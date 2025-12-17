import logging
import random
import re
import math
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import rstr
from bson import ObjectId
from transformers import AutoTokenizer

from src.helpers.document import DocumentSchema

LOGGER = logging.getLogger(__name__)
PLACEHOLDER_PATTERN = re.compile(r"\{(?P<key>[^:{}]+)(?::[^{}]*)?\}")
BULK_INSERT_SIZE = 500
TOKEN_PATTERN = re.compile(r"[0-9A-Za-zÀ-ÖØ-öø-ÿ_/-]+|[^\s]", re.UNICODE)


def normalize_requirements(requirement_spec: Any) -> List[Mapping[str, Any]]:
    if isinstance(requirement_spec, Mapping):
        return [dict(requirement_spec)]
    if isinstance(requirement_spec, Iterable) and not isinstance(
        requirement_spec, (str, bytes)
    ):
        normalized: List[Mapping[str, Any]] = []
        for requirement in requirement_spec:
            if isinstance(requirement, Mapping):
                normalized.append(dict(requirement))
        return normalized
    return []


def run_task(*, doc: Optional[Mapping[str, Any]] = None, db=None, MAX_WORKERS: int = 2) -> None:
    if not doc or db is None:
        LOGGER.warning("builder: tâche ignorée (doc ou db manquant)")
        return

    datasets = db.get_collection("datasets")
    data_collection = db.get_collection("datasets_data")
    models = db.get_collection("models")
    configs = db.get_collection("models_configurations")

    dataset_id = doc["_id"]
    if not isinstance(dataset_id, ObjectId):
        dataset_id = ObjectId(dataset_id)

    model_id = doc.get("model") or doc.get("model_id")
    if not model_id:
        raise ValueError("Model configuration is missing")

    if not isinstance(model_id, ObjectId):
        model_id = ObjectId(model_id)

    model = models.find_one({"_id": model_id})
    if not model:
        raise ValueError(f"Model introuvable: {model_id}")

    configuration_id = model.get("configuration") or doc.get("configuration")
    if not configuration_id:
        raise ValueError("Model configuration is missing")

    if not isinstance(configuration_id, ObjectId):
        configuration_id = ObjectId(configuration_id)

    configuration = configs.find_one({"_id": configuration_id})
    if not configuration:
        raise ValueError(f"Configuration introuvable: {configuration_id}")

    negative_config_ids = configuration.get("negative_configurations") or []
    negative_configurations = []
    if negative_config_ids:
        negative_ids = [ObjectId(nid) for nid in negative_config_ids if nid]
        if negative_ids:
            negative_configurations = list(configs.find({"_id": {"$in": negative_ids}}))
            LOGGER.info(f"Chargement de {len(negative_configurations)} configurations de bruit (négatives).")

    entity_keys: List[str] = []

    mapper_spec = model.get("mapper")
    if mapper_spec:
        try:
            schema = DocumentSchema.from_mapping(mapper_spec)
            entity_keys = [label for label in schema.entity_labels() if label]
        except Exception as exc:
            LOGGER.warning("Impossible de lire le mapper pour déterminer les entités: %s", exc)

    if not entity_keys:
        entity_keys = list((model.get("entities") or {}).keys())

    if not entity_keys and configuration:
        LOGGER.info("Aucune entité définie dans le modèle, utilisation des attributs de la configuration.")
        attributes = configuration.get("attributes") or []
        entity_keys = [attr.get("key") for attr in attributes if attr.get("key")]

    randomizers = model.get("randomizers") or []

    train_params = doc.get("parameters") or {}

    tokenizer_path = train_params.get("base_model", "camembert/camembert-base")
    tokenizer = None
    if tokenizer_path:
        try:
            LOGGER.info(f"Chargement du tokenizer depuis: {tokenizer_path}")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
        except Exception as e:
            LOGGER.warning(f"Impossible de charger le tokenizer: {e}. La génération se fera sans tokens.")

    builder = DatasetBuilder(
        configuration=configuration, 
        db=db, 
        entity_keys=entity_keys, 
        randomizers=randomizers,
        tokenizer=tokenizer,
        negative_configurations=negative_configurations,
        train_params=train_params,
    )
    dataset_requirements = builder.requirements

    size_info = train_params.get("dataset_size", 1000)
    
    negative_ratio = float(train_params.get("negative_ratio", 0.0))
    negative_ratio = max(0.0, min(1.0, negative_ratio))

    max_possibilities = int(configuration.get("possibilities", 1e5))
    formats_count = len(configuration.get("formats") or [])
    dataset_size = determine_dataset_size(size_info, max_possibilities, formats_count)

    LOGGER.info("builder[%s]: génération de %s entrées (Ratio négatif: %.2f)", dataset_id, dataset_size, negative_ratio)
    datasets.update_one(
        {"_id": dataset_id},
        {"$set": {"status": "in-building", "progress": 0.0, "requirements": dataset_requirements}},
    )

    samples: List[Dict[str, Any]] = []
    update_interval = max(1, dataset_size // 100)
    
    for index in range(dataset_size):
        should_be_negative = (random.random() < negative_ratio) and (len(negative_configurations) > 0)
        
        samples.append(builder.generate_sample(is_negative=should_be_negative))
        
        if (index + 1) % update_interval == 0 or index + 1 == dataset_size:
            progress = (index + 1) / dataset_size
            datasets.update_one({"_id": dataset_id}, {"$set": {"progress": progress}})

    payloads = [
        {"dataset": dataset_id, "data": sample, "created_at": datetime.utcnow()}
        for sample in samples
    ]

    for start in range(0, len(payloads), BULK_INSERT_SIZE):
        chunk = payloads[start : start + BULK_INSERT_SIZE]
        if chunk:
            data_collection.insert_many(chunk)

    datasets.update_one(
        {"_id": dataset_id},
        {"$set": {"status": "to-validate", "progress": 1.0}},
    )


def determine_dataset_size(size_info: Any, max_size: int, formats_count: int) -> int:
    formats_count = max(1, formats_count)
    if isinstance(size_info, Mapping):
        requested = size_info.get("size", max_size)
    else:
        requested = size_info or max_size

    if is_integer(requested):
        value = max(1, int(requested))
        return value

    keyword = str(requested).lower()
    return calculate_size_from_keyword(keyword, max_size, formats_count)


def calculate_size_from_keyword(keyword: str, max_size: int, formats_size: int) -> int:
    match keyword:
        case "complete":
            return max_size
        case "advanced":
            return max_size // 2
        case "recommended":
            return max_size // formats_size
        case "small":
            return max_size // formats_size // 2
        case "tiny":
            return max(max_size // formats_size // 5, 1)
        case _:
            return min(max_size, 1000)


class DatasetBuilder:
    def __init__(
        self,
        *,
        configuration: Mapping[str, Any],
        db,
        entity_keys: Sequence[str],
        randomizers: Sequence[Mapping[str, Any]],
        tokenizer=None,
        negative_configurations: Optional[List[Mapping[str, Any]]] = None,
        train_params: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.configuration = configuration
        self.db = db
        self.entity_keys = list(entity_keys)
        self.randomizers = list(randomizers)
        self.tokenizer = tokenizer
        self.negative_configurations = negative_configurations or []
        self.train_params = dict(train_params or {})

        injection_mode = str(self.train_params.get("negative_injection_mode", "mixed"))
        if injection_mode not in {"append", "prepend", "mixed"}:
            injection_mode = "append"
        self.negative_injection_mode = injection_mode
        self.negative_injection_separator = str(self.train_params.get("negative_injection_separator", "\n"))

        raw_probability = self.train_params.get("negative_injection_probability")
        self.negative_injection_probability: Optional[float] = None
        if raw_probability is not None:
            try:
                prob_value = max(0.0, min(1.0, float(raw_probability)))
                self.negative_injection_probability = prob_value
            except (TypeError, ValueError):
                self.negative_injection_probability = None
        
        self.requirements_map: Dict[str, List[Mapping[str, Any]]] = {}
        self._visited_config_ids: Set[str] = set()
        
        self._collect_requirements(self.configuration)
        for neg_conf in self.negative_configurations:
            self._collect_requirements(neg_conf)

    @property
    def requirements(self) -> Dict[str, List[Mapping[str, Any]]]:
        return {
            key: [dict(requirement) for requirement in requirements]
            for key, requirements in self.requirements_map.items()
        }

    def generate_sample(self, is_negative: bool = False) -> Dict[str, Any]:
        target_config = self.configuration
        context: Dict[str, Any] = {}

        constants = target_config.get("constants") or []
        for const_def in constants:
            const_key = const_def.get("key")
            if not const_key:
                continue
            
            val, _ = self._build_dynamic_value(const_def.get("value"), context=context)
            val = coerce_type(const_def.get("type", "string"), val)
            context[const_key] = val

        template = random.choice(target_config.get("formats") or [""])
        attributes_defs = target_config.get("attributes") or []
        built_attributes: List[Dict[str, Any]] = []

        for attr_def in attributes_defs:
            built_attr, extra_attrs = self._build_attribute(attr_def, context)
            
            if built_attr.get("key"):
                context[built_attr["key"]] = built_attr.get("value")
                
            built_attributes.append(built_attr)
            built_attributes.extend(extra_attrs)

        detection_keys = list({
            *(self.entity_keys),
            *(attr.get("key") for attr in attributes_defs if attr.get("key")),
        })

        resolved_text, entities = self._render_entity(
            template, built_attributes, entity_keys=detection_keys
        )

        final_text = resolved_text
        final_entities = [list(entity) for entity in entities]

        should_inject_noise = is_negative and bool(self.negative_configurations)
        if should_inject_noise:
            if self.negative_injection_probability is None or random.random() < self.negative_injection_probability:
                max_noises = max(1, len(self.negative_configurations))
                noise_count = random.randint(1, max_noises)
                LOGGER.debug("Preparing to inject %d negative noise block(s).", noise_count)

                for _ in range(noise_count):
                    negative_config = random.choice(self.negative_configurations)
                    noise_text = self._generate_negative_noise(negative_config)
                    if not noise_text:
                        continue

                    mode_choice = self.negative_injection_mode
                    if mode_choice == "mixed":
                        weights = self.train_params.get("negative_injection_mode_weights") or {
                            "prepend": 0.6,
                            "append": 0.4,
                        }
                        mode_choice = random.choices(
                            population=["append", "prepend"],
                            weights=[weights["append"], weights["prepend"]],
                            k=1
                        )[0]

                    final_text, final_entities = self._inject_noise(
                        final_text,
                        final_entities,
                        noise_text,
                        mode_choice,
                        self.negative_injection_separator,
                    )

        final_text = self._apply_randomizer(final_text)

        stripped_text, stripped_entities = self._strip_text_and_entities(final_text, final_entities)

        result = {"text": stripped_text, "entities": stripped_entities}

        drop = {"prefixe"}

        gliner_ready = self._build_gliner_entry_from_text(
            result["text"], result["entities"], drop_labels=drop
        )
        if gliner_ready:
            result["gliner"] = gliner_ready

        return result

    def _generate_negative_noise(self, neg_conf: Mapping[str, Any]) -> str:
        context: Dict[str, Any] = {}

        constants = neg_conf.get("constants") or []
        for const_def in constants:
            const_key = const_def.get("key")
            if not const_key:
                continue

            val, _ = self._build_dynamic_value(const_def.get("value"), context=context)
            val = coerce_type(const_def.get("type", "string"), val)
            context[const_key] = val

        template = random.choice(neg_conf.get("formats") or [""])
        attributes_defs = neg_conf.get("attributes") or []
        built_attributes: List[Dict[str, Any]] = []

        for attr_def in attributes_defs:
            built_attr, extra_attrs = self._build_attribute(attr_def, context)
            if built_attr.get("key"):
                context[built_attr["key"]] = built_attr.get("value")

            built_attributes.append(built_attr)
            built_attributes.extend(extra_attrs)

        noise_text, _ = self._render_entity(template, built_attributes, entity_keys=set())
        cleaned = re.sub(r"\s+", " ", noise_text).strip()
        return cleaned

    def _inject_noise(
        self,
        base_text: str,
        base_entities: List[List[Any]],
        noise_text: str,
        mode: str,
        sep: str,
    ) -> Tuple[str, List[List[Any]]]:
        if not noise_text:
            LOGGER.debug("No noise to inject; returning base text unchanged.")
            return base_text, [list(entity) for entity in base_entities]

        effective_mode = mode if mode in {"append", "prepend"} else "append"
        insertion_sep = "" if sep is None else sep
        entities = [list(entity) for entity in base_entities]

        if effective_mode == "append":
            final_text = f"{base_text}{insertion_sep}{noise_text}"
            LOGGER.debug(
                "Injected noise (mode=%s) length=%d, entities=%d, shift=%d",
                effective_mode,
                len(noise_text),
                len(entities),
                0,
            )
            return final_text, entities

        if effective_mode == "prepend":
            shift = len(noise_text + insertion_sep)
            shifted_entities = [[start + shift, end + shift, label] for start, end, label in entities]
            final_text = f"{noise_text}{insertion_sep}{base_text}"
            LOGGER.debug(
                "Injected noise (mode=%s) length=%d, entities=%d, shift=%d",
                effective_mode,
                len(noise_text),
                len(entities),
                shift,
            )
            return final_text, shifted_entities

        insertion = f"{insertion_sep}{noise_text}{insertion_sep}"
        shift = len(insertion)

        candidate_positions: List[int] = []
        for match in TOKEN_PATTERN.finditer(base_text):
            candidate_positions.append(match.end())
        candidate_positions.extend([m.start() for m in re.finditer(r"\s", base_text)])

        def _is_inside_entity(pos: int) -> bool:
            for start, end, _ in entities:
                if start < pos < end:
                    return True
            return False

        candidate_positions = [pos for pos in candidate_positions if not _is_inside_entity(pos)]
        if not candidate_positions:
            for _ in range(5):
                pos = random.randint(0, len(base_text)) if base_text else 0
                if not _is_inside_entity(pos):
                    candidate_positions.append(pos)
                    break
        pos = random.choice(candidate_positions) if candidate_positions else len(base_text)

        shifted_entities: List[List[Any]] = []
        for start, end, label in entities:
            if start >= pos:
                shifted_entities.append([start + shift, end + shift, label])
            elif start < pos < end:
                shifted_entities.append([start, end + shift, label])
            else:
                shifted_entities.append([start, end, label])

        final_text = f"{base_text[:pos]}{insertion}{base_text[pos:]}"
        LOGGER.debug(
            "Injected noise (mode=%s) length=%d, entities=%d, shift=%d at pos=%d",
            effective_mode,
            len(noise_text),
            len(entities),
            shift,
            pos,
        )
        return final_text, shifted_entities

    def _strip_text_and_entities(
        self, text: str, entities: List[List[Any]]
    ) -> Tuple[str, List[List[Any]]]:
        leading_trim = len(text) - len(text.lstrip())
        trailing_trim = len(text) - len(text.rstrip())
        stripped_text = text.strip()

        if leading_trim == 0 and trailing_trim == 0:
            return stripped_text, entities

        adjusted_entities: List[List[Any]] = []
        for start, end, label in entities:
            new_start = max(0, start - leading_trim)
            new_end = max(0, end - leading_trim)
            new_end = min(new_end, len(stripped_text))
            if new_end > new_start:
                adjusted_entities.append([new_start, new_end, label])

        LOGGER.debug(
            "Stripped text leading=%d trailing=%d; entities adjusted=%d",
            leading_trim,
            trailing_trim,
            len(adjusted_entities),
        )
        return stripped_text, adjusted_entities

    def _tokenize_for_gliner(self, text: str) -> Tuple[List[str], List[Tuple[int, int]]]:
        tokens: List[str] = []
        offsets: List[Tuple[int, int]] = []
        for m in TOKEN_PATTERN.finditer(text):
            tok = m.group(0)
            tokens.append(tok)
            offsets.append((m.start(), m.end()))
        return tokens, offsets

    def _bio_tags_from_char_spans(
        self,
        offsets: List[Tuple[int, int]],
        entities: List[List[Any]],
        *,
        drop_labels: Optional[Set[str]] = None,
    ) -> List[str]:
        drop_labels = drop_labels or set()
        tags = ["O"] * len(offsets)

        for start_char, end_char, raw_label in sorted(entities, key=lambda x: x[0]):
            label = str(raw_label)
            label = label.replace("B-", "").replace("I-", "").strip()
            if not label or label in drop_labels:
                continue

            begun = False
            for i, (ts, te) in enumerate(offsets):
                if te <= start_char or ts >= end_char:
                    continue
                if not begun:
                    tags[i] = f"B-{label}"
                    begun = True
                else:
                    tags[i] = f"I-{label}"
        return tags

    def _build_gliner_entry_from_text(
        self,
        text: str,
        entities: List[List[Any]],
        *,
        drop_labels: Optional[Set[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        tokens, offsets = self._tokenize_for_gliner(text)
        if not tokens:
            return None

        tags = self._bio_tags_from_char_spans(offsets, entities, drop_labels=drop_labels)

        ner: List[List[Any]] = []
        start_idx: Optional[int] = None
        current_label: Optional[str] = None

        def flush(end_idx: int):
            nonlocal start_idx, current_label
            if current_label is not None and start_idx is not None:
                ner.append([start_idx, end_idx, current_label])
            start_idx = None
            current_label = None

        for i, tag in enumerate(tags):
            if tag == "O":
                if current_label is not None:
                    flush(i - 1)
                continue

            bio, lab = tag.split("-", 1)
            if bio == "B":
                if current_label is not None:
                    flush(i - 1)
                start_idx = i
                current_label = lab
            else:
                if current_label is None:
                    start_idx = i
                    current_label = lab
                elif lab != current_label:
                    flush(i - 1)
                    start_idx = i
                    current_label = lab

        if current_label is not None:
            flush(len(tags) - 1)

        return {"tokenized_text": tokens, "ner": ner}

    def _build_attribute(self, attribute: Mapping[str, Any], context: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        key = attribute.get("key")
        frequency = float(attribute.get("frequency", 1))
        include = random.random() <= frequency
        requirements = normalize_requirements(attribute.get("requirements"))

        value_spec = attribute.get("value") if include else None
        extra_attrs: List[Dict[str, Any]] = []
        value: Any = ""

        if isinstance(value_spec, Mapping):
            value, extra_attrs = self._build_dynamic_value(value_spec, context)
        elif value_spec is not None:
            value = value_spec

        value = coerce_type(attribute.get("type", "string"), value)

        attribute_payload: Dict[str, Any] = {"key": key, "value": "" if value is None else value}
        if requirements:
            attribute_payload["requirements"] = requirements

        return attribute_payload, extra_attrs

    def _build_dynamic_value(self, spec: Mapping[str, Any], context: Dict[str, Any]) -> Tuple[Any, List[Dict[str, Any]]]:
        value_type = spec.get("type", "string")
        rule = spec.get("rule")
        parameters = spec.get("parameters") or {}

        match rule:
            case "constant":
                const_key = parameters.get("const_key")
                # On récupère la valeur depuis le contexte généré
                if const_key and const_key in context:
                    val = context[const_key]
                    return coerce_type(value_type, val), []
                return "", []

            case "calculation":
                formula = parameters.get("formula", "")
                if not formula:
                    return "", []
                try:
                    safe_globals = {
                        "__builtins__": None,
                        "math": math,
                        "int": int,
                        "float": float,
                        "str": str,
                        "round": round,
                        "abs": abs,
                        "min": min,
                        "max": max
                    }
                    eval_context = {**safe_globals, **context}
                    
                    value = eval(formula, eval_context)
                    return coerce_type(value_type, value), []
                except Exception as e:
                    LOGGER.warning(f"Erreur lors du calcul de la formule '{formula}': {e}")
                    return "", []

            case "randint":
                minimum = int(parameters.get("min", 0))
                maximum = int(parameters.get("max", 100))
                if minimum > maximum:
                    minimum, maximum = maximum, minimum
                value = random.randint(minimum, maximum)
                return coerce_type(value_type, value), []

            case "alphanumeric":
                regex = parameters.get("constraint", "")
                value = rstr.xeger(regex) if regex else ""
                return coerce_type(value_type, value), []

            case "data":
                data_id = parameters.get("object_id")
                if data_id:
                    record = self.db.get_collection("models_data").find_one({"_id": ObjectId(data_id)})
                    if record and record.get("data"):
                        value = random.choice(record["data"])
                        return coerce_type(value_type, value), []
                return "", []

            case "configuration":
                config_id = parameters.get("object_id")
                if config_id:
                    nested = self.db.get_collection("models_configurations").find_one({"_id": ObjectId(config_id)})
                    if nested:
                        built = self._build_configuration(nested)
                        return built.get("template", ""), built.get("attributes", [])
                return "", []

            case _:
                return "", []

    def _build_configuration(self, configuration: Mapping[str, Any]) -> Dict[str, Any]:
        template = random.choice(configuration.get("formats") or [""])
        attributes = configuration.get("attributes") or []
        built_attributes: List[Dict[str, Any]] = []
        
        local_context = {}
        constants = configuration.get("constants") or []
        for const_def in constants:
            key = const_def.get("key")
            if key:
                val, _ = self._build_dynamic_value(const_def.get("value"), context=local_context)
                local_context[key] = coerce_type(const_def.get("type"), val)

        for attribute in attributes:
            built_attr, extra_attrs = self._build_attribute(attribute, local_context)
            if built_attr.get("key"):
                local_context[built_attr["key"]] = built_attr.get("value")
            built_attributes.append(built_attr)
            built_attributes.extend(extra_attrs)

        return {"template": re.sub(r"\s+", " ", template.strip()), "attributes": built_attributes}

    def _collect_requirements(self, configuration: Mapping[str, Any]) -> None:
        config_identifier = configuration.get("_id")
        identifier_str: Optional[str] = None
        if isinstance(config_identifier, ObjectId):
            identifier_str = str(config_identifier)
        elif config_identifier is not None:
            try:
                identifier_str = str(ObjectId(config_identifier))
            except Exception:
                identifier_str = None

        if identifier_str:
            if identifier_str in self._visited_config_ids:
                return
            self._visited_config_ids.add(identifier_str)

        attributes = configuration.get("attributes") or []
        for attribute in attributes:
            key = attribute.get("key")
            if key:
                requirements = normalize_requirements(attribute.get("requirements"))
                if key not in self.requirements_map or requirements:
                    self.requirements_map[key] = requirements

            value_spec = attribute.get("value")
            if isinstance(value_spec, Mapping):
                self._collect_requirements_from_value(value_spec)

    def _collect_requirements_from_value(self, value_spec: Mapping[str, Any]) -> None:
        rule = value_spec.get("rule")
        if rule != "configuration":
            return

        parameters = value_spec.get("parameters") or {}
        config_id = parameters.get("object_id")
        if config_id is None:
            return

        object_id: Optional[ObjectId] = None
        if isinstance(config_id, ObjectId):
            object_id = config_id
        else:
            try:
                object_id = ObjectId(config_id)
            except Exception:
                return

        if object_id:
            nested = self.db.get_collection("models_configurations").find_one({"_id": object_id})
            if nested:
                self._collect_requirements(nested)

    def _check_requirements(self, value: Any, requirements: Iterable[Mapping[str, Any]]) -> bool:
        for requirement in requirements or []:
            rule = requirement.get("rule")
            constraint = requirement.get("constraint")
            try:
                val_str = str(value)
                const_str = str(constraint)
                if rule == "regex":
                    if not re.match(const_str, val_str):
                        return False
                elif rule == "eq" and val_str != const_str:
                    return False
                elif rule == "neq" and val_str == const_str:
                    return False
                elif rule == "gt" and float(value) <= float(constraint):
                    return False
                elif rule == "lt" and float(value) >= float(constraint):
                    return False
                elif rule == "gte" and float(value) < float(constraint):
                    return False
                elif rule == "lte" and float(value) > float(constraint):
                    return False
                elif rule == "in":
                    if val_str not in split_constraint(constraint):
                        return False
                elif rule == "nin":
                    if val_str in split_constraint(constraint):
                        return False
                elif rule == "contains" and const_str not in val_str:
                    return False
                elif rule == "ncontains" and const_str in val_str:
                    return False
            except Exception:
                return False
        return True

    def _render_entity(
        self,
        template: str,
        attributes: Sequence[Mapping[str, Any]],
        *,
        entity_keys: Optional[Sequence[str]] = None,
    ) -> Tuple[str, List[List[Any]]]:
        effective_entity_keys = set(entity_keys or self.entity_keys)
        attr_map = {attr.get("key"): attr for attr in attributes}
        resolved_values: Dict[str, Dict[str, Any]] = {}

        def resolve_value(
            key: str, stack: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            stack = stack or []
            if key in resolved_values:
                return resolved_values[key]

            attr = attr_map.get(key)
            if not attr:
                result = {"text": "", "entities": []}
                resolved_values[key] = result
                return result

            if key in stack:
                result = {"text": "", "entities": []}
                resolved_values[key] = result
                return result

            raw_value = str(attr.get("value", ""))
            parts: List[str] = []
            nested_entities: List[Tuple[int, int, str]] = []
            last_index = 0
            offset = 0

            for match in PLACEHOLDER_PATTERN.finditer(raw_value):
                literal = raw_value[last_index:match.start()]
                if literal:
                    parts.append(literal)
                    offset += len(literal)

                nested_key = match.group("key")
                nested_result = resolve_value(nested_key, stack + [key])
                nested_text = nested_result.get("text", "")
                parts.append(nested_text)

                nested_length = len(nested_text)
                if nested_length:
                    nested_entities.append((offset, offset + nested_length, nested_key))

                for nested_start, nested_end, deeper_key in nested_result.get("entities", []):
                    nested_entities.append(
                        (offset + nested_start, offset + nested_end, deeper_key)
                    )

                offset += nested_length
                last_index = match.end()

            tail = raw_value[last_index:]
            if tail:
                parts.append(tail)
                offset += len(tail)

            resolved = "".join(parts)
            
            attr["requirements_met"] = self._check_requirements(
                resolved, attr.get("requirements")
            )

            result = {"text": resolved, "entities": nested_entities}
            resolved_values[key] = result
            return result

        parts: List[str] = []
        entities: List[List[Any]] = []
        cursor = 0
        last_index = 0

        for match in PLACEHOLDER_PATTERN.finditer(template):
            literal = template[last_index : match.start()]
            if literal:
                parts.append(literal)
                cursor += len(literal)

            key = match.group("key")
            value_info = resolve_value(key)
            value = value_info.get("text", "")
            attr = attr_map.get(key)
            requirements_met = True if attr is None else attr.get("requirements_met", True)

            raw_start = cursor
            raw_end = raw_start + len(value)

            start, end = _trim_span(value, raw_start, raw_end)

            has_nested_entities = bool(value_info.get("entities"))

            if (
                key in effective_entity_keys
                and key != "prefixe"
                and value
                and requirements_met
                and not has_nested_entities
            ):
                entities.append([start, end, f"B-{key}"])

            for nested_start, nested_end, nested_key in value_info.get("entities", []):
                absolute_start = raw_start + nested_start
                absolute_end = raw_start + nested_end
                if absolute_end <= absolute_start:
                    continue
                
                nested_attr = attr_map.get(nested_key)
                nested_requirements_met = (
                    True if nested_attr is None else nested_attr.get("requirements_met", True)
                )
                
                if (
                    nested_key in effective_entity_keys
                    and nested_key != "prefixe"
                    and nested_requirements_met
                ):
                    trimmed_start, trimmed_end = _trim_span(
                        value[nested_start:nested_end], absolute_start, absolute_end
                    )
                    if trimmed_end > trimmed_start:
                        entities.append([trimmed_start, trimmed_end, f"B-{nested_key}"])

            cursor = end
            parts.append(value)
            last_index = match.end()

        parts.append(template[last_index:])
        final_text = "".join(parts)
        
        return final_text, entities

    def _tokenize_and_align(
        self, text: str, entities: List[List[Any]], *, raw_entities: Optional[List[List[Any]]] = None
    ) -> Dict[str, Any]:
        if self.tokenizer:
            encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
            tokens = encoding.tokens()
            offsets = encoding["offset_mapping"]
        else:
            # Fallback simple pour garantir des tokens pour GLiNER même sans tokenizer HF
            tokens = text.split()
            offsets = []
            cursor = 0
            for token in tokens:
                start = text.find(token, cursor)
                end = start + len(token)
                offsets.append((start, end))
                cursor = end

        ner_tags = ["O"] * len(tokens)
        entities.sort(key=lambda x: x[0])

        for start_char, end_char, label in entities:
            entity_type = label.replace("B-", "").replace("I-", "")
            found_start = False
            for idx, (token_start, token_end) in enumerate(offsets):
                if token_start == 0 and token_end == 0:
                    continue
                if token_start >= start_char and token_end <= end_char:
                    if not found_start:
                        ner_tags[idx] = f"B-{entity_type}"
                        found_start = True
                    else:
                        ner_tags[idx] = f"I-{entity_type}"

        entity_token_ids: Dict[str, List[int]] = {}
        for start_char, end_char, label in raw_entities or entities:
            entity_type = label.replace("B-", "").replace("I-", "")
            token_indices: List[int] = []
            for idx, (token_start, token_end) in enumerate(offsets):
                if token_start == 0 and token_end == 0:
                    continue
                if token_start >= start_char and token_end <= end_char:
                    token_indices.append(idx)
            if token_indices:
                existing = entity_token_ids.setdefault(entity_type, [])
                existing.extend(token_indices)

        if entity_token_ids:
            for key in entity_token_ids:
                entity_token_ids[key] = sorted(set(entity_token_ids[key]))

        result: Dict[str, Any] = {"tokens": tokens, "ner_tags": ner_tags}
        if entity_token_ids:
            result["entity_token_ids"] = entity_token_ids

        return result

    def _apply_randomizer(self, text: str) -> str:
        if not self.randomizers:
            return text
        randomizer = random.choice(self.randomizers)
        frequency = float(randomizer.get("frequency", 1))
        if random.random() > frequency:
            return text
        rule = randomizer.get("rule")
        if rule == "upper":
            return text.upper()
        if rule == "lower":
            return text.lower()
        return text


def coerce_type(value_type: str, value: Any) -> Any:
    if value is None:
        return None
    if value_type == "number":
        try:
            return int(float(value)) if "." not in str(value) else float(value)
        except (TypeError, ValueError):
            return value
    if value_type == "string":
        return str(value)
    return value


def _trim_span(raw_value: str, start: int, end: int) -> Tuple[int, int]:
    """Retire les espaces en bordure pour éviter de taguer les préfixes."""

    if not raw_value:
        return start, end

    leading = 0
    trailing = 0

    for char in raw_value:
        if char.isspace():
            leading += 1
        else:
            break

    for char in reversed(raw_value):
        if char.isspace():
            trailing += 1
        else:
            break

    new_start = start + leading
    new_end = end - trailing

    if new_end < new_start:
        return start, end

    return new_start, new_end


def split_constraint(constraint: Any) -> List[str]:
    if isinstance(constraint, str):
        return [part.strip() for part in constraint.split(",") if part.strip()]
    if isinstance(constraint, Iterable):
        return [str(item) for item in constraint]
    return [str(constraint)]


def is_integer(value: Any) -> bool:
    try:
        int(value)
        return True
    except (ValueError, TypeError):
        return False
