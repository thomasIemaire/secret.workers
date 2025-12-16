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
        negative_configurations=negative_configurations
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
        negative_configurations: Optional[List[Mapping[str, Any]]] = None
    ) -> None:
        self.configuration = configuration
        self.db = db
        self.entity_keys = list(entity_keys)
        self.randomizers = list(randomizers)
        self.tokenizer = tokenizer
        self.negative_configurations = negative_configurations or []
        
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
        if is_negative and self.negative_configurations:
            target_config = random.choice(self.negative_configurations)

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

        resolved_text, entities = self._render_entity(template, built_attributes)
        
        resolved_text = self._apply_randomizer(resolved_text)
        
        if is_negative:
            entities = []
        
        result = {"text": resolved_text.strip(), "entities": entities}

        tokenized_data = self._tokenize_and_align(result["text"], result["entities"])
        if tokenized_data:
            result.update(tokenized_data)

        gliner_ready = self._build_gliner_entry(
            result.get("tokens") or [], result.get("ner_tags") or []
        )
        if gliner_ready:
            result["gliner"] = gliner_ready

        return result

    def _build_gliner_entry(
        self, tokens: Sequence[str], ner_tags: Sequence[str]
    ) -> Optional[Dict[str, Any]]:
        if not tokens:
            return None

        tags = list(ner_tags) if ner_tags else ["O"] * len(tokens)
        if len(tags) < len(tokens):
            tags.extend(["O"] * (len(tokens) - len(tags)))
        tags = tags[: len(tokens)]

        entities: List[List[Any]] = []
        start_idx: Optional[int] = None
        current_label: Optional[str] = None

        for idx, tag in enumerate(tags):
            normalized = str(tag or "").strip()
            base = normalized.split("-", 1)[-1] if normalized else ""

            if not base or base.upper() == "O":
                if current_label is not None:
                    entities.append([start_idx, idx - 1, current_label])
                    current_label = None
                    start_idx = None
                continue

            if normalized.upper().startswith(("B-", "I-")):
                if current_label is None:
                    start_idx = idx
                    current_label = base
                    continue
                if base != current_label:
                    entities.append([start_idx, idx - 1, current_label])
                    start_idx = idx
                    current_label = base
            else:
                # Valeur non BIO mais non vide : on démarre/continue l'entité courante
                if current_label is None:
                    start_idx = idx
                    current_label = base

        if current_label is not None and start_idx is not None:
            entities.append([start_idx, len(tags) - 1, current_label])

        return {"tokenized_text": list(tokens), "ner": entities}

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
    ) -> Tuple[str, List[List[Any]]]:
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

            start = cursor
            end = start + len(value)

            if key in self.entity_keys and value and requirements_met:
                entities.append([start, end, f"B-{key}"])

            for nested_start, nested_end, nested_key in value_info.get("entities", []):
                absolute_start = start + nested_start
                absolute_end = start + nested_end
                if absolute_end <= absolute_start:
                    continue
                
                nested_attr = attr_map.get(nested_key)
                nested_requirements_met = (
                    True if nested_attr is None else nested_attr.get("requirements_met", True)
                )
                
                if nested_key in self.entity_keys and nested_requirements_met:
                    entities.append([absolute_start, absolute_end, f"B-{nested_key}"])

            cursor = end
            parts.append(value)
            last_index = match.end()

        parts.append(template[last_index:])
        final_text = "".join(parts)
        
        return final_text, entities

    def _tokenize_and_align(self, text: str, entities: List[List[Any]]) -> Dict[str, Any]:
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

        return {"tokens": tokens, "ner_tags": ner_tags}

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
