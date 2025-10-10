"""Training utilities for sequence tagging models."""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import evaluate
import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2
from transformers import (
    CamembertForMaskedLM,
    CamembertForTokenClassification,
    CamembertTokenizerFast,
    DataCollatorForLanguageModeling,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    SchedulerType,
    Trainer,
    TrainingArguments,
)

from .callbacks import MongoTrainLogger
from .document import (
    DocumentExtractionAgent,
    DocumentSchema,
    DocumentVocabulary,
    save_descriptor,
)

try:  # pragma: no cover - optional dependency
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    LoraConfig = TaskType = get_peft_model = None
    PEFT_AVAILABLE = False

LOGGER = logging.getLogger(__name__)
MODEL_NAME = "camembert-base"
MAX_SEQ_LENGTH = 512
O_LABEL = "O"


@dataclass
class CleanedExample:
    """Representation of a validated training example."""

    text: str
    entities: List[Tuple[int, int, str]]
    original: Mapping[str, Any]


def _normalise_labels(label_names: Sequence[str]) -> List[str]:
    unique = []
    seen = set()
    for label in label_names:
        if label not in seen:
            unique.append(label)
            seen.add(label)

    if O_LABEL not in seen:
        unique.insert(0, O_LABEL)
        seen.add(O_LABEL)

    others = sorted(label for label in unique if label != O_LABEL)
    return [O_LABEL, *others]


def _ensure_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _ensure_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _ensure_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if value is None:
        return default
    return bool(value)


def _resolve_scheduler_type(value: Any) -> SchedulerType:
    if isinstance(value, SchedulerType):
        return value

    if value is None:
        return SchedulerType.LINEAR

    try:
        scheduler_value = str(value).strip()
    except Exception:
        scheduler_value = ""

    if not scheduler_value:
        return SchedulerType.LINEAR

    try:
        return SchedulerType(scheduler_value.lower())
    except ValueError:
        LOGGER.warning(
            "Type de scheduler invalide '%s', utilisation de 'linear'",
            value,
        )
        return SchedulerType.LINEAR


def _sanitize_for_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {key: _sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize_for_json(item) for item in value]
    return str(value)


def _clean_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    allowed_labels: Iterable[str],
) -> Tuple[List[CleanedExample], Counter]:
    """Validate and deduplicate raw dataset entries."""

    allowed = {
        label.split("-", 1)[-1] if label.startswith(("B-", "I-")) else label
        for label in allowed_labels
    }
    cleaned: List[CleanedExample] = []
    stats: Counter = Counter()
    seen_examples: set[Tuple[str, Tuple[Tuple[int, int, str], ...]]] = set()

    for example in examples:
        data = example.get("data") or {}
        text = data.get("text", "")
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            stats["empty_text"] += 1
            continue

        normalised_text = " ".join(text.split())
        entities = data.get("entities") or []
        cleaned_entities: List[Tuple[int, int, str]] = []
        for raw in entities:
            try:
                start, end, label = raw
            except (TypeError, ValueError):
                stats["malformed_entity"] += 1
                continue

            try:
                start = int(start)
                end = int(end)
            except (TypeError, ValueError):
                stats["non_numeric_span"] += 1
                continue

            if end <= start:
                stats["invalid_span"] += 1
                continue

            raw_label = str(label or "").strip()
            base_label = raw_label.split("-", 1)[-1]
            if not base_label or base_label not in allowed:
                stats["unknown_label"] += 1
                continue

            start = max(0, start)
            end = min(len(normalised_text), end)
            if start >= end:
                stats["out_of_bounds"] += 1
                continue

            cleaned_entities.append((start, end, base_label))

        cleaned_entities.sort()
        dedup_key = (normalised_text, tuple(cleaned_entities))
        if dedup_key in seen_examples:
            stats["duplicates"] += 1
            continue
        seen_examples.add(dedup_key)

        cleaned.append(
            CleanedExample(text=normalised_text, entities=cleaned_entities, original=example)
        )

        if not cleaned_entities:
            stats["no_entities"] += 1

    return cleaned, stats


def prepare_dataset(
    examples: Sequence[Mapping[str, Any]],
    label_names: Sequence[str],
    tokenizer: CamembertTokenizerFast,
    *,
    schema: Optional[DocumentSchema] = None,
    vocabulary: Optional[DocumentVocabulary] = None,
) -> Tuple[Dataset, Dict[str, int], Dict[int, str], Dict[str, Any]]:
    label_names = _normalise_labels(label_names)
    label2id = {name: i for i, name in enumerate(label_names)}
    id2label = {i: name for name, i in label2id.items()}

    cleaned_examples, stats = _clean_examples(examples, allowed_labels=label_names)
    if stats:
        LOGGER.info("Nettoyage des données: %s", dict(stats))

    records: List[MutableMapping[str, Any]] = []
    for cleaned in cleaned_examples:
        text = cleaned.text

        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        offsets = enc.pop("offset_mapping")

        labels = [label2id[O_LABEL]] * len(enc["input_ids"])
        assigned_labels: List[Optional[str]] = [None] * len(offsets)
        assigned_spans: List[Optional[Tuple[int, int]]] = [None] * len(offsets)
        for i, (start, end) in enumerate(offsets):
            if start == end == 0:
                labels[i] = -100

        for start, end, label in cleaned.entities:
            saw_begin = False
            for idx, (tok_start, tok_end) in enumerate(offsets):
                if tok_start == tok_end == 0:
                    continue
                if tok_end <= start or tok_start >= end:
                    continue
                tag = (
                    f"B-{label}"
                    if not saw_begin and (tok_start <= start < tok_end)
                    else f"I-{label}"
                )
                if tag in label2id:
                    previous_label = assigned_labels[idx]
                    previous_span = assigned_spans[idx]
                    current_span = (start, end)
                    if previous_label is not None and (
                        previous_label != label or previous_span != current_span
                    ):
                        previous_desc = (
                            f"{previous_label} {previous_span}"
                            if previous_span is not None
                            else previous_label
                        )
                        conflict_desc = f"{label} {current_span}"
                        raise ValueError(
                            "Les entités qui se chevauchent ne sont pas supportées : "
                            f"{previous_desc} vs {conflict_desc} dans l'exemple '{text}'. "
                            "Le modèle de token classification ne peut encoder qu'une seule étiquette par token."
                        )

                    assigned_labels[idx] = label
                    assigned_spans[idx] = current_span
                    labels[idx] = label2id[tag]
                    saw_begin = True

        if all(label == -100 for label in labels):
            continue

        enc["labels"] = [int(value) for value in labels]
        records.append(enc)

    if not records:
        raise ValueError("Dataset vide après parsing")

    dataset = Dataset.from_list(records)
    metadata = {
        "texts": [example.text for example in cleaned_examples],
        "cleaning_stats": dict(stats),
        "schema": {"text": "str", "entities": "List[Tuple[int, int, str]]"},
        "document_schema": schema.to_dict() if schema else None,
        "document_vocabulary": vocabulary.to_metadata() if vocabulary else None,
    }

    LOGGER.info(label2id)

    return dataset, label2id, id2label, metadata


def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray], id2label: Dict[int, str]):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_preds: List[List[str]] = []
    true_labels: List[List[str]] = []
    for pred_seq, label_seq in zip(predictions, labels):
        seq_preds: List[str] = []
        seq_labels: List[str] = []
        for pred, label in zip(pred_seq, label_seq):
            if label == -100:
                continue
            seq_preds.append(id2label[int(pred)])
            seq_labels.append(id2label[int(label)])
        true_preds.append(seq_preds)
        true_labels.append(seq_labels)

    results = {
        "precision": precision_score(true_labels, true_preds, mode="strict", scheme=IOB2),
        "recall": recall_score(true_labels, true_preds, mode="strict", scheme=IOB2),
        "f1": f1_score(true_labels, true_preds, mode="strict", scheme=IOB2),
    }

    try:
        import evaluate
        metric = evaluate.load("seqeval")
        extra = metric.compute(predictions=true_preds, references=true_labels)
        for k, v in extra.items():
            if k not in results:
                results[k] = v
    except Exception as e:
        LOGGER.warning("Impossible de charger le metric 'seqeval' via evaluate: %s", e)

    LOGGER.debug(
        "Rapport strict:\n%s",
        classification_report(true_labels, true_preds, mode="strict", scheme=IOB2, digits=4),
    )

    return results


def _collect_label_stats(dataset: Dataset) -> Counter:
    counter = Counter()
    sample_size = min(50, len(dataset))
    if sample_size:
        for record in dataset.select(range(sample_size)):
            counter.update(record["labels"])
    return counter


def trainer(
    dataset: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    *,
    parameters: Optional[Mapping[str, Any]] = None,
    eval_dataset: Optional[Dataset] = None,
    version: Optional[str] = None,
) -> Trainer:
    if not dataset:
        raise ValueError("Dataset vide: aucune donnée à entraîner")

    parameters = parameters or {}
    mapper_spec = model.get("mapper")
    schema = DocumentSchema.from_mapping(mapper_spec)

    raw_label_names = list(model.get("labels") or [])
    if not raw_label_names and schema.entity_labels():
        generated = []
        for label in schema.entity_labels():
            generated.append(f"B-{label}")
            generated.append(f"I-{label}")
        raw_label_names = [O_LABEL, *generated]

    label_names = _normalise_labels(raw_label_names)
    if len(label_names) <= 1:
        raise ValueError("Aucune étiquette valide n'a été fournie pour l'entraînement")

    model_reference = model.get("reference", "model")
    resolved_version = version or model.get("version", "1.0")

    base_model_name = str(model.get("base_model") or MODEL_NAME)
    tokenizer = CamembertTokenizerFast.from_pretrained(base_model_name)

    additional_terms = model.get("vocabulary") or []
    vocabulary = DocumentVocabulary.from_examples(dataset, additional_terms=additional_terms)
    added_tokens = vocabulary.apply_to_tokenizer(tokenizer)
    if added_tokens:
        LOGGER.info("Vocabulaire documentaire: %s jetons ajoutés", added_tokens)

    if schema.fields or schema.collections:
        LOGGER.info(
            "Schéma documentaire chargé: %s champs, %s collections",
            len(schema.fields),
            len(schema.collections),
        )

    train_ds, label2id, id2label, metadata = prepare_dataset(
        dataset,
        label_names,
        tokenizer,
        schema=schema,
        vocabulary=vocabulary,
    )

    LOGGER.info(
        "Schéma dataset: %s", metadata.get("schema", {"text": "str", "entities": "list"})
    )

    texts_for_pretraining = metadata.get("texts") or []

    LOGGER.info(
        "Dataset prêt: %s exemples, %s étiquettes",
        len(train_ds),
        len(label2id),
    )

    eval_ratio = _ensure_float(parameters.get("eval_ratio"), 0.2)
    if eval_dataset is None and 0.0 < eval_ratio < 0.5 and len(train_ds) > 10:
        split_seed = _ensure_int(parameters.get("seed"), 42)
        LOGGER.info(
            "Découpage automatique du dataset: %.0f%% pour l'évaluation",
            eval_ratio * 100,
        )
        splitted = train_ds.train_test_split(test_size=eval_ratio, seed=split_seed)
        train_ds = splitted["train"]
        eval_dataset = splitted["test"]

    output_dir = Path("sardine.agents") / model_reference / resolved_version
    output_dir.mkdir(parents=True, exist_ok=True)

    domain_pretraining_path = run_domain_adaptive_pretraining(
        texts_for_pretraining,
        tokenizer,
        output_dir=output_dir,
        parameters=parameters.get("continued_pretraining"),
    )

    base_model_path = domain_pretraining_path or base_model_name

    ner_model = CamembertForTokenClassification.from_pretrained(
        base_model_path,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    ner_model = maybe_apply_peft(ner_model, parameters.get("peft"))

    collator = DataCollatorForTokenClassification(tokenizer)

    use_fp16 = _ensure_bool(parameters.get("fp16", False)) and torch.cuda.is_available()
    if parameters.get("fp16") and not torch.cuda.is_available():
        LOGGER.warning("fp16 demandé mais CUDA indisponible -> désactivation")

    logging_dir = Path("logs") / model_reference / resolved_version
    logging_dir.mkdir(parents=True, exist_ok=True)

    warmup_ratio = _ensure_float(parameters.get("warmup_ratio"), 0.0)
    warmup_steps = _ensure_int(parameters.get("warmup_steps"), 0)
    lr_scheduler_type = _resolve_scheduler_type(parameters.get("lr_scheduler_type"))

    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=_ensure_float(parameters.get("learning_rate"), 5e-5),
        per_device_train_batch_size=_ensure_int(parameters.get("batch_size"), 16),
        num_train_epochs=_ensure_float(parameters.get("epochs"), 5),
        weight_decay=_ensure_float(parameters.get("weight_decay"), 0.01),
        save_strategy="epoch",
        evaluation_strategy=(
            "no"
            if eval_dataset is None
            else str(parameters.get("eval_strategy", "epoch"))
        ),
        logging_dir=str(logging_dir),
        logging_steps=_ensure_int(parameters.get("logging_steps", 10), 10),
        fp16=use_fp16,
        gradient_accumulation_steps=_ensure_int(parameters.get("grad_accum"), 1),
        group_by_length=True,
        dataloader_pin_memory=False,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        lr_scheduler_type=lr_scheduler_type,
        load_best_model_at_end=_ensure_bool(
            parameters.get("load_best_model"), eval_dataset is not None
        ),
        metric_for_best_model=parameters.get("metric_for_best_model", "f1"),
        greater_is_better=_ensure_bool(parameters.get("greater_is_better"), True),
        save_total_limit=_ensure_int(parameters.get("save_total_limit"), 2),
    )

    label_stats = _collect_label_stats(train_ds)
    kept = sum(value for key, value in label_stats.items() if key != -100)
    LOGGER.info("Premiers comptes d'étiquettes (50 échantillons): %s", dict(label_stats))
    LOGGER.info("Tokens conservés: %s", kept)

    trainer_instance = Trainer(
        model=ner_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=(
            None
            if eval_dataset is None
            else (lambda predictions: compute_metrics(predictions, id2label))
        ),
    )

    if eval_dataset is not None and _ensure_bool(parameters.get("early_stopping"), False):
        patience = _ensure_int(parameters.get("early_stopping_patience"), 2)
        threshold = _ensure_float(parameters.get("early_stopping_threshold"), 0.0)
        trainer_instance.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=patience,
                early_stopping_threshold=threshold,
            )
        )

    mongo_uri = os.getenv("MONGO_URI")
    dataset_id = dataset[0].get("dataset") if dataset else None
    callback: Optional[MongoTrainLogger] = None
    if mongo_uri and dataset_id:
        callback = MongoTrainLogger(
            mongo_uri=mongo_uri,
            dataset=str(dataset_id),
            model=model.get("name", model_reference),
            version=resolved_version,
        )
        trainer_instance.add_callback(callback)

    trainer_instance.train()
    trainer_instance.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    threshold: Optional[float] = None
    if eval_dataset is not None:
        threshold = calibrate_probability_threshold(
            trainer_instance,
            eval_dataset,
            id2label,
            output_dir=output_dir,
            parameters=parameters.get("thresholds"),
        )
        if threshold is not None:
            LOGGER.info("Seuil de probabilité calibré: %.3f", threshold)

    quantized_dir = maybe_quantize_model(
        trainer_instance.model,
        parameters.get("quantization"),
        output_dir=output_dir,
    )
    if quantized_dir:
        LOGGER.info("Modèle quantifié enregistré dans %s", quantized_dir)

    run_active_learning_loop(
        trainer_instance,
        tokenizer,
        parameters.get("active_learning"),
        output_dir=output_dir,
    )

    thresholds_config = parameters.get("thresholds") if isinstance(parameters, Mapping) else None
    if threshold is None:
        if isinstance(thresholds_config, Mapping) and thresholds_config.get("default") is not None:
            threshold = _ensure_float(thresholds_config.get("default"), 0.5)
        else:
            threshold = _ensure_float(parameters.get("default_threshold"), 0.5)

    descriptor: Optional[dict[str, Any]] = None
    if schema.fields or schema.collections:
        descriptor = DocumentExtractionAgent(
            name=model.get("name", model_reference),
            schema=schema,
            threshold=threshold,
            vocabulary=vocabulary,
            description=str(model.get("description", "")),
        ).to_descriptor(base_model=str(base_model_path))
        save_descriptor(output_dir / "agent.json", descriptor)
        save_descriptor(output_dir / "schema.json", schema.to_dict())

    if vocabulary is not None:
        save_descriptor(output_dir / "vocabulary.json", vocabulary.to_metadata())

    if descriptor is not None:
        LOGGER.info("Descripteur d'agent enregistré dans %s", output_dir / "agent.json")

    if callback is not None:
        callback.close()

    return trainer_instance


def run_domain_adaptive_pretraining(
    texts: Sequence[str],
    tokenizer: CamembertTokenizerFast,
    *,
    output_dir: Path,
    parameters: Optional[Mapping[str, Any]],
) -> Optional[str]:
    if parameters is None:
        return None

    if isinstance(parameters, bool):
        enabled = parameters
        config: Mapping[str, Any] = {}
    else:
        enabled = _ensure_bool(parameters.get("enabled", True), True)
        config = parameters

    if not enabled:
        LOGGER.info("Pré-entraînement continu désactivé")
        return None

    unique_texts = [text for text in dict.fromkeys(texts) if text]
    min_samples = _ensure_int(config.get("min_samples", 50), 50)
    if len(unique_texts) < min_samples:
        LOGGER.info(
            "Pas assez d'exemples (%s) pour le pré-entraînement (min=%s)",
            len(unique_texts),
            min_samples,
        )
        return None

    pretrain_dir = output_dir / "continued-pretraining"
    pretrain_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(
        "Lancement du pré-entraînement continu (%s échantillons)", len(unique_texts)
    )

    corpus = Dataset.from_dict({"text": unique_texts})
    max_samples = _ensure_int(config.get("max_samples", 5000), 5000)
    if len(corpus) > max_samples:
        corpus = corpus.shuffle(seed=_ensure_int(config.get("seed"), 42))
        corpus = corpus.select(range(max_samples))

    def tokenize(batch: Mapping[str, List[str]]):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_special_tokens_mask=True,
        )

    tokenized = corpus.map(tokenize, batched=True, remove_columns=["text"])

    mlm_model = CamembertForMaskedLM.from_pretrained(MODEL_NAME)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=_ensure_float(config.get("mlm_probability", 0.15), 0.15),
    )

    args = TrainingArguments(
        output_dir=str(pretrain_dir),
        per_device_train_batch_size=_ensure_int(config.get("batch_size", 16), 16),
        learning_rate=_ensure_float(config.get("learning_rate", 5e-5), 5e-5),
        num_train_epochs=_ensure_float(config.get("epochs", 1.0), 1.0),
        weight_decay=_ensure_float(config.get("weight_decay", 0.01), 0.01),
        logging_steps=_ensure_int(config.get("logging_steps", 20), 20),
        save_strategy="no",
        warmup_steps=_ensure_int(config.get("warmup_steps", 0), 0),
        warmup_ratio=_ensure_float(config.get("warmup_ratio", 0.0), 0.0),
        max_steps=_ensure_int(config.get("max_steps", -1), -1),
        dataloader_pin_memory=False,
        report_to=[],
    )

    mlm_trainer = Trainer(
        model=mlm_model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    mlm_trainer.train()
    mlm_trainer.save_model(str(pretrain_dir))

    return str(pretrain_dir)


def maybe_apply_peft(
    model: CamembertForTokenClassification,
    config: Optional[Mapping[str, Any]],
) -> CamembertForTokenClassification:
    if not config:
        return model

    if not PEFT_AVAILABLE:
        LOGGER.warning("PEFT demandé mais dépendance introuvable")
        return model

    enabled = _ensure_bool(config.get("enabled", True), True)
    if not enabled:
        return model

    lora_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=_ensure_int(config.get("r", 8), 8),
        lora_alpha=_ensure_float(config.get("alpha", 16.0), 16.0),
        lora_dropout=_ensure_float(config.get("dropout", 0.1), 0.1),
        target_modules=config.get("target_modules", ["query", "value"]),
    )

    LOGGER.info(
        "Activation du mode PEFT (LoRA): r=%s alpha=%s dropout=%.2f",
        lora_config.r,
        lora_config.lora_alpha,
        lora_config.lora_dropout,
    )

    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def calibrate_probability_threshold(
    trainer: Trainer,
    eval_dataset: Dataset,
    id2label: Mapping[int, str],
    *,
    output_dir: Path,
    parameters: Optional[Mapping[str, Any]],
) -> Optional[float]:
    if parameters is None:
        enabled = True
        config: Mapping[str, Any] = {}
    elif isinstance(parameters, bool):
        enabled = parameters
        config = {}
    else:
        enabled = _ensure_bool(parameters.get("enabled", True), True)
        config = parameters

    if not enabled:
        return None

    LOGGER.info("Calibration du seuil de décision sur l'ensemble d'évaluation")
    prediction_output = trainer.predict(eval_dataset)
    logits = prediction_output.predictions
    labels = prediction_output.label_ids

    probabilities = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    o_label_id = next((idx for idx, name in id2label.items() if name == O_LABEL), 0)

    if config and "search_space" in config:
        search_space = [float(val) for val in config["search_space"]]
    else:
        search_space = [round(x, 2) for x in np.linspace(0.3, 0.9, 13)]

    best_threshold = 0.5
    best_score = -1.0

    for threshold in search_space:
        predictions: List[List[int]] = []
        true_labels: List[List[int]] = []
        for prob_seq, label_seq in zip(probabilities, labels):
            seq_pred: List[int] = []
            seq_true: List[int] = []
            for probs, true_label in zip(prob_seq, label_seq):
                if true_label == -100:
                    continue
                best_label = int(np.argmax(probs))
                best_prob = float(np.max(probs))
                if best_prob < threshold:
                    seq_pred.append(o_label_id)
                else:
                    seq_pred.append(best_label)
                seq_true.append(int(true_label))
            predictions.append(seq_pred)
            true_labels.append(seq_true)

        mapped_preds = [[id2label[idx] for idx in seq] for seq in predictions]
        mapped_labels = [[id2label[idx] for idx in seq] for seq in true_labels]
        score = f1_score(mapped_labels, mapped_preds, mode="strict", scheme=IOB2)
        if score > best_score:
            best_score = score
            best_threshold = threshold

    threshold_file = output_dir / "probability_threshold.json"
    threshold_file.write_text(
        json.dumps({"threshold": best_threshold}, indent=2), encoding="utf-8"
    )

    return best_threshold


def maybe_quantize_model(
    model: CamembertForTokenClassification,
    config: Optional[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> Optional[Path]:
    if not config:
        return None

    if isinstance(config, bool):
        enabled = config
        dtype_name = "qint8"
    else:
        enabled = _ensure_bool(config.get("enabled", True), True)
        dtype_name = str(config.get("dtype", "qint8"))

    if not enabled:
        return None

    dtype_map = {
        "qint8": torch.qint8,
        "float16": torch.float16,
    }
    dtype = dtype_map.get(dtype_name, torch.qint8)

    LOGGER.info("Quantification dynamique du modèle (%s)", dtype_name)
    quantized = torch.quantization.quantize_dynamic(
        model.cpu(), {torch.nn.Linear}, dtype=dtype
    )

    quant_dir = output_dir / "quantized"
    quant_dir.mkdir(parents=True, exist_ok=True)
    quantized.save_pretrained(str(quant_dir))
    return quant_dir


def run_active_learning_loop(
    trainer: Trainer,
    tokenizer: CamembertTokenizerFast,
    config: Optional[Mapping[str, Any]],
    *,
    output_dir: Path,
) -> None:
    if not config:
        return

    pool = config.get("pool") if isinstance(config, Mapping) else None
    if not pool:
        LOGGER.info("Aucun pool de données fourni pour l'active learning")
        return

    selection_size = _ensure_int(config.get("selection_size", 25), 25)

    encodings: List[Dict[str, Any]] = []
    texts: List[str] = []
    original_examples: List[Mapping[str, Any]] = []
    for sample in pool:
        text = (sample.get("text") or "").strip()
        if not text:
            continue
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )
        encodings.append(encoded)
        texts.append(text)
        original_examples.append(sample)

    if not encodings:
        LOGGER.info("Pool d'active learning vide après nettoyage")
        return

    dataset = Dataset.from_list(encodings)
    prediction_output = trainer.predict(dataset)
    probabilities = torch.softmax(
        torch.tensor(prediction_output.predictions), dim=-1
    ).numpy()

    candidates: List[Dict[str, Any]] = []
    for text, probs, enc, original in zip(texts, probabilities, encodings, original_examples):
        mask = np.array(enc.get("attention_mask", []))
        valid_probs = probs[mask == 1]
        if valid_probs.size == 0:
            continue
        token_uncertainty = 1.0 - valid_probs.max(axis=-1)
        candidates.append(
            {
                "text": text,
                "uncertainty": float(mean(token_uncertainty)),
                "max_uncertainty": float(np.max(token_uncertainty)),
                "source": _sanitize_for_json({k: v for k, v in original.items() if k != "text"}),
            }
        )

    if not candidates:
        LOGGER.info("Impossible de calculer des incertitudes pour l'active learning")
        return

    candidates.sort(key=lambda item: item["uncertainty"], reverse=True)
    selected = candidates[:selection_size]

    target_file = output_dir / "active_learning_candidates.json"
    target_file.write_text(
        json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    LOGGER.info(
        "Top %s échantillons d'active learning sauvegardés dans %s",
        len(selected),
        target_file,
    )
