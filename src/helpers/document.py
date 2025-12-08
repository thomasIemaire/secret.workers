from __future__ import annotations

from dataclasses import dataclass, field
import math
from statistics import mean
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence

import copy
import json
from pathlib import Path
import re


def _normalise_path(path: Sequence[Any] | None, *, fallback: str) -> tuple[str, ...]:
    values: list[str] = []
    if path is None:
        path = ()
    for part in path:
        if part is None:
            continue
        part_str = str(part).strip()
        if part_str:
            values.append(part_str)
    if not values:
        fallback_value = fallback.strip()
        if not fallback_value:
            raise ValueError("Document paths require at least one segment")
        values.append(fallback_value)
    return tuple(values)


def _normalise_label(label: str) -> str:
    stripped = str(label or "").strip()
    if not stripped:
        raise ValueError("Labels must be non empty strings")
    if stripped.upper().startswith(("B-", "I-")):
        return stripped.split("-", 1)[1]
    return stripped


def _deduplicate_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if not value:
            continue
        lower = value.lower()
        if lower in seen:
            continue
        seen.add(lower)
        ordered.append(value)
    return ordered


def _empty_payload() -> dict[str, Any]:
    return {"value": None, "confidence": 0.0, "provenance": None}


def _is_payload(value: Any) -> bool:
    if not isinstance(value, Mapping):
        return False
    return {"value", "confidence"}.issubset(value.keys())


@dataclass(frozen=True)
class DocumentField:
    """Describe a scalar piece of information to extract from a document."""

    name: str
    label: str
    path: Sequence[str] | None = None
    description: str = ""
    required: bool = False
    multiple: bool = False

    def __post_init__(self) -> None:
        cleaned_name = str(self.name or "").strip()
        if not cleaned_name:
            raise ValueError("Field names must be non empty")
        object.__setattr__(self, "name", cleaned_name)
        object.__setattr__(self, "label", _normalise_label(self.label))
        object.__setattr__(self, "path", _normalise_path(self.path, fallback=cleaned_name))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "path": list(self.path or ()),
            "description": self.description,
            "required": self.required,
            "multiple": self.multiple,
        }


@dataclass(frozen=True)
class DocumentCollection:
    """Describe a repeated structure such as invoice lines."""

    name: str
    path: Sequence[str] | None = None
    fields: Sequence[DocumentField] = field(default_factory=tuple)
    description: str = ""

    def __post_init__(self) -> None:
        cleaned_name = str(self.name or "").strip()
        if not cleaned_name:
            raise ValueError("Collection names must be non empty")
        object.__setattr__(self, "name", cleaned_name)
        object.__setattr__(self, "path", _normalise_path(self.path, fallback=cleaned_name))

        labels = [field.label for field in self.fields]
        duplicates = {label for label in labels if labels.count(label) > 1}
        if duplicates:
            raise ValueError(f"Duplicate labels in collection '{self.name}': {sorted(duplicates)}")

    def label_map(self) -> dict[str, DocumentField]:
        return {field.label: field for field in self.fields}

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": list(self.path or ()),
            "description": self.description,
            "fields": [field.to_dict() for field in self.fields],
        }


@dataclass
class DocumentSchema:
    """Container describing how to project entities into JSON."""

    fields: Sequence[DocumentField] = field(default_factory=tuple)
    collections: Sequence[DocumentCollection] = field(default_factory=tuple)
    metadata: MutableMapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        simple_labels = [field.label for field in self.fields]
        duplicates = {label for label in simple_labels if simple_labels.count(label) > 1}
        if duplicates:
            raise ValueError(f"Duplicate labels for fields: {sorted(duplicates)}")

        collection_labels = []
        for collection in self.collections:
            collection_labels.extend(field.label for field in collection.fields)

        conflicts = set(simple_labels) & set(collection_labels)
        if conflicts:
            raise ValueError(f"Labels must be unique across schema: {sorted(conflicts)}")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "DocumentSchema":
        if not mapping:
            return cls()

        field_specs = mapping.get("fields", []) if isinstance(mapping, Mapping) else []
        fields: list[DocumentField] = []
        for spec in field_specs:
            if not isinstance(spec, Mapping):
                continue
            fields.append(
                DocumentField(
                    name=str(spec.get("name") or spec.get("label") or "field"),
                    label=str(spec.get("label") or spec.get("name") or "field"),
                    path=spec.get("path"),
                    description=str(spec.get("description", "")),
                    required=bool(spec.get("required", False)),
                    multiple=bool(spec.get("multiple", False)),
                )
            )

        collections_specs = mapping.get("collections", []) if isinstance(mapping, Mapping) else []
        collections: list[DocumentCollection] = []
        for collection_spec in collections_specs:
            if not isinstance(collection_spec, Mapping):
                continue
            name = str(collection_spec.get("name") or "collection")
            field_defs = []
            for field_spec in collection_spec.get("fields", []):
                if not isinstance(field_spec, Mapping):
                    continue
                field_defs.append(
                    DocumentField(
                        name=str(field_spec.get("name") or field_spec.get("label") or "field"),
                        label=str(field_spec.get("label") or field_spec.get("name") or "field"),
                        path=field_spec.get("path"),
                        description=str(field_spec.get("description", "")),
                        required=bool(field_spec.get("required", False)),
                    )
                )
            collections.append(
                DocumentCollection(
                    name=name,
                    path=collection_spec.get("path"),
                    description=str(collection_spec.get("description", "")),
                    fields=tuple(field_defs),
                )
            )

        metadata = mapping.get("metadata") if isinstance(mapping, Mapping) else None
        if not isinstance(metadata, Mapping):
            metadata = {}
        return cls(fields=tuple(fields), collections=tuple(collections), metadata=dict(metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "fields": [field.to_dict() for field in self.fields],
            "collections": [collection.to_dict() for collection in self.collections],
            "metadata": dict(self.metadata),
        }

    def entity_labels(self) -> list[str]:
        labels = [field.label for field in self.fields]
        for collection in self.collections:
            labels.extend(field.label for field in collection.fields)
        return sorted(set(labels))

    def label_names(self) -> list[str]:
        labels = []
        for label in self.entity_labels():
            labels.append(f"B-{label}")
            labels.append(f"I-{label}")
        return labels

    def empty(self) -> dict[str, Any]:
        structure: dict[str, Any] = {}
        for field in self.fields:
            container, key = self._ensure_container(structure, field.path)
            container[key] = [] if field.multiple else _empty_payload()
        for collection in self.collections:
            container, key = self._ensure_container(structure, collection.path)
            container[key] = []
        return structure

    def _ensure_container(
        self, structure: MutableMapping[str, Any], path: Sequence[str]
    ) -> tuple[MutableMapping[str, Any], str]:
        current = structure
        for segment in path[:-1]:
            if segment not in current or not isinstance(current[segment], MutableMapping):
                current[segment] = {}
            current = current[segment]
        return current, path[-1]

    def _build_payload(
        self,
        *,
        value: str,
        score: float,
        start: Optional[int],
        end: Optional[int],
        label: str,
        agent: Optional[str],
        metadata: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        provenance: dict[str, Any] = {"label": label}
        if agent:
            provenance["agent"] = agent
        if start is not None:
            provenance["start"] = start
        if end is not None:
            provenance["end"] = end
        if metadata:
            provenance["metadata"] = dict(metadata)
        payload = {"value": value, "confidence": float(score), "provenance": provenance or None}
        return payload

    def map_predictions(
        self,
        predictions: Iterable[Mapping[str, Any]],
        *,
        threshold: float = 0.5,
        agent: Optional[str] = None,
        text: Optional[str] = None,
    ) -> dict[str, Any]:
        structure = self.empty()
        simple_fields = {field.label: field for field in self.fields}
        collection_field_map: dict[str, tuple[DocumentCollection, DocumentField]] = {}
        for collection in self.collections:
            for field in collection.fields:
                collection_field_map[field.label] = (collection, field)

        collection_rows: dict[str, list[dict[str, Any]]] = {collection.name: [] for collection in self.collections}

        normalised_predictions = []
        for raw in predictions:
            if not isinstance(raw, Mapping):
                continue
            label = raw.get("label") or raw.get("entity") or raw.get("type")
            if not label:
                continue
            try:
                cleaned_label = _normalise_label(label)
            except ValueError:
                continue
            text_value = str(raw.get("text") or raw.get("value") or "").strip()
            start = raw.get("start")
            end = raw.get("end")
            try:
                start_int = int(start) if start is not None else None
            except (TypeError, ValueError):
                start_int = None
            try:
                end_int = int(end) if end is not None else None
            except (TypeError, ValueError):
                end_int = None
            score = raw.get("confidence")
            if score is None:
                score = raw.get("score")
            try:
                score_value = float(score)
            except (TypeError, ValueError):
                score_value = 0.0
            metadata = raw.get("metadata") or raw.get("provenance") or {}
            if not isinstance(metadata, Mapping):
                metadata = {}
            metadata = dict(metadata)
            for key in ("group", "row", "index"):
                if key not in metadata and key in raw:
                    metadata[key] = raw[key]
            normalised_predictions.append(
                {
                    "label": cleaned_label,
                    "text": text_value,
                    "start": start_int,
                    "end": end_int,
                    "score": score_value,
                    "metadata": metadata,
                }
            )

        def _sort_key(item: Mapping[str, Any]) -> tuple[float, float]:
            start_pos = item.get("start")
            if start_pos is None:
                start_value = math.inf
            else:
                start_value = float(start_pos)
            return (start_value, -float(item.get("score") or 0.0))

        normalised_predictions.sort(key=_sort_key)

        for prediction in normalised_predictions:
            score_value = float(prediction.get("score") or 0.0)
            if score_value < threshold:
                continue
            label = prediction["label"]
            payload = self._build_payload(
                value=prediction.get("text", ""),
                score=score_value,
                start=prediction.get("start"),
                end=prediction.get("end"),
                label=label,
                agent=agent,
                metadata=prediction.get("metadata"),
            )

            if label in simple_fields:
                field = simple_fields[label]
                container, key = self._ensure_container(structure, field.path)
                if field.multiple:
                    container.setdefault(key, [])
                    container[key].append(payload)
                else:
                    current_payload = container.get(key)
                    current_confidence = (
                        float(current_payload.get("confidence", 0.0))
                        if isinstance(current_payload, Mapping)
                        else 0.0
                    )
                    if score_value >= current_confidence:
                        container[key] = payload
                continue

            if label not in collection_field_map:
                continue

            collection, field = collection_field_map[label]
            rows = collection_rows[collection.name]

            metadata = prediction.get("metadata") or {}
            row_identifier = metadata.get("group")
            if row_identifier is None:
                row_identifier = metadata.get("row")

            target_row: Optional[dict[str, Any]] = None
            if row_identifier is not None:
                for row in rows:
                    if row.get("_row_id") == row_identifier:
                        target_row = row
                        break
                if target_row is None:
                    target_row = {"_row_id": row_identifier}
                    rows.append(target_row)
            else:
                for row in rows:
                    if field.name not in row:
                        target_row = row
                        break
                if target_row is None:
                    target_row = {}
                    rows.append(target_row)

            provenance = payload.get("provenance") or {}
            start_position = provenance.get("start")
            if start_position is not None:
                anchor = target_row.get("_anchor")
                if anchor is None or start_position < anchor:
                    target_row["_anchor"] = start_position

            target_row[field.name] = payload

        for collection in self.collections:
            container, key = self._ensure_container(structure, collection.path)
            rows = collection_rows.get(collection.name, [])
            rows.sort(key=lambda row: (row.get("_row_id"), row.get("_anchor", math.inf)))
            normalised_rows: list[dict[str, Any]] = []
            for row in rows:
                row_payload: dict[str, Any] = {}
                confidences = []
                for field in collection.fields:
                    payload = row.get(field.name, _empty_payload())
                    row_payload[field.name] = payload
                    confidence = float(payload.get("confidence") or 0.0)
                    if confidence:
                        confidences.append(confidence)
                row_payload["_confidence"] = mean(confidences) if confidences else 0.0
                if "_row_id" in row:
                    row_payload["_row_id"] = row["_row_id"]
                normalised_rows.append(row_payload)
            container[key] = normalised_rows

        return structure


class DocumentVocabulary:
    """Build a document oriented vocabulary to extend base tokenizers."""

    DEFAULT_TERMS = [
        "facture",
        "adresse",
        "total",
        "montant",
        "quantité",
        "prix",
        "unitaire",
        "tva",
        "siren",
        "siret",
        "iban",
        "bic",
        "banque",
        "client",
        "fournisseur",
        "paiement",
        "échéance",
        "date",
        "numéro",
        "ligne",
        "description",
        "code",
        "postal",
        "ville",
        "pays",
    ]

    # Modification de la regex pour ne PLUS inclure le tiret
    TOKEN_PATTERN = re.compile(r"[\w']+", re.UNICODE)

    def __init__(
        self,
        tokens: Sequence[str],
        *,
        frequencies: Mapping[str, int] | None = None,
        domain_terms: Iterable[str] | None = None,
    ) -> None:
        self.tokens = list(_deduplicate_preserve_order(tokens))
        if frequencies is None:
            frequencies = {}
        self.frequencies = {str(k): int(v) for k, v in frequencies.items()}
        if domain_terms is None:
            domain_terms = []
        self.domain_terms = _deduplicate_preserve_order([str(term) for term in domain_terms])

    @classmethod
    def from_examples(
        cls,
        examples: Iterable[Mapping[str, Any]],
        *,
        additional_terms: Iterable[str] | None = None,
        max_terms: int = 200,
    ) -> "DocumentVocabulary":
        from collections import Counter

        counter: Counter[str] = Counter()
        for example in examples:
            data = example.get("data") if isinstance(example, Mapping) else None
            text = ""
            if isinstance(data, Mapping):
                text = str(data.get("text") or "")
            for match in cls.TOKEN_PATTERN.finditer(text.lower()):
                token = match.group(0)
                if len(token) <= 1:
                    continue
                counter[token] += 1

        most_common = counter.most_common(max_terms)
        inferred_tokens = [token for token, _ in most_common]
        frequencies = {token: freq for token, freq in most_common}

        tokens: list[str] = list(cls.DEFAULT_TERMS)
        tokens.extend(inferred_tokens)

        extra_terms = list(additional_terms or [])
        tokens.extend(extra_terms)

        return cls(tokens, frequencies=frequencies, domain_terms=extra_terms)

    def apply_to_tokenizer(self, tokenizer: Any) -> int:
        if hasattr(tokenizer, "add_tokens"):
            try:
                return int(tokenizer.add_tokens(self.tokens))
            except Exception:
                return 0
        return 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "size": len(self.tokens),
            "tokens": list(self.tokens),
            "frequencies": dict(self.frequencies),
            "domain_terms": list(self.domain_terms),
        }


@dataclass
class DocumentExtractionAgent:
    """Light-weight representation of a specialised extraction agent."""

    name: str
    schema: DocumentSchema
    threshold: float = 0.5
    vocabulary: Optional[DocumentVocabulary] = None
    description: str = ""

    def map_predictions(
        self,
        predictions: Iterable[Mapping[str, Any]],
        *,
        text: Optional[str] = None,
    ) -> dict[str, Any]:
        return self.schema.map_predictions(
            predictions,
            threshold=self.threshold,
            agent=self.name,
            text=text,
        )

    def empty_output(self) -> dict[str, Any]:
        return self.schema.empty()

    def to_descriptor(self, *, base_model: Optional[str] = None) -> dict[str, Any]:
        payload = {
            "name": self.name,
            "threshold": float(self.threshold),
            "schema": self.schema.to_dict(),
        }
        if self.vocabulary is not None:
            payload["vocabulary"] = self.vocabulary.to_metadata()
        if self.description:
            payload["description"] = self.description
        if base_model:
            payload["base_model"] = base_model
        return payload

    def label_names(self) -> list[str]:
        return ["O", *self.schema.label_names()]


class CompositeAgent:
    """Combine the outputs of multiple specialised agents."""

    def __init__(self, agents: Sequence[DocumentExtractionAgent]):
        self._agents = list(agents)

    def combine(self, predictions: Mapping[str, Iterable[Mapping[str, Any]]]) -> dict[str, Any]:
        combined: dict[str, Any] = {}
        for agent in self._agents:
            agent_predictions = predictions.get(agent.name, [])
            result = agent.map_predictions(agent_predictions)
            combined = self._merge(combined, result)
        return combined

    def _merge(self, base: dict[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
        merged = copy.deepcopy(base)
        for key, value in new.items():
            if key not in merged:
                merged[key] = copy.deepcopy(value)
                continue
            existing = merged[key]
            if _is_payload(value):
                if not _is_payload(existing) or float(existing.get("confidence", 0.0)) < float(
                    value.get("confidence", 0.0)
                ):
                    merged[key] = copy.deepcopy(value)
                continue
            if isinstance(value, list):
                if not isinstance(existing, list):
                    merged[key] = list(value)
                else:
                    merged[key].extend(copy.deepcopy(value))
                continue
            if isinstance(value, Mapping):
                if not isinstance(existing, Mapping):
                    merged[key] = copy.deepcopy(value)
                else:
                    merged[key] = self._merge(dict(existing), value)
                continue
            merged[key] = copy.deepcopy(value)
        return merged


def save_descriptor(path: Any, descriptor: Mapping[str, Any]) -> None:
    target = Path(path)
    target.write_text(json.dumps(descriptor, indent=2, ensure_ascii=False), encoding="utf-8")