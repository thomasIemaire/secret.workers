from __future__ import annotations

import json
import logging
import os
import shutil
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional

from bson import ObjectId

from src.helpers.trainer import trainer as run_trainer

LOGGER = logging.getLogger(__name__)


def run_task(*, doc: Optional[Mapping[str, Any]] = None, db=None, MAX_WORKERS: int = 2) -> None:
    if not doc or db is None:
        LOGGER.warning("trainer: tâche ignorée (doc ou db manquant)")
        return

    datasets = db.get_collection("datasets")
    data_collection = db.get_collection("datasets_data")
    models = db.get_collection("models")
    agents = db.get_collection("agents")
    configs = db.get_collection("models_configurations")

    dataset_id = doc.get("_id")
    if not dataset_id:
        raise ValueError("Identifiant de dataset manquant")
    if not isinstance(dataset_id, ObjectId):
        dataset_id = ObjectId(dataset_id)

    job_id = str(dataset_id)

    try:
        total_cpus = os.cpu_count() or 4
        per_worker = max(1, total_cpus // max(1, MAX_WORKERS))
        os.environ.setdefault("OMP_NUM_THREADS", str(per_worker))
        os.environ.setdefault("MKL_NUM_THREADS", str(per_worker))

        dataset = list(data_collection.find({"dataset": dataset_id}))
        if not dataset:
            raise ValueError("Dataset vide: impossible d'entraîner le modèle")

        model_id = doc.get("model") or doc.get("model_id")
        if not model_id:
            raise ValueError("model_id manquant")
        if not isinstance(model_id, ObjectId):
            model_id = ObjectId(model_id)

        model = models.find_one({"_id": model_id})
        if not model:
            raise ValueError(f"Modèle introuvable: {model_id}")

        # --- CORRECTION ---
        # Si le modèle n'a pas d'étiquettes, on tente de les récupérer depuis la configuration
        if not model.get("labels") and not model.get("mapper"):
            LOGGER.info("Aucune étiquette dans le modèle, tentative d'inférence depuis la configuration...")
            config_id = doc.get("configuration") or model.get("configuration")
            if config_id:
                if not isinstance(config_id, ObjectId):
                    config_id = ObjectId(config_id)
                configuration = configs.find_one({"_id": config_id})
                if configuration:
                    attributes = configuration.get("attributes") or []
                    inferred_labels = [attr.get("key") for attr in attributes if attr.get("key")]
                    if inferred_labels:
                        model["labels"] = inferred_labels
                        LOGGER.info("Etiquettes inférées: %s", inferred_labels)
        # ------------------

        version = doc.get("version", "1.0")
        parameters = doc.get("parameters") or {}

        datasets.update_one(
            {"_id": dataset_id},
            {"$set": {"status": "in-training", "started_at": datetime.utcnow()}},
        )
        LOGGER.info(
            "[%s] lancement de l'entraînement du modèle %s (version %s) avec %s exemples",
            job_id,
            model.get("name"),
            version,
            len(dataset),
        )

        run_trainer(dataset, model, parameters=parameters, version=version)

        data_collection.delete_many({"dataset": dataset_id})

        datasets.update_one(
            {"_id": dataset_id},
            {"$set": {"status": "completed", "finished_at": datetime.utcnow()}},
        )

        # MODIFICATION ICI : Utilisation de .resolve() pour obtenir le chemin absolu
        target_path = (Path("sardine.agents") / doc.get("reference", "agent") / version).resolve()

        descriptor_data: dict[str, Any] = {}
        descriptor_path = target_path / "agent.json"
        if descriptor_path.exists():
            try:
                descriptor_data = json.loads(descriptor_path.read_text(encoding="utf-8"))
            except Exception as error:  # pragma: no cover - defensive logging
                LOGGER.warning("Impossible de lire le descripteur d'agent: %s", error)
                descriptor_data = {}

        payload = {
            "created_by": doc.get("created_by"),
            "created_at": datetime.utcnow(),
            "model": model_id,
            "version": version,
            "name": doc.get("name"),
            "reference": doc.get("reference"),
            "description": doc.get("description"),
            "path": str(target_path),
            "mapper": descriptor_data.get("schema") or model.get("mapper"),
            "requirements": doc.get("requirements", []),
            "status": "enabled",
        }
        if "threshold" in descriptor_data:
            payload["confidence_threshold"] = descriptor_data.get("threshold")
        if "vocabulary" in descriptor_data:
            payload["vocabulary"] = descriptor_data.get("vocabulary")
        if "base_model" in descriptor_data:
            payload["base_model"] = descriptor_data.get("base_model")

        agents.insert_one(payload)

        cleanup_checkpoints(target_path)
        LOGGER.info("[%s] entraînement terminé", job_id)

    except Exception as exc:
        error_message = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        datasets.update_one(
            {"_id": dataset_id},
            {
                "$set": {
                    "status": "train-failed",
                    "error": error_message,
                    "finished_at": datetime.utcnow(),
                }
            },
        )
        LOGGER.error("[%s] entraînement échoué: %s", job_id, error_message)


def cleanup_checkpoints(path: Path) -> None:
    if not path.exists():
        return
    for item in path.iterdir():
        if item.name.startswith("checkpoint-") and item.is_dir():
            shutil.rmtree(item, ignore_errors=True)