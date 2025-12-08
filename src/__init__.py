"""Application entry-point and worker orchestration."""

from __future__ import annotations

import logging
import os
import signal
import time
from threading import Event

from dotenv import load_dotenv
from pymongo import MongoClient, ReturnDocument

from src.models.worker import Worker
from src.tasks.builder import run_task as run_builder
from src.tasks.trainer import run_task as run_trainer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s: %(message)s",
)

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI must be defined in the environment")

MAX_WORKERS_BUILDER = int(os.getenv("MAX_WORKERS_BUILDER", "5"))
MAX_WORKERS_TRAINER = int(os.getenv("MAX_WORKERS_TRAINER", "2"))
POLL_DELAY = float(os.getenv("POLL_DELAY", "2"))

client = MongoClient(MONGO_URI)
db = client.get_database()

stop_event = Event()


def _stop(*_: object) -> None:
    """Signal handler that requests a graceful shutdown."""

    stop_event.set()


signal.signal(signal.SIGTERM, _stop)
signal.signal(signal.SIGINT, _stop)


def claim_one_status(status: str, new_status: str):
    """Atomically claim a dataset with the given status and update it."""

    return db.get_collection("datasets").find_one_and_update(
        {"status": status},
        {"$set": {"status": new_status}},
        sort=[("created_at", 1)],
        return_document=ReturnDocument.AFTER,
    )


def main() -> None:
    """Start background workers and keep the process alive."""

    logging.info("Lancement des workers…")

    workers = [
        Worker(
            "builder",
            lambda: claim_one_status("to-build", "in-building"),
            run_builder,
            MAX_WORKERS_BUILDER,
            POLL_DELAY,
            stop_event,
            db=db,
        ).start(),
        Worker(
            "trainer",
            lambda: claim_one_status("to-train", "in-training"),
            run_trainer,
            MAX_WORKERS_TRAINER,
            POLL_DELAY,
            stop_event,
            db=db,
        ).start(),
    ]

    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logging.info("Signal reçu, arrêt en cours…")
    finally:
        stop_event.set()
        for worker in workers:
            worker.join()
        logging.info("Arrêt terminé.")
