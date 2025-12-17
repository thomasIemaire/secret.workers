#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def _load_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_texts(args: argparse.Namespace) -> List[str]:
    if args.text:
        return [args.text.strip()]
    if args.file:
        p = Path(args.file)
        lines = p.read_text(encoding="utf-8").splitlines()
        return [l.strip() for l in lines if l.strip()]
    # stdin fallback
    import sys
    data = sys.stdin.read()
    data = (data or "").strip()
    return [data] if data else []


def _pick_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[WARN] CUDA demandé mais indisponible -> CPU")
        return torch.device("cpu")
    if device_arg == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[WARN] MPS demandé mais indisponible -> CPU")
        return torch.device("cpu")
    return torch.device("cpu")


def _load_threshold(model_dir: Path, override: Optional[float]) -> float:
    if override is not None:
        return float(override)
    thr_file = model_dir / "probability_threshold.json"
    if thr_file.exists():
        data = _load_json(thr_file) or {}
        if "threshold" in data:
            return float(data["threshold"])
    return 0.5


# ---------------------------
# GLiNER
# ---------------------------
def _infer_gliner(
    model_dir: Path,
    texts: List[str],
    labels: List[str],
    threshold: float,
    device: torch.device,
) -> List[Dict[str, Any]]:
    try:
        from gliner import GLiNER
    except ImportError as e:
        raise SystemExit(
            "GLiNER non installé. Fais: pip install gliner"
        ) from e

    model = GLiNER.from_pretrained(str(model_dir))
    model.to(str(device))

    outputs: List[Dict[str, Any]] = []
    for text in texts:
        ents = model.predict_entities(text, labels, threshold=threshold)
        # ents: list of dicts: {start, end, text, label, score}
        outputs.append({"text": text, "entities": ents})
    return outputs


def _guess_gliner_labels(model_dir: Path) -> List[str]:
    """
    Heuristique: essaie de trouver une liste de labels dans des fichiers courants.
    Si rien trouvé, on renverra [] (et l'utilisateur devra fournir --labels).
    """
    # 1) gliner_config.json (parfois contient une clé utile)
    cfg = _load_json(model_dir / "gliner_config.json") or {}
    for key in ("labels", "entity_labels", "ner_labels"):
        val = cfg.get(key)
        if isinstance(val, list) and all(isinstance(x, str) for x in val):
            return [x for x in val if x.strip()]

    # 2) schema.json / agent.json (si présents)
    schema = _load_json(model_dir / "schema.json") or {}
    # selon ta structure, les labels peuvent être imbriqués ; on tente quelques chemins
    for key in ("entity_labels", "labels"):
        val = schema.get(key)
        if isinstance(val, list):
            return [str(x) for x in val if str(x).strip()]

    agent = _load_json(model_dir / "agent.json") or {}
    for key in ("labels", "entity_labels"):
        val = agent.get(key)
        if isinstance(val, list):
            return [str(x) for x in val if str(x).strip()]

    return []


# ---------------------------
# CamemBERT TokenClassification (HF)
# ---------------------------
def _infer_hf_token_cls(
    model_dir: Path,
    texts: List[str],
    threshold: float,
    device: torch.device,
    max_length: int,
) -> List[Dict[str, Any]]:
    from transformers import CamembertForTokenClassification, CamembertTokenizerFast, pipeline

    tokenizer = CamembertTokenizerFast.from_pretrained(str(model_dir))
    model = CamembertForTokenClassification.from_pretrained(str(model_dir))
    model.eval()
    model.to(device)

    # pipeline: regroupe automatiquement les sous-tokens en entités
    device_id = 0 if device.type == "cuda" else -1
    nlp = pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device_id,
    )

    outputs: List[Dict[str, Any]] = []
    for text in texts:
        preds = nlp(text, truncation=True, max_length=max_length)
        # preds: list[dict] avec start/end/word/entity_group/score
        ents = []
        for p in preds:
            score = float(p.get("score", 0.0))
            if score < threshold:
                continue
            ents.append(
                {
                    "start": int(p["start"]),
                    "end": int(p["end"]),
                    "text": text[int(p["start"]) : int(p["end"])],
                    "label": p.get("entity_group") or p.get("entity") or "UNK",
                    "score": score,
                }
            )
        outputs.append({"text": text, "entities": ents})
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser(description="Test/inférence sur checkpoint GLiNER ou CamemBERT TokenClassification")
    ap.add_argument("--model_dir", type=str, required=True, help="Dossier du modèle (ex: checkpoint-10000 ou sardine.agents/.../1.0)")
    ap.add_argument("--text", type=str, default=None, help="Texte à tester")
    ap.add_argument("--file", type=str, default=None, help="Fichier texte (1 ligne = 1 exemple)")
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--threshold", type=float, default=None, help="Seuil score; sinon lit probability_threshold.json si présent, sinon 0.5")
    ap.add_argument("--max_length", type=int, default=512, help="Max tokens (HF token classification)")
    ap.add_argument("--labels", type=str, default=None, help="Labels GLiNER séparés par des virgules (ex: PERSON,ORG,DATE)")
    ap.add_argument("--json_out", type=str, default=None, help="Si défini, écrit la sortie JSON dans ce fichier")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise SystemExit(f"model_dir introuvable: {model_dir}")

    texts = _read_texts(args)
    if not texts:
        raise SystemExit("Aucun texte fourni. Utilise --text, --file, ou pipe stdin.")

    device = _pick_device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))
    threshold = _load_threshold(model_dir, args.threshold)

    is_gliner = (model_dir / "gliner_config.json").exists()
    if is_gliner:
        if args.labels:
            labels = [x.strip() for x in args.labels.split(",") if x.strip()]
        else:
            labels = _guess_gliner_labels(model_dir)

        if not labels:
            raise SystemExit(
                "Checkpoint GLiNER détecté, mais impossible de deviner les labels.\n"
                "Relance avec: --labels \"LABEL1,LABEL2,...\""
            )

        results = _infer_gliner(model_dir, texts, labels=labels, threshold=threshold, device=device)
    else:
        results = _infer_hf_token_cls(model_dir, texts, threshold=threshold, device=device, max_length=args.max_length)

    # Affichage
    for i, item in enumerate(results, start=1):
        print(f"\n=== Exemple {i} ===")
        print(item["text"])
        if not item["entities"]:
            print("Aucune entité.")
            continue
        for e in item["entities"]:
            print(f"- {e['label']:>12} [{e['start']:>4}:{e['end']:<4}]  score={e['score']:.3f}  -> {repr(e['text'])}")

    # JSON output
    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] JSON écrit dans: {out_path}")


if __name__ == "__main__":
    main()
