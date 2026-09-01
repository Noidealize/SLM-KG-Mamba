"""Generate a frozen-SLM sensor embedding cache for MA-RDG-Mamba."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cards(path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not data:
        raise ValueError("sensor card file must be a non-empty JSON list")
    sensors, texts = [], []
    for row in data:
        sensor = str(row["sensor_column"])
        text = str(row["text"]).strip()
        if not text:
            raise ValueError(f"empty text for {sensor}")
        sensors.append(sensor); texts.append(text)
    if len(set(sensors)) != len(sensors):
        raise ValueError("sensor_column values must be unique")
    return sensors, texts


@torch.inference_mode()
def encode(model_path: Path, texts: list[str], device: str, max_length: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device).eval()
    rows = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        hidden = model(**inputs).last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        rows.append(pooled[0].float().cpu().numpy())
    return np.stack(rows).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--sensor-cards", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    sensors, texts = load_cards(args.sensor_cards)
    embeddings = encode(args.model_path, texts, args.device, args.max_length)
    metadata = {
        "model_path": str(args.model_path.resolve()),
        "sensor_cards": str(args.sensor_cards.resolve()),
        "sensor_cards_sha256": _file_hash(args.sensor_cards),
        "pooling": "attention-mask mean of last_hidden_state",
        "max_length": args.max_length,
        "embedding_dim": int(embeddings.shape[1]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embeddings=embeddings,
        sensor_columns=np.asarray(sensors),
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )
    print(json.dumps({**metadata, "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

