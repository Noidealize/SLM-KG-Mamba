"""Unified matched-seed trainer for MA-RDG-Mamba semantic variants."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import hashlib

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from semantic_backbone import BackboneSpec, build_backbone
from data_protocol import prepare_train_validation
from graph_paths import GraphPathSpec
from predictor_adapter import create_feasibility_model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_model(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_original(root: Path):
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import config  # type: ignore
    from cuda_compat import resolve_device  # type: ignore
    from models.ma_rdg_mamba import MARDGMamba, gaussian_nll  # type: ignore
    return config, resolve_device, MARDGMamba, gaussian_nll


def loader(x, c, y, batch_size: int, shuffle: bool, seed: int):
    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(c), torch.from_numpy(y))
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator,
                      num_workers=0, pin_memory=torch.cuda.is_available())


@torch.no_grad()
def predict(model, batches, device):
    model.eval(); ys, mus, sigmas = [], [], []
    for x, c, y in batches:
        mu, log_var = model(x.to(device), c.to(device))
        ys.append(y.numpy()); mus.append(mu.cpu().numpy())
        sigmas.append(torch.exp(0.5 * log_var).cpu().numpy())
    return np.concatenate(ys), np.concatenate(mus), np.concatenate(sigmas)


def train(args) -> dict:
    config, resolve_device, Model, gaussian_nll = import_original(args.original_root)
    seed_model(args.model_seed)
    device = resolve_device(config.DEVICE)
    data = prepare_train_validation(args.data_root, args.subset, config)
    backbone = build_backbone(
        args.original_root,
        data.sensor_cols,
        BackboneSpec(mode=args.semantic_mode, kg_seed=args.kg_seed,
                     slm_cache=args.slm_cache, shuffle_seed=args.shuffle_seed,
                     reference_kg_matrix=args.reference_kg_matrix),
    )
    # Knowledge construction restores RNG state; model initialization therefore
    # depends only on model_seed and stays matched across semantic variants.
    model = create_feasibility_model(
        Model, len(data.sensor_cols), backbone.matrix,
        GraphPathSpec(args.graph_path, args.fixed_fusion_alpha),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=config.WEIGHT_DECAY)
    train_batches = loader(data.x_train, data.c_train, data.y_train, config.BATCH_SIZE,
                           True, args.model_seed + 11)
    val_batches = loader(data.x_val, data.c_val, data.y_val, config.BATCH_SIZE,
                         False, args.model_seed + 12)
    best_value, best_state = float("inf"), None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for batch_index, (x, c, y) in enumerate(train_batches):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            x, c, y = x.to(device), c.to(device), y.to(device)
            mu, log_var = model(x, c)
            loss = gaussian_nll(mu, log_var, y)
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            optimizer.step(); losses.append(float(loss.item()))
        model.eval(); val_y, val_mu, val_nll = [], [], []
        with torch.no_grad():
            for x, c, y in val_batches:
                x, c, y = x.to(device), c.to(device), y.to(device)
                mu, log_var = model(x, c)
                val_nll.append(float(gaussian_nll(mu, log_var, y).item()))
                val_y.append(y.cpu().numpy()); val_mu.append(mu.cpu().numpy())
        yv, muv = np.concatenate(val_y), np.concatenate(val_mu)
        rmse = float(np.sqrt(np.mean((yv - muv) ** 2)))
        mae = float(np.mean(np.abs(yv - muv)))
        selection = (rmse + mae) / float(config.RUL_CAP)
        history.append({"epoch": epoch, "train_nll": float(np.mean(losses)),
                        "val_nll": float(np.mean(val_nll)), "val_rmse": rmse,
                        "val_mae": mae, "selection": selection})
        if selection < best_value:
            best_value = selection
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    config_payload = {
        "subset": args.subset, "semantic_mode": args.semantic_mode,
        "graph_path": args.graph_path, "model_seed": args.model_seed,
        "kg_seed": args.kg_seed, "shuffle_seed": args.shuffle_seed,
        "epochs": args.epochs, "learning_rate": args.learning_rate,
        "fixed_fusion_alpha": args.fixed_fusion_alpha,
        "max_train_batches": args.max_train_batches,
        "backbone_matrix_sha256": backbone.audit["matrix_sha256"],
        "train_data_sha256": data.audit["train_data_sha256"],
    }
    config_hash = hashlib.sha256(json.dumps(config_payload, sort_keys=True).encode()).hexdigest()[:12]
    run_dir = (args.output_root / args.subset / args.semantic_mode / args.graph_path /
               f"model_seed_{args.model_seed}" / f"kg_seed_{args.kg_seed}" / config_hash)
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = run_dir / "best_model.pt"
    torch.save({"model": best_state, "a_k": torch.as_tensor(backbone.matrix),
                "sensor_cols": data.sensor_cols}, checkpoint_path)
    reloaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(reloaded["model"])
    np.savez_compressed(run_dir / "knowledge_components.npz", a_k=backbone.matrix,
                        **backbone.components)
    protocol = {
        "subset": args.subset, "semantic_mode": args.semantic_mode,
        "model_seed": args.model_seed, "kg_seed": args.kg_seed,
        "shuffle_seed": args.shuffle_seed, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "rul_cap": config.RUL_CAP,
        "window_size": config.WINDOW_SIZE, "selection_metric": "(RMSE+MAE)/RUL_CAP",
        "backbone_audit": backbone.audit,
        "graph_path": args.graph_path,
        "fixed_fusion_alpha": args.fixed_fusion_alpha,
        "config_hash": config_hash,
        "data_audit": data.audit,
        "experiment_status": "diagnostic_feasibility_only",
        "paper_evidence": False,
        "official_test_used": False,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_reload_verified": True,
    }
    result = {**protocol, "best_validation_objective": best_value}
    (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="C-MAPSS directory; only train_<subset>.txt is opened")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--semantic-mode", choices=["transe", "slm", "slm_shuffled", "none",
                        "semantic_similarity_slm", "semantic_similarity_shuffled", "no_semantic",
                        "reference_kg_projected", "no_kg"], default="transe")
    parser.add_argument("--slm-cache", type=Path)
    parser.add_argument("--reference-kg-matrix", type=Path)
    parser.add_argument("--model-seed", type=int, default=72)
    parser.add_argument("--kg-seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=1042)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--graph-path", choices=["F0", "F1", "F2", "F3"], default="F0")
    parser.add_argument("--fixed-fusion-alpha", type=float, default=0.5)
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
