"""Unified matched-seed trainer for MA-RDG-Mamba semantic variants."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from semantic_backbone import BackboneSpec, build_backbone


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
    from data.data_preprocessing import prepare_subset  # type: ignore
    from evaluate import probabilistic_metrics  # type: ignore
    from models.ma_rdg_mamba import MARDGMamba, gaussian_nll  # type: ignore
    return config, resolve_device, prepare_subset, probabilistic_metrics, MARDGMamba, gaussian_nll


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
    config, resolve_device, prepare_subset, metrics_fn, Model, gaussian_nll = import_original(args.original_root)
    seed_model(args.model_seed)
    device = resolve_device(config.DEVICE)
    data = prepare_subset(args.subset)
    backbone = build_backbone(
        args.original_root,
        data.sensor_cols,
        BackboneSpec(mode=args.semantic_mode, kg_seed=args.kg_seed,
                     slm_cache=args.slm_cache, shuffle_seed=args.shuffle_seed),
    )
    # Knowledge construction restores RNG state; model initialization therefore
    # depends only on model_seed and stays matched across semantic variants.
    model = Model(len(data.sensor_cols), backbone.matrix).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=config.WEIGHT_DECAY)
    train_batches = loader(data.x_train, data.c_train, data.y_train, config.BATCH_SIZE,
                           True, args.model_seed + 11)
    val_batches = loader(data.x_val, data.c_val, data.y_val, config.BATCH_SIZE,
                         False, args.model_seed + 12)
    test_batches = loader(data.x_test, data.c_test, data.y_test, config.BATCH_SIZE,
                          False, args.model_seed + 13)
    best_value, best_state = float("inf"), None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); losses = []
        for x, c, y in train_batches:
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
    y, mu, sigma = predict(model, test_batches, device)
    metrics, lower, upper = metrics_fn(y, mu, sigma, config.CI_Z)
    run_dir = (args.output_root / args.subset / args.semantic_mode /
               f"model_seed_{args.model_seed}")
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.save({"model": best_state, "a_k": backbone.matrix,
                "sensor_cols": data.sensor_cols}, run_dir / "best_model.pt")
    np.savez_compressed(run_dir / "knowledge_components.npz", a_k=backbone.matrix,
                        **backbone.components)
    np.savetxt(run_dir / "test_predictions.csv",
               np.column_stack([data.test_units, y, mu, sigma, lower, upper]), delimiter=",",
               header="unit,true_rul,mu,sigma,lower90,upper90", comments="")
    protocol = {
        "subset": args.subset, "semantic_mode": args.semantic_mode,
        "model_seed": args.model_seed, "kg_seed": args.kg_seed,
        "shuffle_seed": args.shuffle_seed, "epochs": args.epochs,
        "learning_rate": args.learning_rate, "rul_cap": config.RUL_CAP,
        "window_size": config.WINDOW_SIZE, "selection_metric": "(RMSE+MAE)/RUL_CAP",
        "backbone_audit": backbone.audit,
    }
    result = {**protocol, "best_validation_objective": best_value, **metrics}
    (run_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (run_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2)); return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--subset", choices=["FD001", "FD002", "FD003", "FD004"], default="FD001")
    parser.add_argument("--semantic-mode", choices=["transe", "slm", "slm_shuffled", "none"], default="transe")
    parser.add_argument("--slm-cache", type=Path)
    parser.add_argument("--model-seed", type=int, default=72)
    parser.add_argument("--kg-seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=1042)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
