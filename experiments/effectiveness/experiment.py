"""Training and one-shot evaluation for the isolated effectiveness experiment."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from data_provider.data_factory import data_dict
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.device import resolve_device

from effectiveness_model import ControlledSMETimes, VARIANT_SPECS


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Confirmatory runs fail closed if an operation has no deterministic
        # implementation.  A warning would silently weaken the paired-seed
        # design and make an apparently reproducible run non-reproducible.
        torch.use_deterministic_algorithms(True, warn_only=False)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SplitLocalContextPermutation(Dataset):
    """Pairs each sample with another context from the same split.

    The permutation is fixed by seed and is independent of DataLoader order.
    Targets and numerical inputs remain untouched.  This preserves the context
    marginal distribution while breaking sample-context alignment.
    """

    def __init__(self, base: Dataset, seed: int):
        self.base = base
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(len(base))
        if len(base) > 1 and np.all(permutation == np.arange(len(base))):
            permutation = np.roll(permutation, 1)
        self.permutation = permutation

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        seq_x, seq_y, _, _ = self.base[index]
        _, _, permuted_x_mark, permuted_y_mark = self.base[
            int(self.permutation[index])
        ]
        return seq_x, seq_y, permuted_x_mark, permuted_y_mark


def _base_dataset(dataset: Dataset) -> Dataset:
    return dataset.base if isinstance(dataset, SplitLocalContextPermutation) else dataset


class EffectivenessExperiment(Exp_Long_Term_Forecast):
    """No-test-peeking trainer with deterministic, split-specific loaders."""

    def _build_model(self):
        if self.args.use_multi_gpu:
            raise ValueError("This evidence runner intentionally supports one GPU per run.")
        self.device = resolve_device(self.args)
        return ControlledSMETimes(self.args).to(self.device)

    def _get_data(self, flag: str):
        Data = data_dict[self.args.data]
        if flag in {"train", "val"}:
            size = [self.args.seq_len, self.args.label_len, self.args.token_len]
        else:
            size = [
                self.args.test_seq_len,
                self.args.test_label_len,
                self.args.test_pred_len,
            ]

        dataset = Data(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag=flag,
            size=size,
            seasonal_patterns=self.args.seasonal_patterns,
            drop_short=self.args.drop_short,
        )

        split_offset = {"train": 0, "val": 10_000, "test": 20_000}[flag]
        if VARIANT_SPECS[self.args.variant].permute_context:
            dataset = SplitLocalContextPermutation(
                dataset, seed=self.args.seed + 50_000 + split_offset
            )

        generator = torch.Generator()
        generator.manual_seed(self.args.seed + split_offset)
        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=flag == "train",
            num_workers=self.args.num_workers,
            drop_last=self.args.drop_last if flag == "train" else False,
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.args.num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )
        print(f"{flag}: {len(dataset)} samples, {len(loader)} batches")
        return dataset, loader

    def train_validation_only(self, run_dir: Path) -> Dict[str, Any]:
        """Train with validation early stopping and never instantiate test data."""
        set_global_seed(self.args.seed, self.args.deterministic)
        train_data, train_loader = self._get_data("train")
        val_data, val_loader = self._get_data("val")
        del train_data, val_data

        checkpoint_dir = run_dir / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "trainable_state.pth"

        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args.tmax, eta_min=1e-8
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=self.args.use_amp and self.device.type == "cuda"
        )

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()

        best_val = math.inf
        best_epoch = 0
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, self.args.train_epochs + 1):
            self.model.train()
            loss_sum = 0.0
            sample_count = 0
            epoch_started = time.perf_counter()

            for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if self.args.debug_max_batches and batch_index >= self.args.debug_max_batches:
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.args.use_amp and self.device.type == "cuda",
                ):
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    loss = criterion(outputs, batch_y)

                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite loss in epoch {epoch}; the run is invalid."
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                batch_n = int(batch_x.shape[0])
                loss_sum += float(loss.detach().cpu()) * batch_n
                sample_count += batch_n

            train_mse = loss_sum / sample_count
            val_mse = self._validation_mse(val_loader, criterion)
            epoch_seconds = time.perf_counter() - epoch_started
            history.append(
                {
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "seconds": epoch_seconds,
                }
            )
            print(
                f"epoch={epoch} train_mse={train_mse:.8f} "
                f"val_mse={val_mse:.8f} seconds={epoch_seconds:.1f}"
            )

            if val_mse < best_val:
                best_val = val_mse
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_trainable_checkpoint(checkpoint_path)
            else:
                epochs_without_improvement += 1

            if self.args.cosine:
                scheduler.step()
            if epochs_without_improvement >= self.args.patience:
                print(f"early stopping at epoch {epoch}")
                break

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        train_seconds = time.perf_counter() - started
        train_peak_mib = self._peak_memory_mib()
        self._load_trainable_checkpoint(checkpoint_path)

        checkpoint_sha256 = sha256_file(checkpoint_path)
        summary = {
            "status": "trained_validation_only",
            "variant": self.args.variant,
            "seed": self.args.seed,
            "best_val_mse": best_val,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "train_seconds": train_seconds,
            "train_peak_allocated_mib": train_peak_mib,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
            "diagnostic_truncated": bool(self.args.debug_max_batches),
            "history": history,
        }
        _write_json(run_dir / "train_summary.json", summary)
        return summary

    def evaluate_test_once(self, run_dir: Path) -> Dict[str, Any]:
        if self.args.test_pred_len != self.args.token_len:
            raise ValueError(
                "Evidence phase requires test_pred_len == token_len (96). "
                "Longer rolling forecasts reuse future statistic embeddings upstream."
            )

        checkpoint_path = run_dir / "checkpoint" / "trainable_state.pth"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing trained checkpoint: {checkpoint_path}")
        train_summary_path = run_dir / "train_summary.json"
        if not train_summary_path.exists():
            raise FileNotFoundError(f"Missing training summary: {train_summary_path}")
        with train_summary_path.open("r", encoding="utf-8") as handle:
            train_summary = json.load(handle)
        actual_checkpoint_sha256 = sha256_file(checkpoint_path)
        if actual_checkpoint_sha256 != train_summary.get("checkpoint_sha256"):
            raise RuntimeError(f"Checkpoint SHA256 does not match train summary: {checkpoint_path}")
        self._load_trainable_checkpoint(checkpoint_path)
        test_data, test_loader = self._get_data("test")

        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        predictions = []
        targets = []
        self.model.eval()
        with torch.no_grad():
            for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if self.args.debug_max_batches and batch_index >= self.args.debug_max_batches:
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.args.use_amp and self.device.type == "cuda",
                ):
                    output = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                prediction = output[:, -self.args.test_pred_len :, :]
                target = batch_y[:, -self.args.test_pred_len :, :]
                predictions.append(prediction.detach().float().cpu())
                targets.append(target.detach().float().cpu())

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        test_seconds = time.perf_counter() - started
        test_peak_mib = self._peak_memory_mib()

        pred = torch.cat(predictions, dim=0).numpy()
        true = torch.cat(targets, dim=0).numpy()
        mse = float(np.mean(np.square(pred - true), dtype=np.float64))
        mae = float(np.mean(np.abs(pred - true), dtype=np.float64))
        rmse = float(math.sqrt(mse))

        _atomic_save_npy(run_dir / "predictions.npy", pred)
        _atomic_save_npy(run_dir / "targets.npy", true)
        sample_mse = np.mean(np.square(pred - true), axis=(1, 2), dtype=np.float64)
        sample_mae = np.mean(np.abs(pred - true), axis=(1, 2), dtype=np.float64)
        base = _base_dataset(test_data)
        total_length = int(getattr(base, "tot_len", len(base)))
        indices = np.arange(pred.shape[0], dtype=np.int64)
        _atomic_savez_compressed(
            run_dir / "errors_by_sample.npz",
            sample_mse=sample_mse,
            sample_mae=sample_mae,
            feature_id=indices // total_length,
            forecast_origin=indices % total_length,
        )

        metrics = {
            "status": "test_evaluated_once",
            "variant": self.args.variant,
            "seed": self.args.seed,
            "mse": mse,
            "mae": mae,
            "rmse": rmse,
            "test_samples": int(pred.shape[0]),
            "expected_full_test_samples": int(len(base)),
            "prediction_shape": list(pred.shape),
            "test_seconds": test_seconds,
            "test_samples_per_second": float(pred.shape[0] / test_seconds),
            "test_peak_allocated_mib": test_peak_mib,
            "fusion_alpha": self._fusion_alpha(),
            "diagnostic_truncated": bool(self.args.debug_max_batches),
            "predictions_sha256": sha256_file(run_dir / "predictions.npy"),
            "targets_sha256": sha256_file(run_dir / "targets.npy"),
            "errors_by_sample_sha256": sha256_file(run_dir / "errors_by_sample.npz"),
        }
        _write_json(run_dir / "metrics.json", metrics)
        return metrics

    def _validation_mse(self, loader: DataLoader, criterion: nn.Module) -> float:
        weighted_loss = 0.0
        sample_count = 0
        self.model.eval()
        with torch.no_grad():
            for batch_index, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(loader):
                if self.args.debug_max_batches and batch_index >= self.args.debug_max_batches:
                    break
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.float().to(self.device, non_blocking=True)
                batch_x_mark = batch_x_mark.float().to(self.device, non_blocking=True)
                batch_y_mark = batch_y_mark.float().to(self.device, non_blocking=True)
                with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=self.args.use_amp and self.device.type == "cuda",
                ):
                    outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)
                    # Select checkpoints on the same 96-step endpoint used at test.
                    # Training still follows upstream's full next-patch objective.
                    outputs = outputs[:, -self.args.test_pred_len :, :]
                    forecast_target = batch_y[:, -self.args.test_pred_len :, :]
                    loss = criterion(outputs, forecast_target)
                batch_n = int(batch_x.shape[0])
                weighted_loss += float(loss.detach().cpu()) * batch_n
                sample_count += batch_n
        self.model.train()
        if sample_count == 0:
            raise RuntimeError("Validation loader produced no samples.")
        return weighted_loss / sample_count

    def _save_trainable_checkpoint(self, path: Path) -> None:
        trainable_names = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        state = {
            name: value.detach().cpu()
            for name, value in self.model.state_dict().items()
            if name in trainable_names
        }
        temporary = _temporary_path(path)
        torch.save(state, temporary)
        os.replace(temporary, path)

    def _load_trainable_checkpoint(self, path: Path) -> None:
        try:
            state = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(path, map_location=self.device)
        expected_keys = {
            name for name, parameter in self.model.named_parameters() if parameter.requires_grad
        }
        actual_keys = set(state)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unexpected = sorted(actual_keys - expected_keys)
            raise RuntimeError(
                f"Trainable checkpoint key mismatch for {path}; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
        incompatible = self.model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(
                f"Unexpected checkpoint keys for {path}: {incompatible.unexpected_keys[:5]}"
            )

    def _peak_memory_mib(self) -> float:
        if self.device.type != "cuda":
            return 0.0
        return float(torch.cuda.max_memory_allocated(self.device) / (1024**2))

    def _fusion_alpha(self):
        gate = getattr(self.model, "fusion_gate", None)
        if gate is None:
            return None
        return float(torch.sigmoid(gate.detach()).cpu())


def model_audit(model: nn.Module) -> Dict[str, Any]:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for _, parameter in trainable)
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "frozen_params": int(total_params - trainable_params),
        "trainable_signature": [
            {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in trainable
        ],
        "trainable_initial_sha256": _parameter_hash(trainable, sample_only=False),
        "backbone_sample_sha256": _parameter_hash(
            [
                (name, parameter)
                for name, parameter in model.named_parameters()
                if not parameter.requires_grad and name.startswith("llama.model.")
            ],
            sample_only=True,
        ),
    }


def _parameter_hash(
    named_parameters: Iterable[Tuple[str, torch.Tensor]], sample_only: bool
) -> str | None:
    digest = hashlib.sha256()
    found = False
    for name, parameter in named_parameters:
        found = True
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        flat = parameter.detach().reshape(-1)
        if sample_only:
            flat = flat[: min(flat.numel(), 64)]
        array = flat.float().cpu().numpy()
        digest.update(array.tobytes())
    return digest.hexdigest() if found else None


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_path(path)
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _temporary_path(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.tmp")


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    temporary = _temporary_path(path)
    with temporary.open("wb") as handle:
        np.save(handle, array)
    os.replace(temporary, path)


def _atomic_savez_compressed(path: Path, **arrays: np.ndarray) -> None:
    temporary = _temporary_path(path)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
