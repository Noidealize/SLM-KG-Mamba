"""C-MAPSS train/validation-only loader that never opens official test files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrainValidationData:
    x_train: np.ndarray
    c_train: np.ndarray
    y_train: np.ndarray
    train_units: np.ndarray
    x_val: np.ndarray
    c_val: np.ndarray
    y_val: np.ndarray
    val_units: np.ndarray
    sensor_cols: list[str]
    audit: dict


def _windows(frame, sensors, conditions, window_size):
    xs, cs, ys, units = [], [], [], []
    for unit, group in frame.groupby("unit", sort=True):
        group = group.sort_values("cycle")
        x = group[sensors].to_numpy(np.float32)
        c = group[conditions].to_numpy(np.float32)
        y = group["RUL"].to_numpy(np.float32)
        for end in range(window_size, len(group) + 1):
            xs.append(x[end-window_size:end]); cs.append(c[end-window_size:end])
            ys.append(y[end-1]); units.append(unit)
    return np.stack(xs), np.stack(cs), np.asarray(ys, np.float32), np.asarray(units)


def prepare_train_validation(data_root: Path, subset: str, config) -> TrainValidationData:
    train_path = Path(data_root) / f"train_{subset}.txt"
    if not train_path.is_file():
        raise FileNotFoundError(f"missing C-MAPSS training file: {train_path}")
    frame = pd.read_csv(train_path, sep=r"\s+", header=None, engine="python")
    if frame.shape[1] < len(config.RAW_COLUMNS):
        raise ValueError("training file has fewer columns than RAW_COLUMNS")
    frame = frame.iloc[:, :len(config.RAW_COLUMNS)]
    frame.columns = config.RAW_COLUMNS
    frame["RUL"] = (frame.groupby("unit")["cycle"].transform("max") - frame["cycle"]).clip(
        upper=config.RUL_CAP
    )
    units = np.asarray(sorted(frame.unit.unique()))
    rng = np.random.default_rng(config.DATA_SPLIT_SEED); rng.shuffle(units)
    n_val = max(1, round(len(units) * config.VALIDATION_UNIT_RATIO))
    val_ids = set(units[:n_val].tolist())
    fit = frame[~frame.unit.isin(val_ids)].copy()
    val = frame[frame.unit.isin(val_ids)].copy()
    if set(fit.unit.unique()).intersection(val.unit.unique()):
        raise RuntimeError("engine-level split leaked units")
    sensors = list(config.USED_SENSOR_COLUMNS)
    conditions = list(config.CONDITION_COLUMNS)
    stats = {column: (float(fit[column].min()), float(fit[column].max()))
             for column in sensors + conditions}
    lo, hi = config.NORM_RANGE
    for target in (fit, val):
        for column, (minimum, maximum) in stats.items():
            target[column] = ((target[column] - minimum) / max(maximum - minimum, 1e-8)) * (hi - lo) + lo
    xtr, ctr, ytr, utr = _windows(fit, sensors, conditions, config.WINDOW_SIZE)
    xva, cva, yva, uva = _windows(val, sensors, conditions, config.WINDOW_SIZE)
    return TrainValidationData(xtr, ctr, ytr, utr, xva, cva, yva, uva, sensors, {
        "subset": subset, "split_unit": "engine", "normalization_fit": "training_engines_only",
        "official_test_used": False, "train_path": str(train_path.resolve()),
        "train_data_sha256": _sha256_file(train_path),
        "train_engine_count": int(len(set(utr.tolist()))),
        "validation_engine_count": int(len(set(uva.tolist()))),
    })
