from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
PROJECT = HERE.parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = json.loads((HERE / "formal_experiment_protocol.json").read_text(encoding="utf-8"))
    policy = json.loads((HERE / protocol["sensor_policy"]).read_text(encoding="utf-8"))
    errors = []
    sensors = policy["sensor_columns"]
    if len(sensors) != policy["sensor_count"] or len(set(sensors)) != len(sensors):
        errors.append("fixed sensor count/order is inconsistent")
    expected = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
                "s13", "s14", "s15", "s17", "s20", "s21"]
    if sensors != expected:
        errors.append("sensor order differs from audited external predictor contract")

    knowledge = protocol["knowledge_baseline"]
    manifest = (HERE / knowledge["manifest"]).resolve()
    matrix_path = (HERE / knowledge["projection_matrix"]).resolve()
    for path, expected_hash in ((manifest, knowledge["manifest_sha256"]),
                                (matrix_path, knowledge["projection_matrix_sha256"])):
        if not path.is_file() or sha256(path) != expected_hash:
            errors.append(f"missing or changed frozen artifact: {path}")
    if matrix_path.is_file():
        with np.load(matrix_path, allow_pickle=False) as payload:
            matrix_sensors = payload["sensor_columns"].astype(str).tolist()
            matrix = payload["matrix"]
        if matrix_sensors != sensors or matrix.shape != (14, 14):
            errors.append("projection matrix violates the fixed14 contract")

    # Read train files only. Test and RUL files are intentionally neither resolved nor opened.
    data_root = PROJECT / "data" / "CMAPSS"
    train_audit = {}
    for subset in policy["subsets"]:
        path = data_root / f"train_{subset}.txt"
        if not path.is_file():
            errors.append(f"missing training file for {subset}")
            continue
        first_row = path.open("r", encoding="utf-8").readline().split()
        if len(first_row) < 26:
            errors.append(f"{subset} has fewer than the expected 26 columns")
        train_audit[subset] = {"sha256": sha256(path), "first_row_columns": len(first_row)}

    report = {
        "valid": not errors,
        "errors": errors,
        "protocol_id": protocol["protocol_id"],
        "sensor_policy_id": policy["policy_id"],
        "sensor_count": len(sensors),
        "subsets": policy["subsets"],
        "train_files": train_audit,
        "official_test_used": False,
        "official_rul_used": False,
        "ready_for_pilot": not errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
