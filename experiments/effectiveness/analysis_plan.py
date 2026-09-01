"""Shared helpers for a pre-test, fail-closed analysis-plan lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict


LOCK_FILENAME = "analysis_plan.lock.json"
PRIMARY_TREATMENT = "pretrained_no_context"
PRIMARY_CONTROL = "random_no_context"
COMPARISON_EXCLUDED_FIELDS = {"variant", "seed", "backbone_seed"}


def canonical_comparison_protocol(protocol: Dict[str, Any]) -> Dict[str, Any]:
    """Remove only the intentionally paired-varying fields."""
    return {
        key: value
        for key, value in protocol.items()
        if key not in COMPARISON_EXCLUDED_FIELDS
    }


def comparison_protocol_hash(protocol: Dict[str, Any]) -> str:
    return sha256_json(canonical_comparison_protocol(protocol))


def sha256_json(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_content_hash(plan: Dict[str, Any]) -> str:
    content = {key: value for key, value in plan.items() if key != "plan_content_sha256"}
    return sha256_json(content)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def comparable_environment_hash(payload: Dict[str, Any]) -> str:
    fields = [
        "python",
        "platform",
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "transformers",
        "cuda_available",
        "torch_cuda",
        "cuda_device_index",
        "cuda_device",
    ]
    return sha256_json({field: payload.get(field) for field in fields})


def read_locked_plan(output_root: Path) -> Dict[str, Any]:
    path = output_root / LOCK_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"Missing pre-test analysis plan: {path}. Run lock_analysis_plan.py "
            "after training and before any confirmatory test."
        )
    with path.open("r", encoding="utf-8") as handle:
        plan = json.load(handle)
    if plan.get("status") != "locked_before_test":
        raise RuntimeError(f"Analysis plan is not locked: {path}")
    expected_hash = plan_content_hash(plan)
    if plan.get("plan_content_sha256") != expected_hash:
        raise RuntimeError(f"Analysis plan content hash mismatch: {path}")
    for source_path, expected_sha in plan.get("analysis_source_sha256", {}).items():
        source = Path(source_path)
        if not source.exists() or sha256_file(source) != expected_sha:
            raise RuntimeError(f"Analysis code changed after plan lock: {source}")
    return plan


def verify_test_authorization(
    output_root: Path,
    protocol_hash: str,
    protocol: Dict[str, Any],
) -> Dict[str, Any]:
    plan = read_locked_plan(output_root)
    if Path(plan.get("output_root", "")).resolve() != output_root.resolve():
        raise RuntimeError("Analysis plan was locked for a different output root.")
    run_key = f"{protocol['variant']}:seed_{protocol['seed']}"
    expected_run = plan.get("expected_runs", {}).get(run_key)
    if expected_run is None:
        raise RuntimeError(f"Run is not preregistered in the analysis plan: {run_key}")
    if expected_run.get("protocol_hash") != protocol_hash:
        raise RuntimeError(f"Locked protocol hash mismatch for {run_key}")
    if plan.get("comparison_protocol_hash") != comparison_protocol_hash(protocol):
        raise RuntimeError(f"Comparison protocol mismatch for {run_key}")

    run_dir = output_root / "runs" / protocol["variant"] / f"seed_{protocol['seed']}"
    checkpoint_path = run_dir / "checkpoint" / "trainable_state.pth"
    train_summary_path = run_dir / "train_summary.json"
    environment_path = run_dir / "environment.json"
    audit_path = run_dir / "model_audit.json"
    for path in [checkpoint_path, train_summary_path, environment_path, audit_path]:
        if not path.exists():
            raise FileNotFoundError(f"Locked run artifact is missing: {path}")
    checkpoint_sha = sha256_file(checkpoint_path)
    if checkpoint_sha != expected_run.get("checkpoint_sha256"):
        raise RuntimeError(f"Checkpoint changed after analysis-plan lock: {checkpoint_path}")
    with train_summary_path.open("r", encoding="utf-8") as handle:
        train_summary = json.load(handle)
    if train_summary.get("checkpoint_sha256") != checkpoint_sha:
        raise RuntimeError(f"Train summary/checkpoint mismatch: {run_dir}")
    with environment_path.open("r", encoding="utf-8") as handle:
        environment = json.load(handle)
    environment_sha = comparable_environment_hash(environment)
    if environment_sha != expected_run.get("environment_hash") or environment_sha != plan.get(
        "environment_hash"
    ):
        raise RuntimeError(f"Environment changed relative to analysis-plan lock: {run_dir}")
    with audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)
    locked_audit = plan["pair_audit"][str(protocol["seed"])][protocol["variant"]]
    for field, expected_value in locked_audit.items():
        if audit.get(field) != expected_value:
            raise RuntimeError(f"Model audit changed after lock: {run_key} field={field}")
    return plan
