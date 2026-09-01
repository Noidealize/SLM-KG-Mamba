"""Lock the primary analysis plan after training and before any test access."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from analysis_plan import (
    LOCK_FILENAME,
    PRIMARY_CONTROL,
    PRIMARY_TREATMENT,
    comparison_protocol_hash,
    plan_content_hash,
    sha256_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--expected-seeds", nargs="+", type=int, default=list(range(2025, 2035))
    )
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.01)
    parser.add_argument("--mae-noninferiority-margin", type=float, default=0.01)
    parser.add_argument("--minimum-win-fraction", type=float, default=0.80)
    return parser


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def environment_hash(payload: Dict[str, Any]) -> str:
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


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    lock_path = output_root / LOCK_FILENAME
    if lock_path.exists():
        raise FileExistsError(
            f"Analysis plan already exists and will not be overwritten: {lock_path}"
        )
    if not 0 < args.minimum_relative_improvement < 1:
        raise ValueError("minimum_relative_improvement must be between 0 and 1.")
    if not 0 <= args.mae_noninferiority_margin < 1:
        raise ValueError("mae_noninferiority_margin must be in [0, 1).")
    if not 0 < args.minimum_win_fraction <= 1:
        raise ValueError("minimum_win_fraction must be in (0, 1].")
    expected_seeds = sorted(set(args.expected_seeds))
    if len(expected_seeds) < 10:
        raise ValueError("Confirmatory plan requires at least 10 distinct matched seeds.")

    existing_test_files = list((output_root / "runs").glob("*/seed_*/metrics.json"))
    if existing_test_files:
        raise RuntimeError(
            "Cannot lock after test evidence exists. Use a fresh output root; first file: "
            f"{existing_test_files[0]}"
        )

    expected_runs: Dict[str, Any] = {}
    comparison_hashes = set()
    environment_hashes = set()
    pair_audit: Dict[str, Any] = {}
    for seed in expected_seeds:
        per_variant = {}
        for variant in [PRIMARY_TREATMENT, PRIMARY_CONTROL]:
            run_dir = output_root / "runs" / variant / f"seed_{seed}"
            config_path = run_dir / "config.json"
            audit_path = run_dir / "model_audit.json"
            train_path = run_dir / "train_summary.json"
            environment_path = run_dir / "environment.json"
            checkpoint_path = run_dir / "checkpoint" / "trainable_state.pth"
            for path in [config_path, audit_path, train_path, environment_path, checkpoint_path]:
                if not path.exists():
                    raise FileNotFoundError(f"Incomplete preregistered run: {path}")

            config = read_json(config_path)
            protocol = config["protocol"]
            audit = read_json(audit_path)
            train = read_json(train_path)
            run_environment_hash = environment_hash(read_json(environment_path))
            environment_hashes.add(run_environment_hash)
            if protocol.get("variant") != variant or int(protocol.get("seed")) != seed:
                raise RuntimeError(f"Config identity mismatch in {run_dir}")
            if protocol.get("deterministic") is not True:
                raise RuntimeError(f"Confirmatory run is not deterministic: {run_dir}")
            if int(protocol.get("debug_max_batches", 0)) != 0:
                raise RuntimeError(f"Diagnostic/truncated run cannot be locked: {run_dir}")
            if bool(train.get("diagnostic_truncated")):
                raise RuntimeError(f"Diagnostic train summary cannot be locked: {run_dir}")
            checkpoint_sha = sha256_file(checkpoint_path)
            if checkpoint_sha != train.get("checkpoint_sha256"):
                raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint_path}")

            comp_hash = comparison_protocol_hash(protocol)
            comparison_hashes.add(comp_hash)
            run_key = f"{variant}:seed_{seed}"
            expected_runs[run_key] = {
                "protocol_hash": config["protocol_hash"],
                "checkpoint_sha256": checkpoint_sha,
                "environment_hash": run_environment_hash,
            }
            per_variant[variant] = {
                "trainable_initial_sha256": audit.get("trainable_initial_sha256"),
                "total_params": audit.get("total_params"),
                "trainable_params": audit.get("trainable_params"),
                "backbone_sample_sha256": audit.get("backbone_sample_sha256"),
            }

        treatment = per_variant[PRIMARY_TREATMENT]
        control = per_variant[PRIMARY_CONTROL]
        for field in ["trainable_initial_sha256", "total_params", "trainable_params"]:
            if treatment[field] != control[field]:
                raise RuntimeError(f"Primary pair seed={seed} mismatches on {field}")
        if treatment["backbone_sample_sha256"] == control["backbone_sample_sha256"]:
            raise RuntimeError(
                f"Primary pair seed={seed} has identical pretrained/random backbone hashes."
            )
        pair_audit[str(seed)] = per_variant

    if len(comparison_hashes) != 1:
        raise RuntimeError(
            "Runs do not share one comparison protocol after excluding variant/seed/backbone_seed."
        )
    if len(environment_hashes) != 1:
        raise RuntimeError("Primary runs were not trained in one matched software/hardware environment.")

    pretrained_hashes = {
        pair_audit[str(seed)][PRIMARY_TREATMENT]["backbone_sample_sha256"]
        for seed in expected_seeds
    }
    random_hashes = {
        pair_audit[str(seed)][PRIMARY_CONTROL]["backbone_sample_sha256"]
        for seed in expected_seeds
    }
    if None in pretrained_hashes or None in random_hashes:
        raise RuntimeError("A primary run is missing its frozen-backbone audit hash.")
    if len(pretrained_hashes) != 1:
        raise RuntimeError(
            "Pretrained frozen-backbone hash changed across matched training seeds."
        )
    if len(random_hashes) != len(expected_seeds):
        raise RuntimeError(
            "Random frozen-backbone hashes must be unique across backbone seeds."
        )
    if pretrained_hashes & random_hashes:
        raise RuntimeError("A random frozen backbone matches the pretrained audit hash.")

    plan: Dict[str, Any] = {
        "schema_version": 1,
        "status": "locked_before_test",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_root": str(output_root),
        "primary_contrast": {
            "treatment": PRIMARY_TREATMENT,
            "control": PRIMARY_CONTROL,
            "endpoint": "test_mse",
            "secondary_gate": "test_mae_noninferiority",
        },
        "expected_seeds": expected_seeds,
        "minimum_relative_improvement": args.minimum_relative_improvement,
        "mae_noninferiority_margin": args.mae_noninferiority_margin,
        "minimum_win_fraction": args.minimum_win_fraction,
        "confidence_level": 0.95,
        "independent_unit": "matched training seed",
        "comparison_protocol_hash": next(iter(comparison_hashes)),
        "environment_hash": next(iter(environment_hashes)),
        "expected_runs": expected_runs,
        "pair_audit": pair_audit,
        "test_metrics_present_at_lock": False,
        "analysis_source_sha256": {
            str(path.resolve()): sha256_file(path)
            for path in [
                Path(__file__).resolve(),
                Path(__file__).resolve().parent / "analysis_plan.py",
                Path(__file__).resolve().parent / "summarize_results.py",
            ]
        },
        "notes": [
            "Sign-flip inference assumes exchangeability/symmetry of paired log-error differences.",
            "Student-t log-ratio CI assumes approximately iid, normal seed-level log ratios.",
            "Exploratory MSE and MAE contrasts form separate Holm families.",
        ],
    }
    plan["plan_content_sha256"] = plan_content_hash(plan)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    print(f"locked: {lock_path}")
    print(f"plan sha256: {plan['plan_content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
