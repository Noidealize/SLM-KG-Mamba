"""Aggregate matched-seed validation/test evidence without pseudoreplication."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
from scipy import stats

from analysis_plan import (
    LOCK_FILENAME,
    comparison_protocol_hash,
    read_locked_plan,
    sha256_json,
)


CONTRASTS = [
    {
        "contrast": "pretraining_primary",
        "treatment": "pretrained_no_context",
        "control": "random_no_context",
        "primary": True,
        "interpretation": "net contribution of pretrained weights at matched architecture",
    },
    {
        "contrast": "context_branch_exploratory",
        "treatment": "pretrained_full",
        "control": "pretrained_no_context",
        "primary": False,
        "interpretation": "aligned timestamp/OT-statistic embedding branch",
    },
    {
        "contrast": "context_alignment_exploratory",
        "treatment": "pretrained_full",
        "control": "pretrained_shuffled_context",
        "primary": False,
        "interpretation": "sample-context alignment, not language semantics alone",
    },
    {
        "contrast": "backbone_necessity_exploratory",
        "treatment": "pretrained_no_context",
        "control": "identity_no_context",
        "primary": False,
        "interpretation": "frozen pretrained transformation versus identity path",
    },
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=None,
        help="Only checks equality with a pre-test lock; cannot redefine it after test.",
    )
    return parser


def read_runs(output_root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    runs_root = output_root / "runs"
    if not runs_root.exists():
        return pd.DataFrame()
    for config_path in sorted(runs_root.glob("*/seed_*/config.json")):
        run_dir = config_path.parent
        config = read_json(config_path)
        protocol = config["protocol"]
        if config.get("protocol_hash") != sha256_json(protocol):
            raise RuntimeError(f"Protocol hash mismatch: {config_path}")
        row: Dict[str, Any] = {
            "run_dir": str(run_dir.resolve()),
            "protocol_hash": config["protocol_hash"],
            **{key: value for key, value in protocol.items() if key != "source_sha256"},
            "comparison_protocol_hash": comparison_protocol_hash(protocol),
        }
        for source_path, expected_sha in protocol.get("source_sha256", {}).items():
            path = Path(source_path)
            if not path.exists() or cached_sha256_file(path) != expected_sha:
                raise RuntimeError(f"Source file drift relative to run config: {path}")
        audit_path = run_dir / "model_audit.json"
        train_path = run_dir / "train_summary.json"
        metrics_path = run_dir / "metrics.json"
        environment_path = run_dir / "environment.json"
        if environment_path.exists():
            environment = read_json(environment_path)
            environment_fields = [
                "platform",
                "torch",
                "numpy",
                "pandas",
                "transformers",
                "cuda_available",
                "torch_cuda",
                "cuda_device_0",
            ]
            row["environment_hash"] = sha256_json(
                {field: environment.get(field) for field in environment_fields}
            )
        if audit_path.exists():
            audit = read_json(audit_path)
            for key in [
                "total_params",
                "trainable_params",
                "frozen_params",
                "trainable_initial_sha256",
                "backbone_sample_sha256",
            ]:
                row[key] = audit.get(key)
        if train_path.exists():
            train = read_json(train_path)
            if train.get("variant") != protocol["variant"] or int(train.get("seed")) != int(protocol["seed"]):
                raise RuntimeError(f"Train summary identity mismatch: {train_path}")
            for key in [
                "best_val_mse",
                "best_epoch",
                "epochs_completed",
                "train_seconds",
                "train_peak_allocated_mib",
                "checkpoint_sha256",
                "diagnostic_truncated",
            ]:
                row[f"train_{key}" if key == "diagnostic_truncated" else key] = train.get(key)
            checkpoint_path = run_dir / "checkpoint" / "trainable_state.pth"
            if not checkpoint_path.exists() or cached_sha256_file(checkpoint_path) != train.get("checkpoint_sha256"):
                raise RuntimeError(f"Checkpoint missing or hash mismatch: {checkpoint_path}")
        if metrics_path.exists():
            metrics = read_json(metrics_path)
            if metrics.get("variant") != protocol["variant"] or int(metrics.get("seed")) != int(protocol["seed"]):
                raise RuntimeError(f"Metrics identity mismatch: {metrics_path}")
            for key in [
                "mse",
                "mae",
                "rmse",
                "test_samples",
                "test_seconds",
                "test_samples_per_second",
                "test_peak_allocated_mib",
                "fusion_alpha",
                "diagnostic_truncated",
            ]:
                row[f"test_{key}" if key == "diagnostic_truncated" else key] = metrics.get(key)
            for array_name, expected_shape in [
                ("predictions.npy", metrics.get("prediction_shape")),
                ("targets.npy", metrics.get("prediction_shape")),
            ]:
                array_path = run_dir / array_name
                if not array_path.exists() or list(np.load(array_path, mmap_mode="r").shape) != expected_shape:
                    raise RuntimeError(f"Prediction artifact shape mismatch: {array_path}")
            predictions_path = run_dir / "predictions.npy"
            targets_path = run_dir / "targets.npy"
            errors_path = run_dir / "errors_by_sample.npz"
            for path, key in [
                (predictions_path, "predictions_sha256"),
                (targets_path, "targets_sha256"),
                (errors_path, "errors_by_sample_sha256"),
            ]:
                if not path.exists() or cached_sha256_file(path) != metrics.get(key):
                    raise RuntimeError(f"Test artifact hash mismatch: {path}")
            predictions = np.load(predictions_path)
            targets = np.load(targets_path)
            recomputed_mse = float(
                np.mean(np.square(predictions - targets), dtype=np.float64)
            )
            recomputed_mae = float(
                np.mean(np.abs(predictions - targets), dtype=np.float64)
            )
            recomputed_rmse = float(math.sqrt(recomputed_mse))
            for key, recomputed in [
                ("mse", recomputed_mse),
                ("mae", recomputed_mae),
                ("rmse", recomputed_rmse),
            ]:
                if not math.isclose(
                    float(metrics.get(key)), recomputed, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise RuntimeError(
                        f"metrics.json {key} does not match saved arrays: {metrics_path}"
                    )
            if not metrics.get("diagnostic_truncated") and int(metrics.get("test_samples", -1)) != int(
                metrics.get("expected_full_test_samples", -2)
            ):
                raise RuntimeError(f"Full test sample count mismatch: {metrics_path}")
            if not metrics.get("diagnostic_truncated"):
                complete_path = run_dir / "TEST_COMPLETE.json"
                claim_path = run_dir / "TEST_CLAIM.lock"
                if not complete_path.exists() or not claim_path.exists():
                    raise RuntimeError(f"Missing atomic full-test completion record: {run_dir}")
                complete = read_json(complete_path)
                expected_complete = {
                    "protocol_hash": config["protocol_hash"],
                    "metrics_sha256": cached_sha256_file(metrics_path),
                    "predictions_sha256": metrics["predictions_sha256"],
                    "targets_sha256": metrics["targets_sha256"],
                    "errors_by_sample_sha256": metrics["errors_by_sample_sha256"],
                }
                if complete.get("status") != "test_complete" or any(
                    complete.get(key) != value for key, value in expected_complete.items()
                ):
                    raise RuntimeError(f"Full-test completion record mismatch: {complete_path}")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not frame.empty and frame.duplicated(["variant", "seed"]).any():
        duplicates = frame.loc[frame.duplicated(["variant", "seed"], keep=False), ["variant", "seed"]]
        raise ValueError(f"Duplicate variant/seed runs found:\n{duplicates}")
    return frame


def mean_summary(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    if frame.empty or metric not in frame:
        return pd.DataFrame()
    available = frame.dropna(subset=[metric])
    for variant, group in available.groupby("variant", sort=True):
        values = group[metric].astype(float).to_numpy()
        mean = float(values.mean())
        sd = float(values.std(ddof=1)) if len(values) > 1 else math.nan
        low, high = t_interval(values, confidence=0.95)
        rows.append(
            {
                "variant": variant,
                "metric": metric,
                "n_seeds": len(values),
                "mean": mean,
                "sd": sd,
                "ci95_low": low,
                "ci95_high": high,
            }
        )
    return pd.DataFrame(rows)


def audit_runs_against_locked_plan(
    frame: pd.DataFrame, plan: Dict[str, Any], output_root: Path
) -> None:
    if Path(plan.get("output_root", "")).resolve() != output_root.resolve():
        raise RuntimeError("Locked analysis plan belongs to a different output root.")
    for run_key, expected in plan.get("expected_runs", {}).items():
        variant, seed_text = run_key.split(":seed_", maxsplit=1)
        seed = int(seed_text)
        matched = frame[(frame["variant"] == variant) & (frame["seed"].astype(int) == seed)]
        if len(matched) != 1:
            raise RuntimeError(f"Locked run missing or duplicated in summary: {run_key}")
        row = matched.iloc[0]
        for field in ["protocol_hash", "checkpoint_sha256", "environment_hash"]:
            if row.get(field) != expected.get(field):
                raise RuntimeError(f"Locked run drift on {field}: {run_key}")
        locked_audit = plan["pair_audit"][str(seed)][variant]
        for field, expected_value in locked_audit.items():
            if row.get(field) != expected_value:
                raise RuntimeError(f"Locked model audit drift on {field}: {run_key}")

    expected_seeds = [int(seed) for seed in plan["expected_seeds"]]
    pretrained_hashes = set()
    random_hashes = set()
    for seed in expected_seeds:
        pretrained_hashes.add(
            plan["pair_audit"][str(seed)]["pretrained_no_context"][
                "backbone_sample_sha256"
            ]
        )
        random_hashes.add(
            plan["pair_audit"][str(seed)]["random_no_context"][
                "backbone_sample_sha256"
            ]
        )
    if len(pretrained_hashes) != 1:
        raise RuntimeError("Pretrained backbone changed across locked seeds.")
    if len(random_hashes) != len(expected_seeds):
        raise RuntimeError("Random backbone hashes are not unique across locked seeds.")
    if pretrained_hashes & random_hashes:
        raise RuntimeError("A random backbone equals the locked pretrained backbone.")


def paired_comparisons(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    if frame.empty or metric not in frame:
        return pd.DataFrame()
    for contrast in CONTRASTS:
        audit_columns = [
            "seed",
            metric,
            "comparison_protocol_hash",
            "trainable_initial_sha256",
            "total_params",
            "trainable_params",
            "environment_hash",
        ]
        missing_columns = [column for column in audit_columns if column not in frame]
        if missing_columns:
            raise RuntimeError(f"Cannot form auditable pairs; missing columns: {missing_columns}")
        treatment = frame.loc[
            frame["variant"] == contrast["treatment"], audit_columns
        ].dropna()
        control = frame.loc[
            frame["variant"] == contrast["control"], audit_columns
        ].dropna()
        paired = treatment.merge(control, on="seed", suffixes=("_treatment", "_control"))
        if paired.empty:
            continue
        if not np.all(
            paired["comparison_protocol_hash_treatment"]
            == paired["comparison_protocol_hash_control"]
        ):
            raise RuntimeError(f"Protocol mismatch inside contrast {contrast['contrast']}")
        if not np.all(paired["environment_hash_treatment"] == paired["environment_hash_control"]):
            raise RuntimeError(f"Environment mismatch inside contrast {contrast['contrast']}")
        if contrast["primary"]:
            for field in ["trainable_initial_sha256", "total_params", "trainable_params"]:
                if not np.all(paired[f"{field}_treatment"] == paired[f"{field}_control"]):
                    raise RuntimeError(
                        f"Primary matched control is unequal on {field}; refusing inference."
                    )
        treatment_values = paired[f"{metric}_treatment"].astype(float).to_numpy()
        control_values = paired[f"{metric}_control"].astype(float).to_numpy()
        if np.any(treatment_values <= 0) or np.any(control_values <= 0):
            raise ValueError(f"{metric} values must be positive for log-ratio analysis.")

        # Positive benefit means the treatment has lower error.
        benefit = control_values - treatment_values
        log_ratio = np.log(treatment_values / control_values)
        benefit_low, benefit_high = t_interval(benefit, confidence=0.95)
        log_low, log_high = t_interval(log_ratio, confidence=0.95)
        log90_low, log90_high = t_interval(log_ratio, confidence=0.90)
        bootstrap_low, bootstrap_high = bootstrap_log_ratio_interval(log_ratio)
        geometric_ratio = float(np.exp(log_ratio.mean()))
        row = {
            **contrast,
            "metric": metric,
            "n_pairs": len(paired),
            "matched_seeds": ",".join(str(int(seed)) for seed in paired["seed"]),
            "wins": int(np.sum(treatment_values < control_values)),
            "ties": int(np.sum(treatment_values == control_values)),
            "mean_absolute_benefit": float(benefit.mean()),
            "benefit_ci95_low": benefit_low,
            "benefit_ci95_high": benefit_high,
            "geometric_mean_ratio": geometric_ratio,
            "relative_improvement_pct": float((1.0 - geometric_ratio) * 100.0),
            "ratio_ci95_low": float(np.exp(log_low)) if math.isfinite(log_low) else math.nan,
            "ratio_ci95_high": float(np.exp(log_high)) if math.isfinite(log_high) else math.nan,
            "ratio_ci90_low": float(np.exp(log90_low)) if math.isfinite(log90_low) else math.nan,
            "ratio_ci90_high": float(np.exp(log90_high)) if math.isfinite(log90_high) else math.nan,
            "ratio_bootstrap_ci95_low": float(np.exp(bootstrap_low)),
            "ratio_bootstrap_ci95_high": float(np.exp(bootstrap_high)),
            "paired_sign_flip_p_two_sided": exact_sign_flip_p(-log_ratio),
            "sign_flip_estimand": "paired log(control/treatment)",
            "holm_family": f"{metric}_exploratory_contrasts",
        }
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_adjusted_p_exploratory"] = math.nan
        exploratory_mask = ~result["primary"].astype(bool)
        result.loc[exploratory_mask, "holm_adjusted_p_exploratory"] = holm_adjust(
            result.loc[exploratory_mask, "paired_sign_flip_p_two_sided"].to_numpy(dtype=float)
        )
    return result


def t_interval(values: np.ndarray, confidence: float) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return math.nan, math.nan
    mean = float(values.mean())
    sem = float(stats.sem(values))
    critical = float(stats.t.ppf((1.0 + confidence) / 2.0, df=len(values) - 1))
    return mean - critical * sem, mean + critical * sem


def exact_sign_flip_p(values: Iterable[float]) -> float:
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return math.nan
    observed = abs(float(values.mean()))
    n = len(values)
    if n <= 20:
        extreme = 0
        total = 2**n
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            statistic = abs(float(np.mean(values * np.asarray(signs))))
            if statistic >= observed - 1e-15:
                extreme += 1
        return extreme / total
    rng = np.random.default_rng(2025)
    sign_matrix = rng.choice((-1.0, 1.0), size=(100_000, n))
    statistics = np.abs(np.mean(sign_matrix * values, axis=1))
    return float((np.sum(statistics >= observed) + 1) / (len(statistics) + 1))


def bootstrap_log_ratio_interval(
    log_ratio: np.ndarray, confidence: float = 0.95, resamples: int = 20_000
) -> tuple[float, float]:
    values = np.asarray(log_ratio, dtype=float)
    if len(values) < 2:
        return math.nan, math.nan
    rng = np.random.default_rng(2025)
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return tuple(float(value) for value in np.quantile(means, [alpha, 1.0 - alpha]))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    if len(p_values) == 0:
        return p_values
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        adjusted = (len(p_values) - rank) * p_values[index]
        running = max(running, adjusted)
        adjusted_sorted[rank] = min(1.0, running)
    adjusted = np.empty(len(p_values), dtype=float)
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
    return adjusted


def decision_payload(
    test_mse: pd.DataFrame,
    test_mae: pd.DataFrame,
    analysis_plan: Dict[str, Any] | None,
    unlocked_threshold: float = 0.01,
) -> Dict[str, Any]:
    locked = analysis_plan is not None
    minimum_relative_improvement = (
        float(analysis_plan["minimum_relative_improvement"])
        if locked
        else unlocked_threshold
    )
    payload: Dict[str, Any] = {
        "primary_contrast": "pretraining_primary",
        "independent_unit": "matched training seed",
        "minimum_relative_improvement": minimum_relative_improvement,
        "threshold_status": "locked_before_test" if locked else "unlocked_descriptive_only",
        "decision": "insufficient_evidence",
        "reason": "No complete n>=10 paired confirmatory test evidence.",
    }
    if locked:
        payload["analysis_plan_sha256"] = analysis_plan["plan_content_sha256"]
    if test_mse.empty:
        return payload
    primary = test_mse[test_mse["contrast"] == "pretraining_primary"]
    if primary.empty:
        return payload
    mse_row = primary.iloc[0]
    payload["mse_primary"] = _jsonable_row(mse_row)
    if not locked:
        payload["reason"] = (
            "Test results are descriptive only because no pre-test analysis plan was locked."
        )
        return payload
    expected_seeds = {int(seed) for seed in analysis_plan["expected_seeds"]}
    mse_seeds = {int(seed) for seed in str(mse_row["matched_seeds"]).split(",") if seed}
    if mse_seeds != expected_seeds:
        payload["reason"] = "MSE seed set does not exactly match the locked analysis plan."
        return payload

    mae_primary = test_mae[test_mae["contrast"] == "pretraining_primary"]
    mae_ok = False
    if not mae_primary.empty:
        mae_row = mae_primary.iloc[0]
        payload["mae_primary"] = _jsonable_row(mae_row)
        mae_seeds = {int(seed) for seed in str(mae_row["matched_seeds"]).split(",") if seed}
        mae_margin = float(analysis_plan["mae_noninferiority_margin"])
        mae_ok = (
            mae_seeds == expected_seeds
            and int(mae_row["n_pairs"]) == len(expected_seeds)
            and float(mae_row["ratio_ci95_high"]) <= 1.0 + mae_margin
        )

    mse_ok = float(mse_row["ratio_ci95_high"]) < 1.0 - minimum_relative_improvement
    required_wins = math.ceil(float(analysis_plan["minimum_win_fraction"]) * len(expected_seeds))
    wins_ok = int(mse_row["wins"]) >= required_wins
    if mse_ok and wins_ok and mae_ok:
        payload["decision"] = "supports_pretrained_advantage_under_locked_protocol"
        payload["reason"] = (
            "MSE ratio CI clears the locked practical threshold, the locked win rule is met, "
            "and MAE meets the locked noninferiority gate on the same seeds."
        )
    else:
        payload["decision"] = "does_not_meet_prespecified_support_rule"
        payload["reason"] = (
            f"criteria: mse_ci={mse_ok}, wins={wins_ok}, mae_noninferiority={mae_ok}. "
            "This may mean harm, equivalence, or remaining uncertainty; inspect the CIs."
        )
    return payload


def _jsonable_row(row: pd.Series) -> Dict[str, Any]:
    payload = {}
    for key, value in row.items():
        if isinstance(value, (np.integer,)):
            payload[key] = int(value)
        elif isinstance(value, (np.floating,)):
            payload[key] = None if not np.isfinite(value) else float(value)
        else:
            payload[key] = value
    return payload


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


_FILE_HASH_CACHE: Dict[tuple[str, int, int], str] = {}


def cached_sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _FILE_HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        _FILE_HASH_CACHE[key] = digest.hexdigest()
    return _FILE_HASH_CACHE[key]


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> int:
    args = build_parser().parse_args()
    output_root = args.output_root.resolve()
    summary_dir = output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs = read_runs(output_root)
    runs.to_csv(summary_dir / "runs.csv", index=False)

    diagnostic = pd.Series(False, index=runs.index)
    for column in ["train_diagnostic_truncated", "test_diagnostic_truncated"]:
        if column in runs:
            diagnostic = diagnostic | runs[column].fillna(False).astype(bool)
    evidence_runs = runs[~diagnostic].copy()

    lock_path = output_root / LOCK_FILENAME
    analysis_plan = read_locked_plan(output_root) if lock_path.exists() else None
    if analysis_plan is not None:
        audit_runs_against_locked_plan(runs, analysis_plan, output_root)
    if analysis_plan is not None and args.minimum_relative_improvement is not None:
        if not math.isclose(
            args.minimum_relative_improvement,
            float(analysis_plan["minimum_relative_improvement"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(
                "CLI threshold differs from the pre-test locked analysis plan; refusing redefinition."
            )

    validation_summary = mean_summary(evidence_runs, "best_val_mse")
    validation_pairs = paired_comparisons(evidence_runs, "best_val_mse")
    test_mse_summary = mean_summary(evidence_runs, "mse")
    test_mae_summary = mean_summary(evidence_runs, "mae")
    test_mse_pairs = paired_comparisons(evidence_runs, "mse")
    test_mae_pairs = paired_comparisons(evidence_runs, "mae")

    validation_summary.to_csv(summary_dir / "validation_variant_summary.csv", index=False)
    validation_pairs.to_csv(summary_dir / "validation_paired_comparisons.csv", index=False)
    pd.concat([test_mse_summary, test_mae_summary], ignore_index=True).to_csv(
        summary_dir / "test_variant_summary.csv", index=False
    )
    pd.concat([test_mse_pairs, test_mae_pairs], ignore_index=True).to_csv(
        summary_dir / "test_paired_comparisons.csv", index=False
    )
    decision = decision_payload(
        test_mse_pairs,
        test_mae_pairs,
        analysis_plan,
        unlocked_threshold=args.minimum_relative_improvement or 0.01,
    )
    write_json(summary_dir / "decision.json", decision)
    print(f"runs found: {len(runs)}")
    print(f"summary: {summary_dir}")
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
