"""Run one auditable SMETimes effectiveness experiment.

Stages are deliberately separated:

* ``validate`` checks files/protocol without loading the 1B weights.
* ``train`` uses train/validation only and never opens the test split.
* ``test`` reconstructs the frozen backbone and evaluates a saved checkpoint once.
* ``both`` is convenient for non-confirmatory smoke tests only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

EXPERIMENT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = Path(__file__).resolve().parents[2]
UPSTREAM_DIR = WORKSPACE_DIR
if str(UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_DIR))

import numpy as np
import pandas as pd
import scipy
import torch
import transformers

from effectiveness_model import VARIANT_SPECS
from analysis_plan import verify_test_authorization
from experiment import (
    EffectivenessExperiment,
    model_audit,
    set_global_seed,
    sha256_file,
)


DEFAULT_DATA_ROOT = UPSTREAM_DIR / "data" / "ETT-small"
DEFAULT_LLM_PATH = UPSTREAM_DIR / "models" / "llm" / "llama-3.2-1b-instruct"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["validate", "train", "test", "both"], default="validate"
    )
    parser.add_argument("--variant", choices=list(VARIANT_SPECS), default="pretrained_no_context")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--backbone-seed", type=int)
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_DIR / "outputs" / "manual")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--data-path", default="ETTh1.csv")
    parser.add_argument("--llm-path", type=Path, default=DEFAULT_LLM_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu", type=int, default=0)

    parser.add_argument("--seq-len", type=int, default=672)
    parser.add_argument("--label-len", type=int, default=576)
    parser.add_argument("--token-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tmax", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--debug-max-batches",
        type=int,
        default=0,
        help="Truncate each split for a non-evidential smoke test; 0 uses the full split.",
    )
    parser.add_argument("--mlp-hidden-dim", type=int, default=256)
    parser.add_argument("--mlp-hidden-layers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-experts", type=int, default=1)
    parser.add_argument("--lambda-reg", type=float, default=0.01)
    parser.add_argument("--amp", dest="use_amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--deterministic", dest="deterministic", action="store_true", default=True)
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument(
        "--allow-repeat-test",
        action="store_true",
        help="Explicitly permit replacing an existing test metrics file.",
    )
    return parser


def validate_protocol(args: argparse.Namespace) -> Dict[str, Any]:
    data_root = args.data_root.resolve()
    csv_path = data_root / args.data_path
    embedding_path = data_root / f"{Path(args.data_path).stem}.pt"
    llm_path = args.llm_path.resolve()
    llm_config_path = llm_path / "config.json"

    for path in [csv_path, embedding_path, llm_config_path]:
        if not path.exists():
            raise FileNotFoundError(f"Required file does not exist: {path}")
    if args.seq_len - args.label_len != args.token_len:
        raise ValueError("seq_len - label_len must equal token_len for upstream training.")
    if args.seq_len % args.token_len != 0:
        raise ValueError("seq_len must be divisible by token_len.")
    if args.pred_len != args.token_len:
        raise ValueError(
            "The evidence phase is locked to pred_len == token_len (96) to avoid "
            "the upstream long-horizon future-statistics leakage path."
        )
    if args.num_experts < 1:
        raise ValueError("num_experts must be at least 1.")
    if args.debug_max_batches < 0:
        raise ValueError("debug_max_batches cannot be negative.")

    with llm_config_path.open("r", encoding="utf-8") as handle:
        llm_config = json.load(handle)
    hidden_size = int(llm_config["hidden_size"])
    try:
        embeddings = torch.load(embedding_path, map_location="cpu", weights_only=True)
    except TypeError:
        embeddings = torch.load(embedding_path, map_location="cpu")
    if embeddings.ndim != 2:
        raise ValueError(f"Expected rank-2 embeddings, got shape {tuple(embeddings.shape)}")
    if int(embeddings.shape[1]) != hidden_size:
        raise ValueError(
            f"Embedding dim {embeddings.shape[1]} does not match LLM hidden size {hidden_size}."
        )
    finite = bool(torch.isfinite(embeddings).all())
    if not finite:
        raise ValueError("Timestamp embeddings contain NaN or Inf.")

    frame = pd.read_csv(csv_path)
    if len(frame) != int(embeddings.shape[0]):
        raise ValueError(
            f"CSV rows {len(frame)} do not match embedding rows {embeddings.shape[0]}."
        )
    feature_count = len(frame.columns) - 1
    test_span = 4 * 30 * 24 + args.seq_len
    per_feature_test_samples = test_span - args.seq_len - args.pred_len + 1
    expected_test_samples = per_feature_test_samples * feature_count
    weight_files = sorted(llm_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No local safetensors weights found in {llm_path}")
    weight_index_files = sorted(llm_path.glob("*.safetensors.index.json"))

    report = {
        "status": "valid",
        "dataset": "ETTh1",
        "csv_path": str(csv_path),
        "csv_rows": len(frame),
        "feature_count": feature_count,
        "data_sha256": sha256_file(csv_path),
        "embedding_path": str(embedding_path),
        "embedding_sha256": sha256_file(embedding_path),
        "embedding_shape": list(embeddings.shape),
        "embedding_dtype": str(embeddings.dtype),
        "embedding_finite": finite,
        "llm_path": str(llm_path),
        "llm_hidden_size": hidden_size,
        "llm_config_sha256": sha256_file(llm_config_path),
        "llm_weight_sha256": {
            path.name: sha256_file(path) for path in weight_files
        },
        "llm_weight_index_sha256": {
            path.name: sha256_file(path) for path in weight_index_files
        },
        "expected_test_samples": expected_test_samples,
        "locked_horizon": f"{args.seq_len}->{args.pred_len}",
    }
    return report


def make_upstream_args(args: argparse.Namespace, run_dir: Path) -> SimpleNamespace:
    device_text = str(args.device)
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device_text}, but CUDA is unavailable.")
    backbone_seed = args.backbone_seed
    if backbone_seed is None:
        backbone_seed = args.seed + 100_000
    return SimpleNamespace(
        task_name="long_term_forecast",
        is_training=1,
        model_id="ETTh1_672_96_effectiveness",
        model="SMETimes_Llama",
        data="ETTh1",
        root_path=str(args.data_root.resolve()),
        data_path=args.data_path,
        test_data_path=args.data_path,
        checkpoints=str((run_dir / "checkpoint").resolve()),
        drop_last=False,
        val_set_shuffle=False,
        drop_short=False,
        seq_len=args.seq_len,
        label_len=args.label_len,
        token_len=args.token_len,
        test_seq_len=args.seq_len,
        test_label_len=args.label_len,
        test_pred_len=args.pred_len,
        seasonal_patterns="Monthly",
        llm_ckp_dir=str(args.llm_path.resolve()),
        local_files_only=True,
        mlp_hidden_dim=args.mlp_hidden_dim,
        mlp_hidden_layers=args.mlp_hidden_layers,
        mlp_activation="tanh",
        dropout=args.dropout,
        num_experts=args.num_experts,
        lambda_reg=args.lambda_reg,
        num_workers=args.num_workers,
        train_epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        lradj="type1",
        cosine=True,
        tmax=args.tmax,
        use_amp=args.use_amp,
        use_multi_gpu=False,
        local_rank=0,
        gpu=args.gpu,
        device=device_text,
        mix_embeds=VARIANT_SPECS[args.variant].use_context,
        variant=args.variant,
        seed=args.seed,
        backbone_seed=backbone_seed,
        deterministic=args.deterministic,
        debug_max_batches=args.debug_max_batches,
        visualize=False,
    )


def protocol_payload(args: argparse.Namespace, validation: Dict[str, Any]) -> Dict[str, Any]:
    source_files = [
        Path(__file__).resolve(),
        EXPERIMENT_DIR / "effectiveness_model.py",
        EXPERIMENT_DIR / "experiment.py",
        UPSTREAM_DIR / "models" / "SMETimes_Llama.py",
        UPSTREAM_DIR / "data_provider" / "data_loader.py",
        UPSTREAM_DIR / "data_provider" / "data_factory.py",
        UPSTREAM_DIR / "exp" / "exp_basic.py",
        UPSTREAM_DIR / "exp" / "exp_long_term_forecasting.py",
        UPSTREAM_DIR / "layers" / "mlp.py",
        UPSTREAM_DIR / "utils" / "device.py",
        UPSTREAM_DIR / "utils" / "llm.py",
        UPSTREAM_DIR / "utils" / "tools.py",
    ]
    source_sha256 = {str(path): sha256_file(path) for path in source_files}
    return {
        "schema_version": 1,
        "variant": args.variant,
        "seed": args.seed,
        "backbone_seed": args.backbone_seed if args.backbone_seed is not None else args.seed + 100_000,
        "dataset": "ETTh1",
        "data_root": str(args.data_root.resolve()),
        "data_path": args.data_path,
        "data_sha256": validation["data_sha256"],
        "embedding_sha256": validation["embedding_sha256"],
        "llm_path": str(args.llm_path.resolve()),
        "llm_hidden_size": validation["llm_hidden_size"],
        "llm_config_sha256": validation["llm_config_sha256"],
        "llm_weight_sha256": validation["llm_weight_sha256"],
        "llm_weight_index_sha256": validation["llm_weight_index_sha256"],
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "token_len": args.token_len,
        "pred_len": args.pred_len,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "tmax": args.tmax,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "mlp_hidden_layers": args.mlp_hidden_layers,
        "dropout": args.dropout,
        "num_experts": args.num_experts,
        "lambda_reg": args.lambda_reg,
        "use_amp": args.use_amp,
        "deterministic": args.deterministic,
        "debug_max_batches": args.debug_max_batches,
        "num_workers": args.num_workers,
        "device_request": args.device,
        "gpu": args.gpu,
        "source_sha256": source_sha256,
    }


def payload_hash(payload: Dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def environment_payload(gpu: int) -> Dict[str, Any]:
    cuda_name = None
    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(gpu)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "cuda_device_index": gpu if torch.cuda.is_available() else None,
        "cuda_device": cuda_name,
        "command": sys.argv,
    }


def comparable_environment(payload: Dict[str, Any]) -> Dict[str, Any]:
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
    return {field: payload.get(field) for field in fields}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    args = build_parser().parse_args()
    args.data_root = args.data_root.resolve()
    args.llm_path = args.llm_path.resolve()
    args.output_root = args.output_root.resolve()
    validation = validate_protocol(args)

    if args.stage == "validate":
        args.output_root.mkdir(parents=True, exist_ok=True)
        report_path = args.output_root / "validation_report.json"
        write_json(report_path, validation)
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        print(f"saved: {report_path}")
        return 0

    run_dir = args.output_root / "runs" / args.variant / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.json"
    protocol = protocol_payload(args, validation)
    protocol_sha256 = payload_hash(protocol)
    config = {"protocol_hash": protocol_sha256, "protocol": protocol}

    if args.stage in {"train", "both"}:
        checkpoint_path = run_dir / "checkpoint" / "trainable_state.pth"
        if (run_dir / "metrics.json").exists():
            raise RuntimeError(
                "This run directory already contains test evidence. Retraining here would "
                "make it stale; choose a new --output-root instead."
            )
        if checkpoint_path.exists() and not args.force_train:
            raise FileExistsError(
                f"Checkpoint already exists: {checkpoint_path}. "
                "Use --force-train only if replacement is intentional."
            )
        if config_path.exists() and read_json(config_path).get("protocol_hash") != protocol_sha256:
            if not args.force_train:
                raise RuntimeError("Existing run directory has a different protocol hash.")
        write_json(config_path, config)
        write_json(run_dir / "environment.json", environment_payload(args.gpu))
    else:
        if not config_path.exists():
            raise FileNotFoundError(f"Missing training config: {config_path}")
        saved_config = read_json(config_path)
        if saved_config.get("protocol_hash") != protocol_sha256:
            raise RuntimeError(
                "Test command does not match the locked training protocol. "
                "Use the exact same hyperparameters and source version."
            )
        environment_path = run_dir / "environment.json"
        if not environment_path.exists():
            raise FileNotFoundError(f"Missing training environment: {environment_path}")
        if comparable_environment(read_json(environment_path)) != comparable_environment(
            environment_payload(args.gpu)
        ):
            raise RuntimeError("Test environment differs from the locked training environment.")

    if args.stage in {"test", "both"}:
        metrics_path = run_dir / "metrics.json"
        if args.stage == "both" and not args.debug_max_batches:
            raise RuntimeError(
                "A full confirmatory run cannot train and test in one command. "
                "Train, lock_analysis_plan.py, then run --stage test."
            )
        if metrics_path.exists() and not args.debug_max_batches:
            raise RuntimeError(
                "Full test evidence already exists and cannot be overwritten. "
                "Use a fresh output root for a new protocol."
            )
        if metrics_path.exists() and not args.allow_repeat_test:
            print(f"test already evaluated; leaving untouched: {metrics_path}")
            return 0
        if not args.debug_max_batches:
            verify_test_authorization(args.output_root, protocol_sha256, protocol)

    test_claim_path = None
    if args.stage in {"test", "both"} and not args.debug_max_batches:
        test_claim_path = run_dir / "TEST_CLAIM.lock"
        try:
            with test_claim_path.open("x", encoding="utf-8") as handle:
                json.dump(
                    {
                        "protocol_hash": protocol_sha256,
                        "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
                        "pid": os.getpid(),
                    },
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
        except FileExistsError as error:
            raise RuntimeError(
                f"Test has already been claimed or a prior test crashed: {test_claim_path}. "
                "Failing closed; audit the run and use a fresh output root if needed."
            ) from error

    set_global_seed(args.seed, args.deterministic)
    upstream_args = make_upstream_args(args, run_dir)
    experiment = EffectivenessExperiment(upstream_args)
    audit = model_audit(experiment.model)

    audit_path = run_dir / "model_audit.json"
    if args.stage == "test":
        if not audit_path.exists():
            raise FileNotFoundError(f"Missing locked training model audit: {audit_path}")
        trained_audit = read_json(audit_path)
        for key in ["trainable_signature", "trainable_initial_sha256", "backbone_sample_sha256"]:
            if trained_audit.get(key) != audit.get(key):
                raise RuntimeError(f"Reconstructed model mismatch for audit field: {key}")
    else:
        write_json(audit_path, audit)

    print(
        f"variant={args.variant} seed={args.seed} "
        f"trainable={audit['trainable_params']:,} total={audit['total_params']:,}"
    )
    if args.stage in {"train", "both"}:
        train_summary = experiment.train_validation_only(run_dir)
        print(json.dumps(train_summary, ensure_ascii=False, indent=2))
    if args.stage in {"test", "both"}:
        metrics = experiment.evaluate_test_once(run_dir)
        if test_claim_path is not None:
            completion = {
                "status": "test_complete",
                "protocol_hash": protocol_sha256,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "metrics_sha256": sha256_file(run_dir / "metrics.json"),
                "predictions_sha256": metrics["predictions_sha256"],
                "targets_sha256": metrics["targets_sha256"],
                "errors_by_sample_sha256": metrics["errors_by_sample_sha256"],
            }
            write_json(run_dir / "TEST_COMPLETE.json", completion)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
