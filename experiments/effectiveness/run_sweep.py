"""Sequential matrix runner for matched-seed effectiveness experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from effectiveness_model import VARIANT_SPECS


HERE = Path(__file__).resolve().parent

PRESETS = {
    "smoke": {
        "seeds": [2025],
        "variants": list(VARIANT_SPECS),
        "epochs": 1,
        "debug_max_batches": 2,
    },
    "pilot": {
        "seeds": [2025, 2026, 2027],
        "variants": list(VARIANT_SPECS),
        "epochs": 10,
        "debug_max_batches": 0,
    },
    "confirmatory": {
        "seeds": list(range(2025, 2035)),
        "variants": ["pretrained_no_context", "random_no_context"],
        "epochs": 10,
        "debug_max_batches": 0,
    },
}


_HASH_CACHE = {}


def file_sha256(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        _HASH_CACHE[key] = digest.hexdigest()
    return _HASH_CACHE[key]


def run_is_complete(run_dir: Path, stage: str, expected: dict | None = None) -> bool:
    try:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        protocol = config["protocol"]
        if expected and any(protocol.get(key) != value for key, value in expected.items()):
            return False
        for source_path, expected_sha in protocol.get("source_sha256", {}).items():
            path = Path(source_path)
            if not path.exists() or file_sha256(path) != expected_sha:
                return False

        train = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
        checkpoint = run_dir / "checkpoint" / "trainable_state.pth"
        train_complete = (
            train.get("status") == "trained_validation_only"
            and train.get("variant") == protocol.get("variant")
            and int(train.get("seed")) == int(protocol.get("seed"))
            and checkpoint.exists()
            and file_sha256(checkpoint) == train.get("checkpoint_sha256")
        )
        if stage == "train":
            return train_complete

        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        test_complete = (
            metrics.get("status") == "test_evaluated_once"
            and metrics.get("variant") == protocol.get("variant")
            and int(metrics.get("seed")) == int(protocol.get("seed"))
            and all(
                (run_dir / name).exists()
                for name in ["predictions.npy", "targets.npy", "errors_by_sample.npz"]
            )
        )
        if test_complete and not metrics.get("diagnostic_truncated"):
            claim_path = run_dir / "TEST_CLAIM.lock"
            complete_path = run_dir / "TEST_COMPLETE.json"
            if not claim_path.exists() or not complete_path.exists():
                return False
            complete = json.loads(complete_path.read_text(encoding="utf-8"))
            test_complete = (
                complete.get("status") == "test_complete"
                and complete.get("protocol_hash") == config.get("protocol_hash")
                and complete.get("metrics_sha256") == file_sha256(run_dir / "metrics.json")
                and complete.get("predictions_sha256")
                == file_sha256(run_dir / "predictions.npy")
                and complete.get("targets_sha256") == file_sha256(run_dir / "targets.npy")
                and complete.get("errors_by_sample_sha256")
                == file_sha256(run_dir / "errors_by_sample.npz")
            )
        return train_complete and test_complete if stage in {"test", "both"} else False
    except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=list(PRESETS), default="smoke")
    parser.add_argument("--stage", choices=["validate", "train", "test", "both"], default="train")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--variants", nargs="+", choices=list(VARIANT_SPECS))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--llm-path", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--debug-max-batches", type=int)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--allow-repeat-test", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    preset = PRESETS[args.preset]
    seeds = args.seeds or preset["seeds"]
    variants = args.variants or preset["variants"]
    epochs = args.epochs if args.epochs is not None else preset["epochs"]
    debug_max_batches = (
        args.debug_max_batches
        if args.debug_max_batches is not None
        else preset["debug_max_batches"]
    )
    output_root = (args.output_root or HERE / "outputs" / args.preset).resolve()

    if args.stage == "both" and args.preset != "smoke":
        print(
            "WARNING: --stage both opens the test set immediately. "
            "For pilot/confirmatory work, use separate train and test commands."
        )

    commands = []
    for variant in variants:
        for seed in seeds:
            run_dir = output_root / "runs" / variant / f"seed_{seed}"
            expected_protocol = {
                "variant": variant,
                "seed": seed,
                "epochs": epochs,
                "batch_size": args.batch_size,
                "patience": args.patience,
                "learning_rate": args.learning_rate,
                "num_workers": args.num_workers,
                "debug_max_batches": debug_max_batches,
                "device_request": args.device,
                "gpu": args.gpu,
                "use_amp": not args.no_amp,
            }
            if args.data_root:
                expected_protocol["data_root"] = str(args.data_root.resolve())
            if args.llm_path:
                expected_protocol["llm_path"] = str(args.llm_path.resolve())
            replacement_requested = (
                args.force_train and args.stage in {"train", "both"}
            ) or (args.allow_repeat_test and args.stage in {"test", "both"})
            if not replacement_requested and run_is_complete(
                run_dir, args.stage, expected_protocol
            ):
                print(f"skip completed: {variant} seed={seed}")
                continue

            command_stage = args.stage
            if (
                args.stage == "both"
                and debug_max_batches
                and run_is_complete(run_dir, "train", expected_protocol)
                and not run_is_complete(run_dir, "both", expected_protocol)
            ):
                command_stage = "test"
                print(f"resume test after completed diagnostic train: {variant} seed={seed}")

            command = [
                sys.executable,
                str(HERE / "run_one.py"),
                "--stage",
                command_stage,
                "--variant",
                variant,
                "--seed",
                str(seed),
                "--epochs",
                str(epochs),
                "--batch-size",
                str(args.batch_size),
                "--patience",
                str(args.patience),
                "--learning-rate",
                str(args.learning_rate),
                "--num-workers",
                str(args.num_workers),
                "--debug-max-batches",
                str(debug_max_batches),
                "--device",
                args.device,
                "--gpu",
                str(args.gpu),
                "--output-root",
                str(output_root),
            ]
            if args.data_root:
                command.extend(["--data-root", str(args.data_root.resolve())])
            if args.llm_path:
                command.extend(["--llm-path", str(args.llm_path.resolve())])
            if args.no_amp:
                command.append("--no-amp")
            if args.force_train:
                command.append("--force-train")
            if args.allow_repeat_test:
                command.append("--allow-repeat-test")
            commands.append((variant, seed, run_dir, command))

    if args.dry_run:
        for _, _, _, command in commands:
            print(subprocess.list2cmdline(command))
        print(f"dry-run matrix: {len(commands)} commands")
        return 0

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "preset": args.preset,
        "stage": args.stage,
        "variants": variants,
        "seeds": seeds,
        "epochs": epochs,
        "debug_max_batches": debug_max_batches,
        "output_root": str(output_root),
    }
    with (output_root / f"sweep_{args.stage}_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    failures = []
    for index, (variant, seed, run_dir, command) in enumerate(commands, start=1):
        run_dir.mkdir(parents=True, exist_ok=True)
        log_path = run_dir / f"{args.stage}.log"
        print(f"[{index}/{len(commands)}] {variant} seed={seed}")
        with log_path.open("w", encoding="utf-8") as log_handle:
            result = subprocess.run(
                command,
                cwd=HERE,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            failures.append({"variant": variant, "seed": seed, "returncode": result.returncode})
            print(f"FAILED ({result.returncode}); inspect {log_path}")
            if args.stop_on_error:
                break

    summary_command = [
        sys.executable,
        str(HERE / "summarize_results.py"),
        "--output-root",
        str(output_root),
    ]
    summary_result = subprocess.run(summary_command, cwd=HERE, check=False)
    if summary_result.returncode != 0:
        failures.append(
            {"variant": "__summary__", "seed": None, "returncode": summary_result.returncode}
        )
    if failures:
        with (output_root / f"sweep_{args.stage}_failures.json").open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, ensure_ascii=False, indent=2)
        print(f"completed with {len(failures)} failed run(s)")
        return 1
    print(f"completed {len(commands)} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
