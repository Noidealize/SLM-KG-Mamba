from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).parent
PROJECT = HERE.parents[1]


def matrix_content_sha256(path: Path) -> str:
    with np.load(path, allow_pickle=False) as payload:
        array = np.ascontiguousarray(payload["matrix"])
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def completed(output_root: Path, subset: str, path: str, seed: int,
              epochs: int, matrix_hash: str) -> bool:
    root = output_root / subset / "reference_kg_projected" / path / f"model_seed_{seed}"
    matches = list(root.glob("kg_seed_42/*/metrics.json"))
    for result_path in matches:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if (result.get("graph_path") == path and result.get("model_seed") == seed
                and result.get("epochs") == epochs
                and result.get("backbone_audit", {}).get("matrix_sha256") == matrix_hash
                and result.get("checkpoint_reload_verified") is True
                and result.get("official_test_used") is False):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["pilot", "confirmatory"], default="pilot")
    parser.add_argument("--subsets", nargs="+", choices=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads((HERE / "formal_experiment_protocol.json").read_text(encoding="utf-8"))
    seeds = protocol[f"{args.stage}_model_seeds"]
    subsets = args.subsets or protocol["subsets"]
    paths = list(protocol["graph_paths"])
    epochs = int(protocol["training"]["epochs"])
    matrix_path = (HERE / protocol["knowledge_baseline"]["projection_matrix"]).resolve()
    # metrics.json records the numeric matrix hash, while the protocol also locks
    # the enclosing NPZ file hash. They are intentionally different hashes.
    matrix_hash = matrix_content_sha256(matrix_path)
    jobs = [(subset, path, seed) for subset in subsets for path in paths for seed in seeds]
    print(json.dumps({"stage": args.stage, "jobs": len(jobs), "subsets": subsets,
                      "paths": paths, "seeds": seeds, "epochs": epochs,
                      "official_test_used": False}, indent=2), flush=True)

    failures = []
    for index, (subset, path, seed) in enumerate(jobs, 1):
        if completed(args.output_root, subset, path, seed, epochs, matrix_hash):
            print(f"[{index}/{len(jobs)}] skip completed {subset} {path} seed={seed}", flush=True)
            continue
        command = [
            sys.executable, str(HERE / "train_semantic.py"),
            "--original-root", str(args.original_root),
            "--data-root", str(args.data_root),
            "--output-root", str(args.output_root),
            "--subset", subset,
            "--semantic-mode", "reference_kg_projected",
            "--reference-kg-matrix", str(matrix_path),
            "--graph-path", path,
            "--epochs", str(epochs),
            "--learning-rate", str(protocol["training"]["learning_rate"]),
            "--model-seed", str(seed),
            "--kg-seed", str(protocol["kg_seed"]),
            "--shuffle-seed", str(protocol["shuffle_seed"]),
        ]
        print(f"[{index}/{len(jobs)}] {subset} {path} seed={seed}", flush=True)
        if args.dry_run:
            print(subprocess.list2cmdline(command), flush=True)
            continue
        log_path = args.output_root / "logs" / subset / path / f"seed_{seed}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            log.write(subprocess.list2cmdline(command) + "\n")
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace", bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line); log.flush()
            returncode = process.wait()
        if returncode:
            failures.append({"subset": subset, "path": path, "seed": seed,
                             "returncode": returncode, "log": str(log_path)})
            break
    summary = {"stage": args.stage, "planned": len(jobs), "failures": failures,
               "completed_without_failure": not failures, "official_test_used": False}
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
