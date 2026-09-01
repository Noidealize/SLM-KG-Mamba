from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


HERE = Path(__file__).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / "formal_experiment_protocol.json").read_text(encoding="utf-8"))
    expected = {(subset, path, seed) for subset in protocol["subsets"]
                for path in protocol["graph_paths"]
                for seed in protocol["pilot_model_seeds"]}
    rows, found = [], set()
    for metrics_path in args.output_root.glob("FD*/reference_kg_projected/F*/model_seed_*/kg_seed_*/*/metrics.json"):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        history_path = metrics_path.with_name("history.json")
        history = json.loads(history_path.read_text(encoding="utf-8"))
        best = min(history, key=lambda row: row["selection"])
        key = (metrics["subset"], metrics["graph_path"], int(metrics["model_seed"]))
        if key not in expected:
            continue
        found.add(key)
        rows.append({"subset": key[0], "graph_path": key[1], "model_seed": key[2],
                     "best_epoch": best["epoch"], "val_rmse": best["val_rmse"],
                     "val_mae": best["val_mae"], "selection": best["selection"],
                     "checkpoint_sha256": metrics["checkpoint_sha256"],
                     "metrics_path": str(metrics_path.resolve())})
    rows.sort(key=lambda row: (row["subset"], row["graph_path"], row["model_seed"]))
    summary_dir = args.output_root / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    csv_path = summary_dir / "pilot_validation_runs.csv"
    fields = list(rows[0]) if rows else ["subset", "graph_path", "model_seed"]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    missing = [{"subset": s, "graph_path": p, "model_seed": seed}
               for s, p, seed in sorted(expected - found)]
    report = {"expected_runs": len(expected), "completed_runs": len(found),
              "complete": found == expected, "missing_runs": missing,
              "official_test_used": False, "paper_evidence": False,
              "runs_csv": str(csv_path.resolve())}
    report_path = summary_dir / "pilot_status.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
