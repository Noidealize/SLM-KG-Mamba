from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np

from validators.validate_kg import project_sensor_graph, read_jsonl, validate


ROOT = Path(__file__).parent
PROJECT_ROOT = ROOT.parent
SENSORS = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12",
           "s13", "s14", "s15", "s17", "s20", "s21"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    schema_path = ROOT / "schema.json"
    evidence_path = ROOT / "evidence_registry.jsonl"
    source_path = ROOT / "reference_kg_v1_draft.jsonl"
    rules_path = ROOT / "projection_rules.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    evidence = read_jsonl(evidence_path)
    edges = read_jsonl(source_path)
    report = validate(schema, edges, evidence)
    if not report["valid"] or report["warnings"]:
        raise RuntimeError(f"KG is not freezeable: {report}")

    accepted = [edge for edge in edges if edge["review_status"] == "accepted"]
    pending = [edge for edge in edges if edge["review_status"] == "needs_review"]
    if any(not edge["reviewer"].strip() for edge in accepted):
        raise RuntimeError("Every accepted edge must have a reviewer")

    snapshot_path = ROOT / "reference_kg_v1_reviewed_feasibility.jsonl"
    snapshot_path.write_text("\n".join(json.dumps(row, ensure_ascii=False,
                                                   separators=(",", ":"))
                                       for row in edges) + "\n", encoding="utf-8")
    rules = json.loads(rules_path.read_text(encoding="utf-8"))
    matrix = project_sensor_graph(edges, SENSORS, rules)
    matrix_path = PROJECT_ROOT / "results" / "rul_feasibility" / \
        "reference_kg_v1_reviewed_feasibility_14sensors.npz"
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(matrix_path, matrix=matrix,
                        sensor_columns=np.asarray(SENSORS))

    manifest = {
        "baseline_name": "reference_kg_v1_reviewed_feasibility",
        "frozen_on": date.today().isoformat(),
        "status": "reviewed_feasibility_baseline_not_gold_kg",
        "reviewer": sorted({edge["reviewer"] for edge in accepted}),
        "edge_counts": {"total": len(edges), "accepted": len(accepted),
                        "needs_review": len(pending)},
        "accepted_triple_ids": [edge["triple_id"] for edge in accepted],
        "pending_triple_ids": [edge["triple_id"] for edge in pending],
        "unresolved_conflict_triple_ids": [edge["triple_id"] for edge in edges
                                            if edge["conflict_status"] == "unresolved"],
        "projection": {"sensor_order": SENSORS, "shape": list(matrix.shape),
                       "nonzero": int(np.count_nonzero(matrix))},
        "sha256": {
            "snapshot": sha256(snapshot_path),
            "evidence_registry": sha256(evidence_path),
            "schema": sha256(schema_path),
            "projection_rules": sha256(rules_path),
            "projection_matrix": sha256(matrix_path),
        },
        "validation": report,
        "constraints": {"paper_evidence": False, "official_test_used": False,
                        "gold_kg": False},
    }
    manifest_path = ROOT / "reference_kg_v1_reviewed_feasibility.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({**manifest, "snapshot": str(snapshot_path),
                      "matrix": str(matrix_path), "manifest": str(manifest_path)},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
