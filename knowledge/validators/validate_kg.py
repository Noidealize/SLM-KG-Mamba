from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def validate(schema: dict, edges: Iterable[dict], evidence_rows: Iterable[dict]) -> dict:
    errors, warnings = [], []
    required = set(schema["edge_required_fields"])
    entity_types = set(schema["entity_types"])
    evidence_rows = list(evidence_rows)
    evidence_ids = set()
    evidence_span_ids = set()
    span_records: dict[str, tuple[str, dict]] = {}
    document_records: dict[str, dict] = {}
    document_required = set(schema.get("evidence_document_required_fields", []))
    span_required = set(schema.get("evidence_span_required_fields", []))
    for row_index, row in enumerate(evidence_rows, 1):
        missing_document = document_required.difference(row)
        if missing_document:
            errors.append(f"evidence-row-{row_index}: missing fields {sorted(missing_document)}")
            continue
        evidence_id = row["evidence_id"]
        if evidence_id in evidence_ids:
            errors.append(f"{evidence_id}: duplicate evidence document id")
        evidence_ids.add(evidence_id)
        document_records[evidence_id] = row
        for span_index, span in enumerate(row["spans"], 1):
            missing_span = span_required.difference(span)
            if missing_span:
                errors.append(f"{evidence_id}/span-{span_index}: missing fields {sorted(missing_span)}")
                continue
            span_id = span["evidence_span_id"]
            if span_id in evidence_span_ids:
                errors.append(f"{span_id}: duplicate evidence span id")
            evidence_span_ids.add(span_id)
            span_records[span_id] = (evidence_id, span)
    triple_ids, triples, entities = set(), set(), set()
    for index, edge in enumerate(edges, 1):
        label = edge.get("triple_id") or f"row-{index}"
        missing = required.difference(edge)
        if missing:
            errors.append(f"{label}: missing fields {sorted(missing)}")
            continue
        if edge["head_type"] not in entity_types or edge["tail_type"] not in entity_types:
            errors.append(f"{label}: illegal entity type")
        allowed = schema["relations"].get(edge["relation"])
        if allowed is None:
            errors.append(f"{label}: illegal relation {edge['relation']}")
        elif [edge["head_type"], edge["tail_type"]] not in allowed:
            errors.append(f"{label}: relation direction/type mismatch")
        confidence = edge["evidence_confidence"]
        if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
            errors.append(f"{label}: evidence_confidence outside [0,1]")
        if edge["extraction_method"] not in schema["extraction_methods"]:
            errors.append(f"{label}: illegal extraction_method")
        if edge["review_status"] not in schema["review_statuses"]:
            errors.append(f"{label}: illegal review_status")
        if edge["conflict_status"] not in schema["conflict_statuses"]:
            errors.append(f"{label}: illegal conflict_status")
        if edge["reference_scope"] not in schema["reference_scopes"]:
            errors.append(f"{label}: illegal reference_scope")
        if not isinstance(edge["projection_eligible"], bool):
            errors.append(f"{label}: projection_eligible must be boolean")
        if edge["conflict_status"] == "unresolved" and edge["projection_eligible"]:
            errors.append(f"{label}: unresolved conflict cannot be projection eligible")
        if edge["review_status"] == "accepted" and not edge["reviewer"].strip():
            errors.append(f"{label}: accepted edge requires a reviewer")
        triple = (edge["head"], edge["relation"], edge["tail"])
        if label in triple_ids or triple in triples:
            errors.append(f"{label}: duplicate triple or triple_id")
        triple_ids.add(label); triples.add(triple)
        entities.update([edge["head"], edge["tail"]])
        if edge["evidence_id"] and edge["evidence_id"] not in evidence_ids:
            errors.append(f"{label}: referenced evidence_id does not exist")
        if edge["evidence_span_id"] and edge["evidence_span_id"] not in evidence_span_ids:
            errors.append(f"{label}: referenced evidence_span_id does not exist")
        elif edge["evidence_span_id"]:
            span_evidence_id, span = span_records[edge["evidence_span_id"]]
            if span_evidence_id != edge["evidence_id"]:
                errors.append(f"{label}: evidence span belongs to a different document")
            if label not in span["supports"]:
                errors.append(f"{label}: evidence span does not declare support for this triple")
            if not document_records[span_evidence_id]["usable_for_edges"]:
                errors.append(f"{label}: cited document is not usable for edges")
        if not edge["evidence_id"]:
            if edge["review_status"] == "accepted":
                errors.append(f"{label}: accepted edge has no evidence")
            else:
                warnings.append(f"{label}: unreviewed edge has no evidence")
    return {"valid": not errors, "errors": errors, "warnings": warnings,
            "edge_count": len(triple_ids), "entity_count": len(entities)}


def project_sensor_graph(edges: Iterable[dict], sensors: list[str], rules: dict) -> np.ndarray:
    if len(set(sensors)) != len(sensors):
        raise ValueError("sensor order contains duplicates")
    allowed_status = set(rules["accepted_review_statuses"])
    graph: dict[str, set[str]] = {}
    for edge in edges:
        if (edge["review_status"] not in allowed_status
                or edge.get("projection_eligible", True) is False):
            continue
        graph.setdefault(edge["head"], set()).add(edge["tail"])
        if rules.get("symmetrize", True):
            graph.setdefault(edge["tail"], set()).add(edge["head"])
    max_hops = int(rules.get("max_hops_from_sensor", 1))
    neighbors: dict[str, set[str]] = {}
    sensor_set = set(sensors)
    for sensor in sensors:
        seen, frontier = {sensor}, {sensor}
        for _ in range(max_hops):
            frontier = {node for current in frontier for node in graph.get(current, set())} - seen
            seen.update(frontier)
        neighbors[sensor] = seen - sensor_set
    matrix = np.zeros((len(sensors), len(sensors)), dtype=np.float32)
    np.fill_diagonal(matrix, float(rules["self_loop_weight"]))
    for i in range(len(sensors)):
        for j in range(i + 1, len(sensors)):
            if neighbors[sensors[i]].intersection(neighbors[sensors[j]]):
                matrix[i, j] = matrix[j, i] = float(rules["shared_neighbor_weight"])
    lo, hi = rules["range"]
    if not np.all(np.isfinite(matrix)) or np.any(matrix < lo) or np.any(matrix > hi):
        raise ValueError("projected graph violates configured numeric range")
    return matrix


def main() -> int:
    parser = argparse.ArgumentParser()
    root = Path(__file__).parents[1]
    parser.add_argument("--knowledge-root", type=Path, default=root)
    parser.add_argument("--project-sensors", nargs="*")
    parser.add_argument("--output-matrix", type=Path)
    args = parser.parse_args()
    schema = json.loads((args.knowledge_root / "schema.json").read_text(encoding="utf-8"))
    edges = read_jsonl(args.knowledge_root / "reference_kg_v1_draft.jsonl")
    evidence = read_jsonl(args.knowledge_root / "evidence_registry.jsonl")
    report = validate(schema, edges, evidence)
    if args.project_sensors:
        rules = json.loads((args.knowledge_root / "projection_rules.json").read_text(encoding="utf-8"))
        matrix = project_sensor_graph(edges, args.project_sensors, rules)
        if args.output_matrix:
            args.output_matrix.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.output_matrix, matrix=matrix,
                                sensor_columns=np.asarray(args.project_sensors))
        report["projection"] = {"shape": list(matrix.shape), "min": float(matrix.min()),
                                "max": float(matrix.max()), "nonzero": int(np.count_nonzero(matrix)),
                                "output_matrix": str(args.output_matrix.resolve()) if args.output_matrix else None}
    report.update({"experiment_status": "diagnostic_feasibility_only",
                   "paper_evidence": False, "official_test_used": False})
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
