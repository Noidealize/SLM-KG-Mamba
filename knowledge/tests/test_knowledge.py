from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from validators.validate_kg import project_sensor_graph, read_jsonl, validate


class KnowledgeDraftTests(unittest.TestCase):
    def setUp(self):
        self.schema = json.loads((ROOT / "schema.json").read_text(encoding="utf-8"))
        self.edges = read_jsonl(ROOT / "reference_kg_v1_draft.jsonl")
        self.evidence = read_jsonl(ROOT / "evidence_registry.jsonl")
        self.rules = json.loads((ROOT / "projection_rules.json").read_text(encoding="utf-8"))

    def test_evidenced_draft_validates_without_missing_evidence_warnings(self):
        report = validate(self.schema, self.edges, self.evidence)
        self.assertTrue(report["valid"], report["errors"])
        self.assertEqual(report["warnings"], [])

    def test_invalid_confidence_and_direction_fail(self):
        edge = dict(self.edges[0], evidence_confidence=2.0, head_type="PhysicalQuantity")
        report = validate(self.schema, [edge], [])
        self.assertFalse(report["valid"])

    def test_missing_evidence_span_reference_fails(self):
        edge = dict(self.edges[0], evidence_span_id="missing-span")
        report = validate(self.schema, [edge], self.evidence)
        self.assertFalse(report["valid"])
        self.assertTrue(any("evidence_span_id" in error for error in report["errors"]))

    def test_span_must_explicitly_support_triple(self):
        edge = dict(self.edges[0], triple_id="not-declared-by-span")
        report = validate(self.schema, [edge], self.evidence)
        self.assertFalse(report["valid"])
        self.assertTrue(any("does not declare support" in error for error in report["errors"]))

    def test_unresolved_conflict_cannot_be_projected(self):
        edge = dict(self.edges[-1], projection_eligible=True)
        report = validate(self.schema, [edge], self.evidence)
        self.assertFalse(report["valid"])
        self.assertTrue(any("unresolved conflict" in error for error in report["errors"]))

    def test_projection_shape_range_diagonal_and_shared_neighbor(self):
        sensors = ["s8", "s13", "s2"]
        matrix = project_sensor_graph(self.edges, sensors, self.rules)
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(np.all((matrix >= 0) & (matrix <= 1)))
        np.testing.assert_allclose(np.diag(matrix), 1.0)
        self.assertEqual(matrix[0, 1], 1.0)
        self.assertEqual(matrix[0, 2], 0.0)

    def test_empty_graph_is_identity(self):
        matrix = project_sensor_graph([], ["s1", "s2"], self.rules)
        np.testing.assert_array_equal(matrix, np.eye(2, dtype=np.float32))

    def test_conflicted_w32_edge_is_excluded_from_projection(self):
        matrix = project_sensor_graph(self.edges, ["s20", "s21"], self.rules)
        np.testing.assert_array_equal(matrix, np.eye(2, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
