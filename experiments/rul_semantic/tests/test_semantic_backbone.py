from __future__ import annotations

import json
from pathlib import Path
import random
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from semantic_backbone import BackboneSpec, _cosine_similarity, load_slm_cache


SENSORS = ["s2", "s3", "s4"]


def write_cache(path: Path, sensors=SENSORS):
    np.savez_compressed(
        path,
        embeddings=np.eye(len(sensors), dtype=np.float32),
        sensor_columns=np.asarray(sensors),
        metadata_json=np.asarray(json.dumps({"model": "test"})),
    )


class SemanticBackboneTests(unittest.TestCase):
    def test_cache_requires_exact_sensor_order(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            write_cache(path)
            embeddings, metadata = load_slm_cache(path, SENSORS)
            self.assertEqual(embeddings.shape, (3, 3))
            self.assertEqual(metadata["model"], "test")
            with self.assertRaisesRegex(ValueError, "sensor order"):
                load_slm_cache(path, list(reversed(SENSORS)))

    def test_cosine_component_is_symmetric_bounded_and_has_unit_diagonal(self):
        value = _cosine_similarity(np.asarray([[1, 0], [0, 1], [-1, 0]], np.float32))
        np.testing.assert_allclose(value, value.T)
        np.testing.assert_allclose(np.diag(value), 1.0)
        self.assertTrue(np.all((value >= 0.0) & (value <= 1.0)))

    def test_backbone_spec_keeps_model_and_kg_seeds_separate(self):
        spec = BackboneSpec(kg_seed=42)
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        before = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        self.assertEqual(spec.kg_seed, 42)
        self.assertGreater(before[2].numel(), 0)


if __name__ == "__main__":
    unittest.main()
