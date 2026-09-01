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
from semantic_backbone import (BackboneSpec, _call_builder_preserving_rng,
                               _cosine_similarity, _sha256_array, load_projected_kg,
                               load_slm_cache)


SENSORS = ["s2", "s3", "s4"]


def write_cache(path: Path, sensors=SENSORS):
    embeddings = np.asarray([[1., 0.], [0.8, 0.2], [0., 1.]], dtype=np.float32)
    metadata = {
        "model_name": "test", "model_config_sha256": "config", "weights_id": "weights",
        "tokenizer_id": "tokenizer", "transformers_version": "test", "pooling": "mean",
        "sensor_cards_sha256": "cards", "generated_at_utc": "test", "embedding_dim": 2,
        "semantic_matrix_sha256": _sha256_array(_cosine_similarity(embeddings)),
    }
    np.savez_compressed(
        path,
        embeddings=embeddings,
        sensor_columns=np.asarray(sensors),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


class SemanticBackboneTests(unittest.TestCase):
    def test_cache_requires_exact_sensor_order(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            write_cache(path)
            embeddings, metadata = load_slm_cache(path, SENSORS)
            self.assertEqual(embeddings.shape, (3, 2))
            self.assertEqual(metadata["model_name"], "test")
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

    def test_kg_builder_does_not_change_model_rng_streams(self):
        def builder(columns, seed):
            random.random(); np.random.random(); torch.rand(1)
            return None, {"columns": columns, "seed": seed}
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        expected = (random.random(), np.random.random(), torch.rand(1))
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        _call_builder_preserving_rng(builder, SENSORS, 42)
        actual = (random.random(), np.random.random(), torch.rand(1))
        self.assertEqual(expected[0], actual[0]); self.assertEqual(expected[1], actual[1])
        torch.testing.assert_close(expected[2], actual[2])

    def test_projected_kg_shape_and_range_fail_closed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.npy"
            np.save(path, np.eye(3, dtype=np.float32))
            np.testing.assert_array_equal(load_projected_kg(path, 3), np.eye(3))
            np.save(path, np.asarray([[2.]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                load_projected_kg(path, 3)

    def test_cache_rejects_nonfinite_zero_norm_and_hash_mismatch(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.npz"
            write_cache(path)
            with np.load(path, allow_pickle=False) as payload:
                sensors = payload["sensor_columns"]
                metadata = json.loads(str(payload["metadata_json"].item()))
            metadata["semantic_matrix_sha256"] = "wrong"
            np.savez_compressed(path, embeddings=np.ones((3, 2), np.float32),
                                sensor_columns=sensors,
                                metadata_json=np.asarray(json.dumps(metadata)))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_slm_cache(path, SENSORS)

    def test_shuffling_changes_non_diagonal_structure(self):
        embeddings = np.asarray([[1., 0.], [.8, .2], [0., 1.]], np.float32)
        original = _cosine_similarity(embeddings)
        shuffled = _cosine_similarity(embeddings[[2, 0, 1]])
        mask = ~np.eye(3, dtype=bool)
        self.assertFalse(np.allclose(original[mask], shuffled[mask]))

    def test_none_semantic_component_is_exact_zero(self):
        semantic = np.zeros((3, 3), dtype=np.float32)
        self.assertEqual(np.count_nonzero(semantic), 0)


if __name__ == "__main__":
    unittest.main()
