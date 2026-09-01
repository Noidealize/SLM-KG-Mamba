from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from summarize_results import (
    decision_payload,
    exact_sign_flip_p,
    holm_adjust,
    paired_comparisons,
)


class StatisticsTests(unittest.TestCase):
    def test_exact_sign_flip_detects_consistent_benefit(self):
        p_value = exact_sign_flip_p(np.ones(10))
        self.assertAlmostEqual(p_value, 2 / 1024)

    def test_holm_adjustment_is_monotone_and_bounded(self):
        adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
        self.assertTrue(np.all(adjusted >= np.array([0.01, 0.04, 0.03])))
        self.assertTrue(np.all(adjusted <= 1.0))

    def test_primary_decision_uses_matched_seeds_not_windows(self):
        rows = []
        for seed in range(2025, 2035):
            audit_fields = {
                "comparison_protocol_hash": "same_protocol",
                "environment_hash": "same_environment",
                "trainable_initial_sha256": f"adapter_{seed}",
                "total_params": 100,
                "trainable_params": 10,
            }
            rows.extend(
                [
                    {
                        "variant": "pretrained_no_context",
                        "seed": seed,
                        "mse": 0.95 + (seed - 2025) * 0.0001,
                        "mae": 0.98 + (seed - 2025) * 0.0001,
                        **audit_fields,
                    },
                    {
                        "variant": "random_no_context",
                        "seed": seed,
                        "mse": 1.00 + (seed - 2025) * 0.0001,
                        "mae": 1.00 + (seed - 2025) * 0.0001,
                        **audit_fields,
                    },
                ]
            )
        frame = pd.DataFrame(rows)
        mse_pairs = paired_comparisons(frame, "mse")
        mae_pairs = paired_comparisons(frame, "mae")
        plan = {
            "minimum_relative_improvement": 0.01,
            "mae_noninferiority_margin": 0.01,
            "minimum_win_fraction": 0.8,
            "expected_seeds": list(range(2025, 2035)),
            "plan_content_sha256": "test_plan",
        }
        decision = decision_payload(mse_pairs, mae_pairs, plan)
        self.assertEqual(
            decision["decision"],
            "supports_pretrained_advantage_under_locked_protocol",
        )
        primary = mse_pairs[mse_pairs["contrast"] == "pretraining_primary"].iloc[0]
        self.assertEqual(primary["n_pairs"], 10)
        self.assertEqual(primary["wins"], 10)

    def test_unlocked_results_can_never_emit_support_decision(self):
        rows = []
        for seed in range(2025, 2035):
            audit = {
                "comparison_protocol_hash": "same_protocol",
                "environment_hash": "same_environment",
                "trainable_initial_sha256": f"adapter_{seed}",
                "total_params": 100,
                "trainable_params": 10,
            }
            rows.append(
                {"variant": "pretrained_no_context", "seed": seed, "mse": 0.5, "mae": 0.5, **audit}
            )
            rows.append(
                {"variant": "random_no_context", "seed": seed, "mse": 1.0, "mae": 1.0, **audit}
            )
        frame = pd.DataFrame(rows)
        decision = decision_payload(
            paired_comparisons(frame, "mse"), paired_comparisons(frame, "mae"), None
        )
        self.assertEqual(decision["threshold_status"], "unlocked_descriptive_only")
        self.assertEqual(decision["decision"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main()
