from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from analysis_plan import (
    LOCK_FILENAME,
    comparable_environment_hash,
    comparison_protocol_hash,
    plan_content_hash,
    sha256_file,
    sha256_json,
    verify_test_authorization,
)


class AnalysisPlanTests(unittest.TestCase):
    def test_locked_artifacts_are_verified_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_root = Path(temporary).resolve()
            variant = "pretrained_no_context"
            seed = 2025
            run_dir = output_root / "runs" / variant / f"seed_{seed}"
            checkpoint = run_dir / "checkpoint" / "trainable_state.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint-v1")
            checkpoint_sha = sha256_file(checkpoint)

            environment = {
                "platform": "test",
                "torch": "test",
                "numpy": "test",
                "pandas": "test",
                "transformers": "test",
                "cuda_available": False,
                "torch_cuda": None,
                "cuda_device_0": None,
            }
            audit = {
                "trainable_initial_sha256": "adapter",
                "total_params": 100,
                "trainable_params": 10,
                "backbone_sample_sha256": "pretrained",
            }
            (run_dir / "environment.json").write_text(
                json.dumps(environment), encoding="utf-8"
            )
            (run_dir / "model_audit.json").write_text(json.dumps(audit), encoding="utf-8")
            (run_dir / "train_summary.json").write_text(
                json.dumps({"checkpoint_sha256": checkpoint_sha}), encoding="utf-8"
            )

            protocol = {"variant": variant, "seed": seed, "backbone_seed": 102025, "x": 1}
            protocol_sha = sha256_json(protocol)
            env_sha = comparable_environment_hash(environment)
            plan = {
                "status": "locked_before_test",
                "output_root": str(output_root),
                "comparison_protocol_hash": comparison_protocol_hash(protocol),
                "environment_hash": env_sha,
                "expected_runs": {
                    f"{variant}:seed_{seed}": {
                        "protocol_hash": protocol_sha,
                        "checkpoint_sha256": checkpoint_sha,
                        "environment_hash": env_sha,
                    }
                },
                "pair_audit": {str(seed): {variant: audit}},
            }
            plan["plan_content_sha256"] = plan_content_hash(plan)
            (output_root / LOCK_FILENAME).write_text(json.dumps(plan), encoding="utf-8")

            verified = verify_test_authorization(output_root, protocol_sha, protocol)
            self.assertEqual(verified["plan_content_sha256"], plan["plan_content_sha256"])

            checkpoint.write_bytes(b"tampered")
            with self.assertRaises(RuntimeError):
                verify_test_authorization(output_root, protocol_sha, protocol)


if __name__ == "__main__":
    unittest.main()

