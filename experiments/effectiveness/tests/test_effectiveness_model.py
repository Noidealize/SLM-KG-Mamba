from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn


HERE = Path(__file__).resolve().parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from effectiveness_model import ControlledSMETimes
from experiment import model_audit, set_global_seed


class FakeCore(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.projection = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, inputs_embeds):
        return (torch.tanh(self.projection(inputs_embeds)),)


class FakeLlama(nn.Module):
    def __init__(self, hidden_size: int = 8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.model = FakeCore(hidden_size)
        # Reproduce the guard set on weights loaded by Transformers 5.
        for parameter in self.parameters():
            parameter._is_hf_initialized = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def configs(variant: str):
    return SimpleNamespace(
        token_len=2,
        device="cpu",
        gpu=0,
        use_amp=False,
        use_multi_gpu=False,
        local_rank=0,
        llm_ckp_dir=str(HERE),
        local_files_only=True,
        mix_embeds=True,
        variant=variant,
        seed=7,
        backbone_seed=1007,
        num_experts=2,
        lambda_reg=0.01,
        mlp_hidden_layers=0,
        mlp_hidden_dim=4,
        dropout=0.0,
        mlp_activation="tanh",
    )


class EffectivenessModelTests(unittest.TestCase):
    def build(self, variant: str):
        set_global_seed(7)
        with patch(
            "models.SMETimes_Llama.LlamaForCausalLM.from_pretrained",
            side_effect=lambda *args, **kwargs: FakeLlama(),
        ):
            return ControlledSMETimes(configs(variant)).eval()

    def test_all_variants_have_the_same_output_shape(self):
        x = torch.randn(2, 4, 1)
        x_mark = torch.randn(2, 2, 8)
        for variant in [
            "pretrained_full",
            "pretrained_no_context",
            "random_no_context",
            "identity_no_context",
            "pretrained_shuffled_context",
        ]:
            with self.subTest(variant=variant):
                output = self.build(variant)(x, x_mark, None, None)
                self.assertEqual(tuple(output.shape), (2, 4, 1))

    def test_no_context_is_invariant_to_context_tensor(self):
        model = self.build("pretrained_no_context")
        x = torch.randn(2, 4, 1)
        first = model(x, torch.randn(2, 2, 8), None, None)
        second = model(x, torch.randn(2, 2, 8) * 1000, None, None)
        torch.testing.assert_close(first, second)

    def test_matched_controls_have_identical_trainable_initialization(self):
        pretrained = model_audit(self.build("pretrained_no_context"))
        random_control = model_audit(self.build("random_no_context"))
        identity = model_audit(self.build("identity_no_context"))
        self.assertEqual(pretrained["trainable_signature"], random_control["trainable_signature"])
        self.assertEqual(
            pretrained["trainable_initial_sha256"],
            random_control["trainable_initial_sha256"],
        )
        self.assertEqual(
            pretrained["trainable_initial_sha256"], identity["trainable_initial_sha256"]
        )

    def test_random_backbone_is_reproducible(self):
        first = model_audit(self.build("random_no_context"))
        second = model_audit(self.build("random_no_context"))
        self.assertEqual(first["backbone_sample_sha256"], second["backbone_sample_sha256"])

    def test_random_backbone_differs_from_loaded_pretrained_weights(self):
        pretrained = model_audit(self.build("pretrained_no_context"))
        random_control = model_audit(self.build("random_no_context"))
        self.assertNotEqual(
            pretrained["backbone_sample_sha256"],
            random_control["backbone_sample_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
