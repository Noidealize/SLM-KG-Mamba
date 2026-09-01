"""Controlled SMETimes variants for causal ablation experiments.

This module intentionally leaves the upstream SMETimes source untouched.  The
pretrained variants delegate to the original implementation, while the random
and identity controls change only the frozen Llama backbone.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
from torch import nn

_EXPERIMENT_DIR = Path(__file__).resolve().parent
_UPSTREAM_DIR = Path(__file__).resolve().parents[2]
if str(_UPSTREAM_DIR) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_DIR))

from models.SMETimes_Llama import Model as UpstreamSMETimes


@dataclass(frozen=True)
class VariantSpec:
    use_context: bool
    backbone: str
    permute_context: bool = False


VARIANT_SPECS: Dict[str, VariantSpec] = {
    "pretrained_full": VariantSpec(use_context=True, backbone="pretrained"),
    "pretrained_no_context": VariantSpec(use_context=False, backbone="pretrained"),
    "random_no_context": VariantSpec(use_context=False, backbone="random"),
    "identity_no_context": VariantSpec(use_context=False, backbone="identity"),
    "pretrained_shuffled_context": VariantSpec(
        use_context=True, backbone="pretrained", permute_context=True
    ),
}


class ControlledSMETimes(UpstreamSMETimes):
    """Upstream SMETimes with explicit context/backbone controls.

    ``random_no_context`` first constructs the same local pretrained model and
    trainable heads as ``pretrained_no_context``.  It then reinitializes only
    the frozen Llama weights inside a forked RNG context.  This preserves the
    trainable-head initialization for a matched seed.

    ``identity_no_context`` constructs the same heads, removes the Llama, and
    passes encoded numerical patches directly to the unchanged MoE/decoder.
    """

    def __init__(self, configs):
        variant = str(getattr(configs, "variant", ""))
        if variant not in VARIANT_SPECS:
            valid = ", ".join(VARIANT_SPECS)
            raise ValueError(f"Unknown variant {variant!r}. Choose one of: {valid}")

        self.variant = variant
        self.variant_spec = VARIANT_SPECS[variant]
        configs.mix_embeds = self.variant_spec.use_context
        super().__init__(configs)

        if self.variant_spec.backbone == "random":
            self._reinitialize_frozen_backbone(
                int(getattr(configs, "backbone_seed", getattr(configs, "seed", 0) + 100_000))
            )
        elif self.variant_spec.backbone == "identity":
            del self.llama
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _reinitialize_frozen_backbone(self, seed: int) -> None:
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [self.device.index if self.device.index is not None else 0]

        # Do not let random-backbone construction change adapter/dropout/data RNG.
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
            self._standard_llama_random_init()

        for parameter in self.llama.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def _standard_llama_random_init(self) -> None:
        """Apply Llama's standard initializer without HF's loaded-weight guard.

        Transformers 5 marks loaded tensors with ``_is_hf_initialized`` and its
        public initialization helpers intentionally skip them.  Direct
        ``torch.nn.init`` calls are required for a genuine randomized control.
        """
        std = float(getattr(self.llama.config, "initializer_range", 0.02) or 0.02)
        for module in self.llama.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    nn.init.zeros_(module.weight[module.padding_idx])
            elif "RMSNorm" in module.__class__.__name__ or "LayerNorm" in module.__class__.__name__:
                if getattr(module, "weight", None) is not None:
                    nn.init.ones_(module.weight)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.variant_spec.backbone != "identity":
            return super().forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        batch_size, _, n_vars = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1).reshape(batch_size * n_vars, -1)
        fold_out = x_enc.unfold(
            dimension=-1, size=self.token_len, step=self.token_len
        )
        token_num = fold_out.shape[1]

        # The identity control removes only the frozen Llama transformation.
        outputs = self.encoder(fold_out)
        outputs = self._apply_moe(outputs)
        dec_out = self.decoder(outputs)
        dec_out = dec_out.reshape(batch_size, n_vars, -1).permute(0, 2, 1)
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(
            1, token_num * self.token_len, 1
        )
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(
            1, token_num * self.token_len, 1
        )
        return dec_out
