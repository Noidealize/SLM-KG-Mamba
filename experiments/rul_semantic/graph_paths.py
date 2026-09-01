"""Explicit F0-F3 graph routing for feasibility validation.

This module is intentionally independent of the external Mamba package so the
graph-control contract can be tested with synthetic tensors on CPU.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


GraphPath = Literal["F0", "F1", "F2", "F3"]


@dataclass(frozen=True)
class GraphPathSpec:
    path: GraphPath
    fixed_fusion_alpha: float = 0.5

    def __post_init__(self):
        if self.path not in {"F0", "F1", "F2", "F3"}:
            raise ValueError(f"unsupported graph path: {self.path}")
        if not 0.0 <= self.fixed_fusion_alpha <= 1.0:
            raise ValueError("fixed_fusion_alpha must be in [0,1]")


def route_graph_context(graph, h: torch.Tensor, conditions: torch.Tensor,
                        reference_kg: torch.Tensor, spec: GraphPathSpec):
    """Return graph context and an audit dictionary for a real F0-F3 route."""
    n = h.shape[2]
    identity = torch.eye(n, dtype=h.dtype, device=h.device)
    if spec.path == "F0":
        zeros = h.new_zeros((*h.shape[:3], n))
        return torch.zeros_like(h), {
            "path": "F0", "graph_called": False, "knowledge_used": False,
            "dynamic_used": False, "a_t": zeros,
        }
    if spec.path == "F1":
        context, a_t, a_data, gate = graph(h, conditions, identity, False)
        return context, {"path": "F1", "graph_called": True, "knowledge_used": False,
                         "dynamic_used": True, "a_t": a_t, "a_data": a_data, "gate": gate}
    if spec.path == "F2":
        context, a_t, a_data, gate = graph(h, conditions, reference_kg, True)
        return context, {"path": "F2", "graph_called": True, "knowledge_used": True,
                         "dynamic_used": False, "a_t": a_t, "a_data": a_data, "gate": gate}

    _, _, a_data, gate = graph(h, conditions, identity, False)
    alpha = spec.fixed_fusion_alpha
    base = reference_kg.view(1, 1, n, n)
    a_t = alpha * base + (1.0 - alpha) * a_data
    a_norm = a_t / a_t.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    context = torch.einsum("btij,btjd->btid", a_norm, h)
    return context, {"path": "F3", "graph_called": True, "knowledge_used": True,
                     "dynamic_used": True, "fixed_fusion_alpha": alpha,
                     "a_t": a_t, "a_data": a_data, "gate": gate}
