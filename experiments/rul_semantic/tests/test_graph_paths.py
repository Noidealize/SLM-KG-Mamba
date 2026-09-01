from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from graph_paths import GraphPathSpec, route_graph_context


class FakeGraph(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.3))
        self.calls = 0

    def forward(self, h, conditions, a_k, disable_dynamic=False):
        self.calls += 1
        n = h.shape[2]
        a_data = torch.softmax(self.scale * torch.ones((*h.shape[:2], n, n)), dim=-1)
        base = a_k.view(1, 1, n, n)
        a_t = base if disable_dynamic else base + (1-base) * a_data
        a_norm = a_t / a_t.sum(-1, keepdim=True).clamp_min(1e-8)
        context = torch.einsum("btij,btjd->btid", a_norm, h)
        return context, a_t, a_data, torch.ones_like(a_data)


class GraphPathTests(unittest.TestCase):
    def setUp(self):
        self.h = torch.arange(12.0).view(1, 2, 3, 2).requires_grad_()
        self.c = torch.zeros(1, 2, 3)
        self.kg = torch.tensor([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])

    def test_f0_is_real_bypass_and_kg_invariant(self):
        graph = FakeGraph()
        first, audit = route_graph_context(graph, self.h, self.c, self.kg, GraphPathSpec("F0"))
        second, _ = route_graph_context(graph, self.h, self.c, torch.zeros_like(self.kg), GraphPathSpec("F0"))
        self.assertEqual(graph.calls, 0); self.assertFalse(audit["graph_called"])
        torch.testing.assert_close(first, second); torch.testing.assert_close(first, torch.zeros_like(first))

    def test_f1_is_kg_invariant_but_dynamic_path_runs(self):
        graph = FakeGraph()
        first, _ = route_graph_context(graph, self.h, self.c, self.kg, GraphPathSpec("F1"))
        second, _ = route_graph_context(graph, self.h, self.c, 1-self.kg, GraphPathSpec("F1"))
        torch.testing.assert_close(first, second); self.assertEqual(graph.calls, 2)

    def test_f2_and_f3_are_kg_sensitive(self):
        for path in ("F2", "F3"):
            graph = FakeGraph()
            first, _ = route_graph_context(graph, self.h, self.c, self.kg, GraphPathSpec(path))
            second, _ = route_graph_context(graph, self.h, self.c, torch.eye(3), GraphPathSpec(path))
            self.assertFalse(torch.allclose(first, second), path)

    def test_backward_reaches_input_on_all_paths(self):
        for path in ("F0", "F1", "F2", "F3"):
            h = self.h.detach().clone().requires_grad_()
            context, _ = route_graph_context(FakeGraph(), h, self.c, self.kg, GraphPathSpec(path))
            (context.sum() + h.sum()).backward()
            self.assertIsNotNone(h.grad)


if __name__ == "__main__":
    unittest.main()
