"""Local adapter adding real F0-F3 paths without editing the external project."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from graph_paths import GraphPathSpec, route_graph_context


def create_feasibility_model(base_class, num_sensors: int, reference_kg,
                             path_spec: GraphPathSpec):
    class FeasibilityMARDGMamba(base_class):
        def __init__(self):
            super().__init__(num_sensors, reference_kg)
            self.feasibility_path_spec = path_spec

        def forward(self, x, conditions, return_graph=False):
            ids = torch.arange(self.num_sensors, device=x.device)
            h = self.sensor_value(x.unsqueeze(-1)) + self.sensor_identity(ids).view(
                1, 1, self.num_sensors, -1
            )
            h = h + self.condition_node(conditions).unsqueeze(2)
            h = self.local_norm(F.gelu(h))
            kctx, graph_audit = route_graph_context(
                self.graph, h, conditions, self.a_k, self.feasibility_path_spec
            )
            fused = self.token_fusion(torch.cat([
                h, kctx, conditions.unsqueeze(2).expand(-1, -1, self.num_sensors, -1)
            ], dim=-1))
            weights = torch.softmax(self.node_score(fused).squeeze(-1), dim=-1)
            tokens = torch.einsum("btn,btnd->btd", weights, fused)
            for block in self.blocks:
                tokens = block(tokens)
            raw = self.head(self.final_norm(tokens)[:, -1])
            mu = F.softplus(raw[:, 0])
            log_var = raw[:, 1].clamp(-8.0, 8.0)
            if return_graph:
                return mu, log_var, {**graph_audit, "node_weights": weights}
            return mu, log_var

    return FeasibilityMARDGMamba()
