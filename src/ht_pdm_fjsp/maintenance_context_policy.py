"""Matched-capacity allocation heads with/without per-technician peer context."""

import torch
from torch import nn
from torch.distributions import Categorical

from ht_pdm_fjsp.maintenance_dispatch_policy import DispatchActorCritic

TIE_TOLERANCE = 1e-5


class ContextActorCritic(DispatchActorCritic):
    def __init__(self, hidden=64, *, technician_context=False):
        super().__init__(hidden)
        self.technician_context = technician_context
        self.global_encoder = nn.Sequential(
            nn.Linear(11, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.technician_head = nn.Sequential(
            nn.Linear(4 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def deterministic_choice(self, logits):
        maximum = logits.max(-1, keepdim=True).values
        near = torch.isfinite(logits) & (logits >= maximum - TIE_TOLERANCE)
        if not near.any(-1).all():
            raise RuntimeError("no finite deterministic choice")
        return near.to(torch.int64).argmax(-1)

    def technician_distribution(self, encoded, batch, mask, machine, active, prefix):
        m, t, e, g, value = encoded
        b, n, k, h = e.shape
        indices = torch.arange(b)
        selected = machine.clamp_max(n - 1)
        per_technician = t if self.technician_context else torch.zeros_like(t)
        logits = self.technician_head(
            torch.cat(
                (
                    e[indices, selected],
                    g[:, None].expand(-1, k, -1),
                    prefix[:, None].expand(-1, k, -1),
                    per_technician,
                ),
                -1,
            )
        ).squeeze(-1)
        permitted = mask[indices, selected].clone()
        permitted[~active] = False
        permitted[~active, 0] = True
        return Categorical(logits=logits.masked_fill(~permitted, -torch.inf))
