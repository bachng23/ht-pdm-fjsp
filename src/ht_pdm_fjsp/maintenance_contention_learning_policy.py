"""Centralized, permutation-equivariant complete-subset PPO policy.

Only current compact machine states and known model parameters enter the actor.
No technician identity or future event array is part of its input contract.
"""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from ht_pdm_fjsp.maintenance_contention_model import subsets, validate_model

TIE_TOLERANCE = 1e-6

FEATURE_CONTRACT = dict(
    machine_features=8,
    global_features=11,
    machine_scales=[8, 1, 12, 12, 3, 1, 1, 1],
    global_scales=[12, 4, 4, 8, 1, 20, 20, 20, 20, 12, 20],
    future_events=False,
    worker_identity=False,
    action="joint_machine_subset",
    deterministic_tie_tolerance=TIE_TOLERANCE,
)


def feature_batch(configs, states, remaining_horizons):
    if not len(configs) == len(states) == len(remaining_horizons):
        raise ValueError("feature batch dimensions")
    n = configs[0].machines
    machine, global_rows, masks = [], [], []
    candidate = subsets(n)
    for c, state, h in zip(configs, states, remaining_horizons, strict=True):
        validate_model(c)
        if c.machines != n or not 1 <= h <= c.horizon:
            raise ValueError("homogeneous N and positive remaining horizon required")
        busy = {m for m, row in enumerate(state) if row[3]}
        free = c.technicians - len(busy)
        machine.append(
            [
                [
                    age / 8,
                    failed,
                    wait / 12,
                    remaining / 12,
                    c.service_time[m][0] / 3,
                    c.restoration[m][0],
                    float(remaining > 0),
                    c.failure_probability
                    if not failed and not remaining and age + 1 >= c.failure_age
                    else 0,
                ]
                for m, (age, failed, wait, remaining) in enumerate(state)
            ]
        )
        global_rows.append(
            [
                h / 12,
                c.technicians / 4,
                free / 4,
                c.failure_age / 8,
                c.failure_probability,
                c.preventive_cost / 20,
                c.corrective_cost / 20,
                c.unavailable_cost / 20,
                c.failure_cost / 20,
                c.waiting_limit / 12,
                c.waiting_price / 20,
            ]
        )
        masks.append(
            [
                len(choice) <= free and not busy.intersection(choice)
                for choice in candidate
            ]
        )
    return dict(
        machines=torch.from_numpy(np.asarray(machine, dtype=np.float32)),
        global_features=torch.from_numpy(np.asarray(global_rows, dtype=np.float32)),
        mask=torch.from_numpy(np.asarray(masks, dtype=bool)),
    )


class SubsetActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.machine_encoder = nn.Sequential(
            nn.Linear(8, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(11, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
        )
        self.actor = nn.Sequential(
            nn.Linear(4 * hidden + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def distribution(self, batch):
        machine = self.machine_encoder(batch["machines"])
        global_rows = self.global_encoder(batch["global_features"])
        b, n, width = machine.shape
        choices = subsets(n)
        membership = machine.new_zeros(len(choices), n)
        for i, choice in enumerate(choices):
            membership[i, list(choice)] = 1
        counts = membership.sum(-1)
        selected = (
            torch.einsum("an,bnh->bah", membership, machine)
            / counts.clamp_min(1)[None, :, None]
        )
        other = (
            torch.einsum("an,bnh->bah", 1 - membership, machine)
            / (n - counts).clamp_min(1)[None, :, None]
        )
        pooled = machine.mean(1)
        context = torch.cat([pooled, global_rows], -1)
        logits = self.actor(
            torch.cat(
                [
                    context[:, None].expand(-1, len(choices), -1),
                    selected,
                    other,
                    (counts / 4)[None, :, None].expand(b, -1, -1),
                ],
                -1,
            )
        ).squeeze(-1)
        if batch["mask"].shape != logits.shape or not batch["mask"][:, 0].all():
            raise ValueError("action mask contract including STOP")
        distribution = Categorical(
            logits=logits.masked_fill(~batch["mask"], -torch.inf)
        )
        return distribution, self.critic(context).squeeze(-1)

    def evaluate(self, batch, action):
        distribution, value = self.distribution(batch)
        if not batch["mask"].gather(1, action[:, None]).all():
            raise ValueError("infeasible recorded action")
        return distribution.log_prob(action), distribution.entropy(), value

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False):
        distribution, value = self.distribution(batch)
        if deterministic:
            # Canonical subset order resolves numerically indistinguishable
            # logits, including economically symmetric machine choices.
            maximum = distribution.logits.max(-1, keepdim=True).values
            tied = batch["mask"] & (distribution.logits >= maximum - TIE_TOLERANCE)
            action = tied.to(torch.int64).argmax(-1)
        else:
            action = torch.multinomial(
                distribution.probs, 1, generator=generator
            ).squeeze(-1)
        return action, distribution.log_prob(action), value

    @torch.no_grad()
    def select(self, c, state, h):
        batch = feature_batch([c], [state], [h])
        action, logprob, value = self.sample(batch, deterministic=True)
        index = int(action[0])
        return subsets(c.machines)[index], dict(
            action_index=index, log_probability=float(logprob[0]), value=float(value[0])
        )
