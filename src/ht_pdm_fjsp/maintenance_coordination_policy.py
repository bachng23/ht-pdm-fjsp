"""Simultaneous machine agents with explicit local/global/message contracts."""

import json
import torch
from torch import nn
from torch.distributions import Categorical

from ht_pdm_fjsp import maintenance_dispatch_policy as bp
from ht_pdm_fjsp.maintenance_context_policy import TIE_TOLERANCE

PROPOSAL_ARMS = ("central_proposal", "ippo_local", "mappo_local", "mappo_message")


def urgency(obs, machine):
    row = obs["machines"][machine]
    return float(100 * row[1] + 10 * row[2] + row[0] + row[5])


def resolve(obs, proposals):
    """Public fixed arbitration; losing machines never silently change worker."""
    n, k = len(obs["machines"]), len(obs["technicians"])
    if len(proposals) != n:
        raise ValueError("one proposal per machine")
    contenders = [[] for _ in range(k)]
    for m, j in enumerate(proposals):
        if j == k:
            continue
        if not (0 <= j < k) or not (
            obs["edges"][m, j, 0] > 0
            and obs["machines"][m, 3] < 0.5
            and obs["technicians"][j, 0] > 0.5
        ):
            raise ValueError("infeasible proposal")
        contenders[j].append(m)
    accepted = tuple(
        sorted(
            (min(ms, key=lambda m: (-urgency(obs, m), m)), j)
            for j, ms in enumerate(contenders)
            if ms
        )
    )
    # Maximum cardinality using any currently feasible edge of requesting machines.
    requested = {m for m, p in enumerate(proposals) if p != k}
    masks = {0}
    for m in sorted(requested):
        following = set(masks)
        for mask in masks:
            for j in range(k):
                if (
                    not mask & (1 << j)
                    and obs["edges"][m, j, 0] > 0
                    and obs["technicians"][j, 0] > 0.5
                ):
                    following.add(mask | (1 << j))
        masks = following
    maximum = max(mask.bit_count() for mask in masks)
    return accepted, dict(
        proposals=json.dumps(proposals),
        proposal_count=len(requested),
        contested_technicians=sum(len(ms) > 1 for ms in contenders),
        rejected_proposals=len(requested) - len(accepted),
        avoidable_unassigned_requests=maximum - len(accepted),
    )


class ProposalActorCritic(nn.Module):
    """Parameter-sharing machine agents. Same modules in all four arms."""

    def __init__(self, hidden=64, arm="ippo_local"):
        super().__init__()
        if arm not in PROPOSAL_ARMS:
            raise ValueError(arm)
        self.arm = arm
        self.message_enabled = True
        self.last_diagnostics = {}

        def encoder(inputs):
            return nn.Sequential(
                nn.Linear(inputs, hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
            )

        self.machine_encoder = encoder(7)
        self.technician_encoder = encoder(3)
        self.global_encoder = encoder(11)
        self.edge_encoder = encoder(2 * hidden + 4)
        self.message_encoder = nn.Sequential(nn.Linear(hidden, 4), nn.Tanh())
        self.message_lift = encoder(4)
        self.proposal_head = nn.Sequential(
            nn.Linear(5 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.stop_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def distributions(self, batch):
        exists = batch["machine_exists"]
        b, n = exists.shape
        k = batch["technicians"].shape[1]
        counts = exists.sum(-1)
        ids = torch.arange(n)[None].expand(b, -1) / (counts[:, None] - 1).clamp_min(1)
        m = self.machine_encoder(torch.cat((batch["machines"], ids[..., None]), -1))
        # Public resource broadcast explicitly excludes occupant's failure status.
        public_tech = batch["technicians"].clone()
        public_tech[..., 2] = 0
        t = self.technician_encoder(public_tech)
        g = self.global_encoder(
            torch.cat(
                (
                    batch["global_features"],
                    counts[:, None] / 4,
                    batch["technician_exists"].sum(-1, keepdim=True) / 2,
                ),
                -1,
            )
        )
        edges = self.edge_encoder(
            torch.cat(
                (
                    m[:, :, None].expand(-1, -1, k, -1),
                    t[:, None].expand(-1, n, -1, -1),
                    batch["edges"],
                ),
                -1,
            )
        )
        compatible = (
            (batch["edges"][..., 0] > 0)
            & exists[:, :, None]
            & batch["technician_exists"][:, None]
        )
        own_edges = bp._mean(edges, compatible, 2)
        mask = bp.feasible(
            batch,
            torch.zeros_like(exists),
            torch.zeros_like(batch["technician_exists"]),
        )
        peer = torch.zeros_like(edges)
        if self.arm == "central_proposal" or (
            self.arm == "mappo_message" and self.message_enabled
        ):
            messages = (
                edges if self.arm == "central_proposal" else self.message_encoder(edges)
            )
            contributions = messages * mask[..., None]
            total = contributions.sum(1, keepdim=True) - contributions
            peers = (mask.sum(1, keepdim=True) - mask.to(torch.int64)).clamp_min(1)
            peer = total / peers[..., None]
            if self.arm == "mappo_message":
                peer = self.message_lift(peer)
                # No packet means no peer information, including lift bias.
                peer = (
                    peer
                    * ((mask.sum(1, keepdim=True) - mask.to(torch.int64)) > 0)[
                        ..., None
                    ]
                )
        logits = self.proposal_head(
            torch.cat(
                (
                    m[:, :, None].expand(-1, -1, k, -1),
                    t[:, None].expand(-1, n, -1, -1),
                    edges,
                    g[:, None, None].expand(-1, n, k, -1),
                    peer,
                ),
                -1,
            )
        ).squeeze(-1)
        stop = self.stop_head(
            torch.cat((m, own_edges, g[:, None].expand(-1, n, -1)), -1)
        ).squeeze(-1)
        logits = torch.cat((logits.masked_fill(~mask, -torch.inf), stop[..., None]), -1)
        distribution = Categorical(logits=logits)
        if self.arm == "ippo_local":
            critic = torch.cat((m, own_edges, g[:, None].expand(-1, n, -1)), -1)
        else:
            critic = torch.cat(
                (bp._mean(m, exists, 1), bp._mean(own_edges, exists, 1), g), -1
            )[:, None].expand(-1, n, -1)
        values = self.value_head(critic).squeeze(-1)
        return distribution, values

    def evaluate(self, batch, proposals):
        distribution, values = self.distributions(batch)
        if not torch.isfinite(distribution.log_prob(proposals)).all():
            raise RuntimeError("infeasible replayed proposal")
        return distribution.log_prob(proposals), distribution.entropy(), values

    @torch.no_grad()
    def propose(self, obs, *, deterministic=False, generator=None):
        batch = bp.batch_observations([obs])
        distribution, values = self.distributions(batch)
        if deterministic:
            logits = distribution.logits
            near = torch.isfinite(logits) & (
                logits >= logits.max(-1, keepdim=True).values - TIE_TOLERANCE
            )
            proposals = near.to(torch.int64).argmax(-1)
        else:
            proposals = torch.multinomial(
                distribution.probs.flatten(0, 1), 1, generator=generator
            ).reshape(1, -1)
        logprob = distribution.log_prob(proposals)
        return proposals[0].tolist(), logprob[0], values[0]

    @torch.no_grad()
    def act(self, obs, mode=None, *, deterministic=False, generator=None):
        proposals, probs, values = self.propose(
            obs, deterministic=deterministic, generator=generator
        )
        pairs, diagnostics = resolve(obs, proposals)
        eligible = bp.feasible(
            bp.batch_observations([obs]),
            torch.zeros(1, len(proposals), dtype=torch.bool),
            torch.zeros(1, len(obs["technicians"]), dtype=torch.bool),
        )
        diagnostics["message_scalars"] = (
            int(eligible.sum()) * 4
            if self.arm == "mappo_message" and self.message_enabled
            else 0
        )
        self.last_diagnostics = diagnostics
        return pairs, [*pairs, (-1, -1)], float(probs.sum()), float(values.mean())
