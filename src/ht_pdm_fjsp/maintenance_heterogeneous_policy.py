"""Equal-parameter centralized and machine-agent cooperative policies."""

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from ht_pdm_fjsp.maintenance_dispatch import validate_matching, eligible_edges
from ht_pdm_fjsp.maintenance_waiting import observation
from ht_pdm_fjsp.maintenance_heterogeneous_model import (
    catalogue,
    membership,
    fixed_candidates,
    resolve,
)

CENTRAL = "central_matching_ppo"
FIXED = "central_fixed_allocation_ppo"
MARL = "machine_cooperative_ppo"
ARMS = (CENTRAL, FIXED, MARL)
TIE = 1e-6
FEATURE_CONTRACT = dict(
    machine=7,
    technician=3,
    edge=4,
    global_features=11,
    max_machines=4,
    max_technicians=4,
    full_information=True,
    future_events=False,
    public_rotating_arbitration=True,
    joint_ratio_proposal_likelihood=True,
    deterministic_tie_tolerance=TIE,
)
CATALOGUE = catalogue(4, 4)
INDEX = {p: i for i, p in enumerate(CATALOGUE)}


def feature_batch(configs, states, remaining_horizons):
    b = len(configs)
    if not b == len(states) == len(remaining_horizons) or not b:
        raise ValueError("feature dimensions")
    machines = np.zeros((b, 4, 7), np.float32)
    techs = np.zeros((b, 4, 3), np.float32)
    edges = np.zeros((b, 4, 4, 4), np.float32)
    global_rows = []
    exists = np.zeros((b, 4), bool)
    proposal = np.zeros((b, 4, 5), bool)
    proposal[:, :, 0] = True
    joint = np.zeros((b, 209), bool)
    fixed = np.zeros_like(joint)
    for i, (c, state, h) in enumerate(
        zip(configs, states, remaining_horizons, strict=True)
    ):
        if c.machines != 4 or not 1 <= c.technicians <= 4 or not 1 <= h <= c.horizon:
            raise ValueError("N4/K<=4/positive horizon")
        time = c.horizon - h
        obs = observation(c, state, time)
        k = c.technicians
        machines[i, :, :6] = obs["machines"]
        machines[i, :, 6] = [(m - time) % 4 / 4 for m in range(4)]
        techs[i, :k] = obs["technicians"]
        edges[i, :, :k] = obs["edges"]
        exists[i, :k] = True
        valid = (
            (obs["edges"][:, :, 0] > 0.5)
            & (obs["machines"][:, 3, None] < 0.5)
            & (obs["technicians"][None, :, 0] > 0.5)
        )
        proposal[i, :, 1 : k + 1] = valid
        padded = np.zeros((4, 4), bool)
        padded[:, :k] = valid
        joint[i] = np.einsum("amj,mj->a", membership(4, 4), ~padded) == 0
        for pairs in fixed_candidates(c, state):
            fixed[i, INDEX[pairs]] = True
        global_rows.append([*obs["global_features"], c.machines / 4, c.technicians / 4])
    return dict(
        machines=torch.from_numpy(machines),
        technicians=torch.from_numpy(techs),
        edges=torch.from_numpy(edges),
        global_features=torch.tensor(global_rows, dtype=torch.float32),
        technician_exists=torch.from_numpy(exists),
        proposal_mask=torch.from_numpy(proposal),
        matching_mask=torch.from_numpy(joint),
        fixed_mask=torch.from_numpy(fixed),
    )


class MatchingActorCritic(nn.Module):
    def __init__(self, algorithm, hidden=64):
        super().__init__()
        if algorithm not in ARMS:
            raise ValueError(algorithm)
        self.algorithm = algorithm
        self.hidden = hidden

        def encode(d):
            return nn.Sequential(
                nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh()
            )

        self.machine_encoder = encode(7)
        self.technician_encoder = encode(3)
        self.edge_encoder = encode(4)
        self.global_encoder = encode(11)
        self.pair_encoder = encode(3 * hidden)
        self.critic = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(6 * hidden + 2, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.register_buffer(
            "members", torch.from_numpy(membership(4, 4).copy()), persistent=False
        )

    def distribution(self, batch):
        raw_edge = self.edge_encoder(batch["edges"])
        exists = batch["technician_exists"].float()
        machine = (
            self.machine_encoder(batch["machines"])
            + (raw_edge * exists[:, None, :, None]).sum(2)
            / exists.sum(1).clamp_min(1)[:, None, None]
        )
        tech = self.technician_encoder(batch["technicians"]) + raw_edge.mean(1)
        g = self.global_encoder(batch["global_features"])
        all_m = machine.mean(1)
        all_t = (tech * exists[:, :, None]).sum(1) / exists.sum(1).clamp_min(1)[:, None]
        pair = self.pair_encoder(
            torch.cat(
                [
                    machine[:, :, None].expand(-1, 4, 4, -1),
                    tech[:, None].expand(-1, 4, 4, -1),
                    raw_edge,
                ],
                -1,
            )
        )
        value = self.critic(torch.cat([g, all_m, all_t], -1)).squeeze(-1)
        b = len(g)
        free = batch["technicians"][:, :, 0].sum(1) / 4
        if self.algorithm == MARL:
            edge_options = torch.cat([pair.new_zeros(b, 4, 1, self.hidden), pair], 2)
            own = machine[:, :, None].expand(-1, 4, 5, -1)
            other = (machine.sum(1)[:, None] - machine) / 3
            context = torch.cat(
                [
                    g[:, None, None].expand(-1, 4, 5, -1),
                    all_m[:, None, None].expand(-1, 4, 5, -1),
                    all_t[:, None, None].expand(-1, 4, 5, -1),
                    edge_options,
                    own,
                    other[:, :, None].expand(-1, 4, 5, -1),
                    (torch.arange(5, device=g.device) > 0)
                    .float()[None, None, :, None]
                    .expand(b, 4, -1, -1)
                    / 4,
                    free[:, None, None, None].expand(-1, 4, 5, -1),
                ],
                -1,
            )
            logits = self.actor(context).squeeze(-1)
            mask = batch["proposal_mask"]
        else:
            mem = self.members
            count = mem.sum((1, 2))
            chosen_m = mem.sum(2)
            selected_pair = (
                torch.einsum("amj,bmjh->bah", mem, pair)
                / count.clamp_min(1)[None, :, None]
            )
            selected_m = (
                torch.einsum("am,bmh->bah", chosen_m, machine)
                / count.clamp_min(1)[None, :, None]
            )
            other_m = (
                torch.einsum("am,bmh->bah", 1 - chosen_m, machine)
                / (4 - count).clamp_min(1)[None, :, None]
            )
            context = torch.cat(
                [
                    g[:, None].expand(-1, 209, -1),
                    all_m[:, None].expand(-1, 209, -1),
                    all_t[:, None].expand(-1, 209, -1),
                    selected_pair,
                    selected_m,
                    other_m,
                    (count / 4)[None, :, None].expand(b, -1, -1),
                    free[:, None, None].expand(-1, 209, -1),
                ],
                -1,
            )
            logits = self.actor(context).squeeze(-1)
            mask = (
                batch["fixed_mask"]
                if self.algorithm == FIXED
                else batch["matching_mask"]
            )
        if not mask[..., 0].all():
            raise ValueError("STOP mask")
        return Categorical(logits=logits.masked_fill(~mask, -torch.inf)), value

    def evaluate(self, batch, action):
        dist, value = self.distribution(batch)
        mask = (
            batch["proposal_mask"]
            if self.algorithm == MARL
            else batch["fixed_mask"]
            if self.algorithm == FIXED
            else batch["matching_mask"]
        )
        if not mask.gather(-1, action[..., None]).all():
            raise ValueError("infeasible recorded proposal/action")
        prob = dist.log_prob(action)
        entropy = dist.entropy()
        if self.algorithm == MARL:
            prob = prob.sum(-1)
            entropy = entropy.sum(-1)
        return prob, entropy, value

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False):
        dist, value = self.distribution(batch)
        if deterministic:
            maximum = dist.logits.max(-1, keepdim=True).values
            action = (dist.logits >= maximum - TIE).long().argmax(-1)
        else:
            shape = dist.probs.shape[:-1]
            action = torch.multinomial(
                dist.probs.reshape(-1, dist.probs.shape[-1]), 1, generator=generator
            ).reshape(shape)
        prob = dist.log_prob(action)
        return action, prob.sum(-1) if self.algorithm == MARL else prob, value

    def decode(self, c, state, h, action):
        if self.algorithm == MARL:
            proposals = tuple(int(x) for x in action)
            edges = eligible_edges(c, state)
            if len(proposals) != c.machines or any(
                x < 0 or x > c.technicians or (x > 0 and not edges[m][x - 1])
                for m, x in enumerate(proposals)
            ):
                raise ValueError("infeasible raw proposal")
            pairs, rejected = resolve(proposals, c.horizon - h)
        else:
            proposals = None
            rejected = 0
            pairs = CATALOGUE[int(action)]
        validate_matching(c, state, pairs)
        return pairs, dict(
            raw_action=proposals if proposals is not None else int(action),
            collision_rejections=rejected,
            proposal_attempts=sum(x > 0 for x in proposals)
            if proposals is not None
            else len(pairs),
        )

    @torch.no_grad()
    def select(self, c, state, h):
        action, logprob, value = self.sample(
            feature_batch([c], [state], [h]), deterministic=True
        )
        pairs, diag = self.decode(c, state, h, action[0])
        return pairs, {
            **diag,
            "log_probability": float(logprob[0]),
            "value": float(value[0]),
        }
