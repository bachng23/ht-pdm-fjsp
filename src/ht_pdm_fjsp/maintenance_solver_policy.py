"""Central controls and reservation-safe serial machine policies.

Every arm instantiates identical tensors, including an unused local-critic
branch for cooperative/central arms. Equal total parameters are not equal
active capacity. Central heads preserve the previous matching behavior.
"""
import torch
from torch import nn
from torch.distributions import Categorical
from ht_pdm_fjsp.maintenance_dispatch import validate_matching
from ht_pdm_fjsp import maintenance_heterogeneous_policy as inherited

CENTRAL = inherited.CENTRAL
FIXED = inherited.FIXED
INDEPENDENT = "independent_local_ppo"
LOCAL = "cooperative_local_ppo"
MARL = "cooperative_full_ppo"
ARMS = (CENTRAL, FIXED, INDEPENDENT, LOCAL, MARL)
SERIAL = (INDEPENDENT, LOCAL, MARL)
CATALOGUE = inherited.CATALOGUE
INDEX = inherited.INDEX
TIE = inherited.TIE
FEATURE_CONTRACT = {
    **inherited.FEATURE_CONTRACT,
    "full_information": "central and cooperative_full only",
    "public_rotating_arbitration": False,
    "public_rotating_reservations": True,
    "joint_ratio_proposal_likelihood": False,
    "ratio": "per-machine independent; joint sequence cooperative",
    "local_worker_features": "idle and remaining; no assigned machine failure",
    "local_edges": "own machine only",
    "identical_total_parameters_not_active_capacity": True,
}


def feature_batch(configs, states, remaining_horizons):
    batch = inherited.feature_batch(configs, states, remaining_horizons)
    batch["order"] = torch.tensor(
        [[(c.horizon - h + i) % 4 for i in range(4)]
         for c, h in zip(configs, remaining_horizons, strict=True)], dtype=torch.long)
    return batch


class MatchingActorCritic(inherited.MatchingActorCritic):
    def __init__(self, algorithm, hidden=64):
        if algorithm not in ARMS:
            raise ValueError(algorithm)
        super().__init__(algorithm if algorithm in (CENTRAL, FIXED) else CENTRAL, hidden)
        self.algorithm = algorithm
        self.actor_information = "local" if algorithm in (INDEPENDENT, LOCAL) else "full"
        self.critic_type = "local per-machine" if algorithm == INDEPENDENT else "central team"
        self.local_critic = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def distribution(self, batch):
        if self.algorithm in SERIAL:
            raise ValueError("serial policies require a reservation prefix; use serial_distribution")
        return super().distribution(batch)

    def _encoded(self, batch, local=False):
        edge = self.edge_encoder(batch["edges"])
        exists = batch["technician_exists"].float()
        machine = self.machine_encoder(batch["machines"]) + (
            edge * exists[:, None, :, None]).sum(2) / exists.sum(1).clamp_min(1)[:, None, None]
        tech_features = batch["technicians"]
        if local:
            tech_features = tech_features.clone()
            tech_features[:, :, 2] = 0
        tech = self.technician_encoder(tech_features)
        if not local:
            tech = tech + edge.mean(1)
        g = self.global_encoder(batch["global_features"])
        all_t = (tech * exists[:, :, None]).sum(1) / exists.sum(1).clamp_min(1)[:, None]
        pair = self.pair_encoder(torch.cat([
            machine[:, :, None].expand(-1, 4, 4, -1),
            tech[:, None].expand(-1, 4, 4, -1), edge], -1))
        return machine, tech, g, all_t, pair

    def _values(self, batch, encoded):
        machine, _, g, all_t, _ = encoded
        if self.algorithm == INDEPENDENT:
            return self.local_critic(torch.cat([
                g[:, None].expand(-1, 4, -1), machine,
                all_t[:, None].expand(-1, 4, -1)], -1)).squeeze(-1)
        if self.algorithm == LOCAL:
            machine, _, g, all_t, _ = self._encoded(batch, local=False)
        return self.critic(torch.cat([g, machine.mean(1), all_t], -1)).squeeze(-1)

    def serial_distribution(self, batch, machine_index, reserved, encoded=None):
        """Conditional distribution given public teacher-forced reservations."""
        local = self.algorithm in (INDEPENDENT, LOCAL)
        machine, tech, g, _, pair = (
            self._encoded(batch, local) if encoded is None else encoded)
        b = len(g)
        rows = torch.arange(b, device=g.device)
        own = machine[rows, machine_index]
        options = torch.cat([pair.new_zeros(b, 1, self.hidden), pair[rows, machine_index]], 1)
        available = (batch["technician_exists"] & ~reserved &
                     (batch["technicians"][:, :, 0] > .5)).float()
        free_pool = (tech * available[:, :, None]).sum(1) / available.sum(1).clamp_min(1)[:, None]
        all_m = own if local else machine.mean(1)
        other = torch.zeros_like(own) if local else (machine.sum(1) - own) / 3
        context = torch.cat([
            g[:, None].expand(-1, 5, -1), all_m[:, None].expand(-1, 5, -1),
            free_pool[:, None].expand(-1, 5, -1), options,
            own[:, None].expand(-1, 5, -1), other[:, None].expand(-1, 5, -1),
            (torch.arange(5, device=g.device) > 0).float()[None, :, None].expand(b, -1, -1) / 4,
            (available.sum(1) / 4)[:, None, None].expand(-1, 5, -1)], -1)
        mask = batch["proposal_mask"][rows, machine_index].clone()
        mask[:, 1:] &= ~reserved
        return Categorical(logits=self.actor(context).squeeze(-1).masked_fill(~mask, -torch.inf)), mask

    def _serial(self, batch, action=None, generator=None, deterministic=False, compute_value=True):
        encoded = self._encoded(batch, self.algorithm in (INDEPENDENT, LOCAL))
        b, device = len(batch["machines"]), batch["machines"].device
        rows = torch.arange(b, device=device)
        reserved = torch.zeros((b, 4), dtype=torch.bool, device=device)
        sampled = torch.zeros((b, 4), dtype=torch.long, device=device)
        probabilities, entropies, indices = [], [], []
        if action is not None and action.shape != sampled.shape:
            raise ValueError("recorded serial action dimensions")
        for position in range(4):
            m = batch["order"][:, position]
            dist, mask = self.serial_distribution(batch, m, reserved, encoded)
            if action is not None:
                selected = action[rows, m]
                if ((selected < 0) | (selected > 4)).any() or not mask.gather(1, selected[:, None]).all():
                    raise ValueError("infeasible recorded serial action/prefix")
            elif deterministic:
                maximum = dist.logits.max(-1, keepdim=True).values
                selected = (dist.logits >= maximum - TIE).long().argmax(-1)
            else:
                selected = torch.multinomial(dist.probs, 1, generator=generator).squeeze(-1)
            probabilities.append(dist.log_prob(selected))
            entropies.append(dist.entropy())
            indices.append(m)
            sampled[rows, m] = selected
            reserved = reserved.clone()
            choosing = selected > 0
            reserved[rows[choosing], selected[choosing] - 1] = True
        order = torch.stack(indices, 1)
        logprob = torch.zeros((b, 4), device=device).scatter(1, order, torch.stack(probabilities, 1))
        entropy = torch.zeros_like(logprob).scatter(1, order, torch.stack(entropies, 1))
        if self.algorithm != INDEPENDENT:
            logprob, entropy = logprob.sum(-1), entropy.sum(-1)
        return sampled, logprob, entropy, self._values(batch, encoded) if compute_value else None

    def evaluate(self, batch, action):
        if self.algorithm not in SERIAL:
            return super().evaluate(batch, action)
        _, logprob, entropy, value = self._serial(batch, action=action)
        return logprob, entropy, value

    @torch.no_grad()
    def sample(self, batch, generator=None, deterministic=False):
        if self.algorithm not in SERIAL:
            return super().sample(batch, generator, deterministic)
        action, logprob, _, value = self._serial(batch, generator=generator, deterministic=deterministic)
        return action, logprob, value

    def decode(self, c, state, h, action):
        if self.algorithm not in SERIAL:
            return super().decode(c, state, h, action)
        raw = tuple(int(x) for x in action)
        if len(raw) != 4 or any(x < 0 or x > c.technicians for x in raw):
            raise ValueError("infeasible serial action")
        pairs = tuple((m, j - 1) for m, j in enumerate(raw) if j)
        validate_matching(c, state, pairs)
        return pairs, dict(raw_action=raw, collision_rejections=0,
            proposal_attempts=len(pairs), reservation_rounds=4,
            reservation_order=tuple((c.horizon - h + i) % 4 for i in range(4)))

    @torch.no_grad()
    def select(self, c, state, h):
        batch = feature_batch([c], [state], [h])
        if self.algorithm in SERIAL:
            # Execution needs only the actor: local policies must not acquire
            # private state for a centralized critic diagnostic. Training retains
            # values in sample/evaluate. Central arms retain inherited full-state
            # inference, including its critic (reported in the timing contract).
            action, logprob, _, value = self._serial(
                batch, deterministic=True, compute_value=False)
        else:
            action, logprob, value = self.sample(batch, deterministic=True)
        pairs, diag = self.decode(c, state, h, action[0])
        return pairs, {**diag, "log_probability": logprob[0].tolist(),
                      "value": None if value is None else value[0].tolist(),
                      "critic_evaluated": value is not None}
