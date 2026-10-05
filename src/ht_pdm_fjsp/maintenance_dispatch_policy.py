"""Entity actor: choose a machine, allocate a technician, reserve, repeat."""

from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Categorical

MODES = ("learned_both", "fixed_priority", "fixed_allocation")


def batch_observations(rows):
    n = max(len(r["machines"]) for r in rows)
    k = max(len(r["technicians"]) for r in rows)
    b = len(rows)
    result = dict(
        machines=torch.zeros(b, n, 6),
        technicians=torch.zeros(b, k, 3),
        edges=torch.zeros(b, n, k, 4),
        global_features=torch.stack(
            [torch.from_numpy(r["global_features"]) for r in rows]
        ),
        machine_exists=torch.zeros(b, n, dtype=torch.bool),
        technician_exists=torch.zeros(b, k, dtype=torch.bool),
    )
    for i, r in enumerate(rows):
        m, t = len(r["machines"]), len(r["technicians"])
        result["machines"][i, :m] = torch.from_numpy(r["machines"])
        result["technicians"][i, :t] = torch.from_numpy(r["technicians"])
        result["edges"][i, :m, :t] = torch.from_numpy(r["edges"])
        result["machine_exists"][i, :m] = True
        result["technician_exists"][i, :t] = True
    return result


def _mean(values, mask, dimension):
    return (values * mask.unsqueeze(-1)).sum(dimension) / mask.sum(dimension).clamp_min(
        1
    ).unsqueeze(-1)


def feasible(batch, used_m, used_t):
    return (
        (batch["edges"][..., 0] > 0)
        & batch["machine_exists"][:, :, None]
        & batch["technician_exists"][:, None, :]
        & (batch["machines"][..., 3] < 0.5)[:, :, None]
        & (batch["technicians"][..., 0] > 0.5)[:, None, :]
        & ~used_m[:, :, None]
        & ~used_t[:, None, :]
    )


def fixed_machine(batch, mask):
    machines = batch["machines"]
    threshold = batch["global_features"][:, 1] * 8.0
    urgent = (machines[..., 1] > 0.5) | (
        machines[..., 0] * 8.0 >= threshold[:, None] - 1.0 - 1e-5
    )
    scores = (
        100.0 * machines[..., 1]
        + 10.0 * machines[..., 2]
        + machines[..., 0]
        + machines[..., 5]
    )
    scores = scores.masked_fill(~(urgent & mask.any(-1)), -torch.inf)
    chosen = scores.argmax(-1)
    return torch.where(
        torch.isfinite(scores.max(-1).values),
        chosen,
        torch.full_like(chosen, machines.shape[1]),
    )


def fixed_technician(batch, mask, machine):
    b = torch.arange(len(machine))
    m = machine.clamp_max(batch["machines"].shape[1] - 1)
    edges = batch["edges"][b, m]
    score = edges[..., 1] / edges[..., 2].clamp_min(0.01)
    score = score.masked_fill(~mask[b, m], torch.inf)
    return score.argmin(-1)


def rule_action(observation):
    batch = batch_observations([observation])
    n, k = len(observation["machines"]), len(observation["technicians"])
    used_m = torch.zeros(1, n, dtype=torch.bool)
    used_t = torch.zeros(1, k, dtype=torch.bool)
    pairs = []
    for _ in range(k + 1):
        mask = feasible(batch, used_m, used_t)
        m = int(fixed_machine(batch, mask)[0])
        if m == n:
            break
        j = int(fixed_technician(batch, mask, torch.tensor([m]))[0])
        pairs.append((m, j))
        used_m[0, m] = True
        used_t[0, j] = True
    return tuple(pairs)


class DispatchActorCritic(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden

        def mlp(a, b):
            return nn.Sequential(nn.Linear(a, b), nn.Tanh(), nn.Linear(b, b), nn.Tanh())

        self.machine_encoder = mlp(6, hidden)
        self.technician_encoder = mlp(3, hidden)
        self.global_encoder = mlp(9, hidden)
        self.edge_encoder = mlp(2 * hidden + 4, hidden)
        self.machine_update = mlp(2 * hidden, hidden)
        self.technician_update = mlp(2 * hidden, hidden)
        self.context = mlp(3 * hidden, hidden)
        self.machine_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.technician_head = nn.Sequential(
            nn.Linear(3 * hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.stop_head = nn.Sequential(
            nn.Linear(2 * hidden + 1, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1)
        )

    def encode(self, batch):
        m = self.machine_encoder(batch["machines"])
        t = self.technician_encoder(batch["technicians"])
        b, n, h = m.shape
        k = t.shape[1]
        e = self.edge_encoder(
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
            & batch["machine_exists"][:, :, None]
            & batch["technician_exists"][:, None, :]
        )
        m = self.machine_update(torch.cat((m, _mean(e, compatible, 2)), -1))
        t = self.technician_update(torch.cat((t, _mean(e, compatible, 1)), -1))
        g = self.context(
            torch.cat(
                (
                    _mean(m, batch["machine_exists"], 1),
                    _mean(t, batch["technician_exists"], 1),
                    self.global_encoder(
                        torch.cat(
                            (
                                batch["global_features"],
                                batch["machine_exists"].sum(-1, keepdim=True) / 4.0,
                                batch["technician_exists"].sum(-1, keepdim=True) / 2.0,
                            ),
                            -1,
                        )
                    ),
                ),
                -1,
            )
        )
        return m, t, e, g, self.value_head(g).squeeze(-1)

    def distributions(self, encoded, batch, used_m, used_t, prefix, count):
        m, t, e, g, value = encoded
        b, n, h = m.shape
        mask = feasible(batch, used_m, used_t)
        context = torch.cat((g, prefix), -1)
        logits = self.machine_head(
            torch.cat((m, context[:, None].expand(-1, n, -1)), -1)
        ).squeeze(-1)
        logits = logits.masked_fill(~mask.any(-1), -torch.inf)
        stop = self.stop_head(torch.cat((context, count[:, None]), -1))
        return Categorical(logits=torch.cat((logits, stop), -1)), mask

    def technician_distribution(self, encoded, batch, mask, machine, active, prefix):
        m, t, e, g, value = encoded
        b, n, k, h = e.shape
        indices = torch.arange(b)
        selected = machine.clamp_max(n - 1)
        logits = self.technician_head(
            torch.cat(
                (
                    e[indices, selected],
                    g[:, None].expand(-1, k, -1),
                    prefix[:, None].expand(-1, k, -1),
                ),
                -1,
            )
        ).squeeze(-1)
        permitted = mask[indices, selected].clone()
        # STOP/inactive rows are ignored in log probability; give them a finite
        # dummy categorical rather than creating an all-negative-infinity row.
        permitted[~active] = False
        permitted[~active, 0] = True
        return Categorical(logits=logits.masked_fill(~permitted, -torch.inf))

    def decode(self, batch, mode, sequences=None, generator=None, deterministic=False):
        if mode not in MODES:
            raise ValueError(mode)
        if sequences is not None:
            if (
                sequences.ndim != 3
                or sequences.shape[0] != len(batch["machines"])
                or sequences.shape[2] != 2
            ):
                raise RuntimeError("sequence shape")
            valid_pair = (sequences[..., 0] >= 0) & (sequences[..., 1] >= 0)
            stop = (sequences[..., 0] == -1) & (sequences[..., 1] == -1)
            padding = (sequences[..., 0] == -2) & (sequences[..., 1] == -2)
            if not (valid_pair | stop | padding).all():
                raise RuntimeError("sequence token")
        encoded = self.encode(batch)
        m, t, e, g, value = encoded
        b, n, h = m.shape
        k = t.shape[1]
        used_m = torch.zeros(b, n, dtype=torch.bool)
        used_t = torch.zeros(b, k, dtype=torch.bool)
        prefix = g.new_zeros(b, h)
        count = g.new_zeros(b)
        active = torch.ones(b, dtype=torch.bool)
        logprob = g.new_zeros(b)
        entropy = g.new_zeros(b)
        result = []
        stages = sequences.shape[1] if sequences is not None else k + 1
        for stage in range(stages):
            machine_dist, mask = self.distributions(
                encoded, batch, used_m, used_t, prefix, count
            )
            if sequences is None:
                if mode == "fixed_priority":
                    machine = fixed_machine(batch, mask)
                elif deterministic:
                    machine = machine_dist.logits.argmax(-1)
                else:
                    machine = torch.multinomial(
                        machine_dist.probs, 1, generator=generator
                    ).squeeze(-1)
                machine = torch.where(active, machine, torch.full_like(machine, n))
            else:
                requested = sequences[:, stage, 0]
                if not torch.equal(active, requested != -2):
                    raise RuntimeError("sequence termination/padding mismatch")
                machine = torch.where(
                    requested < 0, torch.full_like(requested, n), requested
                )
                if ((machine < 0) | (machine > n)).any():
                    raise RuntimeError("invalid sequence machine")
                if mode == "fixed_priority" and not torch.equal(
                    machine[active], fixed_machine(batch, mask)[active]
                ):
                    raise RuntimeError("fixed priority sequence mismatch")
            pair_active = active & (machine != n)
            chosen_allowed = (
                torch.cat((mask.any(-1), torch.ones(b, 1, dtype=torch.bool)), -1)
                .gather(1, machine[:, None])
                .squeeze(-1)
            )
            if not chosen_allowed[active].all():
                raise RuntimeError("infeasible machine sequence")
            if mode != "fixed_priority":
                logprob = logprob + machine_dist.log_prob(machine) * active
                entropy = entropy + machine_dist.entropy() * active
            technician_dist = self.technician_distribution(
                encoded, batch, mask, machine, pair_active, prefix
            )
            if sequences is None:
                if mode == "fixed_allocation":
                    technician = fixed_technician(batch, mask, machine)
                elif deterministic:
                    technician = technician_dist.logits.argmax(-1)
                else:
                    technician = torch.multinomial(
                        technician_dist.probs, 1, generator=generator
                    ).squeeze(-1)
                technician = torch.where(
                    pair_active, technician, torch.zeros_like(technician)
                )
            else:
                technician = torch.where(
                    pair_active, sequences[:, stage, 1], torch.zeros_like(machine)
                )
                if ((technician < 0) | (technician >= k)).any():
                    raise RuntimeError("invalid sequence technician")
                if mode == "fixed_allocation" and not torch.equal(
                    technician[pair_active],
                    fixed_technician(batch, mask, machine)[pair_active],
                ):
                    raise RuntimeError("fixed allocation sequence mismatch")
            indices = torch.arange(b)
            selected = machine.clamp_max(n - 1)
            if not mask[
                indices[pair_active], selected[pair_active], technician[pair_active]
            ].all():
                raise RuntimeError("infeasible technician sequence")
            if mode != "fixed_allocation":
                logprob = logprob + technician_dist.log_prob(technician) * pair_active
                entropy = entropy + technician_dist.entropy() * pair_active
            result.append(
                torch.stack(
                    (
                        torch.where(
                            active,
                            torch.where(
                                pair_active, machine, torch.full_like(machine, -1)
                            ),
                            torch.full_like(machine, -2),
                        ),
                        torch.where(
                            active,
                            torch.where(
                                pair_active, technician, torch.full_like(machine, -1)
                            ),
                            torch.full_like(machine, -2),
                        ),
                    ),
                    -1,
                )
            )
            # Clones avoid in-place version changes in masks used by autograd.
            next_m, next_t = used_m.clone(), used_t.clone()
            next_m[indices[pair_active], selected[pair_active]] = True
            next_t[indices[pair_active], technician[pair_active]] = True
            used_m, used_t = next_m, next_t
            new = e[indices, selected, technician]
            prefix = torch.where(
                pair_active[:, None],
                (prefix * count[:, None] + new) / (count + 1).clamp_min(1)[:, None],
                prefix,
            )
            count = count + pair_active.float()
            active = pair_active
        if active.any():
            raise RuntimeError("sequence missing terminal STOP")
        return torch.stack(result, 1), logprob, entropy, value

    def act(self, observation, mode, generator=None, deterministic=False):
        with torch.no_grad():
            sequence, logprob, entropy, value = self.decode(
                batch_observations([observation]),
                mode,
                generator=generator,
                deterministic=deterministic,
            )
        tokens = sequence[0].tolist()
        pairs = tuple((int(m), int(j)) for m, j in tokens if m >= 0)
        tokens = [tuple(a) for a in tokens if a[0] != -2]
        return pairs, tokens, float(logprob[0]), float(value[0])


def batch_sequences(rows):
    stages = max(len(row) for row in rows)
    result = torch.full((len(rows), stages, 2), -2, dtype=torch.long)
    for i, row in enumerate(rows):
        result[i, : len(row)] = torch.tensor(row, dtype=torch.long)
    return result
