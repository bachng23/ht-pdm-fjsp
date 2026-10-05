"""Shared pending demand, feasible immediate matching and completion-time repair."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
import numpy as np

ENV_VERSION = "pending_matching_completion_v1"
OBSERVATION_CONTRACT = dict(
    machine=6,
    technician=3,
    edge=4,
    global_features=7,
    age_scale=8.0,
    time_scale=12.0,
    cost_scale=20.0,
    machine_count_scale=4.0,
    technician_count_scale=2.0,
    identity_embeddings=False,
    future_events=False,
)


def role_seed(seed, role):
    return int.from_bytes(hashlib.sha256(f"{seed}:{role}".encode()).digest()[:8], "big")


@dataclass(frozen=True)
class DispatchConfig:
    machines: int
    technicians: int
    horizon: int
    failure_age: int
    failure_probability: float
    service_time: tuple[tuple[int, ...], ...]
    restoration: tuple[tuple[float, ...], ...]
    max_age: int = 8
    preventive_cost: float = 1.0
    corrective_cost: float = 2.0
    unavailable_cost: float = 6.0
    failure_cost: float = 15.0

    def validate(self):
        if min(self.machines, self.technicians, self.horizon, self.failure_age) < 1:
            raise ValueError("positive dimensions/horizon/threshold required")
        if self.max_age < self.failure_age or not 0 <= self.failure_probability <= 1:
            raise ValueError("failure law")
        if (
            len(self.service_time) != self.machines
            or len(self.restoration) != self.machines
        ):
            raise ValueError("one row per machine")
        for durations, effects in zip(self.service_time, self.restoration):
            if len(durations) != self.technicians or len(effects) != self.technicians:
                raise ValueError("one column per technician")
            if any(not isinstance(d, int) or d < 0 for d in durations):
                raise ValueError(
                    "integer nonnegative durations; zero means incompatible"
                )
            if any(not math.isfinite(e) or not 0 < e <= 1 for e in effects):
                raise ValueError("restoration in (0,1]")
            if not any(durations):
                raise ValueError("machine without compatible resource")
        if any(
            not any(row[j] for row in self.service_time)
            for j in range(self.technicians)
        ):
            raise ValueError("technician without compatible machine")
        if any(
            not math.isfinite(c) or c < 0
            for c in (
                self.preventive_cost,
                self.corrective_cost,
                self.unavailable_cost,
                self.failure_cost,
            )
        ):
            raise ValueError("finite nonnegative costs")


@dataclass(frozen=True)
class DispatchState:
    ages: tuple[int, ...]
    failed: tuple[bool, ...]
    pending_wait: tuple[int, ...]
    remaining: tuple[int, ...]
    assigned: tuple[int, ...]


def validate_state(config, state):
    if any(
        len(x) != config.machines
        for x in (state.ages, state.failed, state.pending_wait)
    ):
        raise ValueError("machine state dimensions")
    if any(len(x) != config.technicians for x in (state.remaining, state.assigned)):
        raise ValueError("technician state dimensions")
    occupied = [m for m in state.assigned if m >= 0]
    if len(set(occupied)) != len(occupied):
        raise ValueError("machine assigned twice")
    if any(not 0 <= a <= config.max_age for a in state.ages) or any(
        w < 0 for w in state.pending_wait
    ):
        raise ValueError("age/wait state")
    for j, (remaining, machine) in enumerate(zip(state.remaining, state.assigned)):
        if remaining < 0 or (remaining == 0) != (machine == -1):
            raise ValueError("occupancy state")
        if machine != -1 and (
            not 0 <= machine < config.machines or not config.service_time[machine][j]
        ):
            raise ValueError("invalid assigned edge")


def eligible_edges(config, state):
    assigned = set(state.assigned)
    return tuple(
        tuple(
            m not in assigned
            and state.remaining[j] == 0
            and config.service_time[m][j] > 0
            for j in range(config.technicians)
        )
        for m in range(config.machines)
    )


def validate_matching(config, state, pairs):
    machines, technicians = set(), set()
    mask = eligible_edges(config, state)
    for m, j in pairs:
        if (
            not isinstance(m, int)
            or not isinstance(j, int)
            or not 0 <= m < config.machines
            or not 0 <= j < config.technicians
        ):
            raise ValueError("matching index")
        if m in machines or j in technicians or not mask[m][j]:
            raise ValueError("infeasible matching")
        machines.add(m)
        technicians.add(j)


def transition(config, state, pairs, events):
    """One physical interval; actions start together, restoration at its end."""
    pairs = tuple(pairs)
    validate_matching(config, state, pairs)
    if len(events) != config.machines:
        raise ValueError("event dimension")
    ages, failed, pending = (
        list(state.ages),
        list(state.failed),
        list(state.pending_wait),
    )
    remaining, assigned = list(state.remaining), list(state.assigned)
    pm = cm = 0
    for m, j in pairs:
        remaining[j], assigned[j] = config.service_time[m][j], m
        if failed[m]:
            cm += 1
        else:
            pm += 1
    servicing = {m for m in assigned if m >= 0}
    unavailable = sum(failed[m] or m in servicing for m in range(config.machines))
    waiting = sum(failed[m] and m not in servicing for m in range(config.machines))
    failures = 0
    for m in range(config.machines):
        if m in servicing:
            continue
        if failed[m]:
            pending[m] += 1
            continue
        ages[m] = min(config.max_age, ages[m] + 1)
        if ages[m] >= config.failure_age and events[m]:
            failed[m] = True
            pending[m] = 0
            failures += 1
    completed = 0
    for j, m in enumerate(assigned):
        if m == -1:
            continue
        remaining[j] -= 1
        if remaining[j] == 0:
            ages[m] = math.floor(ages[m] * (1 - config.restoration[m][j]))
            failed[m] = False
            pending[m] = 0
            assigned[j] = -1
            completed += 1
    components = dict(
        maintenance_cost=pm * config.preventive_cost + cm * config.corrective_cost,
        unavailability_cost=unavailable * config.unavailable_cost,
        failure_cost=failures * config.failure_cost,
    )
    cost = sum(components.values())
    info = dict(
        objective=cost,
        **components,
        preventive_jobs=pm,
        corrective_jobs=cm,
        jobs=pm + cm,
        completed_jobs=completed,
        failures=failures,
        unavailable_ticks=unavailable,
        failed_waiting_ticks=waiting,
        service_ticks=len(servicing),
        invalid_assignments=0,
    )
    following = DispatchState(
        tuple(ages), tuple(failed), tuple(pending), tuple(remaining), tuple(assigned)
    )
    validate_state(config, following)
    return following, cost, info


def observation(config, state, time):
    assigned_to = {m: j for j, m in enumerate(state.assigned) if m >= 0}
    machines = [
        [
            state.ages[m] / 8.0,
            float(state.failed[m]),
            state.pending_wait[m] / 12.0,
            float(m in assigned_to),
            state.remaining[assigned_to[m]] / 12.0 if m in assigned_to else 0.0,
            config.failure_probability
            if not state.failed[m]
            and m not in assigned_to
            and state.ages[m] + 1 >= config.failure_age
            else 0.0,
        ]
        for m in range(config.machines)
    ]
    technicians = [
        [
            float(state.remaining[j] == 0),
            state.remaining[j] / 12.0,
            float(state.failed[m]) if (m := state.assigned[j]) >= 0 else 0.0,
        ]
        for j in range(config.technicians)
    ]
    edges = [
        [
            [
                float(config.service_time[m][j] > 0),
                config.service_time[m][j] / 12.0,
                config.restoration[m][j],
                float(state.assigned[j] == m),
            ]
            for j in range(config.technicians)
        ]
        for m in range(config.machines)
    ]
    global_features = [
        (config.horizon - time) / 12.0,
        config.failure_age / 8.0,
        config.failure_probability,
        config.preventive_cost / 20.0,
        config.corrective_cost / 20.0,
        config.unavailable_cost / 20.0,
        config.failure_cost / 20.0,
    ]
    return dict(
        machines=np.asarray(machines, dtype=np.float32),
        technicians=np.asarray(technicians, dtype=np.float32),
        edges=np.asarray(edges, dtype=np.float32),
        global_features=np.asarray(global_features, dtype=np.float32),
    )


class DispatchEnv:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.time = 0
        self.reset(0)

    def reset(self, seed, initial=None):
        cfg = self.config
        rng = random.Random(role_seed(seed, "initial"))
        self.state = initial or DispatchState(
            tuple(rng.randrange(cfg.failure_age) for _ in range(cfg.machines)),
            (False,) * cfg.machines,
            (0,) * cfg.machines,
            (0,) * cfg.technicians,
            (-1,) * cfg.technicians,
        )
        validate_state(cfg, self.state)
        self.time = 0
        shocks = random.Random(role_seed(seed, "failures"))
        self.events = tuple(
            tuple(
                shocks.random() < cfg.failure_probability for _ in range(cfg.machines)
            )
            for _ in range(cfg.horizon)
        )
        self.metrics = {
            k: 0.0
            for k in (
                "objective",
                "maintenance_cost",
                "unavailability_cost",
                "failure_cost",
                "preventive_jobs",
                "corrective_jobs",
                "jobs",
                "completed_jobs",
                "failures",
                "unavailable_ticks",
                "failed_waiting_ticks",
                "service_ticks",
                "invalid_assignments",
            )
        }
        return observation(cfg, self.state, self.time)

    def step(self, pairs):
        if self.time >= self.config.horizon:
            raise RuntimeError("episode finished")
        self.state, cost, info = transition(
            self.config, self.state, pairs, self.events[self.time]
        )
        self.time += 1
        for k, v in info.items():
            self.metrics[k] += v
        if (
            abs(
                self.metrics["objective"]
                - sum(
                    self.metrics[k]
                    for k in ("maintenance_cost", "unavailability_cost", "failure_cost")
                )
            )
            > 1e-8
        ):
            raise RuntimeError("cost reconciliation")
        return (
            observation(self.config, self.state, self.time),
            -cost,
            self.time == self.config.horizon,
            info,
        )


def _matrix(rng, n, k, low, high, sparse=False):
    durations = []
    effects = []
    for m in range(n):
        preferred = rng.randrange(k)
        durations.append(
            [
                rng.randint(low, high)
                if j == preferred
                else min(high, rng.randint(low, high) + 1)
                for j in range(k)
            ]
        )
        effects.append([rng.choice((0.5, 0.75, 1.0)) for _ in range(k)])
        if sparse:
            for j in range(k):
                if j != preferred and rng.random() < 0.15:
                    durations[-1][j] = 0
    for j in range(k):
        if not any(row[j] for row in durations):
            durations[rng.randrange(n)][j] = rng.randint(low, high)
    return tuple(tuple(r) for r in durations), tuple(tuple(r) for r in effects)


def training_config(seed, profile):
    rng = random.Random(role_seed(seed, "configuration"))
    n = rng.choice((2, 3, 4))
    k = rng.choice((1, 2))
    duration, effect = _matrix(rng, n, k, 1, 4, True)
    return DispatchConfig(
        n,
        k,
        12 if profile == "full" else 4,
        rng.choice((3, 4, 5)),
        rng.choice((0.25, 0.45, 0.60)),
        duration,
        effect,
    )


FAMILIES = (
    "n3_nominal",
    "n3_pressure",
    "n4_nominal",
    "n4_pressure",
    "n5_pressure",
    "n5_k3_sparse",
    "small",
)


def family_config(family, seed, profile):
    if family == "small":
        return DispatchConfig(
            2,
            2,
            5 if profile == "full" else 3,
            3,
            0.45,
            ((1, 2), (2, 1)),
            ((0.5, 1.0), (1.0, 0.5)),
        )
    if family == "development":
        n, k, low, high, age, p = 3, 2, 1, 3, 4, 0.45
    elif family in FAMILIES:
        n = int(family[1])
        k = 3 if family == "n5_k3_sparse" else 2
        nominal = "nominal" in family
        low, high = (1, 2) if nominal else (2, 4)
        age = 5 if nominal else 3
        p = 0.25 if nominal else 0.6
    else:
        raise ValueError(family)
    duration, effect = _matrix(
        random.Random(role_seed(seed, "evaluation_config:" + family)),
        n,
        k,
        low,
        high,
        family == "n5_k3_sparse",
    )
    return DispatchConfig(
        n, k, 12 if profile == "full" else 4, age, p, duration, effect
    )


def matching_actions(config, state):
    mask = eligible_edges(config, state)
    result = []

    def visit(m, pairs, used):
        if m == config.machines:
            result.append(tuple(pairs))
            return
        visit(m + 1, pairs, used)
        for j, valid in enumerate(mask[m]):
            if valid and j not in used:
                visit(m + 1, pairs + [(m, j)], used | {j})

    visit(0, [], set())
    return result
