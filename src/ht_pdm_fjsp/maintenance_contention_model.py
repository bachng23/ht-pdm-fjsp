"""Identical-worker contention model and common-random-number rollout planner.

Compact states store (age, failed, pending wait, service remaining) per machine.
Worker identities are intentionally quotiented out, never capacity constraints.
"""

import itertools
import math
import random
from functools import lru_cache

import numpy as np

from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig
from ht_pdm_fjsp.maintenance_coordination_references import isolated_value

REFERENCES = (
    "risk_skill_rule",
    "deadline_matching",
    "lookahead_matching",
    "independent_dp_matching",
)
CAPACITIES = {"low": 4, "middle": 2, "high": 1}


def cohort_config(seed, capacity, horizon, machines=4):
    rng = random.Random(role_seed(seed, "contention_configuration"))
    durations = [rng.randint(1, 3) for _ in range(4)]
    effects = [rng.choice((0.5, 0.75, 1.0)) for _ in range(4)]
    k = min(CAPACITIES[capacity], machines)
    return WaitingConfig(
        machines,
        k,
        horizon,
        3,
        0.6,
        tuple((d,) * k for d in durations[:machines]),
        tuple((r,) * k for r in effects[:machines]),
        waiting_price=12.0,
    )


def validate_model(c):
    c.validate()
    if (
        c.max_age != 8
        or any(len(set(row)) != 1 or row[0] < 1 for row in c.service_time)
        or any(len(set(row)) != 1 for row in c.restoration)
    ):
        raise ValueError("requires identical compatible workers and max_age=8")


def compact(state):
    remaining = [0] * len(state.ages)
    for m, r in zip(state.assigned, state.remaining):
        if m >= 0:
            remaining[m] = r
    return tuple(zip(state.ages, map(int, state.failed), state.pending_wait, remaining))


def physical_action(state, selected):
    free = [j for j, r in enumerate(state.remaining) if r == 0]
    return tuple(zip(sorted(selected), free))


@lru_cache(maxsize=32)
def subsets(n):
    # Lexicographic subset order gives stable ties and includes STOP.
    return tuple(
        sorted(
            s for size in range(n + 1) for s in itertools.combinations(range(n), size)
        )
    )


def actions(c, state):
    free = c.technicians - sum(row[3] > 0 for row in state)
    return tuple(
        s
        for s in subsets(c.machines)
        if len(s) <= free and all(state[m][3] == 0 for m in s)
    )


def tick(c, state, selected, shocks):
    """Independent scalar compact transition, cross-checked against WaitingEnv."""
    rows = []
    cost = 0.0
    for m, (age, failed, wait, remaining) in enumerate(state):
        start = m in selected
        if start:
            cost += c.corrective_cost if failed else c.preventive_cost
            remaining = c.service_time[m][0]
        cost += c.unavailable_cost if failed or remaining else 0
        if remaining:
            remaining -= 1
            if remaining == 0:
                age = math.floor(age * (1 - c.restoration[m][0]))
                failed, wait = 0, 0
        elif failed:
            wait += 1
            cost += c.waiting_price if wait > c.waiting_limit else 0
        else:
            age = min(c.max_age, age + 1)
            if age >= c.failure_age and shocks[m]:
                failed, wait = 1, 0
                cost += c.failure_cost
        rows.append((age, failed, wait, remaining))
    return tuple(rows), cost


def branches(c, state, selected):
    eligible = [
        m
        for m, row in enumerate(state)
        if not row[1]
        and not row[3]
        and m not in selected
        and min(c.max_age, row[0] + 1) >= c.failure_age
    ]
    for bits in itertools.product((False, True), repeat=len(eligible)):
        p = math.prod(
            c.failure_probability if b else 1 - c.failure_probability for b in bits
        )
        if p:
            events = [False] * c.machines
            for m, b in zip(eligible, bits):
                events[m] = b
            following, cost = tick(c, state, selected, events)
            yield p, following, cost


def isolated_args(c, m):
    return (
        c.service_time[m][0],
        c.restoration[m][0],
        c.failure_age,
        c.failure_probability,
        c.preventive_cost,
        c.corrective_cost,
        c.unavailable_cost,
        c.failure_cost,
        c.waiting_limit,
        c.waiting_price,
    )


def relaxed_value(c, state, h):
    return sum(
        isolated_value(h, *row, *isolated_args(c, m)) for m, row in enumerate(state)
    )


@lru_cache(maxsize=200000)
def bid(c, m, h, age, failed, wait):
    args = isolated_args(c, m)
    h -= 1
    if failed:
        defer = (
            c.unavailable_cost
            + (c.waiting_price if wait + 1 > c.waiting_limit else 0)
            + isolated_value(h, age, 1, wait + 1, 0, *args)
        )
    else:
        new_age = min(c.max_age, age + 1)
        p = c.failure_probability if new_age >= c.failure_age else 0
        defer = p * (c.failure_cost + isolated_value(h, new_age, 1, 0, 0, *args)) + (
            1 - p
        ) * isolated_value(h, new_age, 0, 0, 0, *args)
    duration, effect = args[:2]
    following = (
        (math.floor(age * (1 - effect)), 0, 0, 0)
        if duration == 1
        else (age, failed, wait, duration - 1)
    )
    start = (
        (c.corrective_cost if failed else c.preventive_cost)
        + c.unavailable_cost
        + isolated_value(h, *following, *args)
    )
    return defer - start


def reference_batch(c, states, h, name):
    """Reference decisions for B compact states; stable machine-ID ties."""
    if name not in REFERENCES:
        raise ValueError(name)
    age, failed, wait, remaining = (states[:, :, i] for i in range(4))
    urgent = (failed > 0) | (age >= c.failure_age - 1)
    hazard = np.where(
        (age + 1 >= c.failure_age) & (failed == 0) & (remaining == 0),
        c.failure_probability,
        0,
    )
    if name == "risk_skill_rule":
        # Match original float32 observation and torch arithmetic.
        weights = (
            100 * failed.astype(np.float32)
            + 10 * (wait.astype(np.float32) / 12)
            + age.astype(np.float32) / 8
            + hazard.astype(np.float32)
        )
        weights = np.where(urgent, weights, -np.inf)
    elif name == "deadline_matching":
        weights = (
            100 * failed
            + 20 * np.maximum(0, (wait.astype(np.float32) / 12).astype(float) * 12 - 3)
            + 10 * hazard.astype(np.float32).astype(float)
            + age / 8
        )
        weights += np.asarray(
            [c.restoration[m][0] / c.service_time[m][0] for m in range(c.machines)]
        )
        weights = np.where(urgent, weights, -np.inf)
    else:
        weights = np.zeros(age.shape)
        horizon = min(6, h) if name == "lookahead_matching" else h
        for m in range(c.machines):
            values, inverse = np.unique(states[:, m, :3], axis=0, return_inverse=True)
            gains = [bid(c, m, horizon, *map(int, row)) for row in values]
            weights[:, m] = np.asarray(gains)[inverse]
    weights = np.where(remaining == 0, weights, -np.inf)
    free = c.technicians - (remaining > 0).sum(axis=1)
    order = np.argsort(-weights, axis=1, kind="stable")
    chosen = np.zeros(age.shape, dtype=bool)
    for rank in range(c.machines):
        idx = order[:, rank]
        keep = (rank < free) & (weights[np.arange(len(states)), idx] > 1e-9)
        chosen[np.arange(len(states)), idx] = keep
    return chosen


def reference(c, state, h, name):
    mask = reference_batch(c, np.asarray([state]), h, name)[0]
    return tuple(np.flatnonzero(mask).tolist())


def batch_tick(c, states, selected, shocks):
    following = states.copy()
    age, failed, wait, remaining = (following[:, :, i] for i in range(4))
    duration = np.asarray([row[0] for row in c.service_time])
    effect = np.asarray([row[0] for row in c.restoration])
    costs = (selected * np.where(failed, c.corrective_cost, c.preventive_cost)).sum(
        axis=1
    )
    remaining[:] = np.where(selected, duration, remaining)
    servicing = remaining > 0
    costs += ((failed > 0) | servicing).sum(axis=1) * c.unavailable_cost
    waiting = (failed > 0) & ~servicing
    wait[:] += waiting
    costs += (waiting & (wait > c.waiting_limit)).sum(axis=1) * c.waiting_price
    healthy = (failed == 0) & ~servicing
    age[:] = np.where(healthy, np.minimum(c.max_age, age + 1), age)
    new_failure = healthy & (age >= c.failure_age) & shocks
    failed[new_failure] = 1
    wait[new_failure] = 0
    costs += new_failure.sum(axis=1) * c.failure_cost
    remaining[:] -= servicing
    completed = servicing & (remaining == 0)
    age[:] = np.where(completed, np.floor(age * (1 - effect)).astype(int), age)
    failed[completed], wait[completed] = 0, 0
    return following, costs


def plan(c, state, h, continuation, scenario_seed, scenarios):
    candidates = actions(c, state)
    n_actions = len(candidates)
    # No environment object or its future events enter this function.
    rng = np.random.default_rng(scenario_seed)
    events = rng.random((h, scenarios, c.machines)) < c.failure_probability
    states = np.repeat(np.asarray([state]), n_actions * scenarios, axis=0)
    root = np.zeros((n_actions, c.machines), dtype=bool)
    for i, selected in enumerate(candidates):
        root[i, list(selected)] = True
    selected = np.repeat(root, scenarios, axis=0)
    totals = np.zeros(n_actions * scenarios)
    for t in range(h):
        if t:
            selected = reference_batch(c, states, h - t, continuation)
        states, costs = batch_tick(
            c, states, selected, np.tile(events[t], (n_actions, 1))
        )
        totals += costs
    totals = totals.reshape(n_actions, scenarios)
    means = totals.mean(axis=1)
    best = int(np.argmin(means))
    baseline = candidates.index(reference(c, state, h, continuation))
    se = totals.std(axis=1, ddof=1) / math.sqrt(scenarios)
    return candidates[best], dict(
        candidates=candidates,
        means=means.tolist(),
        mc_se=se.tolist(),
        selected_score=float(means[best]),
        baseline_score=float(means[baseline]),
        selected_mc_se=float(se[best]),
        score_spread=float(np.ptp(means)),
        paired_selected_baseline_mc_se=float(
            (totals[best] - totals[baseline]).std(ddof=1) / math.sqrt(scenarios)
        ),
        scenario_seed=scenario_seed,
    )
