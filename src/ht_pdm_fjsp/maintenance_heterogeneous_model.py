"""Heterogeneous worker identities, strong matching references and CRN forecasts."""

from functools import lru_cache
import math
import random
import numpy as np
from ht_pdm_fjsp.maintenance_dispatch import role_seed, eligible_edges
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig

CONDITIONS = {
    "homogeneous": ("homogeneous", 2),
    "specialized": ("specialized", 2),
    "nominal": ("homogeneous", 4),
    "skill_mask": ("skill_mask", 2),
}
REFERENCES = (
    "risk_matching",
    "deadline_matching",
    "lookahead_matching",
    "adaptive_isolated_matching",
)


def cohort_config(seed, condition, horizon):
    regime, k = CONDITIONS[condition]
    rng = random.Random(role_seed(seed, "heterogeneous_configuration"))
    base = [rng.choice((2, 3)) for _ in range(4)]
    common = rng.randrange(2)
    exception = rng.randrange(4)
    favorite = [1 - common if m == exception else common for m in range(4)]
    duration = []
    effect = []
    excluded = [m for m in range(4) if favorite[m] == common][:2]
    for m in range(4):
        if regime == "homogeneous":
            duration.append((base[m],) * k)
            effect.append((0.75,) * k)
        else:
            ds = [base[m] - 1 if j == favorite[m] else base[m] + 1 for j in range(k)]
            if regime == "skill_mask" and m in excluded:
                ds[1 - common] = 0
            duration.append(tuple(ds))
            effect.append(tuple(1.0 if j == favorite[m] else 0.5 for j in range(k)))
    c = WaitingConfig(
        4, k, horizon, 3, 0.6, tuple(duration), tuple(effect), waiting_price=12.0
    )
    c.validate()
    return c


@lru_cache(maxsize=16)
def catalogue(n, k):
    result = []

    def visit(m, used, pairs):
        if m == n:
            result.append(pairs)
            return
        visit(m + 1, used, pairs)
        for j in range(k):
            if j not in used:
                visit(m + 1, used | {j}, (*pairs, (m, j)))

    visit(0, set(), ())
    return tuple(sorted(result))


@lru_cache(maxsize=16)
def membership(n, k):
    result = np.zeros((len(catalogue(n, k)), n, k), dtype=np.float32)
    for a, pairs in enumerate(catalogue(n, k)):
        for m, j in pairs:
            result[a, m, j] = 1
    return result


def feasible(c, state):
    edge = eligible_edges(c, state)
    return tuple(
        p for p in catalogue(c.machines, c.technicians) if all(edge[m][j] for m, j in p)
    )


def fixed_candidates(c, state):
    groups = {}
    for pairs in feasible(c, state):
        subset = tuple(m for m, j in pairs)
        score = sum(c.service_time[m][j] / c.restoration[m][j] for m, j in pairs)
        if subset not in groups or (score, pairs) < groups[subset]:
            groups[subset] = (score, pairs)
    return tuple(sorted(p for score, p in groups.values()))


def profile(c, m):
    return (
        c.service_time[m],
        c.restoration[m],
        c.max_age,
        c.failure_age,
        c.failure_probability,
        c.preventive_cost,
        c.corrective_cost,
        c.unavailable_cost,
        c.failure_cost,
        c.waiting_limit,
        c.waiting_price,
    )


@lru_cache(maxsize=500000)
def isolated(h, age, failed, wait, remaining, worker, args):
    if h == 0:
        return 0.0
    if remaining:
        return advance(h, age, failed, wait, remaining, worker, -1, args)
    return min(
        advance(h, age, failed, wait, 0, -1, start, args)
        for start in (-1, *[j for j, d in enumerate(args[0]) if d])
    )


def advance(h, age, failed, wait, remaining, worker, start, args):
    duration, effect, max_age, threshold, p, pm, cm, down, failure, limit, price = args
    if start >= 0:
        remaining = duration[start]
        worker = start
    cost = (cm if failed else pm) if start >= 0 else 0.0
    cost += down if failed or remaining else 0.0
    if remaining:
        r = remaining - 1
        next_state = (
            (math.floor(age * (1 - effect[worker])), 0, 0, 0, -1)
            if r == 0
            else (age, failed, wait, r, worker)
        )
        return cost + isolated(h - 1, *next_state, args)
    if failed:
        cost += price if wait + 1 > limit else 0.0
        return cost + isolated(h - 1, age, 1, wait + 1, 0, -1, args)
    a = min(max_age, age + 1)
    chance = p if a >= threshold else 0.0
    return (
        cost
        + chance * (failure + isolated(h - 1, a, 1, 0, 0, -1, args))
        + (1 - chance) * isolated(h - 1, a, 0, 0, 0, -1, args)
    )


@lru_cache(maxsize=200000)
def gain(h, age, failed, wait, j, args):
    return advance(h, age, failed, wait, 0, -1, -1, args) - advance(
        h, age, failed, wait, 0, -1, j, args
    )


def arrays(state, batch=1):
    return {
        k: np.repeat(
            np.asarray([getattr(state, k)], dtype=bool if k == "failed" else np.int64),
            batch,
            axis=0,
        )
        for k in ("ages", "failed", "pending_wait", "remaining", "assigned")
    }


def reference_batch(c, s, h, name):
    if name not in REFERENCES:
        raise ValueError(name)
    age, failed, wait = s["ages"], s["failed"], s["pending_wait"]
    b, n = age.shape
    k = c.technicians
    occupied = np.zeros((b, n), dtype=bool)
    for j in range(k):
        m = s["assigned"][:, j]
        occupied[np.arange(b)[m >= 0], m[m >= 0]] = True
    valid = (
        (~occupied)[:, :, None]
        & (s["remaining"] == 0)[:, None, :]
        & (np.asarray(c.service_time) > 0)[None]
    )
    weights = np.zeros((b, n, k))
    if name in ("risk_matching", "deadline_matching"):
        urgent = failed | (age >= c.failure_age - 1)
        hazard = ((age + 1 >= c.failure_age) & ~failed) * c.failure_probability
        urgency = 100 * failed + 10 * hazard + age / 8
        urgency += (
            20 * np.maximum(0, wait - 3)
            if name == "deadline_matching"
            else 10 * wait / 12
        )
        quality = np.asarray(c.restoration) / np.maximum(1, np.asarray(c.service_time))
        weights = urgency[:, :, None] + quality[None]
        valid = valid & urgent[:, :, None]
    else:
        horizon = min(h, 6) if name == "lookahead_matching" else h
        for m in range(n):
            states = np.column_stack([age[:, m], failed[:, m].astype(int), wait[:, m]])
            values, inverse = np.unique(states, axis=0, return_inverse=True)
            args = profile(c, m)
            for j in range(k):
                if c.service_time[m][j]:
                    gains = [gain(horizon, *map(int, row), j, args) for row in values]
                    weights[:, m, j] = np.asarray(gains)[inverse]
    weights = np.where(valid, weights, -np.inf)
    members = membership(n, k)
    score = np.einsum(
        "amj,bmj->ba",
        members,
        np.where(np.isfinite(weights), weights, 0),
        optimize=True,
    )
    bad = np.einsum("amj,bmj->ba", members, ~np.isfinite(weights), optimize=True) > 0
    index = np.where(bad, -np.inf, score).argmax(-1)
    return members[index].astype(bool)


def reference(c, state, h, name):
    chosen = reference_batch(c, arrays(state), h, name)[0]
    return tuple((int(m), int(j)) for m, j in zip(*np.nonzero(chosen)))


def batch_tick(c, s, selected, shocks):
    f = {k: v.copy() for k, v in s.items()}
    b, n = f["ages"].shape
    k = c.technicians
    ix = np.arange(b)
    failed_before = f["failed"].copy()
    starting = selected.any(2)
    cost = (
        starting * np.where(failed_before, c.corrective_cost, c.preventive_cost)
    ).sum(1)
    duration = np.asarray(c.service_time)
    effect = np.asarray(c.restoration)
    for j in range(k):
        start = selected[:, :, j].any(1)
        m = selected[:, :, j].argmax(1)
        f["assigned"][start, j] = m[start]
        f["remaining"][start, j] = duration[m[start], j]
    servicing = np.zeros((b, n), dtype=bool)
    for j in range(k):
        m = f["assigned"][:, j]
        servicing[ix[m >= 0], m[m >= 0]] = True
    cost += ((f["failed"] | servicing).sum(1)) * c.unavailable_cost
    waiting = f["failed"] & ~servicing
    f["pending_wait"] += waiting
    cost += (waiting & (f["pending_wait"] > c.waiting_limit)).sum(1) * c.waiting_price
    healthy = ~f["failed"] & ~servicing
    f["ages"] = np.where(healthy, np.minimum(c.max_age, f["ages"] + 1), f["ages"])
    new = healthy & (f["ages"] >= c.failure_age) & shocks
    f["failed"][new] = True
    f["pending_wait"][new] = 0
    cost += new.sum(1) * c.failure_cost
    for j in range(k):
        busy = f["remaining"][:, j] > 0
        f["remaining"][:, j] -= busy
        complete = busy & (f["remaining"][:, j] == 0)
        r = ix[complete]
        m = f["assigned"][complete, j]
        f["ages"][r, m] = np.floor(f["ages"][r, m] * (1 - effect[m, j])).astype(int)
        f["failed"][r, m] = False
        f["pending_wait"][r, m] = 0
        f["assigned"][complete, j] = -1
    return f, cost


def plan(c, state, h, continuation, scenario_seed, scenarios, fixed=False):
    candidates = fixed_candidates(c, state) if fixed else feasible(c, state)
    a = len(candidates)
    s = arrays(state, a * scenarios)
    roots = np.zeros((a, c.machines, c.technicians), dtype=bool)
    for i, pairs in enumerate(candidates):
        for m, j in pairs:
            roots[i, m, j] = True
    selected = np.repeat(roots, scenarios, 0)
    events = (
        np.random.default_rng(scenario_seed).random((h, scenarios, c.machines))
        < c.failure_probability
    )
    total = np.zeros(a * scenarios)
    for t in range(h):
        if t:
            selected = reference_batch(c, s, h - t, continuation)
        s, cost = batch_tick(c, s, selected, np.tile(events[t], (a, 1)))
        total += cost
    total = total.reshape(a, scenarios)
    means = total.mean(1)
    best = int(means.argmin())
    se = total.std(1, ddof=1) / math.sqrt(scenarios)
    return candidates[best], dict(
        candidates=candidates,
        means=means.tolist(),
        mc_se=se.tolist(),
        selected_score=float(means[best]),
        score_spread=float(np.ptp(means)),
        selected_mc_se=float(se[best]),
        scenario_seed=scenario_seed,
        restricted_allocation=fixed,
    )


def resolve(proposals, time):
    used = set()
    pairs = []
    rejected = 0
    n = len(proposals)
    for m in sorted(range(n), key=lambda m: (m - time) % n):
        j = int(proposals[m]) - 1
        if j < 0:
            continue
        if j in used:
            rejected += 1
        else:
            used.add(j)
            pairs.append((m, j))
    return tuple(sorted(pairs)), rejected


def contention(c, state):
    """Preference-graph deficit; treat equally capable workers as substitutes."""
    occupied = {m for m in state.assigned if m >= 0}
    preferred = np.zeros((c.machines, c.technicians), bool)
    busy_only = 0
    demand = 0
    for m in range(c.machines):
        if m in occupied or not (state.failed[m] or state.ages[m] >= c.failure_age - 1):
            continue
        demand += 1
        scores = [
            c.service_time[m][j] / c.restoration[m][j]
            if c.service_time[m][j]
            else math.inf
            for j in range(c.technicians)
        ]
        best = min(scores)
        choices = [j for j, v in enumerate(scores) if abs(v - best) < 1e-9]
        free = [j for j in choices if not state.remaining[j]]
        busy_only += not free
        preferred[m, free] = True
    members = membership(c.machines, c.technicians)
    invalid = np.einsum("amj,mj->a", members, ~preferred) > 0
    sizes = members.sum((1, 2))
    maximum = int(np.where(invalid, 0, sizes).max())
    return demand - maximum, busy_only
