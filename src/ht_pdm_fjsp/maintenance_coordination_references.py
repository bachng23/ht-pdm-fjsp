"""Strong, deterministic matching references; no future-shock access."""

import math
from functools import lru_cache

from ht_pdm_fjsp.maintenance_dispatch_policy import rule_action

REFERENCES = ("risk_skill_rule", "deadline_matching", "lookahead_matching")


def maximum_weight_matching(weights):
    """Optional partial matching, O(N K 2^K); lexicographic exact ties."""
    n = len(weights)
    k = len(weights[0]) if n else 0
    dp = {0: (0.0, ())}
    for m in range(n):
        following = dict(dp)
        for mask, (score, pairs) in dp.items():
            for j in range(k):
                gain = weights[m][j]
                if mask & (1 << j) or not math.isfinite(gain) or gain <= 0:
                    continue
                key = mask | (1 << j)
                candidate = score + gain, (*pairs, (m, j))
                previous = following.get(key)
                if (
                    previous is None
                    or candidate[0] > previous[0] + 1e-9
                    or (
                        abs(candidate[0] - previous[0]) <= 1e-9
                        and candidate[1] < previous[1]
                    )
                ):
                    following[key] = candidate
        dp = following
    return min(dp.values(), key=lambda row: (-round(row[0], 9), row[1]))[1]


@lru_cache(maxsize=200000)
def isolated_value(
    horizon,
    age,
    failed,
    wait,
    remaining,
    duration,
    restoration,
    failure_age,
    probability,
    pm,
    cm,
    downtime,
    failure_cost,
    limit,
    price,
):
    """Exact one-machine adaptive wait/start DP, worker always available in future."""
    if horizon == 0:
        return 0.0
    args = (
        duration,
        restoration,
        failure_age,
        probability,
        pm,
        cm,
        downtime,
        failure_cost,
        limit,
        price,
    )

    def advance(start):
        service = remaining or (duration if start else 0)
        current = (cm if failed else pm) if start else 0.0
        current += downtime if service or failed else 0.0
        current += price if failed and not service and wait + 1 > limit else 0.0
        if service:
            new_remaining = service - 1
            new_age = math.floor(age * (1 - restoration)) if new_remaining == 0 else age
            new_failed = False if new_remaining == 0 else failed
            new_wait = 0 if new_remaining == 0 else wait
            return current + isolated_value(
                horizon - 1, new_age, new_failed, new_wait, new_remaining, *args
            )
        if failed:
            return current + isolated_value(horizon - 1, age, True, wait + 1, 0, *args)
        new_age = min(8, age + 1)
        chance = probability if new_age >= failure_age else 0.0
        return (
            current
            + chance
            * (failure_cost + isolated_value(horizon - 1, new_age, True, 0, 0, *args))
            + (1 - chance) * isolated_value(horizon - 1, new_age, False, 0, 0, *args)
        )

    wait_cost = advance(False)
    return min(wait_cost, advance(True)) if remaining == 0 else wait_cost


def marginal_gain(obs, m, j):
    """Compare forcing start now with forcing wait now under the same isolated DP."""
    g, row, edge = obs["global_features"], obs["machines"][m], obs["edges"][m, j]
    horizon = min(6, round(float(g[0]) * 12))
    age, failed, wait = (
        round(float(row[0]) * 8),
        bool(row[1]),
        round(float(row[2]) * 12),
    )
    duration, restoration = round(float(edge[1]) * 12), round(float(edge[2]), 6)
    failure_age, probability = round(float(g[1]) * 8), round(float(g[2]), 6)
    pm, cm, downtime, failure_cost = (round(float(x) * 20, 6) for x in g[3:7])
    limit, price = round(float(g[7]) * 12), round(float(g[8]) * 20, 6)
    args = (
        duration,
        restoration,
        failure_age,
        probability,
        pm,
        cm,
        downtime,
        failure_cost,
        limit,
        price,
    )
    h = horizon - 1
    if h < 0:
        return 0.0
    if failed:
        defer = (
            downtime
            + (price if wait + 1 > limit else 0)
            + isolated_value(h, age, True, wait + 1, 0, *args)
        )
    else:
        new_age = min(8, age + 1)
        chance = probability if new_age >= failure_age else 0.0
        defer = chance * (
            failure_cost + isolated_value(h, new_age, True, 0, 0, *args)
        ) + (1 - chance) * isolated_value(h, new_age, False, 0, 0, *args)
    if duration == 1:
        start = (
            (cm if failed else pm)
            + downtime
            + isolated_value(h, math.floor(age * (1 - restoration)), False, 0, 0, *args)
        )
    else:
        start = (
            (cm if failed else pm)
            + downtime
            + isolated_value(h, age, failed, wait, duration - 1, *args)
        )
    return defer - start


class MatchingReference:
    def __init__(self, name):
        if name not in REFERENCES:
            raise ValueError(name)
        self.name = name
        self.last_diagnostics = {}

    def act(self, obs, mode=None, **kwargs):
        if self.name == "risk_skill_rule":
            pairs = rule_action(obs)
        else:
            n, k = len(obs["machines"]), len(obs["technicians"])
            weights = [[-math.inf] * k for _ in range(n)]
            threshold = float(obs["global_features"][1]) * 8
            for m, row in enumerate(obs["machines"]):
                for j, tech in enumerate(obs["technicians"]):
                    edge = obs["edges"][m, j]
                    if row[3] > 0.5 or tech[0] < 0.5 or edge[0] < 0.5:
                        continue
                    if self.name == "lookahead_matching":
                        weights[m][j] = marginal_gain(obs, m, j)
                    elif row[1] > 0.5 or row[0] * 8 >= threshold - 1 - 1e-5:
                        duration = max(1, round(float(edge[1]) * 12))
                        urgency = (
                            100 * float(row[1])
                            + 20 * max(0, float(row[2]) * 12 - 3)
                            + 10 * float(row[5])
                            + float(row[0])
                        )
                        weights[m][j] = urgency + float(edge[2]) / duration
            pairs = maximum_weight_matching(weights)
        return pairs, [*pairs, (-1, -1)], 0.0, 0.0
