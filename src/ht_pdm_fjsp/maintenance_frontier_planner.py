"""Vectorized root-action rollout with nested, non-oracle forecast budgets."""

import numpy as np

from ht_pdm_fjsp.maintenance_heterogeneous_model import (
    arrays,
    batch_tick,
    feasible,
    reference_batch,
)
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_solver_policy import CENTRAL
from ht_pdm_fjsp.maintenance_solver_model import reference


def forecasts(c, h, seed, budget, maximum):
    if not 2 <= budget <= maximum or not 1 <= h <= c.horizon:
        raise ValueError("forecast budget/horizon")
    # Scenario-major draw: smaller budgets are exact prefixes at every time step.
    return (
        np.random.default_rng(seed).random((budget, h, c.machines))
        < c.failure_probability
    ).transpose(1, 0, 2)


def score_rollouts(c, state, h, continuation, events):
    """All feasible root candidates; unchanged rule continuation and physics."""
    candidates = feasible(c, state)
    scenarios = events.shape[1]
    if events.shape != (h, scenarios, c.machines) or scenarios < 2:
        raise ValueError("forecast dimensions")
    roots = np.zeros((len(candidates), c.machines, c.technicians), dtype=bool)
    for i, pairs in enumerate(candidates):
        for m, j in pairs:
            roots[i, m, j] = True
    selected = np.repeat(roots, scenarios, axis=0)
    s = arrays(state, len(candidates) * scenarios)
    total = np.zeros(len(candidates) * scenarios)
    for t in range(h):
        if t:
            selected = reference_batch(c, s, h - t, continuation)
        s, cost = batch_tick(c, s, selected, np.tile(events[t], (len(candidates), 1)))
        total += cost
    return candidates, total.reshape(len(candidates), scenarios)


def plan(c, state, h, continuation, seed, budget, maximum):
    candidates, total = score_rollouts(
        c, state, h, continuation, forecasts(c, h, seed, budget, maximum)
    )
    means = total.mean(1)
    best = int(means.argmin())
    return candidates[best], dict(
        score_spread=float(np.ptp(means)),
        selected_mc_se=float(total[best].std(ddof=1) / np.sqrt(budget)),
        forecast_seed=seed,
        scenarios=budget,
        candidate_count=len(candidates),
    )


class Controller:
    """Same raw current-state interface; frontend and validity checks timed."""

    def __init__(self, name, rules, continuations, maximum, *, model=None, shock=0):
        self.name, self.rules, self.continuations = name, rules, continuations
        self.maximum, self.model, self.shock = maximum, model, shock
        self.condition = None

    def select(self, c, state, h):
        if self.name == CENTRAL:
            pairs, diag = self.model.select(c, state, h)
        elif self.name == "selected_rule":
            pairs, diag = reference(c, state, h, self.rules[self.condition]), {}
        else:
            budget = int(self.name.removeprefix("planner_"))
            seed = role_seed(self.shock, f"solver_frontier_v1_forecast:{c.horizon - h}")
            pairs, diag = plan(
                c,
                state,
                h,
                self.continuations[self.name][self.condition],
                seed,
                budget,
                self.maximum,
            )
        validate_matching(c, state, pairs)
        return pairs, diag
