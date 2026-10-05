"""Capped exact stochastic joint DP on the worker-symmetry quotient."""

import gzip
import itertools
import json
from functools import lru_cache

from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_contention_model import (
    REFERENCES,
    actions,
    branches,
    reference,
    relaxed_value,
    validate_model,
)


class StateCap(RuntimeError):
    pass


class JointOracle:
    def __init__(self, config, cap=250000, progress=True):
        validate_model(config)
        self.config, self.cap = config, cap
        self.nodes = {}
        self.residual = 0.0
        self.bar = tqdm(
            total=cap, desc="exact states (cap)", leave=False, disable=not progress
        )

    def value(self, h, state):
        if h == 0:
            return 0.0
        key = h, state
        if key in self.nodes:
            return self.nodes[key][0]
        if len(self.nodes) >= self.cap:
            raise StateCap(f"exact state cap {self.cap}")
        self.nodes[key] = None
        self.bar.update()
        choices = actions(self.config, state)
        q = tuple(
            sum(
                p * (cost + self.value(h - 1, following))
                for p, following, cost in branches(self.config, state, choice)
            )
            for choice in choices
        )
        value = min(q)
        self.nodes[key] = (value, choices, q)
        return value

    def evaluate(self, name, initials):
        c = self.config

        @lru_cache(maxsize=None)
        def visit(h, state):
            if not h:
                return (0.0, 0.0, 0.0)
            value, choices, q = self.nodes[h, state]
            selected = reference(c, state, h, name)
            regret = q[choices.index(selected)] - value
            early = c.horizon - h < c.horizon // 2
            total, first, last = 0.0, regret if early else 0.0, 0.0 if early else regret
            for p, following, cost in branches(c, state, selected):
                next_cost, e, late = visit(h - 1, following)
                total += p * (cost + next_cost)
                first += p * e
                last += p * late
            return total, first, last

        rows = [visit(c.horizon, state) for state in initials]
        result = tuple(sum(row[i] for row in rows) / len(rows) for i in range(3))
        visit.cache_clear()
        return result

    def save(self, path):
        """Partial nodes have null values, never mislabeled exact solutions."""
        with gzip.open(path, "wt") as stream:
            for (h, state), node in self.nodes.items():
                row = dict(
                    horizon_remaining=h,
                    state=state,
                    status="SOLVED" if node else "INCOMPLETE",
                )
                if node:
                    value, choices, q = node
                    row.update(value=value, actions=choices, q=q)
                    self.residual = max(self.residual, abs(value - min(q)))
                    for action, saved_q in zip(choices, q):
                        recomputed = 0.0
                        for p, following, cost in branches(self.config, state, action):
                            child = self.nodes.get((h - 1, following))
                            # A solved node always has solved descendants.
                            if h > 1 and child is None:
                                raise AssertionError("missing solved descendant")
                            recomputed += p * (cost + (child[0] if h > 1 else 0.0))
                        self.residual = max(self.residual, abs(saved_q - recomputed))
                stream.write(json.dumps(row) + "\n")


def run_exact(c, table_path, cap=250000, progress=True):
    oracle = JointOracle(c, cap, progress)
    initials = tuple(
        tuple((a, 0, 0, 0) for a in ages)
        for ages in itertools.product(range(c.failure_age), repeat=c.machines)
    )
    job = dict(
        status="RUNNING",
        horizon=c.horizon,
        machines=c.machines,
        technicians=c.technicians,
        initial_states=len(initials),
        cap=cap,
    )
    rows = []
    try:
        optimum = sum(oracle.value(c.horizon, s) for s in initials) / len(initials)
        lower = sum(relaxed_value(c, s, c.horizon) for s in initials) / len(initials)
        if lower > optimum + 1e-7:
            raise AssertionError("invalid capacity relaxation")
        if c.technicians >= c.machines and abs(optimum - lower) > 1e-7:
            raise AssertionError("unconstrained negative control")
        for name in REFERENCES:
            cost, early, late = oracle.evaluate(name, initials)
            if abs(cost - optimum - early - late) > 1e-7 or cost < optimum - 1e-7:
                raise AssertionError("regret decomposition")
            if (
                c.technicians >= c.machines
                and name == "independent_dp_matching"
                and abs(cost - optimum) > 1e-7
            ):
                raise AssertionError("independent policy negative control")
            rows.append(
                dict(
                    controller=name,
                    priced_cost=cost,
                    joint_optimum=optimum,
                    relaxed_lower_bound=lower,
                    unavoidable_scarcity_cost=optimum - lower,
                    avoidable_decision_cost=cost - optimum,
                    priority_regret=early + late,
                    allocation_regret=0.0,
                    early_regret=early,
                    late_regret=late,
                    objective="base_plus_overdue_price",
                )
            )
        job.update(status="COMPLETED", joint_optimum=optimum, relaxed_lower_bound=lower)
    except StateCap as exc:
        job.update(status="CAPPED", reason=str(exc))
    finally:
        oracle.bar.close()
        oracle.save(table_path)
    job.update(
        states=len(oracle.nodes),
        solved_states=sum(x is not None for x in oracle.nodes.values()),
        bellman_residual=oracle.residual,
    )
    return job, rows
