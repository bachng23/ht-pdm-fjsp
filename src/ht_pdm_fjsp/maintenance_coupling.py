"""Frozen model-based timing/assignment experiment. See locked protocol in docs."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import math
import platform
import random
import statistics
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

PROTOCOL = "maintenance_coupling_v1"
POLICIES = ("A_health_fastest", "B_health_queue_assignment", "C_joint_lookahead", "D_fixed_timing_lookahead")
CELLS = tuple(itertools.product(("low", "high"), ("uniform", "heterogeneous")))
COMPONENTS = ("maintenance", "planned_down", "failed_down", "failure", "queue")


@dataclass(frozen=True)
class Config:
    service: tuple[tuple[int, ...], ...]
    initial_ages: tuple[int, ...] = (0, 1)
    failure_age: int = 4
    probability: float = .35
    horizon: int = 12
    maintenance_cost: float = 2.
    planned_cost: float = 1.
    failed_cost: float = 8.
    failure_cost: float = 15.
    waiting_cost: float = .5
    request_order: tuple[int, ...] = (0, 1)

    def __post_init__(self):
        n, k = len(self.service), len(self.service[0])
        assert n and k and len(self.initial_ages) == n
        assert all(len(row) == k and all(isinstance(d, int) and d >= 1 for d in row) for row in self.service)
        assert sorted(self.request_order) == list(range(n))
        assert 0 <= self.probability <= 1 and self.horizon > 0
        assert all(0 <= a < self.failure_age for a in self.initial_ages)
        assert all(getattr(self, key) >= 0 for key in ("maintenance_cost", "planned_cost", "failed_cost", "failure_cost", "waiting_cost"))


@dataclass(frozen=True)
class State:
    ages: tuple[int, ...]
    failed: tuple[bool, ...]
    remaining: tuple[int, ...]
    assigned: tuple[int, ...]
    queues: tuple[tuple[int, ...], ...]


def initial(cfg):
    k = len(cfg.service[0])
    return State(cfg.initial_ages, (False,) * len(cfg.service), (0,) * k, (-1,) * k, ((),) * k)


def audit(cfg, s):
    members = [m for m in s.assigned if m >= 0] + [m for q in s.queues for m in q]
    assert len(members) == len(set(members)) and all(0 <= m < len(cfg.service) for m in members)
    assert all(r >= 0 and (r > 0) == (m >= 0) for r, m in zip(s.remaining, s.assigned))
    assert all(a >= 0 for a in s.ages)


def options(cfg, s):
    occupied = set(s.assigned) | {m for q in s.queues for m in q}
    return tuple((0,) if m in occupied else tuple(range(len(cfg.service[0]) + 1)) for m in range(len(cfg.service)))


def transition(cfg, s, action, events):
    """One step: requests -> dispatch -> down/operation -> completion/reset."""
    assert len(action) == len(s.ages) and len(events) == len(s.ages)
    assert all(a in opt for a, opt in zip(action, options(cfg, s)))
    rem, assigned, queues = list(s.remaining), list(s.assigned), [list(q) for q in s.queues]
    ages, failed = list(s.ages), list(s.failed)
    costs = dict.fromkeys(COMPONENTS, 0.)
    metrics = dict.fromkeys(("failures", "waiting", "planned_steps", "failed_steps", "jobs", "busy_requests", "invalid_requests"), 0)
    for m in cfg.request_order:
        if action[m]:
            t = action[m] - 1
            metrics["busy_requests"] += int(rem[t] > 0)
            queues[t].append(m)
    for t, queue in enumerate(queues):
        if rem[t] == 0 and queue:
            m = queue.pop(0)
            assigned[t], rem[t] = m, cfg.service[m][t]
            metrics["jobs"] += 1
            costs["maintenance"] += cfg.maintenance_cost
    servicing = {m for m in assigned if m >= 0}
    metrics["waiting"] = sum(map(len, queues))
    costs["queue"] = metrics["waiting"] * cfg.waiting_cost
    for m in range(len(ages)):
        if m in servicing:
            metrics["planned_steps"] += 1
            costs["planned_down"] += cfg.planned_cost
        elif failed[m]:
            metrics["failed_steps"] += 1
            costs["failed_down"] += cfg.failed_cost
        else:
            ages[m] = min(cfg.failure_age, ages[m] + 1)
            if ages[m] >= cfg.failure_age and events[m]:
                failed[m] = True
                metrics["failures"] += 1
                costs["failure"] += cfg.failure_cost
    for t, m in enumerate(assigned):
        if m >= 0:
            rem[t] -= 1
            if rem[t] == 0:
                ages[m], failed[m], assigned[t] = 0, False, -1
    ns = State(tuple(ages), tuple(failed), tuple(rem), tuple(assigned), tuple(tuple(q) for q in queues))
    audit(cfg, ns)
    return ns, sum(costs.values()), {**metrics, **{"cost_" + k: v for k, v in costs.items()}}


def closeout(cfg, s):
    """Explicit finite-horizon outstanding-work accounting, not future shock access."""
    cost = sum(s.remaining) * cfg.planned_cost
    served = {m for m in s.assigned if m >= 0}
    for t, queue in enumerate(s.queues):
        delay = s.remaining[t]
        for m in queue:
            served.add(m)
            cost += cfg.maintenance_cost + cfg.service[m][t] * cfg.planned_cost
            cost += delay * (cfg.waiting_cost + (cfg.failed_cost if s.failed[m] else 0.))
            delay += cfg.service[m][t]
    cost += sum(cfg.failure_cost for m, f in enumerate(s.failed) if f and m not in served)
    return cost


def scenario(seed, pressure, skill, horizon):
    rng = random.Random(seed)
    fast = rng.choice((1, 2))
    preference = rng.choice((0, 1))
    k = 2 if pressure == "high" else 4
    rows = []
    for m in range(2):
        favored = 0 if m == 0 else preference
        row = tuple(fast if t % 2 == favored else fast + 2 for t in range(k))
        rows.append(row if skill == "heterogeneous" else (fast + 1,) * k)
    age_limit = rng.choice((3, 4, 5))
    return Config(tuple(rows), (rng.randrange(age_limit - 1), rng.randrange(age_limit - 1)), age_limit,
                  rng.choice((.25, .4, .6)), horizon, failure_cost=rng.choice((12., 18., 24.)))


class Solver:
    def __init__(self, cfg, offset=1, depth=3, tail=2, max_states=500_000):
        self.cfg, self.offset, self.depth, self.tail, self.max_states = cfg, offset, depth, tail, max_states
        # Caches belong to this solver instance; released between cells.
        self.base_value = lru_cache(None)(self._base_value)
        self.value = lru_cache(None)(self._value)
        self.oracle_value = lru_cache(None)(self._oracle_value)
        self.fixed_value = lru_cache(None)(self._fixed_value)

    def guard(self):
        count = sum(f.cache_info().currsize for f in (self.base_value, self.value, self.oracle_value, self.fixed_value))
        if count >= self.max_states:
            raise RuntimeError(f"Solver cache budget {self.max_states} exceeded")

    def triggered(self, s):
        return tuple(m for m, opt in enumerate(options(self.cfg, s)) if len(opt) > 1 and
                     (s.failed[m] or s.ages[m] >= max(0, self.cfg.failure_age - self.offset)))

    def baseline(self, s, policy):
        triggered = self.triggered(s)
        action = [0] * len(s.ages)
        if policy == POLICIES[0]:
            for m in triggered:
                action[m] = min(range(len(s.remaining)), key=lambda t: (self.cfg.service[m][t], t)) + 1
            return tuple(action)
        assert policy == POLICIES[1]
        candidates = itertools.product(*(tuple(range(1, len(s.remaining) + 1)) if m in triggered else (0,) for m in range(len(s.ages))))
        def completion(a):
            delays = [s.remaining[t] + sum(self.cfg.service[m][t] for m in q) for t, q in enumerate(s.queues)]
            score = 0
            for m in self.cfg.request_order:
                if a[m]:
                    t = a[m] - 1
                    delays[t] += self.cfg.service[m][t]
                    score += delays[t]
            return score
        return min(candidates, key=lambda a: (completion(a), a))

    def branches(self, s, a):
        # Hazard-active machines after dispatch; skip irrelevant shock combinations.
        # Compute directly from pre-step dispatch to avoid completion-edge ambiguity.
        servicing = {m for m in s.assigned if m >= 0}
        qs = [list(q) for q in s.queues]
        for m in self.cfg.request_order:
            if a[m]: qs[a[m] - 1].append(m)
        for t, q in enumerate(qs):
            if s.remaining[t] == 0 and q: servicing.add(q[0])
        active = [m for m in range(len(s.ages)) if not s.failed[m] and m not in servicing and s.ages[m] + 1 >= self.cfg.failure_age]
        total = 0.
        for bits in itertools.product((False, True), repeat=len(active)):
            p = math.prod(self.cfg.probability if b else 1 - self.cfg.probability for b in bits)
            if p == 0: continue
            events = [False] * len(s.ages)
            for m, b in zip(active, bits): events[m] = b
            ns, cost, _ = transition(self.cfg, s, a, tuple(events))
            total += p
            yield p, ns, cost
        assert math.isclose(total, 1., abs_tol=1e-12)

    def expectation(self, s, a, continuation):
        return sum(p * (cost + continuation(ns)) for p, ns, cost in self.branches(s, a))

    def _base_value(self, s, left):
        self.guard()
        if left == 0: return closeout(self.cfg, s)
        return self.expectation(s, self.baseline(s, POLICIES[1]), lambda ns: self.base_value(ns, left - 1))

    def _value(self, s, depth, tail):
        self.guard()
        if depth == 0: return self.base_value(s, tail)
        return min(self.expectation(s, a, lambda ns: self.value(ns, depth - 1, tail)) for a in itertools.product(*options(self.cfg, s)))

    def choose(self, s, left):
        d = min(self.depth, left)
        tail = min(self.tail, left - d)
        return min(itertools.product(*options(self.cfg, s)), key=lambda a: (
            self.expectation(s, a, lambda ns: self.value(ns, d - 1, tail)), a))

    def fixed_actions(self, s):
        triggered = self.triggered(s)
        return itertools.product(*(tuple(range(1, len(s.remaining) + 1)) if m in triggered else (0,) for m in range(len(s.ages))))

    def _fixed_value(self, s, depth, tail):
        self.guard()
        if depth == 0: return self.base_value(s, tail)
        return min(self.expectation(s, a, lambda ns: self.fixed_value(ns, depth - 1, tail)) for a in self.fixed_actions(s))

    def choose_fixed(self, s, left):
        d = min(self.depth, left)
        tail = min(self.tail, left - d)
        return min(self.fixed_actions(s), key=lambda a: (
            self.expectation(s, a, lambda ns: self.fixed_value(ns, d - 1, tail)), a))

    def action(self, s, left, policy):
        if policy == POLICIES[2]: return self.choose(s, left)
        if policy == POLICIES[3]: return self.choose_fixed(s, left)
        return self.baseline(s, policy)

    def _oracle_value(self, s, left):
        self.guard()
        if left == 0: return closeout(self.cfg, s)
        return min(self.expectation(s, a, lambda ns: self.oracle_value(ns, left - 1)) for a in itertools.product(*options(self.cfg, s)))

    def exact_policy_cost(self, policy, horizon):
        @lru_cache(None)
        def ev(s, left):
            if left == 0: return closeout(self.cfg, s)
            a = self.action(s, left, policy)
            return self.expectation(s, a, lambda ns: ev(ns, left - 1))
        return ev(initial(self.cfg), horizon)


def uniforms(seed, step, n):
    return tuple(random.Random(int.from_bytes(hashlib.sha256(f"{seed}:{step}:{m}".encode()).digest()[:8], "big")).random() for m in range(n))


def episode(solver, policy, seed, trace=False):
    cfg, s = solver.cfg, initial(solver.cfg)
    totals = {"cost_" + k: 0. for k in COMPONENTS}
    totals.update(dict.fromkeys(("failures", "waiting", "planned_steps", "failed_steps", "jobs", "busy_requests", "invalid_requests"), 0.))
    elapsed, traces = 0., []
    for step in range(cfg.horizon):
        start = time.perf_counter()
        a = solver.action(s, cfg.horizon - step, policy)
        elapsed += time.perf_counter() - start
        ns, _, info = transition(cfg, s, a, tuple(u < cfg.probability for u in uniforms(seed, step, len(s.ages))))
        if trace:
            traces.append({"step": step, "state": json.dumps(asdict(s)), "action": json.dumps(a), "next_state": json.dumps(asdict(ns)), **info})
        for key, value in info.items(): totals[key] += value
        s = ns
    stage = sum(totals["cost_" + k] for k in COMPONENTS)
    terminal = closeout(cfg, s)
    return {**totals, "stage_cost": stage, "terminal_cost": terminal, "cost": stage + terminal,
            "decision_seconds": elapsed, "technician_utilization": totals["planned_steps"] / (cfg.horizon * len(s.remaining)), "terminal_share": terminal / (stage + terminal) if stage + terminal else 0.}, traces


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def git(*args):
    return subprocess.check_output(("git", *args), text=True).strip()


def panels(profile):
    if profile == "smoke":
        return {"dev_configs": list(range(770100, 770102)), "dev_episodes": list(range(771100, 771103)),
                "configs": [770200], "episodes": list(range(771200, 771203)), "horizon": 5, "depth": 2, "tail": 1, "oracle_horizon": 3}
    return {"dev_configs": list(range(780100, 780103)), "dev_episodes": list(range(781100, 781108)),
            "configs": list(range(780200, 780212)), "episodes": list(range(781200, 781240)),
            "horizon": 12, "depth": 3, "tail": 2, "oracle_horizon": 4}


def bootstrap(values, seed=782100, draws=10000):
    if len(values) < 2: return {"mean": statistics.mean(values), "ci95": None, "n_instances": len(values)}
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(draws))
    return {"mean": statistics.mean(values), "ci95": [means[int(.025 * draws)], means[int(.975 * draws)]], "n_instances": len(values)}


def summarize(rows, profile):
    grouped = {}
    for r in rows:
        key = (r["instance_seed"], r["pressure"], r["skill"], r["policy"])
        grouped.setdefault(key, []).append(r["cost"])
    means = {key: statistics.mean(v) for key, v in grouped.items()}
    instances = sorted({r["instance_seed"] for r in rows})
    deltas, cells = {}, {}
    for p, skill in CELLS:
        savings = [means[i, p, skill, POLICIES[1]] - means[i, p, skill, POLICIES[2]] for i in instances]
        deltas[p, skill] = savings
        relatives = [d / means[i, p, skill, POLICIES[1]] for d, i in zip(savings, instances) if means[i, p, skill, POLICIES[1]] > 0]
        cells[p + "_" + skill] = {"B_minus_C": bootstrap(savings), "relative_savings": statistics.mean(relatives) if relatives else None,
                                  "relative_valid_instances": len(relatives), "A_minus_B": bootstrap([means[i,p,skill,POLICIES[0]] - means[i,p,skill,POLICIES[1]] for i in instances]),
                                  "D_minus_C": bootstrap([means[i,p,skill,POLICIES[3]] - means[i,p,skill,POLICIES[2]] for i in instances]),
                                  "cost_means": {policy: statistics.mean(means[i,p,skill,policy] for i in instances) for policy in POLICIES}}
    interaction = bootstrap([deltas["high", "heterogeneous"][j] - deltas["high", "uniform"][j] -
                             deltas["low", "heterogeneous"][j] + deltas["low", "uniform"][j] for j in range(len(instances))])
    high = cells["high_heterogeneous"]
    ci = high["B_minus_C"]["ci95"]
    primary = profile == "full" and ci is not None and ci[0] > 0 and high["relative_valid_instances"] == len(instances) and high["relative_savings"] >= .05
    return {"cells": cells, "interaction": interaction, "primary_gate": bool(primary),
            "stronger_mechanism_gate": bool(primary and interaction["ci95"] and interaction["ci95"][0] > 0 and high["D_minus_C"]["ci95"] and high["D_minus_C"]["ci95"][0] > 0),
            "scientific_status": "ENGINEERING_ONLY" if profile == "smoke" else "CONFIRMATORY_DIAGNOSTIC",
            "inference_unit": "independent instance means; episodes are paired repeated shocks, not independent instances"}


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError("Output directory must be empty; never overwrite a run")
    panel = panels(args.profile)
    source = Path(__file__).read_bytes()
    manifest = {"protocol": PROTOCOL, "profile": args.profile, "status": "RUNNING", "source_commit": git("rev-parse", "HEAD"),
                "source_dirty": bool(git("status", "--porcelain")), "source_sha256": hashlib.sha256(source).hexdigest(),
                "protocol_sha256": hashlib.sha256(Path("docs/maintenance_coupling_v1_plan.md").read_bytes()).hexdigest(),
                "python": platform.python_version(), "platform": platform.platform(), "device": args.device,
                "panel": panel, "full_panel_opened": args.profile == "full", "training": "not applicable: model-based policies",
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(out / "manifest.json", manifest)
    try:
        if args.profile == "full" and manifest["source_dirty"]:
            raise RuntimeError("Full experiment requires clean source checkout")
        write_json(out / "benchmark_config.json", {"model": "completion_recovery_fifo_v1", "protocol": PROTOCOL, "cells": CELLS, "cost_components": COMPONENTS})
        configs = []
        for seed in panel["configs"]:
            for pressure, skill in CELLS:
                cfg = scenario(seed, pressure, skill, panel["horizon"])
                configs.append({"instance_seed": seed, "pressure": pressure, "skill": skill, "config": asdict(cfg)})
        write_json(out / "resolved_config.json", configs)
        dev = []
        for offset, seed, (pressure, skill) in tqdm(list(itertools.product((0,1,2), panel["dev_configs"], CELLS)), desc="development threshold selection", unit="cell"):
            solver = Solver(scenario(seed, pressure, skill, panel["horizon"]), offset)
            for e in panel["dev_episodes"]:
                metrics, _ = episode(solver, POLICIES[1], e)
                dev.append({"offset": offset, "instance_seed": seed, "pressure": pressure, "skill": skill, "episode_seed": e, **metrics})
        write_csv(out / "development.csv", dev)
        offset = min((0,1,2), key=lambda o: (statistics.mean(r["cost"] for r in dev if r["offset"] == o), o))
        policy_data = {"protocol": PROTOCOL, "offset": offset, "depth": panel["depth"], "tail": panel["tail"], "threshold_selection": "B development costs only"}
        write_json(out / "policy.json", policy_data)
        loaded = json.loads((out / "policy.json").read_text())
        assert loaded == policy_data
        rows, traces, coordination, oracle = [], [], [], []
        for seed, (pressure, skill) in tqdm(list(itertools.product(panel["configs"], CELLS)), desc="evaluate paired instances", unit="cell"):
            cfg = scenario(seed, pressure, skill, panel["horizon"])
            solver = Solver(cfg, loaded["offset"], loaded["depth"], loaded["tail"])
            for e in tqdm(panel["episodes"], desc=f"{seed}/{pressure}/{skill}", unit="seed", leave=False):
                for policy in POLICIES:
                    metrics, trace_rows = episode(solver, policy, e, trace=e == panel["episodes"][0])
                    ident = {"instance_seed": seed, "pressure": pressure, "skill": skill, "episode_seed": e, "policy": policy}
                    rows.append({**ident, **metrics})
                    traces.extend({**ident, **t} for t in trace_rows)
            coordination.append({"instance_seed": seed, "pressure": pressure, "skill": skill, "invalid_requests": 0, "state_audit": "PASS", "probability_audit": "PASS"})
            write_csv(out / "episodes.partial.csv", rows)
            write_csv(out / "decisions.csv", traces)
            del solver
            gc.collect()
        for seed, (pressure, skill) in tqdm(list(itertools.product(panel["configs"][:2], CELLS)), desc="tiny exact oracle", unit="cell"):
            cfg = scenario(seed, pressure, skill, panel["oracle_horizon"])
            solver = Solver(cfg, offset, panel["depth"], panel["tail"])
            optimal = solver.oracle_value(initial(cfg), cfg.horizon)
            for policy in POLICIES:
                value = solver.exact_policy_cost(policy, cfg.horizon)
                assert value >= optimal - 1e-8
                oracle.append({"instance_seed": seed, "pressure": pressure, "skill": skill, "horizon": cfg.horizon,
                               "policy": policy, "exact_cost": value, "optimal_cost": optimal, "gap": value - optimal})
            write_csv(out / "exact_oracle.csv", oracle)
        expected = len(panel["configs"]) * 4 * len(panel["episodes"]) * len(POLICIES)
        keys = {(r["instance_seed"], r["pressure"], r["skill"], r["episode_seed"], r["policy"]) for r in rows}
        assert len(keys) == len(rows) == expected
        assert all(math.isclose(r["cost"], r["terminal_cost"] + sum(r["cost_" + k] for k in COMPONENTS), abs_tol=1e-8) for r in rows)
        assert all(math.isfinite(r["cost"]) and r["invalid_requests"] == 0 for r in rows)
        for seed in panel["configs"]:
            avgs = {tuple(statistics.mean(row) for row in scenario(seed,p,k,panel["horizon"]).service) for p,k in CELLS}
            assert len(avgs) == 1
        assert not (set(panel["configs"]) & set(panel["dev_configs"]))
        assert not (set(panel["episodes"]) & set(panel["dev_episodes"]))
        write_csv(out / "episodes.csv", rows)
        write_csv(out / "coordination.csv", coordination)
        summary = summarize(rows, args.profile)
        summary["audits"] = {"expected_rows": expected, "actual_rows": len(rows), "unique_rows": True, "cost_reconciles": True,
                             "matched_service_means": True, "sealed_panel_closed": args.profile == "smoke", "all_passed": True}
        write_json(out / "summary.json", summary)
        manifest.update(status="COMPLETED", finished_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        outputs=sorted(p.name for p in out.iterdir()))
        write_json(out / "manifest.json", manifest)
        print(json.dumps({"status": "COMPLETED", "episodes": len(rows), "offset": offset,
                          "primary_gate": summary["primary_gate"], "scientific_status": summary["scientific_status"]}))
    except BaseException:
        manifest.update(status="FAILED", error=traceback.format_exc())
        write_json(out / "manifest.json", manifest)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
