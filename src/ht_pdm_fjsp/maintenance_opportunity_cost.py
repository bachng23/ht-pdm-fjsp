"""Locked non-learning diagnostic of technician opportunity costs.

Run smoke locally; full execution belongs to the user on the Ubuntu lab.
"""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_dispatch import (
    DispatchState, eligible_edges, role_seed, validate_matching, validate_state,
)
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, transition, ENV_VERSION
from ht_pdm_fjsp.maintenance_heterogeneous_model import (
    arrays, batch_tick, feasible, fixed_candidates, gain, plan, profile, reference,
)
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests

VERSION = "maintenance_opportunity_cost_v1"
ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("homogeneous", "specialized", "skill_mask")
RULES = ("risk_matching", "deadline_matching", "adaptive_isolated_matching",
         "scarcity_0", "scarcity_1", "scarcity_3", "scarcity_6")
CONTROLLERS = ("isolated", "scarcity", "best_rule", "fixed_rollout", "flexible_rollout")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        fields = list(rows[0]) if rows else ["status"]
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


class Journal:
    def __init__(self, path):
        self.stream = path.open("w", newline="")
        self.writer = None

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()

    def close(self):
        if self.writer is None:
            self.stream.write("status\n")
        self.stream.close()


def settings(execution_profile):
    if execution_profile not in ("smoke", "full"):
        raise ValueError("unknown execution profile")
    full = execution_profile == "full"
    return dict(
        profile=execution_profile, horizon=12 if full else 4,
        development_cohorts=list(range(9100000, 9100032)) if full else [9500000, 9500001],
        development_shocks=list(range(9200000, 9200008)) if full else [9600000, 9600001],
        test_cohorts=list(range(9300000, 9300064)) if full else [9500100],
        test_shocks=list(range(9400000, 9400032)) if full else [9600100, 9600101],
        scenarios=128 if full else 8, exact_state_cap=250000,
        exact_horizons=[6, 8] if full else [4],
        exact_a_ages=[3, 5] if full else [3],
        exact_b_ages=list(range(6)) if full else [0, 3],
        exact_probabilities=[.25, .6] if full else [.6],
        bootstrap_samples=20000 if full else 200,
        analysis_seed=9700000 if full else 9700001,
        conditions=list(CONDITIONS), controllers=list(CONTROLLERS), rules=list(RULES),
        train_seeds=[], training="NOT_APPLICABLE", terminal_value=0,
        discount=1, continuation="adaptive_isolated_matching",
        practical_relative_margin=.01, overdue_guard=.25, backlog_guard=.1,
    )


def make_config(cohort, condition, horizon, attempt=0):
    rng = random.Random(role_seed(cohort, f"{VERSION}:configuration:{attempt}"))
    base = [rng.randint(2, 5) for _ in range(4)]
    common = rng.randrange(2)
    exception = rng.randrange(4)
    threshold, probability = rng.choice((3, 4, 5)), rng.choice((.25, .45, .6))
    excluded = [m for m in range(4) if m != exception][:2]
    durations, effects = [], []
    for m in range(4):
        favorite = 1 - common if m == exception else common
        if condition == "homogeneous":
            durations.append((base[m], base[m]))
            effects.append((.75, .75))
        elif condition in ("specialized", "skill_mask"):
            ds = [base[m] if j == favorite else base[m] + 1 for j in range(2)]
            if condition == "skill_mask" and m in excluded:
                ds[1 - common] = 0
            durations.append(tuple(ds))
            effects.append(tuple(1. if j == favorite else .75 for j in range(2)))
        else:
            raise ValueError(condition)
    c = WaitingConfig(4, 2, horizon, threshold, probability,
                      tuple(durations), tuple(effects), waiting_price=12.)
    c.validate()
    return c


def canonical(c):
    """Physical signature invariant to arbitrary machine and worker relabeling."""
    rows = []
    for workers in itertools.permutations(range(c.technicians)):
        rows.append(tuple(sorted(tuple((c.service_time[m][j], c.restoration[m][j])
                                      for j in workers) for m in range(c.machines))))
    return (c.failure_age, c.failure_probability, c.horizon, min(rows))


def build_panels(cfg):
    seen = {cond: set() for cond in CONDITIONS}
    panels = {}
    registry = []
    for split in ("development", "test"):
        for cohort in cfg[f"{split}_cohorts"]:
            for attempt in range(10000):
                configs = {cond: make_config(cohort, cond, cfg["horizon"], attempt)
                           for cond in CONDITIONS}
                if all(canonical(c) not in seen[cond] for cond, c in configs.items()):
                    break
            else:
                raise RuntimeError("cannot construct unique profile panel")
            for cond, c in configs.items():
                seen[cond].add(canonical(c))
                panels[split, cohort, cond] = c
                registry.append(dict(split=split, cohort=cohort, condition=cond,
                                     generator_attempt=attempt, config=asdict(c)))
    return panels, registry


def seed_audit(cfg):
    panels = [set(cfg[k]) for k in ("development_cohorts", "development_shocks",
                                   "test_cohorts", "test_shocks")]
    if any(a & b for a, b in itertools.combinations(panels, 2)):
        raise ValueError("seed panels overlap")
    def integers(value):
        if isinstance(value, dict):
            return set().union(set(), *(integers(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(set(), *(integers(v) for v in value))
        return {value} if isinstance(value, int) and not isinstance(value, bool) else set()
    historical = set()
    hashes = {}
    for path in sorted((ROOT / "configs").glob("*seed_registry.json")):
        if path.name == f"{VERSION}_seed_registry.json":
            continue
        historical |= integers(json.loads(path.read_text()))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    if set.union(*panels) & historical:
        raise ValueError("historical seed overlap")
    return dict(status="PASS", historical_registry_hashes=hashes,
                scope="raw seed panels disjoint from inherited registries; derived seeds domain separated")


class Exact:
    """Finite stochastic DP using the original scalar transition contract."""
    def __init__(self, config, cap=250000):
        config.validate()
        self.c, self.cap, self.states = config, cap, 0
        self.value = lru_cache(maxsize=None)(self._value)

    def branches(self, state, action):
        validate_matching(self.c, state, action)
        servicing = set(state.assigned) | {m for m, _ in action}
        active = [m for m in range(self.c.machines) if m not in servicing
                  and not state.failed[m] and state.ages[m] + 1 >= self.c.failure_age]
        for bits in itertools.product((False, True), repeat=len(active)):
            events = [False] * self.c.machines
            probability = 1.
            for m, bit in zip(active, bits):
                events[m] = bit
                probability *= self.c.failure_probability if bit else 1 - self.c.failure_probability
            if probability:
                following, cost, _ = transition(self.c, state, action, events)
                yield probability, following, cost

    def q(self, state, action, horizon):
        if horizon < 1:
            raise ValueError("positive Q horizon required")
        return sum(p * (cost + self.value(following, horizon - 1))
                   for p, following, cost in self.branches(state, action))

    def _value(self, state, horizon):
        self.states += 1
        if self.states > self.cap:
            raise RuntimeError("exact DP state cap exceeded")
        if horizon == 0:
            return 0.
        return min(self.q(state, action, horizon) for action in feasible(self.c, state))


def scarcity_action(c, state, h, penalty):
    mask = eligible_edges(c, state)
    weights = np.zeros((c.machines, c.technicians))
    for m in range(c.machines):
        for j in range(c.technicians):
            if not mask[m][j]:
                continue
            external = 0.
            for other in range(c.machines):
                if other == m or not mask[other][j]:
                    continue
                risk = (1. if state.failed[other] else c.failure_probability
                        if state.ages[other] + 1 >= c.failure_age else 0.)
                external += risk / sum(mask[other])
            weights[m, j] = (gain(h, state.ages[m], int(state.failed[m]),
                                  state.pending_wait[m], j, profile(c, m))
                             - penalty * c.service_time[m][j] * external)
    actions = feasible(c, state)
    return max(actions, key=lambda pairs: sum(weights[m, j] for m, j in pairs))


def rule_action(c, state, h, name):
    if name.startswith("scarcity_"):
        return scarcity_action(c, state, h, float(name.split("_")[1]))
    return reference(c, state, h, name)


def select(c, state, h, controller, chosen, forecast_seed, scenarios):
    if controller in ("fixed_rollout", "flexible_rollout"):
        return plan(c, state, h, "adaptive_isolated_matching", forecast_seed,
                    scenarios, fixed=controller == "fixed_rollout")
    name = {"isolated": "adaptive_isolated_matching", "scarcity": chosen.get("scarcity"),
            "best_rule": chosen.get("best_rule")}.get(controller, controller)
    return rule_action(c, state, h, name), dict(candidates=(), scenario_seed=None)


def compact(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False)


def episode(c, cohort, shock, condition, controller, chosen, cfg, journals=None):
    env = WaitingEnv(c)
    env_seed = role_seed(shock, f"{VERSION}:physical:{cohort}")
    env.reset(env_seed)
    requests = Requests(env.state)
    identity = dict(cohort=cohort, shock=shock, condition=condition, controller=controller)
    latencies = []
    for t in range(c.horizon):
        before = env.state
        forecast_seed = role_seed(shock, f"{VERSION}:forecast:{cohort}:{t}")
        start = time.perf_counter_ns()
        action, diag = select(c, before, c.horizon - t, controller, chosen,
                              forecast_seed, cfg["scenarios"])
        validate_matching(c, before, action)
        latency = (time.perf_counter_ns() - start) / 1e6
        latencies.append(latency)
        _, _, _, info = env.step(action)
        # Alternate vectorized implementation verifies scalar dynamics each tick.
        selected = np.zeros((1, c.machines, c.technicians), dtype=bool)
        for m, j in action:
            selected[0, m, j] = True
        following, cost = batch_tick(c, arrays(before), selected, np.asarray([env.events[t]]))
        if not all(np.array_equal(following[k][0], getattr(env.state, k)) for k in following):
            raise RuntimeError("scalar/batch state replay mismatch")
        if not np.isclose(cost[0], info["objective"], atol=1e-8, rtol=0):
            raise RuntimeError("scalar/batch cost mismatch")
        requests.step(t, before, action, env.state)
        if journals:
            journals["decisions"].add({**identity, "time": t, "env_seed": env_seed,
                "forecast_seed": forecast_seed, "state": compact(asdict(before)),
                "action": compact(action), "following": compact(asdict(env.state)),
                "physical_metrics": compact(info)})
            journals["latency"].add({**identity, "time": t, "latency_ms": latency,
                "candidate_count": len(diag["candidates"]),
                "forecast_scenarios": cfg["scenarios"] if controller.endswith("rollout") else 0})
    demand = requests.finish(c.horizon)
    if journals:
        for request in demand:
            journals["requests"].add({**identity, **request,
                                     "overdue": request["wait"] > c.waiting_limit})
        journals["coordination"].add({**identity, "decisions": c.horizon,
                                     "invalid_actions": 0, "scalar_batch_replay": "PASS"})
    served = [r["wait"] for r in demand if r["served"]]
    return {**identity, **env.metrics,
        "cost_per_machine_tick": env.metrics["objective"] / (c.machines * c.horizon),
        "terminal_pending": sum(env.state.failed[m] and m not in env.state.assigned
                                for m in range(c.machines)),
        "terminal_failed": sum(env.state.failed),
        "terminal_servicing": sum(r > 0 for r in env.state.remaining),
        "requests": len(demand), "served_requests": len(served),
        "censored_requests": sum(r["censored"] for r in demand),
        "overdue_requests": sum(r["wait"] > c.waiting_limit for r in demand),
        "observed_wait_all_requests": sum(r["wait"] for r in demand),
        "served_wait_mean": float(np.mean(served)) if served else None,
        "served_wait_p95": float(np.quantile(served, .95, method="inverted_cdf")) if served else None,
        "latency_mean_ms": float(np.mean(latencies)),
        "latency_p50_ms": float(np.quantile(latencies, .5)),
        "latency_p95_ms": float(np.quantile(latencies, .95)),
        "latency_p99_ms": float(np.quantile(latencies, .99)),
        "eligible_edges": sum(d > 0 for row in c.service_time for d in row),
        "single_option_machines": sum(sum(d > 0 for d in row) == 1 for row in c.service_time),
        "sum_best_service_rates": sum(max(c.restoration[m][j] / d if d else 0.
                                            for j, d in enumerate(row))
                                      for m, row in enumerate(c.service_time)),
    }


def exact_diagnostic(cfg):
    rows = []
    grid = list(itertools.product(cfg["exact_probabilities"], cfg["exact_horizons"],
                                  cfg["exact_a_ages"], cfg["exact_b_ages"]))
    solvers = {}
    for probability, horizon, a_age, b_age in tqdm(grid, desc="Exact assignment values"):
        key = probability, horizon
        if key not in solvers:
            c = WaitingConfig(2, 2, horizon, 4, probability,
                              ((2, 3), (3, 0)), ((1., 1.), (1., 1.)), waiting_price=12.)
            solvers[key] = Exact(c, cfg["exact_state_cap"])
        solver = solvers[key]
        state = DispatchState((a_age, b_age), (True, False), (0, 0), (0, 0), (-1, -1))
        q = {action: solver.q(state, action, horizon) for action in feasible(solver.c, state)}
        best = min(q, key=q.get)
        fixed = min(fixed_candidates(solver.c, state), key=q.get)
        fast, reserve = q[((0, 0),)], q[((0, 1),)]
        rows.append(dict(probability=probability, horizon=horizon, a_age=a_age, b_age=b_age,
                         state=compact(asdict(state)), q_fast=fast, q_reserve=reserve,
                         fast_minus_reserve=fast-reserve, optimum=q[best],
                         best_action=compact(best), best_fixed_action=compact(fixed),
                         fixed_root_gap=q[fixed]-q[best], fast_regret=fast-q[best],
                         reserve_regret=reserve-q[best], value_states=solver.states))
    return rows


def select_development(rows):
    chosen = {}
    for condition in CONDITIONS:
        scores = {name: float(np.mean([r["objective"] for r in rows
                                      if r["condition"] == condition and r["controller"] == name]))
                  for name in RULES}
        chosen[condition] = dict(best_rule=min(RULES, key=scores.get),
                                 scarcity=min(RULES[3:], key=scores.get), scores=scores)
    return chosen


def paired_summary(rows, cfg):
    """Profile-cluster bootstrap conditional on the locked paired shock panel."""
    indexed = {(r["condition"], r["cohort"], r["shock"], r["controller"]): r for r in rows}
    expected = len(CONDITIONS) * len(cfg["test_cohorts"]) * len(cfg["test_shocks"]) * len(CONTROLLERS)
    if len(indexed) != len(rows) or len(rows) != expected:
        raise RuntimeError("test grid incomplete or duplicated")
    diffs, contrasts = [], {}
    for cond in CONDITIONS:
        for left in ("fixed_rollout", "best_rule", "isolated", "scarcity"):
            primary = cond == "skill_mask" and left in ("fixed_rollout", "best_rule")
            confidence = .975 if primary else .95
            profile_left, profile_right = [], []
            for cohort in cfg["test_cohorts"]:
                ls = [indexed[cond, cohort, s, left]["objective"] for s in cfg["test_shocks"]]
                rs = [indexed[cond, cohort, s, "flexible_rollout"]["objective"] for s in cfg["test_shocks"]]
                profile_left.append(np.mean(ls)); profile_right.append(np.mean(rs))
                diffs.append(dict(condition=cond, reference=left, cohort=cohort,
                                  reference_cost=float(np.mean(ls)), flexible_cost=float(np.mean(rs)),
                                  paired_difference=float(np.mean(np.subtract(ls, rs)))))
            a, b = np.asarray(profile_left), np.asarray(profile_right)
            rng = np.random.default_rng(role_seed(cfg["analysis_seed"], f"{cond}:{left}"))
            draws = rng.integers(len(a), size=(cfg["bootstrap_samples"], len(a)))
            changes = (a-b)[draws].mean(1)
            bases = a[draws].mean(1)
            relative = np.divide(changes, bases, out=np.zeros_like(changes), where=bases != 0)
            tail = (1-confidence)/2
            ci = np.quantile(changes, [tail, 1-tail]).tolist()
            relative_ci = np.quantile(relative, [tail, 1-tail]).tolist()
            contrasts[f"{cond}:{left}_minus_flexible"] = dict(
                primary=primary, confidence=confidence, profiles=len(a),
                mean_difference=float((a-b).mean()), confidence_interval=ci,
                relative_improvement=float((a-b).mean()/a.mean()) if a.mean() else 0.,
                relative_confidence_interval=relative_ci,
                profile_win_fraction=float(np.mean(a>b)),
                status=("ENGINEERING_ONLY" if cfg["profile"] == "smoke" else
                        "PASS_RESEARCH_MARGIN" if relative_ci[0] > cfg["practical_relative_margin"] else
                        "BELOW_RESEARCH_MARGIN" if relative_ci[1] < cfg["practical_relative_margin"] else
                        "INCONCLUSIVE"),
            )
    means = {cond: {name: {metric: float(np.mean([r[metric] for r in rows
                        if r["condition"] == cond and r["controller"] == name]))
                        for metric in ("objective", "base_cost", "maintenance_cost", "failure_cost",
                                       "unavailability_cost", "waiting_cost", "failures",
                                       "overdue_waiting_ticks", "terminal_pending", "terminal_failed",
                                       "terminal_servicing", "censored_requests", "served_requests",
                                       "observed_wait_all_requests", "latency_mean_ms")}
                    for name in CONTROLLERS} for cond in CONDITIONS}
    return dict(contrasts=contrasts, means=means,
                uncertainty_scope="profile bootstrap conditional on fixed shock panel"), diffs


def mechanism_summary(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["probability"], r["horizon"], r["a_age"]), []).append(r)
    reversals = []
    for key, group in groups.items():
        values = [r["fast_minus_reserve"] for r in group]
        if min(values) < -1e-8 and max(values) > 1e-8:
            reversals.append(dict(probability=key[0], horizon=key[1], a_age=key[2]))
    return dict(sign_reversal=bool(reversals), reversal_groups=reversals,
                interpretation="conditional exact assignment diagnostic, not state prevalence")


def service_latency_summary(output):
    """Pool physical decisions/requests, retaining censoring denominators."""
    latency = {(cond, name): [] for cond in CONDITIONS for name in CONTROLLERS}
    requests = {key: [] for key in latency}
    with (output / "latency.csv").open() as stream:
        for row in csv.DictReader(stream):
            latency[row["condition"], row["controller"]].append(float(row["latency_ms"]))
    with (output / "requests.csv").open() as stream:
        for row in csv.DictReader(stream):
            requests[row["condition"], row["controller"]].append(row)
    result = {}
    for (cond, name), samples in latency.items():
        demand = requests[cond, name]
        served = [int(r["wait"]) for r in demand if r["served"] == "True"]
        result[f"{cond}:{name}"] = dict(
            decision_count=len(samples),
            latency_p50_p95_p99_ms=np.quantile(samples, [.5, .95, .99]).tolist(),
            request_count=len(demand), served_count=len(served),
            censored_count=sum(r["censored"] == "True" for r in demand),
            observed_wait_all_requests=sum(int(r["wait"]) for r in demand),
            overdue_request_count=sum(r["overdue"] == "True" for r in demand),
            served_wait_mean=float(np.mean(served)) if served else None,
            served_wait_p95=float(np.quantile(served, .95, method="inverted_cdf")) if served else None,
        )
    return result


def run(output, execution_profile="smoke", device="cpu"):
    if device != "cpu":
        raise ValueError("CPU only")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("new/empty output directory required")
    output.mkdir(parents=True, exist_ok=True)
    cfg = settings(execution_profile)
    now = lambda: datetime.now(timezone.utc).isoformat()
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    manifest = dict(protocol=VERSION, status="RUNNING", started=now(), settings=cfg,
                    commit=git("rev-parse", "HEAD"), dirty=bool(git("status", "--porcelain")),
                    environment_version=ENV_VERSION, device="cpu", platform=platform.platform(),
                    machine=platform.machine(), python=platform.python_version(), numpy=np.__version__,
                    logical_cpus=os.cpu_count(), processor=platform.processor(),
                    thread_environment={k: os.environ.get(k) for k in
                                        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                    training="NOT_APPLICABLE", full_test_outcomes_opened=False)
    write_json(output / "manifest.json", manifest)
    journals = {}
    try:
        manifest["seed_audit"] = seed_audit(cfg)
        panels, registry = build_panels(cfg)
        write_json(output / "resolved_config.json", dict(settings=cfg, physical_profiles=registry))
        write_json(output / "seed_registry.json", {k: v for k, v in cfg.items() if "cohort" in k or "shock" in k})
        shutil.copyfile(ROOT / "docs" / f"{VERSION}_plan.md", output / "plan.md")
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for filename in ("maintenance_opportunity_cost.py", "maintenance_heterogeneous_model.py",
                         "maintenance_waiting.py", "maintenance_dispatch.py", "maintenance_solver_diagnostics.py"):
            shutil.copyfile(ROOT / "src/ht_pdm_fjsp" / filename, snapshot / filename)
        for filename in ("pyproject.toml", "uv.lock"):
            shutil.copyfile(ROOT / filename, snapshot / filename)
        exact = exact_diagnostic(cfg)
        write_csv(output / "action_values.csv", exact)
        development = []
        journals["development"] = Journal(output / "development_episodes.csv")
        grid = list(itertools.product(CONDITIONS, cfg["development_cohorts"], cfg["development_shocks"], RULES))
        for cond, cohort, shock, rule in tqdm(grid, desc="Development rule evaluation"):
            row = episode(panels["development", cohort, cond], cohort, shock, cond, rule, {}, cfg)
            development.append(row); journals["development"].add(row)
        chosen = select_development(development)
        write_json(output / "development_selection.json", dict(locked_before_test=True, selected=chosen))
        for name in ("episodes.partial", "decisions", "requests", "latency", "coordination"):
            journals[name] = Journal(output / f"{name}.csv")
        rows = []
        grid = list(itertools.product(CONDITIONS, cfg["test_cohorts"], cfg["test_shocks"], CONTROLLERS))
        # Only full execution (never tests/smoke) opens full test OUTCOMES.
        manifest["full_test_outcomes_opened"] = execution_profile == "full"
        write_json(output / "manifest.json", manifest)
        for cond, cohort, shock, controller in tqdm(grid, desc="Paired multi-seed evaluation"):
            row = episode(panels["test", cohort, cond], cohort, shock, cond, controller,
                          chosen[cond], cfg, journals)
            journals["episodes.partial"].add(row); rows.append(row)
        summary, differences = paired_summary(rows, cfg)
        summary["service_and_latency"] = service_latency_summary(output)
        summary["mechanism"] = mechanism_summary(exact)
        primary = [v for v in summary["contrasts"].values() if v["primary"]]
        means = summary["means"]["skill_mask"]
        guard = (means["flexible_rollout"]["overdue_waiting_ticks"] <=
                 means["best_rule"]["overdue_waiting_ticks"] + cfg["overdue_guard"]
                 and means["flexible_rollout"]["terminal_pending"] <=
                 means["best_rule"]["terminal_pending"] + cfg["backlog_guard"])
        summary["service_guard"] = dict(pass_descriptive_guard=guard,
                                         interpretation="research screen, not safety guarantee")
        summary["learning_investment_screen"] = ("ENGINEERING_ONLY" if execution_profile == "smoke" else
                "PASS" if all(v["status"] == "PASS_RESEARCH_MARGIN" for v in primary)
                and guard and summary["mechanism"]["sign_reversal"] else "NOT_ESTABLISHED")
        write_json(output / "summary.json", summary)
        write_csv(output / "profile_differences.csv", differences)
        write_csv(output / "episodes.csv", rows)
        manifest.update(status="COMPLETED", audit="PASS", actual_counts=dict(
            development_episodes=len(development), test_episodes=len(rows),
            test_decisions=len(rows)*cfg["horizon"], exact_states=len(exact)))
    except BaseException as error:
        manifest.update(status="FAILED", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for journal in journals.values():
            journal.close()
        manifest["finished"] = now()
        manifest["artifacts"] = {str(path.relative_to(output)): dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
        write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run(args.output_dir, args.profile, args.device)


if __name__ == "__main__":
    main()
