import csv
from dataclasses import asdict, replace
import itertools
import hashlib
import json
import math

import numpy as np
import pytest

from ht_pdm_fjsp import maintenance_opportunity_cost as experiment
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, transition
from ht_pdm_fjsp.maintenance_heterogeneous_model import feasible, fixed_candidates, plan


def tiny_config(horizon=3, probability=.6):
    return WaitingConfig(2, 2, horizon, 2, probability,
                         ((1, 2), (2, 0)), ((1., 1.), (1., 1.)), waiting_price=12.)


def brute_value(c, state, horizon):
    if not horizon:
        return 0.
    def q(action):
        result = 0.
        for events in itertools.product((False, True), repeat=c.machines):
            p = np.prod([c.failure_probability if e else 1-c.failure_probability for e in events])
            following, cost, _ = transition(c, state, action, events)
            result += p * (cost + brute_value(c, following, horizon-1))
        return result
    return min(map(q, feasible(c, state)))


@pytest.mark.parametrize("probability", [0., .6, 1.])
def test_exact_matches_full_event_bruteforce(probability):
    c = tiny_config(probability=probability)
    state = DispatchState((2, 1), (True, False), (3, 0), (0, 0), (-1, -1))
    exact = experiment.Exact(c)
    assert exact.value(state, 3) == pytest.approx(brute_value(c, state, 3))
    assert sum(p for p, _, _ in exact.branches(state, ())) == pytest.approx(1)
    assert exact.value(state, 0) == 0
    assert min(exact.q(state, a, 3) for a in fixed_candidates(c, state)) >= exact.value(state, 3)-1e-8
    with pytest.raises(RuntimeError, match="cap"):
        experiment.Exact(c, cap=1).value(state, 3)


def test_exact_value_is_invariant_under_entity_relabeling():
    c = tiny_config()
    state = DispatchState((2, 1), (True, False), (0, 0), (0, 0), (-1, -1))
    swapped = replace(c, service_time=tuple(tuple(reversed(row)) for row in reversed(c.service_time)),
                      restoration=tuple(tuple(reversed(row)) for row in reversed(c.restoration)))
    s = replace(state, ages=state.ages[::-1], failed=state.failed[::-1])
    assert experiment.canonical(c) == experiment.canonical(swapped)
    assert experiment.Exact(c).value(state, 3) == pytest.approx(experiment.Exact(swapped).value(s, 3))


def test_largest_exact_horizon_on_one_engineering_state():
    # Engineering capacity check, not the full mechanism grid or sealed panel.
    c = WaitingConfig(2, 2, 8, 4, .6, ((2, 3), (3, 0)),
                      ((1., 1.), (1., 1.)), waiting_price=12.)
    state = DispatchState((4, 2), (True, False), (0, 0), (0, 0), (-1, -1))
    solver = experiment.Exact(c, cap=250000)
    assert math.isfinite(solver.value(state, 8))
    assert solver.states < 250000


def test_forecasts_share_scores_and_do_not_use_actual_future():
    c = tiny_config(horizon=4)
    env = WaitingEnv(c); env.reset(71)
    action, full = plan(c, env.state, 4, "adaptive_isolated_matching", 81, 16)
    fixed_action, fixed = plan(c, env.state, 4, "adaptive_isolated_matching", 81, 16, fixed=True)
    scores = dict(zip(full["candidates"], full["means"]))
    for a, q in zip(fixed["candidates"], fixed["means"]):
        assert q == pytest.approx(scores[a])
    assert full["selected_score"] <= fixed["selected_score"] + 1e-8
    env.events = tuple((True, True) for _ in range(c.horizon))
    action_after, replay = plan(c, env.state, 4, "adaptive_isolated_matching", 81, 16)
    assert action_after == action and replay == full
    validate_matching(c, env.state, fixed_action)
    for a in feasible(c, env.state):
        validate_matching(c, env.state, a)
    assert () in feasible(c, env.state)


def test_zero_scarcity_recovers_isolated_rule_and_busy_mask():
    c = tiny_config()
    env = WaitingEnv(c); env.reset(9)
    assert experiment.scarcity_action(c, env.state, 3, 0) == experiment.rule_action(
        c, env.state, 3, "adaptive_isolated_matching")
    occupied = DispatchState((1, 2), (False, True), (0, 0), (1, 0), (0, -1))
    for penalty in (0, 1, 3, 6):
        action = experiment.scarcity_action(c, occupied, 3, penalty)
        assert action == ()
        validate_matching(c, occupied, action)


def test_full_panels_only_audit_metadata_and_seeds_are_disjoint():
    cfg = experiment.settings("full")
    panels, registry = experiment.build_panels(cfg)
    assert len(registry) == 96*3
    for cond in experiment.CONDITIONS:
        dev = {experiment.canonical(c) for (split, _, co), c in panels.items()
               if split == "development" and co == cond}
        test = {experiment.canonical(c) for (split, _, co), c in panels.items()
                if split == "test" and co == cond}
        assert len(dev) == 32 and len(test) == 64 and not dev & test
    assert experiment.seed_audit(cfg)["status"] == "PASS"
    smoke = experiment.settings("smoke")
    for key in ("development_cohorts", "development_shocks", "test_cohorts", "test_shocks"):
        assert not set(cfg[key]) & set(smoke[key])


def test_profile_pairing_and_bootstrap_do_not_count_shocks_as_profiles():
    cfg = experiment.settings("smoke")
    cfg["test_cohorts"] = [11, 12]
    cfg["test_shocks"] = [21, 22]
    rows = []
    for cond, cohort, shock, controller in itertools.product(
            experiment.CONDITIONS, cfg["test_cohorts"], cfg["test_shocks"], experiment.CONTROLLERS):
        rows.append(dict(condition=cond, cohort=cohort, shock=shock, controller=controller,
            objective=90 if controller == "flexible_rollout" else 100,
            **{key: 0 for key in ("base_cost", "maintenance_cost", "failure_cost",
                "unavailability_cost", "waiting_cost", "failures", "overdue_waiting_ticks",
                "terminal_pending", "terminal_failed", "terminal_servicing", "censored_requests",
                "served_requests", "observed_wait_all_requests", "latency_mean_ms")}))
    summary, diffs = experiment.paired_summary(rows, cfg)
    primary = [v for v in summary["contrasts"].values() if v["primary"]]
    assert len(primary) == 2
    assert all(v["profiles"] == 2 and v["confidence"] == .975 for v in primary)
    assert all(v["mean_difference"] == 10 and v["relative_improvement"] == .1 for v in primary)
    assert all(v["status"] == "ENGINEERING_ONLY" for v in primary)
    assert len(diffs) == 3*4*2
    with pytest.raises(RuntimeError, match="grid"):
        experiment.paired_summary(rows + rows[:1], cfg)
    with pytest.raises(RuntimeError, match="grid"):
        experiment.paired_summary(rows[:-1], cfg)


def test_integrated_smoke_replay_censoring_hashes_and_immutability(tmp_path):
    out = tmp_path / "smoke"
    manifest = experiment.run(out)
    assert manifest["status"] == "COMPLETED" and manifest["audit"] == "PASS"
    assert not manifest["full_test_outcomes_opened"]
    assert manifest["actual_counts"] == dict(development_episodes=84, test_episodes=30,
                                             test_decisions=120, exact_states=2)
    for name, metadata in manifest["artifacts"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == metadata["sha256"]
    panels, _ = experiment.build_panels(experiment.settings("smoke"))
    for row in csv.DictReader((out / "decisions.csv").open()):
        c = panels["test", int(row["cohort"]), row["condition"]]
        state = DispatchState(**{k: tuple(v) for k, v in json.loads(row["state"]).items()})
        pairs = tuple(tuple(p) for p in json.loads(row["action"]))
        env = WaitingEnv(c); env.reset(int(row["env_seed"]))
        following, cost, info = transition(c, state, pairs, env.events[int(row["time"])])
        assert json.loads(json.dumps(asdict(following))) == json.loads(row["following"])
        assert info == json.loads(row["physical_metrics"])
    for row in csv.DictReader((out / "requests.csv").open()):
        assert int(row["wait"]) == int(row["end"]) - int(row["arrival"])
        assert (row["censored"] == "True") != (row["served"] == "True")
    assert json.loads((out / "summary.json").read_text())["learning_investment_screen"] == "ENGINEERING_ONLY"
    with pytest.raises(ValueError, match="new/empty"):
        experiment.run(out)


def test_failure_preserves_manifest_and_partial_output(tmp_path, monkeypatch):
    def crash(*args):
        raise RuntimeError("injected diagnostic failure")
    monkeypatch.setattr(experiment, "exact_diagnostic", crash)
    out = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        experiment.run(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "FAILED" and "injected" in manifest["error"]
    assert not manifest["full_test_outcomes_opened"]
    assert (out / "resolved_config.json").exists()


def test_interrupted_evaluation_preserves_flushed_episode_rows(tmp_path, monkeypatch):
    original = experiment.episode
    def interrupt(c, cohort, shock, condition, controller, *args):
        if controller == "flexible_rollout":
            raise KeyboardInterrupt("injected user interruption")
        return original(c, cohort, shock, condition, controller, *args)
    monkeypatch.setattr(experiment, "episode", interrupt)
    out = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        experiment.run(out)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["status"] == "FAILED" and "KeyboardInterrupt" in manifest["error"]
    partial = list(csv.DictReader((out / "episodes.partial.csv").open()))
    assert len(partial) == 4
    assert not (out / "episodes.csv").exists()
