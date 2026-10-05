import gzip
import itertools
import json
from dataclasses import replace

import numpy as np
import pytest

from ht_pdm_fjsp import maintenance_contention_model as model
from ht_pdm_fjsp.maintenance_contention_headroom import episode, settings, summarize
from ht_pdm_fjsp.maintenance_contention_oracle import JointOracle, run_exact
from ht_pdm_fjsp.maintenance_coordination_references import MatchingReference
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, matching_actions, role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, observation, transition


@pytest.mark.parametrize("capacity", model.CAPACITIES)
def test_scalar_vector_and_original_physics(capacity):
    c = model.cohort_config(137000, capacity, 24)
    env = WaitingEnv(c)
    rng = np.random.default_rng(53)
    env.reset(47)
    for t in range(24):
        s = env.state
        small = model.compact(s)
        choices = model.actions(c, small)
        original_choices = matching_actions(c, s)
        assert set(choices) == {
            tuple(sorted(m for m, _ in a)) for a in original_choices
        }
        for selected in choices:
            pairs = model.physical_action(s, selected)
            for shocks in itertools.product((False, True), repeat=4):
                actual, cost, _ = transition(c, s, pairs, shocks)
                following, compact_cost = model.tick(c, small, selected, shocks)
                mask = np.zeros((1, 4), bool)
                mask[0, list(selected)] = True
                batched, costs = model.batch_tick(
                    c, np.asarray([small]), mask, np.asarray([shocks])
                )
                assert model.compact(actual) == following
                assert cost == compact_cost == costs[0]
                assert np.array_equal(batched[0], following)
        env.step(model.physical_action(s, choices[int(rng.integers(len(choices)))]))


@pytest.mark.parametrize("capacity", model.CAPACITIES)
def test_original_reference_equivalence(capacity):
    for cohort in (137000, 137001, 137002):
        c = model.cohort_config(cohort, capacity, 24)
        env = WaitingEnv(c)
        for name in model.REFERENCES[:3]:
            env.reset(13)
            for t in range(c.horizon):
                state = env.state
                obs = observation(c, state, t)
                old = MatchingReference(name).act(obs)[0]
                new = model.reference(c, model.compact(state), c.horizon - t, name)
                assert tuple(sorted(m for m, _ in old)) == new
                env.step(old)


def test_worker_permutation_invariance():
    c = model.cohort_config(137000, "low", 6)
    state = DispatchState(
        (2, 3, 1, 4),
        (False, True, False, True),
        (0, 2, 0, 3),
        (0, 2, 0, 1),
        (-1, 1, -1, 3),
    )
    variants = []
    for perm in itertools.permutations(range(4)):
        swapped = replace(
            state,
            assigned=tuple(state.assigned[j] for j in perm),
            remaining=tuple(state.remaining[j] for j in perm),
        )
        following, cost, _ = transition(
            c, swapped, model.physical_action(swapped, (0, 2)), (True,) * 4
        )
        variants.append((model.compact(following), cost))
    assert len(set(variants)) == 1


def test_paired_streams_and_capacity_intervention():
    envs = []
    for cap, h in itertools.product(model.CAPACITIES, (12, 24)):
        c = model.cohort_config(134000, cap, h)
        env = WaitingEnv(c)
        env.reset(136000)
        envs.append(env)
    assert len({e.state.ages for e in envs}) == 1
    assert len({e.events[:12] for e in envs}) == 1
    assert len({tuple(row[0] for row in e.config.service_time) for e in envs}) == 1
    assert len({tuple(row[0] for row in e.config.restoration) for e in envs}) == 1
    assert role_seed(136000, "failures") != role_seed(136000, "contention_planner:0")


def test_planner_scenario_replay_against_scalar_rollouts():
    c = model.cohort_config(137000, "middle", 4)
    state = model.compact(WaitingEnv(c).state)
    seed, scenarios = 51, 12
    selected, diag = model.plan(c, state, 4, "independent_dp_matching", seed, scenarios)
    assert (selected, diag) == model.plan(
        c, state, 4, "independent_dp_matching", seed, scenarios
    )
    shocks = (
        np.random.default_rng(seed).random((4, scenarios, 4)) < c.failure_probability
    )
    scores = []
    for first in model.actions(c, state):
        totals = []
        for i in range(scenarios):
            current, total = state, 0
            for t in range(4):
                action = (
                    first
                    if t == 0
                    else model.reference(c, current, 4 - t, "independent_dp_matching")
                )
                current, cost = model.tick(c, current, action, shocks[t, i])
                total += cost
            totals.append(total)
        scores.append(np.mean(totals))
    assert np.allclose(scores, diag["means"])
    assert selected == model.actions(c, state)[int(np.argmin(scores))]


@pytest.mark.parametrize("capacity", model.CAPACITIES)
def test_exact_bounds_regret_and_export(tmp_path, capacity):
    c = model.cohort_config(137000, capacity, 3, 3)
    path = tmp_path / "oracle.jsonl.gz"
    job, rows = run_exact(c, path, progress=False)
    assert job["status"] == "COMPLETED"
    assert len(rows) == 4
    for row in rows:
        assert row["joint_optimum"] >= row["relaxed_lower_bound"] - 1e-8
        assert row["avoidable_decision_cost"] == pytest.approx(
            row["early_regret"] + row["late_regret"]
        )
        assert row["allocation_regret"] == 0
    with gzip.open(path, "rt") as stream:
        nodes = [json.loads(line) for line in stream]
    for node in nodes:
        assert node["value"] == min(node["q"])
    if capacity == "low":
        assert rows[-1]["avoidable_decision_cost"] == pytest.approx(0)
        assert rows[-1]["unavoidable_scarcity_cost"] == pytest.approx(0)


def test_cap_is_explicit_and_keeps_partial_graph(tmp_path):
    c = model.cohort_config(137000, "middle", 3, 3)
    path = tmp_path / "capped.jsonl.gz"
    job, rows = run_exact(c, path, cap=2, progress=False)
    assert job["status"] == "CAPPED"
    assert rows == []
    assert job["states"] == 2
    assert "joint_optimum" not in job
    with gzip.open(path, "rt") as stream:
        assert len(list(stream)) == 2


def test_exact_values_against_original_brute_force():
    c = model.cohort_config(137000, "middle", 2, 2)
    oracle = JointOracle(c, progress=False)

    def brute(h, state):
        if not h:
            return 0
        values = []
        for pairs in matching_actions(c, state):
            value = 0
            for shocks in itertools.product((False, True), repeat=c.machines):
                probability = np.prod(
                    [
                        c.failure_probability if b else 1 - c.failure_probability
                        for b in shocks
                    ]
                )
                following, cost, _ = transition(c, state, pairs, shocks)
                value += probability * (cost + brute(h - 1, following))
            values.append(value)
        return min(values)

    state = WaitingEnv(c).state
    assert oracle.value(2, model.compact(state)) == pytest.approx(brute(2, state))
    oracle.bar.close()


def test_episode_reconciliation_and_summary():
    cfg = settings("smoke")
    rows = []
    for cap, h, seed, name in itertools.product(
        model.CAPACITIES,
        cfg["horizons"],
        cfg["test_seeds"],
        ("independent_dp_matching", "joint_rollout"),
    ):
        c = model.cohort_config(137000, cap, h)
        row = episode(
            c,
            seed,
            name,
            "independent_dp_matching",
            4,
            dict(cohort=137000, capacity=cap, horizon=h),
        )
        assert row["objective"] == row["base_cost"] + row["waiting_cost"]
        assert 0 <= row["utilization"] <= 1
        rows.append(row)
    paired, results = summarize(
        rows, dict.fromkeys(model.CAPACITIES, "independent_dp_matching"), cfg
    )
    assert len(paired) == 6
    assert all(r["status"] == "ENGINEERING_ONLY" for r in results.values())
    with pytest.raises(AssertionError, match="incomplete paired"):
        summarize(
            rows[:-1], dict.fromkeys(model.CAPACITIES, "independent_dp_matching"), cfg
        )


def test_artifact_contract_and_scalar_replay(tmp_path):
    import csv
    import hashlib

    from ht_pdm_fjsp.maintenance_contention_headroom import run, seed_audit

    output = tmp_path / "new_run"
    run(output, "smoke")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["actual_counts"] == dict(development=12, test=90, exact_jobs=3)
    assert manifest["exact_capped_jobs"] == 0
    assert seed_audit(settings("full"))["status"] == "PASS"
    for path, metadata in manifest["artifacts"].items():
        assert (
            hashlib.sha256((output / path).read_bytes()).hexdigest()
            == metadata["sha256"]
        )
    checkpoint = json.loads(
        (output / "controller_checkpoints/controllers.json").read_text()
    )
    with (output / "decisions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 450
    envs = {}
    totals = {}
    for row in rows:
        cohort, seed, horizon, t = (
            int(row[k]) for k in ("cohort", "seed", "horizon", "time")
        )
        capacity, name = row["capacity"], row["controller"]
        key = cohort, capacity, horizon, seed, name
        if key not in envs:
            c = model.cohort_config(cohort, capacity, horizon)
            envs[key] = WaitingEnv(c)
            envs[key].reset(seed)
            totals[key] = 0
        env = envs[key]
        state_data = json.loads(row["state"])
        assert model.compact(env.state) == model.compact(
            DispatchState(**{k: tuple(v) for k, v in state_data.items()})
        )
        if name == "joint_rollout":
            selected, diag = model.plan(
                env.config,
                model.compact(env.state),
                horizon - t,
                checkpoint["references"][capacity],
                role_seed(seed, f"contention_planner:{t}"),
                checkpoint["scenarios"],
            )
            # JSON turns tuples into lists; compare canonical serialized values.
            assert json.loads(json.dumps(diag)) == json.loads(row["planner"])
        else:
            selected = model.reference(
                env.config, model.compact(env.state), horizon - t, name
            )
        pairs = model.physical_action(env.state, selected)
        assert [list(pair) for pair in pairs] == json.loads(row["action"])
        _, reward, _, info = env.step(pairs)
        assert info == json.loads(row["physical_metrics"])
        totals[key] -= reward
    with (output / "episodes.csv").open() as stream:
        for row in csv.DictReader(stream):
            key = (
                int(row["cohort"]),
                row["capacity"],
                int(row["horizon"]),
                int(row["seed"]),
                row["controller"],
            )
            assert totals[key] == float(row["objective"])
    with pytest.raises(ValueError, match="new/empty"):
        run(output, "smoke")


def test_failed_run_manifest_preserves_partial(tmp_path, monkeypatch):
    from ht_pdm_fjsp import maintenance_contention_headroom as runner

    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(runner, "episode", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.run(tmp_path / "failed", "smoke")
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert "injected failure" in manifest["error"]
    assert (tmp_path / "failed/source_snapshot").is_dir()
    assert (tmp_path / "failed/episodes.partial.csv").exists()


def test_invalid_seed_panel_is_rejected():
    from ht_pdm_fjsp.maintenance_contention_headroom import seed_audit

    cfg = settings("smoke")
    cfg["test_seeds"] = cfg["development_seeds"]
    with pytest.raises(ValueError, match="seed panel overlap"):
        seed_audit(cfg)
    cfg = settings("smoke")
    cfg["test_seeds"] = [201]
    with pytest.raises(ValueError, match="seed panel overlap"):
        seed_audit(cfg)
