"""Physics, proposal likelihood, matched controls and artifact replay audits."""

import csv
from dataclasses import asdict, replace
import hashlib
import itertools
import json

import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_heterogeneous_marl as runner
from ht_pdm_fjsp import maintenance_heterogeneous_model as model
from ht_pdm_fjsp import maintenance_heterogeneous_policy as policy
from ht_pdm_fjsp import maintenance_heterogeneous_training as training
from ht_pdm_fjsp.maintenance_dispatch import DispatchState, matching_actions, role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition


@pytest.fixture(autouse=True)
def single_thread():
    torch.set_num_threads(1)


def test_paired_profiles_hold_row_means_and_shocks_with_separate_skill_mask():
    for seed in range(155000, 155016):
        a = model.cohort_config(seed, "homogeneous", 12)
        b = model.cohort_config(seed, "specialized", 12)
        masked = model.cohort_config(seed, "skill_mask", 12)
        assert (
            np.asarray(a.service_time).mean(1).tolist()
            == np.asarray(b.service_time).mean(1).tolist()
        )
        assert (
            np.asarray(a.restoration).mean(1).tolist()
            == np.asarray(b.restoration).mean(1).tolist()
        )
        favorites = np.asarray(b.service_time).argmin(1)
        assert sorted(np.bincount(favorites, minlength=2)) == [1, 3]
        assert (np.asarray(masked.service_time) == 0).sum() == 2
        assert (np.asarray(masked.service_time) > 0).any(0).all()
        assert (np.asarray(masked.service_time) > 0).any(1).all()
        envs = [WaitingEnv(c) for c in (a, b, masked)]
        for e in envs:
            e.reset(155030)
        assert envs[0].state == envs[1].state == envs[2].state
        assert np.array_equal(envs[0].events, envs[1].events)
        assert np.array_equal(envs[0].events, envs[2].events)
        long = WaitingEnv(replace(b, horizon=24))
        long.reset(155030)
        assert np.array_equal(envs[1].events, long.events[:12])


@pytest.mark.parametrize("condition", model.CONDITIONS)
def test_physical_catalogue_masks_and_proposals_reach_every_matching(condition):
    c = model.cohort_config(155000, condition, 12)
    env = WaitingEnv(c)
    env.reset(155010)
    rng = np.random.default_rng(6)
    for t in range(12):
        candidates = model.feasible(c, env.state)
        assert set(candidates) == set(matching_actions(c, env.state))
        batch = policy.feature_batch([c], [env.state], [12 - t])
        masked = {
            policy.CATALOGUE[i]
            for i in torch.nonzero(batch["matching_mask"][0]).flatten().tolist()
        }
        assert masked == set(candidates)
        fixed = {
            policy.CATALOGUE[i]
            for i in torch.nonzero(batch["fixed_mask"][0]).flatten().tolist()
        }
        assert fixed == set(model.fixed_candidates(c, env.state))
        assert len(fixed) == len({tuple(m for m, j in p) for p in candidates})
        for pairs in candidates:
            raw = [0] * 4
            for m, j in pairs:
                raw[m] = j + 1
            assert all(bool(batch["proposal_mask"][0, m, j]) for m, j in enumerate(raw))
            assert model.resolve(raw, t) == (pairs, 0)
        env.step(candidates[int(rng.integers(len(candidates)))])
    assert len(model.catalogue(4, 4)) == 209 and len(model.catalogue(4, 2)) == 21


def test_rotating_collision_arbitration():
    for t in range(4):
        assert model.resolve([1, 1, 1, 1], t) == (((t, 0),), 3)
    assert model.resolve([1, 2, 0, 0], 2) == (((0, 0), (1, 1)), 0)
    assert model.resolve([0, 0, 0, 0], 0) == ((), 0)


@pytest.mark.parametrize("arm", policy.ARMS)
def test_matched_parameters_raw_likelihood_and_stop_only_gradient(arm):
    cfg = runner.settings("smoke")
    models = [runner.model_for(cfg, 155000, a) for a in policy.ARMS]
    assert len({sum(p.numel() for p in m.parameters()) for m in models}) == 1
    assert all(
        torch.equal(v, m.state_dict()[k])
        for k, v in models[0].state_dict().items()
        for m in models[1:]
    )
    net = runner.model_for(cfg, 155000, arm)
    c = model.cohort_config(155000, "specialized", 12)
    env = WaitingEnv(c)
    env.reset(155010)
    batch = policy.feature_batch([c] * 8, [env.state] * 8, [12] * 8)
    action, oldprob, oldvalue = net.sample(
        batch, generator=torch.Generator().manual_seed(19)
    )
    prob, ent, value = net.evaluate(batch, action)
    assert torch.allclose(prob, oldprob) and torch.allclose(value, oldvalue)
    if arm == policy.MARL:
        dist, _ = net.distribution(batch)
        assert torch.allclose(prob, dist.log_prob(action).sum(-1))
        # Different raw proposals can map to the same matching; their likelihoods must remain distinct events.
        assert model.resolve([1, 0, 0, 0], 0)[0] == model.resolve([1, 1, 0, 0], 0)[0]
        assert action.shape == (8, 4)
    else:
        assert action.shape == (8,)
    altered = {k: v.clone() for k, v in batch.items()}
    for key in ("proposal_mask", "matching_mask", "fixed_mask"):
        altered[key][:] = False
        altered[key][..., 0] = True
    action, logprob, _ = net.sample(altered)
    assert (action == 0).all() and (logprob == 0).all()
    prob, ent, value = net.evaluate(altered, action)
    assert (ent == 0).all()
    (prob.mean() + value.square().mean()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None
    )
    with pytest.raises(ValueError, match="infeasible"):
        net.evaluate(altered, torch.ones_like(action))


@pytest.mark.parametrize("condition", model.CONDITIONS)
def test_forecast_physics_matches_scalar_service_ownership_and_all_costs(condition):
    c = model.cohort_config(155000, condition, 24)
    envs = [WaitingEnv(c) for _ in range(9)]
    for i, env in enumerate(envs):
        env.reset(155040 + i)
    rng = np.random.default_rng(7)
    for t in range(24):
        s = {
            k: np.asarray(
                [getattr(e.state, k) for e in envs],
                dtype=bool if k == "failed" else int,
            )
            for k in ("ages", "failed", "pending_wait", "remaining", "assigned")
        }
        selected = np.zeros((len(envs), 4, c.technicians), bool)
        pairs = []
        for i, e in enumerate(envs):
            candidates = model.feasible(c, e.state)
            p = candidates[int(rng.integers(len(candidates)))]
            pairs.append(p)
            for m, j in p:
                selected[i, m, j] = True
        following, cost = model.batch_tick(
            c, s, selected, np.asarray([e.events[t] for e in envs])
        )
        for i, (env, p) in enumerate(zip(envs, pairs, strict=True)):
            _, reward, _, info = env.step(p)
            assert -reward == cost[i] == info["objective"]
            for key in following:
                assert tuple(following[key][i]) == getattr(env.state, key)


def test_latched_worker_restoration_only_on_completion():
    c = model.cohort_config(155000, "specialized", 4)
    slow = int(np.argmax(c.service_time[0]))
    duration = c.service_time[0][slow]
    state = DispatchState(
        (7, 0, 0, 0), (True, False, False, False), (3, 0, 0, 0), (0, 0), (-1, -1)
    )
    for t in range(duration):
        state, cost, info = transition(
            c, state, ((0, slow),) if t == 0 else (), (False,) * 4
        )
        assert state.ages[0] == (7 if t < duration - 1 else 3)
        assert state.failed[0] == (t < duration - 1)
        assert info["unavailable_ticks"] == 1
    assert state.assigned[slow] == -1 and state.pending_wait[0] == 0


def test_adaptive_one_machine_terminal_gain_and_forecast_without_future_events():
    c = model.cohort_config(155000, "specialized", 4)
    args = model.profile(c, 0)
    for j in range(c.technicians):
        # Last interval: repairing failed machine costs corrective+down vs down+overdue if deferred.
        assert model.gain(1, 3, 1, 5, j, args) == c.waiting_price - c.corrective_cost
    env = WaitingEnv(c)
    env.reset(155040)
    pairs, diag = model.plan(c, env.state, 4, "adaptive_isolated_matching", 123, 4)
    assert pairs in model.feasible(c, env.state) and diag["scenario_seed"] == 123
    env.events = tuple(tuple(not v for v in row) for row in env.events)
    assert model.plan(c, env.state, 4, "adaptive_isolated_matching", 123, 4) == (
        pairs,
        diag,
    )
    for condition in ("homogeneous", "nominal"):
        config = model.cohort_config(155000, condition, 6)
        e = WaitingEnv(config)
        e.reset(155040)
        for t in range(6):
            full, fd = model.plan(
                config, e.state, 6 - t, "adaptive_isolated_matching", t + 12, 4
            )
            fixed, xd = model.plan(
                config, e.state, 6 - t, "adaptive_isolated_matching", t + 12, 4, True
            )
            assert fd["selected_score"] == xd["selected_score"]
            fs, fc, _ = transition(config, e.state, full, e.events[t])
            xs, xc, _ = transition(config, e.state, fixed, e.events[t])
            assert fc == xc and fs == xs
            e.step(full)


def test_seeds_full_population_and_complete_returns():
    cfg = runner.settings("full")
    audit, forbidden = runner.seed_audit(cfg)
    assert audit["status"] == "PASS"
    specs = [training.episode_spec(155000, i, 12, forbidden) for i in range(1, 11)]
    assert sum(s[1] == "specialized" for s in specs) == 8
    assert sum(s[1] == "nominal" for s in specs) == 2
    with pytest.raises(ValueError, match="seed overlap"):
        training.episode_spec(155000, 1, 12, forbidden | {specs[0][2]})
    cfg["test_seeds"] = cfg["development_seeds"]
    with pytest.raises(ValueError, match="seed overlap"):
        runner.seed_audit(cfg)
    assert runner.counts(runner.settings("full")) == dict(
        models=30,
        training_steps=14745600,
        training_episodes=1228800,
        optimizer_steps=307200,
        development_episodes=63200,
        test_episodes=144000,
        test_decision_intervals=2592000,
        test_machine_rows=576000,
    )
    assert torch.equal(
        training.monte_carlo(torch.tensor([[-1.0, -10.0], [-2.0, -20.0]])),
        torch.tensor([[-3.0, -30.0], [-2.0, -20.0]]),
    )


@pytest.mark.parametrize("arm", policy.ARMS)
def test_training_full_shapes_and_optimizer_budget_without_sealed_seeds(tmp_path, arm):
    cfg = runner.settings("full")
    cfg.update(env_steps=1536, checkpoint_updates=[1])
    _, forbidden = runner.seed_audit(cfg)
    net = runner.model_for(cfg, 155000, arm)
    journals = {
        key: runner.Journal(tmp_path / f"{key}.csv")
        for key in ("training_episodes", "training_progress")
    }
    try:
        record = training.fit(net, cfg, 155000, forbidden, journals, lambda *args: None)
    finally:
        for j in journals.values():
            j.close()
    assert record["env_steps"] == 1536 and record["episodes"] == 128
    assert record["optimizer_steps"] == 32 and record["parameters_changed"]
    assert record["mixture_counts"] == dict(nominal=25, specialized=103)
    with pytest.raises(ValueError, match="seed overlap"):
        training.fit(
            net,
            cfg,
            155000,
            forbidden | {role_seed(155000, "heterogeneous_learning_actions")},
            journals,
            lambda *args: None,
        )


def test_primary_gates_pair_training_seeds_and_apply_bonferroni():
    cfg = runner.settings("full")
    cfg["test_cohorts"] = [155010]
    cfg["test_seeds"] = [155040]
    rows = []
    refs = dict.fromkeys(model.CONDITIONS, model.REFERENCES[0])
    for condition, h in itertools.product(model.CONDITIONS, cfg["horizons"]):
        for name in cfg["controls"]:
            rows.append(
                dict(
                    train_seed=-1,
                    cohort=155010,
                    condition=condition,
                    horizon=h,
                    seed=155040,
                    controller=name,
                    objective=100.0,
                    base_cost=100.0,
                    waiting_violation=0.0,
                )
            )
        for i, seed in enumerate(cfg["train_seeds"]):
            for arm, cost in (
                (policy.FIXED, 91),
                (policy.CENTRAL, 87),
                (policy.MARL, 84),
            ):
                cost = 100 if condition == "nominal" else cost + 0.01 * i
                rows.append(
                    dict(
                        train_seed=seed,
                        cohort=155010,
                        condition=condition,
                        horizon=h,
                        seed=155040,
                        controller=arm,
                        objective=cost,
                        base_cost=cost,
                        waiting_violation=0.0,
                    )
                )
    paired, cohorts, contrasts = runner.summarize(rows, refs, cfg)
    primary = [
        v
        for v in contrasts.values()
        if v["condition"] == "specialized" and v["horizon"] == 12
    ]
    assert len(primary) == 5 and all(v["status"] == "PASS" for v in primary)
    assert all(
        v["confidence_level"] == 0.975 for v in primary if v["right"] in policy.ARMS
    )
    assert len(paired) == 400 and len(cohorts) == 400
    with pytest.raises(RuntimeError, match="incomplete learned panel"):
        runner.summarize(rows[:-1], refs, cfg)
    for r in rows:
        if r["controller"] == policy.MARL and r["condition"] == "nominal":
            r["base_cost"] = 120.0
    _, _, contrasts = runner.summarize(rows, refs, cfg)
    primary = [
        v
        for v in contrasts.values()
        if v["left"] == policy.MARL
        and v["condition"] == "specialized"
        and v["horizon"] == 12
    ]
    assert all(
        v["status"] == "FAIL" and not v["gates"]["nominal_ratio_110pct"]
        for v in primary
    )


def test_selection_excludes_initial_and_checks_nominal_waiting_guards():
    refs = [
        dict(condition=cond, objective=100, base_cost=100, waiting_violation=0)
        for cond in ("specialized", "nominal")
    ]

    def entry(update, cost, nominal=100, wait=0):
        return dict(
            update=update,
            steps=128 * update,
            episodes=[
                dict(
                    condition="specialized",
                    objective=cost,
                    base_cost=cost,
                    waiting_violation=wait,
                ),
                dict(
                    condition="nominal",
                    objective=nominal,
                    base_cost=nominal,
                    waiting_violation=0,
                ),
            ],
        )

    records = [
        entry(0, 0),
        entry(1, 90, 111),
        entry(2, 92, 100, 0.03),
        entry(3, 95),
        entry(4, 95),
    ]
    chosen, status = runner.select_checkpoint(records, refs)
    assert chosen["update"] == 3 and status == "FEASIBLE"
    chosen, status = runner.select_checkpoint(records[:3], refs)
    assert chosen["update"] == 2 and status == "DEVELOPMENT_INFEASIBLE"


def test_smoke_artifacts_physics_raw_proposals_and_checkpoint_replay(tmp_path):
    output = tmp_path / "smoke"
    runner.run(output, "smoke")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and manifest[
        "actual_counts"
    ] == runner.counts(runner.settings("smoke"))
    for name, meta in manifest["artifacts"].items():
        assert (
            hashlib.sha256((output / name).read_bytes()).hexdigest() == meta["sha256"]
        )
    assert (output / "episodes.csv").read_bytes() == (
        output / "episodes.partial.csv"
    ).read_bytes()
    selection = json.loads((output / "checkpoint_selection.json").read_text())[
        "selections"
    ]
    models = {}
    for arm in policy.ARMS:
        record = selection[f"{arm}:155000"]
        assert record["selected_steps"] in (128, 256)
        models[arm], payload = runner.load_checkpoint(output / record["checkpoint"])
        assert (
            payload["algorithm"] == arm
            and payload["physical_steps"] == record["selected_steps"]
        )
    coverage = json.loads((output / "training_coverage.json").read_text())
    assert len({r["initial_parameters_sha256"] for r in coverage}) == 1
    assert len({r["parameter_count"] for r in coverage}) == 1
    assert all(r["parameters_changed"] and r["optimizer_steps"] == 16 for r in coverage)
    decisions = list(csv.DictReader((output / "decisions.csv").open()))
    assert len(decisions) == 1080
    assert len(list(csv.DictReader((output / "machine_metrics.csv").open()))) == 864
    last = None
    totals = {}
    for row in decisions:
        key = (
            int(row["train_seed"]),
            int(row["cohort"]),
            row["condition"],
            int(row["horizon"]),
            int(row["seed"]),
            row["controller"],
        )
        if key != last:
            c = model.cohort_config(key[1], key[2], key[3])
            env = WaitingEnv(c)
            env.reset(key[4])
            last = key
        assert asdict(env.state) == {
            k: tuple(v) for k, v in json.loads(row["state"]).items()
        }
        pairs = tuple(tuple(x) for x in json.loads(row["action"]))
        t = int(row["time"])
        if key[-1] in policy.ARMS:
            chosen, diag = models[key[-1]].select(c, env.state, key[3] - t)
            assert chosen == pairs and json.loads(json.dumps(diag)) == json.loads(
                row["controller_diagnostics"]
            )
        _, _, _, info = env.step(pairs)
        assert info == json.loads(row["physical_metrics"])
        assert asdict(env.state) == {
            k: tuple(v) for k, v in json.loads(row["following"]).items()
        }
        totals[key] = env.metrics["objective"]
    for row in csv.DictReader((output / "episodes.csv").open()):
        key = (
            int(row["train_seed"]),
            int(row["cohort"]),
            row["condition"],
            int(row["horizon"]),
            int(row["seed"]),
            row["controller"],
        )
        assert float(row["objective"]) == totals[key]
    summary = json.loads((output / "summary.json").read_text())
    assert all(v["status"] == "ENGINEERING_ONLY" for v in summary["contrasts"].values())
    with pytest.raises(ValueError, match="new/empty"):
        runner.run(output, "smoke")
    payload["feature_contract"] = {}
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="contract mismatch"):
        runner.load_checkpoint(tmp_path / "bad.pt")


def test_failure_preserves_partial_artifacts(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected failure")

    monkeypatch.setattr(runner, "fit", fail)
    with pytest.raises(RuntimeError, match="injected failure"):
        runner.run(tmp_path / "failed", "smoke")
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert manifest["status"] == "FAILED" and "injected" in manifest["error"]
    assert (tmp_path / "failed/episodes.partial.csv").exists()
    assert list((tmp_path / "failed" / policy.CENTRAL).rglob("*.pt"))


def test_contention_respects_interchangeable_workers_and_detects_specialist_bottleneck():
    homo = model.cohort_config(155000, "homogeneous", 4)
    hetero = model.cohort_config(155000, "specialized", 4)
    favorites = np.asarray(hetero.service_time).argmin(1)
    common = int(np.bincount(favorites).argmax())
    machines = [m for m in range(4) if favorites[m] == common][:2]
    ages = tuple(2 if m in machines else 0 for m in range(4))
    state = DispatchState(ages, (False,) * 4, (0,) * 4, (0, 0), (-1, -1))
    assert model.contention(homo, state) == (0, 0)
    assert model.contention(hetero, state) == (1, 0)
    other = next(m for m in range(4) if m not in machines)
    remaining = tuple(1 if j == common else 0 for j in range(2))
    assigned = tuple(other if j == common else -1 for j in range(2))
    busy = replace(state, remaining=remaining, assigned=assigned)
    assert model.contention(hetero, busy) == (2, 2)
    net = runner.model_for(runner.settings("smoke"), 155000, policy.MARL)
    raw = torch.zeros(4, dtype=torch.long)
    raw[machines[0]] = common + 1
    with pytest.raises(ValueError, match="infeasible raw proposal"):
        net.decode(hetero, busy, 4, raw)
