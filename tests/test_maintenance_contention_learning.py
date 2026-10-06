import csv
from dataclasses import replace
import hashlib
import itertools
import json

import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_contention_learning as runner
from ht_pdm_fjsp import maintenance_contention_learning_policy as policy
from ht_pdm_fjsp import maintenance_contention_learning_training as training
from ht_pdm_fjsp.maintenance_contention_model import (
    actions,
    cohort_config,
    compact,
    physical_action,
    subsets,
)
from ht_pdm_fjsp.maintenance_dispatch import matching_actions
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv


@pytest.fixture(autouse=True)
def one_torch_thread():
    torch.set_num_threads(1)


def test_mask_covers_full_physical_action_space_without_worker_duplicates():
    for cap in ("low", "middle", "high"):
        c = cohort_config(143000, cap, 12)
        env = WaitingEnv(c)
        env.reset(143010)
        rng = np.random.default_rng(1)
        for t in range(12):
            state = env.state
            batch = policy.feature_batch([c], [compact(state)], [12 - t])
            chosen = {
                subsets(4)[i]
                for i in torch.nonzero(batch["mask"][0]).flatten().tolist()
            }
            expected = {
                tuple(sorted(m for m, j in pairs))
                for pairs in matching_actions(c, state)
            }
            assert chosen == expected == set(actions(c, compact(state)))
            assert () in chosen
            choice = sorted(chosen)[int(rng.integers(len(chosen)))]
            env.step(physical_action(state, choice))


def test_sampling_probability_value_reevaluation_and_stop_only():
    c = cohort_config(143000, "middle", 12)
    env = WaitingEnv(c)
    env.reset(143010)
    model = runner.model_for(runner.settings("smoke"), 143000)
    states = [compact(env.state)] * 16
    batch = policy.feature_batch([c] * 16, states, [12] * 16)
    generator = torch.Generator().manual_seed(19)
    action, oldprob, oldvalue = model.sample(batch, generator=generator)
    prob, entropy, value = model.evaluate(batch, action)
    assert torch.allclose(prob, oldprob) and torch.allclose(value, oldvalue)
    assert torch.isfinite(prob).all() and torch.isfinite(entropy).all()
    altered = {k: v.clone() for k, v in batch.items()}
    altered["mask"][:] = False
    altered["mask"][:, 0] = True
    action, logprob, _ = model.sample(altered, generator=generator)
    assert (action == 0).all() and (logprob == 0).all()
    prob, entropy, value = model.evaluate(altered, action)
    assert (entropy == 0).all()
    (prob.mean() + value.square().mean()).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    with pytest.raises(ValueError, match="infeasible"):
        model.evaluate(altered, torch.ones(16, dtype=torch.int64))


def test_machine_permutation_equivariance_of_logits():
    c = cohort_config(143000, "middle", 12)
    state = ((2, 0, 0, 0), (3, 1, 4, 0), (1, 0, 0, 1), (0, 0, 0, 0))
    model = runner.model_for(runner.settings("smoke"), 143000)
    original, _ = model.distribution(policy.feature_batch([c], [state], [8]))
    original_probs = original.probs.detach()[0]
    for perm in itertools.permutations(range(4)):
        altered = replace(
            c,
            service_time=tuple(c.service_time[i] for i in perm),
            restoration=tuple(c.restoration[i] for i in perm),
        )
        mapped = tuple(state[i] for i in perm)
        distribution, _ = model.distribution(
            policy.feature_batch([altered], [mapped], [8])
        )
        for index, choice in enumerate(subsets(4)):
            previous = tuple(sorted(perm[i] for i in choice))
            assert float(distribution.probs.detach()[0, index]) == pytest.approx(
                float(original_probs[subsets(4).index(previous)]), abs=1e-7
            )


def test_monte_carlo_returns_keep_parallel_episode_boundaries_and_sign():
    rewards = torch.tensor([[-1.0, -10.0], [-2.0, -20.0], [-3.0, -30.0]])
    result = training.monte_carlo(rewards)
    assert torch.equal(
        result, torch.tensor([[-6.0, -60.0], [-5.0, -50.0], [-3.0, -30.0]])
    )
    with pytest.raises(ValueError):
        training.monte_carlo(torch.ones(6))


def test_training_counter_roles_mixture_and_reserved_seed_guard():
    cfg = runner.settings("full")
    audit, forbidden = runner.seed_audit(cfg)
    assert audit["status"] == "PASS"
    specs = [training.episode_spec(138000, i, 12, forbidden) for i in range(1, 11)]
    assert sum(s[1] == "middle" for s in specs) == 8
    assert sum(s[1] == "low" for s in specs) == 2
    assert len({s[2] for s in specs}) == 10 and len({s[3] for s in specs}) == 10
    assert not {s[2] for s in specs} & {s[3] for s in specs}
    with pytest.raises(ValueError, match="seed overlap"):
        training.episode_spec(138000, 1, 12, forbidden | {specs[0][2]})
    cfg["test_seeds"] = cfg["development_seeds"]
    with pytest.raises(ValueError, match="seed overlap"):
        runner.seed_audit(cfg)
    counts = runner.counts(runner.settings("full"))
    assert counts == dict(
        models=10,
        training_steps=4915200,
        training_episodes=409600,
        optimizer_steps=102400,
        development_episodes=22400,
        test_episodes=45000,
        test_decision_intervals=810000,
        test_machine_rows=180000,
    )


def test_checkpoint_roundtrip_and_contract_rejection(tmp_path):
    cfg = runner.settings("smoke")
    model = runner.model_for(cfg, 143000)
    path = tmp_path / "checkpoint.pt"
    restored, payload = runner.save_checkpoint(path, model, cfg, 143000, 0, 0)
    assert payload["physical_steps"] == 0
    assert all(
        torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items()
    )
    payload["feature_contract"] = {}
    torch.save(payload, path)
    with pytest.raises(ValueError, match="contract mismatch"):
        runner.load_checkpoint(path)


def test_checkpoint_selection_uses_guards_and_excludes_initial():
    refs = [
        dict(capacity="middle", base_cost=100, objective=100, waiting_violation=0.01),
        dict(capacity="low", base_cost=100, objective=100, waiting_violation=0),
    ]

    def record(update, cost, nominal=100, violation=0.01):
        return dict(
            update=update,
            steps=100 * update,
            episodes=[
                dict(
                    capacity="middle",
                    base_cost=cost,
                    objective=cost,
                    waiting_violation=violation,
                ),
                dict(
                    capacity="low",
                    base_cost=nominal,
                    objective=nominal,
                    waiting_violation=0,
                ),
            ],
        )

    records = [
        record(0, 1),
        record(1, 90, 111),
        record(2, 91, 100, 0.04),
        record(3, 95),
        record(4, 95),
    ]
    chosen, status = runner.select_checkpoint(records, refs)
    assert chosen["update"] == 3 and status == "FEASIBLE"
    records = [record(0, 1), record(1, 90, 120), record(2, 91, 100, 0.04)]
    chosen, status = runner.select_checkpoint(records, refs)
    assert chosen["update"] == 2 and status == "DEVELOPMENT_INFEASIBLE"


def test_primary_gates_and_no_pseudo_replication():
    cfg = runner.settings("full")
    cfg["test_cohorts"] = [141000]
    cfg["test_seeds"] = [142000]
    rows = []
    selected = dict.fromkeys(("low", "middle", "high"), "independent_dp_matching")
    for cap, h in itertools.product(cfg["capacities"], cfg["horizons"]):
        for name in cfg["controls"]:
            cost = 90 if name == "joint_rollout" else 100
            rows.append(
                dict(
                    train_seed=-1,
                    cohort=141000,
                    capacity=cap,
                    horizon=h,
                    seed=142000,
                    controller=name,
                    objective=cost,
                    base_cost=cost,
                    waiting_violation=0,
                )
            )
        for i, seed in enumerate(cfg["train_seeds"]):
            cost = 101 if cap == "low" else 93 + i * 0.1
            rows.append(
                dict(
                    train_seed=seed,
                    cohort=141000,
                    capacity=cap,
                    horizon=h,
                    seed=142000,
                    controller=runner.LEARNER,
                    objective=cost,
                    base_cost=cost,
                    waiting_violation=0,
                )
            )
    paired, cohorts, contrasts = runner.summarize(rows, selected, cfg)
    primary = contrasts["middle_H12"]
    assert primary["status"] == "PASS" and primary["training_seed_wins"] == 10
    assert primary["priced_reduction"] == pytest.approx(0.0655)
    assert primary["ci95"][1] < 0
    assert len(paired) == 60 and len(cohorts) == 60
    with pytest.raises(RuntimeError, match="incomplete learned panel"):
        runner.summarize(rows[:-1], selected, cfg)
    for row in rows:
        if row["controller"] == runner.LEARNER and row["capacity"] == "low":
            row["base_cost"] = 120
    _, _, contrasts = runner.summarize(rows, selected, cfg)
    assert contrasts["middle_H12"]["status"] == "FAIL"
    assert not contrasts["middle_H12"]["gates"]["nominal_ratio_110pct"]


def test_local_smoke_artifacts_and_selected_model_replay(tmp_path):
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
    coverage = json.loads((output / "training_coverage.json").read_text())
    assert coverage[0]["parameters_changed"] and coverage[0]["optimizer_steps"] == 16
    selection = json.loads((output / "checkpoint_selection.json").read_text())[
        "selections"
    ]["143000"]
    assert selection["selected_steps"] in (128, 256)
    model, payload = runner.load_checkpoint(output / selection["checkpoint"])
    assert payload["physical_steps"] == selection["selected_steps"]
    decisions = list(csv.DictReader((output / "decisions.csv").open()))
    machines = list(csv.DictReader((output / "machine_metrics.csv").open()))
    assert len(decisions) == 540 and len(machines) == 432
    last = None
    env = None
    totals = {}
    for row in decisions:
        key = (
            int(row["train_seed"]),
            int(row["cohort"]),
            row["capacity"],
            int(row["horizon"]),
            int(row["seed"]),
            row["controller"],
        )
        if key != last:
            c = cohort_config(key[1], key[2], key[3])
            env = WaitingEnv(c)
            env.reset(key[4])
            last = key
        t = int(row["time"])
        pairs = tuple(tuple(x) for x in json.loads(row["action"]))
        if row["controller"] == runner.LEARNER:
            choice, diag = model.select(env.config, compact(env.state), key[3] - t)
            assert physical_action(env.state, choice) == pairs
            assert diag == json.loads(row["controller_diagnostics"])
        _, _, _, info = env.step(pairs)
        assert info == json.loads(row["physical_metrics"])
        totals[key] = env.metrics["objective"]
    for row in csv.DictReader((output / "episodes.csv").open()):
        key = (
            int(row["train_seed"]),
            int(row["cohort"]),
            row["capacity"],
            int(row["horizon"]),
            int(row["seed"]),
            row["controller"],
        )
        assert float(row["objective"]) == totals[key]
    summary = json.loads((output / "summary.json").read_text())
    assert all(r["status"] == "ENGINEERING_ONLY" for r in summary["contrasts"].values())
    with pytest.raises(ValueError, match="new/empty"):
        runner.run(output, "smoke")


def test_training_failure_preserves_failed_manifest(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected training failure")

    monkeypatch.setattr(runner, "fit", fail)
    with pytest.raises(RuntimeError, match="injected training failure"):
        runner.run(tmp_path / "failed", "smoke")
    manifest = json.loads((tmp_path / "failed/manifest.json").read_text())
    assert manifest["status"] == "FAILED" and "injected training" in manifest["error"]
    assert (tmp_path / "failed/episodes.partial.csv").exists()
    assert list((tmp_path / "failed" / runner.LEARNER).rglob("*.pt"))


def test_vector_training_full_shapes_and_exact_optimizer_budget(tmp_path):
    cfg = runner.settings("full")
    cfg.update(env_steps=1536, checkpoint_updates=[1])
    _, forbidden = runner.seed_audit(cfg)
    model = runner.model_for(cfg, 143000)
    journals = {
        key: runner.Journal(tmp_path / f"{key}.csv")
        for key in ("training_episodes", "training_progress")
    }
    saved = []
    try:
        record = training.fit(
            model,
            cfg,
            143000,
            forbidden,
            journals,
            lambda m, update, steps: saved.append((update, steps)),
        )
    finally:
        for journal in journals.values():
            journal.close()
    assert record["env_steps"] == 1536 and record["episodes"] == 128
    assert record["optimizer_steps"] == 32 and record["parameters_changed"]
    assert record["mixture_counts"] == dict(low=25, middle=103)
    assert saved == [(1, 1536)]
    rows = list(csv.DictReader((tmp_path / "training_episodes.csv").open()))
    assert len(rows) == 128
    for row in rows:
        assert int(row["config_seed"]) not in forbidden
        assert int(row["environment_seed"]) not in forbidden


def test_deterministic_near_ties_choose_canonical_feasible_subset(monkeypatch):
    from torch.distributions import Categorical

    cfg = runner.settings("smoke")
    model = runner.model_for(cfg, 143000)
    c = cohort_config(143000, "middle", 4)
    state = compact(WaitingEnv(c).state)
    batch = policy.feature_batch([c], [state], [4])
    valid = torch.nonzero(batch["mask"][0]).flatten().tolist()
    first, second = valid[:2]
    logits = torch.full(batch["mask"].shape, -100.0)
    logits[0, first] = 0.0
    logits[0, second] = policy.TIE_TOLERANCE / 2
    monkeypatch.setattr(
        model,
        "distribution",
        lambda batch: (
            Categorical(logits=logits.masked_fill(~batch["mask"], -torch.inf)),
            torch.zeros(1),
        ),
    )
    assert int(model.sample(batch, deterministic=True)[0][0]) == first
    logits[0, second] = 4 * policy.TIE_TOLERANCE
    assert int(model.sample(batch, deterministic=True)[0][0]) == second
