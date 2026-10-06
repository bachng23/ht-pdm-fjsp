import csv
import hashlib
import itertools
import json

import pytest
import torch

from ht_pdm_fjsp import maintenance_contention_imitation as runner
from ht_pdm_fjsp import maintenance_contention_imitation_training as training
from ht_pdm_fjsp.maintenance_contention_learning_policy import feature_batch
from ht_pdm_fjsp.maintenance_contention_model import (
    cohort_config,
    compact,
    physical_action,
    plan,
)
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(1)


def journals(path):
    return {
        k: runner.Journal(path / f"{k}.csv")
        for k in ("teacher_rows", "teacher_noise", "bc_progress", "bc_diagnostics")
    }


def test_set_supervision_accepts_ties_mask_and_has_finite_gradients():
    cfg = runner.settings("smoke")
    model = runner.model_for(cfg, 149000)
    c = cohort_config(149000, "middle", 4)
    env = WaitingEnv(c)
    env.reset(149030)
    features = feature_batch([c], [compact(env.state)], [4])
    valid = torch.nonzero(features["mask"][0]).flatten().tolist()
    targets = torch.zeros_like(features["mask"])
    targets[0, valid[:2]] = True
    dist, _ = model.distribution(features)
    expected = -dist.probs[0, valid[:2]].sum().log()
    loss = training.set_loss(model, features, targets)
    assert float(loss.detach()) == pytest.approx(float(expected.detach()))
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert all(p.grad is None for p in model.critic.parameters())
    with pytest.raises(ValueError, match="empty set"):
        training.set_loss(model, features, torch.zeros_like(targets))
    invalid = torch.nonzero(~features["mask"][0]).flatten().tolist()
    targets[0, invalid[0]] = True
    with pytest.raises(ValueError, match="infeasible"):
        training.set_loss(model, features, targets)
    mask = features["mask"].clone()
    mask[:] = False
    mask[:, 0] = True
    features = {**features, "mask": mask}
    loss = training.set_loss(model, features, mask.clone())
    assert float(loss.detach()) == 0
    loss.backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def test_target_tolerance_and_seed_guard():
    choices = ((), (0,), (1,))
    target = training.accepted_targets(choices, [10, 10 + 0.5e-6, 11], 2)
    assert torch.nonzero(target).flatten().tolist() == [0, 1]
    seed = training.checked_seed(149000, "imitation_train_configuration:1", set())
    with pytest.raises(ValueError, match="reserved"):
        training.checked_seed(149000, "imitation_train_configuration:1", {seed})
    cfg = runner.settings("full")
    audit, forbidden = runner.seed_audit(cfg)
    assert audit["status"] == "PASS" and 138000 in forbidden and 143042 in forbidden
    assert not set(cfg["test_seeds"]) & set(cfg["development_seeds"])
    cfg["test_cohorts"] = [141000]
    with pytest.raises(ValueError, match="seed overlap"):
        runner.seed_audit(cfg)


def test_full_and_smoke_exact_budgets():
    expected = dict(
        models=30,
        training_steps=9830400,
        training_episodes=819200,
        optimizer_steps=256000,
        teacher_physical_steps=138240,
        teacher_trajectories=11520,
        teacher_training_rows=122880,
        teacher_validation_rows=15360,
        teacher_noise_probes=320,
        development_episodes=62400,
        test_episodes=105000,
        test_decision_intervals=1890000,
        test_machine_rows=420000,
    )
    assert runner.counts(runner.settings("full")) == expected
    smoke = runner.counts(runner.settings("smoke"))
    assert smoke["development_episodes"] == 34 and smoke["optimizer_steps"] == 40
    assert smoke["test_decision_intervals"] == 720


def test_teacher_current_state_labels_do_not_use_actual_future(tmp_path, monkeypatch):
    cfg = runner.settings("smoke")
    cfg.update(teacher_episodes=1, teacher_validation_episodes=1, noise_probes=1)
    _, forbidden = runner.seed_audit(cfg)
    refs = dict.fromkeys(("low", "middle", "high"), "independent_dp_matching")
    first = tmp_path / "one"
    first.mkdir()
    js = journals(first)
    try:
        original, cov = training.collect(cfg, 149000, forbidden, refs, js)
    finally:
        for j in js.values():
            j.close()
    assert cov["physical_steps"] == 8 and cov["noise_probes"] == 1
    original_reset = WaitingEnv.reset

    def changed_reset(self, seed):
        result = original_reset(self, seed)
        self.events = tuple(tuple(not event for event in row) for row in self.events)
        return result

    # Only the first state's label/features must stay identical: later states
    # legitimately change after the different actual shock history.
    monkeypatch.setattr(WaitingEnv, "reset", changed_reset)
    second = tmp_path / "two"
    second.mkdir()
    js = journals(second)
    try:
        changed, _ = training.collect(cfg, 149000, forbidden, refs, js)
    finally:
        for j in js.values():
            j.close()
    for split in ("train", "validation"):
        assert torch.equal(original[split]["targets"][0], changed[split]["targets"][0])
        assert torch.equal(original[split]["scores"][0], changed[split]["scores"][0])
        for key in original[split]["features"]:
            assert torch.equal(
                original[split]["features"][key][0], changed[split]["features"][key][0]
            )


def test_validation_labels_never_change_bc_weights(tmp_path):
    cfg = runner.settings("smoke")
    _, forbidden = runner.seed_audit(cfg)
    js = journals(tmp_path)
    try:
        data, _ = training.collect(
            cfg,
            149000,
            forbidden,
            dict.fromkeys(("low", "middle", "high"), "independent_dp_matching"),
            js,
        )
        first = runner.model_for(cfg, 149000)
        second = runner.model_for(cfg, 149000)
        coverage = training.fit_imitation(first, cfg, 149000, data, js, lambda *a: None)
        changed = {
            **data,
            "validation": {
                **data["validation"],
                "targets": data["validation"]["features"]["mask"].clone(),
            },
        }
        training.fit_imitation(second, cfg, 149000, changed, js, lambda *a: None)
        assert coverage["optimizer_steps"] == 8 and coverage["critic_head_unchanged"]
        assert coverage["validation_training_rows"] == 0
        assert all(
            torch.equal(v, second.state_dict()[k])
            for k, v in first.state_dict().items()
        )
    finally:
        for j in js.values():
            j.close()


def test_primary_uses_paired_replicas_and_separate_economic_gate():
    cfg = runner.settings("full")
    cfg.update(test_cohorts=[147000], test_seeds=[148000])
    refs = dict.fromkeys(("low", "middle", "high"), "independent_dp_matching")
    rows = []
    for cap, h in itertools.product(cfg["capacities"], cfg["horizons"]):
        for name in cfg["controls"]:
            rows.append(
                dict(
                    controller=name,
                    train_seed=-1,
                    cohort=147000,
                    seed=148000,
                    capacity=cap,
                    horizon=h,
                    objective=90 if name == "joint_rollout" else 100,
                    base_cost=100,
                    waiting_violation=0,
                )
            )
        for arm, seed in itertools.product(runner.ARMS, cfg["train_seeds"]):
            cost = 99 if arm == runner.DIRECT else 96
            rows.append(
                dict(
                    controller=arm,
                    train_seed=seed,
                    cohort=147000,
                    seed=148000,
                    capacity=cap,
                    horizon=h,
                    objective=cost,
                    base_cost=99,
                    waiting_violation=0,
                )
            )
    paired, cohorts, summary = runner.summarize(rows, refs, cfg)
    assert len(paired) == 180 and len(cohorts) == 180
    assert summary["primary"]["status"] == "PASS" and summary["primary"]["wins"] == 10
    assert len(summary["primary"]["paired_differences"]) == 10
    assert (
        summary["arms"][runner.HYBRID]["middle_H12"]["status"] == "FAIL"
    )  # <5% economic gain
    with pytest.raises(RuntimeError, match="incomplete learned panel"):
        runner.summarize(rows[:-1], refs, cfg)
    for row in rows:
        if row["controller"] == runner.HYBRID and row["capacity"] == "low":
            row["base_cost"] = 120
    assert runner.summarize(rows, refs, cfg)[2]["primary"]["status"] == "FAIL"


def test_smoke_all_artifacts_models_teacher_and_physics_replay(tmp_path):
    output = tmp_path / "smoke"
    runner.run(output, "smoke")
    cfg = runner.settings("smoke")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and manifest[
        "actual_counts"
    ] == runner.counts(cfg)
    for name, meta in manifest["artifacts"].items():
        assert (
            hashlib.sha256((output / name).read_bytes()).hexdigest() == meta["sha256"]
        )
    selection = json.loads((output / "checkpoint_selection.json").read_text())[
        "selections"
    ]
    models = {
        arm: runner.load_checkpoint(output / selection[f"{arm}:149000"]["checkpoint"])[
            0
        ]
        for arm in runner.ARMS
    }
    bc = selection[f"{runner.IMITATION}:149000"]
    hybrid = selection[f"{runner.HYBRID}:149000"]
    assert hybrid["parent_checkpoint_sha256"] == bc["checkpoint_sha256"]
    init, _ = runner.load_checkpoint(
        output / runner.HYBRID / "train_seed_149000/checkpoints/counter_00000000.pt"
    )
    assert all(
        torch.equal(v, init.state_dict()[k])
        for k, v in models[runner.IMITATION].state_dict().items()
    )
    direct, _ = runner.load_checkpoint(
        output / runner.DIRECT / "train_seed_149000/checkpoints/counter_00000000.pt"
    )
    initial_bc, _ = runner.load_checkpoint(
        output / runner.IMITATION / "train_seed_149000/checkpoints/counter_00000000.pt"
    )
    assert all(
        torch.equal(v, initial_bc.state_dict()[k])
        for k, v in direct.state_dict().items()
    )
    totals = {}
    env = None
    last = None
    decisions = list(csv.DictReader((output / "decisions.csv").open()))
    assert len(decisions) == 720
    for row in decisions:
        key = (
            row["controller"],
            int(row["train_seed"]),
            int(row["cohort"]),
            row["capacity"],
            int(row["horizon"]),
            int(row["seed"]),
        )
        if key != last:
            env = WaitingEnv(cohort_config(key[2], key[3], key[4]))
            env.reset(key[5])
            last = key
        pairs = tuple(tuple(x) for x in json.loads(row["action"]))
        if key[0] in models:
            chosen, diag = models[key[0]].select(
                env.config, compact(env.state), key[4] - int(row["time"])
            )
            assert physical_action(env.state, chosen) == pairs and diag == json.loads(
                row["controller_diagnostics"]
            )
        _, _, _, info = env.step(pairs)
        assert info == json.loads(row["physical_metrics"])
        assert asdict_state(env.state) == json.loads(row["following"])
        totals[key] = env.metrics["objective"]
    for row in csv.DictReader((output / "episodes.csv").open()):
        key = (
            row["controller"],
            int(row["train_seed"]),
            int(row["cohort"]),
            row["capacity"],
            int(row["horizon"]),
            int(row["seed"]),
        )
        assert float(row["objective"]) == totals[key]
    refs = json.loads((output / "reference_selection.json").read_text())["selected"]
    last = None
    teacher = list(csv.DictReader((output / "teacher_rows.csv").open()))
    assert len(teacher) == 80
    for row in teacher:
        key = (row["split"], int(row["episode"]))
        c = cohort_config(int(row["config_seed"]), row["capacity"], 4)
        if key != last:
            env = WaitingEnv(c)
            env.reset(int(row["environment_seed"]))
            last = key
        state = compact(env.state)
        assert [list(r) for r in state] == json.loads(row["state"])
        chosen, diag = plan(
            c,
            state,
            int(row["remaining_horizon"]),
            refs[row["capacity"]],
            int(row["planner_seed"]),
            4,
        )
        assert json.loads(json.dumps(diag)) == json.loads(row["teacher_diagnostics"])
        executed = tuple(json.loads(row["executed_subset"]))
        _, _, _, info = env.step(physical_action(env.state, executed))
        assert info == json.loads(row["physical_metrics"])
        assert [list(r) for r in compact(env.state)] == json.loads(row["following"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["primary"]["status"] == "ENGINEERING_ONLY"
    assert (output / "episodes.csv").read_bytes() == (
        output / "episodes.partial.csv"
    ).read_bytes()
    assert len(list(csv.DictReader((output / "machine_metrics.csv").open()))) == 576
    with pytest.raises(ValueError, match="new/empty"):
        runner.run(output, "smoke")


def asdict_state(state):
    from dataclasses import asdict

    return json.loads(json.dumps(asdict(state)))


def test_failure_preserves_partial_manifest(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected BC failure")

    monkeypatch.setattr(runner, "fit_imitation", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected BC"):
        runner.run(output, "smoke")
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "FAILED" and "injected BC" in manifest["error"]
    assert (output / "teacher_rows.csv").exists()
    assert list(output.rglob("*.pt"))


def test_full_shape_teacher_and_bc_use_smoke_seed_only(tmp_path):
    cfg = runner.settings("full")
    cfg.update(
        teacher_episodes=16,
        teacher_validation_episodes=2,
        bc_epochs=1,
        bc_checkpoints=[1],
        noise_probes=2,
    )
    _, forbidden = runner.seed_audit(cfg)
    js = journals(tmp_path)
    try:
        data, coverage = training.collect(
            cfg,
            149000,
            forbidden,
            dict.fromkeys(("low", "middle", "high"), "independent_dp_matching"),
            js,
        )
        assert data["train"]["features"]["machines"].shape == (192, 4, 8)
        assert data["train"]["targets"].shape == (192, 16)
        assert coverage["physical_steps"] == 216 and coverage["noise_probes"] == 2
        model = runner.model_for(cfg, 149000)
        record = training.fit_imitation(model, cfg, 149000, data, js, lambda *a: None)
        assert record["optimizer_steps"] == 1 and record["critic_head_unchanged"]
        diagnostic = training.diagnostics(model, data["validation"])
        assert diagnostic["choice_opportunity_rows"] > 0
        assert diagnostic["choice_mean_teacher_score_regret"] >= 0
    finally:
        for j in js.values():
            j.close()
