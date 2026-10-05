import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace

import pytest
import torch

from ht_pdm_fjsp import maintenance_dispatch as physics
from ht_pdm_fjsp import maintenance_dispatch_policy as bp
from ht_pdm_fjsp import maintenance_waiting as env
from ht_pdm_fjsp import maintenance_context_waiting as study
from ht_pdm_fjsp.maintenance_context_policy import ContextActorCritic


def config(price=12.0):
    return env.WaitingConfig(
        2,
        2,
        6,
        3,
        0.5,
        ((2, 2), (1, 2)),
        ((0.75, 0.75), (1.0, 0.5)),
        waiting_price=price,
    )


def failed_state(wait=4):
    return physics.DispatchState((4, 0), (True, False), (wait, 0), (0, 0), (-1, -1))


def test_overdue_cost_boundary_and_service_start():
    cfg = config()
    before = failed_state(3)
    following, cost, info = env.transition(cfg, before, (), (False, False))
    assert following.pending_wait == (4, 0)
    assert cost == 6 and info["overdue_waiting_ticks"] == 0
    following, cost, info = env.transition(cfg, following, (), (False, False))
    assert following.pending_wait == (5, 0)
    assert cost == 18 and info["waiting_cost"] == 12 and info["base_cost"] == 6
    following, cost, info = env.transition(
        cfg, failed_state(), ((0, 0),), (False, False)
    )
    assert (
        cost == 8
        and info["overdue_waiting_ticks"] == 0
        and info["failed_waiting_ticks"] == 0
    )
    assert following.failed[0] and following.remaining[0] == 1
    following, cost, info = env.transition(cfg, following, (), (False, False))
    assert cost == 6 and info["waiting_cost"] == 0 and not following.failed[0]
    assert following.pending_wait[0] == 0 and following.ages[0] == 1


def test_new_failure_does_not_charge_waiting_early():
    cfg = config()
    state = physics.DispatchState((2, 0), (False, False), (0, 0), (0, 0), (-1, -1))
    following, cost, info = env.transition(cfg, state, (), (True, False))
    assert following.failed[0] and following.pending_wait[0] == 0
    assert cost == 15 and info["overdue_waiting_ticks"] == 0


def test_common_adjusted_score_is_not_training_objective():
    _, cost, info = env.transition(config(0), failed_state(), (), (False, False))
    assert cost == info["objective"] == info["base_cost"] == 6
    assert info["waiting_cost"] == 0 and info["service_adjusted_cost"] == 18


@pytest.mark.parametrize("wait,price", [(-1, 0), (1, -1), (1, float("inf")), (1.5, 0)])
def test_invalid_waiting_contract(wait, price):
    with pytest.raises(ValueError, match="waiting"):
        replace(config(), waiting_limit=wait, waiting_price=price).validate()


def test_zero_price_keeps_physics_and_events_exactly():
    cfg = config(0)
    b = physics.DispatchEnv(cfg)
    w = env.WaitingEnv(cfg)
    b.reset(991, failed_state())
    obs = w.reset(991, failed_state())
    assert b.events == w.events and b.state == w.state
    for pairs in [(), ((0, 1),), (), ((1, 0),), (), ()]:
        _, reward_b, done_b, info_b = b.step(pairs)
        obs, reward_w, done_w, info_w = w.step(pairs)
        assert b.state == w.state and reward_b == reward_w and done_b == done_w
        assert all(info_w[k] == v for k, v in info_b.items())
    assert len(obs["global_features"]) == 9 and obs["global_features"][-1] == 0
    assert w.metrics["objective"] == b.metrics["objective"]


def test_full_smoke_seeds_counts_and_long_horizon_contract():
    for profile in ("full", "smoke"):
        cfg = study.settings(profile)
        assert all(
            study.seed_audit(cfg)[k]
            for k in (
                "panels_disjoint",
                "prior_declared_disjoint",
                "sealed_panels_closed",
            )
        )
    cfg = study.settings("full")
    assert 4 * len(cfg["train_seeds"]) * cfg["env_steps"] == 4800000
    assert env.family_config("n4_pressure_long", 124000, "full", 12).horizon == 24
    assert env.family_config("n4_pressure", 124000, "full", 12).horizon == 12
    assert set(study.ARMS) == {
        "local_cost",
        "context_cost",
        "local_wait",
        "context_wait",
    }
    for seed in (800, 801):
        assert asdict(env.training_config(seed, "full", 0)) | {
            "waiting_price": 12
        } == asdict(env.training_config(seed, "full", 12))
    cfg["evaluation_seeds"] = [116000]
    with pytest.raises(ValueError, match="historical"):
        study.seed_audit(cfg)


def test_context_can_distinguish_identical_own_edges_and_equivariance():
    obs = env.observation(
        config(),
        physics.DispatchState((4, 0), (False, False), (0, 0), (0, 0), (-1, -1)),
        0,
    )
    batch = bp.batch_observations([obs])
    used_m = torch.zeros(1, 2, dtype=torch.bool)
    used_t = torch.zeros(1, 2, dtype=torch.bool)
    mask = bp.feasible(batch, used_m, used_t)
    torch.manual_seed(17)
    local = ContextActorCritic(8, technician_context=False)
    context = ContextActorCritic(8, technician_context=True)
    context.load_state_dict(local.state_dict())
    encoded = local.encode(batch)
    args = (batch, mask, torch.tensor([0]), torch.tensor([True]), torch.zeros(1, 8))
    probabilities_local = local.technician_distribution(encoded, *args).probs
    probabilities_context = context.technician_distribution(encoded, *args).probs
    assert torch.allclose(
        probabilities_local, torch.tensor([[0.5, 0.5]]), atol=1e-7, rtol=0
    )
    assert abs(float((probabilities_context[0, 0] - 0.5).detach())) > 1e-7
    permuted = {
        **batch,
        "technicians": batch["technicians"][:, [1, 0]],
        "edges": batch["edges"][:, :, [1, 0]],
    }
    reversed_p = context.technician_distribution(
        context.encode(permuted),
        permuted,
        mask[:, :, [1, 0]],
        torch.tensor([0]),
        torch.tensor([True]),
        torch.zeros(1, 8),
    ).probs
    assert torch.allclose(
        reversed_p, probabilities_context[:, [1, 0]], atol=1e-7, rtol=1e-6
    )
    assert sum(p.numel() for p in local.parameters()) == sum(
        p.numel() for p in context.parameters()
    )


def test_near_ties_choose_first_finite_index_only_at_deployment():
    model = ContextActorCritic(8)
    logits = torch.tensor([[-float("inf"), 2.0, 2.0000005], [-3.0, -2.0, -1.0]])
    assert model.deterministic_choice(logits).tolist() == [1, 2]
    assert bp.DispatchActorCritic(8).deterministic_choice(logits).tolist() == [2, 2]
    with pytest.raises(RuntimeError, match="finite"):
        model.deterministic_choice(torch.full((1, 2), -float("inf")))


@pytest.mark.parametrize("arm", study.ARMS)
def test_sampling_replay_has_valid_matching_and_gradients(arm):
    torch.set_num_threads(1)
    _, price = study.factors(arm)
    cfg = study.settings("smoke")
    model = study.model_for(cfg, 125000, arm)
    obs = env.WaitingEnv(
        env.family_config("n4_pressure", 125010, "smoke", price)
    ).reset(125010)
    pairs, tokens, logprob, value = model.act(
        obs, "learned_both", generator=torch.Generator().manual_seed(77)
    )
    sequence = bp.batch_sequences([tokens])
    _, replayed, entropy, replayed_value = model.decode(
        bp.batch_observations([obs]), "learned_both", sequence
    )
    assert float(replayed[0].detach()) == pytest.approx(logprob, abs=1e-5)
    assert float(replayed_value[0].detach()) == pytest.approx(value, abs=1e-5)
    physics.validate_matching(
        env.family_config("n4_pressure", 125010, "smoke", price),
        env.WaitingEnv(env.family_config("n4_pressure", 125010, "smoke", price)).state,
        pairs,
    )
    (
        -replayed.mean() + replayed_value.square().mean() - entropy.mean() * 0.01
    ).backward()
    assert all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )


def synthetic():
    cfg = study.settings("full")
    cfg["evaluation_seeds"] = cfg["evaluation_seeds"][:1]
    cfg["development_seeds"] = cfg["development_seeds"][:1]
    episodes, machines, refs = [], [], []
    for arm in study.ARMS:
        economic = (
            90.0
            if arm == "context_cost"
            else (99.0 if arm == "context_wait" else 100.0)
        )
        overdue = 0.1 if arm == "context_wait" else 1.0
        for seed in cfg["train_seeds"]:
            for f in env.FAMILIES:
                key = dict(
                    algorithm=arm,
                    train_seed=seed,
                    family=f,
                    eval_seed=cfg["evaluation_seeds"][0],
                )
                episodes.append(
                    dict(
                        **key,
                        base_cost=economic,
                        objective=economic,
                        waiting_cost=0.0,
                        service_adjusted_cost=economic + 12 * overdue,
                        overdue_ticks_per_machine=overdue,
                        failed_waiting_ticks=overdue,
                        terminal_failed_never_serviced_fraction=0.0,
                        wait_violation_machine_fraction=0.0,
                    )
                )
                for m in range(2 if f == "small" else int(f[1])):
                    machines.append(
                        dict(
                            **key,
                            machine=m,
                            max_pending_wait=4 if arm == "context_wait" else 6,
                        )
                    )
    for f in (*env.FAMILIES, "development"):
        refs.append(
            dict(
                family=f,
                eval_seed=(
                    cfg["development_seeds"]
                    if f == "development"
                    else cfg["evaluation_seeds"]
                )[0],
                base_cost=100.0,
            )
        )
    return cfg, episodes, machines, refs


def test_hypotheses_are_separate_and_base_cost_is_primary():
    cfg, eps, machines, refs = synthetic()
    pairs, summary = study.summarize(eps, machines, cfg, True, refs)
    assert len(pairs) == 20
    assert all(r["passed"] for r in summary["hypotheses"].values())
    # Changing the arm's reward objective must not alter economic inference.
    for r in eps:
        r["objective"] = 100000.0 if r["algorithm"] == "context_cost" else 0.0
    _, second = study.summarize(eps, machines, cfg, True, refs)
    assert second["hypotheses"] == summary["hypotheses"]
    for r in eps:
        if r["algorithm"] == "context_cost":
            r["base_cost"] = 96.0
    _, third = study.summarize(eps, machines, cfg, True, refs)
    assert not third["hypotheses"]["H_context"]["passed"]


@pytest.mark.parametrize(
    "failure", ["p95", "cost", "nominal", "zero_control", "uncertainty"]
)
def test_waiting_gate_does_not_hide_tradeoff_or_uncertainty(failure):
    cfg, eps, machines, refs = synthetic()
    for r in eps:
        if failure == "cost" and r["algorithm"] == "context_wait":
            r["base_cost"] = 101.0
        if (
            failure == "nominal"
            and r["algorithm"] == "context_wait"
            and r["family"] in study.NOMINAL
        ):
            r["base_cost"] = 101.0
        if failure == "zero_control" and r["algorithm"] == "context_cost":
            r["overdue_ticks_per_machine"] = 0.0
        if (
            failure == "uncertainty"
            and r["algorithm"] == "context_wait"
            and r["train_seed"] == 122000
        ):
            r["overdue_ticks_per_machine"] = 20.0
    if failure == "p95":
        for r in machines:
            if r["algorithm"] == "context_wait" and r["family"] == "n4_pressure":
                r["max_pending_wait"] = 5
    _, summary = study.summarize(eps, machines, cfg, True, refs)
    assert not summary["hypotheses"]["H_wait"]["passed"]


def test_incomplete_panels_are_rejected_and_smoke_has_no_inference():
    cfg, eps, machines, refs = synthetic()
    with pytest.raises(RuntimeError, match="evaluation"):
        study.summarize(eps[:-1], machines, cfg, True, refs)
    with pytest.raises(RuntimeError, match="machine"):
        study.summarize(eps, machines[:-1], cfg, True, refs)
    _, summary = study.summarize(eps, machines, cfg, False, refs)
    assert all(r["passed"] is None for r in summary["hypotheses"].values())


def test_small_oracle_policy_retains_own_price_and_base_floor():
    class Recorder:
        def act(self, obs, mode, **kwargs):
            assert obs["global_features"][-1] == pytest.approx(0.6)
            return (), [(-1, -1)], 0.0, 0.0

    cfg = physics.DispatchConfig(
        2, 2, 1, 1, 1.0, ((1, 1), (1, 1)), ((1.0, 1.0), (1.0, 1.0))
    )
    oracle = study.core.SmallOracle(cfg)
    result = oracle.evaluate("learned_both", study.OracleAdapter(Recorder(), 12))
    assert result["exact_optimum"] == 14 and result["exact_policy_cost"] == 30


def test_full_dirty_gate_and_nonempty_output_fail_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(study.helpers, "_git_state", lambda: ("sha", True))
    args = argparse.Namespace(
        profile="full", device="cpu", output_dir=str(tmp_path / "run")
    )
    with pytest.raises(RuntimeError, match="clean Git"):
        study.run(args)
    assert not (tmp_path / "run").exists()
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "old.txt").write_text("preserve")
    args.profile = "smoke"
    with pytest.raises(FileExistsError):
        study.run(args)
    assert (tmp_path / "run" / "old.txt").read_text() == "preserve"


def test_smoke_contract_checkpoint_and_trace_replay(tmp_path):
    output = study.run(
        argparse.Namespace(
            profile="smoke", device="cpu", output_dir=str(tmp_path / "smoke")
        )
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED" and all(manifest["audits"].values())
    assert manifest["actual_models"] == 4 and manifest["actual_training_steps"] == 192
    assert (
        manifest["actual_test_episodes"] == 96
        and manifest["actual_development_episodes"] == 4
    )
    assert (
        manifest["actual_reference_episodes"] == 25
        and manifest["actual_optimizer_steps"] == 32
    )
    assert manifest["actual_training_episodes"] == 48
    for name, digest in manifest["source_hashes"].items():
        assert (
            hashlib.sha256((output / "source_snapshot" / name).read_bytes()).hexdigest()
            == digest
        )
    for arm in study.ARMS:
        restored, payload = study.load_checkpoint(
            output / arm / "train_seed_125000" / "model.pt"
        )
        assert restored.technician_context == study.factors(arm)[0]
        assert payload["source_bundle_sha256"] == manifest["source_bundle_sha256"]
    for name in ("decisions.csv", "reference_decisions.csv"):
        grouped = {}
        with (output / name).open() as f:
            for row in csv.DictReader(f):
                key = row["algorithm"], row["family"], int(row["eval_seed"])
                grouped.setdefault(key, []).append(row)
        for (arm, family, ev), rows in grouped.items():
            price = 0 if arm == study.RULE else study.factors(arm)[1]
            cfg = env.family_config(family, ev, "smoke", price)
            simulator = env.WaitingEnv(cfg)
            simulator.reset(ev)
            for row in rows:
                assert tuple(json.loads(row["ages"])) == simulator.state.ages
                assert (
                    tuple(json.loads(row["pending_wait"]))
                    == simulator.state.pending_wait
                )
                pairs = tuple(tuple(p) for p in json.loads(row["pairs"]))
                _, _, _, info = simulator.step(pairs)
                for k in (
                    "objective",
                    "base_cost",
                    "waiting_cost",
                    "overdue_waiting_ticks",
                    "service_adjusted_cost",
                ):
                    assert info[k] == pytest.approx(float(row[k]), abs=1e-9)


def test_ppo_return_reconciliation_includes_positive_wait_penalties(tmp_path):
    cfg = study.settings("smoke")
    model = study.model_for(cfg, 125000, "context_wait")
    with torch.no_grad():
        model.stop_head[-1].bias.fill_(30.0)

    class OverdueEnv(env.WaitingEnv):
        def reset(self, seed, initial=None):
            n, k = self.config.machines, self.config.technicians
            initial = physics.DispatchState(
                (4,) * n, (True,) * n, (4,) * n, (0,) * k, (-1,) * k
            )
            return super().reset(seed, initial)

    _, coverage = study.core.fit(
        model,
        "learned_both",
        cfg,
        125000,
        "smoke",
        tmp_path,
        [],
        environment_factory=OverdueEnv,
        training_configuration=lambda s, p: env.training_config(s, p, 12),
        algorithm="context_wait",
    )
    with (tmp_path / "training_episodes.csv").open() as f:
        records = list(csv.DictReader(f))
    assert coverage["env_steps"] == 48
    assert all(float(r["waiting_cost"]) > 0 for r in records)
    assert all(
        float(r["objective"])
        == pytest.approx(float(r["base_cost"]) + float(r["waiting_cost"]))
        for r in records
    )
