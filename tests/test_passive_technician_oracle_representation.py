"""Tests for exact labels, structural certificates and the artifact contract."""
import itertools
import json
from argparse import Namespace
from dataclasses import replace
from functools import lru_cache

import numpy as np
import pytest
import torch

from ht_pdm_fjsp import passive_technician_oracle_representation as experiment
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig, PassiveState, PassiveTechnicianEnv


def tiny_config():
    return PassiveConfig(machines=2, technicians=2, horizon=2, failure_age=1,
                         failure_probability=.4, max_age=3, service_time=((1, 2), (2, 1)))


def brute_value(config, gamma):
    """Independent uncanonicalized expectation, feasible complete action tree."""
    @lru_cache(None)
    def solve(time, state):
        if time == config.horizon:
            return 0.
        env = PassiveTechnicianEnv(config)
        env.time, env.state = time, state
        actions = itertools.product(*[[i for i, b in enumerate(mask) if b] for mask in env.action_masks()])
        values = []
        for action in actions:
            value = 0.
            for events in itertools.product((False, True), repeat=config.machines):
                probability = 1.
                for machine, event in enumerate(events):
                    eligible = not state.failed[machine] and state.ages[machine]+1 >= config.failure_age
                    probability *= (config.failure_probability if event else 1-config.failure_probability) if eligible else float(not event)
                if probability:
                    nxt, cost, _ = env.transition(state, action, time, events)
                    value += probability*(-cost+gamma*solve(time+1, nxt))
            values.append(value)
        return max(values)
    return solve(0, PassiveTechnicianEnv(config).initial_state())


@pytest.mark.parametrize("gamma,name", [(1., "q1"), (.95, "q95")])
def test_oracle_matches_independent_bruteforce_and_policy_expectation(gamma, name):
    cfg = tiny_config()
    oracle = experiment.ExactOracle(cfg, progress=False)
    value = max(oracle.nodes[oracle.initial][name])
    assert value == pytest.approx(brute_value(cfg, gamma), abs=1e-10)
    optimal_indices = {k: int(np.argmax(n[name])) for k, n in oracle.nodes.items()}
    assert oracle.policy_cost(optimal_indices, gamma) == pytest.approx(-value, abs=1e-10)
    assert oracle.bellman_residual() < 1e-10


def test_stochastic_failure_support_and_eligibility():
    cfg = tiny_config()
    state = PassiveTechnicianEnv(cfg).initial_state()
    outcomes = dict(experiment.failure_outcomes(cfg, state))
    assert outcomes[(True, True)] == pytest.approx(.16)
    assert outcomes[(False, False)] == pytest.approx(.36)
    assert sum(outcomes.values()) == pytest.approx(1.)
    failed = replace(state, failed=(True, False))
    assert all(not events[0] for events, _ in experiment.failure_outcomes(cfg, failed))


def test_canonicalization_preserves_observations_masks_and_future():
    cfg = tiny_config()
    old = PassiveState((1, 0), (False, False), (0, 4), (-1, 1), ((0,), ()))
    new = experiment.canonical(2, old)
    env = PassiveTechnicianEnv(cfg)
    env.time, env.state = 2, old
    obs, masks = env.observations(), env.action_masks()
    env.state = new
    assert env.observations() == obs
    assert env.action_masks() == masks
    for events, _ in experiment.failure_outcomes(cfg, old):
        a = env.transition(old, (0, 0), 2, events)
        b = env.transition(new, (0, 0), 2, events)
        assert experiment.canonical(3, a[0]) == experiment.canonical(3, b[0])
        assert a[1:] == b[1:]


def test_strict_rank_reversal_certificate_and_no_false_positive():
    actions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    witness = experiment.rank_reversal(actions, [0., 3., 2., 1.])
    assert witness["positive_context"]["delta"] > 0
    assert witness["negative_context"]["delta"] < 0
    assert witness["minimum_reversal_margin"] == 2.
    assert experiment.rank_reversal(actions, [0., 1., 2., 3.]) is None
    assert experiment.rank_reversal(actions, [0., 0., 0., 0.]) is None


def test_observation_alias_targets_rejected():
    oracle = experiment.ExactOracle(tiny_config(), progress=False)
    initial_node = oracle.nodes[oracle.initial]
    other = next(k for k in oracle.keys if k != oracle.initial)
    oracle.nodes[other] = dict(initial_node, q95=[q+1 for q in initial_node["q95"]])
    with pytest.raises(RuntimeError, match="aliases"):
        oracle.audit_observations()


def test_split_is_by_state_and_target_statistics_are_train_only():
    oracle = experiment.ExactOracle(experiment.cells("smoke")["pressure"], progress=False)
    data = experiment.dataset(oracle, 95100)
    indices = set(data["train_indices"].tolist())
    for key, block in data["slices"].items():
        assert {i in indices for i in range(block.start, block.stop)} == {data["split"][key] == "train"}
    assert float(data["y"][data["train_indices"]].mean()) == pytest.approx(0., abs=1e-6)
    assert float(data["y"][data["train_indices"]].std(unbiased=False)) == pytest.approx(1., abs=1e-6)
    assert set(data["split"].values()) == {"train", "heldout"}


def test_signed_changes_constraint_without_changing_parameters_or_information():
    config, cfg = tiny_config(), experiment.settings("smoke")
    mono = experiment.model_for("monotonic_no_cf", config, cfg, 19)
    signed = experiment.model_for("signed_no_cf", config, cfg, 19)
    for k, v in mono.state_dict().items():
        assert torch.equal(v, signed.state_dict()[k])
    # Force one negative first-layer weight: signed can have a negative derivative.
    for mixer in (mono.mixer, signed.mixer):
        with torch.no_grad():
            mixer.hyper_w1.weight.zero_(); mixer.hyper_w1.bias.fill_(-1.)
            mixer.hyper_w2.weight.zero_(); mixer.hyper_w2.bias.fill_(1.)
            mixer.hyper_b1.weight.zero_(); mixer.hyper_b1.bias.zero_()
    state = torch.zeros(1, config.machines*config.observation_dim)
    for mixer, positive in ((mono.mixer, True), (signed.mixer, False)):
        q = torch.zeros(1, config.machines, requires_grad=True)
        grad = torch.autograd.grad(mixer(q, state).sum(), q)[0]
        assert bool((grad > 0).all()) == positive
        assert bool((grad < 0).all()) != positive


def test_joint_capacity_matching_and_action_indexing():
    config, cfg = tiny_config(), experiment.settings("full")
    mono = experiment.model_for("monotonic_no_cf", config, cfg, 19)
    joint = experiment.model_for("joint_q", config, cfg, 19)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert abs(count(joint)/count(mono)-1) <= .05
    x = torch.zeros(2, config.machines, config.observation_dim)
    actions = torch.tensor([[1, 2], [2, 1]])
    direct = joint.network(x.flatten(start_dim=1))
    assert torch.equal(joint.total_q(x, actions), direct[torch.arange(2), torch.tensor([5, 7])])


def test_gate_uses_training_seeds_not_states_and_control_is_required():
    cfg = experiment.settings("full")
    rows = [dict(train_seed=seed, algorithm=arm, heldout_regret=.5 if arm == "monotonic_no_cf" else .05,
                 heldout_normalized_rmse=.01) for seed in cfg["train_seeds"] for arm in experiment.ARMS]
    paired, summary = experiment.summarize(rows, cfg, {})
    assert len(paired) == 10
    assert summary["primary"]["hypothesis_passed"]
    assert summary["empirical_attribution_supported"]
    for r in rows:
        if r["algorithm"] == "joint_q":
            r["heldout_normalized_rmse"] = 1.
    assert not experiment.summarize(rows, cfg, {})[1]["empirical_attribution_supported"]


def test_seed_freshness_and_disjointness():
    assert experiment.seed_audit(experiment.settings("full"))["prior_panels_disjoint"]
    cfg = experiment.settings("smoke")
    cfg["evaluation_seeds"] = cfg["train_seeds"]
    with pytest.raises(ValueError, match="overlap"):
        experiment.seed_audit(cfg)


def test_small_runner_artifacts_checkpoint_and_overwrite(monkeypatch, tmp_path):
    original = experiment.settings
    monkeypatch.setattr(experiment, "settings", lambda profile: dict(original(profile), updates=2, log_interval=1))
    output = tmp_path / "run"
    args = Namespace(profile="smoke", device="cpu", output_dir=str(output))
    experiment.run(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["actual_models"] == 12
    assert manifest["actual_evaluation_rows"] == 36
    assert len(list(output.rglob("model.pt"))) == 12
    summary = json.loads((output / "summary.json").read_text())
    assert summary["primary"]["hypothesis_passed"] is None
    assert summary["audits"]["zero_invalid_requests"]
    assert all((output / path).is_file() for path in manifest["outputs"])
    with pytest.raises(FileExistsError):
        experiment.run(args)


def test_failure_preserves_manifest_and_does_not_retry(monkeypatch, tmp_path):
    calls = []
    def fail(*args):
        calls.append(True)
        raise RuntimeError("injected training failure")
    monkeypatch.setattr(experiment, "fit", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="injected"):
        experiment.run(Namespace(profile="smoke", device="cpu", output_dir=str(output)))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "FAILED"
    assert manifest["actual_models"] == 0
    assert len(calls) == 1
    assert (output / "nominal/oracle_q.csv").is_file()
