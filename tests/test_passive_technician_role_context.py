"""Information intervention, optimal-policy feasibility and handoff artifacts."""
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch
from ortools.sat.python import cp_model

from ht_pdm_fjsp import passive_technician_role_context as experiment
from ht_pdm_fjsp import passive_technician_oracle_representation as oracle_exp
from ht_pdm_fjsp.passive_technician_value_decomposition import PassiveValueDecomposition, TechnicianAwareMixer


def config():
    return oracle_exp.cells("smoke")["shared_preference"]


def test_feature_toggles_only_reveal_declared_information():
    local = torch.arange(30, dtype=torch.float32).view(1, 2, 15)
    for role, peer in experiment.FLAGS.values():
        x = experiment.conditioned_inputs(local, role, peer)
        assert x.shape == (1, 2, 32)
        assert torch.equal(x[..., :15], local)
        assert torch.equal(x[..., 15:17], torch.eye(2)[None] if role else torch.zeros(1, 2, 2))
        assert torch.equal(x[..., 17:], local.flip(1) if peer else torch.zeros_like(local))
        changed = local.clone(); changed[:, 1] += 100
        changed_x = experiment.conditioned_inputs(changed, role, peer)
        assert torch.equal(x[:, 0], changed_x[:, 0]) == (not peer)
    with pytest.raises(ValueError):
        experiment.conditioned_inputs(torch.zeros(1, 3, 15), True, True)


def test_parameters_initialization_and_mixer_state_are_identical():
    cfg, seed = experiment.settings("smoke"), 27
    baseline = experiment.model_for("monotonic_base", config(), cfg, seed)
    for arm in experiment.FLAGS:
        model = experiment.model_for(arm, config(), cfg, seed)
        assert isinstance(model.mixer, TechnicianAwareMixer)
        assert not model.use_counterfactual
        assert set(model.state_dict()) == set(baseline.state_dict())
        assert all(torch.equal(v, model.state_dict()[k]) for k, v in baseline.state_dict().items())
    torch.manual_seed(seed)
    original = PassiveValueDecomposition(config(), "tqmix_no_cf", cfg["hidden"], cfg["mixer_hidden"])
    assert all(torch.equal(v, baseline.mixer.state_dict()[k]) for k, v in original.mixer.state_dict().items())


def test_peer_does_not_leak_into_local_agent_when_disabled():
    cfg = experiment.settings("smoke")
    local = torch.zeros(1, 2, 15)
    changed = local.clone(); changed[:, 1] = 1.
    for arm in ("monotonic_base", "monotonic_role"):
        model = experiment.model_for(arm, config(), cfg, 27)
        assert torch.equal(model.agent_q(local)[:, 0], model.agent_q(changed)[:, 0])


def test_identity_can_break_symmetry_that_peer_alone_cannot():
    cfg = experiment.settings("smoke")
    local = torch.ones(1, 2, 15)
    for arm in ("monotonic_base", "monotonic_context"):
        model = experiment.model_for(arm, config(), cfg, 27)
        assert torch.equal(model.agent_q(local)[:, 0], model.agent_q(local)[:, 1])
    model = experiment.model_for("monotonic_role", config(), cfg, 27)
    with torch.no_grad():
        for parameter in model.agent.parameters():
            parameter.zero_()
        model.agent.trunk[0].weight[0, 15] = 1.
        model.agent.trunk[2].weight[0, 0] = 1.
        model.agent.defer.weight[0, 0] = 2.
        model.agent.defer.bias.fill_(-1.)
    # Identity enables different masked greedy decisions under identical locals.
    assert model.agent_q(local).argmax(-1).tolist() == [[0, 1]]


def fake_oracle(nodes):
    keys = [(i, str(i)) for i in range(len(nodes))]
    return SimpleNamespace(config=config(), keys=keys, key_to_id={k: i for i, k in enumerate(keys)},
                           nodes=dict(zip(keys, nodes)))


def test_exact_same_input_conflict_is_resolved_by_identity():
    obs = (0,)*15
    oracle = fake_oracle([dict(observations=(obs, obs), masks=((True,)*3,)*2,
                              actions=[(0, 0), (1, 2), (2, 1)], q95=[0., 1., 1.])])
    for arm in ("monotonic_base", "monotonic_context"):
        result = experiment.policy_feasibility(oracle, arm, 96100)
        assert result["status"] == "INFEASIBLE"
        assert result["method"] == "direct_same_input_conflict"
    for arm in ("monotonic_role", "monotonic_role_context"):
        result = experiment.policy_feasibility(oracle, arm, 96100)
        assert result["status"] in ("OPTIMAL", "FEASIBLE")
        assert result["assignment_verified"]


def test_identity_alone_cannot_resolve_hidden_peer_state_conflict():
    own = (0,)*15
    nodes = [dict(observations=(own, (value,)*15), masks=((True,)*3,)*2,
                  actions=[(action, 0)], q95=[1.]) for value, action in ((1, 1), (2, 2))]
    oracle = fake_oracle(nodes)
    role_only = experiment.policy_feasibility(oracle, "monotonic_role", 96100)
    both = experiment.policy_feasibility(oracle, "monotonic_role_context", 96100)
    assert role_only["status"] == "INFEASIBLE"
    assert role_only["method"] == "exact_finite_constraint_system"
    assert both["status"] in ("OPTIMAL", "FEASIBLE")
    assert both["assignment_verified"]


def test_unknown_solver_is_not_reported_as_infeasible():
    own = (0,)*15
    oracle = fake_oracle([dict(observations=(own, own), masks=((True,)*3,)*2,
                              actions=[(1, 2)], q95=[1.])])
    class UnknownSolver:
        def __init__(self): self.parameters = SimpleNamespace()
        def Solve(self, model): return cp_model.UNKNOWN
        def StatusName(self, status): return "UNKNOWN"
        def WallTime(self): return 30.
    result = experiment.policy_feasibility(oracle, "monotonic_role_context", 96100, solver_factory=UnknownSolver)
    assert result["status"] == "UNKNOWN"
    assert result["assignment_verified"] is None
    assert "witness_policy" not in result


def test_actual_oracle_full_information_lookup_is_verified():
    oracle = oracle_exp.ExactOracle(oracle_exp.cells("smoke")["pressure"], progress=False)
    result = experiment.policy_feasibility(oracle, "monotonic_role_context", 96102)
    assert result["assignment_verified"]
    assert result["state_count"] == len(oracle.keys)


def test_full_joint_capacity_and_seed_panels():
    cfg = experiment.settings("full")
    mono = experiment.model_for("monotonic_base", config(), cfg, 96000)
    joint = experiment.model_for("joint_q", config(), cfg, 96000)
    count = lambda m: sum(p.numel() for p in m.parameters())
    assert abs(count(joint)/count(mono)-1) <= .05
    assert experiment.seed_audit(cfg)["prior_panels_disjoint"]
    cfg["evaluation_seeds"] = cfg["train_seeds"]
    with pytest.raises(ValueError, match="overlap"):
        experiment.seed_audit(cfg)


def good_metrics():
    cfg = experiment.settings("full")
    regrets = dict(monotonic_base=1., monotonic_role=.6, monotonic_context=.9, monotonic_role_context=.3, joint_q=.01)
    costs = dict(monotonic_base=5., monotonic_role=4.5, monotonic_context=4.8, monotonic_role_context=4.4, joint_q=4.)
    return cfg, [dict(cell=cell, algorithm=arm, train_seed=seed, heldout_regret=regrets[arm],
                      undiscounted_policy_cost=4. if cell == "nominal" else costs[arm],
                      undiscounted_policy_gap=.1, heldout_normalized_rmse=.5)
                 for seed in cfg["train_seeds"] for cell in ("nominal", "shared_preference", "pressure") for arm in experiment.ARMS]


def test_two_primary_contrasts_and_policy_control_gates():
    cfg, rows = good_metrics()
    paired, summary = experiment.summarize(rows, cfg, True)
    assert len(paired) == 10
    assert summary["primary"]["role"]["hypothesis_passed"]
    assert summary["primary"]["peer_given_role"]["hypothesis_passed"]
    assert summary["positive_control"]["passed"]  # RMSE is descriptive in this policy diagnostic.
    assert all(summary["empirical_attribution_supported"].values())
    for row in rows:
        if row["algorithm"] == "joint_q" and row["cell"] == "pressure": row["heldout_regret"] = .11
    summary = experiment.summarize(rows, cfg, True)[1]
    assert not summary["positive_control"]["passed"]
    assert not any(summary["empirical_attribution_supported"].values())


def test_nominal_and_trajectory_preservation_are_not_replaced_by_state_regret():
    cfg, rows = good_metrics()
    for row in rows:
        if row["algorithm"] == "monotonic_role_context" and row["cell"] == "nominal":
            row["undiscounted_policy_cost"] = 5.
    assert not experiment.summarize(rows, cfg, True)[1]["primary"]["peer_given_role"]["hypothesis_passed"]
    cfg, rows = good_metrics()
    for row in rows:
        if row["algorithm"] == "monotonic_role_context" and row["cell"] == "pressure":
            row["undiscounted_policy_cost"] = 50.
    assert not experiment.summarize(rows, cfg, True)[1]["primary"]["peer_given_role"]["hypothesis_passed"]


def test_mini_runner_saves_reload_information_and_disables_scientific_gate(monkeypatch, tmp_path):
    original = experiment.settings
    monkeypatch.setattr(experiment, "settings", lambda profile: dict(original(profile), updates=2, log_interval=1))
    output = tmp_path/"run"
    args = Namespace(profile="smoke", device="cpu", output_dir=str(output))
    experiment.run(args)
    manifest = json.loads((output/"manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["actual_models"] == 15
    assert manifest["actual_evaluation_rows"] == 45
    assert manifest["actual_optimizer_updates"] == 30
    assert len(list(output.rglob("model.pt"))) == 15
    summary = json.loads((output/"summary.json").read_text())
    assert summary["primary"]["role"]["hypothesis_passed"] is None
    assert summary["positive_control"]["passed"] is None
    assert summary["audits"]["verified_sat_assignments"]
    assert all((output/p).is_file() for p in manifest["outputs"])
    checkpoint = next(output.glob("*/monotonic_role_context/*/model.pt"))
    payload = torch.load(checkpoint, weights_only=True)
    payload["information_flags"] = [False, False]
    torch.save(payload, tmp_path/"bad.pt")
    with pytest.raises(ValueError, match="information flags"):
        experiment.load_checkpoint(tmp_path/"bad.pt")
    with pytest.raises(FileExistsError):
        experiment.run(args)


def test_failed_training_preserves_outputs_and_does_not_retry(monkeypatch, tmp_path):
    calls = []
    def fail(*args):
        calls.append(True)
        raise RuntimeError("injected failure")
    monkeypatch.setattr(oracle_exp, "fit", fail)
    output = tmp_path/"failed"
    with pytest.raises(RuntimeError, match="injected"):
        experiment.run(Namespace(profile="smoke", device="cpu", output_dir=str(output)))
    assert len(calls) == 1
    assert json.loads((output/"manifest.json").read_text())["status"] == "FAILED"
    assert (output/"nominal/policy_feasibility.json").is_file()
