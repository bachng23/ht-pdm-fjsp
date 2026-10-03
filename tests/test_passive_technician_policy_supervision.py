"""Consistent teachers, objective gradients, regimes, and artifact contracts."""
import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from ht_pdm_fjsp import passive_technician_policy_supervision as exp
from ht_pdm_fjsp import passive_technician_role_context as role
from ht_pdm_fjsp import passive_technician_oracle_representation as base
from ht_pdm_fjsp.passive_technician_value_decomposition import PassiveValueDecomposition


def fixture():
    config = base.cells("smoke")["shared_preference"]
    oracle = base.ExactOracle(config, progress=False)
    witness = role.policy_feasibility(oracle,"monotonic_role_context",97101)
    teacher = exp.teacher_labels(oracle,witness)
    original = base.dataset(oracle,97101)
    return config,oracle,witness,teacher,original


def test_teacher_consistency_optimality_and_rejection():
    _,oracle,witness,teacher,_ = fixture()
    assert len(teacher) == len(oracle.keys)
    for key,action in teacher.items():
        node = oracle.nodes[key]
        assert max(node["q95"])-node["q95"][node["actions"].index(action)] <= 1e-8
    with pytest.raises(RuntimeError,match="verified SAT"):
        exp.teacher_labels(oracle,dict(witness,status="UNKNOWN"))
    wrong = json.loads(json.dumps(witness))
    item = wrong["witness_policy"][0]
    wrong["witness_policy"].append(dict(item,action=(item["action"]+1)%3))
    with pytest.raises(RuntimeError,match="inconsistent"):
        exp.teacher_labels(oracle,wrong)


def test_teacher_rejects_nonoptimal_assignment_even_if_marked_verified():
    config = base.cells("smoke")["shared_preference"]
    obs = ((0,)*15,)*2
    key = (0,"state")
    oracle = SimpleNamespace(config=config,keys=[key],nodes={key:dict(observations=obs,masks=((True,)*3,)*2,
        actions=[(0,0),(1,2)],q95=[0.,1.])})
    witness = dict(status="OPTIMAL",assignment_verified=True,witness_policy=[
        dict(inputs=role.raw_feature_key(obs,i,True,True),action=0) for i in range(2)])
    with pytest.raises(RuntimeError,match="not oracle-optimal"):
        exp.teacher_labels(oracle,witness)


def test_split_normalization_and_training_indices_do_not_use_heldout_targets():
    _,oracle,_,teacher,original = fixture()
    split = exp.regime_data(original,oracle,teacher,"split")
    altered = dict(original,raw_y=original["raw_y"].clone())
    heldout = torch.ones(len(original["raw_y"]),dtype=torch.bool)
    heldout[original["train_indices"]] = False
    altered["raw_y"][heldout] += 10000
    split2 = exp.regime_data(altered,oracle,teacher,"split")
    assert split["mean"] == split2["mean"] and split["std"] == split2["std"]
    assert torch.equal(split["train_indices"],original["train_indices"])
    assert all(original["split"][oracle.keys[original["ids"][i]]] == "train" for i in split["train_indices"].tolist())
    ceiling = exp.regime_data(original,oracle,teacher,"ceiling")
    assert len(ceiling["train_indices"]) == len(original["raw_y"])
    assert ceiling["mean"] == original["raw_y"].mean().item()
    assert torch.equal(split["teacher"],ceiling["teacher"])


def test_objective_arms_same_network_initialization_and_original_mixer():
    config = base.cells("smoke")["nominal"]
    cfg = exp.settings("smoke")
    models = [exp.model_for(a,config,cfg,97400) for a in exp.ARMS[:3]]
    assert all(torch.equal(v,m.state_dict()[k]) for m in models for k,v in models[0].state_dict().items())
    torch.manual_seed(97400)
    original = PassiveValueDecomposition(config,"tqmix_no_cf",cfg["hidden"],cfg["mixer_hidden"])
    assert all(torch.equal(v,models[0].mixer.state_dict()[k]) for k,v in original.mixer.state_dict().items())
    for a in exp.ARMS:
        m = exp.model_for(a,config,exp.settings("full"),97000)
        reference = exp.model_for("monotonic_q",config,exp.settings("full"),97000)
        assert abs(sum(p.numel() for p in m.parameters())/sum(p.numel() for p in reference.parameters())-1) <= .05


@pytest.mark.parametrize("arm",exp.ARMS)
def test_objectives_mask_teacher_and_route_gradients(arm):
    config,oracle,_,teacher,original = fixture()
    data = exp.regime_data(original,oracle,teacher,"ceiling")
    model = exp.model_for(arm,config,exp.settings("smoke"),97400)
    selected = torch.arange(min(64,len(data["y"])))
    loss,mse,ce = exp.objective(model,data,selected,arm)
    assert torch.isfinite(loss)
    assert torch.equal(loss,mse+ce)
    loss.backward()
    if arm != "joint_q":
        assert any(p.grad is not None and torch.count_nonzero(p.grad)>0 for p in model.agent.parameters())
        assert any(p.grad is not None for p in model.mixer.parameters()) == (arm != "monotonic_policy")
    if arm in ("monotonic_policy","monotonic_hybrid"):
        bad = dict(data,masks=data["masks"].clone())
        bad["masks"][0,0,data["teacher"][0,0]] = False
        with pytest.raises(RuntimeError,match="masked-out"):
            exp.objective(model,bad,torch.tensor([0]),arm)


def good_metrics():
    cfg = exp.settings("full")
    rows = []
    for c in ("nominal",*role.DIAGNOSTIC_CELLS):
        for regime in exp.REGIMES:
            for a in exp.ARMS:
                for s in cfg["train_seeds"]:
                    rows.append(dict(cell=c,regime=regime,algorithm=a,train_seed=s,
                        all_regret=1. if a == "monotonic_q" else .01,
                        all_oracle_optimal=1.,undiscounted_policy_cost=4.,undiscounted_policy_gap=.1))
    return cfg,rows


def test_primary_gates_and_control_keep_independent_meanings():
    cfg,rows = good_metrics()
    paired,summary = exp.summarize(rows,cfg,True)
    assert len(paired) == 10
    assert summary["primary"]["learnability"]["passed"]
    assert summary["primary"]["objective"]["hypothesis_passed"]
    assert summary["empirical_objective_attribution_supported"]
    for row in rows:
        if row["regime"] == "ceiling" and row["algorithm"] == "joint_q" and row["cell"] == "pressure":
            row["undiscounted_policy_gap"] = 2.1
    summary = exp.summarize(rows,cfg,True)[1]
    assert not summary["positive_control"]["passed"]
    assert not summary["empirical_objective_attribution_supported"]
    assert summary["primary"]["learnability"]["passed"]
    for row in rows:
        if row["regime"] == "ceiling" and row["algorithm"] == "monotonic_policy" and row["cell"] == "nominal":
            row["all_oracle_optimal"] = .98
    assert not exp.summarize(rows,cfg,True)[1]["primary"]["learnability"]["passed"]


def test_nominal_guardrail_not_hidden_by_regret_and_split_not_primary():
    cfg,rows = good_metrics()
    for row in rows:
        if row["regime"] == "split":
            row["all_regret"] = 1000.; row["all_oracle_optimal"] = 0.
    assert exp.summarize(rows,cfg,True)[1]["primary"]["learnability"]["passed"]
    for row in rows:
        if row["cell"] == "nominal" and row["regime"] == "ceiling" and row["algorithm"] == "monotonic_hybrid":
            row["undiscounted_policy_cost"] = 5.
    assert not exp.summarize(rows,cfg,True)[1]["primary"]["objective"]["hypothesis_passed"]


def test_seed_registry_and_panels():
    cfg = exp.settings("full")
    assert exp.seed_audit(cfg)["prior_panels_disjoint"]
    cfg["train_seeds"] = [96000]
    with pytest.raises(ValueError,match="prior local"):
        exp.seed_audit(cfg)


def test_mini_runner_matches_streams_reload_and_preserves_artifacts(monkeypatch,tmp_path):
    original = exp.settings
    monkeypatch.setattr(exp,"settings",lambda p:dict(original(p),updates=2,log_interval=1))
    out = tmp_path/"run"
    args = Namespace(profile="smoke",device="cpu",output_dir=str(out))
    exp.run(args)
    manifest = json.loads((out/"manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["actual_models"] == 24
    assert manifest["actual_evaluation_rows"] == 72
    assert manifest["actual_optimizer_updates"] == 48
    assert manifest["audits"]["matched_minibatch_streams"]
    assert manifest["audits"]["policy_only_mixer_unchanged"]
    assert len(list(out.rglob("model.pt"))) == 24
    assert all((out/p).is_file() for p in manifest["outputs"])
    summary = json.loads((out/"summary.json").read_text())
    assert summary["primary"]["learnability"]["passed"] is None
    assert summary["primary"]["objective"]["hypothesis_passed"] is None
    ckpt = next(out.glob("*/ceiling/monotonic_policy/*/model.pt"))
    payload = torch.load(ckpt,weights_only=True)
    payload["objective"] = "monotonic_hybrid"
    torch.save(payload,tmp_path/"bad.pt")
    with pytest.raises(ValueError,match="objective"):
        exp.load_checkpoint(tmp_path/"bad.pt")
    with pytest.raises(FileExistsError):
        exp.run(args)


def test_failure_is_recorded_without_retry(monkeypatch,tmp_path):
    calls = []
    def fail(*args):
        calls.append(True)
        raise RuntimeError("injected training failure")
    monkeypatch.setattr(exp,"fit",fail)
    out = tmp_path/"failed"
    with pytest.raises(RuntimeError,match="injected"):
        exp.run(Namespace(profile="smoke",device="cpu",output_dir=str(out)))
    assert len(calls) == 1
    assert json.loads((out/"manifest.json").read_text())["status"] == "FAILED"
    assert (out/"nominal/teacher_actions.json").is_file()


def test_full_population_consistent_teacher_attains_discounted_oracle():
    for (cell,config),seed in zip(base.cells("full").items(),exp.DATASET_SEEDS):
        oracle = base.ExactOracle(config,progress=False)
        witness = role.policy_feasibility(oracle,"monotonic_role_context",seed)
        teacher = exp.teacher_labels(oracle,witness)
        indices = {k:oracle.nodes[k]["actions"].index(a) for k,a in teacher.items()}
        assert abs(oracle.policy_cost(indices,base.GAMMA) + max(oracle.nodes[oracle.initial]["q95"])) < 1e-8
