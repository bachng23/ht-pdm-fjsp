import csv
from dataclasses import replace
import json
import math

import pytest
import torch
from torch.distributions import Categorical

from ht_pdm_fjsp import maintenance_decoding_audit as audit
from ht_pdm_fjsp import maintenance_solver_comparison as original
from ht_pdm_fjsp.maintenance_solver_policy import MatchingActorCritic, LOCAL, MARL, CENTRAL, CATALOGUE, feature_batch
from ht_pdm_fjsp.maintenance_solver_model import cohort_config
from ht_pdm_fjsp.maintenance_joint_map import Decoder, matching_log_probabilities, best_index, RAW_ACTIONS
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_contention_headroom import file_hash


@torch.no_grad()
@pytest.mark.parametrize("algorithm",[LOCAL,MARL])
@pytest.mark.parametrize("condition",["specialized","nominal","skill_mask"])
def test_exhaustive_scores_match_independent_teacher_forcing(algorithm,condition):
    torch.set_num_threads(1)
    torch.manual_seed(81)
    model=MatchingActorCritic(algorithm,hidden=8).eval()
    c=cohort_config(165020,condition,6,profile="smoke")
    env=WaitingEnv(c);env.reset(173000)
    for t in range(c.horizon):
        ids,scores,mass=matching_log_probabilities(model,c,env.state,c.horizon-t)
        batch=feature_batch([c],[env.state],[c.horizon-t])
        independent=[]
        for idx in ids:
            _,p,_,_=model._serial(batch,action=RAW_ACTIONS[idx][None],compute_value=False)
            independent.append(float(p[0]))
        assert scores.tolist()==pytest.approx(independent,abs=2e-6)
        assert mass==pytest.approx(1.,abs=1e-5)
        assert abs(sum(math.exp(v) for v in independent)-1)<1e-5
        assert float(scores.max()-scores[best_index(scores)])<=1.1e-6
        pairs,_=Decoder(model,"joint_map").select(c,env.state,c.horizon-t)
        assert pairs==CATALOGUE[int(ids[best_index(scores)])]
        env.step(pairs)


def test_greedy_can_miss_joint_map(monkeypatch):
    torch.set_num_threads(1)
    model=MatchingActorCritic(LOCAL,hidden=4).eval()
    def conditional(batch,machine,reserved,encoded=None):
        n=len(machine)
        logits=torch.full((n,5),-100.)
        for i,m in enumerate(machine.tolist()):
            if m==0:logits[i,:3]=torch.tensor([math.log(.4),math.log(.6),-100.])
            elif m==1:logits[i,:3]=torch.tensor([math.log(.005),math.log(.99),math.log(.005)])
            else:logits[i,0]=0.
        mask=batch["proposal_mask"][torch.arange(n),machine].clone()
        mask[:,1:] &= ~reserved
        return Categorical(logits=logits.masked_fill(~mask,-torch.inf)),mask
    monkeypatch.setattr(model,"serial_distribution",conditional)
    c=cohort_config(165020,"specialized",4,profile="smoke")
    env=WaitingEnv(c)
    greedy,_=Decoder(model,"greedy").select(c,env.state,4)
    mapped,diag=Decoder(model,"joint_map").select(c,env.state,4)
    assert greedy==((0,0),)
    assert mapped==((1,0),)
    assert math.exp(diag["log_probability"])==pytest.approx(.4*.99,abs=1e-6)
    assert diag["probability_mass"]==pytest.approx(1.,abs=1e-5)


def test_greedy_identity_no_critic_for_map_wait_and_ties(monkeypatch):
    model=MatchingActorCritic(LOCAL,hidden=8).eval()
    c=cohort_config(165020,"specialized",4,profile="smoke")
    env=WaitingEnv(c);env.reset(173000)
    assert Decoder(model,"greedy").select(c,env.state,4)==model.select(c,env.state,4)
    def forbidden(*args,**kwargs):raise AssertionError("critic accessed")
    monkeypatch.setattr(model,"_values",forbidden)
    Decoder(model,"joint_map").select(c,env.state,4)
    blocked=replace(env.state,remaining=(1,1),assigned=(0,1))
    ids,scores,mass=matching_log_probabilities(model,c,blocked,4)
    assert len(ids)==1 and CATALOGUE[int(ids[0])]==() and float(scores[0])==0 and mass==1
    assert best_index(torch.tensor([.0,.0000005,.0000002]))==0
    with pytest.raises(ValueError,match="serial"):
        matching_log_probabilities(MatchingActorCritic(CENTRAL,hidden=4),c,env.state,4)


@pytest.fixture(scope="module")
def source_run(tmp_path_factory):
    # Small real training fixture from the existing tested runner, smoke profiles only.
    torch.set_num_threads(1)
    path=tmp_path_factory.mktemp("decoding_source")/"run"
    original.run(path,"smoke")
    return path


def test_panel_counts_sealing_and_source_rejection(source_run,tmp_path):
    full=audit.settings("full",original.settings("full"))
    assert audit.counts(full)==dict(checkpoints=40,training_steps=0,optimizer_steps=0,
        episodes=317440,decision_intervals=5713920,decoding_audits=960,latency_rows=8928)
    assert audit.seed_audit(full,original.settings("full"))["new_shocks_disjoint"]
    smoke=audit.settings("smoke",original.settings("full"))
    assert not set(smoke["shocks"]) & set(full["shocks"])
    assert smoke["cohorts"]==[165020]
    assert audit.preflight(source_run,"smoke")[0]["train_seeds"]==[165000]
    with pytest.raises(ValueError,match="source run"):
        audit.preflight(source_run,"full")
    with pytest.raises(ValueError,match="unsafe"):
        audit.checked_path(source_run,"../outside.pt")


def test_pair_statistics_and_missing_grid():
    cfg=audit.settings("full",original.settings("full"))
    cfg.update(cohorts=[163000],shocks=[171000],conditions=["specialized"],horizons=[12])
    rows=[]
    for a,mode,seed in audit.controllers(cfg):
        r=dict(controller=audit.label(a,mode),algorithm=a,decoder=mode,train_seed=seed,cohort=163000,
               condition="specialized",horizon=12,seed=171000)
        for metric in ["objective","base_cost","maintenance_cost","failure_cost","unavailability_cost","waiting_cost",
            "failures","waiting_violation","terminal_pending","corrective_requests","served_requests","censored_requests",
            "terminal_zero_wait_censored","request_overdue_observed","max_wait","inference_seconds",
            "decision_latency_p50","decision_latency_p95","invalid_assignments","maximum_mass_error","maximum_map_selection_regret"]:
            r[metric]=0.
        r["objective"]=r["base_cost"]=95. if mode=="joint_map" else 100.
        rows.append(r)
    s,p=audit.summarize(rows,cfg)
    assert len(p)==20
    assert all(v["status"]=="PASS" and v["replicates"]==10 and v["confidence"]==.975 for v in s["contrasts"].values())
    assert audit.confidence_interval(list(range(10)),.975)==pytest.approx([1.9292978306381237,7.070702169361876])
    with pytest.raises(RuntimeError,match="grid"):audit.summarize(rows+rows[:1],cfg)
    with pytest.raises(RuntimeError,match="grid"):audit.summarize(rows[:-1],cfg)


def test_integrated_smoke_preserves_inputs_and_censoring(source_run,tmp_path):
    before={p:file_hash(p) for p in source_run.rglob("*") if p.is_file()}
    out=tmp_path/"audit"
    manifest=audit.run(out,source_run,"smoke")
    assert manifest["status"]=="COMPLETED"
    assert manifest["actual_counts"]==manifest["expected_counts"]
    assert all(file_hash(p)==h for p,h in before.items())
    assert all(file_hash(out/p)==v["sha256"] for p,v in manifest["artifacts"].items())
    summary=json.loads((out/"summary.json").read_text())
    assert all(v["status"]=="ENGINEERING_ONLY" and v["ci"] is None for v in summary["contrasts"].values())
    assert not json.loads((out/"seed_audit.json").read_text())["full_shock_panel_opened"]
    for row in csv.DictReader((out/"request_metrics.csv").open()):
        assert int(row["wait"])==int(row["end"])-int(row["arrival"])
        assert (row["served"]=="True") != (row["censored"]=="True")
    with pytest.raises(ValueError,match="new/empty"):audit.run(out,source_run,"smoke")
    with pytest.raises(ValueError,match="separate"):audit.run(source_run/"inside",source_run,"smoke")
    # Same old checkpoint/decoder and same environment produce identical physical metrics.
    first=next(csv.DictReader((out/"episodes.csv").open()))
    c=cohort_config(int(first["cohort"]),first["condition"],int(first["horizon"]),profile="smoke")
    model,_=original.load_checkpoint(source_run/f'{CENTRAL}/train_seed_165000/model.pt')
    baseline=original.episode(c,int(first["seed"]),CENTRAL,None,4,{},model)
    for metric in ("objective","base_cost","waiting_cost","max_wait","waiting_violation","terminal_pending"):
        assert float(first[metric])==baseline[metric]


def test_tampered_checkpoint_and_failed_run(source_run,tmp_path,monkeypatch):
    selection=json.loads((source_run/"checkpoint_selection.json").read_text())
    target=source_run/selection["selections"][f'{LOCAL}:165000']["checkpoint"]
    content=target.read_bytes()
    try:
        target.write_bytes(content+b'changed')
        with pytest.raises(ValueError,match="hash mismatch"):audit.preflight(source_run,"smoke")
    finally:target.write_bytes(content)
    def crash(*args,**kwargs):raise RuntimeError("injected diagnostic error")
    monkeypatch.setattr(audit,"diagnostics",crash)
    out=tmp_path/"failed"
    with pytest.raises(RuntimeError,match="injected"):audit.run(out,source_run,"smoke")
    m=json.loads((out/"manifest.json").read_text())
    assert m["status"]=="FAILED" and "injected" in m["error"]
    assert (out/"episodes.partial.csv").exists()
