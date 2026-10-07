import csv
from dataclasses import asdict, replace
import hashlib
import json
import itertools
import numpy as np
import pytest
import torch

from ht_pdm_fjsp import maintenance_solver_comparison as runner
from ht_pdm_fjsp import maintenance_solver_model as model
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests, communication, interval
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, transition


def test_profiles_disjoint_and_smoke_never_uses_full_test():
    p=model.profile_partitions()
    assert list(map(len,p.values()))==[80,16,32]
    assert not any(set(a)&set(b) for a,b in itertools.combinations(p.values(),2))
    def signature(seed,split):
        c=model.cohort_config(seed,"specialized",12,split=split)
        return c.service_time,c.restoration
    dev={signature(s,"development") for s in range(161000,161016)}
    test={signature(s,"test") for s in range(163000,163032)}
    train={signature(s,"train") for s in range(3000)}
    assert len(train)==80 and len(dev)==16 and len(test)==32
    assert not dev&test and not dev&train and not test&train
    for split in p:
        assert model.profile_id(165010,split,"smoke") in p["train"]
    for condition in model.CONDITIONS:
        a=WaitingEnv(model.cohort_config(163000,condition,12))
        b=WaitingEnv(model.cohort_config(163000,condition,24))
        a.reset(164000);b.reset(164000)
        assert np.array_equal(a.events,b.events[:12])


def test_requests_count_last_boundary_and_start_not_completion():
    healthy=DispatchState((2,2,2,2),(False,)*4,(0,)*4,(0,0),(-1,-1))
    failed=replace(healthy,failed=(True,False,False,False))
    requests=Requests(healthy)
    requests.step(11,healthy,(),failed)
    assert requests.finish(12)==[dict(machine=0,request=1,arrival=12,end=12,wait=0,served=False,censored=True)]
    requests=Requests(failed)
    servicing=replace(failed,remaining=(3,0),assigned=(0,-1))
    requests.step(0,failed,((0,0),),servicing)
    row=requests.finish(1)[0]
    assert row["served"] and row["wait"]==0 and not row["censored"]


def test_statistics_quantiles_and_wire_accounting():
    for n,confidence,expected in [(10,.9875,3.110934823179474),(32,.9875,2.6519126890308002)]:
        x=np.arange(n,dtype=float)
        ci=interval(x,confidence)
        assert (ci[1]-x.mean())/(x.std(ddof=1)/np.sqrt(n))==pytest.approx(expected)
    assert interval([1],.95) is None
    c=model.cohort_config(165020,"nominal",4,profile="smoke")
    for name in runner.ARMS:
        wire=communication(c,name,((0,0),),True)
        assert wire["communication_bytes"]==sum(v for k,v in wire.items() if k.endswith("_bytes") and k!="communication_bytes")
        assert wire["communication_rounds"]>=2


def test_locked_counts_and_seed_audit():
    full=runner.settings("full")
    assert runner.counts(full)==dict(models=50,training_steps=24576000,training_episodes=2048000,optimizer_steps=512000,development_episodes=82560,test_episodes=286720,test_decision_intervals=5160960,test_machine_rows=1146880)
    assert runner.seed_audit(full)[0]["status"]=="PASS"
    assert runner.seed_audit(runner.settings("smoke"))[0]["status"]=="PASS"


def test_full_statistics_uses_profiles_or_replicas_not_episodes():
    # Synthetic arithmetic fixture only: no full environment or held-out rollout.
    from ht_pdm_fjsp.maintenance_solver_statistics import summarize
    cfg=runner.settings("full")
    cfg["test_seeds"]=cfg["test_seeds"][:1]
    cfg["horizons"]=[12]
    costs={name:120. for name in cfg["controls"]}
    costs.update(joint_rollout=95.,fixed_allocation_rollout=95.,central_matching_ppo=100.,central_fixed_allocation_ppo=100.,independent_local_ppo=120.,cooperative_local_ppo=90.,cooperative_full_ppo=90.)
    rows=[]
    for name,cost in costs.items():
        for seed,cs,cond in itertools.product(cfg["train_seeds"] if name in runner.ARMS else [-1],cfg["test_cohorts"],model.CONDITIONS):
            rows.append(dict(controller=name,train_seed=seed,cohort=cs,condition=cond,horizon=12,seed=cfg["test_seeds"][0],objective=cost,base_cost=cost,waiting_violation=0.))
    paired,cohorts,contrasts=summarize(rows,{c:"adaptive_isolated_matching" for c in model.CONDITIONS},cfg)
    primary=[v for v in contrasts.values() if v["confirmatory"]]
    assert len(primary)==4 and all(v["status"]=="PASS" for v in primary)
    assert sorted(v["replicates"] for v in primary)==[10,10,10,32]
    assert all(v["confidence_level"]==.9875 for v in primary)
    with pytest.raises(RuntimeError,match="grid"):
        summarize(rows+rows[:1],{c:"adaptive_isolated_matching" for c in model.CONDITIONS},cfg)


def test_failed_run_preserves_partial_manifest(tmp_path,monkeypatch):
    def crash(*args,**kwargs):
        raise RuntimeError("injected training failure")
    monkeypatch.setattr(runner,"fit",crash)
    out=tmp_path/"failed"
    with pytest.raises(RuntimeError,match="injected"):
        runner.run(out,"smoke")
    m=json.loads((out/"manifest.json").read_text())
    assert m["status"]=="FAILED" and "injected" in m["error"]
    assert (out/"episodes.partial.csv").exists()
    assert "development_episodes.csv" in m["artifacts"]


def test_integrated_smoke_and_immutable_artifacts(tmp_path):
    torch.set_num_threads(1)
    out=tmp_path/"smoke"
    runner.run(out,"smoke")
    manifest=json.loads((out/"manifest.json").read_text())
    assert manifest["status"]=="COMPLETED"
    assert manifest["actual_counts"]==runner.counts(runner.settings("smoke"))
    for file,meta in manifest["artifacts"].items():
        assert hashlib.sha256((out/file).read_bytes()).hexdigest()==meta["sha256"]
    summary=json.loads((out/"summary.json").read_text())
    assert all(c["status"]=="ENGINEERING_ONLY" for c in summary["contrasts"].values())
    decisions=list(csv.DictReader((out/"decisions.csv").open()))
    for row in decisions:
        state=DispatchState(**{k:tuple(v) for k,v in json.loads(row["state"]).items()})
        c=model.cohort_config(int(row["cohort"]),row["condition"],int(row["horizon"]),profile="smoke")
        env=WaitingEnv(c);env.reset(int(row["seed"]))
        pairs=tuple(tuple(p) for p in json.loads(row["action"]))
        following,cost,info=transition(c,state,pairs,env.events[int(row["time"])])
        assert json.loads(json.dumps(asdict(following)))==json.loads(row["following"])
        assert info==json.loads(row["physical_metrics"])
    requests=list(csv.DictReader((out/"request_metrics.csv").open()))
    assert all(int(r["wait"])==int(r["end"])-int(r["arrival"]) for r in requests)
    with pytest.raises(ValueError,match="new/empty"):
        runner.run(out,"smoke")
