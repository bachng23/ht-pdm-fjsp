import csv
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import itertools
import json

import numpy as np
import pytest

from ht_pdm_fjsp import maintenance_boundary_sensitivity as experiment
from ht_pdm_fjsp.maintenance_dispatch import DispatchState
from ht_pdm_fjsp.maintenance_heterogeneous_model import plan
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests
from ht_pdm_fjsp.maintenance_waiting import WaitingConfig, WaitingEnv, transition


def tiny(probability=.6):
    return WaitingConfig(2, 2, 7, 2, probability, ((3, 2), (2, 0)),
                         ((1., 1.), (1., 1.)), waiting_price=12.)


@pytest.mark.parametrize("fixed", [False, True])
def test_zero_matches_inherited_and_tail_preserves_base_scores(fixed):
    c = tiny()
    state = DispatchState((2, 1), (True, False), (3, 0), (0, 0), (-1, -1))
    action, old = plan(c, state, 4, "adaptive_isolated_matching", 81, 16, fixed=fixed)
    zero_action, zero = experiment.forecast_plan(c, state, 4, 0, 81, 16, fixed=fixed)
    _, tail = experiment.forecast_plan(c, state, 4, 3, 81, 16, fixed=fixed)
    assert zero_action == action
    assert zero["candidates"] == old["candidates"] == tail["candidates"]
    np.testing.assert_array_equal(zero["means"], old["means"])
    np.testing.assert_array_equal(zero["mc_se"], old["mc_se"])
    np.testing.assert_array_equal(tail["base_means"], zero["means"])
    np.testing.assert_allclose(tail["means"], np.array(tail["base_means"])+tail["tail_means"])


def test_common_candidate_scores_and_no_actual_future_access():
    c = tiny()
    env = WaitingEnv(c); env.reset(71)
    action, full = experiment.forecast_plan(c, env.state, 4, 3, 81, 16)
    _, fixed = experiment.forecast_plan(c, env.state, 4, 3, 81, 16, fixed=True)
    scores = dict(zip(full["candidates"], full["means"]))
    assert all(q == scores[a] for a, q in zip(fixed["candidates"], fixed["means"]))
    env.events = tuple((True, True) for _ in range(c.horizon))
    after_action, after = experiment.forecast_plan(c, env.state, 4, 3, 81, 16)
    assert after_action == action and after == full
    with pytest.raises(ValueError):
        experiment.forecast_plan(c, env.state, 0, 3, 81, 16)


@pytest.mark.parametrize("probability", [0., 1.])
def test_deterministic_tail_scores_equal_scalar_policy_rollout(probability):
    c = tiny(probability)
    # An ongoing nonpreemptive job crosses the cutoff, while CM waits elsewhere.
    state = DispatchState((2, 2), (False, True), (0, 3), (3, 0), (0, -1))
    _, diag = experiment.forecast_plan(c, state, 1, 4, 19, 8)
    for root, predicted_base, predicted_tail in zip(diag["candidates"],diag["base_means"],diag["tail_means"]):
        current = state
        base = tail = 0.
        for t in range(5):
            action = root if t == 0 else experiment.prior.rule_action(c,current,5-t,experiment.TAIL_RULE)
            current, cost, _ = transition(c,current,action,(bool(probability),)*c.machines)
            if t == 0:
                base += cost
                assert current.remaining[0] == 2 and current.assigned[0] == 0
            else:
                tail += cost
        assert predicted_base == base and predicted_tail == tail


def test_full_budgets_and_physical_exclusion_ignore_horizon_and_relabeling():
    cfg = experiment.settings("full")
    assert experiment.counts(cfg) == dict(development_episodes=4032,development_decisions=161280,
        test_episodes=23040,base_decisions=645120,tail_decisions=276480,test_decisions=921600)
    assert experiment.seed_audit(cfg)["status"] == "PASS"
    assert experiment.prior.seed_audit(experiment.prior.settings("full"))["status"] == "PASS"
    altered = {**cfg, "analysis_seed": 42}
    with pytest.raises(ValueError,match="registry"):
        experiment.seed_audit(altered)
    panels, registry = experiment.build_panels(cfg)
    old = experiment.exclusion_registry()["previous_physical_signatures"]
    smoke, _ = experiment.build_panels(experiment.settings("smoke"))
    assert len(registry) == 80*3
    for condition in experiment.CONDITIONS:
        signatures = [experiment.signature(c) for (_,_,cond), c in panels.items() if cond == condition]
        assert len(set(signatures)) == 80
        assert not set(signatures) & {experiment.frozen(s) for s in old[condition]}
        assert not set(signatures) & {experiment.signature(c) for (_,_,cond),c in smoke.items() if cond == condition}
    c = next(iter(panels.values()))
    swapped = replace(c,horizon=999,
        service_time=tuple(tuple(reversed(row)) for row in reversed(c.service_time)),
        restoration=tuple(tuple(reversed(row)) for row in reversed(c.restoration)))
    assert experiment.signature(c) == experiment.signature(swapped)


def test_actual_physical_tapes_share_prefix_across_horizons():
    c = tiny()
    short,long = WaitingEnv(replace(c,horizon=6)), WaitingEnv(replace(c,horizon=60))
    short.reset(432); long.reset(432)
    assert short.state == long.state
    assert short.events == long.events[:6]


def synthetic_rows(cfg):
    rows=[]
    for cond,h,c,s,name in itertools.product(experiment.CONDITIONS,cfg["horizons"],cfg["test_cohorts"],cfg["test_shocks"],experiment.CONTROLLERS):
        row=dict(condition=cond,horizon=h,cohort=c,shock=s,controller=name)
        for key in ("base_objective","tail_objective","extended_objective","base_cost_per_machine_tick",
                    "extended_cost_per_machine_tick","cutoff_pending","cutoff_failed","cutoff_cm_in_service",
                    "post_tail_pending","post_tail_failed","interior_cost_per_machine_tick",
                    "interior_pending_mean","interior_failed_mean","interior_overdue_per_tick",
                    "cutoff_censored_requests","extended_censored_requests","carryover_requests",
                    "carryover_started_in_tail","carryover_censored_after_tail","latency_mean_ms",
                    "base_maintenance_cost","base_failure_cost","base_unavailability_cost","base_waiting_cost",
                    "base_overdue_waiting_ticks"):
            row[key]=100 if name in ("best_rule","fixed_zero","flexible_zero") else 90 if name=="fixed_tail" else 80
        rows.append(row)
    return rows


def test_bootstrap_three_primary_contrasts_and_profile_units():
    cfg=experiment.settings("smoke")
    cfg["test_cohorts"]=[11,12];cfg["test_shocks"]=[21,22]
    rows=synthetic_rows(cfg)
    result,differences=experiment.summary(rows,cfg)
    primary=[v for v in result["contrasts"].values() if v["primary"]]
    assert len(primary)==3
    assert all(v["profiles"]==2 and v["confidence"]==1-.05/3 for v in primary)
    assert all(v["mean_difference"]==10 and v["status"]=="ENGINEERING_ONLY" for v in primary)
    assert len(differences)==3*2*3*2
    with pytest.raises(RuntimeError,match="grid"):
        experiment.summary(rows+rows[:1],cfg)
    with pytest.raises(RuntimeError,match="grid"):
        experiment.summary(rows[:-1],cfg)
    cfg["profile"]="full"
    result,_=experiment.summary(rows,cfg)
    assert result["timing_learning_screen"]=="PASS"
    for row in rows:
        if row["controller"]=="fixed_tail":
            row["interior_failed_mean"]=101
    result,_=experiment.summary(rows,cfg)
    assert result["timing_learning_screen"]=="NOT_ESTABLISHED"
    assert not result["service_guard"]["total_failed"]


def test_integrated_smoke_replay_requests_costs_hashes_and_immutability(tmp_path):
    out=tmp_path/"smoke"
    manifest=experiment.run(out)
    assert manifest["status"]=="COMPLETED" and manifest["audit"]=="PASS"
    assert not manifest["full_test_outcomes_opened"]
    assert manifest["actual_counts"]==dict(development_episodes=168,development_decisions=1344,
        test_episodes=60,base_decisions=360,tail_decisions=120,test_decisions=480)
    for name,metadata in manifest["artifacts"].items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==metadata["sha256"]
    cfg=experiment.settings("smoke");panels,_=experiment.build_panels(cfg)
    decisions=list(csv.DictReader((out/"decisions.csv").open()))
    episode_rows=list(csv.DictReader((out/"episodes.csv").open()))
    requests=list(csv.DictReader((out/"requests.csv").open()))
    carryovers=list(csv.DictReader((out/"carryover.csv").open()))
    grouped=defaultdict(list)
    key=lambda r:tuple(r[k] for k in ("condition","cohort","shock","horizon","controller"))
    for row in decisions:
        grouped[key(row)].append(row)
    reconstructed=[]
    reconstructed_carryovers=[]
    for result in episode_rows:
        condition,cohort,shock,h,controller=key(result);h=int(h)
        c=replace(panels["test",int(cohort),condition],horizon=h+cfg["tail_ticks"])
        env=WaitingEnv(c);env.reset(int(grouped[key(result)][0]["env_seed"]))
        base_requests,all_requests=Requests(env.state),Requests(env.state)
        phase_cost=defaultdict(float)
        for row in grouped[key(result)]:
            state=DispatchState(**{k:tuple(v) for k,v in json.loads(row["state"]).items()})
            assert state==env.state
            action=tuple(tuple(p) for p in json.loads(row["action"]));t=int(row["time"])
            if t>=h:
                assert action==experiment.prior.rule_action(c,state,c.horizon-t,experiment.TAIL_RULE)
            following,cost,info=transition(c,state,action,env.events[t])
            env.step(action)
            assert following==env.state
            assert json.loads(json.dumps(asdict(following)))==json.loads(row["following"])
            assert info==json.loads(row["physical_metrics"])
            phase_cost[row["phase"]]+=cost
            all_requests.step(t,state,action,following)
            if t<h:
                base_requests.step(t,state,action,following)
            if t+1==h:
                cut=base_requests.finish(h)
                for metric,value in experiment.snapshot(following).items():
                    assert float(result[f"cutoff_{metric}"])==value
        extended=all_requests.finish(c.horizon)
        for scope,records in (("cutoff",cut),("extended",extended)):
            for record in records:
                reconstructed.append({**{k:result[k] for k in ("condition","cohort","shock","horizon","controller")},
                    "scope":scope,**{k:str(v) for k,v in record.items()},"overdue":str(record["wait"]>c.waiting_limit)})
        for record in cut:
            if record["censored"]:
                later=next(r for r in extended if (r["machine"],r["request"])==(record["machine"],record["request"]))
                reconstructed_carryovers.append({**{k:result[k] for k in ("condition","cohort","shock","horizon","controller")},
                    **{k:str(v) for k,v in dict(machine=record["machine"],request=record["request"],arrival=record["arrival"],
                    wait_at_cutoff=record["wait"],started_in_tail=later["served"],wait_at_start_or_end=later["wait"],
                    censored_after_tail=later["censored"]).items()}})
        assert float(result["base_objective"])==phase_cost["base"]
        assert float(result["tail_objective"])==phase_cost["tail"]
        assert float(result["extended_objective"])==sum(phase_cost.values())
    assert reconstructed==requests
    assert reconstructed_carryovers==carryovers
    assert any(r["censored"]=="True" for r in requests)
    assert carryovers and any(r["started_in_tail"]=="True" for r in carryovers)
    assert json.loads((out/"summary.json").read_text())["timing_learning_screen"]=="ENGINEERING_ONLY"
    with pytest.raises(ValueError,match="new/empty"):
        experiment.run(out)


def test_interruption_preserves_partial_output_and_failed_manifest(tmp_path,monkeypatch):
    original=experiment.episode
    def interrupt(c,cohort,shock,condition,horizon,controller,*args):
        if controller=="flexible_zero":
            raise KeyboardInterrupt("injected interruption")
        return original(c,cohort,shock,condition,horizon,controller,*args)
    monkeypatch.setattr(experiment,"episode",interrupt)
    out=tmp_path/"interrupted"
    with pytest.raises(KeyboardInterrupt):
        experiment.run(out)
    manifest=json.loads((out/"manifest.json").read_text())
    assert manifest["status"]=="FAILED" and "KeyboardInterrupt" in manifest["error"]
    assert not manifest["full_test_outcomes_opened"]
    assert len(list(csv.DictReader((out/"episodes.partial.csv").open())))==2
    assert not (out/"episodes.csv").exists()


def test_largest_forecast_on_engineering_profile_only():
    panels,_=experiment.build_panels(experiment.settings("smoke"))
    c=replace(panels["test",10500100,"skill_mask"],horizon=60)
    env=WaitingEnv(c);env.reset(10600100)
    action,diag=experiment.forecast_plan(c,env.state,48,12,10600101,128)
    assert diag["forecast_ticks"]==60 and diag["terminal_ticks"]==12
    assert np.isfinite(diag["means"]).all()
    assert action in diag["candidates"]
