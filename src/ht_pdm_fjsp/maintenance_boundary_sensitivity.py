"""Finite-horizon boundary diagnostic with a shared policy-evaluation tail."""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_opportunity_cost as prior
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, ENV_VERSION
from ht_pdm_fjsp.maintenance_heterogeneous_model import (
    arrays, batch_tick, feasible, fixed_candidates, reference_batch,
)
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests

VERSION = "maintenance_boundary_sensitivity_v1"
ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = prior.CONDITIONS
RULES = prior.RULES
CONTROLLERS = ("best_rule", "fixed_zero", "flexible_zero", "fixed_tail", "flexible_tail")
TAIL_RULE = "deadline_matching"


def settings(execution_profile):
    if execution_profile not in ("smoke", "full"):
        raise ValueError("unknown profile")
    full = execution_profile == "full"
    return dict(profile=execution_profile, horizons=[12, 24, 48] if full else [4, 8],
        tail_ticks=12 if full else 2, scenarios=128 if full else 8,
        development_cohorts=list(range(10100000, 10100016)) if full else [10500000, 10500001],
        development_shocks=list(range(10200000, 10200004)) if full else [10600000, 10600001],
        test_cohorts=list(range(10300000, 10300064)) if full else [10500100],
        test_shocks=list(range(10400000, 10400008)) if full else [10600100, 10600101],
        analysis_seed=10700000 if full else 10700001,
        bootstrap_samples=20000 if full else 200,
        conditions=list(CONDITIONS), controllers=list(CONTROLLERS), rules=list(RULES),
        continuation="adaptive_isolated_matching", tail_rule=TAIL_RULE,
        discount=1., cost_margin=.01, pending_margin=.1,
        service_guard_pending=.1, service_guard_failed=.1, service_guard_overdue_per_tick=.25/12,
        training="NOT_APPLICABLE", train_seeds=[], primary_confidence=1-.05/3)


def counts(cfg):
    d = len(cfg["conditions"])*len(cfg["development_cohorts"])*len(cfg["development_shocks"])*len(RULES)
    t = len(cfg["conditions"])*len(cfg["test_cohorts"])*len(cfg["test_shocks"])*len(CONTROLLERS)
    base = t*sum(cfg["horizons"])
    tail = t*len(cfg["horizons"])*cfg["tail_ticks"]
    return dict(development_episodes=d*len(cfg["horizons"]),
                development_decisions=d*sum(h+cfg["tail_ticks"] for h in cfg["horizons"]),
                test_episodes=t*len(cfg["horizons"]), base_decisions=base,
                tail_decisions=tail, test_decisions=base+tail)


def frozen(value):
    return tuple(frozen(v) for v in value) if isinstance(value, (tuple, list)) else value


def signature(c):
    variants = [tuple(sorted(tuple((c.service_time[m][j], c.restoration[m][j] if c.service_time[m][j] else 0.)
                                  for j in order) for m in range(c.machines)))
                for order in itertools.permutations(range(c.technicians))]
    return c.failure_age, c.failure_probability, min(variants)


def exclusion_registry():
    return json.loads((ROOT / f"configs/{VERSION}_profile_exclusions.json").read_text())


def build_panels(cfg):
    registry = exclusion_registry()
    seen = {cond: {frozen(s) for s in registry["previous_physical_signatures"][cond]} for cond in CONDITIONS}
    if cfg["profile"] == "smoke":
        # Metadata-only construction never runs a full test environment.
        full_panels, _ = build_panels(settings("full"))
        for (_, _, cond), c in full_panels.items():
            seen[cond].add(signature(c))
    panels, rows = {}, []
    for split in ("development", "test"):
        for cohort in cfg[f"{split}_cohorts"]:
            for attempt in range(10000):
                configs = {cond: prior.make_config(cohort, cond, max(cfg["horizons"]), attempt)
                           for cond in CONDITIONS}
                if all(signature(c) not in seen[cond] for cond, c in configs.items()):
                    break
            else:
                raise RuntimeError("physical profile pool exhausted")
            for cond, c in configs.items():
                seen[cond].add(signature(c))
                panels[split, cohort, cond] = c
                rows.append(dict(split=split, cohort=cohort, condition=cond,
                                 generator_attempt=attempt, config=asdict(c)))
    return panels, rows


def seed_audit(cfg):
    def integers(value):
        if isinstance(value, dict):
            return set().union(set(), *(integers(v) for v in value.values()))
        if isinstance(value, list):
            return set().union(set(), *(integers(v) for v in value))
        return {value} if isinstance(value, int) and not isinstance(value, bool) else set()
    historical = set()
    hashes = {}
    for path in sorted((ROOT / "configs").glob("*seed_registry.json")):
        if path.name == f"{VERSION}_seed_registry.json":
            continue
        historical |= integers(json.loads(path.read_text()))
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    historical |= integers(exclusion_registry()["previous_seed_panels"])
    panels = [set(cfg[k]) for k in ("development_cohorts", "development_shocks", "test_cohorts", "test_shocks")]
    panels.append({cfg["analysis_seed"]})
    if any(a & b for a, b in itertools.combinations(panels, 2)) or set.union(*panels) & historical:
        raise ValueError("seed overlap")
    own_path = ROOT / f"configs/{VERSION}_seed_registry.json"
    own = json.loads(own_path.read_text())["panels"]
    if any(cfg[k] != value for k, value in own[cfg["profile"]].items()):
        raise ValueError("settings differ from committed seed registry")
    if integers(own["smoke"]) & integers(own["full"]):
        raise ValueError("smoke/full seed overlap")
    return dict(status="PASS", historical_registry_hashes=hashes,
                raw_seed_registry_sha256=hashlib.sha256(own_path.read_bytes()).hexdigest(),
                prior_physical_exclusion_sha256=hashlib.sha256((ROOT / f"configs/{VERSION}_profile_exclusions.json").read_bytes()).hexdigest())


def forecast_plan(c, state, h, tail, forecast_seed, scenarios, fixed=False):
    """Add tail values without changing ANY pre-cutoff continuation decisions."""
    if h < 1 or tail < 0 or scenarios < 2:
        raise ValueError("positive horizon, nonnegative tail and >=2 scenarios required")
    candidates = fixed_candidates(c, state) if fixed else feasible(c, state)
    a = len(candidates)
    s = arrays(state, a*scenarios)
    roots = np.zeros((a, c.machines, c.technicians), dtype=bool)
    for i, pairs in enumerate(candidates):
        for m, j in pairs:
            roots[i, m, j] = True
    selected = np.repeat(roots, scenarios, axis=0)
    events = np.random.default_rng(forecast_seed).random((h+tail, scenarios, c.machines)) < c.failure_probability
    base_cost = np.zeros(a*scenarios)
    tail_cost = np.zeros(a*scenarios)
    for i in range(h+tail):
        if i:
            selected = reference_batch(c, s, h-i if i < h else h+tail-i,
                                       "adaptive_isolated_matching" if i < h else TAIL_RULE)
        s, costs = batch_tick(c, s, selected, np.tile(events[i], (a, 1)))
        if i < h:
            base_cost += costs
        else:
            tail_cost += costs
    base_cost = base_cost.reshape(a, scenarios)
    tail_cost = tail_cost.reshape(a, scenarios)
    total = base_cost+tail_cost
    means = total.mean(1)
    best = int(means.argmin())
    return candidates[best], dict(candidates=candidates, means=means.tolist(),
        base_means=base_cost.mean(1).tolist(), tail_means=tail_cost.mean(1).tolist(),
        mc_se=(total.std(1, ddof=1)/math_sqrt(scenarios)).tolist(),
        selected_score=float(means[best]), scenario_seed=forecast_seed,
        forecast_ticks=h+tail, terminal_ticks=tail, restricted_allocation=fixed)


def math_sqrt(value):
    return float(np.sqrt(value))


def snapshot(state):
    servicing = {m for m in state.assigned if m >= 0}
    return dict(pending=sum(f and m not in servicing for m, f in enumerate(state.failed)),
                failed=sum(state.failed), servicing=len(servicing),
                cm_in_service=sum(state.failed[m] for m in servicing))


def request_metrics(rows):
    served = [r["wait"] for r in rows if r["served"]]
    return dict(requests=len(rows), served_requests=len(served),
                censored_requests=sum(r["censored"] for r in rows),
                overdue_requests=sum(r["overdue"] for r in rows),
                observed_wait=sum(r["wait"] for r in rows),
                served_wait_mean=float(np.mean(served)) if served else None,
                served_wait_p95=float(np.quantile(served,.95,method="inverted_cdf")) if served else None)


def episode(config, cohort, shock, condition, horizon, controller, best_rule, cfg,
            journals=None, time_totals=None):
    tail = cfg["tail_ticks"]
    c = replace(config, horizon=horizon+tail)
    env = WaitingEnv(c)
    env_seed = role_seed(shock, f"{VERSION}:physical:{cohort}")
    env.reset(env_seed)
    base_requests, all_requests = Requests(env.state), Requests(env.state)
    identity = dict(condition=condition, cohort=cohort, shock=shock, horizon=horizon, controller=controller)
    base_metrics = None
    interior_begin, interior_end = horizon//4, horizon-horizon//4
    interior = defaultdict(float)
    online_latency = []
    tail_latency = []
    for t in range(c.horizon):
        state = env.state
        phase = "base" if t < horizon else "tail"
        forecast_seed = role_seed(shock, f"{VERSION}:forecast:{cohort}:{t}")
        start = time.perf_counter_ns()
        if phase == "tail":
            action = prior.rule_action(c, state, c.horizon-t, TAIL_RULE)
            diag = dict(candidates=(), forecast_ticks=0, terminal_ticks=0, selected_score=None)
        elif controller in CONTROLLERS[1:]:
            action, diag = forecast_plan(c, state, horizon-t,
                tail if controller.endswith("tail") else 0, forecast_seed,
                cfg["scenarios"], fixed=controller.startswith("fixed"))
        else:
            action = prior.rule_action(c, state, horizon-t, best_rule if controller == "best_rule" else controller)
            diag = dict(candidates=(), forecast_ticks=0, terminal_ticks=0, selected_score=None)
        validate_matching(c, state, action)
        elapsed = (time.perf_counter_ns()-start)/1e6
        (online_latency if phase == "base" else tail_latency).append(elapsed)
        _, _, _, info = env.step(action)
        selected = np.zeros((1, c.machines, c.technicians), dtype=bool)
        for m, j in action:
            selected[0, m, j] = True
        following, costs = batch_tick(c, arrays(state), selected, np.asarray([env.events[t]]))
        if not all(np.array_equal(following[k][0], getattr(env.state, k)) for k in following):
            raise RuntimeError("scalar/batch state mismatch")
        if not np.isclose(costs[0], info["objective"], atol=1e-8, rtol=0):
            raise RuntimeError("scalar/batch cost mismatch")
        all_requests.step(t, state, action, env.state)
        if phase == "base":
            base_requests.step(t, state, action, env.state)
        if interior_begin <= t < interior_end:
            for metric in ("objective", "maintenance_cost", "failure_cost", "unavailability_cost", "waiting_cost", "overdue_waiting_ticks"):
                interior[metric] += info[metric]
            for metric, value in snapshot(state).items():
                interior[metric] += value
        if time_totals is not None:
            totals = time_totals[condition, horizon, controller, t, phase]
            totals["episodes"] += 1
            for metric, value in info.items():
                totals[metric] += value
            for metric, value in snapshot(state).items():
                totals[f"before_{metric}"] += value
            for metric, value in snapshot(env.state).items():
                totals[f"after_{metric}"] += value
        if journals:
            journals["decisions"].add({**identity, "time": t, "phase": phase,
                "env_seed": env_seed, "forecast_seed": forecast_seed,
                "state": prior.compact(asdict(state)), "action": prior.compact(action),
                "following": prior.compact(asdict(env.state)), "physical_metrics": prior.compact(info)})
            journals["latency"].add({**identity, "time": t, "phase": phase,
                "latency_ms": elapsed, "candidate_count": len(diag["candidates"]),
                "forecast_scenarios": cfg["scenarios"] if diag["forecast_ticks"] else 0,
                "forecast_ticks": diag["forecast_ticks"], "terminal_ticks": diag["terminal_ticks"],
                "forecast_transition_samples": len(diag["candidates"])*cfg["scenarios"]*diag["forecast_ticks"],
                "predicted_action_cost": diag["selected_score"]})
        if t+1 == horizon:
            base_metrics = env.metrics.copy()
            cutoff = snapshot(env.state)
            cut_requests = [{**r, "overdue": r["wait"] > c.waiting_limit} for r in base_requests.finish(horizon)]
    extended_requests = [{**r, "overdue": r["wait"] > c.waiting_limit} for r in all_requests.finish(c.horizon)]
    extended_index = {(r["machine"],r["request"]): r for r in extended_requests}
    carryover = []
    for r in cut_requests:
        if r["censored"]:
            later = extended_index[r["machine"],r["request"]]
            carryover.append(dict(machine=r["machine"], request=r["request"], arrival=r["arrival"],
                                  wait_at_cutoff=r["wait"], started_in_tail=later["served"],
                                  wait_at_start_or_end=later["wait"], censored_after_tail=later["censored"]))
    if journals:
        for scope, rows in (("cutoff",cut_requests),("extended",extended_requests)):
            for r in rows:
                journals["requests"].add({**identity, "scope": scope, **r})
        for r in carryover:
            journals["carryover"].add({**identity, **r})
        journals["coordination"].add({**identity, "base_decisions": horizon,
            "tail_decisions": tail, "invalid_actions": 0, "scalar_batch_replay": "PASS"})
    window = interior_end-interior_begin
    result = {**identity,
        **{f"base_{k}": v for k,v in base_metrics.items()},
        **{f"tail_{k}": v-base_metrics[k] for k,v in env.metrics.items()},
        **{f"extended_{k}": v for k,v in env.metrics.items()},
        **{f"cutoff_{k}": v for k,v in cutoff.items()},
        **{f"post_tail_{k}": v for k,v in snapshot(env.state).items()},
        **{f"cutoff_{k}": v for k,v in request_metrics(cut_requests).items()},
        **{f"extended_{k}": v for k,v in request_metrics(extended_requests).items()},
        "base_cost_per_machine_tick": base_metrics["objective"]/(c.machines*horizon),
        "extended_cost_per_machine_tick": env.metrics["objective"]/(c.machines*c.horizon),
        "interior_begin": interior_begin, "interior_end": interior_end,
        "interior_cost_per_machine_tick": interior["objective"]/(c.machines*window),
        "interior_pending_mean": interior["pending"]/window,
        "interior_failed_mean": interior["failed"]/window,
        "interior_overdue_per_tick": interior["overdue_waiting_ticks"]/window,
        **{f"interior_{k}": interior[k] for k in
           ("objective","maintenance_cost","failure_cost","unavailability_cost","waiting_cost")},
        "carryover_requests": len(carryover),
        "carryover_started_in_tail": sum(r["started_in_tail"] for r in carryover),
        "carryover_censored_after_tail": sum(r["censored_after_tail"] for r in carryover),
        "latency_mean_ms": float(np.mean(online_latency)),
        "latency_p50_ms": float(np.quantile(online_latency,.5)),
        "latency_p95_ms": float(np.quantile(online_latency,.95)),
        "latency_p99_ms": float(np.quantile(online_latency,.99)),
        "tail_rule_latency_mean_ms": float(np.mean(tail_latency)),
    }
    if not np.isclose(result["extended_objective"],result["base_objective"]+result["tail_objective"],atol=1e-8,rtol=0):
        raise RuntimeError("base/tail reconciliation")
    return result


def select_development(rows, cfg):
    chosen = {}
    for condition,horizon in itertools.product(CONDITIONS,cfg["horizons"]):
        metric = "interior_cost_per_machine_tick" if horizon==max(cfg["horizons"]) else "extended_objective"
        scores = {rule:float(np.mean([r[metric] for r in rows if r["condition"]==condition
                                     and r["horizon"]==horizon and r["controller"]==rule])) for rule in RULES}
        chosen[f"{condition}:{horizon}"] = dict(metric=metric,best_rule=min(RULES,key=scores.get),scores=scores)
    return chosen


def summary(rows, cfg):
    indexed={(r["condition"],r["horizon"],r["cohort"],r["shock"],r["controller"]):r for r in rows}
    expected=set(itertools.product(CONDITIONS,cfg["horizons"],cfg["test_cohorts"],cfg["test_shocks"],CONTROLLERS))
    if len(indexed)!=len(rows) or set(indexed)!=expected:
        raise RuntimeError("incomplete or duplicated test grid")
    metrics=("base_objective","tail_objective","extended_objective","base_cost_per_machine_tick",
             "extended_cost_per_machine_tick","cutoff_pending","cutoff_failed","cutoff_cm_in_service",
             "post_tail_pending","post_tail_failed","interior_cost_per_machine_tick",
             "interior_pending_mean","interior_failed_mean","interior_overdue_per_tick",
             "cutoff_censored_requests","extended_censored_requests","carryover_requests",
             "carryover_started_in_tail","carryover_censored_after_tail","latency_mean_ms",
             "base_maintenance_cost","base_failure_cost","base_unavailability_cost","base_waiting_cost",
             "base_overdue_waiting_ticks")
    means={}
    contrasts={}
    diffs=[]
    short,long=min(cfg["horizons"]),max(cfg["horizons"])
    for condition,horizon in itertools.product(CONDITIONS,cfg["horizons"]):
        cell=f"{condition}:{horizon}"
        means[cell]={name:{metric:float(np.mean([indexed[condition,horizon,c,s,name][metric]
                 for c in cfg["test_cohorts"] for s in cfg["test_shocks"]])) for metric in metrics} for name in CONTROLLERS}
        definitions=(("pending_terminal", "fixed_zero","fixed_tail","cutoff_pending",horizon==short),
                     ("interior_headroom", "best_rule","fixed_tail","interior_cost_per_machine_tick",horizon==long),
                     ("interior_assignment", "fixed_tail","flexible_tail","interior_cost_per_machine_tick",horizon==long))
        for label,left,right,metric,target in definitions:
            primary=condition=="skill_mask" and target
            confidence=cfg["primary_confidence"] if primary else .95
            a=np.array([np.mean([indexed[condition,horizon,c,s,left][metric] for s in cfg["test_shocks"]]) for c in cfg["test_cohorts"]])
            b=np.array([np.mean([indexed[condition,horizon,c,s,right][metric] for s in cfg["test_shocks"]]) for c in cfg["test_cohorts"]])
            rng=np.random.default_rng(role_seed(cfg["analysis_seed"],f"{cell}:{label}"))
            draws=rng.integers(len(a),size=(cfg["bootstrap_samples"],len(a)))
            delta=(a-b)[draws].mean(1)
            quantiles=[(1-confidence)/2,(1+confidence)/2]
            ci=np.quantile(delta,quantiles).tolist()
            relative_ci=None
            relative=None
            if a.mean()>0 and np.all(a[draws].mean(1)>0):
                relative=float((a-b).mean()/a.mean())
                relative_ci=np.quantile(delta/a[draws].mean(1),quantiles).tolist()
            screen_ci=ci if label=="pending_terminal" else relative_ci
            margin=cfg["pending_margin"] if label=="pending_terminal" else cfg["cost_margin"]
            status=("ENGINEERING_ONLY" if cfg["profile"]=="smoke" else
                    "INCONCLUSIVE" if screen_ci is None else
                    "PASS_RESEARCH_MARGIN" if screen_ci[0]>margin else
                    "BELOW_RESEARCH_MARGIN" if screen_ci[1]<margin else "INCONCLUSIVE")
            contrasts[f"{cell}:{label}"]=dict(primary=primary,label=label,left=left,right=right,
                metric=metric,confidence=confidence,profiles=len(a),mean_difference=float((a-b).mean()),
                confidence_interval=ci,relative_improvement=relative,relative_confidence_interval=relative_ci,
                profile_wins=int(np.sum(a>b)),profile_ties=int(np.sum(a==b)),status=status)
            for cohort,x,y in zip(cfg["test_cohorts"],a,b):
                diffs.append(dict(condition=condition,horizon=horizon,contrast=label,cohort=cohort,
                                  left_mean=float(x),right_mean=float(y),difference=float(x-y)))
    guard_means=means[f"skill_mask:{long}"]
    x,y=guard_means["fixed_tail"],guard_means["best_rule"]
    guard=dict(pending=x["interior_pending_mean"]<=y["interior_pending_mean"]+cfg["service_guard_pending"],
               total_failed=x["interior_failed_mean"]<=y["interior_failed_mean"]+cfg["service_guard_failed"],
               overdue=x["interior_overdue_per_tick"]<=y["interior_overdue_per_tick"]+cfg["service_guard_overdue_per_tick"])
    p2=contrasts[f"skill_mask:{long}:interior_headroom"]
    return dict(means=means,contrasts=contrasts,service_guard=guard,
                timing_learning_screen="ENGINEERING_ONLY" if cfg["profile"]=="smoke" else
                   "PASS" if p2["status"]=="PASS_RESEARCH_MARGIN" and all(guard.values()) else "NOT_ESTABLISHED",
                uncertainty_scope="profile bootstrap conditional on fixed shock panel",
                terminal_value_scope="finite continuation under shared deadline rule, not an infinite-horizon optimum"),diffs


def run(output, execution_profile="smoke", device="cpu"):
    if device!="cpu":
        raise ValueError("CPU only")
    output=Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("new/empty output directory required")
    output.mkdir(parents=True,exist_ok=True)
    cfg=settings(execution_profile)
    now=lambda:datetime.now(timezone.utc).isoformat()
    git=lambda *args:subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
    manifest=dict(protocol=VERSION,status="RUNNING",started=now(),commit=git("rev-parse","HEAD"),
                  dirty=bool(git("status","--porcelain")),settings=cfg,expected_counts=counts(cfg),
                  environment_version=ENV_VERSION,device=device,python=platform.python_version(),
                  platform=platform.platform(),machine=platform.machine(),logical_cpus=os.cpu_count(),
                  numpy=np.__version__,training="NOT_APPLICABLE",full_test_outcomes_opened=False,
                  thread_environment={k:os.environ.get(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")})
    prior.write_json(output/"manifest.json",manifest)
    journals={}
    try:
        manifest["seed_audit"]=seed_audit(cfg)
        panels,registry=build_panels(cfg)
        prior.write_json(output/"resolved_config.json",dict(settings=cfg,physical_profiles=registry))
        prior.write_json(output/"seed_registry.json",dict(raw_panels={k:v for k,v in cfg.items() if "cohort" in k or "shock" in k},
                         exclusion_registry=exclusion_registry()))
        shutil.copyfile(ROOT/f"docs/{VERSION}_plan.md",output/"plan.md")
        snapshot_dir=output/"source_snapshot"; snapshot_dir.mkdir()
        for filename in ("maintenance_boundary_sensitivity.py","maintenance_opportunity_cost.py",
                         "maintenance_dispatch.py","maintenance_waiting.py","maintenance_heterogeneous_model.py","maintenance_solver_diagnostics.py"):
            shutil.copyfile(ROOT/f"src/ht_pdm_fjsp/{filename}",snapshot_dir/filename)
        for filename in ("pyproject.toml","uv.lock"):
            shutil.copyfile(ROOT/filename,snapshot_dir/filename)
        for suffix in ("seed_registry", "profile_exclusions"):
            shutil.copyfile(ROOT/f"configs/{VERSION}_{suffix}.json",snapshot_dir/f"{VERSION}_{suffix}.json")
        journals["development"]=prior.Journal(output/"development_episodes.csv")
        development=[]
        grid=list(itertools.product(CONDITIONS,cfg["horizons"],cfg["development_cohorts"],cfg["development_shocks"],RULES))
        for cond,horizon,cohort,shock,rule in tqdm(grid,desc="Development horizon/rule evaluation"):
            row=episode(panels["development",cohort,cond],cohort,shock,cond,horizon,rule,None,cfg)
            development.append(row); journals["development"].add(row)
        chosen=select_development(development,cfg)
        prior.write_json(output/"development_selection.json",dict(locked_before_test=True,selected=chosen))
        for name in ("episodes.partial","decisions","requests","carryover","latency","coordination"):
            journals[name]=prior.Journal(output/f"{name}.csv")
        manifest["full_test_outcomes_opened"]=execution_profile=="full"
        prior.write_json(output/"manifest.json",manifest)
        rows=[]
        time_totals=defaultdict(lambda:defaultdict(float))
        grid=list(itertools.product(CONDITIONS,cfg["horizons"],cfg["test_cohorts"],cfg["test_shocks"],CONTROLLERS))
        for cond,horizon,cohort,shock,controller in tqdm(grid,desc="Paired multi-seed boundary evaluation"):
            row=episode(panels["test",cohort,cond],cohort,shock,cond,horizon,controller,
                        chosen[f"{cond}:{horizon}"]["best_rule"],cfg,journals,time_totals)
            rows.append(row); journals["episodes.partial"].add(row)
        result,differences=summary(rows,cfg)
        prior.write_json(output/"summary.json",result)
        prior.write_csv(output/"episodes.csv",rows)
        prior.write_csv(output/"profile_differences.csv",differences)
        time_rows=[]
        for (cond,horizon,controller,t,phase),values in sorted(time_totals.items()):
            time_rows.append(dict(condition=cond,horizon=horizon,controller=controller,time=t,phase=phase,
                                 episodes=int(values["episodes"]),
                                 **{k:v/values["episodes"] for k,v in values.items() if k!="episodes"}))
        prior.write_csv(output/"time_profiles.csv",time_rows)
        actual=dict(development_episodes=len(development),
            development_decisions=sum(r["horizon"]+cfg["tail_ticks"] for r in development),
            test_episodes=len(rows),base_decisions=sum(r["horizon"] for r in rows),
            tail_decisions=len(rows)*cfg["tail_ticks"],test_decisions=sum(r["horizon"]+cfg["tail_ticks"] for r in rows))
        if actual!=counts(cfg):
            raise RuntimeError("count audit mismatch")
        manifest.update(status="COMPLETED",audit="PASS",actual_counts=actual)
    except BaseException as error:
        manifest.update(status="FAILED",error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for journal in journals.values():
            journal.close()
        manifest["finished"]=now()
        manifest["artifacts"]={str(p.relative_to(output)):dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                               for p in sorted(output.rglob("*")) if p.is_file() and p.name!="manifest.json"}
        prior.write_json(output/"manifest.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile",choices=("smoke","full"),default="smoke")
    parser.add_argument("--device",choices=("cpu",),default="cpu")
    parser.add_argument("--output-dir",required=True,type=Path)
    args=parser.parse_args()
    run(args.output_dir,args.profile,args.device)


if __name__=="__main__":
    main()
