"""Frozen-checkpoint greedy versus exhaustive joint-MAP; no training or reselection."""
import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_solver_comparison as source
from ht_pdm_fjsp.maintenance_contention_headroom import Journal, file_hash, write_csv, write_json
from ht_pdm_fjsp.maintenance_solver_policy import CENTRAL, FIXED, LOCAL, MARL, CATALOGUE, TIE
from ht_pdm_fjsp.maintenance_solver_model import CONDITIONS, cohort_config, plan, reference
from ht_pdm_fjsp.maintenance_solver_diagnostics import Requests
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_heterogeneous_model import isolated, gain
from ht_pdm_fjsp.maintenance_joint_map import Decoder, matching_log_probabilities, best_index

VERSION = "maintenance_decoding_audit_v1"
SOURCE_COMMIT = "4b1cffb0cc1e2413e27f0e2c3a7730f8d3e3d7a8"
ROOT = Path(__file__).resolve().parents[2]
MODEL_ARMS = (CENTRAL, FIXED, LOCAL, MARL)


def settings(profile, source_config):
    full = profile == "full"
    return dict(profile=profile, training_steps=0, optimizer_steps=0,
        train_seeds=source_config["train_seeds"] if full else source_config["train_seeds"][:1],
        development_cohort=161000 if full else 165020,
        development_shock=170000 if full else 173000,
        cohorts=list(range(163000, 163032)) if full else [165020],
        shocks=list(range(171000, 171020)) if full else [173000, 173001],
        conditions=list(CONDITIONS), horizons=[12, 24] if full else [4, 6],
        scenarios=source_config["scenarios"], torch_threads=1,
        primary_condition="specialized", primary_horizon=12, primary_confidence=.975,
        practical_reduction=.02, waiting_margin=.02, pending_margin=.05,
        latency_repeats=3, decoding_tie_tolerance=TIE, normalization_tolerance=1e-5)


def controllers(cfg):
    return [(a, mode, seed) for a in MODEL_ARMS for seed in cfg["train_seeds"]
            for mode in (("greedy", "joint_map") if a in (LOCAL, MARL) else ("greedy",))] + [
                ("selected_rule", "reference", -1), ("joint_rollout", "reference", -1)]


def label(algorithm, decoder):
    return algorithm if decoder == "reference" else f"{algorithm}__{decoder}"


def counts(cfg):
    n = len(controllers(cfg))
    panel = len(cfg["cohorts"])*len(cfg["shocks"])*len(cfg["conditions"])
    states = len(cfg["conditions"])*cfg["horizons"][0]
    return dict(checkpoints=len(MODEL_ARMS)*len(cfg["train_seeds"]), training_steps=0,
        optimizer_steps=0, episodes=n*panel*len(cfg["horizons"]),
        decision_intervals=n*panel*sum(cfg["horizons"]),
        decoding_audits=2*len(cfg["train_seeds"])*states,
        latency_rows=n*states*cfg["latency_repeats"])


def seed_audit(cfg, source_config):
    registry_path = ROOT / "configs" / f"{VERSION}_seed_registry.json"
    registry = json.loads(registry_path.read_text())
    parent = ROOT / "configs" / registry["historical_registry"]
    old = set(json.loads(parent.read_text())["declared_seed_values"])
    for low, high in registry["inherited_solver_panels"]:
        old.update(range(low, high))
    for key in ("train_seeds", "development_cohorts", "development_seeds", "test_cohorts", "test_seeds"):
        old.update(source_config[key])
    full = {170000} | set(range(171000, 171020))
    smoke = set(range(173000, 173002))
    if full & smoke or (full | smoke) & old:
        raise ValueError("new shock seed overlaps historical/source panel")
    permitted = full if cfg["profile"] == "full" else smoke
    if not (set(cfg["shocks"]) | {cfg["development_shock"]}) <= permitted:
        raise ValueError("shock outside declared panel")
    if cfg["profile"] == "smoke" and set(cfg["cohorts"]) & set(range(163000, 163032)):
        raise ValueError("smoke opened full test profile")
    return dict(status="PASS", new_shocks_disjoint=True, original_model_and_profile_seeds_reused=True,
        full_shock_panel_opened=False, registry_sha256=file_hash(registry_path),
        historical_registry_sha256=file_hash(parent))


def checked_path(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"unsafe or missing source artifact: {relative}")
    return path


def preflight(directory, profile):
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "COMPLETED" or manifest["experiment"] != source.VERSION:
        raise ValueError("source run must be COMPLETED solver-comparison v1")
    if profile == "full" and (manifest["commit"] != SOURCE_COMMIT or manifest["config"]["profile"] != "full"
                              or manifest["dirty"] or manifest["expected_counts"] != manifest["actual_counts"]):
        raise ValueError("full requires the clean completed original full source run")
    cfg = settings(profile, manifest["config"])
    if profile == "full" and cfg["train_seeds"] != list(range(160000, 160010)):
        raise ValueError("source replica panel mismatch")
    verified = {}

    def verify(relative):
        path = checked_path(directory, relative)
        meta = manifest["artifacts"][relative]
        digest = file_hash(path)
        if digest != meta["sha256"] or path.stat().st_size != meta["bytes"]:
            raise ValueError(f"source hash mismatch: {relative}")
        verified[relative] = dict(sha256=digest, bytes=path.stat().st_size)
        return path

    for filename in ("checkpoint_selection.json", "reference_selection.json", "benchmark_config.json"):
        verify(filename)
    selection = json.loads((directory / "checkpoint_selection.json").read_text())
    refs = json.loads((directory / "reference_selection.json").read_text())
    if not selection["frozen_before_test"] or not refs["frozen_before_test"]:
        raise ValueError("source selection was not frozen")
    if json.loads((directory / "benchmark_config.json").read_text()) != manifest["config"]:
        raise ValueError("source config mismatch")
    for relative in manifest["artifacts"]:
        if relative.startswith("source_snapshot/"):
            path = verify(relative)
            name = path.name
            candidates = [ROOT / "src/ht_pdm_fjsp" / name, ROOT / name,
                          ROOT / "docs" / name, ROOT / "configs" / name]
            current = next((p for p in candidates if p.is_file()), None)
            if current is None or file_hash(current) != file_hash(path):
                raise ValueError(f"inherited source changed: {name}")
    models, checkpoints = {}, {}
    for arm, seed in itertools.product(MODEL_ARMS, cfg["train_seeds"]):
        record = selection["selections"][f"{arm}:{seed}"]
        path = verify(record["checkpoint"])
        model, payload = source.load_checkpoint(path)
        if (record["algorithm"] != arm or record["train_seed"] != seed or
            record["checkpoint_sha256"] != verified[record["checkpoint"]]["sha256"] or
            payload["algorithm"] != arm or payload["train_seed"] != seed or
            payload["physical_steps"] != record["selected_steps"] or
            payload["settings"] != manifest["config"]):
            raise ValueError("checkpoint seed/algorithm/steps/settings provenance mismatch")
        models[(arm, seed)] = model
        checkpoints[f"{arm}:{seed}"] = record
    audits = seed_audit(cfg, manifest["config"])
    return cfg, refs["selected"], models, checkpoints, dict(
        source_run=str(directory), source_manifest_sha256=file_hash(manifest_path),
        source_commit=manifest["commit"], source_profile=manifest["config"]["profile"],
        verified_artifacts=verified, checkpoint_selection=checkpoints, seed_audit=audits)


def episode(c, shock, algorithm, mode, seed, cohort, condition, actor, selected_ref, scenarios,
            request_journal):
    env = WaitingEnv(c)
    env.reset(shock)
    requests = Requests(env.state)
    max_wait = 0
    elapsed = []
    max_mass_error = max_regret = 0.
    identity = dict(controller=label(algorithm, mode), algorithm=algorithm, decoder=mode,
                    train_seed=seed, cohort=cohort, condition=condition, horizon=c.horizon, seed=shock)
    for t in range(c.horizon):
        state = env.state
        begin = time.perf_counter_ns()
        if actor is None:
            if algorithm == "joint_rollout":
                pairs, diag = plan(c, state, c.horizon-t, selected_ref,
                    role_seed(shock, f"{VERSION}_planner:{t}"), scenarios)
            else:
                pairs, diag = reference(c, state, c.horizon-t, selected_ref), {}
        else:
            pairs, diag = actor.select(c, state, c.horizon-t)
        elapsed.append((time.perf_counter_ns()-begin)/1e9)
        validate_matching(c, state, pairs)
        serving = set(state.assigned) | {m for m, _ in pairs}
        max_wait = max(max_wait, max(state.pending_wait[m] + int(state.failed[m] and m not in serving)
                                     for m in range(c.machines)))
        max_mass_error = max(max_mass_error, abs(diag.get("probability_mass", 1.)-1.))
        max_regret = max(max_regret, diag.get("numerical_selection_regret", 0.))
        env.step(pairs)
        requests.step(t, state, pairs, env.state)
    request_rows = requests.finish(c.horizon)
    pending = sum(env.state.failed[m] and m not in env.state.assigned for m in range(c.machines))
    if pending != sum(r["censored"] for r in request_rows) or len(request_rows) != env.metrics["failures"]:
        raise RuntimeError("request/physical failure/terminal reconciliation")
    for r in request_rows:
        request_journal.add({**identity, **r, "overdue_observed":r["wait"] > c.waiting_limit})
    return dict(**identity, **env.metrics, max_wait=max_wait, waiting_violation=int(max_wait>c.waiting_limit),
        terminal_pending=pending, terminal_failed=sum(env.state.failed),
        corrective_requests=len(request_rows), served_requests=sum(r["served"] for r in request_rows),
        censored_requests=sum(r["censored"] for r in request_rows),
        request_overdue_observed=sum(r["wait"]>c.waiting_limit for r in request_rows),
        terminal_zero_wait_censored=sum(r["censored"] and r["wait"]==0 for r in request_rows),
        inference_seconds=sum(elapsed), decision_latency_p50=float(np.percentile(elapsed,50)),
        decision_latency_p95=float(np.percentile(elapsed,95)),
        maximum_mass_error=max_mass_error, maximum_map_selection_regret=max_regret)


def common_states(cfg, refs):
    states = []
    h = cfg["horizons"][0]
    for cond in cfg["conditions"]:
        c = cohort_config(cfg["development_cohort"], cond, h, profile=cfg["profile"])
        env = WaitingEnv(c)
        env.reset(cfg["development_shock"])
        for t in range(h):
            states.append((cond, c, env.state, h-t))
            env.step(reference(c, env.state, h-t, refs[cond]))
    return states


def diagnostics(out, cfg, refs, models, audit_journal, latency_journal):
    states = common_states(cfg, refs)
    write_json(out / "latency_states.json", [dict(condition=cond, config=asdict(c), state=asdict(s), remaining=h)
                                             for cond,c,s,h in states])
    audits = latency_count = 0
    for (arm, seed), model in tqdm(list(models.items()), desc="common-state decoder audit", unit="model"):
        if arm not in (LOCAL, MARL):
            continue
        for i,(cond,c,s,h) in enumerate(states):
            indices, scores, mass = matching_log_probabilities(model,c,s,h)
            greedy, _ = model.select(c,s,h)
            pairs = [CATALOGUE[int(k)] for k in indices]
            j = pairs.index(greedy)
            m = best_index(scores)
            audit_journal.add(dict(algorithm=arm,train_seed=seed,state_index=i,condition=cond,remaining=h,
                probability_mass=mass,candidate_count=len(pairs),greedy_action=json.dumps(greedy),
                map_action=json.dumps(pairs[m]),disagreement=greedy!=pairs[m],
                greedy_log_probability=float(scores[j]),map_log_probability=float(scores[m]),
                maximum_log_probability=float(scores.max()),greedy_rank=1+int((scores>scores[j]+TIE).sum()),
                map_log_probability_gain=float(scores[m]-scores[j]),
                candidate_actions=json.dumps(pairs),candidate_log_probabilities=json.dumps(scores.tolist())))
            audits += 1
    for arm,mode,seed in tqdm(controllers(cfg), desc="common-state warm latency",unit="controller"):
        actor = Decoder(models[(arm,seed)],mode) if seed >= 0 else None
        isolated.cache_clear(); gain.cache_clear()
        for repeat in range(cfg["latency_repeats"]+1):
            for i,(cond,c,s,h) in enumerate(states):
                start = time.perf_counter_ns()
                if actor is not None:
                    pairs,_ = actor.select(c,s,h)
                elif arm == "joint_rollout":
                    pairs,_ = plan(c,s,h,refs[cond],role_seed(cfg["development_shock"], f"{VERSION}_latency:{i}"),cfg["scenarios"])
                else:
                    pairs = reference(c,s,h,refs[cond])
                seconds = (time.perf_counter_ns()-start)/1e9
                validate_matching(c,s,pairs)
                if repeat:
                    latency_journal.add(dict(controller=label(arm,mode),train_seed=seed,repeat=repeat,
                        state_index=i,condition=cond,remaining=h,seconds=seconds,action=json.dumps(pairs)))
                    latency_count += 1
    isolated.cache_clear(); gain.cache_clear()
    return audits, latency_count


def confidence_interval(values, confidence):
    if len(values) != 10:
        return None
    critical = {0.95:2.2621571627982055, 0.975:2.685010846816457}
    half = critical[confidence]*statistics.stdev(values)/math.sqrt(10)
    mean = statistics.mean(values)
    return [mean-half,mean+half]


def summarize(rows, cfg):
    expected = {(label(a,m),seed,cs,cond,h,shock) for a,m,seed in controllers(cfg)
                for cs,cond,h,shock in itertools.product(cfg["cohorts"],cfg["conditions"],cfg["horizons"],cfg["shocks"])}
    keys = [(r["controller"],r["train_seed"],r["cohort"],r["condition"],r["horizon"],r["seed"]) for r in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise RuntimeError("incomplete or duplicated closed-loop grid")
    metrics = ["objective","base_cost","maintenance_cost","failure_cost","unavailability_cost","waiting_cost",
        "failures","waiting_violation","terminal_pending","corrective_requests","served_requests","censored_requests",
        "terminal_zero_wait_censored","request_overdue_observed","max_wait","inference_seconds",
        "decision_latency_p50","decision_latency_p95"]
    groups = defaultdict(list)
    replicas = defaultdict(list)
    for r in rows:
        groups[(r["controller"],r["condition"],r["horizon"])].append(r)
        replicas[(r["controller"],r["condition"],r["horizon"],r["train_seed"])].append(r)
    def mean(panel,metric="objective"):
        return statistics.mean(r[metric] for r in panel)
    means = {f'{a}:{c}:H{h}':{m:mean(v,m) for m in metrics} for (a,c,h),v in groups.items()}
    contrasts, paired = {}, []
    for arm,cond,h in itertools.product((LOCAL,MARL),cfg["conditions"],cfg["horizons"]):
        left,right = label(arm,"joint_map"),label(arm,"greedy")
        differences = []
        for seed in cfg["train_seeds"]:
            a,b = replicas[(left,cond,h,seed)],replicas[(right,cond,h,seed)]
            delta = mean(a)-mean(b)
            differences.append(delta)
            paired.append(dict(algorithm=arm,condition=cond,horizon=h,train_seed=seed,
                map_cost=mean(a),greedy_cost=mean(b),difference=delta))
        a,b = groups[(left,cond,h)],groups[(right,cond,h)]
        primary = cond==cfg["primary_condition"] and h==cfg["primary_horizon"]
        confidence = cfg["primary_confidence"] if primary else .95
        ci = confidence_interval(differences,confidence) if cfg["profile"]=="full" else None
        reduction = 1-mean(a)/mean(b)
        gates = dict(priced_reduction_2pct=reduction>=cfg["practical_reduction"],
            wins_80pct=sum(d < -1e-9 for d in differences)>=.8*len(differences),
            ci_upper_below_zero=ci is not None and ci[1]<0,
            waiting_guard=mean(a,"waiting_violation")<=mean(b,"waiting_violation")+cfg["waiting_margin"],
            terminal_pending_guard=mean(a,"terminal_pending")<=mean(b,"terminal_pending")+cfg["pending_margin"])
        contrasts[f'{arm}:{cond}:H{h}'] = dict(map_cost=mean(a),greedy_cost=mean(b),
            difference=statistics.mean(differences),priced_reduction=reduction,
            base_reduction=1-mean(a,"base_cost")/mean(b,"base_cost"),
            confidence=confidence,ci=ci,wins=sum(d < -1e-9 for d in differences),replicates=len(differences),
            confirmatory=primary and cfg["profile"]=="full",gates=gates,
            status="ENGINEERING_ONLY" if cfg["profile"]=="smoke" else
                ("PASS" if all(gates.values()) else "FAIL") if primary else "EXPLORATORY")
    references = {}
    for arm,mode,cond,h in itertools.product((LOCAL,MARL),("greedy","joint_map"),cfg["conditions"],cfg["horizons"]):
        left = label(arm,mode)
        for right in (label(CENTRAL,"greedy"),label(FIXED,"greedy"),"selected_rule","joint_rollout"):
            differences = []
            for seed in cfg["train_seeds"]:
                a = replicas[(left,cond,h,seed)]
                b = replicas[(right,cond,h,seed)] if right.endswith("__greedy") else groups[(right,cond,h)]
                delta = mean(a)-mean(b)
                differences.append(delta)
            references[f'{left}_vs_{right}:{cond}:H{h}'] = dict(
                difference=statistics.mean(differences),
                ci=confidence_interval(differences,.95) if cfg["profile"]=="full" else None,
                confidence=.95,replicates=len(differences),
                wins=sum(d < -1e-9 for d in differences),
                status="ENGINEERING_ONLY" if cfg["profile"]=="smoke" else "EXPLORATORY")
    return dict(aggregate=means,contrasts=contrasts,reference_contrasts=references,
        uncertainty_scope="variation across original frozen training replicas, conditional on familiar labeled profiles and shared fresh shocks",
        interpretation="MAP is centralized probability maximization, not optimal cost or decentralized deployment. No new training or test checkpoint selection.",
        feasibility=dict(invalid_assignments=sum(r["invalid_assignments"] for r in rows),
            maximum_mass_error=max(r["maximum_mass_error"] for r in rows),
            maximum_map_selection_regret=max(r["maximum_map_selection_regret"] for r in rows))), paired


def auxiliary_summaries(output):
    """Request-level quantiles retain censoring; pooled latency is common-state only."""
    requests = defaultdict(Counter)
    wait_histograms = defaultdict(Counter)
    with (output / "request_metrics.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            key=f'{r["controller"]}:{r["condition"]}:H{r["horizon"]}'
            wait=int(r["wait"]); served=r["served"]=="True"; censored=r["censored"]=="True"
            if served==censored or wait!=int(r["end"])-int(r["arrival"]):
                raise RuntimeError("invalid request record")
            requests[key].update(requests=1,served=served,censored=censored,
                observed_overdue=r["overdue_observed"]=="True",terminal_zero_wait_censored=censored and wait==0)
            if served:wait_histograms[key][wait]+=1
    summary={}
    for key,count in requests.items():
        hist=wait_histograms[key]
        values={**count,"censored_fraction":count["censored"]/count["requests"]}
        if count["served"]:
            target=math.ceil(.95*count["served"]);cumulative=0
            for wait,n in sorted(hist.items()):
                cumulative+=n
                if cumulative>=target:break
            values.update(served_wait_mean=sum(w*n for w,n in hist.items())/count["served"],
                          served_wait_p95_nearest_rank=wait,served_wait_max=max(hist))
        summary[key]=values
    write_json(output / "request_summary.json",dict(groups=summary,
        caveat="Wait ends at service START. Served quantiles exclude censored requests; no starvation guarantee."))
    times=defaultdict(list)
    with (output / "latency.csv").open(newline="") as f:
        for r in csv.DictReader(f):times[f'{r["controller"]}:{r["condition"]}'].append(float(r["seconds"]))
    write_json(output / "latency_summary.json",dict(groups={key:dict(n=len(v),mean_seconds=statistics.mean(v),
        p50_seconds=float(np.percentile(v,50)),p95_seconds=float(np.percentile(v,95))) for key,v in times.items()},
        caveat="Same development states; one warm pass and three timed repeats/controller; CPU one thread; includes frontend. MAP centralized diagnostic; no communication measurement."))
    audits=defaultdict(Counter)
    with (output / "decoding_audit.csv").open(newline="") as f:
        for r in csv.DictReader(f):
            key=f'{r["algorithm"]}:{r["condition"]}'
            audits[key].update(states=1,disagreements=r["disagreement"]=="True",
                log_probability_gain=float(r["map_log_probability_gain"]))
    write_json(output / "decoding_summary.json",dict(groups={key:dict(**v,
        disagreement_fraction=v["disagreements"]/v["states"],mean_log_probability_gain=v["log_probability_gain"]/v["states"])
        for key,v in audits.items()},caveat="Common development states only; probability gain is not physical cost gain."))


def run(output, directory, profile):
    output,directory = Path(output).resolve(),Path(directory).resolve()
    if output == directory or output.is_relative_to(directory) or directory.is_relative_to(output):
        raise ValueError("output must be separate from read-only source run")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new/empty; no resume")
    def git(*args):
        return subprocess.check_output(["git",*args],cwd=ROOT,text=True).strip()
    dirty = git("status","--porcelain")
    if profile=="full" and (platform.system()!="Linux" or dirty):
        raise ValueError("full requires human-run Linux lab and clean committed Git")
    torch.set_num_threads(1)
    cfg,refs,models,checkpoints,provenance = preflight(directory,profile)
    expected = counts(cfg)
    output.mkdir(parents=True,exist_ok=True)
    manifest = dict(experiment=VERSION,status="RUNNING",commit=git("rev-parse","HEAD"),dirty=dirty,
        started=datetime.now(timezone.utc).isoformat(),config=cfg,expected_counts=expected,
        runtime=dict(python=platform.python_version(),platform=platform.platform(),torch=str(torch.__version__),
                     device="cpu",torch_threads=1),source_manifest_sha256=provenance["source_manifest_sha256"])
    write_json(output / "manifest.json",manifest)
    journals = {}
    actual = dict(checkpoints=len(models),training_steps=0,optimizer_steps=0,episodes=0,
                  decision_intervals=0,decoding_audits=0,latency_rows=0)
    try:
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for path in [*Path(__file__).parent.glob("maintenance_decoding*.py"),
                     Path(__file__).with_name("maintenance_joint_map.py"),
                     ROOT / "docs" / f"{VERSION}_plan.md",ROOT / "configs" / f"{VERSION}_seed_registry.json"]:
            shutil.copy2(path,snapshot/path.name)
        shutil.copytree(directory / "source_snapshot",output / "inherited_source_snapshot")
        for name in ("manifest.json","checkpoint_selection.json","reference_selection.json"):
            dest=output/"source_metadata"/name;dest.parent.mkdir(exist_ok=True);shutil.copy2(directory/name,dest)
        write_json(output / "source_provenance.json",provenance)
        write_json(output / "benchmark_config.json",cfg)
        write_json(output / "seed_audit.json",provenance["seed_audit"])
        configs = {f'{cs}:{cond}:H{h}':cohort_config(cs,cond,h,profile=profile)
                   for cs,cond,h in itertools.product(cfg["cohorts"],cfg["conditions"],cfg["horizons"])}
        write_json(output / "evaluation_configs.json",{k:asdict(v) for k,v in configs.items()})
        for key,filename in [("episodes","episodes.partial.csv"),("requests","request_metrics.csv"),
                             ("audits","decoding_audit.csv"),("latency","latency.csv")]:
            journals[key]=Journal(output/filename)
        actual["decoding_audits"],actual["latency_rows"] = diagnostics(output,cfg,refs,models,journals["audits"],journals["latency"])
        # This is the first access to the fresh full closed-loop shock panel.
        provenance["seed_audit"]["full_shock_panel_opened"] = profile=="full"
        write_json(output / "seed_audit.json",provenance["seed_audit"])
        rows = []
        for arm,mode,seed in tqdm(controllers(cfg),desc="closed-loop controller replicas",unit="controller"):
            actor = Decoder(models[(arm,seed)],mode) if seed>=0 else None
            panel = list(itertools.product(cfg["cohorts"],cfg["conditions"],cfg["horizons"],cfg["shocks"]))
            for cs,cond,h,shock in tqdm(panel,desc=f'{arm}/{mode}/seed{seed}',leave=False,unit="episode"):
                c = configs[f'{cs}:{cond}:H{h}']
                row = episode(c,shock,arm,mode,seed,cs,cond,actor,refs[cond],cfg["scenarios"],journals["requests"])
                rows.append(row);journals["episodes"].add(row)
                actual["episodes"]+=1;actual["decision_intervals"]+=h
        summary,paired = summarize(rows,cfg)
        write_json(output / "summary.json",summary)
        write_csv(output / "paired_replica_metrics.csv",paired)
        auxiliary_summaries(output)
        # Verify every input used for provenance still identical after evaluation.
        for relative,meta in provenance["verified_artifacts"].items():
            if file_hash(checked_path(directory,relative))!=meta["sha256"]:
                raise RuntimeError("source artifact changed during evaluation")
        if file_hash(directory/"manifest.json")!=provenance["source_manifest_sha256"]:
            raise RuntimeError("source manifest changed during evaluation")
        if actual!=expected:
            raise RuntimeError(f"run count mismatch: {actual} != {expected}")
        journals["episodes"].close();del journals["episodes"]
        shutil.copy2(output / "episodes.partial.csv",output / "episodes.csv")
        manifest["status"]="COMPLETED"
    except BaseException as exc:
        manifest.update(status="FAILED",error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for journal in journals.values():journal.close()
        manifest.update(finished=datetime.now(timezone.utc).isoformat(),actual_counts=actual)
        manifest["artifacts"]={str(p.relative_to(output)):dict(bytes=p.stat().st_size,sha256=file_hash(p))
            for p in output.rglob("*") if p.is_file() and p!=output/"manifest.json"}
        write_json(output/"manifest.json",manifest)
    return manifest


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument("--profile",choices=("smoke","full"),default="smoke")
    parser.add_argument("--device",choices=("cpu",),default="cpu")
    parser.add_argument("--source-run",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path)
    args=parser.parse_args()
    output=args.output_dir or ROOT/"artifacts"/f'{VERSION}_{args.profile}_cpu_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}'
    run(output,args.source_run,args.profile)


if __name__=="__main__":
    main()
