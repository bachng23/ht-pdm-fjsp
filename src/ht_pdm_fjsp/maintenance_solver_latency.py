"""Common development states, cache-isolated warm CPU inference benchmark."""

from dataclasses import asdict
import json
import time
import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_contention_headroom import write_csv, write_json
from ht_pdm_fjsp.maintenance_heterogeneous_model import isolated, gain
from ht_pdm_fjsp.maintenance_solver_model import CONDITIONS, cohort_config, reference, plan
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv


def benchmark_latency(output, cfg, selected_refs, selections, load_checkpoint):
    states = []
    h = cfg["horizons"][0]
    seed = cfg["development_seeds"][0]
    for cond in CONDITIONS:
        c = cohort_config(cfg["development_cohorts"][0], cond, h, profile=cfg["profile"])
        env = WaitingEnv(c)
        env.reset(seed)
        for t in range(h):
            states.append((cond, c, env.state, h-t))
            env.step(reference(c, env.state, h-t, selected_refs[cond]))
    write_json(output / "latency_states.json", [dict(condition=cond, config=asdict(c), state=asdict(s), remaining=h) for cond,c,s,h in states])
    controllers = [(name,-1,None) for name in cfg["controls"]] + [
        (v["algorithm"],v["train_seed"],v["checkpoint"]) for v in selections.values()
    ]
    rows, summaries = [], []
    for name, train_seed, path in tqdm(controllers, desc="common-state warm latency", unit="controller"):
        model = load_checkpoint(output / path)[0] if path else None
        isolated.cache_clear()
        gain.cache_clear()
        for repeat in range(4):
            for i,(cond,c,state,h) in enumerate(states):
                begin = time.perf_counter_ns()
                if model is not None:
                    pairs,_ = model.select(c,state,h)
                elif name in ("joint_rollout","fixed_allocation_rollout"):
                    pairs,_ = plan(c,state,h,selected_refs[cond],role_seed(seed,f"solver_latency:{i}"),cfg["scenarios"],fixed=name=="fixed_allocation_rollout")
                else:
                    pairs = reference(c,state,h,name)
                seconds = (time.perf_counter_ns()-begin)/1e9
                validate_matching(c,state,pairs)
                if repeat:
                    rows.append(dict(controller=name,train_seed=train_seed,repeat=repeat,state_index=i,condition=cond,remaining=h,seconds=seconds,action=json.dumps(pairs)))
        measured = [r["seconds"] for r in rows if r["controller"]==name and r["train_seed"]==train_seed]
        summaries.append(dict(controller=name,train_seed=train_seed,count=len(measured),p50_seconds=float(np.percentile(measured,50)),p95_seconds=float(np.percentile(measured,95)),mean_seconds=float(np.mean(measured)),isolated_cache=isolated.cache_info()._asdict(),gain_cache=gain.cache_info()._asdict()))
        write_csv(output / "latency.csv",rows)
    write_json(output / "latency_summary.json",dict(
        protocol="same development states; per-controller cache clear; one untimed pass then three timed passes; one CPU torch thread",
        caveat="Includes shared full-tensor frontend for local policies; no measured distributed network deployment. States are selected-rule trajectories, not all possible operating states.",
        controllers=summaries,
    ))
    # Scientific rollout timings begin with empty DP caches after diagnostic work.
    isolated.cache_clear()
    gain.cache_clear()
