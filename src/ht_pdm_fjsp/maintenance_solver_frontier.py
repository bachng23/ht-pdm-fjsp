"""Frozen centralized quality/compute frontier. No scientific retraining or scaling."""

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_solver_comparison as base
from ht_pdm_fjsp import maintenance_training_coverage as coverage
from ht_pdm_fjsp.maintenance_contention_headroom import write_json, write_csv, file_hash
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv
from ht_pdm_fjsp.maintenance_solver_model import (
    CONDITIONS,
    REFERENCES,
    cohort_config,
    reference,
    profile_audit,
)
from ht_pdm_fjsp.maintenance_solver_policy import CENTRAL
from ht_pdm_fjsp.maintenance_heterogeneous_model import isolated, gain
from ht_pdm_fjsp.maintenance_frontier_planner import Controller
from ht_pdm_fjsp.maintenance_frontier_statistics import summarize

VERSION = "maintenance_solver_frontier_v1"
ROOT = base.ROOT
SOURCE_COMMIT = "53d180a5fc18c53e850eff7278d753fbe295d647"
SOURCE_RUN = "maintenance_training_coverage_v1_full_cpu_20261009T044236Z"


def settings(profile):
    if profile not in ("full", "smoke"):
        raise ValueError("unknown profile")
    full = profile == "full"
    budgets = [8, 32, 128] if full else [2, 4, 8]
    return dict(
        profile=profile,
        conditions=list(CONDITIONS),
        horizons=[12, 24] if full else [4, 6],
        model_seeds=list(range(180000, 180010)) if full else [180000],
        development_cohorts=list(range(161000, 161016)) if full else [165020],
        test_cohorts=list(range(163000, 163032)) if full else [165010],
        development_seeds=list(range(204000, 204020)) if full else [209020],
        test_seeds=list(range(205000, 205020)) if full else [209040, 209041],
        planner_budgets=budgets,
        controls=["selected_rule", *[f"planner_{b}" for b in budgets]],
        latency_repeats=16 if full else 2,
        timing_seed=206000 if full else 209060,
        bootstrap_seed=207000 if full else 209061,
        bootstrap_draws=20000,
        cost_margin=0.02,
        waiting_margin=0.02,
        pending_margin=0.05,
        latency_ratio_threshold=0.5,
        torch_threads=1,
        selection_horizon=12 if full else 4,
        planner_implementation="vectorized all feasible roots, rule continuation, full remaining horizon",
    )


def counts(cfg):
    cells = len(cfg["conditions"]) * len(cfg["horizons"])
    jobs = len(cfg["controls"]) + len(cfg["model_seeds"])
    panel = len(cfg["test_cohorts"]) * len(cfg["test_seeds"])
    states = (
        len(cfg["development_cohorts"]) * len(cfg["conditions"]) * sum(cfg["horizons"])
    )
    return dict(
        new_training_steps=0,
        frozen_models=len(cfg["model_seeds"]),
        development_episodes=len(cfg["development_cohorts"])
        * len(cfg["development_seeds"])
        * len(cfg["conditions"])
        * len(REFERENCES)
        * (1 + len(cfg["planner_budgets"])),
        test_episodes=jobs * cells * panel,
        test_decision_intervals=jobs
        * len(cfg["conditions"])
        * sum(cfg["horizons"])
        * panel,
        test_machine_rows=jobs * cells * panel * 4,
        common_states=states,
        warm_latency_rows=jobs * states * cfg["latency_repeats"],
        cold_latency_rows=jobs * cells,
    )


def seed_audit(cfg):
    registry = json.loads((ROOT / f"configs/{VERSION}_seed_registry.json").read_text())
    historical = json.loads(
        (
            ROOT / "configs/maintenance_solver_comparison_v1_seed_registry.json"
        ).read_text()
    )
    used = set(historical["declared_seed_values"])
    parent = json.loads(
        (
            ROOT / "configs/maintenance_training_horizon_v1_seed_registry.json"
        ).read_text()
    )
    for lo, hi in parent["inherited_panels"] + parent["new_panels"]:
        used.update(range(lo, hi))
    panels = [set(range(lo, hi)) for lo, hi in registry["new_panels"]]
    if any(p & used for p in panels) or any(
        a & b for a, b in itertools.combinations(panels, 2)
    ):
        raise ValueError("fresh seed overlap")
    frozen = settings(cfg["profile"])
    for key in ["development_seeds", "test_seeds", "timing_seed", "bootstrap_seed"]:
        if cfg[key] != frozen[key]:
            raise ValueError("unregistered evaluation seed")
    if cfg["model_seeds"] != frozen["model_seeds"]:
        raise ValueError("frozen model replicas")
    reserved = used | set().union(*panels)
    derived = {
        role_seed(s, f"solver_frontier_v1_forecast:{t}")
        for s in cfg["development_seeds"] + cfg["test_seeds"]
        for t in range(max(cfg["horizons"]))
    }
    if (
        len(derived)
        != (len(cfg["development_seeds"]) + len(cfg["test_seeds"]))
        * max(cfg["horizons"])
        or derived & reserved
    ):
        raise ValueError("forecast seed overlap")
    return dict(
        status="PASS",
        derived_forecast_seed_count=len(derived),
        new_panels=registry["new_panels"],
        registry_sha256=file_hash(ROOT / f"configs/{VERSION}_seed_registry.json"),
        fresh_shock_panel_opened=False,
        familiar_profiles_reused=True,
        training_seed_reuse="intentional frozen checkpoints; no new training",
    )


def preflight(source, cfg):
    source = Path(source)
    manifest = json.loads((source / "manifest.json").read_text())
    if (
        manifest["status"] != "COMPLETED"
        or manifest["experiment"] != coverage.VERSION
        or manifest["commit"] != SOURCE_COMMIT
        or manifest["dirty"]
        or manifest["config"]["profile"] != "full"
        or manifest["actual_counts"] != coverage.counts(manifest["config"])
    ):
        raise ValueError("source run contract")

    def verified(rel):
        p = source / rel
        meta = manifest["artifacts"][rel]
        if (
            not p.is_file()
            or p.stat().st_size != meta["bytes"]
            or file_hash(p) != meta["sha256"]
        ):
            raise ValueError(f"source artifact hash/size: {rel}")
        return p

    selection = json.loads(verified("checkpoint_selection.json").read_text())
    training = json.loads(verified("training_coverage.json").read_text())
    jobs = []
    for s in cfg["model_seeds"]:
        rel = f"covered/{CENTRAL}/train_seed_{s}/model.pt"
        path = verified(rel)
        chosen = selection[f"covered:{CENTRAL}:{s}"]
        if (
            chosen["checkpoint"] != rel
            or chosen["selection"] != "FINAL_ONLY"
            or chosen["selected_steps"] != 491520
            or chosen["checkpoint_sha256"] != file_hash(path)
        ):
            raise ValueError("final checkpoint selection")
        model, p = coverage.load_checkpoint(path)
        if (
            p["algorithm"] != CENTRAL
            or p["training_regime"] != "covered"
            or p["train_seed"] != s
            or p["physical_steps"] != 491520
            or p["rollout_update"] != 320
            or p["settings"]["hidden"] != 64
            or p["settings"]["profile"] != "full"
            or not all(torch.isfinite(v).all() for v in model.state_dict().values())
        ):
            raise ValueError("frozen source checkpoint identity/finite tensors")
        record = [
            r
            for r in training
            if r["training_regime"] == "covered"
            and r["algorithm"] == CENTRAL
            and r["train_seed"] == s
        ]
        if len(record) != 1 or record[0]["env_steps"] != 491520:
            raise ValueError("source training record")
        jobs.append(
            dict(
                name=CENTRAL,
                train_seed=s,
                checkpoint=rel,
                sha256=file_hash(path),
                source_training_wall_seconds=record[0]["training_wall_seconds"],
            )
        )
    return jobs, dict(
        source_commit=manifest["commit"],
        source_manifest_sha256=file_hash(source / "manifest.json"),
        source_runtime=manifest["runtime"],
        source_run=str(source.resolve()),
        relevant_artifact_hashes_verified=True,
        historical_training_cost="reported wall time; CPU seconds not available, no amortization break-even claim",
    )


def clear_caches():
    isolated.cache_clear()
    gain.cache_clear()


def controller(job, rules, continuations, cfg, output, shock):
    model = (
        coverage.load_checkpoint(output / job["checkpoint"])[0]
        if job["checkpoint"]
        else None
    )
    return Controller(
        job["name"],
        rules,
        continuations,
        max(cfg["planner_budgets"]),
        model=model,
        shock=shock,
    )


class DecisionJournal:
    def __init__(self, output, keep_trace):
        self.latency = coverage.CountedJournal(output / "closed_loop_latency.csv")
        self.trace = (
            coverage.CountedJournal(output / "decisions.csv") if keep_trace else None
        )
        self.samples = defaultdict(list)

    def add(self, row):
        key = (row["controller"], row["train_seed"], row["condition"], row["horizon"])
        self.samples[key].append(row["inference_seconds"])
        self.latency.add(
            {
                k: row[k]
                for k in [
                    "controller",
                    "train_seed",
                    "cohort",
                    "condition",
                    "horizon",
                    "seed",
                    "time",
                    "inference_seconds",
                ]
            }
        )
        if self.trace:
            self.trace.add(row)

    def close(self):
        self.latency.close()
        if self.trace:
            self.trace.close()


def evaluate(cfg, cs, cond, h, shock, job, ctrl, journals=None):
    ctrl.condition = cond
    ctrl.shock = shock
    identity = dict(
        training_regime="covered" if job["name"] == CENTRAL else "reference",
        train_seed=job["train_seed"],
        cohort=cs,
        condition=cond,
        horizon=h,
        checkpoint_step=491520 if job["checkpoint"] else -1,
    )
    tagged = (
        {k: coverage.Tagged(v, controller=job["name"]) for k, v in journals.items()}
        if journals
        else None
    )
    if journals:
        journals["requests"].episode.clear()
    # The inherited episode performs physical/cost/request audits. Controller
    # adapter supplies every algorithm through the same timed select interface.
    row = base.episode(
        cohort_config(cs, cond, h, profile=cfg["profile"]),
        shock,
        CENTRAL,
        None,
        2,
        identity,
        ctrl,
        tagged,
    )
    row["controller"] = job["name"]
    if journals:
        q = journals["requests"].episode
        if (
            q["overdue_ticks"] != row["overdue_waiting_ticks"]
            or q["requests"] != row["corrective_requests"]
            or q["censored"] != row["terminal_pending"]
            or q["served_waiting_cost"] + q["censored_waiting_cost"]
            != row["waiting_cost"]
        ):
            raise RuntimeError("request waiting reconciliation")
        row.update(
            served_waiting_cost=q["served_waiting_cost"],
            censored_waiting_cost=q["censored_waiting_cost"],
        )
    return row


def select_development(cfg, output, journal):
    scores = defaultdict(list)
    jobs = list(
        itertools.product(
            cfg["development_cohorts"],
            cfg["conditions"],
            cfg["development_seeds"],
            REFERENCES,
            ["selected_rule", *[f"planner_{b}" for b in cfg["planner_budgets"]]],
        )
    )
    rng = np.random.default_rng(cfg["timing_seed"])
    rng.shuffle(jobs)
    started, cpu = time.perf_counter(), time.process_time()
    for cs, cond, shock, rule, name in tqdm(
        jobs, desc="development rule/continuation", unit="episode"
    ):
        job = dict(name=name, train_seed=-1, checkpoint=None)
        ctrl = Controller(
            name,
            {cond: rule},
            {name: {cond: rule}},
            max(cfg["planner_budgets"]),
            shock=shock,
        )
        row = evaluate(cfg, cs, cond, cfg["selection_horizon"], shock, job, ctrl)
        journal.add({**row, "continuation": rule})
        scores[(name, cond, rule)].append(row["objective"])
    names = cfg["controls"]
    selected = {
        name: {
            cond: min(REFERENCES, key=lambda r: (np.mean(scores[(name, cond, r)]), r))
            for cond in cfg["conditions"]
        }
        for name in names
    }
    payload = dict(
        selected=selected,
        criterion="development mean objective; H12; lexical name ties",
        frozen_before_test=True,
        all_budgets_reported=True,
        scores={f"{a}:{c}:{r}": float(np.mean(v)) for (a, c, r), v in scores.items()},
    )
    write_json(output / "development_selection.json", payload)
    write_json(
        output / "reference_selection.json", dict(selected=selected["selected_rule"])
    )
    write_json(
        output / "planner_selection.json",
        dict(selected={a: selected[a] for a in names if a != "selected_rule"}),
    )
    return (
        selected["selected_rule"],
        {a: selected[a] for a in names if a != "selected_rule"},
        dict(
            phase="development_selection",
            wall_seconds=time.perf_counter() - started,
            cpu_seconds=time.process_time() - cpu,
        ),
    )


def peak_rss_bytes():
    x = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(x if platform.system() == "Darwin" else x * 1024)


def _latency_worker(output, cfg, job, rules, continuations, states, token):
    torch.set_num_threads(1)
    directory = output / "latency_workers" / token
    directory.mkdir(parents=True)
    start, cpu = time.perf_counter(), time.process_time()
    ctrl = controller(
        job, rules, continuations, cfg, output, cfg["development_seeds"][0]
    )
    load_seconds = time.perf_counter() - start
    warm = coverage.CountedJournal(directory / "warm.csv")
    cold = coverage.CountedJournal(directory / "cold.csv")
    samples = defaultdict(list)
    cold_seen = set()
    for i, (cond, c, state, h) in enumerate(states):
        cell = (cond, c.horizon)
        if cell in cold_seen:
            continue
        cold_seen.add(cell)
        clear_caches()
        ctrl.condition = cond
        began = time.perf_counter_ns()
        action, _ = ctrl.select(c, state, h)
        cold.add(
            dict(
                controller=job["name"],
                train_seed=job["train_seed"],
                condition=cond,
                horizon=c.horizon,
                state_index=i,
                seconds=(time.perf_counter_ns() - began) / 1e9,
                action=json.dumps(action),
            )
        )
    clear_caches()
    for cond, c, state, h in states:
        ctrl.condition = cond
        ctrl.select(c, state, h)
    rng = np.random.default_rng(cfg["timing_seed"])
    try:
        for repeat in range(cfg["latency_repeats"]):
            for i in rng.permutation(len(states)):
                cond, c, state, h = states[i]
                ctrl.condition = cond
                began = time.perf_counter_ns()
                action, _ = ctrl.select(c, state, h)
                seconds = (time.perf_counter_ns() - began) / 1e9
                samples[(cond, c.horizon)].append(seconds)
                warm.add(
                    dict(
                        controller=job["name"],
                        train_seed=job["train_seed"],
                        condition=cond,
                        horizon=c.horizon,
                        repeat=repeat,
                        state_index=int(i),
                        seconds=seconds,
                        action=json.dumps(action),
                    )
                )
    finally:
        warm.close()
        cold.close()
    write_json(
        directory / "summary.json",
        dict(
            cells={f"{c}_H{h}": latency_stats(v) for (c, h), v in samples.items()},
            resource=dict(
                phase="isolated_common_state_latency",
                controller=job["name"],
                train_seed=job["train_seed"],
                wall_seconds=time.perf_counter() - start,
                cpu_seconds=time.process_time() - cpu,
                peak_rss_bytes=peak_rss_bytes(),
                model_load_seconds=load_seconds,
                checkpoint_bytes=(output / job["checkpoint"]).stat().st_size
                if job["checkpoint"]
                else 0,
            ),
        ),
    )


def latency_stats(values):
    values = np.asarray(values, float)
    return dict(
        count=len(values),
        mean_seconds=float(values.mean()),
        p50_seconds=float(np.percentile(values, 50)),
        p95_seconds=float(np.percentile(values, 95)),
        p99_seconds=float(np.percentile(values, 99)),
        max_seconds=float(values.max()),
        tail_samples_above_p99=int(np.sum(values > np.percentile(values, 99))),
        p99_descriptive_only=True,
        p99_caveat="repeats on correlated states; sample count is not independent effective sample size",
    )


def benchmark_latency(output, cfg, jobs, rules, continuations):
    states = []
    for cs, cond, h in itertools.product(
        cfg["development_cohorts"], cfg["conditions"], cfg["horizons"]
    ):
        c = cohort_config(cs, cond, h, profile=cfg["profile"])
        env = WaitingEnv(c)
        env.reset(cfg["development_seeds"][0])
        for t in range(h):
            states.append((cond, c, env.state, h - t))
            env.step(reference(c, env.state, h - t, rules[cond]))
    write_json(
        output / "latency_states.json",
        [
            dict(condition=cond, config=asdict(c), state=asdict(s), remaining=h)
            for cond, c, s, h in states
        ],
    )
    records = defaultdict(list)
    resources = []
    context = mp.get_context("spawn")
    order = np.random.default_rng(cfg["timing_seed"]).permutation(len(jobs))
    for j in tqdm(order, desc="isolated common-state latency", unit="controller"):
        job = jobs[j]
        token = f"{job['name']}_{job['train_seed']}"
        p = context.Process(
            target=_latency_worker,
            args=(output, cfg, job, rules, continuations, states, token),
        )
        p.start()
        try:
            p.join()
        except BaseException:
            p.terminate()
            p.join()
            raise
        if p.exitcode:
            raise RuntimeError(f"latency worker failed {token}: exit {p.exitcode}")
        summary = json.loads(
            (output / "latency_workers" / token / "summary.json").read_text()
        )
        resources.append(summary["resource"])
        for cond, h in itertools.product(cfg["conditions"], cfg["horizons"]):
            records[(job["name"], cond, h)].append(summary["cells"][f"{cond}_H{h}"])
    common = {
        k: {
            metric: float(np.mean([r[metric] for r in rs]))
            for metric in ["mean_seconds", "p50_seconds", "p95_seconds", "p99_seconds"]
        }
        for k, rs in records.items()
    }
    write_json(
        output / "latency_summary.json",
        dict(
            aggregate={f"{a}:{c}_H{h}": v for (a, c, h), v in common.items()},
            protocol="serial isolated CPU processes; one cold per cell, one warm pass, randomized state order each measured pass",
            caveat="Aggregate RL quantiles are means of replica quantiles; all per-replica quantiles stored. Model select retains critic diagnostics. Child interpreter startup excluded.",
        ),
    )
    return common, resources, len(states)


def write_plot(output, summary, profile):
    # Static scientific artifact; no new plotting dependency needed.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    cells = list(
        itertools.product(
            CONDITIONS,
            sorted({int(k.rsplit("H", 1)[1]) for k in summary["aggregates"]}),
        )
    )
    fig, axes = plt.subplots(len(CONDITIONS), 2, figsize=(12, 13), squeeze=False)
    for ax, (cond, h) in zip(axes.flat, cells, strict=True):
        for key, v in summary["aggregates"].items():
            name, cell = key.split(":")
            if cell != f"{cond}_H{h}":
                continue
            ax.scatter(
                v["common_state_mean_seconds"] * 1000, v["objective"], label=name
            )
        ax.set_xscale("log")
        points = [
            v["common_state_mean_seconds"] * 1000
            for key, v in summary["aggregates"].items()
            if key.endswith(f":{cond}_H{h}")
        ]
        ticks = np.geomspace(min(points) * 0.9, max(points) * 1.1, 4)
        ax.set_xticks(ticks, [f"{x:.2g}" for x in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Common-state mean decision latency (ms)")
        ax.set_ylabel("Mean episode cost")
        ax.set_title(f"{cond}, H{h}")
        ax.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(
        f"{profile.upper()}: central solver quality / compute — descriptive points"
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(output / "quality_compute.png", dpi=180)
    fig.savefig(output / "quality_compute.pdf")
    plt.close(fig)


def run(output, source, profile):
    output, source = Path(output), Path(source)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new/empty; no resume")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if profile == "full" and (platform.system() != "Linux" or dirty):
        raise ValueError("full requires human-run clean committed Linux lab checkout")
    cfg = settings(profile)
    if profile == "full" and any(
        os.environ.get(k) != "1"
        for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
    ):
        raise ValueError("full requires explicit OMP/OPENBLAS/MKL_NUM_THREADS=1")
    torch.set_num_threads(1)
    audit = seed_audit(cfg)
    frozen, provenance = preflight(source, cfg)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        experiment=VERSION,
        status="RUNNING",
        config=cfg,
        expected_counts=counts(cfg),
        dirty=dirty,
        commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        started=datetime.now(timezone.utc).isoformat(),
        runtime=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            torch=str(torch.__version__),
            numpy=np.__version__,
            cpu=platform.processor(),
            logical_cpu_count=os.cpu_count(),
            threads=torch.get_num_threads(),
            thread_env={
                k: os.environ.get(k)
                for k in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]
            },
        ),
    )
    write_json(output / "manifest.json", manifest)
    journals = {}
    try:
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        files = [
            *Path(__file__).parent.glob("maintenance_frontier*.py"),
            Path(__file__),
            *Path(__file__).parent.glob("maintenance_solver*.py"),
            *Path(__file__).parent.glob("maintenance_heterogeneous*.py"),
            Path(coverage.__file__),
            ROOT / "src/ht_pdm_fjsp/maintenance_waiting.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_dispatch.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_dispatch_policy.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_coordination_references.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_contention_headroom.py",
            ROOT / "pyproject.toml",
            ROOT / "uv.lock",
            ROOT / f"docs/{VERSION}_plan.md",
            ROOT / f"configs/{VERSION}_seed_registry.json",
            ROOT / "configs/maintenance_training_horizon_v1_seed_registry.json",
            ROOT / "configs/maintenance_solver_comparison_v1_seed_registry.json",
        ]
        for path in dict.fromkeys(files):
            shutil.copy2(path, snapshot / path.name)
        write_json(output / "resolved_config.json", cfg)
        write_json(output / "seed_audit.json", audit)
        pa = profile_audit(profile)
        pa.update(familiar_test_profiles=True, fresh_shock_panel_opened=False)
        write_json(output / "profile_audit.json", pa)
        jobs = [dict(name=a, train_seed=-1, checkpoint=None) for a in cfg["controls"]]
        for job in frozen:
            target = output / job["checkpoint"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / job["checkpoint"], target)
            if file_hash(target) != job["sha256"]:
                raise RuntimeError("copied checkpoint hash")
        jobs.extend(frozen)
        write_json(
            output / "checkpoint_provenance.json",
            dict(**provenance, models=frozen, no_new_training=True),
        )
        for key, name in [
            ("development", "development_episodes.csv"),
            ("episodes", "episodes.partial.csv"),
            ("machines", "machine_metrics.csv"),
            ("requests", "request_metrics.csv"),
        ]:
            journals[key] = coverage.CountedJournal(output / name)
        rules, continuations, dev_resource = select_development(
            cfg, output, journals["development"]
        )
        common, resources, nstates = benchmark_latency(
            output, cfg, jobs, rules, continuations
        )
        resources.append(dev_resource)
        decisions = DecisionJournal(output, profile == "smoke")
        journals["decisions"] = decisions
        requests = coverage.RequestJournal(journals["requests"])
        eval_journals = dict(
            decisions=decisions, machines=journals["machines"], requests=requests
        )
        audit["fresh_shock_panel_opened"] = profile == "full"
        pa.update(
            full_test_rollouts_opened=profile == "full",
            fresh_shock_panel_opened=profile == "full",
        )
        write_json(output / "seed_audit.json", audit)
        write_json(output / "profile_audit.json", pa)
        rows = []
        rng = np.random.default_rng(cfg["timing_seed"])
        panel = list(
            itertools.product(
                cfg["test_cohorts"],
                cfg["conditions"],
                cfg["horizons"],
                cfg["test_seeds"],
            )
        )
        for j in tqdm(
            rng.permutation(len(jobs)), desc="sealed evaluation", unit="controller"
        ):
            job = jobs[j]
            clear_caches()
            ctrl = controller(job, rules, continuations, cfg, output, 0)
            start, cpu = time.perf_counter(), time.process_time()
            for index in tqdm(
                rng.permutation(len(panel)),
                desc=f"test {job['name']}/{job['train_seed']}",
                leave=False,
            ):
                cs, cond, h, shock = panel[index]
                row = evaluate(cfg, cs, cond, h, shock, job, ctrl, eval_journals)
                rows.append(row)
                journals["episodes"].add(row)
            resources.append(
                dict(
                    phase="closed_loop_evaluation",
                    controller=job["name"],
                    train_seed=job["train_seed"],
                    wall_seconds=time.perf_counter() - start,
                    cpu_seconds=time.process_time() - cpu,
                    peak_rss_bytes=peak_rss_bytes(),
                    peak_rss_scope="parent process lifetime high water, not isolated solver memory",
                )
            )
            write_csv(output / "resource_usage.csv", resources)
        journals["episodes"].close()
        shutil.copy2(output / "episodes.partial.csv", output / "episodes.csv")
        write_json(
            output / "closed_loop_latency_summary.json",
            {
                f"{a}:{s}:{c}_H{h}": latency_stats(v)
                for (a, s, c, h), v in decisions.samples.items()
            },
        )
        paired, summary = summarize(rows, cfg, common)
        write_csv(output / "paired_seed_metrics.csv", paired)
        write_json(output / "summary.json", summary)
        write_plot(output, summary, profile)
        write_json(
            output / "request_wait_decomposition.json",
            {
                f"{r}:{a}:{c}_H{h}": dict(v)
                for (r, a, c, h), v in requests.groups.items()
            },
        )
        write_json(
            output / "feasibility_audit.json",
            dict(
                status="PASS",
                all_actions_valid=True,
                physical_and_request_cost_reconciliation=True,
                complete_grid=True,
                pending_is_not_permanent_starvation=True,
            ),
        )
        actual = dict(
            new_training_steps=0,
            frozen_models=len(frozen),
            development_episodes=journals["development"].count,
            test_episodes=journals["episodes"].count,
            test_decision_intervals=decisions.latency.count,
            test_machine_rows=journals["machines"].count,
            common_states=nstates,
            warm_latency_rows=sum(
                sum(v["count"] for v in json.loads(p.read_text())["cells"].values())
                for p in output.glob("latency_workers/*/summary.json")
            ),
            cold_latency_rows=sum(
                sum(1 for _ in p.open()) - 1
                for p in output.glob("latency_workers/*/cold.csv")
            ),
        )
        if actual != counts(cfg):
            raise RuntimeError(f"population mismatch: {actual} != {counts(cfg)}")
        manifest.update(
            status="COMPLETED", actual_counts=actual, request_rows=requests.count
        )
    except BaseException as exc:
        manifest.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        for journal in journals.values():
            journal.close()
        manifest["finished"] = datetime.now(timezone.utc).isoformat()
        manifest["artifacts"] = {
            str(p.relative_to(output)): dict(
                bytes=p.stat().st_size, sha256=file_hash(p)
            )
            for p in output.rglob("*")
            if p.is_file() and p.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
    print(
        json.dumps(dict(output=str(output), status=manifest["status"], counts=actual))
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument(
        "--source-run", type=Path, default=ROOT / "artifacts" / SOURCE_RUN
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (
        args.output_dir
        or ROOT
        / "artifacts"
        / f"{VERSION}_{args.profile}_cpu_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    run(output.resolve(), args.source_run.resolve(), args.profile)


if __name__ == "__main__":
    main()
