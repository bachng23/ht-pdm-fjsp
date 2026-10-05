"""Contention/headroom diagnostic. No learning; run with --profile smoke first."""

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.maintenance_contention_model import (
    CAPACITIES,
    REFERENCES,
    actions,
    cohort_config,
    compact,
    physical_action,
    plan,
    reference,
    validate_model,
)
from ht_pdm_fjsp.maintenance_contention_oracle import run_exact
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv

VERSION = "maintenance_contention_headroom_v1"
ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields or ["status"])
        writer.writeheader()
        writer.writerows(rows)


class Journal:
    def __init__(self, path):
        self.stream = path.open("w", newline="")
        self.writer = None

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.stream, list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.stream.flush()

    def close(self):
        self.stream.close()


def settings(profile):
    full = profile == "full"
    return dict(
        profile=profile,
        cohorts=list(range(134000, 134010)) if full else [137000],
        development_seeds=list(range(135000, 135020)) if full else [137020],
        test_seeds=list(range(136000, 136050)) if full else list(range(137010, 137013)),
        horizons=[12, 24] if full else [4, 6],
        exact_horizon=6 if full else 3,
        scenarios=32 if full else 4,
        exact_state_cap=250000,
        controllers=[*REFERENCES, "joint_rollout"],
        capacities=CAPACITIES,
        seed_roles={
            "configuration": "contention_configuration",
            "initial": "initial",
            "hazard": "failures",
            "planner": "contention_planner:<time>",
        },
    )


def episode(c, seed, controller, continuation, scenarios, identity, journals=None):
    env = WaitingEnv(c)
    env.reset(seed)
    samples = [[] for _ in range(c.machines)]
    machine = [
        dict(
            machine=m,
            service_starts=0,
            unavailable_ticks=0,
            failed_waiting_ticks=0,
            overdue_waiting_ticks=0,
            max_wait=0,
        )
        for m in range(c.machines)
    ]
    excess = choices = backlog = utilization = subset_total = 0
    spreads, errors, planning_seconds = [], [], 0.0
    for t in range(c.horizon):
        state = env.state
        reduced = compact(state)
        feasible = actions(c, reduced)
        free = c.technicians - sum(r > 0 for r in state.remaining)
        urgent = sum(
            not row[3] and (row[1] or row[0] >= c.failure_age - 1) for row in reduced
        )
        urgent_excess = max(0, urgent - free)
        before = time.perf_counter()
        diagnostics = {}
        if controller == "joint_rollout":
            selected, diagnostics = plan(
                c,
                reduced,
                c.horizon - t,
                continuation,
                role_seed(seed, f"contention_planner:{t}"),
                scenarios,
            )
            spreads.append(diagnostics["score_spread"])
            errors.append(diagnostics["selected_mc_se"])
        else:
            selected = reference(c, reduced, c.horizon - t, controller)
        elapsed = time.perf_counter() - before
        planning_seconds += elapsed
        pairs = physical_action(state, selected)
        if selected not in feasible or len(pairs) != len(selected):
            raise AssertionError("infeasible decision")
        serving = {m for m in state.assigned if m >= 0} | set(selected)
        for m, row in enumerate(machine):
            waiting = bool(state.failed[m] and m not in serving)
            wait = state.pending_wait[m] + int(waiting)
            samples[m].append(wait)
            row["service_starts"] += int(m in selected)
            row["unavailable_ticks"] += int(state.failed[m] or m in serving)
            row["failed_waiting_ticks"] += int(waiting)
            row["overdue_waiting_ticks"] += int(waiting and wait > c.waiting_limit)
            row["max_wait"] = max(row["max_wait"], wait)
        _, _, _, info = env.step(pairs)
        excess += urgent_excess
        choices += len(feasible) > 1
        subset_total += len(feasible)
        backlog += sum(state.failed)
        utilization += len(serving)
        if journals:
            journals["decisions"].add(
                dict(
                    **identity,
                    seed=seed,
                    controller=controller,
                    time=t,
                    state=json.dumps(asdict(state)),
                    action=json.dumps(pairs),
                    following=json.dumps(asdict(env.state)),
                    physical_metrics=json.dumps(info),
                    urgent_excess=urgent_excess,
                    feasible_subsets=len(feasible),
                    planning_seconds=elapsed,
                    planner=json.dumps(diagnostics),
                )
            )
    for key in ("unavailable_ticks", "failed_waiting_ticks", "overdue_waiting_ticks"):
        if sum(row[key] for row in machine) != env.metrics[key]:
            raise AssertionError(f"machine reconciliation {key}")
    for m, row in enumerate(machine):
        row.update(
            terminal_failed=int(env.state.failed[m]),
            terminal_pending=int(env.state.failed[m] and m not in env.state.assigned),
            p95_wait=float(np.percentile(samples[m], 95)),
            violation=int(row["max_wait"] > c.waiting_limit),
        )
        if journals:
            journals["machines"].add(
                dict(**identity, seed=seed, controller=controller, **row)
            )
    return dict(
        **identity,
        seed=seed,
        controller=controller,
        **env.metrics,
        p95_wait=float(np.percentile(list(itertools.chain.from_iterable(samples)), 95)),
        max_wait=max(row["max_wait"] for row in machine),
        waiting_violation=int(any(row["violation"] for row in machine)),
        terminal_pending=sum(row["terminal_pending"] for row in machine),
        terminal_failed=sum(env.state.failed),
        utilization=utilization / (c.horizon * c.technicians),
        mean_failed_backlog=backlog / c.horizon,
        mean_urgent_excess=excess / c.horizon,
        choice_opportunity_frequency=choices / c.horizon,
        mean_feasible_subsets=subset_total / c.horizon,
        planner_mean_score_spread=float(np.mean(spreads)) if spreads else 0.0,
        planner_mean_mc_se=float(np.mean(errors)) if errors else 0.0,
        planning_seconds=planning_seconds,
    )


def summarize(rows, selected, cfg):
    paired, hypotheses = [], {}
    for capacity in CAPACITIES:
        baseline = selected[capacity]
        for horizon in cfg["horizons"]:
            differences, costs, basecosts = [], [], []
            for cohort in cfg["cohorts"]:
                panel = [
                    r
                    for r in rows
                    if r["capacity"] == capacity
                    and r["horizon"] == horizon
                    and r["cohort"] == cohort
                ]
                groups = {
                    name: [r for r in panel if r["controller"] == name]
                    for name in (baseline, "joint_rollout")
                }
                for group in groups.values():
                    if sorted(r["seed"] for r in group) != cfg["test_seeds"]:
                        raise AssertionError("incomplete paired panel")
                control, planner = (
                    float(np.mean([r["objective"] for r in groups[n]]))
                    for n in (baseline, "joint_rollout")
                )
                cb, pb = (
                    float(np.mean([r["base_cost"] for r in groups[n]]))
                    for n in (baseline, "joint_rollout")
                )
                differences.append(planner - control)
                costs.append((control, planner))
                basecosts.append((cb, pb))
                paired.append(
                    dict(
                        capacity=capacity,
                        horizon=horizon,
                        cohort=cohort,
                        reference=baseline,
                        reference_cost=control,
                        planner_cost=planner,
                        difference=planner - control,
                        reference_base_cost=cb,
                        planner_base_cost=pb,
                    )
                )
            control, planner = np.mean(costs, axis=0)
            cb, pb = np.mean(basecosts, axis=0)
            mean = float(np.mean(differences))
            half = (
                3.2498355415921254
                * float(np.std(differences, ddof=1))
                / math.sqrt(len(differences))
                if len(differences) == 10
                else None
            )
            reduction = float(1 - planner / control) if control else 0.0
            wins = sum(x < -1e-9 for x in differences)
            passed = (
                half is not None
                and reduction >= 0.05
                and wins >= 8
                and mean + half < 0
                and pb <= 1.1 * cb
            )
            hypotheses[f"{capacity}_H{horizon}"] = dict(
                reference=baseline,
                reduction=reduction,
                cohort_wins=wins,
                paired_difference=mean,
                ci99=[mean - half, mean + half] if half is not None else None,
                base_cost_ratio=float(pb / cb) if cb else None,
                status=("PASS" if passed else "FAIL")
                if cfg["profile"] == "full" and horizon == cfg["horizons"][0]
                else "ENGINEERING_ONLY"
                if cfg["profile"] == "smoke"
                else "EXPLORATORY",
                reference_priced_cost=float(control),
                planner_priced_cost=float(planner),
            )
    return paired, hypotheses


def seed_audit(cfg):
    path = ROOT / "configs/maintenance_contention_headroom_v1_seed_registry.json"
    registry = json.loads(path.read_text())
    historical = (
        set(registry["declared_seed_values"])
        | set(range(201, 301))
        | set(range(601, 701))
    )
    panels = [set(cfg[k]) for k in ("cohorts", "development_seeds", "test_seeds")]
    if any(p & historical for p in panels) or any(
        a & b for a, b in itertools.combinations(panels, 2)
    ):
        raise ValueError("seed panel overlap")
    return dict(
        status="PASS",
        registry_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope=registry["scope"],
        historical_values=len(historical),
    )


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output, profile):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new/empty; resume is unsupported")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    dirty = git("status", "--porcelain")
    if profile == "full" and dirty:
        raise ValueError("full run requires clean committed Git")
    cfg = settings(profile)
    seeds = seed_audit(cfg)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        experiment=VERSION,
        status="RUNNING",
        commit=git("rev-parse", "HEAD"),
        dirty=dirty,
        started=datetime.now(timezone.utc).isoformat(),
        runtime=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            numpy=np.__version__,
        ),
        config=cfg,
    )
    expected = dict(
        development=len(cfg["cohorts"]) * len(cfg["development_seeds"]) * 3 * 4,
        test=len(cfg["cohorts"]) * len(cfg["test_seeds"]) * 3 * 5 * 2,
        exact_jobs=len(cfg["cohorts"]) * 3,
    )
    manifest["expected_counts"] = expected
    manifest["seed_audit"] = seeds
    write_json(output / "manifest.json", manifest)
    journals = {}
    try:
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        sources = [
            *Path(__file__).parent.glob("maintenance_contention*.py"),
            ROOT / "src/ht_pdm_fjsp/maintenance_dispatch.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_waiting.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_coordination_references.py",
            ROOT / "docs/maintenance_contention_headroom_v1_plan.md",
            ROOT / "configs/maintenance_contention_headroom_v1_seed_registry.json",
            ROOT / "uv.lock",
        ]
        for path in sources:
            shutil.copy2(path, snapshot / path.name)
        write_json(output / "benchmark_config.json", cfg)
        write_json(
            output / "resolved_config.json",
            {**cfg, "device": "cpu", "objective": "base_plus_overdue_price"},
        )
        configs = {
            f"{s}_{cap}_H{h}": asdict(cohort_config(s, cap, h))
            for s, cap, h in itertools.product(
                cfg["cohorts"], CAPACITIES, cfg["horizons"]
            )
        }
        configs.update(
            {
                f"exact_{s}_{cap}": asdict(
                    cohort_config(s, cap, cfg["exact_horizon"], 3)
                )
                for s, cap in itertools.product(cfg["cohorts"], CAPACITIES)
            }
        )
        write_json(output / "evaluation_configs.json", configs)
        write_json(
            output / "budget.json",
            dict(
                training_transitions=0,
                optimizer_steps=0,
                teacher_labels=0,
                training_episodes=0,
                scenario_forecasts_per_decision=cfg["scenarios"],
                exact_state_cap=cfg["exact_state_cap"],
                **expected,
            ),
        )
        for key, filename in (
            ("development", "development_episodes.csv"),
            ("episodes", "episodes.partial.csv"),
            ("decisions", "decisions.csv"),
            ("machines", "machine_metrics.csv"),
        ):
            journals[key] = Journal(output / filename)
        dev = []
        panel = itertools.product(
            cfg["cohorts"], CAPACITIES, cfg["development_seeds"], REFERENCES
        )
        for cohort, capacity, seed, name in tqdm(
            panel, total=expected["development"], desc="development references"
        ):
            c = cohort_config(cohort, capacity, cfg["horizons"][0])
            validate_model(c)
            row = episode(
                c,
                seed,
                name,
                None,
                cfg["scenarios"],
                dict(cohort=cohort, capacity=capacity, horizon=c.horizon),
            )
            dev.append(row)
            journals["development"].add(row)
        means = {
            cap: {
                name: float(
                    np.mean(
                        [
                            r["objective"]
                            for r in dev
                            if r["capacity"] == cap and r["controller"] == name
                        ]
                    )
                )
                for name in REFERENCES
            }
            for cap in CAPACITIES
        }
        selected = {
            cap: min(REFERENCES, key=lambda n: (means[cap][n], n)) for cap in CAPACITIES
        }
        write_json(
            output / "reference_selection.json",
            dict(
                selected=selected,
                development_means=means,
                frozen_before_test=True,
                objective="priced_cost",
                selection_horizon=cfg["horizons"][0],
            ),
        )
        # Executable controller specification: reload is used for every test episode.
        checkpoint = dict(
            version=VERSION,
            references=selected,
            scenarios=cfg["scenarios"],
            seed_roles=cfg["seed_roles"],
            config=cfg,
        )
        checkpoint_path = output / "controller_checkpoints/controllers.json"
        write_json(checkpoint_path, checkpoint)
        loaded = json.loads(checkpoint_path.read_text())
        if loaded != checkpoint:
            raise AssertionError("controller specification reload")
        c = cohort_config(cfg["cohorts"][0], "middle", cfg["horizons"][0])
        state = compact(WaitingEnv(c).state)
        args = (
            c,
            state,
            c.horizon,
            selected["middle"],
            role_seed(cfg["test_seeds"][0], "checkpoint_replay"),
        )
        if plan(*args, cfg["scenarios"]) != plan(*args, loaded["scenarios"]):
            raise AssertionError("controller action reload")
        rows = []
        panel = itertools.product(
            cfg["cohorts"],
            CAPACITIES,
            cfg["horizons"],
            cfg["test_seeds"],
            cfg["controllers"],
        )
        for cohort, capacity, horizon, seed, name in tqdm(
            panel, total=expected["test"], desc="paired test episodes"
        ):
            c = cohort_config(cohort, capacity, horizon)
            row = episode(
                c,
                seed,
                name,
                loaded["references"][capacity],
                loaded["scenarios"],
                dict(cohort=cohort, capacity=capacity, horizon=horizon),
                journals,
            )
            journals["episodes"].add(row)
            rows.append(row)
        journals["episodes"].close()
        shutil.copy2(output / "episodes.partial.csv", output / "episodes.csv")
        paired, hypotheses = summarize(rows, selected, cfg)
        write_csv(output / "paired_cohort_metrics.csv", paired)
        jobs, exact = [], []
        tables = output / "oracle_tables"
        tables.mkdir()
        for cohort, capacity in tqdm(
            list(itertools.product(cfg["cohorts"], CAPACITIES)), desc="exact jobs"
        ):
            c = cohort_config(cohort, capacity, cfg["exact_horizon"], 3)
            job, metrics = run_exact(
                c, tables / f"{cohort}_{capacity}.jsonl.gz", cfg["exact_state_cap"]
            )
            jobs.append(dict(cohort=cohort, capacity=capacity, **job))
            exact.extend(dict(cohort=cohort, capacity=capacity, **r) for r in metrics)
            write_csv(output / "exact_jobs.csv", jobs)
            write_csv(output / "exact_metrics.csv", exact)
        actual = dict(development=len(dev), test=len(rows), exact_jobs=len(jobs))
        if actual != expected:
            raise AssertionError("panel count")
        capped = sum(j["status"] == "CAPPED" for j in jobs)
        write_json(
            output / "summary.json",
            dict(
                hypotheses=hypotheses,
                exact_capped_jobs=capped,
                exact_completed_jobs=len(jobs) - capped,
                interpretation="Planner gains identify achievable optimization headroom, not RL/MARL necessity. Exact ceiling applies only to completed N3 jobs; N4 rollout is not an optimum.",
                audits=dict(
                    complete_episode_panels=True,
                    feasible_actions=True,
                    physical_cost_reconciliation=True,
                    machine_reconciliation=True,
                    controller_reload=True,
                    zero_learning_budget=True,
                ),
            ),
        )
        manifest.update(
            status="COMPLETED",
            actual_counts=actual,
            exact_capped_jobs=capped,
            exact_coverage="PARTIAL" if capped else "COMPLETE",
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
        json.dumps(
            dict(
                output=str(output),
                status=manifest["status"],
                counts=actual,
                capped=capped,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = (
        args.output_dir
        or ROOT
        / "artifacts"
        / f"{VERSION}_{args.profile}_cpu_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
    )
    run(output.resolve(), args.profile)


if __name__ == "__main__":
    main()
