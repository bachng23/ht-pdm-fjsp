"""Planner imitation versus direct and initialized PPO; independent held-out test."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp import maintenance_contention_learning as base
from ht_pdm_fjsp.maintenance_contention_headroom import (
    Journal,
    write_json,
    write_csv,
    file_hash,
)
from ht_pdm_fjsp.maintenance_contention_model import (
    CAPACITIES,
    REFERENCES,
    cohort_config,
    compact,
    physical_action,
)
from ht_pdm_fjsp.maintenance_contention_learning_policy import (
    FEATURE_CONTRACT,
)
from ht_pdm_fjsp.maintenance_contention_learning_training import fit
from ht_pdm_fjsp.maintenance_contention_imitation_training import collect, fit_imitation
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, ENV_VERSION

VERSION = "maintenance_contention_imitation_v1"
DIRECT = "direct_ppo"
IMITATION = "planner_imitation"
HYBRID = "imitation_ppo"
ARMS = (DIRECT, IMITATION, HYBRID)
ROOT = Path(__file__).resolve().parents[2]
model_for = base.model_for
mean = base.mean
select_checkpoint = base.select_checkpoint


def settings(profile):
    cfg = base.settings(profile)
    full = profile == "full"
    cfg.update(
        train_seeds=list(range(144000, 144010)) if full else [149000],
        development_cohorts=list(range(145000, 145010)) if full else [149020],
        development_seeds=list(range(146000, 146020)) if full else [149030],
        test_cohorts=list(range(147000, 147010)) if full else [149010],
        test_seeds=list(range(148000, 148050)) if full else list(range(149040, 149043)),
        teacher_episodes=1024 if full else 16,
        teacher_validation_episodes=128 if full else 4,
        bc_epochs=80 if full else 4,
        bc_minibatch=192 if full else 32,
        bc_checkpoints=[20, 40, 60, 80] if full else [1, 2, 3, 4],
        noise_probes=32 if full else 2,
        noise_scenarios=128 if full else 16,
        arms=list(ARMS),
        supervised_tie_tolerance=1e-6,
        primary_relative_reduction=0.02,
    )
    return cfg


def seed_audit(cfg):
    path = ROOT / f"configs/{VERSION}_seed_registry.json"
    registry = json.loads(path.read_text())
    historical = (
        set(registry["declared_seed_values"])
        | set(range(201, 301))
        | set(range(601, 701))
    )
    panels = [
        set(cfg[k])
        for k in (
            "train_seeds",
            "development_cohorts",
            "development_seeds",
            "test_cohorts",
            "test_seeds",
        )
    ]
    if any(a & b for a, b in itertools.combinations(panels, 2)) or any(
        p & historical for p in panels
    ):
        raise ValueError("historical/reserved/new seed overlap")
    return dict(
        status="PASS",
        registry_sha256=file_hash(path),
        scope=registry["scope"],
        panels_disjoint=True,
        historical_disjoint=True,
    ), historical | set().union(*panels)


def counts(cfg):
    n = len(cfg["train_seeds"])
    h = cfg["horizons"][0]
    dev = len(cfg["development_cohorts"]) * len(cfg["development_seeds"])
    test = len(cfg["test_cohorts"]) * len(cfg["test_seeds"])
    bc = n * cfg["bc_epochs"] * cfg["teacher_episodes"] * h // cfg["bc_minibatch"]
    ppo = (
        2
        * n
        * (cfg["env_steps"] // cfg["rollout"])
        * cfg["epochs"]
        * cfg["rollout"]
        // cfg["minibatch"]
    )
    episodes = (5 + 3 * n) * 3 * 2 * test
    return dict(
        models=3 * n,
        training_steps=2 * n * cfg["env_steps"],
        training_episodes=2 * n * cfg["env_steps"] // h,
        optimizer_steps=bc + ppo,
        teacher_physical_steps=n
        * (cfg["teacher_episodes"] + cfg["teacher_validation_episodes"])
        * h,
        teacher_trajectories=n
        * (cfg["teacher_episodes"] + cfg["teacher_validation_episodes"]),
        teacher_training_rows=n * cfg["teacher_episodes"] * h,
        teacher_validation_rows=n * cfg["teacher_validation_episodes"] * h,
        teacher_noise_probes=n * cfg["noise_probes"],
        development_episodes=12 * dev
        + n
        * (2 * (len(cfg["checkpoint_updates"]) + 1) + len(cfg["bc_checkpoints"]) + 1)
        * 2
        * dev,
        test_episodes=episodes,
        test_decision_intervals=(5 + 3 * n) * 3 * test * sum(cfg["horizons"]),
        test_machine_rows=4 * episodes,
    )


class RenameJournal:
    def __init__(self, journal, name):
        self.journal, self.name = journal, name

    def add(self, row):
        self.journal.add({**row, "controller": self.name})


def episode(
    c, seed, name, continuation, scenarios, identity, model=None, journals=None
):
    learned = name in ARMS
    proxy = (
        {k: RenameJournal(j, name) for k, j in journals.items()}
        if learned and journals
        else journals
    )
    row = base.episode(
        c,
        seed,
        base.LEARNER if learned else name,
        continuation,
        scenarios,
        identity,
        model,
        proxy,
    )
    return {**row, "controller": name}


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["protocol"] != VERSION
        or payload["feature_contract"] != FEATURE_CONTRACT
        or payload["environment_version"] != ENV_VERSION
    ):
        raise ValueError("checkpoint contract mismatch")
    model = model_for(payload["settings"], payload["train_seed"])
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


def save_checkpoint(path, model, cfg, seed, update, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            protocol=VERSION,
            environment_version=ENV_VERSION,
            feature_contract=FEATURE_CONTRACT,
            algorithm=cfg["algorithm"],
            counter_kind="bc_optimizer_steps"
            if cfg["algorithm"] == IMITATION
            else "ppo_physical_steps",
            parent_checkpoint_sha256=cfg.get("parent_checkpoint_sha256"),
            train_seed=seed,
            rollout_update=update,
            physical_steps=steps,
            settings=cfg,
            state_dict=model.state_dict(),
        ),
        path,
    )
    restored, payload = load_checkpoint(path)
    if any(
        not torch.equal(v, restored.state_dict()[k])
        for k, v in model.state_dict().items()
    ):
        raise RuntimeError("checkpoint parameter reload mismatch")
    c = cohort_config(cfg["development_cohorts"][0], "middle", cfg["horizons"][0])
    env = WaitingEnv(c)
    env.reset(cfg["development_seeds"][0])
    for t in range(c.horizon):
        chosen, diag = model.select(c, compact(env.state), c.horizon - t)
        if (chosen, diag) != restored.select(c, compact(env.state), c.horizon - t):
            raise RuntimeError("checkpoint probability/value/action reload mismatch")
        env.step(physical_action(env.state, chosen))
    return restored, payload


def summarize(rows, selected_refs, cfg):
    paired = []
    cohorts = []
    arm_results = {}
    controls = [r for r in rows if r["train_seed"] == -1]
    if any(r["controller"] not in (*cfg["controls"], *ARMS) for r in rows):
        raise RuntimeError("unknown controller")
    for arm in ARMS:
        own = [
            {**r, "controller": base.LEARNER} for r in rows if r["controller"] == arm
        ]
        p, c, result = base.summarize(controls + own, selected_refs, cfg)
        paired.extend(dict(arm=arm, **r) for r in p)
        cohorts.extend(dict(arm=arm, **r) for r in c)
        arm_results[arm] = result
    cap = "middle"
    h = cfg["horizons"][0]
    by_arm = {
        arm: {
            seed: mean(
                [
                    r
                    for r in rows
                    if r["controller"] == arm
                    and r["train_seed"] == seed
                    and r["capacity"] == cap
                    and r["horizon"] == h
                ],
                "objective",
            )
            for seed in cfg["train_seeds"]
        }
        for arm in ARMS
    }
    deltas = [by_arm[HYBRID][s] - by_arm[DIRECT][s] for s in cfg["train_seeds"]]
    avg = float(np.mean(deltas))
    half = (
        2.2621571628540993 * float(np.std(deltas, ddof=1)) / math.sqrt(10)
        if len(deltas) == 10
        else None
    )
    reduction = 1 - np.mean(list(by_arm[HYBRID].values())) / np.mean(
        list(by_arm[DIRECT].values())
    )
    hybrid = arm_results[HYBRID][f"middle_H{h}"]
    gates = dict(
        relative_priced_reduction_2pct=bool(
            reduction >= cfg["primary_relative_reduction"]
        ),
        wins_8_of_10=sum(d < -1e-9 for d in deltas) >= 8,
        ci95_upper_below_zero=half is not None and avg + half < 0,
        nominal_guard=hybrid["gates"]["nominal_ratio_110pct"],
        waiting_guard=hybrid["gates"]["waiting_violation_guard"],
    )
    primary = dict(
        contrast=f"{HYBRID} vs {DIRECT}",
        capacity=cap,
        horizon=h,
        paired_replica_costs=by_arm,
        paired_differences=deltas,
        priced_reduction=float(reduction),
        mean_difference=avg,
        ci95=[avg - half, avg + half] if half is not None else None,
        wins=sum(d < -1e-9 for d in deltas),
        gates=gates,
        status="ENGINEERING_ONLY"
        if cfg["profile"] == "smoke"
        else ("PASS" if all(gates.values()) else "FAIL"),
    )
    return paired, cohorts, dict(primary=primary, arms=arm_results)


def run(output, profile):
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be new/empty; no resume")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    dirty = git("status", "--porcelain")
    if profile == "full" and dirty:
        raise ValueError("full requires clean committed Git")
    cfg = settings(profile)
    torch.set_num_threads(cfg["torch_threads"])
    audits, forbidden = seed_audit(cfg)
    expected = counts(cfg)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(
        experiment=VERSION,
        status="RUNNING",
        commit=git("rev-parse", "HEAD"),
        dirty=dirty,
        started=datetime.now(timezone.utc).isoformat(),
        config=cfg,
        expected_counts=expected,
        runtime=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            torch=str(torch.__version__),
            numpy=np.__version__,
        ),
    )
    write_json(output / "manifest.json", manifest)
    journals = {}
    try:
        source = output / "source_snapshot"
        source.mkdir()
        sources = [
            *Path(__file__).parent.glob("maintenance_contention*.py"),
            ROOT / "src/ht_pdm_fjsp/maintenance_dispatch.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_waiting.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_coordination_references.py",
            ROOT / f"docs/{VERSION}_plan.md",
            ROOT / f"configs/{VERSION}_seed_registry.json",
            ROOT / "uv.lock",
        ]
        for path in sources:
            shutil.copy2(path, source / path.name)
        write_json(output / "benchmark_config.json", cfg)
        write_json(
            output / "resolved_config.json",
            {**cfg, "device": "cpu", "feature_contract": FEATURE_CONTRACT},
        )
        write_json(output / "seed_audit.json", audits)
        write_json(
            output / "budget.json",
            {
                **expected,
                "resume_supported": False,
                "teacher_data_shared_with_hybrid": True,
            },
        )
        evaluation_configs = {
            f"{phase}_{cs}_{cap}_H{h}": asdict(cohort_config(cs, cap, h))
            for phase, seeds in [
                ("development", cfg["development_cohorts"]),
                ("test", cfg["test_cohorts"]),
            ]
            for cs, cap, h in itertools.product(
                seeds,
                CAPACITIES,
                cfg["horizons"] if phase == "test" else cfg["horizons"][:1],
            )
        }
        write_json(output / "evaluation_configs.json", evaluation_configs)
        for key in (
            "training_episodes",
            "training_progress",
            "development",
            "episodes",
            "decisions",
            "machines",
            "teacher_rows",
            "teacher_noise",
            "bc_progress",
            "bc_diagnostics",
        ):
            filename = {
                "episodes": "episodes.partial.csv",
                "development": "development_episodes.csv",
                "machines": "machine_metrics.csv",
            }.get(key, f"{key}.csv")
            journals[key] = Journal(output / filename)
        dev = []
        jobs = list(
            itertools.product(
                cfg["development_cohorts"],
                CAPACITIES,
                cfg["development_seeds"],
                REFERENCES,
            )
        )
        for cs, cap, seed, name in tqdm(jobs, desc="development references"):
            c = cohort_config(cs, cap, cfg["horizons"][0])
            row = episode(
                c,
                seed,
                name,
                None,
                cfg["scenarios"],
                dict(
                    train_seed=-1,
                    checkpoint_step=-1,
                    cohort=cs,
                    capacity=cap,
                    horizon=c.horizon,
                ),
            )
            dev.append(row)
            journals["development"].add(row)
        means = {
            cap: {
                name: mean(
                    [
                        r
                        for r in dev
                        if r["capacity"] == cap and r["controller"] == name
                    ],
                    "objective",
                )
                for name in REFERENCES
            }
            for cap in CAPACITIES
        }
        selected_refs = {
            cap: min(REFERENCES, key=lambda name: (means[cap][name], name))
            for cap in CAPACITIES
        }
        write_json(
            output / "reference_selection.json",
            dict(
                selected=selected_refs, development_means=means, frozen_before_test=True
            ),
        )
        selected_dev = [
            r for r in dev if r["controller"] == selected_refs[r["capacity"]]
        ]
        selections = {}
        coverage = []
        teacher_coverage = []
        development_count = len(dev)
        for seed in tqdm(cfg["train_seeds"], desc="training replicas", unit="seed"):
            data, teacher_record = collect(
                cfg, seed, forbidden, selected_refs, journals
            )
            teacher_coverage.append(teacher_record)
            write_json(output / "teacher_coverage.json", teacher_coverage)
            data_dir = output / "teacher" / f"train_seed_{seed}"
            data_dir.mkdir(parents=True)
            torch.save(data, data_dir / "teacher_data.pt")
            restored_data = torch.load(
                data_dir / "teacher_data.pt", map_location="cpu", weights_only=True
            )
            for split in ("train", "validation"):
                for key in ("targets", "scores"):
                    if not torch.equal(data[split][key], restored_data[split][key]):
                        raise RuntimeError("teacher dataset reload mismatch")
                for key, value in data[split]["features"].items():
                    if not torch.equal(value, restored_data[split]["features"][key]):
                        raise RuntimeError("teacher feature reload mismatch")
            for arm in (IMITATION, DIRECT, HYBRID):
                arm_cfg = {**cfg, "algorithm": arm}
                if arm == HYBRID:
                    parent = selections[f"{IMITATION}:{seed}"]
                    model, _ = load_checkpoint(output / parent["checkpoint"])
                    arm_cfg["parent_checkpoint_sha256"] = parent["checkpoint_sha256"]
                else:
                    model = model_for(cfg, seed)
                directory = output / arm / f"train_seed_{seed}"
                records = []

                def checkpoint(current, update, steps):
                    nonlocal development_count
                    path = directory / "checkpoints" / f"counter_{steps:08d}.pt"
                    restored, _ = save_checkpoint(
                        path, current, arm_cfg, seed, update, steps
                    )
                    panel = []
                    jobs = list(
                        itertools.product(
                            cfg["development_cohorts"],
                            ("middle", "low"),
                            cfg["development_seeds"],
                        )
                    )
                    for cs, cap, shock in tqdm(
                        jobs, desc=f"dev {arm}/{seed}/{steps}", leave=False
                    ):
                        c = cohort_config(cs, cap, cfg["horizons"][0])
                        row = episode(
                            c,
                            shock,
                            arm,
                            None,
                            cfg["scenarios"],
                            dict(
                                train_seed=seed,
                                checkpoint_step=steps,
                                cohort=cs,
                                capacity=cap,
                                horizon=c.horizon,
                            ),
                            restored,
                        )
                        panel.append(row)
                        journals["development"].add(row)
                        development_count += 1
                    records.append(
                        dict(
                            update=update,
                            steps=steps,
                            path=str(path.relative_to(output)),
                            episodes=panel,
                        )
                    )

                checkpoint(model, 0, 0)
                training_started = time.perf_counter()
                if arm == IMITATION:
                    record = fit_imitation(model, cfg, seed, data, journals, checkpoint)
                else:
                    record = fit(
                        model,
                        cfg,
                        seed,
                        forbidden,
                        ArmJournals(journals, arm),
                        checkpoint,
                    )
                record["training_and_development_seconds"] = (
                    time.perf_counter() - training_started
                )
                record["arm"] = arm
                coverage.append(record)
                write_json(output / "training_coverage.json", coverage)
                chosen, status = select_checkpoint(records, selected_dev)
                shutil.copy2(output / chosen["path"], directory / "model.pt")
                _, payload = load_checkpoint(directory / "model.pt")
                if (
                    payload["physical_steps"] != chosen["steps"]
                    or payload["algorithm"] != arm
                ):
                    raise RuntimeError("selected checkpoint provenance")
                selections[f"{arm}:{seed}"] = dict(
                    arm=arm,
                    train_seed=seed,
                    status=status,
                    selected_steps=chosen["steps"],
                    selected_update=chosen["update"],
                    counter_kind=payload["counter_kind"],
                    parent_checkpoint_sha256=payload["parent_checkpoint_sha256"],
                    checkpoint=str((directory / "model.pt").relative_to(output)),
                    checkpoint_sha256=file_hash(directory / "model.pt"),
                    development_cost=chosen["development_primary_cost"],
                    development_guards=chosen["development_guards"],
                    candidates=[
                        {k: v for k, v in r.items() if k != "episodes"} for r in records
                    ],
                )
                write_json(
                    output / "checkpoint_selection.json",
                    dict(frozen_before_test=True, selections=selections),
                )
            del data, restored_data
        # All30 selected checkpoints are frozen before the first test rollout.
        rows = []
        jobs = list(
            itertools.product(
                cfg["test_cohorts"],
                CAPACITIES,
                cfg["horizons"],
                cfg["test_seeds"],
                cfg["controls"],
            )
        )
        for cs, cap, h, seed, name in tqdm(jobs, desc="test references/planner"):
            c = cohort_config(cs, cap, h)
            row = episode(
                c,
                seed,
                name,
                selected_refs[cap],
                cfg["scenarios"],
                dict(
                    train_seed=-1,
                    checkpoint_step=-1,
                    cohort=cs,
                    capacity=cap,
                    horizon=h,
                ),
                journals=journals,
            )
            rows.append(row)
            journals["episodes"].add(row)
        for arm, seed in tqdm(
            list(itertools.product(ARMS, cfg["train_seeds"])),
            desc="test learned replicas",
        ):
            chosen = selections[f"{arm}:{seed}"]
            model, _ = load_checkpoint(output / chosen["checkpoint"])
            jobs = list(
                itertools.product(
                    cfg["test_cohorts"], CAPACITIES, cfg["horizons"], cfg["test_seeds"]
                )
            )
            for cs, cap, h, shock in tqdm(jobs, desc=f"test {arm}/{seed}", leave=False):
                row = episode(
                    cohort_config(cs, cap, h),
                    shock,
                    arm,
                    None,
                    cfg["scenarios"],
                    dict(
                        train_seed=seed,
                        checkpoint_step=chosen["selected_steps"],
                        cohort=cs,
                        capacity=cap,
                        horizon=h,
                    ),
                    model,
                    journals,
                )
                rows.append(row)
                journals["episodes"].add(row)
        journals["episodes"].close()
        shutil.copy2(output / "episodes.partial.csv", output / "episodes.csv")
        paired, cohorts, results = summarize(rows, selected_refs, cfg)
        write_csv(output / "paired_seed_metrics.csv", paired)
        write_csv(output / "cohort_metrics.csv", cohorts)
        actual = dict(
            models=len(coverage),
            training_steps=sum(r["env_steps"] for r in coverage),
            training_episodes=sum(r["episodes"] for r in coverage),
            optimizer_steps=sum(r["optimizer_steps"] for r in coverage),
            teacher_physical_steps=sum(r["physical_steps"] for r in teacher_coverage),
            teacher_trajectories=sum(r["trajectories"] for r in teacher_coverage),
            teacher_training_rows=sum(r["train_rows"] for r in teacher_coverage),
            teacher_validation_rows=sum(r["validation_rows"] for r in teacher_coverage),
            teacher_noise_probes=sum(r["noise_probes"] for r in teacher_coverage),
            development_episodes=development_count,
            test_episodes=len(rows),
            test_decision_intervals=sum(r["horizon"] for r in rows),
            test_machine_rows=len(rows) * 4,
        )
        if actual != expected:
            raise RuntimeError(
                f"experiment population/budget mismatch: {actual} != {expected}"
            )
        write_json(
            output / "summary.json",
            dict(
                **results,
                development_infeasible=[
                    k for k, r in selections.items() if r["status"] != "FEASIBLE"
                ],
                interpretation="Supervision and initialization diagnostic with extra teacher information/compute; no MARL claim.",
                audits=dict(
                    complete_grids=True,
                    exact_training_budget=True,
                    checkpoint_reload=True,
                    reserved_seed_training_transitions_zero=True,
                    validation_training_rows_zero=True,
                    teacher_future_event_access_zero=True,
                    feasible_actions=True,
                    cost_machine_reconciliation=True,
                ),
            ),
        )
        manifest.update(status="COMPLETED", actual_counts=actual)
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


class ArmJournal:
    def __init__(self, journal, arm):
        self.journal, self.arm = journal, arm

    def add(self, row):
        self.journal.add(dict(arm=self.arm, **row))


def ArmJournals(journals, arm):
    return {k: ArmJournal(v, arm) for k, v in journals.items()}


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
