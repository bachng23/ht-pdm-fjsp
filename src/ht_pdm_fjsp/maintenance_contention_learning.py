"""Learn shared-capacity scheduling with centralized subset PPO; CPU only."""

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

from ht_pdm_fjsp.maintenance_contention_headroom import (
    Journal,
    write_json,
    write_csv,
    file_hash,
)
from ht_pdm_fjsp.maintenance_contention_model import (
    CAPACITIES,
    REFERENCES,
    actions,
    cohort_config,
    compact,
    physical_action,
    plan,
    reference,
)
from ht_pdm_fjsp.maintenance_contention_learning_policy import (
    SubsetActorCritic,
    FEATURE_CONTRACT,
)
from ht_pdm_fjsp.maintenance_contention_learning_training import fit
from ht_pdm_fjsp.maintenance_dispatch import role_seed
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, ENV_VERSION

VERSION = "maintenance_contention_learning_v1"
LEARNER = "central_subset_ppo"
ROOT = Path(__file__).resolve().parents[2]


def settings(profile):
    full = profile == "full"
    return dict(
        profile=profile,
        train_seeds=list(range(138000, 138010)) if full else [143000],
        development_cohorts=list(range(139000, 139010)) if full else [143020],
        development_seeds=list(range(140000, 140020)) if full else [143030],
        test_cohorts=list(range(141000, 141010)) if full else [143010],
        test_seeds=list(range(142000, 142050)) if full else list(range(143040, 143043)),
        horizons=[12, 24] if full else [4, 6],
        scenarios=32 if full else 4,
        env_steps=491520 if full else 256,
        rollout=1536 if full else 128,
        vector_envs=32 if full else 8,
        minibatch=192 if full else 32,
        epochs=4 if full else 2,
        hidden=64 if full else 16,
        checkpoint_updates=[80, 160, 240, 320] if full else [1, 2],
        learning_rate=3e-4,
        clip=0.2,
        entropy_weight=0.01,
        value_weight=0.5,
        gradient_clip=0.5,
        discount=1.0,
        reward_scale=20.0,
        torch_threads=1,
        capacities=CAPACITIES,
        controls=[*REFERENCES, "joint_rollout"],
        training_capacity_mixture={"middle": 0.8, "low": 0.2},
        training_seed_roles=[
            "learning_configuration:<episode>",
            "learning_environment:<episode>",
            "learning_actions",
            "learning_minibatches",
        ],
    )


def seed_audit(cfg):
    registry_path = ROOT / f"configs/{VERSION}_seed_registry.json"
    registry = json.loads(registry_path.read_text())
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
    forbidden = historical | set().union(*panels)
    return dict(
        status="PASS",
        registry_sha256=file_hash(registry_path),
        scope=registry["scope"],
        panels_disjoint=True,
        historical_disjoint=True,
    ), forbidden


def counts(cfg):
    n = len(cfg["train_seeds"])
    dev = len(cfg["development_cohorts"]) * len(cfg["development_seeds"])
    test = len(cfg["test_cohorts"]) * len(cfg["test_seeds"])
    controls = len(cfg["controls"]) * 3 * 2 * test
    learned = n * 3 * 2 * test
    return dict(
        models=n,
        training_steps=n * cfg["env_steps"],
        training_episodes=n * cfg["env_steps"] // cfg["horizons"][0],
        optimizer_steps=n
        * (cfg["env_steps"] // cfg["rollout"])
        * cfg["epochs"]
        * cfg["rollout"]
        // cfg["minibatch"],
        development_episodes=4 * 3 * dev
        + n * (len(cfg["checkpoint_updates"]) + 1) * 2 * dev,
        test_episodes=controls + learned,
        test_decision_intervals=(5 + n) * 3 * test * sum(cfg["horizons"]),
        test_machine_rows=(controls + learned) * 4,
    )


def model_for(cfg, seed):
    torch.manual_seed(seed)
    return SubsetActorCritic(cfg["hidden"]).eval()


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
            algorithm=LEARNER,
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


def episode(
    c, seed, name, continuation, scenarios, identity, model=None, journals=None
):
    env = WaitingEnv(c)
    env.reset(seed)
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
    samples = [[] for _ in machine]
    excess = choices = backlog = utilization = subset_total = 0
    planning_seconds = 0.0
    spreads = []
    errors = []
    for t in range(c.horizon):
        state = env.state
        reduced = compact(state)
        feasible = actions(c, reduced)
        free = c.technicians - sum(r > 0 for r in state.remaining)
        urgent = sum(
            not row[3] and (row[1] or row[0] >= c.failure_age - 1) for row in reduced
        )
        urgent_excess = max(0, urgent - free)
        begin = time.perf_counter()
        diag = {}
        if name == LEARNER:
            if model is None:
                raise ValueError("learned policy needs checkpoint")
            selected, diag = model.select(c, reduced, c.horizon - t)
        elif name == "joint_rollout":
            selected, diag = plan(
                c,
                reduced,
                c.horizon - t,
                continuation,
                role_seed(seed, f"contention_planner:{t}"),
                scenarios,
            )
            spreads.append(diag["score_spread"])
            errors.append(diag["selected_mc_se"])
        else:
            selected = reference(c, reduced, c.horizon - t, name)
        elapsed = time.perf_counter() - begin
        planning_seconds += elapsed
        pairs = physical_action(state, selected)
        if selected not in feasible or len(pairs) != len(selected):
            raise RuntimeError("infeasible controller output")
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
                    controller=name,
                    time=t,
                    state=json.dumps(asdict(state)),
                    action=json.dumps(pairs),
                    following=json.dumps(asdict(env.state)),
                    physical_metrics=json.dumps(info),
                    urgent_excess=urgent_excess,
                    feasible_subsets=len(feasible),
                    inference_seconds=elapsed,
                    controller_diagnostics=json.dumps(diag),
                )
            )
    for metric in (
        "unavailable_ticks",
        "failed_waiting_ticks",
        "overdue_waiting_ticks",
    ):
        if sum(row[metric] for row in machine) != env.metrics[metric]:
            raise RuntimeError("machine-cost reconciliation")
    for m, row in enumerate(machine):
        row.update(
            terminal_failed=int(env.state.failed[m]),
            terminal_pending=int(env.state.failed[m] and m not in env.state.assigned),
            p95_wait=float(np.percentile(samples[m], 95)),
            violation=int(row["max_wait"] > c.waiting_limit),
        )
        if journals:
            journals["machines"].add(
                dict(**identity, seed=seed, controller=name, **row)
            )
    return dict(
        **identity,
        seed=seed,
        controller=name,
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
        inference_seconds=planning_seconds,
    )


def mean(rows, metric):
    if not rows:
        raise ValueError("empty evaluation panel")
    return float(np.mean([r[metric] for r in rows]))


def select_checkpoint(records, reference_rows):
    ref_middle = [r for r in reference_rows if r["capacity"] == "middle"]
    ref_low = [r for r in reference_rows if r["capacity"] == "low"]
    viable = []
    for record in records:
        if record["update"] == 0:
            continue
        panel = record["episodes"]
        middle = [r for r in panel if r["capacity"] == "middle"]
        low = [r for r in panel if r["capacity"] == "low"]
        nominal = mean(low, "base_cost") <= 1.1 * mean(ref_low, "base_cost")
        waiting = (
            mean(middle, "waiting_violation")
            <= mean(ref_middle, "waiting_violation") + 0.02
        )
        record["development_guards"] = dict(nominal=nominal, waiting=waiting)
        record["development_primary_cost"] = mean(middle, "objective")
        if nominal and waiting:
            viable.append(record)
    if viable:
        chosen = min(viable, key=lambda r: (r["development_primary_cost"], r["steps"]))
        return chosen, "FEASIBLE"
    return max(
        (r for r in records if r["update"] > 0), key=lambda r: r["steps"]
    ), "DEVELOPMENT_INFEASIBLE"


def summarize(rows, selected_refs, cfg):
    expected_controls = set(
        itertools.product(
            cfg["test_cohorts"],
            CAPACITIES,
            cfg["horizons"],
            cfg["test_seeds"],
            cfg["controls"],
        )
    )
    controls = [r for r in rows if r["train_seed"] == -1]
    control_keys = {
        (r["cohort"], r["capacity"], r["horizon"], r["seed"], r["controller"])
        for r in controls
    }
    if len(control_keys) != len(controls) or control_keys != expected_controls:
        raise RuntimeError("incomplete control panel")
    learners = [r for r in rows if r["controller"] == LEARNER]
    expected_learned = set(
        itertools.product(
            cfg["train_seeds"],
            cfg["test_cohorts"],
            CAPACITIES,
            cfg["horizons"],
            cfg["test_seeds"],
        )
    )
    learned_keys = {
        (r["train_seed"], r["cohort"], r["capacity"], r["horizon"], r["seed"])
        for r in learners
    }
    if len(learned_keys) != len(learners) or learned_keys != expected_learned:
        raise RuntimeError("incomplete learned panel")
    paired = []
    cohort = []
    contrasts = {}
    for cap, horizon in itertools.product(CAPACITIES, cfg["horizons"]):
        ref = [
            r
            for r in controls
            if r["capacity"] == cap
            and r["horizon"] == horizon
            and r["controller"] == selected_refs[cap]
        ]
        planner = [
            r
            for r in controls
            if r["capacity"] == cap
            and r["horizon"] == horizon
            and r["controller"] == "joint_rollout"
        ]
        deltas = []
        learner_means = []
        learner_base = []
        learner_violation = []
        for seed in cfg["train_seeds"]:
            panel = [
                r
                for r in learners
                if r["train_seed"] == seed
                and r["capacity"] == cap
                and r["horizon"] == horizon
            ]
            cost = mean(panel, "objective")
            base = mean(panel, "base_cost")
            delta = cost - mean(ref, "objective")
            deltas.append(delta)
            learner_means.append(cost)
            learner_base.append(base)
            learner_violation.append(mean(panel, "waiting_violation"))
            gap = mean(ref, "objective") - mean(planner, "objective")
            basegap = mean(ref, "base_cost") - mean(planner, "base_cost")
            paired.append(
                dict(
                    train_seed=seed,
                    capacity=cap,
                    horizon=horizon,
                    reference=selected_refs[cap],
                    reference_cost=mean(ref, "objective"),
                    learner_cost=cost,
                    planner_cost=mean(planner, "objective"),
                    difference=delta,
                    reference_base_cost=mean(ref, "base_cost"),
                    learner_base_cost=base,
                    planner_base_cost=mean(planner, "base_cost"),
                    priced_gap_capture=(mean(ref, "objective") - cost) / gap
                    if gap > 0
                    else None,
                    base_gap_capture=(mean(ref, "base_cost") - base) / basegap
                    if basegap > 0
                    else None,
                )
            )
            for cs in cfg["test_cohorts"]:
                own = [r for r in panel if r["cohort"] == cs]
                rule = [r for r in ref if r["cohort"] == cs]
                cohort.append(
                    dict(
                        train_seed=seed,
                        capacity=cap,
                        horizon=horizon,
                        cohort=cs,
                        learner_cost=mean(own, "objective"),
                        reference_cost=mean(rule, "objective"),
                        difference=mean(own, "objective") - mean(rule, "objective"),
                    )
                )
        avg = float(np.mean(deltas))
        half = (
            2.2621571628540993 * float(np.std(deltas, ddof=1)) / math.sqrt(10)
            if len(deltas) == 10
            else None
        )
        reduction = 1 - float(np.mean(learner_means)) / mean(ref, "objective")
        base_reduction = 1 - float(np.mean(learner_base)) / mean(ref, "base_cost")
        nominal = [
            r
            for r in learners
            if r["capacity"] == "low" and r["horizon"] == cfg["horizons"][0]
        ]
        nominal_ref = [
            r
            for r in controls
            if r["capacity"] == "low"
            and r["horizon"] == cfg["horizons"][0]
            and r["controller"] == selected_refs["low"]
        ]
        nominal_ratio = mean(nominal, "base_cost") / mean(nominal_ref, "base_cost")
        gates = dict(
            priced_reduction_5pct=reduction >= 0.05,
            win_8_of_10=sum(d < -1e-9 for d in deltas) >= 8,
            ci95_upper_below_zero=half is not None and avg + half < 0,
            base_reduction_3pct=base_reduction >= 0.03,
            nominal_ratio_110pct=nominal_ratio <= 1.1,
            waiting_violation_guard=float(np.mean(learner_violation))
            <= mean(ref, "waiting_violation") + 0.02,
        )
        primary = cap == "middle" and horizon == cfg["horizons"][0]
        status = (
            "ENGINEERING_ONLY"
            if cfg["profile"] == "smoke"
            else ("PASS" if all(gates.values()) else "FAIL")
            if primary
            else "EXPLORATORY"
        )
        contrasts[f"{cap}_H{horizon}"] = dict(
            reference=selected_refs[cap],
            learner_cost=float(np.mean(learner_means)),
            reference_cost=mean(ref, "objective"),
            planner_cost=mean(planner, "objective"),
            priced_reduction=reduction,
            base_reduction=base_reduction,
            training_seed_wins=sum(d < -1e-9 for d in deltas),
            paired_difference=avg,
            ci95=[avg - half, avg + half] if half is not None else None,
            nominal_base_ratio=nominal_ratio,
            learner_waiting_violation=float(np.mean(learner_violation)),
            reference_waiting_violation=mean(ref, "waiting_violation"),
            gates=gates,
            status=status,
        )
    return paired, cohort, contrasts


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
        runtime=dict(
            python=platform.python_version(),
            platform=platform.platform(),
            torch=str(torch.__version__),
            numpy=np.__version__,
        ),
        config=cfg,
        expected_counts=expected,
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
            {**expected, "teacher_labels": 0, "resume_supported": False},
        )
        evaluation_configs = {
            f"{phase}_{cs}_{cap}_H{h}": asdict(cohort_config(cs, cap, h))
            for phase, cohorts in [
                ("development", cfg["development_cohorts"]),
                ("test", cfg["test_cohorts"]),
            ]
            for cs, cap, h in itertools.product(
                cohorts,
                CAPACITIES,
                cfg["horizons"] if phase == "test" else cfg["horizons"][:1],
            )
        }
        write_json(output / "evaluation_configs.json", evaluation_configs)
        for key, filename in [
            ("training_episodes", "training_episodes.csv"),
            ("training_progress", "training_progress.csv"),
            ("development", "development_episodes.csv"),
            ("episodes", "episodes.partial.csv"),
            ("decisions", "decisions.csv"),
            ("machines", "machine_metrics.csv"),
        ]:
            journals[key] = Journal(output / filename)
        dev = []
        panel = itertools.product(
            cfg["development_cohorts"], CAPACITIES, cfg["development_seeds"], REFERENCES
        )
        for cs, cap, seed, name in tqdm(
            panel,
            total=4
            * 3
            * len(cfg["development_cohorts"])
            * len(cfg["development_seeds"]),
            desc="development references",
        ):
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
        dev_means = {
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
            cap: min(REFERENCES, key=lambda name: (dev_means[cap][name], name))
            for cap in CAPACITIES
        }
        write_json(
            output / "reference_selection.json",
            dict(
                selected=selected_refs,
                development_means=dev_means,
                frozen_before_test=True,
            ),
        )
        selected_dev = [
            r for r in dev if r["controller"] == selected_refs[r["capacity"]]
        ]
        selections = {}
        coverage = []
        development_count = len(dev)
        for train_seed in tqdm(
            cfg["train_seeds"], desc="learning replicas", unit="seed"
        ):
            model = model_for(cfg, train_seed)
            directory = output / LEARNER / f"train_seed_{train_seed}"
            records = []

            def checkpoint(current, update, steps):
                nonlocal development_count
                path = directory / "checkpoints" / f"step_{steps:08d}.pt"
                restored, _ = save_checkpoint(
                    path, current, cfg, train_seed, update, steps
                )
                panel = []
                jobs = list(
                    itertools.product(
                        cfg["development_cohorts"],
                        ("middle", "low"),
                        cfg["development_seeds"],
                    )
                )
                for cs, cap, seed in tqdm(
                    jobs, desc=f"dev seed{train_seed}/step{steps}", leave=False
                ):
                    c = cohort_config(cs, cap, cfg["horizons"][0])
                    row = episode(
                        c,
                        seed,
                        LEARNER,
                        None,
                        cfg["scenarios"],
                        dict(
                            train_seed=train_seed,
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
            record = fit(model, cfg, train_seed, forbidden, journals, checkpoint)
            coverage.append(record)
            write_json(output / "training_coverage.json", coverage)
            chosen, status = select_checkpoint(records, selected_dev)
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / chosen["path"], directory / "model.pt")
            restored, payload = load_checkpoint(directory / "model.pt")
            if payload["physical_steps"] != chosen["steps"]:
                raise RuntimeError("selected checkpoint provenance")
            selections[str(train_seed)] = dict(
                status=status,
                selected_steps=chosen["steps"],
                selected_update=chosen["update"],
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
        # All selections are now frozen; first test access occurs here.
        rows = []
        control_panel = itertools.product(
            cfg["test_cohorts"],
            CAPACITIES,
            cfg["horizons"],
            cfg["test_seeds"],
            cfg["controls"],
        )
        for cs, cap, horizon, seed, name in tqdm(
            control_panel,
            total=5 * 3 * 2 * len(cfg["test_cohorts"]) * len(cfg["test_seeds"]),
            desc="test references/planner",
        ):
            c = cohort_config(cs, cap, horizon)
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
                    horizon=horizon,
                ),
                journals=journals,
            )
            rows.append(row)
            journals["episodes"].add(row)
        for train_seed in tqdm(
            cfg["train_seeds"], desc="test learned replicas", unit="seed"
        ):
            chosen = selections[str(train_seed)]
            model, _ = load_checkpoint(output / chosen["checkpoint"])
            panel = list(
                itertools.product(
                    cfg["test_cohorts"], CAPACITIES, cfg["horizons"], cfg["test_seeds"]
                )
            )
            for cs, cap, horizon, seed in tqdm(
                panel, desc=f"test PPO seed{train_seed}", leave=False
            ):
                c = cohort_config(cs, cap, horizon)
                row = episode(
                    c,
                    seed,
                    LEARNER,
                    None,
                    cfg["scenarios"],
                    dict(
                        train_seed=train_seed,
                        checkpoint_step=chosen["selected_steps"],
                        cohort=cs,
                        capacity=cap,
                        horizon=horizon,
                    ),
                    model,
                    journals,
                )
                rows.append(row)
                journals["episodes"].add(row)
        journals["episodes"].close()
        shutil.copy2(output / "episodes.partial.csv", output / "episodes.csv")
        paired, cohorts, contrasts = summarize(rows, selected_refs, cfg)
        write_csv(output / "paired_seed_metrics.csv", paired)
        write_csv(output / "cohort_metrics.csv", cohorts)
        actual = dict(
            models=len(coverage),
            training_steps=sum(r["env_steps"] for r in coverage),
            training_episodes=sum(r["episodes"] for r in coverage),
            optimizer_steps=sum(r["optimizer_steps"] for r in coverage),
            development_episodes=development_count,
            test_episodes=len(rows),
            test_decision_intervals=sum(r["horizon"] for r in rows),
            test_machine_rows=len(rows) * 4,
        )
        if actual != expected:
            raise RuntimeError("experiment population or budget mismatch")
        write_json(
            output / "summary.json",
            dict(
                contrasts=contrasts,
                development_infeasible_seeds=[
                    int(s) for s, r in selections.items() if r["status"] != "FEASIBLE"
                ],
                interpretation="Centralized PPO learnability on a finite held-out seed panel; no MARL comparison or necessity claim.",
                audits=dict(
                    complete_grids=True,
                    exact_training_budget=True,
                    checkpoint_reload=True,
                    reserved_seed_training_transitions_zero=True,
                    zero_teacher_labels=True,
                    probability_return_reconciliation=True,
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
