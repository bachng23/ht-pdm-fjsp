"""Compare centralized matching PPO and cooperative machine agents with heterogeneous workers."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import itertools
import hashlib
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
from ht_pdm_fjsp.maintenance_heterogeneous_model import (
    CONDITIONS,
    REFERENCES,
    feasible,
    cohort_config,
    plan,
    reference,
    contention,
)
from ht_pdm_fjsp.maintenance_heterogeneous_policy import (
    MatchingActorCritic,
    FEATURE_CONTRACT,
    ARMS,
    CENTRAL,
    FIXED,
    MARL,
)
from ht_pdm_fjsp.maintenance_heterogeneous_training import fit
from ht_pdm_fjsp.maintenance_dispatch import role_seed, validate_matching
from ht_pdm_fjsp.maintenance_waiting import WaitingEnv, ENV_VERSION

VERSION = "maintenance_heterogeneous_marl_v1"
CI97 = 2.685010846816525
CI95 = 2.2621571628540993
ROOT = Path(__file__).resolve().parents[2]


def settings(profile):
    full = profile == "full"
    return dict(
        profile=profile,
        train_seeds=list(range(150000, 150010)) if full else [155000],
        development_cohorts=list(range(151000, 151010)) if full else [155020],
        development_seeds=list(range(152000, 152020)) if full else [155030],
        test_cohorts=list(range(153000, 153010)) if full else [155010],
        test_seeds=list(range(154000, 154050)) if full else list(range(155040, 155043)),
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
        conditions=list(CONDITIONS),
        arms=list(ARMS),
        controls=[*REFERENCES, "joint_rollout", "fixed_allocation_rollout"],
        training_condition_mixture={"specialized": 0.8, "nominal": 0.2},
        training_seed_roles=[
            "heterogeneous_learning_configuration:<episode>",
            "heterogeneous_learning_environment:<episode>",
            "heterogeneous_learning_actions",
            "heterogeneous_learning_minibatches",
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
    n = len(cfg["train_seeds"]) * len(ARMS)
    dev = len(cfg["development_cohorts"]) * len(cfg["development_seeds"])
    test = len(cfg["test_cohorts"]) * len(cfg["test_seeds"])
    episodes = (
        (len(cfg["controls"]) + n) * len(CONDITIONS) * len(cfg["horizons"]) * test
    )
    return dict(
        models=n,
        training_steps=n * cfg["env_steps"],
        training_episodes=n * cfg["env_steps"] // cfg["horizons"][0],
        optimizer_steps=n
        * (cfg["env_steps"] // cfg["rollout"])
        * cfg["epochs"]
        * cfg["rollout"]
        // cfg["minibatch"],
        development_episodes=len(REFERENCES) * len(CONDITIONS) * dev
        + n * (len(cfg["checkpoint_updates"]) + 1) * 2 * dev,
        test_episodes=episodes,
        test_decision_intervals=(len(cfg["controls"]) + n)
        * len(CONDITIONS)
        * test
        * sum(cfg["horizons"]),
        test_machine_rows=episodes * 4,
    )


def model_for(cfg, seed, algorithm=CENTRAL):
    torch.manual_seed(seed)
    return MatchingActorCritic(algorithm, cfg["hidden"]).eval()


def load_checkpoint(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["protocol"] != VERSION
        or payload["feature_contract"] != FEATURE_CONTRACT
        or payload["environment_version"] != ENV_VERSION
        or payload["algorithm"] not in ARMS
    ):
        raise ValueError("checkpoint contract mismatch")
    model = model_for(payload["settings"], payload["train_seed"], payload["algorithm"])
    model.load_state_dict(payload["state_dict"])
    return model.eval(), payload


def save_checkpoint(path, model, cfg, seed, update, steps):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            protocol=VERSION,
            environment_version=ENV_VERSION,
            feature_contract=FEATURE_CONTRACT,
            algorithm=model.algorithm,
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
    c = cohort_config(cfg["development_cohorts"][0], "specialized", cfg["horizons"][0])
    env = WaitingEnv(c)
    env.reset(cfg["development_seeds"][0])
    for t in range(c.horizon):
        chosen, diag = model.select(c, env.state, c.horizon - t)
        if (chosen, diag) != restored.select(c, env.state, c.horizon - t):
            raise RuntimeError("checkpoint probability/value/action reload mismatch")
        env.step(chosen)
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
            maintenance_cost=0.0,
            failure_cost=0.0,
            unavailability_cost=0.0,
            waiting_cost=0.0,
        )
        for m in range(c.machines)
    ]
    samples = [[] for _ in machine]
    excess = choices = backlog = utilization = subset_total = 0
    preferred = busy_preferred = rejections = attempts = 0
    worker_ticks = [0] * c.technicians
    planning_seconds = 0.0
    spreads = []
    errors = []
    for t in range(c.horizon):
        state = env.state
        candidates = feasible(c, state)
        free = c.technicians - sum(r > 0 for r in state.remaining)
        urgent = sum(
            m not in state.assigned
            and (state.failed[m] or state.ages[m] >= c.failure_age - 1)
            for m in range(c.machines)
        )
        urgent_excess = max(0, urgent - free)
        begin = time.perf_counter()
        diag = {}
        if name in ARMS:
            if model is None:
                raise ValueError("learned policy needs checkpoint")
            selected, diag = model.select(c, state, c.horizon - t)
        elif name in ("joint_rollout", "fixed_allocation_rollout"):
            selected, diag = plan(
                c,
                state,
                c.horizon - t,
                continuation,
                role_seed(seed, f"heterogeneous_planner:{t}"),
                scenarios,
                fixed=name == "fixed_allocation_rollout",
            )
            spreads.append(diag["score_spread"])
            errors.append(diag["selected_mc_se"])
        else:
            selected = reference(c, state, c.horizon - t, name)
        elapsed = time.perf_counter() - begin
        planning_seconds += elapsed
        pairs = selected
        validate_matching(c, state, pairs)
        if pairs not in candidates:
            raise RuntimeError("infeasible controller output")
        selected_m = {m for m, j in pairs}
        serving = {m for m in state.assigned if m >= 0} | selected_m
        pe, bp = contention(c, state)
        preferred += pe
        busy_preferred += bp
        rejections += diag.get("collision_rejections", 0)
        attempts += diag.get("proposal_attempts", len(pairs))
        active_workers = {j for j, r in enumerate(state.remaining) if r} | {
            j for m, j in pairs
        }
        for j in active_workers:
            worker_ticks[j] += 1
        for m, row in enumerate(machine):
            waiting = bool(state.failed[m] and m not in serving)
            wait = state.pending_wait[m] + int(waiting)
            samples[m].append(wait)
            row["service_starts"] += int(m in selected_m)
            row["unavailable_ticks"] += int(state.failed[m] or m in serving)
            row["failed_waiting_ticks"] += int(waiting)
            row["overdue_waiting_ticks"] += int(waiting and wait > c.waiting_limit)
            row["max_wait"] = max(row["max_wait"], wait)
        _, _, _, info = env.step(pairs)
        for m, row in enumerate(machine):
            row["maintenance_cost"] += (
                (c.corrective_cost if state.failed[m] else c.preventive_cost)
                if m in selected_m
                else 0.0
            )
            row["unavailability_cost"] += c.unavailable_cost * int(
                state.failed[m] or m in serving
            )
            new_failure = (
                not state.failed[m] and m not in serving and env.state.failed[m]
            )
            row["failure_cost"] += c.failure_cost * int(new_failure)
            row["waiting_cost"] += c.waiting_price * int(
                state.failed[m]
                and m not in serving
                and state.pending_wait[m] + 1 > c.waiting_limit
            )
        excess += urgent_excess
        choices += len(candidates) > 1
        subset_total += len(candidates)
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
                    feasible_matchings=len(candidates),
                    preferred_worker_excess=pe,
                    busy_preferred_requests=bp,
                    inference_seconds=elapsed,
                    controller_diagnostics=json.dumps(diag),
                )
            )
    for metric in (
        "unavailable_ticks",
        "failed_waiting_ticks",
        "overdue_waiting_ticks",
        "maintenance_cost",
        "failure_cost",
        "unavailability_cost",
        "waiting_cost",
    ):
        if sum(row[metric] for row in machine) != env.metrics[metric]:
            raise RuntimeError("machine-cost reconciliation")
    if (
        sum(row["service_starts"] for row in machine) != env.metrics["jobs"]
        or sum(worker_ticks) != env.metrics["service_ticks"]
    ):
        raise RuntimeError("worker/job reconciliation")
    for m, row in enumerate(machine):
        row["base_cost"] = (
            row["maintenance_cost"] + row["failure_cost"] + row["unavailability_cost"]
        )
        row["objective"] = row["base_cost"] + row["waiting_cost"]
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
        mean_feasible_matchings=subset_total / c.horizon,
        mean_preferred_worker_excess=preferred / c.horizon,
        mean_busy_preferred_requests=busy_preferred / c.horizon,
        proposal_attempts=attempts,
        collision_rejections=rejections,
        proposal_rejection_rate=rejections / attempts if attempts else 0.0,
        worker_utilization=json.dumps([v / c.horizon for v in worker_ticks]),
        planner_mean_score_spread=float(np.mean(spreads)) if spreads else 0.0,
        planner_mean_mc_se=float(np.mean(errors)) if errors else 0.0,
        inference_seconds=planning_seconds,
    )


def mean(rows, metric):
    if not rows:
        raise ValueError("empty evaluation panel")
    return float(np.mean([r[metric] for r in rows]))


def select_checkpoint(records, reference_rows):
    ref_middle = [r for r in reference_rows if r["condition"] == "specialized"]
    ref_low = [r for r in reference_rows if r["condition"] == "nominal"]
    viable = []
    for record in records:
        if record["update"] == 0:
            continue
        panel = record["episodes"]
        middle = [r for r in panel if r["condition"] == "specialized"]
        low = [r for r in panel if r["condition"] == "nominal"]
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
    controls = [r for r in rows if r["train_seed"] == -1]
    expected = set(
        itertools.product(
            cfg["test_cohorts"],
            CONDITIONS,
            cfg["horizons"],
            cfg["test_seeds"],
            cfg["controls"],
        )
    )
    keys = {
        (r["cohort"], r["condition"], r["horizon"], r["seed"], r["controller"])
        for r in controls
    }
    if len(keys) != len(controls) or keys != expected:
        raise RuntimeError("incomplete control panel")
    learners = [r for r in rows if r["controller"] in ARMS]
    expected = set(
        itertools.product(
            ARMS,
            cfg["train_seeds"],
            cfg["test_cohorts"],
            CONDITIONS,
            cfg["horizons"],
            cfg["test_seeds"],
        )
    )
    keys = {
        (
            r["controller"],
            r["train_seed"],
            r["cohort"],
            r["condition"],
            r["horizon"],
            r["seed"],
        )
        for r in learners
    }
    if len(keys) != len(learners) or keys != expected:
        raise RuntimeError("incomplete learned panel")
    # Index once: averaging replicated rows over thousands of episodes is not inference.
    groups = {}
    for r in rows:
        groups.setdefault(
            (r["controller"], r["train_seed"], r["condition"], r["horizon"]), []
        ).append(r)

    def panel(arm, seed, condition, h):
        return groups[(arm, seed, condition, h)]

    paired = []
    cohorts = []
    contrasts = {}
    H = cfg["horizons"][0]
    for condition, h in itertools.product(CONDITIONS, cfg["horizons"]):
        ref = selected_refs[condition]
        comparisons = [
            ("assignment", CENTRAL, FIXED, 0.03, 0.02, CI97),
            ("cooperative", MARL, CENTRAL, 0.02, None, CI97),
            *(("economic", arm, ref, 0.05, 0.03, CI95) for arm in ARMS),
        ]
        for label, left, right, threshold, base_threshold, critical in comparisons:
            left_panels = [
                panel(left, seed, condition, h) for seed in cfg["train_seeds"]
            ]
            right_panels = [
                panel(right, seed if right in ARMS else -1, condition, h)
                for seed in cfg["train_seeds"]
            ]
            lc = [mean(p, "objective") for p in left_panels]
            rc = [mean(p, "objective") for p in right_panels]
            lb = [mean(p, "base_cost") for p in left_panels]
            rb = [mean(p, "base_cost") for p in right_panels]
            deltas = np.asarray(lc) - rc
            avg = float(deltas.mean())
            half = (
                critical * float(deltas.std(ddof=1)) / math.sqrt(10)
                if len(deltas) == 10
                else None
            )
            reduction = 1 - float(np.mean(lc)) / float(np.mean(rc))
            base_reduction = 1 - float(np.mean(lb)) / float(np.mean(rb))
            nominal = mean(
                [
                    r
                    for seed in cfg["train_seeds"]
                    for r in panel(left, seed, "nominal", H)
                ],
                "base_cost",
            )
            nominal_ref = mean(
                panel(selected_refs["nominal"], -1, "nominal", H), "base_cost"
            )
            violation = float(
                np.mean([mean(p, "waiting_violation") for p in left_panels])
            )
            waiting_ref = mean(panel(ref, -1, condition, h), "waiting_violation")
            gates = dict(
                priced_reduction=reduction >= threshold,
                win_8_of_10=int((deltas < -1e-9).sum()) >= 8,
                ci_upper_below_zero=half is not None and avg + half < 0,
                nominal_ratio_110pct=nominal / nominal_ref <= 1.1,
                waiting_violation_guard=violation <= waiting_ref + 0.02,
            )
            if base_threshold is not None:
                gates["base_reduction"] = base_reduction >= base_threshold
            primary = condition == "specialized" and h == H
            status = (
                "ENGINEERING_ONLY"
                if cfg["profile"] == "smoke"
                else ("PASS" if all(gates.values()) else "FAIL")
                if primary
                else "EXPLORATORY"
            )
            key = f"{label}:{left}_vs_{right}:{condition}_H{h}"
            contrasts[key] = dict(
                left=left,
                right=right,
                condition=condition,
                horizon=h,
                left_cost=float(np.mean(lc)),
                right_cost=float(np.mean(rc)),
                priced_reduction=reduction,
                base_reduction=base_reduction,
                paired_difference=avg,
                training_seed_wins=int((deltas < -1e-9).sum()),
                confidence_level=0.975 if label != "economic" else 0.95,
                ci=[avg - half, avg + half] if half is not None else None,
                uncertainty_scope="training-replica variation conditional on fixed held-out cohort/shock panel",
                nominal_base_ratio=nominal / nominal_ref,
                left_waiting_violation=violation,
                reference_waiting_violation=waiting_ref,
                gates=gates,
                status=status,
            )
            for seed, lp, rp, lcost, rcost, delta in zip(
                cfg["train_seeds"],
                left_panels,
                right_panels,
                lc,
                rc,
                deltas,
                strict=True,
            ):
                paired.append(
                    dict(
                        contrast=key,
                        train_seed=seed,
                        left=left,
                        right=right,
                        condition=condition,
                        horizon=h,
                        left_cost=lcost,
                        right_cost=rcost,
                        difference=float(delta),
                    )
                )
                for cs in cfg["test_cohorts"]:
                    lrows = [r for r in lp if r["cohort"] == cs]
                    rrows = [r for r in rp if r["cohort"] == cs]
                    cohorts.append(
                        dict(
                            contrast=key,
                            train_seed=seed,
                            cohort=cs,
                            left=left,
                            right=right,
                            condition=condition,
                            horizon=h,
                            left_cost=mean(lrows, "objective"),
                            right_cost=mean(rrows, "objective"),
                            difference=mean(lrows, "objective")
                            - mean(rrows, "objective"),
                        )
                    )
    homogeneous = {}
    for r in controls:
        if r["condition"] == "homogeneous" and r["controller"] in (
            "joint_rollout",
            "fixed_allocation_rollout",
        ):
            homogeneous.setdefault((r["cohort"], r["horizon"], r["seed"]), {})[
                r["controller"]
            ] = r["objective"]
    if any(
        abs(v["joint_rollout"] - v["fixed_allocation_rollout"]) > 1e-9
        for v in homogeneous.values()
    ):
        raise RuntimeError("homogeneous allocation invariance")
    return paired, cohorts, contrasts


def mechanisms(rows, selected_refs, cfg):
    metrics = (
        "objective",
        "base_cost",
        "maintenance_cost",
        "unavailability_cost",
        "failure_cost",
        "waiting_cost",
        "failures",
        "waiting_violation",
        "max_wait",
        "p95_wait",
        "terminal_pending",
        "utilization",
        "mean_urgent_excess",
        "mean_preferred_worker_excess",
        "mean_busy_preferred_requests",
        "proposal_rejection_rate",
        "inference_seconds",
    )
    panels = {}
    for r in rows:
        panels.setdefault((r["condition"], r["horizon"], r["controller"]), []).append(r)
    aggregate = {
        f"{cond}_H{h}:{name}": {k: mean(p, k) for k in metrics}
        for (cond, h, name), p in panels.items()
    }
    headroom = {}
    for cond, h in itertools.product(CONDITIONS, cfg["horizons"]):
        full = mean(panels[(cond, h, "joint_rollout")], "objective")
        fixed = mean(panels[(cond, h, "fixed_allocation_rollout")], "objective")
        ref = mean(panels[(cond, h, selected_refs[cond])], "objective")
        headroom[f"{cond}_H{h}"] = dict(
            full_matching_rollout_cost=full,
            fixed_allocation_rollout_cost=fixed,
            selected_rule_cost=ref,
            allocation_rollout_savings=fixed - full,
            relative_allocation_savings=1 - full / fixed if fixed else None,
            interpretation="Common root forecasts; subsequent continuation identical. Not optimality or causal proof of MARL necessity.",
        )
    return dict(aggregate=aggregate, allocation_headroom=headroom)


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
            *Path(__file__).parent.glob("maintenance_heterogeneous*.py"),
            *Path(__file__).parent.glob("maintenance_contention*.py"),
            ROOT / "src/ht_pdm_fjsp/maintenance_dispatch.py",
            ROOT / "src/ht_pdm_fjsp/maintenance_dispatch_policy.py",
            ROOT / "pyproject.toml",
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
                CONDITIONS,
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
            cfg["development_cohorts"], CONDITIONS, cfg["development_seeds"], REFERENCES
        )
        for cs, cap, seed, name in tqdm(
            panel,
            total=4
            * len(CONDITIONS)
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
                    condition=cap,
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
                        if r["condition"] == cap and r["controller"] == name
                    ],
                    "objective",
                )
                for name in REFERENCES
            }
            for cap in CONDITIONS
        }
        selected_refs = {
            cap: min(REFERENCES, key=lambda name: (dev_means[cap][name], name))
            for cap in CONDITIONS
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
            r for r in dev if r["controller"] == selected_refs[r["condition"]]
        ]
        selections = {}
        coverage = []
        development_count = len(dev)
        initial_hashes = {}
        for arm, train_seed in tqdm(
            list(itertools.product(ARMS, cfg["train_seeds"])),
            desc="learning replicas",
            unit="model",
        ):
            model = model_for(cfg, train_seed, arm)

            digest = hashlib.sha256(
                b"".join(
                    v.detach().numpy().tobytes() for v in model.state_dict().values()
                )
            ).hexdigest()
            if train_seed in initial_hashes and initial_hashes[train_seed] != digest:
                raise RuntimeError("unmatched initial parameters")
            initial_hashes[train_seed] = digest
            directory = output / arm / f"train_seed_{train_seed}"
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
                        ("specialized", "nominal"),
                        cfg["development_seeds"],
                    )
                )
                for cs, cap, seed in tqdm(
                    jobs, desc=f"dev {arm} seed{train_seed}/step{steps}", leave=False
                ):
                    c = cohort_config(cs, cap, cfg["horizons"][0])
                    row = episode(
                        c,
                        seed,
                        arm,
                        None,
                        cfg["scenarios"],
                        dict(
                            train_seed=train_seed,
                            checkpoint_step=steps,
                            cohort=cs,
                            condition=cap,
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
            if (
                coverage
                and sum(p.numel() for p in model.parameters())
                != coverage[0]["parameter_count"]
            ):
                raise RuntimeError("unmatched parameter count")
            coverage.append(
                {
                    **record,
                    "algorithm": arm,
                    "initial_parameters_sha256": digest,
                    "parameter_count": sum(p.numel() for p in model.parameters()),
                }
            )
            write_json(output / "training_coverage.json", coverage)
            chosen, status = select_checkpoint(records, selected_dev)
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copy2(output / chosen["path"], directory / "model.pt")
            restored, payload = load_checkpoint(directory / "model.pt")
            if payload["physical_steps"] != chosen["steps"]:
                raise RuntimeError("selected checkpoint provenance")
            selections[f"{arm}:{train_seed}"] = dict(
                algorithm=arm,
                train_seed=train_seed,
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
            CONDITIONS,
            cfg["horizons"],
            cfg["test_seeds"],
            cfg["controls"],
        )
        for cs, cap, horizon, seed, name in tqdm(
            control_panel,
            total=len(cfg["controls"])
            * len(CONDITIONS)
            * len(cfg["horizons"])
            * len(cfg["test_cohorts"])
            * len(cfg["test_seeds"]),
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
                    condition=cap,
                    horizon=horizon,
                ),
                journals=journals,
            )
            rows.append(row)
            journals["episodes"].add(row)
        for arm, train_seed in tqdm(
            list(itertools.product(ARMS, cfg["train_seeds"])),
            desc="test learned replicas",
            unit="model",
        ):
            chosen = selections[f"{arm}:{train_seed}"]
            model, _ = load_checkpoint(output / chosen["checkpoint"])
            panel = list(
                itertools.product(
                    cfg["test_cohorts"], CONDITIONS, cfg["horizons"], cfg["test_seeds"]
                )
            )
            for cs, cap, horizon, seed in tqdm(
                panel, desc=f"test {arm} seed{train_seed}", leave=False
            ):
                c = cohort_config(cs, cap, horizon)
                row = episode(
                    c,
                    seed,
                    arm,
                    None,
                    cfg["scenarios"],
                    dict(
                        train_seed=train_seed,
                        checkpoint_step=chosen["selected_steps"],
                        cohort=cs,
                        condition=cap,
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
                    s for s, r in selections.items() if r["status"] != "FEASIBLE"
                ],
                mechanisms=mechanisms(rows, selected_refs, cfg),
                interpretation="Assignment value and cooperative machine-policy benefit on a fixed held-out panel; full public information and rotating arbitration at execution; no MARL necessity claim. Heterogeneity preserves row means, not effective throughput. Skill masks are separate capacity sensitivity.",
                audits=dict(
                    complete_grids=True,
                    homogeneous_allocation_invariance=True,
                    matched_parameter_counts=True,
                    matched_initial_parameters=True,
                    raw_proposal_likelihood=True,
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
