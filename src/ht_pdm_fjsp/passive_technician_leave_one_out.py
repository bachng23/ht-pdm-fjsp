"""Locked leave-one-component-out experiment for RA-QMIX."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import math
import platform
import statistics
import subprocess
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import _obs_tensor, stress_config
from ht_pdm_fjsp.passive_technician_marl import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    PassiveConfig,
    PassiveTechnicianEnv,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    COMPONENT_FLAGS,
    PassiveValueDecomposition,
    ValueTrainSettings,
    train_value_decomposition_checkpoints,
)


PROTOCOL_VERSION = "ra_qmix_leave_one_out_v1"
ALGORITHMS = ("qmix", "tqmix_no_edge", "tqmix_no_queue", "tqmix_no_cf", "tqmix")
REPORTING_NAMES = {
    "qmix": "Standard QMIX",
    "tqmix_no_edge": "RA-QMIX - edge",
    "tqmix_no_queue": "RA-QMIX - queue",
    "tqmix_no_cf": "RA-QMIX - CF",
    "tqmix": "Full RA-QMIX",
}
FULL_BUDGETS = (20_000, 50_000)
SMOKE_BUDGETS = (8, 16)
SEALED_TEST_SEEDS = tuple(range(201, 301))
COST_TOLERANCE = 1e-6


def environment_cells() -> dict[str, PassiveConfig]:
    base = stress_config()
    return {
        "in_distribution": base,
        "early_failure": replace(
            base, horizon=18, failure_age=4, failure_probability=0.60
        ),
        "slow_service": replace(
            base, horizon=18, service_time=((3, 5), (5, 3), (4, 4))
        ),
        "combined_pressure": replace(
            base,
            horizon=18,
            failure_age=4,
            failure_probability=0.60,
            service_time=((3, 5), (5, 3), (4, 4)),
        ),
    }


def profile_settings(
    profile: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], ValueTrainSettings]:
    if profile == "smoke":
        budgets = SMOKE_BUDGETS
        return (
            (11,),
            (101, 102, 103),
            budgets,
            ValueTrainSettings(
                episodes=max(budgets),
                hidden_dim=16,
                mixer_hidden_dim=8,
                replay_capacity=512,
                batch_size=4,
                learning_starts=4,
                train_frequency=1,
                target_update_interval=16,
            ),
        )
    if profile == "full":
        budgets = FULL_BUDGETS
        return (
            (11, 12, 13, 14, 15),
            tuple(range(101, 201)),
            budgets,
            ValueTrainSettings(episodes=max(budgets)),
        )
    raise ValueError(profile)


def _git_state() -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return revision, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: tuple[str, ...] | list[str] | None = None,
) -> None:
    names = list(fieldnames or dict.fromkeys(key for row in rows for key in row))
    if not names:
        raise ValueError(f"fieldnames required for empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _actions(
    model: PassiveValueDecomposition,
    env: PassiveTechnicianEnv,
    observations: tuple[tuple[int, ...], ...],
    device: torch.device,
) -> tuple[int, ...]:
    local = _obs_tensor(observations, env.config).to(device).unsqueeze(0)
    masks = torch.as_tensor(env.action_masks(), dtype=torch.bool, device=device)
    with torch.no_grad():
        values = model.agent_q(local)[0].masked_fill(~masks, -1e9)
    return tuple(int(action) for action in values.argmax(dim=-1).cpu().tolist())


def evaluate_diagnostic_episode(
    config: PassiveConfig,
    model: PassiveValueDecomposition,
    eval_seed: int,
    train_seed: int,
    scenario: str,
    budget: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate greedily and derive metrics from the simulator's exact timeline."""
    env = PassiveTechnicianEnv(config, seed=eval_seed)
    observations = env.reset()
    joint_actions: set[tuple[int, ...]] = set()
    defer_actions = 0
    action_steps = 0
    queue_length_total = 0
    max_queue_length = 0
    downtime_steps = 0
    technician_busy_steps = [0] * config.technicians
    service_starts = 0
    preventive_starts = 0
    corrective_starts = 0
    request_started_at: dict[tuple[int, int], int] = {}
    completed_waiting_times: list[int] = []
    completion_times: list[int] = []
    cost_failure = 0.0
    cost_downtime = 0.0
    cost_maintenance = 0.0
    cost_queue_waiting = 0.0
    cost_collision = 0.0
    done = False
    while not done:
        time = env.time
        state = env.state
        actions = _actions(model, env, observations, device)
        joint_actions.add(actions)
        defer_actions += sum(action == 0 for action in actions)
        action_steps += 1
        downtime_steps += sum(state.failed)

        queues = [list(queue) for queue in state.queues]
        queued = {machine for queue in queues for machine in queue}
        assigned = {machine for machine in state.assigned_machine if machine >= 0}
        for machine, action in enumerate(actions):
            if action == 0 or machine in queued or machine in assigned:
                continue
            technician = action - 1
            queues[technician].append(machine)
            queued.add(machine)
            request_started_at[(technician, machine)] = time
        for technician in range(config.technicians):
            if state.busy_until[technician] <= time and queues[technician]:
                machine = queues[technician].pop(0)
                service_starts += 1
                if state.failed[machine]:
                    corrective_starts += 1
                else:
                    preventive_starts += 1
                completed_waiting_times.append(
                    time - request_started_at.pop((technician, machine), time)
                )
                completion_times.append(time + config.service_time[machine][technician])

        observations, _, done, info = env.step(actions)
        queue_length = sum(len(queue) for queue in env.state.queues)
        queue_length_total += queue_length
        max_queue_length = max(max_queue_length, queue_length)
        for technician in range(config.technicians):
            technician_busy_steps[technician] += int(
                env.state.busy_until[technician] > time
            )
        cost_failure += config.failure_cost * info["failures"]
        cost_downtime += config.downtime_cost * sum(state.failed)
        cost_maintenance += config.maintenance_cost * info["jobs"]
        cost_queue_waiting += config.queue_waiting_cost * info["waiting"]
        if config.semantics in {"priority_resolver", "invalid_collision"}:
            cost_collision += config.collision_cost * info["collisions"]

    objective = float(env.metrics["objective"])
    component_sum = (
        cost_failure
        + cost_downtime
        + cost_maintenance
        + cost_queue_waiting
        + cost_collision
    )
    utilizations = [steps / config.horizon for steps in technician_busy_steps]
    total_actions = action_steps * config.machines
    return {
        "algorithm": model.algorithm,
        "reporting_name": REPORTING_NAMES[model.algorithm],
        "train_seed": train_seed,
        "scenario": scenario,
        "budget": budget,
        "eval_seed": eval_seed,
        "objective": objective,
        "cost_per_timestep": objective / config.horizon,
        "failure_cost": cost_failure,
        "downtime_cost": cost_downtime,
        "maintenance_cost": cost_maintenance,
        "queue_waiting_cost": cost_queue_waiting,
        "collision_penalty": cost_collision,
        "other_penalty": objective - component_sum,
        "cost_component_sum": component_sum,
        "cost_reconciliation_error": objective - component_sum,
        "failure_events": int(env.metrics["failures"]),
        "machine_downtime_steps": downtime_steps,
        "mean_queue_length": queue_length_total / config.horizon,
        "max_queue_length": max_queue_length,
        "queue_waiting_steps": int(env.metrics["waiting"]),
        "completed_waiting_time": sum(completed_waiting_times),
        "mean_waiting_time_per_started_request": (
            statistics.fmean(completed_waiting_times) if completed_waiting_times else 0.0
        ),
        "pending_requests_at_horizon": len(request_started_at),
        "request_count": total_actions - defer_actions,
        "busy_requests": int(env.metrics["busy_requests"]),
        "invalid_requests": int(env.metrics["invalid_requests"]),
        "collisions": int(env.metrics["collisions"]),
        "service_starts": service_starts,
        "service_completions": sum(value <= config.horizon for value in completion_times),
        "preventive_starts": preventive_starts,
        "corrective_starts": corrective_starts,
        "defer_fraction": defer_actions / total_actions if total_actions else 0.0,
        "unique_joint_actions": len(joint_actions),
        "action_steps": action_steps,
        **{
            f"technician_{index}_busy_steps": steps
            for index, steps in enumerate(technician_busy_steps)
        },
        **{
            f"technician_{index}_utilization": utilization
            for index, utilization in enumerate(utilizations)
        },
        "utilization_gap": max(utilizations) - min(utilizations),
    }


def _budget_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["algorithm"]),
            int(row["train_seed"]),
            str(row["scenario"]),
            int(row["budget"]),
        )
        groups.setdefault(key, []).append(row)
    metrics = (
        "objective",
        "cost_per_timestep",
        "failure_cost",
        "downtime_cost",
        "maintenance_cost",
        "queue_waiting_cost",
        "failure_events",
        "machine_downtime_steps",
        "mean_queue_length",
        "max_queue_length",
        "queue_waiting_steps",
        "completed_waiting_time",
        "mean_waiting_time_per_started_request",
        "pending_requests_at_horizon",
        "request_count",
        "busy_requests",
        "invalid_requests",
        "collisions",
        "service_starts",
        "service_completions",
        "preventive_starts",
        "corrective_starts",
        "defer_fraction",
        "unique_joint_actions",
        "technician_0_utilization",
        "technician_1_utilization",
        "utilization_gap",
    )
    result: list[dict[str, Any]] = []
    for (algorithm, train_seed, scenario, budget), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "algorithm": algorithm,
            "reporting_name": REPORTING_NAMES[algorithm],
            "train_seed": train_seed,
            "scenario": scenario,
            "budget": budget,
            "evaluation_count": len(group),
        }
        for metric in metrics:
            item[f"{metric}_mean"] = statistics.fmean(float(row[metric]) for row in group)
        result.append(item)
    return result


def _paired_differences(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    means = {
        (str(row["algorithm"]), int(row["train_seed"]), str(row["scenario"]), int(row["budget"])): float(row["objective_mean"])
        for row in summary_rows
    }
    rows: list[dict[str, Any]] = []
    contrasts = (
        ("edge_ablation_minus_full", "tqmix_no_edge", "tqmix"),
        ("queue_ablation_minus_full", "tqmix_no_queue", "tqmix"),
        ("cf_ablation_minus_full", "tqmix_no_cf", "tqmix"),
        ("qmix_minus_full", "qmix", "tqmix"),
    )
    cells = sorted({(key[2], key[3]) for key in means})
    train_seeds = sorted({key[1] for key in means})
    for contrast, left, right in contrasts:
        for scenario, budget in cells:
            deltas: list[float] = []
            for train_seed in train_seeds:
                delta = means[left, train_seed, scenario, budget] - means[right, train_seed, scenario, budget]
                deltas.append(delta)
                rows.append({
                    "row_type": "training_seed",
                    "contrast": contrast,
                    "left_algorithm": left,
                    "right_algorithm": right,
                    "scenario": scenario,
                    "budget": budget,
                    "train_seed": train_seed,
                    "delta_objective": delta,
                    "delta_mean": "",
                    "delta_sd": "",
                    "delta_min": "",
                    "delta_max": "",
                })
            rows.append({
                "row_type": "across_training_seeds",
                "contrast": contrast,
                "left_algorithm": left,
                "right_algorithm": right,
                "scenario": scenario,
                "budget": budget,
                "train_seed": "",
                "delta_objective": "",
                "delta_mean": statistics.fmean(deltas),
                "delta_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
                "delta_min": min(deltas),
                "delta_max": max(deltas),
            })
    return rows


def _runtime_metadata() -> dict[str, Any]:
    packages = {}
    for name in ("numpy", "torch", "tqdm"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "torch_cuda_version": torch.version.cuda,
    }


def run(args: argparse.Namespace) -> Path:
    train_seeds, evaluation_seeds, budgets, settings = profile_settings(args.profile)
    if set(train_seeds) & set(evaluation_seeds) or set(train_seeds) & set(SEALED_TEST_SEEDS) or set(evaluation_seeds) & set(SEALED_TEST_SEEDS):
        raise ValueError("training, evaluation, and sealed seed panels must be disjoint")
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    cells = environment_cells()
    train_config = cells["in_distribution"]
    revision, dirty = _git_state()
    component_mapping = {
        algorithm: {
            "edge_utilities": COMPONENT_FLAGS[algorithm][0],
            "queue_mixer": COMPONENT_FLAGS[algorithm][1],
            "counterfactual_loss": COMPONENT_FLAGS[algorithm][2],
            "lambda_cf": settings.lambda_cf if COMPONENT_FLAGS[algorithm][2] else 0.0,
        }
        for algorithm in ALGORITHMS
    }
    expected_rows = len(ALGORITHMS) * len(train_seeds) * len(budgets) * len(cells) * len(evaluation_seeds)
    expected_checkpoints = len(ALGORITHMS) * len(train_seeds) * len(budgets)
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "experiment": "ra_qmix_leave_one_out",
        "protocol_version": PROTOCOL_VERSION,
        "profile": args.profile,
        "hypotheses": {
            "edge": "Removing technician-edge utilities increases nominal 50k objective cost.",
            "queue": "Removing queue-conditioned mixing increases nominal 50k objective cost.",
            "counterfactual": "Removing counterfactual consistency loss increases nominal 50k objective cost.",
            "reference": "Full RA-QMIX has lower nominal 50k objective cost than standard QMIX.",
        },
        "primary_endpoint": "mean objective cost by training seed on in_distribution at 50000 episodes",
        "secondary_metrics": [
            "cost decomposition", "failure events", "downtime steps", "queue metrics",
            "request diagnostics", "maintenance starts/completions", "technician utilization",
            "TD/raw-CF/weighted-CF/total training losses",
        ],
        "stopping_rule": "Train continuously to the fixed maximum budget; stop only for error, non-finite loss, or audit violation.",
        "algorithms": list(ALGORITHMS),
        "reporting_names": REPORTING_NAMES,
        "component_mapping": component_mapping,
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "checkpoint_budgets": list(budgets),
        "training_scenario": "in_distribution",
        "environment_cells": {name: asdict(config) for name, config in cells.items()},
        "recovery_semantics": "age and failure reset at maintenance start; machine remains assigned/unavailable until service completion; downtime is charged from pre-service state",
        "value_settings": asdict(settings),
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device,
        "resolved_device": str(device),
        "git_revision": revision,
        "git_dirty": dirty,
        "runtime": _runtime_metadata(),
        "cost_reconciliation_tolerance": COST_TOLERANCE,
        "expected_counts": {
            "evaluation_rows": expected_rows,
            "checkpoints": expected_checkpoints,
            "training_trajectories": len(ALGORITHMS) * len(train_seeds),
        },
        "started_at": datetime.now(UTC).isoformat(),
    }
    _write_json(output / "manifest.json", manifest)
    _write_json(
        output / "benchmark_config.json",
        {
            "protocol_version": PROTOCOL_VERSION,
            "environment_cells": manifest["environment_cells"],
            "objective_version": OBJECTIVE_VERSION,
            "observation_version": OBSERVATION_VERSION,
        },
    )
    _write_json(
        output / "resolved_config.json",
        {
            "profile": args.profile,
            "algorithms": list(ALGORITHMS),
            "component_mapping": component_mapping,
            "train_seeds": train_seeds,
            "evaluation_seeds": evaluation_seeds,
            "sealed_test_seeds": SEALED_TEST_SEEDS,
            "budgets": budgets,
            "settings": asdict(settings),
            "device": str(device),
        },
    )

    try:
        parameter_rows = []
        for algorithm in ALGORITHMS:
            model = PassiveValueDecomposition(
                train_config, algorithm, settings.hidden_dim, settings.mixer_hidden_dim
            )
            parameter_rows.append(
                {
                    "algorithm": algorithm,
                    "reporting_name": REPORTING_NAMES[algorithm],
                    **component_mapping[algorithm],
                    **model.parameter_counts(),
                }
            )
        _write_csv(output / "parameter_counts.csv", parameter_rows)

        episode_rows: list[dict[str, Any]] = []
        checkpoint_paths: list[str] = []
        optimizer_updates = {
            (algorithm, seed): 0 for algorithm in ALGORITHMS for seed in train_seeds
        }
        losses_finite = True
        for algorithm in ALGORITHMS:
            for train_seed in tqdm(train_seeds, desc=f"train seeds/{algorithm}", unit="seed"):
                root = output / algorithm / f"train_seed_{train_seed}" / "checkpoints"
                checkpoints, progress = train_value_decomposition_checkpoints(
                    train_config,
                    algorithm,
                    train_seed,
                    budgets,
                    root,
                    settings,
                    device,
                )
                trajectory_rows = [{"algorithm": algorithm, **row} for row in progress]
                _append_csv(output / "training_progress.csv", trajectory_rows)
                _append_csv(output / "training_episodes.csv", trajectory_rows)
                optimizer_updates[algorithm, train_seed] = sum(
                    int(row["update_count"]) for row in trajectory_rows
                )
                losses_finite = losses_finite and all(
                    math.isfinite(float(row[name]))
                    for row in trajectory_rows
                    for name in ("td_loss", "raw_cf_loss", "weighted_cf_loss", "total_loss")
                    if row.get(name) is not None
                )
                for budget in budgets:
                    checkpoint = checkpoints[budget]
                    checkpoint_paths.append(str(checkpoint.relative_to(output)))
                    model = PassiveValueDecomposition.load(checkpoint, train_config, device)
                    for scenario, config in cells.items():
                        for eval_seed in tqdm(
                            evaluation_seeds,
                            desc=f"evaluate {algorithm}/{train_seed}/{budget}/{scenario}",
                            unit="episode",
                            leave=False,
                        ):
                            episode_rows.append(
                                evaluate_diagnostic_episode(
                                    config,
                                    model,
                                    eval_seed,
                                    train_seed,
                                    scenario,
                                    budget,
                                    device,
                                )
                            )
                        _write_csv(output / "episodes.partial.csv", episode_rows)

        _write_csv(output / "episodes.csv", episode_rows)
        coordination_rows = [
            {
                "algorithm": row["algorithm"],
                "train_seed": row["train_seed"],
                "scenario": row["scenario"],
                "budget": row["budget"],
                "eval_seed": row["eval_seed"],
                "invalid_requests": row["invalid_requests"],
                "collisions": row["collisions"],
                "cost_reconciliation_error": row["cost_reconciliation_error"],
                "cost_reconciled": abs(float(row["cost_reconciliation_error"])) <= COST_TOLERANCE,
                "resource_semantics_valid": all(
                    0.0 <= float(row[f"technician_{index}_utilization"]) <= 1.0
                    for index in range(train_config.technicians)
                ),
            }
            for row in episode_rows
        ]
        _write_csv(output / "coordination.csv", coordination_rows)
        budget_summary = _budget_summary(episode_rows)
        paired = _paired_differences(budget_summary)
        _write_csv(output / "budget_summary.csv", budget_summary)
        _write_csv(output / "paired_differences.csv", paired)

        keys = [
            (
                row["algorithm"], row["train_seed"], row["scenario"],
                row["budget"], row["eval_seed"],
            )
            for row in episode_rows
        ]
        audits = {
            "expected_evaluation_rows": expected_rows,
            "actual_evaluation_rows": len(episode_rows),
            "all_expected_rows_present": len(episode_rows) == expected_rows,
            "unique_episode_keys": len(keys) == len(set(keys)),
            "expected_checkpoints": expected_checkpoints,
            "actual_checkpoints": len(checkpoint_paths),
            "all_checkpoints_present": len(checkpoint_paths) == expected_checkpoints,
            "training_trajectory_count": len(optimizer_updates),
            "all_trajectories_updated": all(value > 0 for value in optimizer_updates.values()),
            "losses_finite": losses_finite,
            "cost_reconciliation_passed": all(row["cost_reconciled"] for row in coordination_rows),
            "resource_semantics_passed": all(row["resource_semantics_valid"] for row in coordination_rows),
            "sealed_test_panel_closed": not any(int(row["eval_seed"]) in SEALED_TEST_SEEDS for row in episode_rows),
        }
        if not all(
            audits[key]
            for key in (
                "all_expected_rows_present", "unique_episode_keys", "all_checkpoints_present",
                "all_trajectories_updated", "losses_finite", "cost_reconciliation_passed",
                "resource_semantics_passed", "sealed_test_panel_closed",
            )
        ):
            raise RuntimeError(f"experiment audit failed: {audits}")
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "primary_endpoint": manifest["primary_endpoint"],
            "delta_definition": "left algorithm mean objective minus right algorithm mean objective; positive ablation-minus-full supports the component",
            "budget_summary": budget_summary,
            "paired_differences": paired,
            "audits": audits,
        }
        _write_json(output / "summary.json", summary)
        manifest.update(
            {
                "status": "COMPLETED",
                "finished_at": datetime.now(UTC).isoformat(),
                "actual_counts": {
                    "evaluation_rows": len(episode_rows),
                    "checkpoints": len(checkpoint_paths),
                    "training_trajectories": len(optimizer_updates),
                },
                "checkpoint_paths": checkpoint_paths,
                "audits": audits,
                "outputs": sorted(
                    str(path.relative_to(output))
                    for path in output.rglob("*")
                    if path.is_file()
                ),
            }
        )
        _write_json(output / "manifest.json", manifest)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return output
    except BaseException as error:
        manifest.update(
            {
                "status": "FAILED",
                "finished_at": datetime.now(UTC).isoformat(),
                "failure_type": type(error).__name__,
                "failure_message": str(error),
            }
        )
        _write_json(output / "manifest.json", manifest)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
