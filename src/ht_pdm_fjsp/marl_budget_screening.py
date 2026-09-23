"""Screen 100k--500k MARL budgets before launching replicated full runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config
from ht_pdm_fjsp.three_way_marl_experiment import (
    IQL,
    QMIX,
    evaluate_episode,
)
from ht_pdm_fjsp.value_decomposition import (
    ValueDecompositionPolicy,
    ValueLearningSettings,
    train_value_policy,
)


LOCKED_FULL_CHECKPOINTS = (100_000, 200_000, 300_000, 400_000, 500_000)
SCREENING_CONDITIONS = (IQL, QMIX)
SEALED_TEST_SEEDS = tuple(range(62_000, 62_100))
OBJECTIVE_TOLERANCE = 0.05
MAKESPAN_TOLERANCE = 0.05
FAILURE_TOLERANCE = 0.10


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 400,
            "train_seed": 76_100,
            "evaluation_seeds": tuple(range(61_990, 61_993)),
            "n_envs": 4,
            "replay_capacity": 400,
            "learning_starts": 32,
            "value_batch_size": 16,
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "train_seed": 76_000,
            "evaluation_seeds": tuple(range(61_900, 61_950)),
            "n_envs": 8,
            "replay_capacity": 20_000,
            "learning_starts": 2_000,
            "value_batch_size": 256,
        }
    raise ValueError(f"Unknown profile: {profile}")


def _checkpoint_targets(total_timesteps: int) -> tuple[int, ...]:
    return tuple(total_timesteps * part // 5 for part in range(1, 6))


def _checkpoint_path(cell: Path, target: int) -> Path:
    return cell / "checkpoints" / f"model_{target}_steps.pt"


def _load_policy(path: Path, device: str) -> ValueDecompositionPolicy:
    return ValueDecompositionPolicy.load(path, device=device)


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _constraint_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    joint_steps = len(rows)
    if joint_steps == 0:
        raise ValueError("Constraint metrics require at least one decision row")
    production_conflicts = sum(int(row["production_conflicts"]) for row in rows)
    technician_conflicts = sum(int(row["technician_conflicts"]) for row in rows)
    precedence_blocked = sum(
        int(row["precedence_blocked_operations"]) for row in rows
    )
    maintenance_wait = sum(float(row["maintenance_wait_delta"]) for row in rows)
    production_steps = sum(int(row["production_conflicts"]) > 0 for row in rows)
    technician_steps = sum(int(row["technician_conflicts"]) > 0 for row in rows)
    precedence_steps = sum(
        int(row["precedence_blocked_operations"]) > 0 for row in rows
    )
    maintenance_wait_steps = sum(
        float(row["maintenance_wait_delta"]) > 0 for row in rows
    )
    machine_technician_steps = sum(
        int(row["production_conflicts"]) > 0
        and int(row["technician_conflicts"]) > 0
        for row in rows
    )
    machine_precedence_steps = sum(
        int(row["production_conflicts"]) > 0
        and int(row["precedence_blocked_operations"]) > 0
        for row in rows
    )
    technician_precedence_steps = sum(
        int(row["technician_conflicts"]) > 0
        and int(row["precedence_blocked_operations"]) > 0
        for row in rows
    )
    strict_three_way_steps = sum(int(row["three_way"]) for row in rows)
    scale = 1_000 / joint_steps
    return {
        "joint_steps": joint_steps,
        "production_conflicts": production_conflicts,
        "technician_conflicts": technician_conflicts,
        "precedence_blocked_operations": precedence_blocked,
        "maintenance_wait_total": maintenance_wait,
        "production_conflicts_per_1000_joint_steps": production_conflicts * scale,
        "technician_conflicts_per_1000_joint_steps": technician_conflicts * scale,
        "precedence_blocked_operations_per_joint_step": precedence_blocked
        / joint_steps,
        "production_conflict_step_incidence": production_steps / joint_steps,
        "technician_conflict_step_incidence": technician_steps / joint_steps,
        "precedence_blocked_step_incidence": precedence_steps / joint_steps,
        "maintenance_wait_step_incidence": maintenance_wait_steps / joint_steps,
        "maintenance_wait_per_1000_joint_steps": maintenance_wait * scale,
        "machine_technician_steps_per_1000": machine_technician_steps * scale,
        "machine_precedence_steps_per_1000": machine_precedence_steps * scale,
        "technician_precedence_steps_per_1000": technician_precedence_steps
        * scale,
        "strict_three_way_steps": strict_three_way_steps,
        "strict_three_way_steps_per_1000": strict_three_way_steps * scale,
    }


def summarize(
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    coordination_rows: list[dict[str, Any]],
    *,
    checkpoint_targets: tuple[int, ...],
    evaluation_count: int,
    profile: str,
) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    selections: dict[str, int | None] = {}
    audits_passed = True
    for condition in SCREENING_CONDITIONS:
        points: list[dict[str, Any]] = []
        for checkpoint in checkpoint_targets:
            selected = [
                row
                for row in episode_rows
                if row["condition"] == condition
                and int(row["checkpoint_steps"]) == checkpoint
            ]
            selected_decisions = [
                row
                for row in decision_rows
                if row["condition"] == condition
                and int(row["checkpoint_steps"]) == checkpoint
            ]
            point = {
                "checkpoint_steps": checkpoint,
                "episodes": len(selected),
                "objective_mean": _mean(selected, "objective"),
                "makespan_mean": _mean(selected, "makespan"),
                "maintenance_wait_time_mean": _mean(
                    selected, "maintenance_wait_time"
                ),
                "failures_mean": _mean(selected, "failures"),
                "three_way_steps_mean": _mean(selected, "three_way_steps"),
                "three_way_episode_incidence": _mean(
                    selected, "episode_has_three_way"
                ),
                "constraint_metrics": _constraint_metrics(selected_decisions),
            }
            points.append(point)
        condition_coordination = [
            row for row in coordination_rows if row["condition"] == condition
        ]
        invalid = sum(
            int(row[metric])
            for row in condition_coordination
            for metric in (
                "invalid_executions",
                "duplicate_operation_executions",
                "duplicate_technician_executions",
            )
        )
        audits_passed &= invalid == 0
        selected_budget: int | None = None
        if profile == "full":
            best_objective = min(point["objective_mean"] for point in points)
            best_makespan = min(point["makespan_mean"] for point in points)
            best_failures = min(point["failures_mean"] for point in points)
            for candidate in (200_000, 300_000):
                point = next(
                    item for item in points if item["checkpoint_steps"] == candidate
                )
                if (
                    point["objective_mean"] <= (1 + OBJECTIVE_TOLERANCE) * best_objective
                    and point["makespan_mean"]
                    <= (1 + MAKESPAN_TOLERANCE) * best_makespan
                    and point["failures_mean"]
                    <= best_failures + FAILURE_TOLERANCE
                ):
                    selected_budget = candidate
                    break
        selections[condition] = selected_budget
        curves[condition] = {
            "points": points,
            "selected_replicated_run_budget": selected_budget,
        }
    expected = (
        len(SCREENING_CONDITIONS) * len(checkpoint_targets) * evaluation_count
    )
    all_selected = all(value is not None for value in selections.values())
    common_budget = max(selections.values()) if all_selected else None  # type: ignore[arg-type]
    checks = {
        "all_expected_checkpoint_evaluations_completed": len(episode_rows)
        == expected,
        "all_coordination_audits_passed": audits_passed,
        "screening_uses_one_development_training_seed_only": True,
        "sealed_test_panel_remains_closed": True,
    }
    if profile == "full":
        checks["all_algorithms_support_200k_or_300k"] = all_selected
    return {
        "purpose": "Engineering budget screening, not a performance claim.",
        "primary_endpoint": "mean objective learning curve by checkpoint",
        "secondary_endpoints": [
            "makespan",
            "maintenance_wait_time",
            "failures",
            "production_conflicts_per_1000_joint_steps",
            "technician_conflicts_per_1000_joint_steps",
            "precedence_blocked_operations_per_joint_step",
            "maintenance_wait_step_incidence",
            "pairwise_constraint_intersections",
            "three_way_steps",
            "three_way_episode_incidence",
        ],
        "locked_plateau_rule": {
            "eligible_budgets": [200_000, 300_000],
            "objective_within_fraction_of_best_100k_to_500k": OBJECTIVE_TOLERANCE,
            "makespan_within_fraction_of_best_100k_to_500k": MAKESPAN_TOLERANCE,
            "failures_within_mean_of_best": FAILURE_TOLERANCE,
            "selection": "earliest eligible per algorithm; common budget is their maximum",
        },
        "curves": curves,
        "selected_common_replicated_run_budget": common_budget,
        "gate": {
            "checks": checks,
            "passed": all(checks.values()),
            "note": "Use the sealed 62000:62100 panel only after replicated training.",
        },
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = _profile(args.profile)
    total_timesteps = args.total_timesteps or profile["total_timesteps"]
    train_seed = args.train_seed or profile["train_seed"]
    evaluation_seeds = (
        tuple(parse_seeds(args.evaluation_seeds))
        if args.evaluation_seeds
        else profile["evaluation_seeds"]
    )
    if set(evaluation_seeds) & set(SEALED_TEST_SEEDS):
        raise ValueError("The sealed future test panel cannot be opened")
    if total_timesteps % (5 * profile["n_envs"]):
        raise ValueError("total_timesteps must be divisible by 5 * n_envs")
    base_path = Path(args.config).resolve()
    raw_config = base_path.read_bytes()
    base = BenchmarkConfig.from_json(base_path)
    config = build_condition_config(base, topology="two_specialists", multiplier=2.0)
    if len(config.machines) != 6:
        raise AssertionError("Budget screening protocol is locked to six machines")
    scaled_payload = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    (output_dir / "scaled_config.json").write_text(scaled_payload)
    device = resolve_device(args.device)
    value_settings = ValueLearningSettings(
        total_timesteps=total_timesteps,
        replay_capacity=profile["replay_capacity"],
        learning_starts=min(profile["learning_starts"], total_timesteps // 2),
        batch_size=profile["value_batch_size"],
        train_frequency=4,
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_interval=max(32, total_timesteps // 100),
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_fraction=0.5,
        hidden_dim=128,
        mixer_hidden_dim=64,
        device=device,
        n_envs=profile["n_envs"],
    )
    checkpoint_targets = _checkpoint_targets(total_timesteps)
    manifest_path = output_dir / "marl_budget_screening_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "git_commit": _git_revision(),
        "base_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "scaled_config_sha256": hashlib.sha256(scaled_payload.encode()).hexdigest(),
        "capacity_condition": "two_specialists_x2_0",
        "machine_count": len(config.machines),
        "conditions": SCREENING_CONDITIONS,
        "train_seed": train_seed,
        "evaluation_seeds": evaluation_seeds,
        "reserved_future_test_seeds": SEALED_TEST_SEEDS,
        "future_test_panel_opened": False,
        "checkpoint_targets": checkpoint_targets,
        "value_settings": asdict(value_settings),
        "requested_device": args.device,
        "resolved_device": device,
        "completed_training_algorithms": [],
        "completed_evaluation_cells": [],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "torch")},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    coordination_rows: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}
    for condition in tqdm(
        SCREENING_CONDITIONS, desc="Budget-screen algorithms", unit="algorithm"
    ):
        cell = output_dir / condition
        _, elapsed = train_value_policy(
            config,
            value_settings,
            cell,
            algorithm="iql" if condition == IQL else "qmix",
            train_seed=train_seed,
            show_progress=True,
        )
        training_seconds[condition] = elapsed
        manifest["completed_training_algorithms"].append(condition)
        manifest["training_seconds"] = training_seconds
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for checkpoint in tqdm(
            checkpoint_targets,
            desc=f"Evaluate checkpoints {condition}",
            unit="checkpoint",
        ):
            policy = _load_policy(_checkpoint_path(cell, checkpoint), device)
            for seed in tqdm(
                evaluation_seeds,
                desc=f"{condition}/{checkpoint}",
                unit="episode",
                leave=False,
            ):
                episode, decisions, coordination = evaluate_episode(
                    policy,
                    config,
                    condition=condition,
                    train_seed=train_seed,
                    seed=seed,
                    device=device,
                )
                episode["checkpoint_steps"] = checkpoint
                coordination["checkpoint_steps"] = checkpoint
                for row in decisions:
                    row["checkpoint_steps"] = checkpoint
                episode_rows.append(episode)
                decision_rows.extend(decisions)
                coordination_rows.append(coordination)
            del policy
            manifest["completed_evaluation_cells"].append(
                f"{condition}:{checkpoint}"
            )
            _write_csv(
                episode_rows, output_dir / "marl_budget_screening_episodes.partial.csv"
            )
            _write_csv(
                decision_rows,
                output_dir / "marl_budget_screening_decisions.partial.csv",
            )
            _write_csv(
                coordination_rows,
                output_dir / "marl_budget_screening_coordination.partial.csv",
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
    summary = summarize(
        episode_rows,
        decision_rows,
        coordination_rows,
        checkpoint_targets=checkpoint_targets,
        evaluation_count=len(evaluation_seeds),
        profile=args.profile,
    )
    _write_csv(episode_rows, output_dir / "marl_budget_screening_episodes.csv")
    _write_csv(decision_rows, output_dir / "marl_budget_screening_decisions.csv")
    _write_csv(
        coordination_rows, output_dir / "marl_budget_screening_coordination.csv"
    )
    (output_dir / "marl_budget_screening_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        status="COMPLETED",
        completed_at_utc=datetime.now(UTC).isoformat(),
        episode_count=len(episode_rows),
        decision_count=len(decision_rows),
        training_seconds=training_seconds,
        gate=summary["gate"],
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brandimarte_mk01_ht_pdm.json")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--evaluation-seeds")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
