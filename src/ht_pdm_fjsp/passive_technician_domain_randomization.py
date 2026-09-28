"""Equal-compute screen of nominal versus domain-randomized RA-QMIX training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_leave_one_out import (
    COST_TOLERANCE,
    SEALED_TEST_SEEDS,
    _git_state,
    _runtime_metadata,
    _write_csv,
    _write_json,
    environment_cells,
    evaluate_diagnostic_episode,
)
from ht_pdm_fjsp.passive_technician_marl import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    COMPONENT_FLAGS,
    PassiveValueDecomposition,
    ValueTrainSettings,
    train_value_decomposition_step_checkpoints,
)


PROTOCOL_VERSION = "ra_qmix_domain_randomization_v1"
ALGORITHM = "tqmix"
REGIMES = ("nominal_only", "uniform_four_scenario")
FULL_TRAIN_SEEDS = tuple(range(83_800, 83_810))
FULL_EVALUATION_SEEDS = tuple(range(83_900, 84_000))
SMOKE_TRAIN_SEEDS = (83_710,)
SMOKE_EVALUATION_SEEDS = (83_720, 83_721, 83_722)
FULL_STEP_BUDGETS = (240_240, 360_360, 480_480)
SMOKE_STEP_BUDGETS = (132, 264, 396)
FROZEN_REFERENCES = {
    "in_distribution": 7.6,
    "early_failure": 102.496,
    "slow_service": 71.568,
    "combined_pressure": 150.25,
}
PRIOR_USED_SEEDS = frozenset(
    (*range(83_090, 83_105), *range(83_490, 83_703))
)
SUMMARY_METRICS = (
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


def profile_settings(
    profile: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], ValueTrainSettings]:
    if profile == "smoke":
        return (
            SMOKE_TRAIN_SEEDS,
            SMOKE_EVALUATION_SEEDS,
            SMOKE_STEP_BUDGETS,
            ValueTrainSettings(
                episodes=33,
                hidden_dim=16,
                mixer_hidden_dim=8,
                replay_capacity=512,
                batch_size=4,
                learning_starts=4,
                train_frequency=1,
                target_update_interval=12,
            ),
        )
    if profile == "full":
        return (
            FULL_TRAIN_SEEDS,
            FULL_EVALUATION_SEEDS,
            FULL_STEP_BUDGETS,
            ValueTrainSettings(episodes=40_040),
        )
    raise ValueError(profile)


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def training_schedule(
    regime: str,
    train_seed: int,
    maximum_steps: int,
    cells: dict[str, Any],
) -> tuple[str, ...]:
    if regime == "nominal_only":
        horizon = cells["in_distribution"].horizon
        if maximum_steps % horizon:
            raise ValueError("nominal step budget must end on an episode boundary")
        return ("in_distribution",) * (maximum_steps // horizon)
    if regime != "uniform_four_scenario":
        raise ValueError(regime)
    names = tuple(cells)
    block_steps = sum(cells[name].horizon for name in names)
    if maximum_steps % block_steps:
        raise ValueError("mixture step budget must end on a balanced block boundary")
    rng = np.random.default_rng(train_seed + 91_337_000)
    schedule: list[str] = []
    for _ in range(maximum_steps // block_steps):
        schedule.extend(str(name) for name in rng.permutation(names))
    return tuple(schedule)


def _schedule_hash(schedule: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(schedule).encode()).hexdigest()


def budget_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["training_regime"]),
            int(row["train_seed"]),
            str(row["scenario"]),
            int(row["budget"]),
        )
        groups.setdefault(key, []).append(row)
    result: list[dict[str, Any]] = []
    for (regime, train_seed, scenario, budget), group in sorted(groups.items()):
        item: dict[str, Any] = {
            "training_regime": regime,
            "algorithm": ALGORITHM,
            "train_seed": train_seed,
            "scenario": scenario,
            "step_budget": budget,
            "evaluation_count": len(group),
        }
        for metric in SUMMARY_METRICS:
            item[f"{metric}_mean"] = statistics.fmean(
                float(row[metric]) for row in group
            )
        result.append(item)
    return result


def robust_scores(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    means = {
        (
            str(row["training_regime"]),
            int(row["train_seed"]),
            str(row["scenario"]),
            int(row["step_budget"]),
        ): float(row["objective_mean"])
        for row in summary_rows
    }
    seeds = sorted({key[1] for key in means})
    budgets = sorted({key[3] for key in means})
    scenarios = tuple(FROZEN_REFERENCES)
    result: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[tuple[float, float]]] = {}
    for regime in REGIMES:
        for budget in budgets:
            for train_seed in seeds:
                regrets = {
                    scenario: means[regime, train_seed, scenario, budget]
                    / FROZEN_REFERENCES[scenario]
                    - 1.0
                    for scenario in scenarios
                }
                maximum = max(regrets.values())
                average = statistics.fmean(regrets.values())
                grouped.setdefault((regime, budget), []).append((maximum, average))
                result.append(
                    {
                        "row_type": "training_seed",
                        "training_regime": regime,
                        "algorithm": ALGORITHM,
                        "step_budget": budget,
                        "train_seed": train_seed,
                        **{
                            f"{scenario}_relative_regret": regrets[scenario]
                            for scenario in scenarios
                        },
                        "max_relative_regret": maximum,
                        "mean_relative_regret": average,
                        "max_regret_mean": "",
                        "max_regret_sd": "",
                        "mean_regret_mean": "",
                        "mean_regret_sd": "",
                    }
                )
    for (regime, budget), pairs in sorted(grouped.items()):
        maxima = [pair[0] for pair in pairs]
        averages = [pair[1] for pair in pairs]
        result.append(
            {
                "row_type": "across_training_seeds",
                "training_regime": regime,
                "algorithm": ALGORITHM,
                "step_budget": budget,
                "train_seed": "",
                **{
                    f"{scenario}_relative_regret": "" for scenario in scenarios
                },
                "max_relative_regret": "",
                "mean_relative_regret": "",
                "max_regret_mean": statistics.fmean(maxima),
                "max_regret_sd": (
                    statistics.stdev(maxima) if len(maxima) > 1 else 0.0
                ),
                "mean_regret_mean": statistics.fmean(averages),
                "mean_regret_sd": (
                    statistics.stdev(averages) if len(averages) > 1 else 0.0
                ),
            }
        )
    return result


def primary_results(
    score_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    primary_budget: int,
    screening_gate_applicable: bool,
) -> list[dict[str, Any]]:
    scores = {
        (str(row["training_regime"]), int(row["train_seed"])): float(
            row["max_relative_regret"]
        )
        for row in score_rows
        if row["row_type"] == "training_seed"
        and int(row["step_budget"]) == primary_budget
    }
    nominal = {
        (str(row["training_regime"]), int(row["train_seed"])): float(
            row["objective_mean"]
        )
        for row in summary_rows
        if row["scenario"] == "in_distribution"
        and int(row["step_budget"]) == primary_budget
    }
    seeds = sorted(seed for regime, seed in scores if regime == REGIMES[0])
    result: list[dict[str, Any]] = []
    deltas: list[float] = []
    for train_seed in seeds:
        control = scores[REGIMES[0], train_seed]
        treatment = scores[REGIMES[1], train_seed]
        delta = treatment - control
        deltas.append(delta)
        result.append(
            {
                "row_type": "training_seed",
                "train_seed": train_seed,
                "primary_step_budget": primary_budget,
                "control_max_regret": control,
                "treatment_max_regret": treatment,
                "delta_treatment_minus_control": delta,
                "delta_mean": "",
                "delta_sd": "",
                "treatment_better_count": "",
                "control_nominal_objective_mean": "",
                "treatment_nominal_objective_mean": "",
                "nominal_relative_degradation": "",
                "screening_gate_applicable": "",
                "screening_gate_passed": "",
            }
        )
    control_nominal = statistics.fmean(
        nominal[REGIMES[0], seed] for seed in seeds
    )
    treatment_nominal = statistics.fmean(
        nominal[REGIMES[1], seed] for seed in seeds
    )
    nominal_degradation = treatment_nominal / control_nominal - 1.0
    wins = sum(delta < 0 for delta in deltas)
    gate = (
        statistics.fmean(deltas) < 0
        and wins >= 7
        and nominal_degradation <= 0.05
    )
    result.append(
        {
            "row_type": "across_training_seeds",
            "train_seed": "",
            "primary_step_budget": primary_budget,
            "control_max_regret": "",
            "treatment_max_regret": "",
            "delta_treatment_minus_control": "",
            "delta_mean": statistics.fmean(deltas),
            "delta_sd": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "treatment_better_count": wins,
            "control_nominal_objective_mean": control_nominal,
            "treatment_nominal_objective_mean": treatment_nominal,
            "nominal_relative_degradation": nominal_degradation,
            "screening_gate_applicable": screening_gate_applicable,
            "screening_gate_passed": gate if screening_gate_applicable else "",
        }
    )
    return result


def run(args: argparse.Namespace) -> Path:
    train_seeds, evaluation_seeds, budgets, settings = profile_settings(args.profile)
    panels = (set(train_seeds), set(evaluation_seeds), set(SEALED_TEST_SEEDS))
    if any(
        panels[left] & panels[right]
        for left in range(len(panels))
        for right in range(left + 1, len(panels))
    ):
        raise ValueError("training, evaluation, and sealed seed panels must be disjoint")
    if (set(train_seeds) | set(evaluation_seeds)) & PRIOR_USED_SEEDS:
        raise ValueError("new seed panels overlap a prior RA-QMIX protocol")
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)

    cells = environment_cells()
    train_config = cells["in_distribution"]
    schedules = {
        (regime, seed): training_schedule(
            regime, seed, max(budgets), cells
        )
        for regime in REGIMES
        for seed in train_seeds
    }
    expected_training_episodes = sum(len(value) for value in schedules.values())
    expected_rows = (
        len(REGIMES)
        * len(train_seeds)
        * len(budgets)
        * len(cells)
        * len(evaluation_seeds)
    )
    expected_checkpoints = len(REGIMES) * len(train_seeds) * len(budgets)
    revision, dirty = _git_state()
    component_mapping = {
        "edge_utilities": COMPONENT_FLAGS[ALGORITHM][0],
        "queue_mixer": COMPONENT_FLAGS[ALGORITHM][1],
        "counterfactual_loss": COMPONENT_FLAGS[ALGORITHM][2],
        "lambda_cf": settings.lambda_cf,
    }
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "experiment": "ra_qmix_domain_randomization",
        "protocol_version": PROTOCOL_VERSION,
        "profile": args.profile,
        "hypothesis": "At equal environment-step compute, uniform four-scenario training lowers paired per-seed worst-scenario regret relative to nominal-only training.",
        "primary_endpoint": "paired per-training-seed difference in maximum frozen-reference relative regret at the final step checkpoint: uniform_four_scenario minus nominal_only",
        "screening_gate": "mean primary delta < 0, treatment better in at least 7 of 10 seeds, and aggregate nominal degradation <= 5%",
        "screening_gate_applicable": args.profile == "full",
        "stopping_rule": "Fixed environment-step budget; stop only for error, non-finite loss, missing checkpoint, or audit violation.",
        "algorithm": ALGORITHM,
        "training_regimes": list(REGIMES),
        "intervention": "training environment distribution only",
        "component_mapping": component_mapping,
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "step_budgets": list(budgets),
        "primary_step_budget": max(budgets),
        "frozen_scenario_references": FROZEN_REFERENCES,
        "environment_cells": {name: asdict(config) for name, config in cells.items()},
        "training_schedule": "nominal-only episodes or seed-specific randomized balanced four-scenario blocks",
        "value_settings": asdict(settings),
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device,
        "resolved_device": str(device),
        "torch_cpu_threads": torch.get_num_threads(),
        "git_revision": revision,
        "git_dirty": dirty,
        "runtime": _runtime_metadata(),
        "cost_reconciliation_tolerance": COST_TOLERANCE,
        "expected_counts": {
            "evaluation_rows": expected_rows,
            "checkpoints": expected_checkpoints,
            "training_trajectories": len(REGIMES) * len(train_seeds),
            "training_episodes": expected_training_episodes,
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
            "algorithm": ALGORITHM,
            "training_regimes": REGIMES,
            "component_mapping": component_mapping,
            "train_seeds": train_seeds,
            "evaluation_seeds": evaluation_seeds,
            "sealed_test_seeds": SEALED_TEST_SEEDS,
            "step_budgets": budgets,
            "settings": asdict(settings),
            "frozen_scenario_references": FROZEN_REFERENCES,
            "device": str(device),
        },
    )

    try:
        parameter_rows = []
        for regime in REGIMES:
            model = PassiveValueDecomposition(
                train_config,
                ALGORITHM,
                settings.hidden_dim,
                settings.mixer_hidden_dim,
            )
            parameter_rows.append(
                {
                    "training_regime": regime,
                    "algorithm": ALGORITHM,
                    **component_mapping,
                    **model.parameter_counts(),
                }
            )
        _write_csv(output / "parameter_counts.csv", parameter_rows)

        schedule_rows = []
        for regime in REGIMES:
            for train_seed in train_seeds:
                schedule = schedules[regime, train_seed]
                schedule_hash = _schedule_hash(schedule)
                for scenario in cells:
                    count = schedule.count(scenario)
                    schedule_rows.append(
                        {
                            "training_regime": regime,
                            "train_seed": train_seed,
                            "scenario": scenario,
                            "episode_count": count,
                            "environment_steps": count * cells[scenario].horizon,
                            "schedule_sha256": schedule_hash,
                        }
                    )
        _write_csv(output / "training_schedule.csv", schedule_rows)

        episode_rows: list[dict[str, Any]] = []
        progress_rows: list[dict[str, Any]] = []
        checkpoint_paths: list[str] = []
        final_steps: dict[tuple[str, int], int] = {}
        optimizer_updates: dict[tuple[str, int], int] = {}
        training_counts: dict[tuple[str, int, str], int] = {}
        losses_finite = True
        for regime in REGIMES:
            for train_seed in tqdm(
                train_seeds, desc=f"train seeds/{regime}", unit="seed"
            ):
                root = (
                    output
                    / regime
                    / f"train_seed_{train_seed}"
                    / "checkpoints"
                )
                checkpoints, progress = train_value_decomposition_step_checkpoints(
                    train_config,
                    ALGORITHM,
                    train_seed,
                    budgets,
                    root,
                    settings,
                    device,
                    cells,
                    schedules[regime, train_seed],
                )
                trajectory_rows = [
                    {"training_regime": regime, **row} for row in progress
                ]
                _append_csv(output / "training_progress.csv", trajectory_rows)
                _append_csv(output / "training_episodes.csv", trajectory_rows)
                final_steps[regime, train_seed] = int(progress[-1]["environment_steps"])
                optimizer_updates[regime, train_seed] = sum(
                    int(row["update_count"]) for row in progress
                )
                for scenario in cells:
                    training_counts[regime, train_seed, scenario] = sum(
                        row["training_scenario"] == scenario for row in progress
                    )
                losses_finite = losses_finite and all(
                    math.isfinite(float(row[name]))
                    for row in progress
                    for name in (
                        "td_loss",
                        "raw_cf_loss",
                        "weighted_cf_loss",
                        "total_loss",
                    )
                    if row.get(name) is not None
                )
                for budget in budgets:
                    checkpoint = checkpoints[budget]
                    checkpoint_paths.append(str(checkpoint.relative_to(output)))
                    model = PassiveValueDecomposition.load(
                        checkpoint, train_config, device
                    )
                    if model.algorithm != ALGORITHM:
                        raise ValueError(f"checkpoint algorithm mismatch: {checkpoint}")
                    for scenario, evaluation_config in cells.items():
                        for eval_seed in tqdm(
                            evaluation_seeds,
                            desc=(
                                f"evaluate {regime}/{train_seed}/{budget}/{scenario}"
                            ),
                            unit="episode",
                            leave=False,
                        ):
                            row = evaluate_diagnostic_episode(
                                evaluation_config,
                                model,
                                eval_seed,
                                train_seed,
                                scenario,
                                budget,
                                device,
                            )
                            episode_rows.append(
                                {"training_regime": regime, **row}
                            )
                        progress_row = {
                            "training_regime": regime,
                            "train_seed": train_seed,
                            "step_budget": budget,
                            "scenario": scenario,
                            "completed_evaluation_episodes": len(evaluation_seeds),
                        }
                        progress_rows.append(progress_row)
                        _append_csv(
                            output / "evaluation_progress.csv", [progress_row]
                        )
                        _write_csv(output / "episodes.partial.csv", episode_rows)

        _write_csv(output / "episodes.csv", episode_rows)
        coordination_rows = [
            {
                "training_regime": row["training_regime"],
                "algorithm": row["algorithm"],
                "train_seed": row["train_seed"],
                "scenario": row["scenario"],
                "step_budget": row["budget"],
                "eval_seed": row["eval_seed"],
                "invalid_requests": row["invalid_requests"],
                "collisions": row["collisions"],
                "cost_reconciliation_error": row["cost_reconciliation_error"],
                "cost_reconciled": (
                    abs(float(row["cost_reconciliation_error"])) <= COST_TOLERANCE
                ),
                "resource_semantics_valid": all(
                    0.0 <= float(row[f"technician_{index}_utilization"]) <= 1.0
                    for index in range(train_config.technicians)
                ),
            }
            for row in episode_rows
        ]
        _write_csv(output / "coordination.csv", coordination_rows)
        summary_rows = budget_summary(episode_rows)
        score_rows = robust_scores(summary_rows)
        primary = primary_results(
            score_rows,
            summary_rows,
            max(budgets),
            args.profile == "full",
        )
        _write_csv(output / "budget_summary.csv", summary_rows)
        _write_csv(output / "robust_scores.csv", score_rows)
        _write_csv(output / "primary_results.csv", primary)

        keys = [
            (
                row["training_regime"],
                row["train_seed"],
                row["scenario"],
                row["budget"],
                row["eval_seed"],
            )
            for row in episode_rows
        ]
        expected_uniform_count = len(schedules[REGIMES[1], train_seeds[0]]) // len(cells)
        architecture_rows = [
            {key: value for key, value in row.items() if key != "training_regime"}
            for row in parameter_rows
        ]
        audits = {
            "expected_evaluation_rows": expected_rows,
            "actual_evaluation_rows": len(episode_rows),
            "all_expected_rows_present": len(episode_rows) == expected_rows,
            "unique_episode_keys": len(keys) == len(set(keys)),
            "expected_checkpoints": expected_checkpoints,
            "actual_checkpoints": len(checkpoint_paths),
            "all_checkpoints_present": len(checkpoint_paths) == expected_checkpoints,
            "expected_training_episodes": expected_training_episodes,
            "actual_training_episodes": sum(training_counts.values()),
            "all_training_episodes_present": (
                sum(training_counts.values()) == expected_training_episodes
            ),
            "training_trajectory_count": len(optimizer_updates),
            "all_trajectories_updated": all(
                value > 0 for value in optimizer_updates.values()
            ),
            "exact_environment_step_budget": all(
                value == max(budgets) for value in final_steps.values()
            ),
            "uniform_training_balanced": all(
                training_counts[REGIMES[1], seed, scenario]
                == expected_uniform_count
                for seed in train_seeds
                for scenario in cells
            ),
            "architecture_and_loss_identical": architecture_rows[0]
            == architecture_rows[1],
            "losses_finite": losses_finite,
            "cost_reconciliation_passed": all(
                row["cost_reconciled"] for row in coordination_rows
            ),
            "resource_semantics_passed": all(
                row["resource_semantics_valid"] for row in coordination_rows
            ),
            "evaluation_progress_complete": len(progress_rows)
            == len(REGIMES) * len(train_seeds) * len(budgets) * len(cells),
            "seed_panels_disjoint": not any(
                panels[left] & panels[right]
                for left in range(len(panels))
                for right in range(left + 1, len(panels))
            ),
            "prior_seed_panels_disjoint": not (
                (set(train_seeds) | set(evaluation_seeds)) & PRIOR_USED_SEEDS
            ),
            "sealed_test_panel_closed": not any(
                int(row["eval_seed"]) in SEALED_TEST_SEEDS
                for row in episode_rows
            ),
        }
        required_audits = (
            "all_expected_rows_present",
            "unique_episode_keys",
            "all_checkpoints_present",
            "all_training_episodes_present",
            "all_trajectories_updated",
            "exact_environment_step_budget",
            "uniform_training_balanced",
            "architecture_and_loss_identical",
            "losses_finite",
            "cost_reconciliation_passed",
            "resource_semantics_passed",
            "evaluation_progress_complete",
            "seed_panels_disjoint",
            "prior_seed_panels_disjoint",
            "sealed_test_panel_closed",
        )
        if not all(audits[name] for name in required_audits):
            raise RuntimeError(f"experiment audit failed: {audits}")
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "primary_endpoint": manifest["primary_endpoint"],
            "frozen_scenario_references": FROZEN_REFERENCES,
            "primary_results": primary,
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
                    "training_episodes": sum(training_counts.values()),
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
