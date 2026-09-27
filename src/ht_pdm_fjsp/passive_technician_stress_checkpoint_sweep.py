"""Locked stress-generalization checkpoint sweep for RA-QMIX."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_leave_one_out import (
    COST_TOLERANCE,
    SEALED_TEST_SEEDS,
    _budget_summary,
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
    train_value_decomposition_checkpoints,
)


PROTOCOL_VERSION = "ra_qmix_stress_checkpoint_sweep_v1"
ALGORITHMS = ("qmix", "tqmix_no_queue", "tqmix_no_cf", "tqmix")
REPORTING_NAMES = {
    "qmix": "Standard QMIX",
    "tqmix_no_queue": "RA-QMIX - queue",
    "tqmix_no_cf": "RA-QMIX - CF",
    "tqmix": "Full RA-QMIX",
}
FULL_BUDGETS = (5_000, 10_000, 20_000, 30_000, 40_000, 50_000)
SMOKE_BUDGETS = (2, 4, 6, 8, 10, 12)
PRIMARY_SCENARIO = "combined_pressure"
PRIMARY_BASELINE_BUDGET = 20_000
PRIMARY_FINAL_BUDGET = 50_000


def profile_settings(
    profile: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], ValueTrainSettings]:
    if profile == "smoke":
        return (
            (83_090,),
            (83_490, 83_491, 83_492),
            SMOKE_BUDGETS,
            ValueTrainSettings(
                episodes=max(SMOKE_BUDGETS),
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
            (83_100, 83_101, 83_102, 83_103, 83_104),
            tuple(range(83_500, 83_600)),
            FULL_BUDGETS,
            ValueTrainSettings(episodes=max(FULL_BUDGETS)),
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


def checkpoint_changes(
    summary_rows: list[dict[str, Any]], baseline_budget: int, final_budget: int
) -> list[dict[str, Any]]:
    means = {
        (
            str(row["algorithm"]),
            int(row["train_seed"]),
            str(row["scenario"]),
            int(row["budget"]),
        ): float(row["objective_mean"])
        for row in summary_rows
    }
    cells = sorted({(key[0], key[1], key[2]) for key in means})
    result: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    for algorithm, train_seed, scenario in cells:
        baseline = means[algorithm, train_seed, scenario, baseline_budget]
        final = means[algorithm, train_seed, scenario, final_budget]
        change = final - baseline
        grouped.setdefault((algorithm, scenario), []).append(change)
        result.append(
            {
                "row_type": "training_seed",
                "algorithm": algorithm,
                "reporting_name": REPORTING_NAMES[algorithm],
                "scenario": scenario,
                "train_seed": train_seed,
                "baseline_budget": baseline_budget,
                "final_budget": final_budget,
                "baseline_objective": baseline,
                "final_objective": final,
                "objective_change": change,
                "change_mean": "",
                "change_sd": "",
                "change_min": "",
                "change_max": "",
                "positive_seed_count": "",
            }
        )
    for (algorithm, scenario), values in sorted(grouped.items()):
        result.append(
            {
                "row_type": "across_training_seeds",
                "algorithm": algorithm,
                "reporting_name": REPORTING_NAMES[algorithm],
                "scenario": scenario,
                "train_seed": "",
                "baseline_budget": baseline_budget,
                "final_budget": final_budget,
                "baseline_objective": "",
                "final_objective": "",
                "objective_change": "",
                "change_mean": statistics.fmean(values),
                "change_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                "change_min": min(values),
                "change_max": max(values),
                "positive_seed_count": sum(value > 0 for value in values),
            }
        )
    return result


def component_interactions(change_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    per_seed = {
        (str(row["algorithm"]), int(row["train_seed"]), str(row["scenario"])): float(
            row["objective_change"]
        )
        for row in change_rows
        if row["row_type"] == "training_seed"
    }
    contrasts = (
        ("queue_full_minus_no_queue", "tqmix", "tqmix_no_queue", True),
        ("cf_full_minus_no_cf", "tqmix", "tqmix_no_cf", True),
        ("full_minus_qmix", "tqmix", "qmix", False),
    )
    scenarios = sorted({key[2] for key in per_seed})
    train_seeds = sorted({key[1] for key in per_seed})
    result: list[dict[str, Any]] = []
    for contrast, left, right, component_contrast in contrasts:
        for scenario in scenarios:
            values: list[float] = []
            for train_seed in train_seeds:
                value = per_seed[left, train_seed, scenario] - per_seed[
                    right, train_seed, scenario
                ]
                values.append(value)
                result.append(
                    {
                        "row_type": "training_seed",
                        "contrast": contrast,
                        "left_algorithm": left,
                        "right_algorithm": right,
                        "scenario": scenario,
                        "train_seed": train_seed,
                        "interaction": value,
                        "interaction_mean": "",
                        "interaction_sd": "",
                        "interaction_min": "",
                        "interaction_max": "",
                        "positive_seed_count": "",
                        "directional_replication": "",
                        "component_contrast": component_contrast,
                    }
                )
            positive_count = sum(value > 0 for value in values)
            result.append(
                {
                    "row_type": "across_training_seeds",
                    "contrast": contrast,
                    "left_algorithm": left,
                    "right_algorithm": right,
                    "scenario": scenario,
                    "train_seed": "",
                    "interaction": "",
                    "interaction_mean": statistics.fmean(values),
                    "interaction_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "interaction_min": min(values),
                    "interaction_max": max(values),
                    "positive_seed_count": positive_count,
                    "directional_replication": (
                        statistics.fmean(values) > 0 and positive_count >= 4
                    ),
                    "component_contrast": component_contrast,
                }
            )
    return result


def primary_results(interaction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    primary_contrasts = {"queue_full_minus_no_queue", "cf_full_minus_no_cf"}
    return [
        row
        for row in interaction_rows
        if row["row_type"] == "across_training_seeds"
        and row["scenario"] == PRIMARY_SCENARIO
        and row["contrast"] in primary_contrasts
    ]


def run(args: argparse.Namespace) -> Path:
    train_seeds, evaluation_seeds, budgets, settings = profile_settings(args.profile)
    panels = (set(train_seeds), set(evaluation_seeds), set(SEALED_TEST_SEEDS))
    if panels[0] & panels[1] or panels[0] & panels[2] or panels[1] & panels[2]:
        raise ValueError("training, evaluation, and sealed seed panels must be disjoint")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
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
    expected_rows = (
        len(ALGORITHMS)
        * len(train_seeds)
        * len(budgets)
        * len(cells)
        * len(evaluation_seeds)
    )
    expected_checkpoints = len(ALGORITHMS) * len(train_seeds) * len(budgets)
    baseline_budget = 6 if args.profile == "smoke" else PRIMARY_BASELINE_BUDGET
    final_budget = 12 if args.profile == "smoke" else PRIMARY_FINAL_BUDGET
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "experiment": "ra_qmix_stress_checkpoint_sweep",
        "protocol_version": PROTOCOL_VERSION,
        "profile": args.profile,
        "hypotheses": {
            "queue": "Full RA-QMIX degrades more from 20k to 50k under combined pressure than RA-QMIX without the queue mixer.",
            "counterfactual": "Full RA-QMIX degrades more from 20k to 50k under combined pressure than RA-QMIX without the CF loss.",
        },
        "primary_endpoint": "paired interaction in combined_pressure objective change from 20000 to 50000 episodes",
        "primary_contrasts": [
            "queue_full_minus_no_queue",
            "cf_full_minus_no_cf",
        ],
        "directional_replication_rule": "interaction mean > 0 and positive in at least 4 of 5 training seeds",
        "exploratory_scope": "checkpoint trajectory at 5k, 10k, 20k, 30k, 40k, and 50k",
        "stopping_rule": "Fixed maximum training budget; stop only for error, non-finite loss, or audit violation.",
        "algorithms": list(ALGORITHMS),
        "reporting_names": REPORTING_NAMES,
        "component_mapping": component_mapping,
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "checkpoint_budgets": list(budgets),
        "change_baseline_budget": baseline_budget,
        "change_final_budget": final_budget,
        "training_scenario": "in_distribution",
        "environment_cells": {name: asdict(config) for name, config in cells.items()},
        "recovery_semantics": "age and failure reset at maintenance start; machine remains unavailable until service completion; downtime uses pre-service state",
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
            for train_seed in tqdm(
                train_seeds, desc=f"train seeds/{algorithm}", unit="seed"
            ):
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
                "cost_reconciled": abs(float(row["cost_reconciliation_error"]))
                <= COST_TOLERANCE,
                "resource_semantics_valid": all(
                    0.0 <= float(row[f"technician_{index}_utilization"]) <= 1.0
                    for index in range(train_config.technicians)
                ),
            }
            for row in episode_rows
        ]
        _write_csv(output / "coordination.csv", coordination_rows)
        budget_summary = _budget_summary(episode_rows)
        changes = checkpoint_changes(budget_summary, baseline_budget, final_budget)
        interactions = component_interactions(changes)
        primary = primary_results(interactions)
        _write_csv(output / "budget_summary.csv", budget_summary)
        _write_csv(output / "checkpoint_changes.csv", changes)
        _write_csv(output / "component_interactions.csv", interactions)
        _write_csv(output / "primary_results.csv", primary)

        keys = [
            (
                row["algorithm"],
                row["train_seed"],
                row["scenario"],
                row["budget"],
                row["eval_seed"],
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
            "all_trajectories_updated": all(
                value > 0 for value in optimizer_updates.values()
            ),
            "losses_finite": losses_finite,
            "cost_reconciliation_passed": all(
                row["cost_reconciled"] for row in coordination_rows
            ),
            "resource_semantics_passed": all(
                row["resource_semantics_valid"] for row in coordination_rows
            ),
            "sealed_test_panel_closed": not any(
                int(row["eval_seed"]) in SEALED_TEST_SEEDS for row in episode_rows
            ),
            "seed_panels_disjoint": not any(
                panels[left] & panels[right]
                for left, right in ((0, 1), (0, 2), (1, 2))
            ),
        }
        required_audits = (
            "all_expected_rows_present",
            "unique_episode_keys",
            "all_checkpoints_present",
            "all_trajectories_updated",
            "losses_finite",
            "cost_reconciliation_passed",
            "resource_semantics_passed",
            "sealed_test_panel_closed",
            "seed_panels_disjoint",
        )
        if not all(audits[key] for key in required_audits):
            raise RuntimeError(f"experiment audit failed: {audits}")
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "primary_endpoint": manifest["primary_endpoint"],
            "change_definition": "mean objective at final budget minus baseline budget; positive means degradation",
            "interaction_definition": "Full change minus comparator change; positive means Full degraded more",
            "directional_replication_rule": manifest["directional_replication_rule"],
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
