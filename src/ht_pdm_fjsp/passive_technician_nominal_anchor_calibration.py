"""Calibrate nominal-anchored transition mixtures for Full RA-QMIX."""

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

from ht_pdm_fjsp.passive_technician_domain_randomization import (
    FROZEN_REFERENCES,
    budget_summary,
)
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


PROTOCOL_VERSION = "ra_qmix_nominal_anchor_calibration_v1"
ALGORITHM = "tqmix"
CONTROL_REGIME = "nominal_100"
REGIME_SPECS = {
    "nominal_100": {
        "nominal_transition_share": 1.0,
        "nominal_episodes_per_block": 1,
        "stress_episodes_per_cell_per_block": 0,
    },
    "nominal_87_5": {
        "nominal_transition_share": 0.875,
        "nominal_episodes_per_block": 63,
        "stress_episodes_per_cell_per_block": 2,
    },
    "nominal_75": {
        "nominal_transition_share": 0.75,
        "nominal_episodes_per_block": 27,
        "stress_episodes_per_cell_per_block": 2,
    },
    "nominal_50": {
        "nominal_transition_share": 0.5,
        "nominal_episodes_per_block": 9,
        "stress_episodes_per_cell_per_block": 2,
    },
}
REGIMES = tuple(REGIME_SPECS)
FULL_TRAIN_SEEDS = tuple(range(84_000, 84_005))
FULL_EVALUATION_SEEDS = tuple(range(84_100, 84_150))
SMOKE_TRAIN_SEEDS = (84_200,)
SMOKE_EVALUATION_SEEDS = (84_210, 84_211, 84_212)
FULL_STEP_BUDGETS = (240_192, 360_288, 480_384)
SMOKE_STEP_BUDGETS = (864, 1_728, 2_592)
PRIOR_USED_SEEDS = frozenset((*range(1, 301), *range(83_090, 84_000)))
STRESS_SCENARIOS = ("early_failure", "slow_service", "combined_pressure")


def profile_settings(
    profile: str,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], ValueTrainSettings]:
    if profile == "smoke":
        return (
            SMOKE_TRAIN_SEEDS,
            SMOKE_EVALUATION_SEEDS,
            SMOKE_STEP_BUDGETS,
            ValueTrainSettings(
                episodes=216,
                hidden_dim=16,
                mixer_hidden_dim=8,
                replay_capacity=1_024,
                batch_size=4,
                learning_starts=4,
                train_frequency=1,
                target_update_interval=24,
            ),
        )
    if profile == "full":
        return (
            FULL_TRAIN_SEEDS,
            FULL_EVALUATION_SEEDS,
            FULL_STEP_BUDGETS,
            ValueTrainSettings(episodes=40_032),
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
    if regime not in REGIME_SPECS:
        raise ValueError(regime)
    spec = REGIME_SPECS[regime]
    if regime == CONTROL_REGIME:
        horizon = cells["in_distribution"].horizon
        if maximum_steps % horizon:
            raise ValueError("nominal step budget must end on an episode boundary")
        return ("in_distribution",) * (maximum_steps // horizon)

    nominal_count = int(spec["nominal_episodes_per_block"])
    stress_count = int(spec["stress_episodes_per_cell_per_block"])
    block = ["in_distribution"] * nominal_count
    for scenario in STRESS_SCENARIOS:
        block.extend([scenario] * stress_count)
    block_steps = sum(cells[name].horizon for name in block)
    if maximum_steps % block_steps:
        raise ValueError(f"step budget does not close {regime} blocks")
    regime_index = REGIMES.index(regime)
    rng = np.random.default_rng(train_seed + 92_113_000 + regime_index * 10_000)
    schedule: list[str] = []
    for _ in range(maximum_steps // block_steps):
        schedule.extend(str(name) for name in rng.permutation(block))
    return tuple(schedule)


def _schedule_hash(schedule: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(schedule).encode()).hexdigest()


def regret_scores(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    grouped: dict[tuple[str, int], list[tuple[float, float, float]]] = {}
    for regime in REGIMES:
        for budget in budgets:
            for train_seed in seeds:
                regrets = {
                    scenario: means[regime, train_seed, scenario, budget]
                    / FROZEN_REFERENCES[scenario]
                    - 1.0
                    for scenario in scenarios
                }
                stress_max = max(regrets[name] for name in STRESS_SCENARIOS)
                overall_max = max(regrets.values())
                average = statistics.fmean(regrets.values())
                grouped.setdefault((regime, budget), []).append(
                    (stress_max, overall_max, average)
                )
                result.append(
                    {
                        "row_type": "training_seed",
                        "training_regime": regime,
                        "nominal_transition_share": REGIME_SPECS[regime][
                            "nominal_transition_share"
                        ],
                        "step_budget": budget,
                        "train_seed": train_seed,
                        **{
                            f"{scenario}_relative_regret": regrets[scenario]
                            for scenario in scenarios
                        },
                        "stress_max_relative_regret": stress_max,
                        "overall_max_relative_regret": overall_max,
                        "mean_relative_regret": average,
                        "stress_max_regret_mean": "",
                        "overall_max_regret_mean": "",
                        "mean_regret_mean": "",
                    }
                )
    for (regime, budget), triples in sorted(grouped.items()):
        result.append(
            {
                "row_type": "across_training_seeds",
                "training_regime": regime,
                "nominal_transition_share": REGIME_SPECS[regime][
                    "nominal_transition_share"
                ],
                "step_budget": budget,
                "train_seed": "",
                **{
                    f"{scenario}_relative_regret": "" for scenario in scenarios
                },
                "stress_max_relative_regret": "",
                "overall_max_relative_regret": "",
                "mean_relative_regret": "",
                "stress_max_regret_mean": statistics.fmean(
                    value[0] for value in triples
                ),
                "overall_max_regret_mean": statistics.fmean(
                    value[1] for value in triples
                ),
                "mean_regret_mean": statistics.fmean(value[2] for value in triples),
            }
        )
    return result


def primary_results(
    score_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    primary_budget: int,
    gate_applicable: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    scores = {
        (str(row["training_regime"]), int(row["train_seed"])): float(
            row["stress_max_relative_regret"]
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
    seeds = sorted(seed for regime, seed in scores if regime == CONTROL_REGIME)
    result: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for regime in REGIMES:
        if regime == CONTROL_REGIME:
            continue
        stress_deltas: list[float] = []
        nominal_degradations: list[float] = []
        candidate_scores: list[float] = []
        for train_seed in seeds:
            control_score = scores[CONTROL_REGIME, train_seed]
            candidate_score = scores[regime, train_seed]
            stress_delta = candidate_score - control_score
            nominal_degradation = (
                nominal[regime, train_seed] / nominal[CONTROL_REGIME, train_seed]
                - 1.0
            )
            stress_deltas.append(stress_delta)
            nominal_degradations.append(nominal_degradation)
            candidate_scores.append(candidate_score)
            result.append(
                {
                    "row_type": "training_seed",
                    "candidate_regime": regime,
                    "nominal_transition_share": REGIME_SPECS[regime][
                        "nominal_transition_share"
                    ],
                    "train_seed": train_seed,
                    "primary_step_budget": primary_budget,
                    "control_stress_max_regret": control_score,
                    "candidate_stress_max_regret": candidate_score,
                    "stress_delta_candidate_minus_control": stress_delta,
                    "nominal_relative_degradation": nominal_degradation,
                    "stress_delta_mean": "",
                    "stress_delta_sd": "",
                    "stress_better_count": "",
                    "nominal_degradation_mean": "",
                    "nominal_degradation_sd": "",
                    "nominal_within_10pct_count": "",
                    "gate_applicable": "",
                    "feasible": "",
                    "selected": "",
                }
            )
        stress_mean = statistics.fmean(stress_deltas)
        nominal_mean = statistics.fmean(nominal_degradations)
        stress_better = sum(value < 0 for value in stress_deltas)
        nominal_within = sum(value <= 0.10 for value in nominal_degradations)
        feasible = (
            nominal_mean <= 0.05
            and nominal_within >= 4
            and stress_mean < 0
            and stress_better >= 4
        )
        candidates.append(
            {
                "candidate_regime": regime,
                "nominal_transition_share": REGIME_SPECS[regime][
                    "nominal_transition_share"
                ],
                "stress_score_mean": statistics.fmean(candidate_scores),
                "stress_delta_mean": stress_mean,
                "nominal_degradation_mean": nominal_mean,
                "feasible": feasible,
            }
        )
        result.append(
            {
                "row_type": "across_training_seeds",
                "candidate_regime": regime,
                "nominal_transition_share": REGIME_SPECS[regime][
                    "nominal_transition_share"
                ],
                "train_seed": "",
                "primary_step_budget": primary_budget,
                "control_stress_max_regret": "",
                "candidate_stress_max_regret": "",
                "stress_delta_candidate_minus_control": "",
                "nominal_relative_degradation": "",
                "stress_delta_mean": stress_mean,
                "stress_delta_sd": (
                    statistics.stdev(stress_deltas)
                    if len(stress_deltas) > 1
                    else 0.0
                ),
                "stress_better_count": stress_better,
                "nominal_degradation_mean": nominal_mean,
                "nominal_degradation_sd": (
                    statistics.stdev(nominal_degradations)
                    if len(nominal_degradations) > 1
                    else 0.0
                ),
                "nominal_within_10pct_count": nominal_within,
                "gate_applicable": gate_applicable,
                "feasible": feasible if gate_applicable else "",
                "selected": False if gate_applicable else "",
            }
        )
    feasible_candidates = [item for item in candidates if item["feasible"]]
    selected = (
        min(
            feasible_candidates,
            key=lambda item: (
                item["stress_score_mean"],
                item["nominal_degradation_mean"],
                -item["nominal_transition_share"],
                item["candidate_regime"],
            ),
        )
        if gate_applicable and feasible_candidates
        else None
    )
    selected_regime = selected["candidate_regime"] if selected else None
    for row in result:
        if row["row_type"] == "across_training_seeds" and gate_applicable:
            row["selected"] = row["candidate_regime"] == selected_regime
    selection = {
        "gate_applicable": gate_applicable,
        "calibration_passed": selected is not None,
        "selected_regime": selected_regime,
        "selection_rule": "among feasible candidates, minimize (mean stress-max regret, mean nominal degradation, negative nominal share, regime ID)",
        "candidate_summaries": candidates,
    }
    return result, selection


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
        (regime, seed): training_schedule(regime, seed, max(budgets), cells)
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
        "experiment": "ra_qmix_nominal_anchor_calibration",
        "protocol_version": PROTOCOL_VERSION,
        "profile": args.profile,
        "hypothesis": "At least one nominal-anchored transition mixture satisfies the nominal constraint and improves paired stress-max regret versus nominal-only training.",
        "primary_endpoint": "paired candidate-minus-control stress-max relative regret at the final step checkpoint, subject to nominal degradation constraints",
        "calibration_rule": "nominal mean <= 5%, nominal <= 10% in at least 4/5 seeds, stress delta mean < 0, and stress better in at least 4/5 seeds",
        "calibration_applicable": args.profile == "full",
        "stopping_rule": "Fixed environment-step budget; stop only for error, non-finite loss, missing checkpoint, or audit violation.",
        "algorithm": ALGORITHM,
        "control_regime": CONTROL_REGIME,
        "regime_specs": REGIME_SPECS,
        "intervention": "training transition share only",
        "component_mapping": component_mapping,
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "step_budgets": list(budgets),
        "primary_step_budget": max(budgets),
        "frozen_scenario_references": FROZEN_REFERENCES,
        "stress_scenarios": list(STRESS_SCENARIOS),
        "environment_cells": {name: asdict(config) for name, config in cells.items()},
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
            "regime_specs": REGIME_SPECS,
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
        mixture_rows = [
            {
                "training_regime": regime,
                **spec,
                "stress_transition_share_per_cell": (
                    (1.0 - float(spec["nominal_transition_share"])) / 3.0
                ),
            }
            for regime, spec in REGIME_SPECS.items()
        ]
        _write_csv(output / "mixture_registry.csv", mixture_rows)
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
                total_steps = sum(cells[name].horizon for name in schedule)
                for scenario in cells:
                    count = schedule.count(scenario)
                    scenario_steps = count * cells[scenario].horizon
                    schedule_rows.append(
                        {
                            "training_regime": regime,
                            "train_seed": train_seed,
                            "scenario": scenario,
                            "episode_count": count,
                            "environment_steps": scenario_steps,
                            "transition_share": scenario_steps / total_steps,
                            "schedule_sha256": schedule_hash,
                        }
                    )
        _write_csv(output / "training_schedule.csv", schedule_rows)

        evaluation_rows: list[dict[str, Any]] = []
        evaluation_progress: list[dict[str, Any]] = []
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
                            evaluation_rows.append(
                                {"training_regime": regime, **row}
                            )
                        progress_row = {
                            "training_regime": regime,
                            "train_seed": train_seed,
                            "step_budget": budget,
                            "scenario": scenario,
                            "completed_evaluation_episodes": len(evaluation_seeds),
                        }
                        evaluation_progress.append(progress_row)
                        _append_csv(
                            output / "evaluation_progress.csv", [progress_row]
                        )
                        _write_csv(output / "episodes.partial.csv", evaluation_rows)

        _write_csv(output / "episodes.csv", evaluation_rows)
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
            for row in evaluation_rows
        ]
        _write_csv(output / "coordination.csv", coordination_rows)
        summary_rows = budget_summary(evaluation_rows)
        score_rows = regret_scores(summary_rows)
        primary, selection = primary_results(
            score_rows,
            summary_rows,
            max(budgets),
            args.profile == "full",
        )
        _write_csv(output / "budget_summary.csv", summary_rows)
        _write_csv(output / "regret_scores.csv", score_rows)
        _write_csv(output / "primary_results.csv", primary)
        _write_json(output / "selection_result.json", selection)

        keys = [
            (
                row["training_regime"],
                row["train_seed"],
                row["scenario"],
                row["budget"],
                row["eval_seed"],
            )
            for row in evaluation_rows
        ]
        architecture_rows = [
            {key: value for key, value in row.items() if key != "training_regime"}
            for row in parameter_rows
        ]
        exact_schedule = all(
            training_counts[regime, seed, scenario]
            == schedules[regime, seed].count(scenario)
            for regime in REGIMES
            for seed in train_seeds
            for scenario in cells
        )
        exact_shares = all(
            math.isclose(
                sum(
                    training_counts[regime, seed, scenario]
                    * cells[scenario].horizon
                    for scenario in cells
                    if scenario == "in_distribution"
                )
                / max(budgets),
                float(REGIME_SPECS[regime]["nominal_transition_share"]),
                abs_tol=1e-12,
            )
            for regime in REGIMES
            for seed in train_seeds
        )
        audits = {
            "expected_evaluation_rows": expected_rows,
            "actual_evaluation_rows": len(evaluation_rows),
            "all_expected_rows_present": len(evaluation_rows) == expected_rows,
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
            "exact_training_schedule": exact_schedule,
            "exact_nominal_transition_shares": exact_shares,
            "architecture_and_loss_identical": all(
                row == architecture_rows[0] for row in architecture_rows[1:]
            ),
            "paired_training_seeds": set(final_steps)
            == {(regime, seed) for regime in REGIMES for seed in train_seeds},
            "losses_finite": losses_finite,
            "cost_reconciliation_passed": all(
                row["cost_reconciled"] for row in coordination_rows
            ),
            "resource_semantics_passed": all(
                row["resource_semantics_valid"] for row in coordination_rows
            ),
            "evaluation_progress_complete": len(evaluation_progress)
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
                for row in evaluation_rows
            ),
        }
        required_audits = (
            "all_expected_rows_present",
            "unique_episode_keys",
            "all_checkpoints_present",
            "all_training_episodes_present",
            "all_trajectories_updated",
            "exact_environment_step_budget",
            "exact_training_schedule",
            "exact_nominal_transition_shares",
            "architecture_and_loss_identical",
            "paired_training_seeds",
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
            "selection_result": selection,
            "primary_results": primary,
            "audits": audits,
        }
        _write_json(output / "summary.json", summary)
        manifest.update(
            {
                "status": "COMPLETED",
                "finished_at": datetime.now(UTC).isoformat(),
                "actual_counts": {
                    "evaluation_rows": len(evaluation_rows),
                    "checkpoints": len(checkpoint_paths),
                    "training_trajectories": len(optimizer_updates),
                    "training_episodes": sum(training_counts.values()),
                },
                "selection_result": selection,
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
