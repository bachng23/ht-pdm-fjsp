"""Confirm the selected RA-QMIX curriculum-replay recipe on ten fresh seeds."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_anchor_preservation import (
    ALGORITHM,
    CONTROL_REGIME,
    STRESS_SCENARIOS,
    _append_csv,
    _checkpoint_states_equal,
    _normalized_progress_row,
    _schedule_hash,
    training_schedule,
)
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
    AnchorAdaptationSettings,
    PassiveValueDecomposition,
    ValueTrainSettings,
    train_value_decomposition_anchor_checkpoints,
    train_value_decomposition_step_checkpoints,
)


PROTOCOL_VERSION = "ra_qmix_curriculum_confirmation_v1"
SOURCE_DEVELOPMENT_PROTOCOL = "ra_qmix_anchor_preservation_v1"
SOURCE_DEVELOPMENT_COMMIT = "63210742052ed08f6bd6dbedf4ce55ebf0fb99ed"
CANDIDATE_REGIME = "curriculum_replay"
REGIMES = (CONTROL_REGIME, CANDIDATE_REGIME)
REGIME_SPECS = {
    CONTROL_REGIME: {
        "training_design": "nominal_only",
        "overall_nominal_transition_share": 1.0,
        "phase_two_nominal_replay_share": 1.0,
        "anchor_lambda": 0.0,
    },
    CANDIDATE_REGIME: {
        "training_design": "nominal_pretrain_then_stratified_adaptation",
        "overall_nominal_transition_share": 0.9375,
        "phase_two_nominal_replay_share": 0.875,
        "anchor_lambda": 0.0,
    },
}
FULL_TRAIN_SEEDS = tuple(range(84_600, 84_610))
FULL_EVALUATION_SEEDS = tuple(range(84_700, 84_750))
SMOKE_TRAIN_SEEDS = (84_800,)
SMOKE_EVALUATION_SEEDS = (84_810, 84_811, 84_812)
FULL_STEP_BUDGETS = (240_192, 360_288, 480_384)
SMOKE_STEP_BUDGETS = (1_728, 2_592, 3_456)
PRIOR_USED_SEEDS = frozenset((*range(1, 301), *range(83_090, 84_513)))
MINIMUM_CONFIRMATION_WINS = 8
MINIMUM_NOMINAL_WITHIN_COUNT = 8
NOMINAL_MEAN_DEGRADATION_LIMIT = 0.05
NOMINAL_PER_SEED_DEGRADATION_LIMIT = 0.10


def profile_settings(
    profile: str,
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    ValueTrainSettings,
    int,
]:
    if profile == "smoke":
        return (
            SMOKE_TRAIN_SEEDS,
            SMOKE_EVALUATION_SEEDS,
            SMOKE_STEP_BUDGETS,
            1_728,
            ValueTrainSettings(
                episodes=288,
                hidden_dim=16,
                mixer_hidden_dim=8,
                replay_capacity=1_024,
                batch_size=8,
                learning_starts=8,
                train_frequency=2,
                target_update_interval=24,
            ),
            7,
        )
    if profile == "full":
        return (
            FULL_TRAIN_SEEDS,
            FULL_EVALUATION_SEEDS,
            FULL_STEP_BUDGETS,
            240_192,
            ValueTrainSettings(episodes=40_032),
            112,
        )
    raise ValueError(profile)


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
                "step_budget": budget,
                "train_seed": "",
                **{
                    f"{scenario}_relative_regret": "" for scenario in scenarios
                },
                "stress_max_relative_regret": "",
                "overall_max_relative_regret": "",
                "mean_relative_regret": "",
                "stress_max_regret_mean": statistics.fmean(x[0] for x in triples),
                "overall_max_regret_mean": statistics.fmean(x[1] for x in triples),
                "mean_regret_mean": statistics.fmean(x[2] for x in triples),
            }
        )
    return result


def _paired_t_interval(values: list[float]) -> tuple[float | str, float | str]:
    if len(values) != 10:
        return "", ""
    mean = statistics.fmean(values)
    margin = 2.2621571628540993 * statistics.stdev(values) / math.sqrt(10)
    return mean - margin, mean + margin


def confirmation_results(
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
    stress_deltas = [
        scores[CANDIDATE_REGIME, seed] - scores[CONTROL_REGIME, seed]
        for seed in seeds
    ]
    nominal_degradations = [
        nominal[CANDIDATE_REGIME, seed] / nominal[CONTROL_REGIME, seed] - 1.0
        for seed in seeds
    ]
    stress_mean = statistics.fmean(stress_deltas)
    nominal_mean = statistics.fmean(nominal_degradations)
    stress_better = sum(value < 0 for value in stress_deltas)
    nominal_within = sum(
        value <= NOMINAL_PER_SEED_DEGRADATION_LIMIT
        for value in nominal_degradations
    )
    feasible = (
        nominal_mean <= NOMINAL_MEAN_DEGRADATION_LIMIT
        and nominal_within >= MINIMUM_NOMINAL_WITHIN_COUNT
        and stress_mean < 0
        and stress_better >= MINIMUM_CONFIRMATION_WINS
    )
    stress_ci = _paired_t_interval(stress_deltas)
    nominal_ci = _paired_t_interval(nominal_degradations)
    rows: list[dict[str, Any]] = []
    for seed, stress_delta, nominal_degradation in zip(
        seeds, stress_deltas, nominal_degradations, strict=True
    ):
        rows.append(
            {
                "row_type": "training_seed",
                "candidate_regime": CANDIDATE_REGIME,
                "train_seed": seed,
                "primary_step_budget": primary_budget,
                "control_stress_max_regret": scores[CONTROL_REGIME, seed],
                "candidate_stress_max_regret": scores[CANDIDATE_REGIME, seed],
                "stress_delta_candidate_minus_control": stress_delta,
                "nominal_relative_degradation": nominal_degradation,
                "stress_delta_mean": "",
                "stress_delta_sd": "",
                "stress_delta_ci95_low": "",
                "stress_delta_ci95_high": "",
                "stress_better_count": "",
                "nominal_degradation_mean": "",
                "nominal_degradation_sd": "",
                "nominal_degradation_ci95_low": "",
                "nominal_degradation_ci95_high": "",
                "nominal_within_10pct_count": "",
                "gate_applicable": "",
                "confirmation_passed": "",
            }
        )
    rows.append(
        {
            "row_type": "across_training_seeds",
            "candidate_regime": CANDIDATE_REGIME,
            "train_seed": "",
            "primary_step_budget": primary_budget,
            "control_stress_max_regret": "",
            "candidate_stress_max_regret": "",
            "stress_delta_candidate_minus_control": "",
            "nominal_relative_degradation": "",
            "stress_delta_mean": stress_mean,
            "stress_delta_sd": (
                statistics.stdev(stress_deltas) if len(stress_deltas) > 1 else 0.0
            ),
            "stress_delta_ci95_low": stress_ci[0],
            "stress_delta_ci95_high": stress_ci[1],
            "stress_better_count": stress_better,
            "nominal_degradation_mean": nominal_mean,
            "nominal_degradation_sd": (
                statistics.stdev(nominal_degradations)
                if len(nominal_degradations) > 1
                else 0.0
            ),
            "nominal_degradation_ci95_low": nominal_ci[0],
            "nominal_degradation_ci95_high": nominal_ci[1],
            "nominal_within_10pct_count": nominal_within,
            "gate_applicable": gate_applicable,
            "confirmation_passed": feasible if gate_applicable else "",
        }
    )
    decision = {
        "gate_applicable": gate_applicable,
        "primary_hypothesis_passed": feasible if gate_applicable else None,
        "confirmation_passed": feasible if gate_applicable else None,
        "confirmed_regime": (
            CANDIDATE_REGIME if gate_applicable and feasible else None
        ),
        "candidate_regime": CANDIDATE_REGIME,
        "criteria": {
            "nominal_degradation_mean": nominal_mean,
            "nominal_degradation_mean_limit": NOMINAL_MEAN_DEGRADATION_LIMIT,
            "nominal_within_10pct_count": nominal_within,
            "minimum_nominal_within_10pct_count": MINIMUM_NOMINAL_WITHIN_COUNT,
            "stress_delta_mean": stress_mean,
            "stress_better_count": stress_better,
            "minimum_stress_better_count": MINIMUM_CONFIRMATION_WINS,
        },
        "supportive_uncertainty": {
            "method": (
                "paired Student t interval across ten training seeds; "
                "not an additional gate"
            ),
            "stress_delta_ci95": list(stress_ci),
            "nominal_degradation_ci95": list(nominal_ci),
        },
    }
    return rows, decision


def run(args: argparse.Namespace, *, algorithm: str = ALGORITHM,
        profile_override: tuple | None = None, protocol_version: str = PROTOCOL_VERSION,
        prior_used_seeds: frozenset[int] = PRIOR_USED_SEEDS,
        gradient_diagnostic_interval: int = 0) -> Path:
    (
        train_seeds,
        evaluation_seeds,
        budgets,
        phase_boundary_steps,
        settings,
        nominal_batch_size,
    ) = profile_override if profile_override is not None else profile_settings(args.profile)
    panels = (
        set(train_seeds),
        set(evaluation_seeds),
        set(SEALED_TEST_SEEDS),
    )
    if any(
        panels[left] & panels[right]
        for left in range(len(panels))
        for right in range(left + 1, len(panels))
    ):
        raise ValueError(
            "training, evaluation, smoke, and sealed panels must be disjoint"
        )
    if (set(train_seeds) | set(evaluation_seeds)) & prior_used_seeds:
        raise ValueError("confirmation panels overlap a prior RA-QMIX protocol")
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
            regime, seed, max(budgets), phase_boundary_steps, cells
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
    adaptation = AnchorAdaptationSettings(
        phase_boundary_steps=phase_boundary_steps,
        nominal_batch_size=nominal_batch_size,
        anchor_lambda=0.0,
        anchor_temperature=1.0,
    )
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "experiment": "ra_qmix_curriculum_confirmation",
        "protocol_version": protocol_version,
        "source_development_protocol": SOURCE_DEVELOPMENT_PROTOCOL,
        "source_development_commit": SOURCE_DEVELOPMENT_COMMIT,
        "profile": args.profile,
        "hypothesis": (
            "The locked curriculum-replay recipe repeats the nominal-preservation "
            "and stress-robustness gate on ten fresh paired training seeds."
        ),
        "primary_endpoint": (
            "feasibility of curriculum_replay at 480384 steps versus paired "
            "nominal_100 on ten fresh training seeds"
        ),
        "gate": (
            "nominal mean <= 5%, nominal <= 10% in >=8/10 seeds, stress delta "
            "mean < 0, and stress better in >=8/10 seeds"
        ),
        "gate_applicable": args.profile == "full",
        "stopping_rule": (
            "Fixed environment-step budget; stop only for error, non-finite loss, "
            "missing checkpoint, or audit violation."
        ),
        "algorithm": algorithm,
        "regime_specs": REGIME_SPECS,
        "phase_boundary_steps": phase_boundary_steps,
        "adaptation_settings": asdict(adaptation),
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "step_budgets": list(budgets),
        "primary_step_budget": max(budgets),
        "frozen_scenario_references": FROZEN_REFERENCES,
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
            "protocol_version": protocol_version,
            "environment_cells": manifest["environment_cells"],
            "objective_version": OBJECTIVE_VERSION,
            "observation_version": OBSERVATION_VERSION,
        },
    )
    _write_json(
        output / "resolved_config.json",
        {
            "profile": args.profile,
            "algorithm": algorithm,
            "regime_specs": REGIME_SPECS,
            "phase_boundary_steps": phase_boundary_steps,
            "adaptation_settings": asdict(adaptation),
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
        _write_csv(
            output / "regime_registry.csv",
            [
                {"training_regime": regime, **spec}
                for regime, spec in REGIME_SPECS.items()
            ],
        )
        parameter_rows = []
        for regime in REGIMES:
            model = PassiveValueDecomposition(
                train_config,
                algorithm,
                settings.hidden_dim,
                settings.mixer_hidden_dim,
            )
            parameter_rows.append(
                {
                    "training_regime": regime,
                    "algorithm": algorithm,
                    "edge_utilities": COMPONENT_FLAGS[algorithm][0],
                    "queue_mixer": COMPONENT_FLAGS[algorithm][1],
                    "counterfactual_loss": COMPONENT_FLAGS[algorithm][2],
                    "lambda_cf": settings.lambda_cf,
                    "anchor_lambda": 0.0,
                    **model.parameter_counts(),
                }
            )
        _write_csv(output / "parameter_counts.csv", parameter_rows)

        schedule_rows = []
        for regime in REGIMES:
            for seed in train_seeds:
                schedule = schedules[regime, seed]
                schedule_hash = _schedule_hash(schedule)
                steps = 0
                phase_counts: dict[tuple[str, str], int] = {}
                phase_steps: dict[tuple[str, str], int] = {}
                for scenario in schedule:
                    phase = "phase_one" if steps < phase_boundary_steps else "phase_two"
                    key = (phase, scenario)
                    phase_counts[key] = phase_counts.get(key, 0) + 1
                    phase_steps[key] = phase_steps.get(key, 0) + cells[scenario].horizon
                    steps += cells[scenario].horizon
                for phase in ("phase_one", "phase_two"):
                    phase_total = sum(
                        value
                        for (row_phase, _), value in phase_steps.items()
                        if row_phase == phase
                    )
                    for scenario in cells:
                        key = (phase, scenario)
                        schedule_rows.append(
                            {
                                "training_regime": regime,
                                "train_seed": seed,
                                "training_phase": phase,
                                "scenario": scenario,
                                "episode_count": phase_counts.get(key, 0),
                                "environment_steps": phase_steps.get(key, 0),
                                "transition_share_within_phase": phase_steps.get(key, 0)
                                / phase_total,
                                "schedule_sha256": schedule_hash,
                            }
                        )
        _write_csv(output / "training_schedule.csv", schedule_rows)

        evaluation_rows: list[dict[str, Any]] = []
        evaluation_progress: list[dict[str, Any]] = []
        checkpoint_paths: list[str] = []
        checkpoint_map: dict[tuple[str, int, int], Path] = {}
        final_steps: dict[tuple[str, int], int] = {}
        update_counts: dict[tuple[str, int], int] = {}
        training_counts: dict[tuple[str, int, str], int] = {}
        training_phase_counts: dict[tuple[str, int, str, str], int] = {}
        adaptation_diagnostics: dict[int, dict[str, Any]] = {}
        losses_finite = True
        all_anchor_losses_zero = True
        for regime in REGIMES:
            for train_seed in tqdm(
                train_seeds, desc=f"train seeds/{regime}", unit="seed"
            ):
                gradient_records: list[dict[str, Any]] = []
                root = output / regime / f"train_seed_{train_seed}" / "checkpoints"
                if regime == CANDIDATE_REGIME:
                    checkpoints, raw_progress, diagnostics = (
                        train_value_decomposition_anchor_checkpoints(
                            train_config,
                            algorithm,
                            train_seed,
                            budgets,
                            root,
                            settings,
                            device,
                            cells,
                            schedules[regime, train_seed],
                            adaptation,
                            gradient_diagnostic_interval=gradient_diagnostic_interval,
                            gradient_records=gradient_records,
                        )
                    )
                    adaptation_diagnostics[train_seed] = diagnostics
                else:
                    checkpoints, raw_progress = (
                        train_value_decomposition_step_checkpoints(
                            train_config,
                            algorithm,
                            train_seed,
                            budgets,
                            root,
                            settings,
                            device,
                            cells,
                            schedules[regime, train_seed],
                            gradient_diagnostic_interval=gradient_diagnostic_interval,
                            gradient_records=gradient_records,
                        )
                    )
                if gradient_records:
                    _append_csv(output / "gradient_diagnostics.csv", [
                        {"training_regime": regime, "algorithm": algorithm, **row}
                        for row in gradient_records
                    ])
                progress = [
                    _normalized_progress_row(regime, row, phase_boundary_steps)
                    for row in raw_progress
                ]
                cumulative_updates = 0
                for row in progress:
                    cumulative_updates += int(row["update_count"])
                    row["episode_optimizer_updates"] = int(row["update_count"])
                    row["cumulative_optimizer_updates"] = cumulative_updates
                _append_csv(output / "training_progress.csv", progress)
                _append_csv(output / "training_episodes.csv", progress)
                final_steps[regime, train_seed] = int(
                    progress[-1]["environment_steps"]
                )
                update_counts[regime, train_seed] = sum(
                    int(row["update_count"]) for row in progress
                )
                for scenario in cells:
                    training_counts[regime, train_seed, scenario] = sum(
                        row["training_scenario"] == scenario for row in progress
                    )
                for phase in {str(row["training_phase"]) for row in progress}:
                    for scenario in cells:
                        key = (regime, train_seed, phase, scenario)
                        training_phase_counts[key] = sum(
                            row["training_phase"] == phase
                            and row["training_scenario"] == scenario
                            for row in progress
                        )
                losses_finite = losses_finite and all(
                    math.isfinite(float(row[name]))
                    for row in progress
                    for name in (
                        "td_loss",
                        "raw_cf_loss",
                        "weighted_cf_loss",
                        "raw_anchor_loss",
                        "weighted_anchor_loss",
                        "total_loss",
                    )
                    if row.get(name) not in (None, "")
                )
                all_anchor_losses_zero = all_anchor_losses_zero and all(
                    float(row["weighted_anchor_loss"]) == 0 for row in progress
                )
                for budget in budgets:
                    checkpoint = checkpoints[budget]
                    checkpoint_map[regime, train_seed, budget] = checkpoint
                    checkpoint_paths.append(str(checkpoint.relative_to(output)))
                    model = PassiveValueDecomposition.load(
                        checkpoint, train_config, device
                    )
                    for scenario, evaluation_config in cells.items():
                        for eval_seed in tqdm(
                            evaluation_seeds,
                            desc=f"evaluate {regime}/{train_seed}/{budget}/{scenario}",
                            unit="episode",
                            leave=False,
                        ):
                            evaluation_rows.append(
                                {
                                    "training_regime": regime,
                                    **evaluate_diagnostic_episode(
                                        evaluation_config,
                                        model,
                                        eval_seed,
                                        train_seed,
                                        scenario,
                                        budget,
                                        device,
                                    ),
                                }
                            )
                        progress_row = {
                            "training_regime": regime,
                            "train_seed": train_seed,
                            "step_budget": budget,
                            "scenario": scenario,
                            "completed_evaluation_episodes": len(evaluation_seeds),
                        }
                        evaluation_progress.append(progress_row)
                        _append_csv(output / "evaluation_progress.csv", [progress_row])
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
                "cost_reconciled": abs(float(row["cost_reconciliation_error"]))
                <= COST_TOLERANCE,
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
        primary, decision = confirmation_results(
            score_rows, summary_rows, max(budgets), args.profile == "full"
        )
        _write_csv(output / "budget_summary.csv", summary_rows)
        _write_csv(output / "regret_scores.csv", score_rows)
        _write_csv(output / "primary_results.csv", primary)
        _write_json(output / "confirmation_result.json", decision)

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
        parameter_signatures = [
            {key: value for key, value in row.items() if key != "training_regime"}
            for row in parameter_rows
        ]
        phase_boundary_equal = all(
            _checkpoint_states_equal(
                checkpoint_map[CONTROL_REGIME, seed, phase_boundary_steps],
                checkpoint_map[CANDIDATE_REGIME, seed, phase_boundary_steps],
            )
            for seed in train_seeds
        )
        replay_stratification_valid = all(
            diagnostics["phase_two_stratified_updates"] > 0
            and diagnostics["stress_batch_size"]
            == settings.batch_size - nominal_batch_size
            and diagnostics["nominal_batch_size"] == nominal_batch_size
            and diagnostics["phase_one_replay_capacity"] == settings.replay_capacity
            and diagnostics["phase_two_total_replay_capacity"]
            == settings.replay_capacity
            and diagnostics["total_stress_batch_samples"]
            == diagnostics["phase_two_stratified_updates"]
            * diagnostics["stress_batch_size"]
            for diagnostics in adaptation_diagnostics.values()
        )
        exact_training_schedule = all(
            training_counts[regime, seed, scenario]
            == schedules[regime, seed].count(scenario)
            for regime in REGIMES
            for seed in train_seeds
            for scenario in cells
        )
        exact_nominal_transition_shares = all(
            math.isclose(
                training_counts[regime, seed, "in_distribution"]
                * cells["in_distribution"].horizon
                / max(budgets),
                float(REGIME_SPECS[regime]["overall_nominal_transition_share"]),
                abs_tol=1e-12,
            )
            for regime in REGIMES
            for seed in train_seeds
        )
        curriculum_phase_schedule_exact = all(
            training_phase_counts[
                CANDIDATE_REGIME, seed, "nominal_pretrain", "in_distribution"
            ]
            * cells["in_distribution"].horizon
            == phase_boundary_steps
            and all(
                training_phase_counts.get(
                    (CANDIDATE_REGIME, seed, "nominal_pretrain", scenario), 0
                )
                == 0
                for scenario in STRESS_SCENARIOS
            )
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
            "all_training_episodes_present": sum(training_counts.values())
            == expected_training_episodes,
            "training_trajectory_count": len(update_counts),
            "all_trajectories_updated": all(
                value > 0 for value in update_counts.values()
            ),
            "matched_optimizer_updates": len(set(update_counts.values())) == 1,
            "exact_environment_step_budget": all(
                value == max(budgets) for value in final_steps.values()
            ),
            "exact_training_schedule": exact_training_schedule,
            "exact_nominal_transition_shares": exact_nominal_transition_shares,
            "curriculum_phase_schedule_exact": curriculum_phase_schedule_exact,
            "architecture_and_base_loss_identical": all(
                row == parameter_signatures[0] for row in parameter_signatures[1:]
            ),
            "phase_boundary_states_identical": phase_boundary_equal,
            "phase_two_replay_stratification_valid": replay_stratification_valid,
            "all_anchor_losses_zero": all_anchor_losses_zero,
            "paired_training_seeds": set(final_steps)
            == {(regime, seed) for regime in REGIMES for seed in train_seeds},
            "profile_seed_count_valid": len(train_seeds)
            == (10 if args.profile == "full" else 1),
            "locked_recipe_only": set(REGIMES)
            == {CONTROL_REGIME, CANDIDATE_REGIME},
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
                (set(train_seeds) | set(evaluation_seeds)) & prior_used_seeds
            ),
            "sealed_test_panel_closed": not any(
                int(row["eval_seed"]) in SEALED_TEST_SEEDS
                for row in evaluation_rows
            ),
        }
        required = tuple(
            name
            for name, value in audits.items()
            if isinstance(value, bool)
        )
        if not all(audits[name] for name in required):
            raise RuntimeError(f"experiment audit failed: {audits}")
        diagnostics_rows = [
            {
                "training_regime": CANDIDATE_REGIME,
                "train_seed": seed,
                **values,
            }
            for seed, values in adaptation_diagnostics.items()
        ]
        _write_csv(output / "adaptation_diagnostics.csv", diagnostics_rows)
        summary = {
            "protocol_version": protocol_version,
            "primary_endpoint": manifest["primary_endpoint"],
            "confirmation_result": decision,
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
                    "training_trajectories": len(update_counts),
                    "training_episodes": sum(training_counts.values()),
                },
                "confirmation_result": decision,
                "adaptation_diagnostics": diagnostics_rows,
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
