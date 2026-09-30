"""Test curriculum replay and nominal-policy anchoring for Full RA-QMIX."""

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
    AnchorAdaptationSettings,
    PassiveValueDecomposition,
    ValueTrainSettings,
    train_value_decomposition_anchor_checkpoints,
    train_value_decomposition_step_checkpoints,
)


PROTOCOL_VERSION = "ra_qmix_anchor_preservation_v1"
ALGORITHM = "tqmix"
CONTROL_REGIME = "nominal_100"
FIXED_COMPARATOR = "fixed_87_5"
CURRICULUM_REGIMES = ("curriculum_replay", "curriculum_kl_anchor")
REGIME_SPECS = {
    CONTROL_REGIME: {
        "training_design": "nominal_only",
        "overall_nominal_transition_share": 1.0,
        "phase_two_nominal_replay_share": 1.0,
        "anchor_lambda": 0.0,
        "selection_eligible": False,
    },
    FIXED_COMPARATOR: {
        "training_design": "fixed_mixture",
        "overall_nominal_transition_share": 0.875,
        "phase_two_nominal_replay_share": 0.875,
        "anchor_lambda": 0.0,
        "selection_eligible": False,
    },
    "curriculum_replay": {
        "training_design": "nominal_pretrain_then_stratified_adaptation",
        "overall_nominal_transition_share": 0.9375,
        "phase_two_nominal_replay_share": 0.875,
        "anchor_lambda": 0.0,
        "selection_eligible": True,
    },
    "curriculum_kl_anchor": {
        "training_design": "nominal_pretrain_then_stratified_adaptation_with_kl",
        "overall_nominal_transition_share": 0.9375,
        "phase_two_nominal_replay_share": 0.875,
        "anchor_lambda": 10.0,
        "selection_eligible": True,
    },
}
REGIMES = tuple(REGIME_SPECS)
FULL_TRAIN_SEEDS = tuple(range(84_300, 84_305))
FULL_EVALUATION_SEEDS = tuple(range(84_400, 84_450))
SMOKE_TRAIN_SEEDS = (84_500,)
SMOKE_EVALUATION_SEEDS = (84_510, 84_511, 84_512)
FULL_STEP_BUDGETS = (240_192, 360_288, 480_384)
SMOKE_STEP_BUDGETS = (1_728, 2_592, 3_456)
PRIOR_USED_SEEDS = frozenset((*range(1, 301), *range(83_090, 84_300)))
STRESS_SCENARIOS = ("early_failure", "slow_service", "combined_pressure")
ANCHOR_TEMPERATURE = 1.0


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


def _append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _mixed_blocks(
    train_seed: int,
    block_count: int,
    regime_index: int,
) -> tuple[str, ...]:
    block = ["in_distribution"] * 63
    for scenario in STRESS_SCENARIOS:
        block.extend([scenario] * 2)
    rng = np.random.default_rng(train_seed + 93_117_000 + regime_index * 10_000)
    schedule: list[str] = []
    for _ in range(block_count):
        schedule.extend(str(name) for name in rng.permutation(block))
    return tuple(schedule)


def training_schedule(
    regime: str,
    train_seed: int,
    maximum_steps: int,
    phase_boundary_steps: int,
    cells: dict[str, Any],
) -> tuple[str, ...]:
    if regime not in REGIME_SPECS:
        raise ValueError(regime)
    nominal_horizon = cells["in_distribution"].horizon
    if regime == CONTROL_REGIME:
        if maximum_steps % nominal_horizon:
            raise ValueError("nominal budget must close an episode")
        return ("in_distribution",) * (maximum_steps // nominal_horizon)
    block_steps = 63 * nominal_horizon + sum(
        2 * cells[name].horizon for name in STRESS_SCENARIOS
    )
    if regime == FIXED_COMPARATOR:
        if maximum_steps % block_steps:
            raise ValueError("fixed mixture budget must close a block")
        return _mixed_blocks(
            train_seed, maximum_steps // block_steps, REGIMES.index(regime)
        )
    if phase_boundary_steps % nominal_horizon:
        raise ValueError("phase boundary must close a nominal episode")
    adaptation_steps = maximum_steps - phase_boundary_steps
    if adaptation_steps <= 0 or adaptation_steps % block_steps:
        raise ValueError("adaptation budget must close a positive number of blocks")
    return (
        ("in_distribution",) * (phase_boundary_steps // nominal_horizon)
        + _mixed_blocks(
            train_seed,
            adaptation_steps // block_steps,
            REGIMES.index("curriculum_replay"),
        )
    )


def _schedule_hash(schedule: tuple[str, ...]) -> str:
    return hashlib.sha256("\n".join(schedule).encode()).hexdigest()


def _normalized_progress_row(
    regime: str, row: dict[str, Any], phase_boundary_steps: int
) -> dict[str, Any]:
    phase = row.get("training_phase")
    if phase is None:
        if regime == CONTROL_REGIME:
            phase = "nominal_training"
        elif regime == FIXED_COMPARATOR:
            phase = "fixed_mixture_training"
        else:
            phase = (
                "nominal_pretrain"
                if int(row["environment_steps"]) <= phase_boundary_steps
                else "stress_adaptation"
            )
    ordered = {
        "training_regime": regime,
        "policy": row["policy"],
        "train_seed": row["train_seed"],
        "training_scenario": row["training_scenario"],
        "training_phase": phase,
        "episode": row["episode"],
        "environment_steps": row["environment_steps"],
        "update_count": row["update_count"],
        "batch_nominal_samples": row.get("batch_nominal_samples", ""),
        "batch_stress_samples": row.get("batch_stress_samples", ""),
        "stratified_update_count": row.get("stratified_update_count", ""),
        "objective": row["objective"],
        "td_loss": row["td_loss"],
        "raw_cf_loss": row["raw_cf_loss"],
        "weighted_cf_loss": row["weighted_cf_loss"],
        "raw_anchor_loss": row.get("raw_anchor_loss", 0.0),
        "weighted_anchor_loss": row.get("weighted_anchor_loss", 0.0),
        "total_loss": row["total_loss"],
        "loss": row["loss"],
        "epsilon": row["epsilon"],
    }
    excluded = set(ordered) | {"training_phase"}
    ordered.update(
        (key, row[key]) for key in sorted(row) if key not in excluded
    )
    return ordered


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
        stress_deltas = [
            scores[regime, seed] - scores[CONTROL_REGIME, seed] for seed in seeds
        ]
        nominal_degradations = [
            nominal[regime, seed] / nominal[CONTROL_REGIME, seed] - 1.0
            for seed in seeds
        ]
        for seed, stress_delta, nominal_degradation in zip(
            seeds, stress_deltas, nominal_degradations, strict=True
        ):
            result.append(
                {
                    "row_type": "training_seed",
                    "candidate_regime": regime,
                    "selection_eligible": REGIME_SPECS[regime][
                        "selection_eligible"
                    ],
                    "train_seed": seed,
                    "primary_step_budget": primary_budget,
                    "control_stress_max_regret": scores[CONTROL_REGIME, seed],
                    "candidate_stress_max_regret": scores[regime, seed],
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
        candidate = {
            "candidate_regime": regime,
            "selection_eligible": REGIME_SPECS[regime]["selection_eligible"],
            "stress_score_mean": statistics.fmean(
                scores[regime, seed] for seed in seeds
            ),
            "stress_delta_mean": stress_mean,
            "nominal_degradation_mean": nominal_mean,
            "feasible": feasible,
        }
        candidates.append(candidate)
        result.append(
            {
                "row_type": "across_training_seeds",
                "candidate_regime": regime,
                "selection_eligible": candidate["selection_eligible"],
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
    feasible = [
        item
        for item in candidates
        if item["selection_eligible"] and item["feasible"]
    ]
    complexity = {"curriculum_replay": 0, "curriculum_kl_anchor": 1}
    selected = (
        min(
            feasible,
            key=lambda item: (
                item["stress_score_mean"],
                item["nominal_degradation_mean"],
                complexity[item["candidate_regime"]],
            ),
        )
        if gate_applicable and feasible
        else None
    )
    selected_regime = selected["candidate_regime"] if selected else None
    for row in result:
        if row["row_type"] == "across_training_seeds" and gate_applicable:
            row["selected"] = row["candidate_regime"] == selected_regime
    anchored = next(
        item for item in candidates if item["candidate_regime"] == "curriculum_kl_anchor"
    )
    selection = {
        "gate_applicable": gate_applicable,
        "primary_hypothesis_passed": bool(
            gate_applicable and anchored["feasible"]
        ),
        "preservation_intervention_passed": selected is not None,
        "selected_regime": selected_regime,
        "selection_rule": "among feasible curriculum arms, minimize (mean stress-max regret, mean nominal degradation, intervention complexity)",
        "candidate_summaries": candidates,
    }
    return result, selection


def _checkpoint_states_equal(left: Path, right: Path) -> bool:
    left_state = torch.load(left, map_location="cpu", weights_only=True)["state_dict"]
    right_state = torch.load(right, map_location="cpu", weights_only=True)[
        "state_dict"
    ]
    return left_state.keys() == right_state.keys() and all(
        torch.equal(left_state[key], right_state[key]) for key in left_state
    )


def run(args: argparse.Namespace) -> Path:
    (
        train_seeds,
        evaluation_seeds,
        budgets,
        phase_boundary_steps,
        settings,
        nominal_batch_size,
    ) = profile_settings(args.profile)
    panels = (set(train_seeds), set(evaluation_seeds), set(SEALED_TEST_SEEDS))
    if any(
        panels[left] & panels[right]
        for left in range(len(panels))
        for right in range(left + 1, len(panels))
    ):
        raise ValueError("training, evaluation, and sealed panels must be disjoint")
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
    adaptation_by_regime = {
        regime: AnchorAdaptationSettings(
            phase_boundary_steps=phase_boundary_steps,
            nominal_batch_size=nominal_batch_size,
            anchor_lambda=float(REGIME_SPECS[regime]["anchor_lambda"]),
            anchor_temperature=ANCHOR_TEMPERATURE,
        )
        for regime in CURRICULUM_REGIMES
    }
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "experiment": "ra_qmix_anchor_preservation",
        "protocol_version": PROTOCOL_VERSION,
        "profile": args.profile,
        "hypothesis": "Nominal pretraining plus stratified adaptation and behavior-KL anchoring preserves nominal performance while improving paired stress-max regret.",
        "primary_endpoint": "feasibility of curriculum_kl_anchor at the final step checkpoint versus paired nominal_100",
        "gate": "nominal mean <= 5%, nominal <= 10% in >=4/5 seeds, stress delta mean < 0, and stress better in >=4/5 seeds",
        "gate_applicable": args.profile == "full",
        "stopping_rule": "Fixed environment-step budget; stop only for error, non-finite loss, missing checkpoint, or audit violation.",
        "algorithm": ALGORITHM,
        "regime_specs": REGIME_SPECS,
        "phase_boundary_steps": phase_boundary_steps,
        "adaptation_settings": {
            regime: asdict(value) for regime, value in adaptation_by_regime.items()
        },
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
            "phase_boundary_steps": phase_boundary_steps,
            "adaptation_settings": manifest["adaptation_settings"],
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
                ALGORITHM,
                settings.hidden_dim,
                settings.mixer_hidden_dim,
            )
            parameter_rows.append(
                {
                    "training_regime": regime,
                    "algorithm": ALGORITHM,
                    "edge_utilities": COMPONENT_FLAGS[ALGORITHM][0],
                    "queue_mixer": COMPONENT_FLAGS[ALGORITHM][1],
                    "counterfactual_loss": COMPONENT_FLAGS[ALGORITHM][2],
                    "lambda_cf": settings.lambda_cf,
                    "anchor_lambda": REGIME_SPECS[regime]["anchor_lambda"],
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
                    phase = (
                        "phase_one"
                        if steps < phase_boundary_steps
                        else "phase_two"
                    )
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
                                "transition_share_within_phase": (
                                    phase_steps.get(key, 0) / phase_total
                                ),
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
        adaptation_diagnostics: dict[tuple[str, int], dict[str, Any]] = {}
        losses_finite = True
        anchor_active = False
        unanchored_anchor_zero = True
        for regime in REGIMES:
            for train_seed in tqdm(
                train_seeds, desc=f"train seeds/{regime}", unit="seed"
            ):
                root = output / regime / f"train_seed_{train_seed}" / "checkpoints"
                if regime in CURRICULUM_REGIMES:
                    checkpoints, raw_progress, diagnostics = (
                        train_value_decomposition_anchor_checkpoints(
                            train_config,
                            ALGORITHM,
                            train_seed,
                            budgets,
                            root,
                            settings,
                            device,
                            cells,
                            schedules[regime, train_seed],
                            adaptation_by_regime[regime],
                        )
                    )
                    adaptation_diagnostics[regime, train_seed] = diagnostics
                else:
                    checkpoints, raw_progress = (
                        train_value_decomposition_step_checkpoints(
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
                    )
                progress = [
                    _normalized_progress_row(regime, row, phase_boundary_steps)
                    for row in raw_progress
                ]
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
                        training_phase_counts[regime, train_seed, phase, scenario] = sum(
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
                if regime == "curriculum_kl_anchor":
                    anchor_active = anchor_active or any(
                        float(row["weighted_anchor_loss"]) > 0 for row in progress
                    )
                else:
                    unanchored_anchor_zero = unanchored_anchor_zero and all(
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
                "cost_reconciled": abs(
                    float(row["cost_reconciliation_error"])
                )
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
        primary, selection = primary_results(
            score_rows, summary_rows, max(budgets), args.profile == "full"
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
        parameter_signatures = [
            {
                key: value
                for key, value in row.items()
                if key not in ("training_regime", "anchor_lambda")
            }
            for row in parameter_rows
        ]
        phase_boundary_equal = all(
            _checkpoint_states_equal(
                checkpoint_map[
                    "curriculum_replay", train_seed, phase_boundary_steps
                ],
                checkpoint_map[
                    "curriculum_kl_anchor", train_seed, phase_boundary_steps
                ],
            )
            for train_seed in train_seeds
        )
        replay_stratification_valid = all(
            diagnostics["phase_two_stratified_updates"] > 0
            and diagnostics["stress_batch_size"]
            == settings.batch_size - nominal_batch_size
            and diagnostics["nominal_batch_size"] == nominal_batch_size
            and diagnostics["phase_one_replay_capacity"]
            == settings.replay_capacity
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
                float(
                    REGIME_SPECS[regime]["overall_nominal_transition_share"]
                ),
                abs_tol=1e-12,
            )
            for regime in REGIMES
            for seed in train_seeds
        )
        curriculum_phase_schedule_exact = all(
            training_phase_counts[
                regime, seed, "nominal_pretrain", "in_distribution"
            ]
            * cells["in_distribution"].horizon
            == phase_boundary_steps
            and all(
                training_phase_counts.get(
                    (regime, seed, "nominal_pretrain", scenario), 0
                )
                == 0
                for scenario in STRESS_SCENARIOS
            )
            and schedules["curriculum_replay", seed]
            == schedules["curriculum_kl_anchor", seed]
            for regime in CURRICULUM_REGIMES
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
            "training_trajectory_count": len(update_counts),
            "all_trajectories_updated": all(value > 0 for value in update_counts.values()),
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
            "curriculum_phase_boundary_states_identical": phase_boundary_equal,
            "phase_two_replay_stratification_valid": replay_stratification_valid,
            "anchor_loss_active_only_in_anchor_arm": (
                anchor_active and unanchored_anchor_zero
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
        required = (
            "all_expected_rows_present",
            "unique_episode_keys",
            "all_checkpoints_present",
            "all_training_episodes_present",
            "all_trajectories_updated",
            "matched_optimizer_updates",
            "exact_environment_step_budget",
            "exact_training_schedule",
            "exact_nominal_transition_shares",
            "curriculum_phase_schedule_exact",
            "architecture_and_base_loss_identical",
            "curriculum_phase_boundary_states_identical",
            "phase_two_replay_stratification_valid",
            "anchor_loss_active_only_in_anchor_arm",
            "paired_training_seeds",
            "losses_finite",
            "cost_reconciliation_passed",
            "resource_semantics_passed",
            "evaluation_progress_complete",
            "seed_panels_disjoint",
            "prior_seed_panels_disjoint",
            "sealed_test_panel_closed",
        )
        if not all(audits[name] for name in required):
            raise RuntimeError(f"experiment audit failed: {audits}")
        diagnostics_rows = [
            {"training_regime": regime, "train_seed": seed, **values}
            for (regime, seed), values in adaptation_diagnostics.items()
        ]
        _write_csv(output / "adaptation_diagnostics.csv", diagnostics_rows)
        summary = {
            "protocol_version": PROTOCOL_VERSION,
            "primary_endpoint": manifest["primary_endpoint"],
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
                    "training_trajectories": len(update_counts),
                    "training_episodes": sum(training_counts.values()),
                },
                "selection_result": selection,
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
