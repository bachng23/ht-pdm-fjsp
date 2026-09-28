from __future__ import annotations

import csv
import json
from argparse import Namespace

import pytest

from ht_pdm_fjsp.passive_technician_domain_randomization import (
    FULL_EVALUATION_SEEDS,
    FULL_STEP_BUDGETS,
    FULL_TRAIN_SEEDS,
    REGIMES,
    budget_summary,
    primary_results,
    profile_settings,
    robust_scores,
    run,
    training_schedule,
)
from ht_pdm_fjsp.passive_technician_leave_one_out import environment_cells


def test_protocol_panels_budgets_and_regimes_are_locked() -> None:
    train_seeds, evaluation_seeds, budgets, settings = profile_settings("full")
    assert train_seeds == FULL_TRAIN_SEEDS == tuple(range(83_800, 83_810))
    assert evaluation_seeds == FULL_EVALUATION_SEEDS == tuple(range(83_900, 84_000))
    assert budgets == FULL_STEP_BUDGETS == (240_240, 360_360, 480_480)
    assert settings.episodes == 40_040
    assert REGIMES == ("nominal_only", "uniform_four_scenario")
    panels = (
        set(train_seeds),
        set(evaluation_seeds),
        set(range(201, 301)),
    )
    assert all(
        panels[left].isdisjoint(panels[right])
        for left in range(len(panels))
        for right in range(left + 1, len(panels))
    )


def test_training_schedules_have_equal_steps_and_exact_balance() -> None:
    cells = environment_cells()
    maximum_steps = 396
    nominal = training_schedule("nominal_only", 83_710, maximum_steps, cells)
    mixture = training_schedule(
        "uniform_four_scenario", 83_710, maximum_steps, cells
    )
    assert sum(cells[name].horizon for name in nominal) == maximum_steps
    assert sum(cells[name].horizon for name in mixture) == maximum_steps
    assert len(nominal) == 33
    assert len(mixture) == 24
    assert {name: mixture.count(name) for name in cells} == {
        name: 6 for name in cells
    }
    assert mixture == training_schedule(
        "uniform_four_scenario", 83_710, maximum_steps, cells
    )
    assert all(
        set(mixture[start : start + 4]) == set(cells)
        for start in range(0, len(mixture), 4)
    )


def _episode_row(
    regime: str,
    train_seed: int,
    scenario: str,
    budget: int,
    objective: float,
) -> dict[str, object]:
    row: dict[str, object] = {
        "training_regime": regime,
        "algorithm": "tqmix",
        "train_seed": train_seed,
        "scenario": scenario,
        "budget": budget,
    }
    for metric in (
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
    ):
        row[metric] = objective if metric == "objective" else 0.0
    return row


def test_primary_screening_gate_uses_paired_seeds_and_nominal_margin() -> None:
    scenarios = tuple(environment_cells())
    episodes = []
    for seed_index, seed in enumerate(FULL_TRAIN_SEEDS):
        for regime in REGIMES:
            for scenario in scenarios:
                control_value = {
                    "in_distribution": 10.0,
                    "early_failure": 140.0,
                    "slow_service": 100.0,
                    "combined_pressure": 190.0,
                }[scenario]
                treatment_value = control_value * (
                    1.04 if scenario == "in_distribution" else 0.90
                )
                if regime == REGIMES[0]:
                    value = control_value
                else:
                    value = treatment_value if seed_index < 8 else control_value * 1.02
                episodes.append(
                    _episode_row(regime, seed, scenario, 480_480, value)
                )
    summaries = budget_summary(episodes)
    scores = robust_scores(summaries)
    primary = primary_results(scores, summaries, 480_480, True)
    aggregate = primary[-1]
    assert aggregate["delta_mean"] < 0
    assert aggregate["treatment_better_count"] == 8
    assert aggregate["nominal_relative_degradation"] <= 0.05
    assert aggregate["screening_gate_passed"] is True


def test_runner_rejects_nonempty_output_directory(tmp_path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(
            Namespace(
                profile="smoke",
                device="cpu",
                output_dir=str(output),
            )
        )


def test_smoke_runner_meets_counts_and_audits(tmp_path) -> None:
    output = run(
        Namespace(
            profile="smoke",
            device="cpu",
            output_dir=str(tmp_path / "smoke"),
        )
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["resolved_device"] == "cpu"
    assert manifest["actual_counts"] == {
        "evaluation_rows": 72,
        "checkpoints": 6,
        "training_trajectories": 2,
        "training_episodes": 57,
    }
    required = (
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
    assert all(manifest["audits"][name] for name in required)
    with (output / "episodes.csv").open(encoding="utf-8") as stream:
        assert sum(1 for _ in csv.DictReader(stream)) == 72
    assert manifest["screening_gate_applicable"] is False
    assert (output / "primary_results.csv").is_file()
