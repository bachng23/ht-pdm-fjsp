import argparse
import json

import pytest

from ht_pdm_fjsp.passive_technician_leave_one_out import (
    SEALED_TEST_SEEDS,
    environment_cells,
)
from ht_pdm_fjsp.passive_technician_nominal_anchor_calibration import (
    CONTROL_REGIME,
    FULL_EVALUATION_SEEDS,
    FULL_STEP_BUDGETS,
    FULL_TRAIN_SEEDS,
    PRIOR_USED_SEEDS,
    PROTOCOL_VERSION,
    REGIMES,
    REGIME_SPECS,
    SMOKE_EVALUATION_SEEDS,
    SMOKE_STEP_BUDGETS,
    SMOKE_TRAIN_SEEDS,
    primary_results,
    run,
    training_schedule,
)


def test_protocol_and_seed_panels_are_locked_and_disjoint():
    assert PROTOCOL_VERSION == "ra_qmix_nominal_anchor_calibration_v1"
    assert REGIMES == (
        "nominal_100",
        "nominal_87_5",
        "nominal_75",
        "nominal_50",
    )
    assert FULL_STEP_BUDGETS == (240_192, 360_288, 480_384)
    assert SMOKE_STEP_BUDGETS == (864, 1_728, 2_592)
    assert len(FULL_TRAIN_SEEDS) == 5
    assert len(FULL_EVALUATION_SEEDS) == 50
    panels = (
        set(FULL_TRAIN_SEEDS),
        set(FULL_EVALUATION_SEEDS),
        set(SMOKE_TRAIN_SEEDS),
        set(SMOKE_EVALUATION_SEEDS),
        set(SEALED_TEST_SEEDS),
    )
    assert all(
        not panels[left] & panels[right]
        for left in range(len(panels))
        for right in range(left + 1, len(panels))
    )
    assert not set().union(*panels[:-1]) & PRIOR_USED_SEEDS


def test_smoke_schedules_have_exact_transition_shares_and_counts():
    cells = environment_cells()
    expected = {
        "nominal_100": {"in_distribution": 216},
        "nominal_87_5": {
            "in_distribution": 189,
            "early_failure": 6,
            "slow_service": 6,
            "combined_pressure": 6,
        },
        "nominal_75": {
            "in_distribution": 162,
            "early_failure": 12,
            "slow_service": 12,
            "combined_pressure": 12,
        },
        "nominal_50": {
            "in_distribution": 108,
            "early_failure": 24,
            "slow_service": 24,
            "combined_pressure": 24,
        },
    }
    total_episodes = 0
    for regime in REGIMES:
        schedule = training_schedule(regime, SMOKE_TRAIN_SEEDS[0], 2_592, cells)
        assert schedule == training_schedule(
            regime, SMOKE_TRAIN_SEEDS[0], 2_592, cells
        )
        assert {name: schedule.count(name) for name in cells} == {
            name: expected[regime].get(name, 0) for name in cells
        }
        total_steps = sum(cells[name].horizon for name in schedule)
        nominal_steps = schedule.count("in_distribution") * cells[
            "in_distribution"
        ].horizon
        assert total_steps == 2_592
        assert nominal_steps / total_steps == pytest.approx(
            REGIME_SPECS[regime]["nominal_transition_share"]
        )
        total_episodes += len(schedule)
    assert total_episodes == 801


def test_primary_rule_selects_best_feasible_candidate():
    score_by_regime = {
        CONTROL_REGIME: 0.50,
        "nominal_87_5": 0.40,
        "nominal_75": 0.35,
        "nominal_50": 0.30,
    }
    nominal_by_regime = {
        CONTROL_REGIME: 10.0,
        "nominal_87_5": 10.4,
        "nominal_75": 11.2,
        "nominal_50": 13.0,
    }
    score_rows = [
        {
            "row_type": "training_seed",
            "training_regime": regime,
            "train_seed": seed,
            "step_budget": max(FULL_STEP_BUDGETS),
            "stress_max_relative_regret": score,
        }
        for regime, score in score_by_regime.items()
        for seed in FULL_TRAIN_SEEDS
    ]
    summary_rows = [
        {
            "training_regime": regime,
            "train_seed": seed,
            "scenario": "in_distribution",
            "step_budget": max(FULL_STEP_BUDGETS),
            "objective_mean": objective,
        }
        for regime, objective in nominal_by_regime.items()
        for seed in FULL_TRAIN_SEEDS
    ]
    rows, selection = primary_results(
        score_rows, summary_rows, max(FULL_STEP_BUDGETS), True
    )
    assert selection["calibration_passed"] is True
    assert selection["selected_regime"] == "nominal_87_5"
    aggregates = {
        row["candidate_regime"]: row
        for row in rows
        if row["row_type"] == "across_training_seeds"
    }
    assert aggregates["nominal_87_5"]["feasible"] is True
    assert aggregates["nominal_87_5"]["selected"] is True
    assert aggregates["nominal_75"]["feasible"] is False
    assert aggregates["nominal_50"]["feasible"] is False


def test_runner_rejects_nonempty_output_directory(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "sentinel.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(argparse.Namespace(profile="smoke", device="cpu", output_dir=output))


def test_smoke_run_completes_with_expected_artifacts(tmp_path):
    output = tmp_path / "smoke"
    run(argparse.Namespace(profile="smoke", device="cpu", output_dir=output))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["expected_counts"] == manifest["actual_counts"] == {
        "evaluation_rows": 144,
        "checkpoints": 12,
        "training_trajectories": 4,
        "training_episodes": 801,
    }
    assert all(summary["audits"].values())
    assert summary["selection_result"]["gate_applicable"] is False
    assert summary["selection_result"]["selected_regime"] is None
    for name in (
        "training_schedule.csv",
        "training_progress.csv",
        "episodes.csv",
        "budget_summary.csv",
        "regret_scores.csv",
        "primary_results.csv",
        "selection_result.json",
    ):
        assert (output / name).is_file()
