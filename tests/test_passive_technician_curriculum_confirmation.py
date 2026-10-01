import argparse
import json

import pytest

from ht_pdm_fjsp.passive_technician_anchor_preservation import training_schedule
from ht_pdm_fjsp.passive_technician_curriculum_confirmation import (
    CANDIDATE_REGIME,
    CONTROL_REGIME,
    FULL_EVALUATION_SEEDS,
    FULL_STEP_BUDGETS,
    FULL_TRAIN_SEEDS,
    MINIMUM_CONFIRMATION_WINS,
    MINIMUM_NOMINAL_WITHIN_COUNT,
    PRIOR_USED_SEEDS,
    PROTOCOL_VERSION,
    REGIMES,
    REGIME_SPECS,
    SMOKE_EVALUATION_SEEDS,
    SMOKE_STEP_BUDGETS,
    SMOKE_TRAIN_SEEDS,
    confirmation_results,
    run,
)
from ht_pdm_fjsp.passive_technician_leave_one_out import (
    SEALED_TEST_SEEDS,
    environment_cells,
)


def test_confirmation_protocol_and_seed_panels_are_locked():
    assert PROTOCOL_VERSION == "ra_qmix_curriculum_confirmation_v1"
    assert REGIMES == ("nominal_100", "curriculum_replay")
    assert FULL_STEP_BUDGETS == (240_192, 360_288, 480_384)
    assert SMOKE_STEP_BUDGETS == (1_728, 2_592, 3_456)
    assert len(FULL_TRAIN_SEEDS) == 10
    assert len(FULL_EVALUATION_SEEDS) == 50
    assert MINIMUM_CONFIRMATION_WINS == 8
    assert MINIMUM_NOMINAL_WITHIN_COUNT == 8
    assert REGIME_SPECS[CANDIDATE_REGIME] == {
        "training_design": "nominal_pretrain_then_stratified_adaptation",
        "overall_nominal_transition_share": 0.9375,
        "phase_two_nominal_replay_share": 0.875,
        "anchor_lambda": 0.0,
    }
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


def test_confirmation_schedules_preserve_selected_recipe():
    cells = environment_cells()
    expected = {
        CONTROL_REGIME: {"in_distribution": 288},
        CANDIDATE_REGIME: {
            "in_distribution": 270,
            "early_failure": 4,
            "slow_service": 4,
            "combined_pressure": 4,
        },
    }
    total = 0
    for regime in REGIMES:
        schedule = training_schedule(
            regime, SMOKE_TRAIN_SEEDS[0], 3_456, 1_728, cells
        )
        assert {name: schedule.count(name) for name in cells} == {
            name: expected[regime].get(name, 0) for name in cells
        }
        assert sum(cells[name].horizon for name in schedule) == 3_456
        total += len(schedule)
    assert total == 570


def test_full_confirmation_episode_budget_is_locked():
    cells = environment_cells()
    total = sum(
        len(training_schedule(regime, seed, 480_384, 240_192, cells))
        for regime in REGIMES
        for seed in FULL_TRAIN_SEEDS
    )
    assert total == 792_300


def _decision_inputs(candidate_scores: list[float]):
    score_rows = [
        {
            "row_type": "training_seed",
            "training_regime": regime,
            "train_seed": seed,
            "step_budget": max(FULL_STEP_BUDGETS),
            "stress_max_relative_regret": (
                0.5 if regime == CONTROL_REGIME else candidate_scores[index]
            ),
        }
        for regime in REGIMES
        for index, seed in enumerate(FULL_TRAIN_SEEDS)
    ]
    summary_rows = [
        {
            "training_regime": regime,
            "train_seed": seed,
            "scenario": "in_distribution",
            "step_budget": max(FULL_STEP_BUDGETS),
            "objective_mean": 10.0 if regime == CONTROL_REGIME else 10.4,
        }
        for regime in REGIMES
        for seed in FULL_TRAIN_SEEDS
    ]
    return score_rows, summary_rows


def test_confirmation_gate_requires_eight_directional_wins():
    scores, summaries = _decision_inputs([0.2] * 10)
    rows, decision = confirmation_results(
        scores, summaries, max(FULL_STEP_BUDGETS), True
    )
    assert decision["confirmation_passed"] is True
    assert decision["confirmed_regime"] == CANDIDATE_REGIME
    aggregate = [row for row in rows if row["row_type"] == "across_training_seeds"][0]
    assert aggregate["stress_better_count"] == 10
    assert aggregate["nominal_within_10pct_count"] == 10
    assert aggregate["stress_delta_ci95_high"] < 0

    scores, summaries = _decision_inputs([0.4] * 7 + [0.6] * 3)
    _, decision = confirmation_results(
        scores, summaries, max(FULL_STEP_BUDGETS), True
    )
    assert decision["criteria"]["stress_delta_mean"] < 0
    assert decision["criteria"]["stress_better_count"] == 7
    assert decision["confirmation_passed"] is False


def test_confirmation_runner_rejects_nonempty_output_directory(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "sentinel.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(argparse.Namespace(profile="smoke", device="cpu", output_dir=output))


def test_confirmation_smoke_completes_all_audits(tmp_path):
    output = tmp_path / "smoke"
    run(argparse.Namespace(profile="smoke", device="cpu", output_dir=output))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["expected_counts"] == manifest["actual_counts"] == {
        "evaluation_rows": 72,
        "checkpoints": 6,
        "training_trajectories": 2,
        "training_episodes": 570,
    }
    assert all(
        value for value in summary["audits"].values() if isinstance(value, bool)
    )
    assert summary["confirmation_result"]["gate_applicable"] is False
    assert summary["confirmation_result"]["confirmation_passed"] is None
    assert len(manifest["adaptation_diagnostics"]) == 1
    assert manifest["adaptation_diagnostics"][0]["anchor_lambda"] == 0.0
    for name in (
        "regime_registry.csv",
        "training_schedule.csv",
        "training_progress.csv",
        "adaptation_diagnostics.csv",
        "episodes.csv",
        "budget_summary.csv",
        "regret_scores.csv",
        "primary_results.csv",
        "confirmation_result.json",
    ):
        assert (output / name).is_file()
