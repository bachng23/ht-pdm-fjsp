import argparse
import json

import numpy as np
import pytest

from ht_pdm_fjsp.passive_technician_anchor_preservation import (
    CONTROL_REGIME,
    CURRICULUM_REGIMES,
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
from ht_pdm_fjsp.passive_technician_leave_one_out import (
    SEALED_TEST_SEEDS,
    environment_cells,
)
from ht_pdm_fjsp.passive_technician_value_decomposition import ReplayBuffer


def test_protocol_seed_panels_and_interventions_are_locked():
    assert PROTOCOL_VERSION == "ra_qmix_anchor_preservation_v1"
    assert REGIMES == (
        "nominal_100",
        "fixed_87_5",
        "curriculum_replay",
        "curriculum_kl_anchor",
    )
    assert CURRICULUM_REGIMES == (
        "curriculum_replay",
        "curriculum_kl_anchor",
    )
    assert FULL_STEP_BUDGETS == (240_192, 360_288, 480_384)
    assert SMOKE_STEP_BUDGETS == (1_728, 2_592, 3_456)
    assert REGIME_SPECS["curriculum_kl_anchor"]["anchor_lambda"] == 10.0
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


def test_smoke_schedules_have_exact_phase_and_transition_counts():
    cells = environment_cells()
    expected = {
        "nominal_100": {"in_distribution": 288},
        "fixed_87_5": {
            "in_distribution": 252,
            "early_failure": 8,
            "slow_service": 8,
            "combined_pressure": 8,
        },
        "curriculum_replay": {
            "in_distribution": 270,
            "early_failure": 4,
            "slow_service": 4,
            "combined_pressure": 4,
        },
        "curriculum_kl_anchor": {
            "in_distribution": 270,
            "early_failure": 4,
            "slow_service": 4,
            "combined_pressure": 4,
        },
    }
    total_episodes = 0
    curriculum_schedules = []
    for regime in REGIMES:
        schedule = training_schedule(
            regime, SMOKE_TRAIN_SEEDS[0], 3_456, 1_728, cells
        )
        assert schedule == training_schedule(
            regime, SMOKE_TRAIN_SEEDS[0], 3_456, 1_728, cells
        )
        assert {name: schedule.count(name) for name in cells} == {
            name: expected[regime].get(name, 0) for name in cells
        }
        assert sum(cells[name].horizon for name in schedule) == 3_456
        if regime in CURRICULUM_REGIMES:
            curriculum_schedules.append(schedule)
            steps = 0
            phase_one = []
            for scenario in schedule:
                if steps < 1_728:
                    phase_one.append(scenario)
                steps += cells[scenario].horizon
            assert set(phase_one) == {"in_distribution"}
            nominal_steps = schedule.count("in_distribution") * 12
            assert nominal_steps / 3_456 == pytest.approx(0.9375)
        total_episodes += len(schedule)
    assert total_episodes == 1_128
    assert curriculum_schedules[0] == curriculum_schedules[1]


def test_full_schedule_episode_budget_is_locked():
    cells = environment_cells()
    total = sum(
        len(training_schedule(regime, seed, 480_384, 240_192, cells))
        for regime in REGIMES
        for seed in FULL_TRAIN_SEEDS
    )
    assert total == 783_960


def test_replay_resize_retains_latest_transitions_in_order():
    replay = ReplayBuffer(5, machines=1, observation_dim=1, action_dim=1)
    local = np.zeros((1, 1), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.bool_)
    for value in range(7):
        replay.add(local, masks, (0,), float(value), local, masks, False)
    resized = replay.retain_latest(3)
    assert resized.capacity == 3
    assert resized.size == 3
    assert resized.position == 0
    assert resized.rewards.tolist() == [4.0, 5.0, 6.0]


def test_primary_rule_excludes_fixed_comparator_and_selects_anchor():
    scores = {
        CONTROL_REGIME: 0.50,
        "fixed_87_5": 0.20,
        "curriculum_replay": 0.40,
        "curriculum_kl_anchor": 0.35,
    }
    nominal = {
        CONTROL_REGIME: 10.0,
        "fixed_87_5": 10.2,
        "curriculum_replay": 10.4,
        "curriculum_kl_anchor": 10.3,
    }
    score_rows = [
        {
            "row_type": "training_seed",
            "training_regime": regime,
            "train_seed": seed,
            "step_budget": max(FULL_STEP_BUDGETS),
            "stress_max_relative_regret": value,
        }
        for regime, value in scores.items()
        for seed in FULL_TRAIN_SEEDS
    ]
    summary_rows = [
        {
            "training_regime": regime,
            "train_seed": seed,
            "scenario": "in_distribution",
            "step_budget": max(FULL_STEP_BUDGETS),
            "objective_mean": value,
        }
        for regime, value in nominal.items()
        for seed in FULL_TRAIN_SEEDS
    ]
    rows, selection = primary_results(
        score_rows, summary_rows, max(FULL_STEP_BUDGETS), True
    )
    assert selection["primary_hypothesis_passed"] is True
    assert selection["selected_regime"] == "curriculum_kl_anchor"
    aggregates = {
        row["candidate_regime"]: row
        for row in rows
        if row["row_type"] == "across_training_seeds"
    }
    assert aggregates["fixed_87_5"]["selection_eligible"] is False
    assert aggregates["fixed_87_5"]["selected"] is False
    assert aggregates["curriculum_kl_anchor"]["selected"] is True


def test_runner_rejects_nonempty_output_directory(tmp_path):
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "sentinel.txt").write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(argparse.Namespace(profile="smoke", device="cpu", output_dir=output))


def test_smoke_run_completes_with_anchor_and_replay_audits(tmp_path):
    output = tmp_path / "smoke"
    run(argparse.Namespace(profile="smoke", device="cpu", output_dir=output))
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["expected_counts"] == manifest["actual_counts"] == {
        "evaluation_rows": 144,
        "checkpoints": 12,
        "training_trajectories": 4,
        "training_episodes": 1_128,
    }
    assert all(summary["audits"].values())
    assert summary["selection_result"]["gate_applicable"] is False
    assert summary["selection_result"]["selected_regime"] is None
    diagnostics = manifest["adaptation_diagnostics"]
    assert len(diagnostics) == 2
    assert all(row["phase_two_stratified_updates"] > 0 for row in diagnostics)
    assert {row["anchor_lambda"] for row in diagnostics} == {0.0, 10.0}
    for name in (
        "regime_registry.csv",
        "training_schedule.csv",
        "training_progress.csv",
        "adaptation_diagnostics.csv",
        "episodes.csv",
        "budget_summary.csv",
        "regret_scores.csv",
        "primary_results.csv",
        "selection_result.json",
    ):
        assert (output / name).is_file()
