from __future__ import annotations

import pytest

from ht_pdm_fjsp.checkpoint_selection import (
    compare_selected_with_final,
    select_checkpoints,
)


def _validation_row(train_seed: int, step: int, seed: int, objective: float):
    return {
        "split": "checkpoint_validation",
        "train_seed": train_seed,
        "checkpoint_steps": step,
        "seed": seed,
        "objective": objective,
    }


def _test_row(train_seed: int, seed: int, objective: float):
    return {
        "split": "test",
        "policy": "maskable_ppo",
        "train_seed": train_seed,
        "seed": seed,
        "objective": objective,
        "makespan": objective / 2,
        "total_tardiness": 1.0,
        "failures": 1.0,
        "preventive_maintenance": 1.0,
        "corrective_maintenance": 1.0,
        "total_cost": objective / 2,
    }


def test_selects_lowest_validation_mean_and_earlier_tie() -> None:
    rows = [
        _validation_row(10, 100, 20, 8.0),
        _validation_row(10, 100, 21, 10.0),
        _validation_row(10, 200, 20, 9.0),
        _validation_row(10, 200, 21, 9.0),
        _validation_row(11, 100, 20, 12.0),
        _validation_row(11, 100, 21, 14.0),
        _validation_row(11, 200, 20, 10.0),
        _validation_row(11, 200, 21, 12.0),
    ]
    selected = select_checkpoints(
        rows, train_seeds=(10, 11), validation_seeds=(20, 21)
    )
    assert [row["checkpoint_steps"] for row in selected] == [100, 200]
    assert [row["mean_validation_objective"] for row in selected] == [9.0, 11.0]


def test_rejects_incomplete_validation_panel() -> None:
    rows = [_validation_row(10, 100, 20, 8.0)]
    with pytest.raises(ValueError, match="Incomplete validation panel"):
        select_checkpoints(rows, train_seeds=(10,), validation_seeds=(20, 21))


def test_selected_vs_final_is_paired_by_training_and_test_seed() -> None:
    selected = [_test_row(10, 30, 8.0), _test_row(10, 31, 10.0)]
    final = [_test_row(10, 31, 13.0), _test_row(10, 30, 9.0)]
    result = compare_selected_with_final(selected, final)
    delta = result["objective_delta_across_training_seeds"]
    assert delta["mean"] == -2.0
    assert result["pooled_objective_win_rate"] == 1.0


def test_selected_vs_final_rejects_unpaired_episode() -> None:
    selected = [_test_row(10, 30, 8.0)]
    final = [_test_row(10, 31, 9.0)]
    with pytest.raises(ValueError, match="Missing paired final PPO episode"):
        compare_selected_with_final(selected, final)
