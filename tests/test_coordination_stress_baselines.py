from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from ht_pdm_fjsp.committed_conflict_rl import CommittedConflictCTDEEnv
from ht_pdm_fjsp.conflict_consequence import (
    ConflictConsequenceEnv,
    build_cell_config,
)
from ht_pdm_fjsp.coordination_stress_baselines import (
    CELLS,
    SCREEN_POLICIES,
    paired_interval,
    random_feasible_episode,
    run,
    summarize_screen,
)
from ht_pdm_fjsp.parallel_maintenance import ParallelMaintenanceConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "parallel_maintenance.json"


@pytest.mark.parametrize("cell", CELLS)
def test_wrapper_matches_kernel_in_each_screen_cell(cell) -> None:
    base = ParallelMaintenanceConfig.from_json(CONFIG)
    raw = ConflictConsequenceEnv(build_cell_config(base, cell), cell)
    wrapped = CommittedConflictCTDEEnv(base, cell)
    raw.reset(seed=27)
    observation, _ = wrapped.reset(seed=27)
    rng = np.random.default_rng(123)
    for _ in range(25):
        actions = np.asarray([
            rng.choice(np.flatnonzero(observation["action_masks"][agent]))
            for agent in range(wrapped.num_agents)
        ])
        raw_reward, raw_done, raw_info = raw.step(actions)
        observation, reward, done, truncated, info = wrapped.step(actions)
        assert reward == pytest.approx(raw_reward)
        assert done == raw_done
        assert not truncated
        assert info["objective"] == pytest.approx(raw_info["objective"])
        assert np.array_equal(observation["action_masks"], raw.action_masks())


def test_random_feasible_is_reproducible_and_safe() -> None:
    base = ParallelMaintenanceConfig.from_json(CONFIG)
    cell = CELLS[1]
    config = build_cell_config(base, cell)
    first = random_feasible_episode(config, cell, 65_001)
    second = random_feasible_episode(config, cell, 65_001)
    assert first == second
    assert first["return_objective_error"] <= 1e-6
    assert first["duplicate_technician_assignments"] == 0
    assert first["proposal_conflicts"] == 0
    assert first["invalid_executions"] == 0


def test_screen_selection_prefers_largest_qualifying_relative_gap() -> None:
    rows = []
    for cell, gap in zip(CELLS, (10.0, 20.0, 30.0), strict=True):
        for seed in (1, 2, 3):
            for policy in SCREEN_POLICIES:
                objective = (
                    100.0 if policy == "coordinated_preventive"
                    else 100.0 + gap if policy == "independent_preventive"
                    else 200.0
                )
                rows.append({
                    "cell_id": cell.cell_id,
                    "policy": policy,
                    "seed": seed,
                    "objective": objective,
                    "conflict_steps": 5 if policy == "independent_preventive" else 0,
                    "proposal_conflicts": 5 if policy == "independent_preventive" else 0,
                    "return_objective_error": 0.0,
                    "invalid_executions": 0,
                    "duplicate_machine_assignments": 0,
                    "duplicate_technician_assignments": 0,
                    **{metric: 0 for metric in (
                        "production", "downtime", "failures", "preventive",
                        "corrective", "waiting", "rejected_requests",
                        "committed_wait_steps", "rework", "workload_imbalance",
                        "missed_windows", "overdue_steps",
                    )},
                })
    result = summarize_screen(rows, (1, 2, 3), profile="smoke")
    assert result["gate"]["passed"]
    assert result["selected_cell"] == CELLS[2].cell_id


def test_training_seed_interval_uses_t9_critical_value() -> None:
    values = [float(index) for index in range(10)]
    interval = paired_interval(values)
    assert interval["mean"] == pytest.approx(4.5)
    assert interval["upper"] - interval["mean"] == pytest.approx(
        2.262157 * np.std(values, ddof=1) / np.sqrt(10)
    )


def test_smoke_runner_writes_audited_selection_and_models(tmp_path: Path) -> None:
    output = tmp_path / "coordination_smoke"
    run(argparse.Namespace(
        config=str(CONFIG), profile="smoke", device="cpu", output_dir=str(output)
    ))
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["smoke_selection_override"]
    assert manifest["selected_cell"] == CELLS[0].cell_id
    assert manifest["screen_episode_count"] == 3 * 3 * 3
    assert manifest["learned_episode_count"] == 4 * 1 * 2 * 2
    assert manifest["heuristic_episode_count"] == 3 * 2
    assert manifest["decision_count"] == 4 * 168
    assert summary["gate"]["passed"]
    assert not manifest["sealed_test_evaluated"]


def test_full_screen_only_outcome_skips_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ht_pdm_fjsp.coordination_stress_baselines as experiment

    tiny = experiment.profile_settings("smoke")
    monkeypatch.setattr(experiment, "profile_settings", lambda _: tiny)
    original = experiment.summarize_screen

    def no_selection(*args, **kwargs):
        result = original(*args, **kwargs)
        result["selected_cell"] = None
        return result

    monkeypatch.setattr(experiment, "summarize_screen", no_selection)
    output = tmp_path / "screen_only"
    run(argparse.Namespace(
        config=str(CONFIG), profile="full", device="cpu", output_dir=str(output)
    ))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["outcome"] == "SCREEN_ONLY"
    assert not manifest["completed_training_cells"]
    assert not (output / "learned_episodes.csv").exists()
    assert (output / "summary.json").is_file()


def test_selected_stress_cell_trains_under_full_selection_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ht_pdm_fjsp.coordination_stress_baselines as experiment

    tiny = experiment.profile_settings("smoke")
    monkeypatch.setattr(experiment, "profile_settings", lambda _: tiny)
    original = experiment.summarize_screen

    def force_stress_selection(*args, **kwargs):
        result = original(*args, **kwargs)
        result["selected_cell"] = CELLS[2].cell_id
        return result

    monkeypatch.setattr(experiment, "summarize_screen", force_stress_selection)
    output = tmp_path / "stress_selected"
    run(argparse.Namespace(
        config=str(CONFIG), profile="full", device="cpu", output_dir=str(output)
    ))
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["selected_cell"] == CELLS[2].cell_id
    assert manifest["learned_episode_count"] == 16
    assert json.loads((output / "resolved_config.json").read_text()) != json.loads(
        CONFIG.read_text()
    )


def test_sealed_seed_rejected_before_artifact_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ht_pdm_fjsp.coordination_stress_baselines as experiment

    original = experiment.profile_settings("smoke")
    monkeypatch.setattr(experiment, "profile_settings", lambda _: {
        **original, "evaluation_seeds": (63_200,)
    })
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="sealed future test panel"):
        run(argparse.Namespace(
            config=str(CONFIG), profile="smoke", device="cpu", output_dir=str(output)
        ))
    assert not output.exists()
