from __future__ import annotations

import argparse
import json
from pathlib import Path

from ht_pdm_fjsp.marl_budget_screening import (
    SCREENING_CONDITIONS,
    run,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def test_smoke_budget_screen_writes_five_checkpoints_per_algorithm(
    tmp_path: Path,
) -> None:
    output = tmp_path / "marl_budget_screen_smoke_20260923T000000Z"
    run(
        argparse.Namespace(
            config=str(CONFIG_PATH),
            profile="smoke",
            train_seed=76100,
            evaluation_seeds="61990:61992",
            total_timesteps=200,
            device="cpu",
            output_dir=str(output),
        )
    )
    manifest = json.loads(
        (output / "marl_budget_screening_manifest.json").read_text()
    )
    summary = json.loads(
        (output / "marl_budget_screening_summary.json").read_text()
    )
    assert manifest["status"] == "COMPLETED"
    assert manifest["machine_count"] == 6
    assert manifest["conditions"] == list(SCREENING_CONDITIONS)
    assert manifest["episode_count"] == 20
    assert manifest["value_settings"]["n_envs"] == 4
    assert "mappo_settings" not in manifest
    assert summary["selected_common_replicated_run_budget"] is None
    assert summary["gate"]["passed"]
    for condition in SCREENING_CONDITIONS:
        assert len(list((output / condition / "checkpoints").glob("*.pt"))) == 5
        for point in summary["curves"][condition]["points"]:
            metrics = point["constraint_metrics"]
            assert metrics["joint_steps"] > 0
            assert metrics["precedence_blocked_operations_per_joint_step"] >= 0
            assert metrics["strict_three_way_steps_per_1000"] >= 0


def test_full_summary_selects_budget_from_objective_makespan_and_failures() -> None:
    checkpoints = (100_000, 200_000, 300_000, 400_000, 500_000)
    episode_rows = []
    decision_rows = []
    coordination_rows = []
    for condition in SCREENING_CONDITIONS:
        for checkpoint, objective, makespan, failures in zip(
            checkpoints,
            (120.0, 103.0, 101.0, 100.0, 102.0),
            (115.0, 104.0, 101.0, 100.0, 103.0),
            (1.0, 0.5, 0.4, 0.4, 0.5),
            strict=True,
        ):
            episode_rows.append(
                {
                    "condition": condition,
                    "checkpoint_steps": checkpoint,
                    "objective": objective,
                    "makespan": makespan,
                    "maintenance_wait_time": 1.0,
                    "failures": failures,
                    "three_way_steps": 0,
                    "episode_has_three_way": 0,
                }
            )
            decision_rows.append(
                {
                    "condition": condition,
                    "checkpoint_steps": checkpoint,
                    "production_conflicts": 1,
                    "technician_conflicts": 0,
                    "precedence_blocked_operations": 2,
                    "maintenance_wait_delta": 0.5,
                    "three_way": 0,
                }
            )
            coordination_rows.append(
                {
                    "condition": condition,
                    "checkpoint_steps": checkpoint,
                    "invalid_executions": 0,
                    "duplicate_operation_executions": 0,
                    "duplicate_technician_executions": 0,
                }
            )
    summary = summarize(
        episode_rows,
        decision_rows,
        coordination_rows,
        checkpoint_targets=checkpoints,
        evaluation_count=1,
        profile="full",
    )
    assert summary["selected_common_replicated_run_budget"] == 200_000
    assert summary["gate"]["passed"]
    assert "three_way_incidence_within_absolute_fraction_of_500k" not in summary[
        "locked_plateau_rule"
    ]
