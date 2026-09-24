from __future__ import annotations

import argparse
import json
from pathlib import Path

from ht_pdm_fjsp.parallel_maintenance_experiment import (
    HEURISTICS,
    LEARNED,
    run,
    summarize,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "parallel_maintenance.json"


def test_smoke_screen_trains_reloads_evaluates_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "parallel_maintenance_smoke_20260924T000000Z"
    summary = run(
        argparse.Namespace(
            config=str(CONFIG_PATH),
            profile="smoke",
            training_seeds="77100",
            development_seeds="63190:63192",
            total_timesteps=100,
            device="cpu",
            output_dir=str(output),
        )
    )
    manifest = json.loads(
        (output / "parallel_maintenance_manifest.json").read_text()
    )
    assert manifest["status"] == "COMPLETED"
    assert manifest["sealed_test_evaluated"] is False
    assert manifest["episode_count"] == 16
    assert manifest["training_runs"] == 3
    assert summary["gate"]["passed"]
    for filename in (
        "resolved_config.json",
        "episodes.partial.csv",
        "episodes.csv",
        "decisions.csv",
        "coordination.csv",
        "training_progress.csv",
        "summary.json",
    ):
        assert (output / filename).is_file()
    assert (output / "masked_ppo" / "77100" / "model.zip").is_file()
    assert (output / "iql" / "77100" / "model.pt").is_file()
    assert (output / "qmix" / "77100" / "model.pt").is_file()


def test_summary_detects_failed_audit() -> None:
    rows = []
    for condition in (*HEURISTICS, *LEARNED):
        rows.append(
            {
                "condition": condition,
                "train_seed": "" if condition in HEURISTICS else 1,
                "seed": 10,
                "objective": 10.0,
                "production": 5.0,
                "downtime": 2.0,
                "failures": 1,
                "preventive": 1,
                "corrective": 1,
                "waiting": 1.0,
                "proposal_conflicts": 0,
                "conflict_steps": 0,
                "rework": 0,
                "early_pm": 0.0,
                "workload_imbalance": 0.0,
                "inference_seconds": 0.01,
                "return_objective_error": 0.0,
                "invalid_executions": int(condition == "qmix"),
                "duplicate_machine_assignments": 0,
                "duplicate_technician_assignments": 0,
            }
        )
    result = summarize(
        rows,
        expected_episodes=len(rows),
        training_seed_count=1,
        profile="smoke",
    )
    assert not result["audits"]["no_invalid_executions"]
    assert not result["gate"]["passed"]
