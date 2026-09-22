from __future__ import annotations

import argparse
import json
from pathlib import Path

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.technician_conflict_experiment import (
    REACTIVE,
    THRESHOLD_018,
    choose_joint_actions,
    evaluate_episode,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def _descriptor(env: MachineAgentsCTDEEnv, agent: int, action: int):
    global_action = env.local_to_global(agent, action)
    assert global_action is not None
    return env.core.actions[global_action]


def test_threshold_policy_creates_a_measurable_joint_technician_conflict() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    env = MachineAgentsCTDEEnv(config)
    env.reset(seed=1)
    for machine_id in ("M1", "M2"):
        env.core.machines[machine_id].effective_age = 80.0
        env.core.machines[machine_id].preventive_armed = True
    observation = env._ctde_observation(env.core._observation())

    actions = choose_joint_actions(
        env, observation, preventive_threshold=0.18
    )
    first = _descriptor(env, 0, int(actions[0]))
    second = _descriptor(env, 1, int(actions[1]))
    assert (first.kind, first.technician_id) == ("preventive", "T1")
    assert (second.kind, second.technician_id) == ("preventive", "T1")

    _, _, _, _, info = env.step(actions)
    assert info["coordination"]["technician_conflicts"] >= 1
    assert info["coordination"]["duplicate_technician_executions"] == 0


def test_reactive_policy_does_not_voluntarily_select_preventive() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    env = MachineAgentsCTDEEnv(config)
    observation, _ = env.reset(seed=2)
    env.core.machines["M1"].effective_age = 35.0
    env.core.machines["M1"].preventive_armed = True
    observation = env._ctde_observation(env.core._observation())

    reasons: list[str] = []
    actions = choose_joint_actions(
        env, observation, preventive_threshold=None, reasons=reasons
    )
    assert _descriptor(env, 0, int(actions[0])).kind == "production"
    assert "threshold_preventive" not in reasons


def test_episode_records_decisions_and_passes_coordination_audit() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    episode, decisions, coordination = evaluate_episode(
        config,
        condition=THRESHOLD_018,
        seed=61_400,
        threshold=0.18,
    )
    assert decisions
    assert episode["joint_steps"] == len(decisions)
    assert episode["maintenance_proposals"] == (
        episode["proposed_preventive"] + episode["proposed_corrective"]
    )
    assert episode["proposed_preventive"] == (
        episode["threshold_preventive_proposals"]
        + episode["forced_preventive_proposals"]
    )
    assert coordination["invalid_executions"] == 0
    assert coordination["duplicate_operation_executions"] == 0
    assert coordination["duplicate_technician_executions"] == 0


def test_smoke_run_writes_completed_timestamp_ready_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "technician_conflict_smoke_20260923T000000Z"
    run(
        argparse.Namespace(
            profile="smoke",
            config=str(CONFIG_PATH),
            seeds="61400:61402",
            output_dir=str(output),
        )
    )
    manifest = json.loads(
        (output / "technician_conflict_manifest.json").read_text()
    )
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 6
    assert manifest["future_test_panel_opened"] is False
    assert {
        "technician_conflict_episodes.csv",
        "technician_conflict_episodes.partial.csv",
        "technician_conflict_decisions.csv",
        "technician_conflict_decisions.partial.csv",
        "technician_conflict_coordination.csv",
        "technician_conflict_coordination.partial.csv",
        "technician_conflict_summary.json",
    }.issubset(set(manifest["output_files"]))
    summary = json.loads((output / "technician_conflict_summary.json").read_text())
    assert set(summary["per_condition"]) == {
        REACTIVE,
        THRESHOLD_018,
        "threshold_005_joint_spt",
    }
    contrast = summary["paired_contrasts"]["threshold_018_minus_reactive"]
    assert contrast["seed_count"] == 2
    assert contrast["degrees_of_freedom"] == 1
    assert set(contrast["metrics"]) >= {
        "technician_conflicts",
        "maintenance_proposals",
        "objective",
    }
