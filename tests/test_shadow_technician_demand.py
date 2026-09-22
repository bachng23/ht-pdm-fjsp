from __future__ import annotations

import argparse
import json
from pathlib import Path

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.shadow_technician_demand import (
    SHADOW_018,
    _shadow_onsets,
    evaluate_episode,
    reconstruct_shadow_requests,
    run,
)
from ht_pdm_fjsp.technician_noop_experiment import (
    NOOP_018,
    evaluate_episode as evaluate_noop_episode,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def test_shadow_reconstructs_corrective_request_hidden_by_busy_technician() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    env = MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
    env.reset(seed=1)
    env.core.machines["M1"].status = "waiting_corrective"
    env.core.technicians["T1"].status = "busy"
    observation = env._ctde_observation(env.core._observation())
    requests = reconstruct_shadow_requests(
        env, observation, preventive_threshold=None
    )
    request = next(item for item in requests if item.machine_id == "M1")
    assert request.kind == "corrective"
    assert request.technician_id == "T1"
    assert request.all_qualified_technicians_busy


def test_shadow_onsets_are_deduplicated_across_consecutive_snapshots() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    env = MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
    env.reset(seed=2)
    env.core.machines["M1"].status = "waiting_corrective"
    env.core.technicians["T1"].status = "busy"
    observation = env._ctde_observation(env.core._observation())
    requests = reconstruct_shadow_requests(
        env, observation, preventive_threshold=None
    )
    first, active, suppressed, contested = _shadow_onsets(
        requests,
        previous_active=set(),
        previous_suppressed=set(),
        previous_contested=set(),
    )
    second, _, _, _ = _shadow_onsets(
        requests,
        previous_active=active,
        previous_suppressed=suppressed,
        previous_contested=contested,
    )
    assert first["shadow_request_onsets"] >= 1
    assert first["shadow_busy_suppression_onsets"] >= 1
    assert second["shadow_request_onsets"] == 0
    assert second["shadow_busy_suppression_onsets"] == 0


def test_shadow_episode_preserves_safe_noop_audits() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    episode, decisions, coordination = evaluate_episode(
        config,
        condition=SHADOW_018,
        seed=61_420,
        threshold=0.18,
    )
    assert decisions
    assert episode["forced_preventive_proposals"] == 0
    assert episode["shadow_request_onsets"] <= episode[
        "shadow_request_snapshots"
    ]
    assert coordination["invalid_executions"] == 0
    assert coordination["duplicate_operation_executions"] == 0
    assert coordination["duplicate_technician_executions"] == 0


def test_shadow_instrumentation_does_not_change_safe_noop_trajectory() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    shadow, shadow_decisions, _ = evaluate_episode(
        config,
        condition=SHADOW_018,
        seed=61_420,
        threshold=0.18,
    )
    baseline, baseline_decisions, _ = evaluate_noop_episode(
        config,
        condition=NOOP_018,
        seed=61_420,
        threshold=0.18,
        wait_policy="safe_noop",
    )
    assert len(shadow_decisions) == len(baseline_decisions)
    for key in (
        "episode_return",
        "objective",
        "makespan",
        "failures",
        "preventive_maintenance",
        "corrective_maintenance",
        "joint_steps",
        "production_conflicts",
        "technician_conflicts",
        "rejected",
    ):
        assert shadow[key] == baseline[key]


def test_shadow_smoke_writes_complete_artifact_set(tmp_path: Path) -> None:
    output = tmp_path / "mk01_shadow_demand_smoke_20260923T000000Z"
    run(
        argparse.Namespace(
            profile="smoke",
            config=str(CONFIG_PATH),
            seeds="61420:61422",
            output_dir=str(output),
        )
    )
    manifest = json.loads((output / "shadow_demand_manifest.json").read_text())
    summary = json.loads((output / "shadow_demand_summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 6
    assert manifest["future_test_panel_opened"] is False
    assert summary["per_condition"][SHADOW_018]["audit"][
        "request_kinds_sum_to_request_onsets"
    ]
    assert {
        "shadow_demand_episodes.csv",
        "shadow_demand_episodes.partial.csv",
        "shadow_demand_decisions.csv",
        "shadow_demand_decisions.partial.csv",
        "shadow_demand_coordination.csv",
        "shadow_demand_coordination.partial.csv",
        "shadow_demand_summary.json",
    }.issubset(set(manifest["output_files"]))
