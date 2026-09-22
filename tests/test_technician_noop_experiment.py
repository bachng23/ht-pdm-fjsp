from __future__ import annotations

import argparse
import json
from pathlib import Path

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.technician_conflict_experiment import choose_joint_actions
from ht_pdm_fjsp.technician_noop_experiment import (
    NOOP_018,
    _conflict_sources,
    evaluate_episode,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def test_voluntary_preventive_conflict_is_classified_exactly_once() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    env = MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
    env.reset(seed=1)
    for machine_id in ("M1", "M2"):
        env.core.machines[machine_id].effective_age = 80.0
        env.core.machines[machine_id].preventive_armed = True
    observation = env._ctde_observation(env.core._observation())
    reasons: list[str] = []
    actions = choose_joint_actions(
        env, observation, preventive_threshold=0.18, reasons=reasons
    )
    sources = _conflict_sources(env, actions, reasons)
    _, _, _, _, info = env.step(actions)
    assert sources["preventive_preventive_conflicts"] == 1
    assert sources["forced_technician_conflicts"] == 0
    assert sum(sources.values()) == info["coordination"]["technician_conflicts"]


def test_safe_noop_episode_has_complete_conflict_source_accounting() -> None:
    config = BenchmarkConfig.from_json(CONFIG_PATH)
    episode, decisions, coordination = evaluate_episode(
        config,
        condition=NOOP_018,
        seed=61_410,
        threshold=0.18,
        wait_policy="safe_noop",
    )
    source_keys = (
        "forced_technician_conflicts",
        "preventive_preventive_conflicts",
        "corrective_corrective_conflicts",
        "preventive_corrective_conflicts",
        "other_technician_conflicts",
    )
    assert decisions
    assert sum(episode[key] for key in source_keys) == episode[
        "technician_conflicts"
    ]
    assert episode["forced_preventive_proposals"] == 0
    assert episode["forced_technician_conflicts"] == 0
    assert coordination["invalid_executions"] == 0


def test_noop_smoke_writes_completed_non_overwriting_artifacts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "mk01_safe_noop_smoke_20260923T000000Z"
    run(
        argparse.Namespace(
            profile="smoke",
            config=str(CONFIG_PATH),
            seeds="61410:61412",
            output_dir=str(output),
        )
    )
    manifest = json.loads((output / "technician_noop_manifest.json").read_text())
    summary = json.loads((output / "technician_noop_summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 8
    assert manifest["future_test_panel_opened"] is False
    assert summary["per_condition"][NOOP_018]["totals"][
        "forced_preventive_proposals"
    ] == 0
    assert {
        "technician_noop_episodes.csv",
        "technician_noop_episodes.partial.csv",
        "technician_noop_decisions.csv",
        "technician_noop_decisions.partial.csv",
        "technician_noop_coordination.csv",
        "technician_noop_coordination.partial.csv",
        "technician_noop_summary.json",
    }.issubset(set(manifest["output_files"]))
