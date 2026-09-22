from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config
from ht_pdm_fjsp.three_way_marl_experiment import (
    CONDITIONS,
    precedence_blocked_operations,
    run,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def _config() -> BenchmarkConfig:
    return build_condition_config(
        BenchmarkConfig.from_json(CONFIG_PATH),
        topology="two_specialists",
        multiplier=2.0,
    )


def test_three_way_definition_detects_both_conflicts_and_precedence() -> None:
    env = MachineAgentsCTDEEnv(_config(), wait_policy="safe_noop")
    env.reset(seed=1)
    for machine_id in ("M1", "M2"):
        env.core.machines[machine_id].effective_age = 80.0
    observation = env._ctde_observation(env.core._observation())
    production: dict[tuple[str, int], list[tuple[int, int]]] = defaultdict(list)
    technicians: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for agent, catalog in enumerate(env.local_action_catalogs):
        for local_action, global_action in enumerate(catalog):
            if global_action is None or not observation["action_masks"][agent, local_action]:
                continue
            descriptor = env.core.actions[global_action]
            if descriptor.kind == "production":
                production[(str(descriptor.job_id), int(descriptor.operation_index))].append(
                    (agent, local_action)
                )
            elif descriptor.kind == "preventive":
                technicians[str(descriptor.technician_id)].append((agent, local_action))
    maintenance_pair = next(values for values in technicians.values() if len(values) >= 2)
    occupied = {agent for agent, _ in maintenance_pair[:2]}
    production_pair = next(
        values for values in production.values()
        if len([item for item in values if item[0] not in occupied]) >= 2
    )
    production_pair = [item for item in production_pair if item[0] not in occupied][:2]
    actions = np.zeros(env.num_agents, dtype=np.int64)
    for agent, local_action in (*maintenance_pair[:2], *production_pair):
        actions[agent] = local_action
    assert precedence_blocked_operations(env) > 0
    _, _, _, _, info = env.step(actions)
    assert info["coordination"]["production_conflicts"] == 1
    assert info["coordination"]["technician_conflicts"] == 1


def test_smoke_run_writes_all_algorithm_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "three_way_marl_smoke_20260923T000000Z"
    run(
        argparse.Namespace(
            config=str(CONFIG_PATH),
            profile="smoke",
            train_seeds="75000",
            evaluation_seeds="61990:61992",
            total_timesteps=64,
            device="cpu",
            output_dir=str(output),
        )
    )
    manifest = json.loads((output / "three_way_marl_manifest.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["episode_count"] == 6
    assert manifest["future_test_panel_opened"] is False
    assert set(manifest["conditions"]) == set(CONDITIONS)
    assert {
        "three_way_marl_episodes.csv",
        "three_way_marl_decisions.csv",
        "three_way_marl_coordination.csv",
        "three_way_marl_summary.json",
    }.issubset(set(manifest["output_files"]))
