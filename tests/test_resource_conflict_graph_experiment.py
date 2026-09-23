from __future__ import annotations

from pathlib import Path

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.resource_conflict_graph_experiment import (
    CONDITIONS,
    connected_components,
    resource_conflict_graph,
    summarize,
)
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config


ROOT = Path(__file__).resolve().parents[1]


def test_connected_components_include_isolated_agents() -> None:
    assert connected_components(6, {(0, 2), (1, 4), (2, 3)}) == (
        (0, 2, 3),
        (1, 4),
        (5,),
    )


def test_resource_graph_partitions_all_six_machine_agents() -> None:
    base = BenchmarkConfig.from_json(
        ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"
    )
    config = build_condition_config(base, topology="two_specialists", multiplier=2.0)
    env = MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
    observation, _ = env.reset(seed=61_990)
    graph = resource_conflict_graph(env, observation)
    sizes = graph["component_sizes"]
    assert sum(sizes) == 6
    assert graph["largest_component_size"] == max(sizes)
    assert 0.0 <= graph["graph_density"] <= 1.0
    env.close()


def test_summary_applies_locked_qmix_architecture_rule() -> None:
    episodes = []
    epochs = []
    for condition in CONDITIONS:
        for seed in (1, 2):
            episodes.append(
                {
                    "condition": condition,
                    "seed": seed,
                    "joint_steps": 2,
                    "coordination_audit_failures": 0,
                    "mean_largest_component_size": 2.5,
                    "fraction_epochs_with_component_size_2_or_3": 1.0,
                    "fraction_epochs_fully_connected": 0.0,
                }
            )
            epochs.extend(
                {
                    "condition": condition,
                    "seed": seed,
                    "largest_component_size": largest,
                    "has_component_size_2_or_3": 1,
                    "fully_connected": 0,
                    "mean_component_size_including_singletons": 1.5,
                    "mean_nontrivial_component_size": float(largest),
                    "isolated_agent_count": 3,
                    "graph_density": 0.2,
                }
                for largest in (2, 3)
            )
    summary = summarize(episodes, epochs, expected_episodes=4)
    assert summary["gate"]["passed"]
    assert summary["locked_dynamic_subteam_rule"]["supports_dynamic_subteams"]
    assert (
        summary["conditions"]["qmix"]["epoch_weighted"]
        ["mean_largest_connected_component_size"]
        == 2.5
    )
