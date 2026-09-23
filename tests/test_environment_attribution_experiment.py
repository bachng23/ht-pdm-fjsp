from __future__ import annotations

from pathlib import Path

import numpy as np

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.environment_attribution_experiment import (
    GRAPH_VARIANTS,
    feasible_action_sets,
    graph_snapshot,
    summarize,
    top_k_intent_actions,
)
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config


ROOT = Path(__file__).resolve().parents[1]


def test_top_k_intent_respects_masks_and_stable_ties() -> None:
    q_values = np.asarray([[1.0, 5.0, 5.0, -2.0], [3.0, 2.0, 9.0, 4.0]])
    masks = np.asarray([[True, True, True, False], [True, False, False, True]])
    greedy, selected = top_k_intent_actions(q_values, masks, top_k=2)
    assert greedy.tolist() == [1, 3]
    assert selected == ((1, 2), (3, 0))


def test_production_only_graph_removes_technician_edges() -> None:
    base = BenchmarkConfig.from_json(
        ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"
    )
    config = build_condition_config(base, topology="two_specialists", multiplier=2.0)
    env = MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
    observation, _ = env.reset(seed=61_990)
    action_sets = feasible_action_sets(observation)
    all_graph, _ = graph_snapshot(
        env,
        action_sets,
        include_production=True,
        include_technician=True,
    )
    production_graph, _ = graph_snapshot(
        env,
        action_sets,
        include_production=True,
        include_technician=False,
    )
    assert sum(production_graph["component_sizes"]) == 6
    assert production_graph["technician_edge_count"] == 0
    assert production_graph["total_edge_count"] <= all_graph["total_edge_count"]
    env.close()


def test_summary_selects_only_graph_passing_topology_and_prediction() -> None:
    episodes = []
    graph_rows = []
    for condition in ("cooperative_iql", "qmix"):
        episodes.append(
            {
                "condition": condition,
                "seed": 1,
                "joint_steps": 1,
                "coordination_audit_failures": 0,
            }
        )
        for variant in GRAPH_VARIANTS:
            passing = variant == "policy_intent_top2"
            graph_rows.append(
                {
                    "condition": condition,
                    "graph_variant": variant,
                    "seed": 1,
                    "joint_step": 0,
                    "largest_component_size": 3 if passing else 5,
                    "has_component_size_2_or_3": int(passing),
                    "fully_connected": 0,
                    "mean_component_size_including_singletons": 1.5,
                    "mean_nontrivial_component_size": 3.0,
                    "isolated_agent_count": 3,
                    "graph_density": 0.2,
                    "edge_true_positive": 1,
                    "edge_false_positive": 1 if passing else 9,
                    "edge_false_negative": 0,
                    "predicted_conflict_epoch": 1,
                    "actual_conflict_epoch": 1,
                }
            )
    summary = summarize(episodes, graph_rows, expected_episodes=2)
    assert summary["gate"]["passed"]
    decisions = summary["locked_architecture_rule"]["decisions"]
    assert decisions["policy_intent_top2"]["supports_hard_dynamic_subteams"]
    assert not decisions["all_feasible"]["supports_hard_dynamic_subteams"]
    assert summary["locked_architecture_rule"][
        "any_refined_graph_supports_hard_dynamic_subteams"
    ]
