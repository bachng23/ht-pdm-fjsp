from __future__ import annotations

from ht_pdm_fjsp.qplex_soft_graph_experiment import (
    CONDITIONS,
    QPLEX_GRAPHS,
    QPLEX_POLICY_INTENT,
    summarize,
)


def test_full_summary_applies_locked_promotion_rule() -> None:
    train_seeds = (76_000, 77_000, 78_000, 79_000, 80_000)
    evaluation_seeds = (61_700,)
    checkpoints = (100_000, 200_000, 300_000)
    objectives = {
        "qmix": 101.0,
        "qplex_null_graph": 100.0,
        "qplex_all_feasible": 99.0,
        QPLEX_POLICY_INTENT: 95.0,
    }
    episodes = []
    decisions = []
    coordination = []
    graph_rows = []
    for condition in CONDITIONS:
        for train_seed in train_seeds:
            for checkpoint in checkpoints:
                episode = {
                    "condition": condition,
                    "train_seed": train_seed,
                    "checkpoint_steps": checkpoint,
                    "seed": evaluation_seeds[0],
                    "joint_steps": 1,
                    "objective": objectives[condition],
                    "makespan": 50.0,
                    "total_cost": 45.0,
                    "failures": 2.0,
                    "preventive_maintenance": 1.0,
                    "corrective_maintenance": 2.0,
                    "maintenance_wait_time": 0.5,
                    "maximum_rejection_streak": 1,
                    "coordination_audit_failures": 0,
                }
                episodes.append(episode)
                decisions.append(
                    {
                        "condition": condition,
                        "train_seed": train_seed,
                        "checkpoint_steps": checkpoint,
                        "proposals": 4,
                        "rejected": 1,
                        "production_conflicts": 1,
                        "technician_conflicts": 0,
                    }
                )
                coordination.append(
                    {
                        "invalid_executions": 0,
                        "duplicate_operation_executions": 0,
                        "duplicate_technician_executions": 0,
                    }
                )
                if condition in QPLEX_GRAPHS:
                    graph_rows.append(
                        {
                            "condition": condition,
                            "train_seed": train_seed,
                            "checkpoint_steps": checkpoint,
                            "seed": evaluation_seeds[0],
                            "joint_step": 0,
                            "graph_density": 0.2,
                            "largest_component_size": 2,
                            "lambda_mean": 1.0,
                            "lambda_max": 1.5,
                            "edge_true_positive": 1,
                            "edge_false_positive": 1,
                            "edge_false_negative": 0,
                        }
                    )
    reward_rows = [{"policy": policy} for policy in ("a", "b")]
    result = summarize(
        episodes,
        decisions,
        coordination,
        graph_rows,
        reward_rows,
        profile="full",
        train_seeds=train_seeds,
        evaluation_seeds=evaluation_seeds,
        checkpoints=checkpoints,
        checkpoint_files_complete=True,
    )
    assert result["integrity_gate"]["passed"]
    assert result["locked_decision"]["promote_policy_intent_soft_graph"]
    assert result["locked_decision"]["intent_winning_training_seeds"] == 5
