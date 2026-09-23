"""Train a capacity-matched QPLEX soft resource-graph ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.advanced_baselines import (
    HealthThresholdPolicy,
    MaskedSPTPolicy,
    rollout_policy,
)
from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.environment_attribution_experiment import graph_snapshot
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.qplex import (
    ALL_FEASIBLE_GRAPH,
    NULL_GRAPH,
    POLICY_INTENT_TOP2_GRAPH,
    QPLEXPolicy,
    train_qplex,
)
from ht_pdm_fjsp.resource_conflict_graph_experiment import connected_components
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config
from ht_pdm_fjsp.three_way_marl_experiment import precedence_blocked_operations
from ht_pdm_fjsp.value_decomposition import (
    ValueDecompositionPolicy,
    ValueLearningSettings,
    train_value_policy,
)


QMIX = "qmix"
QPLEX_NULL = "qplex_null_graph"
QPLEX_ALL_FEASIBLE = "qplex_all_feasible"
QPLEX_POLICY_INTENT = "qplex_policy_intent_top2"
CONDITIONS = (QMIX, QPLEX_NULL, QPLEX_ALL_FEASIBLE, QPLEX_POLICY_INTENT)
QPLEX_GRAPHS = {
    QPLEX_NULL: NULL_GRAPH,
    QPLEX_ALL_FEASIBLE: ALL_FEASIBLE_GRAPH,
    QPLEX_POLICY_INTENT: POLICY_INTENT_TOP2_GRAPH,
}
SEALED_TEST_SEEDS = tuple(range(62_000, 62_100))
PRIMARY_CHECKPOINT = 300_000
MINIMUM_OBJECTIVE_IMPROVEMENT = 0.03
MINIMUM_WINNING_TRAIN_SEEDS = 4
MAXIMUM_FAILURE_REGRESSION = 0.05
SUMMARY_METRICS = (
    "objective",
    "makespan",
    "total_cost",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
    "maintenance_wait_time",
    "maximum_rejection_streak",
)


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile(name: str) -> dict[str, Any]:
    if name == "smoke":
        return {
            "total_timesteps": 1_200,
            "train_seeds": (76_100,),
            "evaluation_seeds": tuple(range(61_990, 61_993)),
            "n_envs": 4,
            "replay_capacity": 1_200,
            "learning_starts": 64,
            "batch_size": 32,
        }
    if name == "full":
        return {
            "total_timesteps": 300_000,
            "train_seeds": (76_000, 77_000, 78_000, 79_000, 80_000),
            "evaluation_seeds": tuple(range(61_700, 61_750)),
            "n_envs": 8,
            "replay_capacity": 20_000,
            "learning_starts": 2_000,
            "batch_size": 256,
        }
    raise ValueError(f"Unknown profile: {name}")


def _checkpoint_targets(total_timesteps: int) -> tuple[int, ...]:
    if total_timesteps % 3:
        raise ValueError("Training budget must be divisible by three")
    return tuple(total_timesteps * part // 3 for part in range(1, 4))


def _checkpoint_path(cell: Path, target: int) -> Path:
    return cell / "checkpoints" / f"model_{target}_steps.pt"


def _reward_sanity(
    config: BenchmarkConfig, seeds: tuple[int, ...]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in tqdm(
        (MaskedSPTPolicy(), HealthThresholdPolicy()),
        desc="Reward sanity policies",
        unit="policy",
    ):
        for seed in tqdm(
            seeds, desc=policy.name, unit="episode", leave=False
        ):
            result, episode_return = rollout_policy(
                HTPdmFjspEnv(config), policy, seed=seed
            )
            rows.append(
                {
                    "policy": policy.name,
                    "seed": seed,
                    "episode_return": episode_return,
                    **result.metrics,
                }
            )
    return rows


def _actions(policy: Any, observation: dict[str, np.ndarray], device: str) -> np.ndarray:
    return np.asarray(
        policy.act(observation, deterministic=True, device=device),
        dtype=np.int64,
    )


def evaluate_episode(
    policy: ValueDecompositionPolicy | QPLEXPolicy,
    config: BenchmarkConfig,
    *,
    condition: str,
    train_seed: int,
    checkpoint_steps: int,
    seed: int,
    device: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, Any],
    list[dict[str, Any]],
]:
    env = MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
    observation, _ = env.reset(seed=seed)
    decisions: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    episode_return = 0.0
    joint_step = 0
    rejection_streaks = np.zeros(env.num_agents, dtype=np.int64)
    maximum_rejection_streak = 0
    while not env._done:
        actions = _actions(policy, observation, device)
        selected_sets = tuple((int(action),) for action in actions)
        _, actual_edges = graph_snapshot(
            env,
            selected_sets,
            include_production=True,
            include_technician=True,
        )
        diagnostics: dict[str, Any] | None = None
        predicted_edges: set[tuple[int, int]] = set()
        if isinstance(policy, QPLEXPolicy):
            diagnostics = policy.mixer_diagnostics(
                observation, actions, device=device
            )
            adjacency = np.asarray(diagnostics["adjacency"], dtype=np.bool_)
            if not np.array_equal(adjacency, adjacency.T):
                raise AssertionError("Logged graph is not symmetric")
            if np.diag(adjacency).any():
                raise AssertionError("Logged graph contains self edges")
            predicted_edges = {
                (left, right)
                for left in range(env.num_agents)
                for right in range(left + 1, env.num_agents)
                if adjacency[left, right]
            }
            components = connected_components(env.num_agents, predicted_edges)
            lambdas = np.asarray(diagnostics["lambdas"], dtype=np.float64)
            degree = adjacency.sum(axis=1)
            connected = degree > 0
            graph_rows.append(
                {
                    "condition": condition,
                    "graph_variant": policy.graph_variant,
                    "train_seed": train_seed,
                    "checkpoint_steps": checkpoint_steps,
                    "seed": seed,
                    "joint_step": joint_step,
                    "joint_q": diagnostics["joint_q"],
                    "predicted_edge_count": len(predicted_edges),
                    "actual_conflict_edge_count": len(actual_edges),
                    "edge_true_positive": len(predicted_edges & actual_edges),
                    "edge_false_positive": len(predicted_edges - actual_edges),
                    "edge_false_negative": len(actual_edges - predicted_edges),
                    "graph_density": len(predicted_edges)
                    / (env.num_agents * (env.num_agents - 1) / 2),
                    "largest_component_size": max(map(len, components)),
                    "lambda_mean": float(lambdas.mean()),
                    "lambda_max": float(lambdas.max()),
                    "lambda_connected_mean": (
                        float(lambdas[connected].mean()) if connected.any() else 0.0
                    ),
                    "lambda_isolated_mean": (
                        float(lambdas[~connected].mean())
                        if (~connected).any()
                        else 0.0
                    ),
                }
            )
        before_time = env.core.now
        blocked = precedence_blocked_operations(env)
        observation, reward, _, _, info = env.step(actions)
        resolution = info["coordination"]
        resolver_has_conflict = bool(
            resolution["production_conflicts"]
            or resolution["technician_conflicts"]
        )
        if resolver_has_conflict != bool(actual_edges):
            raise AssertionError("Proposal conflict graph disagrees with resolver")
        outcomes = np.asarray(info["agent_outcomes"], dtype=np.int64)
        rejection_streaks = np.where(outcomes == -1, rejection_streaks + 1, 0)
        maximum_rejection_streak = max(
            maximum_rejection_streak, int(rejection_streaks.max())
        )
        decisions.append(
            {
                "condition": condition,
                "train_seed": train_seed,
                "checkpoint_steps": checkpoint_steps,
                "seed": seed,
                "joint_step": joint_step,
                "simulation_time_before": before_time,
                "production_conflicts": resolution["production_conflicts"],
                "technician_conflicts": resolution["technician_conflicts"],
                "proposals": resolution["proposals"],
                "accepted": resolution["accepted"],
                "rejected": resolution["rejected"],
                "waits": resolution["waits"],
                "precedence_blocked_operations": blocked,
                "reward": reward,
            }
        )
        episode_return += float(reward)
        joint_step += 1
    result = env.result(condition)
    if not np.isclose(episode_return, -float(result.metrics["objective"])):
        raise AssertionError("Reward/objective identity failed")
    invalid = sum(
        int(env.coordination_totals[key])
        for key in (
            "invalid_executions",
            "duplicate_operation_executions",
            "duplicate_technician_executions",
        )
    )
    episode = {
        "condition": condition,
        "train_seed": train_seed,
        "checkpoint_steps": checkpoint_steps,
        "seed": seed,
        "joint_steps": joint_step,
        "episode_return": episode_return,
        **result.metrics,
        **env.coordination_totals,
        "maximum_rejection_streak": maximum_rejection_streak,
        "coordination_audit_failures": invalid,
    }
    coordination = {
        "condition": condition,
        "train_seed": train_seed,
        "checkpoint_steps": checkpoint_steps,
        "seed": seed,
        **env.coordination_totals,
        "maximum_rejection_streak": maximum_rejection_streak,
    }
    env.close()
    return episode, decisions, coordination, graph_rows


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return statistics.fmean(float(row[key]) for row in rows)


def _cell_rows(
    episodes: list[dict[str, Any]],
    condition: str,
    train_seed: int,
    checkpoint: int,
) -> list[dict[str, Any]]:
    return [
        row
        for row in episodes
        if row["condition"] == condition
        and int(row["train_seed"]) == train_seed
        and int(row["checkpoint_steps"]) == checkpoint
    ]


def summarize(
    episodes: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    coordination: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    reward_rows: list[dict[str, Any]],
    *,
    profile: str,
    train_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    checkpoints: tuple[int, ...],
    checkpoint_files_complete: bool,
) -> dict[str, Any]:
    curves: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        points: list[dict[str, Any]] = []
        for train_seed in train_seeds:
            for checkpoint in checkpoints:
                selected = _cell_rows(
                    episodes, condition, train_seed, checkpoint
                )
                selected_decisions = [
                    row
                    for row in decisions
                    if row["condition"] == condition
                    and int(row["train_seed"]) == train_seed
                    and int(row["checkpoint_steps"]) == checkpoint
                ]
                joint_steps = len(selected_decisions)
                proposals = sum(int(row["proposals"]) for row in selected_decisions)
                rejected = sum(int(row["rejected"]) for row in selected_decisions)
                production_conflicts = sum(
                    int(row["production_conflicts"])
                    for row in selected_decisions
                )
                technician_conflicts = sum(
                    int(row["technician_conflicts"])
                    for row in selected_decisions
                )
                points.append(
                    {
                        "train_seed": train_seed,
                        "checkpoint_steps": checkpoint,
                        "episodes": len(selected),
                        **{
                            f"{metric}_mean": _mean(selected, metric)
                            for metric in SUMMARY_METRICS
                        },
                        "rejections_per_1000_joint_steps": (
                            1_000 * rejected / joint_steps
                        ),
                        "production_conflicts_per_1000_joint_steps": (
                            1_000 * production_conflicts / joint_steps
                        ),
                        "technician_conflicts_per_1000_joint_steps": (
                            1_000 * technician_conflicts / joint_steps
                        ),
                        "proposal_acceptance_ratio": (
                            (proposals - rejected) / proposals if proposals else 1.0
                        ),
                    }
                )
        curves[condition] = points

    final_checkpoint = checkpoints[-1]
    per_seed: list[dict[str, Any]] = []
    for train_seed in train_seeds:
        means = {
            condition: {
                metric: _mean(
                    _cell_rows(episodes, condition, train_seed, final_checkpoint),
                    metric,
                )
                for metric in SUMMARY_METRICS
            }
            for condition in CONDITIONS
        }
        per_seed.append(
            {
                "train_seed": train_seed,
                "objective_delta_qplex_null_minus_qmix": (
                    means[QPLEX_NULL]["objective"] - means[QMIX]["objective"]
                ),
                "objective_delta_intent_minus_null": (
                    means[QPLEX_POLICY_INTENT]["objective"]
                    - means[QPLEX_NULL]["objective"]
                ),
                "objective_delta_intent_minus_all_feasible": (
                    means[QPLEX_POLICY_INTENT]["objective"]
                    - means[QPLEX_ALL_FEASIBLE]["objective"]
                ),
                "failure_delta_intent_minus_null": (
                    means[QPLEX_POLICY_INTENT]["failures"]
                    - means[QPLEX_NULL]["failures"]
                ),
                **{
                    f"{condition}_{metric}_mean": value
                    for condition, condition_means in means.items()
                    for metric, value in condition_means.items()
                },
            }
        )

    expected_episodes = (
        len(CONDITIONS)
        * len(train_seeds)
        * len(checkpoints)
        * len(evaluation_seeds)
    )
    expected_graph_rows = sum(
        int(row["joint_steps"])
        for row in episodes
        if row["condition"] in QPLEX_GRAPHS
    )
    observed_graph_keys = {
        (
            row["condition"],
            int(row["train_seed"]),
            int(row["checkpoint_steps"]),
            int(row["seed"]),
            int(row["joint_step"]),
        )
        for row in graph_rows
    }
    audit_checks = {
        "all_expected_evaluations_completed": len(episodes) == expected_episodes,
        "coordination_rows_match_episodes": len(coordination) == len(episodes),
        "all_checkpoints_exist": checkpoint_files_complete,
        "every_episode_has_joint_steps": all(
            int(row["joint_steps"]) > 0 for row in episodes
        ),
        "one_graph_row_per_qplex_epoch": (
            len(graph_rows) == expected_graph_rows
            and len(observed_graph_keys) == len(graph_rows)
        ),
        "all_coordination_audits_passed": all(
            int(row["coordination_audit_failures"]) == 0 for row in episodes
        ) and all(
            sum(
                int(row[key])
                for key in (
                    "invalid_executions",
                    "duplicate_operation_executions",
                    "duplicate_technician_executions",
                )
            )
            == 0
            for row in coordination
        ),
        "reward_objective_identity_checked": len(reward_rows)
        == 2 * len(evaluation_seeds),
        "sealed_test_panel_remains_closed": True,
    }

    locked_decision: dict[str, Any] = {
        "eligible": profile == "full",
        "thresholds": {
            "minimum_objective_improvement_fraction": MINIMUM_OBJECTIVE_IMPROVEMENT,
            "minimum_winning_training_seeds": MINIMUM_WINNING_TRAIN_SEEDS,
            "maximum_failure_regression_fraction": MAXIMUM_FAILURE_REGRESSION,
        },
    }
    if profile == "full":
        final_objectives = {
            condition: statistics.fmean(
                float(row[f"{condition}_objective_mean"]) for row in per_seed
            )
            for condition in CONDITIONS
        }
        final_failures = {
            condition: statistics.fmean(
                float(row[f"{condition}_failures_mean"]) for row in per_seed
            )
            for condition in CONDITIONS
        }
        null_objective = final_objectives[QPLEX_NULL]
        objective_improvement = (
            (null_objective - final_objectives[QPLEX_POLICY_INTENT])
            / null_objective
        )
        intent_wins = sum(
            float(row["objective_delta_intent_minus_null"]) < 0
            for row in per_seed
        )
        null_failures = final_failures[QPLEX_NULL]
        failure_limit = (
            0.0
            if null_failures == 0
            else null_failures * (1.0 + MAXIMUM_FAILURE_REGRESSION)
        )
        promotion_checks = {
            "intent_improves_objective_by_at_least_3_percent": (
                objective_improvement >= MINIMUM_OBJECTIVE_IMPROVEMENT
            ),
            "intent_wins_at_least_four_of_five_training_seeds": (
                intent_wins >= MINIMUM_WINNING_TRAIN_SEEDS
            ),
            "intent_beats_all_feasible_grand_mean": (
                final_objectives[QPLEX_POLICY_INTENT]
                < final_objectives[QPLEX_ALL_FEASIBLE]
            ),
            "intent_failure_mean_within_5_percent_of_null": (
                final_failures[QPLEX_POLICY_INTENT] <= failure_limit
            ),
            "integrity_gate_passed": all(audit_checks.values()),
        }
        mixer_deltas = [
            float(row["objective_delta_qplex_null_minus_qmix"])
            for row in per_seed
        ]
        locked_decision.update(
            final_objective_grand_means=final_objectives,
            final_failure_grand_means=final_failures,
            intent_objective_improvement_fraction=objective_improvement,
            intent_winning_training_seeds=intent_wins,
            qplex_null_mixer_noninferiority={
                "mean_delta_null_minus_qmix": statistics.fmean(mixer_deltas),
                "nonpositive_seed_count": sum(delta <= 0 for delta in mixer_deltas),
                "passed": (
                    statistics.fmean(mixer_deltas) <= 0
                    and sum(delta <= 0 for delta in mixer_deltas) >= 3
                ),
            },
            promotion_checks=promotion_checks,
            promote_policy_intent_soft_graph=all(promotion_checks.values()),
        )

    graph_summary: dict[str, Any] = {}
    for condition in QPLEX_GRAPHS:
        selected = [row for row in graph_rows if row["condition"] == condition]
        graph_summary[condition] = {
            "epochs": len(selected),
            "mean_graph_density": _mean(selected, "graph_density"),
            "mean_largest_component_size": _mean(
                selected, "largest_component_size"
            ),
            "mean_lambda": _mean(selected, "lambda_mean"),
            "mean_max_lambda": _mean(selected, "lambda_max"),
            "edge_true_positive": sum(
                int(row["edge_true_positive"]) for row in selected
            ),
            "edge_false_positive": sum(
                int(row["edge_false_positive"]) for row in selected
            ),
            "edge_false_negative": sum(
                int(row["edge_false_negative"]) for row in selected
            ),
        }

    return {
        "experiment": "QPLEX soft-graph ablation",
        "primary_checkpoint": final_checkpoint,
        "curves": curves,
        "final_paired_differences_by_training_seed": per_seed,
        "graph_mechanism": graph_summary,
        "integrity_gate": {
            "checks": audit_checks,
            "passed": all(audit_checks.values()),
        },
        "locked_decision": locked_decision,
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = _profile(args.profile)
    train_seeds = protocol["train_seeds"]
    evaluation_seeds = protocol["evaluation_seeds"]
    if set(evaluation_seeds) & set(SEALED_TEST_SEEDS):
        raise ValueError("The sealed future test panel cannot be opened")
    checkpoints = _checkpoint_targets(protocol["total_timesteps"])
    base_path = Path(args.config).resolve()
    raw_config = base_path.read_bytes()
    base = BenchmarkConfig.from_json(base_path)
    config = build_condition_config(base, topology="two_specialists", multiplier=2.0)
    scaled_payload = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    (output_dir / "scaled_config.json").write_text(scaled_payload)
    device = resolve_device(args.device)
    settings = ValueLearningSettings(
        total_timesteps=protocol["total_timesteps"],
        replay_capacity=protocol["replay_capacity"],
        learning_starts=protocol["learning_starts"],
        batch_size=protocol["batch_size"],
        train_frequency=4,
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_interval=max(32, protocol["total_timesteps"] // 100),
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_fraction=0.5,
        hidden_dim=128,
        mixer_hidden_dim=64,
        device=device,
        n_envs=protocol["n_envs"],
    )
    manifest_path = output_dir / "qplex_soft_graph_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "git_commit": _git_revision(),
        "base_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "scaled_config_sha256": hashlib.sha256(
            scaled_payload.encode()
        ).hexdigest(),
        "capacity_condition": "two_specialists_x2_0",
        "conditions": CONDITIONS,
        "qplex_graph_variants": QPLEX_GRAPHS,
        "train_seeds": train_seeds,
        "evaluation_seeds": evaluation_seeds,
        "reserved_future_test_seeds": SEALED_TEST_SEEDS,
        "future_test_panel_opened": False,
        "checkpoint_targets": checkpoints,
        "primary_checkpoint": checkpoints[-1],
        "settings": asdict(settings),
        "promotion_thresholds": {
            "minimum_objective_improvement_fraction": MINIMUM_OBJECTIVE_IMPROVEMENT,
            "minimum_winning_training_seeds": MINIMUM_WINNING_TRAIN_SEEDS,
            "maximum_failure_regression_fraction": MAXIMUM_FAILURE_REGRESSION,
        },
        "requested_device": args.device,
        "resolved_device": device,
        "completed_training_cells": [],
        "completed_evaluation_cells": [],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "torch")},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    reward_rows = _reward_sanity(config, evaluation_seeds)
    _write_csv(reward_rows, output_dir / "reward_sanity_episodes.csv")
    episodes: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    coordination: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}
    for condition in tqdm(CONDITIONS, desc="QPLEX conditions", unit="condition"):
        for train_seed in tqdm(
            train_seeds, desc=condition, unit="training-seed", leave=False
        ):
            cell = output_dir / condition / f"train_seed_{train_seed}"
            if condition == QMIX:
                _, elapsed = train_value_policy(
                    config,
                    settings,
                    cell,
                    algorithm="qmix",
                    train_seed=train_seed,
                    show_progress=True,
                    checkpoint_targets=checkpoints,
                )
            else:
                _, elapsed = train_qplex(
                    config,
                    settings,
                    cell,
                    graph_variant=QPLEX_GRAPHS[condition],
                    train_seed=train_seed,
                    show_progress=True,
                    checkpoint_targets=checkpoints,
                )
            cell_key = f"{condition}:{train_seed}"
            training_seconds[cell_key] = elapsed
            manifest["completed_training_cells"].append(cell_key)
            manifest["training_seconds"] = training_seconds
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            )
            for checkpoint in tqdm(
                checkpoints,
                desc=f"Evaluate {cell_key}",
                unit="checkpoint",
                leave=False,
            ):
                path = _checkpoint_path(cell, checkpoint)
                policy: ValueDecompositionPolicy | QPLEXPolicy
                policy = (
                    ValueDecompositionPolicy.load(path, device=device)
                    if condition == QMIX
                    else QPLEXPolicy.load(path, device=device)
                )
                for seed in tqdm(
                    evaluation_seeds,
                    desc=f"{condition}/{checkpoint}",
                    unit="episode",
                    leave=False,
                ):
                    episode, episode_decisions, audit, episode_graphs = (
                        evaluate_episode(
                            policy,
                            config,
                            condition=condition,
                            train_seed=train_seed,
                            checkpoint_steps=checkpoint,
                            seed=seed,
                            device=device,
                        )
                    )
                    episodes.append(episode)
                    decisions.extend(episode_decisions)
                    coordination.append(audit)
                    graph_rows.extend(episode_graphs)
                manifest["completed_evaluation_cells"].append(
                    f"{cell_key}:{checkpoint}"
                )
                _write_csv(
                    episodes,
                    output_dir / "qplex_soft_graph_episodes.partial.csv",
                )
                _write_csv(
                    decisions,
                    output_dir / "qplex_soft_graph_decisions.partial.csv",
                )
                _write_csv(
                    coordination,
                    output_dir / "qplex_soft_graph_coordination.partial.csv",
                )
                _write_csv(
                    graph_rows,
                    output_dir / "qplex_soft_graph_mixer_epochs.partial.csv",
                )
                manifest_path.write_text(
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
                )
                del policy

    checkpoint_files_complete = all(
        _checkpoint_path(
            output_dir / condition / f"train_seed_{train_seed}", checkpoint
        ).is_file()
        for condition in CONDITIONS
        for train_seed in train_seeds
        for checkpoint in checkpoints
    )
    summary = summarize(
        episodes,
        decisions,
        coordination,
        graph_rows,
        reward_rows,
        profile=args.profile,
        train_seeds=train_seeds,
        evaluation_seeds=evaluation_seeds,
        checkpoints=checkpoints,
        checkpoint_files_complete=checkpoint_files_complete,
    )
    _write_csv(episodes, output_dir / "qplex_soft_graph_episodes.csv")
    _write_csv(decisions, output_dir / "qplex_soft_graph_decisions.csv")
    _write_csv(coordination, output_dir / "qplex_soft_graph_coordination.csv")
    _write_csv(graph_rows, output_dir / "qplex_soft_graph_mixer_epochs.csv")
    (output_dir / "qplex_soft_graph_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        status="COMPLETED",
        completed_at_utc=datetime.now(UTC).isoformat(),
        episode_count=len(episodes),
        decision_count=len(decisions),
        graph_row_count=len(graph_rows),
        training_seconds=training_seconds,
        integrity_gate=summary["integrity_gate"],
        locked_decision=summary["locked_decision"],
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["integrity_gate"], indent=2, sort_keys=True))
    print(json.dumps(summary["locked_decision"], indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brandimarte_mk01_ht_pdm.json")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
