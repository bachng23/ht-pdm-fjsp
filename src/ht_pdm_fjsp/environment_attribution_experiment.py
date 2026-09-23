"""Attribute resource-graph topology to feasibility and policy intent."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
from collections import defaultdict
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch as th
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.resource_conflict_graph_experiment import (
    CHECKPOINT_STEPS,
    CONDITIONS,
    FULLY_CONNECTED_MAXIMUM,
    MEAN_LARGEST_LIMIT,
    PRIMARY_CONDITION,
    SEALED_TEST_SEEDS,
    SIZE_TWO_THREE_MINIMUM,
    _sha256,
    _validate_source_run,
    connected_components,
)
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.value_decomposition import ValueDecompositionPolicy


GRAPH_VARIANTS = ("all_feasible", "production_only", "policy_intent_top2")
INTENT_TOP_K = 2
EDGE_RECALL_MINIMUM = 0.95
EDGE_PRECISION_MINIMUM = 0.30


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile(name: str) -> tuple[int, ...]:
    if name == "smoke":
        return tuple(range(61_990, 61_993))
    if name == "full":
        return tuple(range(61_900, 61_950))
    raise ValueError(f"Unknown profile: {name}")


def top_k_intent_actions(
    q_values: np.ndarray, masks: np.ndarray, *, top_k: int
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Return deterministic greedy actions and stable top-k feasible sets."""

    if q_values.shape != masks.shape or q_values.ndim != 2:
        raise ValueError("Q values and masks must be matching agent-action matrices")
    if top_k < 1:
        raise ValueError("top_k must be positive")
    greedy: list[int] = []
    selected: list[tuple[int, ...]] = []
    for agent in range(q_values.shape[0]):
        valid = np.flatnonzero(masks[agent])
        if not len(valid):
            raise ValueError("Every agent must have a feasible action")
        ordered = sorted(
            valid.tolist(),
            key=lambda action: (-q_values[agent, action], action),
        )
        greedy.append(int(ordered[0]))
        selected.append(tuple(map(int, ordered[:top_k])))
    return np.asarray(greedy, dtype=np.int64), tuple(selected)


def policy_intent(
    policy: ValueDecompositionPolicy,
    observation: dict[str, np.ndarray],
    *,
    device: str,
    top_k: int = INTENT_TOP_K,
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    local = th.as_tensor(
        observation["local_observations"], dtype=th.float32, device=device
    ).unsqueeze(0)
    with th.no_grad():
        values = policy.q_values(local).squeeze(0).cpu().numpy()
    masks = np.asarray(observation["action_masks"], dtype=np.bool_)
    greedy, action_sets = top_k_intent_actions(values, masks, top_k=top_k)
    acted = np.asarray(
        policy.act(observation, deterministic=True, device=device), dtype=np.int64
    )
    if not np.array_equal(greedy, acted):
        raise AssertionError(
            "Intent ranking disagrees with deterministic policy action"
        )
    return greedy, action_sets


def feasible_action_sets(
    observation: dict[str, np.ndarray]
) -> tuple[tuple[int, ...], ...]:
    masks = np.asarray(observation["action_masks"], dtype=np.bool_)
    return tuple(
        tuple(map(int, np.flatnonzero(agent_mask))) for agent_mask in masks
    )


def graph_snapshot(
    env: MachineAgentsCTDEEnv,
    local_action_sets: Iterable[Iterable[int]],
    *,
    include_production: bool,
    include_technician: bool,
) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    production: list[set[tuple[str, int]]] = []
    technicians: list[set[str]] = []
    action_sets = tuple(tuple(actions) for actions in local_action_sets)
    if len(action_sets) != env.num_agents:
        raise ValueError("Action sets must cover every agent")
    for agent_index, local_actions in enumerate(action_sets):
        production_resources: set[tuple[str, int]] = set()
        technician_resources: set[str] = set()
        for local_action in local_actions:
            global_action = env.local_to_global(agent_index, int(local_action))
            if global_action is None:
                continue
            descriptor = env.core.actions[global_action]
            if include_production and descriptor.kind == "production":
                production_resources.add(
                    (str(descriptor.job_id), int(descriptor.operation_index))
                )
            elif include_technician and descriptor.kind in {
                "preventive",
                "corrective",
            }:
                technician_resources.add(str(descriptor.technician_id))
        production.append(production_resources)
        technicians.append(technician_resources)

    production_edges: set[tuple[int, int]] = set()
    technician_edges: set[tuple[int, int]] = set()
    for left in range(env.num_agents):
        for right in range(left + 1, env.num_agents):
            edge = (left, right)
            if production[left] & production[right]:
                production_edges.add(edge)
            if technicians[left] & technicians[right]:
                technician_edges.add(edge)
    edges = production_edges | technician_edges
    components = connected_components(env.num_agents, edges)
    sizes = tuple(len(component) for component in components)
    nontrivial = tuple(size for size in sizes if size >= 2)
    maximum_edges = env.num_agents * (env.num_agents - 1) / 2
    return (
        {
            "component_sizes": sizes,
            "largest_component_size": max(sizes),
            "mean_component_size_including_singletons": env.num_agents
            / len(sizes),
            "mean_nontrivial_component_size": (
                statistics.fmean(nontrivial) if nontrivial else 0.0
            ),
            "component_count": len(sizes),
            "isolated_agent_count": sum(size == 1 for size in sizes),
            "has_component_size_2_or_3": int(
                any(2 <= size <= 3 for size in sizes)
            ),
            "fully_connected": int(len(components) == 1),
            "production_edge_count": len(production_edges),
            "technician_edge_count": len(technician_edges),
            "total_edge_count": len(edges),
            "graph_density": len(edges) / maximum_edges,
        },
        edges,
    )


def evaluate_episode(
    policy: ValueDecompositionPolicy,
    config: BenchmarkConfig,
    *,
    condition: str,
    train_seed: int,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
    observation, _ = env.reset(seed=seed)
    graph_rows: list[dict[str, Any]] = []
    episode_return = 0.0
    joint_step = 0
    while not env._done:
        actions, intent_sets = policy_intent(policy, observation, device=device)
        all_feasible = feasible_action_sets(observation)
        selected_sets = tuple((int(action),) for action in actions)
        actual_graph, actual_edges = graph_snapshot(
            env,
            selected_sets,
            include_production=True,
            include_technician=True,
        )
        variants = {
            "all_feasible": graph_snapshot(
                env,
                all_feasible,
                include_production=True,
                include_technician=True,
            ),
            "production_only": graph_snapshot(
                env,
                all_feasible,
                include_production=True,
                include_technician=False,
            ),
            "policy_intent_top2": graph_snapshot(
                env,
                intent_sets,
                include_production=True,
                include_technician=True,
            ),
        }
        observation, reward, _, _, info = env.step(actions)
        resolution = info["coordination"]
        resolver_has_conflict = int(
            resolution["production_conflicts"] > 0
            or resolution["technician_conflicts"] > 0
        )
        if resolver_has_conflict != int(bool(actual_edges)):
            raise AssertionError("Proposal conflict edges disagree with resolver")
        for variant in GRAPH_VARIANTS:
            graph, predicted_edges = variants[variant]
            true_positive = len(predicted_edges & actual_edges)
            false_positive = len(predicted_edges - actual_edges)
            false_negative = len(actual_edges - predicted_edges)
            graph_rows.append(
                {
                    "condition": condition,
                    "graph_variant": variant,
                    "train_seed": train_seed,
                    "seed": seed,
                    "joint_step": joint_step,
                    "component_sizes": json.dumps(graph["component_sizes"]),
                    **{
                        key: value
                        for key, value in graph.items()
                        if key != "component_sizes"
                    },
                    "actual_conflict_edge_count": len(actual_edges),
                    "edge_true_positive": true_positive,
                    "edge_false_positive": false_positive,
                    "edge_false_negative": false_negative,
                    "predicted_conflict_epoch": int(bool(predicted_edges)),
                    "actual_conflict_epoch": resolver_has_conflict,
                    "production_conflicts": resolution["production_conflicts"],
                    "technician_conflicts": resolution["technician_conflicts"],
                    "rejected": resolution["rejected"],
                    "reward": reward,
                }
            )
        if actual_graph["total_edge_count"] != len(actual_edges):
            raise AssertionError("Actual graph edge count is inconsistent")
        episode_return += float(reward)
        joint_step += 1
    result = env.result(condition)
    if not np.isclose(episode_return, -float(result.metrics["objective"])):
        raise AssertionError("Reward/objective identity failed")
    audit_failures = sum(
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
        "seed": seed,
        "joint_steps": joint_step,
        "episode_return": episode_return,
        **result.metrics,
        **env.coordination_totals,
        "coordination_audit_failures": audit_failures,
    }
    env.close()
    return episode, graph_rows


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 1.0


def _interval(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    half = 0.0
    if len(values) > 1:
        half = 1.96 * statistics.stdev(values) / len(values) ** 0.5
    return {"mean": mean, "lower_95": mean - half, "upper_95": mean + half}


def _variant_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Variant summary requires epoch rows")
    true_positive = sum(int(row["edge_true_positive"]) for row in rows)
    false_positive = sum(int(row["edge_false_positive"]) for row in rows)
    false_negative = sum(int(row["edge_false_negative"]) for row in rows)
    epoch_tp = sum(
        int(row["predicted_conflict_epoch"])
        and int(row["actual_conflict_epoch"])
        for row in rows
    )
    epoch_fp = sum(
        int(row["predicted_conflict_epoch"])
        and not int(row["actual_conflict_epoch"])
        for row in rows
    )
    epoch_fn = sum(
        not int(row["predicted_conflict_epoch"])
        and int(row["actual_conflict_epoch"])
        for row in rows
    )
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[int(row["seed"])].append(row)
    episode_metrics = {
        "mean_largest_component_size": [],
        "fraction_epochs_with_component_size_2_or_3": [],
        "fraction_epochs_fully_connected": [],
    }
    for seed_rows in by_seed.values():
        episode_metrics["mean_largest_component_size"].append(
            statistics.fmean(
                float(row["largest_component_size"]) for row in seed_rows
            )
        )
        episode_metrics["fraction_epochs_with_component_size_2_or_3"].append(
            statistics.fmean(
                float(row["has_component_size_2_or_3"]) for row in seed_rows
            )
        )
        episode_metrics["fraction_epochs_fully_connected"].append(
            statistics.fmean(float(row["fully_connected"]) for row in seed_rows)
        )
    return {
        "episodes": len(by_seed),
        "epochs": len(rows),
        "epoch_weighted": {
            "mean_largest_connected_component_size": statistics.fmean(
                float(row["largest_component_size"]) for row in rows
            ),
            "fraction_epochs_with_component_size_2_or_3": statistics.fmean(
                float(row["has_component_size_2_or_3"]) for row in rows
            ),
            "fraction_epochs_fully_connected": statistics.fmean(
                float(row["fully_connected"]) for row in rows
            ),
            "mean_component_size_including_singletons": statistics.fmean(
                float(row["mean_component_size_including_singletons"])
                for row in rows
            ),
            "mean_nontrivial_component_size": statistics.fmean(
                float(row["mean_nontrivial_component_size"]) for row in rows
            ),
            "mean_isolated_agent_count": statistics.fmean(
                float(row["isolated_agent_count"]) for row in rows
            ),
            "mean_graph_density": statistics.fmean(
                float(row["graph_density"]) for row in rows
            ),
        },
        "episode_weighted_intervals": {
            key: _interval(values) for key, values in episode_metrics.items()
        },
        "edge_prediction": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": _ratio(true_positive, true_positive + false_positive),
            "recall": _ratio(true_positive, true_positive + false_negative),
        },
        "epoch_prediction": {
            "true_positive": epoch_tp,
            "false_positive": epoch_fp,
            "false_negative": epoch_fn,
            "precision": _ratio(epoch_tp, epoch_tp + epoch_fp),
            "recall": _ratio(epoch_tp, epoch_tp + epoch_fn),
        },
    }


def summarize(
    episodes: list[dict[str, Any]],
    graph_rows: list[dict[str, Any]],
    *,
    expected_episodes: int,
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_rows = [
            row for row in graph_rows if row["condition"] == condition
        ]
        conditions[condition] = {
            variant: _variant_summary(
                [row for row in condition_rows if row["graph_variant"] == variant]
            )
            for variant in GRAPH_VARIANTS
        }
    decisions: dict[str, Any] = {}
    for variant in GRAPH_VARIANTS:
        result = conditions[PRIMARY_CONDITION][variant]
        topology = result["epoch_weighted"]
        prediction = result["edge_prediction"]
        checks = {
            "mean_largest_component_size_at_most_3_5": (
                topology["mean_largest_connected_component_size"]
                <= MEAN_LARGEST_LIMIT
            ),
            "at_least_half_epochs_have_size_2_or_3_component": (
                topology["fraction_epochs_with_component_size_2_or_3"]
                >= SIZE_TWO_THREE_MINIMUM
            ),
            "at_most_ten_percent_epochs_are_fully_connected": (
                topology["fraction_epochs_fully_connected"]
                <= FULLY_CONNECTED_MAXIMUM
            ),
            "edge_recall_at_least_95_percent": (
                prediction["recall"] >= EDGE_RECALL_MINIMUM
            ),
            "edge_precision_at_least_30_percent": (
                prediction["precision"] >= EDGE_PRECISION_MINIMUM
            ),
        }
        decisions[variant] = {
            "checks": checks,
            "supports_hard_dynamic_subteams": all(checks.values()),
        }
    expected_graph_keys = {
        (row["condition"], int(row["seed"]), joint_step, variant)
        for row in episodes
        for joint_step in range(int(row["joint_steps"]))
        for variant in GRAPH_VARIANTS
    }
    observed_graph_keys = {
        (
            row["condition"],
            int(row["seed"]),
            int(row["joint_step"]),
            row["graph_variant"],
        )
        for row in graph_rows
    }
    audit_checks = {
        "all_expected_episodes_completed": len(episodes) == expected_episodes,
        "every_episode_has_epochs": all(
            int(row["joint_steps"]) > 0 for row in episodes
        ),
        "all_graph_variants_logged_per_epoch": (
            len(graph_rows) == len(expected_graph_keys)
            and observed_graph_keys == expected_graph_keys
        ),
        "all_coordination_audits_passed": all(
            int(row["coordination_audit_failures"]) == 0 for row in episodes
        ),
        "sealed_test_panel_remains_closed": True,
    }
    return {
        "purpose": "Environment-attribution diagnostic, not a performance claim.",
        "primary_condition": PRIMARY_CONDITION,
        "graph_variants": GRAPH_VARIANTS,
        "conditions": conditions,
        "locked_architecture_rule": {
            "thresholds": {
                "mean_largest_component_size_max": MEAN_LARGEST_LIMIT,
                "fraction_epochs_with_size_2_or_3_min": SIZE_TWO_THREE_MINIMUM,
                "fraction_epochs_fully_connected_max": FULLY_CONNECTED_MAXIMUM,
                "edge_recall_min": EDGE_RECALL_MINIMUM,
                "edge_precision_min": EDGE_PRECISION_MINIMUM,
            },
            "decisions": decisions,
            "any_refined_graph_supports_hard_dynamic_subteams": any(
                decisions[variant]["supports_hard_dynamic_subteams"]
                for variant in GRAPH_VARIANTS[1:]
            ),
        },
        "gate": {"checks": audit_checks, "passed": all(audit_checks.values())},
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_run = Path(args.source_run).resolve()
    source_manifest, base_path, scaled_path = _validate_source_run(source_run)
    seeds = (
        tuple(parse_seeds(args.evaluation_seeds))
        if args.evaluation_seeds
        else _profile(args.profile)
    )
    if set(seeds) & set(SEALED_TEST_SEEDS):
        raise ValueError("The sealed future test panel cannot be opened")
    checkpoint_steps = int(args.checkpoint_steps)
    if checkpoint_steps != CHECKPOINT_STEPS:
        raise ValueError(f"Protocol is locked to checkpoint {CHECKPOINT_STEPS}")
    config = BenchmarkConfig.from_json(scaled_path)
    if len(config.machines) != 6:
        raise AssertionError("Diagnostic is locked to six machine agents")
    device = resolve_device(args.device)
    train_seed = int(source_manifest["train_seed"])
    model_paths = {
        condition: source_run
        / condition
        / "checkpoints"
        / f"model_{checkpoint_steps}_steps.pt"
        for condition in CONDITIONS
    }
    missing = [str(path) for path in model_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing source checkpoints: {missing}")
    (output_dir / "benchmark_config.json").write_bytes(base_path.read_bytes())
    (output_dir / "scaled_config.json").write_bytes(scaled_path.read_bytes())
    manifest_path = output_dir / "environment_attribution_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "profile": args.profile,
        "git_commit": _git_revision(),
        "source_run": str(source_run),
        "source_git_commit": source_manifest.get("git_commit"),
        "source_manifest_sha256": _sha256(
            source_run / "marl_budget_screening_manifest.json"
        ),
        "source_checkpoint_sha256": {
            condition: _sha256(path) for condition, path in model_paths.items()
        },
        "conditions": CONDITIONS,
        "primary_condition": PRIMARY_CONDITION,
        "graph_variants": GRAPH_VARIANTS,
        "intent_top_k": INTENT_TOP_K,
        "architecture_thresholds": {
            "mean_largest_component_size_max": MEAN_LARGEST_LIMIT,
            "fraction_epochs_with_size_2_or_3_min": SIZE_TWO_THREE_MINIMUM,
            "fraction_epochs_fully_connected_max": FULLY_CONNECTED_MAXIMUM,
            "edge_recall_min": EDGE_RECALL_MINIMUM,
            "edge_precision_min": EDGE_PRECISION_MINIMUM,
        },
        "train_seed": train_seed,
        "checkpoint_steps": checkpoint_steps,
        "evaluation_seeds": seeds,
        "reserved_future_test_seeds": SEALED_TEST_SEEDS,
        "future_test_panel_opened": False,
        "requested_device": args.device,
        "resolved_device": device,
        "completed_conditions": [],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "torch")},
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    episodes: list[dict[str, Any]] = []
    graph_rows: list[dict[str, Any]] = []
    for condition in tqdm(
        CONDITIONS, desc="Attribution policies", unit="policy"
    ):
        policy = ValueDecompositionPolicy.load(model_paths[condition], device=device)
        for seed in tqdm(
            seeds,
            desc=f"Evaluate attribution/{condition}",
            unit="episode",
            leave=False,
        ):
            episode, episode_graphs = evaluate_episode(
                policy,
                config,
                condition=condition,
                train_seed=train_seed,
                seed=seed,
                device=device,
            )
            episodes.append(episode)
            graph_rows.extend(episode_graphs)
        manifest["completed_conditions"].append(condition)
        _write_csv(
            episodes, output_dir / "environment_attribution_episodes.partial.csv"
        )
        _write_csv(
            graph_rows, output_dir / "environment_attribution_epochs.partial.csv"
        )
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        del policy
    summary = summarize(
        episodes, graph_rows, expected_episodes=len(CONDITIONS) * len(seeds)
    )
    _write_csv(episodes, output_dir / "environment_attribution_episodes.csv")
    _write_csv(graph_rows, output_dir / "environment_attribution_epochs.csv")
    (output_dir / "environment_attribution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        status="COMPLETED",
        completed_at_utc=datetime.now(UTC).isoformat(),
        episode_count=len(episodes),
        graph_row_count=len(graph_rows),
        gate=summary["gate"],
        architecture_decision=summary["locked_architecture_rule"],
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["locked_architecture_rule"], indent=2, sort_keys=True))
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--checkpoint-steps", type=int, default=CHECKPOINT_STEPS)
    parser.add_argument("--evaluation-seeds")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
