"""Measure dynamic resource-conflict graph topology under trained MARL policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import statistics
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.value_decomposition import ValueDecompositionPolicy


CONDITIONS = ("cooperative_iql", "qmix")
PRIMARY_CONDITION = "qmix"
SEALED_TEST_SEEDS = tuple(range(62_000, 62_100))
CHECKPOINT_STEPS = 300_000
MEAN_LARGEST_LIMIT = 3.5
SIZE_TWO_THREE_MINIMUM = 0.50
FULLY_CONNECTED_MAXIMUM = 0.10


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile(name: str) -> tuple[int, ...]:
    if name == "smoke":
        return tuple(range(61_990, 61_993))
    if name == "full":
        return tuple(range(61_900, 61_950))
    raise ValueError(f"Unknown profile: {name}")


def connected_components(
    agent_count: int, edges: set[tuple[int, int]]
) -> tuple[tuple[int, ...], ...]:
    """Return a deterministic partition containing isolated agents."""

    adjacency = [set() for _ in range(agent_count)]
    for left, right in edges:
        if not 0 <= left < agent_count or not 0 <= right < agent_count:
            raise ValueError("Graph edge contains an invalid agent index")
        if left == right:
            raise ValueError("Resource-conflict graph cannot contain self edges")
        adjacency[left].add(right)
        adjacency[right].add(left)
    components: list[tuple[int, ...]] = []
    unseen = set(range(agent_count))
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component, reverse=True))
        unseen -= component
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda item: (item[0], len(item))))


def resource_conflict_graph(
    env: MachineAgentsCTDEEnv, observation: dict[str, np.ndarray]
) -> dict[str, Any]:
    """Build the feasible action overlap graph before a joint decision."""

    masks = np.asarray(observation["action_masks"], dtype=np.bool_)
    production: list[set[tuple[str, int]]] = []
    technicians: list[set[str]] = []
    for agent_index, catalog in enumerate(env.local_action_catalogs):
        production_resources: set[tuple[str, int]] = set()
        technician_resources: set[str] = set()
        for local_action, global_action in enumerate(catalog):
            if global_action is None or not masks[agent_index, local_action]:
                continue
            descriptor = env.core.actions[global_action]
            if descriptor.kind == "production":
                production_resources.add(
                    (str(descriptor.job_id), int(descriptor.operation_index))
                )
            elif descriptor.kind in {"preventive", "corrective"}:
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
    flattened = sorted(node for component in components for node in component)
    if flattened != list(range(env.num_agents)):
        raise AssertionError("Connected components do not partition all agents")
    sizes = tuple(len(component) for component in components)
    nontrivial = tuple(size for size in sizes if size >= 2)
    maximum_edges = env.num_agents * (env.num_agents - 1) / 2
    return {
        "component_sizes": sizes,
        "largest_component_size": max(sizes),
        "mean_component_size_including_singletons": env.num_agents / len(sizes),
        "mean_nontrivial_component_size": (
            statistics.fmean(nontrivial) if nontrivial else 0.0
        ),
        "component_count": len(sizes),
        "isolated_agent_count": sum(size == 1 for size in sizes),
        "has_component_size_2_or_3": int(any(2 <= size <= 3 for size in sizes)),
        "fully_connected": int(len(components) == 1),
        "production_edge_count": len(production_edges),
        "technician_edge_count": len(technician_edges),
        "total_edge_count": len(edges),
        "graph_density": len(edges) / maximum_edges,
    }


def _actions(
    policy: ValueDecompositionPolicy,
    observation: dict[str, np.ndarray],
    device: str,
) -> np.ndarray:
    return np.asarray(
        policy.act(observation, deterministic=True, device=device), dtype=np.int64
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
    epoch_rows: list[dict[str, Any]] = []
    episode_return = 0.0
    joint_step = 0
    while not env._done:
        graph = resource_conflict_graph(env, observation)
        actions = _actions(policy, observation, device)
        observation, reward, _, _, info = env.step(actions)
        resolution = info["coordination"]
        epoch_rows.append(
            {
                "condition": condition,
                "train_seed": train_seed,
                "seed": seed,
                "joint_step": joint_step,
                "component_sizes": json.dumps(graph["component_sizes"]),
                **{
                    key: value
                    for key, value in graph.items()
                    if key != "component_sizes"
                },
                "production_conflicts": resolution["production_conflicts"],
                "technician_conflicts": resolution["technician_conflicts"],
                "rejected": resolution["rejected"],
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
        "seed": seed,
        "joint_steps": joint_step,
        "episode_return": episode_return,
        **result.metrics,
        **env.coordination_totals,
        "coordination_audit_failures": invalid,
        "mean_largest_component_size": statistics.fmean(
            float(row["largest_component_size"]) for row in epoch_rows
        ),
        "fraction_epochs_with_component_size_2_or_3": statistics.fmean(
            float(row["has_component_size_2_or_3"]) for row in epoch_rows
        ),
        "fraction_epochs_fully_connected": statistics.fmean(
            float(row["fully_connected"]) for row in epoch_rows
        ),
    }
    env.close()
    return episode, epoch_rows


def _episode_interval(values: list[float]) -> dict[str, float]:
    mean = statistics.fmean(values)
    half = 0.0
    if len(values) > 1:
        half = 1.96 * statistics.stdev(values) / len(values) ** 0.5
    return {
        "mean": mean,
        "lower_95": mean - half,
        "upper_95": mean + half,
    }


def summarize(
    episodes: list[dict[str, Any]],
    epochs: list[dict[str, Any]],
    *,
    expected_episodes: int,
) -> dict[str, Any]:
    conditions: dict[str, Any] = {}
    for condition in CONDITIONS:
        selected_epochs = [row for row in epochs if row["condition"] == condition]
        selected_episodes = [
            row for row in episodes if row["condition"] == condition
        ]
        if not selected_epochs or not selected_episodes:
            raise ValueError(f"Missing graph observations for {condition}")
        conditions[condition] = {
            "episodes": len(selected_episodes),
            "epochs": len(selected_epochs),
            "epoch_weighted": {
                "mean_largest_connected_component_size": statistics.fmean(
                    float(row["largest_component_size"])
                    for row in selected_epochs
                ),
                "fraction_epochs_with_component_size_2_or_3": statistics.fmean(
                    float(row["has_component_size_2_or_3"])
                    for row in selected_epochs
                ),
                "fraction_epochs_fully_connected": statistics.fmean(
                    float(row["fully_connected"]) for row in selected_epochs
                ),
                "mean_component_size_including_singletons": statistics.fmean(
                    float(row["mean_component_size_including_singletons"])
                    for row in selected_epochs
                ),
                "mean_nontrivial_component_size": statistics.fmean(
                    float(row["mean_nontrivial_component_size"])
                    for row in selected_epochs
                ),
                "mean_isolated_agent_count": statistics.fmean(
                    float(row["isolated_agent_count"])
                    for row in selected_epochs
                ),
                "mean_graph_density": statistics.fmean(
                    float(row["graph_density"]) for row in selected_epochs
                ),
            },
            "episode_weighted_intervals": {
                key: _episode_interval(
                    [float(row[key]) for row in selected_episodes]
                )
                for key in (
                    "mean_largest_component_size",
                    "fraction_epochs_with_component_size_2_or_3",
                    "fraction_epochs_fully_connected",
                )
            },
        }
    primary = conditions[PRIMARY_CONDITION]["epoch_weighted"]
    decision_checks = {
        "mean_largest_component_size_at_most_3_5": (
            primary["mean_largest_connected_component_size"]
            <= MEAN_LARGEST_LIMIT
        ),
        "at_least_half_epochs_have_size_2_or_3_component": (
            primary["fraction_epochs_with_component_size_2_or_3"]
            >= SIZE_TWO_THREE_MINIMUM
        ),
        "at_most_ten_percent_epochs_are_fully_connected": (
            primary["fraction_epochs_fully_connected"]
            <= FULLY_CONNECTED_MAXIMUM
        ),
    }
    audit_checks = {
        "all_expected_episodes_completed": len(episodes) == expected_episodes,
        "every_episode_has_epochs": all(
            int(row["joint_steps"]) > 0 for row in episodes
        ),
        "all_coordination_audits_passed": all(
            int(row["coordination_audit_failures"]) == 0 for row in episodes
        ),
        "sealed_test_panel_remains_closed": True,
    }
    return {
        "purpose": "Architecture diagnostic, not a performance claim.",
        "graph_definition": (
            "Six machine-agent nodes; an edge denotes overlap among currently "
            "feasible non-wait actions on a production operation or technician."
        ),
        "primary_condition": PRIMARY_CONDITION,
        "conditions": conditions,
        "locked_dynamic_subteam_rule": {
            "thresholds": {
                "mean_largest_component_size_max": MEAN_LARGEST_LIMIT,
                "fraction_epochs_with_size_2_or_3_min": SIZE_TWO_THREE_MINIMUM,
                "fraction_epochs_fully_connected_max": FULLY_CONNECTED_MAXIMUM,
            },
            "checks": decision_checks,
            "supports_dynamic_subteams": all(decision_checks.values()),
        },
        "gate": {"checks": audit_checks, "passed": all(audit_checks.values())},
    }


def _validate_source_run(source_run: Path) -> tuple[dict[str, Any], Path, Path]:
    manifest_path = source_run / "marl_budget_screening_manifest.json"
    scaled_config = source_run / "scaled_config.json"
    base_config = source_run / "benchmark_config.json"
    if not (
        manifest_path.is_file()
        and scaled_config.is_file()
        and base_config.is_file()
    ):
        raise FileNotFoundError(
            "Source run is missing its manifest or config snapshots"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "COMPLETED":
        raise ValueError("Source budget screen must be COMPLETED")
    if int(manifest.get("machine_count", -1)) != 6:
        raise ValueError("Source run must contain six machine agents")
    if manifest.get("future_test_panel_opened") is not False:
        raise ValueError("Source run did not preserve the sealed test panel")
    return manifest, base_config, scaled_config


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
    manifest_path = output_dir / "resource_graph_manifest.json"
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
    epochs: list[dict[str, Any]] = []
    for condition in tqdm(CONDITIONS, desc="Resource-graph policies", unit="policy"):
        policy = ValueDecompositionPolicy.load(model_paths[condition], device=device)
        for seed in tqdm(
            seeds, desc=f"Evaluate graph/{condition}", unit="episode", leave=False
        ):
            episode, episode_epochs = evaluate_episode(
                policy,
                config,
                condition=condition,
                train_seed=train_seed,
                seed=seed,
                device=device,
            )
            episodes.append(episode)
            epochs.extend(episode_epochs)
        manifest["completed_conditions"].append(condition)
        _write_csv(episodes, output_dir / "resource_graph_episodes.partial.csv")
        _write_csv(epochs, output_dir / "resource_graph_epochs.partial.csv")
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        del policy
    summary = summarize(
        episodes, epochs, expected_episodes=len(CONDITIONS) * len(seeds)
    )
    _write_csv(episodes, output_dir / "resource_graph_episodes.csv")
    _write_csv(epochs, output_dir / "resource_graph_epochs.csv")
    (output_dir / "resource_graph_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        status="COMPLETED",
        completed_at_utc=datetime.now(UTC).isoformat(),
        episode_count=len(episodes),
        epoch_count=len(epochs),
        gate=summary["gate"],
        architecture_decision=summary["locked_dynamic_subteam_rule"],
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["locked_dynamic_subteam_rule"], indent=2, sort_keys=True))
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
