"""Counterfactual diagnostic for route-preserving production reroutes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.gym_env import ActionDescriptor, HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.rl_experiment import resolve_device


POLICY = "route_preserving_residual_entropy"
SHARED_POLICY = "shared_scorer_entropy"
METRICS = (
    "objective",
    "makespan",
    "total_tardiness",
    "failures",
    "preventive_maintenance",
    "corrective_maintenance",
    "total_cost",
)
ACTION_FEATURE_NAMES = (
    "processing_time_normalized",
    "load_factor",
    "failure_probability",
    "effective_age_normalized",
    "slack_normalized",
)
EVENT_FIELDS = (
    "train_seed",
    "seed",
    "decision_index",
    "time",
    "feasible_action_count",
    "feasible_production_count",
    "shared_action",
    "residual_action",
    "shared_job_id",
    "shared_operation_index",
    "shared_operation_id",
    "shared_machine_id",
    "residual_job_id",
    "residual_operation_index",
    "residual_operation_id",
    "residual_machine_id",
    "base_probability_shared",
    "base_probability_residual",
    "production_probability_shared",
    "production_probability_residual",
    "final_probability_shared",
    "final_probability_residual",
    "base_logit_margin_shared_minus_residual",
    "residual_logit_advantage_residual_minus_shared",
    "combined_logit_advantage_residual_minus_shared",
    *(f"shared_{name}" for name in ACTION_FEATURE_NAMES),
    *(f"residual_{name}" for name in ACTION_FEATURE_NAMES),
    *(f"delta_{metric}_residual_minus_shared" for metric in METRICS),
    *(f"residual_branch_{metric}" for metric in METRICS),
    *(f"shared_branch_{metric}" for metric in METRICS),
    "residual_branch_trajectory_sha256",
    "shared_branch_trajectory_sha256",
)


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_manifest(path: Path, name: str) -> dict[str, Any]:
    manifest_path = path / name
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"Source run is not marked COMPLETED: {path}")
    if manifest.get("future_test_panel_opened"):
        raise ValueError(f"Source run opened the reserved test panel: {path}")
    return manifest


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    rows: list[dict[str, Any]], path: Path, *, fieldnames: Iterable[str] | None = None
) -> None:
    names = list(fieldnames or ())
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    if not names:
        raise ValueError(f"CSV schema is empty: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _base_observation(
    observation: dict[str, np.ndarray], model: MaskablePPO
) -> dict[str, np.ndarray]:
    return {key: observation[key] for key in model.observation_space.spaces}


def _predict(
    model: MaskablePPO,
    observation: dict[str, np.ndarray],
    action_mask: np.ndarray,
) -> int:
    action, _ = model.predict(
        _base_observation(observation, model),
        action_masks=action_mask,
        deterministic=True,
    )
    return int(np.asarray(action).item())


def _distribution_state(
    model: MaskablePPO,
    observation: dict[str, np.ndarray],
    action_mask: np.ndarray,
) -> dict[str, Any]:
    obs_tensor, _ = model.policy.obs_to_tensor(
        _base_observation(observation, model)
    )
    with th.no_grad():
        distribution = model.policy._distribution(obs_tensor, action_mask)
        action = int(distribution.mode().item())
        baseline = distribution.last_baseline_actions
        if baseline is None:
            raise AssertionError("Route-preserving distribution omitted base action")

        def array(value: th.Tensor) -> np.ndarray:
            return value.detach().cpu().numpy().reshape(-1)

        if distribution.base_logits is None or distribution.residual_logits is None:
            raise AssertionError("Route-preserving logits are unavailable")
        return {
            "action": action,
            "baseline_action": int(baseline.item()),
            "base_logits": array(distribution.base_logits),
            "residual_logits": array(distribution.residual_logits),
            "base_probabilities": array(distribution.base_distribution.probs),
            "production_probabilities": array(
                distribution.production_distribution.probs
            ),
            "final_probabilities": array(distribution.distribution.probs),
        }


def _rollout_branch(
    env: HTPdmFjspEnv,
    model: MaskablePPO,
    first_action: int,
    *,
    policy_name: str,
) -> dict[str, Any]:
    episode_return = float(env.cumulative_reward)
    observation, reward, _, _, info = env.step(first_action)
    if info["invalid_action"]:
        raise AssertionError("Counterfactual branch began with an invalid action")
    episode_return += float(reward)
    while not env._done:
        mask = env.action_masks()
        action = _predict(model, observation, mask)
        observation, reward, _, _, info = env.step(action)
        if info["invalid_action"]:
            raise AssertionError("Policy selected an invalid branch action")
        episode_return += float(reward)
    result = env.result(policy_name)
    metrics = {key: float(value) for key, value in result.metrics.items()}
    if not np.isclose(episode_return, -metrics["objective"], atol=1e-9):
        raise AssertionError("Branch reward/objective identity failed")
    metrics["trajectory_sha256"] = _trajectory_sha256(result)
    return metrics


def _trajectory_sha256(result: Any) -> str:
    payload = asdict(result)
    # Policy labels differ between the actual and forced branches by design.
    payload.pop("policy", None)
    serialized = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _descriptor_fields(prefix: str, descriptor: ActionDescriptor) -> dict[str, Any]:
    return {
        f"{prefix}_job_id": descriptor.job_id,
        f"{prefix}_operation_index": descriptor.operation_index,
        f"{prefix}_operation_id": descriptor.operation_id,
        f"{prefix}_machine_id": descriptor.machine_id,
    }


def _action_feature_fields(
    prefix: str, observation: dict[str, np.ndarray], action: int
) -> dict[str, float]:
    values = observation["action_features"][action, 4:9]
    return {
        f"{prefix}_{name}": float(value)
        for name, value in zip(ACTION_FEATURE_NAMES, values, strict=True)
    }


def _metrics_match(
    left: dict[str, float], right: dict[str, Any], *, atol: float = 1e-8
) -> bool:
    return all(
        np.isclose(float(left[key]), float(right[key]), rtol=1e-10, atol=atol)
        for key in METRICS
    )


def collect_episode(
    *,
    model: MaskablePPO,
    shared: MaskablePPO,
    config: BenchmarkConfig,
    train_seed: int,
    environment_seed: int,
    source_row: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = HTPdmFjspEnv(config=config, include_production_context=True)
    observation, _ = env.reset(seed=environment_seed)
    episode_return = 0.0
    events: list[dict[str, Any]] = []
    audit = {
        "decision_count": 0,
        "production_baseline_decisions": 0,
        "reroute_count": 0,
        "decision_kind_mismatches": 0,
        "nonproduction_action_mismatches": 0,
        "invalid_actions": 0,
        "internal_shared_action_mismatches": 0,
    }
    while not env._done:
        action_mask = env.action_masks()
        shared_action = _predict(shared, observation, action_mask)
        state = _distribution_state(model, observation, action_mask)
        residual_action = int(state["action"])
        shared_kind = env.actions[shared_action].kind
        residual_kind = env.actions[residual_action].kind
        audit["decision_count"] += 1
        audit["invalid_actions"] += int(not action_mask[residual_action])
        audit["internal_shared_action_mismatches"] += int(
            int(state["baseline_action"]) != shared_action
        )
        audit["decision_kind_mismatches"] += int(shared_kind != residual_kind)
        if shared_kind == "production":
            audit["production_baseline_decisions"] += 1
        else:
            audit["nonproduction_action_mismatches"] += int(
                shared_action != residual_action
            )

        if shared_kind == "production" and shared_action != residual_action:
            audit["reroute_count"] += 1
            shared_branch = _rollout_branch(
                env.clone(), model, shared_action, policy_name="forced_shared_choice"
            )
            residual_branch = _rollout_branch(
                env.clone(), model, residual_action, policy_name="residual_choice"
            )
            base_logits = state["base_logits"]
            residual_logits = state["residual_logits"]
            row: dict[str, Any] = {
                "train_seed": train_seed,
                "seed": environment_seed,
                "decision_index": env.decision_count + 1,
                "time": env.now,
                "feasible_action_count": int(action_mask.sum()),
                "feasible_production_count": int(
                    sum(
                        bool(action_mask[index]) and descriptor.kind == "production"
                        for index, descriptor in enumerate(env.actions)
                    )
                ),
                "shared_action": shared_action,
                "residual_action": residual_action,
                **_descriptor_fields("shared", env.actions[shared_action]),
                **_descriptor_fields("residual", env.actions[residual_action]),
                "base_probability_shared": state["base_probabilities"][shared_action],
                "base_probability_residual": state["base_probabilities"][residual_action],
                "production_probability_shared": state["production_probabilities"][shared_action],
                "production_probability_residual": state["production_probabilities"][residual_action],
                "final_probability_shared": state["final_probabilities"][shared_action],
                "final_probability_residual": state["final_probabilities"][residual_action],
                "base_logit_margin_shared_minus_residual": (
                    base_logits[shared_action] - base_logits[residual_action]
                ),
                "residual_logit_advantage_residual_minus_shared": (
                    residual_logits[residual_action] - residual_logits[shared_action]
                ),
                "combined_logit_advantage_residual_minus_shared": (
                    base_logits[residual_action]
                    + residual_logits[residual_action]
                    - base_logits[shared_action]
                    - residual_logits[shared_action]
                ),
                **_action_feature_fields("shared", observation, shared_action),
                **_action_feature_fields("residual", observation, residual_action),
            }
            for metric in METRICS:
                row[f"residual_branch_{metric}"] = residual_branch[metric]
                row[f"shared_branch_{metric}"] = shared_branch[metric]
                row[f"delta_{metric}_residual_minus_shared"] = (
                    residual_branch[metric] - shared_branch[metric]
                )
            row["residual_branch_trajectory_sha256"] = residual_branch[
                "trajectory_sha256"
            ]
            row["shared_branch_trajectory_sha256"] = shared_branch[
                "trajectory_sha256"
            ]
            events.append(row)

        observation, reward, _, _, info = env.step(residual_action)
        audit["invalid_actions"] += int(info["invalid_action"])
        episode_return += float(reward)

    result = env.result(POLICY)
    metrics = {key: float(value) for key, value in result.metrics.items()}
    if not np.isclose(episode_return, -metrics["objective"], atol=1e-9):
        raise AssertionError("Replay reward/objective identity failed")
    source_replay_match = _metrics_match(metrics, source_row) and int(
        source_row["decision_count"]
    ) == int(result.metrics["decision_count"])
    actual_trajectory_sha256 = _trajectory_sha256(result)
    residual_branch_replay_mismatches = sum(
        not (
            _metrics_match(
                metrics,
                {metric: event[f"residual_branch_{metric}"] for metric in METRICS},
            )
            and event["residual_branch_trajectory_sha256"]
            == actual_trajectory_sha256
        )
        for event in events
    )
    episode_row = {
        "train_seed": train_seed,
        "seed": environment_seed,
        "episode_return": episode_return,
        **{metric: metrics[metric] for metric in METRICS},
        **audit,
        "source_replay_match": source_replay_match,
        "residual_branch_replay_mismatches": residual_branch_replay_mismatches,
        "trajectory_sha256": actual_trajectory_sha256,
    }
    env.close()
    return episode_row, events


def _describe(values: Iterable[float]) -> dict[str, Any]:
    items = [float(value) for value in values]
    if not items:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(items),
        "mean": fmean(items),
        "min": min(items),
        "max": max(items),
    }


def summarize(
    episode_rows: list[dict[str, Any]], event_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    training_seeds = sorted({int(row["train_seed"]) for row in episode_rows})
    seed_rows: list[dict[str, Any]] = []
    for train_seed in training_seeds:
        episodes = [
            row for row in episode_rows if int(row["train_seed"]) == train_seed
        ]
        events = [row for row in event_rows if int(row["train_seed"]) == train_seed]
        objective_deltas = [
            float(row["delta_objective_residual_minus_shared"]) for row in events
        ]
        seed_rows.append(
            {
                "train_seed": train_seed,
                "episodes": len(episodes),
                "decisions": sum(int(row["decision_count"]) for row in episodes),
                "production_baseline_decisions": sum(
                    int(row["production_baseline_decisions"]) for row in episodes
                ),
                "reroutes": len(events),
                "rerouted_episodes": sum(int(row["reroute_count"]) > 0 for row in episodes),
                "mean_delta_objective_residual_minus_shared": (
                    fmean(objective_deltas) if objective_deltas else None
                ),
                "beneficial_reroutes": sum(value < -1e-8 for value in objective_deltas),
                "tied_reroutes": sum(abs(value) <= 1e-8 for value in objective_deltas),
                "harmful_reroutes": sum(value > 1e-8 for value in objective_deltas),
            }
        )

    totals = {
        "episodes": len(episode_rows),
        "decisions": sum(int(row["decision_count"]) for row in episode_rows),
        "production_baseline_decisions": sum(
            int(row["production_baseline_decisions"]) for row in episode_rows
        ),
        "reroutes": len(event_rows),
        "rerouted_episodes": sum(int(row["reroute_count"]) > 0 for row in episode_rows),
        "beneficial_reroutes": sum(
            float(row["delta_objective_residual_minus_shared"]) < -1e-8
            for row in event_rows
        ),
        "tied_reroutes": sum(
            abs(float(row["delta_objective_residual_minus_shared"])) <= 1e-8
            for row in event_rows
        ),
        "harmful_reroutes": sum(
            float(row["delta_objective_residual_minus_shared"]) > 1e-8
            for row in event_rows
        ),
    }
    totals["production_reroute_rate"] = (
        totals["reroutes"] / totals["production_baseline_decisions"]
        if totals["production_baseline_decisions"]
        else 0.0
    )
    checks = {
        "zero_decision_kind_mismatches": sum(
            int(row["decision_kind_mismatches"]) for row in episode_rows
        )
        == 0,
        "zero_nonproduction_action_mismatches": sum(
            int(row["nonproduction_action_mismatches"]) for row in episode_rows
        )
        == 0,
        "zero_invalid_actions": sum(
            int(row["invalid_actions"]) for row in episode_rows
        )
        == 0,
        "frozen_base_matches_shared_model": sum(
            int(row["internal_shared_action_mismatches"]) for row in episode_rows
        )
        == 0,
        "source_episode_replay_matches": all(
            bool(row["source_replay_match"]) for row in episode_rows
        ),
        "counterfactual_residual_branch_replays": sum(
            int(row["residual_branch_replay_mismatches"]) for row in episode_rows
        )
        == 0,
    }
    event_statistics = {
        f"delta_{metric}_residual_minus_shared": _describe(
            row[f"delta_{metric}_residual_minus_shared"] for row in event_rows
        )
        for metric in METRICS
    }
    event_statistics.update(
        {
            "base_logit_margin_shared_minus_residual": _describe(
                row["base_logit_margin_shared_minus_residual"] for row in event_rows
            ),
            "residual_logit_advantage_residual_minus_shared": _describe(
                row["residual_logit_advantage_residual_minus_shared"]
                for row in event_rows
            ),
        }
    )
    return seed_rows, {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "totals": totals,
        "event_statistics": event_statistics,
        "replication_unit": "independent PPO training seed",
        "interpretation_note": (
            "Reroute events and counterfactual branches are repeated descriptive "
            "observations within training seeds, not independent replications."
        ),
        "future_test_panel_opened": False,
    }


def _source_episode_index(
    route_dir: Path,
) -> dict[tuple[int, int], dict[str, str]]:
    rows = _read_csv(route_dir / "route_preserving_residual_episodes.csv")
    selected = [row for row in rows if row.get("condition") == POLICY]
    index = {(int(row["train_seed"]), int(row["seed"])): row for row in selected}
    if len(index) != len(selected):
        raise ValueError("Duplicate route-preserving source episode rows")
    return index


def run(args: argparse.Namespace) -> Path:
    started = time.monotonic()
    route_dir = Path(args.route_source_run).resolve()
    shared_dir = Path(args.shared_source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    route_manifest = _load_manifest(
        route_dir, "route_preserving_residual_manifest.json"
    )
    shared_manifest = _load_manifest(shared_dir, "architecture_manifest.json")
    if Path(route_manifest["shared_source_run"]).name != shared_dir.name:
        raise ValueError("Shared source provenance does not match route source")

    available_train = tuple(map(int, route_manifest["train_seeds"]))
    available_validation = tuple(map(int, route_manifest["validation_seeds"]))
    if args.train_seeds:
        train_seeds = tuple(parse_seeds(args.train_seeds))
    elif args.profile == "smoke":
        train_seeds = available_train[:1]
    else:
        train_seeds = available_train
    if args.validation_seeds:
        validation_seeds = tuple(parse_seeds(args.validation_seeds))
    elif args.profile == "smoke":
        validation_seeds = available_validation[:5]
    else:
        validation_seeds = available_validation
    if not train_seeds or not set(train_seeds) <= set(available_train):
        raise ValueError("Requested training seed is absent from route source")
    if not validation_seeds or not set(validation_seeds) <= set(available_validation):
        raise ValueError("Diagnostic may only replay the opened source validation panel")
    reserved_test = set(range(50_000, 50_100))
    if set(validation_seeds) & reserved_test:
        raise ValueError("Reserved future test panel must remain unopened")

    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if route_manifest.get("config_sha256") != config_sha256:
        raise ValueError("Benchmark config does not match route source")
    shared_hash = shared_manifest.get("config_sha256")
    if shared_hash is not None and shared_hash != config_sha256:
        raise ValueError("Benchmark config does not match shared source")
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    config = BenchmarkConfig.from_json(config_path)
    source_index = _source_episode_index(route_dir)
    requested_keys = {
        (train_seed, seed)
        for train_seed in train_seeds
        for seed in validation_seeds
    }
    if not requested_keys <= set(source_index):
        raise ValueError("Source episode table does not cover requested panel")

    device = resolve_device(args.device)
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "git_commit": _git_revision(),
        "config_sha256": config_sha256,
        "route_source_run": str(route_dir),
        "route_source_git_commit": route_manifest.get("git_commit"),
        "shared_source_run": str(shared_dir),
        "shared_source_git_commit": shared_manifest.get("git_commit"),
        "profile": args.profile,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "route_contract": route_manifest.get("route_contract"),
        "diagnostic_design": (
            "At each production reroute, clone the identical simulator state; "
            "force either residual or shared choice once, then continue both "
            "branches with the frozen route-preserving policy."
        ),
        "replication_unit": "independent PPO training seed",
        "completed_training_seeds": [],
        "requested_device": args.device,
        "resolved_device": device,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{
                name: version(name)
                for name in (
                    "numpy",
                    "stable-baselines3",
                    "sb3-contrib",
                    "torch",
                )
            },
        },
    }
    manifest_path = output_dir / "reroute_diagnostic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    episode_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(train_seeds) * len(validation_seeds),
        desc="Reroute counterfactuals",
        unit="episode",
        disable=args.no_progress,
    )
    try:
        for train_seed in train_seeds:
            model = MaskablePPO.load(
                route_dir
                / POLICY
                / f"train_seed_{train_seed}"
                / "maskable_ppo.zip",
                device=device,
            )
            shared = MaskablePPO.load(
                shared_dir
                / SHARED_POLICY
                / f"train_seed_{train_seed}"
                / "maskable_ppo.zip",
                device=device,
            )
            for environment_seed in validation_seeds:
                episode, events = collect_episode(
                    model=model,
                    shared=shared,
                    config=config,
                    train_seed=train_seed,
                    environment_seed=environment_seed,
                    source_row=source_index[(train_seed, environment_seed)],
                )
                episode_rows.append(episode)
                event_rows.extend(events)
                progress.update(1)
            del model, shared
            manifest["completed_training_seeds"].append(train_seed)
            _write_csv(
                episode_rows, output_dir / "episodes.partial.csv"
            )
            _write_csv(
                event_rows,
                output_dir / "reroute_events.partial.csv",
                fieldnames=EVENT_FIELDS,
            )
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    finally:
        progress.close()

    seed_rows, summary = summarize(episode_rows, event_rows)
    if summary["status"] != "PASS":
        raise AssertionError("Reroute diagnostic integrity checks failed")
    _write_csv(episode_rows, output_dir / "episodes.csv")
    _write_csv(event_rows, output_dir / "reroute_events.csv", fieldnames=EVENT_FIELDS)
    _write_csv(seed_rows, output_dir / "seed_summary.csv")
    (output_dir / "diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        status="COMPLETED",
        episode_count=len(episode_rows),
        reroute_event_count=len(event_rows),
        integrity=summary["checks"],
        elapsed_seconds=time.monotonic() - started,
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed. Artifacts: {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-source-run", required=True)
    parser.add_argument("--shared-source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/minimal_benchmark.json")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--train-seeds")
    parser.add_argument("--validation-seeds")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
