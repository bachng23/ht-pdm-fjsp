"""Train IQL, QMIX and MAPPO and measure three-way MK01 contention."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
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
from ht_pdm_fjsp.marl_diagnostic import (
    DiagnosticPolicy,
    DiagnosticSettings,
    train_diagnostic_policy,
)
from ht_pdm_fjsp.marl_diagnostic_experiment import _git_revision
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config
from ht_pdm_fjsp.value_decomposition import (
    ValueDecompositionPolicy,
    ValueLearningSettings,
    train_value_policy,
)


IQL = "cooperative_iql"
QMIX = "qmix"
MAPPO = "independent_actor_mappo_global_critic"
CONDITIONS = (IQL, QMIX, MAPPO)
T_CRITICAL_975_DF4 = 2.776


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _settings_without_device(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key != "device"}


def _load_resume_state(
    output_dir: Path,
    *,
    profile: str,
    base_config_sha256: str,
    scaled_config_sha256: str,
    train_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
    total_timesteps: int,
    value_settings: ValueLearningSettings,
    mappo_settings: DiagnosticSettings,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    manifest_path = output_dir / "three_way_marl_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Resume manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    checks = {
        "status": manifest.get("status") == "RUNNING",
        "profile": manifest.get("profile") == profile,
        "base_config": manifest.get("base_config_sha256") == base_config_sha256,
        "scaled_config": manifest.get("scaled_config_sha256") == scaled_config_sha256,
        "conditions": tuple(manifest.get("conditions", ())) == CONDITIONS,
        "train_seeds": tuple(
            int(seed) for seed in manifest.get("train_seeds", ())
        )
        == train_seeds,
        "evaluation_seeds": tuple(
            int(seed) for seed in manifest.get("evaluation_seeds", ())
        )
        == evaluation_seeds,
        "holdout_closed": manifest.get("future_test_panel_opened") is False,
        "total_timesteps": int(manifest.get("total_timesteps_per_model", -1))
        == total_timesteps,
        "value_settings": _settings_without_device(manifest.get("value_settings", {}))
        == _settings_without_device(value_settings.__dict__),
        "mappo_settings": _settings_without_device(manifest.get("mappo_settings", {}))
        == _settings_without_device(mappo_settings.__dict__),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Resume contract mismatch: {', '.join(failed)}")

    episode_rows = _read_csv(output_dir / "three_way_marl_episodes.partial.csv")
    decision_rows = _read_csv(output_dir / "three_way_marl_decisions.partial.csv")
    coordination_rows = _read_csv(
        output_dir / "three_way_marl_coordination.partial.csv"
    )
    completed = list(dict.fromkeys(map(str, manifest.get("completed_cells", ()))))
    expected_keys = {
        f"{condition}:{seed}" for condition in CONDITIONS for seed in train_seeds
    }
    if not set(completed) <= expected_keys:
        raise ValueError("Resume manifest contains an unknown completed cell")
    for cell_key in completed:
        condition, raw_seed = cell_key.rsplit(":", 1)
        train_seed = int(raw_seed)
        episodes = [
            row
            for row in episode_rows
            if row["condition"] == condition
            and int(row["train_seed"]) == train_seed
        ]
        coordination = [
            row
            for row in coordination_rows
            if row["condition"] == condition
            and int(row["train_seed"]) == train_seed
        ]
        model_name = "model.pt" if condition in (IQL, QMIX) else "policy.pt"
        model_path = output_dir / condition / f"train_seed_{train_seed}" / model_name
        if (
            len(episodes) != len(evaluation_seeds)
            or len(coordination) != len(evaluation_seeds)
            or not model_path.is_file()
        ):
            raise ValueError(f"Completed cell is missing artifacts: {cell_key}")
    row_cells = {
        f"{row['condition']}:{int(row['train_seed'])}" for row in episode_rows
    }
    if not row_cells <= set(completed):
        raise ValueError("Partial episode CSV contains an uncommitted cell")
    return manifest, episode_rows, decision_rows, coordination_rows


def defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 512,
            "train_seeds": (75_000,),
            "evaluation_seeds": tuple(range(61_990, 61_993)),
            "replay_capacity": 512,
            "learning_starts": 32,
            "batch_size": 16,
            "n_steps": 64,
            "n_epochs": 1,
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "train_seeds": tuple(range(70_000, 75_000, 1_000)),
            "evaluation_seeds": tuple(range(61_700, 61_900)),
            "replay_capacity": 20_000,
            "learning_starts": 2_000,
            "batch_size": 256,
            "n_steps": 1_024,
            "n_epochs": 10,
        }
    raise ValueError(f"Unknown profile: {profile}")


def precedence_blocked_operations(env: MachineAgentsCTDEEnv) -> int:
    """Count downstream operations with idle capacity but unmet precedence."""

    blocked = 0
    for job in env.config.jobs:
        state = env.core.jobs[job.job_id]
        for operation_index in range(state.next_operation + 1, len(job.operations)):
            operation = job.operations[operation_index]
            if any(
                env.core.machines[alternative.machine_id].status == "idle"
                for alternative in operation.alternatives
            ):
                blocked += 1
    return blocked


def _actions(policy: Any, observation: dict[str, np.ndarray], device: str) -> np.ndarray:
    output = policy.act(observation, deterministic=True, device=device)
    return np.asarray(output[0] if isinstance(output, tuple) else output, dtype=np.int64)


def evaluate_episode(
    policy: Any,
    config: BenchmarkConfig,
    *,
    condition: str,
    train_seed: int,
    seed: int,
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    env = MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
    observation, _ = env.reset(seed=seed)
    rows: list[dict[str, Any]] = []
    totals = {
        "three_way_steps": 0,
        "precedence_blocked_at_three_way": 0,
        "maintenance_wait_at_three_way": 0.0,
        "elapsed_time_at_three_way": 0.0,
        "makespan_increment_at_three_way": 0.0,
    }
    episode_return = 0.0
    joint_step = 0
    while not env._done:
        actions = _actions(policy, observation, device)
        before_time = env.core.now
        before_wait = env.core.maintenance_wait_time
        before_makespan = env.core.production_makespan
        blocked = precedence_blocked_operations(env)
        observation, reward, _, _, info = env.step(actions)
        resolution = info["coordination"]
        three_way = int(
            resolution["production_conflicts"] > 0
            and resolution["technician_conflicts"] > 0
            and blocked > 0
        )
        elapsed = env.core.now - before_time
        wait_delta = env.core.maintenance_wait_time - before_wait
        makespan_delta = env.core.production_makespan - before_makespan
        if min(elapsed, wait_delta, makespan_delta) < -1e-9:
            raise AssertionError("Cumulative diagnostic metric decreased")
        totals["three_way_steps"] += three_way
        totals["precedence_blocked_at_three_way"] += three_way * blocked
        totals["maintenance_wait_at_three_way"] += three_way * wait_delta
        totals["elapsed_time_at_three_way"] += three_way * elapsed
        totals["makespan_increment_at_three_way"] += three_way * makespan_delta
        rows.append(
            {
                "condition": condition,
                "train_seed": train_seed,
                "seed": seed,
                "joint_step": joint_step,
                "simulation_time_before": before_time,
                "elapsed_time": elapsed,
                "production_conflicts": resolution["production_conflicts"],
                "technician_conflicts": resolution["technician_conflicts"],
                "precedence_blocked_operations": blocked,
                "three_way": three_way,
                "maintenance_wait_delta": wait_delta,
                "makespan_increment": makespan_delta,
                "reward": reward,
            }
        )
        episode_return += reward
        joint_step += 1
    result = env.result(condition)
    if not np.isclose(episode_return, -float(result.metrics["objective"])):
        raise AssertionError("Reward/objective identity failed")
    episode = {
        "condition": condition,
        "train_seed": train_seed,
        "seed": seed,
        "episode_return": episode_return,
        **result.metrics,
        **totals,
        "episode_has_three_way": int(totals["three_way_steps"] > 0),
    }
    coordination = {
        "condition": condition,
        "train_seed": train_seed,
        "seed": seed,
        **env.coordination_totals,
        **totals,
    }
    env.close()
    return episode, rows, coordination


def _describe(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def _per_train_means(rows: list[dict[str, Any]], condition: str) -> dict[int, dict[str, float]]:
    metrics = (
        "objective",
        "makespan",
        "maintenance_wait_time",
        "three_way_steps",
        "episode_has_three_way",
        "precedence_blocked_at_three_way",
        "maintenance_wait_at_three_way",
        "elapsed_time_at_three_way",
    )
    output: dict[int, dict[str, float]] = {}
    seeds = sorted({int(row["train_seed"]) for row in rows if row["condition"] == condition})
    for train_seed in seeds:
        selected = [
            row for row in rows
            if row["condition"] == condition and int(row["train_seed"]) == train_seed
        ]
        output[train_seed] = {
            metric: statistics.fmean(float(row[metric]) for row in selected)
            for metric in metrics
        }
    return output


def _paired_contrast(
    rows: list[dict[str, Any]], treatment: str, control: str
) -> dict[str, Any]:
    treatment_values = _per_train_means(rows, treatment)
    control_values = _per_train_means(rows, control)
    if treatment_values.keys() != control_values.keys():
        raise AssertionError("Algorithms do not share training seeds")
    if len(treatment_values) < 2:
        return {"status": "SMOKE_ONLY", "training_seed_count": len(treatment_values)}
    metrics: dict[str, Any] = {}
    for metric in treatment_values[next(iter(treatment_values))]:
        deltas = [
            treatment_values[seed][metric] - control_values[seed][metric]
            for seed in treatment_values
        ]
        mean = statistics.fmean(deltas)
        critical = T_CRITICAL_975_DF4 if len(deltas) == 5 else 1.96
        half = critical * statistics.stdev(deltas) / math.sqrt(len(deltas))
        metrics[metric] = {"mean": mean, "lower_95": mean - half, "upper_95": mean + half}
    return {
        "status": "DEVELOPMENT_PAIRED_INTERVAL",
        "training_seed_count": len(treatment_values),
        "metrics": metrics,
    }


def summarize(
    episode_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    coordination_rows: list[dict[str, Any]],
    *,
    expected_episodes: int,
) -> dict[str, Any]:
    per_condition: dict[str, Any] = {}
    audit_pass = True
    mechanism_pass = False
    for condition in CONDITIONS:
        episodes = [row for row in episode_rows if row["condition"] == condition]
        decisions = [row for row in decision_rows if row["condition"] == condition]
        coordination = [row for row in coordination_rows if row["condition"] == condition]
        three_way_steps = sum(int(row["three_way_steps"]) for row in episodes)
        three_way_episodes = sum(int(row["episode_has_three_way"]) for row in episodes)
        invalid = sum(int(row["invalid_executions"]) for row in coordination)
        duplicate_operations = sum(
            int(row["duplicate_operation_executions"]) for row in coordination
        )
        duplicate_technicians = sum(
            int(row["duplicate_technician_executions"]) for row in coordination
        )
        condition_audit = invalid == duplicate_operations == duplicate_technicians == 0
        audit_pass &= condition_audit
        incidence = three_way_episodes / len(episodes)
        mechanism_pass |= incidence >= 0.01 and three_way_steps >= 20
        three_way_rows = [row for row in decisions if int(row["three_way"]) == 1]
        per_condition[condition] = {
            "episodes": len(episodes),
            "joint_steps": len(decisions),
            "three_way_steps": three_way_steps,
            "three_way_steps_per_1000": 1000 * three_way_steps / len(decisions),
            "episodes_with_three_way": three_way_episodes,
            "three_way_episode_incidence": incidence,
            "precedence_blocked_at_three_way": sum(
                int(row["precedence_blocked_operations"]) for row in three_way_rows
            ),
            "maintenance_wait_at_three_way": sum(
                float(row["maintenance_wait_delta"]) for row in three_way_rows
            ),
            "elapsed_time_at_three_way": sum(
                float(row["elapsed_time"]) for row in three_way_rows
            ),
            "makespan_increment_at_three_way": sum(
                float(row["makespan_increment"]) for row in three_way_rows
            ),
            "metrics": {
                metric: _describe([float(row[metric]) for row in episodes])
                for metric in (
                    "objective", "makespan", "maintenance_wait_time", "failures",
                    "preventive_maintenance", "corrective_maintenance",
                )
            },
            "per_training_seed": _per_train_means(episode_rows, condition),
            "coordination_audit_passed": condition_audit,
        }
    all_precedence_positive = all(
        int(row["precedence_blocked_operations"]) > 0
        for row in decision_rows
        if int(row["three_way"]) == 1
    )
    checks = {
        "all_expected_episodes_completed": len(episode_rows) == expected_episodes,
        "all_coordination_audits_passed": audit_pass,
        "at_least_one_algorithm_has_measurable_three_way_contention": mechanism_pass,
        "all_three_way_steps_have_precedence_blocking": all_precedence_positive,
    }
    return {
        "primary_endpoint": "three-way steps per 1,000 joint steps and episode incidence",
        "per_condition": per_condition,
        "paired_contrasts_vs_mappo": {
            condition: _paired_contrast(episode_rows, condition, MAPPO)
            for condition in (IQL, QMIX)
        },
        "gate": {
            "checks": checks,
            "supports_three_way_marl_comparison": all(checks.values()),
            "note": "Development diagnostic; coincident step attribution is not counterfactual causality.",
        },
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    resume = bool(getattr(args, "resume", False))
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = defaults(args.profile)
    train_seeds = tuple(parse_seeds(args.train_seeds)) if args.train_seeds else profile["train_seeds"]
    evaluation_seeds = tuple(parse_seeds(args.evaluation_seeds)) if args.evaluation_seeds else profile["evaluation_seeds"]
    if set(evaluation_seeds) & set(range(62_000, 62_100)):
        raise ValueError("The sealed future test panel cannot be opened")
    base_path = Path(args.config).resolve()
    raw_config = base_path.read_bytes()
    base = BenchmarkConfig.from_json(base_path)
    config = build_condition_config(base, topology="two_specialists", multiplier=2.0)
    config_payload = json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    (output_dir / "benchmark_config.json").write_bytes(raw_config)
    (output_dir / "scaled_config.json").write_text(config_payload)
    device = resolve_device(args.device)
    total_timesteps = args.total_timesteps or profile["total_timesteps"]
    value_settings = ValueLearningSettings(
        total_timesteps=total_timesteps,
        replay_capacity=profile["replay_capacity"],
        learning_starts=min(profile["learning_starts"], total_timesteps // 2),
        batch_size=profile["batch_size"],
        train_frequency=4,
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_interval=max(32, total_timesteps // 100),
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_fraction=0.5,
        hidden_dim=128,
        mixer_hidden_dim=64,
        device=device,
    )
    mappo_settings = DiagnosticSettings(
        total_timesteps=total_timesteps,
        n_envs=1 if args.profile == "smoke" else 4,
        n_steps=profile["n_steps"],
        batch_size=profile["batch_size"],
        n_epochs=profile["n_epochs"],
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        max_grad_norm=0.5,
        actor_hidden_dim=128,
        critic_hidden_dim=128,
        device=device,
    )
    manifest_path = output_dir / "three_way_marl_manifest.json"
    new_manifest: dict[str, Any] = {
        "status": "RUNNING",
        "profile": args.profile,
        "git_commit": _git_revision(),
        "base_config_sha256": hashlib.sha256(raw_config).hexdigest(),
        "scaled_config_sha256": hashlib.sha256(config_payload.encode()).hexdigest(),
        "capacity_condition": "two_specialists_x2_0",
        "conditions": CONDITIONS,
        "train_seeds": train_seeds,
        "evaluation_seeds": evaluation_seeds,
        "reserved_future_test_seeds": list(range(62_000, 62_100)),
        "future_test_panel_opened": False,
        "total_timesteps_per_model": total_timesteps,
        "value_settings": value_settings.__dict__,
        "mappo_settings": mappo_settings.__dict__,
        "requested_device": args.device,
        "resolved_device": device,
        "completed_cells": [],
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{name: version(name) for name in ("numpy", "torch")},
        },
    }
    if resume:
        manifest, episode_rows, decision_rows, coordination_rows = _load_resume_state(
            output_dir,
            profile=args.profile,
            base_config_sha256=new_manifest["base_config_sha256"],
            scaled_config_sha256=new_manifest["scaled_config_sha256"],
            train_seeds=train_seeds,
            evaluation_seeds=evaluation_seeds,
            total_timesteps=total_timesteps,
            value_settings=value_settings,
            mappo_settings=mappo_settings,
        )
        completed_set = set(map(str, manifest["completed_cells"]))
        incomplete_directories = sorted(
            cell_key
            for cell_key in (
                f"{condition}:{seed}"
                for condition in CONDITIONS
                for seed in train_seeds
            )
            if cell_key not in completed_set
            and (
                output_dir
                / cell_key.rsplit(":", 1)[0]
                / f"train_seed_{cell_key.rsplit(':', 1)[1]}"
            ).exists()
        )
        manifest.setdefault("resume_history", []).append(
            {
                "resumed_at_utc": datetime.now(UTC).isoformat(),
                "git_commit": _git_revision(),
                "requested_device": args.device,
                "resolved_device": device,
                "skipped_completed_cells": sorted(completed_set),
                "restarted_incomplete_cells": incomplete_directories,
            }
        )
        manifest["requested_device"] = args.device
        manifest["resolved_device"] = device
        manifest["runtime"] = new_manifest["runtime"]
        manifest["value_settings"]["device"] = device
        manifest["mappo_settings"]["device"] = device
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        manifest = new_manifest
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        episode_rows = []
        decision_rows = []
        coordination_rows = []
    training_seconds: dict[str, float] = {
        key: float(value)
        for key, value in manifest.get("training_seconds", {}).items()
    }
    completed_cells = set(map(str, manifest["completed_cells"]))
    for condition in CONDITIONS:
        for train_seed in tqdm(train_seeds, desc=f"Train/evaluate {condition}", unit="model"):
            cell_key = f"{condition}:{train_seed}"
            if cell_key in completed_cells:
                continue
            cell = output_dir / condition / f"train_seed_{train_seed}"
            if condition in (IQL, QMIX):
                policy, elapsed = train_value_policy(
                    config,
                    value_settings,
                    cell,
                    algorithm="iql" if condition == IQL else "qmix",
                    train_seed=train_seed,
                    show_progress=True,
                )
            else:
                policy, elapsed = train_diagnostic_policy(
                    config,
                    mappo_settings,
                    cell,
                    train_seed=train_seed,
                    actor_mode="independent",
                    critic_mode="global",
                    include_broadcast_context=False,
                    show_progress=True,
                    wait_policy="safe_noop",
                )
            for seed in tqdm(evaluation_seeds, desc=f"Evaluate {condition}/{train_seed}", unit="episode"):
                episode, decisions, coordination = evaluate_episode(
                    policy,
                    config,
                    condition=condition,
                    train_seed=train_seed,
                    seed=seed,
                    device=device,
                )
                episode_rows.append(episode)
                decision_rows.extend(decisions)
                coordination_rows.append(coordination)
            training_seconds[cell_key] = elapsed
            manifest["completed_cells"].append(cell_key)
            completed_cells.add(cell_key)
            manifest["training_seconds"] = training_seconds
            _write_csv(episode_rows, output_dir / "three_way_marl_episodes.partial.csv")
            _write_csv(decision_rows, output_dir / "three_way_marl_decisions.partial.csv")
            _write_csv(coordination_rows, output_dir / "three_way_marl_coordination.partial.csv")
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            del policy
    expected = len(CONDITIONS) * len(train_seeds) * len(evaluation_seeds)
    unique = {(row["condition"], row["train_seed"], row["seed"]) for row in episode_rows}
    if len(episode_rows) != expected or len(unique) != expected:
        raise AssertionError("Incomplete evaluation panel")
    summary = summarize(
        episode_rows,
        decision_rows,
        coordination_rows,
        expected_episodes=expected,
    )
    _write_csv(episode_rows, output_dir / "three_way_marl_episodes.csv")
    _write_csv(decision_rows, output_dir / "three_way_marl_decisions.csv")
    _write_csv(coordination_rows, output_dir / "three_way_marl_coordination.csv")
    (output_dir / "three_way_marl_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    manifest.update(
        status="COMPLETED",
        episode_count=len(episode_rows),
        decision_count=len(decision_rows),
        training_seconds=training_seconds,
        gate=summary["gate"],
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary["gate"], indent=2, sort_keys=True))
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/brandimarte_mk01_ht_pdm.json")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--train-seeds")
    parser.add_argument("--evaluation-seeds")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
