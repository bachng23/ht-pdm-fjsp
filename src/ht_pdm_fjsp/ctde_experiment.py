"""Train and compare genuine machine-agent MAPPO with CTDE."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.ctde_mappo import MAPPOSettings, evaluate_mappo, train_mappo
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import evaluate_ppo, resolve_device
from ht_pdm_fjsp.shared_policy_experiment import summarize_architectures


CENTRALIZED = "shared_scorer_entropy"
CTDE = "machine_agents_mappo_ctde"
CONDITIONS = (CENTRALIZED, CTDE)


def defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 512,
            "n_envs": 1,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 2,
            "train_seeds": (10_000,),
            "validation_seeds": tuple(range(51_900, 51_905)),
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "n_envs": 4,
            "n_steps": 1_024,
            "batch_size": 256,
            "n_epochs": 10,
            "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
            "validation_seeds": tuple(range(52_000, 52_200)),
        }
    raise ValueError(f"Unknown profile: {profile}")


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _load_source(path: Path) -> dict[str, Any]:
    manifest_path = path / "architecture_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError("Centralized source run is not marked COMPLETED")
    if manifest.get("future_test_panel_opened"):
        raise ValueError("Centralized source opened the reserved test panel")
    return manifest


def promotion_decision(summary: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    comparison = summary["comparisons"][f"{CTDE}_minus_{CENTRALIZED}"]
    per_seed = comparison["per_training_seed_objective_delta"]
    objective_delta = comparison["delta_across_training_seeds"]["objective"][
        "mean"
    ]
    failure_delta = comparison["delta_across_training_seeds"]["failures"][
        "mean"
    ]
    checks = {
        "mean_objective_below_centralized": objective_delta < 0,
        "at_least_four_of_five_seed_deltas_nonpositive": sum(
            float(value) <= 0 for value in per_seed.values()
        )
        >= 4,
        "mean_failures_not_increased": failure_delta <= 0,
        "coordination_audit_passed": audit["status"] == "PASS",
    }
    return {
        "eligible_for_future_held_out_test": all(checks.values()),
        "checks": checks,
        "note": "Engineering promotion rule; not a significance claim.",
    }


def run(args: argparse.Namespace) -> Path:
    source_dir = Path(args.shared_source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _load_source(source_dir)
    profile = defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds)
        if args.train_seeds
        else profile["train_seeds"]
    )
    available = set(map(int, source_manifest["train_seeds"]))
    if not train_seeds or not set(train_seeds) <= available:
        raise ValueError("A requested training seed is missing from the source run")
    validation_seeds = tuple(
        parse_seeds(args.validation_seeds)
        if args.validation_seeds
        else profile["validation_seeds"]
    )
    forbidden = (
        set(range(20_000, 48_200))
        | set(range(50_000, 50_100))
        | set(map(int, source_manifest.get("validation_seeds", ())))
    )
    if not validation_seeds or set(validation_seeds) & forbidden:
        raise ValueError("Validation panel overlaps prior or reserved data")
    total_timesteps = args.total_timesteps or profile["total_timesteps"]
    n_envs = args.n_envs or profile["n_envs"]
    n_steps = args.n_steps or profile["n_steps"]
    batch_size = args.batch_size or profile["batch_size"]
    n_epochs = args.n_epochs or profile["n_epochs"]
    if total_timesteps % n_envs:
        raise ValueError("total_timesteps must be divisible by n_envs")
    if n_steps * n_envs % batch_size:
        raise ValueError("n_steps * n_envs must be divisible by batch_size")

    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    source_config_path = source_dir / "benchmark_config.json"
    if source_config_path.is_file() and hashlib.sha256(
        source_config_path.read_bytes()
    ).hexdigest() != config_sha256:
        raise ValueError("Benchmark config does not match centralized source")
    config = BenchmarkConfig.from_json(config_path)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    device = resolve_device(args.device)
    environment_probe = MachineAgentsCTDEEnv(config)
    architecture = {
        "agent_type": "machine",
        "agent_count": environment_probe.num_agents,
        "parameter_sharing": True,
        "actor_execution": "decentralized local observations",
        "critic_training": "centralized global state",
        "joint_action": True,
        "conflict_resolver": "deterministic round-robin machine arbitration",
        "local_feature_dim": environment_probe.LOCAL_FEATURE_DIM,
        "actor_global_state_input_dim": 0,
        "global_state_dim": environment_probe.global_state_dim,
        "max_local_actions": environment_probe.max_local_actions,
        "reward": "shared linear scalarization from benchmark",
    }
    environment_probe.close()
    settings = MAPPOSettings(
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        entropy_coefficient=args.entropy_coefficient,
        value_coefficient=args.value_coefficient,
        max_grad_norm=args.max_grad_norm,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        device=device,
    )
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "git_commit": _git_revision(),
        "config_sha256": config_sha256,
        "shared_source_run": str(source_dir),
        "conditions": CONDITIONS,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "architecture": architecture,
        "settings": settings.__dict__,
        "completed_training_cells": [],
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
    manifest_path = output_dir / "ctde_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    episode_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for train_seed in tqdm(
        train_seeds,
        desc="Centralized source validation",
        unit="model",
        disable=args.no_progress,
    ):
        model = MaskablePPO.load(
            source_dir
            / CENTRALIZED
            / f"train_seed_{train_seed}"
            / "maskable_ppo.zip",
            device=device,
        )
        evaluated = evaluate_ppo(
            model,
            config,
            validation_seeds,
            split="ctde_validation",
            show_progress=False,
        )
        for row in evaluated:
            row.update(
                policy=CENTRALIZED,
                condition=CENTRALIZED,
                train_seed=train_seed,
            )
        episode_rows.extend(evaluated)
        del model
    _write_csv(episode_rows, output_dir / "ctde_episodes.partial.csv")

    training_seconds: dict[str, float] = {}
    for train_seed in tqdm(
        train_seeds,
        desc="Machine-agent MAPPO",
        unit="model",
        disable=args.no_progress,
    ):
        cell_dir = output_dir / CTDE / f"train_seed_{train_seed}"
        model, elapsed = train_mappo(
            config,
            settings,
            cell_dir,
            train_seed=train_seed,
            show_progress=not args.no_progress,
        )
        evaluated, audit = evaluate_mappo(
            model,
            config,
            validation_seeds,
            condition=CTDE,
            device=device,
            show_progress=False,
        )
        for row in evaluated:
            row["train_seed"] = train_seed
        episode_rows.extend(evaluated)
        audit_rows.append({"train_seed": train_seed, **audit})
        training_seconds[str(train_seed)] = elapsed
        manifest["completed_training_cells"].append(train_seed)
        manifest["training_seconds"] = training_seconds
        _write_csv(episode_rows, output_dir / "ctde_episodes.partial.csv")
        _write_csv(audit_rows, output_dir / "ctde_coordination_audit.partial.csv")
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        del model

    expected = len(CONDITIONS) * len(train_seeds) * len(validation_seeds)
    unique = {
        (row["condition"], int(row["train_seed"]), int(row["seed"]))
        for row in episode_rows
    }
    if len(episode_rows) != expected or len(unique) != expected:
        raise ValueError("Incomplete CTDE validation panel")
    combined_audit = {
        key: sum(int(row.get(key, 0)) for row in audit_rows)
        for key in (
            "joint_steps",
            "proposals",
            "accepted",
            "waits",
            "production_conflicts",
            "technician_conflicts",
            "rejected",
            "invalid_executions",
            "duplicate_operation_executions",
            "duplicate_technician_executions",
        )
    }
    audit_checks = {
        "zero_invalid_executions": combined_audit["invalid_executions"] == 0,
        "zero_duplicate_operation_executions": (
            combined_audit["duplicate_operation_executions"] == 0
        ),
        "zero_duplicate_technician_executions": (
            combined_audit["duplicate_technician_executions"] == 0
        ),
        "actor_has_no_global_state_input": architecture[
            "actor_global_state_input_dim"
        ]
        == 0,
        "one_agent_per_machine": architecture["agent_count"]
        == len(config.machines),
        "centralized_critic_present": architecture["global_state_dim"] > 0,
    }
    audit = {
        "status": "PASS" if all(audit_checks.values()) else "FAIL",
        "checks": audit_checks,
        "totals": combined_audit,
    }
    if audit["status"] != "PASS":
        raise AssertionError("CTDE coordination audit failed")
    summary = summarize_architectures(
        episode_rows,
        conditions=CONDITIONS,
        baseline_condition=CENTRALIZED,
    )
    comparison = summary["comparisons"][f"{CTDE}_minus_{CENTRALIZED}"]
    comparison["per_training_seed_objective_delta"] = {
        str(seed): (
            summary["per_condition"][CTDE]["per_training_seed"][str(seed)][
                "objective"
            ]
            - summary["per_condition"][CENTRALIZED]["per_training_seed"][
                str(seed)
            ]["objective"]
        )
        for seed in train_seeds
    }
    summary["primary_endpoint"] = (
        "mean validation objective by independent training seed: "
        "machine-agent MAPPO CTDE minus centralized shared scorer"
    )
    summary["coordination_audit"] = audit
    summary["promotion_decision"] = promotion_decision(summary, audit)
    summary["future_test_panel_opened"] = False
    _write_csv(episode_rows, output_dir / "ctde_episodes.csv")
    _write_csv(audit_rows, output_dir / "ctde_coordination_audit.csv")
    (output_dir / "ctde_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        status="COMPLETED",
        episode_count=len(episode_rows),
        coordination_audit=audit,
        promotion_decision=summary["promotion_decision"],
        training_seconds=training_seconds,
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed. Artifacts: {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/minimal_benchmark.json")
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--train-seeds")
    parser.add_argument("--validation-seeds")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--actor-hidden-dim", type=int, default=64)
    parser.add_argument("--critic-hidden-dim", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
