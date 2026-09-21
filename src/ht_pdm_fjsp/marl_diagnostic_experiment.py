"""Diagnose centralized critic, parameter sharing, and actor context in MARL."""

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
from ht_pdm_fjsp.ctde_mappo import MachineMAPPO, evaluate_mappo
from ht_pdm_fjsp.marl_diagnostic import (
    DiagnosticSettings,
    evaluate_diagnostic_policy,
    train_diagnostic_policy,
)
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import evaluate_ppo, resolve_device
from ht_pdm_fjsp.shared_policy_experiment import summarize_architectures


CENTRALIZED = "centralized_shared_scorer_entropy"
CURRENT_MAPPO = "parameter_shared_mappo_global_critic"
PS_IPPO = "parameter_shared_ippo_local_critic"
INDEPENDENT_MAPPO = "independent_actor_mappo_global_critic"
BROADCAST_MAPPO = "parameter_shared_mappo_broadcast_context"
CONDITIONS = (
    CENTRALIZED,
    CURRENT_MAPPO,
    PS_IPPO,
    INDEPENDENT_MAPPO,
    BROADCAST_MAPPO,
)
TRAINED_SPECS: dict[str, dict[str, Any]] = {
    PS_IPPO: {
        "actor_mode": "shared",
        "critic_mode": "local",
        "include_broadcast_context": False,
        "diagnostic_axis": "centralized critic",
    },
    INDEPENDENT_MAPPO: {
        "actor_mode": "independent",
        "critic_mode": "global",
        "include_broadcast_context": False,
        "diagnostic_axis": "parameter sharing",
    },
    BROADCAST_MAPPO: {
        "actor_mode": "shared",
        "critic_mode": "global",
        "include_broadcast_context": True,
        "diagnostic_axis": "actor information bottleneck",
    },
}


def defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 256,
            "n_envs": 1,
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 1,
            "train_seeds": (10_000,),
            "validation_seeds": tuple(range(52_900, 52_905)),
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "n_envs": 4,
            "n_steps": 1_024,
            "batch_size": 256,
            "n_epochs": 10,
            "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
            "validation_seeds": tuple(range(53_000, 53_200)),
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


def _read_completed_manifest(path: Path, filename: str) -> dict[str, Any]:
    manifest_path = path / filename
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"Source is not marked COMPLETED: {manifest_path}")
    if manifest.get("future_test_panel_opened"):
        raise ValueError(f"Source opened the reserved test panel: {manifest_path}")
    return manifest


def _audit_status(totals: dict[str, int]) -> dict[str, Any]:
    checks = {
        "zero_invalid_executions": totals.get("invalid_executions", 0) == 0,
        "zero_duplicate_operation_executions": totals.get(
            "duplicate_operation_executions", 0
        )
        == 0,
        "zero_duplicate_technician_executions": totals.get(
            "duplicate_technician_executions", 0
        )
        == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "totals": totals,
    }


def _combine_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = (
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
    by_condition: dict[str, dict[str, int]] = {}
    for row in rows:
        condition = str(row["condition"])
        totals = by_condition.setdefault(condition, {key: 0 for key in keys})
        for key in keys:
            totals[key] += int(row.get(key, 0))
    return {
        condition: _audit_status(totals)
        for condition, totals in by_condition.items()
    }


def _validate_source_config(
    source_dir: Path, config_sha256: str, source_name: str
) -> None:
    source_config = source_dir / "benchmark_config.json"
    if not source_config.is_file():
        raise FileNotFoundError(f"Missing {source_name} config: {source_config}")
    source_sha = hashlib.sha256(source_config.read_bytes()).hexdigest()
    if source_sha != config_sha256:
        raise ValueError(f"Benchmark config does not match {source_name} source")


def run(args: argparse.Namespace) -> Path:
    centralized_dir = Path(args.centralized_source_run).resolve()
    ctde_dir = Path(args.ctde_source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    centralized_manifest = _read_completed_manifest(
        centralized_dir, "architecture_manifest.json"
    )
    ctde_manifest = _read_completed_manifest(ctde_dir, "ctde_manifest.json")
    profile = defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds) if args.train_seeds else profile["train_seeds"]
    )
    available_centralized = set(map(int, centralized_manifest["train_seeds"]))
    available_ctde = set(map(int, ctde_manifest["completed_training_cells"]))
    if (
        not train_seeds
        or not set(train_seeds) <= available_centralized
        or not set(train_seeds) <= available_ctde
    ):
        raise ValueError("A requested training seed is missing from a source run")
    validation_seeds = tuple(
        parse_seeds(args.validation_seeds)
        if args.validation_seeds
        else profile["validation_seeds"]
    )
    prior_validation = set(map(int, centralized_manifest.get("validation_seeds", ())))
    prior_validation |= set(map(int, ctde_manifest.get("validation_seeds", ())))
    forbidden = set(range(20_000, 52_200)) | set(range(50_000, 50_100))
    forbidden |= prior_validation
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
    _validate_source_config(ctde_dir, config_sha256, "CTDE")
    config = BenchmarkConfig.from_json(config_path)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    device = resolve_device(args.device)
    base_probe = MachineAgentsCTDEEnv(config)
    context_probe = MachineAgentsCTDEEnv(config, include_broadcast_context=True)
    architectures = {
        CENTRALIZED: {
            "source": "completed shared-action PPO checkpoint",
            "actor_observation": "global shop state",
            "critic": "global",
        },
        CURRENT_MAPPO: {
            "source": "completed machine-agent MAPPO checkpoint",
            "actor_mode": "shared",
            "critic_mode": "global",
            "local_feature_dim": base_probe.local_feature_dim,
        },
        **{
            condition: {
                **spec,
                "local_feature_dim": (
                    context_probe.local_feature_dim
                    if spec["include_broadcast_context"]
                    else base_probe.local_feature_dim
                ),
            }
            for condition, spec in TRAINED_SPECS.items()
        },
    }
    base_probe.close()
    context_probe.close()
    settings = DiagnosticSettings(
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
        "centralized_source_run": str(centralized_dir),
        "ctde_source_run": str(ctde_dir),
        "conditions": CONDITIONS,
        "trained_conditions": tuple(TRAINED_SPECS),
        "architectures": architectures,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "settings": settings.__dict__,
        "completed_training_cells": [],
        "requested_device": args.device,
        "resolved_device": device,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            **{
                name: version(name)
                for name in ("numpy", "stable-baselines3", "sb3-contrib", "torch")
            },
        },
    }
    manifest_path = output_dir / "marl_diagnostic_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episode_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for train_seed in tqdm(
        train_seeds,
        desc="Centralized reference",
        unit="model",
        disable=args.no_progress,
    ):
        model = MaskablePPO.load(
            centralized_dir
            / "shared_scorer_entropy"
            / f"train_seed_{train_seed}"
            / "maskable_ppo.zip",
            device=device,
        )
        evaluated = evaluate_ppo(
            model,
            config,
            validation_seeds,
            split="marl_diagnostic_validation",
            show_progress=False,
        )
        for row in evaluated:
            row.update(policy=CENTRALIZED, condition=CENTRALIZED, train_seed=train_seed)
        episode_rows.extend(evaluated)
        del model
    _write_csv(episode_rows, output_dir / "marl_diagnostic_episodes.partial.csv")

    for train_seed in tqdm(
        train_seeds,
        desc="Current MAPPO reference",
        unit="model",
        disable=args.no_progress,
    ):
        model = MachineMAPPO.load(
            ctde_dir
            / "machine_agents_mappo_ctde"
            / f"train_seed_{train_seed}"
            / "mappo.pt",
            device=device,
        )
        evaluated, audit = evaluate_mappo(
            model,
            config,
            validation_seeds,
            condition=CURRENT_MAPPO,
            device=device,
            show_progress=False,
        )
        for row in evaluated:
            row.update(
                split="marl_diagnostic_validation",
                policy=CURRENT_MAPPO,
                train_seed=train_seed,
            )
        episode_rows.extend(evaluated)
        audit_rows.append({"condition": CURRENT_MAPPO, "train_seed": train_seed, **audit})
        del model
    _write_csv(episode_rows, output_dir / "marl_diagnostic_episodes.partial.csv")
    _write_csv(audit_rows, output_dir / "marl_diagnostic_coordination.partial.csv")

    training_seconds: dict[str, float] = {}
    for condition, spec in TRAINED_SPECS.items():
        for train_seed in tqdm(
            train_seeds,
            desc=condition,
            unit="model",
            disable=args.no_progress,
        ):
            cell_dir = output_dir / condition / f"train_seed_{train_seed}"
            model, elapsed = train_diagnostic_policy(
                config,
                settings,
                cell_dir,
                train_seed=train_seed,
                actor_mode=spec["actor_mode"],
                critic_mode=spec["critic_mode"],
                include_broadcast_context=spec["include_broadcast_context"],
                show_progress=not args.no_progress,
            )
            evaluated, audit = evaluate_diagnostic_policy(
                model,
                config,
                validation_seeds,
                condition=condition,
                device=device,
                show_progress=False,
            )
            for row in evaluated:
                row["train_seed"] = train_seed
            episode_rows.extend(evaluated)
            audit_rows.append({"condition": condition, "train_seed": train_seed, **audit})
            cell_key = f"{condition}:{train_seed}"
            training_seconds[cell_key] = elapsed
            manifest["completed_training_cells"].append(cell_key)
            manifest["training_seconds"] = training_seconds
            _write_csv(
                episode_rows, output_dir / "marl_diagnostic_episodes.partial.csv"
            )
            _write_csv(
                audit_rows, output_dir / "marl_diagnostic_coordination.partial.csv"
            )
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
        raise ValueError("Incomplete MARL diagnostic validation panel")
    coordination_audits = _combine_audits(audit_rows)
    if any(item["status"] != "PASS" for item in coordination_audits.values()):
        raise AssertionError("A MARL coordination audit failed")
    summary = summarize_architectures(
        episode_rows,
        conditions=CONDITIONS,
        baseline_condition=CURRENT_MAPPO,
    )
    for treatment in CONDITIONS:
        if treatment == CURRENT_MAPPO:
            continue
        comparison = summary["comparisons"][
            f"{treatment}_minus_{CURRENT_MAPPO}"
        ]
        comparison["per_training_seed_objective_delta"] = {
            str(seed): (
                summary["per_condition"][treatment]["per_training_seed"][str(seed)][
                    "objective"
                ]
                - summary["per_condition"][CURRENT_MAPPO]["per_training_seed"][
                    str(seed)
                ]["objective"]
            )
            for seed in train_seeds
        }
    summary.update(
        primary_endpoint=(
            "mean validation objective by independent training seed; each diagnostic "
            "condition minus current parameter-shared MAPPO"
        ),
        diagnostic_axes={
            PS_IPPO: "effect of replacing the global critic with a local critic",
            INDEPENDENT_MAPPO: "effect of removing actor parameter sharing",
            BROADCAST_MAPPO: "effect of adding observable aggregate shop context",
        },
        coordination_audits=coordination_audits,
        future_test_panel_opened=False,
    )
    _write_csv(episode_rows, output_dir / "marl_diagnostic_episodes.csv")
    _write_csv(audit_rows, output_dir / "marl_diagnostic_coordination.csv")
    (output_dir / "marl_diagnostic_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        status="COMPLETED",
        episode_count=len(episode_rows),
        coordination_audits=coordination_audits,
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
    parser.add_argument("--centralized-source-run", required=True)
    parser.add_argument("--ctde-source-run", required=True)
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
