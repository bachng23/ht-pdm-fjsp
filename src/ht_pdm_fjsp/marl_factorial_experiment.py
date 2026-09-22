"""Run the MARL actor-sharing by broadcast-context factorial follow-up."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.ctde_mappo import MachineMAPPO, evaluate_mappo
from ht_pdm_fjsp.marl_diagnostic import (
    DiagnosticPolicy,
    DiagnosticSettings,
    evaluate_diagnostic_policy,
    train_diagnostic_policy,
)
from ht_pdm_fjsp.marl_diagnostic_experiment import (
    BROADCAST_MAPPO,
    CENTRALIZED,
    CURRENT_MAPPO,
    INDEPENDENT_MAPPO,
    _combine_audits,
    _git_revision,
    _read_completed_manifest,
    _validate_source_config,
)
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.multi_seed_experiment import _stats
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import evaluate_ppo, resolve_device
from ht_pdm_fjsp.shared_policy_experiment import METRICS, summarize_architectures


COMBINED_MAPPO = "independent_actor_mappo_broadcast_context"
MATCHED_SHARED_MAPPO = "matched_capacity_shared_mappo_global_critic"
CONDITIONS = (
    CENTRALIZED,
    CURRENT_MAPPO,
    INDEPENDENT_MAPPO,
    BROADCAST_MAPPO,
    COMBINED_MAPPO,
    MATCHED_SHARED_MAPPO,
)
SOURCE_DIAGNOSTIC_CONDITIONS = (INDEPENDENT_MAPPO, BROADCAST_MAPPO)
TRAINED_SPECS: dict[str, dict[str, Any]] = {
    COMBINED_MAPPO: {
        "actor_mode": "independent",
        "critic_mode": "global",
        "include_broadcast_context": True,
        "actor_hidden_dim": 64,
        "diagnostic_axis": "actor-sharing by broadcast-context interaction",
    },
    MATCHED_SHARED_MAPPO: {
        "actor_mode": "shared",
        "critic_mode": "global",
        "include_broadcast_context": False,
        "actor_hidden_dim": None,
        "diagnostic_axis": "actor capacity matched to independent actors",
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
            "validation_seeds": tuple(range(53_900, 53_905)),
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "n_envs": 4,
            "n_steps": 1_024,
            "batch_size": 256,
            "n_epochs": 10,
            "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
            "validation_seeds": tuple(range(54_000, 54_200)),
        }
    raise ValueError(f"Unknown profile: {profile}")


def actor_parameter_count(
    input_dim: int, hidden_dim: int, actor_count: int = 1
) -> int:
    per_actor = (
        input_dim * hidden_dim
        + hidden_dim
        + hidden_dim * hidden_dim
        + hidden_dim
        + hidden_dim
        + 1
    )
    return actor_count * per_actor


def matched_shared_hidden_dim(
    *, input_dim: int, independent_hidden_dim: int, agent_count: int
) -> int:
    target = actor_parameter_count(
        input_dim, independent_hidden_dim, actor_count=agent_count
    )
    return min(
        range(1, 513),
        key=lambda hidden: abs(actor_parameter_count(input_dim, hidden) - target),
    )


def _available_diagnostic_seeds(
    manifest: dict[str, Any], condition: str
) -> set[int]:
    prefix = f"{condition}:"
    return {
        int(cell.removeprefix(prefix))
        for cell in manifest["completed_training_cells"]
        if str(cell).startswith(prefix)
    }


def _paired_contrast(
    summary: dict[str, Any],
    *,
    treatment: str,
    control: str,
    train_seeds: tuple[int, ...],
) -> dict[str, Any]:
    per_metric: dict[str, Any] = {}
    per_seed: dict[str, dict[str, float]] = {}
    for seed in train_seeds:
        treatment_values = summary["per_condition"][treatment]["per_training_seed"][
            str(seed)
        ]
        control_values = summary["per_condition"][control]["per_training_seed"][
            str(seed)
        ]
        per_seed[str(seed)] = {
            metric: treatment_values[metric] - control_values[metric]
            for metric in METRICS
        }
    for metric in METRICS:
        per_metric[metric] = _stats(
            per_seed[str(seed)][metric] for seed in train_seeds
        )
    return {
        "treatment": treatment,
        "control": control,
        "delta_across_training_seeds": per_metric,
        "per_training_seed": per_seed,
    }


def _factorial_interaction(
    summary: dict[str, Any], train_seeds: tuple[int, ...]
) -> dict[str, Any]:
    per_seed: dict[str, dict[str, float]] = {}
    for seed in train_seeds:
        values = {
            condition: summary["per_condition"][condition]["per_training_seed"][
                str(seed)
            ]
            for condition in (
                CURRENT_MAPPO,
                INDEPENDENT_MAPPO,
                BROADCAST_MAPPO,
                COMBINED_MAPPO,
            )
        }
        per_seed[str(seed)] = {
            metric: (
                values[COMBINED_MAPPO][metric]
                - values[INDEPENDENT_MAPPO][metric]
                - values[BROADCAST_MAPPO][metric]
                + values[CURRENT_MAPPO][metric]
            )
            for metric in METRICS
        }
    return {
        "definition": "combined - independent - broadcast + shared_no_context",
        "delta_across_training_seeds": {
            metric: _stats(per_seed[str(seed)][metric] for seed in train_seeds)
            for metric in METRICS
        },
        "per_training_seed": per_seed,
    }


def promotion_decision(
    primary_contrast: dict[str, Any], coordination_audits: dict[str, Any]
) -> dict[str, Any]:
    per_seed = primary_contrast["per_training_seed"]
    checks = {
        "mean_objective_below_independent_incumbent": primary_contrast[
            "delta_across_training_seeds"
        ]["objective"]["mean"]
        < 0,
        "at_least_four_of_five_seed_deltas_nonpositive": sum(
            values["objective"] <= 0 for values in per_seed.values()
        )
        >= 4,
        "mean_failures_not_increased": primary_contrast[
            "delta_across_training_seeds"
        ]["failures"]["mean"]
        <= 0,
        "coordination_audit_passed": coordination_audits[COMBINED_MAPPO]["status"]
        == "PASS",
    }
    return {
        "eligible_for_future_held_out_test": all(checks.values()),
        "checks": checks,
        "note": "Engineering gate only; the final test remains separately authorized.",
    }


def run(args: argparse.Namespace) -> Path:
    centralized_dir = Path(args.centralized_source_run).resolve()
    ctde_dir = Path(args.ctde_source_run).resolve()
    diagnostic_dir = Path(args.diagnostic_source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    centralized_manifest = _read_completed_manifest(
        centralized_dir, "architecture_manifest.json"
    )
    ctde_manifest = _read_completed_manifest(ctde_dir, "ctde_manifest.json")
    diagnostic_manifest = _read_completed_manifest(
        diagnostic_dir, "marl_diagnostic_manifest.json"
    )
    profile = defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds) if args.train_seeds else profile["train_seeds"]
    )
    available = set(map(int, centralized_manifest["train_seeds"]))
    available &= set(map(int, ctde_manifest["completed_training_cells"]))
    for condition in SOURCE_DIAGNOSTIC_CONDITIONS:
        available &= _available_diagnostic_seeds(diagnostic_manifest, condition)
    if not train_seeds or not set(train_seeds) <= available:
        raise ValueError("A requested training seed is missing from a source run")
    validation_seeds = tuple(
        parse_seeds(args.validation_seeds)
        if args.validation_seeds
        else profile["validation_seeds"]
    )
    prior_validation = set(map(int, centralized_manifest.get("validation_seeds", ())))
    prior_validation |= set(map(int, ctde_manifest.get("validation_seeds", ())))
    prior_validation |= set(map(int, diagnostic_manifest.get("validation_seeds", ())))
    forbidden = set(range(20_000, 53_200)) | set(range(50_000, 50_100))
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
    _validate_source_config(diagnostic_dir, config_sha256, "diagnostic")
    config = BenchmarkConfig.from_json(config_path)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    device = resolve_device(args.device)
    base_probe = MachineAgentsCTDEEnv(config)
    context_probe = MachineAgentsCTDEEnv(config, include_broadcast_context=True)
    matched_hidden_dim = matched_shared_hidden_dim(
        input_dim=base_probe.local_feature_dim,
        independent_hidden_dim=args.actor_hidden_dim,
        agent_count=base_probe.num_agents,
    )
    independent_actor_parameters = actor_parameter_count(
        base_probe.local_feature_dim,
        args.actor_hidden_dim,
        actor_count=base_probe.num_agents,
    )
    matched_actor_parameters = actor_parameter_count(
        base_probe.local_feature_dim, matched_hidden_dim
    )
    relative_parameter_gap = abs(
        matched_actor_parameters - independent_actor_parameters
    ) / independent_actor_parameters
    if relative_parameter_gap > 0.005:
        raise ValueError("Unable to match shared and independent actor capacity")
    architectures = {
        CENTRALIZED: {"source": "completed centralized checkpoint"},
        CURRENT_MAPPO: {"source": "completed MAPPO checkpoint"},
        INDEPENDENT_MAPPO: {"source": "completed diagnostic checkpoint"},
        BROADCAST_MAPPO: {"source": "completed diagnostic checkpoint"},
        COMBINED_MAPPO: {
            **TRAINED_SPECS[COMBINED_MAPPO],
            "local_feature_dim": context_probe.local_feature_dim,
            "actor_parameter_count": actor_parameter_count(
                context_probe.local_feature_dim,
                args.actor_hidden_dim,
                actor_count=context_probe.num_agents,
            ),
        },
        MATCHED_SHARED_MAPPO: {
            **TRAINED_SPECS[MATCHED_SHARED_MAPPO],
            "actor_hidden_dim": matched_hidden_dim,
            "local_feature_dim": base_probe.local_feature_dim,
            "actor_parameter_count": matched_actor_parameters,
            "target_independent_actor_parameter_count": independent_actor_parameters,
            "relative_parameter_gap": relative_parameter_gap,
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
        "diagnostic_source_run": str(diagnostic_dir),
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
    manifest_path = output_dir / "marl_factorial_manifest.json"
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
            split="marl_factorial_validation",
            show_progress=False,
        )
        for row in evaluated:
            row.update(policy=CENTRALIZED, condition=CENTRALIZED, train_seed=train_seed)
        episode_rows.extend(evaluated)
        del model

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
                split="marl_factorial_validation",
                policy=CURRENT_MAPPO,
                train_seed=train_seed,
            )
        episode_rows.extend(evaluated)
        audit_rows.append({"condition": CURRENT_MAPPO, "train_seed": train_seed, **audit})
        del model

    for condition in SOURCE_DIAGNOSTIC_CONDITIONS:
        for train_seed in tqdm(
            train_seeds,
            desc=f"Source {condition}",
            unit="model",
            disable=args.no_progress,
        ):
            model = DiagnosticPolicy.load(
                diagnostic_dir / condition / f"train_seed_{train_seed}" / "policy.pt",
                device=device,
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
            del model
    _write_csv(episode_rows, output_dir / "marl_factorial_episodes.partial.csv")
    _write_csv(audit_rows, output_dir / "marl_factorial_coordination.partial.csv")

    training_seconds: dict[str, float] = {}
    for condition, spec in TRAINED_SPECS.items():
        actor_hidden_dim = (
            matched_hidden_dim
            if condition == MATCHED_SHARED_MAPPO
            else int(spec["actor_hidden_dim"])
        )
        condition_settings = replace(settings, actor_hidden_dim=actor_hidden_dim)
        for train_seed in tqdm(
            train_seeds,
            desc=condition,
            unit="model",
            disable=args.no_progress,
        ):
            cell_dir = output_dir / condition / f"train_seed_{train_seed}"
            model, elapsed = train_diagnostic_policy(
                config,
                condition_settings,
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
                episode_rows, output_dir / "marl_factorial_episodes.partial.csv"
            )
            _write_csv(
                audit_rows, output_dir / "marl_factorial_coordination.partial.csv"
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
        raise ValueError("Incomplete MARL factorial validation panel")
    coordination_audits = _combine_audits(audit_rows)
    if any(item["status"] != "PASS" for item in coordination_audits.values()):
        raise AssertionError("A MARL coordination audit failed")
    summary = summarize_architectures(
        episode_rows,
        conditions=CONDITIONS,
        baseline_condition=CURRENT_MAPPO,
    )
    primary_contrast = _paired_contrast(
        summary,
        treatment=COMBINED_MAPPO,
        control=INDEPENDENT_MAPPO,
        train_seeds=train_seeds,
    )
    capacity_contrast = _paired_contrast(
        summary,
        treatment=INDEPENDENT_MAPPO,
        control=MATCHED_SHARED_MAPPO,
        train_seeds=train_seeds,
    )
    factorial_interaction = _factorial_interaction(summary, train_seeds)
    summary.update(
        primary_endpoint=(
            "mean validation objective by independent training seed: combined "
            "independent+broadcast MAPPO minus independent-actor incumbent"
        ),
        primary_contrast=primary_contrast,
        capacity_contrast=capacity_contrast,
        factorial_interaction=factorial_interaction,
        coordination_audits=coordination_audits,
        promotion_decision=promotion_decision(primary_contrast, coordination_audits),
        future_test_panel_opened=False,
    )
    _write_csv(episode_rows, output_dir / "marl_factorial_episodes.csv")
    _write_csv(audit_rows, output_dir / "marl_factorial_coordination.csv")
    (output_dir / "marl_factorial_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        status="COMPLETED",
        episode_count=len(episode_rows),
        coordination_audits=coordination_audits,
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
    parser.add_argument("--centralized-source-run", required=True)
    parser.add_argument("--ctde-source-run", required=True)
    parser.add_argument("--diagnostic-source-run", required=True)
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
