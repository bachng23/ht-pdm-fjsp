"""Train and evaluate conflict-aware route-preserving residual PPO."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.conflict_aware_residual_policy import (
    ConflictAwareRoutePreservingResidualPolicy,
    build_production_conflict_matrix,
)
from ht_pdm_fjsp.entity_conditioned_experiment import _paired_summary
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import (
    ExperimentSettings,
    evaluate_ppo,
    resolve_device,
    train_model,
)
from ht_pdm_fjsp.shared_policy_experiment import summarize_architectures


SHARED = "shared_scorer_entropy"
CURRENT = "route_preserving_residual_entropy"
TREATMENT = "conflict_aware_route_preserving_residual_entropy"
CONDITIONS = (SHARED, CURRENT, TREATMENT)


def defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 256,
            "n_envs": 1,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 2,
            "train_seeds": (10_000,),
            "validation_seeds": tuple(range(48_000, 48_005)),
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "n_envs": 4,
            "n_steps": 1_024,
            "batch_size": 256,
            "n_epochs": 10,
            "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
            "validation_seeds": tuple(range(48_000, 48_200)),
        }
    raise ValueError(f"Unknown profile: {profile}")


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


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _base_observation(
    observation: dict[str, np.ndarray], model: MaskablePPO
) -> dict[str, np.ndarray]:
    return {key: observation[key] for key in model.observation_space.spaces}


def _tag(
    target: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    condition: str,
    train_seed: int,
) -> None:
    for row in rows:
        row.update(condition=condition, policy=condition, train_seed=train_seed)
    target.extend(rows)


def evaluate_with_conflict_audit(
    model: MaskablePPO,
    baseline: MaskablePPO,
    config: BenchmarkConfig,
    seeds: Iterable[int],
    *,
    condition: str,
    enforce_conflicts: bool,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env_template = HTPdmFjspEnv(config=config, include_production_context=True)
    conflicts = np.asarray(build_production_conflict_matrix(env_template.actions))
    rows: list[dict[str, Any]] = []
    audit = {
        "condition": condition,
        "enforce_conflicts": enforce_conflicts,
        "episodes": 0,
        "decisions": 0,
        "production_baseline_decisions": 0,
        "production_reroutes": 0,
        "same_machine_reroutes": 0,
        "same_operation_reroutes": 0,
        "nonconflicting_production_reroutes": 0,
        "decision_kind_mismatches": 0,
        "nonproduction_action_mismatches": 0,
        "invalid_actions": 0,
    }
    for seed in tqdm(
        tuple(seeds),
        desc=f"Audit {condition}",
        unit="episode",
        disable=not show_progress,
    ):
        env = HTPdmFjspEnv(config=config, include_production_context=True)
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        while not env._done:
            mask = env.action_masks()
            baseline_action, _ = baseline.predict(
                _base_observation(observation, baseline),
                action_masks=mask,
                deterministic=True,
            )
            action, _ = model.predict(
                observation, action_masks=mask, deterministic=True
            )
            baseline_index = int(np.asarray(baseline_action).item())
            action_index = int(np.asarray(action).item())
            baseline_descriptor = env.actions[baseline_index]
            descriptor = env.actions[action_index]
            audit["decisions"] += 1
            audit["invalid_actions"] += int(not mask[action_index])
            audit["decision_kind_mismatches"] += int(
                baseline_descriptor.kind != descriptor.kind
            )
            if baseline_descriptor.kind == "production":
                audit["production_baseline_decisions"] += 1
                if baseline_index != action_index:
                    audit["production_reroutes"] += 1
                    same_machine = (
                        baseline_descriptor.machine_id == descriptor.machine_id
                    )
                    same_operation = (
                        baseline_descriptor.job_id == descriptor.job_id
                        and baseline_descriptor.operation_index
                        == descriptor.operation_index
                    )
                    audit["same_machine_reroutes"] += int(same_machine)
                    audit["same_operation_reroutes"] += int(same_operation)
                    audit["nonconflicting_production_reroutes"] += int(
                        not conflicts[baseline_index, action_index]
                    )
            else:
                audit["nonproduction_action_mismatches"] += int(
                    baseline_index != action_index
                )
            observation, reward, _, _, _ = env.step(action_index)
            episode_return += float(reward)
        result = env.result(condition)
        if not np.isclose(episode_return, -float(result.metrics["objective"])):
            raise AssertionError("PPO reward/objective identity failed")
        rows.append(
            {
                "split": "conflict_aware_validation",
                "policy": condition,
                "seed": seed,
                "episode_return": episode_return,
                **result.metrics,
            }
        )
        audit["episodes"] += 1
        env.close()
    env_template.close()
    if enforce_conflicts and audit["nonconflicting_production_reroutes"]:
        raise AssertionError("Treatment rerouted outside a production conflict set")
    return rows, audit


def summarize_audit(row: dict[str, Any]) -> dict[str, Any]:
    checks = {
        "zero_decision_kind_mismatches": row["decision_kind_mismatches"] == 0,
        "zero_nonproduction_action_mismatches": (
            row["nonproduction_action_mismatches"] == 0
        ),
        "zero_invalid_actions": row["invalid_actions"] == 0,
    }
    if row["enforce_conflicts"]:
        checks["zero_nonconflicting_production_reroutes"] = (
            row["nonconflicting_production_reroutes"] == 0
        )
    production = int(row["production_baseline_decisions"])
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "totals": {
            key: value
            for key, value in row.items()
            if key not in {"condition", "enforce_conflicts"}
        }
        | {
            "production_reroute_rate": (
                int(row["production_reroutes"]) / production
                if production
                else 0.0
            )
        },
    }


def promotion_decision(
    summary: dict[str, Any], treatment_audit: dict[str, Any]
) -> dict[str, Any]:
    comparison = summary["comparisons"][f"{TREATMENT}_minus_{SHARED}"]
    seed_deltas = comparison["per_training_seed_objective_delta"]
    objective_mean = comparison["delta_across_training_seeds"]["objective"][
        "mean"
    ]
    failures_mean = comparison["delta_across_training_seeds"]["failures"][
        "mean"
    ]
    treatment_std = summary["per_condition"][TREATMENT][
        "across_training_seeds"
    ]["objective"]["std"]
    current_std = summary["per_condition"][CURRENT]["across_training_seeds"][
        "objective"
    ]["std"]
    checks = {
        "mean_objective_below_shared": objective_mean < 0,
        "at_least_four_of_five_seed_deltas_nonpositive": sum(
            float(value) <= 0 for value in seed_deltas.values()
        )
        >= 4,
        "mean_failures_not_increased": failures_mean <= 0,
        "objective_seed_dispersion_not_above_current": (
            treatment_std <= current_std
        ),
        "treatment_route_and_conflict_audit_passed": (
            treatment_audit["status"] == "PASS"
        ),
    }
    return {
        "eligible_for_future_held_out_test": all(checks.values()),
        "checks": checks,
        "note": "Engineering promotion rule; not a significance claim.",
    }


def run(args: argparse.Namespace) -> Path:
    shared_dir = Path(args.shared_source_run).resolve()
    route_dir = Path(args.route_source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_manifest = _load_manifest(shared_dir, "architecture_manifest.json")
    route_manifest = _load_manifest(
        route_dir, "route_preserving_residual_manifest.json"
    )
    if Path(route_manifest["shared_source_run"]).name != shared_dir.name:
        raise ValueError("Shared source provenance does not match route source")

    profile = defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds)
        if args.train_seeds
        else profile["train_seeds"]
    )
    available = set(map(int, shared_manifest["train_seeds"])) & set(
        map(int, route_manifest["train_seeds"])
    )
    if not train_seeds or not set(train_seeds) <= available:
        raise ValueError("A requested training seed is missing from source runs")
    validation_seeds = tuple(
        parse_seeds(args.validation_seeds)
        if args.validation_seeds
        else profile["validation_seeds"]
    )
    forbidden = (
        set(range(20_000, 48_000))
        | set(map(int, shared_manifest["validation_seeds"]))
        | set(map(int, route_manifest["validation_seeds"]))
        | set(range(50_000, 50_100))
    )
    if not validation_seeds or set(validation_seeds) & forbidden:
        raise ValueError("Validation panel overlaps prior or reserved data")

    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if route_manifest.get("config_sha256") != config_sha256:
        raise ValueError("Benchmark config does not match route source")
    config = BenchmarkConfig.from_json(config_path)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    catalog_env = HTPdmFjspEnv(config=config, include_production_context=True)
    conflict_matrix = build_production_conflict_matrix(catalog_env.actions)
    catalog_env.close()
    conflict_sha256 = hashlib.sha256(
        json.dumps(conflict_matrix, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    device = resolve_device(args.device)
    common = {
        "profile": args.profile,
        "total_timesteps": args.total_timesteps or profile["total_timesteps"],
        "n_envs": args.n_envs or profile["n_envs"],
        "n_steps": args.n_steps or profile["n_steps"],
        "batch_size": args.batch_size or profile["batch_size"],
        "n_epochs": args.n_epochs or profile["n_epochs"],
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "validation_seeds": validation_seeds,
        "test_seeds": (),
        "device": device,
        "include_slow_baselines": False,
    }
    if common["n_steps"] * common["n_envs"] % common["batch_size"]:
        raise ValueError("Invalid PPO batch geometry")

    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "git_commit": _git_revision(),
        "config_sha256": config_sha256,
        "production_conflict_matrix_sha256": conflict_sha256,
        "shared_source_run": str(shared_dir),
        "route_source_run": str(route_dir),
        "conditions": CONDITIONS,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "common_settings": common,
        "entropy_coefficient": args.entropy_coefficient,
        "context_hidden_dim": args.context_hidden_dim,
        "max_residual_logit": args.max_residual_logit,
        "freeze_base_actor": True,
        "conflict_rule": (
            "same machine or same job-operation as the shared production action"
        ),
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
    manifest_path = output_dir / "conflict_aware_residual_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    episode_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for train_seed in tqdm(
        train_seeds,
        desc="Shared source validation",
        unit="model",
        disable=args.no_progress,
    ):
        shared = MaskablePPO.load(
            shared_dir / SHARED / f"train_seed_{train_seed}" / "maskable_ppo.zip",
            device=device,
        )
        evaluated = evaluate_ppo(
            shared,
            config,
            validation_seeds,
            split="conflict_aware_validation",
            show_progress=False,
        )
        _tag(episode_rows, evaluated, SHARED, train_seed)
        del shared

    for train_seed in tqdm(
        train_seeds,
        desc="Current residual validation",
        unit="model",
        disable=args.no_progress,
    ):
        shared = MaskablePPO.load(
            shared_dir / SHARED / f"train_seed_{train_seed}" / "maskable_ppo.zip",
            device=device,
        )
        current = MaskablePPO.load(
            route_dir / CURRENT / f"train_seed_{train_seed}" / "maskable_ppo.zip",
            device=device,
        )
        evaluated, audit = evaluate_with_conflict_audit(
            current,
            shared,
            config,
            validation_seeds,
            condition=CURRENT,
            enforce_conflicts=False,
            show_progress=False,
        )
        _tag(episode_rows, evaluated, CURRENT, train_seed)
        audit_rows.append({"train_seed": train_seed, **audit})
        del current, shared
    _write_csv(
        episode_rows, output_dir / "conflict_aware_residual_episodes.partial.csv"
    )

    training_seconds: dict[str, float] = {}
    for train_seed in tqdm(
        train_seeds,
        desc="Conflict-aware residual PPO",
        unit="model",
        disable=args.no_progress,
    ):
        baseline = MaskablePPO.load(
            shared_dir / SHARED / f"train_seed_{train_seed}" / "maskable_ppo.zip",
            device=device,
        )

        def initialize(model: MaskablePPO, source=baseline) -> None:
            model.policy.load_shared_base(source.policy)

        model, elapsed = train_model(
            config,
            ExperimentSettings(train_seed=train_seed, **common),
            output_dir / TREATMENT / f"train_seed_{train_seed}",
            show_progress=not args.no_progress,
            policy=ConflictAwareRoutePreservingResidualPolicy,
            policy_kwargs={
                "production_conflict_matrix": conflict_matrix,
                "context_hidden_dim": args.context_hidden_dim,
                "max_residual_logit": args.max_residual_logit,
                "freeze_base_actor": True,
            },
            ent_coef=args.entropy_coefficient,
            env_kwargs={"include_production_context": True},
            model_initializer=initialize,
        )
        training_seconds[str(train_seed)] = elapsed
        evaluated, audit = evaluate_with_conflict_audit(
            model,
            baseline,
            config,
            validation_seeds,
            condition=TREATMENT,
            enforce_conflicts=True,
            show_progress=False,
        )
        _tag(episode_rows, evaluated, TREATMENT, train_seed)
        audit_rows.append({"train_seed": train_seed, **audit})
        del model, baseline
        _write_csv(
            episode_rows,
            output_dir / "conflict_aware_residual_episodes.partial.csv",
        )
        _write_csv(
            audit_rows,
            output_dir / "conflict_aware_route_audit.partial.csv",
        )
        manifest["completed_training_cells"].append(train_seed)
        manifest["training_seconds"] = training_seconds
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    expected = len(CONDITIONS) * len(train_seeds) * len(validation_seeds)
    unique = {
        (row["condition"], int(row["train_seed"]), int(row["seed"]))
        for row in episode_rows
    }
    if len(episode_rows) != expected or len(unique) != expected:
        raise ValueError("Incomplete validation panel")

    audits: dict[str, Any] = {}
    for condition in (CURRENT, TREATMENT):
        rows = [row for row in audit_rows if row["condition"] == condition]
        combined = {
            "condition": condition,
            "enforce_conflicts": condition == TREATMENT,
            **{
                key: sum(int(row[key]) for row in rows)
                for key in (
                    "episodes",
                    "decisions",
                    "production_baseline_decisions",
                    "production_reroutes",
                    "same_machine_reroutes",
                    "same_operation_reroutes",
                    "nonconflicting_production_reroutes",
                    "decision_kind_mismatches",
                    "nonproduction_action_mismatches",
                    "invalid_actions",
                )
            },
        }
        audits[condition] = summarize_audit(combined)
        if audits[condition]["status"] != "PASS":
            raise AssertionError(f"Route audit failed for {condition}")

    summary = summarize_architectures(
        episode_rows, conditions=CONDITIONS, baseline_condition=SHARED
    )
    summary["primary_endpoint"] = (
        "mean validation objective by training seed: conflict-aware residual "
        "minus shared entropy"
    )
    summary["comparisons"][f"{TREATMENT}_minus_{SHARED}"] = _paired_summary(
        episode_rows, TREATMENT, SHARED
    )
    summary["comparisons"][f"{CURRENT}_minus_{SHARED}"] = _paired_summary(
        episode_rows, CURRENT, SHARED
    )
    summary["comparisons"][f"{TREATMENT}_minus_{CURRENT}"] = _paired_summary(
        episode_rows, TREATMENT, CURRENT
    )
    summary["route_audits"] = audits
    summary["promotion_decision"] = promotion_decision(
        summary, audits[TREATMENT]
    )
    summary["future_test_panel_opened"] = False

    _write_csv(
        episode_rows, output_dir / "conflict_aware_residual_episodes.csv"
    )
    _write_csv(audit_rows, output_dir / "conflict_aware_route_audit.csv")
    (output_dir / "conflict_aware_residual_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        status="COMPLETED",
        episode_count=len(episode_rows),
        route_audits=audits,
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
    parser.add_argument("--route-source-run", required=True)
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
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--context-hidden-dim", type=int, default=64)
    parser.add_argument("--max-residual-logit", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
