"""Train and evaluate route-preserving residual production ranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sb3_contrib import MaskablePPO
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
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
from ht_pdm_fjsp.route_preserving_residual_policy import (
    RoutePreservingResidualContextPolicy,
)
from ht_pdm_fjsp.shared_policy_experiment import summarize_architectures


CONDITIONS = (
    "shared_scorer_entropy",
    "production_context_entropy",
    "residual_context_entropy",
    "route_preserving_residual_entropy",
)


def defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 256,
            "n_envs": 1,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 2,
            "train_seeds": (10_000,),
            "validation_seeds": tuple(range(47_000, 47_005)),
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "n_envs": 4,
            "n_steps": 1_024,
            "batch_size": 256,
            "n_epochs": 10,
            "train_seeds": (10_000, 11_000, 12_000, 13_000, 14_000),
            "validation_seeds": tuple(range(47_000, 47_200)),
        }
    raise ValueError(f"Unknown profile: {profile}")


def _load_manifest(path: Path, name: str) -> dict[str, Any]:
    manifest_path = path / name
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETED":
        raise ValueError(f"Source run is incomplete: {path}")
    if manifest.get("future_test_panel_opened"):
        raise ValueError(f"Source run opened the reserved test panel: {path}")
    return manifest


def _git_revision() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _tag(
    rows: list[dict[str, Any]],
    evaluated: list[dict[str, Any]],
    condition: str,
    training_seed: int,
) -> None:
    for row in evaluated:
        row.update(
            policy=condition,
            condition=condition,
            train_seed=training_seed,
        )
    rows.extend(evaluated)


def _base_observation(
    observation: dict[str, np.ndarray], model: MaskablePPO
) -> dict[str, np.ndarray]:
    return {
        key: observation[key]
        for key in model.observation_space.spaces
    }


def evaluate_with_route_audit(
    model: MaskablePPO,
    baseline: MaskablePPO,
    config: BenchmarkConfig,
    seeds: Iterable[int],
    *,
    split: str,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate while comparing every deterministic route to the frozen base."""

    rows: list[dict[str, Any]] = []
    audit = {
        "episodes": 0,
        "decisions": 0,
        "production_baseline_decisions": 0,
        "production_reroutes": 0,
        "decision_kind_mismatches": 0,
        "nonproduction_action_mismatches": 0,
        "invalid_actions": 0,
    }
    for seed in tqdm(
        tuple(seeds),
        desc=f"Route audit {split}",
        unit="episode",
        disable=not show_progress,
    ):
        env = HTPdmFjspEnv(config=config, include_production_context=True)
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        while not env._done:
            action_mask = env.action_masks()
            baseline_action, _ = baseline.predict(
                _base_observation(observation, baseline),
                action_masks=action_mask,
                deterministic=True,
            )
            action, _ = model.predict(
                observation,
                action_masks=action_mask,
                deterministic=True,
            )
            baseline_index = int(np.asarray(baseline_action).item())
            action_index = int(np.asarray(action).item())
            baseline_kind = env.actions[baseline_index].kind
            action_kind = env.actions[action_index].kind
            audit["decisions"] += 1
            audit["invalid_actions"] += int(not action_mask[action_index])
            audit["decision_kind_mismatches"] += int(
                baseline_kind != action_kind
            )
            if baseline_kind == "production":
                audit["production_baseline_decisions"] += 1
                audit["production_reroutes"] += int(
                    baseline_index != action_index
                )
            else:
                audit["nonproduction_action_mismatches"] += int(
                    baseline_index != action_index
                )
            observation, reward, _, _, _ = env.step(action_index)
            episode_return += reward
        result = env.result("route_preserving_residual_entropy")
        if not np.isclose(episode_return, -float(result.metrics["objective"])):
            raise AssertionError("PPO reward/objective identity failed")
        rows.append(
            {
                "split": split,
                "policy": "route_preserving_residual_entropy",
                "seed": seed,
                "episode_return": episode_return,
                **result.metrics,
            }
        )
        audit["episodes"] += 1
        env.close()
    return rows, audit


def summarize_route_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        key: int(sum(int(row[key]) for row in rows))
        for key in (
            "episodes",
            "decisions",
            "production_baseline_decisions",
            "production_reroutes",
            "decision_kind_mismatches",
            "nonproduction_action_mismatches",
            "invalid_actions",
        )
    }
    production_count = totals["production_baseline_decisions"]
    totals["production_reroute_rate"] = (
        totals["production_reroutes"] / production_count
        if production_count
        else 0.0
    )
    checks = {
        "zero_decision_kind_mismatches": (
            totals["decision_kind_mismatches"] == 0
        ),
        "zero_nonproduction_action_mismatches": (
            totals["nonproduction_action_mismatches"] == 0
        ),
        "zero_invalid_actions": totals["invalid_actions"] == 0,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "totals": totals,
        "replication_unit": "independent PPO training seed",
    }


def run(args: argparse.Namespace) -> Path:
    shared_dir = Path(args.shared_source_run).resolve()
    production_dir = Path(args.production_source_run).resolve()
    residual_dir = Path(args.residual_source_run).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shared_manifest = _load_manifest(
        shared_dir, "architecture_manifest.json"
    )
    production_manifest = _load_manifest(
        production_dir, "production_context_manifest.json"
    )
    residual_manifest = _load_manifest(
        residual_dir, "residual_context_manifest.json"
    )
    if Path(production_manifest["shared_source_run"]).name != shared_dir.name:
        raise ValueError("Production source provenance mismatch")
    if Path(residual_manifest["shared_source_run"]).name != shared_dir.name:
        raise ValueError("Residual source provenance mismatch")
    if Path(residual_manifest["production_source_run"]).name != production_dir.name:
        raise ValueError("Residual/production source provenance mismatch")

    profile = defaults(args.profile)
    training_seeds = tuple(
        parse_seeds(args.train_seeds)
        if args.train_seeds
        else profile["train_seeds"]
    )
    available = (
        set(map(int, shared_manifest["train_seeds"]))
        & set(map(int, production_manifest["train_seeds"]))
        & set(map(int, residual_manifest["train_seeds"]))
    )
    if not training_seeds or not set(training_seeds) <= available:
        raise ValueError("A requested training seed is missing from a source run")
    validation_seeds = tuple(
        parse_seeds(args.validation_seeds)
        if args.validation_seeds
        else profile["validation_seeds"]
    )
    forbidden = set(range(40_000, 47_000)) | set(range(50_000, 50_100))
    for manifest in (
        shared_manifest,
        production_manifest,
        residual_manifest,
    ):
        forbidden.update(map(int, manifest["validation_seeds"]))
    if not validation_seeds or set(validation_seeds) & forbidden:
        raise ValueError("Validation panel overlaps prior or reserved data")

    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config_sha256 = hashlib.sha256(config_bytes).hexdigest()
    if production_manifest.get("config_sha256") != config_sha256:
        raise ValueError("Benchmark config does not match source runs")
    if residual_manifest.get("config_sha256") != config_sha256:
        raise ValueError("Benchmark config does not match residual source")
    config = BenchmarkConfig.from_json(config_path)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)

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
        "shared_source_run": str(shared_dir),
        "production_source_run": str(production_dir),
        "residual_source_run": str(residual_dir),
        "conditions": CONDITIONS,
        "train_seeds": training_seeds,
        "validation_seeds": validation_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "common_settings": common,
        "entropy_coefficient": args.entropy_coefficient,
        "context_hidden_dim": args.context_hidden_dim,
        "max_residual_logit": args.max_residual_logit,
        "freeze_base_actor": True,
        "route_contract": (
            "sample shared action first; preserve every non-production action; "
            "rerank only feasible production alternatives"
        ),
        "completed_training_cells": [],
        "requested_device": args.device,
        "resolved_device": device,
        "runtime": {
            name: version(name)
            for name in (
                "numpy",
                "stable-baselines3",
                "sb3-contrib",
                "torch",
            )
        },
    }
    manifest_path = output_dir / "route_preserving_residual_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    episode_rows: list[dict[str, Any]] = []
    route_audit_rows: list[dict[str, Any]] = []
    source_specs = (
        ("shared_scorer_entropy", shared_dir, {}),
        (
            "production_context_entropy",
            production_dir,
            {"include_production_context": True},
        ),
        (
            "residual_context_entropy",
            residual_dir,
            {"include_production_context": True},
        ),
    )
    source_jobs = [
        (condition, path, kwargs, seed)
        for condition, path, kwargs in source_specs
        for seed in training_seeds
    ]
    for condition, path, env_kwargs, seed in tqdm(
        source_jobs,
        desc="Source validation",
        unit="model",
        disable=args.no_progress,
    ):
        model = MaskablePPO.load(
            path / condition / f"train_seed_{seed}" / "maskable_ppo.zip",
            device=device,
        )
        evaluated = evaluate_ppo(
            model,
            config,
            validation_seeds,
            split="route_preserving_validation",
            show_progress=False,
            env_kwargs=env_kwargs,
        )
        _tag(episode_rows, evaluated, condition, seed)
        del model
        _write_csv(
            episode_rows,
            output_dir / "route_preserving_residual_episodes.partial.csv",
        )

    training_seconds: dict[str, float] = {}
    for seed in tqdm(
        training_seeds,
        desc="Route-preserving residual PPO",
        unit="model",
        disable=args.no_progress,
    ):
        baseline = MaskablePPO.load(
            shared_dir
            / "shared_scorer_entropy"
            / f"train_seed_{seed}"
            / "maskable_ppo.zip",
            device=device,
        )

        def initialize(model: MaskablePPO, source=baseline) -> None:
            model.policy.load_shared_base(source.policy)

        model, elapsed = train_model(
            config,
            ExperimentSettings(train_seed=seed, **common),
            output_dir
            / "route_preserving_residual_entropy"
            / f"train_seed_{seed}",
            show_progress=not args.no_progress,
            policy=RoutePreservingResidualContextPolicy,
            policy_kwargs={
                "context_hidden_dim": args.context_hidden_dim,
                "max_residual_logit": args.max_residual_logit,
                "freeze_base_actor": True,
            },
            ent_coef=args.entropy_coefficient,
            env_kwargs={"include_production_context": True},
            model_initializer=initialize,
        )
        training_seconds[str(seed)] = elapsed
        evaluated, audit = evaluate_with_route_audit(
            model,
            baseline,
            config,
            validation_seeds,
            split="route_preserving_validation",
            show_progress=False,
        )
        _tag(
            episode_rows,
            evaluated,
            "route_preserving_residual_entropy",
            seed,
        )
        route_audit_rows.append({"train_seed": seed, **audit})
        del model, baseline
        _write_csv(
            episode_rows,
            output_dir / "route_preserving_residual_episodes.partial.csv",
        )
        _write_csv(
            route_audit_rows,
            output_dir / "route_preserving_route_audit.partial.csv",
        )
        manifest["completed_training_cells"].append(seed)
        manifest["training_seconds"] = training_seconds
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    expected = len(CONDITIONS) * len(training_seeds) * len(validation_seeds)
    unique = {
        (row["condition"], row["train_seed"], row["seed"])
        for row in episode_rows
    }
    if len(episode_rows) != expected or len(unique) != expected:
        raise ValueError("Incomplete validation panel")

    route_audit = summarize_route_audits(route_audit_rows)
    if route_audit["status"] != "PASS":
        raise AssertionError("Route-preservation audit failed")
    summary = summarize_architectures(
        episode_rows,
        conditions=CONDITIONS,
        baseline_condition="shared_scorer_entropy",
    )
    summary["primary_endpoint"] = (
        "mean validation objective by training seed: "
        "route-preserving residual minus shared entropy"
    )
    summary["comparisons"][
        "route_preserving_residual_entropy_minus_residual_context_entropy"
    ] = _paired_summary(
        episode_rows,
        "route_preserving_residual_entropy",
        "residual_context_entropy",
    )
    summary["route_audit"] = route_audit
    summary["future_test_panel_opened"] = False

    _write_csv(
        episode_rows, output_dir / "route_preserving_residual_episodes.csv"
    )
    _write_csv(
        route_audit_rows, output_dir / "route_preserving_route_audit.csv"
    )
    (output_dir / "route_preserving_residual_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest.update(
        status="COMPLETED",
        episode_count=len(episode_rows),
        route_audit=route_audit,
        training_seconds=training_seconds,
        output_files=sorted(path.name for path in output_dir.iterdir()),
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Completed. Artifacts: {output_dir}", flush=True)
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-source-run", required=True)
    parser.add_argument("--production-source-run", required=True)
    parser.add_argument("--residual-source-run", required=True)
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
