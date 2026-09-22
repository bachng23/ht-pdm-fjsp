"""Run the fresh independent-versus-broadcast MARL replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.marl_diagnostic import (
    DiagnosticSettings,
    evaluate_diagnostic_policy,
    train_diagnostic_policy,
)
from ht_pdm_fjsp.marl_diagnostic_experiment import (
    INDEPENDENT_MAPPO,
    _combine_audits,
    _git_revision,
    _read_completed_manifest,
    _validate_source_config,
)
from ht_pdm_fjsp.marl_factorial_experiment import COMBINED_MAPPO, _paired_contrast
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import _write_csv
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.shared_policy_experiment import METRICS, summarize_architectures


CONDITIONS = (INDEPENDENT_MAPPO, COMBINED_MAPPO)
FAILURE_NONINFERIORITY_MARGIN = 0.05
EXPECTED_FULL_TRAIN_SEEDS = 10

# One-sided 95% Student-t critical values indexed by degrees of freedom.
_T_CRITICAL_95 = {
    1: 6.313752,
    2: 2.919986,
    3: 2.353363,
    4: 2.131847,
    5: 2.015048,
    6: 1.943180,
    7: 1.894579,
    8: 1.859548,
    9: 1.833113,
    10: 1.812461,
    11: 1.795885,
    12: 1.782288,
    13: 1.770933,
    14: 1.761310,
    15: 1.753050,
    16: 1.745884,
    17: 1.739607,
    18: 1.734064,
    19: 1.729133,
    20: 1.724718,
    21: 1.720743,
    22: 1.717144,
    23: 1.713872,
    24: 1.710882,
    25: 1.708141,
    26: 1.705618,
    27: 1.703288,
    28: 1.701131,
    29: 1.699127,
    30: 1.697261,
}


def defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 256,
            "n_envs": 1,
            "n_steps": 32,
            "batch_size": 16,
            "n_epochs": 1,
            "train_seeds": (25_000, 26_000),
            "validation_seeds": tuple(range(55_900, 55_905)),
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "n_envs": 4,
            "n_steps": 1_024,
            "batch_size": 256,
            "n_epochs": 10,
            "train_seeds": tuple(range(15_000, 25_000, 1_000)),
            "validation_seeds": tuple(range(55_000, 55_200)),
        }
    raise ValueError(f"Unknown profile: {profile}")


def one_sided_upper_95(values: Iterable[float]) -> dict[str, float | int]:
    sample = tuple(float(value) for value in values)
    if len(sample) < 2:
        raise ValueError("At least two independent training seeds are required")
    mean = statistics.fmean(sample)
    standard_deviation = statistics.stdev(sample)
    degrees_of_freedom = len(sample) - 1
    critical = _T_CRITICAL_95.get(degrees_of_freedom, 1.644854)
    standard_error = standard_deviation / math.sqrt(len(sample))
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "degrees_of_freedom": degrees_of_freedom,
        "critical_value": critical,
        "upper_95": mean + critical * standard_error,
    }


def confirmation_decision(
    contrast: dict[str, Any], coordination_audits: dict[str, Any]
) -> dict[str, Any]:
    per_seed = contrast["per_training_seed"]
    objective = one_sided_upper_95(
        values["objective"] for values in per_seed.values()
    )
    failures = one_sided_upper_95(values["failures"] for values in per_seed.values())
    favorable_seed_count = sum(
        values["objective"] <= 0 for values in per_seed.values()
    )
    checks = {
        "exactly_ten_independent_training_seeds": len(per_seed)
        == EXPECTED_FULL_TRAIN_SEEDS,
        "objective_superiority_upper_95_below_zero": objective["upper_95"] < 0,
        "at_least_seven_of_ten_objective_deltas_nonpositive": (
            len(per_seed) == EXPECTED_FULL_TRAIN_SEEDS and favorable_seed_count >= 7
        ),
        "failure_noninferiority_upper_95_at_most_margin": failures["upper_95"]
        <= FAILURE_NONINFERIORITY_MARGIN,
        "all_coordination_audits_passed": all(
            coordination_audits[condition]["status"] == "PASS"
            for condition in CONDITIONS
        ),
    }
    return {
        "eligible_for_separately_authorized_held_out_test": all(checks.values()),
        "checks": checks,
        "objective_delta_inference": objective,
        "failure_delta_inference": failures,
        "failure_noninferiority_margin": FAILURE_NONINFERIORITY_MARGIN,
        "favorable_objective_seed_count": favorable_seed_count,
        "note": (
            "Engineering confirmation gate only; passing does not open the "
            "reserved held-out test automatically."
        ),
    }


def run(args: argparse.Namespace) -> Path:
    prior_dir = Path(args.prior_factorial_run).resolve()
    prior_manifest = _read_completed_manifest(
        prior_dir, "marl_factorial_manifest.json"
    )
    if prior_manifest.get("future_test_panel_opened"):
        raise ValueError("Prior factorial run reports an opened future test panel")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    profile = defaults(args.profile)
    train_seeds = tuple(
        parse_seeds(args.train_seeds) if args.train_seeds else profile["train_seeds"]
    )
    validation_seeds = tuple(
        parse_seeds(args.validation_seeds)
        if args.validation_seeds
        else profile["validation_seeds"]
    )
    if len(train_seeds) < 2 or len(set(train_seeds)) != len(train_seeds):
        raise ValueError("At least two unique training seeds are required")
    if set(train_seeds) & set(map(int, prior_manifest["train_seeds"])):
        raise ValueError("Training seeds overlap the prior factorial experiment")
    forbidden_validation = set(range(20_000, 55_000))
    forbidden_validation |= set(range(50_000, 50_100))
    forbidden_validation |= set(map(int, prior_manifest["validation_seeds"]))
    if (
        not validation_seeds
        or len(set(validation_seeds)) != len(validation_seeds)
        or set(validation_seeds) & forbidden_validation
    ):
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
    _validate_source_config(prior_dir, config_sha256, "factorial")
    config = BenchmarkConfig.from_json(config_path)
    (output_dir / "benchmark_config.json").write_bytes(config_bytes)
    device = resolve_device(args.device)
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
    condition_specs = {
        INDEPENDENT_MAPPO: {
            "actor_mode": "independent",
            "critic_mode": "global",
            "include_broadcast_context": False,
        },
        COMBINED_MAPPO: {
            "actor_mode": "independent",
            "critic_mode": "global",
            "include_broadcast_context": True,
        },
    }
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "profile": args.profile,
        "git_commit": _git_revision(),
        "config_sha256": config_sha256,
        "prior_factorial_run": str(prior_dir),
        "prior_factorial_git_commit": prior_manifest.get("git_commit"),
        "conditions": CONDITIONS,
        "condition_specs": condition_specs,
        "train_seeds": train_seeds,
        "validation_seeds": validation_seeds,
        "reserved_future_test_seeds": list(range(50_000, 50_100)),
        "future_test_panel_opened": False,
        "failure_noninferiority_margin": FAILURE_NONINFERIORITY_MARGIN,
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
    manifest_path = output_dir / "marl_replication_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    episode_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    training_seconds: dict[str, float] = {}

    for condition, spec in condition_specs.items():
        for train_seed in tqdm(
            train_seeds,
            desc=f"Train/evaluate {condition}",
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
                show_progress=not args.no_progress,
            )
            for row in evaluated:
                row.update(
                    split="marl_combined_replication_validation",
                    policy=condition,
                    train_seed=train_seed,
                )
            episode_rows.extend(evaluated)
            audit_rows.append({"condition": condition, "train_seed": train_seed, **audit})
            cell_key = f"{condition}:{train_seed}"
            training_seconds[cell_key] = elapsed
            manifest["completed_training_cells"].append(cell_key)
            manifest["training_seconds"] = training_seconds
            _write_csv(
                episode_rows, output_dir / "marl_replication_episodes.partial.csv"
            )
            _write_csv(
                audit_rows, output_dir / "marl_replication_coordination.partial.csv"
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
        raise ValueError("Incomplete MARL replication validation panel")
    if any(abs(float(row["episode_return"]) + float(row["objective"])) > 1e-8 for row in episode_rows):
        raise AssertionError("Reward/objective identity failed")
    coordination_audits = _combine_audits(audit_rows)
    if any(item["status"] != "PASS" for item in coordination_audits.values()):
        raise AssertionError("A MARL coordination audit failed")

    summary = summarize_architectures(
        episode_rows,
        conditions=CONDITIONS,
        baseline_condition=INDEPENDENT_MAPPO,
    )
    contrast = _paired_contrast(
        summary,
        treatment=COMBINED_MAPPO,
        control=INDEPENDENT_MAPPO,
        train_seeds=train_seeds,
    )
    summary.update(
        primary_endpoint=(
            "paired per-training-seed validation objective delta: independent "
            "actors with broadcast context minus independent actors"
        ),
        safety_endpoint=(
            "paired per-training-seed failure delta with +0.05 failures/episode "
            "non-inferiority margin"
        ),
        paired_contrast=contrast,
        coordination_audits=coordination_audits,
        confirmation_decision=confirmation_decision(contrast, coordination_audits),
        future_test_panel_opened=False,
    )
    _write_csv(episode_rows, output_dir / "marl_replication_episodes.csv")
    _write_csv(audit_rows, output_dir / "marl_replication_coordination.csv")
    (output_dir / "marl_replication_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest.update(
        status="COMPLETED",
        episode_count=len(episode_rows),
        coordination_audits=coordination_audits,
        confirmation_decision=summary["confirmation_decision"],
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
    parser.add_argument("--prior-factorial-run", required=True)
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
