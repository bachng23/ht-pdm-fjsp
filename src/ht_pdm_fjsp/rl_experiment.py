"""Train and evaluate a MaskablePPO baseline with reproducible artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm

from ht_pdm_fjsp.advanced_baselines import (
    CPSATReactivePolicy,
    HealthThresholdPolicy,
    JointRiskGreedyPolicy,
    MaskedSPTPolicy,
    RollingHorizonPolicy,
    rollout_policy,
)
from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig


@dataclass(frozen=True)
class ExperimentSettings:
    profile: str
    total_timesteps: int
    train_seed: int
    n_envs: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    validation_seeds: tuple[int, ...]
    test_seeds: tuple[int, ...]
    device: str
    include_slow_baselines: bool


def resolve_device(requested: str) -> str:
    """Resolve SB3 device while rejecting unsupported CUDA architectures."""

    if requested == "cpu":
        return "cpu"
    wants_cuda = requested == "auto" or requested.startswith("cuda")
    if not wants_cuda:
        return requested
    if not torch.cuda.is_available():
        if requested.startswith("cuda"):
            raise RuntimeError("CUDA was requested explicitly but is not available.")
        return "cpu"
    device_index = 0
    if ":" in requested:
        device_index = int(requested.split(":", maxsplit=1)[1])
    major, minor = torch.cuda.get_device_capability(device_index)
    required_arch = f"sm_{major}{minor}"
    supported_arches = set(torch.cuda.get_arch_list())
    if required_arch not in supported_arches:
        message = (
            f"GPU {torch.cuda.get_device_name(device_index)} requires {required_arch}, "
            f"but this PyTorch build supports {sorted(supported_arches)}."
        )
        if requested.startswith("cuda"):
            raise RuntimeError(message)
        print(f"WARNING: {message} Falling back to CPU.", file=sys.stderr)
        return "cpu"
    return "cuda" if requested == "auto" else requested


def profile_defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {
            "total_timesteps": 512,
            "n_envs": 1,
            "n_steps": 64,
            "batch_size": 32,
            "n_epochs": 2,
            "validation_seeds": tuple(range(20_000, 20_005)),
            "test_seeds": tuple(range(30_000, 30_005)),
            "include_slow_baselines": False,
        }
    if profile == "full":
        return {
            "total_timesteps": 500_000,
            "n_envs": 4,
            "n_steps": 1024,
            "batch_size": 256,
            "n_epochs": 10,
            "validation_seeds": tuple(range(20_000, 20_020)),
            "test_seeds": tuple(range(30_000, 30_100)),
            "include_slow_baselines": True,
        }
    raise ValueError(f"Unknown profile: {profile}")


def _make_training_env(
    config: BenchmarkConfig,
    *,
    seed: int,
    monitor_path: Path,
):
    def factory():
        env = HTPdmFjspEnv(config=config)
        env.reset(seed=seed)
        return Monitor(env, filename=str(monitor_path))

    return factory


def train_model(
    config: BenchmarkConfig,
    settings: ExperimentSettings,
    output_dir: Path,
    *,
    show_progress: bool,
    policy: Any = "MultiInputPolicy",
    policy_kwargs: dict[str, Any] | None = None,
    ent_coef: float = 0.0,
) -> tuple[MaskablePPO, float]:
    set_random_seed(settings.train_seed)
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    env = DummyVecEnv(
        [
            _make_training_env(
                config,
                seed=settings.train_seed + rank,
                monitor_path=monitor_dir / f"env_{rank}",
            )
            for rank in range(settings.n_envs)
        ]
    )
    model = MaskablePPO(
        policy,
        env,
        learning_rate=settings.learning_rate,
        n_steps=settings.n_steps,
        batch_size=settings.batch_size,
        n_epochs=settings.n_epochs,
        gamma=settings.gamma,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        seed=settings.train_seed,
        device=settings.device,
        verbose=0,
    )
    model.set_logger(configure(str(output_dir / "training_log"), ["csv"]))
    checkpoint_callback = CheckpointCallback(
        save_freq=max(
            1, math.ceil(settings.total_timesteps / (5 * settings.n_envs))
        ),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="maskable_ppo",
    )
    started = perf_counter()
    model.learn(
        total_timesteps=settings.total_timesteps,
        callback=checkpoint_callback,
        progress_bar=show_progress,
    )
    elapsed = perf_counter() - started
    model.save(output_dir / "maskable_ppo")
    env.close()
    return model, elapsed


def evaluate_ppo(
    model: MaskablePPO,
    config: BenchmarkConfig,
    seeds: Iterable[int],
    *,
    split: str,
    show_progress: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in tqdm(
        list(seeds), desc=f"PPO {split}", unit="episode", disable=not show_progress
    ):
        env = HTPdmFjspEnv(config=config)
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        while not env._done:
            action, _ = model.predict(
                observation,
                action_masks=env.action_masks(),
                deterministic=True,
            )
            observation, reward, _, _, _ = env.step(int(np.asarray(action).item()))
            episode_return += reward
        result = env.result("maskable_ppo")
        if not np.isclose(episode_return, -float(result.metrics["objective"])):
            raise AssertionError("PPO reward/objective identity failed.")
        rows.append(
            {
                "split": split,
                "policy": "maskable_ppo",
                "seed": seed,
                "episode_return": episode_return,
                **result.metrics,
            }
        )
        env.close()
    return rows


def evaluate_baselines(
    config: BenchmarkConfig,
    seeds: Iterable[int],
    *,
    include_slow: bool,
    show_progress: bool,
) -> list[dict[str, Any]]:
    policies: list[Any] = [
        MaskedSPTPolicy(),
        HealthThresholdPolicy(config.preventive_probability_threshold),
        JointRiskGreedyPolicy(),
        CPSATReactivePolicy(
            probability_threshold=config.preventive_probability_threshold
        ),
    ]
    if include_slow:
        policies.append(RollingHorizonPolicy())
    seed_list = list(seeds)
    rows: list[dict[str, Any]] = []
    progress = tqdm(
        total=len(policies) * len(seed_list),
        desc="Baselines test",
        unit="episode",
        disable=not show_progress,
    )
    for policy in policies:
        for seed in seed_list:
            result, episode_return = rollout_policy(
                HTPdmFjspEnv(config=config), policy, seed=seed
            )
            rows.append(
                {
                    "split": "test",
                    "policy": result.policy,
                    "seed": seed,
                    "episode_return": episode_return,
                    **result.metrics,
                }
            )
            progress.update(1)
    progress.close()
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_names = (
        "episode_return",
        "objective",
        "makespan",
        "total_tardiness",
        "failures",
        "preventive_maintenance",
        "corrective_maintenance",
        "total_cost",
    )
    groups = sorted({(str(row["split"]), str(row["policy"])) for row in rows})
    for split, policy in groups:
        selected = [
            row for row in rows if row["split"] == split and row["policy"] == policy
        ]
        summary[f"{split}/{policy}"] = {
            "episodes": len(selected),
            "metrics": {
                metric: {
                    "mean": fmean(float(row[metric]) for row in selected),
                    "std": pstdev(float(row[metric]) for row in selected),
                    "min": min(float(row[metric]) for row in selected),
                    "max": max(float(row[metric]) for row in selected),
                }
                for metric in metric_names
            },
        }
    return summary


def _write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output_dir}. Use --overwrite explicitly."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    defaults = profile_defaults(args.profile)
    resolved_device = resolve_device(args.device)
    settings = ExperimentSettings(
        profile=args.profile,
        total_timesteps=args.total_timesteps or defaults["total_timesteps"],
        train_seed=args.train_seed,
        n_envs=args.n_envs or defaults["n_envs"],
        n_steps=args.n_steps or defaults["n_steps"],
        batch_size=args.batch_size or defaults["batch_size"],
        n_epochs=args.n_epochs or defaults["n_epochs"],
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        validation_seeds=tuple(
            parse_seeds(args.validation_seeds)
            if args.validation_seeds
            else defaults["validation_seeds"]
        ),
        test_seeds=tuple(
            parse_seeds(args.test_seeds)
            if args.test_seeds
            else defaults["test_seeds"]
        ),
        device=resolved_device,
        include_slow_baselines=(
            defaults["include_slow_baselines"] and not args.skip_slow_baselines
        ),
    )
    if settings.n_steps * settings.n_envs % settings.batch_size:
        raise ValueError("n_steps * n_envs must be divisible by batch_size.")
    config = BenchmarkConfig.from_json(args.config)
    manifest = {
        "status": "RUNNING",
        "config_path": str(Path(args.config).resolve()),
        "requested_device": args.device,
        "settings": asdict(settings),
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "gymnasium": version("gymnasium"),
            "stable_baselines3": version("stable-baselines3"),
            "sb3_contrib": version("sb3-contrib"),
            "torch": version("torch"),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model, training_seconds = train_model(
        config, settings, output_dir, show_progress=not args.no_progress
    )
    rows = evaluate_ppo(
        model,
        config,
        settings.validation_seeds,
        split="validation",
        show_progress=not args.no_progress,
    )
    rows.extend(
        evaluate_ppo(
            model,
            config,
            settings.test_seeds,
            split="test",
            show_progress=not args.no_progress,
        )
    )
    rows.extend(
        evaluate_baselines(
            config,
            settings.test_seeds,
            include_slow=settings.include_slow_baselines,
            show_progress=not args.no_progress,
        )
    )
    _write_rows(rows, output_dir / "episodes.csv")
    results = summarize(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest["status"] = "COMPLETED"
    manifest["training_seconds"] = training_seconds
    manifest["output_files"] = sorted(path.name for path in output_dir.iterdir())
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed. Artifacts: {output_dir}")
    return output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/minimal_benchmark.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--train-seed", type=int, default=10_000)
    parser.add_argument("--validation-seeds")
    parser.add_argument("--test-seeds")
    parser.add_argument("--n-envs", type=int)
    parser.add_argument("--n-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--n-epochs", type=int)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--skip-slow-baselines", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    run_experiment(build_parser().parse_args())


if __name__ == "__main__":
    main()
