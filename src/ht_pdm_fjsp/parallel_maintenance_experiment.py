"""Run the validated parallel-maintenance heuristic and RL baseline screen."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

import numpy as np
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
from tqdm.auto import tqdm

from ht_pdm_fjsp.benchmark import parse_seeds
from ht_pdm_fjsp.parallel_maintenance import (
    FAILED,
    MAINTENANCE,
    WORKING,
    CentralizedParallelMaintenanceEnv,
    ParallelMaintenanceConfig,
    ParallelMaintenanceEnv,
)
from ht_pdm_fjsp.rl_experiment import resolve_device
from ht_pdm_fjsp.value_decomposition import (
    ValueDecompositionPolicy,
    ValueLearningSettings,
    train_value_policy,
)


HEURISTICS = (
    "random_feasible",
    "fifo",
    "fibt",
    "risk_first",
    "balanced_greedy",
)
LEARNED = ("masked_ppo", "iql", "qmix")
SEALED_TEST_SEEDS = tuple(range(63_200, 63_300))


@dataclass(frozen=True)
class Profile:
    total_timesteps: int
    training_seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    n_envs: int
    ppo_n_steps: int
    ppo_batch_size: int
    ppo_epochs: int
    replay_capacity: int
    learning_starts: int
    value_batch_size: int


def profile_settings(profile: str) -> Profile:
    if profile == "smoke":
        return Profile(
            total_timesteps=800,
            training_seeds=(77_100,),
            development_seeds=tuple(range(63_190, 63_193)),
            n_envs=1,
            ppo_n_steps=100,
            ppo_batch_size=50,
            ppo_epochs=2,
            replay_capacity=800,
            learning_starts=64,
            value_batch_size=32,
        )
    if profile == "full":
        return Profile(
            total_timesteps=100_000,
            training_seeds=tuple(range(77_000, 77_005)),
            development_seeds=tuple(range(63_100, 63_150)),
            n_envs=4,
            ppo_n_steps=125,
            ppo_batch_size=100,
            ppo_epochs=5,
            replay_capacity=20_000,
            learning_starts=2_000,
            value_batch_size=256,
        )
    raise ValueError(f"Unknown profile: {profile}")


def _git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _available_technicians(env: ParallelMaintenanceEnv) -> list[int]:
    return [
        technician
        for technician in range(env.technician_count)
        if env.technician_available(technician)
    ]


def heuristic_actions(
    name: str,
    env: ParallelMaintenanceEnv,
    rng: np.random.Generator,
) -> np.ndarray:
    masks = env._action_masks()
    if name == "random_feasible":
        return np.asarray(
            [int(rng.choice(np.flatnonzero(row))) for row in masks], dtype=np.int64
        )
    actions = np.zeros(env.num_agents, dtype=np.int64)
    available = _available_technicians(env)
    candidates = [
        machine
        for machine in range(env.num_agents)
        if env.mode[machine] != MAINTENANCE
        and (
            env.mode[machine] == FAILED
            or env.conditional_failure_probability(machine)
            >= env.config.maintenance.pm_risk_threshold
        )
    ]
    if name in {"fifo", "fibt"}:
        candidates.sort(
            key=lambda machine: (
                0 if env.mode[machine] == FAILED else 1,
                -env.wait[machine],
                machine,
            )
        )
    elif name in {"risk_first", "balanced_greedy"}:
        candidates.sort(
            key=lambda machine: (
                0 if env.mode[machine] == FAILED else 1,
                -env.wait[machine],
                -env.conditional_failure_probability(machine),
                machine,
            )
        )
    else:
        raise ValueError(f"Unknown heuristic: {name}")
    for machine in candidates:
        if not available:
            break
        if name == "fifo":
            technician = min(available)
        elif name == "balanced_greedy":
            technician = min(
                available,
                key=lambda item: (
                    env.technician_busy_time[item],
                    env.expected_duration(machine, item),
                    item,
                ),
            )
        else:
            technician = min(
                available,
                key=lambda item: (env.expected_duration(machine, item), item),
            )
        actions[machine] = technician + 1
        available.remove(technician)
    return actions


def _episode_row(
    info: dict[str, Any],
    *,
    condition: str,
    seed: int,
    train_seed: int | None,
    inference_seconds: float,
) -> dict[str, Any]:
    utilization = info["technician_utilization"]
    return {
        "condition": condition,
        "train_seed": "" if train_seed is None else train_seed,
        "seed": seed,
        "objective": info["objective"],
        "episode_return": info["episode_return"],
        "production": info["production"],
        "downtime": info["downtime"],
        "failures": info["failures"],
        "preventive": info["preventive"],
        "corrective": info["corrective"],
        "waiting": info["waiting"],
        "proposal_conflicts": info["proposal_conflicts"],
        "conflict_steps": info["conflict_steps"],
        "rework": info["rework"],
        "early_pm": info["early_pm"],
        "workload_imbalance": info["workload_imbalance"],
        "technician_0_utilization": utilization[0],
        "technician_1_utilization": utilization[1],
        "return_objective_error": info["return_objective_error"],
        "invalid_executions": info["invalid_executions"],
        "duplicate_machine_assignments": info["duplicate_machine_assignments"],
        "duplicate_technician_assignments": info[
            "duplicate_technician_assignments"
        ],
        "inference_seconds": inference_seconds,
    }


def evaluate_local_policy(
    config: ParallelMaintenanceConfig,
    seeds: Iterable[int],
    *,
    condition: str,
    train_seed: int | None,
    action_function: Callable[[ParallelMaintenanceEnv, np.random.Generator], np.ndarray],
    show_progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    seed_list = list(seeds)
    progress = tqdm(
        seed_list,
        desc=f"{condition} evaluation",
        unit="episode",
        disable=not show_progress,
    )
    for seed in progress:
        env = ParallelMaintenanceEnv(config)
        _, _ = env.reset(seed=seed)
        policy_rng = np.random.default_rng(
            np.random.SeedSequence([seed, 901, 0 if train_seed is None else train_seed])
        )
        inference_seconds = 0.0
        done = False
        while not done:
            started = perf_counter()
            actions = action_function(env, policy_rng)
            inference_seconds += perf_counter() - started
            _, _, done, _, info = env.step(actions)
            decision_rows.append(
                {
                    "condition": condition,
                    "train_seed": "" if train_seed is None else train_seed,
                    "seed": seed,
                    "step": env.last_decision["step"],
                    "actions": json.dumps(env.last_decision["actions"]),
                    "accepted": json.dumps(env.last_decision["accepted"]),
                    "proposal_conflicts": env.last_decision["proposal_conflicts"],
                    "failures": env.last_decision["failures"],
                    "reworks": env.last_decision["reworks"],
                    "incremental_cost": env.last_decision["incremental_cost"],
                    "reward": env.last_decision["reward"],
                }
            )
        episode_rows.append(
            _episode_row(
                info,
                condition=condition,
                seed=seed,
                train_seed=train_seed,
                inference_seconds=inference_seconds,
            )
        )
        env.close()
    return episode_rows, decision_rows


def train_ppo(
    config: ParallelMaintenanceConfig,
    profile: Profile,
    output_dir: Path,
    *,
    train_seed: int,
    device: str,
    show_progress: bool,
) -> tuple[MaskablePPO, float]:
    set_random_seed(train_seed)
    monitor_dir = output_dir / "monitor"
    monitor_dir.mkdir(parents=True, exist_ok=True)

    def factory(rank: int):
        def make() -> Monitor:
            env = CentralizedParallelMaintenanceEnv(config)
            env.reset(seed=train_seed + rank)
            return Monitor(env, filename=str(monitor_dir / f"env_{rank}"))

        return make

    vector_env = DummyVecEnv([factory(rank) for rank in range(profile.n_envs)])
    model = MaskablePPO(
        "MlpPolicy",
        vector_env,
        learning_rate=3e-4,
        n_steps=profile.ppo_n_steps,
        batch_size=profile.ppo_batch_size,
        n_epochs=profile.ppo_epochs,
        gamma=0.99,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [128, 128]},
        seed=train_seed,
        device=device,
        verbose=0,
    )
    model.set_logger(configure(str(output_dir / "training_log"), ["csv"]))
    callback = CheckpointCallback(
        save_freq=max(1, profile.total_timesteps // (5 * profile.n_envs)),
        save_path=str(output_dir / "checkpoints"),
        name_prefix="masked_ppo",
    )
    started = perf_counter()
    model.learn(
        total_timesteps=profile.total_timesteps,
        callback=callback,
        progress_bar=show_progress,
    )
    elapsed = perf_counter() - started
    model.save(output_dir / "model")
    vector_env.close()
    loaded = MaskablePPO.load(output_dir / "model", device=device)
    return loaded, elapsed


def evaluate_ppo(
    model: MaskablePPO,
    config: ParallelMaintenanceConfig,
    seeds: Iterable[int],
    *,
    train_seed: int,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for seed in tqdm(
        list(seeds),
        desc="masked_ppo evaluation",
        unit="episode",
        disable=not show_progress,
    ):
        env = CentralizedParallelMaintenanceEnv(config)
        observation, _ = env.reset(seed=seed)
        done = False
        inference_seconds = 0.0
        while not done:
            started = perf_counter()
            action, _ = model.predict(
                observation,
                action_masks=env.action_masks(),
                deterministic=True,
            )
            inference_seconds += perf_counter() - started
            observation, _, done, _, info = env.step(int(np.asarray(action).item()))
            decision_rows.append(
                {
                    "condition": "masked_ppo",
                    "train_seed": train_seed,
                    "seed": seed,
                    "step": env.core.last_decision["step"],
                    "actions": json.dumps(env.core.last_decision["actions"]),
                    "accepted": json.dumps(env.core.last_decision["accepted"]),
                    "proposal_conflicts": env.core.last_decision[
                        "proposal_conflicts"
                    ],
                    "failures": env.core.last_decision["failures"],
                    "reworks": env.core.last_decision["reworks"],
                    "incremental_cost": env.core.last_decision["incremental_cost"],
                    "reward": env.core.last_decision["reward"],
                }
            )
        episode_rows.append(
            _episode_row(
                info,
                condition="masked_ppo",
                seed=seed,
                train_seed=train_seed,
                inference_seconds=inference_seconds,
            )
        )
        env.close()
    return episode_rows, decision_rows


def summarize(
    episode_rows: list[dict[str, Any]],
    *,
    expected_episodes: int,
    training_seed_count: int,
    profile: str,
) -> dict[str, Any]:
    metrics = (
        "objective",
        "production",
        "downtime",
        "failures",
        "preventive",
        "corrective",
        "waiting",
        "proposal_conflicts",
        "conflict_steps",
        "rework",
        "early_pm",
        "workload_imbalance",
        "inference_seconds",
    )
    conditions = (*HEURISTICS, *LEARNED)
    condition_summary: dict[str, Any] = {}
    for condition in conditions:
        selected = [row for row in episode_rows if row["condition"] == condition]
        condition_summary[condition] = {
            "episodes": len(selected),
            "training_replicates": len(
                {row["train_seed"] for row in selected if row["train_seed"] != ""}
            ),
            "metrics": {
                metric: {
                    "mean": statistics.fmean(float(row[metric]) for row in selected),
                    "std": statistics.pstdev(float(row[metric]) for row in selected),
                    "min": min(float(row[metric]) for row in selected),
                    "max": max(float(row[metric]) for row in selected),
                }
                for metric in metrics
            },
        }
    best_heuristic = min(
        HEURISTICS,
        key=lambda name: condition_summary[name]["metrics"]["objective"]["mean"],
    )
    best_learned = min(
        LEARNED,
        key=lambda name: condition_summary[name]["metrics"]["objective"]["mean"],
    )
    heuristic_objective = condition_summary[best_heuristic]["metrics"]["objective"][
        "mean"
    ]
    learned_objective = condition_summary[best_learned]["metrics"]["objective"][
        "mean"
    ]
    h1_improvement = (heuristic_objective - learned_objective) / max(
        abs(heuristic_objective), 1e-9
    )
    iql_conflicts = condition_summary["iql"]["metrics"]["conflict_steps"]["mean"]
    qmix_conflicts = condition_summary["qmix"]["metrics"]["conflict_steps"]["mean"]
    winners = {
        metric: min(
            HEURISTICS,
            key=lambda name: (
                -condition_summary[name]["metrics"][metric]["mean"]
                if metric == "production"
                else condition_summary[name]["metrics"][metric]["mean"]
            ),
        )
        for metric in ("objective", "downtime", "rework", "workload_imbalance")
    }
    audits = {
        "expected_episodes_completed": len(episode_rows) == expected_episodes,
        "training_replicates_complete": all(
            condition_summary[name]["training_replicates"] == training_seed_count
            for name in LEARNED
        ),
        "reward_objective_identity": all(
            float(row["return_objective_error"]) <= 1e-6 for row in episode_rows
        ),
        "no_invalid_executions": all(
            int(row["invalid_executions"]) == 0 for row in episode_rows
        ),
        "no_duplicate_machine_assignments": all(
            int(row["duplicate_machine_assignments"]) == 0 for row in episode_rows
        ),
        "no_duplicate_technician_assignments": all(
            int(row["duplicate_technician_assignments"]) == 0
            for row in episode_rows
        ),
        "sealed_test_panel_remains_closed": not any(
            int(row["seed"]) in SEALED_TEST_SEEDS for row in episode_rows
        ),
    }
    return {
        "purpose": "Development baseline screening, not a final performance claim.",
        "profile": profile,
        "conditions": condition_summary,
        "best_heuristic": best_heuristic,
        "best_learned": best_learned,
        "hypotheses": {
            "H1_learned_improves_best_heuristic_by_5_percent": {
                "relative_improvement": h1_improvement,
                "passed": h1_improvement >= 0.05,
            },
            "H2_qmix_has_lower_conflict_incidence_than_iql": {
                "iql_mean_conflict_steps": iql_conflicts,
                "qmix_mean_conflict_steps": qmix_conflicts,
                "passed": qmix_conflicts < iql_conflicts,
            },
            "H3_no_fixed_heuristic_uniformly_best": {
                "metric_winners": winners,
                "passed": len(set(winners.values())) > 1,
            },
        },
        "audits": audits,
        "gate": {"passed": all(audits.values())},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = ParallelMaintenanceConfig.from_json(args.config)
    defaults = profile_settings(args.profile)
    training_seeds = (
        tuple(parse_seeds(args.training_seeds))
        if args.training_seeds
        else defaults.training_seeds
    )
    development_seeds = (
        tuple(parse_seeds(args.development_seeds))
        if args.development_seeds
        else defaults.development_seeds
    )
    total_timesteps = args.total_timesteps or defaults.total_timesteps
    profile = Profile(
        **{
            **asdict(defaults),
            "training_seeds": training_seeds,
            "development_seeds": development_seeds,
            "total_timesteps": total_timesteps,
        }
    )
    if total_timesteps % profile.n_envs:
        raise ValueError("Total timesteps must be divisible by n_envs")
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    started_at = datetime.now(UTC)
    manifest_path = output_dir / "parallel_maintenance_manifest.json"
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "purpose": "development_baseline_screen",
        "profile": args.profile,
        "git_revision": _git_revision(),
        "started_at": started_at.isoformat(),
        "config": str(Path(args.config).resolve()),
        "training_seeds": list(training_seeds),
        "development_seeds": list(development_seeds),
        "sealed_test_seeds": list(SEALED_TEST_SEEDS),
        "sealed_test_evaluated": False,
        "settings": asdict(profile),
        "device": device,
        "conditions": [*HEURISTICS, *LEARNED],
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": version("numpy"),
            "torch": version("torch"),
            "gymnasium": version("gymnasium"),
            "stable_baselines3": version("stable-baselines3"),
            "sb3_contrib": version("sb3-contrib"),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output_dir / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    episode_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    training_rows: list[dict[str, Any]] = []
    try:
        for name in HEURISTICS:
            rows, decisions = evaluate_local_policy(
                config,
                development_seeds,
                condition=name,
                train_seed=None,
                action_function=lambda env, rng, heuristic=name: heuristic_actions(
                    heuristic, env, rng
                ),
                show_progress=True,
            )
            episode_rows.extend(rows)
            decision_rows.extend(decisions)
            _write_csv(episode_rows, output_dir / "episodes.partial.csv")

        for train_seed in training_seeds:
            ppo_dir = output_dir / "masked_ppo" / str(train_seed)
            ppo, elapsed = train_ppo(
                config,
                profile,
                ppo_dir,
                train_seed=train_seed,
                device=device,
                show_progress=True,
            )
            training_rows.append(
                {
                    "condition": "masked_ppo",
                    "train_seed": train_seed,
                    "timesteps": total_timesteps,
                    "wall_seconds": elapsed,
                }
            )
            rows, decisions = evaluate_ppo(
                ppo,
                config,
                development_seeds,
                train_seed=train_seed,
                show_progress=True,
            )
            episode_rows.extend(rows)
            decision_rows.extend(decisions)
            _write_csv(episode_rows, output_dir / "episodes.partial.csv")

            for algorithm in ("iql", "qmix"):
                cell = output_dir / algorithm / str(train_seed)
                value_settings = ValueLearningSettings(
                    total_timesteps=total_timesteps,
                    replay_capacity=profile.replay_capacity,
                    learning_starts=profile.learning_starts,
                    batch_size=profile.value_batch_size,
                    train_frequency=4,
                    gradient_steps=1,
                    learning_rate=3e-4,
                    gamma=0.99,
                    target_update_interval=max(100, total_timesteps // 20),
                    epsilon_start=1.0,
                    epsilon_end=0.05,
                    epsilon_fraction=0.8,
                    hidden_dim=128,
                    mixer_hidden_dim=64,
                    device=device,
                    n_envs=profile.n_envs,
                )
                checkpoint_targets = tuple(
                    total_timesteps * part // 5 for part in range(1, 6)
                )
                model, elapsed = train_value_policy(
                    config,
                    value_settings,
                    cell,
                    algorithm=algorithm,
                    train_seed=train_seed,
                    show_progress=True,
                    checkpoint_targets=checkpoint_targets,
                    env_factory=lambda: ParallelMaintenanceEnv(config),
                )
                loaded = ValueDecompositionPolicy.load(cell / "model.pt", device=device)
                loaded.eval()
                training_rows.append(
                    {
                        "condition": algorithm,
                        "train_seed": train_seed,
                        "timesteps": total_timesteps,
                        "wall_seconds": elapsed,
                    }
                )
                rows, decisions = evaluate_local_policy(
                    config,
                    development_seeds,
                    condition=algorithm,
                    train_seed=train_seed,
                    action_function=lambda env, rng, policy=loaded: policy.act(
                        env._observation(), deterministic=True, device=device
                    ),
                    show_progress=True,
                )
                episode_rows.extend(rows)
                decision_rows.extend(decisions)
                _write_csv(episode_rows, output_dir / "episodes.partial.csv")

        expected_episodes = len(development_seeds) * (
            len(HEURISTICS) + len(LEARNED) * len(training_seeds)
        )
        summary = summarize(
            episode_rows,
            expected_episodes=expected_episodes,
            training_seed_count=len(training_seeds),
            profile=args.profile,
        )
        _write_csv(episode_rows, output_dir / "episodes.csv")
        _write_csv(decision_rows, output_dir / "decisions.csv")
        _write_csv(
            [
                {
                    key: row[key]
                    for key in (
                        "condition",
                        "train_seed",
                        "seed",
                        "invalid_executions",
                        "duplicate_machine_assignments",
                        "duplicate_technician_assignments",
                        "return_objective_error",
                    )
                }
                for row in episode_rows
            ],
            output_dir / "coordination.csv",
        )
        _write_csv(training_rows, output_dir / "training_progress.csv")
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        manifest.update(
            {
                "status": "COMPLETED" if summary["gate"]["passed"] else "FAILED",
                "finished_at": datetime.now(UTC).isoformat(),
                "episode_count": len(episode_rows),
                "decision_count": len(decision_rows),
                "training_runs": len(training_rows),
                "audit_gate": summary["gate"],
                "outputs": sorted(
                    str(path.relative_to(output_dir))
                    for path in output_dir.rglob("*")
                    if path.is_file()
                ),
            }
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        if not summary["gate"]["passed"]:
            raise RuntimeError("Experiment audit gate failed")
        return summary
    except Exception:
        manifest.update(
            {"status": "FAILED", "finished_at": datetime.now(UTC).isoformat()}
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/parallel_maintenance.json")
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--training-seeds")
    parser.add_argument("--development-seeds")
    parser.add_argument("--total-timesteps", type=int)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
