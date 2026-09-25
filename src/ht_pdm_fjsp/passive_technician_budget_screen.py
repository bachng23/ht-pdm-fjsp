"""Screen whether passive-technician learners are undertrained."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import random
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import (
    ALGORITHMS,
    ActorCritic,
    CentralizedPPO,
    PPOSettings,
    _obs_tensor,
    _masked_logits,
    _returns,
    evaluate_centralized_ppo,
    evaluate_fixed,
    evaluate_independent_ppo,
    load_centralized_ppo,
    load_independent_ppo,
    stress_config,
)
from ht_pdm_fjsp.passive_technician_marl import (
    IndependentQ,
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    PassiveConfig,
    PassiveTechnicianEnv,
)


BUDGETS = (5_000, 10_000, 20_000)
SMOKE_BUDGETS = (8, 16, 32)
FIXED_POLICIES = ("random_feasible", "skill_aware_fifo")
FIELDNAMES = (
    "policy",
    "budget",
    "seed",
    "train_seed",
    "objective",
    "failures",
    "jobs",
    "collisions",
    "waiting",
    "invalid_requests",
    "busy_requests",
)


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: tuple[str, ...] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    names = list(fieldnames or rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _profile(profile: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if profile == "smoke":
        return (11,), (101, 102, 103), SMOKE_BUDGETS
    if profile == "pilot":
        return (11, 12, 13), tuple(range(101, 111)), (100, 200, 400)
    if profile == "full":
        return (11, 12, 13), tuple(range(101, 201)), BUDGETS
    if profile == "replication":
        return tuple(range(14, 24)), tuple(range(101, 201)), BUDGETS
    raise ValueError(f"Unknown profile: {profile}")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is false")
    return torch.device(requested)


def _save_q_checkpoint(learner: IndependentQ, root: Path, budget: int) -> Path:
    path = root / f"budget_{budget}" / "model.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    learner.save(path)
    return path


def _train_q_checkpoints(
    config: PassiveConfig,
    seed: int,
    budgets: tuple[int, ...],
    root: Path,
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    learner = IndependentQ(config, seed)
    progress: list[dict[str, Any]] = []
    checkpoints: dict[int, Path] = {}
    targets = set(budgets)
    for episode in tqdm(
        range(1, max(budgets) + 1), desc=f"Independent Q {seed}", unit="episode", leave=False
    ):
        env = PassiveTechnicianEnv(config, seed=seed + episode - 1)
        observations = env.reset()
        rewards: list[float] = []
        done = False
        epsilon = max(0.05, 1.0 - (episode - 1) / max(1, max(budgets) * 0.8))
        while not done:
            masks = env.action_masks()
            actions = tuple(
                learner.action(machine, observations[machine], epsilon, masks[machine])
                for machine in range(config.machines)
            )
            next_observations, reward, done, _ = env.step(actions)
            next_masks = env.action_masks()
            for machine in range(config.machines):
                learner.update(
                    machine,
                    observations[machine],
                    actions[machine],
                    -reward,
                    next_observations[machine],
                    next_masks[machine],
                )
            observations = next_observations
            rewards.append(float(reward))
        progress.append(
            {"policy": "independent_q", "train_seed": seed, "episode": episode, "objective": -sum(rewards)}
        )
        if episode in targets:
            checkpoints[episode] = _save_q_checkpoint(learner, root, episode)
    return checkpoints, progress


def _save_independent_ppo(policies: list[ActorCritic], seed: int, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policies": [policy.state_dict() for policy in policies], "seed": seed}, path)
    return path


def _train_independent_ppo_checkpoints(
    config: PassiveConfig,
    seed: int,
    budgets: tuple[int, ...],
    root: Path,
    settings: PPOSettings,
    device: torch.device,
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    policies = [ActorCritic(config.observation_dim, config.technicians + 1).to(device) for _ in range(config.machines)]
    optimizers = [torch.optim.Adam(policy.parameters(), lr=settings.learning_rate) for policy in policies]
    checkpoints: dict[int, Path] = {}
    progress: list[dict[str, Any]] = []
    targets = set(budgets)
    for episode in tqdm(
        range(1, max(budgets) + 1), desc=f"Independent PPO {seed}", unit="episode", leave=False
    ):
        env = PassiveTechnicianEnv(config, seed=seed * 100_000 + episode - 1)
        observations = env.reset()
        rollout_obs: list[list[torch.Tensor]] = [[] for _ in policies]
        rollout_actions: list[list[torch.Tensor]] = [[] for _ in policies]
        old_log_probs: list[list[torch.Tensor]] = [[] for _ in policies]
        rollout_masks: list[list[torch.Tensor]] = [[] for _ in policies]
        rewards: list[float] = []
        done = False
        while not done:
            obs = _obs_tensor(observations, config).to(device)
            masks = env.action_masks()
            actions: list[int] = []
            for machine, policy in enumerate(policies):
                logits, _ = policy(obs[machine : machine + 1])
                distribution = Categorical(logits=_masked_logits(logits, masks[machine]))
                action = distribution.sample()
                actions.append(int(action.item()))
                rollout_obs[machine].append(obs[machine].detach())
                rollout_actions[machine].append(action.detach().squeeze(0))
                old_log_probs[machine].append(distribution.log_prob(action).detach().squeeze(0))
                rollout_masks[machine].append(torch.tensor(masks[machine], dtype=torch.bool, device=device))
            observations, reward, done, _ = env.step(actions)
            rewards.append(float(reward))
        returns = _returns(rewards, settings.gamma, device=device)
        for machine, policy in enumerate(policies):
            observations_tensor = torch.stack(rollout_obs[machine])
            actions_tensor = torch.stack(rollout_actions[machine])
            old_log_probs_tensor = torch.stack(old_log_probs[machine])
            masks_tensor = torch.stack(rollout_masks[machine])
            with torch.no_grad():
                _, old_values = policy(observations_tensor)
                advantages = returns - old_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            for _ in range(settings.update_epochs):
                logits, values_tensor = policy(observations_tensor)
                distribution = Categorical(logits=_masked_logits(logits, masks_tensor))
                log_probs_tensor = distribution.log_prob(actions_tensor)
                ratio = torch.exp(log_probs_tensor - old_log_probs_tensor)
                clipped = torch.clamp(ratio, 1.0 - settings.clip_ratio, 1.0 + settings.clip_ratio)
                policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
                value_loss = (values_tensor - returns).pow(2).mean()
                loss = policy_loss + settings.value_weight * value_loss - settings.entropy_weight * distribution.entropy().mean()
                optimizers[machine].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizers[machine].step()
        progress.append(
            {"policy": "independent_ppo", "train_seed": seed, "episode": episode, "objective": -sum(rewards)}
        )
        if episode in targets:
            checkpoints[episode] = _save_independent_ppo(
                policies, seed, root / f"budget_{episode}" / "model.pt"
            )
    return checkpoints, progress


def _save_centralized_ppo(policy: CentralizedPPO, seed: int, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "seed": seed}, path)
    return path


def _train_centralized_ppo_checkpoints(
    config: PassiveConfig,
    seed: int,
    budgets: tuple[int, ...],
    root: Path,
    settings: PPOSettings,
    device: torch.device,
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    policy = CentralizedPPO(config.machines, config.observation_dim, config.technicians + 1).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
    checkpoints: dict[int, Path] = {}
    progress: list[dict[str, Any]] = []
    targets = set(budgets)
    for episode in tqdm(
        range(1, max(budgets) + 1), desc=f"Centralized PPO {seed}", unit="episode", leave=False
    ):
        env = PassiveTechnicianEnv(config, seed=seed * 100_000 + episode - 1)
        observations = env.reset()
        rollout_obs: list[torch.Tensor] = []
        rollout_actions: list[torch.Tensor] = []
        old_log_probs: list[torch.Tensor] = []
        rollout_masks: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        while not done:
            flattened = _obs_tensor(observations, config).flatten().unsqueeze(0).to(device)
            logits, _ = policy(flattened)
            masks = env.action_masks()
            distributions = [
                Categorical(logits=_masked_logits(logits[0, machine], masks[machine]))
                for machine in range(config.machines)
            ]
            actions = [int(distribution.sample().item()) for distribution in distributions]
            rollout_obs.append(flattened.squeeze(0).detach())
            rollout_actions.append(torch.tensor(actions, dtype=torch.long, device=device))
            rollout_masks.append(torch.tensor(masks, dtype=torch.bool, device=device))
            old_log_probs.append(
                torch.stack(
                    [distribution.log_prob(torch.tensor(action, device=device)) for distribution, action in zip(distributions, actions)]
                ).sum().detach()
            )
            observations, reward, done, _ = env.step(actions)
            rewards.append(float(reward))
        returns = _returns(rewards, settings.gamma, device=device)
        observations_tensor = torch.stack(rollout_obs)
        actions_tensor = torch.stack(rollout_actions)
        old_log_probs_tensor = torch.stack(old_log_probs)
        masks_tensor = torch.stack(rollout_masks)
        with torch.no_grad():
            _, old_values = policy(observations_tensor)
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(settings.update_epochs):
            logits, values_tensor = policy(observations_tensor)
            distributions = [
                Categorical(logits=_masked_logits(logits[:, machine], masks_tensor[:, machine]))
                for machine in range(config.machines)
            ]
            log_probs_tensor = torch.stack(
                [distribution.log_prob(actions_tensor[:, machine]) for machine, distribution in enumerate(distributions)], dim=1
            ).sum(dim=1)
            entropy = torch.stack([distribution.entropy() for distribution in distributions], dim=1).mean()
            ratio = torch.exp(log_probs_tensor - old_log_probs_tensor)
            clipped = torch.clamp(ratio, 1.0 - settings.clip_ratio, 1.0 + settings.clip_ratio)
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = (values_tensor - returns).pow(2).mean()
            loss = policy_loss + settings.value_weight * value_loss - settings.entropy_weight * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        progress.append(
            {"policy": "centralized_ppo", "train_seed": seed, "episode": episode, "objective": -sum(rewards)}
        )
        if episode in targets:
            checkpoints[episode] = _save_centralized_ppo(
                policy, seed, root / f"budget_{episode}" / "model.pt"
            )
    return checkpoints, progress


def _with_budget(row: dict[str, Any], budget: int, train_seed: int | str) -> dict[str, Any]:
    return {
        "policy": row["policy"],
        "budget": budget,
        "seed": row["seed"],
        "train_seed": train_seed,
        "objective": row["objective"],
        "failures": row["failures"],
        "jobs": row["jobs"],
        "collisions": row["collisions"],
        "waiting": row["waiting"],
        "invalid_requests": row.get("invalid_requests", 0),
        "busy_requests": row.get("busy_requests", 0),
    }


def _budget_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["policy"]), int(row["budget"]), str(row["train_seed"]))
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for (policy, budget, train_seed), group in sorted(groups.items()):
        output.append(
            {
                "policy": policy,
                "budget": budget,
                "train_seed": train_seed,
                "evaluation_count": len(group),
                "objective_mean": statistics.fmean(float(row["objective"]) for row in group),
                "failures_mean": statistics.fmean(float(row["failures"]) for row in group),
                "jobs_mean": statistics.fmean(float(row["jobs"]) for row in group),
                "collisions_mean": statistics.fmean(float(row["collisions"]) for row in group),
                "waiting_mean": statistics.fmean(float(row["waiting"]) for row in group),
                "invalid_requests_mean": statistics.fmean(
                    float(row.get("invalid_requests", 0)) for row in group
                ),
                "busy_requests_mean": statistics.fmean(
                    float(row.get("busy_requests", 0)) for row in group
                ),
            }
        )
    return output


def _summary(rows: list[dict[str, Any]], train_seeds: tuple[int, ...], eval_seeds: tuple[int, ...], budgets: tuple[int, ...]) -> dict[str, Any]:
    budget_rows = _budget_summary(rows)
    curves: dict[str, list[dict[str, Any]]] = {}
    for policy in ALGORITHMS:
        points: list[dict[str, Any]] = []
        for budget in budgets:
            selected = [row for row in budget_rows if row["policy"] == policy and int(row["budget"]) == budget]
            seed_means = [float(row["objective_mean"]) for row in selected]
            points.append(
                {
                    "budget": budget,
                    "train_seed_count": len(seed_means),
                    "objective_mean": statistics.fmean(seed_means) if seed_means else None,
                    "objective_std_across_train_seeds": statistics.stdev(seed_means) if len(seed_means) > 1 else 0.0,
                    "train_seed_means": seed_means,
                }
            )
        curves[policy] = points
    expected = len(FIXED_POLICIES) * len(eval_seeds) + len(ALGORITHMS) * len(train_seeds) * len(budgets) * len(eval_seeds)
    learned = [row for row in rows if row["policy"] in ALGORITHMS]
    budget_by_seed: dict[str, dict[str, dict[int, float]]] = {}
    for row in budget_rows:
        budget_by_seed.setdefault(str(row["policy"]), {}).setdefault(
            str(row["train_seed"]), {}
        )[int(row["budget"])] = float(row["objective_mean"])
    paired_objective_contrasts: dict[str, dict[str, Any]] = {}
    t_critical_95 = {3: 4.302653, 10: 2.262157}
    for policy in ALGORITHMS:
        seed_curves = budget_by_seed.get(policy, {})
        deltas = [
            values[max(budgets)] - values[min(budgets)]
            for values in seed_curves.values()
            if min(budgets) in values and max(budgets) in values
        ]
        delta_mean = statistics.fmean(deltas) if deltas else None
        standard_error = (
            statistics.stdev(deltas) / len(deltas) ** 0.5 if len(deltas) > 1 else 0.0
        )
        critical = t_critical_95.get(len(deltas), 1.96)
        half_width = critical * standard_error
        paired_objective_contrasts[policy] = {
            "low_budget": min(budgets),
            "high_budget": max(budgets),
            "training_seed_count": len(deltas),
            "training_seed_deltas": deltas,
            "mean_delta_high_minus_low": delta_mean,
            "confidence_interval_95": (
                [delta_mean - half_width, delta_mean + half_width]
                if delta_mean is not None
                else None
            ),
            "supports_objective_reduction": bool(
                delta_mean is not None and delta_mean + half_width < 0
            ),
        }
    return {
        "purpose": "Training-budget diagnostic; not a final algorithm comparison.",
        "primary_metric": "mean objective cost on the common evaluation panel",
        "curves": curves,
        "paired_objective_contrasts": paired_objective_contrasts,
        "audits": {
            "expected_episode_count": expected,
            "episode_count": len(rows),
            "all_expected_rows_present": len(rows) == expected,
            "all_learned_train_seeds_recorded": all(str(row["train_seed"]) for row in learned),
            "sealed_test_panel_closed": True,
            "nonnegative_objectives": all(float(row["objective"]) >= 0 for row in rows),
        },
    }


def run(args: argparse.Namespace) -> Path:
    train_seeds, evaluation_seeds, budgets = _profile(args.profile)
    config = stress_config()
    device = resolve_device(args.device)
    settings = PPOSettings(episodes=max(budgets))
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "status": "RUNNING",
        "experiment": (
            "passive_technician_budget_replication"
            if args.profile == "replication"
            else "passive_technician_budget_screen"
        ),
        "profile": args.profile,
        "stress_config": asdict(config),
        "budgets": list(budgets),
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "algorithms": list(ALGORITHMS),
        "primary_contrast": (
            "centralized_ppo objective at 20,000 minus 5,000 episodes across fresh training seeds"
            if args.profile == "replication"
            else None
        ),
        "fixed_policies": list(FIXED_POLICIES),
        "settings": asdict(settings),
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device,
        "resolved_device": str(device),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "started_at": datetime.now(UTC).isoformat(),
        "sealed_test_evaluated": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "benchmark_config.json").write_text(
        json.dumps({"stress_config": asdict(config), "budgets": budgets, "train_seeds": train_seeds, "evaluation_seeds": evaluation_seeds, "settings": asdict(settings)}, indent=2, sort_keys=True) + "\n"
    )
    rows: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    fixed_eval_seeds = tqdm(evaluation_seeds, desc="fixed policies", unit="episode")
    for seed in fixed_eval_seeds:
        for policy in FIXED_POLICIES:
            rows.append(_with_budget(evaluate_fixed(config, policy, seed), 0, ""))
    _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)
    for algorithm in ALGORITHMS:
        for train_seed in tqdm(train_seeds, desc=algorithm, unit="seed"):
            root = output / algorithm / f"train_seed_{train_seed}"
            if algorithm == "independent_q":
                checkpoints, training_rows = _train_q_checkpoints(config, train_seed, budgets, root)
            elif algorithm == "independent_ppo":
                checkpoints, training_rows = _train_independent_ppo_checkpoints(config, train_seed, budgets, root, settings, device)
            else:
                checkpoints, training_rows = _train_centralized_ppo_checkpoints(config, train_seed, budgets, root, settings, device)
            progress.extend(training_rows)
            for budget in budgets:
                checkpoint = checkpoints[budget]
                if algorithm == "independent_q":
                    policy = IndependentQ.load(checkpoint, config, seed=train_seed)
                elif algorithm == "independent_ppo":
                    policy = load_independent_ppo(config, checkpoint)
                    policy = [item.to(device) for item in policy]
                else:
                    policy = load_centralized_ppo(config, checkpoint)
                    policy = policy.to(device)
                for seed in tqdm(evaluation_seeds, desc=f"evaluate {algorithm}/{train_seed}/{budget}", unit="episode", leave=False):
                    if algorithm == "independent_q":
                        evaluated = dict(_evaluate_q(config, policy, seed, train_seed))
                        evaluated["policy"] = algorithm
                    elif algorithm == "independent_ppo":
                        evaluated = _evaluate_independent_ppo(config, policy, seed, train_seed, device)
                    else:
                        evaluated = _evaluate_centralized_ppo(config, policy, seed, train_seed, device)
                    rows.append(_with_budget(evaluated, budget, train_seed))
                _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)
    _write_csv(rows, output / "episodes.csv", FIELDNAMES)
    _write_csv(rows, output / "coordination.csv", FIELDNAMES)
    _write_csv(progress, output / "training_progress.csv")
    budget_rows = _budget_summary(rows)
    _write_csv(budget_rows, output / "budget_summary.csv")
    summary = _summary(rows, train_seeds, evaluation_seeds, budgets)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest.update(
        {
            "status": "COMPLETED",
            "finished_at": datetime.now(UTC).isoformat(),
            "episode_count": len(rows),
            "checkpoint_count": sum(1 for path in output.rglob("model.*") if path.is_file()),
            "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def _evaluate_q(config: PassiveConfig, learner: IndependentQ, seed: int, train_seed: int) -> dict[str, Any]:
    env = PassiveTechnicianEnv(config, seed=seed)
    observations = env.reset()
    done = False
    while not done:
        masks = env.action_masks()
        actions = tuple(
            learner.action(machine, observations[machine], 0.0, masks[machine])
            for machine in range(config.machines)
        )
        observations, _, done, _ = env.step(actions)
    return {"policy": "independent_q", "seed": seed, "train_seed": train_seed, **env.metrics}


def _evaluate_independent_ppo(
    config: PassiveConfig,
    policies: list[ActorCritic],
    seed: int,
    train_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    env = PassiveTechnicianEnv(config, seed=seed)
    observations = env.reset()
    done = False
    while not done:
        obs = _obs_tensor(observations, config).to(device)
        masks = env.action_masks()
        with torch.no_grad():
            actions = [
                int(
                    torch.argmax(
                        _masked_logits(policy(obs[machine : machine + 1])[0], masks[machine]),
                        dim=-1,
                    ).item()
                )
                for machine, policy in enumerate(policies)
            ]
        observations, _, done, _ = env.step(actions)
    return {"policy": "independent_ppo", "seed": seed, "train_seed": train_seed, **env.metrics}


def _evaluate_centralized_ppo(
    config: PassiveConfig,
    policy: CentralizedPPO,
    seed: int,
    train_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    env = PassiveTechnicianEnv(config, seed=seed)
    observations = env.reset()
    done = False
    while not done:
        masks = env.action_masks()
        with torch.no_grad():
            logits, _ = policy(_obs_tensor(observations, config).flatten().unsqueeze(0).to(device))
            actions = [
                int(torch.argmax(_masked_logits(logits[0, machine], masks[machine])).item())
                for machine in range(config.machines)
            ]
        observations, _, done, _ = env.step(actions)
    return {"policy": "centralized_ppo", "seed": seed, "train_seed": train_seed, **env.metrics}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full", "replication"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
