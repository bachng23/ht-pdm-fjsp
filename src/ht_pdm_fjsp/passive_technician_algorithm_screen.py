"""Screen cooperative MARL families on the corrected passive-technician cell."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import (
    ActorCritic,
    CentralizedPPO,
    PPOSettings,
    _action_diagnostics,
    _masked_logits,
    _obs_tensor,
    _returns,
    evaluate_fixed,
    load_centralized_ppo,
    load_independent_ppo,
    stress_config,
)
from ht_pdm_fjsp.passive_technician_budget_screen import (
    _evaluate_centralized_ppo,
    _evaluate_independent_ppo,
    _evaluate_q,
    _train_centralized_ppo_checkpoints,
    _train_independent_ppo_checkpoints,
    _train_q_checkpoints,
)
from ht_pdm_fjsp.passive_technician_marl import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    IndependentQ,
    PassiveConfig,
    PassiveTechnicianEnv,
)


ALGORITHMS = (
    "independent_q",
    "independent_ppo",
    "mappo_ctde",
    "coma_counterfactual",
)
FIXED_POLICIES = ("random_feasible", "skill_aware_fifo")
SMOKE_EPISODES = 32
FULL_EPISODES = 5_000
FIELDNAMES = (
    "policy",
    "seed",
    "train_seed",
    "objective",
    "failures",
    "jobs",
    "collisions",
    "waiting",
    "invalid_requests",
    "busy_requests",
    "unique_joint_actions",
    "defer_fraction",
    "action_steps",
    "request_count",
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


class ComaPolicy(nn.Module):
    """Small COMA-style policy with a centralized joint-action critic."""

    def __init__(self, config: PassiveConfig, hidden_dim: int = 64) -> None:
        super().__init__()
        self.machines = config.machines
        self.observation_dim = config.observation_dim
        self.action_dim = config.technicians + 1
        self.hidden_dim = hidden_dim
        global_dim = config.machines * config.observation_dim
        joint_dim = config.machines * self.action_dim
        self.actors = nn.ModuleList(
            ActorCritic(self.observation_dim, self.action_dim)
            for _ in range(config.machines)
        )
        self.critic = nn.Sequential(
            nn.Linear(global_dim + joint_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def actor_logits(self, observations: torch.Tensor, machine: int) -> torch.Tensor:
        return self.actors[machine](observations[:, machine])[0]

    def critic_value(self, global_observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        one_hot = F.one_hot(actions, num_classes=self.action_dim).float().flatten(start_dim=1)
        features = torch.cat((global_observations, one_hot), dim=-1)
        return self.critic(features).squeeze(-1)

    def save(self, path: Path, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"actors": [actor.state_dict() for actor in self.actors], "critic": self.critic.state_dict(), "hidden_dim": self.hidden_dim, "seed": seed}, path)

    @classmethod
    def load(cls, path: Path, config: PassiveConfig, device: torch.device) -> "ComaPolicy":
        payload = torch.load(path, map_location=device, weights_only=True)
        model = cls(config, hidden_dim=int(payload.get("hidden_dim", 64))).to(device)
        for actor, state in zip(model.actors, payload["actors"], strict=True):
            actor.load_state_dict(state)
        model.critic.load_state_dict(payload["critic"])
        model.eval()
        return model


def _train_coma(
    config: PassiveConfig,
    seed: int,
    episodes: int,
    path: Path,
    settings: PPOSettings,
    device: torch.device,
) -> tuple[Path, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = ComaPolicy(config).to(device)
    actor_optimizers = [
        torch.optim.Adam(actor.parameters(), lr=settings.learning_rate)
        for actor in model.actors
    ]
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=settings.learning_rate)
    progress: list[dict[str, Any]] = []
    for episode in tqdm(range(1, episodes + 1), desc=f"COMA {seed}", unit="episode", leave=False):
        env = PassiveTechnicianEnv(config, seed=seed * 100_000 + episode - 1)
        observations = env.reset()
        rollout_observations: list[torch.Tensor] = []
        rollout_masks: list[torch.Tensor] = []
        rollout_actions: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        while not done:
            observation_tensor = _obs_tensor(observations, config).to(device)
            masks = torch.tensor(env.action_masks(), dtype=torch.bool, device=device)
            actions: list[int] = []
            for machine in range(config.machines):
                logits = model.actor_logits(observation_tensor.unsqueeze(0), machine)
                distribution = Categorical(logits=_masked_logits(logits, masks[machine]))
                actions.append(int(distribution.sample().item()))
            rollout_observations.append(observation_tensor.detach())
            rollout_masks.append(masks)
            rollout_actions.append(torch.tensor(actions, dtype=torch.long, device=device))
            observations, reward, done, _ = env.step(actions)
            rewards.append(float(reward))

        global_observations = torch.stack(rollout_observations).flatten(start_dim=1)
        masks_tensor = torch.stack(rollout_masks)
        actions_tensor = torch.stack(rollout_actions)
        returns = _returns(rewards, settings.gamma, device=device)
        for _ in range(settings.update_epochs):
            critic_loss = (model.critic_value(global_observations, actions_tensor) - returns).pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            critic_optimizer.step()

        with torch.no_grad():
            actual_q = model.critic_value(global_observations, actions_tensor)
            advantages: list[torch.Tensor] = []
            for machine in range(config.machines):
                baseline_values: list[torch.Tensor] = []
                for timestep in range(len(rewards)):
                    valid_actions = torch.where(masks_tensor[timestep, machine])[0]
                    counterfactual = actions_tensor[timestep].repeat(len(valid_actions), 1)
                    counterfactual[:, machine] = valid_actions
                    global_batch = global_observations[timestep].repeat(len(valid_actions), 1)
                    logits = model.actor_logits(
                        rollout_observations[timestep].unsqueeze(0), machine
                    )
                    distribution = Categorical(logits=_masked_logits(logits, masks_tensor[timestep, machine]))
                    probabilities = distribution.probs.squeeze(0)[valid_actions]
                    baseline_values.append(
                        (probabilities * model.critic_value(global_batch, counterfactual)).sum()
                    )
                advantages.append(actual_q - torch.stack(baseline_values))
        for machine, actor in enumerate(model.actors):
            logits = torch.stack(
                [model.actor_logits(observation.unsqueeze(0), machine).squeeze(0) for observation in rollout_observations]
            )
            distributions = Categorical(logits=_masked_logits(logits, masks_tensor[:, machine]))
            log_probs = distributions.log_prob(actions_tensor[:, machine])
            loss = -(log_probs * advantages[machine].detach()).mean() - settings.entropy_weight * distributions.entropy().mean()
            actor_optimizers[machine].zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            actor_optimizers[machine].step()
        progress.append(
            {"policy": "coma_counterfactual", "train_seed": seed, "episode": episode, "objective": -sum(rewards)}
        )
    model.save(path, seed)
    return path, progress


def _evaluate_coma(
    config: PassiveConfig,
    model: ComaPolicy,
    seed: int,
    train_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    env = PassiveTechnicianEnv(config, seed=seed)
    observations = env.reset()
    joint_actions: set[tuple[int, ...]] = set()
    defer_actions = 0
    action_steps = 0
    done = False
    while not done:
        observation_tensor = _obs_tensor(observations, config).to(device)
        masks = env.action_masks()
        actions: list[int] = []
        with torch.no_grad():
            for machine, actor in enumerate(model.actors):
                logits = actor(observation_tensor[machine : machine + 1])[0]
                actions.append(int(torch.argmax(_masked_logits(logits, masks[machine])).item()))
        joint_action = tuple(actions)
        joint_actions.add(joint_action)
        defer_actions += sum(action == 0 for action in joint_action)
        action_steps += 1
        observations, _, done, _ = env.step(actions)
    return {
        "policy": "coma_counterfactual",
        "seed": seed,
        "train_seed": train_seed,
        **env.metrics,
        **_action_diagnostics(config, joint_actions, defer_actions, action_steps),
    }


def _profile(profile: str) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if profile == "smoke":
        return (11,), (101, 102, 103), SMOKE_EPISODES
    if profile == "full":
        return (11, 12, 13), tuple(range(101, 131)), FULL_EPISODES
    raise ValueError(profile)


def _with_policy(row: dict[str, Any], policy: str, train_seed: int | str) -> dict[str, Any]:
    return {
        field: row.get(field, 0 if field not in {"policy", "seed", "train_seed"} else "")
        for field in FIELDNAMES
    } | {"policy": policy, "train_seed": train_seed}


def _summary(
    rows: list[dict[str, Any]],
    train_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
) -> dict[str, Any]:
    learned = [row for row in rows if row["policy"] in ALGORITHMS]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in learned:
        grouped.setdefault((str(row["policy"]), str(row["train_seed"])), []).append(row)
    per_seed: dict[str, list[dict[str, Any]]] = {}
    for (policy, train_seed), group in sorted(grouped.items()):
        per_seed.setdefault(policy, []).append(
            {
                "train_seed": int(train_seed),
                "evaluation_count": len(group),
                "objective_mean": statistics.fmean(float(row["objective"]) for row in group),
                "waiting_mean": statistics.fmean(float(row["waiting"]) for row in group),
                "invalid_requests_mean": statistics.fmean(float(row["invalid_requests"]) for row in group),
                "unique_joint_actions_mean": statistics.fmean(float(row["unique_joint_actions"]) for row in group),
            }
        )
    curves = {
        policy: {
            "objective_mean": statistics.fmean(item["objective_mean"] for item in values),
            "objective_std_across_train_seeds": statistics.stdev(item["objective_mean"] for item in values) if len(values) > 1 else 0.0,
            "waiting_mean": statistics.fmean(item["waiting_mean"] for item in values),
            "invalid_requests_mean": statistics.fmean(item["invalid_requests_mean"] for item in values),
            "unique_joint_actions_mean": statistics.fmean(item["unique_joint_actions_mean"] for item in values),
            "train_seed_count": len(values),
        }
        for policy, values in per_seed.items()
    }
    expected = len(FIXED_POLICIES) * len(evaluation_seeds) + len(ALGORITHMS) * len(train_seeds) * len(evaluation_seeds)
    return {
        "purpose": "Algorithm-family screening; not a final algorithm comparison.",
        "primary_metric": "mean objective cost on the common evaluation panel",
        "per_training_seed": per_seed,
        "curves": curves,
        "audits": {
            "expected_episode_count": expected,
            "episode_count": len(rows),
            "all_expected_rows_present": len(rows) == expected,
            "all_learned_train_seeds_recorded": all(str(row["train_seed"]) for row in learned),
            "invalid_requests_zero": all(float(row["invalid_requests"]) == 0 for row in learned),
            "sealed_test_panel_closed": True,
            "nonnegative_objectives": all(float(row["objective"]) >= 0 for row in rows),
        },
    }


def run(args: argparse.Namespace) -> Path:
    train_seeds, evaluation_seeds, episodes = _profile(args.profile)
    config = stress_config()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = PPOSettings(episodes=episodes)
    manifest = {
        "status": "RUNNING",
        "experiment": "passive_technician_algorithm_screen",
        "profile": args.profile,
        "algorithms": list(ALGORITHMS),
        "fixed_policies": list(FIXED_POLICIES),
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "episodes_per_train_seed": episodes,
        "stress_config": asdict(config),
        "settings": asdict(settings),
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device,
        "resolved_device": str(device),
        "git_revision": _git_revision(),
        "sealed_test_evaluated": False,
        "started_at": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "benchmark_config.json").write_text(
        json.dumps({"stress_config": asdict(config), "train_seeds": train_seeds, "evaluation_seeds": evaluation_seeds, "episodes": episodes, "settings": asdict(settings)}, indent=2, sort_keys=True) + "\n"
    )
    rows: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    for seed in tqdm(evaluation_seeds, desc="fixed policies", unit="episode"):
        for policy in FIXED_POLICIES:
            rows.append(_with_policy(evaluate_fixed(config, policy, seed), policy, ""))
    _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)
    for algorithm in ALGORITHMS:
        for train_seed in tqdm(train_seeds, desc=algorithm, unit="seed"):
            root = output / algorithm / f"train_seed_{train_seed}"
            if algorithm == "independent_q":
                checkpoints, training_rows = _train_q_checkpoints(config, train_seed, (episodes,), root)
                checkpoint = checkpoints[episodes]
                policy = IndependentQ.load(checkpoint, config, seed=train_seed)
            elif algorithm == "independent_ppo":
                checkpoints, training_rows = _train_independent_ppo_checkpoints(config, train_seed, (episodes,), root, settings, device)
                checkpoint = checkpoints[episodes]
                policy = load_independent_ppo(config, checkpoint)
                policy = [item.to(device) for item in policy]
            elif algorithm == "mappo_ctde":
                checkpoints, training_rows = _train_centralized_ppo_checkpoints(config, train_seed, (episodes,), root, settings, device)
                checkpoint = checkpoints[episodes]
                policy = load_centralized_ppo(config, checkpoint).to(device)
            else:
                checkpoint, training_rows = _train_coma(config, train_seed, episodes, root / f"budget_{episodes}" / "model.pt", settings, device)
                policy = ComaPolicy.load(checkpoint, config, device)
            progress.extend({**row, "policy": algorithm} for row in training_rows)
            for seed in tqdm(evaluation_seeds, desc=f"evaluate {algorithm}/{train_seed}", unit="episode", leave=False):
                if algorithm == "independent_q":
                    evaluated = _evaluate_q(config, policy, seed, train_seed)
                elif algorithm == "independent_ppo":
                    evaluated = _evaluate_independent_ppo(config, policy, seed, train_seed, device)
                elif algorithm == "mappo_ctde":
                    evaluated = _evaluate_centralized_ppo(config, policy, seed, train_seed, device)
                else:
                    evaluated = _evaluate_coma(config, policy, seed, train_seed, device)
                rows.append(_with_policy(evaluated, algorithm, train_seed))
            _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)
    _write_csv(rows, output / "episodes.csv", FIELDNAMES)
    _write_csv(rows, output / "coordination.csv", FIELDNAMES)
    _write_csv(progress, output / "training_progress.csv")
    summary = _summary(rows, train_seeds, evaluation_seeds)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    budget_rows = []
    for policy in ALGORITHMS + FIXED_POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        if group:
            budget_rows.append({"policy": policy, "evaluation_count": len(group), "objective_mean": statistics.fmean(float(row["objective"]) for row in group), "waiting_mean": statistics.fmean(float(row["waiting"]) for row in group), "invalid_requests_mean": statistics.fmean(float(row["invalid_requests"]) for row in group), "unique_joint_actions_mean": statistics.fmean(float(row["unique_joint_actions"]) for row in group)})
    _write_csv(budget_rows, output / "budget_summary.csv")
    manifest.update({"status": "COMPLETED", "finished_at": datetime.now(UTC).isoformat(), "episode_count": len(rows), "checkpoint_count": sum(1 for path in output.rglob("model.*") if path.is_file()), "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
