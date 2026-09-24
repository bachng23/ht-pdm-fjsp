"""Baseline suite for the passive shared-technician research simulator."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_marl import (
    IndependentQ,
    PassiveConfig,
    PassiveTechnicianEnv,
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    dispatcher_action,
    exact_optimum,
    evaluate_marl,
    train_independent,
)


ALGORITHMS = ("independent_q", "independent_ppo", "centralized_ppo")


def stress_config() -> PassiveConfig:
    return PassiveConfig(
        machines=3,
        technicians=2,
        horizon=12,
        failure_age=5,
        max_age=8,
        service_time=((2, 4), (4, 2), (3, 3)),
        semantics="fifo_queue",
    )


def oracle_config() -> PassiveConfig:
    return PassiveConfig(
        machines=2,
        technicians=2,
        horizon=6,
        failure_age=3,
        max_age=5,
        service_time=((2, 4), (4, 2)),
        semantics="fifo_queue",
    )


def random_feasible_action(env: PassiveTechnicianEnv) -> tuple[int, ...]:
    available = [technician for technician in range(env.config.technicians) if env.available(technician)]
    env.rng.shuffle(available)
    actions = [0] * env.config.machines
    for machine in env.rng.sample(range(env.config.machines), env.config.machines):
        if not available:
            break
        if env.state.failed[machine] or env.state.ages[machine] >= env.config.failure_age - 1:
            technician = available.pop()
            actions[machine] = technician + 1
    return tuple(actions)


def evaluate_fixed(config: PassiveConfig, policy_name: str, seed: int) -> dict[str, Any]:
    env = PassiveTechnicianEnv(config, seed=seed)
    env.reset()
    done = False
    while not done:
        action = random_feasible_action(env) if policy_name == "random_feasible" else dispatcher_action(env)
        _, _, done, _ = env.step(action)
    return {"policy": policy_name, "seed": seed, "train_seed": "", **env.metrics}


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(nn.Linear(observation_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.policy = nn.Linear(64, action_dim)
        self.value = nn.Linear(64, 1)

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(observations)
        return self.policy(hidden), self.value(hidden).squeeze(-1)


@dataclass(frozen=True)
class PPOSettings:
    episodes: int
    learning_rate: float = 3e-4
    gamma: float = 0.99
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.01
    update_epochs: int = 4


def _obs_tensor(
    observations: tuple[tuple[int, ...], ...],
    config: PassiveConfig,
) -> torch.Tensor:
    finite_service_times = [
        service_time
        for row in config.service_time
        for service_time in row
        if np.isfinite(service_time)
    ]
    scale = max(
        1,
        config.max_age,
        config.horizon,
        max(finite_service_times, default=1),
    )
    return torch.tensor(observations, dtype=torch.float32) / float(scale)


def _returns(rewards: list[float], gamma: float) -> torch.Tensor:
    output: list[float] = []
    running = 0.0
    for reward in reversed(rewards):
        running = reward + gamma * running
        output.append(running)
    return torch.tensor(list(reversed(output)), dtype=torch.float32)


def train_independent_ppo(config: PassiveConfig, seed: int, settings: PPOSettings, output: Path) -> Path:
    torch.manual_seed(seed)
    random.seed(seed)
    policies = [
        ActorCritic(config.observation_dim, config.technicians + 1)
        for _ in range(config.machines)
    ]
    optimizers = [torch.optim.Adam(policy.parameters(), lr=settings.learning_rate) for policy in policies]
    progress: list[dict[str, float | int]] = []
    for episode in tqdm(range(settings.episodes), desc=f"Independent PPO {seed}", unit="episode", leave=False):
        env = PassiveTechnicianEnv(config, seed=seed * 100_000 + episode)
        observations = env.reset()
        rollout_obs: list[list[torch.Tensor]] = [[] for _ in policies]
        rollout_actions: list[list[torch.Tensor]] = [[] for _ in policies]
        old_log_probs: list[list[torch.Tensor]] = [[] for _ in policies]
        rewards: list[float] = []
        done = False
        while not done:
            obs = _obs_tensor(observations, config)
            actions: list[int] = []
            for machine, policy in enumerate(policies):
                logits, value = policy(obs[machine : machine + 1])
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                actions.append(int(action.item()))
                rollout_obs[machine].append(obs[machine].detach())
                rollout_actions[machine].append(action.detach().squeeze(0))
                old_log_probs[machine].append(distribution.log_prob(action).detach().squeeze(0))
            observations, reward, done, _ = env.step(actions)
            rewards.append(float(reward))
        returns = _returns(rewards, settings.gamma)
        for machine, policy in enumerate(policies):
            observations_tensor = torch.stack(rollout_obs[machine])
            actions_tensor = torch.stack(rollout_actions[machine])
            old_log_probs_tensor = torch.stack(old_log_probs[machine])
            with torch.no_grad():
                _, old_values = policy(observations_tensor)
                advantages = returns - old_values
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            for _ in range(settings.update_epochs):
                logits, values_tensor = policy(observations_tensor)
                distribution = Categorical(logits=logits)
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
        progress.append({"episode": episode + 1, "return": sum(rewards), "objective": -sum(rewards)})
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"policies": [policy.state_dict() for policy in policies], "seed": seed}, output / "model.pt")
    _write_csv(progress, output / "training_progress.csv")
    return output / "model.pt"


def load_independent_ppo(config: PassiveConfig, path: Path) -> list[ActorCritic]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    policies = [
        ActorCritic(config.observation_dim, config.technicians + 1)
        for _ in range(config.machines)
    ]
    for policy, state in zip(policies, payload["policies"]):
        policy.load_state_dict(state)
        policy.eval()
    return policies


def evaluate_independent_ppo(config: PassiveConfig, policies: list[ActorCritic], seed: int, train_seed: int) -> dict[str, Any]:
    env = PassiveTechnicianEnv(config, seed=seed)
    observations = env.reset()
    done = False
    while not done:
        obs = _obs_tensor(observations, config)
        actions = []
        with torch.no_grad():
            for machine, policy in enumerate(policies):
                logits, _ = policy(obs[machine : machine + 1])
                actions.append(int(torch.argmax(logits, dim=-1).item()))
        observations, _, done, _ = env.step(actions)
    return {"policy": "independent_ppo", "seed": seed, "train_seed": train_seed, **env.metrics}


class CentralizedPPO(nn.Module):
    def __init__(self, machines: int, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(machines * observation_dim, 96),
            nn.Tanh(),
            nn.Linear(96, 96),
            nn.Tanh(),
        )
        self.policy = nn.Linear(96, machines * action_dim)
        self.value = nn.Linear(96, 1)
        self.machines = machines
        self.observation_dim = observation_dim
        self.action_dim = action_dim

    def forward(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(observations)
        return self.policy(hidden).view(-1, self.machines, self.action_dim), self.value(hidden).squeeze(-1)


def train_centralized_ppo(config: PassiveConfig, seed: int, settings: PPOSettings, output: Path) -> Path:
    torch.manual_seed(seed)
    random.seed(seed)
    policy = CentralizedPPO(config.machines, config.observation_dim, config.technicians + 1)
    optimizer = torch.optim.Adam(policy.parameters(), lr=settings.learning_rate)
    progress: list[dict[str, float | int]] = []
    for episode in tqdm(range(settings.episodes), desc=f"Centralized PPO {seed}", unit="episode", leave=False):
        env = PassiveTechnicianEnv(config, seed=seed * 100_000 + episode)
        observations = env.reset()
        rollout_obs: list[torch.Tensor] = []
        rollout_actions: list[torch.Tensor] = []
        old_log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        while not done:
            logits, value = policy(_obs_tensor(observations, config).flatten().unsqueeze(0))
            distributions = [Categorical(logits=logits[0, machine]) for machine in range(config.machines)]
            actions = [int(distribution.sample().item()) for distribution in distributions]
            rollout_obs.append(_obs_tensor(observations, config).flatten().detach())
            rollout_actions.append(torch.tensor(actions, dtype=torch.long))
            old_log_probs.append(torch.stack([distribution.log_prob(torch.tensor(action)) for distribution, action in zip(distributions, actions)]).detach().sum())
            observations, reward, done, _ = env.step(actions)
            rewards.append(float(reward))
        returns = _returns(rewards, settings.gamma)
        observations_tensor = torch.stack(rollout_obs)
        actions_tensor = torch.stack(rollout_actions)
        old_log_probs_tensor = torch.stack(old_log_probs)
        with torch.no_grad():
            _, old_values = policy(observations_tensor)
            advantages = returns - old_values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        for _ in range(settings.update_epochs):
            logits, value_tensor = policy(observations_tensor)
            distributions = [Categorical(logits=logits[:, machine]) for machine in range(config.machines)]
            log_probs_tensor = torch.stack(
                [distribution.log_prob(actions_tensor[:, machine]) for machine, distribution in enumerate(distributions)],
                dim=1,
            ).sum(dim=1)
            entropy = torch.stack([distribution.entropy() for distribution in distributions], dim=1).mean()
            ratio = torch.exp(log_probs_tensor - old_log_probs_tensor)
            clipped = torch.clamp(ratio, 1.0 - settings.clip_ratio, 1.0 + settings.clip_ratio)
            policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
            value_loss = (value_tensor - returns).pow(2).mean()
            loss = policy_loss + settings.value_weight * value_loss - settings.entropy_weight * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
        progress.append({"episode": episode + 1, "return": sum(rewards), "objective": -sum(rewards)})
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "seed": seed}, output / "model.pt")
    _write_csv(progress, output / "training_progress.csv")
    return output / "model.pt"


def load_centralized_ppo(config: PassiveConfig, path: Path) -> CentralizedPPO:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    policy = CentralizedPPO(config.machines, config.observation_dim, config.technicians + 1)
    policy.load_state_dict(payload["policy"])
    policy.eval()
    return policy


def evaluate_centralized_ppo(config: PassiveConfig, policy: CentralizedPPO, seed: int, train_seed: int) -> dict[str, Any]:
    env = PassiveTechnicianEnv(config, seed=seed)
    observations = env.reset()
    done = False
    while not done:
        with torch.no_grad():
            logits, _ = policy(_obs_tensor(observations, config).flatten().unsqueeze(0))
            actions = [int(torch.argmax(logits[0, machine]).item()) for machine in range(config.machines)]
        observations, _, done, _ = env.step(actions)
    return {"policy": "centralized_ppo", "seed": seed, "train_seed": train_seed, **env.metrics}


def _write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _profile(profile: str) -> tuple[tuple[int, ...], tuple[int, ...], PPOSettings]:
    if profile == "smoke":
        return (11,), (101, 102, 103), PPOSettings(episodes=8)
    if profile == "pilot":
        return (11, 12, 13), tuple(range(101, 111)), PPOSettings(episodes=500)
    return tuple(range(11, 21)), tuple(range(101, 201)), PPOSettings(episodes=5_000)


def run(args: argparse.Namespace) -> Path:
    train_seeds, evaluation_seeds, settings = _profile(args.profile)
    config = stress_config()
    oracle = oracle_config()
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "status": "RUNNING",
        "experiment": "passive_technician_baselines",
        "profile": args.profile,
        "stress_config": asdict(config),
        "oracle_config": asdict(oracle),
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "settings": asdict(settings),
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "sealed_test_evaluated": False,
        "started_at": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "benchmark_config.json").write_text(
        json.dumps(
            {
                "stress_config": asdict(config),
                "oracle_config": asdict(oracle),
                "settings": asdict(settings),
                "objective_version": OBJECTIVE_VERSION,
                "observation_version": OBSERVATION_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    exact = exact_optimum(oracle)
    rows: list[dict[str, Any]] = []
    fixed_rows = [evaluate_fixed(config, policy, seed) for policy in ("random_feasible", "skill_aware_fifo") for seed in tqdm(evaluation_seeds, desc=policy, unit="episode")]
    rows.extend(fixed_rows)
    _write_csv(rows, output / "episodes.partial.csv")
    for algorithm in ALGORITHMS:
        for train_seed in tqdm(train_seeds, desc=algorithm, unit="seed"):
            model_dir = output / algorithm / f"train_seed_{train_seed}"
            if algorithm == "independent_q":
                learner = train_independent(config, train_seed, settings.episodes)
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "model.json"
                learner.save(model_path)
                policy = learner
            elif algorithm == "independent_ppo":
                model_path = train_independent_ppo(config, train_seed, settings, model_dir)
                policy = load_independent_ppo(config, model_path)
            else:
                model_path = train_centralized_ppo(config, train_seed, settings, model_dir)
                policy = load_centralized_ppo(config, model_path)
            for seed in tqdm(evaluation_seeds, desc=f"evaluate {algorithm}/{train_seed}", unit="episode", leave=False):
                if algorithm == "independent_q":
                    row = evaluate_marl(config, policy, seed, train_seed)
                    row["policy"] = "independent_q"
                elif algorithm == "independent_ppo":
                    row = evaluate_independent_ppo(config, policy, seed, train_seed)
                else:
                    row = evaluate_centralized_ppo(config, policy, seed, train_seed)
                rows.append(row)
            _write_csv(rows, output / "episodes.partial.csv")
    _write_csv(rows, output / "episodes.csv")
    _write_csv(rows, output / "coordination.csv")
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(str(row["policy"]), []).append(float(row["objective"]))
    summary = {
        "oracle_expected_cost_2x2": exact,
        "stress_mean_cost": {key: statistics.fmean(value) for key, value in grouped.items()},
        "audits": {
            "episode_count": len(rows),
            "sealed_panel_closed": True,
            "checkpoint_count": sum(
                1 for path in output.glob("*/train_seed_*/*")
                if path.is_file() and path.name.startswith("model.")
            ),
            "all_train_seeds_recorded": all(row["train_seed"] != "" for row in rows if row["policy"] in ALGORITHMS),
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest.update({"status": "COMPLETED", "finished_at": datetime.now(UTC).isoformat(), "oracle_expected_cost_2x2": exact, "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file())})
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
