"""Parameter-sharing MAPPO with decentralized actors and a centralized critic."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np
import torch as th
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig


@dataclass(frozen=True)
class MAPPOSettings:
    total_timesteps: int
    n_envs: int
    n_steps: int
    batch_size: int
    n_epochs: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coefficient: float
    value_coefficient: float
    max_grad_norm: float
    actor_hidden_dim: int
    critic_hidden_dim: int
    device: str


class MachineMAPPO(nn.Module):
    """Shared local actor plus global-state value function for CTDE."""

    def __init__(
        self,
        *,
        local_feature_dim: int,
        global_state_dim: int,
        actor_hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.local_feature_dim = local_feature_dim
        self.global_state_dim = global_state_dim
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        self.actor = nn.Sequential(
            nn.Linear(local_feature_dim, actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.Tanh(),
            nn.Linear(actor_hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(global_state_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.Tanh(),
            nn.Linear(critic_hidden_dim, 1),
        )
        for layer in self.actor:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        for layer in self.critic:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def distribution(
        self, local_observations: th.Tensor, action_masks: th.Tensor
    ) -> Categorical:
        logits = self.actor(local_observations).squeeze(-1)
        logits = logits.masked_fill(~action_masks.bool(), -1e9)
        return Categorical(logits=logits)

    def value(self, global_state: th.Tensor) -> th.Tensor:
        return self.critic(global_state).squeeze(-1)

    def act(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        local = th.as_tensor(
            observation["local_observations"], dtype=th.float32, device=device
        )
        masks = th.as_tensor(
            observation["action_masks"], dtype=th.bool, device=device
        )
        global_state = th.as_tensor(
            observation["global_state"], dtype=th.float32, device=device
        )
        with th.no_grad():
            distribution = self.distribution(local, masks)
            actions = (
                th.argmax(distribution.logits, dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy().mean(dim=-1)
            value = self.value(global_state)
        return (
            actions.cpu().numpy(),
            log_prob.cpu().numpy(),
            value.cpu().numpy(),
            entropy.cpu().numpy(),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(
            {
                "state_dict": self.state_dict(),
                "local_feature_dim": self.local_feature_dim,
                "global_state_dim": self.global_state_dim,
                "actor_hidden_dim": self.actor_hidden_dim,
                "critic_hidden_dim": self.critic_hidden_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str) -> "MachineMAPPO":
        payload = th.load(path, map_location=device, weights_only=False)
        model = cls(
            local_feature_dim=int(payload["local_feature_dim"]),
            global_state_dim=int(payload["global_state_dim"]),
            actor_hidden_dim=int(payload["actor_hidden_dim"]),
            critic_hidden_dim=int(payload["critic_hidden_dim"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def _stack_observations(
    observations: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_mappo(
    config: BenchmarkConfig,
    settings: MAPPOSettings,
    output_dir: Path,
    *,
    train_seed: int,
    show_progress: bool,
) -> tuple[MachineMAPPO, float]:
    """Train a CTDE policy with PPO on the factorized joint actor."""

    if settings.n_steps * settings.n_envs % settings.batch_size:
        raise ValueError("n_steps * n_envs must be divisible by batch_size")
    np.random.seed(train_seed)
    th.manual_seed(train_seed)
    rng = np.random.default_rng(train_seed)
    envs = [MachineAgentsCTDEEnv(config) for _ in range(settings.n_envs)]
    episode_counts = [0 for _ in envs]
    observations = [
        env.reset(seed=train_seed + rank)[0] for rank, env in enumerate(envs)
    ]
    template = envs[0]
    model = MachineMAPPO(
        local_feature_dim=template.LOCAL_FEATURE_DIM,
        global_state_dim=template.global_state_dim,
        actor_hidden_dim=settings.actor_hidden_dim,
        critic_hidden_dim=settings.critic_hidden_dim,
    ).to(settings.device)
    optimizer = th.optim.Adam(model.parameters(), lr=settings.learning_rate)
    episode_returns = np.zeros(settings.n_envs, dtype=np.float64)
    episode_lengths = np.zeros(settings.n_envs, dtype=np.int64)
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_targets = [
        math.ceil(settings.total_timesteps * fraction / 5)
        for fraction in range(1, 6)
    ]
    next_checkpoint = 0
    completed_steps = 0
    update_index = 0
    progress = tqdm(
        total=settings.total_timesteps,
        desc=f"CTDE-MAPPO seed {train_seed}",
        unit="joint-step",
        disable=not show_progress,
    )
    started = perf_counter()
    while completed_steps < settings.total_timesteps:
        rollout: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "local",
                "masks",
                "global",
                "actions",
                "log_probs",
                "values",
                "rewards",
                "dones",
            )
        }
        rollout_steps = min(
            settings.n_steps,
            math.ceil((settings.total_timesteps - completed_steps) / settings.n_envs),
        )
        for _ in range(rollout_steps):
            batch = _stack_observations(observations)
            actions, log_probs, values, _ = model.act(
                batch, deterministic=False, device=settings.device
            )
            rewards = np.zeros(settings.n_envs, dtype=np.float32)
            dones = np.zeros(settings.n_envs, dtype=np.float32)
            next_observations: list[dict[str, np.ndarray]] = []
            for rank, env in enumerate(envs):
                next_observation, reward, terminated, truncated, _ = env.step(
                    actions[rank]
                )
                done = terminated or truncated
                rewards[rank] = reward
                dones[rank] = float(done)
                episode_returns[rank] += reward
                episode_lengths[rank] += 1
                if done:
                    episode_rows.append(
                        {
                            "train_seed": train_seed,
                            "environment": rank,
                            "episode": episode_counts[rank],
                            "episode_return": episode_returns[rank],
                            "joint_steps": episode_lengths[rank],
                        }
                    )
                    episode_counts[rank] += 1
                    episode_returns[rank] = 0.0
                    episode_lengths[rank] = 0
                    next_seed = (
                        train_seed
                        + rank
                        + episode_counts[rank] * settings.n_envs
                    )
                    next_observation, _ = env.reset(seed=next_seed)
                next_observations.append(next_observation)
            rollout["local"].append(batch["local_observations"])
            rollout["masks"].append(batch["action_masks"])
            rollout["global"].append(batch["global_state"])
            rollout["actions"].append(actions)
            rollout["log_probs"].append(log_probs)
            rollout["values"].append(values)
            rollout["rewards"].append(rewards)
            rollout["dones"].append(dones)
            observations = next_observations
            increment = min(
                settings.n_envs, settings.total_timesteps - completed_steps
            )
            completed_steps += increment
            progress.update(increment)
            if completed_steps >= settings.total_timesteps:
                break

        final_batch = _stack_observations(observations)
        with th.no_grad():
            final_values = model.value(
                th.as_tensor(
                    final_batch["global_state"],
                    dtype=th.float32,
                    device=settings.device,
                )
            ).cpu().numpy()
        rewards_array = np.asarray(rollout["rewards"], dtype=np.float32)
        dones_array = np.asarray(rollout["dones"], dtype=np.float32)
        values_array = np.asarray(rollout["values"], dtype=np.float32)
        advantages = np.zeros_like(rewards_array)
        last_advantage = np.zeros(settings.n_envs, dtype=np.float32)
        next_values = final_values
        for step in reversed(range(len(rewards_array))):
            not_done = 1.0 - dones_array[step]
            delta = (
                rewards_array[step]
                + settings.gamma * next_values * not_done
                - values_array[step]
            )
            last_advantage = (
                delta
                + settings.gamma
                * settings.gae_lambda
                * not_done
                * last_advantage
            )
            advantages[step] = last_advantage
            next_values = values_array[step]
        returns = advantages + values_array

        flat = {
            "local": np.concatenate(rollout["local"], axis=0),
            "masks": np.concatenate(rollout["masks"], axis=0),
            "global": np.concatenate(rollout["global"], axis=0),
            "actions": np.concatenate(rollout["actions"], axis=0),
            "log_probs": np.concatenate(rollout["log_probs"], axis=0),
            "advantages": advantages.reshape(-1),
            "returns": returns.reshape(-1),
        }
        sample_count = len(flat["advantages"])
        advantage_mean = float(flat["advantages"].mean())
        advantage_std = float(flat["advantages"].std())
        flat["advantages"] = (
            flat["advantages"] - advantage_mean
        ) / max(advantage_std, 1e-8)
        losses: list[tuple[float, float, float, float, float]] = []
        for _ in range(settings.n_epochs):
            permutation = rng.permutation(sample_count)
            for start in range(0, sample_count, settings.batch_size):
                indices = permutation[start : start + settings.batch_size]
                local = th.as_tensor(
                    flat["local"][indices], dtype=th.float32, device=settings.device
                )
                masks = th.as_tensor(
                    flat["masks"][indices], dtype=th.bool, device=settings.device
                )
                global_state = th.as_tensor(
                    flat["global"][indices],
                    dtype=th.float32,
                    device=settings.device,
                )
                actions_tensor = th.as_tensor(
                    flat["actions"][indices], dtype=th.long, device=settings.device
                )
                old_log_prob = th.as_tensor(
                    flat["log_probs"][indices],
                    dtype=th.float32,
                    device=settings.device,
                )
                advantage = th.as_tensor(
                    flat["advantages"][indices],
                    dtype=th.float32,
                    device=settings.device,
                )
                return_tensor = th.as_tensor(
                    flat["returns"][indices],
                    dtype=th.float32,
                    device=settings.device,
                )
                distribution = model.distribution(local, masks)
                new_log_prob = distribution.log_prob(actions_tensor)
                entropy = distribution.entropy().mean()
                ratio = th.exp(new_log_prob - old_log_prob)
                agent_advantage = advantage.unsqueeze(-1)
                unclipped = ratio * agent_advantage
                clipped = th.clamp(
                    ratio, 1.0 - settings.clip_range, 1.0 + settings.clip_range
                ) * agent_advantage
                policy_loss = -th.minimum(unclipped, clipped).mean()
                values = model.value(global_state)
                value_loss = nn.functional.mse_loss(values, return_tensor)
                loss = (
                    policy_loss
                    + settings.value_coefficient * value_loss
                    - settings.entropy_coefficient * entropy
                )
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
                optimizer.step()
                log_ratio = new_log_prob - old_log_prob
                approximate_kl = float(
                    ((th.exp(log_ratio) - 1.0) - log_ratio).mean().item()
                )
                clip_fraction = float(
                    ((ratio - 1.0).abs() > settings.clip_range)
                    .float()
                    .mean()
                    .item()
                )
                losses.append(
                    (
                        float(policy_loss.item()),
                        float(value_loss.item()),
                        float(entropy.item()),
                        approximate_kl,
                        clip_fraction,
                    )
                )
        update_index += 1
        update_rows.append(
            {
                "update": update_index,
                "total_timesteps": completed_steps,
                "policy_loss": np.mean([item[0] for item in losses]),
                "value_loss": np.mean([item[1] for item in losses]),
                "entropy": np.mean([item[2] for item in losses]),
                "approx_kl": np.mean([item[3] for item in losses]),
                "clip_fraction": np.mean([item[4] for item in losses]),
                "completed_episodes": len(episode_rows),
            }
        )
        _write_csv(update_rows, output_dir / "training_progress.csv")
        _write_csv(episode_rows, output_dir / "training_episodes.csv")
        while (
            next_checkpoint < len(checkpoint_targets)
            and completed_steps >= checkpoint_targets[next_checkpoint]
        ):
            model.save(
                checkpoint_dir
                / (
                    f"mappo_target_{checkpoint_targets[next_checkpoint]}"
                    f"_at_{completed_steps}_steps.pt"
                )
            )
            next_checkpoint += 1
    elapsed = perf_counter() - started
    progress.close()
    model.save(output_dir / "mappo.pt")
    for env in envs:
        env.close()
    return model, elapsed


def evaluate_mappo(
    model: MachineMAPPO,
    config: BenchmarkConfig,
    seeds: Iterable[int],
    *,
    condition: str,
    device: str,
    show_progress: bool,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    audit: dict[str, int] = {}
    for seed in tqdm(
        tuple(seeds),
        desc=f"MAPPO {condition}",
        unit="episode",
        disable=not show_progress,
    ):
        env = MachineAgentsCTDEEnv(config)
        observation, _ = env.reset(seed=seed)
        episode_return = 0.0
        while not env._done:
            actions, _, _, _ = model.act(
                observation, deterministic=True, device=device
            )
            observation, reward, _, _, _ = env.step(actions)
            episode_return += reward
        result = env.result(condition)
        if not np.isclose(episode_return, -float(result.metrics["objective"])):
            raise AssertionError("MAPPO reward/objective identity failed")
        rows.append(
            {
                "split": "ctde_validation",
                "policy": condition,
                "condition": condition,
                "seed": seed,
                "episode_return": episode_return,
                **result.metrics,
            }
        )
        for key, value in env.coordination_totals.items():
            audit[key] = audit.get(key, 0) + int(value)
        env.close()
    return rows, audit
