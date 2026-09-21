"""Matched-budget MARL diagnostic policies and PPO training loop."""

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
class DiagnosticSettings:
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


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def _initialize(module: nn.Sequential, final_gain: float) -> None:
    for layer in module:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(module[-1].weight, gain=final_gain)


class DiagnosticPolicy(nn.Module):
    """Actor/critic variants that change one MARL assumption at a time."""

    VALID_ACTORS = {"shared", "independent"}
    VALID_CRITICS = {"global", "local"}

    def __init__(
        self,
        *,
        agent_count: int,
        max_local_actions: int,
        local_feature_dim: int,
        global_state_dim: int,
        actor_mode: str,
        critic_mode: str,
        include_broadcast_context: bool,
        actor_hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if actor_mode not in self.VALID_ACTORS:
            raise ValueError(f"Unknown actor mode: {actor_mode}")
        if critic_mode not in self.VALID_CRITICS:
            raise ValueError(f"Unknown critic mode: {critic_mode}")
        self.agent_count = agent_count
        self.max_local_actions = max_local_actions
        self.local_feature_dim = local_feature_dim
        self.global_state_dim = global_state_dim
        self.actor_mode = actor_mode
        self.critic_mode = critic_mode
        self.include_broadcast_context = include_broadcast_context
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        actor_count = 1 if actor_mode == "shared" else agent_count
        self.actors = nn.ModuleList(
            [_mlp(local_feature_dim, actor_hidden_dim, 1) for _ in range(actor_count)]
        )
        if critic_mode == "global":
            critic_input_dim = global_state_dim
            self.critic = _mlp(critic_input_dim, critic_hidden_dim, 1)
        else:
            # Consume the same constructor RNG as the completed MAPPO model
            # before initializing a shared base actor. This keeps the PS-IPPO
            # actor initialization matched to MAPPO for the same training seed.
            _mlp(global_state_dim, critic_hidden_dim, 1)
            for actor in self.actors:
                _initialize(actor, 0.01)
            critic_input_dim = max_local_actions * (local_feature_dim + 1)
            self.critic = _mlp(critic_input_dim, critic_hidden_dim, 1)
        if critic_mode == "global":
            for actor in self.actors:
                _initialize(actor, 0.01)
        _initialize(self.critic, 1.0)

    def actor_logits(self, local_observations: th.Tensor) -> th.Tensor:
        if self.actor_mode == "shared":
            return self.actors[0](local_observations).squeeze(-1)
        return th.stack(
            [
                actor(local_observations[:, agent]).squeeze(-1)
                for agent, actor in enumerate(self.actors)
            ],
            dim=1,
        )

    def distribution(
        self, local_observations: th.Tensor, action_masks: th.Tensor
    ) -> Categorical:
        logits = self.actor_logits(local_observations)
        return Categorical(logits=logits.masked_fill(~action_masks.bool(), -1e9))

    def values(
        self,
        local_observations: th.Tensor,
        action_masks: th.Tensor,
        global_state: th.Tensor,
    ) -> th.Tensor:
        if self.critic_mode == "global":
            team_value = self.critic(global_state).squeeze(-1)
            return team_value.unsqueeze(-1).expand(-1, self.agent_count)
        critic_input = th.cat(
            (local_observations, action_masks.unsqueeze(-1).float()), dim=-1
        ).flatten(start_dim=2)
        return self.critic(critic_input).squeeze(-1)

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
        single = local.ndim == 3
        if single:
            local = local.unsqueeze(0)
            masks = masks.unsqueeze(0)
            global_state = global_state.unsqueeze(0)
        with th.no_grad():
            distribution = self.distribution(local, masks)
            actions = (
                th.argmax(distribution.logits, dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_prob = distribution.log_prob(actions)
            values = self.values(local, masks, global_state)
            entropy = distribution.entropy()
        outputs = (actions, log_prob, values, entropy)
        if single:
            outputs = tuple(item.squeeze(0) for item in outputs)
        return tuple(item.cpu().numpy() for item in outputs)  # type: ignore[return-value]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(
            {
                "state_dict": self.state_dict(),
                "agent_count": self.agent_count,
                "max_local_actions": self.max_local_actions,
                "local_feature_dim": self.local_feature_dim,
                "global_state_dim": self.global_state_dim,
                "actor_mode": self.actor_mode,
                "critic_mode": self.critic_mode,
                "include_broadcast_context": self.include_broadcast_context,
                "actor_hidden_dim": self.actor_hidden_dim,
                "critic_hidden_dim": self.critic_hidden_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str) -> "DiagnosticPolicy":
        payload = th.load(path, map_location=device, weights_only=False)
        model = cls(
            **{
                key: payload[key]
                for key in (
                    "agent_count",
                    "max_local_actions",
                    "local_feature_dim",
                    "global_state_dim",
                    "actor_mode",
                    "critic_mode",
                    "include_broadcast_context",
                    "actor_hidden_dim",
                    "critic_hidden_dim",
                )
            }
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def _stack(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
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


def train_diagnostic_policy(
    config: BenchmarkConfig,
    settings: DiagnosticSettings,
    output_dir: Path,
    *,
    train_seed: int,
    actor_mode: str,
    critic_mode: str,
    include_broadcast_context: bool,
    show_progress: bool,
) -> tuple[DiagnosticPolicy, float]:
    """Train one matched-budget PPO MARL diagnostic cell."""

    if settings.n_steps * settings.n_envs % settings.batch_size:
        raise ValueError("n_steps * n_envs must be divisible by batch_size")
    np.random.seed(train_seed)
    th.manual_seed(train_seed)
    rng = np.random.default_rng(train_seed)
    envs = [
        MachineAgentsCTDEEnv(
            config, include_broadcast_context=include_broadcast_context
        )
        for _ in range(settings.n_envs)
    ]
    observations = [
        env.reset(seed=train_seed + rank)[0] for rank, env in enumerate(envs)
    ]
    template = envs[0]
    model = DiagnosticPolicy(
        agent_count=template.num_agents,
        max_local_actions=template.max_local_actions,
        local_feature_dim=template.local_feature_dim,
        global_state_dim=template.global_state_dim,
        actor_mode=actor_mode,
        critic_mode=critic_mode,
        include_broadcast_context=include_broadcast_context,
        actor_hidden_dim=settings.actor_hidden_dim,
        critic_hidden_dim=settings.critic_hidden_dim,
    ).to(settings.device)
    optimizer = th.optim.Adam(model.parameters(), lr=settings.learning_rate)
    episode_returns = np.zeros(settings.n_envs, dtype=np.float64)
    episode_lengths = np.zeros(settings.n_envs, dtype=np.int64)
    episode_counts = np.zeros(settings.n_envs, dtype=np.int64)
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
        desc=f"{actor_mode}/{critic_mode} seed {train_seed}",
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
            batch = _stack(observations)
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
                            "episode": int(episode_counts[rank]),
                            "episode_return": episode_returns[rank],
                            "joint_steps": episode_lengths[rank],
                        }
                    )
                    episode_counts[rank] += 1
                    episode_returns[rank] = 0.0
                    episode_lengths[rank] = 0
                    next_seed = train_seed + rank + int(
                        episode_counts[rank]
                    ) * settings.n_envs
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
            increment = min(settings.n_envs, settings.total_timesteps - completed_steps)
            completed_steps += increment
            progress.update(increment)
            if completed_steps >= settings.total_timesteps:
                break

        final = _stack(observations)
        with th.no_grad():
            final_values = model.values(
                th.as_tensor(
                    final["local_observations"],
                    dtype=th.float32,
                    device=settings.device,
                ),
                th.as_tensor(
                    final["action_masks"], dtype=th.bool, device=settings.device
                ),
                th.as_tensor(
                    final["global_state"], dtype=th.float32, device=settings.device
                ),
            ).cpu().numpy()
        rewards_array = np.asarray(rollout["rewards"], dtype=np.float32)
        dones_array = np.asarray(rollout["dones"], dtype=np.float32)
        values_array = np.asarray(rollout["values"], dtype=np.float32)
        advantages = np.zeros_like(values_array)
        last_advantage = np.zeros_like(final_values)
        next_values = final_values
        for step in reversed(range(len(rewards_array))):
            not_done = (1.0 - dones_array[step]).reshape(-1, 1)
            reward = rewards_array[step].reshape(-1, 1)
            delta = reward + settings.gamma * next_values * not_done - values_array[step]
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
            "advantages": advantages.reshape(-1, template.num_agents),
            "returns": returns.reshape(-1, template.num_agents),
        }
        flat["advantages"] = (
            flat["advantages"] - flat["advantages"].mean()
        ) / max(float(flat["advantages"].std()), 1e-8)
        sample_count = len(flat["advantages"])
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
                    flat["global"][indices], dtype=th.float32, device=settings.device
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
                    flat["returns"][indices], dtype=th.float32, device=settings.device
                )
                distribution = model.distribution(local, masks)
                new_log_prob = distribution.log_prob(actions_tensor)
                ratio = th.exp(new_log_prob - old_log_prob)
                unclipped = ratio * advantage
                clipped = th.clamp(
                    ratio, 1.0 - settings.clip_range, 1.0 + settings.clip_range
                ) * advantage
                policy_loss = -th.minimum(unclipped, clipped).mean()
                predicted_values = model.values(local, masks, global_state)
                value_loss = nn.functional.mse_loss(predicted_values, return_tensor)
                entropy = distribution.entropy().mean()
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
                losses.append(
                    (
                        float(policy_loss.item()),
                        float(value_loss.item()),
                        float(entropy.item()),
                        float(
                            ((th.exp(log_ratio) - 1.0) - log_ratio).mean().item()
                        ),
                        float(
                            ((ratio - 1.0).abs() > settings.clip_range)
                            .float()
                            .mean()
                            .item()
                        ),
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
                / f"policy_target_{checkpoint_targets[next_checkpoint]}_at_{completed_steps}_steps.pt"
            )
            next_checkpoint += 1
    elapsed = perf_counter() - started
    progress.close()
    model.save(output_dir / "policy.pt")
    for env in envs:
        env.close()
    return model, elapsed


def evaluate_diagnostic_policy(
    model: DiagnosticPolicy,
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
        desc=f"Evaluate {condition}",
        unit="episode",
        disable=not show_progress,
    ):
        env = MachineAgentsCTDEEnv(
            config,
            include_broadcast_context=model.include_broadcast_context,
        )
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
            raise AssertionError("Diagnostic reward/objective identity failed")
        rows.append(
            {
                "split": "marl_diagnostic_validation",
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
