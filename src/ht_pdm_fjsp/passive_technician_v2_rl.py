"""PPO policies and multi-agent adapter for passive-technician simulator v2."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_v2 import (
    MachineMode,
    PassiveTechnicianV2Config,
    PassiveTechnicianV2Env,
)


ALGORITHMS = ("ps_ippo", "mappo", "centralized_ppo")


@dataclass(frozen=True)
class PassivePPOSettings:
    total_timesteps: int
    n_envs: int
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    max_grad_norm: float = 0.5
    actor_hidden_dim: int = 64
    critic_hidden_dim: int = 128
    reward_scale: float = 0.05
    device: str = "cpu"


def selected_learning_config() -> PassiveTechnicianV2Config:
    return PassiveTechnicianV2Config(
        machines=4,
        technicians=2,
        horizon=36,
        failure_age=6,
        failure_probability=0.30,
        max_age=10,
        service_time=((2, 5), (2, 3), (3, 2), (5, 2)),
        eligibility=((True, False), (True, True), (True, True), (False, True)),
        initial_ages=(0, 0, 0, 0),
    )


class PassiveV2MultiAgentEnv:
    """Array adapter with local observations, global state, and action masks."""

    def __init__(self, config: PassiveTechnicianV2Config, *, seed: int = 0) -> None:
        self.config = config
        self.env = PassiveTechnicianV2Env(config, seed=seed)
        self.num_agents = config.machines
        self.action_count = config.action_count
        self.local_feature_dim = config.observation_dim
        self.global_state_dim = config.machines * config.observation_dim

    def _encode(self) -> dict[str, np.ndarray]:
        raw = np.asarray(self.env.observations(), dtype=np.float32)
        encoded = raw.copy()
        cfg = self.config
        max_service = max(max(row) for row in cfg.service_time)
        encoded[:, 0] /= cfg.horizon
        encoded[:, 1] /= cfg.horizon
        encoded[:, 2] /= cfg.max_age
        encoded[:, 9] /= cfg.technicians
        encoded[:, 10] /= cfg.technicians
        encoded[:, 11] /= max_service
        for technician in range(cfg.technicians):
            start = 12 + 7 * technician
            encoded[:, start + 1] /= max_service
            encoded[:, start + 3] /= max_service
            encoded[:, start + 4] /= cfg.machines
            encoded[:, start + 5] /= cfg.machines
        masks = np.asarray(self.env.action_masks(), dtype=np.bool_)
        return {
            "local_observations": encoded,
            "global_state": encoded.reshape(-1),
            "action_masks": masks,
        }

    def reset(self, *, seed: int) -> dict[str, np.ndarray]:
        self.env.seed = int(seed)
        self.env.reset()
        return self._encode()

    def step(
        self, actions: Iterable[int]
    ) -> tuple[dict[str, np.ndarray], float, bool, dict[str, object]]:
        _, reward, done, info = self.env.step(actions)
        return self._encode(), reward, done, info


def _mlp(input_dim: int, hidden_dim: int, output_dim: int) -> nn.Sequential:
    network = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )
    for layer in network:
        if isinstance(layer, nn.Linear):
            nn.init.orthogonal_(layer.weight, gain=math.sqrt(2.0))
            nn.init.zeros_(layer.bias)
    nn.init.orthogonal_(network[-1].weight, gain=0.01)
    return network


class PassivePPOPolicy(nn.Module):
    def __init__(
        self,
        *,
        algorithm: str,
        agent_count: int,
        action_count: int,
        local_feature_dim: int,
        global_state_dim: int,
        actor_hidden_dim: int = 64,
        critic_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if algorithm not in ALGORITHMS:
            raise ValueError(algorithm)
        self.algorithm = algorithm
        self.agent_count = agent_count
        self.action_count = action_count
        self.local_feature_dim = local_feature_dim
        self.global_state_dim = global_state_dim
        self.actor_hidden_dim = actor_hidden_dim
        self.critic_hidden_dim = critic_hidden_dim
        if algorithm == "centralized_ppo":
            self.actor = _mlp(
                global_state_dim, actor_hidden_dim, agent_count * action_count
            )
        else:
            self.actor = _mlp(local_feature_dim, actor_hidden_dim, action_count)
        critic_input = local_feature_dim if algorithm == "ps_ippo" else global_state_dim
        self.critic = _mlp(critic_input, critic_hidden_dim, 1)
        nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)

    def logits(
        self, local_observations: torch.Tensor, global_state: torch.Tensor
    ) -> torch.Tensor:
        if self.algorithm == "centralized_ppo":
            return self.actor(global_state).reshape(
                -1, self.agent_count, self.action_count
            )
        return self.actor(local_observations)

    def distribution(
        self,
        local_observations: torch.Tensor,
        global_state: torch.Tensor,
        action_masks: torch.Tensor,
    ) -> Categorical:
        logits = self.logits(local_observations, global_state)
        return Categorical(logits=logits.masked_fill(~action_masks.bool(), -1e9))

    def value(
        self, local_observations: torch.Tensor, global_state: torch.Tensor
    ) -> torch.Tensor:
        if self.algorithm == "ps_ippo":
            return self.critic(local_observations).squeeze(-1)
        return self.critic(global_state).squeeze(-1)

    def act_batch(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
        device: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        local = torch.as_tensor(
            observation["local_observations"], dtype=torch.float32, device=device
        )
        masks = torch.as_tensor(
            observation["action_masks"], dtype=torch.bool, device=device
        )
        global_state = torch.as_tensor(
            observation["global_state"], dtype=torch.float32, device=device
        )
        with torch.no_grad():
            distribution = self.distribution(local, global_state, masks)
            actions = (
                torch.argmax(distribution.logits, dim=-1)
                if deterministic
                else distribution.sample()
            )
            log_prob = distribution.log_prob(actions)
            entropy = distribution.entropy().mean(dim=-1)
            values = self.value(local, global_state)
        return tuple(
            tensor.detach().cpu().numpy()
            for tensor in (actions, log_prob, values, entropy)
        )  # type: ignore[return-value]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "algorithm": self.algorithm,
                "agent_count": self.agent_count,
                "action_count": self.action_count,
                "local_feature_dim": self.local_feature_dim,
                "global_state_dim": self.global_state_dim,
                "actor_hidden_dim": self.actor_hidden_dim,
                "critic_hidden_dim": self.critic_hidden_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str) -> "PassivePPOPolicy":
        payload = torch.load(path, map_location=device, weights_only=False)
        model = cls(
            algorithm=str(payload["algorithm"]),
            agent_count=int(payload["agent_count"]),
            action_count=int(payload["action_count"]),
            local_feature_dim=int(payload["local_feature_dim"]),
            global_state_dim=int(payload["global_state_dim"]),
            actor_hidden_dim=int(payload["actor_hidden_dim"]),
            critic_hidden_dim=int(payload["critic_hidden_dim"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def _stack(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def _write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_policy(
    config: PassiveTechnicianV2Config,
    settings: PassivePPOSettings,
    output_dir: Path,
    *,
    algorithm: str,
    train_seed: int,
    show_progress: bool,
) -> tuple[
    PassivePPOPolicy,
    dict[int, tuple[Path, int]],
    list[dict[str, object]],
    list[dict[str, object]],
    float,
]:
    if algorithm not in ALGORITHMS:
        raise ValueError(algorithm)
    if settings.total_timesteps % settings.n_envs:
        raise ValueError("total_timesteps must be divisible by n_envs")
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_seed)
    rng = np.random.default_rng(train_seed)
    envs = [PassiveV2MultiAgentEnv(config) for _ in range(settings.n_envs)]
    episode_counts = [0] * settings.n_envs

    def episode_seed(rank: int) -> int:
        return train_seed * 1_000_000 + episode_counts[rank] * settings.n_envs + rank

    observations = [env.reset(seed=episode_seed(rank)) for rank, env in enumerate(envs)]
    template = envs[0]
    model = PassivePPOPolicy(
        algorithm=algorithm,
        agent_count=template.num_agents,
        action_count=template.action_count,
        local_feature_dim=template.local_feature_dim,
        global_state_dim=template.global_state_dim,
        actor_hidden_dim=settings.actor_hidden_dim,
        critic_hidden_dim=settings.critic_hidden_dim,
    ).to(settings.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    episode_returns = np.zeros(settings.n_envs, dtype=np.float64)
    episode_lengths = np.zeros(settings.n_envs, dtype=np.int64)
    training_episodes: list[dict[str, object]] = []
    training_progress: list[dict[str, object]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    targets = [
        math.ceil(settings.total_timesteps * fraction / 5) for fraction in range(1, 6)
    ]
    checkpoint_paths: dict[int, tuple[Path, int]] = {}
    next_checkpoint = 0
    completed_steps = 0
    update_index = 0
    progress = tqdm(
        total=settings.total_timesteps,
        desc=f"{algorithm} seed {train_seed}",
        unit="joint-step",
        disable=not show_progress,
    )
    started = perf_counter()
    while completed_steps < settings.total_timesteps:
        rollout: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "local",
                "global",
                "masks",
                "actions",
                "log_probs",
                "values",
                "rewards",
                "dones",
            )
        }
        rollout_steps = min(
            settings.n_steps,
            (settings.total_timesteps - completed_steps) // settings.n_envs,
        )
        for _ in range(rollout_steps):
            batch = _stack(observations)
            actions, log_probs, values, _ = model.act_batch(
                batch, deterministic=False, device=settings.device
            )
            rewards = np.zeros(settings.n_envs, dtype=np.float32)
            dones = np.zeros(settings.n_envs, dtype=np.float32)
            next_observations: list[dict[str, np.ndarray]] = []
            for rank, env in enumerate(envs):
                next_observation, reward, done, _ = env.step(actions[rank])
                rewards[rank] = reward * settings.reward_scale
                dones[rank] = float(done)
                episode_returns[rank] += reward
                episode_lengths[rank] += 1
                if done:
                    training_episodes.append(
                        {
                            "algorithm": algorithm,
                            "train_seed": train_seed,
                            "environment": rank,
                            "episode": episode_counts[rank],
                            "episode_return": episode_returns[rank],
                            "objective": -episode_returns[rank],
                            "joint_steps": episode_lengths[rank],
                        }
                    )
                    episode_counts[rank] += 1
                    episode_returns[rank] = 0.0
                    episode_lengths[rank] = 0
                    next_observation = env.reset(seed=episode_seed(rank))
                next_observations.append(next_observation)
            rollout["local"].append(batch["local_observations"])
            rollout["global"].append(batch["global_state"])
            rollout["masks"].append(batch["action_masks"])
            rollout["actions"].append(actions)
            rollout["log_probs"].append(log_probs)
            rollout["values"].append(values)
            rollout["rewards"].append(rewards)
            rollout["dones"].append(dones)
            observations = next_observations
            completed_steps += settings.n_envs
            progress.update(settings.n_envs)

        final_batch = _stack(observations)
        with torch.no_grad():
            final_values = model.value(
                torch.as_tensor(
                    final_batch["local_observations"],
                    dtype=torch.float32,
                    device=settings.device,
                ),
                torch.as_tensor(
                    final_batch["global_state"],
                    dtype=torch.float32,
                    device=settings.device,
                ),
            ).cpu().numpy()
        rewards_array = np.asarray(rollout["rewards"], dtype=np.float32)
        dones_array = np.asarray(rollout["dones"], dtype=np.float32)
        values_array = np.asarray(rollout["values"], dtype=np.float32)
        if algorithm == "ps_ippo":
            rewards_array = np.repeat(
                rewards_array[..., None], config.machines, axis=-1
            )
            dones_for_gae = np.repeat(
                dones_array[..., None], config.machines, axis=-1
            )
        else:
            dones_for_gae = dones_array
        advantages = np.zeros_like(values_array)
        last_advantage = np.zeros_like(final_values)
        next_values = final_values
        for step in reversed(range(len(rewards_array))):
            not_done = 1.0 - dones_for_gae[step]
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
        flat_local = np.concatenate(rollout["local"], axis=0)
        flat_global = np.concatenate(rollout["global"], axis=0)
        flat_masks = np.concatenate(rollout["masks"], axis=0)
        flat_actions = np.concatenate(rollout["actions"], axis=0)
        flat_log_probs = np.concatenate(rollout["log_probs"], axis=0)
        flat_advantages = advantages.reshape((-1,) + advantages.shape[2:])
        flat_returns = returns.reshape((-1,) + returns.shape[2:])
        advantage_mean = float(flat_advantages.mean())
        advantage_std = float(flat_advantages.std())
        flat_advantages = (flat_advantages - advantage_mean) / max(
            advantage_std, 1e-8
        )
        sample_count = len(flat_local)
        losses: list[tuple[float, float, float, float, float]] = []
        for _ in range(settings.n_epochs):
            permutation = rng.permutation(sample_count)
            for start in range(0, sample_count, settings.batch_size):
                indices = permutation[start : start + settings.batch_size]
                local = torch.as_tensor(
                    flat_local[indices], dtype=torch.float32, device=settings.device
                )
                global_state = torch.as_tensor(
                    flat_global[indices], dtype=torch.float32, device=settings.device
                )
                masks = torch.as_tensor(
                    flat_masks[indices], dtype=torch.bool, device=settings.device
                )
                actions_tensor = torch.as_tensor(
                    flat_actions[indices], dtype=torch.long, device=settings.device
                )
                old_log_prob = torch.as_tensor(
                    flat_log_probs[indices], dtype=torch.float32, device=settings.device
                )
                advantage = torch.as_tensor(
                    flat_advantages[indices], dtype=torch.float32, device=settings.device
                )
                return_tensor = torch.as_tensor(
                    flat_returns[indices], dtype=torch.float32, device=settings.device
                )
                distribution = model.distribution(local, global_state, masks)
                new_log_prob = distribution.log_prob(actions_tensor)
                entropy = distribution.entropy().mean()
                if algorithm != "ps_ippo":
                    advantage = advantage.unsqueeze(-1)
                ratio = torch.exp(new_log_prob - old_log_prob)
                unclipped = ratio * advantage
                clipped = torch.clamp(
                    ratio, 1.0 - settings.clip_range, 1.0 + settings.clip_range
                ) * advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                predicted_values = model.value(local, global_state)
                value_loss = nn.functional.mse_loss(predicted_values, return_tensor)
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
                    ((torch.exp(log_ratio) - 1.0) - log_ratio).mean().item()
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
        training_progress.append(
            {
                "algorithm": algorithm,
                "train_seed": train_seed,
                "update": update_index,
                "total_timesteps": completed_steps,
                "policy_loss": float(np.mean([item[0] for item in losses])),
                "value_loss": float(np.mean([item[1] for item in losses])),
                "entropy": float(np.mean([item[2] for item in losses])),
                "approx_kl": float(np.mean([item[3] for item in losses])),
                "clip_fraction": float(np.mean([item[4] for item in losses])),
                "completed_episodes": len(training_episodes),
            }
        )
        _write_csv(training_progress, output_dir / "training_progress.csv")
        _write_csv(training_episodes, output_dir / "training_episodes.csv")
        while next_checkpoint < len(targets) and completed_steps >= targets[next_checkpoint]:
            target = targets[next_checkpoint]
            path = checkpoint_dir / f"target_{target}_at_{completed_steps}.pt"
            model.save(path)
            checkpoint_paths[target] = (path, completed_steps)
            next_checkpoint += 1
    elapsed = perf_counter() - started
    progress.close()
    model.save(output_dir / "final_model.pt")
    return model, checkpoint_paths, training_progress, training_episodes, elapsed


def reservation_aware_action(env: PassiveTechnicianV2Env) -> tuple[int, ...]:
    cfg = env.config
    masks = env.action_masks()
    actions = [0] * cfg.machines
    candidates = []
    for machine, raw_mode in enumerate(env.state.modes):
        mode = MachineMode(raw_mode)
        if mode == MachineMode.FAILED or (
            mode == MachineMode.OPERATING
            and env.state.ages[machine] >= cfg.failure_age - 2
        ):
            flexibility = sum(cfg.eligibility[machine])
            candidates.append((flexibility, -int(mode == MachineMode.FAILED), machine))
    candidates.sort()
    virtual_load = [
        env.technician_remaining(env.state, technician)
        + sum(
            cfg.service_time[machine][technician]
            for machine in env.state.queues[technician]
        )
        for technician in range(cfg.technicians)
    ]
    for _, _, machine in candidates:
        eligible = [
            technician
            for technician in range(cfg.technicians)
            if masks[machine][technician + 1]
        ]
        technician = min(
            eligible,
            key=lambda item: (
                virtual_load[item] + cfg.service_time[machine][item],
                cfg.service_time[machine][item],
                item,
            ),
        )
        actions[machine] = technician + 1
        virtual_load[technician] += cfg.service_time[machine][technician]
    return tuple(actions)


def evaluate_model(
    model: PassivePPOPolicy,
    config: PassiveTechnicianV2Config,
    seeds: Iterable[int],
    *,
    policy_name: str,
    split: str,
    train_seed: int,
    checkpoint_target_steps: int,
    checkpoint_actual_steps: int,
    device: str,
    show_progress: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in tqdm(
        tuple(seeds),
        desc=f"evaluate {policy_name}/{train_seed}/{checkpoint_target_steps}",
        unit="episode",
        leave=False,
        disable=not show_progress,
    ):
        adapter = PassiveV2MultiAgentEnv(config, seed=seed)
        observation = adapter.reset(seed=seed)
        episode_return = 0.0
        requests = 0
        slower_requests = 0
        while adapter.env.time < config.horizon:
            batch = {key: value[None, ...] for key, value in observation.items()}
            actions = model.act_batch(
                batch, deterministic=True, device=device
            )[0][0]
            if not all(
                observation["action_masks"][machine, action]
                for machine, action in enumerate(actions)
            ):
                raise AssertionError("learned policy selected a masked action")
            for machine, action in enumerate(actions):
                if action <= 0:
                    continue
                requests += 1
                fastest = min(
                    config.service_time[machine][technician]
                    for technician in range(config.technicians)
                    if config.eligibility[machine][technician]
                )
                slower_requests += int(
                    config.service_time[machine][int(action) - 1] > fastest
                )
            observation, reward, _, _ = adapter.step(actions)
            episode_return += reward
        identity_error = abs(episode_return + adapter.env.metrics["objective"])
        rows.append(
            {
                "split": split,
                "policy": policy_name,
                "train_seed": train_seed,
                "checkpoint_target_steps": checkpoint_target_steps,
                "checkpoint_actual_steps": checkpoint_actual_steps,
                "seed": seed,
                **adapter.env.metrics,
                "episode_return": episode_return,
                "identity_error": identity_error,
                "slower_request_fraction": slower_requests / requests if requests else 0.0,
                "technician_0_busy_steps": adapter.env.technician_busy_steps[0],
                "technician_1_busy_steps": adapter.env.technician_busy_steps[1],
                "technician_0_starts": adapter.env.technician_starts[0],
                "technician_1_starts": adapter.env.technician_starts[1],
            }
        )
    return rows


def policy_metadata(settings: PassivePPOSettings) -> dict[str, object]:
    return asdict(settings)
