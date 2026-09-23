"""Masked cooperative IQL and QMIX baselines for machine agents."""

from __future__ import annotations

import copy
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch as th
from torch import nn
from tqdm.auto import tqdm

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig


@dataclass(frozen=True)
class ValueLearningSettings:
    total_timesteps: int
    replay_capacity: int
    learning_starts: int
    batch_size: int
    train_frequency: int
    gradient_steps: int
    learning_rate: float
    gamma: float
    target_update_interval: int
    epsilon_start: float
    epsilon_end: float
    epsilon_fraction: float
    hidden_dim: int
    mixer_hidden_dim: int
    device: str
    n_envs: int = 1


def _mlp(input_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class QMixer(nn.Module):
    def __init__(self, agent_count: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.agent_count = agent_count
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.hyper_w1 = nn.Linear(state_dim, agent_count * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_q: th.Tensor, state: th.Tensor) -> th.Tensor:
        batch = agent_q.shape[0]
        w1 = self.hyper_w1(state).abs().view(
            batch, self.agent_count, self.hidden_dim
        )
        b1 = self.hyper_b1(state).view(batch, 1, self.hidden_dim)
        hidden = th.nn.functional.elu(th.bmm(agent_q.unsqueeze(1), w1) + b1)
        w2 = self.hyper_w2(state).abs().view(batch, self.hidden_dim, 1)
        return (th.bmm(hidden, w2).squeeze((1, 2)) + self.value(state).squeeze(-1))


class ValueDecompositionPolicy(nn.Module):
    VALID_ALGORITHMS = {"iql", "qmix"}

    def __init__(
        self,
        *,
        algorithm: str,
        agent_count: int,
        local_feature_dim: int,
        global_state_dim: int,
        hidden_dim: int = 128,
        mixer_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if algorithm not in self.VALID_ALGORITHMS:
            raise ValueError(f"Unknown value-decomposition algorithm: {algorithm}")
        self.algorithm = algorithm
        self.agent_count = agent_count
        self.local_feature_dim = local_feature_dim
        self.global_state_dim = global_state_dim
        self.hidden_dim = hidden_dim
        self.mixer_hidden_dim = mixer_hidden_dim
        network_count = agent_count if algorithm == "iql" else 1
        self.agent_networks = nn.ModuleList(
            [_mlp(local_feature_dim, hidden_dim) for _ in range(network_count)]
        )
        self.mixer = (
            QMixer(agent_count, global_state_dim, mixer_hidden_dim)
            if algorithm == "qmix"
            else None
        )

    def q_values(self, local_observations: th.Tensor) -> th.Tensor:
        if self.algorithm == "qmix":
            return self.agent_networks[0](local_observations).squeeze(-1)
        return th.stack(
            [
                network(local_observations[:, index]).squeeze(-1)
                for index, network in enumerate(self.agent_networks)
            ],
            dim=1,
        )

    def act(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
        device: str,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        local = th.as_tensor(
            observation["local_observations"], dtype=th.float32, device=device
        )
        masks = th.as_tensor(
            observation["action_masks"], dtype=th.bool, device=device
        )
        single = local.ndim == 3
        if single:
            local = local.unsqueeze(0)
            masks = masks.unsqueeze(0)
        with th.no_grad():
            values = self.q_values(local)
            greedy = values.masked_fill(~masks, -1e9).argmax(dim=-1).cpu().numpy()
        if deterministic or epsilon <= 0:
            return greedy.squeeze(0) if single else greedy
        generator = rng or np.random.default_rng()
        actions = greedy.copy()
        mask_array = np.asarray(observation["action_masks"])
        if single:
            mask_array = mask_array[None, ...]
        for env_index in range(actions.shape[0]):
            for agent in range(self.agent_count):
                if generator.random() < epsilon:
                    valid = np.flatnonzero(mask_array[env_index, agent])
                    actions[env_index, agent] = int(generator.choice(valid))
        return actions.squeeze(0) if single else actions

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(
            {
                "state_dict": self.state_dict(),
                "algorithm": self.algorithm,
                "agent_count": self.agent_count,
                "local_feature_dim": self.local_feature_dim,
                "global_state_dim": self.global_state_dim,
                "hidden_dim": self.hidden_dim,
                "mixer_hidden_dim": self.mixer_hidden_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str) -> "ValueDecompositionPolicy":
        payload = th.load(path, map_location=device, weights_only=False)
        model = cls(
            **{
                key: payload[key]
                for key in (
                    "algorithm",
                    "agent_count",
                    "local_feature_dim",
                    "global_state_dim",
                    "hidden_dim",
                    "mixer_hidden_dim",
                )
            }
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class ReplayBuffer:
    def __init__(self, capacity: int, observation: dict[str, np.ndarray]) -> None:
        self.capacity = capacity
        local_shape = observation["local_observations"].shape
        mask_shape = observation["action_masks"].shape
        global_shape = observation["global_state"].shape
        self.local = np.empty((capacity, *local_shape), dtype=np.float16)
        self.masks = np.empty((capacity, *mask_shape), dtype=np.bool_)
        self.global_state = np.empty((capacity, *global_shape), dtype=np.float16)
        self.actions = np.empty((capacity, mask_shape[0]), dtype=np.int16)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_local = np.empty((capacity, *local_shape), dtype=np.float16)
        self.next_masks = np.empty((capacity, *mask_shape), dtype=np.bool_)
        self.next_global = np.empty((capacity, *global_shape), dtype=np.float16)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: dict[str, np.ndarray],
        actions: np.ndarray,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        index = self.position
        self.local[index] = observation["local_observations"]
        self.masks[index] = observation["action_masks"]
        self.global_state[index] = observation["global_state"]
        self.actions[index] = actions
        self.rewards[index] = reward
        self.next_local[index] = next_observation["local_observations"]
        self.next_masks[index] = next_observation["action_masks"]
        self.next_global[index] = next_observation["global_state"]
        self.dones[index] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "local": self.local[indices],
            "masks": self.masks[indices],
            "global": self.global_state[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_local": self.next_local[indices],
            "next_masks": self.next_masks[indices],
            "next_global": self.next_global[indices],
            "dones": self.dones[indices],
        }


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _stack_observations(
    observations: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def train_value_policy(
    config: BenchmarkConfig,
    settings: ValueLearningSettings,
    output_dir: Path,
    *,
    algorithm: str,
    train_seed: int,
    show_progress: bool,
) -> tuple[ValueDecompositionPolicy, float]:
    if settings.n_envs < 1:
        raise ValueError("n_envs must be positive")
    if settings.total_timesteps % settings.n_envs:
        raise ValueError("total_timesteps must be divisible by n_envs")
    np.random.seed(train_seed)
    th.manual_seed(train_seed)
    rng = np.random.default_rng(train_seed)
    envs = [
        MachineAgentsCTDEEnv(config, wait_policy="safe_noop")
        for _ in range(settings.n_envs)
    ]
    observations = [
        env.reset(seed=train_seed + rank)[0] for rank, env in enumerate(envs)
    ]
    env = envs[0]
    model = ValueDecompositionPolicy(
        algorithm=algorithm,
        agent_count=env.num_agents,
        local_feature_dim=env.local_feature_dim,
        global_state_dim=env.global_state_dim,
        hidden_dim=settings.hidden_dim,
        mixer_hidden_dim=settings.mixer_hidden_dim,
    ).to(settings.device)
    target = copy.deepcopy(model).to(settings.device).eval()
    optimizer = th.optim.Adam(model.parameters(), lr=settings.learning_rate)
    buffer = ReplayBuffer(settings.replay_capacity, observations[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    targets = [math.ceil(settings.total_timesteps * part / 5) for part in range(1, 6)]
    next_checkpoint = 0
    episode_counts = [0] * settings.n_envs
    episode_returns = [0.0] * settings.n_envs
    episode_lengths = [0] * settings.n_envs
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    started = perf_counter()
    progress = tqdm(
        total=settings.total_timesteps,
        desc=f"{algorithm.upper()} seed {train_seed}",
        unit="joint-step",
        disable=not show_progress,
    )
    completed_steps = 0
    next_update = max(
        settings.train_frequency,
        math.ceil(settings.learning_starts / settings.train_frequency)
        * settings.train_frequency,
    )
    next_target_update = settings.target_update_interval
    next_log = 1_000
    pending_losses: list[float] = []
    while completed_steps < settings.total_timesteps:
        fraction = min(
            1.0,
            completed_steps
            / max(1, settings.epsilon_fraction * settings.total_timesteps),
        )
        epsilon = settings.epsilon_start + fraction * (
            settings.epsilon_end - settings.epsilon_start
        )
        batched_observation = _stack_observations(observations)
        actions = model.act(
            batched_observation,
            deterministic=False,
            device=settings.device,
            epsilon=epsilon,
            rng=rng,
        )
        for rank, current_env in enumerate(envs):
            next_observation, reward, terminated, truncated, _ = current_env.step(
                actions[rank]
            )
            done = terminated or truncated
            buffer.add(
                observations[rank], actions[rank], reward, next_observation, done
            )
            episode_returns[rank] += reward
            episode_lengths[rank] += 1
            observations[rank] = next_observation
            if done:
                episode_rows.append(
                    {
                        "train_seed": train_seed,
                        "environment_rank": rank,
                        "episode": episode_counts[rank],
                        "episode_return": episode_returns[rank],
                        "joint_steps": episode_lengths[rank],
                    }
                )
                episode_counts[rank] += 1
                episode_returns[rank] = 0.0
                episode_lengths[rank] = 0
                reset_seed = (
                    train_seed
                    + rank
                    + episode_counts[rank] * settings.n_envs
                )
                observations[rank], _ = current_env.reset(seed=reset_seed)
        completed_steps += settings.n_envs
        progress.update(settings.n_envs)

        while (
            next_update <= completed_steps
            and buffer.size >= settings.batch_size
        ):
            for _ in range(settings.gradient_steps):
                batch = buffer.sample(settings.batch_size, rng)
                local = th.as_tensor(batch["local"], dtype=th.float32, device=settings.device)
                actions_tensor = th.as_tensor(batch["actions"], dtype=th.long, device=settings.device)
                rewards = th.as_tensor(batch["rewards"], dtype=th.float32, device=settings.device)
                next_local = th.as_tensor(batch["next_local"], dtype=th.float32, device=settings.device)
                next_masks = th.as_tensor(batch["next_masks"], dtype=th.bool, device=settings.device)
                dones = th.as_tensor(batch["dones"], dtype=th.float32, device=settings.device)
                state = th.as_tensor(batch["global"], dtype=th.float32, device=settings.device)
                next_state = th.as_tensor(batch["next_global"], dtype=th.float32, device=settings.device)
                chosen = model.q_values(local).gather(-1, actions_tensor.unsqueeze(-1)).squeeze(-1)
                with th.no_grad():
                    next_q = target.q_values(next_local).masked_fill(~next_masks, -1e9).max(dim=-1).values
                if algorithm == "qmix":
                    assert model.mixer is not None and target.mixer is not None
                    predicted = model.mixer(chosen, state)
                    with th.no_grad():
                        target_value = rewards + settings.gamma * (1.0 - dones) * target.mixer(next_q, next_state)
                else:
                    predicted = chosen
                    target_value = rewards.unsqueeze(-1) + settings.gamma * (1.0 - dones).unsqueeze(-1) * next_q
                loss = nn.functional.smooth_l1_loss(predicted, target_value)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                pending_losses.append(float(loss.item()))
            next_update += settings.train_frequency
        while next_update <= completed_steps:
            next_update += settings.train_frequency
        while completed_steps >= next_target_update:
            target.load_state_dict(model.state_dict())
            next_target_update += settings.target_update_interval
        if pending_losses and (
            completed_steps >= next_log
            or completed_steps == settings.total_timesteps
        ):
            update_rows.append(
                {
                    "step": completed_steps,
                    "loss": float(np.mean(pending_losses)),
                    "epsilon": epsilon,
                    "replay_size": buffer.size,
                    "completed_episodes": sum(episode_counts),
                }
            )
            pending_losses.clear()
            while next_log <= completed_steps:
                next_log += 1_000
            _write_csv(update_rows, output_dir / "training_progress.csv")
            _write_csv(episode_rows, output_dir / "training_episodes.csv")
        while (
            next_checkpoint < len(targets)
            and completed_steps >= targets[next_checkpoint]
        ):
            model.save(checkpoints / f"model_{targets[next_checkpoint]}_steps.pt")
            next_checkpoint += 1
    progress.close()
    elapsed = perf_counter() - started
    model.save(output_dir / "model.pt")
    (output_dir / "settings.json").write_text(
        json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n"
    )
    _write_csv(episode_rows, output_dir / "training_episodes.csv")
    _write_csv(update_rows, output_dir / "training_progress.csv")
    for current_env in envs:
        current_env.close()
    return model, elapsed
