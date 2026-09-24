"""Centralized joint-action DQN baseline for small cooperative action spaces."""

from __future__ import annotations

import copy
import csv
import itertools
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import torch as th
from torch import nn
from tqdm.auto import tqdm

from ht_pdm_fjsp.value_decomposition import ValueLearningSettings


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_training_episodes(rows: list[dict[str, Any]], path: Path) -> None:
    if rows:
        _write_csv(rows, path)
        return
    path.write_text(
        "train_seed,environment_rank,episode,episode_return,joint_steps\n",
        encoding="utf-8",
    )


class CentralizedJointDQNPolicy(nn.Module):
    def __init__(
        self,
        *,
        agent_count: int,
        action_count: int,
        global_state_dim: int,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.agent_count = agent_count
        self.action_count = action_count
        self.global_state_dim = global_state_dim
        self.hidden_dim = hidden_dim
        self.joint_actions = np.asarray(
            list(itertools.product(range(action_count), repeat=agent_count)),
            dtype=np.int64,
        )
        self.joint_action_count = len(self.joint_actions)
        self.network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.joint_action_count),
        )

    def valid_joint_masks(self, masks: np.ndarray) -> np.ndarray:
        mask_array = np.asarray(masks, dtype=np.bool_)
        single = mask_array.ndim == 2
        if single:
            mask_array = mask_array[None, ...]
        valid = np.ones(
            (mask_array.shape[0], self.joint_action_count), dtype=np.bool_
        )
        for agent in range(self.agent_count):
            valid &= mask_array[:, agent, self.joint_actions[:, agent]]
        if not valid.any(axis=1).all():
            raise AssertionError("Centralized policy has no feasible joint action")
        return valid[0] if single else valid

    def actions_to_indices(self, actions: np.ndarray) -> np.ndarray:
        action_array = np.asarray(actions, dtype=np.int64)
        single = action_array.ndim == 1
        if single:
            action_array = action_array[None, :]
        powers = self.action_count ** np.arange(
            self.agent_count - 1, -1, -1, dtype=np.int64
        )
        indices = (action_array * powers).sum(axis=1)
        return indices[0] if single else indices

    def act(
        self,
        observation: dict[str, np.ndarray],
        *,
        deterministic: bool,
        device: str,
        epsilon: float = 0.0,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        state_array = np.asarray(observation["global_state"], dtype=np.float32)
        single = state_array.ndim == 1
        if single:
            state_array = state_array[None, :]
        valid = self.valid_joint_masks(observation["action_masks"])
        if valid.ndim == 1:
            valid = valid[None, :]
        with th.no_grad():
            state = th.as_tensor(state_array, dtype=th.float32, device=device)
            values = self.network(state).cpu().numpy()
        values[~valid] = -np.inf
        indices = values.argmax(axis=1)
        if not deterministic and epsilon > 0:
            generator = rng or np.random.default_rng()
            for row in range(len(indices)):
                if generator.random() < epsilon:
                    indices[row] = int(generator.choice(np.flatnonzero(valid[row])))
        actions = self.joint_actions[indices]
        return actions[0] if single else actions

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(
            {
                "state_dict": self.state_dict(),
                "agent_count": self.agent_count,
                "action_count": self.action_count,
                "global_state_dim": self.global_state_dim,
                "hidden_dim": self.hidden_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str) -> "CentralizedJointDQNPolicy":
        payload = th.load(path, map_location=device, weights_only=False)
        model = cls(
            **{
                key: payload[key]
                for key in (
                    "agent_count",
                    "action_count",
                    "global_state_dim",
                    "hidden_dim",
                )
            }
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class _CentralReplayBuffer:
    def __init__(self, capacity: int, observation: dict[str, np.ndarray]) -> None:
        self.capacity = capacity
        state_shape = observation["global_state"].shape
        mask_shape = observation["action_masks"].shape
        self.state = np.empty((capacity, *state_shape), dtype=np.float16)
        self.masks = np.empty((capacity, *mask_shape), dtype=np.bool_)
        self.action = np.empty(capacity, dtype=np.int32)
        self.reward = np.empty(capacity, dtype=np.float32)
        self.next_state = np.empty((capacity, *state_shape), dtype=np.float16)
        self.next_masks = np.empty((capacity, *mask_shape), dtype=np.bool_)
        self.done = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def add(
        self,
        observation: dict[str, np.ndarray],
        action: int,
        reward: float,
        next_observation: dict[str, np.ndarray],
        done: bool,
    ) -> None:
        index = self.position
        self.state[index] = observation["global_state"]
        self.masks[index] = observation["action_masks"]
        self.action[index] = action
        self.reward[index] = reward
        self.next_state[index] = next_observation["global_state"]
        self.next_masks[index] = next_observation["action_masks"]
        self.done[index] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "state": self.state[indices],
            "action": self.action[indices],
            "reward": self.reward[indices],
            "next_state": self.next_state[indices],
            "next_masks": self.next_masks[indices],
            "done": self.done[indices],
        }


def _stack(observations: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def train_centralized_joint_dqn(
    settings: ValueLearningSettings,
    output_dir: Path,
    *,
    train_seed: int,
    show_progress: bool,
    checkpoint_targets: tuple[int, ...],
    env_factory: Callable[[], Any],
) -> tuple[CentralizedJointDQNPolicy, float]:
    if settings.total_timesteps % settings.n_envs:
        raise ValueError("total_timesteps must be divisible by n_envs")
    np.random.seed(train_seed)
    th.manual_seed(train_seed)
    rng = np.random.default_rng(train_seed)
    envs = [env_factory() for _ in range(settings.n_envs)]
    observations = [
        env.reset(seed=train_seed + rank)[0] for rank, env in enumerate(envs)
    ]
    env = envs[0]
    model = CentralizedJointDQNPolicy(
        agent_count=env.num_agents,
        action_count=env.action_count,
        global_state_dim=env.global_state_dim,
        hidden_dim=settings.hidden_dim,
    ).to(settings.device)
    target = copy.deepcopy(model).to(settings.device).eval()
    optimizer = th.optim.Adam(model.parameters(), lr=settings.learning_rate)
    buffer = _CentralReplayBuffer(settings.replay_capacity, observations[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    next_checkpoint = 0
    episode_counts = [0] * settings.n_envs
    episode_returns = [0.0] * settings.n_envs
    episode_lengths = [0] * settings.n_envs
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    pending = {key: [] for key in ("loss", "chosen_q_mean", "chosen_q_max", "predicted_value_mean", "gradient_norm")}
    completed_steps = 0
    next_update = max(
        settings.train_frequency,
        math.ceil(settings.learning_starts / settings.train_frequency)
        * settings.train_frequency,
    )
    next_target_update = settings.target_update_interval
    next_log = 1_000
    started = perf_counter()
    progress = tqdm(
        total=settings.total_timesteps,
        desc=f"CENTRALIZED_DQN seed {train_seed}",
        unit="joint-step",
        disable=not show_progress,
    )
    while completed_steps < settings.total_timesteps:
        fraction = min(
            1.0,
            completed_steps
            / max(1, settings.epsilon_fraction * settings.total_timesteps),
        )
        epsilon = settings.epsilon_start + fraction * (
            settings.epsilon_end - settings.epsilon_start
        )
        actions = model.act(
            _stack(observations),
            deterministic=False,
            device=settings.device,
            epsilon=epsilon,
            rng=rng,
        )
        indices = model.actions_to_indices(actions)
        for rank, current_env in enumerate(envs):
            next_observation, reward, terminated, truncated, _ = current_env.step(
                actions[rank]
            )
            done = terminated or truncated
            buffer.add(
                observations[rank],
                int(indices[rank]),
                reward,
                next_observation,
                done,
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
                reset_seed = train_seed + rank + episode_counts[rank] * settings.n_envs
                observations[rank], _ = current_env.reset(seed=reset_seed)
        completed_steps += settings.n_envs
        progress.update(settings.n_envs)

        while next_update <= completed_steps and buffer.size >= settings.batch_size:
            for _ in range(settings.gradient_steps):
                batch = buffer.sample(settings.batch_size, rng)
                state = th.as_tensor(batch["state"], dtype=th.float32, device=settings.device)
                action = th.as_tensor(batch["action"], dtype=th.long, device=settings.device)
                reward = th.as_tensor(batch["reward"], dtype=th.float32, device=settings.device)
                next_state = th.as_tensor(batch["next_state"], dtype=th.float32, device=settings.device)
                done = th.as_tensor(batch["done"], dtype=th.float32, device=settings.device)
                predicted = model.network(state).gather(1, action[:, None]).squeeze(1)
                next_valid = th.as_tensor(
                    model.valid_joint_masks(batch["next_masks"]),
                    dtype=th.bool,
                    device=settings.device,
                )
                with th.no_grad():
                    next_q = target.network(next_state).masked_fill(~next_valid, -1e9).max(dim=1).values
                    target_value = reward + settings.gamma * (1.0 - done) * next_q
                loss = nn.functional.smooth_l1_loss(predicted, target_value)
                optimizer.zero_grad()
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                pending["loss"].append(float(loss.item()))
                pending["chosen_q_mean"].append(float(predicted.mean().item()))
                pending["chosen_q_max"].append(float(predicted.max().item()))
                pending["predicted_value_mean"].append(float(predicted.mean().item()))
                pending["gradient_norm"].append(float(gradient_norm))
            next_update += settings.train_frequency
        while next_update <= completed_steps:
            next_update += settings.train_frequency
        while completed_steps >= next_target_update:
            target.load_state_dict(model.state_dict())
            next_target_update += settings.target_update_interval
        if pending["loss"] and (
            completed_steps >= next_log or completed_steps == settings.total_timesteps
        ):
            update_rows.append(
                {
                    "step": completed_steps,
                    **{key: float(np.mean(values)) for key, values in pending.items()},
                    "epsilon": epsilon,
                    "replay_size": buffer.size,
                    "completed_episodes": sum(episode_counts),
                }
            )
            for values in pending.values():
                values.clear()
            while next_log <= completed_steps:
                next_log += 1_000
            _write_csv(update_rows, output_dir / "training_progress.csv")
            _write_csv(episode_rows, output_dir / "training_episodes.csv")
        while (
            next_checkpoint < len(checkpoint_targets)
            and completed_steps >= checkpoint_targets[next_checkpoint]
        ):
            model.save(
                checkpoints
                / f"model_{checkpoint_targets[next_checkpoint]}_steps.pt"
            )
            next_checkpoint += 1
    progress.close()
    elapsed = perf_counter() - started
    model.save(output_dir / "model.pt")
    (output_dir / "settings.json").write_text(
        json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n"
    )
    _write_training_episodes(episode_rows, output_dir / "training_episodes.csv")
    _write_csv(update_rows, output_dir / "training_progress.csv")
    for current_env in envs:
        current_env.close()
    return model, elapsed
