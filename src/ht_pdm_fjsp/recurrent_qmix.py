"""Recurrent candidate-conditioned QMIX used by the baseline-fidelity audit."""

from __future__ import annotations

import copy
import csv
import json
import math
from collections import deque
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
from ht_pdm_fjsp.value_decomposition import QMixer


@dataclass(frozen=True)
class RecurrentQMIXSettings:
    total_timesteps: int
    replay_capacity: int
    learning_starts: int
    episode_batch_size: int
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
    agent_embedding_dim: int
    checkpoint_targets: tuple[int, ...]
    device: str
    n_envs: int = 1


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class RecurrentQMIXPolicy(nn.Module):
    """Shared recurrent utility network plus the original monotonic mixer.

    Candidate features remain the FJSP-specific action interface. A masked
    candidate pool summarizes the current local observation, while the GRU also
    receives learned machine identity, previous proposed-action features, and
    the previous resolver outcome (-1 rejected, 0 wait, 1 accepted).
    """

    def __init__(
        self,
        *,
        agent_count: int,
        local_feature_dim: int,
        global_state_dim: int,
        hidden_dim: int = 128,
        mixer_hidden_dim: int = 64,
        agent_embedding_dim: int = 16,
    ) -> None:
        super().__init__()
        self.agent_count = agent_count
        self.local_feature_dim = local_feature_dim
        self.global_state_dim = global_state_dim
        self.hidden_dim = hidden_dim
        self.mixer_hidden_dim = mixer_hidden_dim
        self.agent_embedding_dim = agent_embedding_dim
        self.candidate_encoder = nn.Sequential(
            nn.Linear(local_feature_dim, hidden_dim), nn.ReLU()
        )
        self.agent_embedding = nn.Embedding(agent_count, agent_embedding_dim)
        self.gru = nn.GRUCell(
            hidden_dim + local_feature_dim + 1 + agent_embedding_dim,
            hidden_dim,
        )
        self.q_head = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.mixer = QMixer(agent_count, global_state_dim, mixer_hidden_dim)
        self._online_hidden: th.Tensor | None = None
        self._online_prev_action: th.Tensor | None = None
        self._online_prev_outcome: th.Tensor | None = None
        self._last_local: th.Tensor | None = None

    def initial_hidden(self, batch_size: int, *, device: str | th.device) -> th.Tensor:
        return th.zeros(
            batch_size, self.agent_count, self.hidden_dim, device=device
        )

    def q_step(
        self,
        local: th.Tensor,
        masks: th.Tensor,
        previous_action: th.Tensor,
        previous_outcome: th.Tensor,
        hidden: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        batch, agents, actions, _ = local.shape
        if agents != self.agent_count:
            raise ValueError("Agent dimension does not match recurrent QMIX")
        encoded = self.candidate_encoder(local)
        weights = masks.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=2) / weights.sum(dim=2).clamp_min(1.0)
        agent_ids = th.arange(agents, device=local.device)
        identity = self.agent_embedding(agent_ids).unsqueeze(0).expand(batch, -1, -1)
        gru_input = th.cat(
            [pooled, previous_action, previous_outcome.unsqueeze(-1), identity],
            dim=-1,
        )
        next_hidden = self.gru(
            gru_input.reshape(batch * agents, -1),
            hidden.reshape(batch * agents, self.hidden_dim),
        ).view(batch, agents, self.hidden_dim)
        context = next_hidden.unsqueeze(2).expand(-1, -1, actions, -1)
        q_values = self.q_head(th.cat([encoded, context], dim=-1)).squeeze(-1)
        return q_values, next_hidden

    def unroll(
        self,
        local: th.Tensor,
        masks: th.Tensor,
        previous_action: th.Tensor,
        previous_outcome: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        batch, timesteps = local.shape[:2]
        hidden = self.initial_hidden(batch, device=local.device)
        values: list[th.Tensor] = []
        hidden_norms: list[th.Tensor] = []
        for timestep in range(timesteps):
            q_values, hidden = self.q_step(
                local[:, timestep],
                masks[:, timestep],
                previous_action[:, timestep],
                previous_outcome[:, timestep],
                hidden,
            )
            values.append(q_values)
            hidden_norms.append(hidden.norm(dim=-1).mean(dim=-1))
        return th.stack(values, dim=1), th.stack(hidden_norms, dim=1)

    def reset_hidden(self, *, batch_size: int, device: str) -> None:
        self._online_hidden = self.initial_hidden(batch_size, device=device)
        self._online_prev_action = th.zeros(
            batch_size,
            self.agent_count,
            self.local_feature_dim,
            device=device,
        )
        self._online_prev_outcome = th.zeros(
            batch_size, self.agent_count, device=device
        )
        self._last_local = None

    def reset_rows(self, rows: list[int]) -> None:
        if not rows or self._online_hidden is None:
            return
        self._online_hidden[rows] = 0
        assert self._online_prev_action is not None
        assert self._online_prev_outcome is not None
        self._online_prev_action[rows] = 0
        self._online_prev_outcome[rows] = 0

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
        if self._online_hidden is None or self._online_hidden.shape[0] != local.shape[0]:
            self.reset_hidden(batch_size=local.shape[0], device=device)
        assert self._online_hidden is not None
        assert self._online_prev_action is not None
        assert self._online_prev_outcome is not None
        with th.no_grad():
            values, self._online_hidden = self.q_step(
                local,
                masks,
                self._online_prev_action,
                self._online_prev_outcome,
                self._online_hidden,
            )
            actions = values.masked_fill(~masks, -1e9).argmax(dim=-1).cpu().numpy()
        if not deterministic and epsilon > 0:
            generator = rng or np.random.default_rng()
            mask_array = masks.cpu().numpy()
            for batch_index in range(actions.shape[0]):
                for agent_index in range(self.agent_count):
                    if generator.random() < epsilon:
                        valid = np.flatnonzero(mask_array[batch_index, agent_index])
                        actions[batch_index, agent_index] = int(generator.choice(valid))
        self._last_local = local.detach()
        return actions.squeeze(0) if single else actions

    def observe_outcome(
        self, actions: np.ndarray, agent_outcomes: tuple[int, ...] | list[int]
    ) -> None:
        if self._last_local is None or self._online_prev_action is None:
            raise RuntimeError("act() must precede observe_outcome()")
        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.ndim == 1:
            action_array = action_array[None, :]
        action_tensor = th.as_tensor(
            action_array, dtype=th.long, device=self._last_local.device
        )
        gather_index = action_tensor[..., None, None].expand(
            -1, -1, 1, self.local_feature_dim
        )
        self._online_prev_action = self._last_local.gather(2, gather_index).squeeze(2)
        outcome = th.as_tensor(
            np.asarray(agent_outcomes)[None, :],
            dtype=th.float32,
            device=self._last_local.device,
        )
        self._online_prev_outcome = outcome

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(
            {
                "state_dict": self.state_dict(),
                "agent_count": self.agent_count,
                "local_feature_dim": self.local_feature_dim,
                "global_state_dim": self.global_state_dim,
                "hidden_dim": self.hidden_dim,
                "mixer_hidden_dim": self.mixer_hidden_dim,
                "agent_embedding_dim": self.agent_embedding_dim,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str) -> "RecurrentQMIXPolicy":
        payload = th.load(path, map_location=device, weights_only=False)
        model = cls(
            **{
                key: payload[key]
                for key in (
                    "agent_count",
                    "local_feature_dim",
                    "global_state_dim",
                    "hidden_dim",
                    "mixer_hidden_dim",
                    "agent_embedding_dim",
                )
            }
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class EpisodeReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.episodes: deque[list[dict[str, Any]]] = deque()
        self.transition_count = 0

    def add(self, episode: list[dict[str, Any]]) -> None:
        if not episode:
            return
        self.episodes.append(episode)
        self.transition_count += len(episode)
        while self.transition_count > self.capacity and len(self.episodes) > 1:
            self.transition_count -= len(self.episodes.popleft())

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> dict[str, np.ndarray]:
        if not self.episodes:
            raise ValueError("Cannot sample an empty episode replay buffer")
        chosen = [
            self.episodes[int(index)]
            for index in rng.integers(0, len(self.episodes), size=batch_size)
        ]
        max_steps = max(map(len, chosen))
        first = chosen[0][0]
        local_shape = first["local"].shape
        mask_shape = first["masks"].shape
        global_shape = first["global"].shape
        agent_count = mask_shape[0]
        feature_dim = local_shape[-1]
        local = np.zeros((batch_size, max_steps + 1, *local_shape), dtype=np.float32)
        masks = np.zeros((batch_size, max_steps + 1, *mask_shape), dtype=np.bool_)
        masks[..., 0] = True
        states = np.zeros((batch_size, max_steps + 1, *global_shape), dtype=np.float32)
        previous_action = np.zeros(
            (batch_size, max_steps + 1, agent_count, feature_dim), dtype=np.float32
        )
        previous_outcome = np.zeros(
            (batch_size, max_steps + 1, agent_count), dtype=np.float32
        )
        actions = np.zeros((batch_size, max_steps, agent_count), dtype=np.int64)
        rewards = np.zeros((batch_size, max_steps), dtype=np.float32)
        dones = np.ones((batch_size, max_steps), dtype=np.float32)
        valid = np.zeros((batch_size, max_steps), dtype=np.float32)
        for batch_index, episode in enumerate(chosen):
            for timestep, transition in enumerate(episode):
                local[batch_index, timestep] = transition["local"]
                masks[batch_index, timestep] = transition["masks"]
                states[batch_index, timestep] = transition["global"]
                previous_action[batch_index, timestep] = transition["previous_action"]
                previous_outcome[batch_index, timestep] = transition["previous_outcome"]
                actions[batch_index, timestep] = transition["actions"]
                rewards[batch_index, timestep] = transition["reward"]
                dones[batch_index, timestep] = transition["done"]
                valid[batch_index, timestep] = 1.0
            last = episode[-1]
            end = len(episode)
            local[batch_index, end] = last["next_local"]
            masks[batch_index, end] = last["next_masks"]
            states[batch_index, end] = last["next_global"]
            previous_action[batch_index, end] = last["next_previous_action"]
            previous_outcome[batch_index, end] = last["next_previous_outcome"]
        return {
            "local": local,
            "masks": masks,
            "states": states,
            "previous_action": previous_action,
            "previous_outcome": previous_outcome,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "valid": valid,
        }


def _stack_observations(
    observations: list[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    return {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }


def _epsilon_actions(
    q_values: th.Tensor,
    masks: th.Tensor,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> np.ndarray:
    actions = q_values.masked_fill(~masks, -1e9).argmax(dim=-1).cpu().numpy()
    mask_array = masks.cpu().numpy()
    for batch_index in range(actions.shape[0]):
        for agent_index in range(actions.shape[1]):
            if rng.random() < epsilon:
                valid = np.flatnonzero(mask_array[batch_index, agent_index])
                actions[batch_index, agent_index] = int(rng.choice(valid))
    return actions


def train_recurrent_qmix(
    config: BenchmarkConfig,
    settings: RecurrentQMIXSettings,
    output_dir: Path,
    *,
    train_seed: int,
    show_progress: bool,
) -> tuple[RecurrentQMIXPolicy, float]:
    if settings.n_envs < 1 or settings.total_timesteps % settings.n_envs:
        raise ValueError("Timesteps must be divisible by a positive n_envs")
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
    template = envs[0]
    model = RecurrentQMIXPolicy(
        agent_count=template.num_agents,
        local_feature_dim=template.local_feature_dim,
        global_state_dim=template.global_state_dim,
        hidden_dim=settings.hidden_dim,
        mixer_hidden_dim=settings.mixer_hidden_dim,
        agent_embedding_dim=settings.agent_embedding_dim,
    ).to(settings.device)
    target = copy.deepcopy(model).to(settings.device).eval()
    optimizer = th.optim.Adam(model.parameters(), lr=settings.learning_rate)
    replay = EpisodeReplayBuffer(settings.replay_capacity)
    current_episodes: list[list[dict[str, Any]]] = [
        [] for _ in range(settings.n_envs)
    ]
    hidden = model.initial_hidden(settings.n_envs, device=settings.device)
    previous_action = np.zeros(
        (settings.n_envs, template.num_agents, template.local_feature_dim),
        dtype=np.float32,
    )
    previous_outcome = np.zeros(
        (settings.n_envs, template.num_agents), dtype=np.float32
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    episode_counts = [0] * settings.n_envs
    episode_returns = [0.0] * settings.n_envs
    episode_lengths = [0] * settings.n_envs
    episode_rows: list[dict[str, Any]] = []
    update_rows: list[dict[str, Any]] = []
    pending: dict[str, list[float]] = {
        key: []
        for key in (
            "loss",
            "chosen_q_mean",
            "chosen_q_max",
            "q_tot_mean",
            "gradient_norm",
            "hidden_norm",
        )
    }
    completed_steps = 0
    next_update = max(
        settings.train_frequency,
        math.ceil(settings.learning_starts / settings.train_frequency)
        * settings.train_frequency,
    )
    next_target_update = settings.target_update_interval
    checkpoint_index = 0
    next_log = 1_000
    started = perf_counter()
    progress = tqdm(
        total=settings.total_timesteps,
        desc=f"RECURRENT-QMIX seed {train_seed}",
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
        batched = _stack_observations(observations)
        local_tensor = th.as_tensor(
            batched["local_observations"], dtype=th.float32, device=settings.device
        )
        masks_tensor = th.as_tensor(
            batched["action_masks"], dtype=th.bool, device=settings.device
        )
        with th.no_grad():
            q_values, next_hidden = model.q_step(
                local_tensor,
                masks_tensor,
                th.as_tensor(previous_action, dtype=th.float32, device=settings.device),
                th.as_tensor(previous_outcome, dtype=th.float32, device=settings.device),
                hidden,
            )
            actions = _epsilon_actions(
                q_values, masks_tensor, epsilon=epsilon, rng=rng
            )
        batch_indices = np.arange(settings.n_envs)[:, None]
        agent_indices = np.arange(template.num_agents)[None, :]
        chosen_features = batched["local_observations"][
            batch_indices, agent_indices, actions
        ]
        outcomes = np.zeros_like(previous_outcome)
        reset_rows: list[int] = []
        for rank, current_env in enumerate(envs):
            prior_observation = observations[rank]
            next_observation, reward, terminated, truncated, info = current_env.step(
                actions[rank]
            )
            done = terminated or truncated
            outcome = np.asarray(info["agent_outcomes"], dtype=np.float32)
            outcomes[rank] = outcome
            current_episodes[rank].append(
                {
                    "local": prior_observation["local_observations"].copy(),
                    "masks": prior_observation["action_masks"].copy(),
                    "global": prior_observation["global_state"].copy(),
                    "previous_action": previous_action[rank].copy(),
                    "previous_outcome": previous_outcome[rank].copy(),
                    "actions": actions[rank].copy(),
                    "reward": float(reward),
                    "done": float(done),
                    "next_local": next_observation["local_observations"].copy(),
                    "next_masks": next_observation["action_masks"].copy(),
                    "next_global": next_observation["global_state"].copy(),
                    "next_previous_action": chosen_features[rank].copy(),
                    "next_previous_outcome": outcome.copy(),
                }
            )
            episode_returns[rank] += reward
            episode_lengths[rank] += 1
            observations[rank] = next_observation
            if done:
                replay.add(current_episodes[rank])
                current_episodes[rank] = []
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
                observations[rank], _ = current_env.reset(
                    seed=train_seed
                    + rank
                    + episode_counts[rank] * settings.n_envs
                )
                reset_rows.append(rank)
        hidden = next_hidden.detach()
        previous_action = chosen_features.astype(np.float32, copy=True)
        previous_outcome = outcomes
        if reset_rows:
            hidden[reset_rows] = 0
            previous_action[reset_rows] = 0
            previous_outcome[reset_rows] = 0
        completed_steps += settings.n_envs
        progress.update(settings.n_envs)

        while (
            next_update <= completed_steps
            and replay.transition_count >= settings.learning_starts
            and len(replay.episodes) >= settings.episode_batch_size
        ):
            for _ in range(settings.gradient_steps):
                batch = replay.sample(settings.episode_batch_size, rng)
                local = th.as_tensor(batch["local"], dtype=th.float32, device=settings.device)
                masks = th.as_tensor(batch["masks"], dtype=th.bool, device=settings.device)
                states = th.as_tensor(batch["states"], dtype=th.float32, device=settings.device)
                prev_action_tensor = th.as_tensor(
                    batch["previous_action"], dtype=th.float32, device=settings.device
                )
                prev_outcome_tensor = th.as_tensor(
                    batch["previous_outcome"], dtype=th.float32, device=settings.device
                )
                actions_tensor = th.as_tensor(
                    batch["actions"], dtype=th.long, device=settings.device
                )
                rewards = th.as_tensor(batch["rewards"], dtype=th.float32, device=settings.device)
                dones = th.as_tensor(batch["dones"], dtype=th.float32, device=settings.device)
                valid = th.as_tensor(batch["valid"], dtype=th.float32, device=settings.device)
                online_q, hidden_norms = model.unroll(
                    local, masks, prev_action_tensor, prev_outcome_tensor
                )
                chosen = online_q[:, :-1].gather(
                    -1, actions_tensor.unsqueeze(-1)
                ).squeeze(-1)
                with th.no_grad():
                    target_q, _ = target.unroll(
                        local, masks, prev_action_tensor, prev_outcome_tensor
                    )
                    next_actions = online_q[:, 1:].detach().masked_fill(
                        ~masks[:, 1:], -1e9
                    ).argmax(dim=-1)
                    next_q = target_q[:, 1:].gather(
                        -1, next_actions.unsqueeze(-1)
                    ).squeeze(-1)
                batch_size, timesteps, agents = chosen.shape
                predicted = model.mixer(
                    chosen.reshape(batch_size * timesteps, agents),
                    states[:, :-1].reshape(batch_size * timesteps, -1),
                ).view(batch_size, timesteps)
                with th.no_grad():
                    target_total = target.mixer(
                        next_q.reshape(batch_size * timesteps, agents),
                        states[:, 1:].reshape(batch_size * timesteps, -1),
                    ).view(batch_size, timesteps)
                    td_target = rewards + settings.gamma * (1.0 - dones) * target_total
                element_loss = nn.functional.smooth_l1_loss(
                    predicted, td_target, reduction="none"
                )
                loss = (element_loss * valid).sum() / valid.sum().clamp_min(1.0)
                if not th.isfinite(loss):
                    raise FloatingPointError("Non-finite recurrent QMIX loss")
                optimizer.zero_grad()
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                selected = valid.bool()
                pending["loss"].append(float(loss.item()))
                pending["chosen_q_mean"].append(float(chosen[selected].mean().item()))
                pending["chosen_q_max"].append(float(chosen[selected].max().item()))
                pending["q_tot_mean"].append(float(predicted[selected].mean().item()))
                pending["gradient_norm"].append(float(gradient_norm))
                pending["hidden_norm"].append(
                    float(hidden_norms[:, :-1][selected].mean().item())
                )
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
                    "replay_transitions": replay.transition_count,
                    "replay_episodes": len(replay.episodes),
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
            checkpoint_index < len(settings.checkpoint_targets)
            and completed_steps >= settings.checkpoint_targets[checkpoint_index]
        ):
            target_step = settings.checkpoint_targets[checkpoint_index]
            model.save(checkpoints / f"model_{target_step}_steps.pt")
            checkpoint_index += 1
    progress.close()
    elapsed = perf_counter() - started
    model.save(output_dir / "model.pt")
    (output_dir / "settings.json").write_text(
        json.dumps(asdict(settings), indent=2, sort_keys=True) + "\n"
    )
    _write_csv(episode_rows, output_dir / "training_episodes.csv")
    _write_csv(update_rows, output_dir / "training_progress.csv")
    for env in envs:
        env.close()
    return model, elapsed
