"""Value-decomposition learners for the passive-technician benchmark."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import (
    _action_diagnostics,
    _obs_tensor,
)
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig, PassiveTechnicianEnv


VALUE_ALGORITHMS = (
    "vdn",
    "qmix",
    "qmix_edge",
    "qmix_queue",
    "qmix_counterfactual",
    "tqmix",
)


@dataclass(frozen=True)
class ValueTrainSettings:
    episodes: int
    hidden_dim: int = 64
    mixer_hidden_dim: int = 64
    learning_rate: float = 3e-4
    gamma: float = 0.95
    replay_capacity: int = 50_000
    batch_size: int = 128
    learning_starts: int = 256
    train_frequency: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_fraction: float = 0.8


class TechnicianEdgeQ(nn.Module):
    """Q network with an explicit defer score and technician-edge scores."""

    def __init__(self, observation_dim: int, technicians: int, hidden_dim: int) -> None:
        super().__init__()
        self.technicians = technicians
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.tech_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.defer = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(observations)
        slots = observations[..., 3:].reshape(
            *observations.shape[:-1], self.technicians, 6
        )
        encoded = self.tech_encoder(slots)
        edge = (
            self.query(hidden).unsqueeze(-2) * self.key(encoded)
        ).sum(dim=-1) / float(self.hidden_dim) ** 0.5
        return torch.cat((self.defer(hidden), edge), dim=-1)


class LocalQ(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class PassiveQMixer(nn.Module):
    def __init__(self, agents: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.agents = agents
        self.hidden_dim = hidden_dim
        self.hyper_w1 = nn.Linear(state_dim, agents * hidden_dim)
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        self.value = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, agent_q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = agent_q.shape[0]
        w1 = self.hyper_w1(state).abs().view(batch, self.agents, self.hidden_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.hidden_dim)
        hidden = torch.nn.functional.elu(torch.bmm(agent_q.unsqueeze(1), w1) + b1)
        w2 = self.hyper_w2(state).abs().view(batch, self.hidden_dim, 1)
        return torch.bmm(hidden, w2).squeeze((1, 2)) + self.value(state).squeeze(-1)


class TechnicianAwareMixer(PassiveQMixer):
    """QMIX mixer whose hypernetwork sees explicit technician congestion."""

    def __init__(
        self,
        agents: int,
        observation_dim: int,
        technicians: int,
        hidden_dim: int,
    ) -> None:
        context_dim = technicians * 3
        super().__init__(agents, agents * observation_dim + context_dim, hidden_dim)
        self.observation_dim = observation_dim
        self.technicians = technicians

    def _state(self, state: torch.Tensor) -> torch.Tensor:
        local = state.view(-1, self.agents, self.observation_dim)
        slots = local[..., 3:].reshape(
            state.shape[0], self.agents, self.technicians, 6
        )
        # availability, remaining busy time, and queue length are the most
        # direct congestion signals; mean and maximum preserve global load.
        context = torch.stack(
            (
                slots[..., 0].mean(dim=1),
                slots[..., 1].amax(dim=1),
                slots[..., 3].mean(dim=1),
            ),
            dim=-1,
        ).flatten(start_dim=1)
        return torch.cat((state, context), dim=-1)

    def forward(self, agent_q: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return super().forward(agent_q, self._state(state))


class PassiveValueDecomposition(nn.Module):
    def __init__(
        self,
        config: PassiveConfig,
        algorithm: str,
        hidden_dim: int = 64,
        mixer_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        if algorithm not in VALUE_ALGORITHMS:
            raise ValueError(algorithm)
        self.algorithm = algorithm
        self.use_edge_q = algorithm in {"qmix_edge", "tqmix"}
        self.use_queue_mixer = algorithm in {"qmix_queue", "tqmix"}
        self.use_counterfactual = algorithm in {"qmix_counterfactual", "tqmix"}
        self.machines = config.machines
        self.observation_dim = config.observation_dim
        self.action_dim = config.technicians + 1
        self.hidden_dim = hidden_dim
        self.mixer_hidden_dim = mixer_hidden_dim
        if self.use_edge_q:
            self.agent = TechnicianEdgeQ(
                config.observation_dim, config.technicians, hidden_dim
            )
        else:
            self.agent = LocalQ(config.observation_dim, self.action_dim, hidden_dim)
        if self.use_queue_mixer:
            self.mixer = TechnicianAwareMixer(
                config.machines,
                config.observation_dim,
                config.technicians,
                mixer_hidden_dim,
            )
        else:
            self.mixer = PassiveQMixer(
                config.machines,
                config.machines * config.observation_dim,
                mixer_hidden_dim,
            )

    def agent_q(self, local: torch.Tensor) -> torch.Tensor:
        batch, machines, features = local.shape
        return self.agent(local.reshape(batch * machines, features)).view(
            batch, machines, self.action_dim
        )

    def total_q(self, local: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        values = self.agent_q(local)
        chosen = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        if self.algorithm == "vdn":
            return chosen.sum(dim=-1)
        return self.mixer(chosen, local.flatten(start_dim=1))

    def local_edge_consistency(
        self,
        local: torch.Tensor,
        actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        """Align local technician differences with joint counterfactual values."""
        if not self.use_counterfactual:
            return torch.zeros((), device=local.device)
        values = self.agent_q(local)
        chosen = values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        losses: list[torch.Tensor] = []
        for machine in range(self.machines):
            valid = masks[:, machine]
            selected = chosen[:, machine]
            for alternative in range(self.action_dim):
                alternative_valid = valid[:, alternative]
                if not bool(alternative_valid.any()):
                    continue
                counterfactual = actions.clone()
                counterfactual[:, machine] = alternative
                joint_delta = self.total_q(local, actions) - self.total_q(local, counterfactual)
                local_delta = selected - values[:, machine, alternative]
                losses.append((local_delta[alternative_valid] - joint_delta[alternative_valid].detach()).pow(2).mean())
        return torch.stack(losses).mean() if losses else torch.zeros((), device=local.device)

    def save(self, path: Path, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "algorithm": self.algorithm,
                "machines": self.machines,
                "observation_dim": self.observation_dim,
                "action_dim": self.action_dim,
                "hidden_dim": self.hidden_dim,
                "mixer_hidden_dim": self.mixer_hidden_dim,
                "seed": seed,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, config: PassiveConfig, device: torch.device) -> "PassiveValueDecomposition":
        payload = torch.load(path, map_location=device, weights_only=True)
        model = cls(
            config,
            str(payload["algorithm"]),
            int(payload["hidden_dim"]),
            int(payload["mixer_hidden_dim"]),
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


class ReplayBuffer:
    def __init__(self, capacity: int, machines: int, observation_dim: int, action_dim: int) -> None:
        self.capacity = capacity
        self.local = np.empty((capacity, machines, observation_dim), dtype=np.float16)
        self.next_local = np.empty_like(self.local)
        self.masks = np.empty((capacity, machines, action_dim), dtype=np.bool_)
        self.next_masks = np.empty_like(self.masks)
        self.actions = np.empty((capacity, machines), dtype=np.int16)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.bool_)
        self.position = 0
        self.size = 0

    def add(
        self,
        local: np.ndarray,
        masks: np.ndarray,
        actions: tuple[int, ...],
        reward: float,
        next_local: np.ndarray,
        next_masks: np.ndarray,
        done: bool,
    ) -> None:
        i = self.position
        self.local[i] = local
        self.masks[i] = masks
        self.actions[i] = actions
        self.rewards[i] = reward
        self.next_local[i] = next_local
        self.next_masks[i] = next_masks
        self.dones[i] = done
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
        indices = rng.integers(0, self.size, size=batch_size)
        return {
            "local": self.local[indices],
            "masks": self.masks[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_local": self.next_local[indices],
            "next_masks": self.next_masks[indices],
            "dones": self.dones[indices],
        }


def train_value_decomposition_checkpoints(
    config: PassiveConfig,
    algorithm: str,
    seed: int,
    budgets: tuple[int, ...],
    root: Path,
    settings: ValueTrainSettings,
    device: torch.device,
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    rng = np.random.default_rng(seed)
    online = PassiveValueDecomposition(
        config, algorithm, settings.hidden_dim, settings.mixer_hidden_dim
    ).to(device)
    target = PassiveValueDecomposition(
        config, algorithm, settings.hidden_dim, settings.mixer_hidden_dim
    ).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=settings.learning_rate)
    replay = ReplayBuffer(
        settings.replay_capacity,
        config.machines,
        config.observation_dim,
        config.technicians + 1,
    )
    checkpoints: dict[int, Path] = {}
    progress: list[dict[str, Any]] = []
    targets = set(budgets)
    global_step = 0
    for episode in tqdm(
        range(1, max(budgets) + 1),
        desc=f"{algorithm} {seed}",
        unit="episode",
        leave=False,
    ):
        env = PassiveTechnicianEnv(config, seed=seed * 100_000 + episode - 1)
        observations = env.reset()
        done = False
        episode_objective = 0.0
        losses: list[float] = []
        while not done:
            local = _obs_tensor(observations, config).numpy()
            masks = np.asarray(env.action_masks(), dtype=np.bool_)
            epsilon = max(
                settings.epsilon_end,
                settings.epsilon_start
                - (settings.epsilon_start - settings.epsilon_end)
                * global_step
                / max(1, int(settings.episodes * 12 * settings.epsilon_fraction)),
            )
            with torch.no_grad():
                local_tensor = torch.as_tensor(local, dtype=torch.float32, device=device).unsqueeze(0)
                q_values = online.agent_q(local_tensor)[0].masked_fill(
                    ~torch.as_tensor(masks, dtype=torch.bool, device=device), -1e9
                )
                greedy = q_values.argmax(dim=-1).cpu().tolist()
            actions = tuple(
                int(rng.choice(np.flatnonzero(masks[machine])))
                if rng.random() < epsilon
                else int(greedy[machine])
                for machine in range(config.machines)
            )
            next_observations, reward, done, _ = env.step(actions)
            next_local = _obs_tensor(next_observations, config).numpy()
            next_masks = np.asarray(env.action_masks(), dtype=np.bool_)
            replay.add(local, masks, actions, float(reward), next_local, next_masks, done)
            observations = next_observations
            episode_objective += -float(reward)
            global_step += 1
            if replay.size >= settings.learning_starts and global_step % settings.train_frequency == 0:
                for _ in range(settings.gradient_steps):
                    batch = replay.sample(settings.batch_size, rng)
                    batch_local = torch.as_tensor(batch["local"], dtype=torch.float32, device=device)
                    batch_masks = torch.as_tensor(batch["masks"], dtype=torch.bool, device=device)
                    batch_actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=device)
                    batch_next_local = torch.as_tensor(batch["next_local"], dtype=torch.float32, device=device)
                    batch_next_masks = torch.as_tensor(batch["next_masks"], dtype=torch.bool, device=device)
                    chosen_q = online.total_q(batch_local, batch_actions)
                    with torch.no_grad():
                        next_online = online.agent_q(batch_next_local).masked_fill(~batch_next_masks, -1e9)
                        next_actions = next_online.argmax(dim=-1)
                        next_q = target.total_q(batch_next_local, next_actions)
                        target_q = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device) + settings.gamma * (~torch.as_tensor(batch["dones"], dtype=torch.bool, device=device)).float() * next_q
                    td_loss = (chosen_q - target_q).pow(2).mean()
                    consistency = online.local_edge_consistency(batch_local, batch_actions, batch_masks)
                    loss = td_loss + (0.05 * consistency if online.use_counterfactual else 0.0)
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                    optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                if global_step % settings.target_update_interval == 0:
                    target.load_state_dict(online.state_dict())
        progress.append(
            {
                "policy": algorithm,
                "train_seed": seed,
                "episode": episode,
                "objective": episode_objective,
                "loss": float(np.mean(losses)) if losses else 0.0,
                "epsilon": epsilon,
            }
        )
        if episode in targets:
            path = root / f"budget_{episode}" / "model.pt"
            online.save(path, seed)
            checkpoints[episode] = path
    return checkpoints, progress


def evaluate_value_decomposition(
    config: PassiveConfig,
    model: PassiveValueDecomposition,
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
        local = _obs_tensor(observations, config).to(device).unsqueeze(0)
        masks = env.action_masks()
        with torch.no_grad():
            values = model.agent_q(local)[0].masked_fill(
                ~torch.as_tensor(masks, dtype=torch.bool, device=device), -1e9
            )
            actions = tuple(int(action) for action in values.argmax(dim=-1).cpu().tolist())
        joint_actions.add(actions)
        defer_actions += sum(action == 0 for action in actions)
        action_steps += 1
        observations, _, done, _ = env.step(actions)
    return {
        "policy": model.algorithm,
        "seed": seed,
        "train_seed": train_seed,
        **env.metrics,
        **_action_diagnostics(config, joint_actions, defer_actions, action_steps),
    }
