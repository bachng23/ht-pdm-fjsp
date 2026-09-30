"""Value-decomposition learners for the passive-technician benchmark."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    "tqmix_no_edge",
    "tqmix_no_queue",
    "tqmix_no_cf",
    "tqmix",
)


COMPONENT_FLAGS = {
    "vdn": (False, False, False),
    "qmix": (False, False, False),
    "qmix_edge": (True, False, False),
    "qmix_queue": (False, True, False),
    "qmix_counterfactual": (False, False, True),
    "tqmix_no_edge": (False, True, True),
    "tqmix_no_queue": (True, False, True),
    "tqmix_no_cf": (True, True, False),
    "tqmix": (True, True, True),
}


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
    lambda_cf: float = 0.05


@dataclass(frozen=True)
class AnchorAdaptationSettings:
    phase_boundary_steps: int
    nominal_batch_size: int
    anchor_lambda: float
    anchor_temperature: float = 1.0


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
        (
            self.use_edge_q,
            self.use_queue_mixer,
            self.use_counterfactual,
        ) = COMPONENT_FLAGS[algorithm]
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

    def save(
        self,
        path: Path,
        seed: int,
        settings: ValueTrainSettings | None = None,
    ) -> None:
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
                "component_flags": {
                    "edge_utilities": self.use_edge_q,
                    "queue_mixer": self.use_queue_mixer,
                    "counterfactual_loss": self.use_counterfactual,
                },
                "settings": asdict(settings) if settings is not None else None,
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
        model.checkpoint_settings = payload.get("settings")
        model.eval()
        return model

    def parameter_counts(self) -> dict[str, int]:
        agent = sum(parameter.numel() for parameter in self.agent.parameters() if parameter.requires_grad)
        mixer = sum(parameter.numel() for parameter in self.mixer.parameters() if parameter.requires_grad)
        return {"agent_parameters": agent, "mixer_parameters": mixer, "total_parameters": agent + mixer}


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

    def retain_latest(self, capacity: int) -> "ReplayBuffer":
        if capacity <= 0:
            raise ValueError("replay capacity must be positive")
        resized = ReplayBuffer(
            capacity,
            self.local.shape[1],
            self.local.shape[2],
            self.masks.shape[2],
        )
        if not self.size:
            return resized
        if self.size < self.capacity:
            chronological = np.arange(self.size)
        else:
            chronological = (
                self.position + np.arange(self.size)
            ) % self.capacity
        retained = chronological[-min(self.size, capacity) :]
        count = len(retained)
        for name in (
            "local",
            "next_local",
            "masks",
            "next_masks",
            "actions",
            "rewards",
            "dones",
        ):
            getattr(resized, name)[:count] = getattr(self, name)[retained]
        resized.size = count
        resized.position = count % capacity
        return resized


def _concatenate_replay_batches(
    batches: Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("at least one replay batch is required")
    return {
        key: np.concatenate([batch[key] for batch in batches], axis=0)
        for key in batches[0]
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
        total_losses: list[float] = []
        td_losses: list[float] = []
        raw_cf_losses: list[float] = []
        weighted_cf_losses: list[float] = []
        episode_updates = 0
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
                    weighted_consistency = (
                        settings.lambda_cf * consistency
                        if online.use_counterfactual
                        else torch.zeros((), device=device)
                    )
                    loss = td_loss + weighted_consistency
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                    optimizer.step()
                    td_losses.append(float(td_loss.detach().cpu()))
                    if online.use_counterfactual:
                        raw_cf_losses.append(float(consistency.detach().cpu()))
                    weighted_cf_losses.append(float(weighted_consistency.detach().cpu()))
                    total_losses.append(float(loss.detach().cpu()))
                    episode_updates += 1
                if global_step % settings.target_update_interval == 0:
                    target.load_state_dict(online.state_dict())
        progress.append(
            {
                "policy": algorithm,
                "train_seed": seed,
                "episode": episode,
                "environment_steps": global_step,
                "update_count": episode_updates,
                "objective": episode_objective,
                "td_loss": float(np.mean(td_losses)) if td_losses else None,
                "raw_cf_loss": (
                    float(np.mean(raw_cf_losses))
                    if raw_cf_losses
                    else None
                ),
                "weighted_cf_loss": (
                    float(np.mean(weighted_cf_losses))
                    if weighted_cf_losses
                    else 0.0
                ),
                "total_loss": float(np.mean(total_losses)) if total_losses else None,
                "loss": float(np.mean(total_losses)) if total_losses else 0.0,
                "epsilon": epsilon,
                **env.metrics,
            }
        )
        if episode in targets:
            path = root / f"budget_{episode}" / "model.pt"
            online.save(path, seed, settings)
            checkpoints[episode] = path
    return checkpoints, progress


def train_value_decomposition_step_checkpoints(
    config: PassiveConfig,
    algorithm: str,
    seed: int,
    step_budgets: tuple[int, ...],
    root: Path,
    settings: ValueTrainSettings,
    device: torch.device,
    training_cells: Mapping[str, PassiveConfig],
    episode_scenarios: Sequence[str],
) -> tuple[dict[int, Path], list[dict[str, Any]]]:
    """Train to exact environment-step checkpoints under a fixed scenario schedule."""
    if not step_budgets or tuple(sorted(set(step_budgets))) != step_budgets:
        raise ValueError("step_budgets must be positive, sorted, and unique")
    if step_budgets[0] <= 0:
        raise ValueError("step_budgets must be positive")
    if not episode_scenarios:
        raise ValueError("episode_scenarios cannot be empty")
    unknown = sorted(set(episode_scenarios) - set(training_cells))
    if unknown:
        raise ValueError(f"unknown training scenarios: {unknown}")
    for name, cell in training_cells.items():
        if (
            cell.machines != config.machines
            or cell.technicians != config.technicians
            or cell.observation_dim != config.observation_dim
        ):
            raise ValueError(f"training cell {name} has incompatible dimensions")
    scheduled_steps = sum(training_cells[name].horizon for name in episode_scenarios)
    if scheduled_steps < max(step_budgets):
        raise ValueError(
            f"scenario schedule has {scheduled_steps} steps, below {max(step_budgets)}"
        )

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
    targets = set(step_budgets)
    maximum_steps = max(step_budgets)
    global_step = 0

    for episode, scenario in enumerate(
        tqdm(
            episode_scenarios,
            desc=f"{algorithm} {seed}",
            unit="episode",
            leave=False,
        ),
        start=1,
    ):
        episode_config = training_cells[scenario]
        env = PassiveTechnicianEnv(
            episode_config, seed=seed * 100_000 + episode - 1
        )
        observations = env.reset()
        done = False
        episode_objective = 0.0
        total_losses: list[float] = []
        td_losses: list[float] = []
        raw_cf_losses: list[float] = []
        weighted_cf_losses: list[float] = []
        episode_updates = 0
        while not done:
            local = _obs_tensor(observations, episode_config).numpy()
            masks = np.asarray(env.action_masks(), dtype=np.bool_)
            epsilon = max(
                settings.epsilon_end,
                settings.epsilon_start
                - (settings.epsilon_start - settings.epsilon_end)
                * global_step
                / max(1, int(maximum_steps * settings.epsilon_fraction)),
            )
            with torch.no_grad():
                local_tensor = torch.as_tensor(
                    local, dtype=torch.float32, device=device
                ).unsqueeze(0)
                q_values = online.agent_q(local_tensor)[0].masked_fill(
                    ~torch.as_tensor(masks, dtype=torch.bool, device=device), -1e9
                )
                greedy = q_values.argmax(dim=-1).cpu().tolist()
            actions = tuple(
                int(rng.choice(np.flatnonzero(masks[machine])))
                if rng.random() < epsilon
                else int(greedy[machine])
                for machine in range(episode_config.machines)
            )
            next_observations, reward, done, _ = env.step(actions)
            next_local = _obs_tensor(next_observations, episode_config).numpy()
            next_masks = np.asarray(env.action_masks(), dtype=np.bool_)
            replay.add(
                local,
                masks,
                actions,
                float(reward),
                next_local,
                next_masks,
                done,
            )
            observations = next_observations
            episode_objective += -float(reward)
            global_step += 1
            if (
                replay.size >= settings.learning_starts
                and global_step % settings.train_frequency == 0
            ):
                for _ in range(settings.gradient_steps):
                    batch = replay.sample(settings.batch_size, rng)
                    batch_local = torch.as_tensor(
                        batch["local"], dtype=torch.float32, device=device
                    )
                    batch_masks = torch.as_tensor(
                        batch["masks"], dtype=torch.bool, device=device
                    )
                    batch_actions = torch.as_tensor(
                        batch["actions"], dtype=torch.long, device=device
                    )
                    batch_next_local = torch.as_tensor(
                        batch["next_local"], dtype=torch.float32, device=device
                    )
                    batch_next_masks = torch.as_tensor(
                        batch["next_masks"], dtype=torch.bool, device=device
                    )
                    chosen_q = online.total_q(batch_local, batch_actions)
                    with torch.no_grad():
                        next_online = online.agent_q(batch_next_local).masked_fill(
                            ~batch_next_masks, -1e9
                        )
                        next_actions = next_online.argmax(dim=-1)
                        next_q = target.total_q(batch_next_local, next_actions)
                        target_q = torch.as_tensor(
                            batch["rewards"], dtype=torch.float32, device=device
                        ) + settings.gamma * (
                            ~torch.as_tensor(
                                batch["dones"], dtype=torch.bool, device=device
                            )
                        ).float() * next_q
                    td_loss = (chosen_q - target_q).pow(2).mean()
                    consistency = online.local_edge_consistency(
                        batch_local, batch_actions, batch_masks
                    )
                    weighted_consistency = (
                        settings.lambda_cf * consistency
                        if online.use_counterfactual
                        else torch.zeros((), device=device)
                    )
                    loss = td_loss + weighted_consistency
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                    optimizer.step()
                    td_losses.append(float(td_loss.detach().cpu()))
                    if online.use_counterfactual:
                        raw_cf_losses.append(float(consistency.detach().cpu()))
                    weighted_cf_losses.append(
                        float(weighted_consistency.detach().cpu())
                    )
                    total_losses.append(float(loss.detach().cpu()))
                    episode_updates += 1
                if global_step % settings.target_update_interval == 0:
                    target.load_state_dict(online.state_dict())
            if global_step in targets:
                if not done:
                    raise RuntimeError(
                        f"step checkpoint {global_step} is not an episode boundary"
                    )
                path = root / f"step_{global_step}" / "model.pt"
                online.save(path, seed, settings)
                checkpoints[global_step] = path

        progress.append(
            {
                "policy": algorithm,
                "train_seed": seed,
                "training_scenario": scenario,
                "episode": episode,
                "environment_steps": global_step,
                "update_count": episode_updates,
                "objective": episode_objective,
                "td_loss": float(np.mean(td_losses)) if td_losses else None,
                "raw_cf_loss": (
                    float(np.mean(raw_cf_losses)) if raw_cf_losses else None
                ),
                "weighted_cf_loss": (
                    float(np.mean(weighted_cf_losses))
                    if weighted_cf_losses
                    else 0.0
                ),
                "total_loss": (
                    float(np.mean(total_losses)) if total_losses else None
                ),
                "loss": float(np.mean(total_losses)) if total_losses else 0.0,
                "epsilon": epsilon,
                **env.metrics,
            }
        )
        if global_step >= maximum_steps:
            break

    if set(checkpoints) != targets or global_step != maximum_steps:
        raise RuntimeError(
            f"step training incomplete: steps={global_step}, checkpoints={sorted(checkpoints)}"
        )
    return checkpoints, progress


def train_value_decomposition_anchor_checkpoints(
    config: PassiveConfig,
    algorithm: str,
    seed: int,
    step_budgets: tuple[int, ...],
    root: Path,
    settings: ValueTrainSettings,
    device: torch.device,
    training_cells: Mapping[str, PassiveConfig],
    episode_scenarios: Sequence[str],
    adaptation: AnchorAdaptationSettings,
    nominal_scenario: str = "in_distribution",
) -> tuple[dict[int, Path], list[dict[str, Any]], dict[str, Any]]:
    """Train a nominal curriculum with stratified phase-two replay and KL anchor."""
    if not step_budgets or tuple(sorted(set(step_budgets))) != step_budgets:
        raise ValueError("step_budgets must be positive, sorted, and unique")
    if step_budgets[0] <= 0:
        raise ValueError("step_budgets must be positive")
    if adaptation.phase_boundary_steps not in step_budgets:
        raise ValueError("phase boundary must be a requested checkpoint")
    if not 0 < adaptation.nominal_batch_size < settings.batch_size:
        raise ValueError("nominal batch size must be between zero and batch size")
    if adaptation.anchor_lambda < 0:
        raise ValueError("anchor lambda must be non-negative")
    if adaptation.anchor_temperature <= 0:
        raise ValueError("anchor temperature must be positive")
    if nominal_scenario not in training_cells:
        raise ValueError(f"unknown nominal scenario: {nominal_scenario}")
    if not episode_scenarios:
        raise ValueError("episode_scenarios cannot be empty")
    unknown = sorted(set(episode_scenarios) - set(training_cells))
    if unknown:
        raise ValueError(f"unknown training scenarios: {unknown}")
    for name, cell in training_cells.items():
        if (
            cell.machines != config.machines
            or cell.technicians != config.technicians
            or cell.observation_dim != config.observation_dim
        ):
            raise ValueError(f"training cell {name} has incompatible dimensions")
    scheduled_steps = sum(training_cells[name].horizon for name in episode_scenarios)
    if scheduled_steps < max(step_budgets):
        raise ValueError(
            f"scenario schedule has {scheduled_steps} steps, below {max(step_budgets)}"
        )

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
    stress_capacity = (
        settings.replay_capacity
        * (settings.batch_size - adaptation.nominal_batch_size)
        // settings.batch_size
    )
    nominal_adaptation_capacity = settings.replay_capacity - stress_capacity
    if stress_capacity <= 0:
        raise ValueError("stratified stress replay capacity must be positive")
    nominal_replay = ReplayBuffer(
        settings.replay_capacity,
        config.machines,
        config.observation_dim,
        config.technicians + 1,
    )
    stress_replay = ReplayBuffer(
        stress_capacity,
        config.machines,
        config.observation_dim,
        config.technicians + 1,
    )
    teacher: PassiveValueDecomposition | None = None
    checkpoints: dict[int, Path] = {}
    progress: list[dict[str, Any]] = []
    targets = set(step_budgets)
    maximum_steps = max(step_budgets)
    global_step = 0
    phase_two_updates_before_stress = 0
    phase_two_stratified_updates = 0
    total_nominal_batch_samples = 0
    total_stress_batch_samples = 0

    for episode, scenario in enumerate(
        tqdm(
            episode_scenarios,
            desc=f"{algorithm} anchor {seed}",
            unit="episode",
            leave=False,
        ),
        start=1,
    ):
        episode_config = training_cells[scenario]
        env = PassiveTechnicianEnv(
            episode_config, seed=seed * 100_000 + episode - 1
        )
        observations = env.reset()
        done = False
        episode_objective = 0.0
        total_losses: list[float] = []
        td_losses: list[float] = []
        raw_cf_losses: list[float] = []
        weighted_cf_losses: list[float] = []
        raw_anchor_losses: list[float] = []
        weighted_anchor_losses: list[float] = []
        episode_updates = 0
        episode_nominal_batch_samples = 0
        episode_stress_batch_samples = 0
        episode_stratified_updates = 0
        while not done:
            local = _obs_tensor(observations, episode_config).numpy()
            masks = np.asarray(env.action_masks(), dtype=np.bool_)
            epsilon = max(
                settings.epsilon_end,
                settings.epsilon_start
                - (settings.epsilon_start - settings.epsilon_end)
                * global_step
                / max(1, int(maximum_steps * settings.epsilon_fraction)),
            )
            with torch.no_grad():
                local_tensor = torch.as_tensor(
                    local, dtype=torch.float32, device=device
                ).unsqueeze(0)
                q_values = online.agent_q(local_tensor)[0].masked_fill(
                    ~torch.as_tensor(masks, dtype=torch.bool, device=device), -1e9
                )
                greedy = q_values.argmax(dim=-1).cpu().tolist()
            actions = tuple(
                int(rng.choice(np.flatnonzero(masks[machine])))
                if rng.random() < epsilon
                else int(greedy[machine])
                for machine in range(episode_config.machines)
            )
            next_observations, reward, done, _ = env.step(actions)
            next_local = _obs_tensor(next_observations, episode_config).numpy()
            next_masks = np.asarray(env.action_masks(), dtype=np.bool_)
            active_replay = (
                nominal_replay if scenario == nominal_scenario else stress_replay
            )
            active_replay.add(
                local,
                masks,
                actions,
                float(reward),
                next_local,
                next_masks,
                done,
            )
            observations = next_observations
            episode_objective += -float(reward)
            global_step += 1
            if (
                nominal_replay.size + stress_replay.size >= settings.learning_starts
                and global_step % settings.train_frequency == 0
            ):
                for _ in range(settings.gradient_steps):
                    phase_two = global_step > adaptation.phase_boundary_steps
                    if phase_two and stress_replay.size:
                        nominal_batch_size = adaptation.nominal_batch_size
                        stress_batch_size = settings.batch_size - nominal_batch_size
                        batch = _concatenate_replay_batches(
                            (
                                nominal_replay.sample(nominal_batch_size, rng),
                                stress_replay.sample(stress_batch_size, rng),
                            )
                        )
                        phase_two_stratified_updates += 1
                        episode_stratified_updates += 1
                    else:
                        nominal_batch_size = settings.batch_size
                        stress_batch_size = 0
                        batch = nominal_replay.sample(settings.batch_size, rng)
                        if phase_two:
                            phase_two_updates_before_stress += 1
                    episode_nominal_batch_samples += nominal_batch_size
                    episode_stress_batch_samples += stress_batch_size
                    total_nominal_batch_samples += nominal_batch_size
                    total_stress_batch_samples += stress_batch_size

                    batch_local = torch.as_tensor(
                        batch["local"], dtype=torch.float32, device=device
                    )
                    batch_masks = torch.as_tensor(
                        batch["masks"], dtype=torch.bool, device=device
                    )
                    batch_actions = torch.as_tensor(
                        batch["actions"], dtype=torch.long, device=device
                    )
                    batch_next_local = torch.as_tensor(
                        batch["next_local"], dtype=torch.float32, device=device
                    )
                    batch_next_masks = torch.as_tensor(
                        batch["next_masks"], dtype=torch.bool, device=device
                    )
                    chosen_q = online.total_q(batch_local, batch_actions)
                    with torch.no_grad():
                        next_online = online.agent_q(batch_next_local).masked_fill(
                            ~batch_next_masks, -1e9
                        )
                        next_actions = next_online.argmax(dim=-1)
                        next_q = target.total_q(batch_next_local, next_actions)
                        target_q = torch.as_tensor(
                            batch["rewards"], dtype=torch.float32, device=device
                        ) + settings.gamma * (
                            ~torch.as_tensor(
                                batch["dones"], dtype=torch.bool, device=device
                            )
                        ).float() * next_q
                    td_loss = (chosen_q - target_q).pow(2).mean()
                    consistency = online.local_edge_consistency(
                        batch_local, batch_actions, batch_masks
                    )
                    weighted_consistency = (
                        settings.lambda_cf * consistency
                        if online.use_counterfactual
                        else torch.zeros((), device=device)
                    )
                    anchor_loss = torch.zeros((), device=device)
                    if (
                        phase_two
                        and teacher is not None
                        and adaptation.anchor_lambda > 0
                    ):
                        nominal_local = batch_local[:nominal_batch_size]
                        nominal_masks = batch_masks[:nominal_batch_size]
                        temperature = adaptation.anchor_temperature
                        with torch.no_grad():
                            teacher_logits = teacher.agent_q(nominal_local).masked_fill(
                                ~nominal_masks, -1e9
                            ) / temperature
                            teacher_probabilities = torch.softmax(
                                teacher_logits, dim=-1
                            )
                            teacher_log_probabilities = torch.log_softmax(
                                teacher_logits, dim=-1
                            )
                        online_log_probabilities = torch.log_softmax(
                            online.agent_q(nominal_local).masked_fill(
                                ~nominal_masks, -1e9
                            )
                            / temperature,
                            dim=-1,
                        )
                        anchor_loss = (
                            teacher_probabilities
                            * (
                                teacher_log_probabilities
                                - online_log_probabilities
                            )
                        ).sum(dim=-1).mean() * (temperature**2)
                    weighted_anchor = adaptation.anchor_lambda * anchor_loss
                    loss = td_loss + weighted_consistency + weighted_anchor
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 5.0)
                    optimizer.step()
                    td_losses.append(float(td_loss.detach().cpu()))
                    if online.use_counterfactual:
                        raw_cf_losses.append(float(consistency.detach().cpu()))
                    weighted_cf_losses.append(
                        float(weighted_consistency.detach().cpu())
                    )
                    raw_anchor_losses.append(float(anchor_loss.detach().cpu()))
                    weighted_anchor_losses.append(
                        float(weighted_anchor.detach().cpu())
                    )
                    total_losses.append(float(loss.detach().cpu()))
                    episode_updates += 1
                if global_step % settings.target_update_interval == 0:
                    target.load_state_dict(online.state_dict())
            if global_step in targets:
                if not done:
                    raise RuntimeError(
                        f"step checkpoint {global_step} is not an episode boundary"
                    )
                path = root / f"step_{global_step}" / "model.pt"
                online.save(path, seed, settings)
                checkpoints[global_step] = path
                if global_step == adaptation.phase_boundary_steps:
                    nominal_replay = nominal_replay.retain_latest(
                        nominal_adaptation_capacity
                    )
                    teacher = PassiveValueDecomposition(
                        config,
                        algorithm,
                        settings.hidden_dim,
                        settings.mixer_hidden_dim,
                    ).to(device)
                    teacher.load_state_dict(online.state_dict())
                    teacher.eval()
                    for parameter in teacher.parameters():
                        parameter.requires_grad_(False)

        progress.append(
            {
                "policy": algorithm,
                "train_seed": seed,
                "training_scenario": scenario,
                "training_phase": (
                    "nominal_pretrain"
                    if global_step <= adaptation.phase_boundary_steps
                    else "stress_adaptation"
                ),
                "episode": episode,
                "environment_steps": global_step,
                "update_count": episode_updates,
                "batch_nominal_samples": episode_nominal_batch_samples,
                "batch_stress_samples": episode_stress_batch_samples,
                "stratified_update_count": episode_stratified_updates,
                "objective": episode_objective,
                "td_loss": float(np.mean(td_losses)) if td_losses else None,
                "raw_cf_loss": (
                    float(np.mean(raw_cf_losses)) if raw_cf_losses else None
                ),
                "weighted_cf_loss": (
                    float(np.mean(weighted_cf_losses))
                    if weighted_cf_losses
                    else 0.0
                ),
                "raw_anchor_loss": (
                    float(np.mean(raw_anchor_losses))
                    if raw_anchor_losses
                    else 0.0
                ),
                "weighted_anchor_loss": (
                    float(np.mean(weighted_anchor_losses))
                    if weighted_anchor_losses
                    else 0.0
                ),
                "total_loss": (
                    float(np.mean(total_losses)) if total_losses else None
                ),
                "loss": float(np.mean(total_losses)) if total_losses else 0.0,
                "epsilon": epsilon,
                **env.metrics,
            }
        )
        if global_step >= maximum_steps:
            break

    if set(checkpoints) != targets or global_step != maximum_steps:
        raise RuntimeError(
            f"anchor training incomplete: steps={global_step}, checkpoints={sorted(checkpoints)}"
        )
    if teacher is None:
        raise RuntimeError("nominal teacher was not captured at the phase boundary")
    diagnostics = {
        "phase_boundary_steps": adaptation.phase_boundary_steps,
        "nominal_batch_size": adaptation.nominal_batch_size,
        "stress_batch_size": settings.batch_size - adaptation.nominal_batch_size,
        "anchor_lambda": adaptation.anchor_lambda,
        "anchor_temperature": adaptation.anchor_temperature,
        "phase_one_replay_capacity": settings.replay_capacity,
        "phase_two_nominal_replay_capacity": nominal_replay.capacity,
        "phase_two_stress_replay_capacity": stress_replay.capacity,
        "phase_two_total_replay_capacity": (
            nominal_replay.capacity + stress_replay.capacity
        ),
        "phase_two_updates_before_stress": phase_two_updates_before_stress,
        "phase_two_stratified_updates": phase_two_stratified_updates,
        "total_nominal_batch_samples": total_nominal_batch_samples,
        "total_stress_batch_samples": total_stress_batch_samples,
    }
    return checkpoints, progress, diagnostics


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
