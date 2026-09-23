"""Feed-forward QPLEX with capacity-matched soft resource-graph context."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch as th
from torch import nn
from tqdm.auto import tqdm

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.value_decomposition import (
    ReplayBuffer,
    ValueLearningSettings,
    _mlp,
    _stack_observations,
    _write_csv,
)


NULL_GRAPH = "null_graph"
ALL_FEASIBLE_GRAPH = "all_feasible"
POLICY_INTENT_TOP2_GRAPH = "policy_intent_top2"
GRAPH_VARIANTS = (
    NULL_GRAPH,
    ALL_FEASIBLE_GRAPH,
    POLICY_INTENT_TOP2_GRAPH,
)


def build_resource_conflict_tensor(env: MachineAgentsCTDEEnv) -> np.ndarray:
    """Return static action-pair resource conflicts, including padded actions.

    The tensor is indexed by ``agent_i, agent_j, action_i, action_j``.  Wait and
    padded actions have no edges.  Production actions conflict when they target
    the same job operation; maintenance actions conflict when they use the same
    technician.  The tensor depends only on the fixed action catalog, never on
    resolver outcomes.
    """

    count = env.num_agents
    actions = env.max_local_actions
    tensor = np.zeros((count, count, actions, actions), dtype=np.bool_)
    for left in range(count):
        for right in range(left + 1, count):
            for left_action, left_global in enumerate(
                env.local_action_catalogs[left]
            ):
                if left_global is None:
                    continue
                left_descriptor = env.core.actions[left_global]
                for right_action, right_global in enumerate(
                    env.local_action_catalogs[right]
                ):
                    if right_global is None:
                        continue
                    right_descriptor = env.core.actions[right_global]
                    production = (
                        left_descriptor.kind == "production"
                        and right_descriptor.kind == "production"
                        and str(left_descriptor.job_id)
                        == str(right_descriptor.job_id)
                        and int(left_descriptor.operation_index)
                        == int(right_descriptor.operation_index)
                    )
                    technician = (
                        left_descriptor.kind in {"preventive", "corrective"}
                        and right_descriptor.kind in {"preventive", "corrective"}
                        and str(left_descriptor.technician_id)
                        == str(right_descriptor.technician_id)
                    )
                    if production or technician:
                        tensor[left, right, left_action, right_action] = True
                        tensor[right, left, right_action, left_action] = True
    return tensor


class QPLEXMixer(nn.Module):
    """Duplex-dueling QPLEX mixer with graph-conditioned positive lambdas.

    This follows the original DMAQ construction: state-dependent positive
    transforms produce per-agent values, while a multi-kernel state/action
    network produces strictly positive advantage weights.  The graph is an
    additional soft input to that weight network.  It cannot mask an action or
    remove an agent, and positivity preserves the Individual-Global-Max
    property.
    """

    def __init__(
        self,
        agent_count: int,
        action_count: int,
        state_dim: int,
        hidden_dim: int,
        *,
        kernels: int = 4,
    ) -> None:
        super().__init__()
        self.agent_count = agent_count
        self.action_count = action_count
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.kernels = kernels
        self.hyper_w = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, agent_count),
        )
        self.hyper_b = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, agent_count),
        )
        action_graph_dim = agent_count * action_count + agent_count * agent_count
        self.key_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(kernels)
            ]
        )
        self.agent_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, agent_count),
                )
                for _ in range(kernels)
            ]
        )
        self.action_graph_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(state_dim + action_graph_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, agent_count),
                )
                for _ in range(kernels)
            ]
        )

    def advantage_weights(
        self,
        state: th.Tensor,
        actions_onehot: th.Tensor,
        adjacency: th.Tensor,
    ) -> th.Tensor:
        batch = state.shape[0]
        context = th.cat(
            [
                state,
                actions_onehot.reshape(batch, -1),
                adjacency.to(state.dtype).reshape(batch, -1),
            ],
            dim=-1,
        )
        heads: list[th.Tensor] = []
        for key_net, agent_net, context_net in zip(
            self.key_extractors,
            self.agent_extractors,
            self.action_graph_extractors,
            strict=True,
        ):
            key = key_net(state).abs() + 1e-10
            agent = th.sigmoid(agent_net(state))
            action_graph = th.sigmoid(context_net(context))
            heads.append(key * agent * action_graph)
        return th.stack(heads, dim=1).sum(dim=1) + 1e-10

    def forward(
        self,
        chosen_q: th.Tensor,
        max_q: th.Tensor,
        state: th.Tensor,
        actions_onehot: th.Tensor,
        adjacency: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor]:
        weight = self.hyper_w(state).abs() + 1e-10
        bias = self.hyper_b(state)
        transformed_chosen = weight * chosen_q + bias
        transformed_max = weight * max_q + bias
        # The detached local advantage matches the reference DMAQ learner.  The
        # minus-one form is algebraically V_tot + sum(lambda_i * A_i), while
        # retaining the reference gradient path through the chosen utilities.
        advantage = (transformed_chosen - transformed_max).detach()
        lambdas = self.advantage_weights(state, actions_onehot, adjacency)
        total = transformed_chosen.sum(dim=-1) + (
            advantage * (lambdas - 1.0)
        ).sum(dim=-1)
        return total, lambdas


class QPLEXPolicy(nn.Module):
    def __init__(
        self,
        *,
        graph_variant: str,
        agent_count: int,
        action_count: int,
        local_feature_dim: int,
        global_state_dim: int,
        resource_conflicts: np.ndarray | th.Tensor,
        hidden_dim: int = 128,
        mixer_hidden_dim: int = 64,
        mixer_kernels: int = 4,
    ) -> None:
        super().__init__()
        if graph_variant not in GRAPH_VARIANTS:
            raise ValueError(f"Unknown QPLEX graph variant: {graph_variant}")
        expected = (agent_count, agent_count, action_count, action_count)
        conflict_tensor = th.as_tensor(resource_conflicts, dtype=th.bool)
        if tuple(conflict_tensor.shape) != expected:
            raise ValueError(
                f"Resource-conflict tensor has shape {tuple(conflict_tensor.shape)}, "
                f"expected {expected}"
            )
        if not th.equal(
            conflict_tensor,
            conflict_tensor.permute(1, 0, 3, 2),
        ):
            raise ValueError("Resource-conflict tensor must be symmetric")
        self.graph_variant = graph_variant
        self.agent_count = agent_count
        self.action_count = action_count
        self.local_feature_dim = local_feature_dim
        self.global_state_dim = global_state_dim
        self.hidden_dim = hidden_dim
        self.mixer_hidden_dim = mixer_hidden_dim
        self.mixer_kernels = mixer_kernels
        self.agent_network = _mlp(local_feature_dim, hidden_dim)
        self.mixer = QPLEXMixer(
            agent_count,
            action_count,
            global_state_dim,
            mixer_hidden_dim,
            kernels=mixer_kernels,
        )
        self.register_buffer("resource_conflicts", conflict_tensor)

    def q_values(self, local_observations: th.Tensor) -> th.Tensor:
        return self.agent_network(local_observations).squeeze(-1)

    def graph_adjacency(
        self, q_values: th.Tensor, masks: th.Tensor
    ) -> th.Tensor:
        if q_values.shape != masks.shape or q_values.ndim != 3:
            raise ValueError("Q values and masks must be batch-agent-action tensors")
        batch, agents, actions = q_values.shape
        if agents != self.agent_count or actions != self.action_count:
            raise ValueError("Q-value shape does not match QPLEX policy")
        if self.graph_variant == NULL_GRAPH:
            return th.zeros(
                batch,
                agents,
                agents,
                dtype=th.bool,
                device=q_values.device,
            )
        if self.graph_variant == ALL_FEASIBLE_GRAPH:
            selected = masks
        else:
            masked = q_values.detach().masked_fill(~masks, -th.inf)
            ordered = th.argsort(masked, dim=-1, descending=True, stable=True)
            valid_count = masks.sum(dim=-1, keepdim=True).clamp_max(2)
            ranks = th.arange(actions, device=q_values.device).view(1, 1, -1)
            take_rank = (ranks < valid_count).expand_as(ordered)
            selected = th.zeros_like(masks).scatter(-1, ordered, take_rank)
        adjacency = th.einsum(
            "bia,ijac,bjc->bij",
            selected.to(th.float32),
            self.resource_conflicts.to(th.float32),
            selected.to(th.float32),
        ) > 0
        diagonal = th.eye(agents, dtype=th.bool, device=q_values.device)
        adjacency = adjacency & ~diagonal.unsqueeze(0)
        if not th.equal(adjacency, adjacency.transpose(1, 2)):
            raise AssertionError("Dynamic resource graph must be symmetric")
        return adjacency

    def mix_from_q(
        self,
        q_values: th.Tensor,
        masks: th.Tensor,
        actions: th.Tensor,
        state: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        masked = q_values.masked_fill(~masks, -1e9)
        max_q = masked.max(dim=-1).values
        chosen_q = q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        adjacency = self.graph_adjacency(q_values, masks)
        onehot = nn.functional.one_hot(
            actions, num_classes=self.action_count
        ).to(q_values.dtype)
        total, lambdas = self.mixer(
            chosen_q, max_q, state, onehot, adjacency
        )
        return total, lambdas, adjacency

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
        mask_array = masks.cpu().numpy()
        for batch_index in range(actions.shape[0]):
            for agent_index in range(self.agent_count):
                if generator.random() < epsilon:
                    valid = np.flatnonzero(mask_array[batch_index, agent_index])
                    actions[batch_index, agent_index] = int(generator.choice(valid))
        return actions.squeeze(0) if single else actions

    def mixer_diagnostics(
        self,
        observation: dict[str, np.ndarray],
        actions: np.ndarray,
        *,
        device: str,
    ) -> dict[str, np.ndarray | float]:
        local = th.as_tensor(
            observation["local_observations"], dtype=th.float32, device=device
        ).unsqueeze(0)
        masks = th.as_tensor(
            observation["action_masks"], dtype=th.bool, device=device
        ).unsqueeze(0)
        state = th.as_tensor(
            observation["global_state"], dtype=th.float32, device=device
        ).unsqueeze(0)
        action_tensor = th.as_tensor(
            np.asarray(actions), dtype=th.long, device=device
        ).unsqueeze(0)
        with th.no_grad():
            values = self.q_values(local)
            total, lambdas, adjacency = self.mix_from_q(
                values, masks, action_tensor, state
            )
        return {
            "joint_q": float(total.item()),
            "lambdas": lambdas.squeeze(0).cpu().numpy(),
            "adjacency": adjacency.squeeze(0).cpu().numpy(),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        th.save(
            {
                "state_dict": self.state_dict(),
                "graph_variant": self.graph_variant,
                "agent_count": self.agent_count,
                "action_count": self.action_count,
                "local_feature_dim": self.local_feature_dim,
                "global_state_dim": self.global_state_dim,
                "hidden_dim": self.hidden_dim,
                "mixer_hidden_dim": self.mixer_hidden_dim,
                "mixer_kernels": self.mixer_kernels,
                "resource_conflicts": self.resource_conflicts.cpu(),
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str) -> "QPLEXPolicy":
        payload = th.load(path, map_location=device, weights_only=False)
        model = cls(
            **{
                key: payload[key]
                for key in (
                    "graph_variant",
                    "agent_count",
                    "action_count",
                    "local_feature_dim",
                    "global_state_dim",
                    "hidden_dim",
                    "mixer_hidden_dim",
                    "mixer_kernels",
                    "resource_conflicts",
                )
            }
        ).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        return model


def train_qplex(
    config: BenchmarkConfig,
    settings: ValueLearningSettings,
    output_dir: Path,
    *,
    graph_variant: str,
    train_seed: int,
    show_progress: bool,
    checkpoint_targets: tuple[int, ...],
) -> tuple[QPLEXPolicy, float]:
    if settings.n_envs < 1 or settings.total_timesteps % settings.n_envs:
        raise ValueError("Training budget must be divisible by a positive n_envs")
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
    reference = envs[0]
    conflicts = build_resource_conflict_tensor(reference)
    model = QPLEXPolicy(
        graph_variant=graph_variant,
        agent_count=reference.num_agents,
        action_count=reference.max_local_actions,
        local_feature_dim=reference.local_feature_dim,
        global_state_dim=reference.global_state_dim,
        resource_conflicts=conflicts,
        hidden_dim=settings.hidden_dim,
        mixer_hidden_dim=settings.mixer_hidden_dim,
    ).to(settings.device)
    target = copy.deepcopy(model).to(settings.device).eval()
    optimizer = th.optim.Adam(model.parameters(), lr=settings.learning_rate)
    buffer = ReplayBuffer(settings.replay_capacity, observations[0])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    targets = list(checkpoint_targets)
    if (
        not targets
        or targets != sorted(set(targets))
        or targets[-1] > settings.total_timesteps
    ):
        raise ValueError("Checkpoint targets must be unique, ordered and in budget")

    episode_counts = [0] * settings.n_envs
    episode_returns = [0.0] * settings.n_envs
    episode_lengths = [0] * settings.n_envs
    episode_rows: list[dict[str, float | int]] = []
    update_rows: list[dict[str, float | int]] = []
    pending: dict[str, list[float]] = {
        key: []
        for key in (
            "loss",
            "chosen_q_mean",
            "chosen_q_max",
            "predicted_value_mean",
            "gradient_norm",
            "lambda_mean",
            "lambda_max",
            "graph_density",
        )
    }
    completed_steps = 0
    next_checkpoint = 0
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
        desc=f"QPLEX-{graph_variant} seed {train_seed}",
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
            _stack_observations(observations),
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
                    train_seed + rank + episode_counts[rank] * settings.n_envs
                )
                observations[rank], _ = current_env.reset(seed=reset_seed)
        completed_steps += settings.n_envs
        progress.update(settings.n_envs)

        while next_update <= completed_steps and buffer.size >= settings.batch_size:
            for _ in range(settings.gradient_steps):
                batch = buffer.sample(settings.batch_size, rng)
                local = th.as_tensor(
                    batch["local"], dtype=th.float32, device=settings.device
                )
                masks = th.as_tensor(
                    batch["masks"], dtype=th.bool, device=settings.device
                )
                action_tensor = th.as_tensor(
                    batch["actions"], dtype=th.long, device=settings.device
                )
                rewards = th.as_tensor(
                    batch["rewards"], dtype=th.float32, device=settings.device
                )
                next_local = th.as_tensor(
                    batch["next_local"], dtype=th.float32, device=settings.device
                )
                next_masks = th.as_tensor(
                    batch["next_masks"], dtype=th.bool, device=settings.device
                )
                dones = th.as_tensor(
                    batch["dones"], dtype=th.float32, device=settings.device
                )
                state = th.as_tensor(
                    batch["global"], dtype=th.float32, device=settings.device
                )
                next_state = th.as_tensor(
                    batch["next_global"], dtype=th.float32, device=settings.device
                )
                q_values = model.q_values(local)
                predicted, lambdas, adjacency = model.mix_from_q(
                    q_values, masks, action_tensor, state
                )
                chosen = q_values.gather(
                    -1, action_tensor.unsqueeze(-1)
                ).squeeze(-1)
                with th.no_grad():
                    target_q = target.q_values(next_local)
                    next_actions = target_q.masked_fill(
                        ~next_masks, -1e9
                    ).argmax(dim=-1)
                    next_total, _, _ = target.mix_from_q(
                        target_q, next_masks, next_actions, next_state
                    )
                    target_value = rewards + settings.gamma * (1.0 - dones) * next_total
                loss = nn.functional.smooth_l1_loss(predicted, target_value)
                if not th.isfinite(loss):
                    raise FloatingPointError("Non-finite QPLEX loss")
                optimizer.zero_grad()
                loss.backward()
                gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
                possible_edges = model.agent_count * (model.agent_count - 1)
                pending["loss"].append(float(loss.item()))
                pending["chosen_q_mean"].append(float(chosen.mean().item()))
                pending["chosen_q_max"].append(float(chosen.max().item()))
                pending["predicted_value_mean"].append(float(predicted.mean().item()))
                pending["gradient_norm"].append(float(gradient_norm))
                pending["lambda_mean"].append(float(lambdas.mean().item()))
                pending["lambda_max"].append(float(lambdas.max().item()))
                pending["graph_density"].append(
                    float(adjacency.sum().item() / (adjacency.shape[0] * possible_edges))
                )
            next_update += settings.train_frequency
        while next_update <= completed_steps:
            next_update += settings.train_frequency
        while completed_steps >= next_target_update:
            target.load_state_dict(model.state_dict())
            next_target_update += settings.target_update_interval
        if pending["loss"] and (
            completed_steps >= next_log
            or completed_steps == settings.total_timesteps
        ):
            update_rows.append(
                {
                    "step": completed_steps,
                    **{
                        key: float(np.mean(values))
                        for key, values in pending.items()
                    },
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
        while next_checkpoint < len(targets) and completed_steps >= targets[next_checkpoint]:
            model.save(checkpoints / f"model_{targets[next_checkpoint]}_steps.pt")
            next_checkpoint += 1
    progress.close()
    elapsed = perf_counter() - started
    model.save(output_dir / "model.pt")
    settings_payload = {
        **asdict(settings),
        "graph_variant": graph_variant,
        "mixer_kernels": model.mixer_kernels,
    }
    (output_dir / "settings.json").write_text(
        json.dumps(settings_payload, indent=2, sort_keys=True) + "\n"
    )
    _write_csv(episode_rows, output_dir / "training_episodes.csv")
    _write_csv(update_rows, output_dir / "training_progress.csv")
    for current_env in envs:
        current_env.close()
    return model, elapsed
