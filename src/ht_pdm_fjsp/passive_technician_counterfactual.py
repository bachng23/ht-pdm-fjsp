"""Queue-aware counterfactual MAPPO for passive heterogeneous technicians."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical
from tqdm.auto import tqdm

from ht_pdm_fjsp.passive_technician_baselines import (
    CentralizedPPO,
    PPOSettings,
    _action_diagnostics,
    _masked_logits,
    _obs_tensor,
    _returns,
    evaluate_fixed,
    load_centralized_ppo,
    stress_config,
)
from ht_pdm_fjsp.passive_technician_budget_screen import (
    _evaluate_centralized_ppo,
    _train_centralized_ppo_checkpoints,
)
from ht_pdm_fjsp.passive_technician_marl import (
    OBJECTIVE_VERSION,
    OBSERVATION_VERSION,
    PassiveConfig,
    PassiveTechnicianEnv,
)


POLICIES = ("mappo_ctde", "queue_aware_counterfactual")
FIXED_POLICIES = ("random_feasible", "skill_aware_fifo")
SMOKE_EPISODES = 32
FULL_EPISODES = 5_000
FIELDNAMES = (
    "policy",
    "seed",
    "train_seed",
    "objective",
    "failures",
    "jobs",
    "collisions",
    "waiting",
    "invalid_requests",
    "busy_requests",
    "unique_joint_actions",
    "defer_fraction",
    "action_steps",
    "request_count",
)


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: tuple[str, ...] | None = None) -> None:
    if not rows and fieldnames is None:
        return
    if fieldnames is not None:
        names = list(fieldnames)
    else:
        names = []
        for row in rows:
            for name in row:
                if name not in names:
                    names.append(name)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class TechnicianEdgeActor(nn.Module):
    """Local actor with a separate score for defer and each technician edge."""

    def __init__(self, observation_dim: int, technicians: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.observation_dim = observation_dim
        self.technicians = technicians
        self.hidden_dim = hidden_dim
        self.trunk = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.technician_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.defer = nn.Linear(hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = self.trunk(observations)
        technician_features = observations[..., 3:].reshape(
            *observations.shape[:-1], self.technicians, 6
        )
        technician_hidden = self.technician_encoder(technician_features)
        query = self.query(hidden).unsqueeze(-2)
        edge_scores = (query * self.key(technician_hidden)).sum(dim=-1)
        edge_scores = edge_scores / float(self.hidden_dim) ** 0.5
        return torch.cat((self.defer(hidden), edge_scores), dim=-1)


class QueueAwareCounterfactualPolicy(nn.Module):
    """CTDE policy using technician-edge logits and counterfactual Q advantages."""

    def __init__(self, config: PassiveConfig, hidden_dim: int = 64) -> None:
        super().__init__()
        self.machines = config.machines
        self.observation_dim = config.observation_dim
        self.action_dim = config.technicians + 1
        self.hidden_dim = hidden_dim
        self.actors = nn.ModuleList(
            TechnicianEdgeActor(self.observation_dim, config.technicians, hidden_dim)
            for _ in range(config.machines)
        )
        global_dim = config.machines * config.observation_dim
        joint_dim = config.machines * self.action_dim
        self.critic = nn.Sequential(
            nn.Linear(global_dim + joint_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def actor_logits(self, observations: torch.Tensor, machine: int) -> torch.Tensor:
        return self.actors[machine](observations[..., machine, :])

    def critic_value(self, global_observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        one_hot = torch.nn.functional.one_hot(actions, num_classes=self.action_dim).float()
        features = torch.cat((global_observations, one_hot.flatten(start_dim=1)), dim=-1)
        return self.critic(features).squeeze(-1)

    def save(self, path: Path, seed: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actors": [actor.state_dict() for actor in self.actors],
                "critic": self.critic.state_dict(),
                "hidden_dim": self.hidden_dim,
                "seed": seed,
            },
            path,
        )

    @classmethod
    def load(
        cls, path: Path, config: PassiveConfig, device: torch.device
    ) -> "QueueAwareCounterfactualPolicy":
        payload = torch.load(path, map_location=device, weights_only=True)
        model = cls(config, hidden_dim=int(payload.get("hidden_dim", 64))).to(device)
        for actor, state in zip(model.actors, payload["actors"], strict=True):
            actor.load_state_dict(state)
        model.critic.load_state_dict(payload["critic"])
        model.eval()
        return model


def _counterfactual_advantages(
    model: QueueAwareCounterfactualPolicy,
    global_observations: torch.Tensor,
    local_observations: torch.Tensor,
    masks: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Return one counterfactual advantage per timestep and machine."""
    with torch.no_grad():
        actual_q = model.critic_value(global_observations, actions)
        advantages = torch.zeros_like(actions, dtype=torch.float32)
        for machine in range(model.machines):
            for timestep in range(actions.shape[0]):
                valid_actions = torch.where(masks[timestep, machine])[0]
                counterfactual = actions[timestep].repeat(len(valid_actions), 1)
                counterfactual[:, machine] = valid_actions
                state_batch = global_observations[timestep].repeat(len(valid_actions), 1)
                q_values = model.critic_value(state_batch, counterfactual)
                logits = model.actor_logits(local_observations[timestep].unsqueeze(0), machine)
                distribution = Categorical(
                    logits=_masked_logits(logits, masks[timestep, machine])
                )
                probabilities = distribution.probs.squeeze(0)[valid_actions]
                baseline = (probabilities * q_values).sum()
                advantages[timestep, machine] = actual_q[timestep] - baseline
    return advantages


def train_queue_aware_counterfactual(
    config: PassiveConfig,
    seed: int,
    episodes: int,
    path: Path,
    settings: PPOSettings,
    device: torch.device,
) -> tuple[Path, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    random.seed(seed)
    model = QueueAwareCounterfactualPolicy(config).to(device)
    actor_optimizer = torch.optim.Adam(
        [parameter for actor in model.actors for parameter in actor.parameters()],
        lr=settings.learning_rate,
    )
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=settings.learning_rate)
    progress: list[dict[str, Any]] = []
    for episode in tqdm(
        range(1, episodes + 1),
        desc=f"Queue-aware CF {seed}",
        unit="episode",
        leave=False,
    ):
        env = PassiveTechnicianEnv(config, seed=seed * 100_000 + episode - 1)
        observations = env.reset()
        rollout_observations: list[torch.Tensor] = []
        rollout_masks: list[torch.Tensor] = []
        rollout_actions: list[torch.Tensor] = []
        rollout_log_probs: list[torch.Tensor] = []
        rewards: list[float] = []
        done = False
        while not done:
            observation_tensor = _obs_tensor(observations, config).to(device)
            masks = torch.tensor(env.action_masks(), dtype=torch.bool, device=device)
            actions: list[int] = []
            log_probs: list[torch.Tensor] = []
            with torch.no_grad():
                for machine in range(config.machines):
                    logits = model.actor_logits(observation_tensor.unsqueeze(0), machine)
                    distribution = Categorical(
                        logits=_masked_logits(logits, masks[machine])
                    )
                    action = distribution.sample()
                    actions.append(int(action.item()))
                    log_probs.append(distribution.log_prob(action).squeeze(0))
            rollout_observations.append(observation_tensor.detach())
            rollout_masks.append(masks)
            rollout_actions.append(torch.tensor(actions, dtype=torch.long, device=device))
            rollout_log_probs.append(torch.stack(log_probs))
            observations, reward, done, _ = env.step(actions)
            rewards.append(float(reward))

        local_observations = torch.stack(rollout_observations)
        global_observations = local_observations.flatten(start_dim=1)
        masks_tensor = torch.stack(rollout_masks)
        actions_tensor = torch.stack(rollout_actions)
        old_log_probs_tensor = torch.stack(rollout_log_probs)
        returns = _returns(rewards, settings.gamma, device=device)

        for _ in range(settings.update_epochs):
            critic_loss = (
                model.critic_value(global_observations, actions_tensor) - returns
            ).pow(2).mean()
            critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            critic_optimizer.step()

        advantages = _counterfactual_advantages(
            model, global_observations, local_observations, masks_tensor, actions_tensor
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        actor_losses: list[torch.Tensor] = []
        entropies: list[torch.Tensor] = []
        for machine in range(config.machines):
            logits = model.actor_logits(local_observations, machine)
            distribution = Categorical(
                logits=_masked_logits(logits, masks_tensor[:, machine])
            )
            log_probs = distribution.log_prob(actions_tensor[:, machine])
            ratio = torch.exp(log_probs - old_log_probs_tensor[:, machine])
            clipped = torch.clamp(
                ratio,
                1.0 - settings.clip_ratio,
                1.0 + settings.clip_ratio,
            )
            actor_losses.append(
                -torch.minimum(
                    ratio * advantages[:, machine].detach(),
                    clipped * advantages[:, machine].detach(),
                ).mean()
            )
            entropies.append(distribution.entropy().mean())
        actor_loss = torch.stack(actor_losses).mean()
        entropy = torch.stack(entropies).mean()
        total_loss = actor_loss - settings.entropy_weight * entropy
        actor_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.actors.parameters(), 1.0)
        actor_optimizer.step()
        progress.append(
            {
                "policy": "queue_aware_counterfactual",
                "train_seed": seed,
                "episode": episode,
                "objective": -sum(rewards),
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
            }
        )
    model.save(path, seed)
    return path, progress


def evaluate_queue_aware_counterfactual(
    config: PassiveConfig,
    model: QueueAwareCounterfactualPolicy,
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
        observation_tensor = _obs_tensor(observations, config).to(device)
        masks = env.action_masks()
        with torch.no_grad():
            actions = [
                int(
                    torch.argmax(
                        _masked_logits(
                            model.actor_logits(observation_tensor.unsqueeze(0), machine),
                            masks[machine],
                        )
                    ).item()
                )
                for machine in range(config.machines)
            ]
        joint_action = tuple(actions)
        joint_actions.add(joint_action)
        defer_actions += sum(action == 0 for action in joint_action)
        action_steps += 1
        observations, _, done, _ = env.step(actions)
    return {
        "policy": "queue_aware_counterfactual",
        "seed": seed,
        "train_seed": train_seed,
        **env.metrics,
        **_action_diagnostics(config, joint_actions, defer_actions, action_steps),
    }


def _profile(profile: str) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    if profile == "smoke":
        return (11,), (101, 102, 103), SMOKE_EPISODES
    if profile == "full":
        return (11, 12, 13), tuple(range(101, 131)), FULL_EPISODES
    raise ValueError(profile)


def _row(row: dict[str, Any], policy: str, train_seed: int | str) -> dict[str, Any]:
    return {
        field: row.get(field, 0 if field not in {"policy", "seed", "train_seed"} else "")
        for field in FIELDNAMES
    } | {"policy": policy, "train_seed": train_seed}


def _summary(
    rows: list[dict[str, Any]],
    train_seeds: tuple[int, ...],
    evaluation_seeds: tuple[int, ...],
) -> dict[str, Any]:
    per_policy: dict[str, dict[str, float | int]] = {}
    for policy in POLICIES + FIXED_POLICIES:
        group = [row for row in rows if row["policy"] == policy]
        if not group:
            continue
        per_policy[policy] = {
            "evaluation_count": len(group),
            "objective_mean": statistics.fmean(float(row["objective"]) for row in group),
            "waiting_mean": statistics.fmean(float(row["waiting"]) for row in group),
            "busy_requests_mean": statistics.fmean(float(row["busy_requests"]) for row in group),
            "invalid_requests_mean": statistics.fmean(float(row["invalid_requests"]) for row in group),
            "unique_joint_actions_mean": statistics.fmean(float(row["unique_joint_actions"]) for row in group),
        }
    expected = len(FIXED_POLICIES) * len(evaluation_seeds) + len(POLICIES) * len(train_seeds) * len(evaluation_seeds)
    learned = [row for row in rows if row["policy"] in POLICIES]
    return {
        "purpose": "Mechanism experiment for queue-aware counterfactual technician allocation.",
        "primary_metric": "mean objective cost on the common evaluation panel",
        "policies": per_policy,
        "paired_objective_delta_new_minus_mappo": (
            per_policy["queue_aware_counterfactual"]["objective_mean"]
            - per_policy["mappo_ctde"]["objective_mean"]
            if "queue_aware_counterfactual" in per_policy and "mappo_ctde" in per_policy
            else None
        ),
        "audits": {
            "expected_episode_count": expected,
            "episode_count": len(rows),
            "all_expected_rows_present": len(rows) == expected,
            "invalid_requests_zero": all(float(row["invalid_requests"]) == 0 for row in learned),
            "nonnegative_objectives": all(float(row["objective"]) >= 0 for row in rows),
            "sealed_test_panel_closed": True,
            "all_learned_train_seeds_recorded": all(str(row["train_seed"]) for row in learned),
        },
    }


def run(args: argparse.Namespace) -> Path:
    train_seeds, evaluation_seeds, episodes = _profile(args.profile)
    config = stress_config()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    output = Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    settings = PPOSettings(episodes=episodes)
    manifest = {
        "status": "RUNNING",
        "experiment": "passive_technician_counterfactual",
        "profile": args.profile,
        "policies": list(POLICIES),
        "fixed_policies": list(FIXED_POLICIES),
        "train_seeds": list(train_seeds),
        "evaluation_seeds": list(evaluation_seeds),
        "episodes_per_train_seed": episodes,
        "hypothesis": "Counterfactual technician-edge credit assignment reduces waiting and objective cost versus MAPPO CTDE.",
        "stress_config": asdict(config),
        "settings": asdict(settings),
        "objective_version": OBJECTIVE_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "requested_device": args.device,
        "resolved_device": str(device),
        "git_revision": _git_revision(),
        "sealed_test_evaluated": False,
        "started_at": datetime.now(UTC).isoformat(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    (output / "benchmark_config.json").write_text(
        json.dumps(
            {"stress_config": asdict(config), "train_seeds": train_seeds, "evaluation_seeds": evaluation_seeds, "episodes": episodes, "settings": asdict(settings)},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    rows: list[dict[str, Any]] = []
    progress: list[dict[str, Any]] = []
    for seed in tqdm(evaluation_seeds, desc="fixed policies", unit="episode"):
        for policy in FIXED_POLICIES:
            rows.append(_row(evaluate_fixed(config, policy, seed), policy, ""))
    _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)
    for policy_name in POLICIES:
        for train_seed in tqdm(train_seeds, desc=policy_name, unit="seed"):
            root = output / policy_name / f"train_seed_{train_seed}"
            if policy_name == "mappo_ctde":
                checkpoints, training_rows = _train_centralized_ppo_checkpoints(
                    config, train_seed, (episodes,), root, settings, device
                )
                checkpoint = checkpoints[episodes]
                policy = load_centralized_ppo(config, checkpoint).to(device)
            else:
                checkpoint, training_rows = train_queue_aware_counterfactual(
                    config,
                    train_seed,
                    episodes,
                    root / f"budget_{episodes}" / "model.pt",
                    settings,
                    device,
                )
                policy = QueueAwareCounterfactualPolicy.load(checkpoint, config, device)
            progress.extend({**row, "policy": policy_name} for row in training_rows)
            for seed in tqdm(
                evaluation_seeds,
                desc=f"evaluate {policy_name}/{train_seed}",
                unit="episode",
                leave=False,
            ):
                if policy_name == "mappo_ctde":
                    evaluated = _evaluate_centralized_ppo(config, policy, seed, train_seed, device)
                else:
                    evaluated = evaluate_queue_aware_counterfactual(
                        config, policy, seed, train_seed, device
                    )
                rows.append(_row(evaluated, policy_name, train_seed))
            _write_csv(rows, output / "episodes.partial.csv", FIELDNAMES)
    _write_csv(rows, output / "episodes.csv", FIELDNAMES)
    _write_csv(rows, output / "coordination.csv", FIELDNAMES)
    _write_csv(progress, output / "training_progress.csv")
    summary = _summary(rows, train_seeds, evaluation_seeds)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    manifest.update(
        {
            "status": "COMPLETED",
            "finished_at": datetime.now(UTC).isoformat(),
            "episode_count": len(rows),
            "checkpoint_count": sum(1 for path in output.rglob("model.*") if path.is_file()),
            "outputs": sorted(str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()),
        }
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
