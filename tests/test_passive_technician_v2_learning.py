from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

import ht_pdm_fjsp.passive_technician_v2_learning as experiment
from ht_pdm_fjsp.passive_technician_v2_learning import run
from ht_pdm_fjsp.passive_technician_v2_rl import (
    ALGORITHMS,
    PassivePPOPolicy,
    PassivePPOSettings,
    PassiveV2MultiAgentEnv,
    evaluate_model,
    reservation_aware_action,
    selected_learning_config,
    train_policy,
)


def tiny_settings() -> PassivePPOSettings:
    return PassivePPOSettings(
        total_timesteps=256,
        n_envs=4,
        n_steps=32,
        batch_size=64,
        n_epochs=1,
        actor_hidden_dim=16,
        critic_hidden_dim=24,
        device="cpu",
    )


def test_selected_config_matches_locked_calibration_candidate() -> None:
    config = selected_learning_config()
    assert config.machines == 4
    assert config.technicians == 2
    assert config.horizon == 36
    assert config.initial_ages == (0, 0, 0, 0)
    assert config.service_time == ((2, 5), (2, 3), (3, 2), (5, 2))
    assert config.eligibility == (
        (True, False),
        (True, True),
        (True, True),
        (False, True),
    )


def test_multi_agent_adapter_shapes_masks_and_reward_identity() -> None:
    config = selected_learning_config()
    adapter = PassiveV2MultiAgentEnv(config)
    observation = adapter.reset(seed=7)
    assert observation["local_observations"].shape == (
        config.machines,
        config.observation_dim,
    )
    assert observation["global_state"].shape == (
        config.machines * config.observation_dim,
    )
    assert observation["action_masks"].shape == (
        config.machines,
        config.action_count,
    )
    episode_return = 0.0
    done = False
    while not done:
        actions = np.zeros(config.machines, dtype=np.int64)
        observation, reward, done, _ = adapter.step(actions)
        episode_return += reward
    assert episode_return == -adapter.env.metrics["objective"]


def test_actor_and_critic_information_boundaries() -> None:
    config = selected_learning_config()
    adapter = PassiveV2MultiAgentEnv(config)
    observation = adapter.reset(seed=1)
    local = torch.as_tensor(observation["local_observations"]).unsqueeze(0)
    global_state = torch.as_tensor(observation["global_state"]).unsqueeze(0)
    changed_global = global_state + 10.0

    for algorithm in ("ps_ippo", "mappo"):
        model = PassivePPOPolicy(
            algorithm=algorithm,
            agent_count=config.machines,
            action_count=config.action_count,
            local_feature_dim=config.observation_dim,
            global_state_dim=config.machines * config.observation_dim,
        )
        assert torch.equal(
            model.logits(local, global_state), model.logits(local, changed_global)
        )
        values = model.value(local, global_state)
        assert values.shape == ((1, config.machines) if algorithm == "ps_ippo" else (1,))

    centralized = PassivePPOPolicy(
        algorithm="centralized_ppo",
        agent_count=config.machines,
        action_count=config.action_count,
        local_feature_dim=config.observation_dim,
        global_state_dim=config.machines * config.observation_dim,
    )
    assert not torch.equal(
        centralized.logits(local, global_state),
        centralized.logits(local, changed_global),
    )


def test_reservation_policy_protects_inflexible_machine_capacity() -> None:
    config = selected_learning_config()
    adapter = PassiveV2MultiAgentEnv(config)
    adapter.reset(seed=1)
    for _ in range(config.failure_age - 2):
        adapter.step((0, 0, 0, 0))
    assert reservation_aware_action(adapter.env) == (1, 1, 2, 2)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_tiny_training_saves_loads_and_evaluates(
    tmp_path: Path, algorithm: str
) -> None:
    config = selected_learning_config()
    root = tmp_path / algorithm
    _, checkpoints, _, _, elapsed = train_policy(
        config,
        tiny_settings(),
        root,
        algorithm=algorithm,
        train_seed=11000,
        show_progress=False,
    )
    assert elapsed >= 0.0
    assert len(checkpoints) == 5
    assert (root / "final_model.pt").is_file()
    checkpoint_target, checkpoint_data = next(iter(checkpoints.items()))
    checkpoint_path, checkpoint_actual = checkpoint_data
    loaded = PassivePPOPolicy.load(checkpoint_path, device="cpu")
    rows = evaluate_model(
        loaded,
        config,
        (9130, 9131),
        policy_name=algorithm,
        split="test",
        train_seed=11000,
        checkpoint_target_steps=checkpoint_target,
        checkpoint_actual_steps=checkpoint_actual,
        device="cpu",
        show_progress=False,
    )
    assert len(rows) == 2
    assert all(row["invalid_actions"] == 0 for row in rows)
    assert all(row["identity_error"] <= 1e-9 for row in rows)


def test_smoke_runner_writes_complete_artifacts(tmp_path: Path, monkeypatch) -> None:
    settings = tiny_settings()

    def small_profile(name: str, device: str):
        assert name == "smoke"
        assert device == "cpu"
        return (11000,), (9100,), (9130,), settings

    monkeypatch.setattr(experiment, "profile", small_profile)
    output = tmp_path / "passive_v2_learning_smoke_20260925T000000Z"
    result = run(Namespace(profile="smoke", device="cpu", output_dir=str(output)))
    assert result == output.resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["hard_gate_passed"] is True
    assert manifest["reporting_episode_count"] == 7
    assert manifest["checkpoint_evaluation_count"] == 15
    assert manifest["selection_count"] == 3
    assert manifest["checkpoint_count"] == 15
    assert manifest["sealed_test_evaluated"] is False
    assert summary["audits"]["invalid_actions"] == 0
    assert summary["audits"]["reload_action_mismatches"] == 0
    for name in (
        "training_progress.csv",
        "training_episodes.csv",
        "checkpoint_evaluations.csv",
        "checkpoint_selection.csv",
        "episodes.partial.csv",
        "episodes.csv",
        "coordination.csv",
        "resolved_config.json",
    ):
        assert (output / name).is_file()
