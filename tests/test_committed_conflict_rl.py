from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from ht_pdm_fjsp.centralized_joint_dqn import (
    CentralizedJointDQNPolicy,
    train_centralized_joint_dqn,
)
from ht_pdm_fjsp.committed_conflict_rl import COMMITTED_CELL, CommittedConflictCTDEEnv
from ht_pdm_fjsp.committed_conflict_rl_experiment import ALGORITHMS, run
from ht_pdm_fjsp.conflict_consequence import (
    SEALED_TEST_SEEDS,
    ConflictConsequenceEnv,
    build_cell_config,
)
from ht_pdm_fjsp.parallel_maintenance import ParallelMaintenanceConfig
from ht_pdm_fjsp.value_decomposition import ValueLearningSettings


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "parallel_maintenance.json"


def _base() -> ParallelMaintenanceConfig:
    return ParallelMaintenanceConfig.from_json(CONFIG_PATH)


def _tiny_settings() -> ValueLearningSettings:
    return ValueLearningSettings(
        total_timesteps=40,
        replay_capacity=40,
        learning_starts=8,
        batch_size=4,
        train_frequency=4,
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_interval=8,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_fraction=0.5,
        hidden_dim=16,
        mixer_hidden_dim=8,
        device="cpu",
        n_envs=4,
    )


def test_wrapper_observation_is_finite_and_keeps_conflicts_reachable() -> None:
    env = CommittedConflictCTDEEnv(_base())
    observation, _ = env.reset(seed=1)
    assert observation["local_observations"].shape == (
        env.num_agents,
        env.action_count,
        env.local_feature_dim,
    )
    assert observation["global_state"].shape == (env.global_state_dim,)
    assert np.isfinite(observation["local_observations"]).all()
    assert observation["action_masks"][:, 1].all()
    env.core.age[:] = 120.0
    _, reward, _, _, info = env.step(np.ones(env.num_agents, dtype=np.int64))
    assert np.isfinite(reward)
    assert info["proposal_conflicts"] == env.num_agents - 1
    assert info["committed_wait_steps"] > 0
    assert info["return_objective_error"] == pytest.approx(0.0)


def test_wrapper_matches_validated_kernel_for_identical_actions() -> None:
    base = _base()
    raw = ConflictConsequenceEnv(build_cell_config(base, COMMITTED_CELL), COMMITTED_CELL)
    wrapped = CommittedConflictCTDEEnv(base)
    raw.reset(seed=17)
    observation, _ = wrapped.reset(seed=17)
    rng = np.random.default_rng(99)
    for _ in range(40):
        actions = np.asarray(
            [
                rng.choice(np.flatnonzero(observation["action_masks"][agent]))
                for agent in range(wrapped.num_agents)
            ],
            dtype=np.int64,
        )
        raw_reward, raw_done, raw_info = raw.step(actions)
        observation, reward, terminated, truncated, info = wrapped.step(actions)
        assert reward == pytest.approx(raw_reward)
        assert terminated == raw_done
        assert not truncated
        for key in (
            "objective",
            "production",
            "downtime",
            "failures",
            "waiting",
            "proposal_conflicts",
            "rework",
        ):
            assert info[key] == pytest.approx(raw_info[key])
        assert np.array_equal(raw.action_masks(), observation["action_masks"])
        if terminated:
            break


def test_centralized_joint_policy_masks_only_individual_infeasibility() -> None:
    env = CommittedConflictCTDEEnv(_base())
    observation, _ = env.reset(seed=3)
    policy = CentralizedJointDQNPolicy(
        agent_count=env.num_agents,
        action_count=env.action_count,
        global_state_dim=env.global_state_dim,
        hidden_dim=16,
    )
    valid = policy.valid_joint_masks(observation["action_masks"])
    conflicting = np.ones(env.num_agents, dtype=np.int64)
    index = int(policy.actions_to_indices(conflicting))
    assert np.array_equal(policy.joint_actions[index], conflicting)
    assert valid[index]
    actions = policy.act(observation, deterministic=True, device="cpu")
    assert all(
        observation["action_masks"][agent, action]
        for agent, action in enumerate(actions)
    )


def test_tiny_centralized_training_saves_loadable_checkpoints(tmp_path: Path) -> None:
    output = tmp_path / "centralized"
    train_centralized_joint_dqn(
        _tiny_settings(),
        output,
        train_seed=5,
        show_progress=False,
        checkpoint_targets=(20, 40),
        env_factory=lambda: CommittedConflictCTDEEnv(_base()),
    )
    loaded = CentralizedJointDQNPolicy.load(
        output / "checkpoints" / "model_40_steps.pt", device="cpu"
    )
    observation, _ = CommittedConflictCTDEEnv(_base()).reset(seed=6)
    actions = loaded.act(observation, deterministic=True, device="cpu")
    assert actions.shape == (6,)
    assert (output / "model.pt").is_file()


def test_tiny_runner_writes_complete_audited_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ht_pdm_fjsp.committed_conflict_rl_experiment as experiment

    protocol = {
        "total_timesteps": 40,
        "train_seeds": (91_900,),
        "evaluation_seeds": (64_390,),
        "n_envs": 4,
        "replay_capacity": 40,
        "learning_starts": 8,
        "batch_size": 4,
        "train_frequency": 4,
        "target_update_interval": 8,
        "checkpoints": (20, 40),
    }
    monkeypatch.setattr(experiment, "_profile", lambda _: protocol)
    output = tmp_path / "tiny_runner"
    run(
        argparse.Namespace(
            config=str(CONFIG_PATH),
            profile="smoke",
            device="cpu",
            output_dir=str(output),
        )
    )
    manifest = json.loads((output / "committed_rl_manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    assert manifest["status"] == "COMPLETED"
    assert manifest["learned_episode_count"] == len(ALGORITHMS) * 2
    assert manifest["heuristic_episode_count"] == 3
    assert manifest["decision_count"] == len(ALGORITHMS) * 2 * 168
    assert len(manifest["completed_training_cells"]) == len(ALGORITHMS)
    assert summary["gate"]["passed"]
    assert not manifest["sealed_test_evaluated"]


def test_runner_rejects_sealed_panel_before_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ht_pdm_fjsp.committed_conflict_rl_experiment as experiment

    protocol = {
        **experiment._profile("smoke"),
        "evaluation_seeds": (SEALED_TEST_SEEDS[0],),
    }
    monkeypatch.setattr(experiment, "_profile", lambda _: protocol)
    output = tmp_path / "sealed"
    with pytest.raises(ValueError, match="sealed future test panel"):
        run(
            argparse.Namespace(
                config=str(CONFIG_PATH),
                profile="smoke",
                device="cpu",
                output_dir=str(output),
            )
        )
    assert not output.exists()
