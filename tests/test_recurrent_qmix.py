from __future__ import annotations

from pathlib import Path

import numpy as np

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.recurrent_qmix import (
    EpisodeReplayBuffer,
    RecurrentQMIXPolicy,
    RecurrentQMIXSettings,
    train_recurrent_qmix,
)
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def _config() -> BenchmarkConfig:
    base = BenchmarkConfig.from_json(CONFIG_PATH)
    return build_condition_config(base, topology="two_specialists", multiplier=2.0)


def test_recurrent_policy_respects_masks_and_round_trips(tmp_path: Path) -> None:
    env = MachineAgentsCTDEEnv(_config(), wait_policy="safe_noop")
    observation, _ = env.reset(seed=10)
    policy = RecurrentQMIXPolicy(
        agent_count=env.num_agents,
        local_feature_dim=env.local_feature_dim,
        global_state_dim=env.global_state_dim,
        hidden_dim=16,
        mixer_hidden_dim=8,
        agent_embedding_dim=4,
    )
    policy.reset_hidden(batch_size=1, device="cpu")
    actions = policy.act(observation, deterministic=True, device="cpu")
    assert all(
        observation["action_masks"][agent, action]
        for agent, action in enumerate(actions)
    )
    _, _, _, _, info = env.step(actions)
    policy.observe_outcome(actions, info["agent_outcomes"])
    path = tmp_path / "policy.pt"
    policy.save(path)
    loaded = RecurrentQMIXPolicy.load(path, device="cpu")
    assert loaded.agent_count == env.num_agents
    assert loaded.hidden_dim == 16


def test_episode_replay_pads_and_marks_valid_transitions() -> None:
    transition = {
        "local": np.ones((2, 3, 4), dtype=np.float32),
        "masks": np.ones((2, 3), dtype=np.bool_),
        "global": np.ones(5, dtype=np.float32),
        "previous_action": np.zeros((2, 4), dtype=np.float32),
        "previous_outcome": np.zeros(2, dtype=np.float32),
        "actions": np.zeros(2, dtype=np.int64),
        "reward": 1.0,
        "done": 1.0,
        "next_local": np.ones((2, 3, 4), dtype=np.float32),
        "next_masks": np.ones((2, 3), dtype=np.bool_),
        "next_global": np.ones(5, dtype=np.float32),
        "next_previous_action": np.ones((2, 4), dtype=np.float32),
        "next_previous_outcome": np.ones(2, dtype=np.float32),
    }
    replay = EpisodeReplayBuffer(capacity=10)
    replay.add([transition])
    batch = replay.sample(2, np.random.default_rng(1))
    assert batch["local"].shape == (2, 2, 2, 3, 4)
    assert batch["valid"].sum() == 2


def test_tiny_recurrent_training_saves_locked_checkpoints(tmp_path: Path) -> None:
    settings = RecurrentQMIXSettings(
        total_timesteps=60,
        replay_capacity=60,
        learning_starts=16,
        episode_batch_size=1,
        train_frequency=4,
        gradient_steps=1,
        learning_rate=3e-4,
        gamma=0.99,
        target_update_interval=16,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_fraction=0.5,
        hidden_dim=16,
        mixer_hidden_dim=8,
        agent_embedding_dim=4,
        checkpoint_targets=(20, 40, 60),
        device="cpu",
        n_envs=4,
    )
    output = tmp_path / "recurrent"
    train_recurrent_qmix(
        _config(), settings, output, train_seed=11, show_progress=False
    )
    assert len(list((output / "checkpoints").glob("*.pt"))) == 3
    loaded = RecurrentQMIXPolicy.load(output / "model.pt", device="cpu")
    assert loaded.hidden_dim == 16
