from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.technician_capacity_screening import build_condition_config
from ht_pdm_fjsp.value_decomposition import (
    QMixer,
    ValueDecompositionPolicy,
    ValueLearningSettings,
    train_value_policy,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def _config() -> BenchmarkConfig:
    base = BenchmarkConfig.from_json(CONFIG_PATH)
    return build_condition_config(base, topology="two_specialists", multiplier=2.0)


def test_iql_and_qmix_respect_action_masks() -> None:
    env = MachineAgentsCTDEEnv(_config(), wait_policy="safe_noop")
    observation, _ = env.reset(seed=1)
    for algorithm in ("iql", "qmix"):
        policy = ValueDecompositionPolicy(
            algorithm=algorithm,
            agent_count=env.num_agents,
            local_feature_dim=env.local_feature_dim,
            global_state_dim=env.global_state_dim,
        )
        actions = policy.act(observation, deterministic=True, device="cpu")
        assert all(
            observation["action_masks"][agent, action]
            for agent, action in enumerate(actions)
        )


def test_iql_and_qmix_batch_action_inference_respects_masks() -> None:
    envs = [MachineAgentsCTDEEnv(_config(), wait_policy="safe_noop") for _ in range(4)]
    observations = [env.reset(seed=index)[0] for index, env in enumerate(envs)]
    batched = {
        key: np.stack([observation[key] for observation in observations])
        for key in observations[0]
    }
    for algorithm in ("iql", "qmix"):
        policy = ValueDecompositionPolicy(
            algorithm=algorithm,
            agent_count=envs[0].num_agents,
            local_feature_dim=envs[0].local_feature_dim,
            global_state_dim=envs[0].global_state_dim,
        )
        actions = policy.act(batched, deterministic=True, device="cpu")
        assert actions.shape == (4, envs[0].num_agents)
        assert all(
            batched["action_masks"][rank, agent, action]
            for rank, row in enumerate(actions)
            for agent, action in enumerate(row)
        )
    for env in envs:
        env.close()


def test_qmix_mixer_is_monotonic_in_agent_values() -> None:
    th.manual_seed(1)
    mixer = QMixer(agent_count=3, state_dim=5, hidden_dim=8)
    state = th.randn(4, 5)
    low = th.randn(4, 3)
    high = low + th.rand(4, 3)
    assert th.all(mixer(high, state) >= mixer(low, state) - 1e-6)


def test_tiny_value_training_saves_loadable_models(tmp_path: Path) -> None:
    settings = ValueLearningSettings(
        total_timesteps=40,
        replay_capacity=40,
        learning_starts=8,
        batch_size=4,
        train_frequency=2,
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
    for algorithm in ("iql", "qmix"):
        output = tmp_path / algorithm
        train_value_policy(
            _config(),
            settings,
            output,
            algorithm=algorithm,
            train_seed=7,
            show_progress=False,
        )
        loaded = ValueDecompositionPolicy.load(output / "model.pt", device="cpu")
        assert loaded.algorithm == algorithm
        assert len(list((output / "checkpoints").glob("*.pt"))) == 5
