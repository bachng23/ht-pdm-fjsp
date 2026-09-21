from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.ctde_mappo import (
    MAPPOSettings,
    MachineMAPPO,
    evaluate_mappo,
    train_mappo,
)
from ht_pdm_fjsp.models import BenchmarkConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs/minimal_benchmark.json")


def _settings() -> MAPPOSettings:
    return MAPPOSettings(
        total_timesteps=64,
        n_envs=1,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        max_grad_norm=0.5,
        actor_hidden_dim=32,
        critic_hidden_dim=64,
        device="cpu",
    )


def test_actor_distribution_uses_local_tensor_only() -> None:
    env = MachineAgentsCTDEEnv(CONFIG)
    observation, _ = env.reset(seed=7)
    model = MachineMAPPO(
        local_feature_dim=env.LOCAL_FEATURE_DIM,
        global_state_dim=env.global_state_dim,
    )
    local = th.as_tensor(observation["local_observations"]).unsqueeze(0)
    masks = th.as_tensor(observation["action_masks"]).unsqueeze(0)
    first = model.distribution(local, masks).logits
    changed_global = observation["global_state"].copy()
    changed_global[:] = 123.0
    second = model.distribution(local, masks).logits
    assert th.equal(first, second)
    assert model.critic[0].in_features == env.global_state_dim
    assert model.actor[0].in_features == env.LOCAL_FEATURE_DIM


def test_mappo_trains_saves_loads_and_evaluates(tmp_path: Path) -> None:
    output = tmp_path / "cell"
    model, elapsed = train_mappo(
        CONFIG,
        _settings(),
        output,
        train_seed=10_000,
        show_progress=False,
    )
    assert elapsed >= 0
    assert (output / "mappo.pt").is_file()
    assert len(list((output / "checkpoints").glob("*.pt"))) == 5
    loaded = MachineMAPPO.load(output / "mappo.pt", device="cpu")
    rows, audit = evaluate_mappo(
        loaded,
        CONFIG,
        [52_000, 52_001],
        condition="machine_agents_mappo_ctde",
        device="cpu",
        show_progress=False,
    )
    assert len(rows) == 2
    assert all(np.isclose(row["episode_return"], -row["objective"]) for row in rows)
    assert audit["invalid_executions"] == 0
    assert model.local_feature_dim == loaded.local_feature_dim
