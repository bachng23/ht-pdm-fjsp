from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.shared_action_policy import SharedActionMaskablePolicy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs" / "minimal_benchmark.json")


def _model() -> MaskablePPO:
    return MaskablePPO(
        SharedActionMaskablePolicy,
        Monitor(HTPdmFjspEnv(config=CONFIG)),
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        seed=123,
        device="cpu",
        verbose=0,
    )


def test_shared_action_logits_are_permutation_equivariant() -> None:
    env = HTPdmFjspEnv(config=CONFIG)
    observation, _ = env.reset(seed=123)
    model = _model()
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with th.no_grad():
        original = model.policy.action_logits(obs_tensor)
        permuted_obs = {key: value.clone() for key, value in obs_tensor.items()}
        permuted_obs["action_features"][:, [1, 2]] = permuted_obs[
            "action_features"
        ][:, [2, 1]]
        permuted = model.policy.action_logits(permuted_obs)
    assert th.allclose(original[:, 1], permuted[:, 2])
    assert th.allclose(original[:, 2], permuted[:, 1])
    assert th.allclose(original[:, 3:], permuted[:, 3:])
    env.close()


def test_shared_action_policy_trains_saves_and_loads(tmp_path: Path) -> None:
    model = _model()
    model.learn(total_timesteps=64)
    path = tmp_path / "shared_policy"
    model.save(path)
    loaded = MaskablePPO.load(path, device="cpu")
    env = HTPdmFjspEnv(config=CONFIG)
    observation, _ = env.reset(seed=456)
    mask = env.action_masks()
    action, _ = loaded.predict(
        observation, action_masks=mask, deterministic=True
    )
    assert mask[int(np.asarray(action).item())]
    env.close()
