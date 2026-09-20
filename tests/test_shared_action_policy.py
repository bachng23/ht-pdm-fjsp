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


def _model(
    *, entity_conditioned: bool = False, production_context: bool = False
) -> MaskablePPO:
    if entity_conditioned and production_context:
        raise ValueError("Select at most one context ablation.")
    extra_key = None
    if entity_conditioned:
        extra_key = "action_context"
    elif production_context:
        extra_key = "production_context"
    return MaskablePPO(
        SharedActionMaskablePolicy,
        Monitor(
            HTPdmFjspEnv(
                config=CONFIG,
                include_action_context=entity_conditioned,
                include_production_context=production_context,
            )
        ),
        policy_kwargs={
            "extra_action_feature_keys": (extra_key,) if extra_key else ()
        },
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


def test_entity_context_links_actions_to_their_entities() -> None:
    env = HTPdmFjspEnv(config=CONFIG, include_action_context=True)
    observation, _ = env.reset(seed=123)
    context = observation["action_context"]
    assert context.shape == env.observation_space["action_context"].shape
    assert np.count_nonzero(context[0]) == 0
    production_index = next(
        index
        for index, descriptor in enumerate(env.actions)
        if descriptor.kind == "production"
    )
    descriptor = env.actions[production_index]
    job_index = env.job_indices[str(descriptor.job_id)]
    machine_index = env.machine_indices[str(descriptor.machine_id)]
    assert context[production_index, 0] == 1.0
    assert np.array_equal(
        context[production_index, 1:5], observation["jobs"][job_index]
    )
    assert context[production_index, 5] == 1.0
    assert np.array_equal(
        context[production_index, 6:13], observation["machines"][machine_index]
    )
    env.close()


def test_entity_conditioned_logits_are_permutation_equivariant() -> None:
    env = HTPdmFjspEnv(config=CONFIG, include_action_context=True)
    observation, _ = env.reset(seed=321)
    model = _model(entity_conditioned=True)
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with th.no_grad():
        original = model.policy.action_logits(obs_tensor)
        permuted_obs = {key: value.clone() for key, value in obs_tensor.items()}
        for key in ("action_features", "action_context"):
            permuted_obs[key][:, [1, 2]] = permuted_obs[key][:, [2, 1]]
        permuted = model.policy.action_logits(permuted_obs)
    assert th.allclose(original[:, 1], permuted[:, 2])
    assert th.allclose(original[:, 2], permuted[:, 1])
    assert th.allclose(original[:, 3:], permuted[:, 3:])
    model.learn(total_timesteps=64)
    env.close()


def test_production_context_only_links_production_actions() -> None:
    env = HTPdmFjspEnv(config=CONFIG, include_production_context=True)
    observation, _ = env.reset(seed=123)
    context = observation["production_context"]
    assert env.observation_space.contains(observation)
    assert context.shape == env.observation_space["production_context"].shape
    nonproduction = [
        index
        for index, descriptor in enumerate(env.actions)
        if descriptor.kind != "production"
    ]
    assert np.count_nonzero(context[nonproduction]) == 0
    production_index = next(
        index
        for index, descriptor in enumerate(env.actions)
        if descriptor.kind == "production"
    )
    descriptor = env.actions[production_index]
    job_index = env.job_indices[str(descriptor.job_id)]
    machine_index = env.machine_indices[str(descriptor.machine_id)]
    assert np.array_equal(
        context[production_index, :4], observation["jobs"][job_index]
    )
    machine_one_hot = context[production_index, 6:]
    assert machine_one_hot.sum() == 1.0
    assert machine_one_hot[machine_index] == 1.0
    env.close()


def test_production_context_logits_are_permutation_equivariant(
    tmp_path: Path,
) -> None:
    env = HTPdmFjspEnv(config=CONFIG, include_production_context=True)
    observation, _ = env.reset(seed=321)
    model = _model(production_context=True)
    obs_tensor, _ = model.policy.obs_to_tensor(observation)
    with th.no_grad():
        original = model.policy.action_logits(obs_tensor)
        permuted_obs = {key: value.clone() for key, value in obs_tensor.items()}
        for key in ("action_features", "production_context"):
            permuted_obs[key][:, [1, 2]] = permuted_obs[key][:, [2, 1]]
        permuted = model.policy.action_logits(permuted_obs)
    assert th.allclose(original[:, 1], permuted[:, 2])
    assert th.allclose(original[:, 2], permuted[:, 1])
    assert th.allclose(original[:, 3:], permuted[:, 3:])
    model.learn(total_timesteps=64)
    path = tmp_path / "production_context_policy_test"
    model.save(path)
    loaded = MaskablePPO.load(path, device="cpu")
    action, _ = loaded.predict(
        observation, action_masks=env.action_masks(), deterministic=True
    )
    assert env.action_masks()[int(np.asarray(action).item())]
    env.close()
