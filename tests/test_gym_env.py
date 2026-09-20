from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

import ht_pdm_fjsp
import ht_pdm_fjsp.rl_experiment as rl_experiment
from ht_pdm_fjsp.advanced_baselines import (
    CPSATReactivePolicy,
    HealthThresholdPolicy,
    JointRiskGreedyPolicy,
    MaskedSPTPolicy,
    RollingHorizonPolicy,
    rollout_policy,
    solve_deterministic_cp_sat,
)
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.rl_experiment import resolve_device


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs" / "minimal_benchmark.json")


def test_environment_passes_gymnasium_checker() -> None:
    env = HTPdmFjspEnv(config=CONFIG)
    check_env(env, skip_render_check=True)
    env.close()


def test_registered_environment_loads() -> None:
    env = gym.make("HTPdMFJSP-v0")
    observation, info = env.reset(seed=1)
    assert env.observation_space.contains(observation)
    assert info["feasible_action_count"] > 0
    env.close()


def test_action_masks_alias_matches_native_mask() -> None:
    env = HTPdmFjspEnv(config=CONFIG)
    env.reset(seed=1)
    assert np.array_equal(env.action_masks(), env.action_mask())


def test_unseeded_resets_use_reproducible_seed_stream() -> None:
    first = HTPdmFjspEnv(config=CONFIG)
    second = HTPdmFjspEnv(config=CONFIG)
    first.reset(seed=123)
    second.reset(seed=123)
    first.reset()
    second.reset()
    assert first.root_seed == second.root_seed
    assert first.root_seed != 0


def test_explicit_cpu_device_is_preserved() -> None:
    assert resolve_device("cpu") == "cpu"


def test_auto_device_falls_back_for_unsupported_gpu(monkeypatch) -> None:
    monkeypatch.setattr(rl_experiment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        rl_experiment.torch.cuda, "get_device_capability", lambda index: (6, 1)
    )
    monkeypatch.setattr(
        rl_experiment.torch.cuda, "get_arch_list", lambda: ["sm_75", "sm_80"]
    )
    monkeypatch.setattr(
        rl_experiment.torch.cuda, "get_device_name", lambda index: "Quadro P2200"
    )
    assert resolve_device("auto") == "cpu"


def test_action_mask_enforces_machine_and_job_exclusivity() -> None:
    env = HTPdmFjspEnv(config=CONFIG)
    observation, _ = env.reset(seed=2)
    production = [
        index
        for index, descriptor in enumerate(env.actions)
        if observation["action_mask"][index] and descriptor.kind == "production"
    ]
    action = production[0]
    descriptor = env.actions[action]
    observation, _, _, _, _ = env.step(action)
    for index, candidate in enumerate(env.actions):
        if candidate.kind != "production":
            continue
        if candidate.machine_id == descriptor.machine_id or candidate.job_id == descriptor.job_id:
            assert observation["action_mask"][index] == 0


def test_invalid_action_is_safe_and_penalized() -> None:
    env = HTPdmFjspEnv(config=CONFIG)
    env.reset(seed=3)
    observation, reward, terminated, truncated, info = env.step(0)
    assert reward == -env.env_config.invalid_action_penalty
    assert info["invalid_action"]
    assert not terminated and not truncated
    assert env.observation_space.contains(observation)


def test_cp_sat_plan_covers_every_operation() -> None:
    plan = solve_deterministic_cp_sat(CONFIG, time_limit_seconds=2.0)
    assert len(plan) == sum(len(job.operations) for job in CONFIG.jobs)
    for job in CONFIG.jobs:
        starts = [plan[(job.job_id, index)][1] for index in range(len(job.operations))]
        assert starts == sorted(starts)


def test_all_baselines_complete_and_reward_identity_holds() -> None:
    policies = (
        MaskedSPTPolicy(),
        HealthThresholdPolicy(CONFIG.preventive_probability_threshold),
        JointRiskGreedyPolicy(),
        CPSATReactivePolicy(
            probability_threshold=CONFIG.preventive_probability_threshold,
            time_limit_seconds=2.0,
        ),
        RollingHorizonPolicy(depth=2, branch_width=5),
    )
    for policy in policies:
        result, episode_return = rollout_policy(
            HTPdmFjspEnv(config=CONFIG), policy, seed=13
        )
        assert result.metrics["makespan"] > 0
        assert np.isclose(episode_return, -float(result.metrics["objective"]))


def test_rollout_is_exactly_reproducible() -> None:
    first, first_return = rollout_policy(
        HTPdmFjspEnv(config=CONFIG), JointRiskGreedyPolicy(), seed=99
    )
    second, second_return = rollout_policy(
        HTPdmFjspEnv(config=CONFIG), JointRiskGreedyPolicy(), seed=99
    )
    assert first.to_dict() == second.to_dict()
    assert first_return == second_return


def test_rolling_horizon_does_not_use_episode_future_seed() -> None:
    first = HTPdmFjspEnv(config=CONFIG)
    second = HTPdmFjspEnv(config=CONFIG)
    first.reset(seed=7)
    second.reset(seed=999)
    policy = RollingHorizonPolicy(depth=1, scenario_count=2)
    assert policy.action(first) == policy.action(second)


def test_nonterminal_states_always_have_a_feasible_action() -> None:
    env = HTPdmFjspEnv(config=CONFIG)
    env.reset(seed=17)
    policy = JointRiskGreedyPolicy()
    policy.reset(env)
    while not env._done:
        assert env.action_mask().sum() > 0
        observation, _, _, _, _ = env.step(policy.action(env))
        assert env.observation_space.contains(observation)
