from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.models import BenchmarkConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs/minimal_benchmark.json")


def _local_action(
    env: MachineAgentsCTDEEnv,
    agent: int,
    *,
    kind: str,
    job_id: str | None = None,
) -> int:
    for local_action, global_action in enumerate(
        env.local_action_catalogs[agent]
    ):
        if global_action is None:
            continue
        descriptor = env.core.actions[global_action]
        if descriptor.kind == kind and descriptor.job_id == job_id:
            return local_action
    raise AssertionError("Requested local action is absent")


def test_ctde_observations_separate_local_actor_and_global_critic() -> None:
    env = MachineAgentsCTDEEnv(CONFIG)
    observation, _ = env.reset(seed=1)
    assert observation["local_observations"].shape == (
        len(CONFIG.machines),
        env.max_local_actions,
        env.LOCAL_FEATURE_DIM,
    )
    assert observation["global_state"].shape == (env.global_state_dim,)
    assert env.num_agents == len(CONFIG.machines)
    assert env.LOCAL_FEATURE_DIM < env.global_state_dim
    assert observation["action_masks"].all(axis=1).sum() == 0


def test_broadcast_context_adds_only_locked_aggregate_features() -> None:
    base = MachineAgentsCTDEEnv(CONFIG)
    context = MachineAgentsCTDEEnv(CONFIG, include_broadcast_context=True)
    base_observation, _ = base.reset(seed=11)
    context_observation, _ = context.reset(seed=11)
    assert context.local_feature_dim == (
        base.LOCAL_FEATURE_DIM + base.BROADCAST_CONTEXT_DIM
    )
    assert np.array_equal(
        context_observation["local_observations"][..., : base.LOCAL_FEATURE_DIM],
        base_observation["local_observations"],
    )
    for agent in range(context.num_agents):
        catalog_size = len(context.local_action_catalogs[agent])
        broadcast = context_observation["local_observations"][
            agent, :catalog_size, base.LOCAL_FEATURE_DIM :
        ]
        assert np.allclose(broadcast, broadcast[0])
        assert np.isfinite(broadcast).all()


def test_same_operation_proposals_are_resolved_once() -> None:
    env = MachineAgentsCTDEEnv(CONFIG)
    env.reset(seed=2)
    actions = [
        _local_action(env, 0, kind="production", job_id="J1"),
        _local_action(env, 1, kind="production", job_id="J1"),
    ]
    _, _, _, _, info = env.step(actions)
    assert info["coordination"]["production_conflicts"] == 1
    assert info["coordination"]["accepted"] == 1
    assert info["coordination"]["invalid_executions"] == 0


def test_distinct_job_proposals_start_as_one_joint_decision() -> None:
    env = MachineAgentsCTDEEnv(CONFIG)
    env.reset(seed=3)
    actions = [
        _local_action(env, 0, kind="production", job_id="J1"),
        _local_action(env, 1, kind="production", job_id="J2"),
    ]
    _, _, _, _, info = env.step(actions)
    assert info["coordination"]["accepted"] == 2
    assert info["coordination"]["rejected"] == 0
    assert sum(state.in_process for state in env.core.jobs.values()) == 1
    assert len(env.core.events) == 1


def test_masked_local_action_is_rejected_before_simulator_execution() -> None:
    env = MachineAgentsCTDEEnv(CONFIG)
    observation, _ = env.reset(seed=4)
    masked = int(np.flatnonzero(~observation["action_masks"][0])[0])
    valid = int(np.flatnonzero(observation["action_masks"][1])[0])
    with pytest.raises(ValueError, match="Masked local action"):
        env.step([masked, valid])


def test_safe_noop_requires_exactly_one_progress_anchor_without_events() -> None:
    env = MachineAgentsCTDEEnv(CONFIG, wait_policy="safe_noop")
    observation, _ = env.reset(seed=4)
    wait_mask = observation["action_masks"][:, 0]
    assert np.count_nonzero(~wait_mask) == 1
    required_agent = int(np.flatnonzero(~wait_mask)[0])
    required_nonwait = np.flatnonzero(
        observation["action_masks"][required_agent, 1:]
    )
    assert len(required_nonwait) > 0


def test_safe_noop_allows_idle_peer_to_wait_while_anchor_progresses() -> None:
    env = MachineAgentsCTDEEnv(CONFIG, wait_policy="safe_noop")
    observation, _ = env.reset(seed=8)
    required_agent = int(np.flatnonzero(~observation["action_masks"][:, 0])[0])
    peer = 1 - required_agent
    assert observation["action_masks"][peer, 0]
    nonwait = int(
        np.flatnonzero(observation["action_masks"][required_agent, 1:])[0] + 1
    )
    actions = [0, 0]
    actions[required_agent] = nonwait
    _, _, _, _, info = env.step(actions)
    assert info["coordination"]["accepted"] == 1
    assert info["coordination"]["waits"] == 1


def test_decentralized_greedy_rollout_completes_with_reward_identity() -> None:
    env = MachineAgentsCTDEEnv(CONFIG)
    observation, _ = env.reset(seed=5)
    episode_return = 0.0
    while not env._done:
        actions = []
        for mask in observation["action_masks"]:
            valid = np.flatnonzero(mask)
            nonwait = valid[valid != 0]
            actions.append(int(nonwait[0] if len(nonwait) else valid[0]))
        observation, reward, _, _, _ = env.step(actions)
        episode_return += reward
    result = env.result("decentralized_greedy")
    assert np.isclose(episode_return, -float(result.metrics["objective"]))
    assert env.coordination_totals["invalid_executions"] == 0
