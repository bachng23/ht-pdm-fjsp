from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ht_pdm_fjsp.parallel_maintenance import (
    FAILED,
    MAINTENANCE,
    WORKING,
    CentralizedParallelMaintenanceEnv,
    ParallelMaintenanceConfig,
    ParallelMaintenanceEnv,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "parallel_maintenance.json"


def deterministic_config() -> ParallelMaintenanceConfig:
    config = ParallelMaintenanceConfig.from_json(CONFIG_PATH)
    technicians = tuple(
        replace(
            technician,
            absence_probability=0.0,
            duration_sigma=tuple(0.0 for _ in technician.duration_sigma),
        )
        for technician in config.technicians
    )
    return replace(config, horizon=12, technicians=technicians)


def test_config_observation_shapes_and_masks() -> None:
    config = deterministic_config()
    env = ParallelMaintenanceEnv(config)
    observation, info = env.reset(seed=3)
    assert observation["local_observations"].shape == (6, 3, 15)
    assert observation["action_masks"].shape == (6, 3)
    assert observation["global_state"].shape == (env.global_state_dim,)
    assert observation["action_masks"].all()
    assert info["return_objective_error"] == 0


def test_conditional_weibull_probability_matches_formula_and_increases_with_age() -> None:
    env = ParallelMaintenanceEnv(deterministic_config())
    env.reset(seed=4)
    machine = env.config.machines[0]
    increment = machine.age_rate * machine.load * env.config.time_step
    expected_at_zero = 1 - np.exp(-((increment / machine.weibull_scale) ** machine.weibull_shape))
    assert env.conditional_failure_probability(0) == pytest.approx(expected_at_zero)
    env.age[0] = 60.0
    assert env.conditional_failure_probability(0) > expected_at_zero


def test_resolver_prioritizes_corrective_and_queues_loser_without_restart() -> None:
    env = ParallelMaintenanceEnv(deterministic_config())
    env.reset(seed=5)
    env.mode[0] = WORKING
    env.age[0] = 70.0
    env.mode[1] = FAILED
    env.wait[1] = 2.0
    _, _, terminated, _, info = env.step([1, 1, 0, 0, 0, 0])
    assert not terminated
    assert env.mode[1] == MAINTENANCE
    assert env.machine_technician[1] == 0
    assert env.machine_technician[0] == -1
    assert info["proposal_conflicts"] == 1
    assert info["invalid_executions"] == 0


def test_busy_technician_is_masked_and_assignments_are_reciprocal() -> None:
    env = ParallelMaintenanceEnv(deterministic_config())
    observation, _ = env.reset(seed=6)
    assert observation["action_masks"][:, 1].all()
    observation, _, _, _, _ = env.step([1, 0, 0, 0, 0, 0])
    if env.technician_machine[0] >= 0:
        assigned_machine = int(env.technician_machine[0])
        assert env.machine_technician[assigned_machine] == 0
        assert not observation["action_masks"][:, 1].any()


def test_successful_repair_restores_age_and_updates_experience() -> None:
    config = deterministic_config()
    env = ParallelMaintenanceEnv(config)
    env.reset(seed=7)
    env.mode[0] = FAILED
    env.age[0] = 40.0
    env._start_maintenance(0, 0)
    env.repair_will_succeed[0] = True
    env.technician_remaining[0] = 1.0
    env.step([0] * env.num_agents)
    assert env.mode[0] == WORKING
    assert env.age[0] == pytest.approx(0.0)
    assert env.experience[0, 0] == pytest.approx(config.technicians[0].learning_rate)


def test_failed_repair_creates_rework_and_keeps_machine_failed() -> None:
    env = ParallelMaintenanceEnv(deterministic_config())
    env.reset(seed=8)
    env.mode[2] = FAILED
    env._start_maintenance(2, 1)
    env.repair_will_succeed[2] = False
    env.technician_remaining[1] = 1.0
    _, _, _, _, info = env.step([0] * env.num_agents)
    assert env.mode[2] == FAILED
    assert info["rework"] == 1
    assert env.machine_technician[2] == -1


def test_return_equals_negative_objective_and_run_is_reproducible() -> None:
    config = deterministic_config()

    def rollout() -> tuple[list[dict[str, object]], dict[str, object]]:
        env = ParallelMaintenanceEnv(config)
        observation, _ = env.reset(seed=91)
        trace = []
        done = False
        while not done:
            actions = [0] * env.num_agents
            for machine in range(env.num_agents):
                if env.mode[machine] == FAILED:
                    feasible = np.flatnonzero(observation["action_masks"][machine, 1:])
                    if len(feasible):
                        actions[machine] = int(feasible[0] + 1)
            observation, _, done, _, info = env.step(actions)
            trace.append(info["last_decision"])
        return trace, info

    trace_a, info_a = rollout()
    trace_b, info_b = rollout()
    assert trace_a == trace_b
    assert info_a == info_b
    assert info_a["episode_return"] == pytest.approx(-info_a["objective"], abs=1e-9)
    assert info_a["return_objective_error"] < 1e-9


def test_centralized_catalog_masks_technician_collisions() -> None:
    env = CentralizedParallelMaintenanceEnv(deterministic_config())
    observation, _ = env.reset(seed=9)
    masks = env.action_masks()
    collision = env.joint_actions.index((1, 1, 0, 0, 0, 0))
    feasible = env.joint_actions.index((1, 2, 0, 0, 0, 0))
    assert observation.shape == env.observation_space.shape
    assert not masks[collision]
    assert masks[feasible]
    with pytest.raises(ValueError, match="Masked centralized action"):
        env.step(collision)


def test_invalid_action_and_post_terminal_step_are_rejected() -> None:
    config = replace(deterministic_config(), horizon=1)
    env = ParallelMaintenanceEnv(config)
    env.reset(seed=10)
    with pytest.raises(ValueError, match="Masked or out-of-range"):
        env.step([99, 0, 0, 0, 0, 0])
    env.step([0] * env.num_agents)
    with pytest.raises(RuntimeError, match="Episode is done"):
        env.step([0] * env.num_agents)


def test_noop_cost_is_exact_negative_production_without_failure() -> None:
    config = deterministic_config()
    machines = tuple(
        replace(machine, weibull_scale=1e12) for machine in config.machines
    )
    env = ParallelMaintenanceEnv(replace(config, machines=machines, horizon=1))
    env.reset(seed=11)
    _, reward, done, _, info = env.step([0] * env.num_agents)
    expected_production = sum(
        machine.production_rate * machine.load * config.time_step
        for machine in machines
    )
    assert done
    assert info["production"] == pytest.approx(expected_production)
    assert info["downtime"] == 0
    assert info["objective"] == pytest.approx(-expected_production)
    assert reward == pytest.approx(expected_production)


def test_randomized_multiseed_rollouts_preserve_all_resource_audits() -> None:
    config = replace(deterministic_config(), horizon=40)
    for seed in range(25):
        env = ParallelMaintenanceEnv(config)
        observation, _ = env.reset(seed=seed)
        policy_rng = np.random.default_rng(seed + 1000)
        done = False
        while not done:
            actions = [
                int(policy_rng.choice(np.flatnonzero(mask)))
                for mask in observation["action_masks"]
            ]
            observation, _, done, _, info = env.step(actions)
        assert info["invalid_executions"] == 0
        assert info["duplicate_machine_assignments"] == 0
        assert info["duplicate_technician_assignments"] == 0
        assert info["return_objective_error"] < 1e-9
