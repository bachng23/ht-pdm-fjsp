from pathlib import Path

from ht_pdm_fjsp.passive_technician_baselines import (
    ActorCritic,
    CentralizedPPO,
    oracle_config,
    random_feasible_action,
    stress_config,
)
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig, PassiveTechnicianEnv, exact_optimum


def test_stress_config_has_shared_heterogeneous_resource() -> None:
    config = stress_config()
    assert config.machines == 3
    assert config.technicians == 2
    assert config.service_time[0][0] != config.service_time[0][1]


def test_random_feasible_actions_do_not_duplicate_available_technicians() -> None:
    env = PassiveTechnicianEnv(stress_config(), seed=1)
    env.reset()
    actions = random_feasible_action(env)
    selected = [action for action in actions if action]
    assert len(selected) == len(set(selected))


def test_baseline_networks_and_oracle() -> None:
    config = stress_config()
    assert config.observation_dim == 3 + 6 * config.technicians
    assert ActorCritic(config.observation_dim, config.technicians + 1)(
        __import__("torch").zeros(1, config.observation_dim)
    )[0].shape == (1, 3)
    assert CentralizedPPO(config.machines, config.observation_dim, config.technicians + 1)(
        __import__("torch").zeros(1, config.machines * config.observation_dim)
    )[0].shape == (1, 3, 3)
    assert exact_optimum(oracle_config()) >= 0


def test_observation_exposes_service_time_and_queue_position() -> None:
    config = stress_config()
    env = PassiveTechnicianEnv(config, seed=1)
    observations = env.reset()
    assert len(observations) == config.machines
    assert len(observations[0]) == config.observation_dim
    assert observations[0][-2:] == (0, config.service_time[0][config.technicians - 1])


def test_fifo_waiting_cost_is_in_environment_objective() -> None:
    config = PassiveConfig(
        machines=2,
        technicians=1,
        horizon=1,
        failure_age=10,
        max_age=10,
        service_time=((1,), (1,)),
        queue_waiting_cost=0.5,
    )
    env = PassiveTechnicianEnv(config, seed=1)
    env.reset()
    env.state = env.state.__class__((0, 0), (False, False), (0,), (-1,), ((), ()))
    _, _, _, info = env.step((1, 1))
    assert info["waiting"] == 1
    assert info["objective"] == config.maintenance_cost + config.queue_waiting_cost
