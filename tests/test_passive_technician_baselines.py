from pathlib import Path

from ht_pdm_fjsp.passive_technician_baselines import (
    ActorCritic,
    CentralizedPPO,
    oracle_config,
    random_feasible_action,
    stress_config,
)
from ht_pdm_fjsp.passive_technician_marl import PassiveTechnicianEnv, exact_optimum


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
    assert ActorCritic(1, config.technicians + 1)(__import__("torch").zeros(1, 1))[0].shape == (1, 3)
    assert CentralizedPPO(config.machines, config.technicians + 1)(__import__("torch").zeros(1, 3))[0].shape == (1, 3, 3)
    assert exact_optimum(oracle_config()) >= 0
