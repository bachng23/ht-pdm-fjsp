from pathlib import Path

import torch

from ht_pdm_fjsp.passive_technician_algorithm_screen import (
    ALGORITHMS,
    ComaPolicy,
    _profile,
)
from ht_pdm_fjsp.passive_technician_baselines import stress_config


def test_screen_profile_is_locked() -> None:
    train, evaluation, episodes = _profile("full")

    assert train == (11, 12, 13)
    assert evaluation == tuple(range(101, 131))
    assert episodes == 5_000
    assert ALGORITHMS == (
        "independent_q",
        "independent_ppo",
        "mappo_ctde",
        "coma_counterfactual",
    )


def test_coma_policy_has_joint_critic_and_round_trip(tmp_path: Path) -> None:
    config = stress_config()
    model = ComaPolicy(config, hidden_dim=16)
    observations = torch.zeros(2, config.machines * config.observation_dim)
    actions = torch.zeros(2, config.machines, dtype=torch.long)

    assert model.critic_value(observations, actions).shape == (2,)
    path = tmp_path / "coma.pt"
    model.save(path, seed=11)
    loaded = ComaPolicy.load(path, config, torch.device("cpu"))
    assert loaded.machines == config.machines
    assert loaded.action_dim == config.technicians + 1
