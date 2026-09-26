from pathlib import Path

import torch

from ht_pdm_fjsp.passive_technician_baselines import stress_config
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition,
    TechnicianEdgeQ,
    ValueTrainSettings,
    train_value_decomposition_checkpoints,
)
from ht_pdm_fjsp.passive_technician_long_comparison import _profile, environment_cells


def test_long_comparison_protocol_is_locked() -> None:
    train, evaluation, budgets = _profile("full")

    assert train == tuple(range(11, 16))
    assert evaluation == tuple(range(101, 201))
    assert budgets == (5_000, 10_000, 20_000, 50_000)
    assert set(environment_cells()) == {
        "in_distribution",
        "early_failure",
        "slow_service",
        "combined_pressure",
    }


def test_technician_edge_q_and_tqmix_checkpoint_round_trip(tmp_path: Path) -> None:
    config = stress_config()
    actor = TechnicianEdgeQ(config.observation_dim, config.technicians, hidden_dim=16)
    assert actor(torch.zeros(2, config.observation_dim)).shape == (
        2,
        config.technicians + 1,
    )

    model = PassiveValueDecomposition(config, "tqmix", hidden_dim=16, mixer_hidden_dim=8)
    path = tmp_path / "model.pt"
    model.save(path, seed=11)
    loaded = PassiveValueDecomposition.load(path, config, torch.device("cpu"))
    assert loaded.algorithm == "tqmix"
    assert loaded.agent_q(torch.zeros(2, config.machines, config.observation_dim)).shape == (
        2,
        config.machines,
        config.technicians + 1,
    )


def test_tqmix_smoke_train_writes_all_checkpoints(tmp_path: Path) -> None:
    config = stress_config()
    checkpoints, progress = train_value_decomposition_checkpoints(
        config,
        "tqmix",
        seed=11,
        budgets=(2, 4),
        root=tmp_path,
        settings=ValueTrainSettings(
            episodes=4,
            hidden_dim=16,
            mixer_hidden_dim=8,
            replay_capacity=64,
            batch_size=4,
            learning_starts=4,
            train_frequency=1,
            target_update_interval=4,
        ),
        device=torch.device("cpu"),
    )

    assert set(checkpoints) == {2, 4}
    assert all(path.is_file() for path in checkpoints.values())
    assert len(progress) == 4
