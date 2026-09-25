from pathlib import Path

import torch

from ht_pdm_fjsp.passive_technician_ablation import ABLATION_ALGORITHMS, _profile
from ht_pdm_fjsp.passive_technician_baselines import stress_config
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition,
    ValueTrainSettings,
    train_value_decomposition_checkpoints,
)


def test_ablation_protocol_and_component_mapping() -> None:
    train, evaluation, budgets = _profile("full")
    assert train == (11, 12, 13)
    assert evaluation == tuple(range(101, 201))
    assert budgets == (20_000, 50_000)
    assert ABLATION_ALGORITHMS == (
        "qmix",
        "qmix_edge",
        "qmix_queue",
        "qmix_counterfactual",
        "tqmix",
    )


def test_each_ablation_has_expected_components() -> None:
    config = stress_config()
    expected = {
        "qmix": (False, False, False),
        "qmix_edge": (True, False, False),
        "qmix_queue": (False, True, False),
        "qmix_counterfactual": (False, False, True),
        "tqmix": (True, True, True),
    }
    for algorithm, flags in expected.items():
        model = PassiveValueDecomposition(config, algorithm, hidden_dim=16, mixer_hidden_dim=8)
        assert (model.use_edge_q, model.use_queue_mixer, model.use_counterfactual) == flags


def test_ablation_variants_isolate_the_mixer_component() -> None:
    config = stress_config()
    standard = PassiveValueDecomposition(config, "qmix")
    edge = PassiveValueDecomposition(config, "qmix_edge")
    queue = PassiveValueDecomposition(config, "qmix_queue")
    full = PassiveValueDecomposition(config, "tqmix")
    assert type(standard.mixer) is type(edge.mixer)
    assert type(queue.mixer) is type(full.mixer)
    assert type(standard.mixer) is not type(queue.mixer)


def test_ablation_smoke_train_save_reload(tmp_path: Path) -> None:
    config = stress_config()
    for algorithm in ABLATION_ALGORITHMS:
        checkpoints, progress = train_value_decomposition_checkpoints(
            config,
            algorithm,
            seed=11,
            budgets=(2, 4),
            root=tmp_path / algorithm,
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
        assert len(progress) == 4
        loaded = PassiveValueDecomposition.load(checkpoints[4], config, torch.device("cpu"))
        assert loaded.algorithm == algorithm
