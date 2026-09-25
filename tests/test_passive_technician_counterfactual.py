from pathlib import Path

import torch

from ht_pdm_fjsp.passive_technician_baselines import PPOSettings, _obs_tensor, stress_config
from ht_pdm_fjsp.passive_technician_counterfactual import (
    QueueAwareCounterfactualPolicy,
    TechnicianEdgeActor,
    _counterfactual_advantages,
    evaluate_queue_aware_counterfactual,
    train_queue_aware_counterfactual,
)
from ht_pdm_fjsp.passive_technician_marl import PassiveTechnicianEnv


def test_technician_edge_actor_has_defer_plus_one_logit_per_technician() -> None:
    config = stress_config()
    actor = TechnicianEdgeActor(config.observation_dim, config.technicians, hidden_dim=16)
    logits = actor(torch.zeros(4, config.observation_dim))

    assert logits.shape == (4, config.technicians + 1)


def test_counterfactual_advantages_and_model_round_trip(tmp_path: Path) -> None:
    config = stress_config()
    device = torch.device("cpu")
    model = QueueAwareCounterfactualPolicy(config, hidden_dim=16).to(device)
    env = PassiveTechnicianEnv(config, seed=11)
    observations = env.reset()
    local = _obs_tensor(observations, config).unsqueeze(0)
    masks = torch.tensor([env.action_masks()], dtype=torch.bool)
    actions = torch.zeros(1, config.machines, dtype=torch.long)
    advantages = _counterfactual_advantages(
        model, local.flatten(start_dim=1), local, masks, actions
    )

    assert advantages.shape == (1, config.machines)
    path = tmp_path / "model.pt"
    model.save(path, seed=11)
    loaded = QueueAwareCounterfactualPolicy.load(path, config, device)
    assert loaded.action_dim == config.technicians + 1


def test_counterfactual_smoke_train_save_reload_evaluate(tmp_path: Path) -> None:
    config = stress_config()
    path, progress = train_queue_aware_counterfactual(
        config,
        seed=11,
        episodes=2,
        path=tmp_path / "budget_2" / "model.pt",
        settings=PPOSettings(episodes=2, update_epochs=1),
        device=torch.device("cpu"),
    )
    model = QueueAwareCounterfactualPolicy.load(path, config, torch.device("cpu"))
    row = evaluate_queue_aware_counterfactual(config, model, 101, 11, torch.device("cpu"))

    assert path.is_file()
    assert len(progress) == 2
    assert row["invalid_requests"] == 0
    assert row["objective"] >= 0
