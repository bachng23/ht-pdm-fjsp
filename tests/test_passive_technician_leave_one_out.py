from __future__ import annotations

import json
import random
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

from ht_pdm_fjsp.passive_technician_baselines import stress_config
from ht_pdm_fjsp.passive_technician_leave_one_out import (
    ALGORITHMS,
    evaluate_diagnostic_episode,
    environment_cells,
    profile_settings,
    run,
)
from ht_pdm_fjsp.passive_technician_marl import PassiveConfig, PassiveTechnicianEnv
from ht_pdm_fjsp.passive_technician_value_decomposition import (
    PassiveValueDecomposition,
    ValueTrainSettings,
)


def test_protocol_and_leave_one_out_mapping_are_locked() -> None:
    train_seeds, evaluation_seeds, budgets, settings = profile_settings("full")
    assert train_seeds == (11, 12, 13)
    assert evaluation_seeds == tuple(range(101, 201))
    assert budgets == (20_000, 50_000)
    assert settings.lambda_cf == 0.05
    assert ALGORITHMS == (
        "qmix",
        "tqmix_no_edge",
        "tqmix_no_queue",
        "tqmix_no_cf",
        "tqmix",
    )
    expected = {
        "qmix": (False, False, False),
        "tqmix_no_edge": (False, True, True),
        "tqmix_no_queue": (True, False, True),
        "tqmix_no_cf": (True, True, False),
        "tqmix": (True, True, True),
    }
    config = stress_config()
    for algorithm, flags in expected.items():
        model = PassiveValueDecomposition(config, algorithm, 16, 8)
        assert (model.use_edge_q, model.use_queue_mixer, model.use_counterfactual) == flags
        local = torch.zeros(2, config.machines, config.observation_dim)
        assert model.agent_q(local).shape == (
            2,
            config.machines,
            config.technicians + 1,
        )
    assert set(environment_cells()) == {
        "in_distribution",
        "early_failure",
        "slow_service",
        "combined_pressure",
    }


def test_counterfactual_branch_has_gradient_only_when_enabled() -> None:
    config = stress_config()
    local = torch.randn(3, config.machines, config.observation_dim)
    actions = torch.zeros(3, config.machines, dtype=torch.long)
    masks = torch.ones(3, config.machines, config.technicians + 1, dtype=torch.bool)
    enabled = PassiveValueDecomposition(config, "tqmix_no_edge", 16, 8)
    loss = enabled.local_edge_consistency(local, actions, masks)
    assert loss.isfinite() and loss.requires_grad
    loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in enabled.agent.parameters()
    )
    disabled = PassiveValueDecomposition(config, "tqmix_no_cf", 16, 8)
    assert disabled.local_edge_consistency(local, actions, masks).item() == 0.0


def test_checkpoint_round_trip_restores_variant_settings_and_q_values(tmp_path: Path) -> None:
    config = stress_config()
    settings = ValueTrainSettings(episodes=16, hidden_dim=16, mixer_hidden_dim=8)
    model = PassiveValueDecomposition(config, "tqmix_no_queue", 16, 8)
    fixed_input = torch.randn(2, config.machines, config.observation_dim)
    expected = model.agent_q(fixed_input).detach()
    path = tmp_path / "model.pt"
    model.save(path, seed=11, settings=settings)
    loaded = PassiveValueDecomposition.load(path, config, torch.device("cpu"))
    assert loaded.algorithm == "tqmix_no_queue"
    assert loaded.checkpoint_settings == settings.__dict__
    assert torch.equal(expected, loaded.agent_q(fixed_input).detach())


def test_recovery_availability_and_cost_timeline_are_explicit() -> None:
    config = PassiveConfig(
        machines=1,
        technicians=1,
        horizon=3,
        failure_age=1,
        failure_probability=1.0,
        max_age=4,
        service_time=((2,),),
    )
    env = PassiveTechnicianEnv(config, seed=7)
    env.reset()
    env.step((0,))
    assert env.state.failed == (True,)
    _, _, _, info = env.step((1,))
    assert info["objective"] == config.downtime_cost + config.maintenance_cost
    assert env.state.ages == (0,)
    assert env.state.failed == (False,)
    assert env.state.assigned_machine == (0,)
    assert env.action_masks()[0] == (True, False)
    env.step((0,))
    assert env.state.assigned_machine == (-1,)
    assert env.metrics["objective"] == (
        2 * config.failure_cost + config.downtime_cost + config.maintenance_cost
    )


class _SequencePolicy:
    algorithm = "qmix"

    def __init__(self) -> None:
        self.step = 0

    def agent_q(self, local: torch.Tensor) -> torch.Tensor:
        result = torch.zeros(local.shape[0], local.shape[1], 2)
        action = 0 if self.step == 0 else 1
        result[..., action] = 1.0
        self.step += 1
        return result


def test_diagnostic_metrics_reconcile_cost_and_preserve_model_state() -> None:
    config = PassiveConfig(
        machines=1,
        technicians=1,
        horizon=3,
        failure_age=1,
        failure_probability=1.0,
        max_age=4,
        service_time=((2,),),
    )
    torch_state = torch.get_rng_state().clone()
    row = evaluate_diagnostic_episode(
        config,
        _SequencePolicy(), 101, 11, "known", 16, torch.device("cpu")
    )
    assert row["objective"] == row["cost_component_sum"]
    assert row["cost_reconciliation_error"] == 0.0
    assert row["failure_events"] == 2
    assert row["machine_downtime_steps"] == 1
    assert row["service_starts"] == 1
    assert row["service_completions"] == 1
    assert row["corrective_starts"] == 1
    assert row["technician_0_busy_steps"] == 2
    assert row["pending_requests_at_horizon"] == 0
    assert torch.equal(torch_state, torch.get_rng_state())


def test_queue_waiting_censoring_and_utilization_on_known_trajectory() -> None:
    config = PassiveConfig(
        machines=2,
        technicians=1,
        horizon=3,
        failure_age=10,
        failure_probability=0.0,
        service_time=((3,), (3,)),
    )

    class _BothRequest:
        algorithm = "qmix"

        def agent_q(self, local: torch.Tensor) -> torch.Tensor:
            result = torch.zeros(local.shape[0], local.shape[1], 2)
            result[..., 1] = 1.0
            return result

    row = evaluate_diagnostic_episode(
        config, _BothRequest(), 101, 11, "known_queue", 16, torch.device("cpu")
    )
    assert row["queue_waiting_steps"] == 3
    assert row["mean_queue_length"] == 1.0
    assert row["max_queue_length"] == 1
    assert row["pending_requests_at_horizon"] == 1
    assert row["service_starts"] == 1
    assert row["service_completions"] == 1
    assert row["technician_0_utilization"] == 1.0
    assert row["queue_waiting_cost"] == 3 * config.queue_waiting_cost


def test_evaluation_does_not_modify_weights_or_global_rng_state() -> None:
    config = stress_config()
    model = PassiveValueDecomposition(config, "tqmix", 16, 8)
    weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
    torch_state = torch.get_rng_state().clone()
    numpy_state = np.random.get_state()
    python_state = random.getstate()
    evaluate_diagnostic_episode(
        config, model, 101, 11, "in_distribution", 16, torch.device("cpu")
    )
    assert all(torch.equal(weights[name], value) for name, value in model.state_dict().items())
    assert torch.equal(torch_state, torch.get_rng_state())
    assert np.array_equal(numpy_state[1], np.random.get_state()[1])
    assert python_state == random.getstate()


def test_runner_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "occupied"
    output.mkdir()
    (output / "keep.txt").write_text("user data", encoding="utf-8")
    with pytest.raises(FileExistsError):
        run(Namespace(profile="smoke", device="cpu", output_dir=str(output)))


def test_smoke_runner_meets_locked_counts_and_audits(tmp_path: Path) -> None:
    output = run(
        Namespace(profile="smoke", device="cpu", output_dir=str(tmp_path / "smoke"))
    )
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "COMPLETED"
    assert manifest["actual_counts"] == {
        "evaluation_rows": 120,
        "checkpoints": 10,
        "training_trajectories": 5,
    }
    assert manifest["sealed_test_evaluated"] is False
    assert all(manifest["audits"].values())
    with (output / "episodes.csv").open(encoding="utf-8") as stream:
        assert sum(1 for _ in stream) - 1 == 120
    with (output / "training_progress.csv").open(encoding="utf-8") as stream:
        text = stream.read()
    assert "raw_cf_loss" in text
    assert "weighted_cf_loss" in text
    assert np.isfinite(
        [
            value
            for row in manifest["component_mapping"].values()
            for value in [row["lambda_cf"]]
        ]
    ).all()
