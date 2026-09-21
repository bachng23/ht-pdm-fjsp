from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch as th

from ht_pdm_fjsp.ctde_env import MachineAgentsCTDEEnv
from ht_pdm_fjsp.ctde_mappo import MachineMAPPO
from ht_pdm_fjsp.marl_diagnostic import (
    DiagnosticPolicy,
    DiagnosticSettings,
    evaluate_diagnostic_policy,
    train_diagnostic_policy,
)
from ht_pdm_fjsp.marl_diagnostic_experiment import (
    BROADCAST_MAPPO,
    INDEPENDENT_MAPPO,
    PS_IPPO,
    _audit_status,
    defaults,
)
from ht_pdm_fjsp.models import BenchmarkConfig


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs/minimal_benchmark.json")


def _settings() -> DiagnosticSettings:
    return DiagnosticSettings(
        total_timesteps=64,
        n_envs=1,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        entropy_coefficient=0.01,
        value_coefficient=0.5,
        max_grad_norm=0.5,
        actor_hidden_dim=32,
        critic_hidden_dim=64,
        device="cpu",
    )


@pytest.mark.parametrize(
    ("condition", "actor_mode", "critic_mode", "broadcast"),
    [
        (PS_IPPO, "shared", "local", False),
        (INDEPENDENT_MAPPO, "independent", "global", False),
        (BROADCAST_MAPPO, "shared", "global", True),
    ],
)
def test_diagnostic_variants_train_save_load_and_evaluate(
    tmp_path: Path,
    condition: str,
    actor_mode: str,
    critic_mode: str,
    broadcast: bool,
) -> None:
    output = tmp_path / condition
    model, elapsed = train_diagnostic_policy(
        CONFIG,
        _settings(),
        output,
        train_seed=10_000,
        actor_mode=actor_mode,
        critic_mode=critic_mode,
        include_broadcast_context=broadcast,
        show_progress=False,
    )
    assert elapsed >= 0
    assert (output / "policy.pt").is_file()
    assert len(list((output / "checkpoints").glob("*.pt"))) == 5
    loaded = DiagnosticPolicy.load(output / "policy.pt", device="cpu")
    rows, audit = evaluate_diagnostic_policy(
        loaded,
        CONFIG,
        [52_900, 52_901],
        condition=condition,
        device="cpu",
        show_progress=False,
    )
    assert len(rows) == 2
    assert all(np.isclose(row["episode_return"], -row["objective"]) for row in rows)
    assert audit["invalid_executions"] == 0
    assert model.actor_mode == loaded.actor_mode
    assert model.critic_mode == loaded.critic_mode


def test_full_profile_uses_fresh_panel_and_keeps_test_closed() -> None:
    profile = defaults("full")
    assert profile["train_seeds"] == (10_000, 11_000, 12_000, 13_000, 14_000)
    assert profile["validation_seeds"] == tuple(range(53_000, 53_200))
    assert set(profile["validation_seeds"]).isdisjoint(range(50_000, 50_100))


def test_ippo_shared_actor_initialization_matches_current_mappo() -> None:
    env = MachineAgentsCTDEEnv(CONFIG)
    th.manual_seed(10_000)
    current = MachineMAPPO(
        local_feature_dim=env.local_feature_dim,
        global_state_dim=env.global_state_dim,
        actor_hidden_dim=32,
        critic_hidden_dim=64,
    )
    th.manual_seed(10_000)
    ippo = DiagnosticPolicy(
        agent_count=env.num_agents,
        max_local_actions=env.max_local_actions,
        local_feature_dim=env.local_feature_dim,
        global_state_dim=env.global_state_dim,
        actor_mode="shared",
        critic_mode="local",
        include_broadcast_context=False,
        actor_hidden_dim=32,
        critic_hidden_dim=64,
    )
    assert all(
        th.equal(current_parameter, ippo_parameter)
        for current_parameter, ippo_parameter in zip(
            current.actor.parameters(), ippo.actors[0].parameters()
        )
    )


def test_coordination_audit_rejects_invalid_execution() -> None:
    assert _audit_status({"invalid_executions": 0})["status"] == "PASS"
    assert _audit_status({"invalid_executions": 1})["status"] == "FAIL"
