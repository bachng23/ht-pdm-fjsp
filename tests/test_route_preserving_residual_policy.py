from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.route_preserving_residual_experiment import (
    defaults,
    summarize_route_audits,
)
from ht_pdm_fjsp.route_preserving_residual_policy import (
    RoutePreservingCategoricalDistribution,
    RoutePreservingResidualContextPolicy,
)
from ht_pdm_fjsp.shared_action_policy import SharedActionMaskablePolicy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs/minimal_benchmark.json")


def _model(policy, *, context: bool = False) -> MaskablePPO:
    return MaskablePPO(
        policy,
        Monitor(
            HTPdmFjspEnv(
                config=CONFIG,
                include_production_context=context,
            )
        ),
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        seed=1,
        device="cpu",
        verbose=0,
    )


def test_distribution_preserves_route_mass_and_nonproduction_probabilities() -> None:
    base = th.tensor([[1.2, 0.4, -0.3, 0.8, -0.1]])
    residual = th.tensor([[0.0, 2.0, -2.0, 0.0, 0.0]])
    production = th.tensor([[False, True, True, False, False]])
    valid = np.asarray([[True, True, True, True, False]])
    distribution = RoutePreservingCategoricalDistribution(5).proba_distribution(
        base,
        residual_logits=residual,
        production_mask=production,
    )
    distribution.apply_masking(valid)
    before = distribution.base_distribution.probs
    after = distribution.distribution.probs
    assert th.allclose(before[:, ~production[0]], after[:, ~production[0]])
    assert th.allclose(
        before[:, production[0]].sum(dim=1),
        after[:, production[0]].sum(dim=1),
    )
    assert after[0, 1] > before[0, 1]
    assert after[0, 2] < before[0, 2]
    assert after[0, 4] == 0


def test_deterministic_mode_preserves_base_kind_and_nonproduction_action() -> None:
    production = th.tensor(
        [[False, True, True, False], [False, True, True, False]]
    )
    base = th.tensor(
        [[3.0, 2.0, 1.0, 0.0], [0.0, 3.0, 2.0, 1.0]]
    )
    residual = th.tensor(
        [[0.0, 20.0, -20.0, 0.0], [0.0, -20.0, 20.0, 0.0]]
    )
    distribution = RoutePreservingCategoricalDistribution(4).proba_distribution(
        base,
        residual_logits=residual,
        production_mask=production,
    )
    actions = distribution.mode()
    assert actions.tolist() == [0, 2]
    assert distribution.last_baseline_actions.tolist() == [0, 1]


def test_stochastic_sampling_couples_every_route_kind() -> None:
    th.manual_seed(7)
    base = th.tensor([[0.3, 0.2, 0.1, 0.4]]).repeat(512, 1)
    residual = th.tensor([[0.0, -4.0, 4.0, 0.0]]).repeat(512, 1)
    production = th.tensor([[False, True, True, False]]).repeat(512, 1)
    distribution = RoutePreservingCategoricalDistribution(4).proba_distribution(
        base,
        residual_logits=residual,
        production_mask=production,
    )
    actions = distribution.sample()
    baseline = distribution.last_baseline_actions
    assert baseline is not None
    baseline_production = production.gather(1, baseline[:, None]).squeeze(1)
    action_production = production.gather(1, actions[:, None]).squeeze(1)
    assert th.equal(baseline_production, action_production)
    assert th.equal(actions[~baseline_production], baseline[~baseline_production])


def test_policy_starts_at_shared_route_and_can_only_rerank_production(
    tmp_path: Path,
) -> None:
    shared = _model(SharedActionMaskablePolicy)
    residual = _model(RoutePreservingResidualContextPolicy, context=True)
    residual.policy.load_shared_base(shared.policy)
    env = HTPdmFjspEnv(config=CONFIG, include_production_context=True)
    observation, _ = env.reset(seed=17)
    tensor, _ = residual.policy.obs_to_tensor(observation)
    mask = env.action_masks()
    with th.no_grad():
        distribution = residual.policy._distribution(tensor, mask)
        action = distribution.mode()
    baseline_action = distribution.last_baseline_actions
    assert baseline_action is not None
    assert action.item() == baseline_action.item()
    assert th.count_nonzero(residual.policy.context_residual(tensor)) == 0

    residual.policy.context_scorer.bias.data.fill_(0.5)
    residual.learn(total_timesteps=64)
    path = tmp_path / "route_preserving_policy_test"
    residual.save(path)
    loaded = MaskablePPO.load(path, device="cpu")
    predicted, _ = loaded.predict(
        observation,
        action_masks=mask,
        deterministic=True,
    )
    assert mask[int(np.asarray(predicted).item())]
    env.close()


def test_training_updates_residual_but_not_frozen_base_actor() -> None:
    model = _model(RoutePreservingResidualContextPolicy, context=True)
    base_before = {
        name: parameter.detach().clone()
        for module_name in ("global_encoder", "action_encoder", "action_scorer")
        for name, parameter in getattr(model.policy, module_name).named_parameters(
            prefix=module_name
        )
    }
    residual_before = {
        name: parameter.detach().clone()
        for name, parameter in model.policy.context_scorer.named_parameters()
    }
    model.learn(total_timesteps=128)
    base_after = {
        name: parameter.detach()
        for module_name in ("global_encoder", "action_encoder", "action_scorer")
        for name, parameter in getattr(model.policy, module_name).named_parameters(
            prefix=module_name
        )
    }
    assert all(th.equal(base_before[name], base_after[name]) for name in base_before)
    assert any(
        not th.equal(residual_before[name], parameter.detach())
        for name, parameter in model.policy.context_scorer.named_parameters()
    )


def test_route_audit_summary_requires_all_invariants() -> None:
    rows = [
        {
            "episodes": 5,
            "decisions": 50,
            "production_baseline_decisions": 30,
            "production_reroutes": 10,
            "decision_kind_mismatches": 0,
            "nonproduction_action_mismatches": 0,
            "invalid_actions": 0,
        }
    ]
    assert summarize_route_audits(rows)["status"] == "PASS"
    rows[0]["decision_kind_mismatches"] = 1
    assert summarize_route_audits(rows)["status"] == "FAIL"


def test_full_profile_uses_fresh_validation_and_keeps_test_closed() -> None:
    full = defaults("full")
    assert full["train_seeds"] == (10_000, 11_000, 12_000, 13_000, 14_000)
    assert len(full["validation_seeds"]) == 200
    assert set(full["validation_seeds"]).isdisjoint(range(50_000, 50_100))
