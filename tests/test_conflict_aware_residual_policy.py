from __future__ import annotations

from pathlib import Path

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from ht_pdm_fjsp.conflict_aware_residual_policy import (
    ConflictAwareRoutePreservingDistribution,
    ConflictAwareRoutePreservingResidualPolicy,
    build_production_conflict_matrix,
)
from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.shared_action_policy import SharedActionMaskablePolicy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs/minimal_benchmark.json")


def _conflicts() -> th.Tensor:
    # Actions 1/2 conflict; action 3 is a separate production conflict set.
    return th.tensor(
        [
            [False, False, False, False, False],
            [False, True, True, False, False],
            [False, True, True, False, False],
            [False, False, False, True, False],
            [False, False, False, False, False],
        ]
    )


def test_conflict_distribution_preserves_route_mass_and_exact_marginal() -> None:
    base = th.tensor([[0.1, 1.5, 0.8, 0.6, 0.2]])
    residual = th.tensor([[0.0, -1.0, 1.0, 20.0, 0.0]])
    production = th.tensor([[False, True, True, True, False]])
    distribution = ConflictAwareRoutePreservingDistribution(5).proba_distribution(
        base,
        residual_logits=residual,
        production_mask=production,
        conflict_matrix=_conflicts(),
    )
    before = distribution.base_distribution.probs
    after = distribution.distribution.probs
    assert th.allclose(before[:, ~production[0]], after[:, ~production[0]])
    assert th.allclose(
        before[:, production[0]].sum(dim=1),
        after[:, production[0]].sum(dim=1),
    )
    # Action 3 cannot absorb probability from the separate 1/2 conflict set.
    assert th.allclose(before[:, 3], after[:, 3])
    # Within the 1/2 set, the residual transfers probability toward action 2.
    assert after[0, 2] > before[0, 2]


def test_deterministic_mode_cannot_cross_conflict_sets() -> None:
    base = th.tensor([[0.0, 3.0, 2.0, 1.0, 0.5]])
    residual = th.tensor([[0.0, -2.0, 2.0, 100.0, 0.0]])
    production = th.tensor([[False, True, True, True, False]])
    distribution = ConflictAwareRoutePreservingDistribution(5).proba_distribution(
        base,
        residual_logits=residual,
        production_mask=production,
        conflict_matrix=_conflicts(),
    )
    action = distribution.mode()
    assert distribution.last_baseline_actions is not None
    assert distribution.last_baseline_actions.item() == 1
    assert action.item() == 2


def test_stochastic_sampler_respects_each_sampled_conflict_set() -> None:
    th.manual_seed(11)
    repeats = 1024
    base = th.tensor([[0.3, 0.2, 0.1, 0.4, 0.5]]).repeat(repeats, 1)
    residual = th.tensor([[0.0, -4.0, 4.0, 20.0, 0.0]]).repeat(repeats, 1)
    production = th.tensor(
        [[False, True, True, True, False]]
    ).repeat(repeats, 1)
    distribution = ConflictAwareRoutePreservingDistribution(5).proba_distribution(
        base,
        residual_logits=residual,
        production_mask=production,
        conflict_matrix=_conflicts(),
    )
    actions = distribution.sample()
    baseline = distribution.last_baseline_actions
    assert baseline is not None
    for sampled_base, sampled_action in zip(
        baseline.tolist(), actions.tolist(), strict=True
    ):
        if sampled_base in {1, 2}:
            assert sampled_action in {1, 2}
        elif sampled_base == 3:
            assert sampled_action == 3
        else:
            assert sampled_action == sampled_base


def test_environment_conflict_graph_is_symmetric_and_nontrivial() -> None:
    env = HTPdmFjspEnv(config=CONFIG, include_production_context=True)
    matrix = np.asarray(build_production_conflict_matrix(env.actions))
    assert matrix.shape == (env.action_space.n, env.action_space.n)
    assert np.array_equal(matrix, matrix.T)
    for index, descriptor in enumerate(env.actions):
        assert matrix[index, index] == (descriptor.kind == "production")
    production_pairs = [
        (left, right)
        for left, left_descriptor in enumerate(env.actions)
        for right, right_descriptor in enumerate(env.actions)
        if left_descriptor.kind == right_descriptor.kind == "production"
        and left != right
    ]
    assert any(matrix[left, right] for left, right in production_pairs)
    assert any(not matrix[left, right] for left, right in production_pairs)


def test_policy_trains_and_roundtrips_with_static_conflict_graph(
    tmp_path: Path,
) -> None:
    base_env = HTPdmFjspEnv(config=CONFIG)
    context_env = HTPdmFjspEnv(config=CONFIG, include_production_context=True)
    shared = MaskablePPO(
        SharedActionMaskablePolicy,
        Monitor(base_env),
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        seed=1,
        device="cpu",
        verbose=0,
    )
    conflicts = build_production_conflict_matrix(context_env.actions)
    model = MaskablePPO(
        ConflictAwareRoutePreservingResidualPolicy,
        Monitor(context_env),
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        seed=1,
        device="cpu",
        verbose=0,
        policy_kwargs={"production_conflict_matrix": conflicts},
    )
    model.policy.load_shared_base(shared.policy)
    model.learn(total_timesteps=64)
    path = tmp_path / "conflict_policy"
    model.save(path)
    loaded = MaskablePPO.load(path, device="cpu")
    observation, _ = context_env.reset(seed=48_000)
    mask = context_env.action_masks()
    action, _ = loaded.predict(observation, action_masks=mask, deterministic=True)
    assert mask[int(np.asarray(action).item())]
