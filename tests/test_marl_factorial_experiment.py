from __future__ import annotations

from ht_pdm_fjsp.marl_factorial_experiment import (
    COMBINED_MAPPO,
    INDEPENDENT_MAPPO,
    MATCHED_SHARED_MAPPO,
    _factorial_interaction,
    _paired_contrast,
    actor_parameter_count,
    defaults,
    matched_shared_hidden_dim,
    promotion_decision,
)
from ht_pdm_fjsp.shared_policy_experiment import METRICS


def _summary() -> dict:
    conditions = (
        "parameter_shared_mappo_global_critic",
        INDEPENDENT_MAPPO,
        "parameter_shared_mappo_broadcast_context",
        COMBINED_MAPPO,
        MATCHED_SHARED_MAPPO,
    )
    offsets = {
        "parameter_shared_mappo_global_critic": 10.0,
        INDEPENDENT_MAPPO: 8.0,
        "parameter_shared_mappo_broadcast_context": 7.0,
        COMBINED_MAPPO: 4.0,
        MATCHED_SHARED_MAPPO: 9.0,
    }
    return {
        "per_condition": {
            condition: {
                "per_training_seed": {
                    str(seed): {
                        metric: offsets[condition] + seed / 100_000
                        for metric in METRICS
                    }
                    for seed in (10_000, 11_000, 12_000, 13_000, 14_000)
                }
            }
            for condition in conditions
        }
    }


def test_full_profile_uses_fresh_panel_and_keeps_final_test_closed() -> None:
    profile = defaults("full")
    assert profile["validation_seeds"] == tuple(range(54_000, 54_200))
    assert set(profile["validation_seeds"]).isdisjoint(range(50_000, 50_100))


def test_matched_shared_actor_capacity_is_within_half_percent() -> None:
    width = matched_shared_hidden_dim(
        input_dim=23, independent_hidden_dim=64, agent_count=2
    )
    independent = actor_parameter_count(23, 64, actor_count=2)
    matched = actor_parameter_count(23, width)
    assert width == 95
    assert independent == 11_522
    assert matched == 11_496
    assert abs(matched - independent) / independent < 0.005


def test_primary_capacity_and_factorial_contrasts() -> None:
    summary = _summary()
    seeds = (10_000, 11_000, 12_000, 13_000, 14_000)
    primary = _paired_contrast(
        summary,
        treatment=COMBINED_MAPPO,
        control=INDEPENDENT_MAPPO,
        train_seeds=seeds,
    )
    capacity = _paired_contrast(
        summary,
        treatment=INDEPENDENT_MAPPO,
        control=MATCHED_SHARED_MAPPO,
        train_seeds=seeds,
    )
    interaction = _factorial_interaction(summary, seeds)
    assert primary["delta_across_training_seeds"]["objective"]["mean"] == -4.0
    assert capacity["delta_across_training_seeds"]["objective"]["mean"] == -1.0
    assert interaction["delta_across_training_seeds"]["objective"]["mean"] == -1.0


def test_promotion_requires_all_combined_policy_gates() -> None:
    contrast = {
        "delta_across_training_seeds": {
            "objective": {"mean": -1.0},
            "failures": {"mean": 0.0},
        },
        "per_training_seed": {
            str(seed): {"objective": delta}
            for seed, delta in zip(
                (10_000, 11_000, 12_000, 13_000, 14_000),
                (-1.0, -0.5, 0.0, -0.2, 0.1),
            )
        },
    }
    audits = {COMBINED_MAPPO: {"status": "PASS"}}
    assert promotion_decision(contrast, audits)[
        "eligible_for_future_held_out_test"
    ]
    contrast["delta_across_training_seeds"]["failures"]["mean"] = 0.1
    assert not promotion_decision(contrast, audits)[
        "eligible_for_future_held_out_test"
    ]
