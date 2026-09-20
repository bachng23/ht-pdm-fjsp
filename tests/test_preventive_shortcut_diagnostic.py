from __future__ import annotations

from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.preventive_shortcut_diagnostic import (
    PAIRS,
    collect_episode,
    diagnostic_defaults,
    summarize_shortcut,
)
from ht_pdm_fjsp.shared_action_policy import SharedActionMaskablePolicy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs" / "minimal_benchmark.json")


def _model(*, entity_conditioned: bool) -> MaskablePPO:
    return MaskablePPO(
        SharedActionMaskablePolicy,
        Monitor(
            HTPdmFjspEnv(
                config=CONFIG,
                include_action_context=entity_conditioned,
            )
        ),
        policy_kwargs={
            "extra_action_feature_keys": (
                ("action_context",) if entity_conditioned else ()
            )
        },
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        seed=123,
        device="cpu",
        verbose=0,
    )


def test_diagnostic_profiles_keep_future_test_separate() -> None:
    smoke = diagnostic_defaults("smoke")
    full = diagnostic_defaults("full")
    assert len(smoke["train_seeds"]) == 2
    assert len(full["diagnostic_seeds"]) == 100
    assert set(full["diagnostic_seeds"]).isdisjoint(range(40_000, 43_000))
    assert set(full["diagnostic_seeds"]).isdisjoint(range(50_000, 50_100))


def test_counterfactual_scoring_uses_identical_states() -> None:
    episode, states = collect_episode(
        control_model=_model(entity_conditioned=False),
        treatment_model=_model(entity_conditioned=True),
        config=CONFIG,
        pair="plain",
        behavior_role="control",
        train_seed=123,
        environment_seed=43_000,
    )
    assert episode["decision_count"] == len(states)
    assert states
    assert all(
        abs(
            sum(float(row[f"control_probability_{kind}"]) for kind in (
                "advance",
                "production",
                "preventive",
                "corrective",
            ))
            - 1.0
        )
        < 1e-6
        for row in states
    )


def test_shortcut_summary_uses_training_seed_as_replication_unit() -> None:
    state_rows = []
    episode_rows = []
    for pair in PAIRS:
        for role in ("control", "treatment"):
            for train_seed, delta in ((10, 0.1), (11, 0.3)):
                state_rows.append(
                    {
                        "pair": pair,
                        "behavior_role": role,
                        "train_seed": train_seed,
                        "preventive_opportunity": True,
                        "control_probability_preventive": 0.2,
                        "treatment_probability_preventive": 0.2 + delta,
                        "delta_probability_preventive": delta,
                        "control_action_kind": "production",
                        "treatment_action_kind": "preventive",
                        "policies_agree": False,
                        "M1_preventive_feasible": True,
                        "M1_age_ratio": 0.5,
                        "control_probability_preventive_M1": 0.2,
                        "treatment_probability_preventive_M1": 0.2 + delta,
                        "delta_probability_preventive_M1": delta,
                        "max_feasible_production_risk": 0.25,
                    }
                )
                episode_rows.append(
                    {
                        "pair": pair,
                        "behavior_role": role,
                        "train_seed": train_seed,
                        **{
                            metric: 1.0
                            for metric in (
                                "objective",
                                "makespan",
                                "total_tardiness",
                                "failures",
                                "preventive_maintenance",
                                "corrective_maintenance",
                                "total_cost",
                            )
                        },
                    }
                )
    summary = summarize_shortcut(state_rows, episode_rows)
    result = summary["pairs"]["plain"]["control"]
    assert result["delta_preventive_probability_across_training_seeds"][
        "mean"
    ] == 0.2
    assert result["delta_preventive_take_rate_across_training_seeds"][
        "mean"
    ] == 1.0
