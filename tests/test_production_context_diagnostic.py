from __future__ import annotations

import math
from pathlib import Path

from sb3_contrib import MaskablePPO
from stable_baselines3.common.monitor import Monitor

from ht_pdm_fjsp.gym_env import HTPdmFjspEnv
from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.production_context_diagnostic import (
    EPISODE_METRICS,
    STATE_METRICS,
    collect_episode,
    diagnostic_defaults,
    model_weight_diagnostics,
    summarize_diagnostic,
)
from ht_pdm_fjsp.shared_action_policy import SharedActionMaskablePolicy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs" / "minimal_benchmark.json")


def _model(*, production_context: bool) -> MaskablePPO:
    return MaskablePPO(
        SharedActionMaskablePolicy,
        Monitor(
            HTPdmFjspEnv(
                config=CONFIG,
                include_production_context=production_context,
            )
        ),
        policy_kwargs={
            "extra_action_feature_keys": (
                ("production_context",) if production_context else ()
            )
        },
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        seed=123,
        device="cpu",
        verbose=0,
    )


def test_diagnostic_profile_reserves_fresh_panel() -> None:
    smoke = diagnostic_defaults("smoke")
    full = diagnostic_defaults("full")
    assert len(smoke["train_seeds"]) == 2
    assert len(full["diagnostic_seeds"]) == 100
    assert set(full["diagnostic_seeds"]).isdisjoint(range(40_000, 45_000))
    assert set(full["diagnostic_seeds"]).isdisjoint(range(50_000, 50_100))


def test_collect_episode_scores_normal_and_zero_context_on_same_states() -> None:
    control = _model(production_context=False)
    treatment = _model(production_context=True)
    episode, states = collect_episode(
        control_model=control,
        treatment_model=treatment,
        config=CONFIG,
        behavior_role="control",
        train_seed=123,
        environment_seed=45_000,
    )
    assert states
    assert episode["decision_count"] == len(states)
    assert all(0.0 <= row["control_normalized_entropy"] <= 1.0 for row in states)
    assert all(0.0 <= row["treatment_normalized_entropy"] <= 1.0 for row in states)
    assert all(row["normal_zero_js_divergence"] >= 0.0 for row in states)
    assert all(
        math.isclose(
            sum(row[f"treatment_probability_{kind}"] for kind in (
                "advance",
                "production",
                "preventive",
                "corrective",
            )),
            1.0,
            abs_tol=1e-6,
        )
        for row in states
    )
    weights = model_weight_diagnostics(treatment)
    assert weights["base_weight_rms"] > 0.0
    assert weights["context_weight_rms"] > 0.0


def test_summary_uses_training_seed_as_replication_unit() -> None:
    state_rows = []
    episode_rows = []
    model_rows = []
    for train_seed, value in ((10, 0.1), (11, 0.3)):
        model_rows.append(
            {
                "train_seed": train_seed,
                "base_weight_rms": 1.0,
                "context_weight_rms": value,
                "context_base_weight_rms_ratio": value,
            }
        )
        for role in ("control", "treatment"):
            state_rows.append(
                {
                    "behavior_role": role,
                    "train_seed": train_seed,
                    **{metric: value for metric in STATE_METRICS},
                }
            )
            episode_rows.append(
                {
                    "behavior_role": role,
                    "train_seed": train_seed,
                    **{metric: 1.0 for metric in EPISODE_METRICS},
                }
            )
    summary = summarize_diagnostic(state_rows, episode_rows, model_rows)
    result = summary["behavior_distributions"]["control"]
    assert result["across_training_seeds"]["delta_normalized_entropy"][
        "mean"
    ] == 0.2
    assert summary["model_weights"]["across_training_seeds"][
        "context_base_weight_rms_ratio"
    ]["mean"] == 0.2
