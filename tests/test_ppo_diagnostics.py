from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.ppo_diagnostics import summarize_training_log, trace_policy


ROOT = Path(__file__).resolve().parents[1]
CONFIG = BenchmarkConfig.from_json(ROOT / "configs" / "minimal_benchmark.json")


class FirstFeasiblePolicy:
    def predict(self, observation, *, action_masks, deterministic):
        del observation, deterministic
        return np.flatnonzero(action_masks)[0], None


def test_training_log_summary_tracks_entropy_and_reward_gap(tmp_path: Path) -> None:
    path = tmp_path / "progress.csv"
    rows = [
        {
            "time/total_timesteps": "100",
            "rollout/ep_rew_mean": "-20",
            "train/entropy_loss": "-1.5",
        },
        {
            "time/total_timesteps": "200",
            "rollout/ep_rew_mean": "-10",
            "train/entropy_loss": "-1.0",
        },
        {
            "time/total_timesteps": "300",
            "rollout/ep_rew_mean": "-15",
            "train/entropy_loss": "-0.5",
        },
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_training_log(path, train_seed=10)
    assert summary["entropy_final"] == 0.5
    assert summary["rollout_reward_best"] == -10.0
    assert summary["rollout_reward_best_step"] == 200
    assert summary["rollout_reward_final_gap_from_best"] == -5.0


def test_trace_policy_records_actions_masks_and_reward_components() -> None:
    episodes, decisions = trace_policy(
        FirstFeasiblePolicy(), CONFIG, (41_000,), train_seed=10
    )
    assert len(episodes) == 1
    assert decisions
    episode = episodes[0]
    assert episode["decision_count"] == len(decisions)
    assert episode["invalid_actions"] == 0
    assert episode["min_feasible_actions"] >= 1
    assert np.isclose(episode["episode_return"], -episode["objective"])
    assert sum(
        episode[f"actions_{kind}"]
        for kind in ("advance", "production", "preventive", "corrective")
    ) == len(decisions)
    assert all(row["feasible_actions"] >= 1 for row in decisions)
