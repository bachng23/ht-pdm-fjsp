from __future__ import annotations

import argparse
import json
from pathlib import Path

from ht_pdm_fjsp.marl_budget_screening import run


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "brandimarte_mk01_ht_pdm.json"


def test_smoke_budget_screen_writes_five_checkpoints_per_algorithm(
    tmp_path: Path,
) -> None:
    output = tmp_path / "marl_budget_screen_smoke_20260923T000000Z"
    run(
        argparse.Namespace(
            config=str(CONFIG_PATH),
            profile="smoke",
            train_seed=76100,
            evaluation_seeds="61990:61992",
            total_timesteps=200,
            device="cpu",
            output_dir=str(output),
        )
    )
    manifest = json.loads(
        (output / "marl_budget_screening_manifest.json").read_text()
    )
    summary = json.loads(
        (output / "marl_budget_screening_summary.json").read_text()
    )
    assert manifest["status"] == "COMPLETED"
    assert manifest["machine_count"] == 6
    assert manifest["episode_count"] == 30
    assert manifest["value_settings"]["n_envs"] == 4
    assert summary["selected_common_replicated_run_budget"] is None
    assert summary["gate"]["passed"]
    for condition in manifest["conditions"]:
        assert len(list((output / condition / "checkpoints").glob("*.pt"))) == 5
