"""Minimal health- and technician-aware PdM-FJSP benchmark."""

from pathlib import Path

import gymnasium as gym

from ht_pdm_fjsp.models import BenchmarkConfig
from ht_pdm_fjsp.policies import HealthThresholdSPTPolicy, ProductionFirstSPTPolicy
from ht_pdm_fjsp.gym_env import GymEnvConfig, HTPdmFjspEnv
from ht_pdm_fjsp.simulator import SimulationResult, Simulator, audit_result

__all__ = [
    "BenchmarkConfig",
    "HealthThresholdSPTPolicy",
    "GymEnvConfig",
    "HTPdmFjspEnv",
    "ProductionFirstSPTPolicy",
    "SimulationResult",
    "Simulator",
    "audit_result",
]

__version__ = "0.1.0"

if "HTPdMFJSP-v0" not in gym.envs.registry:
    gym.register(
        id="HTPdMFJSP-v0",
        entry_point="ht_pdm_fjsp.gym_env:HTPdmFjspEnv",
        kwargs={
            "config_path": str(
                Path(__file__).resolve().parents[2]
                / "configs"
                / "minimal_benchmark.json"
            )
        },
    )
