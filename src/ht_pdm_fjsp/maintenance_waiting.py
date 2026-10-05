"""Completion dispatch with an explicit soft cost of overdue corrective waiting."""

from dataclasses import asdict, dataclass
import math

import numpy as np

from ht_pdm_fjsp import maintenance_dispatch as base

ENV_VERSION = "pending_matching_overdue_waiting_v1"
WAIT_LIMIT = 4
WAIT_PRICE = 12.0
OBSERVATION_CONTRACT = {
    **base.OBSERVATION_CONTRACT,
    "global_features": 9,
    "waiting_limit_scale": 12.0,
    "waiting_price_scale": 20.0,
}
FAMILIES = (*base.FAMILIES, "n4_pressure_long")


@dataclass(frozen=True)
class WaitingConfig(base.DispatchConfig):
    waiting_limit: int = WAIT_LIMIT
    waiting_price: float = 0.0

    def validate(self):
        super().validate()
        if (
            not isinstance(self.waiting_limit, int)
            or self.waiting_limit < 0
            or not math.isfinite(self.waiting_price)
            or self.waiting_price < 0
        ):
            raise ValueError("nonnegative waiting limit and finite price required")


def priced(config, price):
    return WaitingConfig(
        **{**asdict(config), "waiting_limit": WAIT_LIMIT, "waiting_price": price}
    )


def training_config(seed, profile, price):
    return priced(base.training_config(seed, profile), price)


def family_config(family, seed, profile, price):
    if family == "n4_pressure_long":
        config = base.family_config("n4_pressure", seed, profile)
        return WaitingConfig(
            **{
                **asdict(config),
                "horizon": 24 if profile == "full" else 8,
                "waiting_price": price,
            }
        )
    return priced(base.family_config(family, seed, profile), price)


def overdue_ticks(config, state, pairs):
    servicing = {m for m in state.assigned if m >= 0} | {m for m, _ in pairs}
    return sum(
        failed
        and m not in servicing
        and state.pending_wait[m] + 1 > config.waiting_limit
        for m, failed in enumerate(state.failed)
    )


def transition(config, state, pairs, events):
    pairs = tuple(pairs)
    following, economic, info = base.transition(config, state, pairs, events)
    overdue = overdue_ticks(config, state, pairs)
    waiting = config.waiting_price * overdue
    info.update(
        base_cost=economic,
        waiting_cost=waiting,
        overdue_waiting_ticks=overdue,
        service_adjusted_cost=economic + WAIT_PRICE * overdue,
        objective=economic + waiting,
    )
    return following, economic + waiting, info


def augment_observation(obs, limit, price):
    return {
        **obs,
        "global_features": np.concatenate(
            (
                obs["global_features"],
                np.asarray([limit / 12.0, price / 20.0], dtype=np.float32),
            )
        ),
    }


def observation(config, state, time):
    return augment_observation(
        base.observation(config, state, time),
        config.waiting_limit,
        config.waiting_price,
    )


class WaitingEnv:
    def __init__(self, config):
        config.validate()
        self.config = config
        self.reset(0)

    def reset(self, seed, initial=None):
        # Use precisely the original streams, state and eligibility-independent shocks.
        env = base.DispatchEnv(self.config)
        env.reset(seed, initial)
        self.state, self.events, self.time = env.state, env.events, 0
        self.metrics = {
            **env.metrics,
            "base_cost": 0.0,
            "waiting_cost": 0.0,
            "overdue_waiting_ticks": 0.0,
            "service_adjusted_cost": 0.0,
        }
        return observation(self.config, self.state, self.time)

    def step(self, pairs):
        if self.time >= self.config.horizon:
            raise RuntimeError("episode finished")
        self.state, cost, info = transition(
            self.config, self.state, pairs, self.events[self.time]
        )
        self.time += 1
        for key, value in info.items():
            self.metrics[key] += value
        economic = sum(
            self.metrics[k]
            for k in ("maintenance_cost", "unavailability_cost", "failure_cost")
        )
        if (
            abs(self.metrics["base_cost"] - economic) > 1e-8
            or abs(self.metrics["objective"] - economic - self.metrics["waiting_cost"])
            > 1e-8
        ):
            raise RuntimeError("waiting objective reconciliation")
        return (
            observation(self.config, self.state, self.time),
            -cost,
            self.time == self.config.horizon,
            info,
        )
