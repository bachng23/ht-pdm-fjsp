"""Shared scorer with a bounded, zero-initialized production-context residual."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
from torch import nn

from ht_pdm_fjsp.shared_action_policy import SharedActionMaskablePolicy


class ResidualContextMaskablePolicy(SharedActionMaskablePolicy):
    """Preserve a shared actor and learn only a bounded production correction."""

    def __init__(
        self,
        *args: Any,
        context_feature_key: str = "production_context",
        context_hidden_dim: int = 64,
        max_residual_logit: float = 1.0,
        freeze_base_actor: bool = True,
        **kwargs: Any,
    ) -> None:
        observation_space = args[0]
        if context_feature_key not in observation_space.spaces:
            raise ValueError(f"Missing residual context: {context_feature_key}.")
        self.context_feature_key = context_feature_key
        self.context_feature_dim = observation_space[context_feature_key].shape[1]
        self.context_hidden_dim = context_hidden_dim
        self.max_residual_logit = max_residual_logit
        self.freeze_base_actor = freeze_base_actor
        super().__init__(*args, extra_action_feature_keys=(), **kwargs)

    def _build(self, lr_schedule) -> None:
        super()._build(lr_schedule)
        activation = self.activation_fn
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_feature_dim, self.context_hidden_dim),
            activation(),
        ).to(self.device)
        self.context_scorer = nn.Linear(self.context_hidden_dim, 1).to(self.device)
        self.context_encoder.apply(partial(self.init_weights, gain=np.sqrt(2)))
        nn.init.zeros_(self.context_scorer.weight)
        nn.init.zeros_(self.context_scorer.bias)
        if self.freeze_base_actor:
            for module in (
                self.global_encoder,
                self.action_encoder,
                self.action_scorer,
            ):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
        self.optimizer = self.optimizer_class(
            (parameter for parameter in self.parameters() if parameter.requires_grad),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def load_shared_base(self, source: SharedActionMaskablePolicy) -> None:
        """Copy the incumbent actor and critic into the compatible base modules."""

        for target, origin in (
            (self.global_encoder, source.global_encoder),
            (self.action_encoder, source.action_encoder),
            (self.action_scorer, source.action_scorer),
            (self.value_net, source.value_net),
        ):
            target.load_state_dict(origin.state_dict())

    def context_residual(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        latent = self.context_encoder(obs[self.context_feature_key])
        raw = self.context_scorer(latent).squeeze(-1)
        production_indicator = obs["action_features"][..., 1]
        return (
            self.max_residual_logit
            * th.tanh(raw)
            * production_indicator
        )

    def base_action_logits(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        """Return the frozen shared-actor logits without any residual."""

        return super().action_logits(obs)

    def action_logits(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        return self.base_action_logits(obs) + self.context_residual(obs)
