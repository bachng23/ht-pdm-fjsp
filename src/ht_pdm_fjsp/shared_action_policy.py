"""MaskablePPO policy with a permutation-equivariant shared action scorer."""

from __future__ import annotations

from functools import partial
from typing import Any

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from sb3_contrib.common.maskable.distributions import MaskableDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs, Schedule


class SharedActionMaskablePolicy(MaskableMultiInputActorCriticPolicy):
    """Score each action with the same network using state and action features."""

    GLOBAL_KEYS = ("time", "jobs", "machines", "technicians")

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        *,
        global_hidden_dim: int = 128,
        action_hidden_dim: int = 64,
        scorer_hidden_dim: int = 128,
        **kwargs: Any,
    ) -> None:
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError("SharedActionMaskablePolicy requires a discrete action space.")
        action_feature_space = observation_space.spaces["action_features"]
        if len(action_feature_space.shape) != 2:
            raise ValueError("action_features must have shape (actions, features).")
        if action_feature_space.shape[0] != action_space.n:
            raise ValueError("Action feature count does not match action space.")
        self.action_count = action_space.n
        # The final feature is a duplicate of the separately applied action mask.
        self.action_feature_dim = action_feature_space.shape[1] - 1
        self.global_input_dim = sum(
            int(np.prod(observation_space.spaces[key].shape))
            for key in self.GLOBAL_KEYS
        )
        self.critic_input_dim = self.global_input_dim + int(
            np.prod(action_feature_space.shape)
        ) + action_space.n
        self.global_hidden_dim = global_hidden_dim
        self.action_hidden_dim = action_hidden_dim
        self.scorer_hidden_dim = scorer_hidden_dim
        kwargs.setdefault("ortho_init", False)
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            **kwargs,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        """Build a shared actor scorer and a separate global critic."""

        activation = self.activation_fn
        self.global_encoder = nn.Sequential(
            nn.Linear(self.global_input_dim, self.global_hidden_dim),
            activation(),
            nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
            activation(),
        ).to(self.device)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.action_feature_dim, self.action_hidden_dim),
            activation(),
        ).to(self.device)
        self.action_scorer = nn.Sequential(
            nn.Linear(
                self.global_hidden_dim + self.action_hidden_dim,
                self.scorer_hidden_dim,
            ),
            activation(),
            nn.Linear(self.scorer_hidden_dim, 1),
        ).to(self.device)
        self.value_net = nn.Sequential(
            nn.Linear(self.critic_input_dim, self.global_hidden_dim),
            activation(),
            nn.Linear(self.global_hidden_dim, self.global_hidden_dim),
            activation(),
            nn.Linear(self.global_hidden_dim, 1),
        ).to(self.device)
        # Compatibility attributes; all actor/critic calls are overridden below.
        self.mlp_extractor = nn.Identity()
        self.action_net = nn.Identity()

        for module in (self.global_encoder, self.action_encoder):
            module.apply(partial(self.init_weights, gain=np.sqrt(2)))
        self.action_scorer[0].apply(partial(self.init_weights, gain=np.sqrt(2)))
        self.action_scorer[-1].apply(partial(self.init_weights, gain=0.01))
        for layer in self.value_net[:-1]:
            layer.apply(partial(self.init_weights, gain=np.sqrt(2)))
        self.value_net[-1].apply(partial(self.init_weights, gain=1.0))
        self.optimizer = self.optimizer_class(
            self.parameters(),
            lr=lr_schedule(1),
            **self.optimizer_kwargs,
        )

    def _global_input(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        return th.cat(
            [obs[key].flatten(start_dim=1) for key in self.GLOBAL_KEYS], dim=1
        )

    def _critic_input(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        return th.cat(
            (
                self._global_input(obs),
                obs["action_features"].flatten(start_dim=1),
                obs["action_mask"].flatten(start_dim=1),
            ),
            dim=1,
        )

    def action_logits(self, obs: dict[str, th.Tensor]) -> th.Tensor:
        """Return one logit per action using a shared scoring network."""

        global_latent = self.global_encoder(self._global_input(obs))
        action_features = obs["action_features"][..., : self.action_feature_dim]
        action_latent = self.action_encoder(action_features)
        expanded_global = global_latent.unsqueeze(1).expand(
            -1, self.action_count, -1
        )
        scorer_input = th.cat((expanded_global, action_latent), dim=-1)
        return self.action_scorer(scorer_input).squeeze(-1)

    def _distribution(
        self,
        obs: dict[str, th.Tensor],
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> MaskableDistribution:
        distribution = self.action_dist.proba_distribution(
            action_logits=self.action_logits(obs)
        )
        if action_masks is not None:
            distribution.apply_masking(action_masks)
        return distribution

    def forward(
        self,
        obs: dict[str, th.Tensor],
        deterministic: bool = False,
        action_masks: np.ndarray | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        distribution = self._distribution(obs, action_masks)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(self._critic_input(obs))
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob

    def evaluate_actions(
        self,
        obs: dict[str, th.Tensor],
        actions: th.Tensor,
        action_masks: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor | None]:
        distribution = self._distribution(obs, action_masks)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(self._critic_input(obs))
        return values, log_prob, distribution.entropy()

    def get_distribution(
        self,
        obs: PyTorchObs,
        action_masks: np.ndarray | None = None,
    ) -> MaskableDistribution:
        if not isinstance(obs, dict):
            raise TypeError("Shared action policy requires dict observations.")
        return self._distribution(obs, action_masks)

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        if not isinstance(obs, dict):
            raise TypeError("Shared action policy requires dict observations.")
        return self.value_net(self._critic_input(obs))
