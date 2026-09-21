"""Residual production ranking with the shared route decision held fixed."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch as th

from sb3_contrib.common.maskable.distributions import (
    MaskableCategorical,
    MaskableCategoricalDistribution,
    MaybeMasks,
)

from ht_pdm_fjsp.residual_context_policy import ResidualContextMaskablePolicy


class RoutePreservingCategoricalDistribution(MaskableCategoricalDistribution):
    """Keep the base route sample and rerank only production alternatives.

    The resulting categorical probabilities preserve the base probability of
    every non-production action and the base total probability mass assigned to
    production.  Sampling is coupled: first sample from the base distribution;
    only a sampled production action is replaced by a draw from the residual
    production-conditional distribution.  Deterministic mode follows the same
    rule using argmax.
    """

    def __init__(self, action_dim: int) -> None:
        super().__init__(action_dim)
        self.base_logits: th.Tensor | None = None
        self.residual_logits: th.Tensor | None = None
        self.production_mask: th.Tensor | None = None
        self.base_distribution: MaskableCategorical
        self.production_distribution: MaskableCategorical
        self.last_baseline_actions: th.Tensor | None = None

    def proba_distribution(
        self,
        action_logits: th.Tensor,
        residual_logits: th.Tensor | None = None,
        production_mask: th.Tensor | None = None,
    ) -> "RoutePreservingCategoricalDistribution":
        base = action_logits.view(-1, self.action_dim)
        residual = (
            th.zeros_like(base)
            if residual_logits is None
            else residual_logits.view(-1, self.action_dim)
        )
        if production_mask is None:
            raise ValueError("production_mask is required")
        production = production_mask.to(device=base.device, dtype=th.bool).view(
            -1, self.action_dim
        )
        self.base_logits = base
        self.residual_logits = residual
        self.production_mask = production
        self.apply_masking(None)
        return self

    def apply_masking(self, masks: MaybeMasks) -> None:
        if self.base_logits is None or self.production_mask is None:
            return
        valid = th.ones_like(self.production_mask)
        if masks is not None:
            valid = th.as_tensor(
                masks, dtype=th.bool, device=self.base_logits.device
            ).reshape(self.base_logits.shape)
        self.base_distribution = MaskableCategorical(
            logits=self.base_logits, masks=valid
        )
        production_valid = valid & self.production_mask
        assert self.residual_logits is not None
        self.production_distribution = MaskableCategorical(
            logits=self.base_logits + self.residual_logits,
            masks=production_valid,
        )
        base_probabilities = self.base_distribution.probs
        production_mass = (
            base_probabilities * self.production_mask.to(base_probabilities.dtype)
        ).sum(dim=1, keepdim=True)
        reranked_production = production_mass * self.production_distribution.probs
        probabilities = th.where(
            self.production_mask,
            reranked_production,
            base_probabilities,
        )
        # MaskableCategorical normalizes a second time, protecting against small
        # floating-point drift while retaining gradients through the residual.
        self.distribution = MaskableCategorical(probs=probabilities, masks=valid)

    def sample(self) -> th.Tensor:
        baseline = self.base_distribution.sample()
        reranked = self.production_distribution.sample()
        baseline_is_production = self.production_mask.gather(
            1, baseline.unsqueeze(1)
        ).squeeze(1)
        self.last_baseline_actions = baseline
        return th.where(baseline_is_production, reranked, baseline)

    def mode(self) -> th.Tensor:
        baseline = th.argmax(self.base_distribution.probs, dim=1)
        reranked = th.argmax(self.production_distribution.probs, dim=1)
        baseline_is_production = self.production_mask.gather(
            1, baseline.unsqueeze(1)
        ).squeeze(1)
        self.last_baseline_actions = baseline
        return th.where(baseline_is_production, reranked, baseline)

    def actions_from_params(
        self,
        action_logits: th.Tensor,
        deterministic: bool = False,
        *,
        residual_logits: th.Tensor | None = None,
        production_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        self.proba_distribution(
            action_logits,
            residual_logits=residual_logits,
            production_mask=production_mask,
        )
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(
        self,
        action_logits: th.Tensor,
        *,
        residual_logits: th.Tensor | None = None,
        production_mask: th.Tensor | None = None,
    ) -> tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(
            action_logits,
            residual_logits=residual_logits,
            production_mask=production_mask,
        )
        return actions, self.log_prob(actions)


class RoutePreservingResidualContextPolicy(ResidualContextMaskablePolicy):
    """Freeze the shared actor and learn production-conditional reranking only."""

    def _distribution(
        self,
        obs: dict[str, th.Tensor],
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> RoutePreservingCategoricalDistribution:
        base_logits = self.base_action_logits(obs)
        residual_logits = self.context_residual(obs)
        production_mask = obs["action_features"][..., 1] > 0.5
        distribution = RoutePreservingCategoricalDistribution(
            self.action_count
        ).proba_distribution(
            base_logits,
            residual_logits=residual_logits,
            production_mask=production_mask,
        )
        distribution.apply_masking(action_masks)
        return distribution
