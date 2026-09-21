"""Route-preserving residual ranking restricted to real production conflicts."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import torch as th

from sb3_contrib.common.maskable.distributions import (
    MaskableCategorical,
    MaskableCategoricalDistribution,
    MaybeMasks,
)

from ht_pdm_fjsp.route_preserving_residual_policy import (
    RoutePreservingResidualContextPolicy,
)
from ht_pdm_fjsp.gym_env import ActionDescriptor


def build_production_conflict_matrix(
    actions: Sequence[ActionDescriptor],
) -> list[list[bool]]:
    """Return the static same-machine-or-operation production conflict graph."""

    matrix: list[list[bool]] = []
    for left in actions:
        row: list[bool] = []
        for right in actions:
            both_production = left.kind == right.kind == "production"
            same_machine = left.machine_id == right.machine_id
            same_operation = (
                left.job_id == right.job_id
                and left.operation_index == right.operation_index
            )
            row.append(bool(both_production and (same_machine or same_operation)))
        matrix.append(row)
    return matrix


class ConflictAwareRoutePreservingDistribution(MaskableCategoricalDistribution):
    """Couple a shared route draw with conflict-local production reranking.

    For each possible shared production action, the residual conditional is
    normalized only over feasible actions that share its machine or its
    job-operation.  The categorical distribution exposed to PPO is the exact
    marginal of this coupled two-stage sampler.
    """

    def __init__(self, action_dim: int) -> None:
        super().__init__(action_dim)
        self.base_logits: th.Tensor | None = None
        self.residual_logits: th.Tensor | None = None
        self.production_mask: th.Tensor | None = None
        self.conflict_matrix: th.Tensor | None = None
        self.valid_mask: th.Tensor | None = None
        self.base_distribution: MaskableCategorical
        self.conditional_probabilities: th.Tensor | None = None
        self.last_baseline_actions: th.Tensor | None = None

    def proba_distribution(
        self,
        action_logits: th.Tensor,
        *,
        residual_logits: th.Tensor,
        production_mask: th.Tensor,
        conflict_matrix: th.Tensor,
    ) -> "ConflictAwareRoutePreservingDistribution":
        base = action_logits.view(-1, self.action_dim)
        residual = residual_logits.view(-1, self.action_dim)
        production = production_mask.to(device=base.device, dtype=th.bool).view(
            -1, self.action_dim
        )
        conflicts = conflict_matrix.to(device=base.device, dtype=th.bool)
        if conflicts.shape != (self.action_dim, self.action_dim):
            raise ValueError(
                "conflict_matrix must have shape (action_dim, action_dim)"
            )
        if not th.equal(conflicts, conflicts.T):
            raise ValueError("conflict_matrix must be symmetric")
        production_indices = th.nonzero(production[0], as_tuple=False).flatten()
        if production_indices.numel() and not th.all(
            conflicts[production_indices, production_indices]
        ):
            raise ValueError("Every production action must conflict with itself")
        self.base_logits = base
        self.residual_logits = residual
        self.production_mask = production
        self.conflict_matrix = conflicts
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
        self.valid_mask = valid
        self.base_distribution = MaskableCategorical(
            logits=self.base_logits, masks=valid
        )
        assert self.residual_logits is not None
        assert self.conflict_matrix is not None
        combined = self.base_logits + self.residual_logits
        candidate_masks = (
            self.conflict_matrix.unsqueeze(0)
            & valid.unsqueeze(1)
            & self.production_mask.unsqueeze(1)
        )
        candidate_logits = combined.unsqueeze(1).expand(
            -1, self.action_dim, -1
        )
        masked_logits = candidate_logits.masked_fill(~candidate_masks, -1e8)
        conditional = th.softmax(masked_logits, dim=-1)
        conditional = conditional * candidate_masks.to(conditional.dtype)
        conditional = conditional / conditional.sum(dim=-1, keepdim=True).clamp_min(
            1e-12
        )
        self.conditional_probabilities = conditional

        base_probabilities = self.base_distribution.probs
        production_weights = (
            base_probabilities * self.production_mask.to(base_probabilities.dtype)
        )
        reranked_production = th.einsum(
            "bi,bij->bj", production_weights, conditional
        )
        probabilities = th.where(
            self.production_mask,
            reranked_production,
            base_probabilities,
        )
        self.distribution = MaskableCategorical(probs=probabilities, masks=valid)

    def _conditional_actions(
        self, baseline: th.Tensor, *, deterministic: bool
    ) -> th.Tensor:
        if self.conditional_probabilities is None:
            raise RuntimeError("Conditional probabilities are unavailable")
        batch = th.arange(baseline.shape[0], device=baseline.device)
        probabilities = self.conditional_probabilities[batch, baseline]
        fallback = th.nn.functional.one_hot(
            baseline, num_classes=self.action_dim
        ).to(probabilities.dtype)
        probabilities = th.where(
            probabilities.sum(dim=1, keepdim=True) > 0,
            probabilities,
            fallback,
        )
        if deterministic:
            return th.argmax(probabilities, dim=1)
        return th.multinomial(probabilities, num_samples=1).squeeze(1)

    def sample(self) -> th.Tensor:
        baseline = self.base_distribution.sample()
        reranked = self._conditional_actions(baseline, deterministic=False)
        assert self.production_mask is not None
        baseline_is_production = self.production_mask.gather(
            1, baseline.unsqueeze(1)
        ).squeeze(1)
        self.last_baseline_actions = baseline
        return th.where(baseline_is_production, reranked, baseline)

    def mode(self) -> th.Tensor:
        baseline = th.argmax(self.base_distribution.probs, dim=1)
        reranked = self._conditional_actions(baseline, deterministic=True)
        assert self.production_mask is not None
        baseline_is_production = self.production_mask.gather(
            1, baseline.unsqueeze(1)
        ).squeeze(1)
        self.last_baseline_actions = baseline
        return th.where(baseline_is_production, reranked, baseline)


class ConflictAwareRoutePreservingResidualPolicy(
    RoutePreservingResidualContextPolicy
):
    """Freeze the shared actor and rerank only conflicting production actions."""

    def __init__(
        self,
        *args: Any,
        production_conflict_matrix: list[list[bool]],
        **kwargs: Any,
    ) -> None:
        matrix = np.asarray(production_conflict_matrix, dtype=bool)
        action_count = int(args[1].n)
        if matrix.shape != (action_count, action_count):
            raise ValueError(
                "production_conflict_matrix must match the action space"
            )
        self._production_conflict_matrix_array = matrix
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "production_conflict_matrix",
            th.as_tensor(matrix, dtype=th.bool, device=self.device),
        )

    def _distribution(
        self,
        obs: dict[str, th.Tensor],
        action_masks: np.ndarray | th.Tensor | None = None,
    ) -> ConflictAwareRoutePreservingDistribution:
        base_logits = self.base_action_logits(obs)
        residual_logits = self.context_residual(obs)
        production_mask = obs["action_features"][..., 1] > 0.5
        distribution = ConflictAwareRoutePreservingDistribution(
            self.action_count
        ).proba_distribution(
            base_logits,
            residual_logits=residual_logits,
            production_mask=production_mask,
            conflict_matrix=self.production_conflict_matrix,
        )
        distribution.apply_masking(action_masks)
        return distribution
