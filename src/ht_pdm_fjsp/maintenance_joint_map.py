"""Exhaustive joint-MAP diagnostic for the frozen N4 serial reservation actors.

Uses the original actor, order and masks. Never consults a critic or future shocks.
Maximizing probability does not guarantee better expected physical cost.
"""
import torch

from ht_pdm_fjsp.maintenance_solver_policy import (
    CATALOGUE, LOCAL, SERIAL, TIE, feature_batch,
)
from ht_pdm_fjsp.maintenance_dispatch import validate_matching

RAW_ACTIONS = torch.tensor([
    [next((j + 1 for m, j in pairs if m == i), 0) for i in range(4)]
    for pairs in CATALOGUE
], dtype=torch.long)
MASS_TOLERANCE = 1e-5


@torch.no_grad()
def matching_log_probabilities(model, c, state, remaining):
    """Single-state batching with one shared encoding; all feasible joint actions."""
    if model.algorithm not in SERIAL:
        raise ValueError("joint-MAP requires a serial actor")
    batch = feature_batch([c], [state], [remaining])
    indices = torch.where(batch["matching_mask"][0])[0]
    actions = RAW_ACTIONS[indices]
    n = len(indices)
    expanded = {k: v.expand(n, *v.shape[1:]) for k, v in batch.items()}
    encoded = tuple(v.expand(n, *v.shape[1:]) for v in model._encoded(
        batch, local=model.algorithm == LOCAL or model.algorithm == "independent_local_ppo"))
    reserved = torch.zeros((n, 4), dtype=torch.bool)
    rows = torch.arange(n)
    score = torch.zeros(n)
    for position in range(4):
        machine = expanded["order"][:, position]
        distribution, mask = model.serial_distribution(expanded, machine, reserved, encoded)
        chosen = actions[rows, machine]
        if not mask.gather(1, chosen[:, None]).all():
            raise RuntimeError("feasible matching has invalid reservation prefix")
        score += distribution.log_prob(chosen)
        uses = chosen > 0
        reserved[rows[uses], chosen[uses] - 1] = True
    if not torch.isfinite(score).all():
        raise RuntimeError("nonfinite joint log probability")
    mass = float(torch.logsumexp(score.double(), 0).exp())
    if abs(mass - 1) > MASS_TOLERANCE:
        raise RuntimeError(f"joint probability mass {mass} is not one")
    return indices, score, mass


def best_index(scores):
    """Stable existing catalogue order; numerical tie tolerance matches legacy."""
    precise = scores.double()
    return int(torch.where(precise >= precise.max() - TIE)[0][0])


class Decoder:
    def __init__(self, model, mode):
        if mode not in ("greedy", "joint_map"):
            raise ValueError("unknown decoder")
        if mode == "joint_map" and model.algorithm not in SERIAL:
            raise ValueError("MAP requires serial actor")
        self.model, self.mode = model, mode

    @torch.no_grad()
    def select(self, c, state, remaining):
        if self.mode == "greedy":
            return self.model.select(c, state, remaining)
        indices, scores, mass = matching_log_probabilities(self.model, c, state, remaining)
        chosen = best_index(scores)
        pairs = CATALOGUE[int(indices[chosen])]
        validate_matching(c, state, pairs)
        return pairs, dict(
            decoder="joint_map", candidate_count=len(indices), probability_mass=mass,
            log_probability=float(scores[chosen]),
            maximum_log_probability=float(scores.max()),
            numerical_selection_regret=float(scores.max() - scores[chosen]),
            critic_evaluated=False, collision_rejections=0, proposal_attempts=len(pairs),
        )
