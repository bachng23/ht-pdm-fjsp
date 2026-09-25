# Passive technician algorithm-family screening

## Research question and hypotheses

Which cooperative MARL family is worth extending for decentralized maintenance
with shared heterogeneous technicians? The screening compares masked
independent Q-learning, masked Independent PPO, centralized-critic PPO
(MAPPO-lite), and a COMA-style counterfactual policy-gradient learner. The
primary hypothesis is that methods with centralized coordination or
counterfactual credit assignment will reduce queue waiting and allocation
failures relative to independent learners.

## Locked protocol

All methods use the corrected three-machine/two-technician FIFO stress cell,
the same resource-aware observations, duplicate-request action mask, failure
process, costs, and deterministic evaluation rule. Smoke uses train seed 11,
evaluation seeds 101--103, and 32 training episodes. The full screening uses
train seeds 11--13, evaluation seeds 101--130, and 5,000 episodes per seed.
Training stops at the fixed budget; there is no checkpoint selection after
looking at results.

The primary screening metric is mean objective cost on the common evaluation
panel. Mechanism metrics are failures, service jobs, waiting, collisions,
invalid requests, requests to busy technicians, unique joint actions, defer
fraction, and request count. Fixed skill-aware FIFO and random-feasible
dispatchers are references, not learned candidates.

## Selection rule

Before comparing objective means, a candidate must pass these gates: all
checkpoints load, `invalid_requests=0`, objectives are nonnegative, and the
sealed test panel remains closed. Among candidates that pass, the preferred
family is the one with the lowest mean objective, lower waiting, non-degenerate
joint-action diversity, and lower across-seed variation. This is a screening
decision, not a final algorithm claim; the winning family must be re-evaluated
on a larger locked panel before publication claims.

## Artifact schema and limits

Each timestamped run contains a manifest, benchmark configuration,
`episodes.csv`, `episodes.partial.csv`, `budget_summary.csv`,
`training_progress.csv`, `coordination.csv`, and serialized checkpoints. The
COMA candidate is an implementation screen for counterfactual credit
assignment, not a claim of exact reproduction of every detail in the original
COMA paper. The FIFO queue remains an environment/reference semantics; the
screen evaluates which learner can make better assignments under that shared
resource process.
