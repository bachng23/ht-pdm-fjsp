# Frozen-base residual context PPO

## Material Passport
- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED
- Version Label: residual_context_ppo_v1

## Locked design
- Hypothesis: a bounded context residual can improve dispatching without the seed instability caused by relearning the entire scorer.
- The actor and critic start from the matching completed `shared_scorer_entropy` seed.
- The base actor is frozen. A production-only residual head is zero-initialized and bounded to ±1 logit; non-production residuals are exactly zero.
- Controls: completed shared entropy baseline and completed unconstrained production-context negative control.
- Five training seeds, 500,000 steps, `ent_coef=0.01`, fresh validation seeds 46000--46199.
- Primary endpoint: per-training-seed mean objective, residual minus shared.
- Fixed budget; no early stopping, checkpoint selection, retry, or test evaluation. Seeds 50000--50099 remain unopened.

## Expected artifacts
- `residual_context_manifest.json`: completed, five cells, test unopened.
- `residual_context_episodes.csv`: 3 × 5 × 200 = 3,000 unique rows.
- `residual_context_summary.json`: paired seed-level comparisons.
- Five loadable residual policy models plus config snapshot and partial files.
