# Route-preserving residual production ranking

## Material Passport

- Origin: completed shared-action, production-context, context diagnostic, and
  frozen-base residual experiments.
- Verification status: UNVERIFIED until the full lab run completes.
- Version label: `route_preserving_residual_v1`.
- Scope: one synthetic HT-PdM-FJSP benchmark instance; architecture development.

## Question and hypothesis

The first residual-context policy changed only production logits, but those
logit changes also changed the probability and deterministic selection of the
high-level production-versus-maintenance-versus-advance route. Its mean
validation objective was only 0.175 lower than the shared baseline and it won
only two of five training-seed comparisons.

This experiment asks whether context is more useful when it is restricted to
ranking feasible production alternatives after the completed shared policy has
already chosen the route. The hypothesis is that this restriction preserves the
shared baseline's maintenance behavior while allowing job-machine rerouting to
improve objective without increasing failures.

## Locked policy contract

The actor and critic start from the matching completed
`shared_scorer_entropy` model. The shared actor remains frozen. At every action:

1. sample or deterministically select an action from the masked shared policy;
2. if it is advance, preventive, or corrective, execute that exact action;
3. if it is production, use a bounded, zero-initialized context residual to
   rerank only the currently feasible production alternatives.

The implemented categorical distribution preserves every non-production
action probability and the total production probability mass of the shared
policy. Its coupled sampler preserves the sampled route on every decision.
The critic and residual head may train; the shared actor may not.

## Locked experiment

- Treatment: `route_preserving_residual_entropy`, `ent_coef=0.01`, residual
  bound ±1 logit and 64-unit context encoder.
- Controls: completed `shared_scorer_entropy`, `production_context_entropy`,
  and first-generation `residual_context_entropy` models.
- Training seeds: 10000, 11000, 12000, 13000, 14000.
- Budget: 500,000 requested PPO steps per treatment model with the source PPO
  batch geometry; no early stopping, checkpoint selection, retry, or adaptive
  tuning.
- Fresh validation seeds: 47000–47199.
- Reserved future test seeds: 50000–50099 remain unopened.
- Independent replication unit: PPO training seed, not episode or decision.

Primary endpoint: mean validation objective by training seed, treatment minus
shared baseline. Secondary endpoints: failure, corrective/preventive
maintenance, total cost, tardiness, makespan, CV across training seeds, and the
paired comparison with first-generation residual context.

## Mandatory route audit and stopping rule

On every treatment validation decision, score the identical state with the
matching frozen shared policy. The run is invalid unless all three counts are
zero:

- high-level decision-kind mismatches;
- non-production action mismatches;
- invalid treatment actions.

Report the number and rate of production reroutes descriptively. Complete the
fixed budget for all five seeds. Do not open the future-test panel regardless of
the validation result.

## Expected artifacts

- `route_preserving_residual_manifest.json` with provenance and completion state;
- `benchmark_config.json` and its SHA-256;
- `route_preserving_residual_episodes.csv` with 4 × 5 × 200 unique rows;
- `route_preserving_route_audit.csv` with one row per training seed;
- `route_preserving_residual_summary.json` with seed-level paired effects;
- five loadable models, checkpoints, monitor files, and training logs;
- partial episode and route-audit tables written after every treatment cell.

## Lab workflow

Run the full experiment from the existing Ubuntu checkout with one command. The
three completed source runs are provenance-checked before training begins.

```bash
cd "$HOME/ht-pdm-fjsp" && git fetch origin && (git switch codex/route-preserving-residual-context 2>/dev/null || git switch --track origin/codex/route-preserving-residual-context) && git pull --ff-only && uv sync --frozen --extra dev && test -f artifacts/shared_action_ppo_20260920T123821Z/architecture_manifest.json && test -f artifacts/production_context_ppo_20260920T181054Z/production_context_manifest.json && test -f artifacts/residual_context_ppo_20260921T025313Z/residual_context_manifest.json && RUN_ID="route_preserving_residual_$(date -u +%Y%m%dT%H%M%SZ)" && test ! -e "artifacts/$RUN_ID" && mkdir -p artifacts/lab_logs && uv run ht-pdm-fjsp-route-preserving-residual --shared-source-run artifacts/shared_action_ppo_20260920T123821Z --production-source-run artifacts/production_context_ppo_20260920T181054Z --residual-source-run artifacts/residual_context_ppo_20260921T025313Z --profile full --device cpu --output-dir "artifacts/$RUN_ID" 2>&1 | tee "artifacts/lab_logs/$RUN_ID.log"
```

Pull the complete artifact tree to the Mac with the canonical resumable command:

```bash
mkdir -p "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results" && rsync -avhP --partial bachng@100.111.83.52:~/ht-pdm-fjsp/artifacts/ "/Users/bachng/Coding/Reinforcement Learning/marl/lab_results/"
```

## Interpretation boundary

This is a validation-stage architecture experiment on the existing synthetic
benchmark. A favorable result supports freezing this architecture before one
evaluation on the reserved test panel; it is not industrial evidence and does
not justify selecting a new threshold or budget from validation outcomes.
