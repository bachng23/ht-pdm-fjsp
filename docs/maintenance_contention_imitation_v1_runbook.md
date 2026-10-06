# Maintenance contention imitation v1 — runbook

## Material Passport

Schema: ARS 9. Material: experiment execution handoff. Origin: authorized code
implementation. Status: ENGINEERING_VERIFIED; full experiment NOT_RUN.

Frozen plan: maintenance_contention_imitation_v1_plan.md. This diagnostic compares
fresh direct PPO, planner imitation, and imitation-initialized PPO. Same encoder,
masked subset action and new paired test panel. Hybrid has additional teacher
information/compute; do not describe this as a matched-total-budget comparison
or a MARL necessity test.

## Local engineering entry point

`python -m ht_pdm_fjsp.maintenance_contention_imitation --profile smoke --device cpu`
creates a new UTC artifact directory. Explicit --output-dir must be new/empty.
Full profile requires clean committed Git; CPU only, one torch thread. No resume.
A failed run retains manifest FAILED, journals and available checkpoints.

## Full handoff and scope

Use the final compound command pinned to the verified SHA in a fresh detached
lab worktree with its own .venv. The output and tee log live under the original
$HOME/ht-pdm-fjsp/artifacts, so the canonical rsync retrieves them. Keep the lab
terminal open; no tmux. Do not switch/reset the primary lab checkout.

30 selected models (10 per arm). PPO total9830400 physical steps/819200 episodes;
BC+PPO256000 optimizer steps. Teacher data138240 physical intervals,122880 training
labels and15360 validation labels. Three arms62400 development and105000 test
episodes,1890000 test intervals/420000 machine rows. Detailed CSV traces and
teacher datasets can require several GB; exact size depends on output fields.
No external checkpoint or downloaded dataset is needed.

## What to inspect after transfer

1. manifest.json status COMPLETED, actual_counts==expected_counts and file hashes.
2. seed_audit.json, training_coverage.json and teacher_coverage.json: exact budgets,
   no held-out seeds, no validation gradients, BC critic head untouched.
3. checkpoint_selection.json: all30 selected before test; hybrid parent hash equals
   the selected BC model hash. BC counter_kind is bc_optimizer_steps; PPO is
   ppo_physical_steps. Initial models are not selectable. Infeasible seeds stay.
4. teacher_rows.csv / teacher_data.pt: separate training and validation current
   states, independent planner forecasts, accepted tied subsets and actual
   executed trajectories. teacher_noise.csv holds independent128-forecast probes
   on predetermined validation rows, never training targets.
5. bc_progress.csv and bc_diagnostics.csv: agreement and root-score regret on
   training/validation, separately conditioned on >1 feasible subset. High overall
   agreement can be inflated by forced STOP-only states.
6. training_progress.csv / training_episodes.csv: arm-tagged direct/hybrid PPO.
   Same RNG role seeds do not imply identical state trajectories after policies
   diverge. Training timings include checkpoint development evaluation; teacher
   generation and test inference timings are separately available.
7. episodes.csv/partial, decisions.csv and machine_metrics.csv: cost, failures,
   waiting violations, backlog/utilization and feasible physical replay.
8. summary.json.primary: the single confirmatory hybrid-vs-direct >=2% contrast,
   paired10 training-replica CI, >=8 wins and nominal/waiting guards.
   summary.json.arms gives the separate absolute >=5% priced/>=3% base economic
   gates vs the chosen reference, plus K1/H24 exploratory sensitivities.
   Report both verdicts; a relative PASS does not imply absolute success.

All full test cohorts/shocks are fresh compared with learning v1. Physical
profiles may repeat under the finite generator. Uncertainty is conditional on
this fixed test panel; cohort consistency is separately exploratory. Good BC
with poor PPO supports a supervision/optimization explanation, without uniquely
identifying credit assignment; poor BC does not uniquely prove encoder failure.

## Local verification

- Whole existing+new suite:514 passed, one pre-existing tqdm rich warning.
- Final focused suite:9 passed, including full-width64/H12 teacher and BC tensor
  shapes using engineering seed149000 only, with32/128 forecast budgets.
- Final CLI smoke: maintenance_contention_imitation_v1_smoke_cpu_20261006T063325Z,
  COMPLETED,3 models,512 PPO steps,128 PPO episodes,40 optimizer steps,
  80 teacher steps,64 train/16 validation labels,2 independent noise probes,
  34 development episodes,144 test episodes,720 intervals and576 machine rows.
- Integration replay covers every smoke selected-policy action and physical
  transition/cost, every teacher label and executed transition, all artifact
  hashes, matching initial weights and exact BC->hybrid initial parameters.
- Tests also check tied/masked supervision gradients, no future-event access,
  validation labels not affecting training weights, primary paired gates,
  checkpoint provenance, failure preservation and output overwrite refusal.
- Smoke results are ENGINEERING_ONLY; no full test trajectory was evaluated
  locally. No remote run or transfer has been started.
