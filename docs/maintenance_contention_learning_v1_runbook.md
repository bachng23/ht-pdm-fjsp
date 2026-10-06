# Maintenance contention learning v1 — runbook

Use the frozen plan `maintenance_contention_learning_v1_plan.md` for the
hypothesis, budget and seed panels. This run trains one centralized subset-PPO
architecture with10 independent seeds. It evaluates RL learnability, with K2
as primary and K4 nominal guard; it makes no MARL comparison.

Local engineering CLI:

```bash
uv run --frozen --no-sync python -m ht_pdm_fjsp.maintenance_contention_learning \
  --profile smoke --device cpu
```

For Ubuntu lab use the final handoff compound command pinned to the verified
commit, detached worktree, its own `.venv`, and a fresh timestamped artifact
path under `$HOME/ht-pdm-fjsp/artifacts`. CPU only. Keep the terminal open.
The tee log is a sibling of the run directory. Full requires clean Git. The
runner refuses nonempty outputs and has no resume mode. FAILED preserves
partial journals and saved checkpoints; a new run gets a new timestamp.

Full:4,915,200 physical training steps,409,600 training episodes,102,400 optimizer
steps;22,400 development episodes;45,000 test episodes and810,000 test intervals.
Model checkpoints include initialization plus4 fixed training budgets per seed,
and the chosen `model.pt`. Initialization is not eligible for selection.
Checkpoint selection and per-capacity reference selection use only development.
Selections for all10 models are saved before test evaluations begin.

`checkpoint_selection.json` records the selected step and guards. If no trained
checkpoint satisfies development guards, the final model is used and marked
DEVELOPMENT_INFEASIBLE. This model remains in evaluation and uncertainty; it is
not dropped. Engineering smoke does not require economic success after256
training steps. Every smoke contrast stays ENGINEERING_ONLY.

`training_episodes.csv` records episode index, configuration/hazard seeds, config
hash, capacity and cost components. Its physical_steps field is cumulative at
the parallel group's completion, so multiple episodes share that counter.
`training_progress.csv` gives optimizer/physical counters, cost and PPO losses
at every rollout. Rewards=-priced cost/20; undiscounted complete-episode returns
keep parallel episodes separate. Teacher budget is zero. Counter-derived seeds
are checked against historical/reserved values at generation time.

The generator has finitely many physical parameter profiles. Independent
configuration seeds may yield the same profile in training and evaluation.
The guarantee is no reserved development/test seed is used for training and no
test-result feedback; it is not unseen-profile generalization. K1 and H24 are
explicit secondary sensitivities, including capacity/horizon distribution shift.

`episodes.csv` and matching `episodes.partial.csv` contain selected learned
models and all4 references plus planner. Each deterministic controller is
run once per test panel, rather than fabricated as10 independent replicas.
`decisions.csv` contains state/action/next state, physical metrics and controller
diagnostics; `machine_metrics.csv` reconciles waits/cost intervals per machine.
Wait p95 is pooled pending-wait state over machine intervals, as in the parent
headroom diagnostic. Controller/planner runtime is reported separately from cost.

`paired_seed_metrics.csv` reports seed-level differences and priced/base gap
capture relative to joint rollout. Gap capture is undefined when planner does
not beat reference; it may otherwise be negative or exceed1. This planner is
achievable model-based control, not an exact N4 optimum. `summary.json` reports
all primary gates and labels K1/H24 exploratory. The95% CI has training-seed
units conditional on the finite fixed test panel, not independent episode units.

Primary PASS requires priced gain≥5%, base gain≥3%,≥8/10 seed wins, upper95%
paired CI<0, nominal base ratio≤1.1 and primary waiting-violation rate no more
than2 percentage points above the selected reference. Learning curves and gap
capture help interpret a FAIL without altering these thresholds after the run.

Local macOS engineering validation:

- CLI smoke `maintenance_contention_learning_v1_smoke_cpu_20261006T020724Z`
  COMPLETED:1 model,256 steps,64 train episodes,16 optimizer steps,18 development
  episodes,108 test episodes,540 test intervals,432 machine rows.
- Tests verify full physical action coverage modulo identical-worker symmetry,
  masked sampling/log-probability reevaluation, STOP-only entropy/gradient
  stability, permutation equivariance, deterministic near-ties at1e-6,
  Monte Carlo episode boundaries, reserved
  seed guards, checkpoint reload/selection, primary gates and complete panels.
- Artifact replay checks selected-model decisions and all scalar physical
  transitions/costs across the entire smoke panel, hashes and overwrite refusal.
- Full-shape engineering test uses only smoke seed143000:1536-step/H12 rollout,
  width64,32 vector envs,128 episodes and32 optimizer updates; finite gradients,
  changed weights, exact budgets and reserved-seed checks pass. It does not tune
  full hyperparameters or open sealed test episodes.
