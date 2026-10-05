# Context × waiting v1 — local verification

Branch: `codex/maintenance-context-waiting-v1`, based on priority/allocation v2
`cb1a655063843befeff0da0667eb6d190f2f5f07`. Main checkout's concurrent/unrelated
changes were preserved. No remote full run or transfer was launched.

## Protocol and implementation

See `maintenance_context_waiting_v1_plan.md` for the locked factorial design,
synthetic waiting assumption, separate H_context/H_wait gates, seed panels and
limits. The original environment remains unchanged. The new waiting environment
adds observable limit/price and records original economic cost separately.

The existing PPO trainer and rollout helper accept explicit environment/config
factories and an artifact algorithm label. Existing defaults retain v2 behavior.
The actor exposes a deterministic-choice hook whose original default is argmax;
the new context actor adds a matched-size technician context input and a 1e-5
near-tie deployment rule. Sampling and PPO log probabilities use the original
categorical distribution. New modules have a separate environment/checkpoint
contract and the old checkpoint architecture remains loadable.

A read-only audit tool is available at
`scripts/audit_maintenance_context_waiting.py`. It replays state transitions and
independently calculates tick-level maintenance/unavailability/failure/overdue
cost, reconciles machine waits and starts, verifies source hashes, checkpoints,
training populations and exact-regret arithmetic. Replay shares the environment
implementation; it is not independent simulator reproduction.

## Verification

- Ruff check/format and `git diff --check`: passed on changed/new Python files.
- Entire local suite: **440 passed**, 172.17 seconds. One existing SB3 rich/tqdm
  warning; no new warnings after fixing a detached test scalar.
- 26 added test cases include overdue boundary/start-service semantics, new
  failure timing, zero-price state/shock/cost equivalence, invalid costs,
  allocation-head peer-information symmetry and technician permutation,
  near-tie handling, honest sampled logprob replay with gradients, both
  confirmatory gates and their uncertainty/economic/nominal/p95 failure cases,
  output/dirty-Git guards, base-objective exact DP and positive-penalty PPO MC
  reconciliation. Tests use synthetic data, never full test outcomes.
- Additional read-only scan of 75 local historical manifests found no collision
  with any new full/smoke seed panel.

Actual macOS CLI smoke directory:
`artifacts/maintenance_context_waiting_v1_smoke_cpu_20261005T063708Z`.
Command used the worktree's existing frozen environment via
`uv run --frozen --no-sync --cache-dir /private/tmp/marl-context-waiting-uv-cache
python -m ht_pdm_fjsp.maintenance_context_waiting --profile smoke --device cpu
--output-dir artifacts/<run-id>` with pipefail and a preserved tqdm console log.
An initial uv invocation could not write the sandbox-protected shared cache;
its failure log remains at the preceding timestamp, before runner execution.
No dependency/lockfile change was needed.

Smoke status COMPLETED; **23/23 engineering audits passed**:
4 models, 192 physical training steps, 32 optimizer updates, 48 training
episodes, 8 rollout logs, 96 learned test episodes, 4 development episodes,
25 heuristic episodes and 5 exact rows. All arms have equal allocated parameters,
identical initial weights and identical physical training config/initial/hazard
panels. No teacher/test-training transitions or sealed-panel access.

Source bundle SHA256:
`e140cb275faa0b99a531bcdb5d7262c8e276f2eef59f2ebb3958ce893dcde264`.
The separate audit passed on 120 test episodes and 525 physical ticks, all
48 training records, source snapshots, four loaded checkpoints and exact rows.
Local audit report:
`reports/maintenance_context_waiting_v1_preparation_20261005T063708Z/smoke_audit.json`
in the user's main checkout.

The H=4 smoke training episodes do not reach a waiting age >4 from healthy
initial states, so all 48 training rows have zero waiting penalty. The longer
smoke evaluation does reach overdue waiting (12 learned test records), and an
additional forced-overdue training test explicitly verifies positive penalties
are included in PPO returns and episode reconciliation. The four tiny-trained
policies produce identical aggregate smoke cost; this is engineering evidence,
not a failed/passed scientific hypothesis. No training budget or hyperparameter
was adjusted in response.

## Full handoff

Run `uv run --frozen --no-sync python -m
ht_pdm_fjsp.maintenance_context_waiting --profile full --device cpu
--output-dir <new-timestamp-directory>` after `uv sync --frozen` in a clean,
separate detached worktree pinned to the pushed commit. CPU is intentional:
GPU support is not implemented. The handoff keeps artifacts under
`$HOME/ht-pdm-fjsp/artifacts/`, regardless of the run checkout, and uses a
separate .venv so concurrent experiments can continue.

Expected full artifacts: 40 checkpoints, 4.8 million training transitions,
400000 training episodes, 10000 rollout records, 160000 optimizer steps,
16000 learned test episodes, 800 development episodes, 420 fixed-rule
reference episodes, 41 exact rows, physical traces and per-machine waiting
metrics. Terminal must remain open; no resume contract. The previous pod SSH
endpoint/port is reused for the user-run rsync, never contacted by the assistant.
