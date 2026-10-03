# Oracle representation diagnostic — Mac smoke verification

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: run/engineering verification.
- Origin Date: 2026-10-03.
- Verification Status: VERIFIED engineering smoke; full scientific hypotheses not evaluated.
- Protocol: `ra_qmix_oracle_representation_v1`.
- Base commit: `9abc1f64f178938301fa9758e31b94445ac5b951`; smoke manifest records dirty implementation before the experiment commit.

## Verification

- Dependency setup: `uv sync --frozen --extra dev`.
- Targeted tests: 13 passed in 10.29 seconds.
- Full repository suite: 314 passed, 3 dependency warnings, 190.66 seconds.
- Compile check and `git diff --check`: passed.
- Host: macOS 27.0.1 arm64, Python 3.12.13; torch 2.14.0 CPU, numpy 2.5.3, tqdm 4.70.1.
- Executable: `ht-pdm-fjsp-passive-oracle-representation --profile smoke --device cpu`.
- Artifact: `artifacts/ra_qmix_oracle_representation_smoke_20261003T035120Z/`; adjacent console log has oracle, training and multi-seed evaluation tqdm bars.
- Run: 03:51:21–03:51:25 UTC, 03/10/2026 (11:51 Taipei).
- Status COMPLETED; 12/12 models, 12 reload-verified checkpoints, 36/36 unique evaluation rows, 1920 optimizer updates in total.
- Every declared output exists; zero invalid requests; exact cost reconciliation; sealed 201–300 closed.
- Artifacts total 468396 bytes excluding console log.

Oracle graph state/action-row counts: nominal 33/109, shared preference 33/105, pressure 50/138. Bellman residual ≤8.9e-16; no joint-observation target alias detected. Rank-reversal witnesses exist in 6/5/6 states respectively. They certify the full-Q-table restriction at those states, not optimal-policy impossibility or the cause of previous adaptation regressions. Scientific fitting/control thresholds are disabled for smoke and have not been tuned from smoke performance.

Smoke factorized models: 4450 parameters each; joint-Q 4419 (−0.697%). Full configuration parameter-count preflight (no full training): factorized 30082, joint-Q 30039 (−0.143%). Full scope: 3 cells × 4 arms × 10 initialization/training seeds = 120 fits; 1.2 million optimizer updates; 6000 seeded evaluation episodes; 120 checkpoints around 15–20 MB plus oracle tables, state metrics, progress, summaries and logs. CSV size depends on reachable-state counts (100000-state cap per cell). This smoke duration is not a reliable full-run ETA.

No full run, SSH, or rsync was executed. The lab run and retrieval remain user-owned under `docs/experiment_workflow.md`.
