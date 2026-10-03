# Role × peer-information diagnostic — Mac smoke verification

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent.
- Origin Mode: engineering verification.
- Origin Date: 2026-10-03.
- Verification Status: VERIFIED engineering smoke; full scientific gates not evaluated.
- Protocol: `ra_qmix_role_context_v1`.
- Base commit: `a541b2dd677bbf59244e008498c88122549d2b90`; smoke manifest records dirty implementation before the new experiment commit.

## Validation

- Setup: `uv sync --frozen --extra dev`.
- Targeted tests: 13 passed in 2.45 seconds.
- Entire repository: 327 passed, 1 dependency warning, 163.68 seconds.
- Compile and `git diff --check`: passed.
- Host: macOS 27.0.1 arm64, Python 3.12.13; torch 2.14.0 CPU, numpy 2.5.3, tqdm 4.70.1. CP-SAT dependency comes from the unchanged frozen ortools lock entry.
- Executable: `ht-pdm-fjsp-passive-role-context --profile smoke --device cpu`.
- Artifacts: `artifacts/ra_qmix_role_context_smoke_20261003T130727Z/` and adjacent console log.
- Start/end: 13:07:29–13:07:32 UTC, 03/10/2026 (21:07 Taipei).
- COMPLETED: 15/15 models/checkpoints, 45/45 evaluation rows, exactly 2400 optimizer updates.
- All manifest outputs present; checkpoint reload gives bitwise-identical predictions; training/oracle/evaluation tqdm bars appear in console log.
- Zero invalid requests; exact cost reconciliation; finite logs; CF losses zero; all engineering audits pass; sealed seeds 201–300 closed.
- Artifact directory 897494 bytes excluding log.

Factorial arms share 4722 parameters and identical initialization; joint-Q 4701 (−0.445%). Full configuration parameter-count preflight, without full fitting: factorial 31170 each, joint-Q 31095 (−0.241%). Unit tests explicitly compare mixer state_dict to the original no-CF model and verify input toggles, identity symmetry breaking, no peer leakage into local policy when disabled, CP-SAT feasibility/conflicts/UNKNOWN, primary gates and failure/overwrite behavior.

In the short smoke graph, nominal lookup policies are feasible for all four information sets; shared-preference/pressure have direct same-input infeasibility certificates for base and peer-only, while identity and identity+peer are feasible with verified assignments. These concern horizon3 lookup-policy feasibility, not horizon6 neural fitting or performance. No full outcome, thresholds or budgets were selected from these results.

## Full-run scope and handoff

Three cells × five arms × ten fresh training seeds = 150 fits and checkpoints; 10000 updates each, 1.5 million total updates, 7500 rollout evaluations. Expect 67000 per-state model metric rows for the unchanged horizon6 finite graphs, plus oracle tables, 12 information-feasibility reports, training progress and paired gates. Checkpoint files around 20–25 MB; total artifacts estimated 50–100 MB including CSV/certificates/logs. Smoke duration is not a reliable lab ETA.

Four factorial arms use local masked greedy decisions; mixer stays unchanged. Identity denotes the fixed machine index; peer telemetry is explicit observation sharing, with no hidden simulator state. This run tests the information/execution contract with CF off in supervised oracle fitting, not learned role discovery or proof that online adaptation is solved.

No full run, SSH, or rsync was executed. After push, user runs the Ubuntu CPU compound command and rsync command under `docs/experiment_workflow.md`.
