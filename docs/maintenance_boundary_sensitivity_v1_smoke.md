# Maintenance boundary sensitivity v1 — Mac engineering gate

Material Passport: academic-research-suite / experiment-agent / validation;
2026-10-10; VERIFIED engineering evidence. Full scientific outcomes remain
unopened. Research settings are locked in the companion plan.

## Verification

- Frozen environment: `uv sync --frozen --extra dev`.
- Runtime: macOS 27.0.1 arm64, CPython 3.12.13, NumPy 2.5.3; CPU with
  OMP/OPENBLAS/MKL thread environment set to 1.
- Focused old/new experiment tests: **23 passed in 3.10s**.
- Final repository suite: **596 passed, 1 warning in 189.40s**.
  The warning is the inherited SB3 rich/tqdm experimental warning.
- CLI smoke: **COMPLETED**, audit **PASS**; development and evaluation
  progress bars reached 100%.

Final smoke directory:
`artifacts/maintenance_boundary_sensitivity_v1_smoke_20261010T101238Z/`.
An identical copy is preserved in the primary Mac repository's artifacts
directory. The console log is beside the directory.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
uv run --frozen python -m ht_pdm_fjsp.maintenance_boundary_sensitivity \
  --profile smoke --device cpu \
  --output-dir artifacts/<new-timestamped-directory>
```

H={4,8}, L=2, S=8; disjoint engineering cohorts and shocks. Counts:
168 development episodes / 1,344 development ticks;
60 paired evaluation episodes / 360 base + 120 tail decisions.
Training, checkpoint persistence and model reload are NOT_APPLICABLE:
this diagnostic has no trained model.

## Audit evidence

All 25 listed artifact checksums pass, including source/configuration/plan
snapshots; 1,237,628 bytes excluding the manifest and console log. Snapshot
source bytes match the verified working tree. Smoke ran before commit, so its
manifest records base commit f25876eab7d5ba138a1cfe05d5dc8eb7cb96f736 and
dirty=true; source snapshots identify the actual tested code.

480 decisions and latency rows; 60 episodes and coordination rows;
240 time-profile rows, with episode counts reconciling to 480 physical ticks;
229 request records across cutoff/extended scopes and 75 carryover records.
All actions are feasible, invalid-action count is zero, and every transition
passes scalar/batch state and cost reconciliation.

Integration tests independently replay all smoke physical decisions and
reconstruct both request histories and every carryover request, verify
continuation under the frozen deadline rule, and reconcile base + tail costs.
Unit tests verify that zero-terminal scores equal the inherited planner,
tail addition preserves all base forecast scores, common candidate scores
agree across allocator restrictions, and deterministic tail scores match
scalar policy rollouts while retaining an ongoing job across the cutoff.
They also verify shared physical prefixes, three corrected primary contrasts,
profile bootstrap units, guard failure behavior, partial journals on interrupt,
nonempty-directory rejection and full budget counts.

Largest lookahead engineering check: H48 + L12, 128 scenarios, on a smoke
physical profile; finite scores and a feasible root action. This does not run
the new full-test population. Full profile metadata was checked for exclusion
of all previous v1 profiles and relabeling-invariant dev/test disjointness.
`full_test_outcomes_opened=false`; all smoke screens are ENGINEERING_ONLY.

## Compatibility fix and handoff

The first aggregate test run exposed an inherited audit reading prior seed
exclusions as newly reused seeds. Prior physical exclusions are now stored in
`configs/maintenance_boundary_sensitivity_v1_profile_exclusions.json`, and the
separate seed registry contains only this experiment's raw panels. The final
smoke and entire suite were rerun after this metadata fix. Hypotheses,
physical profiles, budgets and controller behavior did not change.

Full budget: 4,032 development episodes and 23,040 test episodes; 921,600
test physical decisions. Expect roughly 1–2 GB of detailed artifacts,
depending on request counts and CSV sizes. This is a size estimate, not a
measured lab runtime or full-run result. CPU/thread=1, timestamped output,
no automatic retry/resume. Commit and push after this gate, then hand off the
canonical lab and Mac rsync commands without executing either.
