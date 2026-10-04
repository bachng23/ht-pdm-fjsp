## Material Passport
- Type: experiment engineering report; status: VERIFIED (macOS smoke only).
- Protocol: ra_qmix_heuristic_assignment_v1. Full scientific run: NOT RUN.
- Existing TD-assignment scientific FAIL is unchanged.

## Verification
Full repository suite: 369 passed, one pre-existing SB3 tqdm experimental warning, 182.60 seconds. New module: 18 tests passed; rerun after disjoint RNG stream amendment: 18 passed in 2.66 seconds. Tests cover ordered peer padding at all four sizes, observable-only teacher access, deterministic assignment/deferral, busy queue workload, masks/matching, exogenous failure noise, configuration/seed splits, TD baseline teacher isolation, labeled replay ring alignment, checkpoint contract, monotonic mixer, nonempty-output rejection, and inference/practical-effect gates.

Final smoke directory: `artifacts/ra_qmix_heuristic_assignment_smoke_20261004T124336Z`; status COMPLETED. Final source SHA256: `406bbfed41a7fe257ee81301623b8b318399d4fe063d4153dcc85376cdb1e4f7`.
The first smoke remains preserved; final smoke follows the documented engineering RNG amendment. No heuristic, gate, matrix or hyperparameter was tuned from smoke performance.

| Artifact/metric | Expected and actual |
|---|---:|
| Saved/reloaded models | 9 |
| Real environment steps | 1,080 |
| Optimizer updates | 243 |
| Training episodes | 360 |
| Training progress rows | 54 |
| Test episodes | 81 |
| Development episodes | 9 |
| Heuristic reference episodes | 40 |
| Small exact evaluation rows | 12 |
| Paired training seed rows | 1 |

All 20 manifest engineering audits PASS. Independent CSV/count checks stored in smoke `independent_smoke_checks.json`; training and evaluation tqdm verified in console log. All evaluation/reference and recorded training episode invalid requests are zero. Cost components reconcile. Labels are absent from TD-only and recorded for exactly 120 visited transitions in each supervised smoke model. Initialization/parameter counts match within size. Checkpoint logits and total-Q reload exactly, including rollout states on all six configurations; monotonic local vs joint max audit checks initial states. Oracle labels are confined to the two training matrices at N=2. No sealed evaluation seeds opened. Smoke scientific pass and CI fields are null; no efficacy interpretation.

## Small oracle capacity preflight
Without fitting models or evaluating their costs, built full-horizon N=2 DAGs and checked Bellman residual: train A/B 413 states each; development 375; nominal 297; medium 381; high 437. All residuals <1e-9 and all counts below cap 100,000. Larger sizes never call exact oracle. Exact evaluation DAGs are cached once per small configuration, not rebuilt per trained model.

## Full handoff scope
90 models, 5,400,000 real training transitions, 1,338,570 optimizer updates, 600,000 training episodes, 13,500 held-out test episodes, 1,800 development episodes and 680 heuristic references. 10 training replicates; inference unit is paired training seed, not rollout episodes. CPU only. Separate models for each N: configuration transfer within size, no claim of unseen-size transfer. Heuristic is training-only; execution uses local masked argmax with peer telemetry and fixed machine ID. Primary N=3–5 diagnostic gate is locked at >=5% reduction, >=8/10 wins, paired CI upper <0, nominal regression <=10%.

The run must be launched by the user in the lab terminal; user also runs the canonical progress-preserving rsync on Mac. No remote launch or transfer was performed by the assistant. No retries/resume; failed partial directories are preserved and a new attempt needs a fresh timestamp.

## Provenance repair validation
The interrupted lab run at first-release commit 1abdf28 reread the runner source while serializing checkpoints. That source path disappeared during training; the traceback establishes disappearance, not whether another checkout operation caused it. Capture source bytes/SHA256 before training, save source_snapshot.py, and reuse the frozen hash for checkpoints and final teacher audit. This also prevents a later edited file from being mislabeled as the running code. Scientific settings unchanged. No resume/retry behavior added and failed full artifacts remain preserved.

Targeted suite after repair: 19 passed in 3.77 seconds, including an end-to-end regression deleting a temporary source path during fit and asserting COMPLETED plus unchanged snapshot/hash in every checkpoint and final audit. Previous repository suite (369 tests) belongs to first release; no claim of a new full-suite rerun.
Fresh macOS smoke: `artifacts/ra_qmix_heuristic_assignment_smoke_20261004T151117Z`, COMPLETED, all audits PASS. 9 models; 1,080 real steps; 243 updates; 81 test, 9 development and 40 reference episodes. Snapshot SHA256 verified against manifest: `94b99bd6db339a82d9a6a00073dbb750e93f909b0ce60d3dad0ec46ca6541d7f`. Old smoke artifacts retained. Full rerun remains human-launched with a new timestamp directory and the original 90-model/5.4m-step budget.
