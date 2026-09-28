# Experiment 2B — Robust model-checkpoint selection holdout

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan → run
- Origin Date: 2026-09-28
- Verification Status: IMPLEMENTED / LOCAL CPU SMOKE VERIFIED — full holdout pending
- Version Label: `ra_qmix_robust_checkpoint_holdout_v1`

## 1. Research question and status

The stress checkpoint sweep showed that the nominal optimum occurs at 50k while
the stress optima usually occur earlier. It also showed that the original
20k→50k component hypotheses did not replicate on fresh training seeds.

This experiment asks whether a model-checkpoint candidate selected by a locked
multi-scenario minimax-regret rule generalizes to a fresh evaluation panel
better than the default Full RA-QMIX 50k endpoint.

This is a prospective holdout evaluation specified after viewing development
seeds 83500–83599. It performs no training and never reselects a candidate after
opening the holdout panel.

## 2. Locked source and selection rule

Source experiment:

- protocol: `ra_qmix_stress_checkpoint_sweep_v1`;
- Git commit: `72bd25efa63f8df843d0dec1cd7ca7f81ece75f6`;
- training seeds: 83100–83104;
- development evaluation seeds: 83500–83599;
- algorithms: `qmix`, `tqmix_no_queue`, `tqmix_no_cf`, `tqmix`;
- checkpoints: 5k, 10k, 20k, 30k, 40k, 50k.

For candidate `(algorithm, checkpoint)` and scenario `s`, average objective
first over development evaluation seeds within each training seed and then over
the five training seeds. Define the scenario reference as the minimum mean
among all 24 candidates and relative regret as:

`regret(candidate,s) = mean_cost(candidate,s) / reference(s) - 1`.

The robust selection score is maximum regret across the four scenarios. Select
the candidate lexicographically by `(max regret, mean regret, lower checkpoint,
algorithm ID)`. Recomputing this rule on the completed source artifacts must
select **Full RA-QMIX at 40k**. Any mismatch aborts before evaluating holdout
seeds.

The scenario reference values and all 24 selection scores are frozen in the new
artifacts. They scale different scenarios; they are not recomputed from holdout
performance.

## 3. Candidates and hypotheses

Only five candidates are evaluated:

| Candidate ID | Algorithm | Checkpoint | Role |
|---|---|---:|---|
| `selected_full_40k` | `tqmix` | 40k | Development-selected treatment |
| `full_50k` | `tqmix` | 50k | Primary comparator |
| `qmix_50k` | `qmix` | 50k | Standard baseline |
| `no_queue_50k` | `tqmix_no_queue` | 50k | Ablation comparator |
| `no_cf_50k` | `tqmix_no_cf` | 50k | Parameter-matched comparator |

For each training seed, compute scenario means on the fresh panel, normalize by
the frozen development references, and take maximum regret across scenarios.

Primary estimand:

`delta_R(seed) = R(selected_full_40k, seed) - R(full_50k, seed)`.

H-primary predicts a negative mean delta. The locked directional replication
criterion is mean delta below zero and selected Full@40k better in at least four
of five training seeds. This descriptive rule is not a p-value. Failing it does
not trigger re-selection.

Secondary endpoints are mean regret, per-scenario objective and cost per
timestep, cost decomposition, reliability, queue and utilization metrics, and
paired comparisons to the three other 50k candidates. Mechanism metrics remain
diagnostic rather than causal mediation evidence.

## 4. Seeds, stopping rule, and scale

| Item | Full CPU holdout | macOS smoke |
|---|---|---|
| Source train seeds | 83100–83104 | 83100 |
| Evaluation seeds | 83600–83699 | 83700–83702 |
| Candidates | 5 | 5 |
| Scenarios | 4 | 4 |
| Evaluation episodes | 10,000 | 60 |
| Training | None | None |
| Sealed seeds | 201–300, closed | closed |

A textual audit of repository protocols, code, tests, and pulled manifests found
no prior use of evaluation seeds 83600–83699. They are disjoint from training,
development, smoke, and sealed panels.

The run stops only for missing/mutated source artifacts, selection mismatch,
evaluation error, non-finite result, or failed audit. It does not retry, resume,
or substitute candidates. A rerun uses a new timestamped directory.

## 5. Artifact and audit contract

```text
manifest.json
source_manifest_snapshot.json
benchmark_config.json
resolved_config.json
selection_scores.csv
scenario_references.csv
candidate_registry.csv
source_checkpoint_inventory.csv
evaluation_progress.csv
episodes.partial.csv
episodes.csv
coordination.csv
budget_summary.csv
robust_scores.csv
primary_results.csv
summary.json
```

The runner must:

- reject a non-empty output directory;
- validate the source protocol, commit, completion status, counts and audits;
- require every selected checkpoint before opening holdout seeds;
- record SHA-256 for source manifest, budget summary and checkpoints, and verify
  they are unchanged after evaluation;
- independently recompute and lock Full@40k selection;
- use `tqdm` for multi-seed evaluation and stream partial/progress CSVs;
- verify expected row counts, unique keys, finite objectives, cost
  reconciliation, resource semantics, panel disjointness and sealed status;
- record `FAILED` metadata on exceptions without modifying source artifacts.

Smoke passes with 60 rows, five source checkpoints, all candidates/scenarios,
successful save/load through the existing serializer, visible evaluation
progress, unchanged source hashes and all audits true. Full passes with 10,000
rows and 25 source checkpoints.

The full evaluation is CPU-first on the Ubuntu lab and depends on the existing
timestamped source artifact. Codex will test and smoke locally, commit and push,
then provide one compound lab command with explicit source-file preflights plus
the canonical `rsync` command. Codex will not SSH to the lab or launch the run.

## 6. Local verification record

Verified on 2026-09-28 before the full holdout was opened:

- repository suite: 260 tests passed;
- real-source CPU smoke: 60/60 evaluation rows and 5/5 checkpoint files;
- the locked rule independently reproduced Full RA-QMIX@40k;
- source manifest, summary and checkpoint hashes were unchanged;
- all row-count, uniqueness, finite-objective, cost, resource, progress,
  panel-disjointness and sealed-panel audits passed;
- local smoke artifact:
  `/private/tmp/ra_qmix_robust_holdout_smoke.FBZpeg/run`.

The smoke primary delta is an engineering diagnostic from one training seed and
three new evaluation seeds per scenario; it is not interpreted as evidence for
or against the full five-seed hypothesis.
