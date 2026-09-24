# Clean Technician-Conflict Attribution Follow-up

## Material Passport

- Experiment ID: `clean-conflict-attribution-v1`
- Mode: code experiment / confirmatory follow-up
- Parent evidence: `technician_conflict_consequence_full_cpu_20260924T070201Z`
- Verification target: fresh development seeds only
- Sealed status: seeds `63200:63300` remain closed

## Research question and scope

Does technician proposal collision itself cause a material objective loss when
maintenance requests commit machines to wait, after removing the stronger
technician-matching policy from the primary comparison?

This is a protocol amendment motivated by the parent diagnostic. The parent
primary endpoint compared independent greedy dispatch with coordinated matching
and therefore bundled collision avoidance with a different matching rule. This
follow-up uses coordinated greedy dispatch as the comparator. It is a simulator
mechanism experiment, not an RL algorithm claim.

## Locked hypotheses

- **H1 (clean committed loss):** under committed requests, the paired mean
  objective difference

  \[
  \Delta J_s = J_{independent\ greedy,s}-J_{coordinated\ greedy,s}
  \]

  is positive, its paired 95% t-interval has a lower bound above zero, its mean
  is at least 5% of the absolute coordinated objective, and independent greedy
  is worse on at least 70% of seeds.
- **H2 (commitment amplification):** the paired difference-in-differences

  \[
  I_C = \Delta J_{commit=1}-\Delta J_{commit=0}
  \]

  has a paired 95% t-interval lower bound above zero.
- **H3 (operational rather than penalty-only loss):** less than half of the
  committed mean objective gap is explained by the explicit conflict penalty;
  the remainder arises from downstream production, downtime, waiting,
  maintenance, failure, and rework components.

Failure of H1 or H2 means commitment has not cleanly established consequential
coordination pressure and baseline RL replication should not be promoted.

## Fixed conditions and policies

Both cells use the promoted parent configuration: failure cost 36, PM duration
factor 0.35, PM risk threshold 0.005. Maintenance windows and strong
substitution are disabled in both cells.

| Cell | Request commitment | Window | Strong substitution |
|---|---:|---:|---:|
| `commit0_window0_sub0` | no | no | no |
| `commit1_window0_sub0` | yes | no | no |

The fixed policies are:

1. `reactive_fibt`, used only as a preventive-value sanity comparator;
2. `coordinated_preventive`, which visits the common priority ordering and
   reserves the fastest available technician after each assignment;
3. `independent_preventive`, in which every eligible machine uses the same
   fastest-technician preference without reservation, allowing collisions.

No coordinated-matching policy is included. The two preventive policies share
the same transition kernel, candidate set, priority rule, and local
fastest-technician preference. Their intended treatment difference is
one-to-one resource reservation before execution.

## Metrics and attribution

The primary endpoint is paired `independent_preventive -
coordinated_preventive` objective on the committed cell. Lower objective is
better, so a positive difference is independent loss.

Secondary metrics are conflict incidence, win/loss rate, production, downtime,
failures, preventive and corrective actions, waiting, committed waiting,
rework, technician utilization, and workload imbalance. The objective gap is
decomposed exactly using the locked objective weights:

\[
\Delta J = 1.5\Delta conflicts + 3\Delta PM + 8\Delta CM
+2\Delta downtime +36\Delta failures +0.75\Delta waiting
+12\Delta rework +5\Delta earlyPM -\Delta production.
\]

## Seeds, stopping rule, and multiplicity

- Smoke: `63990:63993` (three seeds).
- Full fresh development panel: `63800:63900` (100 seeds).
- Sealed confirmation panel: `63200:63300`, rejected by the runner if supplied.
- Fixed workload: two cells x three policies x seeds x 168 decisions.
- No adaptive stopping, seed extension, or result-dependent condition changes.
- H1 is the primary confirmatory test. H2 and H3 are prespecified mechanism
  checks; no omnibus search or cell selection is performed.

The previous 3% conflict-incidence lower bound was an arbitrary screening edge
and the parent commitment-only estimate was 2.96%. Before examining these fresh
seeds, the practical sanity band is amended and locked to 2.5%--10%.

## Promotion and audit gates

The full run promotes the commitment mechanism only when:

1. committed independent conflict-step incidence is in `[0.025, 0.10]`;
2. H1's relative mean gap is at least 5%;
3. H1's paired 95% lower bound is above zero;
4. independent is worse on at least 70% of committed paired seeds;
5. H2's paired 95% lower bound is above zero;
6. coordinated preventive improves at least 5% over reactive FIBT;
7. coordinated preventive has zero proposal conflicts;
8. H3's explicit-conflict share is below 50%; and
9. completion, reward identity, feasibility, duplicate-resource, and sealed-seed
   audits all pass.

Smoke success is engineering-only: correct counts and schema, progress display,
zero feasibility violations, reward/objective identity, and sealed-panel
closure. Smoke performance never changes the locked full protocol.

## Artifact schema

Each invocation refuses a non-empty output directory and writes:

```text
artifacts/<timestamped-run-id>/
├── clean_conflict_manifest.json
├── source_config.json
├── resolved_base_config.json
├── cells.json
├── episodes.partial.csv
├── episodes.csv
├── decisions.csv
└── summary.json
```

The manifest records the Git revision, runtime, device, fixed policies, seed
panels, counts, completion status, audit status, and promotion verdict. Full
results remain development evidence; the sealed panel is not opened by this
experiment.
