# Maintenance staggering v1 — Mac verification

## Material Passport
- Origin Skill: academic-research-suite / experiment-agent, run.
- Origin Date: 2026-10-04; Verification Status: VERIFIED engineering smoke/tests only.
- Branch: codex/maintenance-stagger-v1; protocol: maintenance_stagger_v1_plan.md.

## Evidence
- 14 new tests: backward-reservation phase-stagger witness, no early service with zero lead, busy/backlog calendar, failed-task priority/price sensitivity, beyond-horizon deadline, noncontiguous interval search, reservation feasibility/no overlaps, policy serialization and operating proxy, disjoint seeds/unchanged simulator hash, controlled factorial changes, instance-level noninferiority+operating guard, common shocks, artifact completion/refused overwrite, undefined zero-cost ratios.
- Complete repository suite: **272 passed**, 30.06 seconds; one third-party tqdm warning.
- Final smoke: artifacts/maintenance_stagger_smoke_20261004T140046Z/ . COMPLETED with 128 development rows, 48 evaluation episodes, 120 decisions, 8 coordination rows, all audit gates passed. Source, protocol and dependency hashes reconcile. Partial/final episodes identical. Full sealed panel NOT opened.
- Smoke noninferiority/operational gates are intentionally disabled; not a negative scientific result.
- Freshness scan checked 86 previously synced local manifest files: no seed overlaps with either new full/development or smoke panel. Coverage is limited to this known local registry, not all possible external runs.
- Horizon-24/depth-3 engineering probe used smoke seeds 850100/851100 only: E approximately 0.9–1.0 ms/episode; C approximately 26–27 ms/episode for two price families. These are engineering timings, not matched hardware performance claims or full-run ETA.
- maintenance_coupling.py unchanged, SHA256 4dcbf3acefd4ae4258c591da1f5e0ed11ae74ffbd0de059ee557b4ccff63bc52. Old results untouched. New experiment uses a clean tracked branch and new timestamped artifacts.

## Scope and handoff
Full run: 16 new independent instances × four cells × three downtime prices × two horizons × 30 common shock seeds × B/E/C = 34,560 episode rows. Separate 7,680-row development tuning selects B offset/E lead per price from mean cost per step. There is no RL training. Full trace 20,736 rows; 384 coordination rows. Primary is fixed at high/heterogeneous, price1, horizon24. Cost noninferiority margin 5%; additional operating-time proxy guard allows at most 2 percentage points loss relative to B at the CI lower bound. Gates are diagnostic and do not establish measured production throughput, broad equivalence, or universal heuristic optimality.

Use the lab compound command supplied after push: fetch verified branch, create a NEW detached Git worktree for this experiment, uv sync --frozen, then run `python -m ht_pdm_fjsp.maintenance_stagger --profile full --device cpu` with absolute output directory under ~/ht-pdm-fjsp/artifacts. This keeps an existing checkout available to another experiment. Exact-commit test and full-run clean-source gate prevent accidental use of different source. No SSH/full lab launch/rsync performed by assistant. User pulls artifacts using the canonical rsync -avhP --partial command.
