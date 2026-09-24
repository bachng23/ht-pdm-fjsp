# Passive shared-technician MARL pilot

This is a new benchmark isolated from the existing simulators. Machines are
learning agents. Technicians are passive shared resources with fixed,
machine-dependent service times. There is no technician learning, absence,
substitution, experience, window, or commitment mechanism.

The research question is whether independent machine agents can learn when to
request maintenance and which heterogeneous technician to request. The same
finite-horizon transition function is used by independent MARL, a skill-aware
risk-first dispatcher, and exact dynamic programming on the small instance.

The primary metric is total maintenance objective (lower is better). Secondary
metrics are failures, maintenance jobs, waiting requests, and collision count.
The pilot uses training seeds 11--13 and evaluation seeds 101--110; smoke uses
one training seed and three evaluation seeds. Full uses ten and one hundred
seeds respectively, but is not required before reviewing simulator behavior.
The stopping rule is a fixed episode budget (1,500 pilot; 10,000 full), with no
adaptive extension or result-dependent tuning. Every run writes a timestamped
manifest, episodes CSV, summary, and exact optimum cost.

The independent MARL implementation is a small tabular independent Q learner
with simultaneous actions and deterministic collision resolution. It is a
mechanism-level Rodríguez-style baseline, not a claim of numerical replication
of Rodríguez et al.'s PPO implementation.
