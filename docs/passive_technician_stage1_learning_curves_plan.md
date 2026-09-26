# Stage 1 six-method learning curves

## Research question and hypothesis

Does RA-QMIX learn a lower-cost maintenance policy than Random feasible,
Skill-aware FIFO, Independent Q-learning, VDN, and standard QMIX as the fixed
training budget increases? The falsifiable hypothesis is that RA-QMIX has the
lowest mean objective cost among the learned methods at 50,000 episodes in
each of the four locked environment cells.

## Locked comparison

The environment cells are in-distribution, early-failure, slow-service, and
combined-pressure. Every learned method uses the same FIFO transition
semantics, architecture budget, training-seed panel, evaluation-seed panel,
and checkpoint protocol. Random feasible and Skill-aware FIFO are fixed
policies evaluated on the same evaluation seeds.

Methods are Random feasible, Skill-aware FIFO, Independent Q-learning, VDN,
standard QMIX, and RA-QMIX (`tqmix`). Learned methods train on five seeds
11--15.
Every checkpoint is evaluated on common development seeds 101--200. Seeds
201--300 are reserved as a sealed future panel and are never evaluated by this
run. Checkpoints are fixed at 5,000, 10,000, 20,000, and 50,000 episodes;
training always runs to 50,000 episodes with no adaptive stopping or
checkpoint selection from evaluation performance.

## Metrics and stopping rule

The primary metric is mean total objective cost per training seed, averaged
over evaluation seeds. Secondary metrics are failures, maintenance jobs,
queue waiting, collisions, invalid requests, busy-technician requests,
unique joint actions, defer fraction, and request count. The stopping rule is
the fixed 50,000-episode budget.

The expected row count is 4 cells times 2 fixed policies times 100 evaluation
seeds, plus 4 cells times 4 learned methods times 5 training seeds times 4
checkpoints times 100 evaluation seeds. The primary learning-curve artifact
is `budget_summary.csv`, grouped by cell, policy, checkpoint, and training
seed.

## Smoke gate

The Mac smoke profile uses one training seed, three evaluation seeds, and
checkpoints 8, 16, and 32. It must complete all expected rows, save and load
every checkpoint, write the manifest and CSV/JSON schema, show `tqdm` progress
for training and evaluation, keep invalid requests at zero, and leave seeds
201--300 unopened. Smoke output is engineering evidence only.

## Interpretation limits

The learning curves establish a controlled Stage 1 comparison on the four
synthetic cells. They do not establish transfer to unseen system sizes,
dynamic resource availability, long-horizon queue prediction, or deployment
safety. Fixed policies are reference policies and are not training-seed
replicates; inference about learned methods uses training seeds as the
independent units.
