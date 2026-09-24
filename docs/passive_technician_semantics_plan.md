# Collision semantics screen

This development experiment selects the transition semantics for the passive
shared-technician benchmark. It replays the same seeded independent proposals
under three modes: lowest-index priority resolution, FIFO technician queues,
and Rodríguez-style invalid duplicate assignment. Random-independent and
skill-greedy-independent proposal policies expose contention without adding a
second learning algorithm.

The primary metrics are proposal conflict events, invalid proposals, waiting,
objective, and jobs. Machine-level accepted-job counts diagnose index bias.
Smoke uses three seeds and pilot uses thirty seeds. Every mode and policy sees
the same seed panel. The locked choice after the pilot is FIFO queue semantics
for the main research simulator. Invalid collision is retained as the
Rodríguez reproduction mode, and priority resolution is diagnostic only because
it gives the lowest-index machine preferential access under random proposals.
