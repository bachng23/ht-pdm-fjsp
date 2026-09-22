# Standard FJSP source instances

`brandimarte_mk01.fjs` is an unmodified mirror of Brandimarte MK01 retrieved
from the MIT-licensed SchedulingLab/fjsp-instances repository on 2026-09-22:

<https://github.com/SchedulingLab/fjsp-instances/blob/main/brandimarte/mk01.txt>

SHA-256:

```text
449ada8093e03a84bf2255fa2a1673ccbc80cce7b82e855c5216533a330751ff
```

Original reference:

P. Brandimarte, “Routing and Scheduling in a Flexible Job Shop by Tabu
Search,” *Annals of Operations Research*, 41(3), 157–183, 1993.

The derived `configs/brandimarte_mk01_ht_pdm.json` preserves all job routes,
machine alternatives, and processing times. Degradation, load factors, due
dates, technicians, maintenance durations, restoration factors, and costs are
repository-defined extensions and are not part of the original MK01 instance.
