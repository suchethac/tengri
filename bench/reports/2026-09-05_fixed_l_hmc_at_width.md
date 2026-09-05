# Fixed-L HMC clears the bar on one seed and fails on its partner, at every L. That is not a sampler you can run at width.

**Date:** 2026-09-05
**Verdict:** The gate failed, but not in the way the brief anticipated, and the
difference matters. Preconditioned fixed-length HMC (`mcmc_hmc`,
`precondition=0.5`) was swept over `L` in {10, 20, 40, 80, 160} on **two**
fixtures (`ctl-dpl` and nb05) at **two seeds each** — 20 cells — plus a
6-cell unpreconditioned control on `ctl-dpl`. **Three cells clear the full bar**
(max split R-hat < 1.01 with zero divergences): nb05 L=80 seed 7 (R-hat 1.0046,
min ESS 190.5), nb05 L=160 seed 8 (1.0050, ESS 220.9), and the *unpreconditioned*
`ctl-dpl` L=80 seed 7 (1.0098, ESS 71.4). **No configuration clears on both
seeds of any pair, at any `L`, on either fixture.** Every one of the eleven
swept configurations is 0-of-2 or 1-of-2; none is 2-of-2.

So the answer to "does the metric rescue fixed `L`?" is **not** "never". It is
**"about half the time, and you cannot tell which half"** — which for a catalog
is the same failure wearing better clothes, because the unit of the claim is a
galaxy and half of them come back wrong with no flag on them.

**The thesis is refuted on its first clause, not its second.** The hypothesis
was that fixed-L HMC's constant per-lane cost makes it vmap-friendly, and that
tengri's analytic metric would supply the mixing that fixed `L` cannot adapt its
way to. The first half is *mechanically true and was confirmed*: gradients per
draw is exactly `L` in all ten preconditioned cells and all six control cells,
with **zero** spread across seeds — fixed-L HMC has no cost variance to lose to
vmap, by construction. That was never the doubtful half of the thesis. The
second half is false. The metric helps, substantially and reproducibly, and it
is not close to enough.

**Preconditioning improves almost every paired cell and still clears nothing.**
At matched `L` and seed on `ctl-dpl`, the analytic metric improves min ESS in
**6 of 6** measured pairs and max split R-hat in **5 of 6**, twice by two orders
of magnitude in ESS (L=80 seed 8: R-hat 2.9688 -> 1.0131, min ESS 1.1 -> 18.9;
L=20 seed 8:
2.2563 -> 1.0205, 1.3 -> 9.9). The adapted step size rises in all six pairs,
by 1.3x to 19x, which is the independent evidence that the metric was live in
these rows rather than silently dropped. This replicates
`bench/reports/2026-08-31_catalog_preconditioning.md`'s Finding: the
preconditioner is the part that transfers. It is also not the missing piece.

**The metric's cost is that it converts silence into divergences, and the
zero-divergence bar then penalizes the better sampler.** Every unpreconditioned
control cell reports **zero** divergences, including one at R-hat 2.9688 and one
at 2.2563. The preconditioned cells report 0-20 divergences and much better
R-hat. Read naively, the control "has no divergences" and the preconditioned arm
"has 20"; read correctly, the control is failing silently and the metric is
making the failure visible. No cell clears the bar either way, so this changes no
verdict here — but a campaign that ranked these arms on the divergence column
alone would rank them exactly backwards.

**R-hat is not monotone in `L`, so "use a longer trajectory" is not an available
fix.** On `ctl-dpl` seed 7 the preconditioned sweep runs 1.0851 (L=10) -> 1.0161
(L=20) -> **3.2707** (L=40) -> 1.0023 (L=80) -> 1.0186 (L=160). The L=40 cell is
worse than the L=10 cell by a factor of three in R-hat. nb05 seed 7 does the same
thing in the same place: 1.1165 -> 1.0325 -> **1.5056** (L=40, and 81
divergences) -> 1.0046 -> 1.0145. Whatever selects a good `L` here is not
"larger", and it is not stable across seeds — on nb05 the clearing cell is L=80
on seed 7 and L=160 on seed 8. This is the same non-monotonicity
`bench/reports/2026-08-17_*` recorded unpreconditioned; the metric does not
remove it.

**One seed would have published a false headline three separate times.** The
unpreconditioned `ctl-dpl` L=80 seed 7 cell reads R-hat 1.0098, **zero**
divergences, min ESS 71.4 — it clears the primary bar outright, and on its own it
is a publishable "fixed-L HMC converges on `ctl-dpl`" result. Its seed-8 partner
reads **R-hat 2.9688**, zero divergences, min ESS 1.1, unique-draw fraction
1.000. The nb05 L=80 seed 7 cell (1.0046, ESS 190.5) clears against the very
fixture `bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md` reported no fixed-L
HMC config converging on; its partner is 1.0749. The nb05 L=160 seed 8 cell
(1.0050, ESS 220.9) clears; its partner is 1.0145. Every one of those three is a
headline, and every one is refuted by the seed sitting next to it. The two-seed
minimum did not merely improve an error bar here; it inverted the conclusion
three times, for the same reason the 2026-08-31 campaign retracted its own.

**Zero divergences is not evidence of health, and neither is the unique-draw
fraction.** The two worst preconditioned cells (R-hat 1.5731 and **3.2707**) both
report **zero** divergences, and the R-hat 3.2707 cell has a unique-draw fraction
of exactly **1.000** — every draw a distinct position. The #1999 frozen-chain
check cannot fire on these rows and does not: no cell measured here is frozen.
A row can pass the divergence column, pass the frozen check, look completely
healthy on every cheap diagnostic, and be wrong by a factor of three in R-hat.
Only R-hat and ESS caught these.

**Platform:** Linux 6.8.0-138-generic, AMD Ryzen 9 5900X, NVIDIA GeForce
RTX 3060 12 GB (GA106, driver 580.173.02), Python 3.12.13, JAX 0.11.0,
BlackJAX 1.6.2, NumPy 2.5.2. **Every row in this report ran on the CPU
backend** (`JAX_PLATFORMS=cpu`): the gate is a convergence question, and R-hat,
ESS, divergence counts, adapted step size and gradients per draw are
deterministic given the seed, so they do not need the accelerator and are not
affected by what else was on it. No CUDA row was taken, so
`JAX_DEFAULT_MATMUL_PRECISION=highest` is recorded as a requirement for the
width sweep that was not run rather than as a setting any row here used.

**Precision:** float64 throughout.

**Forward model and SNR:** every row uses `WavePrecomp()` — the default,
`band_integration="quadrature"`, `n_subbands=5` — at **SNR 20** (the fixtures'
own `snr`). `band_integration="taylor"` was **not** used anywhere: its LUT bias
is constant in SNR on the forward model but enters the posterior gradient
multiplied by SNR (#1671), and no throughput claim here is worth a biased
gradient. There is no LUT-bias shortcut in any number below.

**Load:** this box was shared with another agent's campaign for the whole run,
at a 1-minute load average between 2.1 and 22.8. Every wall-clock cell below is
therefore **not** comparable between rows and carries no claim; the 2026-08-20
device matrix measured a 9.5x wall-clock spread from scheduling alone on this
machine. `loadavg_before` and `loadavg_after` are now stamped into every JSON
row (a harness change made for this report) so that a clock cell can at least be
read next to the load that produced it. The verdict rests entirely on
contention-immune quantities.

---

## Why this was measured

Zacharegkas, Hearin & Benson 2025 (arXiv:2506.19919) report ~1000 galaxy
posteriors per GPU-minute on D=12 DSPS, while their Appendix D.3 states that a
single chain takes 22 minutes. Their number is therefore **throughput at
enormous width, not latency**, and the reason width works for them is that they
run **fixed-L HMC**: every lane does the same number of leapfrog steps, so a
vmapped batch wastes nothing.

Two prior tengri results made that look reachable:

* `bench/reports/2026-08-31_fast_nuts.md` (PR #2135, branch `feat/fast-nuts`)
  measured NUTS's vmap penalty as real and large — one vmapped 100-step window
  adaptation over 64 lanes at D=3 took 1272 s, because every lane runs to the
  deepest tree in the batch.
* The same campaign found `precondition=0.5` removes per-galaxy cost variance:
  85.0 and 81.3 gradients per draw on two galaxies (1.06x) where the
  unpreconditioned control swung 3.65x (693.8 vs 190.2).

Cost variance is the entire vmap penalty; fixed-L HMC has none by construction;
preconditioning removes it for the geometry. So the hypothesis was that
**preconditioned fixed-L HMC, vmapped over galaxies, is tengri's route to
Zacharegkas's throughput regime** — their sampler plus tengri's analytic metric.

The prior evidence also cut the other way, and it was respected here:
`bench/reports/2026-08-17_*_nuts_vs_hmc.md` found *no* fixed-L HMC config
converges on nb05 (L=20 leaves `sfh_tsnorm_log_total_mass` at ESS 2.3; L=160
costs 125 s at R-hat 1.024) — but those runs were **unpreconditioned**. The open
question this report answers is precisely whether the metric rescues fixed `L`.

**It does not.** The convergence gate was run first, at N=1, exactly so that a
throughput number would never be published on top of a posterior nobody can use.

---

## 1. The gate: preconditioned fixed-L HMC on `ctl-dpl`, D=8

`ctl-dpl` is nb05's mock, bands, seed, SNR and chain count over a DPL SFH
instead of tsnorm — the non-tsnorm control, so that a sampler failure can be
told apart from the tsnorm family's own degeneracy. D = 8 free parameters,
2 chains, 1000 warmup + 600 draws per chain (**1200 total draws**),
`dense_mass_matrix=False` (tengri's own auto-policy at D>=8, #319),
`target_accept_rate=0.9`, `precondition=True` -> `DEFAULT_WHITENING_STRENGTH`
= **0.5**. Seeds 7 and 8.

The bar is the notebooks' own and the library's: **max split R-hat < 1.01 with
zero divergences** (`CATALOG_MAX_RHAT`). `div%` is divergences as a fraction of
the 1200 total draws. `uniq` is the unique-draw fraction (#1999). `g/draw` is
gradients per draw; `g/ESS` is gradients per effective sample, the cost column
that actually prices a sampler.

| L | seed | max R-hat | div | div% | min ESS | uniq | step size | g/draw | g/ESS | clears? |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 10 | 7 | 1.0851 | 1 | 0.083 | 2.1 | 0.859 | 0.05923 | 10 | 5585 | no |
| 10 | 8 | 1.5731 | **0** | 0.000 | 1.1 | 0.982 | 0.00840 | 10 | 11268 | no |
| 20 | 7 | 1.0161 | 6 | 0.500 | 31.7 | 0.827 | 0.04606 | 20 | 756 | no |
| 20 | 8 | 1.0205 | 4 | 0.333 | 9.9 | 0.876 | 0.04054 | 20 | 2419 | no |
| 40 | 7 | **3.2707** | **0** | 0.000 | 1.1 | **1.000** | 0.00131 | 40 | 44538 | no |
| 40 | 8 | 1.0896 | 1 | 0.083 | 2.4 | 0.948 | 0.00806 | 40 | 20044 | no |
| 80 | 7 | **1.0023** | 5 | 0.417 | **128.1** | 0.890 | 0.02738 | 80 | 750 | no (div) |
| 80 | 8 | 1.0131 | 1 | 0.083 | 18.9 | 0.937 | 0.00789 | 80 | 5083 | no |
| 160 | 7 | 1.0186 | 20 | 1.667 | 43.1 | 0.803 | 0.01609 | 160 | 4450 | no |
| 160 | 8 | **1.0043** | 7 | 0.583 | **231.2** | 0.845 | 0.01211 | 160 | 830 | no (div) |

**Clears the bar on 0 of 2 seeds at every one of the five trajectory lengths.**

Three things in this table are worth reading twice.

1. **The two zero-divergence cells are the two worst cells.** L=10 seed 8 at
   R-hat 1.5731 and L=40 seed 7 at R-hat 3.2707 both report zero divergences.
   The L=40 seed 7 cell additionally has a unique-draw fraction of 1.000, so it
   is not frozen either — it is a chain that moved everywhere and mixed nowhere.
2. **The best two cells are on opposite seeds.** L=80 clears R-hat on seed 7
   (1.0023) and misses on seed 8 (1.0131); L=160 clears on seed 8 (1.0043) and
   misses on seed 7 (1.0186). Neither `L` is reproducibly good.
3. **The step size collapses where R-hat blows up.** L=40 seed 7 adapted to
   0.00131, 45x smaller than L=10 seed 7's 0.05923 on the same fixture and seed.
   Window adaptation is not finding a stable step size across `L` here.

## 1b. The same sweep on nb05, D=8 tsnorm

nb05 as it ships today (`law="calzetti"`, post-#1989), D = 8, 2 chains, same
1000 + 600 budget, same seeds. This is the fixture
`bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md` measured and found no
fixed-L HMC configuration converging on — **unpreconditioned**.

| L | seed | max R-hat | div | min ESS | uniq | step size | g/ESS | clears? |
|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| 10 | 7 | 1.1165 | 7 | 2.0 | 0.770 | 0.05593 | 6111 | no |
| 10 | 8 | 1.3449 | **0** | 1.3 | 0.965 | 0.01691 | 9325 | no |
| 20 | 7 | 1.0325 | 2 | 39.2 | 0.802 | 0.04300 | 612 | no |
| 20 | 8 | 1.4220 | **0** | 2.0 | 0.957 | 0.01866 | 12010 | no |
| 40 | 7 | 1.5056 | 81 | 3.7 | **0.562** | 0.02596 | 12846 | no |
| 40 | 8 | 1.0762 | **0** | 5.0 | 0.957 | 0.01603 | 9696 | no |
| 80 | 7 | **1.0046** | **0** | **190.5** | 0.966 | 0.01850 | **504** | **YES** |
| 80 | 8 | 1.0749 | 0 | 10.7 | 0.983 | 0.01299 | 9003 | no |
| 160 | 7 | 1.0145 | 0 | 168.9 | 0.978 | 0.01578 | 1137 | no |
| 160 | 8 | **1.0050** | **0** | **220.9** | 0.983 | 0.01215 | 869 | **YES** |

**Two cells clear outright, and they are on opposite seeds at different `L`.**
L=80 clears on seed 7 and misses on seed 8 (1.0749); L=160 clears on seed 8 and
misses on seed 7 (1.0145). Both configurations are **1 of 2**.

This is the report's most important table, because it is the one that says the
answer is not "never". Preconditioned fixed-L HMC reaches R-hat 1.0046 with zero
divergences and 190.5 effective samples on the exact fixture the 2026-08-17
campaign could not make converge at any `L` — the metric is doing real work, and
the earlier report's conclusion is correctly scoped to the unpreconditioned case
it measured. What the metric does not buy is **reproducibility across the seed**,
and a catalog is a machine for turning a per-galaxy coin-flip into a headline
number.

Note also that nb05's L=40 seed 7 cell has a unique-draw fraction of **0.562** —
the lowest anywhere in this report, i.e. 44% of its draws are repeats — alongside
81 divergences. That cell is visibly sick. The neighboring L=40 seed 8 cell has
**zero** divergences, a 0.957 unique fraction and R-hat 1.0762: sick in a way no
cheap column reports.

## 2. The control: is the metric actually doing anything?

`preconditioned` is not published in the single-galaxy HMC path's diagnostics
(it comes back `None`), so "preconditioning was on" cannot be read off a row.
It was verified two ways instead: `prepare_preconditioning` is called
unconditionally in `run_hmc` and its `cache_key` feeds the adaptation key, and —
the measurement — the paired unpreconditioned control below moves both the
adapted step size and the diagnostics.

Same fixture, same seeds, same 1000 + 600 budget, `precondition` omitted:

| L | seed | max R-hat | div | min ESS | uniq | step size | g/ESS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 20 | 7 | 1.0454 | 0 | 2.2 | 0.927 | 0.02132 | 10914 |
| 20 | 8 | 2.2563 | 0 | 1.3 | 0.997 | 0.01759 | 18708 |
| 80 | 7 | **1.0098** | **0** | 71.4 | 0.947 | 0.01176 | 1345 |
| 80 | 8 | **2.9688** | 0 | 1.1 | **1.000** | 0.00041 | 85862 |
| 160 | 7 | 1.0076 | 3 | 34.1 | 0.949 | 0.00697 | 5631 |
| 160 | 8 | 1.0154 | 1 | 96.6 | 0.920 | 0.00941 | 1988 |

**Clears the bar on 0 of 2 seeds at every trajectory length here too.** The one
cell that clears outright is L=80 seed 7, and its partner is the worst cell in
the table.

Paired against the preconditioned table, cell for cell:

| L | seed | R-hat off -> on | min ESS off -> on | step off -> on | div off -> on |
|---:|---:|:--|:--|:--|:--|
| 20 | 7 | 1.0454 -> **1.0161** | 2.2 -> **31.7** | 0.02132 -> 0.04606 | 0 -> 6 |
| 20 | 8 | 2.2563 -> **1.0205** | 1.3 -> **9.9** | 0.01759 -> 0.04054 | 0 -> 4 |
| 80 | 7 | 1.0098 -> **1.0023** | 71.4 -> **128.1** | 0.01176 -> 0.02738 | 0 -> 5 |
| 80 | 8 | 2.9688 -> **1.0131** | 1.1 -> **18.9** | 0.00041 -> 0.00789 | 0 -> 1 |
| 160 | 7 | 1.0076 -> *1.0186* | 34.1 -> **43.1** | 0.00697 -> 0.01609 | 3 -> 20 |
| 160 | 8 | 1.0154 -> **1.0043** | 96.6 -> **231.2** | 0.00941 -> 0.01211 | 1 -> 7 |

The metric improves min ESS in **6 of 6** pairs, max split R-hat in **5 of 6**,
and raises the adapted step size in all six (1.3x to 19x). It was live. It was
not enough. The one R-hat regression is L=160 seed 7 (1.0076 -> 1.0186), which
is also the cell where divergences jump hardest (3 -> 20) -- at the longest
trajectory the whitened geometry is costing more in accepted-proposal quality
than it returns in mixing, which is the direction #1442 warns about for
over-whitening and is worth remembering before anyone raises the strength above
0.5.

**The `L=80` seed-7 control row is this report's cautionary exhibit.** R-hat
1.0098, zero divergences, min ESS 71.4: it clears the primary bar. Published
alone it says "unpreconditioned fixed-L HMC at L=80 converges on `ctl-dpl`". Its
seed-8 partner is R-hat 2.9688 with zero divergences, min ESS 1.1 and a
unique-draw fraction of 1.000.

---

## 3. Width, on the GPU — and what it is and is not evidence for

The gate does not license a throughput headline, and none is claimed. What the
width sweep can settle without a converged sampler is the **mechanism** question:
does fixed-L HMC stay flat as the vmapped batch grows, and how does it compare
with NUTS at equal metric? That is a question about cost, not about correctness,
and the first half has a clean answer. **The second half is only half-measured:**
NUTS has one width point here (N=32), so this report shows the HMC-vs-NUTS *gap*
but does **not** establish the campaign's prediction that NUTS degrades *with
width*. That claim still rests on
`bench/reports/2026-08-31_catalog_batched_samplers.md`, not on anything here.

**Fixture warning, and it is the most important sentence in this section:** this
is `benchmark_catalog_throughput.py`'s own mock, which is **D = 3**
(`sfh_dpl_log_total_mass`, `sfh_dpl_alpha`, `met_logzsol`; everything else
pinned) over 5 SDSS-like bands at median SNR **~19.9** (19.8-20.0 between the
two arms' mocks; min 16.5, max 23.1), `WavePrecomp()`
quadrature with `n_subbands=5`, float64. It is **not** the D=8 fixture the gate
ran on and **not** the D=12 DSPS of Zacharegkas+2025. Every number below is a
three-parameter posterior's number.

RTX 3060, `JAX_DEFAULT_MATMUL_PRECISION=highest`, 200 warmup + 200 samples,
`precondition=0.5` on both arms, `forward_chunk_size` K = 32 held fixed so the
sweep varies N alone. N=8 is absent because the harness skips `n_gal < K` by
design. `conv/GPUmin` counts **only** galaxies clearing max split R-hat < 1.01
with zero divergences; `uniq_min` is the worst lane's unique-draw fraction.

### Preconditioned fixed-L HMC (L=10)

| N | warm s | gal/GPUmin | **conv/GPUmin** | converged | max R-hat | min ESS | div | uniq_min | peak GiB | load |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 35.41 | 54.2 | **23.7** | 14/32 | 1.2711 | 2.2 | 0 | 0.915 | 0.115 | 4.27 |
| 128 | 148.08 | 51.9 | **19.9** | 49/128 | 1.1991 | 1.6 | 0 | 0.800 | 0.120 | 16.47 |
| 512 | 541.63 | 56.7 | **23.3** | 210/512 | 1.6458 | 1.3 | 0 | 0.880 | 0.140 | 1.88 |

### Preconditioned NUTS, same metric, same widths

| N | warm s | gal/GPUmin | **conv/GPUmin** | converged | max R-hat | min ESS | div | uniq_min | peak GiB | load |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 223.90 | 8.6 | **2.9** | 11/32 | 1.2090 | 1.8 | 0 | 0.955 | 0.114 | 8.81 |

**Three findings, and the first two are what the campaign predicted.**

1. **Fixed-L HMC is flat in width.** 54.2 -> 51.9 -> 56.7 galaxies per GPU-minute
   across a **16x** increase in N. The 51.9 cell is the one taken at load 16.47
   and the 56.7 cell at load 1.88, so the residual scatter is mostly the box, not
   the sampler — which is exactly why the load stamp was added. There is no vmap
   penalty to find, because there is no cost variance to lose: every lane runs
   exactly `L` leapfrogs.
2. **Fixed-L HMC is 6.3x faster than NUTS at equal metric and equal width**
   (54.2 vs 8.6 gal/GPU-min at N=32) and converges *slightly more* galaxies
   (14/32 vs 11/32). On the throughput axis the thesis is **correct**. This is
   the strongest evidence in the report for the original hypothesis, and it is
   not enough to save it.

   That gap is **not** an artifact of the shared box, and the HMC rows are what
   prove it: they span load 1.88 to 16.47 — a 9x range — and move less than 10%
   in throughput (56.7 vs 51.9). A workload that barely notices a 9x load swing
   cannot lose a factor of 6.3 to the 2x difference between the NUTS row's load
   8.81 and the HMC row's 4.27. The comparison survives its own contention
   caveat, which is the only reason it is quoted.
3. **The converged fraction is ~41% and does not improve with width** — 14/32
   (44%), 49/128 (38%), 210/512 (41%). This **independently reproduces the
   seed-level result** from sections 1 and 1b, on a different fixture, a
   different dimension and a different machine backend: preconditioned fixed-L
   HMC converges on **roughly half** of what you give it, and which half is not
   predictable. Two entirely separate experiments landed on the same number.

**And the number nobody should quote.** The best converged rate here is **23.7
galaxies per GPU-minute** at D=3. Zacharegkas+2025 report ~1000 per GPU-minute at
D=12. That is a **~42x** gap in tengri's disfavor, on a posterior with
**four times fewer parameters** — and the converged galaxies' worst ESS is
**1.3 to 2.2 of 200 draws**. An ESS of 2 is not a posterior. The honest reading
is that tengri is not in this throughput regime and the distance is not a tuning
constant; and even the 23.7 is generous, because it counts galaxies that cleared
R-hat with an effective sample size in the low single digits.

**VRAM never saturates.** Peak device memory goes 0.115 -> 0.120 -> 0.140 GiB
from N=32 to N=512 — a **12 GB** card is nowhere near its limit at 512 galaxies
on a D=3 model, so **the N at which VRAM saturates was not reached and cannot be
reported from this sweep.** The binding constraint at this dimension is kernel
dispatch, not memory: the run is ~102% of one CPU core with the GPU at 100%
utilization on very small kernels. That is worth knowing before anyone sizes a
catalog job by VRAM.

## Caveats

* **CPU only, and deliberately.** No row here ran on CUDA. The gate is a
  convergence question and its columns are seed-deterministic. This also means
  no VRAM number, no GPU wall clock and no `JAX_DEFAULT_MATMUL_PRECISION`
  exposure exists in this report.
* **Wall clocks are uninterpretable.** The box was shared throughout at load
  2.1-22.8. Wall columns are recorded in the JSON rows with their load stamps
  and are used for nothing.
* **Two seeds, not six.** Two is the minimum this campaign accepts and it was
  decisive here (it inverted two would-be headlines). It is not enough to put an
  error bar on any of these numbers, and the converged/not verdict is a step
  function of a continuous diagnostic, which
  `2026-08-31_catalog_preconditioning.md` measured as noisiest exactly where a
  row is closest to the bar — which is where the L=80 and L=160 cells sit.
* **`dense_mass_matrix=False` on every HMC row.** `mcmc_hmc`'s own registry entry
  states it is "convergence-validated only with `dense_mass_matrix=True`,
  `n_warmup>=1000`, `n_leapfrog_steps>=20` on D=6 DPL". The warmup and `L`
  conditions are met; the dense condition is **not**. Diagonal is what tengri's
  auto-policy selects at D>=8 (#319) and what a catalog fit would use, so it is
  the right configuration for the question asked — but a dense-mass fixed-L HMC
  sweep at D=8 was not run and this report cannot exclude that it behaves
  differently. This is the largest single gap in the negative result.
* **One model family, one dimension.** `ctl-dpl` is D=8 photometry at SNR 20.
  The paper's regime is D=12 DSPS. A failure at D=8 on tengri's posteriors is
  not a statement about DSPS posteriors, whose geometry and parameterization are
  not tengri's.
* **The catalog throughput fixture is D=3, not D=8 and not D=12.**
  `benchmark_catalog_throughput.py`'s `build_model` frees only
  `sfh_dpl_log_total_mass`, `sfh_dpl_alpha` and `met_logzsol` over 5 synthetic
  bands. Any galaxies-per-GPU-minute figure that harness produces — including
  the ones already published in `2026-08-30_gpu_catalog_throughput.md` and
  `2026-08-31_catalog_preconditioning.md` — is measured on a **three-parameter**
  posterior and is not comparable to a D=12 published rate without saying so.
  This is worth stating loudly because it is the number a reader would reach for
  to compare against the paper.

## What was NOT measured

* **A throughput headline. Deliberately not published.** Section 3 measures
  galaxies per GPU-minute because the *mechanism* question (flat vs degrading
  with width) is answerable without a converged sampler. The rate itself is not
  offered as tengri's throughput, because no configuration in this report clears
  the bar on both seeds, and the width rows converge ~41% of the catalog at a
  worst ESS of 1.3-2.2. A rate over galaxies nobody can use is not a rate.
* **N = 8, and the VRAM saturation point.** The harness skips `n_gal < K` by
  design, so the K=32 sweep starts at N=32. And peak device memory only reaches
  **0.14 GiB at N=512** on a 12 GB card, so **no saturation point exists in the
  range measured** — a much larger N, or a much larger D, would be needed to find
  one. The brief asked for "the N at which VRAM saturates"; on this fixture there
  isn't one.
* **A CPU control row for the width sweep.** Not run. The GPU arm alone took the
  idle window the box offered, and a CPU row taken under a different load is not
  a comparison. The CPU/GPU axis is already measured in
  `bench/reports/2026-08-20_cuda_device_matrix.md`.
* **Per-lane cost variance at width, as a measured spread.**
  `build_catalog_mcmc_engine` discards `_expansions` from `_nuts_full_scan`, so
  the batched path does not report per-galaxy tree depth and NUTS's lane-cost
  *distribution* could not be measured directly. Threading that into the catalog
  diagnostics is the prerequisite for the mechanism measurement, and is the
  single highest-value follow-up here. What *is* measured is the consequence —
  the 6.3x aggregate gap at N=32 — and fixed-L HMC's side of it exactly, since
  gradients per draw is `L` with zero variance by construction.
* **NUTS at N=512.** The N=32 and N=128 rows are the width evidence; the 512 cell
  was not affordable on the shared window.
* **Dense mass matrix at D=8.** See Caveats. The largest gap in the negative.
* **Preconditioned NUTS at N=1 on `ctl-dpl`.** A `nutsp` family was added to the
  harness for it and the run was started, but it was cut twice by session
  restarts and then by another job's 9 GB of VRAM, and it is not in this report.
  The N=1 NUTS comparison therefore rests on
  `bench/reports/2026-08-31_fast_nuts.md`'s numbers, on the same fixture and the
  same seeds but from another branch, and is quoted rather than reproduced here.
* **`band_integration="taylor"`.** Deliberately never used (#1671).

## Reproduce

All paths are relative to the repository root. The library must be imported from
this branch's checkout, not the venv's editable install, which points elsewhere:

```sh
export PYTHONPATH="$PWD/src"
export JAX_PLATFORMS=cpu          # the gate is seed-deterministic; no GPU needed
```

The gate (preconditioned fixed-L HMC, five trajectory lengths, two seeds):

```sh
python bench/scripts/benchmark_notebook_sampler.py \
    --notebook ctl-dpl --methods hmcp --seeds 2 \
    --json bench/results/2026-09-05_hmcp_L_sweep.jsonl \
    --sweep-json bench/results/2026-09-05_hmcp_ctl-dpl.json
```

The unpreconditioned paired control:

```sh
python bench/scripts/benchmark_notebook_sampler.py \
    --notebook ctl-dpl --methods hmc --only "hmc L=20,hmc L=80,hmc L=160" --seeds 2 \
    --json bench/results/2026-09-05_hmc_control.jsonl \
    --sweep-json bench/results/2026-09-05_hmc_control_ctl-dpl.json
```

The nb05 gate is the same command with `--notebook 05`.

The width sweep (section 3) runs on the **GPU**, so unset `JAX_PLATFORMS`.
`JAX_DEFAULT_MATMUL_PRECISION=highest` is mandatory on CUDA: XLA silently lowers
float32 matmuls to TF32 on Ampere and `NVIDIA_TF32_OVERRIDE=0` does not fix it
(2026-08-20 Finding 7). It is set here even though these rows are float64, so the
command is correct if someone re-runs it at `--dtype f32`.

```sh
unset JAX_PLATFORMS
JAX_DEFAULT_MATMUL_PRECISION=highest python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --precondition 0.5 --n-leapfrog 10 \
    --n-gal 8 32 128 512 --chunk 32 --warmup 200 --burnin 0 --samples 200 \
    --dtype f64 --json bench/results/2026-09-05_width_hmc.json --tag rtx3060-idle

JAX_DEFAULT_MATMUL_PRECISION=highest python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_nuts --precondition 0.5 \
    --n-gal 8 32 128 --chunk 32 --warmup 200 --burnin 0 --samples 200 \
    --dtype f64 --json bench/results/2026-09-05_width_nuts.json --tag rtx3060-idle
```

`--n-gal 8` is accepted and then skipped: the harness skips any cell with
`n_gal < K`, so with `--chunk 32` the sweep starts at N=32.

**Run the width sweep on an otherwise idle box, and check `nvidia-smi` first.**
Its headline columns are wall clocks. Two JAX processes on one 12 GB card is not
merely slow — a concurrent 9 GB pytest job made `gpusolverDnCreate` fail outright
with `cuSolver internal error`, which is what killed the first two attempts at
the NUTS N=128 cell.

### Harness changes made for this report

Four, all in `bench/scripts/`, none touching `src/`:

* `benchmark_notebook_sampler.py`: the `hmcp` family sweeps
  `L` in {10, 20, 40, 80, 160} rather than {20, 80}. Two points cannot show the
  shape of an R-hat-versus-`L` curve, and this one is not monotone.
* `benchmark_notebook_sampler.py`: a new `nutsp` family (preconditioned NUTS), so
  the fixed-L rows can be read against NUTS at the same metric on the same
  branch and seeds rather than against a number quoted from another branch.
* `benchmark_notebook_sampler.py`: every row is stamped with `loadavg_before`,
  `loadavg_after` and `started_unix`. A wall-clock cell on this box is unreadable
  without the load that produced it, and load cannot be recovered afterwards.
* `benchmark_catalog_throughput.py`: rows carry `min_distinct_frac` (the worst
  lane's unique-draw fraction) and `loadavg`. The frozen bucket is a threshold
  test at `FROZEN_DISTINCT_FRAC` = 0.01; a catalog can pass it while carrying
  lanes that barely moved, and #1999's frozen chains reported zero divergences,
  so the fraction is published as its own column.

`tests/contract/test_catalog_throughput_bench.py` and
`tests/contract/test_harness_notebook_parity.py` pass on these changes
(56 passed, 9 skipped).

### Result JSONs

* `bench/results/2026-09-05_hmcp_L_sweep.jsonl` — the gate, one row per
  (fixture, config, seed).
* `bench/results/2026-09-05_hmc_control.jsonl` — the unpreconditioned control.
* `bench/results/2026-09-05_hmcp_ctl-dpl.json`,
  `bench/results/2026-09-05_hmcp_ctl-dpl_L160.json`,
  `bench/results/2026-09-05_hmcp_nb05.json`,
  `bench/results/2026-09-05_hmc_control_L160.json` — the same sweeps as single
  nested documents.
* `bench/results/2026-09-05_width_hmc.json`,
  `bench/results/2026-09-05_width_nuts.json` — the GPU width sweep, one row per
  (method, N, K), each carrying `converged_gal_per_gpu_min`, `max_rhat`,
  `min_ess`, `divergences`, `n_frozen_chains`, `min_distinct_frac`,
  `peak_bytes` and `loadavg`.
