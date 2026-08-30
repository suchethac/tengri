# GPU catalog throughput: 73 % of galaxies pass R-hat and none of them are usable

**Date:** 2026-08-30
**Verdict:** tengri's first galaxies-per-GPU-minute figure on an RTX 3060 is
**304 galaxies/GPU-minute** raw, of which **222/GPU-minute clear max split-R-hat
< 1.01 with zero divergences** (`mcmc_hmc`, float32, N = 512, K = 512, 400
warmup + 500 draws, D = 3). **Do not quote either number as a posterior rate.**
The converged galaxies have **min ESS 2.6 of 500 draws** — R-hat passes and the
chains are still unusable, so the honest count of usable posteriors per
GPU-minute is **zero** and no comparison to Zacharegkas+2025's ~1000/GPU-min is
available. Three results that are solid: float64 costs **3.6x** on the posterior
gradient at batch 2048 (1.9x at 512, ~1.25x below 128) — the 1/64 FP64-rate
hypothesis does not transfer; `forward_chunk_size` is worth **8-13x** and is the
largest lever measured; and **catalog `mcmc_nuts` did not complete a single
cell** on either device, at any tree-depth cap.
**Platform:** Linux 6.8, NVIDIA RTX 3060 12 GB (GA106, driver 580.173.02) against
an AMD Ryzen 9 5900X control (`JAX_PLATFORMS=cpu`). JAX 0.11.0 / jaxlib 0.11.0,
blackjax 1.6.2. Code version: `main` at `fe6bda468`, i.e. **after #2090**
(`DeadFitError`, `total_draws`); every row in the table was re-measured on that
version, because #2090 changes what a dead fit does and rows either side of it
are not comparable.
**Precision:** both arms measured, each proven on an output array's dtype
(`probe: float32` / `probe: float64` in the header line), not on the config flag
— see Finding 0, which is why that matters. Every float32 row runs with
**`JAX_DEFAULT_MATMUL_PRECISION=highest`** (recorded in each row's
`dtype_flags`); without it Ampere lowers float32 matmuls to TF32 and tengri's own
float32 Fisher-matrix test fails by 4.5 % on the error bars. Every row also runs
with `XLA_PYTHON_CLIENT_PREALLOCATE=false`.
**Data / model:** the benchmark's own fixture —
`data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`, five SDSS bands
(*ugriz*), **double-power-law SFH** (deliberately *not* `tsnorm`, whose
degeneracies are what PR #2027 Finding 15 blamed for its own convergence
numbers), **D = 3** free parameters (`sfh_dpl_log_total_mass`, `sfh_dpl_alpha`,
`met_logzsol`), *z* fixed at 0.1.
**SNR = 20 per band** (mock 1-sigma is 5 % of the flux; measured median 20.0).
**Approximation:** `CatalogFitter`'s default `approx="auto"`, which resolves to
`WavePrecomp()` — **`band_integration="quadrature"`, `n_subbands=5`**, verified
off the built model's `ApproxState`, not assumed. `ztable` is off (fixed *z*).
No `PrecompBiasWarning` fired at this SNR. See **Caveat 3** before quoting any
number here at a different SNR.
**Wall clocks** are the warm (second) call. This box was shared with two other
agents during the campaign — see **Caveat 4**, and note the two rows with a
*negative* compile column, where the warm call came out slower than the cold one.

## Why this was measured

`bench/scripts/benchmark_catalog_throughput.py` has existed, device-agnostic,
since the catalog path landed. It had **never been run on a GPU and had no
committed result**, and `docs/internal/getting_started/gpu.md` said so:

> GPU performance is functionally tested but not characterized with published
> wall-clocks.

The immediate prompt is Zacharegkas, Hearin & Benson (2025), *Bayesian
Posteriors with Stellar Population Synthesis on GPUs*
([arXiv:2506.19919](https://arxiv.org/abs/2506.19919)), which reports **~1000
galaxy posteriors per GPU-minute** on a 12-parameter DSPS model. tengri had no
comparable number at all. The second question is precision: a GeForce die runs
float64 at **1/64** the float32 rate, and `notebooks/12_simulation_populations.py`
prices float32 on *bandwidth* ("bytes halve, so ~1.8x") for the forward model
only, while refusing to extrapolate to inference:

> Inference is a separate question again — gradients, mass matrices and
> log-likelihood differences have their own conditioning, and nothing here
> measures any of it.

This report measures it. The 1/64 figure is a hypothesis here, not an
assumption, and it does not survive.

## Finding 0 — the float32 axis did not exist until it was proven on an array

This is first because it invalidates the obvious way to add a precision axis,
and because it is the reason every row carries a `probe_dtype`.

The harness originally did `jax.config.update("jax_enable_x64", True)` at module
scope. Moving that into `main()` and flipping it from `--dtype` **looks** like a
precision axis and is not one: `tengri/__init__.py` re-enables x64 **on import**
unless `JAX_ENABLE_X64` is present in the environment (#1840), and the import
has to happen before the model is built. The first `--dtype f32` run therefore
produced float64 arrays and reported them as float32, silently. It was caught by
a test that asserts on the output array's dtype, not on the flag —
`tests/contract/test_catalog_throughput_bench.py::test_grad_mode_honors_dtype_end_to_end`
— which is the standard `2026-08-20_cuda_device_matrix.md` set.

The switch that binds is the environment variable, applied above `import jax`.
The harness now does that from its own argv, and `set_precision()` refuses to
proceed if `jnp` does not actually allocate the requested dtype. Consequence for
anyone reusing this harness: **one precision per process.** A two-precision
sweep is two processes writing into the same `--json`.

## Finding 1 — the posterior gradient in float64 costs 3.6x, not 64x

The number the plan asked for, and the one the sampler rows cannot give on their
own: the *same* flat log-posterior the catalog sampler calls
(`backends/mcmc/catalog._get_flat_logdensity`) and its gradient, `jax.vmap`-ed
over galaxies, timed in isolation. Microseconds for the whole batch, warm,
minimum over 4 repetitions of 20 blocked calls.

| batch | log-post f64 | log-post f32 | f64/f32 | **gradient f64** | **gradient f32** | **f64/f32** |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9913.9 | 7799.5 | 1.27 | 9808.1 | 7874.3 | 1.25 |
| 8 | 9858.4 | 7964.1 | 1.24 | 9910.5 | 8051.7 | 1.23 |
| 32 | 9622.7 | 7613.8 | 1.26 | 9830.6 | 7591.1 | 1.30 |
| 64 | 9927.9 | 7930.3 | 1.25 | 10073.7 | 8188.9 | 1.23 |
| 128 | 9727.6 | 8224.3 | 1.18 | 10423.3 | 8130.3 | 1.28 |
| 512 | 11408.6 | 8107.2 | 1.41 | 15660.4 | 8362.4 | **1.87** |
| 2048 | 21712.8 | 9098.5 | 2.39 | 37976.8 | 10615.1 | **3.58** |

Per galaxy, the same gradient column: 9808 us at batch 1 down to **18.5 us
(f64) and 5.2 us (f32)** at batch 2048.

**The 1/64 hypothesis does not survive.** GA106 runs float64 at 1/64 the float32
rate, and the plan's expectation was that this would make float32 inference far
better than the forward model's ~1.8x bandwidth framing. The measured gradient
ratio is **3.58x at batch 2048 and 1.87x at batch 512** — above the bandwidth
estimate, and nowhere near 64x. A pre-merge run of the identical sweep gave 3.60x
and 1.97x, so the large-batch figure reproduces to ~5 %.

The reason is in the shape of the curve, not in the ratio. Up to batch 128 both
precisions sit on a **flat floor of ~8-10 ms**: 128x the work for 1.06x the time.
That floor is kernel launch, dispatch and (here) on-demand allocation, not
arithmetic, and a 1/64 FP64 *ALU* rate is invisible to a kernel that is not
ALU-bound. The ratio only climbs where the work exceeds the floor — 512, then
2048 — and even there it tracks bytes moved far more closely than the FLOP rate.
The forward model runs at ~0.12 FLOP/byte (`2026-08-20_cuda_device_matrix.md`)
and its gradient inherits that.

So: **quote 3.6x, and only at batch 2048.** Below ~128 galaxies float32 buys
essentially nothing on this card, because there is nothing to buy.

Two subsidiary results worth stating separately:

- **float64 hurts the gradient more than it hurts the log-posterior** (3.58x vs
  2.39x at batch 2048). The reverse pass moves more state per galaxy, so
  doubling the width of every array costs it more — again a bytes story.
- **The posterior gradient in float32 is not zero.** `gpu.md` records that
  `jax.grad` of a raw observable (`sum(predict_photometry)`) returns identically
  zero in float32. That defect does not reach here: the log-posterior
  standardizes by sigma before squaring, and every row above has
  `grad_all_zero = false` and `grad_finite = true` in the JSON. Checked, not
  assumed — the harness records both flags per row.

**Method caveat on the small-batch column.** These rows ran with
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, which adds per-call allocator work. The
pre-merge sweep with preallocation on put the float64 floor at ~7.4 ms rather
than ~9.9 ms, so the sub-128 ratios here (~1.25) are inflated relative to the
~1.05 measured with preallocation on. The 512 and 2048 ratios are unaffected
(1.97 / 3.60 with preallocation, 1.87 / 3.58 without) because at those sizes the
kernel, not the allocator, dominates.

## Finding 2 — throughput, with the R-hat *and* the ESS column attached

`CatalogFitter.run("mcmc_hmc", forward_chunk_size=K)`, one chain per galaxy, MAP
warm start (`DEFAULT_MAP_INIT_STEPS = 300` ADAM steps, inside the vmap), diagonal
mass matrix, `n_leapfrog_steps=10` (the shipped default), `PRNGKey(0)`, 400
warmup + 500 draws. `warm` is the second identical call.

Every galaxy is placed in exactly one of three buckets — **converged** (its own
max split-R-hat < 1.01 and no divergence), **frozen** (every draw identical, so
`Posterior.rhat()` raises), **unconverged** (moved, did not mix) — because two of
the three are invisible in a wall clock and one of them is invisible to R-hat.

| device | dtype | N | K | cold s | warm s | compile s | gal/min | **converged gal/min** | **converged / N** | frozen | max split-R-hat | **min ESS** | min ESS (converged only) | div | peak VRAM |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GPU | f32 | 512 | 512 | 108.18 | **101.03** | 7.15 | **304** | **222** | 374/512 (73 %) | 0 | 1.1421 | **2.63** | 2.63 | 0 | 131 MB |
| GPU | f64 | 512 | 512 | 158.53 | 152.39 | 6.13 | 202 | 138 | 350/512 (68 %) | 0 | 1.1392 | 2.21 | 2.21 | 0 | 258 MB |
| GPU | f32 | 64 | 32 | 183.81 | 168.79 | 15.02 | 23 | 16 | 46/64 (72 %) | 0 | 1.1244 | 2.97 | 3.27 | 0 | 96 MB |
| GPU | f64 | 64 | 32 | 181.21 | 155.30 | 25.91 | 25 | 16 | 42/64 (66 %) | 0 | 1.1300 | 2.82 | 11.19 | 0 | 111 MB |
| **CPU** | f64 | 64 | 32 | 98.94 | **105.28** | -6.33 | 36 | 30 | 53/64 (83 %) | 0 | 1.0756 | 2.75 | 7.66 | 0 | n/a |

### The R-hat column is a trap without the ESS column

`bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md` established that *"the s/ESS
column is a trap without the R-hat column"*. This campaign found the converse,
and it is the most important line in the report:

> **73 % of galaxies pass max split-R-hat < 1.01 with zero divergences, and the
> minimum ESS among exactly those galaxies is 2.63 of 500 draws.**

That is the `min ESS (converged only)` column, and it exists because the
catalog-wide minimum would otherwise be attributable to the unconverged tail. It
is not: on the headline row the converged subset's worst ESS equals the
catalog's. An effective sample size of 2.6 from 500 draws is an autocorrelation
time of ~190 — a chain that has taken about two independent steps. Split-R-hat
over two halves of 250 such draws reads 1.00 because both halves are equally
badly mixed; it compares variances, and the variance is right even when the chain
is not.

So `converged gal/min = 222` is the rate at which this configuration produces
galaxies that **pass half the bar**. The number of galaxies per GPU-minute for
which tengri produced a usable posterior in this campaign is **zero**, and that
is the honest answer to "what is tengri's galaxies-per-GPU-minute figure". The
raw 304 and the R-hat-passing 222 are both reported so nobody has to re-derive
them, and both are labelled.

The failure is not the fixture's degeneracy. This is a `dpl` SFH at D = 3, chosen
specifically because #2027's Finding 15 attributed its own convergence numbers to
`tsnorm`'s degeneracies. The same failure appears here on a non-degenerate
three-parameter model, which means it belongs to **`mcmc_hmc` at its shipped
`n_leapfrog_steps=10`**, not to the SFH family. #2027's Finding 13 swept L on the
other fixture and found L = 150 the first setting to approach the bar; L was not
swept here (a single L = 150 cell at K = 512 was queued and cut for time), so
this report cannot say whether a longer trajectory fixes it on `dpl` too. That is
the obvious next measurement.

### Two things #2090 makes newly visible, and both came out clean

- **Zero frozen chains, everywhere.** #2027 Finding 14 measured `mcmc_nuts`
  returning a completely frozen chain for **3.1 %** of galaxies with zero
  divergences, and noted a catalog fit has no aggregate convergence gate. On this
  fixture that mode does not appear at all: 0 of 1216 galaxies across five cells.
- **Zero divergences, and the rate is now computed correctly.** #2087 established
  that `n_divergent` is summed over chains while `n_samples` is per chain, so a
  hand-rolled rate over-reports by the chain count. The harness uses
  `total_draws()`; the catalog path runs one chain per galaxy so the two agree
  here, and the rate is exactly 0.0 on every row. **A zero divergence count is
  not evidence of convergence** — every row above has both.
- **No cell was refused.** #2090's `DeadFitError` fires when the final warmup
  window is >= 90 % divergent; nothing here came close. The harness catches it
  and records a `refused` row rather than aborting the sweep. Note the
  catalog-vectorized path cannot raise per *galaxy* — `run_one` lives inside
  `lax.map`, where a Python raise is not expressible — so a refusal fails the
  whole cell, and `refused` is a property of the row rather than a per-galaxy
  count.

### K is the throughput knob, and it is worth 8-13x

| dtype | K = 32 (N = 64) | K = 512 (N = 512) | ratio |
|---|---:|---:|---:|
| f64 | 25 gal/min | 202 | **8.1x** |
| f32 | 23 | 304 | **13.2x** |

16x the chunk width buys 8.1x (f64) and 13.2x (f32), which is the same story
Finding 1 tells in isolation: per-step cost is nearly flat in K until the batch
outgrows the dispatch floor, so widening the chunk is almost free. **Use the
largest K your VRAM allows.** It is a larger lever than precision.

Precision itself is worth **1.51x** end-to-end at K = 512 (152.39 s to 101.03 s)
and nothing at K = 32. A pre-merge run of the same pair gave 1.84x; the spread
between the two is contention, not code (Caveat 4).

### VRAM does not saturate — the question has no answer in this range

The plan asked for "the K at which VRAM saturates". On this fixture there is no
such K below 512: the allocator high-water mark is **96-258 MB** across every
cell, on a 12 GB card, and it tracks the model and catalog more than K.

What *is* binding is XLA's default **75 % preallocation**: a tengri GPU process
reserves ~9.1 GB of the 12 GB up front, so a second concurrent process OOMs
immediately (`Failed to allocate device memory of 8.71GiB`, hit twice during this
campaign before the runs were serialized). The practical rule on this card is one
tengri GPU process at a time, or `XLA_PYTHON_CLIENT_PREALLOCATE=false` /
`XLA_PYTHON_CLIENT_MEM_FRACTION` set explicitly — which is what every row above
uses.

### The CPU wins the narrow cell

Same cell, same seed, same budget, `JAX_PLATFORMS=cpu` on a Ryzen 9 5900X:
**105.28 s against the GPU's 155.30 s, a 1.48x CPU win at K = 32**, with a higher
converged fraction (83 % vs 66 %). That is the same conclusion
`2026-08-20_cuda_device_matrix.md` reached from the other direction: the GPU is a
width instrument. At K = 512 the GPU does win, at 304 gal/min against the CPU's
36 at K = 32 — not a matched pair, but the direction is unambiguous.

## Finding 3 — catalog NUTS did not complete a single cell, on either device

**No `mcmc_nuts` cell in this campaign produced a row.** Every attempt was killed
by its timeout, on the GPU and on the CPU.

| device | cell | tree-depth cap | leapfrogs/step | warmup+draws | timeout | outcome |
|---|---|---:|---:|---|---:|---|
| GPU | N = 512, K = 512, f64 | 5 | <= 31 | 400+500 | 2400 s | timed out |
| GPU | N = 32, K = 32, f64 | 4 | <= 15 | 400+500 | 1500 s | timed out |
| GPU | N = 32, K = 32, f64 | **2** | **<= 3** | 400+500 | 600 s | timed out |
| GPU | N = 32, K = 32, f64 | 4 | <= 15 | 100+200 | 900 s | timed out |
| **CPU** | N = 32, K = 32, f64 | 4 | <= 15 | 100+200 | 900 s | timed out |
| GPU | **N = 8, K = 8**, f64 | **3** | **<= 7** | **50+100** | 600 s | timed out |

Read the fourth row against an HMC cell of the identical shape and budget:
**HMC at 10 leapfrogs per step finished N = 32, K = 32, 100+200 in 30 s.** NUTS
capped to at most 15 leapfrogs per step did not finish it in 900 s, and NUTS
capped to at most **three** did not finish the longer budget in 600 s. So:

1. **The tree-depth cap is not the cost driver.** Cap 2 (<= 3 leapfrogs) is
   *cheaper* in leapfrogs than the HMC row it loses to by more than 20x. The
   obvious hypothesis — that vmapped NUTS is expensive because the batch pays the
   deepest tree any chain asks for, so capping the tree fixes it — is **wrong**,
   and it was worth measuring rather than assuming.
2. **It is not a GPU dispatch artifact.** The CPU cell fails the same way.
3. **It is not the batch width or the step count either.** The last row is the
   smallest NUTS cell this harness can express — 8 galaxies, 8 wide, 50 warmup
   plus 100 draws, at most 7 leapfrogs per step — and it also failed to finish in
   600 s. Everything that scales with the problem was turned down by one to two
   orders of magnitude and the cell still did not complete, which points at a
   **fixed cost per NUTS build**, most plausibly compile.
4. **That last step was not isolated.** Because no row ever printed, the cold
   (compile) and warm (sampling) halves were never separated, so this campaign
   cannot prove it is compile. Timing `build_catalog_mcmc_engine` for `nuts`
   against `hmc` with `TENGRI_DISABLE_JAX_CACHE=1`, at one galaxy, is the obvious
   next measurement. It is left undone rather than guessed at.

This corroborates and sharpens `2026-08-20_cuda_device_matrix.md` Finding 14,
which measured catalog NUTS at **26x more expensive per galaxy and 84x per
iteration** than HMC on a different fixture (`tsnorm`, D = 7). This one is `dpl`
at D = 3. Two fixtures, two SFH families, two devices, the same ratio: Finding 15
was careful to scope its *convergence* claims to its fixture, and this report can
now say the **cost** ratio generalizes.

**The consequence for the plan is direct.** `mcmc_nuts` is `tier="primary"` and
is the sampler the notebooks rely on to converge — and at catalog scale it is the
one you cannot run. `mcmc_hmc` runs and does not mix. That is precisely the gap
the lock-step samplers (ChEES-HMC, GHMC+MEADS, MCLMC) and FSM-NUTS are meant to
close, and this is the measurement that says they are needed rather than merely
interesting.

## Finding 4 — the sharded path works, and does not change the posterior

This box has one CUDA device, so multi-GPU scaling is not measurable here. What
*is* measurable is that `run(devices="all")` is wired and correct, on four
emulated CPU devices (`XLA_FLAGS=--xla_force_host_platform_device_count=4`),
`mcmc_hmc`, N = 32, K = 8, 50 warmup + 100 draws:

| | wall | galaxies/s | max split-R-hat |
|---|---:|---:|---:|
| single device | 30.57 s | 1.0 | 2.6443641 |
| `devices="all"` (4) | **12.68 s** | 2.5 | 2.6443491 |

**2.41x on 4 devices** (60 % of linear, on emulated devices sharing one socket),
and the two R-hats agree to six decimal places — the shard produces the same
posterior, not merely a faster one. Both are far outside the bar; the point of
this cell is the seam, not the sampler.

This also corrects a documentation claim that was already stale before this
campaign: `docs/internal/getting_started/gpu.md` said "Multi-GPU sharding via
`jax.pmap` or `shard_map` is not yet wired" until PR #2027 replaced it.
`CatalogFitter._sharded_vmap` and `_resolve_devices` map the galaxy axis over a
`Mesh` via GSPMD for `mcmc_nuts` and `mcmc_hmc`; `shard_map` specifically is not
used because BlackJAX's NUTS carries a `lax.cond` that trips manual varying-axis
tracking.

## Caveats

**Caveat 1 — D = 3 is not the paper's D = 12.** The fixture is the benchmark's
own: a double-power-law SFH with mass, slope and metallicity free, everything
else pinned, five broadbands. The paper fits twelve parameters. A
galaxies-per-GPU-minute number from a 3-parameter posterior is not comparable to
one from a 12-parameter posterior on cost *or* on difficulty. The fixture was
kept deliberately non-degenerate (not `tsnorm`) so the convergence failure could
not be blamed on the SFH family, which is the one thing it does buy.

**Caveat 2 — one chain per galaxy.** `CatalogFitter._run_native_mcmc` gives each
galaxy a single chain, so "split R-hat" here is the within-chain split
diagnostic, not the between-chain one. It is what `Posterior.rhat()` exposes and
what the 2026-08-17 reports used, but it is weaker than four chains from
dispersed inits and cannot see a chain stuck in the wrong mode. Finding 2's ESS
result is the concrete cost of that weakness.

**Caveat 3 — every number here is at SNR 20 under a quadrature LUT.** The fit
runs on `WavePrecomp(band_integration="quadrature", n_subbands=5)`, whose forward
bias is constant in SNR but enters the posterior gradient **multiplied** by SNR
(#1671; ~5 % relative gradient error at SNR 30, ~50 % at SNR 300). At this mock's
SNR 20 the runtime estimator (`_warn_if_lut_bias_amplified`) stayed below its
warning threshold and no `PrecompBiasWarning` fired, which is why these numbers
are quotable at all. **They are not quotable at higher SNR**: a deeper catalog
would want `approx=None`, a different and slower forward model, so the wall
clocks would not carry over either. The harness records the median SNR and the
resolved `band_integration` in every JSON row so this cannot be lost.

**Caveat 4 — this box was shared.** The GPU was also driving the desktop, and two
other agents ran on the same host during the campaign. Two rows have a *negative*
compile column — the warm call came out slower than the cold one — which is the
signature of that contention, and the same f32 N = 512 cell measured 83.47 s in
one pass and 101.03 s in another (21 %). **Treat wall clocks as indicative to
~20 %, and ratios between rows measured in the same pass as better than that.**
R-hat, ESS, the convergence counts and the divergence counts are deterministic
given the seed and are not affected.

**Caveat 5 — the cold column is a cache load, not a compile.** `import tengri`
enables JAX's persistent compilation cache by default. Set
`TENGRI_DISABLE_JAX_CACHE=1` to measure a real first compile. Every JSON row
records `jax_persistent_cache` so the two cannot be confused.

## What was NOT measured, and why

- **`mcmc_hmc` at L != 10.** The single lever #2027 Finding 13 identified as
  moving the needle (L = 150 took max R-hat from 1.93 to 1.037 on its fixture)
  was queued as one cell at N = 512, K = 512, float32 and cut at ~80 % complete
  for time. It is the highest-value next cell in this harness.
- **N = 2048.** Queued three times at K = 512 with 1500 s, 2700 s and 3600 s
  timeouts and killed by all of them. **N was reduced to 512**, which is the
  reduction the plan asks for first, and this is that reduction said out loud
  rather than a quiet drop. The rate should not move much — N = 2048 at K = 512
  is four `lax.map` iterations of the N = 512 cell — but the wall time disagreed
  with that model by more than 2x, and the un-modelled part is per-galaxy
  **host-side** work: `CatalogFitter` builds one `Posterior` object and one
  summary block per galaxy, in Python, after the device work finishes. That O(N)
  host cost is real and is not in the 304 figure. Measure it before quoting a
  rate at N in the thousands.
- **`K = 1` and `K = 128`.** Both queued, both cut for time. At K = 1 the cost is
  exactly N sequential `lax.map` iterations, so galaxies/second there is
  independent of N and one small-N cell would settle it.
- **Multi-GPU sharding.** One CUDA device on this box; Finding 4 is the
  CPU-emulated substitute.
- **`approx=None` (the exact projector).** A different forward model; its own
  sweep.

## Reproduce

All commands from the repo root, on `main` at or after `fe6bda468` (#2090).
Precision is process-global and is chosen **before** `import jax`, so one
`--dtype` per process; the script does that translation from its own argv and
refuses to run if `jnp` does not then allocate the dtype asked for. Results merge
into one JSON keyed on the configuration. Run the cells **one at a time** — two
concurrent tengri GPU processes OOM this card.

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# 1. the precision question, in isolation (Finding 1)
python bench/scripts/benchmark_catalog_throughput.py --mode grad --dtype f64 \
    --n-gal 1 8 32 64 128 512 2048 --reps 4 --runs 20 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060
python bench/scripts/benchmark_catalog_throughput.py --mode grad --dtype f32 \
    --n-gal 1 8 32 64 128 512 2048 --reps 4 --runs 20 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060

# 2. the headline throughput cell, and its float64 arm (Finding 2)
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --dtype f32 --n-gal 512 --chunk 512 \
    --warmup 400 --burnin 0 --samples 500 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --dtype f64 --n-gal 512 --chunk 512 \
    --warmup 400 --burnin 0 --samples 500 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060

# 3. the narrow-K cells, both precisions
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --dtype f32 --n-gal 64 --chunk 32 \
    --warmup 400 --burnin 0 --samples 500 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --dtype f64 --n-gal 64 --chunk 32 \
    --warmup 400 --burnin 0 --samples 500 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060

# 4. the CPU control (same cell, same seed, same budget)
JAX_PLATFORMS=cpu python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --dtype f64 --n-gal 64 --chunk 32 \
    --warmup 400 --burnin 0 --samples 500 \
    --json bench/results/gpu_catalog_throughput.json --tag ryzen9-5900x

# 5. the NUTS cells that time out (Finding 3). Expect exit 124, not a row.
timeout 600 python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_nuts --dtype f64 --n-gal 32 --chunk 32 --max-doublings 2 \
    --warmup 400 --burnin 0 --samples 500 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060
JAX_PLATFORMS=cpu timeout 900 python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_nuts --dtype f64 --n-gal 32 --chunk 32 --max-doublings 4 \
    --warmup 100 --burnin 0 --samples 200 \
    --json bench/results/gpu_catalog_throughput.json --tag ryzen9-5900x

# 6. exercise the sharded path without a second GPU (4 emulated CPU devices)
XLA_FLAGS=--xla_force_host_platform_device_count=4 JAX_PLATFORMS=cpu \
    python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --n-gal 32 --chunk 8 --shard \
    --warmup 50 --burnin 0 --samples 100

# 7. the cell this report did NOT finish, and wants finished next
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --dtype f32 --n-gal 512 --chunk 512 --n-leapfrog 150 \
    --warmup 400 --burnin 0 --samples 500 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060

# 8. to measure a real first compile rather than a persistent-cache load
TENGRI_DISABLE_JAX_CACHE=1 python bench/scripts/benchmark_catalog_throughput.py ...
```

Or via the dispatcher, which forwards every flag:

```bash
python -m tengri.bench catalog_throughput --method mcmc_hmc --dtype f32 \
    --n-gal 512 --chunk 512 --warmup 400 --burnin 0 --samples 500
```

The harness's own guarantees are pinned by
`tests/contract/test_catalog_throughput_bench.py`:

```bash
python -m pytest tests/contract/test_catalog_throughput_bench.py \
    tests/contract/test_bench_cli.py -v -m "slow or not slow" -n 0
```
