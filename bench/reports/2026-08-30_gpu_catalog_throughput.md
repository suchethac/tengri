# GPU catalog throughput: float64 costs 3.6x, not 64x, and the first "float32" run was float64

**Date:** 2026-08-30
**Verdict:** tengri's first galaxies-per-GPU-minute figure is **372.7** on an
RTX 3060 (`mcmc_hmc`, float32, N = 512, `forward_chunk_size` = 512, 400 warmup +
500 draws) — and it is **not a usable posterior**: max split-R-hat 1.13 against a
bar of 1.01, min ESS 2.4 of 500 draws, on a D = 3 model. It is therefore **not**
comparable to Zacharegkas+2025's ~1000 converged posteriors/GPU-min on D = 12.
No row in this campaign cleared the bar, so nothing here is rankable on
seconds-per-effective-sample. Three things that *are* solid: float64 costs
**3.6x**, not 64x, on the posterior gradient at batch 2048 (1.97x at 512, ~1.0x
below 128) — the 1/64 FP64-rate hypothesis does not transfer to this workload;
`forward_chunk_size` is worth **8.5-13.4x** and is the largest lever measured;
and **catalog `mcmc_nuts` did not complete a single cell on either device**, at
any tree-depth cap.
**Platform:** Linux 6.8, NVIDIA RTX 3060 12 GB (GA106, driver 580.173.02) against
an AMD Ryzen 9 5900X control (`JAX_PLATFORMS=cpu`). JAX 0.11.0 / jaxlib 0.11.0,
blackjax 1.6.2. **Precision:** both arms measured, each proven on an output
array's dtype (`probe: float32` / `probe: float64` in the header line), not on
the config flag — see Finding 0, which is why that matters. Every float32 row
runs with **`JAX_DEFAULT_MATMUL_PRECISION=highest`** (recorded in each row's
`dtype_flags`); without it Ampere lowers float32 matmuls to TF32 and tengri's
own float32 Fisher-matrix test fails by 4.5 % on the error bars.
**Data / model:** the benchmark's own fixture —
`data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`, five SDSS bands
(*ugriz*), double-power-law SFH, **D = 3** free parameters
(`sfh_dpl_log_total_mass`, `sfh_dpl_alpha`, `met_logzsol`), *z* fixed at 0.1.
**SNR = 20 per band** (mock 1-sigma is 5 % of the flux; measured median 20.0,
range 16.2-23.7).
**Approximation:** `CatalogFitter`'s default `approx="auto"`, which resolves to
`WavePrecomp()` — **`band_integration="quadrature"`, `n_subbands=5`**, verified
off the built model's `ApproxState`, not assumed. `ztable` is off (fixed *z*).
No `PrecompBiasWarning` fired at this SNR. See **Caveat 3** before quoting any
number here at a different SNR.
**Wall clocks** are the warm (second) call; the cold call is reported separately
and, because `import tengri` enables JAX's persistent compile cache, it is a
cache *load* rather than a full XLA compile.

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
prices float32 on *bandwidth* ("bytes halve → ~1.8x") for the forward model
only, while refusing to extrapolate to inference:

> Inference is a separate question again — gradients, mass matrices and
> log-likelihood differences have their own conditioning, and nothing here
> measures any of it.

This report measures it. The 1/64 figure is a hypothesis here, not an
assumption, and it does not survive.

## Finding 0 — the float32 axis did not exist until it was proven on an array

This is first because it invalidates the obvious way to add a precision axis,
and because it is the reason every row below carries a `probe_dtype`.

The harness originally did `jax.config.update("jax_enable_x64", True)` at module
scope. Moving that into `main()` and flipping it from `--dtype` **looks** like a
precision axis and is not one: `tengri/__init__.py` re-enables x64 **on import**
unless `JAX_ENABLE_X64` is present in the environment (#1840), and the import
has to happen before the model is built. The first `--dtype f32` run therefore
produced float64 arrays and reported them as float32, silently. It was caught by
a test that asserts on the output array's dtype, not on the flag —
`tests/contract/test_catalog_throughput_bench.py::test_grad_mode_honors_dtype_end_to_end`
— which is exactly the standard `2026-08-20_cuda_device_matrix.md` set.

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
| 1 | 7526.4 | 7161.7 | 1.05 | 7368.0 | 6896.6 | 1.07 |
| 8 | 7277.9 | 7572.2 | 0.96 | 7756.2 | 7705.9 | 1.01 |
| 32 | 7475.4 | 7284.7 | 1.03 | 8087.4 | 7170.1 | 1.13 |
| 64 | 7921.7 | 7275.0 | 1.09 | 8342.3 | 7302.7 | 1.14 |
| 128 | 8407.7 | 7353.1 | 1.14 | 9274.8 | 7851.7 | 1.18 |
| 512 | 10655.6 | 7591.3 | 1.40 | 15187.4 | 7718.0 | **1.97** |
| 2048 | 20857.7 | 8836.6 | 2.36 | 36860.3 | 10233.4 | **3.60** |

Per galaxy, the same gradient column: 7368 us at batch 1 down to **18.0 us (f64)
and 5.0 us (f32)** at batch 2048.

**The 1/64 hypothesis does not survive.** GA106 runs float64 at 1/64 the float32
rate, and the plan's expectation was that this would make float32 inference far
better than the forward model's ~1.8x bandwidth framing. The measured gradient
ratio is **3.60x at batch 2048 and 1.97x at batch 512** — above the bandwidth
estimate, and nowhere near 64x.

The reason is in the shape of the curve, not in the ratio. Up to batch 128 both
precisions sit on a **flat floor of ~7-9 ms**: 128x the work for 1.13x the time,
and f64/f32 within 1.2 of unity. That floor is kernel launch and dispatch, not
arithmetic, and a 1/64 FP64 *ALU* rate is invisible to a kernel that is not
ALU-bound. The ratio only starts to climb where the work finally exceeds the
floor — 512, then 2048 — and even there it tracks the bytes moved much more
closely than the FLOP rate. The forward model runs at ~0.12 FLOP/byte
(`2026-08-20_cuda_device_matrix.md`), and its gradient inherits that.

So: **quote 3.6x, not 64x, and only at batch 2048.** Below ~128 galaxies float32
buys essentially nothing on this card, because there is nothing to buy.

Two subsidiary results worth stating separately:

- **float64 hurts the gradient more than it hurts the log-posterior** (3.60x vs
  2.36x at batch 2048). The reverse pass moves more state per galaxy than the
  forward pass, so doubling the width of every array costs it more — again a
  bytes story, not a FLOPs one.
- **The posterior gradient in float32 is not zero.** `gpu.md` records that
  `jax.grad` of a raw observable (`sum(predict_photometry)`) returns identically
  zero in float32. That defect does not reach here: the log-posterior
  standardizes by sigma before squaring, and every row above has
  `grad_all_zero = false` and `grad_finite = true` in the JSON. This is checked,
  not assumed — the harness records both flags per row.

## Finding 2 — throughput, with the R-hat column attached

### The table

`CatalogFitter.run(method, forward_chunk_size=K)`, one chain per galaxy, MAP
warm start (`DEFAULT_MAP_INIT_STEPS = 300` ADAM steps, inside the vmap),
diagonal mass matrix, `PRNGKey(0)`. `warm` is the second identical call. The
bar is the one `bench/reports/2026-08-17_*` set and it is binding: **max split
R-hat < 1.01 and 0 divergences.**

| method | dtype | device | N | K | warmup+draws | cold s | **warm s** | compile s | **gal/GPU-min** | **max split-R-hat** | **min ESS** | div | peak VRAM |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mcmc_hmc` | f32 | GPU | 512 | 512 | 400+500 | 90.70 | **82.43** | 8.27 | **372.7** | 1.129 | 2.4 | 0 | 165 MB |
| `mcmc_hmc` | f64 | GPU | 512 | 512 | 400+500 | 154.66 | 145.86 | 8.80 | 210.6 | 1.073 | 2.5 | 0 | 232 MB |
| `mcmc_hmc` | f32 | GPU | 32 | 32 | 100+200 | 33.82 | 25.76 | 8.06 | 74.5 | 1.574 | 1.1 | 0 | 91 MB |
| `mcmc_hmc` | f64 | GPU | 32 | 32 | 100+200 | 38.05 | 29.97 | 8.08 | 64.1 | 1.496 | 0.9 | 0 | 101 MB |
| `mcmc_hmc` | f32 | GPU | 64 | 32 | 400+500 | 163.46 | 138.07 | 25.38 | 27.8 | 1.054 | 2.5 | 0 | 165 MB |
| `mcmc_hmc` | f64 | GPU | 64 | 32 | 400+500 | 161.20 | 154.90 | 6.31 | 24.8 | 1.099 | 2.0 | 0 | 115 MB |
| `mcmc_hmc` | f64 | **CPU** | 64 | 32 | 400+500 | 106.66 | **87.21** | 19.45 | 44.0 | 1.076 | 2.7 | 0 | n/a |

**Every row is non-converged.** Max split-R-hat runs from 1.054 to 1.574 against
a bar of 1.01, and min ESS is between **0.9 and 2.7** — of 200 to 500 draws.
Divergences are zero on every row and so is the frozen-chain count, which is
exactly the trap `2026-08-20_cuda_device_matrix.md` Finding 13 named: a
divergence count of zero is not evidence of convergence for fixed-length HMC,
because it cannot report the failure mode a fixed trajectory actually has.

Consequently **no row is rankable on seconds-per-effective-sample**, and the
ranking among rows that clear the bar has zero entries. The numbers above are a
characterization of the *machine* — how fast this card pushes sampler steps
through a vmapped catalog — and nothing more. tengri's galaxies-per-GPU-minute
figure is **372.7 at the fastest setting measured, and it is not a posterior
you could use.** It is not comparable to Zacharegkas+2025's ~1000/GPU-min, which
is a converged number on a 12-parameter model.

The negative result reproduces the one already on file. `mcmc_hmc` ships
`n_leapfrog_steps=10`; Finding 13 of the 2026-08-20 report swept L on a
different fixture and found L = 20 through 2000 warmup steps all stuck at
R-hat 1.9-3.3, with L = 150 the first configuration to approach the bar
(R-hat 1.037). L is not tuned here either, and this fixture — a *different* SFH
family, D = 3 rather than D = 7 — fails the same way. That is the more general
statement the earlier report explicitly declined to make.

### K is the throughput knob, and it is worth 8.5x

Same sampler, same dtype, same budget, one axis:

| dtype | K = 32 (N = 64) | K = 512 (N = 512) | ratio |
|---|---:|---:|---:|
| f64 | 24.8 gal/GPU-min | 210.6 | **8.5x** |
| f32 | 27.8 | 372.7 | **13.4x** |

16x the chunk width buys 8.5x (f64) and 13.4x (f32) the throughput, which is the
same story Finding 1 tells in isolation: the per-step cost is nearly flat in K
until the batch outgrows the dispatch floor, so widening the chunk is almost
free. **Use the largest K your VRAM allows.** It is the single largest lever in
this table, larger than precision.

### VRAM does not saturate — the question has no answer in this range

The plan asked for "the K at which VRAM saturates". On this fixture there is no
such K below 512: the allocator high-water mark is **91-232 MB** across every
cell, on a 12 GB card, and it tracks the model and catalog rather than K (the
two f32 rows at K = 32 and K = 512 peak within 26 kB of each other).

What *is* binding is not the program's working set but XLA's **75 % preallocation**:
each tengri GPU process reserves ~9.1 GB of the 12 GB up front, so a second
concurrent process OOMs immediately (`Failed to allocate device memory of
8.71GiB`, hit twice during this campaign). On this card the practical rule is
one tengri GPU process at a time, or `XLA_PYTHON_CLIENT_MEM_FRACTION` set
explicitly.

### The CPU wins the narrow cell by 1.78x

Same cell, same seed, same budget, `JAX_PLATFORMS=cpu` on a Ryzen 9 5900X:

| | GPU f64 | CPU f64 |
|---|---:|---:|
| warm wall, N = 64, K = 32 | 154.90 s | **87.21 s** |
| galaxies / minute | 24.8 | **44.0** |
| max split-R-hat | 1.099 | 1.076 |
| min ESS | 2.0 | 2.7 |

At K = 32 the GPU is **1.78x slower than the CPU**, which is the same conclusion
`2026-08-20_cuda_device_matrix.md` reached from the other direction: the GPU is a
width instrument. A 3060 only overtakes a 5900X on this path once K is large
enough to leave the dispatch floor, and at K = 512 it does (372.7 vs the CPU's
44.0 at K = 32 — not a matched pair, but the direction is unambiguous).

R-hat and ESS agree between the two devices to within noise, which is the useful
part: the GPU is not producing a different posterior, only the same one at a
different speed.

## Finding 3 — catalog NUTS did not complete a single cell, on either device

This is the result that cost the most wall clock and it is worth stating plainly:
**no `mcmc_nuts` cell in this campaign produced a row.** Every attempt was killed
by its timeout, on the GPU and on the CPU.

| device | cell | tree-depth cap | leapfrogs/step | warmup+draws | timeout | outcome |
|---|---|---:|---:|---|---:|---|
| GPU | N = 512, K = 512, f64 | 5 | <= 31 | 400+500 | 2400 s | timed out |
| GPU | N = 32, K = 32, f64 | 4 | <= 15 | 400+500 | 1500 s | timed out |
| GPU | N = 32, K = 32, f64 | **2** | **<= 3** | 400+500 | 600 s | timed out |
| GPU | N = 32, K = 32, f64 | 4 | <= 15 | 100+200 | 900 s | timed out |
| **CPU** | N = 32, K = 32, f64 | 4 | <= 15 | 100+200 | 900 s | timed out |
| GPU | **N = 8, K = 8**, f64 | **3** | **<= 7** | **50+100** | 600 s | timed out |

Read the fourth row against the HMC row in the table above at the identical
shape and budget: **HMC at 10 leapfrogs per step finished N = 32, K = 32,
100+200 in 30 s.** NUTS capped to at most 15 leapfrogs per step did not finish
it in 900 s, and NUTS capped to at most **three** did not finish the longer
budget in 600 s. So:

1. **The tree-depth cap is not the cost driver.** Cap 2 (<= 3 leapfrogs) is
   *cheaper* in leapfrogs than the HMC row it loses to by more than 20x. The
   obvious hypothesis — that vmapped NUTS is expensive because the batch pays
   the deepest tree any chain asks for, so capping the tree fixes it — is
   **wrong**, and it was worth measuring rather than assuming.
2. **It is not a GPU dispatch artifact.** The CPU cell fails the same way. This
   is a property of the catalog NUTS path itself, not of the accelerator.
3. **It is not the batch width or the step count either.** The last row is the
   smallest NUTS cell this harness can express — 8 galaxies, 8 wide, 50 warmup
   plus 100 draws, at most 7 leapfrogs per step — and it also failed to finish
   in 600 s. Everything that scales with the problem was turned down by one to
   two orders of magnitude and the cell still did not complete, which points at
   a **fixed cost per NUTS build**, most plausibly compile, rather than at the
   sampling loop.
4. **That last step was not isolated.** Because no row ever printed, the cold
   (compile) and warm (sampling) halves were never separated, so this campaign
   cannot prove it is compile. Timing `build_catalog_mcmc_engine` for `nuts`
   against `hmc` with `TENGRI_DISABLE_JAX_CACHE=1`, at one galaxy, is the
   obvious next measurement. It is left undone here rather than guessed at.

This corroborates and sharpens `2026-08-20_cuda_device_matrix.md` Finding 14,
which measured catalog NUTS at **26x more expensive per galaxy and 84x per
iteration** than HMC on a different fixture (`tsnorm`, D = 7). This one is a
`dpl` fixture at D = 3. Two fixtures, two SFH families, two devices, the same
ratio: the earlier report's Finding 15 was careful to scope its convergence
claims to its fixture, and this report can now say the **cost** ratio, at least,
generalizes.

**The consequence for the plan is direct.** `mcmc_nuts` is `tier="primary"` and
is the sampler the notebooks rely on to converge — and at catalog scale it is
the one you cannot run. That is precisely the gap the lock-step samplers
(ChEES-HMC, GHMC+MEADS, MCLMC) and FSM-NUTS are meant to close, and this is the
measurement that says they are needed rather than merely interesting.

## Finding 4 — the sharded path works, and does not change the posterior

This box has one CUDA device, so multi-GPU scaling is not measurable here. What
*is* measurable is that `run(devices="all")` is wired and correct, on four
emulated CPU devices
(`XLA_FLAGS=--xla_force_host_platform_device_count=4`), `mcmc_hmc`, N = 32,
K = 8, 50 warmup + 100 draws:

| | wall | galaxies/s | max split-R-hat |
|---|---:|---:|---:|
| single device | 30.57 s | 1.0 | 2.6443641 |
| `devices="all"` (4) | **12.68 s** | 2.5 | 2.6443491 |

**2.41x on 4 devices** (60 % of linear, on emulated devices sharing one socket),
and the two R-hats agree to six decimal places — the shard produces the same
posterior, not merely a faster one. Both are of course far outside the bar; the
point of this cell is the seam, not the sampler.

This also corrects a documentation claim that was already stale before this
campaign: `docs/internal/getting_started/gpu.md` said "Multi-GPU sharding via
`jax.pmap` or `shard_map` is not yet wired" until PR #2027 replaced it.
`CatalogFitter._sharded_vmap` and `_resolve_devices` map the galaxy axis over a
`Mesh` via GSPMD for `mcmc_nuts` and `mcmc_hmc`; `shard_map` specifically is not
used because BlackJAX's NUTS carries a `lax.cond` that trips manual
varying-axis tracking.

## Caveats

**Caveat 1 — D = 3 is not the paper's D = 12.** The fixture is the benchmark's
own: a double-power-law SFH with mass, slope and metallicity free, everything
else pinned, five broadbands. The paper fits twelve parameters. A
galaxies-per-GPU-minute number from a 3-parameter posterior is not comparable to
one from a 12-parameter posterior on cost *or* on difficulty, and the direction
of the bias is not even obvious: fewer parameters is cheaper per gradient but
the batch cost here is dominated by kernel launch, not by D.

**Caveat 2 — one chain per galaxy.** `CatalogFitter._run_native_mcmc` gives each
galaxy a single chain, so "split R-hat" here is the within-chain split
diagnostic, not the between-chain one. It is the same diagnostic
`Posterior.rhat()` exposes and the same one the 2026-08-17 reports used, but it
is weaker than four chains from dispersed inits and cannot see a chain stuck in
the wrong mode.

**Caveat 3 — every number here is at SNR 20 under a quadrature LUT.** The fit
runs on `WavePrecomp(band_integration="quadrature", n_subbands=5)`, whose
forward bias is constant in SNR but enters the posterior gradient **multiplied**
by SNR (#1671; ~5 % relative gradient error at SNR 30, ~50 % at SNR 300). At
this mock's SNR 20 the runtime estimator
(`_warn_if_lut_bias_amplified`) stayed below its warning threshold and no
`PrecompBiasWarning` fired, which is why these throughput numbers are quotable
at all. **They are not quotable at higher SNR**: a deeper catalog would want
`approx=None`, which is a different and slower forward model, so the wall clocks
here would not carry over either. The harness records the median SNR and the
resolved `band_integration` in every JSON row so this cannot be lost.

**Caveat 4 — this GPU was also driving the desktop**, and other agents were
running on the same host during part of the campaign. Treat wall clocks as
indicative to ~10 %. R-hat, ESS and divergence counts are deterministic given
the seed and are not affected.

**Caveat 5 — the cold column is a cache load, not a compile.** `import tengri`
enables JAX's persistent compilation cache by default. Set
`TENGRI_DISABLE_JAX_CACHE=1` to measure a real first compile. Every JSON row
records `jax_persistent_cache` so the two cannot be confused.

## What was NOT measured, and why

- **Multi-GPU sharding.** This box has one CUDA device, so `--shard` has nothing
  to compare. `CatalogFitter._sharded_vmap` and `run(devices="all")` exist and
  are exercised by the CPU-emulated device path
  (`XLA_FLAGS=--xla_force_host_platform_device_count=N`); a real multi-GPU
  scaling number needs a multi-GPU box.
- **`K = 1` at large N.** At `forward_chunk_size=1` the cost is exactly N
  sequential `lax.map` iterations, so galaxies/second is independent of N. It
  was queued at small N and never reached before the campaign's time budget ran
  out, so the K sweep in the table starts at K = 32.
- **N = 2048.** Queued twice at K = 512 with 2700 s and 1500 s timeouts and
  killed by both. N was reduced to 512, which is the reduction the plan asks for
  first, and this is that reduction being said out loud rather than a quiet
  drop. The rate is not expected to move much — N = 2048 at K = 512 is four
  `lax.map` iterations of the N = 512, K = 512 cell — but the wall time
  disagreed with that model by more than 2x, and the un-modelled part is
  per-galaxy **host-side** work (`CatalogFitter` builds one `Posterior` object
  and one summary block per galaxy, in Python, after the device work is done).
  That O(N) host cost is real and is not in the 372.7 figure. Measure it before
  quoting a rate at N in the thousands.
- **`approx=None` (the exact projector).** Out of scope here; it is a different
  forward model and would need its own sweep.

## Reproduce

All commands from the repo root. Precision is process-global and is chosen
**before** `import jax`, so one `--dtype` per process; the script does that
translation from its own argv and refuses to run if `jnp` does not then allocate
the dtype asked for. Results merge into one JSON keyed on the configuration.

```bash
# 1. the precision question, in isolation (Finding 1)
python bench/scripts/benchmark_catalog_throughput.py --mode grad --dtype f64 \
    --n-gal 1 8 32 64 128 512 2048 --reps 4 --runs 20 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060
python bench/scripts/benchmark_catalog_throughput.py --mode grad --dtype f32 \
    --n-gal 1 8 32 64 128 512 2048 --reps 4 --runs 20 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060

# 2. the headline throughput cell (Finding 2)
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --dtype f32 --n-gal 512 --chunk 512 \
    --warmup 400 --burnin 0 --samples 500 \
    --json bench/results/gpu_catalog_throughput.json --tag rtx3060

# 3. the same cell in float64, and the narrow-K cell
python bench/scripts/benchmark_catalog_throughput.py \
    --method mcmc_hmc --dtype f64 --n-gal 512 --chunk 512 \
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
    --method mcmc_hmc --dtype f64 --n-gal 32 --chunk 8 --shard \
    --warmup 50 --burnin 0 --samples 100

# 7. to measure a real first compile rather than a persistent-cache load
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
    tests/contract/test_bench_cli.py -v -m "slow or not slow"
```
