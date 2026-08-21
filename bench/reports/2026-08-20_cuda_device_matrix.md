# CUDA: the CPU wins one galaxy by 33x, the GPU wins 2048 by 15x, and float32 buys nothing until it does

**Date:** 2026-08-20
**Verdict:** GPU is a width instrument, not a speed instrument. A batched forward
or gradient crosses between 128 and 512 galaxies and reaches 14.7x by 2048; catalog
NUTS crosses between 64 and 256 but is only 1.22x ahead at 256, because a sampler's
sequential half does not batch. Single-galaxy work — including a single fit —
belongs on the CPU. One real defect found: the float32 photometry gradient is
identically zero.
**Platform:** Linux, RTX 3060 12 GB (driver 580.173.02, CUDA 13) vs AMD Ryzen 9
5900X (12 cores / 24 threads), 62 GB RAM. JAX 0.11.0 / jaxlib 0.11.0.
**Precision:** both arms measured — float64 (tengri's default) and float32 via
`JAX_ENABLE_X64=0`, each proven on an output array's dtype, not on the config flag.
**Wall clocks** are warm steady state, minimum of 4 repetitions of 30 timed calls,
every call `block_until_ready`. **FLOPs** are off
`jax.jit(fn).lower(...).compile().cost_analysis()["flops"]`.
**Caveat up front:** this GPU was also driving the desktop. And the sampler
findings (9, 13, 14) are measured on a fixture chosen for cheap forward passes,
which turns out to be very hard to sample — see **Finding 15** before quoting any
of their convergence or cost-per-posterior numbers.

## Question

`docs/performance/index.md` said "GPU/TPU work without modification but are not
benchmarked", and `docs/internal/getting_started/gpu.md` asserted that
`predict_*_batch` is where the GPU wins without a number behind it. Two questions,
kept apart: does tengri *run* on CUDA, and *where* is it faster.

## Configuration

`recipes.mock_recovery_minimal()` — truncated skew-normal SFH, Calzetti dust,
nebular and AGN off, redshift fixed at z = 0.05, seven free parameters — on
`data/fsps_prsc_miles_chabrier.h5`, five SDSS bands (*ugriz*), `approx=WavePrecomp()`
unless stated. AGN is deliberately absent: the pure-float32 inventory in
`tests/regression/precision/` carries strict xfails for three AGN discs and SKIRTOR
interpolation failures, so an AGN model would measure a known-broken path.

The galaxy is **fixed, not sampled**: each free parameter sits at the median of its
declared prior. See Caveat 1 — this is the single most important methodological
choice in the report.

One shape per process; the driver spawns a child per cell and each child gets its
own `TENGRI_PRECOMP_CACHE_DIR`, since that cache is keyed on neither dtype nor
backend.

## Finding 1 — one galaxy: the CPU wins by 33x forward, 13x gradient

Microseconds, warm. A/A is the same arm against itself — the resolution floor.

| | CPU f64 | CPU f32 | GPU f64 | GPU f32 |
|---|---:|---:|---:|---:|
| forward `predict_photometry` | **227.0** | **162.3** | 7422.0 | 7308.2 |
| gradient of the sum | **587.0** | **479.9** | 7755.3 | 7398.8 |
| first call (compile) [ms] | 467 | 408 | 527 | 428 |
| A/A floor (forward) | 1.023 | 1.242 | 1.015 | 1.010 |

CPU/GPU is 32.7x (forward) and 13.2x (gradient) in float64 — both two orders of
magnitude clear of the floor. The gradient gap is the smaller one, which is what
arithmetic intensity predicts: the reverse pass does more work per byte already
moved.

**Compile time is a wash: 467 ms against 527 ms.** Worth stating because on Apple
MPS the GPU compiled ~10x *faster*, which was that backend's one honest win. It
does not transfer to CUDA.

## Finding 2 — the batch sweep, and the crossover

Microseconds **per galaxy**; bold is the faster device in the row.

Forward, `predict_photometry_batch`:

| batch | CPU f64 | CPU f32 | GPU f64 | GPU f32 |
|---:|---:|---:|---:|---:|
| 1 | **234.0** | **142.6** | 7344.4 | 7361.7 |
| 8 | **113.5** | **95.9** | 927.3 | 899.3 |
| 32 | **55.3** | **53.1** | 238.1 | 226.9 |
| 128 | **27.6** | **26.1** | 65.7 | 56.8 |
| 512 | 45.7 | 16.7 | **22.0** | **15.2** |
| 2048 | 48.3 | 20.0 | **11.2** | **4.5** |

Gradient, `vmap` of `grad(sum(predict_photometry))`:

| batch | CPU f64 | CPU f32 | GPU f64 | GPU f32 |
|---:|---:|---:|---:|---:|
| 1 | **499.4** | **435.6** | 7584.7 | 7473.8 |
| 8 | **299.8** | **164.0** | 977.0 | 933.1 |
| 32 | **131.2** | **93.6** | 250.8 | 233.2 |
| 128 | 211.8 | **43.7** | **77.9** | 58.5 |
| 512 | 206.6 | 67.4 | **31.5** | **16.3** |
| 2048 | 172.2 | 80.7 | **19.8** | **5.5** |

The crossover sits between 128 and 512 and depends on the shape: the gradient
crosses at 128 in float64 (GPU 2.7x ahead), the forward pass at 512. At 2048 the
GPU leads by 4.3x (forward f64), 8.7x (gradient f64) and **14.7x** (gradient f32).

The mechanism is in the totals, not the ratios. From 1 to 2048 galaxies, GPU f32
forward **total** goes 7.36 ms → 9.12 ms: 2048x the work for 1.24x the time. The
CPU goes 0.14 ms → 40.9 ms, linear. The GPU is not getting faster; it is finally
given enough work to be worth waking up.

VRAM peaked at 3.0 GB (2048, f64) and 1.8 GB (2048, f32) against 12 GB. **Batch
size was never the memory constraint in this sweep** — which contradicts the
order-of-magnitude allowance in `inference/_batching.py` by a wide margin on this
model.

## Finding 3 — a single MAP fit belongs on the CPU, and the device does not change the answer

300 adam steps, `Fitter.run("map")`, resolved `approx` printed and identical across
arms (`wave_precomp=True, n_subbands=5`).

| | CPU f64 | CPU f32 | GPU f64 | GPU f32 |
|---|---:|---:|---:|---:|
| cold (includes compile) [s] | **2.79** | **2.25** | 13.12 | 11.14 |
| warm [s] | **0.27** | **0.23** | 2.38 | 2.22 |

CPU wins warm by 8.8x. A fit is a few hundred *sequential* steps, each one a
dispatch — the worst shape for a GPU, and batching does not apply to one galaxy.

The correctness result matters more than the timing: **GPU float64 reproduced CPU
float64 to six decimals on all seven parameters.** Same optimum, same trajectory.

## Finding 4 — float32 is accurate forward, and buys nothing here

Against a float64 **CPU** reference, errors masked at `|reference| > 1e-45`:

| arm | max rel. error (photometry) | median | finite |
|---|---:|---:|---|
| GPU f64 | 2.5e-16 | 1.2e-16 | yes |
| CPU f32 | 3.1e-07 | 1.7e-07 | yes |
| GPU f32 | 3.2e-07 | 1.6e-07 | yes |

float32 costs float32 epsilon and nothing worse — the log-offset treatment of the
cosmological flux scale holds at z = 0.05, where a linear `4*pi*d_L^2` would
overflow to `inf`.

But it buys nothing at small width: every f32-vs-f64 comparison in Finding 1 sits
*inside* its own A/A floor (the sole exception is the GPU gradient at ~5%). The
57x float32 advantage this card has in dense arithmetic —

| dense 2048³ matmul | CPU | GPU |
|---|---:|---:|
| float32 | 988 GFLOP/s | **10,740 GFLOP/s** |
| float64 | **379 GFLOP/s** | 189 GFLOP/s |

— is real and simply never reached, because the workload is ~0.12 FLOP/byte (that
intensity is measured in `notebooks/apple_mps.py`, on the compiled graph rather
than the device, which is why it predicts this backend too). Note
the float64 row: **this GPU is half the CPU's float64 throughput**, because GeForce
cards run fp64 at 1/64 rate. On A100/H100 it is ~1/2, so that row is the one result
here that does not transfer.

float32 does pay above the crossover: at 2048 it is a further 2.5x on the GPU
forward pass and 3.6x on the gradient.

## Finding 5 — the float32 photometry gradient is identically zero

`jax.grad(lambda p: jnp.sum(model.predict_photometry(p)))` returns **exactly zero
for all seven parameters in float32** — on CPU and GPU, on the exact path and under
`WavePrecomp`, signs preserved as `-0.0`/`+0.0`. In float64 the same seven
derivatives are 1e-26 to 1e-28 and all nonzero. Nothing raises, nothing warns.

The magnitudes are not the explanation: 1e-26 is comfortably inside float32
(smallest normal 1.2e-38). The reverse pass forms the linear cosmological flux
factor, ~1e-57, which the forward path deliberately carries as a log10 offset and
which is 0 in float32.

What it does and does not break:

| differentiate | float32 |
|---|---|
| `sum(predict_photometry)` — bare forward surface | **identically zero** |
| `neg_log_posterior_fn` — what a fit descends | healthy (-32.2), nonzero, finite |

The objective gradient comes back finite and nonzero (−32.2 on the threaded
objective), and a 300-step float32 MAP moves all seven parameters. **That is not
the same as being right, and I originally drew the wrong conclusion from it.**

This is already open as **#1415**, which is more careful: it verifies against
central finite differences and finds the likelihood-path gradient wrong by
*structured factors* — "~2x on stellar mass" — not merely finite. So **float32
fitting is not safe**, and the correct statement is #1415's, not "fits are
unaffected". The root cause is tracked in #1388 (`apply_log10_scale` is
gradient-unsafe above ~1e38).

`tests/regression/precision/test_inference_grad_float32.py` pins that objective
gradient **finite** — and zero is finite, so existing coverage could not have
caught the bare-observable case either.

## Finding 6 — catalog NUTS crosses between 64 and 256 galaxies, but converts little of the batch advantage

`CatalogFitter.run("mcmc_nuts", forward_chunk_size=K)`, `K = n_gal`, 10 warmup +
10 burnin + 20 samples, `dense_mass_matrix=False`. Warm wall clock, seconds.

| galaxies | CPU f32 | GPU f32 | faster | GPU gal/s |
|---:|---:|---:|---|---:|
| 16 | **72.8** | 247.3 | CPU 3.40x | 0.06 |
| 64 | **139.0** | 261.6 | CPU 1.88x | 0.24 |
| 256 | 334.7 | **274.5** | GPU 1.22x | 0.93 |

At 16 galaxies in float64: CPU 105.7 s against GPU 279.4 s, CPU by 2.64x.

The crossover is between 64 and 256, consistent with the 128–512 of Finding 2, and
the flatness is again the mechanism: 16x the work costs the GPU 1.11x the time.

**The size of the win is the part worth carrying.** At 256 the card leads by 1.22x,
against the 8.7x the bare gradient showed at 2048. The CPU amortizes as well —
72.8 → 334.7 s is 4.6x for 16x the galaxies — because a vectorized NUTS over a
wider axis is more efficient on either device, and a sampler interleaves its wide
forward passes with sequential leapfrog steps and per-iteration control flow that
do not shrink with width. A catalog fit is where a GPU begins to pay; it is not
where it pays like a batched gradient.

**Every arm returned finite posterior draws, float32 included.** That was not
obvious in advance: float32 coverage pins the *objective gradient* finite, and a
converging fit with NaN posterior draws is a documented float32 geoVI failure mode.
Catalog NUTS in float32 is not asserted anywhere in the suite; here it worked.

## Finding 7 — float32 on CUDA silently becomes TF32, and it costs accuracy but not speed

On Ampere and later, XLA lowers float32 matmuls to TF32 (19 bits, 10-bit mantissa)
by default. tengri's own float32 Fisher test fails on CUDA out of the box and passes
on the CPU:

```text
AssertionError: float32 FIM differs from float64 by 3.922e-03 relative
AssertionError: float32 error bars differ from float64 by 4.494e-02
```

A 4.5% error on parameter error bars. Both pass with

```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

Two measured details that make this actionable:

- **`NVIDIA_TF32_OVERRIDE=0` alone does not fix it** — 2 of 6 still fail. XLA picks
  its own algorithm, so the JAX-level knob is the one that binds. Setting both is
  the same as setting the JAX one.
- **It costs no measurable speed.** Re-running Finding 2's GPU f32 batch arm with
  the knob on: 4.42 vs 4.45 us/galaxy forward and 5.51 vs 5.49 gradient at 2048,
  i.e. inside the A/A floor. float32's advantage in tengri is halved memory
  traffic, not tensor cores, so there is nothing to trade. **Set it
  unconditionally for float32 work on CUDA.**

## Finding 8 — the capability sweep: 5 of 559 float32 tests fail on CUDA, 2 after the knob

`tests/regression/precision/` (57 files) is the part of the suite most exposed to a
device change, and `tests/conftest.py` pins no platform, so it already runs on
whatever JAX picks.

| | CUDA | CUDA + `matmul_precision=highest` | CPU |
|---|---:|---:|---:|
| passed | 547 | 550 | **552** |
| failed | **5** | **2** | 0 |
| skipped / xfailed | 3 / 4 | 3 / 4 | 3 / 4 |

Three distinct causes:

1. **Two are TF32** (Finding 7). Fixed by the knob.
2. **Two are a hard cuBLAS error** in the geoVI metric's emission-line
   marginalization, and the knob does not help:
   `INTERNAL: GEMM is not supported by cublasLt and legacy cublas fallback is
   removed.` Not precision — the GEMM shape is one cuBLASLt refuses and JAX 0.11
   has dropped the fallback that used to absorb it. **float32 geoVI with
   marginalized emission lines does not run on CUDA.** It fails loudly, which is
   the right failure mode.
3. **One is the #1392 cross-precision kernel-cache guard**, and it is
   **intermittent** — it failed the first CUDA run of the tree and passed the
   second. The discrepancy is 9.9e-07 relative, about 8 ulp in float32, where
   #1392 itself was a wrong-precision kernel producing NaNs. `assert_array_equal`
   is exact, and GPU reduction order and autotuning need not repeat run to run.
   That points at nondeterminism, not at the cache serving the wrong kernel.

In **float64** — the default — nothing in this tree fails on CUDA at all.

## Finding 9 — HMC is ~50x cheaper per draw and does not mix: the speed was a dead chain

**This finding replaces an earlier version of itself, and the correction is the
point.** Measured at a token budget (10 warmup, 10 burnin, 20 samples, 256
galaxies, float32, `K = n_gal`), swapping `mcmc_nuts` for `mcmc_hmc` looked like a
transformation:

| | CPU f32 | GPU f32 |
|---|---:|---:|
| `mcmc_nuts` | 334.7 s | 274.5 s |
| `mcmc_hmc` | 4.92 s | 5.71 s |

48x on the GPU, 68x on the CPU. That number is real and it means nothing, because
20 draws cannot tell you whether a chain moved. Re-run at a real budget — 1000
galaxies, 300 warmup, 1000 samples, GPU float32, 149 s — and the diagnostics say
the sampler is not sampling:

| | value |
|---|---:|
| ESS_min, median galaxy | **1.5** (of 1000 draws) |
| ESS_min, worst galaxy | 0.7 |
| split R-hat, max | **3.22** |
| galaxies with R-hat > 1.01 | **100%** |

Two further attempts, both worse than they look:

- **`HMC_VALIDATED` from `notebooks/_setup.py`** (1000 warmup, 20 leapfrog steps,
  `target_accept_rate=0.9`) produces a **completely dead chain** — all 600 draws
  identical. tengri raises on it rather than reporting a number, and the message
  is worth quoting because it names the trap: *"This is a dead fit, not a
  converged one — R-hat cannot detect it (both variances are zero, so it reads
  ~1.0)."* That recipe is validated for single-galaxy notebook fits, not for this
  catalog.

  **That guard is #1438 ("a frozen chain must not report as converged"), and the
  convention that would have caught this at the 20-draw budget already exists:**
  `docs/dev/hierarchical-flat-seam.md` prescribes asserting that draws *move*
  (`unique > 1`), for exactly this failure — "a tuner that returns NaN at short
  warmup and hands back a frozen chain that looks like a posterior". So the
  symptom is a known class with a working guard; what is not on record is the
  catalog path failing to mix at these settings. `benchmark_device_matrix.py` now
  records `draws_moved` / `n_unique_draws` alongside every cost number, which is
  the convention it should have followed from the start.
- **Lowering `target_accept_rate` to 0.7** (300 warmup, 600 samples, 20 leapfrog,
  183 s) improves nothing that matters: ESS_min median 2.3, max R-hat 1.94, 87.5%
  of galaxies above 1.01.

So the honest ordering inverts. **NUTS is expensive because it is doing the work
the geometry requires**; fixed-length HMC is cheap here because it is failing, and
at a 20-draw budget that failure is invisible. Per-draw cost is not a sampler
comparison. ESS per second is, and a cost ratio quoted without a convergence
diagnostic beside it is exactly the kind of number this report exists to avoid.

The general lesson for the device question: **an accelerator cannot rescue a
sampler that is not mixing, and it will happily make a dead chain 48x faster.**

## Finding 10 — a million galaxies' photometry in 0.68 s, which is 67x the CPU

Forward prediction over 10^3 and 10^6 galaxies, chunked, each chunk reduced to a
scalar before the next is dispatched so memory stays bounded. Each device at its
own best chunk size — they do not agree, see Finding 11.

| | 1000 galaxies | 1,000,000 galaxies | galaxies/s | best chunk |
|---|---:|---:|---:|---:|
| CPU f64 | 53.5 ms | 45.67 s | 21,900 | 50,000 |
| CPU f32 | 18.1 ms | 22.99 s | 43,500 | 1,000 |
| GPU f64 | 16.6 ms | 7.55 s | 132,500 | 50,000 |
| GPU f32 | **8.5 ms** | **0.679 s** | **1,472,000** | 100,000 |

**67x** against CPU float64, the default precision — far more than the 4.3-14.7x
the per-galaxy sweep of Finding 2 suggested, because at batch 2048 the card was
still partly overhead-bound and 100,000 is where it stops being. This is the one
regime where the GPU is unambiguously the right tool.

Two notes. GPU float64 is 11x slower than GPU float32 here, tracking the 1/64 fp64
rate, and lands almost exactly on CPU float32 — a coincidence worth remembering
when someone reports "the GPU was no faster". And CPU float32 showed a 1.6x
run-to-run spread (14.3 s once, then 22.99 / 23.50 / 23.53 s on a quiet box); the
table takes the reproducible value and no ordering depends on the choice.

A billion was started and abandoned: at this rate it is ~11 minutes, which is
measurable but was not worth the session time. Do not quote it as measured.

## Finding 11 — `forward_chunk_size` is worth 107x on the GPU and nothing on the CPU

One million galaxies, float32, varying only the chunk — the number of galaxies in
one `vmap`:

| chunk | GPU f32 | CPU f32 | GPU VRAM |
|---:|---:|---:|---:|
| 100 | 72.81 s | 29.88 s | 0.8 GB |
| 1,000 | 8.17 | **22.99** | 0.8 |
| 5,000 | 2.29 | 24.22 | 1.1 |
| 10,000 | 1.48 | 26.38 | 1.1 |
| 25,000 | 0.97 | — | 2.6 |
| 50,000 | 0.76 | 25.04 | 4.7 |
| 100,000 | **0.68** | — | 8.8 |
| 200,000 | `RESOURCE_EXHAUSTED` | — | — |

**107x on the GPU from one integer**, and the CPU is flat: 23-30 s across the whole
range, best at 1,000, because it is already saturated by its cores and larger
chunks only cost it cache. There is therefore no single good default — the GPU
wants the largest chunk that fits, the CPU wants about a thousand.

The card saturates near 100,000 (the last doubling buys 12% for twice the memory)
and dies at 200,000 trying to allocate 4.51 GiB. Note `forward_chunk_size` derives
from a **2 GB** budget (`inference/_batching.py`), which on a 12 GB card lands near
chunk 25,000 — about 30% off the best available. Raise
`TENGRI_FORWARD_MEMORY_BUDGET_GB` on a GPU.

## Finding 12 — spectroscopy is the shape the GPU is waiting for

A single galaxy, `predict_spectrum` on 2000 pixels (3800-9200 A, R = 2000) under
`SpectrumPrecomp`, against the 5-band photometry of Finding 1:

| | FLOPs | GPU f32 | CPU f32 | CPU f64 |
|---|---:|---:|---:|---:|
| photometry, 5 bands | 534,050 | 7308 us | 162 | 227 |
| spectrum, 2000 pixels | 12,040,605 | 7927 us | 896 | 2056 |
| ratio | **22.5x** | **1.08x** | 5.5x | 9.1x |

**22.5x the arithmetic for 8% more GPU time.** The CPU pays 5.5x. That is the
dispatch-bound regime stated as plainly as it can be: on the GPU the spectrum is
very nearly free relative to the photometry, because the card was idle either way.

The single-galaxy spectrum still loses to the CPU (7927 us against 896), so this is
not yet a GPU win — it is the reason to expect one at far smaller catalogs than
photometry needs, since each galaxy already carries 22x more work. **The batched
spectroscopy sweep was not run**, and it is the most promising unmeasured cell in
this report.

## Finding 13 — catalog `mcmc_hmc` does not converge on this model at any warmup, and #1999 is the same signature

Finding 9 established that the cheap HMC draws were not samples. This is the
matrix behind that, and it also connects to an open issue. All rows are the same
D=7 model (`mock_recovery_minimal`, 5 SDSS bands, z fixed), float32,
`K = n_gal`, `mcmc_hmc`.

| device | n_gal | warmup | samples | L | target | mass | ESS_min med | R-hat max | divergent med | verdict |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---|
| GPU | 1000 | 300 | 1000 | 10 | 0.85 | diag | 1.5 | 3.22 | — | not mixing |
| GPU | 1000 | 1000 | 600 | 20 | 0.9 | **dense** | — | — | — | **fully frozen**, #1438 guard raised |
| GPU | 1000 | 300 | 600 | 20 | 0.7 | diag | 2.3 | 1.94 | — | not mixing |
| CPU | 64 | 1000 | 600 | 20 | 0.9 | diag | 1.9 | 1.93 | 0 | not mixing |
| CPU | 64 | 200 | 200 | 20 | 0.9 | diag | 1.4 | 2.66 | 0 | not mixing |
| CPU | 64 | 500 | 200 | 20 | 0.9 | diag | 1.3 | 3.30 | 0 | not mixing |
| CPU | 64 | 2000 | 200 | 20 | 0.9 | diag | 1.3 | 3.01 | 0 | not mixing |
| CPU | 64 | 1000 | 600 | 20 | 0.9 | **dense** | 2.2 | 1.39 | 0 | not mixing |
| CPU | 64 | 300 | 300 | **150** | 0.8 | diag | 3.0 | **1.037** | 0 | closest to converged |

Read off the rows, in order of what each rules out:

1. **Warmup length is not the lever.** 200 → 2000 warmup at L=20 changes nothing
   (ESS_min 1.3-1.4, R-hat 2.7-3.3). I expected the opposite and was wrong.
2. **A dense mass matrix is not the lever either** — 1.9 → 2.2 ESS_min at 64
   galaxies. It helps R-hat (1.93 → 1.39) and does not fix it.
3. **Trajectory length is the lever that moves the needle.** L=20 → 150 takes max
   R-hat from 1.93 to **1.037** and the fraction of unconverged galaxies from
   100% to 37.5%. This matches #1986, which measured median min ESS 13 at 20
   steps against 219 at 150 on a 9-D problem. **`HMC_VALIDATED`'s L=20 is simply
   too short for this posterior**, and it is scoped to notebook fits in its own
   docstring.
4. **The chains are not frozen at 64 galaxies** — all 7 free parameters move in
   every diagnosed galaxy (`frac_galaxies_fully_frozen = 0.0`). They mix badly.
   The *fully frozen* case, where every draw of every parameter is identical and
   `rhat()` raises, appeared only at 1000 galaxies under `HMC_VALIDATED`.
5. **Divergences are ~0 throughout**, so nothing in the sampler's own reporting
   flags any of this. As `bench/reports/2026-08-17_nb01_nb05_nuts_vs_hmc.md` put
   it: "A divergence count of zero is not evidence of convergence for
   fixed-length HMC — it cannot report the failure mode that a fixed trajectory
   actually has, which is not exploring the direction at all."

**This is very likely the same defect as open issue #1999**
("mcmc_hmc freezes on 06_fitting_spectroscopy: 600/600 identical draws, both
precondition arms now broken"): same `HMC_VALIDATED` recipe, same 600/600
identical-draw signature, and #1999 reports 600 divergent transitions where the
catalog path reports ~0 — which is itself worth reconciling. #1999 is scoped to
one notebook and to single-galaxy spectroscopy; the catalog path, both devices
and photometry all show it too, so **the scope in #1999 is narrower than the
defect**.

Two wiring differences make the catalog path more exposed, both from reading
`catalog_fitter.py` rather than from measurement:

- **No MAP initialization.** Single-fit HMC runs an 8-restart ADAM MAP first
  (`hmc.py` `_maybe_map_init`); the catalog path starts every galaxy from
  `_initialize_unbounded`, i.e. `0.1 * normal` about the prior centre
  (`fitter.py:2810-2812`).
- **`dense_mass_matrix` is hardcoded `False`** (`catalog_fitter.py:1412`) and
  applied as `bool(dense_mass_matrix)` (`:1478`), where single-fit resolves
  `None` → `n_dim < 8` → **dense at D=7** (`nuts.py:138-140`). Passing the
  documented default `None` to the catalog path therefore silently selects
  diagonal, and unlike `run_nuts` the catalog path never logs which it chose.
- `precondition` — the whitening escape hatch for exactly this kind of geometry
  — is **unreachable** from `CatalogFitter` (`_run_native_mcmc` takes no
  `**kwargs`).

Neither is confirmed as *the* cause; item 3 is the only lever measured to help.

## Finding 14 — NUTS does not converge here either, and freezes 3% of galaxies outright

The obvious reply to Finding 13 is "use NUTS". Measured, 1000 galaxies, GPU
float32, `K = 1000`, 300 warmup + 100 samples:

| | `mcmc_nuts` | `mcmc_hmc` |
|---|---:|---:|
| wall clock | **3861.9 s** (64 min) | 149.1 s |
| per galaxy | 3.86 s | **0.149 s** |
| per iteration (whole batch) | 9.65 s | 0.115 s |
| ESS_min, median galaxy | 2.1 (of 100 draws) | 1.5 (of 1000) |
| split R-hat, max / median | 1.19 / 1.069 | 3.22 / — |
| galaxies with R-hat > 1.01 | 96.8% | 100% |
| **galaxies fully frozen** | **3.1%** | 0% at 64 gal; all at 1000 under `HMC_VALIDATED` |
| divergences, median | 0 | 0 |
| ESS/s (catalog-wide) | 0.53 | **10.1** |

NUTS is **26x more expensive per galaxy** and **84x per iteration**. It buys better
per-draw quality — 2.1 effective of 100 draws (2%) against 1.5 of 1000 (0.15%),
so about 14x more efficient per draw — but not enough to cover 84x more cost, so
HMC is ahead on ESS per second by ~19x. **Neither is converged**, and the honest
reading is that this is not a choice between a fast wrong answer and a slow right
one: both are wrong at practical budgets.

The 3.1% is the part worth acting on: **NUTS returned a completely frozen chain
for 1 galaxy in 32**, every draw of every parameter identical. In a 1000-galaxy
run that is ~31 galaxies whose posteriors are their initial point, and nothing in
the output says so — `n_divergent` is 0 for them, and a per-galaxy `rhat()` call
is the only thing that raises. A catalog fit has no aggregate convergence gate.

### What a usable posterior would cost

Scaling to 100 effective samples per galaxy, which is the low end of usable, and
assuming ESS grows linearly with draws (optimistic — with R-hat at 3.2 it may
not):

| | 1000 galaxies to ESS_min = 100/galaxy |
|---|---:|
| `mcmc_hmc` (L=10, GPU f32) | ~2.8 hours |
| `mcmc_nuts` (GPU f32) | ~51 hours |

So: **hours to days for one catalog, with no validated configuration at the end
of it.** That is the answer to "how long does a catalog posterior take" on this
model, and the fix is not a faster device — the GPU is already doing 1.47M
forward predictions a second (Finding 10). It is a sampler that mixes.

## Finding 15 — the convergence numbers above are a property of this fixture, not of tengri

Merged PR #2014 re-measured the single-galaxy sampler table under blackjax 1.6.2
and reports **min ESS median 118 at L=150** and **10 at L=20**. Findings 13-14
report 1.3-3.0 on the same settings. Two orders of magnitude apart, so one of them
is not measuring what its label says. It is mine, and this is what closed it out.

First, the environment is not the difference. #2014 records that the shared venv
had been rebuilt **below** the declared `blackjax>=1.6` floor on 2026-08-18, which
invalidated the #1986 campaign. This venv runs **blackjax 1.6.2**, above the
floor, so nothing here is that bug.

Second, the amount of data is not the difference either. The obvious suspicion is
that `mock_recovery_minimal` is under-determined — **7 free parameters against 5
broadband fluxes** — so I re-ran the identical model with a 260-pixel spectrum
instead, taking it comfortably over-determined. Same galaxy, same settings
(L=150, 1000 warmup, 600 samples, float64):

| observable | data points | single-galaxy ESS_min | catalog (n_gal=1) ESS_min |
|---|---:|---:|---:|
| photometry | 5 | 1.7 | 1.9 |
| spectrum | 260 | 4.3 | 1.7 |

52x the data moves ESS_min from 1.7 to 4.3. **It is not a data-volume problem.**

What is left is the SFH parameterization. `mock_recovery_minimal` uses `tsnorm`,
whose `skew`, `trunc` and `width_gyr` are strongly degenerate with each other and
with `peak_lbt_gyr`; #2014's page uses a different family. So **Findings 9, 13 and
14 characterize the samplers on a fixture chosen for cheap forward passes, and
that fixture happens to be very hard to sample.** They are not a general statement
about catalog inference in tengri, and the 2.8-hour and 51-hour projections in
Finding 14 should not be quoted as tengri's cost for a 1000-galaxy posterior. A
benchmark fixture picked for speed is the wrong instrument for a convergence
claim, and that is the methodological error here.

Three things do survive, because they are qualitative and reproduce independently
of ESS:

1. `HMC_VALIDATED` at 1000 galaxies returns 600/600 **identical** draws and trips
   the dead-fit guard — the same signature as open issue #1999.
2. `mcmc_nuts` returned a completely frozen chain for **3.1% of galaxies** with 0
   divergences reported, so the freeze is not specific to fixed-length HMC.
3. A catalog fit has **no aggregate convergence gate**: only a per-galaxy
   `rhat()` call raises, so those frozen galaxies are silent in a catalog result.
   (#2008, merged after this work began, now announces a dead fit at `Posterior`
   construction — which addresses exactly this, on the single-fit path.)

The catalog-versus-single comparison is suggestive and **under-powered**: on
identical inputs the catalog path was worse at L=20 photometry (ESS 1.8 vs 3.0,
R-hat 1.47 vs 1.04) and at L=150 spectroscopy (1.7 vs 4.3), and marginally better
at L=150 photometry (1.9 vs 1.7). One seed each. It is consistent with the missing
MAP init and forced diagonal mass, and it does not establish them.

## Interpretation

The GPU question is a shape question. tengri's forward model is memory- and
dispatch-bound at ~0.12 FLOP/byte, so a device with 10 TFLOP/s of float32 ALUs has
nothing to do until enough galaxies are in flight to hide its dispatch latency.
Below ~128 galaxies the CPU wins by one to one-and-a-half orders of magnitude;
above ~512 the GPU wins by up to 15x. Nothing in between is worth arguing about.

Practically: keep single-galaxy fitting and interactive work on the CPU in float64.
Reach for the GPU for catalogs, posterior-predictive sweeps and mock generation —
work that is wide by construction.

## Verdict

**Capability: PASS in float64, PASS with two caveats in float32.** No
tengri-side change is needed; the forward model, gradients, MAP and the catalog
sampler all run on CUDA in both precisions, and float64 results are bit-comparable
with the CPU — 552/552 of the float32 regression tree passes on CPU and, in float64,
nothing in it fails on CUDA either. In float32 on CUDA, set
`JAX_DEFAULT_MATMUL_PRECISION=highest` (Finding 7) and avoid marginalized emission
lines in geoVI (Finding 8).

**Speedup: conditional, and shape-dependent.** 0.03x–0.08x (a large loss) at one
galaxy; 4.3x–14.7x for a batched forward or gradient at 2048; only 1.22x for
catalog NUTS at 256, because a sampler's sequential half does not batch. Use width,
and expect a sampler to convert less of it than a forward pass does.

**And do not read a cost ratio as a sampler recommendation.** Fixed-length HMC is
~50x cheaper per draw than NUTS on this catalog and does not mix — ESS_min ~1.5 of
1000 draws, split R-hat to 3.2, and a dead chain under the repo's own validated
recipe (Finding 9). NUTS costs what it costs because it is doing the work. An
accelerator will make a non-mixing chain 48x faster and tell you nothing.

## Caveats

1. **A sampled fixture is not a valid cross-precision control.** `jax.random`
   returns different numbers for the same key at different widths, so
   `spec.sample(key)` hands the two arms *different galaxies*. The first version of
   Finding 4 read a factor of **152** as precision error for exactly this reason.
   Every number here uses a fixed parameter vector at the prior medians.
2. **The driver process must be pinned to the CPU.** `benchmark_device_matrix.py`
   imports jax at module scope, so an unpinned driver preallocates 75% of the card
   and every GPU child then competes with its own parent — measured at 9–10 GiB
   held by the driver alone, which also made the first VRAM column read 11.8 GB
   instead of 3.0 GB.
3. **This GPU was driving a desktop** (Xorg, gnome-shell, a browser: 1–3 GB,
   20–40% utilization). An idle card would do better in absolute terms; the
   direction and the crossover scale are what to carry.
4. **`cost_analysis` FLOPs are not comparable across backends.** The same call
   reports 682,006 FLOPs on CPU and 547,864 on GPU because the compilers fuse
   differently. They are an invariant *within* a platform, not between platforms —
   so the device comparison here is wall clock, as it has to be.
5. **The float64 penalty is consumer-specific.** 1/64 rate on GeForce, ~1/2 on
   A100/H100.
6. **NIFTy `vi*` and `map(optimizer="lbfgs_scipy")` are excluded by construction,
   not overlooked.** `jft.optimize_kl` is a Python-level outer loop with
   per-iteration host syncs and ~20 GB of *host* RSS; scipy's L-BFGS-B drives a host
   loop and converts every gradient to `np.float64`. Neither can benefit from a
   device.
7. **Comparing float32 and float64 fit *optima* is confounded here.** Initial points
   come from `ctx.initial_params(key)`, which is itself a random draw and therefore
   differs between widths. Finding 3's cross-*device* comparison is clean; a
   cross-*precision* optimum comparison would need a pinned start.
8. **Single device.** `devices="all"` sharding exists for `mcmc_nuts`/`mcmc_hmc`
   (`catalog_fitter.py`) but one card cannot exercise it.
9. **Finding 6 has no A/A floor.** Each cell is one cold and one warm run, which
   agreed to 1–2%, so the 1.22x at 256 is probably real but rests on a thinner
   control than Findings 1–2. It is also a single K per width (`K = n_gal`), not a K
   sweep, so it does not locate the K that saturates the card.
10. **Catalog widths above 256 were not measured.** Extrapolating the two scalings
    (GPU 1.11x, CPU 4.6x per 16x) puts the GPU ~3x ahead near 1024, but that is
    arithmetic on two points, not a measurement.

## Filed

All of it, once `gh` became available. Two of the four things I was about to open
turned out to exist already, which is the useful part of checking first.

**Comments on existing issues**

* **#1999** (`mcmc_hmc freezes on 06_fitting_spectroscopy`) —
  [comment](https://github.com/suchethac/tengri/issues/1999#issuecomment-5365319116).
  The 600/600-identical signature reproduces on catalog *photometry*, on both CPU
  and CUDA, and `mcmc_nuts` froze 3.1% of galaxies with 0 divergences — so the
  scope in #1999 is narrower than the defect. Includes the Finding 15 caveat, so
  nobody reads my ESS magnitudes as tengri's.
* **#1415** (`Pure float32: photometry gradients are silently zero/wrong`) —
  [comment](https://github.com/suchethac/tengri/issues/1415#issuecomment-5365328835).
  Finding 5 is this issue, already open with finite-difference verification.
  Added: it reproduces on CUDA and on the exact path, the signs survive as ±0.0,
  and the float64 magnitudes are well inside float32 range so "underflow" does not
  mean the output was too small. Also flagged that `notebooks/apple_mps.py` times
  this exact shape in pure float32.

**Opened**

* **#2022** — float32 on CUDA silently becomes TF32: 4.5% on parameter error bars,
  and `NVIDIA_TF32_OVERRIDE=0` does not fix it (Findings 7-8).
* **#2023** — float32 geoVI emission-line marginalization dies on CUDA, cuBLASLt
  refuses the GEMM (Finding 8).
* **#2024** — `~/.cache/tengri_precomp` is keyed on neither dtype nor backend.
* **#2025** — `forward_chunk_size`'s 2 GB budget leaves ~30% on the table on a GPU
  where the knob is worth 107x (Findings 10-11).
* **#2026** — the only catalog `mcmc_hmc` test is D=1 and asserts shape and
  finiteness, so a frozen chain passes it.

**Not filed, deliberately**

The catalog-path wiring differences (no MAP init, `dense_mass_matrix` hardcoded
`False` and read as `bool(...)`, `precondition` unreachable, step size and
acceptance discarded) went into the #1999 comment as leads rather than a separate
issue: they are unconfirmed as the cause, and splitting them from the freeze they
might explain would fragment the discussion. The `method_selection.md` and
`known_limitations.md` staleness is left alone pending #2014's re-measurement,
which already moved that page.

## Reproduce

```bash
# one cell
JAX_PLATFORMS=cuda .venv/bin/python bench/scripts/benchmark_device_matrix.py --shape C

# float32
JAX_ENABLE_X64=0 JAX_PLATFORMS=cuda .venv/bin/python \
    bench/scripts/benchmark_device_matrix.py --shape C --precision f32

# the matrix (pin the DRIVER to cpu — see Caveat 2)
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_device_matrix.py --all

# the accuracy table, from the shape-G dumps
.venv/bin/python bench/scripts/benchmark_device_matrix.py --compare
```

## Published

`notebooks/nvidia_cuda.py` is registered in the docs spine
(`docs/spine/experimental/nvidia_cuda.ipynb`), executed on this CUDA box with its
figure captured. Re-render after editing it with

```bash
python scripts/execute_notebooks.py nvidia_cuda   # never set MPLBACKEND
```

Executing needs `nbclient`, `ipykernel` and `nbconvert`, which are not in the base
environment; they were installed into `.venv` for this work.

One warning learned by doing it: `scripts/sync_spine_notebooks_for_docs.py`
rewrites *every* published render, not only the one you added. Two unrelated
notebooks (`08_emission_lines`, `09_parameter_sweeps`) picked up 1226 lines of diff
from source drift predating this work. `scripts/execute_notebooks.py` writes the
render itself and is the safer entry point; if you do run the sync, check
`git status` afterwards.

## See also

- `notebooks/nvidia_cuda.py` — the same measurements as a narrative, with the
  crossover figure.
- `notebooks/apple_mps.py` — the Apple GPU counterpart. float32-only, and reaches
  parity rather than a win.
- `bench/reports/2026-05-06_forward_model_speedup.md` — the CPU-only forward
  baseline this sits beside.
- `docs/internal/getting_started/gpu.md` — install and device selection.
