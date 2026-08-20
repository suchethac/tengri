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
**Caveat up front:** this GPU was also driving the desktop.

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

— is real and simply never reached, because the workload is ~0.12 FLOP/byte. Note
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

A fit is safe: the likelihood standardizes the residual by σ *before* squaring,
lifting the magnitudes back into range, and a 300-step float32 MAP moves all seven
parameters. So float32 *inference* works; float32 `jax.grad` of a raw observable
does not, and it fails silently.

`tests/regression/precision/test_inference_grad_float32.py` pins that objective
gradient **finite** — and zero is finite, so existing coverage could not have
caught this.

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

## To file

Not filed from this session (no `gh` available). Ready to open:

1. **float32 `jax.grad` of photometry returns exactly zero** — `area:perf`,
   `area:forward`, `bug`, `silent-failure`. Evidence: Finding 5. Note the existing
   test asserts finiteness, which zero satisfies; a regression test should assert
   *nonzero* against a float64 reference.
2. **`~/.cache/tengri_precomp` is keyed on neither dtype nor backend** —
   `area:perf`, `bug`, `silent-failure`. Two values in the z-table (`dl_cm`,
   `igm_transmission`) go through `jnp`, so a float32 run can write an entry a
   float64 run later reads.
3. **Nine bench scripts hard-pin `JAX_PLATFORMS=cpu`** — `area:perf`, `area:tests`,
   `type:refactor`. They cannot be run on a device without editing.
   `benchmark_catalog_throughput.py` and `benchmark_device_matrix.py` are the two
   that select the platform from the environment.
4. **`bench/README.md`'s script table is missing entries**, including
   `benchmark_catalog_throughput.py` — `area:tests`, `documentation`.
5. **float32 geoVI emission-line marginalization dies on cuBLASLt** —
   `area:inference`, `area:nebular`, `bug`. `INTERNAL: GEMM is not supported by
   cublasLt and legacy cublas fallback is removed`, two tests in
   `test_geovi_metric_float32.py`, CUDA only, not precision-related.
6. **`test_float64_gradient_does_not_poison_a_later_float32_gradient` is
   GPU-intermittent** — `area:tests`, `area:perf`. Exact `assert_array_equal` at
   ~8 ulp on a device that need not reproduce its reduction order; wants a GPU
   tolerance.
7. **tengri should consider defaulting `jax_default_matmul_precision` to
   `highest` on GPU** when x64 is off — `area:api`, `area:perf`,
   `silent-failure`. It costs nothing measurable (Finding 7) and its absence
   silently degrades every float32 matmul on Ampere+.

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

## Not done

`notebooks/nvidia_cuda.py` is **not registered in the published docs spine.**
Publishing it requires an executed render carrying `image/png` outputs —
`tools/check_notebook_renders.py` enforces that, correctly — and this environment
has no kernel stack to produce one (`ipykernel`, `jupyter_client`, `nbclient` and
`nbconvert` are all absent from the venv). The notebook itself runs end to end on
CUDA; only the render is missing. To publish:

```bash
pip install nbclient ipykernel                       # not installed here
python scripts/execute_notebooks.py nvidia_cuda      # never set MPLBACKEND
```

then add `"nvidia_cuda"` to `EXPERIMENTAL_SLUGS` in
`scripts/sync_spine_notebooks_for_docs.py` and a bullet plus toctree entry in
`docs/spine/experimental/index.md`.

One warning learned by doing it: `scripts/sync_spine_notebooks_for_docs.py`
rewrites *every* published render, not just the one you added. Two unrelated
notebooks (`08_emission_lines`, `09_parameter_sweeps`) picked up 1226 lines of
diff from source drift that predates this work. Check `git status` after running
it and restore anything you did not mean to touch.

## See also

- `notebooks/nvidia_cuda.py` — the same measurements as a narrative, with the
  crossover figure.
- `notebooks/apple_mps.py` — the Apple GPU counterpart. float32-only, and reaches
  parity rather than a win.
- `bench/reports/2026-05-06_forward_model_speedup.md` — the CPU-only forward
  baseline this sits beside.
- `docs/internal/getting_started/gpu.md` — install and device selection.
