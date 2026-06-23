# Performance

The forward model is pure JAX, so every backend (MAP, NUTS, geoVI, …)
runs against the same compiled graph. "How fast is tengri?" therefore
reduces to a small set of numbers that travel together.

Headline numbers (last full run May 2026) and how to reproduce them.
Scripts live under
[`bench/scripts/benchmark_*.py`](https://github.com/suchethac/tengri/tree/main/bench/scripts);
the single entry point is [Health check & dispatcher](#health-check-and-dispatcher).

```{warning}
Numbers below were measured April–May 2026 on JAX 0.9 / Apple M-series
and some pre-date recent forward-model changes. Re-run the relevant
script before quoting in a paper or PR;
[`bench/reports/`](https://github.com/suchethac/tengri/tree/main/bench/reports)
carries the date of every measurement.
```

## Headline numbers (Apple M-series CPU, x64, JAX 0.9, last run May 2026)

Forward photometric prediction on SDSS *ugriz* at z = 0.1, 5 bands,
running on a single CPU core:

| Configuration | Exact | Compositional | Hybrid (precomputed) |
|---|---:|---:|---:|
| Stellar only | 23.9 ms | 1.5 ms | **59 µs** (408×) |
| + nebular (BakedIn) | 24.7 ms | 1.5 ms | 58 µs (424×) |
| + nebular (Cue emulator) | 61.6 ms | 2.4 ms | 567 µs (109×) |
| + dust IR (THEMIS) | 27.3 ms | 2.0 ms | 158 µs (173×) |
| + radio + X-ray + AGN | 76.4 ms | 4.6 ms | 2.44 ms (31×) |
| Kitchen sink (all emitters) | **76.1 ms** | **4.6 ms** | **2.45 ms** (31×) |

The columns are the three internal forward strategies, selected at build time:
**Exact** = `approx=None` (default; exact wave-grid integration), **Hybrid
(precomputed)** = `approx=WavePrecomp()` (the SSP × filter lookup-table fast
path), and **Compositional** = the default fused JIT kernel when no `approx=`
LUT is requested. There is no `mode=` argument on `predict_photometry`; the
strategy is fixed when the `SEDModel` is constructed.

— *full table at [`bench/reports/2026-05-06_forward_model_speedup.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-05-06_forward_model_speedup.md)*

Inference backends on a 7-parameter mock fit (compile + sample wall):

| Backend | First call | Steady-state |
|---|---:|---:|
| MAP (L-BFGS) | ~5 s | < 1 s |
| Laplace | ~5 s | < 1 s |
| Pathfinder | ~10 s | ~2 s |
| NUTS (1k samples) | ~30 s | ~5 s |
| `vi_native` (geoVI, JAX-native) | ~10 s | **2.3 s** |
| `vi` (NIFTy.re) | ~75 s | 43.7 s |

— *full breakdowns: [`2026-04-17_native_vs_nifty.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-04-17_native_vs_nifty.md), [`2026-04-22_pathfinder_vs_window_nuts.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-04-22_pathfinder_vs_window_nuts.md), [`2026-05-06_compile_vs_sampling_breakdown.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-05-06_compile_vs_sampling_breakdown.md)*

`vi_native` is **19–25× faster** than the NIFTy path on smooth-SFH fits
but is **not drop-in posterior-equivalent**: PSD-timescale parameters
differ by an order of magnitude on stochastic fits. Validate per problem
before swapping.

## Persistent compile cache

JAX recompiles XLA programs on every cold start. Tengri auto-enables a
persistent on-disk cache at `~/.cache/tengri_jax_cache` so notebook
restarts, slurm tasks, and benchmark runs all skip the expensive first
compile (geoVI ~75 s, MGVI ~10 s, NUTS warmup tens of seconds).

```bash
export TENGRI_JAX_CACHE_DIR=/scratch/$USER/jax_cache  # custom location
export TENGRI_DISABLE_JAX_CACHE=1                     # opt out
```

After upgrading JAX, wipe stale entries:

```python
import tengri
tengri.clear_cache()
```

Default `min_compile_time_secs=0.05` persists per-filter kernels and
component precompute compiles. See
[compilation_cache.md](https://github.com/suchethac/tengri/blob/main/docs/inference/compilation_cache.md)
and [compilation_diagnostics.md](https://github.com/suchethac/tengri/blob/main/docs/inference/compilation_diagnostics.md)
for full details.

## Health check and dispatcher

A one-command quick read of *your* install:

```bash
python -m tengri.bench
```

prints the JAX backend, default device, persistent compile-cache size,
and a 1-galaxy + 100-galaxy timing on SDSS *ugriz*. ~30 s on CPU after
the cache is warm.

Every benchmark script under `bench/scripts/` is also reachable
through one entry point:

```bash
python -m tengri.bench list                      # show all
python -m tengri.bench help forward_model        # what does it measure?
python -m tengri.bench forward_model             # run it
```

Available benchmarks (`bench list`):

| Name | What it measures |
|---|---|
| `forward_model` | Forward photometry: exact / compositional / hybrid across all emitters |
| `components` | Per-component (stellar, dust, nebular, AGN, ...) wall-clock timing |
| `jit_compile` | Population-scale JIT compile time vs N galaxies |
| `jit_real_path` | Compile time on the production forward-model path |
| `inference_engines` | MAP / Laplace / NUTS / VI / NSS at D = 7, 12, 20 |
| `vi_native_vs_nifty` | geoVI: pure-JAX `vi_native` vs the NIFTy.re reference path |
| `vi_xlarge` | VI scaling on stochastic-SFH problems with D >> 100 |
| `population_native` | Hierarchical PopulationFitter: per-iteration cost vs N galaxies |
| `adam_vs_lbfgs` | MAP optimizers head-to-head |
| `cue` | Cue (Li+2025) nebular emulator timing in isolation |
| `loss_timing` | Per-call loss / negative-log-posterior timing |
| `joint_indices_e2e` | End-to-end timing for joint photometry + spectral indices |
| `precompute_analytic` | Analytic precompute lookup vs full-spectrum integration |
| `precompute_quad` | Quadrature precompute: accuracy vs grid resolution |
| `ztable_interp` | Metallicity-table interpolation kernel timing |

## Reproducing the headline numbers

```bash
JAX_PLATFORMS=cpu python -m tengri.bench forward_model
JAX_PLATFORMS=cpu python -m tengri.bench inference_engines
```

Each script writes its dated report to `bench/reports/` (or to
stdout, depending on the script). The reports there are the source of
truth for every number quoted on this page.
[`bench/RERUN.md`](https://github.com/suchethac/tengri/blob/main/bench/RERUN.md)
tracks which scripts are due for a re-run.

## Hardware notes

- All numbers above are **single CPU core** on Apple M-series hardware.
  Tengri runs on JAX, so the same code executes on GPU/TPU without
  modification — but those platforms have not been benchmarked. See the
  [JAX installation guide](https://docs.jax.dev/en/latest/installation.html)
  for GPU/TPU setup.
- JAX Metal (Apple GPU) is experimental and causes test failures; CPU
  is the supported reference platform for benchmarks. Set
  `JAX_PLATFORMS=cpu` to be explicit.
- Memory: smooth D = 7 fits run in ~100 MB; stochastic D = 137 in
  ~1.5 GB. NUTS warmup with `dense_mass=True` peaks 3–6× steady state
  on small models and can hit 20+ GB on D ≥ 8 with `dense_basis` SFHs;
  multi-fit notebooks need `dense_mass=False`. See
  [Memory expectations](memory.md) for the full table and the two
  recurring OOM patterns.

## When numbers look wrong

If `python -m tengri.bench` shows a much slower 1-galaxy timing than
the table above:

1. Confirm `x64: True` (some downstream behaviour assumes 64-bit).
2. Confirm `default device: cpu` — Metal sometimes silently picks
   itself up and slows things down. Force CPU with `JAX_PLATFORMS=cpu`.
3. Check the cache size — if it's in the GB range with hundreds of
   files, `tengri.clear_cache()` after a JAX upgrade is sometimes the
   fix.
4. The default `bench` SSP grid is whatever first matches `data/ssp_*.h5`;
   a multi-Z, full-α/Fe grid is meaningfully slower than the
   `prsc_miles` grid used in `notebooks/00_quickstart`. The relative
   numbers (vmap speedup, exact-vs-hybrid ratio) are what matters.
