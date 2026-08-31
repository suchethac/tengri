# Performance guide

The forward model is pure JAX, so every backend (MAP, NUTS, geoVI, …)
runs against the same compiled graph. "How fast is tengri?" therefore
reduces to a small set of numbers that travel together.

Headline numbers (last full run August 2026) and how to reproduce them.
Scripts live under
[`bench/scripts/benchmark_*.py`](https://github.com/suchethac/tengri/tree/main/bench/scripts);
the single entry point is [Health check & dispatcher](#health-check-and-dispatcher).

```{warning}
Numbers below were measured August 2026 on JAX 0.11 / Apple M-series and may predate recent changes. Re-run the relevant script before citing in papers or PRs. Dates and full results: [`bench/reports/`](https://github.com/suchethac/tengri/tree/main/bench/reports).
```

## Headline numbers (Apple M-series CPU, x64, JAX 0.11, last run August 2026)

Forward photometric prediction on SDSS *ugriz* at z = 0.1, 5 bands,
running on a single CPU core (DPL parametric SFH, D=6):

| Configuration | Exact | WavePrecomp (precomputed) | Speedup |
|---|---:|---:|---:|
| Stellar only | 9.8 ms | 719 µs | 13.6× |
| + nebular (baked-in SSP) | 11.1 ms | 731 µs | 15.2× |
| + dust IR (THEMIS) | 11.4 ms | 794 µs | 14.3× |
| + radio + X-ray | 10.7 ms + 10.7 ms | (integrated) | 3.1–13.4× |
| **Typical: neb+THEMIS+radio+xray** | **12.6 ms** | **3.4 ms** | **3.7×** |
| **Kitchen sink (all emitters)** | **19.1 ms** | **12.5 ms** | **1.5×** |

The forward path is fixed at `SEDModel` construction:
**Exact** (`approx=None`, default) does full-wavelength SED + filter integration.
**WavePrecomp** (`approx=WavePrecomp()`) uses a precomputed SSP×filter LUT.

— *full table at [`bench/reports/2026-08-31_forward_model_speedup.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-08-31_forward_model_speedup.md)*

### AGN dense integrators: where precompute does not help

WavePrecomp delivers no speedup (K&D 3-zone: **1.0×**, SKIRTOR torus: **0.9×**)
because these AGN components require dense integration of the full-resolution SED
per call. They do not take the band-projection fast branch in
`observation/_band_projection.py` (#1022) — instead, every band integral is computed
by dense quadrature on the full wavelength grid, and the precompute LUT lookup
cost drowns any savings. When AGN-dominated models are slow, the precompute does
not help. Use exact mode, or trim the AGN complexity to composable (disc+torus)
or analytic (QSOgen) variants, which do benefit from precompute (2–2.1×).

Inference backends on a 7-parameter mock fit (compile + sample wall):

| Backend | First call | Steady-state |
|---|---:|---:|
| MAP (L-BFGS) | ~5 s | < 1 s |
| Laplace | ~5 s | < 1 s |
| Pathfinder | ~10 s | ~2 s |
| NUTS (1k samples) | ~30 s | ~5 s |
| `vi_nonlinear_fast` (geoVI, JAX-native) | ~10 s | **2.3 s** |
| `vi` (NIFTy.re) | ~75 s | 43.7 s |

— *full breakdowns: [`2026-04-17_native_vs_nifty.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-04-17_native_vs_nifty.md), [`2026-04-22_pathfinder_vs_window_nuts.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-04-22_pathfinder_vs_window_nuts.md), [`2026-05-06_compile_vs_sampling_breakdown.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-05-06_compile_vs_sampling_breakdown.md)*

`vi_nonlinear_fast` is **19–25× faster** than the NIFTy path on
smooth-SFH fits on some problems but may show differences in posterior geometry
on stochastic fits. The native backends (`vi_nonlinear_fast`, `vi_linear_fast`)
are optimized JAX implementations — validate per problem before swapping to ensure
posterior equivalence on your science case.

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
[Compilation: caching and diagnostics](compilation) for full details,
including how to trace what is recompiling and why.

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
| `forward_model` | Forward photometry: exact vs WavePrecomp across all emitters |
| `components` | Per-component (stellar, dust, nebular, AGN, ...) wall-clock timing |
| `jit_compile` | Population-scale JIT compile time vs N galaxies |
| `jit_real_path` | Compile time on the production forward-model path |
| `inference_engines` | MAP / Laplace / NUTS / VI / NSS at D = 7, 12, 20 |
| `vi_native_vs_nifty` | geoVI: pure-JAX `vi_nonlinear_fast` vs the NIFTy.re reference path |
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

- All numbers above are **single CPU core** on Apple M-series. See
  [JAX installation](https://docs.jax.dev/en/latest/installation.html) for setup.
- **CUDA GPUs are benchmarked** as of 2026-08-20, on an RTX 3060 against a Ryzen 9
  5900X: see
  [`bench/reports/2026-08-20_cuda_device_matrix.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-08-20_cuda_device_matrix.md)
  and `notebooks/nvidia_cuda.py`. Nothing needs changing to run on CUDA, and
  float64 results are bit-comparable with the CPU — but the GPU is a *width*
  instrument. One galaxy: the CPU wins by 33x (forward) and 13x (gradient), and a
  single MAP fit by 8.8x. The crossover is between 128 and 512 galaxies; at 2048 the
  GPU leads by 4.3x (forward) to 14.7x (gradient, float32). tengri's forward model
  runs at ~0.12 FLOP/byte, so the card is waiting on memory and dispatch, not
  arithmetic. Note also that consumer GeForce cards run float64 at 1/64 rate, which
  puts this GPU *below* this CPU on dense float64 arithmetic.
- JAX Metal (Apple GPU) is experimental and causes test failures. CPU is the
  reference platform. Set `JAX_PLATFORMS=cpu` explicitly.
- **Memory:** D = 7 smooth fits ~100 MB; D = 137 stochastic ~1.5 GB. NUTS
  warmup with `dense_mass_matrix=True` peaks 3–6× steady state; can hit 20+ GB on D
  ≥ 8 with `dense_basis` SFHs. Multi-fit notebooks need `dense_mass_matrix=False`.
  See [Memory expectations](memory.md).

## When numbers look wrong

If `python -m tengri.bench` shows slower 1-galaxy timing than the table:

1. Confirm `x64: True`.
2. Confirm `default device: cpu`. Metal sometimes silently activates; force
   CPU with `JAX_PLATFORMS=cpu`.
3. Check cache size. If in the GB range, try `tengri.clear_cache()` after JAX upgrade.
4. The default SSP grid is the first `data/ssp_*.h5` found. Multi-Z,
   full-α/Fe grids are slower than `prsc_miles`. The relative numbers (vmap
   speedup, exact-vs-hybrid ratio) matter most.
