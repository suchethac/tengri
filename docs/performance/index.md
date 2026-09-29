# Performance guide

The forward model is pure JAX, so every backend (MAP, NUTS, geoVI, …) runs against the same compiled graph. The speed of tengri therefore reduces to a small set of numbers that travel together.

Headline numbers (last full run August 2026) and how to reproduce them live under
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
| + radio (SF + AGN) | 10.9 ms | 809 µs | 13.4× |
| + X-ray (XRB + corona) | 10.7 ms | 3.4 ms | 3.1× |
| **Typical: neb+THEMIS+radio+xray** | **12.6 ms** | **3.4 ms** | **3.7×** |
| **Kitchen sink (all emitters)** | **19.1 ms** | **12.5 ms** | **1.5×** |

The forward path is fixed at `SEDModel` construction:
**Exact** (`approx=None`, default) does full-wavelength SED plus filter integration. **WavePrecomp** (`approx=WavePrecomp()`) uses a precomputed SSP×filter LUT. Construction defaults to exact, but since 2026-08-10, every fit surface (`Fitter`, `PopulationFitter`, `CatalogFitter`) resolves `approx="auto"` to the LUT at fit time. Pass `approx=None` to a fitter to force the exact path. The full table is at [`bench/reports/2026-08-31_forward_model_speedup.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-08-31_forward_model_speedup.md).

### AGN dense integrators: where precompute does not help

WavePrecomp delivers no speedup (K&D 3-zone: **1.0×**, SKIRTOR torus: **0.9×**) because these AGN components require dense integration of the full-resolution SED per call. They skip the band-projection fast branch in `components/_band_projection.py` (#1022), so every band integral must be computed by dense quadrature on the full wavelength grid. The precompute LUT lookup cost drowns any savings, so when AGN-dominated models are slow, the precompute does not help. Use exact mode, or trim AGN complexity to composable (disc+torus) or analytic (QSOgen) variants, which do benefit from precompute (2–2.1×).

Inference backends on a 7-parameter mock fit (compile + sample wall):

| Backend | First call | Steady-state |
|---|---:|---:|
| MAP (L-BFGS) | ~5 s | < 1 s |
| Laplace | ~5 s | < 1 s |
| Pathfinder | ~10 s | ~2 s |
| NUTS (1k samples) | ~30 s | ~5 s |
| `vi_nonlinear_fast` (geoVI, NIFTy fast path) | ~10 s | **2.3 s** |
| `vi` (NIFTy.re) | ~75 s | 43.7 s |

Full breakdowns: see [`2026-04-17_native_vs_nifty.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-04-17_native_vs_nifty.md), [`2026-04-22_pathfinder_vs_window_nuts.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-04-22_pathfinder_vs_window_nuts.md), and [`2026-05-06_compile_vs_sampling_breakdown.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-05-06_compile_vs_sampling_breakdown.md).

`vi_nonlinear_fast` is **19–25× faster** than the full NIFTy path on smooth-SFH fits on some problems, but may show differences in posterior geometry on stochastic fits. These backends (`vi_nonlinear_fast`, `vi_linear_fast`) are the NIFTy geoVI/MGVI paths with Python-level logging overhead removed. Validate per problem before swapping to ensure posterior equivalence on your science case.

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

Run one command to quickly read your install:

```bash
python -m tengri.bench
```

This prints the JAX backend, default device, persistent compile-cache size, and a 1-galaxy and 100-galaxy timing on SDSS *ugriz*. Expect about 30 seconds on CPU after the cache is warm.

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
- **CUDA GPUs are benchmarked** as of 2026-08-20 on an RTX 3060 against a Ryzen 9
  5900X; see
  [`bench/reports/2026-08-20_cuda_device_matrix.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-08-20_cuda_device_matrix.md)
  and `notebooks/nvidia_cuda.py`. Nothing needs changing to run on CUDA, and
  float64 results are bit-comparable with the CPU. The GPU is a *width* instrument. On one galaxy, the CPU wins by 33× (forward) and 13× (gradient), and a single MAP fit by 8.8×. The crossover is between 128 and 512 galaxies; at 2048 the GPU leads by 4.3× (forward) to 14.7× (gradient, float32). Because tengri's forward model runs at ~0.12 FLOP/byte, the card waits on memory and dispatch, not arithmetic. Consumer GeForce cards run float64 at 1/64 rate, which puts this GPU below this CPU on dense float64 arithmetic.
- Apple's own `jax-metal` (0.1.1, 2024-10) is not viable against this JAX version, so CPU is the reference platform here. Set `JAX_PLATFORMS=cpu` explicitly. For the Apple GPU, the community `jax-mps` plugin is the supported path. See `notebooks/apple_mps.py` for the install recipe and measured throughput, and `bench/scripts/benchmark_float32_mps_parity.py` for the float32 accuracy check.
- **Pure float32** (`JAX_ENABLE_X64=0` before Python starts) is supported end-to-end as of 2026-09 (#1206). The full panchromatic model and the default photometry plus emission-line fit converge to the float64 optimum to ~1e-5 on CPU and CUDA, and to ~4e-5 on Apple GPU via `jax-mps`. This is a *memory* knob, not a clock: 2.02× galaxies per GiB at batch 8192 on the fitting path, 1.25× at 2048, nothing at one galaxy. See `docs/dev/float32-tier-b-boundary.md` for measurements and the acceptance criterion.
- **Memory:** D = 7 smooth fits ~100 MB; D = 137 stochastic ~1.5 GB. NUTS warmup with `dense_mass_matrix=True` peaks 3–6× steady state and can hit 20+ GB on D ≥ 8 with `dense_basis` SFHs. Multi-fit notebooks need `dense_mass_matrix=False`. See [Memory expectations](memory.md).

## When numbers look wrong

If `python -m tengri.bench` shows slower 1-galaxy timing than the table:

1. Confirm `x64: True`.
2. Confirm `default device: cpu`. Metal sometimes silently activates; force
   CPU with `JAX_PLATFORMS=cpu`.
3. Check cache size. If in the GB range, try `tengri.clear_cache()` after JAX upgrade.
4. The default SSP grid is the first `data/ssp_*.h5` found. Multi-Z,
   full-α/Fe grids are slower than `prsc_miles`. The relative numbers (vmap
   speedup, exact-vs-precomp ratio) matter most.
