# bench/ — performance scripts, reports, and raw results

Performance data organized as:

```
bench/
├── README.md       — you are here
├── scripts/        — runnable Python benchmarks
├── reports/        — dated, human-readable results (Markdown)
└── results/        — machine-readable raw outputs (JSON)
```

User-facing summary:
[`docs/performance/index.md`](../docs/performance/index.md). This README is
for contributors: benchmark placement, where results come from, and tidy
conventions.

## scripts/

| Script | What it measures |
|---|---|
| `benchmark_device_matrix.py` | CPU vs GPU, float64 vs float32, across prediction, gradients and inference |
| `benchmark_forward_model.py` | Forward photometry: exact / compositional / hybrid across all emitters and 3 SFH families |
| `benchmark_components.py` | Per-component (stellar, dust, nebular, AGN, ...) wall-clock timing |
| `benchmark_jit_compile.py` | Population-scale JIT compile time vs N galaxies, batching strategies |
| `benchmark_jit_real_path.py` | Compile time on the production forward-model path |
| `benchmark_inference_engines.py` | MAP / Laplace / NUTS / VI / NSS at D = 7, 12, 20 |
| `benchmark_vi_native_vs_nifty.py` | geoVI: pure-JAX `native_vi_nonlinear` vs the NIFTy.re reference path |
| `benchmark_vi_xlarge.py` | VI scaling on stochastic-SFH problems with D >> 100 |
| `benchmark_population_native.py` | Hierarchical PopulationFitter: per-iteration cost vs N galaxies |
| `benchmark_catalog_throughput.py` | Catalog MCMC: galaxies/s vs `forward_chunk_size`, sampler, precision and devices — with R-hat / ESS on every row |
| `benchmark_adam_vs_lbfgs.py` | MAP optimizers head-to-head |
| `benchmark_cue.py` | Cue (Li+2025) nebular emulator timing in isolation |
| `benchmark_loss_timing.py` | Per-call loss / negative-log-posterior timing |
| `benchmark_joint_indices_e2e.py` | End-to-end timing for joint photometry + spectral indices |
| `benchmark_precompute_analytic.py` | Analytic precompute lookup vs full-spectrum integration |
| `benchmark_precompute_quad.py` | Quadrature precompute: accuracy vs grid resolution |
| `benchmark_ztable_interp.py` | Metallicity-table interpolation kernel timing |
| `benchmark_dust_laws.py` | Dust-law variants on a common forward-model harness |
| `bench_pathfinder_warmstart.py` | Pathfinder as a warm-start for NUTS |

Run a script directly:

```bash
JAX_PLATFORMS=cpu python bench/scripts/benchmark_forward_model.py
```

…or via the consolidated dispatcher:

```bash
python -m tengri.bench list                          # see all
python -m tengri.bench help forward_model            # script docstring
python -m tengri.bench forward_model                 # run it
```

The dispatcher catalogue lives in
[`src/tengri/bench/__init__.py`](../src/tengri/bench/__init__.py). New
scripts must be added there too — one line per script — or the
dispatcher won't find them.

## reports/

Markdown write-ups, one file per benchmark *event*. File names are
`YYYY-MM-DD_<short-topic>.md`, the date is when the benchmark was run.

| File | Topic | Last run | Status |
|---|---|---|---|
| `2026-04-17_native_vs_nifty.md` | First geoVI native-vs-NIFTy comparison | 2026-04-17 | likely stale (forward-model refactor since) |
| `2026-04-22_pathfinder_vs_window_nuts.md` | Pathfinder warm-start for NUTS | 2026-04-22 | likely stale |
| `2026-04-24_population_native_vs_nifty.md` | Hierarchical fits, native vs NIFTy | 2026-04-24 | likely stale |
| `2026-05-06_compile_vs_sampling_breakdown.md` | Compile time vs sample time across backends | 2026-05-06 | recent — probably current |
| `2026-05-06_forward_model_speedup.md` | Hybrid path: 30-400× over exact across all emitters | 2026-05-06 | recent — source of headline tables in `docs/performance/index.md` |
| `2026-05-06_notebook_renewal.md` | Spine-notebook compile and run wall-clock survey | 2026-05-06 | recent — but post-dates this round's notebook prose edits, so SFH-fan and 2-D grid timings need a re-run |

Anything older than ~2 weeks should be assumed stale until re-verified;
forward-model and inference internals move quickly.

Conventions:

- One report per benchmark *intent*, not per script run. If you re-run
  the same benchmark on a new platform, append a section to the existing
  report rather than creating a new file.
- Always include hardware (CPU model, RAM, JAX version, x64/x32) at the
  top.
- The headline numbers in `docs/performance/index.md` cite back to
  reports here — keep that document in sync when a new run lands.

## results/

Raw JSON outputs that `bench/scripts/*.py` write before producing
plots or summary stats.

```
bench/results/
├── jit_compile_benchmark.json
├── jit_real_path_benchmark.json
├── orchestrator_jit_benchmark.json
├── vi_scaling_benchmark.json
├── vi_scaling_benchmark_joint.json
├── vi_scaling_benchmark_pre_vmap.json
├── vi_scaling_benchmark_rich.json
└── vi_scaling_benchmark_spec.json
```

These are not human-readable but are useful when re-rendering figures
(see `analysis/render_vi_scaling.py`). They are committed to the repo
because regenerating them takes hours; treat them as expensive build
artefacts.

## Adding a new benchmark

1. Add `benchmark_<topic>.py` under `bench/scripts/`. Top docstring answers
   "what does this measure?" (one paragraph); downstream tools (`bench help
   <name>`) read it.
2. Register in `BENCHMARK_SCRIPTS` (`src/tengri/bench/__init__.py`) so the dispatcher finds it.
3. Write the human report to `bench/reports/YYYY-MM-DD_<topic>.md` and raw
   JSON to `bench/results/`.
4. If numbers change headline claims, update `docs/performance/index.md`.
