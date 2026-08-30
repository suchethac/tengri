# Benchmarks to rerun

A running list of benchmark events that should be redone before their
numbers are quoted again. Sorted by priority. Update this file when you
re-run something (delete or move to "done").

## High priority

- [ ] **`forward_model`** — last clean run 2026-05-06; the headline table
  in `docs/performance/index.md` cites it. Forward-model precompute
  internals have moved since (see PRs touching
  `src/tengri/forward/precompute/` and AGN/dust components). One CPU
  run takes ~3 min after warmup.

  ```bash
  JAX_PLATFORMS=cpu python -m tengri.bench forward_model
  ```

- [ ] **`inference_engines`** — D = 7 / 12 / 20 wall-clocks for
  MAP/Laplace/NUTS/VI/NSS. The "first call" / "steady-state" inference
  table on `docs/performance/index.md` is interpolated from older data;
  needs a fresh end-to-end pass.

  ```bash
  JAX_PLATFORMS=cpu python -m tengri.bench inference_engines
  ```

## Medium priority

- [ ] **`vi_native_vs_nifty`** — last comparison was 2026-04-17 (well
  before the SFH-registry rework and the dust-IR component split). The
  19–25× speedup claim should be verified on the current code.

- [ ] **`population_native`** — hierarchical PopulationFitter timing.
  Last run 2026-04-24, before block-Gibbs convergence fixes.

- [ ] **`jit_compile`** — population-scale JIT compile-time scan vs N
  galaxies. The persistent compile cache may have grown / shrunk
  since the last run.

## Low priority (mostly stable)

- [ ] **`components`** — per-component timings. Useful to re-run when
  any component's internal kernel changes.
- [ ] **`precompute_analytic`** / **`precompute_quad`** — accuracy /
  cost trade-offs for precompute kernels. Re-run when precompute
  registry changes.
- [ ] **`adam_vs_lbfgs`** — MAP optimizer head-to-head; numbers move
  only when optax versions change.
- [ ] **`cue`** — Cue emulator timing in isolation. Stable unless
  Cue weights are repacked.
- [ ] **`ztable_interp`** — metallicity-table interp kernel. Stable.
- [ ] **`loss_timing`** — log-posterior timing. Stable unless the
  parameter set changes.

## Capacity-permitting

These are full-fat runs that take meaningful wall time:

- [ ] **`vi_xlarge`** — VI scaling on D >> 100 stochastic-SFH problems.
  ~30 min, GB of RAM, single CPU.
- [ ] **`benchmark_dust_laws`** — full forward-model pass per attenuation
  law. ~5 min.

## After re-running

When a benchmark is rerun:

1. Append (or replace) the relevant section in `bench/reports/<date>_<topic>.md`
   with the new numbers and the new hardware/JAX-version metadata.
2. If the new numbers materially differ, update
   `docs/performance/index.md`. If they don't, just bump the "last run"
   date in `bench/README.md`'s status table.
3. Move the entry from this file into a "done" section at the bottom or
   delete it.

## Done

- [x] **`catalog_throughput`** — first GPU run, 2026-08-30, RTX 3060 12 GB
  (GA106) against a Ryzen 9 5900X control. The script had never been run on
  an accelerator and had no committed result; it now has
  [`bench/reports/2026-08-30_gpu_catalog_throughput.md`](reports/2026-08-30_gpu_catalog_throughput.md)
  and `bench/results/gpu_catalog_throughput.json`. The run added a
  `--dtype f32|f64` axis, a `--method` axis, a `--max-doublings` axis, and
  R-hat / ESS / divergence columns on every row.

  ```bash
  # the whole campaign is in the report's Reproduce section; the headline cell:
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python bench/scripts/benchmark_catalog_throughput.py \
      --method mcmc_hmc --dtype f32 --n-gal 512 --chunk 512 \
      --warmup 400 --burnin 0 --samples 500 \
      --json bench/results/gpu_catalog_throughput.json --tag rtx3060
  ```

  The headline is **304 galaxies/GPU-minute raw, 222 of them clearing max
  split-R-hat < 1.01, and none of them usable** (min ESS 2.6 of 500 draws among
  exactly those). Every `mcmc_nuts` cell timed out. Measured on `main` at
  `fe6bda468`, i.e. after #2090.

  **Re-run when** any of these move, because each one invalidates the table:
  the catalog MCMC engine (`inference/backends/mcmc/catalog.py`), the
  `DEFAULT_MAP_INIT_STEPS = 300` warm start, `mcmc_hmc`'s default
  `n_leapfrog_steps = 10`, `WavePrecomp`'s default `band_integration`, or the
  blackjax version. Note the numbers are for the
  benchmark's own D = 3 dpl fixture at SNR 20 — they are a throughput
  characterization of the *machine*, not a convergence claim about tengri,
  and the report says so at length.
