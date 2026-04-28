# Population VI scaling

This page documents how `PopulationFitter`'s pure-JAX variational engines —
`native_vi_linear` (MGVI) and `native_vi_nonlinear` (geoVI) — scale in
**wall time**, **peak memory**, **convergence**, and **hyperparameter
recovery** as the catalog size $N$ and the forward-chunk size $K$ change.

The data is produced once by

```bash
JAX_PLATFORMS=cpu python scripts/benchmark_vi_xlarge.py
```

and rendered by `analysis/render_vi_scaling.py`. The figures below are
checked into the repo so the docs build never has to re-run the
benchmark.

## Setup

* **Forward model.** Stochastic SFH = `tsnorm + field`: a parametric
  truncated-skew-normal SFR peak plus a 128-node correlated-field
  residual whose PSD is governed by hyperparameters
  $(\sigma_{\rm PSD}, \tau_{\rm PSD})$.
* **Per-galaxy free parameters** (10): peak SFR, peak lookback time,
  width, skew, truncation, $\sigma_{\rm PSD}$, $\tau_{\rm PSD}$,
  $\log Z/Z_\odot$, $\tau_{\rm BC}$, $\tau_{\rm diff}$.
* **Truth injected per galaxy.** $\sigma_{\rm PSD} = 2.0$,
  $\tau_{\rm PSD} = 20$ Myr, all other params drawn from the prior.
* **Population-shared hyperparameters.** $\sigma_{\rm PSD}$ and
  $\tau_{\rm PSD}$ are promoted to *shared* by `PopulationFitter`.
* **Observation (basic run).** SDSS $u, g, r, i, z$ at 10% noise
  (5 bands).
* **Observation (rich run).** GALEX FUV/NUV + SDSS $ugriz$ + 2MASS
  $JHK_s$ at 5% noise (10 bands). UV constrains recent SFR; NIR pins
  stellar mass; together they carry far more PSD information than
  optical-only SDSS.
* **Inference.** $K$ is the `forward_chunk_size` passed to
  `PopulationFitter.run`. $K=1$ streams one galaxy at a time through
  `lax.map` (memory $\mathcal{O}(1)$ in $N$ for the linear engine);
  $K>1$ vmaps $K$ galaxies per inner step.
* **Convergence policy.** Both engines stop early via `kl_rtol=1e-2`.
  The benchmark uses an iteration cap of 50 (auto-retry at 100 if the
  cap is hit). A row is *converged* iff `iters_used < cap`.

## Basic run — SDSS only

```{figure} ../../analysis/figures/vi_scaling.png
:name: fig-vi-scaling-basic
:width: 100%
:align: center

Population VI scaling on 5-band SDSS photometry at 10% noise.
**Top**: warm wall-time, peak ΔRSS, and VI iterations vs $N$, one line
per (method, $K$). **Bottom**: $\sigma_{\rm PSD}$ and $\tau_{\rm PSD}$
posterior medians (with 68% bands) vs $N$ at $K=1$, plus the
$\sigma_{\rm PSD}$ 68% half-width vs $N$ against a $1/\sqrt{N}$
reference.
```

**What to take away from the basic run**:

* **Memory.** `native_vi_linear` peak ΔRSS is essentially flat in $N$
  across three orders of magnitude (5–8 GB band from $N=4$ to
  $N=8192$). Memory budget at 30 GB is never approached.
* **Wall time.** Linear in $N$ past $N \sim 128$ for both engines;
  geoVI is $\sim 3\times$ slower per iteration than MGVI.
* **Iterations.** Both engines plateau at 6 iterations once $N \gtrsim
  64$ — the population-shared posterior tightens and stops moving.
* **Hyperparameter recovery (red flag).** With SDSS-only photometry
  the $\tau_{\rm PSD}$ marginal sits at $\sim 150$ Myr, *not* at the
  injected truth of 20 Myr — i.e. the posterior is the prior. The
  $\sigma_{\rm PSD}$ marginal lands on truth only because its prior
  mean ($2.05$) happens to be near truth. **Photometric SDSS
  broadband data carries almost no information about stochastic-SFH
  PSD on Myr timescales.** The constraint half-width does not follow
  $1/\sqrt N$ — it follows the prior.

## Rich run — FUV/NUV + SDSS + JHK at 5% noise

```{figure} ../../analysis/figures/vi_scaling_rich.png
:name: fig-vi-scaling-rich
:width: 100%
:align: center

Same benchmark with 10-band photometry (FUV, NUV, $ugriz$, $JHK_s$)
at 5% noise. UV anchors recent SFR (5–100 Myr); NIR anchors stellar
mass. With this information budget the $\sigma_{\rm PSD}$ posterior
visibly tightens with $N$ and tracks the $1/\sqrt{N}$ reference; the
$\tau_{\rm PSD}$ marginal is partly recovered (still
prior-dominated at large $\tau$ because broadband photometry is
agnostic to correlation length once $\tau$ is short compared to
filter age-sensitivity widths).
```

## Reproducing

```bash
# Basic SDSS-only run, full K sweep:
JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py

# Rich-obs run (the one that actually constrains σ):
JAX_PLATFORMS=cpu .venv/bin/python scripts/benchmark_vi_xlarge.py \
    --rich-obs --noise-frac 0.05 --ks 1

# Render figures:
.venv/bin/python analysis/render_vi_scaling.py
.venv/bin/python analysis/render_vi_scaling.py --rich-obs
```

The driver is **idempotent** — cached `(method, N, K)` cells are
skipped instantly. Pass `--force` to overwrite cached cells, or
restrict with `--ns 16384,32768 --ks 1` to backfill specific cells.
The `--watch` mode of the renderer regenerates the figure whenever the
JSON file changes, which is useful during long sweeps.

## Choosing `forward_chunk_size`

`forward_chunk_size` (`K`) is a runtime tuning knob, not a physical
parameter — it controls how many galaxies are vmapped per inner
`lax.map` step in the catalog VI engine. K=1 streams one galaxy at a
time (memory $\mathcal{O}(1)$ in N for `native_vi_linear`); K>1 trades
memory for chunk parallelism.

The library default is **K=1** because it is the only choice that's
universally safe across all (N, hardware) combinations. The empirical
sweet spot is **K=32** *if* N is large and memory permits. Past K=32
the speedup saturates; past K=64 it regresses.

| K | Warm wall-time at N=8192 (MGVI) | Speedup vs K=1 | Peak ΔRSS |
|---|---------------------------------|----------------|-----------|
| 1 | 419 s | — | 5–6 GB |
| 2 | 417 s | ≈0% | ~6 GB |
| 4 | 388 s | ~7% | ~7 GB |
| 8 | 322 s | ~23% | ~8 GB |
| 16 | 272 s | ~35% | ~9 GB |
| **32** | **208 s** | **~50%** | **~10 GB** |
| 64 | 207 s | ~50% (saturated) | ~11 GB |
| 128 | 229 s | ~45% (regression) | ~12 GB |

Source: `data/vi_scaling_benchmark.json`, MGVI rows at N=8192, CPU.

**How to read this:**

* **Below N≈512**, every K column ties — the per-iteration forward
  cost is too small for chunk parallelism to overcome `lax.map(K=1)`'s
  low overhead. Use K=1.
* **At N=512–4096**, K=8–16 starts winning measurably. Pick whichever
  fits your memory budget.
* **At N≥4096**, K=32 is the elbow of the speedup curve — ~50% faster
  than K=1, double the memory footprint, smallest HLO of the
  high-throughput options. Recommended for large catalog runs.
* **Past K=32** XLA's auto-fusion saturates the available CPU SIMD /
  cache width. K=64 ties K=32; K=128 spends extra compile and memory
  for slightly *worse* throughput (cache-tile thrashing).

The K=128 regression is reproducible and is the practical reason the
library does not pick a high K by default — there's no "always-better"
choice.

**Recommendation table:**

| Catalog size | Memory budget | Recommended K |
|--------------|---------------|---------------|
| N ≤ 256 | any | 1 |
| 256 < N ≤ 1024 | ≤ 8 GB | 1–4 |
| 256 < N ≤ 1024 | ≥ 16 GB | 8–16 |
| N > 1024 | ≤ 8 GB | 4 |
| N > 1024 | 16 GB | 16 |
| N > 1024 | ≥ 32 GB | 32 |

geoVI follows the same shape but with longer absolute wall-times and
slightly tighter memory headroom (Newton-CG inner solve dominates
graph size). The K=32 elbow is the same.

## Caveats

* **What's not yet measured here.** Joint photometry + emission-line
  inference. `PopulationFitter` currently restricts `data_type` to
  `"photometry"` or `"spectroscopy"`; adding Hα or other line fluxes
  per galaxy needs a small extension to the per-galaxy predict path
  in `tengri.inference.hierarchical`. With Hα directly fit, the
  $\tau_{\rm PSD}$ posterior should additionally tighten on
  short-timescale recent burstiness — see {doc}`../advanced/hierarchical`.
* **CPU only.** All numbers above are on CPU. The pure-JAX engines
  carry the same kernels to GPU; chunk parallelism (large $K$) gives
  far larger speedups there.
