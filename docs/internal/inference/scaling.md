# Population VI scaling

How `PopulationFitter`'s pure-JAX variational engines —
`native_vi_linear` (MGVI) and `native_vi_nonlinear` (geoVI) — scale in
**wall time**, **peak memory**, **convergence**, and **hyperparameter
recovery** as the catalog size $N$ and the forward-chunk size $K$ change.

The data is produced once by

```bash
JAX_PLATFORMS=cpu python bench/scripts/benchmark_vi_xlarge.py
```

and rendered by `analysis/render_vi_scaling.py`. The figures below are
checked into the repo so the docs build never has to re-run the
benchmark.

## Setup

* **Forward model.** Stochastic SFH: truncated-skew-normal SFR peak + 128-node
  correlated-field residual with PSD governed by hyperparameters
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

Memory is essentially flat in $N$ for `native_vi_linear` (5–8 GB ΔRSS
from $N=4$ to $N=8192$, well under the 30 GB budget). Wall time goes
linear in $N$ past $N \sim 128$ for both engines, with geoVI ~3× slower
per iteration than MGVI; both plateau at 6 iterations once $N \gtrsim 64$.

The headline scientific result is negative: with SDSS-only photometry,
the $\tau_{\rm PSD}$ marginal sits at ~150 Myr versus the injected truth
of 20 Myr — the posterior collapses to the prior. The $\sigma_{\rm PSD}$
marginal lands near truth only because its prior mean ($2.05$) happens to
match. The constraint half-width does not follow $1/\sqrt{N}$; it follows
the prior. **Photometric SDSS broadband data cannot constrain
stochastic-SFH PSD on Myr timescales.**

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
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_vi_xlarge.py

# Rich-obs run (the one that actually constrains σ):
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_vi_xlarge.py \
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
memory for parallelism.

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

Source: `bench/results/vi_scaling_benchmark.json`, MGVI rows at N=8192, CPU.

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

geoVI follows the same elbow; wall times are ~3× longer and memory
headroom tighter (Newton-CG dominates the graph). The K=32 sweet spot
holds.

## Spectroscopic run — covers Hα, Hβ, [OIII], 4000 Å break

Basic and rich runs are broadband-only. Even rich (FUV→Ks) cannot
constrain $\tau_{\rm PSD}$ on Myr timescales: every photometric band is
an *integral* over time-weighted SFR (UV: ~5–100 Myr, NIR: stellar mass).
Neither resolves the *correlation length* of SFH fluctuations. The Hα/UV
ratio pins $\tau_{\rm PSD}$: Hα probes ~10 Myr SFR, FUV ~100 Myr. Their
mismatch is the burstiness signal (Mehta+23, Asada+23, Faisst+19).

`PopulationFitter` supports `data_type="spectroscopy"`, which
self-consistently produces continuum *and* emission lines from a
single forward model:

1. Stellar SED → ionizing photon rate $Q_{\rm ion}$ from the <912 Å
   luminosity (cumulative integral, JIT-traceable).
2. Nebular backend (Cue/BPASS+CLOUDY) maps
   $(Q_{\rm ion}, \log U, \log Z_{\rm gas}) \to$ line luminosities.
3. Birth-cloud and diffuse dust attenuation applied to lines (HII
   region geometry → typically higher than continuum dust).
4. Continuum + lines combined, LSF convolved to the requested
   resolution.

The `--spec-obs` benchmark uses rest-frame 3000–7500 Å at $R \approx
500$ (covers Hβ, [OIII], Hα, and the 4000 Å break). It runs at K=1
only (spec memory ~13 GB at small N already; K>1 exceeds the 30 GB
budget) and N up to 1024 (per-cell wall time is ~5–10× longer than
photometry).

```{figure} ../../analysis/figures/vi_scaling_spec.png
:name: fig-vi-scaling-spec
:width: 100%
:align: center

Spectroscopic mode: 3000–7500 Å rest at R≈500, 5% noise. Hα, Hβ, [OIII],
and the 4000 Å break allow σ_PSD and τ_PSD posteriors to recover the
injected truth (σ=2, τ=20 Myr) in a way broadband photometry cannot. Run
with `bench/scripts/benchmark_vi_xlarge.py --spec-obs --noise-frac 0.05 --ks 1
--ns 4,8,...,1024`.
```

## Joint photometry + emission-line luminosity run

A lighter-weight alternative to a full optical spectrum is to use the
rich-obs photometry plus four integrated emission-line luminosities
(Hα, Hβ, [OIII] λ5007, [OII] λ3727) extracted directly from the wNE-SSP
spectrum at line centers (no separate Cue/CLOUDY backend — the emission
is already baked into the FSPS/MIST continuum). The 4-line + 10-band
joint vector is consumed by `PopulationFitter(data_type="photometry")`
via a monkey-patched `predict_photometry`.

```{figure} ../../analysis/figures/vi_scaling_joint.png
:name: fig-vi-scaling-joint
:width: 100%
:align: center

Joint mode: 10-band photometry + 4 emission-line luminosities (Hα, Hβ,
[OIII] 5007, [OII] 3727), 5% noise, MGVI K=1, N=4..512. **σ_PSD median
brackets truth (σ=2.0) at every N from 4 onward** — broadband alone lands
on truth by prior coincidence; joint mode lands there because the data
demands it. **The constraint width does not tighten with N**: σ half-width
plateaus ~0.85 (≈ prior width) across 7 doublings of N (bottom-right panel).
Adding more galaxies adds independent realizations of an information-saturated
posterior. The four lines fix σ on average but supply no extra independent
dimensions to shrink it. **τ_PSD remains pinned at the prior mean** (~150 Myr):
four scalar line fluxes cannot discriminate correlation length on Myr
timescales. Run with `bench/scripts/benchmark_vi_xlarge.py --joint-obs
--noise-frac 0.05 --ks 1 --ns 4,8,...,512`.
```

**Key takeaway from the joint experiment:** *getting the median right
is not the same as getting the uncertainty right.* All three modes
basic / rich / joint land on σ ≈ 2 at large N, but only because:

* basic and rich's prior mean happens to be near 2 (pure coincidence),
* joint actively pulls the median to truth via line-flux information.

In all three, the **posterior width stays at the prior width**. The
spec mode (or true time-domain data) is the only configuration that
can plausibly shrink uncertainty 1/√N.

## Caveats

* **Joint photometry + line-fluxes** for `PopulationFitter` is not yet
  wired — spec-mode covers the same physics by including Hα as
  part of the spectrum. A direct joint photometry+`LineFluxData` mode
  is a small (~30-line) extension to the per-galaxy predict path.
* **CPU only.** All benchmarks are CPU. The pure-JAX engines carry the same
  kernels to GPU; large $K$ gives far larger speedups there.
