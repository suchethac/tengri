# Quickstart: NUTS → HMC does not pay, unlike nb06/nb07

**Date:** 2026-08-17
**Verdict:** DO NOT MIGRATE. No HMC configuration beats NUTS on `00_quickstart`.
**Platform:** macOS, CPU (`JAX_PLATFORMS=cpu`), x64. Machine at load ~6 running
other suites, so **wall times are indicative; the ESS and R̂ columns are not.**

## Why this was tried

[`2026-05-06_compile_vs_sampling_breakdown.md`](2026-05-06_compile_vs_sampling_breakdown.md)
moved `06_fitting_spectroscopy` and `07_joint_photo_spec` from NUTS to
fixed-length HMC for 6.3× and 3.4×. Three notebooks never followed:
`00_quickstart`, `01_why_jax`, `05_fitting_photometry`. Finishing that
migration looked like the largest remaining win on the quickstart.

It is not a win. The earlier result does not transfer, and this report exists
so nobody re-derives the same wrong conclusion from it.

## Configuration

`00_quickstart`'s own model: 12 broadbands (GALEX/SDSS/2MASS/WISE), truncated
skew-normal SFH, two-component Calzetti dust, nebular off, *z* fixed at 0.05,
`approx=WavePrecomp()`. **D = 7.** Four chains, all seeded from the same MAP
point (`n_restarts=8, n_steps=800`). Adoption bar is the notebook's own claim:
**0 divergences and max split-R̂ < 1.01.**

> **Correction to this description, 2026-08-31 (#2096). The measurements below
> are unchanged and remain correct for the model they measured.** Two things
> about the paragraph above have stopped being true, and neither touches a
> number.
>
> 1. **This is no longer `00_quickstart`'s model.** #2044 (`36d7189cf`,
>    2026-08-23) rewrote the quickstart to a DPL SFH with a *single* Calzetti
>    screen, nebular on via the wNE grid, at **D = 6**. Read every row here as
>    the **pre-#2044** quickstart. It is the fixture now named `00` in
>    `bench/scripts/benchmark_notebook_sampler.py`; `00now` is today's.
> 2. **The script that produced this table had stopped building this model.**
>    `benchmark_quickstart_sampler.build_model` as committed had no `met` group,
>    so it built D = 6 — while the D = 7 stated here, and the `met_logzsol` named
>    as the L=160 row's worst-mixing parameter, both require free metallicity. A
>    parameter cannot be worst-mixing in a table produced by a model without it,
>    so the *builder* is what drifted after this report, not the table. Free
>    metallicity is restored in the `00` fixture and this table is reachable
>    again. The published numbers are not restated or adjusted.
>
> `tools/check_harness_parity.py` now fails when a fixture stops matching the
> notebook it names, so a third instance of this cannot pass unnoticed.

## Result

| config | wall s | max split-R̂ | divergences | min ESS | gradients / ESS | worst-mixing parameter |
|---|---:|---:|---:|---:|---:|---|
| **NUTS** (1500 warmup, 250 draws) | 137.1 | 1.0087 | 1 | **231.5** | variable | `sfh_tsnorm_skew` |
| HMC L=20, 600 draws | 15.6 | 1.0263 | 0 | 18.8 | 1698 | `sfh_tsnorm_skew` |
| HMC L=40, 600 draws | 54.7 | 1.0162 | 0 | 97.3 | **658** | `sfh_tsnorm_peak_lbt_gyr` |
| HMC L=80, 600 draws | 143.6 | 1.0082 | 0 | 116.0 | 1104 | `sfh_tsnorm_skew` |
| HMC L=160, 600 draws | 174.9 | 1.0012 | 0 | 266.3 | 961 | `met_logzsol` |

HMC at the shipped `_setup.HMC_VALIDATED` recipe (L=20) is **8.8× faster in
wall** — which is exactly the trap. It draws 600 samples of which **18.8 are
effective**, and misses the R̂ bar. Ranked on seconds per effective sample
rather than wall, NUTS wins outright: 0.59 s/ESS against HMC L=20's 0.83.

Longer trajectories do fix the mixing — L=160 reaches min ESS 266 and R̂
1.0012, beating NUTS on both — but costs 174.9 s against NUTS's 137.1 s. **The
only two configs that clear the bar are both slower than what they replace**,
and the one that clears it most cheaply (L=80) mixes half as well (ESS 116 vs
231).

## Why it transfers to nb06/nb07 and not here

The worst-mixing parameter is `sfh_tsnorm_skew`, a shape parameter of the
truncated skew-normal SFH that this mock constrains only weakly. NUTS spends
leapfrogs on such a direction *only when the trajectory needs them*; a fixed L
must either underspend — L=20 leaves that one parameter at ESS 18.8 while the
well-constrained ones are fine — or overspend on all seven directions at once,
which is what L=160 buys and why it costs more than NUTS.

Fixed-length HMC wins where the posterior's correlation length is roughly
uniform across directions, which is what nb06 and nb07 have and this does not.
Trajectory length is a property of the posterior, not of the notebook, so
"HMC was 6.3× on nb06" is not evidence about any other fit.

## What this means for the other two

`01_why_jax` and `05_fitting_photometry` were not measured. They share this
notebook's SFH family, so the same result is *likely* — but that is precisely
the extrapolation this report exists to warn against. Measure before switching.

## Reproduce

```bash
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_quickstart_sampler.py
```

Ranks on seconds per effective sample and prints the worst-mixing parameter per
configuration, because both wall time and mean ESS hide the failure mode here:
one weakly-identified direction dragging while everything else looks healthy.
