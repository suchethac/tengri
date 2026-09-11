# Marginalize the mass, and a photometry posterior costs 15-20 s on eight CPU cores; the metric was never the problem

**Date:** 2026-09-11
**Verdict:** **Reachable on the non-degenerate fixtures, not yet at the strict
bar.** With the total-stellar-mass amplitude integrated out analytically
(`profile_mass`), a dense window adaptation of 150 steps, and 4 chains × 300
draws with the chains `pmap`ped across four CPU devices, every one of twelve
seeds on `ctl-dpl` (D = 8) and `ctl-jwst` (D = 9) fits in **9.5-17.2 s** wall
(MAP + warmup + sampling, warm compile) — **11 of 12 at min ESS ≥ 100**, split
R-hat **1.009-1.043**, no frozen lane — on a machine that was running three
other sweeps and a test suite at the time. One of the twelve cleared the
strict R-hat < 1.01 bar in 1200 draws (`ctl-dpl` seed 12: 15.7 s, ESS 263,
R-hat 1.0088); the rest sit at 1.013-1.043. The degenerate `tsnorm` fixture
(`05`) is just as fast and does not mix on its shape parameters, as every
report since 2026-08-20 has said.

**The whole gain is the parameterization, not the sampler.** Before profiling,
the best metric on this posterior — a dense window adaptation the shipped
auto-policy disables at D ≥ 8 and the 2026-09-06 campaign never ran — costs
**66-228 s** per galaxy, converges on **zero of six** seeds, and **freezes one**
(seed 10: unique-draw fraction 0.089, 411 divergences, R-hat 3.51). The
diagnostic that explains it: in Laplace-whitened coordinates the three
*stiffest* eigen-directions of the posterior have std **2.3, 2.6 and 10.0**
against a Gaussian ideal of 1 — the mass-age direction is a curved ridge whose
curvature at the MAP over-constrains it 10×. Photometry is exactly linear in
mass (`max|flux ratio − 10| = 1.8e-13` per dex), so with a Gaussian likelihood
the mass has a closed-form conditional and integrates out with a 48-node
quadrature and **no extra forward-model evaluation**. The Hessian condition
number drops **3.7e4 → 1.2e3**, every seed needs **3-7× fewer gradients**, and
the frozen galaxy samples (unique fraction 1.000, truth recovered at 0.2σ).

**The mass marginal is exact in practice, not only in algebra.** On every seed
where the joint (unprofiled) run converged, the profiled run's marginal mass
agrees with it to **≤ 0.16σ in the mean and ~5 % in width**; on the seed where
the joint run was frozen, they disagree — the joint "posterior" has std
0.0054, the frozen chain's artefact, and the profiled one 0.032 with the truth
inside it.

**Three refutations on the way, each measured.** A *fixed* dense Laplace metric
(negative Hessian at the MAP) is **worse** than the diagonal control: 668 s,
370 gradients/draw, R-hat 1.09 against 247 s / 131. The warmup tree-depth cap
that bought the diagonal metric 3.2× does **not compose** with the dense one
(adaptation 52 → 6 s, but gradients/draw 34 → 67 and ESS 83 → 50; net ~10 %).
And the GPU is refuted for a single galaxy at every batch size tried: the
same ~38 k gradients that take 25 s on two CPU cores take **222 s** on the RTX
3060 at 4 chains (170 gradients/s — the ~8 ms kernel-launch floor per
leapfrog), and 256 vmapped chains run in lock-step to depth-10 trees at
**1900 gradients/s, slower than two CPU cores**.

**One correction to the shipped defaults that stands on its own:** the MAP
default (Adam, and 8 × 800 steps in every benchmark harness) **does not
converge** — 2.9 s warm to neg-log-posterior 6.33, against scipy L-BFGS-B's
6.0008 in **0.42 s** from one start; on `ctl-jwst` the Adam point has a
negative Hessian eigenvalue. L-BFGS is now the MAP default (commit
`f62d94576` on this branch).

**Platform:** Linux 6.8, AMD Ryzen (24 logical / 12 physical cores), NVIDIA
RTX 3060 12 GB. JAX 0.11.0, BlackJAX 1.6.2, float64 throughout (see Finding 8
for float32). Branch `feat/laplace-metric-nuts` off `main` at `f26f739c2`.

**Load.** Every wall clock below was taken on a shared box under 6-14
concurrent fits plus test runs; the earlier campaigns measured a 9.5×
scheduling spread on this machine. The one series taken one-fit-at-a-time is
labelled *idle* in Finding 5, and its sampling phase ran at 1900 gradients/s,
the two-core ceiling. **Gradients lead and seconds follow** in every table;
the gradient, ESS, R-hat, divergence and unique-fraction columns are
deterministic given the seed and reproduced to the printed digit across
reruns (e.g. `ctl-dpl` seed 7 profiled: 42 805 gradients, ESS 82.8, R-hat
1.0558 in three separate processes).

**Fixtures** (`bench/scripts/benchmark_notebook_sampler.py`'s registry,
dimensions read off the built model):

| fixture | D | bands | SFH | what it is |
|---|---|---|---|---|
| `ctl-dpl` | 8 | 14 | DPL | nb05's bands and dust over a DPL SFH; the non-degenerate control |
| `ctl-jwst` | 9 | 19 | continuity | 19 JWST bands at z = 1.5; the healthy control |
| `05` | 8 | 14 | tsnorm | `05_fitting_photometry` as shipped; the degenerate reference |

Six seeds per fixture (7-12; `ctl-jwst` 4-9), one fit per subprocess, each
seed a different mock galaxy.

## Why this was measured

The requirement: a single-galaxy photometry posterior in under **20 s**, the
20 s covering MAP + warmup + sampling on a warm compile (agreed with the user
at the start), on CPU and/or GPU. `bench/reports/2026-09-06_photometry_20s.md`
closed at **635 s** (32× over) after a 120-fit sweep of warmup caps and
half-strength whitening, and its 2026-09-11 amendment (#2252) records that it
never ran a dense mass matrix. This report starts from what that campaign's
own numbers implied and it did not test: 100-350 gradients per draw at D = 8
is a geometry problem, not a sampler-settings problem.

## How a row becomes a claim

Same rules as the 2026-09-06 report. **Every row carries max split R-hat (over
2 × n_chains halves), min ESS with the worst parameter named, divergences and
unique-draw fraction together**; none is read alone (#1999: a frozen chain can
report zero divergences and R-hat near 1). The bar is **min ESS ≥ 100 and
max split R-hat < 1.01 on every seed**; a cell missing it on any seed is `>=`
and its projection is a lower bound. ESS is over all parameters *including the
post-hoc mass draws* when the mass is profiled. Walls are `map + adapt +
sample`, each the second (warm) call. Rows: `bench/results/2026-09-11_laplace_nuts_20s_cpu.jsonl`
(119), `..._gpu.jsonl` (3), scored by `bench/scripts/score_laplace_nuts_20s.py`
into `..._scored.json`.

## Finding 1 — the shipped MAP is not converged, and L-BFGS is 7× faster to a better optimum

`ctl-dpl` seed 7, warm walls, two CPU cores:

| MAP | wall | neg-log-posterior |
|---|---|---|
| Adam 1 × 100 | 0.25 s | 115.5 |
| Adam 1 × 300 | 0.39 s | 7.88 |
| Adam 8 × 300 | 1.49 s | 7.03 |
| Adam 8 × 800 (the harness's call) | 2.91 s | 6.33 |
| **scipy L-BFGS-B, 1 start** (`optimizer="lbfgs"`, already shipped) | **0.42 s** | **6.0008** |
| scipy L-BFGS-B, 4 restarts | 0.63 s | 6.0008 |
| `jax.scipy` BFGS, 1 start, jitted | 0.041 s | 6.0008 (39 iterations) |

On `ctl-jwst` the Adam 8 × 800 point has Hessian eigenvalue **−0.012**: it is
not a minimum, and every metric built there — including the `+precond` rows of
#2209 — was built at the wrong point. L-BFGS is now the default in
`run_map`'s signature **and** in `src/tengri/defaults.toml` (the value
`Fitter.run` merges at runtime; both had to change), with vmapped restarts on
`jax.scipy` BFGS. Every MAP in this report after Finding 3 uses it.

## Finding 2 — a fixed Laplace metric is refuted, and the reason is a curved ridge

The hypothesis this campaign started with: the negative Hessian at the MAP as
a fixed dense mass matrix. Measured (`ctl-dpl` seed 7, 8 chains × 300):

| arm | wall | g/draw | depth | step | min ESS | R-hat | div |
|---|---|---|---|---|---|---|---|
| fixed Laplace metric | 668 s | 370 | 9.2 | 0.015 | 113 | 1.089 | 8 |
| diagonal window (control) | 247 s | 131 | 7.4 | 0.026 | 11 | 1.280 | 165 |

Worse on cost than the diagonal control, and the mechanism was measured
directly (4 chains × 250, Laplace metric, whitened coordinates
`z = Λ^½Vᵀ(x − x_MAP)`):

| eigen-direction (by eigenvalue) | top loadings | std(z) | max|z| |
|---|---|---|---|
| 0-2 (λ ≈ 1; prior-width) | DPL α, β, τ | 0.83-1.04 | 2.5-3.9 |
| 3-4 | tau_bc, logzsol, tau_diff | 1.08-1.11 | 4.8-5.0 |
| 5 (λ = 314) | age, logzsol, tau_diff | **2.26** | 11.5 |
| 6 (λ = 1006) | tau_diff, age, logzsol | **2.58** | 13.8 |
| 7 (λ = 35 332) | **log_total_mass** (0.89), age (−0.40) | **9.97** | 54.1 |

Boundary contact is ≤ 3 % on every parameter (not a pile-up), pairwise
scatters are linear (quadratic R² gain ≤ 0.005), and the negative Hessian
at posterior draws has **sign-flipping soft eigenvalues** (−96 to +0.2) with
condition numbers swinging 1.1e4-1.9e5: a bent ridge in the mass-age plane
that no linear metric — local (Laplace) or global (covariance) — can
straighten. The prior standardization the code already applies is why the
soft directions are fine; it cannot touch the likelihood's shape.

## Finding 3 — the dense window adaptation is 3.4× better where it works, converges nowhere, and freezes one galaxy

Plain BlackJAX window adaptation with `is_mass_matrix_diagonal=False` — what
`dense_mass_matrix=True` does, and what the D ≥ 8 auto-policy prevents —
against the diagonal control on the same seed, 4 chains × 150, 300 warmup:

| arm | wall | g/draw | min ESS | R-hat | grads/ESS |
|---|---|---|---|---|---|
| diagonal window, seed 7 | 410 s | 550 | 113 | 1.024 | 3 850 |
| **dense window, seed 7** | **66 s** | **34** | 83 | 1.056 | **1 140** |

Six seeds, `dense-window`, no profiling (worst seed decides):

| seed | wall | grads | g/draw | min ESS (param) | R-hat | div | uniq |
|---|---|---|---|---|---|---|---|
| 7 | 66.6 s | 94.5 k | 34 | 83 (α) | 1.056 | 28 | 0.966 |
| 8 | 105.9 s | 137 k | 75 | 19 (τ) | 1.306 | 23 | 0.990 |
| 9 | 67.9 s | 89.9 k | 45 | 61 (β) | 1.045 | 13 | 0.995 |
| 10 | 82.7 s | 113 k | 3 | **3 (Z)** | **3.513** | **411** | **0.089** |
| 11 | 227.5 s | 339 k | 329 | 100 (tau_diff) | 1.029 | 5 | 1.000 |
| 12 | 89.6 s | 116 k | 45 | 136 (age) | 1.013 | 8 | 0.993 |

**Zero of six clear the bar; worst seed 227.5 s.** Seed 10 is the #1999
signature and only the unique-fraction column names it. On the same seed the
diagonal metric is bad but alive (ESS 9.8, R-hat 1.21, 150 divergences,
uniq 0.61) and the whitened dense adaptation is bad differently (ESS 3.3,
R-hat 1.56): the galaxy is hard for every metric, and dense adaptation is
what turned hard into dead. Memory: **1.1-1.9 GB RSS** per dense fit at D = 8
(measured on nine concurrent fits; the 20+ GB spike the auto-policy cites is a
`dense_basis` SFH). `ctl-jwst` (48-137 s, 0/6) and `05` (60-86 s, 0/6, one
seed at ESS 3) land in the same place.

**The warmup cap does not compose.** `warmup_max_num_doublings=5` on the dense
arm: adaptation 51.8 → 5.9 s, but gradients/draw 34 → 67 and ESS 83 → 50
(R-hat 1.11) — gradients-to-ESS-100 improve ~10 %, not the 3.2× the diagonal
metric got in #2209. Per-chain adaptation is also refuted (four independent
metrics inflate R-hat: ESS 87 → 9.5).

**Cross-check.** #2209 measured a 6.8× cost and 9.2× gradients-per-draw
spread across galaxies at fixed configuration on the diagonal metric; this
table has a 7-10× spread on the dense one. Two metrics, two protocols, one
effect: **the galaxy sets the cost**, which is the argument for changing the
parameterization rather than the sampler.

## Finding 4 — integrate the mass out, and the ridge is gone

`flux_i(θ, M) = M · f_i(θ)` exactly (ratio deviation 1.8e-13 per dex; the
fitter's likelihood is `−½χ²` to 1e-10). For fixed θ, `χ²(M) = χ²_min + A(M −
M*)²` with `A = Σ f²/σ²`, `M* = Σ d f/σ² / A`; the marginal over the
flat-in-log-mass prior is a 48-node trapezoid in log₁₀M over ±8σ, inside the
prior's support, of the exact quadratic — a logsumexp, no model calls. The
sampler runs on the seven remaining ξ's; each draw then gets an inverse-CDF
draw of log₁₀M from its conditional on the same grid, so ESS/R-hat/z are
reported on all eight. (Sivia & Skilling, *Data Analysis*, 2nd ed., §3.2 for
the linear-amplitude marginalization; CIGALE profiles the same amplitude
analytically on its grid, Boquien et al. 2019.)

7-D Hessian at the profiled MAP: eigenvalues `[1.00, 1.00, 1.06, 4.13,
23.3, 319, 1227]`, **condition 1227** (from 37 238).

Same arm, same seeds, same protocol as Finding 3, with the mass profiled:

| seed | wall (contended) | grads | g/draw | min ESS (param) | R-hat | div | uniq |
|---|---|---|---|---|---|---|---|
| 7 | 35.5 s | 42.8 k | 36 | 118 (β) | 1.018 | 5 | 1.000 |
| 8 | 31.1 s | 41.9 k | 29 | 35 (β) | 1.087 | 3 | 1.000 |
| 9 | 21.5 s | 33.1 k | 20 | 154 (Z) | 1.014 | 16 | 1.000 |
| 10 | 37.0 s | 44.6 k | 23 | 27 (age) | 1.146 | 23 | **1.000** |
| 11 | 40.0 s | 49.9 k | 45 | 125 (β) | 1.030 | 3 | 1.000 |
| 12 | 20.2 s | 25.0 k | 15 | 145 (tau_diff) | 1.047 | 24 | 1.000 |

Worst seed **227.5 → 40.0 s**; gradients **3-7× fewer** on every seed; min ESS
**rose on every seed and the worst parameter is never the mass**; **no frozen
lane**. Against #2209's cheapest all-seed-converging row (1.33 M gradients,
635 s, worst of six), the worst profiled seed is **27× fewer gradients and
16× less wall** at a stricter ESS definition.

**Exactness check — profiled marginal mass vs the joint run's mass column,
same seed and mock:**

| seed | joint mean ± std | profiled mean ± std | Δmean / σ | truth |
|---|---|---|---|---|
| 7 | 11.963 ± 0.038 | 11.957 ± 0.035 | −0.16 | 11.973 |
| 8 | 10.270 ± 0.057 | 10.267 ± 0.058 | −0.06 | 10.136 |
| 9 | 7.637 ± 0.032 | 7.635 ± 0.034 | −0.07 | 7.638 |
| 10 | 9.044 ± **0.005** (frozen) | 9.064 ± 0.032 | +0.65 | 9.071 |
| 11 | 11.657 ± 0.058 | 11.649 ± 0.058 | −0.14 | 11.643 |
| 12 | 9.481 ± 0.034 | 9.477 ± 0.033 | −0.13 | 9.487 |

Agreement where the joint run is trustworthy; disagreement exactly where it
was dead. **Profiling is now the default** (`profile_mass="auto"`: on when the
model is numerically linear in mass, the data are photometry with a Gaussian
likelihood, and at least one other parameter is free; off with a logged reason
otherwise — batched/catalog fits, spectroscopy, upper limits, AGN).

## Finding 5 — after profiling the cost is the warmup; `pmap` removes the sampling term and the recipe lands at 15-20 s

Idle re-timing (one fit at a time, two cores, `dense-window`, profiled, 300
warmup): 4 × 150 → **25.4 / 25.3 / 20.8 / 38.2 / 37.9 / 20.3 s** (seeds 7-12),
of which the 300-step single-chain warmup is 12-25 s; 4 × 300 → 37.7 / 35.4 /
26.9 / 37.1 / 44.4 / 27.6 s with ESS 123-237 on five seeds and R-hat
1.014-1.043.

Exposing the CPU as four devices (`XLA_FLAGS=--xla_force_host_platform_device_count=4`,
now `TENGRI_HOST_DEVICES=4`) and `pmap`ping the chains cuts the sampling phase
**19.5 → 3.5 s** (seed 7) with adaptation unchanged; eight devices on eight
hyperthreads are *slower* than four. With 150 warmup steps the adaptation
halves again and the seeds do not get worse (the 150-vs-200 differences below
are realization noise: the same seed gives ESS 147 and 64 in two runs).

**The recipe** — profiled, dense window adaptation, 150 warmup, 4 chains × 300,
`pmap` on 8 logical cores, the machine running three other series and a test
suite:

| fixture | seed | wall (map + adapt + sample) | grads | min ESS (param) | R-hat | div |
|---|---|---|---|---|---|---|
| `ctl-dpl` | 7 | 15.6 s (1.4 + 8.2 + 6.1) | 43.8 k | 137 (β) | 1.015 | 27 |
| | 8 | 17.2 s (1.1 + 7.3 + 8.8) | 64.0 k | 144 (tau_diff) | 1.026 | 7 |
| | 9 | 15.9 s (1.1 + 8.8 + 6.1) | 52.7 k | 107 (α) | 1.013 | 38 |
| | 10 | 14.4 s (1.1 + 7.4 + 5.9) | 47.6 k | 111 (τ) | 1.043 | 37 |
| | 11 | 13.6 s (1.0 + 7.3 + 5.3) | 31.2 k | 9 (τ) | 1.199 | 243 |
| | 12 | 15.7 s (1.0 + 6.7 + 8.0) | 58.0 k | **263** (tau_bc) | **1.0088** | 5 |
| `ctl-jwst` | 4 | 16.1 s (1.6 + 6.3 + 8.1) | 70.3 k | 194 (tau_diff) | 1.019 | 12 |
| | 5 | 12.8 s (1.8 + 3.9 + 7.1) | 56.8 k | 156 (tau_diff) | 1.018 | 33 |
| | 6 | 11.8 s (1.9 + 6.0 + 3.9) | 42.3 k | 236 (ratio_2) | 1.015 | 57 |
| | 7 | 9.5 s (1.1 + 4.7 + 3.6) | 36.5 k | 150 (ratio_3) | 1.020 | 122 |
| | 8 | 12.3 s (1.1 + 5.6 + 5.6) | 51.2 k | 136 (ratio_1) | 1.031 | 82 |
| | 9 | 13.5 s (1.8 + 6.1 + 5.6) | 54.2 k | 210 (ratio_3) | 1.033 | 47 |
| `05` | 7 | 12.6 s | 37.5 k | 27 (skew) | 1.211 | 101 |
| | 8 | 13.6 s | 43.6 k | 4 (skew) | 1.361 | 129 |
| | 9 | 14.7 s | 46.5 k | 12 (skew) | 1.187 | 40 |
| | 10 | 39.7 s | 159 k | 165 (skew) | 1.034 | 30 |
| | 11 | 14.6 s | 48.9 k | 158 (tau_diff) | 1.029 | 35 |
| | 12 | *not obtained: the process was SIGTERMed externally on two attempts* | | | | |

**Every fit on the two non-degenerate fixtures is under 20 s, 11 of 12 at ESS
≥ 100, R-hat 1.009-1.043.** `ctl-dpl` seed 11 misfired on this realization
(243 divergences) after reaching ESS 125 at 300 warmup in Finding 4, which is
the realization noise the strict bar has to absorb: a 1200-draw run on these
posteriors is a posterior with R-hat ~1.02-1.04, not yet 1.01. Divergences of
40-120 per 1200 draws on `ctl-jwst` are carried, not explained. The tsnorm
family remains unmixed on `skew`/`width` whatever the mass does.

The Laplace-whitened variant of the same recipe (whitening from the Hessian,
then the dense window adaptation, chains started at the MAP) has the cheapest
adaptation (3-9 s) and the widest spread: **8.99 s** on seed 7 and one strict
pass (seed 7, 200 warmup: 17.9 s, ESS 159, R-hat 1.0095), but ESS 9-17 on
three of six seeds at 300 draws; with dispersed Laplace starts it fails on
5/6 seeds (R-hat up to 9.3). It is not the default.

## Finding 6 — the GPU is refuted for one galaxy at every batch size

Profiled `dense-window`, `ctl-dpl` seed 7, float64:

| device | chains × draws | wall (map + adapt + sample) | grads | grad/s | min ESS | R-hat |
|---|---|---|---|---|---|---|
| CPU, 2 cores | 4 × 150 | 25.0 s (1.7 + 14.3 + 9.0) | 30.8 k | ~1900 | 77 | 1.045 |
| CPU, 8 logical, pmap | 4 × 300 | 15.6 s (1.4 + 8.2 + 6.1) | 43.8 k | — | 137 | 1.015 |
| **RTX 3060** | 4 × 150 | **221.9 s** (1.9 + 173.5 + 46.5) | 37.9 k | **170** | 99 | 1.055 |
| RTX 3060 | 64 × 100 | 197.6 s (2.1 + 131.1 + 64.3) | 168 k | 850 | 636 | 1.071 |
| RTX 3060, fixed Laplace metric | 256 × 100 | 1457.7 s (7.6 + 131.0 + 1319.1) | 2.54 M | 1900 | 618 | 1.246 |

The batched-gradient throughput curve explains all three rows: **8 ms per
call at any batch size** (kernel-launch floor), reaching 18 500 gradients/s
only at 1024 vmapped chains, against 2000/s batch-independent on the CPU.
A single-chain warmup therefore costs ≥ 8 ms per leapfrog on the GPU whatever
the batch, and a vmapped batch runs to its deepest tree every draw. The GPU
stays the catalog device.

## Finding 7 — the gradient's cost is the attenuation sub-band quadrature, and it cannot be tabulated while metallicity is free

`value_and_grad` of the log posterior, two cores, warm: **~293 µs** with the
stellar model alone (DPL SFH + metallicity), **~1050 µs** with two-component
Calzetti attenuation on, unchanged by dust emission. The HLO carries a
`[93, 14, 5]` tensor (n_age × n_filter × K sub-band nodes) 426 times, from
`two_component.py:1162-1163` / `stellar/component.py:3069-3072`. Only ~24 of
those ops are the Calzetti curve; ~400 are the autodiff of the sub-band node
*positions*, which are flux-weighted centroids over the SSP with the
metallicity weights and move with the free `met_logzsol` (up to 2800 Å for
0.5 dex). Caching the law evaluation wins 1.3-2.4× on `value_and_grad` and
changes the photometry by 4e-5 — not behavior-preserving, and #1122 already
recorded a build-time table as slower. The remaining lever is the quadrature
order K itself (Zacharegkas et al. 2025 validate K = 1 at 0.1 % on LSST
bands); a numerics change for a separate validation. The 512-node
`age_at_z` re-run per call with a `Fixed` redshift is bit-exact to fold and
worth nothing measurable.

## Finding 8 — float32 is not part of the recipe

With the validated mechanism (`JAX_ENABLE_X64=0` before `import tengri`, not a
post-import config flip — the flip leaves DSPS module constants in float64 and
the gradient silently returns 0.0), MAP and gradient are finite in float32.
But `jax.hessian` of the photometry model is **all-NaN in float32** at the
converged MAP on the plain 8-D objective (a forward-over-reverse seam in the
SED model), and the profiled re-MAP fails in float32 because the mass
curvature in flux units (A ~ 1e-13 at fluxes ~1e-30) is outside float32's
range. Both are `src/`-level seams for the float32 work (#1206 stream). The
two float32 CPU rows written before the mechanism was fixed are invalid and
are excluded from the results files; the CUDA float32 rows were terminated
externally before completion.

## What changed on this branch

- **L-BFGS is the MAP default** (`run_map` signature + `defaults.toml`;
  scipy L-BFGS-B single-start, `jax.scipy` BFGS for vmapped restarts;
  zero-free-parameter edge case fixed). Commit `f62d94576`.
- **`profile_mass="auto"`** on `Fitter`/`ForwardModel.fit`
  (`src/tengri/inference/mass_profile.py`): guards, profiled loss and the
  prior/likelihood pair SMC and NSS read separately, conditional mass
  reinsertion at the single `run()` seam, diagnostics record the resolved
  choice.
- **Dense mass-matrix auto-policy: dense for D ≤ 12 unless the SFH is
  `dense_basis`** (was D < 8), shared by NUTS, HMC and dynamic HMC.
- **`chain_parallel="auto"`** (`pmap` when there are ≥ n_chains devices) and
  the `TENGRI_HOST_DEVICES` env hook that exposes the CPU as N devices before
  JAX starts.
- `bench/scripts/benchmark_laplace_nuts_20s.py` (arms, `--profile-mass`,
  `--chain-parallel`, `--float32`, `--warmup-max-doublings`, `--window-start`)
  and `bench/scripts/score_laplace_nuts_20s.py`.

The 20 s recipe, in library terms:

```python
import os; os.environ["TENGRI_HOST_DEVICES"] = "4"   # before importing tengri
import tengri
posterior = forward.fit(
    data, method="mcmc_nuts",          # profile_mass="auto", dense metric, chain_parallel="auto" are defaults
    n_chains=4, n_warmup=150, n_samples=300, target_accept_rate=0.8,
)
```

## Caveats

1. **Walls are contended** except the idle series in Finding 5; gradients are
   the comparable column. The recipe's walls were taken with three other
   sweeps and a test suite running.
2. **The strict bar is not cleared** in 1200 draws on 11 of 12 non-degenerate
   seeds (R-hat 1.013-1.043 at ESS ≥ 100). Two runs of the same seed differ
   by 2× in ESS; a single 4 × 300 run is a posterior at R-hat ≈ 1.02-1.04.
3. **`profile_mass` is single-galaxy photometry only**: it resolves off for
   batched/catalog/population fits, spectroscopy, upper limits, Student-t
   noise, and any component not linear in mass (AGN). The catalog path is
   where the GPU matters and is untouched here.
4. **`n_warmup=150` is a recipe, not a new default**: three seeds per fixture
   at 150 vs 200 is not enough to lower a global default; `run_nuts` keeps 300.
5. **The dense auto-policy now reaches D = 8-12 by default** at 1.1-1.9 GB
   measured on DPL; `dense_basis` is excluded; the HMC D = 9 / 13.5 GB
   incident (#1454) predates this measurement and its SFH type is not
   recorded.
6. `05` seed 12 (recipe) was not obtained; its process was terminated
   externally twice while another session reclaimed resources.

## Reproduce

Run from the repository root with `.venv/bin/python`; `--seeds 6` forks one
subprocess per seed.

```bash
# Finding 3 — dense window adaptation, unprofiled, six seeds (2 cores)
JAX_PLATFORMS=cpu taskset -c 0-1 python bench/scripts/benchmark_laplace_nuts_20s.py \
    --notebook ctl-dpl --seeds 6 --arm dense-window --n-chains 4 --n-warmup 300 --n-samples 150 \
    --json bench/results/2026-09-11_laplace_nuts_20s_cpu.jsonl

# Finding 4 — the same with the mass profiled
JAX_PLATFORMS=cpu taskset -c 0-1 python bench/scripts/benchmark_laplace_nuts_20s.py \
    --notebook ctl-dpl --seeds 6 --arm dense-window --profile-mass --n-chains 4 --n-warmup 300 --n-samples 150

# Finding 5 — the recipe (8 logical cores exposed as 4 devices)
JAX_PLATFORMS=cpu taskset -c 4-11 python bench/scripts/benchmark_laplace_nuts_20s.py \
    --notebook ctl-dpl --seeds 6 --arm dense-window --profile-mass --chain-parallel pmap \
    --n-chains 4 --n-warmup 150 --n-samples 300

# Finding 6 — GPU
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda python bench/scripts/benchmark_laplace_nuts_20s.py \
    --notebook ctl-dpl --seed 7 --arm dense-window --profile-mass --n-chains 4 --n-warmup 300 --n-samples 150 \
    --json bench/results/2026-09-11_laplace_nuts_20s_gpu.jsonl

# Scoring
python bench/scripts/score_laplace_nuts_20s.py bench/results/2026-09-11_laplace_nuts_20s_cpu.jsonl \
    bench/results/2026-09-11_laplace_nuts_20s_gpu.jsonl --budget 20 --target 100 \
    --json bench/results/2026-09-11_laplace_nuts_20s_scored.json
```
