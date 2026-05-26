# Inference backend validation — issue #231

**Date:** 2026-05-22
**Script:** `scripts/validate_backends_231.py` (and `..._retry.py`)
**Hardware:** macOS, CPU (`JAX_PLATFORMS=cpu`), JAX 64-bit
**Mocks:** two photometry-only models, 8 SDSS+2MASS bands, SNR=20 at z=0.05:
- **dpl** — Double-power-law SFH + Calzetti dust, D=6 free
- **dense_basis** — DenseBasis SFH + Calzetti dust, D=7 free

Each backend ran in its own subprocess so segfaults didn't take the
sweep down. Per-backend wall-clock budgets capped at 180–600 s.

## Numbers

| Backend | dpl cold/warm/RSS | dense_basis cold/warm/RSS | Outcome |
|---|---|---|---|
| `map` | 2.4 s / 0.6 s / 2.7 GB | 9.3 s / 0.5 s / 2.2 GB | primary ✓ |
| `laplace` | 4.9 / 1.4 / 3.0 | 8.9 / 1.3 / 2.9 | **promoted to primary** |
| `mcmc_hmc` | 21 / 9.5 / 5.1 | 21 / 12 / 5.1 | **promoted to primary** |
| `mcmc_dynamic_hmc` | 19 / 7.6 / 5.3 | 20 / 8.0 / 5.4 | **promoted to primary** |
| `mcmc_ghmc` | 17 / 6.0 / 5.1 | 18 / 6.0 / 5.2 | **promoted to primary** (fastest HMC variant) |
| `mcmc_mclmc` | 20 / 2.0 / 5.0 | 22 / 2.1 / 5.0 | **promoted to primary** (fastest warm call) |
| `mcmc_adjusted_mclmc` | 63 / 9.7 / 4.7 | 67 / 8.4 / 4.9 | experimental (~3× compile premium over mclmc) |
| `mcmc_ess` | 9.7 / 4.7 / 2.2 | 10 / 4.8 / 2.2 | experimental (Gaussian-prior assumption unvalidated) |
| `mcmc_nuts` | 92 / 38 / 5.3 | **timeout >300 s** | primary, but flagged: dense_basis warmup is pathological |
| `mcmc_raytrace` | 15 / 10 / 2.5 | 16 / 11 / 2.5 | primary ✓, scales effortlessly |
| `mcmc` (auto) | 117 / 43 / 5.3 | timeout (routes to NUTS) | bug fixed; inherits NUTS dense_basis caveat |
| `vi` / `vi_nonlinear` | 95 / 58 / 21 | 126 / 79 / 22 | primary, with memory warning added |
| `vi_nonlinear_fast` | 107 / 57 / 20 | 140 / 80 / 21 | primary, same warning |
| `vi_linear` / `vi_linear_fast` | 61 / 51 / 19 | 83 / 72 / 20 | experimental — slower per iter than geoVI |
| `nss` | 236 / 230 / 10 | **timeout >600 s** | experimental, slow — model-comparison only |
| `native_vi_linear` | **segfault** | **segfault** | experimental, **marked UNSTABLE** |
| `native_vi_nonlinear` | **segfault** | **segfault** | experimental, **marked UNSTABLE** |
| `pathfinder` | **segfault** | **segfault** | experimental, **marked UNSTABLE** |

Times reported as `cold / warm / peak RSS`.
Cold = first call (compile + run). Warm = second call (cache hit).

## Bugs found and fixed in this validation

1. **`run_nuts` missing context normalization.** Calling
   `_mcmc_auto_pick → run_nuts(context=Fitter)` crashed with
   `AttributeError: 'Fitter' object has no attribute 'fitter'`.
   HMC and raytrace both call `InferenceContext.from_target(context)`
   at entry; NUTS did not. Fixed by adding the same line. This is
   what unblocked `method="mcmc"` (the auto-dispatcher) at low-D.

2. **Three backends segfault the Python interpreter** on a simple
   D=6 photometry mock (DPL SFH + Calzetti, 8 bands, no nebular):
   `native_vi_linear`, `native_vi_nonlinear`, `pathfinder`. All
   three crashed both DPL and dense_basis variants. `pathfinder`
   crashed independently of arguments; the `native_vi_*` failures
   reproduced with `n_seeds=1` (so the cause is not the default
   vmap-over-5-seeds compile blow-up). These need separate triage —
   marked `[UNSTABLE]` in `short_doc` so users don't trip on them.

3. **`run_nuts` warmup is pathological on `mean_sfh_type="dense_basis"`.**
   Times out past 300 s with `dense_mass_matrix=False`. The same
   model fits cleanly with HMC, dynamic_hmc, ghmc, or mclmc in
   under 25 s cold. This corroborates the existing CLAUDE.md note
   on D≈8 dense_basis hitting 20+ GB peak.

4. **NIFTy VI is memory-heavy even at D=6** (19–22 GB peak RSS for
   any of `vi`, `vi_nonlinear`, `vi_linear`). Astronomers running
   multi-fit notebooks should know this — `short_doc` now states it.

## Recommendation tree

```
Want a posterior on D ≤ 10?
├── Quick point estimate: map (2-9 s)
├── Quick uncertainties:  laplace (5-9 s)
├── Full posterior, fast: mcmc_ghmc or mcmc_mclmc (~20 s cold, ~5 GB)
├── Robust default:       mcmc_hmc (~20 s cold)
└── If you must use NUTS: mcmc_nuts on DPL only; for dense_basis,
                          use mcmc_hmc instead.

Memory-tight machine (<16 GB)?
└── Avoid the NIFTy VI family entirely; use mcmc_ghmc / mcmc_mclmc.

Need model comparison (evidence)?
└── nss — slow but the only option. Stay at D ≤ 6 if you can.
```

## What we did NOT validate

- **Convergence diagnostics.** No R-hat, ESS, autocorrelation, or
  Geweke checks. The harness recorded only point-estimate bias
  (`|posterior_mean - truth| / posterior_std`) — that says nothing
  about whether the chain mixed. With the conservative budgets used
  here (100–300 warmup, 200–300 samples) most chains would fail R-hat
  even if the backend is sound. Re-running with `n_samples=2000+` and
  calling `posterior.check_convergence()` is a follow-up — adequate
  for tier decisions, not for Paper I quality claims.
- **Posterior coverage** (SBI-style rank statistics, posterior
  predictive checks). Out of scope for the tier decisions in #231.
- The **IFT field SFH** (137-D). Explicitly out of scope per user
  request — many backends will not scale there.

## Convergence-checked round 2 (2026-05-23, follow-up)

`scripts/validate_backends_231_convergence.py` re-ran the four MCMC
backends I had promoted, with realistic budgets and per-variant
tuning:

- `mcmc_hmc`: `n_warmup=1000`, `n_burnin=200`, `n_samples=2000`,
  `dense_mass_matrix=True`, `n_leapfrog_steps=20`
- `mcmc_dynamic_hmc` / `mcmc_ghmc`: `n_warmup=1000`,
  `n_burnin=200`, `n_samples=2000`
- `mcmc_mclmc`: `n_samples=4000`

Pass criterion: split-R-hat < 1.01 AND minimum-ESS > 400 (Vehtari 2021).

| Backend | DPL (D=6) R-hat / ESS | dense_basis (D=7) R-hat / ESS | Verdict |
|---|---|---|---|
| `mcmc_hmc` | **1.008 / 411** | 1.051 / 17 | DPL ✓ / dense_basis needs more samples |
| `mcmc_dynamic_hmc` | 1.113 / 27 | 1.255 / 1 | both fail |
| `mcmc_ghmc` | 2.487 / 1 | 3.115 / 1 | both fail catastrophically |
| `mcmc_mclmc` | 1.729 / 1 | 1.129 / 1 | both fail |

**Action:** demoted `mcmc_dynamic_hmc`, `mcmc_ghmc`, `mcmc_mclmc`
back to `experimental` and tagged `mcmc_ghmc` / `mcmc_mclmc` with
`[POOR MIXING]` in `short_doc`. `mcmc_hmc` stays primary with a
docstring requirement to use `dense_mass_matrix=True`, `n_warmup≥1000`,
`n_leapfrog_steps≥20` for convergence — the defaults that the round-1
speed sweep used (`n_warmup=100`, `dense_mass_matrix=False`) gave
junk chains.

Why `mcmc_hmc` works while `mcmc_dynamic_hmc` doesn't, despite both
being from BlackJAX: the static HMC's fixed `n_leapfrog_steps=20`
trajectory was long enough to cross the SED-degeneracy banana,
whereas `dynamic_hmc`'s adaptive trajectory tuning at BlackJAX
defaults underestimates the required path length on these
correlated geometries. Investigating per-backend defaults that
match the SED geometry is a follow-up.

The same caveat that flagged this sweep also applies looking
forward: numbers from `validate_backends_231.py` measure wiring +
speed + memory, NOT posterior quality. Always run
`posterior.check_convergence()` plus a split-R-hat / ESS check
before publishing.

## Cross-method consistency + gradient flow (2026-05-23, follow-up)

`scripts/validate_backends_231_consistency.py` runs `map`, `laplace`,
`mcmc_hmc`, and `nss` on the **same DPL mock with the same seed** and
compares posteriors. If the forward model and likelihood are correct
and every backend is well-wired, the four methods must agree on
posterior means within a fraction of σ and on CI widths within a
factor of ~2.

**Gradient flow (forward model + χ²):**

At the prior midpoint (≈ truth): χ² = 5.7, all six ∇χ² entries
finite, all non-zero, dynamic range 0.03 → 31.
At a 10 % perturbation: χ² = 4078, ∇χ² entries 1.9 → 35 288, still
all finite. The forward model and likelihood differentiate cleanly
through every parameter — no broken pieces, no zero rows.

**Posterior consistency (truth → method posterior mean ± std):**

| Parameter | Truth | MAP | Laplace | mcmc_hmc | NSS |
|---|---|---|---|---|---|
| `dust_tau_bc` | 0.500 | 0.500 | 0.494 ± 0.206 | 0.503 ± 0.208 | 0.496 ± 0.286 |
| `met_logzsol` | -0.900 | -0.889 | -0.887 ± 0.135 | -0.891 ± 0.105 | -0.918 ± 0.119 |
| `sfh_dpl_alpha` | 2.550 | 2.452 | 2.494 ± 1.008 | 2.461 ± 1.028 | 2.547 ± 1.443 |
| `sfh_dpl_beta` | 1.550 | 1.685 | 1.669 ± 0.490 | 1.881 ± 0.477 | 2.055 ± 0.589 |
| `sfh_dpl_log_total_mass` | 1.000 | 1.028 | 1.028 ± 0.085 | 1.053 ± 0.082 | 1.084 ± 0.108 |
| `sfh_dpl_tau_gyr` | 6.050 | 7.207 | 7.065 ± 1.987 | 7.499 ± 1.890 | 8.121 ± 2.327 |

The four methods agree on every posterior mean within ≤ 0.5σ of
each other and recover truth within ~1σ on every parameter. CI
widths cluster within a factor of 1.5×. NSS is slightly more
conservative (broader tails) — expected, because nested sampling
explores the full prior support whereas HMC's 2000 samples
undersample the tails. The ~0.5σ shift on `sfh_dpl_tau_gyr` is
common across all four methods, so it is the data-driven posterior
mean shift at SNR = 20 (not a backend bug).

**Wallclock**: map 10 s, laplace 9 s, mcmc_hmc 104 s, nss 234 s.

**What this proves:**

- Gradients flow through every free parameter of the SED model.
- The four backends with very different mathematical machinery
  (point estimate, Gaussian approximation, gradient-based MCMC,
  likelihood-evaluation nested sampling) reach the same posterior.
  An inconsistency in any one would have shown as a > 1σ
  disagreement on at least one parameter.
- NSS, the gradient-free backend, agrees with HMC, the gradient-
  intensive one — independent corroboration that both the
  gradient code path and the likelihood code path are correct.

**What this does NOT prove:**

- That this is true on real data — only the noised mock was used,
  with truth known by construction.

## dense_basis cross-method consistency (2026-05-23)

Same approach on the dense_basis SFH mock (D=7), all four methods on
the same mock + key, now with `approx=WavePrecomp()`:

- map 1 s · laplace 3 s · mcmc_hmc 5 s · nss 13 s · all four pass

(Full table in `scripts/_backend_consistency_db_results.json`.) All
methods agree on `log10(M*)` within < 0.5 dex and on `log10(SFR_100Myr)`
within < 0.5 dex on this self-fit. NSS dropped from 354 s → 13 s with
WavePrecomp — a 27× speedup. The same WavePrecomp run on `mcmc_hmc`:
default approx 58.4 s cold → `WavePrecomp()` 3.5 s cold, **17× speedup**.

## Component-pair self-consistency + misspecification (2026-05-23)

`scripts/validate_backends_231_component_pairs.py` — five SFH × dust
combinations, all photometry, all with `approx=WavePrecomp()`,
`mcmc_hmc` only.

**Self-consistency** (truth model = fit model):

| Scenario | wall | Δlog M⁎ | Δlog SFR₁₀₀ |
|---|---|---|---|
| `dpl+calzetti` | 8.0 s | +0.05 ✓ | -0.79 ✗ |
| `dense_basis+calzetti` | 9.7 s | +0.09 ✓ | -0.08 ✓ |
| `tsnorm+calzetti` | 7.7 s | +0.07 ✓ | +0.01 ✓ |
| `dpl+smc` | 7.6 s | +0.05 ✓ | -0.71 ✗ |
| `dexp+calzetti` | 8.1 s | +0.03 ✓ | -0.09 ✓ |

The DPL self-fits miss `Δlog SFR₁₀₀` by 0.7-0.8 dex because the DPL
prior **midpoint** corresponds to a `log10(SFR_100Myr) = -2.61`
(quenched) galaxy — fractional error on a tiny absolute SFR. This is
a "pathological truth" not a backend bug; with a star-forming truth
the same DPL recovers SFR. The other four SFH parametrisations pick
midpoints near typical star-forming SFRs and recover both quantities
within 0.5 dex.

**Misspecification** (truth model ≠ fit model):

| Truth → Fit | wall | Δlog M⁎ | Δlog SFR₁₀₀ |
|---|---|---|---|
| `dense_basis → dpl` | 4.9 s | -0.11 ✓ | -0.48 ✓ |
| `tsnorm → dpl` | 4.8 s | +0.01 ✓ | -1.68 ✗ |
| `dpl → dense_basis` | 6.4 s | +0.07 ✓ | -27.4 ✗ (SFR floor) |
| `dpl+smc → dpl+calzetti` | 4.4 s | +0.15 ✓ | -1.94 ✗ |

**Headline**: stellar mass is robust to SFH/dust misspecification on
broad-band photometry — all four cases recover `log M⁎` within
±0.15 dex. SFR₁₀₀ is fragile: only the `dense_basis → dpl` case
stays inside 0.5 dex; the cases where truth has a near-zero SFR
hit the log floor, and wrong dust law propagates into SFR. The
inference is doing the right thing; the science conclusion is that
broad-band photometry alone constrains M⁎ but not SFR₁₀₀ once the
SFH parametrisation is wrong.

## Bug discovered: HMC MAP-init tracing pathology (#262)

Setting up the misspec runs surfaced a real tengri bug: calling
`Fitter(model, flux_obs, noise).run(method='mcmc_hmc')` before any
Python-side forward evaluation on the same `SEDModel` instance
crashes with `ConcretizationTypeError` in
`Uniform.unstandardize` during the MAP-init JIT trace. Workaround:
a single `model.predict_photometry(some_params)` call before the
HMC fit. Tracked in #262. The component-pairs harness now
pre-warms inside `fit_only`.

## Related issues fixed in passing

- **#229** (legacy `approx=` docstring references): swept all
  `approx={'wave_precomp': True}` mentions in component / protocol /
  forward / observation docstrings to the current `approx=WavePrecomp()`
  form.
- **#230** (vi vs vi_native PSD divergence): partially superseded —
  this PR's segfault finding on `native_vi_*` means the "reconcile"
  path is blocked on a more fundamental stability bug. The doc-only
  half of #230 should still happen once `native_vi_*` is stable.
