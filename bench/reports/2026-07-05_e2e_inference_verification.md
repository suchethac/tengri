# End-to-end inference verification — DESI lines + GALEX/DES/WISE

**Date:** 2026-07-05 · **Branch:** `cs/inference` · **Platform:** macOS, JAX_PLATFORMS=cpu

Goal: verify the full inference pipeline for the headline use case — fit
**GALEX→WISE photometry + DESI emission-line fluxes jointly, fast, at catalog
scale, with limits and a noise model** — and that it is mathematically correct
(proper prior transforms, recovery to truth), fast (sub-minute per galaxy), and
free of pathological JIT / OOM. Fixes for everything found are on this branch.

## Executive summary

| Aspect | Status | Evidence |
|--------|--------|----------|
| Prior transformations correct | ✅ FIXED | pushforward KS FAIL→PASS for LogUniform/StudentT/truncated; 35 contract tests |
| Joint phot+lines runs under JIT | ✅ FIXED | eager==jit to 1e-14, finite grads (was crashing: dropped line catalog) |
| Line upper/lower limits in the fit | ✅ FIXED | censored ln Φ term now wired (was silently ignored) |
| Catalog data threading | ✅ FIXED | each galaxy fit against its own data (was baking galaxy-1's data) |
| Per-galaxy speed < 1 min | ✅ | MAP 1.4–6.4 s; NUTS(300/500) 25 s on D=7 |
| No OOM | ✅ | peak RSS ≤ 10.3 GB (headline), 1.1 GB (minimal) under 20 GB monitor |
| Catalog throughput | ✅ | ~1066 galaxies/hour MAP, 1 core, no recompile |
| WavePrecomp recovery-unbiased | ✅ | exact ≡ precomp MAP to 0.002/param at z=0.05 |
| Sampling recovery to truth | ✅ (caveats) | R-hat 1.004; 4/7 in 68% CI; misses = SFH degeneracy + low ESS |

## 1. Prior transformations (parameters/priors.py)

The standardized parameterization requires `unstandardize(ξ) = F⁻¹(Φ(ξ))` so the
½ξᵀξ prior term corresponds to the *declared* prior (MGVI, arXiv:1901.11033
Eqs. 18-25). Pushforward audit (push N(0,1) through `unstandardize`, KS vs
`sample()` and vs the `log_prob` CDF):

| Distribution | KS D before | KS D after | Bug |
|--------------|-------------|-----------|-----|
| LogUniform(1,300) | 0.118 (p≈0) | 0.0025 (PASS) | `sigmoid` → logit-normal in log space, not log-uniform |
| StudentT(0,0.3,df=2) | 0.176 (p≈0) | 0.0033 (PASS) | variance-matched Gaussian; heavy tails discarded |
| StudentT(0,1,df=5) | 0.039 | 0.0026 (PASS) | same |
| Gaussian truncated | 0.157 (p≈0) | 0.0026 (PASS) | `clip` → point mass + zero gradient at bounds |
| Uniform / Gaussian / LogNormal | pass | pass | (already correct) |

Fixes: LogUniform sigmoid→Φ CDF; StudentT exact quantile (df∈{1,2} closed
form, general df via incomplete-beta table); Gaussian/LogNormal/StudentT true
truncation via inverse-CDF reparam + normalized `log_prob` (so nested-sampling
evidence compares priors correctly). Regression:
`tests/contract/test_prior_standardization_pushforward.py` (35 tests).

**Affected science:** LogUniform → PSD burstiness hyperpriors, AGN log-Lbol,
GRAHSP; StudentT df=2 → all continuity-SFH log-ratio priors (Leja+2019).
Posteriors for stochastic-SFH and continuity-SFH fits shift → breaking change.

## 2. Speed & catalog throughput (headline model: 11-band + 5 DESI lines)

MAP (L-BFGS) single fit **6.4 s**; forward eval 3.4 ms warm (WavePrecomp),
33 ms exact. Catalog loop (same model, 6 galaxies, different data):

    galaxy 0..5 MAP wall: 3.08 / 3.27 / 3.38 / 3.77 / 3.30 / 5.98 s
    warm median 3.38 s · recompile: NO · ≈ 1066 galaxies/hour (MAP, 1 core)

Compiled kernel is reused across galaxies (no recompile) — the three-layer
cache + the cache-key fix below make independent per-galaxy fitting the fast
catalog path. Peak RSS 10.3 GB.

## 3. WavePrecomp parity & recovery bias

Per-band exact↔precomp forward error grows in rest-UV at high z (GALEX FUV up to
~16× at z=1 — the known dust-Taylor rest-UV bias) but is small at low z. Decisive
recovery test: mock generated with the **exact** model, MAP-fit with exact vs
WavePrecomp — recovered parameters agree to **0.002/param** at z=0.05, and
WavePrecomp is **13× faster** (1.65 s vs 21.6 s). WavePrecomp is recovery-safe
for low-z catalogs; use `approx=None` for rest-UV-critical high-z GALEX work.

## 4. Sampling recovery (minimal recipe, D=7)

MAP×24 restarts (1.4 s) → NUTS(300 warmup / 500 samples, diagonal mass) **25 s**,
peak 1.1 GB, **R-hat 1.004**. 4/7 params in 68% CI. The 3 misses (mass, met,
peak-age) are 2-3σ but only ~0.1 dex / 0.02 dex — the age–dust–metallicity
degeneracy plus low ESS (21, single short chain), confirmed unbiased by the
exact≡precomp control (§3). Not a transform or approximation defect.

## 5. Limits as data points + noise model

- `LineFluxData` now carries `is_upper_limit` / `is_lower_limit`; `from_dict`
  accepts `(flux, err, 'upper'|'lower')`. Limits enter as censored `ln Φ` terms
  (zero penalty when the model sits the right side of the limit). Before: the
  flag was silently ignored — identical loss with and without it.
- `Fitter(data_mask=…)` now rejects boolean masks (True would silently mean
  "upper limit", the opposite of include/exclude) and documents the trinary
  convention (0 detected / +1 upper / −1 lower).
- Free calibration floor (`NoiseModel(calibration_floor=Uniform(...))`) is live
  in the objective and threaded per galaxy.

## 6. Bugs fixed (this branch)

1. **Standardization priors** (§1) — LogUniform/StudentT/truncation.
2. **Nebular line-catalog dropped under JIT** — the air→vacuum guard used
   numpy/boolean-indexing on what is a Tracer under a jitted sampler; the
   `except` swallowed it, dropping `line_waves`/`line_lums` → joint fits crashed
   with a misleading "backend did not publish" error. Now trace-safe
   (jnp air→vacuum + Balmer vote); failures warn instead of silently dropping.
3. **`predict_line_fluxes`/`_ratios` spurious `×L_SUN`** — `line_lums` are
   already erg/s (per the `DerivedKey` contract); the factor was a 33.6-dex
   flux error invisible to self-cancelling mocks but wrong against real data.
4. **Catalog data threading** — likelihood adapters baked their obs/err/mask
   arrays into the shared compiled loss; every galaxy after the first was fit
   against the first galaxy's data. Adapters now read from `data_args` at call
   time (`resolve_channel_data`).
5. **Engine/loss cache key missing observation channels** — a joint phot+lines
   Fitter could reuse a photometry-only compiled engine (line term dropped, or
   a missing-key crash). `_engine_cache_key` + `compile_signature` now include
   line-flux/ratio/index/censoring structure.
6. **Compile-time constant-folding** — nebular line-render grid-spacing median
   was constant-folded (O(n log n) sort) at every kernel compile; now evaluated
   eagerly when the wave grid is concrete.

## 7. Deferred (issues to file)

- Hierarchical population catalog fitting — verification + O(N) benchmark
  (`area:population`, `area:inference`).
- Survey-depth noise abstraction / per-band floors (synference-inspired,
  `area:observation`).
- Additional bijective-transform priors (InverseGamma / Gamma / Laplace / Beta)
  following the NIFTy `special_distributions` pattern (`area:api`).
- Full stochastic/continuity-SFH before/after recovery quantifying the
  LogUniform/StudentT de-biasing end-to-end (verification-only, >1 min NUTS).

## Reproduce

    # prior pushforward audit (fast)
    PYTHONPATH=src JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
      tests/contract/test_prior_standardization_pushforward.py -q
    # headline MAP + catalog loop, recovery — scratch scripts, OOM-guarded:
    LIMIT_GB=20 scripts/run_with_oom_monitor.sh -- <venv>/python headline_map.py
