# Tengri API & Naming Design Document

**Date:** 2026-04-03
**Status:** Approved — implementation tiers active
**Purpose:** Retrospective design analysis — what we would do differently from scratch, and a
forward-looking vocabulary spec for the codebase. This document covers the full codebase: class
names, method conventions, parameter hierarchy, module organization, and pre-work that should
precede any future major feature.

---

## Part I — What Should Have Happened Before Starting

### 1. Write the three usage stories first

Before any class definition, write runnable pseudocode for three user archetypes:

```python
# STORY A — Grad student, first fit
# NOTE: uses full prefixed param names (no short names); "logzsol" is the one permitted alias
model = tengri.SEDModel.from_config(
    ssp="data/ssp.h5", sfh="tsnorm", filters=["sdss_g", "sdss_r"],
    redshift=0.1, priors=dict(sfh_tsnorm_log_total_mass=Uniform(-1, 2.5), logzsol=Uniform(-2, 0.2))
)
result = model.fit(flux, noise)
result.plot_corner()

# STORY B — Expert, full control
params = tengri.Parameters(sfh="dpl+field", dust="two_component", nebular="cue")
params.set("sfh_dpl_alpha", Uniform(0.5, 5))
params.set("sfh_field_psd_sigma", LogUniform(0.1, 3))
model = tengri.SEDModel(params, ssp, observation)
result = tengri.Fitter(model, data, noise).run("vi").refine("mcmc_raytrace")

# STORY C — Population, joint hierarchical fit
# NOTE: fit_batch() for independent fits; fit_population() for shared hyperparameters
observations = [tengri.Observation.from_row(row, ssp="data/ssp.h5") for row in table]
model = tengri.SEDModel.from_config(ssp="data/ssp.h5", sfh="tsnorm+field", ...)
result = model.fit_population(observations)   # shares PSD hyperpriors across the population
result.plot_hyperparameters()
```

Writing these before implementation forces every class name, argument order, and method
to be decided as a coherent user journey rather than piecemeal as features accumulate.
The `fit_catalog / fit_batch / fit_population` confusion exists precisely because Story C
was never written in full before implementation.

### 2. Write the suffix convention table

One markdown file, agreed on before any code:

| Suffix | Meaning | Examples (current) |
|--------|---------|-------------------|
| *(none)* | Data container / physics object | `Photometry`, `Spectroscopy`, `Observation` |
| `Model` | A physical forward model | `SEDModel`, `NoiseModel` |
| `Parameters` / `Params` | Prior specification, what gets fit | `Parameters` (currently `ParamSpec`) |
| `Fitter` | Runs inference | `Fitter`, `PopulationFitter` |
| `Posterior` | Result of inference (samples + diagnostics) | `Posterior`, `PopulationPosterior` |
| `Config` | Static structural choice (not a fitted param) | `AGNConfig`, `VIConfig` |
| `Backend` | Interchangeable computation engine | `CueBackend`, `CloudyGridBackend` |

Currently violated: `SpectroscopyConfig` is a data container (should have no suffix or
`Spectroscopy`), `NoiseConfig` is a model (should be `NoiseModel`), `HierarchicalResult`
and `Posterior` are the same concept with different suffixes.

### 3. Draw the module boundary tree

```
src/tengri/
├── core/           ← Model assembly + SED computation (no physics)
├── physics/        ← Pure differentiable functions (currently models/)
│   ├── sfh/
│   ├── stellar/    ← (currently sps/)
│   ├── dust/
│   ├── nebular/
│   ├── agn/
│   ├── observation/
│   ├── igm.py
│   ├── radio.py
│   └── xray.py
├── inference/      ← One file per method family (not a 4300-line god file)
│   ├── map.py
│   ├── vi.py       ← geoVI + MGVI + NIFTy fast path
│   ├── mcmc.py     ← NUTS + Ray Tracing + ESS
│   ├── nested.py   ← NSS evidence
│   ├── laplace.py  ← Laplace + Pathfinder
│   └── batch.py    ← fit_batch population dispatch
└── utils/
```

### 4. Define "settings vs parameters" explicitly

The codebase conflates two concepts inside `ParamSpec` and `Model.__init__`:

- **Settings** — which sub-models are active (dust law, nebular backend, AGN type). These do
  not appear in the gradient tape. Stored in `AGNConfig`, `SpectroscopyConfig`, etc.
- **Parameters** — scalars with priors that get fitted. Stored in `ParamSpec`.

A clean pre-design would have made `AGNConfig`, `DustConfig`, and `NebularConfig` first,
then `Parameters` contains only fittable quantities, and `SEDModel.__init__` takes explicit
config objects rather than a soup of kwargs.

---

## Part II — Model Rename

### Recommendation: `SEDModel`

| Option | Pros | Cons |
|--------|------|------|
| `SEDModel` | Self-documenting, short, accurate | Implies only SEDs (but that's what it is) |
| `MultiwavelengthModel` | Future-proof, describes scope | Too long, unwieldy |
| `EmissionModel` | Evokes spectrum | Confused with "emission lines" specifically |
| `GalaxyModel` | Clear intent | Too restrictive (AGN-only use cases) |
| `PhysicalModel` | General | Too vague |

**Decision:** `SEDModel`. When spatial models arrive, they can be `SpatialSEDModel` or
`IFUModel` — and combining them is `tengri.combine(sed_model, spatial_model)`. The
`Model` name becomes available for a future abstract base.

Migration path: `Model` becomes a deprecated alias pointing to `SEDModel` with a
`DeprecationWarning`, removed in v1.0.

---

## Part III — Full Class Naming Audit

### Priority 1: Rename (breaking, needs deprecation aliases)

| Current | Rename to | Reason |
|---------|-----------|--------|
| `Model` | `SEDModel` | Too generic; future spatial models need the namespace |
| `ParamSpec` | `Parameters` | "Spec" is vague; `Parameters` is what it is |
| `SpectroscopyConfig` | `Spectroscopy` | Data container, not a structural config; matches `Photometry` |
| `NoiseConfig` | `NoiseModel` | It's a model (Student-t vs Gaussian), not a config choice |
| `HierarchicalResult` | `PopulationPosterior` | Matches `Posterior` naming; "hierarchical" is an impl detail |
| `HierarchicalFitter` | `PopulationFitter` | Matches `Fitter`; "hierarchical" is an inference approach |
| `LineCatalog` | `LineList` | "Catalog" sounds like a database table; it's a list of lines |

### Priority 2: Standardize (non-breaking additions first)

| Current | Issue | Fix |
|---------|-------|-----|
| `double_powerlaw()` + `dpl()` | Both exported, redundancy | Deprecate `dpl`, keep `double_powerlaw` as canonical |
| `VIConfig` | Exported but rarely user-facing | Rename to `VISettings` for clarity, or keep as-is (less urgent) |
| `BakedInBackend` | What's "baked in"? | Rename to `SSPLineBackend` (lines from SSP grids) |
| `DoubletConstraint` | Fine internally, shouldn't be in public API | Un-export from `__init__.py` |
| `FilterCurve` | Fine internally, shouldn't be in public API | Un-export from `__init__.py` |

### Priority 3: Add missing exports

| Symbol | File | Issue |
|--------|------|-------|
| `uses_student_t()` | `core/noise.py` | Imported in root `__init__.py` but not in `__all__` |
| `variable_noise_hamiltonian()` | `core/noise.py` | Same — add or remove |
| `powerlaw_sfh()` | `models/sfh/mean_sfh.py` | In sfh `__init__` but not `__all__` |
| `continuity_sfh()`, `dirichlet_sfh()` | `models/sfh/nonparametric.py` | Missing from sfh `__all__` |
| `closed_box_metallicity()` family | `models/sfh/chemical_evolution.py` | Missing from sfh `__all__` |

### SFH function names (low priority, but worth documenting)

These function names are consistent with the paper but opaque to new users:

| Current | What it is | Consider aliasing |
|---------|-----------|-------------------|
| `tsnorm` | Truncated skew-normal | `truncated_skewnormal_sfh` as canonical, `tsnorm` as alias |
| `snorm` | Skew-normal | `skewnormal_sfh` |
| `lnorm` | Log-normal | `lognormal_sfh` |
| `norm` | Normal (Gaussian) | `gaussian_sfh` |
| `dpl` | Double power-law | Deprecate; use `double_powerlaw` |

---

## Part IV — Method Naming Audit

### Critical: `summary()` return type mismatch

```python
Model.summary()     → str   # human-readable text
Fitter.summary()    → str   # human-readable text
Posterior.summary() → dict  # ← WRONG — breaks principle of least surprise
```

**Fix:** Rename `Posterior.summary()` → `Posterior.stats()` (returns dict of per-parameter
statistics). Keep `Posterior.summary_table()` as the `str` method (already exists). Users
expecting `str` from `.summary()` will be surprised by a `dict`.

### Medium: `Posterior` property vs method inconsistency

```python
result.derived          # @cached_property — no parens
result.line_fluxes      # @cached_property — no parens
result.bpt_nii()        # method — requires parens
result.balmer_decrement()  # method — requires parens
result.equivalent_widths() # method — requires parens
```

**Rule to adopt:** Properties for quantities that are always cheap and always valid.
Methods for quantities that may fail (non-detections), require arguments, or do
non-trivial computation. The current split is correct in intent — but needs a docstring
convention that explains which is which for every item.

### Low: `sed_pipeline.py` dispatch suffix inconsistency

```python
interp_metallicity()              # no suffix
interp_metallicity_evolving()     # no suffix
interp_met_alpha_dispatch()       # _dispatch suffix
interp_met_alpha_evolving_dispatch()  # _dispatch suffix
```

**Fix:** Drop `_dispatch` suffix everywhere — it's redundant (dispatch is implied by the
branching logic). Since these are internal functions, this is a non-breaking change.

### Method grouping for `SEDModel` (ordered by user mental model)

The 35+ methods on `Model` currently have no documented grouping. The intended grouping:

```
Construction:      __init__(), from_config()
Display:           summary(), tree(), recommend_method()
Prior checks:      prior_predictive()
Prediction:        predict(), predict_sed(), predict_photometry(), predict_spectrum(),
                   predict_sfh(), predict_derived(), predict_magnitudes(),
                   predict_luminosity(), predict_sed_quantities(), predict_sfh_quantities()
Mocks:             mock(), mock_spectrum(), mock_batch()
Fitting:           fit()            ← single object, returns Posterior
                   fit_batch()      ← independent multi-object batch (no shared params)
                   fit_population() ← hierarchical, shared hyperparameters (PSD, dust prior)
Precomputation:    precompute_spectroscopy(), precompute_ztable()
```

### `fit_batch` vs `fit_population` (document prominently everywhere)

These two methods have fundamentally different semantics and must not be conflated:

- **`fit_batch(observations)`** — fits N galaxies **independently**. No shared parameters.
  Returns a list of `Posterior` objects. Use for: catalogs where you want individual
  posteriors fast. Internally uses `vmap` or `HierarchicalFitter` with fixed priors.

- **`fit_population(observations)`** — fits N galaxies **jointly** with shared
  hyperparameters (e.g., population-level PSD amplitude, dust prior). Returns a single
  `PopulationPosterior`. Use for: hierarchical inference, recovering the PSD of a galaxy
  population, constraining the dust prior from data.

The removed `fit_catalog()` was an undocumented alias for `fit_batch()` — it is gone.

---

## Part V — Parameter Naming Conventions

### What works (keep as-is)

The underscore-prefix system is correct and should be preserved:

| Domain | Prefix | Example |
|--------|--------|---------|
| Metallicity | `met_` | `met_logzsol`, `met_alpha_fe` |
| Dust attenuation | `dust_` | `dust_tau_bc`, `dust_tau_diff`, `dust_slope` |
| Star formation | `sfh_{type}_` | `sfh_tsnorm_log_total_mass`, `sfh_dpl_alpha` |
| GP field | `sfh_field_` | `sfh_field_psd_sigma`, `sfh_field_psd_tau_myr` |
| Nebular | `neb_` | `neb_logU`, `neb_logZ_gas`, `neb_fesc` |
| Emission lines | `eline_` | `eline_sigma_kms`, `eline_broad_sigma_kms` |
| AGN | `agn_` | `agn_log_lbol`, `agn_tau_skirtor` |
| Radio | `radio_` | `radio_q_ir`, `radio_loudness` |
| X-ray | `xray_` | `xray_gamma_agn`, `xray_alpha_ox` |
| Shock | `shock_` | `shock_frac`, `shock_velocity` |
| Chemical evolution | `chem_` | `chem_yield`, `chem_eta_outflow` |

### What to eliminate: the short name layer

The three-layer system is the main source of confusion for new users:

```
Layer 1: short names     "log_total_mass"            (user convenience)
Layer 2: public names    "sfh_tsnorm_log_total_mass"  (canonical public)
Layer 3: internal names  "log_total_mass"            (internal computation)
```

Layer 1 and 3 are identical in spelling but mean different things — and the mapping is
SFH-type-dependent. This creates silent bugs when users mix SFH types.

**Recommendation:** Eliminate `resolve_short_names()` from the public API. Remove short
names from `Model.from_config()`. Users write the full name; autocomplete handles the
rest. This is a small verbosity cost for a large correctness and clarity gain.

**One exception:** `"logzsol"` → `"met_logzsol"` is so universal it stays as a
documented alias. Everything else requires the full prefixed name.

### Unit conventions (canonical reference)

| Quantity | User-facing unit | Internal unit | Conversion |
|----------|-----------------|---------------|------------|
| Lookback times | Gyr | yr | ×10⁹ |
| PSD timescale | Myr | yr | ×10⁶ |
| Metallicity | log(Z/Z☉) | log(Z) absolute | +LOG10_ZSUN |
| Wavelength | Å | Å | 1:1 |
| SFR | M☉/yr | M☉/yr | 1:1 |
| Luminosity | L☉ | L☉ | 1:1 |

The Myr/yr split for PSD timescale is the most error-prone: `sfh_field_psd_tau_myr` in
the public API converts to `psd_tau_yr` internally via ×10⁶. This conversion must appear
in every docstring that touches this parameter.

---

## Part VI — Module Organization (if starting fresh)

### `fitter.py` (4,305 lines) → a package

The single biggest architectural mistake. Every inference method is independent code that
shares only the `log_prob` function signature — none of them share implementation. This
makes `fitter.py` effectively four separate files stapled together:

```
inference/
├── __init__.py       ← Fitter class, run() dispatch table only (~150 lines)
├── map.py            ← MAP + optax optimizers
├── vi.py             ← geoVI + MGVI + NIFTy fast path (all VI variants)
├── mcmc.py           ← NUTS + Ray Tracing + Elliptical Slice
├── nested.py         ← NSS evidence computation
├── laplace.py        ← Laplace approximation + Pathfinder
├── batch.py          ← fit_batch independent dispatch
├── hierarchical.py   ← PopulationFitter + PopulationPosterior
└── common.py         ← shared: standardize, log_prob, InferenceResult
```

### `model.py` (2,484 lines) → split by concern

```
core/
├── sed_model.py      ← SEDModel class: __init__, from_config, tree, summary (~400 lines)
├── prediction.py     ← predict_*() methods (already partially extracted)
├── fitting.py        ← fit(), fit_batch(), fit_population() delegation
└── mock.py           ← mock(), mock_spectrum(), mock_batch(), PriorPredictive
```

### `models/observation/` (13 files) → lines/ subpackage

```
observation/
├── __init__.py
├── photometry.py         ← Photometry, FilterCurve, filter loading
├── spectroscopy.py       ← Spectroscopy (was SpectroscopyConfig), LSF, calibration
├── observation.py        ← Observation container
├── noise.py              ← NoiseModel (was NoiseConfig)
└── lines/
    ├── __init__.py
    ├── list.py           ← LineList (was LineCatalog)
    ├── priors.py         ← CLOUDY/Cue interpolated priors
    └── marginalization.py ← design matrix + analytical marginalization
```

---

## Part VII — Inference Method Taxonomy (keep as-is, document better)

The NIFTy and geoVI names are preserved because they appear in the paper. The canonical
taxonomy is correct — it needs only a documentation fix to explain the relationship
between the library (NIFTy) and the algorithm (geoVI):

> "NIFTy is the computational library implementing geoVI; the `vi` method uses its fast
> path by default."

```
Approximate posteriors (fast):
  "map"             — maximum a posteriori (point estimate)
  "laplace"         — Gaussian approximation at MAP via Hessian
  "pathfinder"      — L-BFGS path approximate posterior (Zhang+2022)
  "vi"              — geometric variational inference (geoVI, NIFTy fast path) ← DEFAULT
  "vi_linear"       — linear response variant (MGVI/EVI)
  "vi_nifty"        — NIFTy geoVI with full logging (debugging/paper reproducibility)
  "vi_nifty_linear" — NIFTy MGVI with full logging

Exact posteriors (slow):
  "mcmc"            — auto-select based on dimensionality
  "mcmc_raytrace"   — Ray Tracing HMC (Behroozi 2025); stochastic-gradient resilient
  "mcmc_nuts"       — No-U-Turn Sampler (gold standard, D ≤ 30)
  "mcmc_ess"        — Elliptical Slice Sampling (Murray+2010, Gaussian-prior models)

Model selection:
  "evidence"        — log Z via Nested Slice Sampling (Yallup+2026)
```

---

## Part VIII — Implementation Plan

Changes fall into four tiers, ordered by risk and impact:

### Tier 1 — Documentation only (no code changes)

- Write this design doc to `docs/superpowers/specs/` ✓
- Add method groupings comment block to `SEDModel` docstring
- Document `fit_batch` vs `fit_population` distinction prominently in both methods
- Add NIFTy/geoVI relationship sentence to all inference docstrings
- Document unit conventions in one canonical location (this file + param_translate.py)

### Tier 2 — Non-breaking additions (backward-compatible, do now)

- Add `Posterior.stats()` alongside `Posterior.summary()` (new name for dict return)
- Add `PopulationFitter` and `PopulationPosterior` as aliases for `HierarchicalFitter`/`HierarchicalResult`
- Add `LineList` as alias for `LineCatalog`
- Add `Spectroscopy` as alias for `SpectroscopyConfig`
- Add `NoiseModel` as alias for `NoiseConfig`
- Add `Parameters` as alias for `ParamSpec`
- Add `SEDModel` as alias for `Model` (with DeprecationWarning on `Model`)
- Fix `__all__` gaps: add `uses_student_t`, `powerlaw_sfh`, `continuity_sfh`, `dirichlet_sfh`, chemical_evolution functions
- Deprecate `dpl()` in favor of `double_powerlaw()` (emit DeprecationWarning)

### Tier 3 — Breaking renames (v1.0 target, deprecation warnings now)

- `Model` → `SEDModel`
- `ParamSpec` → `Parameters`
- `SpectroscopyConfig` → `Spectroscopy`
- `NoiseConfig` → `NoiseModel`
- `HierarchicalResult` → `PopulationPosterior`
- `HierarchicalFitter` → `PopulationFitter`
- `LineCatalog` → `LineList`
- `Posterior.summary()` → `Posterior.stats()` (return type change: dict → use .stats(), str → use .summary_table())
- Remove `resolve_short_names()` from public API (keep `"logzsol"` alias only)
- Drop `_dispatch` suffix from `sed_pipeline.py` internal functions

### Tier 4 — Structural refactors (future sprint, not this cycle)

- Split `fitter.py` into `inference/` subpackage (see Part VI)
- Split `model.py` into `core/` subpackage (see Part VI)
- Reorganize `observation/` with `lines/` subpackage (see Part VI)
- Move `param_translate.py` to per-module ownership
- Rename `models/` → `physics/` and `models/sps/` → `physics/stellar/`
