# Tengri Refactor Plan: Lessons from Organic Growth

## Why This Document Exists

Tengri grew organically without a design contract. The physics and math are excellent (geoVI,
PSD-correlated SFH, differentiable Ray Tracing). The engineering structure is fragmented. This
document records the root causes, what should have been done upfront, and a prioritized refactor
plan with scoped subagent tasks.

---

## Root Causes (in severity order)

### 1. No design contract before code

Class naming conventions were never agreed on. The result:

- `SpectroscopyConfig` is a data container (should be `Spectroscopy`)
- `NoiseConfig` is a statistical model (should be `NoiseModel`)
- `ParamSpec` is a parameter registry (should be `Parameters`)
- `HierarchicalFitter` is an orchestrator (should be `PopulationFitter`)
- `LineCatalog` sounds like a DB table (should be `LineList`)

Parameter naming grew into 3 layers with no explicit contract:

- Short user-facing: `log_peak_sfr`
- Full prefixed public: `sfh_tsnorm_log_peak_sfr`
- Internal: `log_peak_sfr` (same spelling, type-dependent meaning — silent bug factory)

Module boundaries were never defined. `fitter.py` became a 4305-line god file because nobody
decided what "inference" owns vs. what "model" owns.

### 2. Settings and parameters were never separated at the type level

Settings = *which sub-model is active* (dust law, nebular backend) — not in gradient tape.
Parameters = *scalars with priors that get fitted* — in gradient tape.

These are fundamentally different types. They were conflated in `ParamSpec` and `Model.__init__`,
making both enormous. `DustConfig`, `NebularConfig`, `AGNConfig` were designed as frozen
dataclasses much later and retrofitted. This is the single largest structural mistake.

### 3. Physics categories accumulated without defaults or selection guidance

- SFH models: 8+ (dpl, tsnorm, snorm, norm, lnorm, exp, dexp, const, field, field+burst)
- Dust attenuation: 6 laws
- Dust emission: 5+ models
- AGN: 4 models (multicolor_disc, unified_nlr_blr, kubota_done_full, qsogen)
- Inference: 14+ named methods for 5 distinct algorithms

No guidance on which to use. No recommended defaults. Each was added as "another option" rather
than replacing or superseding a prior option.

### 4. The "tier 2" SED path was never built

- **Tier 1**: Fused photometry (fixed-z + fixed filters) → ~140 µs, 20× speedup
- **Tier 2**: Compositional rest-frame SED (all physics, free-z, JIT'd) → ~300–500 µs ← **MISSING**
- **Tier 3**: Exact fallback (tabulated SFH, evolving Z, Python dispatch) → ~500–1000 µs

Designed in `docs/design_compositional_sed.md`. Never implemented. All non-tier-1 work falls
through to slow tier 3.

### 5. Gradient correctness was never systematically tested

All gradient tests check `isfinite()` only. They would pass with wrong signs or missing terms. The
CLOUDY marginalization bug (wrong ln_L, biases MAP/VI gradients) passed all tests.
`cloudy_grid_line_priors()` has zero coverage.

### 6. External dependency traps

NIFTy had 99.8% Python overhead (partial wrapping, type checking, dict dispatch). Required full
reimplementation in JAX primitives. The lesson: isolate external library APIs behind internal
abstractions from day one; do not let NIFTy types leak into inference logic.

### 7. Documentation outran implementation

`docs/models/` has full write-ups for: chemical evolution Z(t), shock emission (MAPPINGS), ADAF
disc, MAGPHYS dust, THEMIS dust, patchy IGM reionization, alpha-enhanced SSPs — all with zero or
near-zero code. Creates false scope expectations and makes the project feel unmanageable.

---

## What Should Have Been Done First (Pre-Implementation Checklist)

### Step 0A — Write 3 usage archetypes as runnable scripts

The scripts define every class name, method name, and parameter name before implementation. Run
them as the "north star."

```python
# Grad student: photometric fit, parametric SFH
model = SEDModel.from_config(
    ssp="fsps_default", sfh="dpl", dust="charlot_fall",
    filters=["sdss_u", "sdss_g", "sdss_r"], redshift=0.1,
)
result = model.fit(flux, noise)

# Expert: stochastic SFH, spectroscopy, MCMC validation
params = Parameters(sfh="field", dust="two_component", nebular="cloudy")
params.add("sfh_field_psd_sigma", LogUniform(0.1, 3.0))
result = SEDModel(ssp=grid, settings=settings, params=params, obs=obs).fit().refine("mcmc")

# Population: hierarchical PSD hyperparameters
pop = SEDModel.fit_population(
    obs_list, shared=["sfh_field_psd_sigma", "sfh_field_psd_tau_myr"]
)
pop.plot_population()
```

### Step 0B — Write the suffix convention table before any class definitions

| Suffix      | Meaning                                          | Examples                          |
|-------------|--------------------------------------------------|-----------------------------------|
| (none)      | Concrete class with behavior                     | `SEDModel`, `Fitter`, `Posterior` |
| `Config`    | Frozen settings, NOT in gradient tape            | `DustConfig`, `NebularConfig`     |
| `Backend`   | Swappable physics implementation                 | `CloudyBackend`, `CueBackend`     |
| `Model`     | Statistical model                                | `NoiseModel`, `CalibrationModel`  |
| `List`      | Registry/catalog                                 | `LineList`, `FilterList`          |
| `-er` suffix | Active orchestrator                             | `Fitter`, `PopulationFitter`      |

Forbidden: `Spec` (vague), `Config` on data containers, `Result` (use `Posterior`).

### Step 0C — Define the unit contract table

Every parameter docstring must state: user-facing unit / internal unit / conversion.

| Parameter           | User-facing | Internal   | Conversion  |
|---------------------|------------|------------|-------------|
| `sfh_field_psd_tau` | Myr        | yr         | × 10⁶       |
| `sfh_*_tau_peak`    | Gyr        | yr         | × 10⁹       |
| `met_logzsol`       | log(Z/Z☉)  | log(Z) abs | + LOG10_ZSUN |

### Step 0D — Draw the module boundary tree with ownership rules

```
tengri/
├── core/
│   ├── model.py         < 300 lines: thin orchestrator, no physics
│   ├── parameters.py    parameter registry + prior definitions only
│   ├── settings.py      DustConfig, NebularConfig, AGNConfig
│   ├── param_map.py     name translation (one layer: public → internal)
│   ├── fused_kernels.py JIT kernel builders
│   ├── sed_pipeline.py  SED computation engine
│   └── noise.py         NoiseModel
├── inference/
│   ├── fitter.py        < 150 lines: dispatch table only
│   ├── map.py           MAP + optimizer
│   ├── vi.py            geoVI + MGVI (JIT path + NIFTy path)
│   ├── mcmc.py          NUTS + Ray Tracing + ESS
│   ├── evidence.py      Nested Slice Sampling
│   ├── population.py    PopulationFitter
│   └── posterior.py     Posterior + PopulationPosterior
└── models/              physics (unchanged structure, but with 1 default per category)
```

**Dependency rule**: `core/` never imports from `inference/`. `inference/` never imports from
`models/` directly (only through `core/`). Physics modules only import from `utils/`.

### Step 0E — Define the parameter namespace before any Parameters code

```
{sfh_type}_{param}      sfh_dpl_alpha, sfh_field_psd_sigma
met_{param}             met_logzsol
dust_{param}            dust_tau_bc, dust_tau_diff, dust_slope
neb_{param}             neb_logU, neb_logZ_gas, neb_fesc
agn_{submodel}_{param}  agn_disc_log_mbh, agn_torus_frac
eline_{param}           eline_broad
noise_{param}           noise_f_cal, noise_dof
```

No short names in the public API except `logzsol` → `met_logzsol` (documented explicitly). No
internal names that shadow public names.

---

## Prioritized Refactor Plan

### Phase 1 — Structural Naming Contract (P0, Non-Breaking)

Add new names alongside old with `DeprecationWarning`. Remove old names in v1.0.

| Current                  | Rename to             | Reason                                          |
|--------------------------|-----------------------|-------------------------------------------------|
| `ParamSpec`              | `Parameters`          | Vague abbreviation                              |
| `SpectroscopyConfig`     | `Spectroscopy`        | Data container, not config                      |
| `NoiseConfig`            | `NoiseModel`          | It's a statistical model                        |
| `HierarchicalFitter`     | `PopulationFitter`    | "Hierarchical" is an implementation detail      |
| `HierarchicalResult`     | `PopulationPosterior` | "Hierarchical" is an implementation detail      |
| `LineCatalog`            | `LineList`            | Sounds like a database table                    |
| `Posterior.summary()`    | `Posterior.stats()`   | Returns dict (inconsistent with other `.summary()` → str) |

**Files**: `core/param_spec.py`, `inference/fitter.py`, `inference/posterior.py`,
`inference/hierarchical.py`, `models/observation/line_catalog.py`, `__init__.py`

---

### Phase 2 — Split God Files (P0)

#### 2A: Split `fitter.py` (4305 lines) → inference subpackage

Dispatch table in `fitter.py` (< 150 lines). Each algorithm in its own file:

- `inference/map.py` — MAP + optax optimizers (extract from `map_optimizer.py` + fitter internals)
- `inference/vi.py` — geoVI + MGVI, JIT fast path + NIFTy backend (extract `_run_fast_vi`,
  `_run_native_vi`, `_run_nifty_vi`)
- `inference/mcmc.py` — NUTS + Ray Tracing + ESS (extract `_run_raytrace`, `_run_nuts`,
  `_run_elliptical_slice`)
- `inference/evidence.py` — NSS (extract `_run_nss`)
- `inference/population.py` — extract from `hierarchical.py`

#### 2B: Split `core/model.py` (2538 lines)

- `core/model.py` — thin orchestrator (< 300 lines): `__init__`, `from_config()`, `fit()`,
  dispatch to `sed_pipeline`
- `core/convenience.py` — `prior_predictive()`, `fit_batch()`, `fit_population()`,
  `fit_catalog()` wrappers
- `core/mock.py` — mock galaxy generation (already partially extracted)

---

### Phase 3 — Settings/Parameters Split (P0, Breaking)

Introduce explicit `core/settings.py` with frozen dataclasses. Migrate sub-model selection out of
`ParamSpec.__init__`.

```python
# Before (conflated)
spec = ParamSpec(sfh="field", nebular=True, cloudy_grid_path="...")
spec.add("neb_logU", Uniform(-4, -1))

# After (explicit split)
settings = ModelConfig(
    sfh=FieldSFHConfig(n_grid=50),
    nebular=NebularConfig(backend="cloudy", grid_path="..."),
)
params = Parameters()
params.add("neb_logU", Uniform(-4, -1))
model = SEDModel(ssp=grid, settings=settings, params=params, obs=obs)
```

`AGNConfig`, `DustConfig`, `NebularConfig` already exist — the work is making them the *primary*
interface rather than a retrofit. Move all sub-model selection flags out of `ParamSpec.__init__`
entirely.

**Files**: `core/param_spec.py` (split into `parameters.py` + `settings.py`), `core/model.py`

---

### Phase 4 — Canonical Method Names (DONE — do not change)

The canonical inference method strings are settled and documented. They are **not** candidates
for further collapse into a `method+backend` flag API.

**Current canonical names** (as of 2026-04-02):

| Canonical string  | Algorithm                          | Deprecated aliases removed in v1.0      |
|-------------------|------------------------------------|-----------------------------------------|
| `"vi"`            | geoVI (NIFTy fast path, default)   | `geovi`, `native_geovi`, `fast_geovi`   |
| `"vi_linear"`     | MGVI / EVI (linear response)       | `mgvi`, `native_mgvi`, `fast_mgvi`      |
| `"vi_nifty"`      | NIFTy geoVI with full logging      | `nifty_geovi`                           |
| `"vi_nifty_linear"` | NIFTy MGVI with full logging     | `nifty_mgvi`                            |
| `"mcmc_raytrace"` | Ray Tracing HMC (Behroozi 2025)    | `raytrace`                              |
| `"mcmc_nuts"`     | No-U-Turn Sampler (blackjax)       | `nuts`                                  |
| `"mcmc_ess"`      | Elliptical Slice Sampling          | `elliptical_slice`                      |
| `"evidence"`      | Nested Slice Sampling (log Z)      | `nss`                                   |
| `"map"`           | MAP point estimate                 | —                                       |

**Why this design is better than `method="vi", backend="nifty"`:**

1. **Strings are unambiguous.** `fitter.run("vi_nifty")` is self-documenting in a notebook cell or
   log line. `fitter.run("vi", backend="nifty")` requires the reader to hold two arguments in mind
   simultaneously and understand the cross-product of valid `(method, backend)` combinations.

2. **Not all combinations are valid.** `method="mcmc", sampler="nuts"` and
   `method="mcmc", sampler="raytrace"` differ in more than the sampler — they have different keyword
   arguments (`n_warmup` vs `n_steps`, `step_size`, etc.). Flattening them into a shared namespace
   creates an argument collision problem. String dispatch keeps the call signatures independent.

3. **Paper reproducibility.** MCMC results in the paper are cited as "mcmc\_raytrace with
   `n_steps=1000`". That string appears in notebooks, CLAUDE.md, and the methods section. Changing
   it to `method="mcmc", sampler="raytrace"` would break that traceability without benefit.

4. **The proliferation was already solved.** The 14+ old method strings were aliases for 5 distinct
   algorithms. The aliases are now deprecated with `DeprecationWarning` and removed in v1.0. What
   remains is 9 canonical strings for 9 distinct dispatch paths — not proliferation.

**Action:** Remove `_DEPRECATED_METHOD_ALIASES` dict from `fitter.py` in v1.0. Update any
remaining alias callsites in `tests/` and `notebooks/` at that time. No API change needed now.

---

### Phase 5 — Build Tier 2 SED Path (P1)

Implement `build_fused_rest_sed(settings)` factory in `core/fused_kernels.py`:

- Produces rest-frame SED at SSP resolution, end-to-end JIT'd (~300–500 µs)
- Supports ALL physics components (unlike tier 1 which requires fixed-z)
- Called at `SEDModel.__init__` with graceful tier-3 fallback
- Dispatch logic: tier-1 check (fixed-z + photometry) → tier-2 check (all components JIT-able?)
  → tier-3 exact fallback

Component subfunctions to extract from `core/sed_pipeline.py`:

```python
_compute_stellar_sed(params, ssp, sfh_weights)
_apply_dust_attenuation(sed, params, config)
_add_dust_emission(sed, params, config)
_add_agn(sed, params, config)
_add_nebular(sed, params, config, nebular_backend)
```

**Files**: `core/fused_kernels.py`, `core/sed_pipeline.py`, `core/model.py`

---

### Phase 6 — Science Correctness Fixes (P0, Run in Parallel with Phases 1–2)

**6A: CLOUDY 2D interpolation** (`models/nebular/cloudy_grid.py:169–180`)

- Bug: `cloudy_line_priors()` does 1D interpolation over Z only; misses logU dimension
- Fix: Replace with 2D bilinear interpolation over (Z, logU) grid
- Test: Interpolated values match reference grid values at grid points

**6B: Marginalization ln_L normalization** (`models/observation/eline_priors.py:248–278`)

- Bug: `marginalize_emission_lines_cloudy` missing `+0.5 × (μ² / σ²)` normalization term for
  non-zero-mean prior — biases MAP/VI gradients; MCMC unaffected (uses likelihood ratios)
- Fix: Add normalization term; add finite-difference gradient test

**6C: Gradient correctness test infrastructure** (`tests/unit/test_gradients.py`)

- Every transform inside a JIT boundary needs a finite-difference test
- Minimum suite: `test_dust_attenuation_gradient`, `test_nebular_marginalization_gradient`,
  `test_sfh_transform_gradient`, `test_igm_transmission_gradient`

---

### Phase 7 — Prune Documentation (P2)

Move any model documented with zero code from `docs/models/` → `ROADMAP.md`:

- `docs/models/chemical_evolution.md`
- `docs/models/shock_emission.md`
- `docs/models/adaf_disc.md`
- `docs/models/magphys_dust.md`
- `docs/models/themis_dust.md`
- `docs/models/patchy_igm.md`
- `docs/models/pah_features.md` (partial: keep API design, move implementation section)

---

## Subagent Scopes

All scopes are independent except G (depends on C completing first).

### Subagent A — Naming Refactor (Phase 1)

**Scope**: Rename `ParamSpec→Parameters`, `SpectroscopyConfig→Spectroscopy`,
`NoiseConfig→NoiseModel`, `HierarchicalFitter→PopulationFitter`,
`HierarchicalResult→PopulationPosterior`, `LineCatalog→LineList`,
`Posterior.summary()→Posterior.stats()`.

**Strategy**: Add new names with `DeprecationWarning` on old; update `__init__.py` exports;
update all tests, notebooks, and CLAUDE.md.

**Files**: `core/param_spec.py`, `inference/fitter.py`, `inference/posterior.py`,
`inference/hierarchical.py`, `models/observation/line_catalog.py`, `__init__.py`, `tests/`,
`notebooks/`

**Constraint**: Non-breaking — old names still work, just warn.

---

### Subagent B — Split fitter.py (Phase 2A)

**Scope**: Extract `_run_fast_vi` + `_run_native_vi` + `_run_nifty_vi` →
`inference/vi.py`; extract `_run_raytrace` + `_run_nuts` + `_run_elliptical_slice` →
`inference/mcmc.py`; extract `_run_nss` → `inference/evidence.py`; extract MAP logic →
`inference/map.py`. Leave `fitter.py` as a < 150-line dispatch table.

**Files**: `inference/fitter.py`, `inference/vi.py` (new), `inference/mcmc.py` (new),
`inference/evidence.py` (new), `inference/map.py` (new), `inference/__init__.py`

**Constraint**: All 1764 existing tests must pass. Zero behavior changes.

---

### Subagent C — Split model.py (Phase 2B)

**Scope**: Extract convenience methods (`prior_predictive`, `fit_batch`, `fit_population`,
`fit_catalog`) → `core/convenience.py`. Leave `model.py` as thin orchestrator < 400 lines.

**Files**: `core/model.py`, `core/convenience.py` (new), `core/__init__.py`

**Constraint**: Public API unchanged. All tests pass.

---

### Subagent D — Science Fix: CLOUDY 2D Interpolation (Phase 6A)

**Scope**: Fix `cloudy_line_priors()` in `models/observation/eline_priors.py:169–180` to
perform true bilinear interpolation over the (logZ_gas, logU) grid.

**Root cause**: The function blends `ratios_logU3` (Z-varying, logU=-3) with
`ratios_solar_u` (solar Z only, logU=-2). At `u_frac=1` (logU=-2) the result is pure solar
regardless of `log_z` — the metallicity dimension is completely dropped at one grid edge.
The fix is adding a fourth grid point `_CLOUDY_SUBSOLAR_LOGU2` and wiring proper bilinear
interpolation:
```python
ratios_logu3 = lerp(subsolar_u3, solar_u3, z_frac)   # Z interpolation at logU=-3
ratios_logu2 = lerp(subsolar_u2, solar_u2, z_frac)   # Z interpolation at logU=-2
result = lerp(ratios_logu3, ratios_logu2, u_frac)    # logU interpolation
```

**Files**: `models/observation/eline_priors.py`, `tests/unit/test_eline_priors.py`

**Note**: This is NOT in `models/nebular/cloudy_grid.py` — that file handles the HDF5 grid
interpolation via `cloudy_grid_line_priors()`, which is a separate function. The bug is in
the simpler analytic prior `cloudy_line_priors()` in `eline_priors.py`.

**Constraint**: New interpolation path must pass finite-difference gradient check. Regression
test from `docs/known_bugs.md` (NEW-01) must be green after fix:
```python
def test_cloudy_priors_metallicity_effect_at_high_logu():
    means_solar, _ = cloudy_line_priors(log_z=0.0, neb_logU=-2.0)
    means_subsolar, _ = cloudy_line_priors(log_z=-0.7, neb_logU=-2.0)
    assert means_subsolar[8] < 0.5 * means_solar[8]  # [NII]6583 weaker at sub-solar Z
```

---

### Subagent E — Science Fix: Marginalization ln_L (Phase 6B)

**Scope**: Fix `marginalize_emission_lines_cloudy` in `models/observation/eline_priors.py` to
include missing `+0.5 × μ² / σ²` normalization term in ln_L for non-zero-mean prior.

**Files**: `models/observation/eline_priors.py`, `tests/unit/test_eline_priors.py`

**Constraint**: MCMC is unaffected (likelihood ratios cancel normalization). MAP/VI gradients
must be correct per finite-difference check after fix.

---

### Subagent F — Gradient Test Infrastructure (Phase 6C)

**Scope**: Create `tests/unit/test_gradients.py` with finite-difference gradient checks for:
(1) CSP mass weights, (2) dust attenuation transforms, (3) nebular emission marginalization,
(4) IGM transmission, (5) AGN disc spectrum. Use `jax.test_util.check_grads` or manual FD
with `eps=1e-5`.

**Files**: `tests/unit/test_gradients.py` (new)

**Constraint**: Tests must run without SSP data (use mock inputs). Full suite must complete in
< 30 s.

---

### Subagent G — Tier 2 SED Path (Phase 5, depends on Subagent C)

**Scope**: Implement `build_fused_rest_sed(settings)` factory in `core/fused_kernels.py`.
Extract 5 component subfunctions from `core/sed_pipeline.py`. Wire tier-2 dispatch in
`core/model.py`. Add benchmark test asserting free-z forward pass < 600 µs on CPU.

**Files**: `core/fused_kernels.py`, `core/sed_pipeline.py`, `core/model.py`

**Constraint**: Tier-1 (precomputed photometry) behavior unchanged. Tier-3 fallback still works
for tabulated SFH and evolving metallicity.

---

---

## Items Not Covered Above

### BUG-29: `_mstar` uses formed mass, not surviving mass (`sed_pipeline.py:651`)

`jnp.sum(weights)` is total formed stellar mass. XRB luminosity is calibrated against
surviving stellar mass (which is ~30–50% lower for old galaxies). This causes systematic
X-ray overestimation for evolved systems. Fix requires computing surviving mass fraction
from DSPS output. Moderate impact; not blocking Paper I but needs tracking.

**Not assigned to any subagent above.** Add to a future science-correctness subagent or fix
inline when touching `sed_pipeline.py` for Phase 5.

### Phase 7 clarification: what to do with each stub doc

"Move to ROADMAP.md" is underspecified. Concrete rule:

- **Delete** the implementation section entirely (code examples, parameter tables).
- **Keep** a 1–3 sentence physical motivation blurb in `ROADMAP.md` under a `## Planned
  Physics Modules` heading.
- **Keep** the API design section only if the interface contract was agreed on and is stable
  (e.g. if a `NebularConfig` already exists that would wrap it).

Files affected: `docs/models/chemical_evolution.md`, `shock_emission.md`, `adaf_disc.md`,
`magphys_dust.md`, `themis_dust.md`, `patchy_igm.md`, `pah_features.md` (partial).

### Phase 6C gradient test gaps

The listed tests in Subagent F are the minimum. Also missing:
- `cloudy_grid_line_priors()` — zero test coverage (NEW-07 in `docs/known_bugs.md`)
- `marginalize_emission_lines_cloudy` finite-difference gradient (NEW-09)
- `cloudy_line_priors` bilinear interpolation gradient (needed after Subagent D fix)

---

## Rule for New Features Going Forward

Before adding any new model or inference method:

1. **Write the docstring first.** If you cannot describe it in 3 lines, the abstraction is wrong.
2. **Add the parameter to the namespace table** in `docs/parameter_reference.md`. If the name
   clashes or doesn't fit the prefix scheme, rename before writing code.
3. **Write a finite-difference gradient test.** If the transform is not differentiable, it cannot
   go inside a JIT boundary.
4. **If the code does not exist, do not document it in `docs/models/`.** Put it in `ROADMAP.md`.
