# Tengri Naming Contract

> **Authority:** This is the single source of truth for naming conventions.
> All new code, renames, and refactors MUST comply. Referenced by CLAUDE.md.
>
> **Last updated:** 2026-06-24
> **Current version:** v0.1-dev
> **Derived from:** `docs/dev/sessions/2026-04-03-api-naming-design.md`, `docs/dev/REFACTOR.md`

---

## 0. Versioning & Deprecation Policy

- **Current release:** pre-v1.0 (development).
- **Deprecation horizon:** Deprecated aliases emit `DeprecationWarning` throughout v0.x. All deprecated aliases will be **removed in v1.0**.
- **Changelog:** See `CHANGELOG.md` at repository root.

---

## 1. Class Suffix Convention

| Suffix | Role | Examples |
|--------|------|----------|
| (none) | Data container / physics object | `Photometry`, `Spectroscopy`, `Observation` |
| `Model` | Physical forward model or statistical model | `SEDModel`, `NoiseModel` |
| `Config` | Frozen structural choice (not fitted) | `DustConfig`, `NebularConfig`, `AGNConfig`, `VIConfig` |
| `Backend` | Swappable computation engine | `CloudyGridBackend`, `CueBackend` |
| `Fitter` / `-er` | Active orchestrator | `Fitter`, `PopulationFitter` |
| `Posterior` | Inference result (samples + diagnostics) | `Posterior`, `PopulationPosterior` |
| `List` | Registry / catalog | `LineList`, `FilterList` |
| `Error` | Exception classes (see §8) | `TengriError`, `ParameterError` |

**Forbidden suffixes:**
- `Spec` — too vague; use `Parameters` or a domain-specific name.
- `Result` — use `Posterior`.
- `Config` on **data containers** — `Config` is reserved for frozen structural choices only. Data containers (e.g., `Photometry`, `Spectroscopy`) carry no suffix.

---

## 2. Canonical Class Names

These are the **canonical** names. Old names are deprecated aliases that emit `DeprecationWarning` and will be removed in v1.0.

| Canonical name | Deprecated alias(es) | Defining module |
|---------------|---------------------|-----------------|
| `SEDModel` | `Model` | `core/model.py` |
| `Parameters` | `ParamSpec` | `core/parameters.py` |
| `Spectroscopy` | `SpectroscopyConfig` | `models/observation/spectroscopy.py` |
| `NoiseModel` | `NoiseConfig` | `models/observation/noise_model.py` |
| `LineList` | `LineCatalog` | `models/observation/line_list.py` |
| `PopulationFitter` | `HierarchicalFitter` | `inference/hierarchical.py` |
| `PopulationPosterior` | `HierarchicalResult` | `inference/hierarchical.py` |
| `SEDModelConfig` | `ModelConfig` | `config/settings.py` |

**File path stability rule:** Module filenames must match the canonical class name in `snake_case`. If a class is renamed, the file is renamed in the same commit. The table above is the ground truth; if it conflicts with the filesystem, the filesystem is wrong.

**Rule:** New code must use canonical names. Old names must never appear in new code, tests, docs, or examples.

---

## 3. Parameter Namespace

### Two layers only

| Layer | Purpose | Example |
|-------|---------|---------|
| **User-facing (prefixed)** | What users write in `Parameters` / `from_config()` | `sfh_dpl_alpha`, `dust_tau_bc`, `met_logzsol` |
| **Internal** | What physics functions receive after translation | `alpha`, `tau_bc`, `log_z_abs` |

### Prefix table

| Domain | Prefix | Examples |
|--------|--------|---------|
| Star formation | `sfh_{type}_` | `sfh_dpl_alpha`, `sfh_tsnorm_log_total_mass` |
| GP field | `sfh_field_` | `sfh_field_psd_sigma`, `sfh_field_psd_tau_myr`, `sfh_field_xi` |
| Metallicity | `met_` | `met_logzsol`, `met_alpha_fe` |
| Chemistry | `chem_` | *(reserved; populate when chemical evolution module is added)* |
| Dust | `dust_` | `dust_tau_bc`, `dust_tau_diff`, `dust_slope` |
| Nebular | `neb_` | `neb_logU`, `neb_logZ_gas`, `neb_fesc` |
| AGN | `agn_` | `agn_log_lbol`, `agn_tau_skirtor`, `agn_disc_log_mbh` |
| Emission lines | `eline_` | `eline_sigma_kms`, `eline_broad` |
| Noise | `noise_` | `noise_f_cal`, `noise_dof` |
| Radio | `radio_` | `radio_q_ir` |
| X-ray | `xray_` | `xray_gamma_agn` |
| Shock | `shock_` | `shock_frac` |
| Redshift | `redshift` | `redshift` (standalone; no trailing underscore) |

**CI-enforceable regex:** `^(sfh_|met_|dust_|neb_|agn_|eline_|noise_|radio_|xray_|shock_|chem_|redshift)`

This regex validates that every user-facing parameter name begins with a recognized domain prefix. It does **not** enforce deeper structure beyond the prefix; per-domain structural validation is the responsibility of `Parameters.validate()`.

### What is forbidden

- **Short names in public API** (e.g., bare `psd_sigma` instead of `sfh_field_psd_sigma`). Short names exist ONLY inside `resolve_short_names()` for `from_config()` convenience. One exception: `logzsol` → `met_logzsol`.
- **`_u` suffix params** (e.g., `psd_sigma_u`) leaking outside optimizer internals in `hierarchical.py`.
- **Internal names shadowing public names** (e.g., `log_total_mass` meaning different things at different layers).

### Unit contract

All unit translations are implemented in `core/parameters.py :: translate_to_internal()`. The internal variable names listed below are the canonical internal names used downstream of that function.

| User-facing param | User unit | Internal variable | Internal unit | Conversion |
|-------------------|-----------|-------------------|---------------|------------|
| `sfh_field_psd_tau_myr` | Myr | `psd_tau_yr` | yr | ×10⁶ |
| `sfh_*_tau_peak_gyr` | Gyr | `tau_sfh` | yr | ×10⁹ |
| `met_logzsol` | log(Z/Z☉) | `log_z_abs` | log(Z) absolute | + `LOG10_ZSUN` |
| `dust_tau_bc` | optical depth | `tau_bc` | optical depth | (identity) |
| Wavelength (all) | Å | — | Å | (identity) |
| SFR (all) | M☉/yr | — | M☉/yr | (identity) |

---

## 4. Function Verb Convention

| Verb | Meaning | Examples |
|------|---------|---------|
| `compute_*` | Pure function, returns new array | `compute_blr_sed()`, `compute_csp_sed()` |
| `load_*` | Reads from disk | `load_ssp_data()`, `load_cloudy_grid()` |
| `build_*` | Factory returning a callable/kernel | `build_kernel()`, `build_dl07_photometry_lookup()` |
| `apply_*` | Transform a spectrum in-pipeline | `apply_lsf()`, `apply_lyman_cutoff()` |
| `resolve_*` | Registry lookup by string key | `resolve_dust_law()`, `resolve_sfh()` |
| bare name | SFH shape functions registered as **registry keys** in `SFH_REGISTRY` ONLY (see §6) | `tsnorm`, `dpl` |

### Current violations to fix

| Current | Should be | File |
|---------|-----------|------|
| `blr_emission()` | `compute_blr_sed()` | `models/agn/blr.py` |
| `nlr_emission()` | `compute_nlr_sed()` | `models/agn/nlr.py` |
| `shock_emission_sed()` | `compute_shock_sed()` | `models/nebular/shock.py` |
| `qsogen_sed()` | `compute_qsogen_sed()` | `models/agn/qsogen.py` |
| `get_dust_law()` | `resolve_dust_law()` | `models/dust/attenuation.py` |
| `get_agn_model()` | `resolve_agn_model()` | `models/agn/unified.py` |
| `get_emission_model()` | `resolve_emission_model()` | `models/dust/emission.py` |

---

## 4b. Public Method Verbs & API Contracts

### 4b.1 Method verbs on public surfaces

The public surface uses exactly these verbs. Anything else is a contract violation:

| Verb | Meaning | Module | Examples |
|------|---------|--------|----------|
| `predict_*` | Forward model → observable or derived property | core | `predict_photometry()`, `predict_spectrum()` |
| `measure_*` | Extraction requiring no `SEDModel` — array in, number out | `measure` | `measure_index_jax()`, `measure_line_flux_jax()` |
| `mock*` | Synthesize data | `results` | `generate_mock()` |
| `fit*` | Inference entry point | `inference` | `Fitter.run()`, `PopulationFitter.run()` |
| `plot_*` | Visualization | `plot` | `plot_sed_fit()`, `plot_sfh()` |

**Rationale:** These verbs signal intent immediately. A user reading `model.predict_*` knows the call is deterministic and returns observables; `measure_*` signals extraction that needs no model; `fit*` signals inference.

**"Model-free" means no `SEDModel` — it does NOT mean "observed data."** This
distinction has already cost one wrong turn, so it is written down here. `measure`
is defined by what it does *not* require — a model and a forward pass — never by
where its input came from. Every operator is array-in / number-out, and the same
one measures a spectrum tengri just predicted, one exported to disk, and one the
user reduced themselves; `measure.from_prediction()` measures a `Prediction`
directly. Reading "model-free" as "consumes observed data" is what motivated a
proposed `from_observed` entry point during API Phase 3 — a split by input
provenance that nothing needed and that was correctly never built. Do not
reintroduce it. ("Model-free" also does not mean *reduction* tools: continuum
placement, sky subtraction and bad-pixel repair are a pipeline's job. See the
`tengri.measure` module docstring, which is the long form of this paragraph.)

**The verb can be carried by the module rather than the prefix.** The public
façade spells them `measure.spectral_index()`, `measure.line_flux()` and
`measure.photometry()` — no `measure_` prefix, because `tengri.measure` already
supplies it. The `measure_*`-prefixed engines they dispatch to live in
`observation/`. Note also that measuring an equivalent width is a *mode* of
`measure.spectral_index()` (break / EW / slope), not a function of its own; the
array-returning diagnostic is `compute_equivalent_widths()` under the `compute_*`
verb.

### 4b.2 The property-catalog contract

Every derived physical quantity (stellar mass, SFR, ionization parameter, etc.) is declared by exactly ONE component via a `Property(units, group, doc, fn)` declaration in the protocol. Violations are architecture-level bugs.

**Registry integrity:**
- Flat snake_case keys, Bagpipes-compatible where the concepts overlap (e.g., `stellar_mass`, `sfr_total`)
- Name collisions are a **BUILD-TIME error** caught at `SEDModel.build()`
- Accessing an absent property raises `KeyError` listing the available names (never silently returns None/NaN/0 — this is the `silent-failure` class)
- Metadata (units, group, doc, provenance, component name) is queryable via `tengri.list_properties()`, `describe_property()`

**Topology contract — same keys everywhere:**
- `Prediction` topology → scalar or band array
- `Posterior` topology → `(n_samples,)` or `(n_samples, n_bands)`
- `PopulationPosterior` topology → `(n_galaxies, n_samples,)` or `(n_galaxies, n_samples, n_bands)`

Only the **axes change**; the **key name is identical** across topologies.

### 4b.3 SED vs spectrum (frame & units)

Naming rule for panchromatic quantities:

- **SED** = panchromatic, model-grid array, **frame prefix required**: `rest_sed`, `obs_sed`. The frame prefix means "raw array"; every docstring must state frame and units.
- **spectrum** = instrument-grid, LSF-convolved, calibrated observable. No frame prefix (the instrument frame is implied). Called via `pred.spectrum()`.

**Every observable on `Prediction` is a uniform callable with a default.** No `_at` / `_for` / `_on` coinages, no bare properties that a user then has to resample by hand:

```python
pred.rest_sed()             # L_nu, rest-frame axis            [erg/s/Hz]
pred.rest_sed(wave)         # resampled onto YOUR rest-frame grid   [Angstrom]
pred.obs_sed()              # L_nu, observed-frame axis + IGM  [erg/s/Hz]  <- still L_nu!
pred.obs_sed(wave_obs)      # resampled onto YOUR observed-frame grid
pred.photometry(filters=None, fast=False)   # F_nu  [erg/s/cm2/Hz]
pred.magnitudes(filters=None, fast=False)   # AB mag
pred.spectrum(wave_obs=None)                # F_nu  [erg/s/cm2/Hz]
```

### 4b.3b Units: the distance is applied at PROJECTION, not on the SED

This is the single easiest thing to get wrong, and the docstring got it wrong for a long time.

| surface | quantity | units |
|---|---|---|
| `pred.rest_sed()` | L_ν, rest-frame axis | erg/s/Hz |
| `pred.obs_sed()` | **L_ν** — observed-frame *axis* + IGM | **erg/s/Hz** |
| `pred.photometry()` / `magnitudes()` / `spectrum()` | F_ν | erg/s/cm²/Hz (AB mag for `magnitudes`) |

**`obs_sed` is NOT a flux.** "Observed" names the *frame*, not a flux conversion. It does **not** apply `(1+z)/(4π d_L²)`; that factor lives in the projection layer (`observation/redshift_kernel.py`). Measured, not assumed: at z = 3, `obs_sed` differs from `rest_sed` at exactly the 172 grid points below rest-frame Lyman-α — IGM absorption, and nothing else.

Integrating `obs_sed()` as if it were a flux is wrong by ~57 orders of magnitude. If you want a flux, use `photometry()` or `spectrum()`.

**The wavelength argument is in the accessor's own frame.** `rest_sed(wave)` takes rest-frame Angstrom; `obs_sed(wave_obs)` takes **observed**-frame Angstrom. (The deprecated `model.predict_obs_sed(params, wave=...)` took a *rest*-frame grid and redshifted it — an observed-frame result with a rest-frame argument. That asymmetry was a footgun and is deliberately **not** reproduced.)

**The SED arrays do not carry their axis.** The axis is a separate property:

| array | its axis |
|---|---|
| `pred.rest_sed()` | `pred.wave_rest` |
| `pred.obs_sed()`  | `pred.wave_obs` |

Never reconstruct the observed axis by hand as `wave * (1 + params["redshift"])` — see §4b.6.

**Resampling is `jnp.interp` onto the requested grid**, bit-exact with the wavelength argument of the deprecated `predict_rest_sed`. Migrating a call site changes no number.

### 4b.3a Misuse must fail loudly, never silently

`pred.rest_sed` **without the parentheses** is a method object, not an array. Left alone, `np.asarray(bound_method)` yields a `dtype=object` array that plots and arithmetic will happily turn into garbage. It therefore raises `TypeError` with the fix spelled out (`_SEDCallable` in `forward/prediction.py`).

This is the general rule, not a one-off: **a public accessor that can be misused must raise, not fail open.** This package has shipped enough silent NaN-and-carry-on bugs (the `silent-failure` label) to have earned the paranoia.

### 4b.4 Exact vs fast

Analysis and post-fit prediction is **EXACT by default** using the full wave grid.

The fast path is a build-time optimization (e.g., `approx=WavePrecomp(...)`) that LUTs photometry and routes `predict_photometry` through precomputed sub-band weights. The fast path is **opted into explicitly** with a keyword argument at BUILD time. A speed knob must **never silently change the physics**.

**Contract:**
- Default `model.predict_photometry(params)` uses exact wave-grid integration
- Fast path `model_fast = SEDModel.build(..., approx=WavePrecomp(...))` precomputes filters and returns results via LUT
- Accuracy difference is bounded: approx-vs-exact tolerance is documented in the `approx` class docstring
- User can never accidentally call the fast path; it must be explicit at model construction

### 4b.5 The two prediction surfaces

There are exactly two, and nothing else is public:

| surface | what it is | when |
|---|---|---|
| `model.predict(params)` → `Prediction` | rich, cached; ONE forward pass, everything hangs off it | exploration, plotting, post-fit |
| `model.predict_photometry / predict_spectrum / predict_line_fluxes / predict_properties(params)` | lean shortcuts, JIT/vmap-safe | the inference hot path (what the likelihood calls) |

**`model.predict()` takes `params` and nothing else.** It has no `wave=` argument. (Resampling belongs to the accessor: `pred.rest_sed(wave)`.) Three separate agents have "migrated" a call site to `model.predict(p, wave=...)`; it raises `TypeError`, and `py_compile` does not catch it.

`predict_properties(params, names=...)` is the **single** JIT/vmap surface for derived quantities.

### 4b.6 Two names that are not what they look like

These have each caused real, shipped bugs. Read them before touching prediction code.

**1. `params.get("redshift", 0.0)` is forbidden.** A `Fixed` redshift is legitimately **absent** from the user's `params` dict (the spec holds it). Reading it back with a `0.0` default puts the galaxy at 10 pc and inflates the flux by ~1e17 — silently, with no NaN and no exception. This exact mistake shipped three times (#1097, #1124, #1127). **Always resolve through the spec**: `model._get_redshift(params)`, which lets an explicit value win, falls back to the fixed value, and *raises* if the model has neither.

Do **not** reach for `_get_dl_cm` as "the obvious helper": it short-circuits to a precomputed distance and silently discards an explicit override.

**2. `state.derived` is NOT `Posterior.derived`.**

| expression | what it is | status |
|---|---|---|
| `state.derived["sed_agn"]`, `state.derived["L_absorbed"]`, … | `ForwardState.derived` — the internal cross-component pipeline dict (ADR-0009) | **not deprecated.** Leave it alone. |
| `posterior.derived["stellar_mass"]` | the old `Posterior` accessor | **deprecated** → `posterior.properties[...]` |

An audit that greps for `.derived` and "migrates" every hit will rewrite the physics accessors in the reproduction notebooks. Grep for `posterior.derived` / `\.derived\[` on a `Posterior`, never bare `.derived`.

---

## 5. Inference Method Names

### Canonical method strings (13 total)

```
Approximate (fast):
  "map"                  — point estimate
  "laplace"              — Gaussian at MAP via Hessian
  "pathfinder"           — L-BFGS path (Zhang+2022)
  "vi"                   — geoVI via NIFTy optimize_kl (DEFAULT)
  "vi_linear"            — MGVI via NIFTy optimize_kl
  "vi_nifty_fast"        — geoVI fast path (~35% faster, no logging)
  "vi_nifty_fast_linear" — MGVI fast path (~35% faster, no logging)
  "vi_native"            — Native JAX geoVI (experimental, multi-seed)
  "vi_native_linear"     — Native JAX MGVI (experimental, multi-seed)

Exact (slow):
  "mcmc"             — auto-select (NUTS if D<=20, else Ray Tracing)
  "mcmc_raytrace"    — Ray Tracing HMC (Behroozi 2025)
  "mcmc_nuts"        — No-U-Turn Sampler
  "mcmc_ess"         — Elliptical Slice Sampling

Model selection:
  "nss"              — log Z via Nested Slice Sampling

Auto:
  "auto"             — D<=20: mcmc_nuts, else: vi
```

**Dimensionality D:** For the purpose of `"auto"` selection and `"mcmc"` auto-dispatch, D is the number of **free (sampled) parameters**, excluding deterministic transforms and derived quantities. For IFT/correlated-field models, D counts the latent excitation field DOF, not the pixel-space dimension.

### Deprecated aliases (emit DeprecationWarning, removed v1.0)

| Old name | Maps to | Note |
|----------|---------|------|
| `vi_nifty` | `vi` | |
| `vi_nifty_linear` | `vi_linear` | |
| `geovi`, `fast_geovi`, `nifty_geovi`, `geovi_nuts` | `vi` | |
| `mgvi`, `fast_mgvi`, `nifty_mgvi`, `evi` | `vi_linear` | |
| `native_geovi` | `vi_native` | |
| `native_mgvi`, `native_evi` | `vi_native_linear` | |
| `raytrace` | `mcmc_raytrace` | |
| `nuts` | `mcmc_nuts` | |
| `elliptical_slice` | `mcmc_ess` | |
| `evidence` | `nss` | |
| `cat3d_wind_analytic` | `cat3d_wind_sed` | "_analytic" was a misnomer — grid interpolation, not closed-form |
| `silva04_analytic` | `silva04_sed` | as above |
| `skirtor_analytic` | `skirtor_sed` | as above |
| `fritz_analytic` | `fritz_sed` | as above |

Internal-only (not public API; renamed without alias):
`skirtor_agnfitter_analytic` → `skirtor_agnfitter_sed`,
`slone_netzer_analytic` → `slone_netzer_sed`.

---

## 6. SFH Function Names

| Canonical (Python function name) | Registry key (short alias) | Description |
|----------------------------------|---------------------------|-------------|
| `truncated_skewnormal_sfh` | `tsnorm` | Truncated skew-normal |
| `skewnormal_sfh` | `snorm` | Skew-normal |
| `lognormal_sfh` | `lnorm` | Log-normal |
| `gaussian_sfh` | `norm` | Normal (Gaussian) |
| `double_powerlaw` | `dpl` | Double power-law |

**Clarification:** The canonical names are the actual Python function names. The short aliases are string keys in `SFH_REGISTRY`, retained for backward compatibility and config-file brevity. `resolve_sfh("tsnorm")` and `resolve_sfh("truncated_skewnormal_sfh")` both resolve to the same function. New code and documentation should use the canonical names; the registry resolves both.

---

## 7. Module Boundary Rules

```
core/       -> never imports inference/
inference/  -> never imports models/ directly (only through core/ public API)
models/     -> only imports utils/
utils/      -> only imports stdlib + third-party (jax, numpy, etc.)
io/         -> may import core/, models/, utils/; never imports inference/
plotting/   -> may import core/, utils/; never imports inference/ or models/
```

**What `core/` exposes to `inference/`:** The `core/` package provides the public interface to `models/` via `SEDModel`, `Parameters`, and `build_*` factories. `inference/` must access all physics functionality exclusively through these `core/` entry points. Direct `from models.dust import ...` in `inference/` is forbidden.

No exceptions. Enforce via import-linter or CI grep.

---

## 8. Exception Hierarchy

```
TengriError (base)
├── ParameterError      — invalid parameter names, values, or conflicts
├── ConfigError         — invalid Config construction or missing fields
├── BackendError        — backend initialization or computation failure
├── InferenceError      — sampler/optimizer failures (convergence, NaN, etc.)
└── IOError             — file I/O, missing data files, format mismatch
```

All Tengri exceptions inherit from `TengriError`. New code must raise domain-specific exceptions, not bare `ValueError` or `RuntimeError`. The base class lives in `core/exceptions.py`.

---

## 9. JAX & Array Conventions

### JAX internals

- `_tree_flatten` / `_tree_unflatten` methods on pytree-registered classes: no prefix beyond the leading underscore.
- Custom differentiation rules: use `@jax.custom_jvp` / `@jax.custom_vjp` decorators. Name the custom rule function `_{original_name}_jvp` or `_{original_name}_fwd` / `_{original_name}_bwd`.
- Donated-buffer arguments: suffix `_donate` on the wrapper if it sets `donate_argnums`.

### Shape annotations

All functions operating on arrays must document shapes in the docstring using the following dimension names:

| Symbol | Meaning |
|--------|---------|
| `N` | Number of galaxies / objects in batch |
| `W` | Number of wavelength bins |
| `T` | Number of time steps (SFH grid) |
| `B` | Number of photometric bands |
| `D` | Number of free parameters |
| `S` | Number of posterior samples |

Example docstring:
```python
def compute_sed(params, ssp_grid):
    """
    Parameters
    ----------
    params : array, shape (N, D)
    ssp_grid : array, shape (T, W)

    Returns
    -------
    sed : array, shape (N, W)
    """
```

Adopt `jaxtyping` annotations where practical for static checking.

---

## 10. Spelling Convention (American English)

**Rule:** All identifiers (classes, functions, parameters, kwargs, constants,
registry keys, module filenames) and all prose (docstrings, comments, docs,
notebooks, example scripts) use **American English** spelling. British spellings
must never appear in new code.

| British (forbidden) | American (canonical) |
|---------------------|----------------------|
| `colour`, `behaviour`, `favour`, `neighbour` | `color`, `behavior`, `favor`, `neighbor` |
| `normalise`/`normalisation`, `marginalise`, `optimise`, `initialise`, `finalise` | `normalize`/`normalization`, `marginalize`, `optimize`, `initialize`, `finalize` |
| `catalogue`/`catalogued`, `analogue` | `catalog`/`cataloged`, `analog` |
| `centre`, `metre`, `fibre` | `center`, `meter`, `fiber` |
| `modelling`/`modelled`, `labelled` | `modeling`/`modeled`, `labeled` |
| `grey`, `analyse` (verb) | `gray`, `analyze` |

**Exception — external data contracts.** Strings that index a third-party file
or upstream API keep that source's spelling, because the literal must match the
bytes on disk. The known cases are the Synthesizer grid HDF5 dataset keys
`ionisation_parameter` and `log10_specific_ionising_luminosity`
(`components/nebular/agn_nebular.py`). Document any such exception inline.

**Note on American-invariant words:** do not "correct" words that are already
American or invariant — `noise`, `raise`, `exercise`, `precise`, `surprise`,
`revise`, `analyses` (plural noun of *analysis*), the `-wise` suffix
(`piecewise`, `otherwise`), and the matplotlib `Greys`/`Greens` colormap tokens.

### Renames applied (#819, no deprecation alias)

These were **hard-renamed** (no alias) under the pre-v1.0 policy of §0 — old
British spellings are gone, not deprecated. Per §2's file-path stability rule the
`marginalised.py` module was renamed in the same change.

| Old | Canonical | Module |
|-----|-----------|--------|
| `cue_full_catalogue` / builder key `full_catalogue` | `cue_full_catalog` / `full_catalog` | `parameters/`, `forward/`, `components/nebular/` |
| `rest_frame_colour()` | `rest_frame_color()` | `analysis/diagnostics/spectral.py` |
| `CalibrationMarginalisedLikelihood` (+ `ELine`, `CloudyELine`, `CalibrationELine`) | `…MarginalizedLikelihood` | `inference/likelihoods/marginalized.py` |
| `normalised_excess_variance()` | `normalized_excess_variance()` | `components/agn/grahsp/variability.py` |
| `rank_normalise()` / `rank_normalised_rhat()` | `rank_normalize()` / `rank_normalized_rhat()` | `analysis/diagnostics/autocorrelation.py` |
| `finalise()` | `finalize()` | `inference/backends/nested/utils.py` |
| `SSP_CATALOGUE_URL` | `SSP_CATALOG_URL` | `data/__init__.py` |

See `CHANGELOG.md` for the full record.

### Enforcement

`tools/check_british_spelling.py` is a CI guard (in the `lint` job) that fails
the build on any British spelling in `src/`, `tests/`, hand-written docs,
`examples/`, and active notebooks. Run `python tools/check_british_spelling.py`
locally, or `--fix` to rewrite case-preserving. Data-contract exceptions live in
`ALLOWED_TOKENS`; the rename-ledger docs and this guard's own test fixtures are
in `EXCLUDE_FILES`.

---

## 11. Prose Notation

**Governing rule: Unicode in prose, ASCII in code.** Bare ASCII stays inside code spans and literal parameter names — `tau_diff` is the actual kwarg and must never be "corrected" to `τ_diff` when it appears as an identifier. This rule explicitly forbids blanket find-and-replace across all contexts.

### Canonical notation table

| Concept | Prose | Code / literal |
|---|---|---|
| Citation | `Calzetti+2000`; multiples `Calzetti+2000; Draine+2007` | full ref in `Reference:` line |
| micron | `μm` (U+03BC GREEK SMALL LETTER MU only) | `um` in identifiers |
| Angstrom | `Å` | `AA` |
| Solar mass | `M☉` | `Msun` |
| Solar metallicity | `Z☉` | `Zsun` |
| Optical depth | `τ_diff`, `τ_bc`, `τ_V` | `tau_diff`, `tau_bc`, `tau_V` |
| Redshift, exact | `z = 3` (spaced) | — |
| Redshift, approximate | `z ≈ 3` (spaced); never `z~3` | — |
| Balmer alpha | `Hα` | `H_alpha` |
| Gelman-Rubin | `R̂` | `r_hat` |
| Multiplier | `1.46×` | — |
| Signal-to-noise | `S/N` | `snr` |
| Numeric range | en-dash: `16–84%` | — |
| Chi-square | `χ²` | `chi2` |
| Library name | `tengri` lowercase; `Tengri` only sentence-initial | `tengri` |
| Reference codes | each project's own styling: `BAGPIPES`, `Prospector`, `CIGALE`, `AGNfitter`, `Synthesizer`, `FSPS`, `DSPS` | — |

### Enforcement

Enforcement targets confusable codepoints — U+00B5 MICRO SIGN (which is why U+03BC is canonical), ligatures, homoglyphs, and non-breaking spaces — with intentional typography (superscripts, subscripts, ellipsis) explicitly permitted. The CI guard `tools/check_docs_voice.py` enforces this contract.

### Motivation

Measured across the 325-file user-facing corpus:
- 472 citations split three ways (inconsistent separators and spacing)
- 167 redshift forms split six ways (spacing, `~` vs `≈`, bare values)
- 126 micron uses split across two visually identical codepoints: `µm` (U+00B5 MICRO SIGN, 53 uses) vs `μm` (U+03BC GREEK SMALL LETTER MU, 56 uses) plus bare `um` and full `micron`
