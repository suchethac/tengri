# Tengri Naming Contract

> **Authority:** This is the single source of truth for naming conventions.
> All new code, renames, and refactors MUST comply. Referenced by CLAUDE.md.
>
> **Last updated:** 2026-04-05
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
| Star formation | `sfh_{type}_` | `sfh_dpl_alpha`, `sfh_tsnorm_log_peak_sfr` |
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
- **Internal names shadowing public names** (e.g., `log_peak_sfr` meaning different things at different layers).

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
| `compute_*` | Pure function, returns new array | `compute_blr_sed()`, `compute_dust_attenuation()` |
| `load_*` | Reads from disk | `load_ssp_data()`, `load_cloudy_grid()` |
| `build_*` | Factory returning a callable/kernel | `build_fused_photometry()` |
| `apply_*` | Transform a spectrum in-pipeline | `apply_lsf()`, `apply_igm()` |
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