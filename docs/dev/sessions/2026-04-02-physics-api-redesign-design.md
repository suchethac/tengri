# Physics Model API Redesign Plan

**Date:** 2026-04-02  
**Companion:** `api_redesign.md` (high-level inference API)  
**Principle:** Additive where possible. Fix breaking inconsistencies only where the current behavior is wrong.

---

## Core design philosophy: The model IS the physics hierarchy

The single most important design rule is: **the API should mirror the physical hierarchy**. An astronomer reading a tengri model definition should immediately see what physics is present at each layer, which sub-models are active, which parameters are free, and which inference method is appropriate.

Currently, the model is assembled from flat keyword arguments scattered across `ParamSpec`, `Model`, and `Observation`. The physical layers — SFH, SPS, dust, nebular, AGN, observation — are implicit. Users have to know the full parameter name (`sfh_tsnorm_log_peak_sfr`, `dust_tau_bc`, `neb_logU`) and mentally reconstruct the hierarchy from the prefix.

The redesign makes the hierarchy **explicit and introspectable**.

### What the ideal API looks like

```python
# The model as a physics tree — each layer is a named sub-model
model = tengri.Model.from_config(
    ssp="data/ssp.h5",

    # Layer 1: Star Formation History
    sfh="dpl+field",          # sub-model: double power law + stochastic GP field
    priors={
        "alpha":        Uniform(0.5, 5),
        "beta":         Uniform(0.5, 5),
        "log_peak_sfr": Uniform(-1, 2.5),
        "psd_sigma":    Uniform(0.01, 1.5),
        "psd_tau_myr":  Uniform(10, 500),
    },

    # Layer 2: SPS — always DSPS, SSP grid is the choice
    # (already specified above as ssp=)

    # Layer 3: Dust
    dust={
        "attenuation": "charlot_fall",     # which curve family
        "law_bc":      "power_law",        # birth cloud law
        "law_diff":    "calzetti",         # diffuse ISM law
        "emission":    "dl07",             # dust emission model
        "priors": {
            "tau_bc":   Uniform(0, 2),
            "tau_diff": Uniform(0, 1.5),
            "umin":     Uniform(0.1, 25),  # DL07 minimum radiation field
            "qpah":     Fixed(3.5),
        }
    },

    # Layer 4: Nebular
    nebular={
        "backend": "cue",                   # BakedIn | CloudyGrid | Cue
        "priors": {
            "logU":  Uniform(-4, -1),
            "logNO": Uniform(-0.5, 0.5),    # Cue only
        }
    },

    # Layer 5: AGN [optional]
    agn={
        "disc":  "kubota_done",
        "torus": "skirtor",
        "nlr":   "cue",                     # disc EUV → Cue NLR chain
        "blr":   True,
        "priors": {
            "log_mbh":  Uniform(6, 10),
            "log_ledd": Uniform(-2, 0),
            "cos_inc":  Uniform(0.05, 0.99),
        }
    },

    # Layer 6: Observation
    redshift=0.1,
    filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
)
```

The model now reads like a physics description. Every layer is present. Sub-models are named. Parameters are grouped by the layer they belong to.

### Introspection: `model.describe()` and `model.tree()`

After construction, the model should be introspectable:

```
>>> model.tree()

Model  [D=152, stochastic=True]
│
├── SFH: dpl + field (GP)
│   ├── dpl_alpha          ~ Uniform(0.5, 5.0)
│   ├── dpl_beta           ~ Uniform(0.5, 5.0)
│   ├── dpl_log_peak_sfr   ~ Uniform(-1.0, 2.5)
│   ├── field_psd_sigma    ~ Uniform(0.01, 1.5)
│   ├── field_psd_tau_myr  ~ Uniform(10, 500)
│   └── field_xi           [128-dim GP latent, xi ~ N(0,I)]
│
├── SPS: DSPS  [BC03 + Chabrier IMF, 67 age bins, 6 metallicities]
│
├── Dust
│   ├── Attenuation: Charlot & Fall (bc=power_law, diff=calzetti)
│   │   ├── tau_bc         ~ Uniform(0, 2)
│   │   └── tau_diff       ~ Uniform(0, 1.5)
│   └── Emission: Draine & Li 2007 (tabulated)
│       ├── umin           ~ Uniform(0.1, 25)
│       └── qpah           = Fixed(3.5)
│
├── Nebular: Cue  [12 free params, Li+2025]
│   ├── neb_logU           ~ Uniform(-4, -1)
│   └── neb_logNO          ~ Uniform(-0.5, 0.5)
│       [ionspec params computed from SSP; gas params at defaults]
│
├── AGN: Kubota & Done disc + SKIRTOR torus + Cue NLR + BLR
│   ├── agn_log_mbh        ~ Uniform(6, 10)
│   ├── agn_log_ledd       ~ Uniform(-2, 0)
│   └── agn_cos_inc        ~ Uniform(0.05, 0.99)
│
└── Observation: Photometry [5 SDSS bands] at z=0.1 [fixed]
    Precomputed: YES (21.6× speedup)

Recommended inference:
  → model.fit(data, noise, method="vi")   [D=152, stochastic, default]
  → model.fit(data, noise, method="mcmc_raytrace")  [exact MCMC, validation]
```

The **recommended inference** at the bottom is the key addition: the model knows its own dimensionality and whether it's stochastic, so it can suggest the right method. This is the connection between physical model complexity and inference choice that the user is asking for.

### Inference method selection tied to model structure

The inference method recommendation is not arbitrary — it follows from the physics:

| Model configuration | D | Recommended method | Why |
|---|---|---|---|
| Smooth parametric (tsnorm, dpl) | 5–10 | `"vi"` or `"laplace"` | Low-D, fast convergence |
| Smooth + AGN (no field) | 15–25 | `"vi"` | Moderate-D, differentiable |
| Stochastic field (field only) | ~137 | `"vi"` (native_geovi) | High-D, GP structure helps geoVI |
| Stochastic + AGN + Cue | ~150 | `"vi"` | Same, more parameters |
| Hierarchical (N galaxies × D) | N×D | `"vi_nifty"` (CFM) | CorrelatedFieldMaker handles shared PSD |
| Any model, validation | any | `"mcmc_raytrace"` | Gradient-noise tolerant, exact |
| Bayesian evidence | D ≲ 30 | `"evidence"` (NSS) | Only method that gives log Z |

`model.recommend_method()` returns this string with a one-line explanation. `model.fit(data, noise)` uses it automatically.

---

## Summary of findings

After reading all physics model source files, five categories of issues stand out:

1. **Physical hierarchy is correct but badly surfaced** — the right chain exists (SFH → SPS → dust → nebular → observation), but the abstraction levels in AGN and nebular models leak implementation details into the public API.
2. **Parameter naming is inconsistent between backends** — `neb_logU` (CloudyGrid) vs `gas_logu` (Cue) is the same physical quantity. Users setting this in `ParamSpec` shouldn't need to know which backend uses which name.
3. **`agn_nlr_emission()` has a union return type** — returns either an array on a wavelength grid OR a `(wavelengths, luminosities)` tuple depending on which backend is called. This is the most serious composability bug.
4. **Emission line measurement API is new and under-documented** — `eline_priors.py` and `eline_marginalization.py` together form a powerful analytical line-marginalization pipeline, but the connection between them and the rest of the model is not clear.
5. **`dust/emission.py` is a 2800-line monolith** — mixing physical model implementations, template loaders, registration machinery, and lazy backward-compat shims.

---

## Part 1: Physical Hierarchy — What Should the Layers Be

The correct physical hierarchy for the full panchromatic model:

```
Model(ParamSpec, SSPData, observation)
│
├── SFH registry → SFR(t), Z(t)
│     Models: dpl, tsnorm, snorm, const, field (GP), burst, ...
│
├── SPS (DSPS) → Stellar SED L_star(λ)
│     Input: SFR(t), Z(t), SSP grid
│
├── Dust attenuation → L_attenuated(λ)                [tengri.components.dust.attenuation]
│     ├── Attenuation curves: calzetti, power_law, kriek_conroy, smc, ...
│     └── Two-component (Charlot & Fall): birth cloud + diffuse ISM
│
├── Nebular emission → L_nebular(λ)                   [tengri.components.nebular]
│     ├── Q_H from stellar SED (ionizing photons below 912 Å)
│     └── Backend:
│           BakedIn  → pre-baked in SSP templates (zero free params)
│           CloudyGrid → grid interpolation (3 params: logU, Z_gas, fesc)
│           Cue      → neural emulator (12 params: ionspec × 7 + gas × 5)
│
├── Dust emission → L_dust(λ)                         [tengri.components.dust.emission]
│     Energy balance: L_absorbed = ∫(L_star - L_attenuated) dν → L_IR
│     Models: modified_blackbody, dl07, dl14, dale2014, astrodust, bosa, themis, ...
│
├── AGN → L_AGN(λ)  [optional]                        [tengri.components.agn]
│     ├── Disc: powerlaw_disc | multicolor_disc | kubota_done_disc | adaf_disc
│     ├── Torus: simple_torus | two_temperature_torus | skirtor_analytic
│     ├── NLR (isotropic, always visible):
│     │     Option A: analytic Gaussian profiles (nlr.py) — zero extra params
│     │     Option B: Cue chain — disc EUV → ionspec → NLR luminosities
│     ├── BLR (blocked by torus when edge-on):
│     │     Analytic Gaussian profiles with geometric sigmoid masking
│     ├── Fe II pseudo-continuum [optional]
│     └── Polar dust [optional]
│
├── IGM absorption → T_IGM(λ, z)                      [tengri.components.igm]
│
├── Radio / X-ray [optional]                          [tengri.components.radio, .xray]
│
└── Observation
      ├── Photometry: filter convolution → f_ν [n_filters]
      ├── Spectroscopy: pixel-level prediction → f_λ [n_pix]
      │     + LSF convolution
      │     + Chebyshev calibration polynomial
      └── Emission line marginalization [optional]    [tengri.observation.eline_*]
            ├── Design matrix: Gaussian line profiles → A [n_pix × n_lines]
            ├── Prior: CLOUDY-based ratios relative to Hβ
            └── Analytical marginalization: integrates out line amplitudes
```

**What's currently correct:** The overall chain is implemented. The registry pattern for SFH, dust, and AGN models is clean.

**What needs fixing:** AGN NLR return types, parameter name unification, and the emission line measurement pipeline exposure.

---

## Part 2: AGN Model API Issues and Fixes

### Issue 2.1 — Duplicated `_planck_lnu()` in disc.py and torus.py

Both modules define an identical internal helper. Should be moved to a shared `tengri/models/agn/_phys.py` or `tengri/models/_constants.py`.

**Fix:** Extract to `src/tengri/models/agn/_phys.py`:
```python
# _phys.py
def planck_lnu(nu: jnp.ndarray, temperature: float) -> jnp.ndarray:
    """Planck function L_nu [Lsun/Hz] per unit area."""
    ...
```
Import in both `disc.py` and `torus.py`. No change to public API.

---

### Issue 2.2 — Hardcoded line widths and efficiencies in nlr.py and blr.py

`_NLR_FWHM_KMS = 500.0` and `_NLR_LINE_EFFICIENCY = 0.10` are physics parameters hidden as module constants. Users running high-spectral-resolution fits or fitting AGN with unusual line widths cannot adjust these without monkey-patching.

**Fix:** Expose as optional function parameters with the current values as defaults:
```python
# nlr.py
def nlr_emission(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    covering_fraction: float = 0.10,
    fwhm_kms: float = 500.0,           # NEW — was hardcoded
    line_efficiency: float = 0.10,     # NEW — was hardcoded
    ...
) -> jnp.ndarray:

# blr.py — same pattern
def blr_emission(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    covering_fraction: float = 0.08,
    fwhm_kms: float = 5000.0,          # NEW — was hardcoded
    fe2_strength: float = 0.0,
    ...
) -> jnp.ndarray:
```

Default values unchanged → backward compatible.

---

### Issue 2.3 — `agn_nlr_emission()` returns a union type [CRITICAL]

**Current broken behavior:**
```python
# Backend "cue": returns tuple (line_wavelengths, line_luminosities) in Lsun
# Backend "feltre": returns jnp.ndarray on input wavelength grid in Lsun/Hz
# Backend "analytic": returns jnp.ndarray on input wavelength grid
```

This union type means every calling site must check `isinstance(result, tuple)`. The `sed_pipeline.py` currently handles this with special-case logic — a code smell that propagates forever.

**Fix:** Standardize to always return `(line_wavelengths: jnp.ndarray, line_luminosities: jnp.ndarray)`. The caller is responsible for convolving to the observation wavelength grid:

```python
# New contract for ALL AGN emission line functions:
def nlr_emission(
    wavelength: jnp.ndarray,  # observation grid (used for BROADENING only, not output)
    agn_log_lbol: float,
    ...
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (line_wavelengths_aa, line_luminosities_lsun)."""

def blr_emission(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    ...
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (line_wavelengths_aa, line_luminosities_lsun)."""

def agn_nlr_emission(
    wavelength: jnp.ndarray,
    ...
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Always returns (line_wavelengths_aa, line_luminosities_lsun).
    The caller convolves to the SED wavelength grid."""
```

Add a shared utility:
```python
# src/tengri/models/agn/_phys.py
def lines_to_sed(
    line_wavelengths: jnp.ndarray,
    line_luminosities: jnp.ndarray,
    wave_obs: jnp.ndarray,
    fwhm_kms: float,
) -> jnp.ndarray:
    """Convolve line list to SED wavelength grid via Gaussian profiles."""
```

The `sed_pipeline.py` always calls `lines_to_sed()` after getting lines — removes the isinstance check.

---

### Issue 2.4 — `agn_nlr_cue()` takes a `wavelength` parameter it ignores

The function signature accepts `wavelength` but the Cue emulator returns its own wavelength array — the input wavelength grid is not used. This is confusing and misleads callers.

**Fix:** Remove `wavelength` from `agn_nlr_cue()`. The function signature becomes:
```python
def agn_nlr_cue(
    cue_backend,
    l_acc_erg: float,
    covering_fraction: float = 0.1,
    alpha_pl: float = -1.7,
    gas_logu: float = -3.0,
    gas_logn: float = 3.0,
    gas_logz: float = 0.0,
    gas_logno: float = 0.0,
    gas_logco: float = 0.0,
    ionspec_params: dict | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (line_wavelengths_aa, line_luminosities_lsun)."""
```

---

### Issue 2.5 — No top-level AGN config object

A galaxy with a full AGN has 12–15 parameters: disc params, torus params, BLR/NLR covering fractions, inclination, polar dust. These are currently passed as a flat list of `agn_*` kwargs through multiple layers.

**Proposed:** A frozen dataclass `AGNConfig` for advanced users who want to assemble AGN models manually (distinct from the `ParamSpec` free-parameter system):
```python
@dataclass(frozen=True)
class AGNConfig:
    disc: str = "multicolor"           # "powerlaw" | "multicolor" | "kubota_done" | "adaf"
    torus: str = "two_temperature"     # "simple" | "two_temperature" | "skirtor"
    nlr: str = "analytic"              # "analytic" | "cue"
    blr: bool = True                   # include BLR (masked by inclination)
    polar_dust: bool = False
    fe2: bool = False                  # Fe II pseudo-continuum
```

This is passed to `Model(spec, ssp, agn_config=AGNConfig(...))` as a static configuration (not a free parameter). The free parameters (`agn_log_mbh`, `agn_log_ledd`, etc.) still come from `ParamSpec`.

---

## Part 3: Nebular Backend Naming Unification [HIGH PRIORITY]

### Issue 3.1 — Same ionization parameter has different names in different backends

| Backend | Parameter name | What it is |
|---------|---------------|------------|
| `CloudyGridBackend` | `neb_logU` | log₁₀(ionization parameter) |
| `CueBackend` (via `**neb_params`) | `gas_logu` | same quantity |
| User-facing `ParamSpec` | `neb_logU` | same quantity |
| `ionizing_spectrum.py` | `log_u` | same quantity |

**Fix:** Standardize to `neb_logU` everywhere. The Cue internal parameters (`gas_logu`, `gas_logn`, `gas_logz`, `gas_logno`, `gas_logco`) are fine as internal names, but the backend's public `predict_nebular_sed()` interface should use the standard `neb_*` namespace:

```python
class CueBackend:
    def predict_nebular_sed(
        self,
        ...,
        neb_logU: float = -3.0,        # was gas_logu — standardized
        neb_logZ_gas: float | None = None,  # was gas_logz — standardized
        neb_log_nH: float = 2.0,       # was gas_logn — standardized
        neb_logNO: float = 0.0,        # was gas_logno — standardized
        neb_logCO: float = 0.0,        # was gas_logco — standardized
        ...
    ) -> jnp.ndarray:
```

Internal translation to `gas_logu` etc. happens inside the method, not in the caller. This means `ParamSpec` names map cleanly to backend method names.

---

### Issue 3.2 — Line width units inconsistency

| Location | Parameter name | Units |
|----------|---------------|-------|
| `nebular/` modules | `line_sigma_aa` | Ångström |
| `observation/eline_marginalization.py` | `eline_sigma_kms` | km/s |

These are the same physical quantity (emission line width) in different units. Both are valid physically but the inconsistency forces users to convert.

**Fix:** Standardize on `line_sigma_kms` (km/s) throughout, since km/s is instrument-native (resolving power R = c / Δv). Add unit conversion inside `nebular/` modules where needed.

---

## Part 4: Emission Line Measurement API — The New Pipeline

`eline_priors.py` and `eline_marginalization.py` together form tengri's emission line marginalization pipeline. This is explicitly based on Johnson et al. (2021) / Prospector's approach and is new relative to most SED codes. The API needs to be clearly documented and exposed.

### What the pipeline does (physical description)

When fitting a galaxy spectrum that contains emission lines (Hα, [OIII], etc.), you face a choice:
1. **Model the lines**: add line flux as free parameters → 10+ extra dimensions per galaxy
2. **Mask the lines**: discard information  
3. **Marginalize analytically**: treat line amplitudes as nuisance parameters and integrate them out of the likelihood in closed form → retains all spectral information, adds zero free parameters

Tengri implements option 3. The pipeline:

1. **Build a design matrix** `A` (shape `n_pixels × n_lines`): each column is a Gaussian line profile at the observed wavelength of one emission line.
2. **Set priors on line amplitudes**: either flat (`marginalize_emission_lines`) or physically informed from CLOUDY photoionization grids (`marginalize_emission_lines_cloudy`).
3. **Analytically marginalize**: given the residual spectrum `r = data - continuum_model`, solve for the posterior over line amplitudes and integrate it out. Returns a scalar log-likelihood.

### Current API (what exists)

```python
# Step 1: Build design matrix
A = build_eline_design_matrix(
    wave_obs,                    # observed wavelength grid
    line_wavelengths,            # rest-frame line wavelengths (Å)
    spectral_resolution,         # R = λ/Δλ of the spectrograph
    redshift,                    # to shift lines to observed frame
    eline_sigma_kms=0.0,         # intrinsic line width (km/s, 0 = instrument-limited)
    eline_delta_v_kms=0.0,       # systematic velocity offset
)  # → jnp.ndarray shape (n_pix, n_lines)

# Step 2a: Flat priors (no physics)
ln_L = marginalize_emission_lines(
    residual, noise, A,
    prior_mean=jnp.zeros(n_lines),
    prior_sigma=jnp.ones(n_lines) * 1e10,  # effectively flat
)

# Step 2b: CLOUDY-informed priors
prior_means, prior_sigmas = cloudy_line_priors(
    log_z=met_logzsol,
    neb_logU=-3.0,
    line_wavelengths=DEFAULT_LINE_WAVELENGTHS,
    prior_width_dex=0.3,
)
ln_L = marginalize_emission_lines_cloudy(
    residual, noise, A,
    log_z=met_logzsol, neb_logU=-3.0,
    l_hbeta=hbeta_luminosity,      # anchor: scales all ratios to physical units
    prior_width_dex=0.3,
)
```

### Issues with the current API

**Issue 4.1 — `l_hbeta` is opaque.**
`marginalize_emission_lines_cloudy()` requires `l_hbeta`, the predicted Hβ luminosity from the continuum model. This is needed to convert CLOUDY's ratio-relative-to-Hβ priors into physical units. But nowhere in the documentation is it explained that the user needs to compute this from the current model parameters. A new user will have no idea where this number comes from.

**Fix:** The `l_hbeta` parameter should either be:
- Computed internally: `marginalize_emission_lines_cloudy(..., model=model, params=params)` — computes `l_hbeta` from the model
- Or documented clearly with an example showing: `l_hbeta = model.predict_line_luminosity(params, line="Hbeta")`

Add `Model.predict_hbeta(params)` as a convenience method.

**Issue 4.2 — Two separate functions for narrow and broad lines.**
`build_eline_design_matrix()` is for narrow lines (HII regions). `build_broad_design_matrix()` is for broad lines (AGN BLR). When fitting an AGN host galaxy you need both simultaneously. There should be a unified builder:

```python
# Proposed unified design matrix builder
def build_line_design_matrix(
    wave_obs: jnp.ndarray,
    narrow_wavelengths: jnp.ndarray,
    broad_wavelengths: jnp.ndarray | None = None,   # None = no BLR
    spectral_resolution: float = 2000.0,
    redshift: float = 0.0,
    narrow_sigma_kms: float = 0.0,
    broad_sigma_kms: float = 5000.0,
    delta_v_kms: float = 0.0,
) -> jnp.ndarray:
    """Returns (n_pix, n_narrow + n_broad) design matrix.
    Broad lines appear as additional columns after narrow lines."""
```

**Issue 4.3 — Line wavelength catalog is scattered.**
`DEFAULT_LINE_NAMES` and `DEFAULT_LINE_WAVELENGTHS` (13 lines) live in `eline_marginalization.py`. `CLOUDY_LINE_NAMES` and `CLOUDY_LINE_WAVELENGTHS` (11 lines) live in `eline_priors.py`. These two catalogs partially overlap. There should be a single `EMISSION_LINE_CATALOG` dict in a shared location.

**Proposed:**
```python
# src/tengri/models/observation/eline_catalog.py  [NEW FILE]

EMISSION_LINES = {
    # name: (rest_wavelength_aa, type, default_prior_width_dex)
    "Lya":      (1215.67, "recombination", 0.3),
    "OII3726":  (3726.03, "forbidden",     0.3),
    "OII3729":  (3728.82, "forbidden",     0.3),
    "Hbeta":    (4861.33, "recombination", 0.15),   # anchor — tighter prior
    "OIII4959": (4958.92, "forbidden",     0.15),   # ratio to OIII5007 fixed
    "OIII5007": (5006.84, "forbidden",     0.15),
    "OI6300":   (6300.30, "forbidden",     0.4),
    "NII6548":  (6548.05, "forbidden",     0.3),
    "Halpha":   (6562.80, "recombination", 0.15),   # anchor — tight
    "NII6583":  (6583.45, "forbidden",     0.3),
    "SII6716":  (6716.44, "forbidden",     0.35),
    "SII6731":  (6730.82, "forbidden",     0.35),
}

# Named line groups for convenience
LINE_GROUPS = {
    "optical_narrow":  ["OII3726", "OII3729", "Hbeta", "OIII4959", "OIII5007",
                        "OI6300", "NII6548", "Halpha", "NII6583", "SII6716", "SII6731"],
    "bpt":             ["Hbeta", "OIII5007", "Halpha", "NII6583"],
    "balmer":          ["Lya", "Hbeta", "Halpha"],
    "blr_broad":       ["Lya", "CIV1549", "CIII1909", "MgII2798", "Hbeta", "Halpha"],
}
```

This catalog is then imported by both `eline_marginalization.py` and `eline_priors.py`, eliminating the duplicate/inconsistent line lists.

---

## Part 5: Dust API Cleanup

### Issue 5.1 — `apply_dust_emission(**kwargs)` has no type contract

The function signature accepts `**kwargs` which are passed to whichever dust emission model was registered. DL07, Dale, Astrodust, BOSA, and THEMIS all take different parameter sets. There's no way to know at call time what parameters are valid.

**Fix:** Add a `TypedDict` per model, or a validation step:
```python
# Each emission model's parameters documented with defaults
DL07_PARAMS = {"dust_umin": 1.0, "dust_gamma_dl": 0.01, "dust_qpah": 3.5}
DALE_PARAMS = {"dust_alpha_dale": 2.0}
MAGPHYS_PARAMS = {"dust_T_warm": 45.0, "dust_T_cold": 18.0, ...}
```

Store these in a `_EMISSION_MODEL_PARAMS` dict in the registry, so `model.summary()` can print "dust_umin=1.0 (default)" for the active model.

### Issue 5.2 — `dust/emission.py` is 2800 lines (split it)

Proposed split:
```
src/tengri/models/dust/
├── attenuation.py          (keep, but split further — see below)
├── emission/
│   ├── __init__.py         (re-exports: draine_li2007, dale2014, etc.)
│   ├── _registry.py        (register_emission_model, get_emission_model)
│   ├── analytic.py         (modified_blackbody, casey2012, magphys)
│   ├── draine_li.py        (dl07, dl14 — template + analytic)
│   ├── dale.py             (dale2014)
│   ├── astrodust.py        (astrodust + PAH)
│   ├── bosa.py             (bosa templates)
│   └── themis.py           (themis templates)
├── priors.py               (keep)
```

### Issue 5.3 — `dust/attenuation.py` mixes abstraction levels

The file contains: (a) 14 individual attenuation law functions, (b) two-component Charlot & Fall logic, (c) Witt & Gordon geometry functions, (d) pre-computation helpers. These are all at different abstraction levels.

Proposed split (non-breaking, file kept as re-export shim):
```
src/tengri/models/dust/
├── attenuation/
│   ├── __init__.py         (re-exports everything — backward compat)
│   ├── _registry.py        (register_dust_law, get_dust_law, DUST_LAWS)
│   ├── laws.py             (all 14 individual curves: calzetti, power_law, etc.)
│   ├── two_component.py    (two_component_dust, precompute_dust_age_weights)
│   └── geometry.py         (wg00_shell, wg00_cloudy, wg00_dusty)
```

---

## Part 6: Physical Hierarchy — Registry Alignment

All three domains (SFH, dust, AGN) use a registry pattern (`register_sfh_model`, `register_dust_law`, `register_agn_model`, `register_emission_model`). The pattern is good but the APIs are not consistent across domains.

### Current registry APIs (inconsistent):

```python
# SFH registry
resolve_sfh(sfh_model: str, **params) -> Callable

# Dust attenuation registry
get_dust_law(name: str) -> Callable      # returns fn(wavelength, **kwargs) -> k(λ)

# Dust emission registry
get_emission_model(name: str) -> Callable  # returns fn(wavelength, **kwargs) -> L_nu

# AGN registry
get_agn_model(name: str) -> Callable    # returns fn(wavelength, agn_log_lbol, **kwargs) -> L_nu
```

### Fix: Align all registries to the same pattern

```python
# Unified pattern: registry.get(name) → fn with consistent signature contract
# Physical law functions: fn(wavelength, ...) → output [units documented]
# The registry stores: name → (fn, default_params, description)

class ModelRegistry:
    def register(self, name: str, fn: Callable, defaults: dict, description: str)
    def get(self, name: str) -> Callable
    def list(self) -> list[str]
    def describe(self, name: str) -> str     # human-readable description
    def defaults(self, name: str) -> dict    # default parameter values

# Usage:
tengri.sfh_registry.list()             # all registered SFH models
tengri.dust_registry.list()            # all dust laws
tengri.emission_registry.list()        # all dust emission models
tengri.agn_registry.list()             # all AGN models
```

This enables `model.summary()` to dynamically show "active dust law: calzetti (Calzetti+2000), params: none" instead of hardcoded string.

---

## Part 7: ParamSpec ↔ Physics Model Alignment

The biggest user-facing inconsistency: `ParamSpec` uses `neb_logU` but `CueBackend.predict_nebular_sed()` accepts `gas_logu`. The translation happens silently inside the model. This means:
- Errors in parameter names give cryptic messages like "unexpected keyword argument 'neb_logU'"
- It's impossible to test backends in isolation without knowing the internal names

**Fix:** Add a `ParamSpec.validate_against_backends(model)` method that checks all free parameters can be routed to their target functions without name mismatches. This is a development/debugging tool that catches integration errors early.

Also: expose the translation layer in `param_translate.py` as a documented `PARAM_MAP` dict so both the source code and the documentation point to the same truth:

```python
# param_translate.py — already has _EVOLVING_ALPHA_PARAM_MAP etc.
# Add documented public maps:

NEB_PARAM_MAP = {
    # ParamSpec name → backend kwarg name
    "neb_logU":    "neb_logU",     # uniform for all backends after fix
    "neb_logZ_gas": "neb_logZ_gas",
    "neb_log_nH":  "neb_log_nH",
    "neb_logNO":   "neb_logNO",
    "neb_logCO":   "neb_logCO",
    "neb_fesc":    "neb_fesc",
    "neb_fesc_lya": "neb_fesc_lya",
}
```

---

## Implementation order

| Priority | Change | Files | Why |
|----------|--------|-------|-----|
| 1 | Fix union return type in `agn_nlr_emission()` | `agn/nlr.py`, `agn/blr.py`, `agn/agn_nebular.py`, `core/sed_pipeline.py` | Breaks composability; removes isinstance checks throughout |
| 2 | Unify nebular backend parameter names to `neb_*` | `nebular/cue.py`, `nebular/cloudy_grid.py`, `nebular/dig.py` | Eliminates silent name translation failures |
| 3 | Create `EMISSION_LINE_CATALOG` in `eline_catalog.py` | New file; update `eline_marginalization.py` and `eline_priors.py` | Eliminates duplicate line lists |
| 4 | Add `Model.predict_hbeta()` and document `l_hbeta` | `core/model.py` | Makes emission line marginalization usable without reading source |
| 5 | Unified `build_line_design_matrix()` (narrow + broad) | `observation/eline_marginalization.py` | Needed for AGN host fitting |
| 6 | Expose `fwhm_kms` and `line_efficiency` in nlr/blr | `agn/nlr.py`, `agn/blr.py` | Enables custom line width fitting |
| 7 | Standardize `line_sigma_kms` units everywhere | `nebular/*.py`, `observation/*.py` | Removes user unit conversion |
| 8 | Unified `ModelRegistry` class | New `utils/registry.py` | `tengri.dust_registry.list()` etc. |
| 9 | Split `dust/emission.py` into subdirectory | Multiple files | Maintainability; currently 2800 lines |
| 10 | Extract `_planck_lnu` from disc.py and torus.py | `agn/_phys.py` | DRY |

---

## Documentation and Paper changes from physics API

### Docs to update

| File | Change |
|------|--------|
| `docs/the_model/index.md` | Add physical hierarchy diagram (the tree above); explain the SFH→SPS→dust→nebular→AGN→observation chain explicitly |
| `docs/advanced/extending.md` | Update to show `ModelRegistry` API for registering new dust laws and SFH models |
| `notebooks/specialist/05_emission_line_marginalization.py` | Rewrite to use `EMISSION_LINE_CATALOG`, the new unified `build_line_design_matrix()`, and add documentation of `l_hbeta`. Resync to `.ipynb` after. |
| `notebooks/models/06_nebular.py` | Update all `gas_logu` → `neb_logU` after the naming fix. Resync to `.ipynb`. |
| `notebooks/models/04_agn.py` | Fix `agn_nlr_emission()` return type handling after the union fix; update any `isinstance` checks. Resync to `.ipynb`. |
| `CLAUDE.md` | Update "Gotchas" section: document the `neb_logU` standardization; note that `agn_nlr_emission()` now always returns a tuple. |

### Paper changes from physics API (paper-writing agent instructions)

**IMPORTANT:** Read the final implemented API changes in the source before revising the paper. Do not revise based on this plan alone.

**Paper I** (*(private paper draft)*):
- `3-forward-model.tex` — the forward model section. After the registry unification, update any mentions of how models are selected (e.g., "the user specifies a dust law by name" → show `tengri.dust_registry.list()` or `Model.from_config(dust="calzetti")`).
- `3-forward-model.tex` — the emission line marginalization subsection (if present): update to show the `EMISSION_LINE_CATALOG` and the unified `build_line_design_matrix()`. Add a note on `l_hbeta` and how it connects the continuum model to the line prior.
- `4-inference.tex` — if there is any mention of emission line parameters as free parameters, clarify that they are analytically marginalized (zero inference cost) rather than sampled.
- `6-usage.tex` — the code examples for AGN fitting should show the `AGNConfig` object rather than scattered `agn_*` kwargs.

**Paper II** (*(private paper draft)*):
- No physics API changes affect Paper II directly. However, if Paper II contains any spectral fitting examples that use `fitter.run("native_geovi")` or explicit emission line handling, update those for the new API.

---

## Summary checklist for implementing agent

**AGN model fixes:**
- [ ] `_planck_lnu()` extracted to `agn/_phys.py`, imported by disc.py and torus.py
- [ ] `nlr_emission()` and `blr_emission()` expose `fwhm_kms` and `line_efficiency` as kwargs
- [ ] `agn_nlr_emission()` always returns `tuple[jnp.ndarray, jnp.ndarray]`
- [ ] `agn_nlr_cue()` removes unused `wavelength` parameter
- [ ] `lines_to_sed()` utility added to `agn/_phys.py`
- [ ] `sed_pipeline.py` updated to use `lines_to_sed()` (remove isinstance checks)
- [ ] `AGNConfig` dataclass added for static AGN model selection

**Nebular naming fixes:**
- [ ] `CueBackend.predict_nebular_sed()` uses `neb_logU`, `neb_logZ_gas`, `neb_log_nH`, `neb_logNO`, `neb_logCO`
- [ ] `CloudyGridBackend.predict_nebular_sed()` uses same names (already mostly correct)
- [ ] `mix_dig_emission()` uses consistent `neb_*` names
- [ ] Line widths standardized to `line_sigma_kms` everywhere in nebular/ modules

**Emission line measurement:**
- [ ] `eline_catalog.py` created with `EMISSION_LINES` dict and `LINE_GROUPS`
- [ ] `eline_marginalization.py` imports from `eline_catalog.py`
- [ ] `eline_priors.py` imports from `eline_catalog.py`
- [ ] `build_line_design_matrix()` replaces separate narrow/broad functions
- [ ] `Model.predict_hbeta(params)` added to `model.py`
- [ ] `marginalize_emission_lines_cloudy()` accepts `model=` kwarg to auto-compute `l_hbeta`

**Dust cleanup:**
- [ ] `dust/emission.py` split into `dust/emission/` subdirectory (or at minimum the 2800-line file is refactored into logical sections)
- [ ] Each emission model's default parameters documented in `_EMISSION_MODEL_PARAMS` dict
- [ ] `apply_dust_emission()` validates kwargs against the active model's parameter list

**Registry unification:**
- [ ] `ModelRegistry` class created in `utils/registry.py`
- [ ] `tengri.sfh_registry`, `tengri.dust_registry`, `tengri.emission_registry`, `tengri.agn_registry` exposed at top level
- [ ] `ParamSpec.validate_against_backends(model)` added as debugging utility

**Tests:**
- [ ] `tests/unit/test_agn_lines.py` — tests nlr/blr return types are always tuples
- [ ] `tests/unit/test_nebular_naming.py` — tests `neb_logU` routing works for all backends
- [ ] `tests/unit/test_eline_catalog.py` — tests line catalog consistency
- [ ] `tests/unit/test_physics_api.py` — tests `lines_to_sed()`, `build_line_design_matrix()`

**Notebooks and docs:** (see documentation section above)
