# tengri Parameter System Redesign

**Date:** 2026-03-13
**Status:** Approved (design phase)

## Problem

The current tengri API tightly couples model configuration, parameter definitions, and inference. Parameters use cryptic short names (`tau_ps`, `sfr_norm`), the stochastic GP component is always required, redshift is fixed at construction time, and there's no way to fix/free individual parameters or specify priors.

## Goals

1. **ParamSpec as single source of truth** — one object defines parameter names, values, bounds, priors, and fixed/free state. Used for both mock generation (sampling) and inference (priors).
2. **Optional stochastic SFH** — pure parametric SFH works standalone; GP component is opt-in.
3. **Descriptive parameter names** — prefixed by component (`sfh_`, `psd_`, `met_`, `dust_`).
4. **Redshift as a parameter** — can be fixed or inferred like any other parameter.
5. **Distribution objects** — `Uniform`, `Gaussian`, `LogUniform`, `Fixed` with shorthands.
6. **Separate Fitter** — model defines physics, fitter defines inference strategy. Convenience shortcut on Model.
7. **Posterior object** — samples, derived quantities, resampling, ArviZ integration.
8. **Automatic precomputation** — SSP photometry precomputed based on ParamSpec (fixed vs range).
9. **Parallel mock generation** — vmap-compatible batch generation from ParamSpec or parameter tables.

## Architecture

```
ParamSpec ──→ Model(spec, ssp, filters) ──→ Precomputation (auto)
                    │
                    ├── model.predict(params)             → SED array
                    ├── model.predict_photometry(params)   → flux array
                    ├── model.predict_sfh(params)          → SFH dict
                    ├── model.mock(params, snr, key)       → MockData
                    ├── model.mock_batch(params, snr, key) → MockBatch
                    │
                    ├── model.fit(data, noise, method)     → Posterior  [convenience]
                    │
                    └── Fitter(model, data, noise)         [full control]
                              ├── .run("map", ...)         → Posterior
                              ├── .run("nuts", ...)        → Posterior
                              ├── .run("geovi", ...)       → Posterior
                              └── .run("nuts", init_from=map_result)
                                        │
                                        ▼
                                    Posterior
                                      ├── .samples         → dict of arrays
                                      ├── .derived         → stellar mass, SFR, ...
                                      ├── .params          → best-fit (MAP shortcut)
                                      ├── .summary()       → table
                                      ├── .sample(key, n)  → resample
                                      ├── .to_param_spec() → empirical ParamSpec
                                      └── .to_arviz()      → az.InferenceData
```

## 1. Distribution Classes

Located in `src/tengri/distributions.py`.

```python
class Uniform:
    """Uniform prior on [lo, hi]."""
    def __init__(self, lo: float, hi: float): ...
    def sample(self, key) -> float: ...
    def log_prob(self, x) -> float: ...
    @property
    def bounds(self) -> tuple[float, float]: ...
    @property
    def is_fixed(self) -> bool: return False

class Gaussian:
    """Gaussian prior, optionally clipped to [lo, hi]."""
    def __init__(self, mu: float, sigma: float,
                 lo: float = -jnp.inf, hi: float = jnp.inf): ...
    def sample(self, key) -> float: ...
    def log_prob(self, x) -> float: ...
    @property
    def bounds(self) -> tuple[float, float]: ...

class LogUniform:
    """Uniform in log10 space on [lo, hi]."""
    def __init__(self, lo: float, hi: float): ...
    def sample(self, key) -> float: ...
    def log_prob(self, x) -> float: ...

class Fixed:
    """Fixed value (not sampled, not inferred)."""
    def __init__(self, value: float): ...
    @property
    def is_fixed(self) -> bool: return True
```

**Shorthands** (resolved in `ParamSpec.__init__`):
- Scalar `0.3` → `Fixed(0.3)`
- Tuple `(0.1, 5.0)` → `Uniform(0.1, 5.0)`

All distribution classes inherit from a `Distribution` base class / Protocol with
`sample(key)`, `log_prob(x)`, `bounds`, and `is_fixed` properties. All methods
must be JAX-compatible (jittable). `Gaussian`, `LogUniform` have `is_fixed = False`.

**LogUniform Jacobian note:** The `LogUniform.log_prob(x)` returns `log(1/(x * log(hi/lo)))` —
the full density in linear space, not just a correction. The Fitter's sigmoid transform handles
the bounded→unbounded mapping; the LogUniform log_prob is an *additional* contribution on top,
replacing the implicit flat-in-linear Uniform that the sigmoid Jacobian assumes. Concretely:
`log_posterior += LogUniform.log_prob(x) - Uniform.log_prob(x)` where both are in physical space.

## 2. ParamSpec

Located in `src/tengri/param_spec.py`.

### Constructor

```python
spec = ParamSpec(
    # --- Mean SFH (double power law) ---
    sfh_alpha         = Uniform(0.1, 5.0),
    sfh_beta          = Uniform(0.1, 3.0),
    sfh_tau_peak_gyr  = Uniform(0.1, 12.0),
    sfh_peak_sfr      = (0.01, 200.0),          # shorthand → Uniform

    # --- Stochastic SFH ---
    psd_sigma         = 0.0,                      # shorthand → Fixed(0.0)
    psd_tau_myr       = Fixed(50.0),

    # --- Metallicity ---
    met_logzsol       = Gaussian(-0.3, 0.2, lo=-2.0, hi=0.2),

    # --- Dust ---
    dust_tau_bc       = Uniform(0.0, 4.0),
    dust_tau_diff     = 0.3,
    dust_slope        = -0.7,

    # --- Redshift ---
    redshift          = 0.1,                      # fixed

    # --- Settings (not parameters) ---
    stochastic        = False,
    n_grid            = 256,                      # only used if stochastic=True
    mean_sfh_type     = "double_powerlaw",
)
```

### Parameter name registry

| Name | Description | Units | Physical bounds |
|------|-------------|-------|-----------------|
| `sfh_alpha` | DPL falling slope (cosmic time, peak→present) | dimensionless | > 0 |
| `sfh_beta` | DPL rising slope (cosmic time, early→peak) | dimensionless | > 0 |
| `sfh_tau_peak_gyr` | DPL turnover time | Gyr | > 0 |
| `sfh_peak_sfr` | Peak SFR | Msun/yr | > 0 |
| `psd_sigma` | GP PSD amplitude | dex | >= 0 |
| `psd_tau_myr` | GP PSD damping timescale | Myr | > 0 |
| `met_logzsol` | log10(Z/Zsun) metallicity | dex | typically [-2, 0.5] |
| `dust_tau_bc` | Birth cloud optical depth | dimensionless | >= 0 |
| `dust_tau_diff` | Diffuse ISM optical depth | dimensionless | >= 0 |
| `dust_slope` | Dust attenuation power-law index | dimensionless | typically [-2, 0] |
| `redshift` | Source redshift | dimensionless | >= 0 |

### Internal storage

Parameters are stored internally in the units shown above. The forward model converts to the units needed by each component:
- `sfh_tau_peak_gyr` → multiplied by 1e9 to get years for `double_powerlaw()`
- `psd_tau_myr` → multiplied by 1e6 to get years for `compute_sqrt_power_drw()`

### Properties and methods

```python
spec.free_params      # list of free parameter names
spec.fixed_params     # list of fixed parameter names
spec.all_params       # all parameter names (excluding settings)
spec.n_free           # number of free parameters
spec.stochastic       # bool
spec.n_grid           # int (only meaningful if stochastic=True)

spec.sample(key)              # → dict of parameter values (one draw from priors)
spec.sample_batch(key, n)     # → dict of arrays, each shape (n,)
spec.get_fixed_values()       # → dict of {name: value} for fixed params
spec.get_distribution(name)   # → Distribution object for a parameter
spec.validate(params)         # → raises if params violate bounds
```

### Bound checking

On construction, `ParamSpec` validates:
- All bounds respect physical constraints (e.g., `sfh_alpha > 0`, `redshift >= 0`)
- Gaussian `lo`/`hi` are consistent with `mu ± 5*sigma`
- If `stochastic=True`, `psd_sigma` and `psd_tau_myr` must be present
- If `stochastic=False`, `psd_sigma` and `psd_tau_myr` are present but inactive
- `n_grid` should be a power of 2 for optimal FFT performance (recommended, not enforced)

### psd_xi handling

`psd_xi` is never user-specified. When `stochastic=True`:
- `spec.sample()` auto-generates `psd_xi ~ N(0, I)` of shape `(n_grid,)`
- During inference, `psd_xi` is always free with standard normal prior
- In the parameter dict, `psd_xi` appears alongside the other parameters

When `stochastic=False`:
- `psd_xi` is not generated or included in parameter dicts
- `psd_sigma` and `psd_tau_myr` are ignored by the forward model (GP contribution = 0)
- The Model calls `double_powerlaw()` directly, skipping GP/PSD computation entirely

### psd_xi plumbing in Model and Fitter

`psd_xi` does NOT appear in `ParamSpec.all_params` or `PARAM_MAP`. Instead:
- `Model` knows about `psd_xi` via `spec.stochastic` and `spec.n_grid`
- When `stochastic=True`, `Model._to_internal(params)` expects `psd_xi` in the dict
  and passes it through as `"xi"` to the internal ForwardModel
- `ParamSpec.sample()` auto-appends `psd_xi` when `stochastic=True`
- `Fitter` auto-adds `psd_xi` as a free parameter with shape `(n_grid,)` and
  standard normal prior (`-0.5 * xi^T xi`) when `stochastic=True`
- `Posterior.samples` includes `psd_xi` when `stochastic=True` (shape `(n_samples, n_grid)`)

### Stochastic=False code path

When `stochastic=False`, `Model` does NOT call `ForwardModel.__call__()` (which requires xi).
Instead, it calls the low-level components directly:
1. `double_powerlaw(age_yr, alpha, beta, tau, norm)` → mean SFH
2. `compute_csp_weights(sfr_on_ssp, ssp_ages)` → CSP weights
3. `interpolate_metallicity(...)`, `charlot_fall(...)`, `compute_csp_sed(...)` → SED

This avoids modifying `ForwardModel` and keeps the existing 181 tests intact.
When `stochastic=True`, `Model` delegates to `ForwardModel.__call__()` as before.

## 3. Model

Located in `src/tengri/model.py`. Replaces `forward_model.py`.

### Constructor

```python
model = Model(
    spec,                    # ParamSpec
    ssp_data,                # SSPData from load_ssp_data()
    filters=filters,         # list of FilterCurve, or output of load_filter_set()
    precompute=True,         # True (auto), False, or dict of settings
)
```

**Filters:** `Model` accepts the 3-tuple from `load_filter_set()` (extracts the FilterCurve list
internally) or a plain list of `FilterCurve` objects. Internally stores `filter_waves` and
`filter_trans` lists for the low-level photometry functions.

### Precomputation

Triggered automatically in `__init__` based on `ParamSpec`:

**Redshift:**
- If `redshift` is `Fixed(z)` → precompute SSP photometry at that z
- If `redshift` is a distribution → precompute on a grid, interpolate during forward pass
- `precompute={"redshift_grid": jnp.linspace(0.01, 3.0, 100)}` for explicit control
- `precompute={"redshift_grid": jnp.array([0.5, 1.0, 1.5, 2.0])}` for specific values

**Metallicity:**
- By default, uses all SSP metallicity grid points for interpolation
- `precompute={"met_grid": jnp.array([-1.5, -1.0, -0.5, 0.0])}` for subset

**Disable:** `precompute=False`

### Forward pass

```python
model.predict_sed(params)              # → rest-frame luminosity SED (erg/s/Hz), shape (n_wave,)
                                       #   redshift-independent (intrinsic SED before distance scaling)
model.predict_photometry(params)       # → observed flux densities (erg/s/cm²/Hz), shape (n_filters,)
                                       #   includes (1+z)/(4π dL²) scaling using params["redshift"]
model.predict_spectrum(params, wave_obs)  # → observed spectral flux, shape (n_pix,)
model.predict_sfh(params)              # → {"t_gyr": ..., "sfr_mean": ..., "sfr_full": ...}
model.predict_derived(params)          # → {"stellar_mass": ..., "sfr_100myr": ..., "ssfr": ...}
```

The forward pass handles the unit conversions:
- `sfh_tau_peak_gyr * 1e9` → years
- `psd_tau_myr * 1e6` → years
- If `stochastic=False`: SFH = mean SFH only (no GP, no psd_xi needed)
- If `stochastic=True`: SFH = mean * exp(GP - K(0)/2)

### Mock generation

```python
# Single mock
params = spec.sample(key)
mock = model.mock(params, snr=20.0, key=noise_key)
# Returns MockData(flux_true, flux_obs, noise, sed, params)

# Batch from prior
param_batch = spec.sample_batch(key, n=1000)
mock_batch = model.mock_batch(param_batch, snr=20.0, key=noise_key)
# Uses jax.vmap internally

# Batch from explicit table (dict of arrays)
param_table = {"sfh_alpha": jnp.array([1.0, 1.5]), "sfh_beta": jnp.array([0.8, 1.2]), ...}
mock_batch = model.mock_batch(param_table, snr=20.0, key=noise_key)
```

### Convenience fit

```python
posterior = model.fit(data, noise, method="map", data_type="photometry", **kwargs)
# Delegates to Fitter(model, data, noise, data_type).run(method, **kwargs)
```

## 4. Fitter

Located in `src/tengri/fitter.py`.

```python
fitter = Fitter(model, data, noise, data_type="photometry")
# data_type: "photometry", "spectroscopy", or "joint"
# For "joint", data and noise are concatenated [photometry, spectroscopy]

# MAP optimization
result_map = fitter.run("map", n_steps=1500, learning_rate=0.03)

# NUTS sampling
result_nuts = fitter.run("nuts", n_warmup=500, n_samples=1000)

# geoVI (recommended for stochastic models)
result_geovi = fitter.run("geovi", n_iterations=10, n_samples=6)

# Chain: MAP → NUTS (use MAP as initialization)
result_map = fitter.run("map", n_steps=1000)
result_nuts = fitter.run("nuts", init_from=result_map, n_warmup=300, n_samples=500)
```

### init_from semantics

When `init_from=posterior` is passed:
- The Fitter extracts `posterior.params` (physical space)
- Applies `to_unbounded()` to convert each free parameter to unbounded space
- Uses the result as the initial position for NUTS/geoVI
- For MAP: uses as starting point for Adam
- Does NOT transfer mass matrix or step size (those come from NUTS warmup)

### Stochastic mode warning

When `model.spec.stochastic=True` and `method="nuts"`:

```
⚠️  Stochastic SFH with NUTS: sampling 266 dimensions (256 psd_xi + 10 physical).
    This is computationally expensive. Recommended: method="geovi" (10-100x faster).
    Or reduce n_grid to lower dimensionality.
    Proceeding with NUTS...
```

### Internal parameter handling

The `Fitter` is responsible for:
1. Creating the unbounded parameter space (sigmoid transforms for bounded params)
2. Separating free vs fixed parameters (fixed values baked into the loss function)
3. Building the loss function: chi² + prior log-probabilities
4. Running the chosen optimizer/sampler
5. Converting results back to physical space → `Posterior`

**Prior integration:**
- `Uniform` → flat prior in bounded space (sigmoid transform handles Jacobian)
- `Gaussian` → adds `-0.5 * ((x - mu) / sigma)^2` to the log-posterior
- `LogUniform` → adds `log(1/x)` correction to the log-posterior
- `Fixed` → parameter excluded from optimization, value baked in

## 5. Posterior

Located in `src/tengri/posterior.py`.

```python
class Posterior:
    """Inference results with sampling and diagnostics."""

    # --- Core data ---
    samples: dict[str, jnp.ndarray] | None   # physical params, shape (n_samples, ...)
    params: dict[str, jnp.ndarray]            # best-fit or posterior mean
    method: str                                # "MAP (Adam)", "NUTS (BlackJAX)", "geoVI (NIFTy.re)"
    wall_time_s: float
    diagnostics: dict                          # method-specific

    # --- MAP-specific ---
    loss_history: jnp.ndarray | None          # convergence curve

    # --- Derived quantities (lazy, computed on first access) ---
    @property
    def derived(self) -> dict[str, jnp.ndarray]:
        """Derived quantities for each posterior sample (or single point for MAP).

        For MAP: computes on the single best-fit params → dict of scalars.
        For NUTS/geoVI: computes on all samples → dict of arrays, shape (n_samples,).
        Keys: stellar_mass, sfr_100myr, sfr_10myr, ssfr.
        Requires a reference to the Model (stored internally)."""

    # --- Methods ---
    def summary(self) -> dict:
        """Median and 68% credible intervals for all parameters.
        For MAP: returns point estimates (no intervals)."""

    def resample(self, key, n=1) -> dict:
        """Resample from posterior with replacement.
        Returns dict of arrays, each shape (n, ...)."""

    def to_param_spec(self) -> ParamSpec:
        """Convert posterior to empirical ParamSpec for mock generation.
        Fits a clipped Gaussian to each marginal posterior:
        Gaussian(mu=median, sigma=std, lo=min_sample, hi=max_sample).
        For MAP: returns Fixed values."""

    def to_arviz(self) -> "az.InferenceData":
        """Convert to ArviZ InferenceData for diagnostics.
        Adds chain dimension (1 chain). Includes derived quantities."""
```

## 6. Parameter Name Mapping (Internal)

The forward model components expect specific argument names. The `Model` class maps from the public `ParamSpec` names:

```python
PARAM_MAP = {
    "sfh_alpha":         ("alpha",    1.0),       # (internal_name, unit_scale)
    "sfh_beta":          ("beta",     1.0),
    "sfh_tau_peak_gyr":  ("tau_sfh",  1e9),       # Gyr → yr
    "sfh_peak_sfr":      ("sfr_norm", 1.0),
    "psd_sigma":         ("sigma_ps", 1.0),
    "psd_tau_myr":       ("tau_ps",   1e6),       # Myr → yr
    "met_logzsol":       ("log_z",    1.0),
    "dust_tau_bc":       ("tau_v1",   1.0),
    "dust_tau_diff":     ("tau_v2",   1.0),
    "dust_slope":        ("dust_n",   1.0),
    "redshift":          ("redshift", 1.0),
}
```

This mapping lives inside `Model` and is not exposed to users. The old internal names are kept in the low-level functions (`double_powerlaw`, `charlot_fall`, etc.) to avoid breaking the tested code.

## 7. File Structure

```
src/tengri/
├── distributions.py       # NEW: Uniform, Gaussian, LogUniform, Fixed
├── param_spec.py          # NEW: ParamSpec class
├── model.py               # NEW: Model class (replaces forward_model.py)
├── fitter.py              # NEW: Fitter class
├── posterior.py            # NEW: Posterior class
├── forward_model.py       # KEPT: low-level forward pass (used internally by Model)
├── models/                # UNCHANGED
│   ├── sfh/
│   ├── dust/
│   ├── sps/
│   └── observation/
├── inference/             # MODIFIED: used internally by Fitter
│   ├── common.py
│   ├── map_optimizer.py
│   ├── nuts.py
│   └── geovi.py
├── utils/                 # UNCHANGED
└── diagnostics/           # UNCHANGED
```

**Key principle:** The new API (`ParamSpec`, `Model`, `Fitter`, `Posterior`) is a high-level layer on top of the existing tested code. The low-level modules (`forward_model.py`, `inference/`, `models/`) are kept and used internally. No rewrite of tested code.

## 8. Quickstart Notebook Flow

```python
from tengri import Model, ParamSpec, Uniform, Gaussian, Fitter
from tengri import load_ssp_data, load_filter_set

# 1. Load data
ssp = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# 2. Define model (parametric only, no stochasticity)
spec = ParamSpec(
    sfh_alpha        = Uniform(0.5, 3.0),
    sfh_beta         = Uniform(0.3, 2.0),
    sfh_tau_peak_gyr = Uniform(0.5, 10.0),
    sfh_peak_sfr     = Uniform(0.1, 50.0),
    met_logzsol      = Uniform(-1.5, 0.2),
    dust_tau_bc      = Uniform(0.0, 3.0),
    dust_tau_diff    = Uniform(0.0, 2.0),
    dust_slope       = -0.7,
    redshift         = 0.1,
    stochastic       = False,
)
model = Model(spec, ssp, filters=filters)

# 3. Generate mock
true_params = {
    "sfh_alpha": 1.2, "sfh_beta": 1.0,
    "sfh_tau_peak_gyr": 4.0, "sfh_peak_sfr": 8.0,
    "met_logzsol": -0.3,
    "dust_tau_bc": 1.0, "dust_tau_diff": 0.3, "dust_slope": -0.7,
    "redshift": 0.1,
}
mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(0))

# 4. Fit with MAP
result = model.fit(mock.flux_obs, mock.noise, method="map")

# 5. Plot SFH recovery
sfh_true = model.predict_sfh(true_params)
sfh_fit = model.predict_sfh(result.params)

# 6. Now add stochasticity
spec_stoch = ParamSpec(
    sfh_alpha        = Uniform(0.5, 3.0),
    sfh_beta         = Uniform(0.3, 2.0),
    sfh_tau_peak_gyr = Uniform(0.5, 10.0),
    sfh_peak_sfr     = Uniform(0.1, 50.0),
    psd_sigma        = Uniform(0.1, 3.0),    # NOW FREE
    psd_tau_myr      = Uniform(1.0, 300.0),  # NOW FREE
    met_logzsol      = Uniform(-1.5, 0.2),
    dust_tau_bc      = Uniform(0.0, 3.0),
    dust_tau_diff    = Uniform(0.0, 2.0),
    dust_slope       = -0.7,
    redshift         = 0.1,
    stochastic       = True,
    n_grid           = 128,
)
model_stoch = Model(spec_stoch, ssp, filters=filters)

# 7. Fit with geoVI (recommended for stochastic)
posterior = model_stoch.fit(mock_stoch.flux_obs, mock_stoch.noise, method="geovi")
```

## 9. Deprecation

The old public API (`ForwardModel`, `ModelConfig`, `PriorConfig`, `generate_mock`) is kept
as internal implementation but removed from `__init__.py` exports. The new public API is:

```python
# New public API
from tengri import Model, ParamSpec, Fitter, Posterior
from tengri import Uniform, Gaussian, LogUniform, Fixed
from tengri import load_ssp_data, load_filter_set
```

No deprecation warnings needed — the codebase is private and pre-release.

## 10. SFH Type Extensibility

For now, only `mean_sfh_type="double_powerlaw"` is supported. The parameter registry
(sfh_alpha, sfh_beta, sfh_tau_peak_gyr, sfh_peak_sfr) is specific to DPL. Future SFH
types (delayed-tau, constant) would define their own parameter sets. The `mean_sfh_type`
setting exists to allow this extension without breaking the API. ParamSpec validates that
the provided sfh_* parameters match the chosen SFH type.

## 11. Testing Strategy

### Unit tests
- **distributions.py**: sample, log_prob, bounds, JAX jit compatibility, edge cases
- **param_spec.py**: construction, validation, sampling, batch sampling, bound checks,
  shorthand resolution, stochastic vs parametric, psd_xi auto-generation
- **model.py**: forward pass equivalence with old ForwardModel (stochastic=True),
  parametric-only forward pass (stochastic=False), mock generation, batch generation,
  filter input handling, precomputation
- **fitter.py**: MAP convergence, NUTS runs without error, init_from chaining,
  data_type handling, stochastic warning
- **posterior.py**: summary stats, resampling, to_arviz, to_param_spec, derived for MAP

### Integration tests
- **Round-trip recovery**: generate mock from ParamSpec → fit → verify parameter recovery
  within 3-sigma for MAP, within posterior 95% CI for NUTS
- **Parametric vs stochastic**: same galaxy, fit both ways, verify consistent broad SED shape
- **Batch generation**: generate 100 mocks, verify all finite, physical flux ranges
- **Redshift inference**: fix all params except redshift, verify z recovery from photometry

All existing 181 tests continue to pass (low-level code unchanged).
