(predicting-properties)=

# Predicting and properties

The core workflow after building a model is to generate predictions — forward pass results and derived physical quantities. This page covers the three prediction surfaces, the property catalog, and common workflows.

## The three prediction surfaces

tengri provides three interfaces for computing predictions, each optimized for a different use case:

### 1. Lazy exploration: `model.predict(params)`

For interactive work — plotting, diagnostics, manual inspection — the **recommended** interface is the lazy `Prediction` object:

```python
pred = model.predict(params)

# Lazy evaluation: only computed when first accessed
print(pred.sfh.stellar_mass)      # shape: scalar, triggers SFH computation
print(pred.sed.l_bol)             # shape: scalar, triggers SED computation
print(pred.lines.halpha)          # shape: scalar, triggers nebular computation

# Repeated access reuses cached results (no recomputation)
print(pred.sfh.stellar_mass)      # cached, no re-run
```

The `Prediction` object caches intermediate results, so related quantities share expensive computation. This is ideal for one-off fits or notebook exploration.

**When to use:** Single-galaxy analysis, model diagnostics, fitting a single spectrum.

### 2. Inference hot path: `model.predict_photometry()`, `model.predict_spectrum()`, etc.

For inference loops and likelihood evaluation, use the **lean methods** directly:

```python
# These methods are called by the inference loop
photometry = model.predict_photometry(params)   # the filters built into the model
spectrum = model.predict_spectrum(params, wave_obs)
lines = model.predict_emission_lines(params)
```

These bypass the lazy `Prediction` wrapper and return only what you request, with no caching overhead. They are **JIT-compatible** and safe to call from inside an inference loop. The lean `predict_photometry` uses the filters the model was built with; to evaluate a *different* filter set at runtime, use the rich accessor `pred.photometry(filters=[...])` instead (see [Exact vs fast photometry](#exact-vs-fast-photometry)).

**When to use:** Likelihood evaluation, fitting, parameter sweeps.

### 3. Batch (posterior) computation: `model.predict_properties()` + `jax.vmap`

For computing derived quantities over many parameter sets — posterior chains, mock catalogs, parameter grids — use the JIT-compatible property method with `jax.vmap`:

```python
import jax
import jax.numpy as jnp

# Batch of parameter sets (e.g., from posterior samples or a grid)
params_batch = spec.sample_batch(key, n=10_000)

# vmap over property computation
def predict_one_property(params):
    return model.predict_properties(params, names=("stellar_mass",))["stellar_mass"]

fn = jax.vmap(predict_one_property)
stellar_mass_batch = fn(params_batch)  # shape (10_000,)
```

This path is **100–1000× faster** than calling `.predict()` in a loop, with no Python-level overhead.

**When to use:** Computing posterior summaries, generating mock catalogs, batch inference diagnostics.

## The property catalog

A tengri model publishes **derived quantities** — stellar mass, SFR, emission line luminosities, colors, and diagnostics. These are called **properties**.

### Discovering properties

Use `list_properties()` and `describe_property()` to explore what is available:

```python
from tengri import list_properties, describe_property

# Print all available properties
print(list_properties())

# Filter by group
print(list_properties(group="sfh"))      # star formation history
print(list_properties(group="sed"))      # spectral energy distribution
print(list_properties(group="lines"))    # emission lines
print(list_properties(group="radio"))    # radio
print(list_properties(group="xray"))     # X-ray
print(list_properties(group="ionizing")) # ionizing photon budget

# Get detailed info about one
print(describe_property("stellar_mass"))
# Output:
#   name         stellar_mass
#   units        Msun
#   group        sfh
#   component    stellar
#   description  Total stellar mass formed by the SFH — its time-integral,
#                1.5-1.9x above stellar_mass_surviving
```

```{warning}
`stellar_mass` is the mass **formed** — the time-integral of the SFH. Stellar
evolution returns a third to a half of it to the ISM, so it runs 1.5-1.9x above
`stellar_mass_surviving` on ordinary populations. Quote whichever one your
comparison sample uses; they are not interchangeable.
```

### Accessing properties: two forms

Properties can be accessed in two ways — attribute sugar or dict-like access:

```python
pred = model.predict(params)

# Form 1: Attribute sugar via lazy groups
print(pred.sfh.stellar_mass)                 # via the sfh group
print(pred.sed.l_bol)                        # via the sed group
print(pred.lines.halpha)                     # via the lines group

# Form 2: Direct attribute (shorthand)
print(pred.stellar_mass)                     # same as pred.sfh.stellar_mass
print(pred.l_bol)                            # same as pred.sed.l_bol

# Form 3: Dict-like access (used internally)
print(pred.properties["stellar_mass"])       # explicit dict access
```

### Full property catalog

```{include} _property_table.md
```

## Topology: same names, more axes

One principle unifies the API: **same property names work everywhere**, but with different shapes depending on context.

| Context | Shape | Example |
|---------|-------|---------|
| Prediction (single galaxy) | scalar | `pred.stellar_mass` → `Array(1.5e10, dtype=float64)` |
| Posterior (samples) | `(n_samples,)` | `posterior.properties["stellar_mass"]` → `Array([...], shape=(5000,))` |
| Population catalog | `(n_galaxies,)` or `(n_galaxies, n_apertures)` | Grid or mock catalog |

So you can write **one** function that works on all topologies:

```python
def print_stellar_mass(entity):
    """Works on Prediction, Posterior, or catalog."""
    mass = entity.properties["stellar_mass"]
    print(f"Stellar mass: {mass}")

# All three calls work identically:
print_stellar_mass(pred)           # scalar
print_stellar_mass(posterior)       # (n_samples,) array
print_stellar_mass(population)      # (n_galaxies,) array
```

### Posterior-specific: credible intervals

For posteriors, use the `.ci()` method to compute credible intervals:

```python
result = model.fit(data, noise)
posterior = result.posterior

# Credible interval on a property
lo, median, hi = posterior.properties.ci("stellar_mass")
# Default: 68% (16th/84th percentiles)

# Custom credible level
lo, median, hi = posterior.properties.ci("stellar_mass", level=0.95)
# 95% credible interval (2.5th/97.5th percentiles)

print(f"Stellar mass: {median:.2e} (−{median - lo:.2e} / +{hi - median:.2e}) Msun")
```

## SED vs spectrum: naming and conventions

Two distinct concepts — understand the difference:

### SED: panchromatic model grid

`rest_sed` and `obs_sed` are **callables with a default**, like `photometry()` and `spectrum()`. Call with no argument for the model's own grid, or pass a wavelength grid to resample onto yours:

```python
pred = model.predict(params)

# Rest-frame SED on the model's own grid
lnu  = pred.rest_sed()          # (n_wave,)  [erg/s/Hz]
wave = pred.wave_rest           # (n_wave,)  [Angstrom]  — the matching axis

# ...or on a grid you choose
lnu = pred.rest_sed(np.logspace(3, 5, 500))     # rest-frame Angstrom

# Observed-frame: redshift + IGM attenuation applied
lnu  = pred.obs_sed()           # (n_wave,)  [erg/s/Hz] — STILL L_nu, not a flux
wave = pred.wave_obs            # the matching observed-frame axis
lnu  = pred.obs_sed(np.logspace(3, 5, 500))     # OBSERVED-frame Angstrom
```

**The wavelength argument is in the accessor's own frame** — `rest_sed(wave)` takes rest-frame Å, `obs_sed(wave_obs)` takes observed-frame Å.

**The SED array does not carry its axis.** Use `pred.wave_rest` / `pred.wave_obs`. Never reconstruct the observed axis by hand as `wave * (1 + params["redshift"])`: a `Fixed` redshift is legitimately absent from `params`, and a `0.0` fallback silently puts the galaxy at 10 pc.

The model grid is the SSP grid, auto-extended when dust emission, radio or X-ray components are configured.

```{note}
`pred.rest_sed` **without the parentheses** raises `TypeError`. It is a method, and a bound method coerced with `np.asarray` would otherwise produce a `dtype=object` array that plots silently-wrong results. The error message tells you the fix.
```

### Units: the distance is applied at *projection*, not on the SED

**`obs_sed` is not a flux.** "Observed" names the *frame*, not a flux conversion.

| surface | quantity | units |
|---|---|---|
| `pred.rest_sed()` | L_ν, rest-frame axis | erg/s/Hz |
| `pred.obs_sed()` | **L_ν** — observed-frame *axis* + IGM | **erg/s/Hz** |
| `pred.photometry()` | F_ν | erg/s/cm²/Hz |
| `pred.magnitudes()` | AB magnitude | — |
| `pred.spectrum()` | F_ν | erg/s/cm²/Hz |

`obs_sed()` does **not** apply `(1+z)/(4π d_L²)`; that factor lives in the projection layer. The only differences from `rest_sed()` are the wavelength axis and IGM absorption — at z = 3 the two arrays are identical everywhere above rest-frame Lyman-α.

Integrating `obs_sed()` as if it were a flux is wrong by ~57 orders of magnitude. **If you want a flux, use `photometry()` or `spectrum()`.**

### Spectrum: instrument-specific, LSF-convolved, calibrated

The `.spectrum()` method returns an **instrument-ready** spectrum — convolved with the line-spread function, rebinned to a specific wavelength grid, and calibrated. It requires the model to have a spectroscopy channel: build with `observation=Observation(spectroscopy=...)`, otherwise `pred.spectrum(...)` raises `ValueError` (a photometry-only model has no LSF or calibration to apply). For a bare model SED with no instrument convolution, use `pred.obs_sed(wave_obs)` instead.

```python
# Spectrum at specific observer-frame wavelengths.
# Give the grid you actually observed on — do not build it from a rest-frame
# grid times (1 + z). `wave_obs` is already in the observer frame.
obs_wave = jnp.linspace(4000, 7000, 200)  # Å, observer-frame

spec = pred.spectrum(wave_obs=obs_wave)
# spec.flux — F_ν at observed wavelengths [erg/s/cm²/Å]
# spec.error — noise model prediction [erg/s/cm²/Å]
```

**Key differences from SED:**
- **Grid**: User-specified (not the model grid)
- **Frame**: Observed-frame wavelengths
- **Calibration**: Applied (photometry, line spread, noise)
- **Units**: F_ν (erg/s/cm²/Å), a flux *per unit wavelength*

**Summary table:**

| Aspect | SED (rest_sed, obs_sed) | Spectrum |
|--------|------------------------|----------|
| Grid | Model internal | Observer-specified |
| Frame | Rest (rest_sed) / Obs (obs_sed) | Observer (wavelengths input as observed-frame) |
| Calibration | None | Full (LSF, noise model, photometric calibration) |
| Units | erg/s/Hz (L_ν) | erg/s/cm²/Å (F_ν) — divided by 4π d_L² |
| Use case | Model diagnostics, color/index computation | Spectroscopic fitting, comparison with data |

## Exact vs fast photometry

By default, photometry and spectroscopy predictions are **exact** — integrated over the model's full wavelength grid:

```python
pred = model.predict(params)

# Exact integration over the full SED grid
photometry = pred.photometry()  # filters built into the model
spectrum = pred.spectrum(wave_obs=...)
```

If the model was built with a precomputation (`approx=WavePrecomp(...)`), you can opt into a **fast** lookup-table path:

```python
# ONLY valid if the model was built with approx=WavePrecomp()
photometry_fast = pred.photometry(approx=True)  # uses precomputed LUT
spectrum_fast = pred.spectrum(wave_obs=..., approx=True)  # interpolates precomputed LUT
```

### When to use fast

- **During inference**: Fitting uses the approximation that was baked in at build time. Set it with `approx=` at build (or `model.with_approx(...)`, which returns a clone); read it back with `model.approx`, which reports the LUTs that actually resolved.
- **Post-fit inspection**: Use `approx=True` to match inference exactly.
- **Speed-critical analysis**: e.g., generating a mock catalog of 1 million galaxies.

### When not to use fast

- **Changing filters at runtime** (`pred.photometry(filters=[...])`): incompatible with `approx=True` — raises `ValueError`.
- **Exact science**: Analysis grids where pre-computed filters would bias the results.

### The principle

A speed knob must never silently change the physics. If you pass `approx=True`, you are explicitly saying *"use the approximation baked into the model"*. Changing filters invalidates the precomputation, so it is forbidden.

## Mock catalogs: batch prediction from arbitrary parameters

The model is a pure function of parameters. You can generate mock catalogs from **any** batch of parameter sets — prior samples, a grid, hand-tuned values — without a fit or observed data:

```python
import jax
import jax.numpy as jnp
from tengri import SEDModel, Observation, Photometry

# Build a model with only photometry (mock generation is cheap)
model = SEDModel.build(
    ssp_data=ssp,
    observation=Observation(
        photometry=Photometry.from_names(["sdss-g", "sdss-r", "sdss-i"])
    ),
)

# Method 1: Sample from the prior
key = jax.random.PRNGKey(0)
params_prior = model.spec.sample_batch(key, n=1000)  # shape: dict of (1000,) arrays

# Method 2: Hand-built parameter dict (all parameter names required)
params_grid = {
    "sfh_dpl_alpha": jnp.linspace(0.5, 2.0, 50),
    "sfh_dpl_beta": jnp.linspace(0.1, 1.5, 50),
    "dust_tau_bc": jnp.linspace(0.0, 1.0, 50),
    # ... all free parameters
}

# Batch prediction: vmap over the model forward
def predict_photometry_batch(params_batch):
    def predict_one(p):
        return model.predict_photometry(p)
    return jax.vmap(predict_one)(params_batch)

mocks = predict_photometry_batch(params_prior)
# mocks.photometry shape: (1000,), each element is the photometry dict

# Extract and table
stellar_masses = jax.vmap(lambda p: model.predict_properties(p, names=("stellar_mass",))["stellar_mass"])(params_prior)
sfrs = jax.vmap(lambda p: model.predict_properties(p, names=("sfr_100myr",))["sfr_100myr"])(params_prior)

# Write to file or dataframe
import pandas as pd
catalog = pd.DataFrame({
    "stellar_mass_msun": np.array(stellar_masses),
    "sfr_100myr_msunyr": np.array(sfrs),
    # ... add photometry as columns
})
catalog.to_csv("mock_catalog.csv", index=False)
```

This workflow is pure JAX — no loop, fully differentiable, and trivial to parallelize.

## Error handling: unknown properties raise, never return NaN

If you request a property the model doesn't provide, you get a clear error naming the
available ones — never a silent `NaN` or `None`. Which exception you catch depends on how
you asked:

| Access style | Raises |
|---|---|
| `pred.nonexistent_property` (attribute sugar) | `AttributeError` |
| `pred.properties["nonexistent"]` (catalog) | `KeyError` |

```python
pred = model.predict(params)

try:
    print(pred.nonexistent_property)        # attribute sugar
except AttributeError as e:
    print(f"Error: {e}")
    print("Available properties:", list(pred.properties.keys()))

try:
    print(pred.properties["nonexistent"])   # catalog access
except KeyError as e:
    print(f"Error: {e}")
```

The model always knows which properties are available. Never silently return NaN for an unknown name.

## See also

- `tengri.list_properties()` — discover all available properties
- `tengri.describe_property()` — get detailed info on one
- `Prediction` — lazy exploration object
- `Posterior` — fitting results with credible intervals
- `SEDModel.predict_properties()` — the underlying JIT/vmap-safe method
- `Observation.predict()` — likelihood evaluation (`predict_via_precomp()` on
  the precompute path)
