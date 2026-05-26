# Observation API Guide

A comprehensive guide to tengri's unified Observation API for configuring
photometric and spectroscopic observations.

---

## Overview

The `Observation` class is a frozen, declarative configuration container that
bundles everything about *what you are observing* into a single object:

- **Photometry** -- which broadband filters
- **Spectroscopy** -- wavelength grid, instrument resolution, calibration, emission lines
- **Noise model** -- calibration floor, likelihood shape

It sits at the top of the tengri pipeline:

```
Observation  -->  Model  -->  Fitter  -->  Posterior
  (config)       (physics)   (inference)   (results)
```

**Why it exists.** Before `Observation`, filters were raw lists passed to
`Model`, spectroscopy precomputation was a manual step (easy to forget, causing
a 10--50x performance penalty), joint fitting required fragile manual array
concatenation, and noise/calibration parameters had to be hand-wired into
`ParamSpec`. The `Observation` class eliminates all of these pain points:

- Capability queries (`can_do_photometry`, `can_do_spectroscopy`) replace
  string-based `data_type` dispatch.
- Instrument factories (`nirspec_prism`, `nirspec_g140m`, `constant_r`)
  encode instrument knowledge.
- Observation-driven parameters (calibration coefficients, noise floor) are
  auto-merged into `ParamSpec` by `Model`.
- Spectroscopy precomputation triggers automatically when redshift is fixed.
- `pack_data` / `unpack_prediction` provide validated data concatenation for
  joint fitting.

**Architecture.** `Observation` is a composition of three optional frozen
dataclasses -- no inheritance hierarchy:

```
Observation (frozen dataclass)
+-- Photometry          (optional) -- filters + metadata
+-- SpectroscopyConfig  (optional) -- wave grid, resolution, LSF, calibration, elines
+-- NoiseConfig         (optional) -- calibration floor, Student-t dof
```

At least one of `photometry` or `spectroscopy` must be provided.

The `Observation` never enters JAX-traced code. It configures what the `Model`
precomputes and what the `Fitter` dispatches.

---

## Quick Start

Five lines from import to observation:

```python
from tengri import Observation, Photometry

obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
)
print(obs.summary())
```

Output:

```
Observation
--------------------------------------------------
  Photometry : 5 filters: sdss_u, sdss_g, sdss_r, sdss_i, sdss_z
  Data type  : photometry
  N data     : 5
```

Full pipeline:

```python
from tengri import Model, ParamSpec, Fitter, Observation, Photometry
from tengri import Uniform, Fixed, load_ssp_data

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")

obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
)

spec = ParamSpec(
    sfh_tsnorm_log_total_mass=10.0, 3),
    sfh_tsnorm_tau_rise_gyr=Uniform(0.1, 5.0),
    sfh_tsnorm_tau_fall_gyr=Uniform(0.1, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 3.0),
    redshift=Fixed(0.5),
)

# observation= carries filters; data_type is inferred automatically
model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, flux_obs, noise_obs)
posterior = fitter.run("native_geovi")
print(posterior.summary_table())
```

---

## Photometry Configuration

The `Photometry` class wraps filter transmission curves into a frozen container.

### `Photometry.from_names()` -- load by registry name

The simplest way to configure photometry. Pass a list of short names from the
built-in `FILTER_REGISTRY`:

```python
from tengri import Photometry

phot = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
print(phot.n_filters)   # 5
print(phot.names)        # ('sdss_u', 'sdss_g', 'sdss_r', 'sdss_i', 'sdss_z')
```

Filters are downloaded from the SVO Filter Profile Service on first use and
cached locally in `data/filters/`.

### `Photometry.from_filter_set()` -- backward compatibility

If you have existing filter data from the legacy `load_filter_set()` function
or a list of `FilterCurve` objects:

```python
from tengri import load_filter_set, Photometry

# From the legacy 3-tuple
filter_set = load_filter_set(["sdss_r", "sdss_i"])
phot = Photometry.from_filter_set(filter_set)

# From a list of FilterCurve objects
from tengri.observation.photometry import FilterCurve
curves = [FilterCurve(name="custom", wave=wave_arr, trans=trans_arr)]
phot = Photometry.from_filter_set(curves)
```

### Custom filters

For filters not in the registry, construct `FilterCurve` objects directly:

```python
import jax.numpy as jnp
from tengri.observation.photometry import FilterCurve
from tengri import Photometry

my_filter = FilterCurve(
    name="my_narrowband",
    wave=jnp.linspace(6550, 6580, 100),  # Angstrom
    trans=jnp.ones(100),                   # top-hat transmission
)

phot = Photometry(filters=(my_filter,))
```

### Available filter registries

The built-in `FILTER_REGISTRY` includes 50+ filters across major surveys:

| Prefix     | Instrument / Survey        | Filters                                        |
|------------|---------------------------|------------------------------------------------|
| `sdss_`    | SDSS                      | `u`, `g`, `r`, `i`, `z`                       |
| `lsst_`    | LSST / Rubin              | `u`, `g`, `r`, `i`, `z`, `y`                  |
| `hst_`     | HST ACS/WFC + WFC3/IR     | `f435w`, `f606w`, `f775w`, `f814w`, `f850lp`, `f105w`, `f125w`, `f140w`, `f160w` |
| `jwst_`    | JWST NIRCam               | `f090w`, `f115w`, `f150w`, `f200w`, `f277w`, `f356w`, `f410m`, `f444w` |
| `roman_`   | Roman / WFI               | `f062`, `f087`, `f106`, `f129`, `f158`, `f184`, `f213` |
| `euclid_`  | Euclid VIS + NISP         | `vis`, `y`, `j`, `h`                          |
| `hsc_`     | Subaru HSC                | `g`, `r`, `i`, `z`, `y`                       |
| `des_`     | DES / DECam               | `g`, `r`, `i`, `z`, `Y`                       |
| `galex_`   | GALEX                     | `fuv`, `nuv`                                   |
| `wise_`    | WISE                      | `w1`, `w2`, `w3`, `w4`                         |
| `2mass_`   | 2MASS                     | `j`, `h`, `ks`                                 |
| `herschel_`| Herschel PACS             | `70`, `100`, `160`                             |
| `johnson_` | Generic Johnson           | `u`, `v`, `j`                                  |

To list all available filters programmatically:

```python
from tengri.observation.filters import FILTER_REGISTRY
print(sorted(FILTER_REGISTRY.keys()))
```

---

## Spectroscopy Configuration

The `SpectroscopyConfig` class declares the spectroscopic instrument setup.

### Basic construction

At minimum, provide the observed-frame wavelength grid:

```python
import jax.numpy as jnp
from tengri import SpectroscopyConfig

wave_obs = jnp.linspace(6000, 25000, 1000)  # Angstrom
spec_config = SpectroscopyConfig(wave_obs=wave_obs)
print(spec_config.n_pixels)  # 1000
print(spec_config.has_lsf)   # False (no resolution set)
```

### Instrument factories

Factory methods encode instrument-specific knowledge so you do not need to
look up resolution curves:

#### JWST NIRSpec PRISM

Variable resolution R ~ 30--330 across 0.6--5.3 microns (Jakobsen et al. 2022):

```python
spec_config = SpectroscopyConfig.nirspec_prism(
    wave_obs,                    # observed wavelength grid (Angstrom)
    calibration_order=3,         # optional: Chebyshev calibration
    eline_marginalize=True,      # optional: emission line marginalization
)
# resolution is a per-pixel array computed from the PRISM dispersion model
```

#### JWST NIRSpec G140M

Roughly constant R ~ 1000:

```python
spec_config = SpectroscopyConfig.nirspec_g140m(wave_obs)
# resolution = 1000.0 (scalar)
```

#### Constant-R spectrograph

For any instrument with approximately constant spectral resolution:

```python
spec_config = SpectroscopyConfig.constant_r(wave_obs, R=3000)
```

### Custom resolution

For instruments with a wavelength-dependent resolution not covered by the
built-in factories, pass an array of per-pixel R values:

```python
import jax.numpy as jnp

# Example: linearly increasing resolution
wave_obs = jnp.linspace(3500, 9500, 2000)
R_array = 500 + 1000 * (wave_obs - 3500) / (9500 - 3500)  # R = 500 to 1500

spec_config = SpectroscopyConfig(
    wave_obs=wave_obs,
    resolution=R_array,           # per-pixel R(lambda)
    sigma_lib_kms=70.0,           # SSP library resolution (MILES default)
    lsf_n_bins=16,                # piecewise LSF approximation bins
)
```

### LSF configuration

Two parameters control the line-spread function convolution:

- `sigma_lib_kms` (default: 70.0) -- the velocity dispersion of the SSP
  library in km/s. This is subtracted in quadrature so the model only adds
  the *difference* between library and instrument resolution. Set to 70.0 for
  MILES, 0.0 if your SSP library has negligible broadening.
- `lsf_n_bins` (default: 16) -- number of piecewise-constant bins for
  approximating wavelength-dependent LSF convolution. More bins = more
  accurate but slower compilation.

---

## Calibration Polynomial

### What it does

A multiplicative Chebyshev polynomial that absorbs flux calibration
uncertainties in the spectrum. The model prediction is multiplied by:

```
C(lambda) = 1 + c1*T1(x) + c2*T2(x) + ... + cN*TN(x)
```

where `T_i` are Chebyshev polynomials of the first kind evaluated on the
normalized wavelength grid `x` in [-1, 1], and `c_i` are free parameters
with `Gaussian(0, 0.1)` priors.

### When to use it

Use a calibration polynomial when:

- The flux calibration of the spectrum is uncertain (common for slit
  spectroscopy, especially at the edges of the wavelength range).
- You want to fit spectral *shape* without being sensitive to the overall
  normalization.

Typical orders: 2--5 for PRISM data, 1--3 for grating spectroscopy.

### How calibration_order maps to free params

`calibration_order=N` creates N free parameters named `cal_c1` through
`cal_cN`, each with a `Gaussian(0, 0.1)` prior:

```python
spec_config = SpectroscopyConfig.nirspec_prism(
    wave_obs,
    calibration_order=3,
)

# This auto-generates:
# cal_c1 ~ Gaussian(0, 0.1)
# cal_c2 ~ Gaussian(0, 0.1)
# cal_c3 ~ Gaussian(0, 0.1)

print(spec_config.get_calibration_params())
# {'cal_c1': Gaussian(0.0, 0.1), 'cal_c2': Gaussian(0.0, 0.1), 'cal_c3': Gaussian(0.0, 0.1)}
```

Setting `calibration_order=0` (the default) disables the calibration
polynomial entirely -- no extra parameters are created.

---

## Emission Line Marginalization

### What it does

Analytically marginalizes over emission line amplitudes during the likelihood
computation. This means the model does not need explicit emission line
parameters -- their amplitudes are integrated out, reducing dimensionality
while still accounting for line flux.

### When to use it

Use emission line marginalization when:

- Fitting low-resolution spectroscopy (R < 1000) where individual line
  profiles are unresolved.
- You care about the continuum shape and broadband SED but not individual
  line fluxes.

### Default line list

The default list includes 13 common optical/UV emission lines:

| Line          | Rest wavelength (Angstrom) |
|---------------|---------------------------|
| Ly-alpha      | 1215.67                   |
| H-delta       | 4101.73                   |
| H-gamma       | 4340.46                   |
| H-beta        | 4861.33                   |
| [OIII] 4959   | 4958.91                   |
| [OIII] 5007   | 5006.84                   |
| H-alpha       | 6562.80                   |
| [NII] 6548    | 6548.05                   |
| [NII] 6583    | 6583.45                   |
| [OII] 3726    | 3726.03                   |
| [OII] 3729    | 3728.82                   |
| [SII] 6717    | 6716.44                   |
| [SII] 6731    | 6730.81                   |

### Custom line lists

Pass a custom array of rest-frame wavelengths to override the defaults:

```python
import jax.numpy as jnp

custom_lines = jnp.array([4861.33, 5006.84, 6562.80])  # H-beta, [OIII], H-alpha

spec_config = SpectroscopyConfig.nirspec_prism(
    wave_obs,
    eline_marginalize=True,
    eline_wavelengths=custom_lines,
    eline_prior_sigma=50.0,        # prior width on line amplitudes (default: 100.0)
)
```

---

## Noise Configuration

The `NoiseConfig` class controls the noise model used during inference.

### Fixed calibration floor

A constant fractional floor added in quadrature with the observational noise:

```
sigma_eff = sqrt(sigma_obs^2 + (f_cal * model)^2)
```

```python
from tengri import NoiseConfig

noise = NoiseConfig(calibration_floor=0.02)  # fixed 2% floor
```

This creates a `Fixed(0.02)` parameter named `noise_frac_cal` in the model.

### Free calibration floor (with prior)

Make the calibration floor a free parameter by passing a `Distribution`:

```python
from tengri import NoiseConfig, Uniform

noise = NoiseConfig(calibration_floor=Uniform(0.01, 0.15))
```

This creates a free parameter `noise_frac_cal ~ Uniform(0.01, 0.15)` that
the sampler explores during inference.

### Student-t likelihood

For outlier-robust fitting, use a Student-t likelihood instead of Gaussian:

```python
noise = NoiseConfig(
    calibration_floor=Uniform(0.01, 0.10),
    student_t_dof=5.0,   # degrees of freedom; lower = heavier tails
)
```

This adds a `Fixed(5.0)` parameter named `noise_dof`. Setting
`student_t_dof=None` (the default) uses a standard Gaussian likelihood.

### Default noise

`NoiseConfig()` with no arguments means: no calibration floor, Gaussian
likelihood. No parameters are generated:

```python
noise = NoiseConfig()
print(noise.get_params())  # {}
```

---

## Joint Fitting

### Creating a joint Observation

Provide both `photometry` and `spectroscopy`:

```python
from tengri import Observation, Photometry, SpectroscopyConfig, NoiseConfig

obs = Observation(
    photometry=Photometry.from_names([
        "jwst_f090w", "jwst_f150w", "jwst_f200w",
        "jwst_f277w", "jwst_f356w", "jwst_f444w",
    ]),
    spectroscopy=SpectroscopyConfig.nirspec_prism(
        wave_obs,
        calibration_order=3,
    ),
    noise=NoiseConfig(calibration_floor=0.02),
)

print(obs.is_joint)      # True
print(obs.data_type)     # "joint"
print(obs.n_data_phot)   # 6
print(obs.n_data_spec)   # len(wave_obs)
print(obs.n_data)        # 6 + len(wave_obs)
```

### `pack_data` / `unpack_prediction`

For joint fitting, data must be concatenated in the canonical order:
**photometry first, then spectroscopy**. Use `pack_data` for validated
concatenation and `unpack_prediction` to split results back:

```python
# Pack observed data with shape validation
data = obs.pack_data(phot=phot_flux, spec=spec_flux)
noise = obs.pack_data(phot=phot_noise, spec=spec_noise)

# Fit
model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, data, noise)
posterior = fitter.run("native_geovi")

# Unpack model prediction for plotting
pred = model.predict(posterior.median_params)
result = obs.unpack_prediction(pred)
# result["photometry"]   -- shape (6,)
# result["spectroscopy"] -- shape (n_pixels,)
```

`pack_data` validates array shapes against the observation configuration and
raises `ValueError` if they do not match:

```python
# This raises ValueError: phot shape (5,) doesn't match expected (6,)
obs.pack_data(phot=jnp.ones(5), spec=spec_flux)
```

### Ordering convention

The canonical data order is always `[photometry, spectroscopy]`. This is
enforced by `pack_data` and assumed by `unpack_prediction`. You should never
need to manually concatenate arrays when using the Observation API.

---

## Auto-Merge: How Observation Params Enter ParamSpec

When you create a `Model` with an `Observation`, observation-driven parameters
are automatically merged into `ParamSpec`.

### What gets auto-merged

1. **Calibration coefficients** from `SpectroscopyConfig`:
   `cal_c1`, `cal_c2`, ..., `cal_cN` (one per `calibration_order`)
2. **Noise parameters** from `NoiseConfig`:
   `noise_frac_cal` (calibration floor), `noise_dof` (Student-t dof)

### How it works

```python
obs = Observation(
    spectroscopy=SpectroscopyConfig.nirspec_prism(wave_obs, calibration_order=3),
    noise=NoiseConfig(calibration_floor=Uniform(0.01, 0.15)),
)

# obs.get_all_params() returns:
# {
#     'cal_c1': Gaussian(0.0, 0.1),
#     'cal_c2': Gaussian(0.0, 0.1),
#     'cal_c3': Gaussian(0.0, 0.1),
#     'noise_frac_cal': Uniform(0.01, 0.15),
# }

spec = ParamSpec(
    sfh_tsnorm_log_total_mass=10.0, 3),
    met_logzsol=Uniform(-2.0, 0.2),
    redshift=Fixed(2.0),
)

model = Model(spec, ssp, observation=obs)
# model.spec now contains the physical params PLUS cal_c1..c3 + noise_frac_cal
```

### Precedence rules

**User-defined parameters always win.** If you explicitly define a parameter
in `ParamSpec` that the observation would also auto-generate, your definition
takes precedence:

```python
from tengri import Gaussian

spec = ParamSpec(
    sfh_tsnorm_log_total_mass=10.0, 3),
    cal_c1=Gaussian(0.0, 0.5),  # user override: wider prior than default 0.1
    redshift=Fixed(2.0),
)

model = Model(spec, ssp, observation=obs)
# model.spec's cal_c1 uses Gaussian(0, 0.5) -- user's definition wins
# cal_c2, cal_c3 use Gaussian(0, 0.1) -- auto-merged defaults
```

### Inspecting merged params

```python
# Original ParamSpec -- only physical params
print(spec.summary())

# Augmented ParamSpec on the model -- includes observation params
print(model.spec.summary())
```

---

## Summary and Diagnostics

### `obs.summary()`

Every observation has a `.summary()` method following the same convention as
`spec.summary()`, `model.summary()`, `fitter.summary()`, and
`posterior.summary_table()`:

```python
obs = Observation(
    photometry=Photometry.from_names(["jwst_f200w", "jwst_f356w", "jwst_f444w"]),
    spectroscopy=SpectroscopyConfig.nirspec_prism(wave_obs, calibration_order=3),
    noise=NoiseConfig(calibration_floor=Uniform(0.01, 0.15)),
)

print(obs.summary())
```

Output:

```
Observation
--------------------------------------------------
  Photometry : 3 filters: jwst_f200w, jwst_f356w, jwst_f444w
  Spectroscopy : 1830 pixels, R=30-330, cal order=3
  Noise      : cal floor=Uniform(0.01, 0.15) (free)
  Data type  : joint
  N data     : 1833
               (3 phot + 1830 spec)
  Auto params: cal_c1, cal_c2, cal_c3, noise_frac_cal
```

### Capability queries

```python
obs.can_do_photometry    # bool: are filters configured?
obs.can_do_spectroscopy  # bool: is a wavelength grid configured?
obs.is_joint             # bool: both photometry and spectroscopy?
obs.data_type            # str: "photometry", "spectroscopy", or "joint"
```

### Data dimensions

```python
obs.n_data_phot  # int: number of filters (0 if no photometry)
obs.n_data_spec  # int: number of spectral pixels (0 if no spectroscopy)
obs.n_data       # int: total data points (n_data_phot + n_data_spec)
```

---

## Migration from Old API

The old `filters=` and `data_type=` arguments continue to work unchanged. The
`Observation` API is additive -- nothing is removed.

### Side-by-side comparison

**Photometry-only fitting:**

```python
# OLD
from tengri import Model, Fitter, load_filter_set
filters = load_filter_set(["sdss_r", "sdss_i", "sdss_z"])
model = Model(spec, ssp, filters=filters)
fitter = Fitter(model, flux_obs, noise_obs, data_type="photometry")

# NEW
from tengri import Model, Fitter, Observation, Photometry
obs = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i", "sdss_z"]))
model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, flux_obs, noise_obs)  # data_type inferred
```

**Spectroscopy with calibration:**

```python
# OLD -- noise and calibration params scattered in ParamSpec
spec = ParamSpec(
    ...,
    cal_c1=Gaussian(0, 0.1),
    cal_c2=Gaussian(0, 0.1),
    noise_frac_cal=Uniform(0.01, 0.15),
)
model = Model(spec, ssp)
model.precompute_spectroscopy(wave_obs)  # manual, easy to forget
fitter = Fitter(model, spec_flux, spec_noise, data_type="spectroscopy")

# NEW -- observation declares everything, auto-merge handles params
obs = Observation(
    spectroscopy=SpectroscopyConfig.nirspec_prism(wave_obs, calibration_order=2),
    noise=NoiseConfig(calibration_floor=Uniform(0.01, 0.15)),
)
model = Model(spec, ssp, observation=obs)  # auto-precomputes, auto-merges
fitter = Fitter(model, spec_flux, spec_noise)
```

**Joint fitting:**

```python
# OLD -- manual concatenation, no validation
data = jnp.concatenate([phot_flux, spec_flux])
noise = jnp.concatenate([phot_noise, spec_noise])
fitter = Fitter(model, data, noise, data_type="joint")

# NEW -- validated packing
data = obs.pack_data(phot=phot_flux, spec=spec_flux)
noise = obs.pack_data(phot=phot_noise, spec=spec_noise)
fitter = Fitter(model, data, noise)
```

**Important:** You cannot specify both `filters=` and `observation=` when
creating a `Model` -- this raises `ValueError`.

---

## Complete Examples

### Example 1: SDSS 5-band photometry

```python
import jax
import jax.numpy as jnp
from tengri import (
    Model, ParamSpec, Fitter, Observation, Photometry,
    Uniform, Fixed, load_ssp_data,
)

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")

obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
)

spec = ParamSpec(
    sfh_tsnorm_log_total_mass=10.0, 3),
    sfh_tsnorm_tau_rise_gyr=Uniform(0.1, 5.0),
    sfh_tsnorm_tau_fall_gyr=Uniform(0.1, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 3.0),
    redshift=Fixed(0.1),
)

model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, flux_obs, noise_obs)
posterior = fitter.run("native_geovi")
print(posterior.summary_table())
```

### Example 2: JWST NIRSpec PRISM spectroscopy

```python
import jax.numpy as jnp
from tengri import (
    Model, ParamSpec, Fitter, Observation, SpectroscopyConfig, NoiseConfig,
    Uniform, Fixed, load_ssp_data,
)

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")

obs = Observation(
    spectroscopy=SpectroscopyConfig.nirspec_prism(
        wave_obs,                    # observed wavelength grid (Angstrom)
        calibration_order=3,         # 3rd-order Chebyshev calibration
        eline_marginalize=True,      # marginalize emission line amplitudes
    ),
    noise=NoiseConfig(
        calibration_floor=Uniform(0.01, 0.15),  # free calibration floor
    ),
)

# Physical params only -- cal_c1..c3 and noise_frac_cal auto-merged
spec = ParamSpec(
    sfh_field_psd_sigma=Uniform(0.1, 3.0),
    sfh_field_psd_tau_myr=Uniform(10, 5000),
    sfh_field_xi=Uniform(-3, 3),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 3.0),
    redshift=Fixed(2.0),
)

model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, spec_flux, spec_noise)
posterior = fitter.run("raytrace", n_steps=2000, step_size=0.05)

# Posterior includes physical + observation params
print(posterior.summary_table())
```

### Example 3: JWST photometry + spectroscopy joint fit

```python
import jax.numpy as jnp
from tengri import (
    Model, ParamSpec, Fitter, Observation, Photometry,
    SpectroscopyConfig, NoiseConfig,
    Uniform, Fixed, load_ssp_data,
)

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")

obs = Observation(
    photometry=Photometry.from_names([
        "jwst_f090w", "jwst_f150w", "jwst_f200w",
        "jwst_f277w", "jwst_f356w", "jwst_f444w",
    ]),
    spectroscopy=SpectroscopyConfig.nirspec_prism(
        wave_obs,
        calibration_order=3,
    ),
    noise=NoiseConfig(calibration_floor=0.02),  # fixed 2% floor
)

spec = ParamSpec(
    sfh_field_psd_sigma=Uniform(0.1, 3.0),
    sfh_field_psd_tau_myr=Uniform(10, 5000),
    sfh_field_xi=Uniform(-3, 3),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 3.0),
    redshift=Fixed(3.5),
)

model = Model(spec, ssp, observation=obs)

# Validated packing -- canonical order: [photometry, spectroscopy]
data = obs.pack_data(phot=phot_flux, spec=spec_flux)
noise = obs.pack_data(phot=phot_noise, spec=spec_noise)

fitter = Fitter(model, data, noise)  # data_type="joint" inferred
posterior = fitter.run("native_geovi")

# Unpack prediction for plotting
pred = model.predict(posterior.median_params)
result = obs.unpack_prediction(pred)
phot_pred = result["photometry"]    # shape (6,)
spec_pred = result["spectroscopy"]  # shape (n_pixels,)
```

### Example 4: Mock generation and recovery

```python
import jax
from tengri import (
    Model, ParamSpec, Fitter, Observation, Photometry,
    Uniform, Fixed, load_ssp_data, generate_mock,
)

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")

obs = Observation(
    photometry=Photometry.from_names(["sdss_r", "sdss_i", "sdss_z"]),
)

spec = ParamSpec(
    sfh_tsnorm_log_total_mass=10.0, 3),
    sfh_tsnorm_tau_rise_gyr=Uniform(0.1, 5.0),
    sfh_tsnorm_tau_fall_gyr=Uniform(0.1, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 3.0),
    redshift=Fixed(1.0),
)

model = Model(spec, ssp, observation=obs)

# Generate mock data -- observation tells generate_mock what to produce
key = jax.random.PRNGKey(42)
mock = generate_mock(key, model, snr=20.0)
# mock.flux_obs shape: (3,) -- matches obs.n_data_phot

# Fit the mock
fitter = Fitter(model, mock.flux_obs, mock.noise)
posterior = fitter.run("native_geovi")
print(posterior.summary_table())
```

### Example 5: Batch fitting

```python
from tengri import (
    Model, ParamSpec, Fitter, Observation, Photometry,
    Uniform, Fixed, load_ssp_data,
)

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")

obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
)

spec = ParamSpec(
    sfh_field_psd_sigma=Uniform(0.1, 3.0),
    sfh_field_psd_tau_myr=Uniform(10, 5000),
    sfh_field_xi=Uniform(-3, 3),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 3.0),
    redshift=Fixed(0.5),
)

model = Model(spec, ssp, observation=obs)

# catalog_flux, catalog_noise: shape (n_galaxies, 5)
fitter = Fitter(model, catalog_flux, catalog_noise)
results = fitter.fit_batch(catalog_flux, catalog_noise)
```

### Example 6: Custom ground-based R=5000 spectrograph

```python
import jax.numpy as jnp
from tengri import (
    Model, ParamSpec, Fitter, Observation, SpectroscopyConfig, NoiseConfig,
    Uniform, Fixed, load_ssp_data,
)

ssp = load_ssp_data("data/ssp_fsps_v3.2.h5")

# Custom high-resolution spectrograph
wave_obs = jnp.linspace(3800, 7000, 5000)  # optical range, Angstrom
obs = Observation(
    spectroscopy=SpectroscopyConfig.constant_r(
        wave_obs,
        R=5000,
        sigma_lib_kms=70.0,      # MILES library resolution
        calibration_order=5,      # 5th-order calibration for ground-based data
        eline_marginalize=True,
        lsf_n_bins=32,            # more bins for high-R data
    ),
    noise=NoiseConfig(
        calibration_floor=Uniform(0.01, 0.10),
        student_t_dof=5.0,        # robust to outliers (cosmic rays, sky residuals)
    ),
)

spec = ParamSpec(
    sfh_tsnorm_log_total_mass=10.0, 3),
    sfh_tsnorm_tau_rise_gyr=Uniform(0.1, 5.0),
    sfh_tsnorm_tau_fall_gyr=Uniform(0.1, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 3.0),
    redshift=Fixed(0.05),
)

model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, spec_flux, spec_noise)
posterior = fitter.run("native_geovi")

print(obs.summary())
print(posterior.summary_table())
```

---

## API Reference

### `Observation`

Frozen dataclass. Composes photometric, spectroscopic, and noise configurations.

| Parameter      | Type                          | Default | Description                                      |
|---------------|-------------------------------|---------|--------------------------------------------------|
| `photometry`  | `Photometry` or `None`        | `None`  | Photometric filter configuration                 |
| `spectroscopy`| `SpectroscopyConfig` or `None`| `None`  | Spectroscopic instrument configuration           |
| `noise`       | `NoiseConfig` or `None`       | `None`  | Noise model configuration                        |

At least one of `photometry` or `spectroscopy` must be provided.

**Properties:**

| Property              | Type   | Description                                    |
|-----------------------|--------|------------------------------------------------|
| `can_do_photometry`   | `bool` | Whether photometric filters are configured     |
| `can_do_spectroscopy` | `bool` | Whether a spectroscopic wavelength grid is set  |
| `is_joint`            | `bool` | Whether both photometry and spectroscopy are set|
| `data_type`           | `str`  | `"photometry"`, `"spectroscopy"`, or `"joint"` |
| `n_data_phot`         | `int`  | Number of photometric data points (filters)    |
| `n_data_spec`         | `int`  | Number of spectroscopic data points (pixels)   |
| `n_data`              | `int`  | Total data points (`n_data_phot + n_data_spec`)|

**Methods:**

| Method                                   | Returns              | Description                                           |
|------------------------------------------|----------------------|-------------------------------------------------------|
| `pack_data(phot=None, spec=None)`        | `jnp.ndarray`       | Concatenate phot + spec in canonical order with shape validation |
| `unpack_prediction(predicted)`           | `dict[str, ndarray]` | Split packed array into `{"photometry": ..., "spectroscopy": ...}` |
| `get_all_params()`                       | `dict[str, Distribution]` | Collect all observation-driven parameters          |
| `summary()`                              | `str`                | Human-readable multi-line summary                     |

---

### `Photometry`

Frozen dataclass. Wraps filter transmission curves.

| Parameter | Type                      | Description                        |
|-----------|---------------------------|------------------------------------|
| `filters` | `tuple[FilterCurve, ...]` | Filter transmission curves         |
| `names`   | `tuple[str, ...]`         | Human-readable filter names        |

**Derived fields** (set automatically in `__post_init__`):

| Field          | Type                       | Description                     |
|---------------|----------------------------|---------------------------------|
| `filter_waves` | `tuple[jnp.ndarray, ...]` | Wavelength arrays per filter    |
| `filter_trans` | `tuple[jnp.ndarray, ...]` | Transmission arrays per filter  |
| `n_filters`    | `int`                      | Number of filters               |

**Factory methods:**

| Method                                     | Description                                       |
|-------------------------------------------|---------------------------------------------------|
| `Photometry.from_names(names, cache_dir=)` | Load filters by registry short name (e.g. `"sdss_r"`) |
| `Photometry.from_filter_set(filter_set)`   | Create from legacy `load_filter_set()` output or `FilterCurve` list |

**Other methods:**

| Method      | Returns | Description                       |
|-------------|---------|-----------------------------------|
| `summary()` | `str`   | One-line summary (e.g. `"5 filters: sdss_u, sdss_g, ..."`) |

---

### `SpectroscopyConfig`

Frozen dataclass. Declares the spectroscopic instrument.

| Parameter            | Type                              | Default  | Description                                             |
|---------------------|-----------------------------------|----------|---------------------------------------------------------|
| `wave_obs`          | `jnp.ndarray`                     | required | Observed-frame wavelength grid (Angstrom)               |
| `resolution`        | `float`, `jnp.ndarray`, or `None` | `None`   | Spectral resolution R. Scalar, per-pixel array, or None |
| `sigma_lib_kms`     | `float`                           | `70.0`   | SSP library velocity dispersion (km/s)                  |
| `lsf_n_bins`        | `int`                             | `16`     | Bins for piecewise LSF approximation                    |
| `calibration_order` | `int`                             | `0`      | Chebyshev calibration polynomial order (0 = disabled)   |
| `eline_marginalize` | `bool`                            | `False`  | Analytically marginalize emission line amplitudes       |
| `eline_wavelengths` | `jnp.ndarray` or `None`           | `None`   | Custom emission line wavelengths (rest-frame Angstrom)  |
| `eline_prior_sigma` | `float`                           | `100.0`  | Prior width on emission line amplitudes                 |

**Properties:**

| Property          | Type   | Description                               |
|-------------------|--------|-------------------------------------------|
| `n_pixels`        | `int`  | Number of spectral pixels                 |
| `has_lsf`         | `bool` | Whether LSF convolution is configured     |
| `has_calibration`  | `bool` | Whether calibration polynomial is active  |

**Factory methods:**

| Method                                    | Description                                    |
|------------------------------------------|------------------------------------------------|
| `SpectroscopyConfig.nirspec_prism(wave_obs, **kw)` | JWST NIRSpec PRISM (variable R ~ 30--330)  |
| `SpectroscopyConfig.nirspec_g140m(wave_obs, **kw)` | JWST NIRSpec G140M (R ~ 1000)              |
| `SpectroscopyConfig.constant_r(wave_obs, R, **kw)` | Constant-R spectrograph                     |

All factory methods accept `sigma_lib_kms`, `calibration_order`,
`eline_marginalize`, and additional `**kwargs` passed through to the constructor.

**Other methods:**

| Method                     | Returns                    | Description                                    |
|---------------------------|----------------------------|------------------------------------------------|
| `get_calibration_params()` | `dict[str, Distribution]` | `cal_c1`..`cal_cN` with `Gaussian(0, 0.1)` priors |
| `summary()`                | `str`                      | One-line summary                               |

---

### `NoiseConfig`

Frozen dataclass. Configures the noise model.

| Parameter           | Type                      | Default | Description                                         |
|--------------------|---------------------------|---------|-----------------------------------------------------|
| `calibration_floor` | `float` or `Distribution` | `0.0`   | Fractional calibration floor. Float = fixed, Distribution = free param |
| `student_t_dof`     | `float` or `None`         | `None`  | Student-t degrees of freedom. None = Gaussian likelihood |

**Methods:**

| Method         | Returns                    | Description                                         |
|---------------|----------------------------|-----------------------------------------------------|
| `get_params()` | `dict[str, Distribution]` | Parameter entries: `noise_frac_cal` and/or `noise_dof` |

**Parameter generation rules:**

| Input                                      | Generated parameter                     |
|-------------------------------------------|-----------------------------------------|
| `calibration_floor=0.0`                   | (none)                                  |
| `calibration_floor=0.02`                  | `noise_frac_cal = Fixed(0.02)`          |
| `calibration_floor=Uniform(0.01, 0.15)`  | `noise_frac_cal = Uniform(0.01, 0.15)` |
| `student_t_dof=5.0`                       | `noise_dof = Fixed(5.0)`               |
| `student_t_dof=None`                      | (none)                                  |
