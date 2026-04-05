# Spectroscopy

The `Spectroscopy` class (the canonical name; `SpectroscopyConfig` is a deprecated alias) declares the spectroscopic instrument: wavelength grid,
resolution profile, calibration polynomial, and emission-line marginalization settings.

```python
from tengri import Spectroscopy
```

## Basic usage

At minimum, provide an observed-frame wavelength grid:

```python
import jax.numpy as jnp

wave_obs = jnp.linspace(6000, 25000, 500)  # Angstrom
spec_config = Spectroscopy(wave_obs=wave_obs)
```

This creates a spectroscopy configuration with no LSF convolution, no calibration
polynomial, and no emission-line marginalization.

## Instrument factories

For common instruments, use the built-in factory methods that set the correct
wavelength-dependent resolution profile:

```python
# JWST NIRSpec PRISM (R ~ 30-330, Jakobsen+2022)
spec_config = Spectroscopy.nirspec_prism(wave_obs)

# JWST NIRSpec G140M (R ~ 1000)
spec_config = Spectroscopy.nirspec_g140m(wave_obs)

# Any constant-R spectrograph
spec_config = Spectroscopy.constant_r(wave_obs, R=3000)
```

All factories accept additional keyword arguments that are forwarded to the
`Spectroscopy` constructor (e.g. `calibration_order`, `eline_marginalize`):

```python
spec_config = Spectroscopy.nirspec_prism(
    wave_obs,
    calibration_order=3,
    eline_marginalize=True,
)
```

## Resolution

The `resolution` parameter controls Line Spread Function (LSF) convolution and
accepts three forms:

| Form | Meaning |
|------|---------|
| `None` (default) | No LSF convolution |
| `float` (e.g. `3000`) | Constant R across all pixels |
| `jnp.ndarray` | Per-pixel R array, shape `(n_pix,)` |

When resolution is provided, the SSP templates are convolved with the appropriate
LSF during the forward model.

### LSF settings

Two parameters control the LSF convolution:

- **`sigma_lib_kms`** (default: 70.0) --- SSP library velocity dispersion in km/s,
  subtracted in quadrature. The default of 70 km/s matches the MILES stellar library.
- **`lsf_n_bins`** (default: 16) --- Number of bins for the piecewise-constant
  approximation used when R varies with wavelength.

## Calibration polynomial

Real spectra often have multiplicative calibration offsets relative to the model.
Setting `calibration_order > 0` adds a Chebyshev polynomial correction with free
coefficients:

```python
spec_config = Spectroscopy(
    wave_obs=wave_obs,
    resolution=1000,
    calibration_order=3,    # adds cal_c1, cal_c2, cal_c3
)
```

This generates three free parameters (`cal_c1`, `cal_c2`, `cal_c3`), each with a
`Gaussian(0, 0.1)` prior. These are automatically injected into `Parameters` when the
`Observation` is passed to `Model`.

:::{note}
`calibration_order=0` (the default) means no calibration correction. For most
spectroscopic fitting, order 2--3 is recommended.
:::

## Emission line marginalization

When `eline_marginalize=True`, emission line amplitudes are analytically marginalized
during the likelihood computation. This avoids fitting individual line amplitudes
while still accounting for their contribution to the residuals.

```python
spec_config = Spectroscopy(
    wave_obs=wave_obs,
    eline_marginalize=True,
    eline_prior_sigma=100.0,        # prior width (default)
    eline_wavelengths=my_lines,     # optional custom line list
)
```

If `eline_wavelengths` is `None` (default), a standard 13-line list is used
(Balmer series + forbidden lines).

## Full configuration reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `wave_obs` | `jnp.ndarray` | (required) | Observed-frame wavelength grid (Angstrom) |
| `resolution` | `float`, `array`, `None` | `None` | Spectral resolution R |
| `sigma_lib_kms` | `float` | `70.0` | SSP library velocity dispersion (km/s) |
| `lsf_n_bins` | `int` | `16` | Bins for piecewise-constant LSF |
| `calibration_order` | `int` | `0` | Chebyshev calibration polynomial order |
| `eline_marginalize` | `bool` | `False` | Marginalize emission line amplitudes |
| `eline_wavelengths` | `array` or `None` | `None` | Custom rest-frame line wavelengths |
| `eline_prior_sigma` | `float` | `100.0` | Prior width on line amplitudes |

### Properties

| Property | Description |
|----------|-------------|
| `spec_config.n_pixels` | Number of spectral pixels |
| `spec_config.has_lsf` | Whether LSF convolution is active |
| `spec_config.has_calibration` | Whether calibration polynomial is active |
