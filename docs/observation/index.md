# Observation Guide

The `Observation` class bundles photometric, spectroscopic, and noise configuration
into a single declarative object that drives the entire fitting pipeline. Instead of
passing `filters=`, `data_type=`, and noise parameters separately, you declare
**what you observed** once and let tengri handle the rest.

```python
from tengri import Observation, Photometry, Spectroscopy, NoiseModel
```

## Why Observation?

- **One object, full configuration.** Photometry, spectroscopy, noise, and calibration
  are declared together. No more remembering which arguments go where.
- **Automatic parameter injection.** Calibration polynomial coefficients and noise model
  parameters are generated from the Observation and merged into `Parameters` automatically.
- **Safe data handling.** `pack_data()` and `unpack_prediction()` validate shapes and
  enforce canonical ordering (photometry first, spectroscopy second).
- **Data type inference.** The `Fitter` reads `obs.data_type` directly --- no need for
  `data_type="photometry"` or `data_type="joint"`.

## API at a glance

`Observation` is a frozen dataclass with three optional fields:

| Field | Type | Purpose |
|-------|------|---------|
| `photometry` | `Photometry` | Filter transmission curves |
| `spectroscopy` | `Spectroscopy` | Wavelength grid, resolution, calibration |
| `noise` | `NoiseModel` | Calibration floor, Student-t likelihood |

At least one of `photometry` or `spectroscopy` must be provided.

### Key properties

| Property | Returns |
|----------|---------|
| `obs.can_do_photometry` | `True` if filters are configured |
| `obs.can_do_spectroscopy` | `True` if a wavelength grid is configured |
| `obs.is_joint` | `True` if both are configured |
| `obs.data_type` | `"photometry"`, `"spectroscopy"`, or `"joint"` |
| `obs.n_data` | Total number of data points |

### Key methods

| Method | Purpose |
|--------|---------|
| `obs.pack_data(phot=, spec=)` | Concatenate data arrays with shape validation |
| `obs.unpack_prediction(pred)` | Split prediction back into `{"photometry": ..., "spectroscopy": ...}` |
| `obs.get_all_params()` | Collect all auto-generated parameters |
| `obs.summary()` | Human-readable configuration summary |

```{toctree}
:maxdepth: 1

quickstart
photometry
spectroscopy
noise
joint_fitting
migration
```
