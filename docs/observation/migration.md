# Migration Guide

This page shows how to migrate from the old `filters=` / `data_type=` API to the
new `Observation` pattern.

## Photometry

:::::{grid} 2

::::{grid-item}
**Old API**
```python
from tengri import load_filter_set

filters = load_filter_set(
    ["sdss_u", "sdss_g", "sdss_r"]
)
model = Model(spec, ssp, filters=filters)
fitter = Fitter(
    model, data, noise,
    data_type="photometry",
)
```
::::

::::{grid-item}
**New API**
```python
from tengri import Observation, Photometry

obs = Observation(
    photometry=Photometry.from_names(
        ["sdss_u", "sdss_g", "sdss_r"]
    ),
)
model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, data, noise)
```
::::

:::::

The `Fitter` infers `data_type="photometry"` from the observation automatically.

## Spectroscopy

:::::{grid} 2

::::{grid-item}
**Old API**
```python
model = Model(
    spec, ssp,
    wave_obs=wave_obs,
    resolution=1000,
)
fitter = Fitter(
    model, data, noise,
    data_type="spectroscopy",
)
```
::::

::::{grid-item}
**New API**
```python
obs = Observation(
    spectroscopy=SpectroscopyConfig(
        wave_obs=wave_obs,
        resolution=1000,
    ),
)
model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, data, noise)
```
::::

:::::

## Joint fitting

:::::{grid} 2

::::{grid-item}
**Old API**
```python
import jax.numpy as jnp

model = Model(
    spec, ssp,
    filters=filters,
    wave_obs=wave_obs,
)
data = jnp.concatenate([phot, spec])
noise = jnp.concatenate([phot_n, spec_n])
fitter = Fitter(
    model, data, noise,
    data_type="joint",
)
```
::::

::::{grid-item}
**New API**
```python
obs = Observation(
    photometry=Photometry.from_names([...]),
    spectroscopy=SpectroscopyConfig(wave_obs),
)
data = obs.pack_data(phot=phot, spec=spec)
noise = obs.pack_data(phot=phot_n, spec=spec_n)

model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, data, noise)
```
::::

:::::

The new API validates array shapes and enforces canonical ordering via `pack_data()`.

## Noise model

:::::{grid} 2

::::{grid-item}
**Old API**
```python
spec = ParamSpec(
    ...,
    noise_frac_cal=Uniform(0.01, 0.15),
)
```
::::

::::{grid-item}
**New API**
```python
obs = Observation(
    photometry=Photometry.from_names([...]),
    noise=NoiseConfig(
        calibration_floor=Uniform(0.01, 0.15),
    ),
)
# noise_frac_cal is auto-injected
```
::::

:::::

## Backward compatibility

The old API still works. Both `filters=` and `data_type=` are accepted by `Model`
and `Fitter` respectively. You can migrate incrementally --- the `Observation` pattern
is recommended for new code but not yet required.

:::{note}
If you pass both `observation=` and `filters=` to `Model`, the `observation` takes
precedence and `filters=` is ignored with a warning.
:::
