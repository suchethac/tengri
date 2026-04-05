# Joint Fitting

When both photometry and spectroscopy are available for a galaxy, tengri fits them
simultaneously. The `Observation` class handles the bookkeeping: data concatenation,
shape validation, and prediction splitting.

## Creating a joint observation

Pass both `photometry` and `spectroscopy` to `Observation`:

```python
import jax.numpy as jnp
from tengri import Observation, Photometry, Spectroscopy, NoiseModel, Uniform

wave_obs = jnp.linspace(6000, 25000, 500)

obs = Observation(
    photometry=Photometry.from_names(["jwst_f200w", "jwst_f356w", "jwst_f444w"]),
    spectroscopy=Spectroscopy.nirspec_prism(wave_obs, calibration_order=3),
    noise=NoiseModel(calibration_floor=Uniform(0.01, 0.10)),
)

print(obs.is_joint)     # True
print(obs.data_type)    # "joint"
print(obs.n_data)       # 503 (3 phot + 500 spec)
```

## Packing data

Use `pack_data()` to concatenate photometric and spectroscopic data in the canonical
order (photometry first, then spectroscopy). Shape validation is performed
automatically.

```python
data = obs.pack_data(phot=phot_flux, spec=spec_flux)
noise = obs.pack_data(phot=phot_noise, spec=spec_noise)

fitter = Fitter(model, data, noise)
```

`pack_data()` raises a `ValueError` if:
- `phot` is missing when photometry is configured
- `spec` is missing when spectroscopy is configured
- Array shapes do not match the observation dimensions

## Unpacking predictions

After the forward model produces a concatenated prediction, use `unpack_prediction()`
to split it back into labeled components:

```python
predicted = model(params)
components = obs.unpack_prediction(predicted)

phot_pred = components["photometry"]   # shape (3,)
spec_pred = components["spectroscopy"] # shape (500,)
```

This is particularly useful for plotting residuals or computing per-component
chi-squared values.

## Data ordering

The canonical order is always:

1. **Photometry** (indices `0` to `n_data_phot - 1`)
2. **Spectroscopy** (indices `n_data_phot` to `n_data - 1`)

Both `pack_data()` and `unpack_prediction()` enforce this ordering. You never need
to manually concatenate or slice arrays.

## Auto-generated parameters

A joint observation may generate several automatic parameters:

| Source | Parameters |
|--------|-----------|
| `Spectroscopy(calibration_order=3)` | `cal_c1`, `cal_c2`, `cal_c3` |
| `NoiseModel(calibration_floor=Uniform(...))` | `noise_frac_cal` |

All auto-generated parameters are visible via:

```python
print(obs.get_all_params())
# {'cal_c1': Gaussian(0, 0.1), 'cal_c2': Gaussian(0, 0.1),
#  'cal_c3': Gaussian(0, 0.1), 'noise_frac_cal': Uniform(0.01, 0.1)}
```

These are merged into `Parameters` automatically when `Model` is constructed.

## Observation summary

Call `obs.summary()` for a human-readable overview:

```
Observation
--------------------------------------------------
  Photometry : 3 filters: jwst_f200w, jwst_f356w, jwst_f444w
  Spectroscopy : 500 pixels, R=30-330, cal order=3
  Noise      : cal floor=Uniform(0.01, 0.1) (free)
  Data type  : joint
  N data     : 503
               (3 phot + 500 spec)
  Auto params: cal_c1, cal_c2, cal_c3, noise_frac_cal
```
