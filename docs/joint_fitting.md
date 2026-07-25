# Joint photometry + spectroscopy

Fitting broadband photometry and a spectrum together breaks degeneracies that
neither can break alone: photometry anchors the overall SED shape and
normalization, while absorption-line spectroscopy pins stellar age and
metallicity. [Notebook 07](spine/07_joint_photo_spec) walks through the full
workflow with figures; this page is the reference for the API.

## One observation, both channels

A joint fit is declared by putting both channels in a single `Observation`:

```python
from tengri import SEDModel, Observation, Photometry, Spectroscopy, load_ssp_data

# Set up the model as shown in the notebook 07_joint_photo_spec.py
ssp = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
    spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=2000),
)
model = SEDModel.build(ssp_data=ssp, observation=obs)
```

`model.observation.data_type` now reports `"joint"`, and the spectroscopy grid
is carried on the model (`model.wave_obs`).

## Data ordering and the joint likelihood

The joint likelihood is the **sum** of the photometry and spectroscopy
log-probabilities. You hand the `Fitter` the two data vectors concatenated
**photometry first, then spectroscopy** — the same order the model emits them —
and flag `data_type="joint"` so the loss splits them back into channels:

```python
import numpy as np
from tengri import ForwardModel

data  = np.concatenate([phot_flux,  spec_flux])
noise = np.concatenate([phot_noise, spec_noise])

forward = ForwardModel.build(sed=model, observation=obs)
posterior = forward.fit(
    data, noise,
    method="mcmc_hmc",
    data_type="joint",
    calibration_marginalize=True,   # analytic flux-calibration marginalization
    cal_n_poly=2,
    n_warmup=300, n_samples=600,
    dense_mass_matrix=False
)
```

**Prerequisites:** `phot_flux`, `spec_flux`, `phot_noise`, `spec_noise` are your joint photometry and spectroscopy observations. See [notebook 07 (joint_photo_spec)](spine/07_joint_photo_spec.ipynb) for a complete runnable example.

`calibration_marginalize=True` analytically integrates out a low-order
Chebyshev flux-calibration polynomial on the spectroscopy channel at every
likelihood call (Prospector's optimal-calibration trick, Johnson et al. 2021),
so the calibration nuisance never enters the sampled space. It leaves the
photometry channel untouched.

## Method and memory

Joint posteriors are higher-dimensional and heavier to compile than
photometry-only ones. The tutorial uses `mcmc_hmc` with a diagonal mass matrix
and runs one fit per process — see
[Choosing an inference method](method_selection) and
[Known limitations](known_limitations) for the memory rules.

## Speed: precompute on joint fits

A joint photometry + spectroscopy fit can use the precompute fast path. On a
joint observation, **either** opt-in — `approx=WavePrecomp()` or
`approx=SpectrumPrecomp()` — builds **both** LUT families, and the forward pass
projects the photometry and spectroscopy channels together inside one fused,
persistently-cached JIT kernel:

```python
model = SEDModel.build(
    ssp_data=ssp,
    observation=Observation(photometry=phot, spectroscopy=spec),
    approx=WavePrecomp(),   # joint → both photometry + spectrum LUTs
    # ... plus your sfh / dust / neb groups
)
```

The per-pixel continuum LUT does not apply velocity dispersion / LSF
(`SpectrumPrecomp`'s documented low-to-medium-R domain); use `approx=None` for
the exact LSF-convolved spectrum. See [Known limitations](known_limitations).
