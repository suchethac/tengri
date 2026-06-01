# Joint photometry + spectroscopy

Fitting broadband photometry and a spectrum together breaks degeneracies that
neither can break alone: photometry anchors the overall SED shape and
normalisation, while absorption-line spectroscopy pins stellar age and
metallicity. [Notebook 07](spine/07_joint_photo_spec) walks through the full
workflow with figures; this page is the reference for the API.

## One observation, both channels

A joint fit is declared by putting both channels in a single `Observation`:

```python
from tengri import Observation, Photometry, Spectroscopy

obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
    spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=2000),
)
model = SEDModel(spec, ssp, observation=obs)
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
from tengri import Fitter

data  = np.concatenate([phot_flux,  spec_flux])
noise = np.concatenate([phot_noise, spec_noise])

fitter = Fitter(
    model, data, noise,
    data_type="joint",
    calibration_marginalize=True,   # analytic flux-calibration marginalization
    cal_n_poly=2,
)
posterior = fitter.run("mcmc_hmc", n_warmup=300, n_samples=600,
                       dense_mass_matrix=False)
```

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

## Speed: no precompute on joint fits yet

`WavePrecomp` accelerates **photometry-only** models; its lookup-table path is
bypassed whenever spectroscopy is present, and the spectroscopy lookup table
(`SpectrumPrecomp`) is a Phase-5 work in progress. A joint fit therefore runs
the exact wave-grid forward pass. See [Known limitations](known_limitations).
