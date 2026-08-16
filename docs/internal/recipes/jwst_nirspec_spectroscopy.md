# Recipe: Fit a JWST NIRSpec spectrum

You have a JWST NIRSpec x1d extracted spectrum and want to fit the stellar mass
and star formation history. Extract the 1D spectrum using the specutils bridge,
then build an Observation manually.

For a photometric equivalent using the Galaxy facade, see [SDSS photometry](sdss_photometry_oneliner.md).

## Extract spectrum from FITS

```python
from tengri.io import from_spectrum1d
from specutils import Spectrum1D

# Read JWST x1d FITS
sp = Spectrum1D.read("my_jwst_x1d.fits", format="JWST x1d")

# Extract arrays
wave, flux, err, meta = from_spectrum1d(sp)

print(f"Wave range: {wave.min():.1f} to {wave.max():.1f} Angstrom")
print(f"Flux shape: {flux.shape}, Signal-to-noise: {(flux / err).mean():.1f}")
```

## Fit the spectrum

`Spectroscopy` carries the wavelength grid and instrument setup; the
flux and error arrays go to `Fitter`:

```python
import tengri
from tengri import (
    SEDModel, ForwardModel, Fitter, Spectroscopy, Observation, recipes, load_ssp_data,
)

spec = Spectroscopy(
    wave_obs=wave,
    resolution=1000.0,             # R of your grating/filter combination
    eline_mode="marginalized",     # analytic emission-line amplitudes
)
obs = Observation(spectroscopy=spec)

ssp = load_ssp_data(tengri.download_ssp())
model = SEDModel.build(ssp_data=ssp, observation=obs,
                       **recipes.star_forming_photometry())

forward = ForwardModel.build(sed=model, observation=obs)
fitter = Fitter(forward, flux, err)
result = fitter.run("map")
print(result.summary_table())
```

The full spectroscopic walkthrough (calibration polynomials, line
diagnostics) is [notebook 06](../spine/06_fitting_spectroscopy); joint
NIRSpec + photometry fits are [notebook 07](../spine/07_joint_photo_spec).
