# Recipe: Fit a JWST NIRSpec spectrum

## Scenario

You have a JWST NIRSpec x1d 1D extracted spectrum and want a stellar mass and star formation history posterior.

## Current state

A one-liner `Galaxy.from_spectrum` constructor is planned but not yet shipped. In the interim, use the `from_spectrum1d` IO primitive to extract wavelength, flux, and error arrays, then build an Observation manually.

## Extract spectrum from x1d file

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

See `notebooks/08_fitting_spectra.py` for a complete example showing how to construct a Spectroscopy observation and fit with stellar continuum and emission-line models.

For joint fits combining NIRSpec with photometry (e.g., MIRI photometry), see `notebooks/14_joint_photometry_spectroscopy.py`.

## Emission-line citations

If your fit includes emission-line features, cite the relevant models:

```python
import tengri as tg
tg.cite("cue")   # Emission-line model
tg.cite("nifty") # If using VI inference
```

## Next steps

- Roadmap for one-liner: `Galaxy.from_spectrum(sp, redshift=..., preset=...)`
- Track progress in GitHub issues

## References

- `from_spectrum1d`: Handles JWST x1d, HST/COS, and optical IFU formats
- JWST data guide: https://jwst-docs.stsci.edu/
