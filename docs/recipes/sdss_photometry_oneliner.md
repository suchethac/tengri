# Recipe: Fit SDSS photometry with one function call

You have 5-band SDSS photometry (u, g, r, i, z) for a galaxy and want to fit
the stellar mass, star formation rate, and dust content with a single function
call. The Galaxy facade wraps the full model-building pipeline into one line.

For a spectroscopic equivalent, see [JWST NIRSpec](jwst_nirspec_spectroscopy.md).

```python
import tengri
from tengri import load_ssp_data

ssp = load_ssp_data(tengri.download_ssp())

galaxy = tengri.Galaxy.from_arrays(
    filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    flux=[1.2, 2.5, 3.1, 2.8, 2.3],        # mJy
    flux_err=[0.1, 0.15, 0.12, 0.14, 0.11],
    flux_unit="mJy",
    redshift=0.1,
    ssp=ssp,
    preset="starforming"
)

galaxy = galaxy.fit(backend="map")
print(galaxy.summary())
```

**Parameters:**

- `flux_unit`: one of `"erg/s/cm2/Hz"` (default), `"nJy"`, `"mJy"`, `"Jy"`, `"uJy"`, `"maggies"`
- `redshift`: redshift of the galaxy (or leave None for photo-z mode)
- `preset`: one of `"starforming"`, `"quiescent"`, `"high_z"`, `"photoz"`, `"jwst_spec"`, `"agn_host"`
- `backend`: inference method — `"map"` (point estimate, fast), `"vi"` (full posterior, slower),
  `"mcmc_nuts"`, `"laplace"`, etc. See `tengri.list_inference_methods()` for all available.

**Get credible intervals:**

For credible intervals on photometry (where `map` gives only a point estimate),
the fastest option is `laplace` (Laplace approximation around the MAP):

```python
galaxy = galaxy.fit(backend="laplace")
print(galaxy.summary())    # {param}_median, {param}_lo68, {param}_hi68
```

Laplace is cheap (warm ~1–2 s, ~3 GB) and works well on small photometry problems.
For full posterior samples (VI or MCMC), note that `backend="vi"` is memory-heavy
(cold ~100 s, ~20 GB RSS on D=6–7 photometry) and intended for high-dimensional
problems. For most photometry fits, `laplace` is the practical choice.

**Bibliography:**

Every component, grid, and sampler that ran carries its citation:

```python
print(galaxy.bibliography.report())     # human-readable list
print(galaxy.bibliography.to_bibtex())  # BibTeX for the paper
```

See the [spine notebooks](../spine/00_quickstart) for full end-to-end examples.
