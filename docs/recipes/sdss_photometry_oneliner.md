# Recipe: Fit SDSS photometry with one function call

## Scenario

You have 5-band SDSS photometry (u, g, r, i, z) for a galaxy and want a stellar mass, star formation rate, and dust posterior in under a minute.

## Prerequisites

- tengri installed and up to date
- SSP data cached at the path returned by `tengri.io.load_ssp_data()` (see load_ssp_data documentation for caching options)

## One-liner fit

```python
import tengri as tg

galaxy = tg.Galaxy.from_arrays(
    filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
    flux=[1.2, 2.5, 3.1, 2.8, 2.3],          # nanomaggies
    flux_err=[0.1, 0.15, 0.12, 0.14, 0.11],
    flux_unit="nanomaggies",
    redshift=0.1,
    ssp_path=tg.io.load_ssp_data(),
    preset="starforming"
)

posterior = galaxy.fit(backend="map")
print(posterior.summary())
```

## Key parameters

- **`flux_unit`:** accepts `"maggies"`, `"nanomaggies"`, `"erg/s/cm^2/Angstrom"`, or `"Jy"`
- **`backend`:** use `"map"` for point estimates (fast), `"vi"` for full uncertainty estimates (slower)
- **`preset`:** `"starforming"`, `"quiescent"`, or `"agn"` — sets sensible parameter priors

## Get uncertainties

For posterior samples instead of point estimate:

```python
posterior = galaxy.fit(backend="vi")
samples = posterior.sample(n=5000)
print(f"log M_star: {samples['stellar_mass'].mean():.2f} +/- {samples['stellar_mass'].std():.2f}")
```

## Cite dependencies

At the end of your analysis:

```python
galaxy.cite()  # Prints tengri, DSPS, MIST citations
```

## Runnable example

See `notebooks/00_one_liner.py` for a full executable version with example data.
