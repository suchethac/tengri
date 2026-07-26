# Batch Fitting

Fitting catalogs of galaxies: `Catalog` (independent fits) and
`fit_population` (hierarchical, shared hyperparameters), mock catalog
generation, result aggregation, and practical considerations for real survey data.

:::{important}
**Catalog vs fit_population — choose the right one:**

| Use case | Method | Returns |
|----------|--------|---------|
| N independent galaxy fits, no shared parameters | `Catalog(forward, table, flux_unit=...).fit()` | `CatalogPosterior` |
| N galaxies jointly, shared PSD / dust prior | `fit_population` (future API, #1319) | `PopulationPosterior` |

`Catalog.fit()` is fast and parallelizable, supporting all inference methods.
The hierarchical `fit_population` API is planned for a future release to learn
population-level burstiness priors from data.
:::

## The batch API

`Catalog` fits a table of galaxies independently, sharing the XLA
compilation cache so only the first galaxy pays the compile cost:

```python
from tengri import SEDModel, ForwardModel, Catalog
import jax

model = SEDModel.build(ssp_data=ssp, observation=obs)
forward = ForwardModel.build(sed=model, observation=obs)

# Table is any dict-like with flux and error columns
cat = Catalog(forward, table, flux_unit="cgs_fnu")

key = jax.random.PRNGKey(0)
result = cat.fit(method="map", key=key)
```

**Key points:**
- First galaxy: ~15s (compilation). Subsequent: ~2ms each (cached).
- Any inference method works: `"map"`, `"vi"`, `"mcmc_raytrace"`, `"mcmc_nuts"`, etc.
- Table can be a dict, pandas DataFrame, or any object supporting `__getitem__` and `len()`.

### Parameters

```python
result = cat.fit(
    method="map",               # inference method
    key=jax.random.PRNGKey(42), # reproducibility
    forward_chunk_size=1,       # K galaxies in parallel (native methods only)
    n_pad=None,                 # pad catalog size for cache sharing
    store=None,                 # "full" (all samples) or "summary" (percentiles only)
    percentiles=None,           # custom percentiles for store="summary"
    reducers=None,              # additional summary statistics
)
```

All other kwargs are forwarded to the inference backend (e.g., `n_warmup`, `n_samples`
for MCMC).

### Return value

A `CatalogPosterior` object containing results for all galaxies. Access individual
posteriors via indexing or iteration: `result[0]`, or `for post in result`.

## Generating mock catalogs

Use `generate_mock` to create synthetic photometry for testing:

```python
from tengri import generate_mock
import jax
import numpy as np

# Draw parameters from the prior
key = jax.random.PRNGKey(0)
n_galaxies = 100

catalog = {"flux": [], "noise": []}
for i in range(n_galaxies):
    key, subkey = jax.random.split(key)
    params_i = model.spec.sample(subkey)
    mock_i = generate_mock(model, params_i, key=subkey, snr=20.0)
    catalog["flux"].append(mock_i["flux_obs"])
    catalog["noise"].append(mock_i["noise"])

# Convert to arrays
catalog["flux"] = np.array(catalog["flux"])
catalog["noise"] = np.array(catalog["noise"])
```

Then fit the catalog:

```python
cat = Catalog(forward, catalog, flux_unit="cgs_fnu")
result = cat.fit(method="map", key=jax.random.PRNGKey(0))
```

## Result aggregation

### Per-galaxy posteriors

```python
# Access individual posterior for galaxy i
post_i = result[i]
print(post_i.summary_table())
# A MAP fit reports 'final_loss' and 'converged'; VI/sampling methods also
# populate 'chi2_dof' (access it with post_i.diagnostics.get("chi2_dof")).
print(f"Galaxy {i}: final_loss = {post_i.diagnostics['final_loss']:.2f}")

# Iterate over all galaxies
for i, post in enumerate(result):
    stellar_mass = post.properties["stellar_mass"]
    print(f"Galaxy {i}: stellar_mass = {np.median(stellar_mass):.2e}")
```

### Catalog properties

```python
# Properties are automatically lifted over the galaxy axis
stellar_masses = result.properties["stellar_mass"]  # shape (n_galaxies,) or (n_galaxies, n_samples)
sfr_values = result.properties["sfr_avg"]

# Get credible intervals for each galaxy
ci_table = result.properties.ci("stellar_mass", level=0.68)  # shape (n_galaxies, 3)
# ci_table[:, 0] = lower, ci_table[:, 1] = median, ci_table[:, 2] = upper
```

### Export to table

```python
# Convert results to a table dict (round-trips through catalog ingest)
table = result.to_table()
# table["stellar_mass"] has shape (n_galaxies,)
# If store="summary", also includes percentile columns:
# table["stellar_mass_p16"], table["stellar_mass_p50"], table["stellar_mass_p84"]
```

### Convergence across the catalog

```python
import numpy as np

n_converged = sum(
    post.diagnostics.get("converged", False)
    for post in result
)
print(f"{n_converged}/{result.n_galaxies} galaxies converged")
```

:::{tip}
For galaxies that fail convergence, refit individually using `forward.fit`:

```python
# Reuse the per-galaxy columns from your own table (cgs_fnu = model-native units)
flux_i = catalog["flux"][i]
noise_i = catalog["noise"][i]

# Refit with MAP first, then VI with MAP initialization
result_map = forward.fit(flux_i, noise_i, method="map", n_steps=1500)
result_vi = forward.fit(flux_i, noise_i, method="vi", init_from=result_map, n_iterations=25)
```
:::

## Real data workflows

### SDSS photometry

```python
from tengri import SEDModel, ForwardModel, Catalog, Fixed, Observation, Photometry

obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))
model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    sfh={'type': 'tsnorm'},  # simple parametric SFH
    redshift=Fixed(0.05),  # fixed redshift from spectroscopic catalog
)

forward = ForwardModel.build(sed=model, observation=obs)
cat = Catalog(forward, sdss_table, flux_unit="cgs_fnu")
result = cat.fit(method="map", key=jax.random.PRNGKey(0))
```

### JWST high-redshift considerations

High-redshift JWST fitting introduces specific challenges:

- **IGM absorption**: automatically applied when redshift > 0. Uses
  `igm_transmission(wave_obs, z)` with observed-frame wavelengths.
- **Fewer rest-frame constraints**: at z > 6, optical rest-frame shifts to
  mid-IR. Fewer filters constrain the SFH, so priors matter more.
- **Wider priors**: young galaxies may need wider metallicity and dust priors.
- **Nebular emission**: strong at high-z. Enable with `nebular=True` in `Parameters`.

```python
from tengri import SEDModel, Fixed

obs = Observation(photometry=Photometry.from_names(
    ["jwst_f115w", "jwst_f150w", "jwst_f200w",
     "jwst_f277w", "jwst_f356w", "jwst_f444w"]
))
model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    sfh={'type': 'field'},
    dust={'all_params': FREE},
    neb={'type': 'cue', 'all_params': FREE},
    redshift=Fixed(7.5),  # fixed high-redshift
)
```

:::{note}
Photometry precomputation auto-activates when redshift is fixed and filters are
present, giving a ~21.6x speedup. This makes batch fitting of fixed-redshift
catalogs particularly efficient.
:::

## Performance

| Catalog size | Method | First galaxy | Subsequent | Total (approx) |
|-------------|--------|-------------|-----------|----------------|
| 100 | vi | ~15s | ~2ms | ~15s |
| 1000 | vi | ~15s | ~2ms | ~17s |
| 100 | mcmc_raytrace | ~15s | ~10s | ~17 min |

The XLA compilation cache persists across Python sessions (stored at
`~/.cache/tengri_jax_cache`), so restarting the kernel does not re-trigger compilation
for the same model configuration.

## When to use Catalog vs. fit_population

| Scenario | Use |
|----------|-----|
| Independent fits, no shared parameters | `Catalog.fit()` |
| Shared PSD / dust prior across population | `fit_population` (future, #1319) |
| Quick catalog exploration | `Catalog.fit(method="map")` |
| Population-level SFH constraints | `fit_population` (future, #1319) |

See {doc}`hierarchical` for population-level inference.

For a single-galaxy version of the workflow, see the
[`05_fitting_photometry`](../spine/05_fitting_photometry) and
[`06_fitting_spectroscopy`](../spine/06_fitting_spectroscopy) spine notebooks.
