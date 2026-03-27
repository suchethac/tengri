# Batch Fitting

Fitting catalogs of galaxies efficiently. This page covers the `fit_batch` API,
mock catalog generation, result aggregation, and practical considerations for
real survey data.

## The batch API

`fit_batch` fits a list of galaxies sequentially, sharing the XLA compilation
cache so that only the first galaxy pays the compile cost:

```python
from tengri import Model, ParamSpec, Fitter

model = Model(spec, ssp)
fitter = Fitter(model, flux_obs, noise)  # template fitter

galaxies = [
    {"flux_obs": flux_array_i, "noise": noise_array_i}
    for flux_array_i, noise_array_i in zip(catalog_flux, catalog_noise)
]

results = fitter.fit_batch(galaxies, method="native_geovi", n_iterations=15)
```

**Key points:**
- The first galaxy takes ~15s (XLA compilation). Subsequent galaxies take ~2ms each
  with `native_geovi` thanks to the persistent XLA cache at `~/.cache/tengri_jax_cache`.
- Any inference method works: `native_geovi` (default), `raytrace`, `nuts`, etc.
- For `native_*` methods, `n_seeds=5` is set automatically to improve robustness.

### Parameters

```python
results = fitter.fit_batch(
    galaxies,                   # list of {"flux_obs": ..., "noise": ...}
    method="native_geovi",      # any method from fitter.run()
    key=jax.random.PRNGKey(42), # reproducibility
    verbose=True,               # progress printing
    n_iterations=15,            # passed through to run()
)
```

### Return value

A list of `Posterior` objects, one per galaxy. Each has `.params`, `.samples`,
`.diagnostics`, `.summary_table()`, and `.plot_corner()`.

## Generating mock catalogs

Use `generate_mock` to create synthetic photometry for testing:

```python
from tengri import generate_mock
import jax
import jax.numpy as jnp

# Draw parameters from the prior
key = jax.random.PRNGKey(0)
n_galaxies = 100

catalog = []
for i in range(n_galaxies):
    key, subkey = jax.random.split(key)
    params_i = spec.sample(subkey)
    mock_i = generate_mock(model, params_i, key=subkey, snr=20.0)
    catalog.append(mock_i)

# mock_i contains: "flux_true", "flux_obs", "noise", "params"
```

Then fit the catalog:

```python
galaxies = [
    {"flux_obs": m["flux_obs"], "noise": m["noise"]}
    for m in catalog
]
results = fitter.fit_batch(galaxies)
```

## Result aggregation

### Summary table

```python
import numpy as np

for i, result in enumerate(results):
    table = result.summary_table()
    # table has columns: param, median, lo_68, hi_68, ESS
    print(f"Galaxy {i}: chi2/dof = {result.diagnostics['chi2_dof']:.2f}")
```

### Extracting parameter arrays

```python
# Collect posterior medians across the catalog
medians = {
    param: np.array([
        np.median(r.samples[param]) for r in results
    ])
    for param in results[0].samples.keys()
    if not param.startswith("psd_xi")
}

# Now medians["sfh_dpl_alpha"] is shape (n_galaxies,)
```

### Convergence across the catalog

```python
from _plot_style import convergence_check

n_converged = sum(
    convergence_check(r, verbose=False)["converged"]
    for r in results
)
print(f"{n_converged}/{len(results)} galaxies converged")
```

:::{tip}
For galaxies that fail convergence, try re-fitting with more iterations, a
different method (e.g., `raytrace`), or MAP initialization:

```python
result_map = fitter_i.run("map", n_steps=1500)
result = fitter_i.run("native_geovi", init_from=result_map, n_iterations=25)
```
:::

## Real data workflows

### SDSS photometry

```python
from tengri import Model, ParamSpec, Uniform, Fitter, Observation, Photometry

spec = ParamSpec(
    redshift=0.05,       # fixed redshift from spectroscopic catalog
    sfh="field",         # stochastic SFH
    dust=True,
)

obs = Observation(photometry=Photometry.from_names(
    ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
))
model = Model(spec, ssp, observation=obs)
fitter = Fitter(model, flux_obs, noise)
results = fitter.fit_batch(sdss_galaxies)
```

### JWST high-redshift considerations

High-redshift JWST fitting introduces specific challenges:

- **IGM absorption**: automatically applied when redshift > 0. Uses
  `igm_transmission(wave_obs, z)` with observed-frame wavelengths.
- **Fewer rest-frame constraints**: at z > 6, optical rest-frame shifts to
  mid-IR. Fewer filters constrain the SFH, so priors matter more.
- **Wider priors**: young galaxies may need wider metallicity and dust priors.
- **Nebular emission**: strong at high-z. Enable with `nebular=True` in `ParamSpec`.

```python
spec = ParamSpec(
    redshift=7.5,
    sfh="field",
    dust=True,
    nebular=True,
)

obs = Observation(photometry=Photometry.from_names(
    ["jwst_f115w", "jwst_f150w", "jwst_f200w",
     "jwst_f277w", "jwst_f356w", "jwst_f444w"]
))
model = Model(spec, ssp, observation=obs)
```

:::{note}
Photometry precomputation auto-activates when redshift is fixed and filters are
present, giving a ~21.6x speedup. This makes batch fitting of fixed-redshift
catalogs particularly efficient.
:::

## Performance

| Catalog size | Method | First galaxy | Subsequent | Total (approx) |
|-------------|--------|-------------|-----------|----------------|
| 100 | native_geovi | ~15s | ~2ms | ~15s |
| 1000 | native_geovi | ~15s | ~2ms | ~17s |
| 100 | raytrace | ~15s | ~10s | ~17 min |

The XLA compilation cache persists across Python sessions (stored at
`~/.cache/tengri_jax_cache`), so restarting the kernel does not re-trigger compilation
for the same model configuration.

## When to use batch vs. hierarchical

| Scenario | Use |
|----------|-----|
| Independent fits, no shared parameters | `fitter.fit_batch()` |
| Shared PSD parameters across population | `HierarchicalFitter` |
| Quick catalog exploration | `fit_batch` with `native_geovi` |
| Population-level SFH burstiness constraints | `HierarchicalFitter` |

See {doc}`hierarchical` for population-level inference with shared PSD parameters.

See the demonstration notebooks for worked examples:
{doc}`../demonstrations/index` --- especially notebooks 02 (photometric catalogs),
08 (real data), and 09 (high-z JWST).
