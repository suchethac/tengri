# Internals

Implementation details for contributors working on tengri's core subsystems.

## Inference dispatch

The `Fitter.run(method)` method routes to internal dispatch functions based
on the method name:

| Dispatch function | Canonical methods | Old names (deprecated) |
|-------------------|-------------------|------------------------|
| `_run_vi` | `vi` | `geovi`, `vi_nifty`, `nifty_geovi`, `fast_geovi` |
| `_run_vi_linear` | `vi_linear` | `mgvi`, `evi`, `vi_nifty_linear`, `nifty_mgvi`, `fast_mgvi` |
| `_run_nifty_fast_vi` | `vi_nifty_fast` | — |
| `_run_nifty_fast_vi_linear` | `vi_nifty_fast_linear` | — |
| `_run_vi_native` | `vi_native` | `native_geovi` |
| `_run_vi_native_linear` | `vi_native_linear` | `native_mgvi`, `native_evi` |
| `_run_map` | `map` | — |
| `_run_nuts` | `mcmc_nuts` | `nuts` |
| `_run_raytrace` | `mcmc_raytrace` | `raytrace` |
| `_run_nss` | `nss` | `evidence` |
| `_run_laplace` | `laplace` | — |
| `_run_pathfinder` | `pathfinder` | — |
| `_run_elliptical_slice` | `mcmc_ess` | `elliptical_slice` |

**`vi`** is the default. It uses NIFTy's `optimize_kl` for geoVI
with a resample+update schedule and nonlinear posterior draws.
**`vi_nifty_fast`** uses NIFTy's `OptimizeVI.update` in a tight loop
(~35% faster, no logging). **`vi_native`** is a fully JIT'd native JAX
implementation (experimental, supports multi-seed). Ray Tracing
validates exact posteriors. NUTS validates low-dimensional problems. MAP
provides initialization.

### Inference method hierarchy

| Method | Module | Extra dependency | Exact? | Best dimensionality |
|--------|--------|-----------------|--------|---------------------|
| `map` | `inference/map_optimizer.py` | optax | No (point estimate) | Any |
| `vi` | `inference/fitter.py` | nifty8.re | Approximate | Up to ~100,000 |
| `vi_linear` | `inference/fitter.py` | nifty8.re | Approximate | Up to ~1,000,000 |
| `mcmc_raytrace` | `inference/raytrace.py` | -- | Yes | Up to ~300 |
| `mcmc_nuts` | `inference/nuts.py` | blackjax | Yes | Up to ~20 |
| `mcmc_ess` | `inference/elliptical_slice.py` | -- | Yes | Any (Gaussian prior) |
| `laplace` | `inference/laplace.py` | -- | Approximate (Gaussian) | Any |
| `pathfinder` | `inference/pathfinder.py` | blackjax | Approximate | Up to ~100 |
| `nss` | `inference/ns/nss.py` | -- | Yes + evidence | Up to ~30 |

### Removed method names

These old names have been replaced:

- `geovi_nifty` -> `nifty_geovi`
- `mgvi_nifty` -> `nifty_mgvi`
- `geovi_full` -> `nifty_geovi`
- `mgvi_full` -> `nifty_mgvi`
- `fit_catalog` -> `fit_batch`

## Fused kernel architecture

`core/fused_kernels.py` provides JIT kernel factory functions that fuse
multiple operations into a single `@jax.jit` scope:

- Weights computation
- Metallicity interpolation
- Dust attenuation
- `einsum` for CSP assembly

This eliminates intermediate array materializations and reduces memory
traffic. The fused kernels are created at `Model` initialization time and
reused for every forward model evaluation.

## Parameter translation

`core/param_translate.py` maps between public (user-facing) and internal
parameter names, applying unit conversions where needed:

| Public name | Internal name | Conversion |
|-------------|--------------|------------|
| `met_logzsol` | `log_z_abs` | `+ LOG10_ZSUN` offset |
| `dust_tau_bc` | `tau_bc` | direct |
| `dust_tau_diff` | `tau_diff` | direct |
| `dust_slope` | `dust_slope` | direct |
| `sfh_field_psd_sigma` | `psd_sigma` | direct |
| `sfh_field_psd_tau_myr` | `psd_tau_yr` | `* 1e6` (Myr to yr) |

```{note}
`Parameters` free params use full prefixes: `sfh_dpl_alpha`,
`sfh_dpl_log_total_mass`, `sfh_field_psd_sigma`, `sfh_field_xi` -- not
shorthand like `sfh_alpha` or `psd_xi`. Check with `spec.free_params`
and `spec.sample(key).keys()`.
```

### Internal param name changes (historical)

These internal names were renamed in previous refactors:

- `tau_v1` -> `tau_bc`
- `tau_v2` -> `tau_diff`
- `dust_n` -> `dust_slope`
- `sigma_ps` -> `psd_sigma`
- `tau_ps` -> `psd_tau_yr`
- `log_z` -> `log_z_abs`

## JIT compilation strategy

tengri uses several XLA/JIT optimizations:

**Persistent XLA cache.** A compilation cache at `~/.cache/tengri_jax_cache` is
auto-enabled on import. This avoids recompilation across Python sessions for
the same model configuration.

**Compile-once-run-many.** The forward model and its gradient compile once
on the first call. Subsequent calls reuse the compiled XLA graph.

**Mixed precision — withdrawn.** This entry used to read
"`Model(spec, ssp, forward_dtype="float32")` halves memory usage and provides
roughly 1.5x speed with less than 0.1% error". The knob is retired: it casts
nothing, returns bit-identical results to float64, and still enters the compile
signature, so passing it costs an extra compile and buys nothing (#1433). For
float32 today, run under `jax.enable_x64(False)`.

**Precomputed dust age weights.** The sigmoid of `log10(age)` used for
birth-cloud vs diffuse dust is computed once at `Model.__init__`, not per
forward call.

**Photometry precomputation.** When redshift is fixed and filters are
present, SSP fluxes integrated through each filter are precomputed once.
This eliminates the wavelength-level integral from the inner loop, giving a
21.6x gradient speedup. Check with `model._precomp is not None`.

**Spectroscopy precomputation.** SSPs are pre-interpolated to observed
wavelengths when the wavelength grid is fixed.

### Performance benchmarks (MacBook Pro M-series, CPU)

| Operation | Smooth (D=7) | Stochastic (D=137) |
|-----------|-------------|-------------------|
| Forward model | 140 us | 356 us |
| Gradient | 56 us | 63 us |
| `vi` (10 iter) | 56s compile + 0.3s run | 56s compile + 0.8s run |

## Standardized forward model

`inference/standardized.py` contains `StandardizedForwardModel`, which wraps
`Model` to operate in standardized latent space:

1. Maps `xi ~ N(0, I)` to physical parameters via `Distribution.unstandardize()`
2. Computes the correlated field from PSD parameters and the latent vector
3. Calls `Model.predict_photometry()` or `Model.predict_spectrum()`
4. Builds the loss: `H = 1/2 chi^2 + 1/2 xi^T xi`

The `_correlated_field` key in the params dict allows passing a pre-computed
correlated field to `Model`, bypassing the internal GP computation.

## Hierarchical inference

`PopulationFitter` shares PSD hyperparameters across N galaxies while each
galaxy retains its own latent field `xi_i` and physical parameters. Total
dimensionality: `2 + N * (n_grid + n_phys)`.

Three approaches are available:

1. **CorrelatedFieldMaker + `vi`** (default, recommended): PSD
   hyperparameters are part of the generative model.
2. **`vi_linear`**: Same but faster per iteration for very large N.
3. **`mcmc_raytrace`**: Flat vector with MAP initialization per galaxy for small N.

Batch fitting: `fitter.fit_batch(galaxies)` — default method is `vi`.

## Convergence diagnostics

Every inference result must be checked using `convergence_check()` or
`convergence_table()` from `notebooks/_plot_style.py`.

| Diagnostic | Threshold | Applies to |
|-----------|-----------|------------|
| ESS (bulk) | > 100 per param, > 400 total | RT, NUTS |
| Divergences | 0 ideal; > 5% = serious | NUTS only |
| RT acceptance | 30--70% ideal; > 90% = barely moving | RT only |
| NUTS acceptance | ~80% | NUTS only |

Known difficult parameters: `dust_tau_bc`, `dust_tau_diff`, `met_logzsol`
consistently have low ESS due to the age-dust-metallicity degeneracy. This
is a physical limitation, not a sampler bug.

## Key gotchas

### `hash()` overflow with JAX keys

`jax.random.fold_in(key, hash(string))` overflows `uint32`. Use:

```python
abs(hash(x)) % (2**31)
```

### No Model creation inside gradient tape

`Parameters.__init__` with JAX-traced values fails. Always create `Model` and
`Parameters` objects outside differentiable functions.

### JAX Metal (Apple GPU)

JAX Metal is experimental and causes test failures. Use `JAX_PLATFORMS=cpu`
for reliable results. All benchmarks are CPU numbers.

### IGM wavelength frame

`igm_transmission(wave_obs, z)` takes **observed-frame** wavelengths.
The bagpipes equivalent `get_Inoue14_trans(rest_wavs, z)` takes rest-frame.
Convert: `wave_obs = rest_wavs * (1 + z)`.

### DL07 dust emission

The analytic `draine_li2007()` has incorrect PAH/FIR balance vs tabulated
templates. Use `register_dl07_tabulated("data/dl07_templates.npz")` for
production; analytic is a differentiable fallback only.

### Corner plot axes

`fig.axes` returns a flat list. Reshape with:

```python
np.array(axes).reshape(n, n)
```

### Ray Tracing step size

For stochastic models (D~137), use `step_size=0.05, n_leapfrog_steps=50,
n_steps=2000`. There is a sharp viability cliff at `step_size ~ 0.06` where
acceptance drops from ~98% to 0%.

### NIFTy geoVI samples

Use 4--12 samples per KL iteration, not 80 (literature best practice from
Eberle+2025, Roth+2024, Terveer+2026).

### Metallicity grid

The SSP metallicity grid is `log10(Z)` absolute, not `log10(Z/Zsun)`.
CLOUDY grid metallicities are also converted to absolute at load time.
User-facing `neb_logZ_gas` in `Parameters` is `Z/Zsun` (the param_map adds
`LOG10_ZSUN`).

### macOS timeout

The `timeout` command does not exist on macOS. Use Python-level timeouts or
background tasks instead.
