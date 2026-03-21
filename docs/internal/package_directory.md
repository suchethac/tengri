# tengri Package Directory

Complete reference for every module in the `tengri` package. Covers purpose, key exports, and inter-module dependencies.

---

## Public API

The top-level `from tengri import ...` exposes:

| Category | Exports |
|----------|---------|
| Core classes | `Model`, `ParamSpec`, `Prediction`, `MockData` |
| Prediction containers | `SFHQuantities`, `SEDQuantities`, `DerivedQuantities`, `EmissionLines` |
| Distributions | `Uniform`, `Gaussian`, `LogUniform`, `LogNormal`, `StudentT`, `Fixed` |
| Inference | `Fitter`, `Posterior`, `HierarchicalFitter`, `HierarchicalResult`, `VIConfig`, `sample_raytrace` |
| SFH functions | `tsnorm`, `snorm`, `norm`, `lnorm`, `dpl`, `double_powerlaw`, `delayed_tau`, `constant_sfh`, `exponential_sfh`, `delayed_exponential_sfh`, `triweight_burst` |
| SFH registry | `SFH_REGISTRY`, `FIELD_MODEL_REGISTRY`, `resolve_sfh`, `compute_field_gp` |
| GP / PSD | `generate_gp_fourier`, `generate_gp_batch`, `gp_from_xi`, `compute_sqrt_power_drw`, `make_log_age_grid`, `psd_drw`, `drw_variance`, `drw_acf` |
| SPS | `SSPData`, `load_ssp_data`, `effective_metallicity` |
| Dust | `two_component_dust` |
| Observation | `load_filter_set` |
| Noise model | `compute_effective_noise`, `compute_std_inv`, `has_noise_model`, `uses_student_t`, `variable_noise_hamiltonian` |
| Mock generation | `generate_mock` |
| Constants | `AGEMAX_YR` |
| Plotting | `setup_style`, `plot_sfh`, `plot_sfh_comparison`, `plot_sed_fit`, `plot_spectrum_fit`, `plot_corner_comparison`, `safe_corner`, `diagnostics_table`, `COLORS`, `SDSS_WAVE_EFF`, `SPECTRAL_FEATURES` |

---

## `core/` -- Forward Model Engine

Central orchestration layer. Translates user-facing parameters into observables.

### `core/model.py` -- Model class

High-level wrapper that ties together ParamSpec, SSP data, filters, and all physics modules. Entry point for predictions and fitting.

| Key methods | Purpose |
|-------------|---------|
| `predict(params)` | Returns lazy `Prediction` object |
| `predict_photometry(params)` | Broadband fluxes through filters |
| `predict_spectrum(params)` | Pixel-level spectrum at observed wavelengths |
| `predict_sfh_quantities(params)` | JIT-compatible SFH derived quantities (vmappable) |
| `predict_sed_quantities(params)` | JIT-compatible SED derived quantities (vmappable) |
| `mock(key, params, snr)` | Generate mock data with noise |
| `fit(data, noise, method)` | Convenience wrapper delegating to `Fitter` |

Imports from: `param_spec`, `param_translate`, `sed_pipeline`, `fused_kernels`, `prediction`, `noise`, SFH registry, dust, nebular, AGN, IGM, observation, SPS, utils.

### `core/param_spec.py` -- ParamSpec

Defines model parameters, their prior distributions (or fixed values), and physical bounds. Dynamically selects SFH parameters based on `mean_sfh_type`. Used for both mock generation (sampling) and inference (priors).

| Key methods | Purpose |
|-------------|---------|
| `free_params` | Dict of free (non-fixed) parameter distributions |
| `fixed_params` | Dict of fixed parameter values |
| `sample(key)` | Draw one sample from the joint prior |
| `sample_batch(key, n)` | Draw n samples (vmapped) |
| `log_prior(params)` | Joint log-prior density |

Imports from: `distributions`, SFH `registry`.

### `core/prediction.py` -- Prediction, SFHQuantities, SEDQuantities, DerivedQuantities, EmissionLines

Lazy prediction object with on-demand computation of derived quantities. Properties are computed when first accessed; intermediate results are cached.

| Container | Example properties |
|-----------|--------------------|
| `SFHQuantities` | `stellar_mass`, `sfr_100myr`, `mass_weighted_age_gyr`, `ssfr` |
| `SEDQuantities` | `l_bol`, `uv_slope_beta`, `dn4000`, `rest_uv_magnitude` |
| `EmissionLines` | `halpha`, `hbeta`, `bpt_nii`, `oiii_5007` |
| `DerivedQuantities` | Aggregates SFH + SED + lines |

Two usage modes: (1) lazy single-galaxy exploration; (2) JIT + vmap batch computation.

Imports from: `utils/sed_quantities`, `models/radio`, `models/xray`.

### `core/mock.py` -- MockData, generate_mock

Standalone mock photometry generator. Adds Gaussian noise at a given SNR to true model photometry.

- `generate_mock(model, params, key, snr)` -- returns dict with `flux_true`, `noise`, `flux_obs`, `params`.
- `MockData` -- re-exported from `model.py`.

### `core/noise.py` -- Noise model

IFT-style noise model where noise parameters are jointly inferred with the signal. Implements the variable-covariance Gaussian likelihood from NIFTy.

| Key functions | Purpose |
|---------------|---------|
| `compute_effective_noise(sigma_obs, f_cal, model_flux)` | Combine observational and calibration uncertainty |
| `compute_std_inv(sigma_obs, f_cal, model_flux)` | Inverse noise for the VariableCovarianceGaussian |
| `has_noise_model(spec)` | Check if noise_frac_cal is free |
| `uses_student_t(spec)` | Check if Student-t likelihood is active |
| `variable_noise_hamiltonian(...)` | Full noise-model energy: 0.5 * chi2 - sum(log(tau)) |

Imports from: `distributions`.

### `core/param_translate.py` -- Parameter name translation

Maps between public parameter names (e.g., `met_logzsol`, `sfh_field_psd_tau_myr`) and internal names (e.g., `log_z_abs`, `psd_tau_yr`). Handles unit conversions (Myr to yr, relative to absolute metallicity).

| Key exports | Purpose |
|-------------|---------|
| `LOG10_ZSUN` | Solar metallicity offset (-1.8477, Asplund 2009) |
| `translate_params(spec, params)` | Public params dict to internal params dict |

Imports from: SFH `registry`.

### `core/sed_pipeline.py` -- SED computation engine

The computational core: dispatches SFH, interpolates metallicity, applies dust, adds nebular/AGN emission, applies IGM absorption. All functions take a `model` argument (not `self`) so heavy computation lives outside the Model class.

| Key functions | Purpose |
|---------------|---------|
| `interp_metallicity(model, log_z)` | Dispatch metallicity interpolation |
| `compute_sed(model, params)` | Full rest-frame SED computation |
| `compute_photometry(model, params)` | SED through filters |

Imports from: `dust/attenuation`, `sps/dsps_wrapper`.

### `core/fused_kernels.py` -- Fused JIT kernel builders

Factory functions that build `@jax.jit` closures capturing precomputed arrays (SSP grids, dust weights, filter curves). Eliminates intermediate array materializations for maximum speed.

| Key functions | Purpose |
|---------------|---------|
| `is_fused_compatible(model)` | Check if model config supports fused path |
| `build_fused_photometry_kernel(model)` | Single-JIT photometry: weights + Z-interp + dust + einsum |
| `build_fused_spectrum_kernel(model)` | Single-JIT spectroscopy kernel |

Imports from: dust, SPS modules.

---

## `inference/` -- Fitting and Results

### `inference/fitter.py` -- Fitter class

Unified inference entry point. Builds a loss function from Model predictions and ParamSpec priors, then dispatches to the chosen backend.

| Method | Backend | Best for |
|--------|---------|----------|
| `run("map")` | optax (Adam/SGD) | Fast point estimates, initialization |
| `run("raytrace")` | Ray Tracing Sampler | Exact MCMC, stochastic-gradient resilient |
| `run("nuts")` | BlackJAX NUTS | Gold-standard validation |
| `run("geovi")` | NIFTy.re geoVI | Non-Gaussian posteriors, high-D |
| `run("mgvi")` | NIFTy.re MGVI | Fastest VI, very large D |

Imports from: `distributions`, `utils/transforms`, `inference/common`, and dispatches to each backend module.

### `inference/posterior.py` -- Posterior

Stores inference results (samples or point estimates) with summary statistics, diagnostics, and format conversion.

| Key methods | Purpose |
|-------------|---------|
| `summary()` | Print parameter medians and credible intervals |
| `resample(key)` | Draw a single sample from the posterior |
| `to_arviz()` | Convert to ArviZ InferenceData |
| `to_paramspec()` | Convert posterior mean back to a ParamSpec |

### `inference/hierarchical.py` -- HierarchicalFitter, HierarchicalResult

Shares PSD hyperparameters (sigma_PSD, tau_PSD) across N galaxies while each galaxy retains its own latent field and physical parameters. Supports geoVI and MGVI backends.

| Key exports | Purpose |
|-------------|---------|
| `HierarchicalFitter` | Fit N galaxies jointly with shared PSD |
| `HierarchicalResult` | Stores shared + per-galaxy posterior samples |

Imports from: `utils/transforms`.

### `inference/raytrace.py` -- Ray Tracing Sampler

Physics-inspired MCMC (Behroozi 2025). Propagates light rays through a medium where refractive index encodes the posterior. Orders of magnitude more resilient to stochastic gradients than HMC/NUTS.

| Key exports | Purpose |
|-------------|---------|
| `sample_raytrace(key, x_init, log_prob_fn, ...)` | Run the Ray Tracing sampler |

Licensed under Apache 2.0. Original HMC by Martin Marek; Ray Tracing extensions by Peter Behroozi.

### `inference/map_optimizer.py` -- MAP optimization

Gradient-based point estimation via optax. No uncertainty quantification; used for initialization and sanity checks.

| Key exports | Purpose |
|-------------|---------|
| `fit_map(forward_model, data, noise, ...)` | Run MAP optimization |

Imports from: `inference/common`.

### `inference/nuts.py` -- NUTS sampler

Full Bayesian posterior via BlackJAX NUTS. Supports multi-chain parallelization.

| Key exports | Purpose |
|-------------|---------|
| `fit_nuts(forward_model, data, noise, ...)` | Run NUTS sampling |

Imports from: `inference/common`.

### `inference/geovi.py` -- Geometric Variational Inference

Primary VI method. Uses the Fisher metric to find coordinates where the posterior is approximately Gaussian, then draws samples in that space.

| Key exports | Purpose |
|-------------|---------|
| `fit_geovi(forward_model, data, noise, ...)` | Run geoVI optimization |

Imports from: `inference/common`.

### `inference/geovi_nuts.py` -- geoVI-preconditioned NUTS

Hybrid method: runs geoVI to find a coordinate transform, then runs NUTS in the flattened space with an identity mass matrix. Combines geoVI's geometric insight with NUTS's exactness guarantee.

| Key exports | Purpose |
|-------------|---------|
| `GeoVITransform` | Frozen coordinate transform state |
| `geovi_preconditioned_nuts(...)` | Run NUTS in geoVI-flattened coordinates |

### `inference/standardized.py` -- StandardizedForwardModel

Maps standardized latents xi ~ N(0, I) to predicted observables, absorbing ALL prior structure into the forward model. The loss is always H(xi) = 0.5 * chi2 + 0.5 * xi^T xi. Unifies individual and hierarchical inference.

| Key exports | Purpose |
|-------------|---------|
| `StandardizedForwardModel` | Wraps Model with prior-absorbing transforms |
| `build_standardized_loss(smodel, data, noise)` | Build the standardized loss function |

Imports from: `models/sfh/gp_sfh`, `utils/grid`.

### `inference/common.py` -- Shared inference utilities

Containers and helper functions used by all inference backends.

| Key exports | Purpose |
|-------------|---------|
| `InferenceResult` | NamedTuple: params, samples, loss_history, wall_time, method, diagnostics |
| `PriorConfig` | NamedTuple: prior bounds and distributions |
| `build_loss_fn(...)` | Construct chi-squared + prior loss |
| `initialize_params(...)` | Sample or set initial parameter values |
| `unbounded_to_physical(...)` | Transform optimizer space to physical space |
| `DEFAULT_PRIOR` | Default prior configuration |

### `inference/vi_config.py` -- VIConfig

Configuration dataclass for NIFTy variational inference (geoVI / MGVI / EVI). Stores optimizer kwargs following Philipp Frank's recommendations.

| Key exports | Purpose |
|-------------|---------|
| `VIConfig` | Frozen dataclass with draw_linear, nonlinearly_update, kl kwargs |

---

## `models/` -- Physics Modules

### `models/sfh/` -- Star Formation History

#### `models/sfh/registry.py` -- SFH registry and composition

Registry-driven SFH dispatch. Parametric models are registered by name and composed via `mean_sfh_type` lists. Three composition modes: additive (smooth models summed), mixture (mass-fraction burst), modulator (multiplicative GP field).

| Key exports | Purpose |
|-------------|---------|
| `SFH_REGISTRY` | Dict mapping name to registered SFH model |
| `FIELD_MODEL_REGISTRY` | Dict mapping name to GP field model |
| `resolve_sfh(sfh_type)` | Returns composed (fn, params, param_map, settings) |
| `compute_field_gp(...)` | Compute GP realization from PSD params and xi |

Imports from: `mean_sfh`, `gp_sfh`, `psd_models`, `distributions`.

#### `models/sfh/mean_sfh.py` -- Parametric mean SFH shapes

Smooth parametric SFH components. The GP has zero mean, so these define the overall SFH envelope. Full SFH = mean(t) * exp(GP(t) - K(0)/2).

| Function | Description | Params |
|----------|-------------|--------|
| `tsnorm` | Truncated skew-normal (Bellstedt+2020) | 5: peak, width, skew, trunc, log_peak_sfr |
| `snorm` | Skew-normal | 4: peak, width, skew, log_peak_sfr |
| `norm` | Gaussian | 3: peak, width, log_peak_sfr |
| `lnorm` | Log-normal in log10(age) | 3: log_peak, width, log_peak_sfr |
| `dpl` / `double_powerlaw` | Double power law (Carnall+2018) | 4: alpha, beta, tau, log_peak_sfr |
| `delayed_tau` / `delayed_exponential_sfh` | Peaks at start + tau | 3: tau, t_start, log_peak_sfr |
| `exponential_sfh` | Declining exponential | 3: tau, t_start, log_peak_sfr |
| `constant_sfh` | Flat SFR | 3: t_start, t_end, log_sfr |
| `triweight_burst` | Compact triweight kernel in log-age | 3: log_age_center, log_width, log_mass |

Convention: `t_lookback` in years, SFR in Msun/yr. Constant `AGEMAX_YR = 14e9`.

#### `models/sfh/gp_sfh.py` -- GP realizations from PSD

Implements the NIFTy correlated field model: s = IFFT(sqrt(P) * xi). Two modes: stochastic (random GP draws for mocks) and deterministic (fixed xi for inference).

| Function | Purpose |
|----------|---------|
| `make_log_age_grid(n_grid)` | Create uniform grid in log10(age/yr) |
| `gp_from_xi(xi, sqrt_power, ...)` | Deterministic GP from latent vector (inference) |
| `generate_gp_fourier(key, sqrt_power, ...)` | Random GP draw (mocks) |
| `generate_gp_batch(key, n, ...)` | Batch of random GP draws |
| `compute_sqrt_power_drw(sigma, tau_yr, ...)` | Precompute sqrt(P) for DRW kernel |

Imports from: nothing (self-contained JAX).

#### `models/sfh/psd_models.py` -- Power spectral density kernels

PSD functions for stochastic SFH variability. DRW (damped random walk / Lorentzian) is the primary model.

| Function | Purpose |
|----------|---------|
| `psd_drw(omega, sigma, tau_yr)` | Damped random walk PSD (primary) |
| `drw_variance(sigma, tau_yr)` | Analytic variance of DRW process |
| `drw_acf(lag, sigma, tau_yr)` | Autocorrelation function of DRW |

---

### `models/dust/` -- Dust Attenuation and Emission

#### `models/dust/attenuation.py` -- Two-component dust attenuation

Generalized Charlot & Fall (2000) model with pluggable attenuation curves and optional clumpy geometry (f_obscuration, Lower 2022).

| Key exports | Purpose |
|-------------|---------|
| `two_component_dust(wave, age_weights, tau_bc, tau_diff, ...)` | Full attenuation computation |
| `two_component_dust_fast(...)` | Optimized version with precomputed age weights |
| `DUST_LAWS` | Registry of attenuation curves |
| `register_dust_law(name, fn)` | Register a custom curve |
| `get_dust_law(name)` | Retrieve a curve by name |
| `precompute_dust_age_weights(log_ages)` | Sigmoid weights computed once at init |

Available curves: `power_law`, `calzetti`, `kriek_conroy`, `smc`, `cardelli`, `li08`, `salim`.

Imports from: nothing (self-contained JAX).

#### `models/dust/emission.py` -- IR re-emission

Energy-balance dust emission: total IR luminosity = total absorbed UV/optical. All models are JIT-compatible.

| Key exports | Purpose |
|-------------|---------|
| `apply_dust_emission(wave, sed, L_absorbed, model_name, ...)` | Add IR emission to SED |
| `DUST_EMISSION_MODELS` | Registry of emission models |
| `register_emission_model(name, fn)` | Register a custom model |
| `get_emission_model(name)` | Retrieve by name |
| `compute_absorbed_luminosity(wave, sed_intrinsic, sed_attenuated)` | Compute L_absorbed |
| `compute_absorbed_luminosity_from_tau(wave, sed_intrinsic, tau)` | L_absorbed from optical depth |
| `planck_bnu(wave, T)` | Planck function |
| `cmb_corrected_temperature(T_dust, z)` | CMB heating correction |
| `cmb_contrast_factor(T_dust, z, beta)` | CMB contrast at high-z |

Available models: `modified_blackbody`, `dale2014` (1 param), `draine_li2007` (3 params, analytic), `draine_li2014` (4 params, analytic). Tabulated versions via `register_dale2014_tabulated`, `register_dl14_tabulated`.

---

### `models/agn/` -- Active Galactic Nuclei

Six complexity levels, all accessed through a registry.

#### `models/agn/unified.py` -- Registry and unified combiner

| Key exports | Purpose |
|-------------|---------|
| `AGN_MODELS` | Registry dict of named AGN models |
| `get_agn_model(name)` | Retrieve model function by name |
| `register_agn_model(name, fn)` | Register a custom AGN model |
| `unified_agn(wave, agn_log_lbol, disc_model, ...)` | Generic disc + torus combiner |
| `unified_nlr_blr(wave, ...)` | Full model with NLR/BLR decomposition |

Registered models: `simple` (3 params), `standard` (6 params), `kubota_done` (8+ params), `unified_nlr_blr` (12+ params), `skirtor` (7 params), `qsogen` (7 params).

Imports from: `disc`, `torus`, `blr`, `nlr`, `skirtor`.

#### `models/agn/disc.py` -- Accretion disc models

| Function | Description |
|----------|-------------|
| `powerlaw_disc(wave, agn_log_lbol, alpha_uv, ...)` | Simple power-law + UV cutoff |
| `multicolor_disc(wave, agn_log_lbol, log_mbh, ...)` | Shakura-Sunyaev multi-color disc |

#### `models/agn/torus.py` -- Dust torus models

| Function | Description |
|----------|-------------|
| `simple_torus(wave, L_disc, T_torus, cf, ...)` | Single-temperature modified blackbody |
| `two_temperature_torus(wave, L_disc, ...)` | Hot + warm dust components |

#### `models/agn/blr.py` -- Broad Line Region

Analytic BLR template with broad Gaussian profiles (FWHM ~ 1000-10000 km/s) at strong permitted lines.

| Function | Description |
|----------|-------------|
| `blr_emission(wave, L_bol, ...)` | BLR line emission |

#### `models/agn/nlr.py` -- Narrow Line Region

Power-law continuum + forbidden-line Gaussian profiles (FWHM ~ 500 km/s). Isotropic (not torus-obscured).

| Function | Description |
|----------|-------------|
| `nlr_emission(wave, L_bol, ...)` | NLR line + continuum emission |

#### `models/agn/skirtor.py` -- SKIRTOR clumpy torus

Two approaches: analytic 3-temperature approximation (no data needed) and full template grid interpolation (5D, requires downloaded grid).

| Function | Description |
|----------|-------------|
| `skirtor_analytic(wave, L_disc, ...)` | Analytic 3-temperature approximation |
| `create_skirtor_from_grid(grid_path)` | Build interpolator from template grid |

#### `models/agn/qsogen.py` -- QSOgen empirical quasar SED

Temple, Hewett & Banerji (2021) empirical model. Four additive f_nu components: UV power law, optical power law, hot dust blackbody, emission lines. Produces the characteristic "v-shaped" quasar spectrum.

| Function | Description |
|----------|-------------|
| `qsogen(wave, agn_log_lbol, ...)` | Full QSOgen SED (registered as AGN model) |
| `qsogen_sed(wave, plslp1, plslp2, plbrk, tbb, bbnorm, emline, ebv)` | Low-level SED computation |

---

### `models/nebular/` -- Nebular Emission

Three backends for adding emission lines + continuum to the stellar SED.

#### `models/nebular/baked_in.py` -- BakedInBackend

No-op backend for SSP files that already include nebular emission (wNE files). Returns zero additional contribution. No free parameters.

#### `models/nebular/cloudy_grid.py` -- CloudyGridBackend

Precomputed CLOUDY photoionization grids (Byler+2017). Computes nebular emission from ionization parameter (logU) and gas metallicity (logZ_gas) via grid interpolation.

| Key methods | Purpose |
|-------------|---------|
| `CloudyGridBackend(grid_path, ssp_data)` | Load grid from HDF5 |
| `predict_nebular_sed(ssp_weights, wave, log_z, ...)` | Compute nebular contribution |

Imports from: h5py, numpy.

#### `models/nebular/cue.py` -- CueBackend

Pure JAX re-implementation of the Cue neural emulator (Li et al. 2025). 12 free parameters including ionizing spectrum shape and gas abundance ratios. Uses Speculator neural networks with PCA output basis. No TensorFlow dependency.

| Key methods | Purpose |
|-------------|---------|
| `CueBackend(weights_path)` | Load pre-trained weights from npz |
| `predict_nebular_sed(ssp_weights, wave, log_z, ...)` | Compute nebular lines + continuum |

#### `models/nebular/ionizing_spectrum.py` -- Ionizing spectrum parameterization

Fits a 4-segment piecewise power law to the ionizing SED (lambda < 912 A) to compute the 7 Cue input parameters (4 slopes + 3 log flux ratios).

| Key functions | Purpose |
|---------------|---------|
| `fit_ionizing_spectrum(wave, flux)` | Extract Cue ionizing spectrum parameters |

---

### `models/observation/` -- Observation Models

#### `models/observation/photometry.py` -- Filter convolution

Computes observed flux densities by convolving the rest-frame SED through filter transmission curves.

| Key exports | Purpose |
|-------------|---------|
| `FilterCurve` | NamedTuple: wave, trans, name |
| `compute_photometry(wave, sed, filters, z)` | Integrate SED through filters |

#### `models/observation/filters.py` -- Filter management

Downloads, caches, and loads filter curves from the SVO Filter Profile Service.

| Key exports | Purpose |
|-------------|---------|
| `load_filter_set(names)` | Load list of filters by short name |
| `FILTER_REGISTRY` | Dict mapping short names to SVO IDs |

Supported filter systems: SDSS, JWST (NIRCam, MIRI), HST (ACS, WFC3), 2MASS, WISE, Euclid, Rubin/LSST, GALEX, Spitzer IRAC, HSC, DES, VISTA, UKIRT.

Imports from: `photometry.FilterCurve`.

#### `models/observation/spectroscopy.py` -- Spectroscopic forward model

Pixel-level spectral fitting with LSF convolution and emission-line blending for low-resolution instruments.

| Key exports | Purpose |
|-------------|---------|
| `apply_lsf(wave, flux, sigma_v)` | Line Spread Function convolution |
| `blend_emission_lines(wave, line_waves, line_fluxes, sigma)` | Place blended Gaussian lines |
| `nirspec_prism_resolution(wave)` | JWST NIRSpec PRISM R(lambda) |
| `nirspec_g140m_resolution(wave)` | JWST NIRSpec G140M R(lambda) |
| `SSP_LIBRARY_RESOLUTIONS` | Dict of sigma_v for MILES, C3K, etc. |

#### `models/observation/calibration.py` -- Spectrophotometric calibration

Chebyshev polynomial calibration correction for flux-calibration uncertainties in spectra.

| Key exports | Purpose |
|-------------|---------|
| `chebyshev_basis(x, order)` | Evaluate Chebyshev polynomials |
| `calibration_polynomial(wave, coeffs, wave_range)` | Build calibration correction C(lambda) |
| `apply_calibration(flux, wave, coeffs, wave_range)` | Apply multiplicative correction |

#### `models/observation/eline_marginalization.py` -- Emission-line marginalization

Analytically marginalizes emission-line amplitudes out of the spectral likelihood under a Gaussian prior (following Prospector / Johnson+2021).

| Key exports | Purpose |
|-------------|---------|
| `build_eline_design_matrix(wave, line_waves, sigma)` | Gaussian line profiles matrix |
| `marginalize_emission_lines(data, model, noise, G, ...)` | Marginalized log-likelihood |
| `predict_with_marginalized_lines(data, model, noise, G, ...)` | Best-fit model with lines |
| `DEFAULT_LINE_NAMES` | Standard emission line names |
| `DEFAULT_LINE_WAVELENGTHS` | Rest-frame wavelengths (Angstrom) |

---

### `models/sps/` -- Stellar Population Synthesis

#### `models/sps/dsps_wrapper.py` -- DSPS integration

Wraps the DSPS CSP integral and SSP template loading. Core mapping: SFH weights to composite stellar population spectrum.

| Key exports | Purpose |
|-------------|---------|
| `SSPData` | NamedTuple: ssp_wave, ssp_flux, ssp_lg_age_gyr, ssp_lgmet, ssp_mass_remaining |
| `load_ssp_data(path)` | Load SSP templates from HDF5 |
| `compute_csp_sed(ssp_data, weights, log_z)` | Weighted sum of SSP templates |
| `compute_csp_weights(sfr, log_ages, age_at_z)` | SFR to SSP weights |
| `effective_metallicity(ssp_data, weights, log_z)` | Mass-weighted metallicity |
| `interpolate_metallicity(ssp_data, log_z)` | Interpolate SSP grid to a single Z |
| `interpolate_metallicity_evolving(ssp_data, log_z_array)` | Time-dependent Z(t) |
| `interpolate_metallicity_smooth(...)` | Smooth Z interpolation with scatter |
| `interpolate_metallicity_smooth_evolving(...)` | Smooth + time-dependent |
| `compute_log_z_evolving(...)` | Compute Z(t) from enrichment model |

#### `models/sps/mass_remaining.py` -- Surviving stellar mass

Computes the fraction of formed mass still in living stars + remnants as a function of age and IMF. Independent of SSP spectral library.

| Key exports | Purpose |
|-------------|---------|
| `compute_mass_remaining(log_ages, imf)` | Mass fraction vs. age for given IMF |

Supported IMFs: Chabrier (2003), Salpeter (1955), Kroupa (2001).

#### `models/sps/precompute.py` -- Precomputed SSP observables

Pre-integrates SSPs through filters or rebins to observed wavelengths at fixed or tabulated redshift, eliminating the wavelength dimension from the gradient tape.

| Key exports | Purpose |
|-------------|---------|
| `precompute_photometry(ssp_data, filters, z)` | Fixed-z filter precomputation (30-50x speedup) |
| `precompute_spectroscopy(ssp_data, wave_obs, z)` | Fixed-z spectral rebinning |
| `precompute_photometry_ztable(ssp_data, filters)` | Free-z precomputation on z-grid |
| `fast_photometry_ztable(weights, ztable, z, ...)` | Interpolate precomputed to current z |

---

### `models/igm.py` -- Intergalactic Medium Absorption

Inoue et al. (2014) mean IGM transmission. Accounts for Lyman-series line absorption (LAF + DLA) and Lyman-continuum absorption.

| Key exports | Purpose |
|-------------|---------|
| `igm_transmission(wave_obs, z)` | Mean T_IGM(lambda_obs, z_source) |

Takes **observed-frame** wavelengths (not rest-frame).

### `models/radio.py` -- Radio Emission

Predicts radio SED from SFR (via FIR-radio correlation) and AGN (via radio-loudness).

| Key exports | Purpose |
|-------------|---------|
| `radio_star_forming(wave, L_ir, q_ir, alpha_sf, ...)` | Synchrotron from SF |
| `radio_agn(wave, L_bol, R_radio, alpha_agn, ...)` | AGN radio jets |

### `models/xray.py` -- X-ray Emission

X-ray SED from three components: X-ray binaries (HMXB + LMXB), AGN corona, hot gas.

| Key exports | Purpose |
|-------------|---------|
| `xray_xrb(wave, sfr, stellar_mass, ...)` | XRB emission from SFR + mass |
| `xray_agn_corona(wave, L_bol, ...)` | AGN power-law + cutoff |

---

## `diagnostics/` -- Differentiability-Enabled Analysis

Tools that exploit the end-to-end differentiability of the full pipeline.

### `diagnostics/fisher.py` -- Fisher Information Matrix

Exact FIM via autodiff. Applications: Fisher forecasting, optimal filter design, survey planning, Laplace approximation.

| Key exports | Purpose |
|-------------|---------|
| `compute_jacobian(predict_fn, params, param_keys)` | Jacobian d(model)/d(theta) |
| `compute_fisher_matrix(predict_fn, params, noise, param_keys)` | Full FIM |
| `laplace_posterior(fim, params)` | Gaussian approximation from Hessian at MAP |

### `diagnostics/saliency.py` -- Gradient SEDs

Computes d(flux(lambda))/d(theta) -- sensitivity of each wavelength to each parameter.

| Key exports | Purpose |
|-------------|---------|
| `compute_gradient_sed(forward_model, params, param_name)` | Gradient SED for one parameter |
| `compute_all_gradient_seds(forward_model, params)` | Gradient SEDs for all free params |

### `diagnostics/green_functions.py` -- Green's and window functions

Time-domain sensitivity: which lookback times contribute to flux at each wavelength/filter. Connects to PSD timescale constraints.

| Key exports | Purpose |
|-------------|---------|
| `compute_green_function(ssp_flux, wave, filter, ...)` | G_lambda(t_age): SSP mass-to-light |
| `compute_window_function(green, sfh)` | W_lambda(t_age) = G * SFR |

---

## `utils/` -- Shared Utilities

### `utils/transforms.py` -- Bounded/unbounded transforms

Sigmoid-based bijections connecting physical (bounded) and optimizer (unbounded) parameter spaces.

| Key exports | Purpose |
|-------------|---------|
| `sigmoid(x, x0, k, ymin, ymax)` | R to (ymin, ymax) |
| `inverse_sigmoid(y, x0, k, ymin, ymax)` | (ymin, ymax) to R |
| `to_bounded(x, lo, hi)` | Unbounded to bounded |
| `to_unbounded(y, lo, hi)` | Bounded to unbounded |

Used by: `inference/fitter`, `inference/hierarchical`, `inference/common`.

### `utils/grid.py` -- Log-age grid

Uniform grid in log10(age/yr). Finer resolution at recent times (Myr), coarser at early times (Gyr).

| Key exports | Purpose |
|-------------|---------|
| `make_log_age_grid(n_grid, log_age_min, log_age_max)` | Create the grid |
| `log_age_to_age_yr(log_age)` | Convert log10(age/yr) to years |

Constants: `DEFAULT_N_GRID=256`, `DEFAULT_LOG_AGE_MIN=6.0`, `DEFAULT_LOG_AGE_MAX=10.14`.

### `utils/cosmology.py` -- Flat LCDM cosmology

Minimal pure-JAX cosmology with Planck 2018 defaults (H0=67.4, Om0=0.315).

| Key exports | Purpose |
|-------------|---------|
| `luminosity_distance(z, h0, om0)` | d_L in cm (100-point Gauss-Legendre) |
| `age_at_z(z, h0, om0)` | Age of universe at redshift z |

### `utils/devices.py` -- JAX device management

Platform detection, memory reporting, GPU/CPU/TPU configuration.

| Key exports | Purpose |
|-------------|---------|
| `setup_jax(platform, enable_x64, ...)` | Auto-configure for current hardware |
| `device_info()` | Report available devices and memory |

### `utils/optimizations.py` -- Performance tricks

Memory and speed optimizations from NIFTy.re and Zacharegkas+2025.

| Key exports | Purpose |
|-------------|---------|
| `hartley(x)` | Real-to-real FFT (halves memory vs complex FFT) |
| `gradient_checkpoint(fn)` | Recompute-on-backward to save memory |

### `utils/sed_quantities.py` -- Derived quantity primitives

Pure JAX functions for computing physical quantities from SEDs. Used by both the lazy `Prediction` object and JIT-compatible batch methods.

| Key exports | Purpose |
|-------------|---------|
| `compute_uv_slope_beta(wave, l_nu)` | UV continuum slope beta |
| `compute_dn4000(wave, l_nu)` | 4000-A break strength |
| `compute_balmer_break(wave, l_nu)` | Balmer break index |
| `compute_rest_magnitude(wave, l_nu, filter)` | AB magnitude in a filter |
| `compute_bolometric_luminosity(wave, l_nu)` | L_bol from spectral integration |

---

## Cross-cutting Modules

### `distributions.py` -- Prior distributions

Parameter distribution objects for ParamSpec. Each defines a prior with sampling, log-probability, and standardize/unstandardize transforms for absorbing priors into the forward model.

| Class | Description | Bounds |
|-------|-------------|--------|
| `Uniform(lo, hi)` | Flat prior | [lo, hi] |
| `Gaussian(mu, sigma)` | Normal prior | (-inf, inf) |
| `LogUniform(lo, hi)` | Flat in log space | [lo, hi] |
| `LogNormal(mu, sigma)` | Log-normal | (0, inf) |
| `StudentT(mu, sigma, nu)` | Heavy-tailed | (-inf, inf) |
| `Fixed(value)` | Not free; held constant | N/A |

Key interface per distribution: `bounds`, `sample(key)`, `log_prob(x)`, `unstandardize(xi)`, `standardize(theta)`, `is_fixed`.

Used by: `core/param_spec`, `inference/fitter`, `inference/common`, `models/sfh/registry`.

### `plotting.py` -- Visualization utilities

Publication-quality plotting for SFH recovery, SED fits, spectroscopic fits, corner plots, and diagnostics tables. Colorblind-safe palette.

| Key exports | Purpose |
|-------------|---------|
| `setup_style()` | Configure matplotlib rcParams |
| `plot_sfh(prediction, ...)` | SFH vs lookback time |
| `plot_sfh_comparison(posteriors, ...)` | Compare SFH from multiple methods |
| `plot_sed_fit(model, posterior, data, ...)` | SED + residuals |
| `plot_spectrum_fit(model, posterior, data, ...)` | Spectral fit + residuals |
| `plot_corner_comparison(posteriors, ...)` | Overlay corner plots from multiple methods |
| `safe_corner(samples, labels, ...)` | Wrapper around corner.py with safety checks |
| `diagnostics_table(posteriors, ...)` | Summary table of convergence diagnostics |
| `COLORS` | Dict of colorblind-safe colors by role |
| `SDSS_WAVE_EFF` | Effective wavelengths for SDSS bands |
| `SPECTRAL_FEATURES` | Named spectral feature wavelength ranges |

### `simulate.py` -- Simulation interface

Forward-model SEDs from arbitrary SFH and metallicity arrays (e.g., from cosmological simulations). Bypasses parametric SFH models; directly uses the DSPS CSP integral.

| Key exports | Purpose |
|-------------|---------|
| `sed_from_sfh(t_gyr, sfr, ssp, log_z, ...)` | Rest-frame SED from tabulated SFH |
| `photometry_from_sfh(t_gyr, sfr, ssp, filters, ...)` | Photometry from tabulated SFH |

---

## Dependency Graph

Arrows indicate "imports from". Higher modules depend on lower ones.

```
tengri (top-level __init__)
  |
  +-- core/
  |     model.py ---------> param_spec, param_translate, sed_pipeline,
  |     |                    fused_kernels, prediction, noise
  |     param_spec.py -----> distributions, models/sfh/registry
  |     param_translate.py -> models/sfh/registry
  |     sed_pipeline.py ---> models/dust/attenuation, models/sps/dsps_wrapper
  |     fused_kernels.py --> models/dust, models/sps
  |     prediction.py -----> utils/sed_quantities, models/radio, models/xray
  |     noise.py ----------> distributions
  |     mock.py -----------> model
  |
  +-- inference/
  |     fitter.py ---------> distributions, utils/transforms, inference/common,
  |     |                    inference/raytrace, inference/map_optimizer,
  |     |                    inference/nuts, inference/geovi, inference/standardized
  |     standardized.py ---> models/sfh/gp_sfh, utils/grid
  |     hierarchical.py ---> utils/transforms
  |     posterior.py ------> (standalone, uses jax/numpy)
  |     raytrace.py -------> (standalone, pure JAX)
  |     map_optimizer.py --> inference/common
  |     nuts.py -----------> inference/common
  |     geovi.py ----------> inference/common
  |     geovi_nuts.py -----> (standalone, pure JAX)
  |     common.py ---------> (standalone, uses jax)
  |     vi_config.py ------> (standalone, dataclass)
  |
  +-- models/
  |     sfh/
  |       registry.py -----> distributions, sfh/mean_sfh, sfh/gp_sfh, sfh/psd_models
  |       mean_sfh.py -----> (standalone, pure JAX)
  |       gp_sfh.py -------> (standalone, pure JAX)
  |       psd_models.py ---> (standalone, pure JAX)
  |     dust/
  |       attenuation.py --> (standalone, pure JAX)
  |       emission.py -----> (standalone, pure JAX)
  |     agn/
  |       unified.py ------> agn/disc, agn/torus, agn/blr, agn/nlr, agn/skirtor
  |       disc.py ---------> (standalone, pure JAX)
  |       torus.py --------> (standalone, pure JAX)
  |       blr.py ----------> (standalone, pure JAX)
  |       nlr.py ----------> (standalone, pure JAX)
  |       skirtor.py ------> (standalone, pure JAX)
  |       qsogen.py -------> agn/unified (register_agn_model)
  |     nebular/
  |       baked_in.py -----> (standalone)
  |       cloudy_grid.py --> (h5py, numpy)
  |       cue.py ----------> (standalone, pure JAX)
  |       ionizing_spectrum.py -> (standalone)
  |     observation/
  |       filters.py ------> observation/photometry (FilterCurve)
  |       photometry.py ---> (standalone, pure JAX)
  |       spectroscopy.py -> (standalone, pure JAX)
  |       calibration.py --> (standalone, pure JAX)
  |       eline_marginalization.py -> (standalone, pure JAX)
  |     sps/
  |       dsps_wrapper.py -> (standalone, pure JAX)
  |       mass_remaining.py -> (standalone, pure JAX)
  |       precompute.py ---> (standalone, pure JAX)
  |     igm.py ------------> (standalone, pure JAX)
  |     radio.py ----------> (standalone, pure JAX)
  |     xray.py -----------> (standalone, pure JAX)
  |
  +-- diagnostics/
  |     fisher.py ---------> (standalone, uses jax.jacfwd)
  |     saliency.py -------> (standalone, uses jax.grad)
  |     green_functions.py -> (standalone, pure JAX)
  |
  +-- utils/
  |     transforms.py -----> (standalone, pure JAX)
  |     grid.py -----------> (standalone, pure JAX)
  |     cosmology.py ------> (standalone, pure JAX)
  |     devices.py --------> (standalone, os/warnings)
  |     optimizations.py --> (standalone, pure JAX)
  |     sed_quantities.py -> (standalone, pure JAX)
  |
  +-- distributions.py ----> (standalone, pure JAX)
  +-- plotting.py ---------> (matplotlib, numpy)
  +-- simulate.py ---------> models/sps/dsps_wrapper, models/dust, utils/cosmology
```

Most leaf modules (physics models, utils) are self-contained pure JAX with no internal dependencies, making them independently testable and reusable. Dependencies flow upward: `core/model.py` sits at the top, orchestrating all physics modules through the SED pipeline.
