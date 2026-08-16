# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Quickstart: Spectroscopic SED Fitting
#
# Fit a galaxy spectrum with EVI (Expansion-point Variational Inference).
#
# - 200 spectral pixels, SNR=30
# - Smooth double-power-law SFH (D=7)
# - Full posterior in ~12 seconds

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from tengri import (
    Fitter,
    Fixed,
    SEDModel,
    ParamSpec,
    Uniform,
    load_filter_set,
    load_ssp_data,
)

# %% [markdown]
# ## 1. SEDModel Setup

# %%
ssp_data = load_ssp_data(
    "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

spec = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

# Precompute SSP interpolation for the observed wavelength grid (~8x speedup)
wave_obs = np.linspace(3800, 9200, 200).astype(np.float64)
model = SEDModel(spec, ssp_data, filters=filters).precompute_spectroscopy(wave_obs)

print(f"Free parameters: {spec.free_params}")
print(f"D = {len(spec.free_params)}, n_pixels = {len(wave_obs)}")

# %% [markdown]
# ## 2. Mock Spectrum

# %%
key = jax.random.PRNGKey(42)

true_params = spec.sample(key)
true_params.update(
    sfh_dpl_alpha=1.0,
    sfh_dpl_beta=1.5,
    sfh_dpl_tau_gyr=8.0,
    sfh_dpl_log_peak_sfr=1.5,
    met_logzsol=-0.3,
    dust_tau_bc=0.5,
    dust_tau_diff=0.3,
)

mock = model.mock_spectrum(true_params, wave_obs, snr=30.0, key=key)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(wave_obs, mock.flux_true, "k-", lw=0.8, alpha=0.5, label="Truth")
ax.errorbar(
    wave_obs,
    mock.flux_obs,
    yerr=mock.noise,
    fmt=".",
    ms=2,
    color="C0",
    alpha=0.5,
    label="Observed (SNR=30)",
)
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel(r"Flux [erg/s/cm$^2$/Hz]")
ax.set_title("Mock Galaxy Spectrum")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. EVI Inference
#
# EVI runs a fully JIT-compiled geoVI optimization loop, then draws
# posterior samples via JIT-compiled conjugate gradient.
# With spectroscopy precomputation, the forward model is ~8x faster.

# %%
import time

fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="spectroscopy")

key_fit, key = jax.random.split(key)
t0 = time.perf_counter()
result = fitter.run("native_evi", n_posterior_samples=500, n_seeds=3, key=key_fit)
dt = time.perf_counter() - t0

pred = model.predict_spectrum(result.params)
chi2_dof = float(jnp.sum(((mock.flux_obs - pred) / mock.noise) ** 2)) / len(wave_obs)
print(f"Time: {dt:.1f}s | chi2/dof: {chi2_dof:.2f} | Samples: {result.diagnostics['n_samples']}")

# %% [markdown]
# ## 4. Spectral Fit

# %%
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(10, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

ax1.errorbar(
    wave_obs,
    mock.flux_obs,
    yerr=mock.noise,
    fmt=".",
    ms=2,
    color="gray",
    alpha=0.4,
    label="Observed",
)

# Draw posterior spectra
n_draw = min(50, result.diagnostics["n_samples"])
for i in range(n_draw):
    params_i = dict(true_params)
    for name in spec.free_params:
        if name in result.samples:
            params_i[name] = float(result.samples[name][i])
    pred_i = model.predict_spectrum(params_i)
    ax1.plot(wave_obs, pred_i, "-", color="C1", alpha=0.04, lw=0.5)

ax1.plot(wave_obs, pred, "-", color="C1", lw=1.5, label="EVI posterior")
ax1.set_ylabel(r"Flux [erg/s/cm$^2$/Hz]")
ax1.legend()
ax1.set_title(f"Spectral Fit — EVI ({dt:.1f}s, $\\chi^2$/dof = {chi2_dof:.2f})")

residuals = (mock.flux_obs - pred) / mock.noise
ax2.plot(wave_obs, residuals, ".", ms=2, color="C0", alpha=0.5)
ax2.axhline(0, color="k", lw=0.5)
ax2.axhline(2, color="k", lw=0.3, ls="--")
ax2.axhline(-2, color="k", lw=0.3, ls="--")
ax2.set_ylabel(r"$(f_{\rm obs} - f_{\rm model})/\sigma$")
ax2.set_xlabel(r"Wavelength [$\AA$]")
ax2.set_ylim(-5, 5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Corner Plot

# %%
from tengri.analysis.plotting import safe_corner

truths = {k: float(true_params[k]) for k in spec.free_params}
fig = safe_corner(result, params=spec.free_params, truths=truths)
if fig is not None:
    fig.suptitle(
        f"Spectroscopic Posterior (D={len(spec.free_params)}, {dt:.1f}s)", y=1.02
    )
plt.show()

# %% [markdown]
# ## 6. Parameter Recovery

# %%
print(f"{'Parameter':>25}  {'Truth':>7}  {'Median':>7}  {'Std':>7}  {'Bias(σ)':>8}")
print("-" * 65)
for name in spec.free_params:
    truth = float(true_params[name])
    if name in result.samples:
        samples = np.array(result.samples[name])
        med = np.median(samples)
        std = np.std(samples)
        bias = (med - truth) / std if std > 0 else 0
        print(f"{name:>25}  {truth:7.2f}  {med:7.2f}  {std:7.2f}  {bias:+8.2f}")
