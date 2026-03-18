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
# # EVI (Expansion-point VI) Test
#
# Mock galaxy → EVI. Stochastic model, D ≈ 137.
#
# EVI = MGVI (cheap linear samples) for first half of iterations,
# then geoVI (nonlinear) for second half.

# %%
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from diffsed import (
    Fitter,
    Fixed,
    Model,
    ParamSpec,
    Uniform,
    load_filter_set,
    load_ssp_data,
)

ssp_data = load_ssp_data(
    "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# %% [markdown]
# ## Model + Mock

# %%
spec = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0),
    sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0),
    sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Uniform(0.1, 4.0),
    psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    stochastic=True,
    n_grid=128,
)
model = Model(spec, ssp_data, filters=filters)

key = jax.random.PRNGKey(2026)
true_params = spec.sample(key)
true_params.update(
    sfh_alpha=1.0,
    sfh_beta=1.5,
    sfh_tau_peak_gyr=8.0,
    sfh_peak_sfr=30.0,
    psd_sigma=2.0,
    psd_tau_myr=20.0,
    met_logzsol=-0.3,
    dust_tau_bc=0.5,
    dust_tau_diff=0.3,
)
mock = model.mock(true_params, snr=20.0, key=key)
print(f"D = {spec.n_free}, {len(mock.flux_obs)} data points")

# %%
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

# %% [markdown]
# ## EVI Inference
#
# n_posterior_samples=0 means we only use the samples from optimize_kl
# (these are already valid posterior samples). No expensive extra CG solves.

# %%
key1, key = jax.random.split(key)
t0 = time.perf_counter()
result = fitter.run(
    "evi",
    n_iterations=10,
    n_samples=3,
    n_posterior_samples=2000,
    verbose=False,
    key=key1,
)
t_evi = time.perf_counter() - t0
print(f"\nEVI: {t_evi:.1f} s, {result.diagnostics['n_samples']} samples")

# %% [markdown]
# ## SFH Recovery

# %%
sfh_true = model.predict_sfh(true_params)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(sfh_true["t_gyr"], sfh_true["sfr_full"], "k-", lw=2.5, label="Truth")
ax.plot(
    sfh_true["t_gyr"],
    sfh_true["sfr_mean"],
    "k:",
    lw=1,
    alpha=0.4,
    label="Secular mean",
)
model.plot_sfh_posterior(
    result, true_params=true_params, color="#ff7f0e", label="EVI", ax=ax
)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
ax.set_title(f"EVI SFH Recovery (D ≈ 137, {t_evi:.1f} s)")
ax.set_xlim(0, 13.5)
ax.set_ylim(0, max(3 * float(np.max(np.array(sfh_true["sfr_full"]))), 30))
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("figures/test_geovi_sfh.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/test_geovi_sfh.png")

# %% [markdown]
# ## Corner Plot

# %%
from diffsed.plotting import safe_corner

# Physical params to show (exclude psd_xi latent vector)
param_names = [
    "sfh_alpha",
    "sfh_beta",
    "sfh_tau_peak_gyr",
    "sfh_peak_sfr",
    "psd_sigma",
    "psd_tau_myr",
    "met_logzsol",
    "dust_tau_bc",
    "dust_tau_diff",
]

truths = {k: float(true_params[k]) for k in param_names if k in result.samples}

fig = safe_corner(result, params=param_names, truths=truths)
if fig is not None:
    fig.suptitle(f"EVI Posterior (D ≈ 137, {t_evi:.1f} s)", y=1.02)
    fig.savefig("figures/test_geovi_corner.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/test_geovi_corner.png")
else:
    print("Corner plot skipped (too few samples?)")

# Build arrays for photometry plot
samples_arr = np.column_stack(
    [np.array(result.samples[k]) for k in param_names if k in result.samples]
)
labels = [k for k in param_names if k in result.samples]

# %% [markdown]
# ## Photometry Fit

# %%
wave_eff = np.array([3551, 4686, 6166, 7480, 8932])  # SDSS ugriz
band_names = ["u", "g", "r", "i", "z"]

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

# Observed data
ax1.errorbar(
    wave_eff,
    mock.flux_obs,
    yerr=mock.noise,
    fmt="ko",
    ms=8,
    capsize=3,
    label="Observed",
    zorder=10,
)

# Model predictions from posterior samples
pred_fluxes = []
for i in range(min(len(samples_arr), 50)):
    params_i = dict(true_params)  # start with fixed params
    for j, name in enumerate(labels):
        params_i[name] = float(samples_arr[i, j])
    if "psd_xi" in result.samples:
        params_i["psd_xi"] = result.samples["psd_xi"][i]
    pred = model.predict_photometry(params_i)
    pred_fluxes.append(np.array(pred))
    ax1.plot(wave_eff, pred, "-", color="#ff7f0e", alpha=0.15, lw=0.8)

pred_fluxes = np.array(pred_fluxes)
pred_median = np.median(pred_fluxes, axis=0)

ax1.plot(wave_eff, pred_median, "s-", color="#ff7f0e", ms=6, lw=1.5, label="EVI median")
ax1.set_ylabel("Flux")
ax1.legend(fontsize=9)
ax1.set_title(f"Photometry Fit — EVI ({t_evi:.1f} s)")

# Residuals
residuals = (mock.flux_obs - pred_median) / mock.noise
ax2.bar(wave_eff, residuals, width=300, color="gray", alpha=0.7)
ax2.axhline(0, color="k", lw=0.5)
ax2.set_ylabel(r"$(f_{\rm obs} - f_{\rm model}) / \sigma$")
ax2.set_xlabel(r"Wavelength [$\AA$]")
ax2.set_xticks(wave_eff)
ax2.set_xticklabels(band_names)

plt.tight_layout()
plt.savefig("figures/test_geovi_photometry.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/test_geovi_photometry.png")
