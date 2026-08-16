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
# # geoVI Test — Stochastic SFH Recovery
#
# Geometric Variational Inference (geoVI; Frank et al. 2021) is the
# primary inference method for high-dimensional stochastic SFH models.
# It constructs a coordinate transformation $g(\boldsymbol{\xi}; \bar{\boldsymbol{\xi}})$
# that flattens the posterior metric, making the posterior approximately
# Gaussian in the transformed space.
#
# This notebook demonstrates geoVI on a **bursty mock galaxy**
# ($D \approx 137$: 128 GP latent variables + 9 physical parameters),
# then compares with MGVI (the linearized variant) and EVI (the
# JIT-compiled fast path that starts with MGVI warmup and refines
# with nonlinear geoVI samples).
#
# **Key takeaway:** geoVI recovers the bursty SFH and physical
# parameters in $\sim$60 s on CPU, producing 200+ posterior samples
# without any MCMC tuning.

# %%
import time

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

ssp_data = load_ssp_data(
    "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# %% [markdown]
# ## SEDModel + Mock
#
# Bursty star-forming galaxy: $\sigma_{\rm PS} = 2.0$ (factor $\sim$7
# fluctuations in SFR), $\tau_{\rm PS} = 20$ Myr (SN feedback timescale).

# %%
spec = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_field_psd_sigma=Uniform(0.1, 4.0),
    sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type=["dpl", "field"],
    n_grid=128,
)
model = SEDModel(spec, ssp_data, filters=filters)

key = jax.random.PRNGKey(2026)
true_params = spec.sample(key)
true_params.update(
    sfh_dpl_alpha=1.0,
    sfh_dpl_beta=1.5,
    sfh_dpl_tau_gyr=8.0,
    sfh_dpl_log_peak_sfr=jnp.log10(30.0),
    sfh_field_psd_sigma=2.0,
    sfh_field_psd_tau_myr=20.0,
    met_logzsol=-0.3,
    dust_tau_bc=0.5,
    dust_tau_diff=0.3,
)
mock = model.mock(true_params, snr=20.0, key=key)
print(f"D = {spec.n_free}, {len(mock.flux_obs)} data points")

# %%
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

# %% [markdown]
# ## 1. geoVI (nonlinear)
#
# Standard NIFTy geoVI: each KL iteration draws `n_samples` from the
# current Gaussian approximation, refines the expansion point, and
# updates the posterior metric $\mathcal{M} = \mathbf{J}^T\mathbf{J} + \mathbf{I}$.
# After convergence, `n_posterior_samples` cheap draws give the final posterior.
#
# `sample_mode="nonlinear_resample"` (default) uses the full nonlinear
# coordinate transformation $g$, giving more accurate samples than the
# linear variant.

# %%
key1, key = jax.random.split(key)
t0 = time.perf_counter()
result_geovi = fitter.run(
    "native_geovi",
    n_iterations=15,
    n_samples=6,
    n_posterior_samples=200,
    n_seeds=5,
    verbose=False,
    key=key1,
)
t_geovi = time.perf_counter() - t0
print(f"geoVI: {t_geovi:.1f} s, {result_geovi.diagnostics['n_samples']} samples")

# %% [markdown]
# ## 2. MGVI (linear)
#
# MGVI drops the nonlinear correction and approximates the posterior
# directly as $\mathcal{N}(\bar{\boldsymbol{\xi}},\, \mathcal{M}^{-1})$.
# Cheaper per iteration, but less accurate for non-Gaussian posteriors.
# The `"mgvi"` method is just `"geovi"` with `sample_mode="linear_resample"`.

# %%
key2, key = jax.random.split(key)
t0 = time.perf_counter()
result_mgvi = fitter.run(
    "native_mgvi",
    n_iterations=15,
    n_samples=6,
    n_posterior_samples=200,
    n_seeds=5,
    verbose=False,
    key=key2,
)
t_mgvi = time.perf_counter() - t0
print(f"MGVI: {t_mgvi:.1f} s, {result_mgvi.diagnostics['n_samples']} samples")

# %% [markdown]
# ## 3. EVI (JIT-compiled fast path)
#
# EVI is the production workhorse: a fully JIT-compiled loop that
# auto-stops when KL converges, with ~500x less Python overhead
# than the NIFTy `optimize_kl` path. It starts from MAP automatically.

# %%
key3, key = jax.random.split(key)
t0 = time.perf_counter()
result_evi = fitter.run(
    "native_evi",
    n_iterations=10,
    n_samples=3,
    n_seeds=5,
    n_posterior_samples=2000,
    verbose=False,
    key=key3,
)
t_evi = time.perf_counter() - t0
print(f"EVI: {t_evi:.1f} s, {result_evi.diagnostics['n_samples']} samples")

# %% [markdown]
# ## SFH Recovery — All Three Methods

# %%
sfh_true = model.predict_sfh(true_params)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)

methods = [
    ("geoVI (nonlinear)", result_geovi, "#ff7f0e", t_geovi),
    ("MGVI (linear)", result_mgvi, "#2ca02c", t_mgvi),
    ("EVI (JIT)", result_evi, "#9467bd", t_evi),
]

for ax, (name, result, color, wall) in zip(axes, methods):
    ax.plot(sfh_true["t_gyr"], sfh_true["sfr_full"], "k-", lw=2.5, label="Truth")
    ax.plot(sfh_true["t_gyr"], sfh_true["sfr_mean"],
            "k:", lw=1, alpha=0.4, label="Secular mean")
    model.plot_sfh_posterior(
        result, true_params=true_params, color=color, label=name, ax=ax
    )
    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_title(f"{name} ({wall:.1f} s)")
    ax.set_xlim(0, 13.5)
    ax.legend(fontsize=8)

axes[0].set_ylabel(r"SFR [M$_\odot$ yr$^{-1}$]")
sfr_max = float(np.max(np.array(sfh_true["sfr_full"])))
for ax in axes:
    ax.set_ylim(0, max(3 * sfr_max, 30))

plt.tight_layout()
plt.savefig("figures/test_geovi_sfh.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: figures/test_geovi_sfh.png")

# %% [markdown]
# ## Corner Plot — geoVI vs EVI

# %%
from tengri.analysis.plotting import safe_corner

param_names = [
    "sfh_dpl_alpha",
    "sfh_dpl_beta",
    "sfh_dpl_tau_gyr",
    "sfh_dpl_log_peak_sfr",
    "sfh_field_psd_sigma",
    "sfh_field_psd_tau_myr",
    "met_logzsol",
    "dust_tau_bc",
    "dust_tau_diff",
]

truths = {k: float(true_params[k]) for k in param_names if k in result_geovi.samples}

fig = safe_corner(result_geovi, params=param_names, truths=truths)
if fig is not None:
    fig.suptitle(f"geoVI Posterior (D ≈ 137, {t_geovi:.1f} s)", y=1.02)
    fig.savefig("figures/test_geovi_corner.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: figures/test_geovi_corner.png")
else:
    print("Corner plot skipped (too few samples?)")

# %% [markdown]
# ## Photometry Posterior Predictive Check

# %%
# Build arrays for posterior-predictive photometry
samples_arr = np.column_stack(
    [np.array(result_geovi.samples[k]) for k in param_names if k in result_geovi.samples]
)
labels = [k for k in param_names if k in result_geovi.samples]

wave_eff = np.array([3551, 4686, 6166, 7480, 8932])  # SDSS ugriz
band_names = ["u", "g", "r", "i", "z"]

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
)

ax1.errorbar(
    wave_eff, mock.flux_obs, yerr=mock.noise,
    fmt="ko", ms=8, capsize=3, label="Observed", zorder=10,
)

pred_fluxes = []
for i in range(min(len(samples_arr), 50)):
    params_i = dict(true_params)
    for j, name in enumerate(labels):
        params_i[name] = float(samples_arr[i, j])
    if "sfh_field_xi" in result_geovi.samples:
        params_i["sfh_field_xi"] = result_geovi.samples["sfh_field_xi"][i]
    pred = model.predict_photometry(params_i)
    pred_fluxes.append(np.array(pred))
    ax1.plot(wave_eff, pred, "-", color="#ff7f0e", alpha=0.15, lw=0.8)

pred_fluxes = np.array(pred_fluxes)
pred_median = np.median(pred_fluxes, axis=0)

ax1.plot(wave_eff, pred_median, "s-", color="#ff7f0e", ms=6, lw=1.5, label="geoVI median")
ax1.set_ylabel("Flux")
ax1.legend(fontsize=9)
ax1.set_title(f"Photometry Fit — geoVI ({t_geovi:.1f} s)")

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
