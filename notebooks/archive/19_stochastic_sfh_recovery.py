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
# # Recovering Bursty Star Formation Histories with tengri
#
# This notebook demonstrates the core capability of **tengri**: recovering
# bursty, stochastic star formation histories from broadband photometry
# using a PSD-based Gaussian process SFH model and evidence-based
# variational inference (EVI).
#
# The stochastic SFH model decomposes the star formation rate as:
#
# $$
# \mathrm{SFR}(t) = \mathrm{SFR}_{\mathrm{mean}}(t) \times \exp\!\bigl(\mathrm{GP}(t)\bigr)
# $$
#
# where $\mathrm{SFR}_{\mathrm{mean}}(t)$ is a smooth double-power-law
# backbone and $\mathrm{GP}(t)$ is a Gaussian process whose power spectral
# density is a damped random walk (DRW) controlled by two parameters:
#
# - $\sigma_{\mathrm{PSD}}$: amplitude of stochastic variability
# - $\tau_{\mathrm{PSD}}$: correlation timescale (Myr)
#
# **Outline:**
#
# 1. The stochastic SFH model: GP realizations across burstiness regimes
# 2. Mock galaxy generation with a bursty SFH
# 3. EVI inference and convergence
# 4. SFH recovery (the money figure)
# 5. PSD parameter recovery (corner plots)
# 6. Photometry fit quality
# 7. Parametric vs stochastic model comparison

# %%
import time

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import sys; sys.path.insert(0, ".")
from _plot_style import setup_style, COLORS, safe_corner
setup_style()

import os
os.makedirs("figures", exist_ok=True)

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)
from tengri.sfh.gp_sfh import (
    generate_gp_fourier, compute_sqrt_power_drw, make_log_age_grid,
)
from tengri.sfh.mean_sfh import double_powerlaw

# Reproducibility
KEY_MASTER = jax.random.PRNGKey(2026)

# %% [markdown]
# ## Section 1: The Stochastic SFH SEDModel
#
# The star formation rate is modulated by a Gaussian process whose
# power spectral density follows a damped random walk:
#
# $$
# P(\omega) = \frac{\sigma^2 \, \tau}{1 + (\tau\,\omega)^2}
# $$
#
# Higher $\sigma$ produces larger amplitude bursts; shorter $\tau$
# produces more rapid fluctuations. We visualize four regimes spanning
# the astrophysical range from secular disk evolution to extreme
# starburst activity.

# %%
# Grid setup
N_GRID = 64
log_ages = make_log_age_grid(n_grid=N_GRID)
d_log_age = float(log_ages[1] - log_ages[0])
ages_yr = 10.0 ** log_ages
ages_gyr = ages_yr / 1e9

# Mean SFH backbone (double power law)
mean_sfr = double_powerlaw(ages_yr, alpha=1.0, beta=1.5, tau=8e9, norm=30.0)

# Four burstiness regimes
regimes = [
    {"sigma": 0.3, "tau_myr": 100, "label": r"Smooth ($\sigma$=0.3, $\tau$=100 Myr)"},
    {"sigma": 1.0, "tau_myr": 50,  "label": r"Moderate ($\sigma$=1.0, $\tau$=50 Myr)"},
    {"sigma": 2.0, "tau_myr": 20,  "label": r"Bursty ($\sigma$=2.0, $\tau$=20 Myr)"},
    {"sigma": 3.0, "tau_myr": 10,  "label": r"Highly bursty ($\sigma$=3.0, $\tau$=10 Myr)"},
]

regime_colors = ["#2b6ca3", "#3a9a5b", "#d65f27", "#c03d3e"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

key = KEY_MASTER
for i, (cfg, ax) in enumerate(zip(regimes, axes.ravel())):
    tau_yr = cfg["tau_myr"] * 1e6
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, cfg["sigma"], tau_yr)

    # Draw 3 realizations
    for j in range(3):
        key, subkey = jax.random.split(key)
        gp = generate_gp_fourier(subkey, sqrt_p, N_GRID)
        sfr = mean_sfr * jnp.exp(gp)
        ax.plot(ages_gyr, sfr, color=regime_colors[i], alpha=0.5 + 0.15 * j,
                lw=1.0 + 0.3 * j)

    ax.plot(ages_gyr, mean_sfr, "k--", lw=1.0, alpha=0.4, label="Mean SFH")
    ax.set_title(cfg["label"], fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e4)
    ax.set_xlim(1e-3, 14)
    if i >= 2:
        ax.set_xlabel("Lookback time [Gyr]")
    if i % 2 == 0:
        ax.set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
    if i == 0:
        ax.legend(fontsize=8, loc="lower left")

plt.suptitle("GP-modulated SFH across burstiness regimes", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/19_sfh_regimes.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 2: Mock Galaxy Generation
#
# We generate a mock galaxy in the **bursty** regime
# ($\sigma_{\mathrm{PSD}} = 2.0$, $\tau_{\mathrm{PSD}} = 20$ Myr)
# using SDSS $ugriz$ + WISE W1 photometry at SNR = 20.

# %%
# Load SSP data and filters
ssp_data = load_ssp_data("../data/fsps_prsc_miles_chabrier.h5")
filters = load_filter_set([
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1",
])
filter_names = ["u", "g", "r", "i", "z", "W1"]
wave_eff = np.array([3551, 4686, 6166, 7480, 8932, 33526])

print(f"SSP: {len(ssp_data.ssp_lgmet)} metallicities, "
      f"{len(ssp_data.ssp_lg_age_gyr)} ages")
print(f"Filters: {filter_names}")

# %%
# Define the stochastic model
spec = ParamSpec(
    mean_sfh_type=["dpl", "field"],
    n_grid=64,
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
)

model = SEDModel(spec, ssp_data, filters=filters)

D_total = spec.n_free + spec.n_grid  # physical + GP latent
print(f"Free physical parameters: {spec.n_free}")
print(f"GP latent dimensions: {spec.n_grid}")
print(f"Total dimensions: {D_total}")
print(f"\nFree parameters: {spec.free_params}")

# %%
# Generate mock: sample from prior, override PSD params to bursty regime
true_params = spec.sample(jax.random.PRNGKey(2026))

# Override to known bursty configuration
true_params = {
    **true_params,
    "sfh_dpl_alpha": jnp.array(1.0),
    "sfh_dpl_beta": jnp.array(1.5),
    "sfh_dpl_tau_gyr": jnp.array(8.0),
    "sfh_dpl_log_peak_sfr": jnp.log10(jnp.array(30.0)),
    "sfh_field_psd_sigma": jnp.array(2.0),
    "sfh_field_psd_tau_myr": jnp.array(20.0),
    "met_logzsol": jnp.array(-0.3),
    "dust_tau_bc": jnp.array(0.5),
    "dust_tau_diff": jnp.array(0.3),
}

mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(2026))

print(f"\nTrue PSD: sigma={float(true_params['sfh_field_psd_sigma']):.1f}, "
      f"tau={float(true_params['sfh_field_psd_tau_myr']):.0f} Myr")
print(f"Photometry bands: {len(mock.flux_obs)}")
print(f"SNR: {float(jnp.mean(mock.flux_true / mock.noise)):.1f}")

# %%
# Plot the true SFH and photometry
fig = plt.figure(figsize=(13, 4.5))
gs = GridSpec(1, 2, width_ratios=[1.3, 1], wspace=0.3)

# --- Panel 1: True SFH ---
ax1 = fig.add_subplot(gs[0])
sfh_true = model.predict_sfh(true_params)
ax1.plot(sfh_true["t_gyr"], sfh_true["sfr_full"], color="k", lw=1.5,
         label="Full SFH (mean + GP)")
ax1.plot(sfh_true["t_gyr"], sfh_true["sfr_mean"], "k--", lw=1.0,
         alpha=0.4, label="Mean SFH (DPL backbone)")
ax1.set_xlabel("Lookback time [Gyr]")
ax1.set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
ax1.set_title(r"True SFH: $\sigma_{\rm PSD}=2.0$, $\tau_{\rm PSD}=20$ Myr")
ax1.set_xlim(0, 13.5)
ax1.set_ylim(bottom=0)
ax1.legend(fontsize=9)

# Text box with true parameters
textstr = "\n".join([
    r"$\sigma_{\rm PSD} = 2.0$",
    r"$\tau_{\rm PSD} = 20$ Myr",
    r"$\log Z/Z_\odot = -0.3$",
    r"$\tau_{\rm BC} = 0.5$",
    r"$\tau_{\rm diff} = 0.3$",
])
props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5)
ax1.text(0.97, 0.97, textstr, transform=ax1.transAxes, fontsize=8,
         verticalalignment="top", horizontalalignment="right", bbox=props)

# --- Panel 2: Mock photometry ---
ax2 = fig.add_subplot(gs[1])
ax2.errorbar(wave_eff, mock.flux_obs, yerr=mock.noise,
             fmt="o", ms=7, color="k", capsize=3, label="Observed (SNR=20)", zorder=5)
ax2.scatter(wave_eff, mock.flux_true, marker="D", s=40, facecolors="none",
            edgecolors="C3", linewidths=1.2, zorder=6, label="Truth (noiseless)")

for wl, fn in zip(wave_eff, filter_names):
    ax2.annotate(fn, (wl, float(mock.flux_obs[filter_names.index(fn)])),
                 textcoords="offset points", xytext=(0, 12),
                 ha="center", fontsize=8, color="0.4")

ax2.set_xlabel(r"Wavelength [$\AA$]")
ax2.set_ylabel(r"Flux [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax2.set_title("Mock SDSS + WISE Photometry")
ax2.set_xscale("log")
ax2.legend(fontsize=8)

plt.savefig("figures/19_mock_galaxy.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 3: EVI Recovery
#
# We fit the mock galaxy using two methods:
#
# 1. **MAP**: gradient-descent optimization to find the mode (500 steps)
# 2. **EVI** (Evidence Variational Inference): MGVI-based posterior
#    approximation with JIT-compiled Newton-CG optimization
#
# The EVI method jointly infers all $\sim$73 parameters (9 physical + 64
# GP latent) in a single variational optimization loop.

# %%
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

# --- MAP initialization ---
t0 = time.perf_counter()
result_map = fitter.run("map", n_steps=500, key=jax.random.PRNGKey(0))
t_map = time.perf_counter() - t0
print(f"\nMAP: {t_map:.1f}s")

# --- EVI inference ---
t0 = time.perf_counter()
result_evi = fitter.run(
    "evi",
    n_iterations=15,
    n_samples=4,
    n_posterior_samples=3000,
    key=jax.random.PRNGKey(1),
)
t_evi = time.perf_counter() - t0
print(f"EVI: {t_evi:.1f}s, {result_evi.diagnostics.get('n_iterations', '?')} iterations, "
      f"{result_evi.diagnostics.get('n_samples', '?')} posterior samples")

# %%
# Plot convergence: MAP loss history
if result_map.loss_history is not None:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.array(result_map.loss_history), color="k", lw=1.0)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (negative log posterior)")
    ax.set_title("MAP Convergence")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig("figures/19_map_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()

# %% [markdown]
# ## Section 4: SFH Recovery (The Money Figure)
#
# The key result: can the EVI posterior recover the true bursty SFH,
# including the location and amplitude of recent bursts?
#
# We plot the true SFH (black), posterior median (blue), and 68%/95%
# credible intervals. The dashed line shows the smooth (mean) SFH
# backbone, which by itself would miss all the burst features.

# %%
# Compute SFH draws from posterior
n_total = len(next(iter(result_evi.samples.values())))
sfh_draws = []
sfh_mean_draws = []
for i in range(n_total):
    s_i = {k: result_evi.samples[k][i] for k in result_evi.samples}
    sfh_i = model.predict_sfh(s_i)
    sfh_draws.append(sfh_i["sfr_full"])
    sfh_mean_draws.append(sfh_i["sfr_mean"])

sfh_arr = np.array(sfh_draws)
sfh_mean_arr = np.array(sfh_mean_draws)
t_gyr = np.array(model.predict_sfh(result_evi.params)["t_gyr"])

# Percentiles
median_sfh = np.median(sfh_arr, axis=0)
lo_68 = np.percentile(sfh_arr, 16, axis=0)
hi_68 = np.percentile(sfh_arr, 84, axis=0)
lo_95 = np.percentile(sfh_arr, 2.5, axis=0)
hi_95 = np.percentile(sfh_arr, 97.5, axis=0)

# True SFH
sfh_true = model.predict_sfh(true_params)
t_true = np.array(sfh_true["t_gyr"])
sfr_true = np.array(sfh_true["sfr_full"])
sfr_mean_true = np.array(sfh_true["sfr_mean"])

# %%
# --- THE MONEY FIGURE ---
fig = plt.figure(figsize=(12, 5.5))
gs = GridSpec(1, 2, width_ratios=[2.5, 1], wspace=0.05)

# --- Main panel: full SFH ---
ax_main = fig.add_subplot(gs[0])

# 95% CI
ax_main.fill_between(t_gyr, lo_95, hi_95, color="#2b6ca3", alpha=0.10,
                      edgecolor="none", label="95% CI", rasterized=True)
# 68% CI
ax_main.fill_between(t_gyr, lo_68, hi_68, color="#2b6ca3", alpha=0.25,
                      edgecolor="none", label="68% CI", rasterized=True)
# Posterior median
ax_main.plot(t_gyr, median_sfh, color="#2b6ca3", lw=1.8, label="Posterior median")
# Mean SFH backbone
ax_main.plot(t_true, sfr_mean_true, color="k", ls="--", lw=1.0, alpha=0.4,
             label="Mean SFH (backbone)")
# Truth
ax_main.plot(t_true, sfr_true, color="k", lw=2.2, label="Truth", zorder=10)

ax_main.set_xlabel("Lookback time [Gyr]")
ax_main.set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
ax_main.set_title("Stochastic SFH Recovery (EVI)")
ax_main.set_xlim(0, 13.5)
ax_main.set_ylim(bottom=0)
ax_main.legend(fontsize=9, loc="upper left")

# PSD parameter text box
textstr = "\n".join([
    r"$\sigma_{\rm PSD}^{\rm true} = 2.0$",
    r"$\tau_{\rm PSD}^{\rm true} = 20$ Myr",
    f"D = {D_total} parameters",
    f"EVI: {t_evi:.0f}s",
])
props = dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.5)
ax_main.text(0.97, 0.97, textstr, transform=ax_main.transAxes, fontsize=9,
             verticalalignment="top", horizontalalignment="right", bbox=props)

# --- Inset: zoom on last 200 Myr ---
ax_zoom = fig.add_subplot(gs[1])

mask_zoom = t_gyr < 0.2  # last 200 Myr
mask_true_zoom = t_true < 0.2

ax_zoom.fill_between(t_gyr[mask_zoom], lo_95[mask_zoom], hi_95[mask_zoom],
                      color="#2b6ca3", alpha=0.10, edgecolor="none", rasterized=True)
ax_zoom.fill_between(t_gyr[mask_zoom], lo_68[mask_zoom], hi_68[mask_zoom],
                      color="#2b6ca3", alpha=0.25, edgecolor="none", rasterized=True)
ax_zoom.plot(t_gyr[mask_zoom], median_sfh[mask_zoom],
             color="#2b6ca3", lw=1.8)
ax_zoom.plot(t_true[mask_true_zoom], sfr_true[mask_true_zoom],
             color="k", lw=2.2, zorder=10)
ax_zoom.plot(t_true[mask_true_zoom], sfr_mean_true[mask_true_zoom],
             color="k", ls="--", lw=1.0, alpha=0.4)

ax_zoom.set_xlabel("Lookback time [Gyr]")
ax_zoom.set_title("Last 200 Myr (zoom)", fontsize=11)
ax_zoom.set_xlim(0, 0.2)
ax_zoom.set_ylim(bottom=0)
ax_zoom.yaxis.tick_right()
ax_zoom.yaxis.set_label_position("right")

plt.savefig("figures/19_sfh_recovery_money.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 5: PSD Parameter Recovery
#
# The two PSD hyper-parameters control the burstiness of the SFH.
# $\sigma_{\mathrm{PSD}}$ (amplitude) is typically well-constrained
# from a single galaxy because the variance of the SFH fluctuations
# is visible in the integrated photometry. $\tau_{\mathrm{PSD}}$
# (timescale) is more weakly constrained because changing the
# correlation length while holding the variance fixed produces
# similar broadband colours.

# %%
# Corner plot: PSD parameters + dust
psd_params = ["sfh_field_psd_sigma", "sfh_field_psd_tau_myr"]
dust_params = ["dust_tau_bc", "dust_tau_diff"]
corner_params = psd_params + dust_params

psd_truths = {k: float(true_params[k]) for k in corner_params}

fig_corner = safe_corner(result_evi, params=corner_params, truths=psd_truths,
                         color="#2b6ca3", label="EVI")
if fig_corner is not None:
    plt.suptitle("PSD + Dust Parameter Recovery", y=1.02, fontsize=13)
    plt.savefig("figures/19_corner_psd_dust.png", dpi=150, bbox_inches="tight")
    plt.show()

# %%
# Quantitative recovery summary
print("Parameter recovery (EVI):")
print(f"{'Parameter':30s}  {'Truth':>8s}  {'Median':>8s}  {'68% CI':>18s}  {'Status':>8s}")
print("-" * 78)

for name in spec.free_params:
    if name in result_evi.samples and result_evi.samples[name].ndim == 1:
        samples = np.array(result_evi.samples[name])
        lo, med, hi = np.percentile(samples, [16, 50, 84])
        truth = float(true_params[name])
        covered = "OK" if lo <= truth <= hi else "MISS"
        print(f"  {name:30s}  {truth:8.3f}  {med:8.3f}  [{lo:7.3f}, {hi:7.3f}]  {covered:>8s}")

# %% [markdown]
# ## Section 6: Photometry Fit Quality
#
# Does the model actually fit the data? We overlay posterior predictive
# photometry draws on the observed data and examine the residuals.
# A good fit should have $\chi^2/N_{\mathrm{bands}} \approx 1$.

# %%
# Posterior predictive photometry
n_draw = min(100, n_total)
pred_draws = []
for j in range(n_draw):
    s_j = {k: result_evi.samples[k][j] for k in result_evi.samples}
    pred_j = model.predict_photometry(s_j)
    pred_draws.append(np.array(pred_j))
pred_draws = np.array(pred_draws)

median_pred = np.median(pred_draws, axis=0)
residuals = (np.array(mock.flux_obs) - median_pred) / np.array(mock.noise)
chi2_per_band = float(np.mean(residuals**2))

fig = plt.figure(figsize=(8, 5.5))
gs_fit = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)

# --- Top: photometry ---
ax_top = fig.add_subplot(gs_fit[0])
for draw in pred_draws[:50]:
    ax_top.plot(wave_eff, draw, "-", color="#2b6ca3", alpha=0.06, lw=0.8)

ax_top.errorbar(wave_eff, mock.flux_obs, yerr=mock.noise,
                fmt="o", ms=7, color="k", capsize=3, zorder=5, label="Observed")
ax_top.scatter(wave_eff, mock.flux_true, marker="D", s=40, facecolors="none",
               edgecolors="C3", linewidths=1.2, zorder=6, label="Truth")
ax_top.plot(wave_eff, median_pred, "s", ms=5, color="#2b6ca3",
            zorder=4, label="SEDModel (median)")

for wl, fn in zip(wave_eff, filter_names):
    ax_top.annotate(fn, (wl, float(mock.flux_obs[filter_names.index(fn)])),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, color="0.4")

ax_top.set_ylabel(r"Flux [erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax_top.set_title(f"Posterior Predictive Check ($\\chi^2/N = {chi2_per_band:.2f}$)")
ax_top.set_xscale("log")
ax_top.legend(fontsize=9, loc="upper right")
plt.setp(ax_top.get_xticklabels(), visible=False)

# --- Bottom: residuals ---
ax_bot = fig.add_subplot(gs_fit[1], sharex=ax_top)
ax_bot.axhline(0, color="0.5", ls="--", lw=0.8)
ax_bot.axhspan(-1, 1, alpha=0.05, color="0.5")
ax_bot.axhspan(-2, 2, alpha=0.03, color="0.5")

band_colors = ["#7b3294", "#008837", "#d73027", "#fc8d59", "#4575b4", "#a65628"]
ax_bot.bar(wave_eff, residuals, width=wave_eff * 0.08, color=band_colors, alpha=0.7)
ax_bot.set_xlabel(r"Wavelength [$\AA$]")
ax_bot.set_ylabel(r"Residual [$\sigma$]")
ax_bot.set_ylim(-4, 4)
ax_bot.set_xscale("log")

plt.savefig("figures/19_photometry_fit.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"chi2 / N_bands = {chi2_per_band:.2f}  (expect ~1)")

# %% [markdown]
# ## Section 7: Parametric vs Stochastic Comparison
#
# A critical test: fit the **same bursty mock** with a **parametric-only**
# model (no GP field). The smooth model cannot capture recent bursts,
# leading to systematic bias in the recovered SFH and derived quantities.
#
# This comparison demonstrates the fundamental motivation for the
# stochastic model: not a better $\chi^2$ (parametric models can still
# fit the broadband photometry), but correct physical inference of the
# star formation history.

# %%
# Define parametric-only model (no "field")
spec_param = ParamSpec(
    mean_sfh_type="dpl",
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

model_param = SEDModel(spec_param, ssp_data, filters=filters)
print(f"Parametric model: {spec_param.n_free} free parameters")
print(f"Stochastic model: {spec.n_free} + {spec.n_grid} = {D_total} total parameters")

# Fit the same mock data with parametric model
fitter_param = Fitter(model_param, mock.flux_obs, mock.noise, data_type="photometry")

t0 = time.perf_counter()
result_map_param = fitter_param.run("map", n_steps=500, key=jax.random.PRNGKey(10))
result_rt_param = fitter_param.run(
    "raytrace", init_from=result_map_param,
    n_burnin=100, n_steps=500,
    key=jax.random.PRNGKey(11),
)
t_param = time.perf_counter() - t0
print(f"Parametric fit: {t_param:.1f}s")

# %%
# Compute SFH draws for parametric model
n_param_samples = len(next(iter(result_rt_param.samples.values())))
sfh_param_draws = []
for i in range(n_param_samples):
    s_i = {k: result_rt_param.samples[k][i] for k in result_rt_param.samples}
    sfh_i = model_param.predict_sfh(s_i)
    sfh_param_draws.append(sfh_i["sfr_mean"])

sfh_param_arr = np.array(sfh_param_draws)
t_gyr_param = np.array(model_param.predict_sfh(result_rt_param.params)["t_gyr"])

median_param = np.median(sfh_param_arr, axis=0)
lo_68_param = np.percentile(sfh_param_arr, 16, axis=0)
hi_68_param = np.percentile(sfh_param_arr, 84, axis=0)

# %%
# --- Comparison figure ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

# --- Left: Parametric (wrong model) ---
ax = axes[0]
ax.fill_between(t_gyr_param, lo_68_param, hi_68_param,
                color="#c03d3e", alpha=0.20, edgecolor="none",
                label="Parametric 68% CI", rasterized=True)
ax.plot(t_gyr_param, median_param, color="#c03d3e", lw=1.8,
        label="Parametric median")
ax.plot(t_true, sfr_true, color="k", lw=2.2, label="Truth", zorder=10)
ax.plot(t_true, sfr_mean_true, "k--", lw=1.0, alpha=0.4)

ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]")
ax.set_title("Parametric SEDModel (misses bursts)", fontsize=12)
ax.set_xlim(0, 13.5)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9, loc="upper left")

# --- Right: Stochastic (correct model) ---
ax = axes[1]
ax.fill_between(t_gyr, lo_68, hi_68,
                color="#2b6ca3", alpha=0.20, edgecolor="none",
                label="Stochastic 68% CI", rasterized=True)
ax.plot(t_gyr, median_sfh, color="#2b6ca3", lw=1.8,
        label="Stochastic median")
ax.plot(t_true, sfr_true, color="k", lw=2.2, label="Truth", zorder=10)
ax.plot(t_true, sfr_mean_true, "k--", lw=1.0, alpha=0.4)

ax.set_xlabel("Lookback time [Gyr]")
ax.set_title("Stochastic SEDModel (recovers bursts)", fontsize=12)
ax.set_xlim(0, 13.5)
ax.legend(fontsize=9, loc="upper left")

plt.suptitle("Parametric vs Stochastic SFH Recovery", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("figures/19_parametric_vs_stochastic.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
# Quantitative comparison: photometry chi2
pred_param_draws = []
n_draw_param = min(100, n_param_samples)
for j in range(n_draw_param):
    s_j = {k: result_rt_param.samples[k][j] for k in result_rt_param.samples}
    pred_j = model_param.predict_photometry(s_j)
    pred_param_draws.append(np.array(pred_j))
pred_param_draws = np.array(pred_param_draws)

median_pred_param = np.median(pred_param_draws, axis=0)
residuals_param = (np.array(mock.flux_obs) - median_pred_param) / np.array(mock.noise)
chi2_param = float(np.mean(residuals_param**2))

print(f"\nPhotometry fit quality:")
print(f"  Parametric:  chi2/N = {chi2_param:.2f}")
print(f"  Stochastic:  chi2/N = {chi2_per_band:.2f}")
print(f"\nNote: Both can fit the broadband photometry well.")
print(f"The difference is in the *physical* inference:")
print(f"  - Parametric: smooth SFH that misses recent bursts")
print(f"  - Stochastic: recovers burst timing and amplitude")

# %% [markdown]
# ## Summary
#
# | Aspect | Result |
# |--------|--------|
# | **SFH shape** | EVI recovers the full bursty SFH within 68% CI, including recent burst features |
# | **$\sigma_{\rm PSD}$** | Well-constrained from a single galaxy (amplitude visible in SFH scatter) |
# | **$\tau_{\rm PSD}$** | Weakly constrained per-galaxy (timescale degeneracy); motivates hierarchical inference |
# | **Dust parameters** | Recovered within 68% CI alongside the SFH |
# | **Photometry fit** | Both parametric and stochastic achieve $\chi^2/N \approx 1$ |
# | **SEDModel mismatch** | Parametric model fits the photometry but misses burst features, biasing SFR recovery |
# | **Timing** | EVI converges in $\sim$15 iterations for 73 parameters; total wall time $\sim$1--2 min |
#
# **Key takeaway:** Broadband photometry alone *cannot* distinguish smooth from
# bursty SFHs based on $\chi^2$. The stochastic model's advantage is in
# **physical inference**: it correctly recovers burst features and their
# uncertainties, while the parametric model produces biased SFH estimates
# that miss recent star formation episodes.
#
# **Next steps:**
# - [Tutorial 05](05_hierarchical.ipynb): Hierarchical inference to constrain $\tau_{\rm PSD}$ across a population
# - [Tutorial 08](08_psd_physics.ipynb): Physical interpretation of PSD parameters
