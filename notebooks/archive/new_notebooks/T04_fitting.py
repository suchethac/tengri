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
# # Fitting Galaxies: Photometry and Spectroscopy
#
# This is the practical fitting tutorial.  We fit the **same mock galaxy**
# with broadband photometry alone and then with a $R=100$ spectrum,
# showing how the posterior tightens — especially in the last 200 Myr
# where burstiness lives.
#
# **By the end you will:**
# 1. Generate mock photometry and spectra with `model.mock()`
# 2. Fit with the stochastic GP-PSD model
# 3. See the information gain from photometry → spectroscopy
# 4. Understand PSD parameter constraints from individual galaxies
# 5. Run posterior predictive checks

# %%
import time, os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)
import sys; sys.path.insert(0, ".")
import sys; sys.path.insert(0, "..")
from _plot_style import (setup_style, COLORS, SDSS_WAVE_EFF,
                          plot_sfh, plot_sed_fit, plot_spectrum_fit,
                          plot_corner_comparison, convergence_check,
                          SPECTRAL_FEATURES)
setup_style()

FIG_DIR = "../notebook_figures"
os.makedirs(FIG_DIR, exist_ok=True)
def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"T04_{name}.png"),
                bbox_inches="tight", dpi=72)

ssp_data = load_ssp_data(
    "../../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])


# ── SFH inset helper ─────────────────────────────────────────────
def add_sfh_inset(ax, t_gyr, sfr, inset_range_myr=200, **kw):
    ax_in = inset_axes(ax, width="35%", height="40%", loc="upper right",
                       borderpad=1.5)
    t_myr = np.asarray(t_gyr) * 1e3
    mask = t_myr <= inset_range_myr
    if mask.sum() > 2:
        ax_in.plot(t_myr[mask], np.asarray(sfr)[mask], **kw)
    ax_in.set_xlim(0, inset_range_myr)
    ax_in.set_xlabel("Myr", fontsize=7, labelpad=1)
    ax_in.tick_params(labelsize=6)
    ax_in.axvspan(0, 10, alpha=0.08, color="blue", zorder=0)
    ax_in.axvspan(0, 100, alpha=0.04, color="purple", zorder=0)
    ax_in.set_title("last 200 Myr", fontsize=7, pad=2)
    for sp in ax_in.spines.values():
        sp.set_linewidth(0.6)
    return ax_in


# %% [markdown]
# ## 1. Mock Generation
#
# We generate a moderately bursty galaxy at $z = 0.1$ with both
# SDSS-like photometry and an $R = 100$ rest-frame spectrum.

# %%
spec = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0), sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0), sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Uniform(0.1, 4.0), psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0), dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7), redshift=Fixed(0.1),
    stochastic=True, n_grid=128,
)
model = SEDModel(spec, ssp_data, filters=filters)

key = jax.random.PRNGKey(2026)
true_params = spec.sample(key)
true_params.update(
    sfh_alpha=1.0, sfh_beta=1.5, sfh_tau_peak_gyr=8.0,
    sfh_peak_sfr=30.0, psd_sigma=1.5, psd_tau_myr=30.0,
    met_logzsol=-0.3, dust_tau_bc=0.5, dust_tau_diff=0.3,
)

# Generate both data types
mock_phot = model.mock(true_params, snr=20.0, key=key)

# For spectroscopy, generate an observed-frame wavelength grid at R~100
z_obs = 0.1
wave_rest = jnp.linspace(1000, 8000, 500)
wave_spec_obs = wave_rest * (1.0 + z_obs)
spec_true = model.predict_spectrum(true_params, wave_spec_obs)
key, noise_key = jax.random.split(key)
spec_noise = jnp.abs(spec_true) / 20.0  # SNR = 20
spec_obs = spec_true + spec_noise * jax.random.normal(noise_key, shape=spec_true.shape)

print(f"Photometry: {mock_phot.flux_obs.shape[0]} bands")
print(f"Spectroscopy: {spec_obs.shape[0]} pixels")

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].errorbar(SDSS_WAVE_EFF, mock_phot.flux_obs, yerr=mock_phot.noise,
                 fmt="o", color="k", ms=6, capsize=3, label="Photometry (SNR 20)")
axes[0].plot(SDSS_WAVE_EFF, mock_phot.flux_true, "s", ms=7, mfc="none",
             color="C3", label="Truth")
axes[0].set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
axes[0].set_ylabel("Flux"); axes[0].set_title("SDSS Photometry"); axes[0].legend(fontsize=8)

axes[1].plot(np.array(wave_spec_obs), np.array(spec_obs), color="0.5", lw=0.5)
axes[1].plot(np.array(wave_spec_obs), np.array(spec_true), color="C3", lw=1.0,
             label="Truth")
axes[1].set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
axes[1].set_ylabel("Flux"); axes[1].set_title(r"$R=100$ Spectrum"); axes[1].legend(fontsize=8)

fig.tight_layout(); savefig(fig, "mock_data"); plt.show()

# %% [markdown]
# ## 2. Fitting Photometry Alone

# %%
fitter_phot = Fitter(model, mock_phot.flux_obs, mock_phot.noise,
                     data_type="photometry")
map_phot = fitter_phot.run("map", n_steps=1000)
rt_phot = fitter_phot.run("raytrace", init_from=map_phot,
                          n_burnin=200, n_steps=2000,
                          step_size=0.005, n_leapfrog_steps=200)
convergence_check(rt_phot, "RT (photometry)")

# %% [markdown]
# ## 3. Fitting Spectroscopy

# %%
model._wave_obs = wave_spec_obs  # tell model the wavelength grid for spectroscopy
fitter_spec = Fitter(model, spec_obs, spec_noise,
                     data_type="spectroscopy")
map_spec = fitter_spec.run("map", n_steps=1000)
rt_spec = fitter_spec.run("raytrace", init_from=map_spec,
                          n_burnin=200, n_steps=2000,
                          step_size=0.005, n_leapfrog_steps=200)
convergence_check(rt_spec, "RT (spectroscopy)")

# %% [markdown]
# ## 4. Information Gain: Photometry → Spectroscopy
#
# The key comparison.  The main panel shows the full lookback-time SFH;
# the inset zooms into the last 200 Myr where burstiness matters most.

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Panel 1: photometry only
ax = axes[0]
plot_sfh(model, rt_phot, true_params=true_params, ax=ax, method="RT",
         label="Phot only", color=COLORS["rt"])
ax.set_title("Photometry Only", fontsize=11)
sfh_true = model.predict_sfh(true_params)
add_sfh_inset(ax, sfh_true["t_gyr"], sfh_true["sfr_full"],
              color=COLORS["truth"], lw=2)

# Panel 2: spectroscopy
ax = axes[1]
plot_sfh(model, rt_spec, true_params=true_params, ax=ax, method="RT",
         label="Spectroscopy", color="#d62728")
ax.set_title("Spectroscopy ($R=100$)", fontsize=11)
add_sfh_inset(ax, sfh_true["t_gyr"], sfh_true["sfr_full"],
              color=COLORS["truth"], lw=2)

# Panel 3: overlaid
ax = axes[2]
plot_sfh(model, rt_phot, true_params=true_params, ax=ax,
         label="Phot", color=COLORS["rt"], show_draws=False)
plot_sfh(model, rt_spec, ax=ax,
         label="Spec", color="#d62728", show_draws=False)
ax.set_title("Overlaid: Phot vs. Spec", fontsize=11)
add_sfh_inset(ax, sfh_true["t_gyr"], sfh_true["sfr_full"],
              color=COLORS["truth"], lw=2)

fig.suptitle("Progressive SFH Constraints: Photometry → Spectroscopy",
             fontsize=13, y=1.03)
fig.tight_layout(); savefig(fig, "information_gain"); plt.show()

# %% [markdown]
# ## 5. PSD Parameter Constraints
#
# Can a single galaxy constrain burstiness?  The joint posterior on
# $(\sigma_{\rm PS}, \tau_{\rm PS})$ from photometry vs. spectroscopy.

# %%
fig = plot_corner_comparison(
    [rt_phot, rt_spec],
    ["Photometry", "Spectroscopy"],
    colors=[COLORS["rt"], "#d62728"],
    truths=true_params,
    params=["psd_sigma", "psd_tau_myr"],
)
if fig is not None:
    fig.suptitle(r"PSD Constraints: $(\sigma_{\rm PS}, \tau_{\rm PS})$",
                 fontsize=12, y=1.02)
    savefig(fig, "psd_corner_phot_vs_spec")
plt.show()

# %% [markdown]
# ## 6. Posterior Predictive Check
#
# Does the posterior reproduce the observed data?  We draw model SEDs
# from the posterior and compare to the observations.

# %%
# Generate posterior predictive draws manually
n_draws = min(50, len(list(rt_phot.samples.values())[0]))
pp_draws = []
for i in range(n_draws):
    draw_params = {k: v[i] for k, v in rt_phot.samples.items()}
    pp_draws.append(np.array(model.predict_photometry(draw_params)))
pp_draws = np.array(pp_draws)

fig = plot_sed_fit(SDSS_WAVE_EFF, mock_phot.flux_obs, mock_phot.noise,
                   flux_true=mock_phot.flux_true,
                   posterior_draws=pp_draws)
fig.suptitle("Posterior Predictive Check — Photometry", fontsize=12, y=1.02)
savefig(fig, "ppc_photometry"); plt.show()

# %% [markdown]
# ## Summary
#
# | Data type | Recent SFH | $\sigma_{\rm PS}$ | $\tau_{\rm PS}$ |
# |-----------|-----------|-------------------|-----------------|
# | Photometry | Weakly constrained | Moderate | Prior-dominated |
# | Spectroscopy ($R=100$) | Well constrained | Good | Moderate |
#
# Spectroscopy adds information through emission lines (Hα, Hβ), the
# Balmer break, and continuum shape — features that respond on different
# lookback timescales, jointly constraining the PSD.
#
# For population-level $\tau_{\rm PS}$ constraints, see
# [T05 — Hierarchical](T05_hierarchical.ipynb).
