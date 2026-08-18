# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Checking Your SEDModel Before Fitting
#
# Before fitting, ask: does my model produce plausible galaxies? A prior
# predictive check samples from the prior and runs the forward model. If the
# predicted SEDs don't look like real galaxies, the priors need adjustment —
# no amount of inference can fix a bad model.

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fixed,
    SEDModel,
    Observation,
    Parameters,
    Photometry,
    Uniform,
    load_ssp_data,
)

import sys, os  # noqa: E401
try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
# Change to project root so data/ paths work
# chdir to project root for data/ access
if os.path.exists("data"):
    pass  # already in project root
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

FIGDIR = os.path.join("tutorials", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import COLORS, setup_style

setup_style()

# %%
ssp_data = load_ssp_data(
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)

# %%
# Good prior
spec_good = Parameters(
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model_good = SEDModel(spec_good, ssp_data, observation=obs)
model_good.precompute_spectroscopy(WAVE_OBS)

# %%
# Sample 200 parameter sets from prior and forward-model each
N_PRIOR = 200
keys = jax.random.split(jax.random.PRNGKey(0), N_PRIOR)
prior_spectra = []
prior_phot = []
prior_sfh = []
prior_params_list = []

for i in range(N_PRIOR):
    p = spec_good.sample(keys[i])
    prior_params_list.append(p)
    spec_i = model_good.predict_spectrum(p)
    phot_i = model_good.predict_photometry(p)
    sfh_i = model_good.predict_sfh(p)
    prior_spectra.append(np.array(spec_i))
    prior_phot.append(np.array(phot_i))
    prior_sfh.append(sfh_i)

prior_spectra = np.array(prior_spectra)
prior_phot = np.array(prior_phot)

# %%
# --- FIGURE 1: 200 prior-predictive SEDs ---
fig, ax = plt.subplots(figsize=(10, 4))

# Color by u-r color
u_flux = prior_phot[:, 0]
r_flux = prior_phot[:, 2]
ur_color = -2.5 * np.log10(np.clip(u_flux / r_flux, 1e-10, None))
ur_norm = (ur_color - np.percentile(ur_color, 5)) / (np.percentile(ur_color, 95) - np.percentile(ur_color, 5))
ur_norm = np.clip(ur_norm, 0, 1)

cmap = plt.cm.RdYlBu_r
for i in range(N_PRIOR):
    norm_spec = prior_spectra[i] / np.median(prior_spectra[i])
    ax.plot(np.array(WAVE_OBS), norm_spec, color=cmap(ur_norm[i]), alpha=0.15, lw=0.5)

sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=np.percentile(ur_color, 5),
                                                           vmax=np.percentile(ur_color, 95)))
plt.colorbar(sm, ax=ax, label="u − r color")
ax.set_xlabel("Observed wavelength [Å]")
ax.set_ylabel("Normalized flux")
ax.set_title(f"Prior Predictive: {N_PRIOR} Spectra (colored by u−r)")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig01_prior_predictive_spectra.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 2: Prior-predictive color-color diagram ---
g_flux = prior_phot[:, 1]
ug = -2.5 * np.log10(np.clip(u_flux / g_flux, 1e-10, None))
gr = -2.5 * np.log10(np.clip(g_flux / r_flux, 1e-10, None))

fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(gr, ug, s=5, alpha=0.5, color=COLORS["vi"])
ax.set_xlabel("g − r")
ax.set_ylabel("u − g")
ax.set_title("Prior Predictive Color–Color Diagram")
# Typical SDSS galaxy locus (approximate)
ax.plot([0.0, 0.3, 0.6, 0.9, 1.2], [0.5, 0.8, 1.2, 1.6, 2.0],
        "k--", lw=1, alpha=0.5, label="Approx. SDSS locus")
ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig02_prior_color_color.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 3: Prior-predictive SFH distribution ---
fig, ax = plt.subplots(figsize=(8, 4))
for i in range(min(50, N_PRIOR)):
    t_gyr = np.array(prior_sfh[i]["t_gyr"])
    sfr = np.array(prior_sfh[i]["sfr_mean"])
    ax.plot(t_gyr, sfr, color=COLORS["vi"], alpha=0.15, lw=0.5)
ax.set_xlim(0, 13.5)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.set_title(f"Prior Predictive SFHs (50 of {N_PRIOR})")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig03_prior_sfhs.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Diagnosing Bad Priors
#
# An intentionally bad configuration: dust τ_bc prior too wide (0, 10)
# instead of (0, 2). Many galaxies become absurdly extinguished.

# %%
# Bad prior: dust way too wide
spec_bad = Parameters(
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 10.0),   # BAD: way too wide!
    dust_tau_diff=Uniform(0.0, 5.0),  # BAD: way too wide!
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="tsnorm",
)
model_bad = SEDModel(spec_bad, ssp_data, observation=obs)
model_bad.precompute_spectroscopy(WAVE_OBS)

bad_spectra = []
bad_phot = []
for i in range(N_PRIOR):
    p = spec_bad.sample(keys[i])
    bad_spectra.append(np.array(model_bad.predict_spectrum(p)))
    bad_phot.append(np.array(model_bad.predict_photometry(p)))
bad_spectra = np.array(bad_spectra)
bad_phot = np.array(bad_phot)

# %%
# --- FIGURE 4: Bad prior revealed ---
fig, (ax_sed, ax_cc) = plt.subplots(1, 2, figsize=(12, 4))

# SEDs
for i in range(min(100, N_PRIOR)):
    norm = bad_spectra[i] / np.median(bad_spectra[i]) if np.median(bad_spectra[i]) > 0 else bad_spectra[i]
    ax_sed.plot(np.array(WAVE_OBS), norm, color=COLORS["model"], alpha=0.1, lw=0.5)
ax_sed.set_xlabel("Observed wavelength [Å]")
ax_sed.set_ylabel("Normalized flux")
ax_sed.set_title("Bad Prior: Many Absurdly Extinguished SEDs")

# Color-color
u_bad = bad_phot[:, 0]
g_bad = bad_phot[:, 1]
r_bad = bad_phot[:, 2]
ug_bad = -2.5 * np.log10(np.clip(u_bad / g_bad, 1e-10, None))
gr_bad = -2.5 * np.log10(np.clip(g_bad / r_bad, 1e-10, None))

ax_cc.scatter(gr_bad, ug_bad, s=5, alpha=0.3, color=COLORS["model"], label="Bad prior")
ax_cc.scatter(gr, ug, s=5, alpha=0.3, color=COLORS["vi"], label="Good prior")
ax_cc.plot([0.0, 0.3, 0.6, 0.9, 1.2], [0.5, 0.8, 1.2, 1.6, 2.0],
           "k--", lw=1, alpha=0.5, label="SDSS locus")
ax_cc.set_xlabel("g − r")
ax_cc.set_ylabel("u − g")
ax_cc.legend(fontsize=7)
ax_cc.set_title("Color–Color: Bad Prior vs Good Prior")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig04_bad_prior.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Stochastic SEDModel Prior Predictive
#
# How do PSD priors affect the burstiness range?

# %%
spec_stoch = Parameters(
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    sfh_field_psd_sigma=Uniform(0.1, 4.0),
    sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type=["tsnorm", "field"],
    n_grid=128,
)
model_stoch = SEDModel(spec_stoch, ssp_data, observation=obs)

# %%
# --- FIGURE 5: Stochastic prior SFHs ---
fig, ax = plt.subplots(figsize=(8, 4))
for i in range(20):
    p = spec_stoch.sample(jax.random.PRNGKey(i + 100))
    sfh_i = model_stoch.predict_sfh(p)
    t = np.array(sfh_i["t_gyr"])
    sfr = np.array(sfh_i["sfr_full"])
    sigma_i = float(p["sfh_field_psd_sigma"])
    color = plt.cm.viridis(sigma_i / 4.0)
    ax.plot(t, sfr, color=color, alpha=0.5, lw=0.8)

ax.set_xlim(0, 13.5)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.set_title("Stochastic Prior Predictive SFHs (colored by σ_PS)")

sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, 4))
plt.colorbar(sm, ax=ax, label=r"$\sigma_{\rm PS}$")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig05_stochastic_prior_sfhs.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# The prior predictive check costs seconds and catches configuration errors
# before you waste hours fitting. Make it a habit:
#
# 1. **Sample from the prior** — do the SEDs look like real galaxies?
# 2. **Check color–color** — do they fall on the observed locus?
# 3. **Inspect SFHs** — are they astrophysically reasonable?
# 4. **Diagnose bad priors** — if predictions are pathological, fix the prior.
#
# This completes the tutorial track. You're now ready for the
# **demonstrations/** and **reference/** notebooks.
