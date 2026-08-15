# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # From SFH to Observable SED
#
# The forward model maps latent parameters → SFH → stellar spectrum → dust
# attenuation → redshifted observables. Every step is differentiable. This
# notebook walks through the full pipeline and shows the Jacobian — computed
# for free via JAX autodiff.

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
from tengri.sps.dsps_wrapper import compute_csp_weights

import sys, os  # noqa: E401, E402
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

from _plot_style import COLORS, SPECTRAL_FEATURES, setup_style  # noqa: E402

setup_style()

# %%
ssp_data = load_ssp_data(
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))

# %% [markdown]
# ## SSP Building Blocks
#
# Simple Stellar Populations (SSPs) are the atoms of SED modeling: a single
# burst of star formation at one age and one metallicity. The galaxy spectrum
# is a weighted sum of SSPs.

# %%
# --- FIGURE 1: SSP spectra at 5 ages ---
ages_idx = [0, 20, 40, 60, 80]  # sample indices
age_labels = ["1 Myr", "10 Myr", "100 Myr", "1 Gyr", "10 Gyr"]
met_idx = ssp_data.ssp_flux.shape[0] // 2  # approximately solar

fig, ax = plt.subplots(figsize=(10, 4))
wavelengths = np.array(ssp_data.ssp_wave)

for i, (aidx, label) in enumerate(zip(ages_idx, age_labels)):
    if aidx < ssp_data.ssp_flux.shape[1]:
        flux = np.array(ssp_data.ssp_flux[met_idx, aidx, :])
        flux_norm = flux / np.median(flux[flux > 0]) if np.any(flux > 0) else flux
        ax.plot(wavelengths, flux_norm, lw=0.8, alpha=0.8, label=label)

# Annotate features
for feat_name, feat_wave in SPECTRAL_FEATURES.items():
    if 1000 < feat_wave < 10000:
        ax.axvline(feat_wave, color="gray", ls=":", lw=0.3, alpha=0.5)
        ax.text(feat_wave, ax.get_ylim()[1] * 0.9, feat_name,
                fontsize=5, ha="center", rotation=90, color="gray")

ax.set_xlabel("Rest-frame wavelength [Å]")
ax.set_ylabel("Normalized flux")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(900, 30000)
ax.legend(fontsize=8, title="Age")
ax.set_title("Simple Stellar Populations at Solar Metallicity")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig01_ssp_spectra.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## CSP Construction
#
# The Composite Stellar Population integrates SSPs weighted by the SFH:
# $f_\lambda^{\rm CSP} = \int_0^{t_{\rm cosmic}} \dot{M}_\star(t)\,f_\lambda^{\rm SSP}(t, Z)\,dt$

# %%
# Create a model and compute SFH
spec = Parameters(
    sfh_tsnorm_log_total_mass=Fixed(10.0),
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
model = SEDModel(spec, ssp_data, observation=obs)
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)
model.precompute_spectroscopy(WAVE_OBS)

params = spec.sample(jax.random.PRNGKey(42))
# Override tsnorm to a typical star-forming galaxy (still forming stars now)
params = {**params}
params["sfh_tsnorm_log_total_mass"] = jnp.array(1.2)
params["sfh_tsnorm_peak_lbt_gyr"] = jnp.array(3.0)
params["sfh_tsnorm_width_gyr"] = jnp.array(3.0)
params["sfh_tsnorm_skew"] = jnp.array(0.3)
params["sfh_tsnorm_trunc"] = jnp.array(2.0)
sfh = model.predict_sfh(params)

# %%
# --- FIGURE 2: CSP assembly (1×3) ---
fig, (ax_sfh, ax_weights, ax_csp) = plt.subplots(1, 3, figsize=(15, 4))

t_gyr = np.array(sfh["t_gyr"])
sfr = np.array(sfh["sfr_mean"])

ax_sfh.plot(t_gyr, sfr, color=COLORS["truth"], lw=1.5)
ax_sfh.set_xlabel("Lookback time [Gyr]")
ax_sfh.set_ylabel(r"SFR [$M_\odot$/yr]")
ax_sfh.set_xlim(0, 13.5)
ax_sfh.set_title("(1) Star Formation History")

# Weights = SFR × Δt (contribution of each age bin)
ages_log = np.array(ssp_data.ssp_lg_age_gyr)
ages_gyr = 10**ages_log / 1e9
ax_weights.loglog(ages_gyr, np.abs(sfr[:len(ages_gyr)]) + 1e-10, color=COLORS["sfh_mean"], lw=1.2)
ax_weights.set_xlabel("Lookback time [Gyr]")
ax_weights.set_ylabel("Weight (log)")
ax_weights.set_title("(2) Age Weights")

# CSP = weighted sum
sed = model.predict_spectrum(params)
ax_csp.plot(np.array(WAVE_OBS), np.array(sed), color=COLORS["model"], lw=1)
ax_csp.set_xlabel("Observed wavelength [Å]")
ax_csp.set_ylabel("Flux density")
ax_csp.set_title("(3) Composite Spectrum")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig02_csp_assembly.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Dust Attenuation
#
# Charlot & Fall (2000): two-component model. Young stars (< 10 Myr) see
# both birth cloud + diffuse ISM dust. Older stars see only diffuse.
# $A_\lambda = -2.5 \log_{10}[w(t) \cdot e^{-\tau_{\rm BC}(\lambda/5500)^n} + (1-w(t)) \cdot e^{-\tau_{\rm diff}(\lambda/5500)^n}]$

# %%
# --- FIGURE 3: Dust effects (1×3) ---
fig, (ax_curve, ax_trans, ax_sed) = plt.subplots(1, 3, figsize=(15, 4))

# Attenuation curves
wave_rest = np.linspace(1000, 10000, 500)
for tau_v, label, color in [
    (0.5, "τ = 0.5", COLORS["seq"][0]),
    (1.0, "τ = 1.0", COLORS["seq"][2]),
    (2.0, "τ = 2.0", COLORS["seq"][3]),
    (3.0, "τ = 3.0", COLORS["seq"][4]),
]:
    atten = np.exp(-tau_v * (wave_rest / 5500.0)**(-0.7))
    ax_curve.plot(wave_rest, atten, color=color, lw=1.2, label=label)

ax_curve.set_xlabel("Wavelength [Å]")
ax_curve.set_ylabel("Transmission")
ax_curve.legend(fontsize=7)
ax_curve.set_title("(1) Attenuation Curves")

# Birth cloud transition: sigmoid at ~10 Myr
ages_log_dust = np.linspace(5, 11, 100)
ages_myr = 10**ages_log_dust / 1e6
# Sigmoid transition: w(t) = 1 / (1 + exp((log10(t) - 7) / 0.3))
w = 1.0 / (1.0 + np.exp((ages_log_dust - 7.0) / 0.3))
ax_trans.plot(ages_myr, w, color=COLORS["sfh_mean"], lw=1.5)
ax_trans.axvline(10, color="gray", ls="--", lw=0.8, label="10 Myr")
ax_trans.set_xlabel("Age [Myr]")
ax_trans.set_ylabel("Birth cloud weight w(t)")
ax_trans.set_xscale("log")
ax_trans.legend(fontsize=8)
ax_trans.set_title("(2) Birth Cloud Transition")

# Dusty vs dust-free SED
params_nodust = {**params}
params_nodust["dust_tau_bc"] = jnp.array(0.0)
params_nodust["dust_tau_diff"] = jnp.array(0.0)
sed_nodust = model.predict_spectrum(params_nodust)
sed_dusty = model.predict_spectrum(params)

ax_sed.plot(np.array(WAVE_OBS), np.array(sed_nodust), color=COLORS["seq"][2],
            lw=1, label="Intrinsic", alpha=0.8)
ax_sed.plot(np.array(WAVE_OBS), np.array(sed_dusty), color=COLORS["seq"][4],
            lw=1, label="Attenuated")
ax_sed.set_xlabel("Observed wavelength [Å]")
ax_sed.set_ylabel("Flux density")
ax_sed.legend(fontsize=8)
ax_sed.set_title("(3) Dust Effect on SED")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig03_dust_effects.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Metallicity Effects

# %%
# --- FIGURE 4: Metallicity effects ---
fig, ax = plt.subplots(figsize=(8, 4))
for logz, label, color in [
    (-2.0, "log Z = -2.0", COLORS["seq"][0]),
    (-1.0, "log Z = -1.0", COLORS["seq"][2]),
    (-0.3, "log Z = -0.3 (solar)", COLORS["seq"][3]),
    (0.2, "log Z = +0.2", COLORS["seq"][4]),
]:
    p = {**params}
    p["met_logzsol"] = jnp.array(logz)
    p["dust_tau_bc"] = jnp.array(0.0)
    p["dust_tau_diff"] = jnp.array(0.0)
    sed = model.predict_spectrum(p)
    sed_norm = np.array(sed) / np.median(np.array(sed))
    ax.plot(np.array(WAVE_OBS), sed_norm, lw=1, label=label, color=color)

ax.set_xlabel("Observed wavelength [Å]")
ax.set_ylabel("Normalized flux")
ax.legend(fontsize=8)
ax.set_title("Metallicity Effect on Spectrum (no dust)")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig04_metallicity.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## SED → Photometry

# %%
# --- FIGURE 5: SED with filter curves and photometric points ---
# SDSS effective wavelengths (Angstrom)

fig, ax = plt.subplots(figsize=(10, 4))
sed_full = model.predict_spectrum(params)
phot = model.predict_photometry(params)

ax.plot(np.array(WAVE_OBS), np.array(sed_full), color=COLORS["model"], lw=1, alpha=0.7, label="Spectrum")

wave_eff = np.array([3551, 4686, 6166, 7480, 8932])
band_colors = [COLORS["u"], COLORS["g"], COLORS["r"], COLORS["i"], COLORS["z"]]
band_names = ["u", "g", "r", "i", "z"]
for w, f, c, n in zip(wave_eff, np.array(phot), band_colors, band_names):
    ax.scatter(w, f, s=80, color=c, zorder=5, edgecolors="k", linewidths=0.5)
    ax.text(w, f * 1.05, n, ha="center", fontsize=9, fontweight="bold", color=c)

ax.set_xlabel("Observed wavelength [Å]")
ax.set_ylabel("Flux density")
ax.set_title("Spectrum → Photometry via Filter Convolution")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig05_sed_to_photometry.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Complete Pipeline

# %%
# --- FIGURE 6: Complete pipeline (1×3) ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# SFH
ax1.plot(t_gyr, sfr, color=COLORS["truth"], lw=1.5)
ax1.set_xlabel("Lookback time [Gyr]")
ax1.set_ylabel(r"SFR [$M_\odot$/yr]")
ax1.set_xlim(0, 13.5)
ax1.set_title("SFH")

# SED
ax2.plot(np.array(WAVE_OBS), np.array(sed_full), color=COLORS["model"], lw=1)
ax2.set_xlabel("Observed wavelength [Å]")
ax2.set_ylabel("Flux density")
ax2.set_title("Spectrum")

# Photometry
ax3.errorbar(wave_eff, np.array(phot), yerr=np.array(phot) * 0.05,
             fmt="o", color=COLORS["data"], ms=8, capsize=3)
for w, f, c, n in zip(wave_eff, np.array(phot), band_colors, band_names):
    ax3.text(w, f * 1.08, n, ha="center", fontsize=9, color=c)
ax3.set_xlabel("Wavelength [Å]")
ax3.set_ylabel("Flux density")
ax3.set_title("Photometry (5 bands)")

fig.suptitle("The Complete Forward SEDModel Pipeline", fontsize=12)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig06_complete_pipeline.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## The Jacobian: Why Differentiability Matters
#
# Because everything is JAX, we get ∂f/∂θ for free. This matrix shows which
# observables are sensitive to which parameters — revealing degeneracies and
# informing survey design.

# %%
# --- FIGURE 7: Jacobian heatmap ---
# Compute Jacobian of photometry w.r.t. free params
def phot_from_params(p):
    return model.predict_photometry(p)

J = jax.jacobian(phot_from_params)(params)

# Extract Jacobian into a matrix
param_names = spec.free_params
n_params = len(param_names)
phot_val = model.predict_photometry(params)
n_bands = len(phot_val)
band_labels = ["u", "g", "r", "i", "z"][:n_bands]

J_matrix = np.zeros((n_bands, n_params))
for j, name in enumerate(param_names):
    jval = np.array(J[name]).flatten()
    J_matrix[:, j] = jval[:n_bands]

# Normalize each column
for j in range(n_params):
    col_max = np.max(np.abs(J_matrix[:, j]))
    if col_max > 0:
        J_matrix[:, j] /= col_max

fig, ax = plt.subplots(figsize=(10, 4))
im = ax.imshow(J_matrix.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(n_bands))
ax.set_xticklabels(band_labels, fontsize=10)
ax.set_yticks(range(n_params))
ax.set_yticklabels([n.replace("sfh_tsnorm_", "").replace("met_", "").replace("dust_", "d_")
                     for n in param_names], fontsize=8)
ax.set_xlabel("SDSS Band")
ax.set_ylabel("Parameter")
plt.colorbar(im, ax=ax, label="Normalized sensitivity", shrink=0.8)
ax.set_title("Jacobian: ∂photometry / ∂parameter (signed, normalized)")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig07_jacobian_heatmap.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Degeneracies
#
# The Jacobian reveals:
# - **Age–dust**: dust reddens; so does old age. Similar column patterns.
# - **Z–age**: metal-rich mimics old. D4000 sensitive to both.
# - **SFR–dust**: high SFR + high dust ≈ low SFR + low dust in broadband.
#
# These curved degeneracies are why geoVI's nonlinear coordinate transform
# matters — it straightens the "banana" so inference explores efficiently.

# %% [markdown]
# ## Summary
#
# You now understand every component of the differentiable pipeline:
# SSP → CSP → dust → redshift → filter → photometry. The Jacobian is
# computed for free and reveals the information content of your data.
#
# Next: **tutorials/05** shows how to check your model before fitting
# (prior predictive checks).
