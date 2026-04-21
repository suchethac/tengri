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
# # Gradient Sensitivity and Survey Design (Paper §5.2)
#
# Because the forward model is differentiable, the Jacobian
# $\partial f_\nu / \partial \theta_k$ is available at every wavelength
# and for every parameter at negligible cost.  This enables a
# **multiscale gradient scalogram** — a 2D map of sensitivity across
# wavelength and spectral resolution.
#
# **Paper figures generated:**
# - **Fig 4**: End-to-end gradient sensitivity (Jacobian heatmap)
# - **Fig 9**: Multiscale gradient scalogram

# %%
import os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed,
    load_ssp_data, load_filter_set,
)
import sys; sys.path.insert(0, ".")
from _plot_style import setup_style, COLORS, SPECTRAL_FEATURES
setup_style()

FIG_DIR = "notebook_figures"
os.makedirs(FIG_DIR, exist_ok=True)
def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"A05_{name}.png"),
                bbox_inches="tight", dpi=72)

# Use C3K SSP templates for higher spectral resolution
try:
    ssp_data = load_ssp_data(
        "../data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    print("Loaded C3K SSP templates (high resolution)")
except FileNotFoundError:
    ssp_data = load_ssp_data(
        "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    print("C3K not found — using MILES (lower resolution)")

filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# %% [markdown]
# ## Stochastic SEDModel at Fiducial Parameters

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

key = jax.random.PRNGKey(42)
fiducial = spec.sample(key)
fiducial.update(sfh_alpha=1.0, sfh_beta=1.5, sfh_tau_peak_gyr=8.0,
                sfh_peak_sfr=30.0, psd_sigma=1.5, psd_tau_myr=30.0,
                met_logzsol=-0.3, dust_tau_bc=0.5, dust_tau_diff=0.3)

# %% [markdown]
# ## Paper Figure 4: End-to-End Gradient Sensitivity

# %%
# Compute spectral Jacobian
param_names = ["sfh_alpha", "sfh_beta", "sfh_tau_peak_gyr", "sfh_peak_sfr",
               "psd_sigma", "psd_tau_myr", "met_logzsol",
               "dust_tau_bc", "dust_tau_diff"]
param_labels = [r"$\alpha$", r"$\beta$", r"$\tau_{\rm pk}$", "$A$",
                r"$\sigma_{\rm PS}$", r"$\tau_{\rm PS}$",
                r"$\log Z$", r"$\hat\tau_{\rm bc}$", r"$\hat\tau_{\rm diff}$"]

def spec_fn(pvec):
    p = dict(fiducial)
    for i, k in enumerate(param_names):
        p[k] = pvec[i]
    return model.predict_spectrum(p)["flux"]

pvec = jnp.array([fiducial[k] for k in param_names])
jac = jax.jacobian(spec_fn)(pvec)  # (n_wave, n_params)
jac_np = np.array(jac)

wave = np.array(model.predict_spectrum(fiducial)["wave_obs"])

fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharex=True)

for i, (ax, pname, plabel) in enumerate(zip(axes.flat, param_names, param_labels)):
    grad_i = jac_np[:, i]
    # Signed log-scale
    sign = np.sign(grad_i)
    log_abs = np.log10(np.abs(grad_i) + 1e-30)
    ax.plot(wave, sign * log_abs, lw=0.5, color=COLORS["rt"])
    ax.set_title(plabel, fontsize=10)
    ax.set_xlim(1000, 10000)
    # Mark spectral features
    for fname, fwave in list(SPECTRAL_FEATURES.items())[:5]:
        fobs = fwave * 1.1  # z=0.1
        ax.axvline(fobs, color="0.8", ls=":", lw=0.5)

for ax in axes[-1]:
    ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\partial f_\nu / \partial \theta$ (signed log)")

fig.suptitle("Paper Figure 4: End-to-End Gradient Sensitivity ($z=0.1$)",
             fontsize=13, y=1.01)
fig.tight_layout(); savefig(fig, "paper_fig04_gradient_sensitivity"); plt.show()

# %% [markdown]
# ## Paper Figure 9: Multiscale Gradient Scalogram
#
# At each wavelength $\lambda_0$ and spectral scale $s$, the local
# gradient power is:
#
# $$C_k(\lambda_0, s) = \frac{1}{s} \int_{\lambda_0 - s/2}^{\lambda_0 + s/2} \left|\frac{\partial f_\nu}{\partial \theta_k}\right|^2 d\lambda$$

# %%
# Scalogram computation
scales_ang = np.logspace(np.log10(2), np.log10(3000), 64)
wave_centers = np.linspace(1500, 9000, 200)

# Select key parameters for scalogram
scalogram_params = [0, 4, 5, 6, 7]  # alpha, sigma_PS, tau_PS, logZ, tau_bc
scalogram_labels = [param_labels[i] for i in scalogram_params]

scalograms = {}
for pi, pidx in enumerate(scalogram_params):
    grad_sq = jac_np[:, pidx] ** 2
    C = np.zeros((len(wave_centers), len(scales_ang)))
    for si, s in enumerate(scales_ang):
        for wi, w0 in enumerate(wave_centers):
            mask = (wave >= w0 - s/2) & (wave <= w0 + s/2)
            if mask.sum() > 0:
                C[wi, si] = np.mean(grad_sq[mask])
    scalograms[param_labels[pidx]] = C

fig, axes = plt.subplots(1, len(scalogram_params), figsize=(16, 4.5))

for ax, plabel in zip(axes, scalogram_labels):
    C = scalograms[plabel]
    C_norm = C / (C.max() + 1e-30)
    im = ax.pcolormesh(wave_centers, scales_ang, C_norm.T,
                       cmap="inferno", shading="auto")
    ax.set_yscale("log")
    ax.set_title(plabel, fontsize=10)
    ax.set_xlabel(r"$\lambda_0$ [$\mathrm{\AA}$]")
    # Mark features
    for fname, fwave in SPECTRAL_FEATURES.items():
        fobs = fwave * 1.1
        if 1500 <= fobs <= 9000:
            ax.axvline(fobs, color="w", ls=":", lw=0.5, alpha=0.5)

axes[0].set_ylabel(r"Spectral scale $s$ [$\mathrm{\AA}$]")
fig.colorbar(im, ax=axes[-1], shrink=0.8, label="Normalised sensitivity")

fig.suptitle("Paper Figure 9: Multiscale Gradient Scalogram ($z=0.1$)",
             fontsize=13, y=1.03)
fig.tight_layout(); savefig(fig, "paper_fig09_scalogram"); plt.show()

# %% [markdown]
# ## Interpretation
#
# - **Dust** ($\hat\tau_{\rm bc}$, $\hat\tau_{\rm diff}$): strong
#   gradients at all scales → photometry suffices.
# - **Metallicity** ($\log Z$): concentrated near 4000 Å break and
#   metal absorption features → spectroscopy at $R \gtrsim 50$ required.
# - **SFH shape** ($\alpha$): intermediate scales (~100–500 Å).
# - **PSD parameters** ($\sigma_{\rm PS}$, $\tau_{\rm PS}$): sensitivity
#   concentrated in Hα, UV, and Balmer break regions — spectroscopy
#   strongly helps.
