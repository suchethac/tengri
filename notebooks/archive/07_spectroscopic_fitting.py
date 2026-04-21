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
# # Fitting Galaxy Spectra
#
# Photometry measures the *total light* in a few broad bands.
# Spectroscopy resolves *individual features* — absorption lines,
# emission lines, and continuum breaks — that carry far more
# diagnostic power.
#
# **Why spectroscopy matters:**
#
# | Feature | Wavelength | What it constrains |
# |---------|------------|-------------------|
# | 4000 \AA\ break (D4000) | ~4000 \AA | Age + metallicity |
# | Balmer series (H$\beta$, H$\gamma$, H$\delta$) | 4100--4860 \AA | Recent SFH (last ~1 Gyr) |
# | H$\alpha$ | 6563 \AA | Current SFR, dust |
# | [O III] $\lambda$5007 | 5007 \AA | Ionization state, AGN vs SF |
# | Mg b, Fe lines | 5170--5270 \AA | Metallicity, [$\alpha$/Fe] |
#
# Spectroscopy resolves the **age--dust--metallicity degeneracy** that
# plagues broadband photometry.  This notebook shows how to generate
# mock spectra, fit them with `tengri`, diagnose the fit quality, and
# compare the information content of photometry vs spectroscopy.
#
# **By the end you will understand:**
# 1. How to generate mock galaxy spectra and fit them with tengri
# 2. Which spectral features constrain which physical properties
# 3. How to diagnose fit quality through residual analysis
# 4. The quantitative constraint improvement from photometry to spectroscopy
# 5. How spectral coverage shifts with redshift
#
# > **Prerequisites:** NB02 (forward model), NB03 (inference methods).

# %%
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt

import sys; sys.path.insert(0, ".")
from _plot_style import setup_style, COLORS, SDSS_WAVE_EFF, safe_corner
setup_style()
import os; os.makedirs("notebook_figures", exist_ok=True)

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)

# Reproducibility
key = jax.random.PRNGKey(707)

# Load SSP data and SDSS filters (for later photometric comparison)
ssp_data = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# %% [markdown]
# ## Generating Mock Spectra
#
# A spectroscopic observation is a flux density measured on a
# **wavelength grid** $\lambda_1, \dots, \lambda_{N_{\rm pix}}$ in the
# *observed frame*.  The forward model predicts a rest-frame spectrum,
# redshifts it, and resamples onto the observed grid.
#
# Key choices:
#
# - **Wavelength range:** determined by the spectrograph (e.g.\ SDSS
#   covers 3800--9200 \AA).
# - **Pixel count / resolution:** more pixels $\to$ more information,
#   but also more noise per pixel.
# - **Noise model:** typically $\sigma(\lambda) \propto f(\lambda) /
#   \mathrm{SNR}$ (signal-dependent), so faint continuum regions are
#   noisier in absolute terms.
#
# We use `data_type="spectroscopy"` to tell the `Fitter` that the data
# vector is a spectrum (not photometric fluxes).

# %%
# Define a stochastic model for spectroscopic fitting
spec = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
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

# Ground truth — use spec.sample() to include psd_xi for stochastic model
key, subkey = jax.random.split(key)
true_params = spec.sample(subkey)

# Observed-frame wavelength grid (SDSS-like coverage)
wave_obs = jnp.linspace(3800.0, 9200.0, 500)  # 500 pixels, ~10.8 A/pix

# Generate mock spectrum at SNR = 30
key, subkey = jax.random.split(key)
spec_true = model.predict_spectrum(true_params, wave_obs)
snr = 30.0
noise = spec_true / snr
spec_obs = spec_true + noise * jax.random.normal(subkey, spec_true.shape)

print(f"Wavelength range: {float(wave_obs[0]):.0f} -- {float(wave_obs[-1]):.0f} A")
print(f"N_pix = {len(wave_obs)}, SNR = {snr}")
print(f"D = {spec.n_free} physical params + 128 GP latents")

# Plot the mock spectrum
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wave_obs, spec_obs, "0.6", lw=0.5, label="Observed (noisy)")
ax.plot(wave_obs, spec_true, "C0-", lw=1.2, label="Truth")
ax.fill_between(np.array(wave_obs),
                np.array(spec_true - noise),
                np.array(spec_true + noise),
                color="C0", alpha=0.15, label=r"$\pm 1\sigma$")
ax.set_xlabel(r"Observed wavelength [\AA]")
ax.set_ylabel(r"Flux density [erg/s/cm$^2$/\AA]")
ax.set_title(f"Mock galaxy spectrum (z = 0.1, SNR = {snr:.0f})")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("notebook_figures/07_spectroscopic_fitting_fig01.png", dpi=72, bbox_inches="tight")
plt.show()

# %%
# Annotate key spectral features in the observed frame
z_true = 0.1  # redshift

# Rest-frame feature wavelengths (Angstrom)
features = {
    r"D4000": 4000.0,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4340.0,
    r"H$\beta$": 4861.0,
    r"[O III]": 5007.0,
    r"Mg b": 5175.0,
    r"Na D": 5893.0,
    r"H$\alpha$": 6563.0,
}

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wave_obs, spec_obs, "0.6", lw=0.5)
ax.plot(wave_obs, spec_true, "C0-", lw=1.0)

colors_feat = plt.cm.Set2(np.linspace(0, 1, len(features)))
for (name, lam_rest), color in zip(features.items(), colors_feat):
    lam_obs = lam_rest * (1.0 + z_true)
    if float(wave_obs[0]) <= lam_obs <= float(wave_obs[-1]):
        ax.axvline(lam_obs, color=color, ls="--", lw=0.8, alpha=0.8)
        ax.text(lam_obs + 15, ax.get_ylim()[1] * 0.92, name,
                fontsize=8, color=color, rotation=90, va="top")

ax.set_xlabel(r"Observed wavelength [\AA]")
ax.set_ylabel(r"Flux density [erg/s/cm$^2$/\AA]")
ax.set_title("Key spectral features at z = 0.1")
plt.tight_layout()
plt.savefig("notebook_figures/07_spectroscopic_fitting_fig02.png", dpi=72, bbox_inches="tight")
plt.show()

print("What each feature constrains:")
print("  D4000 break  -> stellar age + metallicity")
print("  Balmer series -> recent SFH (last ~1 Gyr)")
print("  H-alpha      -> current SFR + dust attenuation")
print("  [O III]      -> ionization state (AGN vs star-forming)")
print("  Mg b, Na D   -> metallicity, alpha-element abundance")

# %% [markdown]
# > **SED-fitting wisdom:** Each spectral feature acts as a temporal filter
# > on the SFH. H$\alpha$ responds to the last ~5 Myr (ionizing photons from
# > O stars). Balmer absorption (H$\delta$, H$\gamma$) probes ~100 Myr–1 Gyr
# > (A-star dominated populations). The 4000 Å break responds to ~1 Gyr+
# > (metal-line blanketing in evolved stars). In PSD language, different
# > features probe different frequency ranges of the power spectrum.

# %% [markdown]
# ## Fitting a Spectrum
#
# The workflow is the same as photometric fitting, but we pass
# `data_type="spectroscopy"` and provide the observed wavelength grid:
#
# ```python
# model._wave_obs = wave_obs
# fitter = Fitter(model, spec_obs, noise,
#                 data_type="spectroscopy")
# ```
#
# The likelihood becomes $\chi^2 = \sum_{i=1}^{N_{\rm pix}}
# (d_i - f_i)^2 / \sigma_i^2$ summed over all spectral pixels
# instead of photometric bands.  More data points $\to$ tighter
# constraints $\to$ better-determined SFH.
#
# We use **MAP initialization** followed by **Ray Tracing** for the
# full posterior.

# %%
# Spectroscopic fit: MAP -> Ray Tracing
model._wave_obs = wave_obs
fitter = Fitter(model, spec_obs, noise,
                data_type="spectroscopy")

# MAP initialization
key, subkey = jax.random.split(key)
result_map = fitter.run("map", n_steps=1500, key=subkey)
print(f"MAP: {result_map.wall_time_s:.1f}s, "
      f"final loss = {float(result_map.loss_history[-1]):.2f}")

# Ray Tracing posterior (step_size=0.01 for high-D stochastic model)
key, subkey = jax.random.split(key)
result_rt = fitter.run(
    "raytrace",
    init_from=result_map,
    n_burnin=100,
    n_steps=300,
    step_size=0.05, n_leapfrog_steps=50,
    key=subkey,
)
accept = result_rt.diagnostics.get("accept_rate_post_burnin", 0)
print(f"Ray Tracing: {result_rt.wall_time_s:.1f}s, "
      f"acceptance = {accept:.2%}")

# Overlay best-fit spectrum on data
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(wave_obs, spec_obs, "0.6", lw=0.5, label="Data")

# Posterior median spectrum (draw a few samples)
spec_draws = []
n_draws = min(50, len(next(iter(result_rt.samples.values()))))
for k_idx in range(n_draws):
    draw = {name: (arr[k_idx] if name == 'sfh_field_xi' else float(arr[k_idx])) for name, arr in result_rt.samples.items()}
    spec_draws.append(np.array(model.predict_spectrum(draw, wave_obs)))
spec_draws = np.array(spec_draws)
spec_median = np.median(spec_draws, axis=0)
spec_lo, spec_hi = np.percentile(spec_draws, [16, 84], axis=0)

ax.plot(wave_obs, spec_median, "C3-", lw=1.2, label="RT median")
ax.fill_between(np.array(wave_obs), spec_lo, spec_hi,
                color="C3", alpha=0.2, label="68% CI")
ax.plot(wave_obs, spec_true, "C0--", lw=0.8, alpha=0.6, label="Truth")
ax.set_xlabel(r"Observed wavelength [\AA]")
ax.set_ylabel(r"Flux density [erg/s/cm$^2$/\AA]")
ax.set_title("Spectroscopic fit: data vs model")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig("notebook_figures/07_spectroscopic_fitting_fig03.png", dpi=72, bbox_inches="tight")
plt.show()

# %%
# Corner plot + SFH recovery
phys_params = ["sfh_dpl_alpha", "sfh_dpl_beta", "sfh_dpl_tau_gyr", "sfh_dpl_log_peak_sfr",
               "sfh_field_psd_sigma", "sfh_field_psd_tau_myr", "met_logzsol",
               "dust_tau_bc", "dust_tau_diff"]

fig = safe_corner(result_rt,
    params=phys_params,
    truths=true_params,
    color="C3",
    label="Spectroscopy (RT)",
)
if fig is not None:
    fig.suptitle("Spectroscopic posterior — physical parameters", fontsize=14, y=1.02)
plt.savefig("notebook_figures/07_spectroscopic_fitting_fig04.png", dpi=72, bbox_inches="tight")
plt.show()

# SFH recovery
model.plot_sfh_posterior(result_rt, true_params=true_params)
plt.savefig("notebook_figures/07_spectroscopic_fitting_fig05.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Residual Analysis
#
# A good fit is not just about recovering the right parameters —
# you must verify that the **residuals** $(d_i - f_i) / \sigma_i$
# are consistent with noise.
#
# **What to look for:**
#
# - **$\chi^2 / N_{\rm dof}$** should be close to 1.
# - **Flat residuals:** no systematic trends with wavelength.
# - **Feature-by-feature:** check that individual absorption/emission
#   lines are well-fit (residuals should not spike at known features).
# - **Correlated residuals:** runs of positive or negative residuals
#   indicate model mismatch (e.g.\ wrong continuum shape, missing
#   emission lines).

# %%
# Residual analysis
spec_bestfit = np.median(spec_draws, axis=0)
residuals = (np.array(spec_obs) - spec_bestfit) / np.array(noise)
chi2 = np.sum(residuals**2)
n_dof = len(wave_obs) - spec.n_free
chi2_per_dof = chi2 / n_dof

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                         gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

# Top: spectrum + fit
axes[0].plot(wave_obs, spec_obs, "0.6", lw=0.5, label="Data")
axes[0].plot(wave_obs, spec_bestfit, "C3-", lw=1.0, label="Best fit")
axes[0].set_ylabel(r"Flux density [erg/s/cm$^2$/\AA]")
axes[0].legend(fontsize=9)
axes[0].set_title(f"Spectroscopic fit — $\\chi^2/\\nu$ = {chi2_per_dof:.2f}")

# Bottom: normalized residuals
axes[1].axhline(0, color="k", lw=0.5)
axes[1].axhline(2, color="k", lw=0.3, ls=":")
axes[1].axhline(-2, color="k", lw=0.3, ls=":")
axes[1].plot(wave_obs, residuals, "C0-", lw=0.5)
axes[1].set_ylabel(r"$(d - f) / \sigma$")
axes[1].set_xlabel(r"Observed wavelength [\AA]")
axes[1].set_ylim(-4, 4)

# Highlight spectral features in residuals
z_true = 0.1
for _name, lam_rest in [("D4000", 4000.0), (r"H$\beta$", 4861.0),
                         (r"H$\alpha$", 6563.0)]:
    lam_obs_feat = lam_rest * (1 + z_true)
    for ax in axes:
        ax.axvline(lam_obs_feat, color="0.7", ls=":", lw=0.6)

plt.tight_layout()
plt.savefig("notebook_figures/07_spectroscopic_fitting_fig06.png", dpi=72, bbox_inches="tight")
plt.show()

print(f"chi2 = {chi2:.1f}")
print(f"N_dof = {n_dof}")
print(f"chi2 / N_dof = {chi2_per_dof:.3f}")
print(f"Mean |residual| = {np.mean(np.abs(residuals)):.3f} (expect ~0.80)")
print(f"Fraction |residual| > 2: {np.mean(np.abs(residuals) > 2):.1%} (expect ~5%)")

# %% [markdown]
# ## Signal-to-Noise Ratio
#
# How does spectral SNR affect parameter recovery?  Lower SNR means
# wider posteriors and more degenerate solutions.  Higher SNR drives
# tighter constraints, but with diminishing returns once systematics
# dominate.
#
# We fit the **same galaxy** at SNR = 10, 30, and 100, using MAP + RT
# for each, and compare the posterior widths.

# %%
# Fit at three SNR values
snr_values = [10, 30, 100]
results_snr = {}

for snr_val in snr_values:
    noise_snr = spec_true / snr_val
    key, subkey = jax.random.split(key)
    spec_obs_snr = spec_true + noise_snr * jax.random.normal(subkey, spec_true.shape)

    model._wave_obs = wave_obs
    fitter_snr = Fitter(model, spec_obs_snr, noise_snr,
                        data_type="spectroscopy")

    key, subkey = jax.random.split(key)
    map_snr = fitter_snr.run("map", n_steps=1500, key=subkey)

    key, subkey = jax.random.split(key)
    rt_snr = fitter_snr.run(
        "raytrace",
        init_from=map_snr,
        n_burnin=100,
        n_steps=300,
        step_size=0.05, n_leapfrog_steps=50,
        key=subkey,
    )
    results_snr[snr_val] = rt_snr
    accept = rt_snr.diagnostics.get("accept_rate_post_burnin", 0)
    print(f"SNR={snr_val}: RT {rt_snr.wall_time_s:.1f}s, accept={accept:.2%}")

# 1x3 SFH recovery panels
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
colors_snr = ["C4", "C3", "C2"]
sfh_truth = model.predict_sfh(true_params)

for ax, snr_val, col in zip(axes, snr_values, colors_snr):
    res = results_snr[snr_val]
    ax.plot(sfh_truth["t_gyr"], sfh_truth["sfr_mean"], "k-", lw=2, label="Truth")

    sfr_draws = []
    n_d = min(100, len(next(iter(res.samples.values()))))
    for k_idx in range(n_d):
        draw = {name: (arr[k_idx] if name == 'sfh_field_xi' else float(arr[k_idx])) for name, arr in res.samples.items()}
        sfh_draw = model.predict_sfh(draw)
        sfr_draws.append(sfh_draw["sfr_mean"])
    sfr_draws = np.array(sfr_draws)
    lo, hi = np.percentile(sfr_draws, [16, 84], axis=0)
    ax.fill_between(sfh_truth["t_gyr"], lo, hi, color=col, alpha=0.3, label="68% CI")
    ax.plot(sfh_truth["t_gyr"], np.median(sfr_draws, axis=0),
            color=col, ls="--", lw=1.5, label="Median")

    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_title(f"SNR = {snr_val}", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")

axes[0].set_ylabel(r"SFR [$M_\odot$/yr]")
fig.suptitle("SFH Recovery vs Spectral SNR", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("notebook_figures/07_spectroscopic_fitting_fig07.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Photometry vs Spectroscopy: Head-to-Head
#
# Same galaxy, same ground truth, same inference method (MAP + RT).
# How much tighter are spectroscopic constraints compared to 5-band
# SDSS photometry?
#
# The comparison is not entirely fair — a spectrum with 500 pixels at
# SNR = 30 carries $\sim$500 independent data points, while 5-band
# photometry gives only 5.  But this *is* the real-world tradeoff:
# spectra are expensive but informative.

# %%
# Photometric fit for comparison
key, subkey = jax.random.split(key)
mock_phot = model.mock(true_params, snr=20.0, key=subkey)

fitter_phot = Fitter(model, mock_phot.flux_obs, mock_phot.noise,
                     data_type="photometry")

key, subkey = jax.random.split(key)
map_phot = fitter_phot.run("map", n_steps=1500, key=subkey)

key, subkey = jax.random.split(key)
rt_phot = fitter_phot.run(
    "raytrace",
    init_from=map_phot,
    n_burnin=100,
    n_steps=300,
    step_size=0.05, n_leapfrog_steps=50,
    key=subkey,
)
print(f"Photometry RT: {rt_phot.wall_time_s:.1f}s")

# Overlay corner plots: photometry (blue) vs spectroscopy (red)
# Use the SNR=30 spectroscopic result from the main fit
fig = safe_corner(rt_phot,
    params=phys_params,
    truths=true_params,
    color="C0",
    label="Photometry (5-band)",
)
if fig is not None:
    safe_corner(result_rt,
        params=phys_params,
        truths=true_params,
        color="C3",
        label="Spectroscopy (500 pix)",
        fig=fig,
    )
    fig.suptitle("Photometry vs Spectroscopy — same galaxy, same method",
                 fontsize=14, y=1.02)
plt.savefig("notebook_figures/07_spectroscopic_fitting_fig08.png", dpi=72, bbox_inches="tight")
plt.show()

# Quantify constraint improvement
print("\nPosterior width comparison (68% CI):")
print(f"{'Parameter':<25} {'Phot width':>12} {'Spec width':>12} {'Improvement':>12}")
print("-" * 65)
for p in phys_params:
    phot_chain = np.array(rt_phot.samples[p])
    spec_chain = np.array(result_rt.samples[p])
    w_phot = np.percentile(phot_chain, 84) - np.percentile(phot_chain, 16)
    w_spec = np.percentile(spec_chain, 84) - np.percentile(spec_chain, 16)
    ratio = w_phot / w_spec if w_spec > 0 else np.inf
    print(f"{p:<25} {w_phot:>12.3f} {w_spec:>12.3f} {ratio:>11.1f}x")

# %% [markdown]
# ## Spectroscopy at Different Redshifts
#
# The observable wavelength range is fixed by the instrument, but the
# **rest-frame** coverage shifts with redshift:
# $\lambda_{\rm rest} = \lambda_{\rm obs} / (1 + z)$.
#
# At higher redshift, rest-frame UV features enter the optical window
# while rest-frame optical features shift to the NIR.  This determines
# which physical properties can be constrained:
#
# | Redshift | Survey | $\lambda_{\rm obs}$ range | Rest-frame coverage | Key features |
# |----------|--------|--------------------------|--------------------|----|
# | $z = 0.1$ | SDSS | 3800--9200 \AA | 3450--8360 \AA | D4000, Balmer, H$\alpha$ |
# | $z = 1.0$ | DESI | 3600--9800 \AA | 1800--4900 \AA | UV slope, D4000, Balmer |
# | $z = 6.0$ | JWST/NIRSpec | 6000--53000 \AA | 860--7570 \AA | Ly$\alpha$, UV, D4000 |

# %%
# Visualize spectral feature accessibility at different redshifts
redshifts = [0.1, 1.0, 6.0]
survey_names = ["SDSS", "DESI", "JWST/NIRSpec"]
obs_ranges = [(3800, 9200), (3600, 9800), (6000, 53000)]  # Angstrom

# Rest-frame features
rest_features = {
    r"Ly$\alpha$": 1216.0,
    "UV slope": 1500.0,
    "D4000": 4000.0,
    r"H$\beta$": 4861.0,
    r"[O III]": 5007.0,
    r"H$\alpha$": 6563.0,
}

fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=False)

for ax, z, survey, (lam_lo, lam_hi) in zip(axes, redshifts, survey_names, obs_ranges):
    # Rest-frame range accessible
    rest_lo = lam_lo / (1 + z)
    rest_hi = lam_hi / (1 + z)

    # Draw accessible range
    ax.axvspan(rest_lo, rest_hi, color="C0", alpha=0.15)
    ax.axvline(rest_lo, color="C0", lw=1)
    ax.axvline(rest_hi, color="C0", lw=1)

    # Mark features
    for name, lam_rest in rest_features.items():
        in_range = rest_lo <= lam_rest <= rest_hi
        color = "C2" if in_range else "C3"
        marker = "o" if in_range else "x"
        ax.plot(lam_rest, 0.5, marker, color=color, ms=8, mew=2)
        ax.text(lam_rest, 0.75, name, fontsize=8, ha="center",
                color=color, fontweight="bold" if in_range else "normal")

    ax.set_xlim(500, 10000)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"{survey} at z = {z} "
                 f"(rest: {rest_lo:.0f}--{rest_hi:.0f} " + "\\AA)",
                 fontsize=11)

axes[-1].set_xlabel(r"Rest-frame wavelength [\AA]")
fig.suptitle("Which features are accessible at each redshift?",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("notebook_figures/07_spectroscopic_fitting_fig09.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Practical Guidance
#
# ### Survey Quick-Reference
#
# | Survey | $\lambda_{\rm obs}$ [\AA] | $R$ | Key features at target $z$ | Best-constrained properties |
# |--------|--------------------------|-----|---------------------------|---------------------------|
# | SDSS | 3800--9200 | ~2000 | D4000, Balmer, H$\alpha$ ($z < 0.4$) | Age, metallicity, SFR, dust |
# | DESI | 3600--9800 | 2000--5000 | D4000 ($z < 1.5$), UV slope ($z > 0.8$) | SFH shape, metallicity |
# | JWST/NIRSpec | 6000--53000 | 100--2700 | Ly$\alpha$ to D4000 ($z \sim 4$--$8$) | UV slope, early SFH |
# | VLT/MUSE | 4650--9300 | 1770--3590 | D4000, [O III] ($z < 0.9$) | Spatially resolved SFH |
#
# ### Fitting Checklist
#
# 1. **Always check residuals.** A good $\chi^2/\nu$ does not guarantee
#    a correct model — look for systematic patterns at known features.
# 2. **Wavelength-dependent noise.** Real spectra have noise that varies
#    with wavelength (sky lines, detector edges, atmospheric absorption).
#    Always use a realistic noise vector.
# 3. **Calibration systematics.** Flux calibration errors introduce
#    correlated residuals.  If you see broad wiggles in the residuals,
#    consider a polynomial calibration correction.
# 4. **Emission lines.** The current `tengri` forward model does not
#    include nebular emission.  Mask strong emission lines ([O III],
#    H$\alpha$, [N II]) or fit them separately.
# 5. **Resolution matching.** Convolve the model to the spectrograph's
#    line-spread function before comparing.  Unresolved features lead
#    to biased absorption-line measurements.
# 6. **MAP first, sample second.** Always initialize samplers (RT, geoVI)
#    from a MAP solution.  Cold starts waste burn-in.

# %% [markdown]
# ## What You've Learned
#
# 1. Spectroscopy resolves the age-dust-metallicity degeneracy that plagues photometry
# 2. Individual absorption features (D4000, Balmer) constrain SFH at different timescales
# 3. Residual analysis ($\chi^2/\nu$, feature-by-feature) is essential for validation
# 4. Spectroscopy tightens posteriors by 2–10x over photometry, especially for metallicity
# 5. Rest-frame spectral coverage shifts with redshift — choose your survey accordingly
#
# **Next:** [Tutorial 08 — PSD Physics](08_psd_physics.ipynb) connects PSD
# parameters to astrophysical mechanisms and observable diagnostics.
#
# ## Further Reading
#
# - **NB06: Data Information** — progressive data reveal from 1 band to full spectrum
# - **NB09: Custom Models** — extending tengri with new forward model components
# - **NB03: Inference Methods** — deep dive into all five samplers
