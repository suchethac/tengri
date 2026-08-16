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
# # Spectroscopic Tools: Calibration, Resolution, and Line Spread Functions
#
# tengri provides differentiable (JAX) functions for instrument-level effects
# in pixel-level spectral fitting: flux calibration errors, finite resolution,
# velocity broadening, and emission-line blending. All demos use synthetic
# spectra -- **no external data files required**.

# %%
import warnings
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tengri.observation.calibration import (
    apply_calibration,
    calibration_polynomial,
    chebyshev_basis,
)
from tengri.observation.spectrum import (
    SSP_LIBRARY_RESOLUTIONS,
    apply_lsf,
    blend_emission_lines,
    nirspec_g140m_resolution,
    nirspec_prism_resolution,
    velocity_broaden,
)

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

# %% [markdown]
# ## 1. Chebyshev Calibration Polynomial
#
# A low-order Chebyshev polynomial absorbs wavelength-dependent calibration
# errors (slit losses, telluric residuals) multiplicatively:
# $C(\lambda) = 1 + \sum_{n=1}^{N} a_n T_n(x)$, where $x \in [-1,1]$.
# Coefficients use a Gaussian(0, 0.1) prior toward flat calibration.

# %%
wave = jnp.linspace(3500.0, 10000.0, 500)
basis = chebyshev_basis(wave, order=4, wave_min=3500.0, wave_max=10000.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))
for n in range(5):
    axes[0].plot(wave, basis[n], label=f"$T_{n}$")
axes[0].set(xlabel="Wavelength (A)", ylabel="$T_n(x)$", title="Chebyshev Basis")
axes[0].legend(ncol=5, fontsize=8)
axes[0].axhline(0, color="k", lw=0.5, ls="--")

for label, c in [
    ("flat", jnp.zeros(0)),
    ("$a_1{=}0.05$", jnp.array([0.05])),
    ("$a_1{=}0.05, a_2{=}{-}0.03$", jnp.array([0.05, -0.03])),
    ("3 coeffs", jnp.array([0.08, -0.04, 0.02])),
]:
    y = jnp.ones_like(wave) if c.size == 0 else calibration_polynomial(wave, c, 3500.0, 10000.0)
    axes[1].plot(wave, y, label=label)
axes[1].set(xlabel="Wavelength (A)", ylabel="$C(\\lambda)$", title="Calibration Polynomials")
axes[1].axhline(1, color="k", lw=0.5, ls="--")
axes[1].legend(fontsize=7)
fig.tight_layout()
plt.show()

# %%
# Before/after on a mock blackbody
h, c_cgs, k_B = 6.626e-27, 3e10, 1.381e-16
wc = wave * 1e-8
bb = 2 * h * c_cgs**2 / wc**5 / (jnp.exp(h * c_cgs / (wc * k_B * 6000.0)) - 1)
bb = bb / bb.max()
calibrated = apply_calibration(bb, wave, jnp.array([0.06, -0.04, 0.02]), 3500.0, 10000.0)

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(wave, bb, label="Physical")
ax.plot(wave, calibrated, ls="--", label="Calibrated")
ax.set(xlabel="Wavelength (A)", ylabel="Normalized flux")
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Instrument Resolution Profiles
#
# - **NIRSpec PRISM**: wavelength-dependent, $R \approx 30$--$330$ (0.6--5.3 um)
# - **NIRSpec G140M**: constant $R \approx 1000$

# %%
wave_um = jnp.linspace(0.6, 5.3, 500)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(wave_um, nirspec_prism_resolution(wave_um), lw=2, label="PRISM")
ax.plot(wave_um, nirspec_g140m_resolution(wave_um), lw=2, ls="--", label="G140M")
ax.set(
    xlabel="Wavelength ($\\mu$m)", ylabel="$R$", ylim=(0, 1200), title="JWST NIRSpec Resolution"
)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Line Spread Function (LSF)
#
# Constant-$R$ convolution is a single FFT in $\ln\lambda$. The effective
# width subtracts SSP library resolution in quadrature:
# $\sigma_{\rm eff} = \sqrt{\sigma_{\rm inst}^2 - \sigma_{\rm lib}^2}$.
# For variable $R(\lambda)$, `apply_lsf` uses piecewise-constant bins.

# %%
n_pix = 1000
wl = jnp.linspace(4000.0, 7000.0, n_pix)
raw = jnp.ones(n_pix)
for lc in [4861.0, 5007.0, 6563.0]:
    raw = raw + 5.0 * jnp.exp(-0.5 * ((wl - lc) / 0.5) ** 2)
sm100 = apply_lsf(raw, wl, resolution=100.0)
sm1k = apply_lsf(raw, wl, resolution=1000.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5), sharey=True)
for ax, rng, t in zip(axes, [(4820, 5050), (6500, 6620)], ["H$\\beta$+[OIII]", "H$\\alpha$"]):
    m = (wl >= rng[0]) & (wl <= rng[1])
    ax.plot(wl[m], raw[m], label="Intrinsic", alpha=0.7)
    ax.plot(wl[m], sm1k[m], label="$R=1000$")
    ax.plot(wl[m], sm100[m], ls="--", label="$R=100$")
    ax.set(xlabel="Wavelength (A)", title=t)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Flux")
fig.suptitle("LSF Convolution", y=1.01)
fig.tight_layout()
plt.show()

# %%
print("SSP library resolutions:", SSP_LIBRARY_RESOLUTIONS)

# %% [markdown]
# ## 4. Velocity Broadening
#
# `velocity_broaden` convolves with a Gaussian in $\ln\lambda$ at a given
# stellar velocity dispersion $\sigma$ (km/s).

# %%
wv = jnp.linspace(8400.0, 8700.0, 800)
ab = jnp.ones(800)
for lc in [8498.0, 8542.0, 8662.0]:  # Ca II triplet
    ab = ab - 0.3 * jnp.exp(-0.5 * ((wv - lc) / 1.0) ** 2)
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(wv, ab, label="Intrinsic", alpha=0.6)
for sv in [50.0, 150.0, 300.0]:
    ax.plot(wv, velocity_broaden(ab, wv, sv), label=f"$\\sigma={sv:.0f}$ km/s")
ax.set(xlabel="Wavelength (A)", ylabel="Flux", title="Velocity Broadening: Ca II Triplet")
ax.legend(fontsize=9)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Emission Line Blending
#
# `blend_emission_lines` places lines as Gaussians whose width is set by $R$.
# At low resolution, nearby lines merge (e.g., [NII]+H$\alpha$).

# %%
wb = jnp.linspace(6400.0, 6700.0, 600)
lw = jnp.array([6548.0, 6563.0, 6584.0])  # [NII], Halpha, [NII]
ll = jnp.array([0.3, 1.0, 0.9])
fig, ax = plt.subplots(figsize=(8, 3.5))
for R, ls in [(5000, "-"), (1000, "--"), (100, ":")]:
    bl = blend_emission_lines(lw, ll, float(R), wb)
    ax.plot(wb, bl / bl.max(), ls=ls, label=f"$R={R}$")
for w, n in zip(lw, ["[NII]", "H$\\alpha$", "[NII]"]):
    ax.axvline(w, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax.text(w, 1.05, n, ha="center", fontsize=7, color="gray")
ax.set(
    xlabel="Wavelength (A)",
    ylabel="Normalized flux",
    title="[NII]+H$\\alpha$ Blending",
    ylim=(-0.05, 1.15),
)
ax.legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Analytic Calibration Marginalization
#
# When fitting spectra, the calibration polynomial coefficients are nuisance
# parameters that increase the dimensionality of the sampling problem. The
# Prospector approach (Johnson et al. 2021) integrates them out analytically
# at each likelihood evaluation.
#
# `marginalize_calibration()` takes a model spectrum, observed spectrum, and
# uncertainties, and returns:
# - The **marginalized log-likelihood** (calibration integrated out)
# - The **MAP calibration coefficients** $\hat{c}$ (optimal correction)
# - The **posterior uncertainties** on $\hat{c}$
#
# This is equivalent to sampling over the calibration coefficients with a
# Gaussian prior, but without the extra dimensions in MCMC/VI.

# %%
from tengri.observation.calibration import marginalize_calibration

# Create a synthetic "physical" model spectrum (smooth blackbody + emission lines)
wave_cal = jnp.linspace(4000.0, 9000.0, 800)
h_cal, c_cal, k_cal = 6.626e-27, 3e10, 1.381e-16
wc_cal = wave_cal * 1e-8
bb_cal = (
    2 * h_cal * c_cal**2 / wc_cal**5 / (jnp.exp(h_cal * c_cal / (wc_cal * k_cal * 5500.0)) - 1)
)
# Add emission lines
for lc_cal, amp_cal in [(4861.0, 0.15), (5007.0, 0.25), (6563.0, 0.35)]:
    bb_cal = bb_cal + amp_cal * bb_cal.max() * jnp.exp(-0.5 * ((wave_cal - lc_cal) / 2.0) ** 2)
model_true = bb_cal / bb_cal.max()  # normalize to ~1

# Apply a known calibration offset (the "true" instrument response)
true_coeffs = jnp.array([0.08, -0.05, 0.03])
cal_true = calibration_polynomial(wave_cal, true_coeffs, 4000.0, 9000.0)
observed_noiseless = model_true * cal_true

# Add realistic noise (S/N ~ 30)
key = jax.random.PRNGKey(42)
snr = 30.0
obs_err_cal = model_true / snr
noise = jax.random.normal(key, shape=wave_cal.shape) * obs_err_cal
observed = observed_noiseless + noise

# %%
# Run marginalization
log_like_marg, c_hat, c_hat_err = marginalize_calibration(
    model_true,
    observed,
    obs_err_cal,
    wave_cal,
    n_poly=3,
    prior_sigma=1.0,
)

# Compare to uncalibrated log-likelihood (C = 1, no correction)
chi2_uncal = jnp.sum(((observed - model_true) / obs_err_cal) ** 2)
n_pix = wave_cal.shape[0]
log_like_uncal = (
    -0.5 * n_pix * jnp.log(2.0 * jnp.pi) - jnp.sum(jnp.log(obs_err_cal)) - 0.5 * chi2_uncal
)

print("Analytic calibration marginalization results:")
print(f"  True coefficients:      {np.array(true_coeffs)}")
print(f"  Recovered coefficients: {np.array(c_hat)}")
print(f"  Posterior uncertainties: {np.array(c_hat_err)}")
print(f"  Log-likelihood (uncalibrated):  {float(log_like_uncal):.1f}")
print(f"  Log-likelihood (marginalized):  {float(log_like_marg):.1f}")
print(f"  Improvement:                    {float(log_like_marg - log_like_uncal):.1f}")

# %%
# --- FIGURE: Calibration marginalization demonstration ---
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Top-left: Observed vs model (before and after calibration recovery)
ax = axes[0, 0]
cal_recovered = calibration_polynomial(wave_cal, c_hat, 4000.0, 9000.0)
model_corrected = model_true * cal_recovered
ax.plot(wave_cal, observed, color="#1f77b4", lw=0.5, alpha=0.5, label="Observed")
ax.plot(wave_cal, model_true, color="#2ca02c", lw=1.0, ls="--", label="Physical model")
ax.plot(wave_cal, model_corrected, color="#d62728", lw=1.0, label="Calibrated model")
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel("Normalized flux")
ax.set_title("Spectrum: Before and After Calibration")
ax.legend(fontsize=7, frameon=False)

# Top-right: Calibration polynomial (true vs recovered)
ax = axes[0, 1]
ax.plot(wave_cal, cal_true, color="#2ca02c", lw=2.0, ls="--", label="True $C(\\lambda)$")
ax.plot(wave_cal, cal_recovered, color="#d62728", lw=2.0, label="Recovered $C(\\lambda)$")
ax.axhline(1.0, color="grey", lw=0.5, ls=":")
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel("$C(\\lambda)$")
ax.set_title("Calibration Polynomial Recovery")
ax.legend(fontsize=8, frameon=False)

# Bottom-left: Residuals before and after
ax = axes[1, 0]
resid_before = (observed - model_true) / obs_err_cal
resid_after = (observed - model_corrected) / obs_err_cal
ax.scatter(wave_cal, resid_before, s=1, alpha=0.3, color="#1f77b4", label="Uncalibrated")
ax.scatter(wave_cal, resid_after, s=1, alpha=0.3, color="#d62728", label="After marginalization")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.axhline(2, color="grey", lw=0.3, ls=":")
ax.axhline(-2, color="grey", lw=0.3, ls=":")
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel(r"Residual [$\sigma$]")
ax.set_title("Normalized Residuals")
ax.legend(fontsize=7, frameon=False, markerscale=5)

# Bottom-right: Coefficient recovery bar chart
ax = axes[1, 1]
x_pos = np.arange(len(true_coeffs))
bar_width = 0.35
ax.bar(
    x_pos - bar_width / 2,
    np.array(true_coeffs),
    bar_width,
    color="#2ca02c",
    alpha=0.7,
    label="True",
)
ax.bar(
    x_pos + bar_width / 2,
    np.array(c_hat),
    bar_width,
    color="#d62728",
    alpha=0.7,
    label="Recovered",
    yerr=np.array(c_hat_err),
    capsize=4,
)
ax.set_xticks(x_pos)
ax.set_xticklabels(["$a_1$", "$a_2$", "$a_3$"])
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_ylabel("Coefficient value")
ax.set_title("Chebyshev Coefficient Recovery")
ax.legend(fontsize=8, frameon=False)

fig.suptitle("Analytic Calibration Marginalization (Johnson+2021)", y=1.01)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### Calibration marginalization in practice
#
# In a full tengri fit, calibration coefficients can be either:
#
# 1. **Sampled** as free parameters with Gaussian priors (adds $N_{\rm poly}$
#    dimensions to the posterior):
#    ```python
#    spec = Parameters(cal_c1=Gaussian(0, 0.1), cal_c2=Gaussian(0, 0.1), ...)
#    ```
#
# 2. **Marginalized analytically** via `marginalize_calibration()` in a
#    custom log-likelihood, reducing dimensionality at zero sampling cost.
#
# The marginalized approach is preferred for spectroscopic fitting because
# it avoids the extra MCMC dimensions while correctly accounting for
# calibration uncertainty in the marginalized log-likelihood.

# %% [markdown]
# ## Summary
#
# | Function | Purpose |
# |----------|---------|
# | `chebyshev_basis` | Evaluate $T_0$--$T_N$ basis |
# | `calibration_polynomial` | $C(\lambda)=1+\sum a_n T_n$ |
# | `apply_calibration` | Multiply spectrum by $C(\lambda)$ |
# | `marginalize_calibration` | Analytic marginalization over calibration coefficients |
# | `nirspec_prism_resolution` | Variable $R(\lambda)$ for PRISM |
# | `nirspec_g140m_resolution` | Constant $R\approx1000$ |
# | `apply_lsf` | Instrument LSF convolution |
# | `velocity_broaden` | Stellar velocity dispersion |
# | `blend_emission_lines` | Resolution-broadened line placement |
#
# All functions are pure JAX, JIT-compatible, and differentiable -- suitable
# for gradient-based inference with tengri's `SEDModel` and `Fitter`.
