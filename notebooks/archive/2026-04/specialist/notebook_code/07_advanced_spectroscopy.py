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
# # Noise Models
#
# Real astronomical data has noise properties more complex than simple
# Gaussian errors. tengri provides two noise model extensions:
#
# 1. **Calibration floor** ($f_{\rm cal}$): a fractional uncertainty added
#    in quadrature to observational errors, accounting for systematic
#    calibration uncertainties.
# 2. **Student-t likelihood**: heavy-tailed distribution that is robust
#    to outliers.
#
# Both are implemented as part of the generative model (NIFTy philosophy)
# and jointly inferred with the physical parameters.

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fitter,
    Fixed,
    SEDModel,
    Observation,
    Parameters,
    Photometry,
    Uniform,
    load_ssp_data,
)
from tengri.observation.noise import compute_effective_noise

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

from _plot_style import COLORS, setup_style

setup_style()

FIGDIR = os.path.join(_nb_dir, "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## 1. The Calibration Floor Concept
#
# Photometric calibration uncertainties are typically 1-5% of the flux.
# These systematics are not captured by Poisson/read-noise error bars.
# The effective noise becomes:
#
# $$\sigma_{\rm eff}^2 = \sigma_{\rm obs}^2 + (f_{\rm cal} \cdot m)^2$$
#
# where $m$ is the model flux and $f_{\rm cal}$ is a free parameter.

# %%
# --- FIGURE 1: Calibration floor effect ---
model_flux = jnp.array([1.0, 5.0, 10.0, 50.0, 100.0])
noise_obs = jnp.array([0.5, 0.5, 0.5, 0.5, 0.5])

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Left: effective noise vs model flux for different f_cal
ax = axes[0]
flux_grid = jnp.linspace(0.1, 100, 200)
noise_fixed = jnp.full_like(flux_grid, 1.0)
for f_cal, color, ls in [
    (0.0, COLORS["truth"], "-"),
    (0.02, COLORS["rt"], "--"),
    (0.05, COLORS["geovi"], "-."),
    (0.10, COLORS["nuts"], ":"),
    (0.15, COLORS["mgvi"], "-"),
]:
    eff = compute_effective_noise(noise_fixed, flux_grid, f_cal)
    ax.plot(
        np.array(flux_grid),
        np.array(eff),
        color=color,
        ls=ls,
        lw=1.5,
        label=f"$f_{{cal}}$ = {f_cal:.0%}" if f_cal > 0 else "No cal floor",
    )
ax.set_xlabel("SEDModel flux")
ax.set_ylabel(r"$\sigma_{\rm eff}$")
ax.set_title("Effective Noise vs SEDModel Flux")
ax.legend(fontsize=8, frameon=False)

# Right: SNR with and without cal floor
ax = axes[1]
for f_cal, color, ls in [
    (0.0, COLORS["truth"], "-"),
    (0.05, COLORS["geovi"], "--"),
    (0.10, COLORS["nuts"], ":"),
]:
    eff = compute_effective_noise(noise_fixed, flux_grid, f_cal)
    snr = flux_grid / eff
    ax.plot(
        np.array(flux_grid),
        np.array(snr),
        color=color,
        ls=ls,
        lw=1.5,
        label=f"$f_{{cal}}$ = {f_cal:.0%}" if f_cal > 0 else "No cal floor",
    )
ax.set_xlabel("SEDModel flux")
ax.set_ylabel("Effective SNR")
ax.set_title("SNR Saturation from Calibration Floor")
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "06_calibration_floor.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Student-t Likelihood
#
# The Student-t distribution has heavier tails than a Gaussian, making it
# robust to outliers. It reduces to Gaussian as $\nu \to \infty$.
#
# $$p(d | m, \sigma, \nu) = \frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\pi\nu}\sigma}
# \left(1 + \frac{(d-m)^2}{\nu\sigma^2}\right)^{-(\nu+1)/2}$$

# %%
# --- FIGURE 2: Student-t vs Gaussian comparison ---
from scipy.stats import norm, t as student_t

x = np.linspace(-6, 6, 500)
fig, ax = plt.subplots(figsize=(7, 4.5))

ax.plot(x, norm.pdf(x), "k-", lw=2, label=r"Gaussian ($\nu \to \infty$)")
for nu, color, ls in [
    (3, COLORS["rt"], "--"),
    (5, COLORS["geovi"], "-."),
    (10, COLORS["nuts"], ":"),
]:
    ax.plot(
        x, student_t.pdf(x, df=nu), color=color, ls=ls, lw=1.5, label=f"Student-t ($\\nu$ = {nu})"
    )

ax.set_xlabel("Standardized residual $(d - m)/\\sigma$")
ax.set_ylabel("Probability density")
ax.set_title("Gaussian vs Student-t Likelihood")
ax.set_yscale("log")
ax.set_ylim(1e-5, 1)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "06_student_t.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Fitting with Noise SEDModel
#
# We demonstrate fitting with the calibration floor as a free parameter.

# %%
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

# Parameters with noise model
spec_noise = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    noise_frac_cal=Uniform(0.01, 0.2),
)

# Parameters without noise model (for comparison)
spec_no_noise = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

# Generate mock with 5% systematic floor baked in
model_truth = SEDModel(spec_no_noise, ssp_data, observation=obs)
truth_params = {
    "sfh_tsnorm_log_peak_sfr": 0.8,
    "sfh_tsnorm_peak_lbt_gyr": 4.0,
    "sfh_tsnorm_width_gyr": 2.0,
    "sfh_tsnorm_skew": 0.3,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.2,
    "dust_tau_bc": 0.3,
    "dust_tau_diff": 0.5,
    "dust_slope": -0.7,
    "redshift": 0.1,
}
mock = model_truth.mock(truth_params, snr=30.0, key=jax.random.PRNGKey(42))

# Add 5% systematic scatter
key = jax.random.PRNGKey(99)
sys_scatter = 0.05 * mock.flux_true * jax.random.normal(key, shape=mock.flux_true.shape)
flux_with_sys = mock.flux_obs + sys_scatter

# Fit without noise model
model_no_noise = SEDModel(spec_no_noise, ssp_data, observation=obs)
fitter_no_noise = Fitter(model_no_noise, flux_with_sys, mock.noise)
result_no_noise = fitter_no_noise.run("map", n_steps=500, learning_rate=0.02)

# Fit with noise model
model_noise = SEDModel(spec_noise, ssp_data, observation=obs)
fitter_noise = Fitter(model_noise, flux_with_sys, mock.noise)
result_noise = fitter_noise.run("map", n_steps=500, learning_rate=0.02)

print("Without noise model - final loss:", float(result_no_noise.diagnostics.get("final_loss", 0)))
print("With noise model - final loss:", float(result_noise.diagnostics.get("final_loss", 0)))
if "noise_frac_cal" in result_noise.params:
    print(f"Inferred f_cal: {float(result_noise.params['noise_frac_cal']):.3f} (true: 0.05)")

# %%
# --- FIGURE 3: Residuals comparison ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
wave_eff = np.array([3551, 4686, 6166, 7480, 8932])

for ax, result, title in [
    (axes[0], result_no_noise, "Without noise model"),
    (axes[1], result_noise, "With noise model"),
]:
    model_for_res = model_no_noise if "noise_frac_cal" not in result.params else model_noise
    pred = model_for_res.predict_photometry(result.params)
    residuals = (np.array(flux_with_sys) - np.array(pred)) / np.array(mock.noise)
    ax.bar(np.arange(5), residuals, color=COLORS["rt"], alpha=0.7)
    ax.axhline(0, color="k", ls="-", lw=0.5)
    ax.axhline(2, color="grey", ls=":", lw=0.5)
    ax.axhline(-2, color="grey", ls=":", lw=0.5)
    ax.set_xticks(np.arange(5))
    ax.set_xticklabels(["u", "g", "r", "i", "z"])
    ax.set_ylabel(r"Residual [$\sigma$]")
    ax.set_title(title)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "06_noise_residuals.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Noise SEDModel Comparison Table
#
# | Feature | Gaussian | + Cal floor | + Student-t |
# |---------|----------|-------------|-------------|
# | Outlier robustness | No | No | Yes |
# | Systematic absorption | No | Yes | Partial |
# | Extra parameters | 0 | 1 ($f_{\rm cal}$) | 1 ($\nu$) or 2 |
# | When to use | Clean data | Most real data | Known outliers |
# | NIFTy likelihood | `Gaussian` | `VariableCovGaussian` | Custom |
#
# **Recommendation**: Enable the calibration floor ($f_{\rm cal}$) for all
# real data fitting. Use Student-t only when you know there are flux
# outliers (e.g., contaminated bands, emission line boosting).

# %% [markdown]
# ## Summary
#
# Noise modelling is a first-class citizen in tengri following the NIFTy
# Information Field Theory philosophy. The calibration floor prevents
# over-fitting to systematic residuals and the Student-t likelihood provides
# outlier robustness. Both are jointly inferred with physical parameters.

# %% [markdown]
# ---
# # Part 2 — Spectroscopic Tools
#
# Calibration polynomial marginalization, NIRSpec resolution profiles,
# LSF convolution, velocity broadening, and emission line blending.

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

# %% [markdown]
# ---
# # Part 3 — Alpha Enhancement
#
# 4D SSP grids with [α/Fe] as a fourth axis; spectral effects of
# alpha-enhancement; time-evolving [α/Fe] parameterization; Salaris relation.

# %% [markdown]
# ## 1. Build a Synthetic 4D Alpha-Enhanced SSP Grid
#
# In practice you'd load real templates (sMILES, BPASS v2.3, α-MC).
# Here we build a synthetic grid that captures the key physics:
# - Higher [α/Fe] → stronger Mg features (~5170 Å), weaker Fe lines (~5270 Å)
# - Higher [α/Fe] at fixed [Fe/H] → slightly redder continuum (more total Z)
# - Effect is strongest for old populations, weak for young hot stars

# %%
from tengri.sps.dsps_wrapper import SSPData

# Grid dimensions
n_met, n_alpha, n_age, n_wave = 5, 5, 40, 500
feh_grid = jnp.array([-2.0, -1.5, -1.0, -0.5, 0.0])
alpha_grid = jnp.array([-0.2, 0.0, 0.2, 0.4, 0.6])
lg_age_gyr = jnp.linspace(-1.5, 1.14, n_age)  # 0.03 to 13.8 Gyr
wave = jnp.linspace(3500.0, 9500.0, n_wave)

# Build synthetic SSP flux
# Start from a simple power-law continuum that varies with Z and age
key = jax.random.PRNGKey(42)
base_flux = jnp.abs(jax.random.normal(key, (n_met, n_alpha, n_age, n_wave))) * 1e-6 + 1e-7

# Add physical trends
# Metallicity: redder at higher Z
z_reddening = jnp.exp(-0.0003 * (wave - 5500.0))  # redder = more flux at long λ
met_scale = 10.0 ** (0.3 * feh_grid)  # brighter at higher Z
base_flux = (
    base_flux * met_scale[:, None, None, None] * (1.0 + 0.1 * z_reddening[None, None, None, :])
)

# Age: bluer (more UV) when young
age_gyr = 10.0**lg_age_gyr
uv_boost = jnp.exp(-0.001 * (wave - 3500.0))
age_weight = jnp.clip(1.0 / (age_gyr + 0.01), 0.1, 10.0)
base_flux = base_flux * (
    1.0 + 0.5 * age_weight[None, None, :, None] * uv_boost[None, None, None, :]
)

# Alpha enhancement effects:
# 1. Mg b feature at ~5170 Å (stronger at high [α/Fe])
mg_feature = jnp.exp(-0.5 * ((wave - 5170.0) / 15.0) ** 2)
# 2. Fe 5270 line (weaker at high [α/Fe] because less Fe at fixed total Z)
fe_feature = jnp.exp(-0.5 * ((wave - 5270.0) / 12.0) ** 2)
# 3. Ca H&K at 3933, 3968 Å (stronger at high [α/Fe])
ca_hk = jnp.exp(-0.5 * ((wave - 3950.0) / 20.0) ** 2)

for i_alpha, afe in enumerate(alpha_grid):
    # Absorption features (subtract from continuum)
    alpha_effect = (
        0.15 * float(afe) * mg_feature  # Mg b deeper at high α
        - 0.10 * float(afe) * fe_feature  # Fe weaker at high α (negative = less absorption)
        + 0.08 * float(afe) * ca_hk  # Ca stronger at high α
    )
    # Scale by age: effect stronger for old populations
    old_weight = jnp.clip(age_gyr / 5.0, 0.0, 1.0)
    base_flux = base_flux.at[:, i_alpha, :, :].add(
        -base_flux[:, i_alpha, :, :] * alpha_effect[None, None, :] * old_weight[None, :, None]
    )

# Ensure positive
base_flux = jnp.maximum(base_flux, 1e-10)

ssp_4d = SSPData(
    ssp_wave=wave,
    ssp_flux=base_flux,
    ssp_lg_age_gyr=lg_age_gyr,
    ssp_lgmet=feh_grid,
    ssp_alpha_fe=alpha_grid,
)

print(f"4D SSP grid: {base_flux.shape}")
print(f"[Fe/H] grid: {feh_grid}")
print(f"[α/Fe] grid: {alpha_grid}")
print(f"Ages: {float(10 ** lg_age_gyr[0]):.3f} to {float(10 ** lg_age_gyr[-1]):.1f} Gyr")

# %% [markdown]
# ## 2. How [α/Fe] Changes an SSP Spectrum
#
# At fixed [Fe/H] and age, varying [α/Fe] produces distinct spectral
# signatures — primarily in absorption line strengths, not the continuum.

# %%
from tengri.sps.dsps_wrapper import has_alpha_grid, interpolate_met_alpha

assert has_alpha_grid(ssp_4d), "4D grid should be detected"

# --- FIGURE 1: SSP spectra at different [α/Fe] for an old population ---
fig, axes = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})

feh_val = -0.5
age_idx = 35  # ~8 Gyr — old population where α-effects are strongest

colors_alpha = plt.cm.coolwarm(np.linspace(0.1, 0.9, len(alpha_grid)))

for i, afe in enumerate(alpha_grid):
    sed = interpolate_met_alpha(
        ssp_4d.ssp_flux,
        ssp_4d.ssp_lgmet,
        ssp_4d.ssp_alpha_fe,
        log_z=feh_val,
        alpha_fe=float(afe),
    )
    spec = np.array(sed[age_idx])
    label = f"[α/Fe] = {float(afe):+.1f}"
    axes[0].plot(np.array(wave), spec, color=colors_alpha[i], lw=1.2, label=label)

# Mark key features
for feat_wave, feat_name in [(5170, "Mg b"), (5270, "Fe 5270"), (3950, "Ca H&K")]:
    axes[0].axvline(feat_wave, ls=":", color="grey", alpha=0.4, lw=0.5)
    axes[0].text(
        feat_wave + 10, axes[0].get_ylim()[1] * 0.95, feat_name, fontsize=7, color="grey", va="top"
    )

axes[0].set_ylabel("Flux density [Lsun/Hz/Msun]")
axes[0].legend(fontsize=8, ncol=2, loc="upper right")
axes[0].set_title(
    f"Old SSP (age ≈ {float(10 ** lg_age_gyr[age_idx]):.1f} Gyr, [Fe/H] = {feh_val})"
)
axes[0].set_xlim(3500, 9500)

# Ratio panel: normalized to solar [α/Fe]
sed_solar = interpolate_met_alpha(
    ssp_4d.ssp_flux,
    ssp_4d.ssp_lgmet,
    ssp_4d.ssp_alpha_fe,
    log_z=feh_val,
    alpha_fe=0.0,
)
spec_solar = np.array(sed_solar[age_idx])

for i, afe in enumerate(alpha_grid):
    if float(afe) == 0.0:
        continue
    sed = interpolate_met_alpha(
        ssp_4d.ssp_flux,
        ssp_4d.ssp_lgmet,
        ssp_4d.ssp_alpha_fe,
        log_z=feh_val,
        alpha_fe=float(afe),
    )
    ratio = np.array(sed[age_idx]) / (spec_solar + 1e-30)
    axes[1].plot(np.array(wave), ratio, color=colors_alpha[i], lw=1.0)

axes[1].axhline(1.0, ls="--", color="grey", lw=0.5)
axes[1].set_xlabel("Wavelength [Å]")
axes[1].set_ylabel("Ratio to [α/Fe] = 0")
axes[1].set_ylim(0.85, 1.15)
axes[1].set_xlim(3500, 9500)

for feat_wave in [5170, 5270, 3950]:
    axes[1].axvline(feat_wave, ls=":", color="grey", alpha=0.4, lw=0.5)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "14_alpha_ssp_spectra.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# The key signatures of α-enhancement:
# - **Mg b (5170 Å):** Deepens with increasing [α/Fe] (Mg is an α-element)
# - **Fe 5270:** Weakens with increasing [α/Fe] (less Fe at fixed [Fe/H])
# - **Ca H&K (3933, 3968 Å):** Deepens (Ca is an α-element)
# - **Continuum:** Nearly unchanged — α-enhancement is a LINE effect

# %% [markdown]
# ## 3. Solar [α/Fe] = 0 Reproduces Standard 3D SSPs
#
# A critical consistency check: the 4D grid at [α/Fe] = 0.0 must exactly
# match the solar-scaled (3D) SSP slice.

# %%
from tengri.sps.dsps_wrapper import interpolate_metallicity

# Extract the solar [α/Fe] slice as a 3D grid
ssp_3d_solar = ssp_4d.ssp_flux[:, 1, :, :]  # α index 1 = [α/Fe] = 0.0

# Compare: 4D interpolation at [α/Fe]=0.0 vs direct 3D slice
fig, ax = plt.subplots(figsize=(8, 4))

for feh in [-1.5, -0.5, 0.0]:
    sed_4d = interpolate_met_alpha(
        ssp_4d.ssp_flux,
        ssp_4d.ssp_lgmet,
        ssp_4d.ssp_alpha_fe,
        log_z=feh,
        alpha_fe=0.0,
    )
    sed_3d = interpolate_metallicity(ssp_3d_solar, ssp_4d.ssp_lgmet, feh)

    diff = float(jnp.max(jnp.abs(sed_4d - sed_3d)))
    ax.plot(
        np.array(wave),
        np.array(sed_4d[20] - sed_3d[20]),
        label=f"[Fe/H] = {feh}: max diff = {diff:.2e}",
    )

ax.axhline(0, ls="--", color="grey", lw=0.5)
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("4D([α/Fe]=0) − 3D difference")
ax.set_title("Consistency check: 4D at solar α = 3D (should be zero)")
ax.legend(fontsize=8)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "14_solar_alpha_consistency.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# The difference is machine epsilon — confirming perfect backward compatibility.

# %% [markdown]
# ## 4. Time-Evolving [α/Fe]: Old Stars Are α-Enhanced
#
# In real galaxies, [α/Fe] correlates with stellar age because Type Ia SNe
# (which produce Fe) have a delay time of ~40 Myr to Gyrs. We parameterize
# this as a linear ramp in lookback time.

# %%
from tengri.sps.dsps_wrapper import (
    compute_alpha_fe_evolving,
    interpolate_met_alpha_evolving,
)

# --- FIGURE 3: [α/Fe] evolution ramp ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: the ramp itself
t_universe = 13.7  # Gyr
lookback_gyr = 10.0**lg_age_gyr

for alpha_old in [0.0, 0.2, 0.4, 0.6]:
    afe_per_age = compute_alpha_fe_evolving(lg_age_gyr, alpha_old, 0.0, t_universe)
    axes[0].plot(
        np.array(lookback_gyr),
        np.array(afe_per_age),
        label=f"[α/Fe]$_{{old}}$ = +{alpha_old:.1f}",
        lw=1.5,
    )

axes[0].set_xlabel("Lookback time [Gyr]")
axes[0].set_ylabel("[α/Fe]")
axes[0].set_xlim(0, 14)
axes[0].legend(fontsize=8)
axes[0].set_title("Time-evolving [α/Fe] ramp")

# Right: SED difference between evolving and constant [α/Fe]
feh = -0.5
afe_evolving = compute_alpha_fe_evolving(lg_age_gyr, 0.4, 0.0, t_universe)
afe_constant = jnp.full(n_age, 0.2)  # average of old and young
feh_per_age = jnp.full(n_age, feh)

sed_evolving = interpolate_met_alpha_evolving(
    ssp_4d.ssp_flux,
    ssp_4d.ssp_lgmet,
    ssp_4d.ssp_alpha_fe,
    feh_per_age,
    afe_evolving,
)
sed_constant = interpolate_met_alpha_evolving(
    ssp_4d.ssp_flux,
    ssp_4d.ssp_lgmet,
    ssp_4d.ssp_alpha_fe,
    feh_per_age,
    afe_constant,
)

# Weight by a simple declining SFH to get a CSP
sfr = jnp.exp(-lookback_gyr / 5.0)
weights = (
    sfr
    * jnp.concatenate(
        [
            jnp.array([10 ** lg_age_gyr[1] - 10 ** lg_age_gyr[0]]),
            0.5 * (10 ** lg_age_gyr[2:] - 10 ** lg_age_gyr[:-2]),
            jnp.array([10 ** lg_age_gyr[-1] - 10 ** lg_age_gyr[-2]]),
        ]
    )
    * 1e9
)  # convert Gyr to yr

csp_evolving = jnp.einsum("i,iw->w", weights, sed_evolving)
csp_constant = jnp.einsum("i,iw->w", weights, sed_constant)

ratio = np.array(csp_evolving / (csp_constant + 1e-30))
axes[1].plot(np.array(wave), ratio, color=COLORS.get("geovi", "C0"), lw=1.2)
axes[1].axhline(1.0, ls="--", color="grey", lw=0.5)
axes[1].set_xlabel("Wavelength [Å]")
axes[1].set_ylabel("Evolving / Constant [α/Fe]")
axes[1].set_title("CSP: evolving vs constant [α/Fe]")

for feat_wave, _feat_name in [(5170, "Mg b"), (5270, "Fe"), (3950, "Ca H&K")]:
    axes[1].axvline(feat_wave, ls=":", color="grey", alpha=0.4, lw=0.5)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "14_evolving_alpha.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. The Salaris Relation: [Fe/H] vs [M/H]
#
# Different SSP libraries use different metallicity conventions:
# - **α-MC** (Park+2024): grid in [Fe/H] (iron abundance)
# - **sMILES** (Knowles+2023): grid in [M/H] (total metallicity)
#
# The Salaris relation connects them:
# $$ [\text{M/H}] = [\text{Fe/H}] + 0.66154 \times [\alpha/\text{Fe}] + 0.20465 \times [\alpha/\text{Fe}]^2 $$

# %%
from tengri.sps.dsps_wrapper import salaris_feh_from_mh, salaris_mh_from_feh

# --- FIGURE 4: Salaris relation ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: [M/H] - [Fe/H] offset as function of [α/Fe]
afe_range = np.linspace(-0.3, 0.7, 100)
offsets = [salaris_mh_from_feh(0.0, a) for a in afe_range]
axes[0].plot(afe_range, offsets, "k-", lw=2)
axes[0].axhline(0, ls="--", color="grey", lw=0.5)
axes[0].axvline(0, ls="--", color="grey", lw=0.5)

# Annotate key values
for afe_val, color in [(0.0, "C0"), (0.3, "C1"), (0.4, "C2")]:
    mh = salaris_mh_from_feh(0.0, afe_val)
    axes[0].plot(afe_val, mh, "o", ms=8, color=color)
    axes[0].annotate(
        f"[α/Fe]={afe_val:+.1f}\nΔ={mh:+.3f}",
        (afe_val, mh),
        textcoords="offset points",
        xytext=(10, -15),
        fontsize=7,
        color=color,
    )

axes[0].set_xlabel("[α/Fe] [dex]")
axes[0].set_ylabel("[M/H] − [Fe/H] [dex]")
axes[0].set_title("Salaris relation: offset from total vs iron metallicity")

# Right: same [Fe/H] at different [α/Fe] → different [M/H]
feh_vals = np.array([-2.0, -1.5, -1.0, -0.5, 0.0])
for i, afe in enumerate([0.0, 0.2, 0.4]):
    mh_vals = [salaris_mh_from_feh(f, afe) for f in feh_vals]
    axes[1].plot(feh_vals, mh_vals, "o-", ms=6, label=f"[α/Fe] = +{afe:.1f}", color=f"C{i}")

axes[1].plot([-2.5, 0.5], [-2.5, 0.5], "k--", lw=0.5, label="[M/H] = [Fe/H]")
axes[1].set_xlabel("[Fe/H]")
axes[1].set_ylabel("[M/H]")
axes[1].legend(fontsize=8)
axes[1].set_title("[Fe/H] → [M/H] mapping at different [α/Fe]")
axes[1].set_aspect("equal")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "14_salaris_relation.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# At [α/Fe] = 0 (solar), [M/H] = [Fe/H] exactly. At [α/Fe] = +0.4
# (typical for massive ellipticals), [M/H] is ~0.30 dex higher than [Fe/H].

# %% [markdown]
# ## 6. When Does Alpha-Enhancement Matter?
#
# The effect of [α/Fe] is strongest in spectral line indices and weak in
# broadband colors. This has practical implications for fitting strategies.

# %%
# --- FIGURE 5: Broadband color sensitivity to [α/Fe] ---
# Synthetic "broadband" integration in u, g, r, i, z-like windows
band_centers = [3800, 4800, 6200, 7600, 8800]
band_widths = [500, 800, 700, 700, 700]
band_names = ["u", "g", "r", "i", "z"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Left: broadband color change with [α/Fe] (old pop)
age_idx_old = 35  # ~8 Gyr
colors_by_alpha = {}
for afe_val in alpha_grid:
    sed = interpolate_met_alpha(
        ssp_4d.ssp_flux,
        ssp_4d.ssp_lgmet,
        ssp_4d.ssp_alpha_fe,
        log_z=-0.5,
        alpha_fe=float(afe_val),
    )
    spec = np.array(sed[age_idx_old])
    mags = []
    for bc, bw in zip(band_centers, band_widths):
        mask = (np.array(wave) > bc - bw / 2) & (np.array(wave) < bc + bw / 2)
        flux_band = np.mean(spec[mask]) if mask.sum() > 0 else 1e-30
        mags.append(-2.5 * np.log10(max(flux_band, 1e-30)))
    colors_by_alpha[float(afe_val)] = mags

# Plot g-r and u-g as function of [α/Fe]
gr_colors = [colors_by_alpha[a][2] - colors_by_alpha[a][1] for a in np.array(alpha_grid)]
ug_colors = [colors_by_alpha[a][1] - colors_by_alpha[a][0] for a in np.array(alpha_grid)]

axes[0].plot(np.array(alpha_grid), gr_colors, "o-", color="C0", label="g − r")
axes[0].plot(np.array(alpha_grid), ug_colors, "s-", color="C1", label="u − g")
axes[0].set_xlabel("[α/Fe] [dex]")
axes[0].set_ylabel("Color [mag]")
axes[0].legend()
axes[0].set_title("Broadband colors vs [α/Fe] (old SSP, 8 Gyr)")

# Right: spectral index sensitivity (Mg b vs Fe 5270)
mg_b_ew = []
fe_5270_ew = []
for afe_val in alpha_grid:
    sed = interpolate_met_alpha(
        ssp_4d.ssp_flux,
        ssp_4d.ssp_lgmet,
        ssp_4d.ssp_alpha_fe,
        log_z=-0.5,
        alpha_fe=float(afe_val),
    )
    spec = np.array(sed[age_idx_old])
    # Pseudo-EW: sum of (1 - flux/continuum) in feature window
    for feat_wav, ew_list in [(5170, mg_b_ew), (5270, fe_5270_ew)]:
        feat_mask = (np.array(wave) > feat_wav - 20) & (np.array(wave) < feat_wav + 20)
        cont_mask = (np.array(wave) > feat_wav - 60) & (np.array(wave) < feat_wav - 30)
        if feat_mask.sum() > 0 and cont_mask.sum() > 0:
            cont = np.mean(spec[cont_mask])
            feat = np.mean(spec[feat_mask])
            ew_list.append(1.0 - feat / (cont + 1e-30))
        else:
            ew_list.append(0.0)

axes[1].plot(np.array(alpha_grid), mg_b_ew, "o-", color="C2", label="Mg b (5170 Å)")
axes[1].plot(np.array(alpha_grid), fe_5270_ew, "s-", color="C3", label="Fe 5270 Å")
axes[1].set_xlabel("[α/Fe] [dex]")
axes[1].set_ylabel("Pseudo equivalent width")
axes[1].legend()
axes[1].set_title("Spectral index sensitivity to [α/Fe]")

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "14_alpha_sensitivity.png"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# **Key takeaway:** Broadband colors change by < 0.05 mag across the full
# [α/Fe] range, while spectral indices change by factors of ~2. This means:
# - **Photometric fitting:** [α/Fe] has minimal impact. A single metallicity
#   parameter with `effective_metallicity()` approximation is adequate.
# - **Spectroscopic fitting:** [α/Fe] matters. Use 4D alpha-enhanced grids.

# %% [markdown]
# ## Summary
#
# | Feature | Implementation | Parameters |
# |---------|---------------|------------|
# | Global [α/Fe] | `met_alpha_fe=Uniform(-0.2, 0.6)` | +1 free param |
# | Time-evolving [α/Fe] | `alpha_fe_evolving=True` | +1-2 free params |
# | No alpha (default) | Standard 3D SSPs | 0 extra params |
# | [Fe/H] ↔ [M/H] | `salaris_mh_from_feh()` | Convention conversion |
#
# **When 4D grids are loaded** (detected automatically), bilinear (Z, [α/Fe])
# interpolation is used throughout the pipeline. **When not loaded** (the common
# case), the code uses standard 3D interpolation with zero overhead.
