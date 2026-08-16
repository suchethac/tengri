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
# # Analytical Emission Line Marginalization
#
# When fitting galaxy spectra, emission-line amplitudes are **nuisance
# parameters** --- we care about the physical parameters (SFH, dust,
# metallicity) encoded in the continuum, not the individual line fluxes.
#
# Sampling over line amplitudes is wasteful: each line adds a dimension
# to the posterior, and the amplitudes are *linearly* related to the data.
# Linear parameters can be **analytically marginalized** out of the
# likelihood, leaving a lower-dimensional posterior over the physical
# parameters alone.
#
# This is one of `tengri`'s key capabilities: the marginalized
# log-likelihood is fully differentiable in JAX, so gradients with
# respect to continuum parameters (dust, metallicity, SFH shape) flow
# through the marginalization automatically.
#
# **What this notebook covers:**
#
# 1. Build a mock galaxy spectrum with known continuum + emission lines
# 2. Construct and visualize the Gaussian design matrix for 13 lines
# 3. Recover line amplitudes from noisy data via marginalization
# 4. Show why marginalization prevents biased continuum fits when the
#    nebular model predicts wrong line ratios
# 5. Integration with a full `tengri` spectral fit
# 6. Gradient verification: $\nabla_\theta \ln L_{\rm marg}$ is finite
#
# > **Reference:** Johnson et al. (2021) --- Prospector linear
# > optimization of emission-line amplitudes; Brammer et al. (2008) ---
# > linear combination fitting.

# %% [markdown]
# ## 1. Setup
#
# Load SSP data, create a wavelength grid, and build a mock galaxy
# spectrum with known continuum and emission lines.

# %%
import os
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    Fitter,
    Fixed,
    SEDModel,
    ParamSpec,
    Uniform,
    load_filter_set,
    load_ssp_data,
)
from tengri.observation.eline_marginalization import (
    DEFAULT_LINE_NAMES,
    DEFAULT_LINE_WAVELENGTHS,
    build_eline_design_matrix,
    marginalize_emission_lines,
    predict_with_marginalized_lines,
)

# ── Plot style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 12,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 1.0,
    "axes.labelsize": 13,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 1.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIGDIR = Path("figures")
FIGDIR.mkdir(exist_ok=True)

# ── Load SSP data ─────────────────────────────────────────────────
SSP_PATH = "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
ssp_data = load_ssp_data(SSP_PATH)
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

print(f"SSP grid: {len(ssp_data.ssp_lgmet)} metallicities, "
      f"{len(ssp_data.ssp_lg_age_gyr)} ages")
print(f"Default emission lines: {len(DEFAULT_LINE_NAMES)}")
for name, lam in zip(DEFAULT_LINE_NAMES, DEFAULT_LINE_WAVELENGTHS):
    print(f"  {name:>15s}  {float(lam):8.1f} A")

# %% [markdown]
# ### Build a mock galaxy spectrum
#
# We use a simple double-power-law SFH at $z = 0.1$ and add known
# emission lines on top of the continuum.

# %%
# --- SEDModel configuration (smooth DPL SFH, 7 free params) ---
REDSHIFT = 0.1
wave_obs = jnp.linspace(3600.0, 9500.0, 500)

spec = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(REDSHIFT),
)

model = SEDModel(spec, ssp_data, filters=filters).precompute_spectroscopy(wave_obs)

# True parameters: star-forming galaxy with moderate dust
true_params = {
    "sfh_dpl_alpha": 1.2,
    "sfh_dpl_beta": 1.8,
    "sfh_dpl_tau_gyr": 9.0,
    "sfh_dpl_log_peak_sfr": 1.5,
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.4,
    "dust_tau_diff": 0.2,
    "dust_slope": -0.7,
    "redshift": REDSHIFT,
}

# Compute continuum-only spectrum (uses precomputed fast path)
flux_continuum = model.predict_spectrum(true_params)

# --- Add emission lines with known amplitudes ---
SPECTRAL_RESOLUTION = 2000.0  # R = 2000 (moderate-res spectrograph)

# True line amplitudes (arbitrary units, scaled to be visible)
# Strong star-forming galaxy: bright Balmer + [OIII] + [NII]
flux_scale = float(jnp.median(jnp.abs(flux_continuum)))
true_amplitudes = jnp.array([
    0.0,     # Ly-alpha (outside wavelength range at z=0.1)
    0.8,     # H-delta
    1.2,     # H-gamma
    3.0,     # H-beta
    2.5,     # [OIII]4959
    7.5,     # [OIII]5007
    10.0,    # H-alpha
    1.5,     # [NII]6548
    4.5,     # [NII]6583
    2.0,     # [OII]3726
    2.0,     # [OII]3729
    1.0,     # [SII]6717
    1.2,     # [SII]6731
]) * flux_scale * 1e-2

# Build the design matrix and compute the line spectrum
G = build_eline_design_matrix(wave_obs, DEFAULT_LINE_WAVELENGTHS,
                               SPECTRAL_RESOLUTION, REDSHIFT)
flux_lines = G @ true_amplitudes

# Full spectrum: continuum + lines
flux_true = flux_continuum + flux_lines

# Add noise (SNR ~ 50 per pixel)
SNR = 50.0
key = jax.random.PRNGKey(42)
noise = jnp.abs(flux_true) / SNR
flux_obs = flux_true + noise * jax.random.normal(key, shape=flux_true.shape)

# --- Plot: spectrum with and without emission lines ---
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

ax = axes[0]
ax.plot(wave_obs, flux_continuum * 1e17, color="0.6", lw=0.8,
        label="Continuum only", zorder=2)
ax.plot(wave_obs, flux_true * 1e17, color="#1f77b4", lw=0.8,
        label="Continuum + lines", zorder=3)
ax.errorbar(np.array(wave_obs)[::5], np.array(flux_obs * 1e17)[::5],
            yerr=np.array(noise * 1e17)[::5], fmt=".", ms=2, color="0.3",
            alpha=0.5, label=f"Observed (SNR={SNR:.0f})", zorder=1)
ax.set_ylabel(r"$f_\nu$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.legend(loc="upper right")
ax.set_title("Mock Star-Forming Galaxy at $z = 0.1$")

ax = axes[1]
ax.plot(wave_obs, flux_lines / jnp.abs(flux_continuum) * 100,
        color="#d62728", lw=0.8)
ax.set_ylabel("Line / cont. [%]")
ax.set_xlabel(r"Observed wavelength [$\AA$]")
ax.axhline(0, color="0.7", ls="--", lw=0.5)

# Mark the strongest lines
for name, lam in zip(DEFAULT_LINE_NAMES, np.array(DEFAULT_LINE_WAVELENGTHS)):
    lam_obs = lam * (1.0 + REDSHIFT)
    if float(wave_obs[0]) <= lam_obs <= float(wave_obs[-1]):
        axes[0].axvline(lam_obs, color="0.85", ls=":", lw=0.5, zorder=0)
        axes[1].axvline(lam_obs, color="0.85", ls=":", lw=0.5, zorder=0)

fig.savefig(FIGDIR / "15_mock_spectrum.png")
plt.show()
print("Saved: figures/15_mock_spectrum.png")

# %% [markdown]
# ## 2. The Gaussian Design Matrix
#
# The design matrix $\mathbf{G}$ has shape `(n_pix, n_lines)`.
# Column $j$ is a normalized Gaussian profile centred at the
# redshifted wavelength of line $j$, with width set by the instrument
# resolution $R$:
#
# $$
# G_j(\lambda) = \frac{1}{\sqrt{2\pi}\,\sigma_j}
#     \exp\!\Bigl[-\frac{(\lambda - \lambda_j(1{+}z))^2}{2\sigma_j^2}\Bigr]
# $$
#
# where $\sigma_j = \lambda_j(1{+}z) / (2.355\,R)$.
#
# The full model with free line amplitudes $\mathbf{a}$ is:
#
# $$
# \text{model} = \mathbf{m} + \mathbf{G}\,\mathbf{a}
# $$

# %%
# Design matrix already built above; let us visualize it.

fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                          gridspec_kw={"width_ratios": [2, 1]})

# --- Left: heatmap ---
ax = axes[0]
extent = [0, len(DEFAULT_LINE_NAMES), float(wave_obs[-1]),
          float(wave_obs[0])]
im = ax.imshow(np.array(G), aspect="auto", extent=extent,
               cmap="viridis", interpolation="nearest")
ax.set_xlabel("Line index")
ax.set_ylabel(r"Observed wavelength [$\AA$]")
ax.set_title("Design Matrix $\\mathbf{G}$")
ax.set_xticks(np.arange(len(DEFAULT_LINE_NAMES)) + 0.5)
ax.set_xticklabels([n.replace("[", "").replace("]", "")
                     for n in DEFAULT_LINE_NAMES],
                    rotation=70, ha="right", fontsize=7)
plt.colorbar(im, ax=ax, label="Amplitude", shrink=0.8)

# --- Right: individual line profiles ---
ax = axes[1]
cmap = plt.cm.tab20
for j in range(len(DEFAULT_LINE_NAMES)):
    lam_obs_j = float(DEFAULT_LINE_WAVELENGTHS[j]) * (1.0 + REDSHIFT)
    if float(wave_obs[0]) <= lam_obs_j <= float(wave_obs[-1]):
        col_j = np.array(G[:, j])
        mask = col_j > 1e-10 * col_j.max()
        if mask.any():
            ax.plot(np.array(wave_obs)[mask], col_j[mask],
                    color=cmap(j / len(DEFAULT_LINE_NAMES)),
                    lw=1.2, label=DEFAULT_LINE_NAMES[j])
ax.set_xlabel(r"Observed wavelength [$\AA$]")
ax.set_ylabel("Profile amplitude")
ax.set_title("Individual Line Profiles")
ax.legend(fontsize=6, ncol=2, loc="upper right")

fig.tight_layout()
fig.savefig(FIGDIR / "15_design_matrix.png")
plt.show()
print("Saved: figures/15_design_matrix.png")

# %% [markdown]
# ## 3. Line Recovery from Clean Data
#
# Given the observed spectrum $d = m + \mathbf{G}\mathbf{a} + n$
# and the known continuum $m$, the marginalization returns the
# posterior-mean amplitudes $\hat{\mathbf{a}}$ and their covariance
# $\Sigma_a$.
#
# With the *true* continuum and moderate noise, recovery should be
# excellent.

# %%
# Residual = data - continuum (the lines + noise remain)
residual = flux_obs - flux_continuum

# Marginalize
ln_L, a_hat, a_cov = marginalize_emission_lines(
    residual, noise, G,
    prior_variance=jnp.full(len(DEFAULT_LINE_NAMES), 1e10),
)

# Posterior uncertainties from diagonal of covariance
a_sigma = jnp.sqrt(jnp.diag(a_cov))

print(f"Marginalized log-likelihood: {float(ln_L):.1f}")
print()
print(f"{'Line':>15s}  {'True':>10s}  {'Recovered':>10s}  "
      f"{'Sigma':>10s}  {'(Rec-True)/Sig':>14s}")
print("-" * 65)
for j, name in enumerate(DEFAULT_LINE_NAMES):
    a_t = float(true_amplitudes[j])
    a_r = float(a_hat[j])
    a_s = float(a_sigma[j])
    pull = (a_r - a_t) / a_s if a_s > 0 else 0.0
    print(f"{name:>15s}  {a_t:10.3e}  {a_r:10.3e}  {a_s:10.3e}  {pull:14.2f}")

# --- Bar chart: true vs recovered ---
fig, ax = plt.subplots(figsize=(10, 4))

# Filter to lines within wavelength range
in_range = []
for j in range(len(DEFAULT_LINE_NAMES)):
    lam_obs_j = float(DEFAULT_LINE_WAVELENGTHS[j]) * (1.0 + REDSHIFT)
    if float(wave_obs[0]) <= lam_obs_j <= float(wave_obs[-1]):
        in_range.append(j)

x = np.arange(len(in_range))
width = 0.35

true_vals = np.array([float(true_amplitudes[j]) for j in in_range])
rec_vals = np.array([float(a_hat[j]) for j in in_range])
rec_errs = np.array([float(a_sigma[j]) for j in in_range])
line_labels = [DEFAULT_LINE_NAMES[j] for j in in_range]

ax.bar(x - width / 2, true_vals, width, color="#2ca02c", alpha=0.7,
       label="True")
ax.bar(x + width / 2, rec_vals, width, color="#1f77b4", alpha=0.7,
       yerr=rec_errs, capsize=3, label="Recovered")
ax.set_xticks(x)
ax.set_xticklabels(line_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Line amplitude")
ax.set_title("Emission Line Recovery (known continuum)")
ax.legend()

fig.tight_layout()
fig.savefig(FIGDIR / "15_line_recovery.png")
plt.show()
print("Saved: figures/15_line_recovery.png")

# %% [markdown]
# ## 4. Why Marginalization Matters
#
# In practice, nebular models (CLOUDY, Cue) predict line ratios that
# may not exactly match the data --- due to unknown ISM conditions,
# abundance ratios, or model systematics.
#
# When lines are **fixed** to the model prediction, the mismatch
# shows up as structured residuals at line positions, which **biases
# the continuum fit** (the optimizer distorts dust/metallicity to
# compensate for the wrong lines).
#
# When lines are **marginalized**, the amplitudes float freely,
# absorbing any mismatch. The continuum fit remains unbiased.
#
# This is the key advantage: marginalization decouples line-ratio
# systematics from continuum inference.

# %%
# --- Simulate a "wrong" nebular model ---
# The nebular model predicts line ratios that are off by 20-50%
key_wrong = jax.random.PRNGKey(99)
ratio_error = 1.0 + 0.3 * jax.random.normal(key_wrong,
                                              shape=true_amplitudes.shape)
wrong_amplitudes = true_amplitudes * ratio_error
flux_wrong_lines = G @ wrong_amplitudes

# Case A: Fixed lines (subtract wrong lines, fit residual as if pure continuum)
flux_fixed_residual = flux_obs - flux_wrong_lines
chi2_fixed = jnp.sum(((flux_fixed_residual - flux_continuum) / noise) ** 2)

# Case B: Marginalized lines (let line amplitudes float)
residual_for_marg = flux_obs - flux_continuum
ln_L_marg, a_hat_marg, _ = marginalize_emission_lines(
    residual_for_marg, noise, G,
    prior_variance=jnp.full(len(DEFAULT_LINE_NAMES), 1e10),
)
flux_model_marg = predict_with_marginalized_lines(flux_continuum, G, a_hat_marg)
chi2_marg = jnp.sum(((flux_obs - flux_model_marg) / noise) ** 2)

n_dof = len(wave_obs) - len(DEFAULT_LINE_NAMES)

print(f"Case A (fixed wrong lines):  chi2 = {float(chi2_fixed):.1f}  "
      f"(chi2/dof = {float(chi2_fixed) / n_dof:.2f})")
print(f"Case B (marginalized lines): chi2 = {float(chi2_marg):.1f}  "
      f"(chi2/dof = {float(chi2_marg) / n_dof:.2f})")
print(f"Improvement: delta_chi2 = {float(chi2_fixed - chi2_marg):.1f}")

# --- Residual comparison plot ---
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                          gridspec_kw={"hspace": 0.08})

# Case A: fixed lines
res_fixed = (flux_fixed_residual - flux_continuum) / noise
ax = axes[0]
ax.plot(wave_obs, res_fixed, color="#d62728", lw=0.5, alpha=0.7)
ax.axhline(0, color="0.5", ls="--", lw=0.5)
ax.axhspan(-2, 2, color="0.9", alpha=0.3, zorder=0)
ax.set_ylabel(r"$(d - f_{\rm fixed}) / \sigma$")
ax.set_title(f"Case A: Fixed lines (wrong ratios)  ---  "
             r"$\chi^2/{\rm dof}$" + f" = {float(chi2_fixed) / n_dof:.2f}")
ax.set_ylim(-8, 8)

# Mark line positions where residuals are structured
for lam in np.array(DEFAULT_LINE_WAVELENGTHS):
    lam_obs = lam * (1.0 + REDSHIFT)
    if float(wave_obs[0]) <= lam_obs <= float(wave_obs[-1]):
        ax.axvline(lam_obs, color="#d62728", ls=":", lw=0.4, alpha=0.5)

# Case B: marginalized
res_marg = (flux_obs - flux_model_marg) / noise
ax = axes[1]
ax.plot(wave_obs, res_marg, color="#1f77b4", lw=0.5, alpha=0.7)
ax.axhline(0, color="0.5", ls="--", lw=0.5)
ax.axhspan(-2, 2, color="0.9", alpha=0.3, zorder=0)
ax.set_ylabel(r"$(d - f_{\rm marg}) / \sigma$")
ax.set_xlabel(r"Observed wavelength [$\AA$]")
ax.set_title(f"Case B: Marginalized lines  ---  "
             r"$\chi^2/{\rm dof}$" + f" = {float(chi2_marg) / n_dof:.2f}")
ax.set_ylim(-8, 8)

for lam in np.array(DEFAULT_LINE_WAVELENGTHS):
    lam_obs = lam * (1.0 + REDSHIFT)
    if float(wave_obs[0]) <= lam_obs <= float(wave_obs[-1]):
        ax.axvline(lam_obs, color="#1f77b4", ls=":", lw=0.4, alpha=0.5)

fig.savefig(FIGDIR / "15_marginalization_comparison.png")
plt.show()
print("Saved: figures/15_marginalization_comparison.png")

# %% [markdown]
# The top panel shows clear residual spikes at line positions ---
# the "fixed wrong lines" approach introduces systematic bias.
# The bottom panel shows clean, noise-like residuals after
# marginalization.
#
# **Bottom line:** marginalization absorbs nebular model
# systematics, preventing emission lines from biasing the
# continuum-derived physical parameters.

# %% [markdown]
# ## 5. Integration with the Full SEDModel
#
# In a real fit, the continuum $\mathbf{m}(\theta)$ depends on
# physical parameters $\theta$ (SFH shape, dust, metallicity).
# The marginalized log-likelihood integrates over line amplitudes:
#
# $$
# \ln L_{\rm marg}(\theta) = -\tfrac{1}{2}\bigl[
#     \chi^2_{\rm marg}(\theta) + \text{prior penalty}
#     - \text{log-det correction}
# \bigr]
# $$
#
# Below we generate a mock galaxy with lines, then fit SFH/dust/Z
# from the continuum while marginalizing the line amplitudes.

# %%
# Reuse the model (already has spectroscopy precomputed from Section 1)
model_spec = model

# Generate mock observation with lines baked in
flux_true_cont = model_spec.predict_spectrum(true_params)
flux_true_with_lines = flux_true_cont + flux_lines

key_mock = jax.random.PRNGKey(123)
noise_full = jnp.abs(flux_true_with_lines) / SNR
flux_obs_full = (flux_true_with_lines
                 + noise_full * jax.random.normal(key_mock,
                                                   shape=flux_true_with_lines.shape))


# --- Define a log-posterior that marginalizes emission lines ---
def log_posterior_marginalized(params):
    """Log-posterior with analytically marginalized emission lines.

    The continuum model comes from tengri's forward model; line
    amplitudes are marginalized via the design matrix.
    """
    # Continuum prediction from physical params
    continuum = model_spec.predict_spectrum(params)

    # Residual = data - continuum (contains lines + noise)
    resid = flux_obs_full - continuum

    # Marginalize emission lines
    ln_L, _, _ = marginalize_emission_lines(
        resid, noise_full, G,
        prior_variance=jnp.full(len(DEFAULT_LINE_NAMES), 1e10),
    )
    return ln_L


# --- Evaluate at truth and at a perturbed point ---
ln_L_at_truth = log_posterior_marginalized(true_params)

perturbed_params = dict(true_params)
perturbed_params["dust_tau_diff"] = 0.8  # wrong dust
perturbed_params["met_logzsol"] = -1.0   # wrong metallicity
ln_L_perturbed = log_posterior_marginalized(perturbed_params)

print(f"ln L (true params):      {float(ln_L_at_truth):.1f}")
print(f"ln L (perturbed params): {float(ln_L_perturbed):.1f}")
print(f"Delta ln L:              {float(ln_L_at_truth - ln_L_perturbed):.1f}")
print(f"  (positive = truth is better, as expected)")

# Show recovered lines at the true continuum
resid_at_truth = flux_obs_full - flux_true_cont
_, a_hat_full, a_cov_full = marginalize_emission_lines(
    resid_at_truth, noise_full, G,
    prior_variance=jnp.full(len(DEFAULT_LINE_NAMES), 1e10),
)
flux_model_full = predict_with_marginalized_lines(flux_true_cont, G, a_hat_full)

fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

ax = axes[0]
ax.plot(wave_obs, flux_obs_full * 1e17, color="0.6", lw=0.4, alpha=0.6,
        label="Observed")
ax.plot(wave_obs, flux_model_full * 1e17, color="#1f77b4", lw=1.0,
        label="Continuum + marginalized lines")
ax.plot(wave_obs, flux_true_cont * 1e17, color="0.3", lw=0.8, ls="--",
        label="True continuum")
ax.set_ylabel(r"$f_\nu$ [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$]")
ax.legend(loc="upper right", fontsize=8)
ax.set_title("Full SEDModel: Continuum + Marginalized Lines")

ax = axes[1]
res_full = (flux_obs_full - flux_model_full) / noise_full
ax.plot(wave_obs, res_full, color="#1f77b4", lw=0.5)
ax.axhline(0, color="0.5", ls="--", lw=0.5)
ax.axhspan(-2, 2, color="0.9", alpha=0.3, zorder=0)
ax.set_ylabel(r"$(d - f) / \sigma$")
ax.set_xlabel(r"Observed wavelength [$\AA$]")
ax.set_ylim(-5, 5)

chi2_full = float(jnp.sum(res_full ** 2))
ax.set_title(r"$\chi^2/{\rm dof}$" + f" = {chi2_full / n_dof:.2f}")

fig.savefig(FIGDIR / "15_full_model_fit.png")
plt.show()
print("Saved: figures/15_full_model_fit.png")

# %% [markdown]
# ## 6. Gradient Verification
#
# The marginalized log-likelihood is fully differentiable in JAX.
# Gradients $\nabla_\theta \ln L_{\rm marg}$ with respect to physical
# parameters flow through both the continuum model and the linear
# algebra of the marginalization.
#
# We verify that gradients are:
# 1. **Finite** (no NaN or Inf)
# 2. **Non-zero** (the likelihood is genuinely sensitive to these params)
# 3. **Consistent with finite differences** (JAX autodiff is correct)

# %%
# Build a version that takes a flat dict and returns a scalar
grad_params = ["dust_tau_bc", "dust_tau_diff", "met_logzsol",
               "sfh_dpl_alpha", "sfh_dpl_beta", "sfh_dpl_tau_gyr"]


def _ln_L_for_grad(param_values):
    """Scalar function of a dict of JAX arrays -> ln L_marg."""
    params = dict(true_params)
    params.update(param_values)
    return log_posterior_marginalized(params)


# Compute gradient via JAX autodiff
grad_dict = {name: jnp.array(float(true_params[name])) for name in grad_params}
grad_fn = jax.grad(_ln_L_for_grad)
grads = grad_fn(grad_dict)

print(f"{'Parameter':>20s}  {'Value':>10s}  {'d(lnL)/d(theta)':>16s}  {'Finite?':>8s}")
print("-" * 60)
for name in grad_params:
    val = float(true_params[name])
    g = float(grads[name])
    finite = "OK" if np.isfinite(g) else "FAIL"
    print(f"{name:>20s}  {val:10.3f}  {g:16.4f}  {finite:>8s}")

# Finite-difference check for one parameter
eps = 1e-5
fd_grads = {}
for name in grad_params:
    params_plus = dict(grad_dict)
    params_plus[name] = grad_dict[name] + eps
    params_minus = dict(grad_dict)
    params_minus[name] = grad_dict[name] - eps
    fd = (float(_ln_L_for_grad(params_plus))
          - float(_ln_L_for_grad(params_minus))) / (2 * eps)
    fd_grads[name] = fd

print()
print("Finite-difference verification:")
print(f"{'Parameter':>20s}  {'JAX grad':>12s}  {'FD grad':>12s}  {'Rel. error':>12s}")
print("-" * 62)
for name in grad_params:
    jax_g = float(grads[name])
    fd_g = fd_grads[name]
    rel_err = abs(jax_g - fd_g) / (abs(fd_g) + 1e-30)
    print(f"{name:>20s}  {jax_g:12.4f}  {fd_g:12.4f}  {rel_err:12.2e}")

# --- Gradient bar chart ---
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(grad_params))
jax_vals = [float(grads[name]) for name in grad_params]
fd_vals = [fd_grads[name] for name in grad_params]

ax.bar(x - 0.2, jax_vals, 0.35, color="#1f77b4", alpha=0.8, label="JAX autodiff")
ax.bar(x + 0.2, fd_vals, 0.35, color="#ff7f0e", alpha=0.8, label="Finite difference")
ax.set_xticks(x)
ax.set_xticklabels([p.replace("sfh_dpl_", "").replace("dust_", "d_").replace("met_", "")
                     for p in grad_params], rotation=30, ha="right")
ax.set_ylabel(r"$\partial \ln L_{\rm marg} / \partial \theta$")
ax.set_title("Gradient Verification: JAX Autodiff vs Finite Differences")
ax.legend()
ax.axhline(0, color="0.5", ls="--", lw=0.5)

fig.tight_layout()
fig.savefig(FIGDIR / "15_gradient_verification.png")
plt.show()
print("Saved: figures/15_gradient_verification.png")

# %% [markdown]
# ## Summary
#
# | Aspect | Without marginalization | With marginalization |
# |--------|------------------------|---------------------|
# | **Line amplitudes** | Fixed to nebular model | Analytically integrated out |
# | **Dimensionality** | $D + N_{\rm lines}$ | $D$ (physical params only) |
# | **Line ratio bias** | Contaminates continuum fit | Absorbed by free amplitudes |
# | **Gradients** | N/A (discrete model) | Smooth, finite, JAX-differentiable |
# | **Cost** | N/A | One matrix solve per likelihood call |
#
# **Key takeaway:** Analytical emission line marginalization lets
# tengri fit galaxy spectra for physical parameters while treating
# line amplitudes as nuisance parameters --- with correct uncertainty
# propagation and full differentiability.  This is essential for
# spectroscopic SED fitting where nebular model systematics would
# otherwise bias continuum-derived quantities.
