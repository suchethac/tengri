# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Diagnostics: Fisher, Saliency & Chain Quality
#
# Exploit automatic differentiation to diagnose parameter constraints and chain quality.
#
# ## What you'll learn
#
# - **Fisher Information** — forecast parameter errors from Hessian at MAP
# - **Saliency** (gradient SEDs) — which wavelengths constrain which parameters
# - **Photometry sensitivity** — filter importance ranking for each parameter
# - **Autocorrelation** — assess MCMC chain quality and effective sample size (ESS)
#
# ## Prerequisites
#
# [`04_fitting_spectra.py`](04_fitting_spectra.py) (single-galaxy fitting) and
# [`06_inference_methods.py`](06_inference_methods.py) (inference method overview).
# Post-fit diagnostic tools; run after MCMC/VI to understand your constraints.

# %% [markdown]
# ## 0. Setup and mock data

# %%
import os
import sys
import warnings

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))
except NameError:
    _nb_dir = os.getcwd()
    _repo_root = os.path.abspath(os.path.join(_nb_dir, ".."))

_src = os.path.join(_repo_root, "src")
if os.path.isdir(os.path.join(_src, "tengri")):
    sys.path.insert(0, _src)
sys.path.insert(0, _repo_root)
sys.path.insert(0, _nb_dir)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    generate_mock,
    load_ssp_data,
)
from tengri.analysis.diagnostics.fisher import (
    compute_fisher_matrix,
    fisher_parameter_errors,
    fisher_correlation_matrix,
)
from tengri.analysis.diagnostics.saliency import (
    compute_all_gradient_seds,
    compute_photometry_sensitivity,
)
from tengri.analysis.diagnostics.autocorrelation import (
    check_chain_length,
    effective_sample_size,
)
from tengri.analysis.plotting import setup_style, COLORS, SDSS_WAVE_EFF, SDSS_BANDS
from _plot_style import setup_style as _setup_style

# Use project style
try:
    _setup_style()
except Exception:
    setup_style()

# %% [markdown]
# Build a smooth 7-parameter model and generate mock photometry. We reuse the
# exact setup pattern from ``00_quickstart.py`` so every diagnostic below is
# computed on the same model the tutorial user learned on.

# %%
# Load SSP library (path convention matches the rest of the spine).
SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if not os.path.exists(SSP_PATH):
    SSP_PATH = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
ssp_data = load_ssp_data(SSP_PATH)

# 5-band SDSS photometry observation
BANDS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
obs = Observation(photometry=Photometry.from_names(BANDS))

# 7-D smooth-SFH parameter spec (tsnorm shape + dust + metallicity, fixed z).
spec = Parameters(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Fixed(0.0),
    sfh_tsnorm_trunc=Fixed(3.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.05),
)
model = SEDModel(spec, ssp_data, observation=obs)

# Ground-truth parameters and a mock observation at SNR = 20.
# Use the model's MockData-returning helper so the rest of the
# notebook can use attribute access (mock.flux_obs / mock.noise).
truth = spec.sample(jax.random.PRNGKey(42))
mock = model.mock(truth, snr=20.0, key=jax.random.PRNGKey(43))

print(f"Mock photometry (SNR=20): {BANDS}")
print(f"Fluxes [erg/s/cm²/Hz]: {np.asarray(mock.flux_obs)}")
print(f"Noise  [erg/s/cm²/Hz]: {np.asarray(mock.noise)}")

# %% [markdown]
# Run a quick NUTS fit so the diagnostics below operate on a real posterior.

# %%
fitter = Fitter(model, mock.flux_obs, mock.noise)
result = fitter.run("mcmc_nuts", n_warmup=300, n_samples=600, verbose=False)

# Free-parameter names (in the order ``Fisher`` and saliency helpers expect)
PARAM_NAMES = list(spec.free_params)
print(f"\nFit completed. Free parameters: {PARAM_NAMES}")

# %% [markdown]
# ## 1. Fisher Information Matrix

# %% [markdown]
# The **Fisher Information Matrix** (FIM) quantifies how much each data point constrains each
# parameter. Mathematically, at MAP estimate θ:
#
# $$F_{ij} = \sum_k \frac{1}{\sigma_k^2} \frac{\partial m_k}{\partial \theta_i} \frac{\partial m_k}{\partial \theta_j}$$
#
# where $m_k$ is the model at data point $k$ and $\sigma_k$ is the 1-sigma noise.
# The inverse of FIM is the **Laplace approximation** covariance—useful for quick
# parameter forecasting.

# %%
# Evaluation point: the MAP estimate. Run a cheap MAP fit so Fisher and
# saliency are anchored at the mode of the posterior.
fitter_map = Fitter(model, mock.flux_obs, mock.noise)
result_map = fitter_map.run("map", verbose=False)
# MAP results expose point estimates via .params (not .samples).
map_params = {name: result_map.params[name] for name in PARAM_NAMES}

# Compute the Fisher Information Matrix at MAP.
fim, param_names = compute_fisher_matrix(
    model,
    map_params,
    noise=mock.noise,
    data_type="photometry",
    param_names=PARAM_NAMES,
)

# Diagonal parameter errors (Laplace approximation)
fisher_errors = fisher_parameter_errors(fim)

# Compare to posterior widths from the NUTS run above.
posterior_errors = np.array([np.nanstd(np.asarray(result.samples[name])) for name in PARAM_NAMES])

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Fisher errors
ax = axes[0]
x = np.arange(len(param_names))
ax.bar(x - 0.2, fisher_errors, 0.4, label="Fisher (Laplace)", alpha=0.7, color=COLORS["model"])
ax.bar(x + 0.2, posterior_errors, 0.4, label="NUTS posterior", alpha=0.7, color=COLORS["data"])
ax.set_xticks(x)
ax.set_xticklabels(param_names, rotation=45, ha="right")
ax.set_ylabel(r"1$\sigma$ uncertainty")
ax.set_title("Parameter Errors: Fisher vs Posterior")
ax.legend()
ax.set_yscale("log")
ax.set_ylim(1e-3, 10)

# Correlation matrix (FIM)
ax = axes[1]
corr_fim = fisher_correlation_matrix(fim)
im = ax.imshow(np.abs(corr_fim), cmap="RdBu_r", vmin=0, vmax=1)
ax.set_xticks(np.arange(len(param_names)))
ax.set_yticks(np.arange(len(param_names)))
ax.set_xticklabels(param_names, rotation=45, ha="right")
ax.set_yticklabels(param_names)
ax.set_title("Fisher Correlation Matrix")
plt.colorbar(im, ax=ax, label=r"$|r_{ij}|$")

plt.tight_layout()
plt.show()

print(f"Fisher parameter errors:\n{dict(zip(param_names, fisher_errors))}")

# %% [markdown]
# **Interpretation:** The Fisher errors often **underestimate** true posterior widths,
# especially for non-linear parameters. Compare the two: where do they agree?
# Where does the Laplace approximation break down?

# %% [markdown]
# ## 2. Gradient SEDs (Saliency)

# %% [markdown]
# The **gradient SED** is $\partial \text{SED}(\lambda) / \partial \theta$ — it shows
# which wavelengths are most sensitive to which parameters. This directly answers:
# "Which filters should I use to constrain age vs. dust vs. metallicity?"

# %%
# Compute gradient SED for all free parameters (rest-frame wavelengths)
gradients, wave_rest = compute_all_gradient_seds(
    model,
    map_params,
    param_names=PARAM_NAMES,
)

# Plot gradient SEDs: which parameter drives which wavelength?
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.flatten()

for idx, (pname, grad_sed) in enumerate(gradients.items()):
    if idx >= len(axes):
        break
    ax = axes[idx]
    # Normalize gradient for visibility
    norm_grad = grad_sed / (np.max(np.abs(grad_sed)) + 1e-10)
    ax.plot(wave_rest, norm_grad, color=COLORS["model"], lw=1.5)
    ax.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax.fill_between(wave_rest, 0, norm_grad, alpha=0.3, color=COLORS["model"])
    ax.set_xscale("log")
    ax.set_xlim(wave_rest.min(), wave_rest.max())
    ax.set_xlabel(r"Rest-frame wavelength [$\AA$]")
    ax.set_ylabel(r"$\partial \text{SED} / \partial \theta$")
    ax.set_title(pname)
    ax.grid(True, alpha=0.2)

axes[-1].axis("off")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Photometry Sensitivity Matrix

# %% [markdown]
# Zoom in on photometry: $\partial f_{\text{band}} / \partial \theta$. This matrix
# shows which filter is most sensitive to which parameter—essential for designing
# survey filter sets or explaining why certain parameters are degenerate.

# %%
jac_phot, param_names_phot = compute_photometry_sensitivity(
    model,
    map_params,
    param_names=PARAM_NAMES,
)

# Normalize by parameter and flux magnitude for visibility
jac_norm = jac_phot.copy()
for i in range(jac_norm.shape[1]):
    max_val = np.max(np.abs(jac_norm[:, i])) + 1e-10
    jac_norm[:, i] /= max_val

fig, ax = plt.subplots(figsize=(10, 3.5))
im = ax.imshow(jac_norm.T, cmap="RdBu_r", aspect="auto")
ax.set_xlabel("Filter")
ax.set_ylabel("Parameter")
ax.set_xticks(np.arange(len(BANDS)))
ax.set_xticklabels(BANDS, rotation=45, ha="right")
ax.set_yticks(np.arange(len(param_names_phot)))
ax.set_yticklabels(param_names_phot)
ax.set_title("Photometric Sensitivity: which filter constrains which parameter?")
plt.colorbar(im, ax=ax, label=r"Normalized $\partial f / \partial \theta$")
plt.tight_layout()
plt.show()

print("Photometry Sensitivity (normalized):")
print(f"Filters: {BANDS}")
print(f"Parameters: {param_names_phot}")

# %% [markdown]
# **Reading the matrix:** Red = strong positive sensitivity, Blue = negative.
# E.g., if UV is bright-red for `tau_bc` (dust age), UV photometry strongly constrains dust properties.
# If a parameter column is pale (all near zero), that parameter is weakly constrained by these filters.

# %% [markdown]
# ## 4. MCMC Chain Diagnostics: ESS and Autocorrelation

# %% [markdown]
# How good is your MCMC chain? The **effective sample size (ESS)** accounts for
# autocorrelation: $\text{ESS} = N / \tau_{\text{int}}$ where $\tau_{\text{int}}$
# is the integrated autocorrelation time. A chain with $\tau=1$ is uncorrelated;
# $\tau=10$ means 10 samples per independent draw.

# %%
# Check convergence of the NUTS chain. Posterior.samples is a dict
# {param_name: ndarray of shape (n_samples,)} — pass it straight through.
chain_dict = {name: np.asarray(result.samples[name]) for name in PARAM_NAMES}

convergence_info = check_chain_length(
    chain_dict,
    exclude_prefixes=("psd_xi",),
    verbose=True,
)

# %%
# Extract ESS and plot trace diagnostics
ess_info = effective_sample_size(chain_dict, exclude_prefixes=("psd_xi",))

fig, axes = plt.subplots(len(param_names_phot), 2, figsize=(12, 10))

for idx, (pname, tau_info) in enumerate(ess_info.items()):
    chain = chain_dict[pname]

    # Trace plot
    ax = axes[idx, 0]
    ax.plot(chain, lw=0.5, alpha=0.7, color=COLORS["model"])
    ax.set_ylabel(pname)
    ax.set_title(f"{pname}  (τ={tau_info['tau_max']:.1f}, ESS={tau_info['ess']:.0f})")

    # Autocorrelation
    ax = axes[idx, 1]
    max_lag = min(200, len(chain) // 10)
    lags = np.arange(0, max_lag, 1)
    acf_vals = [np.corrcoef(chain[:-lag], chain[lag:])[0, 1] for lag in lags]
    ax.stem(lags, acf_vals, linefmt="C0-", markerfmt="C0o", basefmt=" ")
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(np.exp(-1), color="r", lw=1, linestyle="--", alpha=0.5, label=r"$e^{-1}$")
    ax.set_xlabel("Lag [samples]")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(f"{pname} ACF")
    ax.set_ylim(-0.2, 1.0)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Summary

# %% [markdown]
# **Why differentiate your SED?**
#
# 1. **Fisher diagnostics** reveal degeneracies and forecast parameter errors without
#    running the sampler—useful for survey design.
# 2. **Gradient SEDs** expose which wavelengths drive which parameter constraints—design
#    your filter set accordingly.
# 3. **Autocorrelation analysis** tells you how many independent samples your chain
#    really has—essential for convergence assessment.
#
# All three tools are **one-liners in JAX** because the forward model is fully differentiable.
# No finite differences, no approximations, no bespoke code.
#
# ## What you learned
#
# - Fisher Hessian forecasts parameter errors and correlations at MAP
# - Saliency (∂SED/∂θ) identifies wavelengths driving each parameter constraint
# - Filter sensitivity ranking guides optimal survey design
# - Autocorrelation and ESS assess MCMC chain quality without rerunning
#
# **Next:** [`06_inference_methods.py`](06_inference_methods.py) (inference algorithm choice) or
# [`07_degeneracies.py`](07_degeneracies.py) (posterior geometry deep dive).
