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
    LogUniform,
    ModelConfig,
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
# Build a smooth 7-parameter model and generate mock photometry.

# %%
# Load SSP library
ssp_data = load_ssp_data("bc03", "chabrier")

# Define the model
config = ModelConfig(
    sfh_model="psd_field",
    sps_model="bc03",
    dust_model="two_component",
    dust_emission=False,
    nebular_emission=True,
    agn_model=None,
)

model = SEDModel(ssp_data, config)

# Free parameters: 7-D smooth galaxy (no bursts, no AGN)
params_free = Parameters(
    psd_sigma=Uniform(0.1, 1.0),
    psd_tau_yr=LogUniform(1e6, 5e9),
    alpha=Uniform(0.5, 3.0),
    beta=Uniform(0.5, 3.0),
    tau_sfh=LogUniform(1e7, 1e10),
    sfr_norm=LogUniform(0.01, 100.0),
    log_z_abs=Uniform(-2.5, 0.5),
    tau_bc=Fixed(10.0),
    tau_diff=Fixed(100.0),
    dust_slope=Fixed(1.0),
)

# Generate mock data: 5 broadband photometry at z=0.05
truth = {
    "psd_sigma": 0.3,
    "psd_tau_yr": 2e8,
    "alpha": 1.5,
    "beta": 1.2,
    "tau_sfh": 1e9,
    "sfr_norm": 5.0,
    "log_z_abs": -0.7,
    "tau_bc": 10.0,
    "tau_diff": 100.0,
    "dust_slope": 1.0,
}

z = 0.05
bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
mock_data = generate_mock(
    model,
    truth,
    redshift=z,
    bands=bands,
    snr=20,
    seed=42,
)
print(f"Mock photometry (SNR=20): {bands}")
print(f"Fluxes [erg/s/cm²/Hz]: {mock_data.flux}")
print(f"Errors [erg/s/cm²/Hz]: {mock_data.error}")

# %% [markdown]
# Run a quick NUTS fit on the mock data (or skip to diagnostics if pre-computed).

# %%
obs = Observation(
    photometry=Photometry(
        flux=mock_data.flux,
        error=mock_data.error,
        bands=bands,
    ),
    redshift=z,
)

fitter = Fitter(model, obs, params_free)

# Run brief NUTS fit
result = fitter.run(
    method="hmc",
    n_samples=1000,
    n_warmup=500,
    seed=43,
    verbose=False,
)

print(f"\nFit completed. Chain shape: {result.chain.shape}")
print(f"Acceptance rate: {result.stats.get('acceptance_rate', 'N/A')}")

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
# Use MAP estimate as evaluation point
map_params = result.map_estimate

# Compute FIM
fim, param_names = compute_fisher_matrix(
    model,
    map_params,
    noise=mock_data.error,
    data_type="photometry",
    param_names=[
        "psd_sigma",
        "psd_tau_yr",
        "alpha",
        "beta",
        "tau_sfh",
        "sfr_norm",
        "log_z_abs",
    ],
)

# Diagonal parameter errors
fisher_errors = fisher_parameter_errors(fim)

# Compare to posterior widths
posterior_errors = np.nanstd(result.chain, axis=0)[:7]  # First 7 free params

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
# Compute gradient SED for all parameters (rest-frame)
gradients, wave_rest = compute_all_gradient_seds(
    model,
    map_params,
    param_names=[
        "psd_sigma",
        "psd_tau_yr",
        "alpha",
        "beta",
        "tau_sfh",
        "sfr_norm",
        "log_z_abs",
    ],
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
    param_names=[
        "psd_sigma",
        "psd_tau_yr",
        "alpha",
        "beta",
        "tau_sfh",
        "sfr_norm",
        "log_z_abs",
    ],
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
ax.set_xticks(np.arange(len(bands)))
ax.set_xticklabels(bands, rotation=45, ha="right")
ax.set_yticks(np.arange(len(param_names_phot)))
ax.set_yticklabels(param_names_phot)
ax.set_title("Photometric Sensitivity: which filter constrains which parameter?")
plt.colorbar(im, ax=ax, label=r"Normalized $\partial f / \partial \theta$")
plt.tight_layout()
plt.show()

print("Photometry Sensitivity (normalized):")
print(f"Filters: {bands}")
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
# Check convergence of the NUTS chain
chain_dict = {name: result.chain[:, i] for i, name in enumerate(param_names_phot)}

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
