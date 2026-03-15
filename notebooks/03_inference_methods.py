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
# # Three Ways to Sample the Posterior
#
# Given photometric data $\mathbf{d}$ and a differentiable forward model
# $f(\boldsymbol{\xi})$, our goal is to characterize the posterior
# $P(\boldsymbol{\xi} \mid \mathbf{d})$.
#
# In `diffsed` every model — parametric or stochastic — is expressed in
# **standardized coordinates** $\boldsymbol{\xi} \sim \mathcal{N}(0, I)$.
# All priors are absorbed into bijective transforms, so the inference
# problem reduces to minimizing a single scalar: the **information
# Hamiltonian**.
#
# This notebook walks through five inference methods, from a quick MAP
# point estimate to full posterior sampling with Ray Tracing, geoVI, NUTS,
# and MGVI. We show when each method shines and where it breaks.
#
# > **Prerequisites:** See *Tutorial 01* for details on the IFT model and
# > *Tutorial 02* for the forward model pipeline.

# %% [markdown]
# ## The Information Hamiltonian
#
# In the standardized picture every free parameter lives in
# $\boldsymbol{\xi}$-space where the prior is
# $P(\boldsymbol{\xi}) = \mathcal{N}(0, I)$.  Taking
# $-\ln P(\boldsymbol{\xi} \mid \mathbf{d})$ and dropping constants gives
# the **information Hamiltonian**
#
# $$
# H(\boldsymbol{\xi})
#   = \frac{1}{2}\,\chi^2(\boldsymbol{\xi})
#   + \frac{1}{2}\,\boldsymbol{\xi}^\top \boldsymbol{\xi}\,,
# $$
#
# where
# $\chi^2 = \sum_i \bigl(d_i - f_i(\boldsymbol{\xi})\bigr)^2 / \sigma_i^2$.
#
# **Why is this elegant?**
#
# | Term | Meaning |
# |------|---------|
# | $\frac{1}{2}\chi^2$ | Data likelihood — how well the model fits the data |
# | $\frac{1}{2}\boldsymbol{\xi}^\top\boldsymbol{\xi}$ | Prior penalty — keeps parameters near zero (i.e.\ the prior center) |
#
# One loss function. Any prior (via transforms). Any sampler.
# All five methods below operate on exactly this $H(\boldsymbol{\xi})$.

# %%
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from diffsed import (
    Model, ParamSpec, Uniform, Gaussian, LogUniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)

# Reproducibility
key = jax.random.PRNGKey(42)

# Load stellar population data and SDSS filters
ssp_data = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# Parametric model — no stochastic SFH
spec = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0),
    sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0),
    sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Fixed(0.0),
    psd_tau_myr=Fixed(50.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    stochastic=False,
)
model = Model(spec, ssp_data, filters=filters)

# Ground truth
true_params = dict(
    sfh_alpha=1.5,
    sfh_beta=1.2,
    sfh_tau_peak_gyr=5.0,
    sfh_peak_sfr=10.0,
    met_logzsol=-0.3,
    dust_tau_bc=0.5,
    dust_tau_diff=0.3,
)

# Generate mock observation (SNR = 20 per band)
key, subkey = jax.random.split(key)
mock = model.mock(true_params, snr=20.0, key=subkey)
print(f"D = {spec.n_free} free parameters (parametric)")
print(f"Mock flux (5 bands): {np.array(mock.flux_obs)}")

# %% [markdown]
# ## Visualizing the Loss Landscape
#
# Let's build intuition by looking at a 2D slice of $H(\boldsymbol{\xi})$.
# We fix all parameters at their true values except two —
# `sfh_alpha` and `met_logzsol` — and sweep a grid.
#
# The contours show where the posterior probability mass lives.
# Note that the MAP (minimum of $H$) does **not** coincide with the
# posterior mean in skewed directions.

# %%
# 2D slice of the loss landscape
fitter_for_grid = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

alpha_vals = np.linspace(0.6, 2.8, 60)
met_vals = np.linspace(-1.8, 0.4, 60)
H_grid = np.zeros((len(met_vals), len(alpha_vals)))

for i, met in enumerate(met_vals):
    for j, alpha in enumerate(alpha_vals):
        params_ij = dict(true_params, sfh_alpha=float(alpha), met_logzsol=float(met))
        flux_pred = model.predict_photometry(params_ij)
        chi2 = jnp.sum(((mock.flux_obs - flux_pred) / mock.noise) ** 2)
        H_grid[i, j] = float(0.5 * chi2)

fig, ax = plt.subplots(figsize=(7, 5))
levels = np.linspace(H_grid.min(), H_grid.min() + 30, 15)
cs = ax.contourf(alpha_vals, met_vals, H_grid, levels=levels, cmap="viridis_r")
ax.contour(alpha_vals, met_vals, H_grid, levels=levels, colors="k", linewidths=0.3)
ax.plot(true_params["sfh_alpha"], true_params["met_logzsol"],
        "w*", ms=14, zorder=5, label="Truth")

# Mark approximate MAP
imin = np.unravel_index(H_grid.argmin(), H_grid.shape)
ax.plot(alpha_vals[imin[1]], met_vals[imin[0]],
        "rx", ms=12, mew=2, zorder=5, label="MAP (grid)")

ax.set_xlabel(r"$\alpha_{\rm SFH}$", fontsize=13)
ax.set_ylabel(r"$\log(Z/Z_\odot)$", fontsize=13)
ax.set_title(r"$H(\xi)$ — 2D slice (other params at truth)", fontsize=13)
plt.colorbar(cs, ax=ax, label=r"$H(\xi)$")
ax.legend(fontsize=11)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## MAP: Gradient Descent on $H$
#
# The simplest approach: find the **mode** (peak) of the posterior by
# minimizing $H(\boldsymbol{\xi})$ with the Adam optimizer.
#
# - **Fast:** converges in seconds for $D \lesssim 10$.
# - **No uncertainties:** a single point estimate.
# - **Use case:** initialization for all sampling methods.
#
# ```
# result_map = fitter.run("map", n_steps=1000)
# ```

# %%
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

key, subkey = jax.random.split(key)
result_map = fitter.run("map", n_steps=1000, key=subkey)

# Convergence curve
fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(np.array(result_map.loss_history), "k-", lw=0.8)
ax.set_xlabel("Step")
ax.set_ylabel(r"$H(\xi)$")
ax.set_title(f"MAP convergence — {result_map.wall_time_s:.1f}s")
ax.set_yscale("log")
plt.tight_layout()
plt.show()

print(f"Wall time: {result_map.wall_time_s:.1f}s")
print(f"Final loss: {float(result_map.loss_history[-1]):.3f}")

# %% [markdown]
# ### Why MAP Isn't Enough
#
# MAP finds the mode of the posterior, but the posterior can be
# **asymmetric** (skewed) or even **multimodal**.  The mode is not the
# mean, and it tells us nothing about the width or shape of the
# distribution.
#
# For science we need **credible intervals**, which requires sampling.
# The next three methods each draw samples from the full posterior, but
# they differ dramatically in how they do it and when they work.

# %%
# MAP SFH vs truth — point estimate only
fig, ax = plt.subplots(figsize=(7, 4))

sfh_true = model.predict_sfh(true_params)
ax.plot(sfh_true["t_gyr"], sfh_true["sfr_mean"], "k-", lw=2, label="Truth")

sfh_map = model.predict_sfh(result_map.params)
ax.plot(sfh_map["t_gyr"], sfh_map["sfr_mean"], "C3--", lw=2, label="MAP")

ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.set_title("MAP gives a point estimate — no uncertainty band")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Ray Tracing Sampler (Behroozi 2025)
#
# The Ray Tracing sampler treats the posterior as an optical medium with
# a spatially varying **refractive index**
#
# $$
# n(\mathbf{x}) = \mathcal{L}(\mathbf{x})^{1/(D-1)}\,,
# $$
#
# where $\mathcal{L}$ is the likelihood and $D$ the dimensionality.
# A "ray of light" travels through this medium, bending toward
# high-likelihood regions via **Snell's law**.
#
# **The key insight:** the ray moves at *constant speed*, so only the
# *direction* needs updating.  This makes the sampler roughly
# **250× more tolerant of noisy gradients** than HMC/NUTS, where both
# speed and direction change at every step.

# %% [markdown]
# ### Algorithm: Drift-Kick-Drift Leapfrog
#
# Each leapfrog step updates position $\mathbf{x}$ and direction
# $\hat{\mathbf{v}}$:
#
# 1. **Half drift:** $\mathbf{x} \leftarrow \mathbf{x} + \frac{\Delta s}{2}\,\hat{\mathbf{v}}$
# 2. **Kick (direction update):**
#
# $$
# \tan\!\bigl(\theta_f/2\bigr)
#   = \tan\!\bigl(\theta_i/2\bigr)
#     \cdot \exp\!\bigl(-\Delta s\,\|\nabla \ln n\|\bigr)
# $$
#
# 3. **Half drift:** $\mathbf{x} \leftarrow \mathbf{x} + \frac{\Delta s}{2}\,\hat{\mathbf{v}}$
#
# A **Metropolis correction** after each trajectory ensures the chain
# samples the exact target density despite discretization errors.
#
# This is analogous to HMC's leapfrog, but the constant-speed constraint
# makes it much more stable when $\nabla \ln \mathcal{L}$ is noisy.

# %% [markdown]
# ### Three Key Advantages of Ray Tracing
#
# | Advantage | Why it matters |
# |-----------|---------------|
# | **Constant speed** | Direction-only updates → ~250× noise tolerance vs HMC |
# | **Barrier crossing** | Rays refract through low-likelihood valleys rather than reflecting off them |
# | **Simple MH correction** | Metropolis accept/reject ensures exact sampling despite discretization |
#
# These properties make Ray Tracing the method of choice for
# **stochastic SFH models** where the gradient is computed via Monte
# Carlo sampling of the latent GP field.

# %%
key, subkey = jax.random.split(key)
result_rt = fitter.run(
    "raytrace",
    init_from=result_map,
    n_burnin=100,
    n_steps=300,
    key=subkey,
)

print(f"Wall time: {result_rt.wall_time_s:.1f}s")
print(f"Acceptance rate: {result_rt.diagnostics.get('accept_rate_post_burnin', 0):.2%}")
ess_rt = result_rt.effective_sample_size()
print("ESS:", {k: f"{v:.0f}" for k, v in ess_rt.items()})

# %%
# Trace plots for a few parameters
trace_params = ["sfh_alpha", "met_logzsol", "dust_tau_bc"]
fig, axes = plt.subplots(len(trace_params), 1, figsize=(8, 2.2 * len(trace_params)),
                         sharex=True)
for ax, name in zip(axes, trace_params):
    chain = np.array(result_rt.samples[name])
    ax.plot(chain, "C0-", lw=0.4, alpha=0.7)
    ax.axhline(true_params[name], color="k", ls="--", lw=1, label="Truth")
    ax.set_ylabel(name.replace("_", " "), fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
axes[-1].set_xlabel("Sample index")
fig.suptitle("Ray Tracing — chain trace", fontsize=13, y=1.01)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Geometric Variational Inference (Frank et al. 2021)
#
# geoVI works in Riemannian geometry.  It builds a local coordinate
# transformation $g(\boldsymbol{\xi})$ using the **Fisher information
# metric**
#
# $$
# M = J^\top J + I\,,
# $$
#
# where $J$ is the Jacobian of the forward model.  In the transformed
# coordinates the banana-shaped posterior becomes approximately
# **spherical** — i.e.\ $P \approx \mathcal{N}(0, I)$.
#
# The algorithm then fits a Gaussian in these curved coordinates, which
# captures non-Gaussian structure that a simple Laplace approximation
# would miss.

# %% [markdown]
# ### The geoVI Loop
#
# 1. **Initialize** at the MAP point $\boldsymbol{\xi}_0$.
# 2. **For each KL iteration** ($\sim$10–25 total):
#    - Draw $n$ samples from the current Gaussian approximation.
#    - Compute the KL divergence $\mathrm{KL}(q \| p)$ via those samples.
#    - Update the expansion point (mean and curvature) to reduce KL.
# 3. **Output:** draw $n_{\rm posterior}$ samples from the final approximation.
#
# **Key tuning:** use 4–12 samples per KL iteration (not 80 — see
# Edenhofer et al. 2024 for the evidence).

# %% [markdown]
# ### Strengths and Limitations
#
# | | geoVI |
# |------|-------|
# | **Strengths** | Scales to $D > 10^5$; cheap samples after optimization; captures non-Gaussianity via coordinate transform |
# | **Limitations** | Approximate (Gaussian in transformed space); can't capture multimodality; accuracy degrades for highly non-Gaussian posteriors |
#
# geoVI is an **equal-priority** primary method alongside Ray Tracing in
# `diffsed`.  It excels when $D$ is large and a good Gaussian
# approximation exists in some coordinate system.

# %%
key, subkey = jax.random.split(key)
result_geovi = fitter.run(
    "geovi",
    init_from=result_map,
    n_iterations=10,
    n_samples=6,
    key=subkey,
)
print(f"Wall time: {result_geovi.wall_time_s:.1f}s")
print(result_geovi.diagnostics_summary())

# %% [markdown]
# ## NUTS: Gold Standard (for Low $D$)
#
# The No-U-Turn Sampler (Hoffman & Gelman 2014) is a variant of
# Hamiltonian Monte Carlo that automatically tunes the trajectory
# length.  It is the **gold standard** for exact posterior sampling
# when $D \lesssim 15$.
#
# - **Exact:** asymptotically unbiased.
# - **Efficient:** exploits gradient information for fast mixing.
# - **Brittle:** struggles with stochastic gradients and $D \gtrsim 20$.
#
# We use it here to **validate** the other methods on this low-$D$
# parametric model.

# %%
key, subkey = jax.random.split(key)
result_nuts = fitter.run(
    "nuts",
    init_from=result_map,
    n_warmup=500,
    n_samples=500,
    key=subkey,
)

print(f"Wall time: {result_nuts.wall_time_s:.1f}s")
ess_nuts = result_nuts.effective_sample_size()
print("ESS:", {k: f"{v:.0f}" for k, v in ess_nuts.items()})
print(result_nuts.diagnostics_summary())

# %% [markdown]
# ## Head-to-Head Comparison
#
# We now compare all three samplers (RT, geoVI, NUTS) on the same mock.
# The MAP solution provides the initialization; the samplers provide
# uncertainty quantification.

# %%
# SFH recovery: 1×4 panel — MAP, RT, geoVI, NUTS
fig, axes = plt.subplots(1, 4, figsize=(18, 4), sharey=True)
titles = ["MAP", "Ray Tracing", "geoVI", "NUTS"]
results = [result_map, result_rt, result_geovi, result_nuts]
colors = ["0.5", "C0", "C1", "C2"]

sfh_truth = model.predict_sfh(true_params)

for ax, title, res, col in zip(axes, titles, results, colors):
    ax.plot(sfh_truth["t_gyr"], sfh_truth["sfr_mean"], "k-", lw=2, label="Truth")

    if res.samples is not None:
        # Draw posterior SFH envelope
        sfr_draws = []
        summary = res.summary()
        for k_idx in range(min(100, len(list(res.samples.values())[0]))):
            draw = {name: float(arr[k_idx]) for name, arr in res.samples.items()}
            sfh_draw = model.predict_sfh(draw)
            sfr_draws.append(sfh_draw["sfr_mean"])
        sfr_draws = np.array(sfr_draws)
        t_plot = sfh_truth["t_gyr"]
        lo, hi = np.percentile(sfr_draws, [16, 84], axis=0)
        ax.fill_between(t_plot, lo, hi, color=col, alpha=0.3, label="68% CI")
        median_sfr = np.median(sfr_draws, axis=0)
        ax.plot(t_plot, median_sfr, color=col, ls="--", lw=1.5, label="Median")
    else:
        sfh_map = model.predict_sfh(res.params)
        ax.plot(sfh_map["t_gyr"], sfh_map["sfr_mean"],
                color=col, ls="--", lw=2, label="MAP")

    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper right")

axes[0].set_ylabel(r"SFR [$M_\odot$/yr]")
fig.suptitle("SFH Recovery — Parametric Model", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Corner plot: RT (blue) + geoVI (orange) + NUTS (green)
fig = result_rt.plot_corner(truths=true_params, color="C0", label="Ray Tracing")
result_geovi.plot_corner(truths=true_params, color="C1", label="geoVI",
                         fig=fig)
result_nuts.plot_corner(truths=true_params, color="C2", label="NUTS",
                        fig=fig)
fig.suptitle("Posterior Comparison — Parametric Model", fontsize=14, y=1.02)
plt.show()

# %%
# Posterior predictive check: overlay model photometry on data
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
method_labels = ["Ray Tracing", "geoVI", "NUTS"]
method_results = [result_rt, result_geovi, result_nuts]
method_colors = ["C0", "C1", "C2"]
band_names = ["u", "g", "r", "i", "z"]

for ax, label, res, col in zip(axes, method_labels, method_results, method_colors):
    # Data with errorbars
    x = np.arange(len(band_names))
    ax.errorbar(x, np.array(mock.flux_obs), yerr=np.array(mock.noise),
                fmt="ko", ms=6, capsize=3, label="Data", zorder=5)

    # Posterior predictive draws
    n_draws = min(50, len(list(res.samples.values())[0]))
    for k_idx in range(n_draws):
        draw = {name: float(arr[k_idx]) for name, arr in res.samples.items()}
        flux_pred = model.predict_photometry(draw)
        ax.plot(x, np.array(flux_pred), "-", color=col, alpha=0.1, lw=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_xlabel("SDSS Band")
    ax.set_title(label, fontsize=12)
    if ax is axes[0]:
        ax.set_ylabel(r"Flux [erg/s/cm$^2$/Hz]")
    ax.legend(fontsize=9)

fig.suptitle("Posterior Predictive Checks", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Summary diagnostics table
print(f"{'Method':<15} {'Wall time':>10} {'ESS (min)':>10} {'ESS (med)':>10} {'Accept %':>10}")
print("-" * 58)

for name, res in [("MAP", result_map), ("Ray Tracing", result_rt),
                  ("geoVI", result_geovi), ("NUTS", result_nuts)]:
    wt = f"{res.wall_time_s:.1f}s"
    if res.samples is not None:
        ess = res.effective_sample_size()
        ess_vals = list(ess.values())
        ess_min = f"{min(ess_vals):.0f}"
        ess_med = f"{np.median(ess_vals):.0f}"
    else:
        ess_min, ess_med = "—", "—"
    accept = res.diagnostics.get("accept_rate_post_burnin",
             res.diagnostics.get("mean_accept_prob", None))
    accept_str = f"{accept:.1%}" if accept is not None else "—"
    print(f"{name:<15} {wt:>10} {ess_min:>10} {ess_med:>10} {accept_str:>10}")

# %% [markdown]
# ## MGVI: Linear Approximation
#
# Metric Gaussian Variational Inference (Knollmüller & Enßlin 2019) is
# the **linearized** cousin of geoVI.  Instead of using the full
# nonlinear coordinate transform, MGVI approximates the posterior with a
# Gaussian whose precision matrix is $J^\top N^{-1} J + I$ evaluated at
# the current expansion point.
#
# - **Faster per iteration** than geoVI (no nonlinear resample step).
# - **Best for $D > 10^5$** where geoVI's nonlinear samples become
#   expensive.
# - **Less accurate** for strongly non-Gaussian posteriors.
#
# ```python
# result_mgvi = fitter.run("mgvi", init_from=result_map, n_iterations=10, n_samples=6)
# ```

# %%
key, subkey = jax.random.split(key)
result_mgvi = fitter.run(
    "mgvi",
    init_from=result_map,
    n_iterations=10,
    n_samples=6,
    key=subkey,
)
print(f"MGVI wall time: {result_mgvi.wall_time_s:.1f}s")
print(result_mgvi.diagnostics_summary())

# %% [markdown]
# ## The Stochastic Model: Where It Gets Interesting
#
# So far we used a parametric SFH with $D = 7$ free parameters.  All
# three samplers handled it easily.  Now let's turn on the **stochastic
# SFH** — a Gaussian Process in log-SFR space governed by a power
# spectral density (PSD).
#
# The latent GP vector $\boldsymbol{\xi}_{\rm GP}$ has `n_grid = 128`
# components, bringing the total dimensionality to $D \sim 137$.  This
# is where:
#
# - **NUTS diverges** (too many leapfrog steps, noisy gradients).
# - **Ray Tracing still works** (constant-speed optics, noise-tolerant).
# - **geoVI still works** (Riemannian geometry, iterative KL).
#
# This is the **defining use case** for `diffsed`.

# %%
# Stochastic model setup
spec_stoch = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0),
    sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0),
    sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Uniform(0.1, 4.0),
    psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    stochastic=True,
    n_grid=128,
)
model_stoch = Model(spec_stoch, ssp_data, filters=filters)

# Ground truth for stochastic model (includes PSD params)
true_params_stoch = dict(
    sfh_alpha=1.5,
    sfh_beta=1.2,
    sfh_tau_peak_gyr=5.0,
    sfh_peak_sfr=10.0,
    psd_sigma=1.5,
    psd_tau_myr=80.0,
    met_logzsol=-0.3,
    dust_tau_bc=0.5,
    dust_tau_diff=0.3,
)

key, subkey = jax.random.split(key)
mock_stoch = model_stoch.mock(true_params_stoch, snr=20.0, key=subkey)
print(f"D = {spec_stoch.n_free} free parameters + 128 GP latents")

# %%
# Stochastic: MAP → Ray Tracing (step_size=0.01 for D>10)
fitter_stoch = Fitter(model_stoch, mock_stoch.flux_obs, mock_stoch.noise,
                      data_type="photometry")

key, subkey = jax.random.split(key)
result_stoch_map = fitter_stoch.run("map", n_steps=2000, key=subkey)
print(f"Stochastic MAP: {result_stoch_map.wall_time_s:.1f}s")

key, subkey = jax.random.split(key)
result_stoch_rt = fitter_stoch.run(
    "raytrace",
    init_from=result_stoch_map,
    n_burnin=100,
    n_steps=300,
    step_size=0.01,
    key=subkey,
)
print(f"Stochastic RT: {result_stoch_rt.wall_time_s:.1f}s")
accept = result_stoch_rt.diagnostics.get("accept_rate_post_burnin", 0)
print(f"Acceptance rate: {accept:.2%}")
ess_stoch_rt = result_stoch_rt.effective_sample_size()
phys_ess = {k: v for k, v in ess_stoch_rt.items() if not k.startswith("psd_xi")}
print("ESS (physical params):", {k: f"{v:.0f}" for k, v in phys_ess.items()})

# %%
# Stochastic: MAP → geoVI
key, subkey = jax.random.split(key)
result_stoch_geovi = fitter_stoch.run(
    "geovi",
    init_from=result_stoch_map,
    n_iterations=10,
    n_samples=6,
    key=subkey,
)
print(f"Stochastic geoVI: {result_stoch_geovi.wall_time_s:.1f}s")
print(result_stoch_geovi.diagnostics_summary())

# %%
# Stochastic: NUTS — expected to struggle at D ~ 137
key, subkey = jax.random.split(key)
try:
    result_stoch_nuts = fitter_stoch.run(
        "nuts",
        init_from=result_stoch_map,
        n_warmup=200,
        n_samples=100,
        key=subkey,
    )
    accept = result_stoch_nuts.diagnostics.get("mean_accept_prob", 0)
    ess_nuts_s = result_stoch_nuts.effective_sample_size()
    phys_ess_nuts = {k: v for k, v in ess_nuts_s.items()
                     if not k.startswith("psd_xi")}
    print(f"NUTS completed in {result_stoch_nuts.wall_time_s:.1f}s")
    print(f"Acceptance rate: {accept:.2%}")
    print("ESS (physical):", {k: f"{v:.0f}" for k, v in phys_ess_nuts.items()})
    print("⚠ NUTS ran but likely has low ESS and/or divergences at this D.")
except Exception as e:
    print(f"NUTS failed at D ~ 137 (as expected): {e}")
    print("This is why Ray Tracing and geoVI exist — they handle high D.")

# %%
# Stochastic SFH comparison: MAP, RT, geoVI
fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
titles_s = ["MAP (stochastic)", "Ray Tracing", "geoVI"]
results_s = [result_stoch_map, result_stoch_rt, result_stoch_geovi]
colors_s = ["0.5", "C0", "C1"]

sfh_truth_s = model_stoch.predict_sfh(true_params_stoch)

for ax, title, res, col in zip(axes, titles_s, results_s, colors_s):
    ax.plot(sfh_truth_s["t_gyr"], sfh_truth_s["sfr_mean"], "k-", lw=2, label="Truth")

    if res.samples is not None:
        sfr_draws = []
        n_draws = min(100, len(list(res.samples.values())[0]))
        for k_idx in range(n_draws):
            draw = {}
            for name, arr in res.samples.items():
                if name == "psd_xi":
                    draw[name] = arr[k_idx]  # array (n_grid,)
                else:
                    draw[name] = float(arr[k_idx])
            sfh_draw = model_stoch.predict_sfh(draw)
            sfr_draws.append(sfh_draw["sfr_mean"])
        sfr_draws = np.array(sfr_draws)
        t_plot = sfh_truth_s["t_gyr"]
        lo, hi = np.percentile(sfr_draws, [16, 84], axis=0)
        ax.fill_between(t_plot, lo, hi, color=col, alpha=0.3, label="68% CI")
        median_sfr = np.median(sfr_draws, axis=0)
        ax.plot(t_plot, median_sfr, color=col, ls="--", lw=1.5, label="Median")
    else:
        sfh_map_s = model_stoch.predict_sfh(res.params)
        ax.plot(sfh_map_s["t_gyr"], sfh_map_s["sfr_mean"],
                color=col, ls="--", lw=2, label="MAP")

    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper right")

axes[0].set_ylabel(r"SFR [$M_\odot$/yr]")
fig.suptitle("SFH Recovery — Stochastic Model ($D \\sim 137$)", fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# %%
# Stochastic model: corner plot of physical + PSD params
# (exclude the 128 GP latents for readability)
phys_params = ["sfh_alpha", "sfh_beta", "sfh_tau_peak_gyr", "sfh_peak_sfr",
               "psd_sigma", "psd_tau_myr", "met_logzsol",
               "dust_tau_bc", "dust_tau_diff"]

fig = result_stoch_rt.plot_corner(
    params=phys_params,
    truths=true_params_stoch,
    color="C0",
    label="Ray Tracing",
)
result_stoch_geovi.plot_corner(
    params=phys_params,
    truths=true_params_stoch,
    color="C1",
    label="geoVI",
    fig=fig,
)
fig.suptitle("Stochastic Model — Physical + PSD Parameters", fontsize=14, y=1.02)
plt.show()

# %% [markdown]
# ## When to Use Which Method
#
# | Criterion | MAP | Ray Tracing | geoVI | NUTS | MGVI |
# |-----------|-----|-------------|-------|------|------|
# | **Speed** | Seconds | Minutes | Minutes | Minutes–hours | Minutes |
# | **Uncertainties** | No | Yes (exact) | Yes (approx.) | Yes (exact) | Yes (approx.) |
# | **Parametric ($D \lesssim 10$)** | Init only | ✓ | ✓ | ✓ Gold standard | ✓ |
# | **Stochastic ($D \sim 100$)** | Init only | ✓ Primary | ✓ Primary | ✗ Diverges | ✓ |
# | **Very large $D$ ($>10^5$)** | Init only | Slow | ✓ | ✗ | ✓ Best |
# | **Noisy gradients** | OK | ✓ Best (~250× tolerance) | ✓ | ✗ | ✓ |
# | **Multimodal** | Finds one mode | ✓ Can cross barriers | ✗ | ✗ | ✗ |
# | **Non-Gaussian** | N/A | ✓ Exact | Partial (coord. transform) | ✓ Exact | ✗ Linear |
# | **Hierarchical (shared PSD)** | Init only | ✓ | ✓ Best | ✗ | ✓ |
#
# ### Step Size Guidance
#
# | Dimensionality | Recommended `step_size` |
# |----------------|------------------------|
# | $D \lesssim 10$ | `0.03` (default: `0.03\sqrt{D}$) |
# | $10 < D \lesssim 100$ | `0.01` |
# | $D > 100$ | `0.005` |
#
# ### Practical Recipe
#
# 1. **Always start with MAP** — fast initialization for all samplers.
# 2. **Parametric models:** run NUTS for gold-standard validation, then
#    RT or geoVI for production.
# 3. **Stochastic models:** run RT and geoVI, cross-check agreement.
# 4. **Hierarchical fits:** use geoVI or MGVI via `HierarchicalFitter`.
# 5. **Very large $D$:** MGVI first (fast), then geoVI for refinement.
#
# > **References:**
# > - Behroozi (2025) — Ray Tracing Sampler
# > - Frank et al. (2021) — geoVI
# > - Knollmüller & Enßlin (2019) — MGVI
# > - Hoffman & Gelman (2014) — NUTS
