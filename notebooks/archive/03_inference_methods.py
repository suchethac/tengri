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
# In `tengri` every model — parametric or stochastic — is expressed in
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

# %% [markdown]
# **By the end you will understand:**
# 1. The Information Hamiltonian — one loss function for any sampler
# 2. Why standardization helps every inference method
# 3. When to use MAP, Ray Tracing, NUTS, geoVI, or MGVI
# 4. How to diagnose convergence (ESS, acceptance rate, divergences)
# 5. The stochastic model challenge: why NUTS fails at $D \sim 137$
#
# **Quick reference:**
#
# | Method | Speed | Exactness | Best for |
# |--------|-------|-----------|----------|
# | MAP | ~seconds | Point estimate | Initialization, catalogs |
# | Ray Tracing | ~seconds | Exact MCMC | Default workhorse |
# | NUTS | ~10s–min | Exact HMC | Gold-standard validation |
# | geoVI | ~minute | Approximate VI | High-D stochastic models |
# | MGVI | ~minute | Approximate VI | Hierarchical problems |

# %% [markdown]
# ## How Standardization Works (and Why It Helps Every Sampler)
#
# Standardization is the reparametrization trick that makes all of this
# possible.  Each physical parameter $\theta_k$ (which may live in a
# bounded, log-spaced, or otherwise awkward space) is generated from a
# standard-normal latent $\xi_k$ via a differentiable bijection:
#
# $$
# \theta_k = h_k(\xi_k), \qquad \xi_k \sim \mathcal{N}(0, 1).
# $$
#
# The bijection $h_k$ is chosen so that pushing $\mathcal{N}(0,1)$
# through it reproduces the desired prior $P(\theta_k)$.  For example:
#
# | Prior | Transform $h(\xi)$ | Inverse (standardize) |
# |-------|--------------------|-----------------------|
# | `Uniform(a, b)` | $a + (b-a)\,\sigma(\xi)$ | $\mathrm{logit}((\theta-a)/(b-a))$ |
# | `Gaussian(μ, σ)` | $\mu + \sigma\,\xi$ | $(\theta - \mu)/\sigma$ |
# | `LogUniform(a, b)` | $\exp(\ln a + (\ln b - \ln a)\,\sigma(\xi))$ | $\mathrm{logit}((\ln\theta - \ln a)/(\ln b - \ln a))$ |
# | `LogNormal(μ, σ)` | $\exp(\mu + \sigma\,\xi)$ | $(\ln\theta - \mu)/\sigma$ |
#
# where $\sigma(\cdot)$ is the sigmoid function.
#
# ### The Jacobian cancellation
#
# By the change-of-variables formula:
#
# $$
# P(h_k(\xi_k))\,\left|\frac{dh_k}{d\xi_k}\right| = \varphi(\xi_k)
# $$
#
# where $\varphi$ is the standard normal density.  This is not an
# approximation — it's the *defining property* of the reparametrization.
# The prior density and the Jacobian exactly cancel, leaving only the
# $\frac{1}{2}\xi^2$ penalty from the standard normal.
#
# ### Why this helps each sampler
#
# **The geometric argument:** In physical space, the prior Hessian can
# span orders of magnitude — e.g., `Uniform(0.5, 3.0)` for SFH α vs
# `Uniform(1, 300)` for PSD τ.  After standardization, the prior
# Hessian is exactly **I** (the identity), so the posterior geometry
# is set by the *data*, not the *parametrization*.
#
# | Sampler | Why isotropy helps |
# |---------|-------------------|
# | **MAP (Adam)** | Same learning rate works for all params; gradient $\nabla H = \nabla\chi^2 + \xi$ has bounded regularizer |
# | **NUTS** | Mass matrix starts near identity → faster warmup, fewer divergences |
# | **Ray Tracing** | Single step size $\Delta s$ works uniformly; $\nabla \ln n$ has comparable magnitude in all directions |
# | **geoVI / MGVI** | *Required*: the metric $M = J^\top J + \mathbf{I}$ assumes prior = identity |
#
# Let's see this in action — we'll compare the condition number of the
# Hessian in physical vs standardized space.

# %%
# --- Demonstrate standardization transforms ---
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from tengri.parameters.priors import Uniform, Gaussian, LogUniform, LogNormal

# Show how each distribution maps xi ~ N(0,1) to physical space
xi_grid = jnp.linspace(-3, 3, 200)

dists = {
    "Uniform(0.5, 3.0)": Uniform(0.5, 3.0),
    "Gaussian(μ=-0.3, σ=0.5)": Gaussian(-0.3, 0.5),
    "LogUniform(1, 300)": LogUniform(1.0, 300.0),
    "LogNormal(μ=3, σ=1)": LogNormal(3.0, 1.0),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 3.5))
for ax, (label, dist) in zip(axes, dists.items()):
    theta = np.array([float(dist.unstandardize(xi)) for xi in xi_grid])
    ax.plot(xi_grid, theta, "C0-", lw=2)
    ax.set_xlabel(r"$\xi$ (standardized)")
    ax.set_ylabel(r"$\theta$ (physical)")
    ax.set_title(label, fontsize=10)
    ax.axhline(dist.bounds[0], color="k", ls=":", alpha=0.4)
    ax.axhline(dist.bounds[1], color="k", ls=":", alpha=0.4)
    ax.axvline(0, color="gray", ls="--", alpha=0.3)

fig.suptitle(
    r"Standardization: $\xi \sim \mathcal{N}(0,1) \to \theta$ via differentiable bijection",
    fontsize=13,
    y=1.02,
)
plt.tight_layout()
plt.savefig("notebook_figures/03_inference_methods_fig01.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# Each curve above is a differentiable bijection $h_k(\xi)$.
# At $\xi = 0$ (the prior center), each parameter sits at the midpoint
# of its physical range.  The sigmoid-based transforms (Uniform,
# LogUniform) have an S-shape that naturally respects bounds, while
# the linear/exponential transforms (Gaussian, LogNormal) are unbounded.
#
# **Key insight:** In $\xi$-space, every parameter has the same
# characteristic scale ($\sim 1$).  A step of $\Delta\xi = 0.1$ means
# the same thing for all parameters — a small perturbation near the
# prior center.  This is why a single step size works for all samplers.

# %% [markdown]
# ### Does standardization weaken the gradients?
#
# A natural concern: by wrapping $\theta = h(\xi)$ in sigmoids and
# exponentials, doesn't the chain rule distort or kill the gradient
# signal?  The answer is **no** — the Jacobian $dh/d\xi$ acts as a
# **natural preconditioner**, not a gradient killer.
#
# The loss gradient in $\xi$-space is:
#
# $$
# \frac{\partial H}{\partial \xi_k}
#   = \underbrace{\sum_j \frac{m_j - d_j}{\sigma_j^2}
#     \cdot \frac{\partial m_j}{\partial \theta_k}}_{\text{physical gradient}}
#   \cdot \underbrace{\frac{d\theta_k}{d\xi_k}}_{\text{Jacobian}}
#   + \underbrace{\xi_k}_{\text{prior}}
# $$
#
# The Jacobian factor is what makes the difference.  Let's visualize it
# for each transform, and then see what kind of *effective step size*
# the optimizer takes in physical space.

# %%
# --- Jacobian dθ/dξ for each transform ---
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri.parameters.priors import Uniform, Gaussian, LogUniform, LogNormal

xi_grid = jnp.linspace(-4, 4, 300)

transforms = {
    "Uniform(0.5, 3.0)": Uniform(0.5, 3.0),
    "Gaussian(μ=-0.3, σ=0.5)": Gaussian(-0.3, 0.5),
    "LogUniform(1, 300)": LogUniform(1.0, 300.0),
    "LogNormal(μ=3, σ=1)": LogNormal(3.0, 1.0),
}

fig, axes = plt.subplots(2, 4, figsize=(16, 6))

for col, (label, dist) in enumerate(transforms.items()):
    # Top row: θ(ξ) — the transform itself
    theta_vals = jnp.array([dist.unstandardize(xi) for xi in xi_grid])
    axes[0, col].plot(xi_grid, theta_vals, "C0-", lw=2)
    axes[0, col].set_title(label, fontsize=10)
    axes[0, col].axvline(0, color="gray", ls="--", alpha=0.3)
    if col == 0:
        axes[0, col].set_ylabel(r"$\theta = h(\xi)$")

    # Bottom row: dθ/dξ — the Jacobian (via JAX autodiff)
    jac_fn = jax.vmap(jax.grad(lambda x, d=dist: d.unstandardize(x)))
    jacobian_vals = jac_fn(xi_grid)
    axes[1, col].plot(xi_grid, jacobian_vals, "C3-", lw=2)
    axes[1, col].axvline(0, color="gray", ls="--", alpha=0.3)
    axes[1, col].set_xlabel(r"$\xi$")
    if col == 0:
        axes[1, col].set_ylabel(r"$d\theta/d\xi$ (Jacobian)")

    # Shade the N(0,1) bulk region
    for row in range(2):
        axes[row, col].axvspan(-2, 2, alpha=0.06, color="blue")

fig.suptitle(
    r"Transform $h(\xi)$ (top) and its Jacobian $dh/d\xi$ (bottom)"
    "\n"
    r"Blue shading = 95% of $\mathcal{N}(0,1)$ mass — where sampling actually happens",
    fontsize=12,
    y=1.04,
)
plt.tight_layout()
plt.savefig("notebook_figures/03_inference_methods_fig02.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# **What the Jacobian panels reveal:**
#
# | Transform | Jacobian $dh/d\xi$ | What it does for the optimizer |
# |-----------|-------------------|-------------------------------|
# | **Uniform** (sigmoid) | Bell-shaped: peaks at $\xi=0$, vanishes at $|\xi| \gg 3$ | **Auto-braking** near boundaries — the optimizer naturally slows down as $\theta$ approaches its bounds. No hard clipping needed. |
# | **Gaussian** (linear) | Constant = $\sigma$ | Pure rescaling — no distortion at all. Gradient in $\xi$ is exactly $\sigma \times$ the physical gradient. |
# | **LogUniform** (sigmoid in log) | Bell × exponential growth | **Log-scale stepping**: large steps when $\theta$ is large (τ = 200 Myr), small steps when $\theta$ is small (τ = 5 Myr). Exactly right for scale parameters. |
# | **LogNormal** (exponential) | $\sigma \cdot \theta$ — grows with $\theta$ | Same log-scale stepping. Effectively does gradient descent in $\ln\theta$, which is the natural geometry for positive quantities. |
#
# **The crucial point**: within the blue-shaded region ($|\xi| < 2$,
# where 95% of the prior mass lives), every Jacobian is well-behaved
# and non-zero.  The gradients are not weakened — they are
# *preconditioned* so that the optimizer takes appropriately sized steps
# in the natural geometry of each parameter.

# %%
# --- Effective step size comparison ---
# If an optimizer takes a step Δξ = 0.1 in standardized space,
# what's the corresponding Δθ in physical space?
# Δθ ≈ (dh/dξ) · Δξ

delta_xi = 0.1
xi_eval_points = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])

print(f"{'Transform':<28} {'ξ':>6}  {'θ':>10}  {'Δθ for Δξ=0.1':>14}  {'Δθ/θ (%)':>10}")
print("-" * 76)

for label, dist in transforms.items():
    for xi_val in xi_eval_points:
        theta = float(dist.unstandardize(xi_val))
        jac = float(jax.grad(lambda x, d=dist: d.unstandardize(x))(xi_val))
        delta_theta = jac * delta_xi
        rel_step = abs(delta_theta / theta) * 100 if abs(theta) > 1e-10 else float("inf")
        print(
            f"{label:<28} {float(xi_val):>6.1f}  {theta:>10.3f}  {delta_theta:>14.4f}  {rel_step:>9.1f}%"
        )
    print()

# %% [markdown]
# **Key observations from the table above:**
#
# - For **Uniform**, $\Delta\theta$ is largest near $\xi = 0$ (center
#   of the range) and shrinks toward the boundaries.  This is natural
#   preconditioning — the optimizer explores freely in the bulk and
#   decelerates near edges.
#
# - For **Gaussian**, $\Delta\theta$ is constant everywhere — no
#   distortion at all.  This is why Gaussian priors are "free" in
#   standardized space.
#
# - For **LogUniform** and **LogNormal**, the *relative* step size
#   $\Delta\theta / \theta$ stays roughly constant across orders of
#   magnitude.  A step of $\Delta\xi = 0.1$ always means "move ~X%
#   in the current value", regardless of whether $\theta = 5$ or
#   $\theta = 200$.  This is exactly the logarithmic scaling you'd
#   want for scale parameters like PSD timescale $\tau$.
#
# ### Comparison with physical-space optimization
#
# Without standardization, an Adam step $\epsilon = 0.001$ in physical
# space means:
# - $\Delta\tau = 0.001$ Myr whether $\tau = 5$ or $\tau = 200$
#   (pathological for a scale parameter)
# - $\Delta\alpha = 0.001$ regardless of whether $\alpha$ is near a
#   boundary (risks overshooting into infeasible regions)
#
# With standardization, the same step $\Delta\xi = 0.001$ automatically
# adapts to each parameter's natural geometry via the Jacobian.
# The transforms are **preconditioners built into the parametrization**.
#
# ### The only edge case
#
# The Jacobian *does* vanish at $|\xi| \gg 3$ for sigmoid-based
# transforms (Uniform, LogUniform).  Could this trap the optimizer?
# In practice, **no**, because:
#
# 1. The $\mathcal{N}(0,1)$ prior penalty $+\xi_k$ in $\nabla H$ grows
#    linearly with $|\xi|$, always pushing back toward the center.
# 2. $|\xi| > 3$ means $\theta$ is at >99.7% of its range — if the
#    MAP is truly there, the prior bounds are probably too tight.
# 3. The Hessian eigenvalue floor (the $+\mathbf{I}$ from the prior)
#    prevents the curvature from collapsing even when the likelihood
#    Jacobian is small.

# %% [markdown]
# The Hessian condition number analysis (comparing physical vs standardized
# space) is shown later in this notebook, after the model and MAP are set up.

# %% [markdown]
# The condition number tells us how anisotropic the posterior is.
# A condition number of 1 means perfectly isotropic (a sphere); larger
# values mean the posterior is elongated along some directions.
#
# In standardized space, the prior contributes $+\mathbf{I}$ to the
# Hessian, which sets a **floor of 1** on every eigenvalue.  This means
# even poorly constrained parameters (where the likelihood Hessian is
# near zero) still have unit curvature from the prior.  No eigenvalue
# can collapse to zero, which prevents the pathological "slab"
# geometries that cause NUTS divergences and HMC instability.
#
# This is the deep reason why standardization benefits all samplers —
# not just the variational ones that require it.

# %%
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

from tengri import (
    SEDModel,
    ParamSpec,
    Uniform,
    Gaussian,
    LogUniform,
    Fixed,
    Fitter,
    load_ssp_data,
    load_filter_set,
)

import sys

sys.path.insert(0, ".")
from _plot_style import convergence_check, convergence_table

# Reproducibility
key = jax.random.PRNGKey(42)

# Load stellar population data and SDSS filters
ssp_data = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# Parametric model — no stochastic SFH
spec = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
)
model = SEDModel(spec, ssp_data, filters=filters)

# Ground truth
true_params = dict(
    sfh_dpl_alpha=1.5,
    sfh_dpl_beta=1.2,
    sfh_dpl_tau_gyr=5.0,
    sfh_dpl_log_peak_sfr=1.0,
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
# `sfh_dpl_alpha` and `met_logzsol` — and sweep a grid.
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
        params_ij = dict(true_params, sfh_dpl_alpha=float(alpha), met_logzsol=float(met))
        flux_pred = model.predict_photometry(params_ij)
        chi2 = jnp.sum(((mock.flux_obs - flux_pred) / mock.noise) ** 2)
        H_grid[i, j] = float(0.5 * chi2)

fig, ax = plt.subplots(figsize=(7, 5))
levels = np.linspace(H_grid.min(), H_grid.min() + 30, 15)
cs = ax.contourf(alpha_vals, met_vals, H_grid, levels=levels, cmap="viridis_r")
ax.contour(alpha_vals, met_vals, H_grid, levels=levels, colors="k", linewidths=0.3)
ax.plot(true_params["sfh_dpl_alpha"], true_params["met_logzsol"], "w*", ms=14, zorder=5, label="Truth")

# Mark approximate MAP
imin = np.unravel_index(H_grid.argmin(), H_grid.shape)
ax.plot(alpha_vals[imin[1]], met_vals[imin[0]], "rx", ms=12, mew=2, zorder=5, label="MAP (grid)")

ax.set_xlabel(r"$\alpha_{\rm SFH}$", fontsize=13)
ax.set_ylabel(r"$\log(Z/Z_\odot)$", fontsize=13)
ax.set_title(r"$H(\xi)$ — 2D slice (other params at truth)", fontsize=13)
plt.colorbar(cs, ax=ax, label=r"$H(\xi)$")
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig("notebook_figures/03_inference_methods_fig03.png", dpi=72, bbox_inches="tight")
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
plt.savefig("notebook_figures/03_inference_methods_fig04.png", dpi=72, bbox_inches="tight")
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
plt.savefig("notebook_figures/03_inference_methods_fig05.png", dpi=72, bbox_inches="tight")
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
trace_params = ["sfh_dpl_alpha", "met_logzsol", "dust_tau_bc"]
fig, axes = plt.subplots(len(trace_params), 1, figsize=(8, 2.2 * len(trace_params)), sharex=True)
for ax, name in zip(axes, trace_params):
    chain = np.array(result_rt.samples[name])
    ax.plot(chain, "C0-", lw=0.4, alpha=0.7)
    ax.axhline(true_params[name], color="k", ls="--", lw=1, label="Truth")
    ax.set_ylabel(name.replace("_", " "), fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
axes[-1].set_xlabel("Sample index")
fig.suptitle("Ray Tracing — chain trace", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("notebook_figures/03_inference_methods_fig06.png", dpi=72, bbox_inches="tight")
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
# `tengri`.  It excels when $D$ is large and a good Gaussian
# approximation exists in some coordinate system.

# %%
key, subkey = jax.random.split(key)
result_geovi = fitter.run(
    "native_geovi",
    init_from=result_map,
    n_iterations=10,
    n_samples=6,
    n_seeds=5,
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
    n_warmup=1000,
    n_samples=1000,
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
        for k_idx in range(min(100, len(next(iter(res.samples.values()))))):
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
        ax.plot(sfh_map["t_gyr"], sfh_map["sfr_mean"], color=col, ls="--", lw=2, label="MAP")

    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper right")

axes[0].set_ylabel(r"SFR [$M_\odot$/yr]")
fig.suptitle("SFH Recovery — Parametric SEDModel", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("notebook_figures/03_inference_methods_fig07.png", dpi=72, bbox_inches="tight")
plt.show()

# %%
# Corner plot: RT (blue) + geoVI (orange) + NUTS (green)
fig = result_rt.plot_corner(truths=true_params, color="C0", label="Ray Tracing")
result_geovi.plot_corner(truths=true_params, color="C1", label="geoVI", fig=fig)
result_nuts.plot_corner(truths=true_params, color="C2", label="NUTS", fig=fig)
fig.suptitle("Posterior Comparison — Parametric SEDModel", fontsize=14, y=1.02)
plt.savefig("notebook_figures/03_inference_methods_fig08.png", dpi=72, bbox_inches="tight")
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
    ax.errorbar(
        x,
        np.array(mock.flux_obs),
        yerr=np.array(mock.noise),
        fmt="ko",
        ms=6,
        capsize=3,
        label="Data",
        zorder=5,
    )

    # Posterior predictive draws
    n_draws = min(50, len(next(iter(res.samples.values()))))
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
plt.savefig("notebook_figures/03_inference_methods_fig09.png", dpi=72, bbox_inches="tight")
plt.show()

# %%
# --- Convergence diagnostics (parametric model) ---
convergence_table(
    {
        "Ray Tracing": result_rt,
        "geoVI": result_geovi,
        "NUTS": result_nuts,
    }
)

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
# ## The Stochastic SEDModel: Where It Gets Interesting
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
# This is the **defining use case** for `tengri`.

# %%
# Stochastic model setup
spec_stoch = ParamSpec(
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
model_stoch = SEDModel(spec_stoch, ssp_data, filters=filters)

# Ground truth for stochastic model (includes PSD params)
true_params_stoch = dict(
    sfh_dpl_alpha=1.5,
    sfh_dpl_beta=1.2,
    sfh_dpl_tau_gyr=5.0,
    sfh_dpl_log_peak_sfr=1.0,
    sfh_field_psd_sigma=1.5,
    sfh_field_psd_tau_myr=80.0,
    met_logzsol=-0.3,
    dust_tau_bc=0.5,
    dust_tau_diff=0.3,
)

key, subkey = jax.random.split(key)
mock_stoch = model_stoch.mock(true_params_stoch, snr=20.0, key=subkey)
print(f"D = {spec_stoch.n_free} free parameters + 128 GP latents")

# %%
# Stochastic: MAP → Ray Tracing (step_size=0.01 for D>10)
fitter_stoch = Fitter(model_stoch, mock_stoch.flux_obs, mock_stoch.noise, data_type="photometry")

key, subkey = jax.random.split(key)
result_stoch_map = fitter_stoch.run("map", n_steps=2000, key=subkey)
print(f"Stochastic MAP: {result_stoch_map.wall_time_s:.1f}s")

key, subkey = jax.random.split(key)
result_stoch_rt = fitter_stoch.run(
    "raytrace",
    init_from=result_stoch_map,
    n_burnin=200,
    n_steps=2000,
    step_size=0.05,
    n_leapfrog_steps=50,
    key=subkey,
)
print(f"Stochastic RT: {result_stoch_rt.wall_time_s:.1f}s")
accept = result_stoch_rt.diagnostics.get("accept_rate_post_burnin", 0)
print(f"Acceptance rate: {accept:.2%}")
ess_stoch_rt = result_stoch_rt.effective_sample_size()
phys_ess = {k: v for k, v in ess_stoch_rt.items() if not k.startswith("sfh_field_xi")}
print("ESS (physical params):", {k: f"{v:.0f}" for k, v in phys_ess.items()})

# %%
# Stochastic: MAP → geoVI
key, subkey = jax.random.split(key)
result_stoch_geovi = fitter_stoch.run(
    "native_geovi",
    init_from=result_stoch_map,
    n_iterations=10,
    n_samples=6,
    n_seeds=5,
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
    phys_ess_nuts = {k: v for k, v in ess_nuts_s.items() if not k.startswith("sfh_field_xi")}
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
        n_draws = min(100, len(next(iter(res.samples.values()))))
        for k_idx in range(n_draws):
            draw = {}
            for name, arr in res.samples.items():
                if name == "sfh_field_xi":
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
        ax.plot(sfh_map_s["t_gyr"], sfh_map_s["sfr_mean"], color=col, ls="--", lw=2, label="MAP")

    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper right")

axes[0].set_ylabel(r"SFR [$M_\odot$/yr]")
fig.suptitle("SFH Recovery — Stochastic SEDModel ($D \\sim 137$)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("notebook_figures/03_inference_methods_fig10.png", dpi=72, bbox_inches="tight")
plt.show()

# %%
# Stochastic model: corner plot of physical + PSD params
# (exclude the 128 GP latents for readability)
phys_params = [
    "sfh_dpl_alpha",
    "sfh_dpl_beta",
    "sfh_dpl_tau_gyr",
    "sfh_dpl_log_peak_sfr",
    "sfh_field_psd_sigma",
    "sfh_field_psd_tau_myr",
    "met_logzsol",
    "dust_tau_bc",
    "dust_tau_diff",
]

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
fig.suptitle("Stochastic SEDModel — Physical + PSD Parameters", fontsize=14, y=1.02)
plt.savefig("notebook_figures/03_inference_methods_fig11.png", dpi=72, bbox_inches="tight")
plt.show()

# %%
# --- Convergence diagnostics (stochastic model) ---
convergence_table(
    {
        "RT (stoch.)": result_stoch_rt,
        "geoVI (stoch.)": result_stoch_geovi,
    }
)

# %% [markdown]
# ## Convergence Diagnostics
#
# Checking that posteriors are converged is essential before trusting
# any inference result.  The standard diagnostics
# (Vehtari et al. 2021; Stan/ArviZ/BlackJAX):
#
# | Diagnostic | Threshold | Applies to |
# |-----------|-----------|------------|
# | **Effective Sample Size** (ESS) | $> 400$ total, $> 100$ per param | RT, NUTS |
# | **Divergent transitions** | 0 ideal; $> 5\%$ = serious | NUTS only |
# | **Acceptance rate** | NUTS $\sim 80\%$; RT $30$–$70\%$ | RT, NUTS |
# | **R-hat** (split $\hat{R}$) | $< 1.01$ | Multi-chain (not used here) |
#
# For **variational inference** (geoVI, MGVI), convergence is assessed by
# monitoring the KL divergence across iterations — it should decrease
# monotonically.  There is no ESS or R-hat equivalent; instead, compare
# geoVI posteriors against an MCMC reference (RT) to assess accuracy.
#
# **Common pitfalls:**
#
# - *NUTS divergences* often signal difficult posterior geometry (e.g.,
#   dust parameters creating funnels/ridges).  Increase `n_warmup` or
#   `target_accept_rate` toward 0.95.
# - *RT high acceptance* ($> 90\%$) means the chain is barely moving.
#   Increase `n_leapfrog_steps` or adjust `step_size`.
# - *Low ESS* on specific parameters (e.g., dust, metallicity) reflects
#   known age-dust-metallicity degeneracies, not sampler failure per se.

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

# %% [markdown]
# ## What You've Learned
#
# 1. The standardized loss $H(\xi) = \frac{1}{2}\chi^2 + \frac{1}{2}\xi^T\xi$ works for any sampler
# 2. MAP is fast but gives no uncertainties — use it for initialization
# 3. Ray Tracing and geoVI are the primary methods for stochastic models ($D \sim 137$)
# 4. NUTS is the gold standard for parametric models ($D < 15$)
# 5. Always check convergence diagnostics before trusting posteriors
#
# **Next:** [Tutorial 04 — Recovery Tests](04_recovery_tests.ipynb) validates
# these methods on mock data across burstiness regimes.
