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
# # Inference: Five Methods, One Loss Function
#
# All five inference methods in tengri operate on the **same
# standardised information Hamiltonian**:
#
# $$H(\boldsymbol{\xi} \mid \mathbf{d}) = \tfrac{1}{2}\chi^2 + \tfrac{1}{2}\boldsymbol{\xi}^\top\boldsymbol{\xi}$$
#
# They differ only in **how they explore the posterior landscape**.
# This tutorial demonstrates each method on the same mock galaxy,
# compares their posteriors, and provides practical guidance on when
# to use which.
#
# | Method | Type | Exact? | D range | Speed |
# |--------|------|--------|---------|-------|
# | **MAP** (Adam) | Optimisation | Point estimate | Any | ~1 s |
# | **Ray Tracing** | MCMC | Yes | ≲300 | ~min |
# | **NUTS** | MCMC (HMC) | Yes | ≲20 | ~min |
# | **geoVI** | Variational | Approximate | ≲10⁴ | ~min |
# | **MGVI** | Variational | Approximate | ≲10⁵ | ~s–min |
#
# **By the end you will know:**
# 1. How each method works (conceptually)
# 2. When each method is appropriate
# 3. How to diagnose convergence
# 4. Where the variational approximations break down

# %%
import time, os
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)
import sys; sys.path.insert(0, ".")
import sys; sys.path.insert(0, "..")
from _plot_style import (setup_style, COLORS, SAMPLER_STYLE, SDSS_WAVE_EFF,
                          plot_sfh, plot_sfh_comparison,
                          plot_corner_comparison, convergence_table)
setup_style()

FIG_DIR = "../notebook_figures"
os.makedirs(FIG_DIR, exist_ok=True)
def savefig(fig, name):
    fig.savefig(os.path.join(FIG_DIR, f"T03_{name}.png"),
                bbox_inches="tight", dpi=72)

ssp_data = load_ssp_data(
    "../../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# %% [markdown]
# ## Setup: Stochastic Mock Galaxy
#
# We use the full stochastic model ($D \approx 137$) so that Ray Tracing
# and geoVI are necessary — NUTS would struggle at this dimensionality.
# We also fit the same galaxy with the parametric model ($D = 7$) to
# demonstrate NUTS.

# %%
# ── Stochastic model ─────────────────────────────────────────────
spec_s = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0), sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0), sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Uniform(0.1, 4.0), psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0), dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7), redshift=Fixed(0.1),
    stochastic=True, n_grid=128,
)
model_s = SEDModel(spec_s, ssp_data, filters=filters)

key = jax.random.PRNGKey(42)
true_s = spec_s.sample(key)
true_s.update(sfh_alpha=1.0, sfh_beta=1.5, sfh_tau_peak_gyr=8.0,
              sfh_peak_sfr=40.0, psd_sigma=2.0, psd_tau_myr=20.0,
              met_logzsol=-0.3, dust_tau_bc=0.5, dust_tau_diff=0.3)
mock_s = model_s.mock(true_s, snr=20.0, key=key)
print(f"Stochastic model: D = {spec_s.n_free}")

# ── Parametric model (for NUTS comparison) ────────────────────────
spec_p = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0), sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0), sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Fixed(0.0), psd_tau_myr=Fixed(50.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0), dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7), redshift=Fixed(0.1), stochastic=False,
)
model_p = SEDModel(spec_p, ssp_data, filters=filters)
true_p = dict(sfh_alpha=1.0, sfh_beta=1.5, sfh_tau_peak_gyr=8.0,
              sfh_peak_sfr=40.0, met_logzsol=-0.3,
              dust_tau_bc=0.5, dust_tau_diff=0.3)
mock_p = model_p.mock(true_p, snr=20.0, key=key)
print(f"Parametric model: D = {spec_p.n_free}")

# %% [markdown]
# ## 1. MAP Optimisation
#
# **MAP** minimises the information Hamiltonian via Adam (Kingma & Ba
# 2015).  It provides a fast point estimate and initialises the samplers.
# MAP is not a posterior — it gives no uncertainty — but it is the
# starting point for everything else.

# %%
fitter_s = Fitter(model_s, mock_s.flux_obs, mock_s.noise, data_type="photometry")
fitter_p = Fitter(model_p, mock_p.flux_obs, mock_p.noise, data_type="photometry")

t0 = time.perf_counter()
map_s = fitter_s.run("map", n_steps=1000)
print(f"MAP (stochastic): {time.perf_counter()-t0:.1f} s")

t0 = time.perf_counter()
map_p = fitter_p.run("map", n_steps=500)
print(f"MAP (parametric): {time.perf_counter()-t0:.1f} s")

# %% [markdown]
# ## 2. Ray Tracing Sampler (Behroozi 2025)
#
# The **Ray Tracing Sampler** propagates proposals along straight-line
# trajectories that refract at iso-probability surfaces via Snell's law.
# Key advantages:
#
# - **~250× more gradient-noise tolerant** than HMC (Behroozi 2025)
# - **Crosses likelihood barriers** that trap HMC walkers
# - **Exact** — produces unbiased posterior samples
#
# It is the recommended method for the stochastic model ($D \sim 137$).

# %%
t0 = time.perf_counter()
rt_s = fitter_s.run("raytrace", init_from=map_s,
                    n_burnin=200, n_steps=2000,
                    step_size=0.005, n_leapfrog_steps=200)
t_rt = time.perf_counter() - t0
accept = rt_s.diagnostics.get("accept_rate_post_burnin", 0)
print(f"Ray Tracing: {t_rt:.1f} s  (acceptance: {accept:.0%})")

# %% [markdown]
# ## 3. NUTS (Hoffman & Gelman 2014)
#
# The **No-U-Turn Sampler** is the gold standard for low-dimensional
# problems.  It automatically tunes trajectory length via the U-turn
# criterion.  For $D \lesssim 20$, it gives exact posteriors with
# well-calibrated uncertainty.
#
# For $D \sim 137$, NUTS becomes impractical — the U-turn criterion
# breaks down and tuning is unreliable.

# %%
t0 = time.perf_counter()
nuts_p = fitter_p.run("nuts", init_from=map_p,
                      n_warmup=1000, n_samples=1000)
t_nuts = time.perf_counter() - t0
n_div = nuts_p.diagnostics.get("n_divergent", 0)
print(f"NUTS (parametric): {t_nuts:.1f} s  ({n_div} divergences)")

# %% [markdown]
# ## 4. geoVI (Frank et al. 2021)
#
# **Geometric Variational Inference** constructs a coordinate
# transformation that flattens the posterior metric, then fits a
# Gaussian in the flattened coordinates.  It is **approximate** but
# scales to $D > 10^4$.  Convergence typically requires ~25 KL
# iterations.
#
# geoVI cannot capture genuinely multimodal posteriors.  Where it
# disagrees with Ray Tracing, prefer the MCMC result.

# %%
t0 = time.perf_counter()
geovi_s = fitter_s.run("native_geovi", init_from=map_s,
                       n_iterations=25, n_samples=6)
t_geovi = time.perf_counter() - t0
print(f"geoVI (stochastic): {t_geovi:.1f} s")

# %% [markdown]
# ## 5. Convergence Diagnostics

# %%
convergence_table({
    "RT (D=137)": rt_s,
    "geoVI (D=137)": geovi_s,
    "NUTS (D=7)": nuts_p,
})

# %% [markdown]
# ## 6. Sampler Comparison
#
# ### SFH recovery: Ray Tracing vs. geoVI

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

ax = axes[0]
plot_sfh(model_s, rt_s, true_params=true_s, ax=ax, method="RT",
         label="Ray Tracing")
ax.set_title("Ray Tracing (exact MCMC)", fontsize=11)

ax = axes[1]
plot_sfh(model_s, geovi_s, true_params=true_s, ax=ax, method="geoVI",
         label="geoVI")
ax.set_title("geoVI (variational)", fontsize=11)

fig.suptitle("Stochastic SFH Recovery — Sampler Comparison", fontsize=12,
             y=1.02)
fig.tight_layout(); savefig(fig, "sfh_comparison_rt_geovi"); plt.show()

# %% [markdown]
# ### Corner plot: physical parameters

# %%
fig = plot_corner_comparison(
    [rt_s, geovi_s],
    ["Ray Tracing", "geoVI"],
    colors=[COLORS["rt"], COLORS["geovi"]],
    truths=true_s,
)
if fig is not None:
    savefig(fig, "corner_rt_vs_geovi")
plt.show()

# %% [markdown]
# ### Timing comparison

# %%
timings = {"MAP": map_s.wall_time_s if hasattr(map_s, 'wall_time_s') else 0,
           "Ray Tracing": t_rt, "geoVI": t_geovi, "NUTS\n(D=7 only)": t_nuts}
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.barh(list(timings.keys()), list(timings.values()),
               color=[COLORS["map"], COLORS["rt"], COLORS["geovi"], COLORS["nuts"]])
ax.set_xlabel("Wall-clock time [s]")
ax.set_title("Inference Time Comparison")
for bar, val in zip(bars, timings.values()):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{val:.1f} s", va="center", fontsize=9)
fig.tight_layout(); savefig(fig, "timing_comparison"); plt.show()

# %% [markdown]
# ## Practical Decision Tree
#
# ```
# Is D < 20?
# ├── Yes → NUTS (gold standard, exact)
# └── No → Is D < 300?
#     ├── Yes → Ray Tracing (exact, noise-tolerant)
#     └── No → Is D < 10⁴?
#         ├── Yes → geoVI (approximate, scalable)
#         └── No → MGVI (linear approximation, hierarchical regime)
# ```
#
# **Always start with MAP for initialisation.**
#
# **Next:** [T04 — Fitting](T04_fitting.ipynb) puts these methods to
# work on photometric and spectroscopic data.
