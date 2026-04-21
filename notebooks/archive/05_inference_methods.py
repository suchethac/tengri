# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,../demonstrations//ipynb
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
# # Inference Methods: A Comprehensive Guide
#
# tengri ships ten inference methods, all optimizing or sampling from the same
# information Hamiltonian. This notebook profiles every method, compares their
# posteriors on identical mock data, and provides a decision tree for choosing
# the right one.
#
# **Methods covered:** MAP, Laplace, Pathfinder, native_geovi, NUTS,
# Ray Tracing, Elliptical Slice Sampling, geoVI-NUTS, Pathfinder-NUTS.

# %%
import os
import sys
import time
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
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
)

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))

# chdir to project root for data/ access
if os.path.exists("data"):
    pass
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

from _plot_style import (
    COLORS,
    add_sfh_inset,
    convergence_table,
    plot_corner_comparison,
    plot_sfh,
    setup_style,
)

setup_style()

FIGDIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else ".",
    "..",
    "figures",
    "demonstrations",
)
os.makedirs(FIGDIR, exist_ok=True)


def savefig(name):
    plt.savefig(os.path.join(FIGDIR, name), dpi=150, bbox_inches="tight")


# %% [markdown]
# ## 1. The Information Hamiltonian
#
# Every inference method in tengri minimizes or samples from the same loss:
#
# $$H(\xi \mid d) = \frac{1}{2}\chi^2(\xi) + \frac{1}{2}\xi^\top\xi$$
#
# where $\xi$ are standardized coordinates (all priors become $\mathcal{N}(0,1)$
# via the reparametrization transform). The first term is data fidelity; the
# second is the prior. Gradients $\nabla_\xi H$ are available via JAX autodiff,
# enabling all gradient-based methods.
#
# The methods differ only in **how** they explore $H$: optimization (MAP),
# Gaussian approximation (Laplace), variational inference (geoVI, Pathfinder),
# or exact MCMC (NUTS, Ray Tracing, Elliptical Slice Sampling).

# %% [markdown]
# ## 2. Setup: D=7 and D=137 Models
#
# We compare all methods on two models built from the same mock galaxy:
#
# - **Parametric (D=7):** Truncated-skew-normal SFH + dust + metallicity.
#   All methods work here, making it ideal for cross-validation.
# - **Stochastic (D=137):** Adds a GP latent field (128 $\xi$ dimensions) for
#   bursty SFH. Only scalable methods (geoVI, RT, ESS) work at this D.

# %%
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)

# --- Parametric model (D = 7) ---
spec_param = ParamSpec(
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
    mean_sfh_type="tsnorm",
)
model_param = SEDModel(spec_param, ssp_data, observation=obs)

key = jax.random.PRNGKey(42)
true_param = spec_param.sample(key)
true_param = {**true_param}
true_param["sfh_tsnorm_log_peak_sfr"] = jnp.array(1.2)
true_param["sfh_tsnorm_peak_lbt_gyr"] = jnp.array(3.0)
true_param["sfh_tsnorm_width_gyr"] = jnp.array(3.0)
true_param["sfh_tsnorm_skew"] = jnp.array(0.3)
true_param["sfh_tsnorm_trunc"] = jnp.array(2.0)

mock_param = model_param.mock(true_param, snr=20.0, key=key)

# --- Stochastic model (D = 137) ---
spec_stoch = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Uniform(-1.0, 2.5),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),
    sfh_field_psd_sigma=Uniform(0.1, 4.0),
    sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type=["tsnorm", "field"],
    n_grid=128,
)
model_stoch = SEDModel(spec_stoch, ssp_data, observation=obs)

true_stoch = spec_stoch.sample(jax.random.PRNGKey(123))
true_stoch = {**true_stoch}
true_stoch["sfh_tsnorm_log_peak_sfr"] = jnp.array(1.2)
true_stoch["sfh_tsnorm_peak_lbt_gyr"] = jnp.array(3.0)
true_stoch["sfh_tsnorm_width_gyr"] = jnp.array(3.0)
true_stoch["sfh_tsnorm_skew"] = jnp.array(0.3)
true_stoch["sfh_tsnorm_trunc"] = jnp.array(2.0)
true_stoch["sfh_field_psd_sigma"] = jnp.array(2.0)
true_stoch["sfh_field_psd_tau_myr"] = jnp.array(20.0)

mock_stoch = model_stoch.mock(true_stoch, snr=20.0, key=jax.random.PRNGKey(456))

print(f"Parametric model: D = {spec_param.n_free}")
print(f"Stochastic model: D = {spec_stoch.n_free}")

# %%
# Create fitters (no data_type= needed with Observation API)
fitter_param = Fitter(model_param, mock_param.flux_obs, mock_param.noise)
fitter_stoch = Fitter(model_stoch, mock_stoch.flux_obs, mock_stoch.noise)

timings = {}
results_d7 = {}
results_d137 = {}

# %% [markdown]
# ## 3. MAP Initialization
#
# Maximum a posteriori -- gradient descent on $H(\xi)$. Fast point estimate
# with no uncertainty quantification. Primarily used as initialization for
# all sampling methods.

# %%
t0 = time.perf_counter()
result_map_p = fitter_param.run("map", n_steps=500, verbose=False)
timings["MAP (D=7)"] = time.perf_counter() - t0
results_d7["MAP"] = result_map_p

t0 = time.perf_counter()
result_map_s = fitter_stoch.run("map", n_steps=1000, verbose=False)
timings["MAP (D=137)"] = time.perf_counter() - t0
results_d137["MAP"] = result_map_s

print(f"MAP (D=7):   {timings['MAP (D=7)']:.1f}s")
print(f"MAP (D=137): {timings['MAP (D=137)']:.1f}s")

# %%
# --- fig01: MAP loss history ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))

if result_map_p.loss_history is not None:
    ax1.plot(np.array(result_map_p.loss_history), color=COLORS["map"], lw=1)
    ax1.set_xlabel("Step")
    ax1.set_ylabel(r"$H(\xi)$")
    ax1.set_title("MAP Loss: D = 7")
    ax1.set_yscale("log")

if result_map_s.loss_history is not None:
    ax2.plot(np.array(result_map_s.loss_history), color=COLORS["map"], lw=1)
    ax2.set_xlabel("Step")
    ax2.set_ylabel(r"$H(\xi)$")
    ax2.set_title("MAP Loss: D = 137")
    ax2.set_yscale("log")

fig.tight_layout()
savefig("05_fig01_map_loss.png")
plt.show()

# %% [markdown]
# ## 4. Laplace Approximation
#
# Compute the Hessian $\mathbf{H}$ of the loss at the MAP, invert to get a
# Gaussian posterior $\mathcal{N}(\hat\theta, \mathbf{H}^{-1})$. The Hessian
# eigenvalue spectrum reveals which parameters are well-constrained (large
# eigenvalues) vs degenerate (small eigenvalues -- the age-dust-metallicity
# banana).
#
# Also provides a Laplace evidence estimate:
# $$\log Z = -H(\hat\theta) + \frac{D}{2}\log(2\pi) - \frac{1}{2}\log\det(\mathbf{H})$$

# %%
t0 = time.perf_counter()
result_laplace_p = fitter_param.run(
    "laplace",
    init_from=result_map_p,
    n_samples=5000,
    verbose=False,
)
timings["Laplace (D=7)"] = time.perf_counter() - t0
results_d7["Laplace"] = result_laplace_p

print(f"Laplace (D=7): {timings['Laplace (D=7)']:.1f}s")
print(f"  Log-evidence: {result_laplace_p.diagnostics.get('log_evidence', 'N/A')}")
print(f"  Condition number: {result_laplace_p.diagnostics.get('condition_number', 'N/A'):.1e}")

# %%
# --- fig02: Hessian eigenvalue spectrum ---
fig, ax = plt.subplots(figsize=(6, 3.5))
if "eigenvalues" in result_laplace_p.diagnostics:
    eigs = np.array(result_laplace_p.diagnostics["eigenvalues"])
    ax.bar(range(len(eigs)), np.sort(eigs)[::-1], color=COLORS["laplace"], alpha=0.8)
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("Hessian eigenvalue")
    ax.set_title("Laplace: Hessian Eigenvalue Spectrum (D = 7)")
    ax.set_yscale("log")
else:
    ax.text(0.5, 0.5, "Eigenvalues not available", transform=ax.transAxes, ha="center")
fig.tight_layout()
savefig("05_fig02_laplace_eigenvalues.png")
plt.show()

# %% [markdown]
# ## 5. Pathfinder
#
# Pathfinder (Zhang et al. 2022) traces the L-BFGS optimization trajectory
# and fits a sequence of Gaussian approximations along the path. It picks
# the best one (by ELBO) and draws samples from it. Roughly 10x faster than
# NUTS for approximate posteriors, and makes an excellent warm-start
# initializer for NUTS chains.

# %%
t0 = time.perf_counter()
result_pathfinder_p = fitter_param.run(
    "pathfinder",
    n_samples=5000,
    maxiter=30,
    verbose=False,
)
timings["Pathfinder (D=7)"] = time.perf_counter() - t0
results_d7["Pathfinder"] = result_pathfinder_p

print(f"Pathfinder (D=7): {timings['Pathfinder (D=7)']:.1f}s")

# %% [markdown]
# ## 6. native_geovi: The Default
#
# Geometric Variational Inference (Frank et al. 2021) constructs a nonlinear
# coordinate transform that flattens curved posterior geometry -- imagine
# straightening the age-dust banana. The "native" implementation is fully
# JIT-compiled in JAX, eliminating 99.8% of Python overhead.
#
# ### Why It's Fast: The JIT Engine
#
# The entire optimization loop runs inside `jax.lax.while_loop` with
# `jax.lax.scan` for sample drawing and Newton-CG for KL minimization.
# Zero Python callbacks. The forward model, gradient, and metric-vector
# products compile into a single XLA program.

# %%
# Compile/runtime separation: compile once, run per galaxy
t0_compile = time.perf_counter()
fitter_param.compile(verbose=False)
t_compile_d7 = time.perf_counter() - t0_compile

t0 = time.perf_counter()
result_geovi_p = fitter_param.run(
    "native_geovi",
    n_iterations=15,
    n_samples=6,
    n_seeds=5,
    n_posterior_samples=5000,
    verbose=False,
)
timings["native_geovi (D=7)"] = time.perf_counter() - t0
results_d7["native_geovi"] = result_geovi_p

print(f"XLA compile (D=7): {t_compile_d7:.1f}s (one-time, cached on disk)")
print(f"native_geovi (D=7): {timings['native_geovi (D=7)']:.1f}s <- runtime per galaxy")

# %%
# native_geovi on D=137
t0_compile = time.perf_counter()
fitter_stoch.compile(verbose=False)
t_compile_d137 = time.perf_counter() - t0_compile

t0 = time.perf_counter()
result_geovi_s = fitter_stoch.run(
    "native_geovi",
    n_iterations=20,
    n_samples=6,
    n_seeds=5,
    n_posterior_samples=5000,
    verbose=False,
)
timings["native_geovi (D=137)"] = time.perf_counter() - t0
results_d137["native_geovi"] = result_geovi_s

print(f"XLA compile (D=137): {t_compile_d137:.1f}s (one-time, cached on disk)")
print(f"native_geovi (D=137): {timings['native_geovi (D=137)']:.1f}s <- runtime per galaxy")

# %%
# --- fig03: KL convergence ---
fig, ax = plt.subplots(figsize=(6, 3.5))
plotted_kl = False
if hasattr(result_geovi_s, "diagnostics") and "kl_history" in result_geovi_s.diagnostics:
    kl = np.array(result_geovi_s.diagnostics["kl_history"])
    ax.plot(kl, color=COLORS["geovi"], lw=1.5, label="D = 137")
    plotted_kl = True
if hasattr(result_geovi_p, "diagnostics") and "kl_history" in result_geovi_p.diagnostics:
    kl_p = np.array(result_geovi_p.diagnostics["kl_history"])
    ax.plot(kl_p, color=COLORS["geovi"], lw=1.5, ls="--", label="D = 7")
    plotted_kl = True

if plotted_kl:
    ax.set_xlabel("Iteration")
    ax.set_ylabel("KL divergence")
    ax.set_title("native_geovi Convergence")
    ax.set_yscale("log")
    ax.legend()
else:
    ax.text(0.5, 0.5, "(KL history not available)", transform=ax.transAxes, ha="center")

fig.tight_layout()
savefig("05_fig03_kl_convergence.png")
plt.show()

# %%
# --- fig04: SFH recovery D=137 with 200 Myr inset ---
fig, ax = plt.subplots(figsize=(8, 4))
plot_sfh(
    model_stoch,
    result_geovi_s,
    true_params=true_stoch,
    ax=ax,
    color=COLORS["geovi"],
    label="native_geovi",
    method="geoVI",
    show_mean_sfh=True,
)
ax.set_title(f"native_geovi: D = 137 in {timings['native_geovi (D=137)']:.1f}s")

# 200 Myr inset
sfh_true_s = model_stoch.predict_sfh(true_stoch)
add_sfh_inset(
    ax,
    np.array(sfh_true_s["t_gyr"]),
    np.array(sfh_true_s["sfr_full"]),
    color=COLORS["truth"],
    lw=1,
)
fig.tight_layout()
savefig("05_fig04_geovi_sfh_137.png")
plt.show()

# %% [markdown]
# ## 7. Ray Tracing: Exact MCMC at Any D
#
# The Ray Tracing Sampler (Behroozi 2025) is a physics-inspired MCMC method.
# It defines a refractive index $n(\mathbf{x}) = \mathcal{L}(\mathbf{x})^{1/(D-1)}$
# and propagates "light rays" through parameter space. Snell's law bends
# trajectories toward high-likelihood regions.
#
# Key advantages:
# - ~250x gradient-noise tolerant vs HMC/NUTS
# - Works at D = 137 where NUTS fails
# - **Exact** posteriors (unbiased, unlike geoVI)
# - Constant-speed propagation: no momentum resampling needed

# %%
# RT on D=7 (step_size=0.05 for parametric models)
t0 = time.perf_counter()
result_rt_p = fitter_param.run(
    "raytrace",
    init_from=result_map_p,
    n_burnin=100,
    n_steps=1000,
    step_size=0.05,
    verbose=False,
)
timings["RT (D=7)"] = time.perf_counter() - t0
results_d7["RT"] = result_rt_p

# RT on D=137 (step_size=0.05, more leapfrog steps)
t0 = time.perf_counter()
result_rt_s = fitter_stoch.run(
    "raytrace",
    init_from=result_map_s,
    n_burnin=200,
    n_steps=2000,
    step_size=0.05,
    n_leapfrog_steps=50,
    verbose=False,
)
timings["RT (D=137)"] = time.perf_counter() - t0
results_d137["RT"] = result_rt_s

acc_p = result_rt_p.diagnostics.get("acceptance_rate", float("nan"))
acc_s = result_rt_s.diagnostics.get("acceptance_rate", float("nan"))
print(f"RT (D=7):   {timings['RT (D=7)']:.1f}s, acceptance = {acc_p:.1%}")
print(f"RT (D=137): {timings['RT (D=137)']:.1f}s, acceptance = {acc_s:.1%}")

# %%
# --- fig05: native_geovi vs RT on D=137 ---
fig, (ax_g, ax_r) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

plot_sfh(
    model_stoch,
    result_geovi_s,
    true_params=true_stoch,
    ax=ax_g,
    color=COLORS["geovi"],
    label="native_geovi",
    method="geoVI",
    show_mean_sfh=True,
)
ax_g.set_title(f"native_geovi ({timings['native_geovi (D=137)']:.1f}s)")

plot_sfh(
    model_stoch,
    result_rt_s,
    true_params=true_stoch,
    ax=ax_r,
    color=COLORS["rt"],
    label="Ray Tracing",
    method="RT",
    show_mean_sfh=True,
)
ax_r.set_title(f"Ray Tracing ({timings['RT (D=137)']:.1f}s)")

# 200 Myr insets
for ax_panel in [ax_g, ax_r]:
    add_sfh_inset(
        ax_panel,
        np.array(sfh_true_s["t_gyr"]),
        np.array(sfh_true_s["sfr_full"]),
        color=COLORS["truth"],
        lw=1,
    )

fig.suptitle("D = 137: Approximate (geoVI) vs Exact (RT)", fontsize=11)
fig.tight_layout()
savefig("05_fig05_geovi_vs_rt_137.png")
plt.show()

# %% [markdown]
# ## 8. NUTS: Gold Standard (D < 20)
#
# No-U-Turn Sampler (Hoffman & Gelman 2014). Exact, unbiased, but scales
# poorly above D ~ 20 due to the U-turn criterion becoming unreliable in
# high dimensions. The go-to validation tool for low-dimensional models.

# %%
t0 = time.perf_counter()
result_nuts_p = fitter_param.run(
    "nuts",
    n_warmup=500,
    n_samples=5000,
    init_from=result_map_p,
    verbose=False,
)
timings["NUTS (D=7)"] = time.perf_counter() - t0
results_d7["NUTS"] = result_nuts_p

print(f"NUTS (D=7): {timings['NUTS (D=7)']:.1f}s")

# %% [markdown]
# ## 9. Elliptical Slice Sampling
#
# Murray, Adams & MacKay (2010). Exact MCMC designed for models with
# Gaussian priors on latent variables. The proposal moves along ellipses
# defined by the prior $\mathcal{N}(0, I)$, guaranteeing acceptance without
# any step-size tuning.
#
# In tengri's unbounded space, all parameters have effective $\mathcal{N}(0, I)$
# priors -- the `psd_xi` latent field explicitly, and bounded physical
# parameters via the sigmoid transform. ESS is therefore mathematically
# appropriate for the full parameter vector.

# %%
t0 = time.perf_counter()
result_ess_p = fitter_param.run(
    "elliptical_slice",
    init_from=result_map_p,
    n_samples=5000,
    n_burnin=500,
    verbose=False,
)
timings["ESS (D=7)"] = time.perf_counter() - t0
results_d7["ESS"] = result_ess_p

print(f"Elliptical Slice (D=7): {timings['ESS (D=7)']:.1f}s")

# %% [markdown]
# ## 10. geoVI-NUTS Hybrid
#
# Run geoVI first to learn the posterior geometry, then use the learned
# coordinate transform to precondition NUTS. Combines the speed of geoVI
# initialization with the exactness of NUTS. Best for D < 20 when you want
# both speed and guarantees.

# %%
t0 = time.perf_counter()
result_gn_p = fitter_param.run(
    "geovi_nuts",
    n_iterations=10,
    n_samples=3,
    n_posterior_samples=5000,
    verbose=False,
)
timings["geoVI-NUTS (D=7)"] = time.perf_counter() - t0
results_d7["geoVI-NUTS"] = result_gn_p

print(f"geoVI-NUTS (D=7): {timings['geoVI-NUTS (D=7)']:.1f}s")

# %% [markdown]
# ## 11. Pathfinder to NUTS Warm-Start
#
# Pathfinder's learned mass matrix and mode estimate can warm-start NUTS,
# reducing warmup time significantly compared to cold-start NUTS.

# %%
# NUTS from scratch (cold start)
t0 = time.perf_counter()
result_nuts_cold = fitter_param.run(
    "nuts",
    n_warmup=500,
    n_samples=2000,
    verbose=False,
)
t_cold = time.perf_counter() - t0

# Pathfinder -> NUTS (warm start)
t0 = time.perf_counter()
result_pf_init = fitter_param.run(
    "pathfinder",
    n_samples=100,
    maxiter=20,
    verbose=False,
)
result_nuts_warm = fitter_param.run(
    "nuts",
    n_warmup=200,
    n_samples=2000,
    init_from=result_pf_init,
    verbose=False,
)
t_warm = time.perf_counter() - t0
timings["Pathfinder-NUTS (D=7)"] = t_warm

print(f"NUTS (cold start):              {t_cold:.1f}s")
print(f"Pathfinder -> NUTS (warm start): {t_warm:.1f}s")
print(f"Speedup: {t_cold / t_warm:.1f}x")

# %%
# --- fig06: cold vs warm NUTS timing ---
fig, ax = plt.subplots(figsize=(6, 3))
labels = ["NUTS (cold)", "Pathfinder + NUTS"]
times_cw = [t_cold, t_warm]
colors_cw = [COLORS["nuts"], COLORS["pathfinder"]]
bars = ax.barh(labels, times_cw, color=colors_cw, alpha=0.85)
for bar, t in zip(bars, times_cw):
    ax.text(
        bar.get_width() + max(times_cw) * 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.1f}s",
        va="center",
        fontsize=9,
    )
ax.set_xlabel("Wall clock [s]")
ax.set_title("Pathfinder Warm-Start vs Cold-Start NUTS")
fig.tight_layout()
savefig("05_fig06_pathfinder_warmstart.png")
plt.show()

# %% [markdown]
# ## 12. Corner Plot: All Methods (D = 7)
#
# Seven methods on the same D = 7 mock, overlaid on a single corner plot.
# NUTS and Ray Tracing are the exact references; geoVI, Laplace, Pathfinder,
# and ESS should bracket or agree with them.

# %%
corner_results = [
    result_laplace_p,
    result_pathfinder_p,
    result_ess_p,
    result_geovi_p,
    result_rt_p,
    result_nuts_p,
    result_gn_p,
]
corner_labels = [
    "Laplace",
    "Pathfinder",
    "ESS",
    "native_geovi",
    "RT",
    "NUTS",
    "geoVI-NUTS",
]
corner_colors = [
    COLORS["laplace"],
    COLORS["pathfinder"],
    COLORS["ess"],
    COLORS["geovi"],
    COLORS["rt"],
    COLORS["nuts"],
    COLORS["mgvi"],
]

fig = plot_corner_comparison(
    corner_results,
    labels=corner_labels,
    colors=corner_colors,
    truths=true_param,
)
if fig is not None:
    fig.suptitle("D = 7: Seven Methods Compared", y=1.02)
    savefig("05_fig07_corner_all_d7.png")
plt.show()

# %% [markdown]
# ### Posterior Predictive Check
#
# Overlay 50 posterior draws on the observed photometry to verify the
# model explains the data.

# %%
# Posterior predictive: photometry
fig, ax = plt.subplots(figsize=(7, 4))

# Get filter effective wavelengths
phot_config = obs.photometry
wave_eff = np.array([float(jnp.mean(fc.wave)) for fc in phot_config.filters]) / 1e4  # microns

# 50 posterior draws from NUTS (exact reference)
n_ppc = min(50, len(next(iter(result_nuts_p.samples.values()))))
for i in range(n_ppc):
    s_i = {k: result_nuts_p.samples[k][i] for k in result_nuts_p.samples}
    phot_i = model_param.predict_photometry(s_i)
    ax.plot(wave_eff, np.array(phot_i) * 1e29, "-", color=COLORS["nuts"], alpha=0.06, lw=0.5)

# Median prediction
phot_best = model_param.predict_photometry(result_nuts_p.params)
ax.plot(
    wave_eff,
    np.array(phot_best) * 1e29,
    "o-",
    color=COLORS["nuts"],
    ms=5,
    lw=1.5,
    label="Posterior median",
)

# Truth
phot_true = model_param.predict_photometry(true_param)
ax.scatter(
    wave_eff,
    np.array(phot_true) * 1e29,
    marker="D",
    s=40,
    facecolors="none",
    edgecolors=COLORS["truth"],
    linewidths=1.2,
    zorder=6,
    label="Truth",
)

# Data
ax.errorbar(
    wave_eff,
    np.array(mock_param.flux_obs) * 1e29,
    yerr=np.array(mock_param.noise) * 1e29,
    fmt="ks",
    ms=7,
    capsize=3,
    label="Data",
    zorder=10,
)

ax.set_xlabel(r"Wavelength ($\mu$m)")
ax.set_ylabel(r"Flux ($\mu$Jy)")
ax.set_title("Posterior Predictive Check (NUTS, D = 7)")
ax.legend(fontsize=9)
fig.tight_layout()
savefig("05_fig07b_posterior_predictive.png")
plt.show()

# %% [markdown]
# ## 13. 1D Marginal Posteriors: Laplace vs Sampling
#
# The corner plot gives a global view but can be hard to read with seven
# overlapping methods. Here we compare the 1D marginal posteriors for each
# physical parameter -- this reveals where Laplace's Gaussian assumption
# holds and where it breaks down (skewed or bounded parameters).

# %%
phys_params = [p for p in spec_param.free_params if "xi" not in p]

_PLABELS = {
    "sfh_tsnorm_log_peak_sfr": r"$\log\,\mathrm{SFR_{peak}}$",
    "sfh_tsnorm_peak_lbt_gyr": r"$t_{\rm peak}$ [Gyr]",
    "sfh_tsnorm_width_gyr": r"$\sigma_{\rm SFH}$ [Gyr]",
    "sfh_tsnorm_skew": "Skew",
    "sfh_tsnorm_trunc": "Trunc",
    "met_logzsol": r"$\log\,(Z/Z_\odot)$",
    "dust_tau_bc": r"$\tau_{\rm BC}$",
    "dust_tau_diff": r"$\tau_{\rm diff}$",
}

all_results_1d = {
    "Laplace": (result_laplace_p, COLORS["laplace"], ":"),
    "Pathfinder": (result_pathfinder_p, COLORS["pathfinder"], "-."),
    "ESS": (result_ess_p, COLORS["ess"], "--"),
    "geoVI": (result_geovi_p, COLORS["geovi"], "-"),
    "RT": (result_rt_p, COLORS["rt"], "--"),
    "NUTS": (result_nuts_p, COLORS["nuts"], "-"),
}

n_cols = min(4, len(phys_params))
n_rows = (len(phys_params) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
axes_flat = np.atleast_1d(axes).ravel()

for idx, pname in enumerate(phys_params):
    ax = axes_flat[idx]
    truth_val = float(true_param.get(pname, np.nan))

    for method_name, (result, color, ls) in all_results_1d.items():
        if result.samples is not None and pname in result.samples:
            s = np.array(result.samples[pname]).ravel()
            lw = 2.5 if method_name == "NUTS" else 1.5
            ax.hist(
                s,
                bins=60,
                density=True,
                histtype="step",
                color=color,
                ls=ls,
                lw=lw,
                label=method_name,
            )
            if method_name == "NUTS":
                ax.hist(s, bins=60, density=True, histtype="stepfilled", color=color, alpha=0.12)

    if np.isfinite(truth_val):
        ax.axvline(truth_val, color=COLORS["truth"], lw=1.5, ls="--", zorder=10)
    ax.set_xlabel(_PLABELS.get(pname, pname))
    ax.set_yticks([])
    if idx == 0:
        ax.legend(fontsize=6, loc="upper right")

for idx in range(len(phys_params), len(axes_flat)):
    axes_flat[idx].set_visible(False)

fig.suptitle("1D Marginal Posteriors: Laplace (Gaussian) vs Sampling", fontsize=12, y=1.02)
fig.tight_layout()
savefig("05_fig08_marginal_posteriors.png")
plt.show()

# %% [markdown]
# ## 14. High-D Comparison (D = 137): ESS vs geoVI vs RT
#
# For high-dimensional models with GP latent fields, only scalable methods
# apply. We compare three approaches with dual 68% + 95% CI bands on the
# SFH recovery.

# %%
# ESS on D=137
t0 = time.perf_counter()
result_ess_s = fitter_stoch.run(
    "elliptical_slice",
    init_from=result_map_s,
    n_samples=2000,
    n_burnin=500,
    verbose=False,
)
timings["ESS (D=137)"] = time.perf_counter() - t0
results_d137["ESS"] = result_ess_s

print(f"ESS (D=137): {timings['ESS (D=137)']:.1f}s")

# %%
# --- fig09: Three-panel SFH with 95%+68% dual CI bands ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

panels = [
    (ax1, result_ess_s, COLORS["ess"], "ESS", timings["ESS (D=137)"]),
    (ax2, result_geovi_s, COLORS["geovi"], "geoVI", timings["native_geovi (D=137)"]),
    (ax3, result_rt_s, COLORS["rt"], "RT", timings["RT (D=137)"]),
]

sfh_true_info = model_stoch.predict_sfh(true_stoch)
t_gyr_s = np.array(sfh_true_info["t_gyr"])

for ax, result, color, name, wall_t in panels:
    if result.samples is not None:
        n_total = len(next(iter(result.samples.values())))
        sfh_draws = []
        sfh_key = "sfr_full" if model_stoch.spec.stochastic else "sfr_mean"
        for i in range(n_total):
            s_i = {k: result.samples[k][i] for k in result.samples}
            sfh_i = model_stoch.predict_sfh(s_i)
            sfh_draws.append(sfh_i[sfh_key])
        sfh_arr = np.array(sfh_draws)

        # 95% CI band (outer, lighter)
        lo95 = np.percentile(sfh_arr, 2.5, axis=0)
        hi95 = np.percentile(sfh_arr, 97.5, axis=0)
        ax.fill_between(
            t_gyr_s,
            lo95,
            hi95,
            color=color,
            alpha=0.12,
            edgecolor="none",
            label="95% CI",
            rasterized=True,
        )

        # 68% CI band (inner, darker)
        lo68 = np.percentile(sfh_arr, 16, axis=0)
        hi68 = np.percentile(sfh_arr, 84, axis=0)
        ax.fill_between(
            t_gyr_s,
            lo68,
            hi68,
            color=color,
            alpha=0.3,
            edgecolor="none",
            label="68% CI",
            rasterized=True,
        )

        # Median
        median = np.median(sfh_arr, axis=0)
        ax.plot(t_gyr_s, median, color=color, lw=1.5)

    # Truth
    ax.plot(
        t_gyr_s,
        np.array(sfh_true_info["sfr_full"]),
        color=COLORS["truth"],
        lw=2.5,
        label="Truth",
        zorder=10,
    )
    ax.plot(
        t_gyr_s,
        np.array(sfh_true_info["sfr_mean"]),
        color=COLORS["truth"],
        lw=0.8,
        ls=":",
        alpha=0.4,
    )

    ax.set_xlabel(r"$\mathrm{Lookback\ time\ /\ Gyr}$")
    ax.set_title(f"{name} ({wall_t:.1f}s)")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(bottom=0.0)
    ax.legend(fontsize=7, loc="upper left")

ax1.set_ylabel(r"$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")
fig.suptitle("D = 137: High-D Methods with 68% + 95% CI Bands", fontsize=12, y=1.02)
fig.tight_layout()
savefig("05_fig09_sfh_high_d.png")
plt.show()

# %% [markdown]
# ## 15. Convergence Diagnostics
#
# Industry-standard diagnostics (Vehtari et al. 2021; Stan/ArviZ conventions).
# Thresholds: ESS > 100 per param, divergences = 0, RT acceptance 30-70%.

# %%
# D=7 convergence table
print("=" * 60)
print("  D = 7: Convergence Diagnostics")
print("=" * 60)
ct_d7 = convergence_table(
    {
        "Laplace": result_laplace_p,
        "Pathfinder": result_pathfinder_p,
        "ESS": result_ess_p,
        "native_geovi": result_geovi_p,
        "RT": result_rt_p,
        "NUTS": result_nuts_p,
        "geoVI-NUTS": result_gn_p,
    }
)

# %%
# D=137 convergence table
print("\n" + "=" * 60)
print("  D = 137: Convergence Diagnostics")
print("=" * 60)
ct_d137 = convergence_table(
    {
        "ESS": result_ess_s,
        "native_geovi": result_geovi_s,
        "RT": result_rt_s,
    }
)

# %%
# ESS per second efficiency metric
print("\nEfficiency: ESS/s (higher = better)")
print(f"  {'Method':<20s} {'D':>4s}  {'ESS(min)':>10s} {'Time[s]':>10s} {'ESS/s':>10s}")
print("  " + "-" * 58)

efficiency_rows = [
    ("native_geovi", "D=7", result_geovi_p, timings["native_geovi (D=7)"]),
    ("RT", "D=7", result_rt_p, timings["RT (D=7)"]),
    ("NUTS", "D=7", result_nuts_p, timings["NUTS (D=7)"]),
    ("ESS", "D=7", result_ess_p, timings["ESS (D=7)"]),
    ("native_geovi", "D=137", result_geovi_s, timings["native_geovi (D=137)"]),
    ("RT", "D=137", result_rt_s, timings["RT (D=137)"]),
    ("ESS", "D=137", result_ess_s, timings["ESS (D=137)"]),
]

for name, dim, result, wall_t in efficiency_rows:
    if result.samples is not None:
        try:
            ess_dict = result.effective_sample_size()
            ess_phys = {k: v for k, v in ess_dict.items() if not k.startswith("sfh_field_xi")}
            if ess_phys:
                ess_min = min(ess_phys.values())
                ess_per_s = ess_min / wall_t if wall_t > 0 else 0.0
                print(
                    f"  {name:<20s} {dim:>4s}  {ess_min:>10.0f} {wall_t:>10.1f} {ess_per_s:>10.1f}"
                )
        except Exception:
            print(f"  {name:<20s} {dim:>4s}  {'(error)':>10s}")

# %% [markdown]
# ## 16. Timing Comparison
#
# Two-panel comparison: all methods on D = 7 (left), scalable methods on
# D = 137 (right).

# %%
# --- fig10: Timing comparison ---
methods_d7_keys = [
    "MAP (D=7)",
    "Laplace (D=7)",
    "Pathfinder (D=7)",
    "native_geovi (D=7)",
    "ESS (D=7)",
    "RT (D=7)",
    "NUTS (D=7)",
    "geoVI-NUTS (D=7)",
    "Pathfinder-NUTS (D=7)",
]
methods_d137_keys = [
    "MAP (D=137)",
    "native_geovi (D=137)",
    "ESS (D=137)",
    "RT (D=137)",
]
# Filter to only keys that exist in timings
methods_d7_keys = [m for m in methods_d7_keys if m in timings]
methods_d137_keys = [m for m in methods_d137_keys if m in timings]

_color_map = {
    "MAP": COLORS["map"],
    "Laplace": COLORS["laplace"],
    "Pathfinder": COLORS["pathfinder"],
    "native_geovi": COLORS["geovi"],
    "ESS": COLORS["ess"],
    "RT": COLORS["rt"],
    "NUTS": COLORS["nuts"],
    "geoVI-NUTS": COLORS["mgvi"],
    "Pathfinder-NUTS": COLORS["pathfinder"],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# D=7
labels_7 = [m.split(" (")[0] for m in methods_d7_keys]
times_7 = [timings[m] for m in methods_d7_keys]
colors_7 = [_color_map.get(l, "0.5") for l in labels_7]
bars = ax1.barh(range(len(labels_7)), times_7, color=colors_7, alpha=0.85)
ax1.set_yticks(range(len(labels_7)))
ax1.set_yticklabels(labels_7)
ax1.set_xlabel("Wall clock [s]")
ax1.set_title("D = 7 (Parametric)")
for bar, t in zip(bars, times_7):
    ax1.text(
        bar.get_width() + max(times_7) * 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.1f}s",
        va="center",
        fontsize=8,
    )

# D=137
labels_137 = [m.split(" (")[0] for m in methods_d137_keys]
times_137 = [timings[m] for m in methods_d137_keys]
colors_137 = [_color_map.get(l, "0.5") for l in labels_137]
bars = ax2.barh(range(len(labels_137)), times_137, color=colors_137, alpha=0.85)
ax2.set_yticks(range(len(labels_137)))
ax2.set_yticklabels(labels_137)
ax2.set_xlabel("Wall clock [s]")
ax2.set_title("D = 137 (Stochastic)")
for bar, t in zip(bars, times_137):
    ax2.text(
        bar.get_width() + max(times_137) * 0.02,
        bar.get_y() + bar.get_height() / 2,
        f"{t:.1f}s",
        va="center",
        fontsize=8,
    )

fig.suptitle("Inference Method Timing Comparison", fontsize=12)
fig.tight_layout()
savefig("05_fig10_timing_comparison.png")
plt.show()

# %% [markdown]
# ## 17. Parameter Recovery Table
#
# All seven D = 7 methods side-by-side: median +/- 68% CI vs truth.

# %%
recovery_methods = {
    "MAP": result_map_p,
    "Laplace": result_laplace_p,
    "Pathfinder": result_pathfinder_p,
    "geoVI": result_geovi_p,
    "ESS": result_ess_p,
    "RT": result_rt_p,
    "NUTS": result_nuts_p,
}

print(f"  {'Parameter':<28s}", end="")
for name in recovery_methods:
    print(f"  {name:>22s}", end="")
print(f"  {'Truth':>8s}")
print("  " + "-" * (28 + 24 * len(recovery_methods) + 10))

for pname in phys_params:
    label = _PLABELS.get(pname, pname)
    print(f"  {label:<28s}", end="")
    for _method_name, result in recovery_methods.items():
        if result.samples is not None and pname in result.samples:
            s = np.array(result.samples[pname]).ravel()
            med = np.median(s)
            lo, hi = np.percentile(s, [16, 84])
            print(f"  {med:>7.2f} [{lo:>6.2f}, {hi:>6.2f}]", end="")
        elif hasattr(result, "params") and pname in result.params:
            val = float(result.params[pname])
            print(f"  {val:>7.2f} {'(MAP)':>14s}", end="")
        else:
            print(f"  {'--':>22s}", end="")
    truth_val = float(true_param.get(pname, np.nan))
    print(f"  {truth_val:>8.2f}")

# %%
# Relative CI width compared to NUTS (NUTS = 1.0)
print("\n  Relative 68% CI width (NUTS = 1.0):")
print(f"  {'Parameter':<28s}", end="")
for name in ["Laplace", "Pathfinder", "geoVI", "ESS", "RT"]:
    print(f"  {name:>12s}", end="")
print()
print("  " + "-" * (28 + 14 * 5))

for pname in phys_params:
    label = _PLABELS.get(pname, pname)
    nuts_s = np.array(result_nuts_p.samples[pname]).ravel()
    nuts_width = np.percentile(nuts_s, 84) - np.percentile(nuts_s, 16)
    if nuts_width < 1e-10:
        continue
    print(f"  {label:<28s}", end="")
    for method_name in ["Laplace", "Pathfinder", "geoVI", "ESS", "RT"]:
        result = recovery_methods[method_name]
        if result.samples is not None and pname in result.samples:
            s = np.array(result.samples[pname]).ravel()
            width = np.percentile(s, 84) - np.percentile(s, 16)
            ratio = width / nuts_width
            print(f"  {ratio:>12.2f}", end="")
        else:
            print(f"  {'--':>12s}", end="")
    print()

# %% [markdown]
# ## 18. Method Selection Decision Tree
#
# | # | Scenario | Recommended | Why |
# |---|----------|------------|-----|
# | 1 | Default for everything | `native_geovi` | Fast, JIT-compiled, any D |
# | 2 | Exact posterior (D < 20) | `nuts` | Gold standard, unbiased |
# | 3 | Exact posterior (D > 20) | `raytrace` | Scales to D = 300, gradient-noise tolerant |
# | 4 | Exact + warm start (D < 20) | `geovi_nuts` | geoVI preconditioning + NUTS exactness |
# | 5 | Quick error bars from MAP | `laplace` | Instant Gaussian, Laplace evidence |
# | 6 | Fast approximate posterior | `pathfinder` | 10x faster than NUTS, good NUTS initializer |
# | 7 | Gaussian-prior latent models | `elliptical_slice` | Natural for GP priors, no step-size tuning |
# | 8 | Bayesian evidence ($\log Z$) | `nss` or `laplace` | NSS for smooth models (D < 30), Laplace for quick estimate |
# | 9 | Hierarchical inference | `native_geovi` | Block Gibbs for shared PSD hyperparameters |
# | 10 | Point estimate only | `map` | Fastest. Initialization for all other methods |
#
# The loss function is always the same -- only the exploration strategy changes.
# Switching methods is one line: `fitter.run("method_name")`.

# %%
# Summary table
print("\n  Method Summary")
print("  " + "=" * 78)
print(f"  {'Method':<22s} {'D range':>8s}  {'Exact?':>6s}  {'Time':>8s}  {'Best for'}")
print("  " + "-" * 78)
rows = [
    (
        "native_geovi",
        "any",
        "No",
        f"{timings.get('native_geovi (D=137)', 0):.1f}s",
        "Default for everything",
    ),
    ("Ray Tracing", "< 300", "Yes", f"{timings.get('RT (D=137)', 0):.1f}s", "Exact high-D MCMC"),
    ("NUTS", "< 20", "Yes", f"{timings.get('NUTS (D=7)', 0):.1f}s", "Low-D validation"),
    (
        "geoVI-NUTS",
        "< 20",
        "Yes",
        f"{timings.get('geoVI-NUTS (D=7)', 0):.1f}s",
        "Exact + warm start",
    ),
    ("ESS", "any", "Yes", f"{timings.get('ESS (D=7)', 0):.1f}s", "Gaussian-prior latent"),
    (
        "Pathfinder",
        "< 50",
        "No",
        f"{timings.get('Pathfinder (D=7)', 0):.1f}s",
        "Fast approx, NUTS init",
    ),
    ("Laplace", "< 50", "No", f"{timings.get('Laplace (D=7)', 0):.1f}s", "Instant from MAP"),
    ("NSS", "< 30", "Yes", "var", "Bayesian evidence"),
    ("MAP", "any", "No", f"{timings.get('MAP (D=7)', 0):.1f}s", "Initialization"),
]
for name, d, exact, t, best in rows:
    print(f"  {name:<22s} {d:>8s}  {exact:>6s}  {t:>8s}  {best}")
print("  " + "=" * 78)
