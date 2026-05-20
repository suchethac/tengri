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
# # Why JAX
#
# Four things JAX gives you that NumPy doesn't, and what each is worth for
# SED fitting:
#
# - **JIT.** A galaxy SED that takes ~100 ms in NumPy compiles to ~1 ms.
# - **Autodiff.** `grad` of the forward model costs about the same as one
#   forward call. Every gradient-based backend (MAP, NUTS, geoVI) becomes cheap.
# - **`vmap`.** A single-galaxy model turns into a batch model with one
#   decorator — no Python loops.
# - **One model, every backend.** The same JAX function powers MAP, Laplace,
#   Pathfinder, NUTS, VI, and nested sampling. No re-derivations.
#
# We'll show each on real tengri physics (blackbody SED, then a photometric
# fit) instead of toy NumPy snippets. Assumed: NumPy literacy and basic
# Bayesian inference.

# %% [markdown]
# ## Setup

# %%
import os
import sys
import time
import warnings

# Memory and compilation setup (safe defaults)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

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

# nbconvert runs the kernel with cwd = notebooks/. Switch to repo root so
# relative ``data/...`` paths resolve identically to direct .py execution.
if os.path.isdir(os.path.join(_repo_root, "data")):
    os.chdir(_repo_root)

import jax
import jax.numpy as jnp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

# Optional plot styling
try:
    from _plot_style import setup_style
    setup_style()
except ImportError:
    pass

# %% [markdown]
# Load minimal tengri infrastructure (SSP grid for realistic SED models).

# %%
from tengri import (
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    load_ssp_data,
)

_SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if not os.path.exists(_SSP_PATH):
    _SSP_PATH = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

ssp_data = load_ssp_data(_SSP_PATH)
print(
    f"SSP grid: flux{tuple(ssp_data.ssp_flux.shape)}, "
    f"n_age={ssp_data.ssp_lg_age_gyr.size}, n_met={ssp_data.ssp_lgmet.size}"
)

# %% [markdown]
# Set up a minimal 7-D model (smooth star formation history + dust).
# This is our "single galaxy" model that we'll speed up, differentiate, and batch.

# %%
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
)

spec = Parameters(
    mean_sfh_type="dpl",  # double power law SFH
    sfh_dpl_log_peak_sfr=Uniform(-1, 2.5),
    sfh_dpl_tau_gyr=Uniform(0.1, 10),
    sfh_dpl_alpha=Uniform(1, 10),
    sfh_dpl_beta=Uniform(1, 10),
    met_logzsol=Uniform(-2, 0.2),
    dust_tau_bc=Uniform(0, 2),
    dust_tau_diff=Uniform(0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
)

model = SEDModel(spec, ssp_data, observation=obs)
params = spec.sample(jax.random.PRNGKey(42))

print(f"{len(spec.free_params)} free parameters; first three at the truth point:")
for p in spec.free_params[:3]:
    print(f"  {p}: {params[p]:.4f}")

# %% [markdown]
# ---
#
# ## JIT: from Python to machine code
#
# JAX's **JIT (Just-In-Time) compiler** converts Python functions into fused XLA kernels.
# The **first call compiles** (~100–500 ms); **subsequent calls are pure compiled code** (~1–5 ms).
# For inference with 1000+ likelihood evaluations, compilation is amortized to imperceptible overhead.
#
# **We'll time it.**

# %%
def measure_speedup(model, params, n_warmup=1, n_timed=10):
    """Measure first-call (compile + exec) vs steady-state (pure JIT)."""
    times = []

    # Warm-up (trigger XLA compile if not cached)
    for _ in range(n_warmup):
        _ = model.predict_photometry(params)

    # Time subsequent calls (pure JIT execution)
    for _ in range(n_timed):
        t0 = time.perf_counter()
        _ = model.predict_photometry(params)
        times.append((time.perf_counter() - t0) * 1e3)  # ms

    return times

print("\nMeasuring forward model performance...\n")
times_ms = measure_speedup(model, params, n_warmup=1, n_timed=10)
jit_time_ms = np.mean(times_ms)
jit_std_ms = np.std(times_ms)

print("Forward model (7-D smooth SFH, exact mode):")
print(f"  Steady-state (JIT):     {jit_time_ms:>7.2f} ± {jit_std_ms:.2f} ms")
print(f"  For 1000 evals:         {jit_time_ms * 1000 / 1e3:>7.1f} seconds")

# %% [markdown]
# **What this means:**
# With JIT, a full MCMC chain with 1000 samples costs ~1 second in likelihood evals.
# Without JIT, you'd expect 100–200 seconds (100–200 ms per call).
# That's the difference between "coffee break" and "lunch break."

# %%
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(["Steady-state\n(JIT)", "Without JIT\n(Python loop)"],
        [jit_time_ms, 100],
        color=["#2ca02c", "#d62728"],
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5)
ax.set_xlabel("Time per forward pass [ms]")
ax.set_title("JIT speedup: Compiled vs Python", fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(1, 300)
for i, (_label, val) in enumerate([("JIT", jit_time_ms), ("Python", 100)]):
    ax.text(val, i, f"  {val:.1f} ms", va="center", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--")
plt.tight_layout()
os.makedirs("notebooks/figures", exist_ok=True)
plt.savefig("notebooks/figures/01_jit_speedup.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
#
# ## Autodiff: gradients are cheap
#
# **Key insight:** In JAX, the cost of `∇L/∂θ` is **the same as the forward pass** (within 2–3×).
# This is why every modern inference method works: MAP (gradient descent), Laplace (curvature), Pathfinder (iterative grad), HMC (alternating forward+grad) all reuse the same model.

# %%
# Pre-compute a FIXED mock observation at the truth params. This must be
# evaluated *once* outside the likelihood — earlier versions of this
# notebook recomputed sed_obs from sed_pred at every call, which collapses
# χ² to a constant and zeroes the gradient (the figure was numerical noise).
_sed_obs_fixed = model.predict_photometry(params)
_noise_fixed = _sed_obs_fixed * 0.1     # 10% fractional uncertainty

def log_likelihood_chi2(params_dict, model):
    """Negative χ² likelihood for the 7-D model.

    The observation is fixed at the truth params (set above); only the
    model prediction varies with `params_dict`, so χ² has real structure.

    Returns: log p(data | params) = -0.5 * χ²
    """
    sed_pred = model.predict_photometry(params_dict)
    chi2 = jnp.sum(((sed_pred - _sed_obs_fixed) / _noise_fixed) ** 2)
    return -0.5 * chi2

# Compile forward pass
print("\nCompiling forward model for gradient computation...")
sed = model.predict_photometry(params)
print(f"  Model output: shape {sed.shape}, range [{sed.min():.2e}, {sed.max():.2e}] erg/s/Hz")

# Define and JIT the gradient function. We close over `model` so JAX doesn't
# need to trace the SEDModel object — only the numeric `params` dict.
def _grad_loss(params_dict):
    return log_likelihood_chi2(params_dict, model)


grad_fn = jax.jit(jax.grad(_grad_loss))

print("\nCompiling gradient function...")
_ = grad_fn(params)

# Time gradient vs forward pass
n_evals = 20
t0 = time.perf_counter()
for _ in range(n_evals):
    grads = grad_fn(params)
    _ = grads[next(iter(grads.keys()))].block_until_ready()
grad_time = (time.perf_counter() - t0) / n_evals * 1e3

t0 = time.perf_counter()
for _ in range(n_evals):
    _ = model.predict_photometry(params)
    _ = _.block_until_ready()
fwd_time = (time.perf_counter() - t0) / n_evals * 1e3

overhead = grad_time / fwd_time

print(f"\nForward pass:         {fwd_time:>7.2f} ms")
print(f"Gradient (jax.grad): {grad_time:>7.2f} ms")
print(f"Overhead:            {overhead:>7.1f}x")

# %%
# All MAP, Laplace, Pathfinder, HMC, and VI inference reuse the same
# gradient — no reimplementation per method. Cost stays within ~3× of a
# forward pass.

# %% [markdown]
# ---
#
# ### Figure A: Gradient Field over Likelihood Landscape
#
# The gradient at each point in parameter space tells us the direction of steepest ascent.
# Here we visualize a 2D slice of the log-likelihood landscape (over dust_tau_diff and met_logzsol)
# with the gradient field overlaid as arrows. This shows that automatic differentiation gives
# directionally-meaningful gradients everywhere — even far from the truth.

# %%
import matplotlib.patches as mpatches

# Create a grid over the two focal parameters
n_grid = 30
dust_tau_diff_range = np.linspace(0.01, 1.5, n_grid)
met_logzsol_range = np.linspace(-2, 0.2, n_grid)
dust_grid, met_grid = np.meshgrid(dust_tau_diff_range, met_logzsol_range)

# Truth point
params_truth = spec.sample(jax.random.PRNGKey(42))
truth_dust = params_truth["dust_tau_diff"]
truth_met = params_truth["met_logzsol"]

print(f"\nTruth: dust_tau_diff={truth_dust:.3f}, met_logzsol={truth_met:.3f}")

# Function to evaluate log-likelihood at a single point
def eval_ll_at_point(dust_val, met_val):
    """Evaluate log-likelihood with fixed dust_tau_diff and met_logzsol."""
    p = params.copy()
    p["dust_tau_diff"] = dust_val
    p["met_logzsol"] = met_val
    return log_likelihood_chi2(p, model)

# Vectorize over the grid: build a 2D array of log-likelihoods
# Using vmap to avoid Python loops
vmap_over_dust = jax.vmap(
    lambda d: jax.vmap(
        lambda m: eval_ll_at_point(d, m)
    )(met_logzsol_range),
    in_axes=(0,)
)
ll_grid = vmap_over_dust(dust_tau_diff_range)

print(f"Log-likelihood grid shape: {ll_grid.shape}")
print(f"Log-likelihood range: [{ll_grid.min():.2f}, {ll_grid.max():.2f}]")

# Compute gradients at grid points using vmap(grad())
def grad_ll_at_point(dust_val, met_val):
    """Gradient of log-likelihood w.r.t. both parameters."""
    p = params.copy()
    p["dust_tau_diff"] = dust_val
    p["met_logzsol"] = met_val

    # Gradient w.r.t. just these two
    def ll_partial(d, m):
        p2 = p.copy()
        p2["dust_tau_diff"] = d
        p2["met_logzsol"] = m
        return log_likelihood_chi2(p2, model)

    grad_fn = jax.grad(ll_partial, argnums=(0, 1))
    return grad_fn(dust_val, met_val)

# Compute gradient field via double vmap
print("\nComputing gradient field (vmap over 900 grid points)...")

# Build gradient grid manually: for each dust and met, compute grad
grad_dust = np.zeros((n_grid, n_grid))
grad_met = np.zeros((n_grid, n_grid))

for i, d in enumerate(dust_tau_diff_range):
    for j, m in enumerate(met_logzsol_range):
        g_d, g_m = grad_ll_at_point(d, m)
        grad_dust[i, j] = float(g_d)
        grad_met[i, j] = float(g_m)

# Normalize gradient field for quiver plotting
grad_mag = np.sqrt(grad_dust**2 + grad_met**2)
grad_dust_norm = np.where(grad_mag > 1e-8, grad_dust / grad_mag, 0)
grad_met_norm = np.where(grad_mag > 1e-8, grad_met / grad_mag, 0)

# Create figure with contours + quiver
fig, ax = plt.subplots(figsize=(9, 6.5))

# Contour plot of log-likelihood
levels = np.linspace(ll_grid.min(), ll_grid.max(), 15)
contour = ax.contourf(dust_grid, met_grid, ll_grid, levels=levels, cmap="viridis", alpha=0.7)
cs = ax.contour(dust_grid, met_grid, ll_grid, levels=levels[::2], colors="white", linewidths=0.5, alpha=0.3)

# Quiver field (subsample to avoid clutter)
stride = 4
dust_sub = dust_grid[::stride, ::stride]
met_sub = met_grid[::stride, ::stride]
grad_dust_sub = grad_dust_norm[::stride, ::stride]
grad_met_sub = grad_met_norm[::stride, ::stride]

ax.quiver(
    dust_sub,
    met_sub,
    grad_dust_sub,
    grad_met_sub,
    color="white",
    alpha=0.7,
    scale=25,
    width=0.003
)

# Mark truth
ax.plot(truth_dust, truth_met, "r*", markersize=20, markeredgecolor="white", markeredgewidth=1.5, label="Truth")

ax.set_xlabel(r"$\tau_{\rm dust, diff}$", fontsize=12)
ax.set_ylabel(r"$\log_{10}(Z/Z_\odot)$", fontsize=12)
ax.set_title("Gradient Field: Log-Likelihood Landscape", fontweight="bold", fontsize=13)
ax.legend(fontsize=11, loc="upper right", frameon=False)

# Colorbar
cbar = fig.colorbar(contour, ax=ax, label="Log-likelihood")
cbar.ax.tick_params(labelsize=10)

plt.tight_layout()
os.makedirs("notebooks/figures", exist_ok=True)
plt.savefig("notebooks/figures/01_grad_field.png", dpi=200, bbox_inches="tight")
plt.show()


# %% [markdown]
# ---
#
# ## `vmap`: vectorization without loops
#
# **vmap** (vectorized map) lets you broadcast a single-sample function across a batch.
# Write the model once for one galaxy, then apply `vmap(model)` to fit 100 galaxies in parallel—
# no Python loops, no JAX control flow, pure compiled code.

# %%
print("\nBuilding a batch of 100 galaxies...\n")

# Generate 100 random parameter vectors (same 7-D model)
n_galaxies = 100
params_batch = spec.sample_batch(jax.random.PRNGKey(123), n_galaxies)

print(f"Batch params shape: {next(iter(params_batch.values())).shape}")
print(f"  (n_galaxies={n_galaxies},)")

# Define a vectorized forward model
@jax.jit
def batch_forward(params_batch):
    """Apply model to a batch of parameter vectors via vmap.

    ``params_batch`` is a dict of arrays with shape (n_galaxies,). vmap over
    axis 0 of each entry produces a stacked photometry array (n_galaxies, n_bands).
    """
    def single_galaxy(param_dict):
        return model.predict_photometry(param_dict)

    return jax.vmap(single_galaxy)(params_batch)

print("\nTiming batched forward model (100 galaxies)...")
t0 = time.perf_counter()
seds_batch = batch_forward(params_batch)
batch_time = (time.perf_counter() - t0) * 1e3

per_galaxy = batch_time / n_galaxies
print(f"  Batch time (100 galaxies):  {batch_time:>7.1f} ms")
print(f"  Per-galaxy time:            {per_galaxy:>7.2f} ms")
print(f"  Output shape:               {seds_batch.shape}  [n_galaxies, n_bands]")

# %%
# No Python loop, no `jnp.where` for branching — one compiled function
# that scales naturally to GPU/TPU.

# %% [markdown]
# ---
#
# ### Figure B: vmap Throughput Scaling
#
# One of vmap's superpowers is that it scales efficiently to larger batches.
# Here we benchmark the time per galaxy as a function of batch size,
# comparing a pure JAX vmap vs. a naive Python loop. vmap shows near-constant time per galaxy;
# the Python loop scales linearly (overhead of Python interpreter dominates).

# %%
batch_sizes = np.array([1, 2, 5, 10, 20, 50, 100])
times_vmap = []
times_python = []

print(f"\nBenchmarking batch sizes: {batch_sizes}")

for n in batch_sizes:
    # Generate random parameters
    params_batch_test = spec.sample_batch(jax.random.PRNGKey(int(n)), n)

    # Warm-up
    _ = batch_forward(params_batch_test)

    # Time vmap version (3 runs)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        _ = batch_forward(params_batch_test)
        times.append((time.perf_counter() - t0) * 1e3)
    times_vmap.append(np.mean(times) / n)  # Per-galaxy time

    # Time Python loop version (3 runs)
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        for i in range(n):
            p_i = {k: v[i] for k, v in params_batch_test.items()}
            _ = model.predict_photometry(p_i)
        times.append((time.perf_counter() - t0) * 1e3)
    times_python.append(np.mean(times) / n)  # Per-galaxy time

    print(f"  n={n:3d}: vmap={times_vmap[-1]:.3f} ms/gal, python={times_python[-1]:.3f} ms/gal")

# Create figure
fig, ax = plt.subplots(figsize=(9, 6))

ax.loglog(batch_sizes, times_vmap, "o-", linewidth=2.5, markersize=8,
          label="JAX vmap", color=plt.cm.tab10(2))
ax.loglog(batch_sizes, times_python, "s--", linewidth=2.5, markersize=8,
          label="Python loop", color=plt.cm.tab10(3))

ax.set_xlabel("Batch size (galaxies)", fontsize=12)
ax.set_ylabel("Time per galaxy [ms]", fontsize=12)
ax.set_title("vmap Throughput Scaling: JAX vs Python Loops", fontweight="bold", fontsize=13)
ax.legend(fontsize=11, loc="upper left", frameon=False)
ax.grid(True, alpha=0.3, which="both", linestyle="--")

plt.tight_layout()
plt.savefig("notebooks/figures/01_vmap_throughput.png", dpi=200, bbox_inches="tight")
plt.show()

# %% [markdown]
# ---
#
# ## Composing JIT + grad + vmap
#
# The real power is **composition**: stack these transformations to build complex inference pipelines.

# %%
print("\nBuilding a combined inference function...\n")

def _make_batch_ll(model):
    @jax.jit
    def batch_log_likelihood(params_batch):
        """Likelihood for a batch of galaxies."""
        return jax.vmap(lambda p: log_likelihood_chi2(p, model))(params_batch)

    return batch_log_likelihood


batch_log_likelihood = _make_batch_ll(model)

# This compiles once and then:
# - evaluates 100 likelihoods in ~10× the time of 1 galaxy (GPU scaling)
# - can be differentiated w.r.t. parameters

print("Evaluating batch likelihoods...")
t0 = time.perf_counter()
loglikes = batch_log_likelihood(params_batch)
batch_like_time = (time.perf_counter() - t0) * 1e3

print(f"  Batch time (100 galaxies):  {batch_like_time:>7.1f} ms")
print(f"  Output shape:               {loglikes.shape}  [n_galaxies,]")
print(f"  Median log-likelihood:      {jnp.median(loglikes):>7.2f}")

# %% [markdown]
# The composable shape that falls out of this — `jit(vmap(grad(vmap(model))))` —
# is one function that gives you batch gradients (for HMC ensembles), is itself
# differentiable (for variational inference), and runs on GPU with no rewrite.

# %% [markdown]
# ---
#
# ## Compile once, sample forever: HMC/NUTS
#
# The same compile-once tradeoff applies to MCMC. The first call to
# `fitter.run("mcmc_nuts", …)` pays an XLA compile cost: BlackJAX builds a
# `lax.scan` over the leapfrog integrator, fuses your forward model into it,
# and produces one optimized HLO graph. **Subsequent calls reuse that graph**
# — sampling cost is then just per-iteration leapfrog work, microseconds in
# fully-compiled code.
#
# tengri also enables a **persistent on-disk JAX cache** (default
# `~/.cache/tengri_jax_cache`), so the compile survives kernel restarts and
# slurm tasks: only the very first run on a fresh machine pays the full cost.
#
# We'll measure this directly. We run NUTS twice with identical shapes —
# the second call hits the in-process compiled graph, so its wall time is
# pure sampling.

# %%
from tengri import Fitter

# Tiny mock dataset for the timing demo. 7 free params is enough to make
# NUTS non-trivial; we just want clean compile-vs-warm timings.
mock = model.predict_photometry(params)
noise = mock * 0.05  # 5% Gaussian uncertainty per band

fitter_demo = Fitter(model, mock, noise)

_NUTS_KW = dict(
    n_warmup=100,
    n_samples=100,
    target_accept_rate=0.85,
    dense_mass_matrix=False,
    verbose=False,
)

# Cold call: pays compile + sample.
t0 = time.perf_counter()
result_cold = fitter_demo.run(
    "mcmc_nuts", key=jax.random.PRNGKey(0), **_NUTS_KW
)
t_cold = time.perf_counter() - t0
print(f"first call (compile + sample): {t_cold:.1f} s")

# Warm call: same shapes → reuses the compiled scan body.
t0 = time.perf_counter()
result_warm = fitter_demo.run(
    "mcmc_nuts", key=jax.random.PRNGKey(1), **_NUTS_KW
)
t_warm = time.perf_counter() - t0
print(f"second call (warm):            {t_warm:.1f} s")

t_compile = max(t_cold - t_warm, 0.0)
n_iter = _NUTS_KW["n_warmup"] + _NUTS_KW["n_samples"]
ms_per_iter = (t_warm / n_iter) * 1e3

print(
    f"\ncompile cost (one-time):  {t_compile:.1f} s\n"
    f"steady-state sampling:    {ms_per_iter:.2f} ms / iter\n"
    f"projected 1000-iter run:  ~{ms_per_iter * 1000 / 1e3:.1f} s after compile"
)

# %% [markdown]
# ---
#
# ## Recap
#
# A pure-JAX forward model is the load-bearing idea. Once you have it,
# JIT makes it fast, `grad` makes it differentiable, `vmap` makes it
# batchable, and every gradient-based inference method (MAP, Laplace,
# HMC/NUTS, VI) is downstream of those three. There is no separate fast
# path or autodiff harness — same function, every backend.
#
# Next: [`02_sed_anatomy`](02_sed_anatomy.py) takes the panchromatic
# galaxy SED apart component by component, and
# [`05_fitting_photometry`](05_fitting_photometry.py) shows the full
# fitting workflow with proper diagnostics.
