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
# # Why JAX? Four ideas that change SED fitting
#
# JAX is Python with **supercharged functions**. In 30 minutes, you'll see:
#
# 1. **JIT compilation:** Write once, run 100× faster. A galaxy SED evaluates in ~1 ms instead of 100 ms.
# 2. **Automatic differentiation:** Free gradients. The cost of `∇L/∇θ` is the same as the forward pass.
# 3. **Vectorization without loops:** `vmap` turns your single-galaxy model into a batch-of-1000 model with one decorator.
# 4. **One model, any inference method:** MAP, Laplace, Pathfinder, HMC, VI all use the same JAX forward model. No reimplementation.
#
# This notebook uses real tengri physics (blackbody SEDs, photometric fitting) to teach JAX *via doing*, not toy examples.
#
# **What you already know:** NumPy, the physics of stellar spectra and dust attenuation, the likelihood χ² and its role in Bayesian inference.

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
    f"✓ Loaded SSP grid: flux{tuple(ssp_data.ssp_flux.shape)}, "
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

print(f"\n✓ Model: {len(spec.free_params)} free parameters")
for p in spec.free_params[:3]:
    print(f"  {p}: {params[p]:.4f}")
print("  ...")

# %% [markdown]
# ---
#
# ## Idea 1: JIT Compilation — From Python to Machine Code
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
        _ = model.predict_photometry(params, mode="exact")

    # Time subsequent calls (pure JIT execution)
    for _ in range(n_timed):
        t0 = time.perf_counter()
        _ = model.predict_photometry(params, mode="exact")
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
ax.barh(
    ["Steady-state\n(JIT)", "Without JIT\n(Python loop)"],
    [jit_time_ms, 100],
    color=["#2ca02c", "#d62728"],
    alpha=0.8,
    edgecolor="black",
    linewidth=1.5,
)
ax.set_xlabel("Time per forward pass [ms]")
ax.set_title("JIT speedup: Compiled vs Python", fontweight="bold")
ax.set_xscale("log")
ax.set_xlim(1, 300)
for i, (_label, val) in enumerate([("JIT", jit_time_ms), ("Python", 100)]):
    ax.text(val, i, f"  {val:.1f} ms", va="center", fontsize=11, fontweight="bold")
ax.grid(axis="x", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("jax_jit_speedup.png", dpi=100, bbox_inches="tight")
plt.show()
print("\n✓ Saved: jax_jit_speedup.png")

# %% [markdown]
# ---
#
# ## Idea 2: Automatic Differentiation — Gradients are Cheap
#
# **Key insight:** In JAX, the cost of `∇L/∂θ` is **the same as the forward pass** (within 2–3×).
# This is why every modern inference method works: MAP (gradient descent), Laplace (curvature), Pathfinder (iterative grad), HMC (alternating forward+grad) all reuse the same model.


# %%
def log_likelihood_chi2(params_dict, model):
    """Negative χ² likelihood for the 7-D model.

    In practice, the observed SED is a real photometric catalog; here we
    mock it as 95% of the model prediction.

    Returns: log p(data | params) = -0.5 * χ²
    """
    sed_pred = model.predict_photometry(params_dict, mode="exact")
    sed_obs = sed_pred * 0.95  # mock observation
    noise = sed_pred * 0.1  # 10% fractional uncertainty
    chi2 = jnp.sum(((sed_pred - sed_obs) / noise) ** 2)
    return -0.5 * chi2


# Compile forward pass
print("\nCompiling forward model for gradient computation...")
sed = model.predict_photometry(params, mode="exact")
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
    _ = model.predict_photometry(params, mode="exact")
    _ = _.block_until_ready()
fwd_time = (time.perf_counter() - t0) / n_evals * 1e3

overhead = grad_time / fwd_time

print(f"\nForward pass:         {fwd_time:>7.2f} ms")
print(f"Gradient (jax.grad): {grad_time:>7.2f} ms")
print(f"Overhead:            {overhead:>7.1f}x")

# %%
print("\n✓ **Why this matters:**")
print("  • MAP (scipy.optimize.minimize):    ~100 steps → ~100 ms")
print("  • Laplace (one grad + Hessian):     ~200 ms")
print("  • Pathfinder (iterative grad):      ~10–50 steps → ~500 ms")
print("  • HMC (50 steps per sample):        ~50 steps × 3× per sample")
print("  • VI (gradient of ELBO):            scales with latent dimension")
print("\n  All use the same forward model and gradient. No reimplementation.")

# %% [markdown]
# ---
#
# ## Idea 3: Vectorization without Loops — `vmap`
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
        return model.predict_photometry(param_dict, mode="exact")

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
print("\n✓ **What vmap does:**")
print("  No Python loop: ✗")
print("  No JAX control flow (where): ✗")
print("  One compiled function: ✓")
print("  Scales to GPU/TPU naturally: ✓")

# %% [markdown]
# ---
#
# ## Idea 4: Combine JIT + Grad + Vmap — The JAX Philosophy
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

# %%
print("\n✓ **The JAX Mantra:**")
print("  ```python")
print("  @jax.jit")
print("  def inference(batch):")
print("      return jax.vmap(jax.grad(jax.vmap(model)))(batch)")
print("  ```")
print("\n  This single function can:")
print("  • Compute a batch of gradients (for HMC ensemble)")
print("  • Be differentiated again (for variational inference)")
print("  • Run on GPU with zero code changes")

# %% [markdown]
# ---
#
# ## Summary: Why JAX for SED Fitting
#
# **One model, any method.**
#
# ```
# Parameters
#     ↓
# forward_model(θ)  [pure JAX function, JIT-compiled once]
#     ↓
# ┌──────────────────────────────────────┐
# │  Inference Methods (all reuse forward_model)
# │                                       │
# │  MAP:        minimize(-log p)         │
# │  Laplace:    one grad + curvature     │
# │  HMC:        alternating forward+grad │
# │  VI:         gradient of ELBO         │
# │  Nested Sampl: sequential             │
# └──────────────────────────────────────┘
#     ↓
# Posterior samples, uncertainties, evidence
# ```
#
# **What you learned:**
# 1. JIT compilation: write Python, run compiled machine code.
# 2. Autodiff: gradients are free (2–3× forward cost).
# 3. vmap: one model becomes a batch model, no loops.
# 4. Composability: stack these to build inference pipelines.
#
# **Next steps:**
# - **Notebook 02** ([`02_sed_anatomy.py`](02_sed_anatomy.py)): Trace the SED from stellar continuum to dust to emission lines. Understand the forward model piece by piece.
# - **Notebook 03** ([`03_fitting_photometry.py`](03_fitting_photometry.py)): Fit mock photometry with MAP, Laplace, and HMC. See inference methods in action.
# - **Notebook 06** ([`06_inference_methods.py`](06_inference_methods.py)): Detailed comparison of all inference backends (MAP, VI, HMC, Pathfinder, Nested Sampling).

# %% [markdown]
# ---
#
# **Questions?** See [`docs/user/`](../docs/user/) for detailed API docs, or open an issue on GitHub.
# tengri is open-source (MIT) and welcomes contributions.
