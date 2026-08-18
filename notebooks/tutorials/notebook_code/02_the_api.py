# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # The tengri API: SEDModel, Fitter, Posterior
#
# tengri has three core objects: `SEDModel` (forward model), `Fitter` (inference
# engine), `Posterior` (results container). This notebook shows how to create,
# configure, inspect, and compose them. You'll also learn the three JAX
# patterns that make everything fast: `jit` (compile), `vmap` (batch),
# `grad` (differentiate).

# %%
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
    Gaussian,
    LogNormal,
    LogUniform,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    StudentT,
    Uniform,
    load_ssp_data,
)

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

FIGDIR = os.path.join("tutorials", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import setup_style, COLORS

setup_style()

# %%
ssp_data = load_ssp_data(
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]))

# %% [markdown]
# ## Parameters: Defining Your Model
#
# Every fit starts with a `Parameters`. It declares which parameters are free,
# what priors they have, and which are fixed. The `Parameters` object is the single
# source of truth for mock generation, inference, and parameter fixing.

# %%
# Build a Parameters spec step by step
spec = Parameters(
    # SFH shape (truncated skew-normal, Bellstedt+2020)
    sfh_tsnorm_log_total_mass=Uniform(7.0, 12.5),      # log10(M*/Msun)
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),        # peak lookback time
    sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),             # Gaussian width
    sfh_tsnorm_skew=Uniform(-3.0, 3.0),                 # skewness
    sfh_tsnorm_trunc=Uniform(1.0, 10.0),                # truncation sharpness
    # Metallicity
    met_logzsol=Uniform(-2.0, 0.2),                     # log(Z/Zsun)
    # Dust (Charlot & Fall 2000)
    dust_tau_bc=Uniform(0.0, 2.0),                      # birth cloud optical depth
    dust_tau_diff=Uniform(0.0, 1.5),                    # diffuse ISM optical depth
    dust_slope=Fixed(-0.7),                              # attenuation slope (fixed)
    # Redshift
    redshift=Fixed(0.1),                                 # Fixed → enables precomputation
    # Model configuration
    mean_sfh_type="tsnorm",
)

# Scalar shorthand: dust_slope=-0.7 is identical to Fixed(-0.7)
print(f"Free parameters: {spec.n_free}")
for name in spec.free_params:
    print(f"  {name}")

# %%
# Inspect the Parameters spec
print(f"\nFree params:  {spec.free_params}")
print(f"Fixed params: {spec.fixed_params}")
print(f"Total free:   {spec.n_free}")
print(f"Stochastic:   {spec.stochastic}")

# %%
# Standardization: the key to universal inference
#
# In latent space, every parameter has a standard Gaussian prior N(0,1).
# This is what makes all inference methods work on the same loss function.
# The physics enters through the forward model f(ξ), not through the prior.
#
# Each Distribution has standardize() and unstandardize() methods.
key = jax.random.PRNGKey(0)
params = spec.sample(key)

print("Round-trip test: physical → latent → physical (should match):")
for name in spec.free_params:
    dist = spec.get_distribution(name)
    theta = params[name]
    xi = dist.standardize(theta)
    theta_rt = dist.unstandardize(xi)
    match = abs(float(theta) - float(theta_rt)) < 1e-10
    print(f"  {name:<35s}: {float(theta):.6f} → ξ={float(xi):.4f} → {float(theta_rt):.6f}  {'✓' if match else '✗'}")

# %%
# Sample from the prior
samples = jax.vmap(spec.sample)(jax.random.split(key, 5))
print("\n5 prior samples:")
print(f"  {'Parameter':<35s} {'S1':>8s} {'S2':>8s} {'S3':>8s} {'S4':>8s} {'S5':>8s}")
for name in spec.free_params:
    vals = [f"{float(samples[name][i]):.3f}" for i in range(5)]
    print(f"  {name:<35s} " + " ".join(f"{v:>8s}" for v in vals))

# %%
# Available distributions
print("\nBuilt-in distributions:")
print(f"  {'Distribution':<15s} {'Usage'}")
print("  " + "-" * 60)
print(f"  {'Uniform':<15s} Most parameters — flat between lo and hi")
print(f"  {'Gaussian':<15s} Informative prior (mean, std)")
print(f"  {'LogUniform':<15s} Scale parameters (e.g., timescales)")
print(f"  {'LogNormal':<15s} Positive quantities with log-space spread")
print(f"  {'StudentT':<15s} Heavy-tailed prior (robust to outliers)")
print(f"  {'Fixed':<15s} Held constant during inference")

# %% [markdown]
# ## SEDModel: The Differentiable Forward Model
#
# The SEDModel wraps the full SPS pipeline — from parameters through SFH, CSP
# integration, dust, and filter convolution — into a single differentiable
# function.

# %%
model = SEDModel(spec, ssp_data, observation=obs)
print(f"Model created: {spec.n_free} free parameters")

# %%
# Predict a spectrum
WAVE_OBS = jnp.linspace(3800.0, 9200.0, 200)
model.precompute_spectroscopy(WAVE_OBS)

params = spec.sample(jax.random.PRNGKey(42))
spectrum = model.predict_spectrum(params)
print(f"Spectrum shape: {spectrum.shape}")

# %%
# Predict photometry
phot = model.predict_photometry(params)
print(f"Photometry ({len(phot)} bands): {np.array(phot)}")

# %%
# Precomputation speedup
# Spectroscopy precomputation is already active.
# Photometry precomputation activates automatically when redshift is fixed.
# Let's measure the speedup:
jit_spec = jax.jit(model.predict_spectrum)
_ = jit_spec(params)  # compile

n = 1000
t0 = time.perf_counter()
for _ in range(n):
    r = jit_spec(params)
    r.block_until_ready()
t_precomp = (time.perf_counter() - t0) / n * 1e6
print(f"Predict spectrum (precomputed): {t_precomp:.0f} µs")

# %%
# JIT compilation
jit_phot = jax.jit(model.predict_photometry)
_ = jit_phot(params)  # compile

t0 = time.perf_counter()
for _ in range(n):
    r = jit_phot(params)
    r.block_until_ready()
t_phot = (time.perf_counter() - t0) / n * 1e6
print(f"Predict photometry (JIT): {t_phot:.0f} µs")

# %%
# vmap: Batch prediction
batch_predict = jax.jit(jax.vmap(model.predict_photometry))
batch_times = {}

for n_batch in [10, 100, 1000]:
    keys = jax.random.split(jax.random.PRNGKey(0), n_batch)
    batch_params = jax.vmap(spec.sample)(keys)
    _ = batch_predict(batch_params)  # compile

    t0 = time.perf_counter()
    for _ in range(50):
        r = batch_predict(batch_params)
        r.block_until_ready()
    t = (time.perf_counter() - t0) / 50 * 1e3
    batch_times[n_batch] = t
    print(f"  vmap({n_batch:>4d} galaxies): {t:.1f} ms  ({t/n_batch*1e3:.1f} µs/galaxy)")

# %%
# --- FIGURE 1: vmap scaling ---
ns = np.array(list(batch_times.keys()))
ts = np.array(list(batch_times.values()))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(ns, ts, s=60, color=COLORS["geovi"], zorder=3)
ax.plot(ns, ts, color=COLORS["geovi"], lw=1)

# Linear reference
linear = ts[0] / ns[0] * ns
ax.plot(ns, linear, ls="--", color="gray", label="Linear scaling")

ax.set_xlabel("Number of galaxies")
ax.set_ylabel("Wall time [ms]")
ax.set_xscale("log")
ax.set_yscale("log")
ax.legend()
ax.set_title("vmap Scaling: Sublinear via XLA Parallelism")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "fig01_vmap_scaling.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# Gradients for free
grad_fn = jax.jit(jax.grad(lambda p: jnp.sum(model.predict_spectrum(p))))
_ = grad_fn(params)

t0 = time.perf_counter()
for _ in range(n):
    g = grad_fn(params)
    jax.tree.map(lambda x: x.block_until_ready(), g)
t_grad = (time.perf_counter() - t0) / n * 1e6
print(f"Gradient:  {t_grad:.0f} µs  ({t_grad/t_precomp:.1f}× forward pass)")

# Full Jacobian (returns a dict of arrays: one per parameter)
J = jax.jacobian(model.predict_spectrum)(params)
first_key = next(iter(J.keys()))
print(f"Jacobian: {len(J)} parameter entries, first shape: {J[first_key].shape}")
print("Gradients cost ~1.5× the forward pass. This is what makes geoVI and RT possible.")

# %%
# SFH and derived quantities
sfh = model.predict_sfh(params)
print(f"\nSFH keys: {list(sfh.keys())}")
print(f"  t_gyr shape: {sfh['t_gyr'].shape}")
print(f"  sfr_mean shape: {sfh['sfr_mean'].shape}")
print(f"  sfr_full shape: {sfh['sfr_full'].shape}")

# %% [markdown]
# ## Fitter: Inference Engine
#
# The Fitter connects a SEDModel to observed data and runs inference. It builds
# the loss function, handles standardization, and dispatches to any of the
# 5+ inference backends.

# %%
# Generate mock data
mock = model.mock_spectrum(params, WAVE_OBS, snr=30.0, key=jax.random.PRNGKey(99))
fitter = Fitter(model, mock.flux_obs, mock.noise)
print(f"Fitter: {len(mock.flux_obs)} data points, {spec.n_free} parameters")

# %%
# The loss function (information Hamiltonian)
# H(ξ) = ½χ² + ½ξᵀξ — every inference method minimizes or samples from this.
# The Fitter builds this internally. All we need to do is call .run():
print("H(ξ) = ½χ² + ½ξᵀξ — every method minimizes or samples from this.")
print(f"  ξ dimension: {spec.n_free}")
print(f"  Data dimension: {len(mock.flux_obs)}")

# %%
# MAP — point estimate
result_map = fitter.run("map", n_steps=500, verbose=False)
print(f"MAP: {result_map.wall_time_s:.1f}s, final loss = {float(result_map.loss_history[-1]):.1f}")

# %%
# native_geovi — approximate posterior
result_geovi = fitter.run(
    "native_geovi", n_iterations=15, n_samples=6, n_seeds=3,
    n_posterior_samples=2000, verbose=False,
)
print(f"native_geovi: {result_geovi.wall_time_s:.1f}s, {len(next(iter(result_geovi.samples.values())))} samples")

# %%
# Ray Tracing — exact MCMC
result_rt = fitter.run(
    "raytrace", init_from=result_map,
    n_burnin=100, n_steps=500, verbose=False,
)
acc = result_rt.diagnostics.get("acceptance_rate", float("nan"))
print(f"Ray Tracing: {result_rt.wall_time_s:.1f}s, acceptance = {acc:.1%}")

# %%
# Switching methods is trivial
print("\nSwitching methods is one line:")
print('  fitter.run("native_geovi")  # approximate, fast')
print('  fitter.run("raytrace")      # exact, any D')
print('  fitter.run("nuts")          # exact, D < 20')
print("\nSame model, same data, same loss — different exploration strategy.")

# %% [markdown]
# ## Posterior: Working with Results
#
# The result object stores samples, diagnostics, and timing. It provides
# methods for summary statistics, derived quantities, and plotting.

# %%
# Inspect the Posterior
result = result_geovi
print(f"Method:    {result.method}")
print(f"Wall time: {result.wall_time_s:.1f}s")
print(f"Samples:   {len(next(iter(result.samples.values())))} draws")

# Summary: median ± 68% CI for each parameter
summary = result.summary()
print(f"\n{'Parameter':<35s} {'Median':>10s} {'68% CI'}")
for name, info in summary.items():
    if "median" in info:
        med = info["median"]
        lo, hi = info.get("lo_68", float("nan")), info.get("hi_68", float("nan"))
        print(f"  {name:<33s} {med:>10.4f}  [{lo:.4f}, {hi:.4f}]")

# %%
# summary_table() gives a nicely formatted overview
print(result.summary_table())

# %%
# SFH from posterior
# We can predict the SFH for each posterior sample
n_sfh_draws = 50
sfh_samples = []
for i in range(n_sfh_draws):
    idx = i % len(result.samples[spec.free_params[0]])
    draw = {k: v[idx] for k, v in result.samples.items()}
    sfh_i = model.predict_sfh(draw)
    sfh_samples.append(np.array(sfh_i["sfr_full" if "sfr_full" in sfh_i else "sfr_mean"]))

sfh_arr = np.array(sfh_samples)
t_gyr = np.array(model.predict_sfh(params)["t_gyr"])
print(f"SFH posterior: {sfh_arr.shape[0]} draws × {sfh_arr.shape[1]} time points")

# %%
# Convergence diagnostics
if hasattr(result, "effective_sample_size"):
    try:
        ess = result.effective_sample_size()
        print("\nEffective Sample Size:")
        for name, val in ess.items():
            print(f"  {name:<35s}: {float(val):.0f}")
    except Exception:
        print("(ESS not available for this method)")

# %% [markdown]
# ## The JAX Mental Model
#
# tengri is built on JAX. Three things to know:
#
# 1. **jit** — compiles functions. Slow first call (tracing), fast after.
# 2. **vmap** — vectorizes over a batch dimension. Use it for catalogs.
# 3. **grad** — automatic differentiation. This is why native_geovi works:
#    gradients flow through the entire SPS pipeline for free.
#
# The forward model runs in ~40 µs JIT'd. Gradients cost ~1.5× that. Batch
# processing is sublinear via XLA parallelism.

# %% [markdown]
# ## Summary
#
# You now know the three core objects and three JAX patterns:
#
# | Object | Purpose | Key methods |
# |--------|---------|-------------|
# | `Parameters` | Define parameters and priors | `.sample()`, `.standardize()` |
# | `SEDModel` | Forward model | `.predict_spectrum()`, `.predict_sfh()`, `.mock()` |
# | `Fitter` | Inference engine | `.run("native_geovi")`, `.run("raytrace")` |
# | `Posterior` | Results container | `.samples`, `.summary()`, `.diagnostics` |
#
# Next: **tutorials/03** explains the IFT/PSD/GP machinery that generates
# bursty SFHs.
