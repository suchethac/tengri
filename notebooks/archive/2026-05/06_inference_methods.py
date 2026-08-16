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
# # Choosing an Inference Method
#
# **What you'll learn:**
# - MAP (mode) vs Laplace (quadratic approximation) vs Pathfinder (approximate posterior)
# - NUTS (exact MCMC) vs Ray Tracing (exact HMC variant) vs NSS (nested sampling, evidence)
# - Decision table: pick method based on problem size, speed/accuracy tradeoff
# - All methods share identical objective: the information Hamiltonian
#
# **Prerequisites:** [`00_quickstart.py`](00_quickstart.py).
# **Next:** Real-data workflows in [`03_fitting_photometry.py`](03_fitting_photometry.py).
#
# ---
#
# Six canonical inference methods on the same smooth 7-D star formation history.
# See where they agree on posterior peaks, and learn when to pick each method.
# Every method minimizes the same **information Hamiltonian**:
# $H(\boldsymbol{\xi}) = \frac{1}{2}\chi^2 + \frac{1}{2}\boldsymbol{\xi}^\top\boldsymbol{\xi}$.

# %% [markdown]
# **Spine location:** `notebooks/06_inference_methods.py`

# %%
import os
import sys
import time
import warnings

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

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fitter,
    Fixed,
    SEDModel,
    Observation,
    Parameters,
    Photometry,
    Uniform,
    load_ssp_data,
)

import importlib.util

_repo_data_root = None
_spec_tengri = importlib.util.find_spec("tengri")
if _spec_tengri is not None and _spec_tengri.origin:
    _walk = os.path.dirname(os.path.abspath(_spec_tengri.origin))
    for _step in range(12):
        _candidate = os.path.join(_walk, "notebooks", "_plot_style.py")
        if os.path.isfile(_candidate):
            sys.path.insert(0, os.path.dirname(_candidate))
            _repo_data_root = os.path.dirname(os.path.dirname(os.path.abspath(_candidate)))
            break
        _parent_walk = os.path.dirname(_walk)
        if _parent_walk == _walk:
            break
        _walk = _parent_walk

if _repo_data_root is None:
    _np_here = os.path.abspath(os.getcwd())
    while True:
        if os.path.isfile(os.path.join(_np_here, "_plot_style.py")):
            sys.path.insert(0, _np_here)
            _repo_data_root = os.path.dirname(_np_here)
            break
        _ppt = os.path.join(_np_here, "notebooks", "_plot_style.py")
        if os.path.isfile(_ppt):
            _nbsd = os.path.dirname(_ppt)
            sys.path.insert(0, _nbsd)
            _repo_data_root = os.path.dirname(_nbsd)
            break
        _parent_here = os.path.dirname(_np_here)
        if _parent_here == _np_here:
            break
        _np_here = _parent_here

if _repo_data_root is not None and os.path.isdir(os.path.join(_repo_data_root, "data")):
    os.chdir(_repo_data_root)
elif os.path.isdir(os.path.join(_repo_root, "data")):
    os.chdir(_repo_root)
elif os.path.isdir("data"):
    pass
elif os.path.isdir(os.path.join("..", "data")):
    os.chdir("..")

FIGDIR = os.path.join("notebooks", "figures", "inference_methods")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import (
    COLORS,
    plot_corner_comparison,
    plot_sfh,
    setup_style,
)

setup_style()

# %%
import tengri as tg

tg.print_logo()
print(f"tengri {tg.__version__}")

# %% [markdown]
# ## Part 0: Shared Setup
#
# We build **one SEDModel** with a smooth 7-D parametrization, generate
# mock photometry, and create a `Fitter`. All methods run on this same problem
# so comparisons are apples-to-apples.

# %%
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

# Multi-wavelength filter set: optical to NIR
_candidate_filters = [
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "twomass_j",
    "twomass_h",
    "twomass_ks",
]

phot_bands_list = []
for band in _candidate_filters:
    try:
        test_phot = Photometry.from_names([band])
        phot_bands_list.append(band)
    except Exception:
        pass

if not phot_bands_list:
    phot_bands_list = ["twomass_j", "twomass_h", "twomass_ks"]

phot_obs = Photometry.from_names(phot_bands_list, cache_dir="data/filters")
obs = Observation(photometry=phot_obs)

print(f"SSP templates: {ssp_data.ssp_flux.shape}")
print(f"Photometric bands ({phot_obs.n_filters}): {', '.join(phot_obs.names)}")

# %%
# Define smooth 7-D parametrization
spec_param = Parameters(
    sfh_db_log_total_mass=Uniform(8, 12),
    sfh_db_log_sfr_inst=Uniform(-2, 3),
    sfh_db_tx_frac_0=Uniform(0.05, 0.95),
    sfh_db_tx_frac_1=Uniform(0.05, 0.95),
    sfh_db_tx_frac_2=Uniform(0.05, 0.95),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type="dense_basis",
)

model_param = SEDModel(spec_param, ssp_data, observation=obs)
print(f"\nModel: {spec_param.n_free} free parameters")
for name in spec_param.free_params:
    print(f"  {name}")

# %%
# Generate mock photometry: monotonically rising SFH
key = jax.random.PRNGKey(42)
true_params_param = spec_param.sample(key)
true_params_param = {**true_params_param}
true_params_param["sfh_db_log_total_mass"] = jnp.array(10.8)
true_params_param["sfh_db_log_sfr_inst"] = jnp.array(1.48)  # log(30) ≈ 1.48 Msun/yr
true_params_param["sfh_db_tx_frac_0"] = jnp.array(0.1)
true_params_param["sfh_db_tx_frac_1"] = jnp.array(0.25)
true_params_param["sfh_db_tx_frac_2"] = jnp.array(0.65)
true_params_param["met_logzsol"] = jnp.array(-0.1)
true_params_param["dust_tau_bc"] = jnp.array(0.5)
true_params_param["dust_tau_diff"] = jnp.array(0.3)

mock_param = model_param.mock(true_params_param, snr=50.0, key=key)

print("\nTrue parameters (monotonically rising SFH, SFR_inst = 30 Msun/yr):")
for name in spec_param.free_params:
    print(f"  {name:30s} = {float(true_params_param[name]):.4f}")

# %%
# Create fitter (shared across all methods)
os.environ["TENGRI_NO_BACKGROUND_COMPILE"] = "1"
fitter_param = Fitter(
    model_param,
    mock_param.flux_obs,
    mock_param.noise,
)

print("\n" + "=" * 70)
print("Running inference methods on 7-D smooth SFH problem...")
print("=" * 70)

# %% [markdown]
# ## Method 1: MAP (Maximum A Posteriori)
#
# **Gradient descent to find the mode.** Fast, no uncertainties.

# %%
t0 = time.perf_counter()
result_map = fitter_param.run("map", n_steps=1000, verbose=False)
t_map = time.perf_counter() - t0
print(f"\nMAP:  {t_map:.2f}s  (point estimate only)")

# %% [markdown]
# ## Method 2: Laplace Approximation
#
# **Gaussian posterior from the Hessian at MAP.** Fast, captures
# local geometry but may miss non-Gaussianity.

# %%
t0 = time.perf_counter()
try:
    result_laplace = fitter_param.run("laplace", init_from=result_map, verbose=False)
    t_laplace = time.perf_counter() - t0
    print(f"Laplace:  {t_laplace:.2f}s  (Gaussian approximation from Hessian)")
except Exception as e:
    result_laplace = None
    t_laplace = None
    print(f"Laplace:  Failed ({type(e).__name__})")

# %% [markdown]
# ## Method 3: Pathfinder
#
# **Variational approximate posterior from optimization.** Fast, good
# warm-start for NUTS. Parametric (Gaussian), but nonlinear geometry
# captured via optimization.

# %%
t0 = time.perf_counter()
try:
    result_pathfinder = fitter_param.run(
        "pathfinder",
        init_from=result_map,
        n_iterations=50,
        verbose=False,
    )
    t_pathfinder = time.perf_counter() - t0
    print(f"Pathfinder:  {t_pathfinder:.2f}s  (variational initialization)")
except Exception as e:
    result_pathfinder = None
    t_pathfinder = None
    print(f"Pathfinder:  Failed ({type(e).__name__})")

# %% [markdown]
# ## Method 4: NUTS (No-U-Turn Sampler)
#
# **Exact HMC sampler, gold standard for low D.** Slow but unbiased.

# %%
t0 = time.perf_counter()
try:
    result_nuts = fitter_param.run(
        "mcmc_nuts",
        init_from=result_map,
        n_warmup=500,
        n_samples=1000,
        verbose=False,
    )
    t_nuts = time.perf_counter() - t0
    print(f"NUTS:  {t_nuts:.2f}s  (gold standard, exact sampler)")
except Exception as e:
    result_nuts = None
    t_nuts = None
    print(f"NUTS:  Failed ({type(e).__name__})")

# %% [markdown]
# ## Method 5: Ray Tracing
#
# **Elliptical slice sampler via optical ray geometry.** Noise-tolerant,
# scales to high D, exact.

# %%
t0 = time.perf_counter()
try:
    result_raytrace = fitter_param.run(
        "mcmc_raytrace",
        init_from=result_map,
        n_warmup=200,
        n_samples=1000,
        step_size=0.05,
        verbose=False,
    )
    t_raytrace = time.perf_counter() - t0
    print(f"Ray Tracing:  {t_raytrace:.2f}s  (optical ESS, noise-tolerant)")
except Exception as e:
    result_raytrace = None
    t_raytrace = None
    print(f"Ray Tracing:  Failed ({type(e).__name__})")

# %% [markdown]
# ## Method 6: Evidence (Nested Sampling Sampler)
#
# **Exact sampler optimized for Bayesian evidence and model comparison.**
# Provides log marginal likelihood for model selection.

# %%
t0 = time.perf_counter()
try:
    result_nss = fitter_param.run(
        "evidence",
        n_live=150,
        n_posterior_samples=500,
        verbose=False,
    )
    t_nss = time.perf_counter() - t0
    print(f"NSS (Evidence):  {t_nss:.2f}s  (nested sampler, model selection)")
except Exception as e:
    result_nss = None
    t_nss = None
    print(f"NSS:  Failed ({type(e).__name__})")

# %% [markdown]
# ## Method 7: Other BlackJAX MCMC variants
#
# In addition to NUTS, the fitter exposes a family of related MCMC samplers.
# These all target the same posterior; differences are in proposal mechanics,
# step-size adaptation, and high-D scaling. Useful when NUTS is the wrong
# trade-off for a given problem (e.g. very high D, expensive likelihoods).
#
# - **HMC**: vanilla Hamiltonian Monte Carlo (fixed integration length).
# - **Dynamic HMC**: adaptive trajectory length without NUTS' tree doubling.
# - **GHMC**: generalised HMC; partial momentum refreshment for better
#   exploration under correlated targets.
# - **MCLMC**: Microcanonical Langevin Monte Carlo; recent unadjusted
#   sampler that scales well to large D.
# - **Elliptical Slice**: exact slice sampler under a Gaussian prior — the
#   classic ESS, distinct from this code's ``mcmc_raytrace`` variant.

# %%
mcmc_variants = (
    ("HMC", "mcmc_hmc"),
    ("Dynamic HMC", "mcmc_dynamic_hmc"),
    ("GHMC", "mcmc_ghmc"),
    ("MCLMC", "mcmc_mclmc"),
    ("Elliptical Slice", "mcmc_ess"),
)

variant_results: dict[str, object] = {}
variant_times: dict[str, float] = {}

for label, method_name in mcmc_variants:
    t0 = time.perf_counter()
    try:
        res = fitter_param.run(
            method_name,
            init_from=result_map,
            n_warmup=200,
            n_samples=500,
            verbose=False,
        )
        dt = time.perf_counter() - t0
        variant_results[label] = res
        variant_times[label] = dt
        print(f"{label:<18} {dt:>8.2f}s")
    except Exception as e:
        variant_results[label] = None
        variant_times[label] = None
        print(f"{label:<18} Failed ({type(e).__name__}: {e})")

# %% [markdown]
# ## Comparison: Posterior Agreement

# %%
# Gather successful sampling results
sampling_results = {}
sampling_methods = {}

if result_nuts is not None:
    sampling_results["NUTS"] = result_nuts
    sampling_methods["NUTS"] = t_nuts

if result_raytrace is not None:
    sampling_results["Ray Tracing"] = result_raytrace
    sampling_methods["Ray Tracing"] = t_raytrace

if result_nss is not None:
    sampling_results["NSS"] = result_nss
    sampling_methods["NSS"] = t_nss

for label, res in variant_results.items():
    if res is not None:
        sampling_results[label] = res
        sampling_methods[label] = variant_times[label]

if sampling_results:
    # Corner plot overlay
    fig = None
    colors_overlay = {
        "NUTS": COLORS.get("mcmc_nuts", "C0"),
        "Ray Tracing": COLORS.get("mcmc_raytrace", "C1"),
        "NSS": COLORS.get("evidence", "C2"),
    }

    for method_name, result in sampling_results.items():
        if fig is None:
            fig = plot_corner_comparison(
                [result],
                labels=[method_name],
                colors=[colors_overlay.get(method_name, "C0")],
                truths=true_params_param,
            )
        else:
            # Note: may need custom overlay; for now just show the first
            pass

    if fig is not None:
        fig.suptitle(
            f"Posterior Comparison ({', '.join(sampling_results.keys())}, D=7)",
            fontsize=12,
            y=1.00,
        )
        plt.tight_layout()
        plt.show()

# %%
# SFH recovery comparison
if sampling_results:
    n_methods = len(sampling_results)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 3.5), sharey=True)
    if n_methods == 1:
        axes = [axes]

    for ax, (method_name, result) in zip(axes, sampling_results.items()):
        color = COLORS.get(f"mcmc_{method_name.lower()}", "C0")
        plot_sfh(
            model_param,
            result,
            true_params=true_params_param,
            ax=ax,
            color=color,
            label=method_name,
            method=method_name,
        )
        ax.set_title(f"{method_name} ({sampling_methods[method_name]:.1f}s)")

    axes[0].set_ylabel(r"SFR [$M_\odot$/yr]")
    fig.suptitle("SFH Recovery: Comparison of Sampling Methods (D=7)", fontsize=13, y=1.00)
    plt.tight_layout()
    plt.show()

# %%
# Wall time summary
print("\n" + "=" * 70)
print("Wall Time Summary (7-D smooth SFH)")
print("=" * 70)
print(f"{'Method':<20} {'Runtime (s)':>12} {'Type':>20}")
print("-" * 70)
print(f"{'MAP':<20} {t_map:>12.2f} {'Point estimate':>20}")

if t_laplace is not None:
    print(f"{'Laplace':<20} {t_laplace:>12.2f} {'Gaussian approx':>20}")
else:
    print(f"{'Laplace':<20} {'failed':>12} {'(N/A)':>20}")

if t_pathfinder is not None:
    print(f"{'Pathfinder':<20} {t_pathfinder:>12.2f} {'Variational init':>20}")
else:
    print(f"{'Pathfinder':<20} {'failed':>12} {'(N/A)':>20}")

if t_nuts is not None:
    print(f"{'NUTS':<20} {t_nuts:>12.2f} {'Exact HMC':>20}")
else:
    print(f"{'NUTS':<20} {'failed':>12} {'(N/A)':>20}")

if t_raytrace is not None:
    print(f"{'Ray Tracing':<20} {t_raytrace:>12.2f} {'Exact optics':>20}")
else:
    print(f"{'Ray Tracing':<20} {'failed':>12} {'(N/A)':>20}")

if t_nss is not None:
    print(f"{'NSS (Evidence)':<20} {t_nss:>12.2f} {'Exact nested':>20}")
else:
    print(f"{'NSS':<20} {'failed':>12} {'(N/A)':>20}")

for label, dt in variant_times.items():
    if dt is not None:
        print(f"{label:<20} {dt:>12.2f} {'BlackJAX variant':>20}")
    else:
        print(f"{label:<20} {'failed':>12} {'(N/A)':>20}")

print("=" * 70)

# %% [markdown]
# ## Decision Table: Pick X When Your Problem is Y
#
# | **Method** | **Best For** | **Wall Time** | **Exact** | **Limit** |
# |:-----------|:-------------|:------------:|:---------:|:---------:|
# | **MAP** | Initialization, catalogs | seconds | ✗ | Always |
# | **Laplace** | Quick uncertainty estimate | seconds | ✗ (Gaussian) | D ≲ 20 |
# | **Pathfinder** | NUTS warm-start, variational | seconds | ✗ | D ≲ 50 |
# | **NUTS** | Validation, exact gold standard | minutes | ✓ | D ≲ 30 |
# | **Ray Tracing** | Default workhorse | minutes | ✓ | D ≲ 200 |
# | **NSS (Evidence)** | Model comparison, Bayes factors | minutes | ✓ | D ≲ 50 |
#
# ### Quick Algorithm
#
# 1. **Always start with MAP** — fast initialization for all samplers.
# 2. **For parametric models (D ≲ 10):**
#    - Use **NUTS** for validation (gold standard).
#    - Then run **Ray Tracing** or **Laplace** for production.
# 3. **For stochastic models (D ≈ 20–200):**
#    - Skip NUTS, use **Ray Tracing** as primary sampler.
#    - Cross-check with **Pathfinder** for speed.
# 4. **For model comparison:**
#    - Use **NSS** to compute Bayesian evidence and Bayes factors.
# 5. **For very quick uncertainty (exploratory):**
#    - Use **Laplace** or **Pathfinder** to get error bars in seconds.

# %% [markdown]
# ## Summary
#
# - The **information Hamiltonian** $H = \frac{1}{2}\chi^2 + \frac{1}{2}\xi^\top\xi$
#   is the same objective for all methods.
# - **MAP** is fast but gives no uncertainties — essential for initialization.
# - **Laplace** and **Pathfinder** are quick variational approximations; use when
#   speed matters.
# - **NUTS** is the exact gold standard for D ≲ 30; slower but unbiased.
# - **Ray Tracing** is the workhorse for D > 20, noise-tolerant, scales well.
# - **NSS** is specialized for Bayesian evidence; required for model selection.
#
# For stochastic (bursty) SFH models at very high D (D ≈ 137), see
# `15_vi_inference.py` (Paper II methods).

# %% [markdown]
# ## What You've Learned
#
# 1. Six canonical inference methods on the same 7-D problem.
# 2. How to interpret wall times and choose by problem size.
# 3. When each method is appropriate (Table above).
# 4. That MAP → NUTS/Ray Tracing/NSS is the standard workflow.
#
# **Next:** `07_fitting_photometry.py` (real photometric inference)
# or `08_fitting_spectra.py` (spectroscopy).
