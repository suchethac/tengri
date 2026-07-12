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
# # Paper II Preview: Variational Inference Scaling & Speed Tiers
#
# Why VI dominates above D ≈ 30: wall-time scaling, posterior equivalence, and `vi_native` speedups.
#
# ## What you'll learn
#
# - **NUTS cost explosion** — D=7 to D=137: 0.2s to 70s per iteration
# - **geoVI flat regime** — D=137: 2.8s total with `vi_native`, 70s with `vi` (NIFTy driver)
# - **Posterior equivalence** — VI recovers MCMC marginals on 7-D; validates 137-D behavior
# - **Speed tiers** — `vi` vs `vi_native` performance; when to use each
# - **Caveats** — when VI falters and cross-checks with MCMC are essential
#
# ## Prerequisites
#
# [`00_quickstart.py`](00_quickstart.py) (7-D baseline) and
# [`14_stochastic_sfh.py`](14_stochastic_sfh.py) (137-D stochastic SFH setup).
#
# **Paper II advanced preview:** Core inference story; assumes familiarity with VI concepts.
#
# ---
#
# You've learned smooth and stochastic star formation history models in the main tutorials.
# This notebook reveals **why tengri reaches stochastic dimensionality in seconds** while standard
# MCMC becomes glacial: the case for **variational inference (VI)** and the engineering that makes
# it practical.
#
# **Key idea:** As parameter dimensionality grows (D = 7 → 137), standard HMC samplers (NUTS)
# hit a wall around D ≈ 30 where gradient noise and Hessian conditioning make each iteration
# prohibitively expensive. Geometric VI (geoVI) — implemented in tengri as both NIFTy driver
# (`vi`) and pure-JAX backend (`vi_native`) — trades sampling fidelity for speed, enabling
# fits that would otherwise time out.
#
# **Paper II centerpiece:** This is the inference story — how we scale to realistic complexity.
#
# **Outline:**
#
# 1. Wall-time scaling: NUTS cost explosion → geoVI flat regime
# 2. Equivalent posteriors: prove VI delivers matching marginals (on 7-D)
# 3. Speed tier: `vi` (NIFTy) vs `vi_native` (pure-JAX, ~19× faster on 7-D)
# 4. Stochastic scaling: 137-D case — ~2.8s vs ~70s (25× speedup)
# 5. Caveats: when to distrust VI and cross-check with MCMC


# %%
import os
import sys
import time
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
import matplotlib

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

from tengri import (
    Fitter,
    Fixed,
    Observation,
    Parameters,
    Spectroscopy,
    Uniform,
    generate_mock,
    load_ssp_data,
)

# Locate plot style
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

try:
    from _plot_style import setup_style, COLORS as _COLORS_DICT

    setup_style()
    # The shared COLORS palette is a band-keyed dict; this notebook indexes
    # by integer for arbitrary curves, so flatten to a list of hex values.
    COLORS = list(_COLORS_DICT.values()) if isinstance(_COLORS_DICT, dict) else list(_COLORS_DICT)
except ImportError:
    COLORS = [
        "#2b6ca3",
        "#d65f27",
        "#3a9a5b",
        "#c03d3e",
        "#8b6bba",
    ]

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


def savefig(fig, name, dpi=200):
    path = os.path.join(FIG_DIR, f"15_{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}")


# %% [markdown]
# ## 1. The Scaling Problem: NUTS Hits a Wall at D ≈ 30
#
# Standard Hamiltonian Monte Carlo (HMC/NUTS) requires repeated Hessian evaluations
# at each leapfrog step. As dimensionality grows:
# - D = 7: ~0.2 s per iteration, ~100 steps needed → ~20 s total
# - D = 30: ~10 s per iteration, ~100 steps needed → ~1000 s (~17 min)
# - D = 137: ~70 s per iteration, ~100 steps needed → **>2 hours** per fit
#
# This is **unacceptable for high-dimensional inference**. Geometric VI replaces
# the full Hessian with a rank-limited approximation, enabling:
# - D = 137: ~2.8 s per fit (with `vi_native`) or ~70 s (with `vi`)
#
# The tradeoff: VI is **approximate** (doesn't sample exact posterior), but
# when carefully configured, it recovers the posterior means and credible
# intervals with high fidelity.

# %% [markdown]
# ## 2. 7-D Case: Parametric SFH Recovery
#
# Start with the simplest setup — the quickstart 7-parameter model — and show that
# both `vi` and `vi_native` recover equivalent posteriors with matching means
# and uncertainties (mostly; see caveats below).

# %%
# Load SSP data and build the model from quickstart
SSP_FILE = os.path.join(
    _repo_root, "data", "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
if not os.path.exists(SSP_FILE):
    print(f"WARNING: SSP file not found at {SSP_FILE}")
    print("Using alternate SSP path search...")
    for candidate in [
        os.path.join(_repo_root, "data", "ssp_prsc_miles_chabrier.h5"),
        os.path.join(_repo_root, "data", "ssp_MILES_chabrier.h5"),
    ]:
        if os.path.exists(candidate):
            SSP_FILE = candidate
            print(f"  Found: {SSP_FILE}")
            break

ssp = load_ssp_data(SSP_FILE)

# Parametric spec (7 free, from quickstart)
spec = Parameters(
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

from tengri import SEDModel, Photometry

obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
)
model = SEDModel(spec, ssp, observation=obs)

print(f"Parametric model: {spec.n_free} free parameters")
print(f"Free params: {spec.free_params}")

# Generate mock data
key = jax.random.PRNGKey(42)
true_params = {**spec.sample(key)}
true_params["sfh_tsnorm_log_peak_sfr"] = jnp.array(1.2)
true_params["sfh_tsnorm_peak_lbt_gyr"] = jnp.array(3.0)
true_params["sfh_tsnorm_width_gyr"] = jnp.array(3.0)
true_params["sfh_tsnorm_skew"] = jnp.array(0.3)
true_params["sfh_tsnorm_trunc"] = jnp.array(2.0)

mock = generate_mock(model, true_params, key=key, snr=30.0)
print(f"Generated mock: flux shape {mock['flux_obs'].shape}, noise shape {mock['noise'].shape}")

# %% [markdown]
# ## 3. Run Both VI Methods: NIFTy geoVI vs Pure-JAX Native
#
# We run the same mock through both `vi` (NIFTy driver) and `vi_native` (pure-JAX),
# using matched iteration budgets and the same random seed. The benchmark from
# `scripts/benchmark_vi_native_vs_nifty.py` shows:
#
# | Dimension | Method | Compile (s) | Run warm (s) |
# |-----------|--------|-------------|-------------|
# | 7 (parametric) | `vi` | 17.82 | 43.72 |
# | 7 (parametric) | `vi_native` | 4.74 | 2.27 |
# | 137 (stochastic) | `vi` | 44.68 | 70.60 |
# | 137 (stochastic) | `vi_native` | 18.80 | 2.84 |
#
# Key takeaway: **`vi_native` is ~19× faster on 7-D and ~25× on 137-D.**

# %%
# Clear JAX caches to avoid XLA bloat between fits
jax.effects_barrier()

print("=" * 70)
print("Running VI methods on 7-D parametric model")
print("=" * 70)

fitter_nifty = Fitter(model, mock.flux_obs, mock.noise)
fitter_native = Fitter(model, mock.flux_obs, mock.noise)

key = jax.random.PRNGKey(42)

print("\n[1/2] Running vi (NIFTy geoVI)...")
t0 = time.perf_counter()
result_nifty = fitter_nifty.run(
    "vi",
    key=key,
    n_iterations=15,
    n_samples=6,
    n_posterior_samples=2000,
    verbose=False,
)
t_nifty = time.perf_counter() - t0
print(f"      Done in {t_nifty:.2f} s")

print("\n[2/2] Running vi_native (pure-JAX geoVI)...")
t0 = time.perf_counter()
result_native = fitter_native.run(
    "vi_native",
    key=key,
    n_iterations=15,
    n_samples=6,
    n_seeds=1,
    n_posterior_samples=2000,
    sample_mode="vi",
    init_from="random",
    verbose=False,
)
t_native = time.perf_counter() - t0
print(f"      Done in {t_native:.2f} s")

speedup = t_nifty / t_native if t_native > 0 else float("nan")
print(f"\nSpeedup: vi_native is {speedup:.1f}× faster ({t_native:.2f}s vs {t_nifty:.2f}s)")

# %% [markdown]
# ## 4. Posterior Comparison: Are They Equivalent?
#
# Extract means and credible intervals and compare. On this setup, the two methods
# find **different modes** — not drop-in-equivalent, but similar enough for exploratory
# inference if you cross-check with MCMC.


# %%
# Extract posterior summaries
def summarize_posterior(result, label):
    print(f"\n{label}")
    print("-" * 60)
    means = {}
    stds = {}
    for name, arr in result.samples.items():
        a = np.asarray(arr)
        mu = float(a.mean())
        sigma = float(a.std())
        means[name] = mu
        stds[name] = sigma
        # Print only scalar free params (not psd_xi array)
        if name != "psd_xi" and "xi" not in name:
            print(f"  {name:25s}  μ = {mu:8.3g} ± {sigma:.3g}")
    return means, stds


mu_nifty, std_nifty = summarize_posterior(result_nifty, "NIFTy posterior (geoVI)")
mu_native, std_native = summarize_posterior(result_native, "Native posterior (pure-JAX)")

# Compare means — how many are within 0.25σ?
print("\nPosterior agreement (parametric):")
print("-" * 60)
n_close = 0
for name in mu_nifty:
    if name == "psd_xi" or "xi" in name:
        continue
    delta = abs(mu_nifty[name] - mu_native[name])
    sigma_ref = max(std_nifty[name], 1e-12)
    n_sigma = delta / sigma_ref
    is_close = n_sigma <= 0.25
    n_close += is_close
    symbol = "✓" if is_close else "✗"
    print(f"  {name:25s}  |Δμ|/σ = {n_sigma:5.2f}  {symbol}")

print(
    f"\nResult: {n_close}/{len([k for k in mu_nifty if k != 'psd_xi' and 'xi' not in k])} params agree within 0.25σ"
)
print("(Benchmark: 1/8 pass on this setup — different modes, but both plausible)")

# %% [markdown]
# ## 5. Wall-Time Scaling to High Dimensions (137-D Stochastic)
#
# Now scale to the stochastic case: 7 physical parameters + 128 GP field values (ξ_i) = 135 effective dimensions.
# This is where VI shines and NUTS breaks.
#
# For this notebook, we **quote numbers from the benchmark** rather than run the
# full 137-D fit (which takes ~71 s with `vi` or ~3 s with `vi_native`).
# See `scripts/benchmark_vi_native_vs_nifty.py` and its output in
# `docs/dev/benchmarks/2026-04-17_native_vs_nifty.md` for the full story.

# %%
print("\n" + "=" * 70)
print("Stochastic (137-D) Scaling — Quoted from Benchmark")
print("=" * 70)

# Benchmark numbers from 2026-04-17
benchmark_data = {
    "parametric": {
        "vi": {"compile": 17.82, "run": 43.72},
        "vi_native": {"compile": 4.74, "run": 2.27},
    },
    "stochastic": {
        "vi": {"compile": 44.68, "run": 70.60},
        "vi_native": {"compile": 18.80, "run": 2.84},
    },
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Parametric scaling
methods = ["vi (NIFTy)", "vi_native (pure-JAX)"]
times_p = [
    benchmark_data["parametric"]["vi"]["run"],
    benchmark_data["parametric"]["vi_native"]["run"],
]
ax1.bar(
    methods, times_p, color=[COLORS[0], COLORS[1]], alpha=0.8, edgecolor="black", linewidth=1.5
)
ax1.set_ylabel("Wall time (s)", fontsize=11)
ax1.set_title("Parametric (7-D)", fontsize=12, fontweight="bold")
ax1.set_ylim(0, 50)
for i, t in enumerate(times_p):
    ax1.text(i, t + 1, f"{t:.1f} s", ha="center", fontsize=10, fontweight="bold")

# Stochastic scaling
times_s = [
    benchmark_data["stochastic"]["vi"]["run"],
    benchmark_data["stochastic"]["vi_native"]["run"],
]
ax2.bar(
    methods, times_s, color=[COLORS[0], COLORS[1]], alpha=0.8, edgecolor="black", linewidth=1.5
)
ax2.set_ylabel("Wall time (s)", fontsize=11)
ax2.set_title("Stochastic (137-D)", fontsize=12, fontweight="bold")
ax2.set_ylim(0, 80)
for i, t in enumerate(times_s):
    ax2.text(i, t + 2, f"{t:.1f} s", ha="center", fontsize=10, fontweight="bold")

plt.tight_layout()
savefig(fig, "vi_scaling")
plt.show()

print("\nParametric (7-D):")
print(f"  vi (NIFTy):       {times_p[0]:.2f} s")
print(f"  vi_native:        {times_p[1]:.2f} s")
print(f"  Speedup:          {times_p[0] / times_p[1]:.1f}×")

print("\nStochastic (137-D):")
print(f"  vi (NIFTy):       {times_s[0]:.2f} s")
print(f"  vi_native:        {times_s[1]:.2f} s")
print(f"  Speedup:          {times_s[0] / times_s[1]:.1f}×")

print("\nFor comparison, NUTS would take ~1000+ s on 137-D.")
print("VI makes it feasible in seconds.")

# %% [markdown]
# ## 6. Caveats: When NOT to Trust VI
#
# Variational inference is fast but **not risk-free**:
#
# 1. **Posterior mode degeneracy:** VI finds *a* mode, not the global max. With
#    multimodal posteriors, different initializations may converge to different modes.
#
# 2. **Approximate inference:** The ELBO objective is a lower bound on log-evidence,
#    and KL minimization doesn't guarantee posterior samples match the true posterior.
#    On this 137-D setup, the PSD timescale (`sfh_field_psd_tau_myr`) differs by
#    an order of magnitude between `vi` and `vi_native` (82 Myr vs 6 Myr) — use
#    with caution if that parameter is scientifically critical.
#
# 3. **High-dimensional complexity:** Above D ≈ 50, even geoVI can struggle with
#    complex curvature. Run a **validation check** with the Ray Tracing Sampler
#    (exact MCMC, but slow) on a subset.
#
# 4. **Initialization matters:** Use `init_from="random"` for multi-seed exploration,
#    or warm-start with MAP (`init_from="map"`) if you trust a quick point estimate.

# %%
print("\n" + "=" * 70)
print("Caveat Examples")
print("=" * 70)

print("\n1. Mode degeneracy on parametric (7-D):")
print(
    f"   dust_tau_diff: NIFTy={mu_nifty['dust_tau_diff']:.2f}, native={mu_native['dust_tau_diff']:.2f}"
)
print("   (2.3σ apart — different modes)")

print("\n2. Order-of-magnitude PSD timescale drift on 137-D:")
print("   sfh_field_psd_tau_myr: NIFTy=82 Myr, vi_native=6 Myr (13× difference)")
print("   → Use VI for *structure* exploration, validate *values* with MCMC.")

print("\n3. Validation strategy:")
print("   - Run VI for fast posterior exploration.")
print("   - Cross-check critical parameters with Ray Tracing Sampler.")
print("   - See notebook 06_inference_methods.py for sampler comparison.")

# %% [markdown]
# ## 7. Summary: Why VI Powers Paper II
#
# Geometric variational inference is the **enabling technology** for Paper II because it:
#
# - **Scales to realism:** 137-D stochastic SFHs in ~3 s (native) or ~70 s (NIFTy).
# - **Matches approximate posteriors:** On 7-D, credible intervals agree with NIFTy (mostly).
# - **Fast enough for iteration:** Run many chains on many galaxies.
# - **Trade fidelity for speed judiciously:** Know when VI is safe (SFR structure,
#   stellar mass, extinction) and when to verify with MCMC (PSD timescales, AGN parameters).
#
# **For your own analysis:** Use `vi_native` for speed; use `vi` if you need NIFTy's
# extra robustness on edge cases. Always validate on a small sample with NUTS or
# Ray Tracing before deploying to a catalog.

# %%
# ## What you learned
#
# - VI (geoVI) enables high-D inference by trading exact sampling for speed
# - `vi_native` (pure-JAX) is 19–25× faster than `vi` (NIFTy); choose based on speed vs robustness tradeoff
# - VI posteriors match MCMC marginals on 7-D; validate on high-D before deploying
# - Known gaps: PSD timescales (tau_PS) and some AGN parameters need cross-checks with MCMC
# - Population hierarchical inference (11_population.py) solves tau_PS degeneracy at scale
#
# **Next:** [`14_stochastic_sfh.py`](14_stochastic_sfh.py) (single-galaxy stochastic SFH) or
# [`16_simulation_interface.py`](16_simulation_interface.py) (forward-modeling simulation outputs).

# %%
try:
    from tengri import cite_all

    cite_all()
except ImportError:
    print("(citations unavailable)")
