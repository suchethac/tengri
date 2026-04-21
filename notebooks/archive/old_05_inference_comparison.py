# ---
# jupyter:
#   jupytext:
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
# # Tutorial 5: Inference Methods — MAP, NUTS, Ray Tracing, and geoVI
#
# tengri supports four gradient-based inference methods. All exploit end-to-end differentiability through the forward model.
#
# | Method | Speed | Uncertainties | Best for |
# |--------|-------|---------------|----------|
# | **MAP** (Adam) | Seconds | No | Large surveys, initial guesses |
# | **NUTS** (BlackJAX) | Minutes | Yes (exact) | Gold-standard posteriors |
# | **Ray Tracing** (Behroozi 2025) | Seconds | Yes (exact) | Robust posteriors, noisy gradients |
# | **geoVI** (NIFTy.re) | Seconds | Yes (approx) | Fast posteriors, hierarchical models |

# %%
# %matplotlib inline
import sys; sys.path.insert(0, "../src")
import warnings; warnings.filterwarnings('ignore')
import logging; logging.disable(logging.WARNING)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import time, io

from tengri import (
    SEDModel, ParamSpec, Fitter, Uniform, Gaussian, Fixed,
    load_ssp_data, load_filter_set,
)

plt.rcParams.update({
    "figure.dpi": 130, "font.size": 11,
    "axes.linewidth": 1.0,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})

ssp = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
filter_names = ["u", "g", "r", "i", "z"]

filter_names = ['u', 'g', 'r', 'i', 'z']
filter_curves = filters[2]
band_centers_arr = jnp.array([float(jnp.mean(fc.wave)) for fc in filter_curves])

# %% [markdown]
# ## Setup: Mock Galaxy at z = 0.1
#
# We fit a single parametric mock galaxy (no stochastic GP) with all four methods and compare.

# %%
spec = ParamSpec(
    sfh_alpha        = Uniform(0.5, 3.0),
    sfh_beta         = Uniform(0.3, 2.0),
    sfh_tau_peak_gyr = Uniform(0.5, 10.0),
    sfh_peak_sfr     = Uniform(0.1, 50.0),
    met_logzsol      = -0.2,             # fixed
    dust_tau_bc      = Uniform(0.0, 3.0),
    dust_tau_diff    = 0.3,              # fixed
    dust_slope       = -0.7,
    redshift         = 0.1,
    stochastic       = False,
)
model = SEDModel(spec, ssp, filters=filters)

# Star-forming galaxy
true_params = {
    "sfh_alpha": 1.0, "sfh_beta": 0.8,
    "sfh_tau_peak_gyr": 3.0, "sfh_peak_sfr": 10.0,
    "met_logzsol": -0.2,
    "dust_tau_bc": 0.8, "dust_tau_diff": 0.3,
    "dust_slope": -0.7, "redshift": 0.1,
}

mock = model.mock(true_params, snr=20.0, key=jax.random.PRNGKey(0))
sfh_true = model.predict_sfh(true_params)

print(f"Mock: {len(filter_names)} bands, SNR=20, z={true_params['redshift']}")
print(f"Free parameters: {spec.n_free}")

# %% [markdown]
# ---
# ## 1. MAP (Adam)
#
# Fast point estimate — seconds per galaxy. No error bars.

# %%
fitter = Fitter(model, mock.flux_obs, mock.noise)

result_map = fitter.run("map", n_steps=5000, learning_rate=0.05, print_every=500, verbose=True)
print(f"\nWall time: {result_map.wall_time_s:.1f}s")

# %% [markdown]
# ---
# ## 2. NUTS (BlackJAX)
#
# Exact posterior via Hamiltonian Monte Carlo. Uses the gradient to efficiently explore the posterior surface. Initialized from MAP for faster convergence.

# %%
try:
    result_nuts = fitter.run(
        "nuts", init_from=result_map,
        n_warmup=500, n_samples=500,
        target_accept_rate=0.9, verbose=True,
    )
    nuts_ok = True
    print(f"\nDivergences: {result_nuts.diagnostics['n_divergent']}/{result_nuts.diagnostics['n_samples']}")
except Exception as e:
    nuts_ok = False
    print(f"NUTS failed: {type(e).__name__}: {str(e)[:200]}")

# %% [markdown]
# ---
# ## 3. geoVI (NIFTy.re)
#
# Approximate posterior via geometric variational inference. Finds a coordinate transform where the posterior is approximately Gaussian. Much faster than NUTS with comparable quality for most SED fits.

# %%
try:
    import warnings, logging
    warnings.filterwarnings('ignore')
    logging.getLogger('nifty8').setLevel(logging.ERROR)

    result_geovi = fitter.run(
        "geovi", init_from=result_map,
        n_iterations=15, n_samples=6, n_posterior_samples=80, verbose=False,
    )
    print(f"geoVI: {result_geovi.diagnostics['n_samples']} samples "
          f"in {result_geovi.wall_time_s:.1f}s")
    geovi_ok = True
except Exception as e:
    geovi_ok = False
    print(f"geoVI failed: {e}")

# %% [markdown]
# ---
# ## 4. Ray Tracing Sampler (Behroozi 2025)
#
# A physics-inspired MCMC sampler that propagates "light rays" through a medium whose refractive index is set by the likelihood surface. Snell's law bends rays toward high-likelihood regions, naturally producing fair posterior samples. Key advantages: resilience to noisy gradients, no energy-conservation issues, and the ability to cross likelihood barriers. Initialized from MAP for faster convergence.

# %%
try:
    result_rt = fitter.run(
        "raytrace", init_from=result_map,
        n_steps=400, n_leapfrog_steps=10, n_burnin=100, verbose=True,
    )
    rt_ok = True
    print(f"\nAcceptance rate: {result_rt.diagnostics['accept_rate']:.1%} (overall), "
          f"{result_rt.diagnostics['accept_rate_post_burnin']:.1%} (post burn-in)")
    print(f"Samples: {result_rt.diagnostics['n_samples']}")
    print(f"Wall time: {result_rt.wall_time_s:.1f}s")
except Exception as e:
    rt_ok = False
    print(f"Ray Tracing failed: {type(e).__name__}: {str(e)[:200]}")

# %% [markdown]
# ---
# ## 5. Comparison: SFH Recovery (Linear Scale)
#
# The key comparison: how well does each method recover the true star formation history? Plotted in **linear** SFR for intuitive interpretation.

# %%
methods = {"MAP": (result_map, "C0")}
if nuts_ok:
    methods["NUTS"] = (result_nuts, "C1")
if rt_ok:
    methods["Ray Tracing"] = (result_rt, "C3")
if geovi_ok:
    methods["geoVI"] = (result_geovi, "C2")

n_methods = len(methods)
fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 5), sharey=True)
if n_methods == 1: axes = [axes]

for ax, (name, (res, color)) in zip(axes, methods.items()):
    model.plot_sfh_posterior(res, true_params=true_params, ax=ax,
                            color=color, label=name)
    ax.set_title(f"{name} ({res.wall_time_s:.1f}s)")
    ax.text(0.98, 0.98, f"z = {true_params['redshift']}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, style="italic")

plt.suptitle("SFH Recovery — Linear Scale (16-84% fill)", fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 6. Parameter Posteriors (Physical Units)

# %%
# Show posteriors for methods that have samples
sampling_results = {}
if nuts_ok:
    sampling_results["NUTS"] = (result_nuts, "C1")
if rt_ok:
    sampling_results["Ray Tracing"] = (result_rt, "C3")
if geovi_ok:
    sampling_results["geoVI"] = (result_geovi, "C2")

if sampling_results:
    phys_params = [
        ("sfh_alpha",         r"$\alpha$",                 1.5),
        ("sfh_beta",          r"$\beta$",                  1.0),
        ("sfh_tau_peak_gyr",  r"$\tau_{\rm peak}$ (Gyr)", 4.0),
        ("sfh_peak_sfr",      r"SFR$_{\rm peak}$",        8.0),
        ("met_logzsol",       r"log Z",                    -0.3),
        ("dust_tau_bc",       r"$\tau_{\rm bc}$",         1.0),
        ("dust_tau_diff",     r"$\tau_{\rm diff}$",       0.3),
    ]

    n_params = len(phys_params)
    fig, axes_p = plt.subplots(2, 4, figsize=(14, 6))
    axes_flat = axes_p.ravel()

    for idx, (key, label, true_val) in enumerate(phys_params):
        ax = axes_flat[idx]
        for mname, (res, color) in sampling_results.items():
            if key in res.samples:
                vals = np.array(res.samples[key])
                ax.hist(vals, bins=min(20, len(vals)//2 + 1), color=color,
                        alpha=0.5, density=True, label=mname)
        ax.axvline(true_val, color="k", ls="--", lw=1.5, label="Truth")
        ax.set_xlabel(label, fontsize=10)
        ax.set_yticks([])
        if idx == 0:
            ax.legend(fontsize=7)

    # Hide extra subplot
    axes_flat[-1].set_visible(False)

    fig.suptitle("Posterior Distributions (Physical Parameters)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.show()
else:
    print("No sampling methods available — install blackjax or nifty8[re]")

# %% [markdown]
# ### Corner Plot (geoVI)
#
# The corner plot shows all pairwise correlations between physical parameters and derived quantities (stellar mass, SFR). Dashed lines mark the true values.

# %%
if geovi_ok:
    # Add derived truths for corner plot
    d_true = model.predict_derived(true_params)
    corner_truths = dict(true_params)
    corner_truths["stellar_mass"] = float(d_true["stellar_mass"])
    corner_truths["sfr_100myr"] = float(d_true["sfr_100myr"])

    fig = result_geovi.plot_corner(truths=corner_truths)
    fig.suptitle("geoVI Posterior Corner Plot (Smooth SEDModel)", fontsize=13, y=1.01)
    plt.show()
else:
    print("Skipping — geoVI not available")

# %% [markdown]
# ### Posterior Predictive Checks
#
# Overlay posterior predicted photometry and spectrum on the data. Good fits should have the data within the posterior spread.

# %%
# Posterior predictive checks: photometry + spectrum
if geovi_ok:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: photometry
    ax = axes[0]
    n_draws = min(50, result_geovi.diagnostics['n_samples'])
    for i in range(n_draws):
        s_i = {k: result_geovi.samples[k][i] for k in result_geovi.samples}
        phot_i = model.predict_photometry(s_i)
        ax.plot(np.array(band_centers_arr) / 1e4, np.array(phot_i) * 1e29,
                '-', color='C0', alpha=0.08, lw=0.5)
    
    phot_best = model.predict_photometry(result_geovi.params)
    ax.plot(np.array(band_centers_arr) / 1e4, np.array(phot_best) * 1e29,
            'o-', color='C0', ms=6, lw=1.5, label='Posterior mean')
    ax.errorbar(np.array(band_centers_arr) / 1e4,
                np.array(mock.flux_obs) * 1e29,
                yerr=np.array(mock.noise) * 1e29,
                fmt='ks', ms=8, capsize=3, label='Data', zorder=10)
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylabel(r'Flux ($\mu$Jy)')
    ax.legend(fontsize=8)
    ax.set_title('Photometry')

    # Right: spectrum
    ax = axes[1]
    wave_obs = jnp.linspace(4000, 10000, 200)
    for i in range(n_draws):
        s_i = {k: result_geovi.samples[k][i] for k in result_geovi.samples}
        spec_i = model.predict_spectrum(s_i, wave_obs)
        ax.plot(np.array(wave_obs) / 1e4, np.array(spec_i) * 1e29,
                '-', color='C2', alpha=0.06, lw=0.3)
    
    spec_best = model.predict_spectrum(result_geovi.params, wave_obs)
    spec_true = model.predict_spectrum(true_params, wave_obs)
    ax.plot(np.array(wave_obs) / 1e4, np.array(spec_best) * 1e29,
            '-', color='C2', lw=1.5, label='Posterior mean')
    ax.plot(np.array(wave_obs) / 1e4, np.array(spec_true) * 1e29,
            'k--', lw=1, alpha=0.5, label='Truth')
    ax.set_xlabel(r'Wavelength ($\mu$m)')
    ax.set_ylabel(r'Flux ($\mu$Jy)')
    ax.legend(fontsize=8)
    ax.set_title('Predicted Spectrum')

    plt.tight_layout()
    plt.savefig("figures/05_posterior_predictive.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("geoVI not available — skipping posterior predictive checks")

# %% [markdown]
# ---
# ## 7. Parameter Recovery Table

# %%
print(f"{'Parameter':<22s} {'True':>8s}", end="")
for name in methods:
    print(f"  {name:>12s}", end="")
print()
print("-" * (22 + 10 + 14 * len(methods)))

for key in ["sfh_alpha", "sfh_beta", "sfh_tau_peak_gyr", "sfh_peak_sfr",
            "met_logzsol", "dust_tau_bc", "dust_tau_diff"]:
    true_v = true_params[key]
    print(f"{key:<22s} {true_v:>8.3g}", end="")
    for name, (res, _) in methods.items():
        s = res.summary()
        if key in s:
            if "median" in s[key]:
                print(f"  {s[key]['median']:>12.3g}", end="")
            else:
                print(f"  {s[key]['value']:>12.3g}", end="")
    print()

print()
print("Wall times:")
for name, (res, _) in methods.items():
    print(f"  {name}: {res.wall_time_s:.1f}s")

# %% [markdown]
# ---
# ## 8. Derived Quantities

# %%
d_true = model.predict_derived(true_params)

print(f"{'Quantity':<18s} {'True':>12s}", end="")
for name in methods:
    print(f"  {name:>12s}", end="")
print()
print("-" * (18 + 14 + 14 * len(methods)))

for key in ["stellar_mass", "sfr_100myr", "ssfr"]:
    print(f"{key:<18s} {float(d_true[key]):>12.3e}", end="")
    for name, (res, _) in methods.items():
        d = model.predict_derived(res.params)
        print(f"  {float(d[key]):>12.3e}", end="")
    print()

# %% [markdown]
# ---
# ## 9. Stochastic SEDModel: geoVI vs NUTS
#
# The stochastic SFH model adds a GP with `n_grid=128` latent variables, making the total dimensionality 128 + 9 = 137. NUTS struggles in this regime (many divergences, slow mixing). geoVI handles it naturally because it approximates the posterior in a transformed Gaussian space.
#
# This is the recommended workflow for the full IFT model: **MAP initialization → geoVI posterior**.

# %%
# Stochastic model: GP with n_grid=128
spec_stoch = ParamSpec(
    sfh_alpha        = Uniform(0.5, 3.0),
    sfh_beta         = Uniform(0.3, 2.0),
    sfh_tau_peak_gyr = Uniform(0.5, 10.0),
    sfh_peak_sfr     = Uniform(0.1, 50.0),
    psd_sigma        = Uniform(0.1, 3.0),
    psd_tau_myr      = Uniform(1.0, 300.0),
    met_logzsol      = Uniform(-1.5, 0.2),
    dust_tau_bc      = Uniform(0.0, 3.0),
    dust_tau_diff    = Uniform(0.0, 2.0),
    dust_slope       = -0.7,
    redshift         = 0.1,
    stochastic       = True,
    n_grid           = 128,
)
model_stoch = SEDModel(spec_stoch, ssp, filters=filters)

# Star-forming galaxy with moderate burstiness
true_stoch = spec_stoch.sample(jax.random.PRNGKey(7))
true_stoch["sfh_alpha"] = jnp.array(1.2)
true_stoch["sfh_beta"] = jnp.array(0.8)
true_stoch["sfh_tau_peak_gyr"] = jnp.array(2.0)
true_stoch["sfh_peak_sfr"] = jnp.array(15.0)
true_stoch["psd_sigma"] = jnp.array(1.5)
true_stoch["psd_tau_myr"] = jnp.array(30.0)
true_stoch["met_logzsol"] = jnp.array(-0.2)
true_stoch["dust_tau_bc"] = jnp.array(0.8)
true_stoch["dust_tau_diff"] = jnp.array(0.3)

mock_stoch = model_stoch.mock(true_stoch, snr=20.0, key=jax.random.PRNGKey(1))

print(f"Stochastic model: {spec_stoch.n_free} free + {spec_stoch.n_grid} psd_xi "
      f"= {spec_stoch.n_free + spec_stoch.n_grid} total dimensions")

# %%
# MAP initialization
fitter_stoch = Fitter(model_stoch, mock_stoch.flux_obs, mock_stoch.noise)
result_map_s = fitter_stoch.run("map", n_steps=2000, learning_rate=0.03, verbose=True)
print(f"MAP: {result_map_s.wall_time_s:.1f}s")

# %%
# geoVI (recommended for stochastic)
try:
    import warnings, logging
    warnings.filterwarnings('ignore')
    logging.getLogger('nifty8').setLevel(logging.ERROR)

    result_geovi_s = fitter_stoch.run(
        "geovi", init_from=result_map_s,
        n_iterations=15, n_samples=6, n_posterior_samples=80, verbose=False,
    )
    print(f"geoVI: {result_geovi_s.diagnostics['n_samples']} samples "
          f"in {result_geovi_s.wall_time_s:.1f}s")
    geovi_stoch_ok = True
except Exception as e:
    geovi_stoch_ok = False
    print(f"geoVI failed: {e}")

# %%
# NUTS (for comparison — expect divergences)
try:
    result_nuts_s = fitter_stoch.run(
        "nuts", init_from=result_map_s,
        n_warmup=200, n_samples=100, verbose=True,
    )
    nuts_stoch_ok = True
    print(f"\nNUTS: {result_nuts_s.diagnostics['n_divergent']}/{result_nuts_s.diagnostics['n_samples']} divergences, "
          f"{result_nuts_s.wall_time_s:.1f}s")
except Exception as e:
    nuts_stoch_ok = False
    print(f"NUTS failed: {type(e).__name__}: {str(e)[:200]}")

# %% [markdown]
# ### Stochastic SFH Recovery (Linear Scale)
#
# With 137 dimensions, geoVI finds the posterior efficiently while NUTS struggles. The SFH is plotted in linear scale.

# %%
methods_s = {"MAP": (result_map_s, "C0")}
if geovi_stoch_ok:
    methods_s["geoVI"] = (result_geovi_s, "C2")
if nuts_stoch_ok:
    methods_s["NUTS"] = (result_nuts_s, "C1")

n_m = len(methods_s)
fig, axes = plt.subplots(1, n_m, figsize=(5 * n_m, 5), sharey=True)
if n_m == 1: axes = [axes]

for ax, (name, (res, color)) in zip(axes, methods_s.items()):
    model_stoch.plot_sfh_posterior(res, true_params=true_stoch, ax=ax,
                                  color=color, label=name)
    ax.set_title(f"{name} — Stochastic ({res.wall_time_s:.1f}s)")

plt.suptitle(f"Stochastic SFH Recovery (n_grid={spec_stoch.n_grid}, 16-84% fill)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.show()

print("\nTiming comparison (stochastic model):")
for name, (res, _) in methods_s.items():
    extra = ""
    if 'n_divergent' in res.diagnostics:
        extra = f", {res.diagnostics['n_divergent']} divergences"
    print(f"  {name:6s}: {res.wall_time_s:6.1f}s{extra}")

# %% [markdown]
# ### Key Observation
#
# For the stochastic model (128 GP dimensions + 9 physical parameters = 137D):
# - **geoVI** handles it naturally — the Gaussian approximation in the transformed coordinate system is efficient
# - **NUTS** struggles with many divergences and slow convergence — the high-dimensional GP space makes HMC trajectories unstable
#
# This is why **geoVI is the recommended inference method** for the full IFT model with stochastic SFHs.

# %% [markdown]
# ---
# ## Key Takeaways
#
# 1. **MAP** is fastest (seconds) — good for catalogs, no uncertainties
# 2. **NUTS** gives exact posteriors (minutes) — 10-100x faster than dynesty/MultiNest
# 3. **Ray Tracing** gives exact posteriors (seconds) — resilient to noisy gradients, no energy conservation issues
# 4. **geoVI** gives approximate posteriors (seconds) — best speed/accuracy trade-off
# 5. All four use the **same loss function** and benefit from end-to-end JAX gradients
# 6. **Photometry alone** constrains the broad SED shape but not individual SFH parameters — spectroscopy helps break degeneracies
