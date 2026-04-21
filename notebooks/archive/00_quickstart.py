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
# # Fit a Galaxy in 60 Seconds
#
# **tengri** recovers bursty star formation histories from broadband
# photometry in seconds.  Where traditional SED fitting codes like BAGPIPES
# and Prospector assume smooth SFHs, tengri adds physically motivated
# stochastic fluctuations governed by a power spectral density (PSD) —
# and keeps everything differentiable in JAX.
#
# This quickstart demonstrates both modes end-to-end:
#
# - **Part A** fits a smooth parametric SFH (7 free parameters) —
#   comparable to BAGPIPES/Prospector, with NUTS as the gold standard.
# - **Part B** fits a stochastic SFH with GP-correlated burstiness
#   (137 free parameters) — the unique capability of tengri, with
#   **geoVI** as the primary inference method.
#
# By the end you will:
# 1. Fit a parametric SFH and compare MAP, geoVI, Ray Tracing, and NUTS
# 2. Fit a stochastic SFH with PSD-governed burstiness
# 3. See corner plots, SFH recovery, and convergence diagnostics
# 4. Understand when to use which inference method

# %%
import time

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Gaussian, LogUniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)

# Publication-quality plot style
import sys; sys.path.insert(0, ".")
from _plot_style import setup_style, COLORS, SDSS_WAVE_EFF, safe_corner
from _plot_style import plot_corner_comparison, convergence_table
setup_style()
import os; os.makedirs("notebook_figures", exist_ok=True)

ssp_data = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
print(f"SSP grid loaded — {len(ssp_data.ssp_lgmet)} metallicities, "
      f"{len(ssp_data.ssp_lg_age_gyr)} ages")
print(f"Filters loaded — {[fc.name for fc in filters[2]]}")

# %% [markdown]
# ## Part A: Parametric SEDModel (catalog-scale fitting)
#
# The parametric model uses a **double power law** for the SFH — a smooth
# function that rises with cosmic time ($\beta$ controls the rise), peaks
# at epoch $\tau_{\rm peak}$, then declines ($\alpha$ controls the decline).
# This is the same SFH shape used by BAGPIPES (Carnall et al. 2018) and
# Prospector (Johnson et al. 2021) in parametric mode.
#
# We use `mean_sfh_type="dpl"` (no stochastic component), leaving **7 free
# parameters**: 4 SFH shape + 1 metallicity + 2 dust.  This is a
# low-dimensional problem where NUTS gives exact, gold-standard posteriors.
#
# > **SED-fitting context:** Most SED fitting in the literature uses smooth
# > parametric SFHs like this.  It works well for integrated quantities
# > (stellar mass, mean SFR) but cannot capture bursty star formation.
# > Part B adds that capability.

# %%
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

# Star-forming galaxy: late-peaking SFH, high current SFR, moderate dust
# (tau_peak=10 Gyr cosmic time ≈ still near peak at z=0.1, alpha=1.0 = slow decline)
key = jax.random.PRNGKey(2026)
true_params = dict(
    sfh_dpl_alpha=1.0,               # slow decline after peak
    sfh_dpl_beta=1.5,                # moderate rise
    sfh_dpl_tau_gyr=10.0,       # peaks late — still actively forming stars at z=0.1
    sfh_dpl_log_peak_sfr=1.176,      # log10(15) Msun/yr at peak
    met_logzsol=-0.2,            # near-solar metallicity
    dust_tau_bc=0.3,             # moderate birth-cloud dust
    dust_tau_diff=0.2,           # moderate diffuse dust
)
mock = model.mock(true_params, snr=20.0, key=key)

print(f"Free parameters: {spec.n_free}")
print(f"Observed bands:  {mock.flux_obs.shape[0]}")

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
wave_eff = SDSS_WAVE_EFF
ax.errorbar(wave_eff, mock.flux_obs, yerr=mock.noise,
            fmt="o", color="k", label="Observed (SNR 20)", zorder=3)
ax.plot(wave_eff, mock.flux_true, "s", ms=6, mfc="none",
        color="C3", label="Truth (no noise)")
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux [arbitrary]")
ax.set_title("Mock SDSS Photometry — Parametric SFH")
ax.legend()
plt.tight_layout()
plt.savefig("notebook_figures/00_quickstart_fig01.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### MAP + geoVI + Ray Tracing + NUTS
#
# We first run **MAP** (Maximum A Posteriori) to find a good starting point
# ($\lesssim 1$ second).  Then we sample the full posterior with three
# complementary methods:
#
# - **geoVI** (Frank et al. 2021) — variational inference on a Riemannian
#   manifold.  Approximate but scales to very high $D$.  The primary
#   method for stochastic SFH problems.
# - **Ray Tracing** (Behroozi 2025) — exact MCMC via Snell's law optics.
#   Fast, noise-tolerant, works at any dimensionality.
# - **NUTS** (Hoffman & Gelman 2014) — gold-standard HMC.  Exact posteriors
#   for low-$D$ problems; impractical above $D \sim 20$.
#
# For the parametric model ($D = 7$), all three give consistent posteriors.
# The differences emerge in Part B where $D \sim 137$.

# %%
fitter = Fitter(model, mock.flux_obs, mock.noise, data_type="photometry")

t0 = time.perf_counter()
result_map = fitter.run("map", n_steps=500)
t_map = time.perf_counter() - t0
print(f"MAP finished in {t_map:.1f}s")

t0 = time.perf_counter()
result_geovi = fitter.run("native_geovi", init_from=result_map,
                          n_iterations=10, n_samples=6, n_seeds=5)
t_geovi = time.perf_counter() - t0
print(f"geoVI (native) finished in {t_geovi:.1f}s")

t0 = time.perf_counter()
result_geovi_nuts = fitter.run("geovi_nuts", init_from=result_map,
                               n_iterations=10, n_samples=3,
                               n_posterior_samples=500)
t_geovi_nuts = time.perf_counter() - t0
print(f"geoVI+NUTS finished in {t_geovi_nuts:.1f}s")

t0 = time.perf_counter()
result_rt = fitter.run("raytrace", init_from=result_map,
                       n_burnin=200, n_steps=2000)
t_rt = time.perf_counter() - t0
print(f"Ray Tracing finished in {t_rt:.1f}s  "
      f"(acceptance: {result_rt.diagnostics.get('accept_rate_post_burnin', 0):.0%})")

t0 = time.perf_counter()
result_nuts = fitter.run("nuts", init_from=result_map,
                         n_warmup=1000, n_samples=1000)
t_nuts = time.perf_counter() - t0
n_div = result_nuts.diagnostics.get("n_divergent", 0)
print(f"NUTS finished in {t_nuts:.1f}s  ({n_div} divergences)")

# %%
# --- Convergence diagnostics ---
convergence_table({
    "geoVI": result_geovi,
    "geoVI+NUTS": result_geovi_nuts,
    "Ray Tracing": result_rt,
    "NUTS": result_nuts,
})

# %%
# --- SFH recovery: geoVI + RT + NUTS overlaid ---
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

ax_sfh = axes[0]
model.plot_sfh_posterior(result_geovi, true_params=true_params,
                         color=COLORS["geovi"], label="geoVI", ax=ax_sfh)
model.plot_sfh_posterior(result_rt, true_params=true_params,
                         color=COLORS["rt"], label="Ray Tracing", ax=ax_sfh)
model.plot_sfh_posterior(result_nuts, true_params=true_params,
                         color=COLORS["nuts"], label="NUTS", ax=ax_sfh)
ax_sfh.set_title("SFH Recovery — Parametric")
ax_sfh.legend(fontsize=9)

# --- Derived quantities: M*, SFR_100, SFR_10, sSFR with truth markers ---
ax_der = axes[1]
derived_truth = model.predict_derived(true_params)
derived_nuts = result_nuts.derived

qty_names = ["stellar_mass", "sfr_100myr", "sfr_10myr", "ssfr"]
qty_labels = [r"$\log M_*/M_\odot$", r"$\log$ SFR$_{100}$",
              r"$\log$ SFR$_{10}$", r"$\log$ sSFR"]

for i, (qty, qlabel) in enumerate(zip(qty_names, qty_labels)):
    y_offset = i * 2.2
    vals = np.array(derived_nuts[qty])
    vals = np.log10(np.clip(vals, 1e-30, None))
    vp = ax_der.violinplot([vals], positions=[y_offset], vert=False,
                           showmedians=True, widths=0.8)
    for pc in vp["bodies"]:
        pc.set_facecolor(COLORS["nuts"])
        pc.set_alpha(0.6)
    truth_val = float(np.log10(np.clip(derived_truth[qty], 1e-30, None)))
    ax_der.axvline(truth_val, color=COLORS["truth"], lw=2, ls="--")
    ax_der.text(0.02, 1 - (i + 0.5) / 4, qlabel, fontsize=10,
                transform=ax_der.transAxes, va="center")

ax_der.set_xlabel("log value")
ax_der.set_title("Derived Quantities (NUTS) — truth dashed")
ax_der.set_yticks([])

plt.tight_layout()
plt.savefig("notebook_figures/00_quickstart_fig02.png", dpi=72, bbox_inches="tight")
plt.show()

# %%
# --- Corner plot: all three samplers overlaid ---
plot_corner_comparison(
    [result_geovi, result_rt, result_nuts],
    ["geoVI", "Ray Tracing", "NUTS"],
    colors=[COLORS["geovi"], COLORS["rt"], COLORS["nuts"]],
    truths=true_params,
)
plt.savefig("notebook_figures/00_quickstart_fig03.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Part B: Stochastic SEDModel (IFT correlated field)
#
# Smooth SFHs are insufficient for many galaxies.  Star formation is
# driven by feedback (supernovae, stellar winds, AGN) and gas accretion,
# producing stochastic fluctuations on timescales from $\sim$10 Myr to
# $\sim$1 Gyr.  The scatter in the star-forming main sequence
# ($\sim$0.3 dex; Speagle et al. 2014) demands variability beyond what
# any smooth function can capture.
#
# This is what makes **tengri** unique.  Instead of a smooth parametric
# SFH, we add a Gaussian-process correlated field whose power spectral
# density (PSD) is governed by two physical hyper-parameters:
#
# $$
# P(k) = \frac{\sigma_{\rm ps}^2 \, \tau_{\rm ps}}{1 + (2\pi k \tau_{\rm ps})^2}
# $$
#
# - $\sigma_{\rm ps}$ — amplitude of stochastic variability (dex)
# - $\tau_{\rm ps}$ — correlation timescale (Myr)
#
# The GP is represented on a grid of $N_{\rm grid} = 128$ points via a
# latent vector $\boldsymbol{\xi} \sim \mathcal{N}(0, I)$, giving a
# total dimensionality of $\sim 137$.  This lets us recover bursty,
# non-parametric SFHs while keeping physically motivated priors.

# %%
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

# Star-forming galaxy with bursty SFH
# Use spec.sample() to get a full parameter set including sfh_field_xi,
# then override physical params to known values
key_stoch = jax.random.PRNGKey(2026)
true_params_stoch = spec_stoch.sample(key_stoch)
true_params_stoch = {**true_params_stoch,
    "sfh_dpl_alpha": 0.8,               # shallow decline — SF continues to present
    "sfh_dpl_beta": 1.5,                # moderate rise
    "sfh_dpl_tau_gyr": 8.0,        # late peak — still actively forming at z=0.1
    "sfh_dpl_log_peak_sfr": 1.903,      # log10(80) — very high peak SFR for dramatic dynamic range
    "sfh_field_psd_sigma": 3.0,               # extreme burstiness — factor ~1000 fluctuations
    "sfh_field_psd_tau_myr": 20.0,            # 20 Myr — SN feedback timescale
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.5,
    "dust_tau_diff": 0.3,
}
mock_stoch = model_stoch.mock(true_params_stoch, snr=20.0, key=key_stoch)

print(f"Free parameters (stochastic): {spec_stoch.n_free}")

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sfh_true = model_stoch.predict_sfh(true_params_stoch)
axes[0].plot(sfh_true["t_gyr"], sfh_true["sfr_full"], color="k", lw=1.2)
axes[0].set_xlabel("Lookback time [Gyr]")
axes[0].set_ylabel("SFR [M$_\\odot$ yr$^{-1}$]")
axes[0].set_title("True Bursty SFH")

wave_eff = SDSS_WAVE_EFF
axes[1].errorbar(wave_eff, mock_stoch.flux_obs, yerr=mock_stoch.noise,
                 fmt="o", color="k", label="Observed", zorder=3)
axes[1].plot(wave_eff, mock_stoch.flux_true, "s", ms=6, mfc="none",
             color="C3", label="Truth")
axes[1].set_xlabel("Wavelength [Å]")
axes[1].set_ylabel("Flux [arbitrary]")
axes[1].set_title("Mock SDSS Photometry — Stochastic SFH")
axes[1].legend()

plt.tight_layout()
plt.savefig("notebook_figures/00_quickstart_fig04.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### geoVI — The Primary Method for Stochastic SFHs
#
# **geoVI** ([Frank et al. 2021](https://arxiv.org/abs/2105.10470))
# performs variational inference on a Riemannian manifold, approximating the
# posterior with a Gaussian in a curved coordinate system.  It is
# *approximate* (unlike Ray Tracing or NUTS) but scales gracefully to
# very high dimensionality and converges in a handful of KL iterations.
#
# For the stochastic model ($D \sim 137$), geoVI is the recommended
# first choice: it produces hundreds of posterior samples without any
# MCMC tuning, step size selection, or burn-in assessment.

# %%
fitter_stoch = Fitter(model_stoch, mock_stoch.flux_obs, mock_stoch.noise,
                      data_type="photometry")

t0 = time.perf_counter()
result_map_stoch = fitter_stoch.run("map", n_steps=1000)
t_map_s = time.perf_counter() - t0
print(f"MAP finished in {t_map_s:.1f}s")

t0 = time.perf_counter()
result_geovi_stoch = fitter_stoch.run("native_geovi", init_from=result_map_stoch,
                                      n_iterations=15, n_samples=6,
                                      n_posterior_samples=200, n_seeds=5)
t_geovi_s = time.perf_counter() - t0
print(f"geoVI finished in {t_geovi_s:.1f}s  "
      f"({result_geovi_stoch.diagnostics['n_samples']} samples)")

# %% [markdown]
# ### Ray Tracing — Exact MCMC Cross-Check
#
# **Ray Tracing** ([Behroozi 2025](https://arxiv.org/abs/2510.25824))
# provides exact, asymptotically unbiased posterior samples via Snell's
# law optics.  It complements geoVI by serving as an independent check:
# where the two posteriors agree, we have high confidence; where they
# disagree, the MCMC result should be preferred.
#
# > **Step size for $D \sim 137$:** Use `step_size=0.005` with 200
# > leapfrog steps.  The viable step-size range narrows sharply at
# > high $D$ — too large and acceptance crashes to 0%.

# %%
t0 = time.perf_counter()
result_rt_stoch = fitter_stoch.run("raytrace", init_from=result_map_stoch,
                                   n_burnin=200, n_steps=2000,
                                   step_size=0.005, n_leapfrog_steps=200)
t_rt_s = time.perf_counter() - t0
accept = result_rt_stoch.diagnostics.get("accept_rate_post_burnin", 0)
print(f"Ray Tracing finished in {t_rt_s:.1f}s  (acceptance: {accept:.0%})")

# %%
# --- Convergence diagnostics (stochastic model) ---
convergence_table({
    "geoVI (stochastic)": result_geovi_stoch,
    "RT (stochastic)": result_rt_stoch,
})

# %%
# --- Overlaid SFH recovery: geoVI primary, RT cross-check ---
fig, ax = plt.subplots(figsize=(10, 5))

# Truth: thick black line
ax.plot(sfh_true["t_gyr"], sfh_true["sfr_full"],
        color=COLORS["truth"], lw=3, label="Truth", zorder=10)
ax.plot(sfh_true["t_gyr"], sfh_true["sfr_mean"],
        color=COLORS["truth"], lw=1.5, ls=":", alpha=0.5, label="Mean SFH", zorder=9)

# MAP point estimate
sfh_map = model_stoch.predict_sfh(result_map_stoch.params)
ax.plot(sfh_map["t_gyr"], sfh_map["sfr_mean"],
        color=COLORS["map"], lw=1.5, ls="--", label="MAP", zorder=5)

# geoVI posterior (primary — plotted first, more prominent)
model_stoch.plot_sfh_posterior(result_geovi_stoch, true_params=true_params_stoch,
                               color=COLORS["geovi"], label="geoVI", ax=ax)

# RT posterior (cross-check)
model_stoch.plot_sfh_posterior(result_rt_stoch, true_params=true_params_stoch,
                               color=COLORS["rt"], label="Ray Tracing", ax=ax)

ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel("SFR [M$_\\odot$ yr$^{-1}$]")
ax.set_title("Stochastic SFH Recovery — geoVI (primary) + RT (cross-check)")
ax.legend(fontsize=9, loc="upper left")

# Clip y-axis: extreme GP samples can push SFR to 1000+ M☉/yr
sfr_truth_max = float(np.max(sfh_true["sfr_full"]))
ax.set_ylim(0, max(5 * sfr_truth_max, 50))

plt.tight_layout()
plt.savefig("notebook_figures/00_quickstart_fig05.png", dpi=72, bbox_inches="tight")
plt.show()

# %%
plot_corner_comparison(
    [result_geovi_stoch, result_rt_stoch],
    ["geoVI", "Ray Tracing"],
    colors=[COLORS["geovi"], COLORS["rt"]],
    truths=true_params_stoch,
)
plt.savefig("notebook_figures/00_quickstart_fig06.png", dpi=72, bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Why not NUTS for the stochastic model?
#
# NUTS relies on Hamiltonian dynamics and the U-turn criterion to set
# trajectory length.  In the 137-D stochastic model, $D$ is simply too
# high: the U-turn criterion becomes unreliable and tuning is impractical,
# so NUTS diverges or mixes poorly.  The tengri forward model is
# *differentiable* — JAX gives exact gradients through the GP — so the
# issue is **high dimensionality**, not noisy gradients.  For $D \gtrsim 20$,
# **geoVI** is the recommended primary method, with **Ray Tracing** as an
# exact MCMC cross-check.
# NUTS remains the gold standard for low-dimensional parametric models (Part A).

# %% [markdown]
# ## What's Next?
#
# This quickstart gave you the 60-second version.  The tutorial series
# goes deeper, building from theory through implementation to science:
#
# - **[NB01 — The IFT SEDModel](01_the_model.ipynb)**: How the PSD governs
#   burstiness — the climate/weather analogy for star formation
# - **[NB02 — Forward SEDModel](02_forward_model.ipynb)**: Step-by-step SPS
#   pipeline from SFH to photometry, with gradient sensitivity analysis
# - **[NB03 — Inference Methods](03_inference_methods.ipynb)**: Deep dive
#   into all five samplers — when each shines and where it breaks
# - **[NB04 — Recovery Tests](04_recovery_tests.ipynb)**: Can you trust
#   the posteriors?  Mock validation across burstiness regimes
# - **[NB05 — Hierarchical](05_hierarchical.ipynb)**: Population-level PSD
#   recovery — constraining $\tau$ by sharing across $N$ galaxies
# - **[NB06 — Data Information](06_data_information.ipynb)**: How much data
#   do you need?  From 1 band to a full spectrum
# - **[NB07 — Spectroscopy](07_spectroscopic_fitting.ipynb)**: What spectra
#   tell you that photometry can't — breaking degeneracies with features
# - **[NB08 — PSD Physics](08_psd_physics.ipynb)**: The observer's
#   translation guide — mapping PSD parameters to astrophysical mechanisms
# - **[NB09 — Custom Models](09_custom_models.ipynb)**: Extending tengri
#   with new priors, PSD models, dust laws, and SSP templates
