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
# **tengri** recovers bursty star formation histories from photometric and
# spectroscopic data using differentiable stellar population synthesis and
# Information Field Theory.  Where traditional SED-fitting codes impose smooth
# or discretely binned SFHs, tengri treats the SFH as a *continuous
# correlated field* governed by a power spectral density (PSD) — and keeps
# the entire forward model differentiable in JAX.
#
# This quickstart shows two fits end-to-end:
#
# | | **Part A: Parametric** | **Part B: Stochastic** |
# |---|---|---|
# | SFH model | Double power law (smooth) | Double power law + GP burstiness |
# | Free parameters | 7 | ~137 |
# | Inference | MAP → NUTS (gold standard) | MAP → Ray Tracing |
# | Comparable to | BAGPIPES / Prospector parametric mode | **Unique to tengri** |
#
# By the end you will have fitted both a smooth and a bursty mock galaxy,
# seen posterior SFH recovery with a zoomed inset of the last 200 Myr
# where burstiness lives, and produced corner plots of the physical
# parameters.  Detailed explanations of every step follow in
# [T01](T01_ift_model.ipynb)–[T06](T06_extending.ipynb).

# %%
import time
import os

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from tengri import (
    SEDModel, ParamSpec, Uniform, Gaussian, LogUniform, Fixed, Fitter,
    load_ssp_data, load_filter_set,
)

# ── Plot style ────────────────────────────────────────────────────
# When _plot_style.py is available, replace this block with:
#   from _plot_style import setup_style, COLORS, ...
#   setup_style()
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.frameon": False,
})

COLORS = {
    "truth": "#222222",
    "map": "#999999",
    "rt": "#1b9e77",
    "nuts": "#d95f02",
    "geovi": "#7570b3",
    "mgvi": "#e7298a",
    "phot": "#4575b4",
    "spec": "#d73027",
}

# Effective wavelengths for SDSS ugriz (Å)
SDSS_WAVE_EFF = np.array([3561.8, 4718.9, 6185.2, 7499.7, 8961.4])

FIG_DIR = "notebook_figures"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fig, name, dpi=150):
    path = os.path.join(FIG_DIR, f"T00_{name}.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"  → {path}")


# ── SFH inset helper ─────────────────────────────────────────────
def add_sfh_inset(ax, t_gyr, sfr, inset_range_myr=200, loc="upper right",
                  width="35%", height="40%", **plot_kwargs):
    """Add an inset axis zooming into the recent SFH.

    Parameters
    ----------
    ax : matplotlib Axes
        Parent axis (lookback time in Gyr on x-axis).
    t_gyr : array
        Lookback time in Gyr.
    sfr : array
        Star formation rate.
    inset_range_myr : float
        Lookback time range for the inset, in Myr.
    loc : str
        Location string for inset_axes.
    width, height : str
        Size of inset as percentage of parent.

    Returns
    -------
    ax_in : matplotlib Axes
        The inset axis.
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_in = inset_axes(ax, width=width, height=height, loc=loc,
                       borderpad=1.5)

    t_myr = np.asarray(t_gyr) * 1e3  # convert Gyr → Myr
    mask = t_myr <= inset_range_myr
    if mask.sum() > 2:
        ax_in.plot(t_myr[mask], np.asarray(sfr)[mask], **plot_kwargs)

    ax_in.set_xlim(0, inset_range_myr)
    ax_in.set_xlabel("Myr", fontsize=7, labelpad=1)
    ax_in.tick_params(labelsize=6)

    # Mark Hα (~10 Myr) and UV (~100 Myr) timescales
    ax_in.axvspan(0, 10, alpha=0.08, color="blue", zorder=0)
    ax_in.axvspan(0, 100, alpha=0.04, color="purple", zorder=0)
    if inset_range_myr >= 100:
        ylim = ax_in.get_ylim()
        y_txt = ylim[1] * 0.92 if ylim[1] > 0 else 0.9
        ax_in.text(5, y_txt, r"H$\alpha$", fontsize=5.5, color="blue",
                   va="top")
        ax_in.text(50, y_txt, "UV", fontsize=5.5, color="purple", va="top")

    # Visual indicator that this is a zoom
    ax_in.set_title("last 200 Myr", fontsize=7, pad=2)
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.6)

    return ax_in


# %% [markdown]
# ## Load SSP Data
#
# tengri requires pre-computed simple stellar population (SSP) spectra.
# The default templates use FSPS with MIST isochrones, MILES spectra, a
# Chabrier IMF, and nebular emission from Cloudy.

# %%
ssp_data = load_ssp_data(
    "../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

print(f"SSP grid: {len(ssp_data.ssp_lgmet)} metallicities × "
      f"{len(ssp_data.ssp_lg_age_gyr)} ages × "
      f"{len(ssp_data.ssp_wave)} wavelengths")
print(f"Filters:  {[fc.name for fc in filters[2]]}")

# %% [markdown]
# ---
# ## Part A — Parametric SEDModel
#
# A smooth double power-law SFH with **7 free parameters**: 4 SFH shape
# + 1 metallicity + 2 dust.  The PSD amplitude is fixed to zero
# (`psd_sigma = 0`), so there is no stochastic component.  This is the
# same class of model used in BAGPIPES parametric mode or Prospector with
# a `continuity` SFH.
#
# For this low-dimensional problem ($D = 7$), NUTS gives exact,
# gold-standard posteriors.

# %%
spec_param = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0),       # decline slope (cosmic time)
    sfh_beta=Uniform(0.5, 3.0),        # rise slope (cosmic time)
    sfh_tau_peak_gyr=Uniform(0.5, 13.0),  # peak epoch
    sfh_peak_sfr=Uniform(0.1, 100.0),  # normalization (Msun/yr at peak)
    psd_sigma=Fixed(0.0),              # no burstiness
    psd_tau_myr=Fixed(50.0),
    met_logzsol=Uniform(-2.0, 0.5),    # stellar metallicity
    dust_tau_bc=Uniform(0.0, 2.0),     # birth-cloud dust
    dust_tau_diff=Uniform(0.0, 2.0),   # diffuse ISM dust
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    stochastic=False,
)
model_param = SEDModel(spec_param, ssp_data, filters=filters)
print(f"Free parameters: {spec_param.n_free}")

# %%
# ── Mock galaxy: late-peaking star-forming spiral ─────────────────
key = jax.random.PRNGKey(2026)
true_param = dict(
    sfh_alpha=1.0,            # slow decline after peak
    sfh_beta=1.5,             # moderate rise
    sfh_tau_peak_gyr=10.0,    # peaks late → actively forming at z=0.1
    sfh_peak_sfr=15.0,        # 15 Msun/yr at peak
    met_logzsol=-0.2,         # near-solar
    dust_tau_bc=0.3,          # moderate birth-cloud dust
    dust_tau_diff=0.2,        # moderate diffuse dust
)
mock_param = model_param.mock(true_param, snr=20.0, key=key)

# %% [markdown]
# ### Mock photometry

# %%
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.errorbar(SDSS_WAVE_EFF, mock_param.flux_obs, yerr=mock_param.noise,
            fmt="o", color="k", ms=6, capsize=3, label="Observed (SNR 20)",
            zorder=3)
ax.plot(SDSS_WAVE_EFF, mock_param.flux_true, "s", ms=7, mfc="none",
        mew=1.5, color="C3", label="Truth (noiseless)")
ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
ax.set_ylabel("Flux [arbitrary]")
ax.set_title("Mock SDSS Photometry — Parametric SFH", fontsize=11)
ax.legend(fontsize=9)
fig.tight_layout()
savefig(fig, "mock_photometry_parametric")
plt.show()

# %% [markdown]
# ### MAP → NUTS
#
# We first run **MAP** (maximum a posteriori via Adam) for a fast point
# estimate, then draw posterior samples with **NUTS** — the gold-standard
# Hamiltonian Monte Carlo sampler for $D \lesssim 20$.

# %%
fitter_param = Fitter(model_param, mock_param.flux_obs, mock_param.noise,
                      data_type="photometry")

t0 = time.perf_counter()
result_map_p = fitter_param.run("map", n_steps=500)
t_map = time.perf_counter() - t0
print(f"MAP: {t_map:.1f} s")

t0 = time.perf_counter()
result_nuts = fitter_param.run("nuts", init_from=result_map_p,
                               n_warmup=1000, n_samples=1000)
t_nuts = time.perf_counter() - t0
n_div = result_nuts.diagnostics.get("n_divergent", 0)
print(f"NUTS: {t_nuts:.1f} s  ({n_div} divergences)")

# %% [markdown]
# ### SFH recovery
#
# The black curve is the true SFH; the coloured band shows the posterior
# median and 68% credible interval from NUTS.  The inset zooms into the
# last 200 Myr — the regime where Hα and UV constrain the recent SFH.
# For a smooth parametric model the inset is featureless, which is
# exactly the limitation that Part B addresses.

# %%
fig, ax = plt.subplots(figsize=(8, 4.5))

# Truth
sfh_true = model_param.predict_sfh(true_param)
t_gyr = sfh_true["t_gyr"]
sfr_true = sfh_true["sfr_mean"]
ax.plot(t_gyr, sfr_true, color=COLORS["truth"], lw=2.5, label="Truth",
        zorder=10)

# NUTS posterior SFH
model_param.plot_sfh_posterior(result_nuts, true_params=true_param,
                               color=COLORS["nuts"], label="NUTS", ax=ax)

ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax.set_title("SFH Recovery — Parametric (NUTS)", fontsize=11)
ax.set_xlim(0, 13.5)
ax.set_ylim(bottom=0)
ax.legend(fontsize=9)

# Inset: last 200 Myr
add_sfh_inset(ax, t_gyr, sfr_true, inset_range_myr=200,
              color=COLORS["truth"], lw=2)

fig.tight_layout()
savefig(fig, "sfh_recovery_parametric")
plt.show()

# %% [markdown]
# ### Corner plot

# %%
from _plot_style import safe_corner, plot_corner_comparison

plot_corner_comparison(
    [result_nuts],
    ["NUTS"],
    colors=[COLORS["nuts"]],
    truths=true_param,
)
plt.suptitle("Parametric SEDModel — NUTS Posterior", fontsize=12, y=1.02)
savefig(plt.gcf(), "corner_parametric")
plt.show()

# %% [markdown]
# ---
# ## Part B — Stochastic SEDModel (the unique capability)
#
# Real galaxies are not smooth.  Feedback from supernovae, stellar winds,
# and gas accretion drives stochastic SFR fluctuations on timescales from
# ~1 Myr to ~1 Gyr.  tengri captures this by adding a Gaussian-process
# correlated field to the smooth mean SFH, governed by a damped random
# walk (DRW) power spectral density with two physical parameters:
#
# - **$\sigma_{\rm PS}$** — amplitude of SFR fluctuations
# - **$\tau_{\rm PS}$** — coherence timescale (Myr)
#
# The GP is represented on a grid of $N_{\rm grid} = 128$ points via
# a latent vector $\boldsymbol{\xi} \sim \mathcal{N}(0, I)$, giving
# $D \approx 137$ total free parameters.  For this high-dimensional
# problem, we use the **Ray Tracing Sampler** (Behroozi 2025) — an
# exact MCMC method with ~250× more gradient-noise tolerance than HMC.

# %%
spec_stoch = ParamSpec(
    sfh_alpha=Uniform(0.5, 3.0),
    sfh_beta=Uniform(0.5, 3.0),
    sfh_tau_peak_gyr=Uniform(0.5, 13.0),
    sfh_peak_sfr=Uniform(0.1, 100.0),
    psd_sigma=Uniform(0.1, 4.0),
    psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    stochastic=True,
    n_grid=128,
)
model_stoch = SEDModel(spec_stoch, ssp_data, filters=filters)
print(f"Free parameters (stochastic): {spec_stoch.n_free}")

# %%
# ── Mock galaxy: bursty star-forming system ──────────────────────
key_s = jax.random.PRNGKey(2026)
true_stoch = spec_stoch.sample(key_s)
true_stoch = {**true_stoch,
    "sfh_alpha": 0.8,            # shallow decline — SF continues
    "sfh_beta": 1.5,             # moderate rise
    "sfh_tau_peak_gyr": 8.0,     # late peak
    "sfh_peak_sfr": 50.0,        # high peak SFR
    "psd_sigma": 2.0,            # bursty (factor ~100 SFR fluctuations)
    "psd_tau_myr": 20.0,         # SN feedback timescale
    "met_logzsol": -0.3,
    "dust_tau_bc": 0.5,
    "dust_tau_diff": 0.3,
}
mock_stoch = model_stoch.mock(true_stoch, snr=20.0, key=key_s)

# %% [markdown]
# ### The true bursty SFH
#
# Note the dramatic SFR fluctuations — factors of $\sim$100 on
# timescales of tens of Myr.  The inset shows the last 200 Myr, which
# is what Hα and UV observations actually probe.

# %%
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Left: true SFH
sfh_s = model_stoch.predict_sfh(true_stoch)
t_gyr_s = sfh_s["t_gyr"]
sfr_full = sfh_s["sfr_full"]
sfr_mean = sfh_s["sfr_mean"]

ax = axes[0]
ax.plot(t_gyr_s, sfr_full, color=COLORS["truth"], lw=1.2,
        label="Full SFH (mean × burstiness)")
ax.plot(t_gyr_s, sfr_mean, color=COLORS["truth"], lw=1.5, ls=":",
        alpha=0.5, label="Secular mean")
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax.set_title(r"True Bursty SFH ($\sigma_{\rm PS}=2.0$, "
             r"$\tau_{\rm PS}=20$ Myr)", fontsize=10)
ax.set_xlim(0, 13.5)
ax.set_ylim(bottom=0)
ax.legend(fontsize=8)

# Inset
add_sfh_inset(ax, t_gyr_s, sfr_full, inset_range_myr=200,
              color=COLORS["truth"], lw=1.0)

# Right: mock photometry
ax = axes[1]
ax.errorbar(SDSS_WAVE_EFF, mock_stoch.flux_obs, yerr=mock_stoch.noise,
            fmt="o", color="k", ms=6, capsize=3, label="Observed (SNR 20)",
            zorder=3)
ax.plot(SDSS_WAVE_EFF, mock_stoch.flux_true, "s", ms=7, mfc="none",
        mew=1.5, color="C3", label="Truth")
ax.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
ax.set_ylabel("Flux [arbitrary]")
ax.set_title("Mock SDSS Photometry — Stochastic SFH", fontsize=10)
ax.legend(fontsize=9)

fig.tight_layout()
savefig(fig, "true_sfh_and_mock_stochastic")
plt.show()

# %% [markdown]
# ### MAP → Ray Tracing
#
# The **Ray Tracing Sampler** (Behroozi 2025) propagates proposals along
# straight-line trajectories that refract at iso-probability surfaces
# (Snell's law).  It is ~250× more tolerant of gradient noise than HMC,
# making it the method of choice for $D \gtrsim 20$.
#
# > **Step-size note:** For $D \sim 137$, the viable step-size range is
# > narrow.  We use `step_size=0.005` with 200 leapfrog steps per
# > trajectory.

# %%
fitter_stoch = Fitter(model_stoch, mock_stoch.flux_obs, mock_stoch.noise,
                      data_type="photometry")

t0 = time.perf_counter()
result_map_s = fitter_stoch.run("map", n_steps=1000)
t_map_s = time.perf_counter() - t0
print(f"MAP: {t_map_s:.1f} s")

t0 = time.perf_counter()
result_rt = fitter_stoch.run("raytrace", init_from=result_map_s,
                             n_burnin=200, n_steps=2000,
                             step_size=0.005, n_leapfrog_steps=200)
t_rt = time.perf_counter() - t0
accept = result_rt.diagnostics.get("accept_rate_post_burnin", 0)
print(f"Ray Tracing: {t_rt:.1f} s  (acceptance: {accept:.0%})")

# %% [markdown]
# ### SFH recovery with 200 Myr inset
#
# The main panel shows the full lookback-time SFH.  The inset zooms into
# the last 200 Myr — this is where burstiness lives and where the GP
# field adds information beyond the smooth mean.  The posterior correctly
# captures the *amplitude* of recent SFR fluctuations even from
# broadband photometry alone, though the *phase* (exact timing of
# individual bursts) is typically unconstrained.

# %%
fig, ax = plt.subplots(figsize=(9, 5))

# Truth
ax.plot(t_gyr_s, sfr_full, color=COLORS["truth"], lw=2.5, label="Truth",
        zorder=10)
ax.plot(t_gyr_s, sfr_mean, color=COLORS["truth"], lw=1.5, ls=":",
        alpha=0.4, label="Secular mean", zorder=9)

# RT posterior
model_stoch.plot_sfh_posterior(result_rt, true_params=true_stoch,
                               color=COLORS["rt"], label="Ray Tracing",
                               ax=ax)

ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax.set_title("Stochastic SFH Recovery — Ray Tracing", fontsize=11)
ax.set_xlim(0, 13.5)
sfr_max = float(np.max(np.asarray(sfr_full)))
ax.set_ylim(0, max(3 * sfr_max, 30))
ax.legend(fontsize=9, loc="upper left")

# Inset: last 200 Myr with truth
ax_in = add_sfh_inset(ax, t_gyr_s, sfr_full, inset_range_myr=200,
                       color=COLORS["truth"], lw=2)

fig.tight_layout()
savefig(fig, "sfh_recovery_stochastic")
plt.show()

# %% [markdown]
# ### Corner plot: physical parameters
#
# We show only the physical parameters (SFH shape, PSD, dust,
# metallicity) — not the 128 GP latent variables $\xi_i$.  The PSD
# amplitude $\sigma_{\rm PS}$ is reasonably constrained even from
# broadband photometry; the coherence timescale $\tau_{\rm PS}$
# typically requires spectroscopy or population-level inference
# (see [T05](T05_hierarchical.ipynb)).

# %%
plot_corner_comparison(
    [result_rt],
    ["Ray Tracing"],
    colors=[COLORS["rt"]],
    truths=true_stoch,
)
plt.suptitle("Stochastic SEDModel — Ray Tracing Posterior", fontsize=12, y=1.02)
savefig(plt.gcf(), "corner_stochastic")
plt.show()

# %% [markdown]
# ---
# ## Summary
#
# | | Part A (parametric) | Part B (stochastic) |
# |---|---|---|
# | **SEDModel** | Double power law | Double power law + GP(PSD) |
# | **Free params** | 7 | ~137 |
# | **Inference** | NUTS (exact, gold standard) | Ray Tracing (exact, noise-tolerant) |
# | **SFH recovery** | Smooth envelope well-recovered | Burst amplitude recovered; phase unconstrained |
# | **Wall time** | ~seconds (NUTS) | ~minutes (Ray Tracing) |
#
# **Key takeaway:** tengri's stochastic mode recovers bursty SFH
# structure that parametric models miss entirely.  The 200 Myr inset
# reveals the regime where Hα and UV observations constrain the SFH —
# and where the PSD-governed GP adds real information.
#
# ## What's next
#
# | Tutorial | Topic |
# |----------|-------|
# | **[T01 — The IFT SEDModel](T01_ift_model.ipynb)** | PSD → GP → SFH: the mathematical framework |
# | **[T02 — Forward SEDModel](T02_forward_model.ipynb)** | SFH → SPS → dust → photometry pipeline |
# | **[T03 — Inference](T03_inference.ipynb)** | All five samplers: when each works and when it breaks |
# | **[T04 — Fitting](T04_fitting.ipynb)** | Photometry + spectroscopy fitting with diagnostics |
# | **[T05 — Hierarchical](T05_hierarchical.ipynb)** | Population-level PSD recovery |
# | **[T06 — Extending](T06_extending.ipynb)** | Custom PSD models, dust laws, SSP templates |
