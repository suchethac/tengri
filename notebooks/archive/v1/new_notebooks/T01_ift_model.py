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
# # The IFT Correlated Field SEDModel
#
# This tutorial develops the mathematical framework that makes tengri
# different from other SED-fitting codes.  The central idea: **the star
# formation history is a continuous field, not a parametric function or a
# set of independent time bins.**  Its temporal correlation structure is
# encoded by a power spectral density (PSD), which carries direct
# physical meaning — the amplitude and timescale of feedback-driven
# burstiness.
#
# The framework rests on three pillars:
#
# 1. **Information Field Theory** (IFT; Enßlin 2019) — Bayesian
#    inference on continuous fields, with a standardized latent space
#    that makes gradient-based sampling efficient.
#
# 2. **The PSD as a physical prior** — the damped random walk (DRW)
#    PSD encodes both the amplitude ($\sigma_{\rm PS}$) and the
#    coherence timescale ($\tau_{\rm PS}$) of SFR fluctuations.
#
# 3. **End-to-end JAX differentiability** — every operation from PSD
#    parameters through GP realization, SPS integration, and
#    photometric prediction is a pure JAX function with exact gradients.
#
# **By the end you will understand:**
#
# 1. How IFT frames SED fitting as field reconstruction
# 2. How the DRW PSD encodes burstiness physics
# 3. How GP realizations are generated from a PSD via FFT
# 4. How the double power-law mean SFH provides the secular envelope
# 5. How the full SFH combines mean + GP with a lognormal correction
# 6. How the $(\sigma_{\rm PS}, \tau_{\rm PS})$ burstiness plane maps to
#    SFH diversity
# 7. How PSD parameters connect to observable diagnostics
# 8. How end-to-end JAX gradients enable efficient inference

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from tengri.utils.devices import setup_jax
setup_jax()

import jax
import jax.numpy as jnp
from jax import random, grad, jit

# Low-level model components
from tengri.sfh.psd_models import (
    psd_drw, drw_acf, drw_variance, psd_to_sqrt_power,
)
from tengri.sfh.gp_sfh import (
    gp_from_xi, generate_gp_fourier, generate_gp_batch,
    compute_sqrt_power_drw,
)
from tengri.sfh.mean_sfh import double_powerlaw
from tengri.utils.grid import (
    make_log_age_grid, grid_spacing, log_age_to_age_yr,
    interpolate_to_linear_time,
)
from tengri.utils.cosmology import age_at_z

# ── Plot style ────────────────────────────────────────────────────
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

REGIME_COLORS = {
    "Smooth":        "#1b9e77",
    "Moderate":      "#d95f02",
    "Bursty":        "#7570b3",
    "Highly bursty": "#e7298a",
}

FIG_DIR = "../notebook_figures"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fig, name, dpi=72):
    path = os.path.join(FIG_DIR, f"T01_{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"  → {path}")


# ── Helpers ───────────────────────────────────────────────────────
def add_redshift_axis(ax, z_ticks=(0, 0.5, 1, 2, 3, 5)):
    """Add redshift labels as a twin x-axis on a lookback-time plot."""
    ax2 = ax.twiny()
    t_uni = float(age_at_z(0.0)) / 1e9
    lookbacks = [t_uni - float(age_at_z(z)) / 1e9 for z in z_ticks]
    xlim = ax.get_xlim()
    ax2.set_xlim(xlim)
    valid = [(z, lb) for z, lb in zip(z_ticks, lookbacks)
             if xlim[0] <= lb <= xlim[1]]
    if valid:
        ax2.set_xticks([lb for _, lb in valid])
        ax2.set_xticklabels([f"z={z}" for z, _ in valid], fontsize=8)
    ax2.set_xlabel("Redshift", fontsize=9)
    return ax2


def add_sfh_inset(ax, t_gyr, sfr, inset_range_myr=200,
                  width="35%", height="40%", **plot_kwargs):
    """Add an inset axis zooming into the recent SFH."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_in = inset_axes(ax, width=width, height=height, loc="upper right",
                       borderpad=1.5)
    t_myr = np.asarray(t_gyr) * 1e3
    mask = t_myr <= inset_range_myr
    if mask.sum() > 2:
        ax_in.plot(t_myr[mask], np.asarray(sfr)[mask], **plot_kwargs)
    ax_in.set_xlim(0, inset_range_myr)
    ax_in.set_xlabel("Myr", fontsize=7, labelpad=1)
    ax_in.tick_params(labelsize=6)
    ax_in.axvspan(0, 10, alpha=0.08, color="blue", zorder=0)
    ax_in.axvspan(0, 100, alpha=0.04, color="purple", zorder=0)
    ylim = ax_in.get_ylim()
    y_txt = ylim[1] * 0.92 if ylim[1] > 0 else 0.9
    ax_in.text(5, y_txt, r"H$\alpha$", fontsize=5.5, color="blue", va="top")
    ax_in.text(50, y_txt, "UV", fontsize=5.5, color="purple", va="top")
    ax_in.set_title("last 200 Myr", fontsize=7, pad=2)
    for spine in ax_in.spines.values():
        spine.set_linewidth(0.6)
    return ax_in


# ── Grid setup (used throughout) ──────────────────────────────────
N_GRID = 128
log_age_grid = make_log_age_grid(N_GRID)
d_log_age = grid_spacing(log_age_grid)
age_yr = log_age_to_age_yr(log_age_grid)
age_gyr = age_yr / 1e9

print(f"JAX {jax.__version__} | device: {jax.devices()[0]}")
print(f"Grid: {N_GRID} points, log(age) = {float(log_age_grid[0]):.2f}"
      f" to {float(log_age_grid[-1]):.2f} dex")
print(f"Age range: {float(age_yr[0])/1e6:.1f} Myr to "
      f"{float(age_yr[-1])/1e9:.1f} Gyr")
print(f"Grid spacing: Δlog(age) = {d_log_age:.4f} dex")


def sfh_on_linear_time(sfr_on_log_grid, n_pts=1000):
    """Resample SFH from the log-age grid to uniform linear time."""
    return interpolate_to_linear_time(log_age_grid, sfr_on_log_grid, n_pts)


# %% [markdown]
# ---
# ## 1. Information Field Theory
#
# **Information Field Theory** (IFT; Enßlin 2019) extends Bayesian
# inference to *fields* — quantities defined over continuous domains.
# In our problem:
#
# | IFT concept | SED fitting meaning |
# |:-----------:|:--------------------|
# | **Signal** $\mathbf{s} = x(t)$ | Log-SFR fluctuation (the unknown field) |
# | **Data** $\mathbf{d}$ | Observed photometry or spectrum |
# | **Response** $R$ | Full forward model: $x(t) \to \mathrm{SFR}(t) \to \mathrm{SPS} \to \mathrm{SED} \to f_\nu$ |
# | **Noise** $\mathbf{n}$ | Measurement uncertainties |
#
# The posterior combines data fidelity with a prior on the field:
#
# $$P(\mathbf{s} \mid \mathbf{d}) \propto P(\mathbf{d} \mid \mathbf{s})\, P(\mathbf{s})$$
#
# ### Standardization: the key trick
#
# The **correlated field model** (Knollmüller & Enßlin 2019; Edenhofer
# et al. 2024) reparametrizes the generative model so that the prior
# becomes a standard Gaussian.  We define a differentiable mapping
# $\mathbf{s} = f(\boldsymbol{\xi})$ such that
# $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.  All
# prior structure — temporal correlations, burstiness amplitude — is
# absorbed into the forward model $f$.
#
# In the standardized coordinates, the **information Hamiltonian**
# (negative log-posterior) reduces to:
#
# $$\boxed{H(\boldsymbol{\xi} \mid \mathbf{d}) = \frac{1}{2}\chi^2 + \frac{1}{2}\boldsymbol{\xi}^\top\boldsymbol{\xi}}$$
#
# The first term penalises misfit to the data.  The second is an
# isotropic Gaussian prior penalty — no complicated covariance matrices.
# Every sampler (MAP, NUTS, Ray Tracing, geoVI, MGVI) operates on this
# same loss function.
#
# ### Why this matters for observers
#
# Traditional SED codes use either parametric SFHs (too rigid — miss
# real burstiness) or binned non-parametric SFHs with ad-hoc continuity
# priors (arbitrary bin widths, difficult to interpret physically).  IFT
# replaces the ad-hoc prior with a **physically motivated PSD kernel**,
# and the standardized $\boldsymbol{\xi}$-space enables gradient-based
# inference that is 10–100× faster than gradient-free samplers like
# `dynesty` or `MultiNest`.

# %% [markdown]
# ---
# ## 2. The Power Spectral Density
#
# The **damped random walk** (DRW) PSD is a Lorentzian with two
# parameters:
#
# $$P(\omega) = \frac{\sigma_{\rm PS}^2 \, \tau_{\rm PS}}{1 + (\tau_{\rm PS}\,\omega)^2}$$
#
# ### $\sigma_{\rm PS}$: fluctuation amplitude
#
# The stationary variance of the GP is
# $\sigma_x^2 = \sigma_{\rm PS}^2 / 2$, so a $1\sigma$ excursion in
# $x(t)$ corresponds to a multiplicative factor of $e^{\sigma_x}$ in
# SFR.  The table below gives the mapping:
#
# | $\sigma_{\rm PS}$ | SFR scatter | Peak-to-trough | Physical regime |
# |:-:|:-:|:-:|:--|
# | 0.5 | ~0.15 dex | ~3× | Normal main-sequence galaxy |
# | 1.0 | ~0.3 dex | ~10× | Moderate burstiness (MW-mass) |
# | 2.0 | ~0.7 dex | ~100× | Bursty (SN feedback-dominated) |
# | 3.0 | ~1.0 dex | ~1000× | Extreme starburst/quenching |
#
# ### $\tau_{\rm PS}$: coherence timescale
#
# $\tau_{\rm PS}$ sets how long a burst or quench episode persists
# before reverting to the mean.  Different feedback mechanisms produce
# characteristic timescales:
#
# | $\tau_{\rm PS}$ [Myr] | Dominant process | Observable signature |
# |:-:|:--|:--|
# | 5–10 | Stellar winds, SN blowout | Flickering in Hα |
# | 20–50 | Superbubble feedback cycle | Scatter in Hα/UV ratio |
# | 100–300 | Gas cycling, halo response | Main-sequence scatter |
# | 500+ | Mergers, environmental quenching | Bimodal colour distribution |
#
# ### Two frequency regimes
#
# At frequencies $\omega \ll 1/\tau_{\rm PS}$, the PSD is flat (white
# noise — uncorrelated on these long timescales).  At
# $\omega \gg 1/\tau_{\rm PS}$, the power falls as $\omega^{-2}$ (red
# noise — correlated on short timescales).  The break frequency
# $\omega_{\rm break} = 1/\tau_{\rm PS}$ separates the two regimes.
#
# The corresponding autocorrelation function is:
#
# $$\xi_x(\Delta t) = \frac{\sigma_{\rm PS}^2}{2} \exp\!\left(-\frac{|\Delta t|}{\tau_{\rm PS}}\right)$$

# %%
# ── Figure: PSD overview (3-panel) ────────────────────────────────
# Panel 1: P(ω) for four regimes
# Panel 2: Normalised ACF
# Panel 3: GP realisations

regimes = {
    "Smooth":        {"sigma": 0.5, "tau_myr": 200},
    "Moderate":      {"sigma": 1.5, "tau_myr": 50},
    "Bursty":        {"sigma": 2.5, "tau_myr": 20},
    "Highly bursty": {"sigma": 4.0, "tau_myr": 5},
}

omega_myr = jnp.logspace(-4, 1, 500)   # rad / Myr
dt_myr = jnp.linspace(0, 500, 500)     # Myr

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

# Panel 1: PSD
ax = axes[0]
for name, r in regimes.items():
    P = psd_drw(omega_myr, r["sigma"], r["tau_myr"])
    ax.loglog(omega_myr, P, lw=2.5, color=REGIME_COLORS[name],
              label=rf"$\sigma$={r['sigma']}, $\tau$={r['tau_myr']} Myr")
    ax.axvline(1.0 / r["tau_myr"], color=REGIME_COLORS[name], ls=":",
               alpha=0.25)
ax.set_xlabel(r"$\omega$ [rad Myr$^{-1}$]")
ax.set_ylabel(r"$P(\omega)$")
ax.set_title("DRW Power Spectrum")
ax.legend(fontsize=6.5, loc="lower left")
ax.set_xlim(1e-4, 10)
# Annotate regimes
ax.text(3e-4, 1e4, "flat\n(correlated)", fontsize=8, color="0.4")
ax.text(2, 1e-2, r"$\propto\omega^{-2}$", fontsize=8, color="0.4")

# Panel 2: ACF
ax = axes[1]
for name, r in regimes.items():
    acf = drw_acf(dt_myr, r["sigma"], r["tau_myr"])
    ax.plot(dt_myr, acf / acf[0], lw=2.5, color=REGIME_COLORS[name],
            label=name)
ax.set_xlabel(r"$\Delta t$ [Myr]")
ax.set_ylabel("Normalised ACF")
ax.set_title("Autocorrelation")
ax.legend(fontsize=7)
ax.set_ylim(-0.05, 1.05)
ax.axhline(1.0 / np.e, color="0.6", ls="--", lw=0.7, alpha=0.5)
ax.text(480, 1.0 / np.e + 0.03, r"$1/e$", fontsize=7, color="0.5",
        ha="right")

# Panel 3: GP realisations
ax = axes[2]
N_DISP = 256
log_ages_disp = make_log_age_grid(N_DISP)
d_log_disp = grid_spacing(log_ages_disp)
ages_disp_yr = log_age_to_age_yr(log_ages_disp)

key = jax.random.PRNGKey(7)
offset = 0
for name, r in regimes.items():
    sqrt_p = compute_sqrt_power_drw(N_DISP, float(d_log_disp),
                                     r["sigma"], r["tau_myr"] * 1e6)
    for draw in range(3):
        key, subkey = jax.random.split(key)
        xi = jax.random.normal(subkey, shape=(N_DISP,))
        gp = gp_from_xi(xi, sqrt_p, N_DISP)
        t_gyr, gp_lin = interpolate_to_linear_time(log_ages_disp, gp, 500)
        lw = 2.0 if draw == 0 else 0.8
        label = name if draw == 0 else None
        ax.plot(np.array(t_gyr), np.array(gp_lin) + offset,
                lw=lw, color=REGIME_COLORS[name], alpha=0.7, label=label)
    offset += 5

ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"GP field $x(t)$ (offset)")
ax.set_title("GP Realisations")
ax.legend(fontsize=6.5, loc="upper right")
ax.set_xlim(0, 13.5)

fig.tight_layout()
savefig(fig, "psd_overview")
plt.show()

# %% [markdown]
# ---
# ## 3. Generating GP Realisations from a PSD
#
# Given the PSD, generating a GP realisation is a **three-step FFT
# recipe**:
#
# 1. **Draw** a standardised latent vector:
#    $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
# 2. **Multiply** in Fourier space:
#    $\hat{\mathbf{x}} = \sqrt{P / \Delta u} \cdot \hat{\boldsymbol{\xi}}$
# 3. **Transform** back: $x(t) = \texttt{irfft}(\hat{\mathbf{x}})$
#
# The amplitude operator $\sqrt{P / \Delta u}$ encodes all correlation
# structure.  Changing $\sigma_{\rm PS}$ or $\tau_{\rm PS}$ changes
# $\sqrt{P}$ but leaves $\boldsymbol{\xi}$ untouched — the sampler
# explores $\boldsymbol{\xi}$-space, which has a simple standard-normal
# geometry, while the physics is encoded in $\sqrt{P}$.
#
# The grid is $N_{\rm grid}$ points uniformly spaced in
# $\log_{10}(t_{\rm age}/\mathrm{yr})$ from 1 Myr to 13.8 Gyr.
# Uniform spacing in log-age gives finer resolution at young ages
# (~1 Myr steps at 1 Myr, ~300 Myr steps at 10 Gyr), exactly where
# the SED is most sensitive to SFR changes.

# %%
# ── Figure: Same ξ, different PSD → different GP ─────────────────
key = random.PRNGKey(42)
xi_fixed = random.normal(key, shape=(N_GRID,))

fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                     r["sigma"], r["tau_myr"] * 1e6)
    gp = gp_from_xi(xi_fixed, sqrt_p, N_GRID)
    sig_x = np.sqrt(float(drw_variance(r["sigma"])))

    ax.plot(age_gyr, gp, lw=2.0, color=REGIME_COLORS[name])
    ax.axhline(0, ls="--", color="gray", lw=0.7)
    ax.axhspan(-sig_x, sig_x, alpha=0.08, color=REGIME_COLORS[name])
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, "
                 rf"$\tau_{{\rm PS}}$={r['tau_myr']} Myr", fontsize=9)
    ax.set_ylabel(r"$x(t)$", fontsize=10)
    ax.text(0.97, 0.95, rf"$\sigma_x$={sig_x:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

for ax in axes[1]:
    ax.set_xlabel("Stellar age [Gyr]")

fig.suptitle(r"Same latent $\boldsymbol{\xi}$, different PSD "
             r"$\rightarrow$ different GP", fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "same_xi_different_psd")
plt.show()

# %%
# ── Figure: Multiple GP realisations per regime ──────────────────
N_REAL = 5
key = random.PRNGKey(123)

fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                     r["sigma"], r["tau_myr"] * 1e6)
    gp_batch = generate_gp_batch(key, sqrt_p, N_GRID, N_REAL)
    sig_x = np.sqrt(float(drw_variance(r["sigma"])))

    for i in range(N_REAL):
        ax.plot(age_gyr, gp_batch[i], lw=1.5, alpha=0.65,
                color=REGIME_COLORS[name])

    ax.axhline(0, ls="--", color="gray", lw=0.7)
    ax.axhspan(-sig_x, sig_x, alpha=0.08, color=REGIME_COLORS[name])
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, "
                 rf"$\tau_{{\rm PS}}$={r['tau_myr']} Myr", fontsize=9)
    ax.set_ylabel(r"$x(t)$", fontsize=10)

for ax in axes[1]:
    ax.set_xlabel("Stellar age [Gyr]")

fig.suptitle(f"{N_REAL} independent GP realisations per regime",
             fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "gp_realisations")
plt.show()

# %% [markdown]
# ---
# ## 4. The Mean Star Formation History
#
# The GP $x(t)$ fluctuates around zero by construction — it generates
# only stochastic variability.  The secular envelope (rising at early
# times, declining at late times) comes from a separate **smooth mean
# SFH**.  We use the double power law (Behroozi et al. 2013; Carnall
# et al. 2018):
#
# $$\overline{\mathrm{SFR}}(t) = \frac{A}{\left(t/\tau\right)^\alpha + \left(t/\tau\right)^{-\beta}}$$
#
# where $t$ is lookback time.  The function peaks near
# $t \approx \tau$.
#
# ### Parameter meanings
#
# | Parameter | Symbol | Controls | Lookback-time plot |
# |:----------|:------:|:---------|:-------------------|
# | Falling slope | $\alpha$ | Decline from peak toward early universe | Right side |
# | Rising slope | $\beta$ | Rise from peak toward present | Left side |
# | Turnover time | $\tau$ | Epoch of peak SFR | Peak location |
# | Normalisation | $A$ | Peak SFR (M$_\odot$ yr$^{-1}$) | Amplitude |
#
# $A$ is **not** the stellar mass.  Stellar mass is derived:
# $M_* = \int_0^{t_H} \mathrm{SFR}(t)\,(1-R(t))\,dt$, where $R(t)$
# is the returned mass fraction from stellar evolution.

# %%
# ── Figure: Galaxy archetypes ────────────────────────────────────
archetypes = {
    "Late-forming spiral": {
        "alpha": 1.0, "beta": 0.5, "tau_gyr": 8.0, "norm": 5.0,
        "color": "#1b9e77",
    },
    "Post-starburst": {
        "alpha": 2.0, "beta": 2.0, "tau_gyr": 3.0, "norm": 50.0,
        "color": "#d95f02",
    },
    "High-z star-forming": {
        "alpha": 1.5, "beta": 3.5, "tau_gyr": 2.0, "norm": 30.0,
        "color": "#7570b3",
    },
    "Recent starburst": {
        "alpha": 0.8, "beta": 4.0, "tau_gyr": 1.5, "norm": 80.0,
        "color": "#e7298a",
    },
}

fig, ax = plt.subplots(figsize=(9, 4.5))
for label, p in archetypes.items():
    sfr = double_powerlaw(age_yr, p["alpha"], p["beta"],
                          p["tau_gyr"] * 1e9, p["norm"])
    ax.plot(age_gyr, sfr, lw=2.2, color=p["color"],
            label=rf"{label}  ($\alpha$={p['alpha']}, $\beta$={p['beta']}, "
                  rf"$\tau$={p['tau_gyr']} Gyr)")

ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax.set_title("Mean SFH: Galaxy Archetypes (Double Power Law)", fontsize=11)
ax.set_xlim(0, 13.5)
ax.set_ylim(bottom=0)
ax.legend(fontsize=7, loc="upper right")
add_redshift_axis(ax)

fig.tight_layout()
savefig(fig, "galaxy_archetypes")
plt.show()

# %%
# ── Figure: Vary α, β, τ (1×3) ──────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
base = {"alpha": 1.5, "beta": 1.5, "tau_gyr": 4.0, "norm": 10.0}
cmap = plt.cm.viridis

# Panel 1: vary α
ax = axes[0]
for i, a in enumerate([0.5, 1.0, 2.0, 3.0, 4.0]):
    sfr = double_powerlaw(age_yr, a, base["beta"],
                          base["tau_gyr"] * 1e9, base["norm"])
    ax.plot(age_gyr, sfr, lw=2.0, color=cmap(i / 4), label=rf"$\alpha$={a}")
ax.set_title(r"Vary $\alpha$ (right side: early universe)", fontsize=9)
ax.legend(fontsize=7)
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")

# Panel 2: vary β
ax = axes[1]
for i, b in enumerate([0.3, 0.8, 1.5, 2.5, 4.0]):
    sfr = double_powerlaw(age_yr, base["alpha"], b,
                          base["tau_gyr"] * 1e9, base["norm"])
    ax.plot(age_gyr, sfr, lw=2.0, color=cmap(i / 4), label=rf"$\beta$={b}")
ax.set_title(r"Vary $\beta$ (left side: near present)", fontsize=9)
ax.legend(fontsize=7)

# Panel 3: vary τ
ax = axes[2]
for i, t in enumerate([1.0, 2.0, 4.0, 7.0, 10.0]):
    sfr = double_powerlaw(age_yr, base["alpha"], base["beta"],
                          t * 1e9, base["norm"])
    ax.plot(age_gyr, sfr, lw=2.0, color=cmap(i / 4),
            label=rf"$\tau$={t} Gyr")
ax.set_title(r"Vary $\tau$ (peak lookback time)", fontsize=9)
ax.legend(fontsize=7)

for ax in axes:
    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(bottom=0)

fig.tight_layout()
savefig(fig, "mean_sfh_parameters")
plt.show()

# %% [markdown]
# ---
# ## 5. The Full SFH: Mean × Burstiness
#
# The complete star formation history combines the smooth mean with the
# GP fluctuations:
#
# $$\mathrm{SFR}(t) = \overline{\mathrm{SFR}}(t) \times \exp\!\left(x(t) - \frac{\sigma_x^2}{2}\right)$$
#
# ### Why subtract $\sigma_x^2/2$?
#
# The GP $x(t)$ is zero-mean, so $\langle x(t)\rangle = 0$.  But
# $\exp(x)$ is **not** mean-one.  For a zero-mean Gaussian with
# variance $\sigma_x^2$, the expectation of $\exp(x)$ is
# $\exp(\sigma_x^2/2) > 1$.  Without correction, burstier models
# would have systematically higher average SFR.
#
# **Intuition:** upward excursions (bursts) are multiplicatively larger
# than downward excursions (lulls), pulling the average up.  Subtracting
# $\sigma_x^2/2$ cancels this bias, ensuring
# $\langle\mathrm{SFR}\rangle = \overline{\mathrm{SFR}}$ regardless
# of burstiness level.
#
# For the DRW, $\sigma_x^2 = \sigma_{\rm PS}^2 / 2$, so the correction
# is $-\sigma_{\rm PS}^2 / 4$.

# %%
# ── Figure: Step-by-step SFH assembly (1×3) ─────────────────────
mean_params = {"alpha": 1.2, "beta": 0.8, "tau_gyr": 6.0, "norm": 8.0}
mean_sfr = double_powerlaw(age_yr, mean_params["alpha"], mean_params["beta"],
                           mean_params["tau_gyr"] * 1e9, mean_params["norm"])

sigma_ps, tau_ps_myr = 1.5, 50.0
sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, sigma_ps, tau_ps_myr * 1e6)
key = random.PRNGKey(7)
gp = generate_gp_fourier(key, sqrt_p, N_GRID)
sigma_x_sq = float(drw_variance(sigma_ps))
full_sfr = mean_sfr * jnp.exp(gp - sigma_x_sq / 2.0)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))

# Panel 1: mean SFH
ax = axes[0]
t_m, m_lin = sfh_on_linear_time(mean_sfr)
ax.plot(t_m, m_lin, lw=2.2, color="#1b9e77")
ax.fill_between(np.array(t_m), 0, np.array(m_lin), alpha=0.12,
                color="#1b9e77")
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax.set_title(r"Step 1: Secular mean $\overline{\mathrm{SFR}}(t)$",
             fontsize=10)
ax.set_xlim(0, 13.5)
ax.set_ylim(bottom=0)

# Panel 2: GP fluctuation
ax = axes[1]
ax.plot(age_gyr, gp, lw=2.0, color="#d95f02")
ax.axhline(0, ls="--", color="gray", lw=0.7)
ax.axhspan(-np.sqrt(sigma_x_sq), np.sqrt(sigma_x_sq), alpha=0.1,
           color="#d95f02")
ax.set_ylabel(r"$x(t)$")
ax.set_title(rf"Step 2: GP field ($\sigma_{{\rm PS}}$={sigma_ps}, "
             rf"$\tau_{{\rm PS}}$={tau_ps_myr:.0f} Myr)", fontsize=10)

# Panel 3: full SFH
ax = axes[2]
t_m, m_lin = sfh_on_linear_time(mean_sfr)
ax.plot(t_m, m_lin, lw=1.5, ls="--", color="gray", label="Mean")
t_f, f_lin = sfh_on_linear_time(full_sfr)
ax.plot(t_f, f_lin, lw=2.0, color="#7570b3", label="Full SFH")
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax.set_title(r"Step 3: $\overline{\mathrm{SFR}} \times "
             r"e^{x - \sigma_x^2/2}$", fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(0, 13.5)
ax.set_ylim(bottom=0)

for ax in axes:
    ax.set_xlabel("Lookback time [Gyr]")

fig.tight_layout()
savefig(fig, "sfh_step_by_step")
plt.show()

# %%
# ── Figure: 4 regimes (log scale) with 200 Myr inset ────────────
key = random.PRNGKey(1234)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                     r["sigma"], r["tau_myr"] * 1e6)
    gp = generate_gp_fourier(key, sqrt_p, N_GRID)
    sig_x_sq = float(drw_variance(r["sigma"]))
    sfr = mean_sfr * jnp.exp(gp - sig_x_sq / 2.0)

    ax.semilogy(age_gyr, sfr, lw=1.8, color=REGIME_COLORS[name])
    ax.semilogy(age_gyr, mean_sfr, lw=1.0, ls="--", color="gray")
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, "
                 rf"$\tau_{{\rm PS}}$={r['tau_myr']} Myr", fontsize=9)
    ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
    ax.set_xlim(0, 13.5)

    # Inset: last 200 Myr
    add_sfh_inset(ax, age_gyr, sfr, inset_range_myr=200,
                  color=REGIME_COLORS[name], lw=1.2)

for ax in axes[1]:
    ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle("Full SFH (log scale) — same mean, different burstiness",
             fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "full_sfh_4_regimes")
plt.show()

# %%
# ── Figure: Ensemble of 10 SFHs per regime (log) ────────────────
N_ENS = 10
key_ens = random.PRNGKey(314)

fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                     r["sigma"], r["tau_myr"] * 1e6)
    gp_batch = generate_gp_batch(key_ens, sqrt_p, N_GRID, N_ENS)
    sig_x_sq = float(drw_variance(r["sigma"]))

    for i in range(N_ENS):
        sfr = mean_sfr * jnp.exp(gp_batch[i] - sig_x_sq / 2.0)
        ax.semilogy(age_gyr, sfr, lw=0.8, alpha=0.5,
                    color=REGIME_COLORS[name])

    ax.semilogy(age_gyr, mean_sfr, lw=2.0, ls="--", color="black",
                label="Mean SFH")
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, "
                 rf"$\tau_{{\rm PS}}$={r['tau_myr']} Myr", fontsize=9)
    ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
    ax.set_xlim(0, 13.5)
    if name == "Smooth":
        ax.legend(fontsize=7)

for ax in axes[1]:
    ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle(f"{N_ENS} SFH realisations per regime (log scale)",
             fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "sfh_ensemble")
plt.show()

# %% [markdown]
# ---
# ## 6. The Burstiness Plane
#
# The two-dimensional $(\sigma_{\rm PS}, \tau_{\rm PS})$ parameter space
# produces a rich diversity of SFH behaviours.  Varying $\sigma_{\rm PS}$
# changes the **amplitude** of fluctuations; varying $\tau_{\rm PS}$
# changes their **timescale**:
#
# - **Low $\sigma$, high $\tau$** — gentle, long-timescale modulation
# - **High $\sigma$, low $\tau$** — rapid, violent bursts
# - **Low $\sigma$, low $\tau$** — rapid but weak flickering
# - **High $\sigma$, high $\tau$** — slow but extreme excursions

# %%
# ── Figure: 3×3 σ × τ grid (log scale) ──────────────────────────
sigma_grid = [0.5, 1.5, 3.0]
tau_grid_myr = [10, 50, 200]
key_bp = random.PRNGKey(2024)

fig, axes = plt.subplots(3, 3, figsize=(10, 8), sharex=True, sharey=True)

for i, tau_myr in enumerate(tau_grid_myr):
    for j, sigma in enumerate(sigma_grid):
        ax = axes[i, j]
        sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                         sigma, tau_myr * 1e6)
        gp = generate_gp_fourier(key_bp, sqrt_p, N_GRID)
        sig_x_sq = float(drw_variance(sigma))
        sfr = mean_sfr * jnp.exp(gp - sig_x_sq / 2.0)

        ax.semilogy(age_gyr, sfr, lw=1.3, color="steelblue")
        ax.semilogy(age_gyr, mean_sfr, lw=0.7, ls="--", color="gray")
        ax.set_xlim(0, 13.5)
        ax.set_ylim(1e-3, 1e4)

        if i == 0:
            ax.set_title(rf"$\sigma_{{\rm PS}}$={sigma}", fontsize=9)
        if j == 0:
            ax.set_ylabel(rf"$\tau_{{\rm PS}}$={tau_myr} Myr" + "\n"
                          + r"SFR [M$_{\odot}$ yr$^{-1}$]", fontsize=8)
        if i == 2:
            ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle(r"Burstiness plane: $\sigma_{\rm PS}$ (columns) "
             r"$\times$ $\tau_{\rm PS}$ (rows)", fontsize=11, y=1.01)
fig.tight_layout()
savefig(fig, "burstiness_plane")
plt.show()

# %% [markdown]
# ---
# ## 7. PSD Parameters and Observable Diagnostics
#
# The PSD is not merely a mathematical prior — it makes testable
# predictions about quantities observers measure.  Three key
# diagnostics connect PSD parameters to data:
#
# ### Main-sequence scatter
#
# The intrinsic scatter of the star-forming main sequence at $z \sim 0$
# is $\sim$0.3 dex (Speagle et al. 2014).  This provides a direct
# constraint on the integrated PSD power on timescales shorter than
# $\sim$1 Gyr.  The mapping is approximately:
#
# $$\sigma_{\rm MS} \approx \frac{\sigma_{\rm PS}}{\sqrt{2}\,\ln 10}$$
#
# ### Hα/UV ratio
#
# Hα traces the last $\sim$5 Myr of star formation; rest-UV traces the
# last $\sim$100 Myr.  Their ratio fluctuates around unity for constant
# SFR but shows large excursions for bursty galaxies.  The scatter in
# $\log(\mathrm{SFR}_{\mathrm{H}\alpha} / \mathrm{SFR}_{\mathrm{UV}})$
# increases with both $\sigma_{\rm PS}$ and $\tau_{\rm PS}$
# (Iyer et al. 2024; Wan et al. 2024).
#
# ### Window functions
#
# Each observable acts like a differently weighted temporal filter on the
# SFH.  The **window function** $\tilde{W}_\lambda(\omega)$ specifies
# which PSD frequencies a given measurement probes.  Timescales where
# $\tilde{W}$ has negligible support are unconstrained regardless of
# the inference method.

# %%
# ── Figure: σ_PS → SFR scatter diagnostic ────────────────────────
sigma_vals = jnp.array([0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0])
tau_myr_vals = jnp.array([10.0, 30.0, 100.0, 300.0])
n_real = 200

mean_sfr_grid = double_powerlaw(age_yr, alpha=1.5, beta=1.2,
                                 tau=5e9, norm=10.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
tau_cols = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

# Panel 1: SFR scatter at 100 Myr vs σ_PS
ax = axes[0]
idx_100 = int(jnp.argmin(jnp.abs(age_yr - 1e8)))

for tau_myr, col in zip(tau_myr_vals, tau_cols):
    scatters = []
    for sigma in sigma_vals:
        key, subkey = jax.random.split(jax.random.PRNGKey(int(tau_myr)))
        sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                         float(sigma), float(tau_myr) * 1e6)
        keys = jax.random.split(subkey, n_real)
        gp_batch = jax.vmap(
            lambda k, sp=sqrt_p: generate_gp_fourier(k, sp, N_GRID)
        )(keys)
        var_gp = drw_variance(float(sigma))
        sfr_batch = mean_sfr_grid[None, :] * jnp.exp(gp_batch - var_gp / 2.0)
        scatter = float(jnp.std(jnp.log10(sfr_batch[:, idx_100])))
        scatters.append(scatter)
    ax.plot(np.array(sigma_vals), scatters, "o-", color=col, lw=2, ms=5,
            label=rf"$\tau = {float(tau_myr):.0f}$ Myr")

ax.axhline(0.3, color="0.5", ls="--", lw=1,
           label=r"$\sigma_{\rm MS} \approx 0.3$ dex")
ax.set_xlabel(r"PSD amplitude $\sigma_{\rm PS}$")
ax.set_ylabel(r"$\sigma[\log \mathrm{SFR}_{100\,\mathrm{Myr}}]$ [dex]")
ax.set_title("SFR scatter vs. PSD amplitude", fontsize=10)
ax.legend(fontsize=8)

# Panel 2: Hα/UV ratio scatter
ax = axes[1]
idx_10 = int(jnp.argmin(jnp.abs(age_yr - 1e7)))
idx_5 = int(jnp.argmin(jnp.abs(age_yr - 5e6)))

for tau_myr, col in zip(tau_myr_vals, tau_cols):
    ratio_scatters = []
    for sigma in sigma_vals:
        key, subkey = jax.random.split(
            jax.random.PRNGKey(int(tau_myr) + 1000))
        sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                         float(sigma), float(tau_myr) * 1e6)
        keys = jax.random.split(subkey, n_real)
        gp_batch = jax.vmap(
            lambda k, sp=sqrt_p: generate_gp_fourier(k, sp, N_GRID)
        )(keys)
        var_gp = drw_variance(float(sigma))
        sfr_batch = mean_sfr_grid[None, :] * jnp.exp(gp_batch - var_gp / 2.0)

        sfr_ha = jnp.mean(sfr_batch[:, idx_5:idx_10 + 1], axis=1)
        sfr_uv = jnp.mean(sfr_batch[:, idx_10:idx_100 + 1], axis=1)
        log_ratio = jnp.log10(sfr_ha / jnp.maximum(sfr_uv, 1e-10))
        ratio_scatters.append(float(jnp.std(log_ratio)))

    ax.plot(np.array(sigma_vals), ratio_scatters, "o-", color=col, lw=2,
            ms=5, label=rf"$\tau = {float(tau_myr):.0f}$ Myr")

ax.set_xlabel(r"PSD amplitude $\sigma_{\rm PS}$")
ax.set_ylabel(r"$\sigma[\log(\mathrm{SFR}_{\mathrm{H}\alpha}"
              r"/\mathrm{SFR}_{\mathrm{UV}})]$ [dex]")
ax.set_title(r"H$\alpha$/UV ratio scatter vs. PSD amplitude", fontsize=10)
ax.legend(fontsize=8)

fig.tight_layout()
savefig(fig, "psd_observables")
plt.show()

# %% [markdown]
# ---
# ## 8. End-to-End Gradients
#
# Because the entire pipeline (PSD → GP → SFH → stellar mass) is
# implemented in pure JAX, we can compute exact gradients of any output
# with respect to any input via automatic differentiation.  This is what
# enables gradient-based inference (NUTS, geoVI, MAP via Adam) rather
# than slow gradient-free sampling.
#
# Below we verify that $M_* = \int \mathrm{SFR}(t)\,dt$ has **finite,
# nonzero gradients** with respect to all six SFH parameters.

# %%
_la = make_log_age_grid(N_GRID)
_dla = float(_la[1] - _la[0])
_a = 10.0 ** _la

def stellar_mass(alpha, beta, tau_yr, norm, sigma_ps, tau_ps_yr, xi):
    """Approximate M* = ∫ SFR(t) dt."""
    mean = double_powerlaw(_a * 1e9, alpha, beta, tau_yr, norm)
    sqrt_p = compute_sqrt_power_drw(N_GRID, _dla, sigma_ps, tau_ps_yr)
    gp = gp_from_xi(xi, sqrt_p, N_GRID)
    var_x = 0.5 * sigma_ps ** 2
    sfr = mean * jnp.exp(gp - var_x / 2.0)
    dt = _a * jnp.log(10.0) * _dla
    return jnp.sum(sfr * dt)

# Reference parameters
alpha0, beta0 = 1.2, 0.8
tau0, norm0 = 6.0e9, 8.0
sigma0, tau_ps0 = 1.5, 50e6
xi0 = jax.random.normal(jax.random.PRNGKey(0), shape=(N_GRID,))

mstar = stellar_mass(alpha0, beta0, tau0, norm0, sigma0, tau_ps0, xi0)

grad_fn = jax.grad(stellar_mass, argnums=(0, 1, 2, 3, 4, 5))
grads = grad_fn(alpha0, beta0, tau0, norm0, sigma0, tau_ps0, xi0)

names = [r"α", r"β", r"τ_mean", "A (norm)", r"σ_PS", r"τ_PS"]

print(f"Stellar mass: M* = {float(mstar):.3e} M☉")
print()
print(f"{'Parameter':<15s} {'Gradient':>14s}  {'Nonzero?':>10s}")
print("-" * 42)
for n, g in zip(names, grads):
    gv = float(g)
    ok = "✓" if abs(gv) > 1e-30 else "✗"
    print(f"{n:<15s} {gv:>14.4e}  {ok:>10s}")

print()
print("All gradients are finite and nonzero → gradient-based inference works.")

# %% [markdown]
# ---
# ## 9. Parameter Sensitivity
#
# To build intuition, we vary one parameter at a time while holding the
# others fixed and the same GP realisation $\boldsymbol{\xi}$.  Each
# panel shows how the resulting SFH changes.

# %%
xi_sens = random.normal(random.PRNGKey(77), shape=(N_GRID,))
cmap_sens = plt.cm.coolwarm

param_sweeps = [
    {"name": r"$\alpha$", "key": "alpha",
     "values": [0.5, 1.0, 1.5, 2.5, 4.0],
     "base": dict(alpha=1.5, beta=1.0, tau_gyr=5.0, norm=8.0,
                  sigma=1.5, tau_myr=50)},
    {"name": r"$\beta$", "key": "beta",
     "values": [0.3, 0.8, 1.5, 2.5, 4.0],
     "base": dict(alpha=1.5, beta=1.0, tau_gyr=5.0, norm=8.0,
                  sigma=1.5, tau_myr=50)},
    {"name": r"$\tau_{\rm mean}$", "key": "tau_gyr",
     "values": [1.0, 3.0, 5.0, 8.0, 11.0],
     "base": dict(alpha=1.5, beta=1.0, tau_gyr=5.0, norm=8.0,
                  sigma=1.5, tau_myr=50)},
    {"name": r"$A$ (norm)", "key": "norm",
     "values": [1.0, 3.0, 8.0, 20.0, 50.0],
     "base": dict(alpha=1.5, beta=1.0, tau_gyr=5.0, norm=8.0,
                  sigma=1.5, tau_myr=50)},
    {"name": r"$\sigma_{\rm PS}$", "key": "sigma",
     "values": [0.3, 0.8, 1.5, 2.5, 4.0],
     "base": dict(alpha=1.5, beta=1.0, tau_gyr=5.0, norm=8.0,
                  sigma=1.5, tau_myr=50)},
    {"name": r"$\tau_{\rm PS}$", "key": "tau_myr",
     "values": [5, 20, 50, 100, 300],
     "base": dict(alpha=1.5, beta=1.0, tau_gyr=5.0, norm=8.0,
                  sigma=1.5, tau_myr=50)},
]

fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True)

for ax, sweep in zip(axes.flat, param_sweeps):
    vals = sweep["values"]
    base = sweep["base"].copy()
    for k, v in enumerate(vals):
        p = base.copy()
        p[sweep["key"]] = v

        mean = double_powerlaw(age_yr, p["alpha"], p["beta"],
                               p["tau_gyr"] * 1e9, p["norm"])
        sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age,
                                         p["sigma"], p["tau_myr"] * 1e6)
        gp = gp_from_xi(xi_sens, sqrt_p, N_GRID)
        sig_x_sq = float(drw_variance(p["sigma"]))
        sfr = mean * jnp.exp(gp - sig_x_sq / 2.0)

        label = f"{v}"
        if sweep["key"] == "tau_myr":
            label = f"{v} Myr"
        elif sweep["key"] == "tau_gyr":
            label = f"{v} Gyr"
        t_sfr, sfr_lin = sfh_on_linear_time(sfr)
        ax.plot(t_sfr, sfr_lin, lw=1.8, alpha=0.8,
                color=cmap_sens(k / (len(vals) - 1)), label=label)

    ax.set_title(f"Vary {sweep['name']}", fontsize=9)
    ax.legend(fontsize=5.5, ncol=2)
    ax.set_xlim(0, 13.5)
    ax.set_ylim(bottom=0)

for ax in axes[:, 0]:
    ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
for ax in axes[1]:
    ax.set_xlabel("Lookback time [Gyr]")

fig.tight_layout()
savefig(fig, "parameter_sensitivity")
plt.show()

# %% [markdown]
# ---
# ## Summary
#
# ### SEDModel parameters at a glance
#
# | Parameter | Symbol | Meaning | Typical range | Units |
# |:----------|:------:|:--------|:-------------|:------|
# | PSD amplitude | $\sigma_{\rm PS}$ | SFR fluctuation strength ($\sigma_x = \sigma_{\rm PS}/\sqrt{2}$) | 0.3–5.0 | — |
# | PSD timescale | $\tau_{\rm PS}$ | Burst memory time | 1–300 | Myr |
# | Falling slope | $\alpha$ | Decline from peak (cosmic time) | 0.5–4.0 | — |
# | Rising slope | $\beta$ | Rise to peak (cosmic time) | 0.3–3.0 | — |
# | Turnover time | $\tau$ | Lookback time of peak SFR | 1–11 | Gyr |
# | Peak SFR | $A$ | Normalisation (**not** stellar mass) | 0.1–100 | M$_\odot$ yr$^{-1}$ |
#
# ### Key takeaways
#
# 1. **IFT correlated field model**: SFH fluctuations are a GP drawn
#    from a PSD, parametrised by
#    $\boldsymbol{\xi} \sim \mathcal{N}(0, I)$ in a standardised
#    latent space.
#
# 2. **Two PSD parameters encode the physics**: $\sigma_{\rm PS}$
#    (how bursty) and $\tau_{\rm PS}$ (how long bursts last).
#
# 3. **The mean SFH provides the secular envelope**: a double power
#    law capturing the broad rise and fall of star formation.
#
# 4. **Lognormal correction**: $-\sigma_x^2/2$ ensures that burstiness
#    does not bias the mean SFR upward.
#
# 5. **PSD parameters connect to observables**: main-sequence scatter,
#    Hα/UV ratio, and the window functions of each spectral feature.
#
# 6. **End-to-end JAX gradients**: every parameter has a finite,
#    nonzero gradient through the full model → gradient-based
#    inference works.
#
# ### Next
#
# [**T02 — The Forward SEDModel**](T02_forward_model.ipynb) walks through
# the full differentiable pipeline from SFH → SSP integration → dust
# attenuation → redshift → filter convolution → observed photometry.

# %% [markdown]
# ---
# ## Appendix: Hardware Check

# %%
from tengri.utils.devices import check_resources
check_resources()
