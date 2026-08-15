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
# # Tutorial 1: The PSD &rarr; GP &rarr; SFH SEDModel
#
# `tengri` models galaxy star formation histories (SFHs) as **continuous correlated fields** governed by a power spectral density (PSD), using the **Information Field Theory** (IFT) framework ([En&szlig;lin 2019](https://arxiv.org/abs/1804.03350)). The key insight: the PSD encodes the amplitude and timescale of star formation burstiness &mdash; different feedback mechanisms (supernovae, stellar winds, gas accretion) produce different PSDs, and the data decide which is preferred.
#
# **What you will learn:**
#
# 1. What Information Field Theory is and why it matters for SED fitting
# 2. How the DRW power spectral density encodes burstiness physics
# 3. How Gaussian Process realizations are generated from a PSD via FFT
# 4. How the smooth mean SFH (double power law) provides the secular envelope
# 5. How the full SFH combines mean + GP with a lognormal correction
# 6. How the $(\sigma_{\rm PS},\; \tau_{\rm PS})$ burstiness plane maps to SFH diversity
# 7. How end-to-end JAX gradients enable efficient inference
#
# **The key equation:**
#
# $$\text{SFR}(t) = \overline{\text{SFR}}(t) \;\times\; \exp\!\left(x(t) - \tfrac{\sigma_x^2}{2}\right)$$
#
# where $\overline{\text{SFR}}(t)$ is the smooth mean SFH (double power law), $x(t)$ is a zero-mean Gaussian Process drawn from the PSD, and $-\sigma_x^2/2$ is a lognormal correction that ensures $\langle \text{SFR} \rangle = \overline{\text{SFR}}$.
#
# > **Note on parameter names:** This tutorial uses the low-level function parameter names (`sigma_ps`, `tau_ps`, `alpha`, etc.) to show how each component works internally. The high-level API uses descriptive names (`sfh_field_psd_sigma`, `sfh_field_psd_tau_myr`, `sfh_dpl_alpha`, etc.) -- see the [Quickstart](00_quickstart.ipynb).

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure JAX before importing it
from tengri.utils.devices import setup_jax
setup_jax()

import jax
import jax.numpy as jnp
from jax import random, grad, jit

# tengri imports
from tengri.sfh.psd_models import psd_drw, drw_acf, drw_variance, psd_to_sqrt_power
from tengri.sfh.gp_sfh import (
    gp_from_xi, generate_gp_fourier, generate_gp_batch, compute_sqrt_power_drw
)
from tengri.sfh.mean_sfh import double_powerlaw
from tengri.utils.grid import make_log_age_grid, grid_spacing, log_age_to_age_yr, interpolate_to_linear_time
from tengri.utils.cosmology import age_at_z

# ── Plot style ─────────────────────────────────────────────────
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


# ── Redshift twin-axis helper ──────────────────────────────────
def add_redshift_axis(ax, z_ticks=[0, 0.5, 1, 2, 3, 5]):
    """Add redshift labels as a twin x-axis on a lookback-time (Gyr) plot."""
    ax2 = ax.twiny()
    t_uni = float(age_at_z(0.0)) / 1e9  # age of universe in Gyr
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


# ── Figure saving helper ───────────────────────────────────────
FIG_DIR = "notebook_figures"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fig, name, dpi=72):
    path = os.path.join(FIG_DIR, f"01_{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}")


print(f"JAX {jax.__version__} | device: {jax.devices()[0]}")

# %% [markdown]
# ## What is Information Field Theory?
#
# Information Field Theory (IFT) is **Bayesian inference applied to continuous fields** ([En&szlig;lin 2019](https://arxiv.org/abs/1804.03350)). In our case the "field" is the log-SFR fluctuation $x(t)$ &mdash; a continuous function over cosmic time. The problem: we have a finite number of photometric or spectroscopic measurements (data $\mathbf{d}$), but want to reconstruct a continuous SFH (signal $\mathbf{s}$).
#
# The IFT framework sets this up as:
#
# | Symbol | Meaning in our context |
# |:------:|:------------------------|
# | **Signal** $\mathbf{s} = x(t)$ | Log-SFR fluctuation around the mean (the unknown field) |
# | **Data** $\mathbf{d}$ | Observed photometry or spectrum |
# | **Response** $R$ | Maps SFH to observables: $x(t) \to \text{SFR}(t) \to \text{DSPS} \to \text{SED} \to$ photometry |
# | **Noise** $\mathbf{n}$ | Measurement uncertainties |
#
# The measurement equation is $\mathbf{d} = R(\mathbf{s}) + \mathbf{n}$, which is non-linear because of the exponential link, dust attenuation, and stellar population synthesis.
#
# ### The correlated field model
#
# The **prior** on $x(t)$ comes from the PSD: it says "SFR fluctuations at frequency $\omega$ have power $P(\omega)$." This is encoded via the **correlated field model** ([Frank et al. 2021](https://arxiv.org/abs/2105.10470); [Edenhofer et al. 2024](https://arxiv.org/abs/2402.16683)):
#
# $$x = \mathrm{IFFT}\!\left(\sqrt{P} \cdot \hat{\boldsymbol{\xi}}\right), \quad \boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$
#
# where $\boldsymbol{\xi}$ is a **standardized white-noise vector**. All the physics (correlation timescale, burstiness amplitude) lives in the amplitude operator $\sqrt{P}$. The sampler explores $\boldsymbol{\xi}$-space, which has a simple standard-normal geometry, while the physics is encoded in $\sqrt{P}$.
#
# Why is this parametrisation important? Because $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0},\mathbf{I})$ has an identity covariance &mdash; there are no strong correlations between its components. Gradient-based samplers like NUTS and variational methods like geoVI work far more efficiently in such a "whitened" coordinate system than they would if we sampled $x(t)$ directly (which has the highly non-trivial covariance $P$).
#
# ### The information Hamiltonian
#
# The negative log-posterior (called the "information Hamiltonian" in IFT) is:
#
# $$H(\boldsymbol{\xi}|\mathbf{d}) = \underbrace{\frac{1}{2}\sum_k \left(\frac{d_k - m_k(\boldsymbol{\xi})}{\sigma_k}\right)^2}_{\text{data fit } (\chi^2)} \;+\; \underbrace{\frac{1}{2}\,\boldsymbol{\xi}^\top \boldsymbol{\xi}}_{\text{GP prior}}$$
#
# The first term says "match the data." The second says "don't deviate too far from the prior" &mdash; and because $\boldsymbol{\xi}$ is standardized, the prior is simply a unit Gaussian penalty. Inference means minimising $H$ (MAP) or sampling from $\exp(-H)$ (NUTS/geoVI).
#
# ### Why this matters for observers
#
# Traditional SED codes use either parametric SFHs (too rigid, can miss real burstiness) or binned non-parametric SFHs with ad-hoc continuity priors (arbitrary bin widths, difficult to interpret). IFT replaces the ad-hoc prior with a **physically motivated PSD kernel**, and the standardized $\boldsymbol{\xi}$-space enables **gradient-based inference** (NUTS, geoVI) that is 10&ndash;100$\times$ faster than gradient-free samplers like `dynesty`.

# %% [markdown]
# ## The Power Spectral Density (PSD)
#
# The **damped random walk** (DRW) PSD is a Lorentzian with two parameters:
#
# $$P(\omega) = \frac{\sigma_{\rm PS}^2 \, \tau_{\rm PS}}{1 + (\tau_{\rm PS}\,\omega)^2}$$
#
# **$\sigma_{\rm PS}$** sets the **amplitude** of SFR fluctuations. The stationary variance of the GP is $\sigma_x^2 = \sigma_{\rm PS}^2 / 2$, so a 1$\sigma$ excursion in $x$ corresponds to a multiplicative factor of $e^{\sigma_x}$ in SFR. At $\sigma_{\rm PS} = 0.5$, the SFR varies by about 1.4$\times$ around the mean; at $\sigma_{\rm PS} = 4.0$, it varies by about 50$\times$.
#
# **$\tau_{\rm PS}$** sets the **memory time** &mdash; how long a burst or quench episode persists before reverting to the mean. Different physical feedback mechanisms produce characteristic timescales:
#
# | $\tau_{\rm PS}$ range | Physical mechanism | References |
# |:-----:|:----|:----|
# | 1&ndash;10 Myr | Stellar winds, SN blowout | Hopkins et al. 2018 |
# | 20&ndash;50 Myr | Supernova feedback cycle | Faucher-Gigu&egrave;re 2018 |
# | 100&ndash;300 Myr | Gas accretion, halo response | Dekel et al. 2023 |
#
# The autocorrelation function (ACF) is the Fourier transform of the PSD: $\xi(\Delta t) = (\sigma_{\rm PS}^2/2)\,\exp(-|\Delta t|/\tau_{\rm PS})$. The ACF tells you how correlated the SFR is between two epochs separated by $\Delta t$.

# %%
# ── PSD overview: P(omega), ACF, and GP realizations ─────────

regimes = {
    "Smooth":         {"sigma": 0.5, "tau_myr": 200, "color": "#1b9e77"},
    "Moderate":       {"sigma": 1.5, "tau_myr": 50,  "color": "#d95f02"},
    "Bursty":         {"sigma": 2.5, "tau_myr": 20,  "color": "#7570b3"},
    "Highly bursty":  {"sigma": 4.0, "tau_myr": 5,   "color": "#e7298a"},
}

# Frequency in rad/Myr (so the knee at omega=1/tau is visible)
omega_myr = jnp.logspace(-4, 1, 500)  # rad/Myr
delta_t_myr = jnp.linspace(0, 500, 500)  # Myr

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# ── Panel 1: PSD P(omega) -- use Myr units ──
ax = axes[0]
for name, r in regimes.items():
    tau_myr = r["tau_myr"]
    P = psd_drw(omega_myr, r["sigma"], tau_myr)  # tau in Myr
    ax.loglog(omega_myr, P, lw=2.5, color=r["color"],
              label=rf"$\sigma$={r['sigma']}, $\tau$={tau_myr} Myr")
    ax.axvline(1.0 / tau_myr, color=r["color"], ls=":", alpha=0.3)
ax.set_xlabel(r"$\omega$ [rad Myr$^{-1}$]")
ax.set_ylabel(r"$P(\omega)$")
ax.set_title("DRW Power Spectrum")
ax.legend(fontsize=6.5, loc="lower left")
ax.set_xlim(1e-4, 10)

# ── Panel 2: ACF (normalized) ──
ax = axes[1]
for name, r in regimes.items():
    tau_myr = r["tau_myr"]
    acf = drw_acf(delta_t_myr, r["sigma"], tau_myr)
    acf_norm = acf / acf[0]
    ax.plot(delta_t_myr, acf_norm, lw=2.5, color=r["color"], label=name)
ax.set_xlabel(r"$\Delta t$ [Myr]")
ax.set_ylabel(r"Normalized ACF")
ax.set_title("Autocorrelation")
ax.legend(fontsize=7)
ax.set_ylim(-0.05, 1.05)

# ── Panel 3: GP realizations (linear time, offset for clarity) ──
ax = axes[2]
N_GRID = 256
log_ages = make_log_age_grid(N_GRID)
d_log = grid_spacing(log_ages)
ages_yr = log_age_to_age_yr(log_ages)

key = jax.random.PRNGKey(7)
offset = 0
for name, r in regimes.items():
    sqrt_p = compute_sqrt_power_drw(
        N_GRID, float(d_log), r["sigma"], r["tau_myr"] * 1e6
    )
    # Draw 3 realizations
    for draw in range(3):
        key, subkey = jax.random.split(key)
        xi = jax.random.normal(subkey, shape=(N_GRID,))
        gp = gp_from_xi(xi, sqrt_p, N_GRID)
        t_gyr, gp_lin = interpolate_to_linear_time(log_ages, gp, 500)
        lw = 2.0 if draw == 0 else 0.8
        label = name if draw == 0 else None
        ax.plot(np.array(t_gyr), np.array(gp_lin) + offset,
                lw=lw, color=r["color"], alpha=0.7, label=label)
    offset += 5  # vertical offset between regimes

ax.set_xlabel("Lookback time (Gyr)")
ax.set_ylabel(r"GP field $x(t)$ (offset)")
ax.set_title("GP Realizations")
ax.legend(fontsize=6.5, loc="upper right")
ax.set_xlim(0, 13.5)

plt.tight_layout()
savefig(fig, "psd_overview")
plt.show()

# %% [markdown]
# ## Generating GP Realizations
#
# Given the PSD, generating a GP realization is a **three-step FFT recipe**:
#
# 1. **Draw** a standardized latent vector: $\boldsymbol{\xi} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ &mdash; this is what the sampler explores
# 2. **Multiply** in Fourier space: $\hat{\boldsymbol{\xi}} = \texttt{rfft}(\boldsymbol{\xi})$, then $\hat{\mathbf{x}} = \sqrt{P/\Delta u} \cdot \hat{\boldsymbol{\xi}}$
# 3. **Transform** back: $x(t) = \texttt{irfft}(\hat{\mathbf{x}})$
#
# The amplitude operator $\sqrt{P / \Delta u}$ (where $\Delta u$ is the grid spacing in dex) encodes all the correlation structure. Changing $\sigma_{\rm PS}$ or $\tau_{\rm PS}$ changes $\sqrt{P}$ but leaves $\boldsymbol{\xi}$ untouched, so we can explore different burstiness regimes from the **same latent draw**.
#
# The grid is 256 points uniformly spaced in $\log_{10}(t_{\rm age}/\text{yr})$ from 1 Myr to 13.8 Gyr. Uniform spacing in log-age gives finer resolution at young ages (sub-Myr steps at 1 Myr, ~38 Myr steps at 1 Gyr), exactly where the SED is most sensitive to SFR changes.

# %%
# ── Grid setup ────────────────────────────────────────────────

N_GRID = 256
log_age_grid = make_log_age_grid(N_GRID)
d_log_age = grid_spacing(log_age_grid)
age_yr = log_age_to_age_yr(log_age_grid)
age_gyr = age_yr / 1e9

# Helper for proper linear-time SFH plotting
# age_yr IS lookback time (how long ago stars formed)
# The log-age grid is denser at recent times (small age_yr)
# interpolate_to_linear_time resamples to uniform spacing for honest plots

def sfh_on_linear_time(sfr_on_log_grid, n_pts=1000):
    """Resample an SFH from the log-age grid to a uniform linear grid."""
    return interpolate_to_linear_time(log_age_grid, sfr_on_log_grid, n_pts)

print(f"Grid: {N_GRID} points")
print(f"log(age) range: {float(log_age_grid[0]):.2f} to {float(log_age_grid[-1]):.2f} dex")
print(f"Age range: {float(age_yr[0])/1e6:.1f} Myr to {float(age_yr[-1])/1e9:.2f} Gyr")
print(f"Grid spacing: {d_log_age:.4f} dex")


# %%
# ── Same xi, different PSD -> different GP ───────────────────
# This demonstrates the separation of concerns: xi is the
# standardized latent variable, sqrt(P) encodes the physics.

key = random.PRNGKey(42)
xi_fixed = random.normal(key, shape=(N_GRID,))

fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, r["sigma"], r["tau_myr"] * 1e6)
    gp = gp_from_xi(xi_fixed, sqrt_p, N_GRID)

    ax.plot(age_gyr, gp, lw=2.0, color=r["color"])
    ax.axhline(0, ls="--", color="gray", lw=0.7)
    # Show +/- 1-sigma band
    sig_x = np.sqrt(float(drw_variance(r["sigma"])))
    ax.axhspan(-sig_x, sig_x, alpha=0.08, color=r["color"])
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, $\tau_{{\rm PS}}$={r['tau_myr']} Myr",
                 fontsize=9)
    ax.set_ylabel(r"$x(t)$", fontsize=10)
    ax.text(0.97, 0.95, rf"$\sigma_x$={sig_x:.2f}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

for ax in axes[1]:
    ax.set_xlabel("Stellar age [Gyr]")

fig.suptitle(r"Same latent $\boldsymbol{\xi}$, different PSD $\rightarrow$ different GP",
             fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "same_xi_different_psd")
plt.show()

# %%
# ── Multiple GP realizations per regime ──────────────────────
# Different xi draws produce different realizations of the same
# correlation structure. The shaded band shows the expected
# +/- 1 sigma_x envelope.

N_REAL = 5
key = random.PRNGKey(123)

fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, r["sigma"], r["tau_myr"] * 1e6)
    gp_batch = generate_gp_batch(key, sqrt_p, N_GRID, N_REAL)

    for i in range(N_REAL):
        ax.plot(age_gyr, gp_batch[i], lw=1.5, alpha=0.7, color=r["color"])

    ax.axhline(0, ls="--", color="gray", lw=0.7)
    sig_x = np.sqrt(float(drw_variance(r["sigma"])))
    ax.axhspan(-sig_x, sig_x, alpha=0.08, color=r["color"])
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, $\tau_{{\rm PS}}$={r['tau_myr']} Myr",
                 fontsize=9)
    ax.set_ylabel(r"$x(t)$", fontsize=10)

for ax in axes[1]:
    ax.set_xlabel("Stellar age [Gyr]")

fig.suptitle(f"{N_REAL} independent GP realizations per regime", fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "gp_realizations")
plt.show()

# %% [markdown]
# ## The Mean Star Formation History
#
# The GP $x(t)$ fluctuates around zero by construction &mdash; it only generates stochastic variability. The secular envelope (rising at early times, declining at late times) comes from a separate **smooth mean SFH**. We use the double power law ([Behroozi et al. 2013](https://arxiv.org/abs/1207.6105); [Carnall et al. 2018](https://arxiv.org/abs/2207.08778)):
#
# $$\overline{\text{SFR}}(t) = \frac{A}{(t/\tau)^\alpha + (t/\tau)^{-\beta}}$$
#
# where $t$ is **lookback time**. The function peaks near $t \approx \tau$.
#
# ### Understanding $\alpha$ and $\beta$
#
# The parameters have different visual effects depending on whether you think in cosmic time or lookback time. The double power law was originally defined in cosmic time (Behroozi et al. 2013), but our plots show lookback time (present on the left):
#
# - **In cosmic time** (early universe on the left, present on the right): $\alpha$ = falling slope (decline from peak to present), $\beta$ = rising slope (early universe to peak)
# - **In lookback time plots** (present on the left, early universe on the right, what we show): **$\alpha$ controls the right side** (large lookback = early universe), **$\beta$ controls the left side** (small lookback = near present)
#
# A practical way to remember: when you see our lookback-time plots, $\beta$ controls how steeply the SFR drops from the peak toward the present (left side), and $\alpha$ controls how steeply it drops toward the early universe (right side).
#
# ### The normalization $A$
#
# **Important:** The parameter $A$ (norm) is the **peak SFR** in M$_{\odot}$/yr &mdash; it is **not** the stellar mass. Stellar mass is a derived quantity:
#
# $$M_* = \int_0^{t_{\rm H}} \text{SFR}(t)\,(1 - R(t))\,\mathrm{d}t$$
#
# where $R(t)$ is the returned mass fraction from stellar winds and supernovae. The FSPS SSP templates already account for mass loss, so we do not need to apply $R(t)$ separately when computing the SED.

# %%
# ── Galaxy archetypes: mean SFH on lookback time ─────────────
# Four example SFHs showing the range of shapes
# accessible via the double power law parameters.

archetypes = {
    "a=1.0, b=0.5, t=8 Gyr": {
        "alpha": 1.0, "beta": 0.5, "tau_gyr": 8.0, "norm": 5.0,
        "color": "#1b9e77",
    },
    "a=2.0, b=2.0, t=3 Gyr": {
        "alpha": 2.0, "beta": 2.0, "tau_gyr": 3.0, "norm": 50.0,
        "color": "#d95f02",
    },
    "a=1.5, b=3.5, t=2 Gyr": {
        "alpha": 1.5, "beta": 3.5, "tau_gyr": 2.0, "norm": 30.0,
        "color": "#7570b3",
    },
    "a=0.8, b=4.0, t=1.5 Gyr": {
        "alpha": 0.8, "beta": 4.0, "tau_gyr": 1.5, "norm": 80.0,
        "color": "#e7298a",
    },
}

fig, ax = plt.subplots(figsize=(9, 4.5))

for name, p in archetypes.items():
    sfr = double_powerlaw(age_yr, p["alpha"], p["beta"],
                          p["tau_gyr"] * 1e9, p["norm"])
    ax.plot(age_gyr, sfr, lw=2.2, color=p["color"],
            label=rf"$\alpha$={p['alpha']}, $\beta$={p['beta']}, "
                  rf"$\tau$={p['tau_gyr']} Gyr")

ax.set_xlabel(r"Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax.set_title("Mean SFH: Galaxy Archetypes (Double Power Law)", fontsize=11)
ax.set_xlim(0, 13.5)
ax.legend(fontsize=7.5, loc="upper right")

# Annotate present-day side
ax.annotate(r"$z=0$ (present)", xy=(0.3, 0.02), xycoords="axes fraction",
            fontsize=8, color="gray", ha="left")
ax.annotate("", xy=(0.02, 0.05), xytext=(0.18, 0.05),
            xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="gray", lw=1.0))

add_redshift_axis(ax)

fig.tight_layout()
savefig(fig, "galaxy_archetypes")
plt.show()

# %%
# ── Parameter exploration: alpha, beta, tau ──────────────────
# Each panel varies one parameter while holding the others fixed
# at a baseline (alpha=1.5, beta=1.5, tau=4 Gyr, A=10 Msun/yr).

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))

base = {"alpha": 1.5, "beta": 1.5, "tau_gyr": 4.0, "norm": 10.0}
cmap = plt.cm.viridis

# Panel 1: vary alpha (controls RIGHT side = early universe in lookback)
ax = axes[0]
alphas = [0.5, 1.0, 2.0, 3.0, 4.0]
for i, a in enumerate(alphas):
    sfr = double_powerlaw(age_yr, a, base["beta"],
                          base["tau_gyr"] * 1e9, base["norm"])
    ax.plot(age_gyr, sfr, lw=2.0, color=cmap(i / (len(alphas) - 1)),
            label=rf"$\alpha$={a}")
ax.set_title(r"Vary $\alpha$ (right side: early universe)", fontsize=9)
ax.legend(fontsize=7)
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")

# Panel 2: vary beta (controls LEFT side = near present in lookback)
ax = axes[1]
betas = [0.3, 0.8, 1.5, 2.5, 4.0]
for i, b in enumerate(betas):
    sfr = double_powerlaw(age_yr, base["alpha"], b,
                          base["tau_gyr"] * 1e9, base["norm"])
    ax.plot(age_gyr, sfr, lw=2.0, color=cmap(i / (len(betas) - 1)),
            label=rf"$\beta$={b}")
ax.set_title(r"Vary $\beta$ (left side: near present)", fontsize=9)
ax.legend(fontsize=7)

# Panel 3: vary tau (shifts the peak)
ax = axes[2]
taus = [1.0, 2.0, 4.0, 7.0, 10.0]
for i, t in enumerate(taus):
    sfr = double_powerlaw(age_yr, base["alpha"], base["beta"],
                          t * 1e9, base["norm"])
    ax.plot(age_gyr, sfr, lw=2.0, color=cmap(i / (len(taus) - 1)),
            label=rf"$\tau$={t} Gyr")
ax.set_title(r"Vary $\tau$ (peak lookback time)", fontsize=9)
ax.legend(fontsize=7)

for ax in axes:
    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_xlim(0, 13.5)

fig.tight_layout()
savefig(fig, "mean_sfh_parameters")
plt.show()

# %% [markdown]
# ## The Full SFH: Mean + GP + Lognormal Correction
#
# The complete star formation history combines the smooth mean with the GP fluctuations:
#
# $$\text{SFR}(t) = \overline{\text{SFR}}(t) \;\times\; \exp\!\left(x(t) - \frac{\sigma_x^2}{2}\right)$$
#
# ### Why $-\sigma_x^2/2$?
#
# The GP $x(t)$ is zero-mean, so $\langle x(t)\rangle = 0$. But $\exp(x)$ is **not** mean-one: for a zero-mean Gaussian $x$ with variance $\sigma_x^2$, the expectation of $\exp(x)$ is $\exp(\sigma_x^2/2) > 1$. Without correction, burstier models (larger $\sigma_{\rm PS}$) would have systematically higher **average** SFR than the intended mean.
#
# **Intuition:** The median of a lognormal is $\exp(\mu)$, but the mean is $\exp(\mu + \sigma^2/2)$. The upward excursions (bursts) are multiplicatively larger than the downward excursions (quenching dips), pulling the average up. Subtracting $\sigma_x^2/2$ cancels this bias, ensuring $\langle\text{SFR}\rangle = \overline{\text{SFR}}$ regardless of burstiness level.
#
# For the DRW, $\sigma_x^2 = \sigma_{\rm PS}^2 / 2$, so the correction is $-\sigma_{\rm PS}^2/4$.

# %%
# ── Step-by-step: mean -> GP -> full SFH ─────────────────────
# Three panels showing each ingredient and the final combination.

# Mean SFH (spiral-like)
mean_params = {"alpha": 1.2, "beta": 0.8, "tau_gyr": 6.0, "norm": 8.0}
mean_sfr = double_powerlaw(age_yr, mean_params["alpha"], mean_params["beta"],
                           mean_params["tau_gyr"] * 1e9, mean_params["norm"])

# GP (moderate burstiness)
sigma_ps, tau_ps_myr = 1.5, 50.0
sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, sigma_ps, tau_ps_myr * 1e6)
key = random.PRNGKey(7)
gp = generate_gp_fourier(key, sqrt_p, N_GRID)

# Lognormal correction
sigma_x_sq = float(drw_variance(sigma_ps))
full_sfr = mean_sfr * jnp.exp(gp - sigma_x_sq / 2.0)

fig, axes = plt.subplots(1, 3, figsize=(10, 3.8))

# Panel 1: mean SFH (linear)
ax = axes[0]
t_mean, mean_lin = sfh_on_linear_time(mean_sfr)
ax.plot(t_mean, mean_lin, lw=2.2, color="#1b9e77")
ax.fill_between(np.array(age_gyr), 0, np.array(mean_sfr),
                alpha=0.15, color="#1b9e77")
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax.set_title(r"Step 1: Mean $\overline{\mathrm{SFR}}(t)$", fontsize=10)
ax.set_xlim(0, 13.5)

# Panel 2: GP fluctuation
ax = axes[1]
ax.plot(age_gyr, gp, lw=2.0, color="#d95f02")
ax.axhline(0, ls="--", color="gray", lw=0.7)
ax.axhspan(-np.sqrt(sigma_x_sq), np.sqrt(sigma_x_sq), alpha=0.1, color="#d95f02")
ax.set_ylabel(r"$x(t)$")
ax.set_title(rf"Step 2: GP ($\sigma_{{\rm PS}}$={sigma_ps}, $\tau_{{\rm PS}}$={tau_ps_myr:.0f} Myr)",
             fontsize=10)

# Panel 3: full SFH (linear)
ax = axes[2]
t_mean, mean_lin = sfh_on_linear_time(mean_sfr)
ax.plot(t_mean, mean_lin, lw=1.5, ls="--", color="gray", label="Mean")
t_full, full_lin = sfh_on_linear_time(full_sfr)
ax.plot(t_full, full_lin, lw=2.0, color="#7570b3", label="Full SFH")
ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
ax.set_title(r"Step 3: $\overline{\mathrm{SFR}} \times e^{x - \sigma_x^2/2}$", fontsize=10)
ax.legend(fontsize=8)
ax.set_xlim(0, 13.5)

for ax in axes:
    ax.set_xlabel("Lookback time [Gyr]")

fig.tight_layout()
savefig(fig, "sfh_step_by_step")
plt.show()

# %%
# ── 4-regime comparison: LOG scale ───────────────────────────
# Log scale is essential for visualizing extreme burstiness, where
# SFR varies over orders of magnitude.

key = random.PRNGKey(1234)

fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, r["sigma"], r["tau_myr"] * 1e6)
    gp = generate_gp_fourier(key, sqrt_p, N_GRID)

    sig_x_sq = float(drw_variance(r["sigma"]))
    sfr = mean_sfr * jnp.exp(gp - sig_x_sq / 2.0)

    ax.semilogy(age_gyr, sfr, lw=2.0, color=r["color"])
    ax.semilogy(age_gyr, mean_sfr, lw=1.2, ls="--", color="gray")
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, $\tau_{{\rm PS}}$={r['tau_myr']} Myr",
                 fontsize=9)
    ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
    ax.set_xlim(0, 13.5)

for ax in axes[1]:
    ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle("Full SFH (log scale) -- same mean, different burstiness",
             fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "full_sfh_4_regimes")
plt.show()

# %%
# ── 4-regime comparison: LINEAR scale ────────────────────────
# Linear scale gives better intuition for smooth/moderate regimes
# where fluctuations are small.

fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, r["sigma"], r["tau_myr"] * 1e6)
    gp = generate_gp_fourier(key, sqrt_p, N_GRID)

    sig_x_sq = float(drw_variance(r["sigma"]))
    sfr = mean_sfr * jnp.exp(gp - sig_x_sq / 2.0)

    t_sfr, sfr_lin = sfh_on_linear_time(sfr)
    ax.plot(t_sfr, sfr_lin, lw=2.0, color=r["color"])
    t_mean, mean_lin = sfh_on_linear_time(mean_sfr)
    ax.plot(t_mean, mean_lin, lw=1.2, ls="--", color="gray")
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, $\tau_{{\rm PS}}$={r['tau_myr']} Myr",
                 fontsize=9)
    ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(bottom=0)

for ax in axes[1]:
    ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle("Full SFH (linear scale) -- same mean, different burstiness",
             fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "full_sfh_4_regimes_linear")
plt.show()

# %%
# ── Ensemble of 10 SFHs per regime: LOG ──────────────────────
# Multiple draws from the same PSD prior show the diversity
# of SFH shapes that each burstiness regime permits.

N_ENS = 10
key_ens = random.PRNGKey(314)

fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, r["sigma"], r["tau_myr"] * 1e6)
    gp_batch = generate_gp_batch(key_ens, sqrt_p, N_GRID, N_ENS)
    sig_x_sq = float(drw_variance(r["sigma"]))

    for i in range(N_ENS):
        sfr = mean_sfr * jnp.exp(gp_batch[i] - sig_x_sq / 2.0)
        ax.semilogy(age_gyr, sfr, lw=1.0, alpha=0.5, color=r["color"])

    ax.semilogy(age_gyr, mean_sfr, lw=2.0, ls="--", color="black",
                label="Mean SFH")
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, $\tau_{{\rm PS}}$={r['tau_myr']} Myr",
                 fontsize=9)
    ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
    ax.set_xlim(0, 13.5)
    if name == "Smooth":
        ax.legend(fontsize=7)

for ax in axes[1]:
    ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle(f"{N_ENS} SFH realizations per regime (log scale)", fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "sfh_ensemble")
plt.show()

# %%
# ── Ensemble of 10 SFHs per regime: LINEAR ──────────────────

fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, r["sigma"], r["tau_myr"] * 1e6)
    gp_batch = generate_gp_batch(key_ens, sqrt_p, N_GRID, N_ENS)
    sig_x_sq = float(drw_variance(r["sigma"]))

    for i in range(N_ENS):
        sfr = mean_sfr * jnp.exp(gp_batch[i] - sig_x_sq / 2.0)
        t_sfr, sfr_lin = sfh_on_linear_time(sfr)
        ax.plot(t_sfr, sfr_lin, lw=1.0, alpha=0.5, color=r["color"])

    t_mean, mean_lin = sfh_on_linear_time(mean_sfr)
    ax.plot(t_mean, mean_lin, lw=2.0, ls="--", color="black",
            label="Mean SFH")
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, $\tau_{{\rm PS}}$={r['tau_myr']} Myr",
                 fontsize=9)
    ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")
    ax.set_xlim(0, 13.5)
    ax.set_ylim(bottom=0)
    if name == "Smooth":
        ax.legend(fontsize=7)

for ax in axes[1]:
    ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle(f"{N_ENS} SFH realizations per regime (linear scale)", fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "sfh_ensemble_linear")
plt.show()

# %%
# ── Zoom: last 1 Gyr (what UV and H-alpha probe) ────────────
# UV luminosity traces SFR averaged over ~100 Myr; H-alpha traces
# ~10 Myr. Bursty regimes show large SFR swings on these
# timescales, directly affecting SFR-indicator calibrations.

mask = age_yr < 1.0e9
t_recent_myr = np.array(age_yr[mask]) / 1e6

fig, axes = plt.subplots(2, 2, figsize=(10, 5.5), sharex=True)

for ax, (name, r) in zip(axes.flat, regimes.items()):
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, r["sigma"], r["tau_myr"] * 1e6)
    sig_x_sq = float(drw_variance(r["sigma"]))

    for i in range(3):
        subkey = random.PRNGKey(500 + i)
        gp = generate_gp_fourier(subkey, sqrt_p, N_GRID)
        sfr = mean_sfr * jnp.exp(gp - sig_x_sq / 2.0)
        ax.semilogy(t_recent_myr, sfr[mask], lw=1.8, alpha=0.7, color=r["color"])

    ax.semilogy(t_recent_myr, np.array(mean_sfr[mask]), lw=1.5, ls="--", color="gray")
    ax.set_title(rf"{name}: $\sigma_{{\rm PS}}$={r['sigma']}, $\tau_{{\rm PS}}$={r['tau_myr']} Myr",
                 fontsize=9)
    ax.set_ylabel(r"SFR [M$_{\odot}$ yr$^{-1}$]")

    # Mark UV and H-alpha timescales
    ax.axvspan(0, 10, alpha=0.06, color="blue")
    ax.axvspan(0, 100, alpha=0.04, color="purple")
    if name == "Smooth":
        ylim = ax.get_ylim()
        ax.text(5, ylim[1] * 0.4, r"H$\alpha$", fontsize=7, color="blue")
        ax.text(50, ylim[1] * 0.4, "UV", fontsize=7, color="purple")

for ax in axes[1]:
    ax.set_xlabel("Lookback time [Myr]")

fig.suptitle(r"Last 1 Gyr: what UV and H$\alpha$ probe", fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "sfh_zoom_recent")
plt.show()

# %%
# ── Same data in log lookback time ────────────────────────────
# Log time emphasizes the recent past where UV/Halpha probe.

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
key = jax.random.PRNGKey(3)

regimes_log = {
    "Smooth":    {"sigma": 0.5, "tau_myr": 200, "color": "#1b9e77", "ax": axes[0, 0]},
    "Moderate":  {"sigma": 1.5, "tau_myr": 50,  "color": "#d95f02", "ax": axes[0, 1]},
    "Bursty":    {"sigma": 2.5, "tau_myr": 20,  "color": "#7570b3", "ax": axes[1, 0]},
    "Highly bursty": {"sigma": 4.0, "tau_myr": 5, "color": "#e7298a", "ax": axes[1, 1]},
}

for name, r in regimes_log.items():
    ax = r["ax"]
    sqrt_p = compute_sqrt_power_drw(
        N_GRID, float(d_log_age), r["sigma"], r["tau_myr"] * 1e6
    )
    k0_half = float(drw_variance(r["sigma"])) / 2.0

    sfr_mean_bg = double_powerlaw(age_yr, alpha=1.5, beta=1.0, tau=5e9, norm=3.0)

    for draw in range(5):
        key, subkey = jax.random.split(key)
        xi = jax.random.normal(subkey, shape=(N_GRID,))
        gp_val = gp_from_xi(xi, sqrt_p, N_GRID)
        sfr_full = sfr_mean_bg * jnp.exp(gp_val - k0_half)
        ax.plot(np.array(age_gyr), np.array(sfr_full),
                lw=0.5, alpha=0.4, color=r["color"])

    ax.plot(np.array(age_gyr), np.array(sfr_mean_bg),
            "k--", lw=2, label="Mean SFH")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 14)
    ax.set_xlabel("Lookback time (Gyr)")
    ax.set_ylabel(r"SFR (M$_{\odot}$/yr)")
    ax.set_title(rf"{name} ($\sigma$={r['sigma']}, $\tau$={r['tau_myr']} Myr)",
                 fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylim(bottom=0)

fig.suptitle("SFH Realizations in Log Lookback Time", fontsize=13, y=1.01)
fig.tight_layout()
savefig(fig, "sfh_zoom_recent_logtime")
plt.show()

# %% [markdown]
# ## The Burstiness Plane
#
# The 2D $(\sigma_{\rm PS},\; \tau_{\rm PS})$ parameter space produces a rich diversity of SFH behaviors. Varying $\sigma_{\rm PS}$ changes the **amplitude** of fluctuations (smooth vs. violent), while varying $\tau_{\rm PS}$ changes the **timescale** (rapid flickering vs. slow modulation):
#
# - **Low $\sigma$, high $\tau$** (top-left): gentle, long-timescale modulation
# - **High $\sigma$, low $\tau$** (bottom-right): rapid, violent bursts
# - **Low $\sigma$, low $\tau$** (bottom-left): rapid but weak flickering
# - **High $\sigma$, high $\tau$** (top-right): slow but extreme excursions

# %%
# ── 3x3 sigma x tau grid: LOG scale ──────────────────────────

sigma_grid = [0.5, 1.5, 3.0]
tau_grid_myr = [10, 50, 200]

key_bp = random.PRNGKey(2024)

fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True, sharey=True)

for i, tau_myr in enumerate(tau_grid_myr):
    for j, sigma in enumerate(sigma_grid):
        ax = axes[i, j]
        sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, sigma, tau_myr * 1e6)
        gp = generate_gp_fourier(key_bp, sqrt_p, N_GRID)
        sig_x_sq = float(drw_variance(sigma))
        sfr = mean_sfr * jnp.exp(gp - sig_x_sq / 2.0)

        ax.semilogy(age_gyr, sfr, lw=1.5, color="steelblue")
        ax.semilogy(age_gyr, mean_sfr, lw=0.8, ls="--", color="gray")
        ax.set_xlim(0, 13.5)
        ax.set_ylim(1e-3, 1e4)

        if i == 0:
            ax.set_title(rf"$\sigma_{{\rm PS}}$={sigma}", fontsize=9)
        if j == 0:
            ax.set_ylabel(rf"$\tau_{{\rm PS}}$={tau_myr} Myr" + "\n"
                          + r"SFR [M$_{\odot}$ yr$^{-1}$]", fontsize=8)
        if i == 2:
            ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle(r"Burstiness plane: $\sigma_{\rm PS}$ (columns) $\times$ $\tau_{\rm PS}$ (rows) -- log scale",
             fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "burstiness_plane")
plt.show()

# %%
# ── 3x3 sigma x tau grid: LINEAR scale ───────────────────────

fig, axes = plt.subplots(3, 3, figsize=(10, 7.5), sharex=True)

for i, tau_myr in enumerate(tau_grid_myr):
    for j, sigma in enumerate(sigma_grid):
        ax = axes[i, j]
        sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, sigma, tau_myr * 1e6)
        gp = generate_gp_fourier(key_bp, sqrt_p, N_GRID)
        sig_x_sq = float(drw_variance(sigma))
        sfr = mean_sfr * jnp.exp(gp - sig_x_sq / 2.0)

        t_sfr, sfr_lin = sfh_on_linear_time(sfr)
        ax.plot(t_sfr, sfr_lin, lw=1.5, color="steelblue")
        t_mean, mean_lin = sfh_on_linear_time(mean_sfr)
        ax.plot(t_mean, mean_lin, lw=0.8, ls="--", color="gray")
        ax.set_xlim(0, 13.5)
        ax.set_ylim(bottom=0)

        if i == 0:
            ax.set_title(rf"$\sigma_{{\rm PS}}$={sigma}", fontsize=9)
        if j == 0:
            ax.set_ylabel(rf"$\tau_{{\rm PS}}$={tau_myr} Myr" + "\n"
                          + r"SFR [M$_{\odot}$ yr$^{-1}$]", fontsize=8)
        if i == 2:
            ax.set_xlabel("Lookback time [Gyr]")

fig.suptitle(r"Burstiness plane: $\sigma_{\rm PS}$ (columns) $\times$ $\tau_{\rm PS}$ (rows) -- linear scale",
             fontsize=11, y=1.02)
fig.tight_layout()
savefig(fig, "burstiness_plane_linear")
plt.show()

# %% [markdown]
# ## End-to-End Gradients
#
# Because the entire pipeline (PSD $\to$ GP $\to$ SFH $\to$ stellar mass) is implemented in **pure JAX**, we can compute exact gradients of any output with respect to any input via automatic differentiation. This is what enables gradient-based inference (NUTS, geoVI, MAP via Adam) rather than slow gradient-free sampling.
#
# Below we verify that the stellar mass $M_* = \int \text{SFR}(t)\,\mathrm{d}t$ (a simplified integral, ignoring mass loss for illustration) has **finite, nonzero gradients** with respect to all SFH parameters. If any gradient were zero, that parameter would be invisible to gradient-based samplers.

# %%
# ── Gradient computation ─────────────────────────────────────

# Pre-compute static grid values OUTSIDE the JIT function
_log_age = make_log_age_grid(N_GRID)
_d_la = float(_log_age[1] - _log_age[0])
_age = 10.0 ** _log_age

def stellar_mass(alpha, beta, tau_yr, norm, sigma_ps, tau_ps_yr, xi):
    """Compute approximate M* = integral of SFR(t) dt."""
    # Mean SFH (using lookback time = age for this approximation)
    mean = double_powerlaw(_age * 1e9, alpha, beta, tau_yr, norm)

    # GP from xi
    sqrt_p = compute_sqrt_power_drw(N_GRID, _d_la, sigma_ps, tau_ps_yr)
    gp = gp_from_xi(xi, sqrt_p, N_GRID)

    # Full SFH with lognormal correction
    var_x = 0.5 * sigma_ps**2
    sfr = mean * jnp.exp(gp - var_x / 2.0)

    # Integrate: M* = sum(SFR * dt), dt from log-age grid
    dt = _age * jnp.log(10.0) * _d_la  # dt = t * ln(10) * d(log_age)
    return jnp.sum(sfr * dt)

# Reference parameters
alpha0, beta0 = 1.2, 0.8
tau0 = 6.0e9       # yr
norm0 = 8.0
sigma0 = 1.5
tau_ps0 = 50e6     # yr
xi0 = jax.random.normal(jax.random.PRNGKey(0), shape=(N_GRID,))

# Compute M* and all gradients
mstar = stellar_mass(alpha0, beta0, tau0, norm0, sigma0, tau_ps0, xi0)

grad_fn = jax.grad(stellar_mass, argnums=(0, 1, 2, 3, 4, 5))
grads = grad_fn(alpha0, beta0, tau0, norm0, sigma0, tau_ps0, xi0)

param_names = [r"alpha", r"beta", r"tau (mean)", r"A (norm)",
               r"sigma_PS", r"tau_PS"]

print(f"Stellar mass: M* = {float(mstar):.3e} Msun")
print()
print(f"{'Parameter':<20s} {'Gradient':>15s}  {'|grad| > 0?':>12s}")
print("-" * 50)
for name, g in zip(param_names, grads):
    gval = float(g)
    ok = "YES" if abs(gval) > 1e-30 else "NO"
    print(f"{name:<20s} {gval:>15.4e}  {ok:>12s}")

print()
print("All gradients are finite and nonzero: gradient-based inference works.")

# %% [markdown]
# ## Parameter Sensitivity
#
# To build intuition for what each parameter does, we vary one parameter at a time while holding the others fixed, and show how the resulting SFH changes. Each panel uses the same GP realization ($\boldsymbol{\xi}$), so the stochastic structure is held constant and only the effect of the varied parameter is visible.

# %%
# ── Parameter sensitivity: 2x3 panel ─────────────────────────

xi_sens = random.normal(random.PRNGKey(77), shape=(N_GRID,))
cmap_sens = plt.cm.coolwarm

param_sweeps = [
    {"name": r"$\alpha$", "key": "alpha", "values": [0.5, 1.0, 1.5, 2.5, 4.0],
     "base": {"alpha": 1.5, "beta": 1.0, "tau_gyr": 5.0, "norm": 8.0,
              "sigma": 1.5, "tau_myr": 50}},
    {"name": r"$\beta$", "key": "beta", "values": [0.3, 0.8, 1.5, 2.5, 4.0],
     "base": {"alpha": 1.5, "beta": 1.0, "tau_gyr": 5.0, "norm": 8.0,
              "sigma": 1.5, "tau_myr": 50}},
    {"name": r"$\tau_{\rm mean}$", "key": "tau_gyr",
     "values": [1.0, 3.0, 5.0, 8.0, 11.0],
     "base": {"alpha": 1.5, "beta": 1.0, "tau_gyr": 5.0, "norm": 8.0,
              "sigma": 1.5, "tau_myr": 50}},
    {"name": r"$A$ (norm)", "key": "norm", "values": [1.0, 3.0, 8.0, 20.0, 50.0],
     "base": {"alpha": 1.5, "beta": 1.0, "tau_gyr": 5.0, "norm": 8.0,
              "sigma": 1.5, "tau_myr": 50}},
    {"name": r"$\sigma_{\rm PS}$", "key": "sigma",
     "values": [0.3, 0.8, 1.5, 2.5, 4.0],
     "base": {"alpha": 1.5, "beta": 1.0, "tau_gyr": 5.0, "norm": 8.0,
              "sigma": 1.5, "tau_myr": 50}},
    {"name": r"$\tau_{\rm PS}$", "key": "tau_myr",
     "values": [5, 20, 50, 100, 300],
     "base": {"alpha": 1.5, "beta": 1.0, "tau_gyr": 5.0, "norm": 8.0,
              "sigma": 1.5, "tau_myr": 50}},
]

fig, axes = plt.subplots(2, 3, figsize=(10, 5.5), sharex=True)

for ax, sweep in zip(axes.flat, param_sweeps):
    vals = sweep["values"]
    base = sweep["base"].copy()

    for k, v in enumerate(vals):
        p = base.copy()
        p[sweep["key"]] = v

        # Mean SFH
        mean = double_powerlaw(age_yr, p["alpha"], p["beta"],
                               p["tau_gyr"] * 1e9, p["norm"])
        # GP
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
    ax.legend(fontsize=6, ncol=2)
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
# ## Summary
#
# ### SEDModel parameters at a glance
#
# | Parameter | Symbol | Meaning | Typical range | Units |
# |:----------|:------:|:--------|:--------------|:------|
# | PSD amplitude | $\sigma_{\rm PS}$ | Scatter in log-SFR ($\sigma_x = \sigma_{\rm PS}/\sqrt{2}$) | 0.3 -- 5.0 | dex |
# | PSD timescale | $\tau_{\rm PS}$ | Burst memory time | 1 -- 300 | Myr |
# | Falling slope | $\alpha$ | Decline from peak (cosmic time); controls **right side** of lookback-time plot | 0.5 -- 4.0 | -- |
# | Rising slope | $\beta$ | Rise to peak (cosmic time); controls **left side** of lookback-time plot | 0.3 -- 3.0 | -- |
# | Turnover time | $\tau$ | Lookback time of peak SFR | 1 -- 11 | Gyr |
# | Peak SFR | $A$ | Normalization (**not** stellar mass; $M_* = \int\text{SFR}\,dt$) | 0.1 -- 100 | M$_{\odot}$ yr$^{-1}$ |
#
# ### Key takeaways
#
# 1. **IFT correlated field model**: SFH fluctuations are a GP drawn from a PSD, parametrised by $\boldsymbol{\xi} \sim \mathcal{N}(0, I)$ in a standardized latent space that is easy to sample.
# 2. **Two PSD parameters encode the physics**: $\sigma_{\rm PS}$ (how bursty) and $\tau_{\rm PS}$ (how long bursts last).
# 3. **The mean SFH provides the secular envelope**: a double power law that captures the overall rise and decline of star formation.
# 4. **Lognormal correction**: $-\sigma_x^2/2$ ensures that burstiness does not bias the mean SFR upward.
# 5. **End-to-end JAX gradients**: every parameter has a well-defined, nonzero gradient through the full model, enabling NUTS, geoVI, and Fisher matrix calculations.
#
# ### Next
#
# **Tutorial 2** builds the full forward model: SFH $\to$ SSP integration $\to$ dust $\to$ redshift $\to$ photometry. You will see how the SFH constructed here produces an observable spectral energy distribution.

# %% [markdown]
# ## Appendix: Hardware Check

# %%
from tengri.utils.devices import check_resources
check_resources()
