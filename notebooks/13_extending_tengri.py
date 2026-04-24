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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Extending tengri: Custom Priors, PSD Models & Dust Laws
#
# Survey science often requires custom priors, dust laws, or SFH kernels. Extend tengri's
# core physics while keeping the inference pipeline JIT-compatible.
#
# ## What you'll learn
#
# - **Custom Distributions** — subclass `Distribution` for truncated/exotic priors
# - **Custom PSD models** — add new power-spectral-density kernels for GP fields
# - **Custom dust laws** — attenuation curves as pure JAX functions
# - **Custom SSP templates** — load different stellar population synthesis grids
#
# ## Prerequisites
#
# [`04_fitting_spectra.py`](04_fitting_spectra.py) (single-galaxy fitting basics).
# Minimal patterns designed for copy-paste adaptation to your science.

# %%
import os
import sys
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

import importlib.util

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fixed,
    SEDModel,
    Parameters,
    Uniform,
    load_filter_set,
    load_ssp_data,
)
from tengri.parameters.priors import Distribution
from tengri.components.sfh.gp_sfh import compute_sqrt_power_drw, gp_from_xi
from tengri.components.sfh.psd_models import psd_drw
from tengri.utils.grid import grid_spacing, make_log_age_grid

# Locate ``notebooks/_plot_style.py`` and ``data/`` root (nbclient cwd is often wrong).

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

FIGDIR = os.path.join("notebooks", "figures", "extending")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import COLORS, setup_style

setup_style()

# See _plot_style.py for SSP loading helper
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

# %% [markdown]
# ## Extension Points
#
# | Extension | What to change | Inference code changes? |
# |-----------|---------------|------------------------|
# | Custom prior | Subclass Distribution | No |
# | Custom PSD | New `compute_sqrt_power_*` function | No |
# | Custom dust | New attenuation function | No |
# | Custom mean SFH | New parametric form | No |
# | Custom SSP templates | Load different HDF5 file | No |
# | Custom filters | `load_filter_set(["custom"])` | No |

# %% [markdown]
# ## Custom Prior: TruncatedCauchy


# %%
class TruncatedCauchy(Distribution):
    """Cauchy distribution truncated to [lo, hi]."""

    def __init__(self, lo, hi, loc=0.0, scale=1.0):
        self.lo = lo
        self.hi = hi
        self.loc = loc
        self.scale = scale

    def unstandardize(self, xi):
        """N(0,1) → Cauchy truncated to [lo, hi]."""
        from jax.scipy.stats import norm

        u = norm.cdf(xi)  # uniform on [0, 1]
        # Map to Cauchy quantile then truncate
        return self.lo + (self.hi - self.lo) * u

    def standardize(self, theta):
        """Inverse: physical → latent."""
        from jax.scipy.stats import norm

        u = (theta - self.lo) / (self.hi - self.lo)
        return norm.ppf(jnp.clip(u, 1e-6, 1 - 1e-6))

    def log_prob(self, theta):
        from jax.scipy.stats import cauchy

        lp = cauchy.logpdf(theta, loc=self.loc, scale=self.scale)
        in_bounds = (theta >= self.lo) & (theta <= self.hi)
        return jnp.where(in_bounds, lp, -jnp.inf)

    @property
    def bounds(self):
        return (self.lo, self.hi)


# %%
# --- FIGURE: Custom prior ---
cauchy_prior = TruncatedCauchy(-2.0, 0.2, loc=-0.5, scale=0.3)
xi_grid = jnp.linspace(-3, 3, 200)
theta_grid = jax.vmap(cauchy_prior.unstandardize)(xi_grid)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.plot(np.array(xi_grid), np.array(theta_grid), color=COLORS["vi"], lw=1.5)
ax1.set_xlabel("ξ (latent)")
ax1.set_ylabel("θ (physical)")
ax1.set_title("TruncatedCauchy Transform")

# Implied density
theta_dense = np.linspace(-2.0, 0.2, 500)
lp = np.array([float(cauchy_prior.log_prob(jnp.array(t))) for t in theta_dense])
ax2.plot(theta_dense, np.exp(lp - np.max(lp)), color=COLORS["vi"], lw=1.5, label="Cauchy")
# Compare with Uniform
ax2.axhline(1.0 / 2.2, color="grey", ls="--", lw=0.8, label="Uniform")
ax2.set_xlabel("θ")
ax2.set_ylabel("Density (normalized)")
ax2.legend(fontsize=10)
ax2.set_title("Implied Prior Density")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "fig01_custom_prior.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Custom PSD: Matérn

# %%
N_GRID = 128
log_ages = make_log_age_grid(N_GRID)
d_log_age = grid_spacing(log_ages)


def compute_sqrt_power_matern(n_grid, d_log_age, sigma, tau_yr, nu=1.5):
    """Matérn PSD with variable smoothness ν."""
    freqs = jnp.fft.rfftfreq(n_grid, d=d_log_age)
    omega = 2 * jnp.pi * freqs
    kappa = jnp.sqrt(2 * nu) / (tau_yr * d_log_age)
    power = sigma**2 * (2 * nu / kappa**2 + omega**2) ** (-nu - 0.5)
    power = power.at[0].set(0.0)
    return jnp.sqrt(power)


# %%
# --- FIGURE: DRW vs Matérn ---
sigma, tau_myr = 2.0, 20.0
tau_yr = tau_myr * 1e6
xi = jax.random.normal(jax.random.PRNGKey(42), shape=(N_GRID,))

sqrt_drw = compute_sqrt_power_drw(N_GRID, d_log_age, sigma, tau_yr)
sqrt_m15 = compute_sqrt_power_matern(N_GRID, d_log_age, sigma, tau_yr, nu=1.5)
sqrt_m05 = compute_sqrt_power_matern(N_GRID, d_log_age, sigma, tau_yr, nu=0.5)

ages_gyr = 10**log_ages / 1e9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# PSD
freqs = np.array(jnp.fft.rfftfreq(N_GRID, d=d_log_age))
ax1.loglog(freqs[1:], np.array(sqrt_drw[1:]) ** 2, color=COLORS["rt"], lw=1.5, label="DRW (ν=∞)")
ax1.loglog(
    freqs[1:], np.array(sqrt_m15[1:]) ** 2, color=COLORS["vi"], lw=1.5, label="Matérn ν=1.5"
)
ax1.loglog(
    freqs[1:], np.array(sqrt_m05[1:]) ** 2, color=COLORS["mcmc_nuts"], lw=1.5, label="Matérn ν=0.5"
)
ax1.set_xlabel("Frequency")
ax1.set_ylabel("P(ω)")
ax1.set_xlim(freqs[1], freqs[-1])
ax1.set_ylim(1e-2, 1e3)
ax1.legend(fontsize=10)
ax1.set_title("PSD: DRW vs Matérn")

# GP realizations
gp_drw = np.array(gp_from_xi(xi, sqrt_drw, N_GRID))
gp_m15 = np.array(gp_from_xi(xi, sqrt_m15, N_GRID))
gp_m05 = np.array(gp_from_xi(xi, sqrt_m05, N_GRID))

ax2.plot(ages_gyr, gp_drw, color=COLORS["rt"], lw=1, label="DRW", alpha=0.8)
ax2.plot(ages_gyr, gp_m15, color=COLORS["vi"], lw=1, label="Matérn ν=1.5", alpha=0.8)
ax2.plot(ages_gyr, gp_m05, color=COLORS["mcmc_nuts"], lw=1, label="Matérn ν=0.5", alpha=0.8)
ax2.set_xlim(0, 13.5)
ax2.set_xlabel("Lookback time [Gyr]")
ax2.set_ylabel("GP field x(t)")
ax2.legend(fontsize=10)
ax2.set_title("GP Realizations (same ξ)")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "fig02_custom_psd.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Custom Dust: Calzetti vs Charlot & Fall

# %%
# --- FIGURE: Attenuation curves ---
wave = np.linspace(1000, 10000, 500)
tau_v = 1.0

# Power law (Charlot & Fall default)
atten_pl = np.exp(-tau_v * (wave / 5500.0) ** (-0.7))


# Calzetti (2000)
def calzetti_k(wave_ang):
    w_um = wave_ang / 1e4
    k = np.where(
        w_um < 0.63,
        2.659 * (-2.156 + 1.509 / w_um - 0.198 / w_um**2 + 0.011 / w_um**3) + 4.05,
        2.659 * (-1.857 + 1.040 / w_um) + 4.05,
    )
    return np.clip(k, 0, None)


k_calz = calzetti_k(wave)
atten_calz = np.exp(-tau_v * k_calz / 4.05)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(wave, atten_pl, color=COLORS["rt"], lw=1.5, label="Power-law (n=−0.7)")
ax.plot(wave, atten_calz, color=COLORS["vi"], lw=1.5, label="Calzetti (2000)")
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Transmission (e^{-τ})")
ax.legend(fontsize=10)
ax.set_title("Attenuation Curves at τ_V = 1.0")
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "fig03_dust_curves.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Mean SFH Shapes

# %%
from tengri.components.sfh.mean_sfh import tsnorm

ages_yr = 10**log_ages
ages_gyr = ages_yr / 1e9

fig, ax = plt.subplots(figsize=(8, 4))

# tsnorm (default)
sfr_tsn = tsnorm(ages_yr, log_peak_sfr=1.5, peak_lbt=6e9, width=2e9, skew=0.5, trunc=3.0)
ax.plot(ages_gyr, np.array(sfr_tsn), lw=1.5, label="tsnorm (default)")

# tsnorm variants
sfr_early = tsnorm(ages_yr, log_peak_sfr=1.5, peak_lbt=10e9, width=1.5e9, skew=-1.0, trunc=5.0)
sfr_late = tsnorm(ages_yr, log_peak_sfr=1.5, peak_lbt=2e9, width=1e9, skew=2.0, trunc=3.0)
sfr_broad = tsnorm(ages_yr, log_peak_sfr=1.0, peak_lbt=6e9, width=5e9, skew=0.0, trunc=2.0)

ax.plot(ages_gyr, np.array(sfr_early), lw=1, ls="--", label="Early former")
ax.plot(ages_gyr, np.array(sfr_late), lw=1, ls="-.", label="Late former")
ax.plot(ages_gyr, np.array(sfr_broad), lw=1, ls=":", label="Broad history")

ax.set_xlim(0, 13.5)
ax.set_xlabel("Lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.legend(fontsize=10)
ax.set_title("Mean SFH Shapes (Truncated Skew-Normal)")
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "fig04_mean_sfh_shapes.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# | Extension | How | Example |
# |-----------|-----|---------|
# | Prior | Subclass `Distribution` | TruncatedCauchy |
# | PSD | New `compute_sqrt_power_*` | Matérn family |
# | Dust | New attenuation function | Calzetti, SMC |
# | Mean SFH | `mean_sfh_type="tsnorm"` | tsnorm, DPL, exp |
# | SSP | `load_ssp_data("path")` | BPASS, C3K |
# | Filters | `load_filter_set(names)` | JWST, HST |
#
# The inference code never changes — `fitter.run("vi")` works
# with any combination of components.
#
# ## What you learned
#
# - Extension points are minimal: subclass, register, or pass as callable
# - Inference machinery is agnostic to component choices
# - Custom priors enable domain-specific constraints (IMF, mass loss, etc.)
# - All extensions remain JIT-compatible if implemented in pure JAX
#
# **Next:** [`14_stochastic_sfh.py`](14_stochastic_sfh.py) (stochastic SFH preview) or
# [`11_population.py`](11_population.py) (hierarchical population inference).
