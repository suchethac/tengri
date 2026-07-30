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
# # Star Formation History Gallery
#
# _sfh_gallery
#
# ## When to Use Which SFH Model
#
# | Model | Best use case | Free params | Stochastic? |
# |-------|--------------|-------------|-------------|
# | `tsnorm` | General galaxy; adjustable skew and truncation | 5 | No |
# | `snorm` | Simple symmetric bell; fast to fit | 4 | No |
# | `norm` | Gaussian SFH; oldest/simplest | 3 | No |
# | `lnorm` | Log-normal; naturally skewed toward early times | 3 | No |
# | `dpl` | Double power law; best for quenched galaxies | 4 | No |
# | `dexp` | Delayed exponential; classic SED fitting prior | 3 | No |
# | `exp` | Single exponential; legacy rising-then-flat | 2 | No |
# | `declining_exp` | Declining tau matching FSPS sfh=1 / bagpipes | 3 | No |
# | `const` | Constant SFR; null hypothesis | 1-3 | No |
# | `const_then_exp` | Constant then quenching exponential | 4 | No |
# | `delayed_tau` | Rising-then-falling; widely used baseline | 2 | No |
# | `delayed_bq` | Delayed-tau + instantaneous burst/quench | 4 | No |
# | `powerlaw` | Pure power law; simple slope model | 2-3 | No |
# | `psb_wild2020` | Post-starburst two-component (Wild+2020) | 7 | No |
# | `triweight_burst` | Recent burst + extended base | 2 | No |
# | `snorm_burst` | Skew-normal + flat recent burst (as in ProSpect) | 6 | No |
# | `tsnorm_burst` | Truncated skew-normal + recent burst (as in ProSpect) | 7 | No |
# | `spline_sfh` | PCHIP spline at user-defined nodes (as in ProSpect) | N_nodes | No |
# | `periodic` | Regularly-spaced SF events; mergers/inflows | 4 | No |
# | `buat08` | Velocity-parameterized chemical-evolution SFH | 1 | No |
# | `continuity` | Non-parametric piecewise; flexible | N_bins | No |
# | `psb_continuity` | Non-parametric PSB with quenching epoch (Suess+2021) | N_bins+2 | No |
# | `dirichlet` | Non-parametric with Dirichlet prior | N_bins | No |
# | **`dense_basis`** | **GP-SFH via mass-time quantiles (default)** | **5** | **No** |
# | `dense_basis` + field | **Stochastic GP on top of quantile SFH (default field mode)** | 4 + 2 + N_grid | **Yes** |
# | `dpl` + field | Stochastic bursty GP field on top of DPL | 4 + 2 + N_grid | **Yes** |
# | `tsnorm` + field | Stochastic bursty GP field on top of tsnorm | 5 + 2 + N_grid | **Yes** |
#
# **Rule of thumb**: Use `dense_basis` (the default) for most galaxies — stellar mass
# is a direct parameter, and the quantile-based shape is flexible enough for rising,
# declining, quenched, and multi-episode SFHs. Add `+field` for stochastic burstiness
# (auto-swaps to `dense_basis_pure`). Use `tsnorm` or `dpl` when you want a specific
# parametric functional form.
#
# **Tabulated $(t,\mathrm{SFR})$ from simulations?** Use [`13_tabulated_sfh_to_mock_sed.py`](13_tabulated_sfh_to_mock_sed.py)
# (`sed_from_sfh` / `photometry_from_sfh`) to generate SEDs and mock fluxes without a parametric prior.
#
# ---
#
# This notebook provides a comprehensive visual catalogue of **every SFH model**
# available in tengri. Models are grouped into four families:
#
# 1. **Parametric** -- smooth analytic functions (tsnorm, snorm, norm, lnorm,
#    dpl, dexp, exp, const, triweight_burst, delayed_tau).
# 2. **Non-parametric** -- flexible models without a fixed functional form
#    (dense_basis, continuity, dirichlet).
# 3. **Stochastic (GP)** -- Gaussian-process modulation governed by a PSD
#    (DRW, Extended Regulator, Matern).
# 4. **Composition** -- additive, burst-mixture, and field-modulator
#    combinations built from the registry.
#
# A final section shows the **chemical evolution** module that derives
# $Z(t)$ self-consistently from the SFH.
#
# **Physical picture:** the SFH sets how much stellar mass formed at each cosmic time,
# feeding the UV continuum, Balmer features, the 4000 Å break, and (indirectly) dust
# heating and nebular emission.

# %%
import importlib.util
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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    compute_sqrt_power_drw,
    constant_sfh,
    delayed_exponential_sfh,
    delayed_tau,
    dpl,
    drw_variance,
    exponential_sfh,
    generate_gp_fourier,
    lnorm,
    make_log_age_grid,
    norm,
    psd_drw,
    snorm,
    snorm_burst,
    spline_sfh,
    triweight_burst,
    tsnorm,
    tsnorm_burst,
)
from tengri.sfh.chemical_evolution import closed_box_metallicity
from tengri.sfh.dense_basis import dense_basis_sfh
from tengri.sfh.mean_sfh import (
    buat08,
    constant_then_exponential_sfh,
    declining_exponential_sfh,
    delayed_bq,
    periodic,
    powerlaw_sfh,
    psb_wild2020,
)
from tengri.sfh.nonparametric import (
    DEFAULT_BIN_EDGES_GYR,
    bursty_continuity_prior_logp,
    continuity_sfh,
    dirichlet_sfh,
    psb_continuity_sfh,
)
from tengri.sfh.psd_models import psd_extended_regulator, psd_matern
from tengri.utils.grid import grid_spacing

_nbooks = os.path.join(_repo_root, "notebooks")
if os.path.isdir(_nbooks):
    sys.path.insert(0, _nbooks)

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

try:
    from _plot_style import COLORS, add_sfh_inset, setup_style
except ModuleNotFoundError:
    from tengri.analysis.plotting import setup_style

    COLORS = {
        "sfh_mean": "C0",
        "sfh_gp": "C1",
        "sfh_full": "C2",
    }

    def add_sfh_inset(ax, t_gyr, sfr, **kwargs):
        return ax


setup_style()

FIGDIR = os.path.join("notebooks", "figures", "sfh_gallery")
os.makedirs(FIGDIR, exist_ok=True)

# Lookback labels match plot_sfh() in _plot_style.py
XLAB_LBT_GYR = r"$\mathrm{Lookback\ time\ /\ Gyr}$"

# %% [markdown]
# ## 0. SFH theory: CSP, lognormal field, and DRW checks
#
# ### 0.1 Star formation as input to the CSP
#
# The forward model integrates the **composite stellar population** (paper §3.1),
# $$L_{\rm CSP}(\lambda) = \sum_i w_i\, L_{\rm SSP}(\lambda \mid t_i, Z_i)\,,$$
# where the weights $w_i \propto \mathrm{SFR}(t)\,\Delta t$ on an age grid. Whatever form
# $\mathrm{SFR}(t)$ takes — parametric, tabulated ([`13_tabulated_sfh_to_mock_sed.py`](13_tabulated_sfh_to_mock_sed.py)),
# or a stochastic draw — it enters only through these weights before dust, IGM, and photometry.
# Wavelength-level intuition: [`01_sed_anatomy.py`](01_sed_anatomy.py).
#
# ### 0.2 Lognormal modulation of a mean SFH
#
# For the correlated-field model (paper §3.2.2),
# $$\ln \dot{M}_\star(t) = \ln \bar{\dot{M}}_\star(t) - \frac{K(0)}{2} + x(t)\,,\qquad x \sim \mathcal{GP}(0, P)\,.$$
# The term $-K(0)/2$ enforces $\mathbb{E}[\dot{M}_\star] = \bar{\dot{M}}_\star$ when fluctuations are
# lognormal ($\mathbb{E}[e^x] = e^{\mathrm{Var}(x)/2}$). In Fourier space, white noise $\hat\xi$
# is coloured by $\sqrt{P}$ (IFT / NIFTy picture); inference explores the same standardized
# latent space as the rest of tengri (information Hamiltonian in paper §2.2).
#
# **Paper II** develops population-level constraints on the PSD and observational recovery tests
# for bursty SFH in depth; this gallery focuses on **prior shapes** you can configure in code.
#
# ### 0.3 Uniform $\log_{10}({\rm age}/{\rm yr})$ grid
#
# The stochastic field lives on $u = \log_{10}(t_{\rm age}/{\rm yr})$ with $N{=}256$ points over
# $[6.0,\,10.14]$ — finer spacing at young ages where the UV–optical SED is most sensitive.

# %%
# --- FIGURE 0a: DRW autocorrelation & stationary variance ---
N_TH = 256
log_ages_th = make_log_age_grid(N_TH)
d_u_th = grid_spacing(log_ages_th)
ages_yr_th = np.array(10**log_ages_th)
sigma_th, tau_myr_th = 1.0, 100.0
tau_yr_th = tau_myr_th * 1e6
sqrt_p_th = compute_sqrt_power_drw(N_TH, d_u_th, sigma_th, tau_yr_th)

n_draw = 64
acf_max_lag = 40
acf_emp = np.zeros(acf_max_lag + 1)
var_emp = []
keys = jax.random.split(jax.random.PRNGKey(2026), n_draw)
for key_i in keys:
    gp_i = np.array(generate_gp_fourier(key_i, sqrt_p_th, N_TH))
    var_emp.append(np.var(gp_i))
    for lag in range(acf_max_lag + 1):
        acf_emp[lag] += float(np.mean(gp_i[: N_TH - lag] * gp_i[lag:]))

acf_emp /= n_draw
lags = np.arange(acf_max_lag + 1)
dt_centers = np.array(
    [np.mean(np.abs(ages_yr_th[: N_TH - ell] - ages_yr_th[ell:])) for ell in lags]
)
acf_theory = (float(drw_variance(sigma_th)) / 2.0) * np.exp(-dt_centers / tau_yr_th)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
ax = axes[0]
ax.plot(lags, acf_emp, "o", ms=4, label="Empirical (GP draws)", color=COLORS.get("sfh_gp", "C1"))
ax.plot(lags, acf_theory, "-", lw=1.5, label=r"$\frac{\sigma_{\rm PS}^2}{2}\exp(-\Delta t/\tau)$")
ax.set_xlabel(r"Lag index on $u$-grid")
ax.set_ylabel(r"$\mathbb{E}[x_i x_{i+\ell}]$")
ax.set_title(rf"DRW ACF check ($\sigma={sigma_th}$, $\tau={tau_myr_th}$ Myr)")
ax.legend(fontsize=7, frameon=False)

ax = axes[1]
pred_var = float(drw_variance(sigma_th)) / 2.0
ax.axhline(pred_var, color="k", ls="--", lw=1, label=rf"Theory $\sigma^2/2 = {pred_var:.3f}$")
ax.scatter(np.arange(n_draw), var_emp, s=12, alpha=0.7, c=COLORS.get("sfh_mean", "C0"))
ax.set_xlabel("Draw index")
ax.set_ylabel(r"Var$(x)$ on grid")
ax.set_title(r"Stationary variance of $x(t)$ across keys")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "sfh_theory_acf_variance.png"), dpi=150, bbox_inches="tight")
plt.show()

# %%
# --- FIGURE 0b: Mean preservation under lognormal modulation ---
mean_sfr_th = np.array(
    tsnorm(
        jnp.array(ages_yr_th),
        log_total_mass=10.0,
        peak_lbt=6e9,
        width=2e9,
        skew=0.2,
        trunc=3.0,
    )
)
k0_half_th = float(drw_variance(sigma_th)) / 2.0
ratio_sum = 0.0
n_m = 48
keys_m = jax.random.split(jax.random.PRNGKey(7), n_m)
for km in keys_m:
    gp_m = generate_gp_fourier(km, sqrt_p_th, N_TH)
    full_m = np.array(jnp.asarray(mean_sfr_th) * jnp.exp(gp_m - k0_half_th))
    ratio_sum += np.mean(full_m) / max(np.mean(mean_sfr_th), 1e-30)

print(f"Mean over draws of ratio  <SFR_full>/<SFR_mean>  = {ratio_sum / n_m:.4f}  (expect ~1)")

fig, ax = plt.subplots(figsize=(6.5, 3.2))
ax.plot(
    ages_yr_th / 1e9,
    mean_sfr_th,
    "--",
    lw=1.5,
    color=COLORS.get("sfh_mean", "C0"),
    label="Mean SFH",
)
for km in keys_m[:5]:
    gp_m = generate_gp_fourier(km, sqrt_p_th, N_TH)
    full_m = np.array(jnp.asarray(mean_sfr_th) * jnp.exp(gp_m - k0_half_th))
    ax.plot(ages_yr_th / 1e9, full_m, lw=0.9, alpha=0.65, color=COLORS.get("sfh_full", "C2"))
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.set_title(r"Lognormal modulation: five draws ($\bar{\mathrm{SFR}}\,e^{x-K(0)/2}$)")
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "sfh_theory_lognormal_draws.png"), dpi=150, bbox_inches="tight")
plt.show()


def add_multi_sfh_inset(ax, t_gyr, y_series, colors=None, lws=None, linestyles=None, ylabel="SFR"):
    """Last 200 Myr inset — same geometry as add_sfh_inset, multiple curves."""
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    ax_in = inset_axes(ax, width="35%", height="40%", loc="upper right", borderpad=1.5)
    t_myr = np.asarray(t_gyr) * 1e3
    mask = t_myr <= 200
    n = len(y_series)
    if colors is None:
        colors = [None] * n
    if lws is None:
        lws = [1.0] * n
    if linestyles is None:
        linestyles = ["-"] * n
    for y, c, lw, ls in zip(y_series, colors, lws, linestyles, strict=True):
        if mask.sum() > 2:
            ax_in.plot(t_myr[mask], np.asarray(y)[mask], color=c, lw=lw, ls=ls)
    ax_in.set_xlim(0, 200)
    ax_in.set_xlabel("Lookback (Myr)", fontsize=6)
    ax_in.set_ylabel(ylabel, fontsize=6)
    ax_in.tick_params(labelsize=5)
    return ax_in


# %%
# Shared lookback-time grid (yr) — linear in lookback (matches plot_sfh / notebook style)
t_yr = np.linspace(10**6.0, 10**10.14, 2000)
t_gyr = t_yr / 1e9

# %% [markdown]
# ## 1. Parametric SFH Models
#
# All ten parametric models on a single comparison plot.  Each is normalised
# so that the peak SFR is comparable, making shape differences easy to see.

# %%
# --- FIGURE 1: All 10 parametric models ---
fig, ax = plt.subplots(figsize=(9, 4.5))

# A palette cycling through colorblind-safe colors
pal = plt.cm.tab10(np.linspace(0, 1, 10))

# 1. delayed_tau: SFR = norm * t * exp(-t/tau)
sfr_dtau = delayed_tau(t_yr, tau=2e9, norm=1.0)
sfr_dtau = sfr_dtau / np.max(sfr_dtau)
ax.plot(t_gyr, sfr_dtau, color=pal[0], lw=1.8, label="delayed_tau")

# 2. delayed_exponential (dexp): peaks at start + tau
sfr_dexp = delayed_exponential_sfh(t_yr, log_total_mass=10.0, tau=3e9, start=0.0)
sfr_dexp = sfr_dexp / np.max(sfr_dexp)
ax.plot(t_gyr, sfr_dexp, color=pal[1], lw=1.8, label="dexp")

# 3. double_powerlaw (dpl)
sfr_dpl = dpl(t_yr, alpha=2.0, beta=1.0, tau=5e9, log_total_mass=10.0)
sfr_dpl = sfr_dpl / np.max(sfr_dpl)
ax.plot(t_gyr, sfr_dpl, color=pal[2], lw=1.8, label="dpl")

# 4. tsnorm (Bellstedt+2020)
sfr_tsn = tsnorm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=0.3, trunc=3.0)
sfr_tsn = sfr_tsn / np.max(sfr_tsn)
ax.plot(t_gyr, sfr_tsn, color=pal[3], lw=1.8, label="tsnorm")

# 5. snorm (skewed Gaussian)
sfr_sn = snorm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=0.5)
sfr_sn = sfr_sn / np.max(sfr_sn)
ax.plot(t_gyr, sfr_sn, color=pal[4], lw=1.8, label="snorm")

# 6. norm (symmetric Gaussian)
sfr_nm = norm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9)
sfr_nm = sfr_nm / np.max(sfr_nm)
ax.plot(t_gyr, sfr_nm, color=pal[5], lw=1.8, label="norm")

# 7. lnorm (log-normal)
sfr_ln = lnorm(t_yr, log_total_mass=10.0, peak_lbt=5e9, width=0.4)
sfr_ln = sfr_ln / np.max(sfr_ln)
ax.plot(t_gyr, sfr_ln, color=pal[6], lw=1.8, label="lnorm")

# 8. constant
sfr_cst = constant_sfh(t_yr, log_sfr=0.0, start=1e9, end=12e9)
sfr_cst = sfr_cst / np.max(sfr_cst + 1e-30)
ax.plot(t_gyr, sfr_cst, color=pal[7], lw=1.8, label="const")

# 9. exponential (declining tau)
sfr_exp = exponential_sfh(t_yr, log_total_mass=10.0, tau=3e9, start=0.0)
sfr_exp = sfr_exp / np.max(sfr_exp)
ax.plot(t_gyr, sfr_exp, color=pal[8], lw=1.8, label="exp")

# 10. triweight_burst
sfr_tw = triweight_burst(t_yr, log_tpeak_myr=2.0, log_tmax_myr=2.5)
sfr_tw = sfr_tw / np.max(sfr_tw + 1e-30)
ax.plot(t_gyr, sfr_tw, color=pal[9], lw=1.8, label="triweight_burst")

ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("All 10 Parametric SFH Models")
ax.legend(ncol=2, fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
ax.set_ylim(-0.05, 1.15)
add_multi_sfh_inset(
    ax,
    t_gyr,
    [sfr_dtau, sfr_dexp, sfr_dpl, sfr_tsn, sfr_sn, sfr_nm, sfr_ln, sfr_cst, sfr_exp, sfr_tw],
    colors=list(pal[:10]),
    lws=[0.9] * 10,
    ylabel="SFR",
)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_parametric_sfh_gallery.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.1 delayed_tau: varying $\tau$
#
# $\text{SFR}(t) = N \cdot t \cdot \exp(-t/\tau)$. Peaks at $t = \tau$.

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))
series_dtau = []
for tau_gyr in [0.5, 1.0, 3.0, 6.0, 10.0]:
    sfr = delayed_tau(t_yr, tau=tau_gyr * 1e9, norm=1.0)
    sfr = sfr / np.max(sfr)
    series_dtau.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"$\tau = {tau_gyr}$ Gyr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title(r"delayed\_tau: varying $\tau$")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_dtau, ylabel="SFR")
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_delayed_tau_vary.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.2 delayed_exponential: varying $\tau$ and peak SFR

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

ax = axes[0]
series_dexp_tau = []
for tau_gyr in [0.5, 1.5, 3.0, 5.0]:
    sfr = delayed_exponential_sfh(t_yr, log_total_mass=10.0, tau=tau_gyr * 1e9)
    sfr = sfr / np.max(sfr)
    series_dexp_tau.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"$\tau = {tau_gyr}$ Gyr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title(r"dexp: varying $\tau$")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_dexp_tau, ylabel="SFR")

ax = axes[1]
series_dexp_lp = []
for log_p in [-0.5, 0.0, 0.5, 1.0]:
    sfr = delayed_exponential_sfh(t_yr, log_total_mass=10.0, tau=3e9)
    series_dexp_lp.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"$\log$ peak = {log_p}")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title(r"dexp: varying log peak SFR ($\tau = 3$ Gyr)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_dexp_lp, ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_dexp_vary.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.3 Double Power Law (DPL): varying $\alpha$, $\beta$, $\tau$
#
# $\text{SFR}(t) = \text{norm} / [(t/\tau)^\alpha + (t/\tau)^{-\beta}]$

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3))

# Vary alpha (falling slope in cosmic time = decline from peak to present)
ax = axes[0]
series_dpl_a = []
for alpha in [0.5, 1.0, 2.0, 4.0]:
    sfr = dpl(t_yr, alpha=alpha, beta=1.0, tau=5e9, log_total_mass=10.0)
    sfr = sfr / np.max(sfr)
    series_dpl_a.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"$\alpha = {alpha}$")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title(r"DPL: vary $\alpha$ ($\beta=1$, $\tau=5$ Gyr)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_dpl_a, ylabel="SFR")

# Vary beta (rising slope in cosmic time)
ax = axes[1]
series_dpl_b = []
for beta in [0.3, 0.7, 1.5, 3.0]:
    sfr = dpl(t_yr, alpha=2.0, beta=beta, tau=5e9, log_total_mass=10.0)
    sfr = sfr / np.max(sfr)
    series_dpl_b.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"$\beta = {beta}$")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title(r"DPL: vary $\beta$ ($\alpha=2$, $\tau=5$ Gyr)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_dpl_b, ylabel="SFR")

# Vary tau (turnover time)
ax = axes[2]
series_dpl_t = []
for tau_gyr in [1.0, 3.0, 6.0, 10.0]:
    sfr = dpl(t_yr, alpha=2.0, beta=1.0, tau=tau_gyr * 1e9, log_total_mass=10.0)
    sfr = sfr / np.max(sfr)
    series_dpl_t.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"$\tau = {tau_gyr}$ Gyr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title(r"DPL: vary $\tau$ ($\alpha=2$, $\beta=1$)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_dpl_t, ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_dpl_vary.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.4 tsnorm (Bellstedt+2020): varying peak, width, skew, truncation

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 6))

# Vary peak lookback time
ax = axes[0, 0]
series_tsn_pk = []
for pk_gyr in [2.0, 4.0, 6.0, 9.0, 12.0]:
    sfr = tsnorm(t_yr, log_total_mass=10.0, peak_lbt=pk_gyr * 1e9, width=2e9, skew=0.0, trunc=3.0)
    sfr = sfr / np.max(sfr + 1e-30)
    series_tsn_pk.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"peak = {pk_gyr} Gyr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("tsnorm: vary peak lookback time")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_tsn_pk, ylabel="SFR")

# Vary width
ax = axes[0, 1]
series_tsn_w = []
for w_gyr in [0.5, 1.0, 2.0, 4.0]:
    sfr = tsnorm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=w_gyr * 1e9, skew=0.0, trunc=3.0)
    sfr = sfr / np.max(sfr + 1e-30)
    series_tsn_w.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"width = {w_gyr} Gyr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("tsnorm: vary width")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_tsn_w, ylabel="SFR")

# Vary skew
ax = axes[1, 0]
series_tsn_sk = []
for sk in [-0.8, -0.3, 0.0, 0.3, 0.8]:
    sfr = tsnorm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=sk, trunc=3.0)
    sfr = sfr / np.max(sfr + 1e-30)
    series_tsn_sk.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"skew = {sk}")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("tsnorm: vary skew")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_tsn_sk, ylabel="SFR")

# Vary truncation
ax = axes[1, 1]
series_tsn_tr = []
for tr in [1.0, 2.0, 5.0, 10.0]:
    sfr = tsnorm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=0.3, trunc=tr)
    sfr = sfr / np.max(sfr + 1e-30)
    series_tsn_tr.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"trunc = {tr}")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("tsnorm: vary truncation")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_tsn_tr, ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_tsnorm_vary.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.5 snorm, norm, lnorm comparison
#
# Three Gaussian-family models: snorm adds skewness, lnorm operates in
# log-age space, norm is the simple symmetric case.

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))

sfr_norm = norm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9)
sfr_snorm = snorm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=0.5)
sfr_lnorm = lnorm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=0.4)

series_gauss = []
for sfr_i, name, ls in [
    (sfr_norm, "norm (Gaussian)", "-"),
    (sfr_snorm, "snorm (skew=0.5)", "--"),
    (sfr_lnorm, "lnorm (log-normal)", "-."),
]:
    sfr_i = sfr_i / np.max(sfr_i)
    series_gauss.append(sfr_i)
    ax.plot(t_gyr, sfr_i, lw=1.8, ls=ls, label=name)

ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("Gaussian-family SFH Models")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(
    ax, t_gyr, series_gauss, linestyles=["-", "--", "-."], lws=[1.2, 1.2, 1.2], ylabel="SFR"
)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_gaussian_family.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.6 constant and exponential

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))

ax = axes[0]
sfr_c1 = constant_sfh(t_yr, log_sfr=0.0)
sfr_c2 = constant_sfh(t_yr, log_sfr=0.0, start=2e9, end=10e9)
ax.plot(t_gyr, np.array(sfr_c1), lw=1.5, label="full (0 - 14 Gyr)")
ax.plot(t_gyr, np.array(sfr_c2), lw=1.5, label="2 - 10 Gyr window")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title("Constant SFH")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, [np.array(sfr_c1), np.array(sfr_c2)], ylabel="SFR")

ax = axes[1]
series_exp = []
for tau_gyr in [1.0, 3.0, 5.0, 10.0]:
    sfr = exponential_sfh(t_yr, log_total_mass=10.0, tau=tau_gyr * 1e9)
    sfr = sfr / np.max(sfr)
    series_exp.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"$\tau = {tau_gyr}$ Gyr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title(r"Exponential (declining $\tau$)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_exp, ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_const_exp.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.6b declining_exponential and constant_then_exponential
#
# **`declining_exponential_sfh`** matches FSPS `sfh=1` / bagpipes `'exponential'`:
# in cosmic time $T$, $\mathrm{SFR}(T) = S_0 e^{-T/\tau}$, so in lookback time
# SFR *increases* toward the past. Distinct from `exponential_sfh` which peaks at $t=0$.
#
# **`constant_then_exponential_sfh`** models quenching: constant SFR from formation
# until $t_\mathrm{quench}$, then exponential decline.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))

ax = axes[0]
series_dexp2 = []
for tau_gyr, age_gyr_val in [(1.0, 12.0), (3.0, 12.0), (6.0, 12.0), (1.0, 6.0)]:
    sfr = declining_exponential_sfh(
        t_yr, log_total_mass=10.0, tau=tau_gyr * 1e9, age=age_gyr_val * 1e9
    )
    sfr = np.array(sfr) / np.max(np.array(sfr) + 1e-30)
    series_dexp2.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"$\tau={tau_gyr}$ Gyr, age={age_gyr_val} Gyr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title(r"declining\_exponential (FSPS sfh=1)")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_dexp2, ylabel="SFR")

ax = axes[1]
series_cte = []
for q_gyr, tau_gyr in [(8.0, 1.0), (5.0, 0.5), (3.0, 2.0)]:
    sfr = constant_then_exponential_sfh(
        t_yr, log_sfr=0.0, tau=tau_gyr * 1e9, quench_age=q_gyr * 1e9, age=12e9
    )
    sfr = np.array(sfr) / np.max(np.array(sfr) + 1e-30)
    series_cte.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"quench={q_gyr} Gyr, $\tau={tau_gyr}$ Gyr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("constant_then_exponential (quenching)")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_cte, ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_declining_const_exp.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.7 triweight_burst: compact burst kernel

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))
series_tw = []
lss = []
for tp, tw, ls in [(1.5, 2.0, "-"), (2.0, 2.5, "--"), (2.5, 3.0, "-.")]:
    sfr = triweight_burst(t_yr, log_tpeak_myr=tp, log_tmax_myr=tw)
    sfr = sfr / np.max(sfr + 1e-30)
    series_tw.append(sfr)
    lss.append(ls)
    ax.plot(t_gyr, sfr, lw=1.5, ls=ls, label=rf"peak = $10^{{{tp}}}$ Myr, dur = $10^{{{tw}}}$ Myr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised kernel")
ax.set_title("Triweight Burst Kernel (Zacharegkas+2025)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_tw, linestyles=lss, ylabel="Kernel")
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_triweight_burst.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.8 ProSpect burst models: snorm_burst and tsnorm_burst (Robotham+2020)
#
# Both models add a **flat recent-burst component** to the base skew-normal SFH:
# $$\text{SFR}(t) = (1 - f_\mathrm{burst})\,\cdot\,\text{base}(t)
#   + f_\mathrm{burst}\,\cdot\,\text{burst}(t)$$
# where $\text{burst}(t) = S / t_\mathrm{burst}$ for $t < t_\mathrm{burst}$, $0$ otherwise,
# and $S$ is set so the total stellar mass is preserved. Implements the same form as the R package
# **ProSpect** (Robotham et al. 2020, MNRAS 495 905).

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))

# --- Left panel: snorm_burst — varying burst_sfr ---
ax = axes[0]
series_snb = []
lss_snb = []
base_snorm = np.array(snorm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=0.3))
base_snorm_n = base_snorm / np.max(base_snorm + 1e-30)
ax.plot(t_gyr, base_snorm_n, "k--", lw=1.3, alpha=0.5, label="snorm (no burst)")
for burst_sfr, ls in [(0.1, "-"), (0.5, "--"), (1.0, "-.")]:
    sfr = np.array(
        snorm_burst(
            jnp.array(t_yr),
            log_total_mass=10.0,
            peak_lbt=6e9,
            width=2e9,
            skew=0.3,
            burst_sfr=burst_sfr,
            burst_age=2e8,
        )
    )
    sfr_n = sfr / np.max(sfr + 1e-30)
    series_snb.append(sfr_n)
    lss_snb.append(ls)
    ax.plot(t_gyr, sfr_n, lw=1.5, ls=ls, label=rf"burst\_sfr = {burst_sfr}")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("snorm_burst (ProSpect): varying burst_sfr")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(
    ax,
    t_gyr,
    [base_snorm_n, *series_snb],
    linestyles=["--", *lss_snb],
    lws=[1.0, *([1.2] * len(lss_snb))],
    ylabel="SFR",
)

# --- Right panel: tsnorm_burst — varying burst_age ---
ax = axes[1]
series_tnb = []
lss_tnb = []
base_tsnorm = np.array(
    tsnorm(t_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=0.3, trunc=3.0)
)
base_tsnorm_n = base_tsnorm / np.max(base_tsnorm + 1e-30)
ax.plot(t_gyr, base_tsnorm_n, "k--", lw=1.3, alpha=0.5, label="tsnorm (no burst)")
for burst_age_myr, ls in [(100.0, "-"), (300.0, "--"), (600.0, "-.")]:
    sfr = np.array(
        tsnorm_burst(
            jnp.array(t_yr),
            log_total_mass=10.0,
            peak_lbt=6e9,
            width=2e9,
            skew=0.3,
            trunc=3.0,
            burst_sfr=0.5,
            burst_age=burst_age_myr * 1e6,
        )
    )
    sfr_n = sfr / np.max(sfr + 1e-30)
    series_tnb.append(sfr_n)
    lss_tnb.append(ls)
    ax.plot(t_gyr, sfr_n, lw=1.5, ls=ls, label=rf"burst\_age = {burst_age_myr:.0f} Myr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("tsnorm_burst (ProSpect): varying burst_age")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(
    ax,
    t_gyr,
    [base_tsnorm_n, *series_tnb],
    linestyles=["--", *lss_tnb],
    lws=[1.0, *([1.2] * len(lss_tnb))],
    ylabel="SFR",
)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_prospect_burst.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.9 psb_wild2020: Post-Starburst SFH (Wild+2020)
#
# Two-component model: declining exponential (old stellar population) + double
# power law (recent burst episode). Components are mass-fraction weighted.
# Reference: Wild et al. 2020, MNRAS 494 529.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

ax = axes[0]
series_psb = []
for fburst in [0.1, 0.3, 0.5, 0.8]:
    sfr = np.array(
        psb_wild2020(
            jnp.array(t_yr),
            log_total_mass=10.0,
            age=12e9,
            tau=3e9,
            burstage=0.3e9,
            alpha=2.0,
            beta=2.0,
            fburst=fburst,
        )
    )
    sfr = sfr / np.max(sfr + 1e-30)
    series_psb.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"$f_\mathrm{{burst}} = {fburst}$")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("psb_wild2020: varying burst fraction")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_psb, ylabel="SFR")

ax = axes[1]
series_psb2 = []
for burstage_gyr in [0.1, 0.3, 0.5, 1.0]:
    sfr = np.array(
        psb_wild2020(
            jnp.array(t_yr),
            log_total_mass=10.0,
            age=12e9,
            tau=3e9,
            burstage=burstage_gyr * 1e9,
            alpha=2.0,
            beta=2.0,
            fburst=0.4,
        )
    )
    sfr = sfr / np.max(sfr + 1e-30)
    series_psb2.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.5, label=rf"burst age = {burstage_gyr} Gyr")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("psb_wild2020: varying burst age")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_psb2, ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_psb_wild2020.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.10 powerlaw_sfh and delayed_bq (Ciesla+2017)
#
# **`powerlaw_sfh`**: $\mathrm{SFR}(t) = S_0\,(t/t_{\rm ref})^\alpha$.
# Positive $\alpha$ → rising toward present; negative $\alpha$ → falling.
#
# **`delayed_bq`**: delayed-tau SFH followed by an instantaneous burst or quench
# at `age_bq_yr`. After the event, SFR is held at $r_\mathrm{sfr}$ times the
# pre-event SFR. $r < 1$ = quench; $r > 1$ = burst. (Ciesla+2017, A&A 608 41.)

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

ax = axes[0]
series_pl = []
for alpha in [-1.5, -0.5, 0.5, 1.5, 3.0]:
    sfr = np.array(powerlaw_sfh(jnp.array(t_yr), alpha=alpha, norm=1.0, t_ref=1e8))
    peak = np.max(np.abs(sfr) + 1e-30)
    sfr = sfr / peak
    series_pl.append(np.clip(sfr, 0, None))
    ax.plot(t_gyr, np.clip(sfr, 0, None), lw=1.5, label=rf"$\alpha = {alpha}$")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title(r"powerlaw\_sfh: $\mathrm{SFR}(t) \propto t^\alpha$")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_pl, ylabel="SFR")

ax = axes[1]
series_dbq = []
for r_sfr, ls in [(0.0, "-"), (0.1, "--"), (2.0, "-.")]:
    sfr = np.array(
        delayed_bq(
            jnp.array(t_yr),
            tau_main_yr=3e9,
            age_main_yr=12e9,
            age_bq_yr=1e9,
            r_sfr=r_sfr,
        )
    )
    sfr = sfr / np.max(sfr + 1e-30)
    series_dbq.append(sfr)
    label = rf"$r_\mathrm{{sfr}} = {r_sfr}$ ({'quench' if r_sfr < 1 else 'burst'})"
    ax.plot(t_gyr, sfr, lw=1.5, ls=ls, label=label)
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("delayed_bq (Ciesla+2017): burst/quench at 1 Gyr")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_dbq, linestyles=["-", "--", "-."], ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_powerlaw_delayed_bq.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.11 periodic and buat08
#
# **`periodic`**: regularly-spaced SF events (exponential, delayed, or rectangular
# pulses). Useful for modelling intermittent or merger-triggered star formation.
#
# **`buat08`**: chemically-motivated SFH parameterized by galaxy rotational velocity
# (Buat+2008, A&A 483 107). Polynomial in $\log_{10}(t)$ with velocity-interpolated
# coefficients.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

ax = axes[0]
series_per = []
lss_per = []
for burst_type, ls, label in [
    (0, "-", "exp pulses"),
    (1, "--", "delayed pulses"),
    (2, "-.", "rect pulses"),
]:
    sfr = np.array(
        periodic(
            jnp.array(t_yr),
            delta_bursts_yr=1.5e9,
            tau_bursts_yr=3e8,
            burst_type=burst_type,
            age_yr=12e9,
        )
    )
    sfr = sfr / np.max(sfr + 1e-30)
    series_per.append(sfr)
    lss_per.append(ls)
    ax.plot(t_gyr, sfr, lw=1.5, ls=ls, label=label)
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("periodic: regularly-spaced SF events")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_per, linestyles=lss_per, ylabel="SFR")

ax = axes[1]
series_buat = []
for v_km_s in [50, 100, 175, 250]:
    sfr = np.array(buat08(jnp.array(t_yr), velocity_km_s=float(v_km_s)))
    sfr_valid = np.where(np.isfinite(sfr), sfr, 0.0)
    sfr_valid = sfr_valid / np.max(sfr_valid + 1e-30)
    series_buat.append(sfr_valid)
    ax.plot(t_gyr, sfr_valid, lw=1.5, label=rf"$v = {v_km_s}$ km/s")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("buat08: velocity-parameterized SFH")
ax.legend(fontsize=7, frameon=False, loc="upper right")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_buat, ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_periodic_buat08.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 1.12 spline_sfh: PCHIP spline interpolation (ProSpect)
#
# User-specified SFR values at fixed lookback-time nodes; smoothly interpolated
# via **PCHIP** (Piecewise Cubic Hermite Interpolating Polynomial) which preserves
# monotonicity locally. Implements ProSpect's `massfunc_spline` form (Robotham+2020).

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.5))

node_ages_yr = jnp.array([0.3e9, 1.0e9, 3.0e9, 6.0e9, 10.0e9, 13.0e9])
scenarios = {
    "Rising": jnp.array([3.0, 2.0, 1.5, 1.0, 0.5, 0.1]),
    "Peaked at 6 Gyr": jnp.array([0.5, 1.0, 2.0, 3.0, 1.5, 0.3]),
    "Two episodes": jnp.array([1.5, 0.3, 2.5, 0.5, 1.0, 0.2]),
    "Declining": jnp.array([0.1, 0.3, 0.8, 1.5, 2.5, 3.0]),
}
series_spl = []
lss_spl = ["-", "--", "-.", ":"]
for (label, sfr_nodes), ls in zip(scenarios.items(), lss_spl):
    sfr = np.array(spline_sfh(jnp.array(t_yr), sfr_nodes, node_ages_yr))
    sfr = np.clip(sfr, 0, None)
    sfr = sfr / np.max(sfr + 1e-30)
    series_spl.append(sfr)
    ax.plot(t_gyr, sfr, lw=1.8, ls=ls, label=label)
    ax.scatter(
        np.array(node_ages_yr) / 1e9,
        np.array(sfr_nodes) / float(jnp.max(sfr_nodes) + 1e-30),
        s=20,
        zorder=5,
        alpha=0.7,
    )

ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel("Normalised SFR")
ax.set_title("spline_sfh: PCHIP spline (as in ProSpect, Robotham+2020)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(ax, t_gyr, series_spl, linestyles=lss_spl, ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_spline.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### Which SFH model should I use?
#
# | Science case | Recommended model | Why |
# |-------------|-------------------|-----|
# | General galaxy fitting | **Dense basis (`dense_basis`)** | **Default. M★ is direct param, flexible shape** |
# | Stochastic / bursty SFH | `dense_basis` + field | Auto-swaps to `dense_basis_pure` + GP modulation |
# | Quick look / catalog fitting | Double power law (`dpl`) | Parametric, 4 params, fast |
# | Post-starburst / quenching | Truncated skew-normal (`tsnorm`) | Captures abrupt truncation |
# | Dwarf irregulars / starbursts | Stochastic GP (`field`) | Allows rapid SFR fluctuations |
# | Simulation calibration | Tabulated (`tabulated`) | Matches hydro output directly |
# | Agnostic / model comparison | Non-parametric bins (`continuity`) | Minimal SFH assumptions |
#
# **When in doubt:** use `dense_basis` (the default). It handles rising, declining,
# quenched, and double-peaked SFHs with just 5 parameters, and stellar mass is a
# direct parameter (not derived from SFR integration).

# %% [markdown]
# ## 2. Non-Parametric SFH Models
#
# ### 2.1 Continuity SFH (Leja+2019)
#
# Free parameters are log-SFR ratios between adjacent bins. A Student-t
# smoothness prior penalises sharp jumps.

# %%
age_yr = np.linspace(10**6.0, 10**10.14, 1200)
age_gyr = age_yr / 1e9

fig, ax = plt.subplots(figsize=(6.5, 3.2))

# Scenario A: Flat SFH (all ratios = 0)
sfr_flat = continuity_sfh(
    jnp.array(age_yr),
    log_total_mass=10.0,
    ratio_0=0.0,
    ratio_1=0.0,
    ratio_2=0.0,
    ratio_3=0.0,
    ratio_4=0.0,
    ratio_5=0.0,
)
ax.plot(age_gyr, np.array(sfr_flat), lw=1.5, label="Flat (all ratios = 0)")

# Scenario B: Rising SFH (positive ratios = younger bins have higher SFR)
sfr_rise = continuity_sfh(
    jnp.array(age_yr),
    log_total_mass=10.0,
    ratio_0=0.3,
    ratio_1=0.3,
    ratio_2=0.2,
    ratio_3=0.1,
    ratio_4=0.1,
    ratio_5=0.0,
)
ax.plot(age_gyr, np.array(sfr_rise), lw=1.5, label="Rising (positive ratios)")

# Scenario C: Declining SFH (negative ratios)
sfr_dec = continuity_sfh(
    jnp.array(age_yr),
    log_total_mass=10.0,
    ratio_0=-0.3,
    ratio_1=-0.3,
    ratio_2=-0.2,
    ratio_3=-0.1,
    ratio_4=-0.1,
    ratio_5=0.0,
)
ax.plot(age_gyr, np.array(sfr_dec), lw=1.5, label="Declining (negative ratios)")

# Scenario D: Post-starburst (one sharp dip)
sfr_psb = continuity_sfh(
    jnp.array(age_yr),
    log_total_mass=10.0,
    ratio_0=-0.8,
    ratio_1=0.5,
    ratio_2=0.2,
    ratio_3=0.0,
    ratio_4=0.0,
    ratio_5=0.0,
)
ax.plot(age_gyr, np.array(sfr_psb), lw=1.5, label="Post-starburst")

ax.set_yscale("log")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title("Continuity SFH (Leja+2019): 7 bins, 6 free ratios")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(age_gyr[-1]))
add_multi_sfh_inset(
    ax,
    age_gyr,
    [np.array(sfr_flat), np.array(sfr_rise), np.array(sfr_dec), np.array(sfr_psb)],
    ylabel="SFR",
)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_continuity_sfh.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 2.1b Bursty Continuity Prior (Tacchella+2022)
#
# The **bursty-continuity prior** uses the same `continuity_sfh` bins but applies
# a wider Student-t scale to young bins ($t_\mathrm{lookback} < t_\mathrm{split}$),
# allowing more rapid SFR fluctuations in the recent universe.
#
# | Regime | Student-t scale |
# |--------|-----------------|
# | Young bins ($< t_\mathrm{split}$) | 1.0 (wide / bursty) |
# | Old bins ($\geq t_\mathrm{split}$) | 0.3 (narrow / smooth) |
#
# Reference: Tacchella et al. 2022, ApJ 926 134 (arXiv:2102.11954).

# %%
from jax.scipy.stats import t as _student_t

bin_edges = DEFAULT_BIN_EDGES_GYR
n_bins = len(bin_edges) - 1
n_ratios = n_bins - 1

# Sweep log-SFR ratio values applied to a single bin while others stay at 0.
ratio_vals = np.linspace(-3.0, 3.0, 200)

# Standard continuity prior: Student-t(df=2, scale=0.3) on all n_ratios
logp_std = np.array(
    [float(n_ratios * _student_t.logpdf(r, 2.0, loc=0.0, scale=0.3)) for r in ratio_vals]
)

# Bursty prior: vary only the youngest ratio (bin 0 → bursty, scale=1.0)
logp_young = np.array(
    [
        float(
            bursty_continuity_prior_logp(
                jnp.concatenate([jnp.array([r]), jnp.zeros(n_ratios - 1)]),
                jnp.array(bin_edges),
                t_split_gyr=1.0,
            )
        )
        for r in ratio_vals
    ]
)

# Bursty prior: vary only the oldest ratio (old bin, scale=0.3 — same as standard)
logp_old = np.array(
    [
        float(
            bursty_continuity_prior_logp(
                jnp.concatenate([jnp.zeros(n_ratios - 1), jnp.array([r])]),
                jnp.array(bin_edges),
                t_split_gyr=1.0,
            )
        )
        for r in ratio_vals
    ]
)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

ax = axes[0]
ax.plot(ratio_vals, logp_std, lw=1.8, label="Standard continuity (scale=0.3 all bins)")
ax.plot(
    ratio_vals,
    logp_young,
    lw=1.8,
    ls="--",
    label=r"Bursty: young bin ratio varied ($t < 1$ Gyr)",
)
ax.plot(
    ratio_vals,
    logp_old,
    lw=1.8,
    ls="-.",
    label=r"Bursty: old bin ratio varied ($t \geq 1$ Gyr)",
)
ax.set_xlabel(r"log-SFR ratio")
ax.set_ylabel(r"$\log p$")
ax.set_title("Bursty vs Standard Continuity Prior\n(Tacchella+2022)")
ax.legend(fontsize=7, frameon=False)
ax.set_xlim(-3, 3)

ax = axes[1]
# Show absolute log-prob difference between bursty young and standard
diff_young = logp_young - logp_std
diff_old = logp_old - logp_std
ax.plot(ratio_vals, diff_young, lw=1.8, ls="--", label=r"Bursty young $-$ standard")
ax.plot(ratio_vals, diff_old, lw=1.8, ls="-.", label=r"Bursty old $-$ standard")
ax.axhline(0, color="k", lw=0.6, ls=":")
ax.set_xlabel(r"log-SFR ratio")
ax.set_ylabel(r"$\Delta \log p$")
ax.set_title("Log-prob difference: bursty $-$ standard")
ax.legend(fontsize=7, frameon=False)
ax.set_xlim(-3, 3)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_bursty_continuity_prior.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 2.1c psb_continuity_sfh: Post-Starburst Non-Parametric SFH (Suess+2021)
#
# Extends `continuity_sfh` with two extra parameters (`tlast_gyr`, `tflex_gyr`)
# that pin the quenching epoch. The youngest bin spans $[0, t_\mathrm{last}]$;
# a flexible zone between $t_\mathrm{last}$ and $t_\mathrm{flex}$ tracks the
# transition. Implements the same transition as Prospector (Johnson+2021).

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))

# Vary tlast: when quenching happened
ax = axes[0]
series_psbc = []
for tlast, ls in [(0.05, "-"), (0.2, "--"), (0.5, "-.")]:
    sfr = np.array(
        psb_continuity_sfh(
            jnp.array(age_yr),
            log_total_mass=10.0,
            tlast_gyr=tlast,
            tflex_gyr=2.0,
            ratio_young=-1.5,
            ratio_old_0=0.3,
            ratio_old_1=0.1,
            ratio_old_2=0.0,
        )
    )
    series_psbc.append(sfr)
    ax.plot(age_gyr, sfr, lw=1.5, ls=ls, label=rf"$t_\mathrm{{last}} = {tlast}$ Gyr")
ax.set_yscale("log")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title(r"psb\_continuity: varying $t_\mathrm{last}$ (quench epoch)")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(age_gyr[-1]))
add_multi_sfh_inset(ax, age_gyr, series_psbc, linestyles=["-", "--", "-."], ylabel="SFR")

# Vary ratio_young: depth of recent quench
ax = axes[1]
series_psbc2 = []
for ry, ls in [(-3.0, "-"), (-1.0, "--"), (0.5, "-.")]:
    sfr = np.array(
        psb_continuity_sfh(
            jnp.array(age_yr),
            log_total_mass=10.0,
            tlast_gyr=0.2,
            tflex_gyr=2.0,
            ratio_young=ry,
            ratio_old_0=0.2,
            ratio_old_1=0.0,
            ratio_old_2=0.0,
        )
    )
    series_psbc2.append(sfr)
    label = rf"ratio\_young = {ry} ({'deep quench' if ry < -1 else 'burst' if ry > 0 else 'mild quench'})"
    ax.plot(age_gyr, sfr, lw=1.5, ls=ls, label=label)
ax.set_yscale("log")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title(r"psb\_continuity: varying ratio\_young")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(age_gyr[-1]))
add_multi_sfh_inset(ax, age_gyr, series_psbc2, linestyles=["-", "--", "-."], ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_psb_continuity.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 2.2 Dirichlet SFH (Leja+2017)
#
# Mass fractions in each bin are derived from auxiliary variables via
# stick-breaking. Uniform auxiliary variables $z_i \in (0, 1)$ give a
# symmetric Dirichlet(1,...,1) prior.

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))

# Uniform mass fraction (all z = 0.5 => roughly equal fractions)
sfr_unif = dirichlet_sfh(
    jnp.array(age_yr),
    log_total_mass=10.0,
    z_frac_0=0.5,
    z_frac_1=0.5,
    z_frac_2=0.5,
    z_frac_3=0.5,
    z_frac_4=0.5,
    z_frac_5=0.5,
)
ax.plot(age_gyr, np.array(sfr_unif), lw=1.5, label="Equal fractions (z=0.5)")

# Mass concentrated in young bins
sfr_young = dirichlet_sfh(
    jnp.array(age_yr),
    log_total_mass=10.0,
    z_frac_0=0.8,
    z_frac_1=0.6,
    z_frac_2=0.3,
    z_frac_3=0.2,
    z_frac_4=0.1,
    z_frac_5=0.1,
)
ax.plot(age_gyr, np.array(sfr_young), lw=1.5, label="Mass in young bins")

# Mass concentrated in old bins
sfr_old = dirichlet_sfh(
    jnp.array(age_yr),
    log_total_mass=10.0,
    z_frac_0=0.1,
    z_frac_1=0.1,
    z_frac_2=0.2,
    z_frac_3=0.3,
    z_frac_4=0.6,
    z_frac_5=0.8,
)
ax.plot(age_gyr, np.array(sfr_old), lw=1.5, label="Mass in old bins")

# Mass in intermediate bins
sfr_mid = dirichlet_sfh(
    jnp.array(age_yr),
    log_total_mass=10.0,
    z_frac_0=0.1,
    z_frac_1=0.3,
    z_frac_2=0.8,
    z_frac_3=0.8,
    z_frac_4=0.3,
    z_frac_5=0.1,
)
ax.plot(age_gyr, np.array(sfr_mid), lw=1.5, label="Mass in intermediate bins")

ax.set_yscale("log")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title("Dirichlet SFH (Leja+2017): stick-breaking mass fractions")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(age_gyr[-1]))
add_multi_sfh_inset(
    ax,
    age_gyr,
    [
        np.array(sfr_unif),
        np.array(sfr_young),
        np.array(sfr_old),
        np.array(sfr_mid),
    ],
    ylabel="SFR",
)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_dirichlet_sfh.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 2.3 Dense Basis GP-SFH (Iyer & Gawiser 2017; Iyer et al. 2019)
#
# The **default SFH model** in tengri. Parameterises the SFH via mass-time
# quantiles: tx_frac_i is the cosmic time fraction at which the galaxy has
# formed (i+1)/(N+1) of its total stellar mass. A GP with Matérn 3/2 +
# Linear kernel smoothly interpolates the cumulative mass curve.
#
# **Key advantage**: stellar mass is a *direct parameter* (not derived from
# SFR integration), making inference more efficient.
#
# When composed with `field` (stochastic GP), automatically swaps to
# `dense_basis_pure` (no SFR constraint points) so the field has full
# control over recent SFR variability.

# %%
age_yr_db = np.linspace(10**6.0, 13.47e9, 1200)
age_gyr_db = age_yr_db / 1e9

fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

# --- Left panel: 6 canonical tutorial shapes (Iyer+2019) ---
ax = axes[0]
tutorial_shapes = {
    "Rising / starburst": (0.5, 0.7, 0.85),
    "Regular star-forming": (0.3, 0.55, 0.8),
    "Post-starburst": (0.5, 0.8, 0.9),
    "Old quenched": (0.15, 0.3, 0.5),
    "Double-peaked (SF)": (0.25, 0.30, 0.7),
    "Double-peaked (Q)": (0.1, 0.6, 0.7),
}
for label, (t0, t1, t2) in tutorial_shapes.items():
    sfr = dense_basis_sfh(
        jnp.array(age_yr_db),
        log_total_mass=10.0,
        log_sfr_inst=0.0,
        tx_frac_0=t0,
        tx_frac_1=t1,
        tx_frac_2=t2,
    )
    ax.plot(age_gyr_db, np.array(sfr), lw=1.5, label=label)

ax.set_xlabel(r"Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title(
    "Dense Basis GP-SFH (Iyer+2017, 2019)\n"
    r"$\log M_\star = 10$, $N_{\rm param} = 3$"
)
ax.legend(fontsize=6.5, frameon=False, ncol=2)
ax.set_xlim(0.0, float(age_gyr_db[-1]))

# --- Right panel: varying log_total_mass ---
ax = axes[1]
for log_m, log_sfr in [(9.0, -1.0), (10.0, 0.0), (11.0, 1.0)]:
    sfr = dense_basis_sfh(
        jnp.array(age_yr_db),
        log_total_mass=log_m,
        log_sfr_inst=log_sfr,
        tx_frac_0=0.3,
        tx_frac_1=0.55,
        tx_frac_2=0.8,
    )
    ax.plot(
        age_gyr_db,
        np.array(sfr),
        lw=1.5,
        label=rf"$\log M_\star = {log_m:.0f}$",
    )

ax.set_yscale("log")
ax.set_xlabel(r"Lookback time [Gyr]")
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title("Dense Basis: varying stellar mass")
ax.legend(fontsize=7, frameon=False)
ax.set_xlim(0.0, float(age_gyr_db[-1]))

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_dense_basis.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Stochastic (GP) SFH Model
#
# The GP modulates a smooth mean SFH:
# $\text{SFR}(t) = \bar{\text{SFR}}(t) \cdot \exp\!\bigl(x(t) - K(0)/2\bigr)$
# where $x(t) \sim \mathcal{GP}(0, P(\omega))$.
#
# ### 3.1 Mean SFH + GP Modulation

# %%
N_GRID = 256
log_ages = make_log_age_grid(N_GRID)
d_log_age = grid_spacing(log_ages)
ages_yr = 10**log_ages
ages_yr_arr = np.array(ages_yr)
ages_yr_lin = np.linspace(float(ages_yr_arr[0]), float(ages_yr_arr[-1]), 1200)
ages_gyr_lin = ages_yr_lin / 1e9


def _sfh_on_lin_grid(y):
    return np.interp(ages_yr_lin, ages_yr_arr, np.asarray(y).ravel())


key = jax.random.PRNGKey(42)

fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))

# Panel A: Mean SFH (tsnorm)
ax = axes[0]
mean_sfr = tsnorm(ages_yr, log_total_mass=10.0, peak_lbt=6e9, width=2e9, skew=0.2, trunc=3.0)
mean_lin = _sfh_on_lin_grid(mean_sfr)
ax.plot(ages_gyr_lin, mean_lin, color=COLORS["sfh_mean"], lw=2, label="Mean SFH")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title("Smooth Mean SFH")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(ages_gyr_lin[-1]))
add_sfh_inset(ax, ages_gyr_lin, mean_lin, color=COLORS["sfh_mean"], lw=1.2)

# Panel B: GP realization
ax = axes[1]
sigma, tau_myr = 0.8, 100.0
sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, sigma, tau_myr * 1e6)
gp = generate_gp_fourier(key, sqrt_p, N_GRID)
k0_half = float(drw_variance(sigma)) / 2.0
gp_lin = _sfh_on_lin_grid(gp)
ax.plot(ages_gyr_lin, gp_lin, color=COLORS["sfh_gp"], lw=1.5, alpha=0.8)
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"$x(t)$")
ax.set_title(rf"GP Realization ($\sigma={sigma}$, $\tau={tau_myr}$ Myr)")
ax.set_xlim(0.0, float(ages_gyr_lin[-1]))
add_sfh_inset(ax, ages_gyr_lin, gp_lin, color=COLORS["sfh_gp"], lw=1.2)

# Panel C: Full stochastic SFH
ax = axes[2]
full_sfr = mean_sfr * jnp.exp(gp - k0_half)
full_lin = _sfh_on_lin_grid(full_sfr)
ax.plot(ages_gyr_lin, mean_lin, color=COLORS["sfh_mean"], lw=1.5, alpha=0.5, label="Mean", ls="--")
ax.plot(ages_gyr_lin, full_lin, color=COLORS["sfh_full"], lw=1.5, label="Full SFH")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title("Mean + GP Modulation")
ax.legend(fontsize=8, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(ages_gyr_lin[-1]))
add_multi_sfh_inset(ax, ages_gyr_lin, [mean_lin, full_lin], ylabel="SFR")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_gp_sfh_demo.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3.2 DRW PSD: varying $\sigma_{\rm PS}$ and $\tau_{\rm PS}$

# %%
fig, axes = plt.subplots(2, 2, figsize=(9, 6))

# Top row: vary sigma
sigmas = [0.3, 0.6, 1.0, 1.5, 2.5]
ax = axes[0, 0]
omega = np.logspace(-3, 2, 500)  # rad / Myr
for sig in sigmas:
    psd = psd_drw(omega / 1e6, sig, 100e6)  # convert to rad/yr
    ax.loglog(omega, psd * 1e6, lw=1.5, label=rf"$\sigma = {sig}$")
ax.set_xlabel(r"$\omega$ [rad / Myr]")
ax.set_ylabel(r"$P(\omega)$ [Myr]")
ax.set_title(r"DRW PSD: vary $\sigma$ ($\tau=100$ Myr)")
ax.legend(fontsize=7, frameon=False)

ax = axes[0, 1]
series_gp_sig = []
for sig in sigmas:
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, sig, 100e6)
    gp_i = generate_gp_fourier(key, sqrt_p, N_GRID)
    gpl = _sfh_on_lin_grid(gp_i)
    series_gp_sig.append(gpl)
    ax.plot(ages_gyr_lin, gpl, lw=0.8, alpha=0.7, label=rf"$\sigma = {sig}$")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"$x(t)$")
ax.set_title(r"GP realizations: vary $\sigma$")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(ages_gyr_lin[-1]))
add_multi_sfh_inset(ax, ages_gyr_lin, series_gp_sig, ylabel=r"$x$")

# Bottom row: vary tau
taus_myr = [5, 30, 100, 300, 1000]
ax = axes[1, 0]
for tau in taus_myr:
    psd = psd_drw(omega / 1e6, 1.0, tau * 1e6)
    ax.loglog(omega, psd * 1e6, lw=1.5, label=rf"$\tau = {tau}$ Myr")
ax.set_xlabel(r"$\omega$ [rad / Myr]")
ax.set_ylabel(r"$P(\omega)$ [Myr]")
ax.set_title(r"DRW PSD: vary $\tau$ ($\sigma=1.0$)")
ax.legend(fontsize=7, frameon=False)

ax = axes[1, 1]
series_gp_tau = []
for tau in taus_myr:
    sqrt_p = compute_sqrt_power_drw(N_GRID, d_log_age, 1.0, tau * 1e6)
    gp_i = generate_gp_fourier(key, sqrt_p, N_GRID)
    gpl = _sfh_on_lin_grid(gp_i)
    series_gp_tau.append(gpl)
    ax.plot(ages_gyr_lin, gpl, lw=0.8, alpha=0.7, label=rf"$\tau = {tau}$ Myr")
ax.axhline(0, color="k", lw=0.5, ls=":")
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"$x(t)$")
ax.set_title(r"GP realizations: vary $\tau$")
ax.legend(fontsize=7, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(ages_gyr_lin[-1]))
add_multi_sfh_inset(ax, ages_gyr_lin, series_gp_tau, ylabel=r"$x$")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_drw_psd_vary.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3.3 Extended Regulator PSD (Tacchella+2020)
#
# Two-timescale PSD: regulator (gas inflow/equilibrium) + dynamical term.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))

f_myr = np.logspace(-4, 0, 500)  # cyclic frequency (1/Myr)

ax = axes[0]
# Fiducial extended regulator
psd_er = psd_extended_regulator(
    f_myr / 1e6, s_reg=0.5, tau_in=200e6, tau_eq=50e6, s_dyn=0.3, tau_dyn=20e6
)
psd_simple = psd_drw(2 * np.pi * f_myr / 1e6, 0.5, 200e6)
ax.loglog(f_myr, psd_er * 1e6, lw=2, label="Extended Regulator")
ax.loglog(f_myr, psd_simple * 1e6, lw=1.5, ls="--", label="DRW (matched)")
ax.set_xlabel(r"$f$ [1 / Myr]")
ax.set_ylabel(r"$P(f)$ [Myr]")
ax.set_title("Extended Regulator vs DRW")
ax.legend(fontsize=8, frameon=False)

# Panel B: Vary dynamical term
ax = axes[1]
for s_dyn, tau_dyn in [(0.1, 10e6), (0.3, 20e6), (0.5, 5e6), (0.8, 30e6)]:
    psd_i = psd_extended_regulator(
        f_myr / 1e6, s_reg=0.5, tau_in=200e6, tau_eq=50e6, s_dyn=s_dyn, tau_dyn=tau_dyn
    )
    td_myr = tau_dyn / 1e6
    ax.loglog(
        f_myr,
        psd_i * 1e6,
        lw=1.5,
        label=rf"$s_{{\rm dyn}}={s_dyn}$, $\tau_{{\rm dyn}}={td_myr:.0f}$ Myr",
    )
ax.set_xlabel(r"$f$ [1 / Myr]")
ax.set_ylabel(r"$P(f)$ [Myr]")
ax.set_title("Varying dynamical component")
ax.legend(fontsize=7, frameon=False)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_extended_regulator.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 3.4 Matern PSD: varying $\nu$
#
# $\nu = 0.5$ recovers the DRW. Higher $\nu$ damps high-frequency
# fluctuations more strongly, giving smoother GP realizations.

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.3))

omega_rad_yr = np.logspace(-9, -4, 500)  # rad/yr

ax = axes[0]
for nu in [0.5, 1.5, 2.5, 5.0]:
    psd_m = psd_matern(omega_rad_yr, variance=1.0, length_scale=100e6, nu=nu)
    label_str = r"$\nu = 0.5$ (DRW)" if nu == 0.5 else rf"$\nu = {nu}$"
    ax.loglog(omega_rad_yr * 1e6, psd_m * 1e6, lw=1.5, label=label_str)
ax.set_xlabel(r"$\omega$ [rad / Myr]")
ax.set_ylabel(r"$P(\omega)$ [Myr]")
ax.set_title(r"Matern PSD: varying $\nu$")
ax.legend(fontsize=8, frameon=False)

# Note on high-frequency slopes
ax = axes[1]
for nu in [0.5, 1.5, 2.5, 5.0]:
    psd_m = psd_matern(omega_rad_yr, variance=1.0, length_scale=100e6, nu=nu)
    slope = -(2 * nu + 1)
    label_str = rf"$\nu = {nu}$, slope $\propto \omega^{{{slope:.0f}}}$"
    ax.loglog(omega_rad_yr * 1e6, psd_m * 1e6, lw=1.5, label=label_str)
ax.set_xlabel(r"$\omega$ [rad / Myr]")
ax.set_ylabel(r"$P(\omega)$ [Myr]")
ax.set_title("Matern: high-frequency spectral slope")
ax.legend(fontsize=7, frameon=False)

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_matern_psd.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Composition System
#
# The SFH registry supports three composition types:
# - **additive**: sum of smooth components (e.g., tsnorm + const)
# - **mixture**: smooth base with burst admixture (e.g., tsnorm + burst)
# - **modulator**: multiplicative GP field (e.g., tsnorm + field)
