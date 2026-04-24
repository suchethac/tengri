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
# # Star Formation History — Advanced Topics
#
# Combine multiple SFH components and evolve metallicity self-consistently from the SFH.
#
# ## What you'll learn
#
# - **Composition system** — additive (multi-component), burst mixture, GP field modulation
# - **Closed-box and leaky-box metallicity evolution** — how Z(t) depends on SFH shape and mass loss
# - **Physical interpretation** — why late-forming histories produce shallower Z(t) gradients
#
# ## Prerequisites
#
# [`02_sed_anatomy.py`](02_sed_anatomy.py) for the forward model; `examples/sfh/` for parametric / non-parametric SFH shapes.
# Useful for galaxy populations with complex assembly histories or when metallicity is a science goal.

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
    dpl,
    drw_variance,
    exponential_sfh,
    generate_gp_fourier,
    make_log_age_grid,
    tsnorm,
    triweight_burst,
)
from tengri.sfh.chemical_evolution import closed_box_metallicity
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
# ## 4. Composition System
#
# The SFH registry supports three composition types:
# - **additive**: sum of smooth components (e.g., tsnorm + const)
# - **mixture**: smooth base with burst admixture (e.g., tsnorm + burst)
# - **modulator**: multiplicative GP field (e.g., tsnorm + field)

# %%
# Setup shared grids for composition and chemical evolution demos
t_yr = np.linspace(10**6.0, 10**10.14, 2000)
t_gyr = t_yr / 1e9

N_GRID = 256
log_ages = make_log_age_grid(N_GRID)
d_log_age = grid_spacing(log_ages)
ages_yr = 10**log_ages
ages_yr_arr = np.array(ages_yr)
ages_yr_lin = np.linspace(float(ages_yr_arr[0]), float(ages_yr_arr[-1]), 1200)
ages_gyr_lin = ages_yr_lin / 1e9


def _sfh_on_lin_grid(y):
    return np.interp(ages_yr_lin, ages_yr_arr, np.asarray(y).ravel())


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
    ax_in.set_xlabel("Lookback (Myr)", fontsize=10)
    ax_in.set_ylabel(ylabel, fontsize=10)
    ax_in.tick_params(labelsize=5)
    return ax_in


# %% [markdown]
# ### 4.1 Additive: DPL + constant

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))

sfr_dpl_only = dpl(t_yr, alpha=2.0, beta=1.0, tau=5e9, log_peak_sfr=0.5)
sfr_const_part = constant_sfh(t_yr, log_sfr=-0.5)
sfr_sum = np.array(sfr_dpl_only) + np.array(sfr_const_part)

ax.plot(
    t_gyr,
    np.array(sfr_dpl_only),
    ls="--",
    lw=1.5,
    color=COLORS["sfh_mean"],
    label="DPL alone",
    alpha=0.6,
)
ax.plot(
    t_gyr,
    np.array(sfr_const_part),
    ls=":",
    lw=1.5,
    color=COLORS["sfh_gp"],
    label="Constant floor",
    alpha=0.6,
)
ax.plot(t_gyr, sfr_sum, lw=2, color=COLORS["sfh_full"], label="DPL + constant")

ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title("Additive Composition: DPL + Constant")
ax.legend(fontsize=10, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(t_gyr[-1]))
add_multi_sfh_inset(
    ax,
    t_gyr,
    [np.array(sfr_dpl_only), np.array(sfr_const_part), sfr_sum],
    colors=[COLORS["sfh_mean"], COLORS["sfh_gp"], COLORS["sfh_full"]],
    linestyles=["--", ":", "-"],
    lws=[1.2, 1.2, 1.5],
    ylabel="SFR",
)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_composition_additive.png", dpi=300, bbox_inches="tight"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 4.2 Burst Mixture: smooth + triweight burst
#
# $\text{SFR}(t) = (1 - f_{\rm burst}) \cdot \text{smooth}(t)
#   + f_{\rm burst} \cdot \text{burst}(t)$

# %%
fig, axes = plt.subplots(1, 3, figsize=(12, 3.3))

smooth_base = tsnorm(t_yr, log_peak_sfr=0.5, peak_lbt=6e9, width=2e9, skew=0.2, trunc=3.0)
burst_shape = triweight_burst(t_yr, log_tpeak_myr=2.0, log_tmax_myr=2.5)
burst_shape_norm = burst_shape * np.max(smooth_base) / (np.max(burst_shape) + 1e-30)

for f_burst, ax in zip([0.01, 0.1, 0.3], axes, strict=True):
    combined = (1.0 - f_burst) * np.array(smooth_base) + f_burst * np.array(burst_shape_norm)
    ax.plot(
        t_gyr,
        np.array(smooth_base),
        ls="--",
        lw=1,
        color=COLORS["sfh_mean"],
        alpha=0.5,
        label="Smooth",
    )
    ax.plot(
        t_gyr,
        combined,
        lw=1.8,
        color=COLORS["sfh_full"],
        label=f"Combined ($f_{{\\rm burst}}={f_burst}$)",
    )
    ax.set_xlabel(XLAB_LBT_GYR)
    ax.set_ylabel(r"SFR [M$_\odot$/yr]")
    ax.set_title(rf"$f_{{\rm burst}} = {f_burst}$")
    ax.legend(fontsize=10, frameon=False, loc="lower left")
    ax.set_xlim(0.0, float(t_gyr[-1]))
    add_multi_sfh_inset(
        ax,
        t_gyr,
        [np.array(smooth_base), combined],
        colors=[COLORS["sfh_mean"], COLORS["sfh_full"]],
        linestyles=["--", "-"],
        lws=[1.0, 1.2],
        ylabel="SFR",
    )

fig.suptitle("Burst Mixture: tsnorm + triweight burst", y=1.02, fontsize=11)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_composition_burst.png", dpi=300, bbox_inches="tight"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ### 4.3 Field Modulation: tsnorm $\times \exp(x(t) - K(0)/2)$
#
# Multiple GP realizations showing how burstiness modulates the smooth SFH.

# %%
fig, ax = plt.subplots(figsize=(6.5, 3.2))

mean_base = tsnorm(ages_yr, log_peak_sfr=0.5, peak_lbt=6e9, width=2e9, skew=0.2, trunc=3.0)
sigma_field = 0.8
tau_field_yr = 100e6
sqrt_p_field = compute_sqrt_power_drw(N_GRID, d_log_age, sigma_field, tau_field_yr)
k0_half_field = float(drw_variance(sigma_field)) / 2.0

mean_base_lin = _sfh_on_lin_grid(mean_base)
ax.plot(
    ages_gyr_lin,
    mean_base_lin,
    color=COLORS["sfh_mean"],
    lw=2.5,
    ls="--",
    label="Mean SFH",
    zorder=5,
)

field_draws_lin = []
for i in range(6):
    key_i = jax.random.PRNGKey(100 + i)
    gp_i = generate_gp_fourier(key_i, sqrt_p_field, N_GRID)
    full_i = mean_base * jnp.exp(gp_i - k0_half_field)
    fl = _sfh_on_lin_grid(full_i)
    field_draws_lin.append(fl)
    ax.plot(ages_gyr_lin, fl, lw=0.8, alpha=0.5, color=COLORS["sfh_full"])

ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"SFR [M$_\odot$/yr]")
ax.set_title(
    rf"Field Modulation: 6 GP draws ($\sigma={sigma_field}$, $\tau={int(tau_field_yr / 1e6)}$ Myr)"
)
ax.legend(fontsize=10, frameon=False, loc="lower left")
ax.set_xlim(0.0, float(ages_gyr_lin[-1]))
add_multi_sfh_inset(ax, ages_gyr_lin, [mean_base_lin, *field_draws_lin], ylabel="SFR")
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_composition_field.png", dpi=300, bbox_inches="tight"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Chemical Evolution
#
# The closed-box / leaky-box model derives $Z(t)$ from the SFH:
# $Z(t) = y_{\rm eff} \cdot \ln(1 / f_{\rm gas}(t))$
# where $y_{\rm eff} = y / (1 + \eta)$.
#
# ### 5.1 Closed-box vs Leaky-box

# %%
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

# Use a tsnorm SFH for the chemical evolution demo
sfr_chem = np.array(
    tsnorm(ages_yr, log_peak_sfr=0.5, peak_lbt=6e9, width=2e9, skew=0.2, trunc=3.0)
)

# Panel A: Closed-box (eta=0) vs leaky-box (eta > 0)
ax = axes[0]
eta_specs = [
    (0.0, "-", "Closed box ($\\eta=0$)"),
    (0.5, "--", "Leaky ($\\eta=0.5$)"),
    (2.0, "-.", "Leaky ($\\eta=2.0$)"),
    (5.0, ":", "Leaky ($\\eta=5.0$)"),
]
series_z_eta = []
for eta, ls, label in eta_specs:
    logz = closed_box_metallicity(ages_yr, jnp.array(sfr_chem), yield_y=0.03, eta_outflow=eta)
    z_lin = _sfh_on_lin_grid(logz)
    series_z_eta.append(z_lin)
    ax.plot(ages_gyr_lin, z_lin, lw=1.5, ls=ls, label=label)
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"$\log_{10}(Z/Z_\odot)$")
ax.set_title("Closed-box vs Leaky-box")
ax.legend(fontsize=10, frameon=False, loc="lower left")
ax.axhline(0.0, color="grey", lw=0.5, ls=":")
ax.set_xlim(0.0, float(ages_gyr_lin[-1]))
add_multi_sfh_inset(ax, ages_gyr_lin, series_z_eta, ylabel=r"$\log Z$")

# Panel B: Effect of different SFH shapes on Z(t)
ax = axes[1]
sfh_set = {
    "DPL (late-forming)": np.array(dpl(ages_yr, alpha=3.0, beta=1.0, tau=3e9, log_peak_sfr=0.5)),
    "tsnorm (peaked)": np.array(
        tsnorm(ages_yr, log_peak_sfr=0.5, peak_lbt=6e9, width=2e9, skew=0.2, trunc=3.0)
    ),
    "Constant": np.array(constant_sfh(ages_yr, log_sfr=0.0)),
    "Declining exp": np.array(exponential_sfh(ages_yr, log_peak_sfr=0.5, tau=3e9)),
}

series_z_shape = []
for name, sfr_i in sfh_set.items():
    logz_i = closed_box_metallicity(ages_yr, jnp.array(sfr_i), yield_y=0.03, eta_outflow=0.5)
    z_lin = _sfh_on_lin_grid(logz_i)
    series_z_shape.append(z_lin)
    ax.plot(ages_gyr_lin, z_lin, lw=1.5, label=name)
ax.set_xlabel(XLAB_LBT_GYR)
ax.set_ylabel(r"$\log_{10}(Z/Z_\odot)$")
ax.set_title(r"$Z(t)$ from different SFH shapes ($\eta=0.5$)")
ax.legend(fontsize=10, frameon=False, loc="lower left")
ax.axhline(0.0, color="grey", lw=0.5, ls=":")
ax.set_xlim(0.0, float(ages_gyr_lin[-1]))
add_multi_sfh_inset(ax, ages_gyr_lin, series_z_shape, ylabel=r"$\log Z$")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "sfh_chemical_evolution.png", dpi=300, bbox_inches="tight"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Summary
#
# **Composition** combines multiple SFH components: additive (base + floor), burst (base + burst mixture),
# and field (mean + GP modulation). The registry enforces constraints: at least one additive model,
# at most one burst and one field. `dense_basis` auto-swaps to `dense_basis_pure` when a field is active.
#
# **Chemical evolution** derives $Z(t)$ from the SFH via closed-box / leaky-box models. The evolution
# depends on SFH shape: late-forming (DPL) produces shallower $Z(t)$ gradients; early-forming (peaked)
# produces steeper ones. The `eta_outflow` parameter controls mass-loss rate; higher $\eta$ produces
# lower final metallicity.

# %%
# %% [markdown]
# ## What you learned
#
# - SFH composition combines smooth, burst, and field-modulated components
# - Chemical evolution via closed-box/leaky-box: Z(t) depends on SFH shape and η_outflow
# - Late-forming histories (DPL) produce shallower [Z/H] gradients; peaked histories produce steeper ones
#
# **Next:** [`09_dust_emission.py`](09_dust_emission.py) (dust IR emission models) or
# [`11_population.py`](11_population.py) (hierarchical inference across galaxy samples).
