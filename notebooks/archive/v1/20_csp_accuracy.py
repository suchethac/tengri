# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NB20 — CSP Integral Accuracy & Metallicity Interpolation
#
# **Goal:** Compare tengri's CSP integral against the DSPS reference,
# diagnose where differences arise, and demonstrate the new smooth
# triweight metallicity interpolation that matches DSPS exactly.
#
# ## The CSP integral
#
# Every SED code computes a composite stellar population by weighting
# SSP templates. Two choices control accuracy:
#
# 1. **Age weights** $w_i$ — how much stellar mass formed at each SSP age
# 2. **Metallicity weights** $\phi_m$ — how to combine SSP metallicity templates
#
# $$L_\mathrm{CSP}(\lambda) = \sum_{i,m} w_i \, \phi_m \, \mathrm{SSP}(\lambda, Z_m, \mathrm{age}_i)$$
#
# This notebook shows that tengri can use either:
# - **2-point linear** interpolation in $\log(Z)$ (same as FSPS/Prospector)
# - **Triweight kernel** (same as DSPS, Hearin+2023) with smooth $C^2$ gradients
#
# Both are available via `ParamSpec(met_interp="linear")` or
# `ParamSpec(met_interp="smooth", lgmet_scatter=0.1)`.
#
# ## Important: metallicity convention
#
# The SSP grid stores **absolute** $\log_{10}(Z)$ values, not solar-relative.
# The user-facing parameter `met_logzsol` is in $\log_{10}(Z/Z_\odot)$.
# The conversion is:
#
# $$\log_{10}(Z)_\mathrm{abs} = \log_{10}(Z/Z_\odot) + \log_{10}(Z_\odot)$$
#
# where $\log_{10}(Z_\odot) = -1.8477$ (Asplund+2009). The low-level functions
# in this notebook use **absolute** $\log_{10}(Z)$ directly.

# %%
import sys
sys.path.insert(0, "../src")

import time
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from _plot_style import setup_style, COLORS

setup_style()
jax.config.update("jax_enable_x64", True)

# %% [markdown]
# ## 1. Load SSP templates

# %%
from tengri.sps.dsps_wrapper import (
    load_ssp_data,
    compute_csp_weights,
    compute_csp_sed,
    interpolate_metallicity,
    interpolate_metallicity_smooth,
    compute_lgmet_weights,
    LSUN_ERG_PER_S,
)

SSP_PATH = "../data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
ssp = load_ssp_data(SSP_PATH)

LOG10_ZSUN = -1.8477  # Asplund+2009

print(f"SSP grid: {ssp.ssp_flux.shape[0]} Z x "
      f"{ssp.ssp_flux.shape[1]} ages x {ssp.ssp_flux.shape[2]} wavelengths")
print(f"Z grid (absolute log10 Z): {np.array(ssp.ssp_lgmet)}")
print(f"Z grid (solar-relative):   {np.array(ssp.ssp_lgmet) - LOG10_ZSUN}")
print(f"Grid spacing: {np.diff(np.array(ssp.ssp_lgmet))}")
print(f"Age range: {10**float(ssp.ssp_lg_age_gyr[0]):.4f} - "
      f"{10**float(ssp.ssp_lg_age_gyr[-1]):.2f} Gyr")

# %% [markdown]
# ## 2. Define test SFHs
#
# Three shapes that stress different aspects of the CSP integral:
# - **Constant**: tests total mass normalization
# - **Burst**: tests single-age accuracy
# - **Declining**: tests weight distribution across the full age range

# %%
from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z

z_obs = 0.1
t_obs_gyr = np.asarray(age_at_z(z_obs, *DEFAULT_COSMOLOGY)).item()
t_table = jnp.linspace(0.01, t_obs_gyr, 200)

sfr_constant = jnp.ones(200)
sfr_burst = jnp.exp(-0.5 * ((t_table - 2.0) / 0.1) ** 2) / (0.1 * jnp.sqrt(2 * jnp.pi)) * 10.0
sfr_declining = jnp.where(t_table > 1.0, 10.0 * jnp.exp(-(t_table - 1.0) / 3.0), 0.0)

sfh_models = {
    "Constant": sfr_constant,
    "Burst (2 Gyr)": sfr_burst,
    r"Declining ($\tau$=3 Gyr)": sfr_declining,
}

fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharey=False)
for ax, (name, sfr) in zip(axes, sfh_models.items()):
    ax.plot(t_table, sfr, color=COLORS["truth"], lw=2)
    ax.set_xlabel("Cosmic time (Gyr)")
    ax.set_ylabel(r"SFR ($M_\odot\,\mathrm{yr}^{-1}$)")
    ax.set_title(name)
    total = float(jnp.trapezoid(sfr, t_table * 1e9))
    ax.text(0.95, 0.90, f"$M_{{\\star,formed}}$ = {total:.1e} $M_\\odot$",
            transform=ax.transAxes, ha="right", fontsize=9)
fig.tight_layout()
plt.savefig("figures/20_test_sfhs.pdf")
plt.show()

# %% [markdown]
# ## 3. How codes differ
#
# | Code | Age integration | Z interpolation |
# |------|----------------|-----------------|
# | **FSPS** (Conroy+2009) | analytic piecewise | 2-point linear in log(Z) |
# | **Bagpipes** (Carnall+2018) | fine-grid histogram | linear in Z |
# | **Prospector** (Johnson+2021) | delegates to FSPS | delegates to FSPS |
# | **CIGALE** (Boquien+2019) | grid-based | nearest-neighbor |
# | **DSPS** (Hearin+2023) | cumulative mass interp | triweight kernel |
# | **tengri** (this work) | DSPS cumulative | **both** (selectable) |

# %% [markdown]
# ## 4. DSPS reference SEDs

# %%
from dsps.sed import calc_rest_sed_sfh_table_lognormal_mdf
from dsps.sed.ssp_weights import calc_age_weights_from_sfh_table

# Test at solar-relative -0.3 → absolute -2.148
log_z_solar = -0.3
log_z_abs = log_z_solar + LOG10_ZSUN  # -2.148 (absolute)
print(f"Test metallicity: log10(Z/Zsun) = {log_z_solar}, log10(Z) = {log_z_abs:.3f}")

def dsps_reference_sed(sfr_table, log_z_absolute, lgmet_scatter):
    """Full DSPS reference SED. log_z_absolute is log10(Z), matching ssp_lgmet."""
    return calc_rest_sed_sfh_table_lognormal_mdf(
        gal_t_table=t_table, gal_sfr_table=sfr_table,
        gal_lgmet=log_z_absolute, gal_lgmet_scatter=lgmet_scatter,
        ssp_lgmet=ssp.ssp_lgmet, ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_flux=ssp.ssp_flux, t_obs=t_obs_gyr,
    )

def _get_dsps_age_weights(sfr_table):
    """Extract DSPS age weights (handles v0.4.x return types)."""
    result = calc_age_weights_from_sfh_table(
        gal_t_table=t_table, gal_sfr_table=sfr_table,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr, t_obs=t_obs_gyr,
    )
    return result.age_weights if hasattr(result, "age_weights") else jnp.asarray(result)

print("Computing DSPS reference SEDs...")
dsps_results = {}
for name, sfr in sfh_models.items():
    dsps_results[name] = {
        "scatter_0.1": dsps_reference_sed(sfr, log_z_abs, 0.1),
        "scatter_0.001": dsps_reference_sed(sfr, log_z_abs, 0.001),
    }
    print(f"  {name}: done")

# %% [markdown]
# ## 5. Four methods compared
#
# | Method | Age weights | Z interpolation |
# |--------|------------|-----------------|
# | **A**: tengri standard | midpoint rule | 2-point linear |
# | **B**: tengri + DSPS age | DSPS cumulative | 2-point linear |
# | **C**: tengri smooth | DSPS cumulative | triweight kernel |
# | **Ref**: full DSPS | DSPS cumulative | triweight kernel |

# %%
wave = np.array(ssp.ssp_wave)

def method_a(sfr_table):
    """tengri standard: midpoint weights + 2-point Z."""
    ssp_ages_gyr = 10.0 ** ssp.ssp_lg_age_gyr
    ssp_ages_yr = ssp_ages_gyr * 1e9
    t_lookback = t_obs_gyr - t_table
    sfr_at_ssp = jnp.interp(ssp_ages_gyr, t_lookback[::-1], sfr_table[::-1],
                              left=0.0, right=0.0)
    weights = compute_csp_weights(sfr_at_ssp, ssp_ages_yr)
    ssp_at_z = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, log_z_abs)
    dust = jnp.ones_like(ssp_at_z)
    return compute_csp_sed(weights, ssp_at_z, dust), weights

def method_b(sfr_table):
    """DSPS age weights + 2-point linear Z."""
    age_w = _get_dsps_age_weights(sfr_table)
    total_mass = jnp.trapezoid(sfr_table, t_table * 1e9)
    weights = age_w * total_mass
    ssp_at_z = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, log_z_abs)
    dust = jnp.ones_like(ssp_at_z)
    return compute_csp_sed(weights, ssp_at_z, dust), weights

def method_c(sfr_table):
    """DSPS age weights + triweight smooth Z (tengri native)."""
    age_w = _get_dsps_age_weights(sfr_table)
    total_mass = jnp.trapezoid(sfr_table, t_table * 1e9)
    weights = age_w * total_mass
    ssp_at_z = interpolate_metallicity_smooth(ssp.ssp_flux, ssp.ssp_lgmet, log_z_abs, 0.1)
    dust = jnp.ones_like(ssp_at_z)
    return compute_csp_sed(weights, ssp_at_z, dust), weights

print("Computing tengri SEDs (3 methods)...")
results = {}
for name, sfr in sfh_models.items():
    sed_a, w_a = method_a(sfr)
    sed_b, w_b = method_b(sfr)
    sed_c, w_c = method_c(sfr)
    results[name] = {
        "A": {"sed": np.array(sed_a), "weights": np.array(w_a)},
        "B": {"sed": np.array(sed_b), "weights": np.array(w_b)},
        "C": {"sed": np.array(sed_c), "weights": np.array(w_c)},
    }
    print(f"  {name}: done")

# %% [markdown]
# ## 6. Age weight comparison

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
ssp_ages_gyr = 10.0 ** np.array(ssp.ssp_lg_age_gyr)

for ax, (name, sfr) in zip(axes, sfh_models.items()):
    w_dsps = np.array(_get_dsps_age_weights(sfr))
    w_midpt = results[name]["A"]["weights"]

    total_midpt = w_midpt.sum()
    total_dsps = max(w_dsps.sum(), 1e-30)

    ax.step(ssp_ages_gyr, w_dsps / total_dsps, where="mid",
            color=COLORS["truth"], lw=2, label="DSPS (cumulative)")
    ax.step(ssp_ages_gyr, w_midpt / total_midpt, where="mid",
            color=COLORS["rt"], lw=1.5, ls="--", label="tengri (midpoint)")
    ax.set_xscale("log")
    ax.set_xlabel("SSP age (Gyr)")
    ax.set_ylabel("Normalized weight")
    ax.set_title(name)
    ax.legend(fontsize=8)

    total_true = float(jnp.trapezoid(sfr, t_table * 1e9))
    ax.text(0.95, 0.70,
            f"$M_{{true}}$ = {total_true:.2e}\n"
            f"$M_{{midpt}}$ = {total_midpt:.2e}\n"
            f"Ratio = {total_midpt / total_true:.3f}",
            transform=ax.transAxes, ha="right", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

fig.suptitle("Age weight comparison", fontsize=14, y=1.02)
fig.tight_layout()
plt.savefig("figures/20_age_weights.pdf")
plt.show()

# %% [markdown]
# ## 7. Full SED comparison

# %%
fig, axes = plt.subplots(2, 3, figsize=(16, 8),
                          gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

method_styles = {
    "A": {"color": COLORS["rt"],    "ls": "--",  "label": "A: midpoint + 2pt"},
    "B": {"color": COLORS["geovi"], "ls": "-.",   "label": "B: DSPS-age + 2pt"},
    "C": {"color": COLORS["nuts"],  "ls": "-",   "label": "C: DSPS-age + triweight"},
}

for col, (name, sfr) in enumerate(sfh_models.items()):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    sed_ref = np.array(dsps_results[name]["scatter_0.1"].rest_sed) * LSUN_ERG_PER_S
    safe = np.where(sed_ref > 0, sed_ref, 1.0)

    ax_top.plot(wave, sed_ref, color=COLORS["truth"], lw=2.5,
                label="DSPS ref", zorder=10)

    for key, sty in method_styles.items():
        sed = results[name][key]["sed"]
        ax_top.plot(wave, sed, color=sty["color"], ls=sty["ls"],
                    lw=1.5, label=sty["label"])
        res = (sed - sed_ref) / safe * 100
        ax_bot.plot(wave, res, color=sty["color"], ls=sty["ls"], lw=1.2)

    ax_top.set_xscale("log"); ax_top.set_yscale("log")
    ax_top.set_title(name); ax_top.set_xlim(900, 30000)
    if col == 0:
        ax_top.set_ylabel(r"$L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)")
        ax_top.legend(fontsize=7, loc="lower left")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_bot.axhline(0, color="0.5", ls="-", lw=0.5)
    ax_bot.axhspan(-2, 2, color="0.9", alpha=0.3)
    ax_bot.set_xscale("log"); ax_bot.set_xlabel(r"$\lambda$ ($\AA$)")
    ax_bot.set_xlim(900, 30000); ax_bot.set_ylim(-15, 15)
    if col == 0:
        ax_bot.set_ylabel("Residual (%)")

    mask = (wave > 1000) & (wave < 20000)
    for key in ["A", "B", "C"]:
        rms = np.sqrt(np.mean(((results[name][key]["sed"][mask] - sed_ref[mask]) / safe[mask]) ** 2)) * 100
        print(f"  {name:20s} | Method {key} | RMS = {rms:.2f}%")

fig.suptitle(f"CSP SED comparison at log(Z/Zsun) = {log_z_solar}", fontsize=14, y=1.02)
fig.tight_layout()
plt.savefig("figures/20_sed_comparison.pdf")
plt.show()

# %% [markdown]
# ## 8. Metallicity weight kernels

# %%
from dsps.sed.ssp_weights import calc_lgmet_weights_from_lognormal_mdf

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: kernels at different scatter values
for scatter, ls, alpha, label in [
    (0.001, ":",  0.5, r"$\sigma\approx 0$"),
    (0.05,  "--", 0.7, r"$\sigma = 0.05$"),
    (0.1,   "-",  1.0, r"$\sigma = 0.1$ (default)"),
    (0.3,   "-.", 0.7, r"$\sigma = 0.3$"),
]:
    w_dsps = np.array(calc_lgmet_weights_from_lognormal_mdf(log_z_abs, scatter, ssp.ssp_lgmet))
    lgmet_solar = np.array(ssp.ssp_lgmet) - LOG10_ZSUN
    axes[0].step(lgmet_solar, w_dsps, where="mid", ls=ls, lw=2, alpha=alpha, label=label)

# 2-point weight
lgmet = np.array(ssp.ssp_lgmet)
idx = np.searchsorted(lgmet, log_z_abs) - 1
idx = max(0, min(idx, len(lgmet) - 2))
frac = (log_z_abs - lgmet[idx]) / (lgmet[idx + 1] - lgmet[idx])
w_2pt = np.zeros(len(lgmet))
w_2pt[idx] = 1 - frac
w_2pt[idx + 1] = frac
axes[0].step(lgmet - LOG10_ZSUN, w_2pt, where="mid", color=COLORS["model"], lw=2.5,
             label="2-point linear")

axes[0].set_xlabel(r"$\log_{10}(Z/Z_\odot)$")
axes[0].set_ylabel("Weight")
axes[0].set_title(r"Metallicity kernels at $\log(Z/Z_\odot)=-0.3$")
axes[0].legend(fontsize=8)

# Right: tengri vs DSPS cross-validation
scatter_vals = [0.05, 0.1, 0.2, 0.3]
max_diffs = []
for sc in scatter_vals:
    w_ours = np.array(compute_lgmet_weights(log_z_abs, ssp.ssp_lgmet, sc))
    w_dsps = np.array(calc_lgmet_weights_from_lognormal_mdf(log_z_abs, sc, ssp.ssp_lgmet))
    max_diffs.append(np.max(np.abs(w_ours - w_dsps)))

axes[1].semilogy(scatter_vals, max_diffs, "o-", color=COLORS["rt"], lw=2, ms=8)
axes[1].axhline(1e-4, color="0.7", ls=":", lw=0.8, label=r"$10^{-4}$")
axes[1].set_xlabel(r"lgmet\_scatter (dex)")
axes[1].set_ylabel("|w_tengri - w_DSPS|_max")
axes[1].set_title("tengri triweight vs DSPS: cross-validation")
axes[1].legend()

fig.tight_layout()
plt.savefig("figures/20_z_kernels.pdf")
plt.show()

for sc, md in zip(scatter_vals, max_diffs):
    print(f"  scatter={sc:.2f}: max_diff = {md:.2e}")

# %% [markdown]
# ## 9. Gradient smoothness
#
# The triweight kernel gives C$^2$ continuous derivatives. The 2-point
# linear scheme has kinks at every Z grid boundary.

# %%
def _make_sed_fn(interp_fn, scatter=None):
    """Create V-band luminosity as f(log_z) for gradient analysis."""
    age_w = _get_dsps_age_weights(sfr_constant)
    total_mass = jnp.trapezoid(sfr_constant, t_table * 1e9)
    weights = age_w * total_mass
    iband = (ssp.ssp_wave > 5000) & (ssp.ssp_wave < 6000)

    def _fn(log_z_absolute):
        if scatter is not None:
            ssp_at_z = interp_fn(ssp.ssp_flux, ssp.ssp_lgmet, log_z_absolute, scatter)
        else:
            ssp_at_z = interp_fn(ssp.ssp_flux, ssp.ssp_lgmet, log_z_absolute)
        dust = jnp.ones_like(ssp_at_z)
        sed = compute_csp_sed(weights, ssp_at_z, dust)
        return jnp.sum(sed * iband)

    return _fn

sed_fn_linear = _make_sed_fn(interpolate_metallicity)
sed_fn_smooth = _make_sed_fn(interpolate_metallicity_smooth, scatter=0.1)

grad_linear = jax.grad(sed_fn_linear)
grad_smooth = jax.grad(sed_fn_smooth)

# Scan in absolute log10(Z) space, within the grid
z_lo, z_hi = float(ssp.ssp_lgmet[2]), float(ssp.ssp_lgmet[-2])
log_z_scan = jnp.linspace(z_lo, z_hi, 80)
log_z_scan_solar = log_z_scan - LOG10_ZSUN  # for x-axis

print("Computing V-band luminosity + gradients...")
lum_linear = np.array([float(sed_fn_linear(z)) for z in log_z_scan])
lum_smooth = np.array([float(sed_fn_smooth(z)) for z in log_z_scan])
g_linear = np.array([float(grad_linear(z)) for z in log_z_scan])
g_smooth = np.array([float(grad_smooth(z)) for z in log_z_scan])
print("  Done.")

# %%
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                          gridspec_kw={"height_ratios": [2, 1.5], "hspace": 0.05})

# Panel 1: luminosity
ax = axes[0]
ax.plot(log_z_scan_solar, lum_linear, color=COLORS["rt"], lw=2, label="2-point linear")
ax.plot(log_z_scan_solar, lum_smooth, color=COLORS["nuts"], lw=2, ls="--", label="triweight (scatter=0.1)")
for z_grid in np.array(ssp.ssp_lgmet) - LOG10_ZSUN:
    ax.axvline(z_grid, color="0.88", ls=":", lw=0.5, zorder=0)
ax.set_ylabel(r"$L_\nu$ (V-band, erg/s/Hz)")
ax.legend()
ax.set_title("V-band luminosity and gradient vs metallicity")

# Panel 2: gradient
ax = axes[1]
ax.plot(log_z_scan_solar, g_linear, color=COLORS["rt"], lw=2,
        label=r"$\partial L/\partial \log Z$ (linear)")
ax.plot(log_z_scan_solar, g_smooth, color=COLORS["nuts"], lw=2, ls="--",
        label=r"$\partial L/\partial \log Z$ (triweight)")
ax.axhline(0, color="0.5", ls="-", lw=0.5)
for z_grid in np.array(ssp.ssp_lgmet) - LOG10_ZSUN:
    ax.axvline(z_grid, color="0.88", ls=":", lw=0.5, zorder=0)
ax.set_xlabel(r"$\log_{10}(Z/Z_\odot)$")
ax.set_ylabel(r"$\partial L/\partial \log Z$")
ax.legend(fontsize=9)

fig.tight_layout()
plt.savefig("figures/20_gradient_smoothness.pdf")
plt.show()

print("Gradient jump (max |diff| of consecutive values):")
print(f"  linear: {np.max(np.abs(np.diff(g_linear))):.3e}")
print(f"  smooth: {np.max(np.abs(np.diff(g_smooth))):.3e}")
print(f"  ratio:  {np.max(np.abs(np.diff(g_linear))) / max(np.max(np.abs(np.diff(g_smooth))), 1e-30):.1f}x smoother")

# %% [markdown]
# ## 10. Speed benchmarks

# %%
# Warmup
_ = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, log_z_abs).block_until_ready()
_ = interpolate_metallicity_smooth(ssp.ssp_flux, ssp.ssp_lgmet, log_z_abs, 0.1).block_until_ready()
_ = dsps_reference_sed(sfr_constant, log_z_abs, 0.1).rest_sed.block_until_ready()

N = 1000

t0 = time.perf_counter()
for _ in range(N):
    r = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, log_z_abs)
r.block_until_ready()
t_linear = (time.perf_counter() - t0) / N * 1e6

t0 = time.perf_counter()
for _ in range(N):
    r = interpolate_metallicity_smooth(ssp.ssp_flux, ssp.ssp_lgmet, log_z_abs, 0.1)
r.block_until_ready()
t_smooth = (time.perf_counter() - t0) / N * 1e6

t0 = time.perf_counter()
for _ in range(100):
    r = dsps_reference_sed(sfr_constant, log_z_abs, 0.1)
r.rest_sed.block_until_ready()
t_dsps = (time.perf_counter() - t0) / 100 * 1e6

print("=" * 55)
print(f"{'Method':<40} {'us':>10}")
print("=" * 55)
print(f"{'interpolate_metallicity (2-point)':<40} {t_linear:>10.0f}")
print(f"{'interpolate_metallicity_smooth':<40} {t_smooth:>10.0f}")
print(f"{'DSPS full 2D (reference)':<40} {t_dsps:>10.0f}")
print("-" * 55)
print(f"{'smooth / linear overhead':<40} {t_smooth / t_linear:>9.1f}x")
print(f"{'smooth / DSPS speedup':<40} {t_dsps / t_smooth:>9.1f}x")
print("=" * 55)

fig, ax = plt.subplots(figsize=(8, 3.5))
methods = ["2-point linear\n(FSPS-style)", "triweight smooth\n(DSPS-style)",
           "full DSPS 2D\n(reference)"]
times = [t_linear, t_smooth, t_dsps]
colors = [COLORS["rt"], COLORS["nuts"], COLORS["truth"]]
bars = ax.barh(methods, times, color=colors, edgecolor="0.3", linewidth=0.5)
ax.set_xlabel(r"Time per call ($\mu$s)")
ax.set_title("Metallicity interpolation speed")
for bar, t in zip(bars, times):
    ax.text(bar.get_width() + max(times)*0.02, bar.get_y() + bar.get_height()/2,
            f"{t:.0f} us", va="center", fontsize=10)
ax.set_xlim(0, max(times) * 1.25)
fig.tight_layout()
plt.savefig("figures/20_speed_benchmark.pdf")
plt.show()

# %% [markdown]
# ## 11. RMS accuracy across metallicities

# %%
# Scan in absolute log10(Z), then convert to solar-relative for plotting
z_scan_abs = np.linspace(float(ssp.ssp_lgmet[2]), float(ssp.ssp_lgmet[-2]), 12)
z_scan_solar = z_scan_abs - LOG10_ZSUN

rms_A = []; rms_B = []; rms_C = []
mask = (wave > 1000) & (wave < 20000)

for lz_abs in z_scan_abs:
    ref = dsps_reference_sed(sfr_constant, lz_abs, 0.1)
    sed_ref = np.array(ref.rest_sed) * LSUN_ERG_PER_S
    safe = np.where(sed_ref[mask] > 0, sed_ref[mask], 1.0)

    age_w = _get_dsps_age_weights(sfr_constant)
    total_mass = float(jnp.trapezoid(sfr_constant, t_table * 1e9))
    weights = age_w * total_mass

    # A: midpoint + linear
    ssp_ages_gyr = 10.0 ** ssp.ssp_lg_age_gyr
    t_lookback = t_obs_gyr - t_table
    sfr_at_ssp = jnp.interp(ssp_ages_gyr, t_lookback[::-1], sfr_constant[::-1], left=0.0, right=0.0)
    w_m = compute_csp_weights(sfr_at_ssp, ssp_ages_gyr * 1e9)
    ssp_z = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, lz_abs)
    sed_A = np.array(compute_csp_sed(w_m, ssp_z, jnp.ones_like(ssp_z)))

    # B: DSPS age + linear
    sed_B = np.array(compute_csp_sed(weights, ssp_z, jnp.ones_like(ssp_z)))

    # C: DSPS age + triweight
    ssp_z_s = interpolate_metallicity_smooth(ssp.ssp_flux, ssp.ssp_lgmet, lz_abs, 0.1)
    sed_C = np.array(compute_csp_sed(weights, ssp_z_s, jnp.ones_like(ssp_z_s)))

    rms_A.append(np.sqrt(np.mean(((sed_A[mask] - sed_ref[mask]) / safe) ** 2)) * 100)
    rms_B.append(np.sqrt(np.mean(((sed_B[mask] - sed_ref[mask]) / safe) ** 2)) * 100)
    rms_C.append(np.sqrt(np.mean(((sed_C[mask] - sed_ref[mask]) / safe) ** 2)) * 100)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(z_scan_solar, rms_A, "o--", color=COLORS["rt"], lw=2, ms=6, label="A: midpoint + 2pt")
ax.plot(z_scan_solar, rms_B, "s-.", color=COLORS["geovi"], lw=2, ms=6, label="B: DSPS-age + 2pt")
ax.plot(z_scan_solar, rms_C, "D-", color=COLORS["nuts"], lw=2, ms=6, label="C: DSPS-age + triweight")
ax.axhline(1, color="0.7", ls=":", lw=0.8, label="1% threshold")

for z_grid in np.array(ssp.ssp_lgmet) - LOG10_ZSUN:
    ax.axvline(z_grid, color="0.92", ls=":", lw=0.5, zorder=0)

ax.set_xlabel(r"$\log_{10}(Z/Z_\odot)$")
ax.set_ylabel("RMS SED residual vs DSPS ref (%)")
ax.set_title("Accuracy vs metallicity (constant SFH)")
ax.legend()
ax.set_ylim(0, None)
fig.tight_layout()
plt.savefig("figures/20_rms_vs_metallicity.pdf")
plt.show()

# %% [markdown]
# ## 12. Usage
#
# ```python
# from tengri import ParamSpec, Uniform
#
# # FSPS-compatible (default)
# spec = ParamSpec(met_interp="linear", met_logzsol=Uniform(-2, 0.2))
#
# # DSPS-compatible (smooth gradients)
# spec = ParamSpec(met_interp="smooth", lgmet_scatter=0.1, met_logzsol=Uniform(-2, 0.2))
# ```
#
# The Zsol/Zabs conversion is automatic: `met_logzsol` is solar-relative
# (user-facing), and the model internally adds `log10(Zsun) = -1.8477`.

# %% [markdown]
# ## 13. Summary
#
# | | Method A | Method B | Method C |
# |---|---------|----------|----------|
# | **Age weights** | midpoint | DSPS cumulative | DSPS cumulative |
# | **Z interpolation** | 2-point linear | 2-point linear | triweight kernel |
# | **Matches** | traditional | FSPS/Prospector | **DSPS** |
# | **Gradient quality** | kinks at Z grid | kinks at Z grid | C$^2$ smooth |
# | **Speed (Z interp)** | ~100 us | ~100 us | ~800 us |
#
# **Key findings:**
#
# 1. The ~10% difference vs DSPS was **not** a Zsol/Zabs convention error.
#    Both codes use the same absolute $\log_{10}(Z)$ internally.
# 2. The difference has two sources: (a) age weights (fixed by DSPS cumulative)
#    and (b) Z interpolation method (now selectable).
# 3. The triweight kernel adds ~700 us but gives smooth C$^2$ gradients that
#    improve NUTS/geoVI convergence for metallicity.
# 4. `met_interp="linear"` is the default (FSPS/Prospector compatible).
#    `met_interp="smooth"` matches DSPS exactly.
