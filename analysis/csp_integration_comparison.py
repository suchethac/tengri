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
# # CSP Integration Comparison
#
# This notebook tests the impact of different CSP integration choices on
# photometry, colors, and spectra. It accompanies `docs/metallicity_interpolation.md`.
#
# We compare:
# - **Age weights**: midpoint rule vs DSPS cumulative mass interpolation
# - **Z interpolation**: 2-point linear (FSPS-style) vs triweight kernel (DSPS-style)
# - **Combined**: all four combinations
#
# The key finding is that age weights matter much more than Z interpolation,
# and neither choice affects colors or spectra at observable precision.

# %%
import sys

sys.path.insert(0, "../src")

import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)

from dsps.cosmology import DEFAULT_COSMOLOGY, age_at_z
from dsps.sed import calc_rest_sed_sfh_table_lognormal_mdf
from dsps.sed.ssp_weights import calc_age_weights_from_sfh_table

from tengri.observation.filters import load_filter_set
from tengri.observation.photometry import compute_flux_density
from tengri.sps.dsps_wrapper import (
    LSUN_ERG_PER_S,
    compute_csp_sed,
    compute_csp_weights,
    compute_lgmet_weights,
    interpolate_metallicity,
    interpolate_metallicity_smooth,
    load_ssp_data,
)
from tengri.utils.cosmology import luminosity_distance

# Style
try:
    import scienceplots

    plt.style.use(["science", "no-latex"])
except ImportError:
    pass
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "figure.dpi": 150,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "legend.frameon": False,
    }
)

C_TRUTH = "#1a1a1a"
C_BLUE = "#1f77b4"
C_ORANGE = "#ff7f0e"
C_GREEN = "#2ca02c"
C_RED = "#d62728"

# %% [markdown]
# ## 1. Setup

# %%
SSP_PATH = "../data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
ssp = load_ssp_data(SSP_PATH)
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
fw, ft, _ = filters
wave = np.array(ssp.ssp_wave)
bands = ["u", "g", "r", "i", "z"]

LOG10_ZSUN = -1.8477

z_obs = 0.3
t_obs_gyr = np.asarray(age_at_z(z_obs, *DEFAULT_COSMOLOGY)).item()
t_table = jnp.linspace(0.01, t_obs_gyr, 200)
dl_cm = luminosity_distance(z_obs)

# Test SFHs
sfr_constant = jnp.ones(200)
sfr_declining = jnp.where(t_table > 1.0, 10.0 * jnp.exp(-(t_table - 1.0) / 3.0), 0.0)

print(f"SSP grid: {ssp.ssp_flux.shape}")
print(f"Z grid (solar-relative): {np.array(ssp.ssp_lgmet) - LOG10_ZSUN}")
print(
    f"Age range: {10 ** float(ssp.ssp_lg_age_gyr[0]):.4f} - {10 ** float(ssp.ssp_lg_age_gyr[-1]):.2f} Gyr"
)
print(f"z_obs = {z_obs}, t_obs = {t_obs_gyr:.2f} Gyr")

# %% [markdown]
# ## 2. Four CSP integration methods
#
# | Method | Age weights | Z interpolation |
# |---|---|---|
# | **AA**: midpoint + linear | midpoint rule | 2-point linear |
# | **AB**: midpoint + smooth | midpoint rule | triweight |
# | **BA**: DSPS-age + linear | DSPS cumulative | 2-point linear |
# | **BB**: DSPS-age + smooth | DSPS cumulative | triweight |
# | **Ref**: full DSPS | DSPS cumulative | triweight + 2D weights |


# %%
def _dsps_age_weights(sfr_table):
    r = calc_age_weights_from_sfh_table(
        gal_t_table=t_table,
        gal_sfr_table=sfr_table,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        t_obs=t_obs_gyr,
    )
    return r.age_weights if hasattr(r, "age_weights") else jnp.asarray(r)


def _midpoint_weights(sfr_table):
    ssp_ages_gyr = 10.0**ssp.ssp_lg_age_gyr
    ssp_ages_yr = ssp_ages_gyr * 1e9
    t_lb = t_obs_gyr - t_table
    sfr_ssp = jnp.interp(ssp_ages_gyr, t_lb[::-1], sfr_table[::-1], left=0.0, right=0.0)
    return compute_csp_weights(sfr_ssp, ssp_ages_yr)


def compute_sed(sfr_table, log_z_solar, age_method, z_method):
    """Compute rest-frame SED with specified integration methods."""
    lz_abs = log_z_solar + LOG10_ZSUN

    # Age weights
    if age_method == "midpoint":
        weights = _midpoint_weights(sfr_table)
    else:
        age_w = _dsps_age_weights(sfr_table)
        total_mass = jnp.trapezoid(sfr_table, t_table * 1e9)
        weights = age_w * total_mass

    # Z interpolation
    if z_method == "linear":
        ssp_at_z = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, lz_abs)
    else:
        ssp_at_z = interpolate_metallicity_smooth(ssp.ssp_flux, ssp.ssp_lgmet, lz_abs, 0.1)

    dust = jnp.ones_like(ssp_at_z)
    return np.array(compute_csp_sed(weights, ssp_at_z, dust))


def compute_photometry(sed):
    """Compute SDSS photometry from rest-frame SED."""
    return np.array(
        [
            float(compute_flux_density(jnp.array(sed), ssp.ssp_wave, fwi, fti, z_obs, dl_cm))
            for fwi, fti in zip(fw, ft)
        ]
    )


def sed_to_mags(sed):
    phot = compute_photometry(sed)
    return -2.5 * np.log10(np.maximum(phot, 1e-40)) - 48.6


# %% [markdown]
# ## 3. Age weight comparison

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ssp_ages_gyr = 10.0 ** np.array(ssp.ssp_lg_age_gyr)

for ax, (sfh_name, sfr) in zip(
    axes, [("Constant", sfr_constant), (r"Declining $\tau$=3 Gyr", sfr_declining)]
):
    w_dsps = np.array(_dsps_age_weights(sfr))
    w_midpt = np.array(_midpoint_weights(sfr))

    t_dsps = max(w_dsps.sum(), 1e-30)
    t_midpt = w_midpt.sum()
    total_true = float(jnp.trapezoid(sfr, t_table * 1e9))

    ax.step(
        ssp_ages_gyr, w_dsps / t_dsps, where="mid", color=C_TRUTH, lw=2, label="DSPS cumulative"
    )
    ax.step(
        ssp_ages_gyr,
        w_midpt / t_midpt,
        where="mid",
        color=C_BLUE,
        lw=1.5,
        ls="--",
        label="Midpoint rule",
    )
    ax.set_xscale("log")
    ax.set_xlabel("SSP age (Gyr)")
    ax.set_ylabel("Normalised weight")
    ax.set_title(sfh_name)
    ax.legend(fontsize=9)
    ax.text(
        0.95,
        0.65,
        f"$M_{{true}}$ = {total_true:.2e}\n$M_{{midpt}}$ = {t_midpt:.2e}\n"
        f"Ratio = {t_midpt / total_true:.3f}",
        transform=ax.transAxes,
        ha="right",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
    )

fig.suptitle("Age weight methods", fontsize=14, y=1.02)
fig.tight_layout()
plt.savefig("figures/csp_age_weights.pdf")
plt.show()

# %% [markdown]
# ## 4. SED comparison (all 4 methods vs DSPS reference)

# %%
methods = {
    "AA: midpoint+linear": ("midpoint", "linear", "--", C_BLUE),
    "AB: midpoint+smooth": ("midpoint", "smooth", "-.", C_ORANGE),
    "BA: DSPS-age+linear": ("dsps", "linear", ":", C_GREEN),
    "BB: DSPS-age+smooth": ("dsps", "smooth", "-", C_RED),
}

fig, axes = plt.subplots(
    2, 2, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05}
)

for col, (sfh_name, sfr) in enumerate(
    [("Constant", sfr_constant), (r"Declining $\tau$=3 Gyr", sfr_declining)]
):
    ax_top = axes[0, col]
    ax_bot = axes[1, col]

    log_z_sol = -0.3
    # DSPS reference
    ref = calc_rest_sed_sfh_table_lognormal_mdf(
        gal_t_table=t_table,
        gal_sfr_table=sfr,
        gal_lgmet=log_z_sol + LOG10_ZSUN,
        gal_lgmet_scatter=0.1,
        ssp_lgmet=ssp.ssp_lgmet,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_flux=ssp.ssp_flux,
        t_obs=t_obs_gyr,
    )
    sed_ref = np.array(ref.rest_sed) * LSUN_ERG_PER_S
    safe = np.where(sed_ref > 0, sed_ref, 1.0)

    ax_top.plot(wave, sed_ref, color=C_TRUTH, lw=2.5, label="DSPS ref", zorder=10)

    mask = (wave > 1000) & (wave < 25000)
    for label, (am, zm, ls, color) in methods.items():
        sed = compute_sed(sfr, log_z_sol, am, zm)
        ax_top.plot(wave, sed, color=color, ls=ls, lw=1.5, label=label)
        res = (sed - sed_ref) / safe * 100
        ax_bot.plot(wave, res, color=color, ls=ls, lw=1.2)
        rms = np.sqrt(np.mean(res[mask] ** 2))
        print(f"  {sfh_name:20s} | {label} | RMS = {rms:.2f}%")

    ax_top.set_xscale("log")
    ax_top.set_yscale("log")
    ax_top.set_title(f"{sfh_name}, [Z/H] = {log_z_sol}")
    ax_top.set_xlim(900, 30000)
    if col == 0:
        ax_top.set_ylabel(r"$L_\nu$ (erg s$^{-1}$ Hz$^{-1}$)")
        ax_top.legend(fontsize=7, loc="lower left")
    plt.setp(ax_top.get_xticklabels(), visible=False)

    ax_bot.axhline(0, color="0.5", lw=0.5)
    ax_bot.axhspan(-2, 2, color="0.9", alpha=0.3)
    ax_bot.set_xscale("log")
    ax_bot.set_xlim(900, 30000)
    ax_bot.set_ylim(-15, 15)
    ax_bot.set_xlabel(r"$\lambda$ ($\AA$)")
    if col == 0:
        ax_bot.set_ylabel("Residual (%)")

fig.suptitle("CSP SED: 4 integration methods vs DSPS reference", fontsize=14, y=1.02)
fig.tight_layout()
plt.savefig("figures/csp_sed_comparison.pdf")
plt.show()

# %% [markdown]
# ## 5. Color differences
#
# Do the methods produce different colors? Colors are magnitude
# differences, so a uniform flux offset cancels.

# %%
z_solar_scan = np.linspace(-1.5, 0.2, 20)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
color_names = ["u-g", "g-r", "r-i", "i-z"]

for method_label, (am, zm, ls, color) in methods.items():
    delta_colors = {c: [] for c in color_names}
    for z_sol in z_solar_scan:
        # Method under test
        sed_test = compute_sed(sfr_declining, z_sol, am, zm)
        # Reference: DSPS
        ref = calc_rest_sed_sfh_table_lognormal_mdf(
            gal_t_table=t_table,
            gal_sfr_table=sfr_declining,
            gal_lgmet=z_sol + LOG10_ZSUN,
            gal_lgmet_scatter=0.1,
            ssp_lgmet=ssp.ssp_lgmet,
            ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
            ssp_flux=ssp.ssp_flux,
            t_obs=t_obs_gyr,
        )
        sed_ref = np.array(ref.rest_sed) * LSUN_ERG_PER_S

        mag_test = sed_to_mags(sed_test)
        mag_ref = sed_to_mags(sed_ref)
        dc = np.diff(mag_test) - np.diff(mag_ref)
        for i, cn in enumerate(color_names):
            delta_colors[cn].append(dc[i])

    for i, cn in enumerate(color_names):
        r, c_idx = divmod(i, 2)
        axes[r, c_idx].plot(
            z_solar_scan, delta_colors[cn], ls=ls, color=color, lw=1.5, label=method_label
        )

for i, cn in enumerate(color_names):
    r, c_idx = divmod(i, 2)
    ax = axes[r, c_idx]
    ax.axhline(0, color="0.5", lw=0.5)
    ax.axhspan(-0.02, 0.02, color="0.9", alpha=0.3, label="SDSS error" if i == 0 else None)
    ax.set_ylabel(f"$\\Delta$({cn}) [mag]")
    ax.set_ylim(-0.06, 0.06)
    if r == 1:
        ax.set_xlabel(r"$\log_{10}(Z/Z_\odot)$")
    if i == 0:
        ax.legend(fontsize=7, loc="upper left")
    ax.set_title(cn)

fig.suptitle(
    r"Color differences vs DSPS reference (declining SFH, $\tau$=3 Gyr)", fontsize=14, y=1.02
)
fig.tight_layout()
plt.savefig("figures/csp_color_differences.pdf")
plt.show()

# %% [markdown]
# ## 6. Spectral differences near absorption features
#
# Zoom into key spectral features to check if the interpolation
# method changes line depths or equivalent widths.

# %%
features = {
    r"D4000 break": (3700, 4300),
    r"H$\beta$ + Mg b": (4800, 5300),
    r"H$\alpha$": (6400, 6700),
    "Ca II triplet": (8400, 8750),
}

log_z_sol = -0.3

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

for ax, (feat_name, (wlo, whi)) in zip(axes.flat, features.items()):
    mask = (wave > wlo) & (wave < whi)
    ww = wave[mask]

    for label, (am, zm, ls, color) in methods.items():
        sed = compute_sed(sfr_declining, log_z_sol, am, zm)
        # Normalise to continuum for shape comparison
        sed_norm = sed[mask] / np.median(sed[mask])
        ax.plot(ww, sed_norm, color=color, ls=ls, lw=1.5, label=label)

    # DSPS reference
    ref = calc_rest_sed_sfh_table_lognormal_mdf(
        gal_t_table=t_table,
        gal_sfr_table=sfr_declining,
        gal_lgmet=log_z_sol + LOG10_ZSUN,
        gal_lgmet_scatter=0.1,
        ssp_lgmet=ssp.ssp_lgmet,
        ssp_lg_age_gyr=ssp.ssp_lg_age_gyr,
        ssp_flux=ssp.ssp_flux,
        t_obs=t_obs_gyr,
    )
    sed_ref = np.array(ref.rest_sed) * LSUN_ERG_PER_S
    sed_ref_norm = sed_ref[mask] / np.median(sed_ref[mask])
    ax.plot(ww, sed_ref_norm, color=C_TRUTH, lw=2.5, label="DSPS ref", zorder=10)

    ax.set_xlabel(r"$\lambda$ ($\AA$)")
    ax.set_ylabel("Normalised flux")
    ax.set_title(feat_name)
    if ax == axes.flat[0]:
        ax.legend(fontsize=6, loc="lower right")

fig.suptitle(f"Absorption features at [Z/H] = {log_z_sol} (declining SFH)", fontsize=14, y=1.02)
fig.tight_layout()
plt.savefig("figures/csp_spectral_features.pdf")
plt.show()

# %% [markdown]
# ## 7. Gradient smoothness
#
# The primary motivation for the triweight kernel: smoother derivatives
# with respect to metallicity.

# %%
# SED at V-band as function of logZ
age_w = _dsps_age_weights(sfr_constant)
total_mass = jnp.trapezoid(sfr_constant, t_table * 1e9)
weights = age_w * total_mass
iband = (ssp.ssp_wave > 5000) & (ssp.ssp_wave < 6000)


def vband_linear(log_z_abs):
    s = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, log_z_abs)
    return jnp.sum(compute_csp_sed(weights, s, jnp.ones_like(s)) * iband)


def vband_smooth(log_z_abs):
    s = interpolate_metallicity_smooth(ssp.ssp_flux, ssp.ssp_lgmet, log_z_abs, 0.1)
    return jnp.sum(compute_csp_sed(weights, s, jnp.ones_like(s)) * iband)


grad_lin = jax.grad(vband_linear)
grad_smo = jax.grad(vband_smooth)

z_lo, z_hi = float(ssp.ssp_lgmet[2]), float(ssp.ssp_lgmet[-2])
z_scan = jnp.linspace(z_lo, z_hi, 80)
z_scan_solar = z_scan - LOG10_ZSUN

print("Computing V-band gradients...")
lum_l = np.array([float(vband_linear(z)) for z in z_scan])
lum_s = np.array([float(vband_smooth(z)) for z in z_scan])
g_l = np.array([float(grad_lin(z)) for z in z_scan])
g_s = np.array([float(grad_smo(z)) for z in z_scan])

fig, axes = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1.5], "hspace": 0.05}
)

ax = axes[0]
ax.plot(z_scan_solar, lum_l, color=C_BLUE, lw=2, label="2-point linear")
ax.plot(z_scan_solar, lum_s, color=C_RED, lw=2, ls="--", label="Triweight (scatter=0.1)")
for zg in np.array(ssp.ssp_lgmet) - LOG10_ZSUN:
    ax.axvline(zg, color="0.88", ls=":", lw=0.5, zorder=0)
ax.set_ylabel(r"$L_\nu$ (V-band, erg/s/Hz)")
ax.legend()
ax.set_title("V-band luminosity and gradient vs metallicity")

ax = axes[1]
ax.plot(z_scan_solar, g_l, color=C_BLUE, lw=2, label=r"$\partial L/\partial \log Z$ (linear)")
ax.plot(
    z_scan_solar,
    g_s,
    color=C_RED,
    lw=2,
    ls="--",
    label=r"$\partial L/\partial \log Z$ (triweight)",
)
ax.axhline(0, color="0.5", lw=0.5)
for zg in np.array(ssp.ssp_lgmet) - LOG10_ZSUN:
    ax.axvline(zg, color="0.88", ls=":", lw=0.5, zorder=0)
ax.set_xlabel(r"$\log_{10}(Z/Z_\odot)$")
ax.set_ylabel(r"$\partial L / \partial \log Z$")
ax.legend(fontsize=9)

fig.tight_layout()
plt.savefig("figures/csp_gradient_smoothness.pdf")
plt.show()

print(
    f"\nGradient max jump — linear: {np.max(np.abs(np.diff(g_l))):.3e}, "
    f"smooth: {np.max(np.abs(np.diff(g_s))):.3e}"
)
print(
    f"Smoothness improvement: {np.max(np.abs(np.diff(g_l))) / max(np.max(np.abs(np.diff(g_s))), 1e-30):.1f}x"
)

# %% [markdown]
# ## 8. Speed benchmark

# %%
# Warmup
_ = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, -2.15).block_until_ready()
_ = interpolate_metallicity_smooth(ssp.ssp_flux, ssp.ssp_lgmet, -2.15, 0.1).block_until_ready()

N = 1000
t0 = time.perf_counter()
for _ in range(N):
    r = interpolate_metallicity(ssp.ssp_flux, ssp.ssp_lgmet, -2.15)
r.block_until_ready()
t_lin = (time.perf_counter() - t0) / N * 1e6

t0 = time.perf_counter()
for _ in range(N):
    r = interpolate_metallicity_smooth(ssp.ssp_flux, ssp.ssp_lgmet, -2.15, 0.1)
r.block_until_ready()
t_smo = (time.perf_counter() - t0) / N * 1e6

print(f"2-point linear:  {t_lin:.0f} us")
print(f"Triweight smooth: {t_smo:.0f} us")
print(f"Overhead: {t_smo / t_lin:.1f}x")
print(
    f"\nIn context of full forward model (~4500 us), this is {(t_smo - t_lin) / 4500 * 100:.1f}% overhead"
)

# %% [markdown]
# ## 9. Triweight weight kernel visualisation

# %%
from dsps.sed.ssp_weights import calc_lgmet_weights_from_lognormal_mdf

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
lgmet_solar = np.array(ssp.ssp_lgmet) - LOG10_ZSUN

# Left: different scatter values
for scatter, ls, alpha, label in [
    (0.001, ":", 0.5, r"$\sigma\approx 0$"),
    (0.05, "--", 0.7, r"$\sigma = 0.05$"),
    (0.1, "-", 1.0, r"$\sigma = 0.1$ (default)"),
    (0.3, "-.", 0.7, r"$\sigma = 0.3$"),
]:
    w = np.array(calc_lgmet_weights_from_lognormal_mdf(-0.3 + LOG10_ZSUN, scatter, ssp.ssp_lgmet))
    axes[0].step(lgmet_solar, w, where="mid", ls=ls, lw=2, alpha=alpha, label=label)

# 2-point linear weight
lgmet = np.array(ssp.ssp_lgmet)
lz_abs = -0.3 + LOG10_ZSUN
idx = np.searchsorted(lgmet, lz_abs) - 1
idx = max(0, min(idx, len(lgmet) - 2))
frac = (lz_abs - lgmet[idx]) / (lgmet[idx + 1] - lgmet[idx])
w_2pt = np.zeros(len(lgmet))
w_2pt[idx] = 1 - frac
w_2pt[idx + 1] = frac
axes[0].step(lgmet_solar, w_2pt, where="mid", color=C_RED, lw=2.5, label="2-point linear")

axes[0].set_xlabel(r"$\log_{10}(Z/Z_\odot)$")
axes[0].set_ylabel("Weight")
axes[0].set_title(r"Z weight kernels at [Z/H] = $-0.3$")
axes[0].legend(fontsize=8)

# Right: cross-validation with DSPS
scatter_vals = np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.3])
max_diffs = []
for sc in scatter_vals:
    w_ours = np.array(compute_lgmet_weights(lz_abs, ssp.ssp_lgmet, sc))
    w_dsps = np.array(calc_lgmet_weights_from_lognormal_mdf(lz_abs, sc, ssp.ssp_lgmet))
    max_diffs.append(np.max(np.abs(w_ours - w_dsps)))

axes[1].semilogy(scatter_vals, max_diffs, "o-", color=C_BLUE, lw=2, ms=8)
axes[1].axhline(1e-4, color="0.7", ls=":", lw=0.8, label=r"$10^{-4}$")
axes[1].set_xlabel(r"lgmet\_scatter (dex)")
axes[1].set_ylabel("|w_tengri - w_DSPS|_max")
axes[1].set_title("tengri vs DSPS cross-validation")
axes[1].legend()

fig.tight_layout()
plt.savefig("figures/csp_z_kernels.pdf")
plt.show()

# %% [markdown]
# ## 10. Summary
#
# | Observable | Age method impact | Z method impact |
# |---|---|---|
# | **Flux (broadband)** | 5--15% (mass normalisation) | <1.3% |
# | **Colors** | <0.05 mag | <0.009 mag |
# | **Spectra (optical)** | 5--15% (normalisation) | <0.3% (median 0.05%) |
# | **Absorption features** | Negligible (shape preserved) | Negligible (shape preserved) |
# | **Gradient smoothness** | N/A | **8.5x improvement** |
#
# **Conclusions:**
#
# 1. **Age weights matter more than Z interpolation** for SED accuracy.
#    The midpoint rule loses 5--15% of total mass on log-spaced grids.
#    DSPS cumulative interpolation is exact.
#
# 2. **Z interpolation does not affect colors or spectral shapes** at
#    any observable precision. The max color difference (0.009 mag) is
#    well below SDSS errors (~0.02 mag). Spectral differences are <0.3%.
#
# 3. **The triweight kernel's value is gradient quality**, not SED accuracy.
#    8.5x smoother gradients improve convergence for MAP, NUTS, geoVI, and EVI.
#
# 4. **Speed overhead is negligible** (<1% of total forward model cost).
#
# 5. **The smooth triweight is the default** in tengri because it provides
#    better inference at no measurable cost to accuracy or speed.
