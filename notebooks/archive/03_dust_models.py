# ---
# jupyter:
#   jupytext:
#     formats: notebook_code//py:percent,ipynb
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
# # Dust Attenuation and Emission
#
# tengri implements a generalized two-component dust model (Charlot & Fall
# 2000) with pluggable attenuation curves. This notebook visualizes all
# seven available curves, explores the two-component model, and shows the
# panchromatic SED from UV through IR.

# %%
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)

from tengri import (
    Fixed,
    SEDModel,
    Observation,
    ParamSpec,
    Photometry,
    Uniform,
    load_ssp_data,
    two_component_dust,
)
from tengri.dust.attenuation import DUST_LAWS, get_dust_law

import sys, os  # noqa: E401

try:
    _nb_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_nb_dir, "..", ".."))
except NameError:
    _nb_dir = os.getcwd()
    sys.path.insert(0, os.path.join(_nb_dir, ".."))
# Change to project root so data/ paths work
# chdir to project root for data/ access
if os.path.exists("data"):
    pass  # already in project root
elif os.path.exists(os.path.join("..", "data")):
    os.chdir("..")
elif os.path.exists(os.path.join("..", "..", "data")):
    os.chdir(os.path.join("..", ".."))
elif os.path.exists(os.path.join("..", "..", "..", "data")):
    os.chdir(os.path.join("..", "..", ".."))

from _plot_style import COLORS, setup_style

setup_style()

FIGDIR = os.path.join(_nb_dir, "..", "figures", "reference")
os.makedirs(FIGDIR, exist_ok=True)

# %% [markdown]
# ## 1. All Seven Attenuation Curves
#
# Each curve $k(\lambda)$ describes the wavelength dependence of dust
# attenuation, normalized at 5500 A. The two-component model then applies
# the curve with separate optical depths for birth clouds and diffuse ISM.

# %%
wavelength = jnp.linspace(1000.0, 30000.0, 2000)

# Define curves and their extra kwargs
CURVES = [
    ("power_law", {}, "Power law (CF00)"),
    ("calzetti", {}, "Calzetti+2000"),
    ("kriek_conroy", {"dust_bump_strength": 1.0, "dust_delta": 0.0}, "Kriek & Conroy 2013"),
    ("smc", {}, "SMC (Gordon+2003)"),
    ("cardelli", {"dust_Rv": 3.1}, "Cardelli+1989 (MW)"),
    ("salim", {}, "Salim+2018"),
    ("li08", {}, "Li+2008"),
]

curve_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]

# %%
# --- FIGURE 1: All attenuation curves ---
fig, ax = plt.subplots(figsize=(9, 5))
for (name, kwargs, label), color in zip(CURVES, curve_colors):
    dust_fn = get_dust_law(name)
    k = dust_fn(wavelength, **kwargs)
    ax.plot(np.array(wavelength) / 1e4, np.array(k), label=label, color=color, lw=1.5)

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$k(\lambda)$ (normalized at 5500 $\AA$)")
ax.set_title("Dust Attenuation Curves in tengri")
ax.axvline(0.55, ls=":", color="grey", lw=0.5, alpha=0.5)
ax.annotate(
    "V-band", xy=(0.55, 0.05), xycoords=("data", "axes fraction"), fontsize=7, color="grey"
)
ax.axvline(0.2175, ls=":", color="grey", lw=0.5, alpha=0.5)
ax.annotate(
    "2175 A bump", xy=(0.22, 0.85), xycoords=("data", "axes fraction"), fontsize=7, color="grey"
)
ax.set_xlim(0.1, 3.0)
ax.set_ylim(0, None)
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_attenuation_curves.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 2. Two-Component SEDModel Exploration
#
# The Charlot & Fall model has two components:
# - **Birth cloud** ($\tau_{\rm bc}$): extra attenuation on young stars
#   (age < $t_{\rm birth} \approx 10$ Myr)
# - **Diffuse ISM** ($\tau_{\rm diff}$): attenuation on all stars
#
# We show how varying each component changes the transmission.

# %%
ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
ssp_ages_yr = 10.0 ** (ssp_data.ssp_lg_age_gyr + 9.0)

# --- FIGURE 2: Two-component dust transmission for different tau values ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: Vary tau_bc with fixed tau_diff
ax = axes[0]
tau_diff_fixed = 0.3
for tau_bc, color in zip([0.0, 0.3, 0.8, 1.5], curve_colors[:4]):
    transmission = two_component_dust(
        ssp_data.ssp_wave,
        ssp_ages_yr,
        tau_v1=tau_bc,
        tau_v2=tau_diff_fixed,
        law_bc="power_law",
        law_diff="power_law",
        n_slope=-0.7,
    )
    # Show for a young population (index ~20, ~10 Myr)
    young_idx = 20
    ax.plot(
        np.array(ssp_data.ssp_wave),
        np.array(transmission[young_idx]),
        label=f"$\\tau_{{bc}}$={tau_bc}",
        color=color,
        lw=1.2,
    )
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel("Transmission")
ax.set_title(f"Birth cloud (young stars, $\\tau_{{diff}}$={tau_diff_fixed})")
ax.set_xlim(1000, 20000)
ax.legend(fontsize=8, frameon=False)

# Panel B: Vary tau_diff with fixed tau_bc
ax = axes[1]
tau_bc_fixed = 0.5
for tau_diff, color in zip([0.0, 0.3, 0.8, 1.5], curve_colors[:4]):
    transmission = two_component_dust(
        ssp_data.ssp_wave,
        ssp_ages_yr,
        tau_v1=tau_bc_fixed,
        tau_v2=tau_diff,
        law_bc="power_law",
        law_diff="power_law",
        n_slope=-0.7,
    )
    # Show for an old population (index ~80, ~1 Gyr)
    old_idx = 80
    ax.plot(
        np.array(ssp_data.ssp_wave),
        np.array(transmission[old_idx]),
        label=f"$\\tau_{{diff}}$={tau_diff}",
        color=color,
        lw=1.2,
    )
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel("Transmission")
ax.set_title(f"Diffuse ISM (old stars, $\\tau_{{bc}}$={tau_bc_fixed})")
ax.set_xlim(1000, 20000)
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_two_component.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 3. Effect of Attenuation Curve Choice on SED
#
# We generate an intrinsic SED and show how different dust laws modify
# the spectrum.

# %%
# Create a model and generate intrinsic + attenuated SEDs
spec = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
    sfh_tsnorm_width_gyr=Fixed(2.0),
    sfh_tsnorm_skew=Fixed(0.0),
    sfh_tsnorm_trunc=Fixed(5.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.0),
    dust_tau_diff=Fixed(0.0),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.0),
)
model_nodust = SEDModel(spec, ssp_data)
params_nodust = {
    "sfh_tsnorm_log_peak_sfr": 1.0,
    "sfh_tsnorm_peak_lbt_gyr": 3.0,
    "sfh_tsnorm_width_gyr": 2.0,
    "sfh_tsnorm_skew": 0.0,
    "sfh_tsnorm_trunc": 5.0,
    "met_logzsol": -0.2,
    "dust_tau_bc": 0.0,
    "dust_tau_diff": 0.0,
    "dust_slope": -0.7,
    "redshift": 0.0,
}
sed_intrinsic = model_nodust.predict_sed(params_nodust)
wave_rest = ssp_data.ssp_wave

# %%
# --- FIGURE 3: Panchromatic SED with different dust laws ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    np.array(wave_rest),
    np.array(sed_intrinsic),
    "k-",
    lw=1.5,
    alpha=0.4,
    label="Intrinsic (no dust)",
)

tau_bc, tau_diff = 0.5, 0.5
for (name, kwargs, label), color in zip(CURVES[:5], curve_colors[:5]):
    transmission = two_component_dust(
        wave_rest,
        ssp_ages_yr,
        tau_v1=tau_bc,
        tau_v2=tau_diff,
        law_bc=name,
        law_diff=name,
        **kwargs,
    )
    # Apply dust to SSP-weighted SED (approximate: use average transmission)
    mean_trans = jnp.mean(transmission, axis=0)
    sed_dusty = sed_intrinsic * mean_trans
    ax.plot(np.array(wave_rest), np.array(sed_dusty), color=color, lw=1.0, label=label)

ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel(r"$L_\nu$ [arbitrary]")
ax.set_title(
    f"SED with Different Dust Laws ($\\tau_{{bc}}$={tau_bc}, $\\tau_{{diff}}$={tau_diff})"
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(900, 50000)
ax.legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_panchromatic_sed.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. UV Bump at 2175 A
#
# The 2175 A feature is a defining characteristic of the Milky Way
# extinction curve. Different dust laws handle it differently.

# %%
# --- FIGURE 4: UV bump zoom ---
fig, ax = plt.subplots(figsize=(7, 4))
wave_uv = jnp.linspace(1500.0, 3500.0, 500)

for (name, kwargs, label), color in zip(CURVES, curve_colors):
    dust_fn = get_dust_law(name)
    k = dust_fn(wave_uv, **kwargs)
    ax.plot(np.array(wave_uv), np.array(k), color=color, lw=1.5, label=label)

ax.axvline(2175, ls=":", color="grey", lw=0.8)
ax.annotate(
    "2175 A", xy=(2175, 0.02), xycoords=("data", "axes fraction"), fontsize=8, color="grey"
)
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel(r"$k(\lambda)$")
ax.set_title("UV Bump Region Detail")
ax.legend(fontsize=7, frameon=False, ncol=2)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_uv_bump.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Dust Parameter Degeneracies
#
# The age-dust degeneracy is one of the most important systematics in SED
# fitting. Higher dust and older stellar populations both make galaxies
# redder. We show the degeneracy direction in tau_diff vs age space.

# %%
# --- FIGURE 5: tau_diff vs age color (simple illustration) ---
fig, ax = plt.subplots(figsize=(7, 5))

# Compute r-band - i-band color for a grid of (tau_diff, age)
obs_ri = Observation(photometry=Photometry.from_names(["sdss_r", "sdss_i"]))
tau_range = np.linspace(0.0, 1.5, 15)
age_range = np.linspace(1.0, 10.0, 15)
color_grid = np.zeros((len(tau_range), len(age_range)))

for i_t, tau in enumerate(tau_range):
    for i_a, age in enumerate(age_range):
        spec_grid = ParamSpec(
            sfh_tsnorm_log_peak_sfr=Fixed(1.0),
            sfh_tsnorm_peak_lbt_gyr=Fixed(float(age)),
            sfh_tsnorm_width_gyr=Fixed(2.0),
            sfh_tsnorm_skew=Fixed(0.0),
            sfh_tsnorm_trunc=Fixed(5.0),
            met_logzsol=Fixed(-0.2),
            dust_tau_bc=Fixed(0.0),
            dust_tau_diff=Fixed(float(tau)),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.1),
        )
        model_grid = SEDModel(spec_grid, ssp_data, observation=obs_ri)
        params_grid = {
            "sfh_tsnorm_log_peak_sfr": 1.0,
            "sfh_tsnorm_peak_lbt_gyr": float(age),
            "sfh_tsnorm_width_gyr": 2.0,
            "sfh_tsnorm_skew": 0.0,
            "sfh_tsnorm_trunc": 5.0,
            "met_logzsol": -0.2,
            "dust_tau_bc": 0.0,
            "dust_tau_diff": float(tau),
            "dust_slope": -0.7,
            "redshift": 0.1,
        }
        phot = model_grid.predict_photometry(params_grid)
        if phot is not None and len(phot) == 2:
            color_grid[i_t, i_a] = float(phot[0] - phot[1])  # r - i color

im = ax.contourf(age_range, tau_range, color_grid, levels=20, cmap="RdYlBu_r")
plt.colorbar(im, ax=ax, label="r - i color (flux)")
ax.set_xlabel("Peak lookback time [Gyr]")
ax.set_ylabel(r"$\tau_{\rm diff}$")
ax.set_title("Age-Dust Degeneracy: Iso-color Contours")
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_age_dust_degeneracy.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. SMC/LMC Pei 1992 Curves
#
# Pei (1992) parameterized the SMC and LMC extinction curves using a sum of
# generalized Drude profiles. The SMC has a very steep UV rise and NO 2175 A
# bump, while the LMC has a weak bump. Comparing these with Calzetti (starburst)
# and Cardelli (MW) illustrates the diversity of dust environments.

# %%
# --- FIGURE 6: SMC/LMC/Calzetti/Cardelli comparison ---
wave_comp = jnp.linspace(900.0, 10000.0, 2000)

pei_curves = [
    ("smc", {}, "SMC (Pei 1992)", "#d62728"),
    ("lmc", {}, "LMC (Pei 1992)", "#9467bd"),
    ("calzetti", {}, "Calzetti+2000", "#ff7f0e"),
    ("cardelli", {"dust_Rv": 3.1}, "Cardelli+1989 (MW)", "#1f77b4"),
]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# Panel A: Full wavelength range
ax = axes[0]
for name, kwargs, label, color in pei_curves:
    dust_fn = get_dust_law(name)
    k = dust_fn(wave_comp, **kwargs)
    ax.plot(np.array(wave_comp), np.array(k), label=label, color=color, lw=1.5)

ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel(r"$k(\lambda)$ (normalized at 5500 $\AA$)")
ax.set_title("Pei 1992 Curves: UV Slope Differences")
ax.axvline(2175, ls=":", color="grey", lw=0.5, alpha=0.5)
ax.set_xlim(900, 10000)
ax.set_ylim(0, None)
ax.legend(fontsize=8, frameon=False)

# Panel B: UV zoom showing 2175 A bump presence/absence
ax = axes[1]
wave_uv_zoom = jnp.linspace(1500.0, 3500.0, 500)
for name, kwargs, label, color in pei_curves:
    dust_fn = get_dust_law(name)
    k = dust_fn(wave_uv_zoom, **kwargs)
    ax.plot(np.array(wave_uv_zoom), np.array(k), label=label, color=color, lw=1.5)

ax.axvline(2175, ls=":", color="grey", lw=0.8)
ax.annotate(
    "2175 $\\AA$ bump",
    xy=(2200, 0.85),
    xycoords=("data", "axes fraction"),
    fontsize=8,
    color="grey",
)
ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel(r"$k(\lambda)$")
ax.set_title("UV Bump: Present (MW, LMC) vs Absent (SMC, Calzetti)")
ax.legend(fontsize=8, frameon=False)

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_pei92_curves.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# The SMC curve is the steepest in the UV, making it the preferred choice for
# high-redshift galaxies where empirical evidence (e.g., Capak+2015,
# Reddy+2018) suggests steep, bump-free attenuation. The LMC sits between
# the bump-free SMC/Calzetti and the strong MW bump.

# %% [markdown]
# ## 7. Witt & Gordon 2000 Dust Geometries
#
# The dust-star geometry matters as much as the dust grain properties. Witt &
# Gordon (2000) showed that the same optical depth produces very different
# transmission curves depending on how dust and stars are mixed:
#
# - **Shell** (foreground screen): standard $T = \exp(-\tau_V k(\lambda))$
# - **Cloudy** (homogeneous slab): $T = (1 - e^{-\tau k}) / (\tau k)$ — greyer
# - **Dusty** (clumpy, Poisson): $T = \exp(-N(1 - e^{-\tau_{cl} k}))$ — greyest
#
# The clumpy geometry is the greyest because some sightlines miss all clumps.

# %%
from tengri.dust import wg00_shell, wg00_cloudy, wg00_dusty

wave_geom = jnp.linspace(1000.0, 20000.0, 1500)
tau_values = [1.0, 2.0, 4.0]
geom_colors = {1.0: "#1f77b4", 2.0: "#ff7f0e", 4.0: "#d62728"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

geometries = [
    ("Shell (foreground screen)", wg00_shell),
    ("Cloudy (homogeneous slab)", wg00_cloudy),
    ("Dusty (clumpy)", wg00_dusty),
]

for ax, (title, geom_fn) in zip(axes, geometries):
    for tau_v in tau_values:
        T = geom_fn(wave_geom, tau_v=tau_v, law="cardelli", dust_Rv=3.1)
        ax.plot(
            np.array(wave_geom) / 1e4,
            np.array(T),
            color=geom_colors[tau_v],
            lw=1.3,
            label=f"$\\tau_V$ = {tau_v:.0f}",
        )
    ax.set_xlabel(r"Wavelength [$\mu$m]")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(0.1, 2.0)
    ax.legend(fontsize=8, frameon=False)

axes[0].set_ylabel(r"Transmission $T(\lambda)$")
fig.suptitle("Witt & Gordon (2000) Dust Geometries (Cardelli MW law)", y=1.02, fontsize=11)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_wg00_geometries.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# At $\tau_V = 4$, the shell geometry transmits essentially zero UV light, while
# the clumpy geometry still transmits $\sim$20% — a factor-of-infinity
# difference. This demonstrates why assuming a foreground screen can
# dramatically over-estimate UV attenuation in clumpy ISM environments.

# %% [markdown]
# ## 8. Casey 2012 MBB + Mid-IR Power Law
#
# A pure modified blackbody (MBB) under-predicts emission at $\lambda < 40\,\mu$m.
# Casey (2012) added a mid-IR power law component joined by a sigmoid transition,
# capturing the warm dust continuum seen in real galaxy SEDs. We compare both
# models at three temperatures.

# %%
from tengri.dust.emission import modified_blackbody, casey2012

# Wavelength grid covering FIR (10 to 1000 micron)
wave_ir = jnp.linspace(1e4, 1e7, 5000)  # 1-1000 um in Angstrom

L_abs = 1e10  # Lsun — arbitrary normalization
temperatures = [25.0, 35.0, 50.0]
temp_colors = {25.0: "#1f77b4", 35.0: "#ff7f0e", 50.0: "#d62728"}

fig, ax = plt.subplots(figsize=(9, 5))

for T_dust in temperatures:
    # Pure modified blackbody
    sed_mbb = modified_blackbody(wave_ir, L_abs, dust_T=T_dust, dust_beta_ir=1.8)
    # Casey 2012 (MBB + mid-IR power law)
    sed_casey = casey2012(wave_ir, L_abs, dust_T=T_dust, dust_beta_ir=1.8, dust_alpha_mir=2.0)

    wave_um = np.array(wave_ir) / 1e4
    ax.plot(
        wave_um,
        np.array(sed_mbb),
        color=temp_colors[T_dust],
        ls="--",
        lw=1.0,
        alpha=0.6,
    )
    ax.plot(
        wave_um,
        np.array(sed_casey),
        color=temp_colors[T_dust],
        ls="-",
        lw=1.5,
        label=f"T = {T_dust:.0f} K",
    )

# Legend entries for line styles
ax.plot([], [], "k--", lw=1.0, alpha=0.6, label="Pure MBB")
ax.plot([], [], "k-", lw=1.5, label="Casey 2012")

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title("Casey (2012) vs Pure Modified Blackbody")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, 1000)
ax.set_ylim(bottom=np.max(np.array(sed_casey)) * 1e-6)
ax.legend(fontsize=8, frameon=False, ncol=2)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_casey2012_vs_mbb.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# The mid-IR excess (solid vs dashed lines) is most prominent at $\lambda \lesssim
# 40\,\mu$m. Hotter dust shifts the MBB peak to shorter wavelengths, but the
# power-law component always extends toward the mid-IR. This excess is
# physically attributed to stochastically heated small grains and warm dust
# continuum.

# %% [markdown]
# ## 9. Energy Balance Relaxation ($\eta_{\rm balance}$)
#
# By default, tengri enforces strict energy balance: $L_{\rm IR} = L_{\rm absorbed}$.
# The `dust_eta_balance` parameter relaxes this:
#
# - $\eta = 1.0$: strict energy balance (default)
# - $\eta < 1.0$: some absorbed UV escapes without re-emission (geometric mismatch)
# - $\eta > 1.0$: extra IR from obscured sources (e.g., embedded AGN, Kokorev+2021)
#
# We generate the full SED (UV through IR) at three $\eta$ values.

# %%
# Build a model with dust emission enabled
spec_eta = ParamSpec(
    sfh_tsnorm_log_peak_sfr=Fixed(1.0),
    sfh_tsnorm_peak_lbt_gyr=Fixed(3.0),
    sfh_tsnorm_width_gyr=Fixed(2.0),
    sfh_tsnorm_skew=Fixed(0.0),
    sfh_tsnorm_trunc=Fixed(5.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.5),
    dust_tau_diff=Fixed(0.5),
    dust_slope=Fixed(-0.7),
    dust_eta_balance=Fixed(1.0),
    redshift=Fixed(0.0),
    dust_emission="modified_blackbody",
)
model_eta = SEDModel(spec_eta, ssp_data)

eta_values = [0.5, 1.0, 2.0]
eta_colors = {0.5: "#1f77b4", 1.0: "#2ca02c", 2.0: "#d62728"}
eta_labels = {
    0.5: r"$\eta = 0.5$ (geometric escape)",
    1.0: r"$\eta = 1.0$ (strict balance)",
    2.0: r"$\eta = 2.0$ (extra obscured sources)",
}

fig, ax = plt.subplots(figsize=(10, 5))

for eta in eta_values:
    params_eta = {
        "sfh_tsnorm_log_peak_sfr": 1.0,
        "sfh_tsnorm_peak_lbt_gyr": 3.0,
        "sfh_tsnorm_width_gyr": 2.0,
        "sfh_tsnorm_skew": 0.0,
        "sfh_tsnorm_trunc": 5.0,
        "met_logzsol": -0.2,
        "dust_tau_bc": 0.5,
        "dust_tau_diff": 0.5,
        "dust_slope": -0.7,
        "dust_eta_balance": float(eta),
        "redshift": 0.0,
    }
    sed_eta = model_eta.predict_sed(params_eta)
    ax.plot(
        np.array(ssp_data.ssp_wave),
        np.array(sed_eta),
        color=eta_colors[eta],
        lw=1.3,
        label=eta_labels[eta],
    )

ax.set_xlabel(r"Wavelength [$\AA$]")
ax.set_ylabel(r"$L_\nu$ [L$_\odot$ / Hz]")
ax.set_title("Energy Balance Relaxation via $\\eta_{\\rm balance}$")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(900, 5e6)
ax.legend(fontsize=8, frameon=False)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_energy_balance_eta.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# The UV/optical part of the SED is identical at all three $\eta$ values
# (same attenuation), but the IR luminosity scales linearly with $\eta$.
# This is useful for galaxies where the UV-derived attenuation and the IR
# luminosity are inconsistent — a common finding in dusty AGN hosts and
# sub-mm galaxies.

# %% [markdown]
# ## 10. Narayanan+2018 Redshift-Dependent Dust Priors
#
# Narayanan et al. (2018) ran cosmological radiative transfer simulations
# through SIMBA galaxies and found systematic trends in the attenuation
# curve shape with redshift:
#
# - **dust_delta** (slope deviation from Calzetti): more negative at high z
#   (steeper curves at early times)
# - **dust_bump_strength** (2175 A bump): weaker at high z
#
# tengri provides `narayanan_prior(z)` which returns Gaussian distributions
# that encode these trends, ready for direct use in ParamSpec.

# %%
from tengri.dust.priors import narayanan_prior

redshifts = [0.0, 1.0, 3.0, 6.0]
z_colors = {0.0: "#1f77b4", 1.0: "#2ca02c", 3.0: "#ff7f0e", 6.0: "#d62728"}

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: dust_delta prior at each redshift
ax = axes[0]
delta_range = np.linspace(-1.5, 0.5, 300)
for z in redshifts:
    priors = narayanan_prior(z)
    delta_dist = priors["dust_delta"]
    # Gaussian pdf: exp(-0.5 * ((x - mu)/sigma)^2) / (sigma * sqrt(2pi))
    mu, sigma = delta_dist.mu, delta_dist.sigma
    pdf = np.exp(-0.5 * ((delta_range - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(delta_range, pdf, color=z_colors[z], lw=1.5, label=f"z = {z:.0f}")
    ax.axvline(mu, color=z_colors[z], ls=":", lw=0.7, alpha=0.5)

ax.set_xlabel(r"$\delta$ (slope deviation)")
ax.set_ylabel("Probability density")
ax.set_title(r"dust\_delta Prior: Steeper at High $z$")
ax.legend(fontsize=8, frameon=False)

# Panel B: dust_bump_strength prior at each redshift
ax = axes[1]
bump_range = np.linspace(-0.5, 2.5, 300)
for z in redshifts:
    priors = narayanan_prior(z)
    bump_dist = priors["dust_bump_strength"]
    mu, sigma = bump_dist.mu, bump_dist.sigma
    pdf = np.exp(-0.5 * ((bump_range - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    ax.plot(bump_range, pdf, color=z_colors[z], lw=1.5, label=f"z = {z:.0f}")
    ax.axvline(mu, color=z_colors[z], ls=":", lw=0.7, alpha=0.5)

ax.axvline(0.0, color="grey", ls="-", lw=0.5, alpha=0.3)
ax.annotate(
    "no bump",
    xy=(0.02, 0.92),
    xycoords=("data", "axes fraction"),
    fontsize=7,
    color="grey",
)
ax.set_xlabel("Bump strength")
ax.set_ylabel("Probability density")
ax.set_title(r"dust\_bump\_strength Prior: Weaker at High $z$")
ax.legend(fontsize=8, frameon=False)

fig.suptitle("Narayanan+2018 Redshift-Dependent Dust Priors", y=1.02, fontsize=11)
fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "03_narayanan_priors.png"), bbox_inches="tight")
plt.show()

# %% [markdown]
# At $z = 0$, the curve shape is close to Calzetti ($\delta \approx -0.2$) with a
# moderate UV bump. By $z = 6$, the curve is much steeper ($\delta \approx -0.8$)
# and the bump has essentially vanished — consistent with the empirical finding
# that high-z galaxies resemble SMC-like attenuation. Using these priors in
# tengri:
#
# ```python
# from tengri.dust.priors import narayanan_prior
# spec = ParamSpec(..., **narayanan_prior(z=2.0))
# ```

# %% [markdown]
# ## Summary
#
# | Dust law | UV bump | Slope freedom | Best for |
# |----------|---------|---------------|----------|
# | Power law | No | Fixed | Fast, simple |
# | Calzetti | No | Fixed | Starbursts |
# | Kriek & Conroy | Yes (tunable) | Yes ($\delta$) | Prospector default |
# | SMC | No | Fixed (steep) | High-z |
# | LMC | Yes (weak) | Fixed | LMC-like environments |
# | Cardelli | Yes (MW) | $R_V$ free | MW sightlines |
# | Salim | No | Modified Calzetti | DSPS default |
# | Li+2008 | Yes ($c_4$ bump term) | 4 coeffs ($c_1$--$c_4$, Li 2008 Eq. 1) | Flexible analytic; MW/SMC/Calzetti presets |
#
# The two-component model separates birth-cloud and diffuse-ISM attenuation,
# which is crucial for correctly interpreting UV-bright young stars versus
# older populations. Beyond the attenuation curve choice, the dust-star
# geometry (Section 7), emission model (Section 8), energy balance (Section 9),
# and redshift-dependent priors (Section 10) all play important roles in
# producing reliable panchromatic SEDs.
