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
# # Tutorial 14: AGN Models and IGM Absorption
#
# This notebook demonstrates two physics modules in **diffsed** that are
# critical for broadband SED fitting beyond the local universe:
#
# 1. **AGN emission** &mdash; accretion disc + dust torus models at three
#    complexity levels (`simple`, `standard`, `kubota_done`).
# 2. **IGM absorption** &mdash; mean intergalactic medium transmission
#    (Inoue et al. 2014) that imprints the Lyman break and
#    Gunn&ndash;Peterson trough on high-$z$ galaxy spectra.
#
# We show how these modules work standalone, how they integrate into the
# `Model` forward model, and how IGM absorption creates the photometric
# dropout signatures used for high-redshift galaxy selection.

# %%
import os

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from diffsed.models.agn import AGN_MODELS, get_agn_model, unified_agn
from diffsed.models.agn.disc import powerlaw_disc, multicolor_disc
from diffsed.models.agn.torus import simple_torus, two_temperature_torus
from diffsed.models.igm import igm_transmission
from diffsed import (
    Model, ParamSpec, Uniform, Fixed,
    load_ssp_data, load_filter_set,
)

# ── Plot style ─────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.linewidth": 1.2,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# ── Figure output directory ────────────────────────────────────
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(fig, name, dpi=150):
    path = os.path.join(FIG_DIR, f"14_{name}.png")
    fig.savefig(path, bbox_inches="tight", dpi=dpi)
    print(f"Saved {path}")

# %% [markdown]
# ## 1. AGN Models: Disc + Torus SEDs
#
# diffsed provides three pre-registered AGN configurations, all combining
# an accretion disc (UV/optical) with a dust torus (MIR):
#
# | Model | Disc | Torus | Free params |
# |-------|------|-------|-------------|
# | `simple` | Power-law + UV cutoff | Single-$T$ modified BB | 3 |
# | `standard` | Multi-color Shakura&ndash;Sunyaev | Two-$T$ (hot + warm) | 5&ndash;6 |
# | `kubota_done` | Multi-color + BH spin | Two-$T$ clumpy | 8+ |
#
# All return $L_\nu$ in $L_\odot\,\mathrm{Hz}^{-1}$.

# %%
# Wavelength grid: 500 Angstrom to 30 micron (UV to MIR)
wave = jnp.geomspace(500.0, 3e5, 2000)  # Angstrom

# Common parameters
log_lbol = 44.0  # log10(L_bol / Lsun) ~ luminous Seyfert

# --- Compute all three AGN SEDs ---
sed_simple = get_agn_model("simple")(
    wave, agn_log_lbol=log_lbol, agn_frac=1.0,
    agn_alpha=-1.0, agn_T_torus=1000.0, agn_torus_frac=0.5,
)
sed_standard = get_agn_model("standard")(
    wave, agn_log_lbol=log_lbol, agn_frac=1.0,
    agn_log_mbh=8.0, agn_log_ledd=-1.0,
    agn_T_hot=1200.0, agn_T_warm=300.0, agn_frac_hot=0.3,
    agn_torus_frac=0.5,
)
sed_kubota = get_agn_model("kubota_done")(
    wave, agn_log_lbol=log_lbol, agn_frac=1.0,
    agn_log_mbh=8.0, agn_log_ledd=-1.0,
    agn_a_spin=0.5, agn_cos_inc=0.5,
    agn_T_hot=1200.0, agn_T_warm=300.0, agn_frac_hot=0.3,
    agn_tau_torus=5.0, agn_torus_frac=0.5,
)

# Convert to numpy for plotting
wave_um = np.array(wave) / 1e4  # Angstrom -> micron
nu = 2.99792458e18 / np.array(wave)  # Hz

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(wave_um, np.array(sed_simple) * nu, label="simple", lw=2.0)
ax.loglog(wave_um, np.array(sed_standard) * nu, label="standard", lw=2.0, ls="--")
ax.loglog(wave_um, np.array(sed_kubota) * nu, label="kubota_done", lw=2.0, ls=":")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$\nu L_\nu$ [$L_\odot$]")
ax.set_title(r"AGN SED models ($\log L_{\rm bol} = 44$)")
ax.set_xlim(0.05, 30)
ax.set_ylim(bottom=1e6)
ax.legend(loc="upper right")
ax.axvline(0.0912, color="0.7", ls=":", lw=0.6, zorder=0)
ax.text(0.0912, ax.get_ylim()[1] * 0.7, r"Ly limit", fontsize=7, color="0.5", ha="right")
ax.axvline(9.7, color="0.7", ls=":", lw=0.6, zorder=0)
ax.text(9.7, ax.get_ylim()[1] * 0.7, r"Si 9.7$\mu$m", fontsize=7, color="0.5", ha="left")
fig.tight_layout()
savefig(fig, "agn_three_models")
plt.show()

# %% [markdown]
# ### 1b. Disc vs Torus Components
#
# The `unified_agn` combiner splits the bolometric luminosity between the
# disc (fraction $1 - f_{\rm torus}$) and the torus ($f_{\rm torus}$).
# Here we show the two components separately for the `simple` model.

# %%
torus_frac = 0.5

# Disc only (power-law)
l_disc = powerlaw_disc(wave, agn_log_lbol=log_lbol, agn_frac=1.0 - torus_frac,
                       agn_alpha=-1.0)
# Torus only (single-T)
l_torus = simple_torus(wave, agn_log_lbol=log_lbol, agn_torus_frac=torus_frac,
                       agn_T_torus=1000.0)
l_total = l_disc + l_torus

fig, ax = plt.subplots(figsize=(8, 5))
ax.loglog(wave_um, np.array(l_disc) * nu, label="Accretion disc", color="#1f77b4", lw=1.8)
ax.loglog(wave_um, np.array(l_torus) * nu, label="Dust torus", color="#d62728", lw=1.8)
ax.loglog(wave_um, np.array(l_total) * nu, label="Total AGN", color="k", lw=2.2)
ax.fill_between(wave_um, 1e-10, np.array(l_disc) * nu, alpha=0.08, color="#1f77b4")
ax.fill_between(wave_um, 1e-10, np.array(l_torus) * nu, alpha=0.08, color="#d62728")
ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$\nu L_\nu$ [$L_\odot$]")
ax.set_title(r"Disc + torus decomposition ($f_{\rm torus} = 0.5$)")
ax.set_xlim(0.05, 30)
ax.set_ylim(bottom=1e6)
ax.legend(loc="upper right")
fig.tight_layout()
savefig(fig, "disc_torus_decomposition")
plt.show()

# %% [markdown]
# ### 1c. Effect of AGN Fraction on the Total SED
#
# The `agn_frac` parameter controls what fraction of the total bolometric
# luminosity comes from the AGN. At low fractions the galaxy SED dominates;
# at high fractions the UV and MIR are boosted by the disc and torus.

# %%
agn_fracs = [0.0, 0.01, 0.05, 0.1, 0.3, 0.5]
colors_frac = plt.cm.plasma(np.linspace(0.15, 0.85, len(agn_fracs)))

fig, ax = plt.subplots(figsize=(8, 5))
for frac, col in zip(agn_fracs, colors_frac):
    if frac == 0.0:
        ax.loglog(wave_um, np.ones_like(wave_um) * 1e-10, color=col,
                  label=f"$f_{{\\rm AGN}} = {frac:.2f}$", lw=0.5)
        continue
    sed_frac = get_agn_model("simple")(
        wave, agn_log_lbol=log_lbol, agn_frac=frac,
        agn_alpha=-1.0, agn_T_torus=1000.0, agn_torus_frac=0.5,
    )
    ax.loglog(wave_um, np.array(sed_frac) * nu, color=col,
              label=f"$f_{{\\rm AGN}} = {frac:.2f}$", lw=1.5)

ax.set_xlabel(r"Wavelength [$\mu$m]")
ax.set_ylabel(r"$\nu L_\nu$ [$L_\odot$]")
ax.set_title(r"Effect of $f_{\rm AGN}$ on AGN SED")
ax.set_xlim(0.05, 30)
ax.set_ylim(1e6, 1e13)
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
savefig(fig, "agn_frac_effect")
plt.show()

# %% [markdown]
# ## 2. AGN in the Forward Model
#
# When `agn_model="simple"` is set in the `ParamSpec`, the `Model` class
# automatically adds AGN emission to the stellar SED. The AGN bolometric
# luminosity is computed as `agn_frac * L_bol_stellar`, so the same
# parameter controls the relative AGN contribution at all wavelengths.
#
# Here we compare a galaxy SED with and without AGN using
# GALEX + SDSS + WISE filters.

# %%
# Load SSP data
ssp_data = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")

# Filter set: GALEX (UV) + SDSS (optical) + WISE (MIR)
filter_names = [
    "galex_fuv", "galex_nuv",
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "wise_w1", "wise_w2", "wise_w3", "wise_w4",
]
filters = load_filter_set(filter_names)
filter_wave_eff = np.array([
    1528, 2271,  # GALEX
    3551, 4686, 6166, 7480, 8932,  # SDSS
    33526, 46028, 115608, 220883,  # WISE
])

# --- Galaxy WITHOUT AGN ---
spec_no_agn = ParamSpec(
    sfh_dpl_log_peak_sfr=Fixed(1.0),
    sfh_dpl_tau_gyr=Fixed(5.0),
    sfh_dpl_alpha=Fixed(1.5),
    sfh_dpl_beta=Fixed(2.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.6),
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
    apply_igm=False,
)
model_no_agn = Model(spec_no_agn, ssp_data, filters=filters)

# --- Galaxy WITH AGN (simple model, 10% AGN fraction) ---
spec_agn = ParamSpec(
    sfh_dpl_log_peak_sfr=Fixed(1.0),
    sfh_dpl_tau_gyr=Fixed(5.0),
    sfh_dpl_alpha=Fixed(1.5),
    sfh_dpl_beta=Fixed(2.0),
    met_logzsol=Fixed(-0.2),
    dust_tau_bc=Fixed(0.3),
    dust_tau_diff=Fixed(0.6),
    agn_frac=Fixed(0.1),
    agn_alpha=Fixed(-1.0),
    agn_T_torus=Fixed(1000.0),
    redshift=Fixed(0.1),
    mean_sfh_type="dpl",
    agn_model="simple",
    apply_igm=False,
)
model_agn = Model(spec_agn, ssp_data, filters=filters)

# Sample at the fixed values
params_no_agn = spec_no_agn.sample(jax.random.PRNGKey(0))
params_agn = spec_agn.sample(jax.random.PRNGKey(0))

# Compute photometry
phot_no_agn = np.array(model_no_agn.predict_photometry(params_no_agn))
phot_agn = np.array(model_agn.predict_photometry(params_agn))

# Compute rest-frame SEDs for context
sed_no_agn = np.array(model_no_agn.predict_sed(params_no_agn))
sed_agn = np.array(model_agn.predict_sed(params_agn))
ssp_wave_um = np.array(ssp_data.ssp_wave) / 1e4  # Angstrom -> micron

# %%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Left panel: rest-frame SEDs ---
nu_ssp = 2.99792458e18 / np.array(ssp_data.ssp_wave)
ax1.loglog(ssp_wave_um, sed_no_agn * nu_ssp, color="#1f77b4", lw=1.2,
           label="Galaxy only", alpha=0.8)
ax1.loglog(ssp_wave_um, sed_agn * nu_ssp, color="#d62728", lw=1.2,
           label="Galaxy + AGN (10%)", alpha=0.8)
ax1.set_xlabel(r"Rest-frame wavelength [$\mu$m]")
ax1.set_ylabel(r"$\nu L_\nu$ [erg s$^{-1}$]")
ax1.set_title("Rest-frame SED")
ax1.set_xlim(0.05, 30)
ax1.legend(loc="upper right")

# --- Right panel: observed photometry ---
ax2.scatter(filter_wave_eff / 1e4, phot_no_agn * 1e29, s=60, marker="o",
            color="#1f77b4", zorder=5, label="Galaxy only")
ax2.scatter(filter_wave_eff / 1e4, phot_agn * 1e29, s=60, marker="D",
            color="#d62728", zorder=5, label="Galaxy + AGN (10%)")
# Connect with lines for clarity
ax2.plot(filter_wave_eff / 1e4, phot_no_agn * 1e29, "-", color="#1f77b4",
         alpha=0.4, lw=1.0)
ax2.plot(filter_wave_eff / 1e4, phot_agn * 1e29, "-", color="#d62728",
         alpha=0.4, lw=1.0)
ax2.set_xscale("log")
ax2.set_yscale("log")
ax2.set_xlabel(r"Observed wavelength [$\mu$m]")
ax2.set_ylabel(r"$f_\nu$ [$\mu$Jy]")
ax2.set_title("Observed photometry (GALEX + SDSS + WISE)")
ax2.legend(loc="upper left")

# Annotate UV and MIR boost
ax2.annotate("UV boost", xy=(0.16, phot_agn[0] * 1e29),
             xytext=(0.08, phot_agn[0] * 1e29 * 5),
             arrowprops=dict(arrowstyle="->", color="0.4"),
             fontsize=8, color="0.4")
ax2.annotate("MIR boost", xy=(12, phot_agn[-2] * 1e29),
             xytext=(5, phot_agn[-2] * 1e29 * 5),
             arrowprops=dict(arrowstyle="->", color="0.4"),
             fontsize=8, color="0.4")

fig.tight_layout()
savefig(fig, "agn_forward_model")
plt.show()

# %% [markdown]
# ## 3. IGM Absorption
#
# The intergalactic medium absorbs photons blueward of Lyman-$\alpha$
# (1216 \AA) through:
#
# - **Lyman-series line absorption** (Ly$\alpha$ forest + DLA systems)
# - **Lyman-continuum absorption** ($\lambda < 912$ \AA)
#
# diffsed implements the Inoue et al. (2014) mean IGM transmission
# $T_{\rm IGM}(\lambda_{\rm obs}, z_{\rm source})$, which is a
# function of observed wavelength and source redshift.

# %%
# Observed wavelength grid
wave_obs = jnp.linspace(800.0, 15000.0, 5000)

# Compute T_IGM at multiple redshifts
redshifts = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
colors_z = plt.cm.viridis(np.linspace(0.1, 0.95, len(redshifts)))

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                          gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})

ax_main = axes[0]
ax_zoom = axes[1]

for z_s, col in zip(redshifts, colors_z):
    t_igm = np.array(igm_transmission(wave_obs, z_s))
    ax_main.plot(np.array(wave_obs), t_igm, color=col, lw=1.2,
                 label=f"$z = {z_s:.0f}$")

# Mark key features
ax_main.axvline(1216.0, color="0.7", ls=":", lw=0.6)
ax_main.text(1216, 1.03, r"Ly$\alpha$", fontsize=8, ha="center", color="0.5")
ax_main.axvline(912.0, color="0.7", ls=":", lw=0.6)
ax_main.text(912, 1.03, r"Ly limit", fontsize=8, ha="center", color="0.5")

ax_main.set_ylabel(r"$T_{\rm IGM}(\lambda_{\rm obs})$")
ax_main.set_ylim(-0.02, 1.1)
ax_main.set_title("Mean IGM transmission (Inoue et al. 2014)")
ax_main.legend(loc="lower right", ncol=4, fontsize=8)
plt.setp(ax_main.get_xticklabels(), visible=False)

# Zoom into the Lyman break region at z=4
t_z4 = np.array(igm_transmission(wave_obs, 4.0))
ax_zoom.plot(np.array(wave_obs), t_z4, color=colors_z[4], lw=1.5)
ax_zoom.set_ylabel(r"$T_{\rm IGM}$ ($z=4$)")
ax_zoom.set_xlabel(r"Observed wavelength [$\AA$]")
ax_zoom.set_ylim(-0.02, 1.1)

# Mark Lyman series lines shifted to z=4
lyman_lines = {"Ly$\\alpha$": 1216.0, "Ly$\\beta$": 1026.0,
               "Ly$\\gamma$": 973.0, "Ly limit": 912.0}
for name, lam_rest in lyman_lines.items():
    lam_obs = lam_rest * (1.0 + 4.0)
    ax_zoom.axvline(lam_obs, color="0.6", ls=":", lw=0.5)
    ax_zoom.text(lam_obs, 1.03, name, fontsize=7, ha="center",
                 color="0.5", rotation=45)

# Shade Gunn-Peterson trough region
wave_np = np.array(wave_obs)
mask_gp = wave_np < 912.0 * (1.0 + 4.0)
ax_zoom.fill_between(wave_np[mask_gp], 0, 1.1, alpha=0.05, color="purple")
ax_zoom.text(912.0 * 5.0 / 2, 0.5, "Gunn-Peterson\ntrough", fontsize=8,
             ha="center", color="purple", alpha=0.6)

fig.tight_layout()
savefig(fig, "igm_transmission")
plt.show()

# %% [markdown]
# ## 4. High-$z$ Galaxy with IGM Absorption
#
# At $z = 6$, the Lyman break at 912 \AA\ is redshifted to
# $\sim 6400$ \AA\ (observed), while Ly$\alpha$ at 1216 \AA\ shifts to
# $\sim 8500$ \AA. This means essentially all flux blueward of the
# $i$-band is absorbed, creating a classic "$i$-band dropout."
#
# We demonstrate this using a full diffsed forward model at $z = 6$.

# %%
# ParamSpec for a z=6 star-forming galaxy
spec_z6 = ParamSpec(
    sfh_dpl_log_peak_sfr=Fixed(1.5),
    sfh_dpl_tau_gyr=Fixed(0.3),
    sfh_dpl_alpha=Fixed(1.0),
    sfh_dpl_beta=Fixed(5.0),
    met_logzsol=Fixed(-1.0),
    dust_tau_bc=Fixed(0.05),
    dust_tau_diff=Fixed(0.1),
    redshift=Fixed(6.0),
    mean_sfh_type="dpl",
    apply_igm=True,
)

spec_z6_noigm = ParamSpec(
    sfh_dpl_log_peak_sfr=Fixed(1.5),
    sfh_dpl_tau_gyr=Fixed(0.3),
    sfh_dpl_alpha=Fixed(1.0),
    sfh_dpl_beta=Fixed(5.0),
    met_logzsol=Fixed(-1.0),
    dust_tau_bc=Fixed(0.05),
    dust_tau_diff=Fixed(0.1),
    redshift=Fixed(6.0),
    mean_sfh_type="dpl",
    apply_igm=False,
)

# Use SDSS + JWST NIRCam filters to span the dropout
filter_names_z6 = [
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w",
    "jwst_f277w", "jwst_f356w", "jwst_f444w",
]
filters_z6 = load_filter_set(filter_names_z6)

model_z6 = Model(spec_z6, ssp_data, filters=filters_z6)
model_z6_noigm = Model(spec_z6_noigm, ssp_data, filters=filters_z6)

params_z6 = spec_z6.sample(jax.random.PRNGKey(42))
params_z6_noigm = spec_z6_noigm.sample(jax.random.PRNGKey(42))

phot_z6 = np.array(model_z6.predict_photometry(params_z6))
phot_z6_noigm = np.array(model_z6_noigm.predict_photometry(params_z6_noigm))

# Effective wavelengths (approximate, in Angstrom)
wave_eff_z6 = np.array([
    3551, 4686, 6166, 7480, 8932,  # SDSS
    9000, 11500, 15000, 20000,     # NIRCam SW
    27700, 35600, 44400,           # NIRCam LW
])

# %%
fig, ax = plt.subplots(figsize=(9, 5))

# Convert to microJy for visibility
phot_uJy = phot_z6 * 1e29
phot_uJy_noigm = phot_z6_noigm * 1e29

# Plot without IGM
ax.scatter(wave_eff_z6 / 1e4, phot_uJy_noigm, s=50, marker="s",
           facecolors="none", edgecolors="#1f77b4", linewidths=1.2,
           zorder=5, label="No IGM absorption")
ax.plot(wave_eff_z6 / 1e4, phot_uJy_noigm, "-", color="#1f77b4",
        alpha=0.3, lw=1.0)

# Plot with IGM
ax.scatter(wave_eff_z6 / 1e4, np.maximum(phot_uJy, 1e-10), s=70,
           marker="o", color="#d62728", zorder=6, label="With IGM (Inoue+2014)")
ax.plot(wave_eff_z6 / 1e4, np.maximum(phot_uJy, 1e-10), "-",
        color="#d62728", alpha=0.3, lw=1.0)

# Mark the Lyman break at z=6
lam_ly_obs = 912.0 * (1.0 + 6.0) / 1e4  # micron
lam_lya_obs = 1216.0 * (1.0 + 6.0) / 1e4
ax.axvline(lam_ly_obs, color="0.6", ls="--", lw=0.8)
ax.text(lam_ly_obs, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-6,
        f"Ly limit\n({lam_ly_obs:.2f} $\\mu$m)", fontsize=7, ha="right",
        color="0.5", va="bottom")
ax.axvline(lam_lya_obs, color="0.6", ls=":", lw=0.8)
ax.text(lam_lya_obs, 1e-4, f"Ly$\\alpha$\n({lam_lya_obs:.2f} $\\mu$m)",
        fontsize=7, ha="left", color="0.5", va="bottom")

# Shade the dropout region
ax.axvspan(0.3, lam_lya_obs, alpha=0.06, color="purple")
ax.text(0.55, 0.02, "$u$-band\ndropout\nregion", fontsize=9,
        ha="center", color="purple", alpha=0.7, transform=ax.transAxes)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"Observed wavelength [$\mu$m]")
ax.set_ylabel(r"$f_\nu$ [$\mu$Jy]")
ax.set_title(r"Galaxy at $z = 6$: IGM creates the Lyman-break dropout")
ax.set_xlim(0.3, 5.5)
ax.legend(loc="upper right")

# Label filter bands
band_labels = ["u", "g", "r", "i", "z",
               "F090W", "F115W", "F150W", "F200W", "F277W", "F356W", "F444W"]
for weff, label in zip(wave_eff_z6 / 1e4, band_labels):
    y_pos = max(phot_uJy_noigm[list(wave_eff_z6 / 1e4).index(weff)] * 1.5, 1e-5)
    ax.text(weff, y_pos, label, fontsize=6, ha="center", color="0.4")

fig.tight_layout()
savefig(fig, "high_z_galaxy_igm")
plt.show()

# %% [markdown]
# ## 5. IGM + Photometric Redshifts: The Dropout Signature
#
# The power of IGM absorption for photometric redshift estimation comes
# from the fact that the Lyman break moves through the filter set as
# redshift increases. At $z \sim 4$ it falls in the $u$-band
# ("$u$-dropout"), at $z \sim 5$ in the $g$-band, at $z \sim 6$ in
# the $r$-band, and at $z \sim 7$ in the $i$-band.
#
# Here we show observed photometry of the same intrinsic galaxy at
# $z = 4, 5, 6, 7$ through JWST NIRCam filters, demonstrating how the
# dropout moves redward.

# %%
# JWST NIRCam filter set (wide-band)
jwst_names = [
    "jwst_f090w", "jwst_f115w", "jwst_f150w", "jwst_f200w",
    "jwst_f277w", "jwst_f356w", "jwst_f444w",
]
jwst_filters = load_filter_set(jwst_names)
# Approximate effective wavelengths (micron)
jwst_wave_eff_um = np.array([0.90, 1.15, 1.50, 2.00, 2.77, 3.56, 4.44])

target_redshifts = [4.0, 5.0, 6.0, 7.0]
colors_panel = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

for ax, z_target, col in zip(axes.flat, target_redshifts, colors_panel):
    # Build model at this redshift
    spec_z = ParamSpec(
        sfh_dpl_log_peak_sfr=Fixed(1.5),
        sfh_dpl_tau_gyr=Fixed(0.3),
        sfh_dpl_alpha=Fixed(1.0),
        sfh_dpl_beta=Fixed(5.0),
        met_logzsol=Fixed(-1.0),
        dust_tau_bc=Fixed(0.05),
        dust_tau_diff=Fixed(0.1),
        redshift=Fixed(z_target),
        mean_sfh_type="dpl",
        apply_igm=True,
    )
    spec_z_noigm = ParamSpec(
        sfh_dpl_log_peak_sfr=Fixed(1.5),
        sfh_dpl_tau_gyr=Fixed(0.3),
        sfh_dpl_alpha=Fixed(1.0),
        sfh_dpl_beta=Fixed(5.0),
        met_logzsol=Fixed(-1.0),
        dust_tau_bc=Fixed(0.05),
        dust_tau_diff=Fixed(0.1),
        redshift=Fixed(z_target),
        mean_sfh_type="dpl",
        apply_igm=False,
    )

    model_z = Model(spec_z, ssp_data, filters=jwst_filters)
    model_z_noigm = Model(spec_z_noigm, ssp_data, filters=jwst_filters)

    p_z = spec_z.sample(jax.random.PRNGKey(int(z_target * 10)))
    p_z_noigm = spec_z_noigm.sample(jax.random.PRNGKey(int(z_target * 10)))

    phot_igm = np.array(model_z.predict_photometry(p_z)) * 1e29  # uJy
    phot_noi = np.array(model_z_noigm.predict_photometry(p_z_noigm)) * 1e29

    # No-IGM as open squares
    ax.scatter(jwst_wave_eff_um, phot_noi, s=40, marker="s",
               facecolors="none", edgecolors="0.5", linewidths=1.0,
               zorder=4, label="No IGM")
    ax.plot(jwst_wave_eff_um, phot_noi, "-", color="0.5", alpha=0.3, lw=0.8)

    # With IGM as filled circles
    ax.scatter(jwst_wave_eff_um, np.maximum(phot_igm, 1e-10), s=60,
               marker="o", color=col, zorder=5, label="With IGM")
    ax.plot(jwst_wave_eff_um, np.maximum(phot_igm, 1e-10), "-",
            color=col, alpha=0.4, lw=1.0)

    # Shade dropout region
    lam_lya_z = 1216.0 * (1.0 + z_target) / 1e4  # Ly-alpha in micron
    ax.axvspan(0.7, lam_lya_z, alpha=0.08, color=col)
    ax.axvline(lam_lya_z, color=col, ls=":", lw=0.8, alpha=0.5)

    ax.set_title(f"$z = {z_target:.0f}$", fontsize=13, fontweight="bold", color=col)
    ax.set_yscale("log")
    ax.set_xlim(0.7, 5.0)
    ax.set_ylim(1e-6, 1e2)
    ax.legend(loc="upper right", fontsize=7)

    # Label filters
    for weff, fname in zip(jwst_wave_eff_um, ["F090W", "F115W", "F150W",
                           "F200W", "F277W", "F356W", "F444W"]):
        ax.text(weff, 50, fname, fontsize=6, ha="center", color="0.4", rotation=45)

axes[1, 0].set_xlabel(r"Observed wavelength [$\mu$m]")
axes[1, 1].set_xlabel(r"Observed wavelength [$\mu$m]")
axes[0, 0].set_ylabel(r"$f_\nu$ [$\mu$Jy]")
axes[1, 0].set_ylabel(r"$f_\nu$ [$\mu$Jy]")

fig.suptitle("Lyman-break dropout through JWST NIRCam filters", fontsize=14, y=1.01)
fig.tight_layout()
savefig(fig, "dropout_signature_jwst")
plt.show()

# %% [markdown]
# ## Summary
#
# | Feature | Module | Key function |
# |---------|--------|-------------|
# | AGN disc emission | `diffsed.models.agn.disc` | `powerlaw_disc`, `multicolor_disc` |
# | AGN torus emission | `diffsed.models.agn.torus` | `simple_torus`, `two_temperature_torus` |
# | Unified AGN SED | `diffsed.models.agn.unified` | `unified_agn`, `get_agn_model` |
# | IGM transmission | `diffsed.models.igm` | `igm_transmission` |
# | Forward model integration | `diffsed.Model` | `agn_model="simple"` in `ParamSpec` |
#
# **Key takeaways:**
#
# 1. The `simple` AGN model (3 free params) is sufficient for most photometric
#    surveys; the `kubota_done` model adds BH physics for detailed AGN studies.
# 2. AGN emission boosts the UV (disc) and MIR (torus) relative to a pure
#    stellar SED &mdash; critical for breaking dust&ndash;AGN degeneracies.
# 3. IGM absorption at $z > 3$ is **not optional**: it creates the Lyman break
#    that dominates broadband photometric colors and enables photometric
#    redshift estimation.
# 4. The dropout wavelength tracks $(1+z) \times 1216$ \AA, moving through
#    successive filters as redshift increases.
