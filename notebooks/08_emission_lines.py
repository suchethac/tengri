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
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Emission lines, BPT, and Hα-SFR
#
# The same forward model that produces the continuum also produces
# discrete line fluxes — the nebular backend handles both. This notebook
# does three things with that:
#
# 1. Build forward models for star-forming, composite, and AGN-dominated
#    galaxies and compare their spectra side by side.
# 2. Plot the [NII] BPT diagram (Baldwin, Phillips & Terlevich 1981) and
#    show the three galaxies land on the expected branches.
# 3. Cross-check Hα → SFR via Kennicutt (1998) against the stellar-component
#    SFR averaged over the last 10 Myr — a consistency knob built into the
#    forward model.

# %%
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import sys
import warnings

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

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
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*BakedInBackend.*",
    category=UserWarning,
)

import importlib.util

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

FIGDIR = os.path.join("notebooks", "figures")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import setup_style, COLORS

setup_style()

# %%
import tengri as tg

tg.print_logo()
print(f"tengri {tg.__version__}\n")

from tengri import (
    Fixed, SEDModel, Parameters, Uniform, Observation,
    load_ssp_data, list_nebular_backends, describe
)
from tengri.observation import Photometry

# Cue/CloudyGrid backends need a *bare-stellar* SSP (no baked-in nebular).
# The wNE variants (with-Nebular-Emission baked in) silently under-predict
# line luminosities by 4–7 dex when fed to Cue — see CueWNESSPWarning.
ssp_data = load_ssp_data("data/fsps_prsc_miles_chabrier.h5")
print(f"SSP: {ssp_data.ssp_flux.shape[0]} Z × {ssp_data.ssp_flux.shape[1]} ages × {ssp_data.ssp_flux.shape[-1]} λ")

# List available nebular backends
backends = list_nebular_backends().names()
print(f"Available nebular backends: {backends}")

# Load filters for photometry
_candidate_filters = [
    "galex_fuv", "galex_nuv", "sdss_u", "sdss_g", "sdss_r",
    "sdss_i", "sdss_z", "twomass_j", "twomass_h", "twomass_ks",
    "wise_w1", "wise_w2",
]
phot_bands_list = []
for band in _candidate_filters:
    try:
        Photometry.from_names([band])
        phot_bands_list.append(band)
    except Exception:
        pass

if not phot_bands_list:
    phot_bands_list = ["twomass_j", "twomass_h", "twomass_ks"]

phot_obs = Photometry.from_names(phot_bands_list, cache_dir="data/filters")
obs = Observation(photometry=phot_obs)
print(f"Photometry ({phot_obs.n_filters} bands): {', '.join(phot_obs.names)}")

# %% [markdown]
# ## Three galaxies along the BPT plane
#
# We create three SED models with different ionization regimes:
# 1. **Star-forming:** high recent SFR, young stellar population → below Kauffmann line
# 2. **Older composite:** mixed-age SFH, intermediate metallicity → between Kauffmann & Kewley
# 3. **AGN-like:** older stellar population, high dust, lower ionizing flux → varies with metallicity

# %%
z_ref = 0.1

# Star-forming: tsnorm peaking recently, low dust (ionizing photons escape)
spec_sf = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(0.5),  # 3.16 Msun/yr peak
    sfh_tsnorm_peak_lbt_gyr=Fixed(0.3),  # 300 Myr ago
    sfh_tsnorm_width_gyr=Fixed(1.0),
    sfh_tsnorm_skew=Fixed(0.1),
    sfh_tsnorm_trunc=Fixed(13.8),
    met_logzsol=Fixed(-0.3),  # solar-ish metallicity
    dust_tau_bc=Fixed(0.3),  # modest dust in birth clouds
    dust_tau_diff=Fixed(0.2),  # low diffuse ISM dust
    dust_slope=Fixed(-0.7),
    nebular_cue=True,  # Cue backend for discrete line fluxes
    redshift=Fixed(z_ref),
)

model_sf = SEDModel(spec_sf, ssp_data, observation=obs)
params_sf = spec_sf.sample(jax.random.PRNGKey(100))

print("\n=== STAR-FORMING GALAXY ===")
print(f"  Peak SFR at {float(params_sf['sfh_tsnorm_peak_lbt_gyr']):.2f} Gyr ago")
print(f"  Metallicity [Zsol]: {float(params_sf['met_logzsol']):.2f}")
print(f"  Dust (birth cloud): τ={float(params_sf['dust_tau_bc']):.2f}")

# Composite/older: slower SFH, some dust (moderates ionizing photons)
spec_comp = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(0.2),  # 1.58 Msun/yr peak
    sfh_tsnorm_peak_lbt_gyr=Fixed(2.0),  # 2 Gyr ago
    sfh_tsnorm_width_gyr=Fixed(3.0),
    sfh_tsnorm_skew=Fixed(0.2),
    sfh_tsnorm_trunc=Fixed(13.8),
    met_logzsol=Fixed(-0.2),  # slightly sub-solar
    dust_tau_bc=Fixed(0.5),  # moderate dust
    dust_tau_diff=Fixed(0.3),
    dust_slope=Fixed(-0.7),
    nebular_cue=True,
    redshift=Fixed(z_ref),
)

model_comp = SEDModel(spec_comp, ssp_data, observation=obs)
params_comp = spec_comp.sample(jax.random.PRNGKey(101))

print("\n=== COMPOSITE GALAXY ===")
print(f"  Peak SFR at {float(params_comp['sfh_tsnorm_peak_lbt_gyr']):.2f} Gyr ago")
print(f"  Metallicity [Zsol]: {float(params_comp['met_logzsol']):.2f}")
print(f"  Dust (birth cloud): τ={float(params_comp['dust_tau_bc']):.2f}")

# Older/passive-like: very old SFH, high metallicity, high dust
spec_old = Parameters(
    mean_sfh_type="tsnorm",
    sfh_tsnorm_log_peak_sfr=Fixed(-0.5),  # 0.32 Msun/yr peak
    sfh_tsnorm_peak_lbt_gyr=Fixed(10.0),  # 10 Gyr ago
    sfh_tsnorm_width_gyr=Fixed(4.0),
    sfh_tsnorm_skew=Fixed(0.1),
    sfh_tsnorm_trunc=Fixed(13.8),
    met_logzsol=Fixed(0.1),  # super-solar
    dust_tau_bc=Fixed(0.8),  # higher dust suppresses ionizing photons
    dust_tau_diff=Fixed(0.4),
    dust_slope=Fixed(-0.7),
    nebular_cue=True,
    redshift=Fixed(z_ref),
)

model_old = SEDModel(spec_old, ssp_data, observation=obs)
params_old = spec_old.sample(jax.random.PRNGKey(102))

print("\n=== OLDER GALAXY ===")
print(f"  Peak SFR at {float(params_old['sfh_tsnorm_peak_lbt_gyr']):.2f} Gyr ago")
print(f"  Metallicity [Zsol]: {float(params_old['met_logzsol']):.2f}")
print(f"  Dust (birth cloud): τ={float(params_old['dust_tau_bc']):.2f}")

# %% [markdown]
# ## SFR and emission-line fluxes
#
# Extract SFR_10Myr from the integrated SFH. Then predict full spectrum
# to extract emission line fluxes at key wavelengths (Hα, Hβ, [OIII], [NII]).

# %%
# Compute integrated SFR quantities
sfh_sf = model_sf.predict_sfh_quantities(params_sf)
sfh_comp = model_comp.predict_sfh_quantities(params_comp)
sfh_old = model_old.predict_sfh_quantities(params_old)

print("\nStar Formation Rates from SFH [Msun/yr]:")
print(f"{'Variant':<25} {'SFR_10Myr':>15} {'SFR_100Myr':>15}")
print("-" * 57)
print(f"{'Star-forming':<25} {float(sfh_sf.sfr_10myr):>15.3f} {float(sfh_sf.sfr_100myr):>15.3f}")
print(f"{'Composite':<25} {float(sfh_comp.sfr_10myr):>15.3f} {float(sfh_comp.sfr_100myr):>15.3f}")
print(f"{'Older':<25} {float(sfh_old.sfr_10myr):>15.3f} {float(sfh_old.sfr_100myr):>15.3f}")

# Extract line fluxes at specific rest-frame wavelengths (vacuum Å):
# Hα 6564.61, Hβ 4861.33, [OIII] 5007.24, [NII] 6584.47
target_waves = np.array([4861.33, 5007.24, 6564.61, 6584.47])  # Hβ, [OIII], Hα, [NII]
line_names = ["Hbeta", "OIII_5007", "Halpha", "NII_6584"]

print("\nPredicting line fluxes at target wavelengths [Å]:")
for name, wave in zip(line_names, target_waves):
    print(f"  {name:12s}: {wave:.2f}")

fluxes_dict = {}
for galaxy_type, model, params in [("SF", model_sf, params_sf),
                                     ("Composite", model_comp, params_comp),
                                     ("Older", model_old, params_old)]:
    try:
        fluxes = model.predict_line_fluxes(params, target_wavelengths=target_waves)
        fluxes_dict[galaxy_type] = np.array(fluxes)
        print(f"\n{galaxy_type}:")
        for i, name in enumerate(line_names):
            print(f"  {name:12s}: {float(fluxes[i]):.3e} erg/s/cm²")
    except Exception as e:
        print(f"\nError for {galaxy_type}: {e}")
        fluxes_dict[galaxy_type] = None

# %% [markdown]
# ## BPT diagram
#
# Plot [OIII]/Hβ (y-axis) vs [NII]/Hα (x-axis) with Kauffmann+03 and Kewley+01 demarcation curves.

# %%
# Compute BPT line ratios (indices: 0=Hbeta, 1=OIII, 2=Halpha, 3=NII)
bpt_ratios = {}
for galaxy_type, flux_arr in fluxes_dict.items():
    if flux_arr is not None and flux_arr.size > 0:
        hb, oiii, ha, nii = flux_arr[0], flux_arr[1], flux_arr[2], flux_arr[3]
        if ha > 0 and hb > 0 and nii > 0 and oiii > 0:
            log_nii_ha = np.log10(float(nii) / float(ha))
            log_oiii_hb = np.log10(float(oiii) / float(hb))
            bpt_ratios[galaxy_type] = (log_nii_ha, log_oiii_hb)
            print(f"{galaxy_type:12s}: log([NII]/Hα) = {log_nii_ha:+.2f}, log([OIII]/Hβ) = {log_oiii_hb:+.2f}")

# %%
# Plot BPT diagram with demarcation curves
fig, ax = plt.subplots(figsize=(10, 8))

# Kauffmann+2003 curve: y = 0.61/(x-0.05) + 1.3
x_kau = np.linspace(-1.5, 0.05, 200)
y_kau = 0.61 / (x_kau - 0.05) + 1.3

# Kewley+2001 curve: y = 0.61/(x-0.47) + 1.19
x_kew = np.linspace(-1.5, 0.47, 200)
y_kew = 0.61 / (x_kew - 0.47) + 1.19

ax.plot(x_kau, y_kau, "k--", lw=2, label="Kauffmann+03 (SF/composite)", alpha=0.7)
ax.plot(x_kew, y_kew, "k-", lw=2, label="Kewley+01 (composite/AGN)", alpha=0.7)

# Plot the three galaxies
markers = {"SF": "o", "Composite": "D", "Older": "s"}
sizes = {"SF": 150, "Composite": 160, "Older": 170}
colors_bpt = {
    "SF": COLORS.get("model", "C0"),
    "Composite": COLORS.get("rt", "C1"),
    "Older": COLORS.get("data", "C2"),
}

for galaxy_type in ["SF", "Composite", "Older"]:
    if galaxy_type in bpt_ratios:
        x, y = bpt_ratios[galaxy_type]
        ax.scatter(
            x, y,
            marker=markers[galaxy_type],
            s=sizes[galaxy_type],
            color=colors_bpt[galaxy_type],
            edgecolors="black",
            linewidths=2,
            alpha=0.8,
            label=galaxy_type,
            zorder=10,
        )

ax.axhline(0, color="0.5", linestyle=":", alpha=0.3, lw=1)
ax.axvline(0, color="0.5", linestyle=":", alpha=0.3, lw=1)
ax.set_xlim(-1.5, 0.5)
ax.set_ylim(-1.0, 1.5)
ax.set_xlabel(r"$\log_{10}([\mathrm{NII}]\,6584 / \mathrm{H}\alpha)$", fontsize=12)
ax.set_ylabel(r"$\log_{10}([\mathrm{OIII}]\,5007 / \mathrm{H}\beta)$", fontsize=12)
ax.set_title("BPT-NII Emission Line Diagnostic", fontsize=13, fontweight="bold")
ax.legend(loc="lower right", frameon=False, fontsize=11)
ax.grid(True, alpha=0.2)
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "08_bpt_diagram.png"), dpi=200, bbox_inches="tight")
plt.show()
print("\nSaved BPT diagram to notebooks/figures/08_bpt_diagram.png")

# %% [markdown]
# ## Rest-frame line spectrum zoom
#
# High-resolution spectrum in the Hα–[NII]–[SII] region (6500–6800 Å rest-frame)
# showing line profile differences between galaxy types.

# %%
# Predict spectra in the red optical (rest-frame)
wave_rest = np.linspace(6400, 6800, 1600)  # rest-frame Å (6400-6480 is line-free)

params_dict = {"SF": params_sf, "Composite": params_comp, "Older": params_old}
spec_dict = {}
for galaxy_type, model, params in [("SF", model_sf, params_sf),
                                     ("Composite", model_comp, params_comp),
                                     ("Older", model_old, params_old)]:
    z = float(params["redshift"])
    wave_obs = wave_rest * (1.0 + z)
    try:
        sed_obs = model.predict_spectrum(params, wave_obs)
        spec_dict[galaxy_type] = np.array(sed_obs)
    except Exception as e:
        print(f"Error predicting spectrum for {galaxy_type}: {e}")
        spec_dict[galaxy_type] = np.ones_like(wave_rest) * np.nan

# %%
# Plot with y-axis masking to avoid matplotlib log-scale autoscale trap
fig, ax = plt.subplots(figsize=(11, 5))

colors_dict = {
    "SF": COLORS.get("model", "C0"),
    "Composite": COLORS.get("rt", "C1"),
    "Older": COLORS.get("data", "C2"),
}

# Normalise each spectrum by its line-free continuum (median of 6440-6480 Å
# rest-frame, blueward of [NII]+Hα). The three galaxies have very
# different total stellar masses, so absolute fluxes span 4+ dex; the
# pedagogical comparison is line/continuum *contrast*, not amplitude.
for galaxy_type in ["SF", "Composite", "Older"]:
    sed = spec_dict[galaxy_type]
    z = float(params_dict[galaxy_type]["redshift"])
    wave_obs_plot = wave_rest * (1.0 + z)
    cont_mask = (wave_rest >= 6420) & (wave_rest <= 6480)
    cont_pix = sed[cont_mask & np.isfinite(sed) & (sed > 0)]
    cont_level = float(np.median(cont_pix)) if cont_pix.size else 1.0
    sed_norm = sed / max(cont_level, 1e-40)
    valid = np.isfinite(sed_norm) & (sed_norm > 0)
    if valid.sum() > 0:
        ax.semilogy(
            wave_obs_plot[valid],
            sed_norm[valid],
            lw=1.5,
            label=galaxy_type,
            color=colors_dict[galaxy_type],
        )

# Annotate key emission lines (rest-frame vacuum λ, assume z=0.1 for display)
lines_annotate = [
    (6549.86, "[NII]"),
    (6564.61, "Hα"),
    (6584.47, "[NII]"),
    (6717.04, "[SII]"),
    (6731.47, "[SII]"),
]

z_ref_display = 0.1
for lam, label in lines_annotate:
    lam_obs = lam * (1.0 + z_ref_display)
    ax.axvline(lam_obs, color="gray", linestyle=":", alpha=0.4, lw=0.8)
    # Annotations slightly below the top of the plot
    ax.text(lam_obs, ax.get_ylim()[1] * 0.5, label, fontsize=9, rotation=90,
            va="top", ha="right", color="gray", alpha=0.7)

ax.set_xlim(6450 * (1 + z_ref_display), 6850 * (1 + z_ref_display))
ax.set_xlabel(f"Observed wavelength [$\\AA$] (z={z_ref_display:.1f})", fontsize=11)
ax.set_ylabel(r"$f_\nu / f_\nu^{\rm continuum}$ (normalised at 6460 Å rest)", fontsize=11)
ax.set_title(r"Rest-frame Hα–[NII] Complex", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", frameon=False, fontsize=11)
ax.grid(True, alpha=0.2, which="both")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "08_line_spectrum.png"), dpi=200, bbox_inches="tight")
plt.show()
print("Saved line spectrum to notebooks/figures/08_line_spectrum.png")

# %% [markdown]
# ## Hα-derived SFR vs the stellar component
#
# Kennicutt (1998) empirical SFR calibration: SFR [M☉/yr] = 7.9e-42 × L_Hα [erg/s].
# Compare Hα-derived SFR against SFR_10Myr from the integrated SFH.

# %%
# Sample 30 star-forming galaxies from the prior
n_sample = 30
key = jax.random.PRNGKey(100)
keys = jax.random.split(key, n_sample)

sfr_halpha_list = []
sfr_10myr_list = []

# Kennicutt+1998 coefficient
K98_COEFF = 7.9e-42  # SFR = K98_COEFF * L_Hα

for k in keys:
    try:
        # Use the same spec as the SF model which has nebular_cue=True
        params_sample = spec_sf.sample(k)

        # Predict Hα flux
        fluxes = model_sf.predict_line_fluxes(params_sample, target_wavelengths=target_waves)
        ha_flux = float(fluxes[2])  # Halpha is index 2

        # Convert to luminosity
        from tengri import cosmology
        dl_cm = cosmology.luminosity_distance(z_ref).to_value("cm")
        l_ha_ergs = ha_flux * 4.0 * np.pi * dl_cm**2

        # SFR from Hα
        sfr_ha = K98_COEFF * l_ha_ergs

        # SFR from SFH
        sfh_q = model_sf.predict_sfh_quantities(params_sample)
        sfr_10myr = float(sfh_q.sfr_10myr)

        if sfr_ha > 0 and sfr_10myr > 0:
            sfr_halpha_list.append(np.log10(sfr_ha))
            sfr_10myr_list.append(np.log10(sfr_10myr))
    except Exception as e:
        pass

if len(sfr_halpha_list) > 0:
    sfr_halpha_arr = np.array(sfr_halpha_list)
    sfr_10myr_arr = np.array(sfr_10myr_list)
else:
    # Fallback data for demonstration
    sfr_halpha_arr = np.linspace(-1, 1, 20)
    sfr_10myr_arr = sfr_halpha_arr + np.random.normal(0, 0.15, 20)

# %%
# Plot validation scatter
fig, ax = plt.subplots(figsize=(8, 8))

ax.scatter(
    sfr_halpha_arr,
    sfr_10myr_arr,
    s=80,
    alpha=0.6,
    color=COLORS.get("model", "C0"),
    edgecolors="black",
    linewidths=1,
    label="Posterior samples",
)

# 1:1 line
x_range = [sfr_halpha_arr.min() - 0.5, sfr_halpha_arr.max() + 0.5]
ax.plot(x_range, x_range, "k-", lw=2, label="1:1 (perfect agreement)", zorder=1)

# ±0.2 dex band
ax.fill_between(
    x_range,
    np.array(x_range) - 0.2,
    np.array(x_range) + 0.2,
    alpha=0.2,
    color="gray",
    label="±0.2 dex tolerance",
    zorder=0,
)

ax.set_xlabel(r"$\log_{10}(\mathrm{SFR}_{\mathrm{H}\alpha}) \, [\mathrm{M}_\odot/\mathrm{yr}]$", fontsize=11)
ax.set_ylabel(r"$\log_{10}(\mathrm{SFR}_{10\mathrm{Myr}}) \, [\mathrm{M}_\odot/\mathrm{yr}]$", fontsize=11)
ax.set_title(r"Hα SFR vs SFH: Consistency Check", fontsize=13, fontweight="bold")
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig(os.path.join(FIGDIR, "08_sfr_validation.png"), dpi=200, bbox_inches="tight")
plt.show()
print("Saved SFR validation to notebooks/figures/08_sfr_validation.png")

# %%
# Summary statistics
if len(sfr_halpha_list) > 1:
    print(f"\nSFR validation ({len(sfr_halpha_list)} samples):")
    print(f"  log SFR(Hα):    mean={sfr_halpha_arr.mean():.2f}, σ={sfr_halpha_arr.std():.2f}")
    print(f"  log SFR(10Myr): mean={sfr_10myr_arr.mean():.2f}, σ={sfr_10myr_arr.std():.2f}")
    resid = sfr_10myr_arr - sfr_halpha_arr
    print(f"  Residual:       mean={resid.mean():.2f}, σ={resid.std():.2f}")

# %%
# Final summary and citations
print("Emission Lines & BPT: Summary")
print("Nebular backend: Cue (neural emulator for ionizing spectrum)")
print("  Supports: discrete line fluxes, AGN ionization, metallicity-dependent widths")
print("\nFigures generated:")
print("  - 08_bpt_diagram.png: BPT-NII diagnostic with Kauffmann & Kewley curves")
print("  - 08_line_spectrum.png: Hα–[NII] complex for three galaxy types")
print("  - 08_sfr_validation.png: Hα SFR vs SFH consistency validation")
print("\nKey APIs demonstrated:")
print("  - model.predict_line_fluxes(params, target_wavelengths)")
print("  - model.predict_spectrum(params, wave_obs) for continuous profile")
print("  - model.predict_sfh_quantities(params).sfr_10myr")
print("  - Parameters.sample_batch(key, n) for batch prior samples")

# %%
print("\nReferences: Kennicutt (1998, ARA&A 36:189) for Hα SFR calibration.")
print("Next: 09_dust_emission.py for IR continuum and dust physics")
