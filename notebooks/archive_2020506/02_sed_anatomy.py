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
# # SED Anatomy: Wavelength → Physics
#
# **What you'll learn:**
# - How stellar + nebular + dust + AGN/radio/X-ray combine into a panchromatic SED
# - Component isolation: toggle modules on/off to understand each wavelength decade
# - Redshift handling and IGM absorption at high z
# - Safe optical-only forward models for spectroscopic fitting
#
# **Prerequisites:** [`00_quickstart.py`](00_quickstart.py) (optional context).
# **Next:** [`03_fitting_photometry.py`](03_fitting_photometry.py) for real-data workflows.
#
# ---
#
# Panchromatic SED from X-ray to radio. Trace how each physical ingredient
# (stellar continuum, nebular lines, dust attenuation, IR re-radiation, radio, X-ray) shapes different wavelengths.
# Then build the same SED by hand, piece by piece, using the tengri API.
#
# **Physics:** Stellar population synthesis (DSPS), nebular emission from ionizing photons, two-component dust (birth cloud + diffuse ISM),
# infrared templates, radio scalings, X-ray from XRBs. All in one JAX function.
#
# **Architecture difference:** In tengri, the full forward model (SSP → SFH → dust → nebular/AGN/radio/X-ray) is one differentiable JAX function,
# not a black-box pipeline. Toggle components, understand every equation, derivatives flow everywhere.


# %% [markdown]
# ## Setup: imports and model configuration

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

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*BakedInBackend.*",
    category=UserWarning,
)

from tengri import (
    Fixed,
    Observation,
    Parameters,
    SEDModel,
    Spectroscopy,
    Uniform,
    load_ssp_data,
)

# Locate ``notebooks/_plot_style.py`` and ``data/`` root (nbclient cwd is often wrong).
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

FIGDIR = os.path.join("notebooks", "figures", "sed_anatomy")
os.makedirs(FIGDIR, exist_ok=True)

from _plot_style import COLORS, setup_style

setup_style()

# %%
SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
ssp_data = load_ssp_data(SSP_PATH)


def _sfh_dust_kwargs():
    """Shared fixed galaxy: star-forming, moderate dust."""
    return dict(
        mean_sfh_type="dense_basis",
        sfh_db_log_total_mass=Fixed(10.2),
        sfh_db_log_sfr_inst=Fixed(0.6),
        sfh_db_tx_frac_0=Fixed(0.2),
        sfh_db_tx_frac_1=Fixed(0.35),
        sfh_db_tx_frac_2=Fixed(0.45),
        met_logzsol=Fixed(0.0),
        dust_tau_bc=Fixed(1.0),
        dust_tau_diff=Fixed(0.3),
        dust_slope=Fixed(-0.7),
        redshift=Fixed(0.05),
    )


# Inference-safe optical path: baked nebular in SSP, two-component attenuation only.
# (Do not add dust_emission / radio / xray when the data are optical pixels only.)
spec_baked_optical = Parameters(**_sfh_dust_kwargs(), nebular_ssp=True)

# Full physics for *visualization*: Draine & Li IR, radio + X-ray scalings
spec_full = Parameters(
    **_sfh_dust_kwargs(),
    nebular_ssp=True,
    dust_emission="draine_li2007",
    dust_T=Fixed(35.0),
    dust_qpah=Fixed(2.5),
    radio=True,
    xray=True,
    radio_q_ir=Fixed(2.64),
)
model_full = SEDModel(spec_full, ssp_data, observation=None)
params = spec_full.sample(jax.random.PRNGKey(42))

# %% [markdown]
# ## Section 1: Panchromatic SED decomposition (X-ray → radio)
#
# We use the forward pipeline's `compute_sed_components()` to obtain each physical
# emission channel in a **single** consistent call. All outputs are in
# erg s⁻¹ Hz⁻¹ (CGS); we convert to νL_ν [L☉] for the plot so different
# wavelength decades are directly comparable.
#
# The figure style follows standard multi-wavelength plots:
# electromagnetic band shading, filled component areas, twin frequency axis,
# and annotated spectral breaks.

# %%
import matplotlib.ticker as ticker
from tengri.forward.pipeline import compute_sed_components

# ── Wavelength / frequency grids ────────────────────────────────────────────
_C_AA_S = 2.99792458e18  # speed of light in Å s⁻¹
_LSUN_ERG = 3.828e33  # IAU 2015 nominal solar luminosity [erg/s]

# 0.12 Å (≈100 keV hard X-ray) → 3×10^11 Å (≈300 MHz radio)
_WAVE_AA = np.logspace(np.log10(0.12), np.log10(3e11), 3000)
_WAVE_UM = _WAVE_AA * 1e-4  # μm  (primary axis)
_NU_HZ = _C_AA_S / _WAVE_AA  # Hz

# ── Compute all SED components at z = 0 (rest-frame) ────────────────────────
_params_z0 = {**params, "redshift": jnp.array(0.0)}
_comp = compute_sed_components(
    model_full, _params_z0, need_intrinsic=True, rest_wavelength=jnp.array(_WAVE_AA)
)


def _nulnu(lnu):
    """L_nu [erg/s/Hz] → νL_ν [L☉], floored at zero."""
    return np.maximum(np.array(lnu) * _NU_HZ / _LSUN_ERG, 0.0)


_nl = {
    "intrinsic": _nulnu(_comp["sed_intrinsic"]),
    "stellar": _nulnu(_comp["sed_attenuated"]),
    "nebular": _nulnu(_comp["sed_nebular"]),
    "dust_ir": _nulnu(_comp["sed_dust_ir"]),
    "xray": _nulnu(_comp["sed_xray"]),
    "radio": _nulnu(_comp["sed_radio"]),
    "total": _nulnu(_comp["sed_total"]),
}

# ── Colour palette (perceptually ordered, colourblind-safe) ─────────────────
_C = {
    "intrinsic": "#aaaaaa",
    "stellar": "#4c78a8",
    "nebular": "#17becf",
    "dust_ir": "#f58518",
    "xray": "#9467bd",
    "radio": "#54a24b",
    "total": "#1a1a1a",
}

# ── Figure ───────────────────────────────────────────────────────────────────
_Y_MIN, _Y_MAX = 4.0, 13.0
_XLIM_UM = (1e-4, 1e6)

fig, ax = plt.subplots(1, 1, figsize=(13, 8))

# Band shading
_BANDS = [
    (1e-4, 0.010, "Hard\nX-ray", "#e8d0f5", 0.97),
    (0.010, 0.020, "Soft\nX-ray", "#d9edf7", 0.97),
    (0.020, 0.091, "EUV", "#f5e6d3", 0.88),
    (0.091, 0.20, "FUV", "#fff0a0", 0.97),
    (0.20, 0.40, "UV", "#faf3c0", 0.88),
    (0.40, 0.70, "Optical", "#e0f5e0", 0.97),
    (0.70, 2.5, "NIR", "#fde8d0", 0.88),
    (2.5, 30.0, "MIR", "#fdd9b0", 0.97),
    (30.0, 1e3, "FIR", "#fcc890", 0.88),
    (1e3, 1e6, "Radio", "#dce8f5", 0.97),
]
for _lo, _hi, _lbl, _col, _yfrac in _BANDS:
    ax.axvspan(_lo, _hi, color=_col, alpha=0.35, zorder=0)
    _lc = np.sqrt(_lo * _hi)
    _yp = 10 ** (_Y_MIN + _yfrac * (_Y_MAX - _Y_MIN))
    ax.text(
        _lc,
        _yp,
        _lbl,
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
        style="italic",
        zorder=5,
    )

# Plotting helpers
_thresh = 10 ** (_Y_MIN - 0.5)


def _splot(x, y, **kw):
    mask = y > _thresh
    if mask.any():
        ax.plot(x[mask], y[mask], **kw)


def _sfill(x, y, **kw):
    floor = np.full_like(y, 10**_Y_MIN)
    ym = np.where(y > _thresh, y, np.nan)
    ax.fill_between(x, floor, ym, **kw)


# ── Components ───────────────────────────────────────────────────────────────
_splot(
    _WAVE_UM,
    _nl["intrinsic"],
    color=_C["intrinsic"],
    lw=1.5,
    ls="--",
    alpha=0.7,
    zorder=2,
    label="Stars (intrinsic, no dust)",
)

_splot(
    _WAVE_UM,
    _nl["nebular"],
    color=_C["nebular"],
    lw=1.5,
    zorder=4,
    label="Nebular emission (baked-in SSP)",
)

_sfill(_WAVE_UM, _nl["stellar"], color=_C["stellar"], alpha=0.22, zorder=2)
_splot(
    _WAVE_UM,
    _nl["stellar"],
    color=_C["stellar"],
    lw=2.0,
    zorder=3,
    label="Stellar continuum (attenuated)",
)

_sfill(_WAVE_UM, _nl["dust_ir"], color=_C["dust_ir"], alpha=0.22, zorder=2)
_splot(
    _WAVE_UM,
    _nl["dust_ir"],
    color=_C["dust_ir"],
    lw=2.0,
    zorder=3,
    label=r"Dust IR (DL07, $q_{\rm PAH}=2.5\%$)",
)

_sfill(_WAVE_UM, _nl["xray"], color=_C["xray"], alpha=0.22, zorder=2)
_splot(_WAVE_UM, _nl["xray"], color=_C["xray"], lw=2.0, ls="-.", zorder=3, label="X-ray (XRBs)")

_sfill(_WAVE_UM, _nl["radio"], color=_C["radio"], alpha=0.22, zorder=2)
_splot(_WAVE_UM, _nl["radio"], color=_C["radio"], lw=2.0, zorder=3, label="Radio (SF synchrotron)")

_splot(_WAVE_UM, _nl["total"], color=_C["total"], lw=3.0, zorder=6, label="Total SED")

# ── Axes ─────────────────────────────────────────────────────────────────────
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*_XLIM_UM)
ax.set_ylim(10**_Y_MIN, 10**_Y_MAX)
ax.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]", fontsize=12)
ax.set_ylabel(r"$\nu L_\nu$ [$L_\odot$]", fontsize=12)
ax.set_title(
    r"SED anatomy: component decomposition (star-forming galaxy, $z=0$)",
    fontsize=13,
    fontweight="bold",
)

# Twin frequency axis on top
_ax_top = ax.twiny()
_ax_top.set_xscale("log")
_ax_top.set_xlim(*_XLIM_UM)
_nu_ticks = np.array([1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18])
_lam_ticks = (_C_AA_S / _nu_ticks) * 1e-4
_in_range = (_lam_ticks >= _XLIM_UM[0]) & (_lam_ticks <= _XLIM_UM[1])
_ax_top.set_xticks(_lam_ticks[_in_range])
_ax_top.set_xticklabels(
    [rf"$10^{{{int(np.log10(nu))}}}$" for nu in _nu_ticks[_in_range]],
    fontsize=10,
)
_ax_top.set_xlabel(r"Rest-frame frequency $\nu$ [Hz]", fontsize=11, labelpad=8)

ax.yaxis.set_minor_locator(ticker.LogLocator(subs=np.arange(2, 10)))
ax.yaxis.set_minor_formatter(ticker.NullFormatter())

ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3,
    fontsize=10,
    framealpha=0.9,
    edgecolor="#888888",
)

# Spectral break markers
for _wbreak, _wlabel, _xside in [
    (912e-4, "Lyman limit\n(912 Å)", "right"),
    (3646e-4, "Balmer break\n(3646 Å)", "left"),
]:
    ax.axvline(_wbreak, color="#555555", lw=1.0, ls=":", alpha=0.6, zorder=1)
    _xoff = _wbreak * (0.88 if _xside == "left" else 1.12)
    ax.text(
        _xoff,
        10**5.4,
        _wlabel,
        fontsize=10,
        color="#555555",
        va="bottom",
        ha=_xside,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
    )

fig.tight_layout()
plt.savefig(os.path.join(FIGDIR, "01_sed_decomposition.png"), dpi=150, bbox_inches="tight")
plt.show()

print("→ Next: build the SED in steps using separate forward models (same SSP, toggled modules).")

# %% [markdown]
# ## Section 2: Building the SED one component at a time
#
# We use **several** `SEDModel` instances with different modules enabled, each time
# calling **`predict_spectrum`** on the same wavelength grid. This avoids private
# component hooks and stays on the supported public API.

# %%
wave_vis = jnp.logspace(2.5, 4.85, 1500)

# Stellar + dust attenuation only (no nebular, no IR/radio/X-ray)
spec_stars = Parameters(**_sfh_dust_kwargs(), nebular_ssp=False)
model_stars = SEDModel(spec_stars, ssp_data, observation=None)
params_s = spec_stars.sample(jax.random.PRNGKey(42))
sed_stars_only = model_stars.predict_spectrum(params_s, wave_vis)

# + Nebular (baked into SSP)
spec_neb = Parameters(**_sfh_dust_kwargs(), nebular_ssp=True)
model_neb = SEDModel(spec_neb, ssp_data, observation=None)
params_n = spec_neb.sample(jax.random.PRNGKey(42))
sed_stars_nebular = model_neb.predict_spectrum(params_n, wave_vis)

# Intrinsic (no diffuse dust) vs attenuated stellar+nebular
params_no_dust = {
    **params_n,
    "dust_tau_bc": jnp.array(0.0),
    "dust_tau_diff": jnp.array(0.0),
}
sed_no_dust = model_neb.predict_spectrum(params_no_dust, wave_vis)
sed_with_atten = model_neb.predict_spectrum(params_n, wave_vis)

# + Dust IR + radio + X-ray (full)
sed_complete = model_full.predict_spectrum(params, wave_vis)

wv = np.array(wave_vis)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
ax.loglog(
    wv, np.array(sed_stars_only), color=COLORS["seq"][2], lw=1.2, label="Stars (+ dust att.)"
)
ax.set_xlim(wv.min(), wv.max())
y_data = np.array(sed_stars_only)
y_valid = y_data[np.isfinite(y_data) & (y_data > 0)]
ax.set_ylim(y_valid.min() * 0.3, y_valid.max() * 3)
ax.set_ylabel(r"$f_\nu$ [cgs]", fontsize=10)
ax.set_title("Step 1: Stellar continuum (no SSP nebular)", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

ax = axes[0, 1]
ax.loglog(
    wv, np.array(sed_stars_only), color=COLORS["seq"][2], lw=0.8, alpha=0.5, label="Previous"
)
ax.loglog(wv, np.array(sed_stars_nebular), color=COLORS["seq"][3], lw=1.2, label="+ Nebular (SSP)")
ax.set_xlim(wv.min(), wv.max())
y_data = np.concatenate([np.array(sed_stars_only), np.array(sed_stars_nebular)])
y_valid = y_data[np.isfinite(y_data) & (y_data > 0)]
ax.set_ylim(y_valid.min() * 0.3, y_valid.max() * 3)
ax.set_title("Step 2: + Nebular emission", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

ax = axes[1, 0]
ax.loglog(wv, np.array(sed_no_dust), color=COLORS["seq"][1], lw=0.8, alpha=0.5, label="Intrinsic")
ax.loglog(wv, np.array(sed_with_atten), color="darkred", lw=1.2, label="+ Dust attenuation")
ax.set_xlim(wv.min(), wv.max())
y_data = np.concatenate([np.array(sed_no_dust), np.array(sed_with_atten)])
y_valid = y_data[np.isfinite(y_data) & (y_data > 0)]
ax.set_ylim(y_valid.min() * 0.3, y_valid.max() * 3)
ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]", fontsize=10)
ax.set_ylabel(r"$f_\nu$ [cgs]", fontsize=10)
ax.set_title("Step 3: Charlot–Fall attenuation", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

ax = axes[1, 1]
ax.loglog(wv, np.array(sed_with_atten), color="darkred", lw=0.8, alpha=0.5, label="Attenuated")
ax.loglog(wv, np.array(sed_complete), color=COLORS["seq"][4], lw=1.2, label="+ IR + radio + X-ray")
ax.set_xlim(wv.min(), wv.max())
y_data = np.concatenate([np.array(sed_with_atten), np.array(sed_complete)])
y_valid = y_data[np.isfinite(y_data) & (y_data > 0)]
ax.set_ylim(y_valid.min() * 0.3, y_valid.max() * 3)
ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]", fontsize=10)
ax.set_title("Step 4: IR + radio + X-ray", fontsize=10, fontweight="bold")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

fig.suptitle(
    "Progressive SED assembly (public predict_spectrum only)",
    fontsize=12,
    fontweight="bold",
    y=1.0,
)
fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "01_sed_anatomy_components.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 3: The same galaxy at different redshifts
#
# We only change `redshift` in the parameter dict and evaluate on an
# observed-frame grid `λ_obs = λ_rest (1+z)`. Luminosity distance and IGM
# handling are applied inside `predict_spectrum`—do **not** apply an extra
# manual `(1+z)^{-2}` factor.

# %%
wave_rest = jnp.logspace(2.5, 5.0, 2000)
redshifts = [0.05, 0.1, 0.5, 1.0, 2.0, 4.0]
colors_z = plt.cm.plasma(np.linspace(0.1, 0.9, len(redshifts)))

fig, ax = plt.subplots(figsize=(12, 5))

for z, col in zip(redshifts, colors_z):
    wave_obs_z = wave_rest * (1.0 + z)
    params_z = {**params, "redshift": jnp.array(z)}
    sed_z = model_full.predict_spectrum(params_z, wave_obs_z)
    ax.loglog(np.array(wave_obs_z), np.array(sed_z), color=col, lw=1.2, label=f"z = {z}")

ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]", fontsize=11)
ax.set_ylabel(r"$f_\nu$ [erg/s/cm$^2$/Hz]", fontsize=11)
ax.set_title(
    "Same physical SED at different redshifts (observed frame)", fontsize=12, fontweight="bold"
)
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, alpha=0.3)
ax.set_xlim(1e3, 1e7)
ax.set_ylim(1e-18, 1e-12)

sdss_bands = {"u": 3600, "g": 4700, "r": 6200, "i": 7500, "z": 9100}
for _band_name, band_center in sdss_bands.items():
    ax.axvspan(band_center * 0.8, band_center * 1.2, alpha=0.06, color="gray")

fig.tight_layout()
# plt.savefig(os.path.join(FIGDIR, "01_sed_anatomy_redshift.png", dpi=300, bbox_inches="tight"), dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Section 3b: IGM transmission (high-$z$ caveat)
#
# The plots above use the full pipeline's redshift handling. **Separately**, the
# intergalactic medium suppresses observed flux blueward of the Lyman series
# (Inoue+2014). `igm_transmission(wave_obs, z)` expects **observed-frame**
# wavelengths in Å — the same convention as photometry through bandpasses.
# See `examples/multiwavelength/` and `examples/igm/` for a longer tour.

# %%
from tengri.igm import igm_transmission

wave_obs_igm = jnp.linspace(500.0, 25000.0, 2000)
fig, ax = plt.subplots(figsize=(9, 3.5))
for z, c in zip([1.0, 3.0, 6.0], plt.cm.plasma([0.2, 0.55, 0.85])):
    t = igm_transmission(wave_obs_igm, z)
    ax.plot(np.array(wave_obs_igm), np.array(t), color=c, lw=1.2, label=f"z = {z:.0f}")
ax.set_xscale("log")
ax.set_xlim(500.0, 25000.0)
ax.set_xlabel(r"Observed wavelength [$\mathrm{\AA}$]")
ax.set_ylabel(r"$T_{\rm IGM}$")
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False)
ax.set_title("IGM transmission (Inoue+2014) — illustration only")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Section 4: Optical spectrum — baked-in nebular only (fit-safe)
#
# For fitting on an optical spectrum only, build `SEDModel` with
# ``spec_baked_optical`` and `Observation(spectroscopy=Spectroscopy(...))`.
# Omitting IR/radio/X-ray matches the information in the data and avoids
# NaN or invalid likelihoods from extraneous SED components.

# %%
WAVE_FIT = jnp.linspace(3800.0, 9200.0, 200)
obs_optical = Observation(spectroscopy=Spectroscopy(wave_obs=WAVE_FIT))
sed_optical = SEDModel(spec_baked_optical, ssp_data, observation=obs_optical)

_key_opt = jax.random.PRNGKey(42)
params_optical = spec_baked_optical.sample(_key_opt)
mock_optical = sed_optical.mock_spectrum(
    params_optical, WAVE_FIT, snr=40.0, key=jax.random.fold_in(_key_opt, 1)
)

# Verify that the forward model is finite at the true parameters
pred_truth = sed_optical.predict_spectrum(params_optical, WAVE_FIT)
resid_sigma = (mock_optical.flux_obs - pred_truth) / mock_optical.noise
log_gauss_truth = float(-0.5 * jnp.sum(resid_sigma**2))
print(
    "Optical-only forward model: log (Gaussian) at truth =",
    log_gauss_truth,
    "(finite:",
    jnp.isfinite(log_gauss_truth).item(),
    ")",
)

# %% [markdown]
# ## What You Learned
#
# - Panchromatic anatomy via `compute_sed_components()`
# - Component isolation: stars, nebular, dust attenuation, IR/radio/X-ray
# - Redshift + IGM handling is automatic in `predict_spectrum()`
# - Safe optical-only models for inference
#
# **Next:** [`03_fitting_photometry.py`](03_fitting_photometry.py) for real-data workflows.
