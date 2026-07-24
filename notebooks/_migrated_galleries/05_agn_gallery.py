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
# # AGN Emission Gallery: Unified Model Tour
#
# _agn_gallery
#
# **What:** Build complete AGN SEDs combining accretion disc, dust torus,
# broad-line region (BLR), and narrow-line region (NLR) using `unified_agn`.
#
# **What you'll see:** End-to-end assembly of a Type 1 (face-on) and Type 2
# (edge-on) AGN SED, showing how geometric masking and torus orientation
# transform the observed spectrum. We'll tour three disc models (power-law,
# multi-color, Kubota & Done), two torus templates (simple MBB, SKIRTOR
# clumpy), and emission-line strengths across the optical and UV.
#
# **Why tengri is different:** Most SED codes ignore AGN lines or use
# angle-independent templates. tengri models inclination-dependent geometry
# (BLR hidden at high inclination), differentiable line profiles, and BH
# physics (mass, spin, Eddington ratio) jointly with the host galaxy SED.
#
# **Prerequisites:** `00_quickstart.py` (forward model flow).
# **Continue with:** `06_multiwavelength_gallery.py` (X-ray + radio).

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

from tengri import load_ssp_data
from tengri.components.agn import (
    blr_emission,
    multicolor_disc,
    nlr_emission,
    powerlaw_disc,
    resolve_agn_model,
    simple_torus,
    two_temperature_torus,
    unified_agn,
)

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
    from _plot_style import COLORS, setup_style
except ModuleNotFoundError:
    from tengri.analysis.plotting import setup_style

    COLORS = {
        "rt": "C0",
        "geovi": "C1",
        "nuts": "C2",
        "model": "C3",
        "seq": ["C0", "C1", "C2", "C3", "C4"],
    }

setup_style()

FIGDIR = os.path.join("notebooks", "figures", "agn_gallery")
os.makedirs(FIGDIR, exist_ok=True)


def _set_reasonable_log_ylim_from_axes(
    axes,
    pad_log=0.12,
    min_xy_points=4,
    floor_below_peak_dex=10.0,
    wide_log_span_threshold=10.0,
    percentile_lo_hi=(5.0, 95.0),
):
    """Tighten log *y* limits from line data inside each subplot's *x* range.

    - Skips ``axvline`` polylines (too few points or zero *x* span) so their
      artificial *y* range does not dominate limits.
    - Per line, drops values more than ``floor_below_peak_dex`` below that
      line's 99th percentile (removes line-model numerical floors / spikes).
    - If the pooled log-span still exceeds ``wide_log_span_threshold``, uses
      ``percentile_lo_hi`` on log10(y) instead of raw min/max.
    """
    ax_list = np.ravel(np.atleast_1d(axes))
    if ax_list.size == 0:
        return
    ys = []
    for ax in ax_list:
        x_lo, x_hi = ax.get_xlim()
        if not np.isfinite(x_lo) or not np.isfinite(x_hi) or x_lo >= x_hi:
            continue
        for line in ax.get_lines():
            x = np.asarray(line.get_xdata(), dtype=float)
            y = np.asarray(line.get_ydata(), dtype=float)
            if x.size < min_xy_points:
                continue
            x_span = float(np.ptp(x))
            x_scale = max(float(np.max(np.abs(x))), 1.0)
            if x_span <= 1e-9 * x_scale or x_span <= 1e-12:
                continue
            m = (x >= x_lo) & (x <= x_hi) & np.isfinite(y) & (y > 0)
            if not np.any(m):
                continue
            y_win = y[m]
            peak = float(np.percentile(y_win, 99.0))
            if not np.isfinite(peak) or peak <= 0:
                continue
            floor = peak * 10 ** (-floor_below_peak_dex)
            y_win = y_win[y_win >= floor]
            if y_win.size == 0:
                continue
            ys.append(y_win)
    if not ys:
        return
    y = np.concatenate(ys)
    logy = np.log10(y)
    if not np.all(np.isfinite(logy)):
        return
    raw_lo = float(np.min(logy))
    raw_hi = float(np.max(logy))
    if raw_hi - raw_lo > wide_log_span_threshold:
        p_lo, p_hi = percentile_lo_hi
        lo_log, hi_log = (float(t) for t in np.percentile(logy, [p_lo, p_hi]))
    else:
        lo_log, hi_log = raw_lo, raw_hi
    lo_log -= pad_log
    hi_log += pad_log
    if hi_log - lo_log < pad_log * 2:
        mid = 0.5 * (lo_log + hi_log)
        lo_log, hi_log = mid - pad_log * 2, mid + pad_log * 2
    y0, y1 = 10**lo_log, 10**hi_log
    if not np.isfinite(y0) or not np.isfinite(y1) or y0 <= 0 or y1 <= y0:
        return
    for ax in ax_list:
        ax.set_ylim(y0, y1)


# Physical constants
_LSUN_ERG = 3.828e33  # Solar luminosity [erg s^-1]

# Common wavelength grids
wavelength = jnp.logspace(np.log10(100), np.log10(1e6), 1000)
wave_um = np.asarray(wavelength) / 1e4
nu_arr = np.asarray(3e18 / wavelength)

# %% [markdown]
# ## 1. Overview: Full Unified AGN SED (Disc + Torus + BLR + NLR)
#
# A complete unified AGN SED spans UV through far-infrared. The `unified_agn`
# function combines four physical components:
#
# 1. **Accretion disc** (UV/optical): thermal emission from the hot inner flow
# 2. **Dust torus** (infrared): reprocesses absorbed disc emission as dust thermal continuum
# 3. **Broad Line Region** (BLR, UV/optical lines): dense gas near the black hole,
#    FWHM ~ 5000 km/s, **hidden at high inclination**
# 4. **Narrow Line Region** (NLR, optical lines): extended photoionized gas,
#    FWHM ~ 500 km/s, **visible at all inclinations**
#
# The torus covering factor `agn_torus_frac` splits the bolometric luminosity:
# the disc emits (1 - agn_torus_frac) and the torus re-emits agn_torus_frac.

# %%
fig, ax = plt.subplots(figsize=(9, 5.5))

# Compute unified AGN manually for labeling
_log_lbol = 44.0
_cos_inc = 0.95  # face-on (Type 1)
_l_bol_erg = 10.0**_log_lbol * _LSUN_ERG
_torus_frac = 0.5

# Disc only (gets 1 - torus_frac of luminosity)
l_disc = np.asarray(
    multicolor_disc(
        wavelength,
        agn_log_lbol=_log_lbol,
        agn_lum_ratio=1.0 - _torus_frac,
        agn_log_mbh=8.0,
        agn_log_ledd=-1.0,
    )
)

# Torus only
l_torus = np.asarray(
    two_temperature_torus(wavelength, agn_log_lbol=_log_lbol, agn_torus_frac=_torus_frac)
)

# BLR: 10% of disc luminosity
l_disc_bol_erg = (1.0 - _torus_frac) * _l_bol_erg
l_blr = (
    np.asarray(
        blr_emission(
            wavelength,
            l_disc_bol_erg=l_disc_bol_erg,
            covering_fraction=0.1,
            fwhm_kms=5000.0,
            agn_fe2_strength=1.0,
        )
    )
    / _LSUN_ERG
)

# NLR: 10% of disc luminosity
l_nlr = (
    np.asarray(
        nlr_emission(
            wavelength,
            l_disc_bol_erg=l_disc_bol_erg,
            covering_fraction=0.1,
            fwhm_kms=500.0,
        )
    )
    / _LSUN_ERG
)

# Total
l_total = l_disc + l_torus + l_blr + l_nlr

# Plot each component
ax.loglog(wave_um, l_disc * nu_arr, color=COLORS["rt"], lw=1.8, label="Accretion disc")
ax.loglog(wave_um, l_torus * nu_arr, color=COLORS["model"], lw=1.8, label="Dust torus")
ax.loglog(wave_um, l_blr * nu_arr, color="C2", lw=1.5, ls="--", label="BLR + Fe II")
ax.loglog(wave_um, l_nlr * nu_arr, color="C1", lw=1.5, ls=":", label="NLR")
ax.loglog(wave_um, l_total * nu_arr, color="black", lw=2.5, label="Total", alpha=0.7)

# Reference wavelengths
for lam_um, _name in [(0.1216, r"Ly$\alpha$"), (9.7, "Si 9.7")]:
    ax.axvline(lam_um, color="gray", ls=":", alpha=0.3, lw=0.7)

ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$\nu L_\nu$ [arb.]",
    title=r"Full Unified AGN SED: disc + torus + BLR + NLR ($\log L_{\rm bol}=44$, Type 1)",
    xlim=(1e-3, 100),
)
_set_reasonable_log_ylim_from_axes(ax)

# Region labels
for label, x_pos, y_axes in [
    ("UV", 0.02, 0.85),
    ("Optical", 0.5, 0.7),
    ("NIR", 2, 0.6),
    ("MIR", 15, 0.8),
    ("FIR", 70, 0.55),
]:
    ax.text(
        x_pos,
        y_axes,
        label,
        fontsize=7,
        color="gray",
        ha="center",
        transform=ax.get_xaxis_transform(),
    )

ax.legend(fontsize=8, ncol=2, loc="upper left")
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 2. Accretion Disc Models
#
# tengri provides three disc models with increasing physical realism:
#
# - **Power-law disc**: Minimal phenomenological model (3 params). Fast, suitable for quick tests.
# - **Multi-color disc (Shakura-Sunyaev)**: Standard thin disc with temperature zones (6 params).
#   Includes black-hole mass and Eddington ratio effects.
# - **Kubota & Done 3-zone**: Full physics with outer cool disc, warm Comptonization layer,
#   and hot X-ray corona (9+ params). See `05b_agn_advanced.py` for details.

# %%
# %% [markdown]
# ### 2a. Power-Law Disc: Spectral Slope & UV Cutoff
#

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (a) Vary spectral slope
alphas = [-0.5, -1.0, -1.5]
alpha_colors = [COLORS["rt"], "C2", COLORS["model"]]
for alpha, c in zip(alphas, alpha_colors):
    lnu = np.asarray(powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_alpha=alpha))
    axes[0].loglog(wave_um, lnu, color=c, lw=1.8, label=rf"$\alpha={alpha}$")

axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"(a) Spectral slope $\alpha$ ($T_{\rm max}=10^5$ K)",
    xlim=(1e-3, 10),
)
axes[0].legend(fontsize=8)

# (b) Vary T_max (UV cutoff)
t_maxs = [1e4, 5e4, 1e5, 1e6]
t_colors = [COLORS["rt"], "C2", COLORS["model"], "C1"]
for t_max, c in zip(t_maxs, t_colors):
    lnu = np.asarray(powerlaw_disc(wavelength, agn_log_lbol=44.0, agn_T_max=t_max))
    axes[1].loglog(wave_um, lnu, color=c, lw=1.8, label=rf"$T_{{\rm max}}={t_max:.0e}$ K")

axes[1].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"(b) UV cutoff temperature $T_{\rm max}$",
    xlim=(1e-3, 10),
)
axes[1].legend(fontsize=8)

fig.tight_layout()
plt.show()

# %% [markdown]
# ### 2b. Multi-Color Disc: Black Hole Mass & Accretion Rate
#
# The multi-color (Shakura-Sunyaev) disc links the spectral shape to the
# accretion physics: the disc temperature profile depends on black hole mass
# M_BH and Eddington ratio L/L_Edd.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (a) Vary M_BH
m_bhs = [7.0, 8.0, 9.0, 10.0]
mbh_colors = [COLORS["rt"], "C2", COLORS["model"], "C1"]
for m_bh, c in zip(m_bhs, mbh_colors):
    lnu = np.asarray(
        multicolor_disc(wavelength, agn_log_lbol=44.0, agn_log_mbh=m_bh, agn_log_ledd=-1.0)
    )
    axes[0].loglog(wave_um, lnu, color=c, lw=1.8, label=rf"$\log M_{{\rm BH}}={m_bh}$")

axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"(a) Black hole mass $M_{\rm BH}$ (fixed $L/L_{Edd}=-1$ dex)",
    xlim=(1e-3, 100),
)
axes[0].legend(fontsize=8)

# (b) Vary L/L_Edd (accretion rate)
l_edds = [-2.0, -1.0, 0.0, 1.0]
ledd_colors = [COLORS["rt"], "C2", COLORS["model"], "C1"]
for l_edd, c in zip(l_edds, ledd_colors):
    lnu = np.asarray(
        multicolor_disc(wavelength, agn_log_lbol=44.0, agn_log_mbh=8.0, agn_log_ledd=l_edd)
    )
    axes[1].loglog(wave_um, lnu, color=c, lw=1.8, label=rf"$\log(L/L_{{\rm Edd}})={l_edd}$")

axes[1].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"(b) Accretion rate $L/L_{Edd}$ (fixed $M_{\rm BH}=10^8 M_\odot$)",
    xlim=(1e-3, 100),
)
axes[1].legend(fontsize=8)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## 3. Dust Torus Models
#
# The torus re-processes disc photons absorbed at short wavelengths as thermal
# continuum at infrared wavelengths. Two options:
#
# - **Simple torus**: Single-temperature blackbody (fast, 2 params).
# - **Two-temperature torus**: Hot + cold components (more realistic, 4 params).

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (a) Single-temperature: vary T
temps = [800, 1000, 1200, 1500]
temp_colors = [COLORS["rt"], "C2", COLORS["model"], "C1"]
for temp, c in zip(temps, temp_colors):
    lnu = np.asarray(
        simple_torus(wavelength, agn_log_lbol=44.0, agn_T_torus=temp, agn_torus_frac=0.5)
    )
    axes[0].loglog(wave_um, lnu, color=c, lw=1.8, label=rf"$T={temp}$ K")

axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"(a) Simple torus: single temperature",
    xlim=(0.1, 1000),
)
axes[0].legend(fontsize=8)

# (b) Two-temperature: vary T_hot and T_warm
t_hots = [1200, 1200, 1200, 1200]
t_warms = [200, 300, 400, 500]
twtemp_colors = [COLORS["rt"], "C2", COLORS["model"], "C1"]
for t_h, t_w, c in zip(t_hots, t_warms, twtemp_colors):
    lnu = np.asarray(
        two_temperature_torus(
            wavelength,
            agn_log_lbol=44.0,
            agn_T_hot=t_h,
            agn_T_warm=t_w,
            agn_frac_hot=0.3,
            agn_torus_frac=0.5,
        )
    )
    axes[1].loglog(wave_um, lnu, color=c, lw=1.8, label=rf"$T_h={t_h}$, $T_w={t_w}$ K")

axes[1].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"(b) Two-temperature torus",
    xlim=(0.1, 1000),
)
axes[1].legend(fontsize=8)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## 4. Broad Line Region (BLR) Emission
#
# The BLR produces broad Balmer lines (H-α, H-β, [OIII]) plus Lyman-α
# and other UV lines. Line strength depends on:
#
# - **Covering fraction**: What fraction of the disc photons are intercepted by BLR gas?
# - **FWHM**: Velocity dispersion (typically 3000–10000 km/s).
# - **Fe II strength**: Pseudo-continuum from iron multiplets (optional).
#
# The BLR is **geometrically hidden at high inclination** (Type 2 AGN).

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Select wavelength range for clear line visibility
wave_blr = jnp.logspace(np.log10(1000), np.log10(7000), 500)
_l_disc_bol = (1.0 - 0.5) * 10.0**44.0 * _LSUN_ERG  # 50% disc luminosity

# (a) Vary covering fraction
cov_fracs = [0.05, 0.1, 0.2, 0.3]
cov_colors = [COLORS["rt"], "C2", COLORS["model"], "C1"]
for cov_frac, c in zip(cov_fracs, cov_colors):
    lnu = np.asarray(
        blr_emission(
            wave_blr,
            l_disc_bol_erg=_l_disc_bol,
            covering_fraction=cov_frac,
            fwhm_kms=5000.0,
            agn_fe2_strength=0.0,
        )
    )
    axes[0].loglog(np.asarray(wave_blr) / 1e4, lnu, color=c, lw=1.5, label=f"f_cov={cov_frac}")

axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title="(a) BLR: varying covering fraction",
)
axes[0].legend(fontsize=8)

# (b) Vary FWHM
fwhms = [2000, 5000, 10000]
fwhm_colors = [COLORS["rt"], "C2", COLORS["model"]]
for fwhm, c in zip(fwhms, fwhm_colors):
    lnu = np.asarray(
        blr_emission(
            wave_blr,
            l_disc_bol_erg=_l_disc_bol,
            covering_fraction=0.1,
            fwhm_kms=fwhm,
            agn_fe2_strength=0.0,
        )
    )
    axes[1].loglog(np.asarray(wave_blr) / 1e4, lnu, color=c, lw=1.5, label=f"FWHM={fwhm}")

axes[1].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title="(b) BLR: varying line width",
)
axes[1].legend(fontsize=8)

# (c) Iron pseudo-continuum strength
fe2_strengths = [0.0, 0.5, 1.0, 2.0]
fe2_colors = [COLORS["rt"], "C2", COLORS["model"], "C1"]
for fe2_str, c in zip(fe2_strengths, fe2_colors):
    lnu = np.asarray(
        blr_emission(
            wave_blr,
            l_disc_bol_erg=_l_disc_bol,
            covering_fraction=0.1,
            fwhm_kms=5000.0,
            agn_fe2_strength=fe2_str,
        )
    )
    axes[2].loglog(np.asarray(wave_blr) / 1e4, lnu, color=c, lw=1.5, label=f"Fe II={fe2_str}")

axes[2].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title="(c) BLR: Iron pseudo-continuum",
)
axes[2].legend(fontsize=8)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## 5. Narrow Line Region (NLR) Emission
#
# The NLR is isotropic (visible at all inclinations) and produces weaker
# forbidden lines: [OIII], [OII], [NII], [SII], Balmer (narrow component).
# NLR parameters:
#
# - **Covering fraction**: Fraction of AGN luminosity ionizing the NLR.
# - **FWHM**: Typically 300–500 km/s (narrower than BLR).
#
# NLR geometry is **not** masked by the torus: it's extended emission beyond
# the torus opening angle.

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# (a) Vary covering fraction
nlr_cov_fracs = [0.05, 0.1, 0.2]
nlr_cov_colors = [COLORS["rt"], "C2", COLORS["model"]]
for cov_frac, c in zip(nlr_cov_fracs, nlr_cov_colors):
    lnu = np.asarray(
        nlr_emission(
            wave_blr,
            l_disc_bol_erg=_l_disc_bol,
            covering_fraction=cov_frac,
            fwhm_kms=500.0,
        )
    )
    axes[0].loglog(np.asarray(wave_blr) / 1e4, lnu, color=c, lw=1.5, label=f"f_cov={cov_frac}")

axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title="(a) NLR: varying covering fraction",
)
axes[0].legend(fontsize=8)

# (b) Vary FWHM
nlr_fwhms = [300, 500, 1000]
nlr_fwhm_colors = [COLORS["rt"], "C2", COLORS["model"]]
for fwhm, c in zip(nlr_fwhms, nlr_fwhm_colors):
    lnu = np.asarray(
        nlr_emission(
            wave_blr,
            l_disc_bol_erg=_l_disc_bol,
            covering_fraction=0.1,
            fwhm_kms=fwhm,
        )
    )
    axes[1].loglog(np.asarray(wave_blr) / 1e4, lnu, color=c, lw=1.5, label=f"FWHM={fwhm}")

axes[1].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title="(b) NLR: varying line width",
)
axes[1].legend(fontsize=8)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## 6. Named Model Registry
#
# The `resolve_agn_model()` function provides convenient access to
# pre-configured AGN models with sensible defaults. Available models:

# %%
# List registered models
from tengri.components.agn.unified import AGN_MODELS

print("Registered AGN models:")
for name in sorted(AGN_MODELS.keys()):
    print(f"  - {name}")

# %%
# %% [markdown]
# ### 6a. Simple Model (3 params)
#
# Minimal: power-law disc + single-temperature torus.

# %%
fig, ax = plt.subplots(figsize=(8, 5))

simple_fn = resolve_agn_model("simple")
lnu_simple = np.asarray(
    simple_fn(
        wavelength,
        agn_log_lbol=44.0,
        agn_lum_ratio=1.0,
        agn_alpha=-1.0,
        agn_T_torus=1000.0,
        agn_torus_frac=0.5,
    )
)

ax.loglog(
    wave_um,
    lnu_simple,
    color=COLORS["rt"],
    lw=2.0,
    label="simple model (pwr-law disc + 1T torus)",
)
ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"Registered Model: 'simple'",
    xlim=(1e-2, 1000),
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ### 6b. Standard Model (6 params)
#
# More realistic: multi-color disc + two-temperature torus.

# %%
fig, ax = plt.subplots(figsize=(8, 5))

standard_fn = resolve_agn_model("standard")
lnu_standard = np.asarray(
    standard_fn(
        wavelength,
        agn_log_lbol=44.0,
        agn_lum_ratio=1.0,
        agn_log_mbh=8.0,
        agn_log_ledd=-1.0,
        agn_T_hot=1200.0,
        agn_T_warm=300.0,
        agn_frac_hot=0.3,
        agn_torus_frac=0.5,
    )
)

ax.loglog(
    wave_um,
    lnu_standard,
    color="C2",
    lw=2.0,
    label="standard model (multi-color disc + 2T torus)",
)
ax.set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title=r"Registered Model: 'standard'",
    xlim=(1e-2, 1000),
)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## 7. Type 1 vs Type 2 AGN: Inclination Effects
#
# The unified model assumes BLR is hidden at high inclination (edge-on).
# Here we show both orientations with the same underlying model.

# %%
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

# Both use the standard model with BLR/NLR included
nlr_cov = 0.1
blr_cov = 0.1

# Type 1: face-on (cos_inc = 1.0) → BLR visible
l_disc_1 = np.asarray(
    multicolor_disc(
        wavelength,
        agn_log_lbol=44.0,
        agn_lum_ratio=0.5,
        agn_log_mbh=8.0,
        agn_log_ledd=-1.0,
    )
)
l_torus_1 = np.asarray(
    two_temperature_torus(
        wavelength,
        agn_log_lbol=44.0,
        agn_torus_frac=0.5,
        agn_T_hot=1200.0,
        agn_T_warm=300.0,
    )
)
l_disc_bol_1 = 0.5 * 10.0**44.0 * _LSUN_ERG
l_blr_1 = (
    np.asarray(
        blr_emission(
            wavelength,
            l_disc_bol_erg=l_disc_bol_1,
            covering_fraction=blr_cov,
            fwhm_kms=5000.0,
        )
    )
    / _LSUN_ERG
)
l_nlr_1 = (
    np.asarray(
        nlr_emission(
            wavelength,
            l_disc_bol_erg=l_disc_bol_1,
            covering_fraction=nlr_cov,
            fwhm_kms=500.0,
        )
    )
    / _LSUN_ERG
)
l_tot_1 = l_disc_1 + l_torus_1 + l_blr_1 + l_nlr_1

# Type 2: edge-on (cos_inc = 0.1) → BLR hidden, NLR still visible
l_blr_2 = l_blr_1 * 0.01  # Suppress BLR at edge-on
l_tot_2 = l_disc_1 + l_torus_1 + l_blr_2 + l_nlr_1

axes[0].loglog(wave_um, l_tot_1, color=COLORS["rt"], lw=2.0, label="Type 1 (face-on)")
axes[0].loglog(wave_um, l_blr_1, color="C2", lw=1.2, ls="--", alpha=0.8, label="  + BLR visible")
axes[0].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title="Type 1 AGN (face-on, cos(i)=1.0)",
    xlim=(1e-2, 1000),
)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)

axes[1].loglog(wave_um, l_tot_2, color=COLORS["model"], lw=2.0, label="Type 2 (edge-on)")
axes[1].loglog(wave_um, l_blr_2, color="C2", lw=1.2, ls="--", alpha=0.3, label="  + BLR hidden")
axes[1].set(
    xlabel=r"Wavelength [$\mu$m]",
    ylabel=r"$L_\nu$ [$L_\odot$ Hz$^{-1}$]",
    title="Type 2 AGN (edge-on, cos(i)=0.1)",
    xlim=(1e-2, 1000),
)
axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.show()

# %% [markdown]
# ## 8. Summary: Unified AGN in `tengri`
#
# The unified AGN model brings together:
#
# | Component | Code | Params | Physics |
# |-----------|------|--------|---------|
# | Disc | `multicolor_disc()`, `powerlaw_disc()` | 3–6 | Accretion, M_BH, L/L_Edd |
# | Torus | `simple_torus()`, `two_temperature_torus()` | 2–4 | IR reprocessing, covering |
# | BLR | `blr_emission()` | 2–3 | Broad lines, Fe II, **inclination-masked** |
# | NLR | `nlr_emission()` | 2–3 | Forbidden lines, **isotropic** |
# | Combined | `unified_agn()`, `resolve_agn_model()` | 7–12+ | Full SED fit |
#
# **Key advantage:** Unlike CIGALE and Prospector, tengri's disc, torus, and
# lines are all JIT-compilable JAX functions, enabling **differentiable AGN
# parameter inference** via HMC or variational inference.
#
# **Next steps:**
# - See `05b_agn_advanced.py` for Kubota & Done 3-zone disc, ADAF, and SKIRTOR torus.
# - See `06_multiwavelength_gallery.py` for X-ray corona and radio jet.
# - See `07_fitting_photometry.py` to fit AGN parameters to data.
