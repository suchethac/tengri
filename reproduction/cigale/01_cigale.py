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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Reproducing CIGALE's physics with tengri
#
# CIGALE (Boquien et al. 2019, A&A 622, A103) is the workhorse for
# panchromatic SED fitting. This study configures tengri's public API to approximate CIGALE's model choices; tengri's implementation is its own, not derived from CIGALE's code, and residual differences are documented below.
# This notebook places its physics modules next to their tengri equivalents at the same parameters, as shown in the module map below.
# Same parameters in, same SED out: any disagreement is physics, not data or fitting.
#
# Both codes consume the same BC03 templates: CIGALE's bundled
# Chabrier-IMF grid (Bruzual & Charlot 2003), repackaged into the DSPS
# HDF5 layout by `_drivers/cigale_ssp_to_dsps.py`.
#
# The fiducial galaxy throughout: τ-delayed SFH with τ = 1 Gyr,
# age = 5 Gyr; Z = Z☉; modified-starburst dust with E(B−V)_lines = 0.3;
# Dale et al. (2014) IR re-emission with α = 2. Each section sweeps one
# physics block around this fiducial.
#
# **What to expect.** The shared SSP grid, the parametric SFHs and the
# attenuation laws agree to floating point or a fraction of a percent; the dust
# IR, the SKIRTOR AGN components, the X-ray corona and the radio synchrotron
# agree to about 1 %. Three things differ, and each is stated where it is
# measured: the age convention of the SFH ⊗ SSP convolution (§3), exact in tengri
# and a discrete 1 Myr sum, equivalent to aging every star by 0.5 Myr, in CIGALE,
# which sets a ~1 % floor under the dust IR and the radio; the nebular emitter
# (Cue trained on Cloudy 22.00, Li et al. 2025, ApJ 986, 9, arXiv:2405.04598)
# compared with CIGALE's bundled Cloudy grids (§8, §11b); and the Fritz (2006)
# torus library (§9e). Two further differences sit on CIGALE's side: its X-ray
# templates are normalized on a truncated wavelength grid (§10), and its polar-dust
# reference is integrated on a coarser axis than tengri's (§9d). The Meiksin IGM
# agrees where one Lyman line is active and differs blueward of Lyβ (§12).
# Every section prints the number it claims.

# %% [markdown]
# ## Setup

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import warnings

# Register CIGALE's own Dale et al. 2014 grid for this reproduction notebook.
# tengri ships two Dale grids. ``dale2014`` (``data/dale2014_templates.h5``) is
# the unmodified Wyoming-source release, which embeds a star-forming radio
# synchrotron continuum rising to 2.2459e9 Å; ``dale2014_cigale``
# (``data/dale2014_templates_cigale.h5``) is CIGALE's own ``dale2014`` template,
# whose red end falls away because CIGALE adds radio in a separate module.
# Use ``dale2014_cigale`` whenever the model also carries a ``radio={...}`` block
# (the radio block supplies the synchrotron once), and to compare with pcigale.
# Both files come from ``scripts/regenerate_dale2014_from_{cigale,official}.py``.
from pathlib import (
    Path,
    Path as _Path,
)

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from reproduction import _validation as V
from reproduction.cigale._drivers import cigale_driver as C, units as U

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform, load_ssp_data
from tengri.components.radio.radio import radio_freefree
from tengri.dust import register_dale2014_tabulated
from tengri.utils.physics_constants import C_AA, L_SUN, LOG10_ZSUN

# Force the inline backend so figures embed on (re-)render regardless of the
# ambient MPLBACKEND. A non-inline backend (e.g. Agg) drops the save_fig()
# auto-display and produces a figure-less notebook. No-op when run as a script.
try:  # noqa: SIM105
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

# nbclient kernels don't bind ``__file__``; fall back to cwd to work around
# environment differences between kernel and file-system contexts.
_CIGALE_DALE_PARENT = (
    _Path(__file__).parent.parent.parent
    if "__file__" in dir()
    else _Path.cwd().resolve().parent.parent
)
_CIGALE_DALE_PATH = _CIGALE_DALE_PARENT / "data" / "dale2014_templates_cigale.h5"
if _CIGALE_DALE_PATH.is_file():
    register_dale2014_tabulated(str(_CIGALE_DALE_PATH), name="dale2014_cigale")

warnings.filterwarnings("ignore")
warnings.filterwarnings("default", module=r"tengri(\.|$)")
tengri.plot.setup_style()
plt.rcParams["figure.dpi"] = 120  # 22 inline figures, 4 MiB cap; ×1.44 scaling

# Unit-sanity guard: every panel below claims percent-level agreement,
# which rests on the CIGALE-W/nm → tengri-erg/s/Hz converter in
# ``_drivers/units.py``. A factor-of-10 or 1e7 bug there would silently
# misshape every comparison. Assert the bolometric round-trip here so the
# entire notebook trips at Setup if the converter ever drifts.
_unit_check = U.verify_unit_conversion(rtol=1e-3)
print(
    f"unit-conversion bolometric round-trip: "
    f"rel_err = {_unit_check['rel_err']:.2e}  (target < 1e-3)"
)

# CIGALE's `sfhdelayed(..., normalise=True)` integrates the τ-delayed
# shape to 1 M☉ formed. tengri's parametric SFHs adopt the Bagpipes /
# Prospector convention: every shape is rescaled so that
# `trapezoid(SFR, t_lookback) = 10**log_total_mass` exactly. Setting
# `log_total_mass = 0.0` is the bit-for-bit equivalent.
#
# Dust mapping: CIGALE's `dustatt_modified_starburst(E_BV_lines)` applies a
# **single** Calzetti screen to the stellar continuum. tengri's `two_component`
# (Charlot & Fall) has a birth-cloud term; to match CIGALE, set
# `tau_bc = 0`, `tau_diff = R_V * 0.44 * E_BV / 1.086`.
_E_BV_LINES = 0.3
_R_V_CALZETTI = 4.05
_F_CONT_OVER_LINES = 0.44
TAU_DIFF_FIDUCIAL = _R_V_CALZETTI * _F_CONT_OVER_LINES * _E_BV_LINES / 1.086
TAU_BC_FIDUCIAL = 0.0  # CIGALE modified_starburst = single continuum screen

# Metallicity pin — CIGALE bc03(metallicity=0.02) is Z_abs = 0.02 (≈ Z_⊙).
# tengri's met_logzsol = log10(Z/Z_⊙) with Z_⊙ = 10**LOG10_ZSUN ≈ 0.0142
# (Asplund+2009). Pin explicitly so the comparison is bit-aligned regardless
# of registry-default convention.
MET_LOGZSOL = float(np.log10(0.02) - LOG10_ZSUN)  # ≈ +0.149
MET_FIDUCIAL = {"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)}

# One fiducial nebular configuration, reused in every build below (§1 excepted):
# Cue at logU = -2, Z_gas matched to the stellar metallicity, f_esc = 0 — the
# same values §8 uses. Only §8b varies logU, Z_gas or f_esc. Every
# tengri/CIGALE residual from here on therefore also carries the Cue-vs-CLOUDY
# nebular-model difference §8 quantifies, on top of whatever else the section
# is measuring.
NEB_FIDUCIAL_TENGRI = {
    "type": "cue",
    "neb_logU": Fixed(-2.0),
    "neb_logZ_gas": Fixed(MET_LOGZSOL),
    "neb_fesc": Fixed(0.0),
    # neb_fdust_frac = f_dust / (1 - f_esc): CIGALE's f_esc = 0, f_dust = 0
    # below maps to 0.0. Stated explicitly, like neb_fesc above, so the
    # pairing is visible in the configuration.
    "neb_fdust_frac": Fixed(0.0),
    "all_params": Fixed(DEFAULT),
}
NEB_FIDUCIAL_CIGALE = (
    "nebular",
    dict(
        logU=-2.0,
        zgas=0.02,
        ne=100,
        f_esc=0.0,
        f_dust=0.0,
        lines_width=300.0,
        emission=True,
        line_list="",
    ),
)

# Notebook-vs-script compatible: ``__file__`` is undefined when this
# is run via nbclient (the kernel's resources path is set to the
# reproduction/cigale/ directory instead), so fall back to the CWD.
_HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd().resolve()
figs_dir = _HERE / "_figs"
figs_dir.mkdir(exist_ok=True)


_FIG_DPI = 120


def save_fig(filename: str) -> None:
    """Save figure to ``_figs/`` and leave it open so inline embeds work."""
    plt.savefig(str(figs_dir / filename), dpi=_FIG_DPI, bbox_inches="tight")


def _lbol_nu(wave_aa, l_nu):
    """∫L_ν dν [erg/s] from L_ν [erg/s/Hz] on a wavelength grid [Å]."""
    w = np.asarray(wave_aa)
    order = np.argsort(w)[::-1]  # → increasing frequency
    return float(np.trapezoid(np.asarray(l_nu)[order], C_AA / w[order]))


def _assert_comparable(arr_ref, arr_t, *, name: str) -> None:
    """Guard against shipping a blank or wildly mis-scaled panel."""
    a_ref = np.asarray(arr_ref)
    a_t = np.asarray(arr_t)
    assert np.isfinite(a_ref).any() and np.isfinite(a_t).any(), f"{name}: NaN-only"
    assert (a_ref > 0).any() and (a_t > 0).any(), f"{name}: zero/negative-only"
    ratio = a_ref.max() / a_t.max()
    assert 1e-3 < ratio < 1e3, f"{name}: y-scale ratio {ratio:.2e} out of range"


def _band_energy_ratio(w_ref, l_ref, w_t, l_t, lo_aa: float, hi_aa: float) -> float:
    """Energy-weighted tengri / reference ratio, ∫L_ν dν over [lo_aa, hi_aa] [Å].

    tengri is regridded onto the reference grid and both are integrated over
    the same points, so the ratio is the ratio of band luminosities. A median
    over grid points would instead weight a band by how many points each code
    places in it, and CIGALE's nebular line grid puts far more points in the
    ultraviolet than in the optical.
    """
    w = np.asarray(w_ref)
    sel = (w >= lo_aa) & (w <= hi_aa)
    if int(sel.sum()) < 2:
        return float("nan")
    l_t_on_ref = U.regrid(np.asarray(w_t), np.asarray(l_t), w)
    num = _lbol_nu(w[sel], l_t_on_ref[sel])
    den = _lbol_nu(w[sel], np.asarray(l_ref)[sel])
    return num / den if den > 0 else float("nan")


# %% [markdown]
# ## Common SSP grid
#
# CIGALE's BC03 Chabrier templates re-shaped into the DSPS HDF5 layout
# that tengri reads, with the same numerical SSPs on both sides.

# %%
ssp_file = _HERE / "_drivers" / "data" / "bc03_from_cigale.h5"
if not ssp_file.is_file():
    raise SystemExit(
        f"SSP grid {ssp_file} is missing. Generate it once with:\n"
        f"    python -m reproduction.cigale._drivers.cigale_ssp_to_dsps"
    )
ssp = load_ssp_data(str(ssp_file.resolve()))
print(
    f"BC03 Chabrier SSP: {ssp.ssp_wave.shape[0]} wavelengths, "
    f"{ssp.ssp_lgmet.shape[0]} metallicities, "
    f"{ssp.ssp_lg_age_gyr.shape[0]} age bins.\n"
    f"repackaged from pcigale {C.cigale_version()}"
)


# %% [markdown]
# ## Module map
#
# The CIGALE module on the left, the tengri registry on the right. The
# tengri side is enumerated live via `tengri.list_*()` so the table reflects
# what the installed version actually exposes.


# %%
def _names(rows):
    return [r["name"] for r in rows]


registries = {
    "SFH": (
        "sfhdelayed · sfh2exp · sfhdelayedbq · sfhperiodic · sfhfromfile · "
        "sfh_buat08 · sfhstochastic_carvajal2025",
        _names(tengri.list_sfh_models()),
    ),
    "Nebular": (
        "nebular (static CLOUDY grids)",
        _names(tengri.list_nebular_backends()),
    ),
    "Dust attenuation": (
        "calzleit · modified_CF00 · modified_starburst · powerlaw · 2powerlaws",
        _names(tengri.list_dust_laws()),
    ),
    "Dust IR emission": (
        "dl2007 · dl2014 · dale2014 · casey2012 · schreiber2016",
        _names(tengri.list_dust_emission_models()),
    ),
    "AGN": (
        "fritz2006 · skirtor2016 · dale2014 (fracAGN)",
        _names(tengri.list_agn_models()),
    ),
    "X-ray": (
        "xray (Yang+2020 corona + XRB + photoelectric N_H)",
        _names(tengri.list_xray_models()),
    ),
    "Radio": (
        "radio (q_IR + radio-loud AGN power-law)",
        _names(tengri.list_radio_models()),
    ),
    "IGM": (
        "redshifting (Meiksin 2006)",
        _names(tengri.list_igm_models()),
    ),
}

for block, (cig, tng) in registries.items():
    print(f"\n{block}")
    print(f"  CIGALE  {cig}")
    print(f"  tengri  {', '.join(tng)}")


# %% [markdown]
# ## §1 Stellar populations
#
# BC03 Chabrier (Bruzual & Charlot 2003) at Z = 0.02 from 1 Myr to 10 Gyr:
# CIGALE's `bc03/Z=0.02_imf=chab.pickle` read directly against the same
# templates in tengri's HDF5. The relative residual |tengri − CIGALE| /
# CIGALE is the float32 round-trip of the repackaged grid and nothing else:
# median 2e-8 at every age, and max 6e-8 at four of the five. Only the 100 Myr
# SSP has points above 1e-7, which are four of its 1218 non-zero points, all in the
# extreme UV below 240 Å where the SSP is 15 decades below its peak and the
# relative measure is reading the last float32 digit of a number near zero
# (the 230 Å point, $L_\nu = 1.5\times10^{-7}$ of the peak, is the 5e-6 spike
# on the residual panel).
#
# Both sides divide by the same speed of light: tengri's `C_AA`, which
# `_drivers/cigale_ssp_to_dsps.py` also uses to write the grid and
# `_drivers/units.py` to convert CIGALE's W/nm. A rounded c on either side
# puts a floor of $|\Delta c|/c$ under this residual at every wavelength and
# every age, 2.5e-5 for the usual 2.998e18, which is three decades above the float32
# floor this panel is measuring.

# %%
import pickle as _pickle
from pathlib import Path as _P

ages_yr = [1e6, 1e7, 1e8, 1e9, 1e10]

# CIGALE side: raw BC03 Chabrier Z=0.02 pickle, converted W/nm/Msun →
# Lsun/Hz/Msun (the exact conversion used by _drivers/cigale_ssp_to_dsps.py).
import sys as _sys

_pkl_path = next(
    (
        _p / "pcigale" / "data" / "bc03" / "Z=0.02_imf=chab.pickle"
        for _p in map(_P, _sys.path)
        if (_p / "pcigale" / "data" / "bc03" / "Z=0.02_imf=chab.pickle").exists()
    ),
    _P(_sys.prefix)
    / "lib"
    / "python3.12"
    / "site-packages"
    / "pcigale"
    / "data"
    / "bc03"
    / "Z=0.02_imf=chab.pickle",
)
with open(_pkl_path, "rb") as _f:
    _raw = _pickle.load(_f)
_wl_aa = np.asarray(_raw.wl) * 10.0  # nm → Å
cigale_ssp = []
for age_yr in ages_yr:
    ia = int(np.argmin(np.abs(np.asarray(_raw.t) - age_yr / 1e6)))  # raw.t in Myr
    lnu = np.asarray(_raw.spec[:, ia]) * 1e6 * _wl_aa**2 / C_AA / L_SUN  # Lsun/Hz/Msun
    cigale_ssp.append((_wl_aa, lnu * L_SUN))  # → erg/s/Hz/Msun for plotting

i_zsun = int(np.argmin(np.abs(ssp.ssp_lgmet - np.log10(0.02))))
tengri_ssp = []
for age_yr in ages_yr:
    i_age = int(np.argmin(np.abs(ssp.ssp_lg_age_gyr - np.log10(age_yr / 1e9))))
    # ssp_flux axes: (n_met, n_age, n_wave) — metallicity first, then age.
    tengri_ssp.append((ssp.ssp_wave, ssp.ssp_flux[i_zsun, i_age, :] * L_SUN))

# Overlay both codes on one SED axis + a residual panel underneath.
fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
colors = plt.cm.viridis(np.linspace(0, 1, len(ages_yr)))
print("§1 SSP residual |tengri − CIGALE| / CIGALE (float32 eps = 1.19e-7):")
for color, age_yr, (w_c, L_c), (w_t, L_t) in zip(colors, ages_yr, cigale_ssp, tengri_ssp):
    label = f"{age_yr / 1e6:g} Myr"
    ax.plot(w_c, L_c, color=color, linewidth=2.0, label=label)
    ax.plot(w_t, L_t, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
    # Residual on the CIGALE wavelength grid (tengri regridded onto it).
    L_t_on_c = U.regrid(w_t, L_t, w_c)
    resid = np.abs(L_t_on_c - L_c) / np.maximum(np.abs(L_c), 1e-30)
    resid[~np.isfinite(resid)] = 0.0
    _pos = L_c > 0
    print(
        f"    {age_yr / 1e6:>6g} Myr: median {np.median(resid[_pos]):.2e}, "
        f"max {resid[_pos].max():.2e}, "
        f"{int((resid[_pos] > 1e-7).sum())}/{int(_pos.sum())} points above 1e-7"
    )
    ax_r.plot(w_c, resid, color=color, linewidth=1.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$\nu L_\nu$ or $L_\nu$ [erg/s/Hz]")
ax.set_title("BC03 Chabrier Z = 0.02 — CIGALE (solid) vs tengri (black dashed)")
ax.legend(fontsize=9, title="SSP age")
ax.grid(True, alpha=0.3)
ax_r.set_xscale("log")
ax_r.set_yscale("log")
ax_r.set_xlabel(r"$\lambda$ [Å]")
ax_r.set_ylabel(r"$|\Delta| / L_{\rm CIGALE}$", fontsize=9)
ax_r.set_ylim(1e-9, 1e-2)
ax_r.axhline(1e-6, color="gray", linestyle=":", alpha=0.6)
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("cigale_01_ssp_bc03.png")


# %% [markdown]
# ## §2 Star formation histories
#
# CIGALE module `sfhdelayed` ↔ tengri `sfh={'type': 'delayed'}`: the same
# τ-delayed shape, SFR(t) ∝ t · exp(−t/τ), peaking at t = τ. Both integrate to
# 1 M☉ formed by `age` (CIGALE via `normalise=True`, tengri via
# `log_total_mass = 0.0`). The figure overlays the two SFR(t) curves; tengri's
# is `state.derived["sfr_history"]` on a 256-point log-spaced lookback grid.
#
# Each subsection prints the SFR(t) shape agreement and the mass integral. The
# two codes sample the same analytic shape on different grids, so CIGALE is
# interpolated onto tengri's points and the median and worst-point ratios are
# reported. The plotted maximum of the delayed curve falls at the log node
# nearest t = τ, which is where the grid is and not a different peak.
#
# **Verification Status:** PARTIAL (11/33): Parametric SFH family physics

# %% [markdown]
# ### τ-delayed

# %%
t_c, sfr_c = C.sfh_curve(
    "sfhdelayed",
    tau_main=1000,
    age_main=5000,
    tau_burst=50,
    age_burst=20,
    f_burst=0.0,
    sfr_A=1.0,
    normalise=True,
)

# tengri's actual pipeline SFR history (not the analytic formula): build
# a minimal SEDModel with the delayed SFH and read sfr_history off the
# resulting state. The pipeline samples on the spec's log-spaced
# lookback grid (n_grid=256 by default) — visibly smooth across the
# rise-and-decay. That grid is what every fit downstream sees, so
# plotting it honestly is the test: if the area under this curve doesn't
# integrate to 1 M☉ formed (= log_total_mass = 0.0), tengri's SFH
# normalization is broken regardless of how clean the analytic shape
# looks.
tau_gyr, age_gyr = 1.0, 5.0
_m_sfh = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(tau_gyr),
        "age_gyr": Fixed(age_gyr),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
    # dense lookback grid: the table compares the SFH form, not the
    # 256-point diagnostic grid
    n_grid=4096,
)
_state_sfh = _m_sfh.predict_state({})
_lbt_yr = np.asarray(_state_sfh.derived["sfh_grid_lbt_yr"])
_sfr_history = np.asarray(_state_sfh.derived["sfr_history"])
# Convert lookback time → cosmic age since SF onset (consistent with the
# CIGALE x-axis above): t_cosmic = age_gyr - lbt
t_t = (age_gyr - _lbt_yr / 1e9) * 1e9  # yr
sfr_t = _sfr_history
# The SFH grid spans the full cosmic lookback; keep the epoch after formation.
_keep_t = t_t >= 0
t_t_valid = t_t[_keep_t]
sfr_t_valid = sfr_t[_keep_t]
assert t_t_valid.min() >= 0, f"negative cosmic age: min={t_t_valid.min()}"
# Verify normalization: trapezoid of SFR over cosmic-age axis should be
# 10**log_total_mass = 1.0 M☉ within numerical accuracy of the n_grid pipeline.
# tengri's pipeline carries sfh_grid in decreasing lookback time, so
# integrate against the increasing-time order.
_idx = np.argsort(t_t_valid)
_mass_formed = float(np.trapezoid(sfr_t_valid[_idx], t_t_valid[_idx]))
print(f"tengri ∫SFR dt = {_mass_formed:.4f} M☉  (log_total_mass=0 → 1.0000)")


def _sfr_shape_report(label, t_c_yr, sfr_c_arr, t_t_yr, sfr_t_arr, age_gyr):
    """Print the SFR(t) shape agreement on tengri's own grid.

    The two codes sample the same shape on different grids — CIGALE on a
    uniform 1-Myr axis, tengri on a 256-point log-spaced lookback axis — so
    the comparison is CIGALE interpolated onto tengri's points, over the
    interior of the history (the first 2 % and last 1 % of the age are
    dropped: both ends are grid edges, not physics).
    """
    _c_on_t = np.interp(
        t_t_yr, np.asarray(t_c_yr), np.asarray(sfr_c_arr), left=np.nan, right=np.nan
    )
    _ok = (
        np.isfinite(_c_on_t)
        & (_c_on_t > 0)
        & (t_t_yr > 0.02 * age_gyr * 1e9)
        & (t_t_yr < 0.99 * age_gyr * 1e9)
    )
    _rr = np.asarray(sfr_t_arr)[_ok] / _c_on_t[_ok]
    _k = int(np.argmax(np.abs(_rr - 1.0)))
    print(
        f"{label} SFR(t) shape, tengri / CIGALE on tengri's grid: "
        f"median {float(np.median(_rr)):.5f}×, max {float(_rr.max()):.4f}× at "
        f"t = {t_t_yr[_ok][_k] / 1e9:.3f} Gyr, "
        f"{int((np.abs(_rr - 1.0) > 0.01).sum())}/{int(_ok.sum())} points more than 1 % off"
    )


_sfr_shape_report("§2 delayed", t_c, sfr_c, t_t_valid, sfr_t_valid, age_gyr)

# Both arms on their common interval.
_t_min_broad = max(t_c.min() / 1e9, t_t_valid.min() / 1e9)
_t_max_broad = min(t_c.max() / 1e9, t_t_valid.max() / 1e9)
_keep_c_1 = (t_c / 1e9 >= _t_min_broad) & (t_c / 1e9 <= _t_max_broad)
_keep_t_1 = (t_t_valid / 1e9 >= _t_min_broad) & (t_t_valid / 1e9 <= _t_max_broad)

_t_c_temp = (t_c / 1e9)[_keep_c_1]
_t_t_temp = (t_t_valid / 1e9)[_keep_t_1]
_t_min_aligned = max(_t_c_temp.min(), _t_t_temp.min())
_keep_final_c = _t_c_temp >= _t_min_aligned
_keep_final_t = _t_t_temp >= _t_min_aligned
_t_c_overlap = t_c[_keep_c_1][_keep_final_c]
_sfr_c_overlap = sfr_c[_keep_c_1][_keep_final_c]
_t_t_valid_overlap = t_t_valid[_keep_t_1][_keep_final_t]
_sfr_t_valid_overlap = sfr_t_valid[_keep_t_1][_keep_final_t]

print(
    f"  common interval: t_c [{_t_c_overlap.min() / 1e9:.4f}, {_t_c_overlap.max() / 1e9:.4f}] Gyr, "
    f"t_t range [{_t_t_valid_overlap.min() / 1e9:.4f}, {_t_t_valid_overlap.max() / 1e9:.4f}] Gyr"
)

fig, (ax, ax_r), ratio = V.sweep_fig(
    [
        (
            "delayed-τ",
            _t_c_overlap / 1e9,
            _sfr_c_overlap,
            _t_t_valid_overlap / 1e9,
            _sfr_t_valid_overlap,
        )
    ],
    ref_label="pcigale",
    title="§2 τ-delayed SFH (τ=1 Gyr, age=5 Gyr)",
    xlabel="Cosmic age since SF onset [Gyr]",
    ylabel=r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]",
    xlim=(0, 5),
    logy=False,
)
ax.axvline(1.0, color="gray", linestyle=":", alpha=0.6, label=r"$\tau$ = 1 Gyr")
ax.legend(fontsize=9)
fig.tight_layout()
save_fig("cigale_02_sfh_tau.png")


# %% [markdown]
# ### Note on the FSPS / BAGPIPES declining-exponential
#
# CIGALE's `sfh2exp(f_burst=0)` and FSPS's `sfh=1` both produce a
# declining exponential peaking at galaxy formation, a *different*
# shape from CIGALE's `sfhdelayed` plotted above. tengri intentionally
# does **not** register this shape as `sfh.tau`, to avoid confusion with
# the τ-delayed model, which has opposite physics; `delayed`
# (= CIGALE `sfhdelayed`) is the only τ-style SFH in the registry. The
# `declining_exponential` function remains importable from the `sfh`
# builder grammar for expert use cases.


# %% [markdown]
# ### Double-exponential with a burst (sfh2exp)
#
# CIGALE's `sfh2exp` superposes an old declining-exponential population and a
# recent exponential burst that carries a fixed fraction `f_burst` of the total
# stellar mass. tengri registers the same form (`sfh.sfh2exp`). At matched
# parameters (τ_main = 4 Gyr, τ_burst = 0.1 Gyr, f_burst = 0.1, burst 0.3 Gyr
# ago, age 10 Gyr) the two histories agree everywhere except at the burst
# onset.
#
# **The burst onset is a step, and the two codes resolve it differently.**
# `f_burst` turns the burst on discontinuously at `burst_age`. CIGALE
# evaluates its SFH on a uniform 1-Myr grid, so it puts a grid point on each
# side of the step; tengri's 256-point log-spaced lookback grid places a point
# on the step itself. The printed comparison gives the median ratio and the
# number of points more than 1 % from CIGALE; the points that differ sit at
# the discontinuity, so this is a sampling difference and not a difference in
# the SFH. The printed ∫SFR dt is the trapezoid sum over tengri's sampled
# history, to be read against the 1 M☉ the shape is normalized to.

# %%
t_c2, sfr_c2 = C.sfh_curve(
    "sfh2exp",
    tau_main=4000,
    tau_burst=100,
    f_burst=0.1,
    age=10000,
    burst_age=300,
    sfr_0=1.0,
    normalise=True,
)

_age_gyr_2exp = 10.0
_m_2exp = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "sfh2exp",
        "tau_main_gyr": Fixed(4.0),
        "tau_burst_gyr": Fixed(0.1),
        "f_burst": Fixed(0.1),
        "age_gyr": Fixed(_age_gyr_2exp),
        "burst_age_gyr": Fixed(0.3),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
    n_grid=4096,
)
_st_2exp = _m_2exp.predict_state({})
_lbt_2exp = np.asarray(_st_2exp.derived["sfh_grid_lbt_yr"])
_sfr_2exp = np.asarray(_st_2exp.derived["sfr_history"])
_t_2exp = (_age_gyr_2exp - _lbt_2exp / 1e9) * 1e9  # cosmic age since SF onset [yr]
# Restrict to the epoch the galaxy exists: keep only samples with _t_2exp >= 0
_keep_2exp = _t_2exp >= 0
_t_2exp_valid = _t_2exp[_keep_2exp]
_sfr_2exp_valid = _sfr_2exp[_keep_2exp]
assert _t_2exp_valid.min() >= 0, f"negative cosmic age: min={_t_2exp_valid.min()}"
_idx2 = np.argsort(_t_2exp_valid)
_mass_2exp = float(np.trapezoid(_sfr_2exp_valid[_idx2], _t_2exp_valid[_idx2]))
print(f"tengri ∫SFR dt = {_mass_2exp:.4f} M☉  (log_total_mass=0 → 1.0000)")
_sfr_shape_report("§2 sfh2exp", t_c2, sfr_c2, _t_2exp_valid, _sfr_2exp_valid, _age_gyr_2exp)

# Both arms on their common interval.
_t2_min_broad = max(t_c2.min() / 1e9, _t_2exp_valid.min() / 1e9)
_t2_max_broad = min(t_c2.max() / 1e9, _t_2exp_valid.max() / 1e9)
_keep2_c_1 = (t_c2 / 1e9 >= _t2_min_broad) & (t_c2 / 1e9 <= _t2_max_broad)
_keep2_t_1 = (_t_2exp_valid / 1e9 >= _t2_min_broad) & (_t_2exp_valid / 1e9 <= _t2_max_broad)

_t2_c_temp = (t_c2 / 1e9)[_keep2_c_1]
_t2_t_temp = (_t_2exp_valid / 1e9)[_keep2_t_1]
_t2_min_aligned = max(_t2_c_temp.min(), _t2_t_temp.min())
_keep2_final_c = _t2_c_temp >= _t2_min_aligned
_keep2_final_t = _t2_t_temp >= _t2_min_aligned
_t_c2_overlap = t_c2[_keep2_c_1][_keep2_final_c]
_sfr_c2_overlap = sfr_c2[_keep2_c_1][_keep2_final_c]
_t_2exp_valid_overlap = _t_2exp_valid[_keep2_t_1][_keep2_final_t]
_sfr_2exp_valid_overlap = _sfr_2exp_valid[_keep2_t_1][_keep2_final_t]

print(
    f"  common interval: t_c2 [{_t_c2_overlap.min() / 1e9:.4f}, {_t_c2_overlap.max() / 1e9:.4f}] Gyr, "
    f"_t_2exp range [{_t_2exp_valid_overlap.min() / 1e9:.4f}, {_t_2exp_valid_overlap.max() / 1e9:.4f}] Gyr"
)

fig, (ax, ax_r), ratio = V.sweep_fig(
    [
        (
            "sfh2exp",
            _t_c2_overlap / 1e9,
            _sfr_c2_overlap,
            _t_2exp_valid_overlap / 1e9,
            _sfr_2exp_valid_overlap,
        )
    ],
    ref_label="pcigale",
    title="§2 sfh2exp (main + burst, age=10 Gyr)",
    xlabel="Cosmic age since SF onset [Gyr]",
    ylabel=r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]",
    xlim=(0, 10),
    logy=False,
    ratio_ylim=(0.8, 1.2),
)
ax.axvline(_age_gyr_2exp - 0.3, color="gray", linestyle=":", alpha=0.6, label="burst onset")
ax.legend(fontsize=9)
ax.set_ylim(bottom=0.0)
fig.tight_layout()
save_fig("cigale_02_sfh2exp.png")


# %% [markdown]
# ### §2c Beyond delayed: delayed_bq, periodic, buat08
#
# Three more parametric SFHs. `delayed_bq` (Ciesla+2017) adds a
# burst/quench step of ratio `r_sfr` at lookback `age_bq`; `periodic`
# repeats one burst shape (`burst_type` 0/1/2 = exponential/delayed/
# rectangular) every `delta_bursts`; `buat08` sets the shape from a
# rotational velocity, with the same parameters in Gyr as `sfh.delayed_bq`,
# `sfh.periodic`, `sfh.buat08`. Seven cases: τ_main=2, age_main=8,
# age_bq=0.5 Gyr, r_sfr ∈ {0.1, 5}; burst_type ∈ {0, 1, 2} at δ=1, τ=0.2,
# age=8 Gyr; velocity ∈ {150, 250} km/s at age=8 Gyr. The shapes agree
# (median ratio 1.000 in every case except periodic exponential, 0.999, and
# periodic rectangular, 0.993). The
# large maximum deviations sit on discontinuities, the delayed_bq
# burst/quench step at T = age_main − age_bq and the periodic burst onsets
# at T = kΔ, where each code's time grid samples the edge differently.
# Tabulated histories enter tengri via `Catalog.from_histories`, so
# `sfhfromfile` is not compared.

# %%
_AGE_2C_GYR = 8.0
_DUST_OFF_2C = {
    "law": "power_law",
    "type": "two_component",
    "tau_bc": Fixed(0.0),
    "tau_diff": Fixed(0.0),
    "all_params": Fixed(DEFAULT),
}
_cases_2c = []

for _r in (0.1, 5.0):
    _t_c, _sfr_c = C.sfh_curve(
        "sfhdelayedbq",
        tau_main=2000,
        age_main=8000,
        age_bq=500,
        r_sfr=_r,
        sfr_A=1.0,
        normalise=True,
    )
    _m = SEDModel.build(
        ssp_data=ssp,
        met=MET_FIDUCIAL,
        sfh={
            "type": "delayed_bq",
            "tau_main_gyr": Fixed(2.0),
            "age_main_gyr": Fixed(_AGE_2C_GYR),
            "age_bq_gyr": Fixed(0.5),
            "r_sfr": Fixed(_r),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=_DUST_OFF_2C,
        neb=NEB_FIDUCIAL_TENGRI,
        redshift=Fixed(0.0),
        n_grid=4096,
    )
    _st = _m.predict_state({})
    _lbt = np.asarray(_st.derived["sfh_grid_lbt_yr"])
    _sfr_t = np.asarray(_st.derived["sfr_history"])
    _t_t = (_AGE_2C_GYR - _lbt / 1e9) * 1e9
    _idx = np.argsort(_t_t)
    _cases_2c.append(
        (f"delayed_bq r_sfr={_r}", np.asarray(_t_c), np.asarray(_sfr_c), _t_t[_idx], _sfr_t[_idx])
    )

for _bt in (0, 1, 2):
    _t_c, _sfr_c = C.sfh_curve(
        "sfhperiodic",
        type_bursts=_bt,
        delta_bursts=1000,
        tau_bursts=200,
        age=8000,
        sfr_A=1.0,
        normalise=True,
    )
    _m = SEDModel.build(
        ssp_data=ssp,
        met=MET_FIDUCIAL,
        sfh={
            "type": "periodic",
            "delta_bursts_gyr": Fixed(1.0),
            "tau_bursts_gyr": Fixed(0.2),
            "burst_type": Fixed(_bt),
            "age_gyr": Fixed(_AGE_2C_GYR),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=_DUST_OFF_2C,
        neb=NEB_FIDUCIAL_TENGRI,
        redshift=Fixed(0.0),
        n_grid=4096,
    )
    _st = _m.predict_state({})
    _lbt = np.asarray(_st.derived["sfh_grid_lbt_yr"])
    _sfr_t = np.asarray(_st.derived["sfr_history"])
    _t_t = (_AGE_2C_GYR - _lbt / 1e9) * 1e9
    _idx = np.argsort(_t_t)
    _label = {0: "exponential", 1: "delayed", 2: "rectangular"}[_bt]
    _cases_2c.append(
        (f"periodic {_label}", np.asarray(_t_c), np.asarray(_sfr_c), _t_t[_idx], _sfr_t[_idx])
    )

for _v in (150, 250):
    _t_c, _sfr_c = C.sfh_curve("sfh_buat08", velocity=_v, age=8000, normalise=True)
    _m = SEDModel.build(
        ssp_data=ssp,
        met=MET_FIDUCIAL,
        sfh={
            "type": "buat08",
            "velocity_km_s": Fixed(float(_v)),
            "age_gyr": Fixed(_AGE_2C_GYR),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=_DUST_OFF_2C,
        neb=NEB_FIDUCIAL_TENGRI,
        redshift=Fixed(0.0),
        n_grid=4096,
    )
    _st = _m.predict_state({})
    _lbt = np.asarray(_st.derived["sfh_grid_lbt_yr"])
    _sfr_t = np.asarray(_st.derived["sfr_history"])
    _t_t = (_AGE_2C_GYR - _lbt / 1e9) * 1e9
    _idx = np.argsort(_t_t)
    _cases_2c.append(
        (f"buat08 v={_v}", np.asarray(_t_c), np.asarray(_sfr_c), _t_t[_idx], _sfr_t[_idx])
    )

for _label, _wr, _lr, _wt, _lt in _cases_2c:
    _assert_comparable(_lr, _lt, name=f"§2c {_label}")

fig, (ax, ax_r), _ratios_2c = V.sweep_fig(
    _cases_2c,
    ref_label="CIGALE",
    title="§2c SFH families beyond delayed",
    xlim=(0, _AGE_2C_GYR),
    x_of_wave=lambda t: t / 1e9,
    xlabel="cosmic age since SF onset [Gyr]",
    ylabel=r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]",
    logy=False,
)
fig.tight_layout()
save_fig("cigale_02c_sfh_families.png")

_rows_2c = V.window_rows(
    _cases_2c, lo=0.02 * _AGE_2C_GYR * 1e9, hi=0.95 * _AGE_2C_GYR * 1e9, rel_to="peak"
)
V.print_window_table(
    _rows_2c,
    ref_name="CIGALE",
    title="§2c SFH families beyond delayed (SFR(t), 2-95% of age; deviation as % of peak SFR)",
    x_unit="Gyr",
    x_scale=1e-9,
)


# %% [markdown]
# ## §3 Integrated stellar SED
#
# CIGALE `sfhdelayed` + `bc03` + `nebular` against tengri `delayed` SFH on the
# shared SSP with the Setup nebular fiducial; no dust. The figure overlays the
# two intrinsic SEDs. CIGALE is normalized to 1 M☉ formed by construction; tengri's stellar
# mass formed is reported in the annotation.
#
# **The one convention difference on this SSP, and where it goes.** The two
# codes convolve the same star formation with the same SSP grid in different
# ways, and the rest of the notebook inherits the result, so it is stated once
# here.
#
# tengri computes the exact SFH ⊗ SSP convolution: each parcel of star formation
# is placed at its own age, and a cloud-in-cell kernel splits it between the two
# SSP nodes that bracket that age. CIGALE samples the SFR at integer Myr
# (`pcigale/sed_modules/sfhdelayed.py`) and pairs each 1 Myr bin with the SSP at
# the bin's older edge (`pcigale/sed_modules/bc03.py`). That ages every star by
# about 0.5 Myr and gives no weight to stars younger than 1 Myr. Q_H and the
# far-UV fall steeply over the first ~10 Myr, so 0.5 Myr is a large fraction of
# the young population's output and a negligible one of the old population's: the
# difference is strongly wavelength-dependent.
#
# The cell after the table tests this on Q_H. Convolving the SFH continuously
# with the BC03 n_ly(t) of §1 reproduces tengri's Q_H, and aging every star by
# 0.5 Myr before the same convolution reproduces CIGALE's. Measured on this pair
# as energy-weighted ratios (the ratio of the luminosity each code puts in the
# band; a median over grid points would weight a band by how many points CIGALE's
# nebular line grid places in it), printed below:
#
# | quantity | tengri / CIGALE |
# |---|---|
# | 912–1200 Å | 1.063× |
# | 1200–3000 Å | 1.030× |
# | 0.3–1 μm | 1.006× |
# | stellar L_bol | 1.016× |
# | Q_H (λ < 911.76 Å) | 1.150× |
#
# The 200–912 Å band is not quoted: with the nebular emitter on, both codes
# absorb nearly every Lyman-continuum photon, so there is nothing to take a
# ratio of. Neither convention is a defect: tengri's is the continuous
# convolution, CIGALE's a 1 Myr discrete sum with the bin's older-edge age.
#
# Three later sections are this table and nothing else. §6's dust IR is
# normalized to `L_absorbed`, which inherits the L_bol offset weighted toward
# the UV, and lands at 1.011× CIGALE's `dust.luminosity`. §11's synchrotron is
# anchored on the same luminosity, so that factor appears there as a flat
# offset. §8's line luminosities scale with Q_H. None of those is a separate
# discrepancy.

# %%
sed_c = C.run_chain(
    [
        (
            "sfhdelayed",
            dict(
                tau_main=1000,
                age_main=5000,
                tau_burst=50,
                age_burst=20,
                f_burst=0.0,
                sfr_A=1.0,
                normalise=True,
            ),
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        NEB_FIDUCIAL_CIGALE,
    ]
)
w_c, L_c = C.to_lnu(sed_c)

m_stellar = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
s_stellar = m_stellar.predict_state({})
_assert_comparable(L_c, s_stellar.sed_intrinsic, name="§3 stellar")

# Shared-axis overlay + tengri/CIGALE ratio panel: a single scale makes any
# normalization offset visible at a glance. Both codes form 1 M_sun; the
# tengri box reports its *surviving* stellar mass, so the ratio panel shows
# whether the M_star = 1.0 vs 0.558 label difference is a real SED offset (it is
# not — formed-vs-surviving-mass convention) or a genuine ~1.8x mismatch.
m_star = 10.0 ** float(s_stellar.derived["log_mstar"])
fig, ax, ax_r, ratio = U.overlay_ratio_fig(
    w_c,
    L_c,
    np.asarray(s_stellar.wave),
    np.asarray(s_stellar.sed_intrinsic),
    title="§3 stellar SED — CIGALE sfhdelayed+bc03 vs tengri (both 1 M$_\\odot$ formed)",
    label_c="CIGALE  sfhdelayed + bc03",
    label_t="tengri  sfh.delayed + bc03",
    xlim=(1e2, 1e6),
)
ax.text(
    0.05,
    0.95,
    rf"formed $M_\star = 1\,M_\odot$ both; tengri surviving $M_\star = {m_star:.3f}\,M_\odot$",
    transform=ax.transAxes,
    fontsize=9,
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
# The age-binning convention, band by band, as energy-weighted ratios: the
# ratio of the luminosity each code puts in the band. This is the table the §3
# prose quotes, and the term every later section inherits. A median over grid
# points is not used: CIGALE's nebular line grid carries many more points in the
# ultraviolet than in the optical, so a point-count median weights the bands by
# grid density.
_L_t_on_c3 = U.regrid(np.asarray(s_stellar.wave), np.asarray(s_stellar.sed_intrinsic), w_c)
print("§3 intrinsic stellar SED, energy-weighted ratio by band (∫L_ν dν, tengri / CIGALE):")
for _lo, _hi, _wname in (
    (912.0, 1200.0, "912–1200 Å "),
    (1200.0, 3000.0, "1200–3000 Å"),
    (3000.0, 10000.0, "0.3–1 µm   "),
    (1000.0, 10000.0, "0.1–1 µm   "),
):
    print(
        f"    {_wname}: "
        f"{_band_energy_ratio(w_c, L_c, np.asarray(s_stellar.wave), np.asarray(s_stellar.sed_intrinsic), _lo, _hi):.3f}×"
    )
_pos3 = L_c > 0
_lbol_c3 = _lbol_nu(w_c[_pos3], np.asarray(L_c)[_pos3])
_lbol_t3 = _lbol_nu(w_c[_pos3], _L_t_on_c3[_pos3])
_n_ly_c3 = float(sed_c.info["stellar.n_ly"])
_q_h_t3 = float(np.asarray(s_stellar.derived["nion"]))
print(
    f"    stellar L_bol: CIGALE {_lbol_c3:.4e}, tengri {_lbol_t3:.4e} erg/s "
    f"→ {_lbol_t3 / _lbol_c3:.3f}×"
)
print(
    f"    Q_H          : CIGALE {_n_ly_c3:.4e}, tengri {_q_h_t3:.4e} ph/s "
    f"→ {_q_h_t3 / _n_ly_c3:.3f}×"
)
fig.tight_layout()
plt.close(fig)  # superseded by §3b's tau x age grid, which replaces this panel


# %%
# The age convention, tested on the ionizing photon rate. The BC03 pickle read in
# §1 carries n_ly(t) on a 1 Myr age axis. Convolving the τ-delayed SFH with it
# continuously (every parcel at its own age, the SSP linear in age between nodes
# and held at its 1 Myr value below) is the exact SFH ⊗ SSP convolution tengri
# computes. Aging every star by 0.5 Myr first is the convention CIGALE's discrete
# 1 Myr sum implies: each 1 Myr bin is paired with the SSP at its older edge.
_t_ssp_myr = np.asarray(_raw.t, dtype=float)  # SSP ages [Myr], 1 Myr spacing
_q_ssp = np.asarray(_raw.info)[2]  # n_ly per M☉ formed [ph/s]


def _q_h_continuous(shift_myr: float, tau_myr=1000.0, age_myr=5000.0, d_age=0.01) -> float:
    """Q_H [ph/s per M☉ formed] of the τ-delayed SFH, every star aged by ``shift_myr``."""
    a = np.arange(0.5 * d_age, age_myr, d_age)  # lookback age of each parcel [Myr]
    sfr = (age_myr - a) * np.exp(-(age_myr - a) / tau_myr)
    sfr = sfr / (sfr.sum() * d_age)  # normalized to 1 M☉ formed
    return float(np.sum(sfr * np.interp(a + shift_myr, _t_ssp_myr, _q_ssp)) * d_age)


_q_exact_3 = _q_h_continuous(0.0)
_q_shift_3 = _q_h_continuous(0.5)
print(
    f"§3 Q_H, continuous SFH ⊗ SSP convolution: {_q_exact_3:.4e} ph/s "
    f"→ {_q_exact_3 / _q_h_t3:.3f}× tengri's {_q_h_t3:.4e}"
)
print(
    f"§3 Q_H, the same convolution with every star aged by 0.5 Myr: {_q_shift_3:.4e} ph/s "
    f"→ {_q_shift_3 / _n_ly_c3:.3f}× CIGALE's {_n_ly_c3:.4e}"
)
assert abs(_q_exact_3 / _q_h_t3 - 1.0) < 0.02, "continuous convolution should match tengri's Q_H"
assert abs(_q_shift_3 / _n_ly_c3 - 1.0) < 0.01, "a 0.5 Myr aging should reproduce CIGALE's Q_H"


# %% [markdown]
# ### §3b τ × age grid
#
# Stellar SED dependence on the delayed SFH's shape parameters: τ ∈
# {0.3, 1, 3} Gyr at age = 1 and 10 Gyr, plus the τ = 1 Gyr, age = 5 Gyr
# fiducial (7 cases). One tengri build per case with `sfh_delayed_tau_gyr` and
# `sfh_delayed_age_gyr` fixed (a free age is expressed as a fraction of the age of the
# universe, so absolute ages are fixed per build), evaluated via `predict_rest_sed`;
# one CIGALE `sfhdelayed` + `bc03` chain per case. Table over the
# GALEX-through-2MASS bands (`UV_TO_NIR`), since this comparison carries
# no dust IR. The worst band ratio in the printed table is τ = 3 Gyr, age = 1 Gyr
# (1.055×, SDSS u). The τ = 1 and 3 Gyr cases at age 1 Gyr
# have one band each outside 5 % (worst 1.052× and 1.055×, both SDSS u), where the
# youngest stars carry the §3 convention; the remaining five cases have none. Of
# those five, τ = 1 Gyr at 10 Gyr ties τ = 0.3 Gyr at 10 Gyr at 1.044×, and the worst
# band of the quiescent cases is in the GALEX far or near UV (printed per case). The
# quiescent cases sit near unity because Cue counts ionizing photons from every
# stellar age, so the old population supplies the nebular far-UV that CIGALE also
# includes.

# %%
_tau_age_cases_grid = [
    (0.3, 1.0),
    (1.0, 1.0),
    (3.0, 1.0),
    (0.3, 10.0),
    (1.0, 10.0),
    (3.0, 10.0),
    (1.0, 5.0),
]

def _build_3b(tau_gyr, age_gyr):
    """One fixed-timing tengri build per (tau, age) case."""
    return SEDModel.build(
        ssp_data=ssp,
        met=MET_FIDUCIAL,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(tau_gyr),
            "age_gyr": Fixed(age_gyr),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        neb=NEB_FIDUCIAL_TENGRI,
        redshift=Fixed(0.0),
    )


_cases_3b = []
for _tau, _age in _tau_age_cases_grid:
    _t_main, _a_main = int(round(_tau * 1000)), int(round(_age * 1000))
    _sed_c3b = C.run_chain(
        [
            (
                "sfhdelayed",
                dict(
                    tau_main=_t_main,
                    age_main=_a_main,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=1.0,
                    normalise=True,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            NEB_FIDUCIAL_CIGALE,
        ]
    )
    _w_c3b, _L_c3b = C.to_lnu(_sed_c3b)
    _m_3b = _build_3b(_tau, _age)
    _o_3b = _m_3b.predict_rest_sed(dict(_m_3b.spec.sample(jax.random.PRNGKey(0))))
    _cases_3b.append(
        (
            f"τ={_tau:g}, age={_age:g} Gyr",
            _w_c3b,
            _L_c3b,
            np.asarray(_o_3b.wavelength),
            np.asarray(_o_3b.sed),
        )
    )
    _assert_comparable(_L_c3b, np.asarray(_o_3b.sed), name=f"§3b tau={_tau} age={_age}")

fig, (ax, ax_r), _ratios_3b = V.sweep_fig(
    _cases_3b, ref_label="CIGALE", title="§3b τ × age grid", xlim=(1e2, 1e5)
)
fig.tight_layout()
save_fig("cigale_03b_tau_age_grid.png")

print("§3b τ × age grid (UV_TO_NIR band ratio per case):")
for _label, _w_ref, _L_ref, _w_t, _L_t in _cases_3b:
    _rows = V.filter_rows_native(
        np.asarray(_w_t), np.asarray(_L_t), _w_ref, _L_ref, filters=V.UV_TO_NIR
    )
    V.print_filter_table(
        _rows, ref_name="CIGALE", title=f"§3b {_label}", compact=True, show_worst_band=True
    )


# %% [markdown]
# ## §4 Dust attenuation curves
#
# CIGALE's three `dustatt_*` families against tengri's, each side evaluating
# its own analytic curve on one grid and normalized to `A(λ)/A_V` at 5500 Å.
# The pairings are by curve, not by module name:
#
# | CIGALE | tengri | shared continuum |
# |---|---|---|
# | `dustatt_calzleit` | `leitherer02` | Calzetti+2000 with the L02 far-UV extension |
# | `dustatt_modified_starburst`, `uv_bump_amplitude = 3` | `noll09(dust_bump_strength=3)` | the same, plus the 2175 Å Drude bump |
# | `dustatt_modified_CF00` | `power_law` twice, `slope_ISM = −0.7` + `slope_BC = −1.3` | (λ/5500 Å)^δ, two screens |
#
# `calzleit` is paired with `leitherer02`, not with tengri's `calzetti`:
# CIGALE's curve carries the Leitherer far-UV extension and tengri's bare
# `calzetti` does not, which is an 8.4 % gap below 1500 Å between two laws
# that are not the same law. `modified_CF00` needs two curves on each side:
# CIGALE attenuates young stars through the birth cloud *and* the ISM
# (`Av_BC = Av_ISM(1−μ)/μ = 1.53` at the defaults), so a single
# (λ/5500 Å)^−0.7 is the wrong object to compare it against.
#
# **The one real convention difference is the Leitherer↔Calzetti crossover.**
# CIGALE hands over at 1500 Å (`a_vs_ebv`), tengri at 1800 Å, which is the
# upper end of the 912–1800 Å range Leitherer et al. (2002) state for their
# fit. That is the whole of the printed max |Δ| below; away from
# 1500–1800 Å the pairs agree to 0.000 %. tengri is the reference here:
# pcigale's own `k_leitherer2002` docstring gives the range as 91.2–180 nm
# while `a_vs_ebv` truncates it at 150 nm.
#
# Below the Lyman limit the two diverge by construction: CIGALE's curves are
# zero there, tengri's polynomial continues unless
# `dust_attenuation={'lyman_cutoff': True}` clips it. Every applied-dust
# section below sets that flag; see §5.
#
# **Verification Status:** CROSSVAL: Attenuation law library

# %%
from tengri.dust import list_laws

_tengri_laws = list_laws(headline=False).to_dict("fn")  # {name: fn(wave_aa) -> k at tau_V=1}
wave_law = np.logspace(np.log10(1000.0), np.log10(30000.0), 2000)


def _norm_AV(wave, A):
    """A(λ) normalized to A_V at 5500 Å."""
    return A / A[np.argmin(np.abs(wave - 5500.0))]


# CIGALE's bump/slope arguments are the module defaults except where a pair
# exercises one of them: bump_wave and bump_width in nm, bump_ampl the
# Milky-Way 3.0 for the second pair, power_slope 0.
_BUMP_KW = dict(bump_wave=217.5, bump_width=35.0, bump_ampl=0.0, power_slope=0.0)

_A_ISM, _MU = 1.2, 0.44
_AV_BC = _A_ISM * (1.0 - _MU) / _MU  # CIGALE ModCF00Att._init_code

# Charlot & Fall: the curve a young star sees is both screens in series.
# CIGALE returns A(λ)/A_V per screen, so compose in magnitudes before
# normalizing; tengri's law functions return k(λ) with k(5500 Å) = 1, so the
# same composition is A_V-weighted there.
_cf00_c = _A_ISM * C.attenuation_curve(
    "dustatt_modified_CF00", wave_law, delta=-0.7
) + _AV_BC * C.attenuation_curve("dustatt_modified_CF00", wave_law, delta=-1.3)
_cf00_t = _A_ISM * np.asarray(
    _tengri_laws["power_law"](wave_law, dust_slope=-0.7)
) + _AV_BC * np.asarray(_tengri_laws["power_law"](wave_law, dust_slope=-1.3))

_law_pairs = [
    (
        "Calzetti + Leitherer far-UV",
        C.attenuation_curve("dustatt_calzleit", wave_law, **_BUMP_KW),
        np.asarray(_tengri_laws["leitherer02"](wave_law)),
        "calzleit ↔ leitherer02",
    ),
    (
        "  + 2175 Å bump (E_b = 3)",
        C.attenuation_curve(
            "dustatt_modified_starburst", wave_law, **{**_BUMP_KW, "bump_ampl": 3.0}
        ),
        np.asarray(_tengri_laws["noll09"](wave_law, dust_bump_strength=3.0)),
        "modified_starburst ↔ noll09",
    ),
    (
        "Charlot & Fall, two screens",
        _cf00_c,
        _cf00_t,
        "modified_CF00 ↔ power_law × 2",
    ),
]

# pcigale thick and translucent, tengri thin on top (the §6-knobs
# convention): agreement reads as a line down the middle of its own halo.
fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
print("§4 attenuation law parity (tengri / CIGALE, A_λ/A_V):")
for (label, A_c, A_t, pair), color in zip(_law_pairs, ("C0", "C1", "C3")):
    a_c, a_t = _norm_AV(wave_law, A_c), _norm_AV(wave_law, A_t)
    ax.plot(wave_law, a_c, color=color, lw=4.0, alpha=0.35, solid_capstyle="round")
    ax.plot(wave_law, a_t, color=color, lw=1.4, label=label)
    _r = a_t / np.where(a_c > 0, a_c, np.nan)
    ax_r.plot(wave_law, _r, color=color, lw=1.0)
    _cross = (wave_law >= 1500.0) & (wave_law <= 1800.0)
    print(
        f"  {pair:34s} max|Δ| {float(np.nanmax(np.abs(_r - 1.0))) * 100:6.3f}%  "
        f"median ratio {float(np.nanmedian(_r)):.5f}  "
        f"(outside 1500–1800 Å: max|Δ| "
        f"{float(np.nanmax(np.abs(_r[~_cross] - 1.0))) * 100:6.3f}%)"
    )

# The mispairing this section avoids, measured rather than asserted: tengri's
# bare ``calzetti`` against the same CIGALE calzleit curve, below 1500 Å.
_a_c_cz = _norm_AV(wave_law, C.attenuation_curve("dustatt_calzleit", wave_law, **_BUMP_KW))
_a_t_cz = _norm_AV(wave_law, np.asarray(_tengri_laws["calzetti"](wave_law)))
_m_fuv = wave_law < 1500.0
print(
    f"  for contrast, calzleit ↔ calzetti (no L02 extension) below 1500 Å: "
    f"max|Δ| {float(np.abs(_a_t_cz[_m_fuv] / _a_c_cz[_m_fuv] - 1.0).max()) * 100:.1f}%"
)

ax.plot([], [], "k-", lw=4.0, alpha=0.35, label="pcigale")
ax.plot([], [], "k-", lw=1.4, label="tengri")
ax.set(xscale="log", yscale="log", xlim=(1e3, 3e4), ylim=(0.05, 20))
ax.set_ylabel(r"$A_\lambda / A_V$")
ax.set_title("§4 attenuation laws — pcigale (band) vs tengri (line)")
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)
ax_r.axhspan(0.99, 1.01, color="0.85", zorder=0)
ax_r.axhline(1.0, color="0.5", lw=0.8)
ax_r.axvspan(1500.0, 1800.0, color="C2", alpha=0.12, zorder=0)
ax_r.set(xscale="log", ylim=(0.99, 1.01))
ax_r.set_xlabel(r"$\lambda$ [Å]")
ax_r.set_ylabel("tengri / CIGALE", fontsize=9)
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("cigale_04_dust_attenuation.png")
plt.show()


# %% [markdown]
# ## §5 Dust attenuation applied
#
# Fiducial galaxy with and without attenuation. CIGALE uses
# `modified_starburst` at E(B−V)_lines = 0.3; tengri uses the
# two-component Calzetti law at τ_BC and τ_diff derived from the
# same E(B−V)_lines via `cigale_ebv_lines_to_tau`.
#
# The printed energy-weighted ratios are the comparison. The printed maximum |Δ|
# over 0.1–1 μm is a single point in an emission-line wing and not a continuum
# difference. The point nearest the vacuum Lyα line (1215.67 Å) sits in a wing that
# the two codes draw differently: tengri renders Cue's Lyα as a flux-conserving
# triweight of σ = 100 km/s on the SSP wavelength grid and interpolates it onto
# CIGALE's grid, while CIGALE uses a Gaussian of FWHM 300 km/s on its own
# line-augmented grid. The cell therefore also prints the maximum outside ±10 Å of
# Lyα; the largest remaining points are on other emission-line wings, for example
# the Hα line wing (6563–6573 Å), and are set by the same difference in profile and sampling.
#
# **Verification Status:** PARTIAL (2/88): Two-component attenuation (birth cloud and diffuse)

# %%
sed_c_nodust = C.run_chain(
    [
        (
            "sfhdelayed",
            dict(
                tau_main=1000,
                age_main=5000,
                tau_burst=50,
                age_burst=20,
                f_burst=0.0,
                sfr_A=1.0,
                normalise=True,
            ),
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        NEB_FIDUCIAL_CIGALE,
    ]
)
w_c_nd, L_c_nd = C.to_lnu(sed_c_nodust)

sed_c_dust = C.run_chain(
    [
        (
            "sfhdelayed",
            dict(
                tau_main=1000,
                age_main=5000,
                tau_burst=50,
                age_burst=20,
                f_burst=0.0,
                sfr_A=1.0,
                normalise=True,
            ),
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        NEB_FIDUCIAL_CIGALE,
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
    ]
)
w_c_d, L_c_d = C.to_lnu(sed_c_dust)

m_nd = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
s_nd = m_nd.predict_state({})

m_d = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        # Match CIGALE's ``dustatt_modified_starburst``, which zeros its curve
        # below the Lyman limit (LyC photons ionize H rather than heat dust).
        # Without this tengri's leitherer02 polynomial extrapolates through the
        # FUV and over-attenuates λ < 912 Å relative to CIGALE.
        "lyman_cutoff": True,
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
s_d = m_d.predict_state({})
_assert_comparable(L_c_d, s_d.derived["sed_dust_attenuated"], name="§5 dust applied")

# The four panels are two comparisons — intrinsic and attenuated — and a
# side-by-side layout with independent y-axes cannot show a normalization
# offset in either. One number each.
_LYA_VAC_AA, _LYA_WINDOW_AA = 1215.67, 10.0  # vacuum Lyα [Å], half-width of the excluded window
print("§5 attenuation applied (tengri / CIGALE, on CIGALE's grid):")
for _name, _wc, _Lc, _wt, _Lt in (
    ("intrinsic ", w_c_nd, L_c_nd, np.asarray(s_nd.wave), np.asarray(s_nd.sed_intrinsic)),
    (
        "attenuated",
        w_c_d,
        L_c_d,
        np.asarray(s_d.wave),
        np.asarray(s_d.derived["sed_dust_attenuated"]),
    ),
):
    _Lt_on_c = U.regrid(_wt, _Lt, _wc)
    _r5 = _Lt_on_c / np.where(_Lc > 0, _Lc, np.nan)
    for _lo, _hi, _wname in ((912.0, 1200.0, "912–1200 Å"), (1000.0, 10000.0, "0.1–1 µm  ")):
        _m5 = (_wc >= _lo) & (_wc <= _hi) & np.isfinite(_r5)
        _d5 = np.abs(_r5[_m5] - 1.0)
        # The band luminosity ratio, and where the worst point is: the extremum
        # is a single point in an emission-line wing, where the two codes draw the
        # line with different profiles (a 100 km/s triweight on the SSP grid in
        # tengri, a 300 km/s FWHM Gaussian on CIGALE's line grid) and different
        # sampling. The maximum is therefore printed twice, over the whole band
        # and outside a ±10 Å window around the vacuum Lyα line.
        _m5_out = _m5 & (np.abs(_wc - _LYA_VAC_AA) > _LYA_WINDOW_AA)
        _d5_out = np.abs(_r5[_m5_out] - 1.0)
        print(
            f"    {_name} {_wname}: energy-weighted "
            f"{_band_energy_ratio(_wc, _Lc, _wt, _Lt, _lo, _hi):.4f}×, "
            f"max |Δ| {float(_d5.max()) * 100:.2f}% at "
            f"{float(_wc[_m5][int(np.argmax(_d5))]):.0f} Å; "
            f"outside Lyα ±{_LYA_WINDOW_AA:g} Å: {float(_d5_out.max()) * 100:.2f}% at "
            f"{float(_wc[_m5_out][int(np.argmax(_d5_out))]):.0f} Å"
        )

fig, ((ax_l1, ax_r1), (ax_l2, ax_r2)) = plt.subplots(2, 2, sharey=True, figsize=(12, 8))
U.panel(ax_l1, ax_r1, label_l="pcigale  intrinsic", label_r="tengri  intrinsic")
U.panel(
    ax_l2,
    ax_r2,
    label_l="pcigale  modified_starburst  (E(B−V)_lines = 0.3)",
    label_r=rf"tengri  two-component Calzetti  ($\tau_{{BC}}$={TAU_BC_FIDUCIAL:.2f}, "
    rf"$\tau_{{diff}}$={TAU_DIFF_FIDUCIAL:.2f})",
)
ax_l1.plot(w_c_nd, L_c_nd, "C0-", linewidth=1.5)
ax_r1.plot(s_nd.wave, s_nd.sed_intrinsic, "C1-", linewidth=1.5)
ax_l2.plot(w_c_d, L_c_d, "C0-", linewidth=1.5)
ax_r2.plot(s_d.wave, s_d.derived["sed_dust_attenuated"], "C1-", linewidth=1.5)
_ymax = float(np.asarray(s_nd.sed_intrinsic).max())
for ax in (ax_l1, ax_r1, ax_l2, ax_r2):
    ax.set_ylim(_ymax * 1e-6, _ymax * 2)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.close(fig)  # superseded by §5b's attenuation-knob grids, which replace this panel


# %% [markdown]
# ### §5b Attenuation knobs
#
# Two more attenuation modules. `dustatt_modified_starburst` at
# E(B−V)_lines ∈ {0.1, 0.3, 0.6} × `uv_bump_amplitude` ∈ {0, 3} (6 cases)
# → tengri `two_component`, law `noll09` on both screens,
# `tau_diff` from the Setup E→τ conversion, `tau_bc = 0`,
# `bump_strength_diff` matching the bump. `dustatt_2powerlaws`
# (Av_BC=1.0, slope_BC=−1.3, BC_to_ISM_factor=0.44) at
# slope_ISM ∈ {−0.4, −0.7, −1.0} (3 cases) → tengri `power_law` on both
# screens, `tau_bc = 1.0/1.086`, `tau_diff = (1.0/0.44)/1.086` (CIGALE sets
# Av_ISM = Av_BC / BC_to_ISM_factor, not the product). Both attenuated
# SEDs against the `UV_TO_NIR` bands; both stand-alone A(λ)/A_V curves
# against `C.attenuation_curve` over 1216–3000 Å. This fiducial galaxy is
# old-star dominated, so the ISM screen carries almost all the attenuation and
# the `2powerlaws` ratio does not move with `slope_ISM`.

# %%
_cases_5b_noll = []
for _ebv, _bump in [(0.1, 0.0), (0.3, 0.0), (0.6, 0.0), (0.1, 3.0), (0.3, 3.0), (0.6, 3.0)]:
    _tau_d = _R_V_CALZETTI * _F_CONT_OVER_LINES * _ebv / 1.086
    _sed_cn = C.run_chain(
        [
            (
                "sfhdelayed",
                dict(
                    tau_main=1000,
                    age_main=5000,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=1.0,
                    normalise=True,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            NEB_FIDUCIAL_CIGALE,
            ("dustatt_modified_starburst", dict(E_BV_lines=_ebv, uv_bump_amplitude=_bump)),
        ]
    )
    _w_cn, _L_cn = C.to_lnu(_sed_cn)
    _m_n = SEDModel.build(
        ssp_data=ssp,
        met=MET_FIDUCIAL,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "noll09",
            "law_diff": "noll09",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(_tau_d),
            "bump_strength_diff": _bump,
            "lyman_cutoff": True,
            "all_params": Fixed(DEFAULT),
        },
        neb=NEB_FIDUCIAL_TENGRI,
        redshift=Fixed(0.0),
    )
    _s_n = _m_n.predict_state({})
    _cases_5b_noll.append(
        (
            f"E(B-V)={_ebv}, bump={_bump:g}",
            _w_cn,
            _L_cn,
            np.asarray(_s_n.wave),
            np.asarray(_s_n.derived["sed_dust_attenuated"]),
        )
    )
    _assert_comparable(
        _L_cn, np.asarray(_s_n.derived["sed_dust_attenuated"]), name=f"§5b noll09 {_ebv},{_bump}"
    )

fig, (ax, ax_r), _ratios_5bn = V.sweep_fig(
    _cases_5b_noll,
    ref_label="CIGALE",
    title="§5b modified_starburst E(B-V) × bump grid",
    xlim=(1e3, 1e4),
)
fig.tight_layout()
save_fig("cigale_05b_noll09_grid.png")
print("§5b modified_starburst E(B-V) × bump grid (UV_TO_NIR):")
for _label, _w_ref, _L_ref, _w_t, _L_t in _cases_5b_noll:
    _rows = V.filter_rows_native(
        np.asarray(_w_t), np.asarray(_L_t), _w_ref, _L_ref, filters=V.UV_TO_NIR
    )
    V.print_filter_table(_rows, ref_name="CIGALE", title=f"§5b {_label}", compact=True)

_cases_5b_2pl = []
for _slope_ism in (-0.4, -0.7, -1.0):
    _sed_c2p = C.run_chain(
        [
            (
                "sfhdelayed",
                dict(
                    tau_main=1000,
                    age_main=5000,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=1.0,
                    normalise=True,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            NEB_FIDUCIAL_CIGALE,
            (
                "dustatt_2powerlaws",
                dict(Av_BC=1.0, slope_BC=-1.3, BC_to_ISM_factor=0.44, slope_ISM=_slope_ism),
            ),
        ]
    )
    _w_c2p, _L_c2p = C.to_lnu(_sed_c2p)
    _m_2p = SEDModel.build(
        ssp_data=ssp,
        met=MET_FIDUCIAL,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "power_law",
            "law_diff": "power_law",
            "slope_bc": -1.3,
            "slope_diff": _slope_ism,
            # CIGALE's dustatt_2powerlaws sets Av_ISM = Av_BC / BC_to_ISM_factor
            # (dustatt_2powerlaws.py _init_code), not Av_BC * factor despite the
            # parameter's "Av ISM / Av BC (<1)" docstring -- verified against the
            # installed module directly.
            "tau_bc": Fixed(1.0 / 1.086),
            "tau_diff": Fixed((1.0 / 0.44) / 1.086),
            "lyman_cutoff": True,
            "all_params": Fixed(DEFAULT),
        },
        neb=NEB_FIDUCIAL_TENGRI,
        redshift=Fixed(0.0),
    )
    _s_2p = _m_2p.predict_state({})
    _cases_5b_2pl.append(
        (
            f"slope_ISM={_slope_ism}",
            _w_c2p,
            _L_c2p,
            np.asarray(_s_2p.wave),
            np.asarray(_s_2p.derived["sed_dust_attenuated"]),
        )
    )
    _assert_comparable(
        _L_c2p,
        np.asarray(_s_2p.derived["sed_dust_attenuated"]),
        name=f"§5b 2powerlaws {_slope_ism}",
    )

fig, (ax, ax_r), _ratios_5b2 = V.sweep_fig(
    _cases_5b_2pl,
    ref_label="CIGALE",
    title="§5b 2powerlaws slope_ISM sweep",
    xlim=(1e3, 1e4),
    cmap="Oranges",
    values=[-0.4, -0.7, -1.0],
    param_label="slope_ISM",
)
fig.tight_layout()
save_fig("cigale_05c_2powerlaws_slope.png")
print("§5b 2powerlaws slope_ISM sweep (UV_TO_NIR):")
for _label, _w_ref, _L_ref, _w_t, _L_t in _cases_5b_2pl:
    _rows = V.filter_rows_native(
        np.asarray(_w_t), np.asarray(_L_t), _w_ref, _L_ref, filters=V.UV_TO_NIR
    )
    V.print_filter_table(_rows, ref_name="CIGALE", title=f"§5b {_label}", compact=True)

# Stand-alone A(λ)/A_V curves (no SFH/SSP, no E(B-V) scale — the ratio is
# shape-only), the same way §4 reads each law's own analytic function.
_wave_5b = np.logspace(np.log10(1000.0), np.log10(30000.0), 2000)
_cases_5b_curve = []
for _bump in (0.0, 3.0):
    _A_c = _norm_AV(
        _wave_5b,
        C.attenuation_curve(
            "dustatt_modified_starburst",
            _wave_5b,
            bump_wave=217.5,
            bump_width=35.0,
            bump_ampl=_bump,
            power_slope=0.0,
        ),
    )
    _A_t = _norm_AV(
        _wave_5b, np.asarray(_tengri_laws["noll09"](_wave_5b, dust_bump_strength=_bump))
    )
    _cases_5b_curve.append((f"bump={_bump:g}", _wave_5b, _A_c, _wave_5b, _A_t))
_rows_5b_curve = V.window_rows(_cases_5b_curve, lo=1216.0, hi=3000.0)
V.print_window_table(
    _rows_5b_curve, ref_name="CIGALE", title="§5b A(λ)/A_V, modified_starburst screen, 1216-3000 Å"
)


# %% [markdown]
# ## §6 Dust IR re-emission and energy balance
#
# CIGALE re-emits absorbed stellar UV/optical through the Dale et al. (2014)
# template family (α = 2); tengri evaluates the same templates and enforces
# energy balance, $L_{\rm IR,\,emitted} \equiv L_{\rm absorbed}$, to
# floating point: the residual is annotated on the right panel and printed
# below, and it is exactly zero.
#
# **The energy anchor.** tengri's `L_absorbed` sits 1.1 % above CIGALE's
# `dust.luminosity` (printed below). That is §3's age convention arriving here:
# the absorbed luminosity is an integral over the attenuated far-UV and optical,
# weighted toward the wavelengths where CIGALE's 0.5 Myr older stars emit less
# than tengri's. The two codes' attenuation *curves* are not the cause:
# §4 measures them agreeing to 0.000 % away from the 1500–1800 Å crossover,
# and the absorbed fractions they produce are printed below (0.359 for CIGALE, 0.357 for tengri).
# Since the IR is normalized to that anchor, it appears in the printed
# 10–100 μm median as a floor under whatever the templates themselves do.
#
# **Lyman continuum.** tengri's attenuation curves polynomial-extend
# through the FUV; CIGALE zeros attenuation below 912 Å, and the models here
# set `dust_attenuation={'lyman_cutoff': True}` to match. This changes the
# emergent far-UV continuum, which is what §7's panel shows, and *not* the IR
# budget: `L_absorbed` masks λ < 912 Å unconditionally, on the reasoning that
# Lyman-continuum photons ionize hydrogen rather than heat dust. The IR here
# is bit-identical with and without the flag.
#
# **Verification Status:** CROSSVAL: Dust IR emission versus BAGPIPES

# %%
sed_c_ir = C.run_chain(
    [
        (
            "sfhdelayed",
            dict(
                tau_main=1000,
                age_main=5000,
                tau_burst=50,
                age_burst=20,
                f_burst=0.0,
                sfr_A=1.0,
                normalise=True,
            ),
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        NEB_FIDUCIAL_CIGALE,
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("dale2014", dict(alpha=2.0)),
    ]
)
w_c_ir, L_c_ir = C.to_lnu(sed_c_ir)

m_ir = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        # Lyman-limit clip (CIGALE parity) — see §5. The dust IR is energy-balance
        # normalized to L_absorbed, whose integral already excludes λ < 912 Å, so
        # this only changes the emergent FUV continuum, not the IR budget.
        "lyman_cutoff": True,
        "all_params": Fixed(DEFAULT),
    },
    dust_emission={
        "type": "dale2014_cigale",
        "alpha_dale": Fixed(2.0),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
s_ir = m_ir.predict_state({})
L_abs = float(s_ir.derived.get("L_absorbed", 0.0))
L_emit = float(s_ir.derived.get("L_ir", 0.0))
residual = abs(L_abs - L_emit) / max(L_abs, 1e-30)
_assert_comparable(L_c_ir, s_ir.sed_intrinsic, name="§6 IR")

# Shared-axis overlay + ratio panel: the FIR-peak partition difference is
# only readable on one scale.
fig, ax, ax_r, ratio = U.overlay_ratio_fig(
    w_c_ir,
    L_c_ir,
    np.asarray(s_ir.wave),
    np.asarray(s_ir.sed_intrinsic),
    title="§6 stellar + Dale+2014 dust IR — CIGALE vs tengri",
    label_c="CIGALE  + Dale+2014 (α = 2)",
    label_t="tengri  + dust.emission.dale2014 (α = 2)",
    xlim=(1e3, 5e6),
    dyn_range=1e-4,
)
ax.text(
    0.98,
    0.05,
    rf"tengri $|L_{{\rm IR}} - L_{{\rm abs}}|/L_{{\rm abs}} = {residual:.1e}$",
    transform=ax.transAxes,
    fontsize=9,
    ha="right",
    va="bottom",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
)
print(
    "§6 dust IR tengri/CIGALE energy-weighted (10–100 µm): "
    f"{_band_energy_ratio(w_c_ir, L_c_ir, np.asarray(s_ir.wave), np.asarray(s_ir.sed_intrinsic), 1e5, 1e6):.3f}×"
)

# The energy anchor the IR is normalized to, and the absorbed fraction that
# sets it. Printed because §6's IR ratio, §11's synchrotron amplitude and the
# capstone's FIR all rest on this one number, and it is not the attenuation
# curve (§4) — it is the §3 age-binning convention reaching the integral.
_L_DUST_C = float(sed_c_ir.info["dust.luminosity"]) * 1e7
_L_STAR_C = float(sed_c_ir.info["stellar.lum"]) * 1e7
# §3's dust-free twin is this galaxy's intrinsic stellar SED — same SFH, same
# SSP, same 1 M☉ — so its bolometric integral is the denominator CIGALE's
# ``stellar.lum`` is.
_L_star_t = _lbol_nu(np.asarray(s_stellar.wave), np.asarray(s_stellar.sed_intrinsic))
print(
    f"§6 energy anchor: CIGALE dust.luminosity {_L_DUST_C:.4e}, "
    f"tengri L_absorbed {L_abs:.4e} erg/s → {L_abs / _L_DUST_C:.4f}×"
)
print(f"§6 absorbed fraction: CIGALE {_L_DUST_C / _L_STAR_C:.6f}, tengri {L_abs / _L_star_t:.6f}")
print(f"§6 energy balance |L_IR − L_abs| / L_abs = {residual:.3e}")
fig.tight_layout()
plt.close(fig)  # superseded by §6c's IR-library grids, which replace this panel


# %% [markdown]
# ### Dust-IR model knobs: AGN heating and the radiation-field slope
#
# Two further CIGALE dust-IR parameters. Both panels sit on the same
# energy anchor as §6: `_knob_model` uses the notebook's fiducial
# attenuation block, a single screen (`tau_bc = 0`, matching
# `dustatt_modified_starburst`, which has no Charlot & Fall birth cloud;
# A_V = R_V × E(B−V)_cont = 4.05 × 0.132 = 0.535 mag) carrying the
# Leitherer-extended Calzetti curve with the 912 Å clip. So the 1.1 % anchor
# offset of §3/§6 enters every ratio printed here, and what the panels add is
# whatever the *templates* do on top of it.
#
# **Left: Dale 2014 AGN fraction (`dale2014.fracAGN`).** `fracAGN` adds an
# AGN-heated source as a separate power budget ($L_{\rm AGN}=L_{\rm
# dust}\,f/(1-f)$) using CIGALE's own `model_quasar` template
# ($SED = L\,T_{\rm SF}(\alpha) + L_{\rm AGN}\,T_{\rm QSO}$, `dale2014.py`).
# The subtlety that makes the mixing work: CIGALE normalizes `model_quasar`
# to unit luminosity over its full native grid (~60 nm onward), where ~46 %
# of the quasar energy is the UV/optical accretion-disc continuum below the
# dust grid's blue edge. Only its ~0.42 IR share enters the dust mixing, and
# tengri carries that same partition. The cell prints the 3–8 μm and
# 8–1000 μm median ratio at each $f_{\rm AGN}$, so the question of whether
# the mid-IR lift drifts with $f_{\rm AGN}$ is answered by three numbers
# rather than by eye.
#
# **Right: THEMIS slope $\alpha$ (`themis.alpha`, $dU/dM \propto
# U^{-\alpha}$, matched `qhac=0.17, umin=1.0, gamma=0.1`).** tengri's THEMIS
# templates are built from the published DustEM grids (Jones+2017) and
# conserve the absorbed energy. The a-C(:H) aromatic fraction `qhac` is
# pinned to CIGALE's 0.17 on both sides. CIGALE quotes it as a fraction
# while the DustEM grid tabulates it in FSPS scaling (`qhac × 100/2.2`), so
# the two must be reconciled before interpolation or the wrong grain model
# is selected. One input is *not* matched and cannot be: CIGALE's own DustEM
# run used `umax = 1e7`, a slightly hotter PDR. The cell prints both an
# 8–30 μm and an 8–1000 μm ratio at each α, so a redistribution of the IR
# within the band can be told apart from a change in its total.

# %%
import jax
import jax.numpy as jnp

_c_aa_dust = C_AA

# Matched fiducial chain (mirrors the §6 dale2014 cell): same SFH + BC03 +
# Calzetti attenuation on both sides, so the absorbed luminosity that feeds
# the IR re-emission template is identical. pcigale's normalise=True forms
# 1 M_sun; tengri's log_total_mass=11 forms 1e11 M_sun, so scale the pcigale
# curve by 1e11 to overlay the two at the same stellar mass.
_KNOB_MASS = 1e11
_SFH_CHAIN = (
    "sfhdelayed",
    dict(
        tau_main=1000,
        age_main=5000,
        tau_burst=50,
        age_burst=20,
        f_burst=0.0,
        sfr_A=1.0,
        normalise=True,
    ),
)
_BC03_CHAIN = ("bc03", dict(imf=1, metallicity=0.02, separation_age=10))
_DUSTATT_CHAIN = ("dustatt_modified_starburst", dict(E_BV_lines=0.3))


def _knob_model(emission_type, **emkw):
    """tengri fiducial twin of the matched pcigale chain (1e11 M_sun).

    The attenuation is the notebook's fiducial: a **single** screen
    (``tau_bc = 0``, matching CIGALE's ``dustatt_modified_starburst``, which
    has no Charlot & Fall birth cloud) carrying the Leitherer-extended
    Calzetti curve with the 912 Å clip, i.e. exactly the block §5–§11 use.
    That matters here rather than being housekeeping: the Dale and THEMIS
    templates are normalized to the *absorbed* starlight, so the IR can only
    match to the extent the attenuation does, and a different curve or a
    missing clip would show up as an IR offset with no dust-side cause.
    """
    return SEDModel.build(
        ssp_data=ssp,
        met=MET_FIDUCIAL,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(11.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "leitherer02",
            "law_diff": "leitherer02",
            "tau_bc": Fixed(TAU_BC_FIDUCIAL),
            "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
            "lyman_cutoff": True,
            "all_params": Fixed(DEFAULT),
        },
        dust_emission=(
            {"type": "none"}
            if emission_type == "none"
            else {"type": emission_type, "all_params": Fixed(DEFAULT), **emkw}
        ),
        neb=NEB_FIDUCIAL_TENGRI,
        redshift=Fixed(0.0),
    )


def _nu_lnu(wave_aa, l_nu):
    w = np.asarray(wave_aa)
    return w, _c_aa_dust / w * np.asarray(l_nu)


def _band_median(ratio, wave_aa, lo_um, hi_um):
    """Median of ``ratio`` over ``[lo_um, hi_um]`` on the Å grid ``wave_aa``."""
    m = (wave_aa >= lo_um * 1e4) & (wave_aa <= hi_um * 1e4) & np.isfinite(ratio)
    return float(np.median(ratio[m])) if m.any() else float("nan")


fig = plt.figure(figsize=(11, 6.5))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3)
ax_l, ax_r = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_lr, ax_rr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
ax_lr.sharex(ax_l)
ax_rr.sharex(ax_r)

# Where the two codes agree, a thin pcigale line simply disappears under
# tengri's — the panel then reads as though only one code were plotted, which
# is the opposite of the point. Draw pcigale as a thick translucent band and
# tengri as a thin line on top (the convention the AGNFITTER-RX notebook uses):
# agreement shows as a crisp line running down the middle of its own halo, and
# any divergence separates the two immediately.
_REF_KW = dict(lw=4.0, alpha=0.35, solid_capstyle="round")
_TNG_LW = 1.4
_peaks = []
_ratios_l, _ratios_r = [], []

# LEFT — Dale 2014 AGN fraction: pcigale dale2014.fracAGN (band) vs tengri (line).
m_frac = _knob_model(
    "dale2014_cigale", alpha_dale=Fixed(2.0), frac_agn=Uniform(0.0, 0.99, default=0.3)
)
p_frac = dict(m_frac.spec.sample(jax.random.PRNGKey(0)))
print("§6 knobs — Dale 2014 fracAGN sweep (tengri / CIGALE, median in band):")
for f, c in zip([0.0, 0.3, 0.6], ["C0", "C1", "C3"]):
    sed = C.run_chain(
        [
            _SFH_CHAIN,
            _BC03_CHAIN,
            NEB_FIDUCIAL_CIGALE,
            _DUSTATT_CHAIN,
            ("dale2014", dict(alpha=2.0, fracAGN=f)),
        ]
    )
    w_c, nl_c = _nu_lnu(*C.to_lnu(sed))
    ax_l.loglog(w_c, nl_c * _KNOB_MASS, color=c, **_REF_KW)
    o = m_frac.predict_rest_sed({**p_frac, "dust_frac_agn": jnp.float64(f)})
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_l.loglog(w_t, nl_t, color=c, lw=_TNG_LW, label=rf"$f_{{\rm AGN}}={f}$")
    _peaks.append(float(np.nanmax(nl_t)))
    _rk = U.regrid(w_t, nl_t, w_c) / np.where(nl_c > 0, nl_c * _KNOB_MASS, np.nan)
    _ratios_l.append((c, w_c, _rk))
    print(
        f"    f_AGN = {f}:  3–8 µm {_band_median(_rk, w_c, 3.0, 8.0):.4f}×"
        f"   8–1000 µm {_band_median(_rk, w_c, 8.0, 1000.0):.4f}×"
    )
ax_l.plot([], [], "k-", **_REF_KW, label="pcigale")
ax_l.plot([], [], "k-", lw=_TNG_LW, label="tengri")
ax_l.set(
    xlim=(1e4, 1e7),
    ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]",
    title="Dale 2014 AGN fraction",
)
ax_l.legend(fontsize=8, frameon=False, ncol=2)
ax_l.tick_params(labelbottom=False)

# RIGHT — THEMIS slope alpha, matched qhac=0.17, umin=1.0, gamma=0.1.
m_alpha = _knob_model(
    "themis", gamma_dl=Fixed(0.1), qhac=Fixed(0.17), alpha=Uniform(1.0, 3.0, default=2.0)
)
p_alpha = dict(m_alpha.spec.sample(jax.random.PRNGKey(0)))
print("§6 knobs — THEMIS α sweep (tengri / CIGALE, median in band):")
for a, c in zip([1.0, 2.0, 3.0], ["C0", "C1", "C3"]):
    sed = C.run_chain(
        [
            _SFH_CHAIN,
            _BC03_CHAIN,
            NEB_FIDUCIAL_CIGALE,
            _DUSTATT_CHAIN,
            ("themis", dict(qhac=0.17, umin=1.0, gamma=0.1, alpha=a)),
        ]
    )
    w_c, nl_c = _nu_lnu(*C.to_lnu(sed))
    ax_r.loglog(w_c, nl_c * _KNOB_MASS, color=c, **_REF_KW)
    o = m_alpha.predict_rest_sed({**p_alpha, "dust_alpha": jnp.float64(a)})
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_r.loglog(w_t, nl_t, color=c, lw=_TNG_LW, label=rf"$\alpha={a}$")
    _peaks.append(float(np.nanmax(nl_t)))
    _rk = U.regrid(w_t, nl_t, w_c) / np.where(nl_c > 0, nl_c * _KNOB_MASS, np.nan)
    _ratios_r.append((c, w_c, _rk))
    print(
        f"    α = {a}:  8–30 µm {_band_median(_rk, w_c, 8.0, 30.0):.4f}×"
        f"   8–1000 µm {_band_median(_rk, w_c, 8.0, 1000.0):.4f}×"
    )
ax_r.plot([], [], "k-", **_REF_KW, label="pcigale")
ax_r.plot([], [], "k-", lw=_TNG_LW, label="tengri")
ax_r.set(xlim=(3e4, 1e7), title=r"THEMIS radiation-field slope $\alpha$")
ax_r.legend(fontsize=8, frameon=False, ncol=2)
ax_r.tick_params(labelbottom=False)

# Ratio panels
for c, w, ratio in _ratios_l:
    pos = w > 0
    ax_lr.plot(w[pos], ratio[pos], color=c, linewidth=1.0)
ax_lr.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_lr.axhline(1.0, color="0.5", linewidth=0.8)
ax_lr.set(
    xlim=(1e4, 1e7), ylim=(0.5, 1.5), yscale="log", xlabel=r"$\lambda$ [Å]", ylabel="tengri/CIGALE"
)
ax_lr.grid(True, alpha=0.3)

for c, w, ratio in _ratios_r:
    pos = w > 0
    ax_rr.plot(w[pos], ratio[pos], color=c, linewidth=1.0)
ax_rr.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_rr.axhline(1.0, color="0.5", linewidth=0.8)
ax_rr.set(
    xlim=(3e4, 1e7), ylim=(0.5, 1.5), yscale="log", xlabel=r"$\lambda$ [Å]", ylabel="tengri/CIGALE"
)
ax_rr.grid(True, alpha=0.3)

_ypk = max(_peaks)
for ax in (ax_l, ax_r):
    ax.set_ylim(_ypk * 1e-5, _ypk * 2.0)

fig.tight_layout()
save_fig("cigale_06_dust_ir_knobs.png")
plt.show()


# %% [markdown]
# ### §6c IR library sweep
#
# Three more dust-IR template families, on the §6-knobs fiducial (1e11 M☉,
# single-screen Calzetti attenuation; the §3 energy-anchor offset sits under
# every ratio). `dl2007` (qpah, umin, γ) at three grid points, cold to warm;
# `dl2014` at α ∈ {1, 2, 3}; `casey2012` at T ∈ {25, 35, 50} K (β=1.6,
# α_mir=2.0); `schreiber2016` at T ∈ {25, 35, 50} K;
# `dale2014` (CIGALE's own grid, via `dale2014_cigale`) at α ∈ {0.5, 2, 4}. One tengri build per
# family, swept knob(s) free, via `predict_rest_sed`. `IR_BANDS` rows plus
# the 8–1000 μm L_ν ratio locate each family's shape. Both implementations are normalized
# on the same absorbed stellar energy (`L_absorbed`), and the 8–1000 μm L_ν ratios printed
# below lie between 1.0109× and 1.0141× across the sweep. The filter-by-filter tables show
# where shapes differ: no row has a band outside 5 % (`f_pah` matches CIGALE's 0.05
# default). `dl2014` is compared as dust emission alone: at α = 1 the dust is only
# part of the total beyond 250 μm, and the rest is the nebular free-free continuum, so
# a total-SED ratio there measures the nebular continuum and not the dust. Each code's
# twin without dust emission, at the same attenuation, is subtracted from its total;
# the dust-only 8–1000 μm medians are 1.0119×, 1.0116× and 1.0116× for α = 1, 2 and 3,
# with worst filter ratios 1.012×, 0.985× and 0.975×.
# `schreiber2016` is the tabulated Schreiber et al. (2018) library as
# CIGALE packages it: per unit dust mass, a dust-continuum and a PAH template at
# each temperature (the library tabulates 15–99 K; pcigale accepts 15–60 K), mixed
# by the PAH mass fraction `f_pah` as pcigale does and normalized to the absorbed
# energy. At T = 25, 35 and 50 K
# its 8–1000 μm ratio is 1.0115× each time, and its filter-by-filter median is 1.011×
# each time, with worst cases of 1.020×, 1.020× and 1.037×.

# %%
_peaks_6c = []
fig = plt.figure(figsize=(11, 6.5))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.3)
ax_l, ax_r = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_lr, ax_rr = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
ax_lr.sharex(ax_l)
ax_rr.sharex(ax_r)
ax_rr.sharey(ax_lr)

print("§6c dl2007 grid (tengri / CIGALE, median in 8-1000 µm band):")
m_dl07 = _knob_model(
    "draine_li2007",
    qpah=Uniform(0.47, 4.58, default=2.5),
    umin=Uniform(0.1, 10.0, default=1.0),
    gamma_dl=Uniform(0.01, 0.5, default=0.1),
)
p_dl07 = dict(m_dl07.spec.sample(jax.random.PRNGKey(0)))
_dl07_cases = []
_ratios_6c_l, _ratios_6c_r = [], []
for (_qpah, _umin, _gamma), _c in zip(
    [(0.47, 0.1, 0.01), (2.5, 1.0, 0.1), (4.58, 10.0, 0.5)], ["C0", "C1", "C3"]
):
    sed = C.run_chain(
        [
            _SFH_CHAIN,
            _BC03_CHAIN,
            NEB_FIDUCIAL_CIGALE,
            _DUSTATT_CHAIN,
            ("dl2007", dict(qpah=_qpah, umin=_umin, gamma=_gamma)),
        ]
    )
    _w_c_raw, _L_c_raw = C.to_lnu(sed)
    w_c, nl_c = _nu_lnu(_w_c_raw, _L_c_raw)
    ax_l.loglog(w_c, nl_c * _KNOB_MASS, color=_c, **_REF_KW)
    o = m_dl07.predict_rest_sed(
        {
            **p_dl07,
            "dust_qpah": jnp.float64(_qpah),
            "dust_umin": jnp.float64(_umin),
            "dust_gamma_dl": jnp.float64(_gamma),
        }
    )
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_l.loglog(w_t, nl_t, color=_c, lw=_TNG_LW, label=rf"qpah={_qpah:g}, umin={_umin:g}")
    _peaks_6c.append(float(np.nanmax(nl_t)))
    _rk = U.regrid(w_t, nl_t, w_c) / np.where(nl_c > 0, nl_c * _KNOB_MASS, np.nan)
    _ratios_6c_l.append((_c, w_c, _rk))
    print(
        f"    qpah={_qpah:g} umin={_umin:g} gamma={_gamma:g}: 8-1000 µm {_band_median(_rk, w_c, 8.0, 1000.0):.4f}×"
    )
    _dl07_cases.append(
        (
            f"dl2007 qpah={_qpah:g}",
            _w_c_raw,
            _L_c_raw * _KNOB_MASS,
            np.asarray(o.wavelength),
            np.asarray(o.sed),
        )
    )
ax_l.plot([], [], "k-", **_REF_KW, label="pcigale")
ax_l.plot([], [], "k-", lw=_TNG_LW, label="tengri")
ax_l.set(xlim=(1e4, 1e7), ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]", title="Draine & Li 2007")
ax_l.legend(fontsize=8, frameon=False)
ax_l.tick_params(labelbottom=False)

print("§6c dl2014 grid (dust emission only, tengri / CIGALE, median in 8-1000 µm band):")
# Both codes add the dust emission to a stellar + nebular + attenuation baseline,
# and at α = 1 the dust is a minor part of the total beyond 250 µm, where the
# nebular free-free continuum carries the rest. The no-dust-emission twin of each
# code is built at the same attenuation and subtracted, so the ratio is the dust
# emission against the dust emission.
_sed_c_twin = C.run_chain([_SFH_CHAIN, _BC03_CHAIN, NEB_FIDUCIAL_CIGALE, _DUSTATT_CHAIN])
_w_twin_c, _L_twin_c = C.to_lnu(_sed_c_twin)
_m_twin_t = _knob_model("none")
_o_twin_t = _m_twin_t.predict_rest_sed(dict(_m_twin_t.spec.sample(jax.random.PRNGKey(0))))
_w_twin_t, _L_twin_t = np.asarray(_o_twin_t.wavelength), np.asarray(_o_twin_t.sed)
m_dl14 = _knob_model(
    "draine_li2014",
    qpah=Fixed(2.5),
    umin=Fixed(1.0),
    gamma_dl=Fixed(0.1),
    alpha_dl14=Uniform(1.0, 3.0, default=2.0),
)
p_dl14 = dict(m_dl14.spec.sample(jax.random.PRNGKey(0)))
_dl14_cases = []
_dl14_medians = []
for _alpha, _c in zip([1.0, 2.0, 3.0], ["C0", "C1", "C3"]):
    sed = C.run_chain(
        [
            _SFH_CHAIN,
            _BC03_CHAIN,
            NEB_FIDUCIAL_CIGALE,
            _DUSTATT_CHAIN,
            ("dl2014", dict(qpah=2.5, umin=1.0, alpha=_alpha, gamma=0.1)),
        ]
    )
    _w_c_raw, _L_c_raw = C.to_lnu(sed)
    w_c, nl_c = _nu_lnu(_w_c_raw, _L_c_raw)
    ax_r.loglog(w_c, nl_c * _KNOB_MASS, color=_c, **_REF_KW)
    o = m_dl14.predict_rest_sed({**p_dl14, "dust_alpha_dl14": jnp.float64(_alpha)})
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_r.loglog(w_t, nl_t, color=_c, lw=_TNG_LW, label=rf"$\alpha$={_alpha:g}")
    _peaks_6c.append(float(np.nanmax(nl_t)))
    # Dust emission alone: total minus the same model without dust emission.
    _L_dust_c = (_L_c_raw - U.regrid(_w_twin_c, _L_twin_c, _w_c_raw)) * _KNOB_MASS
    _L_dust_t = np.asarray(o.sed) - U.regrid(_w_twin_t, _L_twin_t, np.asarray(o.wavelength))
    _, nl_dust_c = _nu_lnu(_w_c_raw, _L_dust_c)
    _, nl_dust_t = _nu_lnu(o.wavelength, _L_dust_t)
    _rk = U.regrid(w_t, nl_dust_t, w_c) / np.where(nl_dust_c > 0, nl_dust_c, np.nan)
    _ratios_6c_r.append((_c, w_c, _rk))
    _dl14_medians.append(_band_median(_rk, w_c, 8.0, 1000.0))
    _band_sel = (_w_c_raw >= 2.5e6) & (_w_c_raw <= 1e7)
    _share_c = _lbol_nu(_w_c_raw[_band_sel], _L_dust_c[_band_sel]) / _lbol_nu(
        _w_c_raw[_band_sel], (_L_c_raw * _KNOB_MASS)[_band_sel]
    )
    print(
        f"    alpha={_alpha:g}: dust-only 8-1000 µm {_dl14_medians[-1]:.4f}×; "
        f"dust is {_share_c * 100:.0f}% of CIGALE's total SED over 250-1000 µm"
    )
    _dl14_cases.append(
        (
            f"dl2014 α={_alpha:g} (dust only)",
            _w_c_raw,
            _L_dust_c,
            np.asarray(o.wavelength),
            _L_dust_t,
        )
    )
assert all(0.98 < _m < 1.04 for _m in _dl14_medians), _dl14_medians
ax_r.plot([], [], "k-", **_REF_KW, label="pcigale")
ax_r.plot([], [], "k-", lw=_TNG_LW, label="tengri")
ax_r.set(xlim=(1e4, 1e7), title="Draine & Li 2014")
ax_r.legend(fontsize=8, frameon=False)
ax_r.tick_params(labelbottom=False)

_ypk6c = max(_peaks_6c)
for ax in (ax_l, ax_r):
    ax.set_ylim(_ypk6c * 1e-5, _ypk6c * 2.0)

# Ratio panels for dl2007 (restrict to dust-dominated range above 10^5 Å to avoid nebular line spikes)
for c, w, ratio in _ratios_6c_l:
    pos = (w >= 1e5) & (w > 0)
    ax_lr.plot(w[pos], ratio[pos], color=c, linewidth=1.0)
ax_lr.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_lr.axhline(1.0, color="0.5", linewidth=0.8)
ax_lr.set(xlim=(1e5, 1e7), ylim=(0.5, 2.0), xlabel=r"$\lambda$ [Å]", ylabel="tengri/CIGALE")
ax_lr.grid(True, alpha=0.3)

# Ratio panels for dl2014 (restrict to dust-dominated range above 10^5 Å to avoid nebular line spikes)
for c, w, ratio in _ratios_6c_r:
    pos = (w >= 1e5) & (w > 0)
    ax_rr.plot(w[pos], ratio[pos], color=c, linewidth=1.0)
ax_rr.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_rr.axhline(1.0, color="0.5", linewidth=0.8)
ax_rr.set(xlim=(1e5, 1e7), ylim=(0.5, 2.0), xlabel=r"$\lambda$ [Å]")
ax_rr.tick_params(labelleft=False)
ax_rr.grid(True, alpha=0.3)

fig.tight_layout()
save_fig("cigale_06c_dl07_dl14.png")

for _name, _cases in (("dl2007", _dl07_cases), ("dl2014", _dl14_cases)):
    for _label, _w_ref, _L_ref, _w_t, _L_t in _cases:
        _rows = V.filter_rows_native(
            np.asarray(_w_t), np.asarray(_L_t), _w_ref, _L_ref, filters=V.IR_BANDS
        )
        V.print_filter_table(_rows, ref_name="CIGALE", title=f"§6c {_label}", compact=True)

fig = plt.figure(figsize=(13.5, 6.5))
gs = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.3)
ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
ax_ar, ax_br, ax_cr = (
    fig.add_subplot(gs[1, 0]),
    fig.add_subplot(gs[1, 1]),
    fig.add_subplot(gs[1, 2]),
)
ax_ar.sharex(ax_a)
ax_br.sharex(ax_b)
ax_cr.sharex(ax_c)
ax_br.sharey(ax_ar)
ax_cr.sharey(ax_ar)
_peaks_6d = []

print("§6c casey2012 T sweep (tengri / CIGALE, median in 8-1000 µm band):")
m_cas = _knob_model(
    "casey2012", T=Uniform(10.0, 80.0, default=35.0), beta_ir=Fixed(1.6), alpha_mir=Fixed(2.0)
)
p_cas = dict(m_cas.spec.sample(jax.random.PRNGKey(0)))
_cas_cases = []
_ratios_6d_a = []
for _T, _c in zip([25.0, 35.0, 50.0], ["C0", "C1", "C3"]):
    sed = C.run_chain(
        [
            _SFH_CHAIN,
            _BC03_CHAIN,
            NEB_FIDUCIAL_CIGALE,
            _DUSTATT_CHAIN,
            ("casey2012", dict(temperature=_T, beta=1.6, alpha=2.0)),
        ]
    )
    _w_c_raw, _L_c_raw = C.to_lnu(sed)
    w_c, nl_c = _nu_lnu(_w_c_raw, _L_c_raw)
    ax_a.loglog(w_c, nl_c * _KNOB_MASS, color=_c, **_REF_KW)
    o = m_cas.predict_rest_sed({**p_cas, "dust_T": jnp.float64(_T)})
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_a.loglog(w_t, nl_t, color=_c, lw=_TNG_LW, label=rf"T={_T:g} K")
    _peaks_6d.append(float(np.nanmax(nl_t)))
    _rk = U.regrid(w_t, nl_t, w_c) / np.where(nl_c > 0, nl_c * _KNOB_MASS, np.nan)
    _ratios_6d_a.append((_c, w_c, _rk))
    print(f"    T={_T:g} K: 8-1000 µm {_band_median(_rk, w_c, 8.0, 1000.0):.4f}×")
    _cas_cases.append(
        (
            f"casey2012 T={_T:g}",
            _w_c_raw,
            _L_c_raw * _KNOB_MASS,
            np.asarray(o.wavelength),
            np.asarray(o.sed),
        )
    )
ax_a.plot([], [], "k-", **_REF_KW, label="pcigale")
ax_a.plot([], [], "k-", lw=_TNG_LW, label="tengri")
ax_a.set(xlim=(1e4, 1e7), ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]", title="Casey 2012")
ax_a.legend(fontsize=8, frameon=False)
ax_a.tick_params(labelbottom=False)

print("§6c schreiber2016 T sweep (tengri / CIGALE, median in 8-1000 µm band):")
m_sch = _knob_model("schreiber2016", T=Uniform(15.0, 60.0, default=30.0))
p_sch = dict(m_sch.spec.sample(jax.random.PRNGKey(0)))
_sch_cases = []
_ratios_6d_b = []
for _T, _c in zip([25.0, 35.0, 50.0], ["C0", "C1", "C3"]):
    sed = C.run_chain(
        [
            _SFH_CHAIN,
            _BC03_CHAIN,
            NEB_FIDUCIAL_CIGALE,
            _DUSTATT_CHAIN,
            ("schreiber2016", dict(tdust=_T)),
        ]
    )
    _w_c_raw, _L_c_raw = C.to_lnu(sed)
    w_c, nl_c = _nu_lnu(_w_c_raw, _L_c_raw)
    ax_b.loglog(w_c, nl_c * _KNOB_MASS, color=_c, **_REF_KW)
    o = m_sch.predict_rest_sed({**p_sch, "dust_T": jnp.float64(_T)})
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_b.loglog(w_t, nl_t, color=_c, lw=_TNG_LW, label=rf"T={_T:g} K")
    _peaks_6d.append(float(np.nanmax(nl_t)))
    _rk = U.regrid(w_t, nl_t, w_c) / np.where(nl_c > 0, nl_c * _KNOB_MASS, np.nan)
    _ratios_6d_b.append((_c, w_c, _rk))
    print(f"    T={_T:g} K: 8-1000 µm {_band_median(_rk, w_c, 8.0, 1000.0):.4f}×")
    _sch_cases.append(
        (
            f"schreiber2016 T={_T:g}",
            _w_c_raw,
            _L_c_raw * _KNOB_MASS,
            np.asarray(o.wavelength),
            np.asarray(o.sed),
        )
    )
ax_b.plot([], [], "k-", **_REF_KW, label="pcigale")
ax_b.plot([], [], "k-", lw=_TNG_LW, label="tengri")
ax_b.set(xlim=(1e4, 1e7), title="Schreiber 2016")
ax_b.legend(fontsize=8, frameon=False)
ax_b.tick_params(labelbottom=False)

print("§6c dale2014 alpha sweep (tengri / CIGALE, median in 8-1000 µm band):")
m_dale = _knob_model("dale2014_cigale", alpha_dale=Uniform(0.0625, 4.0, default=2.0))
p_dale = dict(m_dale.spec.sample(jax.random.PRNGKey(0)))
_dale_cases = []
_ratios_6d_c = []
for _alpha, _c in zip([0.5, 2.0, 4.0], ["C0", "C1", "C3"]):
    sed = C.run_chain(
        [
            _SFH_CHAIN,
            _BC03_CHAIN,
            NEB_FIDUCIAL_CIGALE,
            _DUSTATT_CHAIN,
            ("dale2014", dict(alpha=_alpha)),
        ]
    )
    _w_c_raw, _L_c_raw = C.to_lnu(sed)
    w_c, nl_c = _nu_lnu(_w_c_raw, _L_c_raw)
    ax_c.loglog(w_c, nl_c * _KNOB_MASS, color=_c, **_REF_KW)
    o = m_dale.predict_rest_sed({**p_dale, "dust_alpha_dale": jnp.float64(_alpha)})
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_c.loglog(w_t, nl_t, color=_c, lw=_TNG_LW, label=rf"$\alpha$={_alpha:g}")
    _peaks_6d.append(float(np.nanmax(nl_t)))
    _rk = U.regrid(w_t, nl_t, w_c) / np.where(nl_c > 0, nl_c * _KNOB_MASS, np.nan)
    _ratios_6d_c.append((_c, w_c, _rk))
    print(f"    alpha={_alpha:g}: 8-1000 µm {_band_median(_rk, w_c, 8.0, 1000.0):.4f}×")
    _dale_cases.append(
        (
            f"dale2014 α={_alpha:g}",
            _w_c_raw,
            _L_c_raw * _KNOB_MASS,
            np.asarray(o.wavelength),
            np.asarray(o.sed),
        )
    )
ax_c.plot([], [], "k-", **_REF_KW, label="pcigale")
ax_c.plot([], [], "k-", lw=_TNG_LW, label="tengri")
ax_c.set(xlim=(1e4, 1e7), title="Dale 2014 (CIGALE grid)")
ax_c.legend(fontsize=8, frameon=False)
ax_c.tick_params(labelbottom=False)

_ypk6d = max(_peaks_6d)
for ax in (ax_a, ax_b, ax_c):
    ax.set_ylim(_ypk6d * 1e-5, _ypk6d * 2.0)

# Ratio panels (restrict to dust-dominated range above 10^5 Å to avoid nebular line spikes)
for c, w, ratio in _ratios_6d_a:
    pos = (w >= 1e5) & (w > 0)
    ax_ar.plot(w[pos], ratio[pos], color=c, linewidth=1.0)
ax_ar.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_ar.axhline(1.0, color="0.5", linewidth=0.8)
ax_ar.set(xlim=(1e5, 1e7), ylim=(0.5, 2.0), xlabel=r"$\lambda$ [Å]", ylabel="tengri/CIGALE")
ax_ar.grid(True, alpha=0.3)

for c, w, ratio in _ratios_6d_b:
    pos = (w >= 1e5) & (w > 0)
    ax_br.plot(w[pos], ratio[pos], color=c, linewidth=1.0)
ax_br.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_br.axhline(1.0, color="0.5", linewidth=0.8)
ax_br.set(xlim=(1e5, 1e7), ylim=(0.5, 2.0), xlabel=r"$\lambda$ [Å]")
ax_br.tick_params(labelleft=False)
ax_br.grid(True, alpha=0.3)

for c, w, ratio in _ratios_6d_c:
    pos = (w >= 1e5) & (w > 0)
    ax_cr.plot(w[pos], ratio[pos], color=c, linewidth=1.0)
ax_cr.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_cr.axhline(1.0, color="0.5", linewidth=0.8)
ax_cr.set(xlim=(1e5, 1e7), ylim=(0.5, 2.0), xlabel=r"$\lambda$ [Å]")
ax_cr.tick_params(labelleft=False)
ax_cr.grid(True, alpha=0.3)

fig.tight_layout()
save_fig("cigale_06d_casey_schreiber_dale.png")

for _name, _cases in (
    ("casey2012", _cas_cases),
    ("schreiber2016", _sch_cases),
    ("dale2014", _dale_cases),
):
    for _label, _w_ref, _L_ref, _w_t, _L_t in _cases:
        _rows = V.filter_rows_native(
            np.asarray(_w_t), np.asarray(_L_t), _w_ref, _L_ref, filters=V.IR_BANDS
        )
        V.print_filter_table(_rows, ref_name="CIGALE", title=f"§6c {_label}", compact=True)


# %% [markdown]
# ## §7 Panchromatic SED
#
# The §6 model from the Lyman continuum to 1 m: stellar SED convolved
# with the τ-delayed SFH, dust attenuation and Dale+2014 IR re-emission.
# X-ray and radio blocks arrive in §10 and §11.
#
# **Far-UV (λ < 912 Å).** Calzetti+2000 was fit on 1200–22000 Å; tengri's
# polynomial extrapolates below that, while CIGALE's `dustatt_modified_starburst`
# drops to zero at 912 Å. Setting `dust_attenuation={'lyman_cutoff': True}`
# applies the same 912 Å clip on both sides.
#
# **Beyond 1 mm.** Both Cue and CIGALE's nebular modules carry thermal
# free-free into the radio, and this panel shows the comparison when both
# grids extend to 1 m (tengri with the nebular tail, CIGALE's native reach).
# Both codes' Dale 2014 templates end at 6 mm. The nebular continuum there is
# Q_H times a per-photon continuum, so its ratio is §3's Q_H ratio times the
# Cue-against-Cloudy ratio per ionizing photon that §11b measures.
#
# **Bands.** The ratios printed below are energy-weighted, ∫L_ν dν over each
# band. The 1250–3000 Å band starts at 1250 Å, outside the Lyα window: the SSPs'
# Lyα region differs by a large factor between the two grids at a few points and
# would otherwise set the band.

# %%
# Two panels on independent y-axes read as agreement whatever they contain,
# so the band-by-band ratio is printed beside them. The §6 model, seen from
# the Lyman continuum to the submillimeter.
_L_t_pan = U.regrid(np.asarray(s_ir.wave), np.asarray(s_ir.sed_intrinsic), w_c_ir)
# ``regrid`` zero-fills outside tengri's grid, and a zero divided by CIGALE's
# model is a ratio of 0.000 that reads as total disagreement rather than as
# "the model stops here". Count the points where *both* codes carry a model
# and say so, so the mm tail is reported as missing coverage and not as a
# residual.
_L_c_ir_floor = 1e-6 * L_c_ir.max()
_cov_pan = (L_c_ir > _L_c_ir_floor) & (_L_t_pan > 0)
print("§7 panchromatic energy-weighted ratio, ∫L_ν dν per band (tengri / CIGALE):")
for _lo, _hi, _wname in (
    (200.0, 912.0, "200–912 Å (LyC) "),
    (912.0, 1200.0, "912–1200 Å      "),
    (1250.0, 3000.0, "1250–3000 Å     "),
    (3000.0, 10000.0, "0.3–1 µm        "),
    (3.0e4, 8.0e4, "3–8 µm          "),
    (1.0e5, 1.0e6, "10–100 µm       "),
    (1.0e6, 1.0e7, "100–1000 µm     "),
):
    _band = (w_c_ir >= _lo) & (w_c_ir <= _hi)
    _n_c = int((_band & (L_c_ir > 0)).sum())
    _n_both = int((_band & _cov_pan).sum())
    if _n_both:
        print(
            f"    {_wname}: "
            f"{_band_energy_ratio(w_c_ir[_cov_pan], L_c_ir[_cov_pan], np.asarray(s_ir.wave), np.asarray(s_ir.sed_intrinsic), _lo, _hi):.4f}×  "
            f"({_n_both}/{_n_c} points where both grids carry a model)"
        )
    else:
        print(f"    {_wname}: tengri's grid does not reach this band ({_n_c} CIGALE points)")

fig, ax, ax_r, ratio = V.overlay_ratio_fig(
    w_c_ir,
    L_c_ir,
    np.asarray(s_ir.wave),
    np.asarray(s_ir.sed_intrinsic),
    xlabel=r"$\lambda$ [Å]",
    title="§7 panchromatic SED (τ=1 Gyr, age=5 Gyr, with Dale+2014 dust IR)",
    ref_label="pcigale",
    label_t="tengri",
    xlim=(1e2, 3e6),
)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
fig.tight_layout()
fig.savefig(str(figs_dir / "cigale_07_panchromatic_full.png"), dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## §8 Nebular emission
#
# CIGALE uses static CLOUDY grids (`pcigale.sed_modules.nebular`).
# tengri uses **Cue** (Li et al. 2025), a neural emulator of the same
# physics that exposes logU, gas metallicity and IMF as continuous
# parameters. Cue requires the bare-stellar SSP this notebook loaded.
#
# **Fiducial choice: young population.** Sections §3–§7 use a 5 Gyr
# quiescent τ=1 Gyr galaxy (Boquien+2019 reference), which has almost no
# ionizing budget. Nebular emission lives in stars ≲ 100 Myr old, so §8
# swaps to a **τ=300 Myr, age=100 Myr** delayed SFH where Hα and the
# metal-line forest are physically strong. Only the SFH changes.
#
# The two emitters see the same H II region: `logU = −2.0`, Z_gas = 0.02
# (Cue's `neb_logZ_gas` pinned to `log10(0.02/Z_⊙) ≈ +0.149`, the same
# absolute metallicity as CIGALE's `zgas = 0.02` and as the stars on both
# sides), `f_esc = 0`, `n_H = 100 cm⁻³` (`gas_logn = 2.0`), solar N/O and C/O
# (`gas_logno = gas_logco = 0`). The Q_H reaching Cue is the integral of the
# SSP-convolved ionizing spectrum below 911.76 Å, published by the stellar
# component on every forward pass.
#
# **Read the ionizing budget before the line ratios.** Every line scales with the
# Q_H handed to the emitter, so the cell prints Q_H first: CIGALE's
# `stellar.n_ly` against tengri's `nion`, both on the shared BC03 grid. There the
# ratio is §3's age convention alone, 1.146× for this population. Both emitters
# see the same stellar population, and the lines are compared on that shared
# grid: tengri's from Cue's catalogue, which returns each line's integrated
# luminosity independent of any wavelength grid, and CIGALE's from a window
# integral over its own line-resolved grid.
#
# **Raw ratios and ratios per ionizing photon.** The raw line ratio contains Q_H;
# line / Q_H takes it out and leaves the emitter. Hα and Hβ are 1.12× raw, which is
# the 1.146× of Q_H less the emitter's deficit, and 0.974× and 0.978× per photon:
# Cue's Balmer emission per ionizing photon is 2–3 % below CIGALE's Cloudy grid, a
# difference between two photoionization models.
# [O III] is 0.96× raw and 0.834× per photon.
#
# **[O III] and the U definition.** Two conventions set the [O III] per photon,
# and neither is a normalization error. First, Cue's ionizing spectrum enters as a
# shape fitted to the stellar population, and the fitted shape of this population is
# soft: §8b prints how strongly Cue's [O III]/Hβ response to logU depends on it.
# Second, the two codes may not define the ionization parameter at the same radius.
# Cue's `neb_logU` is $U = Q_{\rm H}/(4\pi R_{\rm inner}^2 n_{\rm H} c)$ at the inner face
# of the cloud, with $R_{\rm inner}$ at the 10¹⁹ cm default of the Cue package's
# `logQ` function (Li et al. 2025). CIGALE's nebular module reads a precomputed Cloudy
# grid in logU, metallicity and density; the module's documentation (its docstring
# and parameter text, pcigale 2025.1) does not state the radius at which logU is
# evaluated. Other
# nebular grids fix it elsewhere: Gutkin et al. (2016) define it at the Strömgren
# radius. The offset between such definitions depends on the cloud, so it is not
# quantified here, and the same
# `logU = −2` need not be the same nebula. The higher-ionization lines move the most.
#
# Each line is quantified with the **integrated** luminosity
# (continuum-subtracted, width-independent), not a single-bin peak ratio,
# which would measure line width and grid resolution (CIGALE broadens to
# `lines_width = 300 km/s`) rather than physics.
#
# **Grid coverage.** Cue's native grid runs ~915 Å – 10⁸ Å, with a thermal
# free-free tail extending to 10¹⁰ Å (radio); CIGALE's CLOUDY grid extends to
# far-IR fine-structure lines ([O III] 88 μm, [C II] 158 μm, [S III] 18.7 μm,
# [Ne III] 15.6 μm out to ~10⁶ Å), which is why the left panel shows line spikes
# the Cue panel does not. For a CLOUDY-vs-CLOUDY match, tengri exposes its own
# static grid via `neb={'type': 'cloudy'}` (`data/cloudy_grid_*.h5`, 166 lines
# to 6.1×10⁶ Å).
#
# **Verification Status:** PARTIAL (1/10): Cue nebular emulator

# %%
# §8 young fiducial: τ=300 Myr, age=100 Myr — Hα-bright. CIGALE accepts
# Myr values directly; tengri takes Gyr via tau_gyr/age_gyr.
_TAU_MAIN_YOUNG_MYR = 300
_AGE_MAIN_YOUNG_MYR = 100
_sfh_args = (
    "sfhdelayed",
    dict(
        tau_main=_TAU_MAIN_YOUNG_MYR,
        age_main=_AGE_MAIN_YOUNG_MYR,
        tau_burst=50,
        age_burst=20,
        f_burst=0.0,
        sfr_A=1.0,
        normalise=True,
    ),
)
sed_c_st = C.run_chain(
    [
        _sfh_args,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
    ]
)
w_c_st, L_c_st = C.to_lnu(sed_c_st)

sed_c_neb = C.run_chain(
    [
        _sfh_args,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        (
            "nebular",
            dict(
                logU=-2.0,
                zgas=0.02,
                ne=100,
                f_esc=0.0,
                f_dust=0.0,
                lines_width=300.0,
                emission=True,
                line_list="",
            ),
        ),
    ]
)
w_c_neb, L_c_neb = C.to_lnu(sed_c_neb)

_neb_sfh_kw = {
    "type": "delayed",
    "tau_gyr": Fixed(_TAU_MAIN_YOUNG_MYR / 1000),
    "age_gyr": Fixed(_AGE_MAIN_YOUNG_MYR / 1000),
    "log_total_mass": Fixed(0.0),
    "all_params": Fixed(DEFAULT),
}

m_neb = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh=_neb_sfh_kw,
    neb=NEB_FIDUCIAL_TENGRI,  # ionspec_* slopes stay at their SSP-derived Fixed values
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    redshift=Fixed(0.0),
)
s_neb = m_neb.predict_state({})

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="pcigale  CLOUDY nebular", label_r="tengri  Cue nebular (Li+2024)")
# CIGALE side — dashed stellar + solid (stellar+nebular) + dotted
# nebular-only (the line forest + smooth continuum that CLOUDY adds).
L_c_neb_only = np.maximum(L_c_neb - U.regrid(w_c_st, L_c_st, w_c_neb), 1e-30)
ax_l.plot(w_c_st, L_c_st, "k--", linewidth=1.0, alpha=0.5, label="stellar only")
ax_l.plot(w_c_neb, L_c_neb, "C0-", linewidth=1.4, alpha=0.7, label="stellar + CLOUDY nebular")
ax_l.plot(w_c_neb, L_c_neb_only, "C0:", linewidth=1.4, label="CLOUDY nebular only")
ax_l.legend(fontsize=8)
# tengri side — same three traces (stellar dashed, stellar+Cue solid,
# Cue-only dotted). Agreement is limited by the two emitters' Cloudy
# models (Cue: Cloudy 22.00), bare-stellar vs wNE-SSP path, and line-broadening kernel;
# see the integrated-line-luminosity ratio printed below.
# ``derived["sed_nebular"]`` publishes the photoionized continuum + lines
# directly, so the stellar-only trace is read off the same nebular-on build
# by subtracting it back out -- no separate stellar-only model needed (every
# build on this page carries the nebular fiducial; see Setup).
L_t_neb_only = np.maximum(np.asarray(s_neb.derived["sed_nebular"]), 1e-30)
_s_stellar_on_neb = np.maximum(np.asarray(s_neb.sed_intrinsic) - L_t_neb_only, 1e-30)
ax_r.plot(s_neb.wave, _s_stellar_on_neb, "k--", linewidth=1.0, alpha=0.5, label="stellar only")
ax_r.plot(s_neb.wave, s_neb.sed_intrinsic, "C1-", linewidth=1.4, alpha=0.7, label="stellar + Cue")
ax_r.plot(s_neb.wave, L_t_neb_only, "C1:", linewidth=1.4, label="Cue nebular only")
ax_r.legend(fontsize=8)
_xmin_n = float(min(w_c_neb.min(), float(np.asarray(s_neb.wave).min())))
_xmax_n = float(max(w_c_neb.max(), float(np.asarray(s_neb.wave).max())))
_ymax_n = max(float(L_c_neb.max()), float(np.asarray(s_neb.sed_intrinsic).max()))
for ax in (ax_l, ax_r):
    ax.set_xlim(_xmin_n, _xmax_n)
    ax.set_ylim(_ymax_n * 1e-6, _ymax_n * 2)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
plt.close(fig)  # superseded by §8b's logU x Z_gas x f_esc grid, which replaces this panel

# Quantify the residual by integrated line luminosity (width- and grid-
# independent). A single-bin peak ratio measures line width, not luminosity:
# CIGALE broadens its lines (lines_width = 300 km/s) while Cue applies its own.
# Both emitters run on the shared BC03 SSP, so the only difference in the
# ionizing budget reaching them is §3's age convention.
def _tengri_line_lum(state, lo_aa: float, hi_aa: float) -> float:
    """Integrated luminosity [erg/s] of the catalogued Cue lines in [lo_aa, hi_aa] (vacuum Å).

    Cue returns each line's integrated luminosity directly, so the value does not
    depend on the wavelength grid of the SSP the model was built on. CIGALE's side
    is the window integral over its own line-resolved grid.
    """
    _lw = np.asarray(state.derived["line_waves"])
    _ll = np.asarray(state.derived["line_lums"])
    _sel = (_lw >= lo_aa) & (_lw <= hi_aa)
    assert _sel.any(), (lo_aa, hi_aa)
    return float(_ll[_sel].sum())


_LINE_WINDOWS = {  # centre [Å] -> (lo, hi) catalogue window
    6564.61: (6563.6, 6565.6),
    5008.24: (5007.2, 5009.2),
    4862.68: (4861.7, 4863.7),
    3728.48: (3726.0, 3730.5),  # the [O II] doublet, 3727.09 + 3729.87
}


# The ionizing budget first. Every line scales with the Q_H handed to the
# emitter, so a line ratio read without it cannot separate "different line
# physics" from "different number of ionizing photons". CIGALE publishes it
# as ``stellar.n_ly``; tengri as ``derived["nion"]``. Both sides form 1 M☉.
_n_ly_c = float(sed_c_neb.info["stellar.n_ly"])
_q_h_neb = float(np.asarray(s_neb.derived["nion"]))
print("§8 ionizing photon rate Q_H reaching the emitter [ph/s per M☉ formed]:")
print(f"    CIGALE  stellar.n_ly on BC03      {_n_ly_c:.4e}")
print(f"    tengri  nion on the shared BC03   {_q_h_neb:.4e}  → {_q_h_neb / _n_ly_c:.3f}×")

# Raw ratios and ratios per ionizing photon, side by side. Dividing each code's
# line by its own Q_H removes the Q_H term and leaves the emitter.
print("§8 integrated line luminosity, tengri Cue / CIGALE Cloudy, shared BC03 SSP:")
_neb_line_rows = {}
for _c, _name in [(6564.61, "Hα"), (5008.24, "[O III]"), (4862.68, "Hβ")]:
    _lc = U.line_lum(w_c_neb, L_c_neb_only, _c)
    _lt = _tengri_line_lum(s_neb, *_LINE_WINDOWS[_c])
    if _lc > 0:
        _neb_line_rows[_name] = (_lt / _lc, (_lt / _q_h_neb) / (_lc / _n_ly_c))
        print(
            f"    {_name} {_c:.0f} Å: CIGALE {_lc:.2e}, tengri {_lt:.2e} erg/s → "
            f"raw {_lt / _lc:.2f}×; per ionizing photon CIGALE {_lc / _n_ly_c:.3e}, "
            f"tengri {_lt / _q_h_neb:.3e} erg → {_neb_line_rows[_name][1]:.3f}×"
        )
    else:
        # Not skipped silently: an empty CIGALE window is a grid-resolution
        # failure of the measurement, not an absent line.
        print(f"    {_name} {_c:.0f} Å: CIGALE window holds < 2 grid points — not measurable here")


# %% [markdown]
# ### §8b logU × Z_gas × f_esc
#
# Same young fiducial and shared BC03 SSP as above. logU ∈ {−3, −2, −1.5}
# at Z_gas = 0.02; Z_gas ∈ {0.004, 0.02, 0.041} (CIGALE's nearest grid
# point to 0.04) at logU = −2; f_esc ∈ {0, 0.5} at the logU/Z_gas
# fiducial (7 cases). One tengri build with `neb_logU`, `neb_logZ_gas`,
# `neb_fesc` free, evaluated per case via `predict_state`; one CIGALE
# `nebular` call per case. Rows: Hα/Hβ, [O III]/Hβ, [O II]/Hβ, tengri and
# CIGALE side by side, and the absolute line / Q_H beneath each; a residual here
# is Cue versus CLOUDY (§8), not a parity check.
#
# **Z_gas = 0.041 is degenerate in CIGALE and is not a comparison.** CIGALE's
# metallicity axis is a staircase: neighbouring Z nodes repeat the same model (the
# printed library table shows Z = 0.016, 0.019, 0.02 and 0.022 identical), and
# toward the high-Z end the library collapses: [O III]/Hβ is 0.0061 and [O II]/Hβ
# 0.0095 at Z = 0.041 and smaller beyond. The ratio-of-ratios is therefore
# reported at the six stable cases, with the 0.041 case printed beside it. The
# absolute Hβ per ionizing photon is flat to 3 % across every case in both codes, so
# the collapse is in the metal lines.
#
# **The f_esc rows test ratios, and ratios do not depend on f_esc.** The absolute
# Hβ / Q_H does: both codes scale it by (1 − f)/(1 + 0.597 f), 0.385 at f = 0.5,
# which the cell asserts against each code's own f = 0 value.
#
# **Reading the [O III]/Hβ response.** `neb_logU` reaches Cue unchanged as its own
# ionization parameter. Cue's [O III]/Hβ response to it depends on the ionizing
# spectrum Cue is fed. At the shape fitted to this population the ratio rises
# modestly with logU (1.154, 1.596, 1.663 in the table above, for logU = −3, −2,
# −1.5). A final cell evaluates the network at Cue's own default (hard)
# young-starburst shape and prints the ratio for it, which rises much more steeply
# (3.416, 5.499, 6.085). The cell shows that the response depends on the ionizing
# shape; it does not by itself reproduce CIGALE's values, whose logU is defined
# at a different radius (§8).

# %%
_m_8b = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh=_neb_sfh_kw,
    neb={
        "type": "cue",
        "neb_logU": Uniform(-4.0, -1.0, default=-2.0),
        "neb_logZ_gas": Uniform(-1.0, 0.5, default=MET_LOGZSOL),
        "neb_fesc": Uniform(0.0, 1.0, default=0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    redshift=Fixed(0.0),
)
_p_8b = dict(_m_8b.spec.sample(jax.random.PRNGKey(0)))

_8b_grid = [
    ("logU=-3", -3.0, 0.02, 0.0),
    ("logU=-2 (fid)", -2.0, 0.02, 0.0),
    ("logU=-1.5", -1.5, 0.02, 0.0),
    ("Zgas=0.004", -2.0, 0.004, 0.0),
    ("Zgas=0.041", -2.0, 0.041, 0.0),
    ("fesc=0", -2.0, 0.02, 0.0),
    ("fesc=0.5", -2.0, 0.02, 0.5),
]
_LINE_RATIOS_8B = (
    ("Ha/Hb", 6564.61, 4862.68),
    ("[OIII]/Hb", 5008.24, 4862.68),
    ("[OII]/Hb", 3728.48, 4862.68),  # center of the vacuum 3727.09 / 3729.88 doublet
)

_ABS_LINES_8B = (("Hb", 4862.68), ("[OIII]", 5008.24), ("[OII]", 3728.48))
_DEGENERATE_CASE_8B = "Zgas=0.041"  # CIGALE's grid collapses here; see the prose above

_rt_8b_all, _rc_8b_all, _at_8b_all, _ac_8b_all = [], [], [], []
print("§8b logU × Z_gas × f_esc (7 cases): line ratios to Hβ, then absolute line / Q_H.")
for _label, _logU, _zgas, _fesc in _8b_grid:
    _sed_c8b = C.run_chain(
        [
            _sfh_args,
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            (
                "nebular",
                dict(
                    logU=_logU,
                    zgas=_zgas,
                    ne=100,
                    f_esc=_fesc,
                    f_dust=0.0,
                    lines_width=300.0,
                    emission=True,
                    line_list="",
                ),
            ),
        ]
    )
    _w_c8b, _L_c8b_full = C.to_lnu(_sed_c8b)
    _L_c8b_st = U.regrid(w_c_st, L_c_st, _w_c8b)
    _L_c8b_neb = np.maximum(_L_c8b_full - _L_c8b_st, 1e-30)

    _logzgas_8b = float(np.log10(_zgas) - LOG10_ZSUN)
    _st_8b = _m_8b.predict_state(
        {
            **_p_8b,
            "neb_logU": jnp.float64(_logU),
            "neb_logZ_gas": jnp.float64(_logzgas_8b),
            "neb_fesc": jnp.float64(_fesc),
        }
    )
    _q_t8b = float(np.asarray(_st_8b.derived["nion"]))
    _q_c8b = float(_sed_c8b.info["stellar.n_ly"])

    _rt, _rc, _at, _ac = {}, {}, {}, {}
    for _name, _num, _den in _LINE_RATIOS_8B:
        _lt_n = _tengri_line_lum(_st_8b, *_LINE_WINDOWS[_num])
        _lt_d = _tengri_line_lum(_st_8b, *_LINE_WINDOWS[_den])
        _lc_n, _lc_d = U.line_lum(_w_c8b, _L_c8b_neb, _num), U.line_lum(_w_c8b, _L_c8b_neb, _den)
        _rt[_name] = _lt_n / _lt_d if _lt_d > 0 else float("nan")
        _rc[_name] = _lc_n / _lc_d if _lc_d > 0 else float("nan")
    for _name, _wl in _ABS_LINES_8B:
        _at[_name] = _tengri_line_lum(_st_8b, *_LINE_WINDOWS[_wl]) / _q_t8b
        _ac[_name] = U.line_lum(_w_c8b, _L_c8b_neb, _wl) / _q_c8b
    _flag = "  [CIGALE grid collapsed: degenerate]" if _label == _DEGENERATE_CASE_8B else ""
    print(
        f"    {_label:<14} tengri Ha/Hb {_rt['Ha/Hb']:.3f}, [OIII]/Hb {_rt['[OIII]/Hb']:.3f}, "
        f"[OII]/Hb {_rt['[OII]/Hb']:.3f}  |  CIGALE Ha/Hb {_rc['Ha/Hb']:.3f}, "
        f"[OIII]/Hb {_rc['[OIII]/Hb']:.3f}, [OII]/Hb {_rc['[OII]/Hb']:.3f}{_flag}"
    )
    print(
        "    " + " " * 14 + " line/Q_H [1e-13 erg], tengri | CIGALE: "
        + ", ".join(f"{_n} {_at[_n] * 1e13:.3f} | {_ac[_n] * 1e13:.3f}" for _n, _ in _ABS_LINES_8B)
    )
    _rt_8b_all.append(_rt)
    _rc_8b_all.append(_rc)
    _at_8b_all.append(_at)
    _ac_8b_all.append(_ac)

# The ratio-of-ratios is reported at the stable nodes only; the degenerate
# Z_gas = 0.041 case is printed beside it so the reader sees what is excluded.
_stable_8b = [i for i, g in enumerate(_8b_grid) if g[0] != _DEGENERATE_CASE_8B]
_degen_8b = [i for i, g in enumerate(_8b_grid) if g[0] == _DEGENERATE_CASE_8B][0]
for _name, _, _ in _LINE_RATIOS_8B:
    _rr_all = [_t[_name] / _c[_name] for _t, _c in zip(_rt_8b_all, _rc_8b_all)]
    _j = max(_stable_8b, key=lambda i: abs(np.log(_rr_all[i])))
    print(
        f"§8b {_name}: largest tengri/CIGALE ratio-of-ratios over the six stable cases "
        f"{_rr_all[_j]:.2f}× at {_8b_grid[_j][0]}; at the degenerate {_DEGENERATE_CASE_8B} "
        f"{_rr_all[_degen_8b]:.2f}×"
    )

# The f_esc rows test ratios, which f_esc leaves untouched. The absolute Hβ per
# ionizing photon tests f_esc itself: each code's Hβ/Q_H at f_esc = 0.5 over its
# own value at f_esc = 0 against the closed form (1 - f)/(1 + 0.597 f).
_k_fesc = (1.0 - 0.5) / (1.0 + 0.597 * 0.5)
_i0 = [g[0] for g in _8b_grid].index("fesc=0")
_i5 = [g[0] for g in _8b_grid].index("fesc=0.5")
_k_t = _at_8b_all[_i5]["Hb"] / _at_8b_all[_i0]["Hb"]
_k_c = _ac_8b_all[_i5]["Hb"] / _ac_8b_all[_i0]["Hb"]
print(
    f"§8b Hβ/Q_H at f_esc = 0.5 over f_esc = 0: tengri {_k_t:.4f}, CIGALE {_k_c:.4f}; "
    f"(1 - f)/(1 + 0.597 f) = {_k_fesc:.4f}"
)
assert abs(_k_t / _k_fesc - 1.0) < 0.01 and abs(_k_c / _k_fesc - 1.0) < 0.01

# CIGALE's metallicity axis, read straight from its line library at logU = -2,
# n_e = 100 cm^-3: neighbouring Z nodes repeat the same model, which is why the
# 0.041 case above is not a point on a smooth curve.
from pcigale.data import SimpleDatabase as _CigaleLineDB

print("§8b CIGALE nebular line library, Z axis at logU = -2, n_e = 100 cm^-3:")
with _CigaleLineDB("nebular_lines") as _line_db:
    for _z in (0.014, 0.016, 0.019, 0.02, 0.022, 0.025, 0.03, 0.033, 0.037, 0.041, 0.046):
        _rec = _line_db.get(Z=_z, logU=-2.0, ne=100.0)
        _d = {str(n): v for n, v in zip(_rec.name, _rec.spec)}
        print(
            f"    Z = {_z:<6g} Hβ {_d['H-beta']:.4e}  [O III] 500.7/Hβ "
            f"{_d['OIII-500.7'] / _d['H-beta']:.4f}  [O II] 372.6+372.9/Hβ "
            f"{(_d['OII-372.6'] + _d['OII-372.9']) / _d['H-beta']:.4f}"
        )

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
_x_8b = np.arange(len(_8b_grid))
for _ax, _rows, _title in (
    (ax_l, _rc_8b_all, "CIGALE (CLOUDY)"),
    (ax_r, _rt_8b_all, "tengri (Cue)"),
):
    for _name, _color in zip(("Ha/Hb", "[OIII]/Hb", "[OII]/Hb"), ("C0", "C1", "C3")):
        _ax.plot(_x_8b, [r[_name] for r in _rows], "o-", color=_color, label=_name)
    _ax.set_xticks(_x_8b)
    _ax.set_xticklabels([g[0] for g in _8b_grid], rotation=60, fontsize=8)
    _ax.set_title(_title)
    _ax.set_yscale("log")
    _ax.grid(True, alpha=0.3)
ax_l.set_ylabel("line ratio (to Hβ)")
ax_l.legend(fontsize=8)
fig.suptitle("§8b logU × Z_gas × f_esc — line ratios")
fig.tight_layout()
save_fig("cigale_08b_neb_grid.png")


# %%
# Cue's [O III]/Hβ response to logU at Cue's own default (hard) young-starburst
# ionizing shape, from the same network `neb_logU` feeds. Only the ionizing
# spectrum differs from the table above. Density (`gas_logn`) and `gas_logqion` are
# left at Cue's own defaults (`gas_logn = 2.0`, the value the table uses;
# `default_gas_logqion = 49.1`), and Z_gas is the table's 0.02.
from tengri._data_setup import data_path as _data_path
from tengri.components.nebular.cue import CueBackend

_cue_hard = CueBackend(str(_data_path("cue_weights.npz")))
_HARD_SHAPE = dict(CueBackend._IONSPEC_DEFAULTS)


def _catalog_line(wave, lum, center_vac, tol=2.0):
    """Luminosity of the catalogued line nearest ``center_vac`` (vacuum Å)."""
    i = int(np.argmin(np.abs(wave - center_vac)))
    assert abs(wave[i] - center_vac) < tol, (center_vac, wave[i])
    return lum[i]


_o3_hard = {}
for _logU in (-3.0, -2.0, -1.5):
    _w_h, _l_h = _cue_hard.predict_nebular_line_luminosities(
        gas_logu=_logU,
        gas_logz=float(np.log10(0.02) - LOG10_ZSUN),
        **_HARD_SHAPE,
    )
    _w_h, _l_h = np.asarray(_w_h), np.asarray(_l_h)
    _o3_hard[_logU] = _catalog_line(_w_h, _l_h, 5008.24) / _catalog_line(_w_h, _l_h, 4862.68)
print(
    "§8b Cue [OIII]/Hb at Cue's default hard shape, logU = -3 / -2 / -1.5: "
    + " / ".join(f"{_o3_hard[k]:.3f}" for k in (-3.0, -2.0, -1.5))
)


# %% [markdown]
# ## §9 AGN
#
# CIGALE module `skirtor2016` ↔ tengri
# `agn={"type": "composable", "disc": schartmann2005, "torus": skirtor,
# "atten": polar_dust}` at the SKIRTOR fiducial (i = 30°, τ_9.7 = 7, oa = 40°,
# p = q = 1). The figure overlays the stellar + dust + AGN SEDs of the two codes;
# §9c prints the disc, torus and polar-dust luminosities separately.
#
# **Disc.** `disc.schartmann2005` matches CIGALE's `skirtor2016 disk_type=1`
# default: the Schartmann (2005) piecewise power law with the 1200 Å bend that
# CIGALE substitutes for the FITS-bundled disc. tengri also ships `disc.skirtor`
# (CIGALE `disk_type=0`, §9b) and `disc.adaf_lopez2024` (`disk_type=2`). For a
# differentiable (M_BH, ṁ, spin) disc use `disc={"type": "multicolor", ...}`; its
# UV continuum differs from CIGALE's, so this notebook uses
# `disc.schartmann2005`.
#
# **Three components, and all three have to be selected.** CIGALE's
# `skirtor2016` emits a disc, a torus and a Casey-2012 polar-dust graybody, and
# publishes them separately. tengri's composable AGN has the same three, but the
# polar graybody is a standalone `atten` block, and `derived["sed_agn_polar"]` is
# a zeros array whenever the block is not selected. The build names all three at
# CIGALE's defaults (`oa = 40°`, τ_9.7 = 7, p = q = 1, i = 30°,
# `agn_polar_ebv = 0.03`, T = 100 K, β = 1.6).
#
# **Inclination and polar cone.** `agn_cos_inc` is the one inclination of every
# torus (cos 30° here). The polar cone shares the opening angle of the selected
# torus (`agn_oa_skirtor`), so the cone's share of the disc follows the torus
# geometry.
#
# **Polar-dust extinction law.** CIGALE's `law=0` is the Bongiorno et al. (2012)
# power law `k = 1.39 (λ/μm)^−1.2` above 100 nm, with the SMC-bar shape of
# Weingartner & Draine (2001) below it. tengri's `polar_law="bongiorno"` is that
# curve and is what the builds here use. tengri's default, `polar_law="smc"`
# (Pei 1992), is a different curve; §9d prints how far apart they are.
#
# **The polar component is disc-shaped by construction.** Both codes set its
# luminosity from `g(oa) × ∫ disc(1 − e^−τ_polar) dλ`, the disc's own spectrum
# seen through the polar screen, so it moves with the disc's UV shape even at
# fixed `oa`, `agn_polar_ebv`, T and β. §9c and §9b print each code's polar share
# of its own AGN dust budget beside the component ratios: pcigale's share moves
# between its two disc types, and tengri's moves with it.
#
# **Energy-balance normalization.** CIGALE ties disc, torus and polar dust to a
# single `agn_power` reference through the fixed SKIRTOR template ratios. tengri's
# default `agn_norm='cigale_joint'` does the same: the tie normalizes the disc
# only (a corona, when the disc block carries one, rides on top; the
# Schartmann disc has none), with the disc set to `agn_power × R`, where
# `R = η(i)·∫disc/∫dust` and `η(i) = cos i (1+2cos i)/3` is the Stalevski+2016
# anisotropy factor. With `agn_ir_frac` set, `agn_power` is derived from
# `L_absorbed × f/(1−f)`, as CIGALE derives it from `fracAGN`, so §6's energy
# anchor enters the AGN normalization too. `agn_log_lbol`, the angle-integrated
# accretion power, is then not a free input: setting both is two statements of
# one power, and the build refuses it. Use `agn_norm='independent'` and
# `agn_log_lbol` for a model with no `agn_ir_frac`.
#
# **Verification Status:** CROSSVAL: SKIRTOR torus (mean 3-param)

# %%
# CIGALE's polar dust (`law=0`, E(B-V) = 0.03) is tengri's `polar_dust` block with
# `polar_law="bongiorno"`; the default `polar_law="smc"` is Pei (1992), a
# different curve (§9d prints the difference).
_POLAR_DUST_BLOCK = {
    "type": "polar_dust",
    "polar_law": "bongiorno",
    "agn_polar_ebv": Fixed(0.03),
    "all_params": Fixed(DEFAULT),
}
_sfh_args_d = (
    "sfhdelayed",
    dict(
        tau_main=1000,
        age_main=5000,
        tau_burst=50,
        age_burst=20,
        f_burst=0.0,
        sfr_A=1.0,
        normalise=True,
    ),
)
sed_c_base = C.run_chain(
    [
        _sfh_args_d,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        NEB_FIDUCIAL_CIGALE,
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
    ]
)
w_base, L_base = C.to_lnu(sed_c_base)

m_agn_base = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        # Match CIGALE's dustatt_modified_starburst, which drops to zero at the
        # Lyman limit; §7/§11 do the same. Without it the Calzetti/Leitherer
        # polynomial extrapolates below 912 Å and the stellar+dust baseline
        # sits above CIGALE in the far-UV, muddying the AGN comparison.
        "lyman_cutoff": True,
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
s_agn_base = m_agn_base.predict_state({})

# Full SED with and without AGN on both sides — clearer than the
# differential plot when one side has X-ray + the other doesn't.
sed_skirtor = C.run_chain(
    [
        _sfh_args_d,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        NEB_FIDUCIAL_CIGALE,
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        (
            "skirtor2016",
            dict(
                t=7,
                pl=1.0,
                q=1.0,
                oa=40,
                R=20,
                Mcl=0.97,
                i=30,
                disk_type=1,
                delta=0,
                fracAGN=0.3,
                lambda_fracAGN="0/0",
                law=0,
                EBV=0.03,
                temperature=100.0,
                emissivity=1.6,
            ),
        ),
    ]
)
w_skirt, L_skirt = C.to_lnu(sed_skirtor)

m_agn = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        # Match CIGALE's dustatt_modified_starburst, which drops to zero at the
        # Lyman limit; §7/§11 do the same. Without it the Calzetti/Leitherer
        # polynomial extrapolates below 912 Å and the stellar+dust baseline
        # sits above CIGALE in the far-UV, muddying the AGN comparison.
        "lyman_cutoff": True,
        "all_params": Fixed(DEFAULT),
    },
    # tengri's AGN defaults match CIGALE skirtor2016's (oa=40, tau=7, p=q=1,
    # i=30, EBV=0.03, T=100, β=1.6, disk_type=1 → ``disc.schartmann2005``),
    # except the polar-dust law, set in ``_POLAR_DUST_BLOCK``.
    agn={
        "type": "composable",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        # The Casey-2012 polar-dust graybody is its own attenuation block, not
        # part of the torus: without this entry ``derived["sed_agn_polar"]`` is
        # zeros. ``agn_polar_ebv = 0.03``, T = 100 K and β = 1.6 are CIGALE's
        # skirtor2016 defaults on both sides.
        "atten": _POLAR_DUST_BLOCK,
        # ``agn_ir_frac = 0.3`` mirrors CIGALE's ``fracAGN`` parameter.
        # tengri's AGN component reads ``state.derived["L_absorbed"]``
        # and computes ``agn_power = L_abs × frac/(1-frac)`` exactly
        # like CIGALE ``skirtor2016.py:498`` (lambda_fracAGN="0/0")
        # via cross-component energy coupling. That coupling *is* the AGN
        # power here, so there is no ``agn_log_lbol`` to set: the two
        # together are a contradiction the build refuses.
        "agn_ir_frac": Fixed(0.3),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
s_agn = m_agn.predict_state({})

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(
    ax_l,
    ax_r,
    label_l="pcigale  + SKIRTOR2016 (i = 30°, τ_9.7 = 7)",
    label_r="tengri  agn[schartmann disc + skirtor torus + polar BB]",
)
# CIGALE side — full chain (stellar+dust dashed, full +SKIRTOR solid,
# AGN-only differential to show what SKIRTOR adds).
L_skirt_only = np.maximum(L_skirt - U.regrid(w_base, L_base, w_skirt), 1e-50)
ax_l.plot(w_base, L_base, "k--", linewidth=1.0, alpha=0.5, label="stellar + dust")
ax_l.plot(w_skirt, L_skirt, "C0-", linewidth=1.5, alpha=0.7, label="stellar + dust + SKIRTOR")
ax_l.plot(w_skirt, L_skirt_only, "C0:", linewidth=1.5, label="SKIRTOR component only")
ax_l.legend(fontsize=9)
ax_l.grid(True, alpha=0.3)
# tengri side — same three-line layout. ``derived['sed_agn']`` carries
# the disc + torus contribution; ``sed_intrinsic`` is the full SED with
# everything; dashed baseline is the no-AGN build.
L_t_agn_only = np.maximum(np.asarray(s_agn.derived["sed_agn"]), 1e-50)
ax_r.plot(
    s_agn_base.wave,
    s_agn_base.sed_intrinsic,
    "k--",
    linewidth=1.0,
    alpha=0.5,
    label="stellar + dust",
)
ax_r.plot(
    s_agn.wave, s_agn.sed_intrinsic, "C1-", linewidth=1.5, alpha=0.7, label="stellar + dust + AGN"
)
ax_r.plot(
    s_agn.wave, L_t_agn_only, "C1:", linewidth=1.5, label="composable disc + SKIRTOR torus only"
)
ax_r.legend(fontsize=9)
ax_r.grid(True, alpha=0.3)

# Bound both axes to the same windows so the panels are visually
# comparable. Without explicit ``set_xlim``, matplotlib auto-scales
# each panel to its widest trace's native grid — the no-AGN baseline
# (SSP grid, 91 Å – 160 µm) is shorter than the +AGN trace (SKIRTOR
# grid extends X-ray and FIR), so the two panels would show different
# x-spans for reasons that look like data but are really cosmetics.
_xmin_a = float(min(w_skirt.min(), float(np.asarray(s_agn.wave).min())))
_xmax_a = float(max(w_skirt.max(), float(np.asarray(s_agn.wave).max())))
_ymax_a = max(float(np.asarray(L_skirt).max()), float(np.asarray(s_agn.sed_intrinsic).max()))
for ax in (ax_l, ax_r):
    ax.set_xlim(_xmin_a, _xmax_a)
    ax.set_ylim(_ymax_a * 1e-6, _ymax_a * 2)

fig.tight_layout()
save_fig("cigale_09_agn_skirtor.png")


# %% [markdown]
# ### §9c AGN parity — full-spectrum ratio
#
# The two `stellar + dust + AGN` SEDs of §9 on a shared grid, read out as
# energy-weighted tengri / CIGALE ratios in three bands (1500–3000 Å, 5–30 μm and
# 100 μm) and then component by component as integrated luminosities. A band
# ratio locates a residual in wavelength but cannot say which component carries
# it: the 1500–3000 Å band is mostly host starlight (full SED 1.027×, host alone
# 1.061×, disc alone 1.009×), and at 100 μm all three AGN components overlap with
# the host. The component ladder separates them. Both codes publish disc, torus
# and polar dust separately. The printed ladder gives the three luminosity
# ratios (about 1 % each), the emergent disc shape deviation and each code's
# polar share.

# %%
w_c_agn, L_c_agn = w_skirt, np.asarray(L_skirt)  # CIGALE stellar+dust+SKIRTOR
w_t_agn, L_t_agn = np.asarray(s_agn.wave), np.asarray(s_agn.sed_intrinsic)  # tengri full
L_t_on_c_agn = U.regrid(w_t_agn, L_t_agn, w_c_agn)
ratio_agn = L_t_on_c_agn / np.maximum(L_c_agn, 1e-50)

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax.plot(w_c_agn, L_c_agn, "C0-", linewidth=1.6, label="CIGALE  skirtor2016")
ax.plot(w_c_agn, L_t_on_c_agn, "C1--", linewidth=1.4, label="tengri  (regridded)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e3, 1e7)
_ymx_agn = float(max(np.nanmax(L_c_agn), np.nanmax(L_t_on_c_agn)))
ax.set_ylim(_ymx_agn * 1e-5, _ymx_agn * 2.0)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title("AGN parity — CIGALE skirtor2016 vs tengri (stellar + dust + AGN)")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax_r.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_r.axhline(1.0, color="0.5", linewidth=0.8)
ax_r.plot(w_c_agn, ratio_agn, "C1-", linewidth=1.0)
ax_r.set_xscale("log")
ax_r.set_ylim(0.5, 1.5)
ax_r.set_xlabel(r"$\lambda$ [Å]")
ax_r.set_ylabel("tengri / CIGALE", fontsize=9)
ax_r.grid(True, alpha=0.3)

# Per-band AGN ratios (median over each AGN-dominated window).
print("§9 AGN per-band parity (tengri / CIGALE, full stellar+dust+AGN SED):")
for _name, _lo, _hi in [
    ("full SED 1500-3000 Å (host + disc)", 1500.0, 3000.0),
    ("full SED 5-30 µm (torus + host)   ", 5.0e4, 3.0e5),
    ("full SED 100 µm (polar + host)    ", 8.0e5, 1.2e6),
]:
    _m = (w_c_agn >= _lo) & (w_c_agn <= _hi) & (L_c_agn > 0)
    if _m.any():
        print(f"  {_name}: {_band_energy_ratio(w_c_agn[_m], L_c_agn[_m], w_t_agn, L_t_agn, _lo, _hi):.3f}×")
# The full SED in the ultraviolet is mostly host starlight, so its ratio is the
# host's, not the disc's. The disc component alone, from each code's own
# published disc spectrum:
_w_disc_c, _L_disc_c = U.wnm_to_erg_per_hz_per_aa(
    np.asarray(sed_skirtor.wavelength_grid),
    np.asarray(sed_skirtor.luminosities["agn.SKIRTOR2016_disk"]),
)
print(
    "  disc component alone 1500-3000 Å: "
    f"{_band_energy_ratio(_w_disc_c, _L_disc_c, np.asarray(s_agn.wave), np.asarray(s_agn.derived['sed_agn_disc']), 1500.0, 3000.0):.3f}×"
)
print(
    "  host component alone 1500-3000 Å (stellar + dust, no AGN): "
    f"{_band_energy_ratio(w_base, L_base, np.asarray(s_agn_base.wave), np.asarray(s_agn_base.sed_intrinsic), 1500.0, 3000.0):.3f}×"
)


# A band ratio locates a residual in wavelength; it does not say which
# component carries it, and in the FIR three of them overlap. CIGALE separates
# its SKIRTOR contributions (``agn.SKIRTOR2016_disk`` / ``_torus`` /
# ``_polar_dust``) and tengri publishes the same three on ``state.derived``,
# so the comparison can be made component by component instead.
def _agn_component_pair(sed_c, state_t, cig_key, tengri_key):
    """(CIGALE, tengri) ∫L_ν dν for one AGN component, on their shared support.

    Both codes are integrated over the same wavelength points — CIGALE's grid,
    masked to where both sides are non-zero — so the ratio is a luminosity
    comparison and not a difference in grid extent.
    """
    _wc, _Lc = U.wnm_to_erg_per_hz_per_aa(
        np.asarray(sed_c.wavelength_grid), np.asarray(sed_c.luminosities[cig_key])
    )
    _Lt_on_c = U.regrid(np.asarray(state_t.wave), np.asarray(state_t.derived[tengri_key]), _wc)
    _m = (_Lc > 0) & (_Lt_on_c > 0)
    if int(_m.sum()) < 2:
        return float("nan"), float("nan")
    return _lbol_nu(_wc[_m], _Lc[_m]), _lbol_nu(_wc[_m], _Lt_on_c[_m])


_AGN_COMPONENTS = (
    ("disc      ", "agn.SKIRTOR2016_disk", "sed_agn_disc"),
    ("torus     ", "agn.SKIRTOR2016_torus", "sed_agn_torus"),
    ("polar dust", "agn.SKIRTOR2016_polar_dust", "sed_agn_polar"),
)


def _disc_shape_dev(sed_c, state_t, lo=1000.0, hi=5000.0):
    """max |Δ| between the emergent disc shapes, each set to 1 at 2500 Å.

    Normalizing at 2500 Å removes the luminosity scale, which the component
    ladder reports separately. What is left is the *shape* of the disc that
    emerges — the analytic disc times the polar-dust screen, because both
    codes redden the disc before publishing it.
    """
    _wc, _Lc = U.wnm_to_erg_per_hz_per_aa(
        np.asarray(sed_c.wavelength_grid),
        np.asarray(sed_c.luminosities["agn.SKIRTOR2016_disk"]),
    )
    _Lt = U.regrid(np.asarray(state_t.wave), np.asarray(state_t.derived["sed_agn_disc"]), _wc)
    _j = int(np.argmin(np.abs(_wc - 2500.0)))
    _r = (_Lt / _Lt[_j]) / (_Lc / _Lc[_j])
    _m = (_wc >= lo) & (_wc <= hi) & (_Lc > 0) & (_Lt > 0)
    return float(np.abs(_r[_m] - 1.0).max()), int(_m.sum())


def _print_agn_report(header, sed_c, state_t):
    """Print the disc-shape deviation, the three component ratios and the polar share.

    The polar share -- polar / (polar + torus), each code against its own
    budget -- is what makes the polar component's disc dependence visible: it
    is a property of one code, not a comparison between them, so §9 and §9b
    printing different values for the *same* code is the disc moving it.
    """
    print(header)
    _dev, _n = _disc_shape_dev(sed_c, state_t)
    print(
        f"  disc shape, 1000–5000 Å, normalized at 2500 Å: max |Δ| = {_dev * 100:.2f}% ({_n} pts)"
    )
    _pairs = {
        _tk: _agn_component_pair(sed_c, state_t, _ck, _tk) for _, _ck, _tk in _AGN_COMPONENTS
    }
    for _name, _ck, _tk in _AGN_COMPONENTS:
        _bc, _bt = _pairs[_tk]
        print(f"  {_name}: CIGALE {_bc:.4e}, tengri {_bt:.4e} erg/s → {_bt / _bc:.4f}×")
    _tor_c, _tor_t = _pairs["sed_agn_torus"]
    _pol_c, _pol_t = _pairs["sed_agn_polar"]
    print(
        f"  polar share polar/(polar+torus): CIGALE {_pol_c / (_pol_c + _tor_c):.4f}, "
        f"tengri {_pol_t / (_pol_t + _tor_t):.4f}"
    )


_print_agn_report(
    "§9 AGN parity, component by component (∫L_ν dν; schartmann2005 disc):",
    sed_skirtor,
    s_agn,
)
fig.tight_layout()
plt.close(fig)  # superseded by §9e's SKIRTOR (τ, oa, i) grid, which replaces this panel


# %% [markdown]
# ### §9b The other disc — `disc.skirtor` ↔ CIGALE `disk_type=0`
#
# CIGALE's `skirtor2016` offers two analytic discs: `disk_type=0` →
# `skirtor_disk()` and `disk_type=1` → `schartmann2005_disk()`. tengri ships both,
# as `disc.skirtor` and `disc.schartmann2005`; §9 pins the `disk_type=1` pairing
# and this section swaps to `disc.skirtor`, the shallower SKIRTOR disc with its
# 1200 Å bend. Only the disc changes: the torus, the polar-dust block and the
# `fracAGN` coupling are §9's, so the printed ladder repeats §9's with one entry
# moved. The figure overlays the two AGN SEDs as in §9.
#
# **What the disc-shape number contains.** Both codes publish the disc *after*
# the polar-dust screen, so the printed deviation is the analytic disc shape
# and the two implementations of the same Bongiorno screen together. The screen
# does not cancel by comparing this section with §9: the same law integrated
# against a different disc spectrum is a different factor, which is why the polar
# share printed here differs from §9's on *both* sides. §9d prints the polar
# reference term by term.

# %%
sed_skirtor0 = C.run_chain(
    [
        _sfh_args_d,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        NEB_FIDUCIAL_CIGALE,
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        (
            "skirtor2016",
            dict(
                t=7,
                pl=1.0,
                q=1.0,
                oa=40,
                R=20,
                Mcl=0.97,
                i=30,
                disk_type=0,  # ← skirtor_disk ↔ tengri disc.skirtor
                delta=0,
                fracAGN=0.3,
                lambda_fracAGN="0/0",
                law=0,
                EBV=0.03,
                temperature=100.0,
                emissivity=1.6,
            ),
        ),
    ]
)
w_sk0, L_sk0 = C.to_lnu(sed_skirtor0)

m_agn_sk = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        # Same Lyman-limit clip as §9, so the stellar+dust floor under this
        # panel is the one §9 and the dashed baseline below already plot.
        "lyman_cutoff": True,
        "all_params": Fixed(DEFAULT),
    },
    agn={
        "type": "composable",
        "disc": {"type": "skirtor", "all_params": Fixed(DEFAULT)},  # ← the SKIRTOR analytic disc
        "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        # Same polar-dust block and same fracAGN coupling as §9: only the disc
        # changes, so the two panels differ in one input.
        "atten": _POLAR_DUST_BLOCK,
        "agn_ir_frac": Fixed(0.3),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
s_agn_sk = m_agn_sk.predict_state({})

_print_agn_report(
    "§9b AGN parity, component by component (∫L_ν dν; skirtor disc):",
    sed_skirtor0,
    s_agn_sk,
)

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(
    ax_l,
    ax_r,
    label_l="pcigale  + SKIRTOR2016 (disk_type = 0)",
    label_r="tengri  agn[skirtor disc + skirtor torus + polar BB]",
)
L_sk0_only = np.maximum(L_sk0 - U.regrid(w_base, L_base, w_sk0), 1e-50)
ax_l.plot(w_base, L_base, "k--", linewidth=1.0, alpha=0.5, label="stellar + dust")
ax_l.plot(w_sk0, L_sk0, "C0-", linewidth=1.5, alpha=0.7, label="stellar + dust + SKIRTOR (dt=0)")
ax_l.plot(w_sk0, L_sk0_only, "C0:", linewidth=1.5, label="SKIRTOR component only")
ax_l.legend(fontsize=9)
ax_l.grid(True, alpha=0.3)
L_sk_only = np.maximum(np.asarray(s_agn_sk.derived["sed_agn"]), 1e-50)
ax_r.plot(
    s_agn_base.wave,
    s_agn_base.sed_intrinsic,
    "k--",
    linewidth=1.0,
    alpha=0.5,
    label="stellar + dust",
)
ax_r.plot(
    s_agn_sk.wave,
    s_agn_sk.sed_intrinsic,
    "C1-",
    linewidth=1.5,
    alpha=0.7,
    label="stellar + dust + AGN",
)
ax_r.plot(
    s_agn_sk.wave, L_sk_only, "C1:", linewidth=1.5, label="skirtor disc + SKIRTOR torus only"
)
ax_r.legend(fontsize=9)
ax_r.grid(True, alpha=0.3)
_xmin_b = float(min(w_sk0.min(), float(np.asarray(s_agn_sk.wave).min())))
_xmax_b = float(max(w_sk0.max(), float(np.asarray(s_agn_sk.wave).max())))
_ymax_b = max(float(np.asarray(L_sk0).max()), float(np.asarray(s_agn_sk.sed_intrinsic).max()))
for ax in (ax_l, ax_r):
    ax.set_xlim(_xmin_b, _xmax_b)
    ax.set_ylim(_ymax_b * 1e-6, _ymax_b * 2)
fig.tight_layout()
save_fig("cigale_09b_disc_skirtor.png")


# %% [markdown]
# ### §9d What separates the polar reference
#
# Both codes set the polar dust's absorbed power from the same product,
# `g(oa) × R_faceon × J`: the polar cone's share of the disc's anisotropic
# emission, the face-on disc integral measured against the SKIRTOR dust
# integral, and `J = ∫disc·(1 − e^−τ) dλ / ∫disc dλ`, the fraction of the disc
# spectrum the polar screen absorbs. `g(oa)` is the same closed form on both
# sides and cancels from the comparison. The cell prints the other two and the
# two terms their ratio factors into, with the polar screen set to
# `polar_law="bongiorno"` (CIGALE's `law=0`) and, for contrast, to the default
# `polar_law="smc"` (Pei 1992).
#
# `R_faceon` agrees: tengri reads 4.553791 off the vendored SKIRTOR grid against
# pcigale's own 4.562211, a ratio of 0.998154× (−0.18 %). What separates the
# polar reference is the pair of terms printed beneath it.
#
# **The wavelength axis, a convention on pcigale's side.** The vendored grid
# carries 136 wavelengths across 10 Å – 10⁸ Å where pcigale's carries 948, and each
# code integrates on its own. With CIGALE's extinction curve held fixed, the
# absorbed fraction J converges to 0.221723 on a dense axis. On tengri's 136
# points it is 0.221928, +0.09 % from converged, and on pcigale's 948 points
# 0.222842, +0.51 %. tengri's axis is the closer of the two to the integral, so
# this term, 0.994058× (−0.59 %), is mostly pcigale's quadrature error.
#
# **The extinction curve.** With `polar_law="bongiorno"` the two curves coincide
# above 100 nm (the printed A(λ)/E(B−V) ratios are 1.000 at 2500 Å, 5500 Å and
# 1 μm) and differ only in the sub-100 nm SMC-bar shape (0.985 at 912 Å); the
# curve term is 1.002761× (+0.28 %). The default `polar_law="smc"` is a different
# curve (A(λ)/E(B−V) ratios 1.076, 1.011, 0.972 and 1.207 at 912 Å, 2500 Å,
# 5500 Å and 1 μm) and its curve term is 0.976712× (−2.33 %). Use
# `polar_law="bongiorno"` to compare with CIGALE's default polar dust.
#
# The two terms multiply to 0.996803× with `polar_law="bongiorno"`: tengri sets
# its polar dust from 0.32 % less absorbed disc power than pcigale does. With
# the default `polar_law="smc"` the product is 0.970908×.

# %%
from pcigale.data import SimpleDatabase as _CigaleDB
from pcigale.sed_modules.skirtor2016 import k_ext as _cigale_k_ext
from pcigale.sed_modules.skirtor2016 import schartmann2005_disk as _cigale_disc

from tengri.components.agn.disc_cigale import schartmann2005_disk_spectrum
from tengri.components.agn.polar_dust import bongiorno2012_extinction_curve
from tengri.components.agn.skirtor import load_skirtor_bundle, skirtor_disc_dust_ratio
from tengri.components.dust.attenuation import smc as _tengri_smc

_OA_SKIRTOR, _EBV_POLAR, _RV_SMC = 40.0, 0.03, 2.93
_COS_I_FIDUCIAL = float(np.cos(np.deg2rad(30.0)))

# tengri's face-on reference, straight off the vendored grid. The tie is given
# an unreddened disc (`ext_fac = 1`), exactly as the composable AGN calls it:
# the polar screen enters through J below, not through R_faceon.
_w_tie = np.logspace(1.0, 7.0, 20000)
_tie = skirtor_disc_dust_ratio(
    _w_tie,
    schartmann2005_disk_spectrum(_w_tie / 10.0),  # the disc spectrum takes nm
    np.ones_like(_w_tie),
    _template=load_skirtor_bundle().disc_dust,
    agn_oa_skirtor=_OA_SKIRTOR,
    agn_cos_inc=_COS_I_FIDUCIAL,
)
_R_faceon_t = float(_tie.R_faceon)
_wave_native = np.asarray(_tie.wave_native)
_disc_shape_t = np.asarray(_tie.faceon_shape_native)

# pcigale's own, rebuilt from its database the way skirtor2016._init_code does:
# the i = 0 record carried onto the observer record's luminosity scale by
# norm(0)/norm(i), divided by that record's dust integral.
with _CigaleDB("skirtor2016") as _db:
    _sk_i = _db.get(t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97, i=30)
    _sk_0 = _db.get(t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97, i=0)
_wave_c_nm = np.asarray(_sk_i.wl)
_R_faceon_c = float(
    np.trapezoid(np.asarray(_sk_0.disk) * (_sk_0.norm / _sk_i.norm), x=np.asarray(_sk_0.wl))
    / np.trapezoid(np.asarray(_sk_i.dust), x=_wave_c_nm)
)

# The curves on one dense axis. `k_ext` rescales its sub-100 nm branch to meet
# the power law at the last grid point below that boundary, so it has to be
# evaluated once on an axis that resolves the boundary rather than sampled point
# by point. tengri's `polar_law="bongiorno"` is the curve the notebook's builds
# use; the default `polar_law="smc"` (Pei 1992 SMC bar, R_V = 2.93) is shown for
# contrast.
_w_dense = np.logspace(1.0, 6.0, 20001)
_k_cigale_dense = _cigale_k_ext(_w_dense / 10.0, 0)
_k_tengri_dense = np.asarray(bongiorno2012_extinction_curve(jnp.asarray(_w_dense)))
_k_pei_dense = np.asarray(_tengri_smc(_w_dense)) * _RV_SMC


def _absorbed_fraction(disc_shape, wave_aa, k_lambda):
    """J = ∫disc·(1 − 10^(−0.4·k·E(B−V))) dλ / ∫disc dλ [dimensionless]."""
    _transmitted = 10.0 ** (-0.4 * k_lambda * _EBV_POLAR)
    return float(
        np.trapezoid(disc_shape * (1.0 - _transmitted), wave_aa)
        / np.trapezoid(disc_shape, wave_aa)
    )


_J_t = _absorbed_fraction(
    _disc_shape_t, _wave_native, np.interp(_wave_native, _w_dense, _k_tengri_dense)
)
_J_pei = _absorbed_fraction(
    _disc_shape_t, _wave_native, np.interp(_wave_native, _w_dense, _k_pei_dense)
)
# CIGALE's curve on tengri's axis: the control that separates the two terms.
_J_c_on_native = _absorbed_fraction(
    _disc_shape_t, _wave_native, np.interp(_wave_native, _w_dense, _k_cigale_dense)
)
_J_c = _absorbed_fraction(
    _cigale_disc(_wave_c_nm), _wave_c_nm * 10.0, _cigale_k_ext(_wave_c_nm, 0)
)

_term_quadrature = (_R_faceon_t * _J_c_on_native) / (_R_faceon_c * _J_c)
_term_curve = _J_t / _J_c_on_native
_polar_reference_ratio = (_R_faceon_t * _J_t) / (_R_faceon_c * _J_c)

print("§9d polar reference g(oa)·R_faceon·J, tengri against pcigale (i = 30°):")
print(
    f"  R_faceon = ∫disc(face-on)/∫dust: tengri {_R_faceon_t:.6f}, "
    f"pcigale {_R_faceon_c:.6f} → {_R_faceon_t / _R_faceon_c:.6f}× "
    f"({(_R_faceon_t / _R_faceon_c - 1) * 100:+.2f}%)"
)
print(
    f"  wavelength axis: vendored grid {_wave_native.size} points, "
    f"pcigale {_wave_c_nm.size}, same 10 Å–10⁸ Å span"
)
print(
    f"  J absorbed fraction: tengri {_J_t:.6f}, pcigale {_J_c:.6f} "
    f"(CIGALE's curve on the vendored axis: {_J_c_on_native:.6f})"
)
print(
    f"  term 1, wavelength quadrature: {_term_quadrature:.6f}× "
    f"({(_term_quadrature - 1) * 100:+.2f}%)"
)
print(
    f"  term 2, polar_law='bongiorno' extinction curve: {_term_curve:.6f}× "
    f"({(_term_curve - 1) * 100:+.2f}%)"
)
print(
    f"  product {_term_quadrature * _term_curve:.6f}× = polar reference ratio "
    f"{_polar_reference_ratio:.6f}× ({(_polar_reference_ratio - 1) * 100:+.2f}%)"
)
_CURVE_POINTS = ((912.0, "912 Å"), (2500.0, "2500 Å"), (5500.0, "5500 Å"), (1e4, "1 µm"))
for _curve_name, _k_dense in (
    ("bongiorno", _k_tengri_dense),
    ("smc (Pei 1992, default)", _k_pei_dense),
):
    print(
        f"  A(λ)/E(B−V), pcigale / tengri polar_law='{_curve_name}': "
        + ", ".join(
            f"{float(np.interp(_x, _w_dense, _k_cigale_dense)) / float(np.interp(_x, _w_dense, _k_dense)):.3f} at {_label}"
            for _x, _label in _CURVE_POINTS
        )
    )
# Which quadrature is closer to the integral? CIGALE's extinction curve on three
# axes: tengri's vendored axis carrying tengri's face-on disc shape (`_J_c_on_native`
# above), pcigale's own axis carrying pcigale's disc, and a dense axis for each
# code's analytic disc, which gives the converged value.
_w_conv = np.logspace(1.0, 8.0, 400001)
_k_conv = _cigale_k_ext(_w_conv / 10.0, 0)
_J_conv_c = _absorbed_fraction(_cigale_disc(_w_conv / 10.0), _w_conv, _k_conv)
_J_conv_t = _absorbed_fraction(
    np.asarray(schartmann2005_disk_spectrum(_w_conv / 10.0)), _w_conv, _k_conv
)
print(
    f"  J with CIGALE's curve, converged: CIGALE's disc {_J_conv_c:.6f}, tengri's disc {_J_conv_t:.6f}"
)
print(
    f"  J on tengri's {_wave_native.size}-point axis {_J_c_on_native:.6f} "
    f"({(_J_c_on_native / _J_conv_t - 1) * 100:+.2f}% from converged); on pcigale's "
    f"{_wave_c_nm.size}-point axis {_J_c:.6f} ({(_J_c / _J_conv_c - 1) * 100:+.2f}%)"
)
assert abs(_J_c_on_native / _J_conv_t - 1.0) < abs(_J_c / _J_conv_c - 1.0)
print(
    f"  for contrast, the default polar_law='smc': J = {_J_pei:.6f}, term 2 = "
    f"{_J_pei / _J_c_on_native:.6f}× ({(_J_pei / _J_c_on_native - 1) * 100:+.2f}%), "
    f"polar reference ratio {(_R_faceon_t * _J_pei) / (_R_faceon_c * _J_c):.6f}×"
)


# %% [markdown]
# ### §9e AGN dust grids: SKIRTOR (τ, oa, i) and Fritz
#
# CIGALE modules `skirtor2016` and `fritz2006` ↔ tengri's `torus.skirtor` and
# `torus.fritz` with the same polar-dust block as §9. Each panel compares AGN dust
# emission, torus plus polar screen summed; the two figures overlay the sums and
# show the tengri/CIGALE ratio. The sum is independent of how each code
# partitions the two: pcigale subtracts the polar re-emission from the torus dust,
# while tengri rescales the torus against a shared budget and carries the polar
# term separately.
#
# SKIRTOR: τ_9.7 ∈ {3, 7, 11}, oa ∈ {20°, 40°, 60°}, i ∈ {0°, 30°, 70°}, each
# varied one at a time from §9's fiducial (7, 40°, 30°). The printed IR-band
# table gives a median ratio of 1.011–1.012× in every case, with no band outside
# 5 % and a worst band of at most 1.013×: optical depth, opening angle and
# inclination all agree.
#
# Fritz 2006: (τ, opening angle, ψ) ∈ {(1, 60°, 0.001°), (1, 100°, 40.1°),
# (6, 60°, 89.99°)}, with r_ratio=60, β=−0.5, γ=4, a Schartmann disc,
# fracAGN=0.3 and EBV=0.03. `agn_cos_inc` is the one inclination: the library's
# viewing elevation is ψ = 90° − i, so the cell passes `agn_cos_inc = sin ψ`.
# CIGALE's opening angle is the full angle and maps to tengri's half-angle via
# `agn_fritz_oa = (180 − opening_angle) / 2`. Under `cigale_joint` the Fritz torus
# carries the same `agn_ir_frac` tie as SKIRTOR. CIGALE's `fritz2006` carries a
# disc-slope parameter δ whose default is −0.36; the cell passes `delta=0.0`, which
# is the Schartmann disc of §9 and of the SKIRTOR runs, so the disc under the torus is
# the same on both sides. The printed table gives band medians of
# 1.009–1.010×, worst bands of 0.973–0.975× and
# none of 12 IR bands outside 5 % in the three cases.

# %%
_AGN_SFH_9E = {
    "type": "delayed",
    "tau_gyr": Fixed(1.0),
    "age_gyr": Fixed(5.0),
    "log_total_mass": Fixed(0.0),
    "all_params": Fixed(DEFAULT),
}
_AGN_DUSTATT_9E = {
    "type": "two_component",
    "law_bc": "leitherer02",
    "law_diff": "leitherer02",
    "tau_bc": Fixed(TAU_BC_FIDUCIAL),
    "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
    "lyman_cutoff": True,
    "all_params": Fixed(DEFAULT),
}


def _cigale_skirtor_sed(t, oa, i):
    return C.run_chain(
        [
            _sfh_args_d,
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            NEB_FIDUCIAL_CIGALE,
            ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
            (
                "skirtor2016",
                dict(
                    t=t,
                    pl=1.0,
                    q=1.0,
                    oa=oa,
                    R=20,
                    Mcl=0.97,
                    i=i,
                    disk_type=1,
                    delta=0,
                    fracAGN=0.3,
                    lambda_fracAGN="0/0",
                    law=0,
                    EBV=0.03,
                    temperature=100.0,
                    emissivity=1.6,
                ),
            ),
        ]
    )


m_skirtor_sweep = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh=_AGN_SFH_9E,
    dust_attenuation=_AGN_DUSTATT_9E,
    agn={
        "type": "composable",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {
            "type": "skirtor",
            "tau_skirtor": Uniform(3.0, 11.0, default=7.0),
            "oa_skirtor": Uniform(10.0, 80.0, default=40.0),
            "cos_inc": Uniform(0.0, 1.0, default=float(np.cos(np.deg2rad(30.0)))),
            "all_params": Fixed(DEFAULT),
        },
        "atten": _POLAR_DUST_BLOCK,
        "agn_ir_frac": Fixed(0.3),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
p_sk_sweep = dict(m_skirtor_sweep.spec.sample(jax.random.PRNGKey(0)))

_skirtor_grid = [
    (3, 40, 30),
    (7, 40, 30),
    (11, 40, 30),
    (7, 20, 30),
    (7, 60, 30),
    (7, 40, 0),
    (7, 40, 70),
]
_cases_9e_sk = []
for _tau, _oa, _i in _skirtor_grid:
    _sed_c9e = _cigale_skirtor_sed(_tau, _oa, _i)
    _w_c9e, _L_torus_c9e = C.to_lnu_contribution(_sed_c9e, "agn.SKIRTOR2016_torus")
    _, _L_polar_c9e = C.to_lnu_contribution(_sed_c9e, "agn.SKIRTOR2016_polar_dust")
    _L_c9e = _L_torus_c9e + _L_polar_c9e  # Sum torus + polar (partition-independent)
    _pred_9e = m_skirtor_sweep.predict(
        {
            **p_sk_sweep,
            "agn_tau_skirtor": jnp.float64(_tau),
            "agn_oa_skirtor": jnp.float64(_oa),
            "agn_cos_inc": jnp.float64(np.cos(np.deg2rad(_i))),
        }
    )
    _w_t_9e = np.asarray(_pred_9e.sed.components["wavelength"])
    _L_torus_t_9e = np.asarray(_pred_9e.sed.components["sed_agn_torus"])
    _L_polar_t_9e = np.asarray(_pred_9e.sed.components["sed_agn_polar"])
    _L_t_9e = _L_torus_t_9e + _L_polar_t_9e  # Sum torus + polar
    _cases_9e_sk.append((f"τ={_tau}, oa={_oa}°, i={_i}°", _w_c9e, _L_c9e, _w_t_9e, _L_t_9e))
    _assert_comparable(_L_c9e, _L_t_9e, name=f"§9e skirtor {_tau},{_oa},{_i}")

fig, (ax, ax_r), _ratios_9e_sk = V.sweep_fig(
    _cases_9e_sk, ref_label="CIGALE", title="§9e SKIRTOR (τ, oa, i) grid", xlim=(1e3, 1e7)
)
fig.tight_layout()
save_fig("cigale_09e_skirtor_grid.png")
print("§9e SKIRTOR (τ, oa, i) grid (IR_BANDS):")
for _label, _w_ref, _L_ref, _w_t, _L_t in _cases_9e_sk:
    _rows = V.filter_rows_native(
        np.asarray(_w_t), np.asarray(_L_t), _w_ref, _L_ref, filters=V.IR_BANDS
    )
    V.print_filter_table(_rows, ref_name="CIGALE", title=f"§9e {_label}", compact=True)


def _cigale_fritz_sed(tau, oa, psy):
    return C.run_chain(
        [
            _sfh_args_d,
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            NEB_FIDUCIAL_CIGALE,
            ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
            (
                "fritz2006",
                dict(
                    r_ratio=60.0,
                    tau=tau,
                    beta=-0.5,
                    gamma=4.0,
                    opening_angle=oa,
                    psy=psy,
                    disk_type=1,
                    # δ = 0 matches the Schartmann disc and the SKIRTOR runs; the
                    # module's own default is −0.36.
                    delta=0.0,
                    fracAGN=0.3,
                    EBV=0.03,
                    temperature=100.0,
                    emissivity=1.6,
                ),
            ),
        ]
    )


# Under `agn_norm='cigale_joint'` the Fritz torus carries the same fracAGN tie as
# SKIRTOR: `agn_ir_frac = 0.3` sets P_AGN = L_absorbed x f/(1-f) and ties the disc
# to the library's disc-to-dust ratio at the viewing angle, as CIGALE's
# `fritz2006` does.
m_fritz_sweep = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh=_AGN_SFH_9E,
    dust_attenuation=_AGN_DUSTATT_9E,
    agn={
        "type": "composable",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {
            "type": "fritz",
            "fritz_tau": Uniform(0.1, 10.0, default=1.0),
            "fritz_oa": Uniform(20.0, 60.0, default=60.0),
            "agn_cos_inc": Uniform(0.0, 1.0, default=0.7071),
            "fritz_r_ratio": Fixed(60.0),
            "fritz_beta": Fixed(-0.5),
            "fritz_gamma": Fixed(4.0),
            "all_params": Fixed(DEFAULT),
        },
        "atten": _POLAR_DUST_BLOCK,
        "agn_ir_frac": Fixed(0.3),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
p_fritz_sweep = dict(m_fritz_sweep.spec.sample(jax.random.PRNGKey(0)))

_fritz_grid = [(1.0, 60.0, 0.001), (1.0, 100.0, 40.1), (6.0, 60.0, 89.99)]
_cases_9e_fr = []
for _tau, _oa, _psy in _fritz_grid:
    _sed_c9f = _cigale_fritz_sed(_tau, _oa, _psy)
    _w_c9f, _L_torus_c9f = C.to_lnu_contribution(_sed_c9f, "agn.fritz2006_torus")
    _, _L_polar_c9f = C.to_lnu_contribution(_sed_c9f, "agn.fritz2006_polar_dust")
    _L_c9f = _L_torus_c9f + _L_polar_c9f  # Sum torus + polar (partition-independent)
    _oa_half = (180.0 - _oa) / 2.0
    _pred_9f = m_fritz_sweep.predict(
        {
            **p_fritz_sweep,
            "agn_fritz_tau": jnp.float64(_tau),
            "agn_fritz_oa": jnp.float64(_oa_half),
            # One inclination: the library elevation psi = 90 deg - i, cos i = sin psi.
            "agn_cos_inc": jnp.float64(np.sin(np.radians(_psy))),
        }
    )
    _w_t_9f = np.asarray(_pred_9f.sed.components["wavelength"])
    _L_torus_t_9f = np.asarray(_pred_9f.sed.components["sed_agn_torus"])
    _L_polar_t_9f = np.asarray(_pred_9f.sed.components["sed_agn_polar"])
    _L_t_9f = _L_torus_t_9f + _L_polar_t_9f  # Sum torus + polar
    _cases_9e_fr.append((f"τ={_tau}, oa={_oa}°, ψ={_psy}°", _w_c9f, _L_c9f, _w_t_9f, _L_t_9f))
    _assert_comparable(_L_c9f, _L_t_9f, name=f"§9e fritz {_tau},{_oa},{_psy}")

fig, (ax, ax_r), _ratios_9e_fr = V.sweep_fig(
    _cases_9e_fr,
    ref_label="CIGALE",
    title="§9e Fritz 2006 (τ, opening angle, ψ) grid",
    xlim=(1e3, 1e7),
)
fig.tight_layout()
save_fig("cigale_09f_fritz_grid.png")
print("§9e Fritz 2006 (τ, opening angle, ψ) grid (IR_BANDS):")
for _label, _w_ref, _L_ref, _w_t, _L_t in _cases_9e_fr:
    _rows = V.filter_rows_native(
        np.asarray(_w_t), np.asarray(_L_t), _w_ref, _L_ref, filters=V.IR_BANDS
    )
    V.print_filter_table(_rows, ref_name="CIGALE", title=f"§9e {_label}", compact=True)


# %% [markdown]
# ## §10 X-ray
#
# CIGALE module `xray` (Yang et al. 2020: an AGN corona power law plus X-ray
# binaries and hot gas) ↔ tengri `xray={"type": "yang20"}`. The figure overlays
# the two coronas against photon energy, with the tengri/CIGALE ratio below.
# Three conventions are matched: α_ox is derived from L_2500 (Just et al. 2007,
# `L_2keV = L_2500 · 10^(α_ox/0.3838)`) on both sides at log₁₀ L_2500 = 29.47;
# inclination follows Yang+2020 at i = 30° (`agn_cos_inc`); and N_H is pinned to
# zero on both sides. The disc power is set through `agn_log_lbol` (the
# angle-integrated accretion power), solved so that both codes' intrinsic L_2500
# agree to the printed digits. At `log N_H = 0` tengri's Ricci et al. (2017)
# scattered floor is 1 % of an unabsorbed corona and is not what the printed
# residual is. The cell prints the ratio at 2 keV and as a median over
# 0.5–10 keV.
#
# **The binary and hot-gas templates.** pcigale normalizes its hot-gas (0.5–2 keV),
# LMXB (2–10 keV) and HMXB (0.5–8 keV) templates with a trapezoid sum over its own
# wavelength grid, restricted to the grid points inside each band
# (`pcigale/sed_modules/yang20.py`). The band edges fall between grid points, so the
# band integral CIGALE normalizes to is a truncated one. The 0.99× of the capstone's
# X-ray rows, where the galaxy carries these templates and no corona, is that
# convention.
#
# **Verification Status:** PARTIAL (3/16): Radio, X-ray and AGN

# %%
_LOG_L2500_TARGET = (2.638 + 1.4) / 0.137  # Just+2007 == CIGALE's -1.4


def _cigale_xray(sfr_a, incl):
    return C.run_chain(
        [
            (
                "sfhdelayed",
                dict(
                    tau_main=1000,
                    age_main=5000,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=sfr_a,
                    normalise=False,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            NEB_FIDUCIAL_CIGALE,
            ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
            ("dale2014", dict(alpha=2.0)),
            (
                "skirtor2016",
                dict(
                    t=7,
                    pl=1.0,
                    q=1.0,
                    oa=40,
                    R=20,
                    Mcl=0.97,
                    i=incl,
                    disk_type=1,
                    delta=-0.36,
                    fracAGN=0.3,
                    law=0,
                    EBV=0.0,
                    temperature=100,
                    emissivity=1.6,
                ),
            ),
            (
                "yang20",
                dict(
                    gam=1.8,
                    E_cut=300.0,
                    alpha_ox=-1.4,
                    max_dev_alpha_ox=0.2,
                    angle_coef="0.5 & 0",
                    det_lmxb=0.0,
                    det_hmxb=0.0,
                ),
            ),
        ]
    )


def _cigale_corona(sed):
    """CIGALE's corona-only component in tengri units (erg/s/Hz, Å)."""
    return U.wnm_to_erg_per_hz_per_aa(
        np.asarray(sed.wavelength_grid), np.asarray(sed.luminosities["xray.agn"])
    )


def _cigale_l2500(sed):
    """Intrinsic disc L_2500 [erg/s/Hz]; CIGALE stores W/Hz."""
    return float(sed.info["agn.intrin_Lnu_2500A_30deg"]) * 1e7


def _tengri_xray(log_lbol, cos_inc):
    return SEDModel.build(
        ssp_data=ssp,
        met=MET_FIDUCIAL,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "leitherer02",
            "law_diff": "leitherer02",
            "tau_bc": Fixed(TAU_BC_FIDUCIAL),
            "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
            # Lyman-limit clip (CIGALE parity) — match §5/§6
            "lyman_cutoff": True,
            "all_params": Fixed(DEFAULT),
        },
        agn={
            "type": "composable",
            "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
            # Polar dust off, matching the ``EBV=0.0`` in this section's
            # skirtor2016 call — stated rather than left to omission, since it
            # is the one AGN input §9 and §10 deliberately differ on.
            # ``type='none'`` and not ``polar_dust`` with ebv = 0: the block
            # refuses to be selected with no extinction to apply.
            "atten": {"type": "none"},
            # No ``agn_ir_frac`` here: this section solves for the disc power
            # directly, so ``agn_log_lbol`` is the knob that acts.
            "agn_log_lbol": Fixed(log_lbol),
            "agn_cos_inc": Fixed(cos_inc),
            "all_params": Fixed(DEFAULT),
        },
        xray={"type": "yang20", "log_nh": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        neb=NEB_FIDUCIAL_TENGRI,
        redshift=Fixed(0.0),
    )


_COS30 = float(np.cos(np.radians(30.0)))

# Solve both codes onto log10 L_2500 = 29.47 from one trial run each —
# the disc scales linearly with sfr_A (CIGALE, via fracAGN) and with
# 10**agn_log_lbol (tengri, schartmann2005 has an L_bol-independent shape).
_trial_c = _cigale_xray(1e8, 30)
_SFR_A_XRAY = 1e8 * 10.0**_LOG_L2500_TARGET / _cigale_l2500(_trial_c)
sed_x = _cigale_xray(_SFR_A_XRAY, 30)
w_x, L_x = _cigale_corona(sed_x)
_l2500_c = _cigale_l2500(sed_x)

_trial_t = _tengri_xray(11.5, _COS30).predict_state({})
_AGN_LOG_LBOL_XRAY = 11.5 + float(
    np.log10(_l2500_c / float(np.asarray(_trial_t.derived["L_2500_intrinsic"])))
)
state_x = _tengri_xray(_AGN_LOG_LBOL_XRAY, _COS30).predict_state({})
_l2500_t = float(np.asarray(state_x.derived["L_2500_intrinsic"]))
print(
    f"§10 matched disc L_2500: CIGALE log10={np.log10(_l2500_c):.4f}  "
    f"tengri log10={np.log10(_l2500_t):.4f}"
)

w_t = np.asarray(state_x.wave)
sed_t = np.asarray(state_x.derived["sed_xray"])
e_kev_c = 12.398 / w_x
e_kev_t = 12.398 / w_t

m_c = (e_kev_c >= 0.3) & (e_kev_c <= 300) & (L_x > 0)
_order = np.argsort(e_kev_c[m_c])
_e = e_kev_c[m_c][_order]
_Lc = L_x[m_c][_order]
_Lt = U.regrid(e_kev_t, sed_t, _e)
_ratio = _Lt / np.maximum(_Lc, 1e-300)
_assert_comparable(_Lc, _Lt, name="§10 corona")

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax.plot(_e, _Lc, "C0-", linewidth=1.6, label="CIGALE  xray.agn corona (Yang+2020)")
ax.plot(_e, _Lt, "C1--", linewidth=1.4, label="tengri  xray.yang20 corona")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title(
    r"X-ray corona at matched disc $L_{2500}$ "
    r"($\log_{10} L_{2500} = 29.47$, $i = 30°$)"
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax_r.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_r.axhline(1.0, color="0.5", linewidth=0.8)
ax_r.plot(_e, _ratio, "C1-", linewidth=1.0)
ax_r.set_xscale("log")
ax_r.set_yscale("log")
ax_r.set_ylim(0.5, 2.0)
ax_r.set_xlabel(r"$E$ [keV]")
ax_r.set_ylabel("tengri / CIGALE", fontsize=9)
ax_r.grid(True, alpha=0.3)
_soft = (_e >= 0.5) & (_e <= 10.0)
print(
    f"§10 corona parity (tengri/CIGALE): 2 keV = "
    f"{_ratio[np.argmin(np.abs(_e - 2.0))]:.3f}x, "
    f"median 0.5-10 keV = {float(np.median(_ratio[_soft])):.3f}x"
)
fig.tight_layout()
save_fig("cigale_10_xray_corona.png")


# %% [markdown]
# ### §10b Inclination sweep
#
# The Yang et al. (2022) corona anisotropy `f(μ) = (a₁μ + 1 − a₁) / (1 − 0.13397 a₁)`
# with a₁ = 0.5 equals 1 at the 30° anchor. Both codes apply the
# same f(cos i) tilt; the sweep rescales each angle to hold the
# corona anchor at the α_ox crossing.

# %%
fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
print(" i     f(mu)   log10 L_2500,c   tengri/CIGALE @ 2 keV")
for _i_deg, _col in zip((0, 30, 60, 80), ("C0", "C2", "C3", "C4")):
    # hold the corona anchor at the alpha_ox crossing: fracAGN raises the
    # intrinsic disc as the torus hides it, so rescale sfr_A per angle
    _probe_i = _cigale_xray(_SFR_A_XRAY, _i_deg)
    sed_i = _cigale_xray(_SFR_A_XRAY * 10.0**_LOG_L2500_TARGET / _cigale_l2500(_probe_i), _i_deg)
    w_ci, L_ci = _cigale_corona(sed_i)
    _l25_i = _cigale_l2500(sed_i)
    _mu = float(np.cos(np.radians(_i_deg)))
    st_i = _tengri_xray(
        _AGN_LOG_LBOL_XRAY + float(np.log10(_l25_i / _l2500_c)), _mu
    ).predict_state({})
    w_ti = np.asarray(st_i.wave)
    L_ti = np.asarray(st_i.derived["sed_xray"])
    e_ci = 12.398 / w_ci
    _mm = (e_ci >= 0.3) & (e_ci <= 300) & (L_ci > 0)
    _oo = np.argsort(e_ci[_mm])
    _ee = e_ci[_mm][_oo]
    _LLc = L_ci[_mm][_oo]
    _LLt = U.regrid(12.398 / w_ti, L_ti, _ee)
    ax.plot(_ee, _LLc, color=_col, linewidth=1.5, label=f"i = {_i_deg}°")
    ax.plot(_ee, _LLt, color=_col, linewidth=1.2, linestyle="--")
    ax_r.plot(_ee, _LLt / np.maximum(_LLc, 1e-300), color=_col, linewidth=1.0)
    _f_mu = (0.5 * _mu + 0.5) / (1.0 - 0.13397 * 0.5)
    _r2 = float(_LLt[np.argmin(np.abs(_ee - 2.0))] / _LLc[np.argmin(np.abs(_ee - 2.0))])
    print(f" {_i_deg:2d}    {_f_mu:.4f}   {np.log10(_l25_i):.4f}          {_r2:.4f}")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]")
ax.set_title("Corona vs inclination — CIGALE solid, tengri dashed")
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax_r.axhspan(0.9, 1.1, color="0.85", zorder=0)
ax_r.axhline(1.0, color="0.5", linewidth=0.8)
ax_r.set_xscale("log")
ax_r.set_ylim(0.8, 1.25)
ax_r.set_xlabel(r"$E$ [keV]")
ax_r.set_ylabel("tengri / CIGALE", fontsize=9)
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("cigale_10b_xray_inclination.png")


# %% [markdown]
# ## §11 Radio
#
# CIGALE module `radio` (`qir_sf`, `alpha_sf`) ↔ tengri
# `radio={"sf": {"type": "bell2003", "freefree": False, "nu_ref": "21cm",
# "ir_window": "total"}, "radio_q_ir": 2.5,
# "radio_alpha_sf": 0.8}`. The figure overlays the star-forming radio spectrum
# of both codes; the lower panel is the tengri/CIGALE ratio.
#
# **What `q_IR` calibrates.** pcigale's `radio` module is synchrotron only: its
# `qir_sf` sets the 21 cm *non-thermal* luminosity,
# `L_ref = L_dust / (3.75e12 · 10^q_IR)`, and CIGALE's thermal free-free comes
# from its `nebular` module. tengri's `radio_q_ir` is by default Bell (2003)'s
# *total* 1.4 GHz calibration (synchrotron plus thermal): the synchrotron is the
# non-thermal share `1 − f_th` of that total, and the thermal part comes from the
# nebular continuum or, with `freefree: True`, from the Murphy et al. (2011)
# free-free term. `freefree: False` switches to CIGALE's convention: `q_IR`
# calibrates the non-thermal term alone, the synchrotron carries all of it, and
# the thermal free-free is whatever the nebular continuum supplies. The
# like-for-like comparison below therefore uses `freefree: False`; tengri's default
# (Bell 2003: TIR 8–1000 μm, 1.4 GHz, thermal from the nebular continuum) is
# overlaid as the dash-dot curve.
#
# **Conventions.** tengri's defaults are Bell (2003)'s: `q_IR` defined at 1.4 GHz
# and the IR luminosity integrated over the model's own paper window (TIR,
# 8–1000 μm). pcigale normalizes the relation at 21 cm (1.4276 GHz) and feeds it
# the total dust luminosity. The comparison builds therefore set
# `nu_ref: "21cm"` and `ir_window: "total"` explicitly in the `sf` block.
#
# Both codes anchor the amplitude on the absorbed dust power, so any mismatch in
# `L_absorbed` lands 1:1 in the radio. With the two conventions matched, the flat
# synchrotron ratio is predicted by the energy anchor alone (`L_absorbed` against
# CIGALE's `dust.luminosity`, §6). The cell prints the measured range beside the
# prediction.
#
# With `freefree: False` the synchrotron carries the whole `q_IR`. In tengri's
# default the cell prints the following. The TIR input to the radio block is 0.944
# of the total dust luminosity. The default synchrotron is 0.840 of the
# CIGALE-convention synchrotron at 1.4 GHz, which combines the thermal share
# removed from Bell's total, that smaller TIR input and the 1.4 against
# 1.4276 GHz anchor. The remaining numbers share one base, Bell's TIR total at
# exactly 1.4 GHz: the synchrotron is 0.9033 of it and the nebular continuum
# 0.0583, so (synchrotron + nebular continuum) / Bell
# total is 0.9616. It falls below 1 because the
# thermal share removed from the total, which is the Murphy et al. (2011) term at
# the TIR input (0.0967), exceeds the nebular continuum by
# 0.0384.
#
# The ratio panel also compares the two thermal terms, the nebular continuum's
# free-free tail and the Murphy et al. (2011) term, with CIGALE's synchrotron: a
# calibration comparison between two implementations of the same physics. §11b
# tests the thermal free-free itself, per ionizing photon, against an analytic
# reference.
#
# **Verification Status:** PARTIAL (3/25): Radio, X-ray, IGM and PSD physics

# %%
sed_r = C.run_chain(
    [
        (
            "sfhdelayed",
            dict(
                tau_main=1000,
                age_main=5000,
                tau_burst=50,
                age_burst=20,
                f_burst=0.0,
                sfr_A=1.0,
                normalise=True,
            ),
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        NEB_FIDUCIAL_CIGALE,
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("dale2014", dict(alpha=2.0)),
        ("radio", dict(qir_sf=2.5, alpha_sf=0.8, R_agn=0.0, alpha_agn=0.7)),
    ]
)
# CIGALE's SF radio is a single synchrotron component: isolate it.
w_r, L_r = U.wnm_to_erg_per_hz_per_aa(
    np.asarray(sed_r.wavelength_grid), np.asarray(sed_r.luminosities["radio.sf_nonthermal"])
)

_dust_radio_cfg = {
    "type": "two_component",
    "law_bc": "leitherer02",
    "law_diff": "leitherer02",
    "tau_bc": Fixed(TAU_BC_FIDUCIAL),
    "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
    "lyman_cutoff": True,
    "all_params": Fixed(DEFAULT),
}


def _radio_model(sf_block):
    """The §11 galaxy with one star-forming radio configuration."""
    return SEDModel.build(
        ssp_data=ssp,
        met=MET_FIDUCIAL,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=_dust_radio_cfg,
        dust_emission={
            "type": "dale2014_cigale",
            "alpha_dale": Fixed(2.0),
            "all_params": Fixed(DEFAULT),
        },
        radio={
            "sf": sf_block,
            "agn": {"type": "powerlaw"},
            "radio_q_ir": Fixed(2.5),
            "radio_alpha_sf": Fixed(0.8),
            "all_params": Fixed(DEFAULT),
        },
        neb=NEB_FIDUCIAL_TENGRI,
        redshift=Fixed(0.0),
    )


# CIGALE's convention: q_IR calibrates the non-thermal term alone.
# pcigale normalizes the relation at 21 cm and feeds it the total dust luminosity;
# tengri's defaults are Bell (2003)'s 1.4 GHz and the TIR (8-1000 um) window, so
# both are set explicitly here.
_CIGALE_RADIO_SF = {
    "type": "bell2003",
    "freefree": False,
    "nu_ref": "21cm",
    "ir_window": "total",
}
state_r = _radio_model(_CIGALE_RADIO_SF).predict_state({})
w_t = np.asarray(state_r.wave)
sed_t = np.asarray(state_r.derived["sed_radio"])  # synchrotron carrying the whole q_IR

# tengri's default (Bell 2003: TIR 8-1000 um window, 1.4 GHz): q_IR calibrates Bell's
# total, and the thermal term comes from the nebular continuum.
state_r_tot = _radio_model({"type": "bell2003"}).predict_state({})
sed_t_tot = np.asarray(state_r_tot.derived["sed_radio"])  # (1 - f_th) of the total

# Shared-axis overlay + ratio panel: the main pair is non-thermal on both sides.
fig, ax, ax_r, ratio = U.overlay_ratio_fig(
    w_r,
    L_r,
    w_t,
    sed_t,
    x_of_wave=lambda w: C_AA / w / 1e9,
    xlabel=r"$\nu$ [GHz]",
    title="§11 SF radio: synchrotron calibrated as CIGALE's `qir_sf`",
    label_c="CIGALE  radio.sf_nonthermal (synchrotron only)",
    label_t="tengri  bell2003, freefree=False (q_IR calibrates the synchrotron)",
    xlim=(0.1, 100.0),
    ratio_ylim=(0.5, 2.0),
)
# Widen CIGALE into a translucent band so the thin curves read against it.
for _ln in ax.get_lines():
    if _ln.get_label().startswith("CIGALE"):
        _ln.set(linewidth=4.0, alpha=0.35, solid_capstyle="round")

# tengri's default build: synchrotron is the non-thermal share of Bell's total.
ax.plot(
    C_AA / w_t / 1e9,
    sed_t_tot,
    color="C3",
    ls="-.",
    lw=1.4,
    label="tengri  bell2003 default (TIR window, 1.4 GHz; synchrotron = 1 − f_th of Bell's total)",
)
ax.legend(fontsize=8, frameon=False)
_nu_r = C_AA / w_r / 1e9
_g14 = (_nu_r >= 1.0) & (_nu_r <= 1.5) & (L_r > 0)
print(f"§11 radio tengri/CIGALE median (1.0–1.5 GHz): {float(np.median(ratio[_g14])):.3f}×")

# With the anchor frequency and the IR window matched to pcigale's radio module,
# the synchrotron ratio reduces to the energy balance of §6.
_L_ir_t = float(np.asarray(state_r.derived["L_ir"]))
_f_lir = _L_ir_t / (float(sed_r.info["dust.luminosity"]) * 1e7)
_syn_t_on_c = U.regrid(w_t, sed_t, w_r)
_valid = (_nu_r >= 0.1) & (_nu_r <= 100.0) & (L_r > 0) & np.isfinite(ratio)
_syn_ratio = _syn_t_on_c[_valid] / L_r[_valid]
print(
    f"§11 synchrotron alone (freefree=False), tengri/CIGALE: "
    f"{_syn_ratio.min():.4f}–{_syn_ratio.max():.4f} across 0.1–100 GHz (flat)"
)
print(
    f"§11   predicted from the energy balance alone (§6 L_absorbed / dust.luminosity): "
    f"×{_f_lir:.4f}"
)

# tengri's default build (Bell 2003, TIR 8-1000 um window, 1.4 GHz): the synchrotron is
# a fixed share of Bell's calibrated total, and synchrotron + nebular thermal term is
# compared with that total at 1.4 GHz. Bell's total is formed from the TIR input the
# radio block published for this build (`radio_L_ir_input`), not from the total dust
# luminosity the CIGALE-convention build above uses.
_sed_neb = np.asarray(state_r.derived["sed_nebular"])  # nebular continuum incl. free-free tail
_W_14 = C_AA / 1.4e9  # rest wavelength of exactly 1.4 GHz [Å]


def _at_14ghz(sed):
    """``sed`` log-log interpolated to exactly 1.4 GHz."""
    return float(np.exp(np.interp(np.log(_W_14), np.log(w_t), np.log(np.maximum(sed, 1e-300)))))


_sync_14, _sync_cig_14, _neb_14 = (_at_14ghz(x) for x in (sed_t_tot, sed_t, _sed_neb))
_L_tir_in = float(np.asarray(state_r_tot.derived["radio_L_ir_input"])) * L_SUN  # Lsun → erg/s
_bell_total_14 = _L_tir_in / (3.75e12 * 10.0**2.5)
print(
    f"§11 tengri default (Bell 2003: TIR 8-1000 um, 1.4 GHz): L_TIR input / total dust "
    f"luminosity = {_L_tir_in / _L_ir_t:.4f}"
)
print(
    f"§11   default synchrotron / CIGALE-convention synchrotron = "
    f"{_sync_14 / _sync_cig_14:.4f} at 1.4 GHz (thermal share, TIR window and "
    f"the 1.4 vs 1.4276 GHz anchor together)"
)
print(
    f"§11   (default synchrotron + nebular continuum) / Bell's calibrated total from the "
    f"TIR input, L(1.4 GHz) = {(_sync_14 + _neb_14) / _bell_total_14:.3f}"
)

# The same residual in one base: every term at exactly 1.4 GHz, as a fraction of Bell's
# TIR total there. The Murphy term is evaluated directly at 1.4 GHz.
_sync_f = _sync_14 / _bell_total_14
_neb_f = _neb_14 / _bell_total_14
_murphy_tir = np.asarray(radio_freefree(jnp.asarray([_W_14]), _L_tir_in, T_e=1.0e4, alpha_ff=-0.1))
_murphy_f = float(_murphy_tir[0] / _bell_total_14)
print(
    f"§11   as fractions of Bell's TIR total at 1.4 GHz: synchrotron {_sync_f:.4f}, "
    f"nebular continuum {_neb_f:.4f}, sum {_sync_f + _neb_f:.4f}"
)
print(
    f"§11   thermal share subtracted from the total {1.0 - _sync_f:.4f}; Murphy+2011 term "
    f"at the TIR input {_murphy_f:.4f}; nebular continuum falls short of it by "
    f"{1.0 - _sync_f - _neb_f:.4f}"
)

# Thermal free-free calibration, component-wise against CIGALE's synchrotron: the
# nebular continuum's tail and the Murphy+2011 term (the free-free the radio block
# adds with `freefree: True`) evaluated at the same L_IR.
_sed_murphy = np.asarray(radio_freefree(jnp.asarray(w_t), _L_ir_t, T_e=1.0e4, alpha_ff=-0.1))
_den = np.where(L_r > 0, L_r, np.nan)
_neb_on_c = np.asarray(U.regrid(w_t, _sed_neb, w_r))
_murphy_on_c = np.asarray(U.regrid(w_t, _sed_murphy, w_r))
ax_r.plot(_nu_r, _neb_on_c / _den, color="C2", ls="--", lw=1.2, label="nebular tail / CIGALE")
ax_r.plot(_nu_r, _murphy_on_c / _den, color="C3", ls="-.", lw=1.2, label="Murphy+2011 / CIGALE")
ax_r.axhline(
    _f_lir,
    color="0.25",
    ls=":",
    lw=1.6,
    label=f"synchrotron only (energy balance = {_f_lir:.3f})",
)
ax_r.legend(fontsize=7, frameon=False, loc="upper left")
_nu_valid = np.asarray(_nu_r[_valid])
_neb_valid = _neb_on_c[_valid] / L_r[_valid]
_murphy_valid = _murphy_on_c[_valid] / L_r[_valid]
print("§11 thermal free-free, component-wise, as a fraction of CIGALE's synchrotron:")
for _f in (0.15, 1.4, 10.0, 100.0):
    _j = int(np.argmin(np.abs(_nu_valid - _f)))
    print(
        f"    {float(_nu_valid[_j]):6.2f} GHz: nebular tail ×{float(_neb_valid[_j]):.3f}, "
        f"Murphy+2011 ×{float(_murphy_valid[_j]):.3f}"
    )
fig.tight_layout()
save_fig("cigale_11_radio_synchrotron.png")


# %% [markdown]
# ### §11b Thermal free-free: four tests
#
# Beyond 1 mm the nebular continuum is thermal free-free, and in the full SED at the end
# of the notebook it is where the two codes differ by more than the synchrotron's 1 %.
# Four tests separate what is Q_H, what is the emitter and what is the radio block, on
# the §3 galaxy (BC03, logU = −2, Z = 0.02, f_esc = 0). Each carries an `assert` against
# the band stated here.
#
# **1. Per ionizing photon, Cue against Cloudy.** Each code's free-free continuum divided
# by its own Q_H, at 3 cm, 1 cm, 3 mm and 1 mm. Cue's continuum per photon over CIGALE's
# is 1.04–1.08× (1.044× at 3 cm, 1.077× at 1 cm, 1.075× at 3 mm and 1 mm), asserted to lie in 0.95–1.12×. The nebular continuum
# ratio of the full SED is this ratio times §3's Q_H ratio, 1.150×: the excess there is
# the age convention and the emitter, and not a normalization error in either code, since
# lines and continuum share one Q_H.
#
# **2. An analytic reference.** The optically thin free-free luminosity of a hydrogen
# plasma at 10⁴ K per ionizing photon, in case B, is the thermal bremsstrahlung
# emissivity of Rybicki & Lightman (1979, §5.2, Eqs. 5.14a and 5.14b),
# $\varepsilon_\nu = \frac{2^5 \pi e^6}{3 m_e c^3}\left(\frac{2\pi}{3 k m_e}\right)^{1/2}
# T^{-1/2}\, n_e n_p\, e^{-h\nu/kT}\, \bar g_{\rm ff}$, divided by the recombination
# rate $\alpha_B n_e n_p$ that sets Q_H. The cell evaluates the prefactor from CODATA
# constants, 6.84e-38 in cgs against the 6.8e-38 of their Eq. 5.14b. The Gaunt
# factor is the fit of Draine (2011); the cell checks it against the Hummer (1988) values
# tabulated by Dickinson, Davies & Davis (2003, Table 2) at 10⁴ K, and it agrees to 1.2 %
# at 0.4 GHz, 0.9 % at 1.4 GHz and 0.5 % at 70 GHz. $\alpha_B$ is the Hui & Gnedin (1997)
# fit to the Ferland et al. (1992) case B coefficient, 2.592e-13 cm³ s⁻¹ at 10⁴ K.
# CIGALE's continuum per photon is 0.78–0.80× this reference and Cue's is
# 0.83–0.86×; both are asserted to lie in 0.70–1.00×. The
# reference is hydrogen at one temperature, so a gas at another temperature or with a
# different ionization structure sits off it; the cell does not separate those.
# CIGALE's library adds a trend that a hydrogen plasma does not have: its continuum per
# photon at 1 mm falls by 3.08× from Z = 0.0004 to Z = 0.041 (printed). That is a
# property of the Cloudy models; that it comes from dust grains is an inference, as the
# library's physics is not exposed here.
#
# **3. The radio block's free-free.** The term the radio block adds with
# `freefree: True` (Murphy et al. 2011), against Cue's nebular free-free at the same
# Q_H. A history that is nearly constant over 10 Myr (τ = 20 Gyr, age 1 Gyr) makes Q_H proportional to the
# star formation rate over the last 10 Myr, so one SFR gives one Q_H; the radio block is
# evaluated at the L_IR that Murphy et al.'s Eq. 4 assigns to that SFR. The ratio of the
# two per unit Q_H is 0.985–1.119× (0.985× at 3 and 1 cm, 1.043× at 3 mm, 1.119× at 1 mm), asserted to lie in 0.8–1.25× at every
# wavelength the radio block reaches.
#
# **4. Continuity through 1 mm.** The radio SED with `freefree: True` has no step
# through 1 mm when synchrotron and free-free both extend below it: the largest
# |d ln L_ν / d ln λ| between neighbouring points over 0.1–3 mm stays below 1.5, and the
# flux at 0.95 mm is within 0.9–1.2× that at 1.05 mm. The test runs only where the radio
# block carries emission shortward of 1 mm, which the cell reads from the data; it
# prints that it was skipped otherwise. Result: the radio block carries emission below 1 mm, so the test runs. The largest |d ln L_ν / d ln λ| over 0.1–3 mm is 0.326 and the flux ratio across 1 mm is 0.9775.

# %%
import scipy.constants as _sc

_FF_LAMBDAS = (("3 cm", 3.0e8), ("1 cm", 1.0e8), ("3 mm", 3.0e7), ("1 mm", 1.0e7))  # [Å]
_FF_BAND_CUE_VS_CLOUDY = (0.95, 1.12)  # tengri / CIGALE, per ionizing photon
_FF_BAND_VS_ANALYTIC = (0.70, 1.00)  # each code over the analytic reference


def _loglog_at(wave_aa, lnu, target_aa):
    """``lnu`` interpolated log-log to ``target_aa`` [Å]."""
    return float(
        np.exp(
            np.interp(
                np.log(target_aa), np.log(np.asarray(wave_aa)), np.log(np.maximum(lnu, 1e-300))
            )
        )
    )


# Check 1. Thermal free-free per ionizing photon, each code over its own Q_H, on
# the §3 galaxy (BC03, logU = -2, Z = 0.02, f_esc = 0). CIGALE's nebular module
# carries the continuum in its emission component; no emission line sits between
# 1 mm and 3 cm, so the sum of its young and old emission components is the
# continuum there.
_, _l_old = C.to_lnu_contribution(sed_c, "nebular.emission_old")
_w_ff_c, _l_young = C.to_lnu_contribution(sed_c, "nebular.emission_young")
_l_ff_c = np.asarray(_l_old) + np.asarray(_l_young)
_w_ff_t = np.asarray(s_stellar.wave)
_l_ff_t = np.asarray(s_stellar.derived["sed_nebular"])

# Check 2. The optically thin free-free emission of a pure hydrogen H II region
# per ionizing photon, case B, T_e = 1e4 K. The emissivity is the thermal
# bremsstrahlung expression of Rybicki & Lightman (1979, Eq. 5.14a),
#   ε_ν = [2^5 π e^6 / (3 m_e c^3)] (2π / 3 k m_e)^(1/2) T^(-1/2) n_e n_p e^(-hν/kT) g_ff,
# with the velocity-averaged Gaunt factor g_ff from the fit of Draine (2011).
# Every recombination to n >= 2 in a gas that absorbs all of Q_H gives
# Q_H = α_B n_e n_p V, so ε_ν / (α_B n_e n_p) is L_ν per ionizing photon. α_B is the
# Hui & Gnedin (1997, MNRAS 292, 27) fit to the case B coefficient of Ferland et al.
# (1992). Constants are CODATA, converted to cgs.
_E_ESU = _sc.e * _sc.c * 10.0  # elementary charge [esu]
_M_E, _C_CGS = _sc.m_e * 1e3, _sc.c * 1e2  # [g], [cm/s]
_K_B, _H_P = _sc.k * 1e7, _sc.h * 1e7  # [erg/K], [erg s]
_FF_PREFACTOR = (
    2**5
    * np.pi
    * _E_ESU**6
    / (3.0 * _M_E * _C_CGS**3)
    * np.sqrt(2.0 * np.pi / (3.0 * _K_B * _M_E))
)  # [erg cm^3 K^(1/2) / s / Hz] per (n_e n_i Z^2)


def _gaunt_ff(nu_hz, t_e=1.0e4, z_ion=1.0):
    """Velocity-averaged free-free Gaunt factor, the fit of Draine (2011)."""
    return np.log(
        np.exp(5.960 - np.sqrt(3.0) / np.pi * np.log(z_ion * nu_hz / 1e9 * (t_e / 1e4) ** -1.5))
        + np.e
    )


def _alpha_b_hydrogen(t_e=1.0e4):
    """Case B recombination coefficient of hydrogen [cm^3/s], Hui & Gnedin (1997)."""
    lam = 2.0 * 157807.0 / t_e
    return 2.753e-14 * lam**1.5 / (1.0 + (lam / 2.74) ** 0.407) ** 2.242


def _ff_per_photon_analytic(wave_aa, t_e=1.0e4):
    """Optically thin free-free L_ν per ionizing photon [erg/s/Hz per ph/s]."""
    nu = C_AA / wave_aa
    return (
        _FF_PREFACTOR
        * t_e**-0.5
        * np.exp(-_H_P * nu / (_K_B * t_e))
        * _gaunt_ff(nu, t_e)
        / _alpha_b_hydrogen(t_e)
    )


print(
    f"§11b analytic free-free: prefactor {_FF_PREFACTOR:.3e} (Rybicki & Lightman Eq. 5.14b quotes 6.8e-38), "
    f"α_B(1e4 K) = {_alpha_b_hydrogen():.3e} cm^3/s"
)
assert abs(_FF_PREFACTOR / 6.8e-38 - 1.0) < 0.01

# The Gaunt fit against the Hummer (1988) values tabulated by Dickinson, Davies &
# Davis (2003, MNRAS 341, 369, their Table 2), at 1e4 K.
_HUMMER88_GAUNT_1E4K = {0.4: 6.39, 1.4: 5.73, 70.0: 3.67}  # {ν [GHz]: g_ff}
for _nu_ghz, _g_tab in _HUMMER88_GAUNT_1E4K.items():
    _g_fit = float(_gaunt_ff(_nu_ghz * 1e9))
    print(f"    g_ff({_nu_ghz:g} GHz, 1e4 K): fit {_g_fit:.3f}, table {_g_tab:.2f} ({_g_fit / _g_tab - 1:+.1%})")
    assert abs(_g_fit / _g_tab - 1.0) < 0.02

print("§11b thermal free-free per ionizing photon [erg/s/Hz per ph/s], §3 galaxy:")
print("    λ        CIGALE      tengri Cue   Cue/CIGALE   analytic    CIGALE/analytic  Cue/analytic")
_ff_rows = []
for _name, _lam in _FF_LAMBDAS:
    _pc = _loglog_at(_w_ff_c, _l_ff_c, _lam) / _n_ly_c3
    _pt = _loglog_at(_w_ff_t, _l_ff_t, _lam) / _q_h_t3
    _pa = float(_ff_per_photon_analytic(_lam))
    _ff_rows.append((_name, _pc, _pt, _pa))
    print(
        f"    {_name:<5} {_pc:11.4e} {_pt:11.4e} {_pt / _pc:10.3f} {_pa:12.4e} "
        f"{_pc / _pa:12.3f} {_pt / _pa:14.3f}"
    )
    assert _FF_BAND_CUE_VS_CLOUDY[0] < _pt / _pc < _FF_BAND_CUE_VS_CLOUDY[1], (_name, _pt / _pc)
    assert _FF_BAND_VS_ANALYTIC[0] < _pc / _pa < _FF_BAND_VS_ANALYTIC[1], (_name, "CIGALE")
    assert _FF_BAND_VS_ANALYTIC[0] < _pt / _pa < _FF_BAND_VS_ANALYTIC[1], (_name, "Cue")

# CIGALE's continuum per photon over its metallicity axis, from its library at
# logU = -2, n_e = 100 cm^-3, at 1 mm. A hydrogen plasma at fixed T_e does not
# depend on Z, so the fall is a property of the Cloudy models, which carry dust.
_cont_z = []
with _CigaleLineDB("nebular_continuum") as _cont_db:
    for _z in (0.0004, 0.004, 0.02, 0.041):
        _rec = _cont_db.get(Z=_z, logU=-2.0, ne=100.0)
        _wl_aa, _l_nu = U.wnm_to_erg_per_hz_per_aa(np.asarray(_rec.wl), np.asarray(_rec.spec))
        _cont_z.append((_z, _loglog_at(_wl_aa, _l_nu, 1.0e7)))
print(
    "§11b CIGALE nebular continuum per ionizing photon at 1 mm vs Z: "
    + ", ".join(f"Z = {_z:g}: {_v:.3e}" for _z, _v in _cont_z)
    + f" (Z = 0.0004 over Z = 0.041: {_cont_z[0][1] / _cont_z[-1][1]:.2f}×)"
)
assert all(a[1] > b[1] for a, b in zip(_cont_z, _cont_z[1:]))

# %%
# Check 3. The free-free term the radio block adds with `freefree: True`
# (Murphy et al. 2011) against the nebular free-free it stands in for, at the same
# Q_H. A history that is nearly constant over 10 Myr (τ = 20 Gyr, age 1 Gyr) makes Q_H
# proportional to the star formation rate over the last 10 Myr, so one SFR gives
# one Q_H. The radio block is evaluated at the L_IR that Murphy et al.'s Eq. 4,
# SFR = 3.88e-44 L_IR [erg/s], assigns to that SFR; the nebular term is Cue's, with
# the nebular emitter on and no radio block.
_MURPHY11_SFR_PER_LIR = 3.88e-44  # [M☉/yr per erg/s], Murphy et al. (2011) Eq. 4
_m_ss = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(20.0),
        "age_gyr": Fixed(1.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
_s_ss = _m_ss.predict_state({})
_sfr_ss = float(np.asarray(_s_ss.derived["sfr_10myr"]))
_q_ss = float(np.asarray(_s_ss.derived["nion"]))
_w_ss = np.asarray(_s_ss.wave)
_l_ss_cue = np.asarray(_s_ss.derived["sed_nebular"])
_l_ss_radio = np.asarray(
    radio_freefree(jnp.asarray(_w_ss), _sfr_ss / _MURPHY11_SFR_PER_LIR, T_e=1.0e4, alpha_ff=-0.1)
)
print(
    f"§11b steady star formation: SFR(10 Myr) = {_sfr_ss:.3e} M☉/yr, Q_H = {_q_ss:.3e} ph/s "
    f"({_q_ss / _sfr_ss:.3e} ph/s per M☉/yr)"
)
print("    λ      radio block per Q_H   Cue per Q_H    radio block / Cue")
_FF_BAND_RADIO_VS_CUE = (0.8, 1.25)
for _name, _lam in _FF_LAMBDAS:
    _cue_pp = _loglog_at(_w_ss, _l_ss_cue, _lam) / _q_ss
    _rad_pp = _loglog_at(_w_ss, _l_ss_radio, _lam) / _q_ss
    if _rad_pp > 1e-30:
        print(f"    {_name:<5} {_rad_pp:16.4e} {_cue_pp:16.4e} {_rad_pp / _cue_pp:12.3f}")
        assert _FF_BAND_RADIO_VS_CUE[0] < _rad_pp / _cue_pp < _FF_BAND_RADIO_VS_CUE[1], _name
    else:
        print(f"    {_name:<5} the radio block's free-free term stops at a longer wavelength")

# Check 4. Continuity of the radio SED through 1 mm. The check runs only when the
# radio block extends below 1 mm, which the data decide: with `freefree: True`
# the star-forming radio SED must be nonzero at 0.5 mm.
_state_ff_on = _radio_model(
    {"type": "bell2003", "freefree": True, "nu_ref": "21cm", "ir_window": "total"}
).predict_state({})
_w_rad = np.asarray(_state_ff_on.wave)
_l_rad = np.asarray(_state_ff_on.derived["sed_radio"])
_extends_below_1mm = bool(_loglog_at(_w_rad, _l_rad, 5.0e6) > 1e-30)


def _max_abs_log_slope(wave_aa, lnu, lo_aa, hi_aa):
    """Largest |d ln L_ν / d ln λ| between neighbouring grid points in [lo_aa, hi_aa]."""
    w, y = np.asarray(wave_aa), np.asarray(lnu)
    sel = (w >= lo_aa) & (w <= hi_aa) & (y > 0)
    return float(np.max(np.abs(np.diff(np.log(y[sel])) / np.diff(np.log(w[sel])))))


if _extends_below_1mm:
    _slope_max = _max_abs_log_slope(_w_rad, _l_rad, 1.0e6, 3.0e7)
    _step_1mm = _loglog_at(_w_rad, _l_rad, 0.95e7) / _loglog_at(_w_rad, _l_rad, 1.05e7)
    print(
        f"§11b radio SED across 1 mm: largest |d ln L_ν / d ln λ| over 0.1-3 mm = {_slope_max:.3f}; "
        f"L_ν(0.95 mm) / L_ν(1.05 mm) = {_step_1mm:.4f}"
    )
    assert _slope_max < 1.5, "the radio SED has a step near 1 mm"
    assert 0.9 < _step_1mm < 1.2, "the radio SED has a step at 1 mm"
else:
    print(
        "§11b radio SED across 1 mm: skipped, the radio block carries no emission "
        "shortward of 1 mm in this build"
    )


# %% [markdown]
# ## §12 IGM transmission
#
# CIGALE module `redshifting` (`igm_transmission`) ↔ tengri
# `igm.meiksin06` (`igm_transmission_meiksin06`): Meiksin (2006) IGM attenuation,
# Lyman series and the diffuse-IGM Lyα forest continuum suppression. The figure
# plots each code's transmission against observed wavelength at z = 3, 5 and 7.
#
# **Where they differ.** CIGALE 2025.1 evaluates the n ≥ 3 Lyman-series optical
# depths at the source redshift (`redshifting.py:63-81`) and chooses Meiksin's
# Eq. 2/3 for Lyα from the source redshift. tengri evaluates every line at its
# absorber redshift z_n = λ_obs/λ_n − 1, as Meiksin (2006) Table 1 defines. The
# printed |ΔT| statistics reach a few percent at z = 3 and 5. Redward of Lyβ
# (rest ≥ 1026 Å) only Lyα is active and the two codes agree there; the cell
# prints the max |ΔT| for that region separately. Meiksin's Table 2 gives
# T(1730 Å, z = 5) = 0.003277; the cell prints that literal beside both codes'
# values at observed 1730 Å.
#
# **Verification Status:** CROSSVAL: Inoue+2014 IGM transmission

# %%
# Both transmission curves come straight from each code's own IGM
# function — no SED build, no flux-ratio reconstruction. CIGALE exposes
# the Meiksin (2006) transmission as
# `pcigale.sed_modules.redshifting.igm_transmission(wave_nm, z)`; tengri
# exposes `igm_transmission_meiksin06(wave_obs_AA, z)`. Same prescription,
# so the curves should overlay.
import jax.numpy as _jnp
from pcigale.sed_modules.redshifting import igm_transmission as cigale_igm

from tengri.igm import igm_transmission_meiksin06

fig, ax_l, ax_r = U.two_panel_fig()
for ax in (ax_l, ax_r):
    ax.set_xlabel(r"$\lambda_{\rm obs}$ [Å]")
    ax.set_ylabel("IGM transmission")
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
ax_l.set_title("pcigale  redshifting.igm_transmission (Meiksin 2006)")
ax_r.set_title("tengri  igm.meiksin06")

wave_obs_aa = np.logspace(np.log10(500.0), np.log10(1e4), 600)
print("§12 Meiksin 2006 transmission (tengri − CIGALE) on the plotted grid:")
for color, z in zip(("C0", "C1", "C2"), (3.0, 5.0, 7.0)):
    # CIGALE igm_transmission takes wavelength in nm.
    T_c = np.asarray(cigale_igm(wave_obs_aa / 10.0, z))
    T_t = np.asarray(igm_transmission_meiksin06(_jnp.asarray(wave_obs_aa), z))
    ax_l.plot(wave_obs_aa, T_c, color=color, linewidth=1.4, label=rf"$z = {z:.0f}$")
    ax_r.plot(wave_obs_aa, T_t, color=color, linewidth=1.4, label=rf"$z = {z:.0f}$")
    _dT = np.abs(T_t - T_c)
    _red = wave_obs_aa >= 1026.0 * (1.0 + z)
    print(
        f"    z = {z:.0f}: max |ΔT| redward of Lyβ (rest ≥ 1026 Å) = {float(_dT[_red].max()):.2e}; "
        f"over the whole grid max |ΔT| = {float(_dT.max()):.2e}, "
        f"median |ΔT| = {float(np.median(_dT)):.2e}, "
        f"{int((_dT > 1e-3).sum())}/{_dT.size} points above 1e-3"
    )

for _z in (3.0, 5.0):
    _t_c = float(np.asarray(cigale_igm(np.array([1730.0 / 10.0]), _z))[0])
    _t_t = float(np.asarray(igm_transmission_meiksin06(_jnp.asarray([1730.0]), _z))[0])
    print(f"    T(1730 Å) at z = {_z:.0f}: tengri {_t_t:.6f}, CIGALE {_t_c:.6f}")

# Meiksin (2006) Table 2, transmission at observed 1730 Å and z = 5 (literal from the paper).
_T_MEIKSIN06_TABLE2_1730_Z5 = 0.003277
_t_c5 = float(np.asarray(cigale_igm(np.array([1730.0 / 10.0]), 5.0))[0])
_t_t5 = float(np.asarray(igm_transmission_meiksin06(_jnp.asarray([1730.0]), 5.0))[0])
print(
    f"    T(1730 Å) at z = 5 against Meiksin (2006) Table 2 ({_T_MEIKSIN06_TABLE2_1730_Z5:.6f}, "
    f"paper literal): tengri {_t_t5:.6f}, CIGALE {_t_c5:.6f}"
)

ax_l.legend(fontsize=9)

ax_r.legend(fontsize=9)
fig.tight_layout()
save_fig("cigale_12_igm_transmission.png")


# %% [markdown]
# ## tengri in CIGALE-mode — the full X-ray → radio SED
#
# The whole chain at once: tengri configured as CIGALE's `sfhdelayed` + `bc03` +
# `nebular` + `dustatt_modified_starburst` + `dale2014` + `xray` (Yang+2020) +
# `radio` chain, overlaid on pcigale at matched parameters. The figure shows
# ν L_ν over eleven decades of wavelength and the tengri/CIGALE residual below.
# The radio block is `freefree: False` with `q_IR = 2.5` (§11), so `q_IR`
# calibrates the synchrotron as in CIGALE's `qir_sf` and the thermal free-free
# comes from the nebular continuum on both sides.
#
# **The stellar-to-FIR core.** Optical agreement is reported as a normalization
# ratio and its 16–84 % spread. With the single-screen dust mapping
# (`tau_bc = 0`) the lower panel's shaded band marks ±25 %. The sub-912 Å
# excursion is the Lyman-continuum extrapolation. The 1 mm to 3 cm decade carries
# thermal free-free from the nebular continuum on both codes (Cue's tail on
# tengri's side, CIGALE's `nebular` module on the other), and its ratio is §3's Q_H
# ratio times §11b's Cue-against-Cloudy ratio per ionizing photon; the radio blocks
# add synchrotron only. The X-ray wing is XRB + hot gas with no AGN corona
# (`alpha_ox` is supplied but there is no disc for it to act on), and its 0.99×
# is pcigale's truncated band normalization (§10). Every number is printed below.

# %%
import chex

# The full X-ray -> radio SED: the §6/§7 galaxy (stellar + dust + Dale IR) plus
# the §10 X-ray (Yang+2020 XRB + hot gas, no AGN corona in this galaxy-only
# chain) and the §11 radio (`freefree=False`: q_IR = 2.5 calibrates the
# synchrotron, as CIGALE's `qir_sf`), so the master grid spans ~0.01 Å (hard
# X-ray) to ~1 m (radio).
sed_c_full = C.run_chain(
    [
        (
            "sfhdelayed",
            dict(
                tau_main=1000,
                age_main=5000,
                tau_burst=50,
                age_burst=20,
                f_burst=0.0,
                sfr_A=1.0,
                normalise=True,
            ),
        ),
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        NEB_FIDUCIAL_CIGALE,
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("dale2014", dict(alpha=2.0)),
        (
            "yang20",
            dict(
                gam=1.8,
                E_cut=300.0,
                alpha_ox=-1.4,  # no AGN disc here -> corona is zero; XRB + hot gas only
                max_dev_alpha_ox=0.2,
                angle_coef="0.5 & 0",
                det_lmxb=0.0,
                det_hmxb=0.0,
            ),
        ),
        ("radio", dict(qir_sf=2.5, alpha_sf=0.8, R_agn=0.0, alpha_agn=0.7)),
        ("redshifting", dict(redshift=0.0)),
    ]
)
_w_full, _L_full = C.to_lnu(sed_c_full)
w_ext, L_ext = np.asarray(_w_full), np.asarray(_L_full)

m_full = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        "lyman_cutoff": True,
        "all_params": Fixed(DEFAULT),
    },
    dust_emission={
        "type": "dale2014_cigale",
        "alpha_dale": Fixed(2.0),
        "all_params": Fixed(DEFAULT),
    },
    xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
    radio={
        "sf": {"type": "bell2003", "freefree": False, "nu_ref": "21cm", "ir_window": "total"},
        "agn": {"type": "powerlaw"},
        "radio_q_ir": Fixed(2.5),
        "radio_alpha_sf": Fixed(0.8),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL_TENGRI,
    redshift=Fixed(0.0),
)
s_full = m_full.predict_state({})
wave_t = np.asarray(s_full.wave)
L_t = np.asarray(s_full.sed_intrinsic)

# Put tengri on CIGALE's wavelength grid so the two compare point for point.
L_t_on_ext = U.regrid(wave_t, L_t, w_ext)
chex.assert_equal_shape([L_ext, L_t_on_ext])

mask = (w_ext > 0) & (L_ext > 0) & (L_t_on_ext > 0)
resid = np.full(w_ext.shape, np.nan, dtype=float)
resid[mask] = L_t_on_ext[mask] / L_ext[mask] - 1.0

# Headline numbers: the optical normalization ratio tengri/CIGALE and its
# 16–84% spread. With the shared BC03 grid, matched mass convention and
# single-screen dust mapping the ratio sits near 1. Three things sit outside
# that window: the far-UV, which is §3's age-binning convention; the sub-912 Å
# excursion, which is the Lyman-continuum extrapolation; and the 1 mm–1 m decade,
# where both codes carry thermal free-free from the nebular continuum (Cue's tail
# extends the continuum from the 1 cm table edge to 1 m; CIGALE's `nebular`
# module's native reach) with the radio blocks calibrated on the synchrotron
# alone. The Cue-vs-CLOUDY nebular residual of §8 is part of the optical window.
opt = mask & (w_ext >= 1000.0) & (w_ext <= 10000.0)
ratio_opt = L_t_on_ext[opt] / L_ext[opt]
norm = float(np.median(ratio_opt))
p16, p84 = float(np.percentile(ratio_opt, 16)), float(np.percentile(ratio_opt, 84))
print(
    f"full-SED head-to-head tengri/CIGALE optical (1000–10000 Å): "
    f"normalization {norm:.2f}×, 16–84% spread {p16:.2f}–{p84:.2f}×"
)


def _ratio_at(target_aa: float) -> float:
    """tengri/CIGALE L_ν ratio at the CIGALE grid point nearest ``target_aa``."""
    j = int(np.argmin(np.abs(w_ext - target_aa)))
    return float(L_t_on_ext[j] / L_ext[j]) if L_ext[j] > 0 else float("nan")


_C_AA_HZ = C_AA
print(
    "  X-ray  1 keV = {:.2f}×, 5 keV = {:.2f}×  (XRB + hot gas; no AGN corona)".format(
        _ratio_at(12.398 / 1.0), _ratio_at(12.398 / 5.0)
    )
)
print(
    "  radio  1.4 GHz = {:.2f}×, 150 MHz = {:.2f}×  (SF synchrotron, q_IR = 2.5 calibrating the non-thermal term)".format(
        _ratio_at(_C_AA_HZ / 1.4e9), _ratio_at(_C_AA_HZ / 0.15e9)
    )
)
print(
    "  mm     1 mm = {:.2f}×, 3 mm = {:.2f}×, 1 cm = {:.2f}×, 3 cm = {:.2f}×  "
    "(nebular free-free on both sides, Q_H ratio × Cue-against-Cloudy per photon, §11b; "
    "the radio blocks add synchrotron only)".format(
        _ratio_at(1e7), _ratio_at(3e7), _ratio_at(1e8), _ratio_at(3e8)
    )
)
_assert_comparable(L_ext, L_t, name="full-SED head-to-head")

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
# nu L_nu, not L_nu. Over eleven decades of wavelength L_nu is a hopeless
# display coordinate: the synchrotron tail climbs as a power law to the grid
# edge, so the SED spans 16.2 decades in L_nu and any peak-anchored y-limit
# that keeps the stellar and dust humps legible amputates the X-ray and the
# far-UV. The same SED spans 8.5 decades in nu L_nu, so everything fits on one
# axis — and equal areas are equal energy, which is what makes the stellar and
# dust humps read as the comparable reservoirs energy balance says they are.
_nu_ext = _C_AA_HZ / w_ext
nuL_ext = _nu_ext * L_ext
# Mask where tengri's grid has run out rather than let regrid's zero-fill draw a
# vertical cliff at the grid edge — the model stops there, it does not go dark.
nuL_t_on_ext = np.where(L_t_on_ext > 0, _nu_ext * L_t_on_ext, np.nan)
ax.plot(w_ext, nuL_ext, "C0-", linewidth=1.5, label="CIGALE")
ax.plot(w_ext, nuL_t_on_ext, "C1--", linewidth=1.5, label="tengri (CIGALE-mode)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-1, 1e12)  # hard X-ray -> radio
_ymax_h = float(max(np.nanmax(nuL_ext), np.nanmax(nuL_t_on_ext)))
ax.set_ylim(_ymax_h * 1e-9, _ymax_h * 3.0)
ax.set_ylabel(r"$\nu L_\nu$ [erg/s]")
ax.set_title(r"tengri in CIGALE-mode vs CIGALE — full panchromatic SED")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(
    0.02,
    0.05,
    rf"tengri/CIGALE $= {norm:.2f}\times$ (16–84%: {p16:.2f}–{p84:.2f})",
    transform=ax.transAxes,
    fontsize=10,
    va="bottom",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)

ax_r.axhspan(-0.25, 0.25, color="0.85", zorder=0)
ax_r.axhline(0.0, color="0.5", linewidth=0.8)
ax_r.axhline(norm - 1.0, color="C1", linestyle=":", linewidth=0.9)
ax_r.plot(w_ext, resid, "C1-", linewidth=1.0)
ax_r.set_xscale("log")
ax_r.set_xlim(1e-1, 1e12)
ax_r.set_ylim(-1.0, 1.0)
ax_r.set_xlabel(r"$\lambda$ [Å]")
ax_r.set_ylabel(r"tengri/CIGALE $-1$")
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("cigale_full_sed_headtohead.png")
plt.show()


# %% [markdown]
# ## Summary
#
# Section by section, tengri / CIGALE as printed above. A ratio is tengri over
# CIGALE; sweep sections quote the median across the swept cases and the worst
# printed case.
#
# | § | Block | Agreement |
# |---|---|---|
# | 1 | SSP | median residual 2.0–2.2e-8 at every age (float32 round-trip) |
# | 2 | SFH | `delayed` median 1.00002×; `sfh2exp` median 1.00015×, 2 of 1969 points beyond 1 % (burst step); §2c median 1.000× except periodic exponential 0.999× and rectangular 0.993× |
# | 3 | stellar SED | energy-weighted 912–1200 Å 1.063×, 1200–3000 Å 1.030×, 0.3–1 μm 1.006×, L_bol 1.016×, Q_H 1.150× (age convention: a continuous convolution gives 1.009× of tengri's Q_H, and aging every star by 0.5 Myr gives 1.000× of CIGALE's); §3b band medians 1.001–1.018×, worst band 1.055× (τ = 3 Gyr, age = 1 Gyr) |
# | 4 | attenuation laws | max \|Δ\| 0.252 % (calzleit), 0.247 % (starburst + bump), 0.000 % (CF00); 0.000 % outside the 1500–1800 Å crossover |
# | 5 | attenuation applied | energy-weighted 0.1–1 μm 1.0165× intrinsic and 0.9923× attenuated (912–1200 Å 1.0631×, 1.0571×); §5b band medians 0.997–1.000×; single-point maxima of 75–942 % over 0.1–1 μm sit on emission-line wings (line profile and grid sampling), not on the continuum |
# | 6 | dust IR | 10–100 μm energy-weighted 1.012×; `L_absorbed` 1.0112× `dust.luminosity`; energy balance residual 0; §6c 8–1000 μm medians 1.0109–1.0141×, worst filter 1.043× (`dl2007`, qpah = 4.58); `dl2014` as dust emission alone 1.0119×, 1.0116×, 1.0116× at α = 1, 2, 3 with no band outside 5 %; `schreiber2016` 1.0115× |
# | 7 | panchromatic | energy-weighted 1.0616× (912–1200 Å), 1.0614× (1250–3000 Å), 1.0122× (0.3–1 μm), 1.0081× (3–8 μm), 1.0125× (10–100 μm), 1.0116× (100–1000 μm) |
# | 8 | nebular | Q_H 1.146× (shared SSP); raw Hα 1.12×, [O III] 0.96×, Hβ 1.12×, and per ionizing photon 0.974×, 0.834×, 0.978×; §8b at the six stable cases the largest ratio-of-ratios is 0.99× (Hα/Hβ), 2.06× ([O III]/Hβ) and 4.23× ([O II]/Hβ), with Z_gas = 0.041 degenerate in CIGALE's grid (18.87× and 44.38×); Hβ/Q_H at f_esc = 0.5 over f_esc = 0 is 0.3851 (tengri) and 0.3862 (CIGALE) against (1 − f)/(1 + 0.597 f) = 0.3851; Cue's [O III]/Hβ response to logU is 1.154 / 1.596 / 1.663 at the fitted shape and 3.416 / 5.499 / 6.085 at Cue's default hard shape, for logU = −3 / −2 / −1.5 |
# | 9 | AGN, SKIRTOR | disc 1.0099×, torus 1.0111×, polar dust 1.0118× (§9); 1.0076×, 1.0147×, 0.9996× with the SKIRTOR disc (§9b); 1500–3000 Å full SED 1.027×, of which host 1.061× and disc 1.009×; §9d `R_faceon` 0.998154×, quadrature term 0.994058×, curve term 1.002761×, polar reference 0.996803× (`polar_law="bongiorno"`; 0.970908× with the default `"smc"`); §9e SKIRTOR grid band medians 1.011–1.012×, worst band 1.013× |
# | 9 | AGN, Fritz | §9e band medians 1.009–1.010×, worst bands 0.973–0.975×, none of 12 bands outside 5 % |
# | 10 | X-ray | 1.010× at 2 keV and as a 0.5–10 keV median, at i = 0°, 30°, 60° and 80° |
# | 11 | radio | synchrotron 1.0112× across 0.1–100 GHz against ×1.0112 predicted from the §6 energy balance alone, with `nu_ref: "21cm"` and `ir_window: "total"` set as in pcigale's `radio` module (`freefree: False`) |
# | 11b | free-free | per ionizing photon Cue / CIGALE 1.04–1.08× (1.044× at 3 cm, 1.077× at 1 cm, 1.075× at 3 mm and 1 mm); against the analytic reference CIGALE 0.78–0.80× and Cue 0.83–0.86×; radio block / Cue 0.985–1.119× (0.985× at 3 and 1 cm, 1.043× at 3 mm, 1.119× at 1 mm); continuity through 1 mm: largest |d ln L_ν / d ln λ| 0.326 over 0.1–3 mm, flux ratio across 1 mm 0.9775 |
# | 12 | IGM | max \|ΔT\| 4.4e-2, 4.6e-2, 2.0e-2 at z = 3, 5, 7; median \|ΔT\| 6.4e-4, 1.5e-4, 4.4e-7 (CIGALE evaluates the n ≥ 3 Lyman lines and the Lyα equation choice at the source redshift, tengri at each absorber's, as Meiksin's Table 1 defines); T(1730 Å, z = 5) is 0.003283 for tengri and 0.000909 for CIGALE against Meiksin's Table 2 0.003277; redward of Lyβ the max \|ΔT\| is ≤ 2.8e-16 |
# | full | CIGALE-mode SED | optical 1.02× (16–84 %: 1.00–1.07×); X-ray 0.99× (1 keV), 0.99× (5 keV); radio 1.02× (1.4 GHz), 1.02× (150 MHz); 1 mm / 3 mm / 1 cm / 3 cm 1.02× / 1.09× / 1.07× / 1.04× |
#
# Four conventions account for most of the residuals and are stated where they are
# measured. The age convention (§3) is exact in tengri and a 1 Myr discrete sum, equivalent
# to aging every star by 0.5 Myr, in CIGALE; it sets the UV, Q_H and, through
# `L_absorbed`, the ~1 % offset under the dust IR and radio (§6, §11). The nebular
# emitter differs by design: per ionizing photon Cue's Balmer lines are 2–3 % below
# CIGALE's, its [O III] is lower because of the fitted ionizing shape and a different
# definition of logU, and its free-free continuum is within about 10 % (§8, §11b). CIGALE's
# own conventions are a polar-dust quadrature coarser than tengri's (§9d) and an X-ray
# band normalization truncated on its wavelength grid (§10). The polar-dust extinction law
# must be set to `polar_law="bongiorno"` to match CIGALE (§9, §9d), and `freefree: False`
# makes `radio_q_ir` calibrate the synchrotron alone, as CIGALE's `qir_sf` does (§11).
# tengri's radio defaults are Bell (2003)'s 1.4 GHz and the model's own paper's IR
# window; the CIGALE comparison sets 21 cm and the total dust luminosity explicitly
# (§11).

# %% [markdown]
# ## References
#
# * Bell 2003, ApJ 586, 794: IR–radio correlation
# * Boquien et al. 2019, A&A 622, A103: CIGALE
# * Bruzual & Charlot 2003, MNRAS 344, 1000: BC03 SSPs
# * Calzetti et al. 2000, ApJ 533, 682: starburst attenuation law
# * Charlot & Fall 2000, ApJ 539, 718: two-component dust
# * Condon 1992, ARA&A 30, 575: radio synchrotron / IR–radio correlation
# * Dale et al. 2014, ApJ 784, 83: IR dust templates
# * Dickinson, Davies & Davis 2003, MNRAS 341, 369: free-free Gaunt factors (Hummer 1988 values)
# * Draine 2011, Physics of the Interstellar and Intergalactic Medium (Princeton University Press): free-free Gaunt-factor fit
# * Fritz et al. 2006, MNRAS 366, 767: AGN torus
# * Gutkin, Charlot & Bruzual 2016, MNRAS 462, 1757: nebular grid, ionization parameter at the Strömgren radius
# * Hui & Gnedin 1997, MNRAS 292, 27: case B recombination coefficient fit
# * Inoue et al. 2014, MNRAS 442, 1805: IGM transmission
# * Li et al. 2025, ApJ, 986, 9 (Cue, arXiv:2405.04598): neural CLOUDY emulator
# * Madau 1995, ApJ 441, 18: original IGM transmission
# * Meiksin 2006, MNRAS 365, 807: updated IGM transmission
# * Murphy et al. 2011, ApJ 737, 67: free-free radio SFR calibration
# * Noll et al. 2009, A&A 507, 1793: modified Calzetti
# * Rybicki & Lightman 1979, Radiative Processes in Astrophysics (Wiley), §5.2, Eqs. 5.14a-b: thermal bremsstrahlung
# * Silva et al. 2004, MNRAS 355, 973: AGN torus
# * Stalevski et al. 2016, MNRAS 458, 2288: SKIRTOR
# * Yang et al. 2020, MNRAS 491, 740: X-ray CIGALE module
