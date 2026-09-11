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
# This notebook places its physics modules next to their tengri equivalents at the same parameters — the module map below pairs them up.
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
# **What to expect.** The shared SSP grid, the parametric SFHs, the
# attenuation laws, the X-ray corona and the Meiksin IGM agree to floating
# point or a fraction of a percent. Three things do not, and each is stated
# where it is measured rather than here: tengri's age-binning convention
# (§3), which sets a ~2 % floor under the dust IR and the radio; the nebular
# emitter, which is Cue — a neural emulator trained on Cloudy 17
# (Li et al. 2025) — against CIGALE's Cloudy 13.x grids (§8); and the AGN
# dust budget (§9). Every section prints the number it claims.

# %% [markdown]
# ## Setup

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import warnings

# Register CIGALE's own Dale et al. 2014 grid for this reproduction notebook.
# tengri ships two Dale grids and they are different data, which is why the
# comparison has to name one. ``data/dale2014_templates.h5`` (tengri's
# ``dale2014``) is the unmodified Wyoming-source release, and it embeds a
# star-forming radio synchrotron continuum rising to 2.2459e9 Å;
# ``data/dale2014_templates_cigale.h5`` (``dale2014_cigale``) is CIGALE's own
# ``dale2014`` SED template, whose red end falls away (measured slope -5.5),
# because CIGALE adds radio in a separate module. This notebook wants the
# latter: it is what pcigale evaluates on the other side of every panel, and
# it is also the only one of the two that can compose with a separate
# ``radio={...}`` block without double-counting the synchrotron between
# ~1.34 and ~10 GHz. tengri refuses that composition by reading the selected
# grid's own red edge, not by its registry name, so this is a choice about
# which templates the reproduction needs rather than a way around a guard.
# Both files come from ``scripts/regenerate_dale2014_from_{cigale,official}.py``.
from pathlib import (
    Path,
    Path as _Path,
)

import matplotlib.pyplot as plt
import numpy as np
from reproduction.cigale._drivers import cigale_driver as C, units as U

import tengri
from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
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

# Notebook-vs-script compatible: ``__file__`` is undefined when this
# is run via nbclient (the kernel's resources path is set to the
# reproduction/cigale/ directory instead), so fall back to the CWD.
_HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd().resolve()
figs_dir = _HERE / "_figs"
figs_dir.mkdir(exist_ok=True)


_FIG_DPI = 150


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


# %% [markdown]
# ## Common SSP grid
#
# CIGALE's BC03 Chabrier templates re-shaped into the DSPS HDF5 layout
# that tengri reads — same numerical SSPs on both sides.

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
# SSP has any point above 1e-7 — four of its 1218 non-zero points, all in the
# extreme UV below 240 Å where the SSP is 15 decades below its peak and the
# relative measure is reading the last float32 digit of a number near zero
# (the 230 Å point, $L_\nu = 1.5\times10^{-7}$ of the peak, is the 5e-6 spike
# on the residual panel).
#
# Both sides divide by the same speed of light — tengri's `C_AA`, which
# `_drivers/cigale_ssp_to_dsps.py` also uses to write the grid and
# `_drivers/units.py` to convert CIGALE's W/nm. A rounded c on either side
# puts a floor of $|\Delta c|/c$ under this residual at every wavelength and
# every age, 2.5e-5 for the usual 2.998e18 — three decades above the float32
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
# tengri's `sfh.delayed` is the same τ-delayed shape CIGALE uses in
# `sfhdelayed`: SFR(t) ∝ t · exp(−t/τ), peak at t = τ. Both integrate to 1 M☉
# formed by `age` — CIGALE via `normalise=True`, tengri via `log_total_mass = 0.0`.
# tengri's curve is `state.derived["sfr_history"]` on a 256-point log-spaced
# lookback grid; the `∫SFR dt = 1.0000 M☉` check verifies the integral matches
# the constraint.
#
# Each subsection prints the SFR(t) shape agreement as well as the mass
# integral: the two codes sample the same analytic shape on different grids,
# so CIGALE is interpolated onto tengri's points and the median and
# worst-point ratios are reported. The plotted maximum of the delayed curve
# falls at whichever log node is nearest t = τ rather than at τ itself, which
# is where the grid is and not a different peak.
#
# **Verification Status:** PARTIAL (11/33) — Parametric SFH family physics

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
    neb={"type": "ssp"}, redshift=Fixed(0.0),
)
_state_sfh = _m_sfh.predict_state({})
_lbt_yr = np.asarray(_state_sfh.derived["sfh_grid_lbt_yr"])
_sfr_history = np.asarray(_state_sfh.derived["sfr_history"])
# Convert lookback time → cosmic age since SF onset (consistent with the
# CIGALE x-axis above): t_cosmic = age_gyr - lbt
t_t = (age_gyr - _lbt_yr / 1e9) * 1e9  # yr
sfr_t = _sfr_history
# Verify normalization: trapezoid of SFR over cosmic-age axis should be
# 10**log_total_mass = 1.0 M☉ within numerical accuracy of the n_grid pipeline.
# tengri's pipeline carries sfh_grid in decreasing lookback time, so
# integrate against the increasing-time order.
_idx = np.argsort(t_t)
_mass_formed = float(np.trapezoid(sfr_t[_idx], t_t[_idx]))
print(f"tengri pipeline ∫SFR dt = {_mass_formed:.4f} M☉ (target: 1.0000 from log_total_mass=0)")


def _sfr_shape_report(label, t_c_yr, sfr_c_arr, t_t_yr, sfr_t_arr, age_gyr):
    """Print the SFR(t) shape agreement on tengri's own grid.

    The two codes sample the same shape on different grids — CIGALE on a
    uniform 1-Myr axis, tengri on a 256-point log-spaced lookback axis — so
    the comparison is CIGALE interpolated onto tengri's points, over the
    interior of the history (the first 2 % and last 1 % of the age are
    dropped: both ends are grid edges, not physics).
    """
    _c_on_t = np.interp(t_t_yr, np.asarray(t_c_yr), np.asarray(sfr_c_arr), left=np.nan, right=np.nan)
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


_sfr_shape_report("§2 delayed", t_c, sfr_c, t_t, sfr_t, age_gyr)

fig, ax_l, ax_r = U.two_panel_fig()
for ax, title in (
    (ax_l, "pcigale.sed_modules.sfhdelayed (τ=1 Gyr, age=5 Gyr)"),
    (ax_r, "tengri pipeline sfr_history (n_grid=256 log-lbt)"),
):
    ax.set_xlabel("Cosmic age since SF onset [Gyr]")
    ax.set_ylabel(r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]")
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
ax_l.plot(t_c / 1e9, sfr_c, "C0-", linewidth=2.0)
ax_l.axvline(1.0, color="gray", linestyle=":", alpha=0.6, label=r"$\tau$ = 1 Gyr")
ax_l.legend(fontsize=9)
ax_r.plot(t_t / 1e9, sfr_t, "C1-", linewidth=2.0)
ax_r.axvline(1.0, color="gray", linestyle=":", alpha=0.6, label=r"$\tau$ = 1 Gyr")
ax_r.legend(fontsize=9)

fig.tight_layout()
save_fig("cigale_02_sfh_tau.png")


# %% [markdown]
# ### Note on the FSPS / BAGPIPES declining-exponential
#
# CIGALE's `sfh2exp(f_burst=0)` and FSPS's `sfh=1` both produce a
# declining exponential peaking at galaxy formation — a *different*
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
# onset, and tengri's pipeline grid still integrates to the requested mass.
#
# **The burst onset is a step, and the two codes resolve it differently.**
# `f_burst` turns the burst on discontinuously at `burst_age`. CIGALE
# evaluates its SFH on a uniform 1-Myr grid, so it puts a grid point on each
# side of the step; tengri's 256-point log-spaced lookback grid has one point
# inside it, and that point straddles the discontinuity. The printed
# comparison is the whole story: one grid point of 122 lands more than 1 %
# from CIGALE, and the median is 1.00043×. It is a sampling difference at a
# discontinuity, not a difference in the SFH — the mass integral, which does
# not care where the grid points fall, agrees to 4 decimal places.

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
    neb={"type": "ssp"}, redshift=Fixed(0.0),
)
_st_2exp = _m_2exp.predict_state({})
_lbt_2exp = np.asarray(_st_2exp.derived["sfh_grid_lbt_yr"])
_sfr_2exp = np.asarray(_st_2exp.derived["sfr_history"])
_t_2exp = (_age_gyr_2exp - _lbt_2exp / 1e9) * 1e9  # cosmic age since SF onset [yr]
_idx2 = np.argsort(_t_2exp)
_mass_2exp = float(np.trapezoid(_sfr_2exp[_idx2], _t_2exp[_idx2]))
print(f"tengri pipeline ∫SFR dt = {_mass_2exp:.4f} M☉ (target: 1.0000 from log_total_mass=0)")
_sfr_shape_report("§2 sfh2exp", t_c2, sfr_c2, _t_2exp, _sfr_2exp, _age_gyr_2exp)

fig, ax_l, ax_r = U.two_panel_fig()
for ax, title in (
    (ax_l, "pcigale.sed_modules.sfh2exp (main + burst)"),
    (ax_r, "tengri pipeline sfr_history (sfh2exp)"),
):
    ax.set_xlabel("Cosmic age since SF onset [Gyr]")
    ax.set_ylabel(r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]")
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
ax_l.plot(t_c2 / 1e9, sfr_c2, "C0-", linewidth=2.0)
ax_l.axvline(_age_gyr_2exp - 0.3, color="gray", linestyle=":", alpha=0.6, label="burst onset")
ax_l.legend(fontsize=9)
ax_r.plot(_t_2exp / 1e9, _sfr_2exp, "C1-", linewidth=2.0)
ax_r.axvline(_age_gyr_2exp - 0.3, color="gray", linestyle=":", alpha=0.6, label="burst onset")
ax_r.legend(fontsize=9)
ax_l.set_ylim(bottom=0.0)
ax_r.set_ylim(bottom=0.0)
fig.tight_layout()
save_fig("cigale_02_sfh2exp.png")


# %% [markdown]
# ## §3 Integrated stellar SED
#
# The τ-delayed SFH convolved with the BC03 SSPs. No dust, no nebular.
# CIGALE is normalized to 1 M☉ formed by construction; tengri's stellar
# mass formed is reported in the annotation.
#
# **The one convention difference on this SSP, and where it goes.** The two
# codes place the same star formation on the same SSP age grid in different
# ways, and the whole rest of the notebook inherits the result, so it is
# stated once here.
#
# tengri integrates the SFH onto the age axis with a cloud-in-cell kernel
# whose integrand extends to lookback zero, so the youngest SSP node (1 Myr
# on this grid) receives the star formation of the whole `[0, 1 Myr]` parcel.
# CIGALE gives that node exactly one 1-Myr bin and ages every star by +1 Myr
# (`pcigale/sed_modules/bc03.py`). tengri therefore carries recent star
# formation that a native-age binning drops, and since the 1–10 Myr
# population dominates the ionizing and far-UV output, the difference is
# strongly wavelength-dependent. Measured on this pair, printed below:
#
# | quantity | tengri / CIGALE |
# |---|---|
# | 200–912 Å | 1.144× |
# | 912–1200 Å | 1.057× |
# | 1200–3000 Å | 1.027× |
# | 0.3–1 µm | 1.001× |
# | stellar L_bol | 1.016× |
# | Q_H (λ < 911.76 Å) | 1.150× |
#
# This is tengri's documented convention, not a defect, and it is ratcheted
# by `tests/crossval/test_dsps_csp_uv_dense_reference.py` — which pins the
# sign and the size of the rest-UV excess against a native-age
# re-integration of tengri's own SFH, on this very grid. The ionizing budget
# is why: the knot moves Q_H *toward* the analytic continuous SFH→SSP
# convolution, so the side that drops the parcel is the one that is
# incomplete.
#
# Three later sections are this table and nothing else. §6's dust IR is
# normalized to `L_absorbed`, which inherits the 1.6 % L_bol offset weighted
# toward the UV, and lands at 1.021× CIGALE's `dust.luminosity`. §11's
# synchrotron is anchored on the same luminosity, so the 1.021 appears there
# as a flat factor. §8's line luminosities scale with Q_H. None of those is
# a separate discrepancy.

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
    neb={"type": "ssp"}, redshift=Fixed(0.0),
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
_opt = (w_c >= 1e3) & (w_c <= 1e4) & (L_c > 0)
print(f"§3 stellar tengri/CIGALE median (0.1–1 µm): {float(np.median(ratio[_opt])):.3f}×")

# The age-binning convention, band by band. This is the table the §3 prose
# quotes, and the term every later section inherits.
_L_t_on_c3 = U.regrid(np.asarray(s_stellar.wave), np.asarray(s_stellar.sed_intrinsic), w_c)
_r3 = _L_t_on_c3 / np.where(L_c > 0, L_c, np.nan)
print("§3 intrinsic stellar SED, median ratio by band (tengri / CIGALE):")
for _lo, _hi, _wname in (
    (200.0, 912.0, "200–912 Å  "),
    (912.0, 1200.0, "912–1200 Å "),
    (1200.0, 3000.0, "1200–3000 Å"),
    (3000.0, 10000.0, "0.3–1 µm   "),
):
    _m3 = (w_c >= _lo) & (w_c <= _hi) & np.isfinite(_r3)
    print(f"    {_wname}: {float(np.median(_r3[_m3])):.3f}×")
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
save_fig("cigale_03_stellar_sed.png")


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
# that are not the same law. `modified_CF00` needs two curves on each side —
# CIGALE attenuates young stars through the birth cloud *and* the ISM
# (`Av_BC = Av_ISM(1−µ)/µ = 1.53` at the defaults), so a single
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
# Below the Lyman limit the two diverge by construction — CIGALE's curves are
# zero there, tengri's polynomial continues unless
# `dust_attenuation={'lyman_cutoff': True}` clips it. Every applied-dust
# section below sets that flag; see §5.
#
# **Verification Status:** CROSSVAL — Attenuation law library

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
        C.attenuation_curve("dustatt_modified_starburst", wave_law, **{**_BUMP_KW, "bump_ampl": 3.0}),
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
# **Verification Status:** PARTIAL (2/88) — Two-component attenuation (birth cloud + diffuse)

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
    neb={"type": "ssp"}, redshift=Fixed(0.0),
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
    neb={"type": "ssp"}, redshift=Fixed(0.0),
)
s_d = m_d.predict_state({})
_assert_comparable(L_c_d, s_d.derived["sed_dust_attenuated"], name="§5 dust applied")

# The four panels are two comparisons — intrinsic and attenuated — and a
# side-by-side layout with independent y-axes cannot show a normalization
# offset in either. One number each.
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
        # Where the worst point is, not just how bad it is: on this ~20 Å
        # optical grid the extremum is a single point in a line or a deep
        # absorption trough, and the median is what the panel shows.
        print(
            f"    {_name} {_wname}: median {float(np.median(_r5[_m5])):.4f}×, "
            f"max |Δ| {float(_d5.max()) * 100:.2f}% at "
            f"{float(_wc[_m5][int(np.argmax(_d5))]):.0f} Å"
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
fig.savefig(str(figs_dir / "cigale_05_dust_attenuation_applied.png"), dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## §6 Dust IR re-emission and energy balance
#
# CIGALE re-emits absorbed stellar UV/optical through the Dale et al. (2014)
# template family (α = 2); tengri evaluates the same templates and enforces
# energy balance, $L_{\rm IR,\,emitted} \equiv L_{\rm absorbed}$, to
# floating-point — the residual is annotated on the right panel and printed
# below, and it is exactly zero.
#
# **The energy anchor.** tengri's `L_absorbed` sits 2.1 % above CIGALE's
# `dust.luminosity` (printed below). That is §3's age-binning convention
# arriving here: the absorbed luminosity is an integral over the attenuated
# far-UV and optical, weighted toward the wavelengths where tengri's youngest
# SSP node adds flux. The two codes' attenuation *curves* are not the cause —
# §4 measures them agreeing to 0.000 % away from the 1500–1800 Å crossover,
# and the absorbed fraction they produce differs by less than half a percent.
# Since the IR is normalized to that anchor, it appears in the printed
# 10–100 µm median as a floor under whatever the templates themselves do.
#
# **Lyman continuum.** tengri's attenuation curves polynomial-extend
# through the FUV; CIGALE zeros attenuation below 912 Å, and the models here
# set `dust_attenuation={'lyman_cutoff': True}` to match. This changes the
# emergent far-UV continuum, which is what §7's panel shows, and *not* the IR
# budget: `L_absorbed` masks λ < 912 Å unconditionally, on the reasoning that
# Lyman-continuum photons ionize hydrogen rather than heat dust. The IR here
# is bit-identical with and without the flag.
#
# **Verification Status:** CROSSVAL — Dust IR emission vs BAGPIPES

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
    dust_emission={"type": "dale2014_cigale", "alpha_dale": Fixed(2.0), "all_params": Fixed(DEFAULT)},
    neb={"type": "ssp"}, redshift=Fixed(0.0),
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
_fir = (w_c_ir >= 1e5) & (w_c_ir <= 1e6) & (L_c_ir > 0)
print(f"§6 dust IR tengri/CIGALE median (10–100 µm): {float(np.median(ratio[_fir])):.3f}×")

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
print(
    f"§6 absorbed fraction: CIGALE {_L_DUST_C / _L_STAR_C:.6f}, "
    f"tengri {L_abs / _L_star_t:.6f}"
)
print(f"§6 energy balance |L_IR − L_abs| / L_abs = {residual:.3e}")
fig.tight_layout()
fig.savefig(str(figs_dir / "cigale_06_dust_ir_dale2014.png"), dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ### Dust-IR model knobs: AGN heating and the radiation-field slope
#
# Two further CIGALE dust-IR parameters. Both panels sit on the same
# energy anchor as §6 — `_knob_model` uses the notebook's fiducial
# attenuation block, a single screen (`tau_bc = 0`, matching
# `dustatt_modified_starburst`, which has no Charlot & Fall birth cloud;
# A_V = R_V × E(B−V)_cont = 4.05 × 0.132 = 0.535 mag) carrying the
# Leitherer-extended Calzetti curve with the 912 Å clip. So the 2.1 % anchor
# offset of §3/§6 enters every ratio printed here, and what the panels add is
# whatever the *templates* do on top of it.
#
# **Left — Dale 2014 AGN fraction (`dale2014.fracAGN`).** `fracAGN` adds an
# AGN-heated source as a separate power budget ($L_{\rm AGN}=L_{\rm
# dust}\,f/(1-f)$) using CIGALE's own `model_quasar` template
# ($SED = L\,T_{\rm SF}(\alpha) + L_{\rm AGN}\,T_{\rm QSO}$, `dale2014.py`).
# The subtlety that makes the mixing work: CIGALE normalizes `model_quasar`
# to unit luminosity over its full native grid (~60 nm onward), where ~46 %
# of the quasar energy is the UV/optical accretion-disc continuum below the
# dust grid's blue edge. Only its ~0.42 IR share enters the dust mixing, and
# tengri carries that same partition. The cell prints the 3–8 µm and
# 8–1000 µm median ratio at each $f_{\rm AGN}$, so the question of whether
# the mid-IR lift drifts with $f_{\rm AGN}$ is answered by three numbers
# rather than by eye.
#
# **Right — THEMIS slope $\alpha$ (`themis.alpha`, $dU/dM \propto
# U^{-\alpha}$, matched `qhac=0.17, umin=1.0, gamma=0.1`).** tengri's THEMIS
# templates are built from the published DustEM grids (Jones+2017) and
# conserve the absorbed energy. The a-C(:H) aromatic fraction `qhac` is
# pinned to CIGALE's 0.17 on both sides. CIGALE quotes it as a fraction
# while the DustEM grid tabulates it in FSPS scaling (`qhac × 100/2.2`), so
# the two must be reconciled before interpolation or the wrong grain model
# is selected. One input is *not* matched and cannot be: CIGALE's own DustEM
# run used `umax = 1e7`, a slightly hotter PDR. The cell prints both an
# 8–30 µm and an 8–1000 µm ratio at each α, so a redistribution of the IR
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
        dust_emission={"type": emission_type, "all_params": Fixed(DEFAULT), **emkw},
        neb={"type": "ssp"}, redshift=Fixed(0.0),
    )


def _nu_lnu(wave_aa, l_nu):
    w = np.asarray(wave_aa)
    return w, _c_aa_dust / w * np.asarray(l_nu)


def _band_median(ratio, wave_aa, lo_um, hi_um):
    """Median of ``ratio`` over ``[lo_um, hi_um]`` on the Å grid ``wave_aa``."""
    m = (wave_aa >= lo_um * 1e4) & (wave_aa <= hi_um * 1e4) & np.isfinite(ratio)
    return float(np.median(ratio[m])) if m.any() else float("nan")


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.4))

# Where the two codes agree, a thin pcigale line simply disappears under
# tengri's — the panel then reads as though only one code were plotted, which
# is the opposite of the point. Draw pcigale as a thick translucent band and
# tengri as a thin line on top (the convention the AGNFITTER-RX notebook uses):
# agreement shows as a crisp line running down the middle of its own halo, and
# any divergence separates the two immediately.
_REF_KW = dict(lw=4.0, alpha=0.35, solid_capstyle="round")
_TNG_LW = 1.4
_peaks = []

# LEFT — Dale 2014 AGN fraction: pcigale dale2014.fracAGN (band) vs tengri (line).
m_frac = _knob_model("dale2014_cigale", alpha_dale=Fixed(2.0))
p_frac = dict(m_frac.spec.sample(jax.random.PRNGKey(0)))
print("§6 knobs — Dale 2014 fracAGN sweep (tengri / CIGALE, median in band):")
for f, c in zip([0.0, 0.3, 0.6], ["C0", "C1", "C3"]):
    sed = C.run_chain(
        [_SFH_CHAIN, _BC03_CHAIN, _DUSTATT_CHAIN, ("dale2014", dict(alpha=2.0, fracAGN=f))]
    )
    w_c, nl_c = _nu_lnu(*C.to_lnu(sed))
    ax_l.loglog(w_c, nl_c * _KNOB_MASS, color=c, **_REF_KW)
    o = m_frac.predict_rest_sed({**p_frac, "dust_frac_agn": jnp.float64(f)})
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_l.loglog(w_t, nl_t, color=c, lw=_TNG_LW, label=rf"$f_{{\rm AGN}}={f}$")
    _peaks.append(float(np.nanmax(nl_t)))
    _rk = U.regrid(w_t, nl_t, w_c) / np.where(nl_c > 0, nl_c * _KNOB_MASS, np.nan)
    print(
        f"    f_AGN = {f}:  3–8 µm {_band_median(_rk, w_c, 3.0, 8.0):.4f}×"
        f"   8–1000 µm {_band_median(_rk, w_c, 8.0, 1000.0):.4f}×"
    )
ax_l.plot([], [], "k-", **_REF_KW, label="pcigale")
ax_l.plot([], [], "k-", lw=_TNG_LW, label="tengri")
ax_l.set(
    xlim=(1e4, 1e7),
    xlabel=r"$\lambda$ [Å]",
    ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]",
    title="Dale 2014 AGN fraction",
)
ax_l.legend(fontsize=8, frameon=False, ncol=2)

# RIGHT — THEMIS slope alpha, matched qhac=0.17, umin=1.0, gamma=0.1.
m_alpha = _knob_model("themis", dust_gamma_dl=Fixed(0.1), dust_qhac=Fixed(0.17))
p_alpha = dict(m_alpha.spec.sample(jax.random.PRNGKey(0)))
print("§6 knobs — THEMIS α sweep (tengri / CIGALE, median in band):")
for a, c in zip([1.0, 2.0, 3.0], ["C0", "C1", "C3"]):
    sed = C.run_chain(
        [
            _SFH_CHAIN,
            _BC03_CHAIN,
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
    print(
        f"    α = {a}:  8–30 µm {_band_median(_rk, w_c, 8.0, 30.0):.4f}×"
        f"   8–1000 µm {_band_median(_rk, w_c, 8.0, 1000.0):.4f}×"
    )
ax_r.plot([], [], "k-", **_REF_KW, label="pcigale")
ax_r.plot([], [], "k-", lw=_TNG_LW, label="tengri")
ax_r.set(xlim=(3e4, 1e7), xlabel=r"$\lambda$ [Å]", title=r"THEMIS radiation-field slope $\alpha$")
ax_r.legend(fontsize=8, frameon=False, ncol=2)

# Five decades below the peak: at alpha = 1 the radiation field is hot enough
# that the FIR bump falls away steeply, and a shallower floor cuts the curve
# off mid-decline as though the model had stopped.
_ypk = max(_peaks)
for ax in (ax_l, ax_r):
    ax.set_ylim(_ypk * 1e-5, _ypk * 2.0)

fig.tight_layout()
save_fig("cigale_06_dust_ir_knobs.png")
plt.show()


# %% [markdown]
# ## §7 Panchromatic SED
#
# Same model, viewed across 1 Å (X-ray) to 10 m (radio). What appears
# in the X-ray and radio panels arrives in §10 and §11.
#
# **Far-UV (λ < 1000 Å) — now matched.** Calzetti+2000 was fit on
# 1200 Å – 22000 Å; tengri's polynomial extrapolates below that, while
# CIGALE's `dustatt_modified_starburst` drops to zero at 912 Å. Setting
# `dust_attenuation={'lyman_cutoff': True}` applies the same 912 Å clip on both sides.

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
_cov_pan = (L_c_ir > 0) & (_L_t_pan > 0)
_r_pan = np.where(_cov_pan, _L_t_pan / np.where(L_c_ir > 0, L_c_ir, 1.0), np.nan)
print("§7 panchromatic median ratio (tengri / CIGALE):")
for _lo, _hi, _wname in (
    (200.0, 912.0, "200–912 Å (LyC) "),
    (912.0, 1200.0, "912–1200 Å      "),
    (1200.0, 3000.0, "1200–3000 Å     "),
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
            f"    {_wname}: {float(np.nanmedian(_r_pan[_band])):.4f}×  "
            f"({_n_both}/{_n_c} points where both grids carry a model)"
        )
    else:
        print(f"    {_wname}: tengri's grid does not reach this band ({_n_c} CIGALE points)")

fig, (ax_l, ax_r) = plt.subplots(1, 2, sharey=True, figsize=(12, 5))
U.panel(
    ax_l, ax_r, label_l="pcigale  fiducial chain", label_r="tengri  sfh.delayed + dust.dale2014"
)
ax_l.plot(w_c_ir, L_c_ir, "C0-", linewidth=1.5)
ax_r.plot(s_ir.wave, s_ir.sed_intrinsic, "C1-", linewidth=1.5)
_xmin_p = float(min(w_c_ir.min(), float(np.asarray(s_ir.wave).min())))
_xmax_p = float(max(w_c_ir.max(), float(np.asarray(s_ir.wave).max())))
# Frame the y-axis on the SED peak. Without this the panchromatic SED cliffs
# to ~0 at the grid edges and the shared log axis autoscales across ~170
# decades, crushing the real SED into a flat line at the top (it spans only
# ~6 decades). Match the peak-anchored framing used by the other §-panels.
_ymax_p = float(max(np.nanmax(L_c_ir), np.nanmax(np.asarray(s_ir.sed_intrinsic))))
for ax in (ax_l, ax_r):
    ax.set_xlim(_xmin_p, _xmax_p)
    ax.set_ylim(_ymax_p * 1e-6, _ymax_p * 2.0)
    ax.grid(True, alpha=0.3)
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
# **Fiducial choice — young population.** Sections §3–§7 use a 5 Gyr
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
# **Read the ionizing budget before the line ratios.** Every line scales with
# the Q_H handed to the emitter, so the cell prints Q_H first: CIGALE's
# `stellar.n_ly` against tengri's `nion`, on the shared BC03 grid and on the
# dense FSPS grid the lines are measured on. Two known terms enter it. On the
# shared grid it is §3's age-binning convention alone. On the dense grid the
# SSP swap adds to it — deliberate, and the reason for it is below.
#
# **What is left after that is the emitter.** Cue was trained on Cloudy 17
# (Li et al. 2025) while CIGALE bundles Cloudy 13.x grids, and Cue's
# bare-stellar path differs from CIGALE's wNE-SSP convolution. The printed
# line ratios divide out neither term, so they are an upper bound on the
# emitter difference rather than a measurement of it — and they do not move
# together: the recombination lines and [O III] behave differently, which is
# what a line-physics or abundance difference looks like and not what a
# uniform normalization offset looks like. Closing this properly needs
# tengri's own static grid at matched Q_H (`neb={'type': 'cloudy'}`, below),
# a Cloudy-against-Cloudy comparison; this notebook does not run it.
#
# Each line is quantified with the **integrated** luminosity
# (continuum-subtracted, width-independent), not a single-bin peak ratio,
# which would measure line width and grid resolution (CIGALE broadens to
# `lines_width = 300 km/s`) rather than physics.
#
# **Grid coverage.** Cue's native grid runs ~915 Å – 10⁸ Å (optical/UV
# forest); CIGALE's CLOUDY grid extends to far-IR fine-structure lines
# ([O III] 88 μm, [C II] 158 μm, [S III] 18.7 μm, [Ne III] 15.6 μm out to
# ~10⁶ Å), which is why the left panel shows line spikes the Cue panel does
# not. For a CLOUDY-vs-CLOUDY match, tengri exposes its own static grid via
# `neb={'type': 'cloudy'}` (`data/cloudy_grid_*.h5`, 166 lines to 6.1×10⁶ Å).
#
# **Verification Status:** PARTIAL (1/10) — Cue nebular emulator

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

m_no_neb = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh=_neb_sfh_kw,
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    neb={"type": "ssp"}, redshift=Fixed(0.0),
)
s_no_neb = m_no_neb.predict_state({})

m_neb = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh=_neb_sfh_kw,
    neb={
        "type": "cue",
        "neb_logU": Fixed(-2.0),
        "neb_logZ_gas": Fixed(MET_LOGZSOL),  # Z_gas = 0.02 ≡ stellar Z
        "neb_fesc": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },  # ionspec_* slopes stay at their SSP-derived Fixed values
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
# Cue-only dotted). Agreement is limited by Cloudy version (17 vs
# 13.x), bare-stellar vs wNE-SSP path, and line-broadening kernel;
# see the integrated-line-luminosity ratio printed below.
# Attaching Cue extends the master grid to ~10⁸ Å via the
# native-grid union (cue_weights.npz/cont_wavelength), so s_neb has more
# points than s_no_neb. Regrid the no-neb baseline onto the with-Cue
# wave grid before subtracting.
_s_no_neb_on_neb = U.regrid(
    np.asarray(s_no_neb.wave),
    np.asarray(s_no_neb.sed_intrinsic),
    np.asarray(s_neb.wave),
)
L_t_neb_only = np.maximum(np.asarray(s_neb.sed_intrinsic) - _s_no_neb_on_neb, 1e-30)
ax_r.plot(
    s_no_neb.wave, s_no_neb.sed_intrinsic, "k--", linewidth=1.0, alpha=0.5, label="stellar only"
)
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
save_fig("cigale_08_nebular_cue_vs_cloudy.png")

# Quantify the residual by integrated line luminosity (width- and grid-
# independent). A single-bin peak ratio measures line width, not luminosity —
# CIGALE broadens its lines (lines_width=300 km/s) while Cue applies its own.
#
# CIGALE's native BC03 grid (repackaged unchanged into `bc03_from_cigale.h5`) is
# non-uniform and only ~20 Å in the optical — that is CIGALE's own spectral
# resolution, not a repackaging artifact, and it cannot resolve an emission line (a
# ±12 Å window holds a single grid point). So the integrated measure is taken on
# the dense FSPS MIST+MILES bare-stellar SSP that Cue was validated on (the same
# SSP swap the ProSpect notebook §8 makes, and for the same reason). The CIGALE
# reference stays CLOUDY-on-BC03.
_ssp_neb_dense = load_ssp_data(
    str(tengri.download_ssp("fsps_mist_miles_chabrier", dest=str(_HERE / "_drivers" / "data")))
)
_m_neb_dense = SEDModel.build(
    ssp_data=_ssp_neb_dense,
    # CIGALE runs this section at Z = 0.02 for both the stars (bc03
    # metallicity=0.02) and the gas (nebular zgas=0.02). Pin both sides of
    # the dense-SSP twin to the same absolute metallicity — a bare
    # ``logzsol = 0.0`` here would be Z = 0.0142, a 0.15 dex mismatch on top
    # of the disclosed SSP swap, and it moves the metal lines much more than
    # the recombination lines.
    met=MET_FIDUCIAL,
    sfh=_neb_sfh_kw,
    neb={
        "type": "cue",
        "neb_logU": Fixed(-2.0),
        "neb_logZ_gas": Fixed(MET_LOGZSOL),
        "neb_fesc": Fixed(0.0),
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
_s_neb_dense = _m_neb_dense.predict_state({})
_w_t_dense = np.asarray(_s_neb_dense.wave)
_L_t_dense = np.asarray(_s_neb_dense.derived["sed_nebular"])

# The ionizing budget first. Every line scales with the Q_H handed to the
# emitter, so a line ratio read without it cannot separate "different line
# physics" from "different number of ionizing photons". CIGALE publishes it
# as ``stellar.n_ly``; tengri as ``derived["nion"]``. Both sides form 1 M☉,
# so the two are directly comparable.
_n_ly_c = float(sed_c_neb.info["stellar.n_ly"])
print("§8 ionizing photon rate Q_H reaching the emitter [ph/s per M☉ formed]:")
print(f"    CIGALE  stellar.n_ly on BC03      {_n_ly_c:.4e}")
for _label, _st in (
    ("tengri  shared BC03 SSP          ", s_neb),
    ("tengri  dense FSPS MIST+MILES    ", _s_neb_dense),
):
    _qh = float(np.asarray(_st.derived["nion"]))
    print(f"    {_label}{_qh:.4e}  → {_qh / _n_ly_c:.3f}×")

print("§8 integrated line luminosity (tengri Cue / CIGALE CLOUDY; tengri on dense FSPS SSP):")
for _c, _name in [(6563.0, "Hα"), (5007.0, "[O III]"), (4861.0, "Hβ")]:
    _lc = U.line_lum(w_c_neb, L_c_neb_only, _c)
    _lt = U.line_lum(_w_t_dense, _L_t_dense, _c)
    if _lc > 0:
        print(
            f"    {_name} {_c:.0f} Å: CIGALE {_lc:.2e}, tengri {_lt:.2e} erg/s → {_lt / _lc:.2f}×"
        )
    else:
        # Not skipped silently: an empty CIGALE window is a grid-resolution
        # failure of the measurement, not an absent line.
        print(f"    {_name} {_c:.0f} Å: CIGALE window holds < 2 grid points — not measurable here")


# %% [markdown]
# ## §9 AGN
#
# CIGALE's `skirtor2016` AGN package on the left vs tengri's composable
# AGN on the right, both at the SKIRTOR fiducial (i = 30°, τ_9.7 = 7,
# oa = 40°, p = q = 1).
#
# **`disc.schartmann2005` matches CIGALE's `skirtor2016 disk_type=1`**
# default: the Schartmann (2005) piecewise power law with the 1200 Å
# bend that CIGALE substitutes for the FITS-bundled disc. tengri also
# ships `disc.skirtor` (CIGALE `disk_type=0`, §9b) and `disc.adaf_lopez2024`
# (`disk_type=2`).
#
# A differentiable alternative (M_BH, ṁ, spin) is available as
# `disc={"type": "multicolor", ...}`, a Shakura-Sunyaev disc. It carries
# a far-UV bump separated from the optical Wien tail by a notch near
# 5000 Å and diverges from CIGALE in the disc UV continuum, though the
# torus IR still matches. This notebook uses `disc.schartmann2005` to
# match CIGALE's default.
#
# **Three components, and all three have to be selected.** CIGALE's
# `skirtor2016` emits a disc, a torus and a Casey-2012 polar-dust graybody,
# and publishes them separately. tengri's composable AGN has the same three,
# but the polar graybody is a standalone `atten` block — it is not bundled
# with the torus, and `derived["sed_agn_polar"]` is a zeros array whenever the
# block is not selected. The build below therefore names all three, at
# CIGALE's own defaults (`oa = 40°`, τ_9.7 = 7, p = q = 1, i = 30°,
# `agn_torus_frac = 0.5`, `agn_polar_ebv = 0.03`, T = 100 K, β = 1.6). §9c
# prints the disc, torus and polar-dust luminosities separately, which is the
# only way a FIR residual can be attributed: at 100 µm all three overlap and
# a band ratio cannot say which one carries it.
#
# **The polar component is disc-shaped by construction.** Both codes set its
# luminosity from `g(oa) × ∫ disc(1 − e^−τ_polar) dλ` — the disc's own spectrum
# seen through the polar screen — so it moves with the disc's UV shape even at
# fixed `oa`, `agn_polar_ebv`, T and β, and it is not a component either code
# can be expected to reproduce independently of which disc is selected. §9c and
# §9b print each code's polar share of its own AGN dust budget beside the
# component ratios: pcigale's share moves between its two disc types, and
# tengri's moves with it.
#
# The torus IR uses the same templates and reproduces at all inclinations:
# at i = 30° both sides peak at ~6–9 μm, and edge-on viewing pushes the dust
# peak out to ~30 μm.
#
# **Energy-balance normalization.** CIGALE ties disc, torus, and polar
# dust to a single `agn_power` reference through the fixed SKIRTOR
# template ratios. tengri's default `norm='cigale_joint'` does the same:
# the disc is tied to `agn_power × R`, where `R = η(i)·∫disc/∫dust` and
# `η(i) = cos i (1+2cos i)/3` is the Stalevski+2016 anisotropy factor
# (η = 0.789 at i = 30°). `agn_power` itself is not a free input here — with
# `agn_ir_frac` set it is derived from `L_absorbed × f/(1−f)`, exactly as
# CIGALE derives it from `fracAGN`, so §6's energy anchor enters the AGN
# normalization too. That coupling exists only under `'cigale_joint'`:
# `norm='independent'` decouples the disc from the absorbed-energy budget and
# is for builds that set no `agn_ir_frac`. Writing `'independent'` or
# `'conserving'` beside an active `agn_ir_frac` states two normalizations at
# once, and the build refuses it rather than silently dropping one.
#
# **Verification Status:** CROSSVAL — SKIRTOR torus (mean 3-param)

# %%
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
    neb={"type": "ssp"}, redshift=Fixed(0.0),
)
s_agn_base = m_agn_base.predict_state({})

# Full SED with and without AGN on both sides — clearer than the
# differential plot when one side has X-ray + the other doesn't.
sed_skirtor = C.run_chain(
    [
        _sfh_args_d,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
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
    # tengri's AGN defaults already match CIGALE skirtor2016's (oa=40,
    # tau=7, p=q=1, i=30, EBV=0.03, T=100, β=1.6, disk_type=1 →
    # ``disc.schartmann2005``). The differentiable multicolor disc remains
    # available — ``disc={"type": "multicolor", ...}``.
    agn={
        "type": "composable",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        # The Casey-2012 polar-dust graybody is its own attenuation block,
        # not a rider on the torus: without this entry the model has no
        # polar component at all and ``derived["sed_agn_polar"]`` is zeros,
        # so the comparison would put disc+torus against CIGALE's
        # disc+torus+polar chain. ``agn_polar_ebv = 0.03``, T = 100 K and
        # β = 1.6 are CIGALE's skirtor2016 defaults on both sides.
        "atten": {
            "type": "polar_dust",
            "agn_polar_ebv": Fixed(0.03),
            "all_params": Fixed(DEFAULT),
        },
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
    neb={"type": "ssp"}, redshift=Fixed(0.0),
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
# The two `stellar + dust + AGN` SEDs on a shared grid, with the tengri / CIGALE
# ratio and per-band readouts (disc UV, torus mid-IR, polar/FIR).

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
    ("disc UV    1500-3000 Å", 1500.0, 3000.0),
    ("torus MIR  5-30 µm    ", 5.0e4, 3.0e5),
    ("polar FIR  100 µm     ", 8.0e5, 1.2e6),
]:
    _m = (w_c_agn >= _lo) & (w_c_agn <= _hi) & (L_c_agn > 0)
    if _m.any():
        print(f"  {_name}: {float(np.median(ratio_agn[_m])):.3f}×")


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
    print(f"  disc shape, 1000–5000 Å, normalized at 2500 Å: max |Δ| = {_dev * 100:.2f}% ({_n} pts)")
    _pairs = {_tk: _agn_component_pair(sed_c, state_t, _ck, _tk) for _, _ck, _tk in _AGN_COMPONENTS}
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
save_fig("cigale_09c_agn_parity_ratio.png")
plt.show()


# %% [markdown]
# ### §9b The other disc — `disc.skirtor` ↔ CIGALE `disk_type=0`
#
# CIGALE's `skirtor2016` offers two analytic discs (`skirtor2016.py`):
# `disk_type=0` → `skirtor_disk()` and `disk_type=1` →
# `schartmann2005_disk()`. tengri ships both, as `disc.skirtor` and
# `disc.schartmann2005`; §9 pins the `disk_type=1` pairing and this one swaps
# to `disc.skirtor`, the shallower SKIRTOR disc with its 1200 Å bend. Only the
# disc changes: the torus, the polar-dust block and the `fracAGN` coupling are
# §9's, so the printed ladder repeats §9's with one entry moved.
#
# **What the disc-shape number contains.** Both codes publish the disc *after*
# the polar-dust screen, so the deviation printed below is the analytic disc
# shape and the two codes' SMC screens together — it is an upper bound on the
# disc difference, not a measurement of it. The screen does not cancel by
# comparing this section with §9: the same SMC law integrated against a
# different disc spectrum is a different factor, which is why the polar share
# printed here differs from §9's on *both* sides. What is left in the polar
# residual once the disc shape is accounted for is the two codes' SMC screens
# and their quadrature of the SKIRTOR face-on reference.

# %%
sed_skirtor0 = C.run_chain(
    [
        _sfh_args_d,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
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
        # Same polar-dust block and same fracAGN coupling as §9 — only the
        # disc changes, so the two panels differ in one input.
        "atten": {
            "type": "polar_dust",
            "agn_polar_ebv": Fixed(0.03),
            "all_params": Fixed(DEFAULT),
        },
        "agn_ir_frac": Fixed(0.3),
        "all_params": Fixed(DEFAULT),
    },
    neb={"type": "ssp"}, redshift=Fixed(0.0),
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
# ## §10 X-ray
#
# CIGALE's `xray` module follows Yang et al. (2020): an AGN corona power law
# plus X-ray binaries and hot gas. tengri ships `xray.yang20`. Three
# conventions are matched: α_ox is derived from L_2500 (Just et al. 2007,
# `L_2keV = L_2500 · 10^(α_ox/0.3838)`) on both sides at log₁₀ L_2500 = 29.47;
# inclination follows Yang+2020 at i = 30°; and N_H is pinned to zero on both
# sides. At `log N_H = 0` tengri's Ricci et al. (2017) scattered floor is 1 %
# of an unabsorbed corona and contributes nothing here, so it is not what the
# printed residual is; the residual is a fraction of a percent and the cell
# prints it at 2 keV and as a median over 0.5–10 keV.
#
# **Verification Status:** PARTIAL (3/16) — Radio + X-ray + AGN

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
        neb={"type": "ssp"}, redshift=Fixed(0.0),
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
# CIGALE's `radio` module is a pure star-forming synchrotron power law tied
# to the IR-radio correlation (`qir_sf`, `alpha_sf`). tengri's `radio.condon92`
# includes Murphy 2011 free-free (Eq. 11) plus synchrotron (Bell 2003). The
# build pins `radio_q_ir = 2.5` and `radio_alpha_sf = 0.8` to match. The
# synchrotron amplitude is anchored on `L_absorbed`: both codes compute
# `L_ref = L_dust / (3.75e12 · 10^q_IR)`, so any mismatch in the absorbed
# energy lands 1:1 in the radio.
#
# tengri sits above CIGALE across the band and the excess *grows with
# frequency* — the four printed total ratios climb monotonically from
# 150 MHz to 100 GHz. The shape is the diagnosis: a normalization error would
# offset the whole band by a constant, and this does not.
#
# Separate the ratio into the only two things it can be made of.
#
# **The non-thermal terms agree, and their offset is pure convention.** Both
# codes emit a synchrotron power law of the same index, so tengri's synchrotron
# over CIGALE's can only be a constant — and it is: flat across all three
# decades (printed, and the dotted line on the ratio panel). Two conventions
# predict that constant without reference to the measurement:
#
# - **Anchor frequency, ×0.985.** Bell 2003 defines q_IR at 1.4 GHz; CIGALE
#   normalizes at 21 cm = 1.4276 GHz — a pure `(1.4276/1.4)^−0.8` offset.
# - **Energy balance.** `L_absorbed` runs 2.1 % above CIGALE's
#   `dust.luminosity` (§6's printed anchor, and §3's age-binning convention
#   behind it — not an attenuation-curve difference). Both codes anchor the
#   synchrotron on the dust luminosity
#   (`L_ref = L_dust / (3.75e12 · 10^q_IR)`), so it lands in the radio 1:1.
#
# Their product is the measured flat offset; the cell prints the prediction
# and the measurement side by side. Nothing is fitted: the prediction comes
# from the two conventions, the measurement from the two SEDs, and they meet.
#
# **Everything above that line is thermal.** tengri's `condon92` carries a
# Murphy+2011 free-free term; CIGALE's `radio` module has none. Free-free
# is flat (α ≈ 0.1) where synchrotron is steep (α = 0.8), so its share climbs
# with frequency, which is the entire rise of the ratio panel — the printed
# thermal fraction at 0.15, 1.4, 10 and 100 GHz is that share. It is a physics
# difference between the two codes rather than a discrepancy in the shared
# physics.
#
# **Verification Status:** PARTIAL (3/25) — Radio / X-ray / IGM / PSD physics

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
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        ("dale2014", dict(alpha=2.0)),
        ("radio", dict(qir_sf=2.5, alpha_sf=0.8, R_agn=0.0, alpha_agn=0.7)),
    ]
)
# CIGALE's SF radio is a single synchrotron component — isolate it.
w_r, L_r = U.wnm_to_erg_per_hz_per_aa(
    np.asarray(sed_r.wavelength_grid), np.asarray(sed_r.luminosities["radio.sf_nonthermal"])
)

m_r = SEDModel.build(
    ssp_data=ssp,
    met=MET_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    # Same attenuation setup as §6 — the radio amplitude is anchored on
    # L_absorbed through q_IR, so a mismatched dust config here would leak
    # straight into the synchrotron normalization.
    dust_attenuation={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        "lyman_cutoff": True,
        "all_params": Fixed(DEFAULT),
    },
    dust_emission={"type": "dale2014_cigale", "alpha_dale": Fixed(2.0), "all_params": Fixed(DEFAULT)},
    # q_IR pinned to CIGALE's qir_sf = 2.5 (tengri bucket default 2.64).
    radio={
        "sf": {"type": "bell2003"},
        "agn": {"type": "powerlaw"},
        "radio_q_ir": Fixed(2.5),
        "radio_alpha_sf": Fixed(0.8),
        "all_params": Fixed(DEFAULT),
    },
    neb={"type": "ssp"}, redshift=Fixed(0.0),
)
state_r = m_r.predict_state({})
w_t = np.asarray(state_r.wave)
sed_t = np.asarray(state_r.derived["sed_radio"])  # synchrotron + Murphy free-free

# Shared-axis overlay + ratio panel: the ratio rising above unity toward high ν
# is tengri's free-free, which CIGALE's synchrotron-only module does not have.
fig, ax, ax_r, ratio = U.overlay_ratio_fig(
    w_r,
    L_r,
    w_t,
    sed_t,
    x_of_wave=lambda w: C_AA / w / 1e9,
    xlabel=r"$\nu$ [GHz]",
    title="§11 SF radio — CIGALE synchrotron vs tengri synchrotron + free-free",
    label_c="CIGALE  radio.sf_nonthermal (synchrotron only)",
    label_t="tengri  radio.condon92 (synchrotron + free-free)",
    xlim=(0.1, 100.0),
    ratio_ylim=(0.5, 2.0),
)
# Overlay tengri's synchrotron-only term (Bell 2003). It is the load-bearing
# curve of this panel: it lands *on* CIGALE's synchrotron-only sf_nonthermal,
# which is what makes the excess at high frequency attributable to the
# Murphy+2011 free-free CIGALE omits rather than to a synchrotron
# normalization error. Landing on top of CIGALE also made it invisible — so
# widen CIGALE into a translucent band and let the two thin curves read
# against it.
from tengri.radio import radio_sfr_bell2003 as _bell03

for _ln in ax.get_lines():
    if _ln.get_label().startswith("CIGALE"):
        _ln.set(linewidth=4.0, alpha=0.35, solid_capstyle="round")

_syn_only = np.asarray(
    _bell03(w_t, float(np.asarray(state_r.derived["L_ir"])), q_ir=2.5, alpha_sf=0.8)
)
ax.plot(
    C_AA / w_t / 1e9,
    _syn_only,
    color="0.25",
    ls=":",
    lw=1.6,
    label="tengri  synchrotron only (Bell 2003)",
)
ax.legend(fontsize=8, frameon=False)
_nu_r = C_AA / w_r / 1e9
_g14 = (_nu_r >= 1.0) & (_nu_r <= 1.5) & (L_r > 0)
print(f"§11 radio tengri/CIGALE median (1.0–1.5 GHz): {float(np.median(ratio[_g14])):.3f}×")

# Close the ratio *at every frequency*, not at one probe point. The prediction
# is purely the two convention/physics factors — thermal fraction (frequency
# dependent) and the fixed 21 cm-vs-1.4 GHz anchor — times the energy-balance
# anchor, which §3 and §6 already account for and which is printed here from
# this section's own SED. If that curve traces the measured ratio across three
# decades, the "radio deviation" is fully accounted for and nothing is left over.
_L_ir_t = float(np.asarray(state_r.derived["L_ir"]))
_f_lir = _L_ir_t / (float(sed_r.info["dust.luminosity"]) * 1e7)
_f_anchor = float((1.4276e9 / 1.4e9) ** (-0.8))

# Split the measured ratio into the two things it can be made of, and check
# each against a number derived *independently* of it.
#
# tengri's synchrotron alone, over CIGALE's synchrotron, is flat — the two
# codes' non-thermal terms have the same spectral index, so their quotient can
# only be a normalization. Predict that normalization from the two conventions
# (anchor frequency and the energy-balance anchor) and compare. Nothing about
# the prediction is fitted to the measurement, so their agreement is a real
# check rather than an identity.
_syn_on_c = np.asarray(_bell03(w_r, _L_ir_t, q_ir=2.5, alpha_sf=0.8))
_valid = (_nu_r >= 0.1) & (_nu_r <= 100.0) & (L_r > 0) & np.isfinite(ratio)
_syn_ratio = _syn_on_c[_valid] / L_r[_valid]
print(
    f"§11 synchrotron alone, tengri/CIGALE: "
    f"{_syn_ratio.min():.4f}–{_syn_ratio.max():.4f} across 0.1–100 GHz (flat)"
)
print(
    f"§11   predicted from conventions: anchor ×{_f_anchor:.4f} · "
    f"energy-balance ×{_f_lir:.4f} = ×{_f_anchor * _f_lir:.4f}"
)

# Everything the total ratio carries above that flat line is the thermal term —
# read off the model itself, not re-derived from a formula whose (T_e, thermal
# fraction) would have to be guessed back out of the component.
_ff_frac = ratio[_valid] / (_f_anchor * _f_lir) - 1.0
_order = np.argsort(_nu_r[_valid])
ax_r.axhline(
    _f_anchor * _f_lir,
    color="0.25",
    ls=":",
    lw=1.6,
    label=f"synchrotron only (anchor × energy balance = {_f_anchor * _f_lir:.3f})",
)
ax_r.legend(fontsize=7, frameon=False, loc="upper left")
print("§11 implied free-free excess over CIGALE (which has no thermal term):")
for _f in (0.15, 1.4, 10.0, 100.0):
    _j = int(np.argmin(np.abs(_nu_r[_valid] - _f)))
    print(
        f"    {_nu_r[_valid][_j]:6.2f} GHz: total ×{ratio[_valid][_j]:.3f} "
        f"→ thermal fraction {_ff_frac[_j] * 100:5.1f}%"
    )
fig.tight_layout()
save_fig("cigale_11_radio_synchrotron.png")


# %% [markdown]
# ## §12 IGM transmission
#
# CIGALE applies Meiksin (2006) IGM attenuation inside its
# `redshifting` module — Lyman series **and** the diffuse-IGM Lyα
# forest continuum suppression, so transmission redward of the Lyman
# limit at z = 3 sits at ~0.18–0.25 rather than 1. tengri ships the
# matching `igm.meiksin06`; this panel uses it directly so both sides
# apply the same Meiksin prescription. The printed max and median |ΔT| at
# z = 3, 5, 7 are at float precision with no point anywhere above 1e-3:
# tengri matches CIGALE's Meiksin transmission to the last digit, not just
# visually.
#
# **Verification Status:** CROSSVAL — Inoue+2014 IGM transmission

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
    print(
        f"    z = {z:.0f}: max |ΔT| = {float(_dT.max()):.2e}, "
        f"median |ΔT| = {float(np.median(_dT)):.2e}, "
        f"{int((_dT > 1e-3).sum())}/{_dT.size} points above 1e-3"
    )

ax_l.legend(fontsize=9)

ax_r.legend(fontsize=9)
fig.tight_layout()
save_fig("cigale_12_igm_transmission.png")


# %% [markdown]
# ## tengri in CIGALE-mode — the full X-ray → radio party SED
#
# The whole chain at once: shared BC03 SSP, fiducial τ-delayed SFH,
# modified-starburst attenuation, Dale+2014 IR re-emission, plus §10 X-ray
# (Yang+2020: XRB + hot gas, no AGN corona in this galaxy-only chain) and
# §11 radio (Condon 1992 SF synchrotron, `q_IR = 2.5`), overlaid on CIGALE
# at matched parameters.
#
# **The stellar-to-FIR core reproduces to a few percent.** Optical agreement
# is reported as a normalization ratio and its 16–84 % spread. With the
# single-screen dust mapping (`tau_bc = 0`) the residual sits inside ±25 %
# from the far-UV through the FIR; the sub-912 Å excursion is the
# Lyman-continuum extrapolation and the mm-tail offset is the Dale template
# cutoff (§6). The X-ray wing here is XRB + hot gas with no AGN corona —
# `alpha_ox` is supplied but there is no disc for it to act on — and the
# Lehmer+2016 LMXB term is scaled by the SSP mass-weighted age of this
# galaxy, not by a default age. The radio wings rest on `q_IR = 2.5`, pinned
# on both sides (§11). Every one of these is printed below.

# %%
import chex

# The full X-ray -> radio party SED: the §6/§7 galaxy (stellar + dust + Dale
# IR) now with the §10 X-ray (Yang+2020 XRB + hot gas — no AGN corona in this
# galaxy-only chain) and §11 radio (Condon 1992 SF synchrotron, q_IR = 2.5)
# bolted on, so the master grid spans ~0.01 Å (hard X-ray) to ~1 m (radio).
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
    dust_emission={"type": "dale2014_cigale", "alpha_dale": Fixed(2.0), "all_params": Fixed(DEFAULT)},
    xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
    radio={
        "sf": {"type": "bell2003"},
        "agn": {"type": "powerlaw"},
        "radio_q_ir": Fixed(2.5),
        "radio_alpha_sf": Fixed(0.8),
        "all_params": Fixed(DEFAULT),
    },
    neb={"type": "ssp"}, redshift=Fixed(0.0),
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
# single-screen dust mapping the ratio sits at ~1 with a few-percent spread.
# Three things sit outside that window, all of them already accounted for:
# the far-UV, which is §3's age-binning convention; the sub-912 Å excursion,
# which is the Lyman-continuum extrapolation; and the mm tail, where the Dale
# template stops. Nothing here is emission lines — this chain carries no
# nebular block.
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
    "  radio  1.4 GHz = {:.2f}×, 150 MHz = {:.2f}×  (SF synchrotron, q_IR = 2.5)".format(
        _ratio_at(_C_AA_HZ / 1.4e9), _ratio_at(_C_AA_HZ / 0.15e9)
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
# Section by section, with the number each one prints.
#
# * **§1 SSP.** The shared BC03 grid round-trips at the float32 level
#   (median 2e-8), and the write and read sides use one speed of light.
# * **§2 SFH.** The `delayed` shape agrees to a flat 1.00044×, `sfh2exp` to a
#   median 1.00043× with a single grid point at the burst step, where tengri's
#   log lookback grid straddles a discontinuity CIGALE's uniform 1-Myr grid
#   brackets. Both mass integrals hit 1.0000 M☉.
# * **§3 stellar SED.** One convention differs, and this is where it is
#   stated: tengri's cloud-in-cell age kernel captures the `[0, 1 Myr]` star
#   formation that CIGALE's native-age binning drops. It is +14 % at
#   200–912 Å, +1.6 % on L_bol, +15 % on Q_H and +0.1 % in the optical —
#   documented, ratcheted by `tests/crossval/test_dsps_csp_uv_dense_reference.py`,
#   and the reason §6, §8 and §11 have the offsets they do.
# * **§4 attenuation laws.** Analytic curve against analytic curve, the three
#   families agree to 0.000 % away from one window: CIGALE hands over from
#   Leitherer to Calzetti at 1500 Å and tengri at 1800 Å, worth 0.25 %
#   between them. tengri follows Leitherer et al.'s stated 912–1800 Å range;
#   pcigale's own docstring says the same and its code does not.
# * **§5–§7 attenuation applied, dust IR, panchromatic.** Energy balance is
#   exact (`|L_IR − L_absorbed| / L_absorbed = 0`). The energy *anchor* runs
#   2.1 % above CIGALE's `dust.luminosity`, which is §3 and not the dust: the
#   attenuation curves agree to 0.000 % and the absorbed fractions to under
#   half a percent. `lyman_cutoff` matches CIGALE's 912 Å clip on the
#   emergent far-UV; it does not touch the IR budget, which masks the Lyman
#   continuum unconditionally on both sides.
# * **§8 nebular.** Different emitters by design — Cue (a Cloudy 17 emulator)
#   against CIGALE's Cloudy 13.x grids — on a deliberately different, denser
#   SSP, since CIGALE's own ~20 Å optical grid cannot resolve a line. The
#   ionizing budget is printed with the lines, because the line ratios bound
#   the emitter difference rather than measuring it. A Cloudy-against-Cloudy
#   comparison at matched Q_H would close it and is not run here.
# * **§9 AGN.** Disc, torus and polar dust are compared as three separate
#   integrated luminosities rather than as band medians of their sum, so a FIR
#   residual can be attributed to a component instead of to a wavelength — at
#   100 µm all three overlap. Under the `cigale_joint` normalization the torus
#   lands 5.2 % (§9) and 7.5 % (§9b) above CIGALE's and the polar graybody
#   11.3 % and 17.6 % below it, while the disc is 2.8 % high with the
#   Schartmann shape and 7.2 % low with the SKIRTOR one. The emergent disc
#   *shape* differs by 4.70 % and 5.95 %, the analytic disc and the two codes'
#   polar SMC screens together. The polar component is disc-shaped by
#   construction — each code builds it from its own disc seen through that
#   screen — so neither the screen's contribution nor the polar share is
#   common to the two pairings: pcigale's own share of its AGN dust budget
#   moves 0.2098 → 0.2307 between its two disc types, and tengri's 0.1830 →
#   0.1868 with it.
# * **§10 X-ray.** Matched to 4 decimal places on disc L_2500, then a
#   fraction of a percent at 2 keV, and the Yang+2022 inclination tilt is the
#   same function on both sides across i = 0–80°.
# * **§11 radio.** The synchrotron terms agree up to a flat factor that two
#   conventions predict without being fitted to it: the 21 cm-vs-1.4 GHz
#   anchor (×0.985) and §6's energy anchor. Everything above that line is
#   tengri's Murphy+2011 free-free, which CIGALE's radio module does not
#   have.
# * **§12 IGM.** Meiksin 2006 on both sides, max |ΔT| ~ 1e-7 at z = 3, 5, 7,
#   median |ΔT| between 1e-17 and 1e-23, and no point anywhere above 1e-3 —
#   the same prescription evaluated twice.

# %% [markdown]
# ## References
#
# * Boquien et al. 2019, A&A 622, A103 — CIGALE
# * Bruzual & Charlot 2003, MNRAS 344, 1000 — BC03 SSPs
# * Calzetti et al. 2000, ApJ 533, 682 — starburst attenuation law
# * Charlot & Fall 2000, ApJ 539, 718 — two-component dust
# * Condon 1992, ARA&A 30, 575 — radio synchrotron / IR–radio correlation
# * Dale et al. 2014, ApJ 784, 83 — IR dust templates
# * Fritz et al. 2006, MNRAS 366, 767 — AGN torus
# * Inoue et al. 2014, MNRAS 442, 1805 — IGM transmission
# * Li et al. 2025, ApJ, 986, 9 (Cue, arXiv:2405.04598) — neural CLOUDY emulator
# * Madau 1995, ApJ 441, 18 — original IGM transmission
# * Meiksin 2006, MNRAS 365, 807 — updated IGM transmission
# * Noll et al. 2009, A&A 507, 1793 — modified Calzetti
# * Silva et al. 2004, MNRAS 355, 973 — AGN torus
# * Stalevski et al. 2016, MNRAS 458, 2288 — SKIRTOR
# * Yang et al. 2020, MNRAS 491, 740 — X-ray CIGALE module
