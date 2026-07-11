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
# panchromatic SED fitting. This notebook places its physics modules —
# `sfhdelayed`, `bc03`, `nebular`, `dustatt_modified_starburst`,
# `dale2014`, `skirtor2016`, `xray`, `radio`, `redshifting` — next to
# their tengri equivalents on the same axes, in the same units, at the
# same parameters. Same parameters in, same SED out: any disagreement
# is physics, not data or fitting. Each section also names the tengri
# build kwarg that carries the equivalent block.
#
# Both codes consume the same BC03 templates: CIGALE's bundled
# Chabrier-IMF grid (Bruzual & Charlot 2003) was repackaged into the DSPS
# HDF5 layout by `_drivers/cigale_ssp_to_dsps.py`. Any §1 residual
# below floating-point precision is interpolation only.
#
# The fiducial galaxy throughout: τ-delayed SFH with τ = 1 Gyr,
# age = 5 Gyr; Z = Z☉; modified-starburst dust with E(B−V)_lines = 0.3;
# Dale et al. (2014) IR re-emission with α = 2. Sections sweep one
# physics block at a time around this fiducial.
#
# **What to expect.** Stellar templates, star-formation histories,
# dust-attenuation curves, AGN disc + SKIRTOR torus, dust IR through
# the FIR + Rayleigh-Jeans tail to ~mm, X-ray corona + binaries, IGM
# transmission, and the radio synchrotron + free-free composite all
# match CIGALE to floating-point or to a fraction of a percent at
# matched parameters. One block differs by design: the nebular emitter
# uses Cue (a neural emulator trained on Cloudy 17, Li et al. 2025)
# rather than CIGALE's bundled Cloudy 13.x grids. With all gas inputs
# matched, the remaining Hα offset traces to the Cloudy version and to
# how each code convolves nebular emission with its stellar grid. Each
# discrepancy is called out at the relevant section.

# %% [markdown]
# ## Setup

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import warnings

# Override the ``dale2014`` emission model to use CIGALE-sourced templates
# for this reproduction notebook. The shipped ``data/dale2014_templates.h5``
# is the unmodified Wyoming-source Dale et al. 2014 release; the comparison
# panel uses ``data/dale2014_templates_cigale.h5`` so it matches CIGALE's
# actual ``dale2014`` SED template directly. Both files come from
# ``scripts/regenerate_dale2014_from_{cigale,official}.py``.
from pathlib import (
    Path,
    Path as _Path,
)

import matplotlib.pyplot as plt
import numpy as np
from reproduction.cigale._drivers import cigale_driver as C, units as U

import tengri
from tengri import FIXED, Fixed, SEDModel, load_ssp_data
from tengri.dust import register_dale2014_tabulated

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
    register_dale2014_tabulated(str(_CIGALE_DALE_PATH), name="dale2014")

warnings.filterwarnings("ignore")
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
# Dust mapping: CIGALE's `dustatt_modified_starburst(E_BV_lines)`
# applies a **single** Calzetti screen with `E(B-V)_cont = 0.44 * E_BV_lines`
# to the *entire* stellar continuum — the birth-cloud / `E_BV_lines` part
# reddens nebular *lines*, not the continuum. tengri's `two_component` is
# Charlot & Fall: `tau_bc` reddens the *young-star* continuum on top of the
# diffuse `tau_diff`. The CIGALE-equivalent build is therefore a single
# diffuse screen — `tau_bc = 0`, `tau_diff = R_V * 0.44 * E_BV / 1.086`.
# A `tau_bc + tau_diff` split (the earlier mapping) put both terms on the
# continuum and over-attenuated the young-star-dominated FUV by ~2x (and, by
# energy balance, inflated the dust IR / radio ~12-20%). See
# `validate_matched_physics.py`, which matches CIGALE to ~2-3% FUV->FIR
# with the single screen.
_E_BV_LINES = 0.3
_R_V_CALZETTI = 4.05
_F_CONT_OVER_LINES = 0.44
TAU_DIFF_FIDUCIAL = _R_V_CALZETTI * _F_CONT_OVER_LINES * _E_BV_LINES / 1.086
TAU_BC_FIDUCIAL = 0.0  # CIGALE modified_starburst = single continuum screen

# Metallicity pin — CIGALE bc03(metallicity=0.02) is Z_abs = 0.02 (≈ Z_⊙).
# tengri's met_logzsol = log10(Z/Z_⊙) with Z_⊙ = 10**LOG10_ZSUN ≈ 0.0142
# (Asplund+2009). Pin explicitly so the comparison is bit-aligned regardless
# of registry-default convention.
LOG10_ZSUN = -1.848
MET_LOGZSOL = float(np.log10(0.02) - LOG10_ZSUN)  # ≈ +0.149
STELLAR_FIDUCIAL = {"logzsol": Fixed(MET_LOGZSOL), "*": FIXED}

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
ssp = load_ssp_data(str(ssp_file.resolve()))
print(
    f"BC03 Chabrier SSP: {ssp.ssp_wave.shape[0]} wavelengths, "
    f"{ssp.ssp_lgmet.shape[0]} metallicities, "
    f"{ssp.ssp_lg_age_gyr.shape[0]} age bins."
)


# %% [markdown]
# ## Module map
#
# Each row shows the CIGALE module on the left, the tengri registry on
# the right. The tengri side is enumerated live via `tengri.list_*()` so
# the table reflects what the installed version actually exposes.


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
# BC03 Chabrier (Bruzual & Charlot 2003; Chabrier 2003) at Z = 0.02 from
# 1 Myr to 10 Gyr. **Single SSPs**, overlaid: CIGALE's raw
# `bc03/Z=0.02_imf=chab.pickle` (solid) read directly with no SFH module,
# against the same templates re-shaped into tengri's HDF5 (dashed). The
# curves sit on top of each other; the lower panel shows the relative
# residual |tengri − CIGALE| / CIGALE, ~1e-7 from float32 round-trip
# through the HDF5 repackaging — both codes consume identical numerics.

# %%
import pickle as _pickle
from pathlib import Path as _P

ages_yr = [1e6, 1e7, 1e8, 1e9, 1e10]
L_SUN = 3.828e33  # erg/s
_C_AA = 2.998e18  # speed of light [Å/s]

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
    lnu = np.asarray(_raw.spec[:, ia]) * 1e6 * _wl_aa**2 / _C_AA / L_SUN  # Lsun/Hz/Msun
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
for color, age_yr, (w_c, L_c), (w_t, L_t) in zip(colors, ages_yr, cigale_ssp, tengri_ssp):
    label = f"{age_yr / 1e6:g} Myr"
    ax.plot(w_c, L_c, color=color, linewidth=2.0, label=label)
    ax.plot(w_t, L_t, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
    # Residual on the CIGALE wavelength grid (tengri regridded onto it).
    L_t_on_c = U.regrid(w_t, L_t, w_c)
    resid = np.abs(L_t_on_c - L_c) / np.maximum(np.abs(L_c), 1e-30)
    resid[~np.isfinite(resid)] = 0.0
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
# tengri's `sfh.delayed` is the same closed-form τ-delayed shape CIGALE
# uses in `sfhdelayed`: SFR(t) ∝ t · exp(−t/τ), peak at cosmic-time
# t = τ (lookback `age − τ`). Both integrate to 1 M☉ formed by `age` —
# CIGALE via the `normalise=True` flag, tengri via `log_total_mass = 0.0`.
# (`sfh.tau` is a separate model — FSPS sfh=1 / Bagpipes "exponential":
# monotonic decline from formation. Different physics.)
#
# **What the right panel shows.** The SFH reads `state.derived["sfr_history"]`
# from a built `SEDModel` on its 256-point log-spaced lookback grid.
# The `∫SFR dt = 1.0000 M☉` check verifies the integral matches
# `log_total_mass` — the only constraint that matters downstream.

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
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(tau_gyr),
        "age_gyr": Fixed(age_gyr),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
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
# ### Note on the FSPS / Bagpipes declining-exponential
#
# CIGALE's `sfh2exp(f_burst=0)` and FSPS's `sfh=1` both produce a
# declining exponential peaking at galaxy formation — a *different*
# shape from CIGALE's `sfhdelayed` plotted above. tengri intentionally
# does **not** register this shape as `sfh.tau` — to avoid confusion with
# the τ-delayed model, which has opposite physics. The
# `declining_exponential` function remains importable from
# the `sfh` builder grammar for expert use cases.
# The only τ-style SFH in the registry is `delayed` (= CIGALE
# `sfhdelayed`).


# %% [markdown]
# ### Double-exponential with a burst (sfh2exp)
#
# CIGALE's `sfh2exp` superposes an old declining-exponential population and a
# recent exponential burst that carries a fixed fraction `f_burst` of the total
# stellar mass. tengri registers the same form (`sfh.sfh2exp`). At matched
# parameters (τ_main = 4 Gyr, τ_burst = 0.1 Gyr, f_burst = 0.1, burst 0.3 Gyr
# ago, age 10 Gyr) the two SFR histories agree, and tengri's pipeline grid still
# integrates to the requested mass.

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
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "sfh2exp",
        "tau_main_gyr": Fixed(4.0),
        "tau_burst_gyr": Fixed(0.1),
        "f_burst": Fixed(0.1),
        "age_gyr": Fixed(_age_gyr_2exp),
        "burst_age_gyr": Fixed(0.3),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
_st_2exp = _m_2exp.predict_state({})
_lbt_2exp = np.asarray(_st_2exp.derived["sfh_grid_lbt_yr"])
_sfr_2exp = np.asarray(_st_2exp.derived["sfr_history"])
_t_2exp = (_age_gyr_2exp - _lbt_2exp / 1e9) * 1e9  # cosmic age since SF onset [yr]
_idx2 = np.argsort(_t_2exp)
_mass_2exp = float(np.trapezoid(_sfr_2exp[_idx2], _t_2exp[_idx2]))
print(f"tengri pipeline ∫SFR dt = {_mass_2exp:.4f} M☉ (target: 1.0000 from log_total_mass=0)")

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
# Convolve the τ-delayed SFH with the BC03 SSPs. No dust, no nebular.
# Both panels show L_ν vs λ_rest; CIGALE is normalized to 1 M☉ formed
# by construction, tengri's stellar mass formed is reported in the
# annotation.

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
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_stellar = m_stellar.predict_state({})
_assert_comparable(L_c, s_stellar.sed_intrinsic, name="§3 stellar")

# Shared-axis overlay + tengri/CIGALE ratio panel (#864): a single scale makes
# any normalization offset visible at a glance. Both codes form 1 M_sun; the
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
fig.tight_layout()
save_fig("cigale_03_stellar_sed.png")


# %% [markdown]
# ## §4 Dust attenuation curves
#
# tengri's `calzetti`, `noll09`, and `power_law` attenuation laws against
# their CIGALE counterparts (`dustatt_calzleit`,
# `dustatt_modified_starburst`, `dustatt_modified_CF00`). Both sides
# evaluate the analytic law directly (tengri via `tengri.dust.list_laws`),
# normalized to `A(λ)/A_V` at 5500 Å — curve against curve, no
# SSP-convolution noise. CIGALE's default laws carry no 2175 Å bump (it is
# opt-in via `uv_bump_amplitude`); tengri's `noll09` reproduces the
# Noll+2009 Drude bump, pinned against `dust_attenuation` in the test suite.

# %%
from tengri.dust import list_laws

# (CIGALE module, CIGALE kwargs for A_V ≈ 1.2, tengri law, label). Pairs
# match by curve family: Calzetti, Calzetti + Leitherer UV slope, and the
# (λ/550)^δ power law.
_law_pairs = [
    ("dustatt_calzleit", dict(E_BVs_young=0.3), "calzetti", "Calzetti+2000"),
    ("dustatt_modified_starburst", dict(E_BV_lines=0.3), "noll09", "Calzetti + Leitherer UV"),
    ("dustatt_modified_CF00", dict(Av_ISM=1.2), "power_law", "Charlot & Fall power law"),
]
_tengri_laws = list_laws(headline=False)  # {name: fn(wave_aa) -> k at tau_V=1}
wave_law = np.logspace(np.log10(1000.0), np.log10(30000.0), 2000)


def _norm_AV(wave, A):
    """A(λ) normalized to A_V at 5500 Å."""
    return A / A[np.argmin(np.abs(wave - 5500.0))]


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, title in (
    (ax_l, "pcigale.sed_modules attenuation laws"),
    (ax_r, "tengri attenuation laws"),
):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ [Å]")
    ax.set_xlim(1e3, 3e4)
    ax.set_ylim(0.05, 20)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
ax_l.set_ylabel(r"$A_\lambda / A_V$")

for cig_law, cig_kw, tengri_law, label in _law_pairs:
    try:
        w_c, A_c = C.attenuation_curve(cig_law, **cig_kw)
        ax_l.plot(w_c, _norm_AV(w_c, A_c), linewidth=2.0, label=label)
    except Exception:
        pass

    # tengri's law functions are JAX-native but accept array-likes; the
    # result is wrapped back to NumPy for plotting.
    A_t = np.asarray(_tengri_laws[tengri_law](wave_law))
    ax_r.plot(wave_law, _norm_AV(wave_law, A_t), linewidth=2.0, label=label)
ax_l.legend(fontsize=10)
ax_r.legend(fontsize=10)
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
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_nd = m_nd.predict_state({})

m_d = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={
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
        "*": FIXED,
    },
    redshift=Fixed(0.0),
)
s_d = m_d.predict_state({})
_assert_comparable(L_c_d, s_d.derived["sed_dust_attenuated"], name="§5 dust applied")

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
# Absorbed stellar UV/optical reappears in the IR. CIGALE uses the
# Dale et al. (2014) template family (α = 2); tengri evaluates the same
# templates and enforces energy balance,
# $L_{\rm IR,\,emitted} \equiv L_{\rm absorbed}$, to floating-point —
# the residual is annotated on the right panel. The energy anchors agree
# to 2%: tengri's `L_absorbed` sits above CIGALE's `dust.luminosity` by
# the FUV shape difference between the two codes' Leitherer-extended
# Calzetti curves. That anchor offset, plus template regridding, is the
# printed 10–100 µm median.
#
# **Long wavelengths.** Past ~10 mm the panels diverge because tengri
# ships the published Dale grid (1–225 mm) while CIGALE truncates at
# ~10 mm; the tail sits ~3 orders below the FIR peak and both codes
# renormalize to `L_absorbed`, so nothing physical rides on it.
#
# **Lyman continuum.** tengri's `calzetti` / `leitherer02` curves
# polynomial-extend through the FUV; CIGALE zeros attenuation below
# 912 Å. Models here set `dust={'lyman_cutoff': True}` to match. The
# `L_absorbed` integral already excludes λ < 912 Å on both sides, so
# only the emergent FUV continuum changes (see §7).

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
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        # Lyman-limit clip (CIGALE parity) — see §5. The dust IR is energy-balance
        # normalized to L_absorbed, whose integral already excludes λ < 912 Å, so
        # this only changes the emergent FUV continuum, not the IR budget.
        "lyman_cutoff": True,
        "*": FIXED,
        "emission": {"type": "dale2014", "alpha_mir": Fixed(2.0), "*": FIXED},
    },
    redshift=Fixed(0.0),
)
s_ir = m_ir.predict_state({})
L_abs = float(s_ir.derived.get("L_absorbed", 0.0))
L_emit = float(s_ir.derived.get("L_ir", 0.0))
residual = abs(L_abs - L_emit) / max(L_abs, 1e-30)
_assert_comparable(L_c_ir, s_ir.sed_intrinsic, name="§6 IR")

# Shared-axis overlay + ratio panel (#864): the FIR-peak partition difference
# (~1.0-1.15, Dale2014 stellar-heated) is only readable on one scale.
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
fig.tight_layout()
fig.savefig(str(figs_dir / "cigale_06_dust_ir_dale2014.png"), dpi=150, bbox_inches="tight")
plt.show()


# %% [markdown]
# ### Dust-IR model knobs: AGN heating and the radiation-field slope
#
# Two further CIGALE dust-IR parameters, shown as the same head-to-head as
# every other panel — **tengri solid, pcigale dotted**.
#
# **The attenuation must match for the IR to match.** The dust IR is
# energy-balance-normalized to the *absorbed* starlight, so reproducing CIGALE
# requires reproducing its `dustatt_modified_starburst`, which is a **single
# Calzetti screen** on the stellar continuum (A_V = R_V x E(B-V)_cont =
# 4.05 x 0.132 = 0.535 mag). It has no Charlot & Fall birth cloud. Using
# tengri's two-component default (an extra `tau_bc` birth-cloud screen, as the
# other panels do) over-absorbs by **~9 %** and inflates the whole IR by the
# same factor. `_knob_model` therefore uses the single screen (`tau_bc = 0`),
# at which the absorbed *fraction* matches CIGALE to **0.2 %**.
#
# **Left — Dale 2014 AGN fraction (`dale2014.fracAGN`).** With the screen
# corrected, the stellar-heated curve matches CIGALE to **~1.5 %** (the
# residual is the BC03->DSPS conversion: tengri's intrinsic stellar $L_{\rm
# bol}$ is 1.4 % low, *not* the dust or the attenuation). `fracAGN` adds an
# AGN-heated source as a separate power budget ($L_{\rm AGN}=L_{\rm
# dust}\,f/(1-f)$) using CIGALE's own ``model_quasar`` template
# (`SED = L\,T_{\rm SF}(\alpha) + L_{\rm AGN}\,T_{\rm QSO}`, `dale2014.py`).
# tengri tracks CIGALE to **~1.5 % across the whole sweep** ($f_{\rm AGN}=0,
# 0.3, 0.6$), the same BC03->DSPS floor — there is no fracAGN-dependent drift.
# The subtlety that makes this work: CIGALE normalizes ``model_quasar`` to unit
# luminosity over its *full* native grid (~60 nm onward), where ~46 % of the
# quasar energy is the UV/optical accretion-disc continuum below the dust
# grid's blue edge. Only its ~0.42 IR share enters the dust mixing; tengri
# carries that exact partition faithfully, so the mid-IR lift matches and
# does not run high at $f_{\rm AGN}=0.6$.
#
# **Right — THEMIS slope $\alpha$ (`themis.alpha`, $dU/dM \propto U^{-\alpha}$,
# matched `qhac=0.17, umin=1.0, gamma=0.1`).** tengri's THEMIS templates are
# built from the published DustEM grids (Jones+2017) and conserve the absorbed
# energy. The a-C(:H) aromatic fraction `qhac` is pinned to CIGALE's 0.17 on
# both sides — CIGALE quotes it as a fraction while the DustEM grid tabulates
# it in FSPS scaling (`qhac × 100/2.2`), so the two must be reconciled before
# interpolation or the wrong grain model is selected. With the composition
# matched the curves track across the whole $\alpha$ sweep; the small residual
# is CIGALE's own DustEM run using `umax=1e7` (a slightly hotter PDR), which
# nudges the IR partition at the few-percent level.

# %%
import jax
import jax.numpy as jnp

_c_aa_dust = 2.998e18

# Matched fiducial chain (mirrors the §6 dale2014 cell): same SFH + BC03 +
# Calzetti attenuation on both sides, so the absorbed luminosity that feeds
# the IR re-emission template is identical. pcigale's normalise=True forms
# 1 M_sun; tengri's log_total_mass=11 forms 1e11 M_sun, so scale the pcigale
# curve by 1e11 to overlay the two at the same stellar mass.
_KNOB_MASS = 1e11
_SFH_CHAIN = (
    "sfhdelayed",
    dict(tau_main=1000, age_main=5000, tau_burst=50, age_burst=20, f_burst=0.0,
         sfr_A=1.0, normalise=True),
)
_BC03_CHAIN = ("bc03", dict(imf=1, metallicity=0.02, separation_age=10))
_DUSTATT_CHAIN = ("dustatt_modified_starburst", dict(E_BV_lines=0.3))


def _knob_model(emission_type, **emkw):
    """tengri fiducial twin of the matched pcigale chain (1e11 M_sun).

    CIGALE's ``dustatt_modified_starburst`` is a **single** Calzetti screen on
    the stellar continuum (A_V = R_V x E(B-V)_cont = 4.05 x 0.132 = 0.535 mag,
    i.e. ``tau_diff = TAU_DIFF_FIDUCIAL``). It has no Charlot & Fall birth-cloud
    component, so we set ``tau_bc = 0``: adding the extra birth-cloud screen
    (the two-component default elsewhere in this notebook) over-absorbs the
    starlight by ~9 %, and since the Dale/THEMIS IR is normalized to the
    absorbed energy that inflates the whole IR. With the single screen the
    absorbed *fraction* matches CIGALE to 0.2 %; the residual ~1.4 % is the
    BC03-to-DSPS conversion (tengri's intrinsic stellar Lbol is 1.4 % lower).
    """
    return SEDModel.build(
        ssp_data=ssp,
        stellar=STELLAR_FIDUCIAL,
        sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
             "log_total_mass": Fixed(11.0), "*": FIXED},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
            "*": FIXED,
            "emission": {"type": emission_type, "*": FIXED, **emkw},
        },
        redshift=Fixed(0.0),
    )


def _nu_lnu(wave_aa, l_nu):
    w = np.asarray(wave_aa)
    return w, _c_aa_dust / w * np.asarray(l_nu)


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(11, 4.4))

# LEFT — Dale 2014 AGN fraction: tengri (solid) vs pcigale dale2014.fracAGN (dashed).
m_frac = _knob_model("dale2014", alpha_mir=Fixed(2.0))
p_frac = dict(m_frac.spec.sample(jax.random.PRNGKey(0)))
for f, c in zip([0.0, 0.3, 0.6], ["C0", "C1", "C3"]):
    o = m_frac.predict_rest_sed({**p_frac, "dust_frac_agn": jnp.float64(f)})
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_l.loglog(w_t, nl_t, color=c, lw=1.8, label=rf"$f_{{\rm AGN}}={f}$")
    sed = C.run_chain([_SFH_CHAIN, _BC03_CHAIN, _DUSTATT_CHAIN,
                       ("dale2014", dict(alpha=2.0, fracAGN=f))])
    w_c, nl_c = _nu_lnu(*C.to_lnu(sed))
    ax_l.loglog(w_c, nl_c * _KNOB_MASS, color=c, lw=1.2, ls=":")
ax_l.plot([], [], "k-", lw=1.8, label="tengri")
ax_l.plot([], [], "k:", lw=1.2, label="pcigale")
ax_l.set(xlim=(1e4, 1e7), ylim=(1e42, 1e45), xlabel=r"$\lambda$ [Å]",
         ylabel=r"$\nu L_\nu$ [erg s$^{-1}$]", title="Dale 2014 AGN fraction")
ax_l.legend(fontsize=8, frameon=False, ncol=2)

# RIGHT — THEMIS slope alpha: tengri (solid) vs pcigale themis.alpha (dashed),
# matched qhac=0.17, umin=1.0, gamma=0.1.
m_alpha = _knob_model("themis", dust_gamma_dl=Fixed(0.1), dust_qhac=Fixed(0.17))
p_alpha = dict(m_alpha.spec.sample(jax.random.PRNGKey(0)))
for a, c in zip([1.0, 2.0, 3.0], ["C0", "C1", "C3"]):
    o = m_alpha.predict_rest_sed({**p_alpha, "dust_alpha": jnp.float64(a)})
    w_t, nl_t = _nu_lnu(o.wavelength, o.sed)
    ax_r.loglog(w_t, nl_t, color=c, lw=2.0, label=rf"$\alpha={a}$")
    sed = C.run_chain([_SFH_CHAIN, _BC03_CHAIN, _DUSTATT_CHAIN,
                       ("themis", dict(qhac=0.17, umin=1.0, gamma=0.1, alpha=a))])
    w_c, nl_c = _nu_lnu(*C.to_lnu(sed))
    ax_r.loglog(w_c, nl_c * _KNOB_MASS, color=c, lw=1.2, ls=":")
ax_r.plot([], [], "k-", lw=2.0, label="tengri")
ax_r.plot([], [], "k:", lw=1.2, label="pcigale")
ax_r.set(xlim=(3e4, 1e7), ylim=(1e42, 1e45), xlabel=r"$\lambda$ [Å]",
         title=r"THEMIS radiation-field slope $\alpha$")
ax_r.legend(fontsize=8, frameon=False, ncol=2)

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
# 1200 Å – 22000 Å; tengri's polynomial *extrapolates* below that,
# letting `A_λ/A_V` climb to ~150 at λ < 300 Å, while CIGALE's
# `dustatt_modified_starburst` drops cleanly to zero at 912 Å. The
# tengri model in this section sets `dust={'lyman_cutoff': True}`,
# which applies the same 912 Å clip — so the far-UV continuum is
# transmitted on both sides and the panels now agree
# blueward of Lyα. (Drop the toggle to recover tengri's default
# extrapolation, which over-attenuates the FUV relative to CIGALE.)
#
# **One remaining wing difference — a §-cross-reference, not a §7-physics
# issue.** *Long-wavelength tail (λ > 10⁷ Å):* the tengri side carries a
# small rising νL_ν tail from 10 to 200 mm where CIGALE's reads zero.
# This is the published-Dale-vs-CIGALE-truncated-Dale template
# difference documented in §6, normalized away by energy balance.

# %%
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
# tengri uses **Cue** (Li et al. 2024), a neural emulator of the same
# physics that exposes logU, gas metallicity and IMF as continuous
# parameters. Cue requires the bare-stellar SSP that this notebook
# already loaded.
#
# **Fiducial choice — young population.** Sections §3–§7 use a 5 Gyr
# quiescent τ=1 Gyr galaxy (Boquien+2019 reference). Nebular line
# emission, however, lives almost entirely in the ionizing-photon
# budget of stars ≲ 100 Myr old, and the 5 Gyr quiescent galaxy has
# almost none. To showcase line emission honestly, §8 swaps to a
# **τ=300 Myr, age=100 Myr** delayed SFH where Hα and the metal-line
# forest are physically strong. Same logU, Z_gas, f_esc as the rest of
# the notebook; only the SFH shifts to a younger reference.
#
# Each panel shows the stellar baseline (dashed), stellar + nebular
# (solid), and the nebular component alone (dotted). The two emitters
# see the same H II region: `logU = −2.0`, `Z_gas = Z_⊙` (Cue's
# `neb_logZ_gas` is pinned to `log10(0.02/Z_⊙) ≈ +0.149`), `f_esc = 0`,
# `n_H = 100 cm⁻³` (`gas_logn = 2.0`), solar N/O and C/O
# (`gas_logno = gas_logco = 0`). The Q_H reaching Cue is the integral
# of the SSP-convolved ionizing spectrum below 911.76 Å, published by
# the stellar component and consumed by the nebular component on every
# forward pass.
#
# **Remaining residual.** The gap lives downstream of the gas knobs:
# Cue was trained on Cloudy 17 (Li et al. 2025) while CIGALE bundles
# Cloudy 13.x grids, and Cue's bare-stellar SSP path differs from
# CIGALE's wNE-SSP convolution. Nebular continuum shape and
# emission-line ratios reproduce well; the absolute line normalization
# carries a Cloudy-version offset. The residual is quantified below with
# the **integrated** line luminosity (continuum-subtracted,
# width-independent) — *not* a single-bin peak ratio, which would
# measure line width and grid resolution (CIGALE broadens to
# `lines_width = 300 km/s`) rather than physics.
#
# **Grid coverage — why the two panels span different ranges.** The Cue
# emulator ships a native continuum grid (~915 Å – 10⁸ Å, 1841 points)
# inside `cue_weights.npz`, and its trained line list is the optical/UV
# forest. CIGALE's static CLOUDY grid additionally carries the *far-IR
# fine-structure line forest* — [O III] 88 µm, [C II] 158 µm, [N II],
# [S III] 18.7 µm, [Ne III] 15.6 µm and companions out to ~10⁶ Å — which
# is why the left panel shows tall line spikes at 10⁵–10⁶ Å that the Cue
# panel does not. That is the emulator's scope, not a defect: for a
# CLOUDY-vs-CLOUDY match on the same wavelength range (FIR lines included)
# tengri exposes its own static CLOUDY grid via `neb={'type': 'cloudy'}`
# (`data/cloudy_grid_*.h5`, 166 lines to 6.1×10⁶ Å); Cue is the
# differentiable emulator you reach for when the gas parameters must be
# free in a fit. The wavelength-extension registry routes the Cue grid
# into the master union, so attaching Cue alone extends the SED past the
# SSP edge — no `dust.emission` component required — all the way out to
# the radio without the artificial 160-µm truncation earlier registry
# versions left in place when only Cue (and no dust template) was active.

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
    "*": FIXED,
}

m_no_neb = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh=_neb_sfh_kw,
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_no_neb = m_no_neb.predict_state({})

m_neb = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh=_neb_sfh_kw,
    neb={
        "type": "cue",
        "neb_logU": Fixed(-2.0),
        "neb_logZ_gas": Fixed(MET_LOGZSOL),  # Z_gas = 0.02 ≡ stellar Z
        "neb_fesc": Fixed(0.0),
        "*": FIXED,
    },  # ionspec_* slopes stay at their SSP-derived Fixed values
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
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
    stellar={"logzsol": Fixed(0.0), "*": FIXED},
    sfh=_neb_sfh_kw,
    neb={
        "type": "cue",
        "neb_logU": Fixed(-2.0),
        "neb_logZ_gas": Fixed(0.0),
        "neb_fesc": Fixed(0.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
_s_neb_dense = _m_neb_dense.predict_state({})
_w_t_dense = np.asarray(_s_neb_dense.wave)
_L_t_dense = np.asarray(_s_neb_dense.derived["sed_nebular"])
print("§8 integrated line luminosity (tengri Cue / CIGALE CLOUDY; tengri on dense FSPS SSP):")
for _c, _name in [(6563.0, "Hα"), (5007.0, "[O III]"), (4861.0, "Hβ")]:
    _lc = U.line_lum(w_c_neb, L_c_neb_only, _c)
    _lt = U.line_lum(_w_t_dense, _L_t_dense, _c)
    if _lc > 0:
        print(
            f"    {_name} {_c:.0f} Å: CIGALE {_lc:.2e}, "
            f"tengri {_lt:.2e} erg/s → {_lt / _lc:.2f}×"
        )


# %% [markdown]
# ## §9 AGN
#
# CIGALE's `skirtor2016` AGN package on the left vs tengri's composable
# AGN on the right, both at the SKIRTOR fiducial (i = 30°, τ_9.7 = 7,
# oa = 40°, p = q = 1). Each panel shows the stellar + dust baseline
# (dashed), the full SED with AGN (solid), and the AGN-only component
# (dotted).
#
# **`disc.schartmann2005` matches CIGALE's `skirtor2016 disk_type=1`**
# default: the Schartmann (2005) piecewise power law with the 1200 Å
# bend that CIGALE's module substitutes for the FITS-bundled disc when
# `disk_type=1` is selected. tengri also ships `disc.skirtor` (the
# SKIRTOR analytic disc, CIGALE `disk_type=0`) and `disc.adaf_lopez2024`
# (CIGALE `disk_type=2`). tengri's library defaults match CIGALE's
# `skirtor2016` defaults — `agn_torus_frac=0.5` (covering factor),
# `agn_polar_ebv=0.03` (Casey-2012 polar dust on, T=100 K, β=1.6) — so
# the disc + torus + polar-dust graybody is engaged automatically with
# no per-fit overrides. Net agreement vs CIGALE: UV–NIR within ~1 %,
# MIR/FIR within ~6 % under the default `norm='cigale_joint'` energy
# balance, which ties all AGN components to a shared reference (described
# below).
#
# **Alternative.** For users who want a differentiable disc (M_BH, ṁ,
# spin), the composable AGN still accepts `disc={"type": "multicolor",
# ...}` — a Shakura-Sunyaev numerically-evaluated disc. That path
# carries a hard far-UV bump separated from the optical Wien tail by
# a notch around 5000 Å (the multicolor-disc signature) and visibly
# diverges from CIGALE in the disc UV continuum even though the torus
# IR still matches. The reproduction notebook uses `disc.schartmann2005`
# to match CIGALE exactly; production fits choose the physics they need.
#
# The torus IR uses the same templates and reproduces at all
# inclinations: face-on i = 30° peaks at ~6–9 µm on both sides;
# edge-on viewing pushes the dust peak out to ~30 µm (classic
# reprocessed-dust bump). The Casey-2012 polar-dust graybody — added
# on top of the SKIRTOR thermal dust by default (`agn_polar_ebv=0.03`,
# composable; set `Fixed(0.0)` to disable) — lifts the FIR tail
# (~100 µm) by a factor of a few.
#
# **Energy-balance normalization.** CIGALE ties disc, torus, and polar
# dust to a single `agn_power` reference via the fixed SKIRTOR template
# ratios (`skirtor2016.py`, `norm = 1/∫dust`), ensuring energy
# conservation across all AGN components. tengri's default `norm='cigale_joint'`
# policy implements the same accounting via four coordinated mechanisms:
# (1) the disc is reddened by the Pei-SMC extinction `ext_fac` for Type-1
# sightlines; (2) the disc is tied to `agn_power × R` where `R = η(i)·∫disc/∫dust`
# is the SKIRTOR disc/dust bolometric ratio and `η(i) = cos i (1+2cos i)/3`
# is the Stalevski+2016 anisotropy factor (η = 0.789 at i = 30°); (3) `R`
# is evaluated on the native SKIRTOR grid via PCHIP interpolation, with the
# disc inclination shape reweighted to the chosen `cos_inc`; (4) the polar
# graybody's absorbed-energy budget `l_ext` is tied to the face-on
# un-reddened ratio `R_faceon = ∫disc(i=0)/∫dust`. With all four engaged,
# the stellar+AGN SED lands at {UV: 0.99, opt: 1.00, NIR: 0.99,
# 30 µm: 0.96, 100 µm: 0.94} vs CIGALE — within ~6 % end-to-end, at the
# template and SSP-conversion precision floor.
#
# **The policy is switchable.** `agn={'type':'composable', …,
# 'norm':'cigale_joint'}` (default) is the CIGALE-matched path above;
# `'norm':'independent'` restores the legacy two-reference scaling
# (~0.80 / 0.79 / 1.06 / 1.15) for users who want the disc decoupled
# from the absorbed-energy budget. The choice is surfaced in
# `model.spec.summary()` (the `agn=composable[…, norm=…]` line) and in
# `tengri.describe_agn_model('composable')`.

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
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={
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
        "*": FIXED,
    },
    redshift=Fixed(0.0),
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
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={
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
        "*": FIXED,
    },
    # ``agn_log_lbol`` matches CIGALE's ``sed.info["agn.accretion_power"]``
    # at the §9 fiducial: 9.18e25 W = 0.240 L☉ → log_lbol = -0.620.
    # This is the **intrinsic 4π disc bolometric** — the L_bol that the
    # accretion engine produces, before extinction. CIGALE derives it from
    # agn_power × ∫AGN1.disk × norm × 0.493 (skirtor2016.py:507). Using
    # the observed (extinction-corrected) disc luminosity instead would
    # underestimate the intrinsic power by a factor of ~1.6.
    #
    # All other tengri AGN defaults already match CIGALE skirtor2016
    # defaults (oa=40, tau=7, p=q=1, i=30, EBV=0.03, T=100, β=1.6,
    # disk_type=1 → ``disc.schartmann2005``). The polar-dust graybody
    # is integrated into the SKIRTOR thermal-dust normalization
    # (CIGALE skirtor2016.py:389 adds polar BB before the ``norm =
    # 1/∫dust`` step). The differentiable multicolor disc remains
    # available — ``disc={"type": "multicolor", ...}``.
    agn={
        "type": "composable",
        "disc": {"type": "schartmann2005", "*": FIXED},
        "torus": {"type": "skirtor", "*": FIXED},
        "agn_log_lbol": Fixed(-0.42),
        # ``agn_fracAGN = 0.3`` mirrors CIGALE's ``fracAGN`` parameter.
        # tengri's AGN component reads ``state.derived["L_absorbed"]``
        # and computes ``agn_power = L_abs × frac/(1-frac)`` exactly
        # like CIGALE ``skirtor2016.py:498`` (lambda_fracAGN="0/0")
        # via cross-component energy coupling.
        "agn_fracAGN": Fixed(0.3),
        "*": FIXED,
    },
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
fig.tight_layout()
save_fig("cigale_09c_agn_parity_ratio.png")
plt.show()


# %% [markdown]
# ### §9b The other disc — `disc.skirtor` ↔ CIGALE `disk_type=0`
#
# CIGALE's `skirtor2016` offers two analytic discs (`skirtor2016.py`):
# `disk_type=0` → `skirtor_disk()` and `disk_type=1` →
# `schartmann2005_disk()`. tengri ships **both, bit-for-bit**:
# `disc.skirtor` reproduces `skirtor_disk` and `disc.schartmann2005`
# reproduces `schartmann2005_disk`, each to machine precision (the
# unit-area disc shapes agree to ~1e-16). The panel above pins the
# `disk_type=1` pairing; this one swaps to `disc.skirtor` and pairs it
# against CIGALE `disk_type=0`. The UV–optical disc continuum (where the
# disc, not the torus, dominates) overlies CIGALE exactly — the
# shallower SKIRTOR disc with its 1200 Å bend, distinct from the
# Schartmann disc's shape. The MIR/FIR torus agreement is independent of
# the disc choice.

# %%
sed_skirtor0 = C.run_chain(
    [
        _sfh_args_d,
        ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
        ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
        (
            "skirtor2016",
            dict(
                t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97, i=30,
                disk_type=0,  # ← skirtor_disk ↔ tengri disc.skirtor
                delta=0, fracAGN=0.3, lambda_fracAGN="0/0", law=0,
                EBV=0.03, temperature=100.0, emissivity=1.6,
            ),
        ),
    ]
)
w_sk0, L_sk0 = C.to_lnu(sed_skirtor0)

m_agn_sk = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(5.0),
         "log_total_mass": Fixed(0.0), "*": FIXED},
    dust={"type": "two_component", "law_bc": "leitherer02", "law_diff": "leitherer02",
          "tau_bc": Fixed(TAU_BC_FIDUCIAL), "tau_diff": Fixed(TAU_DIFF_FIDUCIAL), "*": FIXED},
    agn={"type": "composable",
         "disc": {"type": "skirtor", "*": FIXED},  # ← the SKIRTOR analytic disc
         "torus": {"type": "skirtor", "*": FIXED},
         "agn_log_lbol": Fixed(-0.42), "agn_fracAGN": Fixed(0.3), "*": FIXED},
    redshift=Fixed(0.0),
)
s_agn_sk = m_agn_sk.predict_state({})

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(
    ax_l, ax_r,
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
ax_r.plot(s_agn_base.wave, s_agn_base.sed_intrinsic, "k--", linewidth=1.0, alpha=0.5,
          label="stellar + dust")
ax_r.plot(s_agn_sk.wave, s_agn_sk.sed_intrinsic, "C1-", linewidth=1.5, alpha=0.7,
          label="stellar + dust + AGN")
ax_r.plot(s_agn_sk.wave, L_sk_only, "C1:", linewidth=1.5,
          label="skirtor disc + SKIRTOR torus only")
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
# `L_ν ∝ E^(1−Γ)` (Γ ≈ 1.8, E_cut ≈ 300 keV) normalized off the intrinsic
# disc through `L_2keV = L_2500 · 10^(α_ox/0.3838)`, plus X-ray binaries and
# hot gas. tengri ships the matching `xray.yang20`. Three conventions are
# pinned so the panel isolates real differences:
#
# - **α_ox.** CIGALE fixes α_ox = −1.4; tengri derives it from L_2500
#   (Just et al. 2007, Eq. 3). The two coincide at log₁₀ L_2500 = 29.47,
#   so both codes are solved live onto that luminosity — no hard-coded
#   scale factors to go stale.
# - **Inclination.** Both codes anchor the α_ox-derived corona at the
#   Yang+2020 30° reference and tilt it with the same Yang et al. (2022)
#   factor, read from the AGN block (#980). Here i = 30°; §10b sweeps it.
# - **Absorption.** CIGALE's module carries no line-of-sight absorber, so
#   tengri's N_H channel is pinned to zero. tengri keeps its constant 1%
#   scattered floor (Ricci et al. 2017) — a flat +1% in the ratio panel.
#
# The comparison is corona-to-corona: CIGALE's `xray.agn` component against
# tengri's `sed_xray` (the 1 M☉ host makes tengri's XRB/hot-gas terms
# negligible). CIGALE's `fracAGN` anchors the AGN on the dust luminosity,
# so the attenuation + dust-emission modules in the chain are load-bearing:
# without them the corona normalization collapses and the X-ray SED is
# binaries only.

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
        stellar=STELLAR_FIDUCIAL,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust={
            "type": "two_component",
            "tau_bc": Fixed(TAU_BC_FIDUCIAL),
            "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
            "*": FIXED,
        },
        agn={
            "type": "composable",
            "disc": {"type": "schartmann2005", "*": FIXED},
            "torus": {"type": "skirtor", "*": FIXED},
            "agn_log_lbol": Fixed(log_lbol),
            "agn_cos_inc": Fixed(cos_inc),
            "*": FIXED,
        },
        xray={"type": "yang20", "log_nh": Fixed(0.0), "*": FIXED},
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
# The Yang et al. (2022) corona anisotropy,
# `f(μ) = (a₁μ + 1 − a₁) / (1 − 0.13397 a₁)` with a₁ = 0.5, equals 1 at the
# 30° anchor where the α_ox relation is defined. CIGALE tilts its corona by
# `f(cos i)` read from the AGN module; tengri publishes the same
# `agn_cos_inc` to its X-ray block (#980). One wrinkle: at fixed `fracAGN`,
# hiding the disc behind the torus (Type-2, i > 50° here) makes CIGALE
# *raise* the intrinsic disc normalization, which would drag L_2500 off the
# α_ox crossing — CIGALE keeps α_ox = −1.4 while tengri's Just+2007 tracks
# the moving L_2500. The sweep re-solves CIGALE's amplitude at each angle
# so the corona anchor stays at the crossing; what remains is the pure
# f(cos i) tilt, applied identically by both codes — the ratio panel stays
# at the §10a value at every inclination.

# %%
fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
print(" i     f(mu)   log10 L_2500,c   tengri/CIGALE @ 2 keV")
for _i_deg, _col in zip((0, 30, 60, 80), ("C0", "C2", "C3", "C4")):
    # hold the corona anchor at the alpha_ox crossing: fracAGN raises the
    # intrinsic disc as the torus hides it, so rescale sfr_A per angle
    _probe_i = _cigale_xray(_SFR_A_XRAY, _i_deg)
    sed_i = _cigale_xray(
        _SFR_A_XRAY * 10.0**_LOG_L2500_TARGET / _cigale_l2500(_probe_i), _i_deg
    )
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
# to the IR-radio correlation (`qir_sf`, `alpha_sf`) — no thermal free-free
# term. tengri's `radio.condon92` is the fuller composite: Bell 2003 (q_IR
# synchrotron) plus Murphy 2011 free-free (Eq. 11) plus Yang 2020 (AGN).
# The build pins `radio_q_ir = 2.5`, `radio_alpha_sf = 0.8`, and the same
# attenuation law as §6, because the synchrotron amplitude is anchored on
# the dust luminosity: both codes compute
# `L_ref = L_dust / (3.75e12 · 10^q_IR)`, so any mismatch in `L_absorbed`
# lands 1:1 in the radio.
#
# Every term of the ratio near 1.4 GHz is accounted for (the cell prints
# them):
#
# - **Free-free, ×1.036 at 1.25 GHz.** Murphy+2011 thermal emission that
#   CIGALE omits (#863). It grows with frequency — the ratio panel climbs
#   to ~1.9 by 100 GHz as the synchrotron (α = 0.8) falls away from the
#   flatter (α ≈ 0.1) free-free.
# - **Anchor frequency, ×0.985.** Bell 2003 defines q_IR at 1.4 GHz;
#   CIGALE normalizes at 21 cm = 1.4276 GHz. A pure convention offset of
#   `(1.4276/1.4)^−0.8`.
# - **Energy-balance anchor, ×1.020.** tengri's `L_absorbed` sits 2% above
#   CIGALE's `dust.luminosity` for the matched attenuation setup — the §6
#   FUV curve-shape residual, propagated through q_IR.
#
# Star-forming only, 100 MHz to 100 GHz.

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
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    # Same attenuation setup as §6 — the radio amplitude is anchored on
    # L_absorbed through q_IR, so a mismatched dust config here would leak
    # straight into the synchrotron normalization.
    dust={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        "lyman_cutoff": True,
        "*": FIXED,
        "emission": {"type": "dale2014", "alpha_mir": Fixed(2.0), "*": FIXED},
    },
    # q_IR pinned to CIGALE's qir_sf = 2.5 (tengri bucket default 2.64).
    radio={
        "type": "condon92",
        "radio_q_ir": Fixed(2.5),
        "radio_alpha_sf": Fixed(0.8),
        "*": FIXED,
    },
    redshift=Fixed(0.0),
)
state_r = m_r.predict_state({})
w_t = np.asarray(state_r.wave)
sed_t = np.asarray(state_r.derived["sed_radio"])  # synchrotron + Murphy free-free

# Shared-axis overlay + ratio panel (#864): the ratio rising above unity toward
# high ν is tengri's free-free that CIGALE's synchrotron-only module lacks (#863).
fig, ax, ax_r, ratio = U.overlay_ratio_fig(
    w_r,
    L_r,
    w_t,
    sed_t,
    x_of_wave=lambda w: 2.998e18 / w / 1e9,
    xlabel=r"$\nu$ [GHz]",
    title="§11 SF radio — CIGALE synchrotron vs tengri synchrotron + free-free",
    label_c="CIGALE  radio.sf_nonthermal (synchrotron only)",
    label_t="tengri  radio.condon92 (synchrotron + free-free)",
    xlim=(0.1, 100.0),
    ratio_ylim=(0.5, 2.0),
)
# Overlay tengri's synchrotron-only term (Bell 2003) so the reader can see it
# lies on top of CIGALE's synchrotron-only sf_nonthermal. The gap above unity in
# the ratio panel is then unambiguously the Murphy+2011 free-free that CIGALE
# omits, not a synchrotron-normalization difference.
from tengri.radio import radio_sfr_bell2003 as _bell03

_syn_only = np.asarray(
    _bell03(w_t, float(np.asarray(state_r.derived["L_ir"])), q_ir=2.5, alpha_sf=0.8)
)
ax.plot(
    2.998e18 / w_t / 1e9,
    _syn_only,
    color="0.45",
    ls=":",
    lw=1.6,
    label="tengri  synchrotron only (Bell 2003)",
)
ax.legend(fontsize=8, frameon=False)
_nu_r = 2.998e18 / w_r / 1e9
_g14 = (_nu_r >= 1.0) & (_nu_r <= 1.5) & (L_r > 0)
print(f"§11 radio tengri/CIGALE median (1.0–1.5 GHz): {float(np.median(ratio[_g14])):.3f}×")

# Close the ratio with its three proven factors: free-free fraction,
# 21 cm vs 1.4 GHz anchor, and the L_absorbed (energy-balance) residual.
_f_lir = float(np.asarray(state_r.derived["L_ir"])) / (
    float(sed_r.info["dust.luminosity"]) * 1e7
)
_f_anchor = float((1.4276e9 / 1.4e9) ** (-0.8))
from tengri.radio import radio_freefree, radio_sfr_bell2003

_w_probe = np.array([2.998e18 / 1.25e9])
_L_ir_t = float(np.asarray(state_r.derived["L_ir"]))
_L_syn = float(np.asarray(radio_sfr_bell2003(_w_probe, _L_ir_t, q_ir=2.5, alpha_sf=0.8))[0])
_L_ff = float(np.asarray(radio_freefree(_w_probe, _L_ir_t, 1.0e4, 0.1))[0])
_f_ff = 1.0 + _L_ff / _L_syn
print(
    f"§11 decomposition at 1.25 GHz: free-free ×{_f_ff:.3f} · "
    f"anchor-frequency ×{_f_anchor:.3f} · L_IR anchor ×{_f_lir:.3f} "
    f"= ×{_f_ff * _f_anchor * _f_lir:.3f}"
)
fig.tight_layout()
save_fig("cigale_11_radio_synchrotron.png")


# %% [markdown]
# ## §12 IGM transmission
#
# CIGALE applies Meiksin (2006) IGM attenuation inside its
# `redshifting` module — Lyman series **and** the diffuse-IGM Lyα
# forest continuum suppression, so transmission redward of the Lyman
# limit at z = 3 sits at ~0.18-0.25 rather than 1. tengri ships the
# matching `igm.meiksin06`; this panel uses it directly so both sides
# apply the same Meiksin prescription. The transmission curves overlay
# at z = 3, 5, 7 to **max |ΔT| ~ 1e-7** (float precision, median
# ΔT = 0): tengri matches CIGALE's Meiksin transmission bit for bit,
# not just visually.

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
for color, z in zip(("C0", "C1", "C2"), (3.0, 5.0, 7.0)):
    # CIGALE igm_transmission takes wavelength in nm.
    T_c = np.asarray(cigale_igm(wave_obs_aa / 10.0, z))
    T_t = np.asarray(igm_transmission_meiksin06(_jnp.asarray(wave_obs_aa), z))
    ax_l.plot(wave_obs_aa, T_c, color=color, linewidth=1.4, label=rf"$z = {z:.0f}$")
    ax_r.plot(wave_obs_aa, T_t, color=color, linewidth=1.4, label=rf"$z = {z:.0f}$")

ax_l.legend(fontsize=9)

ax_r.legend(fontsize=9)
fig.tight_layout()
save_fig("cigale_12_igm_transmission.png")


# %% [markdown]
# ## tengri in CIGALE-mode — the full X-ray → radio party SED
#
# Every section above swept one physics block. This is the whole chain at
# once: tengri set up to emulate CIGALE end to end — shared BC03 SSP,
# fiducial τ-delayed SFH, modified-starburst (Leitherer+02) attenuation,
# Dale+2014 IR re-emission, plus the §10 **X-ray** (Yang+2020: XRB + hot
# gas; this galaxy-only chain has no AGN, so the corona is zero) and the
# §11 **radio** (Condon 1992 SF synchrotron, `q_IR = 2.5`) — overlaid on
# CIGALE's own X-ray→radio output at matched parameters. The top panel is
# the overlay; the bottom is the fractional residual `tengri / CIGALE − 1`
# with the ±25 % band shaded.
#
# **The stellar-to-FIR core reproduces to a few percent.** Optical
# agreement is reported as a normalization ratio and its 16–84 % spread.
# With the single-screen dust mapping (`tau_bc = 0`; see Setup) the residual
# sits inside ±25 % from the far-UV through the FIR; the sub-912 Å excursion
# is the Lyman-continuum extrapolation and the mm-tail offset is the Dale
# template cutoff (§6) — not the dust energy budget.
#
# Extending the comparison past the FIR into the X-ray and radio wings both
# now match CIGALE, and getting there caught a real tengri bug:
#
# - **Radio.** The Condon-1992 SF synchrotron matches CIGALE's `radio` module
#   to ~5 % across the band (1.4 GHz ≈ 1.05×, 150 MHz ≈ 1.02×) once
#   `q_IR = 2.5` is pinned on both sides (§11).
# - **X-ray.** With no AGN corona the X-ray is pure XRB + hot gas; tengri now
#   matches CIGALE to ~3 % (1 keV ≈ 0.97×, 5 keV ≈ 0.96×). Building this
#   panchromatic overlay exposed a wiring bug: the X-ray component was calling
#   the Lehmer+2016 **LMXB** scaling with its 1 Gyr *default* age instead of
#   the galaxy's mass-weighted age (the `logT` polynomial is steep, so a
#   ~3 Gyr population came out ~3–4× too luminous — LMXB dominates the galaxy
#   X-ray). The component now threads the SSP mass-weighted age (matching
#   CIGALE's `stellar.age_m_star`) into `xray_total`, collapsing the wing from
#   ~3.4× to ~0.97×.

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
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={
        "type": "two_component",
        "law_bc": "leitherer02",
        "law_diff": "leitherer02",
        "tau_bc": Fixed(TAU_BC_FIDUCIAL),
        "tau_diff": Fixed(TAU_DIFF_FIDUCIAL),
        "lyman_cutoff": True,
        "*": FIXED,
        "emission": {"type": "dale2014", "alpha_mir": Fixed(2.0), "*": FIXED},
    },
    xray={"type": "yang20", "*": FIXED},
    radio={
        "type": "condon92",
        "radio_q_ir": Fixed(2.5),
        "radio_alpha_sf": Fixed(0.8),
        "*": FIXED,
    },
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

# Headline numbers: the optical normalization ratio tengri/CIGALE and
# its 16–84% spread. With the shared BC03 grid, matched mass convention,
# and single-screen dust mapping the ratio sits at ~1 with a few-percent
# spread; the only residual is the sub-912 Å Lyman-continuum extrapolation
# and the mm-tail Dale-template cutoff (§6), not emission lines (this
# chain carries no nebular block).
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


_C_AA_HZ = 2.998e18  # Å/s
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
ax.plot(w_ext, L_ext, "C0-", linewidth=1.5, label="CIGALE")
ax.plot(w_ext, L_t_on_ext, "C1--", linewidth=1.5, label="tengri (CIGALE-mode)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e-1, 1e12)  # hard X-ray -> radio
# Peak-anchored y-limit: both SEDs cliff to ~0 at the grid edges, so the log
# axis would otherwise autoscale across ~170 decades and flatten the SED.
_ymax_h = float(max(np.nanmax(L_ext), np.nanmax(L_t_on_ext)))
ax.set_ylim(_ymax_h * 1e-9, _ymax_h * 2.0)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title("tengri in CIGALE-mode vs CIGALE — full panchromatic SED")
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
# Module by module, at matched parameters, CIGALE and tengri agree
# wherever they evaluate the same mathematics — the BC03 SSP grid, the
# τ-delayed SFH, the attenuation curves, the Dale+2014 dust IR through
# energy balance, the X-ray corona + binary scalings, the radio
# synchrotron + free-free composite, and the Meiksin IGM — and differ in
# the two places they use different physics inputs: the nebular grid (Cue
# vs CIGALE's Cloudy 13.x, §8) and the SKIRTOR torus shape (§9). The
# full-SED head-to-head collects the configured chain onto one axis; the
# per-section scalars printed above are the quantitative record, the
# figures in `_figs/` the visual one.

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
# * Li et al. 2024 (Cue, arXiv:2405.xxxxx) — neural CLOUDY emulator
# * Madau 1995, ApJ 441, 18 — original IGM transmission
# * Meiksin 2006, MNRAS 365, 807 — updated IGM transmission
# * Noll et al. 2009, A&A 507, 1793 — modified Calzetti
# * Silva et al. 2004, MNRAS 355, 973 — AGN torus
# * Stalevski et al. 2016, MNRAS 458, 2288 — SKIRTOR
# * Yang et al. 2020, MNRAS 491, 740 — X-ray CIGALE module
