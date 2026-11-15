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
# # Reproducing BEAGLE's physics with tengri
#
# BEAGLE (Chevallard & Charlot 2016) is a Bayesian SED fitting code. This page compares tengri with
# it block by block: SSP templates, star formation history, the integrated SED, dust attenuation,
# the attenuated SED, and IGM absorption. BEAGLE 0.29.2 runs here, in its Docker image on its own
# licensed templates, and only its outputs appear on the left-hand side; tengri's side uses public
# ingredients only and never reads a BEAGLE template file. BEAGLE runs are executed once by the
# driver and cached outside the repository, so rendering re-runs nothing, and no BEAGLE-derived data
# is committed: the figures and printed numbers are the record.
#
# Not compared here: dust emission, X-ray, radio and shock components, which BEAGLE does not model.

# %% [markdown]
# ## Setup
#
# The code cell below loads the libraries, checks the unit conversion, opens the cached BEAGLE
# runs, and prints the provenance of one run. Every BEAGLE number on this page comes from
# `beagle(section)`, which reads the cache and raises if a section has not been run.

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import itertools
import time
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from astropy.cosmology import Planck18
from IPython.display import Markdown, display

from reproduction import _validation as V
from reproduction.beagle._drivers import beagle_driver, beagle_run
from reproduction.beagle._drivers import units as U
from reproduction.beagle._drivers.beagle_matrix import RUN_MATRIX, by_section

import tengri
from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.utils.physics_constants import L_SUN
from tengri.components.dust.attenuation import calzetti, cardelli, chevallard13, lmc, power_law, smc
from tengri import FREE
from tengri.components.agn import get_feltre_backend
from tengri.components.nebular.agn_nebular import _log_qh_from_lacc

# Force the inline backend so figures embed on (re-)render regardless of the
# ambient MPLBACKEND. A non-inline backend (e.g. Agg) drops the save_fig()
# calls' on-screen copy; as a script this is a no-op.
try:  # noqa: SIM105
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

warnings.filterwarnings("ignore")
tengri.plot.setup_style()
_T0 = time.perf_counter()
_FIG_DPI = 150
_HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd().resolve()
figs_dir = _HERE / "_figs"
figs_dir.mkdir(exist_ok=True)


def save_fig(filename: str) -> None:
    """Save the current figure to ``_figs/`` and leave it open so inline embeds work."""
    plt.savefig(str(figs_dir / filename), dpi=_FIG_DPI, bbox_inches="tight")


_unit_check = U.verify_unit_conversion(rtol=1e-3)
print(f"unit-conversion bolometric round-trip: rel_err = {_unit_check['rel_err']:.2e}  (target < 1e-3)")

beagle_run.require_available()
beagle_driver.require_available()


def beagle(section):
    """The cached BEAGLE run for one matrix section. A cache miss raises; nothing is rerun here."""
    return beagle_run.run(by_section(section)[0])


SSP_PATH = beagle_driver.DATA_DIR / "bc03_pdva_stelib_chabrier.h5"
if not SSP_PATH.is_file():
    raise SystemExit(
        f"{SSP_PATH} is missing. Fetch it with: "
        "python -c \"import tengri; tengri.download_ssp('bc03_pdva_stelib_chabrier')\""
    )
ssp = load_ssp_data(str(SSP_PATH))

_prov = beagle("s1_ssp_bc03").provenance
print(f"BEAGLE image: {_prov['image']}")
print(f"BEAGLE version: {str(_prov['version']).splitlines()[0]}")
print(f"BEAGLE templates: {', '.join(_prov['templates'])}")
print(f"BEAGLE cache: {beagle_run.workdir() / 'cache'}")

# BEAGLE's solar metallicity is 0.01524; tengri's is 0.0142. A BEAGLE log Z/Z☉ value maps to
# tengri's scale by adding LOG10_ZSUN_SHIFT, so both sides request the same absolute Z.
SHIFT = beagle_driver.LOG10_ZSUN_SHIFT
AGE_Z1_GYR = float(Planck18.age(1.0).to_value("Gyr"))
LOG_MASS = 10.0
RECENT_MYR = 10.0
RECENT_LOG_SFR = 1.0
M_FORMED = 10.0**LOG_MASS  # BEAGLE `mass`: total formed, episode included


def bands_candels():
    """CANDELS curves from the committed workshop catalog, each scaled to unit peak.

    HAWKI_KS is stored as published, with values below 100, so peak scaling also removes that factor.
    """
    out = []
    with h5py.File(beagle_driver.WORKSHOP_PATH, "r") as f:
        for key in sorted(f["filters"].keys()):
            w = np.asarray(f["filters"][key]["wave_aa"], float)
            t = np.asarray(f["filters"][key]["response"], float)
            order = np.argsort(w)
            out.append((key, w[order], t[order] / t.max()))
    return tuple(out)


CANDELS = bands_candels()
print(f"CANDELS bands: {len(CANDELS)} ({', '.join(b[0] for b in CANDELS)})")


def candels_rows(w_t, L_t, w_ref, L_ref, *, z):
    """Band-average each SED through the CANDELS curves moved to the rest frame at ``z``.

    Returns rows in the layout of ``V.filter_rows_native``, with the rest-frame pivot [um].
    """
    rows = []
    for name, fw_obs, ft in CANDELS:
        fw = fw_obs / (1.0 + z)
        a = V.band_average(w_t, L_t, fw, ft)
        b = V.band_average(w_ref, L_ref, fw, ft)
        ratio = a / b if np.isfinite(a) and np.isfinite(b) and b > 0 else float("nan")
        rows.append((name, V.pivot_wavelength(fw, ft) / 1e4, a, b, ratio))
    return rows


def sfh_dict(tau_gyr):
    """Delayed history plus the current episode (SFR_10 = 10**log_sfr over 10 Myr); the formed mass is the total."""
    return {
        "type": ["delayed", "recent"],
        "tau_gyr": Fixed(tau_gyr),
        "age_gyr": Fixed(AGE_Z1_GYR),
        "log_total_mass": Fixed(float(np.log10(M_FORMED))),
        "log_sfr": Fixed(RECENT_LOG_SFR),
        "timescale_myr": Fixed(RECENT_MYR),
        "other_params": Fixed(DEFAULT),
    }


def cf00_dict(tau_v, *, law_diff="power_law"):
    """Two-component dust with power-law curves: slopes -1.3 (birth clouds) and -0.7, mu = 0.4."""
    out = {
        "type": "two_component",
        "parameterization": "tauv_mu",
        "tau_v": Fixed(tau_v),
        "mu": Fixed(0.4),
        "law_bc": "power_law",
        "law_diff": law_diff,
        "slope_bc": Fixed(-1.3),
        "t_birth_yr": 1e7,
        "other_params": Fixed(DEFAULT),
    }
    if law_diff == "power_law":
        out["slope_diff"] = Fixed(-0.7)
    return out


def tengri_model(*, sfh, logzsol, dust=None, z=1.0):
    """SEDModel on the shared SSP; BEAGLE's bc03 runs carry no nebular emission, so it is off here."""
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(logzsol), "logzsol_scatter": Fixed(0.001), "all_params": Fixed(DEFAULT)},
        sfh=sfh,
        dust_attenuation=dust or {"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(z),
    )


def window_mean(w, f, lo, hi):
    """Mean of ``f`` over [lo, hi] Angstrom on a 1 Angstrom grid."""
    g = np.arange(lo, hi + 0.5, 1.0)
    return float(np.mean(np.interp(g, w, f)))


print(f"setup complete in {time.perf_counter() - _T0:.1f} s")

# %% [markdown]
# ## §1 SSP templates
#
# BEAGLE's `bc03` set is the MILES-based `bc2003_hr_xmiless` family; tengri's shipped BC03 grid is
# STELIB. The comparison is therefore a library bracket between two BC03 releases that share
# isochrones and IMF, not an identity. The figure shows single stellar populations of 10^10 M☉
# at Z = 0.02 and log age 7, 8 and 9, and the tables give the ratio in four rest-frame windows.
# Optical and near-infrared window medians lie between 0.96 and 1.07 at all three ages, which is
# consistent with the MILES against STELIB libraries and comparable to the spread between BEAGLE's
# own `cb_june16` and `bc03` sets (§1a). The largest entry is the rest UV at 1 Gyr, where the
# 1500–2500 Å median is 0.72 of BEAGLE. That window lies below both libraries' native ranges
# (STELIB from 3200 Å, MILES from 3540 Å), so it is not a MILES-against-STELIB effect; it is a UV
# template difference, not isolated here.

# %%
AGES_LOG = (7.0, 8.0, 9.0)
WINDOWS_AA = ((1500.0, 2500.0), (3500.0, 5500.0), (5500.0, 9000.0), (10000.0, 25000.0))
r_ssp = beagle("s1_ssp_bc03")
rows_ssp = by_section("s1_ssp_bc03")[0].input_table()
z_abs = 10.0 ** (float(rows_ssp["metallicity"][0])) * beagle_driver.Z_SUN_BEAGLE
i_z = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - np.log10(z_abs))))
print(f"requested Z = {z_abs:.4f}; tengri SSP metallicity node Z = {10.0 ** float(ssp.ssp_lgmet[i_z]):.4f}")

w_b = np.asarray(r_ssp.wl_rest_aa)
cases_ssp = []
for k, lg in enumerate(AGES_LOG):
    i_age = int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - (lg - 9.0))))
    print(f"log age {lg:g}: tengri node log age = {float(ssp.ssp_lg_age_gyr[i_age]) + 9.0:.3f}")
    L_t = np.asarray(ssp.ssp_flux[i_z, i_age, :]) * L_SUN * 10.0**LOG_MASS
    cases_ssp.append((f"log age {lg:g}", w_b, np.asarray(r_ssp.sed_lnu[k]), np.asarray(ssp.ssp_wave), L_t))

fig, (ax, ax_r), _ = V.sweep_fig(
    cases_ssp,
    ref_label="BEAGLE bc03 (MILES)",
    title=r"SSP, $10^{10}\,M_\odot$, $Z=0.02$: BEAGLE vs tengri BC03 (STELIB)",
    xlim=(1e3, 2.5e4),
)
save_fig("beagle_01_ssp.png")
plt.show()

for lo, hi in WINDOWS_AA:
    V.print_window_table(
        V.window_rows(cases_ssp, lo=lo, hi=hi), ref_name="BEAGLE", title=f"§1 window {lo:.0f}-{hi:.0f} A"
    )

# %% [markdown]
# The rows flagged in the four §1 windows are template differences: BEAGLE's `bc03` set is MILES-based and tengri's BC03 grid is STELIB, and the spread between them is not a tengri defect. The largest entry, 0.72 in the 1500–2500 Å window at log age 9, is a UV template difference, not isolated, and does not come from the MILES-against-STELIB choice.

# %% [markdown]
# ### §1a CB16 against BC03 inside BEAGLE
#
# BEAGLE's `cb_june16` set is the Charlot & Bruzual update, the stellar set under every BEAGLE
# nebular run. The tables compare it with BEAGLE's own `bc03` set at the same ages and metallicity,
# so they size the spread between template releases that a tengri comparison inherits.

# %%
r_cb = beagle("s1a_ssp_cb16")
cases_cb = [
    (f"log age {lg:g}", w_b, np.asarray(r_ssp.sed_lnu[k]), np.asarray(r_cb.wl_rest_aa), np.asarray(r_cb.sed_lnu[k]))
    for k, lg in enumerate(AGES_LOG)
]
for lo, hi in WINDOWS_AA:
    V.print_window_table(
        V.window_rows(cases_cb, lo=lo, hi=hi), ref_name="BEAGLE bc03", title=f"§1a CB16 / BC03, window {lo:.0f}-{hi:.0f} A"
    )

# %% [markdown]
# The rows flagged in the §1a windows are the spread between BEAGLE's own `cb_june16` and `bc03` templates. That is the library bracket a tengri comparison inherits; it is not a difference between tengri and BEAGLE.

# %% [markdown]
# ## §2 Star formation history
#
# The `s2_sfh_delayed` run is a delayed history with tau of 0.1, 1 and 10 Gyr at z = 1, a formed mass
# of 10^10 M☉ and a current SFR of 10 M☉/yr over the last 10 Myr. tengri builds the same
# delayed member with the `recent` episode on top, the formed mass set to BEAGLE's `mass`. The figure shows the star formation rate against lookback time, and the tables
# compare the 10 Myr episode, the smooth history and the integrated properties. BEAGLE's `M_tot` is
# the total formed mass including the episode and `M_star` the surviving mass. For the 10 Myr episode
# at tau = 0.1 Gyr, tengri's SFR_10 is 5 % above BEAGLE's and SFR_100 is 5 % below; this is numerical on
# tengri's side, because its history grid starts at 1 Myr lookback.

# %%
r_sfh = beagle("s2_sfh_delayed")
rows_in = by_section("s2_sfh_delayed")[0].input_table()
logz_in = float(rows_in["metallicity"][0]) + SHIFT
taus_gyr = [10.0 ** (float(t) - 9.0) for t in rows_in["tau"]]

cases_sfh, prop_rows, m_int = [], [], []
for k, tau in enumerate(taus_gyr):
    lb_b = np.asarray(r_sfh.sfh["lookback_age"][k], float)
    sfr_b = np.asarray(r_sfh.sfh["SFR"][k], float)
    keep_b = (lb_b >= 0) & (sfr_b > 0)
    m = tengri_model(sfh=sfh_dict(tau), logzsol=logz_in)
    st = m.predict_state({})
    lb_t = np.asarray(st.derived["sfh_grid_lbt_yr"], float)
    sfr_t = np.asarray(st.derived["sfr_history"], float)
    keep_t = (lb_t >= 0) & (sfr_t > 0)
    _o = np.argsort(lb_t)
    m_int.append(float(np.trapezoid(sfr_t[_o], lb_t[_o])))
    cases_sfh.append((f"tau = {tau:g} Gyr", lb_b[keep_b] / 1e9, sfr_b[keep_b], lb_t[keep_t] / 1e9, sfr_t[keep_t]))
    prop_rows.append(
        m.predict_properties({}, names=("sfr_10myr", "sfr_100myr", "stellar_mass", "stellar_mass_surviving"))
    )

fig, (ax, ax_r), _ = V.sweep_fig(
    cases_sfh,
    ref_label="BEAGLE",
    title=r"Delayed history plus 10 Myr episode, $z=1$",
    xlabel="lookback time [Gyr]",
    ylabel=r"SFR [$M_\odot$/yr]",
    logy=True,
    xlim=(0.0, 6.0),
    dyn_range=1e-3,
)
save_fig("beagle_02_sfh.png")
plt.show()

print("\n§2 integrated properties, BEAGLE / tengri / ratio")
print("  M_tot(int) is the integral of the evaluated tengri history.")
print(f"  {'tau [Gyr]':>9s} {'quantity':>10s} {'BEAGLE':>12s} {'tengri':>12s} {'ratio':>8s}")
sf, gp = r_sfh.star_formation, r_sfh.galaxy_properties
for k, tau in enumerate(taus_gyr):
    for label, b, t in (
        ("SFR_10", float(sf["SFR_10"][k]), float(prop_rows[k]["sfr_10myr"])),
        ("SFR_100", float(sf["SFR_100"][k]), float(prop_rows[k]["sfr_100myr"])),
        ("M_tot", float(gp["M_tot"][k]), float(prop_rows[k]["stellar_mass"])),
        ("M_tot(int)", float(gp["M_tot"][k]), m_int[k]),
        ("M_star", float(gp["M_star"][k]), float(prop_rows[k]["stellar_mass_surviving"])),
    ):
        print(f"  {tau:9g} {label:>10s} {b:12.4e} {t:12.4e} {t / b:8.3f}")

# Episode and smooth history, as the median tengri/BEAGLE ratio inside each lookback window.
for k, tau in enumerate(taus_gyr):
    for lo, hi, name in ((1e-3, 8e-3, "episode, 1-8 Myr"), (0.015, 5.5, "smooth, 15 Myr-5.5 Gyr")):
        V.print_window_table(
            V.window_rows([cases_sfh[k]], lo=lo, hi=hi, rel_to="peak"),
            ref_name="BEAGLE",
            title=f"§2 tau = {tau:g} Gyr, {name}",
            x_unit="Gyr",
        )

# %% [markdown]
# The two §2 rows flagged for tau = 0.1 Gyr are numerical, from the sampling of the history on tengri's age grid, the same class as the episode SFR_10 and SFR_100 values in the summary. The episode case is 1.054 in the median; the smooth case, 1.94 in the median, is not isolated here and is left at this classification.

# %% [markdown]
# ## §3 Integrated SED
#
# The `s3_fiducial` run is the 1 Gyr delayed history with the current episode at z = 1, with no dust
# and no nebular emission on either side (BEAGLE's `bc03` run carries none), so the SED is the
# stellar template integrated over the history. BEAGLE's age is the age of the universe at z = 1
# under its cosmology and tengri's is its own, 5.851 Gyr. The figure overlays rest-frame L_nu on
# BEAGLE's grid; the table integrates both through the CANDELS curves moved to the rest frame, with
# the rest-frame pivot beside each band (ratios are tengri over BEAGLE). The four bands sampling the
# rest-frame ultraviolet agree to better than 1 %, so normalization, age and star formation history
# match; redward of 3500 Å BEAGLE lies between 5 % below and 0.5 % above tengri (median 2 % below),
# the range of the empirical stellar libraries of the two BC03 variants (MILES in BEAGLE's set,
# STELIB in tengri's).

# %%
r3 = beagle("s3_fiducial")
w3, L3 = np.asarray(r3.wl_rest_aa), np.asarray(r3.sed_lnu[0])
logz3 = float(by_section("s3_fiducial")[0].input_table()["metallicity"][0]) + SHIFT
m3 = tengri_model(sfh=sfh_dict(1.0), logzsol=logz3)
p3 = m3.predict({})
w3_t, L3_t = np.asarray(p3.wave_rest), np.asarray(p3.rest_sed())

fig, ax, ax_r, _ = V.overlay_ratio_fig(
    w3, L3, w3_t, L3_t, title=r"Fiducial, $z=1$, no dust, no nebular emission (rest frame)",
    ref_label="BEAGLE", label_t="tengri", xlim=(1e3, 2.5e4),
)
save_fig("beagle_03_sed.png")
plt.show()

rows3 = candels_rows(w3_t, L3_t, w3, L3, z=1.0)
V.print_filter_table(rows3, ref_name="BEAGLE", title="§3 CANDELS bands (pivot = rest-frame um at z = 1)", tol=0.02)
_r3 = np.array([r[4] for r in rows3])
_uv = np.array([r[4] for r in rows3 if r[1] < 0.35])
_ir = np.array([r[4] for r in rows3 if r[1] >= 0.35])
print(f"§3 rest pivot < 0.35 um: ratio {_uv.min():.3f} to {_uv.max():.3f}; "
      f">= 0.35 um: {_ir.min():.3f} to {_ir.max():.3f}")
for lo, hi in WINDOWS_AA:
    V.print_window_table(
        V.window_rows([("fiducial", w3, L3, w3_t, L3_t)], lo=lo, hi=hi), ref_name="BEAGLE",
        title=f"§3 window {lo:.0f}-{hi:.0f} A",
    )

# %% [markdown]
# The rows flagged in §3 are the library bracket. The CANDELS pivots redward of 0.35 μm differ by 2 to 7 %, the same MILES-against-STELIB spread that §1 shows, and the window maxima of up to 15 % sit in the 1500–2500 Å and 3500–5500 Å windows, where the same template difference applies.

# %% [markdown]
# ## §4 Attenuation curves
#
# Each `s4_*` run applies one attenuation law at a V-band depth of 1 to the §3 history, so the
# effective depth is tau_lambda = -ln(F_dust / F_no-dust), with the dust-free `s3_fiducial` run as the
# reference. For the single-screen laws (Calzetti, and BEAGLE's Milky Way, LMC and SMC tables) this is the
# curve itself. For CF00 and CCWW13 universal, the effective depth of the whole population lies
# between the two stellar groups: old stars see the diffuse ISM alone (mu = 0.4) and stars younger
# than 10 Myr see it plus the birth cloud. The two analytic bounds are drawn dashed, and the solid
# line is the tengri model's own ratio with the same history. The CCWW13 variant with the
# radiative-transfer tables (`ccww13_full`) is not run, because those tables are not distributed with
# BEAGLE. BEAGLE's `Calzetti` is the 1994 curve and tengri's `calzetti` is the 2000 curve, so that
# panel carries a curve difference; the printed table gives it. The Milky Way panel compares BEAGLE's
# shipped Milky Way table with Cardelli et al. (1989) on tengri's side. BEAGLE reads its Milky Way,
# LMC and SMC curves from that one table, and its measured LMC and SMC depths equal the table's
# extinction column to 0.1 %. Redward of 1 μm the LMC and SMC depths of tengri exceed BEAGLE's by
# 24 % and 62 % at 10000 Å; at 20000 Å they are 1.6 and 2.8 times larger. The manual calls the table
# Pei (1992), but over 1 to 2 μm it is 0.62 to 0.81 of the Pei Table 4 Drude sums for the LMC and
# 0.36 to 0.62 for the SMC, which are the sums tengri's `lmc` and `smc` use. The difference lies on
# BEAGLE's side of the comparison: a convention, not a tengri defect.

# %%
MU = 0.4
SECTIONS_S4 = [r.section for r in RUN_MATRIX if r.section.startswith("s4_")]
LAWS = {
    "s4_cf00": ("CF00", None),
    "s4_calzetti": ("Calzetti (1994 | 2000)", calzetti),
    "s4_ccww13_universal": ("CCWW13 universal", None),
    "s4_mw": ("Milky Way (BEAGLE table | CCM89)", cardelli),
    "s4_lmc": ("LMC (BEAGLE table)", lmc),
    "s4_smc": ("SMC (BEAGLE table)", smc),
}
wl4 = np.arange(1000.0, 25001.0, 100.0)
_w = lambda a: np.asarray(a, float)  # noqa: E731
bounds = {
    "s4_cf00": (MU * (wl4 / 5500.0) ** -0.7, MU * (wl4 / 5500.0) ** -0.7 + (1 - MU) * (wl4 / 5500.0) ** -1.3),
    "s4_ccww13_universal": (
        MU * _w(chevallard13(wl4, tau_v=MU)) / float(chevallard13(np.array([5500.0]), tau_v=MU)[0]),
        MU * _w(chevallard13(wl4, tau_v=MU)) / float(chevallard13(np.array([5500.0]), tau_v=MU)[0])
        + (1 - MU) * (wl4 / 5500.0) ** -1.3,
    ),
}
print(f"CCWW13 curve at 5500 A (tau_V = {MU}): {float(chevallard13(np.array([5500.0]), tau_v=MU)[0]):.4f}")

ref3 = np.asarray(r3.sed_lnu[0])
tau_b, tau_t = {}, {}
for sec in SECTIONS_S4:
    rb = beagle(sec)
    tau_b[sec] = -np.log(np.interp(wl4, rb.wl_rest_aa, rb.sed_lnu[0]) / np.interp(wl4, r3.wl_rest_aa, ref3))
    if sec in bounds:
        law = "power_law" if sec == "s4_cf00" else "chevallard13"
        pdm = tengri_model(sfh=sfh_dict(1.0), logzsol=logz3, dust=cf00_dict(1.0, law_diff=law)).predict({})
        tau_t[sec] = -np.log(
            np.interp(wl4, _w(pdm.wave_rest), _w(pdm.rest_sed())) / np.interp(wl4, w3_t, L3_t)
        )
    else:
        k = _w(LAWS[sec][1](wl4))
        k = k / float(np.asarray(LAWS[sec][1](np.array([5500.0])))[0])
        tau_t[sec] = k

fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.4), sharex=True, sharey=True)
for ax, sec in zip(axes.ravel(), SECTIONS_S4):
    ax.plot(wl4, tau_b[sec], "o", ms=3.0, color="C0", alpha=0.7, label="BEAGLE")
    ax.plot(wl4, tau_t[sec], "-", color="C1", lw=1.4, label="tengri")
    if sec in bounds:
        ax.plot(wl4, bounds[sec][0], "--", color="0.45", lw=0.9, label="old stars")
        ax.plot(wl4, bounds[sec][1], ":", color="0.45", lw=0.9, label="young stars")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(LAWS[sec][0], fontsize=9)
    ax.set_xlim(1e3, 2.5e4)
    ax.set_ylim(0.02, 8.0)
for ax in axes[1]:
    ax.set_xlabel(r"$\lambda$ [Å]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$\hat\tau_\lambda$")
axes[0, 0].legend(fontsize=7, frameon=False, loc="lower left")
fig.tight_layout()
save_fig("beagle_04_attenuation_curves.png")
plt.show()

CHECK_AA = (1500.0, 2200.0, 5500.0, 10000.0, 20000.0)
print("§4 effective depth tau_lambda, BEAGLE / tengri / ratio")
print(f"  {'section':<22s}" + "".join(f"{int(a):>26d} A" for a in CHECK_AA))
for sec in SECTIONS_S4:
    cells = []
    for a in CHECK_AA:
        b = float(np.interp(a, wl4, tau_b[sec]))
        t = float(np.interp(a, wl4, tau_t[sec]))
        cells.append(f"{b:8.3f} {t:8.3f} {t / b:7.3f}")
    print(f"  {sec:<22s}" + "".join(f"{c:>28s}" for c in cells))

# The Calzetti 1994 curve is a piecewise form of the same shape; the BEAGLE depth normalized to V
# against the 2000 curve gives the curve difference directly.
tb_cal, tt_cal = tau_b["s4_calzetti"], tau_t["s4_calzetti"]
ratio_cal = tt_cal / tb_cal
for lo, hi in ((1200.0, 2200.0), (2200.0, 5500.0), (5500.0, 12000.0)):
    sel = (wl4 >= lo) & (wl4 <= hi)
    print(f"§4 Calzetti 2000 / 1994 depth ratio, {lo:.0f}-{hi:.0f} A: median {np.median(ratio_cal[sel]):.3f}, "
          f"range {ratio_cal[sel].min():.3f} to {ratio_cal[sel].max():.3f}")
for sec in ("s4_mw", "s4_lmc", "s4_smc"):
    sel = (wl4 >= 1200) & (wl4 <= 12000)
    r_ = (tau_t[sec] / tau_b[sec])[sel]
    print(f"§4 {sec} ratio 1200-12000 A: median {np.median(r_):.3f}, range {r_.min():.3f} to {r_.max():.3f}")

# %% [markdown]
# ## §5 Attenuation applied
#
# The `s5_cf00_tauv` run applies CF00 at V-band depths of 0.3, 1 and 2 (mu = 0.4) to the §3 history.
# tengri applies the same two power laws, with stars younger than 10 Myr seeing the birth cloud
# and the diffuse ISM and older stars the diffuse ISM alone, the split being the exact formed-mass
# fraction. The figure shows the SED ratio for each depth; the tables give the CANDELS bands and the
# attenuation A = -2.5 log10(F_dust / F_no-dust) in 100 Å windows at 1500 and 5500 Å.

# %%
r5 = beagle("s5_cf00_tauv")
tau_vs = [float(t) for t in by_section("s5_cf00_tauv")[0].input_table()["tauV_eff"]]
cases5, A_rows = [], []
for k, tv in enumerate(tau_vs):
    md = tengri_model(sfh=sfh_dict(1.0), logzsol=logz3, dust=cf00_dict(tv))
    pd_ = md.predict({})
    w_d, L_d = np.asarray(pd_.wave_rest), np.asarray(pd_.rest_sed())
    L_b = np.asarray(r5.sed_lnu[k])
    cases5.append((rf"$\tau_V$ = {tv:g}", np.asarray(r5.wl_rest_aa), L_b, w_d, L_d))
    for name, lo, hi in (("A_1500", 1450.0, 1550.0), ("A_V", 5450.0, 5550.0)):
        a_b = -2.5 * np.log10(window_mean(r5.wl_rest_aa, L_b, lo, hi) / window_mean(w3, L3, lo, hi))
        a_t = -2.5 * np.log10(window_mean(w_d, L_d, lo, hi) / window_mean(w3_t, L3_t, lo, hi))
        A_rows.append((tv, name, a_b, a_t))

fig, (ax, ax_r), _ = V.sweep_fig(
    cases5, ref_label="BEAGLE", title=r"CF00 attenuated SED, $\mu=0.4$, $z=1$ (rest frame)",
    xlim=(1e3, 2.5e4), dyn_range=1e-6,
)
save_fig("beagle_05_attenuation_applied.png")
plt.show()

print("§5 attenuation in 100 A windows [mag]: BEAGLE / tengri / difference")
print(f"  {'tau_V':>6s} {'quantity':>8s} {'BEAGLE':>8s} {'tengri':>8s} {'diff':>8s}")
for tv, name, a_b, a_t in A_rows:
    print(f"  {tv:6.1f} {name:>8s} {a_b:8.3f} {a_t:8.3f} {a_t - a_b:8.3f}")

for k, tv in enumerate(tau_vs):
    pd_ = cases5[k]
    V.print_filter_table(
        candels_rows(pd_[3], pd_[4], pd_[1], pd_[2], z=1.0), ref_name="BEAGLE",
        title=f"§5 CANDELS bands, tau_V = {tv:g} (pivot = rest-frame um)", tol=0.02,
    )

# %% [markdown]
# ## 8. Nebular emission
#
# The `s8_nebular` run is a constant star formation rate of 10 M☉/yr over 1 Gyr at z = 0.001, with
# formed mass 10^10 M☉ and no dust. Its 48 rows vary log U_S from -3.5 to -2, Z from 0.001 to 0.03 and
# xi_d from 0.1 to 0.5; stellar and gas metallicity are equal, n_H is 100 cm^-3, C/O is solar and m_up is
# 100 M☉. tengri runs the same history with the `gutkin16` nebular set. One build per metallicity
# declares log U and xi_d free and sets them at each prediction, and a check below rebuilds one row with
# both fixed. The public Gutkin et al. (2016) table is the third column. It is the line set BEAGLE
# interpolates, with no continuum and no Lyman-alpha. Its values are per unit star formation rate, so only
# ratios are compared with it.

# %%
NEB_LOG_U = (-3.5, -3.0, -2.5, -2.0)
NEB_Z = (0.001, 0.004, 0.014, 0.03)
NEB_XI = (0.1, 0.3, 0.5)
NEB_ROWS = tuple(itertools.product(NEB_LOG_U, NEB_Z, NEB_XI))  # BEAGLE's row order
NEB_CURVE_LOG_U = np.linspace(-3.5, -2.0, 7)  # includes the four BEAGLE log U values
GK_LABELS = tuple(label for label, _ in beagle_driver.GUTKIN_LINES)
GK_WAVE_AA = np.array([wave for _, wave in beagle_driver.GUTKIN_LINES])
HB = GK_LABELS.index("Hb")
HA = GK_LABELS.index("Ha")
ALL_LABELS = GK_LABELS + ("He II 4686",)
Z_SOLAR_TENGRI = 0.0142
CASE_B_ERG = 4.78e-13  # H-beta per ionizing photon, case B [erg] (Osterbrock & Ferland 2006)
HEII_4686_AA = (4685.0, 4689.0)  # BEAGLE only: the public table has no He II 4686
MARKERS = ("o", "s", "^")
LINESTYLES = ("-", "--", ":")

# BEAGLE vacuum-wavelength windows that select each Gutkin+2016 line; doublet components are summed.
LINE_WINDOWS_AA = (
    ((3725.0, 3732.0),),  # [OII] 3727 doublet
    ((4861.0, 4864.0),),  # H-beta
    ((4957.0, 4962.0),),  # [OIII] 4959
    ((5005.0, 5010.0),),  # [OIII] 5007
    ((6546.0, 6551.0),),  # [NII] 6548
    ((6562.0, 6566.0),),  # H-alpha
    ((6582.0, 6587.0),),  # [NII] 6584
    ((6715.0, 6720.0),),  # [SII] 6717
    ((6729.0, 6734.0),),  # [SII] 6731
    ((1236.0, 1245.0),),  # N V 1240 doublet
    ((1547.0, 1549.5),),  # C IV 1548
    ((1549.5, 1553.0),),  # C IV 1551
    ((1638.0, 1642.0),),  # He II 1640
    ((1659.0, 1663.0),),  # O III] 1661
    ((1664.0, 1668.0),),  # O III] 1666
    ((1881.0, 1885.0),),  # [Si III] 1883
    ((1881.0, 1885.0), (1890.0, 1894.0)),  # Si III] 1888 column: BEAGLE Si3@1883 + Si3@1892 (no Si3@1888 key)
    ((1904.0, 1910.5),),  # C III] 1908 doublet
)
assert len(LINE_WINDOWS_AA) == len(GK_LABELS)


def beagle_lines(run, windows, kind="_lum_em"):
    """BEAGLE line luminosities [erg/s], shape (rows, len(windows)); each column sums the keys in its windows.

    Keys carry the air label and the vacuum wavelength after the underscore, so the match uses vacuum.
    """
    keys = [k for k in run.line_lums_lsun if k.endswith(kind)]
    vac = {k: float(k.split("@")[1].split("_")[1]) for k in keys}
    cols = []
    for groups in windows:
        sel = [k for k in keys if any(lo <= vac[k] <= hi for lo, hi in groups)]
        if not sel:
            raise KeyError(f"no BEAGLE line in {groups}")
        cols.append(sum(np.asarray(run.line_lums_lsun[k], float) for k in sel))
    return np.stack(cols, axis=1) * beagle_driver.L_SUN_BEAGLE


def neb_model(z_abs, *, continuum="analytic", onset_gyr=1.0):
    """tengri build at one absolute Z: stellar and gas metallicity equal, logU and xi_d free per prediction.

    The constant history runs from lookback ``onset_gyr`` [Gyr] to the present.
    """
    lz = float(np.log10(z_abs / Z_SOLAR_TENGRI))
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(lz), "logzsol_scatter": Fixed(0.001), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "const",
            "log_total_mass": Fixed(LOG_MASS),
            "start_gyr": Fixed(onset_gyr),  # onset at lookback onset_gyr
            "end_gyr": Fixed(1e-4),  # cessation at lookback 0.1 Myr, so the rate runs to the present
            "other_params": Fixed(DEFAULT),
        },
        dust_attenuation={"type": "none"},
        neb={
            "type": "gutkin16",
            "logU": FREE,
            "logZ_gas": Fixed(lz),
            "xi_d": FREE,
            "log_nH": Fixed(2.0),
            "co_ratio": Fixed(1.0),
            "mup": 100,
            "continuum": continuum,
            "other_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.001),
    )


def tengri_curves(model):
    """gutkin16 lines [erg/s], log Q_H and SFR over 10 Myr on the grid NEB_XI x NEB_CURVE_LOG_U."""
    lines, log_qh, sfr10 = [], [], []
    for x in NEB_XI:
        for u in NEB_CURVE_LOG_U:
            pred = model.predict({"neb_logU": float(u), "neb_xi_d": float(x)})
            waves = np.asarray(pred.lines.all_waves, float)[: len(GK_LABELS)]
            if np.max(np.abs(waves - GK_WAVE_AA)) > 5.0:
                raise ValueError("tengri line order no longer matches the Gutkin+2016 table")
            lines.append(np.asarray(pred.lines.all_lums, float)[: len(GK_LABELS)])
            log_qh.append(float(pred.log_q_h))
            sfr10.append(float(pred.sfh.sfr_10myr))
    shape = (len(NEB_XI), len(NEB_CURVE_LOG_U))
    return (
        np.reshape(lines, shape + (len(GK_LABELS),)),
        np.reshape(log_qh, shape),
        np.reshape(sfr10, shape),
    )


NEB_MODELS = {z: neb_model(z) for z in NEB_Z}
CURVES = {z: tengri_curves(model) for z, model in NEB_MODELS.items()}


def tengri_at(z, u, x, *, curves=None):
    """tengri lines [erg/s], log Q_H and SFR_10 at one row, read from the curve grid (default: the 1 Gyr history)."""
    lines, log_qh, sfr10 = (CURVES if curves is None else curves)[z]
    i_x, i_u = NEB_XI.index(x), int(np.argmin(np.abs(NEB_CURVE_LOG_U - u)))
    return lines[i_x, i_u], log_qh[i_x, i_u], sfr10[i_x, i_u]


r_neb = beagle("s8_nebular")
tab = by_section("s8_nebular")[0].input_table()
np.testing.assert_allclose(np.asarray(tab["nebular_logU"], float), [u for u, _, _ in NEB_ROWS])
np.testing.assert_allclose(np.asarray(tab["nebular_xi"], float), [x for _, _, x in NEB_ROWS])
np.testing.assert_allclose(
    np.asarray(tab["nebular_Z"], float), np.log10([z / beagle_driver.Z_SUN_BEAGLE for _, z, _ in NEB_ROWS])
)

B_ALL = beagle_lines(r_neb, LINE_WINDOWS_AA + ((HEII_4686_AA,),))  # [erg/s], 18 Gutkin lines + He II 4686
B_LINES = B_ALL[:, : len(GK_LABELS)]
T_ROWS = [tengri_at(z, u, x) for u, z, x in NEB_ROWS]
T_LINES = np.array([t[0] for t in T_ROWS])
T_LOG_QH = np.array([t[1] for t in T_ROWS])
T_SFR10 = np.array([t[2] for t in T_ROWS])
RAW_LINES = np.array(
    [
        [beagle_driver.gutkin_node(z, u, x, 100.0, 1.0, 100.0)[lab] for lab in GK_LABELS]
        for u, z, x in NEB_ROWS
    ]
)
RB, RT, RR = (a / a[:, [HB]] for a in (B_LINES, T_LINES, RAW_LINES))

# Build check: one row rebuilt with logU and xi_d fixed, against the free build predicted at the same values.
_fixed = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(float(np.log10(0.014 / Z_SOLAR_TENGRI))), "logzsol_scatter": Fixed(0.001), "all_params": Fixed(DEFAULT)},
    sfh={"type": "const", "log_total_mass": Fixed(LOG_MASS), "start_gyr": Fixed(1.0), "end_gyr": Fixed(1e-4), "other_params": Fixed(DEFAULT)},
    dust_attenuation={"type": "none"},
    neb={"type": "gutkin16", "logU": Fixed(-3.0), "logZ_gas": Fixed(float(np.log10(0.014 / Z_SOLAR_TENGRI))),
         "xi_d": Fixed(0.3), "log_nH": Fixed(2.0), "co_ratio": Fixed(1.0), "mup": 100, "other_params": Fixed(DEFAULT)},
    redshift=Fixed(0.001),
)
_hb_fixed = float(_fixed.predict({}).lines.all_lums[HB])
_hb_free = float(T_LINES[NEB_ROWS.index((-3.0, 0.014, 0.3)), HB])
print(f"§8 build check (row logU -3, Z 0.014, xi 0.3): H-beta fixed {_hb_fixed:.6e}, free {_hb_free:.6e} erg/s, "
      f"relative difference {abs(_hb_free / _hb_fixed - 1.0):.1e}")

print(f"§8 row 0 (logU {NEB_ROWS[0][0]}, Z {NEB_ROWS[0][1]}, xi {NEB_ROWS[0][2]}): BEAGLE H-beta "
      f"{B_LINES[0, HB] / beagle_driver.L_SUN_BEAGLE:.4e} L_sun, [OIII]5007/H-beta {RB[0, GK_LABELS.index('[OIII]5007')]:.3f}, "
      f"N_ion {float(np.asarray(r_neb.galaxy_properties['N_ion'])[0]):.3f}")

# %% [markdown]
# ### §8a Line ratios to H-beta
#
# Each dot is one of the 48 rows. The table gives, for each line, the median and the maximum of |log10| of
# each pair of sources. The raw table and tengri agree to 1e-14 dex on every line, so tengri's interpolation
# is exact at these nodes. The BEAGLE differences are therefore set by how BEAGLE drives that table (its ionizing
# spectrum and normalization), not by tengri's build; this is an inference from the agreement, not a separate test. Hα/H-beta agrees to 0.002 dex at most,
# [N II] and [S II] to 0.01 dex in the median, and [O III] to 0.02 dex in the median and 0.24 at most.
# He II 1640 carries the largest spread, 0.49 dex in the median and 1.03 at most. Si III] 1888, the table
# column for the blend of BEAGLE's Si3@1883 and Si3@1892 lines, agrees to 0.04 dex in the median and 0.11
# at most. Signed by metallicity, the He II 1640 gap is log10(BEAGLE / tengri) = −0.13, +0.04, +0.98 and
# +0.87 dex at Z 0.001, 0.004, 0.014 and 0.03, and the [O III] 5007 gap is −0.01, +0.00, +0.05 and +0.13 dex.
# BEAGLE's N_ion_HeII/N_ion is −4.68, −3.34, −3.50 and −3.65 dex at the same four metallicities. This is
# consistent with BEAGLE's updated templates, which give harder ionizing spectra at Z ≥ 0.014; it is not
# proven, and this section does not isolate the cause. §8c gives the one-number ionizing comparison.

# %%
fig, ax = plt.subplots(figsize=(9.0, 3.8))
xs = np.arange(len(GK_LABELS))
for i, (arr, off, col, lab) in enumerate(((RB, -0.22, "C0", "BEAGLE"), (RT, 0.0, "C1", "tengri"), (RR, 0.22, "C2", "raw table"))):
    for j in xs:
        ax.plot(np.full(len(NEB_ROWS), j + off), np.log10(arr[:, j]), ls="none", marker=("o", "+", "x")[i],
                ms=3.0, color=col, alpha=0.7, label=lab if j == 0 else None)
ax.set_xticks(xs, GK_LABELS, rotation=60, fontsize=7)
ax.set_ylabel(r"$\log_{10}$(line / H$\beta$)")
ax.legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
fig.tight_layout()
save_fig("beagle_08a_line_ratios.png")
plt.show()

print("§8a line / H-beta over 48 rows: median and max |dlog10| between sources")
print(f"  {'line':>12s} {'BEAGLE/tengri':>15s} {'max':>7s} {'raw/tengri':>12s} {'max':>7s} {'BEAGLE/raw':>12s} {'max':>7s}")
for j, lab in enumerate(GK_LABELS):
    if j == HB:
        continue
    d_bt = np.abs(np.log10(RB[:, j] / RT[:, j]))
    d_rt = np.abs(np.log10(RR[:, j] / RT[:, j]))
    d_br = np.abs(np.log10(RB[:, j] / RR[:, j]))
    print(f"  {lab:>12s} {np.median(d_bt):15.3f} {d_bt.max():7.3f} {np.median(d_rt):12.2e} {d_rt.max():7.1e} "
          f"{np.median(d_br):12.3f} {d_br.max():7.3f}")

# Signed gap per metallicity: median over the rows at each Z of log10(tengri / BEAGLE) for line / H-beta.
print("§8a He II 1640 and [O III] 5007 gap by Z, median over 12 rows of log10(BEAGLE / tengri) (line / H-beta)")
print(f"  {'Z':>6s} {'HeII1640 gap':>13s} {'[OIII]5007 gap':>15s} {'log N_HeII/N_ion':>17s}")
_N_HEII = np.asarray(r_neb.galaxy_properties["N_ion_HeII"], float)
_N_ION_NEB = np.asarray(r_neb.galaxy_properties["N_ion"], float)
_J_HE, _J_O3 = GK_LABELS.index("HeII1640"), GK_LABELS.index("[OIII]5007")
for z in NEB_Z:
    idx = [i for i, (_, zz, _) in enumerate(NEB_ROWS) if zz == z]
    gap_he = np.median(np.log10(RB[idx, _J_HE] / RT[idx, _J_HE]))
    gap_o3 = np.median(np.log10(RB[idx, _J_O3] / RT[idx, _J_O3]))
    print(f"  {z:6.3f} {gap_he:+13.2f} {gap_o3:+15.2f} {np.median(_N_HEII[idx] - _N_ION_NEB[idx]):17.2f}")

# %% [markdown]
# ### §8b Six ratios against log U_S
#
# Points are BEAGLE; lines are tengri on the seven-node log U grid. Color is metallicity and marker or
# line style is xi_d. He II 4686 is not in the public table, so its panel has BEAGLE points only.

# %%
PANELS_8B = (
    (r"[O III] 5007 / H$\beta$", "[OIII]5007", "Hb"),
    (r"[O II] 3727 / H$\beta$", "[OII]3727", "Hb"),
    (r"[N II] 6584 / H$\alpha$", "[NII]6584", "Ha"),
    (r"H$\alpha$ / H$\beta$", "Ha", "Hb"),
    (r"He II 4686 / H$\beta$", "He II 4686", "Hb"),
    (r"C III] 1908 / H$\beta$", "CIII]1908", "Hb"),
)
fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.0))
for ax, (title, num, den) in zip(axes.ravel(), PANELS_8B):
    n, d = ALL_LABELS.index(num), ALL_LABELS.index(den)
    rb = B_ALL[:, n] / B_ALL[:, d]
    rt_med = []
    for iz, z in enumerate(NEB_Z):
        for ix, x in enumerate(NEB_XI):
            sel = [i for i, (_, zz, xx) in enumerate(NEB_ROWS) if zz == z and xx == x]
            ax.plot([NEB_ROWS[i][0] for i in sel], rb[sel], ls="none", marker=MARKERS[ix], ms=3.5, color=f"C{iz}")
            if n < len(GK_LABELS):
                lines = CURVES[z][0]
                curve = lines[ix, :, n] / lines[ix, :, d]
                ax.plot(NEB_CURVE_LOG_U, curve, ls=LINESTYLES[ix], color=f"C{iz}", lw=1.0)
                rt_med.append(np.abs(np.log10(rb[sel] / (T_LINES[sel, n] / T_LINES[sel, d]))))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"$\log_{10} U_S$", fontsize=8)
    if rt_med:
        print(f"§8b {title}: median |dlog10| BEAGLE/tengri over 48 rows {np.median(np.concatenate(rt_med)):.3f}, "
              f"max {np.concatenate(rt_med).max():.3f}")
    else:
        print(f"§8b {title}: BEAGLE only (no line in the public table)")
for z, c in zip(NEB_Z, range(len(NEB_Z))):
    axes[0, 0].plot([], [], color=f"C{c}", label=f"Z = {z:g}")
for ix, x in enumerate(NEB_XI):
    axes[0, 0].plot([], [], color="0.3", ls=LINESTYLES[ix], marker=MARKERS[ix], label=f"xi_d = {x:g}")
axes[0, 0].legend(fontsize=6, frameon=False, loc="best")
fig.tight_layout()
save_fig("beagle_08b_ratio_panels.png")
plt.show()

# %% [markdown]
# ### §8c H-beta per ionizing photon and per SFR
#
# BEAGLE's photon rate is `N_ion` and tengri's is `log_q_h`. Case B gives 4.78e-13 erg per photon. The
# per-SFR column uses each source's own 10 Myr rate, which is 10 M☉/yr in both, so the per-SFR ratio is the
# per-photon ratio times the photon-rate ratio, row by row. Per photon the two codes agree to 1.4 percent in
# the median, so the line conversion matches once the photons are counted. The photon-rate ratio is 0.78,
# 0.80, 0.64 and 0.75 at Z 0.001, 0.004, 0.014 and 0.03, so the per-SFR ratio (0.73 in the median) is the
# ionizing output per unit SFR: a library bracket between CB16 (BEAGLE) and BC03 STELIB (tengri). tengri's
# stellar Z is weighted onto its nearest SSP node alone (`logzsol_scatter` 0.001 dex): Z 0.014 and 0.03 both
# run at the 0.02 node and Z 0.001 at the 0.0004 node, while BEAGLE's CB16 grid has exact nodes. Z 0.004 is the
# only metallicity on both grids, and there the ionizing output differs by −0.10 dex (ratio 0.80), the clean
# library number.

# %%
N_ION_B = np.asarray(r_neb.galaxy_properties["N_ion"], float)
SFR10_B = np.asarray(r_neb.star_formation["SFR_10"], float)
PER_PHOTON_B = B_LINES[:, HB] / 10.0**N_ION_B
PER_PHOTON_T = T_LINES[:, HB] / 10.0**T_LOG_QH
PER_SFR_B = B_LINES[:, HB] / SFR10_B
PER_SFR_T = T_LINES[:, HB] / T_SFR10

fig, (ax_p, ax_s) = plt.subplots(1, 2, figsize=(10.0, 3.8))
for iz, z in enumerate(NEB_Z):
    sel = [i for i, (_, zz, xx) in enumerate(NEB_ROWS) if zz == z]
    lu = np.array([NEB_ROWS[i][0] for i in sel])
    ax_p.plot(lu, PER_PHOTON_B[sel], "o", color=f"C{iz}", ms=3.5)
    ax_s.plot(lu, PER_SFR_B[sel], "o", color=f"C{iz}", ms=3.5)
    lines, log_qh, sfr10 = CURVES[z]
    ix = NEB_XI.index(0.3)
    ax_p.plot(NEB_CURVE_LOG_U, lines[ix, :, HB] / 10.0**log_qh[ix], color=f"C{iz}", lw=1.0, label=f"Z = {z:g}")
    ax_s.plot(NEB_CURVE_LOG_U, lines[ix, :, HB] / sfr10[ix], color=f"C{iz}", lw=1.0)
ax_p.axhline(CASE_B_ERG, color="0.4", ls=":", lw=1.0, label="case B")
ax_p.set_yscale("log")
ax_p.set_ylabel("H$\\beta$ per ionizing photon [erg]", fontsize=8)
ax_s.set_ylabel(r"H$\beta$ per $\mathrm{SFR}_{10}$ [erg/s / (M$_\odot$/yr)]", fontsize=8)
for ax in (ax_p, ax_s):
    ax.set_xlabel(r"$\log_{10} U_S$", fontsize=8)
ax_p.legend(fontsize=6, frameon=False)
fig.tight_layout()
save_fig("beagle_08c_per_photon.png")
plt.show()

print("§8c H-beta per ionizing photon [erg] and per SFR_10, 48 rows")
print(f"  {'quantity':>24s} {'BEAGLE median':>14s} {'min':>10s} {'max':>10s} {'tengri median':>14s} {'min':>10s} {'max':>10s} {'tengri/BEAGLE':>14s}")
for lab, b, t in (("per photon", PER_PHOTON_B, PER_PHOTON_T), ("per SFR_10", PER_SFR_B, PER_SFR_T)):
    print(f"  {lab:>24s} {np.median(b):14.4e} {b.min():10.3e} {b.max():10.3e} {np.median(t):14.4e} {t.min():10.3e} "
          f"{t.max():10.3e} {np.median(t / b):14.3f}")
print(f"  Case B per photon {CASE_B_ERG:.3e} erg; BEAGLE median per photon / case B {np.median(PER_PHOTON_B) / CASE_B_ERG:.3f}, "
      f"tengri median per photon / case B {np.median(PER_PHOTON_T) / CASE_B_ERG:.3f}")

# Per-SFR ratio = (ionizing-photon ratio) x (per-photon ratio), row by row; both SFRs are 10 M_sun/yr.
# tengri's stellar Z snaps to its nearest SSP node (logzsol_scatter 0.001); BEAGLE's grid has exact nodes.
print("§8c ionizing output by Z (1 Gyr history, 12 rows per Z): log10 photon rate, Q ratio = tengri/BEAGLE, "
      "per-photon and per-SFR ratios")
print(f"  {'Z':>6s} {'log Q tengri':>13s} {'log N BEAGLE':>13s} {'Q ratio':>8s} {'per photon':>11s} "
      f"{'per SFR':>8s} {'product':>8s}")
for z in NEB_Z:
    idx = [i for i, (_, zz, _) in enumerate(NEB_ROWS) if zz == z]
    q_t = float(np.median(T_LOG_QH[idx]))
    n_b = float(np.median(N_ION_B[idx]))
    q_ratio = 10.0 ** (q_t - n_b)
    pp = float(np.median(PER_PHOTON_T[idx] / PER_PHOTON_B[idx]))
    ps = float(np.median(PER_SFR_T[idx] / PER_SFR_B[idx]))
    print(f"  {z:6.3f} {q_t:13.3f} {n_b:13.3f} {q_ratio:8.3f} {pp:11.3f} {ps:8.3f} {pp * q_ratio:8.3f}")

# %% [markdown]
# ### §8d Continuum in line-free windows
#
# BEAGLE's continuum is the stellar plus nebular continuum without lines. tengri's SED includes its
# analytic continuum and lines; the windows sit between lines, and the table divides out the window mean
# at 4000 to 4050 A, so only the shape enters. The shape ratio runs from 0.93 to 1.19 across windows and
# metallicities: it rises toward the red at Z = 0.001 and falls toward the red at Z = 0.03. That is the
# library bracket between MILES (BEAGLE) and STELIB (tengri) at each metallicity, and the metallicity trend
# is not separated from it here. The analytic continuum changes the window means by 1 to 3 percent (last column).

# %%
CONT_WINDOWS_AA = ((3600.0, 3700.0), (4000.0, 4050.0), (5100.0, 5200.0), (6000.0, 6100.0), (7000.0, 7200.0))
wl_b = np.asarray(r_neb.wl_rest_aa, float)
SHAPE = {}
for z in NEB_Z:
    i_b = NEB_ROWS.index((-3.0, z, 0.3))
    flux_b = np.array([window_mean(wl_b, np.asarray(r_neb.continuum_sed[i_b], float), lo, hi) for lo, hi in CONT_WINDOWS_AA])
    sed_t = np.asarray(NEB_MODELS[z].predict({"neb_logU": -3.0, "neb_xi_d": 0.3}).rest_sed(wave=wl_b), float)
    flux_t = np.array([window_mean(wl_b, sed_t, lo, hi) for lo, hi in CONT_WINDOWS_AA])
    ratio = flux_b / flux_t
    SHAPE[z] = ratio / ratio[1]
neb_none = neb_model(0.014, continuum="none")
sed_a = np.asarray(NEB_MODELS[0.014].predict({"neb_logU": -3.0, "neb_xi_d": 0.3}).rest_sed(wave=wl_b), float)
sed_n = np.asarray(neb_none.predict({"neb_logU": -3.0, "neb_xi_d": 0.3}).rest_sed(wave=wl_b), float)
NONE_OVER_ANALYTIC = np.array([window_mean(wl_b, sed_n, lo, hi) / window_mean(wl_b, sed_a, lo, hi) for lo, hi in CONT_WINDOWS_AA])

fig, (ax_s, ax_n) = plt.subplots(1, 2, figsize=(9.5, 3.4))
centers = np.arange(len(CONT_WINDOWS_AA))
for iz, z in enumerate(NEB_Z):
    ax_s.plot(centers, SHAPE[z], "o-", ms=4, lw=0.9, color=f"C{iz}", label=f"Z = {z:g}")
ax_s.axhline(1.0, color="0.6", lw=0.8)
ax_s.set_ylabel("BEAGLE / tengri, shape", fontsize=8)
ax_n.plot(centers, NONE_OVER_ANALYTIC, "o-", ms=4, lw=0.9, color="C3")
ax_n.axhline(1.0, color="0.6", lw=0.8)
ax_n.set_ylabel("tengri continuum none / analytic", fontsize=8)
for ax in (ax_s, ax_n):
    ax.set_xticks(centers, [f"{lo / 1e3:.2f}-{hi / 1e3:.2f}" for lo, hi in CONT_WINDOWS_AA], fontsize=7)
    ax.set_xlabel(r"window [$\mu$m]", fontsize=8)
ax_s.legend(fontsize=6, frameon=False)
fig.tight_layout()
save_fig("beagle_08d_continuum.png")
plt.show()

print("§8d continuum window means at logU -3, xi_d 0.3; shape = ratio over the 4000-4050 A window")
print(f"  {'window [A]':>12s} " + " ".join(f"{'Z=' + format(z, 'g'):>10s}" for z in NEB_Z) + f" {'none/analytic':>14s}")
for k, (lo, hi) in enumerate(CONT_WINDOWS_AA):
    print(f"  {lo:6.0f}-{hi:5.0f} " + " ".join(f"{SHAPE[z][k]:10.3f}" for z in NEB_Z) + f" {NONE_OVER_ANALYTIC[k]:14.4f}")

# %% [markdown]
# ### §8 Notes on conventions
#
# The gutkin16 `logU` is the ionization parameter at the Stromgren radius, U_S, which is the table's own
# axis. Cue's `neb_logU` is the inner-face U at 10^19 cm, so no conversion is applied here. Both codes set
# the ionizing escape fraction to zero (BEAGLE `LyC_f_esc` and tengri `neb_fesc`). The public table has no
# Lyman-alpha and no continuum, so tengri's Lyman-alpha is the Case B value and its continuum is the analytic
# free-free plus two-photon term; the continuum is compared with BEAGLE in §8d. BEAGLE writes lines at air
# wavelengths above 2000 A, and the match above uses the vacuum wavelength in each key.

# %% [markdown]
# ### §8e Recent star formation
#
# The `s8b_episode` runs put a constant current episode on the 1 Gyr delayed history: the episode rate
# is log SFR in {-1, 0, 0.5, 1, 1.5} over 10 Myr, plus log SFR 0.5 over 31.6 and 100 Myr. The formed
# mass is 10^10 M☉ in both codes, and the stellar and gas metallicity are 0.014 with log U = -2.5 and
# xi_d = 0.3. tengri's `recent` episode is the same constant rate over `timescale_myr`. The equivalent
# width of Hα divides the line by the continuum L_λ averaged over the line-free windows 6400–6500 Å
# and 6605–6640 Å, so both codes use the same continuum estimate.

# %%
from tengri.utils.physics_constants import C_AA  # noqa: E402

TENGRI_LOG_ZSOL_NEB = float(np.log10(0.014 / Z_SOLAR_TENGRI))
EPISODE_AGE_GYR = 1.0  # BEAGLE `tau` and `max_stellar_age` are 9 (log yr): a 1 Gyr history
HA_I, HB_I, O3_I = GK_LABELS.index("Ha"), GK_LABELS.index("Hb"), GK_LABELS.index("[OIII]5007")


def episode_model(*, log_sfr, timescale_myr, dust=None, fesc=None):
    """tengri gutkin16 build: 1 Gyr delayed history plus a current episode, at BEAGLE's Z, log U and xi_d."""
    neb = {"type": "gutkin16", "logU": Fixed(-2.5), "logZ_gas": Fixed(TENGRI_LOG_ZSOL_NEB),
           "xi_d": Fixed(0.3), "log_nH": Fixed(2.0), "co_ratio": Fixed(1.0), "other_params": Fixed(DEFAULT)}
    if fesc is not None:
        neb["fesc"] = Fixed(fesc)  # the grammar short key of neb_fesc
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(TENGRI_LOG_ZSOL_NEB), "logzsol_scatter": Fixed(0.001), "all_params": Fixed(DEFAULT)},
        sfh={"type": ["delayed", "recent"], "tau_gyr": Fixed(EPISODE_AGE_GYR), "age_gyr": Fixed(EPISODE_AGE_GYR),
             "log_total_mass": Fixed(LOG_MASS), "log_sfr": Fixed(log_sfr), "timescale_myr": Fixed(timescale_myr),
             "other_params": Fixed(DEFAULT)},
        dust_attenuation=dust or {"type": "none"},
        neb=neb,
        redshift=Fixed(0.001),
    )


def tengri_gk_lines(pred):
    """gutkin16 line luminosities [erg/s] in the Gutkin+2016 table order, checked against its wavelengths."""
    waves = np.asarray(pred.lines.all_waves, float)[: len(GK_LABELS)]
    if np.max(np.abs(waves - GK_WAVE_AA)) > 5.0:
        raise ValueError("tengri line order no longer matches the Gutkin+2016 table")
    return np.asarray(pred.lines.all_lums, float)[: len(GK_LABELS)]


def continuum_lambda_ha(wave, l_nu):
    """Mean continuum L_λ [erg/s/Å] at H-alpha from the line-free windows 6400–6500 and 6605–6640 Å."""
    lnu = 0.5 * (window_mean(wave, l_nu, 6400.0, 6500.0) + window_mean(wave, l_nu, 6605.0, 6640.0))
    return lnu * C_AA / 6564.61**2


def binned_lnu(wave, l_nu, lo, hi, width=20.0):
    """Mean L_ν over ``width`` Å bins from ``lo`` to ``hi``: returns (bin centers [Å], binned L_ν)."""
    edges = np.arange(lo, hi + 0.5 * width, width)
    grid = np.arange(lo, hi, 1.0) + 0.5
    f = np.interp(grid, wave, l_nu)
    idx = np.digitize(grid, edges) - 1
    vals = np.array([f[idx == k].mean() for k in range(len(edges) - 1)])
    return 0.5 * (edges[:-1] + edges[1:]), vals


EP_ROWS = by_section("s8b_episode")[0].input_table()
EP_SFR = np.asarray(EP_ROWS["sfr"], float)
EP_WIN_MYR = 10.0 ** (np.asarray(EP_ROWS["current_sfr_timescale"], float) - 6.0)
r_ep = beagle("s8b_episode")
B_EP_LINES = beagle_lines(r_ep, LINE_WINDOWS_AA)
T_EP = []
for s, w_myr in zip(EP_SFR, EP_WIN_MYR):
    pred = episode_model(log_sfr=float(s), timescale_myr=float(w_myr)).predict({})
    T_EP.append((np.asarray(pred.wave_rest, float), np.asarray(pred.rest_sed(), float), tengri_gk_lines(pred)))

W_EP_B = np.asarray(r_ep.wl_rest_aa, float)
EW_B = np.array([
    B_EP_LINES[i, HA_I] / continuum_lambda_ha(W_EP_B, np.asarray(r_ep.continuum_sed[i], float))
    for i in range(len(EP_SFR))
])
EW_T = np.array([lines[HA_I] / continuum_lambda_ha(w, l) for w, l, lines in T_EP])
O3_B = B_EP_LINES[:, O3_I] / B_EP_LINES[:, HB_I]
O3_T = np.array([lines[O3_I] / lines[HB_I] for _, _, lines in T_EP])

# Constant-history reference at the same logU, Z and xi_d from §8a: isolates the episode history.
_ref = NEB_ROWS.index((-2.5, 0.014, 0.3))
print(f"§8e constant 1 Gyr reference (logU -2.5, Z 0.014, xi 0.3, §8a rows): [OIII]5007/H-beta BEAGLE "
      f"{RB[_ref, GK_LABELS.index('[OIII]5007')]:.3f}, tengri {RT[_ref, GK_LABELS.index('[OIII]5007')]:.3f}")
print("§8e episode rows: H-alpha equivalent width [Å] and [O III] 5007 / H-beta, BEAGLE / tengri")
print(f"  {'log SFR':>8s} {'window [Myr]':>12s} {'EW BEAGLE':>10s} {'EW tengri':>10s} {'T/B':>7s} "
      f"{'O3/Hb B':>8s} {'O3/Hb T':>8s} {'T-B':>8s}")
for i in range(len(EP_SFR)):
    print(f"  {EP_SFR[i]:8.2f} {EP_WIN_MYR[i]:12.1f} {EW_B[i]:10.1f} {EW_T[i]:10.1f} {EW_T[i] / EW_B[i]:7.3f} "
          f"{O3_B[i]:8.3f} {O3_T[i]:8.3f} {O3_T[i] - O3_B[i]:+8.3f}")

_i1 = int(np.where((EP_SFR == 1.0) & (EP_WIN_MYR == 10.0))[0][0])
_w_t, _l_t, _lines_t = T_EP[_i1]
_ha_ratio = _lines_t[HA_I] / B_EP_LINES[_i1, HA_I]
_cont_ratio = continuum_lambda_ha(_w_t, _l_t) / continuum_lambda_ha(W_EP_B, np.asarray(r_ep.continuum_sed[_i1], float))
print(f"§8e row log SFR 1, 10 Myr: H-alpha luminosity tengri/BEAGLE {_ha_ratio:.3f}, continuum L_λ at H-alpha "
      f"tengri/BEAGLE {_cont_ratio:.3f}, EW ratio {EW_T[_i1] / EW_B[_i1]:.3f} (= luminosity / continuum)")

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.5, 3.6))
ten = EP_WIN_MYR == 10.0
ax_a.plot(EP_SFR[ten], EW_B[ten], "o", color="C0", ms=4, label="BEAGLE")
ax_a.plot(EP_SFR[ten], EW_T[ten], "-s", color="C1", ms=3.5, lw=1.0, label="tengri")
ax_a.set_xlabel(r"episode $\log_{10}$ SFR [M$_\odot$/yr] over 10 Myr", fontsize=8)
ax_a.set_ylabel(r"H$\alpha$ EW [Å]", fontsize=8)
ax_a.set_yscale("log")
ax_a.legend(fontsize=7, frameon=False)
mid = (EP_SFR == 0.5)
order = np.argsort(EP_WIN_MYR[mid])
ax_b.plot(EP_WIN_MYR[mid][order], EW_B[mid][order], "o-", color="C0", lw=1.0, ms=4, label="BEAGLE EW")
ax_b.plot(EP_WIN_MYR[mid][order], EW_T[mid][order], "s-", color="C1", lw=1.0, ms=4, label="tengri EW")
ax_b.set_xscale("log")
ax_b.set_xlabel("episode window [Myr], log SFR 0.5", fontsize=8)
ax_b.set_ylabel(r"H$\alpha$ EW [Å]", fontsize=8)
ax_b.legend(fontsize=7, frameon=False)
fig.tight_layout()
save_fig("beagle_08e_episode.png")
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(9.5, 5.2), sharex="col", gridspec_kw={"height_ratios": (3.0, 1.0)})
for col, i in enumerate((3, 6)):
    wb, lb = W_EP_B, np.asarray(r_ep.sed_lnu[i], float)
    wt, lt, _ = T_EP[i]
    cen, bb = binned_lnu(wb, lb, 3600.0, 7000.0)
    _, bt = binned_lnu(wt, lt, 3600.0, 7000.0)
    axes[0, col].plot(cen, bb, color="C0", lw=1.0, label="BEAGLE")
    axes[0, col].plot(cen, bt, color="C1", lw=1.0, label="tengri")
    axes[0, col].set_yscale("log")
    axes[0, col].set_title(f"log SFR {EP_SFR[i]:g}, window {EP_WIN_MYR[i]:g} Myr", fontsize=8)
    axes[1, col].plot(cen, bt / bb, color="k", lw=0.8)
    axes[1, col].axhline(1.0, color="0.6", lw=0.8)
    axes[1, col].set_xlabel(r"rest wavelength [Å], 20 Å bins", fontsize=8)
axes[0, 0].legend(fontsize=7, frameon=False)
axes[0, 0].set_ylabel(r"$L_\nu$ [erg/s/Hz]", fontsize=8)
axes[1, 0].set_ylabel("tengri / BEAGLE", fontsize=8)
fig.tight_layout()
save_fig("beagle_08e_spectra.png")
plt.show()

# %% [markdown]
# The Hα EW ratio tengri/BEAGLE is 0.68 to 0.73 on the rows where BEAGLE's EW exceeds 10 Å. On the
# log SFR 1, 10 Myr row the Hα luminosity ratio is 0.718 and the continuum L_λ at Hα ratio is 1.027,
# so the EW gap sits in the line. The line ratio matches the per-SFR ratio of §8c (0.73): the ionizing output
# of CB16 (BEAGLE) against BC03 STELIB (tengri), a library bracket. On the log SFR -1 row BEAGLE's EW is 1.3 Å
# and tengri's is 1.5 Å; this row is at the continuum-dominated limit and is not classified beyond that.
#
# The [O III]/H-beta ratio does not depend on the episode rate in either code. BEAGLE's stays at 3.649 to
# 3.651, and tengri's is 3.327 on every row, the same value as its constant-history reference at the same
# log U, Z and xi_d (§8a). The -0.04 dex gap is therefore the one §8a measures for the constant history, not
# an episode effect; its source is not isolated here, and the §8a table is the record.

# %% [markdown]
# ### §8f Dust inside H II regions
#
# The `s8c_hiidust_on` and `s8c_hiidust_off` runs share the episode (log SFR 1 over 10 Myr) and CF00 depths
# τ_V,eff of 0.3, 1 and 2 with mu = 0.4. BEAGLE applies dust inside the H II regions only in the `on`
# run. tengri has no separate H II-region dust term: its two-component birth-cloud screen acts on the
# nebular light and the stellar light of ages below 10 Myr, with the power-law slopes -1.3 and -0.7 of
# §5. The table places tengri between the two BEAGLE runs with the fraction (tengri - off) / (on - off).

# %%
TAUV_HII = np.asarray(by_section("s8c_hiidust_on")[0].input_table()["tauV_eff"], float)
np.testing.assert_allclose(np.asarray(by_section("s8c_hiidust_off")[0].input_table()["tauV_eff"], float), TAUV_HII)
r_hon, r_hoff = beagle("s8c_hiidust_on"), beagle("s8c_hiidust_off")
BON = beagle_lines(r_hon, LINE_WINDOWS_AA, kind="_lum_obs")
BOFF = beagle_lines(r_hoff, LINE_WINDOWS_AA, kind="_lum_obs")
PRED_HII = [episode_model(log_sfr=1.0, timescale_myr=10.0, dust=cf00_dict(float(tv))).predict({}) for tv in TAUV_HII]
TLINES_HII = [tengri_gk_lines(p) for p in PRED_HII]

BAL_ON = BON[:, HA_I] / BON[:, HB_I]
BAL_OFF = BOFF[:, HA_I] / BOFF[:, HB_I]
BAL_T = np.array([t[HA_I] / t[HB_I] for t in TLINES_HII])
O3_ON = BON[:, O3_I] / BON[:, HB_I]
O3_OFF = BOFF[:, O3_I] / BOFF[:, HB_I]
O3_TH = np.array([t[O3_I] / t[HB_I] for t in TLINES_HII])

print("§8f H II-region dust: H-alpha / H-beta and [O III] 5007 / H-beta, observed (attenuated) lines")
print(f"  {'tauV':>5s} {'Ha/Hb on':>9s} {'Ha/Hb off':>10s} {'Ha/Hb T':>9s} {'frac':>7s} "
      f"{'O3/Hb on':>9s} {'O3/Hb off':>10s} {'O3/Hb T':>9s} {'frac':>7s}")
for k, tv in enumerate(TAUV_HII):
    frac_bal = (BAL_T[k] - BAL_OFF[k]) / (BAL_ON[k] - BAL_OFF[k])
    # The [O III]/H-beta on-off gap is below 1 %, so its fraction is not defined; the column says so.
    o3_gap = abs(O3_ON[k] - O3_OFF[k]) / O3_OFF[k]
    frac_o3 = f"{(O3_TH[k] - O3_OFF[k]) / (O3_ON[k] - O3_OFF[k]):7.3f}" if o3_gap > 0.01 else f"{'n/a':>7s}"
    print(f"  {tv:5.1f} {BAL_ON[k]:9.3f} {BAL_OFF[k]:10.3f} {BAL_T[k]:9.3f} {frac_bal:7.3f} "
          f"{O3_ON[k]:9.3f} {O3_OFF[k]:10.3f} {O3_TH[k]:9.3f} {frac_o3}")

TIDX = 1  # tauV_eff = 1 for the spectrum ratio
cen_h, b_on = binned_lnu(W_EP_B, np.asarray(r_hon.sed_lnu[TIDX], float), 3600.0, 7000.0)
_, b_off = binned_lnu(W_EP_B, np.asarray(r_hoff.sed_lnu[TIDX], float), 3600.0, 7000.0)
_, b_t = binned_lnu(np.asarray(PRED_HII[TIDX].wave_rest, float), np.asarray(PRED_HII[TIDX].rest_sed(), float), 3600.0, 7000.0)
fig, ax = plt.subplots(figsize=(8.0, 3.6))
ax.plot(cen_h, b_on / b_off, color="C0", lw=1.0, label="BEAGLE on / off")
ax.plot(cen_h, b_t / b_on, color="C1", lw=1.0, label="tengri / BEAGLE on")
ax.plot(cen_h, b_t / b_off, color="C2", lw=1.0, ls="--", label="tengri / BEAGLE off")
ax.axhline(1.0, color="0.6", lw=0.8)
ax.set_xlabel("rest wavelength [Å], 20 Å bins", fontsize=8)
ax.set_ylabel("spectrum ratio", fontsize=8)
ax.set_title(r"$\tau_{V,\rm eff}=1$, $\mu=0.4$", fontsize=9)
ax.legend(fontsize=7, frameon=False)
fig.tight_layout()
save_fig("beagle_08f_hii_dust.png")
plt.show()
print(f"§8f spectrum ratio at tauV_eff 1, 3600-7000 A (20 A bins): BEAGLE on/off median {np.median(b_on / b_off):.3f}, "
      f"tengri/BEAGLE on median {np.median(b_t / b_on):.3f}, tengri/BEAGLE off median {np.median(b_t / b_off):.3f}")

# BEAGLE's own switch for the H II part of the birth-cloud depth, read from its DUST ATTENUATION extension.
from astropy.io import fits  # noqa: E402


def beagle_dust_column(run, column):
    """One column of the DUST ATTENUATION extension in the run's results FITS, as float."""
    path = beagle_run.workdir() / "results" / run.name / f"{run.name}.fits"
    with fits.open(path) as hdul:
        return np.asarray(hdul["DUST ATTENUATION"].data[column], float)


for _label, _run in (("on", by_section("s8c_hiidust_on")[0]), ("off", by_section("s8c_hiidust_off")[0])):
    print(f"§8f BEAGLE DUST ATTENUATION, {_label} run: tauV_HII_max {np.unique(beagle_dust_column(_run, 'tauV_HII_max'))}, "
          f"tauV_HII_accounting {np.unique(beagle_dust_column(_run, 'tauV_HII_accounting'))}")

# %% [markdown]
# BEAGLE's on run has the lower Balmer decrement at every tauV_eff (3.11 against 3.26 at 0.3, 3.87 against 4.05
# at 1, 5.27 against 5.52 at 2). This is BEAGLE's documented convention and not a physical effect: since v0.27
# BEAGLE splits the birth-cloud optical depth into an H II part and an H I part. With `ACCOUNT FOR HII DUST = T`
# only the H I part is applied as an external screen to stars younger than 10 Myr and their lines, because the
# H II templates already carry their own dust through xi_d; with `F` the full birth-cloud depth is applied
# externally, so the off run puts more external dust on the lines. The mechanism is the convention as stated
# by the owner; the local BEAGLE tree does not document it, so it is not checked against BEAGLE source.
#
# tengri's Balmer decrement sits at the off end of that bracket: 3.262 against 3.260 at tauV 0.3, and 4.045 against
# 4.050 at 1. At 2 it lies 16 per cent of the way from off to on (5.484 against 5.523 and 5.273). tengri applies
# the full birth-cloud screen to the nebular light, which is consistent with matching the off run. The
# [O III]/H-beta on-off gap is under 1 %, so its fraction is not defined; tengri's ratio sits 0.04 dex below
# BEAGLE in both runs, the §8e gap. The spectrum ratio over 3600–7000 Å has BEAGLE on/off median 1.008 and
# tengri/BEAGLE median 0.994 against the off run and 0.986 against the on run.

# %% [markdown]
# ### §8g LyC escape
#
# The `s8d_fesc` runs set the escape fraction to 0, 0.1, 0.3 and 0.6 on the 10 Myr episode with no dust.
# BEAGLE's `LyC_f_esc` removes that fraction of the ionizing photons from the line emission. tengri's
# `neb_fesc` scales every nebular amplitude, lines and continuum, by one common factor, so the line
# ratios depend on f_esc only through that factor. The expected line ratio is 1 - f_esc for BEAGLE and
# (1 - f_esc) / (1 + 0.597 f_esc) for tengri, the CIGALE form with alpha_1 / alpha_B = 0.597. The continuum column compares the light between 800 and 900 Å,
# below the Lyman limit, relative to f_esc = 0.

# %%
FESC_VALS = np.asarray(by_section("s8d_fesc")[0].input_table()["LyC_f_esc"], float)
r_fe = beagle("s8d_fesc")
B_FE = beagle_lines(r_fe, LINE_WINDOWS_AA)
W_FE = np.asarray(r_fe.wl_rest_aa, float)
T_FE_PRED = [episode_model(log_sfr=1.0, timescale_myr=10.0, fesc=float(f)).predict({}) for f in FESC_VALS]
T_FE_LINES = np.array([tengri_gk_lines(p) for p in T_FE_PRED])
K0 = int(np.where(FESC_VALS == 0.0)[0][0])
HB_B = B_FE[:, HB_I] / B_FE[K0, HB_I]
HA_B = B_FE[:, HA_I] / B_FE[K0, HA_I]
HB_T = T_FE_LINES[:, HB_I] / T_FE_LINES[K0, HB_I]
HA_T = T_FE_LINES[:, HA_I] / T_FE_LINES[K0, HA_I]
CONT_B = np.array([window_mean(W_FE, np.asarray(r_fe.continuum_sed[i], float), 800.0, 900.0) for i in range(len(FESC_VALS))])
CONT_T = np.array([window_mean(np.asarray(p.wave_rest, float), np.asarray(p.rest_sed(), float), 800.0, 900.0) for p in T_FE_PRED])

print("§8g LyC escape: line ratios to f_esc = 0 and the continuum at 800-900 Å, BEAGLE / tengri")
print(f"  {'f_esc':>5s} {'Hb B':>7s} {'Hb T':>7s} {'Ha B':>7s} {'Ha T':>7s} {'1-f':>7s} {'T theory':>9s}")
for k, f in enumerate(FESC_VALS):
    t_theory = (1.0 - f) / (1.0 + 0.597 * f)
    print(f"  {f:5.2f} {HB_B[k]:7.3f} {HB_T[k]:7.3f} {HA_B[k]:7.3f} {HA_T[k]:7.3f} {1.0 - f:7.3f} {t_theory:9.3f}")
print(f"  {'f_esc':>5s} {'cont 800-900 A B [erg/s/Hz]':>28s} {'tengri':>12s} {'T/B':>8s}")
for k, f in enumerate(FESC_VALS):
    print(f"  {f:5.2f} {CONT_B[k]:28.4e} {CONT_T[k]:12.4e} {CONT_T[k] / CONT_B[k]:8.3f}")


def _linear_in_f(f, y):
    """Least-squares line y = a + b f; returns (b, a, max residual over the largest value)."""
    b, a = np.polyfit(f, y, 1)
    return b, a, float(np.max(np.abs(y - (a + b * f))) / float(np.max(np.abs(y))))


_slope_b, _icpt_b, _res_b = _linear_in_f(FESC_VALS, CONT_B)
_slope_t, _icpt_t, _res_t = _linear_in_f(FESC_VALS, CONT_T)
print("§8g 800-900 A mean L_nu, linear in f_esc: slope [erg/s/Hz per unit f], intercept [erg/s/Hz], "
      "max residual / largest value")
print(f"  BEAGLE  slope {_slope_b:.3e}  intercept {_icpt_b:.3e}  residual {_res_b:.1e}")
print(f"  tengri  slope {_slope_t:.3e}  intercept {_icpt_t:.3e}  residual {_res_t:.1e}")

fig, (ax_l, ax_c) = plt.subplots(1, 2, figsize=(9.5, 3.5))
ax_l.plot(FESC_VALS, HB_B, "o", color="C0", ms=4, label=r"BEAGLE H$\beta$")
ax_l.plot(FESC_VALS, HA_B, "^", color="C0", ms=4, mfc="none", label=r"BEAGLE H$\alpha$")
ax_l.plot(FESC_VALS, HB_T, "s", color="C1", ms=4, label=r"tengri H$\beta$")
ax_l.plot(FESC_VALS, HA_T, "v", color="C1", ms=4, mfc="none", label=r"tengri H$\alpha$")
fine = np.linspace(0.0, 0.6, 61)
ax_l.plot(fine, 1.0 - fine, "k--", lw=0.8, label=r"$1-f_{\rm esc}$")
ax_l.plot(fine, (1.0 - fine) / (1.0 + 0.597 * fine), "k:", lw=0.8, label="tengri form")
ax_l.set_xlabel(r"$f_{\rm esc}$", fontsize=8)
ax_l.set_ylabel("line / value at 0", fontsize=8)
ax_l.legend(fontsize=6, frameon=False)
ax_c.plot(FESC_VALS, CONT_B, "o", color="C0", ms=4, label="BEAGLE")
ax_c.plot(FESC_VALS, CONT_T, "s", color="C1", ms=4, label="tengri")
ax_c.set_yscale("log")
ax_c.set_xlabel(r"$f_{\rm esc}$", fontsize=8)
ax_c.set_ylabel(r"mean $L_\nu$ 800-900 Å [erg/s/Hz]", fontsize=8)
ax_c.legend(fontsize=7, frameon=False)
fig.tight_layout()
save_fig("beagle_08g_lyc_escape.png")
plt.show()

# %% [markdown]
# BEAGLE's H-beta and Hα fall exactly as 1 - f_esc (0.900, 0.700, 0.400). tengri's fall as
# (1 - f_esc) / (1 + 0.597 f_esc), and its printed values match that form to 0.001 at each escape fraction
# (0.849, 0.594, 0.295). The line ratios differ because the two codes remove photons differently: BEAGLE
# removes the ionizing photons from the lines, while tengri applies one factor that also scales the continuum,
# so its Balmer decrement does not change.
#
# The 800–900 Å continuum is linear in f_esc in both codes (the largest residual, relative to the largest value, is 2e-8 for BEAGLE and 7e-10 for tengri). BEAGLE's slope is
# 2.82e28 erg/s/Hz per unit f_esc, and its intercept 1.98e26 erg/s/Hz is the Lyman continuum of its older
# stars, which it does not reprocess: its picket fence acts on stars younger than 10 Myr only. tengri's slope is
# 1.76e28 and its intercept is 2.1e20, so it reprocesses the Lyman continuum of every stellar age. The escaping
# continuum ratio tengri/BEAGLE is 0.585, 0.612 and 0.619 at f_esc 0.1, 0.3 and 0.6, close to the slope ratio of
# 0.63 and to the 0.72 line ratio of §8e and the 0.73 per-SFR ratio of §8c. The convention is which ages' ionizing
# photons are reprocessed; the ionizing-output class is not isolated.

# %% [markdown]
# ### §8h Constant 100 Myr: the origin of the UV-line gaps
#
# The `s8e_const100` run holds a constant star formation rate for exactly 100 Myr, with no current episode, at
# the 48 nodes of §8a. The raw table column is the one §8a uses. tengri's history is the same constant rate from
# lookback 0.1 Gyr to the present, with the gutkin16 gas metallicity set equal to the stellar one. The first table
# gives, for each line, the median and the maximum of |log10| between the sources, in the §8a format. The second sets
# the four lines of the §8a gaps beside the same numbers for the 1 Gyr history.

# %%
H8_ONSET_GYR = 0.1  # constant history from lookback 0.1 Gyr to the present: BEAGLE max_stellar_age 8 (log yr)
H8_LINES = ("[OIII]5007", "HeII1640", "SiIII]1888", "CIII]1908")
r_c100 = beagle("s8e_const100")
tab_c100 = by_section("s8e_const100")[0].input_table()
np.testing.assert_allclose(np.asarray(tab_c100["nebular_logU"], float), [u for u, _, _ in NEB_ROWS])
np.testing.assert_allclose(np.asarray(tab_c100["nebular_xi"], float), [x for _, _, x in NEB_ROWS])
np.testing.assert_allclose(
    np.asarray(tab_c100["nebular_Z"], float), np.log10([z / beagle_driver.Z_SUN_BEAGLE for _, z, _ in NEB_ROWS])
)
CURVES_C100 = {z: tengri_curves(neb_model(z, onset_gyr=H8_ONSET_GYR)) for z in NEB_Z}
T100_LINES = np.array([tengri_at(z, u, x, curves=CURVES_C100)[0] for u, z, x in NEB_ROWS])
B100_LINES = beagle_lines(r_c100, LINE_WINDOWS_AA)
RB100, RT100 = (a / a[:, [HB]] for a in (B100_LINES, T100_LINES))

print("§8h constant 100 Myr: median and max |dlog10| of line / H-beta between sources, 48 rows")
print(f"  {'line':>12s} {'BEAGLE/tengri':>15s} {'max':>7s} {'raw/tengri':>12s} {'max':>7s} {'BEAGLE/raw':>12s} {'max':>7s}")
for j, lab in enumerate(GK_LABELS):
    if j == HB:
        continue
    d_bt = np.abs(np.log10(RB100[:, j] / RT100[:, j]))
    d_rt = np.abs(np.log10(RR[:, j] / RT100[:, j]))
    d_br = np.abs(np.log10(RB100[:, j] / RR[:, j]))
    print(f"  {lab:>12s} {np.median(d_bt):15.3f} {d_bt.max():7.3f} {np.median(d_rt):12.2e} {d_rt.max():7.1e} "
          f"{np.median(d_br):12.3f} {d_br.max():7.3f}")

print("§8h beside §8a (1 Gyr history), median / max |dlog10| over 48 rows")
print(f"  {'line':>12s} {'B/T 8a med':>11s} {'8h med':>8s} {'8a max':>8s} {'8h max':>8s} "
      f"{'B/raw 8a med':>13s} {'8h med':>8s} {'8a max':>8s} {'8h max':>8s}")
for lab in H8_LINES:
    j = GK_LABELS.index(lab)
    a_bt = np.abs(np.log10(RB[:, j] / RT[:, j]))
    h_bt = np.abs(np.log10(RB100[:, j] / RT100[:, j]))
    a_br = np.abs(np.log10(RB[:, j] / RR[:, j]))
    h_br = np.abs(np.log10(RB100[:, j] / RR[:, j]))
    print(f"  {lab:>12s} {np.median(a_bt):11.3f} {np.median(h_bt):8.3f} {a_bt.max():8.3f} {h_bt.max():8.3f} "
          f"{np.median(a_br):13.3f} {np.median(h_br):8.3f} {a_br.max():8.3f} {h_br.max():8.3f}")

PAIRS_H8 = (
    ("BEAGLE / tengri", (RB, RT), (RB100, RT100)),
    ("BEAGLE / raw table", (RB, RR), (RB100, RR)),
)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)
xs = np.arange(len(H8_LINES))
for ax, (title, pair_1g, pair_100) in zip(axes, PAIRS_H8):
    for (num, den), off, col, lab in ((pair_1g, -0.15, "0.6", "1 Gyr, §8a"), (pair_100, 0.15, "C0", "100 Myr, §8h")):
        for k, name in enumerate(H8_LINES):
            j = GK_LABELS.index(name)
            vals = np.log10(num[:, j] / den[:, j])
            ax.plot(np.full(len(vals), xs[k] + off), vals, ls="none", marker="o", ms=3.0, color=col, alpha=0.7,
                    label=lab if k == 0 else None)
    ax.axhline(0.0, color="0.6", lw=0.8)
    ax.set_xticks(xs, H8_LINES, rotation=30, fontsize=7)
    ax.set_title(title, fontsize=9)
axes[0].set_ylabel(r"$\log_{10}$ source ratio of line / H$\beta$", fontsize=8)
axes[0].legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout()
save_fig("beagle_08h_const100.png")
plt.show()

# %% [markdown]
# The He II 1640 gap persists at 100 Myr. Its median is 0.490 dex, the same as on the 1 Gyr history to 0.001 dex,
# and the C III] 1908 maximum is 0.537 dex on both. Si III] 1888 agrees with the table to 0.040 dex in the median
# on both histories. The tengri ratios at 100 Myr equal the raw table to 7.4e-15 dex at most, so tengri's gutkin16
# line ratios do not depend on the constant history, and the comparison reduces to BEAGLE's 100 Myr spectrum
# against the table. BEAGLE's four-line medians and maxima change by 0.001 dex or less between 100 Myr and 1 Gyr,
# so the age of a constant history is not the origin of the He II 1640 or C III] 1908 gaps. BEAGLE's line-to-H-beta
# ratios are the same on both histories, so this test shows only that the gaps are not from the history alone.
# Verdict: the He II 1640 and C III] 1908 gaps persist. The remaining difference is on BEAGLE's side, its nebular
# template set against the public table, which this notebook does not isolate. The [O III] 5007 residual, 0.021 dex
# in the median, is in the same class.

# %% [markdown]
# ### §8i Line evolution with age
#
# The `s8f_age` run holds a constant rate with no current episode and sets the maximum stellar age to 10^6.5,
# 10^7, 10^7.5, 10^8 and 10^9 yr, at logU -2.5, Z 0.014 and xi_d 0.3. tengri runs the same constant history at each
# age, from lookback 10^(log age - 9) Gyr to the present. The public table is time-integrated, so it has no age
# axis and §8i compares the two codes only with each other. The figure shows the two line ratios and the
# Hα equivalent width against age, and the table gives the values.

# %%
AGE_LOG_I = (6.5, 7.0, 7.5, 8.0, 9.0)  # BEAGLE max_stellar_age [log yr]
HE_I = GK_LABELS.index("HeII1640")
r_age = beagle("s8f_age")
tab_age = by_section("s8f_age")[0].input_table()
np.testing.assert_allclose(np.asarray(tab_age["max_stellar_age"], float), AGE_LOG_I)
np.testing.assert_allclose(np.asarray(tab_age["nebular_logU"], float), -2.5)
np.testing.assert_allclose(np.asarray(tab_age["nebular_xi"], float), 0.3)
B_AGE = beagle_lines(r_age, LINE_WINDOWS_AA)
W_AGE_B = np.asarray(r_age.wl_rest_aa, float)
T_AGE = []
for lg in AGE_LOG_I:
    pred = neb_model(0.014, onset_gyr=10.0 ** (lg - 9.0)).predict({"neb_logU": -2.5, "neb_xi_d": 0.3})
    T_AGE.append((np.asarray(pred.wave_rest, float), np.asarray(pred.rest_sed(), float), tengri_gk_lines(pred)))

EW_AGE_B = np.array([
    B_AGE[i, HA_I] / continuum_lambda_ha(W_AGE_B, np.asarray(r_age.continuum_sed[i], float))
    for i in range(len(AGE_LOG_I))
])
EW_AGE_T = np.array([lines[HA_I] / continuum_lambda_ha(w, l) for w, l, lines in T_AGE])
O3_AGE_B = B_AGE[:, O3_I] / B_AGE[:, HB_I]
O3_AGE_T = np.array([lines[O3_I] / lines[HB_I] for _, _, lines in T_AGE])
HE_AGE_B = B_AGE[:, HE_I] / B_AGE[:, HB_I]
HE_AGE_T = np.array([lines[HE_I] / lines[HB_I] for _, _, lines in T_AGE])

print("§8i constant history at logU -2.5, Z 0.014, xi_d 0.3: line ratios and H-alpha EW, BEAGLE / tengri")
print(f"  {'log age':>7s} {'age [Myr]':>10s} {'O3/Hb B':>8s} {'O3/Hb T':>8s} {'He2/Hb B':>9s} {'He2/Hb T':>9s} "
      f"{'EW B':>8s} {'EW T':>8s} {'T/B EW':>7s}")
for i, lg in enumerate(AGE_LOG_I):
    print(f"  {lg:7.1f} {10.0 ** (lg - 6.0):10.2f} {O3_AGE_B[i]:8.3f} {O3_AGE_T[i]:8.3f} {HE_AGE_B[i]:9.5f} "
          f"{HE_AGE_T[i]:9.5f} {EW_AGE_B[i]:8.1f} {EW_AGE_T[i]:8.1f} {EW_AGE_T[i] / EW_AGE_B[i]:7.3f}")

fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(9.5, 3.6))
ax_r.plot(AGE_LOG_I, O3_AGE_B, "o", color="C0", ms=4, label=r"BEAGLE [O III]/H$\beta$")
ax_r.plot(AGE_LOG_I, O3_AGE_T, "s-", color="C1", ms=4, lw=1.0, label=r"tengri [O III]/H$\beta$")
ax_r.plot(AGE_LOG_I, HE_AGE_B, "^", color="C0", ms=4, mfc="none", label=r"BEAGLE He II 1640/H$\beta$")
ax_r.plot(AGE_LOG_I, HE_AGE_T, "v--", color="C1", ms=4, mfc="none", lw=1.0, label=r"tengri He II 1640/H$\beta$")
ax_r.set_yscale("log")
ax_r.set_ylabel("line ratio", fontsize=8)
ax_r.legend(fontsize=6, frameon=False)
ax_e.plot(AGE_LOG_I, EW_AGE_B, "o", color="C0", ms=4, label="BEAGLE")
ax_e.plot(AGE_LOG_I, EW_AGE_T, "s-", color="C1", ms=4, lw=1.0, label="tengri")
ax_e.set_yscale("log")
ax_e.set_ylabel(r"H$\alpha$ EW [Å]", fontsize=8)
ax_e.legend(fontsize=7, frameon=False)
for ax in (ax_r, ax_e):
    ax.set_xlabel(r"maximum stellar age, $\log_{10}$ [yr]", fontsize=8)
fig.tight_layout()
save_fig("beagle_08i_age.png")
plt.show()

# %% [markdown]
# BEAGLE's [O III]/H-beta is 3.400 at 3.2 Myr, 3.651 at 10 Myr and 3.649 from 32 Myr on, and its He II 1640/H-beta
# is 0.00052, 0.00966 and 0.00966 on the same ages. tengri gives 3.327 and 0.00107 at every age, because its gutkin16
# ratios are the time-integrated table at the same logU, Z and xi_d, and do not change with the age of a constant
# history. The age dependence below 10 Myr is therefore model physics: BEAGLE's nebular templates are resolved in age
# and the public table is not, so tengri has no age dependence at all. At the saturated ages the He II 1640/H-beta
# ratio differs by 0.96 dex (a factor of 9) and the [O III]/H-beta ratio by 0.04 dex. The He II value lies inside the
# §8a He II 1640 spread (median 0.49 dex, maximum 1.03 dex) and is classed the same way, as model physics on BEAGLE's
# side, not isolated; the [O III] value is in the §8a class, not classified beyond that. The Hα EW ratio
# tengri/BEAGLE is 0.95 at 3.2 Myr, 0.79 at 10 Myr and 0.69 at 1 Gyr. From 10 Myr on it sits near the ionizing-output
# bracket of §8c (0.73 per SFR), a library bracket; its drift from 0.79 to 0.69 with age is not isolated here. At
# 3.2 Myr the EW gap of 5 per cent is not classified further.

# %% [markdown]
# ## 9. AGN narrow-line region
#
# BEAGLE-AGN (Vidal-Garcia et al. 2024) adds a narrow-line region (NLR) to the H II regions of a galaxy. The
# NLR is the Feltre, Charlot & Gutkin (2016) CLOUDY photoionization grid, and BEAGLE drives it with the
# accretion luminosity `agn_lacc` (log10 erg/s), the ionization parameter `agn_logU`, the metallicity `agn_Z`
# (log10 Z/Z☉, Z☉ 0.01524), `agn_xi` 0.3, n_H 10^3 cm^-3 and alpha -1.7. tengri's composable `nlr` block
# reads the same grid. Its disc and torus are left out, because BEAGLE adds no AGN continuum, so the AGN
# contributes lines only. The section has four parts: the grid check (9a), the NLR normalization and the
# AGN-only line ratios (9b and 9c), and the mixed H II plus NLR population (9d and 9e).

# %% [markdown]
# ### §9a Feltre grid check

# %% [markdown]
# The raw Feltre table is tabulated at L_acc = 10^45 erg/s. The check below evaluates tengri's backend at the
# grid nodes and compares it with the raw table. The BEAGLE rows in 9b to 9e fall between the nodes, so they
# test interpolation as well as normalization.

# %%
AGN_ALPHA = -1.7
AGN_XI = 0.3
AGN_LOG_N = 3.0
AGN_Z = (0.002, 0.014)
AGN_LOG_U = (-3.5, -2.5, -1.5)
AGN_LINES = ("[OII]3727", "[OIII]5007", "[NII]6584", "Halpha", "[SII]6717")
FELTRE = get_feltre_backend()


def tengri_nlr(z_abs, log_u, *, l_acc=1e45, xi=AGN_XI, log_n=AGN_LOG_N, alpha=AGN_ALPHA):
    """tengri NLR lines: wavelengths [A] and luminosities [erg/s] from the Feltre backend."""
    wave, lum_lsun = FELTRE.predict_agn_nlr_lines(
        alpha_pl=alpha,
        neb_logU=log_u,
        neb_logn=log_n,
        neb_logZ_gas=float(np.log10(z_abs)),  # absolute log10 Z
        xi_d=xi,
        log_qh=_log_qh_from_lacc(l_acc, alpha),
        neb_fesc=0.0,
    )
    return np.asarray(wave, float), np.asarray(lum_lsun, float) * L_SUN


def line_at(waves, lum, wave_aa):
    """Luminosity of the line nearest to ``wave_aa``; a miss beyond 1 A raises."""
    i = int(np.argmin(np.abs(waves - wave_aa)))
    if abs(waves[i] - wave_aa) > 1.0:
        raise KeyError(f"no tengri NLR line at {wave_aa} A")
    return float(lum[i])


def feltre_wave(name):
    return dict(beagle_driver.FELTRE_LINES)[name]


hb_wave = feltre_wave("Hbeta")
anchor_raw = beagle_driver.feltre_node(0.014, -2.5, 0.1, 100.0, -1.2)["Hbeta"]
w_a, l_a = tengri_nlr(0.014, -2.5, xi=0.1, log_n=2.0, alpha=-1.2)
anchor_t = line_at(w_a, l_a, hb_wave)
print(f"§9 anchor node (Z 0.014, logU -2.5, xi 0.1, n_H 100, alpha -1.2, L_acc 1e45 erg/s): "
      f"H-beta raw {anchor_raw:.4e}, tengri {anchor_t:.4e} erg/s, ratio {anchor_t / anchor_raw:.4f}")
w1, l1 = tengri_nlr(0.014, -2.5, l_acc=1e44)
w2, l2 = tengri_nlr(0.014, -2.5, l_acc=1e45)
print(f"§9 H-beta linear in L_acc: tengri(1e44)/tengri(1e45) = {line_at(w1, l1, hb_wave) / line_at(w2, l2, hb_wave):.4f} "
      f"(linear: 0.1000)")

print("§9 ratios to H-beta: raw Feltre table vs tengri at L_acc 1e45 erg/s, xi_d 0.3, n_H 1000, alpha -1.7")
print(f"  {'Z':>6s} {'logU':>6s} {'line':>12s} {'raw':>10s} {'tengri':>10s} {'dlog10':>8s}")
for z in AGN_Z:
    for u in AGN_LOG_U:
        raw = beagle_driver.feltre_node(z, u, AGN_XI, 10.0**AGN_LOG_N, AGN_ALPHA)
        w, lum = tengri_nlr(z, u)
        hb_t = line_at(w, lum, hb_wave)
        for name in AGN_LINES:
            r_ratio = raw[name] / raw["Hbeta"]
            t_ratio = line_at(w, lum, feltre_wave(name)) / hb_t
            print(f"  {z:6.3f} {u:6.1f} {name:>12s} {r_ratio:10.4f} {t_ratio:10.4f} {np.log10(t_ratio / r_ratio):8.4f}")

fine_u = np.linspace(-3.5, -1.5, 9)
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.5))
for ax, name in zip(axes, ("[OIII]5007", "[NII]6584")):
    for iz, z in enumerate(AGN_Z):
        t_curve = []
        for u in fine_u:
            w, lum = tengri_nlr(z, float(u))
            t_curve.append(line_at(w, lum, feltre_wave(name)) / line_at(w, lum, hb_wave))
        ax.plot(fine_u, t_curve, color=f"C{iz}", lw=1.2, label=f"tengri, Z = {z:g}")
        raw_pts = [beagle_driver.feltre_node(z, u, AGN_XI, 10.0**AGN_LOG_N, AGN_ALPHA) for u in AGN_LOG_U]
        ax.plot(AGN_LOG_U, [r[name] / r["Hbeta"] for r in raw_pts], "o", color=f"C{iz}", ms=4, label=f"raw table, Z = {z:g}")
    ax.set_yscale("log")
    ax.set_title(f"{name} / H$\\beta$", fontsize=9)
    ax.set_xlabel(r"$\log_{10} U_S$", fontsize=8)
axes[0].legend(fontsize=6, frameon=False)
fig.tight_layout()
save_fig("beagle_09_nlr.png")
plt.show()


# %% [markdown]
# ### §9b NLR normalization against BEAGLE

# %% [markdown]
# BEAGLE's `agn_lacc` is taken as the tengri accretion luminosity directly: `agn_log_lbol` = `agn_lacc` - log10(L_sun)
# with `agn_nlr_cf` 1, so the Feltre grid is evaluated at L_acc = 10^`agn_lacc` erg/s. The column `ratio, cf 0.1` is
# the ratio if a 10 % covering fraction were applied on top of `agn_lacc`. The composable build at one node checks
# that the block's summed line luminosity equals the backend's. The last column is the raw grid at Z 0.014, which
# BEAGLE's Z does not sit on, for the same logU.

# %%
from types import SimpleNamespace  # noqa: E402

AGN_ONLY_LACC = 45.0  # BEAGLE agn_lacc of s9_agn_only, fixed in the matrix
MIX_RUN = beagle("s9_agn")
MIX_ROWS = by_section("s9_agn")[0].input_table()
ONLY_RUN = beagle("s9_agn_only")
ONLY_ROWS = by_section("s9_agn_only")[0].input_table()
ONLY_LOGU = np.asarray(ONLY_ROWS["agn_logU"], float)
ONLY_Z = np.asarray(ONLY_ROWS["agn_Z"], float)

# BEAGLE vacuum-wavelength windows for each AGN line; doublets are summed.
AGN_WIN = {
    "Hb": ((4860.0, 4864.0),), "Ha": ((6562.0, 6566.0),), "CIV": ((1547.0, 1553.0),),
    "HeII1640": ((1639.0, 1641.0),), "CIII]1908": ((1905.0, 1912.0),), "NeV3426": ((3424.0, 3428.0),),
    "OII3727": ((3725.0, 3731.0),), "HeII4686": ((4685.0, 4688.0),), "OIII5007": ((5005.0, 5009.0),),
    "NII6584": ((6582.0, 6586.0),), "SII": ((6714.0, 6734.0),),
}
# tengri Feltre line wavelengths [Angstrom, vacuum, as the grid returns them] for each name; doublets are
# summed. Empty where the Feltre table has no such line. The header names differ from these for [C III] 1910.
TENGRI_NLR_WAVES = {
    "Hb": (4862.69,), "Ha": (6564.61,), "CIV": (1548.2, 1550.77), "HeII1640": (1640.4,),
    "CIII]1908": (1906.68, 1908.73), "NeV3426": (), "OII3727": (3727.09,), "HeII4686": (),
    "OIII5007": (5008.24,), "NII6584": (6585.27,), "SII": (6718.29, 6732.68),
}


def z_abs_of(zrel):
    """Absolute gas metallicity Z for BEAGLE agn_Z (log10 Z/Z_sun, Z_sun 0.01524)."""
    return beagle_driver.Z_SUN_BEAGLE * 10.0**zrel


def beagle_agn(run, name):
    """BEAGLE AGN line luminosity [erg/s] for one AGN_WIN name, shape (rows,)."""
    return beagle_lines(SimpleNamespace(line_lums_lsun=run.agn_line_lums_lsun), (AGN_WIN[name],))[:, 0]


def agn_log_lbol(lacc):
    """tengri agn_log_lbol [log10 L_sun] for BEAGLE agn_lacc [log10 erg/s]."""
    return lacc - np.log10(L_SUN)


def tengri_nlr_lines(z_abs, log_u, lacc, *, xi=AGN_XI, log_n=AGN_LOG_N, alpha=AGN_ALPHA):
    """tengri NLR line luminosities [erg/s] by AGN_WIN name, from the backend at L_acc 10^lacc erg/s."""
    w, lum = tengri_nlr(z_abs, log_u, l_acc=10.0**lacc, xi=xi, log_n=log_n, alpha=alpha)
    out = {}
    for name, waves in TENGRI_NLR_WAVES.items():
        out[name] = sum(line_at(w, lum, wave_aa) for wave_aa in waves) if waves else np.nan
    return out


def agn_mix_model(*, lacc=None, log_u=-2.5, zrel=0.0):
    """tengri build: the BEAGLE H II part (1 Gyr delayed history, 10 Myr episode at log SFR 1) with gutkin16,
    plus the Feltre NLR when ``lacc`` is given. The disc and torus are not built, so the AGN is lines only."""
    agn = None
    if lacc is not None:
        agn = {"type": "composable", "agn_log_lbol": Fixed(agn_log_lbol(lacc)), "all_params": Fixed(DEFAULT),
               "nlr": {"type": "feltre", "agn_nlr_cf": Fixed(1.0), "agn_nlr_alpha_pl": Fixed(AGN_ALPHA),
                       "agn_nlr_logn": Fixed(AGN_LOG_N), "agn_nlr_logU": Fixed(log_u),
                       "agn_nlr_logZ": Fixed(float(np.log10(z_abs_of(zrel)))), "agn_nlr_xi_d": Fixed(AGN_XI),
                       "agn_nlr_fwhm_kms": Fixed(500.0), "other_params": Fixed(DEFAULT)}}
    kwargs = {"agn": agn} if agn is not None else {}
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(TENGRI_LOG_ZSOL_NEB), "logzsol_scatter": Fixed(0.001), "all_params": Fixed(DEFAULT)},
        sfh={"type": ["delayed", "recent"], "tau_gyr": Fixed(EPISODE_AGE_GYR), "age_gyr": Fixed(EPISODE_AGE_GYR),
             "log_total_mass": Fixed(LOG_MASS), "log_sfr": Fixed(RECENT_LOG_SFR),
             "timescale_myr": Fixed(RECENT_MYR), "other_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        neb={"type": "gutkin16", "logU": Fixed(-2.5), "logZ_gas": Fixed(TENGRI_LOG_ZSOL_NEB),
             "xi_d": Fixed(0.3), "log_nH": Fixed(2.0), "co_ratio": Fixed(1.0), "other_params": Fixed(DEFAULT)},
        redshift=Fixed(0.001),
        **kwargs,
    )


_chk_lacc = AGN_ONLY_LACC
mix_check = agn_mix_model(lacc=_chk_lacc, log_u=-2.5, zrel=0.0)
pred_check = mix_check.predict({})
w_c, lum_c = tengri_nlr(z_abs_of(0.0), -2.5, l_acc=10.0**_chk_lacc)
composable_total = float(pred_check.properties["L_agn_lines_lsun"])
backend_total = float(np.sum(lum_c)) / L_SUN
print(f"§9b composable check (agn_lacc {_chk_lacc:g}, logU -2.5, Z 0.01524): summed NLR line luminosity "
      f"composable {composable_total:.6e} L_sun, backend {backend_total:.6e} L_sun, ratio {composable_total / backend_total:.8f}")
print(f"§9b agn_log_lbol for agn_lacc {AGN_ONLY_LACC:g}: {agn_log_lbol(AGN_ONLY_LACC):.4f}; L_sun {L_SUN:.4e} erg/s "
      f"(BEAGLE 3.826e33 erg/s)")

B_HB_ONLY = beagle_agn(ONLY_RUN, "Hb")
print("§9b AGN-only H-beta, BEAGLE vs tengri (erg/s)")
print(f"  {'logU':>6s} {'Z':>6s} {'BEAGLE':>11s} {'tengri':>11s} {'ratio':>8s} {'ratio, cf 0.1':>14s} "
      f"{'table Z 0.014':>14s}")
NORM_RATIO = []
for j, (u, zr) in enumerate(zip(ONLY_LOGU, ONLY_Z)):
    t_hb = tengri_nlr_lines(z_abs_of(zr), u, AGN_ONLY_LACC)["Hb"]
    raw_hb = beagle_driver.feltre_node(0.014, u, AGN_XI, 10.0**AGN_LOG_N, AGN_ALPHA)["Hbeta"]
    NORM_RATIO.append(t_hb / B_HB_ONLY[j])
    print(f"  {u:6.1f} {zr:6.1f} {B_HB_ONLY[j]:11.4e} {t_hb:11.4e} {t_hb / B_HB_ONLY[j]:8.4f} "
          f"{0.1 * t_hb / B_HB_ONLY[j]:14.4f} {raw_hb:14.4e}")

# %% [markdown]
# The two normalizations are not one constant factor apart. The ratio is 0.97 to 0.98 at logU -3.5 and rises
# with logU, reaching 1.82 at logU -1.5 and Z -0.5 and 2.74 at logU -1.5 and Z 0, the largest value. Tengri's H-beta falls
# by 0.17 dex across the logU range, as the raw grid does, while BEAGLE's falls by 0.62 dex at Z 0, so the difference
# lies in BEAGLE's AGN templates and not in the grid interpolation (§9a). BEAGLE scales its AGN templates by
# 10^(agn_lacc − 45) with no further covering factor: its H-beta equals its template interpolated in log Z to four
# decimals over all 30 rows, so `agn_nlr_cf` 1.0 is the matching setting here. BEAGLE's AGN templates
# (`nlr_*_v100_h1_r3`) match the public Feltre lines to 0.01 dex at logU -3.5, except C IV and N V (+0.5 to +1.2 dex
# at every logU), and they diverge toward high logU: the H-beta slope is 0.60 dex against 0.17 dex at Z 0.014. The
# class is template version (measured). Applying a 10 % covering fraction on top of `agn_lacc` would put every ratio
# at 0.10 to 0.27, so this comparison takes `agn_lacc` as the ionizing luminosity.

# %% [markdown]
# ### §9c AGN-only line ratios

# %% [markdown]
# Each line is divided by H-beta, and the table gives log10 of the ratio for BEAGLE and for tengri, and the
# difference, for the six (logU, Z) combinations. He II 4686 and [Ne V] 3426 are not in the Feltre table, so
# tengri has no value for them.

# %%
ONLY_LABELS = [f"{u:+.1f}/{zr:+.1f}" for u, zr in zip(ONLY_LOGU, ONLY_Z)]
T_ONLY = [tengri_nlr_lines(z_abs_of(zr), u, AGN_ONLY_LACC) for u, zr in zip(ONLY_LOGU, ONLY_Z)]
B_ONLY = {k: beagle_agn(ONLY_RUN, k) for k in AGN_WIN}
LINE_NAMES = [k for k in AGN_WIN if k != "Hb"]


def log_ratio_table(title, values):
    print(title)
    print(f"  {'line':>10s} " + " ".join(f"{lab:>11s}" for lab in ONLY_LABELS))
    for name in LINE_NAMES:
        cells = []
        for j in range(len(ONLY_LOGU)):
            v = values[name][j]
            cells.append(f"{'n/a':>11s}" if not np.isfinite(v) else f"{v:11.3f}")
        print(f"  {name:>10s} " + " ".join(cells))


B_LOG = {k: np.log10(B_ONLY[k] / B_ONLY["Hb"]) for k in LINE_NAMES}
T_LOG = {k: np.array([np.log10(t[k] / t["Hb"]) if np.isfinite(t[k]) else np.nan for t in T_ONLY])
         for k in LINE_NAMES}
D_LOG = {k: T_LOG[k] - B_LOG[k] for k in LINE_NAMES}
print("§9c combos are logU/Z (Z = log10 Z/Z_sun)")
log_ratio_table("§9c BEAGLE log10(line / H-beta)", B_LOG)
log_ratio_table("§9c tengri log10(line / H-beta)", T_LOG)
log_ratio_table("§9c difference tengri - BEAGLE (dex)", D_LOG)

fig, ax = plt.subplots(figsize=(4.6, 4.0))
for j, lab in enumerate(ONLY_LABELS):
    xs = [B_LOG[k][j] for k in LINE_NAMES if np.isfinite(T_LOG[k][j])]
    ys = [T_LOG[k][j] for k in LINE_NAMES if np.isfinite(T_LOG[k][j])]
    ax.plot(xs, ys, "o", ms=4, color=f"C{j}", label=f"logU/Z {lab}")
lim = [-2.5, 1.5]
ax.plot(lim, lim, "k:", lw=0.8)
ax.set_xlabel(r"BEAGLE $\log_{10}$(line / H$\beta$)", fontsize=8)
ax.set_ylabel(r"tengri $\log_{10}$(line / H$\beta$)", fontsize=8)
ax.legend(fontsize=6, frameon=False)
fig.tight_layout()
save_fig("beagle_09_agn_lines.png")
plt.show()

# %% [markdown]
# Hα agrees to 0.01 dex in all six combinations, [N II] to 0.04 dex and [O III]/H-beta to 0.05 dex. The [O II] and
# [S II] differences reach 0.17 and 0.60 dex at logU -1.5 and Z -0.5, C IV reaches -1.12 dex at logU -1.5 and Z 0, and
# He II 1640 and C III] reach 0.32 and 0.20 dex. C IV and N V are where BEAGLE's templates differ most: measured
# against the public Feltre lines, BEAGLE's C IV and N V lie +0.5 to +1.2 dex high at every logU, so the C IV residual
# is in the template class. The [O II] and [S II] differences are classed the same way, as a template-version
# candidate and not isolated here. He II 4686 and [Ne V] 3426 are not in the Feltre table tengri reads, which holds
# 20 lines, while BEAGLE's AGN output lists them: a line-set coverage difference.

# %% [markdown]
# ### §9d Mixed H II and NLR: BPT diagram

# %% [markdown]
# The 24 mixed rows add the NLR to the same H II regions. Each line is the H II value plus the AGN value, so the
# H II part is checked first: it is identical in every row, and it sets the H II points of the diagram. Demarcation
# curves are Kauffmann et al. (2003) and Kewley et al. (2001).

# %%
BPT_NAMES = ("Ha", "Hb", "NII6584", "OIII5007")
BPT_WINDOWS = tuple(AGN_WIN[n] for n in BPT_NAMES)
TG_HII_IDX = {"Ha": GK_LABELS.index("Ha"), "Hb": GK_LABELS.index("Hb"),
              "NII6584": GK_LABELS.index("[NII]6584"), "OIII5007": GK_LABELS.index("[OIII]5007")}

B_HII = beagle_lines(MIX_RUN, BPT_WINDOWS)
B_AGN_PART = beagle_lines(SimpleNamespace(line_lums_lsun=MIX_RUN.agn_line_lums_lsun), BPT_WINDOWS)
B_MIX = B_HII + B_AGN_PART
hii_spread = np.max(np.abs(B_HII[:, 1] / B_HII[0, 1] - 1.0))
print(f"§9d BEAGLE H II part of the 24 mixed rows: largest relative spread of H-beta across rows {hii_spread:.2e}")

tg_gk = tengri_gk_lines(pred_check)
T_HII = np.array([tg_gk[TG_HII_IDX[n]] for n in BPT_NAMES])
print("§9d tengri H II part (erg/s): " + ", ".join(f"{n} {T_HII[i]:.4e}" for i, n in enumerate(BPT_NAMES)))
print(f"§9d BEAGLE H II part (erg/s), row 0: " + ", ".join(f"{n} {B_HII[0, i]:.4e}" for i, n in enumerate(BPT_NAMES)))


def bpt_xy(ha, hb, nii, oiii):
    """log10([N II]/H-alpha) and log10([O III]/H-beta)."""
    return np.log10(nii / ha), np.log10(oiii / hb)


MIX_LACC = np.asarray(MIX_ROWS["agn_lacc"], float)
MIX_LOGU = np.asarray(MIX_ROWS["agn_logU"], float)
MIX_Z = np.asarray(MIX_ROWS["agn_Z"], float)
xb, yb = bpt_xy(B_MIX[:, 0], B_MIX[:, 1], B_MIX[:, 2], B_MIX[:, 3])
T_MIX = np.array([T_HII + np.array([tengri_nlr_lines(z_abs_of(zr), u, la)[n] for n in BPT_NAMES])
                  for la, u, zr in zip(MIX_LACC, MIX_LOGU, MIX_Z)])
xt, yt = bpt_xy(T_MIX[:, 0], T_MIX[:, 1], T_MIX[:, 2], T_MIX[:, 3])

ep_idx = np.flatnonzero(np.isclose(EP_SFR, 1.0) & np.isclose(EP_WIN_MYR, 10.0))[0]
B_PURE_HII = beagle_lines(beagle("s8b_episode"), BPT_WINDOWS)[ep_idx]
xb_h, yb_h = bpt_xy(*B_PURE_HII[[0, 1, 2, 3]])
xt_h, yt_h = bpt_xy(*T_HII)
B_ONLY_LINES = np.array([[B_ONLY["Ha"][j], B_ONLY["Hb"][j], B_ONLY["NII6584"][j], B_ONLY["OIII5007"][j]]
                         for j in range(len(ONLY_LOGU))])
xb_a, yb_a = bpt_xy(B_ONLY_LINES[:, 0], B_ONLY_LINES[:, 1], B_ONLY_LINES[:, 2], B_ONLY_LINES[:, 3])
T_ONLY_LINES = np.array([[t["Ha"], t["Hb"], t["NII6584"], t["OIII5007"]] for t in T_ONLY])
xt_a, yt_a = bpt_xy(T_ONLY_LINES[:, 0], T_ONLY_LINES[:, 1], T_ONLY_LINES[:, 2], T_ONLY_LINES[:, 3])

print("§9d mixed rows: agn_lacc, logU, Z; BEAGLE log([NII]/Ha), log([OIII]/Hb); tengri the same; differences")
print(f"  {'lacc':>5s} {'logU':>6s} {'Z':>6s} {'B x':>8s} {'T x':>8s} {'dx':>8s} {'B y':>8s} {'T y':>8s} {'dy':>8s}")
for i in range(len(MIX_LACC)):
    print(f"  {MIX_LACC[i]:5.1f} {MIX_LOGU[i]:6.1f} {MIX_Z[i]:6.1f} {xb[i]:8.3f} {xt[i]:8.3f} {xt[i] - xb[i]:8.3f} "
          f"{yb[i]:8.3f} {yt[i]:8.3f} {yt[i] - yb[i]:8.3f}")
print(f"§9d max |dx| {np.max(np.abs(xt - xb)):.3f}, max |dy| {np.max(np.abs(yt - yb)):.3f} dex over the 24 mixed rows")
print(f"§9d pure H II, BEAGLE (x, y) = ({xb_h:.3f}, {yb_h:.3f}); tengri = ({xt_h:.3f}, {yt_h:.3f})")
print("§9d pure AGN, BEAGLE (x, y) and tengri (x, y): " + "; ".join(
    f"logU/Z {lab}: B ({xb_a[j]:.3f}, {yb_a[j]:.3f}) T ({xt_a[j]:.3f}, {yt_a[j]:.3f})"
    for j, lab in enumerate(ONLY_LABELS)))

kx = np.linspace(-1.6, 0.0, 200)
kewley_x = np.linspace(-2.0, 0.3, 300)
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.plot(kx, 0.61 / (kx - 0.05) + 1.3, "k--", lw=0.8, label="Kauffmann+03")
ax.plot(kewley_x[kewley_x < 0.47], 0.61 / (kewley_x[kewley_x < 0.47] - 0.47) + 1.19, "k:", lw=0.8,
        label="Kewley+01")
for j in range(len(ONLY_LOGU)):
    ax.plot(xb_a[j], yb_a[j], "s", mfc="none", color=f"C{j}", ms=5)
    ax.plot(xt_a[j], yt_a[j], "s", color=f"C{j}", ms=3)
ax.plot(xb[:], yb[:], "o", mfc="none", ms=4, color="0.3", label="BEAGLE, mixed")
ax.plot(xt[:], yt[:], "o", ms=3, color="C3", label="tengri, mixed")
ax.plot(xb_h, yb_h, "*", mfc="none", ms=9, color="k", label="BEAGLE, H II")
ax.plot(xt_h, yt_h, "*", ms=7, color="k", label="tengri, H II")
ax.set_xlim(-1.6, 0.8)
ax.set_ylim(-1.0, 1.6)
ax.set_xlabel(r"$\log_{10}$([N II] 6584 / H$\alpha$)", fontsize=8)
ax.set_ylabel(r"$\log_{10}$([O III] 5007 / H$\beta$)", fontsize=8)
ax.legend(fontsize=6, frameon=False, loc="lower right")
fig.tight_layout()
save_fig("beagle_09_bpt.png")
plt.show()

# %% [markdown]
# The H II points agree to 0.04 dex in [O III]/H-beta, consistent with the §8 line comparison. The pure-AGN points agree
# to 0.05 dex in both axes for all six combinations, so the 9b normalization offset does not move a line ratio. The
# mixed rows differ by at most 0.16 dex in [N II]/Hα and 0.13 dex in [O III]/H-beta, at logU -1.5 where the 9b offset
# is largest (agn_lacc 44 and 45, Z 0 and -0.5). Mixed points depend on the NLR-to-H II H-beta ratio, so this difference is
# the 9b normalization offset carried into the mixture, together with the H II library bracket of §8c, which this section
# does not remove. It is classified as model physics of the NLR normalization.

# %% [markdown]
# ### §9e Spectrum of one mixed row

# %% [markdown]
# The row with agn_lacc 45, logU -2.5 and Z 0 is shown on BEAGLE's native rest-frame grid, binned to 20 Å in both
# codes. BEAGLE's `AGN FULL SED` is the AGN part of its `FULL SED`, checked point by point below. The tengri AGN share
# is the difference from the H II-only build. The H II ratio uses the BEAGLE row with a 10 Myr episode at SFR 1 (§8e).

# %%
sel = int(np.flatnonzero(np.isclose(MIX_LACC, 45.0) & np.isclose(MIX_LOGU, -2.5) & np.isclose(MIX_Z, 0.0))[0])
wl_b = np.asarray(MIX_RUN.wl_rest_aa, float)
lnu_b = np.asarray(MIX_RUN.sed_lnu[sel], float)
agn_b = np.asarray(MIX_RUN.agn_sed[sel], float)
lnu_t = np.asarray(pred_check.rest_sed(wl_b), float)
hii_t = np.asarray(agn_mix_model(lacc=None).predict({}).rest_sed(wl_b), float)
_ep_run = beagle("s8b_episode")
hii_b = np.asarray(_ep_run.sed_lnu[ep_idx], float)
wl_hii_b = np.asarray(_ep_run.wl_rest_aa, float)
print(f"§9e BEAGLE FULL SED minus AGN FULL SED is non-negative in {np.mean(lnu_b - agn_b >= 0) * 100:.1f} % of the grid points")

LO, HI = 1200.0, 7000.0
cw, vb = binned_lnu(wl_b, lnu_b, LO, HI, 20.0)
_, vt = binned_lnu(wl_b, lnu_t, LO, HI, 20.0)
_, va = binned_lnu(wl_b, agn_b, LO, HI, 20.0)
_, vh = binned_lnu(wl_b, hii_t, LO, HI, 20.0)
_, vhb = binned_lnu(wl_hii_b, hii_b, LO, HI, 20.0)
ratio = vt / vb
print("§9e binned L_nu ratio tengri/BEAGLE and AGN share of the total, by window (20 Å bins):")
print(f"  {'window':>14s} {'median T/B':>11s} {'max |log|':>10s} {'AGN share B':>12s} {'AGN share T':>12s} "
      f"{'H II T/B':>9s}")
for lo_w, hi_w in ((1200.0, 2000.0), (2000.0, 3500.0), (3500.0, 5500.0), (5500.0, 7000.0)):
    m = (cw >= lo_w) & (cw < hi_w)
    share_b = va[m].sum() / vb[m].sum()
    share_t = (vt[m] - vh[m]).sum() / vt[m].sum()
    print(f"  {lo_w:6.0f}-{hi_w:5.0f} {np.median(ratio[m]):11.3f} {np.max(np.abs(np.log10(ratio[m]))):10.3f} "
          f"{share_b:12.3f} {share_t:12.3f} {np.median(vh[m] / vhb[m]):9.3f}")
print("§9e BEAGLE AGN light per window: integrated AGN L_nu and the sum of its listed lines [erg/s]")
print(f"  {'window':>14s} {'int L_nu':>11s} {'lines':>11s} {'non-line':>11s} {'non-line share':>15s}")
for lo_w, hi_w in ((1200.0, 2000.0), (2000.0, 3500.0), (3500.0, 5500.0), (5500.0, 7000.0)):
    m = (wl_b >= lo_w) & (wl_b < hi_w)
    int_nu = float(np.trapezoid(agn_b[m] * C_AA / wl_b[m] ** 2, wl_b[m]))
    lines_sum = float(beagle_lines(SimpleNamespace(line_lums_lsun=MIX_RUN.agn_line_lums_lsun),
                                   (((lo_w, hi_w),),))[sel, 0])
    print(f"  {lo_w:6.0f}-{hi_w:5.0f} {int_nu:11.4e} {lines_sum:11.4e} {int_nu - lines_sum:11.4e} "
          f"{(int_nu - lines_sum) / int_nu:15.3f}")

fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
ax0.plot(cw, np.log10(vb), "k-", lw=1.0, label="BEAGLE, H II + NLR")
ax0.plot(cw, np.log10(vt), "C3-", lw=1.0, label="tengri, H II + NLR")
ax0.plot(cw, np.log10(va), "k:", lw=0.8, label="BEAGLE, NLR only")
ax0.set_ylabel(r"$\log_{10} L_\nu$ [erg/s/Hz]", fontsize=8)
ax0.legend(fontsize=6, frameon=False)
ax1.plot(cw, ratio, "C0-", lw=1.0)
ax1.axhline(1.0, color="0.5", lw=0.6)
ax1.set_ylabel("tengri / BEAGLE", fontsize=8)
ax1.set_xlabel(r"rest wavelength [$\mathrm{\AA}$]", fontsize=8)
fig.tight_layout()
save_fig("beagle_09_spectrum.png")
plt.show()

# %% [markdown]
# The optical windows agree. Above 3500 Å the median tengri/BEAGLE ratio is 0.99 to 1.01 and the AGN share agrees to
# within 0.02. Below 3500 Å the ratio is 1.17 and 1.09, and the H II-only ratio is 1.21 and 1.12, so this excess is in
# the H II part and not in the NLR. It is the stellar or nebular continuum; the library bracket and the nebular
# continuum are not separated here. The AGN share below 3500 Å differs: at 1200 to 2000 Å it follows the C IV and He II
# differences of 9c. At 2000 to 3500 Å, where tengri's Feltre table has no line, 40 % of the BEAGLE AGN light is outside
# its listed lines (printed above). Measured against BEAGLE's AGN templates, 27 % of that light is nebular continuum
# (template `lumin_cont`) and 13 % is an Fe II 2400 blend that BEAGLE does not list. tengri's NLR block returns lines
# only, so that continuum and blend are a model gap in tengri.

# %% [markdown]
# ## §12 IGM absorption
#
# BEAGLE applies the IGM to photometry only, so the transmission here is read from twelve top-hat
# bands (full width 2 per cent of the observed center, rest-frame centers 850 to 1216 Å) run with
# and without `IGM ABSORPTION = Inoue` at z = 1, 3 and 6: T = 10^(-0.4 (m_Inoue - m_none)). tengri's
# Inoue et al. (2014) transmission is integrated over the same top-hats with the photon weight and the
# no-IGM BEAGLE spectrum as the source shape. Away from the two bands where the top-hat straddles a step,
# the Lyman limit at 912 Å and Lyman-alpha at 1216 Å, the two codes agree to 0.0002 in transmission. The
# step bands average the step across their width of 18 to 24 Å (rest frame) and carry the largest
# differences: at z = 6 the Lyman-alpha band differs by +0.002 at native template resolution. With the
# BEAGLE templates rebinned to 10 Å (`REBIN TEMPLATES = 10`), BEAGLE's transmission in that band moves by
# -0.14 and the difference becomes +0.12. This is numerical, from BEAGLE's 10 Å sampling of the edge inside
# a 2 per cent band; the native-resolution value is the comparison and the rebinned pair stays on record.
# The figure shows the three redshifts and the table gives the values.

# %%
def igm_tophat_rows(zi, run_i, run_n):
    """Per top-hat at redshift ``zi``: (rest-frame center [A], BEAGLE transmission, tengri transmission)."""
    pi, pn = beagle_run.run(run_i), beagle_run.run(run_n)
    w_rest, L_n = np.asarray(pn.wl_rest_aa), np.asarray(pn.sed_lnu[0])
    rows = []
    for fs in run_i.filters:
        m_i = float(np.asarray(pi.magnitudes_app[fs.name])[0])
        m_n = float(np.asarray(pn.magnitudes_app[fs.name])[0])
        wl_o = np.asarray(fs.wl_aa, float)
        resp = np.asarray(fs.transmission, float)
        live = resp > 0
        wgrid = np.linspace(wl_o[live].min(), wl_o[live].max(), 401)
        weight = np.interp(wgrid, wl_o, resp) / wgrid * np.interp(wgrid / (1.0 + zi), w_rest, L_n)
        t_igm = beagle_driver.inoue14_transmission(wgrid, float(zi))
        cen = float(np.sum(wgrid * weight) / np.sum(weight)) / (1.0 + zi)
        t_b = 10.0 ** (-0.4 * (m_i - m_n))
        t_t = float(np.trapezoid(weight * t_igm, wgrid) / np.trapezoid(weight, wgrid))
        rows.append((cen, t_b, t_t))
    return rows


fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
igm_rows = []
for ax, zi in zip(axes, (1, 3, 6)):
    rows = igm_tophat_rows(zi, by_section(f"s12_igm_inoue_z{zi}")[0], by_section(f"s12_igm_none_z{zi}")[0])
    igm_rows.extend((zi, *r) for r in rows)
    cen, t_b, t_t = (np.array(v) for v in zip(*rows))
    ax.plot(cen, t_b, "o", color="C0", ms=4, label="BEAGLE")
    ax.plot(cen, t_t, "-", color="C1", lw=1.4, label="tengri")
    ax.axvline(911.8, color="0.6", ls=":", lw=0.8)
    ax.set_title(f"z = {zi}", fontsize=9)
    ax.set_xlabel(r"rest-frame center [Å]")
axes[0].set_ylabel("mean IGM transmission")
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
fig.tight_layout()
save_fig("beagle_06_igm.png")
plt.show()

print("§12 IGM transmission through 2 per cent top-hats: BEAGLE / tengri / difference")
print(f"  {'z':>2s} {'rest center [A]':>16s} {'BEAGLE':>9s} {'tengri':>9s} {'diff':>9s}")
for zi, c, tb, tt in igm_rows:
    print(f"  {zi:2d} {c:16.1f} {tb:9.4f} {tt:9.4f} {tt - tb:9.4f}")
_d = np.array([r[3] - r[2] for r in igm_rows])
_edge = np.array([abs(r[1] - 911.76) < 8.0 or abs(r[1] - 1215.67) < 8.0 for r in igm_rows])
print(f"§12 bands away from the Lyman limit and Lyman-alpha: largest |difference| {np.abs(_d[~_edge]).max():.4f}, "
      f"median {np.median(np.abs(_d[~_edge])):.4f}")
for zi in (1, 3, 6):
    sel = np.array([r[0] == zi for r in igm_rows]) & _edge
    print(f"§12 z = {zi}, bands straddling the Lyman limit and Lyman-alpha: differences "
          + ", ".join(f"{x:+.4f}" for x in _d[sel]))

# Lyman-alpha edge band at z = 6, at native template resolution and with REBIN TEMPLATES = 10.
LYA_REST_AA = 1215.67
for label, sec_i, sec_n in (
    ("native", "s12_igm_inoue_z6", "s12_igm_none_z6"),
    ("REBIN TEMPLATES = 10", "s12_igm_inoue_z6_rebin10", "s12_igm_none_z6_rebin10"),
):
    c, tb, tt = min(igm_tophat_rows(6, by_section(sec_i)[0], by_section(sec_n)[0]), key=lambda r: abs(r[0] - LYA_REST_AA))
    print(f"§12 z = 6 Lyman-alpha edge band (rest {c:.1f} A), {label}: BEAGLE {tb:.4f}, tengri {tt:.4f}, "
          f"diff {tt - tb:+.4f}")

# %% [markdown]
# ## 13. Where the codes differ
#
# Each row names one difference, the BEAGLE and tengri choice, its class, and the size this notebook
# measures or prints. A row marked "not run here" is a difference this notebook does not exercise.
#
# | difference | BEAGLE | tengri | class | size measured in this notebook |
# |---|---|---|---|---|
# | stellar template release | MILES-based `bc2003_hr_xmiless` | STELIB-based BC03 | library bracket (optical and NIR); UV template difference, not isolated (rest UV) | §1 optical and NIR window medians 0.96 to 1.07, rest-UV 1500–2500 Å median 0.72 at log age 9; §3 ratio 1.005 to 1.009 below 0.35 μm and 0.995 to 1.049 redward (tengri/BEAGLE) |
# | CB16 against BC03 inside BEAGLE | `cb_june16` | `bc03` (both BEAGLE) | library bracket | §1a window medians 0.84 to 1.20 |
# | nebular templates against the public table | BEAGLE nebular templates, per age | gutkin16 public table, time-integrated, with analytic continuum | library bracket (ionizing output); line set otherwise | §8c per-SFR_10 ratio 0.73 and per-photon 0.99 (medians), photon-rate ratio 0.80 at Z 0.004 (−0.10 dex, the clean number); §8a line medians 0.00 to 0.49 dex, He II 1640 the largest (not isolated here) |
# | continuum and Lyman-alpha | no continuum or Lyman-alpha in the public table | analytic free-free plus two-photon continuum; Case B Lyman-alpha | convention | §8d none/analytic window means 0.97 to 0.99; shape ratio 0.93 to 1.19 across windows and metallicities |
# | line wavelengths | air above 2000 Å | vacuum | convention | matched at vacuum in every key; not measured separately |
# | line luminosity units | L_sun in the HII EMISSION table, header unit erg/s | erg/s | convention | converted at ingestion with BEAGLE's L_sun; §8c uses the converted values |
# | ionization parameter | U_S at the Strömgren radius | Cue inner-face U at 10^19 cm | convention | not run here (Cue is not in this notebook); no conversion applied to gutkin16 |
# | c13.03 table interpolants against Cue | c13.03 table interpolated by BEAGLE | Cue network | numerical | not run here |
# | v0.27 tau_BC split (HII dust) | `ACCOUNT FOR HII DUST` | no HII-dust term in the §4 and §5 runs | convention | not exercised here |
# | template rebinning at the Lyman-alpha edge | native template sampling | `REBIN TEMPLATES = 10` on BEAGLE's side only | numerical | §12 z = 6 band: +0.002 native; with rebinning BEAGLE moves by -0.14 and the difference is +0.12 |
# | Calzetti 1994 against 2000 | `Calzetti` (1994) | `calzetti` (2000) | convention | §4 depth ratio 2000/1994: 1.11 at 1200–2200 Å, 1.05 at 2200–5500 Å, 1.00 at 5500–12000 Å |
# | Milky Way curve | BEAGLE's shipped Milky Way table | CCM89 (`cardelli`) | convention | §4 depth ratio tengri/BEAGLE: 1.06 at 1500 Å, 0.94 at 10000 Å, 1.21 at 20000 Å |
# | LMC and SMC curves | BEAGLE's shipped table: 0.62 to 0.81 (LMC) and 0.36 to 0.62 (SMC) of the Pei (1992) Drude sums over 1 to 2 μm | Pei (1992) Drude sums | convention (BEAGLE side) | §4 depth ratio at 10000 Å: LMC 1.24, SMC 1.62; at 20000 Å: LMC 1.60, SMC 2.79 |
# | CCWW13 full law | `ccww13_full` with the inclination law | not run | convention | not run here; the inclination law is not distributed with the public package |
# | cosmology | Planck 2015 | Planck 2018 (`Planck18`, Setup) | convention | luminosity distance at z = 1: 6791.81 against 6791.27 Mpc, ratio 1.00008 (code cell below) |
# | noise floor | sigma^2 += (0.02 y)^2 | `noise_frac_cal` | convention | not measured here |
# | solar luminosity | 3.826e33 erg/s | `L_SUN` in `tengri.utils.physics_constants` | convention | 3.8260e33 against 3.8280e33 erg/s, ratio 0.99948 (code cell below) |
# | surviving mass | `M_star` | `stellar_mass_surviving` | convention | §2 M_star ratio 0.93 at tau 0.1 Gyr, 0.92 at 1 Gyr, 0.93 at 10 Gyr; residual not separated here |
# | formed mass | `M_tot`, formed mass including the episode | formed mass set to the same value | convention | §2 M_tot ratio 1.000 at each tau |
# | specific SFR | `specific_sfr` relative to total formed mass | `recent_ssfr` episode | convention | not compared here |
# | formation time | `formation_redshift` | `age_gyr` (age at z) | convention | not compared here |
# | Hα EW per episode SFR | episode rate over 10 Myr, EW from BEAGLE line and continuum | episode rate over `timescale_myr`, same continuum window | library bracket (ionizing output) | §8e EW ratio 0.70 at log SFR 1 (0.68 to 0.73 on rows with BEAGLE EW above 10 Å); 1.16 at log SFR -1, not classified |
# | dust inside H II regions | `ACCOUNT FOR HII DUST` on and off (`tauV_HII_max` 0.135 on, 0 off) | no H II-region dust term; birth-cloud screen on all nebular light | convention | §8f Balmer-decrement position between off and on: -0.01, 0.03 and 0.16 at tauV 0.3, 1 and 2 |
# | LyC escape form | `LyC_f_esc` removes that fraction of the lines | `neb_fesc` scales lines and continuum by one factor, (1 − f) / (1 + 0.597 f) (the CIGALE form) | convention | §8g H-beta at f 0.1, 0.3, 0.6: tengri 0.849, 0.594, 0.295 against BEAGLE 0.900, 0.700, 0.400 |
# | ionizing photons below 912 Å at f_esc 0 | picket fence: only stars younger than 10 Myr reprocessed | all stellar ages reprocessed in the gutkin16 build without dust | convention | §8g mean L_ν at 800–900 Å: tengri 2.18e20 against BEAGLE 1.98e26 erg/s/Hz; escaping ratio 0.585 to 0.619 at f 0.1 to 0.6 |
# | BEAGLE nebular templates against the public table, UV lines | BEAGLE's updated templates (README: "an updated version of the stellar population + photoionization models presented in Gutkin et al., 2016") | the published 2016 grid, time-integrated | template version (BEAGLE's updated templates vs the published grid) | §8h at 100 Myr: He II 1640 median 0.490 dex, Si III] 1888 0.040 dex, C III] 1908 maximum 0.537 dex, the same as at 1 Gyr to 0.001 dex; tengri equals the table to 7.4e-15 dex |
# | line age dependence | age-resolved nebular templates | time-integrated table, no age axis | physics (BEAGLE side) | §8i [O III]/H-beta 3.400 at 3.2 Myr and 3.649 from 32 Myr on, tengri 3.327 at every age; EW ratio 0.95, 0.79 and 0.69 at 3.2 Myr, 10 Myr and 1 Gyr |
# | NLR normalization against logU | `agn_logU` drives its AGN templates (`nlr_*_v100_h1_r3`), which match the public Feltre lines to 0.01 dex except C IV and N V | Feltre grid at the same nodes (matches the raw grid to 1e-14 dex, §9a) | template version (measured) | §9b H-beta ratio tengri/BEAGLE 0.97 to 2.74 at cf 1.0 across logU -3.5 to -1.5; BEAGLE falls 0.62 dex across logU at Z 0 against tengri 0.17 dex |
# | AGN covering fraction | scales its templates by 10^(agn_lacc − 45) with no covering factor (H-beta equals its template to four decimals, 30 rows) | `agn_nlr_cf` 1.0, with `agn_lacc` taken as the ionizing luminosity | convention | §9b with `agn_nlr_cf` 0.1 the ratio would be 0.10 to 0.27, so the 10 % covering fraction is not applied |
# | AGN UV and low-ionization lines | BEAGLE's AGN templates (`v100` tags) | Feltre table lines only | template version (measured for C IV and N V; candidate for the rest) | §9c tengri minus BEAGLE at logU -1.5, Z 0: C IV -1.12, He II 1640 -0.32, C III] -0.20, [O II] 0.17, [S II] -0.60 dex; Hα 0.01, [N II] 0.04, [O III] 0.05 dex |
# | AGN line-set coverage | He II 4686 and [Ne V] 3426 in BEAGLE's AGN output | not in the Feltre table tengri reads (20 lines) | convention | §9c n/a for both in every combination |
# | H II continuum below 3500 Å | BEAGLE H II part with the same episode | gutkin16 H II part with BC03 STELIB | library bracket (nebular continuum not separated) | §9e tengri/BEAGLE median 1.17 at 1200–2000 Å and 1.09 at 2000–3500 Å, H II-only 1.21 and 1.12; 0.99 and 1.00 above 3500 Å |
# | AGN non-line light | BEAGLE's AGN output includes 27 % nebular continuum and 13 % an Fe II 2400 blend at 2000–3500 Å | NLR block returns lines only | model gap (tengri) | §9e non-line share of BEAGLE AGN light 0.04, 0.40, 0.03 and 0.06 in the four windows from 1200 to 7000 Å |
# | BEAGLE-AGN, normalization | Feltre NLR at `agn_lacc`, line output | composable `nlr` block on the same grid | convention | §9a–§9e run with the BEAGLE AGN column; disc and torus are not built |

# %%
from astropy.cosmology import Planck15

_dl15 = float(Planck15.luminosity_distance(1.0).to_value("Mpc"))
_dl18 = float(Planck18.luminosity_distance(1.0).to_value("Mpc"))
print(f"luminosity distance at z = 1: Planck 2015 {_dl15:.2f} Mpc, Planck 2018 {_dl18:.2f} Mpc, "
      f"ratio {_dl15 / _dl18:.5f}")
print(f"L_sun: BEAGLE {beagle_driver.L_SUN_BEAGLE:.4e} erg/s, tengri {L_SUN:.4e} erg/s, "
      f"ratio {beagle_driver.L_SUN_BEAGLE / L_SUN:.5f}")

# %% [markdown]
# ## 14. Parameter translation
#
# Each row maps a BEAGLE keyword to the tengri grammar key in `docs/model_configuration.md`. A row is
# left out when its tengri key is not on that page. BEAGLE's `mass` is the formed mass including the
# current episode. nebular_Z + 0.031 gives neb_logZ_gas, because Z☉ is 0.01524 in BEAGLE and 0.0142 in
# tengri. With `current_sfr_timescale = 8`, BEAGLE reports SFR_100 equal to the episode rate (a BEAGLE
# behavior, not re-measured in this notebook).
#
# | BEAGLE keyword | tengri grammar key | note |
# |---|---|---|
# | `sfr` (log ψ over the episode) | `sfh={'type': ['delayed', 'recent'], 'log_sfr': ...}` | episode rate; the configuration page documents `['delayed', 'recent']` |
# | `specific_sfr` (log ψ_S) | `sfh={'type': ['delayed', 'recent_ssfr'], 'log_ssfr': ...}` | BEAGLE normalizes to total formed mass; the tengri normalization is not restated here |
# | `current_sfr_timescale` (log yr) | `'timescale_myr'` | 10^7 yr is 10 Myr; the unit changes from log yr to Myr |
# | `tauV_eff`, `mu` | `dust_attenuation={'type': 'two_component', 'parameterization': 'tauv_mu', 'tau_v': ..., 'mu': ...}` | BEAGLE `mu` default 0.3, workshop run 0.4 |
# | `attenuation_type` calzetti | `'law': 'calzetti'` | tengri is the 2000 curve, BEAGLE the 1994 curve (§4) |
# | `attenuation_type` cf00 | `law_bc: 'power_law'`, `law_diff: 'power_law'` | the two-component pair is named together |
# | `attenuation_type` ccww13_universal | `law_bc: 'power_law'`, `law_diff: 'chevallard13'` | `chevallard13` is accepted as `law_diff` only |
# | `attenuation_type` mw | `'law': 'ccm89'` | BEAGLE's mw is its shipped table (§4) |
# | `nebular_logU` | `neb={'type': 'gutkin16', 'logU': Uniform(-4.0, -1.0)}` | the table extent is -4 to -1 |
# | `nebular_Z` | `neb={'type': 'gutkin16', 'logZ_gas': ...}` | log10(Z/Z☉) with Z☉ 0.0142; BEAGLE value + 0.031 |
# | `nebular_xi` | `'xi_d'` | the table nodes are 0.1, 0.3 and 0.5 |
# | `nebular_CO` | `'co_ratio'` | (C/O)/(C/O)_sun |
# | `IGM ABSORPTION = Inoue` | `igm={'type': 'inoue'}` | the configuration page names `'inoue'`, not `'inoue14'` |
# | `redshift` | `redshift=Fixed(z)` | a fixed redshift |
#
# Rows left out, because their tengri key is not on the configuration page: the delayed-history
# parameters (`tau`, `mass`, `max_stellar_age`; §2 and §3 pass `tau_gyr`, `age_gyr` and `log_total_mass`
# by name), `metallicity` (§2 and §3 pass `logzsol`), `LyC_f_esc` (`neb_fesc` is named only in the
# `lyc_reprocessed_by` context), the AGN keys `agn_*` (no `'feltre'` entry on the page), the `lsf_*` and
# `continuum_coeff` keys (`Spectroscopy(resolution=...)` and `calibration_order` are not on the page; its
# `Spectroscopy` options are the calibration marginal ones), `min_rel_err` (`noise_frac_cal` is not on the
# page), and `cosmology` (§13 prints the Planck difference instead). The `lmc` and `smc` attenuation laws
# are also left out, because the page does not list either as an attenuation law name.

# %% [markdown]
# ## 15. tengri in BEAGLE mode
#
# The four capstone galaxies are the Paper I objects 79 (red), 7837 (blue), 15775 (dusty) and 1826 (mid-infrared AGN host). BEAGLE fitted them on the CANDELS photometry of the workshop catalog, with the priors listed in the driver, and its posterior is read from the cache. tengri is run on the same photometry with the same model family: delayed-tau plus a recent episode set by the specific SFR, a CF00 two-component law with power-law slopes -1.3 and -0.7 and mu = 0.4, Gutkin et al. (2016) nebular emission with a free ionization parameter and the gas metallicity set equal to the stellar one, as in BEAGLE, and the Inoue IGM.
#
# The tengri side is evaluated at BEAGLE's posterior-median parameters and is not a tengri fit. A Laplace fit of one galaxy did not finish inside this notebook's budget, because the compiled likelihood takes minutes to build, so the comparison is between two models at the same parameter values. The tables and the figure are the record; the classification sentences below each table give the cause of each difference.

# %%
from reproduction.beagle._drivers.beagle_capstone import CAPSTONE_IDS, capstone_run
from tengri import Observation, Photometry
from tengri.utils.cosmology import age_at_z, luminosity_distance_mpc
from tengri.utils.physics_constants import C_AA, MPC_CM

CAPSTONE_LABEL = {79: "red", 7837: "blue", 15775: "dusty", 1826: "mid-infrared AGN host"}
CAPSTONE_RUN = capstone_run()
CAPSTONE = beagle_run.run(CAPSTONE_RUN)
CAPSTONE_CATALOG = beagle_driver.workshop_catalog(1)
AB_OFFSET = 48.6
BEAGLE_TO_TENGRI = {
    "CTIO_U": "ctio_u", "VIMOS_U": "vimos_u", "ACS_F435W": "hst_f435w", "ACS_F606W": "hst_f606w",
    "ACS_F775W": "hst_f775w", "ACS_F814W": "hst_f814w", "ACS_F850LP": "hst_f850lp",
    "WFC3_F098M": "hst_f098m", "WFC3_F105W": "hst_f105w", "WFC3_F125W": "hst_f125w",
    "WFC3_F160W": "hst_f160w", "ISAAC_KS": "vista_ks", "HAWKI_KS": "vista_ks",
    "IRAC_CH1": "irac_36", "IRAC_CH2": "irac_45", "IRAC_CH3": "irac_58", "IRAC_CH4": "irac_80",
}


def weighted_quantile(x, w, q):
    """Quantile q of x under normalized weights w: the first sample whose cumulative weight reaches q."""
    order = np.argsort(x)
    cum = np.cumsum(w[order])
    cum = cum / cum[-1]
    return float(x[order][min(np.searchsorted(cum, q), len(x) - 1)])


def beagle_quantiles(gid, ext, col, transform=lambda x: x):
    """Posterior 16, 50 and 84 % of one BEAGLE column, weighted by POSTERIOR PDF probability."""
    per = CAPSTONE.posterior[str(gid)]
    w = np.asarray(per["POSTERIOR PDF"]["probability"], float)
    w = w / w.sum()
    x = transform(np.asarray(per[ext][col], float))
    return tuple(weighted_quantile(x, w, q) for q in (0.16, 0.50, 0.84))


def beagle_medians(gid):
    """BEAGLE posterior-median parameters in tengri's conventions."""
    tau = beagle_quantiles(gid, "POSTERIOR PDF", "tau")[1]
    age = beagle_quantiles(gid, "POSTERIOR PDF", "max_stellar_age")[1]
    return {
        "tau_gyr": 10.0 ** (tau - 9.0),
        "age_frac": 10.0 ** (age - 9.0) / float(age_at_z(beagle_quantiles(gid, "POSTERIOR PDF", "redshift")[1])),
        "log_total_mass": beagle_quantiles(gid, "POSTERIOR PDF", "mass")[1],
        "met_logzsol": beagle_quantiles(gid, "POSTERIOR PDF", "metallicity")[1] + beagle_driver.LOG10_ZSUN_SHIFT,
        "log_ssfr": beagle_quantiles(gid, "POSTERIOR PDF", "specific_sfr")[1],
        "tau_v": beagle_quantiles(gid, "POSTERIOR PDF", "tauv_eff")[1],
        "log_u": beagle_quantiles(gid, "POSTERIOR PDF", "nebular_logu")[1],
        "z": beagle_quantiles(gid, "POSTERIOR PDF", "redshift")[1],
    }


def galaxy_bands(gid):
    """Measured bands of one galaxy as (BEAGLE names, tengri names, fluxes, errors)."""
    row = CAPSTONE_CATALOG[CAPSTONE_CATALOG["ID"] == gid][0]
    names, fnu, sig = beagle_driver.photometry_for_row(row, list(CAPSTONE_CATALOG.meta["bands"]))
    masked = np.ma.getmaskarray(fnu)
    ks = next((k for k in ("ISAAC_KS", "HAWKI_KS") if not masked[names.index(k)]), None)
    keep = [i for i, n in enumerate(names) if not masked[i] and n not in ("ISAAC_KS", "HAWKI_KS")]
    if ks is not None:
        keep.append(names.index(ks))
    keep = sorted(keep)
    return (
        [names[i] for i in keep],
        [BEAGLE_TO_TENGRI[names[i]] for i in keep],
        np.asarray(fnu.data)[keep],
        np.asarray(sig.data)[keep],
    )


def beagle_like_model(p, tengri_bands, *, dust=True):
    """tengri model at BEAGLE parameter values, in the notebook's dict grammar."""
    dust_block = (
        {"type": "two_component", "parameterization": "tauv_mu", "law_bc": "power_law",
         "slope_bc": Fixed(-1.3), "law_diff": "power_law", "slope_diff": Fixed(-0.7),
         "tau_v": Fixed(p["tau_v"]), "mu": Fixed(0.4), "other_params": Fixed(DEFAULT)}
        if dust else {"type": "none"}
    )
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(tengri_bands)),
        met={"logzsol": Fixed(p["met_logzsol"]), "logzsol_scatter": Fixed(0.001), "all_params": Fixed(DEFAULT)},
        sfh={"type": ["delayed", "recent_ssfr"], "all_params": Fixed(DEFAULT),
             "tau_gyr": Fixed(p["tau_gyr"]), "age_frac": Fixed(p["age_frac"]),
             "log_total_mass": Fixed(p["log_total_mass"]), "log_ssfr": Fixed(p["log_ssfr"]),
             "timescale_myr": Fixed(10.0)},
        dust_attenuation=dust_block,
        dust_emission={"type": "none"},
        neb={"type": "gutkin16", "logU": Fixed(p["log_u"]), "xi_d": Fixed(0.3), "log_nH": Fixed(2.0),
             "co_ratio": Fixed(1.0), "logZ_gas": Fixed(p["met_logzsol"]), "all_params": Fixed(DEFAULT)},
        igm={"type": "inoue"},
        redshift=Fixed(p["z"]),
    )

# %%
def tengri_at_beagle(gid):
    """tengri photometry, spectrum and properties at BEAGLE medians, and the dust-free A_V."""
    beagle_names, tengri_bands, fnu, sig = galaxy_bands(gid)
    p = beagle_medians(gid)
    model = beagle_like_model(p, tengri_bands)
    pred = model.predict({})
    dust_free = beagle_like_model(p, tengri_bands, dust=False).predict({})
    wave = np.asarray(pred.wave_rest)
    l_att = np.asarray(pred.rest_sed())
    l_free = np.asarray(dust_free.rest_sed())
    a_v = -2.5 * np.log10(window_mean(wave, l_att, 5450.0, 5550.0) / window_mean(wave, l_free, 5450.0, 5550.0))
    props = pred.properties
    return {
        "beagle_names": beagle_names, "tengri_bands": tengri_bands, "fnu": fnu, "sig": sig,
        "params": p, "pred": pred, "a_v": float(a_v),
        "log_mstar": float(np.log10(props["stellar_mass_surviving"])),
        "log_sfr10": float(np.log10(props["sfr_10myr"])),
        "tau_v": p["tau_v"],
        "photometry": np.asarray(pred.photometry()),
    }

# %%
RESULTS = {}
for gid in CAPSTONE_IDS:
    res = tengri_at_beagle(gid)
    RESULTS[gid] = res
    print(f"galaxy {gid} ({CAPSTONE_LABEL[gid]}), z = {res['params']['z']:.3f}, bands {len(res['tengri_bands'])}")
    print(f"  tengri photometry, first bands: {res['photometry'][:3]}")

# %%
def beagle_values(gid):
    return {
        "log M_star (surviving)": beagle_quantiles(gid, "GALAXY PROPERTIES", "M_star", np.log10),
        "log SFR_10 [M_sun/yr]": beagle_quantiles(gid, "STAR FORMATION", "SFR_10", np.log10),
        "tauV_eff": beagle_quantiles(gid, "DUST ATTENUATION", "tauV_eff"),
        "A_V [mag]": beagle_quantiles(gid, "DUST ATTENUATION", "A_V"),
    }


TABLE_ROWS = {}
for gid in CAPSTONE_IDS:
    res = RESULTS[gid]
    beagle = beagle_values(gid)
    tengri = {
        "log M_star (surviving)": res["log_mstar"],
        "log SFR_10 [M_sun/yr]": res["log_sfr10"],
        "tauV_eff": res["tau_v"],
        "A_V [mag]": res["a_v"],
    }
    TABLE_ROWS[gid] = {k: (beagle[k], tengri[k]) for k in beagle}
    print(f"galaxy {gid} ({CAPSTONE_LABEL[gid]}): BEAGLE posterior median and 16-84 % against tengri at the median")
    print(f"  {'quantity':<24s}{'BEAGLE':>9s}{'16 %':>9s}{'84 %':>9s}{'tengri':>9s}{'diff':>9s}")
    for label, (beagle_triple, t) in TABLE_ROWS[gid].items():
        lo, b, hi = beagle_triple  # 16, 50, 84 %
        print(f"  {label:<24s}{b:9.3f}{lo:9.3f}{hi:9.3f}{t:9.3f}{t - b:+9.3f}")


def implied_young_fraction(a_v, tau_v, mu=0.4):
    """Young-star share f with A_V = -2.5 log10[(1-f) e^(-mu tau) + f e^(-tau)], by bisection on [0, 1]."""
    target = 10.0 ** (-0.4 * a_v)
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if (1.0 - mid) * np.exp(-mu * tau_v) + mid * np.exp(-tau_v) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print("§15 young-star share f from A_V = -2.5 log10[(1-f) e^(-mu tau) + f e^(-tau)], mu 0.4, tau = tauV_eff median")
print(f"  {'galaxy':>8s} {'tauV_eff':>9s} {'tauV_HII_max':>13s} {'A_V BEAGLE':>11s} {'A_V tengri':>11s} "
      f"{'f BEAGLE':>9s} {'f tengri':>9s}")
for gid in CAPSTONE_IDS:
    tau_v = RESULTS[gid]["tau_v"]
    hii_max = beagle_quantiles(gid, "DUST ATTENUATION", "tauV_HII_max")[1]
    a_b = beagle_values(gid)["A_V [mag]"][1]
    a_t = RESULTS[gid]["a_v"]
    print(f"  {gid:8d} {tau_v:9.3f} {hii_max:13.3f} {a_b:11.3f} {a_t:11.3f} "
          f"{implied_young_fraction(a_b, tau_v):9.3f} {implied_young_fraction(a_t, tau_v):9.3f}")

# %% [markdown]
# Each table is BEAGLE's posterior median with its 16 and 84 % range, and tengri's value at that median, so the difference is the model difference at fixed parameters. Tying the gas metallicity to the stellar one leaves every printed value unchanged at the printed precision; the first-band photometry moves by under 0.01 %. Four classes follow, one per quantity.
#
# - log M_star (surviving): the differences of 0.01 to 0.02 dex are a library bracket. The stellar set and the surviving-mass fraction of the two codes differ, and §1 gives the template spread that this inherits; the size of its share is not isolated here.
# - log SFR_10: the differences of up to 0.02 dex are a library bracket too. Both codes use the same 10 Myr episode, so the residual comes from the template spread in §1 and not from the episode definition.
# - tauV_eff: the difference is zero because tengri's `tau_v` input is set to BEAGLE's `tauV_eff` at the median. The row checks that the parameter reaches the model, and carries no physics.
# - A_V: the differences of −0.28 to +0.10 mag come from the young-star dust. Both codes measure A_V over the 100 Å window at 5500 Å against the dust-free SED. For 1826 (BEAGLE median tauV_HII_max 0.83, §8f) the +0.10 mag is BEAGLE applying only the H I part of the birth-cloud depth externally, so BEAGLE's young stars carry the smaller share of the V light: implied young-star fraction 0.10 against 0.24. For 79, 7837 and 15775 tengri's young stars carry the smaller share: 0.68 against 0.62, 0.22 against 0.20 and 0.64 against 0.51 (BEAGLE against tengri). The cause of that smaller share is not isolated here.

# %%
def beagle_sed_observed(gid, z):
    """BEAGLE 16, 50 and 84 % observed-frame F_nu [erg/s/cm2/Hz] on the rest grid, with the observed wavelength [A].

    F_nu_obs = (1 + z) F_lambda(rest, at d_L) lambda_rest^2 / c, the rest-frame flux density at d_L
    re-expressed in the observed frame; the convention check below confirms it against the
    APPARENT MAGNITUDES that BEAGLE writes for the same posterior.
    """
    per = CAPSTONE.posterior[str(gid)]
    wl_rest = np.asarray(per["FULL SED"]["wl"], float)
    out = {}
    for q in ("p16", "p50", "p84"):
        out[q] = (1.0 + z) * np.asarray(per["FULL SED"][q], float) * wl_rest**2 / C_AA
    return wl_rest * (1.0 + z), out


def beagle_model_mags(gid, names):
    per = CAPSTONE.posterior[str(gid)]
    w = np.asarray(per["POSTERIOR PDF"]["probability"], float)
    w = w / w.sum()
    return np.array([weighted_quantile(np.asarray(per["APPARENT MAGNITUDES"][f"{n}_APP"], float), w, 0.5)
                     for n in names])


CANDELS_BY_CURVE = {name: (fw, ft) for name, fw, ft in CANDELS}
# Two catalog bands share a curve (ISAAC_KS is stored under the HAWKI_KS curve), so the band-to-curve map is the catalog's own.
CURVE_FOR_BAND = dict(CAPSTONE_CATALOG.meta["curve_for_band"])
CANDELS_CURVE = {band: CANDELS_BY_CURVE[curve] for band, curve in CURVE_FOR_BAND.items()}
CANDELS_PIVOT = {band: V.pivot_wavelength(fw, ft) for band, (fw, ft) in CANDELS_CURVE.items()}


def sed_convention_check(gid):
    """Median ratio of the BEAGLE median SED, band-averaged on the CANDELS curves, to its APPARENT MAGNITUDES."""
    res = RESULTS[gid]
    wl_obs, sed = beagle_sed_observed(gid, res["params"]["z"])
    mags = 10.0 ** (-0.4 * (beagle_model_mags(gid, res["beagle_names"]) + AB_OFFSET))
    ratios = []
    for name, m in zip(res["beagle_names"], mags):
        fw, ft = CANDELS_CURVE[name]
        ratios.append(V.band_average(wl_obs, sed["p50"], fw, ft) / m)
    return float(np.median(ratios))


print("BEAGLE SED convention check, median band-averaged SED over APPARENT MAGNITUDES:")
for gid in CAPSTONE_IDS:
    print(f"  {gid} ({CAPSTONE_LABEL[gid]}): {sed_convention_check(gid):.3f}")

fig = plt.figure(figsize=(10.5, 8.2))
outer = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
for k, gid in enumerate(CAPSTONE_IDS):
    res = RESULTS[gid]
    z = res["params"]["z"]
    inner = outer[k].subgridspec(2, 1, height_ratios=(3.0, 1.0), hspace=0.05)
    ax = fig.add_subplot(inner[0])
    axr = fig.add_subplot(inner[1], sharex=ax)
    wl_obs, sed = beagle_sed_observed(gid, z)
    ax.fill_between(wl_obs, sed["p16"], sed["p84"], color="C0", alpha=0.2, lw=0)
    ax.plot(wl_obs, sed["p50"], color="C0", lw=1.0, label="BEAGLE median and 16-84 %")
    wl_t = np.asarray(res["pred"].wave_obs)
    d_l_cm = float(luminosity_distance_mpc(z)) * MPC_CM
    f_t = np.asarray(res["pred"].obs_sed()) * (1.0 + z) / (4.0 * np.pi * d_l_cm**2)
    ax.plot(wl_t, f_t, color="C1", lw=1.0, label="tengri at the same parameters")
    pivots = np.array([CANDELS_PIVOT[n] for n in res["beagle_names"]])
    ax.errorbar(pivots, res["fnu"], yerr=res["sig"], fmt="o", color="k", ms=3, capsize=0, label="observed")
    ax.plot(pivots, 10.0 ** (-0.4 * (beagle_model_mags(gid, res["beagle_names"]) + AB_OFFSET)), "s", color="C0", ms=3, mfc="none")
    ax.plot(pivots, res["photometry"][: len(pivots)], "^", color="C1", ms=3, mfc="none")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_title(f"{gid} ({CAPSTONE_LABEL[gid]}), z = {z:.3f}", fontsize=9)
    ax.tick_params(labelbottom=False)
    if k == 0:
        ax.legend(fontsize=7, frameon=False)
    obs = np.asarray(res["fnu"], float)
    axr.axhline(0.0, color="0.6", lw=0.8)
    axr.plot(pivots, (10.0 ** (-0.4 * (beagle_model_mags(gid, res["beagle_names"]) + AB_OFFSET)) - obs) / obs, "s", color="C0", ms=3, mfc="none")
    axr.plot(pivots, (res["photometry"][: len(pivots)] - obs) / obs, "^", color="C1", ms=3, mfc="none")
    axr.set_xscale("log"); axr.set_ylim(-1.0, 1.0)
    axr.set_xlabel(r"observed wavelength [$\mu$m]" if k >= 2 else "")
    axr.set_ylabel("frac.", fontsize=7)
fig.suptitle("BEAGLE posterior median and tengri at the same parameters, CANDELS photometry", fontsize=10)
save_fig("beagle_07_capstone.png")
plt.show()

# %% [markdown]
# ## Summary
#
# | section | quantity | agreement measured | class |
# |---|---|---|---|
# | unit conversion | bolometric round trip | rel_err 3.1e-6 | numerical check |
# | §1 | SSP window medians, tengri/BEAGLE: optical and NIR; rest UV 1500–2500 Å at log age 9 | 0.96 to 1.07; 0.72 | library bracket; UV template difference (not isolated) |
# | §1a | CB16 against BC03 inside BEAGLE, window medians | 0.84 to 1.20 | library bracket |
# | §2 | SFR_10 and SFR_100, 10 Myr episode at tau 0.1 Gyr | 1.054 and 0.948 | numerical (tengri history grid) |
# | §2 | M_tot | 1.000 at each tau | agreement |
# | §3 | CANDELS pivots below 0.35 μm | 1.005 to 1.009 | agreement |
# | §3 | CANDELS pivots redward of 0.35 μm | 0.995 to 1.049 | library bracket |
# | §4 | Calzetti 2000 against 1994 depth | 1.00 to 1.11 | convention (curve) |
# | §4 | Milky Way depth, tengri/BEAGLE | 0.94 to 1.21 | convention (curve) |
# | §4 | LMC and SMC depth at 10000 Å | 1.24 and 1.62 | convention (BEAGLE side) |
# | §5 | CF00 attenuation A_1500 and A_V | up to 0.039 and 0.016 mag | inherits the §3 bracket |
# | §8a | Hα/H-beta, median and max | 0.000 and 0.002 dex | agreement |
# | §8a | [O III] 5007/H-beta, median and max | 0.021 and 0.238 dex | not classified (BEAGLE drive, inferred) |
# | §8a | He II 1640 and Si III] 1888, median | 0.49 and 0.04 dex | He II: consistent with BEAGLE's updated templates, not proven; Si III] 1888: agreement |
# | §8a | He II 1640 gap by Z, log10(BEAGLE/tengri), Z 0.001, 0.004, 0.014, 0.03 | −0.13, +0.04, +0.98, +0.87 dex | consistent with BEAGLE's updated templates (not proven) |
# | §8a | [O III] 5007 gap by Z, same order | −0.01, +0.00, +0.05, +0.13 dex | not classified (BEAGLE drive, inferred) |
# | §8c | H-beta per photon and per SFR_10, median | 0.99 and 0.73 | agreement (per photon); library bracket (per SFR) |
# | §8c | photon-rate ratio tengri/BEAGLE, Z 0.001, 0.004, 0.014, 0.03 | 0.78, 0.80, 0.64, 0.75 (−0.10 dex at 0.004) | library bracket |
# | §8d | continuum shape, window ratios | 0.93 to 1.19 | library bracket |
# | §8e | Hα EW ratio, 10 Myr episode at log SFR 1 | 0.70 (0.68 to 0.73 across rows above 10 Å) | library bracket (ionizing output) |
# | §8f | Balmer-decrement position between off and on, tauV 0.3, 1, 2 | -0.01, 0.03, 0.16 | convention |
# | §8g | H-beta at f_esc 0.1, 0.3, 0.6, tengri against BEAGLE | 0.849, 0.594, 0.295 against 0.900, 0.700, 0.400 | convention (escape form) |
# | §8g | mean L_ν at 800–900 Å, f_esc 0 | 2.18e20 against 1.98e26 erg/s/Hz | convention (picket fence, stars under 10 Myr) |
# | §8g | slope of 800–900 Å L_ν per unit f_esc, BEAGLE and tengri | 2.82e28 and 1.76e28 erg/s/Hz | convention (ages reprocessed) |
# | §8h | He II 1640 and Si III] 1888 medians at 100 Myr; C III] 1908 maximum | 0.490, 0.040 dex; 0.537 dex | template version (He II, C III]) |
# | §8i | [O III]/H-beta at saturated ages; EW ratio at 3.2 Myr, 10 Myr, 1 Gyr | 0.04 dex; 0.95, 0.79, 0.69 | physics (BEAGLE side) |
# | §9 | AGN anchor node, and H-beta linear in L_acc | 1.0000 and 0.1000 | normalization and wiring only |
# | §9b | AGN-only H-beta, tengri/BEAGLE at cf 1.0, logU -3.5 to -1.5 | 0.97 to 2.74 | template version (measured) |
# | §9b | the same ratio with `agn_nlr_cf` 0.1 | 0.10 to 0.27 | convention |
# | §9c | UV and low-ionization lines, tengri minus BEAGLE, logU -1.5 Z 0 | C IV -1.12, He II 1640 -0.32, C III] -0.20, [O II] 0.17, [S II] -0.60 dex | template version (measured for C IV and N V; candidate for the rest) |
# | §9c | Hα, [N II] and [O III] against BEAGLE, all six combinations | 0.01, 0.04, 0.05 dex | agreement |
# | §9e | tengri/BEAGLE median, 1200–2000 and 2000–3500 Å | 1.17 and 1.09 | library bracket (H II continuum) |
# | §9e | tengri/BEAGLE median, 3500–5500 and 5500–7000 Å; NLR share agreement | 0.99 and 1.00; within 0.02 | agreement |
# | §9e | BEAGLE non-line share of AGN light, 2000–3500 Å | 0.40 (27 % continuum, 13 % Fe II 2400) | model gap (tengri) |
# | §12 | IGM transmission, bands away from the steps | max 0.0002 | agreement |
# | §12 | Lyman-alpha band at z = 6, native and rebinned | +0.002 and +0.12 | numerical |
# | §13 | luminosity distance at z = 1, Planck 2015 against 2018 | ratio 1.00008 | convention |
# | §13 | L_sun, BEAGLE against tengri | ratio 0.99948 | convention |
# | §15 | log M_star (surviving), four capstone galaxies | -0.012 to -0.020 dex | library bracket |
# | §15 | log SFR_10, four capstone galaxies | -0.001 to +0.016 dex | library bracket |
# | §15 | tauV_eff, four capstone galaxies | 0.000 | parameter mapping (set by construction) |
# | §15 | A_V, four capstone galaxies | −0.28 to +0.10 mag | young-star dust share (1826: BEAGLE H I-only external screen; 79, 7837, 15775: smaller tengri young share) |
# | §15 | implied young-star fraction f, BEAGLE/tengri, 1826, 79, 7837, 15775 | 0.10/0.24, 0.68/0.62, 0.22/0.20, 0.64/0.51 | cause not isolated |
#
# On the stellar side the two codes agree in the rest-frame ultraviolet to 1 % and in normalization, age and
# star formation history at the level this notebook tests; redward of 0.35 μm they differ by up to 5 %, within the
# spread between BEAGLE's own two template sets in §1a. The attenuation-curve differences are conventions: the
# curve definitions of Calzetti, Milky Way and the Magellanic Clouds. The IGM agrees to 0.0002 away from the
# steps, and the one numerical difference, at the z = 6 Lyman-alpha edge, is set by BEAGLE's template sampling.
#
# On the nebular side the strong lines agree (Hα/H-beta to 0.002 dex, the other strong lines to 0.02 dex in the
# median), and the Hα equivalent width and the ionizing output both sit near 0.7 of BEAGLE's per unit star
# formation rate, a library bracket. The UV and low-ionization lines do not agree. He II 1640 and
# C III] 1908 keep their gaps on a constant 100 Myr history, which points to BEAGLE's updated nebular templates
# rather than to the age of the history. Si III] 1888 agrees to 0.04 dex. BEAGLE's nebular templates and the public 2016 grid differ, so this is a
# template-version class, not a tengri defect. The Lyman-continuum split by stellar age is a convention. The
# AGN normalization does not follow the same function of logU in the two codes, with a ratio from 0.97 to 2.74;
# the AGN line ratios match in Hα, [N II] and [O III] but not in C IV, He II or [O II].
#
# BEAGLE-AGN is run here with its line output, and §9 gives the AGN comparison in full. CCWW13 full and the noise
# model are not run in this notebook. §15 compares the two codes at BEAGLE's posterior medians for the four capstone
# galaxies, with the gas metallicity tied to the stellar one as in BEAGLE; its differences are the library bracket
# and the convention classes above.

# %% [markdown]
# ## References
#
# Entries 1 to 6 are cited as the BEAGLE Terms of Use require.
#
# 1. Chevallard J., Charlot S., 2016, "Modelling and interpreting spectral energy distributions of galaxies with BEAGLE", MNRAS 462, 1415. doi:`10.1093/mnras/stw1756`, arXiv:`1603.03037`.
# 2. Bruzual G., Charlot S., 2003, "Stellar population synthesis at the resolution of 2003", MNRAS 344, 1000. doi:`10.1046/j.1365-8711.2003.06897.x`, arXiv:`astro-ph/0309134`.
# 3. Gutkin J., Charlot S., Bruzual G., 2016, "Modelling the nebular emission from primeval to present-day star-forming galaxies", MNRAS 462, 1757. doi:`10.1093/mnras/stw1716`, arXiv:`1607.06086`.
# 4. Calzetti D., Kinney A. L., Storchi-Bergmann T., 1994, "Dust extinction of the stellar continua in starburst galaxies: The ultraviolet and optical extinction law", ApJ 429, 582. doi:`10.1086/174346`.
# 5. Charlot S., Fall S. M., 2000, "A Simple Model for the Absorption of Starlight by Dust in Galaxies", ApJ 539, 718. doi:`10.1086/309250`, arXiv:`astro-ph/0003128`.
# 6. Chevallard J., Charlot S., Wandelt B., Wild V., 2013, "Insights into the content and spatial distribution of dust from the integrated spectral properties of galaxies", MNRAS 432, 2061. doi:`10.1093/mnras/stt523`, arXiv:`1303.6631`.
# 7. Pei Y. C., 1992, "Interstellar dust from the Milky Way to the Magellanic Clouds", ApJ 395, 130. doi:`10.1086/171637`.
# 8. Inoue A. K., Shimizu I., Iwata I., Tanaka M., 2014, "An updated analytic model for attenuation by the intergalactic medium", MNRAS 442, 1805. doi:`10.1093/mnras/stu936`, arXiv:`1402.0677`.
# 9. Feltre A., Charlot S., Gutkin J., 2016, "Nuclear activity versus star formation: emission-line diagnostics at ultraviolet and optical wavelengths", MNRAS 456, 3354. doi:`10.1093/mnras/stv2794`, arXiv:`1511.08217`.
# 10. Vidal-García A., Plat A., Curtis-Lake E., Feltre A., Hirschmann M., Chevallard J., Charlot S., 2024, "BEAGLE-AGN I: Simultaneous constraints on the properties of gas in star-forming and AGN narrow-line regions in galaxies", MNRAS 527, 7217. doi:`10.1093/mnras/stad3252`, arXiv:`2211.13648`.
# 11. Plat A., Charlot S., Bruzual G., Feltre A., Vidal-García A., Morisset C., Chevallard J., Todt H., 2019, "Constraints on the production and escape of ionizing radiation from the emission-line spectra of metal-poor star-forming galaxies", MNRAS 490, 978. doi:`10.1093/mnras/stz2616`, arXiv:`1909.07386`.
# 12. Pacifici C. et al., 2023, "The Art of Measuring Physical Parameters in Galaxies: A Critical Assessment of Spectral Energy Distribution Fitting Techniques", ApJ 944, 141. doi:`10.3847/1538-4357/acacff`, arXiv:`2212.01915`.
# 13. Planck Collaboration, 2016, "Planck 2015 results. XIII. Cosmological parameters", A&A 594, A13. doi:`10.1051/0004-6361/201525830`, arXiv:`1502.01589`.
# 14. Ferland G. J. et al., 2017, "The 2017 Release of Cloudy", RMxAA 53, 385, arXiv:`1705.10877`.
# 15. Osterbrock D. E., Ferland G. J., 2006, *Astrophysics of Gaseous Nebulae and Active Galactic Nuclei*, 2nd ed., University Science Books.

# %%
print(f"total execution: {time.perf_counter() - _T0:.1f} s")
