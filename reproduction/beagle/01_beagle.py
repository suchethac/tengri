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

# BEAGLE's solar metallicity is 0.01524; tengri's is 0.0142. A BEAGLE log Z/Z_sun value maps to
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
# isochrones and IMF, not an identity. The figure shows single stellar populations of 10^10 M_sun
# at Z = 0.02 and log age 7, 8 and 9, and the tables give the ratio in four rest-frame windows.
# Window medians lie within 8 % in the optical and near-infrared at all three ages; the rest-UV
# at 1 Gyr is the largest difference (tengri 0.72 of BEAGLE), and BEAGLE's own `cb_june16` and
# `bc03` sets differ by as much there (§1a), so this is the library bracket.

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
# ## §2 Star formation history
#
# The `s2_sfh_delayed` run is a delayed history with tau of 0.1, 1 and 10 Gyr at z = 1, a formed mass
# of 10^10 M_sun and a current SFR of 10 M_sun/yr over the last 10 Myr. tengri builds the same
# delayed member with the `recent` episode on top, the formed mass set to BEAGLE's `mass`. The figure shows the star formation rate against lookback time, and the tables
# compare the 10 Myr episode, the smooth history and the integrated properties. BEAGLE's `M_tot` is
# the total formed mass including the episode and `M_star` the surviving mass.

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
# ## §4 Attenuation curves
#
# Each `s4_*` run applies one attenuation law at a V-band depth of 1 to the §3 history, so the
# effective depth is tau_lambda = -ln(F_dust / F_no-dust), with the dust-free `s3_fiducial` run as the
# reference. For the single-screen laws (Calzetti, Pei 1992 Milky Way, LMC and SMC) this is the
# curve itself. For CF00 and CCWW13 universal, the effective depth of the whole population lies
# between the two stellar groups: old stars see the diffuse ISM alone (mu = 0.4) and stars younger
# than 10 Myr see it plus the birth cloud. The two analytic bounds are drawn dashed, and the solid
# line is the tengri model's own ratio with the same history. The CCWW13 variant with the
# radiative-transfer tables (`ccww13_full`) is not run, because those tables are not distributed with
# BEAGLE. BEAGLE's `Calzetti` is the 1994 curve and tengri's `calzetti` is the 2000 curve, so that
# panel carries a curve difference; the printed table gives it.

# %%
MU = 0.4
SECTIONS_S4 = [r.section for r in RUN_MATRIX if r.section.startswith("s4_")]
LAWS = {
    "s4_cf00": ("CF00", None),
    "s4_calzetti": ("Calzetti (1994 | 2000)", calzetti),
    "s4_ccww13_universal": ("CCWW13 universal", None),
    "s4_mw": ("Milky Way (Pei 1992 | CCM89)", cardelli),
    "s4_lmc": ("LMC (Pei 1992)", lmc),
    "s4_smc": ("SMC (Pei 1992)", smc),
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
# (written in the next pass)

# %% [markdown]
# ## 9. AGN narrow-line region
#
# (written in the next pass)

# %% [markdown]
# ## §12 IGM absorption
#
# BEAGLE applies the IGM to photometry only, so the transmission here is read from twelve top-hat
# bands (full width 2 per cent of the observed center, rest-frame centers 850 to 1216 Å) run with
# and without `IGM ABSORPTION = Inoue` at z = 1, 3 and 6: T = 10^(-0.4 (m_Inoue - m_none)). tengri's
# Inoue et al. (2014) transmission is integrated over the same top-hats with the photon weight and the
# no-IGM BEAGLE spectrum as the source shape. The two bands where the top-hat straddles a step, the
# Lyman limit at 912 Å and Lyman-alpha at 1216 Å, average the step across their width of 18 to 24 Å
# (rest frame) and carry the largest differences; the other ten bands agree to 0.6 per cent. The figure shows
# the three redshifts and the table gives the values.

# %%
fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.6), sharey=True)
igm_rows = []
for ax, zi in zip(axes, (1, 3, 6)):
    run_i, run_n = by_section(f"s12_igm_inoue_z{zi}")[0], by_section(f"s12_igm_none_z{zi}")[0]
    pi, pn = beagle_run.run(run_i), beagle_run.run(run_n)
    w_rest, L_n = np.asarray(pn.wl_rest_aa), np.asarray(pn.sed_lnu[0])
    cen, t_b, t_t = [], [], []
    for fs in run_i.filters:
        m_i = float(np.asarray(pi.magnitudes_app[fs.name])[0])
        m_n = float(np.asarray(pn.magnitudes_app[fs.name])[0])
        wl_o = np.asarray(fs.wl_aa, float)
        resp = np.asarray(fs.transmission, float)
        live = resp > 0
        wgrid = np.linspace(wl_o[live].min(), wl_o[live].max(), 401)
        weight = np.interp(wgrid, wl_o, resp) / wgrid * np.interp(wgrid / (1.0 + zi), w_rest, L_n)
        t_igm = beagle_driver.inoue14_transmission(wgrid, float(zi))
        cen.append(float(np.sum(wgrid * weight) / np.sum(weight)) / (1.0 + zi))
        t_b.append(10.0 ** (-0.4 * (m_i - m_n)))
        t_t.append(float(np.trapezoid(weight * t_igm, wgrid) / np.trapezoid(weight, wgrid)))
        igm_rows.append((zi, cen[-1], t_b[-1], t_t[-1]))
    cen, t_b, t_t = np.array(cen), np.array(t_b), np.array(t_t)
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

# %% [markdown]
# ## 13. Where the codes differ
#
# (written in the next pass)

# %% [markdown]
# ## 14. Parameter translation
#
# (written in the next pass)

# %% [markdown]
# ## 15. tengri in BEAGLE mode
#
# (written in the next pass)

# %% [markdown]
# ## Summary
#
# (written in the next pass)

# %% [markdown]
# ## References
#
# (written in the next pass)

# %%
print(f"total execution: {time.perf_counter() - _T0:.1f} s")
