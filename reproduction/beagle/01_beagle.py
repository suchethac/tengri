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
# panel carries a curve difference; the printed table gives it. The Milky Way panel compares Pei (1992)
# on BEAGLE's side with Cardelli et al. (1989) on tengri's side. Redward of 1 micron the LMC and SMC
# depths of tengri exceed BEAGLE's by 24 % and 62 % at 10000 Å; at 20000 Å they are 1.6 and 2.8 times
# larger. tengri's `lmc` and `smc` are the Pei (1992) Drude-sum curves, so this is a difference on
# BEAGLE's side of the comparison (its measured LMC and SMC curves), a convention, not a tengri defect.

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
# The `s8_nebular` run is a constant star formation rate of 10 M_sun/yr over 1 Gyr at z = 0.001, with
# formed mass 10^10 M_sun and no dust. Its 48 rows vary log U_S from -3.5 to -2, Z from 0.001 to 0.03 and
# xi_d from 0.1 to 0.5; stellar and gas metallicity are equal, n_H is 100 cm^-3, C/O is solar and m_up is
# 100 M_sun. tengri runs the same history with the `gutkin16` nebular set. One build per metallicity
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
    ((1890.0, 1894.0),),  # Si III] 1888 (vacuum 1892)
    ((1904.0, 1910.5),),  # C III] 1908 doublet
)
assert len(LINE_WINDOWS_AA) == len(GK_LABELS)


def beagle_lines(run, windows):
    """BEAGLE line luminosities [erg/s], shape (rows, len(windows)); each column sums the keys in its windows.

    Keys carry the air label and the vacuum wavelength after the underscore, so the match uses vacuum.
    """
    keys = [k for k in run.line_lums_lsun if k.endswith("_lum_em")]
    vac = {k: float(k.split("@")[1].split("_")[1]) for k in keys}
    cols = []
    for groups in windows:
        sel = [k for k in keys if any(lo <= vac[k] <= hi for lo, hi in groups)]
        if not sel:
            raise KeyError(f"no BEAGLE line in {groups}")
        cols.append(sum(np.asarray(run.line_lums_lsun[k], float) for k in sel))
    return np.stack(cols, axis=1) * beagle_driver.L_SUN_BEAGLE


def neb_model(z_abs, *, continuum="analytic"):
    """tengri build at one absolute Z: stellar and gas metallicity equal, logU and xi_d free per prediction."""
    lz = float(np.log10(z_abs / Z_SOLAR_TENGRI))
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(lz), "logzsol_scatter": Fixed(0.001), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "const",
            "log_total_mass": Fixed(LOG_MASS),
            "start_gyr": Fixed(1.0),  # onset at lookback 1 Gyr
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


def tengri_at(z, u, x):
    """tengri lines [erg/s], log Q_H and SFR_10 at one row, read from the curve grid."""
    lines, log_qh, sfr10 = CURVES[z]
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
# spectrum and normalization), not by tengri's build; this is an inference from the agreement, not a separate test. H-alpha/H-beta agrees to 0.002 dex at most,
# [N II] and [S II] to 0.01 dex in the median, and [O III] to 0.02 dex in the median and 0.24 at most. He II
# 1640 and Si III] 1888 differ by 0.36 to 0.49 dex in the median and carry the largest spread. This section
# does not isolate their cause; §8c gives the one-number ionizing comparison.

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
# per-SFR column uses each source's own 10 Myr rate, which is 10 M_sun/yr in both. Per photon the two codes
# agree to 1.4 percent in the median, so the line conversion matches once the photons are counted. The
# per-SFR ratio (0.73 in the median) is therefore the ionizing photon output per unit SFR, which is the
# library bracket between CB16 (BEAGLE) and BC03 STELIB (tengri).

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
# ## 9. AGN narrow-line region
#
# The narrow-line region block is tengri's Feltre et al. (2016) photoionization grid; Vidal-Garcia et al.
# (2024) is cited for the narrow-line setup. tengri's ionizing rate comes from an accretion luminosity L_acc
# and a power-law slope alpha, and the raw table is tabulated at L_acc = 10^45 erg/s. There is no BEAGLE AGN
# run in this matrix, so this section has no BEAGLE column. Both sources are evaluated at grid nodes: the
# anchor ratio (1.0000) and the zero differences in the table confirm the normalization and the wiring, but
# they do not test interpolation between nodes.

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
# | stellar template release | MILES-based `bc2003_hr_xmiless` | STELIB-based BC03 | library bracket | §1 window medians 0.72 to 1.07; §3 ratio 1.005 to 1.009 below 0.35 µm and 0.995 to 1.049 redward (tengri/BEAGLE) |
# | CB16 against BC03 inside BEAGLE | `cb_june16` | `bc03` (both BEAGLE) | library bracket | §1a window medians 0.84 to 1.20 |
# | nebular templates against the public table | BEAGLE nebular templates, per age | gutkin16 public table, time-integrated, with analytic continuum | library bracket (ionizing output); line set otherwise | §8c H-beta per SFR_10 ratio 0.73 and per photon 0.99 (medians); §8a line medians 0.00 to 0.49 dex, He II 1640 the largest (not isolated here) |
# | continuum and Lyman-alpha | no continuum or Lyman-alpha in the public table | analytic free-free plus two-photon continuum; Case B Lyman-alpha | convention | §8d none/analytic window means 0.97 to 0.99; shape ratio 0.93 to 1.19 across windows and metallicities |
# | line wavelengths | air above 2000 Å | vacuum | convention | matched at vacuum in every key; not measured separately |
# | line luminosity units | L_sun in the HII EMISSION table, header unit erg/s | erg/s | convention | converted at ingestion with BEAGLE's L_sun; §8c uses the converted values |
# | ionization parameter | U_S at the Strömgren radius | Cue inner-face U at 10^19 cm | convention | not run here (Cue is not in this notebook); no conversion applied to gutkin16 |
# | c13.03 table interpolants against Cue | c13.03 table interpolated by BEAGLE | Cue network | numerical | not run here |
# | v0.27 tau_BC split (HII dust) | `ACCOUNT FOR HII DUST` | no HII-dust term in the §4 and §5 runs | convention | not exercised here |
# | template rebinning at the Lyman-alpha edge | native template sampling | `REBIN TEMPLATES = 10` on BEAGLE's side only | numerical | §12 z = 6 band: +0.002 native; with rebinning BEAGLE moves by -0.14 and the difference is +0.12 |
# | Calzetti 1994 against 2000 | `Calzetti` (1994) | `calzetti` (2000) | convention | §4 depth ratio 2000/1994: 1.11 at 1200-2200 Å, 1.05 at 2200-5500 Å, 1.00 at 5500-12000 Å |
# | Milky Way curve | Pei (1992) | CCM89 (`cardelli`) | convention | §4 depth ratio tengri/BEAGLE: 1.06 at 1500 Å, 0.94 at 10000 Å, 1.21 at 20000 Å |
# | LMC and SMC curves | BEAGLE's measured LMC and SMC curves | Pei (1992) Drude sums | convention (BEAGLE side) | §4 depth ratio at 10000 Å: LMC 1.24, SMC 1.62; at 20000 Å: LMC 1.60, SMC 2.79 |
# | CCWW13 full law | `ccww13_full` with the inclination law | not run | convention | not run here; the inclination law is not distributed with the public package |
# | cosmology | Planck 2015 | Planck 2018 (`Planck18`, Setup) | convention | luminosity distance at z = 1: 6791.81 against 6791.27 Mpc, ratio 1.00008 (code cell below) |
# | noise floor | sigma^2 += (0.02 y)^2 | `noise_frac_cal` | convention | not measured here |
# | solar luminosity | 3.826e33 erg/s | `L_SUN` in `tengri.utils.physics_constants` | convention | 3.8260e33 against 3.8280e33 erg/s, ratio 0.99948 (code cell below) |
# | surviving mass | `M_star` | `stellar_mass_surviving` | convention | §2 M_star ratio 0.93 at tau 0.1 Gyr, 0.92 at 1 Gyr, 0.93 at 10 Gyr; residual not separated here |
# | formed mass | `M_tot`, formed mass including the episode | formed mass set to the same value | convention | §2 M_tot ratio 1.000 at each tau |
# | specific SFR | `specific_sfr` relative to total formed mass | `recent_ssfr` episode | convention | not compared here |
# | formation time | `formation_redshift` | `age_gyr` (age at z) | convention | not compared here |
# | BEAGLE-AGN | not run | not run | convention | not run here; §9 has no BEAGLE column |

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
# current episode. nebular_Z + 0.031 gives neb_logZ_gas, because Z_sun is 0.01524 in BEAGLE and 0.0142 in
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
# | `attenuation_type` mw | `'law': 'ccm89'` | BEAGLE's mw is Pei (1992) (§4) |
# | `nebular_logU` | `neb={'type': 'gutkin16', 'logU': Uniform(-4.0, -1.0)}` | the table extent is -4 to -1 |
# | `nebular_Z` | `neb={'type': 'gutkin16', 'logZ_gas': ...}` | log10(Z/Z_sun) with Z_sun 0.0142; BEAGLE value + 0.031 |
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
# (written in the next pass)

# %% [markdown]
# ## Summary
#
# | section | quantity | agreement measured | class |
# |---|---|---|---|
# | unit conversion | bolometric round trip | rel_err 3.1e-6 | numerical check |
# | §1 | SSP window medians, tengri/BEAGLE | 0.72 to 1.07 | library bracket |
# | §1a | CB16 against BC03 inside BEAGLE, window medians | 0.84 to 1.20 | library bracket |
# | §2 | SFR_10 and SFR_100, 10 Myr episode at tau 0.1 Gyr | 1.054 and 0.948 | numerical (tengri history grid) |
# | §2 | M_tot | 1.000 at each tau | agreement |
# | §3 | CANDELS pivots below 0.35 µm | 1.005 to 1.009 | agreement |
# | §3 | CANDELS pivots redward of 0.35 µm | 0.995 to 1.049 | library bracket |
# | §4 | Calzetti 2000 against 1994 depth | 1.00 to 1.11 | convention (curve) |
# | §4 | Milky Way depth, tengri/BEAGLE | 0.94 to 1.21 | convention (curve) |
# | §4 | LMC and SMC depth at 10000 Å | 1.24 and 1.62 | convention (BEAGLE side) |
# | §5 | CF00 attenuation A_1500 and A_V | up to 0.039 and 0.016 mag | inherits the §3 bracket |
# | §8a | H-alpha/H-beta, median and max | 0.000 and 0.002 dex | agreement |
# | §8a | [O III] 5007/H-beta, median and max | 0.021 and 0.238 dex | not classified (BEAGLE drive, inferred) |
# | §8a | He II 1640 and Si III] 1888, median | 0.49 and 0.36 dex | not classified (not isolated) |
# | §8c | H-beta per photon and per SFR_10, median | 0.99 and 0.73 | agreement (per photon); library bracket (per SFR) |
# | §8d | continuum shape, window ratios | 0.93 to 1.19 | library bracket |
# | §9 | AGN anchor node, and H-beta linear in L_acc | 1.0000 and 0.1000 | normalization and wiring only |
# | §12 | IGM transmission, bands away from the steps | max 0.0002 | agreement |
# | §12 | Lyman-alpha band at z = 6, native and rebinned | +0.002 and +0.12 | numerical |
# | §13 | luminosity distance at z = 1, Planck 2015 against 2018 | ratio 1.00008 | convention |
# | §13 | L_sun, BEAGLE against tengri | ratio 0.99948 | convention |
#
# The rest-frame ultraviolet bands agree to 1 % and the normalization, age and star formation history
# match at the level this notebook tests. Redward of 0.35 µm the two codes differ by up to 5 %, within the
# spread between BEAGLE's own two template sets in §1a. The attenuation-curve differences are conventions:
# the curve definitions of Calzetti, Milky Way and the Magellanic Clouds. The strong nebular lines agree to
# 0.02 dex in the median and H-alpha/H-beta to 0.002 dex, while He II 1640 and Si III] 1888 differ by
# 0.36 to 0.49 dex and are not isolated here. The IGM agrees to 0.0002 away from the steps, and the one
# numerical difference, at the z = 6 Lyman-alpha edge, is set by BEAGLE's template sampling. BEAGLE-AGN,
# CCWW13 full, `ACCOUNT FOR HII DUST` and the noise model are not run in this notebook, and §15 is not
# written here.

# %% [markdown]
# ## References
#
# Entries 1 to 6 are cited as the BEAGLE Terms of Use require.
#
# 1. Chevallard J., Charlot S., 2016, "Modelling and interpreting spectral energy distributions of galaxies with BEAGLE", MNRAS 462, 1415. doi:10.1093/mnras/stw1756, arXiv:1603.03037.
# 2. Bruzual G., Charlot S., 2003, "Stellar population synthesis at the resolution of 2003", MNRAS 344, 1000. doi:10.1046/j.1365-8711.2003.06897.x, arXiv:astro-ph/0309134.
# 3. Gutkin J., Charlot S., Bruzual G., 2016, "Modelling the nebular emission from primeval to present-day star-forming galaxies", MNRAS 462, 1757. doi:10.1093/mnras/stw1716, arXiv:1607.06086.
# 4. Calzetti D., Kinney A. L., Storchi-Bergmann T., 1994, "Dust extinction of the stellar continua in starburst galaxies: The ultraviolet and optical extinction law", ApJ 429, 582. doi:10.1086/174346.
# 5. Charlot S., Fall S. M., 2000, "A Simple Model for the Absorption of Starlight by Dust in Galaxies", ApJ 539, 718. doi:10.1086/309250, arXiv:astro-ph/0003128.
# 6. Chevallard J., Charlot S., Wandelt B., Wild V., 2013, "Insights into the content and spatial distribution of dust from the integrated spectral properties of galaxies", MNRAS 432, 2061. doi:10.1093/mnras/stt523, arXiv:1303.6631.
# 7. Pei Y. C., 1992, "Interstellar dust from the Milky Way to the Magellanic Clouds", ApJ 395, 130. doi:10.1086/171637.
# 8. Inoue A. K., Shimizu I., Iwata I., Tanaka M., 2014, "An updated analytic model for attenuation by the intergalactic medium", MNRAS 442, 1805. doi:10.1093/mnras/stu936, arXiv:1402.0677.
# 9. Feltre A., Charlot S., Gutkin J., 2016, "Nuclear activity versus star formation: emission-line diagnostics at ultraviolet and optical wavelengths", MNRAS 456, 3354. doi:10.1093/mnras/stv2794, arXiv:1511.08217.
# 10. Vidal-García A., Plat A., Curtis-Lake E., Feltre A., Hirschmann M., Chevallard J., Charlot S., 2024, "BEAGLE-AGN I: Simultaneous constraints on the properties of gas in star-forming and AGN narrow-line regions in galaxies", MNRAS 527, 7217. doi:10.1093/mnras/stad3252, arXiv:2211.13648.
# 11. Plat A., Charlot S., Bruzual G., Feltre A., Vidal-García A., Morisset C., Chevallard J., Todt H., 2019, "Constraints on the production and escape of ionizing radiation from the emission-line spectra of metal-poor star-forming galaxies", MNRAS 490, 978. doi:10.1093/mnras/stz2616, arXiv:1909.07386.
# 12. Pacifici C. et al., 2023, "The Art of Measuring Physical Parameters in Galaxies: A Critical Assessment of Spectral Energy Distribution Fitting Techniques", ApJ 944, 141. doi:10.3847/1538-4357/acacff, arXiv:2212.01915.
# 13. Planck Collaboration, 2016, "Planck 2015 results. XIII. Cosmological parameters", A&A 594, A13. doi:10.1051/0004-6361/201525830, arXiv:1502.01589.
# 14. Ferland G. J. et al., 2017, "The 2017 Release of Cloudy", RMxAA 53, 385, arXiv:1705.10877.
# 15. Osterbrock D. E., Ferland G. J., 2006, *Astrophysics of Gaseous Nebulae and Active Galactic Nuclei*, 2nd ed., University Science Books.

# %%
print(f"total execution: {time.perf_counter() - _T0:.1f} s")
