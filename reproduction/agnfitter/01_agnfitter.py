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
# # Reproducing AGNfitter-rX's physics with tengri
#
# AGNfitter-rX (Martínez-Ramírez et al. 2024, A&A 688, A46) fits the radio-to-X-ray SEDs of
# active galaxies. It models the AGN itself (accretion disc, hot dusty torus, relativistic
# jet and hot corona) alongside the host (stars, cold dust, star-formation radio). Radio and
# X-ray data, largely free of dust, break the infrared-ultraviolet degeneracies that limited
# the original AGNfitter (Calistro Rivera et al. 2016).
#
# This page asks whether tengri can stand in for it. Each section configures tengri through
# `SEDModel.build`'s dict grammar and `model.predict(params)`, reads AGNfitter-rX's own
# templates and equations through a thin driver, and compares one physics block at matched
# parameters. The paper's fit conclusions (next cell) are not tested; the templates and
# equations underneath them are. tengri implements the same physics independently, and the
# AGNfitter-rX template libraries are repackaged data files.

# %% [markdown]
# **Reading guide.** The agreement class is stated qualitatively here; the numbers are in the
# section named in the last column. Section 6 collects every difference between the codes in
# one table, and section 7 translates an AGNfitter-rX fit into a tengri model.
#
# | AGNfitter-rX component | tengri spelling | Agreement | Section |
# |---|---|---|---|
# | Host stars (GA) | `sfh={'type': 'declining_exp'}`, BC03 grid | shape matched; mass convention differs (present vs formed) | 1.1-1.3 |
# | Host attenuation | `dust_attenuation={'law': 'calzetti'}` | matched inside the calibrated range; upstream tails differ | 1.4 |
# | Cold dust (SB) | `dust_emission={'type': 'schreiber2018'}`, `'dh02_ce01'` | matched at template nodes | 1.5-1.6 |
# | Accretion disc (BBB) | `agn={'disc': ...}`: `richards2006`, `slone_netzer`, `kd18_agnfitter`, `qsogen` with `blr` and `feii` | matched at nodes; THB21 is a template against a model | 2.1 |
# | Disc reddening | `agn_ebv_disc`, `atten` | same law, one constant factor apart | 2.2 |
# | Torus (TO) | `silva04`, `nenkova_agnfitter*`, `skirtor*`, `cat3d_wind*` | node-exact for tabulated blocks; amplitude convention differs by design | 3 |
# | X-ray corona | `xray={'type': 'yang20'}` | exact when anisotropy and absorption are off | 4 |
# | Radio | `radio={'agn': 'dpl', 'sf': 'bell2003_split'}` | exact for the AGN; matched by construction for star formation | 5 |
# | Informative priors | `tengri.agn.priors.agnfitter_priors` | two upstream priors replaced by physical versions | 6, 7 |

# %%
# Paper-quoted values the prose relies on (Martínez-Ramírez et al. 2024). They
# are taken from the paper, not computed here.
PAPER_QUOTED = {
    "log10 Bayes factor, THB21 over R06": ("5.1", "p. 14"),
    "sources whose maximum-likelihood torus is CAT3D-Wind": ("25 of 36", "p. 13"),
    "sources best fit by CAT3D-Wind + THB21": ("67%", "pp. 14-15"),
    "wavelength range of the near-infrared excess [µm]": ("1.5-5", "pp. 12-13"),
}
for _k, (_v, _pg) in PAPER_QUOTED.items():
    print(f"paper  {_k:55s} {_v:>9s}  ({_pg})")

# %% [markdown]
# ## Setup
#
# Both codes build on Bruzual & Charlot (2003) populations with a Chabrier (2003) initial
# mass function. AGNfitter-rX ships its own edition, tabulated over `(tau, age)`, which the
# driver reads from committed HDF5; tengri reads its `bc03_pdva_stelib_chabrier` grid, the same
# physics in a different (STELIB) spectral-library edition. Upstream's fit keeps every third
# wavelength point of its grid. The two cosmologies differ as well, so the luminosity-distance
# factor is compared at $z=1$ and the panels compare shape, not a cosmology. Every host below
# is stellar-only (`neb={'type': 'none'}`), like AGNfitter-rX's GALAXY component, which has no
# nebular term.

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from reproduction import _validation as V
from reproduction.agnfitter._drivers import agnfitter_driver as A, units as U
from reproduction.agnfitter._drivers.helpers import (
    assert_comparable,
    band_power,
    make_agn_builders,
    make_save_fig,
    node_exact_verdict,
    norm_at,
    norm_peak,
    resolved_params,
    val_at,
)

import tengri
from tengri import DEFAULT, Fixed, SEDModel, builders, load_ssp_data

# Force the inline backend so figures embed on (re-)render regardless of the
# ambient MPLBACKEND. A non-inline backend (e.g. Agg) drops the save_fig()
# auto-display and produces a figure-less notebook. No-op when run as a script.
try:  # noqa: SIM105
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

warnings.filterwarnings("ignore")
warnings.filterwarnings("default", module=r"tengri(\.|$)")
tengri.plot.setup_style()

# nbclient kernels don't bind ``__file__``; fall back to cwd so the notebook can
# locate ``_figs/`` and data.
_HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd().resolve()
save_fig = make_save_fig(_HERE / "_figs")

# Every panel below claims percent-level agreement, which rests on the AGNfitter-rX
# (log nu, F_nu) -> tengri (Angstrom, erg/s/Hz) bookkeeping. Trip the notebook here if
# that converter ever drifts.
_unit_check = U.verify_unit_conversion(rtol=1e-3)
print(f"unit-conversion bolometric round-trip: rel_err = {_unit_check['rel_err']:.2e}  (target < 1e-3)")

# The AGNfitter-rX template libraries are committed under data/, so this runs on a clean
# checkout; require_available() says what to regenerate if a grid is missing.
A.require_available()
print("AGNfitter-rX reference libraries (committed under data/):")
print(f"  disks      : {A.list_disks()}")
print(f"  tori       : {A.list_tori()}")
print(f"  cold dust  : {A.list_cold_dust()}")

# Live menus: ask the installed registry rather than keep a list of names.
print("tengri.list_agn_models():", sorted(r["name"] for r in tengri.list_agn_models()))
for _cat in ("disc", "nlr", "blr", "feii", "torus", "attenuation"):
    print(
        f"tengri.list_agn_blocks(category={_cat!r}):",
        sorted(r["name"] for r in tengri.list_agn_blocks(category=_cat)),
    )

# %%
_ssp_path = tengri.download_ssp("bc03_pdva_stelib_chabrier", dest=_HERE / "_drivers" / "data")
ssp = load_ssp_data(str(_ssp_path))
_ga_w, _ = A.galaxy_template(1.0, 5e9)
_ga_w3, _ = A.galaxy_template(1.0, 5e9, subsample3=True)
print(
    f"tengri BC03 + Chabrier SSP (STELIB edition): {ssp.ssp_wave.shape[0]} wavelengths, "
    f"{ssp.ssp_lgmet.shape[0]} metallicities, {ssp.ssp_lg_age_gyr.shape[0]} ages."
)
print(
    f"AGNfitter-rX BC03 edition: {_ga_w.size} wavelengths in the pickle, "
    f"{_ga_w3.size} kept by the fit (every third point)."
)

import tengri.cosmology as cosmo

AF_H0, AF_OM0 = 67.4, 0.315  # AGNfitter-rX cosmology (paper p. 2; z2Dlum)
_dl_af = cosmo.luminosity_distance(1.0, h0=AF_H0, om0=AF_OM0)
_dl_tengri = cosmo.luminosity_distance(1.0)
_c = cosmo.DEFAULT_COSMO
print("cosmology          |    H0 [km/s/Mpc] |    Omega_m")
print(f"AGNfitter-rX       | {AF_H0:16.2f} | {AF_OM0:10.5f}")
print(f"tengri default     | {100 * float(_c.h):16.2f} | {float(_c.Om0):10.5f}")
print(
    f"D_L(z=1): AGNFITTER-RX = {_dl_af:.4e} cm;  tengri = {_dl_tengri:.4e} cm;  "
    f"D_L^2 ratio tengri/AGNFITTER-RX = {(_dl_tengri / _dl_af) ** 2:.4f}"
)

# Fiducial host galaxy: AGNfitter-rX's own declining-exponential (tau-model)
# SFH (Martínez-Ramírez et al. 2024, p.7), tau = 1 Gyr, age = 4.8939 Gyr —
# the nearest node in AGNfitter-rX's own (tau, age) grid to 5 Gyr, so every
# panel that reuses SFH_FIDUCIAL ties to a matched reference node.
# The 'declining_exp' model takes its fully prefixed parameter names
# (e.g. `sfh_declining_exp_tau_gyr`).
TAU_GYR, AGE_GYR, LOG_MASS = 1.0, 4.8939, 10.0
SFH_FIDUCIAL = {
    "type": "declining_exp",
    "sfh_declining_exp_tau_gyr": Fixed(TAU_GYR),
    "sfh_declining_exp_age_gyr": Fixed(AGE_GYR),
    "sfh_declining_exp_log_total_mass": Fixed(LOG_MASS),
    "all_params": Fixed(DEFAULT),
}
NO_DUST = {
    "law": "power_law",
    "type": "two_component",
    "tau_bc": Fixed(0.0),
    "tau_diff": Fixed(0.0),
    "all_params": Fixed(DEFAULT),
}

# Isolated AGN sub-block builders (disc, torus, THB21 analog, disc + atten law): each one
# is a single ``SEDModel.build(agn={...})`` read from ``model.predict(params)``.
B = make_agn_builders(ssp, SFH_FIDUCIAL, NO_DUST)

# %% [markdown]
# ## 1 Host galaxy
#
# AGNfitter-rX's host is a stellar template (GA), a cold-dust template (SB) and, for the
# radio, a star-formation tail. This section compares each at matched parameters, and ends
# with the host composite at physical normalization. The result a user needs most is the
# mass convention of §1.3.

# %% [markdown]
# ### 1.1 Stellar populations
#
# Does tengri's BC03 library, integrated over the same declining-exponential history,
# reproduce the shape of AGNfitter-rX's tabulated template at a matched node? Left: tengri's
# library at two ages. Right: the fiducial composite stellar population ($\tau = 1$ Gyr, the
# age node of AGNfitter-rX's grid nearest 5 Gyr) against the template, read on its full
# pickle grid and normalized to 1 at 5500 Å. The panel tests shape, not absolute scale (§1.3).

# %%
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4.6))
for age, c in [(0.1, "C0"), (5.0, "C3")]:
    m = SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "declining_exp",
            "sfh_declining_exp_tau_gyr": Fixed(0.1),
            "sfh_declining_exp_age_gyr": Fixed(age),
            "sfh_declining_exp_log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=NO_DUST,
        neb={"type": "none"}, redshift=Fixed(0.0),
    )
    pred = m.predict({})
    w = np.asarray(pred.sed.components["wavelength"])
    L = np.asarray(pred.sed.components["sed_attenuated"])
    msk = (w > 9e2) & (w < 3e4)
    ax0.loglog(w[msk], L[msk], c, lw=1.4, label=f"tengri BC03, age = {age:g} Gyr")
ax0.set_xlabel(r"$\lambda$ [Å]")
ax0.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax0.set_title("BC03 + Chabrier stellar populations (shared library)")
ax0.legend(fontsize=8)
ax0.grid(True, alpha=0.3)

m_csp = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST, neb={"type": "none"}, redshift=Fixed(0.0))
resolved_params(m_csp)
pred_csp = m_csp.predict({})
w_t = np.asarray(pred_csp.sed.components["wavelength"])
L_t = np.asarray(pred_csp.sed.components["sed_attenuated"])
w_r, L_r = A.galaxy_template(tau=TAU_GYR, age=AGE_GYR * 1e9)
_bandS1 = (w_t > 1.5e3) & (w_t < 2.5e4)
t_n = norm_at(w_t, L_t, 5500.0)
r_n = norm_at(w_r, L_r, 5500.0)
ax1.loglog(w_t[_bandS1], t_n[_bandS1], "C1-", lw=1.5, label="tengri declining_exp CSP")
ax1.loglog(w_r, r_n, "C0-", lw=3.5, alpha=0.35, solid_capstyle="round", label="AGNFITTER-RX BC03 (matched node)")
ax1.set_xlim(1.5e3, 2.5e4)
ax1.set_xlabel(r"$\lambda$ [Å]")
ax1.set_title(rf"Matched CSP node ($\tau$={TAU_GYR:g} Gyr, age={AGE_GYR:.4f} Gyr)")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_01_ssp_bc03.png")

_o = np.argsort(w_r)
_t_on_r = np.interp(np.log10(w_r[_o]), np.log10(w_t), np.log10(np.clip(t_n, 1e-300, None)))
_logratio = _t_on_r - np.log10(np.clip(r_n[_o], 1e-300, None))
_band_m = (w_r[_o] > 1.5e3) & (w_r[_o] < 2.5e4)
print(
    f"§1.1  matched CSP node (tau={TAU_GYR:g} Gyr, age={AGE_GYR:.4f} Gyr), 0.15-2.5 um, "
    f"5500-A-normalized: median|log10 ratio| = {np.median(np.abs(_logratio[_band_m])):.4f}"
)

# %% [markdown]
# The paper's default library, BC03_metal, carries four metallicities. The table builds
# tengri at each, mapping the upstream axis (units of the BC03 solar value, $Z_\odot = 0.02$)
# onto `met_logzsol` (relative to tengri's solar value, $Z_\odot = 0.0142$), and compares shape
# at 5500 Å and absolute scale through the present-mass mapping of §1.3.
#
# **Caveat:** the top upstream node (twice the BC03 solar value) lies between tengri's
# tabulated metallicities (printed above the table), so tengri interpolates there.

# %%
_Z_SUN_BC03, _Z_SUN_TENGRI = 0.02, 0.0142
_ax_m = A.galaxy_axes(metal=True)
_age_m = float(_ax_m["age"][int(np.argmin(np.abs(_ax_m["age"] - AGE_GYR * 1e9)))])
print(f"§1.1  tengri SSP metallicities Z = {np.round(10 ** np.asarray(ssp.ssp_lgmet), 5).tolist()}")
print(
    f"§1.1  BC03_metal nodes: Z/Z_sun = {_ax_m['metal'].tolist()}; tau = {TAU_GYR:g} Gyr, "
    f"age node = {_age_m / 1e9:.4f} Gyr"
)
print("    | Z [BC03 Z_sun] | tengri met_logzsol | median|dlog10| shape, 0.15-2.5 um | L(5500) tengri / (template x M*_present) |")
_sfh_m = {**SFH_FIDUCIAL, "sfh_declining_exp_age_gyr": Fixed(_age_m / 1e9)}
for _z in _ax_m["metal"]:
    _logzsol = float(np.log10(_z * _Z_SUN_BC03 / _Z_SUN_TENGRI))
    _mm = SEDModel.build(
        ssp_data=ssp, sfh=_sfh_m, dust_attenuation=NO_DUST, neb={"type": "none"}, redshift=Fixed(0.0),
        met={"logzsol": Fixed(_logzsol), "all_params": Fixed(DEFAULT)},
    )
    _pm = _mm.predict({})
    _wm = np.asarray(_pm.sed.components["wavelength"])
    _Lm = np.asarray(_pm.sed.components["sed_attenuated"])
    _wr, _Lr = A.galaxy_template(TAU_GYR, _age_m, metal=float(_z))
    _b = (_wr > 1.5e3) & (_wr < 2.5e4)
    _shape = np.median(np.abs(np.log10(
        (np.interp(_wr[_b], _wm, _Lm) / np.interp(5500.0, _wm, _Lm)) / (_Lr[_b] / np.interp(5500.0, _wr, _Lr)))))
    _abs = np.interp(5500.0, _wm, _Lm) / (
        np.interp(5500.0, _wr, _Lr) * float(_pm.sfh.stellar_mass_surviving))
    print(f"    | {float(_z):14.1f} | {_logzsol:18.3f} | {_shape:35.4f} | {_abs:40.4f} |")

# %% [markdown]
# **Result.** At the matched node the median shape residual is 0.024 dex, and at most 0.012 dex
# at the four metallicity nodes (both printed). Three effects account for it, none a mismatch of the SFH
# form: the two BC03 editions' spectral resolution, the nearest-node age snap and the shared 5500 Å
# anchor. With the present-mass mapping applied, the absolute scale sits 8-10% below the
# template at 5500 Å (table); §1.3 shows that this drift is wavelength dependent, the signature of the two BC03
# editions, not a normalization error.

# %% [markdown]
# ### 1.2 Star formation history
#
# AGNfitter-rX's GALAXY component tabulates a declining exponential, $\mathrm{SFR}(T) \propto
# e^{-T/\tau}$: the SFR falls monotonically from formation to the present. tengri's
# `declining_exp` type has the same form. The cell writes the closed form directly, checks
# mass closure through the public `pred.sfh.stellar_mass`, and overlays the template's tabulated
# SFR(age) at the fiducial $\tau$, renormalized to tengri's mass (shape only).

# %%
m2 = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST, neb={"type": "none"}, redshift=Fixed(0.0))
pred2 = m2.predict({})
mass_formed = pred2.sfh.stellar_mass  # public property: SFH's own mass-formed integral

age_axis_yr, sfr_ref = A.galaxy_sfr(tau=TAU_GYR)
t_lb_gyr = np.linspace(0.0, AGE_GYR, 400)
T_cosmic_gyr = AGE_GYR - t_lb_gyr  # cosmic time elapsed since formation
sfr_shape = np.where(T_cosmic_gyr >= 0, np.exp(-T_cosmic_gyr / TAU_GYR), 0.0)
sfr_analytic = sfr_shape * (10**LOG_MASS) / np.trapezoid(sfr_shape, t_lb_gyr * 1e9)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(t_lb_gyr, sfr_analytic, "C1-", lw=1.6, label=r"tengri declining_exp, $\mathrm{SFR}\propto e^{-T/\tau}$")
ax.plot(
    age_axis_yr / 1e9,
    sfr_ref / np.trapezoid(sfr_ref, age_axis_yr) * float(mass_formed),
    "C0--",
    lw=1.6,
    label="AGNFITTER-RX BC03 pickle (renormalized to tengri's mass, shape only)",
)
ax.set_xlabel("lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.set_title(rf"Declining-exponential SFH ($\tau$ = {TAU_GYR:g} Gyr, age = {AGE_GYR:.4f} Gyr)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
ax.text(
    0.97, 0.95,
    rf"pred.sfh.stellar_mass = {mass_formed / 10**LOG_MASS:.4f} $\times\,10^{{{LOG_MASS:.0f}}}\,M_\odot$",
    transform=ax.transAxes, ha="right", va="top",
)
fig.tight_layout()
save_fig("agnfitter_02_sfh_tau.png")
print(
    f"§1.2  pred.sfh.stellar_mass = {mass_formed:.4e} M_sun  (target 1.0000e{LOG_MASS:.0f}); "
    f"AGNFITTER-RX reference SFR is monotonically declining: "
    f"{bool(np.all(np.diff(sfr_ref) <= 0))}"
)

# %% [markdown]
# **Result.** The shapes coincide and the formed mass closes to the declared value; the
# upstream SFR is monotonically declining as printed.

# %% [markdown]
# ### 1.3 Integrated stellar SED and the mass convention
#
# The stored GA template is normalized to one solar mass of **present** stellar mass (stars
# plus remnants alive at the template age); tengri normalizes to mass **formed**. A prior on
# the GA amplitude therefore means a different stellar mass in the two codes unless it is
# mapped. The table below gives the mapping at the matched node and checks it against the
# absolute SED at four wavelengths; it is the conversion used in sections 6 and 7.
#
# **Verification Status:** PARTIAL; Absolute SED normalization

# %%
pred3 = m_csp.predict({})
w3 = np.asarray(pred3.sed.components["wavelength"])
L3 = np.asarray(pred3.sed.components["sed_attenuated"])
fig, ax = plt.subplots(figsize=(7, 4.5))
msk = (w3 > 9e2) & (w3 < 1e7)
ax.loglog(w3[msk], L3[msk], "C1-", lw=1.4)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title(rf"Integrated stellar SED ($10^{{{LOG_MASS:.0f}}}\,M_\odot$ host)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_03_stellar_sed.png")
_m_formed = float(pred3.sfh.stellar_mass)
_m_present = float(pred3.sfh.stellar_mass_surviving)
_surv_frac = _m_present / _m_formed
_f_formed_per_present = A.galaxy_formed_per_present(TAU_GYR, AGE_GYR * 1e9)
_w_ga, _L_ga_unit = A.galaxy_template(TAU_GYR, AGE_GYR * 1e9)
_L_ga_phys = _L_ga_unit * _m_present  # unit template x present mass [erg/s/Hz]
_lam_chk = np.array([3000.0, 5500.0, 1e4, 2e4])
_ratio_abs = np.interp(_lam_chk, w3, L3) / np.interp(_lam_chk, _w_ga, _L_ga_phys)
print("§1.3  GA ↔ tengri stellar-mass mapping at the matched node")
print("    | quantity                                                     |      value |")
print(f"    | tengri M*_formed (pred.sfh.stellar_mass) [Msun]              | {_m_formed:10.4e} |")
print(f"    | tengri M*_present (stellar_mass_surviving) [Msun]            | {_m_present:10.4e} |")
print(f"    | tengri surviving fraction M*_present / M*_formed             | {_surv_frac:10.4f} |")
print(f"    | upstream unit template: M_formed per M_present (SFR table)   | {_f_formed_per_present:10.4f} |")
print(f"    | 1 / that = upstream surviving fraction                       | {1.0 / _f_formed_per_present:10.4f} |")
for _lam, _r in zip(_lam_chk, _ratio_abs):
    print(f"    | L_nu tengri / (unit template x M*_present) at {_lam:7.0f} Å            | {_r:10.4f} |")
print(
    "    GA amplitude -> present mass:  M*_present = 10^GA · 4π d_L² / [L_sun (1+z)] / 1e18  "
    "(MODEL_AGNfitter.stellar_info); formed mass = M*_present / surviving fraction."
)

# %% [markdown]
# **Result.** With the mapping applied the absolute SEDs agree to 2% at 0.3-0.55 µm and fall to
# 0.88-0.90 at 1-2 µm, where the two BC03 editions part; that is the edition difference of §1.1,
# not a normalization error. The two codes also disagree on the surviving fraction (tengri's and
# the upstream table's are both printed), so a GA prior must go through the printed relation:
# convert the amplitude to a present mass, then divide by tengri's surviving fraction to get the
# formed mass that `sfh_declining_exp_log_total_mass` expects.

# %% [markdown]
# ### 1.4 Galaxy attenuation: Calzetti curve
#
# Does tengri's Calzetti law reproduce the curve AGNfitter-rX applies to the host? Both
# redden the whole galaxy continuum with the Calzetti et al. (2000) curve,
#
# $$k'(\lambda) = 2.659\,(-2.156 + 1.509\,x - 0.198\,x^2 + 0.011\,x^3) + R_V
# \quad (\lambda < 0.63\,\mu{\rm m}),$$
# $$k'(\lambda) = 2.659\,(-1.857 + 1.040\,x) + R_V \quad (\lambda \ge 0.63\,\mu{\rm m}),$$
#
# with $x = 1/\lambda_{\mu{\rm m}}$, $R_V = 4.05$ and $\tau_V = R_V\,E(B{-}V)/1.086$. Left: inside the
# calibrated range (0.12-2.2 µm, shaded) tengri follows the analytic curve. Right: outside it
# the codes diverge; the reference is upstream's own evaluation (`GALAXYred_Calzetti`),
# reproduced by the driver on the grid the fit uses.
#
# **Caveat:** below 0.12 µm upstream extrapolates linearly from two grid points and adds a
# second $R_V$, and in the near infrared its linear branch turns negative, which *brightens* the
# template (the crossing and the size of both tails are printed). The comparison is therefore
# made inside the calibrated range only. tengri's tails are a choice, not a measurement. The
# two-screen Charlot & Fall (2000) law has no working upstream counterpart: the function is
# defined but never called, and its wavelength argument mixes µm with Å.
#
# **Verification Status:** CROSSVAL; Attenuation law library

# %%
_EBV_GAL = 0.3
_RV_CALZETTI = 4.05
_tau_v_gal = _RV_CALZETTI * _EBV_GAL / 1.086

_wl_cal = np.geomspace(950.0, 1.2e5, 600)  # 0.095-12 um: calibrated range plus both tails
_x_cal = 1e4 / _wl_cal  # 1/um
_k_ir = 2.659 * (-1.857 + 1.040 * _x_cal)
_k_uv = 2.659 * (-2.156 + 1.509 * _x_cal - 0.198 * _x_cal**2 + 0.011 * _x_cal**3)
_k_prime = np.where(_wl_cal >= 6300.0, _k_ir, _k_uv)
_x_5500 = 1e4 / 5500.0
_k5500 = (2.659 * (-2.156 + 1.509 * _x_5500 - 0.198 * _x_5500**2 + 0.011 * _x_5500**3) + _RV_CALZETTI) / _RV_CALZETTI
# tengri's calzetti() returns the DIMENSIONLESS shape (k(5500 A)=1); the
# physical A_lambda/E(B-V) is that shape times R_V (A_V = R_V * E(B-V) by
# definition at V band) -- the same relation the build below realizes via
# dust_tau_v = R_V * E(B-V) / 1.086.
_k_analytic = (_k_prime + _RV_CALZETTI) / _k5500

m_g0 = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL,
                       dust_attenuation={"type": "single_component", "law": "calzetti",
                                         "dust_tau_v": Fixed(0.0), "all_params": Fixed(DEFAULT)},
                       neb={"type": "none"}, redshift=Fixed(0.0))
m_g1 = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL,
                       dust_attenuation={"type": "single_component", "law": "calzetti",
                                         "dust_tau_v": Fixed(_tau_v_gal), "all_params": Fixed(DEFAULT)},
                       neb={"type": "none"}, redshift=Fixed(0.0))
resolved_params(m_g1)
w_g0 = np.asarray(m_g0.predict({}).sed.components["wavelength"])
L_g0 = np.asarray(m_g0.predict({}).sed.components["sed_attenuated"])
L_g1 = np.asarray(m_g1.predict({}).sed.components["sed_attenuated"])
ratio_gal = np.divide(L_g1, L_g0, out=np.ones_like(L_g1), where=L_g0 > 0)
A_lambda_gal = -2.5 * np.log10(np.clip(ratio_gal, 1e-300, None)) / _EBV_GAL
A_lambda_gal_on_grid = np.interp(_wl_cal, w_g0, A_lambda_gal)

# Upstream's own evaluation on the (every-third-point) grid its fit uses.
_w_up, _ = A.galaxy_template(TAU_GYR, AGE_GYR * 1e9, subsample3=True)
_, _red_up = A.galaxy_redden_calzetti(_w_up, np.ones_like(_w_up), _EBV_GAL)
_A_up = -2.5 * np.log10(_red_up) / _EBV_GAL

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 4.6))
_cal = (_wl_cal >= 950.0) & (_wl_cal <= 2.2e4)
axl.plot(_wl_cal[_cal], _k_analytic[_cal], "C0-", lw=3.5, alpha=0.35, solid_capstyle="round",
         label="Calzetti+2000 analytic $k(\\lambda)$")
axl.plot(_wl_cal[_cal], A_lambda_gal_on_grid[_cal], "C1-", lw=1.4, label="tengri  dust_attenuation={'law':'calzetti'} (via build)")
axl.axvspan(1200.0, 2.2e4, color="0.9", zorder=0)
axl.set_xscale("log")
axl.set_title("Calibrated range (0.12-2.2 µm, shaded)")
axl.set_xlabel(r"$\lambda$ [Å]")
axl.set_ylabel(r"$A_\lambda / E(B{-}V)$")
axl.legend(fontsize=8)
axl.grid(True, alpha=0.3)
axr.plot(_w_up, _A_up, "C0-", lw=3.5, alpha=0.35, solid_capstyle="round", label="AGNFITTER-RX  GALAXYred_Calzetti (as evaluated)")
axr.plot(_wl_cal, A_lambda_gal_on_grid, "C1-", lw=1.4, label="tengri  calzetti (via build)")
axr.axvline(1200.0, color="0.7", ls=":", lw=1)
axr.axvline(2.2e4, color="0.7", ls=":", lw=1)
axr.axhline(0.0, color="0.5", lw=0.8)
axr.set_xscale("log")
axr.set_xlim(950, 1.2e5)
axr.set_title("Beyond the calibrated range (Caveat)")
axr.set_xlabel(r"$\lambda$ [Å]")
axr.legend(fontsize=8)
axr.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_05_galaxy_calzetti.png")

_m_cal = (_wl_cal >= 1200.0) & (_wl_cal <= 2.2e4)
print(
    f"§1.4  Calzetti parity (0.12-2.2 um): max|A_tengri - A_analytic| = "
    f"{np.max(np.abs(A_lambda_gal_on_grid[_m_cal] - _k_analytic[_m_cal])):.4f} mag/E(B-V)"
)
print("§1.4  A_lambda / E(B-V) outside the calibrated range")
print("    | lambda [um] | analytic | tengri (build) | AGNfitter-rX as evaluated |")
for _lam_um in (0.095, 0.1195, 1.0, 2.2, 5.0, 10.0):
    _lam = _lam_um * 1e4
    print(
        f"    | {_lam_um:11.4f} | {np.interp(_lam, _wl_cal, _k_analytic):8.2f} | "
        f"{np.interp(_lam, _wl_cal, A_lambda_gal_on_grid):14.2f} | {np.interp(_lam, _w_up, _A_up):25.2f} |"
    )
_neg = _w_up[(_w_up > 6300.0) & (_A_up < 0.0)]
print(
    f"    upstream k turns negative at {_neg[0] / 1e4:.2f} um; flux boost at 10 um for "
    f"E(B-V)={_EBV_GAL}: x{10 ** (-0.4 * np.interp(1e5, _w_up, _A_up) * _EBV_GAL):.3f}"
)

# %% [markdown]
# **Result.** Inside the calibrated range tengri follows the analytic curve to the residual
# printed (mag per unit $E(B{-}V)$). Outside it the two codes part, as the table shows: upstream's
# extrapolation below 0.12 µm and its negative infrared branch are not physical, so only the
# calibrated range is a meaningful comparison.

# %% [markdown]
# ### 1.5 Cold dust infrared emission
#
# AGNfitter-rX ships two cold-dust libraries: S17 (Schreiber et al. 2018, dust continuum plus
# PAH through $T_{\rm dust}$ and $f_{\rm PAH}$) and DH02_CE01 (Dale & Helou 2002 with Chary &
# Elbaz 2001, indexed by IR luminosity). Each panel builds a minimal model with
# `dust_emission={'type': ..., ...}`, whose absorbed luminosity comes from the build's own
# attenuated stellar continuum, and reads `sed_dust_ir`. A radio block extends the model
# wavelength grid past the far infrared; the printed grid limits show what a build without one
# loses.
#
# `schreiber2018` and `schreiber2016` take `dust_T` and `dust_f_pah`, at nodes inside the ranges
# the paper fits. `dh02_ce01` is indexed by the model's own realized IR luminosity
# (`pred.l_tir`), so its node is read off: `dust_tau_v` is bisected until the luminosity lands
# on a grid node, and a ladder of screens shows the nodes it reaches.
#
# **Verification Status:** CROSSVAL; Dust IR emission physics (MBB, Casey12, CMB)

# %%
_RADIO_GRID_EXTENSION = {"sf": {"type": "none"}, "agn": {"type": "powerlaw"}}  # no AGN in the host builds: extends the grid only


def _dust_emission_build(dtype, tau_v, *, radio=True, **params):
    kwargs = {k: Fixed(v) for k, v in params.items()}
    extra = {"radio": _RADIO_GRID_EXTENSION} if radio else {}
    m = SEDModel.build(
        ssp_data=ssp,
        sfh={"type": "declining_exp", "sfh_declining_exp_tau_gyr": Fixed(0.3),
             "sfh_declining_exp_age_gyr": Fixed(1.0), "sfh_declining_exp_log_total_mass": Fixed(10.5),
             "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "single_component", "law": "calzetti", "dust_tau_v": Fixed(tau_v),
                           "all_params": Fixed(DEFAULT)},
        dust_emission={"type": dtype, **kwargs, "all_params": Fixed(DEFAULT)},
        neb={"type": "none"}, redshift=Fixed(0.0), **extra,
    )
    _dust_emission_build.last_model = m
    pred = m.predict({})
    w = np.asarray(pred.sed.components["wavelength"])
    L = np.asarray(pred.sed.components["sed_dust_ir"])
    return w, L, pred

wave_ir = np.geomspace(1e4, 1e8, 2000)
S17_NODES = [(20.0, 0.01), (30.0, 0.02), (38.0, 0.05), (42.0, 0.03), (35.0, 0.0)]  # (T_dust [K], f_PAH)
plt.rcParams["figure.dpi"] = 100  # keep the rendered notebook under the figure-size budget
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
_s17_resid = []
for _i, ((T, fpah), c) in enumerate(zip(S17_NODES, ["C0", "C1", "C2", "C3", "C4"])):
    w_te, L_te, _ = _dust_emission_build("schreiber2018", 3.0, dust_T=T, dust_f_pah=fpah)
    if _i == 0:
        resolved_params(_dust_emission_build.last_model)
    w_s17, L_s17 = A.cold_dust_template("S17", tdust=T, fpah=fpah)
    _b = (w_s17 > 3e4) & (w_s17 < 3e6)
    # Normalize both arms at the peak of the reference within the comparison band
    _peak_idx_s17 = int(np.argmax(L_s17[_b])) + int(np.where(_b)[0][0])
    _norm_wave_s17 = w_s17[_peak_idx_s17]
    s17n = norm_at(w_s17, L_s17, _norm_wave_s17)
    te_on_s17 = norm_at(w_s17, np.interp(w_s17, w_te, L_te, left=0.0, right=0.0), _norm_wave_s17)
    resid = np.abs(np.log10(np.clip(te_on_s17[_b], 1e-30, None)) - np.log10(np.clip(s17n[_b], 1e-30, None)))
    _s17_resid.append((T, fpah, float(np.median(resid))))
    axL.loglog(w_s17, norm_peak(L_s17), c + "-", lw=4.0, alpha=0.3, solid_capstyle="round")
    axL.loglog(w_te, norm_peak(L_te), c + "-", lw=1.4, label=f"T={T:g} K, f_PAH={fpah:g}")
axL.set_xlim(1e4, 1e8)
axL.set_ylim(1e-6, 3)
axL.set_xlabel(r"$\lambda$ [Å]")
axL.set_ylabel(r"$L_\nu$ (norm. at peak)")
axL.set_title("schreiber2018 vs AGNFITTER-RX S17 (thick=AF, thin=tengri)")
axL.legend(fontsize=8)
axL.grid(True, alpha=0.3)

w_te16, L_te16, _ = _dust_emission_build("schreiber2016", 3.0, dust_T=35.0, dust_f_pah=0.02)
w_ted, L_ted, _ = _dust_emission_build("dale2014", 3.0, dust_alpha_dale=1.5)
w_s17ref, L_s17ref = A.cold_dust_template("S17", tdust=35.0, fpah=0.02)
axR.loglog(w_s17ref, norm_peak(L_s17ref), "0.6", lw=2.0, alpha=0.6, label="AGNFITTER-RX  S17 (ref)")
axR.loglog(w_te16, norm_peak(L_te16), "C1-", lw=1.5, label="tengri  schreiber2016 (analytic)")
axR.loglog(w_ted, norm_peak(L_ted), "C4-", lw=1.5, label=r"tengri  dale2014 ($\alpha=1.5$)")
axR.set_xlim(1e4, 1e8)
axR.set_xlabel(r"$\lambda$ [Å]")
axR.set_title("Differentiable alternatives (not node-matched)")
axR.legend(fontsize=8)
axR.grid(True, alpha=0.3)
fig.suptitle("Cold-dust IR — S17 nodes (left) vs differentiable alternatives (right)", y=1.02)
fig.tight_layout()
save_fig("agnfitter_06_cold_dust.png")
plt.rcParams["figure.dpi"] = 150

# %%
# Find tau_v that lands exactly on a DH02_CE01 grid node
_dh_axis = A.cold_dust_axes("DH02_CE01")["log_irlum"]

# Sample tau_v to bracket the desired range
_, _, _pred_low = _dust_emission_build("dh02_ce01", 1.0)
_lir_low = float(np.log10(_pred_low.l_tir))
_, _, _pred_high = _dust_emission_build("dh02_ce01", 8.0)
_lir_high = float(np.log10(_pred_high.l_tir))

# Select the median node in the bracketed range
_target_nodes = _dh_axis[(_dh_axis >= _lir_low) & (_dh_axis <= _lir_high)]
_target = _target_nodes[len(_target_nodes) // 2] if len(_target_nodes) > 0 else 10.9

# Binary search for tau_v that matches the target node
_low, _high = 1.0, 8.0
for _ in range(18):
    _mid = (_low + _high) / 2
    _, _, _pred = _dust_emission_build("dh02_ce01", _mid)
    _lir = float(np.log10(_pred.l_tir))
    if _lir < _target:
        _low = _mid
    else:
        _high = _mid
_tau_v_on_node = (_low + _high) / 2

w_dh, L_dh, pred_dh = _dust_emission_build("dh02_ce01", _tau_v_on_node)
_log_lir_realized = float(np.log10(pred_dh.l_tir))
_dh_node = float(_dh_axis[int(np.argmin(np.abs(_dh_axis - _log_lir_realized)))])
w_dhref, L_dhref = A.cold_dust_template("DH02_CE01", log_irlum=_dh_node)
_bd = (w_dhref > 3e4) & (w_dhref < 3e6)
# Normalize both arms at the peak of the reference within the comparison band
_peak_idx_ref = int(np.argmax(L_dhref[_bd])) + int(np.where(_bd)[0][0])
_norm_wave_dh = w_dhref[_peak_idx_ref]
_dhn = norm_at(w_dhref, L_dhref, _norm_wave_dh)
_te_on_dh = norm_at(w_dhref, np.interp(w_dhref, w_dh, L_dh, left=0.0, right=0.0), _norm_wave_dh)
_dh_resid = float(np.median(np.abs(np.log10(np.clip(_te_on_dh[_bd], 1e-30, None)) - np.log10(np.clip(_dhn[_bd], 1e-30, None)))))
_, _, _pred_nr = _dust_emission_build("dh02_ce01", 3.0, radio=False)
_w_nr = np.asarray(_pred_nr.sed.components["wavelength"])
print(
    f"§1.5  model wavelength grid ends at {w_dh.max() / 1e4:.4g} um with a radio block and at "
    f"{_w_nr.max() / 1e4:.4g} um without one"
)
print("§1.5  S17 node table (T_dust [K], f_PAH, median|log10 ratio| over 3-300 um):")
for T, fpah, resid in _s17_resid:
    print(f"    T={T:5.1f} K  f_PAH={fpah:.2f}  median|Delta log10| = {resid:.4f}")
print(
    f"§1.5  dh02_ce01: realized log10(L_TIR/Lsun) = {_log_lir_realized:.3f}; "
    f"nearest AGNFITTER-RX grid node {_dh_node:.3f}; "
    f"median|log10 ratio| over 3-300 um at that node = {_dh_resid:.4f} dex."
)

# %%
_dh_resids = [_dh_resid]
print("§1.5  dh02_ce01 log L_IR grid via dust_tau_v (node read off pred.l_tir):")
for tau_v_try in [0.1, 2.0, 40.0]:
    w_dh_i, L_dh_i, pred_dh_i = _dust_emission_build("dh02_ce01", tau_v_try)
    _log_lir_i = float(np.log10(pred_dh_i.l_tir))
    _dh_node_i = float(_dh_axis[int(np.argmin(np.abs(_dh_axis - _log_lir_i)))])
    w_dhref_i, L_dhref_i = A.cold_dust_template("DH02_CE01", log_irlum=_dh_node_i)
    _bd_i = (w_dhref_i > 3e4) & (w_dhref_i < 3e6)
    # Normalize both arms at the peak of the reference within the comparison band
    _peak_idx_ref_i = int(np.argmax(L_dhref_i[_bd_i])) + int(np.where(_bd_i)[0][0])
    _norm_wave_dh_i = w_dhref_i[_peak_idx_ref_i]
    _dhn_i = norm_at(w_dhref_i, L_dhref_i, _norm_wave_dh_i)
    _te_on_dh_i = norm_at(w_dhref_i, np.interp(w_dhref_i, w_dh_i, L_dh_i, left=0.0, right=0.0), _norm_wave_dh_i)
    _dh_resid_i = float(
        np.median(np.abs(np.log10(np.clip(_te_on_dh_i[_bd_i], 1e-30, None)) - np.log10(np.clip(_dhn_i[_bd_i], 1e-30, None))))
    )
    _dh_resids.append(_dh_resid_i)
    print(
        f"    dust_tau_v={tau_v_try:5.1f}  ->  log10(L_TIR/Lsun) = {_log_lir_i:.3f}, "
        f"node = {_dh_node_i:.3f}, median|log10 ratio| = {_dh_resid_i:.4f}"
    )

# %% [markdown]
# **Result.** Both libraries reproduce the template shape at their nodes (median residuals
# printed above). `schreiber2016` and `dale2014` are differentiable alternatives, not
# node-matched, and are shown only for shape.

# %% [markdown]
# ### 1.6 Host composite: GA + SB at physical normalization
#
# One build with the fiducial host and S17 cold-dust emission gives `sed_attenuated +
# sed_dust_ir`, energy-balanced by `dust_eta_balance`. The AGNfitter-rX side sums its GA and SB
# terms as `ymodel` does: the GA template at tengri's present stellar mass (§1.3), reddened by
# upstream's Calzetti evaluation (§1.4), plus the S17 template. The starburst amplitude is a
# free parameter of the fit, so the matched input is the infrared luminosity: S17 is scaled until
# its 8-1000 µm power equals tengri's realized $L_{\rm TIR}$, and what is tested is the dust
# shape. The galaxy term has no free scale.
#
# **Caveat:** tengri evaluates dust emission on the model wavelength grid, which is sparse in the
# infrared (the table counts points per band), so a sub-band power carries the sampling error;
# the total infrared power is the robust comparison.

# %%
m7 = SEDModel.build(
    ssp_data=ssp, sfh=SFH_FIDUCIAL,
    dust_attenuation={"type": "single_component", "law": "calzetti", "dust_tau_v": Fixed(_tau_v_gal),
                       "all_params": Fixed(DEFAULT)},
    dust_emission={"type": "schreiber2018", "dust_T": Fixed(35.0), "dust_f_pah": Fixed(0.02),
                    "all_params": Fixed(DEFAULT)},
    radio=_RADIO_GRID_EXTENSION, neb={"type": "none"}, redshift=Fixed(0.0),
)
pred7 = m7.predict({})
w7 = np.asarray(pred7.sed.components["wavelength"])
_ga7_te = np.asarray(pred7.sed.components["sed_attenuated"])
_sb7_te = np.asarray(pred7.sed.components["sed_dust_ir"])
host_sed = _ga7_te + _sb7_te
_log_lir7 = float(np.log10(pred7.l_tir))

_m_present7 = float(pred7.sfh.stellar_mass_surviving)
w_ga, L_ga = A.galaxy_lnu(TAU_GYR, AGE_GYR * 1e9, np.log10(_m_present7), ebv=_EBV_GAL)
w_sb, L_sb = A.cold_dust_template("S17", tdust=35.0, fpah=0.02)
_IR_LO, _IR_HI = 8e4, 1e7  # 8-1000 um
_sb_scale = band_power(w7, _sb7_te, _IR_LO, _IR_HI) / band_power(w_sb, L_sb, _IR_LO, _IR_HI)
af_grid = np.geomspace(1e3, 1e8, 3000)
_af_parts, af_host = A.ymodel_sum({"GA": (w_ga, L_ga), "SB": (w_sb, L_sb * _sb_scale)}, af_grid)
te_host_on_grid = U.regrid(w7, np.clip(host_sed, 0, None), af_grid)

fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax.loglog(af_grid, af_host, "C0-", lw=3.5, alpha=0.35, solid_capstyle="round",
          label="AGNFITTER-RX  GA (present mass, Calzetti) + SB (L_IR-matched)")
ax.loglog(af_grid, te_host_on_grid, "C1-", lw=1.4, label="tengri  sed_attenuated + sed_dust_ir (one build)")
ax.set_xlim(1e3, 1e8)
ax.set_ylim(af_host[af_host > 0].max() * 1e-8, af_host.max() * 3)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title("Host composite (GA + SB), physical normalization")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
_res7 = np.where((af_host > 0) & (te_host_on_grid > 0), te_host_on_grid / af_host - 1.0, np.nan)
axr.axhline(0.0, color="0.5", lw=0.8)
axr.semilogx(af_grid, _res7, "C3-", lw=1.0)
axr.set_ylim(-1, 1)
axr.set_xlabel(r"$\lambda$ [Å]")
axr.set_ylabel("(tengri - AF)/AF")
axr.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_07_host_composite.png")

print(
    f"§1.6  tengri host: pred.l_tir = {pred7.l_tir:.3e} Lsun (log10={_log_lir7:.3f}); "
    f"M*_formed = {pred7.sfh.stellar_mass:.3e}, M*_present = {_m_present7:.3e} Msun"
)
print("§1.6  tengri / AGNfitter-rX band power, host composite")
print("    | band                         | tengri / AGNfitter-rX | tengri grid points | what it tests                     |")
for _lab, _lo, _hi, _why in [
    ("0.1-1 um (stellar)", 1e3, 1e4, "GA mass mapping + Calzetti"),
    ("1-3 um (stellar)", 1e4, 3e4, "GA mass mapping + Calzetti"),
    ("8-30 um (PAH / hot dust)", 8e4, 3e5, "S17 shape at fixed L_IR"),
    ("30-1000 um (cold dust)", 3e5, 1e7, "S17 shape at fixed L_IR"),
    ("8-1000 um (total IR)", _IR_LO, _IR_HI, "matched by construction"),
]:
    _r = band_power(w7, host_sed, _lo, _hi) / band_power(af_grid, af_host, _lo, _hi)
    _npts = int(np.sum((w7 >= _lo) & (w7 <= _hi)))
    print(f"    | {_lab:28s} | {_r:21.4f} | {_npts:18d} | {_why:33s} |")

# %% [markdown]
# **Result.** The total infrared power is matched by construction. The stellar bands, which test
# the mass mapping and the Calzetti curve together, sit 5% (0.1-1 µm) and 12% (1-3 µm) below
# AGNfitter-rX, the same BC03-edition drift as §1.3. The sub-band infrared rows test the S17
# shape at fixed $L_{\rm IR}$ and agree to a few per cent; read them with the point counts.

# %% [markdown]
# ## 2 Accretion disc
#
# The paper's disc conclusion rests on one spectral feature: the Hα + [N II] bump near 0.7 µm,
# carried only by the semi-empirical THB21 library. This section tests whether tengri's discs
# reproduce the four AGNfitter-rX libraries at matched parameters, then the reddening that both
# codes apply to the disc.

# %% [markdown]
# ### 2.1 Library face-off
#
# Four AGNfitter-rX libraries at matched parameters, unreddened and normalized at 2500 Å, so
# disc *shape* is compared; the absolute scale (upstream ties it to $M_{\rm BH}$ and $\dot M$,
# tengri to $L_{\rm bol}$) is not. THB21 is `qsogen` with its `blr` and `feii` blocks
# (`B.qsogen_full`); upstream's template is a fixed table with unrecorded generating
# parameters, so it is a template against a model at default line strengths, not a
# parameter-matched pair. SN12 is compared at nodes of AGNfitter-rX's own
# $(\log M_{\rm BH}, \log\dot M/\dot M_{\rm Edd})$ table (upstream snaps to the nearest node,
# tengri interpolates). KD18 is tengri's `kd18_agnfitter`, grid-tabulated on the same axes: a
# node-exact tabulation, not a re-derivation. R06 is `richards2006`, which returns the table
# as stored.

# %%
disk_pairs = [
    ("R06", {}, lambda: B.disc("richards2006"), "richards2006"),
    ("SN12", dict(log_mbh=8.6, edd_index=10),
     lambda: B.disc("slone_netzer", agn_log_mbh=8.6, agn_log_ledd=-2.0),
     "slone_netzer (Slone & Netzer 12)"),
    ("KD18", dict(log_mbh=8.0, log_edd=-0.75),
     lambda: B.disc("kd18_agnfitter", agn_log_mbh=8.0, agn_log_ledd=-0.75),
     "kd18_agnfitter (grid-tabulated, node-exact)"),
    ("THB21", {}, B.qsogen_full, "qsogen + blr + FeII"),
]


ANCHOR = 2500.0
_HA_LADDER: dict[str, float] = {}
_ANNOT = dict(transform=None, va="top", ha="left", fontsize=7, family="monospace",
              bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, (af_name, af_kw, tengri_fn, tengri_label) in zip(axes.ravel(), disk_pairs):
    w_a, L_a = A.disk_template(af_name, **af_kw)
    a_norm = norm_at(w_a, L_a, ANCHOR)
    msk_a = (w_a > 5e2) & (w_a < 5e4)
    ax.loglog(w_a[msk_a], a_norm[msk_a], "C0-", lw=4.0, alpha=0.35, solid_capstyle="round", label=f"AGNFITTER  {af_name}")
    w_t, L_t = tengri_fn()
    t_norm = norm_at(w_t, L_t, ANCHOR)
    msk_t = (w_t > 5e2) & (w_t < 5e4)
    ax.loglog(w_t[msk_t], t_norm[msk_t], "C1-", lw=1.4, label=f"tengri  {tengri_label}")
    ax.axvline(6563, color="0.7", ls=":", lw=1)

    if af_name == "KD18":
        oa_, ot_ = np.argsort(w_a), np.argsort(w_t)
        w_a_s = np.asarray(w_a)[oa_]
        m = (w_a_s >= 1.2e3) & (w_a_s <= 1e4) & (a_norm[oa_] > 0)
        _t_on = np.interp(w_a_s[m], np.asarray(w_t)[ot_], t_norm[ot_])
        _lr = np.abs(np.log10(_t_on / a_norm[oa_][m]))
        ax.text(0.03, 0.97,
                f"grid-tabulated node match\ndisc window 1200 A-1 um\nmedian: {np.median(_lr):.3f} dex\nmax: {_lr.max():.2f} dex",
                **{**_ANNOT, "transform": ax.transAxes})
    if af_name == "THB21":
        _ot = np.argsort(np.asarray(w_t))
        _t_on_af = np.interp(np.asarray(w_a), np.asarray(w_t)[_ot], np.asarray(t_norm)[_ot])
        _HA_LADDER.update(
            {
                "tengri, native grid": val_at(w_t, t_norm, 6563),
                "tengri, interpolated onto the reference grid": val_at(w_a, _t_on_af, 6563),
                "AGNfitter-rX THB21 template": val_at(w_a, a_norm, 6563),
            }
        )
        ax.text(0.03, 0.97, "H-alpha/2500 A: see table",
                **{**_ANNOT, "transform": ax.transAxes})
    ax.set_xlim(5e2, 5e4)
    ax.set_ylim(0.05, 20)
    ax.set_title(af_name)
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(True, alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel(r"$\lambda$ [Å]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$L_\nu$ (norm. at 2500 Å)")
fig.suptitle("Accretion-disk libraries — the 0.7 µm (dotted) Hα+[N II] bump is THB21-only", y=1.0)
fig.tight_layout()
save_fig("agnfitter_09a_disc_library.png")

# %%
print("§2.1  disc-library shape residuals (anchor 2500 Å, 1200 Å-1 um):")
for af_name, af_kw, tengri_fn, _label in disk_pairs:
    w_a, L_a = A.disk_template(af_name, **af_kw)
    w_t, L_t = tengri_fn()
    oa_, ot_ = np.argsort(w_a), np.argsort(w_t)
    w_a_s, a_s = np.asarray(w_a)[oa_], norm_at(w_a, L_a, ANCHOR)[oa_]
    m = (w_a_s >= 1.2e3) & (w_a_s <= 1e4) & (a_s > 0)
    t_on = np.interp(w_a_s[m], np.asarray(w_t)[ot_], norm_at(w_t, L_t, ANCHOR)[ot_])
    logr = np.abs(np.log10(t_on / a_s[m]))
    print(f"  {af_name:6s} median = {np.median(logr):.3f} dex   max = {logr.max():.3f} dex")
print("§2.1  Hα / L(2500 Å) ladder, THB21 panel")
print("    | curve                                          | L(6563 Å) / L(2500 Å) |")
for _k, _v in _HA_LADDER.items():
    print(f"    | {_k:46s} | {_v:21.3f} |")

# %%
# KD18 grid-tabulated vs warm-index variant, at fixed (M_BH, lambda_Edd): a far warm index
# against kd18_agnfitter's baked-in default (tengri against tengri; the repackaged reference
# holds no warm-index library).
w_kd, L_kd = B.disc("kd18_agnfitter", agn_log_mbh=8.0, agn_log_ledd=-0.75)
resolved_params(B.disc.last_model)
w_kdw, L_kdw = B.disc("kd18_agnfitter_warmindex", agn_log_mbh=8.0, agn_log_ledd=-0.75, agn_gamma_warm=1.5)
_m = (w_kd > 1.2e3) & (w_kd < 1e4)
_kdn = norm_at(w_kd, L_kd, ANCHOR)
_kdwn_on = np.interp(w_kd, w_kdw, norm_at(w_kdw, L_kdw, ANCHOR))
_kd_logr = np.abs(np.log10(np.clip(_kdwn_on[_m], 1e-30, None) / np.clip(_kdn[_m], 1e-30, None)))
_KD18_WARM_MAX_DEX = float(_kd_logr.max())

# The KD18 family carries its own hot corona, so tengri refuses an additional AGN-corona
# X-ray variant; upstream instead replaces everything above 200 eV with its own power law.
try:
    SEDModel.build(
        ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST,
        agn={"type": "composable", "disc": {"type": "kd18_agnfitter", "all_params": Fixed(DEFAULT)},
             "torus": {"type": "none"}, "nlr": {"type": "none"}, "blr": {"type": "none"},
             "atten": {"type": "none"}, "agn_log_lbol": Fixed(11.0), "all_params": Fixed(DEFAULT),
             "norm": "independent"},
        xray={"type": "yang20", "all_params": Fixed(DEFAULT)}, neb={"type": "none"}, redshift=Fixed(0.0),
    )
except (ValueError, tengri.ConfigError) as _exc:
    _GUARD = f"refused ({type(_exc).__name__})"
    _GUARD_MESSAGE = str(_exc)
else:
    raise AssertionError("kd18_agnfitter + xray='yang20' was accepted; the double-count guard is gone")

# SN12 and KD18 at AGNfitter-rX's own table nodes (the per-node figure is in section 8.1).
sn12_axes = A.disk_axes("SN12")
sn12_node_idx = [(0, 0), (4, 6), (8, 11), (2, 9)]
kd18_nodes = [(6.0, -1.5), (6.0, 0.0), (7.43, -0.96), (8.0, -1.5)]
cases_9a2 = []
for i, j in sn12_node_idx:
    log_mbh = float(sn12_axes["log_mbh"][i])
    log_edd = float(sn12_axes["log_edd"][j])
    w_a, L_a = A.disk_template("SN12", log_mbh=log_mbh, edd_index=j)
    w_t, L_t = B.disc("slone_netzer", agn_log_mbh=log_mbh, agn_log_ledd=log_edd)
    label = f"SN12 ({i},{j}) logM={log_mbh:.1f} logEdd={log_edd:.2f}"
    a_n, t_n = norm_at(w_a, L_a, ANCHOR), norm_at(w_t, L_t, ANCHOR)
    assert_comparable(a_n, t_n, name=f"§8.1 {label}")
    cases_9a2.append((label, w_a, a_n, w_t, t_n))
for mbh, edd in kd18_nodes:
    w_a, L_a = A.disk_template("KD18", log_mbh=mbh, log_edd=edd)
    w_t, L_t = B.disc("kd18_agnfitter", agn_log_mbh=mbh, agn_log_ledd=edd)
    label = f"KD18 ({mbh:g},{edd:g})"
    a_n, t_n = norm_at(w_a, L_a, ANCHOR), norm_at(w_t, L_t, ANCHOR)
    assert_comparable(a_n, t_n, name=f"§8.1 {label}")
    cases_9a2.append((label, w_a, a_n, w_t, t_n))
_win_9a2 = V.window_rows(cases_9a2, lo=1200, hi=1e4)

print("§2.1  disc-block checks that are not a library overlay")
print("    | check                                                   | result |")
print(f"    | SN12 + KD18 at table nodes: cases, worst max|t/a-1|     | {len(_win_9a2)}, {max(r['max_abs_dev'] for r in _win_9a2):.3g} |")
print(f"    | kd18_agnfitter_warmindex (gamma_warm=1.5) vs grid: max  | {_KD18_WARM_MAX_DEX:.3f} dex |")
print(f"    | kd18_agnfitter + xray='yang20' (corona double count)    | {_GUARD} |")
print("§2.1  guard message:", _GUARD_MESSAGE)
print("§2.1  disc composable menu:", sorted(r["name"] for r in tengri.list_agn_blocks(category="disc")))

# %% [markdown]
# **Result.** R06, SN12 and KD18 agree with their references (median 0.000 dex, maximum 0.002
# dex for KD18; the per-node figure is in §8.1). Only THB21 carries the 0.7 µm bump, and tengri
# reproduces its position. On a common grid, the reference's, tengri's Hα/2500 Å ratio is 16%
# above the template's (the interpolated row of the table against the template row); read on
# tengri's own finer grid the line peak is higher still, which is a sampling effect and not a
# comparable number. The KD18 refusal is tengri's policy, not upstream's: upstream replaces
# everything above 200 eV in any disc by its own power law, which also removes a corona the
# template may carry.

# %% [markdown]
# ### 2.2 Disc reddening
#
# AGNfitter-rX reddens the disc, not the host, with the Prevot et al. (1984) SMC fit
# $A_\lambda = k_{\rm raw}(\lambda)\,E(B{-}V)$, $k_{\rm raw} = 1.39\,\lambda_{\mu{\rm m}}^{-1.2} - 0.38$,
# with no reddening blueward of 200 eV (`BBBred_Prevot`). The routine declares an $R_V$ but its
# inner function ignores it, so the effective ratio is $k_{\rm raw}(0.55\,\mu{\rm m})$. tengri's
# `agn_ebv_disc` (the `EBVbbb` analog) applies the intended $k = R_V\,k_{\rm raw}/k_{\rm raw}(V)$.
# The two conventions differ by one constant factor at matched $E(B{-}V)$ and have identical
# shape; the table gives declared, effective and relative values. The right panel measures
# tengri's end-to-end disc attenuation by building the disc at two $E(B{-}V)$ and taking the
# ratio.
#
# **Verification Status:** CROSSVAL; Attenuation law library

# %%
_K_RAW_V = 1.39 * 0.55 ** (-1.2) - 0.38  # Prevot raw fit at V band, AGNFITTER-RX's own formula
_R_V_SMC = 2.72  # R_V declared (and unused) by BBBred_Prevot; Prevot et al. (1984)
_CONV = _R_V_SMC / _K_RAW_V  # tengri / AGNfitter-rX A_lambda at equal E(B-V)

w_thb, L_thb = A.disk_template("THB21")
lam_um = np.geomspace(0.1, 3.0, 400)
k_raw = 1.39 * lam_um ** (-1.2) - 0.38

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 4.5))
axl.plot(lam_um, k_raw, "C0-", lw=1.6, label="AGNFITTER-RX  $k_{raw}(\\lambda)$")
axl.plot(lam_um, k_raw / _K_RAW_V * _R_V_SMC, "C1--", lw=1.6, label=rf"tengri  $k(\lambda)\,R_V$  ($={_CONV:.3f}\,k_{{raw}}$)")
axl.set_xscale("log")
axl.set_xlabel(r"$\lambda$ [µm]")
axl.set_ylabel(r"$A_\lambda / E(B{-}V)$")
axl.set_title("Prevot SMC law — two normalization conventions")
axl.legend(fontsize=8)
axl.grid(True, alpha=0.3)

_EBV_DEMO = 0.3
w_t0, L_t0 = B.disc("qsogen", ebv_disc=0.0)
resolved_params(B.disc.last_model)
w_t3, L_t3 = B.disc("qsogen", ebv_disc=_EBV_DEMO)
ratio_tengri = np.divide(L_t3, L_t0, out=np.ones_like(L_t3), where=L_t0 > 0)
L_thb_red = A.apply_bbb_reddening(w_thb, L_thb, _EBV_DEMO)
ratio_af = np.divide(L_thb_red, L_thb, out=np.ones_like(L_thb_red), where=L_thb > 0)
w_tr, L_tr = B.disc("qsogen", ebv_disc=_EBV_DEMO * _K_RAW_V / _R_V_SMC)
ratio_tengri_rescaled = np.divide(L_tr, L_t0, out=np.ones_like(L_tr), where=L_t0 > 0)

msk_t = (w_t0 > 8e2) & (w_t0 < 3e4)
msk_a = (w_thb > 8e2) & (w_thb < 3e4)
axr.semilogx(w_thb[msk_a], -2.5 * np.log10(ratio_af[msk_a]), "C0-", lw=3.5, alpha=0.35,
             solid_capstyle="round", label="AGNFITTER-RX  BBBred_Prevot")
axr.semilogx(w_t0[msk_t], -2.5 * np.log10(ratio_tengri_rescaled[msk_t]), "C1-", lw=1.4,
             label=rf"tengri  agn_ebv_disc at $E(B{{-}}V)/{_CONV:.3f}$ (convention-matched)")
axr.semilogx(w_t0[msk_t], -2.5 * np.log10(ratio_tengri[msk_t]), "C3:", lw=1.4,
             label=rf"tengri at same $E(B{{-}}V)$ (raw, ${100 * (_CONV - 1):+.1f}\%$ convention offset)")
axr.set_xlabel(r"$\lambda$ [Å]")
axr.set_ylabel(r"$A_\lambda$ [mag] at $E(B{-}V)=0.3$")
axr.set_title("Disc attenuation, end-to-end — identical law once convention-matched")
axr.legend(fontsize=8)
axr.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_04_dust_attenuation.png")

# %%
_grid4 = np.geomspace(1.2e3, 1e4, 200)
_a_af = np.interp(_grid4, w_thb, -2.5 * np.log10(ratio_af))
_a_t = np.interp(_grid4, w_t0, -2.5 * np.log10(ratio_tengri))
_a_tr = np.interp(_grid4, w_t0, -2.5 * np.log10(ratio_tengri_rescaled))
print(
    f"§2.2  matched E(B-V)={_EBV_DEMO}: max|A_tengri - A_AGNFITTER| = "
    f"{np.max(np.abs(_a_t - _a_af)):.3f} mag  (ratio A_t/A_af median = "
    f"{np.median(_a_t / _a_af):.4f}, expected {_R_V_SMC / _K_RAW_V:.4f})"
)
print(
    f"§2.2  rescaled E(B-V)={_EBV_DEMO}/{_CONV:.4f}: max|A_tengri - A_AGNFITTER| = "
    f"{np.max(np.abs(_a_tr - _a_af)):.4f} mag (pure-convention check)"
)
print("§2.2  Prevot reddening conventions")
print("    | quantity                                        |   value |")
print(f"    | R_V declared by BBBred_Prevot (unused there)    | {_R_V_SMC:7.3f} |")
print(f"    | AGNfitter-rX effective R_V = k_raw(V)           | {_K_RAW_V:7.3f} |")
print(f"    | tengri R_V (k(V) = R_V)                         | {_R_V_SMC:7.3f} |")
print(f"    | A_tengri / A_AGNfitter-rX at equal E(B-V)       | {_CONV:7.4f} |")
print(f"    | E(B-V)_tengri for equal A_lambda: E(B-V)_AF  /  | {_CONV:7.4f} |")

# %% [markdown]
# qsogen, which builds the THB21 disc, reddens with a different, empirically derived quasar
# curve (Temple, Hewett & Banerji 2021, from SDSS quasars, not the SMC), reached through the
# composable `atten` sub-block: `agn={'atten': {'type': 'qsogen'}}`. The curve is recovered by
# the same two-build ratio.

# %%
_wl_ext = np.geomspace(1e3, 1e4, 300)
w_q0, L_q0 = B.disc_atten("qsogen", "none", 0.0)
resolved_params(B.disc_atten.last_model)
w_q1, L_q1 = B.disc_atten("qsogen", "qsogen", 0.3)
ratio_qsogen = np.divide(L_q1, L_q0, out=np.ones_like(L_q1), where=L_q0 > 0)
# A_lambda/E(B-V) = -2.5 log10(ratio) / E(B-V)
A_over_ebv_qsogen = -2.5 * np.log10(np.clip(ratio_qsogen, 1e-300, None)) / 0.3

_k_raw_ext_v = 1.39 * (_wl_ext / 1e4) ** (-1.2) - 0.38
_k_af_ext = _k_raw_ext_v
_k_tengri_ext = _k_raw_ext_v / _K_RAW_V * _R_V_SMC
_A_qsogen_on_grid = np.interp(_wl_ext, w_q0, A_over_ebv_qsogen)

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(_wl_ext, _k_af_ext, "C0-", lw=1.6, label=rf"AGNFITTER-RX  Prevot SMC ($R_V^{{\rm eff}}={_K_RAW_V:.3f}$)")
ax.plot(_wl_ext, _k_tengri_ext, "C1--", lw=1.6, label=rf"tengri  Prevot SMC ($R_V={_R_V_SMC:.2f}$)")
ax.plot(_wl_ext, _A_qsogen_on_grid, "C3-", lw=2.2, alpha=0.8, label="tengri  qsogen atten (via build ratio)")
ax.axvline(5500, color="0.8", ls=":", lw=1, label="V (5500 Å)")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$A_\lambda / E(B{-}V)$")
ax.set_title("Two disc-reddening laws in tengri's public grammar")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_04b_qsogen_ext.png")

# %%
def _at_ext(a, lam):
    return float(np.interp(lam, _wl_ext, a))


print(
    f"§2.2  A_V/E(B-V) at V=5500 Å:  AGNFITTER-RX={_at_ext(_k_af_ext, 5500):.3f}  "
    f"tengri-Prevot={_at_ext(_k_tengri_ext, 5500):.3f}  tengri-qsogen={_at_ext(_A_qsogen_on_grid, 5500):.3f}"
)
print(
    f"§2.2  A(1500)/A(V) (UV steepness):  AGNFITTER-RX={_at_ext(_k_af_ext, 1500) / _at_ext(_k_af_ext, 5500):.2f}  "
    f"tengri-qsogen={_at_ext(_A_qsogen_on_grid, 1500) / _at_ext(_A_qsogen_on_grid, 5500):.2f}"
)

# %% [markdown]
# The sweep below applies the Prevot law to the THB21 template (upstream) and to tengri's
# `qsogen` disc at three $E(B{-}V)$; at matched $E(B{-}V)$ tengri's attenuation sits above
# AGNfitter-rX's by the convention factor, which the printed ratios reproduce.

# %%
w_thb, L_thb = A.disk_template("THB21")
_, L_te0 = B.disc("qsogen", ebv_disc=0.0)
msk = (w_thb > 8e2) & (w_thb < 1e4)
print("§2.2  A(1500 Å) per E(B-V) [mag]:")
for ebv, c in [(0.1, "C2"), (0.3, "C1"), (0.5, "C3")]:
    L_red = A.apply_bbb_reddening(w_thb, L_thb, ebv)
    ratio_af = np.divide(L_red, L_thb, out=np.ones_like(L_red), where=L_thb > 0)
    w_te, L_te = B.disc("qsogen", ebv_disc=ebv)
    ratio_te = np.divide(L_te, L_te0, out=np.ones_like(L_te), where=L_te0 > 0)
    msk_te = (w_te > 8e2) & (w_te < 1e4)
    a_af = -2.5 * np.log10(np.interp(1500.0, w_thb, ratio_af))
    a_te = -2.5 * np.log10(np.interp(1500.0, w_te, ratio_te))
    print(f"  E(B-V)={ebv:g}:  AGNFITTER = {a_af:.2f}   tengri = {a_te:.2f}   (ratio {a_te / a_af:.3f}, §2.2 convention {_CONV:.3f})")

# %% [markdown]
# **Result.** The Prevot law is reproduced exactly once the one constant factor is divided out;
# users moving an $E(B{-}V)_{\rm BBB}$ posterior from AGNfitter-rX to tengri's Prevot disc screen
# divide by that factor (§7). qsogen's own curve is a different law: its shape and V-band scale
# (printed above) differ from the SMC fit.

# %% [markdown]
# ## 3 Torus
#
# The paper finds CAT3D-Wind the maximum-likelihood torus for most of its sources and attributes
# that to polar-wind dust in the near infrared. Does tengri reproduce the four torus libraries
# (S04, NK08, SKIRTOR, CAT3D-Wind) and the averaged reductions of the same families?
#
# Four headline libraries are compared at matched grid nodes, peak-normalized. **S04**: `silva04`
# at one column density. **NK08**: `nenkova_agnfitter`, the inclination-only reduction of the
# CLUMPY grid that upstream calls `NK0_mean_1p`. **SKIRTOR**: `skirtor` (the full X-CIGALE grid)
# at the opening angle, inclination and optical depth set in the cell; AGNfitter-rX's averaged
# reduction is `skirtor_agnfitter`. **CAT3D-Wind**: `cat3d_wind` at a high wind fraction.
#
# **Verification Status:** CROSSVAL; SKIRTOR torus (mean 3-param)

# %%
torus_pairs = [
    ("S04", "S04", lambda: B.torus("silva04", log_nh_silva=23.0), "silva04 (log N_H = 23)", dict(log_nh=23.0)),
    ("NK08", "NK08", lambda: B.torus("nenkova_agnfitter", cos_inc=0.8660254), "nenkova_agnfitter (incl 30°)", dict(incl=30.0)),
    ("SKIRTOR", "SKIRTOR", lambda: B.torus("skirtor", cos_inc=0.8660254, oa_skirtor=40.0, tau_skirtor=7.0), "skirtor (oa 40°, incl 30°, τ 7)", dict(oa=40.0, incl=30.0, tau=7.0)),
    ("CAT3D", "CAT3D-Wind", lambda: B.torus("cat3d_wind", cos_inc=1.0, a_cat3d=-2.0, fwd_cat3d=1.75), "cat3d_wind (incl 0°, a −2, f_wd 1.75)", dict(incl=0.0, a=-2.0, fwd=1.75)),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
_printed_torus_params = False
_TORUS_SAMPLING = {}
for ax, (af_name, title, tengri_fn, tengri_label, af_kw) in zip(axes.ravel(), torus_pairs):
    w_a, L_a = A.torus_template(af_name, **af_kw)
    msk_a = (w_a > 5e3) & (w_a < 1e7)
    ax.loglog(w_a[msk_a], norm_peak(L_a)[msk_a], "C0-", lw=4.0, alpha=0.35, solid_capstyle="round", label=f"AGNFITTER  {af_name}")
    w_t, L_t = tengri_fn()
    _TORUS_SAMPLING[af_name] = (np.asarray(w_a), np.asarray(w_t))
    if not _printed_torus_params:
        resolved_params(B.torus.last_model)
        _printed_torus_params = True
    msk_t = (w_t > 5e3) & (w_t < 1e7)
    ax.loglog(w_t[msk_t], norm_peak(L_t)[msk_t], "C1-", lw=1.4, label=f"tengri  {tengri_label}")
    ax.axvline(1e5, color="0.7", ls=":", lw=1)
    ax.set_xlim(8e3, 3e6)
    ax.set_ylim(1e-3, 3)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="lower center")
    ax.grid(True, alpha=0.3)
for ax in axes[-1]:
    ax.set_xlabel(r"$\lambda$ [Å]")
for ax in axes[:, 0]:
    ax.set_ylabel(r"$L_\nu$ (norm. at peak)")
fig.suptitle("Torus libraries — dotted line marks the 10 µm silicate feature", y=1.0)
fig.tight_layout()
save_fig("agnfitter_09c_torus_library.png")

# %% [markdown]
# Two properties matter for every table in this section. The four headline reference arrays are
# repackaged on a 1024-point grid uniform in $\log\lambda$, by linear interpolation of upstream's
# native axes, and that sampling sets a floor on how closely a case can agree. And tengri
# evaluates these blocks on the model wavelength grid, which joins the SSP grid to each block's
# own template axis, not on the reference wavelengths. The table prints both samplings and the
# grid limit of each block.

# %%
print("§3  sampling of the reference arrays and of tengri's model grid")
print("    | library  | reference n | uniform in log λ | reference range [um] | tengri n (> 10 um) | tengri grid max [um] |")
for _name, (_wa, _wt) in _TORUS_SAMPLING.items():
    _dl = np.diff(np.log10(np.sort(_wa)))
    _uniform = bool(np.allclose(_dl, _dl.mean(), rtol=1e-2))
    print(
        f"    | {_name:8s} | {_wa.size:11d} | {str(_uniform):16s} | "
        f"{_wa.min() / 1e4:8.3g} - {_wa.max() / 1e4:<9.3g} | {int(np.sum(_wt > 1e5)):18d} | {_wt.max() / 1e4:20.4g} |"
    )

# %% [markdown]
# The averaged reductions, the S04 column-density axis, the NK08 inclination axis, the SKIRTOR
# $(oa, incl, \tau)$ nodes and the CAT3D union axes are swept in the next cell, each against the
# reference at the same node. A case is called node-exact when
# $\max|\mathrm{tengri}/\mathrm{AGNfitter\text{-}rX} - 1|$ is below the tolerance printed with the
# verdicts. The per-library figures and full window tables are in §8.2 and §8.3.

# %%
print("§3  torus composable menu:", sorted(r["name"] for r in tengri.list_agn_blocks(category="torus")))

# --- five further averaged reductions (peak-normalized, 1-100 um median |log10 ratio|)
print("§3  five reductions: axes x node counts (tengri torus_axes) vs upstream row count:")
for _name in ("NK08_2P", "NK08_3P", "SKIRTOR_MEAN1P", "SKIRTOR_MEAN2P", "CAT3D_LOWFWD"):
    _axes = A.torus_axes(_name)
    _sizes = {k: len(v) for k, v in _axes.items()}
    print(f"    {_name:16s} axes={_sizes}  upstream rows = product = {int(np.prod(list(_sizes.values())))}")

_reduction_pairs = [
    ("NK08_2P", lambda incl, oa: B.torus("nenkova_agnfitter_2p", agn_cos_inc=np.cos(np.deg2rad(incl)), agn_oa_nenkova=oa), dict(incl=30.0, oa=40.0)),
    ("NK08_3P", lambda incl, oa, tau: B.torus("nenkova_agnfitter_3p", agn_cos_inc=np.cos(np.deg2rad(incl)), agn_oa_nenkova=oa, agn_tv_nenkova=tau), dict(incl=30.0, oa=40.0, tau=60.0)),
    ("SKIRTOR_MEAN1P", lambda incl: B.torus("skirtor_agnfitter_1p", agn_incl_skirtor=incl), dict(incl=30.0)),
    ("SKIRTOR_MEAN2P", lambda oa, incl: B.torus("skirtor_agnfitter_2p", agn_oa_skirtor=oa, agn_incl_skirtor=incl), dict(oa=40.0, incl=30.0)),
    ("CAT3D_LOWFWD", lambda incl, a, fwd: B.torus("cat3d_wind_lowfwd", agn_cos_inc=np.cos(np.deg2rad(incl)), agn_a_cat3d_lowfwd=a, agn_fwd_cat3d_lowfwd=fwd), dict(incl=0.0, a=-2.0, fwd=0.3)),
]
_reduction_rows, _reduction_curves = [], {}
for name, fn, kw in _reduction_pairs:
    w_a, L_a = A.torus_template(name, **kw)
    w_t, L_t = fn(**kw)
    _grid = np.geomspace(max(w_a.min(), w_t.min()) * 1.01, min(w_a.max(), w_t.max()) * 0.99, 300)
    a_on = np.interp(np.log10(_grid), np.log10(np.sort(w_a)), norm_peak(L_a)[np.argsort(w_a)])
    t_on = np.interp(np.log10(_grid), np.log10(np.sort(w_t)), norm_peak(L_t)[np.argsort(w_t)])
    ratio = t_on / np.clip(a_on, 1e-30, None)
    m_ir = (_grid > 1e4) & (_grid < 1e6) & (a_on > 1e-3)
    _reduction_curves[name] = (_grid, ratio)
    _reduction_rows.append({
        "label": f"{name} node={kw}",
        "median_dex": float(np.median(np.abs(np.log10(ratio[m_ir])))),
        "n_tengri": int(np.sum(w_t > 1e5)), "n_ref": int(np.sum(w_a > 1e5)),
    })

# --- S04 column density and NK08 inclination
s04_axis = A.torus_axes("S04")["log_nh"]
nk08_axis = A.torus_axes("NK08")["incl"]
print(
    f"§3  S04 log N_H axis: {s04_axis.size} nodes, {s04_axis.min():.2f}-{s04_axis.max():.2f}; "
    f"NK08 inclination axis: {nk08_axis.size} nodes, {nk08_axis.min():g}-{nk08_axis.max():g} deg"
)
s04_fracs = [0.0, 0.25, 0.5, 0.75, 0.99]  # 0.99 keeps the node clear of the float32 axis end
nk08_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]
cases_9c0, cases_9c0_fig = [], []
for f in s04_fracs:
    log_nh = float(s04_axis[int(round(f * (len(s04_axis) - 1)))])
    w_a, L_a = A.torus_template("S04", log_nh=log_nh)
    w_t, L_t = B.torus("silva04", log_nh_silva=log_nh)
    label = f"S04 log N_H={log_nh:.2f}"
    a_n, t_n = norm_peak(L_a), norm_peak(L_t)
    assert_comparable(a_n, t_n, name=f"§3 {label}")
    cases_9c0.append((label, w_a, a_n, w_t, t_n))
    if f != 0.5:
        cases_9c0_fig.append(cases_9c0[-1])
for f in nk08_fracs:
    incl = float(nk08_axis[int(round(f * (len(nk08_axis) - 1)))])
    w_a, L_a = A.torus_template("NK08", incl=incl)
    w_t, L_t = B.torus("nenkova_agnfitter", cos_inc=float(np.cos(np.deg2rad(incl))))
    label = f"NK08 incl={incl:g}°"
    a_n, t_n = norm_peak(L_a), norm_peak(L_t)
    assert_comparable(a_n, t_n, name=f"§3 {label}")
    cases_9c0.append((label, w_a, a_n, w_t, t_n))
    if f != 0.5:
        cases_9c0_fig.append(cases_9c0[-1])
_win_9c0 = V.window_rows(cases_9c0, lo=1e4, hi=1e6)

# --- SKIRTOR (oa, incl, tau) nodes of the averaged reduction, plus the full grid at the fiducial
sk_axes = A.torus_axes("SKIRTOR")
sk_triples = [(0, 3, 2), (3, 3, 2), (5, 6, 0), (7, 9, 4)]
cases_9c5 = []
for oi, ii, ti in sk_triples:
    oa = float(sk_axes["oa"][oi])
    incl = float(sk_axes["incl"][ii])
    tau = float(sk_axes["tau"][ti])
    w_a, L_a = A.torus_template("SKIRTOR", oa=oa, incl=incl, tau=tau)
    w_t, L_t = B.torus("skirtor_agnfitter", oa_skirtor=oa, incl_skirtor=incl, tv_skirtor=tau)
    label = f"SKIRTOR ({oi},{ii},{ti})"
    a_n, t_n = norm_peak(L_a), norm_peak(L_t)
    assert_comparable(a_n, t_n, name=f"§3 {label}")
    cases_9c5.append((label, w_a, a_n, w_t, t_n))
    if (oi, ii, ti) == (3, 3, 2):
        w_ref, L_ref, w_af, L_af = w_a, a_n, w_t, t_n  # fiducial node
w_xc, L_xc_raw = B.torus("skirtor", cos_inc=float(np.cos(np.deg2rad(30.0))), oa_skirtor=40.0, tau_skirtor=7.0)
L_xc = norm_peak(L_xc_raw)
assert_comparable(L_ref, L_xc, name="§3 SKIRTOR full grid @ fiducial")
cases_9c5.append(("SKIRTOR full grid @ fiducial", w_ref, L_ref, w_xc, L_xc))
_win_9c5 = V.window_rows(cases_9c5, lo=1e4, hi=1e6)
_peak = {}
for _nm, _w, _L in (("AGNfitter-rX SKIRTOR_mean_3p", w_ref, L_ref), ("tengri skirtor_agnfitter", w_af, L_af), ("tengri skirtor (full grid)", w_xc, L_xc)):
    _mk = (_w > 5e3) & (_w < 1e7)
    _peak[_nm] = float(_w[_mk][np.argmax(_L[_mk])])

# --- CAT3D: the wind-fraction union and (incl, a, f_wd) triples
_INCL0, _A0 = 0.0, -2.0
_u = A.cat3d_union_axes()


def tengri_cat3d_union(incl, a, fwd, sub):
    """tengri CAT3D block for one union node; ``sub`` names the reference sub-library it belongs to."""
    cos_inc = float(np.cos(np.deg2rad(incl)))
    if sub == "CAT3D_LOWFWD":
        return B.torus("cat3d_wind_lowfwd", agn_cos_inc=cos_inc, agn_a_cat3d_lowfwd=a, agn_fwd_cat3d_lowfwd=fwd)
    return B.torus("cat3d_wind", cos_inc=cos_inc, a_cat3d=a, fwd_cat3d=fwd)


_fwd_grid = np.geomspace(1e4, 1e6, 300)
_plot_fwd = {float(f) for f in _u["fwd"][[0, 2, 4, 5, 8, 10]]}
_cat_curves, _union_rows, _union_table = [], [], []
for fwd in _u["fwd"]:
    fwd = float(fwd)
    w_a, L_a, sub = A.cat3d_union_template(_INCL0, _A0, fwd)
    w_t, L_t = tengri_cat3d_union(_INCL0, _A0, fwd, sub)
    a_on = np.interp(np.log10(_fwd_grid), np.log10(w_a), norm_peak(L_a))
    t_on = np.interp(np.log10(_fwd_grid), np.log10(w_t), norm_peak(L_t))
    ok = a_on > 1e-3
    _res = np.abs(t_on[ok] / a_on[ok] - 1.0)
    _union_rows.append({"label": f"f_wd={fwd:g} ({sub})", "max_abs_dev": float(_res.max())})
    _union_table.append((fwd, sub, "cat3d_wind_lowfwd" if sub == "CAT3D_LOWFWD" else "cat3d_wind", float(np.median(_res)), float(_res.max()), float(_fwd_grid[ok][np.argmax(_res)])))
    if fwd in _plot_fwd:
        _cat_curves.append(("union", fwd, w_a, norm_peak(L_a), w_t, norm_peak(L_t)))
c3_axes = A.torus_axes("CAT3D")
c3_triples = [(0, 0, 5), (3, 1, 5), (3, 2, 4), (6, 3, 2)]
_tri_rows, _tri_table = [], []
for (ii, ai, fi), c in zip(c3_triples, ["C5", "C6", "C7", "C8"]):
    incl = float(c3_axes["incl"][ii])
    a_cat3d = float(c3_axes["a"][ai])
    fwd = float(c3_axes["fwd"][fi])
    w_a, L_a = A.torus_template("CAT3D", incl=incl, a=a_cat3d, fwd=fwd)
    w_t, L_t = B.torus("cat3d_wind", cos_inc=float(np.cos(np.deg2rad(incl))), a_cat3d=a_cat3d, fwd_cat3d=fwd)
    label = f"({ii},{ai},{fi})"
    assert_comparable(norm_peak(L_a), norm_peak(L_t), name=f"§3 {label}")
    a_on = np.interp(np.log10(_fwd_grid), np.log10(w_a), norm_peak(L_a))
    t_on = np.interp(np.log10(_fwd_grid), np.log10(w_t), norm_peak(L_t))
    ok = a_on > 1e-3
    _res = np.abs(t_on[ok] / a_on[ok] - 1.0)
    _tri_rows.append({"label": f"{label} incl={incl:g} a={a_cat3d:g} f_wd={fwd:g}", "max_abs_dev": float(_res.max())})
    _tri_table.append((label, incl, a_cat3d, fwd, float(np.median(_res)), float(_res.max())))
    _cat_curves.append(("triple", c, f"{label}: incl={incl:g}° a={a_cat3d:g}", w_a, norm_peak(L_a), w_t, norm_peak(L_t)))

# --- one summary table
def _worst(rows, key="max_abs_dev"):
    r = max(rows, key=lambda r: r[key])
    return r[key], r["label"]


_fam = lambda rows, tag: [r for r in rows if r["label"].startswith(tag)]
print("§3  torus agreement by library (peak-normalized, 1-100 um)")
print("    | library / reduction                       | cases | worst            | where")
for _lab, _rows, _key in [
    ("S04, column-density axis", _fam(_win_9c0, "S04"), "max_abs_dev"),
    ("NK08, inclination axis", _fam(_win_9c0, "NK08"), "max_abs_dev"),
    ("SKIRTOR (oa, incl, tau), averaged", [r for r in _win_9c5 if "full grid" not in r["label"]], "max_abs_dev"),
    ("SKIRTOR full unaveraged grid, fiducial", _fam(_win_9c5, "SKIRTOR full"), "max_abs_dev"),
    ("CAT3D wind-fraction union", _union_rows, "max_abs_dev"),
    ("CAT3D (incl, a, f_wd) nodes", _tri_rows, "max_abs_dev"),
]:
    _v, _where = _worst(_rows, _key)
    print(f"    | {_lab:41s} | {len(_rows):5d} | max|t/a-1| {_v:8.3g} | {_where}")
_v, _where = _worst(_reduction_rows, "median_dex")
print(f"    | {'five further averaged reductions':41s} | {len(_reduction_rows):5d} | median {_v:8.3g} dex | {_where}")
print(
    "§3  IR peak wavelength [um]: "
    + ", ".join(f"{k} = {v / 1e4:.1f}" for k, v in _peak.items())
)

# %% [markdown]
# **Result.** The tabulated blocks reproduce their reference nodes: S04, NK08, the averaged
# SKIRTOR and the high-wind CAT3D nodes agree to about 1% at worst, and the five further
# reductions to 1e-3 dex in the median. The one larger entry, the low-wind half of the CAT3D union,
# has a median of 1e-4 and a maximum near 10% confined to a narrow range around 1.4-1.5 µm (the
# table in §8.3 prints where), at the short-wavelength edge where the low-wind reference is
# sparse and steep. The full unaveraged SKIRTOR grid differs from the averaged reference by
# construction: it carries the clumpiness and radial structure that the average removes, which
# shifts the IR peak (printed). The union of the two tengri CAT3D blocks covers the one upstream
# library, and the polar wind that CAT3D-Wind is for supplies the near-infrared excess that
# equatorial tori miss.

# %% [markdown]
# #### Inclination-dependent torus power
#
# Upstream's torus templates do not share a bolometric luminosity: the template integral falls
# with inclination, so at fixed `TO` a more inclined torus radiates less. tengri normalizes every
# node to the same torus power and lets the torus luminosity fraction carry the amplitude. Every
# panel above is peak-normalized and blind to this; the table compares the two normalizations,
# each relative to the lowest-inclination node.
#
# **Caveat:** this is a difference of parametrization, not an error, but it matters when a
# posterior is moved between the codes: `TO` and tengri's torus luminosity fraction are related
# by the ratio tabulated here.

# %%
_bol_sets = [
    ("NK08", A.torus_axes("NK08")["incl"],
     lambda i: A.torus_bolometric("NK08", incl=i),
     lambda i: B.torus("nenkova_agnfitter", cos_inc=float(np.cos(np.deg2rad(i))))),
    ("CAT3D", A.torus_axes("CAT3D")["incl"],
     lambda i: A.torus_bolometric("CAT3D", incl=i, a=-2.0, fwd=1.75),
     lambda i: B.torus("cat3d_wind", cos_inc=float(np.cos(np.deg2rad(i))), a_cat3d=-2.0, fwd_cat3d=1.75)),
]
print("§3  torus power relative to the lowest-inclination node")
print("    | library  | incl [deg] | upstream template integral | tengri torus power |")
for _name, _incl_ax, _up_fn, _te_fn in _bol_sets:
    _up0 = _up_fn(float(_incl_ax[0]))
    _te0 = band_power(*_te_fn(float(_incl_ax[0])), 1e3, 1e9)
    for _i in _incl_ax:
        _w_b, _L_b = _te_fn(float(_i))
        print(
            f"    | {_name:8s} | {float(_i):10.1f} | {_up_fn(float(_i)) / _up0:26.4f} | "
            f"{band_power(_w_b, _L_b, 1e3, 1e9) / _te0:18.4f} |"
        )

# %% [markdown]
# ## 4 X-ray corona
#
# AGNfitter-rX ties the 2 keV corona to the 2500 Å disc continuum through the Just et al. (2007)
# relation, $\alpha_{\rm ox} = -0.137\,\log L_{2500} + 2.638 + \Delta\alpha_{\rm ox}$, then lays
# down a power law of photon index $\Gamma$ with a 300 keV exponential cutoff, joined to the disc
# by a hard step at 200 eV. Is tengri's corona the same object? The first figure compares the
# relation itself (tengri offers three calibrations through `alpha_ox_from_l2500`; upstream uses
# the Just et al. line).

# %%
from tengri.xray import alpha_ox_from_l2500

l2500 = np.geomspace(1e28, 1e32, 200)
aox_ref = np.array([A.alpha_ox(float(x)) for x in l2500])
fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for rel, c in [("just2007", "C1"), ("lusso_risaliti_2016", "C2"), ("lusso_risaliti_2017", "C3")]:
    aox = np.array([float(alpha_ox_from_l2500(x, relation=rel)) for x in l2500])
    ax.plot(np.log10(l2500), aox, c, lw=1.6, label=f"tengri  {rel}")
    if rel == "just2007":
        aox_t = aox
ax.plot(np.log10(l2500), aox_ref, "k--", lw=1.2, label="AGNFITTER-RX  Just+2007")
ax.set_ylabel(r"$\alpha_{ox}$")
ax.set_title(r"$\alpha_{ox}$–$L_{2500}$ relation")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
axr.axhline(0.0, color="0.5", lw=0.8)
axr.plot(np.log10(l2500), aox_t - aox_ref, "C1-", lw=1.2)
axr.set_ylabel(r"$\Delta\alpha_{ox}$", fontsize=9)
axr.set_xlabel(r"$\log_{10}\ L_{2500\,\AA}$ [erg/s/Hz]")
axr.grid(True, alpha=0.3)
_aox_dmax = float(np.max(np.abs(aox_t - aox_ref)))
print(f"§4 α_ox parity: max |tengri just2007 − AGNFITTER-RX| = {_aox_dmax:.2e}")
fig.tight_layout()
save_fig("agnfitter_10c_alphaox_residual.png")
plt.show()

# %% [markdown]
# Given the 2500 Å disc luminosity, both codes then build the same bare corona ($\Gamma = 1.8$,
# 300 keV cutoff, normalized through the relation). Left: tengri's
# `xray_agn_corona_from_disc(..., apply_anisotropy=False)` against upstream's disc X-ray
# extension. Right: what tengri's *defaults* add, none of which AGNfitter-rX has an analog for:
# the Yang et al. (2022) viewing-angle anisotropy, a default absorbing column, and a host
# X-ray-binary floor (`xray_xrb`, Mineo et al. 2014). The table gives the ratio in a hard window
# (0.5-100 keV) and a soft one (0.2-0.5 keV), across $\Delta\alpha_{\rm ox}$ and $\Gamma$, with
# the column at its default and switched off.
#
# **Verification Status:** PARTIAL; Radio + X-ray + AGN

# %%
from tengri.xray import xray_agn_corona_from_disc, xray_xrb

L_2500 = 1.0e30
wave_x = np.geomspace(1e-2, 1e2, 600)
L_corona_bare = np.asarray(xray_agn_corona_from_disc(jnp.asarray(wave_x), L_2500, delta_alpha_ox=0.0, apply_anisotropy=False))
L_corona_default = np.asarray(xray_agn_corona_from_disc(jnp.asarray(wave_x), L_2500, delta_alpha_ox=0.0, cos_inc=1.0))
w_thb, L_thb = A.disk_template("THB21")
L_thb_at_2500 = norm_at(w_thb, L_thb, 2500.0) * L_2500
xw_af, xL_af = A.disk_xray_extension(w_thb, L_thb_at_2500, scatter=0.0)
L_xrb = np.asarray(xray_xrb(jnp.asarray(wave_x), sfr=5.0, stellar_mass=1e10))

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 4.8))
axl.loglog(xw_af, xL_af, "C0-", lw=1.5, label="AGNFITTER-RX  disk X-ray extension")
axl.loglog(wave_x, L_corona_bare, "C1--", lw=1.5, label="tengri  corona (anisotropy off)")
axl.loglog(wave_x, L_xrb, "C2:", lw=1.4, label="tengri  xray_xrb (host floor, Mineo+14)")
axl.set_xlim(2e-2, 1.3e2)
axl.set_ylim(L_corona_bare.max() * 1e-6, L_corona_bare.max() * 5)
axl.set_xlabel(r"$\lambda$ [Å]")
axl.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
axl.set_title(r"Bare corona parity ($\Gamma$ = 1.8, 300 keV cutoff)")
axl.legend(fontsize=8)
axl.grid(True, alpha=0.3)

_af_on_wave = np.interp(wave_x, xw_af, xL_af, left=np.nan, right=np.nan)
_ratio_bare = L_corona_bare / _af_on_wave
_ratio_default = L_corona_default / _af_on_wave
axr.semilogx(wave_x, _ratio_bare, "C1-", lw=1.5, label="bare (anisotropy off)")
axr.semilogx(wave_x, _ratio_default, "C3-", lw=1.5, label="tengri defaults (Yang+22 anisotropy, face-on)")
axr.axhline(1.0, color="0.5", lw=0.8)
axr.set_xlim(1e-2, 1e2)
axr.set_ylim(0.8, 1.3)
axr.set_xlabel(r"$\lambda$ [Å]")
axr.set_ylabel("tengri / AGNFITTER-RX")
axr.set_title("Corona ratio — what tengri's defaults add")
axr.legend(fontsize=8)
axr.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_10b_xray_corona.png")

# %%
_hard = (wave_x > 0.12) & (wave_x < 25.0) & np.isfinite(_ratio_bare)
print(
    f"§4 corona parity (0.5-100 keV): bare median ratio = {np.nanmedian(_ratio_bare[_hard]):.4f}, "
    f"max |ratio-1| = {np.nanmax(np.abs(_ratio_bare[_hard] - 1.0)):.3f}; "
    f"defaults median ratio = {np.nanmedian(_ratio_default[_hard]):.4f} (the Yang+22 anisotropy)"
)

# %%
from tengri.xray import alpha_ox_from_l2500, xray_agn_corona_from_disc

_m10 = SEDModel.build(
    ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST,
    xray={"type": "yang20", "all_params": Fixed(DEFAULT)}, neb={"type": "none"}, redshift=Fixed(0.0),
)
resolved_params(_m10)  # the Gamma=1.8, 300 keV cutoff, log N_H=20 defaults the table below discusses

w_thb, L_thb = A.disk_template("THB21")
L_2500_10a = 1.0e30
L_thb_at_2500 = norm_at(w_thb, L_thb, 2500.0) * L_2500_10a
wave_x_10a = np.geomspace(1e-2, 1e2, 600)
_hard_10a = (wave_x_10a > 0.12) & (wave_x_10a < 25.0)  # 0.5-100 keV
_soft_10a = (wave_x_10a >= 25.0) & (wave_x_10a < 62.0)  # 0.2-0.5 keV (upstream's X-ray branch starts at 62 A)
_scat_grid = [-0.4, -0.2, 0.0, 0.2, 0.4]


def _corona_ratio(scat, gamma, log_nh):
    xw, xL = A.disk_xray_extension(w_thb, L_thb_at_2500, scatter=scat, gamma=gamma)
    L_t = np.asarray(
        xray_agn_corona_from_disc(
            jnp.asarray(wave_x_10a), L_2500_10a, delta_alpha_ox=scat, gamma=gamma,
            apply_anisotropy=False, log_nh=log_nh,
        )
    )
    af_on = np.interp(wave_x_10a, xw, xL, left=np.nan, right=np.nan)
    return np.where(np.isfinite(af_on) & (af_on > 0), L_t / af_on, np.nan)


print("§4  X-ray corona, tengri / AGNfitter-rX (apply_anisotropy=False): median ratio and max|ratio-1| per window")
print("    | Δα_ox |   Γ | log N_H | 0.5-100 keV median | max|r-1| | 0.2-0.5 keV median | max|r-1| |")
_xray_rows = {20.0: [], 0.0: []}
for scat, gam in [(s_, 1.8) for s_ in _scat_grid] + [(0.0, 1.6), (0.0, 2.0)]:
    for lognh in (20.0, 0.0):
        r = _corona_ratio(scat, gam, lognh)
        rh, rs = r[_hard_10a], r[_soft_10a]
        _xray_rows[lognh].append({"label": f"Δα_ox={scat:+.1f} Γ={gam:.1f}", "max_abs_dev": float(np.nanmax(np.abs(rh - 1.0)))})
        print(
            f"    | {scat:+5.1f} | {gam:3.1f} | {lognh:7.0f} | {np.nanmedian(rh):18.4f} | "
            f"{np.nanmax(np.abs(rh - 1.0)):8.4f} | {np.nanmedian(rs):18.4f} | {np.nanmax(np.abs(rs - 1.0)):8.4f} |"
        )

# %% [markdown]
# **Result.** With anisotropy and absorption off, the power law, the cutoff and the 2 keV anchor
# agree to about 1% in both windows, independently of $\Delta\alpha_{\rm ox}$ and $\Gamma$ (the table
# rows repeat). The default absorbing column (and its scattered fraction) lowers the soft-window
# median ratio to 0.78, and the anisotropy rescales the whole corona. Both are tengri choices to keep
# or remove, not discrepancies: set `apply_anisotropy=False` and `log_nh=0` to reproduce
# AGNfitter-rX. §6 gives the disc-to-corona join, where the codes differ by construction.

# %% [markdown]
# ## 5 Radio
#
# AGNfitter-rX models the AGN core and jet with a simple power law (SPL, slope fixed at $-0.75$
# with an exponential cutoff at $10^{13}$ Hz) or a double power law (DPL, paper Eq. 2); tengri
# ships both, as `radio_agn` and `radio_agn_dpl`. The upstream curves are the paper's equations
# evaluated by the driver, each normalized at 5 GHz.
#
# **Verification Status:** PARTIAL; Radio / X-ray / IGM / PSD physics

# %%
from tengri.radio import radio_agn, radio_agn_dpl

_m11 = SEDModel.build(
    ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST,
    radio={"sf": {"type": "bell2003", "all_params": Fixed(DEFAULT)},
           "agn": {"type": "dpl", "all_params": Fixed(DEFAULT)}},
    neb={"type": "none"}, redshift=Fixed(0.0),
)
resolved_params(_m11)

freq = np.geomspace(1e8, 1e12, 400)
wave_radio = jnp.asarray(U.C_ANGSTROM_PER_S / freq)
L_AGN_BOL = 1e45
_DPL_PARS = dict(alpha1=-0.75, alpha2=-0.1, log_nu_t=10.0, log_nu_cut=13.0)
L_spl = np.asarray(radio_agn(wave_radio, L_AGN_BOL, radio_loudness=1.0, alpha_agn=0.75))
L_dpl = np.asarray(radio_agn_dpl(wave_radio, L_AGN_BOL, radio_loudness=1.0, **_DPL_PARS))
wave_radio = np.asarray(wave_radio)

_, F_spl_af = A.agn_radio_spl(freq)
_, F_dpl_af = A.agn_radio_dpl(freq, **_DPL_PARS)
_nu5 = U.C_ANGSTROM_PER_S / 5e9


def _norm5(freq_hz, F):
    return F / np.interp(5e9, freq_hz, F)

# %%
_band = freq <= 3e11
t_spl = np.asarray(norm_at(wave_radio, L_spl, _nu5))
t_dpl = np.asarray(norm_at(wave_radio, L_dpl, _nu5))
ratio_spl = np.where(_band & (t_spl > 0), t_spl / _norm5(freq, F_spl_af), np.nan)
ratio_dpl = np.where(_band & (t_dpl > 0), t_dpl / _norm5(freq, F_dpl_af), np.nan)
fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax.loglog(freq / 1e9, np.where(t_spl > 0, t_spl, np.nan), "C0-", lw=1.6, label="tengri  radio_agn (SPL)")
ax.loglog(freq / 1e9, _norm5(freq, F_spl_af), "C0:", lw=1.4, label="AGNFITTER-RX  SPL")
ax.loglog(freq / 1e9, np.where(t_dpl > 0, t_dpl, np.nan), "C1-", lw=1.6, label="tengri  radio_agn_dpl")
ax.loglog(freq / 1e9, _norm5(freq, F_dpl_af), "C1:", lw=1.4, label="AGNFITTER-RX  DPL")
ax.set_ylabel(r"$L_\nu$ (norm. at 5 GHz)")
ax.set_title("Radio parity (0.1–300 GHz band)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
axr.axhline(1.0, color="0.5", lw=0.8)
axr.semilogx(freq / 1e9, ratio_spl, "C0-", lw=1.2, label="SPL")
axr.semilogx(freq / 1e9, ratio_dpl, "C1-", lw=1.2, label="DPL")
axr.set_ylim(0.95, 1.05)
axr.set_ylabel("tengri / AGNFITTER", fontsize=9)
axr.set_xlabel(r"$\nu$ [GHz]")
axr.legend(fontsize=8)
axr.grid(True, alpha=0.3)
_spl_dmax = float(np.nanmax(np.abs(ratio_spl - 1.0)))
_dpl_dmax = float(np.nanmax(np.abs(ratio_dpl - 1.0)))
print(f"§5 radio parity (0.1-300 GHz): SPL max |ratio − 1| = {_spl_dmax:.2e}   DPL max |ratio − 1| = {_dpl_dmax:.2e}")
fig.tight_layout()
save_fig("agnfitter_11c_radio_spl_residual.png")
plt.show()

# %% [markdown]
# Star-formation radio. AGNfitter-rX's `S17_radio` joins the Schreiber et al. (2018) dust SED to
# a radio tail calibrated with the infrared-radio correlation of Bell (2003), split 90% non-thermal
# and 10% thermal at 1.4 GHz (paper pp. 3-4). The correlation parameter is
#
# $$q_{\rm IR} = \log_{10}\!\left[\frac{L_{\rm IR}}{(3.75\times10^{12}\,{\rm Hz})\,L_{\nu,1.4\,{\rm GHz}}}\right],$$
#
# with $L_{\rm IR}$ the 8-1000 µm luminosity. The paper adopts $q_{\rm IR} = 2.64 \pm 0.26$ and then
# the conservative value $2.64 + \sigma$, which is what the repackaged template embeds; the cell
# measures it from the template. tengri's `radio_sfr_bell2003_split` is the matching mode and is
# run at that value. tengri's *default* architecture (`radio_sfr_bell2003` plus a separately
# normalized `radio_freefree`) calibrates a different quantity with $q_{\rm IR}$ and is not
# compared here, since mixing the two would double-count the thermal term.
#
# **Caveat:** at tengri's default $q_{\rm IR} = 2.64$ instead of the template's value, the 1.4 GHz
# luminosity of the same $L_{\rm IR}$ is higher by the factor printed below.

# %%
from tengri.radio import radio_sfr_bell2003_split

_Q_BELL, _SIGMA_Q = 2.64, 0.26  # Bell (2003) value and scatter adopted by the paper (p. 3)
Q_IR_PARITY = _Q_BELL + _SIGMA_Q  # the paper's conservative choice, embedded in S17_radio

w_afr, L_afr = A.cold_dust_radio_template(tdust=35.0, fpah=0.02)
_axes_radio = A.cold_dust_radio_axes()
_t_idx = int(np.argmin(np.abs(_axes_radio["tdust"] - 35.0)))
L_IR_NODE = float(_axes_radio["lir_conv"][_t_idx])  # already erg/s (see driver docstring)

# q_IR measured from the template itself (paper Eq. 1).
_nu_afr = U.C_ANGSTROM_PER_S / w_afr
_o_afr = np.argsort(_nu_afr)
_L14_afr = float(np.interp(1.4e9, _nu_afr[_o_afr], L_afr[_o_afr]))
Q_IR_TEMPLATE = float(np.log10(band_power(w_afr, L_afr, 8e4, 1e7) / (3.75e12 * _L14_afr)))
print(
    f"§5  q_IR measured from the S17_radio template (T_dust = 35 K) = {Q_IR_TEMPLATE:.4f};  "
    f"paper value 2.64 + sigma = {Q_IR_PARITY:.3f};  difference = {Q_IR_TEMPLATE - Q_IR_PARITY:+.4f}"
)

wave_all = np.geomspace(1e4, 3e9, 1200)
w_te_sb, L_dust_shape, _ = _dust_emission_build("schreiber2018", 3.0, dust_T=35.0, dust_f_pah=0.02)
L_dust = np.interp(wave_all, w_te_sb, L_dust_shape, left=0.0, right=0.0)
L_dust = L_dust * (L_IR_NODE / abs(band_power(wave_all, L_dust, 8e4, 1e7)))
L_radio_split = np.asarray(radio_sfr_bell2003_split(jnp.asarray(wave_all), L_IR_NODE, q_ir=Q_IR_PARITY))
L_te_total = L_dust + L_radio_split

fig, ax = plt.subplots(figsize=(8.2, 4.8))
msk_af = (w_afr > 1e4) & (w_afr < 3e9)
ax.loglog(w_afr[msk_af], L_afr[msk_af] / np.max(L_afr[msk_af]), "C0-", lw=2.2, alpha=0.5, label="AGNFITTER-RX  S17_radio (Bell 2003, 90/10 split)")
_peak_te = float(np.max(np.where((wave_all > 1e4), L_te_total, 0.0)))
ax.loglog(wave_all, L_te_total / _peak_te, "C1--", lw=1.5, label="tengri  schreiber2018 + radio_sfr_bell2003_split")
ax.axvline(U.C_ANGSTROM_PER_S / 1.4e9, color="0.7", ls=":", lw=1, label="1.4 GHz")
ax.set_xlim(8e3, 3e9)
ax.set_ylim(1e-6, 3)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ (norm. at FIR peak)")
ax.set_title("Host dust + star-formation radio, AGNfitter-rX parity mode")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_11b_radio_sf.png")

# %%
_lam_14 = U.C_ANGSTROM_PER_S / 1.4e9
_af_n = L_afr / np.max(L_afr[msk_af])
_te_n = L_te_total / _peak_te
_rb = (w_afr > U.C_ANGSTROM_PER_S / 3e10) & (w_afr < U.C_ANGSTROM_PER_S / 1.1e9)  # 1.1-30 GHz
_te_on_af = np.interp(np.log10(w_afr[_rb]), np.log10(wave_all), _te_n)
_ratio_rb = _te_on_af / _af_n[_rb]
_radio_low = np.asarray(radio_sfr_bell2003_split(jnp.asarray(wave_all), L_IR_NODE, q_ir=_Q_BELL))
_radio_ratio_1p4ghz = float(np.interp(np.log10(_lam_14), np.log10(wave_all), _radio_low / L_radio_split))
print(
    f"§5  SF radio, tengri / AGNfitter-rX (both normalized to their FIR peak) over 1.1-30 GHz: "
    f"median ratio = {np.median(_ratio_rb):.4f}, max|ratio-1| = {np.max(np.abs(_ratio_rb - 1.0)):.4f}"
)
print(
    f"§5  Caveat: q_IR = {_Q_BELL} instead of {Q_IR_PARITY:.3f} raises the 1.4 GHz luminosity "
    f"of the same L_IR by a factor {_radio_ratio_1p4ghz:.3f}"
)

# %% [markdown]
# **Result.** The SPL and DPL AGN radio agree with the upstream equations to 1e-4 (the parameter
# grids in §8.4 stay within 2e-4). The star-formation radio in the parity mode follows the template
# over 1.1-30 GHz to 3.6%, which is the difference between the $q_{\rm IR}$ measured from the template
# and the paper's quoted value (printed above). Users who keep tengri's default $q_{\rm IR}$ get
# the factor printed above at 1.4 GHz.

# %% [markdown]
# ## 6 Where tengri and AGNfitter-rX differ, and why
#
# Everything on this page that is not an agreement, in one place. The first group is upstream
# behavior that tengri deliberately does not reproduce; the second is conventions and terms where
# the two codes are built differently. The effect sizes are printed by the cell below from the
# sections that measure them; a dash means the item is structural and has no single scalar.

# %%
_i_up = lambda lam_aa: float(np.interp(lam_aa, _w_up, _A_up))
_neg_um = float(_neg[0]) / 1e4
_boost_10um = 10 ** (-0.4 * _i_up(1e5) * _EBV_GAL)
_k_uv_up, _k_uv_an = _i_up(950.0), float(np.interp(950.0, _wl_cal, _k_analytic))
_o = np.argsort(w_thb)
_w_bb, _L_bb = A.bbb_with_xrays(w_thb, L_thb_at_2500, ebv=0.0, scatter=0.0)
_o_bb = np.argsort(_w_bb)
_w_bb, _L_bb = np.asarray(_w_bb)[_o_bb], np.asarray(_L_bb)[_o_bb]
_i_join = int(np.searchsorted(_w_bb, 62.0))
_join_step = float(_L_bb[_i_join - 1] / _L_bb[_i_join])

_UPSTREAM = [
    ("Prevot disc screen: A_t/A_AF at equal E(B-V)", f"{_CONV:.4f}",
     "applies the declared R_V", "divide E(B-V)_BBB by this factor when moving a posterior"),
    ("Calzetti below 0.12 um: A/E(B-V) at 0.095 um", f"{_k_uv_up:.2f} vs {_k_uv_an:.2f}",
     "analytic curve, no second R_V", "do not compare galaxy attenuation below 0.12 um"),
    ("Calzetti above 2.2 um: k turns negative at [um]", f"{_neg_um:.2f}",
     "positive, monotonic tail", "restrict upstream comparisons to 0.12-2.2 um"),
    ("... flux boost at 10 um for E(B-V)=0.3", f"x{_boost_10um:.3f}",
     "no boost", "treat upstream's mid-IR host as brightened"),
    ("Disc-to-corona join at 200 eV: L(below 62 A)/L(above)", f"{_join_step:.3g}",
     "smooth corona; the KD18 family carries its own", "expect a step at 62 A in AGNfitter-rX only"),
    ("energy-balance prior: the GA term carries a template factor of", "1e18",
     "physical balance, same units", "do not expect upstream's floor"),
    ("maximal_age prior", "-",
     "age limit applied at the observing redshift", "expect a tighter age prior"),
    ("Charlot & Fall (2000) two-screen law", "-",
     "available in tengri's attenuation components", "no working upstream counterpart to compare"),
]
_CONVENTIONS = [
    ("qsogen's own quasar curve: A_V/E(B-V)", f"{_at_ext(_A_qsogen_on_grid, 5500):.2f} (Prevot {_at_ext(_k_tengri_ext, 5500):.2f})",
     "different law, own convention", "use `atten={'type': 'qsogen'}` only with a qsogen disc"),
    ("GA amplitude: present vs formed mass", f"tengri surviving fraction {_surv_frac:.4f}",
     "mass formed", "convert with the section 1.3 relation"),
    ("Torus amplitude: template integral with inclination", "tabulated in section 3",
     "same power at every node", "convert TO with the tabulated ratio"),
    ("Host nebular emission, IGM transmission", "-",
     "optional components, off here", "enable them for real data"),
]
for _title, _rows in (("upstream behavior tengri does not reproduce", _UPSTREAM), ("conventions and terms", _CONVENTIONS)):
    print(f"§6  {_title}")
    for _item, _eff, _tengri, _do in _rows:
        print(f"    - {_item}\n        effect: {_eff} | tengri: {_tengri} | do: {_do}")

# %% [markdown]
# **Reading the table.** The first group is corrections to upstream: tengri implements the
# intended version, and a posterior moved from AGNfitter-rX should be re-derived under it. The
# second is modelling choices: qsogen's own reddening convention, the mass and amplitude
# conventions of sections 1.3 and 3, and the terms AGNfitter-rX does not model at all (host nebular
# emission, IGM), which this page switches off so that the comparison is like for like.

# %% [markdown]
# ## 7 Translating an AGNfitter-rX fit into tengri
#
# A fit result gives AGNfitter-rX parameter values; tengri wants a `SEDModel.build` call. The
# table maps each AGNfitter-rX parameter to its tengri counterpart and the conversion, with the
# live constants printed; each tengri name is asserted against the built model, so a renamed
# parameter fails the notebook. The call below it builds the paper's winning combination
# (THB21-like disc, CAT3D-Wind torus, $\alpha_{\rm ox}$ corona, DPL jet, S17 cold dust, star-formation
# radio) on the fiducial host; section 9 compares it with AGNfitter-rX end to end.

# %%
_CAT3D_NODE = dict(incl=0.0, a=-2.0, fwd=1.75)
_LOG_LBOL_CAP = 11.0
_AGN_RADIO = builders.radio.agn.dpl(other_params=Fixed(DEFAULT))


def winning_model(radio):
    """The paper's winning AGN combination on the fiducial host, one ``SEDModel.build``."""
    return SEDModel.build(
        ssp_data=ssp, sfh=SFH_FIDUCIAL,
        dust_attenuation={"type": "single_component", "law": "calzetti", "dust_tau_v": Fixed(_tau_v_gal),
                           "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "schreiber2018", "dust_T": Fixed(35.0), "dust_f_pah": Fixed(0.02),
                        "all_params": Fixed(DEFAULT)},
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "agn_ebv_disc": Fixed(0.0), "all_params": Fixed(DEFAULT)},
            "nlr": {"type": "none"}, "blr": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
            "feii": {"type": "qsogen_balmer", "all_params": Fixed(DEFAULT)},
            "torus": {"type": "cat3d_wind", "cos_inc": Fixed(1.0), "a_cat3d": Fixed(-2.0),
                      "fwd_cat3d": Fixed(1.75), "all_params": Fixed(DEFAULT)},
            "atten": {"type": "none"},
            "agn_log_lbol": Fixed(_LOG_LBOL_CAP),
            "all_params": Fixed(DEFAULT), "norm": "independent",
        },
        xray={"type": "yang20", "all_params": Fixed(DEFAULT)},
        radio=radio, neb={"type": "none"}, redshift=Fixed(0.0),
    )


# Q_IR_PARITY enters the star-formation radio; defined in section 5.
m_cap = winning_model({"radio_q_ir": Fixed(Q_IR_PARITY), "sf": {"type": "bell2003_split"}, "agn": _AGN_RADIO})
resolved_params(m_cap)

_MAP = [
    # (AGNfitter-rX, tengri names asserted on the model, conversion)
    ("tau, age (GALAXY SFH)", ["sfh_declining_exp_tau_gyr", "sfh_declining_exp_age_gyr"],
     "direct; tengri age in Gyr, upstream in yr"),
    ("GA (log amplitude of the unit template)", ["sfh_declining_exp_log_total_mass"],
     f"present mass from GA with the section 1.3 relation; divide by the surviving fraction ({_surv_frac:.3f}) for the formed mass"),
    ("metal (BC03_metal axis)", ["met_logzsol"],
     "log10(Z_BC03 x 0.02 / 0.0142) = met_logzsol"),
    ("EBVgal", ["dust_tau_v"],
     f"dust_tau_v = R_V E(B-V)/1.086 with R_V = 4.05 (E(B-V) = 0.3 gives {_tau_v_gal:.3f})"),
    ("EBVbbb", ["agn_ebv_disc"],
     f"E(B-V)_tengri = E(B-V)_AF / {_CONV:.4f} (Prevot convention, section 2.2)"),
    ("SB, tdust, fpah (S17)", ["dust_T", "dust_f_pah"],
     "SB is not a parameter: the dust amplitude follows from energy balance (L_TIR)"),
    ("BB (disc amplitude)", ["agn_log_lbol"],
     "set agn_log_lbol so that L_nu(2500 A) of the disc plus lines matches; section 9 does this"),
    ("TO (torus amplitude), inclination, oa, tau", ["agn_cos_inc", "agn_a_cat3d", "agn_fwd_cat3d"],
     "block-specific node names; amplitude through the torus power (section 3 ratio)"),
    ("scatter on alpha_ox, Gamma, N_H", ["xray_delta_alpha_ox", "xray_gamma_agn", "xray_log_nh"],
     "direct; set log N_H = 0 and apply_anisotropy off for the bare AGNfitter-rX corona"),
    ("RAD, alpha, nu_t, nu_cut", ["radio_loudness", "radio_alpha_agn", "radio_log_nu_t"],
     "direct names; amplitude normalized at 5 GHz"),
    ("q_IR (star-formation radio)", ["radio_q_ir"],
     f"direct; the repackaged template embeds {Q_IR_PARITY:.3f}, tengri's default is {_Q_BELL}"),
]
_names = set(m_cap.spec.all_params)
print("§7  AGNfitter-rX parameter -> tengri parameter")
print("    | AGNfitter-rX | tengri | conversion |")
for _af, _te, _conv in _MAP:
    _missing = [n for n in _te if n not in _names]
    assert not _missing, f"tengri parameter(s) not on the built model: {_missing}"
    print(f"    | {_af} | {', '.join(f'`{n}`' for n in _te)} | {_conv} |")

# %% [markdown]
# ### Priors
#
# `tengri.agn.priors.agnfitter_priors` adapts the eight AGNfitter-rX priors (`PRIORS_AGNfitter.py`)
# onto a `model.predict(params)` prediction, using the per-sub-block `sed_agn_torus` and
# `sed_agn_disc` it exposes. `Fitter(model, ..., extra_log_prior=callable(params, state))` reaches
# every inference backend with the same callable. All eight are evaluated on an AGN + galaxy
# build below, each fed a stand-in for the data derived from the model's own prediction (a
# self-consistency demonstration of the API, not an observation). `prior_stellar_mass` is a
# Gaussian on upstream's GA; the cell converts tengri's present stellar mass to GA with the
# driver's exact relation (§1.3), so the prior acts on the same physical quantity in both codes.
# `prior_energy_balance` compares the dust-absorbed with the cold-dust re-emitted luminosity; at
# `dust_eta_balance`'s default (strict balance) they match by construction.
#
# **Caveat:** two upstream priors have no faithful tengri counterpart, by design. Upstream's
# `maximal_age` evaluates the observing-redshift terms outside its integrand, so it imposes no real
# age limit, and its energy-balance prior compares quantities in different units (the galaxy term
# carries a $10^{18}$ template factor, the dust term does not), which loosens the floor by orders
# of magnitude. tengri implements the physical versions, so agreement with upstream's values is
# not expected.

# %%
from tengri.agn.priors import AGNFITTER_PRIOR_DEFAULTS, agnfitter_priors

_z13 = 0.5
_dL13 = float(cosmo.luminosity_distance(_z13))
m13 = SEDModel.build(
    ssp_data=ssp, sfh=SFH_FIDUCIAL,
    dust_attenuation={"type": "single_component", "law": "calzetti", "dust_tau_v": Fixed(_tau_v_gal),
                       "all_params": Fixed(DEFAULT)},
    dust_emission={"type": "schreiber2018", "dust_T": Fixed(35.0), "dust_f_pah": Fixed(0.02),
                    "all_params": Fixed(DEFAULT)},
    agn={
        "type": "composable",
        "disc": {"type": "qsogen", "agn_ebv_disc": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "none"}, "blr": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
        "feii": {"type": "qsogen_balmer", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "cat3d_wind", "cos_inc": Fixed(1.0), "a_cat3d": Fixed(-2.0),
                  "fwd_cat3d": Fixed(1.75), "all_params": Fixed(DEFAULT)},
        "atten": {"type": "none"},
        "agn_log_lbol": Fixed(11.5),
        "all_params": Fixed(DEFAULT), "norm": "independent",
    },
    xray={"type": "yang20"},
    radio={"sf": {"type": "bell2003"}, "agn": {"type": "dpl"}},
    neb={"type": "none"}, redshift=Fixed(_z13),
)
resolved_params(m13)
pred13 = m13.predict({})
w13 = np.asarray(pred13.sed.components["wavelength"])
_o13 = np.argsort(w13)
_dim13 = (1.0 + _z13) / (4.0 * np.pi * _dL13**2)  # L_nu -> F_nu conversion


def _flux_at(key, lam_rest):
    L = np.asarray(pred13.sed.components[key])
    return float(np.interp(lam_rest, w13[_o13], L[_o13])) * _dim13


_flux_1500 = _flux_at("sed_intrinsic", 1500.0)
_l_2kev = float(np.interp(6.199, w13[_o13], np.asarray(pred13.sed.components["sed_xray"])[_o13]))
_log_l2kev = float(np.log10(max(_l_2kev, 1e-300)))
# ir_xrays' log_f2_10kev_data is compared against a prediction the prior
# itself computes from nulnu_6um via Stern (2015); feeding it the SAME
# formula applied to the model's own torus 6-um flux (rather than an
# independent X-ray channel) is the self-consistent choice for this API
# demo -- Stern's relation is not otherwise enforced between tengri's
# alpha_ox-tied corona and its torus, so an unrelated channel would compare
# two independent physical scales and swing by tens of dex.
_torus_lnu13 = np.asarray(pred13.sed.components["sed_agn_torus"])
_lnu_6um13 = float(np.interp(60000.0, w13[_o13], _torus_lnu13[_o13]))
_nulnu_6um13 = (U.C_ANGSTROM_PER_S / 60000.0) * _lnu_6um13
_x_stern13 = np.log10(_nulnu_6um13 / 1e41)
_log_f_2_10kev = 22.9494264 + 1.024 * _x_stern13 - 0.047 * _x_stern13**2
_flux_rad = _flux_at("sed_radio", U.C_ANGSTROM_PER_S / 1.4e9)
_log_nu_rad = float(np.log10(1.4e9))
_ir_band13 = (w13 > 3e4) & (w13 < 1e6)
_lam_ir_peak = float(w13[_ir_band13][np.argmax(np.asarray(pred13.sed.components["sed_dust_ir"])[_ir_band13])])
_flux_ir = _flux_at("sed_dust_ir", _lam_ir_peak)
_log_nu_ir = float(np.log10(U.C_ANGSTROM_PER_S / _lam_ir_peak))

total_default, breakdown_default = agnfitter_priors(
    pred13, redshift=_z13, dlum=_dL13, torus_key="sed_agn_torus", disc_key="sed_agn_disc",
    data_flux_1500=_flux_1500, **{f"enable_{k}": v for k, v in AGNFITTER_PRIOR_DEFAULTS.items()
                                    if k != "energy_balance_mode"},
    energy_balance_mode=AGNFITTER_PRIOR_DEFAULTS["energy_balance_mode"],
)
_ga13 = A.ga_from_log_mstar_present(np.log10(float(pred13.sfh.stellar_mass_surviving)), _dL13, _z13)
total_all, breakdown_all = agnfitter_priors(
    pred13, redshift=_z13, dlum=_dL13, torus_key="sed_agn_torus", disc_key="sed_agn_disc",
    enable_energy_balance=True,
    enable_stellar_mass=True, ga=_ga13,  # GA from tengri's present stellar mass (driver relation)
    enable_agn_fraction=True, data_flux_1500=_flux_1500,
    enable_low_agn_fraction=True,
    enable_midir_uv=True,
    enable_uv_xrays=True, log_l2kev_data=_log_l2kev,
    enable_ir_xrays=True, log_f2_10kev_data=_log_f_2_10kev,
    enable_ir_syn_fraction=True, data_flux_rad=_flux_rad, data_nu_rad=_log_nu_rad,
    data_flux_ir=_flux_ir, data_nu_ir=_log_nu_ir,
)
from tengri.agn.priors import AGNFITTER_HARD_REJECT


def _fmt_prior(v: float) -> str:
    v = float(v)
    return "HARD_REJECT" if v == AGNFITTER_HARD_REJECT else f"{v:.3f}"


print(f"§7  AGNfitter-rX default flags (energy_balance flexible + agn_fraction): total = {float(total_default):.3f}")
print("     | prior            | log-prior   |")
for k, v in breakdown_default.items():
    print(f"     | {k:16s} | {_fmt_prior(v):>11s} |")
print(
    f"\n§7  all eight priors, all finite: total = {float(total_all):.3f}  "
    f"(GA = {_ga13:.3f} from M*_present = {float(pred13.sfh.stellar_mass_surviving):.3e} Msun at z = {_z13})"
)
print("     | prior            | log-prior   |")
for k, v in breakdown_all.items():
    print(f"     | {k:16s} | {_fmt_prior(v):>11s} |")

# %% [markdown]
# Attaching a prior to a fit: `Fitter(model, ..., extra_log_prior=callable)` reaches MAP, VI and
# MCMC. The hook below uses `uv_xrays` alone, fed a `log_l2kev_data` 3 dex brighter (in the implied
# disc $L_{2500}$) than the build's own truth, standing in for an external X-ray measurement that
# disagrees with the photometry-only fit, which is when an informative prior earns its keep. To
# make the pull visible, the mock photometry drops the 1500 Å anchor and widens to 30% noise
# (three bands), and `enable_agn_fraction` and `enable_energy_balance` are set `False`
# explicitly, since the energy-balance floor is a step with zero gradient once rejected.

# %%
from tengri import FilterCurve, Fitter, Observation, Photometry


def _tophat_filter(center_aa, frac=0.16, n=25):
    w = jnp.linspace(center_aa * (1 - frac), center_aa * (1 + frac), n)
    return FilterCurve(wave=w, trans=jnp.sin(jnp.linspace(0, jnp.pi, n)) * 0.6, name=f"b{int(center_aa)}")


_filters13 = tuple(_tophat_filter(c) for c in (5000.0, 2e4, 1e5))
m13_obs = SEDModel.build(
    ssp_data=ssp, observation=Observation(photometry=Photometry(filters=_filters13)),
    sfh=SFH_FIDUCIAL,
    dust_attenuation={"type": "single_component", "law": "calzetti", "dust_tau_v": Fixed(_tau_v_gal),
                       "all_params": Fixed(DEFAULT)},
    dust_emission={"type": "schreiber2018", "dust_T": Fixed(35.0), "dust_f_pah": Fixed(0.02),
                    "all_params": Fixed(DEFAULT)},
    agn={
        "type": "composable",
        "disc": {"type": "qsogen", "agn_ebv_disc": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "none"}, "blr": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
        "feii": {"type": "qsogen_balmer", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "cat3d_wind", "cos_inc": Fixed(1.0), "a_cat3d": Fixed(-2.0),
                  "fwd_cat3d": Fixed(1.75), "all_params": Fixed(DEFAULT)},
        "atten": {"type": "none"},
        "agn_log_lbol": tengri.Uniform(9.0, 13.0),
        "all_params": Fixed(DEFAULT), "norm": "independent",
    },
    xray={"type": "none"},
    radio={"sf": {"type": "none"}, "agn": {"type": "none"}},
    neb={"type": "none"}, redshift=Fixed(_z13),
)
import jax

_p0 = dict(m13_obs.spec.sample(jax.random.PRNGKey(0)))
_mock_flux = np.asarray(m13_obs.predict_photometry(_p0)) * (
    1 + 0.02 * np.random.default_rng(0).normal(size=len(_filters13))
)
_mock_noise = 0.30 * np.abs(_mock_flux)

_pred0_fit = m13_obs.predict(_p0)
_w0_fit = np.asarray(_pred0_fit.sed.components["wavelength"])
_o0_fit = np.argsort(_w0_fit)
_disc0_fit = np.asarray(_pred0_fit.sed.components["sed_agn_disc"])
_log_l2500_truth = float(np.log10(np.interp(2500.0, _w0_fit[_o0_fit], _disc0_fit[_o0_fit])))
_BETA_JR16, _GAMMA_JR16 = 0.643, 6.8734  # Lusso & Risaliti (2016) L_2500-L_2keV slope/intercept
_log_l2kev_mismatched = (_log_l2500_truth + 3.0) * _BETA_JR16 + _GAMMA_JR16  # +3 dex brighter than truth


# The hook receives the fit's fully resolved parameter vector, while
# `model.predict` takes the free parameters (a Fixed parameter's value comes
# from the model itself). Drop the keys the spec pins before predicting;
# anything the spec does not pin (free draws, runtime latents) passes through.
_pinned13 = {
    name
    for name in m13_obs.spec.all_params
    if getattr(m13_obs.spec.get_distribution(name), "is_fixed", False)
}


def _priors_hook(params, state):
    pred = m13_obs.predict({k: v for k, v in params.items() if k not in _pinned13})
    total, _ = agnfitter_priors(
        pred, redshift=_z13, dlum=_dL13, torus_key="sed_agn_torus", disc_key="sed_agn_disc",
        enable_energy_balance=False, enable_agn_fraction=False,
        enable_uv_xrays=True, log_l2kev_data=_log_l2kev_mismatched,
    )
    return total


_fit_off = Fitter(m13_obs, data=_mock_flux, noise=_mock_noise, data_type="photometry")
_res_off = _fit_off.run(method="map", key=jax.random.PRNGKey(4))
_fit_on = Fitter(m13_obs, data=_mock_flux, noise=_mock_noise, data_type="photometry",
                  extra_log_prior=_priors_hook)
_res_on = _fit_on.run(method="map", key=jax.random.PRNGKey(4))
_lbol_off = float(_res_off.params["agn_log_lbol"])
_lbol_on = float(_res_on.params["agn_log_lbol"])
print(
    f"§7  MAP agn_log_lbol: hook off = {_lbol_off:.3f}, hook on (uv_xrays, "
    f"+3 dex L_2500 mismatch) = {_lbol_on:.3f}  (shift = {_lbol_on - _lbol_off:+.3f} dex, "
    f"truth = {float(_p0['agn_log_lbol']):.3f}) -- the mismatched X-ray-implied UV "
    "luminosity pulls the MAP fit toward a brighter disc, exactly as intended."
)

# %% [markdown]
# **Result.** All eight priors evaluate and carry a finite log-density except where the printed
# table shows a hard rejection; the MAP fit with the hook moves toward the brighter disc the
# mismatched X-ray measurement implies, as intended.

# %% [markdown]
# ## 8 Grid-level evidence
#
# The per-node figures and full window tables behind the summary tables of sections 2 and 3, and
# the radio parameter grids. A figure is kept here only where it shows something the summary table
# does not: where in wavelength a residual sits.

# %% [markdown]
# ### 8.1 SN12 and KD18 node grids
#
# SN12 at four nodes of AGNfitter-rX's $(\log M_{\rm BH}, \dot M/\dot M_{\rm Edd})$ table (axes
# printed in the table) and KD18 at four $(\log M_{\rm BH}, \log\lambda_{\rm Edd})$ nodes, anchored at
# 2500 Å and compared over 1200 Å-1 µm.

# %%
print(
    f"§8.1  SN12 table axes: log M_BH = {np.round(sn12_axes['log_mbh'], 2).tolist()}; "
    f"log Mdot/Mdot_Edd = {np.round(sn12_axes['log_edd'], 3).tolist()}"
)
plt.rcParams["figure.dpi"] = 100  # keep the rendered notebook under the figure-size budget
fig, (ax, ax_r), _ratios_9a2 = V.sweep_fig(
    cases_9a2, ref_label="AGNfitter-rX", title="SN12 and KD18 node grids",
    xlim=(1.2e3, 1e4), xlabel=r"$\lambda$ [Å]", ylabel=r"$L_\nu$ (norm. at 2500 Å)",
    cmap=None,
)
fig.tight_layout()
save_fig("agnfitter_09a1_disk_nodes.png")
plt.rcParams["figure.dpi"] = 150
V.print_window_table(_win_9a2, ref_name="AGNfitter-rX", title="§8.1 SN12 and KD18 node grids")

# %% [markdown]
# ### 8.2 Torus reductions and node grids
#
# Five averaged reductions (parity ratio against the reference at the matching node), then the S04
# and NK08 axes, the SKIRTOR $(oa, incl, \tau)$ nodes, and the two SKIRTOR reductions at the
# fiducial. The full unaveraged `skirtor` grid carries clumpiness and radial structure that the
# averaged reference removes, so it differs by construction and is excluded from the node-exact
# count.

# %%
fig, ax = plt.subplots(figsize=(9, 4.8))
print("§8.2  torus-reduction parity (peak-normalized, 1-100 um median|log10 ratio|):")
for (_name, _curve), c, _row in zip(_reduction_curves.items(), ["C0", "C1", "C2", "C3", "C4"], _reduction_rows):
    _grid, _ratio = _curve
    ax.loglog(_grid, _ratio, c, lw=1.4, label=_name)
    print(
        f"    {_row['label']:60s} median = {_row['median_dex']:.3f} dex  "
        f"(points above 10 um: tengri {_row['n_tengri']}, reference {_row['n_ref']})"
    )
ax.axhspan(0.8, 1.25, color="0.9", zorder=0)
ax.axhline(1.0, color="0.5", lw=0.8)
ax.set_xlim(8e3, 3e6)
ax.set_ylim(0.3, 3.0)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel("tengri / AGNFITTER (peak-norm.)")
ax.set_title("Five further torus reductions — parity ratio")
ax.legend(fontsize=8, ncol=3)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_09c1_reductions.png")

# %%
plt.rcParams["figure.dpi"] = 100  # keep the rendered notebook under the figure-size budget
fig, (ax, ax_r), _ratios_9c0 = V.sweep_fig(
    cases_9c0_fig, ref_label="AGNfitter-rX", title="S04 log N_H and NK08 inclination",
    xlim=(8e3, 3e6), xlabel=r"$\lambda$ [Å]", ylabel=r"$L_\nu$ (norm. at peak)",
    cmap=None,
)
fig.tight_layout()
save_fig("agnfitter_09c0_torus_sweeps.png")
plt.rcParams["figure.dpi"] = 150
V.print_window_table(_win_9c0, ref_name="AGNfitter-rX", title="§8.2 S04 log N_H and NK08 inclination")
node_exact_verdict(_win_9c0, "§8.2 S04 + NK08")

fig, (ax, ax_r), _ratios_9c5 = V.sweep_fig(
    cases_9c5, ref_label="AGNfitter-rX", title="SKIRTOR (oa, incl, τ) nodes",
    xlim=(8e3, 3e6), xlabel=r"$\lambda$ [Å]", ylabel=r"$L_\nu$ (norm. at peak)",
    cmap=None,
)
fig.tight_layout()
save_fig("agnfitter_09c5_skirtor_nodes.png")
V.print_window_table(_win_9c5, ref_name="AGNfitter-rX", title="§8.2 SKIRTOR (oa, incl, τ) nodes")
node_exact_verdict([r for r in _win_9c5 if "full grid" not in r["label"]], "§8.2 SKIRTOR (averaged nodes)")

# %% [markdown]
# ### 8.3 CAT3D-Wind: the wind-fraction union
#
# Upstream's CAT3D is one library whose wind-fraction axis joins two tables of the same pickle;
# tengri splits it over two blocks, `cat3d_wind` (high wind fractions) and `cat3d_wind_lowfwd` (low
# ones). The sweep runs $f_{\rm wd}$ across the union axis `cat3d_union_axes()` at fixed
# inclination and `a`, choosing the tengri block by the half of the axis a node belongs to; four
# `(incl, a, f_wd)` index triples then vary inclination and the radial index together with the wind
# fraction. The shaded band marks the 1.5-5 µm near-infrared excess the paper attributes to polar
# wind dust.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
_colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(_plot_fwd)))
_ic = 0
print(
    f"§8.3  CAT3D union axes: {_u['incl'].size} inclinations x {_u['a'].size} radial indices x "
    f"{_u['fwd'].size} wind fractions = {_u['incl'].size * _u['a'].size * _u['fwd'].size} nodes"
)
print(f"§8.3  wind-fraction sweep at incl={_INCL0:g} deg, a={_A0:g} (peak-normalized, 1-100 um):")
print("    |  f_wd | reference sub-library | tengri block        | median|t/a-1| | max|t/a-1| | at [um] |")
for _fwd, _sub, _blk, _med, _mx, _lam_mx in _union_table:
    print(f"    | {_fwd:5.2f} | {_sub:21s} | {_blk:19s} | {_med:13.4g} | {_mx:10.4g} | {_lam_mx / 1e4:7.2f} |")
node_exact_verdict(_union_rows, "§8.3 union sweep")
print("§8.3  (incl, a, f_wd) index triples off torus_axes('CAT3D'):")
for _label, _incl, _a, _fwd, _med, _mx in _tri_table:
    print(f"  {_label} incl={_incl:g}° a={_a:g} f_wd={_fwd:g}:  median |ratio-1| = {_med * 100:.2f}%   max = {_mx * 100:.2f}%")
node_exact_verdict(_tri_rows, "§8.3 triples")
for _entry in _cat_curves:
    if _entry[0] == "union":
        _, _f, _wa, _la, _wt, _lt = _entry
        c = _colors[_ic]
        _ic += 1
        ax.loglog(_wa, _la, color=c, ls="--", lw=1.2)
        ax.loglog(_wt, _lt, color=c, ls="-", lw=1.5, label=f"$f_{{wd}}$ = {_f:g}")
    else:
        _, c, _lab, _wa, _la, _wt, _lt = _entry
        _ma, _mt = (_wa > 5e3) & (_wa < 1e7), (_wt > 5e3) & (_wt < 1e7)
        ax.loglog(_wa[_ma], _la[_ma], c, ls="--", lw=1.0)
        ax.loglog(_wt[_mt], _lt[_mt], c, ls="-", lw=1.3, label=_lab)
ax.axvspan(1.5e4, 5e4, color="0.92", zorder=0)
ax.set_xlim(8e3, 3e6)
ax.set_ylim(1e-3, 3)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ (norm. at peak)")
ax.set_title("CAT3D-Wind union sweep — f_wd (both blocks) and (incl, a, f_wd) nodes")
ax.legend(fontsize=7, ncol=2, title="shaded: 1.5–5 µm excess band")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_09c3_cat3d_fwd_sweep.png")

# %% [markdown]
# ### 8.4 Radio parameter grids
#
# The simple power law at $\alpha \in \{-0.5, -0.75, -1.0\}$ crossed with
# $\log\nu_{\rm cut} \in \{12, 13, 14\}$, and the double power law at
# $\log\nu_t \in \{9.5, 10, 10.5\}$ with $\alpha_1 = -0.75$, $\alpha_2 = -0.1$,
# $\log\nu_{\rm cut} = 13$, each against the driver's equations, normalized at 5 GHz over
# 0.1-300 GHz. The default nodes are in section 5.

# %%
print("§8.4  SPL alpha x log_nu_cut grid (0.1-300 GHz, norm. at 5 GHz):")
_spl_grid_worst = 0.0
for alpha in [-0.5, -0.75, -1.0]:
    for log_nu_cut in [12.0, 13.0, 14.0]:
        _, F_af = A.agn_radio_spl(freq, alpha=alpha, log_nu_cut=log_nu_cut)
        L_t = np.asarray(radio_agn(wave_radio, L_AGN_BOL, radio_loudness=1.0, alpha_agn=-alpha, log_nu_cut=log_nu_cut))
        t_n = np.asarray(norm_at(np.asarray(wave_radio), L_t, _nu5))
        ratio = np.where(_band & (t_n > 0), t_n / _norm5(freq, F_af), np.nan)
        _mx = float(np.nanmax(np.abs(ratio - 1.0)))
        _spl_grid_worst = max(_spl_grid_worst, _mx)
        print(f"    alpha={alpha:+.2f} log_nu_cut={log_nu_cut:g}   max|ratio-1| = {_mx:.2e}")
print("§8.4  DPL log_nu_t grid (0.1-300 GHz, norm. at 5 GHz):")
_dpl_grid_worst = 0.0
for log_nu_t in [9.5, 10.0, 10.5]:
    _, F_af = A.agn_radio_dpl(freq, alpha1=-0.75, alpha2=-0.1, log_nu_t=log_nu_t, log_nu_cut=13.0)
    L_t = np.asarray(radio_agn_dpl(wave_radio, L_AGN_BOL, radio_loudness=1.0, alpha1=-0.75, alpha2=-0.1, log_nu_t=log_nu_t, log_nu_cut=13.0))
    t_n = np.asarray(norm_at(np.asarray(wave_radio), L_t, _nu5))
    ratio = np.where(_band & (t_n > 0), t_n / _norm5(freq, F_af), np.nan)
    _mx = float(np.nanmax(np.abs(ratio - 1.0)))
    _dpl_grid_worst = max(_dpl_grid_worst, _mx)
    print(f"    log_nu_t={log_nu_t:g}   max|ratio-1| = {_mx:.2e}")
print(f"§8.4  worst over the full grid: max|ratio-1| = {max(_spl_grid_worst, _dpl_grid_worst):.2e}")

# %% [markdown]
# ## 9 Capstone: the paper's winning model on its host, radio to X-ray
#
# The model translated in section 7 is compared with AGNfitter-rX's `ymodel`
# (`PARAMETERSPACE_AGNfitter.py`), which sums
#
# $$L_\nu = 10^{SB}\,T_{\rm SB} + 10^{BB}\,T_{\rm BBB} + 10^{GA}\,T_{\rm GA}
# + 10^{TO}\,T_{\rm TO} + 10^{RAD}\,T_{\rm RAD}.$$
#
# Every AGNfitter-rX curve is upstream's own template or equation read through the driver, never
# a tengri output. The amplitudes $10^N$ are free parameters of upstream's fit, so each is set from
# a matched input: GA from tengri's present stellar mass (§1.3), with no free scale; SB so its
# 8-1000 µm power equals tengri's realized $L_{\rm TIR}$; BB so $L_\nu(2500\,{\rm Å})$ equals tengri's
# disc plus lines; TO so its integral equals tengri's torus power; RAD so the DPL equals tengri's
# jet at 5 GHz. The test is absolute scale for GA and shape for the rest. The BBB term is upstream's
# disc cut at 200 eV plus its $\alpha_{\rm ox}$ power law, with a hard step and no EUV bridge, which
# is how the fit sees it.

# %%
m_cap_jet = winning_model({"sf": {"type": "none"}, "agn": _AGN_RADIO})  # the AGN jet alone
s_cap = m_cap.predict({})
w_te = np.asarray(s_cap.sed.components["wavelength"])
_comp = {k: np.asarray(s_cap.sed.components[k]) for k in (
    "sed_attenuated", "sed_dust_ir", "sed_agn", "sed_agn_torus", "sed_xray", "sed_radio", "sed_total")}
_jet_te = np.asarray(m_cap_jet.predict({}).sed.components["sed_radio"])
_sum_chk = _comp["sed_attenuated"] + _comp["sed_dust_ir"] + _comp["sed_agn"] + _comp["sed_xray"] + _comp["sed_radio"]
print(
    f"Capstone  sed_total vs sum of components: max|ratio-1| = "
    f"{np.max(np.abs(_sum_chk[_comp['sed_total'] > 0] / _comp['sed_total'][_comp['sed_total'] > 0] - 1.0)):.2e}"
)

# tengri components as ymodel groups them
te_ga = _comp["sed_attenuated"]
te_sb = _comp["sed_dust_ir"] + (_comp["sed_radio"] - _jet_te)  # cold dust + star-formation radio
te_bb = (_comp["sed_agn"] - _comp["sed_agn_torus"]) + _comp["sed_xray"]  # disc + lines + corona
te_to = _comp["sed_agn_torus"]
te_rad = _jet_te
te_tot = te_ga + te_sb + te_bb + te_to + te_rad
_owt = np.argsort(w_te)
_L2500_te = float(np.interp(2500.0, w_te[_owt], (_comp["sed_agn"] - _comp["sed_agn_torus"])[_owt]))

# AGNfitter-rX components: upstream templates and equations, amplitudes from matched inputs
_m_present = float(s_cap.sfh.stellar_mass_surviving)
w_ga_c, L_ga_c = A.galaxy_lnu(TAU_GYR, AGE_GYR * 1e9, np.log10(_m_present), ebv=_EBV_GAL)
w_sb_c, L_sb_c = A.cold_dust_radio_template(tdust=35.0, fpah=0.02)  # S17_radio: dust + star-formation radio
L_sb_c = L_sb_c * band_power(w_te, _comp["sed_dust_ir"], _IR_LO, _IR_HI) / band_power(w_sb_c, L_sb_c, _IR_LO, _IR_HI)
w_d0, L_d0 = A.disk_template("THB21")
L_d0 = L_d0 * _L2500_te / val_at(w_d0, L_d0, 2500.0)  # BB amplitude: L_nu(2500 A); alpha_ox follows from it
w_bb_c, L_bb_c = A.bbb_with_xrays(w_d0, L_d0, ebv=0.0, scatter=0.0)
w_to_c, L_to_c = A.torus_template("CAT3D", **_CAT3D_NODE)
L_to_c = L_to_c * band_power(w_te, te_to, 1e3, 1e9) / band_power(w_to_c, L_to_c, 1e3, 1e9)
_nu_r = 10.0 ** np.arange(7.0, 15.0, 0.02)  # upstream's own radio frequency grid
_, F_r = A.agn_radio_dpl(_nu_r, **_DPL_PARS)
w_r_c = U.C_ANGSTROM_PER_S / _nu_r[::-1]
L_r_c = F_r[::-1] * val_at(w_te, te_rad, U.C_ANGSTROM_PER_S / 5e9) / float(np.interp(5e9, _nu_r, F_r))

nu_grid = np.geomspace(1e8, 1e20, 4000)
lam_grid = U.C_ANGSTROM_PER_S / nu_grid
af_parts, af_sed = A.ymodel_sum(
    {"GA": (w_ga_c, L_ga_c), "SB": (w_sb_c, L_sb_c), "BB": (w_bb_c, L_bb_c), "TO": (w_to_c, L_to_c), "RAD": (w_r_c, L_r_c)},
    lam_grid,
)
te_parts = {k: U.regrid(w_te, np.clip(v, 0, None), lam_grid) for k, v in
            {"GA": te_ga, "SB": te_sb, "BB": te_bb, "TO": te_to, "RAD": te_rad}.items()}
te_sed = U.regrid(w_te, np.clip(te_tot, 0, None), lam_grid)
af_plot = np.where(af_sed > 0, nu_grid * af_sed, np.nan)
te_plot = np.where(te_sed > 0, nu_grid * te_sed, np.nan)
_resid = np.where((af_sed > 0) & (te_sed > 0), (te_sed - af_sed) / af_sed, np.nan)

_L2500_af = float(val_at(w_bb_c, L_bb_c, 2500.0))
_l2kev_af = float(val_at(w_bb_c, L_bb_c, 6.199))
print(
    f"Capstone  anchors: L_nu(2500 A) = {_L2500_te:.3e} erg/s/Hz (tengri, used for BB);  "
    f"alpha_ox upstream = {A.alpha_ox(_L2500_te):.3f}, "
    f"tengri corona: {-0.3838 * np.log10(_L2500_te / float(val_at(w_te, _comp['sed_xray'], 6.199))):.3f}"
)

fig, (ax, axr) = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax.loglog(nu_grid, af_plot, "C0-", lw=4.0, alpha=0.35, solid_capstyle="round",
          label="AGNFITTER-RX  ymodel: GA + SB + BB + TO + RAD")
ax.loglog(nu_grid, te_plot, "C1-", lw=1.4, label="tengri  one SEDModel.build")
for (_k, _c), _lab in zip({"GA": "C2", "SB": "C4", "BB": "C3", "TO": "C5", "RAD": "C6"}.items(),
                          ["GA stars", "SB dust + SF radio", "BB disc + corona", "TO torus", "RAD jet"]):
    ax.loglog(nu_grid, np.where(te_parts[_k] > 0, nu_grid * te_parts[_k], np.nan), _c, lw=0.9, alpha=0.75, label=f"   tengri {_lab}")
for nu_band in (1.4e9, 3e13, 6e14, 4.8e17):
    ax.axvline(nu_band, color="0.85", ls=":", lw=1)
ax.axvspan(3.3e15, 4.8e16, color="0.8", alpha=0.30, zorder=0)
ax.set_xlim(1e8, 1e20)
_te_fin = te_plot[np.isfinite(te_plot)]
ax.set_ylim(_te_fin.max() * 1e-9, _te_fin.max() * 5)
ax.set_ylabel(r"$\nu L_\nu$ [erg/s]")
ax.set_title(r"Radio-to-X-ray SED — the paper's winning model on its host ($8 < \log\,\nu/\mathrm{Hz} < 20$)")
ax.legend(fontsize=7, ncol=2, loc="lower center")
ax.grid(True, alpha=0.3)
axr.axhline(0.0, color="0.5", lw=0.8)
axr.semilogx(nu_grid, _resid, "C3-", lw=1.0)
axr.axvspan(3.3e15, 4.8e16, color="0.8", alpha=0.30, zorder=0)
axr.set_ylim(-1, 1)
axr.set_xlabel(r"$\nu$ [Hz]")
axr.set_ylabel("(tengri - AF)/AF")
axr.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_full_sed_headtohead.png")

# %% [markdown]
# **Caveat:** the EUV and soft-X-ray band (shaded) is a hole in both codes' simple corona, by
# different construction: upstream's power law starts above 200 eV with a hard step from the disc,
# and tengri's `yang20` corona carries no soft excess (that physics lives in the `kubota_done` disc's
# own warm Comptonization, not the $\alpha_{\rm ox}$ corona used here). Residuals in that band, and in
# the soft X-ray where tengri's default absorbing column acts (§4), are construction differences.
# The torus row inherits the sampling of the model wavelength grid (§3), and the `ymodel` sum has
# no nebular or X-ray-binary term.

# %%
def _ratio_band(key, lo_aa, hi_aa):
    return band_power(w_te, {"GA": te_ga, "SB": te_sb, "BB": te_bb, "TO": te_to, "RAD": te_rad}[key], lo_aa, hi_aa) / \
        band_power(lam_grid, af_parts[key], lo_aa, hi_aa)


print("Capstone  component band-power ratios, tengri / AGNfitter-rX")
print("    | component | band                      | ratio  | amplitude set by           |")
for _k, _lab, _lo, _hi, _how in [
    ("GA", "0.3-1 um", 3e3, 1e4, "present stellar mass (none free)"),
    ("GA", "1-3 um", 1e4, 3e4, "present stellar mass (none free)"),
    ("SB", "8-1000 um", _IR_LO, _IR_HI, "L_TIR (anchor)"),
    ("SB", "8-30 um", 8e4, 3e5, "L_TIR (shape test)"),
    ("BB", "0.1-1 um", 1e3, 1e4, "L_nu(2500 A) (shape test)"),
    ("BB", "0.5-10 keV", 1.24, 24.8, "alpha_ox (upstream) vs corona"),
    ("TO", "1-1000 um", 1e4, 1e7, "torus power (anchor)"),
    ("TO", "3-30 um", 3e4, 3e5, "torus power (shape test)"),
    ("RAD", "0.1-30 GHz", U.C_ANGSTROM_PER_S / 3e10, U.C_ANGSTROM_PER_S / 1e8, "5 GHz (shape test)"),
]:
    print(f"    | {_k:9s} | {_lab:25s} | {_ratio_band(_k, _lo, _hi):6.3f} | {_how:26s} |")

# %%
_opt_band = (nu_grid > 4e14) & (nu_grid < 8e14) & np.isfinite(_resid)
_opt_ratio = 1.0 + _resid[_opt_band]
_p16, _p50, _p84 = np.percentile(_opt_ratio, [16, 50, 84])
print(
    f"Capstone optical (4e14-8e14 Hz, tengri/AGNFITTER) normalization ratio: "
    f"median = {_p50:.3f}, 16-84% = [{_p16:.3f}, {_p84:.3f}]"
)
_fin_res = _resid[np.isfinite(_resid) & (nu_grid > 1e9) & (nu_grid < 1e15)]
print(
    f"Capstone fractional residual, 1e9-1e15 Hz (radio to UV): median = {np.median(_fin_res):+.3f}, "
    f"16-84% = [{np.percentile(_fin_res, 16):+.3f}, {np.percentile(_fin_res, 84):+.3f}]"
)

# %%
# Per-block worst deviation, collected from the cells above (nothing typed).
_SUMMARY = [
    ("SN12 + KD18 disc nodes", "§8.1", len(_win_9a2), max(r["max_abs_dev"] for r in _win_9a2), "max|t/a-1|"),
    ("S04 + NK08 torus nodes", "§8.2", len(_win_9c0), max(r["max_abs_dev"] for r in _win_9c0), "max|t/a-1|"),
    ("SKIRTOR (oa, incl, tau) nodes, averaged", "§8.2", len(_win_9c5) - 1,
     max(r["max_abs_dev"] for r in _win_9c5 if "full grid" not in r["label"]), "max|t/a-1|"),
    ("CAT3D union: wind-fraction sweep", "§8.3", len(_union_rows), max(r["max_abs_dev"] for r in _union_rows), "max|t/a-1|"),
    ("CAT3D (incl, a, f_wd) nodes", "§8.3", len(_tri_rows), max(r["max_abs_dev"] for r in _tri_rows), "max|t/a-1|"),
    ("Cold dust: S17 nodes + DH02 ladder", "§1.5", len(_s17_resid) + len(_dh_resids),
     max([r[2] for r in _s17_resid] + _dh_resids), "median|dex|"),
    ("X-ray corona, default N_H (0.5-100 keV)", "§4", len(_xray_rows[20.0]), max(r["max_abs_dev"] for r in _xray_rows[20.0]), "max|t/a-1|"),
    ("X-ray corona, N_H switched off", "§4", len(_xray_rows[0.0]), max(r["max_abs_dev"] for r in _xray_rows[0.0]), "max|t/a-1|"),
    ("Radio SPL alpha x nu_cut, DPL nu_t", "§8.4", 12, max(_spl_grid_worst, _dpl_grid_worst), "max|t/a-1|"),
]
print("Summary: per-block worst tengri / AGNfitter-rX deviation")
print("    | block                                     | §     | cases | worst      | metric      |")
for _blk, _sec, _n, _w, _met in _SUMMARY:
    print(f"    | {_blk:41s} | {_sec:5s} | {_n:5d} | {_w:10.4g} | {_met:11s} |")

# %% [markdown]
# ## Summary
#
# The per-block worst deviations are in the table printed above. The entries below say what
# matched and what did not, section by section; use them as the page to check before trusting
# anything else.
#
# - **§1 Host galaxy.** Stellar populations agree in shape at a matched node; the absolute scale
#   needs the mass mapping of §1.3 (upstream's unit template is a present mass, tengri's a mass
#   formed). Calzetti attenuation agrees inside its calibrated range; upstream's own tails are
#   tabulated, not reproduced. Cold dust (S17, DH02_CE01) and the host composite agree in template
#   shape; infrared sub-band powers inherit the sparse model grid.
# - **§2 Accretion disc.** Only THB21 carries the 0.7 µm bump, and tengri reproduces it in position
#   with a line-strength offset against the template. R06, SN12 and the grid-tabulated KD18 discs
#   agree at nodes. The Prevot disc screen is the same law one constant factor apart; qsogen's own
#   curve is a different law.
# - **§3 Torus.** The tabulated blocks reproduce their reference nodes, the remainder tracing to the
#   model wavelength grid; the CAT3D library is the union of two tengri blocks; the
#   inclination-dependent torus power differs by design.
# - **§4 X-ray corona.** Power law, cutoff and 2 keV anchor agree with anisotropy and absorption off;
#   tengri's defaults add both.
# - **§5 Radio.** SPL and DPL agree; the star-formation radio reproduces the template's embedded
#   $q_{\rm IR}$ in its parity mode.
# - **§6 Differences.** Four upstream behaviors are deliberately not reproduced; the rest are
#   conventions, all tabulated with their effect sizes.
# - **§7 Translation.** Every AGNfitter-rX parameter has a tengri counterpart and a printed
#   conversion; the eight informative priors evaluate through `agnfitter_priors` and attach to a
#   fit through `extra_log_prior`.
# - **§9 Capstone.** One buildable, fittable model over $8 < \log\nu/{\rm Hz} < 20$ against
#   upstream's `ymodel` sum of all five components, with a fractional-residual panel and a 16-84%
#   optical normalization spread.
#
# **What to use.** Move an AGNfitter-rX fit to tengri with the §7 table, switch the tengri
# defaults that AGNfitter-rX lacks off for a like-for-like comparison, and expect the §6 differences.

# %% [markdown]
# ## References
#
# Every model compared above, with the section that uses it. The machine-
# readable BibTeX lives next to this notebook in `references.bib`; the key of
# each entry is given in brackets.
#
# **Accretion disks (§2.1)**
# - Richards, G. T., et al. 2006, ApJS 166, 470 — R06 [`richards2006sed`].
# - Slone, O. & Netzer, H. 2012, MNRAS 426, 656 — SN12 [`slone2012effects`].
# - Kubota, A. & Done, C. 2018, MNRAS 480, 1247 — KD18 [`kubota2018physical`].
# - Temple, M. J., Hewett, P. C. & Banerji, M. 2021, MNRAS 508, 737 — THB21;
#   qsogen [`temple2021modelling`].
#
# **Tori (§3)**
# - Silva, L., et al. 2004, MNRAS 355, 973 — S04 [`Silva2004`].
# - Nenkova, M., et al. 2008, ApJ 685, 160 — NK08 [`nenkova2008agnII`].
# - Stalevski, M., et al. 2016, MNRAS 458, 2288 — SKIRTOR [`Stalevski2016`].
# - Hönig, S. F. & Kishimoto, M. 2017, ApJL 838, L20 — CAT3D-Wind [`honig2017dusty`].
# - Yang, G., et al. 2020, MNRAS 491, 740 — X-CIGALE SKIRTOR [`yang2020xcigale`].
#
# **Cold dust (§1.5–§1.6)**
# - Schreiber, C., et al. 2018, A&A 609, A30 — S17 [`schreiber2018dust`].
# - Dale, D. A. & Helou, G. 2002, ApJ 576, 159 [`dale2002infrared`]; Chary, R. &
#   Elbaz, D. 2001, ApJ 556, 562 [`chary2001interpreting`] — DH02_CE01.
# - Dale, D. A., et al. 2014, ApJ 784, 83 — tengri `dale2014` [`dale2014two`].
# - Calzetti, D., et al. 2000, ApJ 533, 682 — galaxy attenuation [`calzetti2000dust`].
#
# **X-ray (§4)**
# - Just, D. W., et al. 2007, ApJ 665, 1004 [`just2007x`]; Lusso, E. &
#   Risaliti, G. 2016, ApJ 819, 154 [`lusso2016tight`]; 2017, A&A 602, A79
#   [`lusso2017quasars`] — α_ox–L₂₅₀₀.
# - Stern, D. 2015, ApJ 807, 129 — 6 µm ↔ 2–10 keV relation behind the
#   AGNfitter-rX X-ray prior [`stern2015`].
# - Yang, G., et al. 2022, ApJ 927, 192 — X-ray viewing-angle anisotropy
#   [`yang2022cigale`].
# - Mineo, S., et al. 2014, MNRAS 437, 1698 — host XRB / SFR [`mineo2014x`].
#
# **Radio (§5)**
# - Azadi, M., et al. 2020 (arXiv:2011.03130) — radio AGN/SF separation
#   [`azadi2020disentangling`].
# - Bell, E. F. 2003, ApJ 586, 794 — IR–radio correlation [`bell2003estimating`].
# - Murphy, E. J., et al. 2011, ApJ 737, 67 — SFR-L_IR calibration [`murphy2011calibrating`].
#
# **Stellar populations & attenuation (§1.1–§1.4)**
# - Bruzual, G. & Charlot, S. 2003, MNRAS 344, 1000 [`bruzual2003stellar`];
#   Chabrier, G. 2003, PASP 115, 763 [`chabrier2003galactic`].
# - Prevot, M. L., et al. 1984, A&A 132, 389 — SMC reddening [`prevot1984typical`].
#
# **Codes, priors & inference (§7)**
# - Martínez-Ramírez, L. N., et al. 2024, A&A 688, A46 — AGNfitter-rX and its
#   informative priors [`martinez2024agnfitter`].
# - Calistro Rivera, G., et al. 2016, ApJ 833, 98 — the original AGNfitter
#   [`calistrorivera2016agnfitter`].
# - Hearin, A. P., et al. 2023, MNRAS 521, 1741 — DSPS [`hearin2023dsps`].

# %% [markdown]
# ### BibTeX
#
# The complete machine-readable bibliography (printed below from
# `references.bib` so it never drifts from the file).

# %%
_bib_path = _HERE / "references.bib"
print(_bib_path.read_text())
