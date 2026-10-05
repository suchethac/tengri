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
# | Host stars (GA) | `sfh={'type': 'declining_exp'}`, BC03 grid | shape and light per formed mass match; surviving-mass tables differ 12% | 1.1-1.3 |
# | Host attenuation | `dust_attenuation={'law': 'calzetti'}` | matched inside the calibrated range; upstream tails differ | 1.4 |
# | Cold dust (SB) | `dust_emission={'type': 'schreiber2018'}`, `'dh02_ce01'` | matched at template nodes | 1.5-1.6 |
# | Accretion disc (BBB) | `agn={'disc': ...}`: `richards2006`, `slone_netzer`, `kd18_agnfitter`, `qsogen` with `blr` and `feii` | matched at nodes; THB21 is a template against a model | 2.1 |
# | Disc reddening | `agn_ebv_disc`, `atten` | same law, one constant factor apart | 2.2 |
# | Torus (TO) | `silva04`, `nenkova_agnfitter*`, `skirtor*`, `cat3d_wind*` | node-exact at table nodes (CAT3D low-wind: 10% between nodes); amplitude convention differs by design | 3 |
# | X-ray corona | `xray={'type': 'yang20'}` | exact when anisotropy and absorption are off | 4 |
# | Radio | `radio={'agn': 'dpl', 'sf': 'bell2003'}`, `radio_sfr_bell2003_split` | exact for the AGN; the stand-alone 90/10 function matches the star-formation radio | 5 |
# | Informative priors | `tengri.agn.priors.agnfitter_priors` | two upstream priors replaced by physical versions | 6, 7 |

# %%
# Paper-quoted values the prose relies on (Martínez-Ramírez et al. 2024). They
# are taken from the paper, not computed here.
PAPER_QUOTED = {
    "log10 Bayes factor, THB21 over R06": ("5.1", "p. 14"),
    "sources whose maximum-likelihood torus is CAT3D-Wind": ("25 of 36", "p. 13"),
    "sources best fit by CAT3D-Wind + THB21": ("67%", "pp. 14-15"),
    "wavelength range of the near-infrared excess [µm]": ("1.5-5", "pp. 12-13"),
    "q_IR adopted from Bell (2003)": ("2.64", "p. 3"),
    "scatter on q_IR": ("0.26", "p. 3"),
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
# nebular term, and sits at the BC03 solar metallicity, $Z = 0.02$ (the metallicity of upstream's
# single-metallicity template, §1.1).

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
    halpha_line,
    node_exact_verdict,
    norm_at,
    norm_peak,
    resolved_params,
    ssp_file_has_mass_remaining,
    val_at,
)

import tengri
from tengri import DEFAULT, Fixed, SEDModel, builders, load_ssp_data

# Inline backend, so every figure embeds in the page.
try:  # noqa: SIM105
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

warnings.filterwarnings("ignore")
warnings.filterwarnings("default", module=r"tengri(\.|$)")
tengri.plot.setup_style()

# ``_figs/`` and the data sit next to this file (the working directory when run from a kernel).
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
from tengri import units

AF_H0, AF_OM0 = 67.4, 0.315  # AGNfitter-rX cosmology (paper p. 2; z2Dlum)
_dl_af = cosmo.luminosity_distance(1.0, h0=AF_H0, om0=AF_OM0)
_dl_tengri = cosmo.luminosity_distance(1.0)
_c = cosmo.DEFAULT_COSMO
print("cosmology          |    H0 [km/s/Mpc] |    Omega_m")
print(f"AGNfitter-rX       | {AF_H0:16.2f} | {AF_OM0:10.5f}")
print(f"tengri default     | {100 * float(_c.h):16.2f} | {float(_c.Om0):10.5f}")
print(
    f"D_L(z=1): AGNfitter-rX = {_dl_af:.4e} cm;  tengri = {_dl_tengri:.4e} cm;  "
    f"D_L^2 ratio tengri/AGNfitter-rX = {(_dl_tengri / _dl_af) ** 2:.4f}"
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
# The upstream single-metallicity template is BC03 solar (Z = 0.02; shown in §1.1). tengri's
# `met_logzsol` is relative to its own solar value, Z = 0.0142, so matching it needs a shift.
Z_SUN_BC03, Z_SUN_TENGRI = 0.02, 0.0142
MET_BC03_SOLAR = {
    "logzsol": Fixed(float(np.log10(Z_SUN_BC03 / Z_SUN_TENGRI))),
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

m_csp = SEDModel.build(
    ssp_data=ssp, sfh=SFH_FIDUCIAL, met=MET_BC03_SOLAR,
    dust_attenuation=NO_DUST, neb={"type": "none"}, redshift=Fixed(0.0),
)
pred_csp = m_csp.predict({})
w_t = np.asarray(pred_csp.sed.components["wavelength"])
L_t = np.asarray(pred_csp.sed.components["sed_attenuated"])
w_r, L_r = A.galaxy_template(tau=TAU_GYR, age=AGE_GYR * 1e9)
_bandS1 = (w_t > 1.5e3) & (w_t < 2.5e4)
t_n = norm_at(w_t, L_t, 5500.0)
r_n = norm_at(w_r, L_r, 5500.0)
ax1.loglog(w_t[_bandS1], t_n[_bandS1], "C1-", lw=1.5, label="tengri declining_exp CSP")
ax1.loglog(w_r, r_n, "C0-", lw=3.5, alpha=0.35, solid_capstyle="round", label="AGNfitter-rX BC03 (matched node)")
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
# The paper's default library, BC03_metal, carries four metallicities. The table builds tengri
# at each, mapping the upstream axis (units of the BC03 solar value, $Z_\odot = 0.02$) onto
# `met_logzsol` (relative to tengri's solar value, $Z_\odot = 0.0142$), and compares shape at
# 5500 Å and absolute scale at 5500 Å per unit present mass and per unit formed mass (§1.3). The
# last lines identify which BC03 metallicity the single-metallicity template of the panel above
# is.
#
# **Caveat:** the top upstream node (twice the BC03 solar value) lies between tengri's
# tabulated metallicities (printed above the table), so tengri interpolates there.

# %%
_ax_m = A.galaxy_axes(metal=True)
_age_m = float(_ax_m["age"][int(np.argmin(np.abs(_ax_m["age"] - AGE_GYR * 1e9)))])
print(f"§1.1  tengri SSP metallicities Z = {np.round(10 ** np.asarray(ssp.ssp_lgmet), 5).tolist()}")
print(
    f"§1.1  BC03_metal nodes: Z/Z_sun = {_ax_m['metal'].tolist()}; tau = {TAU_GYR:g} Gyr, "
    f"age node = {_age_m / 1e9:.4f} Gyr"
)
print("    | Z [BC03 Z_sun] | tengri met_logzsol | shape median|dlog10| | f_surv tengri | f_surv upstream | L(5500) ratio per present mass | per formed mass |")
_sfh_m = {**SFH_FIDUCIAL, "sfh_declining_exp_age_gyr": Fixed(_age_m / 1e9)}
for _z in _ax_m["metal"]:
    _logzsol = float(np.log10(_z * Z_SUN_BC03 / Z_SUN_TENGRI))
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
    _mf, _mp = float(_pm.sfh.stellar_mass), float(_pm.sfh.stellar_mass_surviving)
    _fpp = A.galaxy_formed_per_present(TAU_GYR, _age_m, metal=float(_z))
    _t5, _u5 = np.interp(5500.0, _wm, _Lm), np.interp(5500.0, _wr, _Lr)
    print(
        f"    | {float(_z):14.1f} | {_logzsol:18.3f} | {_shape:20.4f} | {_mp / _mf:13.4f} | {1.0 / _fpp:15.4f} "
        f"| {_t5 / (_u5 * _mp):30.4f} | {(_t5 / _mf) / (_u5 / _fpp):15.4f} |"
    )

# Which BC03 metallicity is the single-metallicity template that the rest of the page uses?
_w840, _L840 = A.galaxy_template(TAU_GYR, AGE_GYR * 1e9)
_b840 = (_w840 > 3e3) & (_w840 < 2.5e4)
print("§1.1  single-metallicity template against each BC03_metal node (0.3-2.5 um, nearest age nodes):")
for _z in _ax_m["metal"]:
    _wz, _Lz = A.galaxy_template(TAU_GYR, AGE_GYR * 1e9, metal=float(_z))
    _lr = np.log10(_L840[_b840] / np.interp(_w840[_b840], _wz, _Lz))
    print(f"    Z = {float(_z):.1f} Z_sun(BC03): std of log10 ratio = {np.std(_lr):.4f} dex, median ratio = {10 ** np.median(_lr):.3f}")

# %% [markdown]
# **Result.** The single-metallicity template is the BC03 solar one: against the Z = 1 node it
# differs by 0.0023 dex, against the others by 0.057-0.092 dex, so every tengri host on this page
# sits at $Z = 0.02$ (`met_logzsol` = 0.149). At that metallicity the median shape residual is
# 0.0125 dex, and at most 0.012 dex across the four BC03_metal nodes; it is the difference between
# the two BC03 editions, not of the SFH form. The last two table columns separate the absolute
# scale. Per unit present mass tengri sits 8-14% below the template at 5500 Å (0.864-0.921); per
# unit formed mass the two agree to 1-3% (0.990-1.029). The offset therefore sits in the
# surviving fraction (tengri 0.611, template 0.533-0.547), not in the light; §1.3 measures it at
# the fiducial node.

# %% [markdown]
# ### 1.2 Star formation history
#
# AGNfitter-rX's GALAXY component tabulates a declining exponential, $\mathrm{SFR}(T) \propto
# e^{-T/\tau}$, as the instantaneous SFR at each template age; tengri's `declining_exp` type has
# the same form. The test builds tengri at four age nodes of the template's $\tau = 1$ Gyr row at
# one formed mass and compares tengri's own `pred.sfh.sfr_10myr` with the template's tabulated SFR
# scaled to that mass (the unit template holds $f$ times its present mass in formed mass,
# §1.3). The window factor is the mean of $e^{-T/\tau}$ over the last 10 Myr relative to its end
# value, which a 10-Myr average of a declining history must carry.

# %%
_age_axis_yr, _sfr_tab = A.galaxy_sfr(tau=TAU_GYR)
_f_nodes = np.array([A.galaxy_formed_per_present(TAU_GYR, float(_a)) for _a in _age_axis_yr])
_sfr_up_curve = _sfr_tab * (10**LOG_MASS) / _f_nodes  # template SFR(age) at 1e10 Msun formed [Msun/yr]
_win_10 = (TAU_GYR / 0.01) * np.expm1(0.01 / TAU_GYR)

_sfh_rows = []
for _age_req in (0.5736, 1.1721, 2.395, AGE_GYR):
    _a = int(np.argmin(np.abs(_age_axis_yr - _age_req * 1e9)))
    _age_node = float(_age_axis_yr[_a]) / 1e9
    _m2 = SEDModel.build(
        ssp_data=ssp, sfh={**SFH_FIDUCIAL, "sfh_declining_exp_age_gyr": Fixed(_age_node)},
        met=MET_BC03_SOLAR, dust_attenuation=NO_DUST, neb={"type": "none"}, redshift=Fixed(0.0),
    )
    _p2 = _m2.predict({})
    _sfh_rows.append(
        (_age_node, float(_sfr_up_curve[_a]), float(_p2.sfh.sfr_10myr), float(_p2.sfh.stellar_mass) / 10**LOG_MASS)
    )

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.semilogy(_age_axis_yr / 1e9, _sfr_up_curve, "C0-", lw=3.5, alpha=0.35, solid_capstyle="round",
            label="AGNfitter-rX template SFR(age), scaled to $10^{10}\\,M_\\odot$ formed")
ax.semilogy([r[0] for r in _sfh_rows], [r[2] for r in _sfh_rows], "C1o", ms=7,
            label="tengri `pred.sfh.sfr_10myr`")
ax.set_xlabel("template age [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.set_title(rf"Declining-exponential SFH ($\tau$ = {TAU_GYR:g} Gyr)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_02_sfh_tau.png")
print("§1.2  SFR at the template age, tengri / AGNfitter-rX (both at 1e10 Msun formed)")
print("    | age [Gyr] | template SFR [Msun/yr] | tengri sfr_10myr [Msun/yr] | ratio | 10-Myr window factor | stellar_mass / 1e10 |")
for _age_node, _sfr_up, _sfr_te, _closure in _sfh_rows:
    print(
        f"    | {_age_node:9.4f} | {_sfr_up:22.4f} | {_sfr_te:26.4f} | {_sfr_te / _sfr_up:5.4f} "
        f"| {_win_10:20.4f} | {_closure:19.4f} |"
    )
print(
    f"§1.2  template SFR is monotonically declining: {bool(np.all(np.diff(_sfr_tab) <= 0))}"
)

# %% [markdown]
# **Result.** tengri's own 10-Myr SFR matches the template's tabulated SFR at the same formed mass
# to 0.5-0.7% at all four ages (ratios 1.0048-1.0068), against a window factor of 1.0050, so the
# two histories agree to 0.2%. The formed mass closes to 1.0000 at every age, and the template's SFR
# is monotonically declining, as printed.

# %% [markdown]
# ### 1.3 Integrated stellar SED and the mass convention
#
# The stored GA template is normalized to one solar mass of **present** stellar mass (stars
# plus remnants alive at the template age); tengri normalizes to mass **formed**. A prior on
# the GA amplitude therefore means a different stellar mass in the two codes unless it is
# mapped. The table compares the two at the matched node ($\tau = 1$ Gyr, $Z = 0.02$, age
# 4.8939 Gyr), per unit present mass and per unit formed mass at four wavelengths, and prints where
# tengri's mass-loss table comes from. The relation it ends on is the conversion used in sections 6
# and 7.

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
_surv_up = 1.0 / _f_formed_per_present
_w_ga, _L_ga_unit = A.galaxy_template(TAU_GYR, AGE_GYR * 1e9)
_lam_chk = np.array([3000.0, 5500.0, 1e4, 2e4])
_t_chk = np.interp(_lam_chk, w3, L3)
_u_chk = np.interp(_lam_chk, _w_ga, _L_ga_unit)
_own_table = ssp_file_has_mass_remaining(_ssp_path)
_mr = np.asarray(ssp.ssp_mass_remaining)
print("§1.3  GA ↔ tengri stellar-mass mapping at the matched node (tau = 1 Gyr, Z = 0.02)")
print("    | quantity                                                      |      value |")
print(f"    | tengri M*_formed (pred.sfh.stellar_mass) [Msun]               | {_m_formed:10.4e} |")
print(f"    | tengri M*_present (stellar_mass_surviving) [Msun]             | {_m_present:10.4e} |")
print(f"    | tengri surviving fraction M*_present / M*_formed              | {_surv_frac:10.4f} |")
print(f"    | upstream unit template: M_formed per M_present (SFR table)    | {_f_formed_per_present:10.4f} |")
print(f"    | upstream surviving fraction = 1 / that                        | {_surv_up:10.4f} |")
print(f"    | tengri / upstream surviving fraction                          | {_surv_frac / _surv_up:10.4f} |")
for _lam, _t, _u in zip(_lam_chk, _t_chk, _u_chk):
    print(f"    | L_nu ratio per unit PRESENT mass at {_lam:7.0f} Å                   | {(_t / _m_present) / _u:10.4f} |")
for _lam, _t, _u in zip(_lam_chk, _t_chk, _u_chk):
    print(f"    | L_nu ratio per unit FORMED mass at {_lam:7.0f} Å                    | {(_t / _m_formed) / (_u / _f_formed_per_present):10.4f} |")
print(
    f"§1.3  tengri mass-loss table: carried by the SSP file = {_own_table}; "
    f"same at all {_mr.shape[0]} metallicities = {bool(np.allclose(_mr, _mr[0]))}; "
    f"value at 5 Gyr = {float(np.interp(np.log10(5.0), np.asarray(ssp.ssp_lg_age_gyr), _mr[0])):.4f}"
)
print(
    "    GA amplitude -> upstream present mass:  M*_present = 10^GA · 4π d_L² / [L_sun (1+z)] / 1e18  "
    "(MODEL_AGNfitter.stellar_info)."
)
print(
    f"    Matching light: M*_formed = M*_present(upstream) x {_f_formed_per_present:.4f}; "
    f"tengri then reports M*_present = {_surv_frac / _surv_up:.4f} x the upstream present mass."
)

# %% [markdown]
# **Result.** Per unit formed mass the two SEDs agree to 0.2% at 0.3, 1 and 2 µm and to 2.9% at
# 5500 Å (ratios 1.0000, 0.9987, 0.9998 and 1.0292); per unit present mass tengri sits 8-11% below
# (0.894-0.921). The disagreement is in the surviving fraction, 0.6104 for tengri against 0.5464 for
# the template (a ratio of 1.117), not in the light. The cause is the mass-loss table. The BC03 SSP
# file that tengri reads carries none (printed), so tengri uses the DSPS Chabrier fit to FSPS,
# which is the same at every metallicity (0.598 for an SSP at 5 Gyr), while the template's fraction
# follows from BC03's own mass loss. This reproduction cannot split the 12% between isochrones and
# remnant bookkeeping, because the template stores only the SFR. To translate a fit, take the
# formed mass as the upstream present mass times the template's own formed-to-present ratio (1.8300
# here), not as the present mass divided by tengri's 0.610; tengri then reports a present mass 1.117
# times the upstream one. The same ratio converts a `stellar_mass_surviving` read from tengri into
# the upstream convention.

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
# with $x = 1/\lambda_{\mu{\rm m}}$, $R_V = 4.05$ and $\tau_V = R_V\,E(B{-}V)/1.086$ (`calzetti` ignores
# `dust_Rv`; set `dust_tau_v`). Left: inside the
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

m_g0 = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL, met=MET_BC03_SOLAR,
                       dust_attenuation={"type": "single_component", "law": "calzetti",
                                         "dust_tau_v": Fixed(0.0), "all_params": Fixed(DEFAULT)},
                       neb={"type": "none"}, redshift=Fixed(0.0))
m_g1 = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL, met=MET_BC03_SOLAR,
                       dust_attenuation={"type": "single_component", "law": "calzetti",
                                         "dust_tau_v": Fixed(_tau_v_gal), "all_params": Fixed(DEFAULT)},
                       neb={"type": "none"}, redshift=Fixed(0.0))
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
axr.plot(_w_up, _A_up, "C0-", lw=3.5, alpha=0.35, solid_capstyle="round", label="AGNfitter-rX  GALAXYred_Calzetti (as evaluated)")
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
# extrapolation below 0.12 µm and its negative infrared branch are not physical, and tengri's curve is
# held at zero there, so only the calibrated range is a meaningful comparison.

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
    pred = m.predict({})
    w = np.asarray(pred.sed.components["wavelength"])
    L = np.asarray(pred.sed.components["sed_dust_ir"])
    return w, L, pred

wave_ir = np.geomspace(1e4, 1e8, 2000)
S17_NODES = [(20.0, 0.01), (30.0, 0.02), (38.0, 0.05), (42.0, 0.03), (35.0, 0.0)]  # (T_dust [K], f_PAH)
plt.rcParams["figure.dpi"] = 100
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
_s17_resid = []
for _i, ((T, fpah), c) in enumerate(zip(S17_NODES, ["C0", "C1", "C2", "C3", "C4"])):
    w_te, L_te, _ = _dust_emission_build("schreiber2018", 3.0, dust_T=T, dust_f_pah=fpah)
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
axL.set_title("schreiber2018 vs AGNfitter-rX S17 (thick=AF, thin=tengri)")
axL.legend(fontsize=8)
axL.grid(True, alpha=0.3)

w_te16, L_te16, _ = _dust_emission_build("schreiber2016", 3.0, dust_T=35.0, dust_f_pah=0.02)
w_ted, L_ted, _ = _dust_emission_build("dale2014", 3.0, dust_alpha_dale=1.5)
w_s17ref, L_s17ref = A.cold_dust_template("S17", tdust=35.0, fpah=0.02)
axR.loglog(w_s17ref, norm_peak(L_s17ref), "0.6", lw=2.0, alpha=0.6, label="AGNfitter-rX  S17 (ref)")
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
    f"nearest AGNfitter-rX grid node {_dh_node:.3f}; "
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
# terms as `ymodel` does: the GA template at the present mass that holds tengri's formed mass (§1.3), reddened by
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
    ssp_data=ssp, sfh=SFH_FIDUCIAL, met=MET_BC03_SOLAR,
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
_m_present7_af = float(pred7.sfh.stellar_mass) / _f_formed_per_present  # same formed mass, upstream bookkeeping
w_ga, L_ga = A.galaxy_lnu(TAU_GYR, AGE_GYR * 1e9, np.log10(_m_present7_af), ebv=_EBV_GAL)
w_sb, L_sb = A.cold_dust_template("S17", tdust=35.0, fpah=0.02)
_IR_LO, _IR_HI = 8e4, 1e7  # 8-1000 um
_sb_scale = band_power(w7, _sb7_te, _IR_LO, _IR_HI) / band_power(w_sb, L_sb, _IR_LO, _IR_HI)
af_grid = np.geomspace(1e3, 1e8, 3000)
_af_parts, af_host = A.ymodel_sum({"GA": (w_ga, L_ga), "SB": (w_sb, L_sb * _sb_scale)}, af_grid)
te_host_on_grid = U.regrid(w7, np.clip(host_sed, 0, None), af_grid)

fig, (ax, axr) = plt.subplots(2, 1, figsize=(8, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax.loglog(af_grid, af_host, "C0-", lw=3.5, alpha=0.35, solid_capstyle="round",
          label="AGNfitter-rX  GA (matched formed mass, Calzetti) + SB (L_IR-matched)")
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
    f"M*_formed = {pred7.sfh.stellar_mass:.3e}, tengri M*_present = {_m_present7:.3e}, "
    f"upstream-convention M*_present = {_m_present7_af:.3e} Msun"
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
# **Result.** The total infrared power is matched by construction. The stellar bands, which test the
# formed-mass mapping of §1.3 and the Calzetti curve together, agree to 0.3% (0.1-1 µm) and 0.1%
# (1-3 µm). The infrared sub-band rows test the S17 shape at fixed $L_{\rm IR}$: 0.994 at 8-30 µm,
# where the model grid has 220 points, and 1.000 at 30-1000 µm.

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
_HA_LADDER: dict[str, tuple[float, float]] = {}
_ANNOT = dict(transform=None, va="top", ha="left", fontsize=7, family="monospace",
              bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, (af_name, af_kw, tengri_fn, tengri_label) in zip(axes.ravel(), disk_pairs):
    w_a, L_a = A.disk_template(af_name, **af_kw)
    a_norm = norm_at(w_a, L_a, ANCHOR)
    msk_a = (w_a > 5e2) & (w_a < 5e4)
    ax.loglog(w_a[msk_a], a_norm[msk_a], "C0-", lw=4.0, alpha=0.35, solid_capstyle="round", label=f"AGNfitter-rX  {af_name}")
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
        _HA_LADDER["AGNfitter-rX THB21 template"] = halpha_line(w_a, L_a)
        _HA_LADDER["tengri qsogen + blr + FeII"] = halpha_line(w_t, L_t)
        ax.text(0.03, 0.97, "Hα+[N II] line power: see table",
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
print("§2.1  Hα+[N II] bump above its continuum, THB21 panel (each curve on its own grid)")
print("    | curve                                | line power / (nu L_nu at 2500 Å) | equivalent width [Å] |")
for _k, (_pw, _ew) in _HA_LADDER.items():
    print(f"    | {_k:36s} | {_pw:32.4f} | {_ew:20.1f} |")
_ha_up, _ha_te = list(_HA_LADDER.values())
_HA_POWER_RATIO, _HA_EW_RATIO = _ha_te[0] / _ha_up[0], _ha_te[1] / _ha_up[1]
print(f"    tengri / template: line power {_HA_POWER_RATIO:.3f}, equivalent width {_HA_EW_RATIO:.3f}")

# %%
# KD18 grid-tabulated vs warm-index variant, at fixed (M_BH, lambda_Edd): a far warm index
# against kd18_agnfitter's baked-in default (tengri against tengri; the repackaged reference
# holds no warm-index library).
w_kd, L_kd = B.disc("kd18_agnfitter", agn_log_mbh=8.0, agn_log_ledd=-0.75)
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
    raise AssertionError("kd18_agnfitter + xray='yang20' must be refused: the KD18 disc carries its own corona")

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
# **Result.** R06 and SN12 agree with their references (maximum 0.000 dex) and KD18 to 0.002 dex; the
# per-node tables are in §8.1. Only THB21 carries the 0.7 µm Hα + [N II] bump, and tengri reproduces
# its position. With each curve on its own grid, the continuum-subtracted line power relative to
# $\nu L_\nu$ at 2500 Å is 1.34 times the template's and the equivalent width 1.31 times. The cause is
# that the template is a fixed table with unrecorded generating parameters while tengri's `blr` runs
# at its default strengths; freeing `agn_blr_line_efficiency` or `agn_blr_cf` absorbs the offset in a
# fit (§6). The KD18 refusal is tengri's policy, not upstream's: upstream replaces everything above
# 200 eV in any disc by its own power law, which also removes a corona the template may carry.

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
_K_RAW_V = 1.39 * 0.55 ** (-1.2) - 0.38  # Prevot raw fit at V band, AGNfitter-rX's own formula
_R_V_SMC = 2.72  # R_V declared (and unused) by BBBred_Prevot; Prevot et al. (1984)
_CONV = _R_V_SMC / _K_RAW_V  # tengri / AGNfitter-rX A_lambda at equal E(B-V)

w_thb, L_thb = A.disk_template("THB21")
lam_um = np.geomspace(0.1, 3.0, 400)
k_raw = 1.39 * lam_um ** (-1.2) - 0.38

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 4.5))
axl.plot(lam_um, k_raw, "C0-", lw=1.6, label="AGNfitter-rX  $k_{raw}(\\lambda)$")
axl.plot(lam_um, k_raw / _K_RAW_V * _R_V_SMC, "C1--", lw=1.6, label=rf"tengri  $k(\lambda)\,R_V$  ($={_CONV:.3f}\,k_{{raw}}$)")
axl.set_xscale("log")
axl.set_xlabel(r"$\lambda$ [µm]")
axl.set_ylabel(r"$A_\lambda / E(B{-}V)$")
axl.set_title("Prevot SMC law — two normalization conventions")
axl.legend(fontsize=8)
axl.grid(True, alpha=0.3)

_EBV_DEMO = 0.3
w_t0, L_t0 = B.disc("qsogen", ebv_disc=0.0)
w_t3, L_t3 = B.disc("qsogen", ebv_disc=_EBV_DEMO)
ratio_tengri = np.divide(L_t3, L_t0, out=np.ones_like(L_t3), where=L_t0 > 0)
L_thb_red = A.apply_bbb_reddening(w_thb, L_thb, _EBV_DEMO)
ratio_af = np.divide(L_thb_red, L_thb, out=np.ones_like(L_thb_red), where=L_thb > 0)
w_tr, L_tr = B.disc("qsogen", ebv_disc=_EBV_DEMO * _K_RAW_V / _R_V_SMC)
ratio_tengri_rescaled = np.divide(L_tr, L_t0, out=np.ones_like(L_tr), where=L_t0 > 0)

msk_t = (w_t0 > 8e2) & (w_t0 < 3e4)
msk_a = (w_thb > 8e2) & (w_thb < 3e4)
axr.semilogx(w_thb[msk_a], -2.5 * np.log10(ratio_af[msk_a]), "C0-", lw=3.5, alpha=0.35,
             solid_capstyle="round", label="AGNfitter-rX  BBBred_Prevot")
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
    f"§2.2  matched E(B-V)={_EBV_DEMO}: max|A_tengri - A_AGNfitter-rX| = "
    f"{np.max(np.abs(_a_t - _a_af)):.3f} mag  (ratio A_t/A_af median = "
    f"{np.median(_a_t / _a_af):.4f}, expected {_R_V_SMC / _K_RAW_V:.4f})"
)
print(
    f"§2.2  rescaled E(B-V)={_EBV_DEMO}/{_CONV:.4f}: max|A_tengri - A_AGNfitter-rX| = "
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
w_q1, L_q1 = B.disc_atten("qsogen", "qsogen", 0.3)
ratio_qsogen = np.divide(L_q1, L_q0, out=np.ones_like(L_q1), where=L_q0 > 0)
# A_lambda/E(B-V) = -2.5 log10(ratio) / E(B-V)
A_over_ebv_qsogen = -2.5 * np.log10(np.clip(ratio_qsogen, 1e-300, None)) / 0.3

_k_raw_ext_v = 1.39 * (_wl_ext / 1e4) ** (-1.2) - 0.38
_k_af_ext = _k_raw_ext_v
_k_tengri_ext = _k_raw_ext_v / _K_RAW_V * _R_V_SMC
_A_qsogen_on_grid = np.interp(_wl_ext, w_q0, A_over_ebv_qsogen)

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(_wl_ext, _k_af_ext, "C0-", lw=1.6, label=rf"AGNfitter-rX  Prevot SMC ($R_V^{{\rm eff}}={_K_RAW_V:.3f}$)")
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
    f"§2.2  A_V/E(B-V) at V=5500 Å:  AGNfitter-rX={_at_ext(_k_af_ext, 5500):.3f}  "
    f"tengri-Prevot={_at_ext(_k_tengri_ext, 5500):.3f}  tengri-qsogen={_at_ext(_A_qsogen_on_grid, 5500):.3f}"
)
print(
    f"§2.2  A(1500)/A(V) (UV steepness):  AGNfitter-rX={_at_ext(_k_af_ext, 1500) / _at_ext(_k_af_ext, 5500):.2f}  "
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
    print(f"  E(B-V)={ebv:g}:  AGNfitter-rX = {a_af:.2f}   tengri = {a_te:.2f}   (ratio {a_te / a_af:.3f}, §2.2 convention {_CONV:.3f})")

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

# %%
torus_pairs = [
    ("S04", "S04", lambda: B.torus("silva04", log_nh_silva=23.0), "silva04 (log N_H = 23)", dict(log_nh=23.0)),
    ("NK08", "NK08", lambda: B.torus("nenkova_agnfitter", cos_inc=0.8660254), "nenkova_agnfitter (incl 30°)", dict(incl=30.0)),
    ("SKIRTOR", "SKIRTOR", lambda: B.torus("skirtor", cos_inc=0.8660254, oa_skirtor=40.0, tau_skirtor=7.0), "skirtor (oa 40°, incl 30°, τ 7)", dict(oa=40.0, incl=30.0, tau=7.0)),
    ("CAT3D", "CAT3D-Wind", lambda: B.torus("cat3d_wind", cos_inc=1.0, a_cat3d=-2.0, fwd_cat3d=1.75), "cat3d_wind (incl 0°, a −2, f_wd 1.75)", dict(incl=0.0, a=-2.0, fwd=1.75)),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
_TORUS_SAMPLING = {}
for ax, (af_name, title, tengri_fn, tengri_label, af_kw) in zip(axes.ravel(), torus_pairs):
    w_a, L_a = A.torus_template(af_name, **af_kw)
    msk_a = (w_a > 5e3) & (w_a < 1e7)
    ax.loglog(w_a[msk_a], norm_peak(L_a)[msk_a], "C0-", lw=4.0, alpha=0.35, solid_capstyle="round", label=f"AGNfitter-rX  {af_name}")
    w_t, L_t = tengri_fn()
    _TORUS_SAMPLING[af_name] = (np.asarray(w_a), np.asarray(L_a), np.asarray(w_t), np.asarray(L_t))
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
# own template axis, not on the reference wavelengths. The first table prints both samplings and
# the range of each; the second measures what the sampling costs a broadband flux, by integrating
# each reference over a band using only tengri's own wavelengths (linear interpolation between
# them) and dividing by its full integral; the floor row halves the reference sampling and
# compares it with itself.

# %%
print("§3  sampling of the reference arrays and of tengri's model grid")
print("    | library  | reference n | uniform in log λ | reference range [um] | tengri n (> 10 um) | tengri grid max [um] |")
for _name, (_wa, _la, _wt, _lt) in _TORUS_SAMPLING.items():
    _dl = np.diff(np.log10(np.sort(_wa)))
    _uniform = bool(np.allclose(_dl, _dl.mean(), rtol=1e-2))
    print(
        f"    | {_name:8s} | {_wa.size:11d} | {str(_uniform):16s} | "
        f"{_wa.min() / 1e4:8.3g} - {_wa.max() / 1e4:<9.3g} | {int(np.sum(_wt > 1e5)):18d} | {_wt.max() / 1e4:20.4g} |"
    )

_BANDS_UM = [(3.0, 10.0), (10.0, 30.0), (30.0, 100.0), (100.0, 300.0), (300.0, 1000.0)]
print("§3  broadband power from tengri's own wavelengths / full reference integral (peak-normalized)")
print("    | library  | " + " | ".join(f"{lo:g}-{hi:g} um" for lo, hi in _BANDS_UM) + " |")
_SAMPLING_COST = {}
for _name, (_wa, _la, _wt, _lt) in _TORUS_SAMPLING.items():
    _oa = np.argsort(_wa)
    _la_n = norm_peak(_la)[_oa]
    _wt_in = np.sort(_wt[(_wt >= _wa.min()) & (_wt <= _wa.max())])
    _l_on_wt = np.interp(np.log10(_wt_in), np.log10(_wa[_oa]), _la_n)
    _ratios = []
    for _lo, _hi in _BANDS_UM:
        _full = band_power(_wa[_oa], _la_n, _lo * 1e4, _hi * 1e4)
        _ratios.append(band_power(_wt_in, _l_on_wt, _lo * 1e4, _hi * 1e4) / _full if _full > 0 else np.nan)
    _SAMPLING_COST[_name] = _ratios
    print(f"    | {_name:8s} | " + " | ".join(f"{_r:{len(f'{lo:g}-{hi:g} um')}.4f}" for _r, (lo, hi) in zip(_ratios, _BANDS_UM)) + " |")
_floor_rows = []
for _name, (_wa, _la, _wt, _lt) in _TORUS_SAMPLING.items():
    _oa = np.argsort(_wa)
    _wa_s, _la_s = _wa[_oa], norm_peak(_la)[_oa]
    _half = np.interp(np.log10(_wa_s), np.log10(_wa_s[::2]), _la_s[::2])
    _ok = (_wa_s > 1e4) & (_wa_s < 1e6) & (_la_s > 1e-3)
    _floor_rows.append(float(np.max(np.abs(_half[_ok] / _la_s[_ok] - 1.0))))
print(
    "§3  reference-sampling floor (reference at half its sampling vs itself, 1-100 um, L/L_peak > 1e-3): max|ratio-1| = "
    + ", ".join(f"{_n} {_f:.2g}" for _n, _f in zip(_TORUS_SAMPLING, _floor_rows))
)

# %% [markdown]
# **Reading the sampling tables.** Each block ends at its own template limit (S04 at 948 µm, NK08
# at 1000 µm, SKIRTOR at $10^4$ µm, CAT3D at $3.6\times10^4$ µm); the model holds no torus flux beyond
# it, which is outside every band fitted here. Integrated over the full band from its own nodes
# (plus the band edges), every block reproduces the reference band power to 0.6% or better, SKIRTOR
# included (0.994-1.000) despite its 68 points above 10 µm. The sampling therefore costs no
# broadband flux at these bands. The floor set by the reference sampling is 1-2% (the reference at half
# its sampling against itself), and bounds how closely any peak-normalized case can agree.

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
# **Result.** The tabulated blocks reproduce their reference nodes: S04, NK08, the averaged SKIRTOR and
# the high-wind CAT3D nodes agree to 1% at worst (0.0098, 0.0073, 0.0002, 0.0074), and the five
# further reductions to 1e-3 dex in the median (0.00096). The low-wind half of the CAT3D union has a
# median of 1.4e-4 and a maximum of 0.104 at 1.45 µm. §8.3 shows that this is interpolation between
# sparse reference nodes (the reference spaces its nodes 0.085 dex apart there, tengri 0.0015 dex),
# where the reference is at 0.014 of its peak: at each reference node tengri equals the reference to
# four digits, and the 1-5 µm band power agrees to 1.7%. The full unaveraged SKIRTOR grid differs
# from the averaged reference by construction: it carries the clumpiness and radial structure that
# the average removes, which shifts the IR peak (printed). The union of the two tengri CAT3D blocks
# covers the one upstream library, and the polar wind that CAT3D-Wind is for supplies the
# near-infrared excess that equatorial tori miss.

# %% [markdown]
# ### Inclination-dependent torus power
#
# Upstream's torus templates do not share a bolometric luminosity: the template integral changes
# with inclination (the table prints the trend for each library), so at fixed `TO` the torus
# radiates a different power at each inclination. tengri normalizes every
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
_TORUS_POWER_TREND = {}
for _name, _incl_ax, _up_fn, _te_fn in _bol_sets:
    _up0 = _up_fn(float(_incl_ax[0]))
    _te0 = band_power(*_te_fn(float(_incl_ax[0])), 1e3, 1e9)
    _up_r, _te_r = [], []
    for _i in _incl_ax:
        _w_b, _L_b = _te_fn(float(_i))
        _up_r.append(_up_fn(float(_i)) / _up0)
        _te_r.append(band_power(_w_b, _L_b, 1e3, 1e9) / _te0)
        print(f"    | {_name:8s} | {float(_i):10.1f} | {_up_r[-1]:26.4f} | {_te_r[-1]:18.4f} |")
    _TORUS_POWER_TREND[_name] = (min(_up_r), max(_up_r), min(_te_r), max(_te_r))

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
ax.plot(np.log10(l2500), aox_ref, "k--", lw=1.2, label="AGNfitter-rX  Just+2007")
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
print(f"§4 α_ox parity: max |tengri just2007 − AGNfitter-rX| = {_aox_dmax:.2e}")
fig.tight_layout()
save_fig("agnfitter_10c_alphaox_residual.png")
plt.show()

# %% [markdown]
# Given the 2500 Å disc luminosity, both codes then build the same bare corona ($\Gamma = 1.8$,
# 300 keV cutoff, normalized through the relation). Left: tengri's
# `xray_agn_corona_from_disc(..., apply_anisotropy=False, log_nh=0)` against upstream's disc X-ray
# extension. Right: what tengri's *defaults* add, none of which AGNfitter-rX has an analog for:
# the Yang et al. (2022) viewing-angle anisotropy, a default absorbing column, and a host
# X-ray-binary floor (`xray_xrb`, Mineo et al. 2014). The table gives the ratio in a hard window
# (0.5-100 keV) and a soft one (0.2-0.5 keV), across $\Delta\alpha_{\rm ox}$ and $\Gamma$, with
# the column at its default and switched off.

# %%
from tengri.xray import xray_agn_corona_from_disc, xray_xrb

L_2500 = 1.0e30
wave_x = np.geomspace(1e-2, 1e2, 600)
L_corona_bare = np.asarray(
    xray_agn_corona_from_disc(jnp.asarray(wave_x), L_2500, delta_alpha_ox=0.0, apply_anisotropy=False, log_nh=0.0)
)
L_corona_default = np.asarray(xray_agn_corona_from_disc(jnp.asarray(wave_x), L_2500, delta_alpha_ox=0.0, cos_inc=1.0))
w_thb, L_thb = A.disk_template("THB21")
L_thb_at_2500 = norm_at(w_thb, L_thb, 2500.0) * L_2500
xw_af, xL_af = A.disk_xray_extension(w_thb, L_thb_at_2500, scatter=0.0)
L_xrb = np.asarray(xray_xrb(jnp.asarray(wave_x), sfr=5.0, stellar_mass=1e10))

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 4.8))
axl.loglog(xw_af, xL_af, "C0-", lw=1.5, label="AGNfitter-rX  disk X-ray extension")
axl.loglog(wave_x, L_corona_bare, "C1--", lw=1.5, label="tengri  corona (anisotropy and N_H off)")
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
axr.semilogx(wave_x, _ratio_bare, "C1-", lw=1.5, label="bare (anisotropy and N_H off)")
axr.semilogx(wave_x, _ratio_default, "C3-", lw=1.5, label="tengri defaults (Yang+22 anisotropy, face-on)")
axr.axhline(1.0, color="0.5", lw=0.8)
axr.set_xlim(1e-2, 1e2)
axr.set_ylim(0.8, 1.3)
axr.set_xlabel(r"$\lambda$ [Å]")
axr.set_ylabel("tengri / AGNfitter-rX")
axr.set_title("Corona ratio — what tengri's defaults add")
axr.legend(fontsize=8)
axr.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_10b_xray_corona.png")

# %%
_hard = (wave_x > 0.12) & (wave_x < 25.0) & np.isfinite(_ratio_bare)
print(
    f"§4 corona parity (0.5-100 keV, anisotropy and N_H off): bare median ratio = {np.nanmedian(_ratio_bare[_hard]):.4f}, "
    f"max |ratio-1| = {np.nanmax(np.abs(_ratio_bare[_hard] - 1.0)):.3f}; "
    f"defaults (face-on anisotropy, log N_H = 20) median ratio = {np.nanmedian(_ratio_default[_hard]):.4f}"
)

# %%
from tengri.xray import alpha_ox_from_l2500, xray_agn_corona_from_disc

_m10 = SEDModel.build(
    ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST,
    xray={"type": "yang20", "all_params": Fixed(DEFAULT)}, neb={"type": "none"}, redshift=Fixed(0.0),
)

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


_xray_rows = {20.0: [], 0.0: []}
_xray_stats = {20.0: [], 0.0: []}
for scat, gam in [(s_, 1.8) for s_ in _scat_grid] + [(0.0, 1.6), (0.0, 2.0)]:
    for lognh in (20.0, 0.0):
        r = _corona_ratio(scat, gam, lognh)
        rh, rs = r[_hard_10a], r[_soft_10a]
        _xray_rows[lognh].append({"label": f"Δα_ox={scat:+.1f} Γ={gam:.1f}", "max_abs_dev": float(np.nanmax(np.abs(rh - 1.0)))})
        _xray_stats[lognh].append((np.nanmedian(rh), np.nanmax(np.abs(rh - 1.0)), np.nanmedian(rs), np.nanmax(np.abs(rs - 1.0))))
print(
    f"§4  X-ray corona, tengri / AGNfitter-rX (apply_anisotropy=False), {len(_xray_stats[0.0])} cases: "
    "Δα_ox in {-0.4..+0.4} at Γ = 1.8, and Γ = 1.6, 2.0 at Δα_ox = 0; range over the cases"
)
print("    | log N_H | 0.5-100 keV median | 0.5-100 keV max|r-1| | 0.2-0.5 keV median | 0.2-0.5 keV max|r-1| |")
for _nh, _st in _xray_stats.items():
    _a = np.array(_st)
    print(
        f"    | {_nh:7.0f} | {_a[:, 0].min():8.4f} - {_a[:, 0].max():<8.4f} | {_a[:, 1].min():8.4f} - {_a[:, 1].max():<8.4f}"
        f" | {_a[:, 2].min():8.4f} - {_a[:, 2].max():<8.4f} | {_a[:, 3].min():8.4f} - {_a[:, 3].max():<8.4f} |"
    )

# %% [markdown]
# **Result.** With anisotropy and absorption off, the power law, the cutoff and the 2 keV anchor agree
# to 0.9% (median ratio 1.0094, maximum 0.0094) in both windows, and the ratio does not depend on
# $\Delta\alpha_{\rm ox}$ or $\Gamma$ (the range columns collapse to one value). The default
# absorbing column lowers the 0.2-0.5 keV median ratio to 0.776 and leaves 0.5-100 keV at 1.0091; the
# default face-on anisotropy multiplies the corona by 1.072 (§6). Both are tengri choices, not
# discrepancies. Set `xray_log_nh` to 0 in the build for the unabsorbed corona; in a composable build
# the anisotropy follows `agn_cos_inc` (shared with the torus; 1 at 30°), and `apply_anisotropy=False`
# exists only on the function used for this comparison. §6 gives the disc-to-corona join, where the
# codes differ by construction.

# %% [markdown]
# ## 5 Radio
#
# AGNfitter-rX models the AGN core and jet with a simple power law (SPL, slope fixed at $-0.75$
# with an exponential cutoff at $10^{13}$ Hz) or a double power law (DPL, paper Eq. 2); tengri
# ships both, as `radio_agn` and `radio_agn_dpl`. The upstream curves are the paper's equations
# evaluated by the driver, each normalized at 5 GHz.

# %%
from tengri.radio import radio_agn, radio_agn_dpl

_m11 = SEDModel.build(
    ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST,
    radio={"sf": {"type": "bell2003", "all_params": Fixed(DEFAULT)},
           "agn": {"type": "dpl", "all_params": Fixed(DEFAULT)}},
    neb={"type": "none"}, redshift=Fixed(0.0),
)

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
ax.loglog(freq / 1e9, _norm5(freq, F_spl_af), "C0:", lw=1.4, label="AGNfitter-rX  SPL")
ax.loglog(freq / 1e9, np.where(t_dpl > 0, t_dpl, np.nan), "C1-", lw=1.6, label="tengri  radio_agn_dpl")
ax.loglog(freq / 1e9, _norm5(freq, F_dpl_af), "C1:", lw=1.4, label="AGNfitter-rX  DPL")
ax.set_ylabel(r"$L_\nu$ (norm. at 5 GHz)")
ax.set_title("Radio parity (0.1–300 GHz band)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
axr.axhline(1.0, color="0.5", lw=0.8)
axr.semilogx(freq / 1e9, ratio_spl, "C0-", lw=1.2, label="SPL")
axr.semilogx(freq / 1e9, ratio_dpl, "C1-", lw=1.2, label="DPL")
axr.set_ylim(0.95, 1.05)
axr.set_ylabel("tengri / AGNfitter-rX", fontsize=9)
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
# the conservative value $2.64 + \sigma$; the cell measures the value the repackaged template
# actually embeds and prints its difference from that. tengri's `radio_sfr_bell2003_split` is the
# matching function and is run at the measured value. The model's `bell2003` block calibrates the same
# quantity, the total 1.4 GHz luminosity, but takes its thermal share from the Murphy et al. (2011)
# free-free term and its synchrotron slope from `radio_alpha_sf`, so it is not compared here.
#
# **Caveat:** at tengri's default $q_{\rm IR} = 2.64$ instead of the template's value, the 1.4 GHz
# luminosity of the same $L_{\rm IR}$ is higher by the factor printed below.

# %%
from tengri.radio import radio_sfr_bell2003_split

_Q_BELL = float(PAPER_QUOTED["q_IR adopted from Bell (2003)"][0])
_SIGMA_Q = float(PAPER_QUOTED["scatter on q_IR"][0])
Q_IR_PAPER = _Q_BELL + _SIGMA_Q  # the paper's conservative choice

w_afr, L_afr = A.cold_dust_radio_template(tdust=35.0, fpah=0.02)
_axes_radio = A.cold_dust_radio_axes()
_t_idx = int(np.argmin(np.abs(_axes_radio["tdust"] - 35.0)))
L_IR_NODE = float(_axes_radio["lir_conv"][_t_idx])  # already erg/s (see driver docstring)

# q_IR measured from the template itself (paper Eq. 1).
_nu_afr = U.C_ANGSTROM_PER_S / w_afr
_o_afr = np.argsort(_nu_afr)
_L14_afr = float(np.interp(1.4e9, _nu_afr[_o_afr], L_afr[_o_afr]))
Q_IR_TEMPLATE = float(np.log10(band_power(w_afr, L_afr, 8e4, 1e7) / (3.75e12 * _L14_afr)))
Q_IR_PARITY = Q_IR_TEMPLATE  # the parity mode runs at the value the template embeds
print(
    f"§5  q_IR measured from the S17_radio template (T_dust = 35 K) = {Q_IR_TEMPLATE:.4f};  "
    f"paper value {_Q_BELL} + {_SIGMA_Q} = {Q_IR_PAPER:.3f};  difference = {Q_IR_TEMPLATE - Q_IR_PAPER:+.4f}"
)

wave_all = np.geomspace(1e4, 3e9, 1200)
w_te_sb, L_dust_shape, _ = _dust_emission_build("schreiber2018", 3.0, dust_T=35.0, dust_f_pah=0.02)
L_dust = np.interp(wave_all, w_te_sb, L_dust_shape, left=0.0, right=0.0)
L_dust = L_dust * (L_IR_NODE / abs(band_power(wave_all, L_dust, 8e4, 1e7)))
L_radio_split = np.asarray(radio_sfr_bell2003_split(jnp.asarray(wave_all), L_IR_NODE, q_ir=Q_IR_PARITY))
L_te_total = L_dust + L_radio_split

fig, ax = plt.subplots(figsize=(8.2, 4.8))
msk_af = (w_afr > 1e4) & (w_afr < 3e9)
ax.loglog(w_afr[msk_af], L_afr[msk_af] / np.max(L_afr[msk_af]), "C0-", lw=2.2, alpha=0.5, label="AGNfitter-rX  S17_radio (Bell 2003, 90/10 split)")
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
# **Result.** The SPL and DPL AGN radio agree with the upstream equations to 1.0e-4 and 5.0e-5, and
# the parameter grids of §8.4 stay within 1.4e-4. The star-formation radio in the parity mode, run at
# the $q_{\rm IR} = 2.915$ the template embeds (0.015 above the paper's quoted 2.90), follows the
# template over 1.1-30 GHz to 0.09%. Users who keep tengri's default $q_{\rm IR}$ get the factor
# printed above, 1.884, at 1.4 GHz.

# %% [markdown]
# ## 6 Where tengri and AGNfitter-rX differ, and why
#
# Everything on this page that is not an agreement, in one place. The first group is upstream
# behavior that tengri deliberately does not reproduce; the second is conventions and terms where
# the two codes are built differently. The effect sizes are printed by the cell below from the
# sections that measure them; a dash means the item is structural and has no single scalar.

# %%
from scipy.integrate import quad
from tengri.xray import xray_anisotropy

_i_up = lambda lam_aa: float(np.interp(lam_aa, _w_up, _A_up))
_neg_um = float(_neg[0]) / 1e4
_boost_10um = 10 ** (-0.4 * _i_up(1e5) * _EBV_GAL)
_A_te_5um = float(np.interp(5e4, _wl_cal, A_lambda_gal_on_grid))
_k_uv_up, _k_uv_an = _i_up(950.0), float(np.interp(950.0, _wl_cal, _k_analytic))
_w_bb, _L_bb = A.bbb_with_xrays(w_thb, L_thb_at_2500, ebv=0.0, scatter=0.0)
_o_bb = np.argsort(_w_bb)
_w_bb, _L_bb = np.asarray(_w_bb)[_o_bb], np.asarray(_L_bb)[_o_bb]
_i_join = int(np.searchsorted(_w_bb, 62.0))
_join_step = float(_L_bb[_i_join - 1] / _L_bb[_i_join])

# Upstream's energy-balance prior, evaluated as written on the fiducial template. The stored
# GALAXY array is 1e18 times the physical template (`renorm_template('GA')` divides by 1e18); the
# prior integrates the stored array unreddened but passes it times 1e18 to the reddening routine,
# so its "absorbed" term is the difference of two different normalizations.
_w_gu, _L_gu = A.galaxy_template(TAU_GYR, AGE_GYR * 1e9)
_raw = _L_gu * 1e18
_nu_gu = U.C_ANGSTROM_PER_S / _w_gu
_o_gu = np.argsort(_nu_gu)
_int = lambda arr: float(np.trapezoid(np.asarray(arr)[_o_gu], _nu_gu[_o_gu]))
_I_raw = _int(_raw)
_I_up_red = _int(A.galaxy_redden_calzetti(_w_gu, _raw * 1e18, _EBV_GAL)[1])
_I_raw_red = _int(A.galaxy_redden_calzetti(_w_gu, _raw, _EBV_GAL)[1])
_eb_ratio = abs(_I_raw - _I_up_red) / (_I_raw - _I_raw_red)
_worst_low = max((r for r in _union_table if r[1] == "CAT3D_LOWFWD"), key=lambda r: r[4])

# Upstream's maximal_age(z), as written: the observing-redshift terms sit outside the integrand.
_Z_AGE = 0.5
_a_age = 1.0 / (1.0 + _Z_AGE)
_E_age = 0.266 * (1 + _Z_AGE) ** 3 + (1.0 - 0.266)
_H_sec = 70.0 / 3.0857e19
_t_up_age = quad(lambda _z: _a_age / np.sqrt(_E_age), _Z_AGE, 1089)[0] / _H_sec / 31556926 / 1e9 - 1.0
_t_cosmic = float(cosmo.age_at_z(_Z_AGE, h0=70.0, om0=0.266)) - 1.0

# Corona anchor: tengri's corona follows the disc continuum, upstream's amplitude is set from
# disc + lines. Read on the winning model's AGN (face-on, as in section 9).
_m_anchor = SEDModel.build(
    ssp_data=ssp, sfh=SFH_FIDUCIAL, met=MET_BC03_SOLAR, dust_attenuation=NO_DUST,
    agn={
        "type": "composable",
        "disc": {"type": "qsogen", "agn_ebv_disc": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "none"}, "blr": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
        "feii": {"type": "qsogen_balmer", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "cat3d_wind", "cos_inc": Fixed(1.0), "a_cat3d": Fixed(-2.0),
                  "fwd_cat3d": Fixed(1.75), "all_params": Fixed(DEFAULT)},
        "atten": {"type": "none"}, "agn_log_lbol": Fixed(11.0),
        "all_params": Fixed(DEFAULT), "norm": "independent",
    },
    neb={"type": "none"}, redshift=Fixed(0.0),
)
_c_anchor = _m_anchor.predict({}).sed.components
_w_anchor = np.asarray(_c_anchor["wavelength"])
_l2500_disc = val_at(_w_anchor, _c_anchor["sed_agn_disc"], 2500.0)
_l2500_full = val_at(_w_anchor, np.asarray(_c_anchor["sed_agn"]) - np.asarray(_c_anchor["sed_agn_torus"]), 2500.0)
_wave_x6 = jnp.geomspace(1e-2, 1e2, 600)
_cor = lambda l2500: float(np.interp(6.199, np.asarray(_wave_x6), np.asarray(
    xray_agn_corona_from_disc(_wave_x6, l2500, delta_alpha_ox=0.0, gamma=1.8, apply_anisotropy=False, log_nh=0.0))))
_anchor_factor = _cor(_l2500_disc) / _cor(_l2500_full)
_aniso_faceon = float(xray_anisotropy(1.0, 1.0))
_cat_up_lo, _cat_up_hi, _cat_te_lo, _cat_te_hi = _TORUS_POWER_TREND["CAT3D"]

_UPSTREAM = [
    ("Prevot disc screen: A_t / A_AF at equal E(B-V)", f"x{_CONV:.4f}",
     "applies the declared R_V", "divide E(B-V)_BBB by this factor when moving a posterior"),
    ("Calzetti below 0.12 um: A/E(B-V) at 0.095 um", f"{_k_uv_up:.2f} upstream vs {_k_uv_an:.2f} analytic",
     "analytic curve, no second R_V", "do not compare galaxy attenuation below 0.12 um"),
    ("Calzetti above 2.2 um: upstream k turns negative", f"at {_neg_um:.2f} um; x{_boost_10um:.3f} flux at 10 um",
     f"no boost; A = {_A_te_5um:.2f} mag at 5 um", "restrict upstream comparisons to 0.12-2.2 um"),
    ("Disc-to-corona join at 200 eV: L(below 62 A) / L(above)", f"{_join_step:.3g}",
     "smooth corona; the KD18 family carries its own", "expect a step at 62 A in AGNfitter-rX only"),
    ("energy-balance prior: upstream absorbed term / physical absorbed power", f"x{_eb_ratio:.3g}",
     "physical balance (`dust_eta_balance`)", "do not carry upstream's prior tolerance over"),
    ("maximal_age prior at z = 0.5: age cap vs cosmic age - 1 Gyr", f"{_t_up_age:.3g} Gyr vs {_t_cosmic:.2f} Gyr",
     "age limit applied at the observing redshift", "expect a tighter age prior"),
    ("Charlot & Fall (2000) two-screen law", "-",
     "available in tengri's attenuation components", "no working upstream counterpart to compare"),
]
_CONVENTIONS = [
    ("GA mass: surviving fraction M_present / M_formed, tengri vs BC03 template (section 1.3)",
     f"{_surv_frac:.4f} vs {_surv_up:.4f} (x{_surv_frac / _surv_up:.3f})",
     "SSP file carries no mass-loss table: DSPS Chabrier fit to FSPS, same at every Z",
     "convert with the section 1.3 relation; the light per formed mass agrees"),
    ("qsogen's own quasar curve: A_V / E(B-V)", f"{_at_ext(_A_qsogen_on_grid, 5500):.2f} (Prevot {_at_ext(_k_tengri_ext, 5500):.2f})",
     "different law, own convention", "use `atten={'type': 'qsogen'}` only with a qsogen disc"),
    ("THB21 H-alpha + [N II] line power / (nu L_nu at 2500 A), tengri / template",
     f"x{_HA_POWER_RATIO:.2f} (equivalent width x{_HA_EW_RATIO:.2f})",
     "qsogen blr at default line strengths; the template's were generated with unrecorded parameters",
     "free `agn_blr_line_efficiency` (and `agn_blr_cf`) to absorb the offset in a fit"),
    ("CAT3D low-wind union half: max abs(t/a - 1)", f"{_worst_low[4]:.3f} at {_worst_low[5] / 1e4:.2f} um",
     "between sparse reference nodes (section 8.3); exact at the nodes", "band power over 1-5 um agrees to 1.7% (section 8.3)"),
    ("Corona anchor: tengri 2 keV corona / corona anchored on disc + lines", f"x{_anchor_factor:.3f}",
     "corona follows the disc continuum L(2500), upstream's amplitude is set from disc + lines",
     "expect this offset in a capstone-style comparison"),
    ("Corona anisotropy at face-on (the Yang+22 factor)", f"x{_aniso_faceon:.3f}",
     "`agn_cos_inc` sets it (shared with the torus); 30 deg gives 1", "no upstream analog"),
    ("Torus amplitude: CAT3D template integral across inclination",
     f"x{_cat_up_lo:.2f} - x{_cat_up_hi:.2f} upstream; x{_cat_te_lo:.2f} - x{_cat_te_hi:.2f} tengri",
     "same power at every node", "convert TO with the tabulated ratio (section 3)"),
    ("Host nebular emission, IGM transmission", "-", "optional components, off here", "enable them for real data"),
]
print("§6  where tengri and AGNfitter-rX differ")
print("    | item | effect | tengri | what to do |")
print("    |---|---|---|---|")
_n_up = len(_UPSTREAM)
for _group, _rows in (("upstream behavior tengri does not reproduce", _UPSTREAM), ("conventions and library differences", _CONVENTIONS)):
    print(f"    | **{_group}** | | | |")
    for _item, _eff, _tengri, _do in _rows:
        print(f"    | {_item} | {_eff} | {_tengri} | {_do} |")

# %% [markdown]
# **Reading the table.** The first group is upstream behavior that tengri does not reproduce:
# tengri implements the intended version, and a posterior moved from AGNfitter-rX should be
# re-derived under it. The second is conventions and library differences, the ones a translation
# must carry: the surviving mass fraction (12%), the qsogen reddening law, the
# Hα + [N II] line strength of the THB21 template, the corona anchor and its face-on anisotropy,
# and the torus amplitude. The CAT3D low-wind entry is an interpolation excursion that moves the
# 1-5 µm band power by under 2%. The terms AGNfitter-rX does not model at all (host nebular emission,
# IGM) are switched off on this page, so the comparison is like for like.

# %% [markdown]
# ## 7 Translating an AGNfitter-rX fit into tengri
#
# A fit result gives AGNfitter-rX parameter values; tengri wants a `SEDModel.build` call. The
# table maps each AGNfitter-rX parameter to its tengri counterpart and the conversion, with the
# live constants printed; each tengri name is asserted against the built model, so a renamed
# parameter fails the notebook. The call below it builds the paper's winning combination
# (THB21-like disc, CAT3D-Wind torus, $\alpha_{\rm ox}$ corona, DPL jet, S17 cold dust, star-formation
# radio) on the fiducial host, with the corona unabsorbed as section 4 advises; section 9 compares it with AGNfitter-rX end to end.

# %%
_CAT3D_NODE = dict(incl=0.0, a=-2.0, fwd=1.75)
_LOG_LBOL_CAP = 11.0
_AGN_RADIO = builders.radio.agn.dpl(other_params=Fixed(DEFAULT))


def winning_model(radio):
    """The paper's winning AGN combination on the fiducial host, one ``SEDModel.build``.

    The corona is unabsorbed (``xray_log_nh`` = 0), as section 4 advises for a comparison with
    AGNfitter-rX; its anisotropy follows ``agn_cos_inc``.
    """
    return SEDModel.build(
        ssp_data=ssp, sfh=SFH_FIDUCIAL, met=MET_BC03_SOLAR,
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
        xray={"type": "yang20", "xray_log_nh": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        radio=radio, neb={"type": "none"}, redshift=Fixed(0.0),
    )


# Q_IR_PARITY enters the star-formation radio; defined in section 5.
m_cap = winning_model({"radio_q_ir": Fixed(Q_IR_PARITY), "sf": {"type": "bell2003_split"}, "agn": _AGN_RADIO})
resolved_params(m_cap)

from tengri.utils.physics_constants import L_SUN

_cm = m_cap.predict({}).sed.components
_frac_meas = band_power(np.asarray(_cm["wavelength"]), _cm["sed_agn_torus"], 1e3, 1e9) / (10**_LOG_LBOL_CAP * L_SUN)
_wq11, _Lq11 = B.qsogen_full(log_lbol=11.0)
_wq115, _Lq115 = B.qsogen_full(log_lbol=11.5)
_dlog_l2500 = float(np.log10(val_at(_wq115, _Lq115, 2500.0) / val_at(_wq11, _Lq11, 2500.0)) / 0.5)
_MAP = [
    # (AGNfitter-rX, tengri names asserted on the model, conversion)
    ("tau, age (GALAXY SFH)", ["sfh_declining_exp_tau_gyr", "sfh_declining_exp_age_gyr"],
     "direct; tengri age in Gyr, upstream in yr"),
    ("GA (log amplitude of the unit template)", ["sfh_declining_exp_log_total_mass"],
     "log_total_mass = log10(M_present x f), M_present = 10^GA 4 pi d_L^2 / [L_sun (1+z)] / 1e18, "
     f"f = SFR(age) tau (e^(age/tau) - 1) from the template's SFR table ({_f_formed_per_present:.4f} at the "
     f"fiducial node). The formed mass is not M_present / {_surv_frac:.3f}: tengri's surviving fraction "
     f"differs from the template's {_surv_up:.3f} (section 1.3)"),
    ("metal (BC03_metal axis)", ["met_logzsol"],
     f"met_logzsol = log10(Z_BC03 x {Z_SUN_BC03} / {Z_SUN_TENGRI})"),
    ("EBVgal", ["dust_tau_v"],
     f"dust_tau_v = R_V E(B-V)/1.086 with R_V = 4.05 (E(B-V) = 0.3 gives {_tau_v_gal:.3f})"),
    ("EBVbbb", ["agn_ebv_disc"],
     f"agn_ebv_disc = E(B-V)_AF / {_CONV:.4f} (Prevot convention, section 2.2)"),
    ("SB, tdust, fpah (S17)", ["dust_T", "dust_f_pah", "dust_eta_balance"],
     "dust_T, dust_f_pah direct. SB is not a parameter: the infrared power is eta x the absorbed power, "
     "so free `dust_eta_balance` = L_IR,AF / L_absorbed to reproduce a fitted SB"),
    ("BB (disc amplitude)", ["agn_log_lbol"],
     "agn_log_lbol = 11 + log10[L_nu,AF(2500 A) / L_nu,tengri(2500 A; agn_log_lbol = 11)] "
     f"(L_2500 scales as the luminosity to the power {_dlog_l2500:.3f}); section 9 sets BB the other way round"),
    ("TO (torus amplitude), inclination, a, f_wd", ["agn_torus_frac", "agn_cos_inc", "agn_a_cat3d", "agn_fwd_cat3d"],
     "agn_torus_frac = P_TO / (10^agn_log_lbol L_sun) with P_TO = 10^TO x (integral of the unit template T_TO at the "
     "fitted inclination, nu in Hz), so the inclination dependence of the template integral enters through P_TO "
     f"(CAT3D: x{_cat_up_lo:.2f} to x{_cat_up_hi:.2f} over inclination, section 3); agn_cos_inc = cos(incl_AF [deg]) "
     "(tengri takes a cosine, upstream degrees); agn_a_cat3d, agn_fwd_cat3d direct. "
     f"Measured on this model: torus power / (10^lbol L_sun) = {_frac_meas:.4f}"),
    ("scatter on alpha_ox, Gamma, N_H", ["xray_delta_alpha_ox", "xray_gamma_agn", "xray_log_nh"],
     "direct, with xray_log_nh = 0 for the unabsorbed upstream corona. The corona anisotropy follows agn_cos_inc "
     f"(shared with the torus; x{_aniso_faceon:.3f} face-on, 1 at 30 deg)"),
    ("RAD (DPL jet): alpha1, alpha2, nu_t, nu_cut",
     ["radio_alpha_thin", "radio_alpha_thick", "radio_log_nu_t", "radio_log_nu_cut", "radio_loudness"],
     "alpha1 -> radio_alpha_thin, alpha2 -> radio_alpha_thick (same sign, both negative); nu_t, nu_cut as log10 [Hz]; "
     "radio_loudness = log10[L_nu(5 GHz) / L_nu(4400 A) of the disc]"),
    ("RAD (SPL jet): alpha", ["radio_alpha_agn"],
     "radio_alpha_agn = -alpha_AF (opposite sign: upstream -0.75 is 0.75 here)"),
    ("q_IR (star-formation radio)", ["radio_q_ir"],
     f"direct; the repackaged template embeds {Q_IR_PARITY:.3f}, tengri's default is {_Q_BELL}"),
]
_names = set(m_cap.spec.all_params)
print("§7  AGNfitter-rX parameter -> tengri parameter")
print("    | AGNfitter-rX | tengri | conversion |")
print("    |---|---|---|")
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
# Gaussian on upstream's GA; the cell converts tengri's formed mass to GA through the §1.3
# relation, so the prior acts on the same physical quantity in both codes.
# `prior_energy_balance` compares the dust-absorbed with the cold-dust re-emitted luminosity; at
# `dust_eta_balance`'s default (strict balance) they match by construction.
#
# **Caveat:** two upstream priors have no faithful tengri counterpart, by design. Upstream's
# `maximal_age` evaluates the observing-redshift terms outside its integrand, so it imposes no real
# age limit, and its energy-balance prior compares quantities in different normalizations, so its
# "absorbed" term is far from the physical absorbed power (both measured in section 6). tengri
# implements the physical versions, so agreement with upstream's values is not expected.

# %%
from tengri.agn.priors import AGNFITTER_PRIOR_DEFAULTS, agnfitter_priors

_z13 = 0.5
_dL13 = float(cosmo.luminosity_distance(_z13))
m13 = SEDModel.build(
    ssp_data=ssp, sfh=SFH_FIDUCIAL, met=MET_BC03_SOLAR,
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
pred13 = m13.predict({})
w13 = np.asarray(pred13.sed.components["wavelength"])
_o13 = np.argsort(w13)


def _flux_at(key, lam_rest):
    L = np.asarray(pred13.sed.components[key])
    return float(units.lnu_to_fnu(np.interp(lam_rest, w13[_o13], L[_o13]), _dL13, _z13))


_flux_1500 = _flux_at("sed_intrinsic", 1500.0)
_l_2kev = float(np.interp(6.199, w13[_o13], np.asarray(pred13.sed.components["sed_xray"])[_o13]))
_log_l2kev = float(np.log10(max(_l_2kev, 1e-300)))
# The ir_xrays prior compares against Stern (2015) applied to the model's own 6 um torus flux,
# so the demonstration is self-consistent.
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
_ga13 = A.ga_from_log_mstar_present(np.log10(float(pred13.sfh.stellar_mass) / _f_formed_per_present), _dL13, _z13)
total_all, breakdown_all = agnfitter_priors(
    pred13, redshift=_z13, dlum=_dL13, torus_key="sed_agn_torus", disc_key="sed_agn_disc",
    enable_energy_balance=True,
    enable_stellar_mass=True, ga=_ga13,  # GA holding tengri's formed mass (section 1.3 relation)
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
    f"(GA = {_ga13:.3f} from M*_formed = {float(pred13.sfh.stellar_mass):.3e} Msun at z = {_z13})"
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
    sfh=SFH_FIDUCIAL, met=MET_BC03_SOLAR,
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
print("§8.2  torus-reduction parity (peak-normalized, 1-100 um median|log10 ratio|):")
for _row in _reduction_rows:
    print(
        f"    {_row['label']:60s} median = {_row['median_dex']:.3f} dex  "
        f"(points above 10 um: tengri {_row['n_tengri']}, reference {_row['n_ref']})"
    )

# %%
plt.rcParams["figure.dpi"] = 100
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
_w_lo, _L_lo, _sub_lo = A.cat3d_union_template(_INCL0, _A0, _worst_low[0])
_w_tl, _L_tl = tengri_cat3d_union(_INCL0, _A0, _worst_low[0], _sub_lo)
_a_lo, _t_lo = norm_peak(_L_lo), norm_peak(_L_tl)
_lam_w = _worst_low[5]
_i_lo, _i_tl = int(np.argmin(np.abs(_w_lo - _lam_w))), int(np.argmin(np.abs(_w_tl - _lam_w)))
_node_ratio = np.interp(np.log10(_w_lo), np.log10(_w_tl), _t_lo) / _a_lo
_near = (_w_lo > 0.5 * _lam_w) & (_w_lo < 1.6 * _lam_w)
print(
    f"§8.3  worst low-wind case (f_wd = {_worst_low[0]:g}) at {_lam_w / 1e4:.2f} um: "
    f"reference L/L_peak = {float(np.interp(np.log10(_lam_w), np.log10(_w_lo), _a_lo)):.4f}; "
    f"node spacing there: reference {np.log10(_w_lo[_i_lo + 1] / _w_lo[_i_lo - 1]) / 2:.4f} dex, "
    f"tengri {np.log10(_w_tl[_i_tl + 1] / _w_tl[_i_tl - 1]) / 2:.4f} dex"
)
print(
    "§8.3  tengri / reference at the reference's own nodes "
    + ", ".join(f"{w / 1e4:.2f} um: {r:.4f}" for w, r in zip(_w_lo[_near], _node_ratio[_near]))
)
print(
    f"§8.3  band-integrated 1-5 um power, tengri / reference: "
    f"{band_power(_w_tl, _t_lo, 1e4, 5e4) / band_power(_w_lo, _a_lo, 1e4, 5e4):.4f}"
)
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
# a matched input: GA from the present mass that holds tengri's formed mass (§1.3), with no free scale; SB so its
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
_m_present_af = float(s_cap.sfh.stellar_mass) / _f_formed_per_present  # same formed mass, upstream bookkeeping
w_ga_c, L_ga_c = A.galaxy_lnu(TAU_GYR, AGE_GYR * 1e9, np.log10(_m_present_af), ebv=_EBV_GAL)
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
          label="AGNfitter-rX  ymodel: GA + SB + BB + TO + RAD")
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
# own warm Comptonization, not the $\alpha_{\rm ox}$ corona used here). Residuals in that band are
# construction differences. The torus row inherits the sampling of the model wavelength grid (§3),
# the model grid holds few points in the X-ray (a band power there is a sum over them, so the corona
# is compared at 2 keV), and the `ymodel` sum has no nebular or X-ray-binary term.

# %%
def _ratio_band(key, lo_aa, hi_aa):
    return band_power(w_te, {"GA": te_ga, "SB": te_sb, "BB": te_bb, "TO": te_to, "RAD": te_rad}[key], lo_aa, hi_aa) / \
        band_power(lam_grid, af_parts[key], lo_aa, hi_aa)


print("Capstone  component band-power ratios, tengri / AGNfitter-rX")
print("    | component | band                      | ratio  | amplitude set by           |")
for _k, _lab, _lo, _hi, _how in [
    ("GA", "0.3-1 um", 3e3, 1e4, "matched formed mass (none free)"),
    ("GA", "1-3 um", 1e4, 3e4, "matched formed mass (none free)"),
    ("SB", "8-1000 um", _IR_LO, _IR_HI, "L_TIR (anchor)"),
    ("SB", "8-30 um", 8e4, 3e5, "L_TIR (shape test)"),
    ("BB", "0.1-1 um", 1e3, 1e4, "L_nu(2500 A) (shape test)"),
    ("TO", "1-1000 um", 1e4, 1e7, "torus power (anchor)"),
    ("TO", "3-30 um", 3e4, 3e5, "torus power (shape test)"),
    ("RAD", "0.1-30 GHz", U.C_ANGSTROM_PER_S / 3e10, U.C_ANGSTROM_PER_S / 1e8, "5 GHz (shape test)"),
]:
    print(f"    | {_k:9s} | {_lab:25s} | {_ratio_band(_k, _lo, _hi):6.3f} | {_how:26s} |")
_n_xgrid = int(np.sum((w_te >= 1.24) & (w_te <= 24.8)))
_ratio_2kev = val_at(w_te, _comp["sed_xray"], 6.199) / val_at(w_bb_c, L_bb_c, 6.199)
print(f"    | {'BB':9s} | {'2 keV (monochromatic)':25s} | {_ratio_2kev:6.3f} | {'alpha_ox (upstream) vs corona':26s} |")
print(f"    model-grid points in 0.5-10 keV: {_n_xgrid} (a band power there is a sum over those points)")

# %%
# Where the 2 keV offset comes from: the anchor of the corona, then its anisotropy.
_l2500_disc_cap = val_at(w_te, s_cap.sed.components["sed_agn_disc"], 2500.0)
assert abs(_l2500_disc_cap / _l2500_disc - 1.0) < 1e-6, "the winning model's disc differs from the section 6 anchor build"
_wx_cap = np.asarray(_wave_x6)
_cor_cap = lambda l2500: np.asarray(
    xray_agn_corona_from_disc(_wave_x6, l2500, delta_alpha_ox=0.0, gamma=1.8, apply_anisotropy=False, log_nh=0.0)
)
_at2 = lambda arr: float(np.interp(6.199, _wx_cap, arr))
_r_bare = _at2(_cor_cap(_L2500_te)) / val_at(w_bb_c, L_bb_c, 6.199)
_r_anchor = _anchor_factor  # section 6 value, same AGN build
_r_aniso = float(xray_anisotropy(1.0, 1.0))
print("Capstone  2 keV corona, tengri / AGNfitter-rX, step by step")
print("    | quantity                                                                   |  value |")
print(f"    | L_nu(2500 A): disc continuum / disc + lines (tengri)                       | {_l2500_disc_cap / _L2500_te:6.3f} |")
print(f"    | bare corona (anisotropy and N_H off) at the same L_nu(2500 A) as upstream  | {_r_bare:6.3f} |")
print(f"    | x corona anchored on the disc continuum / on disc + lines                  | {_r_anchor:6.3f} |")
print(f"    | x face-on anisotropy factor (agn_cos_inc = 1)                              | {_r_aniso:6.3f} |")
print(f"    | = predicted ratio                                                          | {_r_bare * _r_anchor * _r_aniso:6.3f} |")
print(f"    | measured ratio, tengri sed_xray / AGNfitter-rX                             | {_ratio_2kev:6.3f} |")
_cor_nh = lambda nh: np.asarray(
    xray_agn_corona_from_disc(_wave_x6, _l2500_disc_cap, delta_alpha_ox=0.0, gamma=1.8, apply_anisotropy=False, log_nh=nh)
)
_soft = (_wx_cap >= 25.0) & (_wx_cap < 62.0)
print(
    f"Capstone  what xray_log_nh = 0 changes against the default 20: 2 keV x{_at2(_cor_nh(0.0)) / _at2(_cor_nh(20.0)):.4f}, "
    f"0.2-0.5 keV median x{np.median(_cor_nh(0.0)[_soft] / _cor_nh(20.0)[_soft]):.3f}"
)

# %% [markdown]
# **Result.** The translated model reproduces the AGNfitter-rX sum component by component. The host
# at the matched formed mass closes to 0.1% (GA 1.001 at 0.3-1 µm, 0.999 at 1-3 µm), the cold dust to
# 0.1% (SB 1.000 and 0.999), the torus to 0.1% (TO 1.000 and 1.000), the jet to 0.3% (RAD 1.003), and
# the disc shape at 0.1-1 µm to 2% (BB 1.016). Over 4e14-8e14 Hz the optical normalization ratio has a
# median of 1.024 (16-84%: 0.996-1.046), and from radio to UV the fractional residual has a median of
# +0.003 (-0.009 to +0.021). The corona is the one component that does not close, 0.915 at 2 keV,
# and the step-by-step table accounts for it: tengri anchors its corona on the disc continuum, which
# is 0.770 of the disc plus lines at 2500 Å from which the upstream amplitude is set (a factor
# 0.846), and the face-on anisotropy multiplies it by 1.072; the product, 0.915, equals the measured
# ratio. The resulting $\alpha_{\rm ox}$ is -1.342 for tengri against -1.327 upstream. The absorbing
# column is switched off in this model, as §4 advises; against the default it raises the 2 keV
# corona by 0.4% and the 0.2-0.5 keV median by 30% (printed).

# %%
_opt_band = (nu_grid > 4e14) & (nu_grid < 8e14) & np.isfinite(_resid)
_opt_ratio = 1.0 + _resid[_opt_band]
_p16, _p50, _p84 = np.percentile(_opt_ratio, [16, 50, 84])
print(
    f"Capstone optical (4e14-8e14 Hz, tengri/AGNfitter-rX) normalization ratio: "
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
# The per-block worst deviations are in the table printed above. The entries below say what matched
# and what did not, section by section; use them as the page to check before trusting anything else.
#
# - **§1 Host galaxy.** Stellar populations agree in shape at the matched node (0.0125 dex), and the
#   light per unit formed mass agrees to 3%. The surviving mass fractions do not: 0.610 for tengri
#   against 0.546 for the template, because tengri's BC03 grid takes its mass loss from the DSPS fit
#   to FSPS; convert a GA prior with the formed-to-present ratio of §1.3. Calzetti attenuation agrees
#   inside its calibrated range; upstream's own tails are not reproduced. Cold dust (S17, DH02_CE01)
#   and the host composite agree in template shape.
# - **§2 Accretion disc.** R06, SN12 and the grid-tabulated KD18 discs agree at nodes (at most
#   0.002 dex for KD18). Only THB21 carries the 0.7 µm bump; tengri reproduces its position with a
#   line power 1.34 times the template's. The Prevot disc screen is the same law one constant factor
#   (1.1020) apart; qsogen's own curve is a different law.
# - **§3 Torus.** The tabulated blocks reproduce their reference nodes to 1% at worst. The low-wind
#   half of the CAT3D union reaches 10% between sparse reference nodes and agrees at them. Every
#   block reproduces the reference band power from 3 to 1000 µm to 0.6%, SKIRTOR's 68 far-infrared
#   points included. The inclination-dependent torus power differs by design.
# - **§4 X-ray corona.** Power law, cutoff and 2 keV anchor agree to 0.9% with anisotropy and
#   absorption off; tengri's defaults add both.
# - **§5 Radio.** SPL and DPL agree to 1e-4; the star-formation radio follows the template to 0.09%
#   at its embedded $q_{\rm IR}$.
# - **§6 Differences.** The table lists each difference with its measured effect and what to do.
# - **§7 Translation.** Every AGNfitter-rX parameter has a tengri counterpart and a printed
#   conversion; the eight informative priors evaluate through `agnfitter_priors` and attach to a fit
#   through `extra_log_prior`.
# - **§9 Capstone.** One buildable, fittable model over $8 < \log\nu/{\rm Hz} < 20$ against upstream's
#   `ymodel` sum: every component closes to 2% except the corona at 2 keV (0.915), which the corona
#   anchor and the face-on anisotropy account for.
#
# **What to use.** Move an AGNfitter-rX fit to tengri with the §7 table, remove the tengri defaults
# that AGNfitter-rX lacks (`xray_log_nh` = 0) for a like-for-like comparison, and expect the §6
# differences.

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
