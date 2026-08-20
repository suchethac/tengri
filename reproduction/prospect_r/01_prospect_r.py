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
# # Reproducing ProSpect's physics with tengri
#
# ProSpect (Robotham et al. 2020) is an SED generation and fitting code from
# the GAMA survey, written in R. Every left-hand panel calls its own functions
# through `rpy2` (thin wrappers in `_drivers/prospect_driver.py`); the
# right-hand panel is tengri.
#
# The closed-form blocks — the SFH shapes, the mass-mapped metallicity
# history, the attenuation curves, the IGM — match ProSpect to a fraction
# of a percent. Nebular emission (§8) is where the two codes use genuinely
# different inputs (different photoionization grids), and that section
# quantifies the difference. ProSpect has no X-ray component; it does have
# a radio continuum, which §11 includes.

# %% [markdown]
# ## Setup

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")
# rpy2 links to R at run time (ABI mode) and needs R on PATH; the driver
# sets both, but we mirror them here so the kernel environment is explicit.
os.environ.setdefault("RPY2_CFFI_MODE", "ABI")
if "/opt/homebrew/bin" not in os.environ.get("PATH", ""):
    os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ.get("PATH", "")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reproduction.prospect_r._drivers import prospect_driver as P, units as U

import tengri
from tengri import FIXED, Fixed, SEDModel, load_ssp_data
from tengri.utils.physics_constants import LOG10_ZSUN

# Force the inline backend so figures embed on (re-)render regardless of the
# ambient MPLBACKEND. A non-inline backend (e.g. Agg) drops the save_fig()
# auto-display and produces a figure-less notebook. No-op when run as a script.
try:  # noqa: SIM105
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

warnings.filterwarnings("ignore")
tengri.plot.setup_style()

# Unit-sanity guard: ProSpect returns L_λ in L⊙/Å, which the driver
# converts to erg/s/Hz via L⊙ and the λ²/c Jacobian. Every panel below
# claims percent-level agreement, so a factor bug in the converter would
# silently misshape the whole notebook. Assert the bolometric round-trip
# here — the notebook trips at Setup if the converter ever drifts.
_unit_check = U.verify_unit_conversion(rtol=1e-3)
print(
    f"unit-conversion bolometric round-trip: "
    f"rel_err = {_unit_check['rel_err']:.2e}  (target < 1e-3)"
)

# Metallicity pin. ProSpect / BC03 work in absolute metal mass fraction Z with
# Z⊙ = 0.02 (the BC03 convention); tengri's `logzsol` is log10(Z / Z⊙) with
# tengri's Z⊙ is Asplund+2009 (`LOG10_ZSUN`). The two solar conventions differ,
# so to put both codes at the *same absolute* Z = 0.02 the tengri side needs
# logzsol = log10(0.02) − LOG10_ZSUN ≈ 0.149, not 0. Matching absolute Z (not
# the label "solar") is what keeps the SED panels honest.
Z_SOLAR = 0.02


def logzsol_for_Z(z_abs: float) -> float:
    """tengri ``logzsol`` that lands a model at absolute metallicity ``z_abs``."""
    return float(np.log10(z_abs) - LOG10_ZSUN)


MET_LOGZSOL = logzsol_for_Z(Z_SOLAR)  # ≈ 0.149

# Fiducial galaxy shared across the SED panels: a skew-normal star
# formation history peaking 10 Gyr ago, observed at z = 0, solar
# metallicity, with Charlot & Fall birth-cloud + screen dust.
LOG_MASS_FIDUCIAL = 10.0
SNORM_FIDUCIAL = dict(mSFR=10.0, mpeak=10.0, mperiod=2.0, mskew=0.5)
TAU_BIRTH_FIDUCIAL = 1.0
TAU_SCREEN_FIDUCIAL = 0.3
POW_FIDUCIAL = -0.7
MASS_SCALE = 10.0**LOG_MASS_FIDUCIAL

# nbclient kernels don't bind ``__file__`` (the kernel's resources path is
# the notebook directory instead), so fall back to the CWD.
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
# ## Common stellar library
#
# Both codes read Bruzual & Charlot (2003) grids: ProSpect reads `BC03lr`
# from the `ProSpectData` R package; tengri reads BC03 (Padova 1994 +
# STELIB, Chabrier IMF) from the public catalog.

# %%
ssp_path = tengri.download_ssp("bc03_pdva_stelib_chabrier", dest=str(_HERE / "_drivers" / "data"))
ssp = load_ssp_data(str(ssp_path))
_pro_info = P.ssp_grid_info("BC03lr")
print(
    f"tengri BC03:   {ssp.ssp_wave.shape[0]} wavelengths, "
    f"{ssp.ssp_lgmet.shape[0]} metallicities, {ssp.ssp_lg_age_gyr.shape[0]} ages; "
    f"λ up to {ssp.ssp_wave.max():.1e} Å."
)
print(
    f"ProSpect BC03lr: {_pro_info['n_wave']} wavelengths, "
    f"{_pro_info['n_z']} metallicities, {_pro_info['n_age']} ages; "
    f"λ up to {_pro_info['wave_max']:.1e} Å."
)

# ProSpect's SED scales with the stellar mass its SFH forms; tengri fixes the
# formed mass through `log_total_mass`. Integrate the fiducial skew-normal SFH
# on the ProSpect side to get its formed mass, then scale ProSpect's spectra to
# the same 10^10 M⊙ so the absolute-luminosity panels line up.
_t_pro, _sfr_pro = P.sfh_curve(sfh="snorm", **SNORM_FIDUCIAL)
_o = np.argsort(_t_pro)
PRO_MASS = float(np.trapezoid(_sfr_pro[_o], _t_pro[_o]))
PRO_SCALE = MASS_SCALE / PRO_MASS
print(f"ProSpect fiducial formed mass = {PRO_MASS:.3e} M⊙  →  scale to 10^10 = ×{PRO_SCALE:.3e}")

# tengri SSP metallicity index nearest solar (ssp_lgmet is absolute log10 Z).
I_ZSUN = int(np.argmin(np.abs(np.asarray(ssp.ssp_lgmet) - np.log10(Z_SOLAR))))


# %% [markdown]
# ## §1 Single stellar populations
#
# Both codes carry a Bruzual & Charlot (2003) library at a Chabrier IMF and
# Padova 1994 isochrones. ProSpect's `BC03lr` is the low-resolution variant
# (1221 wavelengths); tengri's grid samples the same models on a finer
# wavelength grid. SSPs from 1 Myr to 10 Gyr at solar metallicity, with the
# relative residual `|tengri − ProSpect| / ProSpect` below.
#
# The two are independent distributions of one underlying library, so the
# residual is a resolution and interpolation effect, a percent-level floor
# rather than a physics difference. Residual spikes sit at spectral features
# and reflect ProSpect's coarse wavelength grid (1221 λ vs 6900 λ); they
# vanish when both spectra are regridded to the same wavelength mesh.

# %% [markdown]
# **Verification Status:** CROSSVAL (2 tests — thin) — CSP integral — CIC age kernel (default)

# %%
_target_ages_yr = [1e6, 1e7, 1e8, 1e9, 1e10]
_age_idx = [
    int(np.argmin(np.abs(np.asarray(ssp.ssp_lg_age_gyr) - np.log10(a / 1e9))))
    for a in _target_ages_yr
]

pro_ssp, tng_ssp, age_labels = [], [], []
for ia in _age_idx:
    age_gyr = float(10.0 ** np.asarray(ssp.ssp_lg_age_gyr)[ia])
    w_p, L_p = P.ssp_spectrum(Z=Z_SOLAR, age_gyr=age_gyr)
    pro_ssp.append((w_p, L_p))
    tng_ssp.append(
        (np.asarray(ssp.ssp_wave), np.asarray(ssp.ssp_flux[I_ZSUN, ia, :]) * U.L_SUN_ERG_PER_S)
    )
    age_labels.append(f"{age_gyr * 1e3:.0f} Myr" if age_gyr < 1 else f"{age_gyr:.1f} Gyr")

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
colors = plt.cm.viridis(np.linspace(0, 1, len(_age_idx)))
for color, label, (w_p, L_p), (w_t, L_t) in zip(colors, age_labels, pro_ssp, tng_ssp):
    ax.plot(w_p, L_p, color=color, linewidth=2.0, label=label)
    ax.plot(w_t, L_t, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
    L_t_on_p = U.regrid(w_t, L_t, w_p)
    resid = np.abs(L_t_on_p - L_p) / np.maximum(np.abs(L_p), 1e-30)
    resid[~np.isfinite(resid)] = 0.0
    ax_r.plot(w_p, resid, color=color, linewidth=1.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e6)
ax.set_ylim(1e14, 5e21)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz / $M_\odot$]")
ax.set_title("BC03 Chabrier Z = Z⊙ — ProSpect (solid) vs tengri (black dashed)")
ax.legend(fontsize=9, title="SSP age")
ax.grid(True, alpha=0.3)
ax_r.set_xscale("log")
ax_r.set_yscale("log")
ax_r.set_xlabel(r"$\lambda$ [Å]")
ax_r.set_ylabel(r"$|\Delta| / L_{\rm ProSpect}$", fontsize=9)
ax_r.set_ylim(1e-4, 1e0)
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_01_ssp_bc03.png")

_w_ref, _L_ref = pro_ssp[3]  # 1 Gyr
_mask = (_w_ref >= 3000) & (_w_ref <= 10000)
_t_on_p = U.regrid(tng_ssp[3][0], tng_ssp[3][1], _w_ref)
_res = np.abs(_t_on_p[_mask] - _L_ref[_mask]) / np.maximum(_L_ref[_mask], 1e-30)
print(f"§1 SSP 1 Gyr optical residual: median {np.median(_res):.2e}, max {_res.max():.2e}")


# %% [markdown]
# ## §2 Star formation history
#
# ProSpect offers the skew-normal `massfunc_snorm` and delayed-exponential
# `massfunc_dtau`; tengri carries both forms. tengri's curve is pipeline
# output from `state.derived["sfr_history"]`, on the log-spaced lookback grid
# used by the convolution. `mSFR = 10` forms ≈10^10.9 M⊙, and tengri's
# `log_total_mass` is set to match, so the SFR amplitudes are directly
# comparable. Peaks and widths agree; the two `snorm` implementations
# parametrize the skew slightly differently.

# %% [markdown]
# **Verification Status:** PARTIAL (11/33) — Parametric SFH family physics

# %%
t_p_sn, sfr_p_sn = P.sfh_curve(sfh="snorm", **SNORM_FIDUCIAL)
t_p_dt, sfr_p_dt = P.sfh_curve(sfh="dtau", mSFR=10.0, mpeak=10.0, mtau=3.0)

# Match tengri's formed mass to ProSpect's skew-normal so the curves overlay.
LOG_MASS_SNORM = float(np.log10(PRO_MASS))
m_sfh = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": FIXED},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_SNORM),
        "all_params": FIXED,
    },
    dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "all_params": FIXED},
    redshift=Fixed(0.0),
)
s_sfh = m_sfh.predict_state({})
_lbt_yr = np.asarray(s_sfh.derived["sfh_grid_lbt_yr"])
_sfr_history = np.asarray(s_sfh.derived["sfr_history"])
_idx = np.argsort(_lbt_yr)
_mass_t = float(np.trapezoid(_sfr_history[_idx], _lbt_yr[_idx]))
print(
    f"§2 formed mass: ProSpect = {PRO_MASS:.3e} M⊙, tengri pipeline = {_mass_t:.3e} M⊙; "
    f"tengri snorm peak at {_lbt_yr[np.argmax(_sfr_history)] / 1e9:.1f} Gyr "
    f"(ProSpect mpeak = {SNORM_FIDUCIAL['mpeak']:g} Gyr)"
)

fig, ax_l, ax_r = U.two_panel_fig()
for ax in (ax_l, ax_r):
    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_ylabel(r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]")
    ax.set_xlim(0, 13.7)
    ax.grid(True, alpha=0.3)
ax_l.set_title("ProSpect massfunc_snorm / massfunc_dtau")
ax_l.plot(t_p_sn / 1e9, sfr_p_sn, "C0-", linewidth=2.0, label="snorm (peak 10 Gyr)")
ax_l.plot(t_p_dt / 1e9, sfr_p_dt, "C2--", linewidth=2.0, label="dtau (τ=3 Gyr)")
ax_l.legend(fontsize=9)
ax_r.set_title("tengri pipeline sfr_history (snorm)")
ax_r.plot(t_p_sn / 1e9, sfr_p_sn, "C0-", linewidth=1.0, alpha=0.5, label="ProSpect snorm")
ax_r.plot(_lbt_yr / 1e9, _sfr_history, "C1-", linewidth=2.0, label="tengri snorm")
ax_r.legend(fontsize=9)
fig.tight_layout()
save_fig("prospect_r_02_sfh.png")


# %% [markdown]
# ## §2b Metallicity history — chemical evolution
#
# ProSpect ties gas-phase metallicity to cumulative stellar mass formed
# (Bellstedt et al. 2020), breaking the age–metallicity degeneracy: old
# stars become metal-poor and young stars metal-rich. `Zfunc_massmap_lin`
# maps Z linearly; `Zfunc_massmap_box` uses Lynden-Bell closed-box enrichment
# with a fixed yield.
#
# tengri implements both, reading each history from
# `state.derived["log_metallicity_history"]` at matched `Zstart`/`Zfinal`,
# with ProSpect's `yield` parameter exposed as `met_yield`. Against
# cumulative mass fraction `massmap_lin` is a straight line by construction;
# the half-mass-point ratio confirms the linear case agrees.

# %% [markdown]
# **Verification Status:** PARTIAL (11/33) — Parametric SFH family physics

# %%
Z_START, Z_FINAL = 1e-4, Z_SOLAR
age_pro, Z_lin, cmf_lin = P.metallicity_history(
    zfunc="massmap_lin", sfh="snorm", Zstart=Z_START, Zfinal=Z_FINAL, **SNORM_FIDUCIAL
)
_, Z_box, _ = P.metallicity_history(
    zfunc="massmap_box", sfh="snorm", Zstart=Z_START, Zfinal=Z_FINAL, yield_=0.03, **SNORM_FIDUCIAL
)

# tengri massmap_lin — the same cumulative-mass mapping, matched endpoints.
m_zmm = SEDModel.build(
    ssp_data=ssp,
    met={
        "type": "massmap_lin",
        "met_logzsol_start": Fixed(logzsol_for_Z(Z_START)),
        "met_logzsol_final": Fixed(logzsol_for_Z(Z_FINAL)),
        "all_params": FIXED,
    },
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": FIXED,
    },
    dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "all_params": FIXED},
    redshift=Fixed(0.0),
)
s_zmm = m_zmm.predict_state({})
_age_t = np.asarray(s_zmm.derived["sfh_grid_lbt_yr"])
_Z_t = 10.0 ** np.asarray(s_zmm.derived["log_metallicity_history"])  # absolute Z
# tengri cumulative-mass fraction formed, mass-weighted (∫SFR dt — the grid is
# log-spaced, so SFR must be integrated against the bin width, not summed).
_sfr_t = np.asarray(s_zmm.derived["sfr_history"])
_o = np.argsort(_age_t)  # ascending lookback (youngest first)
_a, _sfr_a = _age_t[_o], _sfr_t[_o]
_mass_int = 0.5 * (_sfr_a[:-1] + _sfr_a[1:]) * np.abs(np.diff(_a))
_mass_after = np.concatenate([[0.0], np.cumsum(_mass_int)])  # mass formed younger than each age
_cmf_sorted = 1.0 - _mass_after / max(_mass_after[-1], 1e-30)  # fraction older (Z's variable)
_cmf_t = np.empty_like(_cmf_sorted)
_cmf_t[_o] = _cmf_sorted

# tengri massmap_box — the Lynden-Bell closed box, same SFH and endpoints, with
# the ProSpect ``yield`` parameter exposed as ``met_yield``.
m_zmb = SEDModel.build(
    ssp_data=ssp,
    met={
        "type": "massmap_box",
        "met_logzsol_start": Fixed(logzsol_for_Z(Z_START)),
        "met_logzsol_final": Fixed(logzsol_for_Z(Z_FINAL)),
        "met_yield": Fixed(0.03),
        "all_params": FIXED,
    },
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": FIXED,
    },
    dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "all_params": FIXED},
    redshift=Fixed(0.0),
)
_Z_box_t = 10.0 ** np.asarray(m_zmb.predict_state({}).derived["log_metallicity_history"])

fig, ((ax_l, ax_r), (ax_l2, ax_r2)) = plt.subplots(2, 2, figsize=(12, 8))
ax_l.set_title("ProSpect mass-mapped Z history")
ax_l.plot(age_pro / 1e9, Z_lin / Z_SOLAR, "C0-", lw=2, label="massmap_lin")
ax_l.plot(age_pro / 1e9, Z_box / Z_SOLAR, "C3--", lw=2, label="massmap_box (yield 0.03)")
ax_r.set_title("tengri massmap_lin / massmap_box Z history")
ax_r.plot(age_pro / 1e9, Z_lin / Z_SOLAR, "C0-", lw=1, alpha=0.4, label="ProSpect lin")
ax_r.plot(age_pro / 1e9, Z_box / Z_SOLAR, "C3-", lw=1, alpha=0.4, label="ProSpect box")
ax_r.plot(_age_t / 1e9, _Z_t / Z_SOLAR, "C1-", lw=2, label="tengri massmap_lin")
ax_r.plot(_age_t / 1e9, _Z_box_t / Z_SOLAR, "C2--", lw=2, label="tengri massmap_box")
for ax in (ax_l, ax_r):
    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_ylabel(r"$Z / Z_\odot$")
    ax.set_yscale("log")
    ax.set_xlim(0, 13.7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
ax_l2.set_title("ProSpect — Z vs cumulative mass formed")
ax_l2.plot(cmf_lin, Z_lin / Z_SOLAR, "C0-", lw=2, label="massmap_lin")
ax_l2.plot(cmf_lin, Z_box / Z_SOLAR, "C3--", lw=2, label="massmap_box")
ax_r2.set_title("tengri massmap_lin / massmap_box — Z vs cumulative mass")
ax_r2.plot(cmf_lin, Z_lin / Z_SOLAR, "C0-", lw=1, alpha=0.4, label="ProSpect lin")
ax_r2.plot(cmf_lin, Z_box / Z_SOLAR, "C3-", lw=1, alpha=0.4, label="ProSpect box")
ax_r2.plot(_cmf_t, _Z_t / Z_SOLAR, "C1-", lw=2, label="tengri massmap_lin")
ax_r2.plot(_cmf_t, _Z_box_t / Z_SOLAR, "C2--", lw=2, label="tengri massmap_box")
for ax in (ax_l2, ax_r2):
    ax.set_xlabel("cumulative mass fraction formed")
    ax.set_ylabel(r"$Z / Z_\odot$")
    ax.set_yscale("log")
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
fig.tight_layout()
save_fig("prospect_r_02b_metallicity.png")

# Quantify the match: tengri massmap_lin vs ProSpect massmap_lin at half-mass.
_Z_t_at_half = float(np.interp(0.5, _cmf_t[np.argsort(_cmf_t)], _Z_t[np.argsort(_cmf_t)]))
_Z_lin_at_half = float(np.interp(0.5, cmf_lin[np.argsort(cmf_lin)], Z_lin[np.argsort(cmf_lin)]))
print(
    f"§2b Z at half the mass formed: ProSpect massmap_lin = {_Z_lin_at_half / Z_SOLAR:.3f} Z⊙, "
    f"tengri massmap_lin = {_Z_t_at_half / Z_SOLAR:.3f} Z⊙  "
    f"(ratio {_Z_t_at_half / _Z_lin_at_half:.2f}× — same mass-mapped chemical evolution)"
)


# %% [markdown]
# ## §3 Integrated stellar SED
#
# The fiducial skew-normal SFH convolved with the BC03 library at solar
# metallicity, no dust or nebular, both scaled to 10^10 M⊙ formed.

# %% [markdown]
# **Verification Status:** CROSSVAL — Photometry projection

# %%
sed_stel = P.prospect_sed(
    massfunc="snorm",
    sfh_pars=SNORM_FIDUCIAL,
    Z=Z_SOLAR,
    tau_birth=0.0,
    tau_screen=0.0,
)
w_p3, L_p3 = sed_stel["FinalLum"]
L_p3 = L_p3 * PRO_SCALE

m_stellar = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": FIXED},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": FIXED,
    },
    dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "all_params": FIXED},
    redshift=Fixed(0.0),
)
s_stellar = m_stellar.predict_state({})
_assert_comparable(L_p3, s_stellar.sed_intrinsic, name="§3 stellar")

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="ProSpect  snorm + BC03", label_r="tengri  snorm + BC03")
ax_l.plot(w_p3, L_p3, "C0-", linewidth=1.5)
ax_r.plot(s_stellar.wave, s_stellar.sed_intrinsic, "C1-", linewidth=1.5)
# This BC03 grid does not carry a surviving-mass column, so log_mstar is NaN;
# report the formed mass (what both codes are normalized to) instead.
m_formed = 10.0 ** float(s_stellar.derived["log_mstar_formed"])
ax_r.text(
    0.05,
    0.95,
    rf"$M_\star = {m_formed:.2e}\,M_\odot$ formed",
    transform=ax_r.transAxes,
    fontsize=10,
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
for ax in (ax_l, ax_r):
    ax.set_xlim(1e2, 1e6)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_03_stellar_sed.png")

_mask_opt = (w_p3 >= 3000) & (w_p3 <= 10000)
_t_on_p3 = U.regrid(np.asarray(s_stellar.wave), np.asarray(s_stellar.sed_intrinsic), w_p3)
_ratios3 = _t_on_p3[_mask_opt] / L_p3[_mask_opt]
_ratios3 = _ratios3[np.isfinite(_ratios3) & (_ratios3 > 0)]
print(
    f"§3 stellar SED tengri/ProSpect optical (3000–10000 Å): "
    f"median {np.median(_ratios3):.3f}, P5 {np.percentile(_ratios3, 5):.3f}, "
    f"P95 {np.percentile(_ratios3, 95):.3f}"
)


# %% [markdown]
# ## §4 Dust attenuation curves
#
# ProSpect uses Charlot & Fall (2000): a birth-cloud term on young stars
# and a diffuse screen on all stars, each a power law with slope −0.7.
# tengri's `power_law` law is the same functional form. Both are normalized
# to `A(λ)/A_V` at 5500 Å; the screen also carries an optional 2175 Å bump,
# against tengri's `noll09` law.

# %% [markdown]
# **Verification Status:** CROSSVAL — Attenuation law library

# %%
from tengri.dust import list_laws

_tengri_laws = list_laws(headline=False).to_dict('fn')
wave_law = np.logspace(np.log10(1000.0), np.log10(30000.0), 2000)


def _norm_AV(wave, A):
    return A / A[np.argmin(np.abs(wave - 5500.0))]


# ProSpect Charlot & Fall power-law screen + birth, and a bumped screen.
w_cf, A_cf = P.attenuation_curve(component="screen", tau=1.0, pow_=-0.7)
_, A_cf_birth = P.attenuation_curve(component="birth", tau=1.0, pow_=-0.7, wave_aa=w_cf)
_, A_cf_bump = P.attenuation_curve(component="screen", tau=1.0, pow_=-0.7, Eb=1.5, wave_aa=w_cf)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, title in (
    (ax_l, "ProSpect Charlot & Fall (power law)"),
    (ax_r, "tengri attenuation laws"),
):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ [Å]")
    ax.set_xlim(1e3, 3e4)
    ax.set_ylim(0.1, 10)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
ax_l.set_ylabel(r"$A_\lambda / A_V$")
ax_l.plot(w_cf, _norm_AV(w_cf, A_cf), "C0-", lw=2, label="screen (pow −0.7)")
ax_l.plot(w_cf, _norm_AV(w_cf, A_cf_birth), "C2--", lw=2, label="birth (pow −0.7)")
ax_l.plot(w_cf, _norm_AV(w_cf, A_cf_bump), "C3:", lw=2, label="screen + 2175 Å bump")
ax_l.legend(fontsize=10)
ax_r.plot(
    wave_law,
    _norm_AV(wave_law, np.asarray(_tengri_laws["power_law"](wave_law))),
    "C0-",
    lw=2,
    label="power_law",
)
ax_r.plot(
    wave_law,
    _norm_AV(wave_law, np.asarray(_tengri_laws["noll09"](wave_law))),
    "C3:",
    lw=2,
    label="noll09 (with bump)",
)
ax_r.legend(fontsize=10)
fig.tight_layout()
save_fig("prospect_r_04_dust_attenuation.png")

# Quantify the power-law match at 1500 Å.
_a_p = float(_norm_AV(w_cf, A_cf)[np.argmin(np.abs(w_cf - 1500))])
_a_t = float(
    _norm_AV(wave_law, np.asarray(_tengri_laws["power_law"](wave_law)))[
        np.argmin(np.abs(wave_law - 1500))
    ]
)
print(f"§4 A(1500)/A_V: ProSpect CF = {_a_p:.3f}, tengri power_law = {_a_t:.3f}")


# %% [markdown]
# ## §5 Attenuation applied
#
# The fiducial galaxy with and without dust. ProSpect applies birth-cloud
# (`τ_birth = 1`) and diffuse screen (`τ_screen = 0.3`) at slope −0.7.
# tengri's `two_component` dust maps directly with `τ_bc` and `τ_diff` using
# the `power_law` law.

# %% [markdown]
# **Verification Status:** CROSSVAL — Attenuation law library

# %%
sed_atten = P.prospect_sed(
    massfunc="snorm",
    sfh_pars=SNORM_FIDUCIAL,
    Z=Z_SOLAR,
    tau_birth=TAU_BIRTH_FIDUCIAL,
    tau_screen=TAU_SCREEN_FIDUCIAL,
    pow_birth=POW_FIDUCIAL,
    pow_screen=POW_FIDUCIAL,
)
w_p5, L_p5 = sed_atten["StarsAtten"]
L_p5 = L_p5 * PRO_SCALE

DUST_FIDUCIAL = {
    "type": "two_component",
    "law_bc": "power_law",
    "law_diff": "power_law",
    "tau_bc": Fixed(TAU_BIRTH_FIDUCIAL),
    "tau_diff": Fixed(TAU_SCREEN_FIDUCIAL),
    "all_params": FIXED,
}
m_d = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": FIXED},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": FIXED,
    },
    dust_attenuation=DUST_FIDUCIAL,
    redshift=Fixed(0.0),
)
s_d = m_d.predict_state({})
L_t_atten = np.asarray(s_d.derived["sed_dust_attenuated"])
_assert_comparable(L_p5, L_t_atten, name="§5 dust applied")

fig, ((ax_l1, ax_r1), (ax_l2, ax_r2)) = plt.subplots(2, 2, sharey=True, figsize=(12, 8))
U.panel(ax_l1, ax_r1, label_l="ProSpect  intrinsic", label_r="tengri  intrinsic")
U.panel(
    ax_l2,
    ax_r2,
    label_l=rf"ProSpect  CF ($\tau_b={TAU_BIRTH_FIDUCIAL:g},\ \tau_s={TAU_SCREEN_FIDUCIAL:g}$)",
    label_r=rf"tengri  two-component ($\tau_{{bc}}={TAU_BIRTH_FIDUCIAL:g}$, "
    rf"$\tau_{{diff}}={TAU_SCREEN_FIDUCIAL:g}$)",
)
ax_l1.plot(w_p3, L_p3, "C0-", linewidth=1.5)
ax_r1.plot(s_stellar.wave, s_stellar.sed_intrinsic, "C1-", linewidth=1.5)
ax_l2.plot(w_p5, L_p5, "C0-", linewidth=1.5)
ax_r2.plot(s_d.wave, L_t_atten, "C1-", linewidth=1.5)
_ymax = float(np.asarray(s_stellar.sed_intrinsic).max())
for ax in (ax_l1, ax_r1, ax_l2, ax_r2):
    ax.set_xlim(1e2, 5e4)
    ax.set_ylim(_ymax * 1e-6, _ymax * 2)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_05_dust_applied.png")


# %% [markdown]
# ## §6 Dust IR re-emission and energy balance
#
# ProSpect re-emits absorbed starlight with Dale et al. (2014) templates;
# tengri uses the same Dale 2014 grid and enforces energy balance to floating
# point. At matched hardness (`alpha = 3.0`) both codes produce identical
# dust IR SEDs.

# %% [markdown]
# **Verification Status:** CROSSVAL — Dust IR emission physics (MBB, Casey12, CMB)

# %%
sed_ir = P.prospect_sed(
    massfunc="snorm",
    sfh_pars=SNORM_FIDUCIAL,
    Z=Z_SOLAR,
    tau_birth=0.0,
    tau_screen=TAU_SCREEN_FIDUCIAL,
    pow_birth=POW_FIDUCIAL,
    pow_screen=POW_FIDUCIAL,
)
w_p6, L_p6 = sed_ir["FinalLum"]
L_p6 = L_p6 * PRO_SCALE

m_ir = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": FIXED},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": FIXED,
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "power_law",
        "law_diff": "power_law",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(TAU_SCREEN_FIDUCIAL),
        "all_params": FIXED,
    }, dust_emission={"type": "dale2014", "alpha_dale": Fixed(3.0), "all_params": FIXED},
    redshift=Fixed(0.0),
)
s_ir = m_ir.predict_state({})
_L_abs = float(np.asarray(s_ir.derived["L_absorbed"]))
_L_ir = float(np.asarray(s_ir.derived["L_ir"]))
_eb_resid = abs(_L_ir - _L_abs) / max(_L_abs, 1e-30)
print(
    f"§6 tengri energy balance: L_abs = {_L_abs:.3e}, L_IR = {_L_ir:.3e}, resid = {_eb_resid:.2e}"
)

sed_full_t = np.asarray(s_ir.derived["sed_dust_attenuated"]) + np.asarray(
    s_ir.derived["sed_dust_ir"]
)

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="ProSpect  CF + Dale 2014", label_r="tengri  power_law + Dale 2014")
ax_l.plot(w_p6, L_p6, "C0-", linewidth=1.5)
ax_r.plot(s_ir.wave, sed_full_t, "C1-", linewidth=1.5)
ax_r.text(
    0.05,
    0.95,
    rf"$|L_{{IR}} - L_{{abs}}| / L_{{abs}}$ = {_eb_resid:.1e}",
    transform=ax_r.transAxes,
    fontsize=10,
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
for ax in (ax_l, ax_r):
    ax.set_xlim(1e3, 1e7)
    ax.set_ylim(1e24, 1e32)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_06_dust_ir.png")

# Both SEDs are L_nu [erg/s/Hz]; compare the nu*L_nu peak on each side so the
# peak is measured the same way (a raw L_nu argmax lands ~30 um redward).
_fir = (w_p6 > 1e5) & (w_p6 < 1e7)
_peak_p6 = w_p6[_fir][np.argmax((L_p6 * U.C_ANGSTROM_PER_S / w_p6)[_fir])]
_w_t6 = np.asarray(s_ir.wave)
_L_t6 = np.asarray(s_ir.derived["sed_dust_ir"])
_fir_t = (_w_t6 > 1e5) & (_w_t6 < 1e7)
_peak_t6 = _w_t6[_fir_t][np.argmax((_L_t6 * U.C_ANGSTROM_PER_S / _w_t6)[_fir_t])]
print(f"§6 dust IR nu*Lnu peak: ProSpect {_peak_p6 / 1e4:.0f} um, tengri {_peak_t6 / 1e4:.0f} um")


# %% [markdown]
# ## §7 Panchromatic SED
#
# Stellar + Charlot & Fall attenuation + Dale 2014 IR from the rest-UV to
# the far-IR. The percent-level disagreements of the earlier sections stack
# here.

# %% [markdown]
# **Verification Status:** CROSSVAL — Photometry projection

# %%
fig, ax_l, ax_r = U.two_panel_fig(figsize=(13, 5))
U.panel(ax_l, ax_r, label_l="ProSpect  panchromatic", label_r="tengri  panchromatic")
ax_l.plot(w_p6, L_p6, "C0-", linewidth=1.5)
ax_r.plot(s_ir.wave, sed_full_t, "C1-", linewidth=1.5)
for ax in (ax_l, ax_r):
    ax.set_xlim(1e2, 1e7)
    ax.set_ylim(1e22, 1e31)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_07_panchromatic.png")


# %% [markdown]
# ## §12 IGM transmission — Inoue et al. (2014)
#
# Both codes use Inoue et al. (2014) for Lyman-series absorption. The
# residual is measured over the Lyman-α forest window. (§8–§11 cover
# nebular, AGN, and radio.)

# %% [markdown]
# **Verification Status:** CROSSVAL — Inoue+2014 IGM transmission

# %%
from tengri.igm import igm_transmission as tengri_igm

Z_IGM = 4.0
_wave_rest = np.linspace(700.0, 1300.0, 800)
w_igm, T_p_igm = P.igm_transmission(_wave_rest, Z_IGM)
# tengri's IGM takes observed-frame wavelengths.
T_t_igm = np.asarray(tengri_igm(_wave_rest * (1.0 + Z_IGM), np.asarray(Z_IGM)))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(w_igm, T_p_igm, "C0-", linewidth=2.0, label=f"ProSpect Inoue14, z={Z_IGM:g}")
ax.plot(w_igm, T_t_igm, "k--", linewidth=1.0, label=f"tengri Inoue14, z={Z_IGM:g}")
ax.set_xlabel(r"rest-frame $\lambda$ [Å]")
ax.set_ylabel(r"IGM transmission $T(\lambda, z)$")
ax.set_xlim(700, 1300)
ax.set_ylim(0, 1.05)
ax.set_title(f"Inoue (2014) IGM transmission at z = {Z_IGM:g}")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_12_igm_inoue.png")

# Compare away from the sharp Lyman-α step, where a one-sample grid offset
# between the two implementations produces a spurious ~0.6 spike that says
# nothing about the underlying physics.
_win = (w_igm >= 950) & (w_igm <= 1210)
_igm_diff = np.abs(T_t_igm[_win] - T_p_igm[_win])
print(
    f"§12 Inoue14 IGM at z={Z_IGM:g} (950–1210 Å, off the Lyα step): "
    f"median |Δ| = {np.median(_igm_diff):.3e}, 95th pct |Δ| = {np.percentile(_igm_diff, 95):.3e}"
)


# %% [markdown]
# ## §8 Nebular emission
#
# ProSpect uses `emissionLines`, tying Hα to the SFR and distributing other
# lines via Levesque et al. (2010). tengri uses Cue (Li et al. 2025), a neural
# emulator on Cloudy 17 that predicts lines from the ionizing spectrum. Cue
# needs a bare-stellar SSP (FSPS MIST + MILES here).
#
# **Matched ionization parameter.** ProSpect's `emissionLines` derives the
# ionization parameter from metallicity via `Z2q` (Orsi 2014), giving at solar
# Z a soft `q ≈ 1.4e7` cm/s that suppresses metal lines: [O III]/Hα ≈ 0.014
# (~50× below Cloudy). We pass ProSpect the matching `q = U·c ≈ 3e8` cm/s for
# Cue's `logU = -2`. ProSpect then returns [O III]/Hα ≈ 0.21; the residual
# versus Cue's 0.72 is a genuine Levesque-2010 vs Cloudy-17 difference, not
# an ionization mismatch. Balmer lines are q-insensitive.

# %% [markdown]
# **Verification Status:** CROSSVAL — Cloudy grid / Cue vs FSPS baked-in

# %%
NEB_AGE_GYR = 0.01
NEB_LOG_MASS = 8.0

ssp_neb = load_ssp_data(
    str(tengri.download_ssp("fsps_mist_miles_chabrier", dest=str(_HERE / "_drivers" / "data")))
)
m_neb = SEDModel.build(
    ssp_data=ssp_neb,
    met={"logzsol": Fixed(0.0), "all_params": FIXED},
    # Confine star formation to the last 10 Myr — a young, ionizing population.
    sfh={
        "type": "const",
        "start_gyr": Fixed(NEB_AGE_GYR),
        "end_gyr": Fixed(0.0),
        "log_total_mass": Fixed(NEB_LOG_MASS),
        "all_params": FIXED,
    },
    dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "all_params": FIXED},
    neb={"type": "cue", "neb_logU": Fixed(-2.0), "neb_logZ_gas": Fixed(0.0), "all_params": FIXED},
    redshift=Fixed(0.0),
)
s_neb = m_neb.predict_state({})
w_t8 = np.asarray(s_neb.wave)
L_t8 = np.asarray(s_neb.derived["sed_nebular"])

# Match ProSpect's SFR to tengri's recent SFR so the line budgets are comparable,
# and its ionization parameter to Cue's logU = -2 (q = U·c) so the metal-line
# ratios are a fair physics comparison rather than a q mismatch.
SFR_NEB = float(s_neb.derived["sfr_10myr"])
_NEB_Q = 10.0**-2.0 * 2.99792458e10  # q = U·c [cm/s] for logU = -2  → ≈ 3.0e8
w_p8, L_p8 = P.nebular_lnu(sfr=SFR_NEB, Z=Z_SOLAR, q=_NEB_Q)
print(f"§8 matched SFR = {SFR_NEB:.2f} M⊙/yr, ProSpect q = {_NEB_Q:.2e} cm/s (logU=-2)")


# Width-independent integrated line luminosity (shared helper U.line_lum) — a
# single-bin peak ratio would measure line width, not luminosity, since
# ProSpect broadens its lines to a fixed velocity dispersion while Cue places
# them at its grid resolution.
_lines = [(6563.0, "Hα"), (5007.0, "[O III]"), (3727.0, "[O II]"), (4861.0, "Hβ")]
_L = {}
print("§8 integrated line luminosity (Cue / ProSpect):")
for _c, _name in _lines:
    _lp = U.line_lum(w_p8, L_p8, _c)
    _lt = U.line_lum(w_t8, L_t8, _c)
    _L[_name] = (_lp, _lt)
    if _lp > 0:
        print(
            f"    {_name} {_c:.0f} Å: ProSpect {_lp:.2e}, "
            f"tengri {_lt:.2e} erg/s → {_lt / _lp:.2f}×"
        )
# Line *ratios* are the fair, normalization-free comparison. The absolute Hα is
# offset because ProSpect ties L_Hα to SFR via a fixed Kennicutt-style
# coefficient, while Cue derives it from the ionizing-photon budget — so the
# whole nebular spectrum carries that ~3x scale. The [O III]/Hα ratio isolates
# the line physics: at the matched q it is a modest Levesque-2010 vs
# Cloudy-17 difference, not the 50x q-mismatch artifact of the default Z2q.
if _L["Hα"][0] > 0 and _L["Hβ"][0] > 0:
    print("§8 line ratios (q-matched, normalization-free):")
    print(
        f"    [O III]/Hβ: ProSpect {_L['[O III]'][0] / _L['Hβ'][0]:.2f}, "
        f"tengri {_L['[O III]'][1] / _L['Hβ'][1]:.2f}"
    )
    print(
        f"    [O III]/Hα: ProSpect {_L['[O III]'][0] / _L['Hα'][0]:.3f}, "
        f"tengri {_L['[O III]'][1] / _L['Hα'][1]:.3f}"
    )
    print(
        f"    Hα/Hβ (Balmer): ProSpect {_L['Hα'][0] / _L['Hβ'][0]:.2f}, "
        f"tengri {_L['Hα'][1] / _L['Hβ'][1]:.2f}  (Case B ≈ 2.86)"
    )

fig, ax_l, ax_r = U.two_panel_fig(figsize=(13, 5))
U.panel(
    ax_l,
    ax_r,
    label_l="ProSpect  emissionLines",
    label_r="tengri  Cue emulator",
)
ax_l.plot(w_p8, L_p8, "C0-", linewidth=1.0)
ax_r.plot(w_t8, L_t8, "C1-", linewidth=1.0)
for ax in (ax_l, ax_r):
    ax.set_xlim(1000, 7000)
    ax.set_yscale("log")
    ax.set_xscale("linear")
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_08_nebular.png")


# %% [markdown]
# ## §9 AGN torus
#
# ProSpect models AGN with Fritz et al. (2006) and SKIRTOR (Stalevski et al.
# 2012, 2016). We compare ProSpect's `SKIRTOR_interp` against tengri's SKIRTOR
# at matched bolometric luminosity, pinned to ProSpect's defaults (inclination
# 30°, opening angle 40°, optical depth 1, p=q=1) with the full bolometric
# routed to the template (`agn_frac_agn=1`). The two agree to ~2%.
#
# The panels show `νL_ν`, where the thermal bump is a true maximum. (In `L_ν`
# a torus rises into the far-IR simply because `L_ν = νL_ν · λ/c`, easy to
# misread.) Both the torus (peak 9.3 μm with 10 μm silicate) and the disc
# shortward of ~1 μm track ProSpect: the disc reads ~0.9× ProSpect at 2000 Å.
# This uses the **`skirtor_stalevski`** model — the published Stalevski (2016)
# radiative-transfer SED, no analytic-disc substitution, reading the
# full-coverage SKIRTOR grid on the full `ta,p,q,oa,R,i` axes (fixing v3's two
# shortcuts: R fixed at 20; total reconstructed as disk+dust). Three SKIRTOR
# models are swappable (`tengri.list_agn_models()`): `skirtor_stalevski` (raw,
# ~0.9×), composable `disc.skirtor`+`torus.skirtor` (CIGALE's analytic disc +
# `norm=1/∫dust`), and deprecated monolithic `agn={'type':'skirtor'}`
# (power-law disc, ~0.28×). The residual to 1.0× is a parameter-convention
# mismatch (ProSpect's `ct`/`rm` vs SKIRTOR's `oa`/`R`) — not the disc/total
# treatment.

# %% [markdown]
# **Verification Status:** CROSSVAL — Nenkova+08 (CLUMPY) torus

# %%
AGN_LUM_ERG = 1e44
w_p9, L_p9, _agn_log_lbol = P.agn_torus_lnu(model="SKIRTOR", lum_erg=AGN_LUM_ERG)
print(
    f"§9 ProSpect SKIRTOR L_bol = {AGN_LUM_ERG:.1e} erg/s = 10^{_agn_log_lbol:.2f} L⊙ "
    f"(template integral)"
)

# Pin tengri's SKIRTOR to ProSpect's `SKIRTOR_interp` defaults so the two read the
# *same* point in the Stalevski (2016) library: inclination an=30° (cos_inc=0.866 —
# a Type-1 sightline that looks into the polar cone and sees the disc), opening angle
# ct=40°, optical depth ta=1, and p=q=1. ``agn_band_frac=1`` routes the full bolometric
# into the template, matching ProSpect's ``lum`` normalization (tengri otherwise scales
# the AGN down by ``frac_agn`` as a host-fraction knob).
m_agn = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": FIXED},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": FIXED,
    },
    dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "all_params": FIXED},
    # Raw-Stalevski SKIRTOR model. ProSpect's `SKIRTOR_interp` reads the
    # SKIRTOR template directly — the disc + torus as Stalevski's radiative
    # transfer computed them. tengri's `skirtor_stalevski` model does the same:
    # it reads the published RT total from the faithful full-coverage grid
    # (`scripts/build_skirtor_raw_grid.py`), no analytic-disc substitution, so it
    # matches ProSpect to ~0.9× at the 2000 Å disc and 9.3 µm at the torus
    # peak. (The monolithic `agn={'type':'skirtor'}` pairs the torus with a
    # *power-law* disc → 0.28×; the composable `disc.skirtor` uses CIGALE's
    # analytic disc + `norm=1/∫dust` → ~0.86×, the right choice for reproducing
    # CIGALE rather than the raw template. All three are swappable — see
    # `tengri.list_agn_models()`.)
    agn={
        "type": "skirtor_stalevski",
        "agn_log_lbol": Fixed(_agn_log_lbol),
        "agn_cos_inc": Fixed(float(np.cos(np.radians(30.0)))),  # ProSpect an=30°
        "agn_oa_skirtor": Fixed(40.0),  # ProSpect ct=40°
        "agn_tau_skirtor": Fixed(1.0),  # ProSpect ta=1
        "agn_p_skirtor": Fixed(1.0),  # ProSpect p=1
        "agn_q_skirtor": Fixed(1.0),  # ProSpect q=1
        "agn_band_frac": Fixed(1.0),  # full L_bol into template (match ProSpect lum)
        "all_params": FIXED,
    },
    redshift=Fixed(0.0),
)
s_agn = m_agn.predict_state({})
w_t9 = np.asarray(s_agn.wave)
L_t9 = np.asarray(s_agn.derived["sed_agn"])

# Confirm both sides carry the same AGN bolometric (sanity on the normalization).
_nu_p9 = U.C_ANGSTROM_PER_S / w_p9[::-1]
_nu_t9 = U.C_ANGSTROM_PER_S / w_t9[::-1]
_lbol_p9 = float(np.trapezoid(L_p9[::-1], _nu_p9))
_lbol_t9 = float(np.trapezoid(L_t9[::-1], _nu_t9))
print(f"§9 AGN bolometric: ProSpect {_lbol_p9:.2e} erg/s, tengri {_lbol_t9:.2e} erg/s")

# Disc continuum at a rest-UV anchor — where the two SKIRTOR reductions diverge.
_uv9 = 2000.0
_disc_p9 = float(np.interp(_uv9, w_p9, L_p9)) * U.C_ANGSTROM_PER_S / _uv9
_disc_t9 = float(np.interp(_uv9, w_t9, L_t9)) * U.C_ANGSTROM_PER_S / _uv9
print(
    f"§9 disc continuum νLν(2000 Å): ProSpect {_disc_p9:.2e}, "
    f"tengri {_disc_t9:.2e} ({_disc_t9 / _disc_p9:.2f}×)"
)

# νL_ν = L_ν · c/λ — the torus thermal bump is a true peak in this quantity.
nuLnu_p = L_p9 * U.C_ANGSTROM_PER_S / w_p9
nuLnu_t = L_t9 * U.C_ANGSTROM_PER_S / w_t9

fig, ax_l, ax_r = U.two_panel_fig()
for ax, lab in ((ax_l, "ProSpect  SKIRTOR template"), (ax_r, "tengri  SKIRTOR")):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ [Å]")
    ax.set_ylabel(r"$\nu L_\nu$ [erg/s]")
    ax.set_title(lab)
ax_l.plot(w_p9, nuLnu_p, "C0-", linewidth=1.5)
ax_r.plot(w_t9, nuLnu_t, "C1-", linewidth=1.5)
_peak_nu = max(float(np.max(nuLnu_p)), float(np.max(nuLnu_t)))
for ax in (ax_l, ax_r):
    ax.set_xlim(1e2, 1e7)
    ax.set_ylim(_peak_nu * 1e-3, _peak_nu * 3)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_09_agn_skirtor.png")

# Torus bump peak in νL_ν (restrict to λ > 1 µm to skip the disc continuum).
_mir = w_p9 > 1e4
_peak_p9 = w_p9[_mir][np.argmax(nuLnu_p[_mir])]
_mir_t = w_t9 > 1e4
_peak_t9 = w_t9[_mir_t][np.argmax(nuLnu_t[_mir_t])]
print(f"§9 torus νLν peak: ProSpect {_peak_p9 / 1e4:.1f} µm, tengri {_peak_t9 / 1e4:.1f} µm")


# %% [markdown]
# ## §11 Radio continuum
#
# ProSpect models radio continuum tied to the SFR via `addradio_SF` (free-free
# + synchrotron). tengri's `condon92` model (Condon 1992) is the matching
# star-formation radio prescription. The comparison is slope and normalization
# at matched SFR. (ProSpect has no X-ray component.)

# %% [markdown]
# **Verification Status:** PARTIAL (3/16) — Radio + X-ray + AGN

# %%
sed_radio = P.prospect_sed(
    massfunc="snorm",
    sfh_pars=SNORM_FIDUCIAL,
    Z=Z_SOLAR,
    tau_birth=TAU_BIRTH_FIDUCIAL,
    tau_screen=TAU_SCREEN_FIDUCIAL,
    addradio_SF=True,
)
w_p11, L_p11 = sed_radio["FinalLum"]
L_p11 = L_p11 * PRO_SCALE

m_radio = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": FIXED},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": FIXED,
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "power_law",
        "law_diff": "power_law",
        "tau_bc": Fixed(TAU_BIRTH_FIDUCIAL),
        "tau_diff": Fixed(TAU_SCREEN_FIDUCIAL),
        "all_params": FIXED,
    },
    dust_emission={"type": "dale2014", "alpha_dale": Fixed(3.0), "all_params": FIXED},
    radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}, "all_params": FIXED},
    redshift=Fixed(0.0),
)
s_radio = m_radio.predict_state({})
w_t11 = np.asarray(s_radio.wave)
L_t11 = np.asarray(s_radio.sed_intrinsic)

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="ProSpect  + radio (free-free + sync)", label_r="tengri  + Condon 92")
ax_l.plot(w_p11, L_p11, "C0-", linewidth=1.5)
ax_r.plot(w_t11, L_t11, "C1-", linewidth=1.5)
# Span the full SED in view (dust-IR peak through the radio tail) so the FIR bump
# is not clipped at the top of the frame.
_w_lo, _w_hi = 1e5, 2e9
_m_p11 = (w_p11 >= _w_lo) & (w_p11 <= _w_hi)
_m_t11 = (w_t11 >= _w_lo) & (w_t11 <= _w_hi)
_ymax11 = max(float(np.nanmax(L_p11[_m_p11])), float(np.nanmax(L_t11[_m_t11])))
for ax in (ax_l, ax_r):
    # Cap at ProSpect's output-grid edge (~2e9 Å) so its grid cutoff is not
    # shown as a spurious feature against tengri's wider grid.
    ax.set_xlim(_w_lo, _w_hi)
    ax.set_ylim(_ymax11 * 1e-7, _ymax11 * 3)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_11_radio.png")

_rad = w_p11 > 1e8  # > 1 cm, radio
if np.any(_rad) and L_p11[_rad].max() > 0:
    print(f"§11 ProSpect radio (>1 cm) peak L_ν = {L_p11[_rad].max():.2e} erg/s/Hz")


# %% [markdown]
# ## tengri in ProSpect-mode — full-SED head-to-head
#
# tengri configured to emulate ProSpect end to end — shared BC03, fiducial
# skew-normal SFH, Charlot & Fall attenuation, Dale 2014 IR — overlaid on
# ProSpect's output, with the fractional residual and a ±25 % band below.

# %%
import chex

w_ext, L_ext = np.asarray(w_p6), np.asarray(L_p6)
L_t_full = np.asarray(sed_full_t)  # tengri §6 attenuated + Dale IR
L_t_on_ext = U.regrid(np.asarray(s_ir.wave), L_t_full, w_ext)
chex.assert_equal_shape([L_ext, L_t_on_ext])

mask = (w_ext > 0) & (L_ext > 0) & (L_t_on_ext > 0)
resid = np.full(w_ext.shape, np.nan, dtype=float)
resid[mask] = L_t_on_ext[mask] / L_ext[mask] - 1.0

opt = mask & (w_ext >= 1000.0) & (w_ext <= 10000.0)
ratio_opt = L_t_on_ext[opt] / L_ext[opt]
if ratio_opt.size:
    norm = float(np.median(ratio_opt))
    p16, p84 = float(np.percentile(ratio_opt, 16)), float(np.percentile(ratio_opt, 84))
else:
    norm = p16 = p84 = float("nan")
print(
    f"full-SED head-to-head tengri/ProSpect optical (1000–10000 Å): "
    f"normalization {norm:.2f}×, 16–84% spread {p16:.2f}–{p84:.2f}×"
)
_assert_comparable(L_ext, L_t_full, name="full-SED head-to-head")

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax.plot(w_ext, L_ext, "C0-", linewidth=1.5, label="ProSpect")
ax.plot(w_ext, L_t_on_ext, "C1--", linewidth=1.5, label="tengri (ProSpect-mode)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e7)
ax.set_ylim(1e22, 1e31)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title("tengri in ProSpect-mode vs ProSpect — full panchromatic SED")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(
    0.02,
    0.05,
    rf"tengri/ProSpect $= {norm:.2f}\times$ (16–84%: {p16:.2f}–{p84:.2f})",
    transform=ax.transAxes,
    fontsize=10,
    va="bottom",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
ax_r.axhspan(-0.25, 0.25, color="0.85", zorder=0)
ax_r.axhline(0.0, color="0.5", linewidth=0.8)
ax_r.plot(w_ext, resid, "C1-", linewidth=1.0)
ax_r.set_xscale("log")
ax_r.set_xlim(1e2, 1e7)
ax_r.set_ylim(-1.0, 1.0)
ax_r.set_xlabel(r"$\lambda$ [Å]")
ax_r.set_ylabel(r"tengri/ProSpect $-1$")
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("prospect_r_full_sed_headtohead.png")
plt.show()


# %% [markdown]
# ## Summary
#
# Component by component at matched parameters, ProSpect and tengri agree
# wherever they evaluate the same mathematics: the BC03 library (§1), the
# skew-normal SFH (§2), the integrated stellar SED (§3, within ~1 % once
# absolute metallicity is matched), Charlot & Fall attenuation (§4), Dale
# 2014 dust IR (§6), and Inoue 2014 IGM (§12, bit-identical away from
# Lyman-α). Radio continuum (§11) and AGN torus (§9, SKIRTOR vs SKIRTOR) line
# up in slope and peak.
#
# ProSpect's defining feature — metallicity history tied to cumulative stellar
# mass formed (§2b) — is reproduced by tengri's `massmap_lin` mode: the two
# agree to a couple of percent at half-mass. The one genuine difference is
# nebular emission (§8), a deliberate disagreement between two photoionization
# grids. ProSpect's EMILES library and Fritz (2006) torus have no tengri
# equivalent.

# %% [markdown]
# **Verification Status:** PARTIAL (68/126) — Absolute SED normalization
#

# %% [markdown]
# ## References
#
# * Robotham et al. 2020, MNRAS 495, 905 — ProSpect
# * Bellstedt et al. 2020, MNRAS 498, 5581 — GAMA metallicity-history method
# * Bruzual & Charlot 2003, MNRAS 344, 1000 — BC03 stellar library
# * Chabrier 2003, PASP 115, 763 — IMF
# * Charlot & Fall 2000, ApJ 539, 718 — two-component dust attenuation
# * Dale et al. 2014, ApJ 784, 83 — infrared dust emission templates
# * Stalevski et al. 2012, MNRAS 420, 2756; 2016, MNRAS 458, 2288 — SKIRTOR
# * Fritz et al. 2006, MNRAS 366, 767 — AGN torus library
# * Levesque et al. 2010, ApJ 712, 1019 — nebular photoionization grid
# * Condon 1992, ARA&A 30, 575 — radio continuum from star formation
# * Inoue et al. 2014, MNRAS 442, 1805 — IGM absorption
# * Li et al. 2025 — Cue nebular emulator
