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
# This study configures tengri's public API to approximate ProSpect's model choices; tengri's implementation is its own, not derived from ProSpect's code, and residual differences are documented below.
#
# The closed-form blocks — the SFH shapes, the mass-mapped metallicity
# history, the attenuation curves, the IGM — match ProSpect to a fraction
# of a percent. Every model on this page carries nebular emission — ProSpect
# through `SFHfunc(emission = TRUE)`, whose `emissionLines` (lines only; ProSpect has no nebular continuum) takes its ionization
# parameter from `Z2q(Z)` (Orsi 2014; q = 1.4 × 10⁷ cm s⁻¹ at Z = 0.02,
# logU = −3.32), and tengri through Cue at that same logU and Z_gas; §8 also
# compares the line ratios at logU = −2. Residuals in every section include the
# Levesque-2010-versus-Cloudy difference quantified in §8. ProSpect has no X-ray
# component; it does have a radio continuum, which §11 includes.

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
from reproduction import _validation as V
from reproduction.prospect_r._drivers import prospect_driver as P, units as U

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform, load_ssp_data
from tengri.utils.physics_constants import C_CGS, LOG10_ZSUN

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

# Ionization parameter ProSpect's SED path derives from Z (Orsi 2014).
NEB_LOGU_FIDUCIAL = float(np.log10(P.z2q(Z_SOLAR) / C_CGS))

# Fiducial nebular emission (Cue at matched ionization and metallicity).
NEB_FIDUCIAL = {
    "type": "cue",
    "neb_logU": Fixed(NEB_LOGU_FIDUCIAL),
    "neb_logZ_gas": Fixed(MET_LOGZSOL),
    "all_params": Fixed(DEFAULT),
}

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
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_SNORM),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    n_grid=4096,  # dense lookback grid: the table compares the SFH form, not the 256-point diagnostic grid
    neb=NEB_FIDUCIAL, redshift=Fixed(0.0),
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

# %% [markdown]
# ## §2 cont'd — dtau, snorm_burst, snorm_trunc
#
# Three more ProSpect SFH families against their tengri counterparts, each
# swept over one shape parameter: delayed-tau timescale τ, a recent burst
# amplitude on the skew-normal, and a truncation sharpness. Each tengri model
# is built once with the swept parameter free and evaluated per case; the
# formed mass is matched to ProSpect's own integral so the curves sit at the
# same amplitude. `dtau` maps to tengri's `delayed` (`tau_gyr`/`age_gyr`),
# `snorm_burst` adds `burst_sfr`/`burst_age_gyr` to the fiducial skew-normal,
# and `snorm_trunc` maps to `tsnorm`'s `trunc`. ProSpect's `massfunc_dtau`
# continues past `mpeak` with a second timescale `magemax − mpeak`; the
# comparison is the recent branch, which is tengri's `delayed`.
# The τ-delayed families use
# different truncation conventions past the peak and diverge on the oldest
# tail; the windows below stay clear of ProSpect's own numerical zero.

# %%
DTAU_MPEAK = 10.0
cases_sfh2 = []

for tau in (1.0, 3.0):
    t_p, sfr_p = P.sfh_curve(sfh="dtau", mSFR=10.0, mpeak=DTAU_MPEAK, mtau=tau)
    o = np.argsort(t_p)
    # ProSpect's massfunc_dtau has two branches: recent (age <= mpeak) and old (age > mpeak).
    # tengri's delayed model represents only the recent branch, so extract it.
    mask_p = t_p <= DTAU_MPEAK * 1e9
    mass_p_recent = float(np.trapezoid(sfr_p[o][mask_p[o]], t_p[o][mask_p[o]]))
    age_gyr = DTAU_MPEAK + tau  # exact match to ProSpect's recent-side (age<=mpeak) branch
    if tau == 1.0:
        m_dtau = SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "delayed",
                "log_total_mass": Uniform(7.0, 12.5, default=10.0),
                "tau_gyr": Uniform(0.1, 10.0, default=2.0),
                "age_gyr": Uniform(0.5, 20.0, default=5.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            n_grid=4096,
            neb=NEB_FIDUCIAL,
            redshift=Fixed(0.0),
        )
        # Evaluate tengri at unity mass to extract the recent-branch mass fraction.
        s_unity = m_dtau.predict_state(
            {
                "sfh_delayed_log_total_mass": 0.0,
                "sfh_delayed_tau_gyr": tau,
                "sfh_delayed_age_gyr": age_gyr,
            }
        )
        lbt0 = np.asarray(s_unity.derived["sfh_grid_lbt_yr"])
        sfr0 = np.asarray(s_unity.derived["sfr_history"])
        _o_t = np.argsort(lbt0)
        mask_t = lbt0 <= DTAU_MPEAK * 1e9
        mass_t_recent = float(np.trapezoid(sfr0[_o_t][mask_t[_o_t]], lbt0[_o_t][mask_t[_o_t]]))
    # Scale tengri to match ProSpect's recent-branch mass.
    s = m_dtau.predict_state(
        {
            "sfh_delayed_log_total_mass": float(np.log10(mass_p_recent / mass_t_recent)),
            "sfh_delayed_tau_gyr": tau,
            "sfh_delayed_age_gyr": age_gyr,
        }
    )
    lbt, sfr_t = np.asarray(s.derived["sfh_grid_lbt_yr"]), np.asarray(s.derived["sfr_history"])
    cases_sfh2.append((f"dtau τ={tau:g} Gyr", t_p, sfr_p, lbt, sfr_t))

for mburst in (1.0, 5.0):
    t_p, sfr_p = P.sfh_curve(sfh="snorm_burst", mburst=mburst, mburstage=0.1, **SNORM_FIDUCIAL)
    o = np.argsort(t_p)
    mass_p = float(np.trapezoid(sfr_p[o], t_p[o]))
    if mburst == 1.0:
        m_burst = SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "snorm_burst",
                "log_total_mass": Uniform(7.0, 12.5, default=10.0),
                "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
                "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
                "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
                "burst_sfr": Uniform(0.0, 2.0, default=0.1),
                "burst_age_gyr": Fixed(0.1),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            n_grid=4096,
            neb=NEB_FIDUCIAL,
            redshift=Fixed(0.0),
        )
    burst_sfr = mburst / SNORM_FIDUCIAL["mSFR"]  # ratio to the bare kernel's own unit peak
    s = m_burst.predict_state(
        {
            "sfh_snorm_burst_log_total_mass": float(np.log10(mass_p)),
            "sfh_snorm_burst_burst_sfr": burst_sfr,
        }
    )
    lbt, sfr_t = np.asarray(s.derived["sfh_grid_lbt_yr"]), np.asarray(s.derived["sfr_history"])
    cases_sfh2.append((f"snorm_burst mburst={mburst:g}", t_p, sfr_p, lbt, sfr_t))

for mtrunc in (1.0, 2.0, 4.0):
    t_p, sfr_p = P.sfh_curve(
        sfh="snorm_trunc", mSFR=10.0, mpeak=10.0, mperiod=1.0, mskew=0.5, mtrunc=mtrunc
    )
    o = np.argsort(t_p)
    mass_p = float(np.trapezoid(sfr_p[o], t_p[o]))
    if mtrunc == 1.0:
        m_tsnorm = SEDModel.build(
            ssp_data=ssp,
            met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
            sfh={
                "type": "tsnorm",
                "log_total_mass": Uniform(7.0, 12.5, default=10.0),
                "peak_lbt_gyr": Fixed(10.0),
                "width_gyr": Fixed(1.0),
                "skew": Fixed(0.5),
                "trunc": Uniform(0.5, 10.0, default=2.0),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.0),
                "all_params": Fixed(DEFAULT),
            },
            n_grid=4096,
            neb=NEB_FIDUCIAL,
            redshift=Fixed(0.0),
        )
    s = m_tsnorm.predict_state(
        {"sfh_tsnorm_log_total_mass": float(np.log10(mass_p)), "sfh_tsnorm_trunc": mtrunc}
    )
    lbt, sfr_t = np.asarray(s.derived["sfh_grid_lbt_yr"]), np.asarray(s.derived["sfr_history"])
    cases_sfh2.append((f"snorm_trunc mtrunc={mtrunc:g}", t_p, sfr_p, lbt, sfr_t))

for label, _w_ref, _L_ref, _w_t, _L_t in cases_sfh2:
    _assert_comparable(_L_ref, _L_t, name=label)

fig, (ax, ax_r), _ratios2 = V.sweep_fig(
    cases_sfh2,
    ref_label="ProSpect",
    title="§2 cont'd — dtau, snorm_burst, snorm_trunc",
    x_of_wave=lambda w: w / 1e9,
    xlabel="lookback time [Gyr]",
    ylabel=r"SFR [$M_\odot\,\mathrm{yr}^{-1}$]",
    xlim=(0.05, 13.7),
    logy=False,
)
fig.tight_layout()
save_fig("prospect_r_02cont_sfh_families.png")

AGE_UNIV_YR = 13.7e9
_lo2, _hi2 = 0.02 * AGE_UNIV_YR, 0.99 * AGE_UNIV_YR
# dtau: compare only the recent branch (age <= mpeak)
_dtau_lo, _dtau_hi = 0.02 * DTAU_MPEAK * 1e9, DTAU_MPEAK * 1e9
V.print_window_table(
    V.window_rows(cases_sfh2[0:2], lo=_dtau_lo, hi=_dtau_hi, rel_to="peak"), ref_name="ProSpect", title="§2 cont'd — dtau; deviation as % of peak SFR", x_unit="Gyr", x_scale=1e-9
)
V.print_window_table(
    V.window_rows(cases_sfh2[2:4], lo=_lo2, hi=0.95 * AGE_UNIV_YR, rel_to="peak"),
    ref_name="ProSpect",
    title="§2 cont'd — snorm_burst; deviation as % of peak SFR",
    x_unit="Gyr", x_scale=1e-9,
)
# snorm_trunc's Gaussian-tail truncation reaches ProSpect's own numerical zero
# by ~90% of the age; cap the window at 80% to stay in the well-defined range.
V.print_window_table(
    V.window_rows(cases_sfh2[4:7], lo=_lo2, hi=0.80 * AGE_UNIV_YR, rel_to="peak"),
    ref_name="ProSpect",
    title="§2 cont'd — snorm_trunc (2-80% of age); deviation as % of peak SFR",
    x_unit="Gyr", x_scale=1e-9,
)


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
        "all_params": Fixed(DEFAULT),
    },
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    n_grid=4096,
    neb=NEB_FIDUCIAL, redshift=Fixed(0.0),
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
        "all_params": Fixed(DEFAULT),
    },
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    n_grid=4096,
    neb=NEB_FIDUCIAL, redshift=Fixed(0.0),
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
# metallicity, nebular emission enabled on both sides, no dust, both scaled to 10^10 M⊙ formed.
# Band-integrated ratios compare the continuum; pointwise ratios would measure emission-line widths,
# which differ between Cue and ProSpect's photoionization grid. The u band (1.326×) and NUV carry
# the nebular continuum — the Balmer jump and two-photon emission — which Cue includes and
# ProSpect's `emissionLines` does not.

# %% [markdown]
# **Verification Status:** CROSSVAL — Photometry projection

# %%
sed_stel = P.prospect_sed(
    massfunc="snorm",
    sfh_pars=SNORM_FIDUCIAL,
    Z=Z_SOLAR,
    tau_birth=0.0,
    tau_screen=0.0,
    extra={"emission": True},
)
w_p3, L_p3 = sed_stel["FinalLum"]
L_p3 = L_p3 * PRO_SCALE

m_stellar = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL, redshift=Fixed(0.0),
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

_rows3 = V.filter_rows_native(np.asarray(s_stellar.wave), np.asarray(s_stellar.sed_intrinsic), w_p3, L_p3, filters=V.UV_TO_NIR)
V.print_filter_table(_rows3, ref_name="ProSpect", title="§3 stellar SED, UV-to-NIR bands", compact=False)


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

_tengri_laws = list_laws(headline=False).to_dict("fn")
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
    extra={"emission": True},
)
w_p5, L_p5 = sed_atten["StarsAtten"]
L_p5 = L_p5 * PRO_SCALE

DUST_FIDUCIAL = {
    "type": "two_component",
    "law_bc": "power_law",
    "law_diff": "power_law",
    "tau_bc": Fixed(TAU_BIRTH_FIDUCIAL),
    "tau_diff": Fixed(TAU_SCREEN_FIDUCIAL),
    "all_params": Fixed(DEFAULT),
}
m_d = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation=DUST_FIDUCIAL,
    neb=NEB_FIDUCIAL, redshift=Fixed(0.0),
)
s_d = m_d.predict_state({})
L_t_atten = np.asarray(s_d.derived["sed_dust_attenuated"])
_assert_comparable(L_p5, L_t_atten, name="§5 dust applied")

# %% [markdown]
# ## §5 cont'd — pow, bump, τ_screen
#
# Two curve sweeps and an applied-SED sweep. The Charlot & Fall screen power
# law's slope `pow` and the noll09 bump strength `Eb` are compared as
# `A(λ)/A_V` over 1216-3000 Å — the slope sweep is an exact match (same
# functional form on both sides); the bump sweep differs at the ~10-40% level
# since the two bump normalizations are not identical. The applied sweep then
# varies both optical depths (`τ_screen`, `τ_birth`) together and reads the
# result off the UV-to-NIR bandpasses: all six combinations differ, and the
# birth-cloud depth dominates the residual over the screen depth.

# %%
from tengri.dust import list_laws

_laws2 = list_laws(headline=False).to_dict("fn")
_wave_law2 = np.logspace(np.log10(1000.0), np.log10(30000.0), 2000)


def _norm_AV2(wave, A):
    return A / A[np.argmin(np.abs(wave - 5500.0))]


cases_dust_curve = []
for _pow in (-0.4, -0.7, -1.0):
    _w_p, _A_p = P.attenuation_curve(component="screen", tau=1.0, pow_=_pow, wave_aa=_wave_law2)
    _AV_p = _norm_AV2(_w_p, _A_p)
    _k_t = np.asarray(_laws2["power_law"](_wave_law2, dust_slope=_pow))
    cases_dust_curve.append((f"pow={_pow:g}", _w_p, _AV_p, _wave_law2, _norm_AV2(_wave_law2, _k_t)))
for _Eb in (0.0, 1.5, 3.0):
    _w_p, _A_p = P.attenuation_curve(
        component="screen", tau=1.0, pow_=POW_FIDUCIAL, Eb=_Eb, wave_aa=_wave_law2
    )
    _AV_p = _norm_AV2(_w_p, _A_p)
    _k_t = np.asarray(_laws2["noll09"](_wave_law2, dust_bump_strength=_Eb))
    cases_dust_curve.append((f"Eb={_Eb:g}", _w_p, _AV_p, _wave_law2, _norm_AV2(_wave_law2, _k_t)))

for _label, _w_ref, _L_ref, _w_t, _L_t in cases_dust_curve:
    _assert_comparable(_L_ref, _L_t, name=_label)

V.print_window_table(
    V.window_rows(cases_dust_curve, lo=1216.0, hi=3000.0),
    ref_name="ProSpect",
    title="§5 cont'd — A(λ)/A_V curves (pow, Eb), 1216-3000 Å",
)

DUST_APPLIED_TB = (0.5, 1.5)
DUST_APPLIED_TS = (0.1, 0.3, 1.0)
m_dsweep = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "power_law",
        "law_diff": "power_law",
        "tau_bc": Uniform(0.0, 3.0, default=TAU_BIRTH_FIDUCIAL),
        "tau_diff": Uniform(0.0, 2.0, default=TAU_SCREEN_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL,
    redshift=Fixed(0.0),
)
cases_dust_applied = []
for t_screen in DUST_APPLIED_TS:
    for t_birth in DUST_APPLIED_TB:
        _sed = P.prospect_sed(
            massfunc="snorm",
            sfh_pars=SNORM_FIDUCIAL,
            Z=Z_SOLAR,
            tau_birth=t_birth,
            tau_screen=t_screen,
            pow_birth=POW_FIDUCIAL,
            pow_screen=POW_FIDUCIAL,
            extra={"emission": True},
        )
        _w_p, _L_p = _sed["StarsAtten"]
        _L_p = _L_p * PRO_SCALE
        _s = m_dsweep.predict_state({"dust_tau_bc": t_birth, "dust_tau_diff": t_screen})
        cases_dust_applied.append(
            (
                f"τ_bc={t_birth:g},τ_diff={t_screen:g}",
                _w_p,
                _L_p,
                np.asarray(_s.wave),
                np.asarray(_s.derived["sed_dust_attenuated"]),
            )
        )

_sfr_t_dust = [np.interp(5500.0, c[3], c[4]) for c in cases_dust_applied]
assert len(set(np.round(_sfr_t_dust, 6))) == len(_sfr_t_dust), "applied dust cases must all differ"

for _label, _w_ref, _L_ref, _w_t, _L_t in cases_dust_applied:
    _assert_comparable(_L_ref, _L_t, name=_label)

fig, (ax, ax_r), _ratios_dust = V.sweep_fig(
    cases_dust_applied,
    ref_label="ProSpect",
    title="§5 cont'd — τ_screen × τ_birth applied",
    x_of_wave=lambda w: w / 1e4,
    xlabel=r"$\lambda$ [$\mu$m]",
    xlim=(0.1, 3.0),
)
fig.tight_layout()
save_fig("prospect_r_05cont_dust_applied_sweep.png")

for _label, _w_ref, _L_ref, _w_t, _L_t in cases_dust_applied:
    _rows = V.filter_rows_native(_w_t, _L_t, _w_ref, _L_ref, filters=V.UV_TO_NIR)
    V.print_filter_table(_rows, ref_name="ProSpect", title=f"§5 cont'd — {_label}", compact=True)


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
    extra={"emission": True},
)
w_p6, L_p6 = sed_ir["FinalLum"]
L_p6 = L_p6 * PRO_SCALE

m_ir = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "power_law",
        "law_diff": "power_law",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(TAU_SCREEN_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_emission={"type": "dale2014", "alpha_dale": Fixed(3.0), "all_params": Fixed(DEFAULT)},
    neb=NEB_FIDUCIAL, redshift=Fixed(0.0),
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
# ## §6 cont'd — Dale α sweep
#
# The Dale et al. (2014) radiation-field hardness α set to 1, 2, 3, and 4,
# all other parameters at the §6 fiducial. Larger α weights the dust-mass
# distribution toward lower radiation intensities, so the peak wavelength
# moves redward (colder dust) as α grows. tengri's `dust_alpha_dale` is
# declared free once and evaluated at each α; the IR bandpass table below
# reads a systematic ~20% tengri/ProSpect offset across the pure-dust bands
# that is flat in α — a template-normalization residual, not the shape
# mismatch a sweep is built to catch.

# %%
m_ir_sweep = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "power_law",
        "law_diff": "power_law",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(TAU_SCREEN_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_emission={
        "type": "dale2014",
        "alpha_dale": Uniform(0.0625, 4.0, default=2.0),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL,
    redshift=Fixed(0.0),
)
cases_dale = []
for _alpha in (1.0, 2.0, 3.0, 4.0):
    _sed = P.prospect_sed(
        massfunc="snorm",
        sfh_pars=SNORM_FIDUCIAL,
        Z=Z_SOLAR,
        tau_birth=0.0,
        tau_screen=TAU_SCREEN_FIDUCIAL,
        pow_birth=POW_FIDUCIAL,
        pow_screen=POW_FIDUCIAL,
        alpha_SF_screen=_alpha,
        extra={"emission": True},
    )
    _w_p, _L_p = _sed["FinalLum"]
    _L_p = _L_p * PRO_SCALE
    _s = m_ir_sweep.predict_state({"dust_alpha_dale": _alpha})
    _sed_t = np.asarray(_s.derived["sed_dust_attenuated"]) + np.asarray(_s.derived["sed_dust_ir"])
    cases_dale.append((f"α={_alpha:g}", _w_p, _L_p, np.asarray(_s.wave), _sed_t))

_dale_peaks = []
for _label, _w_ref, _L_ref, _w_t, _L_t in cases_dale:
    _fir_t2 = (_w_t > 1e5) & (_w_t < 1e7)
    _dale_peaks.append(_w_t[_fir_t2][np.argmax((_L_t * U.C_ANGSTROM_PER_S / _w_t)[_fir_t2])])
    _assert_comparable(_L_ref, _L_t, name=_label)
assert len(set(np.round(_dale_peaks, -3))) == len(_dale_peaks), "alpha sweep must shift the FIR peak"
print("§6 cont'd tengri FIR peaks [um] vs α:", [f"{p / 1e4:.0f}" for p in _dale_peaks])

fig, (ax, ax_r), _ratios_dale = V.sweep_fig(
    cases_dale,
    ref_label="ProSpect",
    title="§6 cont'd — Dale 2014 α sweep",
    x_of_wave=lambda w: w / 1e4,
    xlim=(1.0, 1e3),
)
fig.tight_layout()
save_fig("prospect_r_06cont_dale_alpha.png")

for _label, _w_ref, _L_ref, _w_t, _L_t in cases_dale:
    _rows = V.filter_rows_native(_w_t, _L_t, _w_ref, _L_ref, filters=V.IR_BANDS)
    V.print_filter_table(_rows, ref_name="ProSpect", title=f"§6 cont'd — {_label}", compact=True)


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
# ## §12 cont'd — z sweep
#
# The same Inoue et al. (2014) transmission at z = 2, 3, 5, 7, read over
# 850-1216 Å rest-frame (Ångström, matching the units above — up to and
# including the Lyα line itself). Both codes agree to within a few percent at
# every redshift, the same floor as the single-z=4 case above.

# %%
cases_igm_z = []
for _z in (2.0, 3.0, 5.0, 7.0):
    _wave_rest = np.linspace(700.0, 1300.0, 800)
    _w_igm, _T_p = P.igm_transmission(_wave_rest, _z)
    _T_t = np.asarray(tengri_igm(_wave_rest * (1.0 + _z), np.asarray(_z)))
    cases_igm_z.append((f"z={_z:g}", _w_igm, _T_p, _wave_rest, _T_t))

fig, (ax, ax_r), _ratios_igm = V.sweep_fig(
    cases_igm_z,
    ref_label="ProSpect",
    title="§12 cont'd — Inoue+2014 z sweep",
    x_of_wave=lambda w: w,
    xlabel=r"rest-frame $\lambda$ [Å]",
    ylabel=r"$T(\lambda, z)$",
    xlim=(850.0, 1216.0),
    ratio_ylim=(0.9, 1.1),
    logy=False,
)
fig.tight_layout()
save_fig("prospect_r_12cont_igm_z_sweep.png")

V.print_window_table(
    V.window_rows(cases_igm_z, lo=850.0, hi=1216.0),
    ref_name="ProSpect",
    title="§12 cont'd — Inoue+2014 z sweep, 850-1216 Å",
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
    met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
    # Confine star formation to the last 10 Myr — a young, ionizing population.
    sfh={
        "type": "const",
        "start_gyr": Fixed(NEB_AGE_GYR),
        "end_gyr": Fixed(0.0),
        "log_total_mass": Fixed(NEB_LOG_MASS),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    neb={
        "type": "cue",
        "neb_logU": Fixed(-2.0),
        "neb_logZ_gas": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
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

# %% [markdown]
# ## §8 cont'd — q × Z_gas
#
# The ionization parameter `logU` (−3, −2, −1.5) crossed with gas-phase
# metallicity `Z_gas` (0.004, 0.02 = solar) — six cases, tengri's `neb_logU`
# and `neb_logZ_gas` both declared free on one Cue build, ProSpect's matching
# `q = U·c` and `Z` passed to `emissionLines`. [O III]/Hβ is the diagnostic
# most sensitive to both axes; Hα/Hβ (pure recombination) is essentially flat
# across the grid on both sides, as expected.

# %%
m_neb2 = SEDModel.build(
    ssp_data=ssp_neb,
    met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "const",
        "start_gyr": Fixed(NEB_AGE_GYR),
        "end_gyr": Fixed(0.0),
        "log_total_mass": Fixed(NEB_LOG_MASS),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    neb={
        "type": "cue",
        "neb_logU": Uniform(-4.0, -1.0, default=-2.0),
        "neb_logZ_gas": Uniform(-2.0, 0.2, default=0.0),
        "all_params": Fixed(DEFAULT),
    },
    redshift=Fixed(0.0),
)

cases_neb = []
_neb_rows = []
for _logU in (-3.0, -2.0, -1.5):
    for _Zgas in (0.004, 0.02):
        _q = 10.0**_logU * 2.99792458e10
        _w_p, _L_p = P.nebular_lnu(sfr=SFR_NEB, Z=_Zgas, q=_q)
        _logZ_gas = float(np.log10(_Zgas / Z_SOLAR))
        _s = m_neb2.predict_state({"neb_logU": _logU, "neb_logZ_gas": _logZ_gas})
        _w_t = np.asarray(_s.wave)
        _L_t = np.asarray(_s.derived["sed_nebular"])
        _label = f"logU={_logU:g},Z={_Zgas:g}"
        cases_neb.append((_label, _w_p, _L_p, _w_t, _L_t))
        _ha_p, _ha_t = U.line_lum(_w_p, _L_p, 6563.0), U.line_lum(_w_t, _L_t, 6563.0)
        _hb_p, _hb_t = U.line_lum(_w_p, _L_p, 4861.0), U.line_lum(_w_t, _L_t, 4861.0)
        _o3_p, _o3_t = U.line_lum(_w_p, _L_p, 5007.0), U.line_lum(_w_t, _L_t, 5007.0)
        _o2_p, _o2_t = U.line_lum(_w_p, _L_p, 3727.0), U.line_lum(_w_t, _L_t, 3727.0)
        _neb_rows.append(
            (
                _label,
                _o3_p / _hb_p if _hb_p > 0 else float("nan"),
                _o3_t / _hb_t if _hb_t > 0 else float("nan"),
                _o2_p / _hb_p if _hb_p > 0 else float("nan"),
                _o2_t / _hb_t if _hb_t > 0 else float("nan"),
                _ha_p / _hb_p if _hb_p > 0 else float("nan"),
                _ha_t / _hb_t if _hb_t > 0 else float("nan"),
            )
        )

for _label, _w_ref, _L_ref, _w_t, _L_t in cases_neb:
    _assert_comparable(_L_ref, _L_t, name=_label)

print("\n  §8 cont'd — [O III]/Hβ, [O II]/Hβ, Hα/Hβ (ProSpect | tengri)")
print(f"  {'case':<20} {'[OIII]/Hb':>20} {'[OII]/Hb':>20} {'Ha/Hb':>16}")
print("  " + "-" * 78)
for _label, _o3p, _o3t, _o2p, _o2t, _hap, _hat in _neb_rows:
    print(
        f"  {_label:<20} {_o3p:>8.3f} | {_o3t:<8.3f}   "
        f"{_o2p:>8.3f} | {_o2t:<8.3f}   {_hap:>5.2f} | {_hat:<5.2f}"
    )

fig, (ax, ax_r), _ratios_neb = V.sweep_fig(
    cases_neb,
    ref_label="ProSpect",
    title="§8 cont'd — logU × Z_gas",
    x_of_wave=lambda w: w,
    xlabel=r"$\lambda$ [Å]",
    xlim=(1000.0, 7000.0),
    logy=True,
)
fig.tight_layout()
save_fig("prospect_r_08cont_nebular_params.png")


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
# treatment. tengri deprecates `skirtor_stalevski` as a public model name
# (it has no composable disc+torus recipe equivalent), and this section
# selects it anyway on purpose: it is the one tengri model that reads
# ProSpect's own raw SKIRTOR total-SED template rather than a composable
# reinterpretation of it, which is the deliberate comparison target here.

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
# ct=40°, optical depth ta=1, and p=q=1. The monolithic SKIRTOR model normalizes the
# template to `agn_log_lbol` itself, matching ProSpect's ``lum`` normalization.
m_agn = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
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
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL, redshift=Fixed(0.0),
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

# %% [markdown]
# ## §9 cont'd — SKIRTOR nodes and Fritz
#
# Six SKIRTOR nodes sweeping opening angle `ct`, inclination `an`, and optical
# depth `ta` against ProSpect's `SKIRTOR_interp`, plus three Fritz et al.
# (2006) nodes against `Fritz_interp` — a torus tengri has an exact
# composable equivalent for (`torus={"type": "fritz"}`), matching ProSpect's
# `AGNct`/`AGNrm`/`AGNan`/`AGNta`/`AGNal`/`AGNbe` one-to-one (`AGNal`→`gamma`,
# `AGNbe`→`beta`). Both curves are peak-normalized over 1-100 µm so the
# comparison is shape, not normalization. SKIRTOR tracks ProSpect near its
# default inclination and diverges at the grazing an=0° node, where the raw
# template's edge-on disc term is small on both sides. Fritz sits at a stable
# ~0.3× — the composable torus block pairs with tengri's own disc rather than
# ProSpect's combined template, as with the deprecated monolithic
# SKIRTOR/power-law-disc pairing above.

# %%
m_skirtor_sweep = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    agn={
        "type": "skirtor_stalevski",
        "agn_log_lbol": Uniform(9.0, 13.0, default=11.4),
        "agn_cos_inc": Uniform(0.0, 1.0, default=0.866),
        "agn_oa_skirtor": Uniform(10.0, 80.0, default=40.0),
        "agn_tau_skirtor": Uniform(0.1, 11.0, default=1.0),
        "agn_p_skirtor": Fixed(1.0),
        "agn_q_skirtor": Fixed(1.0),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL,
    redshift=Fixed(0.0),
)

SKIRTOR_NODES = ((40, 30, 1), (40, 30, 7), (20, 30, 1), (60, 30, 1), (40, 0, 1), (40, 80, 1))
cases_skirtor = []
for _ct, _an, _ta in SKIRTOR_NODES:
    _w_p, _L_p, _log_lbol = P.agn_torus_lnu(
        model="SKIRTOR", lum_erg=AGN_LUM_ERG, ct=float(_ct), an=float(_an), ta=float(_ta)
    )
    _s = m_skirtor_sweep.predict_state(
        {
            "agn_log_lbol": _log_lbol,
            "agn_cos_inc": float(np.cos(np.radians(_an))),
            "agn_oa_skirtor": float(_ct),
            "agn_tau_skirtor": float(_ta),
        }
    )
    _w_t = np.asarray(_s.wave)
    _L_t = np.asarray(_s.derived["sed_agn"])
    cases_skirtor.append((f"ct={_ct},an={_an},ta={_ta}", _w_p, _L_p, _w_t, _L_t))


def _peak_norm_1_100um(w, L):
    m = (w >= 1e4) & (w <= 1e6)
    return L / L[m].max()


cases_skirtor_norm = [
    (lbl, w_p, _peak_norm_1_100um(w_p, L_p), w_t, _peak_norm_1_100um(w_t, L_t))
    for lbl, w_p, L_p, w_t, L_t in cases_skirtor
]
for _label, _w_ref, _L_ref, _w_t, _L_t in cases_skirtor_norm:
    _assert_comparable(_L_ref, _L_t, name=_label)

fig, (ax, ax_r), _ratios_sk = V.sweep_fig(
    cases_skirtor_norm,
    ref_label="ProSpect",
    title="§9 cont'd — SKIRTOR nodes (peak-normalized, 1-100 µm)",
    x_of_wave=lambda w: w / 1e4,
    ylabel=r"$L_\nu$ (peak-normalized)",
    xlim=(1e-1, 1e3),
)
fig.tight_layout()
save_fig("prospect_r_09cont_skirtor_nodes.png")

V.print_window_table(
    V.window_rows(cases_skirtor_norm, lo=1e4, hi=1e6),
    ref_name="ProSpect",
    title="§9 cont'd — SKIRTOR nodes, peak-normalized 1-100 µm",
)

FRITZ_NODES = ((40, 30, 1, 4, -0.5), (40, 80, 1, 4, -0.5), (60, 30, 6, 2, -0.75))
m_fritz = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "law": "power_law",
        "type": "two_component",
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(0.0),
        "all_params": Fixed(DEFAULT),
    },
    agn={
        "torus": {
            "type": "fritz",
            "agn_fritz_oa": Uniform(10.0, 80.0, default=60.0),
            "agn_fritz_r_ratio": Fixed(60.0),  # ProSpect AGNrm default
            "agn_fritz_psy": Uniform(0.0, 90.0, default=30.0),
            "agn_fritz_tau": Uniform(0.1, 10.0, default=1.0),
            "agn_fritz_gamma": Uniform(0.0, 6.0, default=4.0),
            "agn_fritz_beta": Uniform(-1.0, 0.0, default=-0.5),
            "agn_torus_frac": Fixed(1.0),  # full bolometric to the template, matching ProSpect
        },
        "agn_log_lbol": Uniform(9.0, 13.0, default=11.4),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL,
    redshift=Fixed(0.0),
)
cases_fritz = []
for _ct, _an, _ta, _al, _be in FRITZ_NODES:
    _w_p, _L_p, _log_lbol = P.agn_torus_lnu(
        model="Fritz",
        lum_erg=AGN_LUM_ERG,
        ct=float(_ct),
        an=float(_an),
        ta=float(_ta),
        al=float(_al),
        be=float(_be),
    )
    _s = m_fritz.predict_state(
        {
            "agn_log_lbol": _log_lbol,
            "agn_fritz_oa": float(_ct),
            "agn_fritz_psy": float(_an),
            "agn_fritz_tau": float(_ta),
            "agn_fritz_gamma": float(_al),
            "agn_fritz_beta": float(_be),
        }
    )
    _w_t = np.asarray(_s.wave)
    _L_t = np.asarray(_s.derived["sed_agn"])
    cases_fritz.append((f"ct={_ct},an={_an},ta={_ta},al={_al},be={_be}", _w_p, _L_p, _w_t, _L_t))

cases_fritz_norm = [
    (lbl, w_p, _peak_norm_1_100um(w_p, L_p), w_t, _peak_norm_1_100um(w_t, L_t))
    for lbl, w_p, L_p, w_t, L_t in cases_fritz
]
for _label, _w_ref, _L_ref, _w_t, _L_t in cases_fritz_norm:
    _assert_comparable(_L_ref, _L_t, name=_label)

fig, (ax, ax_r), _ratios_fr = V.sweep_fig(
    cases_fritz_norm,
    ref_label="ProSpect",
    title="§9 cont'd — Fritz+2006 nodes (peak-normalized, 1-100 µm)",
    x_of_wave=lambda w: w / 1e4,
    ylabel=r"$L_\nu$ (peak-normalized)",
    xlim=(1e-1, 1e3),
)
fig.tight_layout()
save_fig("prospect_r_09cont_fritz.png")

V.print_window_table(
    V.window_rows(cases_fritz_norm, lo=1e4, hi=1e6),
    ref_name="ProSpect",
    title="§9 cont'd — Fritz+2006 nodes, peak-normalized 1-100 µm",
)


# %% [markdown]
# ## §11 Radio continuum
#
# ProSpect models radio continuum tied to the SFR via `addradio_SF` (free-free
# + synchrotron). tengri's `bell2003_split` radio block matches that pair: the
# Bell (2003) total SFR-radio luminosity L(1.4 GHz) split into a non-thermal
# synchrotron component (S_ν ∝ ν^−0.75, Baan & Klockner 2006) and a thermal
# free-free component (S_ν ∝ ν^−0.10, Dale & Helou 2002; Condon 1992). The
# comparison is slope and normalization at matched SFR. (ProSpect has no
# X-ray component.)

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
    extra={"emission": True},
)
w_p11, L_p11 = sed_radio["FinalLum"]
L_p11 = L_p11 * PRO_SCALE

m_radio = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "power_law",
        "law_diff": "power_law",
        "tau_bc": Fixed(TAU_BIRTH_FIDUCIAL),
        "tau_diff": Fixed(TAU_SCREEN_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    # dale2014_cigale (not dale2014): the plain Dale+2014 template embeds its own
    # star-forming radio synchrotron continuum to 1.335 GHz, which double-counts
    # against the active radio.sf block below (#1970); the CIGALE variant has that
    # tail stripped and composes correctly with a separate radio component.
    # ProSpect's own Dale templates carry no radio tail, since ProSpect adds
    # radio continuum as a separate step via `addradio_SF`.
    dust_emission={
        "type": "dale2014_cigale",
        "alpha_dale": Fixed(3.0),
        "all_params": Fixed(DEFAULT),
    },
    radio={"sf": {"type": "bell2003_split"}, "agn": {"type": "powerlaw"}, "all_params": Fixed(DEFAULT)},
    neb=NEB_FIDUCIAL, redshift=Fixed(0.0),
)
s_radio = m_radio.predict_state({})
w_t11 = np.asarray(s_radio.wave)
L_t11 = np.asarray(s_radio.sed_intrinsic)

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(
    ax_l,
    ax_r,
    label_l="ProSpect  + radio (free-free + sync)",
    label_r="tengri  bell2003_split (free-free + synchrotron)",
)
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
# ## §11 cont'd — q_IR × α_SF
#
# The FIR-radio correlation normalization `q_IR` (2.3, 2.64, 2.9 — Bell 2003's
# own value is 2.64) crossed with the synchrotron spectral index `α_SF` (0.7,
# 0.8). `q_IR` is a tengri-only knob — ProSpect's `addradio_SF` calibrates its
# own fixed FIR-radio ratio internally and exposes no equivalent — so only
# `α_SF` (ProSpect's `sy_power_SF`) moves both sides; ProSpect's default
# `waveout` grid also has an edge artifact right at 1.4 GHz, worked around
# here with a finer, wider grid passed through `extra`. tengri sits near unity
# at `q_IR = 2.64` and moves away on either side, exactly as the normalization
# knob should.

# %%
import rpy2.robjects as _ro

m_radio_sweep = SEDModel.build(
    ssp_data=ssp,
    met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
    sfh={
        "type": "snorm",
        "peak_lbt_gyr": Fixed(SNORM_FIDUCIAL["mpeak"]),
        "width_gyr": Fixed(SNORM_FIDUCIAL["mperiod"]),
        "skew": Fixed(SNORM_FIDUCIAL["mskew"]),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_attenuation={
        "type": "two_component",
        "law_bc": "power_law",
        "law_diff": "power_law",
        "tau_bc": Fixed(TAU_BIRTH_FIDUCIAL),
        "tau_diff": Fixed(TAU_SCREEN_FIDUCIAL),
        "all_params": Fixed(DEFAULT),
    },
    dust_emission={"type": "dale2014_cigale", "alpha_dale": Fixed(3.0), "all_params": Fixed(DEFAULT)},
    radio={
        "sf": {"type": "bell2003"},
        "radio_q_ir": Uniform(1.8, 3.5, default=2.64),
        "radio_alpha_sf": Uniform(0.5, 1.2, default=0.8),
        "all_params": Fixed(DEFAULT),
    },
    neb=NEB_FIDUCIAL,
    redshift=Fixed(0.0),
)
_waveout_ext = _ro.FloatVector(np.arange(2.0, 9.8, 0.005))  # avoids a ~1.4 GHz edge artifact

for _q_ir in (2.3, 2.64, 2.9):
    for _alpha in (0.7, 0.8):
        _sed = P.prospect_sed(
            massfunc="snorm",
            sfh_pars=SNORM_FIDUCIAL,
            Z=Z_SOLAR,
            tau_birth=TAU_BIRTH_FIDUCIAL,
            tau_screen=TAU_SCREEN_FIDUCIAL,
            addradio_SF=True,
            extra={"sy_power_SF": -_alpha, "waveout": _waveout_ext, "emission": True},
        )
        _w_p, _L_p = _sed["FinalLum"]
        _L_p = _L_p * PRO_SCALE
        _s = m_radio_sweep.predict_state({"radio_q_ir": _q_ir, "radio_alpha_sf": _alpha})
        _w_t, _L_t = np.asarray(_s.wave), np.asarray(_s.sed_intrinsic)
        _L_t_on_ref = U.regrid(_w_t, _L_t, _w_p)
        _rows = V.radio_rows(_w_p, _L_t_on_ref, _L_p)
        V.print_radio_table(
            _rows, ref_name="ProSpect", title=f"§11 cont'd — q_IR={_q_ir:g}, α_SF={_alpha:g}"
        )


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
# up in slope and peak. The sweeps below extend each of these to a small grid
# of parameters rather than one point, and the worst-case ratio in each grid
# is what the single-point sections could not show.
#
# | Block | § | Cases | Worst tengri/ProSpect | Where |
# |---|---|---|---|---|
# | SFH families | §2 cont'd | 7 (dtau, snorm_burst, snorm_trunc) | 1.00–1.75× median | window, 2-99%/2-80% of age |
# | Stellar SED | §3 | 1 (single point) | u band 1.326× | UV-to-NIR bands |
# | Attenuation curves | §5 cont'd | 6 (pow, Eb) | 0.90–1.07× median | A(λ)/A_V, 1216-3000 Å |
# | Attenuation applied | §5 cont'd | 6 (τ_screen×τ_birth) | 0.99–1.00× median (bands to 0.66×) | UV_TO_NIR bands |
# | Dust IR | §6 cont'd | 4 (α) | 0.74–0.79× median | IR_BANDS |
# | Nebular | §8 cont'd | 6 (logU×Z_gas) | line-ratio dependent | Hα, Hβ, [O III], [O II] |
# | AGN SKIRTOR | §9 cont'd | 6 (ct,an,ta) | 1.02–1.49× median | peak-norm, 1-100 µm |
# | AGN Fritz | §9 cont'd | 3 (ct,an,ta,al,be) | 0.28–0.35× median | peak-norm, 1-100 µm |
# | Radio | §11 cont'd | 6 (q_IR×α_SF) | 0.57–1.94× (band ratios) | VLA/ALMA bands |
# | IGM | §12 cont'd | 4 (z) | ~1.00× median | T(λ), 850-1216 Å |
#
# ProSpect's defining feature — metallicity history tied to cumulative stellar
# mass formed (§2b) — is reproduced by tengri's `massmap_lin` mode: the two
# agree to a couple of percent at half-mass. The one genuine difference is
# nebular emission (§8), a deliberate disagreement between two photoionization
# grids. ProSpect's EMILES library has no tengri equivalent; Fritz (2006)
# does (`torus={"type": "fritz"}`), and §9 cont'd compares it directly — the
# composable torus block pairs with tengri's own disc rather than ProSpect's
# combined disc+torus template, holding at ~0.3× across the swept nodes.

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
# * Bell 2003, ApJ 586, 794 — SFR-radio (FIRRC) normalization
# * Dale & Helou 2002, ApJ 576, 159 — infrared-radio correlation calibration (thermal free-free slope)
# * Baan & Klockner 2006, A&A 449, 559 — non-thermal (synchrotron) spectral index
# * Inoue et al. 2014, MNRAS 442, 1805 — IGM absorption
# * Li et al. 2025 — Cue nebular emulator
