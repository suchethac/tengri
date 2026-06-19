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
# # Reproducing BAGPIPES with tengri
#
# BAGPIPES (Carnall et al. 2018) is the reference code for galaxy SED
# fitting at JWST cosmic noon, post-starburst quiescent populations,
# and rest-UV continuum work at high redshift. This notebook places
# its physics modules — `delayed`, `constant`, `dust` (Calzetti,
# Cardelli, Salim, CF00), `nebular` (Cloudy 25), `dust_emission`
# (Draine & Li 2007), and the Inoue+2014 IGM — next to their tengri
# equivalents on the same axes, in the same units, at matched
# parameter values.
#
# Both codes consume the same numerical templates: BAGPIPES'
# bundled BC03+MILES Kroupa SSP grid is ported into the DSPS HDF5
# layout `tengri.load_ssp_data` reads (`_drivers/bagpipes_ssp_to_dsps.py`).
# Any §1 residual below floating-point precision is interpolation
# alone.
#
# The fiducial galaxy throughout the SED panels: a τ-delayed star
# formation history with τ = 1 Gyr formed over 5 Gyr; Z = Z☉; a
# Calzetti+2000 attenuation law at A_V = 1; and Draine & Li (2007)
# IR re-emission at (q_PAH, U_min, γ) = (2.5, 1.0, 0.05). Each
# section then sweeps one physics block around this fiducial so the
# disagreement attributable to that block can be read off the figure.
#
# **What to expect.** Stellar templates, star-formation histories, dust
# attenuation, the Draine & Li (2007) dust IR, and the Inoue+2014 IGM
# reproduce BAGPIPES to floating-point or to a few percent at matched
# parameters. (The dust IR shape agreement relies on the DL07 PDR
# luminosity weighting — without it the warm component is
# ~14× under-weighted and the IR comes out spuriously cold.) The nebular
# block is the principal exception by construction: BAGPIPES uses Cloudy
# v25 grids embedded in the stellar population synthesis chain, while
# tengri uses Cue
# (Li et al. 2025), a neural emulator trained on Cloudy v17. The
# resulting Hα ratio is quantified in §9.
#
# BAGPIPES does not provide AGN, X-ray, or radio components, so those
# sections are deliberately omitted here. The CIGALE reproduction
# notebook covers the panchromatic AGN/X-ray/radio stack.

# %% [markdown]
# ## Setup

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reproduction.bagpipes._drivers import bagpipes_driver as B, units as U

import tengri
from tengri import FIXED, Fixed, SEDModel, load_ssp_data

# Force the inline backend so figures embed on (re-)render regardless of the
# ambient MPLBACKEND. A non-inline backend (e.g. Agg) drops the save_fig()
# auto-display and produces a figure-less notebook. No-op when run as a script.
try:  # noqa: SIM105
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

warnings.filterwarnings("ignore")
tengri.plot.setup_style()

# Unit-sanity guard: BAGPIPES' ``spectrum_full`` is in erg/s/Å at z=0.
# Every panel below claims percent-level agreement, which rests on the
# erg/s/Å → erg/s/Hz converter in ``_drivers/units.py``. A factor-of-c
# bug there would silently misshape every comparison. Assert the
# bolometric round-trip here so the entire notebook trips at Setup if
# the converter ever drifts.
_unit_check = U.verify_unit_conversion(rtol=1e-3)
print(
    f"unit-conversion bolometric round-trip: "
    f"rel_err = {_unit_check['rel_err']:.2e}  (target < 1e-3)"
)

# Metallicity pin — BAGPIPES `metallicity=1.0` is Z/Z_⊙ = 1; the bundled
# grid uses Z_⊙ = 10**(-1.848) (Asplund+2009) at HDU `ZMET_1.000ZSOL`.
# tengri's `met_logzsol = log10(Z/Z_⊙) = 0` is the bit-aligned counterpart.
LOG10_ZSUN = -1.848
MET_LOGZSOL = 0.0
STELLAR_FIDUCIAL = {"logzsol": Fixed(MET_LOGZSOL), "*": FIXED}

# Notebook-vs-script compatible: ``__file__`` is undefined when this is
# run via nbclient (the kernel's resources path is set to the
# reproduction/bagpipes/ directory instead), so fall back to the CWD.
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
# BAGPIPES' BC03+MILES Kroupa templates re-shaped into the DSPS HDF5
# layout that tengri reads — same numerical SSPs on both sides.

# %%
ssp_file = _HERE / "_drivers" / "data" / "bc03_miles_from_bagpipes.h5"
if not ssp_file.is_file():
    raise SystemExit(
        f"SSP grid {ssp_file} is missing. Generate it once with:\n"
        f"    python -m reproduction.bagpipes._drivers.bagpipes_ssp_to_dsps"
    )
ssp = load_ssp_data(str(ssp_file.resolve()))
print(
    f"BC03+MILES Kroupa SSP: {ssp.ssp_wave.shape[0]} wavelengths, "
    f"{ssp.ssp_lgmet.shape[0]} metallicities, "
    f"{ssp.ssp_lg_age_gyr.shape[0]} age bins."
)


# %% [markdown]
# ## §1 Stellar populations
#
# BC03+MILES Kroupa (Bruzual & Charlot 2003; Sánchez-Blázquez et al.
# 2006; Kroupa 2001) at Z = Z⊙ from 1 Myr to 10 Gyr. **Single SSPs**,
# overlaid: BAGPIPES' raw `bc03_miles_stellar_grids.fits` (solid) read
# directly with no SFH module, against the same templates re-shaped
# into tengri's HDF5 (dashed). The curves sit on top of each other; the
# lower panel shows the relative residual `|tengri − BAGPIPES| /
# BAGPIPES`, ~1e-7 from float32 round-trip through the HDF5 port — both
# codes consume identical numerics.

# %%
from astropy.io import fits as _fits

ages_yr = [1e6, 1e7, 1e8, 1e9, 1e10]
L_SUN = 3.826e33  # erg/s — match the value bagpipes hard-codes
_C_AA = 2.998e18  # speed of light [Å/s]

_grid_path = Path(B.__file__).resolve().parent / "data"  # not used directly
_bagpipes_grid_dir = Path(__import__("bagpipes").config.grid_dir)
_fits_file = _bagpipes_grid_dir / "bc03_miles_stellar_grids.fits"
with _fits.open(_fits_file) as hdul:
    _wave_aa = np.asarray(hdul["WAVELENGTHS_AA"].data, dtype=np.float64)
    _age_yr_native = np.asarray(hdul["STELLAR_AGE_YR"].data, dtype=np.float64)
    _flux_zsol_aa = np.asarray(hdul["ZMET_1.000ZSOL"].data, dtype=np.float64)

bagpipes_ssp = []
for age_yr in ages_yr:
    ia = int(np.argmin(np.abs(_age_yr_native - age_yr)))
    # Lsun/Å/Msun → erg/s/Hz/Msun: × λ²/c × L_sun
    lnu = _flux_zsol_aa[ia] * _wave_aa**2 / _C_AA * L_SUN
    bagpipes_ssp.append((_wave_aa, lnu))

i_zsun = int(np.argmin(np.abs(ssp.ssp_lgmet - LOG10_ZSUN)))
tengri_ssp = []
for age_yr in ages_yr:
    i_age = int(np.argmin(np.abs(ssp.ssp_lg_age_gyr - np.log10(age_yr / 1e9))))
    tengri_ssp.append((ssp.ssp_wave, ssp.ssp_flux[i_zsun, i_age, :] * L_SUN))

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
colors = plt.cm.viridis(np.linspace(0, 1, len(ages_yr)))
for color, age_yr, (w_b, L_b), (w_t, L_t) in zip(colors, ages_yr, bagpipes_ssp, tengri_ssp):
    label = f"{age_yr / 1e6:g} Myr"
    ax.plot(w_b, L_b, color=color, linewidth=2.0, label=label)
    ax.plot(w_t, L_t, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
    L_t_on_b = U.regrid(w_t, L_t, w_b)
    resid = np.abs(L_t_on_b - L_b) / np.maximum(np.abs(L_b), 1e-30)
    resid[~np.isfinite(resid)] = 0.0
    ax_r.plot(w_b, resid, color=color, linewidth=1.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e6)
ax.set_ylim(1e16, 5e24)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz / $M_\odot$]")
ax.set_title("BC03+MILES Kroupa Z = Z⊙ — BAGPIPES (solid) vs tengri (black dashed)")
ax.legend(fontsize=9, title="SSP age")
ax.grid(True, alpha=0.3)
ax_r.set_xscale("log")
ax_r.set_yscale("log")
ax_r.set_xlabel(r"$\lambda$ [Å]")
ax_r.set_ylabel(r"$|\Delta| / L_{\rm BAGPIPES}$", fontsize=9)
ax_r.set_ylim(1e-9, 1e-2)
ax_r.axhline(1e-6, color="grey", linestyle=":", alpha=0.6, label="float32 round-trip floor")
ax_r.legend(loc="upper right", fontsize=8)
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("bagpipes_01_ssp_bc03_miles.png")


# %% [markdown]
# ## §2 Parametric star formation histories — delayed-τ
#
# BAGPIPES' `delayed` SFH is the same closed-form τ-delayed shape
# tengri uses in `sfh.delayed`: `SFR(t) ∝ t · exp(−t/τ)`, peaking at
# cosmic time `t = τ`. Both integrate to `10**massformed` M☉ formed by
# `age` — BAGPIPES via the `massformed` parameter, tengri via
# `log_total_mass`.
#
# **What the right panel actually plots.** Not a fine-grid analytic
# evaluation — that would be a comparison of two closed-form formulas
# rather than a test of tengri. Instead the panel reads
# `state.derived["sfr_history"]` off a built `SEDModel`, on the
# 256-point log-spaced lookback grid the SFH-convolution code actually
# uses. The stepping near small `t_cosmic` (large lookback) is real;
# every fit downstream sees this same grid. The printed
# `∫SFR dt = 1.0000 × 10¹⁰ M☉` check confirms the area integrates to
# `10**log_total_mass`, the only test that matters for downstream
# physics.

# %%
LOG_MASS_FIDUCIAL = 10.0
TAU_GYR_FIDUCIAL = 1.0
AGE_GYR_FIDUCIAL = 5.0

t_b, sfr_b = B.sfh_curve(
    sfh_type="delayed",
    age=AGE_GYR_FIDUCIAL,
    tau=TAU_GYR_FIDUCIAL,
    massformed=LOG_MASS_FIDUCIAL,
)
# BAGPIPES' sfh.ages is lookback time; convert to cosmic-age-since-onset
# for matched plotting.
t_b_cosmic_gyr = AGE_GYR_FIDUCIAL - t_b / 1e9
_keep = (t_b_cosmic_gyr >= 0) & (t_b_cosmic_gyr <= AGE_GYR_FIDUCIAL)
t_b_cosmic_gyr = t_b_cosmic_gyr[_keep]
sfr_b_keep = sfr_b[_keep]

_m_sfh = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(TAU_GYR_FIDUCIAL),
        "age_gyr": Fixed(AGE_GYR_FIDUCIAL),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
_state_sfh = _m_sfh.predict_state({})
_lbt_yr = np.asarray(_state_sfh.derived["sfh_grid_lbt_yr"])
_sfr_history = np.asarray(_state_sfh.derived["sfr_history"])
t_t = (AGE_GYR_FIDUCIAL - _lbt_yr / 1e9) * 1e9
_idx = np.argsort(t_t)
_mass_formed = float(np.trapezoid(_sfr_history[_idx], t_t[_idx]))
print(
    f"tengri pipeline ∫SFR dt = {_mass_formed:.4e} M☉  (target: 1.0000e+10 from log_total_mass=10)"
)

# BAGPIPES side mass check:
_idx_b = np.argsort(t_b_cosmic_gyr)
_mass_b = float(np.trapezoid(sfr_b_keep[_idx_b], t_b_cosmic_gyr[_idx_b] * 1e9))
print(f"BAGPIPES        ∫SFR dt = {_mass_b:.4e} M☉  (target: 1.0000e+10 from massformed=10)")

fig, ax_l, ax_r = U.two_panel_fig()
for ax, title in (
    (ax_l, "BAGPIPES delayed (τ=1 Gyr, age=5 Gyr, log M=10)"),
    (ax_r, "tengri pipeline sfr_history (256-pt log-lbt)"),
):
    ax.set_xlabel("Cosmic age since SF onset [Gyr]")
    ax.set_ylabel(r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]")
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
ax_l.plot(t_b_cosmic_gyr, sfr_b_keep, "C0-", linewidth=2.0)
ax_l.axvline(
    TAU_GYR_FIDUCIAL,
    color="grey",
    linestyle=":",
    alpha=0.6,
    label=rf"$\tau$ = {TAU_GYR_FIDUCIAL:g} Gyr",
)
ax_l.legend(fontsize=9)
ax_r.plot(t_t / 1e9, _sfr_history, "C1-", linewidth=2.0)
ax_r.axvline(TAU_GYR_FIDUCIAL, color="grey", linestyle=":", alpha=0.6)
ax_r.set_yscale("linear")
ax_l.set_yscale("linear")
fig.tight_layout()
save_fig("bagpipes_02_sfh_delayed.png")


# %% [markdown]
# ## §2 cont'd — double power-law
#
# BAGPIPES' `dblplaw` is the workhorse SFH shape for quiescent-galaxy
# fitting at JWST cosmic noon — a smooth rise + smooth fall, two slopes
# `α` (falling) and `β` (rising), turnover time `τ`:
# :math:`\\mathrm{SFR}(T) \\propto \\bigl[(T/\\tau)^{\\alpha} +
# (T/\\tau)^{-\\beta}\\bigr]^{-1}`. tengri's `dpl` is the same closed-form
# shape with the same `(α, β, τ)` parameterisation.
#
# **Time frame.** Both codes measure the shape in *cosmic time since
# formation* :math:`T`, peaking near
# :math:`T = \\tau\\,(\\beta/\\alpha)^{1/(\\alpha+\\beta)}`. tengri's `dpl`
# takes an explicit `age_gyr` anchor and converts internally,
# :math:`T = \\mathrm{age} - t_\\mathrm{lookback}` — the same way
# `sfhdelayed` does. Feeding it the BAGPIPES age of the universe at the
# source redshift reproduces the BAGPIPES curve directly, with
# no axis flip and no coordinate-system conversion. The two
# panels below overlay the *same* shape.

# %%
DPL_ALPHA = 1.5
DPL_BETA = 1.0
DPL_TAU_GYR = 3.0

# Build the bagpipes side directly via model_galaxy.sfh inspection.
_comp_b_dpl = {
    "redshift": 0.0,
    "dblplaw": {
        "alpha": DPL_ALPHA,
        "beta": DPL_BETA,
        "tau": DPL_TAU_GYR,
        "metallicity": 1.0,
        "massformed": LOG_MASS_FIDUCIAL,
    },
}
_mg_b_dpl = B._build_model(_comp_b_dpl)
t_b_dpl = np.asarray(_mg_b_dpl.sfh.ages, dtype=np.float64)  # yr lookback
sfr_b_dpl = np.asarray(_mg_b_dpl.sfh.sfh, dtype=np.float64)  # Msun/yr

# Anchor tengri's cosmic-time frame to the *same* age of the universe
# BAGPIPES used (Planck cosmology at z = 0). Reading it off the model
# keeps the two codes on one clock regardless of cosmology drift.
AGE_OF_UNIVERSE_GYR = float(_mg_b_dpl.sfh.age_of_universe) / 1e9

m_dpl = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "dpl",
        "alpha": Fixed(DPL_ALPHA),
        "beta": Fixed(DPL_BETA),
        "tau_gyr": Fixed(DPL_TAU_GYR),
        "age_gyr": Fixed(AGE_OF_UNIVERSE_GYR),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_dpl = m_dpl.predict_state({})
_lbt_dpl = np.asarray(s_dpl.derived["sfh_grid_lbt_yr"])
_sfr_dpl = np.asarray(s_dpl.derived["sfr_history"])
_mass_b_dpl = float(np.trapezoid(sfr_b_dpl[::-1], t_b_dpl[::-1]))
_mass_t_dpl = float(np.trapezoid(_sfr_dpl[np.argsort(_lbt_dpl)], _lbt_dpl[np.argsort(_lbt_dpl)]))
print(
    f"§2a ∫SFR dt: BAGPIPES = {abs(_mass_b_dpl):.3e} M☉, "
    f"tengri = {_mass_t_dpl:.3e} M☉ (target 1.0e+10)"
)

fig, ax_l, ax_r = U.two_panel_fig()
for ax, title in (
    (ax_l, f"BAGPIPES dblplaw (α={DPL_ALPHA:g}, β={DPL_BETA:g}, τ={DPL_TAU_GYR:g} Gyr)"),
    (ax_r, f"tengri dpl (age = {AGE_OF_UNIVERSE_GYR:.2f} Gyr)"),
):
    ax.set_xlabel("lookback time [Gyr]")
    ax.set_xlim(0, 13.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11)
ax_l.set_ylabel(r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]")
ax_l.plot(t_b_dpl / 1e9, sfr_b_dpl, "C0-", linewidth=2.0)
ax_r.plot(_lbt_dpl / 1e9, _sfr_dpl, "C1-", linewidth=2.0)
_peak_lbt_dpl = float(_lbt_dpl[np.argmax(_sfr_dpl)] / 1e9)
for ax in (ax_l, ax_r):
    ax.axvline(
        _peak_lbt_dpl,
        color="grey",
        linestyle=":",
        alpha=0.6,
        label=f"peak @ {_peak_lbt_dpl:.1f} Gyr lookback",
    )
    ax.legend(fontsize=9)
fig.tight_layout()
save_fig("bagpipes_10_sfh_dblplaw.png")

print(
    f"§2a peak location: BAGPIPES @ lookback {t_b_dpl[np.argmax(sfr_b_dpl)] / 1e9:.2f} Gyr, "
    f"tengri dpl @ {_peak_lbt_dpl:.2f} Gyr (same cosmic-time frame; "
    f"residual is SFH-grid discretisation)"
)


# %% [markdown]
# ## §2 cont'd — lognormal
#
# Another BAGPIPES standard, popular for "rejuvenation" tests: a
# lognormal SFR(t) peaked at `tmax` with full-width-half-max `fwhm`.
# tengri's `lnorm` is the same family with a slightly different
# parameterisation: peak in cosmic time `peak_gyr` and log-space width
# `width_gyr` (dex). To match BAGPIPES' linear-time FWHM we convert:
# `width_dex ≈ FWHM/(2.355 × tmax × ln 10)` for narrow bursts.
#
# **Same time frame as §2a.** Like `dpl`, tengri's `lnorm` takes
# an explicit `age_gyr` anchor and evaluates the shape in cosmic time
# since formation, `T = age − lookback`, so `peak_gyr` is the cosmic-age
# peak `tmax` — the same direction as BAGPIPES. Feeding the BAGPIPES age
# of the universe reproduces the BAGPIPES curve directly.
#
# **Caveat — shape fidelity.** tengri's `lnorm` is a log10-space Gaussian,
# not the exact Carnall+2018 1/T ln-space lognormal (which carries a 1/T
# Jacobian and solves (t0, σ) from (tmax, fwhm)). The detailed wing shape
# differs slightly from the exact Carnall parameterisation; the functional
# form is a deliberate trade-off prioritising simplicity and numerical
# stability.

# %%
LN_TMAX_GYR = 4.0
LN_FWHM_GYR = 2.0
LN_WIDTH_DEX = LN_FWHM_GYR / (2.355 * LN_TMAX_GYR * np.log(10))

_comp_b_ln = {
    "redshift": 0.0,
    "lognormal": {
        "tmax": LN_TMAX_GYR,
        "fwhm": LN_FWHM_GYR,
        "metallicity": 1.0,
        "massformed": LOG_MASS_FIDUCIAL,
    },
}
_mg_b_ln = B._build_model(_comp_b_ln)
t_b_ln = np.asarray(_mg_b_ln.sfh.ages, dtype=np.float64)
sfr_b_ln = np.asarray(_mg_b_ln.sfh.sfh, dtype=np.float64)

m_ln = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "lnorm",
        "peak_gyr": Fixed(LN_TMAX_GYR),
        "width_gyr": Fixed(LN_WIDTH_DEX),
        "age_gyr": Fixed(AGE_OF_UNIVERSE_GYR),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_ln = m_ln.predict_state({})
_lbt_ln = np.asarray(s_ln.derived["sfh_grid_lbt_yr"])
_sfr_ln = np.asarray(s_ln.derived["sfr_history"])

fig, ax_l, ax_r = U.two_panel_fig()
for ax, title in (
    (ax_l, f"BAGPIPES lognormal (tmax={LN_TMAX_GYR:g}, FWHM={LN_FWHM_GYR:g} Gyr)"),
    (ax_r, f"tengri lnorm (peak={LN_TMAX_GYR:g} Gyr, age={AGE_OF_UNIVERSE_GYR:.2f} Gyr)"),
):
    ax.set_xlabel("lookback time [Gyr]")
    ax.set_xlim(0, 13.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=11)
ax_l.set_ylabel(r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]")
ax_l.plot(t_b_ln / 1e9, sfr_b_ln, "C0-", linewidth=2.0)
ax_r.plot(_lbt_ln / 1e9, _sfr_ln, "C1-", linewidth=2.0)
_peak_lbt_ln = float(_lbt_ln[np.argmax(_sfr_ln)] / 1e9)
for ax in (ax_l, ax_r):
    ax.axvline(
        _peak_lbt_ln,
        color="grey",
        linestyle=":",
        alpha=0.6,
        label=f"peak @ {_peak_lbt_ln:.1f} Gyr lookback",
    )
    ax.legend(fontsize=9)
fig.tight_layout()
save_fig("bagpipes_11_sfh_lognormal.png")

print(
    f"§2b peak location: BAGPIPES @ lookback {t_b_ln[np.argmax(sfr_b_ln)] / 1e9:.2f} Gyr, "
    f"tengri lnorm @ {_peak_lbt_ln:.2f} Gyr (same cosmic-time frame; "
    f"residual is grid discretisation + log10-vs-Carnall wing shape)"
)


# %% [markdown]
# ## §3 Non-parametric continuity SFH (Leja+2019)
#
# BAGPIPES' workhorse non-parametric SFH (see the Carnall+ "Further
# Examples 2" notebook, *The Leja2019 non-parametric continuity SFH
# model*) is the **piecewise-constant SFR with log-ratios between
# adjacent bins**, with a Student-t prior on the ratios pushing
# adjacent bins toward equality. tengri ships the same shape under
# `sfh.type="continuity"` with the same parametrisation.
#
# Both codes operate on a 7-bin grid by default:
#
# ```
# bin edges (Gyr lookback): [0, 0.03, 0.1, 0.3, 1, 3, 6, 13.7]
# free params:              log-ratio between bin i and bin i+1 (×6)
# prior on each ratio:      StudentT(μ=0, σ=0.3, df=2)
# ```
#
# **Convention warning.** BAGPIPES indexes `dsfr_i` from OLDEST to
# YOUNGEST (Leja+2019 paper convention): `dsfr_1 = log10(SFR_bin1 /
# SFR_bin2)` where bin 1 is the oldest. tengri's `ratio_i` indexes from
# YOUNGEST to OLDEST. Setting the same ratio array on both sides
# therefore produces **time-reversed** SFR(t) — same physics, different
# parameter ordering. The panel below uses three configurations with
# the **arrays reversed on the BAGPIPES side** so both panels show the
# same SFH shape.

# %%
# Bin edges shared between codes. Both want them in **increasing**
# order (lookback time from 0 to age of universe). BAGPIPES expects
# Myr; tengri expects Gyr.
_BIN_EDGES_GYR = [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 13.7]
_BIN_EDGES_MYR_ASC = [e * 1e3 for e in _BIN_EDGES_GYR]

_cases = [
    ("flat", [0.0] * 6),
    ("recent burst", [0.0, 0.0, 0.0, 0.0, +0.6, +0.6]),
    ("quenched", [+0.5, +0.5, 0.0, -0.5, -0.5, -0.5]),
]

fig, axes = plt.subplots(len(_cases), 2, figsize=(13, 8), sharex=True)
for row, (label, ratios) in enumerate(_cases):
    # BAGPIPES side
    comp_b = {
        "redshift": 0.0,
        "continuity": {
            "metallicity": 1.0,
            "massformed": LOG_MASS_FIDUCIAL,
            "bin_edges": _BIN_EDGES_MYR_ASC,
            # BAGPIPES indexes dsfr_i from OLDEST to YOUNGEST — reverse
            # the array so the SFR(t) shape matches tengri's young-first
            # convention.
            **{f"dsfr{i + 1}": ratios[5 - i] for i in range(6)},
        },
    }
    mg_b_c = B._build_model(comp_b)
    t_b_c = np.asarray(mg_b_c.sfh.ages, dtype=np.float64)
    sfr_b_c = np.asarray(mg_b_c.sfh.sfh, dtype=np.float64)

    # tengri side
    m_c = SEDModel.build(
        ssp_data=ssp,
        stellar=STELLAR_FIDUCIAL,
        sfh={
            "type": "continuity",
            "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
            **{f"ratio_{i}": Fixed(ratios[i]) for i in range(6)},
            "*": FIXED,
        },
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.0),
    )
    s_c = m_c.predict_state({})
    lbt_t_c = np.asarray(s_c.derived["sfh_grid_lbt_yr"])
    sfr_t_c = np.asarray(s_c.derived["sfr_history"])

    ax_l, ax_r = axes[row]
    ax_l.plot(t_b_c / 1e9, sfr_b_c, "C0-", linewidth=2.0)
    ax_r.plot(lbt_t_c / 1e9, sfr_t_c, "C1-", linewidth=2.0)
    for ax in (ax_l, ax_r):
        ax.set_xlim(0, 13.7)
        ax.set_yscale("log")
        ax.set_ylim(1e-1, 1e2)
        ax.set_ylabel(r"SFR [$M_\odot/\mathrm{yr}$]")
        ax.grid(True, alpha=0.3)
        for edge in _BIN_EDGES_GYR:
            ax.axvline(edge, color="grey", linestyle=":", alpha=0.25)
    ax_l.set_title(f"BAGPIPES continuity — {label}")
    ax_r.set_title(f"tengri continuity — {label}")

for ax in axes[-1]:
    ax.set_xlabel("lookback time [Gyr]")

fig.tight_layout()
save_fig("bagpipes_17_sfh_continuity_leja.png")


# %% [markdown]
# ## §4 Integrated stellar SED
#
# Convolve the τ-delayed SFH with the BC03+MILES Kroupa SSPs. No dust,
# no nebular. Both panels show `L_ν` vs `λ_rest`; BAGPIPES is normalised
# to `10**massformed` M☉ formed by construction, tengri's stellar mass
# formed is reported in the annotation.

# %%
w_b, L_b = B.stellar_only_lnu(
    massformed=LOG_MASS_FIDUCIAL,
    metallicity=1.0,
    age_max=AGE_GYR_FIDUCIAL,
    tau=TAU_GYR_FIDUCIAL,
    sfh_type="delayed",
)

m_stellar = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(TAU_GYR_FIDUCIAL),
        "age_gyr": Fixed(AGE_GYR_FIDUCIAL),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_stellar = m_stellar.predict_state({})
_assert_comparable(L_b, s_stellar.sed_intrinsic, name="§3 stellar")

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(
    ax_l,
    ax_r,
    label_l="BAGPIPES  delayed + BC03+MILES",
    label_r="tengri  sfh.delayed + BC03+MILES",
)
ax_l.plot(w_b, L_b, "C0-", linewidth=1.5)
ax_l.text(
    0.05,
    0.95,
    rf"$M_\star = 10^{{{LOG_MASS_FIDUCIAL:.0f}}}\,M_\odot$ (norm)",
    transform=ax_l.transAxes,
    fontsize=10,
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
ax_r.plot(s_stellar.wave, s_stellar.sed_intrinsic, "C1-", linewidth=1.5)
m_star = 10.0 ** float(s_stellar.derived["log_mstar"])
ax_r.text(
    0.05,
    0.95,
    rf"$M_\star = {m_star:.2e}\,M_\odot$",
    transform=ax_r.transAxes,
    fontsize=10,
    va="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
)
# Frame y to the visible SED. Without this, matplotlib autoscales over the
# full SSP grid (down to ~5 Å, L_nu ~ 1e8) and the panel spans ~21 dead
# decades; the stellar continuum only occupies the top ~6.
_ypk = float(max(np.max(L_b), np.max(np.asarray(s_stellar.sed_intrinsic))))
for ax in (ax_l, ax_r):
    ax.set_xlim(1e2, 1e6)
    ax.set_ylim(_ypk / 1e6, _ypk * 2.0)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("bagpipes_03_stellar_sed.png")

# Median tengri/BAGPIPES ratio in the optical (3000–10000 Å), a useful
# scalar diagnostic for the docs page.
_mask_opt = (w_b >= 3000) & (w_b <= 10000)
_t_on_b = U.regrid(s_stellar.wave, np.asarray(s_stellar.sed_intrinsic), w_b)
_ratios = _t_on_b[_mask_opt] / L_b[_mask_opt]
_ratios = _ratios[np.isfinite(_ratios) & (_ratios > 0)]
if _ratios.size:
    print(
        f"§4 stellar SED tengri/BAGPIPES optical (3000–10000 Å): "
        f"median {np.median(_ratios):.3f}, "
        f"P5 {np.percentile(_ratios, 5):.3f}, "
        f"P95 {np.percentile(_ratios, 95):.3f}"
    )


# %% [markdown]
# ## §5 Metallicity sensitivity (chemical enrichment, single-Z form)
#
# BAGPIPES exposes `metallicity` (Z / Z☉) on every SFH block; tengri
# carries the same knob via `logzsol = log10(Z / Z☉)`. Both codes also
# support time-varying Z (BAGPIPES `metallicity_bins`; tengri's
# `chemical_enrichment_history`), but the simplest reproducible test
# is the single-Z response: sweep `Z ∈ {0.2, 1.0, 2.5} Z☉` at the
# fiducial 5 Gyr delayed-τ SFH and overlay the optical-NIR stellar
# continuum. Both codes should track the standard
# age-metallicity-degeneracy direction: high-Z → redder + deeper
# absorption features.

# %%
_Z_VALUES = [0.2, 1.0, 2.5]
_logzsol_values = [float(np.log10(z)) for z in _Z_VALUES]

fig, (ax_b, ax_t) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
ax_b.set_title("BAGPIPES — Z sweep at 5 Gyr delayed-τ")
ax_t.set_title("tengri — Z sweep at 5 Gyr delayed-τ")
for ax in (ax_b, ax_t):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda_{\rm rest}$ [Å]")
    ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
    ax.set_xlim(2e3, 2e4)
    ax.set_ylim(2e27, 8e29)
    ax.grid(True, alpha=0.3)

_colors = plt.cm.plasma(np.linspace(0.15, 0.85, len(_Z_VALUES)))
for color, z, logz in zip(_colors, _Z_VALUES, _logzsol_values):
    w_b_z, L_b_z = B.stellar_only_lnu(
        massformed=LOG_MASS_FIDUCIAL,
        metallicity=z,
        age_max=AGE_GYR_FIDUCIAL,
        tau=TAU_GYR_FIDUCIAL,
        sfh_type="delayed",
    )
    ax_b.plot(w_b_z, L_b_z, color=color, linewidth=1.7, label=f"Z = {z:g} Z⊙")
    m_z = SEDModel.build(
        ssp_data=ssp,
        stellar={"logzsol": Fixed(logz), "*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(TAU_GYR_FIDUCIAL),
            "age_gyr": Fixed(AGE_GYR_FIDUCIAL),
            "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
            "*": FIXED,
        },
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.0),
    )
    s_z = m_z.predict_state({})
    ax_t.plot(s_z.wave, s_z.sed_intrinsic, color=color, linewidth=1.7, label=f"Z = {z:g} Z⊙")

ax_b.legend(fontsize=10)
ax_t.legend(fontsize=10)
fig.tight_layout()
save_fig("bagpipes_15_metallicity_sweep.png")


# %% [markdown]
# ## §6 Dust attenuation curves
#
# BAGPIPES' bundled dust laws — Calzetti+2000, Cardelli+1989 (MW),
# Charlot & Fall 2000, and the Salim+2018 modification — shown against
# tengri's `calzetti`, `cardelli`, `noll09`, and `salim`. Both sides
# evaluate the analytic law directly (tengri via `tengri.dust.list_laws`),
# normalised to `A(λ)/A_V` at 5500 Å, so the comparison is curve against
# curve with no SSP-convolution noise.

# %%
from tengri.dust import list_laws

# (BAGPIPES dust block, tengri law, label) for the four matched laws.
_law_pairs = [
    ({"type": "Calzetti", "Av": 1.0}, "calzetti", "Calzetti+2000"),
    ({"type": "Cardelli", "Av": 1.0}, "cardelli", "Cardelli+1989 (MW)"),
    ({"type": "CF00", "Av": 1.0, "eta": 2.0, "n": -0.7}, "noll09", "Charlot & Fall 2000"),
    ({"type": "Salim", "Av": 1.0, "delta": 0.0, "B": 0.0}, "salim", "Salim+2018 (δ=0)"),
]
_tengri_laws = list_laws(headline=False)  # {name: fn(wave_aa) -> k at tau_V=1}
wave_law = np.logspace(np.log10(1000.0), np.log10(50000.0), 2000)


def _norm_AV(wave, A):
    """A(λ) normalised to A_V at 5500 Å."""
    return A / A[np.argmin(np.abs(wave - 5500.0))]


fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, title in (
    (ax_l, "bagpipes.model_galaxy attenuation laws"),
    (ax_r, "tengri attenuation laws"),
):
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\lambda$ [Å]")
    ax.set_xlim(1e3, 5e4)
    ax.set_ylim(0.05, 20)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
ax_l.set_ylabel(r"$A_\lambda / A_V$")

for dust_block, tengri_law, label in _law_pairs:
    try:
        w_b, A_b = B.attenuation_curve(dust_block)
        ax_l.plot(w_b, _norm_AV(w_b, A_b), linewidth=2.0, label=label)
    except Exception as exc:
        print(f"  skip BAGPIPES {label!r}: {exc}")

    # tengri's law functions are JAX-native but accept array-likes; the
    # result is wrapped back to NumPy for plotting.
    A_t = np.asarray(_tengri_laws[tengri_law](wave_law))
    ax_r.plot(wave_law, _norm_AV(wave_law, A_t), linewidth=2.0, label=label)
ax_l.legend(fontsize=10)
ax_r.legend(fontsize=10)
fig.tight_layout()
save_fig("bagpipes_04_dust_attenuation.png")


# %% [markdown]
# ## §7 Dust attenuation applied
#
# Fiducial galaxy with and without attenuation. BAGPIPES applies the
# Calzetti law as a single screen at `Av = 1.0` (with `eta = 1`, young
# and old stars see the same Av). tengri matches that by putting the full
# Av on the diffuse component — which attenuates all ages equally — and
# zeroing the birth-cloud term: tengri's `τ_bc` is age-gated to stars
# younger than ~10 Myr, so the single-screen equivalent is
# `τ_diff = Av/1.086`, `τ_bc = 0`. This configuration ensures that the
# old-stellar population dominating the 5 Gyr fiducial receives proper
# attenuation matching the BAGPIPES single-screen treatment.

# %%
AV_FIDUCIAL = 1.0
TAU_DIFF = AV_FIDUCIAL / 1.086  # full single screen on the diffuse (all-age) component
TAU_BC = 0.0  # birth-cloud term off — tengri's τ_bc attenuates only < ~10 Myr stars

w_b_nd, L_b_nd = B.stellar_only_lnu(
    massformed=LOG_MASS_FIDUCIAL,
    metallicity=1.0,
    age_max=AGE_GYR_FIDUCIAL,
    tau=TAU_GYR_FIDUCIAL,
    sfh_type="delayed",
)
w_b_d, L_b_d = B.attenuated_lnu(
    dust_block={"type": "Calzetti", "Av": AV_FIDUCIAL},
    sfh_type="delayed",
    massformed=LOG_MASS_FIDUCIAL,
    metallicity=1.0,
    age=AGE_GYR_FIDUCIAL,
    tau=TAU_GYR_FIDUCIAL,
)

m_nd = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(TAU_GYR_FIDUCIAL),
        "age_gyr": Fixed(AGE_GYR_FIDUCIAL),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
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
        "tau_gyr": Fixed(TAU_GYR_FIDUCIAL),
        "age_gyr": Fixed(AGE_GYR_FIDUCIAL),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "*": FIXED,
    },
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "tau_bc": Fixed(TAU_BC),
        "tau_diff": Fixed(TAU_DIFF),
        "*": FIXED,
    },
    redshift=Fixed(0.0),
)
s_d = m_d.predict_state({})
_assert_comparable(L_b_d, s_d.derived["sed_dust_attenuated"], name="§5 dust applied")

fig, ((ax_l1, ax_r1), (ax_l2, ax_r2)) = plt.subplots(2, 2, sharey=True, figsize=(12, 8))
U.panel(ax_l1, ax_r1, label_l="BAGPIPES  intrinsic", label_r="tengri  intrinsic")
U.panel(
    ax_l2,
    ax_r2,
    label_l=rf"BAGPIPES  Calzetti  ($A_V = {AV_FIDUCIAL:g}$)",
    label_r=rf"tengri  single-screen Calzetti  ($\tau_V={TAU_DIFF:.2f}$)",
)
ax_l1.plot(w_b_nd, L_b_nd, "C0-", linewidth=1.5)
ax_r1.plot(s_nd.wave, s_nd.sed_intrinsic, "C1-", linewidth=1.5)
ax_l2.plot(w_b_d, L_b_d, "C0-", linewidth=1.5)
ax_r2.plot(s_d.wave, s_d.derived["sed_dust_attenuated"], "C1-", linewidth=1.5)
_ymax = float(np.asarray(s_nd.sed_intrinsic).max())
for ax in (ax_l1, ax_r1, ax_l2, ax_r2):
    ax.set_xlim(1e2, 5e4)
    ax.set_ylim(_ymax * 1e-6, _ymax * 2)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("bagpipes_05_dust_attenuation_applied.png")


# %% [markdown]
# ## §8 Dust IR re-emission and energy balance
#
# Absorbed stellar UV/optical reappears in the IR. BAGPIPES uses the
# Draine & Li (2007) template family parametrised by `(qpah, umin,
# gamma)`; tengri uses its own DL07 template grid and enforces energy
# balance, `L_IR_emitted ≡ L_absorbed`, to floating point — the residual
# is annotated on the right panel.
#
# At matched `(qpah, umin, gamma)` the two DL07 SEDs agree in **shape**
# as well as bolometrically: both peak near ~130 µm at `umin = 1` and the
# 30–100 µm and >100 µm bands track BAGPIPES to ~6 %. The agreement
# reflects a key physical detail: `gamma` is a dust-*mass* fraction, but
# the PDR dust emits `R = U_max ln(U_max/U_min) / (U_max − U_min) ≈ 14×`
# more per unit mass (DL07 Eq. 33), so a 5 % mass fraction carries
# ~40 % of the luminosity. Proper accounting of this PDR luminosity
# weighting is essential for shape agreement in the 30–100 µm bands.

# %%
QPAH_FIDUCIAL = 2.5
UMIN_FIDUCIAL = 1.0
GAMMA_FIDUCIAL = 0.05

# Bagpipes: dust + dust_emission triggered by including `dust` block
# with the energy-balance autoflag. We use the same Calzetti fiducial
# as §5, plus the DL07 emission block.
comp_b = {
    "redshift": 0.0,
    "delayed": {
        "metallicity": 1.0,
        "age": AGE_GYR_FIDUCIAL,
        "tau": TAU_GYR_FIDUCIAL,
        "massformed": LOG_MASS_FIDUCIAL,
    },
    "dust": {
        "type": "Calzetti",
        "Av": AV_FIDUCIAL,
        "eta": 1.0,
        "qpah": QPAH_FIDUCIAL,
        "umin": UMIN_FIDUCIAL,
        "gamma": GAMMA_FIDUCIAL,
    },
}
mg_b = B._build_model(comp_b)
w_b_ir, L_b_ir = B.to_lnu(mg_b)

m_ir = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(TAU_GYR_FIDUCIAL),
        "age_gyr": Fixed(AGE_GYR_FIDUCIAL),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "*": FIXED,
    },
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "tau_bc": Fixed(TAU_BC),
        "tau_diff": Fixed(TAU_DIFF),
        "emission": {
            "type": "draine_li2007",
            "qpah": Fixed(QPAH_FIDUCIAL),
            "umin": Fixed(UMIN_FIDUCIAL),
            "gamma_dl": Fixed(GAMMA_FIDUCIAL),
            "*": FIXED,
        },
        "*": FIXED,
    },
    redshift=Fixed(0.0),
)
s_ir = m_ir.predict_state({})

_L_abs = float(np.asarray(s_ir.derived["L_absorbed"]))
_L_ir = float(np.asarray(s_ir.derived["L_ir"]))
_eb_resid = abs(_L_ir - _L_abs) / max(_L_abs, 1e-30)
print(
    f"tengri energy balance: L_absorbed = {_L_abs:.3e}, "
    f"L_IR_emitted = {_L_ir:.3e}, resid = {_eb_resid:.2e}"
)

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(
    ax_l,
    ax_r,
    label_l="BAGPIPES  Calzetti + DL07",
    label_r="tengri  Calzetti + DL07 (energy-balanced)",
)
ax_l.plot(w_b_ir, L_b_ir, "C0-", linewidth=1.5)
sed_full_t = s_ir.derived["sed_dust_attenuated"] + s_ir.derived["sed_dust_ir"]
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
# Frame to the visible SED (stellar + dust IR); the fixed 1e24-1e32 window
# left ~3 empty decades below the continuum.
_ypk = float(max(np.max(L_b_ir), np.max(np.asarray(sed_full_t))))
for ax in (ax_l, ax_r):
    ax.set_xlim(1e3, 1e7)
    ax.set_ylim(_ypk / 1e5, _ypk * 2.0)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("bagpipes_06_dust_ir.png")


# %% [markdown]
# ## §9 Nebular emission
#
# BAGPIPES' bundled nebular grid is Cloudy v25
# (`bc03_miles_nebular_line_grids_extended_logU_nograins_cloudy25.fits`),
# parametrised by `(logU, metallicity)`. tengri's nebular emitter is
# Cue (Li et al. 2025), a neural-network emulator trained on Cloudy
# v17. **They will not agree.** The difference reflects three years
# of Cloudy development plus the different convolution paths (Cue
# operates on bare-stellar SSPs; BAGPIPES on the SFH-integrated
# spectrum). The panel quantifies the residual with the **integrated**
# line luminosity (continuum-subtracted, width-independent) rather than
# papering over it — a single-bin peak ratio would measure line width and
# grid resolution, not physics (the §10 LSF panel handles widths).

# %%
# Young 10 Myr constant-SFR fiducial — the regime where nebular
# lines dominate. Otherwise the Hα signal is buried under the
# evolved stellar continuum.
NEB_AGE = 0.01  # Gyr

# Give bagpipes a dense spec_wavs grid through the optical so its
# Cloudy-v25 lines aren't smeared into broad bumps by the coarse default
# 747-point grid (which spans 1 Å to 1e8 Å — far too sparse around Hα).
_neb_spec_wavs = np.arange(900.0, 7000.0, 1.0)

comp_b_neb_on = {
    "redshift": 0.0,
    "constant": {"metallicity": 1.0, "age_min": 0.0, "age_max": NEB_AGE, "massformed": 9.0},
    "nebular": {"logU": -2.0},
}
mg_b_neb_on = B._build_model(comp_b_neb_on, spec_wavs=_neb_spec_wavs)
# At redshift=0 with spec_wavs set, mg.spectrum carries erg/s/Å directly
# (no cosmological dimming applied). Column 0 is wavelength, column 1
# is L_λ.
w_b_neb = mg_b_neb_on.spectrum[:, 0]
L_b_neb_on = mg_b_neb_on.spectrum[:, 1] * w_b_neb**2 / U.C_ANGSTROM_PER_S

comp_b_neb_off = {
    "redshift": 0.0,
    "constant": {"metallicity": 1.0, "age_min": 0.0, "age_max": NEB_AGE, "massformed": 9.0},
}
mg_b_neb_off = B._build_model(comp_b_neb_off, spec_wavs=_neb_spec_wavs)
_L_stellar_only_aa = mg_b_neb_off.spectrum[:, 1]
L_b_stellar_only = _L_stellar_only_aa * w_b_neb**2 / U.C_ANGSTROM_PER_S
L_b_neb_alone = np.clip(L_b_neb_on - L_b_stellar_only, 0.0, None)

m_neb_on = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "const",
        "start_gyr": Fixed(NEB_AGE),
        "end_gyr": Fixed(0.0),
        "log_total_mass": Fixed(9.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    neb={"type": "cue", "neb_logU": Fixed(-2.0), "neb_logZ_gas": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_neb_on = m_neb_on.predict_state({})

m_neb_off = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "const",
        "start_gyr": Fixed(NEB_AGE),
        "end_gyr": Fixed(0.0),
        "log_total_mass": Fixed(9.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_neb_off = m_neb_off.predict_state({})
# Cue exposes its emission directly under derived["sed_nebular"];
# avoid subtracting two SEDs on different wave grids.
L_t_neb_alone = np.asarray(s_neb_on.derived["sed_nebular"])

fig, ax_l, ax_r = U.two_panel_fig(figsize=(13, 5))
U.panel(
    ax_l,
    ax_r,
    label_l="BAGPIPES  Cloudy v25 nebular (10 Myr CSF)",
    label_r="tengri  Cue v17 emulator (10 Myr CSF)",
)
ax_l.plot(w_b_neb, L_b_neb_alone, "C0-", linewidth=1.0)
ax_r.plot(s_neb_on.wave, L_t_neb_alone, "C1-", linewidth=1.0)
for ax in (ax_l, ax_r):
    ax.set_xlim(900, 7000)
    ax.set_xscale("linear")
    ax.grid(True, alpha=0.3)
# Quantify the residual by integrated line luminosity (width- and grid-
# independent). A single-bin peak ratio measures line width, not luminosity —
# Cue broadens its lines (see §10) while BAGPIPES' grid places them at its
# resolution, so a peak ratio is meaningless. §10 below addresses widths.
_w_t_neb = np.asarray(s_neb_on.wave)
print(
    "§9 integrated line luminosity (tengri Cue v17 / BAGPIPES Cloudy v25; "
    "residual = Cloudy v17→v25 + bare-stellar vs SFH-integrated path):"
)
for _c, _name in [(6563.0, "Hα"), (5007.0, "[O III]"), (4861.0, "Hβ")]:
    _lb = U.line_lum(w_b_neb, L_b_neb_alone, _c)
    _lt = U.line_lum(_w_t_neb, L_t_neb_alone, _c)
    if _lb > 0:
        print(
            f"    {_name} {_c:.0f} Å: BAGPIPES {_lb:.2e}, "
            f"tengri {_lt:.2e} erg/s → {_lt / _lb:.2f}×"
        )
fig.tight_layout()
save_fig("bagpipes_08_nebular.png")


# %% [markdown]
# ## §10 Line-spread function — velocity-broadening parity
#
# BAGPIPES' spectroscopy mode applies a Gaussian velocity broadening via
# the `veldisp` parameter (km/s); the kernel is built in log-wavelength
# (= velocity) space and convolved into the spectrum before fitting.
# tengri's equivalent is `tengri.observation.spectrum.velocity_broaden`,
# which JIT-compiles the same Gaussian convolution in log-λ space.
#
# At matched `veldisp`, the broadened Hα profile should have FWHM
# `2.355 σ_v λ / c`. This panel takes the §8 fiducial nebular SED
# (10 Myr CSF at `logU = −2`), runs each code's velocity-broadening at
# `veldisp = 150 km/s` — a typical late-type-galaxy value — and compares
# the result.
#
# A subtlety worth flagging. BAGPIPES' default internal spectral
# grid has `R_spec = 1000` (FWHM = c/R ≈ 300 km/s, σ ≈ 127 km/s),
# and the `veldisp` Gaussian convolves on top of it. The effective
# Hα width at `veldisp = 150 km/s` is therefore
# σ_eff = sqrt(127² + 150²) ≈ 197 km/s, FWHM ≈ 10 Å.
# tengri's `velocity_broaden` operates on the unbinned input
# spectrum and returns the pure-Gaussian profile at σ = 150 km/s
# (FWHM ≈ 7.7 Å). Both behaviours are correct; the comparison
# requires either oversampling the BAGPIPES grid or subtracting the
# baseline resolution from `veldisp` in quadrature.

# %%
VELDISP_KMS = 150.0
VELDISP_SPEC_WAVS = np.arange(6400.0, 6720.0, 0.5)

# BAGPIPES with veldisp on the same young-CSF fiducial used in §8.
comp_b_lsf = {
    "redshift": 0.0,
    "constant": {"metallicity": 1.0, "age_min": 0.0, "age_max": NEB_AGE, "massformed": 9.0},
    "nebular": {"logU": -2.0},
    "veldisp": VELDISP_KMS,
}
mg_b_lsf = B._build_model(comp_b_lsf, spec_wavs=VELDISP_SPEC_WAVS)
w_b_lsf = mg_b_lsf.spectrum[:, 0]
L_b_lsf_lambda = mg_b_lsf.spectrum[:, 1]  # erg/s/Å at z=0
L_b_lsf = L_b_lsf_lambda * w_b_lsf**2 / U.C_ANGSTROM_PER_S  # erg/s/Hz

# BAGPIPES without veldisp — same spectrum, no broadening, for reference.
comp_b_unb = dict(comp_b_lsf)
del comp_b_unb["veldisp"]
mg_b_unb = B._build_model(comp_b_unb, spec_wavs=VELDISP_SPEC_WAVS)
w_b_unb = mg_b_unb.spectrum[:, 0]
L_b_unb = mg_b_unb.spectrum[:, 1] * w_b_unb**2 / U.C_ANGSTROM_PER_S

# tengri side: take the §8 nebular SED, resample onto a uniform
# log-wavelength grid, apply velocity_broaden at the same sigma.
from tengri import velocity_broaden as _tng_broaden

_w_t_neb_orig = np.asarray(s_neb_on.wave)
_L_t_neb_orig = np.asarray(s_neb_on.derived["sed_nebular"])
# Restrict to the Hα window and resample to uniform log-λ for FFT.
_mask_halpha = (_w_t_neb_orig >= 6400) & (_w_t_neb_orig <= 6720)
_w_t_halpha = _w_t_neb_orig[_mask_halpha]
_L_t_halpha = _L_t_neb_orig[_mask_halpha]
# Uniform log-λ grid for velocity_broaden's FFT.
_n_uni = 4096
_w_t_uni = np.geomspace(_w_t_halpha[0], _w_t_halpha[-1], _n_uni)
_L_t_uni = np.interp(_w_t_uni, _w_t_halpha, _L_t_halpha)
L_t_lsf = np.asarray(_tng_broaden(_L_t_uni, _w_t_uni, VELDISP_KMS))

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
for ax, title in (
    (ax_l, f"BAGPIPES  veldisp = {VELDISP_KMS:.0f} km/s"),
    (ax_r, f"tengri  velocity_broaden(σ = {VELDISP_KMS:.0f} km/s)"),
):
    ax.set_xlabel(r"$\lambda$ [Å]")
    ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
    ax.set_title(title)
    ax.set_xlim(6400, 6720)
    ax.grid(True, alpha=0.3)
ax_l.plot(w_b_unb, L_b_unb, "lightgrey", linewidth=1.0, label="no broadening")
ax_l.plot(w_b_lsf, L_b_lsf, "C0-", linewidth=1.8, label="veldisp = 150 km/s")
ax_l.legend(fontsize=9)
ax_r.plot(_w_t_uni, _L_t_uni, "lightgrey", linewidth=1.0, label="no broadening")
ax_r.plot(_w_t_uni, L_t_lsf, "C1-", linewidth=1.8, label="velocity_broaden 150 km/s")
ax_r.legend(fontsize=9)
# Shared y-scale framed on the BROADENED lines so the two panels are directly
# comparable. The grey unbroadened references are delta-function spikes whose
# height is purely sampling-dependent; letting them drive independent autoscale
# made BAGPIPES (1.2e32) and tengri (4.4e32) look mismatched even though the
# broadened Hα profiles are the physical quantity of interest.
_ymax_vb = 1.15 * float(max(np.max(L_b_lsf), np.max(L_t_lsf)))
for ax in (ax_l, ax_r):
    ax.set_ylim(0, _ymax_vb)
fig.tight_layout()
save_fig("bagpipes_09_lsf_velbroaden.png")


# FWHM check at Hα: σ_v = 150 km/s ↔ FWHM_λ = 2.355 σ_v λ_Hα / c
_expected_fwhm = 2.355 * VELDISP_KMS / 2.998e5 * 6563.0
print(f"§10 expected Hα FWHM at σ_v = {VELDISP_KMS:g} km/s: {_expected_fwhm:.3f} Å")


def _fwhm(wave, spec, line_wave):
    """Half-power FWHM about a line centre, ignoring the local continuum."""
    idx = int(np.argmin(np.abs(wave - line_wave)))
    half = 0.5 * (spec[idx] + np.median(spec))
    above = spec > half
    if not above.any():
        return float("nan")
    lo = wave[np.argmax(above)]
    hi = wave[len(above) - 1 - np.argmax(above[::-1])]
    return float(hi - lo)


print(
    f"§10 measured Hα FWHM: BAGPIPES = {_fwhm(w_b_lsf, L_b_lsf, 6563.0):.3f} Å, "
    f"tengri = {_fwhm(_w_t_uni, L_t_lsf, 6563.0):.3f} Å"
)


# %% [markdown]
# ## §11 Panchromatic SED
#
# Stellar + nebular + dust attenuation + DL07 IR, on a single axis from
# the rest-UV to the far-IR. The percent-level disagreements visible in
# §3–§6 stack; the headline of this panel is the overall shape, not
# bit-for-bit agreement at any one wavelength.

# %%
comp_b_full = {
    "redshift": 0.0,
    "delayed": {
        "metallicity": 1.0,
        "age": AGE_GYR_FIDUCIAL,
        "tau": TAU_GYR_FIDUCIAL,
        "massformed": LOG_MASS_FIDUCIAL,
    },
    "dust": {
        "type": "Calzetti",
        "Av": AV_FIDUCIAL,
        "eta": 1.0,
        "qpah": QPAH_FIDUCIAL,
        "umin": UMIN_FIDUCIAL,
        "gamma": GAMMA_FIDUCIAL,
    },
    "nebular": {"logU": -2.0},
}
mg_b_full = B._build_model(comp_b_full)
w_b_full, L_b_full = B.to_lnu(mg_b_full)

m_full = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(TAU_GYR_FIDUCIAL),
        "age_gyr": Fixed(AGE_GYR_FIDUCIAL),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "*": FIXED,
    },
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "tau_bc": Fixed(TAU_BC),
        "tau_diff": Fixed(TAU_DIFF),
        "emission": {
            "type": "draine_li2007",
            "qpah": Fixed(QPAH_FIDUCIAL),
            "umin": Fixed(UMIN_FIDUCIAL),
            "gamma_dl": Fixed(GAMMA_FIDUCIAL),
            "*": FIXED,
        },
        "*": FIXED,
    },
    neb={"type": "cue", "neb_logU": Fixed(-2.0), "neb_logZ_gas": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_full = m_full.predict_state({})

fig, ax_l, ax_r = U.two_panel_fig(figsize=(13, 5))
U.panel(ax_l, ax_r, label_l="BAGPIPES  panchromatic", label_r="tengri  panchromatic")
ax_l.plot(w_b_full, L_b_full, "C0-", linewidth=1.5)
_sed_full_t = (
    np.asarray(s_full.derived["sed_dust_attenuated"])
    + np.asarray(s_full.derived["sed_dust_ir"])
    + np.asarray(s_full.derived["sed_nebular"])
)
ax_r.plot(s_full.wave, _sed_full_t, "C1-", linewidth=1.5)
for ax in (ax_l, ax_r):
    ax.set_xlim(1e2, 1e7)
    ax.set_ylim(1e22, 1e31)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("bagpipes_07_panchromatic.png")


# %% [markdown]
# ## §12 IGM transmission — Inoue14
#
# Both codes use the Inoue et al. (2014) IGM transmission tables.
# At the same redshift, with the same Lyman-series + DLA opacity
# coefficients, the two should agree to floating-point.

# %%
Z_FIDUCIAL_IGM = 4.0
w_b_igm, T_b_igm = B.igm_transmission(Z_FIDUCIAL_IGM)

# tengri side: evaluate igm.inoue14 at z=4 on the same rest-frame grid.
from tengri import igm_transmission as _tngigm

# tengri's IGM is parametrised on *observed*-frame wavelengths.
wave_obs = w_b_igm * (1.0 + Z_FIDUCIAL_IGM)
T_t_igm = np.asarray(_tngigm(wave_obs, np.asarray(Z_FIDUCIAL_IGM)))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(w_b_igm, T_b_igm, "C0-", linewidth=2.0, label=f"BAGPIPES Inoue14, z={Z_FIDUCIAL_IGM}")
ax.plot(w_b_igm, T_t_igm, "k--", linewidth=1.0, label=f"tengri Inoue14, z={Z_FIDUCIAL_IGM}")
ax.set_xlabel(r"rest-frame $\lambda$ [Å]")
ax.set_ylabel(r"IGM transmission $T(\lambda, z)$")
ax.set_xlim(800, 1300)
ax.set_ylim(0, 1.05)
ax.set_title(f"Inoue+2014 IGM transmission at z = {Z_FIDUCIAL_IGM}")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("bagpipes_12_igm_inoue14.png")

# Quantify agreement.
_igm_diff = np.abs(T_t_igm - T_b_igm)
print(
    f"§12 IGM Inoue14 at z={Z_FIDUCIAL_IGM}: "
    f"max |Δ| = {_igm_diff.max():.3e}, median |Δ| = {np.median(_igm_diff):.3e}"
)


# %% [markdown]
# ## §12 cont'd — Asada+2025 CGM damping wing (tengri-only)
#
# Inoue+2014 captures the *mean* intergalactic medium but does not
# include the damping-wing absorption from neutral hydrogen in the
# **circumgalactic** medium that's important at z > 5 (epoch of
# reionization). tengri ships an experimental Asada+2025 CGM damping-
# wing model (arXiv:2410.21543, accepted ApJL) in
# `tengri.components.igm.igm_transmission` via the `add_cgm=True`
# switch. BAGPIPES has no counterpart.
#
# This panel shows the CGM contribution at z = 7: tengri's Inoue14
# alone (solid) vs Inoue14 + Asada CGM damping wing (dashed). We use
# `cgm_log_nhi = 22.5` — the saturated-IGM regime expected at the
# epoch of reionization.
#
# tengri's implementation follows the full frequency-dependent
# Totani+2006 cross-section that Asada+2025 cite (Eq. 4 of
# Asada+2025):
#
# σ_α(ν) = (3λ²f Λ/8π) · Λ (ν/ν_α)⁴ / [4π²(ν−ν_α)² + Λ²(ν/ν_α)⁶/4]
#
# with the sigmoid evolution `N_HI(z) = 10**log_nhi / (1 + exp(-(z -
# z_mid)/dz))` matched to Asada+2025's calibration at z = 6–8. The
# damping wing kicks in immediately redward of Ly-α and decays over
# ~50 Å rest (~400 km/s), recovering to T = 1 by ~1280 Å rest.

# %%
from tengri import igm_transmission as _tngigm_with_cgm

Z_CGM = 7.0
# Two panels: full window (900–1260 Å rest) showing the IGM cliff at
# Lyα, then a zoom into 1210–1260 Å rest showing the redward damping
# wing shape — the only place where the Asada CGM separates from
# pure Inoue14 at high z (where Inoue14 already kills everything
# blueward).
wave_rest_full = np.linspace(900.0, 1260.0, 7001)
wave_rest_zoom = np.linspace(1215.5, 1280.0, 4001)

T_inoue_full = np.asarray(_tngigm_with_cgm(wave_rest_full * (1.0 + Z_CGM), np.asarray(Z_CGM)))
T_cgm_full = np.asarray(
    _tngigm_with_cgm(
        wave_rest_full * (1.0 + Z_CGM), np.asarray(Z_CGM), add_cgm=True, cgm_log_nhi=22.5
    )
)
T_inoue_zoom = np.asarray(_tngigm_with_cgm(wave_rest_zoom * (1.0 + Z_CGM), np.asarray(Z_CGM)))
T_cgm_zoom = np.asarray(
    _tngigm_with_cgm(
        wave_rest_zoom * (1.0 + Z_CGM), np.asarray(Z_CGM), add_cgm=True, cgm_log_nhi=22.5
    )
)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5))
for ax in (ax_l, ax_r):
    ax.axvline(1215.67, color="grey", linestyle=":", alpha=0.6, label=r"Ly$\alpha$ rest")
    ax.set_xlabel(r"rest-frame $\lambda$ [Å]")
    ax.grid(True, alpha=0.3)
ax_l.plot(wave_rest_full, T_inoue_full, "C0-", linewidth=2.0, label="Inoue14 (no CGM)")
ax_l.plot(wave_rest_full, T_cgm_full, "C1--", linewidth=2.0, label="Inoue14 + Asada CGM")
ax_l.set_ylabel(r"transmission $T(\lambda, z)$")
ax_l.set_xlim(900, 1260)
ax_l.set_ylim(0, 1.05)
ax_l.set_title(rf"Full window at $z = {Z_CGM:g}$ — IGM cuts off at Ly$\alpha$")
ax_l.legend(fontsize=10)
ax_r.plot(wave_rest_zoom, T_inoue_zoom, "C0-", linewidth=2.0, label="Inoue14 (no CGM)")
ax_r.plot(wave_rest_zoom, T_cgm_zoom, "C1--", linewidth=2.0, label="Inoue14 + Asada CGM")
ax_r.set_ylabel(r"transmission $T(\lambda, z)$")
ax_r.set_xlim(1215.5, 1280.0)
ax_r.set_ylim(0.0, 1.05)
ax_r.set_title(r"Redward zoom — Asada damping wing shape")
ax_r.legend(fontsize=10)
fig.tight_layout()
save_fig("bagpipes_16_cgm_asada.png")

# Diagnostic at +5 Å redward of Lyα (1220.67 Å rest), where the
# damping wing is at its peak effect on observable continuum.
_idx_red = np.argmin(np.abs(wave_rest_zoom - 1230.0))
print(
    f"§12b Asada CGM at z={Z_CGM:g}, λ_rest=1230 Å (14 Å redward of Ly-α): "
    f"T(no CGM) = {T_inoue_zoom[_idx_red]:.3f}, "
    f"T(with CGM) = {T_cgm_zoom[_idx_red]:.3f}, "
    f"τ_CGM ≈ {-np.log(max(T_cgm_zoom[_idx_red], 1e-10) / max(T_inoue_zoom[_idx_red], 1e-10)):.3f}"
)


# %% [markdown]
# ## §13 Photometry — SDSS ugriz AB magnitudes
#
# Integrating the panchromatic SED through a filter is the step that
# turns a model spectrum into observable data. BAGPIPES does this via
# `filt_list` + `model_galaxy.photometry`; tengri does it via
# `tengri.Photometry`. Both codes should produce the same AB
# magnitudes for the same input SED and the same filter curves.
#
# Here we share the **same filter set on both sides** (tengri's
# bundled SDSS curves) and convolve each code's §7 panchromatic SED
# through it, placing the source at the standard absolute-magnitude
# distance of 10 pc. Differences trace back to the §7 SED, not to the
# photometry integration — tengri and BAGPIPES use the same
# `∫ F_ν T dν / ∫ T dν` definition of band-averaged flux.
#
# **What the residual panel shows.** With the §7 dust as a single screen
# matching BAGPIPES (`τ_bc = 0`, full `A_V` on the diffuse component — see
# §7) and proper reddening of the nebular continuum through that screen,
# tengri reproduces the BAGPIPES SDSS magnitudes to **≤ 0.02 mag
# in r/i/z** but stays **−0.11 mag (u)** and **−0.15 mag (g)** brighter. The
# two bluest bands carry the strongest nebular emission lines — u spans
# [O II] 3727 and g spans Hβ 4862 + [O III] 4959/5007 — and that is exactly
# where the Cue-vs-Cloudy nebular difference (§9) manifests: §13b shows the
# gap collapses to ≤ 0.02 mag in every band once the nebular block is
# removed. The residual is a nebular **line-strength** difference between
# tengri's Cue emulator and BAGPIPES' Cloudy grid, not a dust or
# stellar-colour effect.

# %%
from tengri.observation.filters import load_filter

_sdss_bands = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
_filters = [load_filter(b) for b in _sdss_bands]

_d_10pc_cm = 10.0 * 3.086e18  # 10 pc in cm
_dimm = 4.0 * np.pi * _d_10pc_cm**2  # cm², F_ν = L_ν / dimm


def _ab_mag(wave_aa, L_nu_erg_per_hz, f_wave, f_trans):
    """Photon-counting AB magnitude at d = 10 pc."""
    F_nu = np.asarray(L_nu_erg_per_hz) / _dimm
    F_nu_at_filter = np.interp(f_wave, wave_aa, F_nu, left=0.0, right=0.0)
    weight = f_trans / f_wave
    F_band = np.trapezoid(F_nu_at_filter * weight, f_wave) / np.trapezoid(weight, f_wave)
    AB_ZP = 3.631e-20  # 3631 Jy in erg/s/cm²/Hz
    return -2.5 * np.log10(F_band / AB_ZP)


bp_mags = [_ab_mag(w_b_full, L_b_full, f.wave, f.trans) for f in _filters]
_L_t_full = (
    np.asarray(s_full.derived["sed_dust_attenuated"])
    + np.asarray(s_full.derived["sed_dust_ir"])
    + np.asarray(s_full.derived["sed_nebular"])
)
tng_mags = [_ab_mag(np.asarray(s_full.wave), _L_t_full, f.wave, f.trans) for f in _filters]

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
_pivot = np.array(
    [np.trapezoid(f.trans * f.wave, f.wave) / np.trapezoid(f.trans, f.wave) for f in _filters]
)
ax_top.plot(_pivot, bp_mags, "o-", color="C0", linewidth=1.7, label="BAGPIPES")
ax_top.plot(_pivot, tng_mags, "s--", color="C1", linewidth=1.7, label="tengri")
ax_top.invert_yaxis()
ax_top.set_ylabel(r"AB magnitude (10 pc, $M_\star = 10^{10}\,M_\odot$)")
ax_top.set_title("SDSS ugriz photometry on the §7 panchromatic SED")
ax_top.grid(True, alpha=0.3)
ax_top.legend(fontsize=10)
ax_bot.plot(_pivot, np.array(tng_mags) - np.array(bp_mags), "k.-", linewidth=1.5)
ax_bot.axhline(0.0, color="grey", linestyle=":")
ax_bot.set_xlabel(r"pivot $\lambda$ [Å]")
ax_bot.set_ylabel("tengri − BAGPIPES [mag]")
ax_bot.set_ylim(-1.0, 0.2)
ax_bot.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("bagpipes_14_photometry_sdss.png")

for band, m_b, m_t in zip(_sdss_bands, bp_mags, tng_mags):
    print(f"§13 {band}: BAGPIPES {m_b:.3f}, tengri {m_t:.3f}, Δ {m_t - m_b:+.3f} mag")


# %% [markdown]
# ### §13b Photometry without nebular — the residual is nebular lines
#
# §13 leaves tengri −0.11 mag (u) and −0.15 mag (g) brighter than BAGPIPES
# while r/i/z agree to ≤ 0.02 mag. This cell pins that on the §9 Cue/CLOUDY
# nebular difference: rebuild both codes' SEDs with the nebular block removed
# and re-run the same convolution. The band-averaged residual drops from
# ⟨Δ⟩ ≈ −0.06 mag (full) to ≈ −0.01 mag (no nebular), and crucially the u/g
# excess collapses with it — confirming the gap is a nebular **line-strength**
# difference (Cue vs CLOUDY) in the bands carrying [O II] 3727 (u) and
# Hβ + [O III] 4959/5007 (g), not a dust or stellar-continuum effect. What
# remains without nebular is the ≈ 0.01 mag residual §4 stellar colour.

# %%
comp_b_nonneb = dict(comp_b_full)
del comp_b_nonneb["nebular"]
mg_b_nonneb = B._build_model(comp_b_nonneb)
w_b_nonneb, L_b_nonneb = B.to_lnu(mg_b_nonneb)

m_full_nonneb = SEDModel.build(
    ssp_data=ssp,
    stellar=STELLAR_FIDUCIAL,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(TAU_GYR_FIDUCIAL),
        "age_gyr": Fixed(AGE_GYR_FIDUCIAL),
        "log_total_mass": Fixed(LOG_MASS_FIDUCIAL),
        "*": FIXED,
    },
    dust={
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "tau_bc": Fixed(TAU_BC),
        "tau_diff": Fixed(TAU_DIFF),
        "emission": {
            "type": "draine_li2007",
            "qpah": Fixed(QPAH_FIDUCIAL),
            "umin": Fixed(UMIN_FIDUCIAL),
            "gamma_dl": Fixed(GAMMA_FIDUCIAL),
            "*": FIXED,
        },
        "*": FIXED,
    },
    redshift=Fixed(0.0),
)
s_full_nonneb = m_full_nonneb.predict_state({})
_L_t_nonneb = np.asarray(s_full_nonneb.derived["sed_dust_attenuated"]) + np.asarray(
    s_full_nonneb.derived["sed_dust_ir"]
)

bp_mags_nn = [_ab_mag(w_b_nonneb, L_b_nonneb, f.wave, f.trans) for f in _filters]
tng_mags_nn = [
    _ab_mag(np.asarray(s_full_nonneb.wave), _L_t_nonneb, f.wave, f.trans) for f in _filters
]

fig, (ax_top, ax_bot) = plt.subplots(
    2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax_top.plot(_pivot, bp_mags, "o-", color="C0", linewidth=1.7, label="BAGPIPES (full)")
ax_top.plot(_pivot, tng_mags, "s--", color="C1", linewidth=1.7, label="tengri (full)")
ax_top.plot(
    _pivot, bp_mags_nn, "o:", color="C0", linewidth=1.2, alpha=0.7, label="BAGPIPES (no nebular)"
)
ax_top.plot(
    _pivot, tng_mags_nn, "s:", color="C1", linewidth=1.2, alpha=0.7, label="tengri (no nebular)"
)
ax_top.invert_yaxis()
ax_top.set_ylabel(r"AB magnitude (10 pc, $M_\star = 10^{10}\,M_\odot$)")
ax_top.set_title("§13b — SDSS ugriz with vs without nebular")
ax_top.legend(fontsize=9)
ax_top.grid(True, alpha=0.3)
ax_bot.plot(
    _pivot, np.array(tng_mags) - np.array(bp_mags), "k.-", linewidth=1.5, label="full pipeline"
)
ax_bot.plot(
    _pivot,
    np.array(tng_mags_nn) - np.array(bp_mags_nn),
    "k.:",
    linewidth=1.5,
    alpha=0.7,
    label="no nebular",
)
ax_bot.axhline(0.0, color="grey", linestyle=":")
ax_bot.set_xlabel(r"pivot $\lambda$ [Å]")
ax_bot.set_ylabel("tengri − BAGPIPES [mag]")
ax_bot.set_ylim(-1.0, 0.2)
ax_bot.legend(fontsize=9)
ax_bot.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("bagpipes_13b_photometry_no_neb.png")

print(
    f"§13b nebular-attribution test:  "
    f"full ⟨Δ⟩ = {np.mean(np.array(tng_mags) - np.array(bp_mags)):+.3f} mag, "
    f"no-neb ⟨Δ⟩ = {np.mean(np.array(tng_mags_nn) - np.array(bp_mags_nn)):+.3f} mag"
)


# %% [markdown]
# ## §14 Forward-model timing — order-of-magnitude sanity check
#
# Same fiducial galaxy on both sides — τ-delayed SFH, Calzetti dust at
# `Av = 1`, DL07 IR, Cloudy v25 (BAGPIPES) / Cue v17 (tengri) nebular,
# Inoue14 IGM at z = 0 — and time a single forward evaluation. Both
# codes finish a full SED in ~10² ms; they are in the **same
# performance class** at the public API for a single forward pass.
#
# tengri's real speed advantage is **not** the forward call. It is the
# gradient: `jax.grad` differentiates the JIT'd objective at the cost
# of roughly one extra forward pass, where any non-JAX code (BAGPIPES,
# CIGALE) has to fall back to finite differences with `2 × n_params`
# forward calls. For a 10-parameter model that is a 20× swing.
#
# Caveats: timings depend on the JAX cache state, the CPU, and whether
# tengri's persistent JAX cache is warm. The numbers below are
# illustrative orders of magnitude, not a benchmark.

# %%
import time

_comp_b_full_timing = {
    "redshift": 0.0,
    "delayed": {
        "metallicity": 1.0,
        "age": AGE_GYR_FIDUCIAL,
        "tau": TAU_GYR_FIDUCIAL,
        "massformed": LOG_MASS_FIDUCIAL,
    },
    "dust": {
        "type": "Calzetti",
        "Av": AV_FIDUCIAL,
        "eta": 1.0,
        "qpah": QPAH_FIDUCIAL,
        "umin": UMIN_FIDUCIAL,
        "gamma": GAMMA_FIDUCIAL,
    },
    "nebular": {"logU": -2.0},
}

# Warm-up
B._build_model(_comp_b_full_timing)
_n_b = 20
_t0 = time.perf_counter()
for _ in range(_n_b):
    _mg = B._build_model(_comp_b_full_timing)
_t_b_per = (time.perf_counter() - _t0) / _n_b
print(f"§14 BAGPIPES model_galaxy build: {_t_b_per * 1000:.1f} ms / call (warm, n={_n_b})")

# tengri warm-up (compile the JIT). Reuse the §7 build.
m_full.predict_state({})  # one warm call
_n_t = 100
_t0 = time.perf_counter()
for _ in range(_n_t):
    _ = m_full.predict_state({})
_t_t_per = (time.perf_counter() - _t0) / _n_t
print(f"§14 tengri SEDModel.predict_state:  {_t_t_per * 1000:.1f} ms / call (warm, n={_n_t})")
print(f"§14 speedup tengri / BAGPIPES: {_t_b_per / _t_t_per:.1f}×")


# %% [markdown]
# ## tengri in BAGPIPES-mode — full-SED head-to-head
#
# Every section above swept one physics block. This is the whole forward
# model at once: tengri configured to emulate BAGPIPES end to end — the
# shared BC03+MILES SSP, the fiducial τ-delayed SFH, a Calzetti
# attenuation law, Draine & Li (2007) IR re-emission, and nebular —
# overlaid on BAGPIPES' own panchromatic output at matched parameters (the
# §11 configuration). The top panel is the overlay; the bottom is the
# fractional residual `tengri / BAGPIPES − 1` with the ±25 % band shaded.
# Optical agreement is reported as a normalization ratio and its 16–84 %
# spread; the nebular emission lines (§9) drive the spread.

# %%
import chex

# Reuse the §11 panchromatic full SED: tengri's BAGPIPES-mode model and
# BAGPIPES' own output, both at the fiducial galaxy.
w_ext, L_ext = np.asarray(w_b_full), np.asarray(L_b_full)
wave_t = np.asarray(s_full.wave)
L_t = (
    np.asarray(s_full.derived["sed_dust_attenuated"])
    + np.asarray(s_full.derived["sed_dust_ir"])
    + np.asarray(s_full.derived["sed_nebular"])
)

# Put tengri on BAGPIPES' wavelength grid so the two compare point for point.
L_t_on_ext = U.regrid(wave_t, L_t, w_ext)
chex.assert_equal_shape([L_ext, L_t_on_ext])

mask = (w_ext > 0) & (L_ext > 0) & (L_t_on_ext > 0)
resid = np.full(w_ext.shape, np.nan, dtype=float)
resid[mask] = L_t_on_ext[mask] / L_ext[mask] - 1.0

# Headline numbers: the optical normalization ratio tengri/BAGPIPES and
# its 16–84% spread. With the §7 dust applied as a single screen, the
# continuum normalization sits at the bottom of the spread (P16 ≈ 1.0×);
# the median and upper end are pulled up by the Cue-vs-Cloudy nebular
# emission lines (§9). Reporting the ratio + spread keeps that line-driven
# scatter distinct from the continuum match.
opt = mask & (w_ext >= 1000.0) & (w_ext <= 10000.0)
ratio_opt = L_t_on_ext[opt] / L_ext[opt]
norm = float(np.median(ratio_opt))
p16, p84 = float(np.percentile(ratio_opt, 16)), float(np.percentile(ratio_opt, 84))
print(
    f"full-SED head-to-head tengri/BAGPIPES optical (1000–10000 Å): "
    f"normalization {norm:.2f}×, 16–84% spread {p16:.2f}–{p84:.2f}×"
)
_assert_comparable(L_ext, L_t, name="full-SED head-to-head")

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax.plot(w_ext, L_ext, "C0-", linewidth=1.5, label="BAGPIPES")
ax.plot(w_ext, L_t_on_ext, "C1--", linewidth=1.5, label="tengri (BAGPIPES-mode)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e7)
ax.set_ylim(1e22, 1e31)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title("tengri in BAGPIPES-mode vs BAGPIPES — full panchromatic SED")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(
    0.02,
    0.05,
    rf"tengri/BAGPIPES $= {norm:.2f}\times$ (16–84%: {p16:.2f}–{p84:.2f})",
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
ax_r.set_xlim(1e2, 1e7)
ax_r.set_ylim(-1.0, 1.0)
ax_r.set_xlabel(r"$\lambda$ [Å]")
ax_r.set_ylabel(r"tengri/BAGPIPES $-1$")
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("bagpipes_full_sed_headtohead.png")
plt.show()


# %% [markdown]
# ## Summary
#
# Section-by-section, at matched parameters, in the logical flow used
# in this notebook (stellar building blocks → SED assembly → cosmology
# → observational → meta):
#
# - **§1 SSPs.** BC03+MILES Kroupa templates port through the DSPS
#   HDF5 layout at float32 round-trip precision (~1e-7). Both codes
#   consume the same numeric SSP arrays.
# - **§2 parametric SFHs.** Delayed-τ, double power-law (§2a) and
#   lognormal (§2b) all match BAGPIPES directly: each takes an explicit
#   `age_gyr` anchor and evaluates the shape in cosmic time since
#   formation, `T = age − lookback`. Anchored to the BAGPIPES
#   age of the universe, the two-panel figures overlay the same curve —
#   no axis flip, no coordinate conversion. (`lnorm` uses a log10-Gaussian
#   rather than the exact Carnall 1/T lognormal, so its wings differ
#   slightly; direction and peak match.) All forms integrate to
#   `10**log_total_mass`.
# - **§3 Leja+19 continuity SFH.** Non-parametric piecewise-constant
#   SFH. Bit-for-bit agreement on SFR(t) at matched parameters once
#   the bin-ordering convention is reconciled (BAGPIPES indexes
#   oldest→youngest, tengri young→old).
# - **§4 stellar SED.** tengri / BAGPIPES median ratio in the optical
#   ≈ 1.010 ± 0.001 — a flat ~1 % systematic at matched SFH and SSP.
# - **§5 metallicity sensitivity.** Both codes track the standard
#   age-metallicity-degeneracy direction. Visual match.
# - **§6–§8 dust attenuation + IR.** Calzetti curves overlap; CF00 /
#   Cardelli / Salim differ by construction. With the §7 single-screen
#   mapping the attenuated optical matches to ~1 %, and the DL07 IR now
#   matches in shape too (both peak ~130 µm; 30–100 µm and submm to ~6 %).
#   Shape agreement relies on proper accounting of the DL07 PDR luminosity
#   physics: `gamma` is a dust-mass fraction but the warm PDR dust emits
#   ~14× more per mass (Eq. 33).
# - **§9 nebular.** Cloudy v25 (BAGPIPES) vs Cloudy v17 (Cue, tengri).
#   tengri Hα ≈ 3.6 × BAGPIPES Hα — the Cloudy generation difference
#   plus bare-stellar vs SFH-integrated convolution path.
# - **§10 LSF.** tengri's `velocity_broaden` matches the analytic
#   Gaussian σ = 150 km/s FWHM to 0.7 %.
# - **§11 panchromatic.** The combined picture; per-section residuals
#   stack.
# - **§12 IGM.** Inoue14 vs Inoue14 agrees redward of the Lyman limit
#   and now extends below 912 Å (LyC opacity restored). The Asada+2025
#   CGM damping wing (§12b, tengri-only) produces the full Totani+06
#   damping-wing shape at z = 7.
# - **§13 SDSS photometry.** With the §7 single-screen dust, tengri
#   reproduces the BAGPIPES ugriz magnitudes to ≤ 0.02 mag in r/i/z but stays
#   −0.11 mag (u) and −0.15 mag (g) brighter — the two bands carrying the
#   strongest nebular lines ([O II] 3727 in u; Hβ + [O III] 4959/5007 in g).
#   §13b attributes this to the §9 Cue-vs-Cloudy nebular line-strength
#   difference: the band-averaged residual drops from ⟨Δ⟩ −0.06 → −0.01 mag
#   with the nebular block removed, leaving only the ≈ 0.01 mag §4 stellar
#   colour mismatch.
# - **§14 timing.** Both codes finish a full SED in 80–120 ms.
# - **full-SED head-to-head.** The whole BAGPIPES-mode forward model on
#   one axis with a fractional-residual panel and an optical normalization
#   ratio + 16–84 % spread. With the dust corrected the continuum sits at
#   ≈ 1.0× (P16); the nebular emission lines drive the upper spread.
#
# Every residual that exceeds the noise floor in a panel above has a
# one-sentence physics explanation attached to it. The companion
# README (`reproduction/bagpipes/README.md`) holds the band-by-band
# comparison table.

# %% [markdown]
# ## References
#
# * Carnall et al. 2018, MNRAS 480, 4379 — BAGPIPES
# * Bruzual & Charlot 2003, MNRAS 344, 1000 — BC03 SSPs
# * Sánchez-Blázquez et al. 2006, MNRAS 371, 703 — MILES library
# * Kroupa 2001, MNRAS 322, 231 — IMF
# * Calzetti et al. 2000, ApJ 533, 682 — starburst attenuation
# * Cardelli, Clayton & Mathis 1989, ApJ 345, 245 — MW extinction
# * Charlot & Fall 2000, ApJ 539, 718 — two-component dust
# * Salim, Boquien & Lee 2018, ApJ 859, 11 — attenuation modification
# * Draine & Li 2007, ApJ 657, 810 — dust IR emission
# * Inoue et al. 2014, MNRAS 442, 1805 — IGM absorption
# * Asada et al. 2025 — CGM damping wing
# * Li et al. 2025 — Cue nebular emulator
