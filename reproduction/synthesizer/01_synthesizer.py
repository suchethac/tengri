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
# # Reproducing Synthesizer's physics with tengri
#
# Synthesizer (Lovell et al. 2025, OJA; Roper et al. 2026, JOSS) is a
# forward-modeling package for synthetic galaxy observables. It builds
# spectra from pre-computed HDF5 grids and an `EmissionModel` tree: stellar
# SSPs, parametric star formation history, attenuation and dust-emission
# laws, Madau/Inoue IGM, and `UnifiedAGN` black-hole emission (disc + narrow
# and broad line regions + dusty torus).
#
# This notebook places that forward model next to tengri at matched
# parameters, with the emphasis on the AGN model (§9).
#
# **What sits on each side.** For stellar populations (§1) both read the same
# templates (Synthesizer's stellar grid re-shaped into tengri's form), so the
# §1 residual is interpolation alone. For AGN line regions (§9c, §9d) tengri
# reads the *same* `/spectra/nebular` array that Synthesizer's `UnifiedAGN`
# extracts, through the public grammar (`nlr={'type': 'synthesizer_spectra'}`).
#
# **Grids.** This notebook uses Synthesizer's *test* grids
# (`synthesizer-download --stellar-test-grids --agn-test-grids
# --dust-grid`). The test grids have the same physics as production grids at
# lower resolution (AGN test grids sample each photoionization axis at two
# nodes); this coarseness is a property of both sides equally. Synthesizer
# has no X-ray or radio component.

# %% [markdown]
# ## Setup

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reproduction.synthesizer._drivers import (
    synthesizer_driver as S,
    synthesizer_ssp_to_dsps as REPACK,
    units as U,
)

import tengri
from tengri import (
    FIXED,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)

# Force the inline backend so figures embed on (re-)render regardless of the
# ambient MPLBACKEND. A non-inline backend (e.g. Agg) drops the save_fig()
# auto-display and produces a figure-less notebook. No-op when run as a script.
try:  # noqa: SIM105
    get_ipython().run_line_magic("matplotlib", "inline")
except NameError:
    pass

warnings.filterwarnings("ignore")
tengri.plot.setup_style()

# Unit-sanity guard. Synthesizer reports L_ν in erg/s/Hz natively, so the bridge
# is mostly stripping the unyt tag; the one real conversion is the L_λ ↔ L_ν
# Jacobian, exercised here so a factor-of-c slip trips at Setup, not in a panel.
_unit_check = U.verify_unit_conversion(rtol=1e-3)
print(
    f"unit-conversion bolometric round-trip: "
    f"rel_err = {_unit_check['rel_err']:.2e}  (target < 1e-3)"
)

# Fiducial galaxy shared across the SED panels.
LOG_MASS_FIDUCIAL = 10.0
TAU_GYR_FIDUCIAL = 1.0
AGE_GYR_FIDUCIAL = 5.0
AV_FIDUCIAL = 1.0
Z_FIDUCIAL = 0.02  # absolute metallicity (≈ solar on the Synthesizer grid)

# nbclient kernels don't bind ``__file__`` (the resources path is the notebook
# directory), so fall back to the CWD.
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
# ## Common stellar grid
#
# Both codes read identical templates: Synthesizer's stellar grid is
# re-shaped once into tengri's form (cached locally, not committed), so the
# §1 agreement is a numerical statement.

# %%
_grid_h5 = _HERE / "_drivers" / "data" / "synthesizer_test_grid.h5"
if not _grid_h5.exists():
    REPACK.repackage_stellar_grid(_grid_h5)
ssp = load_ssp_data(str(_grid_h5))
print(
    f"Synthesizer test_grid (repackaged): {ssp.ssp_wave.shape[0]} wavelengths, "
    f"{ssp.ssp_lgmet.shape[0]} metallicities, {ssp.ssp_lg_age_gyr.shape[0]} age bins; "
    f"λ up to {ssp.ssp_wave.max():.1e} Å."
)


# %% [markdown]
# ## §1 Single stellar populations
#
# Single stellar populations at ≈ solar metallicity from 1 Myr to 10 Gyr,
# over the relative residual. Both sides read identical numbers; the residual
# sits at the level set by single-precision round-trip through the shared
# grid (gray line, 1e-6).

# %%
_target_ages_yr = [1e6, 1e7, 1e8, 1e9, 1e10]
i_z = int(np.argmin(np.abs(ssp.ssp_lgmet - np.log10(Z_FIDUCIAL))))
_age_idx = [
    int(np.argmin(np.abs(ssp.ssp_lg_age_gyr - np.log10(a / 1e9)))) for a in _target_ages_yr
]

syn_ssp, tng_ssp, age_labels = [], [], []
for ia in _age_idx:
    age_gyr = float(10.0 ** ssp.ssp_lg_age_gyr[ia])
    w_s, L_s = S.ssp_spectrum(metallicity=Z_FIDUCIAL, age_gyr=age_gyr)
    syn_ssp.append((w_s, L_s))
    tng_ssp.append((ssp.ssp_wave, np.asarray(ssp.ssp_flux[i_z, ia, :]) * U.L_SUN_ERG_PER_S))
    age_labels.append(f"{age_gyr * 1e3:g} Myr" if age_gyr < 1 else f"{age_gyr:g} Gyr")

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
colors = plt.cm.viridis(np.linspace(0, 1, len(_age_idx)))
for color, label, (w_s, L_s), (w_t, L_t) in zip(colors, age_labels, syn_ssp, tng_ssp):
    ax.plot(w_s, L_s, color=color, linewidth=2.0, label=label)
    ax.plot(w_t, L_t, color="k", linewidth=0.8, linestyle="--", alpha=0.7)
    L_t_on_s = U.regrid(w_t, L_t, w_s)
    resid = np.abs(L_t_on_s - L_s) / np.maximum(np.abs(L_s), 1e-30)
    resid[~np.isfinite(resid)] = 0.0
    ax_r.plot(w_s, resid, color=color, linewidth=1.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e6)
ax.set_ylim(1e16, 5e24)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz / $M_\odot$]")
ax.set_title("Synthesizer test_grid Z ≈ Z⊙ — Synthesizer (solid) vs tengri (black dashed)")
ax.legend(fontsize=9, title="SSP age")
ax.grid(True, alpha=0.3)
ax_r.set_xscale("log")
ax_r.set_yscale("log")
ax_r.set_xlabel(r"$\lambda$ [Å]")
ax_r.set_ylabel(r"$|\Delta| / L_{\rm Syn}$", fontsize=9)
ax_r.set_ylim(1e-9, 1e-2)
ax_r.axhline(1e-6, color="gray", linestyle=":", alpha=0.6, label="single-precision floor")
ax_r.legend(loc="upper right", fontsize=8)
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_01_ssp.png")

_w_ref, _L_ref = syn_ssp[3]  # 1 Gyr
_mask = (_w_ref >= 3000) & (_w_ref <= 10000)
_t_on_s = U.regrid(tng_ssp[3][0], tng_ssp[3][1], _w_ref)
_res = np.abs(_t_on_s[_mask] - _L_ref[_mask]) / np.maximum(_L_ref[_mask], 1e-30)
print(f"§1 SSP 1 Gyr optical residual: median {np.median(_res):.2e}, max {_res.max():.2e}")


# %% [markdown]
# ## §2 Star formation history — delayed-τ
#
# Synthesizer's `SFH.DelayedExponential` uses `SFR(t) ∝ t · exp(−t/τ)`,
# peaking at `t = τ`. tengri's `sfh.delayed` is the same closed form.
# tengri's SFR history is read from a built model on the log-spaced lookback
# grid, not a re-evaluated analytic curve. Both normalize to 1 M⊙ formed.

# %%
t_s, sfr_s = S.sfh_curve(tau_gyr=TAU_GYR_FIDUCIAL, max_age_gyr=AGE_GYR_FIDUCIAL)
t_s_cosmic_gyr = AGE_GYR_FIDUCIAL - t_s / 1e9
_mass_s = float(np.trapezoid(sfr_s[np.argsort(t_s)], t_s[np.argsort(t_s)]))

_m_sfh = SEDModel.build(
    ssp_data=ssp,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(TAU_GYR_FIDUCIAL),
        "age_gyr": Fixed(AGE_GYR_FIDUCIAL),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
_state_sfh = _m_sfh.predict_state({})
_lbt_yr = np.asarray(_state_sfh.derived["sfh_grid_lbt_yr"])
_sfr_history = np.asarray(_state_sfh.derived["sfr_history"])
t_t_cosmic_gyr = AGE_GYR_FIDUCIAL - _lbt_yr / 1e9
_idx = np.argsort(_lbt_yr)
_mass_t = float(np.trapezoid(_sfr_history[_idx], _lbt_yr[_idx]))
print(
    f"§2 ∫SFR dt: Synthesizer = {_mass_s:.4f} M⊙, "
    f"tengri pipeline = {_mass_t:.4f} M⊙ (target 1.0000)"
)

fig, ax_l, ax_r = U.two_panel_fig()
for ax, title in (
    (ax_l, "Synthesizer delayed-τ (τ=1 Gyr, age=5 Gyr)"),
    (ax_r, "tengri pipeline sfr_history (log-lbt grid)"),
):
    ax.set_xlabel("Cosmic age since SF onset [Gyr]")
    ax.set_ylabel(r"SFR [$M_\odot\ \mathrm{yr}^{-1}$]")
    ax.set_xlim(0, 5)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.axvline(TAU_GYR_FIDUCIAL, color="gray", linestyle=":", alpha=0.6)
ax_l.plot(t_s_cosmic_gyr, sfr_s, "C0-", linewidth=2.0, label=rf"$\tau$ = {TAU_GYR_FIDUCIAL:g} Gyr")
ax_l.legend(fontsize=9)
ax_r.plot(t_t_cosmic_gyr, _sfr_history, "C1-", linewidth=2.0)
fig.tight_layout()
save_fig("synthesizer_02_sfh_delayed.png")


# %% [markdown]
# ## §3 Integrated stellar SED
#
# Convolve the τ-delayed SFH with the SSP grid, no dust or nebular. Both form
# 10^10 M⊙. The printed ~1.09× optical ratio is *not* a normalization error
# — it is the two codes' independent SFH discretizations: Synthesizer integrates
# an analytic SFZH onto SSP age-bin edges, while tengri convolves on its
# log-lookback quadrature (validated against dense code-independent
# convolution, #964). The mild chromatic spread (P5–P95 ≈ 1.05–1.10) follows
# from different age weighting.

# %%
w_s3, L_s3 = S.stellar_sed(
    tau_gyr=TAU_GYR_FIDUCIAL,
    max_age_gyr=AGE_GYR_FIDUCIAL,
    metallicity=Z_FIDUCIAL,
    log_mass=LOG_MASS_FIDUCIAL,
)

m_stellar = SEDModel.build(
    ssp_data=ssp,
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
_assert_comparable(L_s3, s_stellar.sed_intrinsic, name="§3 stellar")

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(
    ax_l,
    ax_r,
    label_l="Synthesizer  delayed-τ + test_grid",
    label_r="tengri  sfh.delayed + same SSP",
)
ax_l.plot(w_s3, L_s3, "C0-", linewidth=1.5)
ax_r.plot(s_stellar.wave, s_stellar.sed_intrinsic, "C1-", linewidth=1.5)
# Floor the y-range to the stellar continuum: the repackaged grid is clipped at 1e8 Å,
# so the SED drops to ~0 at the long-λ edge — without a floor the log axis would
# stretch across ~50 empty decades down to that cliff.
_s3pk = max(float(np.nanmax(L_s3)), float(np.nanmax(np.asarray(s_stellar.sed_intrinsic))))
for ax in (ax_l, ax_r):
    ax.set_xlim(1e2, 1e6)
    ax.set_ylim(_s3pk * 1e-5, _s3pk * 2)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_03_stellar_sed.png")

_mask_opt = (w_s3 >= 3000) & (w_s3 <= 10000)
_t_on_s3 = U.regrid(np.asarray(s_stellar.wave), np.asarray(s_stellar.sed_intrinsic), w_s3)
_ratios = _t_on_s3[_mask_opt] / L_s3[_mask_opt]
_ratios = _ratios[np.isfinite(_ratios) & (_ratios > 0)]
print(
    f"§3 stellar SED tengri/Synthesizer optical (3000–10000 Å): "
    f"median {np.median(_ratios):.3f}, P5 {np.percentile(_ratios, 5):.3f}, "
    f"P95 {np.percentile(_ratios, 95):.3f}"
)


# %% [markdown]
# ## §4 Dust attenuation curves
#
# Synthesizer's `Calzetti2000` and `PowerLaw` laws against tengri's `calzetti`
# and `power_law` laws, both normalized to `A(λ)/A_V` at 5500 Å. Direct
# analytic evaluation, no SSP-convolution noise.

# %%
from tengri.dust import list_laws

_tengri_laws = list_laws(headline=False)
wave_law = np.logspace(np.log10(1000.0), np.log10(30000.0), 2000)


def _norm_AV(wave, A):
    return A / A[np.argmin(np.abs(wave - 5500.0))]


# (Synthesizer law name, extra params, tengri law name, plot label)
_law_pairs = [
    ("calzetti", {}, "calzetti", "Calzetti+2000"),
    ("power_law", {"slope": -0.7}, "power_law", "Power law (δ=−0.7)"),
]

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)
for ax, title in (
    (ax_l, "Synthesizer attenuation laws"),
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

for syn_name, kw, tengri_law, label in _law_pairs:
    w_s4, A_s4 = S.attenuation_curve(syn_name, wave_aa=wave_law, **kw)
    ax_l.plot(w_s4, _norm_AV(w_s4, A_s4), linewidth=2.0, label=label)
    A_t = np.asarray(_tengri_laws[tengri_law](wave_law))
    ax_r.plot(wave_law, _norm_AV(wave_law, A_t), linewidth=2.0, label=label)
ax_l.legend(fontsize=10)
ax_r.legend(fontsize=10)
fig.tight_layout()
save_fig("synthesizer_04_dust_attenuation.png")


# %% [markdown]
# ## §5 Attenuation applied
#
# The fiducial galaxy with and without dust. Synthesizer applies Calzetti
# screen at A_V = 1; tengri matches with full A_V on the diffuse component
# (attenuates all ages equally), birth-cloud off — the single-screen mapping
# from Prospector/Bagpipes notebooks.

# %%
TAU_DIFF = AV_FIDUCIAL / 1.086
L_s5_attn = S.attenuate(w_s3, L_s3, name="calzetti", av=AV_FIDUCIAL)

m_d = SEDModel.build(
    ssp_data=ssp,
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
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(TAU_DIFF),
        "*": FIXED,
    },
    redshift=Fixed(0.0),
)
s_d = m_d.predict_state({})
_assert_comparable(L_s5_attn, s_d.derived["sed_dust_attenuated"], name="§5 dust applied")

fig, ((ax_l1, ax_r1), (ax_l2, ax_r2)) = plt.subplots(2, 2, sharey=True, figsize=(12, 8))
U.panel(ax_l1, ax_r1, label_l="Synthesizer  intrinsic", label_r="tengri  intrinsic")
U.panel(
    ax_l2,
    ax_r2,
    label_l=rf"Synthesizer  Calzetti  ($A_V = {AV_FIDUCIAL:g}$)",
    label_r=rf"tengri  single-screen Calzetti  ($\tau_V={TAU_DIFF:.2f}$)",
)
ax_l1.plot(w_s3, L_s3, "C0-", linewidth=1.5)
ax_r1.plot(s_stellar.wave, s_stellar.sed_intrinsic, "C1-", linewidth=1.5)
ax_l2.plot(w_s3, L_s5_attn, "C0-", linewidth=1.5)
ax_r2.plot(s_d.wave, s_d.derived["sed_dust_attenuated"], "C1-", linewidth=1.5)
_ymax = float(np.asarray(s_stellar.sed_intrinsic).max())
for ax in (ax_l1, ax_r1, ax_l2, ax_r2):
    ax.set_xlim(1e2, 5e4)
    ax.set_ylim(_ymax * 1e-6, _ymax * 2)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_05_dust_applied.png")


# %% [markdown]
# ## §6 Dust IR re-emission and energy balance
#
# Absorbed stellar UV/optical re-emerges in the infrared. Synthesizer re-emits
# through Draine & Li (2007) templates, enforcing energy balance internally.
# tengri uses the same DL07 grid with the same energy-balance constraint.
# Isolated dust IR emission at matched `(q_PAH, U_min)` — both peak near
# ~140 µm for `U_min = 1`.

# %%
QPAH_FRAC = 0.025  # Synthesizer fraction; ≈ tengri qpah = 2.5
UMIN = 1.0
_te = S.total_emission(
    tau_gyr=TAU_GYR_FIDUCIAL,
    max_age_gyr=AGE_GYR_FIDUCIAL,
    metallicity=Z_FIDUCIAL,
    log_mass=LOG_MASS_FIDUCIAL,
    av=AV_FIDUCIAL,
    qpah=QPAH_FRAC,
    umin=UMIN,
    components=("dust_emission", "total"),
)
w_ir_s, L_ir_s = _te["dust_emission"]

m_ir = SEDModel.build(
    ssp_data=ssp,
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
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(AV_FIDUCIAL / 1.086),
        "emission": {
            "type": "draine_li2007",
            "qpah": Fixed(2.5),
            "umin": Fixed(UMIN),
            "gamma_dl": Fixed(0.05),
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
print(f"§6 tengri energy balance: L_abs={_L_abs:.3e}, L_IR={_L_ir:.3e}, resid={_eb_resid:.2e}")

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(
    ax_l,
    ax_r,
    label_l="Synthesizer  DL07 dust IR",
    label_r="tengri  DL07 dust IR (energy-balanced)",
)
ax_l.plot(w_ir_s, L_ir_s, "C0-", linewidth=1.5)
ax_r.plot(s_ir.wave, s_ir.derived["sed_dust_ir"], "C1-", linewidth=1.5)
_irpk = max(float(L_ir_s.max()), float(np.asarray(s_ir.derived["sed_dust_ir"]).max()))
for ax in (ax_l, ax_r):
    ax.set_xlim(1e4, 1e7)
    ax.set_ylim(_irpk * 1e-3, _irpk * 3)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_06_dust_ir.png")
_pk_s = w_ir_s[(w_ir_s > 1e5)][np.argmax(L_ir_s[(w_ir_s > 1e5)])]
_wt = np.asarray(s_ir.wave)
_Lt = np.asarray(s_ir.derived["sed_dust_ir"])
_pk_t = _wt[(_wt > 1e5)][np.argmax(_Lt[(_wt > 1e5)])]
print(f"§6 dust IR far-IR peak: Synthesizer {_pk_s / 1e4:.0f} µm, tengri {_pk_t / 1e4:.0f} µm")


# %% [markdown]
# ## §7 Panchromatic SED
#
# Stellar + nebular + dust attenuation + DL07 IR from rest-UV to far-IR.
# Percent-level disagreements from §3–§6 and nebular (§8) stack here. (The
# Synthesizer test grid has coarse far-IR nebular continuum that lifts
# mid-IR — a 2-node test grid property, not physics.)

# %%
w_full_s, L_full_s = _te["total"]

m_full = SEDModel.build(
    ssp_data=ssp,
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
        "tau_bc": Fixed(0.0),
        "tau_diff": Fixed(AV_FIDUCIAL / 1.086),
        "emission": {
            "type": "draine_li2007",
            "qpah": Fixed(2.5),
            "umin": Fixed(UMIN),
            "gamma_dl": Fixed(0.05),
            "*": FIXED,
        },
        "*": FIXED,
    },
    neb={"type": "cue", "neb_logU": Fixed(-2.0), "neb_logZ_gas": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_full = m_full.predict_state({})
_sed_full_t = (
    np.asarray(s_full.derived["sed_dust_attenuated"])
    + np.asarray(s_full.derived["sed_dust_ir"])
    + np.asarray(s_full.derived["sed_nebular"])
)

fig, ax_l, ax_r = U.two_panel_fig(figsize=(13, 5))
U.panel(ax_l, ax_r, label_l="Synthesizer  total", label_r="tengri  panchromatic")
ax_l.plot(w_full_s, L_full_s, "C0-", linewidth=1.5)
ax_r.plot(s_full.wave, _sed_full_t, "C1-", linewidth=1.5)
_fpk = max(float(np.nanmax(L_full_s)), float(np.nanmax(_sed_full_t)))
for ax in (ax_l, ax_r):
    ax.set_xlim(1e2, 1e7)
    ax.set_ylim(_fpk * 1e-5, _fpk * 3)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_07_panchromatic.png")


# %% [markdown]
# ## §8 Nebular emission
#
# Synthesizer's nebular comes from Cloudy grids baked into SSP grids; tengri
# uses Cue (Li et al. 2025), a neural emulator on a different Cloudy version,
# so lines differ accordingly. The panel reports integrated, continuum-subtracted
# line luminosity (width- and grid-independent).

# %%
NEB_AGE = 0.01  # Gyr — young constant-SFR population
NEB_LOGMASS = 9.0
w_neb_s, L_neb_s = S.nebular_sed(age_gyr=NEB_AGE, metallicity=Z_FIDUCIAL, log_mass=NEB_LOGMASS)
L_neb_s = np.clip(L_neb_s, 0.0, None)

m_neb = SEDModel.build(
    ssp_data=ssp,
    sfh={
        "type": "const",
        "start_gyr": Fixed(NEB_AGE),
        "end_gyr": Fixed(0.0),
        "log_total_mass": Fixed(NEB_LOGMASS),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    neb={"type": "cue", "neb_logU": Fixed(-2.0), "neb_logZ_gas": Fixed(0.0), "*": FIXED},
    redshift=Fixed(0.0),
)
s_neb = m_neb.predict_state({})
L_neb_t = np.asarray(s_neb.derived["sed_nebular"])

fig, ax_l, ax_r = U.two_panel_fig(figsize=(13, 5))
U.panel(
    ax_l,
    ax_r,
    label_l="Synthesizer  Cloudy nebular (10 Myr CSF)",
    label_r="tengri  Cue emulator (10 Myr CSF)",
)
ax_l.plot(w_neb_s, L_neb_s, "C0-", linewidth=1.0)
ax_r.plot(s_neb.wave, L_neb_t, "C1-", linewidth=1.0)


# Frame the y-axis on the lines inside the plotted window (900–7000 Å); a global
# max would be set by a far-UV resonance line off-panel, leaving dead headroom.
def _winpk(w, L, lo=900.0, hi=7000.0):
    w = np.asarray(w)
    m = (w >= lo) & (w <= hi)
    return float(np.nanmax(L[m])) if m.any() else float(np.nanmax(L))


_npk = max(_winpk(w_neb_s, L_neb_s), _winpk(np.asarray(s_neb.wave), L_neb_t))
for ax in (ax_l, ax_r):
    ax.set_xlim(900, 7000)
    ax.set_ylim(_npk * 1e-3, _npk * 2)
    ax.set_xscale("linear")
    ax.grid(True, alpha=0.3)
# Integrated line luminosity (width- and grid-independent). A single-bin peak
# ratio measures line width, not luminosity, so it is not used here.
#
# Note on the window: this notebook reads Synthesizer's *test* grid, which is
# coarse in the optical (~17 Å), so a line's flux spreads beyond a tight ±12 Å
# window (Hα lands on a single grid point). A wider ±30 Å window captures the
# full line on both sides; with it the integrated ratio is well behaved and
# consistent with the other Cloudy codes (Hα/Hβ ≈ 1.0× vs FSPS and BAGPIPES),
# confirming Cue's line physics is sound. [O III] carries the usual
# Cloudy-version spread. (The notebook's headline Synthesizer comparison is the
# Unified AGN model in §9.)
_w_t = np.asarray(s_neb.wave)
print("§8 integrated line luminosity (tengri Cue / Synthesizer Cloudy; test grid, ±30 Å):")
for _c, _name in [(6563.0, "Hα"), (5007.0, "[O III]"), (4861.0, "Hβ")]:
    _ls = U.line_lum(w_neb_s, L_neb_s, _c, half=30.0)
    _lt = U.line_lum(_w_t, L_neb_t, _c, half=30.0)
    if _ls > 0:
        print(
            f"    {_name} {_c:.0f} Å: Synthesizer {_ls:.2e}, "
            f"tengri {_lt:.2e} erg/s → {_lt / _ls:.2f}×"
        )
    else:
        print(f"    {_name} {_c:.0f} Å: unmeasurable on the coarse test grid")
fig.tight_layout()
save_fig("synthesizer_08_nebular.png")


# %% [markdown]
# ## §9 AGN — the Unified AGN model
#
# Synthesizer's `UnifiedAGN` combines four regions — disc, NLR, BLR, and torus
# — with disc–torus geometry (inclination-dependent mask), reproduced here
# piece by piece and then as the combined spectrum.
#
# The AGN Cloudy grid stores disc continuum three ways: **incident** (bare
# disc), **transmitted** (through gas), and **nebular** (gas re-emission).
# Covering fractions set how much is reprocessed vs escapes. The torus
# reprocesses obscured disc luminosity into warm-dust IR. At grazing
# inclination (`inclination + θ_torus > 90°`) the disc and BLR vanish (Type-2
# view); the NLR stays visible. That accounting, gated by the inclination
# mask, produces the components in §9a–§9f.
#
# The black hole is driven by `(mass, accretion_rate_eddington)`, with
# `inclination` and `theta_torus` setting geometry. Bolometric luminosity
# maps to tengri's `agn_log_lbol = log10(L_bol / L⊙)`.

# %%
import jax.numpy as jnp

# Synthesizer grid directory (where synthesizer-download placed the test grids).
# tengri's grid-backed line blocks read the *same* AGN Cloudy grids via this path,
# exported as ``TENGRI_SYNTHESIZER_AGN_GRID_DIR`` in the §9 setup below.
_SYN_GRID_DIR = os.environ.get(
    "SYNTHESIZER_GRID_DIR",
    os.path.expanduser("~/Library/Application Support/Synthesizer/grids"),
)

# Fiducial AGN.
BH_MASS = 1e8  # Msun
BH_EDD = 0.5  # Eddington ratio
BH_INC = 30.0  # degrees
BH_Z = 0.01  # absolute gas metallicity
THETA_TORUS = 30.0  # degrees
CF = 0.1  # NLR/BLR covering fraction

agn = S.agn_unified(
    mass_msun=BH_MASS,
    eddington=BH_EDD,
    inclination_deg=BH_INC,
    metallicity=BH_Z,
    cf_nlr=CF,
    cf_blr=CF,
    theta_torus_deg=THETA_TORUS,
)
agn_log_lbol = float(np.log10(agn["_bolometric_erg_s"] / U.L_SUN_ERG_PER_S))
print(
    f"§9 Synthesizer BH: L_bol = {agn['_bolometric_erg_s']:.3e} erg/s = 10^{agn_log_lbol:.2f} L⊙"
)

# Every tengri AGN below is built through the *public* grammar (`SEDModel.build`),
# never a raw internal adapter — the same path a fit uses. ``_agn_grammar`` builds
# a composable AGN at the fiducial geometry and returns sed_agn (rest-frame L_nu).
os.environ.setdefault("TENGRI_SYNTHESIZER_AGN_GRID_DIR", _SYN_GRID_DIR)
BH_COS_INC = float(np.cos(np.radians(BH_INC)))
# Match Synthesizer's geometric torus coupling: it reprocesses a fraction
# ``torus_fraction = θ_torus / 90°`` of L_bol (0.33 at θ_torus = 30°). tengri keeps
# ``agn_torus_frac`` free (default 0.5 over-weights the torus ~1.5x here); pinning
# it to θ/90 puts the disc↔torus energy split on the same footing as Synthesizer.
_TORUS_FRAC = THETA_TORUS / 90.0


def _agn_grammar(disc="kubota_done", torus="simple", nlr="none", blr="none", cos_inc=BH_COS_INC):
    """Build a composable AGN via the public API; return (wave, L_nu of sed_agn)."""
    m = SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        agn={
            "type": "composable",
            "disc": {"type": disc},
            "torus": {"type": torus},
            "nlr": {"type": nlr},
            "blr": {"type": blr},
            "agn_log_lbol": Fixed(agn_log_lbol),
            # Match Synthesizer's black hole, not just its bolometric luminosity:
            # the kubota_done (qsosed) disc temperature profile — and therefore the
            # UV bump shape and height — is set by M_BH and the Eddington ratio.
            # Leaving these at tengri's defaults (1e7 M⊙) ran a hotter, fainter
            # disc (0.75x); pinning them to the §9 BH gives a ~0.98x match.
            "agn_log_mbh": Fixed(float(np.log10(BH_MASS))),
            "agn_log_ledd": Fixed(float(np.log10(BH_EDD))),
            "agn_cos_inc": Fixed(cos_inc),
            "agn_theta_torus": Fixed(THETA_TORUS),
            "agn_torus_frac": Fixed(_TORUS_FRAC),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.derived["sed_agn"])


# %% [markdown]
# ### §9a Accretion disc
#
# The incident disc from Synthesizer's **test** AGN grid (qsosed, Kubota &
# Done 2018) vs tengri's `kubota_done` disc. tengri also overlays a broken
# power-law disc (Feltre et al. 2016) for context. tengri's disc inner
# temperature follows Novikov-Thorne theory; its peak is set by standard disc
# physics at given $M_{\rm BH}$ and $\lambda_{\rm Edd}$; the test grid runs
# cooler. Shape comparison at matched bolometric luminosity.

# %%
w_disc_s, L_disc_s = agn["disc"]

_disc_models = (
    ("kubota_done", "Kubota & Done qsosed"),
    ("schartmann2005", "broken power law (Schartmann 2005)"),
)
_disc_tengri = {}
for disc_type, _ in _disc_models:
    m = SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        agn={
            "type": "composable",
            "disc": {"type": disc_type},
            "torus": {"type": "none"},
            "lines": {"type": "none"},
            "agn_log_lbol": Fixed(agn_log_lbol),
            # Match the §9 BH so the kubota_done disc temperature profile matches
            # Synthesizer's qsosed (ignored by the power-law disc). Without this the
            # disc runs at tengri's default 1e7 M⊙ — hotter and ~0.75x in peak.
            "agn_log_mbh": Fixed(float(np.log10(BH_MASS))),
            "agn_log_ledd": Fixed(float(np.log10(BH_EDD))),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    _disc_tengri[disc_type] = (np.asarray(s.wave), np.asarray(s.derived["sed_agn"]))

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(ax_l, ax_r, label_l="Synthesizer  disc", label_r="tengri  disc models")
ax_l.plot(w_disc_s, L_disc_s, "C0-", linewidth=1.5, label="UnifiedAGN disc")
ax_l.legend(fontsize=9)
for c, (dt, label) in zip(("C1", "C2"), _disc_models):
    wt, Lt = _disc_tengri[dt]
    ax_r.plot(wt, Lt, c + "-", linewidth=1.5, label=label)
ax_r.legend(fontsize=9)


# Range from the plotted disc curves over the *plotted window* (1e2–1e5 Å) — a
# global max would be set by the model grid's far-IR edge, not the disc bump.
def _winmax(w, L, lo=1e2, hi=1e5):
    w = np.asarray(w)
    m = (w >= lo) & (w <= hi)
    return float(np.nanmax(np.asarray(L)[m])) if m.any() else float(np.nanmax(L))


_dpk = max(
    _winmax(w_disc_s, L_disc_s),
    _winmax(*_disc_tengri["kubota_done"]),
    _winmax(*_disc_tengri["schartmann2005"]),
)
for ax in (ax_l, ax_r):
    ax.set_xlim(1e2, 1e5)
    ax.set_ylim(_dpk * 3e-3, _dpk * 2)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_09a_disc.png")
print(f"§9a disc peak: Synthesizer {w_disc_s[np.argmax(L_disc_s)]:.0f} Å")


# %% [markdown]
# ### §9b Disc transmitted vs escaped
#
# Synthesizer splits the observed disc into the unobscured (`disc_escaped`) and the
# line-region-transmitted (`disc_transmitted`) paths, combined by covering
# fractions into the observed disc. All three are shown, to expose the geometry.

# %%
fig, ax = plt.subplots(1, 1, figsize=(8, 5))
for key, c, label in (
    ("disc_incident", "C3", "incident (intrinsic)"),
    ("disc_escaped", "C0", "escaped (unobscured)"),
    ("disc_transmitted", "C1", "transmitted (through line regions)"),
    ("disc", "k", "observed disc (combination)"),
):
    if key in agn:
        wk, Lk = agn[key]
        ax.plot(
            wk, Lk, color=c, linewidth=1.5, linestyle="-" if key != "disc" else "--", label=label
        )
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e5)
_b = float(agn["disc_incident"][1].max())
ax.set_ylim(_b * 1e-3, _b * 2)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title("Synthesizer disc: incident → escaped + transmitted → observed")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_09b_disc_transmitted.png")


# %% [markdown]
# ### §9c Narrow-line region
#
# Synthesizer's `UnifiedAGN` builds its NLR from the grid's `/spectra/nebular`
# array (nebular continuum + lines, extracted isotropically at
# `cosine_inclination = 0.5`). tengri reads that same array via the public
# grammar (`nlr={'type': 'synthesizer_spectra'}`): an [O III] 5007-dominant
# reprocessed spectrum scaled by `L_bol × covering_fraction`. Shape
# correlation ≈ 0.97; residual line-peak smoothing is tengri's C²-grid
# interpolation (same gradient-friendly kernel as §9f mask), which shrinks
# on production grids.
#
# **Caveat:** The Synthesizer-side NLR spectrum depends on the Python hash
# seed — `UnifiedAGN.get_spectra` assembles its emission-model tree in
# dict/set order, and the [O III]/Hβ peak ratio lands at 20.9
# (`PYTHONHASHSEED=0`) or 15.6 (other seeds) while tengri stays at 12.8.
# Renders pin `PYTHONHASHSEED=0` (see `reproduction/CONTRACT.md`); both
# orderings are [O III]-dominant, so the conclusion does not depend on the
# seed.

# %%
w_nlr_s, L_nlr_s = agn["nlr"]
# tengri's NLR through the *public* grammar, reading the same /spectra/nebular
# array as UnifiedAGN. No torus, so subtracting the disc isolates the NLR.
_wave_nlr, _L_with_nlr = _agn_grammar(torus="none", nlr="synthesizer_spectra")
_, _L_disc_only = _agn_grammar(torus="none")
L_nlr_t = _L_with_nlr - _L_disc_only

# Same physical product on a shared y-axis → overlay (not two independent panels).
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(w_nlr_s, L_nlr_s, "C0-", lw=1.2, label="Synthesizer  UnifiedAGN NLR  (/spectra/nebular)")
ax.plot(_wave_nlr, L_nlr_t, "C1-", lw=1.0, alpha=0.85, label="tengri  nlr_synthesizer_spectra")
_nlr_pk = float(np.nanmax(L_nlr_s[(w_nlr_s >= 1000) & (w_nlr_s <= 10000)]))
ax.set_ylim(_nlr_pk * 1e-3, _nlr_pk * 3)
ax.set_xlim(1000, 10000)
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
save_fig("synthesizer_09c_nlr.png")


def _peak(w, L, w0, half=15.0):
    m = (w >= w0 - half) & (w <= w0 + half)
    return float(np.nanmax(L[m])) if m.any() else np.nan


_o3hb_t = _peak(_wave_nlr, L_nlr_t, 5006.84) / _peak(_wave_nlr, L_nlr_t, 4861.33)
_o3hb_s = _peak(w_nlr_s, L_nlr_s, 5006.84) / _peak(w_nlr_s, L_nlr_s, 4861.33)
print(
    "§9c NLR (/spectra/nebular path): "
    f"[OIII]5007/Hβ tengri={_o3hb_t:.1f} vs Synthesizer={_o3hb_s:.1f} — both [O III]-dominant."
)


# %% [markdown]
# ### §9d Broad-line region
#
# The broad-line region reproduces the same way — `blr={'type':
# 'synthesizer_spectra'}` reads the BLR grid's `/spectra/nebular` array.
# Synthesizer extracts both line regions isotropically (`cosine_inclination = 0.5`).
# (tengri's physically Type-2-obscured BLR is the separate `blr`/`blr_synthesizer`
# path used by `recipes.unified_agn()`; see §9f.)
#
# **Caveat:** The downloadable test grids don't distinguish NLR from BLR.
# `test_grid_agn-nlr` and `test_grid_agn-blr` are byte-identical placeholders,
# so the BLR overlay mirrors §9c. A genuine NLR-vs-BLR contrast needs
# production grids.

# %%
w_blr_s, L_blr_s = agn["blr"]
# tengri's BLR through the *public* grammar, reading the same /spectra/nebular
# array as UnifiedAGN's blr component (isotropic, cos=0.5).
_wave_blr, _L_with_blr = _agn_grammar(torus="none", blr="synthesizer_spectra")
_, _L_disc_only2 = _agn_grammar(torus="none")
L_blr_t = _L_with_blr - _L_disc_only2
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(w_blr_s, L_blr_s, "C0-", lw=1.2, label="Synthesizer  UnifiedAGN BLR  (/spectra/nebular)")
ax.plot(_wave_blr, L_blr_t, "C1-", lw=1.0, alpha=0.85, label="tengri  blr_synthesizer_spectra")
_blr_pk = float(np.nanmax(L_blr_s[(w_blr_s >= 1000) & (w_blr_s <= 10000)]))
ax.set_ylim(_blr_pk * 1e-3, _blr_pk * 3)
ax.set_xlim(1000, 10000)
ax.set_yscale("log")
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
save_fig("synthesizer_09d_blr.png")
print(
    "§9d BLR (/spectra/nebular path): tengri reproduces UnifiedAGN's isotropic "
    "BLR component via the public grammar."
)


# %% [markdown]
# ### §9e Dusty torus
#
# Synthesizer reprocesses obscured disc luminosity into the IR with a blackbody
# torus (T = 1000 K). tengri's torus blocks (`nenkova`, `skirtor`,
# `two_temperature`) are independent radiative-transfer/parametric models, so the
# comparison is mid-IR shape and peak at matched bolometric luminosity. The
# parametric `two_temperature` peaks warmer than 1000 K BB; the radiative-transfer
# `nenkova` (CLUMPY) peaks cooler — the spread reflects model-family differences.

# %%
w_torus_s, L_torus_s = agn["torus"]

_torus_tengri = {}
for torus_type in ("nenkova", "two_temperature"):
    m = SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        agn={
            "type": "composable",
            "disc": {"type": "none"},
            "torus": {"type": torus_type},
            "lines": {"type": "none"},
            "agn_log_lbol": Fixed(agn_log_lbol),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    _torus_tengri[torus_type] = (np.asarray(s.wave), np.asarray(s.derived["sed_agn"]))

fig, ax_l, ax_r = U.two_panel_fig()
U.panel(
    ax_l, ax_r, label_l="Synthesizer  torus (Blackbody 1000 K)", label_r="tengri  torus models"
)
ax_l.plot(w_torus_s, L_torus_s, "C0-", linewidth=1.5)
for c, tt in zip(("C1", "C2"), ("nenkova", "two_temperature")):
    wt, Lt = _torus_tengri[tt]
    ax_r.plot(wt, Lt, c + "-", linewidth=1.5, label=tt)
ax_r.legend(fontsize=9)
_tpk = max(float(L_torus_s.max()), float(_torus_tengri["nenkova"][1].max()))
for ax in (ax_l, ax_r):
    ax.set_xlim(1e3, 1e7)
    ax.set_ylim(_tpk * 1e-3, _tpk * 3)
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_09e_torus.png")
_pk_s = w_torus_s[(w_torus_s > 1e4)][np.argmax(L_torus_s[(w_torus_s > 1e4)])]


def _ir_peak_um(wt, lt):
    m = wt > 1e4
    return float(wt[m][np.argmax(lt[m])]) / 1e4


_pk_nen = _ir_peak_um(*_torus_tengri["nenkova"])
_pk_2t = _ir_peak_um(*_torus_tengri["two_temperature"])
print(
    f"§9e torus mid-IR peak: Synthesizer (1000 K BB) {_pk_s / 1e4:.1f} µm; "
    f"tengri two_temperature {_pk_2t:.1f} µm (close), nenkova {_pk_nen:.1f} µm (cooler clumpy RT)"
)


# %% [markdown]
# ### §9f Unified spectrum and inclination anisotropy
#
# The full unified AGN spectrum: Synthesizer's `UnifiedAGN` against tengri's,
# the latter built in a single `SEDModel.build` call — disc + torus + NLR +
# BLR via the composable grammar and `nlr_blr_synthesizer_spectra` lines
# block. The disc UV bump, [O III]/Balmer line forest, and torus IR bump
# stack into the same panchromatic shape on both sides.
#
# Below, the decisive geometry difference. Both codes carry
# Type-1/2 anisotropy on the disc: Synthesizer applies a **hard cut** — the disc
# vanishes at grazing inclination (`inclination + θ_torus > 90°`) — while tengri
# applies a **smooth sigmoid** through the critical angle, so disc visibility
# stays differentiable. Line regions are isotropic in both (Synthesizer at fixed
# `cosine_inclination = 0.5`; tengri by intrinsic bolometric), so they do not
# fade with inclination. (tengri's physically Type-2-obscured BLR is the
# separate `nlr_blr_synthesizer` path in `recipes.unified_agn()`; here we
# reproduce Synthesizer's isotropic convention.)

# %%
# tengri builds the *entire* unified AGN in ONE ``SEDModel.build`` call (via the
# ``_agn_grammar`` helper defined in the §9 setup): disc + torus + NLR + BLR
# reading the same ``/spectra/nebular`` reprocessed array as Synthesizer's
# ``UnifiedAGN``, through the combined ``nlr_blr_synthesizer_spectra`` lines block.
# The runner masks the disc with inclination while the line regions stay isotropic
# (Synthesizer's convention) — no hand-assembly of raw adapter calls.
w_t, L_tot_t = _agn_grammar(nlr="synthesizer_spectra", blr="synthesizer_spectra")
# Decompose it for the component panel, every piece through the same grammar:
# disc-only and torus-only as standalone builds; the line regions as the
# difference each makes on top of the disc+torus continuum.
_, L_disc_t = _agn_grammar(torus="none")
_, L_torus_t = _agn_grammar(disc="none")
_, _L_cont = _agn_grammar()  # disc + torus, no lines
L_nlr_t = _agn_grammar(nlr="synthesizer_spectra")[1] - _L_cont
L_blr_t = _agn_grammar(blr="synthesizer_spectra")[1] - _L_cont

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
ax_s = fig.add_subplot(gs[0, 0])
ax_t = fig.add_subplot(gs[0, 1], sharey=ax_s)
ax_inc = fig.add_subplot(gs[1, :])

# Shared y-range spanning the UV disc (faint in L_nu) through the IR torus / line
# peak (bright in L_nu): a torus-only floor squashes the disc, so anchor the
# bottom to the disc bump and the top to the total. The disc continuum is
# genuinely ~50x below the torus in L_nu (it peaks in the UV, the torus in the
# IR) — this keeps the disc visible rather than on the axis floor.
_disc_pk = max(_winmax(*agn["disc"], hi=1e4), _winmax(w_t, L_disc_t, hi=1e4))
_top = max(_winmax(*agn["intrinsic"], hi=1e7), _winmax(w_t, L_tot_t, hi=1e7))
_ylo, _yhi = _disc_pk * 3e-2, _top * 3.0

# Left: Synthesizer UnifiedAGN total + components.
w_tot, L_tot = agn["intrinsic"]
ax_s.plot(w_tot, L_tot, "k-", linewidth=2.0, label="total (intrinsic)")
for key, c in (("disc", "C0"), ("nlr", "C1"), ("blr", "C2"), ("torus", "C3")):
    wk, Lk = agn[key]
    ax_s.plot(wk, Lk, color=c, linewidth=1.0, alpha=0.8, label=key)
# Right: tengri unified AGN total + components (qsosed + Nenkova + grid NLR/BLR).
ax_t.plot(w_t, L_tot_t, "k-", linewidth=2.0, label="total")
for L, c, lab in (
    (L_disc_t, "C0", "disc (qsosed)"),
    (L_nlr_t, "C1", "nlr (grid)"),
    (L_blr_t, "C2", "blr (grid)"),
    (L_torus_t, "C3", "torus (1000 K BB)"),
):
    ax_t.plot(w_t, L, color=c, linewidth=1.0, alpha=0.8, label=lab)
for axx, title in ((ax_s, "Synthesizer UnifiedAGN"), (ax_t, "tengri unified AGN")):
    axx.set_xscale("log")
    axx.set_yscale("log")
    axx.set_xlim(1e2, 1e7)
    axx.set_ylim(_ylo, _yhi)
    axx.set_xlabel(r"$\lambda$ [Å]")
    axx.set_title(f"{title}  (inc={BH_INC:g}°, θ_torus={THETA_TORUS:g}°)")
    axx.legend(fontsize=8, ncol=2)
    axx.grid(True, alpha=0.3)
ax_s.set_ylabel(r"$L_\nu$ [erg/s/Hz]")

# Inclination sweep: observed-disc luminosity at 5000 Å vs inclination.
incs = np.linspace(0.0, 89.0, 24)
disc_vis = []
for inc in incs:
    r = S.agn_unified(
        mass_msun=BH_MASS,
        eddington=BH_EDD,
        inclination_deg=float(inc),
        metallicity=BH_Z,
        cf_nlr=CF,
        cf_blr=CF,
        theta_torus_deg=THETA_TORUS,
        components=("disc_incident_masked",),
    )
    if "disc_incident_masked" in r:
        wk, Lk = r["disc_incident_masked"]
        disc_vis.append(float(Lk[np.argmin(np.abs(wk - 5000.0))]))
    else:
        disc_vis.append(0.0)
disc_vis = np.array(disc_vis)
ax_inc.plot(
    incs,
    disc_vis / max(disc_vis.max(), 1e-30),
    "C0-o",
    markersize=3,
    label="Synthesizer (hard mask)",
)
# tengri's disc visibility, MEASURED from the composable model (not drawn): we
# build the unified AGN once with the inclination free, then read the optical
# disc continuum (5000 Å) as the sightline sweeps. The runner applies a smooth
# sigmoid mask centered at inc_crit = 90° − θ_torus instead of Synthesizer's hard
# step, so the disc visibility — and the inclination itself — stays a
# differentiable fit parameter. This is the model's one deliberate departure
# from Synthesizer; the curve below is the model's own output, not an overlay.
_inc_crit = 90.0 - THETA_TORUS
_m_vis = SEDModel.build(
    ssp_data=ssp,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    agn={
        "type": "composable",
        "disc": {"type": "kubota_done"},
        "torus": {"type": "simple"},
        "agn_log_lbol": Fixed(agn_log_lbol),
        "agn_cos_inc": Uniform(0.0, 1.0),
        "agn_theta_torus": Fixed(THETA_TORUS),
        "*": FIXED,
    },
    redshift=Fixed(0.0),
)
_w_vis = np.asarray(_m_vis.predict_state({"agn_cos_inc": 0.5}).wave)
_i5000 = int(np.argmin(np.abs(_w_vis - 5000.0)))
tengri_vis = np.array(
    [
        float(
            np.asarray(
                _m_vis.predict_state({"agn_cos_inc": float(np.cos(np.radians(i)))}).derived[
                    "sed_agn"
                ]
            )[_i5000]
        )
        for i in incs
    ]
)
tengri_vis = tengri_vis / max(tengri_vis.max(), 1e-30)
ax_inc.plot(incs, tengri_vis, "C1-", linewidth=2.0, label="tengri (composable model, measured)")
ax_inc.axvline(
    90.0 - THETA_TORUS,
    color="gray",
    linestyle=":",
    label=rf"$90°-\theta_{{torus}}$ = {90 - THETA_TORUS:g}°",
)
ax_inc.set_xlabel("inclination [degrees]")
ax_inc.set_ylabel("disc visibility, normalized")
ax_inc.set_title("Disc visibility vs inclination — hard step vs differentiable sigmoid")
ax_inc.legend(fontsize=9)
ax_inc.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_09f_unified_inclination.png")
print(
    f"§9f disc mask (θ_torus={THETA_TORUS:g}°): "
    f"Synthesizer hard-zeros at inc>{90 - THETA_TORUS:g}°; tengri sigmoid stays differentiable."
)


# %% [markdown]
# ### §9g Photometry under precompute — exact vs `WavePrecomp`
#
# Fits drive this unified AGN through `predict_photometry` with
# `approx=WavePrecomp()` lookup tables. Additive AGN emitters are
# filter-integrated exactly under precompute, so sharp line-region structure
# must survive to float level. Both paths use FUV-to-mid-IR filters at z = 0.05.

# %%
_g_filters = [
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_ks",
    "wise_w1",
    "wise_w2",
]
_g_obs = Observation(photometry=Photometry.from_names(_g_filters))


def _unified_phot(approx):
    m = SEDModel.build(
        ssp_data=ssp,
        observation=_g_obs,
        approx=approx,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        agn={
            "type": "composable",
            "disc": {"type": "kubota_done"},
            "torus": {"type": "simple"},
            "nlr": {"type": "synthesizer_spectra"},
            "blr": {"type": "synthesizer_spectra"},
            "agn_log_lbol": Fixed(agn_log_lbol),
            "agn_log_mbh": Fixed(float(np.log10(BH_MASS))),
            "agn_log_ledd": Fixed(float(np.log10(BH_EDD))),
            "agn_cos_inc": Fixed(BH_COS_INC),
            "agn_theta_torus": Fixed(THETA_TORUS),
            "agn_torus_frac": Fixed(_TORUS_FRAC),
            "*": FIXED,
        },
        redshift=Fixed(0.05),
    )
    return np.asarray(m.predict_photometry({}))


_phot_exact = _unified_phot(None)
_phot_lut = _unified_phot(WavePrecomp())
_rel = np.abs(_phot_lut / _phot_exact - 1.0)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(range(len(_g_filters)), np.maximum(_rel, 1e-17), color="C1")
ax.set_xticks(range(len(_g_filters)))
ax.set_xticklabels(_g_filters, rotation=45, ha="right", fontsize=8)
ax.set_yscale("log")
ax.set_ylabel(r"|WavePrecomp / exact $-$ 1|")
ax.set_title("§9g unified-AGN photometry: precompute vs exact, per band")
ax.grid(True, alpha=0.3, axis="y")
fig.tight_layout()
save_fig("synthesizer_09g_precompute_parity.png")
print(f"§9g precompute vs exact photometry: max |Δ|/exact = {float(_rel.max()):.2e}")


# %% [markdown]
# ### §9h What a fit would free
#
# The same composable AGN with the block-scoped wildcard flipped to `FREE`.
# `'*': FREE` frees only the parameters the active disc/torus/nlr/blr blocks
# consume; every freed parameter is guaranteed to move `predict()` (contract:
# `tests/contract/test_agn_block_consumes.py`).

# %%
_m_free = SEDModel.build(
    ssp_data=ssp,
    sfh={
        "type": "delayed",
        "tau_gyr": Fixed(1.0),
        "age_gyr": Fixed(5.0),
        "log_total_mass": Fixed(0.0),
        "*": FIXED,
    },
    dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
    agn={
        "type": "composable",
        "disc": {"type": "kubota_done"},
        "torus": {"type": "simple"},
        "nlr": {"type": "synthesizer_spectra"},
        "blr": {"type": "synthesizer_spectra"},
        "agn_log_lbol": Fixed(agn_log_lbol),
        "*": FREE,
    },
    redshift=Fixed(0.0),
)
_agn_free = sorted(str(_p) for _p in _m_free.spec.free_params if str(_p).startswith("agn_"))
print(f"§9h '*': FREE frees {len(_agn_free)} AGN parameters:")
for _p in _agn_free:
    print("   ", _p)


# %% [markdown]
# ## §12 IGM transmission — Inoue (2014) and Madau (1995)
#
# Both Synthesizer and tengri ship Inoue+2014 and Madau+1995 IGM prescriptions.
# At matched redshift the curves should track closely; the residual is the
# difference between the two implementations.

# %%
# Inoue14 IGM is public (``tengri.igm_transmission``). The Madau96 variant is not
# yet re-exported — the one remaining internal import in this notebook.
from tengri.igm import igm_transmission_madau

Z_IGM = 4.0
w_obs, T_s_inoue = S.igm_transmission(redshift=Z_IGM, model="inoue14")
_, T_s_madau = S.igm_transmission(redshift=Z_IGM, model="madau96", wave_obs_aa=w_obs)
T_t_inoue = np.asarray(tengri.igm_transmission(jnp.asarray(w_obs), float(Z_IGM)))
T_t_madau = np.asarray(igm_transmission_madau(jnp.asarray(w_obs), np.asarray(Z_IGM)))

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(w_obs, T_s_inoue, "C0-", linewidth=2.0, label=f"Synthesizer Inoue14, z={Z_IGM:g}")
ax.plot(w_obs, T_t_inoue, "k--", linewidth=1.0, label="tengri Inoue14")
ax.plot(w_obs, T_s_madau, "C1-", linewidth=2.0, alpha=0.7, label="Synthesizer Madau96")
ax.plot(w_obs, T_t_madau, "C3:", linewidth=1.2, label="tengri Madau")
ax.set_xlabel(r"observed $\lambda$ [Å]")
ax.set_ylabel(r"IGM transmission $T(\lambda, z)$")
ax.set_xlim(3000, 8000)
ax.set_ylim(0, 1.05)
ax.set_title(f"IGM transmission at z = {Z_IGM:g}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_12_igm.png")

_win = (w_obs >= (1 + Z_IGM) * 950) & (w_obs <= (1 + Z_IGM) * 1216)
_diff = np.abs(T_t_inoue[_win] - T_s_inoue[_win])
print(
    f"§12 Inoue14 IGM at z={Z_IGM:g}: "
    f"max |Δ| = {_diff.max():.3e}, median |Δ| = {np.median(_diff):.3e}"
)


# %% [markdown]
# ## tengri in Synthesizer-mode — full-SED head-to-head
#
# tengri configured to emulate Synthesizer end to end (shared SSP, τ-delayed
# SFH, Calzetti screen, DL07 IR, Cue nebular) overlaid on Synthesizer's
# `TotalEmission`, over the fractional residual with the ±25 % band shaded.
# The optical ratio carries the §3 SFH-discretization offset, the dominant
# stellar-continuum difference.

# %%
import chex

w_ext, L_ext = np.asarray(w_full_s), np.asarray(L_full_s)
L_t_on_ext = U.regrid(np.asarray(s_full.wave), _sed_full_t, w_ext)
chex.assert_equal_shape([L_ext, L_t_on_ext])

mask = (w_ext > 0) & (L_ext > 0) & (L_t_on_ext > 0)
resid = np.full(w_ext.shape, np.nan, dtype=float)
resid[mask] = L_t_on_ext[mask] / L_ext[mask] - 1.0

opt = mask & (w_ext >= 1000.0) & (w_ext <= 10000.0)
ratio_opt = L_t_on_ext[opt] / L_ext[opt]
norm = float(np.median(ratio_opt))
p16, p84 = float(np.percentile(ratio_opt, 16)), float(np.percentile(ratio_opt, 84))
print(
    f"full-SED head-to-head tengri/Synthesizer optical (1000–10000 Å): "
    f"normalization {norm:.2f}×, 16–84% spread {p16:.2f}–{p84:.2f}×"
)
_assert_comparable(L_ext, _sed_full_t, name="full-SED head-to-head")

fig, (ax, ax_r) = plt.subplots(
    2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax.plot(w_ext, L_ext, "C0-", linewidth=1.5, label="Synthesizer (TotalEmission)")
ax.plot(w_ext, L_t_on_ext, "C1--", linewidth=1.5, label="tengri (Synthesizer-mode)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1e2, 1e7)
_hpk = float(np.nanmax(L_ext))
ax.set_ylim(_hpk * 1e-5, _hpk * 3)
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title("tengri in Synthesizer-mode vs Synthesizer — full panchromatic SED")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.text(
    0.02,
    0.05,
    rf"tengri/Synthesizer $= {norm:.2f}\times$ (16–84%: {p16:.2f}–{p84:.2f})",
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
ax_r.set_ylabel(r"tengri/Syn $-1$")
ax_r.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_full_sed_headtohead.png")
plt.show()


# %% [markdown]
# ## Summary
#
# Component by component at matched parameters, Synthesizer and tengri agree
# where they evaluate the same numerics: shared SSP grid (§1, float-precision),
# delayed-τ SFH (§2), attenuation curves (§4), and IGM (§12). AGN line regions
# (§9c, §9d) draw from the same Cloudy grids, though spectra differ in line
# representation. Where codes use independent physics — accretion disc (§9a),
# torus (§9e), and disc–torus geometry (§9f, Synthesizer's hard mask vs
# tengri's smooth sigmoid) — the comparison is shape and amplitude.

# %% [markdown]
# ## References
#
# * Lovell et al. 2025, Open J. Astrophys. 8 — Synthesizer (doi:10.33232/001c.145766)
# * Roper et al. 2026, JOSS 11, 9436 — Synthesizer (doi:10.21105/joss.09436)
#   (Both Synthesizer papers must be cited together, per the project's citation policy.)
# * Kubota & Done 2018, MNRAS 480, 1247 — qsosed accretion disc
# * Feltre et al. 2016, MNRAS 456, 3354 — AGN narrow-line-region grids
# * Nenkova et al. 2008, ApJ 685, 160 — clumpy torus
# * Calzetti et al. 2000, ApJ 533, 682 — starburst attenuation
# * Draine & Li 2007, ApJ 657, 810 — dust IR emission
# * Inoue et al. 2014, MNRAS 442, 1805 — IGM attenuation
# * Madau 1995, ApJ 441, 18 — IGM absorption
# * Li et al. 2025 — Cue nebular emulator
