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
# # Reproducing Synthesizer with tengri
#
# Synthesizer (Roper, Lovell, Wilkins et al. 2025) is a modular forward-modelling
# package for synthetic galaxy observables. It builds spectra by extracting from
# pre-computed HDF5 grids and walking an `EmissionModel` tree — stellar SSPs, a
# parametric star formation history, attenuation and dust-emission laws, the
# Madau / Inoue IGM, and the `UnifiedAGN` black-hole emission model (disc + narrow
# and broad line regions + dusty torus, combined with the disc–torus geometry).
#
# This notebook places that forward model next to tengri on the same axes, in the
# same units, at matched parameters — section by section. tengri and Synthesizer
# are **independent codes** with independent implementations; this is a
# peer-to-peer comparison, exactly like the CIGALE, Prospector, and BAGPIPES
# reproduction notebooks. The emphasis here is the **AGN** model (§9), reproduced
# component by component.
#
# **What sits on each side.** The left panel of every figure is Synthesizer; the
# right panel is tengri. For the stellar populations (§1) both sides read the
# *same* templates — Synthesizer's stellar grid is re-shaped into the form tengri
# reads (a one-off step described in the README) — so the §1 residual is
# interpolation alone, not a different spectral library. For the AGN line regions
# (§9c, §9d) tengri reads the *same* Synthesizer photoionisation grids, so the
# line lists and line luminosities have a common origin; the spectra still differ
# in how each code spreads a line over wavelength (see §9c).
#
# **Grids.** This notebook runs on Synthesizer's *test* grids
# (`synthesizer-download --stellar-test-grids --agn-test-grids --dust-grid`).
# The test grids are the same physics as the production grids at lower resolution
# (the AGN test grids sample each photoionisation axis at just two nodes); the
# coarseness is a property of both sides equally, so it never appears as a
# tengri-vs-Synthesizer disagreement.
#
# Synthesizer has no X-ray or radio component, so those sections are omitted (the
# CIGALE notebook covers the panchromatic X-ray / radio stack).

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
    synthesizer_ssp_to_dsps as PORT,
    units as U,
)

import tengri
from tengri import FIXED, Fixed, SEDModel, load_ssp_data

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
# tengri reads Synthesizer's stellar populations, re-shaped once into the form
# tengri loads (see the README); the re-shaped file is cached locally and not
# committed. Both codes then read identical templates, so the §1 agreement below
# is a numerical statement, not a physical one.

# %%
_grid_h5 = _HERE / "_drivers" / "data" / "synthesizer_test_grid.h5"
if not _grid_h5.exists():
    PORT.port_stellar_grid(_grid_h5)
ssp = load_ssp_data(str(_grid_h5))
print(
    f"Synthesizer test_grid (ported): {ssp.ssp_wave.shape[0]} wavelengths, "
    f"{ssp.ssp_lgmet.shape[0]} metallicities, {ssp.ssp_lg_age_gyr.shape[0]} age bins; "
    f"λ up to {ssp.ssp_wave.max():.1e} Å."
)


# %% [markdown]
# ## §1 Single stellar populations
#
# Single stellar populations at ≈ solar metallicity, from 1 Myr to 10 Gyr,
# overlaid: Synthesizer's spectrum (solid) against the same templates as read by
# tengri (black dashed). The lower panel shows the relative residual
# `|tengri − Synthesizer| / Synthesizer`. Both sides read identical numbers, so
# the residual sits at the level set by the single-precision round-trip through
# the shared grid (grey line, 1e-6) — there is no spectral-library difference to
# see here.

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
ax_r.axhline(1e-6, color="grey", linestyle=":", alpha=0.6, label="single-precision floor")
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
# Synthesizer's `SFH.DelayedExponential` uses `SFR(t) ∝ t · exp(−t/τ)`, peaking at
# `t = τ`. tengri's `sfh.delayed` is the same closed form. Both normalise to 1 M⊙
# formed (Synthesizer via `initial_mass`, tengri via `log_total_mass`).
#
# **What the right panel plots.** Not a fine-grid analytic curve — that would
# compare two formulas. It reads the star formation history straight off a built
# tengri model, on the log-spaced lookback grid the model actually uses (hence
# the slightly jagged look). The printed `∫SFR dt` confirms it integrates to
# 1 M⊙ formed.

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
    ax.axvline(TAU_GYR_FIDUCIAL, color="grey", linestyle=":", alpha=0.6)
ax_l.plot(t_s_cosmic_gyr, sfr_s, "C0-", linewidth=2.0, label=rf"$\tau$ = {TAU_GYR_FIDUCIAL:g} Gyr")
ax_l.legend(fontsize=9)
ax_r.plot(t_t_cosmic_gyr, _sfr_history, "C1-", linewidth=2.0)
fig.tight_layout()
save_fig("synthesizer_02_sfh_delayed.png")


# %% [markdown]
# ## §3 Integrated stellar SED
#
# Convolve the τ-delayed SFH with the SSP grid. No dust, no nebular. Both panels
# show `L_ν` vs `λ_rest` for the 10^10 M⊙ fiducial galaxy.
#
# Both sides form exactly 10^10 M⊙ and read the same SSPs (§1), so the printed
# ~1.16× optical ratio is *not* a normalisation error — it is the two codes'
# independent **SFH discretisations**: Synthesizer integrates
# an analytic SFZH onto the SSP age-bin edges (a third of the mass lands in the
# wide log-spaced bin straddling the SFR peak), while tengri convolves on its
# 64-point log-lookback grid. The mild chromatic spread (P5–P95 ≈ 1.11–1.19)
# follows from the slightly different age weighting, not a shape disagreement.

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
# Floor the y-range to the stellar continuum: the ported grid is clipped at 1e8 Å,
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
# Synthesizer's `Calzetti2000` and `PowerLaw` attenuation laws (and a Calzetti law
# with the 2175 Å Drude bump enabled) against tengri's `calzetti`, `power_law`,
# and bump-modified laws. Both sides evaluate the analytic law directly, normalised
# to `A(λ)/A_V` at 5500 Å — curve against curve, no SSP-convolution noise.

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
# The fiducial galaxy with and without dust. Synthesizer applies a Calzetti screen
# at A_V = 1; tengri matches with the full A_V on the diffuse component (which
# attenuates all ages equally) and the birth-cloud term off — the single-screen
# mapping established in the Prospector / Bagpipes notebooks.

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
# Absorbed stellar UV/optical reappears in the infrared. Synthesizer's
# `TotalEmission` tree attenuates the reprocessed spectrum with a Calzetti screen
# and re-emits the absorbed luminosity through the Draine & Li (2007) templates,
# enforcing energy balance internally (`set_energy_balance`). tengri uses its own
# DL07 grid with the same energy-balance constraint. We compare the isolated dust
# IR emission at matched `(q_PAH, U_min)`; both peak in the far-IR near ~140 µm
# for `U_min = 1`.

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
# Stellar + nebular + dust attenuation + DL07 IR on one axis, from the rest-UV to
# the far-IR. The percent-level disagreements of §3–§6 (and the larger nebular gap
# of §8) stack here; the headline is the overall shape. (The Synthesizer test grid
# carries a coarse far-IR nebular-continuum feature that lifts the mid-IR on the
# Synthesizer side — a property of the 2-node test grid, not a physics
# disagreement.)

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
# Synthesizer's nebular emission comes from the Cloudy grid baked into the SSP
# grid (`NebularEmission`, lines + continuum). tengri's emitter is Cue (Li et al.
# 2025), a neural emulator trained on a different Cloudy version with a different
# ionising-spectrum parametrisation. **They will not agree**, and the gap is the
# point — it reflects the Cloudy version and the different convolution paths. The
# panel quantifies the Hα ratio rather than hiding it.

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
_s_ha = float(L_neb_s[np.argmin(np.abs(w_neb_s - 6563))])
_t_ha = float(L_neb_t[int(np.argmin(np.abs(np.asarray(s_neb.wave) - 6563)))])
if _s_ha > 0:
    print(f"§8 Hα (6563 Å) tengri Cue / Synthesizer Cloudy = {_t_ha / _s_ha:.2f}×")
fig.tight_layout()
save_fig("synthesizer_08_nebular.png")


# %% [markdown]
# ## §9 AGN — the Unified AGN model
#
# This is the focus of the notebook. Synthesizer's `UnifiedAGN` combines four
# regions — accretion **disc**, **narrow-line region** (NLR), **broad-line region**
# (BLR), and dusty **torus** — and folds in the disc–torus geometry (an
# inclination-dependent mask). We reproduce each piece against tengri, then the
# combined spectrum and its inclination dependence.
#
# Under the hood the model is one idea applied to each region. The AGN Cloudy
# grid stores the disc continuum three ways — the **incident** spectrum (the bare
# disc), the **transmitted** spectrum (disc light that passes through the gas),
# and the **nebular** spectrum (the lines and continuum the gas re-emits). For
# the NLR and BLR a covering fraction sets how much of the disc is reprocessed
# (transmitted + nebular) versus escapes untouched; the energy the gas absorbs
# comes back out as emission lines. The torus does the same in the infrared,
# reprocessing the disc luminosity it intercepts into warm-dust emission. The
# geometry then closes the model: for a sightline grazing the torus edge
# (`inclination + θ_torus > 90°`) the disc and BLR are hidden (a Type-2 view),
# while the NLR — sitting outside the torus — stays visible at every angle. That
# incident/transmitted/escaped accounting, repeated per region and gated by the
# inclination mask, is what fans the model out into the components below (§9a–§9f).
#
# The Synthesizer black hole is driven by `(mass, accretion_rate_eddington)` (the
# grid axes), with `inclination` and `theta_torus` setting the geometry. Its
# bolometric luminosity maps to tengri's `agn_log_lbol = log10(L_bol / L⊙)`.

# %%
import jax.numpy as jnp

from tengri.components.agn.nlr_cloudy import compute_nlr_sed_synthesizer

# Synthesizer grid directory (where synthesizer-download placed the test grids),
# so tengri's adapter reads the *same* AGN Cloudy grids.
_SYN_GRID_DIR = os.environ.get(
    "SYNTHESIZER_GRID_DIR",
    os.path.expanduser("~/Library/Application Support/Synthesizer/grids"),
)
_NLR_GRID_PATH = str(Path(_SYN_GRID_DIR) / "test_grid_agn-nlr.hdf5")

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


# %% [markdown]
# ### §9a Accretion disc
#
# These AGN grids were generated with Synthesizer's **qsosed** disc (Kubota &
# Done 2018) — so the matched comparison is against tengri's `kubota_done` disc.
# We also overlay tengri's broken power-law disc (the form the Feltre et al. 2016
# grids assume) for context. The codes use independent disc implementations, so
# this is a comparison of shape at matched bolometric luminosity, not an exact
# match.

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
# fractions into the observed disc. We show all three to expose the geometry.

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
# Here tengri reads the *same* narrow-line-region grid as Synthesizer — the same
# 215 Cloudy lines, and the disc's own ionising luminosity straight from the grid
# (rather than an assumed ionising-spectrum slope), so the normalisation is
# Synthesizer's own. Two things keep the curves from being identical, both by
# design. First, Synthesizer plots the full reprocessed emission (nebular
# continuum plus lines spread over the grid's native bins), while tengri returns
# the lines alone, each narrowed to the ~500 km/s width of the narrow-line region
# — concentrating a line into that profile lifts its peak above the grid-binned
# version (so each panel has its own y-axis). Second, tengri interpolates the
# coarse test grid with a smooth, differentiable kernel rather than a step
# lookup, the same gradient-friendly choice as the inclination mask in §9f; on a
# 2-node-per-axis test grid that smooths the line ratios by tens of percent, an
# offset that shrinks on a finer grid. The strong forbidden lines — [O III] 5007,
# the Balmer series, [O II] 3727 — still line up. (On the downloadable test grid
# the NLR and BLR files are identical placeholders — see the §9d caveat.)

# %%
w_nlr_s, L_nlr_s = agn["nlr"]
_wave_nlr = jnp.asarray(np.logspace(2.7, 6.0, 4000))
L_nlr_t = np.asarray(
    compute_nlr_sed_synthesizer(
        _wave_nlr,
        l_disc_bol_erg=agn["_bolometric_erg_s"],
        covering_fraction=CF,
        grid_path=_NLR_GRID_PATH,
        neb_logU=-2.0,
        neb_logZ_gas=float(np.log10(BH_Z)),
    )
)
_wave_nlr = np.asarray(_wave_nlr)

# Independent y-axes: the Gaussian-convolved line peaks and the grid-binned
# spectrum live on very different L_ν scales (see the markdown above).
fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))
U.panel(
    ax_l,
    ax_r,
    label_l="Synthesizer  NLR (full reprocessed: lines + continuum)",
    label_r="tengri  NLR lines (same grid, 500 km/s)",
)
ax_l.plot(w_nlr_s, L_nlr_s, "C0-", linewidth=1.0)
ax_r.plot(_wave_nlr, L_nlr_t, "C1-", linewidth=1.0)
# Floor the log-y axis at a few dex below each panel's peak so the lines are
# legible (the line spectrum is ~0 between lines → −inf on a raw log axis).
_nlr_pk_s = float(np.nanmax(L_nlr_s[(w_nlr_s >= 1000) & (w_nlr_s <= 10000)]))
_nlr_pk_t = float(np.nanmax(L_nlr_t[(_wave_nlr >= 1000) & (_wave_nlr <= 10000)]))
ax_l.set_ylim(_nlr_pk_s * 1e-3, _nlr_pk_s * 3)
ax_r.set_ylim(_nlr_pk_t * 1e-4, _nlr_pk_t * 3)
for ax in (ax_l, ax_r):
    ax.set_xlim(1000, 10000)
    ax.set_xscale("linear")
    ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("synthesizer_09c_nlr.png")
# Identify the strongest tengri NLR line in the optical-UV window.
_optw = (_wave_nlr >= 1000) & (_wave_nlr <= 10000)
_pk_line = _wave_nlr[_optw][np.argmax(L_nlr_t[_optw])]
print(
    f"§9c NLR: strongest tengri line at {_pk_line:.1f} Å (from the shared Synthesizer Cloudy grid)"
)


# %% [markdown]
# ### §9d Broad-line region
#
# The same comparison for the broad-line region, with tengri reading the
# Synthesizer broad-line grid. Broad-line widths are far larger (~5000 km/s), so
# the permitted lines blend into the quasar-like pseudo-continuum seen in Type-1
# AGN — Lyα, C IV, the Balmer lines — rather than the sharp forbidden lines of §9c.
#
# **Caveat — the test grids don't yet distinguish NLR from BLR.** Synthesizer's
# downloadable `test_grid_agn-nlr` and `test_grid_agn-blr` are *byte-identical*
# placeholders: the BLR file is a copy of the NLR file. So on these grids the
# Synthesizer NLR (§9c, left) and BLR (here, left) panels are the *same* spectrum,
# and the only thing that distinguishes tengri's NLR from its BLR is the velocity
# width applied (500 vs 5000 km/s) — which is why the right-hand panels differ in
# line *width* but share line *positions*. A genuine NLR-vs-BLR physical contrast
# (density, ionisation, line ratios) needs the production grids.

# %%
try:
    from tengri.components.agn.nlr_cloudy import compute_blr_sed_synthesizer

    _BLR_GRID_PATH = str(Path(_SYN_GRID_DIR) / "test_grid_agn-blr.hdf5")
    w_blr_s, L_blr_s = agn["blr"]
    _wave_blr = jnp.asarray(np.logspace(2.7, 6.0, 4000))
    L_blr_t = np.asarray(
        compute_blr_sed_synthesizer(
            _wave_blr,
            l_disc_bol_erg=agn["_bolometric_erg_s"],
            covering_fraction=CF,
            grid_path=_BLR_GRID_PATH,
            neb_logU=-1.0,
            neb_logZ_gas=float(np.log10(BH_Z)),
        )
    )
    _wave_blr = np.asarray(_wave_blr)
    # Independent y-axes (as in §9c): the broad-Gaussian tengri spectrum and the
    # grid-binned Synthesizer spectrum live on different L_ν scales.
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 4.5))
    U.panel(
        ax_l,
        ax_r,
        label_l="Synthesizer  BLR (full reprocessed)",
        label_r="tengri  BLR lines (same grid, 5000 km/s)",
    )
    ax_l.plot(w_blr_s, L_blr_s, "C0-", linewidth=1.0)
    ax_r.plot(_wave_blr, L_blr_t, "C1-", linewidth=1.0)
    _blr_pk_s = float(np.nanmax(L_blr_s[(w_blr_s >= 1000) & (w_blr_s <= 10000)]))
    _blr_pk_t = float(np.nanmax(L_blr_t[(_wave_blr >= 1000) & (_wave_blr <= 10000)]))
    ax_l.set_ylim(_blr_pk_s * 1e-3, _blr_pk_s * 3)
    ax_r.set_ylim(_blr_pk_t * 1e-2, _blr_pk_t * 3)
    for ax in (ax_l, ax_r):
        ax.set_xlim(1000, 10000)
        ax.set_xscale("linear")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig("synthesizer_09d_blr.png")
    print("§9d BLR: tengri compute_blr_sed_synthesizer rendered (broad 5000 km/s lines).")
except ImportError:
    print(
        "§9d BLR: tengri.compute_blr_sed_synthesizer not yet available — pending the "
        "grid-backed BLR adapter (library follow-up). Showing Synthesizer BLR alone."
    )
    w_blr_s, L_blr_s = agn["blr"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(w_blr_s, L_blr_s, "C0-", linewidth=1.0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1e3, 1e6)
    ax.set_xlabel(r"$\lambda$ [Å]")
    ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
    ax.set_title("Synthesizer BLR (tengri grid adapter pending)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    save_fig("synthesizer_09d_blr.png")


# %% [markdown]
# ### §9e Dusty torus
#
# Synthesizer reprocesses the obscured disc luminosity into the IR with a
# blackbody torus (here T = 1000 K). tengri's torus blocks (`nenkova`, `skirtor`,
# `two_temperature`) are independent radiative-transfer / parametric models, so we
# compare the mid-IR shape and peak at matched bolometric luminosity.

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
# The full unified AGN spectrum, assembled. The **top row** puts Synthesizer's
# `UnifiedAGN` (left) next to tengri's own unified AGN (right): the same model
# pieces on both sides — the qsosed accretion disc, a 1000 K blackbody torus
# (Synthesizer's `Blackbody`; tengri's single-temperature `simple` torus), and the
# NLR and
# BLR drawn from the *same* Synthesizer Cloudy grids as §9c/§9d (via the
# grid-backed `nlr_synthesizer` lines block). Each panel shows the total and its
# four components on a shared axis — the disc UV bump, the line forest, and the
# torus IR bump stack into the same broad shape on both sides; the line spikes are
# narrow-Gaussian on tengri's side and grid-binned on Synthesizer's, as in §9c.
#
# The **bottom panel** is the decisive geometry: Synthesizer applies a **hard**
# cut — the disc and broad-line region vanish the moment the sightline grazes the
# torus edge (inclination + θ_torus > 90°), the Type-1 → Type-2 transition — while
# tengri uses a **smooth sigmoid** through the *same* critical angle, keeping disc
# visibility a differentiable function of inclination so it stays a stable fit
# parameter. The two agree everywhere except the few degrees around the
# transition. That is the single deliberate difference between the two AGN models.

# %%
from tengri.components.agn.nlr_cloudy import compute_blr_sed_synthesizer as _blr_syn


# Build tengri's own Unified AGN from the same pieces as the panels above:
# the qsosed disc and Nenkova torus through the composable builder, the NLR via
# the new grid-backed `nlr_synthesizer` lines block (#588) — i.e. the *same*
# Cloudy grid as §9c — and the BLR added from the matching grid. Each component
# is evaluated on the model's own wavelength grid so they sum cleanly.
def _tengri_agn_component(disc, torus, lines):
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
            "lines": {"type": lines},
            "agn_log_lbol": Fixed(agn_log_lbol),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.derived["sed_agn"])


w_t, L_disc_t = _tengri_agn_component("kubota_done", "none", "none")
# Match Synthesizer's torus model: a single-temperature blackbody at 1000 K.
# tengri's "simple" torus is a single-T greybody with agn_T_torus = 1000 K by
# default, so both panels reprocess the absorbed disc luminosity the same way.
_, L_torus_t = _tengri_agn_component("none", "simple", "none")
_blr_path = str(Path(_SYN_GRID_DIR) / "test_grid_agn-blr.hdf5")
L_nlr_t = np.asarray(
    compute_nlr_sed_synthesizer(
        jnp.asarray(w_t),
        l_disc_bol_erg=agn["_bolometric_erg_s"],
        covering_fraction=CF,
        grid_path=_NLR_GRID_PATH,
        neb_logU=-2.0,
        neb_logZ_gas=float(np.log10(BH_Z)),
    )
)
L_blr_t = np.asarray(
    _blr_syn(
        jnp.asarray(w_t),
        l_disc_bol_erg=agn["_bolometric_erg_s"],
        covering_fraction=CF,
        grid_path=_blr_path,
        neb_logU=-1.0,
        neb_logZ_gas=float(np.log10(BH_Z)),
    )
)
L_tot_t = L_disc_t + L_torus_t + L_nlr_t + L_blr_t

fig = plt.figure(figsize=(13, 9))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 2])
ax_s = fig.add_subplot(gs[0, 0])
ax_t = fig.add_subplot(gs[0, 1], sharey=ax_s)
ax_inc = fig.add_subplot(gs[1, :])

# Shared y-range from the smooth disc+torus continuum, with headroom for the
# narrow line spikes (so the continuum isn't squished into the floor).
_cont = max(
    _winmax(*agn["disc"], hi=1e7),
    _winmax(*agn["torus"], hi=1e7),
    _winmax(w_t, L_disc_t, hi=1e7),
    _winmax(w_t, L_torus_t, hi=1e7),
)
_ylo, _yhi = _cont * 5e-3, _cont * 50.0

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
# tengri replaces the hard step with a smooth sigmoid centred on the same
# critical angle, inc_crit = 90° − θ_torus, with a ~2° transition width. The
# sigmoid keeps the disc visibility — and therefore the inclination itself —
# a smoothly varying quantity, so it stays a well-behaved fit parameter rather
# than a flat-then-cliff one. This is the model's one deliberate departure from
# Synthesizer, and the curve below shows it side by side with the hard cut.
_inc_crit = 90.0 - THETA_TORUS
tengri_vis = 1.0 / (1.0 + np.exp((incs - _inc_crit) / 2.0))
ax_inc.plot(incs, tengri_vis, "C1-", linewidth=2.0, label="tengri (smooth sigmoid)")
ax_inc.axvline(
    90.0 - THETA_TORUS,
    color="grey",
    linestyle=":",
    label=rf"$90°-\theta_{{torus}}$ = {90 - THETA_TORUS:g}°",
)
ax_inc.set_xlabel("inclination [degrees]")
ax_inc.set_ylabel("disc visibility, normalised")
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
# ## §12 IGM transmission — Inoue (2014) and Madau (1995)
#
# Synthesizer ships both the Inoue+2014 and Madau+1995 IGM prescriptions
# (`emission_models.attenuation.Inoue14` / `Madau96`). tengri ships the same two.
# At matched redshift the curves should track closely; the residual is the
# difference between the two implementations.

# %%
from tengri.components.igm.igm import igm_transmission, igm_transmission_madau

Z_IGM = 4.0
w_obs, T_s_inoue = S.igm_transmission(redshift=Z_IGM, model="inoue14")
_, T_s_madau = S.igm_transmission(redshift=Z_IGM, model="madau96", wave_obs_aa=w_obs)
T_t_inoue = np.asarray(igm_transmission(jnp.asarray(w_obs), np.asarray(Z_IGM)))
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
# Every section above swept one block. This is the whole stellar+dust forward
# model at once: tengri configured to emulate Synthesizer end to end (the shared
# SSP, the fiducial τ-delayed SFH, a Calzetti screen, DL07 IR, and Cue nebular)
# overlaid on Synthesizer's own `TotalEmission` output at matched parameters (the
# §7 configuration). The top panel is the overlay; the bottom is the fractional
# residual `tengri / Synthesizer − 1` with the ±25 % band shaded. The optical
# normalization ratio and its 16–84 % spread are reported — they carry the §3
# SFH-discretisation offset, which is the dominant stellar-continuum difference.

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
# Component by component, at matched parameters, Synthesizer and tengri agree where
# they evaluate the same numerics: the shared SSP grid (§1, float-precision), the
# delayed-τ SFH (§2), the attenuation curves (§4), and the IGM (§12). The AGN line
# regions (§9c, §9d) draw their line lists and integrated luminosities from the
# same Cloudy grids, though the spectra differ in line representation (tengri's
# narrow Gaussians vs the grid bins). Where the codes use
# independent physics — the accretion disc (§9a), the torus (§9e), and the
# disc–torus geometry (§9f, Synthesizer's hard inclination mask vs tengri's smooth
# treatment) — the comparison is of shape and amplitude, and the differences are
# the expected ones between independent implementations. The §9 panels walk
# through every piece of the unified AGN model — disc, transmitted/escaped split,
# narrow- and broad-line regions, torus, and the inclination anisotropy — matching
# Synthesizer where the inputs are shared and showing exactly where, and why, the
# two part ways.

# %% [markdown]
# ## References
#
# * Roper, Lovell, Vijayan, Wilkins et al. 2025 — Synthesizer (arXiv:2508.03888)
# * Kubota & Done 2018, MNRAS 480, 1247 — qsosed accretion disc
# * Feltre et al. 2016, MNRAS 456, 3354 — AGN narrow-line-region grids
# * Nenkova et al. 2008, ApJ 685, 160 — clumpy torus
# * Calzetti et al. 2000, ApJ 533, 682 — starburst attenuation
# * Draine & Li 2007, ApJ 657, 810 — dust IR emission
# * Inoue et al. 2014, MNRAS 442, 1805 — IGM attenuation
# * Madau 1995, ApJ 441, 18 — IGM absorption
# * Li et al. 2025 — Cue nebular emulator
