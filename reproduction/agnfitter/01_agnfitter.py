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
# # Reproducing AGNFITTER-RX's physics with tengri
#
# AGNFITTER-RX (Martínez-Ramírez et al. 2024, A&A 688, A46) models the
# radio-to-X-ray SEDs of active galaxies. Where CIGALE, BAGPIPES, and
# Prospector are galaxy-centric, AGNFITTER-RX is built to *characterize the
# AGN itself* — its four physical components (accretion disk, hot dusty
# torus, relativistic jets / core radio, and hot corona) alongside a host
# galaxy (stellar populations, cold dust, star-formation radio). Its
# guiding idea is that radio and X-ray data — both largely unaffected by
# dust — are *orthogonal tracers* that break the infrared–ultraviolet
# degeneracies the original submm-to-UV AGNfitter struggled with. The code
# is, in the authors' words, a *model-testing laboratory* for AGN physics.
#
# This notebook places AGNFITTER-RX's model libraries next to tengri's, on
# the same axes and in the same units (erg/s/Hz), with the depth on the AGN
# block. The headline comparisons are the two model *face-offs* that drive
# the paper's conclusions:
#
# * **§9a — accretion-disk libraries.** R06, SN12, KD18, THB21. The paper
#   finds THB21 wins decisively (Bayes factor ≈10⁵·¹ over R06) because it
#   alone carries the broad and narrow emission lines that produce the
#   ≈0.7 µm flux peak (Hα + [N II]) the theory disks miss.
# * **§9c — torus libraries.** S04, NK08, SKIRTOR, CAT3D-Wind. CAT3D-Wind
#   wins (maximum likelihood in 25/36 sources) because its polar-wind dust
#   addresses the 1.5–5 µm near-IR excess that pure equatorial tori cannot.
#
# tengri's `cat3d_wind`, `silva04`, `skirtor_agnfitter`, `slone_netzer`,
# and `schreiber2018` blocks evaluate the same template libraries
# AGNFITTER-RX publishes, so those panels are direct checks of tengri's
# implementations against their source.
#
# Two ground rules keep the comparison honest. First, every claim of
# agreement prints the number that backs it — a peak-normalized residual,
# an implied attenuation in magnitudes, a slope — next to its panel.
# Second, wherever a tengri model is exercised, it is built through the
# public `SEDModel.build` grammar (disc/torus/atten sub-blocks, radio and
# X-ray groups), so the notebook doubles as an end-to-end test that every
# AGNFITTER-RX-parity parameter is actually wired through the public API.
#
# The host sections (§1–§8) are kept compact — AGNFITTER-RX's host (BC03 +
# Chabrier stellar, Schreiber/Dale cold dust) is standard and covered in
# depth by the sibling notebooks. Where tengri and AGNFITTER-RX genuinely
# differ, the difference is stated plainly and quantitatively. IGM is out
# of scope (AGNFITTER-RX applies no IGM attenuation; it instead masks
# photometry blueward of rest-frame Lyman-alpha in its likelihood).

# %% [markdown]
# ## Setup

# %%
import os

os.environ.setdefault("TENGRI_NO_BACKGROUND_COMPILE", "1")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from reproduction.agnfitter._drivers import agnfitter_driver as A, units as U

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

# nbclient kernels don't bind ``__file__``; fall back to cwd so the Setup
# cell can locate ``_figs/`` and data instead of crashing every panel.
_HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd().resolve()
figs_dir = _HERE / "_figs"
figs_dir.mkdir(exist_ok=True)
_FIG_DPI = 150


def save_fig(filename: str) -> None:
    """Save a figure to ``_figs/`` and leave it open for inline embedding."""
    plt.savefig(str(figs_dir / filename), dpi=_FIG_DPI, bbox_inches="tight")


# Unit-sanity guard. Every panel claims percent-level agreement, which rests
# on the AGNFITTER (log nu, F_nu) -> tengri (Angstrom, erg/s/Hz) bookkeeping.
# Trip the whole notebook here if that converter ever drifts.
_unit_check = U.verify_unit_conversion(rtol=1e-3)
print(
    f"unit-conversion bolometric round-trip: rel_err = "
    f"{_unit_check['rel_err']:.2e}  (target < 1e-3)"
)

# AGNFITTER-RX template libraries are read from a checkout (never the fitter).
if not A.available():
    print(A._CLONE_HINT)
print(f"AGNFITTER-RX libraries available: {A.available()}  ({A.AGNFITTER_HOME})")


def norm_at(wave, L, lam_aa):
    """Scale an SED so ``L(lam_aa) = 1`` (log-interp anchor)."""
    wave = np.asarray(wave)
    L = np.asarray(L)
    order = np.argsort(wave)
    ref = float(np.interp(lam_aa, wave[order], L[order]))
    return L / ref if ref > 0 else L


def norm_peak(L):
    """Scale an SED so its maximum is 1."""
    L = np.asarray(L)
    m = float(np.max(L))
    return L / m if m > 0 else L


# %% [markdown]
# ## Common stellar library
#
# Both codes use the Bruzual & Charlot (2003) stellar populations with a
# Chabrier (2003) IMF. tengri reads a BC03 + Chabrier grid in the DSPS
# layout; AGNFITTER-RX ships the same library as IR-luminosity-tagged
# τ-model templates. Because the underlying SSPs are identical published
# models, the host comparison below is a sanity check, not the focus.

# %%
_SSP_CANDIDATES = [
    _HERE / "_drivers" / "data" / "bc03_chabrier.h5",
    Path("data/bc03_pdva_stelib_chabrier.h5"),
    Path(tengri.__file__).resolve().parents[2] / "data" / "bc03_pdva_stelib_chabrier.h5",
    Path("/Users/suchethacooray/Projects/tengri/data/bc03_pdva_stelib_chabrier.h5"),
]
_ssp_file = next((p for p in _SSP_CANDIDATES if p.is_file()), None)
if _ssp_file is None:
    raise FileNotFoundError(
        "No BC03 SSP grid found. Place bc03_pdva_stelib_chabrier.h5 in the tengri data/ directory."
    )
ssp = load_ssp_data(str(_ssp_file.resolve()))
print(
    f"BC03 Chabrier SSP: {ssp.ssp_wave.shape[0]} wavelengths, "
    f"{ssp.ssp_lgmet.shape[0]} metallicities, {ssp.ssp_lg_age_gyr.shape[0]} ages."
)

# Fiducial host galaxy: delayed-τ SFH, τ = 1 Gyr, age = 5 Gyr, solar Z,
# 10^10 M_sun formed. Sections sweep one block around this.
TAU_GYR, AGE_GYR, LOG_MASS = 1.0, 5.0, 10.0
SFH_FIDUCIAL = {
    "type": "delayed",
    "tau_gyr": Fixed(TAU_GYR),
    "age_gyr": Fixed(AGE_GYR),
    "log_total_mass": Fixed(LOG_MASS),
    "*": FIXED,
}
NO_DUST = {"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED}


# %% [markdown]
# ## tengri AGN helpers
#
# Each AGN face-off builds a single tengri block in isolation and reads the
# AGN SED off ``state.derived["sed_agn"]``. We normalize shapes at a common
# anchor (2500 Å for disks, the IR peak for tori) so the comparison is of
# spectral shape at matched parameters, independent of the per-code
# luminosity bookkeeping.
#
# One default matters here: tengri's CIGALE-faithful AGN applies a polar-dust
# screen at E(B−V) = 0.03 by default (`agn_polar_ebv`, X-CIGALE's default).
# AGNFITTER-RX's disc templates carry no such screen — its only disc
# obscuration is the free `EBVbbb` — so every helper pins
# `agn_polar_ebv = 0`. Left at the default, the screen suppresses the
# rest-UV by ~0.05 dex and would masquerade as a disc-shape residual in
# every panel below.


# %%
def tengri_disc(disc_type, *, log_lbol=11.0, ebv_disc=None, **disc_params):
    """Isolated tengri accretion-disc SED. Returns (wave_aa, L_nu).

    ``ebv_disc`` sets the shared disc obscuration ``agn_ebv_disc`` (the
    AGNFITTER-RX ``EBVbbb`` analog) at the top level of the agn group.
    """
    disc = {"type": disc_type, "*": FIXED}
    disc.update({k: Fixed(v) for k, v in disc_params.items()})
    agn = {
        "type": "composable",
        "disc": disc,
        "torus": {"type": "none"},
        "lines": {"type": "none"},
        "agn_log_lbol": Fixed(log_lbol),
        "agn_polar_ebv": Fixed(0.0),  # AGNFITTER-RX templates carry no polar screen
        "*": FIXED,
    }
    if ebv_disc is not None:
        agn["agn_ebv_disc"] = Fixed(ebv_disc)
    m = SEDModel.build(
        ssp_data=ssp,
        sfh=SFH_FIDUCIAL,
        dust=NO_DUST,
        agn=agn,
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.derived["sed_agn"])


def tengri_disc_model(model_type, *, log_lbol=11.0):
    """Isolated tengri top-level disc *model* (e.g. richards2006)."""
    m = SEDModel.build(
        ssp_data=ssp,
        sfh=SFH_FIDUCIAL,
        dust=NO_DUST,
        agn={
            "type": model_type,
            "agn_log_lbol": Fixed(log_lbol),
            "agn_polar_ebv": Fixed(0.0),  # AGNFITTER-RX templates carry no polar screen
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.derived["sed_agn"])


def tengri_qsogen_full(*, log_lbol=11.0):
    """tengri's THB21 analog: qsogen continuum *with* its broad/narrow lines
    and FeII pseudo-continuum. THB21's defining feature is the emission-line
    forest (the 0.7 µm Hα+[N II] bump), so the disc-only continuum alone does
    not reproduce it — the lines and FeII blocks must be switched on."""
    m = SEDModel.build(
        ssp_data=ssp,
        sfh=SFH_FIDUCIAL,
        dust=NO_DUST,
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "*": FIXED},
            "torus": {"type": "none"},
            "lines": {"type": "qsogen", "*": FIXED},
            "feii": {"type": "qsogen_balmer", "*": FIXED},
            "agn_log_lbol": Fixed(log_lbol),
            "agn_polar_ebv": Fixed(0.0),  # AGNFITTER-RX templates carry no polar screen
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.derived["sed_agn"])


def tengri_torus(torus_type, *, log_lbol=11.0, **torus_params):
    """Isolated tengri torus SED. Returns (wave_aa, L_nu)."""
    torus = {"type": torus_type, "*": FIXED}
    torus.update({k: Fixed(v) for k, v in torus_params.items()})
    m = SEDModel.build(
        ssp_data=ssp,
        sfh=SFH_FIDUCIAL,
        dust=NO_DUST,
        agn={
            "type": "composable",
            "disc": {"type": "none"},
            "torus": torus,
            "lines": {"type": "none"},
            "agn_log_lbol": Fixed(log_lbol),
            "agn_polar_ebv": Fixed(0.0),  # AGNFITTER-RX templates carry no polar screen
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.derived["sed_agn"])


# %% [markdown]
# ## §1 Stellar populations
#
# Both codes build on BC03 + Chabrier. The panel shows tengri's BC03 SSP at
# a young (0.1 Gyr) and old (5 Gyr) age — the same templates AGNFITTER-RX
# uses to assemble its τ-model host library. No residual is drawn here
# because the libraries are the identical published models; the host
# comparison exists for continuity with the sibling notebooks.

# %%
fig, ax = plt.subplots(figsize=(7, 4.5))
for age, c in [(0.1, "C0"), (5.0, "C3")]:
    m = SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(0.1),
            "age_gyr": Fixed(age),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust=NO_DUST,
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    w, L = np.asarray(s.wave), np.asarray(s.derived["sed_dust_attenuated"])
    msk = (w > 9e2) & (w < 3e4)
    ax.loglog(w[msk], L[msk], c, lw=1.4, label=f"tengri BC03, age = {age:g} Gyr")
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title("BC03 + Chabrier stellar populations (shared library)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_01_ssp_bc03.png")

# %% [markdown]
# ## §2 Star formation history
#
# AGNFITTER-RX adopts τ-model (declining-exponential and delayed) star
# formation histories. tengri's parametric SFHs rescale every shape so that
# ``∫ SFR dt = 10**log_total_mass``. We read the *pipeline* SFR history off
# ``state.derived["sfr_history"]`` (not an analytic curve) and check the
# mass closes.

# %%
m = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL, dust=NO_DUST, redshift=Fixed(0.0))
s = m.predict_state({})
sfr = np.asarray(s.derived["sfr_history"])
# Lookback-time grid the pipeline actually integrates over.
lbt_yr = np.asarray(s.derived["sfh_grid_lbt_yr"])
order = np.argsort(lbt_yr)
mass_formed = float(np.trapezoid(sfr[order], lbt_yr[order]))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(lbt_yr / 1e9, sfr, "C1-", lw=1.6, label="tengri pipeline SFR history")
ax.set_xlabel("lookback time [Gyr]")
ax.set_ylabel(r"SFR [$M_\odot$/yr]")
ax.set_title(rf"Delayed-$\tau$ SFH ($\tau$ = {TAU_GYR:g} Gyr, age = {AGE_GYR:g} Gyr)")
ax.legend()
ax.grid(True, alpha=0.3)
ax.text(
    0.97,
    0.95,
    rf"$\int$ SFR dt = {mass_formed / 10**LOG_MASS:.4f} $\times\,10^{{{LOG_MASS:.0f}}}\,M_\odot$",
    transform=ax.transAxes,
    ha="right",
    va="top",
)
fig.tight_layout()
save_fig("agnfitter_02_sfh_tau.png")
print(f"§2  int SFR dt = {mass_formed:.4e} M_sun  (target 1.0000e{LOG_MASS:.0f})")

# %% [markdown]
# ## §3 Integrated stellar SED
#
# The fiducial host's stellar continuum from tengri, the same quantity
# AGNFITTER-RX's GA component contributes (reddened by an SMC/Calzetti law
# at fit time). Compact by design — see the CIGALE/Prospector notebooks for
# the full stellar head-to-head.

# %%
m = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL, dust=NO_DUST, redshift=Fixed(0.0))
s = m.predict_state({})
w, L = np.asarray(s.wave), np.asarray(s.derived["sed_dust_attenuated"])
fig, ax = plt.subplots(figsize=(7, 4.5))
msk = (w > 9e2) & (w < 1e7)
ax.loglog(w[msk], L[msk], "C1-", lw=1.4)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ [erg/s/Hz]")
ax.set_title(r"Integrated stellar SED ($10^{10}\,M_\odot$ host)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_03_stellar_sed.png")

# %% [markdown]
# ## §4 Disk reddening law (Prevot SMC)
#
# AGNFITTER-RX reddens the accretion disk (not the host) with the Prevot et
# al. (1984) SMC law, applying ``A_λ = k_raw(λ) · E(B−V)`` with
# ``k_raw(λ) = 1.39 λ_µm^−1.2 − 0.38`` directly, and no reddening blueward
# of 200 eV (``MODEL_AGNfitter.BBBred_Prevot``). tengri's disc obscuration
# (`agn_ebv_disc`, the `EBVbbb` analog) uses the same Prevot curve but pins
# the published SMC total-to-selective ratio: ``A_λ = k(λ) · R_V · E(B−V)``
# with ``k(λ) = k_raw(λ)/k_raw(V)`` and ``R_V = 2.72``. Because the raw fit
# evaluates to ``k_raw(0.55 µm) ≈ 2.468`` — the fit is calibrated in the UV,
# where it was measured, not at V — the two prescriptions differ by the
# uniform factor ``2.72/2.468 ≈ 1.102`` in A_λ at matched E(B−V): the
# *shape* is identical, and the AGNFITTER-RX ``EBVbbb`` posterior maps onto
# tengri's as ``E(B−V)_tengri ≈ E(B−V)_AGNFITTER / 1.102``. The right panel
# demonstrates both statements end-to-end: tengri's reddening is exercised
# through ``SEDModel.build`` (a qsogen disc with `agn_ebv_disc` set), not by
# re-evaluating the formula.

# %%
_K_RAW_V = 1.39 * 0.55 ** (-1.2) - 0.38  # Prevot raw fit at V band ≈ 2.468
_R_V_SMC = 2.72

w_thb, L_thb = A.disk_template("THB21")
lam_um = np.geomspace(0.1, 3.0, 400)
k_raw = 1.39 * lam_um ** (-1.2) - 0.38

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 4.5))
axl.plot(lam_um, k_raw, "C0-", lw=1.6, label="AGNFITTER-RX  $k_{raw}(\\lambda)$")
axl.plot(
    lam_um,
    k_raw / _K_RAW_V * _R_V_SMC,
    "C1--",
    lw=1.6,
    label=r"tengri  $k(\lambda)\,R_V$  ($=1.102\,k_{raw}$)",
)
axl.set_xscale("log")
axl.set_xlabel(r"$\lambda$ [µm]")
axl.set_ylabel(r"$A_\lambda / E(B{-}V)$")
axl.set_title("Prevot SMC law — two normalization conventions")
axl.legend(fontsize=8)
axl.grid(True, alpha=0.3)

# e2e attenuation ratios at E(B-V) = 0.3: tengri through SEDModel.build
# (ratio of two public-API predictions) vs AGNFITTER-RX's BBBred_Prevot.
_EBV_DEMO = 0.3
w_t0, L_t0 = tengri_disc("qsogen", ebv_disc=0.0)
w_t3, L_t3 = tengri_disc("qsogen", ebv_disc=_EBV_DEMO)
ratio_tengri = np.divide(L_t3, L_t0, out=np.ones_like(L_t3), where=L_t0 > 0)
L_thb_red = A.apply_bbb_reddening(w_thb, L_thb, _EBV_DEMO)
ratio_af = np.divide(L_thb_red, L_thb, out=np.ones_like(L_thb_red), where=L_thb > 0)
# tengri at the convention-rescaled E(B-V) — should land on the AGNFITTER curve.
w_tr, L_tr = tengri_disc("qsogen", ebv_disc=_EBV_DEMO * _K_RAW_V / _R_V_SMC)
ratio_tengri_rescaled = np.divide(L_tr, L_t0, out=np.ones_like(L_tr), where=L_t0 > 0)

msk_t = (w_t0 > 8e2) & (w_t0 < 3e4)
msk_a = (w_thb > 8e2) & (w_thb < 3e4)
axr.semilogx(w_thb[msk_a], -2.5 * np.log10(ratio_af[msk_a]), "C0-", lw=1.6,
             label="AGNFITTER-RX  BBBred_Prevot")
axr.semilogx(w_t0[msk_t], -2.5 * np.log10(ratio_tengri[msk_t]), "C1--", lw=1.6,
             label="tengri  agn_ebv_disc (SEDModel.build)")
axr.semilogx(w_t0[msk_t], -2.5 * np.log10(ratio_tengri_rescaled[msk_t]), "C3:", lw=1.6,
             label=r"tengri at $E(B{-}V)/1.102$ (rescaled)")
axr.set_xlabel(r"$\lambda$ [Å]")
axr.set_ylabel(r"$A_\lambda$ [mag] at $E(B{-}V)=0.3$")
axr.set_title("Disc attenuation, end-to-end")
axr.legend(fontsize=8)
axr.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_04_dust_attenuation.png")

# %%
# Quantify: interpolate both A_lambda curves onto a common grid over the
# disc window and report matched-E(B-V) and rescaled residuals.
_grid4 = np.geomspace(1.2e3, 1e4, 200)
_a_af = np.interp(_grid4, w_thb, -2.5 * np.log10(ratio_af))
_a_t = np.interp(_grid4, w_t0, -2.5 * np.log10(ratio_tengri))
_a_tr = np.interp(_grid4, w_t0, -2.5 * np.log10(ratio_tengri_rescaled))
print(
    f"§4  matched E(B-V)={_EBV_DEMO}: max|A_tengri − A_AGNFITTER| = "
    f"{np.max(np.abs(_a_t - _a_af)):.3f} mag  (ratio A_t/A_af median = "
    f"{np.median(_a_t / _a_af):.4f}, expected {_R_V_SMC / _K_RAW_V:.4f})"
)
print(
    f"§4  rescaled E(B-V)={_EBV_DEMO}/1.102: max|A_tengri − A_AGNFITTER| = "
    f"{np.max(np.abs(_a_tr - _a_af)):.4f} mag (pure-convention check)"
)

# %% [markdown]
# ## §6 Cold dust infrared emission
#
# AGNFITTER-RX offers two cold-dust libraries: the legacy DH02_CE01 (Dale &
# Helou 2002 + Chary & Elbaz 2001) and S17 (Schreiber et al. 2018), a
# flexible dust-continuum + PAH model parameterized by dust temperature and
# PAH fraction. The paper frames cold dust through an *energy-balance* prior:
# the cold-dust IR luminosity should at least match the dust-absorbed stellar
# luminosity (a constraint it deliberately allows to relax for
# spatially-disconnected high-z dust).
#
# tengri exposes three cold-dust models that bracket this choice:
#
# * **`schreiber2018`** — AGNFITTER-RX's S17 library evaluated from the same tabulated
#   Schreiber+2018 dust + PAH templates, mixed natively as
#   `(1 − f_PAH)·dust + f_PAH·PAH`). At matched (T_dust, f_PAH) it reproduces
#   AGNFITTER-RX's S17 shape to a median of ~0.05% and a maximum of ~1.3% of
#   the FIR peak, with the dust peak landing at 86.5 µm vs S17's 87.0 µm
#   (0.6%). This is the model to use when reproducing AGNFITTER-RX's cold dust.
# * **`schreiber2016`** — an *analytic* approximation (modified blackbody +
#   a few Drude PAH profiles). Its FIR peak (92 µm) is close, but its PAH
#   forest at 3–13 µm comes out much weaker than the tabulated S17 at matched
#   (T_dust, f_PAH). It is fast and differentiable but not PAH-faithful — the
#   gap that motivated the tabulated `schreiber2018` block.
# * **`dale2014`** — tengri's tabulated Dale+2014 library, the modern relative
#   of AGNFITTER-RX's Dale & Helou 2002 (the "DH" in DH02_CE01). It is
#   parameterized by the radiation-field hardness α rather than a dust
#   temperature, so it is not a node-for-node match to S17; shown here at
#   α = 1.5 it peaks near S17 and carries its own (real, tabulated) PAH
#   features. tengri has no direct DH02_CE01 counterpart — `dale2014` is the closest
#   counterpart.
#
# All curves are normalized at their FIR peak.

# %%
import jax.numpy as jnp

from tengri.dust import DUST_EMISSION_MODELS

wave_ir = np.geomspace(1e4, 1e8, 2000)  # 1 µm – 1 cm
schreiber18 = DUST_EMISSION_MODELS["schreiber2018"]
schreiber16 = DUST_EMISSION_MODELS["schreiber2016"]
dale14 = DUST_EMISSION_MODELS["dale2014"]
L_s18 = np.asarray(schreiber18(jnp.asarray(wave_ir), 1.0, dust_T=35.0, dust_f_pah=0.02))
L_s16 = np.asarray(schreiber16(jnp.asarray(wave_ir), 1.0, dust_T=35.0, dust_f_pah=0.02))
L_d14 = np.asarray(dale14(jnp.asarray(wave_ir), 1.0, dust_alpha_dale=1.5))

w_s17, L_s17 = A.cold_dust_template("S17", tdust=35.0, fpah=0.02)
w_dh, L_dh = A.cold_dust_template("DH02_CE01")

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.loglog(
    w_s17, norm_peak(L_s17), "C0-", lw=2.2, alpha=0.5, label="AGNFITTER-RX  S17 (Schreiber+18)"
)
ax.loglog(wave_ir, norm_peak(L_s18), "C3-", lw=1.4, label="tengri  schreiber2018 (S17 tables)")
ax.loglog(wave_ir, norm_peak(L_s16), "C1--", lw=1.3, label="tengri  schreiber2016 (analytic)")
ax.loglog(wave_ir, norm_peak(L_d14), "C4:", lw=1.5, label=r"tengri  dale2014 ($\alpha=1.5$)")
ax.loglog(w_dh, norm_peak(L_dh), "C2-.", lw=1.2, label="AGNFITTER-RX  DH02_CE01")
ax.set_xlim(1e4, 1e8)
ax.set_ylim(1e-3, 3)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ (norm. at peak)")
ax.set_title("Cold-dust IR: tengri schreiber2018 / schreiber2016 / dale2014 vs AGNFITTER-RX")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_06_cold_dust.png")

# %%
# Quantify the schreiber2018 ↔ S17 agreement at matched nodes (shape, over 3–300 µm).
_band = (w_s17 > 3e4) & (w_s17 < 3e6)
_s17n = L_s17 / L_s17[_band].max()
_s18n = np.asarray(schreiber18(jnp.asarray(w_s17), 1.0, dust_T=35.0, dust_f_pah=0.02))
_s18n = _s18n / _s18n[_band].max()
_resid = np.abs(_s18n[_band] - _s17n[_band])
print(
    f"§6  schreiber2018 vs AGNFITTER-RX S17 (T=35 K, f_PAH=0.02):  "
    f"median |Δ|/peak = {np.median(_resid) * 100:.3f}%   max = {_resid.max() * 100:.2f}%"
)

# %% [markdown]
# ## §9a Accretion-disk library face-off
#
# The four AGNFITTER-RX disk libraries against their tengri counterparts,
# all unreddened and normalized at 2500 Å. The decisive feature is the
# narrow peak near 0.7 µm (Hα λ6563 + [N II] λλ6549,6585): present only in
# the semi-empirical THB21, absent from the theory disks R06/SN12/KD18.
# AGNFITTER-RX finds this single feature drives a ≈10¹⁰ likelihood gap.
#
# SN12 (Slone & Netzer 2012) was, until this work, the one disk library
# tengri lacked — every "Netzer" reference in tengri pointed to the unrelated
# Laor & Netzer (1989) self-gravity radius. tengri's new `slone_netzer` disc
# block ports the SN12 α-disc grid directly from AGNFITTER-RX's `SN12.pickle`
# (the M_BH = 8.6, log Ṁ/Ṁ_edd ≈ −2.0 grid point is shown on both sides).
#
# How well each tengri block matches, panel by panel:
#
# * **THB21 — reproduced.** The 0.7 µm bump is an emission-line feature, so
#   `qsogen` must run *with* its line and FeII blocks (continuum alone misses
#   it entirely). With them on, tengri's Hα/2500 Å contrast sits above the
#   vendored THB21 template's (2.45 — the reference h5's 1024-point common
#   grid undersamples the narrow Hα peak; tengri evaluates qsogen at native
#   resolution) along the qsogen luminosity sequence — matching log_lbol
#   aligns the two templates.
# * **SN12 — reproduced** by the `slone_netzer` block. It interpolates the
#   108-template grid with node-exact bilinear interpolation, so the SN12 peak
#   lands on AGNFITTER-RX's at every grid node (the peak shifts strongly with
#   accretion rate, so the original smooth-kernel interpolation smeared it by
#   30–50% — now fixed).
# * **KD18 — reproduced** (with a parameterization note). tengri's full
#   Kubota & Done 3-zone block (`kubota_done`: a Novikov-Thorne outer disc +
#   warm Comptonization + a hot corona) is *luminosity-first*: the Eddington
#   ratio is derived from `agn_log_lbol` and `agn_log_mbh` (λ_Edd =
#   L_bol/L_Edd), so `agn_log_ledd` is deprecated and ignored for this disc.
#   To land on an AGNFITTER-RX grid node (logM_BH, logλ_Edd) we therefore set
#   L_bol = λ_Edd · L_Edd(M_BH), computed below from the same physical
#   constants tengri uses. At the matched node the two realizations track
#   across the UV–near-IR; the smooth residual (quantified below) is the
#   spread expected between a precomputed qsosed template grid
#   (AGNFITTER-RX) and tengri's from-scratch three-zone integration, whose
#   warm-Comptonization proxy is a documented approximation.
# * **R06 — the same template, in two conventions.** Both sides use the
#   identical Richards+2006 composite. The catch is purely how it is carried:
#   tengri stores it as the physically-correct L_ν (divides the published
#   νL_ν by ν), so its peak sits at 1.2 µm, while AGNFITTER-RX carries the
#   νL_ν array directly, so its R06 peaks at 3050 Å — a factor of ν bluer.
#   The dotted gray curve shows tengri's `richards2006` *put back into
#   AGNFITTER-RX's νL_ν convention* (multiply by ν, re-anchor at 2500 Å): it
#   lands exactly on AGNFITTER-RX's R06 (median residual 0.0% over
#   0.15–3 µm). So the apparent ~20× near-IR gap between the two solid curves
#   is the ν-factor convention alone — the underlying disk template is bit-
#   identical.

# %%
from tengri.utils.physics_constants import C_CGS, G_GRAV, L_SUN, M_PROTON, M_SUN, SIGMA_T

ANCHOR = 2500.0


def kd18_log_lbol(log_mbh, log_edd):
    """L_bol [log10 L_sun] that pins kubota_done's derived λ_Edd to a grid node.

    kubota_done derives λ_Edd = L_bol / L_Edd(M_BH), so matching an
    AGNFITTER-RX (logM_BH, logλ_Edd) node means requesting
    L_bol = λ_Edd · L_Edd, with L_Edd = 4π G M m_p c / σ_T — the same
    constants tengri's disc uses internally.
    """
    l_edd = 4.0 * np.pi * G_GRAV * (10.0**log_mbh * M_SUN) * M_PROTON * C_CGS / SIGMA_T
    return log_edd + np.log10(l_edd / L_SUN)


# Each entry: (AGNFITTER name, AGNFITTER kwargs, tengri loader, tengri label).
# SN12/KD18 are gridded; pick a matched (M_BH, Eddington) point on both sides.
disk_pairs = [
    ("R06", {}, lambda: tengri_disc_model("richards2006"), "richards2006"),
    (
        "SN12",
        dict(log_mbh=8.6, edd_index=10),  # log_edd ≈ -2.0
        lambda: tengri_disc("slone_netzer", agn_log_mbh=8.6, agn_log_ledd=-2.0),
        "slone_netzer (Slone & Netzer 12)",
    ),
    (
        "KD18",
        dict(log_mbh=8.0, log_edd=-1.0),
        lambda: tengri_disc(
            "kubota_done", log_lbol=kd18_log_lbol(8.0, -1.0), agn_log_mbh=8.0
        ),
        "kubota_done (3-zone, matched L_bol)",
    ),
    ("THB21", {}, tengri_qsogen_full, "qsogen + lines + FeII"),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, (af_name, af_kw, tengri_fn, tengri_label) in zip(axes.ravel(), disk_pairs):
    w_a, L_a = A.disk_template(af_name, **af_kw)
    msk_a = (w_a > 5e2) & (w_a < 5e4)
    ax.loglog(
        w_a[msk_a], norm_at(w_a, L_a, ANCHOR)[msk_a], "C0-", lw=1.5, label=f"AGNFITTER  {af_name}"
    )
    w_t, L_t = tengri_fn()
    msk_t = (w_t > 5e2) & (w_t < 5e4)
    ax.loglog(
        w_t[msk_t],
        norm_at(w_t, L_t, ANCHOR)[msk_t],
        "C1-",
        lw=1.5,
        label=f"tengri  {tengri_label}",
    )
    if af_name == "R06":
        # Same template in AGNFITTER-RX's nu-L_nu convention: multiply tengri's
        # physically-correct L_nu by nu and re-anchor. This overlays AGNFITTER's
        # R06 exactly, isolating the offset above as a pure convention choice.
        nu_t = U.C_ANGSTROM_PER_S / w_t
        ax.loglog(
            w_t[msk_t],
            norm_at(w_t, L_t * nu_t, ANCHOR)[msk_t],
            color="0.5",
            ls=":",
            lw=1.6,
            label=r"tengri $\times\,\nu$ (AGNFITTER conv.)",
        )
    ax.axvline(6563, color="0.7", ls=":", lw=1)
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
# Quantify each panel: |log10(tengri/AGNFITTER)| over the disc window
# (2500 Å-anchored shapes, 1200 Å – 1 µm), evaluated AT the AGNFITTER
# reference's own wavelength samples — interpolating the sparser reference
# onto a dense grid would manufacture residual at every line feature. For
# R06 the residual is also reported after putting tengri's L_nu back into
# AGNFITTER's nu·L_nu carriage — that number isolates the pure convention
# offset.
print("§9a  disc-library shape residuals (anchor 2500 Å, 1200 Å–1 µm):")
for af_name, af_kw, tengri_fn, _label in disk_pairs:
    w_a, L_a = A.disk_template(af_name, **af_kw)
    w_t, L_t = tengri_fn()
    oa_, ot_ = np.argsort(w_a), np.argsort(w_t)
    w_a_s, a_s = np.asarray(w_a)[oa_], norm_at(w_a, L_a, ANCHOR)[oa_]
    m = (w_a_s >= 1.2e3) & (w_a_s <= 1e4) & (a_s > 0)
    t_on = np.interp(w_a_s[m], np.asarray(w_t)[ot_], norm_at(w_t, L_t, ANCHOR)[ot_])
    logr = np.abs(np.log10(t_on / a_s[m]))
    line = f"  {af_name:6s} median = {np.median(logr):.3f} dex   max = {logr.max():.3f} dex"
    if af_name == "R06":
        nu_t = U.C_ANGSTROM_PER_S / np.asarray(w_t)
        t_conv = np.interp(w_a_s[m], np.asarray(w_t)[ot_], norm_at(w_t, L_t * nu_t, ANCHOR)[ot_])
        logc = np.abs(np.log10(t_conv / a_s[m]))
        line += f"   (nu-L_nu convention applied: median = {np.median(logc):.4f} dex)"
    print(line)

# %%
# Quantify the line bump.
w_t, L_t = A.disk_template("THB21")
w_r, L_r = A.disk_template("R06")
bump_thb = float(
    np.interp(
        6563, *(lambda o: (np.asarray(w_t)[o], norm_at(w_t, L_t, ANCHOR)[o]))(np.argsort(w_t))
    )
)
bump_r06 = float(
    np.interp(
        6563, *(lambda o: (np.asarray(w_r)[o], norm_at(w_r, L_r, ANCHOR)[o]))(np.argsort(w_r))
    )
)
print(
    f"§9a  L_nu(6563 Å)/L_nu(2500 Å):  THB21 = {bump_thb:.2f}   R06 = {bump_r06:.2f}  "
    f"(ratio {bump_thb / bump_r06:.1f}x)"
)

# %% [markdown]
# ## §9b Accretion-disk reddening sweep
#
# The disk color excess E(B−V)_BBB sweeps the UV continuum via the Prevot
# SMC law on both sides. AGNFITTER-RX applies it to the THB21 template;
# tengri's `agn_ebv_disc` (identically, its `agn.atten = "smc_prevot"`
# block) applies the same curve to whatever disc is configured. Plotting
# the attenuation ratio ``L(E(B−V)) / L(0)`` removes the underlying
# template from both sides, so this panel compares the reddening laws
# themselves, end-to-end: solid curves are AGNFITTER-RX's
# ``BBBred_Prevot``, dashed curves are two `SEDModel.build` predictions
# divided by each other. The visible dashed-above-solid offset is the
# uniform 1.102× normalization convention quantified in §4.

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.8))
w_thb, L_thb = A.disk_template("THB21")
_, L_te0 = tengri_disc("qsogen", ebv_disc=0.0)
msk = (w_thb > 8e2) & (w_thb < 1e4)
print("§9b  A(1500 Å) per E(B-V) [mag]:")
for ebv, c in [(0.1, "C2"), (0.3, "C1"), (0.5, "C3")]:
    L_red = A.apply_bbb_reddening(w_thb, L_thb, ebv)
    ratio_af = np.divide(L_red, L_thb, out=np.ones_like(L_red), where=L_thb > 0)
    ax.loglog(w_thb[msk], ratio_af[msk], c, ls="-", lw=1.4,
              label=f"AGNFITTER  E(B−V) = {ebv:g}")
    w_te, L_te = tengri_disc("qsogen", ebv_disc=ebv)
    ratio_te = np.divide(L_te, L_te0, out=np.ones_like(L_te), where=L_te0 > 0)
    msk_te = (w_te > 8e2) & (w_te < 1e4)
    ax.loglog(w_te[msk_te], ratio_te[msk_te], c, ls="--", lw=1.4,
              label=f"tengri  E(B−V) = {ebv:g}")
    a_af = -2.5 * np.log10(np.interp(1500.0, w_thb, ratio_af))
    a_te = -2.5 * np.log10(np.interp(1500.0, w_te, ratio_te))
    print(f"  E(B-V)={ebv:g}:  AGNFITTER = {a_af:.2f}   tengri = {a_te:.2f}   "
          f"(ratio {a_te / a_af:.3f}, convention 1.102)")
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L(E(B{-}V))\ /\ L(0)$")
ax.set_title("Disk reddening sweep (Prevot SMC) — both codes, template-free")
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_09b_bbb_reddening.png")

# %% [markdown]
# ## §9c Torus library face-off
#
# The four AGNFITTER-RX torus libraries against tengri's, all normalized at
# their mid-IR peak. tengri's `silva04` and `cat3d_wind` blocks evaluate the
# same template libraries AGNFITTER-RX publishes and are pinned here to
# matched grid nodes — S04 at log N_H = 23, CAT3D-Wind at (incl 0°, a = −2,
# f_wd = 1.75); all three CAT3D axes are live through `SEDModel.build` over
# AGNFITTER-RX's genuine rows-210+ domain (a ∈ [−3, −1.5], f_wd ∈ [1.0, 2.25]).
# `nenkova` is the FSPS Nenkova+2008 CLUMPY radiative-transfer grid (the same
# library Prospector uses), interpolated in optical depth. It differs from
# AGNFITTER-RX's NK08 by ~2–4× in the near-IR because the two average
# different corners of the CLUMPY parameter space (Y, N₀, q, σ) — the same
# model family, two different reductions, so that panel is a family-level
# comparison rather than a node-matched check. An inclination-averaged
# `NK0_mean_1p` counterpart is the one torus entry tengri does not yet offer.
#
# For `skirtor`, the IR peak lands slightly redward of AGNFITTER-RX: `skirtor`
# keeps the full Stalevski (2016) grid (τ, p, q, opening angle, inclination,
# following X-CIGALE), while AGNFITTER-RX fits a parameter-averaged reduction
# (`SKIRTOR_mean_3p`), whose averaging warms the effective SED. Two reductions of
# one model.
#
# tengri ships **both**: `skirtor` (full X-CIGALE grid, shown here) and
# `skirtor_agnfitter` (node-exact on the published `SKIRTOR_mean_3p`
# templates, exactly as for `silva04` and `cat3d_wind`). The next panel
# (§9c′) overlays the two reductions against the AGNFITTER-RX reference, so you
# can see the node-exact version land on the averaged library while the
# full-grid block sits at longer wavelengths.
#
# Two residual pathologies the paper emphasizes also live in this plot: the
# 10 µm silicate feature (NK08/CAT3D can over- or under-predict it depending
# on inclination) and the 1.5–5 µm near-IR excess that the CAT3D polar-wind
# component is designed to fill.

# %%
torus_pairs = [
    # Pin both sides to the same column density (log N_H = 23). tengri's
    # `log_nh_silva` is now a live builder parameter, so this is a genuine
    # matched-parameter template check, not a default-vs-default coincidence.
    (
        "S04",
        "S04",
        lambda: tengri_torus("silva04", log_nh_silva=23.0),
        "silva04 (log N_H = 23)",
        dict(log_nh=23.0),
    ),
    ("NK08", "NK08", lambda: tengri_torus("nenkova"), "nenkova", dict(incl=0.0)),
    (
        "SKIRTOR",
        "SKIRTOR",
        # Match AGNFITTER-RX's geometry exactly: oa = 40°, incl = 30°
        # (cos_inc = cos 30° = 0.866), tau_9.7 = 7 — so the panel compares
        # the same Stalevski sightline on both sides, not two geometries.
        lambda: tengri_torus("skirtor", cos_inc=0.8660254, oa_skirtor=40.0, tau_skirtor=7.0),
        "skirtor (oa 40°, incl 30°, τ 7)",
        dict(oa=40.0, incl=30.0, tau=7.0),
    ),
    (
        "CAT3D",
        "CAT3D-Wind",
        # Pin ALL THREE library axes on both sides: incl = 0° (cos_inc = 1),
        # cloud power-law index a = -2, wind fraction fwd = 1.75. tengri's
        # cat3d_wind spans AGNfitter's genuine rows-210+ sub-library
        # (a in [-3, -1.5], fwd in [1.0, 2.25]) with node-exact PCHIP, so a
        # matched node is a genuine template check.
        lambda: tengri_torus("cat3d_wind", cos_inc=1.0, a_cat3d=-2.0, fwd_cat3d=1.75),
        "cat3d_wind (incl 0°, a −2, f_wd 1.75)",
        dict(incl=0.0, a=-2.0, fwd=1.75),
    ),
]
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, (af_name, title, tengri_fn, tengri_label, af_kw) in zip(axes.ravel(), torus_pairs):
    w_a, L_a = A.torus_template(af_name, **af_kw)
    msk_a = (w_a > 5e3) & (w_a < 1e7)
    ax.loglog(w_a[msk_a], norm_peak(L_a)[msk_a], "C0-", lw=1.5, label=f"AGNFITTER  {af_name}")
    w_t, L_t = tengri_fn()
    msk_t = (w_t > 5e3) & (w_t < 1e7)
    ax.loglog(w_t[msk_t], norm_peak(L_t)[msk_t], "C1-", lw=1.5, label=f"tengri  {tengri_label}")
    ax.axvline(1e5, color="0.7", ls=":", lw=1)  # 10 µm silicate
    ax.set_xlim(5e3, 1e7)
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
# ### §9c″ Torus-library parity — full-spectrum shape ratio
#
# The tengri / AGNFITTER-RX peak-normalized shape ratio for each torus library,
# with the 1–100 µm median printed below. S04 and CAT3D-Wind (matched sightline)
# reproduce AGNFITTER-RX; SKIRTOR and NK08 use different library reductions.

# %%
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.axhspan(0.8, 1.25, color="0.9", zorder=0)
ax.axhline(1.0, color="0.5", lw=0.8)
_ratio_grid = np.geomspace(5e3, 1e7, 400)
print("§9c torus-library full-spectrum shape parity (tengri / AGNFITTER, peak-norm):")
for (af_name, _title, tengri_fn, _tlabel, af_kw), _c in zip(torus_pairs, ["C0", "C1", "C2", "C3"]):
    w_a, L_a = A.torus_template(af_name, **af_kw)
    w_t, L_t = tengri_fn()
    a_on = np.interp(
        np.log10(_ratio_grid), np.log10(w_a), norm_peak(L_a), left=np.nan, right=np.nan
    )
    t_on = np.interp(
        np.log10(_ratio_grid), np.log10(w_t), norm_peak(L_t), left=np.nan, right=np.nan
    )
    _ratio = t_on / a_on
    ax.loglog(_ratio_grid, _ratio, _c, lw=1.4, label=_title)
    _m = (_ratio_grid > 1e4) & (_ratio_grid < 1e6) & np.isfinite(_ratio) & (a_on > 1e-3)
    if _m.any():
        print(f"  {_title:12s}: median (1-100 µm) = {float(np.nanmedian(_ratio[_m])):.3f}×")
ax.set_xlim(5e3, 1e7)
ax.set_ylim(0.3, 3.0)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel("tengri / AGNFITTER (peak-norm.)")
ax.set_title("Torus-library full-spectrum shape parity")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_09c2_torus_ratio.png")
plt.show()


# %% [markdown]
# ### §9c′ Both SKIRTOR reductions, side by side
#
# The panel above contrasts AGNFITTER-RX's averaged `SKIRTOR_mean_3p` with
# tengri's *full-grid* `skirtor` (the X-CIGALE reduction) — different by
# design. tengri now also ships `skirtor_agnfitter`, a direct node-exact
# node-exact on the very same `SKIRTOR_mean_3p` templates, so we can show both
# reductions of the Stalevski (2016) models against the AGNFITTER-RX
# reference at one matched geometry (opening angle 40°, inclination 30°,
# τ₉.₇ = 7):
#
# * **`skirtor_agnfitter`** overlays the AGNFITTER-RX curve to node-exact
#   tolerance — the monotone-cubic (PCHIP) interpolant reproduces the
#   tabulated library at its grid nodes rather than smoothing across them.
# * **`skirtor`** (full X-CIGALE grid) carries the un-averaged clumpiness
#   and radial-distribution structure, which broadens and warms the SED and
#   pushes its IR peak to longer wavelengths.
#
# Neither is wrong — they are two intentional reductions of one model, and
# tengri lets you pick either. Use `skirtor_agnfitter` to reproduce an
# AGNFITTER-RX fit bit-for-bit, or `skirtor` to match a CIGALE-family run.

# %%
fig, ax = plt.subplots(figsize=(7.5, 5))
w_ref, L_ref = A.torus_template("SKIRTOR", oa=40.0, incl=30.0, tau=7.0)
msk_ref = (w_ref > 5e3) & (w_ref < 1e7)
ax.loglog(
    w_ref[msk_ref], norm_peak(L_ref)[msk_ref], "C0-", lw=2.0, label="AGNFITTER  SKIRTOR_mean_3p"
)
w_af, L_af = tengri_torus("skirtor_agnfitter", oa_skirtor=40.0, incl_skirtor=30.0, tv_skirtor=7.0)
msk_af = (w_af > 5e3) & (w_af < 1e7)
ax.loglog(
    w_af[msk_af],
    norm_peak(L_af)[msk_af],
    "C1--",
    lw=1.6,
    label="tengri  skirtor_agnfitter (node-exact)",
)
w_xc, L_xc = tengri_torus("skirtor", cos_inc=0.8660254, oa_skirtor=40.0, tau_skirtor=7.0)
msk_xc = (w_xc > 5e3) & (w_xc < 1e7)
ax.loglog(
    w_xc[msk_xc],
    norm_peak(L_xc)[msk_xc],
    "C3-",
    lw=1.6,
    label="tengri  skirtor (full X-CIGALE grid, by design)",
)
ax.axvline(1e5, color="0.7", ls=":", lw=1)  # 10 µm silicate
ax.set_xlim(5e3, 1e7)
ax.set_ylim(1e-3, 3)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ (norm. at peak)")
ax.set_title("SKIRTOR: averaged templates vs full grid (oa 40°, incl 30°, τ 7)")
ax.legend(fontsize=8, loc="lower center")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_09c2_skirtor_port.png")

# %% [markdown]
# ### §9c‴ CAT3D-Wind wind-fraction sweep
#
# The polar wind is what CAT3D-Wind is *for* — the paper adopts it because
# the wind fills the 1.5–5 µm near-IR excess that equatorial tori miss. This
# panel sweeps the wind mass fraction f_wd across AGNFITTER-RX's library
# domain (1.0 → 2.25) on both sides at fixed (incl 0°, a = −2): tengri's
# `cat3d_wind` (solid, through `SEDModel.build`) over the AGNFITTER-RX
# library nodes (dashed). The interpolation is node-exact monotone-cubic,
# so at library nodes the curves coincide; the printed residuals quantify
# that. Raising f_wd lifts exactly the near-IR shoulder.

# %%
fig, ax = plt.subplots(figsize=(8, 5))
print("§9c‴  cat3d_wind wind-fraction sweep, node residuals (peak-norm, 1–100 µm):")
_fwd_grid = np.geomspace(1e4, 1e6, 300)
for fwd, c in [(1.0, "C0"), (1.75, "C1"), (2.25, "C3")]:
    w_a, L_a = A.torus_template("CAT3D", incl=0.0, a=-2.0, fwd=fwd)
    w_t, L_t = tengri_torus("cat3d_wind", cos_inc=1.0, a_cat3d=-2.0, fwd_cat3d=fwd)
    msk_a = (w_a > 5e3) & (w_a < 1e7)
    msk_t = (w_t > 5e3) & (w_t < 1e7)
    ax.loglog(w_a[msk_a], norm_peak(L_a)[msk_a], c, ls="--", lw=1.2)
    ax.loglog(
        w_t[msk_t], norm_peak(L_t)[msk_t], c, ls="-", lw=1.5, label=f"$f_{{wd}}$ = {fwd:g}"
    )
    a_on = np.interp(np.log10(_fwd_grid), np.log10(w_a), norm_peak(L_a))
    t_on = np.interp(np.log10(_fwd_grid), np.log10(w_t), norm_peak(L_t))
    ok = a_on > 1e-3
    _res = np.abs(t_on[ok] / a_on[ok] - 1.0)
    print(f"  f_wd = {fwd:4g}:  median |ratio-1| = {np.median(_res) * 100:.2f}%   "
          f"max = {_res.max() * 100:.2f}%")
ax.axvspan(1.5e4, 5e4, color="0.92", zorder=0)  # the 1.5-5 µm near-IR excess band
ax.set_xlim(5e3, 1e7)
ax.set_ylim(1e-3, 3)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ (norm. at peak)")
ax.set_title("CAT3D-Wind $f_{wd}$ sweep — tengri (solid) on AGNFITTER-RX nodes (dashed)")
ax.legend(fontsize=9, title="shaded: 1.5–5 µm excess band")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_09c3_cat3d_fwd_sweep.png")

# %% [markdown]
# ## §9d Best combination — CAT3D-Wind + THB21
#
# The paper's winning model for 67% of its sample: the CAT3D-Wind torus on
# the THB21 disk. We build the full AGN SED on both sides — AGNFITTER-RX's
# THB21 disk plus its CAT3D-Wind torus, against tengri's composable AGN with
# `disc = qsogen` (with its lines + FeII, i.e. the full THB21 analog) and
# `torus = cat3d_wind` — normalized at the disk's 2500 Å. Carrying the disc's
# emission lines is what reproduces the 0.7 µm bump on top of the torus hump.

# %%
# Matched CAT3D-Wind node on both sides: incl 0°, a = -2, f_wd = 1.75.
_CAT3D_NODE = dict(incl=0.0, a=-2.0, fwd=1.75)
_CAT3D_NODE_TE = dict(cos_inc=1.0, a_cat3d=-2.0, fwd_cat3d=1.75)
w_disc, L_disc = A.disk_template("THB21")
w_tor, L_tor = A.torus_template("CAT3D", **_CAT3D_NODE)
# Co-add on a common grid, torus scaled to ~10% of the disk's 2500 A anchor.
grid = np.geomspace(1e3, 1e7, 3000)
disc_g = U.regrid(w_disc, np.clip(L_disc, 0, None), grid)
tor_g = U.regrid(w_tor, np.clip(L_tor, 0, None), grid)
disc_n = disc_g / np.interp(2500, grid, disc_g)
tor_n = tor_g / np.max(tor_g) * 0.5
af_total = disc_n + tor_n

w_te, L_te = tengri_torus("cat3d_wind", **_CAT3D_NODE_TE)  # torus
# THB21 *is* qsogen with its emission-line forest; the disc continuum alone
# would drop the 0.7 µm Hα+[N II] bump that defines this combination, so use
# the full qsogen + lines + FeII disc to match AGNFITTER-RX's THB21.
w_td, L_td = tengri_qsogen_full()  # disc
te_disc_g = U.regrid(w_td, np.clip(L_td, 0, None), grid)
te_tor_g = U.regrid(w_te, np.clip(L_te, 0, None), grid)
te_disc_n = te_disc_g / np.interp(2500, grid, te_disc_g)
te_tor_n = te_tor_g / np.max(te_tor_g) * 0.5
te_total = te_disc_n + te_tor_n
# tengri's cat3d_wind torus grid ends near 1.5e6 Å; mask the sum beyond it so
# the curve ends cleanly instead of cliffing down to the bare disc tail.
te_total = np.where(te_tor_g > 0, te_total, np.nan)

fig, ax = plt.subplots(figsize=(8.5, 5))
ax.loglog(grid, af_total, "C0-", lw=1.6, label="AGNFITTER-RX  THB21 + CAT3D-Wind")
ax.loglog(grid, te_total, "C1--", lw=1.6, label="tengri  qsogen + cat3d_wind")
# tengri's packaged cat3d_wind grid spans to ~1.5e6 Å (the build's
# common-wavelength intersection); cap the axis there so the comparison runs
# only where both torus libraries have data, not into the bare disc tail.
ax.set_xlim(1e3, 1.5e6)
ax.set_ylim(1e-3, 5)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ (disk norm. at 2500 Å)")
ax.set_title("Best-combination AGN SED (paper's winning model)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_09d_best_combo.png")

# %% [markdown]
# ## §10 X-ray corona via the α_ox–L₂₅₀₀ relation
#
# AGNFITTER-RX ties the 2 keV corona to the 2500 Å disk continuum through
# the Just et al. (2007) / Lusso & Risaliti (2016, 2017) relation,
# ``α_ox = −0.137 log L₂₅₀₀ + 2.638 + Δα_ox``, then lays down a Γ = 1.8
# power law with a 300 keV exponential cutoff. The dispersion Δα_ox ∈
# [−0.4, 0.4] is a free parameter. tengri exposes the same `just2007`,
# `lusso_risaliti_2016/2017` relations in `alpha_ox_from_l2500`. Soft X-ray
# amplitude differences (10–20% level) sit within the corona-model uncertainty
# set by α_ox scatter and cutoff-edge location.
#
# The relation is valid only for type-1, radio-quiet, non-BAL AGN — the
# paper is explicit that it should not be applied to type-2, BAL, or blazar
# sources.

# %%
from tengri.xray import alpha_ox_from_l2500

l2500 = np.geomspace(1e28, 1e32, 200)
fig, (axl, axr) = plt.subplots(1, 2, figsize=(12, 4.6))
for rel, c in [("just2007", "C0"), ("lusso_risaliti_2016", "C2"), ("lusso_risaliti_2017", "C3")]:
    aox = np.array([float(alpha_ox_from_l2500(x, relation=rel)) for x in l2500])
    axl.plot(np.log10(l2500), aox, c, lw=1.5, label=f"tengri {rel}")
aox_af = -0.137 * np.log10(l2500) + 2.638
axl.plot(np.log10(l2500), aox_af, "k--", lw=1.3, label="AGNFITTER-RX (Just+07)")
axl.set_xlabel(r"$\log_{10}\ L_{2500\,\AA}$ [erg/s/Hz]")
axl.set_ylabel(r"$\alpha_{ox}$")
axl.set_title(r"$\alpha_{ox}$–$L_{2500}$ relation")
axl.legend(fontsize=8)
axl.grid(True, alpha=0.3)

# Disk + α_ox X-ray extension on the AGNFITTER side. Δα_ox sets the X-ray
# *amplitude* relative to the disk, so all three curves share one
# normalization (the Δα_ox = 0 peak) — normalizing each at its own peak
# would hide exactly the effect this panel is about.
w_thb, L_thb = A.disk_template("THB21")
_xray = {scat: A.disk_xray_extension(w_thb, L_thb, scatter=scat) for scat in (-0.4, 0.0, 0.4)}
_xref = float(np.max(_xray[0.0][1]))
for scat, c in [(-0.4, "C2"), (0.0, "C0"), (0.4, "C3")]:
    xw, xL = _xray[scat]
    axr.loglog(xw, xL / _xref, c, lw=1.4, label=rf"$\Delta\alpha_{{ox}}$ = {scat:+.1f}")
axr.set_xlim(1e-2, 1e2)
axr.set_ylim(1e-3, 1e3)
axr.set_xlabel(r"$\lambda$ [Å]")
axr.set_ylabel(r"$L_\nu$ (norm.)")
axr.set_title("X-ray corona (Γ = 1.8, 300 keV cutoff)")
axr.legend(fontsize=8)
axr.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_10a_alphaox.png")

# %% [markdown]
# ### §10′ X-ray α_ox–L₂₅₀₀ parity (tengri vs AGNFITTER-RX)
#
# tengri's `just2007` relation vs AGNFITTER-RX's `α_ox = −0.137 log L₂₅₀₀ +
# 2.638`, with the residual below.

# %%
aox_t = np.array([float(alpha_ox_from_l2500(x, relation="just2007")) for x in l2500])
aox_ref = -0.137 * np.log10(l2500) + 2.638
fig, (ax, axr) = plt.subplots(
    2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax.plot(np.log10(l2500), aox_t, "C1-", lw=1.6, label="tengri  just2007")
ax.plot(np.log10(l2500), aox_ref, "k--", lw=1.2, label="AGNFITTER-RX  Just+2007")
ax.set_ylabel(r"$\alpha_{ox}$")
ax.set_title(r"$\alpha_{ox}$–$L_{2500}$ parity")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
axr.axhline(0.0, color="0.5", lw=0.8)
axr.plot(np.log10(l2500), aox_t - aox_ref, "C1-", lw=1.2)
axr.set_ylabel(r"$\Delta\alpha_{ox}$", fontsize=9)
axr.set_xlabel(r"$\log_{10}\ L_{2500\,\AA}$ [erg/s/Hz]")
axr.grid(True, alpha=0.3)
_aox_dmax = float(np.max(np.abs(aox_t - aox_ref)))
print(f"§10 α_ox parity: max |tengri − AGNFITTER-RX| = {_aox_dmax:.2e}")
fig.tight_layout()
save_fig("agnfitter_10c_alphaox_residual.png")
plt.show()

# %% [markdown]
# ## §11 Radio
#
# AGNFITTER-RX models AGN core/jet radio with a simple power law (SPL,
# α = −0.75, exponential cutoff at 10¹³ Hz) or a double power law (DPL,
# Eq. 9–10: a self-absorption turnover ν_t and a synchrotron-aging cutoff
# ν_cut). tengri ships both — `radio_agn` (SPL) and `radio_agn_dpl`, the
# latter implementing AGNFITTER-RX Eq. 9–10 exactly. The driver carries the
# upstream formulas verbatim (`A.agn_radio_spl` / `A.agn_radio_dpl`), so
# both panels overlay tengri against AGNFITTER-RX itself at matched
# parameters, not against a hand-written power law. DPL parameters sit
# inside AGNFITTER-RX's own sampling grid (α₂ ∈ [−1, 0)).
#
# One deliberate difference: above ~300 GHz tengri's radio blocks return
# exactly zero (the far-IR belongs to the dust components), while
# AGNFITTER-RX lets the exponential cutoff decay smoothly through the THz
# range. The comparison band below stops at 300 GHz; the difference beyond
# it is visible in the top panel.

# %%
import jax.numpy as jnp

from tengri.radio import radio_agn, radio_agn_dpl

freq = np.geomspace(1e8, 1e12, 400)  # 0.1–1000 GHz
wave_radio = jnp.asarray(U.C_ANGSTROM_PER_S / freq)
L_AGN_BOL = 1e45  # erg/s
_DPL_PARS = dict(alpha1=-0.75, alpha2=-0.1, log_nu_t=10.0, log_nu_cut=13.0)
L_spl = np.asarray(radio_agn(wave_radio, L_AGN_BOL, radio_loudness=1.0, alpha_agn=0.75))
L_dpl = np.asarray(radio_agn_dpl(wave_radio, L_AGN_BOL, radio_loudness=1.0, **_DPL_PARS))
wave_radio = np.asarray(wave_radio)

_, F_spl_af = A.agn_radio_spl(freq)  # upstream SPL: alpha=-0.75, nu_t=1e9, cutoff 1e13
_, F_dpl_af = A.agn_radio_dpl(freq, **_DPL_PARS)

_nu5 = U.C_ANGSTROM_PER_S / 5e9


def _norm5(freq_hz, F):
    """Normalize a radio shape to 1 at 5 GHz."""
    return F / np.interp(5e9, freq_hz, F)


fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.loglog(freq / 1e9, _norm5(freq, F_spl_af), "C0-", lw=2.2, alpha=0.45,
          label=r"AGNFITTER-RX  SPL ($\alpha=-0.75$)")
ax.loglog(freq / 1e9, norm_at(wave_radio, L_spl, _nu5), "C0--", lw=1.5,
          label="tengri  radio_agn")
ax.loglog(freq / 1e9, _norm5(freq, F_dpl_af), "C1-", lw=2.2, alpha=0.45,
          label="AGNFITTER-RX  DPL (Eq. 9–10)")
ax.loglog(freq / 1e9, norm_at(wave_radio, L_dpl, _nu5), "C1--", lw=1.5,
          label="tengri  radio_agn_dpl")
ax.set_xlabel(r"$\nu$ [GHz]")
ax.set_ylabel(r"$L_\nu$ (norm. at 5 GHz)")
ax.set_title("AGN core/jet radio — tengri on AGNFITTER-RX's own SPL/DPL")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_11a_radio_agn.png")

# %% [markdown]
# ### §11′ Radio SPL/DPL parity ratios (tengri vs AGNFITTER-RX)
#
# The ratio of each tengri radio model to the AGNFITTER-RX formula over
# 0.1–300 GHz (both normalized at 5 GHz). The SPL ratio drifts up to ~3%
# at 300 GHz because AGNFITTER-RX's SPL carries its exponential cutoff
# (exp(−ν/10¹³ Hz)) while tengri's SPL is a pure power law within its
# support; the DPL formulas are identical term by term.

# %%
_band = freq <= 3e11  # 0.1–300 GHz comparison band
t_spl = np.asarray(norm_at(wave_radio, L_spl, _nu5))
t_dpl = np.asarray(norm_at(wave_radio, L_dpl, _nu5))
ratio_spl = np.where(_band & (t_spl > 0), t_spl / _norm5(freq, F_spl_af), np.nan)
ratio_dpl = np.where(_band & (t_dpl > 0), t_dpl / _norm5(freq, F_dpl_af), np.nan)
fig, (ax, axr) = plt.subplots(
    2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)
ax.loglog(freq / 1e9, np.where(t_spl > 0, t_spl, np.nan), "C0-", lw=1.6,
          label="tengri  radio_agn (SPL)")
ax.loglog(freq / 1e9, _norm5(freq, F_spl_af), "C0:", lw=1.4, label="AGNFITTER-RX  SPL")
ax.loglog(freq / 1e9, np.where(t_dpl > 0, t_dpl, np.nan), "C1-", lw=1.6,
          label="tengri  radio_agn_dpl")
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
print(f"§11 radio parity (0.1-300 GHz): SPL max |ratio − 1| = {_spl_dmax:.2e}   "
      f"DPL max |ratio − 1| = {_dpl_dmax:.2e}")
fig.tight_layout()
save_fig("agnfitter_11c_radio_spl_residual.png")
plt.show()

# %% [markdown]
# ## §10b X-ray corona: exact parity, then tengri's default extras
#
# Given the 2500 Å disk luminosity, both codes build the same corona: a
# Γ = 1.8 power law in F_ν with a 300 keV exponential cutoff, normalized at
# 2 keV through the Just+2007 α_ox relation (AGNFITTER-RX's cutoff constant
# ``exp(−ν/7.254·10¹⁹ Hz)`` *is* 300 keV). The **left panel** compares the
# bare prescriptions — tengri's `xray_agn_corona_from_disc` with its
# X-CIGALE extensions switched off (`apply_anisotropy=False`) against
# AGNFITTER-RX's disk X-ray extension at the same L₂₅₀₀. The residual is a
# flat ~+1% (tengri adds a constant 1% Thomson-scattered fraction to the
# absorbed primary) plus soft-X-ray curvature from tengri's tbabs
# photoelectric absorption at its default column log N_H = 20 — both printed.
#
# The **right panel** shows what tengri's *defaults* add on top of the bare
# corona, because they are on by default and AGNFITTER-RX has no analog:
# the Yang et al. (2022) viewing-angle anisotropy (normalized at the 30°
# X-CIGALE anchor, so even a face-on sightline sits ×1.07 above isotropic)
# and the tbabs/Compton line-of-sight absorption. For AGNFITTER-RX-parity
# work, switch the anisotropy off.
#
# On the host side, tengri adds an X-ray-binary floor (`xray_xrb`,
# Mineo et al. 2014) as a bona-fide SED component. AGNFITTER-RX's code has
# no host X-ray component; its X-ray physics beyond the corona lives in two
# *priors* — a Lusso & Risaliti (2016)-calibrated α_ox prior (σ = 0.4) and
# a Stern (2015) 6 µm ↔ 2–10 keV prior (σ = 0.5) — which shape posteriors
# but never emit flux. The XRB curve is plotted for scale.

# %%
from tengri.xray import xray_agn_corona_from_disc, xray_xrb

L_2500 = 1.0e30  # erg/s/Hz — representative bright-Seyfert disk
wave_x = np.geomspace(1e-2, 1e2, 600)  # ~600 keV down to ~0.12 keV
# Bare corona: X-CIGALE anisotropy off; tbabs stays at its log N_H = 20 default.
L_corona_bare = np.asarray(
    xray_agn_corona_from_disc(
        jnp.asarray(wave_x), L_2500, delta_alpha_ox=0.0, apply_anisotropy=False
    )
)
# tengri defaults: Yang+2022 anisotropy on (30°-anchored), face-on sightline.
L_corona_default = np.asarray(
    xray_agn_corona_from_disc(jnp.asarray(wave_x), L_2500, delta_alpha_ox=0.0, cos_inc=1.0)
)
# AGNFITTER-RX disk X-ray extension off the THB21 disk, anchored to the same L_2500.
w_thb, L_thb = A.disk_template("THB21")
L_thb_at_2500 = norm_at(w_thb, L_thb, 2500.0) * L_2500
xw_af, xL_af = A.disk_xray_extension(w_thb, L_thb_at_2500, scatter=0.0)
# Host XRB floor for a star-forming host (SFR ~ 5, M* ~ 1e10).
L_xrb = np.asarray(xray_xrb(jnp.asarray(wave_x), sfr=5.0, stellar_mass=1e10))

fig, (axl, axr) = plt.subplots(1, 2, figsize=(12.5, 4.8))
axl.loglog(xw_af, xL_af, "C0-", lw=1.5, label="AGNFITTER-RX  disk X-ray extension")
axl.loglog(wave_x, L_corona_bare, "C1--", lw=1.5,
           label="tengri  corona (anisotropy off)")
axl.loglog(wave_x, L_xrb, "C2:", lw=1.4, label="tengri  xray_xrb (host floor, Mineo+14)")
axl.set_xlim(1e-2, 1e2)
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
axr.semilogx(wave_x, _ratio_default, "C3-", lw=1.5,
             label="tengri defaults (Yang+22 anisotropy, face-on)")
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
# Quantify over the 0.5-100 keV band (lambda ~ 0.12-25 A).
_hard = (wave_x > 0.12) & (wave_x < 25.0) & np.isfinite(_ratio_bare)
print(
    f"§10b corona parity (0.5-100 keV): bare median ratio = "
    f"{np.nanmedian(_ratio_bare[_hard]):.4f} (the +1% scattered fraction), "
    f"max |ratio-1| = {np.nanmax(np.abs(_ratio_bare[_hard] - 1.0)):.3f}; "
    f"defaults median ratio = {np.nanmedian(_ratio_default[_hard]):.4f} "
    f"(x1.072 = the 30 deg-anchored Yang+22 anisotropy, face-on)"
)

# %% [markdown]
# ## §11b Star-formation radio (Bell 2003) vs AGNFITTER-RX's S17_radio
#
# AGNFITTER-RX delivers the host's radio emission *inside* its starburst
# template: the `S17_radio` model extends each Schreiber+2018 (T_dust,
# f_PAH) dust SED into the radio with the Bell (2003) IR–radio correlation
# (q_IR = 2.64, synchrotron plus a thermal free-free component that
# flattens the slope near 10 GHz). tengri keeps the two pieces separate —
# `schreiber2018` for the dust, `radio_sfr_bell2003` for the radio — so the
# genuine comparison is the *sum*, anchored to the same total IR
# luminosity. The panel overlays tengri's dust + radio against the upstream
# S17_radio template at (T_dust = 35 K, f_PAH = 0.02), and prints the
# 1.4 GHz ratio — which turns out to be a precise measurement of a q_IR
# bookkeeping difference between the codes, quantified below the panel.

# %%
from tengri.dust import DUST_EMISSION_MODELS as _DEM
from tengri.radio import radio_sfr_bell2003

w_afr, L_afr = A.cold_dust_radio_template(tdust=35.0, fpah=0.02)
_axes_radio = A.cold_dust_radio_axes()
_t_idx = int(np.argmin(np.abs(_axes_radio["tdust"] - 35.0)))
# Upstream tabulates LIR per T_dust node in L_sun; convert to erg/s.
L_IR_NODE = float(_axes_radio["lir_conv"][_t_idx]) * 3.826e33

wave_all = np.geomspace(1e4, 3e9, 1200)  # 1 µm – 30 cm
L_dust = np.asarray(
    _DEM["schreiber2018"](jnp.asarray(wave_all), 1.0, dust_T=35.0, dust_f_pah=0.02)
)
# schreiber2018 returns a normalized shape; scale so its 8-1000 µm integral
# equals the upstream node's L_IR, then add the Bell-2003 radio at that L_IR.
_nu_all = U.C_ANGSTROM_PER_S / wave_all
_ir_band = (wave_all > 8e4) & (wave_all < 1e7)
_order = np.argsort(_nu_all)
L_dust_ir = -float(np.trapezoid(np.where(_ir_band, L_dust, 0.0)[_order], _nu_all[_order]))
L_dust = L_dust * (L_IR_NODE / abs(L_dust_ir))
L_radio = np.asarray(radio_sfr_bell2003(jnp.asarray(wave_all), L_IR_NODE, q_ir=2.64))
L_te_total = L_dust + L_radio

fig, ax = plt.subplots(figsize=(8.2, 4.8))
msk_af = (w_afr > 1e4) & (w_afr < 3e9)
ax.loglog(w_afr[msk_af], L_afr[msk_af] / np.max(L_afr[msk_af]), "C0-", lw=2.2, alpha=0.5,
          label="AGNFITTER-RX  S17_radio (dust + Bell 03 radio)")
_peak_te = float(np.max(np.where((wave_all > 1e4), L_te_total, 0.0)))
ax.loglog(wave_all, L_te_total / _peak_te, "C1--", lw=1.5,
          label="tengri  schreiber2018 + radio_sfr_bell2003")
ax.axvline(U.C_ANGSTROM_PER_S / 1.4e9, color="0.7", ls=":", lw=1, label="1.4 GHz")
ax.set_xlim(1e4, 3e9)
ax.set_ylim(1e-8, 3)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ (norm. at FIR peak)")
ax.set_title("Host dust + star-formation radio, joined at matched $L_{IR}$")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_11b_radio_sf.png")

# %%
# Quantify at 1.4 GHz (peak-normalized shapes). The radio-to-FIR offset is a
# pure q_IR bookkeeping difference: the upstream template's baked radio
# corresponds to q_IR ≈ 2.92 measured against its OWN 8-1000 µm integral
# (its Bell normalization is applied to the tabulated LIR_conv, a smaller
# quantity than the template integral), while tengri applies q_IR = 2.64 to
# the 8-1000 µm integral directly. 2.64 + log10(ratio) recovers the
# upstream's effective q — printed as the check.
_lam_14 = U.C_ANGSTROM_PER_S / 1.4e9
_af_14 = np.interp(np.log10(_lam_14), np.log10(w_afr), L_afr / np.max(L_afr[msk_af]))
_te_14 = np.interp(np.log10(_lam_14), np.log10(wave_all), L_te_total / _peak_te)
_nu_afr = U.C_ANGSTROM_PER_S / w_afr
_o_afr = np.argsort(_nu_afr)
_ir_afr = (w_afr > 8e4) & (w_afr < 1e7)
_L_IR_self = np.trapezoid(np.where(_ir_afr, L_afr, 0.0)[_o_afr], _nu_afr[_o_afr])
_L14_abs = 10 ** np.interp(
    np.log10(_lam_14), np.log10(w_afr), np.log10(np.maximum(L_afr, 1e-300))
)
_q_upstream = np.log10((_L_IR_self / 3.75e12) / _L14_abs)
_lam_10g = U.C_ANGSTROM_PER_S / 1.0e10
_af_10 = np.interp(np.log10(_lam_10g), np.log10(w_afr), L_afr / np.max(L_afr[msk_af]))
_slope = np.log10(_af_10 / _af_14) / np.log10(1.0e10 / 1.4e9)
print(
    f"§11b SF radio:  L(1.4 GHz) tengri/AGNFITTER = {_te_14 / _af_14:.3f}  ==  "
    f"upstream effective q_IR = {_q_upstream:.3f} vs Bell's 2.64 "
    f"(2.64 + log10(ratio) = {2.64 + np.log10(_te_14 / _af_14):.3f}); "
    f"upstream 1.4-10 GHz slope alpha = {_slope:.2f} (free-free-flattened; "
    f"tengri's pure synchrotron is -0.80)"
)

# %% [markdown]
# ## Capstone — the radio-to-X-ray SED
#
# AGNFITTER-RX's reason for being: one model spanning `8 < log ν/Hz < 20`. We
# assemble the paper's winning AGN model — THB21 disk + CAT3D-Wind torus, with
# the α_ox X-ray corona and a DPL radio jet — and overlay tengri's matching
# composition (qsogen + cat3d_wind + corona + radio_agn_dpl). The component
# proportions are illustrative (a real fit sets them per source); the point is
# that both codes cover the same eleven decades of frequency with the same
# physical pieces, and the spectral *shapes* track across every band.

# %%
# Common observer grid spanning X-ray (~0.05 keV) to meter-wave radio.
nu_grid = np.geomspace(1e8, 1e20, 4000)
lam_grid = U.C_ANGSTROM_PER_S / nu_grid


def _assemble(disc_wL, torus_wL, xray_wL, radio_fL, *, f_torus=0.5, f_xray=3e-2, f_radio=2e-4):
    """Co-add normalized components onto nu_grid; disk anchored at 2500 Å."""
    disc = U.regrid(disc_wL[0], np.clip(disc_wL[1], 0, None), lam_grid)
    disc = disc / np.interp(2500.0, lam_grid[::-1], disc[::-1])
    tor = U.regrid(torus_wL[0], np.clip(torus_wL[1], 0, None), lam_grid)
    tor = tor / np.max(tor) * f_torus if np.max(tor) > 0 else tor
    xr = U.regrid(xray_wL[0], np.clip(xray_wL[1], 0, None), lam_grid)
    xr = xr / np.max(xr) * f_xray if np.max(xr) > 0 else xr
    # radio supplied as (freq, L_nu)
    rad = np.interp(
        nu_grid, radio_fL[0][::-1], np.clip(radio_fL[1], 0, None)[::-1], left=0, right=0
    )
    rad = rad / np.max(rad) * f_radio if np.max(rad) > 0 else rad
    return disc + tor + xr + rad


# Dedicated radio grid spanning the full DPL (turnover 1e10 → aging cutoff
# 1e13 Hz); the §11 grid stopped at 1e12 and truncated the jet mid-curve.
freq_cap = np.geomspace(1e8, 3e13, 800)
wave_cap = jnp.asarray(U.C_ANGSTROM_PER_S / freq_cap)
_dpl_kw = dict(radio_loudness=1.0, **_DPL_PARS)

# AGNFITTER-RX side.
w_d, L_d = A.disk_template("THB21")
w_t, L_t = A.torus_template("CAT3D", **_CAT3D_NODE)
xw, xL = A.disk_xray_extension(w_d, norm_at(w_d, L_d, 2500.0) * 1e30, scatter=0.0)
L_rad_dpl = np.asarray(radio_agn_dpl(wave_cap, 1e45, **_dpl_kw))
af_sed = _assemble((w_d, L_d), (w_t, L_t), (xw, xL), (freq_cap, L_rad_dpl))

# tengri side.
wd_t, Ld_t = tengri_disc("qsogen")
wt_t, Lt_t = tengri_torus("cat3d_wind", **_CAT3D_NODE_TE)
Lc_t = np.asarray(
    xray_agn_corona_from_disc(
        jnp.asarray(wave_x), 1e30, delta_alpha_ox=0.0, apply_anisotropy=False
    )
)
Lr_t = np.asarray(radio_agn_dpl(wave_cap, 1e45, **_dpl_kw))
te_sed = _assemble((wd_t, Ld_t), (wt_t, Lt_t), (wave_x, Lc_t), (freq_cap, Lr_t))

# Mask where every component is absent (grid edges) so each SED ends cleanly
# at its own coverage rather than plunging vertically to zero.
af_plot = np.where(af_sed > 0, nu_grid * af_sed, np.nan)
te_plot = np.where(te_sed > 0, nu_grid * te_sed, np.nan)

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.loglog(nu_grid, af_plot, "C0-", lw=1.6, label="AGNFITTER-RX  THB21 + CAT3D + corona + DPL")
ax.loglog(nu_grid, te_plot, "C1--", lw=1.6, label="tengri  qsogen + cat3d_wind + corona + DPL")
for nu_band, name in [(1.4e9, "radio"), (3e13, "IR"), (6e14, "opt"), (4.8e17, "2 keV")]:
    ax.axvline(nu_band, color="0.85", ls=":", lw=1)
    ax.text(
        nu_band,
        ax.get_ylim()[1],
        f" {name}",
        rotation=90,
        va="top",
        ha="left",
        fontsize=7,
        color="0.5",
    )
ax.set_xlim(1e8, 1e20)
ax.set_xlabel(r"$\nu$ [Hz]")
ax.set_ylabel(r"$\nu L_\nu$ (disk norm. at 2500 Å)")
ax.set_title(r"Radio-to-X-ray AGN SED ($8 < \log\,\nu/\mathrm{Hz} < 20$)")
ax.legend(fontsize=8, loc="lower center")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_full_sed_headtohead.png")

# %% [markdown]
# ### Capstone′ — the same composition as one `SEDModel.build`
#
# The panel above assembles the components by hand so the two codes can be
# normalized identically. In practice you would not do that: the entire
# AGNFITTER-RX-style model — host stars and dust, THB21-analog disc with
# its lines and FeII, CAT3D-Wind torus, α_ox corona, DPL jet — composes as
# *one* tengri model through the public grammar, and that is the object you
# fit. This cell builds it and plots the pipeline's own total SED across
# the same eleven decades; every AGNFITTER-RX-parity knob used in this
# notebook (`agn_ebv_disc`, the torus axes, the DPL turnover/cutoff) is a
# regular parameter of this one model and could be declared free.

# %%
m_full = SEDModel.build(
    ssp_data=ssp,
    sfh=SFH_FIDUCIAL,
    dust={
        "type": "two_component",
        "tau_bc": Fixed(0.3),
        "tau_diff": Fixed(0.1),
        "emission": {"type": "schreiber2018", "*": FIXED},
        "*": FIXED,
    },
    agn={
        "type": "composable",
        "disc": {"type": "qsogen", "*": FIXED},
        "lines": {"type": "qsogen", "*": FIXED},
        "feii": {"type": "qsogen_balmer", "*": FIXED},
        "torus": {
            "type": "cat3d_wind",
            "cos_inc": Fixed(1.0),
            "a_cat3d": Fixed(-2.0),
            "fwd_cat3d": Fixed(1.75),
            "*": FIXED,
        },
        "agn_log_lbol": Fixed(12.0),
        "agn_ebv_disc": Fixed(0.0),
        "agn_polar_ebv": Fixed(0.0),
        "*": FIXED,
    },
    xray={"type": "yang20"},
    radio={"sf": {"type": "bell2003"}, "agn": {"type": "dpl"}},
    redshift=Fixed(0.0),
)
s_full = m_full.predict_state({})
w_full = np.asarray(s_full.wave)
# The pipeline's total rest-frame SED after every component has applied.
L_full = np.asarray(s_full.sed_intrinsic)
nu_full = U.C_ANGSTROM_PER_S / w_full
anchor_full = float(np.interp(2500.0, w_full, L_full))
nuL = np.where(L_full > 0, nu_full * L_full / (anchor_full * (U.C_ANGSTROM_PER_S / 2500.0)), np.nan)

fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.loglog(nu_full, nuL, "C1-", lw=1.4, label="tengri — one SEDModel.build, all components")
ax.set_xlim(1e8, 1e20)
ax.set_xlabel(r"$\nu$ [Hz]")
ax.set_ylabel(r"$\nu L_\nu$ (norm. at 2500 Å)")
ax.set_title("Full composed model: host + AGN + corona + jet in a single build")
ax.legend(fontsize=8, loc="lower center")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_capstone_composed.png")
print(
    f"capstone′  composed model wavelength coverage: {w_full.min():.2e} – {w_full.max():.2e} Å "
    f"({np.log10(nu_full.max() / nu_full.min()):.1f} decades of frequency)"
)

# %% [markdown]
# ## Summary
#
# At matched parameters, tengri matches AGNFITTER-RX's AGN model
# component by component, with every claim quantified in-notebook and every
# tengri model composed through the public `SEDModel.build` grammar: all
# four accretion-disk libraries (R06, SN12, KD18, THB21), the Prevot disc
# reddening, the four torus libraries (S04, NK08, SKIRTOR, CAT3D-Wind —
# three of them node-exact on these very published templates, including
# all three CAT3D-Wind axes), the Schreiber cold-dust
# family with its S17_radio extension, the α_ox–L₂₅₀₀ X-ray corona, and the
# SPL/DPL radio jets joined to the Bell-2003 star-formation radio. The
# notebook reproduces the paper's two central results directly: THB21's
# 0.7 µm Hα+[N II] bump (§9a) and the near-IR wind excess the CAT3D-Wind
# model is built to address (§9c, §9c‴), and spans the full radio-to-X-ray
# range AGNFITTER-RX was built for (capstone) — as one buildable,
# fittable model (capstone′).
#
# The residuals that remain are understood and stated where they appear:
#
# * **R06** — a pure carriage convention (L_ν vs νL_ν); in AGNFITTER-RX's
#   convention the two overlay to ~0 (§9a).
# * **Prevot reddening** — identical spectral shape; tengri pins the
#   published R_V = 2.72 where AGNFITTER-RX evaluates the raw UV fit at V
#   (k_raw(V) ≈ 2.468), a uniform 1.102× rescaling of the effective E(B−V)
#   (§4, §9b).
# * **KD18** — tengri's disc is luminosity-first (λ_Edd derived from L_bol
#   and M_BH); at matched nodes the residual is the documented
#   warm-Comptonization proxy, ≲0.17 dex in the disc window (§9a).
# * **Corona** — bit-matched prescription up to a flat +1% scattered
#   fraction; tengri's X-CIGALE extras (Yang+2022 anisotropy, tbabs) are
#   additions to switch off for AGNFITTER-RX-parity work (§10b).
# * **NK08** — a family-level comparison only: tengri carries the
#   FSPS/CLUMPY reduction, not AGNFITTER-RX's inclination-averaged
#   NK0_mean_1p (§9c).
# * **Host X-rays** — tengri models an XRB floor (Mineo+2014) as a real
#   component; AGNFITTER-RX has no host X-ray component and instead applies
#   α_ox (Lusso & Risaliti 2016, σ = 0.4) and 6 µm↔2–10 keV (Stern 2015,
#   σ = 0.5) priors during fitting (§10b).
# * **Radio > 300 GHz** — tengri's radio blocks end where the dust begins;
#   AGNFITTER-RX lets the exponential cutoff decay through the THz (§11).
# * **SF radio normalization** — upstream's S17_radio template carries an
#   effective q_IR ≈ 2.92 against its own 8–1000 µm integral (Bell's 2.64
#   applied to the smaller tabulated LIR_conv); tengri applies q_IR = 2.64
#   to the integral directly, so its 1.4 GHz output is 1.9× brighter at
#   matched L_IR — same correlation, different L_IR bookkeeping (§11b).
#
# For the cold dust, tengri's `schreiber2018` matches AGNFITTER-RX's S17
# dust + PAH SED to a median of ~0.05% at fixed (T_dust, f_PAH); the
# analytic `schreiber2016` remains available as a fast, differentiable —
# but PAH-approximate — alternative.
#
# ## References
#
# Every model compared above, with the section that uses it. The machine-
# readable BibTeX for all of these lives next to this notebook in
# `references.bib`; the key per entry is given in brackets.
#
# **Accretion disks (§9a)**
# - Richards, G. T., et al. 2006, ApJS 166, 470 — R06 mean Type-1 quasar SED
#   [`richards2006sed`].
# - Slone, O. & Netzer, H. 2012, MNRAS 426, 656 — SN12 α-disk [`slone2012effects`].
# - Kubota, A. & Done, C. 2018, MNRAS 480, 1247 — KD18 [`kubota2018physical`].
# - Temple, M. J., Hewett, P. C. & Banerji, M. 2021, MNRAS 508, 737 — THB21 /
#   qsogen [`temple2021modelling`].
#
# **Tori (§9c)**
# - Silva, L., et al. 2004, MNRAS 355, 973 — S04 [`Silva2004`].
# - Nenkova, M., et al. 2008, ApJ 685, 160 — NK08 / CLUMPY [`nenkova2008agnII`].
# - Stalevski, M., et al. 2016, MNRAS 458, 2288 — SKIRTOR [`Stalevski2016`].
# - Hönig, S. F. & Kishimoto, M. 2017, ApJL 838, L20 — CAT3D-Wind
#   [`honig2017dusty`].
# - Yang, G., et al. 2020, MNRAS 491, 740 — X-CIGALE SKIRTOR + polar dust,
#   the implementation tengri's `skirtor` follows [`yang2020xcigale`].
#
# **Cold dust (§6)**
# - Schreiber, C., et al. 2018, A&A 609, A30 — S17 [`schreiber2018dust`].
# - Dale, D. A. & Helou, G. 2002, ApJ 576, 159 [`dale2002infrared`]; Chary, R. &
#   Elbaz, D. 2001, ApJ 556, 562 [`chary2001interpreting`] — DH02_CE01.
# - Dale, D. A., et al. 2014, ApJ 784, 83 — tengri `dale2014` [`dale2014two`].
#
# **X-ray (§10)**
# - Just, D. W., et al. 2007, ApJ 665, 1004 [`just2007x`]; Lusso, E. &
#   Risaliti, G. 2016, ApJ 819, 154 [`lusso2016tight`]; 2017, A&A 602, A79
#   [`lusso2017quasars`] — α_ox–L₂₅₀₀.
# - Mineo, S., et al. 2014, MNRAS 437, 1698 — host XRB / SFR [`mineo2014x`].
#
# **Radio (§11)**
# - Azadi, M., et al. 2020 (arXiv:2011.03130) — radio AGN [`azadi2020disentangling`].
# - Bell, E. F. 2003, ApJ 586, 794 — radio–FIR q_IR [`bell2003estimating`].
#
# **Stellar populations & attenuation (§1–§5)**
# - Bruzual, G. & Charlot, S. 2003, MNRAS 344, 1000 [`bruzual2003stellar`];
#   Chabrier, G. 2003, PASP 115, 763 [`chabrier2003galactic`].
# - Calzetti, D., et al. 2000, ApJ 533, 682 [`calzetti2000dust`]; Prevot, M. L.,
#   et al. 1984, A&A 132, 389 — SMC reddening [`prevot1984typical`].
#
# **Codes & inference**
# - Martínez-Ramírez, L. N., et al. 2024, A&A 688, A46 — AGNFITTER-RX
#   [`martinez2024agnfitter`].
# - Hearin, A. P., et al. 2023, MNRAS 521, 1741 — DSPS [`hearin2023dsps`].
# - Buchner, J. 2019, PASP 131, 108005 — UltraNest [`buchner2019collaborative`].

# %% [markdown]
# ### BibTeX
#
# The complete machine-readable bibliography (printed below from
# `references.bib` so it never drifts from the file).

# %%
_bib_path = _HERE / "references.bib"
print(_bib_path.read_text())
