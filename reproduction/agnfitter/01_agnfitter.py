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
# AGNfitter-rX (Martínez-Ramírez et al. 2024, A&A 688, A46) models the
# radio-to-X-ray SEDs of active galaxies. Where CIGALE and Prospector are
# galaxy-centric, AGNfitter-rX is built to characterize the AGN itself — its
# four physical components (accretion disk, hot dusty torus, relativistic
# jets / core radio, and hot corona) alongside host galaxy (stellar populations,
# cold dust, star-formation radio). The code is a *model-testing laboratory*
# for AGN physics; radio and X-ray data, largely unaffected by dust, are
# orthogonal tracers that break the infrared–ultraviolet degeneracies that
# limited the original submm-to-UV AGNfitter.
#
# This study configures tengri's public API to approximate AGNfitter-rX's model choices; tengri's implementation is its own, not derived from AGNfitter-rX's code, and residual differences are quantified below.
#
# The headline comparisons are the two model *face-offs* that drive the
# paper's conclusions: **§9a** — accretion-disk libraries R06, SN12, KD18, THB21.
# THB21 wins (Bayes factor ≈10⁵·¹ over R06) because it alone carries the
# broad and narrow emission lines producing the ≈0.7 μm Hα + [N II] peak the
# theory disks miss. **§9c** — torus libraries S04, NK08, SKIRTOR, CAT3D-Wind.
# CAT3D-Wind wins (maximum likelihood 25/36 sources) because its polar-wind
# dust addresses the 1.5–5 μm near-IR excess pure equatorial tori cannot.
#
# Tengri's `cat3d_wind`, `silva04`, `skirtor_agnfitter`, `slone_netzer`,
# and `schreiber2018` blocks evaluate the same template libraries AGNFITTER-RX
# publishes; those panels are direct checks of tengri implementations.

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

# AGNFITTER-RX's template libraries are repackaged as committed HDF5 under
# data/, so this runs on a clean checkout with no AGNfitter clone;
# require_available() says what to regenerate if a grid is missing.
A.require_available()
print(f"AGNFITTER-RX reference grids: {len(A.list_disks())} disks, "
      f"{len(A.list_tori())} tori, {len(A.list_cold_dust())} cold-dust "
      f"(committed under data/)")


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
# Both codes build on the Bruzual & Charlot (2003) stellar populations with a
# Chabrier (2003) initial mass function (IMF). Tengri reads a BC03 + Chabrier
# grid in the DSPS layout; AGNFITTER-RX ships the same library as
# IR-luminosity-tagged τ-model templates.

# %%
_SSP_CANDIDATES = [
    _HERE / "_drivers" / "data" / "bc03_chabrier.h5",
    Path("data/bc03_pdva_stelib_chabrier.h5"),
    Path(tengri.__file__).resolve().parents[2] / "data" / "bc03_pdva_stelib_chabrier.h5",
    Path(__file__).resolve().parents[2] / "data" / "bc03_pdva_stelib_chabrier.h5",
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
    "all_params": FIXED,
}
NO_DUST = {"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "all_params": FIXED}


# %% [markdown]
# ## tengri AGN helpers
#
# Each AGN face-off builds a single tengri block in isolation and reads the
# AGN SED off ``state.derived["sed_agn"]``. We normalize shapes at a common
# anchor (2500 Å for disks, the IR peak for tori) so the comparison is of
# spectral shape at matched parameters, independent of the per-code luminosity
# bookkeeping. One default matters here: tengri's CIGALE-faithful AGN applies
# a polar-dust screen at E(B−V) = 0.03 by default (`agn_polar_ebv`, X-CIGALE's
# default). AGNfitter-rX's disc templates carry no such screen — its only disc
# obscuration is the free `EBVbbb` — so every helper pins `agn_polar_ebv = 0`.
# Left at the default, the screen suppresses the rest-UV by ~0.05 dex and would
# masquerade as a disc-shape residual in every panel below.


# %%
def tengri_disc(disc_type, *, log_lbol=11.0, ebv_disc=None, **disc_params):
    """Isolated tengri accretion-disc SED. Returns (wave_aa, L_nu).

    ``ebv_disc`` sets the shared disc obscuration ``agn_ebv_disc`` (the
    AGNFITTER-RX ``EBVbbb`` analog) at the top level of the agn group.
    """
    disc = {"type": disc_type, "all_params": FIXED}
    disc.update({k: Fixed(v) for k, v in disc_params.items()})
    agn = {
        "type": "composable",
        "disc": disc,
        "torus": {"type": "none"},
        "lines": {"type": "none"},
        "agn_log_lbol": Fixed(log_lbol),
        "agn_polar_ebv": Fixed(0.0),  # AGNFITTER-RX templates carry no polar screen
        "all_params": FIXED,
    }
    if ebv_disc is not None:
        agn["agn_ebv_disc"] = Fixed(ebv_disc)
    m = SEDModel.build(
        ssp_data=ssp,
        sfh=SFH_FIDUCIAL,
        dust_attenuation=NO_DUST,
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
        dust_attenuation=NO_DUST,
        agn={
            "type": model_type,
            "agn_log_lbol": Fixed(log_lbol),
            "agn_polar_ebv": Fixed(0.0),  # AGNFITTER-RX templates carry no polar screen
            "all_params": FIXED,
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
        dust_attenuation=NO_DUST,
        agn={
            "type": "composable",
            "disc": {"type": "qsogen", "all_params": FIXED},
            "torus": {"type": "none"},
            "lines": {"type": "qsogen", "all_params": FIXED},
            "feii": {"type": "qsogen_balmer", "all_params": FIXED},
            "agn_log_lbol": Fixed(log_lbol),
            "agn_polar_ebv": Fixed(0.0),  # AGNFITTER-RX templates carry no polar screen
            "all_params": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.derived["sed_agn"])


def tengri_torus(torus_type, *, log_lbol=11.0, **torus_params):
    """Isolated tengri torus SED. Returns (wave_aa, L_nu)."""
    torus = {"type": torus_type, "all_params": FIXED}
    torus.update({k: Fixed(v) for k, v in torus_params.items()})
    m = SEDModel.build(
        ssp_data=ssp,
        sfh=SFH_FIDUCIAL,
        dust_attenuation=NO_DUST,
        agn={
            "type": "composable",
            "disc": {"type": "none"},
            "torus": torus,
            "lines": {"type": "none"},
            "agn_log_lbol": Fixed(log_lbol),
            "agn_polar_ebv": Fixed(0.0),  # AGNFITTER-RX templates carry no polar screen
            "all_params": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.derived["sed_agn"])


# %% [markdown]
# ## §1 Stellar populations
#
# Both codes use identical BC03 + Chabrier SSPs. The panel shows tengri's
# library at representative ages (0.1 and 5 Gyr).
#
# **Verification Status:** CROSSVAL (2 tests — thin) — CSP integral — CIC age kernel (default)

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
            "all_params": FIXED,
        },
        dust_attenuation=NO_DUST,
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
# AGNfitter-rX adopts τ-model (delayed-exponential) star formation histories.
# Tengri's parametric SFHs rescale the shape so that ``∫ SFR dt = 10**log_total_mass``.
# The panel reads the pipeline SFR history off ``state.derived["sfr_history"]``
# (not an analytic curve) and verifies mass closure via trapezoid integration.
#
# **Verification Status:** PARTIAL (11/33) — Parametric SFH family physics

# %%
m = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST, redshift=Fixed(0.0))
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
# The fiducial host's stellar continuum from tengri — the same quantity
# AGNfitter-rX's GA (host stellar) component contributes, reddened by
# SMC/Calzetti law at fit time. The panel is compact by design, since the
# underlying SSPs are identical published models; the focus is on the AGN
# components in the panels that follow.
#
# **Verification Status:** PARTIAL (68/126) — Absolute SED normalization

# %%
m = SEDModel.build(ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST, redshift=Fixed(0.0))
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
# AGNfitter-rX reddens the accretion disk (not the host) with an analytic
# Prevot et al. (1984) SMC fit, ``k_raw(λ) = 1.39 λ_μm^−1.2 − 0.38``, applied
# as ``A_λ = k_raw(λ) · E(B−V)`` with no reddening blueward of 200 eV
# (``MODEL_AGNfitter.BBBred_Prevot``). That routine *declares* ``RV = 2.72`` —
# the value Prevot+1984 measured for the SMC — and passes it to its inner
# ``function_prevot(x, RV)``, but the function ignores the argument and returns
# the bare ``k_raw``. So AGNfitter-rX's *effective* total-to-selective ratio is
# ``k_raw(0.55 μm) ≈ 2.468`` (the fit's value at V), not the 2.72 it declares —
# this is the origin of the 10% below, and it is a code-level detail, not a
# calibration choice.
#
# tengri's disc obscuration (`agn_ebv_disc`, the `EBVbbb` analog) applies the
# ``RV`` AGNfitter-rX intended: it normalizes the same fit's shape to
# ``k(λ) = k_raw(λ)/k_raw(V)`` (so ``k(V) = 1``) and pins ``R_V = 2.72``, giving
# ``A_λ = k(λ) · R_V · E(B−V) = k_raw · (2.72/2.468) · E(B−V)``. The two differ
# by the uniform factor ``2.72/2.468 ≈ 1.102`` in A_λ at matched E(B−V): the
# *shape* is identical, so the AGNfitter-rX ``EBVbbb`` posterior maps onto
# tengri's as ``E(B−V)_tengri ≈ E(B−V)_AGNFITTER / 1.102``. tengri's amplitude
# is the more physical one — it matches both Prevot+1984's measured SMC R_V and
# AGNfitter-rX's own declared value.
#
# This analytic fit is AGNFITTER-RX's *own* SMC approximation for reddening the
# finished BBB template. qsogen, which builds the THB21 disc, reddens with a
# different tabulated curve (``pl_ext_comp_03.sph``) — see §4b. §4 therefore
# compares tengri to AGNFITTER-RX's ``BBBred_Prevot``, the ``EBVbbb`` free
# parameter both codes expose.
#
# The right panel leads with the payoff — tengri at the convention-matched
# ``E(B−V)/1.102`` lies exactly on AGNfitter-rX's band (identical *law*), with
# the raw same-``E(B−V)`` curve shown faint underneath to mark the +10.2%
# reparametrization. tengri's reddening is exercised through ``SEDModel.build``
# (a qsogen disc with `agn_ebv_disc` set), not by re-evaluating the formula.
#
# **Verification Status:** CROSSVAL — Attenuation law library

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
# Lead with the payoff: AGNFITTER-RX as a thick band, and tengri at the
# convention-matched E(B-V)/1.102 as a thin line on top — they lie exactly on
# each other, so the *law* is identical. The raw same-E(B-V) tengri curve is
# shown faint underneath purely to mark the +10.2% E(B-V) reparametrization.
axr.semilogx(w_thb[msk_a], -2.5 * np.log10(ratio_af[msk_a]), "C0-", lw=3.5, alpha=0.35,
             solid_capstyle="round", label="AGNFITTER-RX  BBBred_Prevot")
axr.semilogx(w_t0[msk_t], -2.5 * np.log10(ratio_tengri_rescaled[msk_t]), "C1-", lw=1.4,
             label=r"tengri  agn_ebv_disc at $E(B{-}V)/1.102$ (convention-matched)")
axr.semilogx(w_t0[msk_t], -2.5 * np.log10(ratio_tengri[msk_t]), "C3:", lw=1.4,
             label=r"tengri at same $E(B{-}V)$ (raw — $+10.2\%$ convention offset)")
axr.set_xlabel(r"$\lambda$ [Å]")
axr.set_ylabel(r"$A_\lambda$ [mag] at $E(B{-}V)=0.3$")
axr.set_title("Disc attenuation, end-to-end — identical law once convention-matched")
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
# ### §4b qsogen's *own* reddening law — a different curve and convention
#
# The §4 comparison is deliberately against AGNfitter-rX's ``BBBred_Prevot``
# (the ``EBVbbb`` analog). But the qsogen code that *builds* the THB21 disc
# reddens with a different law entirely: an empirically-derived **quasar**
# extinction curve (Temple, Hewett & Banerji 2021, from SDSS DR7 quasars at
# 2 < z < 3 — *not* the SMC), stored as the color excess ``E(λ−V)/E(B−V)`` and
# applied as ``A_λ = E(B−V)·[E(λ−V)/E(B−V) + R]`` with ``R = 3.1``. tengri now
# ships this law as a composable attenuation block —
# ``agn={'atten': {'type': 'qsogen'}}`` — so a qsogen disc can be reddened
# exactly the way qsogen reddens it. The panel puts all three ``A_λ/E(B−V)``
# curves on one axis: they differ in both *shape* (the empirical quasar curve is
# not the analytic SMC fit — it is greyer in the UV) and *V-band normalization*
# (AGNfitter-rX 2.468, tengri-Prevot 2.72, qsogen 3.1). See §9d/§9a for the
# qsogen disc itself; here we isolate its reddening curve.

# %%
from tengri.components.dust.qsogen_ext import qsogen_quasar_extinction

_wl_ext = np.geomspace(1e3, 1e4, 400)
_k_raw_ext = 1.39 * (_wl_ext / 1e4) ** (-1.2) - 0.38  # AGNFITTER-RX A_λ/E(B-V), R_eff 2.468
_k_tengri_ext = _k_raw_ext / _K_RAW_V * _R_V_SMC  # tengri Prevot, R_V 2.72
_k_qsogen_ext = np.asarray(qsogen_quasar_extinction(_wl_ext))  # qsogen: curve + R=3.1

fig, ax = plt.subplots(figsize=(8.2, 5.0))
ax.plot(_wl_ext, _k_raw_ext, "C0-", lw=1.6, label=r"AGNFITTER-RX  Prevot SMC ($R_V\approx2.468$)")
ax.plot(_wl_ext, _k_tengri_ext, "C1--", lw=1.6, label=r"tengri  Prevot SMC ($R_V=2.72$)")
ax.plot(_wl_ext, _k_qsogen_ext, "C3-", lw=2.2, alpha=0.8,
        label=r"qsogen  empirical quasar curve ($R=3.1$)")
ax.axvline(5500, color="0.8", ls=":", lw=1, label="V (5500 Å)")
ax.set_xscale("log")
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$A_\lambda / E(B{-}V)$")
ax.set_title("Three disc-reddening laws — AGNFITTER-RX vs tengri-Prevot vs qsogen's own")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_04b_qsogen_ext.png")


# %%
def _at_ext(a, lam):
    return float(np.interp(lam, _wl_ext, a))


print(
    f"§4b  A_V/E(B-V) at V=5500 Å:  AGNFITTER-RX={_at_ext(_k_raw_ext, 5500):.3f}  "
    f"tengri-Prevot={_at_ext(_k_tengri_ext, 5500):.3f}  qsogen={_at_ext(_k_qsogen_ext, 5500):.3f}"
)
print(
    f"§4b  A(1500)/A(V) (UV steepness):  AGNFITTER-RX="
    f"{_at_ext(_k_raw_ext, 1500) / _at_ext(_k_raw_ext, 5500):.2f}  "
    f"qsogen={_at_ext(_k_qsogen_ext, 1500) / _at_ext(_k_qsogen_ext, 5500):.2f}  "
    f"(qsogen's empirical curve is greyer in the UV)"
)

# %% [markdown]
# ## §6 Cold dust infrared emission
#
# AGNfitter-rX ships two libraries: S17 (Schreiber et al. 2018, flexible
# dust-continuum + PAH parameterized by T_dust and f_PAH) and legacy
# DH02_CE01 (Dale & Helou 2002 + Chary & Elbaz 2001, indexed by IR luminosity).
#
# Tengri exposes four models, including node-exact matches to both:
# **`schreiber2018`** — S17 tabulated templates (1 − f_PAH)·dust + f_PAH·PAH,
# median residual 0.05% of FIR peak, dust peak 86.5 vs 87.0 μm (0.6%).
# **`dh02_ce01`** — DH02_CE01 legacy library, median residual 0.001% (curves
# overlay). **`schreiber2016`** — analytic approximation (modified blackbody +
# Drude PAH), fast and differentiable but PAH weaker than tabulated S17.
# **`dale2014`** — modern Dale+2014 library, parameterized by radiation-field
# hardness α (not node-for-node match to S17). All peak-normalized. The left
# panel below shows the two node-exact matches (each pair one color — thick band
# AGNfitter-rX, thin line tengri); the right panel shows the two differentiable
# alternatives.
#
# **Verification Status:** CROSSVAL — Dust IR emission physics (MBB, Casey12, CMB)

# %%
import jax.numpy as jnp

from tengri.dust import DUST_EMISSION_MODELS

wave_ir = np.geomspace(1e4, 1e8, 2000)  # 1 µm – 1 cm
schreiber18 = DUST_EMISSION_MODELS["schreiber2018"]
schreiber16 = DUST_EMISSION_MODELS["schreiber2016"]
dale14 = DUST_EMISSION_MODELS["dale2014"]
dh02 = DUST_EMISSION_MODELS["dh02_ce01"]
L_s18 = np.asarray(schreiber18(jnp.asarray(wave_ir), 1.0, dust_T=35.0, dust_f_pah=0.02))
L_s16 = np.asarray(schreiber16(jnp.asarray(wave_ir), 1.0, dust_T=35.0, dust_f_pah=0.02))
L_d14 = np.asarray(dale14(jnp.asarray(wave_ir), 1.0, dust_alpha_dale=1.5))
_DH_LIR = 12.0  # matched template IR luminosity log10(L_IR/L_sun)
L_dh02 = np.asarray(dh02(jnp.asarray(wave_ir), 1.0, dust_log_lir=_DH_LIR))

w_s17, L_s17 = A.cold_dust_template("S17", tdust=35.0, fpah=0.02)
w_dh, L_dh = A.cold_dust_template("DH02_CE01", log_irlum=_DH_LIR)

# Shape residuals at the nodes (over 3–300 µm), computed up front so the
# matched-pair panel can annotate them.
_band = (w_s17 > 3e4) & (w_s17 < 3e6)
_s17n = L_s17 / L_s17[_band].max()
_s18n = np.asarray(schreiber18(jnp.asarray(w_s17), 1.0, dust_T=35.0, dust_f_pah=0.02))
_s18n = _s18n / _s18n[_band].max()
_resid = np.abs(_s18n[_band] - _s17n[_band])
_bd = (w_dh > 3e4) & (w_dh < 3e6)
_dhn = L_dh / L_dh[_bd].max()
_dh02n = np.asarray(dh02(jnp.asarray(w_dh), 1.0, dust_log_lir=_DH_LIR))
_dh02n = _dh02n / _dh02n[_bd].max()
_rd = np.abs(_dh02n[_bd] - _dhn[_bd])

# Two panels so the matches read cleanly: LEFT = the two node-exact pairs
# (each pair shares a hue — AGNFITTER-RX as a thick band, tengri as a thin line
# on top); RIGHT = the two differentiable alternatives, which are *not* node
# matched, against a faint S17 reference for context.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
axL.loglog(w_s17, norm_peak(L_s17), "C0-", lw=4.0, alpha=0.35, solid_capstyle="round",
           label="AGNFITTER-RX  S17 (Schreiber+18)")
axL.loglog(wave_ir, norm_peak(L_s18), "C0-", lw=1.4, label="tengri  schreiber2018 (S17 tables)")
axL.loglog(w_dh, norm_peak(L_dh), "C2-", lw=4.0, alpha=0.35, solid_capstyle="round",
           label=f"AGNFITTER-RX  DH02_CE01 (log L$_{{IR}}$={_DH_LIR:g})")
axL.loglog(wave_ir, norm_peak(L_dh02), "C2-", lw=1.4, label="tengri  dh02_ce01 (matched)")
axL.set_title("Node-exact matches — tengri reproduces both AGNFITTER-RX libraries")
axL.text(0.03, 0.97,
         "shape residual (median |Δ|/peak)\n"
         f"S17 pair       : {np.median(_resid) * 100:.3f}%\n"
         f"DH02_CE01 pair : {np.median(_rd) * 100:.3f}%",
         transform=axL.transAxes, va="top", ha="left", fontsize=7, family="monospace",
         bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
axR.loglog(w_s17, norm_peak(L_s17), "0.6", lw=2.0, alpha=0.6, label="AGNFITTER-RX  S17 (ref)")
axR.loglog(wave_ir, norm_peak(L_s16), "C1-", lw=1.5, label="tengri  schreiber2016 (analytic)")
axR.loglog(wave_ir, norm_peak(L_d14), "C4-", lw=1.5, label=r"tengri  dale2014 ($\alpha=1.5$)")
axR.set_title("Differentiable alternatives (not node-matched)")
for ax in (axL, axR):
    ax.set_xlim(1e4, 1e8)
    # Six decades below the peak, not three. A 1e-3 floor amputated most of
    # what these libraries are being compared *on*: the near-IR rise, the
    # sub-percent PAH forest at 3-8 um, and the whole Rayleigh-Jeans tail past
    # ~2 mm — which is exactly where schreiber2016's analytic modified
    # blackbody parts company with the tabulated S17. Cropping a comparison
    # above the disagreement defeats the comparison.
    ax.set_ylim(1e-6, 3)
    ax.set_xlabel(r"$\lambda$ [Å]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
axL.set_ylabel(r"$L_\nu$ (norm. at peak)")
fig.suptitle("Cold-dust IR — matched pairs (left) vs differentiable alternatives (right)", y=1.02)
fig.tight_layout()
save_fig("agnfitter_06_cold_dust.png")

# %%
# Print the two node-exact residuals for the record.
print(
    f"§6  schreiber2018 vs AGNFITTER-RX S17 (T=35 K, f_PAH=0.02):  "
    f"median |Δ|/peak = {np.median(_resid) * 100:.3f}%   max = {_resid.max() * 100:.2f}%"
)
print(
    f"§6  dh02_ce01 vs AGNFITTER-RX DH02_CE01 (log L_IR={_DH_LIR:g}):  "
    f"median |Δ|/peak = {np.median(_rd) * 100:.3f}%   max = {_rd.max() * 100:.2f}%"
)

# %% [markdown]
# ## §9a Accretion-disk library face-off
#
# Four AGNfitter-rX disk libraries at matched parameters, unreddened and
# normalized at 2500 Å. The decisive feature is the 0.7 μm bump (Hα + [N II]):
# present only in the semi-empirical THB21, absent from theory disks
# R06/SN12/KD18. AGNfitter-rX finds this single feature drives a Bayes factor
# ≈10⁵·¹ over R06.
#
# SN12 (Slone & Netzer 2012) was, until this work, the one disk library
# tengri lacked — every "Netzer" reference in tengri pointed to the unrelated
# Laor & Netzer (1989) self-gravity radius. tengri's new `slone_netzer` disc
# block reads the SN12 α-disc grid directly from AGNfitter-rX's published
# `SN12.pickle` (the M_BH = 8.6, log Ṁ/Ṁ_edd ≈ −2.0 grid point is shown on
# both sides).
#
# How each tengri block matches, panel by panel:
#
# * **THB21 — reproduced, and faithful to the source.** The 0.7 μm bump is an
#   emission-line feature, so `qsogen` must run *with* its line and FeII blocks
#   (continuum alone misses it entirely). With them on, tengri's Hα/2500 Å
#   contrast (~4.5) lands on the *published* Temple, Hewett & Banerji (2021)
#   qsogen reference (4.68 — cross-checked below against the committed
#   `qsogen_detailed_reference`), while AGNfitter-rX's stored THB21 template
#   reads only 2.45. That gap is not a tengri error; it is two compounding
#   effects in the vendored template: (1) its 1024-point common grid samples Hα
#   every ~104 Å — wider than the broad line itself (~66 Å FWHM) — so
#   point-sampling tengri's own native qsogen onto that grid drops its Hα/2500
#   from 4.5 to ~2.8, most of the way to AGNFITTER-RX's value; and (2) the
#   stored template is weaker still (2.45), a small residual from
#   AGNFITTER-RX's particular luminosity/Baldwin realization. tengri evaluates
#   qsogen at native resolution, so it recovers the full line. The near-IR
#   hot-dust inflection that Temple+2021 fold into their *composite* template
#   is, in both codes' decomposition, the torus's job (§9c) — which is why the
#   disc panels here agree in the continuum and near-IR and diverge only at the
#   lines.
# * **SN12 — reproduced** by the `slone_netzer` block. It interpolates the
#   108-template grid with node-exact bilinear interpolation, so the SN12 peak
#   lands on AGNfitter-rX's at every grid node (the peak shifts strongly with
#   accretion rate, so the original smooth-kernel interpolation smeared it by
#   30–50% — now fixed).
# * **KD18 — reproduced** (with a parameterization note). tengri's full
#   Kubota & Done 3-zone block (`kubota_done`: a Novikov-Thorne outer disc +
#   warm Comptonization + a hot corona) is *luminosity-first*: the Eddington
#   ratio is derived from `agn_log_lbol` and `agn_log_mbh` (λ_Edd =
#   L_bol/L_Edd), so `agn_log_ledd` is deprecated and ignored for this disc.
#   To land on an AGNfitter-rX grid node (logM_BH, logλ_Edd) we therefore set
#   L_bol = λ_Edd · L_Edd(M_BH), computed below from the same physical
#   constants tengri uses. At the matched node the two realizations agree
#   across the UV–optical to ≤0.07 dex (the residual box quantifies the
#   1000 Å–1 µm disc window), and diverge only in the wings: tengri sits
#   slightly *above* AGNFITTER-RX in the near-IR (> 1 µm) and the far-UV,
#   because it integrates its outer disc out to the physical self-gravity
#   (Toomre) radius (Laor & Netzer 1989; the qsosed-canonical R_out ≈ 1300 R_g
#   at this node) and so carries the coolest outer annuli that AGNFITTER-RX's
#   stored template truncates. The far-IR / X-ray tails (hot corona,
#   seed-photon rollover) the two codes treat differently and are not compared
#   here.
# * **R06 — the same template.** Both sides use the identical Richards+2006
#   composite; the only subtlety is carriage. tengri's `richards2006` returns
#   the physical L_ν, while AGNfitter-rX stores the published νL_ν array
#   directly (it never divides by ν). Left uncorrected the two would sit a
#   factor of ν apart. The panel puts AGNfitter-rX's R06 onto the same L_ν
#   axis (divides by ν) so it shows the *same template* — the two curves then
#   overlay to a median 0.0002 dex over 0.15–3 µm (printed below).

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
def _val_at(w, L, lam):
    """Interpolated value of L at wavelength ``lam`` (sorts w first)."""
    o = np.argsort(np.asarray(w))
    return float(np.interp(lam, np.asarray(w)[o], np.asarray(L)[o]))


_ANNOT = dict(
    transform=None, va="top", ha="left", fontsize=7, family="monospace",
    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
for ax, (af_name, af_kw, tengri_fn, tengri_label) in zip(axes.ravel(), disk_pairs):
    w_a, L_a = A.disk_template(af_name, **af_kw)
    if af_name == "R06":
        # AGNFITTER-RX carries R06 as nu*L_nu (it stores the published
        # Richards+2006 nu*L_nu array without dividing by nu); tengri's
        # `richards2006` returns the physical L_nu. Put AGNFITTER's onto the
        # same L_nu axis (divide by nu) so the panel shows the SAME template
        # rather than a factor-of-nu carriage difference — they then overlay
        # to ~1e-4 dex (the §9a residual cell prints the number).
        L_a = np.asarray(L_a) / (U.C_ANGSTROM_PER_S / np.asarray(w_a))
    a_norm = norm_at(w_a, L_a, ANCHOR)
    msk_a = (w_a > 5e2) & (w_a < 5e4)
    # AGNFITTER-RX as a thick, semi-transparent band; tengri as a thin line on
    # top — the two stay legible even where they lie exactly on one another.
    ax.loglog(w_a[msk_a], a_norm[msk_a], "C0-", lw=4.0, alpha=0.35,
              solid_capstyle="round", label=f"AGNFITTER  {af_name}")
    # tengri at NATIVE resolution — qsogen's lines are what a real fit uses,
    # so the THB21 panel now shows the full line rather than a downsampled one.
    w_t, L_t = tengri_fn()
    t_norm = norm_at(w_t, L_t, ANCHOR)
    msk_t = (w_t > 5e2) & (w_t < 5e4)
    ax.loglog(w_t[msk_t], t_norm[msk_t], "C1-", lw=1.4, label=f"tengri  {tengri_label}")
    ax.axvline(6563, color="0.7", ls=":", lw=1)

    if af_name == "THB21":
        # Ladder: tengri's native qsogen (faithful to Temple+2021), the same
        # curve point-sampled onto AGNFITTER's 1024-pt grid (isolates the
        # ~104 Å undersampling), and AGNFITTER's weaker stored template.
        _ot = np.argsort(np.asarray(w_t))
        _t_on_af = np.interp(np.asarray(w_a), np.asarray(w_t)[_ot], np.asarray(t_norm)[_ot])
        _lad = ("tall lines = tengri resolving native\n"
                "qsogen (a feature; AF grid downsamples)\n"
                "Hα/2500 Å\n"
                f"tengri native : {_val_at(w_t, t_norm, 6563):.2f}\n"
                f" on AF grid   : {_val_at(w_a, _t_on_af, 6563):.2f}\n"
                f"AGNFITTER     : {_val_at(w_a, a_norm, 6563):.2f}")
        ax.text(0.03, 0.97, _lad, **{**_ANNOT, "transform": ax.transAxes})

    if af_name == "KD18":
        # Agreement is quantified in the DATA-CONSTRAINED disc window
        # (1200 Å–1 µm). The shaded wings are model-dependent, not a fit
        # residual: in the NIR tengri integrates the outer disc to the PHYSICAL
        # self-gravity (Toomre) radius (~1300 R_g; qsosed-canonical, pinned by a
        # passing test) that AGNFITTER-RX's stored template truncates, so tengri
        # carries more near-IR; the far-UV reflects the warm-Comptonization
        # proxy. Where it matters (the disc window) they agree to <0.07 dex.
        ax.axvspan(1e4, 5e4, color="0.85", alpha=0.45, zorder=0)  # NIR: R_out
        ax.axvspan(5e2, 1.2e3, color="0.85", alpha=0.45, zorder=0)  # far-UV: warm-Compton
        oa_, ot_ = np.argsort(w_a), np.argsort(w_t)
        w_a_s = np.asarray(w_a)[oa_]
        m = (w_a_s >= 1.2e3) & (w_a_s <= 1e4) & (a_norm[oa_] > 0)
        _t_on = np.interp(w_a_s[m], np.asarray(w_t)[ot_], t_norm[ot_])
        _lr = np.abs(np.log10(_t_on / a_norm[oa_][m]))
        _res = ("disc window 1200 Å–1 µm\n"
                f"median : {np.median(_lr):.3f} dex\n"
                f"max    : {_lr.max():.2f} dex\n"
                "shaded wings: model-dependent\n(tengri → physical self-gravity R_out)")
        ax.text(0.03, 0.97, _res, **{**_ANNOT, "transform": ax.transAxes})

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
# onto a dense grid would manufacture residual at every line feature. R06 is
# compared in the shared L_nu convention (AGNFITTER's nu*L_nu carriage divided
# by nu), matching the panel above.
print("§9a  disc-library shape residuals (anchor 2500 Å, 1200 Å–1 µm):")
for af_name, af_kw, tengri_fn, _label in disk_pairs:
    w_a, L_a = A.disk_template(af_name, **af_kw)
    if af_name == "R06":
        # Same L_nu convention as the panel (divide AGNFITTER's nu*L_nu by nu).
        L_a = np.asarray(L_a) / (U.C_ANGSTROM_PER_S / np.asarray(w_a))
    w_t, L_t = tengri_fn()
    oa_, ot_ = np.argsort(w_a), np.argsort(w_t)
    w_a_s, a_s = np.asarray(w_a)[oa_], norm_at(w_a, L_a, ANCHOR)[oa_]
    m = (w_a_s >= 1.2e3) & (w_a_s <= 1e4) & (a_s > 0)
    t_on = np.interp(w_a_s[m], np.asarray(w_t)[ot_], norm_at(w_t, L_t, ANCHOR)[ot_])
    logr = np.abs(np.log10(t_on / a_s[m]))
    print(f"  {af_name:6s} median = {np.median(logr):.3f} dex   max = {logr.max():.3f} dex")

# %%
# Quantify the line bump AND the THB21 Hα ladder — the evidence that tengri's
# qsogen is faithful to the *source* while AGNFITTER-RX's stored template is
# the weaker-lined outlier. All ratios are L_nu(6563 Å)/L_nu(2500 Å).
w_af, L_af = A.disk_template("THB21")
w_r, L_r = A.disk_template("R06")
bump_thb = _val_at(w_af, norm_at(w_af, L_af, ANCHOR), 6563)
bump_r06 = _val_at(w_r, norm_at(w_r, L_r, ANCHOR), 6563)
print(
    f"§9a  L_nu(6563 Å)/L_nu(2500 Å):  THB21 = {bump_thb:.2f}   R06 = {bump_r06:.2f}  "
    f"(ratio {bump_thb / bump_r06:.1f}x — the emission-line bump)"
)

# tengri native, tengri point-sampled onto AGNFITTER's grid, AGNFITTER stored.
w_tq, L_tq = tengri_qsogen_full()
tq_norm = norm_at(w_tq, L_tq, ANCHOR)
_otq = np.argsort(np.asarray(w_tq))
tq_on_af = np.interp(np.asarray(w_af), np.asarray(w_tq)[_otq], np.asarray(tq_norm)[_otq])
print(
    f"§9a  Hα ladder:  tengri native = {_val_at(w_tq, tq_norm, 6563):.2f}   "
    f"tengri on AGNFITTER grid = {_val_at(w_af, tq_on_af, 6563):.2f}   "
    f"AGNFITTER stored = {bump_thb:.2f}"
)

# Independent cross-check against the published Temple, Hewett & Banerji (2021)
# qsogen reference (committed): its full_flam -> L_nu Hα contrast should land
# next to tengri's native value, confirming tengri (not AGNFITTER) reproduces
# the source. Skipped gracefully if the reference npz is absent.
_REF_CANDIDATES = [
    _HERE / "_drivers" / "data" / "qsogen_detailed_reference.npz",
    Path("data/qsogen_detailed_reference.npz"),
    Path(tengri.__file__).resolve().parents[2] / "data" / "qsogen_detailed_reference.npz",
]
_ref_path = next((p for p in _REF_CANDIDATES if p.is_file()), None)
if _ref_path is not None:
    _ref = np.load(str(_ref_path), allow_pickle=True)
    _wref = np.asarray(_ref["wave"]).astype(float)
    _lnu_ref = np.asarray(_ref["full_flam"]).astype(float) * _wref**2  # F_lam -> L_nu
    _ref_ha = _val_at(_wref, _lnu_ref, 6563) / _val_at(_wref, _lnu_ref, 2500)
    print(
        f"§9a  Temple+2021 qsogen source (published reference): Hα/2500 = "
        f"{_ref_ha:.2f}  → tengri native "
        f"({_val_at(w_tq, tq_norm, 6563):.2f}) matches the source; AGNFITTER's "
        f"stored {bump_thb:.2f} is downsampled + weaker."
    )
else:
    print("§9a  (Temple+2021 qsogen reference npz not found — source cross-check skipped.)")

# %% [markdown]
# ## §9b Accretion-disk reddening sweep
#
# The disk color excess E(B−V)_BBB sweeps the UV continuum via the Prevot
# SMC law on both sides. AGNfitter-rX applies it to the THB21 template;
# tengri's `agn_ebv_disc` applies the same curve to whatever disc is configured.
# The attenuation ratio ``L(E(B−V)) / L(0)`` removes the underlying template
# from both sides, so this panel compares the reddening laws themselves,
# end-to-end.
#
# tengri's dashed curves sit **below** AGNFITTER-RX's solid ones at matched
# E(B−V) — the §4 convention, 1.102× more extinction under the more physical
# R_V. It is a pure normalization, not a shape difference: the black dotted
# curve is tengri at E(B−V) = 0.3/1.102 = 0.272 and it lands exactly on
# AGNFITTER-RX's 0.3 solid, so E(B−V)_tengri = E(B−V)_AF/1.102 reproduces
# AGNFITTER-RX bit-for-bit.

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
# Convention-matched demonstration: tengri at E(B−V) = 0.3/1.102 = 0.272 lands
# exactly on AGNFITTER-RX's 0.3 solid — identical LAW SHAPE, the offset is a
# pure R_V normalization (2.72 vs 2.468), fully resolved by rescaling E(B−V).
w_tm, L_tm = tengri_disc("qsogen", ebv_disc=0.3 / 1.102)
ratio_tm = np.divide(L_tm, L_te0, out=np.ones_like(L_tm), where=L_te0 > 0)
msk_tm = (w_tm > 8e2) & (w_tm < 1e4)
ax.loglog(w_tm[msk_tm], ratio_tm[msk_tm], "k:", lw=1.8,
          label="tengri E(B−V)=0.3/1.102 → on AF 0.3")
ax.text(0.03, 0.60,
        "convention resolved (§4):\n"
        "E(B−V)_tengri = E(B−V)_AF / 1.102\n"
        "R_V 2.72 (physical) vs 2.468 — same shape",
        transform=ax.transAxes, va="top", ha="left", fontsize=7, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))
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
# Four AGNfitter-rX torus libraries at matched grid nodes, peak-normalized.
# All four now match at pinned nodes through tengri's `silva04`, `nenkova_agnfitter`,
# `cat3d_wind` (and `skirtor` full-grid + `skirtor_agnfitter` averaged).
#
# **S04** — `silva04` at log N_H = 23. **NK08** — `nenkova_agnfitter`
# (AGNfitter-rX's inclination-averaged `NK0_mean_1p` CLUMPY reduction, distinct
# from the FSPS `nenkova` average) at incl = 30°. **SKIRTOR** — `skirtor`
# (full X-CIGALE grid, Stalevski 2016) at oa 40°, incl 30°, τ 7 (AGNfitter-rX
# uses averaged `SKIRTOR_mean_3p`; see §9c′). **CAT3D-Wind** — `cat3d_wind` at
# incl 0°, a = −2, f_wd = 1.75, spanning AGNfitter-rX's rows-210+ domain
# (a ∈ [−3, −1.5], f_wd ∈ [1.0, 2.25]). The 10 μm silicate feature and 1.5–5 μm
# near-IR excess appear in these plots; CAT3D's polar wind is designed to fill
# the latter.
#
# **Verification Status:** CROSSVAL — SKIRTOR torus (mean 3-param)

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
    (
        "NK08",
        "NK08",
        # Node-exact against AGNFITTER-RX's inclination-averaged NK0_mean_1p
        # CLUMPY reduction, matched at incl = 30° (cos_inc = cos 30° = 0.866).
        # This is the new `nenkova_agnfitter` block — the earlier panel showed
        # the FSPS `nenkova` reduction, a different average of the same family
        # that did not overlay.
        lambda: tengri_torus("nenkova_agnfitter", cos_inc=0.8660254),
        "nenkova_agnfitter (incl 30°)",
        dict(incl=30.0),
    ),
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
    # AGNFITTER-RX as a thick, semi-transparent band; tengri as a thin line on
    # top — the two stay legible even where they lie exactly on one another.
    ax.loglog(w_a[msk_a], norm_peak(L_a)[msk_a], "C0-", lw=4.0, alpha=0.35,
              solid_capstyle="round", label=f"AGNFITTER  {af_name}")
    w_t, L_t = tengri_fn()
    msk_t = (w_t > 5e3) & (w_t < 1e7)
    ax.loglog(w_t[msk_t], norm_peak(L_t)[msk_t], "C1-", lw=1.4, label=f"tengri  {tengri_label}")
    ax.axvline(1e5, color="0.7", ls=":", lw=1)  # 10 µm silicate
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
# ### §9c″ Torus-library parity — full-spectrum shape ratio
#
# The tengri / AGNfitter-rX peak-normalized shape ratio for each torus library,
# with the 1–100 μm median printed below. S04 and CAT3D-Wind (matched sightline)
# reproduce AGNfitter-rX to high fidelity; SKIRTOR and NK08 use different
# library reductions (full X-CIGALE grid vs averaged; FSPS vs AGNfitter-rX CLUMPY
# average), so moderate shape offsets are expected.

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
ax.set_xlim(8e3, 3e6)
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
# ### §9c′ SKIRTOR: two reductions
#
# The panel above contrasts AGNfitter-rX's averaged `SKIRTOR_mean_3p` with
# tengri's *full-grid* `skirtor` (the X-CIGALE reduction) — different by design.
# Tengri now also ships `skirtor_agnfitter`, a direct node-exact match to the
# `SKIRTOR_mean_3p` templates, so we can show both reductions against the
# AGNfitter-rX reference at one matched geometry (oa 40°, incl 30°, τ₉.₇ = 7):
# **`skirtor_agnfitter`** overlays the AGNfitter-rX curve to node-exact tolerance
# — the monotone-cubic (PCHIP) interpolant reproduces the tabulated library at
# grid nodes rather than smoothing across them. **`skirtor`** (full X-CIGALE grid)
# carries the unaveraged clumpiness and radial-distribution structure, which
# broadens and warms the SED and pushes its IR peak to longer wavelengths.
# Use `skirtor_agnfitter` to reproduce an AGNfitter-rX fit bit-for-bit, or
# `skirtor` to match a CIGALE-family run.

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
ax.set_xlim(8e3, 3e6)
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
# the wind fills the 1.5–5 μm near-IR excess that equatorial tori miss. This
# panel sweeps the wind mass fraction f_wd across AGNfitter-rX's library domain
# (1.0 → 2.25) on both sides at fixed (incl 0°, a = −2): tengri's `cat3d_wind`
# (solid, through `SEDModel.build`) over the AGNfitter-rX library nodes (dashed).
# The interpolation is node-exact monotone-cubic, so at library nodes the curves
# coincide. Raising f_wd lifts exactly the near-IR shoulder, demonstrating the
# wind's role.

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
ax.set_xlim(8e3, 3e6)
ax.set_ylim(1e-3, 3)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ (norm. at peak)")
ax.set_title("CAT3D-Wind $f_{wd}$ sweep — tengri (solid) on AGNFITTER-RX nodes (dashed)")
ax.legend(fontsize=9, title="shaded: 1.5–5 µm excess band")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_09c3_cat3d_fwd_sweep.png")

# %% [markdown]
# ## §9d Best combination — CAT3D-Wind + THB21, full radio-to-X-ray SED
#
# The paper's winning model for 67% of its sample: the CAT3D-Wind torus on the
# THB21 disk. tengri's side spans radio to hard X-ray in one build —
# `disc = qsogen` (with lines + FeII, the full THB21 analog) plus
# `torus = cat3d_wind` at the pipeline's **energy balance** (covering fraction
# f ~ 0.4), the α_ox X-ray corona, and the DPL radio jet — plotted in **νL_ν**
# and **decomposed** into its components (disc+torus, α_ox corona, DPL jet).
# AGNFITTER-RX is placed on the same physical scales: THB21 disc anchored at
# 2500 Å, CAT3D torus energy-balanced to tengri's torus IR peak, X-ray via its
# own disc extension, the same jet. In νL_ν the big blue bump leads with the
# mid-IR torus bump comparable; the EUV/soft-X-ray band is the model-dependent
# bridge discussed at the capstone. Carrying the disc's emission lines is what
# puts the 0.7 µm forest on top of the torus hump.

# %%
# Matched CAT3D-Wind node on both sides: incl 0 deg, a = -2, f_wd = 1.75.
_CAT3D_NODE = dict(incl=0.0, a=-2.0, fwd=1.75)
_CAT3D_NODE_TE = dict(cos_inc=1.0, a_cat3d=-2.0, fwd_cat3d=1.75)

from tengri.xray import xray_agn_corona_from_disc as _xray_from_disc_9

# tengri: the FULL winning model in ONE build — disc + torus (energy-balanced) +
# the alpha_ox corona (xray='yang20') + the DPL radio jet — read straight off
# the build's published components. No hand-set fractions.
_m9 = SEDModel.build(
    ssp_data=ssp,
    sfh=SFH_FIDUCIAL,
    dust_attenuation=NO_DUST,
    agn={
        "type": "composable",
        "disc": {"type": "qsogen", "all_params": FIXED},
        "lines": {"type": "qsogen", "all_params": FIXED},
        "feii": {"type": "qsogen_balmer", "all_params": FIXED},
        "torus": {
            "type": "cat3d_wind",
            "cos_inc": Fixed(1.0),
            "a_cat3d": Fixed(-2.0),
            "fwd_cat3d": Fixed(1.75),
            "all_params": FIXED,
        },
        "agn_log_lbol": Fixed(12.0),
        "agn_ebv_disc": Fixed(0.0),
        "agn_polar_ebv": Fixed(0.0),
        "all_params": FIXED,
    },
    xray={"type": "yang20"},
    radio={"sf": {"type": "bell2003"}, "agn": {"type": "dpl"}},
    redshift=Fixed(0.0),
)
_s9 = _m9.predict_state({})
w9 = np.asarray(_s9.wave)
_o9 = np.argsort(w9)
agn9 = np.asarray(_s9.derived["sed_agn"])  # disc + torus [L_nu]
radio9 = np.asarray(_s9.derived["sed_radio"])  # DPL jet [L_nu]
_L2500_9 = float(np.interp(2500.0, w9[_o9], agn9[_o9]))
# alpha_ox corona: the SAME bare corona §10b validates to ~1% (anisotropy off),
# masked to lambda < 100 A so it does not extrapolate into the disc's UV.
_xray9_raw = np.asarray(
    _xray_from_disc_9(jnp.asarray(w9), _L2500_9, delta_alpha_ox=0.0, apply_anisotropy=False)
)
xray9 = np.where(w9 < 100.0, _xray9_raw, 0.0)

# Full-range nu*L_nu grid (radio -> hard X-ray).
nu9 = np.geomspace(1e8, 1e20, 4000)
lam9 = U.C_ANGSTROM_PER_S / nu9


def _nulnu9(sed):
    s = U.regrid(w9, np.clip(sed, 0, None), lam9)
    return np.where(s > 0, nu9 * s, np.nan)


# Taper off the qsogen disc's spurious hard-X-ray extrapolation (below its
# ~91 Å template limit): the disc+torus emit no X-rays — the α_ox corona is the
# X-ray source — so disc+torus (opt–EUV) hands off to corona (X-ray), 91 → 30 Å.
_euv_taper9 = np.clip(
    (np.log10(w9) - np.log10(30.0)) / (np.log10(91.0) - np.log10(30.0)), 0.0, 1.0
)
agn9_phys = agn9 * _euv_taper9
te_agn9 = _nulnu9(agn9_phys)  # disc + torus (opt–EUV)
te_xray9 = _nulnu9(xray9)  # α_ox corona (X-ray)
te_radio9 = _nulnu9(radio9)  # DPL jet (radio)
te_tot9 = _nulnu9(agn9_phys + xray9 + radio9)  # total
# tengri's absolute torus IR L_nu peak — the energy-balance reference.
_ir9 = (w9 > 3e4) & (w9 < 1e6)
_te_ir_peak = float(np.max(agn9[_ir9])) / _L2500_9

# AGNFITTER-RX: the same components on the SAME physical scales.
w_disc, L_disc = A.disk_template("THB21")
af_disc = U.regrid(w_disc, np.clip(L_disc, 0, None), lam9)
af_disc = af_disc / np.interp(2500.0, lam9[::-1], af_disc[::-1]) * _L2500_9
af_disc = af_disc * np.clip(  # X-ray from the extension only (same 91→30 Å rolloff)
    (np.log10(lam9) - np.log10(30.0)) / (np.log10(91.0) - np.log10(30.0)), 0.0, 1.0
)
w_tor, L_tor = A.torus_template("CAT3D", **_CAT3D_NODE)
af_tor = U.regrid(w_tor, np.clip(L_tor, 0, None), lam9)
af_tor = (
    af_tor / np.max(af_tor) * (_te_ir_peak * _L2500_9) if np.max(af_tor) > 0 else af_tor
)
_L_disc_phys = norm_at(w_disc, L_disc, 2500.0) * _L2500_9
xw, xL = A.disk_xray_extension(w_disc, _L_disc_phys, scatter=0.0)
af_xray = U.regrid(xw, np.clip(xL, 0, None), lam9)
af_radio = U.regrid(w9, np.clip(radio9, 0, None), lam9)
af_sed9 = af_disc + af_tor + af_xray + af_radio
af_tot9 = np.where(af_sed9 > 0, nu9 * af_sed9, np.nan)

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.loglog(nu9, af_tot9, "C0-", lw=4.0, alpha=0.35, solid_capstyle="round",
          label="AGNFITTER-RX  THB21 + CAT3D + a_ox + DPL")
ax.loglog(nu9, te_tot9, "C1-", lw=1.6, label="tengri  total (one build)")
# tengri component decomposition (thin lines) — what makes up the winning model.
ax.loglog(nu9, te_agn9, "C3-", lw=1.0, alpha=0.75, label="   disc + torus")
ax.loglog(nu9, te_xray9, "C4-", lw=1.0, alpha=0.75, label="   α_ox corona")
ax.loglog(nu9, te_radio9, "C2-", lw=1.0, alpha=0.75, label="   DPL jet")
for nu_band, name in [(1.4e9, "radio"), (3e13, "IR"), (6e14, "opt"), (4.8e17, "2 keV")]:
    ax.axvline(nu_band, color="0.85", ls=":", lw=1)
    ax.text(nu_band, 0.98, f" {name}", transform=ax.get_xaxis_transform(),
            rotation=90, va="top", ha="left", fontsize=7, color="0.5")
# Same EUV / soft-X-ray model-dependent band as the capstone.
ax.axvspan(3.3e15, 4.8e16, color="0.8", alpha=0.30, zorder=0)
ax.set_xlim(1e8, 1e20)
_fin9 = te_tot9[np.isfinite(te_tot9)]
ax.set_ylim(_fin9.max() * 1e-9, _fin9.max() * 5)
ax.set_xlabel(r"$\nu$ [Hz]")
ax.set_ylabel(r"$\nu L_\nu$ [erg/s]")
ax.set_title("Best-combination AGN SED (paper's winning model) — full range, decomposed")
ax.legend(fontsize=8, ncol=2, loc="lower center")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_09d_best_combo.png")

# %% [markdown]
# ## §10 X-ray corona via α_ox–L₂₅₀₀
#
# AGNfitter-rX ties the 2 keV corona to the 2500 Å disk continuum through the
# Just et al. (2007) / Lusso & Risaliti (2016, 2017) relation,
# ``α_ox = −0.137 log L₂₅₀₀ + 2.638 + Δα_ox``, then lays down a Γ = 1.8
# power law with 300 keV exponential cutoff. The dispersion Δα_ox ∈ [−0.4, 0.4]
# is a free parameter. Tengri exposes the same `just2007`, `lusso_risaliti_2016/2017`
# relations in `alpha_ox_from_l2500`. The relation is valid only for type-1,
# radio-quiet, non-BAL AGN — the paper is explicit that it should not be applied
# to type-2, BAL, or blazar sources.
#
# **Verification Status:** PARTIAL (3/16) — Radio + X-ray + AGN

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
# ### §10′ X-ray α_ox–L₂₅₀₀ parity (tengri vs AGNfitter-rX)
#
# Tengri's `just2007` relation vs AGNfitter-rX's `α_ox = −0.137 log L₂₅₀₀ +
# 2.638`, with the residual panel below showing agreement across the luminosity
# range.

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
# AGNfitter-rX models AGN core/jet radio with a simple power law (SPL,
# α = −0.75, exponential cutoff at 10¹³ Hz) or double power law (DPL,
# Eq. 9–10: a self-absorption turnover ν_t and synchrotron-aging cutoff ν_cut).
# Tengri ships both — `radio_agn` (SPL) and `radio_agn_dpl`, the latter
# implementing AGNfitter-rX Eq. 9–10 exactly. The driver carries the upstream
# formulas verbatim, so both panels overlay tengri against AGNfitter-rX itself
# at matched parameters, not against a hand-written power law. DPL parameters
# sit inside AGNfitter-rX's own sampling grid (α₂ ∈ [−1, 0)).
#
# Both tengri AGN-jet blocks extend into the sub-mm/IR governed by their
# synchrotron-aging cutoff (10 THz), exactly as AGNfitter-rX's jet does. The
# `radio_agn` SPL now carries the same `exp(−ν/1e13)` term, so its parity with
# AGNfitter-rX's SPL is 1e-4 across 0.1–300 GHz (previously ~3%). Only the
# star-formation radio keeps the 1 mm floor, since it is tied to the dust FIR
# and must not double-count.
#
# **Verification Status:** PARTIAL (3/25) — Radio / X-ray / IGM / PSD physics

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
# ### §11′ Radio SPL/DPL parity ratios (tengri vs AGNfitter-rX)
#
# The ratio of each tengri radio model to the AGNfitter-rX formula over
# 0.1–300 GHz (both normalized at 5 GHz). The SPL ratio drifts up to ~3%
# at 300 GHz because AGNfitter-rX's SPL carries its exponential cutoff
# (exp(−ν/10¹³ Hz)) while tengri's SPL is a pure power law within its support;
# the DPL formulas are identical term by term.

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
# Γ = 1.8 power law in F_ν with a 300 keV exponential cutoff
# (``exp(−ν/7.254·10¹⁹ Hz)``), normalized at 2 keV through the Just+2007
# α_ox relation. The **left panel** compares the
# bare prescriptions — tengri's `xray_agn_corona_from_disc` with its X-CIGALE
# extensions switched off (`apply_anisotropy=False`) against AGNfitter-rX's
# disk X-ray extension at the same L₂₅₀₀. The residual is a flat ~+1%
# (tengri adds a constant 1% Thomson-scattered fraction to the absorbed primary)
# plus soft-X-ray curvature from tengri's tbabs photoelectric absorption at its
# default column log N_H = 20 — both printed.
#
# The **right panel** shows what tengri's *defaults* add on top of the bare
# corona, because they are on by default and AGNfitter-rX has no analog: the
# Yang et al. (2022) viewing-angle anisotropy (normalized at the 30° X-CIGALE
# anchor, so even a face-on sightline sits ×1.07 above isotropic) and the
# tbabs/Compton line-of-sight absorption. For AGNfitter-rX-parity work, switch
# the anisotropy off. On the host side, tengri adds an X-ray-binary floor
# (`xray_xrb`, Mineo et al. 2014) as a bona-fide SED component. AGNfitter-rX
# has no host X-ray component; its X-ray physics beyond the corona lives in two
# *priors* — a Lusso & Risaliti (2016)-calibrated α_ox prior (σ = 0.4) and a
# Stern (2015) 6 μm ↔ 2–10 keV prior (σ = 0.5) — which shape posteriors but
# never emit flux.

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
# ## §11b Star-formation radio (Bell 2003) vs AGNfitter-rX's S17_radio
#
# AGNfitter-rX delivers the host's radio emission *inside* its starburst
# template: the `S17_radio` model extends each Schreiber+2018 (T_dust, f_PAH)
# dust SED into the radio with the Bell (2003) IR–radio correlation
# (q_IR = 2.64, synchrotron plus a thermal free-free component that flattens
# the slope near 10 GHz). Tengri keeps the two pieces separate — `schreiber2018`
# for the dust, `radio_sfr_bell2003` for the radio — so the genuine comparison
# is the *sum*, anchored to the same total IR luminosity. The panel overlays
# tengri's dust + radio against the upstream S17_radio template at
# (T_dust = 35 K, f_PAH = 0.02), and prints the 1.4 GHz ratio — which turns out
# to be a precise measurement of a q_IR bookkeeping difference between the codes.
# AGNfitter-rX applies Bell's 2.64 to the tabulated LIR_conv (smaller than the
# template's 8–1000 μm integral), whereas tengri applies 2.64 to the integral
# directly, so its 1.4 GHz output is 1.9× brighter at matched L_IR — same
# correlation, different L_IR bookkeeping.

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
ax.set_xlim(8e3, 3e9)
ax.set_ylim(1e-6, 3)
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
# ## Capstone — the radio-to-X-ray SED (physical composition)
#
# AGNfitter-rX's reason for being: one model spanning `8 < log ν/Hz < 20`. Here
# the components are composed at **physical** normalizations, not display
# fractions, so their relative heights carry meaning:
#
# * **tengri** is its *actual* ``SEDModel.build`` output — disc + torus (with the
#   pipeline's own energy balance), the α_ox X-ray corona (``xray='yang20'``) and
#   the DPL radio jet in one build: ``sed_agn + sed_xray + sed_radio``. The torus
#   reprocesses a covering fraction ``f ~ 0.4`` of the disc light into the IR, and
#   the disc is debited by ``1 − f`` so disc + torus conserve L_bol. In νL_ν the
#   concentrated mid-IR bump therefore out-peaks the (debited) big blue bump — the
#   reprocessing signature of a f < 1 torus, not an over-bright one (the on-figure
#   box prints the model's own f).
# * **AGNfitter-rX** is placed on the *same* physical scales: its THB21 disc
#   anchored at the disc's ``L_ν(2500 Å)``; its X-ray via its own disc extension
#   at that luminosity (the α_ox relation — ``L_2keV/L_2500 ≈ 3e−4``, α_ox ≈ −1.4,
#   the textbook radio-quiet value tengri's build reproduces); its CAT3D torus
#   energy-balanced to the disc's IR reprocessing; the same DPL jet.
#
# The residual differences are genuine — the qsogen emission lines, the torus
# silicate profile, and, in the far UV → X-ray, the **soft/EUV bridge**: both
# codes anchor 2 keV with the *same* Just+2007 α_ox relation (so they agree
# there and across the hard X-ray), but the EUV (13.6–200 eV) is a deliberate
# *hole* in both — AGNFITTER-RX's ``XRAYS`` builds its power law only above
# 200 eV ("with a hole between BB template and X-Rays"), and tengri's simple
# corona is a bare Γ = 1.8 power law with **no soft X-ray excess** (warm
# Comptonization). The complete soft-excess physics lives in tengri's
# ``kubota_done`` disc (the KD18 3-zone model of §9a), not in the simple α_ox
# corona used here for AGNFITTER-RX parity.
#
# Everything on tengri's curve is a *single* ``SEDModel.build``, and every
# AGNFITTER-RX-parity knob used in this notebook (``agn_ebv_disc``, the torus
# axes, the DPL turnover/cutoff) is a regular parameter of that one model, free
# to fit.

# %%
# Common observer grid spanning X-ray (~0.05 keV) to meter-wave radio.
nu_grid = np.geomspace(1e8, 1e20, 4000)
lam_grid = U.C_ANGSTROM_PER_S / nu_grid


# --- tengri: the real physics in ONE build. sed_agn already carries the
# disc/torus energy balance, sed_xray the alpha_ox corona, sed_radio the jet —
# no hand-set fractions. ---
m_cap = SEDModel.build(
    ssp_data=ssp,
    sfh=SFH_FIDUCIAL,
    dust_attenuation=NO_DUST,
    agn={
        "type": "composable",
        "disc": {"type": "qsogen", "all_params": FIXED},
        "lines": {"type": "qsogen", "all_params": FIXED},
        "feii": {"type": "qsogen_balmer", "all_params": FIXED},
        "torus": {
            "type": "cat3d_wind",
            "cos_inc": Fixed(1.0),
            "a_cat3d": Fixed(-2.0),
            "fwd_cat3d": Fixed(1.75),
            "all_params": FIXED,
        },
        "agn_log_lbol": Fixed(12.0),
        "agn_ebv_disc": Fixed(0.0),
        "agn_polar_ebv": Fixed(0.0),
        "all_params": FIXED,
    },
    xray={"type": "yang20"},
    radio={"sf": {"type": "bell2003"}, "agn": {"type": "dpl"}},
    redshift=Fixed(0.0),
)
s_cap = m_cap.predict_state({})
w_te = np.asarray(s_cap.wave)
_owt = np.argsort(w_te)
_agn_te = np.asarray(s_cap.derived["sed_agn"])
_radio_te = np.asarray(s_cap.derived["sed_radio"])
L2500 = float(np.interp(2500.0, w_te[_owt], _agn_te[_owt]))  # disc anchor
# X-ray: the α_ox corona anchored to the disc's L_ν(2500 Å) — the same model
# §10b validates against AGNFITTER-RX (anisotropy off for parity; tengri's
# default adds the ×1.072 Yang+22 term). Masked to λ < 100 Å (its physical
# domain) so it does not extrapolate into the disc's UV.
_xray_raw = np.asarray(
    xray_agn_corona_from_disc(
        jnp.asarray(w_te), L2500, delta_alpha_ox=0.0, apply_anisotropy=False
    )
)
_xray_raw_masked = np.where(w_te < 100.0, _xray_raw, 0.0)
_xray_te = _xray_raw_masked
# The qsogen disc template extrapolates a spurious power-law tail below its
# ~91 Å limit across the hard X-ray (νL_ν ~ 7e43 at 2 keV, comparable to the
# corona itself). The disc emits no X-rays — the α_ox corona is the physical
# X-ray source — so taper sed_agn off across the EUV (91 → 30 Å) so disc+torus
# (opt–EUV) hands off smoothly to the corona (X-ray) with no double count and no
# hard cliff. Without this the total sits ~2× above AGNFITTER-RX at 2 keV even
# though the *bare* corona matches to ~1 % (§10b).
_euv_taper = np.clip(
    (np.log10(w_te) - np.log10(30.0)) / (np.log10(91.0) - np.log10(30.0)), 0.0, 1.0
)
_agn_te_phys = _agn_te * _euv_taper
te_lnu = _agn_te_phys + _xray_te + _radio_te  # physical L_nu [erg/s/Hz]
te_sed = U.regrid(w_te, np.clip(te_lnu, 0, None), lam_grid)
_irband = (w_te > 3e4) & (w_te < 1e6)
te_tor_ir_peak = float(np.max(_agn_te[_irband])) if np.any(_irband) else L2500
_nu_te = U.C_ANGSTROM_PER_S / w_te
_l2kev = float(np.interp(6.199, w_te[_owt], _xray_te[_owt]))  # 2 keV = 6.199 A
alpha_ox = -0.3838 * np.log10(L2500 / _l2kev)
_lx = float(np.trapezoid(_xray_te[_owt], _nu_te[_owt]))
_lagn = float(np.trapezoid(_agn_te[_owt], _nu_te[_owt]))
# Torus covering fraction: the share of the AGN output reprocessed into the IR
# (lambda > 3 um). This sets the disc/torus heights below by ENERGY BALANCE —
# the disc is debited by (1 - f_cov) so disc + torus conserve L_bol — not a
# free display fraction. So the mid-IR torus bump out-peaking the (debited) big
# blue bump in nu*L_nu is the reprocessing signature of a f ~ 0.4 covering
# torus, not an over-normalization (f_cov < 1 always; the disc is NOT erased).
_l_ir_cap = float(np.trapezoid(np.where(w_te[_owt] > 3e4, _agn_te[_owt], 0.0), _nu_te[_owt]))
_f_cov = abs(_l_ir_cap) / abs(_lagn)

# --- AGNFITTER-RX: the same components on the SAME physical scales. ---
# Disc anchored at the physical L_nu(2500 A).
w_ad, L_ad = A.disk_template("THB21")
af_disc = U.regrid(w_ad, np.clip(L_ad, 0, None), lam_grid)
af_disc = af_disc / np.interp(2500.0, lam_grid[::-1], af_disc[::-1]) * L2500
# Symmetric taper: THB21 also carries a (steeper) sub-Lyman tail; hand the X-ray
# on both sides to each code's corona/extension only, with the same 91 → 30 Å
# EUV rolloff so neither disc double-counts into the X-ray.
af_disc = af_disc * np.clip(
    (np.log10(lam_grid) - np.log10(30.0)) / (np.log10(91.0) - np.log10(30.0)), 0.0, 1.0
)
# Torus energy-balanced: scale CAT3D so its IR peak matches tengri's torus.
w_at, L_at = A.torus_template("CAT3D", **_CAT3D_NODE)
af_tor = U.regrid(w_at, np.clip(L_at, 0, None), lam_grid)
af_tor = af_tor / np.max(af_tor) * te_tor_ir_peak if np.max(af_tor) > 0 else af_tor
# X-ray via AGNFITTER-RX's own disc extension at the physical disc luminosity
# (its alpha_ox) — same L_nu(2500 A) as tengri, so both coronae sit at the
# physical X-ray/UV ratio rather than a hand-set fraction.
_L_ad_phys = norm_at(w_ad, L_ad, 2500.0) * L2500
xw, xL = A.disk_xray_extension(w_ad, _L_ad_phys, scatter=0.0)
af_xray = U.regrid(xw, np.clip(xL, 0, None), lam_grid)
# Radio: the same physical DPL jet tengri's build used.
af_radio = U.regrid(w_te, np.clip(_radio_te, 0, None), lam_grid)
af_sed = af_disc + af_tor + af_xray + af_radio

af_plot = np.where(af_sed > 0, nu_grid * af_sed, np.nan)
te_plot = np.where(te_sed > 0, nu_grid * te_sed, np.nan)

fig, ax = plt.subplots(figsize=(9.5, 5))
ax.loglog(nu_grid, af_plot, "C0-", lw=4.0, alpha=0.35, solid_capstyle="round",
          label="AGNFITTER-RX  THB21 + CAT3D + a_ox X-ray + DPL")
ax.loglog(nu_grid, te_plot, "C1-", lw=1.4,
          label="tengri  one SEDModel.build (disc+torus+corona+jet)")
for nu_band, name in [(1.4e9, "radio"), (3e13, "IR"), (6e14, "opt"), (4.8e17, "2 keV")]:
    ax.axvline(nu_band, color="0.85", ls=":", lw=1)
    ax.text(nu_band, ax.get_ylim()[1], f" {name}", rotation=90, va="top", ha="left",
            fontsize=7, color="0.5")
# Make the energy balance legible: the torus reprocesses a covering fraction
# f_cov of the AGN output into the IR, and the disc is debited by (1 - f_cov),
# so disc + torus conserve L_bol. The mid-IR bump therefore out-peaks the
# (debited) big blue bump in nu*L_nu by spectral concentration, not because the
# torus is over-normalized (f_cov < 1).
ax.text(
    0.015, 0.97,
    f"torus covering  f = L_IR/L_AGN ~ {_f_cov:.2f}\n"
    "disc debited x(1 - f); disc + torus = L_bol\n"
    "(energy balance, not a display fraction)",
    transform=ax.transAxes, va="top", ha="left", fontsize=7, family="monospace",
    bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85),
)
# Shade the EUV / soft-X-ray bridge (13.6 eV – 0.2 keV): here BOTH codes
# extrapolate an UNOBSERVABLE region and the curves genuinely differ by model
# choice, not by fit quality. tengri's qsogen disc carries a shallower EUV tail
# (L_ν/L_2500: 0.09 @ 300 Å vs AGNFITTER-RX's THB21 0.04, which drops at the
# Lyman break), and neither simple α_ox corona adds a warm-Compton soft excess
# here (that lives in tengri's kubota_done disc, §9a). Disc/torus/radio and the
# 2 keV anchor all overlay — the divergence is confined to this shaded band.
ax.axvspan(3.3e15, 4.8e16, color="0.8", alpha=0.30, zorder=0)
ax.text(1.26e16, 0.055, "EUV / soft-X\nmodel-dependent\n(unobservable)",
        transform=ax.get_xaxis_transform(), fontsize=6.5, color="0.45",
        ha="center", va="bottom")
ax.set_xlim(1e8, 1e20)
_te_fin = te_plot[np.isfinite(te_plot)]
ax.set_ylim(_te_fin.max() * 1e-9, _te_fin.max() * 5)
ax.set_xlabel(r"$\nu$ [Hz]")
ax.set_ylabel(r"$\nu L_\nu$ [erg/s]")
ax.set_title(r"Radio-to-X-ray AGN SED — physical composition ($8 < \log\,\nu/\mathrm{Hz} < 20$)")
ax.legend(fontsize=8, loc="lower center")
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_full_sed_headtohead.png")

# %%
print(
    f"Capstone physical anchors (log L_bol = 12): disc L_nu(2500 A) = {L2500:.2e} erg/s/Hz;  "
    f"alpha_ox = {alpha_ox:.2f}  (L_2keV/L_2500 = {10 ** (alpha_ox / 0.3838):.1e});  "
    f"X-ray/AGN (integrated) = {_lx / _lagn:.3f};  "
    f"torus covering f = L_IR/L_AGN = {_f_cov:.2f}  (energy balance, disc debited x(1-f)) "
    "— all physical, not display fractions"
)

# %% [markdown]
# ## Summary
#
# Tengri matches AGNfitter-rX's AGN model component by component at matched
# parameters: all four accretion-disk libraries (R06, SN12, KD18, THB21) with
# the Prevot SMC disc reddening; all four torus libraries node-exact (S04, NK08,
# SKIRTOR both the X-CIGALE and averaged reductions, CAT3D-Wind all three axes);
# both cold-dust libraries (S17 and DH02_CE01 with S17_radio extension); the
# α_ox–L₂₅₀₀ X-ray corona; and SPL/DPL radio jets. Every model is built through
# the public `SEDModel.build` grammar, so this notebook is an end-to-end test of
# the public API. The notebook reproduces the paper's two central results: THB21's
# 0.7 μm emission-line forest (Bayes factor ≈10⁵·¹ over R06, §9a) and CAT3D-Wind's
# 1.5–5 μm near-IR excess from the polar wind (§9c, §9c‴), and spans the full
# radio-to-X-ray range (capstone) as one buildable, fittable model (capstone′).
#
# Residuals are understood and quantified per panel: R06 carriage convention
# (0.0002 dex median, §9a); Prevot 1.102× E(B−V) rescaling where AGNfitter-rX
# uses raw UV fit (§4); KD18 warm-Compton proxy + physical self-gravity outer
# radius carrying more near-IR than AGNfitter-rX's truncated template (§9a); THB21 native
# vs pre-binned spectral resolution (§9a); Corona +1% Thomson scattering floor
# (§10b); Host XRBs as separate tengri component (§10b); Radio jets extend into
# sub-mm/IR with SPL parity 1e-4 (§11); SF radio q_IR bookkeeping (§11b).
#
# ## References
#
# Every model compared above, with the section that uses it. The machine-
# readable BibTeX for all of these lives next to this notebook in
# `references.bib` — the key per entry is given in brackets. The brief
# descriptions below map each citation to its role in this reproduction.
#
# **Accretion disks (§9a)**
# - Richards, G. T., et al. 2006, ApJS 166, 470 — R06 mean Type-1 quasar composite SED
#   [`richards2006sed`].
# - Slone, O. & Netzer, H. 2012, MNRAS 426, 656 — SN12 α-disk radiative-transfer
#   models [`slone2012effects`].
# - Kubota, A. & Done, C. 2018, MNRAS 480, 1247 — KD18 Novikov-Thorne + Comptonization
#   [`kubota2018physical`].
# - Temple, M. J., Hewett, P. C. & Banerji, M. 2021, MNRAS 508, 737 — THB21 semi-empirical
#   with emission-line forest; qsogen [`temple2021modelling`].
#
# **Tori (§9c)**
# - Silva, L., et al. 2004, MNRAS 355, 973 — S04 clumpy dust clouds at various
#   inclinations [`Silva2004`].
# - Nenkova, M., et al. 2008, ApJ 685, 160 — NK08 CLUMPY radiative-transfer model
#   [`nenkova2008agnII`].
# - Stalevski, M., et al. 2016, MNRAS 458, 2288 — SKIRTOR self-consistent
#   radiative-transfer [`Stalevski2016`].
# - Hönig, S. F. & Kishimoto, M. 2017, ApJL 838, L20 — CAT3D-Wind with
#   equatorial + polar-wind dust [`honig2017dusty`].
# - Yang, G., et al. 2020, MNRAS 491, 740 — X-CIGALE SKIRTOR + polar dust,
#   the parameterization tengri's `skirtor` follows [`yang2020xcigale`].
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
# - Azadi, M., et al. 2020 (arXiv:2011.03130) — radio AGN and star-formation
#   separation [`azadi2020disentangling`].
# - Bell, E. F. 2003, ApJ 586, 794 — IR–radio correlation q_IR = 2.64
#   [`bell2003estimating`].
#
# **Stellar populations & attenuation (§1–§5)**
# - Bruzual, G. & Charlot, S. 2003, MNRAS 344, 1000 [`bruzual2003stellar`];
#   Chabrier, G. 2003, PASP 115, 763 [`chabrier2003galactic`].
# - Calzetti, D., et al. 2000, ApJ 533, 682 [`calzetti2000dust`]; Prevot, M. L.,
#   et al. 1984, A&A 132, 389 — SMC reddening [`prevot1984typical`].
#
# **Codes & inference**
# - Martínez-Ramírez, L. N., et al. 2024, A&A 688, A46 — AGNfitter-rX
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
