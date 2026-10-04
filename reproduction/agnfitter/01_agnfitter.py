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
# radio-to-X-ray SEDs of active galaxies. CIGALE and Prospector are
# galaxy-centric; AGNfitter-rX characterizes the AGN itself: its accretion
# disk, hot dusty torus, relativistic jet/core radio, and hot corona, alongside
# the host (stellar populations, cold dust, star-formation radio). Radio and
# X-ray data, largely unaffected by dust, are orthogonal tracers that break
# the infrared–ultraviolet degeneracies that limited the original submm-to-UV
# AGNfitter (Calistro Rivera et al. 2016).
#
# This notebook configures tengri's public API, exclusively `SEDModel.build`'s
# dict grammar, `model.predict(params)`, and the documented `tengri.agn`,
# `tengri.xray`, `tengri.radio`, and `tengri.dust` namespaces, to approximate
# AGNfitter-rX's model choices. tengri implements the same physics
# independently; residual differences are quantified in place, next to the figure or
# printed number that shows them.
#
# The headline comparisons are two model face-offs that drive the paper's
# conclusions. **§9a** compares the accretion-disk libraries R06, SN12, KD18
# and THB21 (plus tengri's KD18 grid-tabulated and warm-index variants): the
# paper prefers THB21, the only library that carries the broad and narrow
# emission lines behind the Hα + [N II] peak near 0.7 μm. **§9c** compares the
# torus libraries S04, NK08, SKIRTOR and CAT3D-Wind and five further averaged
# reductions of the same families: the paper finds CAT3D-Wind the
# maximum-likelihood torus for most of its sources and attributes that to
# polar-wind dust in the near-infrared. Those are the paper's fit conclusions,
# listed with their page numbers in the next cell; this notebook tests the
# templates and equations underneath them, not the fits.

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

# nbclient kernels don't bind ``__file__``; fall back to cwd so the Setup
# cell can locate ``_figs/`` and data instead of crashing every panel.
_HERE = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd().resolve()
figs_dir = _HERE / "_figs"
figs_dir.mkdir(exist_ok=True)
_FIG_DPI = 150


def save_fig(filename: str) -> None:
    """Save a figure to ``_figs/`` and leave it open for inline embedding."""
    plt.savefig(str(figs_dir / filename), dpi=_FIG_DPI, bbox_inches="tight")


def _assert_comparable(arr_ref, arr_t, *, name: str) -> None:
    """Guard against shipping a blank or wildly mis-scaled panel."""
    a_ref = np.asarray(arr_ref)
    a_t = np.asarray(arr_t)
    assert np.isfinite(a_ref).any() and np.isfinite(a_t).any(), f"{name}: NaN-only"
    assert (a_ref > 0).any() and (a_t > 0).any(), f"{name}: zero/negative-only"
    ratio = a_ref.max() / a_t.max()
    assert 1e-3 < ratio < 1e3, f"{name}: y-scale ratio {ratio:.2e} out of range"


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
print("AGNFITTER-RX reference libraries (committed under data/):")
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


def band_power(wave_aa, L_nu, lo_aa, hi_aa) -> float:
    """Power ``|int L_nu d(nu)|`` of an SED between two wavelengths [erg/s when L_nu is erg/s/Hz]."""
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    L_nu = np.asarray(L_nu, dtype=np.float64)
    sel = (wave_aa >= lo_aa) & (wave_aa <= hi_aa)
    nu = U.C_ANGSTROM_PER_S / wave_aa[sel]
    order = np.argsort(nu)
    return float(np.trapezoid(L_nu[sel][order], nu[order]))


NODE_EXACT_TOL = 1e-2  # max|tengri/AGNfitter-rX - 1| below which a case is called node-exact


def node_exact_verdict(rows, title) -> None:
    """Print how many cases of a window table meet the node-exact tolerance, and the worst one."""
    devs = {r["label"]: float(r["max_abs_dev"]) for r in rows}
    n_ok = sum(d < NODE_EXACT_TOL for d in devs.values())
    worst = max(devs, key=devs.get)
    print(
        f"{title}: {n_ok}/{len(devs)} cases within max|t/a-1| < {NODE_EXACT_TOL:g}; "
        f"worst = {worst} ({devs[worst]:.3g})"
    )


def resolved_params(m) -> None:
    """Print the model's full resolved parameter table, so no parameter
    (matters or not) rests on a silent default."""
    m.spec.summary()


# %% [markdown]
# ## Common stellar library
#
# Both codes build on Bruzual & Charlot (2003) stellar populations with a
# Chabrier (2003) initial mass function. AGNfitter-rX ships its own edition
# (`models/GALAXY/BC03_840seds.pickle`), tabulated directly over `(tau, age)`
# and repackaged here as `data/agnfitter_galaxy_reference.h5`; we read it
# through the driver's `galaxy_template` and `galaxy_sfr` accessors, never the
# pickle itself. tengri reads its own `bc03_pdva_stelib_chabrier` grid: the same
# BC03 + Chabrier physics in a different (STELIB) spectral-library edition,
# downloaded on demand. The wavelength samplings of both editions are printed
# below. Upstream's fit does not use its full grid: `MODEL_AGNfitter.GALAXY`
# keeps every third wavelength point and snaps bands to the nearest kept
# point, which `galaxy_template(..., subsample3=True)` reproduces.
#
# The two codes' cosmologies also differ: the parameters are printed below. The
# luminosity-distance-squared factor that enters any flux normalization is
# compared at $z=1$, so the panels compare SED shape rather than a cosmology
# artifact. The paper's Table 1 caption quotes a third cosmology for the
# age prior, which the fit does not otherwise use. Both host galaxies are
# stellar-only (`neb={'type': 'none'}`) to match AGNfitter-rX's GALAXY
# component, which carries no nebular-emission term.

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


# %% [markdown]
# ## tengri AGN helpers
#
# Every AGN face-off below builds one composable `agn={...}` block and reads
# the SED from `model.predict(params).sed.components[...]`, the public
# per-sub-block keys (`sed_agn_disc`, `sed_agn_torus`, `sed_agn` for
# disc+lines+torus combined). Every build states `'norm': 'independent'`
# explicitly: disc scales on `agn_log_lbol`, torus on its own (the
# AGNfitter-style bookkeeping; the default `'cigale_joint'` ties disc/torus
# to one CIGALE `agn_power` reference and would rescale the torus amplitude)
# and `atten={'type': 'none'}` explicitly (AGNfitter-rX disc templates carry no
# polar-dust screen; tengri's `polar_dust` atten type is opt-in only, so the
# explicit entry is for clarity). A monolithic `agn={'type': <model>}` (see
# `tengri.list_agn_models()` above) cannot express `atten` or the per-sub-block
# decomposition this notebook depends on throughout.


# %%
def tengri_disc(disc_type, *, log_lbol=11.0, ebv_disc=None, **disc_params):
    """Isolated tengri accretion-disc SED. Returns (wave_aa, L_nu).

    ``ebv_disc`` sets the shared disc obscuration ``agn_ebv_disc`` (the
    AGNFITTER-RX ``EBVbbb`` analog), a disc-block key independent of
    the `atten` sub-block.
    """
    disc = {"type": disc_type, "all_params": Fixed(DEFAULT)}
    disc.update({k: Fixed(v) for k, v in disc_params.items()})
    agn = {
        "type": "composable",
        "disc": disc,
        "torus": {"type": "none"},
        "nlr": {"type": "none"},
        "blr": {"type": "none"},
        "atten": {"type": "none"},
        "agn_log_lbol": Fixed(log_lbol),
        "all_params": Fixed(DEFAULT),
        "norm": "independent",
    }
    if ebv_disc is not None:
        disc["agn_ebv_disc"] = Fixed(ebv_disc)
    m = SEDModel.build(
        ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST, agn=agn, neb={"type": "none"}, redshift=Fixed(0.0)
    )
    tengri_disc.last_model = m
    pred = m.predict({})
    w = np.asarray(pred.sed.components["wavelength"])
    return w, np.asarray(pred.sed.components["sed_agn_disc"])


def tengri_torus(torus_type, *, log_lbol=11.0, **torus_params):
    """Isolated tengri torus SED. Returns (wave_aa, L_nu)."""
    torus = {"type": torus_type, "all_params": Fixed(DEFAULT)}
    torus.update({k: Fixed(v) for k, v in torus_params.items()})
    m = SEDModel.build(
        ssp_data=ssp,
        sfh=SFH_FIDUCIAL,
        dust_attenuation=NO_DUST,
        agn={
            "type": "composable",
            "disc": {"type": "none"},
            "torus": torus,
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "atten": {"type": "none"},
            "agn_log_lbol": Fixed(log_lbol),
            "all_params": Fixed(DEFAULT),
            "norm": "independent",
        },
        neb={"type": "none"}, redshift=Fixed(0.0),
    )
    tengri_torus.last_model = m
    pred = m.predict({})
    w = np.asarray(pred.sed.components["wavelength"])
    return w, np.asarray(pred.sed.components["sed_agn_torus"])


def tengri_qsogen_full(*, log_lbol=11.0, torus=None):
    """tengri's THB21 analog: qsogen continuum *with* its broad/narrow lines
    and FeII pseudo-continuum. qsogen carries both line families in its `blr`
    block (`nlr` stays off); THB21's defining feature is that forest (the
    0.7 µm Hα+[N II] bump), so the disc-only continuum alone does not
    reproduce it."""
    agn = {
        "type": "composable",
        "disc": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
        "torus": torus if torus is not None else {"type": "none"},
        "nlr": {"type": "none"},
        "blr": {"type": "qsogen", "all_params": Fixed(DEFAULT)},
        "feii": {"type": "qsogen_balmer", "all_params": Fixed(DEFAULT)},
        "atten": {"type": "none"},
        "agn_log_lbol": Fixed(log_lbol),
        "all_params": Fixed(DEFAULT),
        "norm": "independent",
    }
    m = SEDModel.build(
        ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST, agn=agn, neb={"type": "none"}, redshift=Fixed(0.0)
    )
    tengri_qsogen_full.last_model = m
    pred = m.predict({})
    w = np.asarray(pred.sed.components["wavelength"])
    return w, np.asarray(pred.sed.components["sed_agn"])


# %% [markdown]
# ## §1 Stellar populations
#
# tengri's library at representative ages (0.1 and 5 Gyr) and the fiducial
# declining-exponential CSP (`SFH_FIDUCIAL`, tau=1 Gyr, age=4.8939 Gyr)
# are compared against AGNfitter-rX's own tabulated BC03 template at the same node,
# read through the driver's `galaxy_template` accessor on its full pickle grid.
# The fit itself keeps every third point (printed in the setup cell); both
# sides here are normalized to 1 at 5500 Å, so this panel tests SED *shape*.
# The absolute normalization, which depends on whether a template holds one
# solar mass formed or present, is tested in §3.
#
# **Verification Status:** CROSSVAL; CSP integral; CIC age kernel (default)

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
    f"§1  matched CSP node (tau={TAU_GYR:g} Gyr, age={AGE_GYR:.4f} Gyr), 0.15-2.5 um, "
    f"5500-A-normalized: median|log10 ratio| = {np.median(np.abs(_logratio[_band_m])):.4f}"
)

# %% [markdown]
# ### §1.1 BC03_metal, AGNfitter-rX's default library
#
# The paper's default galaxy library carries four metallicities, which the matched
# node above does not exercise. The table builds tengri at each, mapping the
# upstream axis (in units of the BC03 solar value, $Z_\odot = 0.02$) onto tengri's
# `met_logzsol` (relative to tengri's own solar value, $Z_\odot = 0.0142$), and
# compares at the library's nearest age node, shape at 5500 Å and absolute scale
# through the present-mass mapping of §3.
#
# **Caveat:** the top upstream node (twice the BC03 solar value) lies between
# tengri's tabulated metallicities (printed below), so tengri interpolates there.

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
# ## §2 Star formation history
#
# AGNfitter-rX's GALAXY component tabulates a declining-exponential
# (`SFR(T) ∝ exp(-T/τ)`), so SFR falls monotonically from formation to the
# present: the classic tau-model, not a delayed-tau history that rises
# then falls. tengri's `declining_exp` SFH type implements the same
# functional form; the notebook writes the closed form directly and checks
# mass closure through the public `pred.sfh.stellar_mass` property. The
# AGNfitter-rX pickle's own tabulated SFR(age) at the fiducial tau is overlaid,
# read through `galaxy_sfr`, renormalized to tengri's mass (shape only). The
# §1 shape residual has three candidate contributions, none of them a mismatch
# of the SFH form: the nearest-node snap in age, the two BC03 editions'
# different spectral resolution, and the shared 5500 Å anchor.
#
# **Verification Status:** PARTIAL; Parametric SFH family physics

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
    f"§2  pred.sfh.stellar_mass = {mass_formed:.4e} M_sun  (target 1.0000e{LOG_MASS:.0f}); "
    f"AGNFITTER-RX reference SFR is monotonically declining: "
    f"{bool(np.all(np.diff(sfr_ref) <= 0))}"
)

# %% [markdown]
# ## §3 Integrated stellar SED
#
# The fiducial host's stellar continuum is the quantity AGNfitter-rX's GA
# (host stellar) component contributes, reddened at fit time by a Calzetti law
# (§5; upstream wires no other galaxy law). The stored GA template is
# normalized to one solar mass of **present** stellar mass (stars plus remnants
# alive at the template age), whereas tengri normalizes to mass **formed**,
# so the two are compared through the mapping tabulated below, which also fixes
# how a prior on the GA amplitude translates to tengri's stellar mass.
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
print("§3  GA ↔ tengri stellar-mass mapping at the matched node")
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
# ## §4 Disk reddening law (Prevot SMC)
#
# AGNfitter-rX reddens the accretion disk (not the host) with an analytic
# Prevot et al. (1984) SMC fit applied as $A_\lambda = k_{\rm raw}(\lambda)\,E(B{-}V)$,
# with $k_{\rm raw}(\lambda) = 1.39\,\lambda_{\mu{\rm m}}^{-1.2} - 0.38$ and no
# reddening blueward of 200 eV (`MODEL_AGNfitter.BBBred_Prevot`). The routine
# declares an $R_V$ (Prevot et al.'s measured SMC value), but its inner
# function ignores that argument and returns the bare $k_{\rm raw}$, so the
# effective total-to-selective ratio is $k_{\rm raw}(0.55\,\mu{\rm m})$, not the
# declared $R_V$.
#
# tengri's disc obscuration (`agn_ebv_disc`, the `EBVbbb` analog, built through
# the public `SEDModel.build` grammar via `tengri_disc`) applies the intended
# value: $k(\lambda) = R_V\,k_{\rm raw}(\lambda)/k_{\rm raw}(V)$, so $k(V)=R_V$.
# The two conventions therefore differ by one constant factor in $A_\lambda$ at
# matched $E(B{-}V)$ and have identical shape; the table printed below gives the
# declared, effective and relative values.
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
w_t0, L_t0 = tengri_disc("qsogen", ebv_disc=0.0)
resolved_params(tengri_disc.last_model)
w_t3, L_t3 = tengri_disc("qsogen", ebv_disc=_EBV_DEMO)
ratio_tengri = np.divide(L_t3, L_t0, out=np.ones_like(L_t3), where=L_t0 > 0)
L_thb_red = A.apply_bbb_reddening(w_thb, L_thb, _EBV_DEMO)
ratio_af = np.divide(L_thb_red, L_thb, out=np.ones_like(L_thb_red), where=L_thb > 0)
w_tr, L_tr = tengri_disc("qsogen", ebv_disc=_EBV_DEMO * _K_RAW_V / _R_V_SMC)
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
    f"§4  matched E(B-V)={_EBV_DEMO}: max|A_tengri - A_AGNFITTER| = "
    f"{np.max(np.abs(_a_t - _a_af)):.3f} mag  (ratio A_t/A_af median = "
    f"{np.median(_a_t / _a_af):.4f}, expected {_R_V_SMC / _K_RAW_V:.4f})"
)
print(
    f"§4  rescaled E(B-V)={_EBV_DEMO}/{_CONV:.4f}: max|A_tengri - A_AGNFITTER| = "
    f"{np.max(np.abs(_a_tr - _a_af)):.4f} mag (pure-convention check)"
)
print("§4  Prevot reddening conventions")
print("    | quantity                                        |   value |")
print(f"    | R_V declared by BBBred_Prevot (unused there)    | {_R_V_SMC:7.3f} |")
print(f"    | AGNfitter-rX effective R_V = k_raw(V)           | {_K_RAW_V:7.3f} |")
print(f"    | tengri R_V (k(V) = R_V)                         | {_R_V_SMC:7.3f} |")
print(f"    | A_tengri / A_AGNfitter-rX at equal E(B-V)       | {_CONV:7.4f} |")
print(f"    | E(B-V)_tengri for equal A_lambda: E(B-V)_AF  /  | {_CONV:7.4f} |")

# %% [markdown]
# ### §4b qsogen's *own* reddening law — a different curve and convention
#
# qsogen (which builds the THB21 disc) reddens with a different, empirically
# derived quasar extinction curve (Temple, Hewett & Banerji 2021, from
# SDSS DR7 quasars, not the SMC), reached through the composable `atten`
# sub-block: `agn={'atten': {'type': 'qsogen', ...}}`. The qsogen extinction
# curve is recovered here by measuring the ratio of two public `SEDModel.build`
# predictions at $E(B{-}V)$ and 0, the same method as §4.
#
# **Verification Status:** CROSSVAL; Attenuation law library

# %%
def tengri_disc_atten(disc_type, atten_type, ebv, **atten_params):
    """Disc SED with a NAMED atten-block law at a given E(B-V), via the public grammar."""
    atten = {"type": atten_type, "ebv": Fixed(ebv)}
    atten.update({k: Fixed(v) for k, v in atten_params.items()})
    m = SEDModel.build(
        ssp_data=ssp, sfh=SFH_FIDUCIAL, dust_attenuation=NO_DUST,
        agn={
            "type": "composable",
            "disc": {"type": disc_type, "all_params": Fixed(DEFAULT)},
            "torus": {"type": "none"}, "nlr": {"type": "none"}, "blr": {"type": "none"},
            "atten": atten,
            "agn_log_lbol": Fixed(11.0), "all_params": Fixed(DEFAULT), "norm": "independent",
        },
        neb={"type": "none"}, redshift=Fixed(0.0),
    )
    tengri_disc_atten.last_model = m
    pred = m.predict({})
    w = np.asarray(pred.sed.components["wavelength"])
    return w, np.asarray(pred.sed.components["sed_agn_disc"])


_wl_ext = np.geomspace(1e3, 1e4, 300)
w_q0, L_q0 = tengri_disc_atten("qsogen", "none", 0.0)
resolved_params(tengri_disc_atten.last_model)
w_q1, L_q1 = tengri_disc_atten("qsogen", "qsogen", 0.3)
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
    f"§4b  A_V/E(B-V) at V=5500 Å:  AGNFITTER-RX={_at_ext(_k_af_ext, 5500):.3f}  "
    f"tengri-Prevot={_at_ext(_k_tengri_ext, 5500):.3f}  tengri-qsogen={_at_ext(_A_qsogen_on_grid, 5500):.3f}"
)
print(
    f"§4b  A(1500)/A(V) (UV steepness):  AGNFITTER-RX={_at_ext(_k_af_ext, 1500) / _at_ext(_k_af_ext, 5500):.2f}  "
    f"tengri-qsogen={_at_ext(_A_qsogen_on_grid, 1500) / _at_ext(_A_qsogen_on_grid, 5500):.2f}"
)

# %% [markdown]
# ## §5 Galaxy attenuation: Calzetti curve
#
# AGNfitter-rX's GALAXY (host stellar) component is reddened at fit time by a
# Calzetti law applied to the whole galaxy continuum, a knob distinct from §4's
# disc-only `EBVbbb`. tengri's `dust_attenuation={'law': 'calzetti', ...}`
# implements the Calzetti et al. (2000) curve
#
# $$k'(\lambda) = 2.659\,(-2.156 + 1.509\,x - 0.198\,x^2 + 0.011\,x^3) + R_V
# \quad (\lambda < 0.63\,\mu{\rm m}),$$
# $$k'(\lambda) = 2.659\,(-1.857 + 1.040\,x) + R_V \quad (\lambda \ge 0.63\,\mu{\rm m}),$$
#
# with $x = 1/\lambda_{\mu{\rm m}}$, $R_V = 4.05$ and $\tau_V = R_V\,E(B{-}V)/1.086$,
# applied to the fiducial host through the public build at the $E(B{-}V)_{\rm gal}$
# set in the cell. Left: inside the calibrated range (0.12–2.2 µm, shaded) tengri
# follows the analytic curve. Right: outside it the codes diverge, and the
# reference curve is upstream's own evaluation (`GALAXYred_Calzetti`,
# reproduced by the driver on the wavelength grid the fit uses).
#
# **Caveat:** below 0.12 µm upstream extrapolates linearly from two grid
# points and adds a second $R_V$ (the extrapolation depends on the grid
# spacing), and in the near infrared its linear branch turns negative (the
# crossing is printed), which *brightens* the template. Neither is physical, so the comparison is made only
# inside the calibrated range; the printed table gives the size of both tails.
# tengri's tails are a deliberate choice, not a measurement either. The
# two-screen Charlot & Fall (2000) law has no working upstream counterpart:
# the function is defined but never called, and its wavelength argument mixes
# µm with Å.
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
    f"§5  Calzetti parity (0.12-2.2 um): max|A_tengri - A_analytic| = "
    f"{np.max(np.abs(A_lambda_gal_on_grid[_m_cal] - _k_analytic[_m_cal])):.4f} mag/E(B-V)"
)
print("§5  A_lambda / E(B-V) outside the calibrated range")
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
# ## §6 Cold dust infrared emission
#
# AGNfitter-rX ships two libraries: S17 (Schreiber et al. 2018, dust continuum
# plus PAH through $T_{\rm dust}$ and $f_{\rm PAH}$) and DH02_CE01 (Dale & Helou
# 2002 with Chary & Elbaz 2001, indexed by IR luminosity). Each panel builds a
# minimal tengri model (`SEDModel.build(..., dust_emission={'type': ..., ...})`)
# whose absorbed luminosity comes from the build's own dust-attenuated stellar
# continuum, and reads `sed_dust_ir` off the prediction. A radio block is
# included so the model wavelength grid reaches beyond the far infrared; the
# printed grid limits below show what a build without one loses.
#
# `schreiber2018` and `schreiber2016` take `dust_T` and `dust_f_pah`; the S17
# nodes stay inside the $T_{\rm dust}$ and $f_{\rm PAH}$ ranges the paper fits
# (Table 1). `dale2014` takes `dust_alpha_dale`. `dh02_ce01` is indexed by the
# model's own realized IR luminosity (`pred.l_tir`), so its node is read off
# rather than set: `dust_tau_v` is bisected until the realized luminosity lands
# on a grid node, and a further ladder of `dust_tau_v` screens shows the nodes
# it reaches.
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
    f"§6  model wavelength grid ends at {w_dh.max() / 1e4:.4g} um with a radio block and at "
    f"{_w_nr.max() / 1e4:.4g} um without one"
)
print("§6  S17 node table (T_dust [K], f_PAH, median|log10 ratio| over 3-300 um):")
for T, fpah, resid in _s17_resid:
    print(f"    T={T:5.1f} K  f_PAH={fpah:.2f}  median|Delta log10| = {resid:.4f}")
print(
    f"§6  dh02_ce01: realized log10(L_TIR/Lsun) = {_log_lir_realized:.3f}; "
    f"nearest AGNFITTER-RX grid node {_dh_node:.3f}; "
    f"median|log10 ratio| over 3-300 um at that node = {_dh_resid:.4f} dex."
)

# %%
_dh_resids = [_dh_resid]
print("§6  dh02_ce01 log L_IR grid via dust_tau_v (node read off pred.l_tir):")
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
# ## §7 Host composite: GA + SB, physical normalization
#
# One `SEDModel.build` with the fiducial host SFH and S17 cold-dust emission
# gives the host's stellar-plus-dust SED at physical normalization
# (`sed_attenuated + sed_dust_ir`, energy-balanced by the build's own
# `dust_eta_balance`). The AGNfitter-rX side sums its GALAXY and STARBURST terms
# as `ymodel` does (`PARAMETERSPACE_AGNfitter.py`): the GA template at tengri's
# present stellar mass (§3), reddened by upstream's own Calzetti evaluation
# (§5), plus the S17 template. The starburst amplitude `SB` is a free parameter
# of the fit, so the matched input is the infrared luminosity: the S17 template
# is scaled until its 8–1000 µm power equals tengri's realized L_TIR, and what is
# tested is the dust shape. The galaxy term carries no free scale here. The
# band-power ratios are tabulated below the figure.
#
# **Caveat:** tengri evaluates dust emission on the model wavelength grid, which
# is sparse in the infrared (the table counts its points per band), so a
# sub-band power integrated on that grid carries the sampling error; the total
# infrared power, integrated over many more points, is the robust comparison.

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
    f"§7  tengri host: pred.l_tir = {pred7.l_tir:.3e} Lsun (log10={_log_lir7:.3f}); "
    f"M*_formed = {pred7.sfh.stellar_mass:.3e}, M*_present = {_m_present7:.3e} Msun"
)
print("§7  tengri / AGNfitter-rX band power, host composite")
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
# ## §8 Terms upstream does not model: host nebular emission and the IGM
#
# AGNfitter-rX's GALAXY component carries no host nebular emission (neither
# lines nor continuum), and the model has no IGM transmission term. Every host
# build on this page is therefore stellar-only (`neb={'type': 'none'}`) and at
# $z=0$, where tengri's observer-frame IGM component (`igm={'type': ...}`) is
# inactive; tengri's nebular models are compared in the other reproduction
# notebooks. The same holds for galaxy attenuation beyond the Calzetti law (§5).

# %% [markdown]
# ## §9a Accretion-disk library face-off
#
# Four AGNfitter-rX disk libraries at matched parameters, unreddened and
# normalized at 2500 Å, plus tengri's grid-tabulated KD18 discs. The decisive
# feature is the 0.7 µm bump (Hα + [N II]): present only in the semi-empirical
# THB21, absent from the theory discs R06, SN12 and KD18. Only disc *shape* is
# compared: every curve is anchored at 2500 Å, so the absolute scale that
# upstream ties to $(M_{\rm BH}, \dot M)$ and tengri to $L_{\rm bol}$ is not
# tested.
#
# * **THB21**: `qsogen` with its `blr` and `feii` blocks (`tengri_qsogen_full`);
#   the continuum alone misses the line forest. The upstream template is a fixed
#   table whose generating parameters are not recorded, so this is a template
#   against a tengri model at default line strengths, not a parameter-matched
#   pair; the Hα/2500 Å table below gives the difference.
# * **SN12** (Slone & Netzer 2012): `slone_netzer`, compared at nodes of
#   AGNfitter-rX's own $(\log M_{\rm BH}, \log \dot M/\dot M_{\rm Edd})$ table,
#   whose axes are printed in §9a.2. Upstream snaps to the nearest node while
#   tengri interpolates, so only nodes are compared.
# * **KD18** (Kubota & Done 2018): tengri's `kd18_agnfitter` block is
#   grid-tabulated on AGNfitter-rX's own axes (`agn_log_mbh`, `agn_log_ledd`),
#   a node-exact tabulation rather than a physical re-derivation. The KD18-family
#   discs carry their own hot corona, so tengri refuses an additional AGN-corona
#   X-ray variant with them (the message is printed in §9a.1), whereas upstream
#   cuts every disc at 200 eV and appends its own power law.
#   `kd18_agnfitter_warmindex` adds a free warm-Comptonization index
#   `agn_gamma_warm`; the repackaged reference holds no warm-index library, so
#   that variant is compared only with `kd18_agnfitter` (tengri against tengri,
#   printed below), not with upstream.
# * **R06** (Richards et al. 2006): `richards2006` returns the table as stored,
#   $L_\nu$, and the panel compares it with the reference array unchanged.

# %%
disk_pairs = [
    ("R06", {}, lambda: tengri_disc("richards2006"), "richards2006"),
    ("SN12", dict(log_mbh=8.6, edd_index=10),
     lambda: tengri_disc("slone_netzer", agn_log_mbh=8.6, agn_log_ledd=-2.0),
     "slone_netzer (Slone & Netzer 12)"),
    ("KD18", dict(log_mbh=8.0, log_edd=-0.75),
     lambda: tengri_disc("kd18_agnfitter", agn_log_mbh=8.0, agn_log_ledd=-0.75),
     "kd18_agnfitter (grid-tabulated, node-exact)"),
    ("THB21", {}, tengri_qsogen_full, "qsogen + blr + FeII"),
]


def _val_at(w, L, lam):
    o = np.argsort(np.asarray(w))
    return float(np.interp(lam, np.asarray(w)[o], np.asarray(L)[o]))


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
                "tengri, native grid": _val_at(w_t, t_norm, 6563),
                "tengri, interpolated onto the reference grid": _val_at(w_a, _t_on_af, 6563),
                "AGNfitter-rX THB21 template": _val_at(w_a, a_norm, 6563),
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
print("§9a  disc-library shape residuals (anchor 2500 Å, 1200 Å-1 um):")
for af_name, af_kw, tengri_fn, _label in disk_pairs:
    w_a, L_a = A.disk_template(af_name, **af_kw)
    w_t, L_t = tengri_fn()
    oa_, ot_ = np.argsort(w_a), np.argsort(w_t)
    w_a_s, a_s = np.asarray(w_a)[oa_], norm_at(w_a, L_a, ANCHOR)[oa_]
    m = (w_a_s >= 1.2e3) & (w_a_s <= 1e4) & (a_s > 0)
    t_on = np.interp(w_a_s[m], np.asarray(w_t)[ot_], norm_at(w_t, L_t, ANCHOR)[ot_])
    logr = np.abs(np.log10(t_on / a_s[m]))
    print(f"  {af_name:6s} median = {np.median(logr):.3f} dex   max = {logr.max():.3f} dex")
print("§9a  Hα / L(2500 Å) ladder, THB21 panel")
print("    | curve                                          | L(6563 Å) / L(2500 Å) |")
for _k, _v in _HA_LADDER.items():
    print(f"    | {_k:46s} | {_v:21.3f} |")

# %%
# KD18 grid-tabulated vs warm-index variant, at fixed (M_BH, lambda_Edd),
# far warm index vs kd18_agnfitter's baked-in default.
w_kd, L_kd = tengri_disc("kd18_agnfitter", agn_log_mbh=8.0, agn_log_ledd=-0.75)
resolved_params(tengri_disc.last_model)
w_kdw, L_kdw = tengri_disc("kd18_agnfitter_warmindex", agn_log_mbh=8.0, agn_log_ledd=-0.75, agn_gamma_warm=1.5)
_m = (w_kd > 1.2e3) & (w_kd < 1e4)
_kdn = norm_at(w_kd, L_kd, ANCHOR)
_kdwn_on = np.interp(w_kd, w_kdw, norm_at(w_kdw, L_kdw, ANCHOR))
_kd_logr = np.abs(np.log10(np.clip(_kdwn_on[_m], 1e-30, None) / np.clip(_kdn[_m], 1e-30, None)))
print(
    f"§9a  kd18_agnfitter vs kd18_agnfitter_warmindex (gamma_warm=1.5) at "
    f"(log M_BH, log lambda_Edd)=(8.0, -0.75): max|log10 ratio| = {_kd_logr.max():.3f} dex "
    f"(={100 * (10 ** _kd_logr.max() - 1):.0f}%)"
)

# %%
print("§9a  disc composable menu:", sorted(r["name"] for r in tengri.list_agn_blocks(category="disc")))

# %% [markdown]
# ## §9a.1 KD18 grid discs and the corona double-count guard
#
# `kd18_agnfitter` (and `kd18_agnfitter_warmindex`) already carry a Kubota &
# Done (2018) hot corona; asking for an *additional* AGN-corona X-ray variant
# (for instance `xray={'type': 'yang20'}`) is refused at build time. The cell
# below triggers the refusal and prints the message. Every KD18 panel in this
# notebook therefore builds the disc with no X-ray group.
#
# **Caveat:** the refusal is tengri's policy, not upstream's: upstream replaces
# everything above 200 eV in any disc by its own power law, which also removes a
# hot corona the template may carry.

# %%
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
    print(f"§9a.1  refused, as required: {type(_exc).__name__}: {_exc}")
else:
    raise AssertionError("kd18_agnfitter + xray='yang20' was accepted; the double-count guard is gone")

# %% [markdown]
# ### §9a.2 SN12 and KD18 node grids
#
# SN12 at four nodes off `disk_axes('SN12')` (index pairs (0,0), (4,6), (8,11),
# (2,9) in the $(\log M_{\rm BH}, \dot M/\dot M_{\rm Edd})$ table, whose axes are
# printed below), via `tengri_disc('slone_netzer', agn_log_mbh=...,
# agn_log_ledd=...)` against `disk_template('SN12', log_mbh=...,
# edd_index=...)`. KD18 at four $(\log M_{\rm BH}, \log\lambda_{\rm Edd})$ nodes,
# via `tengri_disc('kd18_agnfitter', ...)` against `disk_template('KD18', ...)`.
# Both are anchored at 2500 Å and compared over 1200 Å–1 µm; the window table
# lists every node.

# %%
sn12_axes = A.disk_axes("SN12")
print(
    f"§9a.2  SN12 table axes: log M_BH = {np.round(sn12_axes['log_mbh'], 2).tolist()}; "
    f"log Mdot/Mdot_Edd = {np.round(sn12_axes['log_edd'], 3).tolist()}"
)
sn12_node_idx = [(0, 0), (4, 6), (8, 11), (2, 9)]
kd18_nodes = [(6.0, -1.5), (6.0, 0.0), (7.43, -0.96), (8.0, -1.5)]

cases_9a2 = []
for i, j in sn12_node_idx:
    log_mbh = float(sn12_axes["log_mbh"][i])
    log_edd = float(sn12_axes["log_edd"][j])
    w_a, L_a = A.disk_template("SN12", log_mbh=log_mbh, edd_index=j)
    w_t, L_t = tengri_disc("slone_netzer", agn_log_mbh=log_mbh, agn_log_ledd=log_edd)
    label = f"SN12 ({i},{j}) logM={log_mbh:.1f} logEdd={log_edd:.2f}"
    a_n, t_n = norm_at(w_a, L_a, ANCHOR), norm_at(w_t, L_t, ANCHOR)
    _assert_comparable(a_n, t_n, name=f"§9a.2 {label}")
    cases_9a2.append((label, w_a, a_n, w_t, t_n))
for mbh, edd in kd18_nodes:
    w_a, L_a = A.disk_template("KD18", log_mbh=mbh, log_edd=edd)
    w_t, L_t = tengri_disc("kd18_agnfitter", agn_log_mbh=mbh, agn_log_ledd=edd)
    label = f"KD18 ({mbh:g},{edd:g})"
    a_n, t_n = norm_at(w_a, L_a, ANCHOR), norm_at(w_t, L_t, ANCHOR)
    _assert_comparable(a_n, t_n, name=f"§9a.2 {label}")
    cases_9a2.append((label, w_a, a_n, w_t, t_n))

plt.rcParams["figure.dpi"] = 100  # keep the rendered notebook under the figure-size budget
fig, (ax, ax_r), _ratios_9a2 = V.sweep_fig(
    cases_9a2, ref_label="AGNfitter-rX", title="§9a.2 SN12 and KD18 node grids",
    xlim=(1.2e3, 1e4), xlabel=r"$\lambda$ [Å]", ylabel=r"$L_\nu$ (norm. at 2500 Å)",
    cmap=None,
)
fig.tight_layout()
save_fig("agnfitter_09a1_disk_nodes.png")
plt.rcParams["figure.dpi"] = 150

_win_9a2 = V.window_rows(cases_9a2, lo=1200, hi=1e4)
V.print_window_table(_win_9a2, ref_name="AGNfitter-rX", title="§9a.2 SN12 and KD18 node grids")

# %% [markdown]
# ## §9b Accretion-disk reddening sweep
#
# The disk color excess $E(B{-}V)_{\rm BBB}$ sweeps the UV continuum through the
# Prevot SMC law on both sides, applied to the THB21 template (upstream) and to
# tengri's `qsogen` disc. At matched $E(B{-}V)$ tengri's dashed curves sit
# **below** AGNfitter-rX's solid ones by the convention factor of §4, which the
# printed ratios reproduce.

# %%
fig, ax = plt.subplots(figsize=(7.5, 4.8))
w_thb, L_thb = A.disk_template("THB21")
_, L_te0 = tengri_disc("qsogen", ebv_disc=0.0)
msk = (w_thb > 8e2) & (w_thb < 1e4)
print("§9b  A(1500 Å) per E(B-V) [mag]:")
for ebv, c in [(0.1, "C2"), (0.3, "C1"), (0.5, "C3")]:
    L_red = A.apply_bbb_reddening(w_thb, L_thb, ebv)
    ratio_af = np.divide(L_red, L_thb, out=np.ones_like(L_red), where=L_thb > 0)
    ax.loglog(w_thb[msk], ratio_af[msk], c, ls="-", lw=1.4, label=f"AGNFITTER  E(B−V) = {ebv:g}")
    w_te, L_te = tengri_disc("qsogen", ebv_disc=ebv)
    ratio_te = np.divide(L_te, L_te0, out=np.ones_like(L_te), where=L_te0 > 0)
    msk_te = (w_te > 8e2) & (w_te < 1e4)
    ax.loglog(w_te[msk_te], ratio_te[msk_te], c, ls="--", lw=1.4, label=f"tengri  E(B−V) = {ebv:g}")
    a_af = -2.5 * np.log10(np.interp(1500.0, w_thb, ratio_af))
    a_te = -2.5 * np.log10(np.interp(1500.0, w_te, ratio_te))
    print(f"  E(B-V)={ebv:g}:  AGNFITTER = {a_af:.2f}   tengri = {a_te:.2f}   (ratio {a_te / a_af:.3f}, §4 convention {_CONV:.3f})")
w_tm, L_tm = tengri_disc("qsogen", ebv_disc=0.3 / _CONV)
ratio_tm = np.divide(L_tm, L_te0, out=np.ones_like(L_tm), where=L_te0 > 0)
msk_tm = (w_tm > 8e2) & (w_tm < 1e4)
ax.loglog(w_tm[msk_tm], ratio_tm[msk_tm], "k:", lw=1.8, label=f"tengri E(B−V)=0.3/{_CONV:.3f} → on AF 0.3")
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
# Four headline AGNfitter-rX torus libraries at matched grid nodes,
# peak-normalized, plus a parity table across further averaged reductions of the
# same NK08/SKIRTOR/CAT3D families, read through `torus_template` and
# `torus_axes`. **S04**: `silva04` at one column density. **NK08**:
# `nenkova_agnfitter`, the inclination-only reduction of the CLUMPY grid that
# upstream calls `NK0_mean_1p`; the paper's preferred NK08 variant has three
# parameters and appears among the reductions (§9c.1). **SKIRTOR**: `skirtor`
# (the full X-CIGALE grid) at the opening angle, inclination and optical depth
# set in the cell; AGNfitter-rX's own averaged `SKIRTOR_mean_3p` reduction is
# `skirtor_agnfitter` (§9c.1). **CAT3D-Wind**: `cat3d_wind` at a high wind
# fraction.
#
# Two properties of the comparison matter for every table below. First, the
# four headline reference arrays are repackaged on a 1024-point grid that is
# uniform in $\log\lambda$, by linear-in-wavelength interpolation of upstream's
# native axes; the reductions keep their native axes. The reference sampling is
# printed under the figure, and it sets a floor on how closely a case can agree.
# Second, tengri does not evaluate these blocks on the reference wavelengths but
# on the model grid; the printed point counts show how coarse that grid is
# longward of 10 µm for some blocks.
#
# **Caveat:** tengri's torus blocks are evaluated on the model's rest-frame
# wavelength grid (the SSP grid joined with each block's own template axis, which
# reaches 1 mm for the averaged NK08/SKIRTOR reductions), so the averaged
# reductions are sampled at their template nodes and nowhere else. Residuals larger
# than the reference floor in the tables below reflect that sampling, not the
# template physics.
# A case is called node-exact only when `max|tengri/AGNfitter-rX − 1|` is below
# the tolerance printed under each table.
#
# **Verification Status:** CROSSVAL; SKIRTOR torus (mean 3-param)

# %%
torus_pairs = [
    ("S04", "S04", lambda: tengri_torus("silva04", log_nh_silva=23.0), "silva04 (log N_H = 23)", dict(log_nh=23.0)),
    ("NK08", "NK08", lambda: tengri_torus("nenkova_agnfitter", cos_inc=0.8660254), "nenkova_agnfitter (incl 30°)", dict(incl=30.0)),
    ("SKIRTOR", "SKIRTOR", lambda: tengri_torus("skirtor", cos_inc=0.8660254, oa_skirtor=40.0, tau_skirtor=7.0), "skirtor (oa 40°, incl 30°, τ 7)", dict(oa=40.0, incl=30.0, tau=7.0)),
    ("CAT3D", "CAT3D-Wind", lambda: tengri_torus("cat3d_wind", cos_inc=1.0, a_cat3d=-2.0, fwd_cat3d=1.75), "cat3d_wind (incl 0°, a −2, f_wd 1.75)", dict(incl=0.0, a=-2.0, fwd=1.75)),
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
        resolved_params(tengri_torus.last_model)
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

# %%
print("§9c  sampling of the reference arrays and of tengri's model grid")
print("    | library  | reference n | uniform in log λ | reference range [um] | tengri n (> 10 um) | tengri grid max [um] |")
for _name, (_wa, _wt) in _TORUS_SAMPLING.items():
    _dl = np.diff(np.log10(np.sort(_wa)))
    _uniform = bool(np.allclose(_dl, _dl.mean(), rtol=1e-2))
    print(
        f"    | {_name:8s} | {_wa.size:11d} | {str(_uniform):16s} | "
        f"{_wa.min() / 1e4:8.3g} - {_wa.max() / 1e4:<9.3g} | {int(np.sum(_wt > 1e5)):18d} | {_wt.max() / 1e4:20.4g} |"
    )

# %% [markdown]
# ### §9c.1 Further torus reductions
#
# `nenkova_agnfitter_2p` and `_3p` (NK08 with the opening-angle axis, then also
# the optical-depth axis, retained; `_3p` is the paper's preferred NK08
# variant), `skirtor_agnfitter_1p` and `_2p` (SKIRTOR averaged over fewer axes
# than the headline `skirtor_agnfitter` 3p reduction), and `cat3d_wind_lowfwd`,
# the low-wind-fraction half of upstream's single CAT3D library (joined to the
# high-wind half in §9c.5). Each is built once against the driver's matching
# `torus_axes(reduction)` node and peak-normalized; the reference axes of these
# five are upstream's native ones.

# %%
print("§9c.1  torus composable menu:", sorted(r["name"] for r in tengri.list_agn_blocks(category="torus")))

print("§9c.1  five reductions: axes x node counts (tengri torus_axes) vs upstream row count:")
for _name in ("NK08_2P", "NK08_3P", "SKIRTOR_MEAN1P", "SKIRTOR_MEAN2P", "CAT3D_LOWFWD"):
    _axes = A.torus_axes(_name)
    _sizes = {k: len(v) for k, v in _axes.items()}
    _n_rows = int(np.prod(list(_sizes.values())))
    print(f"    {_name:16s} axes={_sizes}  upstream rows = product = {_n_rows}")

_reduction_pairs = [
    ("NK08_2P", lambda incl, oa: tengri_torus("nenkova_agnfitter_2p", agn_cos_inc=np.cos(np.deg2rad(incl)), agn_oa_nenkova=oa), dict(incl=30.0, oa=40.0)),
    ("NK08_3P", lambda incl, oa, tau: tengri_torus("nenkova_agnfitter_3p", agn_cos_inc=np.cos(np.deg2rad(incl)), agn_oa_nenkova=oa, agn_tv_nenkova=tau), dict(incl=30.0, oa=40.0, tau=60.0)),
    ("SKIRTOR_MEAN1P", lambda incl: tengri_torus("skirtor_agnfitter_1p", agn_incl_skirtor=incl), dict(incl=30.0)),
    ("SKIRTOR_MEAN2P", lambda oa, incl: tengri_torus("skirtor_agnfitter_2p", agn_oa_skirtor=oa, agn_incl_skirtor=incl), dict(oa=40.0, incl=30.0)),
    ("CAT3D_LOWFWD", lambda incl, a, fwd: tengri_torus("cat3d_wind_lowfwd", agn_cos_inc=np.cos(np.deg2rad(incl)), agn_a_cat3d_lowfwd=a, agn_fwd_cat3d_lowfwd=fwd), dict(incl=0.0, a=-2.0, fwd=0.3)),
]
print("§9c.1  torus-reduction parity (peak-normalized, 1-100 um median|log10 ratio|):")
fig, ax = plt.subplots(figsize=(9, 4.8))
for (name, fn, kw), c in zip(_reduction_pairs, ["C0", "C1", "C2", "C3", "C4"]):
    w_a, L_a = A.torus_template(name, **kw)
    w_t, L_t = fn(**kw)
    _grid = np.geomspace(max(w_a.min(), w_t.min()) * 1.01, min(w_a.max(), w_t.max()) * 0.99, 300)
    a_on = np.interp(np.log10(_grid), np.log10(np.sort(w_a)), norm_peak(L_a)[np.argsort(w_a)])
    t_on = np.interp(np.log10(_grid), np.log10(np.sort(w_t)), norm_peak(L_t)[np.argsort(w_t)])
    ratio = t_on / np.clip(a_on, 1e-30, None)
    ax.loglog(_grid, ratio, c, lw=1.4, label=name)
    m_ir = (_grid > 1e4) & (_grid < 1e6) & (a_on > 1e-3)
    print(
        f"    {name:16s} node={kw}  median = {np.median(np.abs(np.log10(ratio[m_ir]))):.3f} dex  "
        f"(points above 10 um: tengri {int(np.sum(w_t > 1e5))}, reference {int(np.sum(w_a > 1e5))})"
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

# %% [markdown]
# ### §9c.2 S04 log N_H and NK08 inclination
#
# S04 at fractions {0, .25, .5, .75, .99} of the `torus_axes('S04')` column
# density axis; NK08 at fractions {0, .25, .5, .75, 1} of its inclination axis
# (the axis lengths are printed below). Built via `tengri_torus('silva04',
# log_nh_silva=...)` and `tengri_torus('nenkova_agnfitter', cos_inc=...)`, each
# against `torus_template` at the same node, peak-normalized over 1–100 µm. The
# figure plots four nodes per library (the middle fraction dropped to stay within
# the panel budget); the table covers all ten. The 0.99 fraction keeps the S04
# node clear of the axis end, where float32 storage of the axis would place it
# outside tengri's bounds.

# %%
s04_axis = A.torus_axes("S04")["log_nh"]
nk08_axis = A.torus_axes("NK08")["incl"]
print(
    f"§9c.2  S04 log N_H axis: {s04_axis.size} nodes, {s04_axis.min():.2f}-{s04_axis.max():.2f}; "
    f"NK08 inclination axis: {nk08_axis.size} nodes, {nk08_axis.min():g}-{nk08_axis.max():g} deg"
)
s04_fracs = [0.0, 0.25, 0.5, 0.75, 0.99]
nk08_fracs = [0.0, 0.25, 0.5, 0.75, 1.0]

cases_9c0, cases_9c0_fig = [], []
for f in s04_fracs:
    idx = int(round(f * (len(s04_axis) - 1)))
    log_nh = float(s04_axis[idx])
    w_a, L_a = A.torus_template("S04", log_nh=log_nh)
    w_t, L_t = tengri_torus("silva04", log_nh_silva=log_nh)
    label = f"S04 log N_H={log_nh:.2f}"
    a_n, t_n = norm_peak(L_a), norm_peak(L_t)
    _assert_comparable(a_n, t_n, name=f"§9c.2 {label}")
    case = (label, w_a, a_n, w_t, t_n)
    cases_9c0.append(case)
    if f != 0.5:
        cases_9c0_fig.append(case)
for f in nk08_fracs:
    idx = int(round(f * (len(nk08_axis) - 1)))
    incl = float(nk08_axis[idx])
    w_a, L_a = A.torus_template("NK08", incl=incl)
    w_t, L_t = tengri_torus("nenkova_agnfitter", cos_inc=float(np.cos(np.deg2rad(incl))))
    label = f"NK08 incl={incl:g}°"
    a_n, t_n = norm_peak(L_a), norm_peak(L_t)
    _assert_comparable(a_n, t_n, name=f"§9c.2 {label}")
    case = (label, w_a, a_n, w_t, t_n)
    cases_9c0.append(case)
    if f != 0.5:
        cases_9c0_fig.append(case)

plt.rcParams["figure.dpi"] = 100  # keep the rendered notebook under the figure-size budget
fig, (ax, ax_r), _ratios_9c0 = V.sweep_fig(
    cases_9c0_fig, ref_label="AGNfitter-rX", title="§9c.2 S04 log N_H and NK08 inclination",
    xlim=(8e3, 3e6), xlabel=r"$\lambda$ [Å]", ylabel=r"$L_\nu$ (norm. at peak)",
    cmap=None,
)
fig.tight_layout()
save_fig("agnfitter_09c0_torus_sweeps.png")
plt.rcParams["figure.dpi"] = 150

_win_9c0 = V.window_rows(cases_9c0, lo=1e4, hi=1e6)
V.print_window_table(_win_9c0, ref_name="AGNfitter-rX", title="§9c.2 S04 log N_H and NK08 inclination")
node_exact_verdict(_win_9c0, "§9c.2")

# %% [markdown]
# ### §9c.3 SKIRTOR (oa, incl, τ) nodes
#
# `skirtor_agnfitter`, the tengri block tabulated from AGNfitter-rX's averaged
# `SKIRTOR_mean_3p`, at four `(oa, incl, τ)` index triples off
# `torus_axes('SKIRTOR')` — (0,3,2), (3,3,2), (5,6,0), (7,9,4) — plus the full
# unaveraged `skirtor` X-CIGALE grid at the (3,3,2) fiducial. `skirtor` carries
# the unaveraged clumpiness and radial-distribution structure, which broadens and
# shifts its IR peak, so that last case differs from the averaged reference by
# construction rather than by discrepancy and is excluded from the node-exact
# count.

# %%
sk_axes = A.torus_axes("SKIRTOR")
sk_triples = [(0, 3, 2), (3, 3, 2), (5, 6, 0), (7, 9, 4)]
cases_9c5 = []
for oi, ii, ti in sk_triples:
    oa = float(sk_axes["oa"][oi])
    incl = float(sk_axes["incl"][ii])
    tau = float(sk_axes["tau"][ti])
    w_a, L_a = A.torus_template("SKIRTOR", oa=oa, incl=incl, tau=tau)
    w_t, L_t = tengri_torus("skirtor_agnfitter", oa_skirtor=oa, incl_skirtor=incl, tv_skirtor=tau)
    label = f"SKIRTOR ({oi},{ii},{ti})"
    a_n, t_n = norm_peak(L_a), norm_peak(L_t)
    _assert_comparable(a_n, t_n, name=f"§9c.3 {label}")
    cases_9c5.append((label, w_a, a_n, w_t, t_n))
    if (oi, ii, ti) == (3, 3, 2):
        w_ref, L_ref, w_af, L_af = w_a, a_n, w_t, t_n  # fiducial node (peak-norm), reused by §9c.4 below

w_xc, L_xc_raw = tengri_torus("skirtor", cos_inc=float(np.cos(np.deg2rad(30.0))), oa_skirtor=40.0, tau_skirtor=7.0)
L_xc = norm_peak(L_xc_raw)
_assert_comparable(L_ref, L_xc, name="§9c.3 SKIRTOR full grid @ fiducial")
cases_9c5.append(("SKIRTOR full grid @ fiducial", w_ref, L_ref, w_xc, L_xc))

plt.rcParams["figure.dpi"] = 100  # keep the rendered notebook under the figure-size budget
fig, (ax, ax_r), _ratios_9c5 = V.sweep_fig(
    cases_9c5, ref_label="AGNfitter-rX", title="§9c.3 SKIRTOR (oa, incl, τ) nodes",
    xlim=(8e3, 3e6), xlabel=r"$\lambda$ [Å]", ylabel=r"$L_\nu$ (norm. at peak)",
    cmap=None,
)
fig.tight_layout()
save_fig("agnfitter_09c5_skirtor_nodes.png")
plt.rcParams["figure.dpi"] = 150

_win_9c5 = V.window_rows(cases_9c5, lo=1e4, hi=1e6)
V.print_window_table(_win_9c5, ref_name="AGNfitter-rX", title="§9c.3 SKIRTOR (oa, incl, τ) nodes")
node_exact_verdict([r for r in _win_9c5 if "full grid" not in r["label"]], "§9c.3 (averaged nodes)")

# %% [markdown]
# ### §9c.4 SKIRTOR: two reductions
#
# `skirtor_agnfitter` (tabulated from AGNfitter-rX's averaged `SKIRTOR_mean_3p`)
# and `skirtor` (the full unaveraged X-CIGALE grid) at the same (oa, incl, τ)
# fiducial plotted in §9c.3. The panel there already shows both curves, so only
# the IR peak wavelengths are printed here.

# %%
msk_ref = (w_ref > 5e3) & (w_ref < 1e7)
msk_af = (w_af > 5e3) & (w_af < 1e7)
msk_xc = (w_xc > 5e3) & (w_xc < 1e7)
_peak_ref = float(w_ref[msk_ref][np.argmax(L_ref[msk_ref])])
_peak_af = float(w_af[msk_af][np.argmax(L_af[msk_af])])
_peak_xc = float(w_xc[msk_xc][np.argmax(L_xc[msk_xc])])
print(
    f"§9c.4  IR peak wavelength: AGNFITTER SKIRTOR_mean_3p = {_peak_ref / 1e4:.1f} um, "
    f"tengri skirtor_agnfitter = {_peak_af / 1e4:.1f} um, "
    f"tengri skirtor (full grid) = {_peak_xc / 1e4:.1f} um"
)

# %% [markdown]
# ### §9c.5 CAT3D-Wind: the wind-fraction union
#
# Upstream's CAT3D is one library whose wind-fraction axis joins two tables of
# the same pickle; tengri splits it over two blocks, `cat3d_wind` (high wind
# fractions) and `cat3d_wind_lowfwd` (low ones). The sweep below runs $f_{\rm wd}$
# across the union axis `cat3d_union_axes()` at fixed inclination and `a`,
# choosing the tengri block by the half of the axis a node belongs to, and the
# table lists every node. Four `(incl, a, f_wd)` index triples off
# `torus_axes('CAT3D')` then vary inclination and the radial index together with
# the wind fraction. The polar wind is what CAT3D-Wind is for: it supplies the
# near-infrared excess that equatorial tori miss.

# %%
plt.rcParams["figure.dpi"] = 100  # keep the rendered notebook under the figure-size budget
_INCL0, _A0 = 0.0, -2.0
_u = A.cat3d_union_axes()


def tengri_cat3d_union(incl, a, fwd, sub):
    """tengri CAT3D block for one union node; ``sub`` names the reference sub-library it belongs to."""
    cos_inc = float(np.cos(np.deg2rad(incl)))
    if sub == "CAT3D_LOWFWD":
        return tengri_torus("cat3d_wind_lowfwd", agn_cos_inc=cos_inc, agn_a_cat3d_lowfwd=a, agn_fwd_cat3d_lowfwd=fwd)
    return tengri_torus("cat3d_wind", cos_inc=cos_inc, a_cat3d=a, fwd_cat3d=fwd)


fig, ax = plt.subplots(figsize=(8, 5))
_fwd_grid = np.geomspace(1e4, 1e6, 300)
_plot_fwd = {float(f) for f in _u["fwd"][[0, 2, 4, 5, 8, 10]]}
_colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(_plot_fwd)))
_union_rows, _ic = [], 0
print(
    f"§9c.5  CAT3D union axes: {_u['incl'].size} inclinations x {_u['a'].size} radial indices x "
    f"{_u['fwd'].size} wind fractions = {_u['incl'].size * _u['a'].size * _u['fwd'].size} nodes"
)
print(f"§9c.5  wind-fraction sweep at incl={_INCL0:g} deg, a={_A0:g} (peak-normalized, 1-100 um):")
print("    |  f_wd | reference sub-library | tengri block        | median|t/a-1| | max|t/a-1| |")
for fwd in _u["fwd"]:
    fwd = float(fwd)
    w_a, L_a, sub = A.cat3d_union_template(_INCL0, _A0, fwd)
    w_t, L_t = tengri_cat3d_union(_INCL0, _A0, fwd, sub)
    a_on = np.interp(np.log10(_fwd_grid), np.log10(w_a), norm_peak(L_a))
    t_on = np.interp(np.log10(_fwd_grid), np.log10(w_t), norm_peak(L_t))
    ok = a_on > 1e-3
    _res = np.abs(t_on[ok] / a_on[ok] - 1.0)
    _union_rows.append({"label": f"f_wd={fwd:g} ({sub})", "max_abs_dev": float(_res.max())})
    _blk = "cat3d_wind_lowfwd" if sub == "CAT3D_LOWFWD" else "cat3d_wind"
    print(f"    | {fwd:5.2f} | {sub:21s} | {_blk:19s} | {np.median(_res):13.4g} | {_res.max():10.4g} |")
    if fwd in _plot_fwd:
        c = _colors[_ic]
        _ic += 1
        ax.loglog(w_a, norm_peak(L_a), color=c, ls="--", lw=1.2)
        ax.loglog(w_t, norm_peak(L_t), color=c, ls="-", lw=1.5, label=f"$f_{{wd}}$ = {fwd:g}")
node_exact_verdict(_union_rows, "§9c.5 union sweep")

c3_axes = A.torus_axes("CAT3D")
c3_triples = [(0, 0, 5), (3, 1, 5), (3, 2, 4), (6, 3, 2)]
print("§9c.5  (incl, a, f_wd) index triples off torus_axes('CAT3D'):")
_tri_rows = []
for (ii, ai, fi), c in zip(c3_triples, ["C5", "C6", "C7", "C8"]):
    incl = float(c3_axes["incl"][ii])
    a_cat3d = float(c3_axes["a"][ai])
    fwd = float(c3_axes["fwd"][fi])
    w_a, L_a = A.torus_template("CAT3D", incl=incl, a=a_cat3d, fwd=fwd)
    w_t, L_t = tengri_torus("cat3d_wind", cos_inc=float(np.cos(np.deg2rad(incl))), a_cat3d=a_cat3d, fwd_cat3d=fwd)
    label = f"({ii},{ai},{fi})"
    _assert_comparable(norm_peak(L_a), norm_peak(L_t), name=f"§9c.5 {label}")
    msk_a = (w_a > 5e3) & (w_a < 1e7)
    msk_t = (w_t > 5e3) & (w_t < 1e7)
    ax.loglog(w_a[msk_a], norm_peak(L_a)[msk_a], c, ls="--", lw=1.0)
    ax.loglog(w_t[msk_t], norm_peak(L_t)[msk_t], c, ls="-", lw=1.3, label=f"{label}: incl={incl:g}° a={a_cat3d:g}")
    a_on = np.interp(np.log10(_fwd_grid), np.log10(w_a), norm_peak(L_a))
    t_on = np.interp(np.log10(_fwd_grid), np.log10(w_t), norm_peak(L_t))
    ok = a_on > 1e-3
    _res = np.abs(t_on[ok] / a_on[ok] - 1.0)
    _tri_rows.append({"label": f"{label} incl={incl:g} a={a_cat3d:g} f_wd={fwd:g}", "max_abs_dev": float(_res.max())})
    print(f"  {label} incl={incl:g}° a={a_cat3d:g} f_wd={fwd:g}:  median |ratio-1| = {np.median(_res) * 100:.2f}%   max = {_res.max() * 100:.2f}%")
node_exact_verdict(_tri_rows, "§9c.5 triples")
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
plt.rcParams["figure.dpi"] = 150

# %% [markdown]
# ### §9c.6 Inclination-dependent torus power
#
# Upstream's torus templates do not share a bolometric luminosity: the template
# integral falls with inclination, so at fixed `TO` a more inclined torus
# radiates less. tengri normalizes every node to the same torus power and lets
# the torus luminosity fraction carry the amplitude. Every panel above is
# peak-normalized and therefore blind to this; the table compares the two
# normalizations directly, each relative to the lowest-inclination node.
#
# **Caveat:** the difference is one of parametrization, not an error, but it
# matters when a posterior is moved between the codes: `TO` and tengri's torus
# luminosity fraction are related by the ratio tabulated here.

# %%
_bol_sets = [
    ("NK08", A.torus_axes("NK08")["incl"],
     lambda i: A.torus_bolometric("NK08", incl=i),
     lambda i: tengri_torus("nenkova_agnfitter", cos_inc=float(np.cos(np.deg2rad(i))))),
    ("CAT3D", A.torus_axes("CAT3D")["incl"],
     lambda i: A.torus_bolometric("CAT3D", incl=i, a=-2.0, fwd=1.75),
     lambda i: tengri_torus("cat3d_wind", cos_inc=float(np.cos(np.deg2rad(i))), a_cat3d=-2.0, fwd_cat3d=1.75)),
]
print("§9c.6  torus power relative to the lowest-inclination node")
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
# ## §10 X-ray corona via α_ox–L₂₅₀₀
#
# AGNfitter-rX ties the 2 keV corona to the 2500 Å disk continuum through the
# Just et al. (2007) relation, $\alpha_{\rm ox} = -0.137\,\log L_{2500} + 2.638
# + \Delta\alpha_{\rm ox}$, then lays down a power law of photon index $\Gamma$
# with a 300 keV exponential cutoff, joined to the disc by a hard step at
# 200 eV. tengri exposes the same relations in the public `alpha_ox_from_l2500`.
# The figure overlays tengri's `xray_agn_corona_from_disc(apply_anisotropy=False)`
# on the driver's `disk_xray_extension` across $\Delta\alpha_{\rm ox}$ (and, in
# the table, $\Gamma$). The table reports the ratio in a hard window (0.5–100 keV)
# and a soft one (0.2–0.5 keV), each with tengri's default absorbing column
# (`log_nh=20`, plus a 1% scattered fraction, neither of which upstream has) and
# with the column switched off (`log_nh=0`): the power law, cutoff and 2 keV
# anchor agree, and the default column is what lowers the ratio at soft energies.
#
# **Verification Status:** PARTIAL; Radio + X-ray + AGN

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
_xray = {scat: A.disk_xray_extension(w_thb, L_thb_at_2500, scatter=scat) for scat in _scat_grid}
_xref = float(np.max(_xray[0.0][1]))

plt.rcParams["figure.dpi"] = 100  # keep the rendered notebook under the figure-size budget
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for scat, c in zip(_scat_grid, ["C2", "C5", "C0", "C6", "C3"]):
    xw, xL = _xray[scat]
    ax.loglog(xw, xL / _xref, c, lw=2.0, alpha=0.4, label=rf"AF  $\Delta\alpha_{{ox}}$={scat:+.1f}")
    L_t = np.asarray(
        xray_agn_corona_from_disc(jnp.asarray(wave_x_10a), L_2500_10a, delta_alpha_ox=scat, gamma=1.8, apply_anisotropy=False)
    )
    ax.loglog(wave_x_10a, L_t / _xref, c, ls="--", lw=1.3)
ax.set_xlim(1e-2, 1e2)
ax.set_ylim(1e-3, 1e3)
ax.set_xlabel(r"$\lambda$ [Å]")
ax.set_ylabel(r"$L_\nu$ (norm.)")
ax.set_title(r"X-ray corona — AF (solid) vs tengri (dashed), $\Gamma$=1.8")
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_fig("agnfitter_10a_alphaox.png")
plt.rcParams["figure.dpi"] = 150

# %%
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


print("§10  X-ray corona, tengri / AGNfitter-rX (apply_anisotropy=False): median ratio and max|ratio-1| per window")
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
# ### §10.1 The α_ox–L₂₅₀₀ relation
#
# tengri offers three calibrations through `alpha_ox_from_l2500`; AGNfitter-rX
# uses the Just et al. (2007) line, evaluated here by the driver's `alpha_ox`.
# Top: the three tengri relations against upstream's; bottom: the difference for
# `just2007`.

# %%
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
print(f"§10.1 α_ox parity: max |tengri just2007 − AGNFITTER-RX| = {_aox_dmax:.2e}")
fig.tight_layout()
save_fig("agnfitter_10c_alphaox_residual.png")
plt.show()

# %% [markdown]
# ## §10b X-ray corona: exact parity, then tengri's default extras
#
# Given the 2500 Å disk luminosity, both codes build the same bare corona
# (Γ=1.8, 300 keV cutoff, normalized via Just+2007). Left: bare prescriptions
# (`xray_agn_corona_from_disc(..., apply_anisotropy=False)`) vs AGNfitter-RX's
# disk X-ray extension. Right: what tengri's *defaults* add — the Yang et al.
# (2022) viewing-angle anisotropy and a host X-ray-binary floor (`xray_xrb`,
# Mineo et al. 2014), neither of which AGNfitter-RX has an analog for.

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
    f"§10b corona parity (0.5-100 keV): bare median ratio = {np.nanmedian(_ratio_bare[_hard]):.4f}, "
    f"max |ratio-1| = {np.nanmax(np.abs(_ratio_bare[_hard] - 1.0)):.3f}; "
    f"defaults median ratio = {np.nanmedian(_ratio_default[_hard]):.4f} (the Yang+22 anisotropy)"
)

# %% [markdown]
# ## §11 Radio
#
# AGNfitter-rX models AGN core/jet radio with a simple power law (SPL, slope
# fixed at $-0.75$ with an exponential cutoff at $10^{13}$ Hz) or a double power
# law (DPL, paper Eq. 2). tengri ships both, `radio_agn` (SPL) and
# `radio_agn_dpl`. The AGNfitter-rX curves below are the upstream equations
# evaluated by the driver (`agn_radio_spl`, `agn_radio_dpl`), each normalized at
# 5 GHz; §11.1 plots them against tengri's with a ratio panel.
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



# %% [markdown]
# ### §11.1 Radio SPL/DPL parity

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
print(f"§11 radio parity (0.1-300 GHz): SPL max |ratio − 1| = {_spl_dmax:.2e}   DPL max |ratio − 1| = {_dpl_dmax:.2e}")
fig.tight_layout()
save_fig("agnfitter_11c_radio_spl_residual.png")
plt.show()

# %% [markdown]
# ### §11.2 SPL α × log ν_cut, DPL log ν_t grid
#
# SPL `alpha_agn=−α` swept over $\alpha \in \{-0.5, -0.75, -1.0\}$ × `log_nu_cut`
# $\in \{12, 13, 14\}$; DPL `log_nu_t` $\in \{9.5, 10, 10.5\}$ at fixed
# `alpha1=−0.75, alpha2=−0.1, log_nu_cut=13`, each against
# `agn_radio_spl`/`agn_radio_dpl`, normalized at 5 GHz over 0.1–300 GHz.
# Table only; §11.1 covers the default nodes.

# %%
print("§11.2  SPL alpha x log_nu_cut grid (0.1-300 GHz, norm. at 5 GHz):")
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
print("§11.2  DPL log_nu_t grid (0.1-300 GHz, norm. at 5 GHz):")
_dpl_grid_worst = 0.0
for log_nu_t in [9.5, 10.0, 10.5]:
    _, F_af = A.agn_radio_dpl(freq, alpha1=-0.75, alpha2=-0.1, log_nu_t=log_nu_t, log_nu_cut=13.0)
    L_t = np.asarray(radio_agn_dpl(wave_radio, L_AGN_BOL, radio_loudness=1.0, alpha1=-0.75, alpha2=-0.1, log_nu_t=log_nu_t, log_nu_cut=13.0))
    t_n = np.asarray(norm_at(np.asarray(wave_radio), L_t, _nu5))
    ratio = np.where(_band & (t_n > 0), t_n / _norm5(freq, F_af), np.nan)
    _mx = float(np.nanmax(np.abs(ratio - 1.0)))
    _dpl_grid_worst = max(_dpl_grid_worst, _mx)
    print(f"    log_nu_t={log_nu_t:g}   max|ratio-1| = {_mx:.2e}")
print(f"§11.2  worst over the full grid: max|ratio-1| = {max(_spl_grid_worst, _dpl_grid_worst):.2e}")

# %% [markdown]
# ## §11b Star-formation radio: bell2003_split parity, then tengri's default
#
# AGNfitter-rX's `S17_radio` joins the Schreiber et al. (2018) dust SED to a
# radio tail calibrated with the infrared–radio correlation of Bell (2003), split
# 90% non-thermal and 10% thermal at 1.4 GHz (paper pp. 3–4). The correlation
# parameter is
#
# $$q_{\rm IR} = \log_{10}\!\left[\frac{L_{\rm IR}}{(3.75\times10^{12}\,{\rm Hz})\,L_{\nu,1.4\,{\rm GHz}}}\right],$$
#
# with $L_{\rm IR}$ the 8–1000 µm luminosity (paper Eq. 1). The paper adopts
# $q_{\rm IR} = 2.64 \pm 0.26$ from Bell (2003) and then takes the conservative
# value $2.64 + \sigma$ (p. 3), which is what the repackaged template embeds; the
# cell measures it from the template. tengri's `radio_sfr_bell2003_split` is the
# matching mode and is run at that value. tengri's *default* architecture
# (`radio_sfr_bell2003` plus a separately normalized `radio_freefree`) uses a
# different convention for what $q_{\rm IR}$ calibrates and is not compared here,
# since mixing the two would double-count the thermal term.
#
# **Caveat:** at tengri's default $q_{\rm IR} = 2.64$ instead of the template's
# value, the 1.4 GHz luminosity of the same $L_{\rm IR}$ is higher by the factor
# printed below.

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
    f"§11b  q_IR measured from the S17_radio template (T_dust = 35 K) = {Q_IR_TEMPLATE:.4f};  "
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
    f"§11b  SF radio, tengri / AGNfitter-rX (both normalized to their FIR peak) over 1.1-30 GHz: "
    f"median ratio = {np.median(_ratio_rb):.4f}, max|ratio-1| = {np.max(np.abs(_ratio_rb - 1.0)):.4f}"
)
print(
    f"§11b  Caveat: q_IR = {_Q_BELL} instead of {Q_IR_PARITY:.3f} raises the 1.4 GHz luminosity "
    f"of the same L_IR by a factor {_radio_ratio_1p4ghz:.3f}"
)

# %% [markdown]
# ## §12 AGNfitter-rX informative priors
#
# `tengri.agn.priors.agnfitter_priors` adapts the eight AGNfitter-rX priors
# (`PRIORS_AGNfitter.py`) onto a public `model.predict(params)` prediction:
# `pred.sed.components["sed_agn_torus"]` and `["sed_agn_disc"]` give the
# per-sub-block torus and disc flux the priors need. `Fitter(model, ...,
# extra_log_prior=callable(params, state))` reaches every inference backend
# (MAP/VI/MCMC) with the same callable.
#
# All eight priors are evaluated on the AGN+galaxy build below, each fed a
# "data" stand-in derived from the model's own prediction (a self-consistency
# demonstration of the API, not an observation). `prior_energy_balance`
# compares the galaxy-attenuated (dust-absorbed) luminosity with the cold-dust
# re-emitted luminosity; at `dust_eta_balance`'s default (`Fixed(1.0)`, strict
# balance) the two match by construction, so it contributes a non-rejecting
# log-density here. `prior_stellar_mass` is a Gaussian on upstream's galaxy
# amplitude `GA`; the cell converts tengri's present stellar mass to `GA` with
# the driver's exact relation (§3), so the prior acts on the same physical
# quantity in both codes.
#
# **Caveat:** two upstream priors have no faithful tengri counterpart, by
# design. Upstream's `maximal_age` evaluates the observing-redshift terms
# outside its integrand, so it imposes no real age limit; and its energy-balance
# prior compares quantities in different units (the galaxy term carries the
# $10^{18}$ template factor, the dust term does not), which loosens the
# "flexible" floor by orders of magnitude. tengri implements the physical
# versions, so agreement with upstream's values is not expected.

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


print(f"§12  AGNfitter-rX default flags (energy_balance flexible + agn_fraction): total = {float(total_default):.3f}")
print("     | prior            | log-prior   |")
for k, v in breakdown_default.items():
    print(f"     | {k:16s} | {_fmt_prior(v):>11s} |")
print(
    f"\n§12  all eight priors, all finite: total = {float(total_all):.3f}  "
    f"(GA = {_ga13:.3f} from M*_present = {float(pred13.sfh.stellar_mass_surviving):.3e} Msun at z = {_z13})"
)
print("     | prior            | log-prior   |")
for k, v in breakdown_all.items():
    print(f"     | {k:16s} | {_fmt_prior(v):>11s} |")

# %% [markdown]
# ### §12.1 Attaching the priors to a fit
#
# `Fitter(model, ..., extra_log_prior=callable)` reaches MAP/VI/MCMC. The
# hook uses `uv_xrays` alone, deliberately fed a `log_l2kev_data` 3 dex
# brighter (in the implied disc L₂₅₀₀) than this build's own truth point —
# a stand-in for a real external X-ray measurement that disagrees with the
# photometry-only fit, which is exactly when an informative prior earns its
# keep. Two more choices make the pull visible rather than swamped: the mock
# photometry drops the 1500 Å anchor band and widens to 30% noise (3 bands
# total), and `enable_agn_fraction`/`enable_energy_balance` — both `True` by
# the adapter's own default unless stated otherwise — are set `False`
# explicitly, since `energy_balance`'s hard floor is a step function with
# zero gradient once rejected and contributes nothing to steer ADAM.

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
    f"§12.1  MAP agn_log_lbol: hook off = {_lbol_off:.3f}, hook on (uv_xrays, "
    f"+3 dex L_2500 mismatch) = {_lbol_on:.3f}  (shift = {_lbol_on - _lbol_off:+.3f} dex, "
    f"truth = {float(_p0['agn_log_lbol']):.3f}) -- the mismatched X-ray-implied UV "
    "luminosity pulls the MAP fit toward a brighter disc, exactly as intended."
)

# %% [markdown]
# ## Capstone — the paper's winning model on its host, radio to X-ray
#
# One `SEDModel.build` with the fiducial host (stellar population, S17 cold dust,
# star-formation radio in the parity mode of §11b) and the paper's winning AGN
# combination (THB21-like disc, CAT3D-Wind torus, α_ox corona, DPL radio jet) is
# compared with AGNfitter-rX's `ymodel` (`PARAMETERSPACE_AGNfitter.py`), which sums
#
# $$L_\nu = 10^{SB}\,T_{\rm SB} + 10^{BB}\,T_{\rm BBB} + 10^{GA}\,T_{\rm GA}
# + 10^{TO}\,T_{\rm TO} + 10^{RAD}\,T_{\rm RAD}.$$
#
# Every AGNfitter-rX curve is upstream's own template or equation read through
# the driver, never a tengri output. The amplitudes $10^N$ are free parameters of
# upstream's fit, so each is set from a matched input: GA from tengri's present
# stellar mass (§3), with no free scale; SB so its 8–1000 µm power equals
# tengri's realized $L_{\rm TIR}$; BB so $L_\nu(2500\,{\rm Å})$ equals tengri's
# disc plus lines; TO so its integral equals tengri's torus power; RAD so the DPL
# equals tengri's jet at 5 GHz. The test is therefore absolute scale for GA and
# shape for the rest. The BBB term is upstream's disc cut at 200 eV plus its
# $\alpha_{\rm ox}$ power law, with a hard step and no EUV bridge, which is how
# the fit sees it.

# %%
_CAT3D_NODE = dict(incl=0.0, a=-2.0, fwd=1.75)
_LOG_LBOL_CAP = 11.0
_AGN_RADIO = builders.radio.agn.dpl(other_params=Fixed(DEFAULT))


def _capstone_build(radio):
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


m_cap = _capstone_build({"radio_q_ir": Fixed(Q_IR_PARITY), "sf": {"type": "bell2003_split"}, "agn": _AGN_RADIO})
m_cap_jet = _capstone_build({"sf": {"type": "none"}, "agn": _AGN_RADIO})  # the AGN jet alone
resolved_params(m_cap)
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
L_d0 = L_d0 * _L2500_te / _val_at(w_d0, L_d0, 2500.0)  # BB amplitude: L_nu(2500 A); alpha_ox follows from it
w_bb_c, L_bb_c = A.bbb_with_xrays(w_d0, L_d0, ebv=0.0, scatter=0.0)
w_to_c, L_to_c = A.torus_template("CAT3D", **_CAT3D_NODE)
L_to_c = L_to_c * band_power(w_te, te_to, 1e3, 1e9) / band_power(w_to_c, L_to_c, 1e3, 1e9)
_nu_r = 10.0 ** np.arange(7.0, 15.0, 0.02)  # upstream's own radio frequency grid
_, F_r = A.agn_radio_dpl(_nu_r, **_DPL_PARS)
w_r_c = U.C_ANGSTROM_PER_S / _nu_r[::-1]
L_r_c = F_r[::-1] * _val_at(w_te, te_rad, U.C_ANGSTROM_PER_S / 5e9) / float(np.interp(5e9, _nu_r, F_r))

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

_L2500_af = float(_val_at(w_bb_c, L_bb_c, 2500.0))
_l2kev_af = float(_val_at(w_bb_c, L_bb_c, 6.199))
print(
    f"Capstone  anchors: L_nu(2500 A) = {_L2500_te:.3e} erg/s/Hz (tengri, used for BB);  "
    f"alpha_ox upstream = {A.alpha_ox(_L2500_te):.3f}, "
    f"tengri corona: {-0.3838 * np.log10(_L2500_te / float(_val_at(w_te, _comp['sed_xray'], 6.199))):.3f}"
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
# **Caveat:** the EUV and soft-X-ray band (shaded) is a hole in both codes' simple
# corona, by different construction: upstream's power law starts above 200 eV
# with a hard step from the disc, and tengri's `yang20` corona carries no soft
# excess (that physics lives in the `kubota_done` disc's own warm
# Comptonization, not the $\alpha_{\rm ox}$ corona used here). Residuals in
# that band, and in the soft X-ray where tengri's default absorbing column acts
# (§10), are construction differences. The torus row inherits the sampling of the
# model wavelength grid discussed in §9c, and the `ymodel` sum has no
# nebular or X-ray-binary term.

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
    ("SN12 + KD18 disc nodes", "§9a.2", len(_win_9a2), max(r["max_abs_dev"] for r in _win_9a2), "max|t/a-1|"),
    ("S04 + NK08 torus nodes", "§9c.2", len(_win_9c0), max(r["max_abs_dev"] for r in _win_9c0), "max|t/a-1|"),
    ("SKIRTOR (oa, incl, tau) nodes, averaged", "§9c.3", len(_win_9c5) - 1,
     max(r["max_abs_dev"] for r in _win_9c5 if "full grid" not in r["label"]), "max|t/a-1|"),
    ("CAT3D union: wind-fraction sweep", "§9c.5", len(_union_rows), max(r["max_abs_dev"] for r in _union_rows), "max|t/a-1|"),
    ("CAT3D (incl, a, f_wd) nodes", "§9c.5", len(_tri_rows), max(r["max_abs_dev"] for r in _tri_rows), "max|t/a-1|"),
    ("Cold dust: S17 nodes + DH02 ladder", "§6", len(_s17_resid) + len(_dh_resids),
     max([r[2] for r in _s17_resid] + _dh_resids), "median|dex|"),
    ("X-ray corona, default N_H (0.5-100 keV)", "§10", len(_xray_rows[20.0]), max(r["max_abs_dev"] for r in _xray_rows[20.0]), "max|t/a-1|"),
    ("X-ray corona, N_H switched off", "§10", len(_xray_rows[0.0]), max(r["max_abs_dev"] for r in _xray_rows[0.0]), "max|t/a-1|"),
    ("Radio SPL alpha x nu_cut, DPL nu_t", "§11.2", 12, max(_spl_grid_worst, _dpl_grid_worst), "max|t/a-1|"),
]
print("Summary: per-block worst tengri / AGNfitter-rX deviation")
print("    | block                                     | §     | cases | worst      | metric      |")
for _blk, _sec, _n, _w, _met in _SUMMARY:
    print(f"    | {_blk:41s} | {_sec:5s} | {_n:5d} | {_w:10.4g} | {_met:11s} |")

# %% [markdown]
# ## Summary
#
# The per-block worst deviations are in the table printed above; the entries
# below say what matched and what did not, section by section.
#
# - **§1–§3** Stellar populations: the matched declining-exponential node agrees
#   in shape; the absolute scale needs the mass mapping of §3 (upstream unit
#   template is a present mass, tengri's a mass formed), and the printed
#   wavelength ladder shows where the two BC03 editions part in the infrared.
#   BC03 metallicity nodes are tabulated in §1.
# - **§4–§4b** Disc reddening: identical law once the declared and effective
#   $R_V$ of the upstream routine are reconciled; qsogen's own quasar curve
#   differs in shape and V-band scale.
# - **§5** Galaxy attenuation: the calibrated range agrees with the analytic
#   curve; upstream's own tails (double $R_V$ below 0.12 µm, negative $k$ in the
#   infrared) are tabulated, not reproduced.
# - **§6–§7** Cold dust (S17, DH02_CE01) and the host composite: template shapes
#   agree with the radio-extended model grid; sub-band powers in the infrared
#   inherit the sparse model grid.
# - **§8** No host nebular emission or IGM term upstream.
# - **§9a–§9b** Four disc libraries and tengri's grid-tabulated KD18 discs at
#   nodes; the Hα bump is reproduced in position, with a line-strength offset
#   tabulated against the THB21 template; disc reddening sweep.
# - **§9c** Torus libraries and reductions: the printed verdict lines state which
#   nodes meet the tolerance, the remainder traces to the model wavelength grid;
#   the CAT3D library is compared as the union of both tengri blocks; the
#   inclination-dependent torus power differs by design.
# - **§10** X-ray corona: power law, cutoff and 2 keV anchor agree; tengri's
#   default absorbing column lowers the soft band.
# - **§11–§11b** SPL/DPL radio agree; the star-formation radio parity mode
#   reproduces the template's embedded $q_{\rm IR}$.
# - **§12** All eight AGNfitter-rX priors are evaluated through
#   `agnfitter_priors`, and attached to a MAP fit through `extra_log_prior`.
# - **Capstone** One buildable, fittable model over $8 < \log\nu/{\rm Hz} < 20$
#   against upstream's `ymodel` sum of all five components, with a
#   fractional-residual panel and a 16–84% optical normalization spread.

# %% [markdown]
# ## References
#
# Every model compared above, with the section that uses it. The machine-
# readable BibTeX lives next to this notebook in `references.bib`; the key of
# each entry is given in brackets.
#
# **Accretion disks (§9a)**
# - Richards, G. T., et al. 2006, ApJS 166, 470 — R06 [`richards2006sed`].
# - Slone, O. & Netzer, H. 2012, MNRAS 426, 656 — SN12 [`slone2012effects`].
# - Kubota, A. & Done, C. 2018, MNRAS 480, 1247 — KD18 [`kubota2018physical`].
# - Temple, M. J., Hewett, P. C. & Banerji, M. 2021, MNRAS 508, 737 — THB21;
#   qsogen [`temple2021modelling`].
#
# **Tori (§9c)**
# - Silva, L., et al. 2004, MNRAS 355, 973 — S04 [`Silva2004`].
# - Nenkova, M., et al. 2008, ApJ 685, 160 — NK08 [`nenkova2008agnII`].
# - Stalevski, M., et al. 2016, MNRAS 458, 2288 — SKIRTOR [`Stalevski2016`].
# - Hönig, S. F. & Kishimoto, M. 2017, ApJL 838, L20 — CAT3D-Wind [`honig2017dusty`].
# - Yang, G., et al. 2020, MNRAS 491, 740 — X-CIGALE SKIRTOR [`yang2020xcigale`].
#
# **Cold dust (§6–§7)**
# - Schreiber, C., et al. 2018, A&A 609, A30 — S17 [`schreiber2018dust`].
# - Dale, D. A. & Helou, G. 2002, ApJ 576, 159 [`dale2002infrared`]; Chary, R. &
#   Elbaz, D. 2001, ApJ 556, 562 [`chary2001interpreting`] — DH02_CE01.
# - Dale, D. A., et al. 2014, ApJ 784, 83 — tengri `dale2014` [`dale2014two`].
# - Calzetti, D., et al. 2000, ApJ 533, 682 — galaxy attenuation [`calzetti2000dust`].
#
# **X-ray (§10)**
# - Just, D. W., et al. 2007, ApJ 665, 1004 [`just2007x`]; Lusso, E. &
#   Risaliti, G. 2016, ApJ 819, 154 [`lusso2016tight`]; 2017, A&A 602, A79
#   [`lusso2017quasars`] — α_ox–L₂₅₀₀.
# - Stern, D. 2015, ApJ 807, 129 — 6 µm ↔ 2–10 keV relation behind the
#   AGNfitter-rX X-ray prior [`stern2015`].
# - Yang, G., et al. 2022, ApJ 927, 192 — X-ray viewing-angle anisotropy
#   [`yang2022cigale`].
# - Mineo, S., et al. 2014, MNRAS 437, 1698 — host XRB / SFR [`mineo2014x`].
#
# **Radio (§11)**
# - Azadi, M., et al. 2020 (arXiv:2011.03130) — radio AGN/SF separation
#   [`azadi2020disentangling`].
# - Bell, E. F. 2003, ApJ 586, 794 — IR–radio correlation [`bell2003estimating`].
# - Murphy, E. J., et al. 2011, ApJ 737, 67 — SFR-L_IR calibration [`murphy2011calibrating`].
#
# **Stellar populations & attenuation (§1–§5)**
# - Bruzual, G. & Charlot, S. 2003, MNRAS 344, 1000 [`bruzual2003stellar`];
#   Chabrier, G. 2003, PASP 115, 763 [`chabrier2003galactic`].
# - Prevot, M. L., et al. 1984, A&A 132, 389 — SMC reddening [`prevot1984typical`].
#
# **Codes, priors & inference (§12)**
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
