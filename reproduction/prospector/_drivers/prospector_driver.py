"""Thin wrappers around Prospector's forward engine for the notebook.

Prospector (Johnson+2021) forward-models galaxy SEDs by setting FSPS
parameters on a ``prospect.sources.CSPSpecBasis`` and calling FSPS
through ``python-fsps``; its dust attenuation curves come from
``sedpy.attenuation``. ``CSPSpecBasis`` is a thin holder whose ``.ssp``
attribute *is* an :class:`fsps.StellarPopulation` — so driving
``python-fsps`` directly reproduces exactly what Prospector evaluates
during a fit, while staying stable against the 2.x alpha API. This
module sets the FSPS flags one physics block at a time and returns the
results in tengri's unit convention (erg/s/Hz, rest-frame Å).

The heavy ``StellarPopulation`` (~30 s to construct, it reads the full
isochrone + spectral grids off disk) is built once and cached at module
level; each call mutates ``.params`` from a clean baseline so sections
do not leak state into one another.

References
----------
.. [1] Johnson, B.D., Leja, J., Conroy, C., Speagle, J.S. (2021).
       Stellar Population Inference with Prospector. ApJS, 254, 22.
       arXiv:2012.01426.
.. [2] Conroy, C., Gunn, J.E., White, M. (2009). FSPS I. ApJ, 699, 486.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import units as U

# FSPS IMF flag: 1 = Chabrier (2003). The downloadable tengri grids are
# ``*_chabrier``; the FSPS default is 2 (Kroupa), so we pin Chabrier
# explicitly to keep the §1 SSP head-to-head bit-identical.
_IMF_CHABRIER = 1

# Module-level cache for the expensive StellarPopulation.
_SP: Any = None


def _require_sps_home() -> None:
    """Raise a clear error if ``SPS_HOME`` is unset before importing FSPS."""
    if not os.environ.get("SPS_HOME"):
        raise RuntimeError(
            "SPS_HOME is not set. python-fsps needs it to locate the FSPS "
            "isochrone and spectral tables. Set it to your FSPS checkout, e.g.\n"
            "    export SPS_HOME=/path/to/fsps\n"
            "before launching the notebook (see reproduction/prospector/README.md)."
        )


# The comparison is only meaningful if both sides read the same stellar
# library. tengri downloads ``fsps_mist_miles_chabrier``; FSPS builds its own
# from ``$SPS_HOME``, and which library that is was fixed when python-fsps was
# compiled. MILES is a compile-time flag, not a runtime parameter --
# ``src/sps_vars.f90`` ships ``#define C3K_LR 1`` and ``#define MILES 0``.
_REQUIRED_LIBRARY = "miles"


def _require_matching_library(sp) -> None:
    """Fail loudly when FSPS was built against a library the grid does not match.

    python-fsps >= 0.5.0 defaults to C3K low-resolution, so a plain
    ``pip install fsps`` silently produces the wrong comparison: C3K-lr gives
    1936 wavelengths over 13 metallicities against the grid's MILES 5994 over
    12. Nothing raises -- the notebook simply reports a §1 SSP residual of
    ~1e-1 where a matched pair gives ~1e-9, and every downstream section moves
    with it. Rebuild with MILES enabled:

        FFLAGS="-DMILES" pip install --no-binary fsps --force-reinstall fsps
    """
    libs = [x.decode() if isinstance(x, bytes) else str(x) for x in sp.libraries]
    if not any(_REQUIRED_LIBRARY in x.lower() for x in libs):
        raise RuntimeError(
            f"FSPS was built with libraries {libs}, but this comparison needs "
            f"'{_REQUIRED_LIBRARY}'. tengri's downloaded grid is MIST+MILES; a "
            "C3K build compares two different stellar libraries and the §1 "
            "residual degrades from ~1e-9 to ~1e-1 with no error raised. "
            'Rebuild with: FFLAGS="-DMILES" pip install --no-binary fsps '
            "--force-reinstall fsps"
        )


def _get_sp():
    """Return the cached :class:`fsps.StellarPopulation` (lazy, ~30 s once)."""
    global _SP
    if _SP is None:
        _require_sps_home()
        import fsps

        _SP = fsps.StellarPopulation(zcontinuous=1, imf_type=_IMF_CHABRIER)
        _require_matching_library(_SP)
    return _SP


def _reset(sp) -> None:
    """Restore a clean single-stellar baseline: no dust, no neb, no AGN, no IGM."""
    sp.params["imf_type"] = _IMF_CHABRIER
    sp.params["sfh"] = 0
    sp.params["logzsol"] = 0.0
    sp.params["dust_type"] = 2  # Calzetti, only consulted when dust2 > 0
    sp.params["dust1"] = 0.0
    sp.params["dust2"] = 0.0
    sp.params["dust_index"] = 0.0
    sp.params["add_neb_emission"] = False
    sp.params["add_neb_continuum"] = False
    sp.params["add_dust_emission"] = False
    # The Nenkova torus has no on/off flag in FSPS — it is active whenever
    # fagn > 0, so resetting fagn to 0 fully disables it.
    sp.params["fagn"] = 0.0
    sp.params["add_igm_absorption"] = False
    sp.params["zred"] = 0.0


def _spectrum(sp, tage: float) -> tuple[np.ndarray, np.ndarray]:
    """Pull L_ν [erg/s/Hz] off FSPS at ``tage`` (Gyr); 0 → integrated CSP."""
    wave, L_nu_lsun = sp.get_spectrum(tage=tage, peraa=False)
    return np.asarray(wave, dtype=np.float64), U.lnu_lsun_to_erg(L_nu_lsun)


def _apply_sfh(
    sp, *, sfh: int, tau: float, tage: float, logzsol: float, const: float = 0.0
) -> None:
    """Set the SFH block. ``sfh=0`` is a single SSP (``tage`` = SSP age).

    ``const`` is the FSPS fraction of mass formed in a constant-SFR mode;
    ``sfh=1`` with ``const=1.0`` gives a pure constant SFR over
    ``[0, tage]`` — the young-burst fiducial used for the nebular panel.
    """
    sp.params["logzsol"] = logzsol
    sp.params["sfh"] = sfh
    if sfh != 0:
        sp.params["tau"] = tau
        sp.params["const"] = const
        sp.params["sf_start"] = 0.0
        sp.params["sf_trunc"] = 0.0
        sp.params["tburst"] = 0.0
        sp.params["fburst"] = 0.0


# ---------------------------------------------------------------------------
# §1 — single stellar populations
# ---------------------------------------------------------------------------
def ssp_spectrum(*, logzsol: float = 0.0, age_gyr: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Return one FSPS single-SSP spectrum :math:`L_\\nu` [erg/s/Hz].

    Parameters
    ----------
    logzsol : float
        :math:`\\log_{10}(Z/Z_\\odot)`. Default 0 (solar).
    age_gyr : float
        SSP age in Gyr.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Spectral luminosity [erg/s/Hz] per unit mass formed.
    """
    sp = _get_sp()
    _reset(sp)
    _apply_sfh(sp, sfh=0, tau=1.0, tage=age_gyr, logzsol=logzsol)
    return _spectrum(sp, age_gyr)


# ---------------------------------------------------------------------------
# §2 — star formation history (analytic delayed-τ, the FSPS sfh=4 shape)
# ---------------------------------------------------------------------------
def sfh_curve(
    *, tau: float = 1.0, tage: float = 5.0, ngrid: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Return the delayed-τ :math:`\\mathrm{SFR}(t_{\\text{look}})`, ∫ = 1 M⊙.

    FSPS ``sfh=4`` (and Prospector's ``parametric_sfh`` template) use the
    delayed-exponential form :math:`\\mathrm{SFR}(t)\\propto t\\,e^{-t/\\tau}`
    where :math:`t` is time since the onset of star formation. The curve
    is normalized so the mass formed over ``[0, tage]`` is exactly 1 M⊙,
    matching tengri's ``log_total_mass = 0``.

    Parameters
    ----------
    tau : float
        e-folding timescale [Gyr].
    tage : float
        Age of the stellar population at observation [Gyr].
    ngrid : int
        Number of lookback-time samples.

    Returns
    -------
    t_lookback_yr : ndarray, shape (ngrid,)
        Lookback time [yr], 0 = observation epoch.
    sfr : ndarray, shape (ngrid,)
        Star formation rate [M⊙/yr], normalized to 1 M⊙ formed.
    """
    t_since_start = np.linspace(0.0, tage, ngrid)  # Gyr, 0 = formation
    shape = t_since_start * np.exp(-t_since_start / tau)
    mass = np.trapezoid(shape, t_since_start * 1e9)  # ∫ over yr
    sfr = shape / mass  # M⊙/yr per M⊙ formed
    t_lookback_yr = (tage - t_since_start) * 1e9
    order = np.argsort(t_lookback_yr)
    return t_lookback_yr[order], sfr[order]


# ---------------------------------------------------------------------------
# §2a–§2d — non-parametric SFHs (Prospector's binned templates → FSPS sfh=3)
#
# Prospector does not evaluate its non-parametric SFHs inside FSPS. It bins
# the history in ``prospect.models.transforms`` (per-bin stellar masses from
# log-SFR ratios or Dirichlet z-fractions) and feeds the resulting
# step-function SFR to FSPS as a *tabular* SFH (``sfh=3``,
# ``set_tabular_sfh``). The wrappers below call the exact transform Prospector
# uses for each family and return ``(agebins, masses)``; :func:`csp_lnu_binned`
# turns that pair into the FSPS spectrum. So the SED on the left of each §2x
# panel is Prospector's own forward model, evaluated at the parameter values
# the right (tengri) panel uses.
#
# Convention (verified against ``transforms.logsfr_ratios_to_masses``):
# ``logsfr_ratios[j] = log10(SFR_j / SFR_{j+1})`` with ``j = 0`` the most
# recent bin in lookback time — identical sign and ordering to tengri's
# ``ratio_i`` (``components/stellar/sfh/nonparametric.py``).
# ---------------------------------------------------------------------------
def agebins_from_edges(bin_edges_gyr: np.ndarray) -> np.ndarray:
    """Build Prospector ``agebins`` from shared lookback-time bin edges.

    Prospector stores age bins as an ``(nbin, 2)`` array of
    :math:`\\log_{10}` lookback-time edges in **years**, ordered youngest
    first. tengri expresses the same grid as ``bin_edges_gyr`` (lookback
    [Gyr], ascending). Passing both sides the same physical edges makes the
    binning identical; the only adjustment is that a ``0 Gyr`` youngest edge
    has no :math:`\\log_{10}(\\mathrm{yr})` value, so it is floored to 1 Myr.

    Parameters
    ----------
    bin_edges_gyr : array_like, shape (nbin+1,)
        Lookback-time bin edges [Gyr], ascending (youngest first).

    Returns
    -------
    agebins : ndarray, shape (nbin, 2)
        ``[log10(t_lower_yr), log10(t_upper_yr)]`` per bin, youngest first.
    """
    edges_yr = np.asarray(bin_edges_gyr, dtype=np.float64) * 1e9
    edges_yr = np.maximum(edges_yr, 1e6)  # floor a 0-Gyr youngest edge at 1 Myr
    log_edges = np.log10(edges_yr)
    return np.stack([log_edges[:-1], log_edges[1:]], axis=1)


def csp_lnu_binned(
    *,
    agebins: np.ndarray,
    masses: np.ndarray,
    logzsol: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """FSPS spectrum for a binned (step-function) SFH via the ``sfh=3`` path.

    This is the non-parametric analog of :func:`csp_lnu`. Per-bin stellar
    masses are converted to a piecewise-constant SFR, handed to FSPS through
    :meth:`fsps.StellarPopulation.set_tabular_sfh`, and the integrated
    composite spectrum is read at the epoch of observation.

    Parameters
    ----------
    agebins : array_like, shape (nbin, 2)
        :math:`\\log_{10}` lookback-time bin edges [yr], youngest first — the
        Prospector convention (see :func:`agebins_from_edges`).
    masses : array_like, shape (nbin,)
        Stellar mass formed in each bin [M⊙], aligned with ``agebins``.
    logzsol : float
        Stellar metallicity :math:`\\log_{10}(Z/Z_\\odot)`. Default 0 (solar).

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Spectral luminosity [erg/s/Hz] for the supplied formed mass (already
        absolute — *not* per 1 M⊙, since ``masses`` carries the normalization).

    Notes
    -----
    FSPS' tabular SFH is cosmic time (Gyr from the start of star formation,
    increasing); the bins are reversed from lookback to cosmic time and a
    duplicated point at each internal edge renders the SFR as a true step.
    """
    agebins = np.asarray(agebins, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    edges_yr = np.concatenate([10.0 ** agebins[:, 0], [10.0 ** agebins[-1, 1]]])
    widths_yr = np.diff(edges_yr)
    sfr_bins = masses / widths_yr  # M⊙/yr, youngest-first

    # Lookback edges → cosmic age (Gyr, increasing): oldest edge is t=0.
    tage_gyr = edges_yr[-1] / 1e9
    lookback_gyr = edges_yr / 1e9
    cosmic_edges = (tage_gyr - lookback_gyr)[::-1]  # ascending, [0 .. tage]
    sfr_cosmic = sfr_bins[::-1]  # SFR per bin, oldest-first to match cosmic_edges

    # Render the step function: two samples per internal edge (ε apart).
    ages, sfrs = [], []
    eps = 1e-6  # Gyr
    for i in range(sfr_cosmic.shape[0]):
        lo, hi = cosmic_edges[i], cosmic_edges[i + 1]
        ages += [lo, max(hi - eps, lo + eps / 2)]
        sfrs += [sfr_cosmic[i], sfr_cosmic[i]]
    age_tab = np.asarray(ages, dtype=np.float64)
    sfr_tab = np.asarray(sfrs, dtype=np.float64)

    sp = _get_sp()
    _reset(sp)
    sp.params["logzsol"] = logzsol
    sp.params["sfh"] = 3
    sp.set_tabular_sfh(age_tab, sfr_tab)
    return _spectrum(sp, tage_gyr)


def continuity_masses(
    *,
    bin_edges_gyr: np.ndarray,
    log_total_mass: float,
    logsfr_ratios: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin masses for the continuity SFH (Leja+2019), fixed bin edges.

    Wraps :func:`prospect.models.transforms.logsfr_ratios_to_masses` — the
    exact transform Prospector applies for ``continuity_sfh``.

    Parameters
    ----------
    bin_edges_gyr : array_like, shape (nbin+1,)
        Shared lookback-time bin edges [Gyr], ascending.
    log_total_mass : float
        :math:`\\log_{10}(M_\\star/M_\\odot)` formed.
    logsfr_ratios : array_like, shape (nbin-1,)
        ``log10(SFR_j / SFR_{j+1})``, youngest first.

    Returns
    -------
    agebins : ndarray, shape (nbin, 2)
        Prospector age bins [log10 yr].
    masses : ndarray, shape (nbin,)
        Stellar mass formed per bin [M⊙].
    """
    from prospect.models import transforms as _T

    agebins = agebins_from_edges(bin_edges_gyr)
    masses = _T.logsfr_ratios_to_masses(
        logmass=float(log_total_mass),
        logsfr_ratios=np.asarray(logsfr_ratios, dtype=np.float64),
        agebins=agebins,
    )
    return agebins, np.asarray(masses, dtype=np.float64)


def dirichlet_masses(
    *,
    bin_edges_gyr: np.ndarray,
    log_total_mass: float,
    z_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin masses for the Dirichlet SFH (Leja+2017), fixed bin edges.

    Wraps :func:`prospect.models.transforms.zfrac_to_masses`.

    Parameters
    ----------
    bin_edges_gyr : array_like, shape (nbin+1,)
        Shared lookback-time bin edges [Gyr], ascending.
    log_total_mass : float
        :math:`\\log_{10}(M_\\star/M_\\odot)` formed.
    z_fraction : array_like, shape (nbin-1,)
        Stick-breaking z-fractions, youngest first.

    Returns
    -------
    agebins : ndarray, shape (nbin, 2)
        Prospector age bins [log10 yr].
    masses : ndarray, shape (nbin,)
        Stellar mass formed per bin [M⊙].
    """
    from prospect.models import transforms as _T

    agebins = agebins_from_edges(bin_edges_gyr)
    masses = _T.zfrac_to_masses(
        total_mass=10.0 ** float(log_total_mass),
        z_fraction=np.asarray(z_fraction, dtype=np.float64),
        agebins=agebins,
    )
    return agebins, np.asarray(masses, dtype=np.float64)


def flex_masses(
    *,
    anchor_edges_gyr: np.ndarray,
    log_total_mass: float,
    logsfr_ratio_young: float,
    logsfr_ratios: np.ndarray,
    logsfr_ratio_old: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin masses + derived bins for continuity_flex (Leja+2019).

    Wraps :func:`prospect.models.transforms.logsfr_ratios_to_masses_flex` and
    :func:`prospect.models.transforms.logsfr_ratios_to_agebins`. Unlike the
    fixed-bin families, the *flex* bin edges are derived from the ratios under
    a constant-mass-per-flex-bin constraint, so this returns the derived
    ``agebins`` too.

    Parameters
    ----------
    anchor_edges_gyr : array_like, shape (3,)
        ``[t_young_end, t_old_start, t_max]`` [Gyr] — the fixed young/old
        anchor edges (tengri's ``bin_edges_gyr`` for ``continuity_flex``).
    log_total_mass : float
        :math:`\\log_{10}(M_\\star/M_\\odot)` formed.
    logsfr_ratio_young : float
        ``log10(SFR_young / SFR_flex[0])``.
    logsfr_ratios : array_like, shape (n_flex-1,)
        Inner flex log-SFR ratios controlling the flex bin widths.
    logsfr_ratio_old : float
        ``log10(SFR_old / SFR_flex[N])``.

    Returns
    -------
    agebins : ndarray, shape (nbin, 2)
        Derived Prospector age bins [log10 yr].
    masses : ndarray, shape (nbin,)
        Stellar mass formed per bin [M⊙].
    """
    from prospect.models import transforms as _T

    a = np.asarray(anchor_edges_gyr, dtype=np.float64)
    # Outer anchor bins: youngest [1 Myr, t_young_end], oldest [t_old_start, t_max].
    outer = np.array(
        [
            [np.log10(1e6), np.log10(a[0] * 1e9)],
            [np.log10(a[1] * 1e9), np.log10(a[2] * 1e9)],
        ]
    )
    derived = _T.logsfr_ratios_to_agebins(
        logsfr_ratios=np.asarray(logsfr_ratios, dtype=np.float64), agebins=outer
    )
    masses = _T.logsfr_ratios_to_masses_flex(
        logmass=float(log_total_mass),
        logsfr_ratios=np.asarray(logsfr_ratios, dtype=np.float64),
        logsfr_ratio_young=np.atleast_1d(np.float64(logsfr_ratio_young)),
        logsfr_ratio_old=np.atleast_1d(np.float64(logsfr_ratio_old)),
        agebins=outer,
    )
    return np.asarray(derived, dtype=np.float64), np.asarray(masses, dtype=np.float64)


# ---------------------------------------------------------------------------
# §3, §5, §6, §7 — composite stellar populations with optional physics
# ---------------------------------------------------------------------------
def csp_lnu(
    *,
    logzsol: float = 0.0,
    tau: float = 1.0,
    tage: float = 5.0,
    sfh: int = 4,
    const: float = 0.0,
    av: float = 0.0,
    dust_type: int = 2,
    dust_index: float = 0.0,
    add_dust_emission: bool = False,
    duste_qpah: float = 2.5,
    duste_umin: float = 1.0,
    duste_gamma: float = 0.05,
    add_neb_emission: bool = False,
    gas_logu: float = -2.0,
    gas_logz: float = 0.0,
    add_agn_dust: bool = False,
    fagn: float = 0.0,
    agn_tau: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a composite-SFH FSPS spectrum with any subset of physics on.

    This is the workhorse for the dust, nebular, AGN, and panchromatic
    panels. Every block defaults off; switch on only what a section needs.

    Parameters
    ----------
    logzsol, tau, tage, sfh : float, float, float, int
        Stellar metallicity [dex], e-folding time [Gyr], age [Gyr], and
        FSPS SFH flag (4 = delayed-τ).
    av : float
        Diffuse :math:`V`-band attenuation [mag]; mapped to FSPS
        ``dust2 = Av / 1.086``.
    dust_type : int
        FSPS dust law (2 = Calzetti, 4 = Kriek & Conroy, 0 = power law).
    dust_index : float
        Power-law / Kriek-Conroy slope modifier.
    add_dust_emission : bool
        Attach the Draine & Li (2007) IR templates.
    duste_qpah, duste_umin, duste_gamma : float
        DL07 parameters: PAH mass fraction [%], minimum radiation-field
        intensity, and fraction of dust at :math:`U > U_{\\min}`.
    add_neb_emission : bool
        Attach the Byler (2017) nebular line + continuum grid.
    gas_logu, gas_logz : float
        Nebular ionization parameter :math:`\\log U` and gas-phase
        :math:`\\log_{10}(Z/Z_\\odot)`.
    add_agn_dust : bool
        Attach the Nenkova (2008) AGN torus.
    fagn : float
        AGN luminosity as a fraction of the stellar bolometric luminosity.
    agn_tau : float
        Nenkova torus optical depth.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Spectral luminosity [erg/s/Hz] per unit mass formed.
    """
    sp = _get_sp()
    _reset(sp)
    _apply_sfh(sp, sfh=sfh, tau=tau, tage=tage, logzsol=logzsol, const=const)
    sp.params["dust_type"] = dust_type
    sp.params["dust2"] = av / 1.086
    sp.params["dust_index"] = dust_index
    if add_dust_emission:
        sp.params["add_dust_emission"] = True
        sp.params["duste_qpah"] = duste_qpah
        sp.params["duste_umin"] = duste_umin
        sp.params["duste_gamma"] = duste_gamma
    if add_neb_emission:
        sp.params["add_neb_emission"] = True
        sp.params["add_neb_continuum"] = True
        sp.params["gas_logu"] = gas_logu
        sp.params["gas_logz"] = gas_logz
    if add_agn_dust:
        # FSPS has no add_agn_dust flag; the Nenkova torus is on when fagn > 0.
        sp.params["fagn"] = fagn
        sp.params["agn_tau"] = agn_tau
    return _spectrum(sp, tage)


def isolate(on: dict[str, Any], off: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the additive contribution of a block: spectrum(on) − spectrum(off).

    Used to pull the nebular-only or torus-only SED out of FSPS, which
    only exposes the summed spectrum.

    Parameters
    ----------
    on, off : dict
        Keyword dicts for :func:`csp_lnu` with and without the block.

    Returns
    -------
    wave_aa : ndarray
        Rest-frame wavelength [Å].
    L_nu_block : ndarray
        The isolated block's :math:`L_\\nu` [erg/s/Hz].
    """
    w_on, L_on = csp_lnu(**on)
    w_off, L_off = csp_lnu(**off)
    assert np.array_equal(w_on, w_off), "wave grid drifted between on/off runs"
    return w_on, L_on - L_off


# ---------------------------------------------------------------------------
# §4 — dust attenuation curves (sedpy.attenuation, Prospector's source)
# ---------------------------------------------------------------------------
def attenuation_curve(
    name: str,
    *,
    wave_aa: np.ndarray | None = None,
    av: float = 1.0,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Return :math:`A_\\lambda` [mag] for a named ``sedpy`` attenuation law.

    Prospector imports its dust laws from ``sedpy.attenuation``. Each
    function returns the optical depth :math:`\\tau(\\lambda)` for a given
    :math:`\\tau_V`; we convert to magnitudes via
    :math:`A_\\lambda = 1.086\\,\\tau(\\lambda)`.

    Parameters
    ----------
    name : {"calzetti", "conroy", "cardelli", "smc", "powerlaw"}
        ``sedpy.attenuation`` function name. ``conroy`` is the
        Kriek & Conroy (2013) law.
    wave_aa : array_like, optional
        Wavelength grid [Å]. Defaults to 1000–25000 Å.
    av : float
        :math:`V`-band attenuation [mag] setting :math:`\\tau_V = A_V/1.086`.
    **kwargs
        Extra law parameters passed through (e.g. ``R_v``, ``dust_index``).

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Wavelength [Å].
    A_lambda : ndarray, shape (n_wave,)
        Attenuation [mag].
    """
    from sedpy import attenuation as _att

    if wave_aa is None:
        wave_aa = np.logspace(np.log10(1000.0), np.log10(25000.0), 2000)
    wave_aa = np.asarray(wave_aa, dtype=np.float64)
    fn = getattr(_att, name)
    tau = fn(wave_aa, tau_v=av / 1.086, **kwargs)
    return wave_aa, 1.086 * np.asarray(tau, dtype=np.float64)


def fsps_kriek_conroy_curve(wave_aa: np.ndarray, *, dust_index: float = 0.0) -> np.ndarray:
    """Return FSPS' ``dust_type=4`` Kriek & Conroy curve, k(λ) at tau_V=1.

    This is the law Prospector actually applies (FSPS ``attn_curve.f90``,
    ``dust_type=4``), not ``sedpy``'s ``conroy``. The two are different
    KC13 implementations — FSPS ties the 2175 Å bump amplitude to the
    slope via KC13 Eqn 3 and divides the Drude by :math:`R_V = 4.05`::

        eb = 0.85 - 1.9 * dust_index
        drude = eb * (lam * dlam) ** 2 / ((lam**2 - lamuvb**2) ** 2 + (lam * dlam) ** 2)
        k = (cal00 + drude / R_V) * (lam / lamv) ** dust_index

    Parameters
    ----------
    wave_aa : array_like, shape (n_wave,)
        Wavelength grid [Å].
    dust_index : float
        Slope modifier :math:`\\delta` (FSPS ``dust_index``). Default 0.

    Returns
    -------
    k : ndarray, shape (n_wave,)
        Attenuation curve at :math:`\\tau_V = 1` (FSPS normalization, not
        renormalized to k(5500)=1 — divide by the V-band value for that).
    """
    wl = np.asarray(wave_aa, dtype=np.float64)
    x = 1e4 / wl  # 1/micron
    below = wl <= 6300.0
    cal = np.where(
        below,
        1.17 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3) + 1.78,
        1.17 * (-1.857 + 1.04 * x) + 1.78,
    )
    cal = np.clip(cal, 0.0, None) / 0.44 / 4.05
    eb = 0.85 - 1.9 * dust_index  # KC13 Eqn 3
    dlam, lamuvb = 350.0, 2175.0
    drude = eb * (wl * dlam) ** 2 / ((wl**2 - lamuvb**2) ** 2 + (wl * dlam) ** 2)
    return (cal + drude / 4.05) * (wl / 5500.0) ** dust_index


# ---------------------------------------------------------------------------
# §12 — IGM transmission (FSPS add_igm_absorption, Madau 1995)
# ---------------------------------------------------------------------------
def igm_transmission(
    *, zred: float, logzsol: float = 0.0, age_gyr: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Return FSPS' Madau (1995) IGM transmission :math:`T(\\lambda_{\\rm rest})`.

    FSPS applies the IGM as a multiplicative factor on the redshifted
    spectrum. We recover the transmission by dividing the spectrum built
    with ``add_igm_absorption`` on by the one with it off, at the same
    ``zred`` — leaving a clean :math:`T \\in [0, 1]` on the rest-frame grid.

    Parameters
    ----------
    zred : float
        Source redshift (> 0).
    logzsol : float
        Stellar metallicity [dex] of the young SSP used as the backlight.
    age_gyr : float
        Age [Gyr] of the backlight SSP (young → bright in the UV).

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    T : ndarray, shape (n_wave,)
        IGM transmission in [0, 1].
    """
    sp = _get_sp()
    _reset(sp)
    _apply_sfh(sp, sfh=0, tau=1.0, tage=age_gyr, logzsol=logzsol)
    sp.params["zred"] = zred
    sp.params["add_igm_absorption"] = False
    wave, L_clear = sp.get_spectrum(tage=age_gyr, peraa=False)
    sp.params["add_igm_absorption"] = True
    sp.params["igm_factor"] = 1.0
    _, L_igm = sp.get_spectrum(tage=age_gyr, peraa=False)
    with np.errstate(divide="ignore", invalid="ignore"):
        T = np.where(L_clear > 0, L_igm / L_clear, 1.0)
    T = np.clip(np.nan_to_num(T, nan=1.0), 0.0, 1.0)
    return np.asarray(wave, dtype=np.float64), T
