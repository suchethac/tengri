# SPDX-License-Identifier: BSD-3-Clause
r"""Thin wrappers around Synthesizer's forward model for the notebook.

Synthesizer (Lovell et al. 2025; Roper et al. 2026 — cite both) builds galaxy SEDs by
extracting from pre-computed HDF5 grids and walking an ``EmissionModel`` tree.
This module drives that machinery one physics block at a time — single stellar
populations, a parametric delayed-:math:`\tau` SFH, the Calzetti / power-law
attenuation curves, the Draine & Li (2007) dust grid, the nebular grid, the
``UnifiedAGN`` black-hole emission tree, and the Madau / Inoue IGM — and
returns everything in tengri's convention (``L_nu`` in erg/s/Hz on a rest-frame
Angstrom grid).

The heavy objects here are the HDF5 grids (the stellar test grid is ~200 MB).
They are loaded once and cached at module level; lightweight ``Stars`` /
``BlackHole`` components are rebuilt per call so sections never leak state.

Grids resolve from ``$SYNTHESIZER_GRID_DIR`` if set, else Synthesizer's default
application-support directory. Fetch them once with the ``synthesizer-download``
CLI (flags ``--stellar-test-grids``, ``--agn-test-grids``, ``--dust-grid``).

References
----------
.. [1] Synthesizer (cite BOTH papers):
       Lovell, C.C., et al. (2025), Open J. Astrophys. 8, doi:10.33232/001c.145766;
       Roper, W.J., et al. (2026), JOSS 11, 9436, doi:10.21105/joss.09436.
.. [2] Nenkova, M., et al. (2008). ApJ, 685, 160 — clumpy torus.
.. [3] Kubota, A., Done, C. (2018). MNRAS, 480, 1247 — qsosed disc.
"""

from __future__ import annotations

import os
from functools import cache, lru_cache
from typing import Any

import numpy as np

from . import units as U

# Grid names shipped by ``synthesizer-download``.
_STELLAR_GRID = "test_grid"
_NLR_GRID = "test_grid_agn-nlr"
_BLR_GRID = "test_grid_agn-blr"
_DUST_GRID = "draine_li_dust_emission_grid_MW_3p1"


def synthesizer_version() -> str:
    """Installed cosmos-synthesizer version, for the grid provenance line.

    The AGN test grids are downloaded, not committed, and they move between
    releases: §9c's Synthesizer [OIII]5007/Hbeta read 20.9 against the grid
    current at the audit and 15.6 under 1.2.0, with the tengri side unchanged
    at 12.8. Printing the version is what separates a grid revision from a
    physics disagreement.

    Note the distribution is ``cosmos-synthesizer``; the ``synthesizer`` name
    on PyPI is an unrelated audio package that shadows it if installed.
    """
    from importlib.metadata import version

    return version("cosmos-synthesizer")


def _grid_dir() -> str:
    """Return the grid directory (env override or Synthesizer's default)."""
    env = os.environ.get("SYNTHESIZER_GRID_DIR")
    if env:
        return env
    # Synthesizer's platform default (macOS application-support path).
    return os.path.expanduser("~/Library/Application Support/Synthesizer/grids")


@cache
def _load_grid(name: str, **kwargs: Any):
    """Load and cache a Synthesizer :class:`Grid` by name."""
    from synthesizer import Grid

    return Grid(name, grid_dir=_grid_dir(), **kwargs)


# ---------------------------------------------------------------------------
# §1 — single stellar populations
# ---------------------------------------------------------------------------
def ssp_grid_axes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(wave_aa, ages_yr, metallicities)`` of the stellar grid."""
    g = _load_grid(_STELLAR_GRID)
    return (
        np.asarray(g.lam.to("angstrom").value, dtype=np.float64),
        np.asarray(g.ages.to("yr").value, dtype=np.float64),
        np.asarray(g.metallicities, dtype=np.float64),
    )


def ssp_spectrum(
    *, metallicity: float = 0.02, age_gyr: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return one Synthesizer single-SSP ``incident`` spectrum.

    The ``incident`` grid spectrum is the pure stellar continuum (no nebular
    reprocessing), normalized to 1 :math:`M_\odot` of stars formed — the direct
    analog of an FSPS/BC03 SSP.

    Parameters
    ----------
    metallicity : float
        Absolute metallicity :math:`Z` (grid is in absolute :math:`Z`, not
        :math:`Z/Z_\odot`). Snapped to the nearest grid node.
    age_gyr : float
        SSP age [Gyr]. Snapped to the nearest grid node.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Spectral luminosity [erg/s/Hz] per :math:`M_\odot` formed.
    """
    g = _load_grid(_STELLAR_GRID)
    wave_aa, ages_yr, mets = ssp_grid_axes()
    ia = int(np.argmin(np.abs(ages_yr - age_gyr * 1e9)))
    iz = int(np.argmin(np.abs(mets - metallicity)))
    L_nu = np.asarray(g.spectra["incident"][ia, iz, :], dtype=np.float64)
    return wave_aa, L_nu


# ---------------------------------------------------------------------------
# §2 — star formation history (delayed-τ, the FSPS sfh=4 analog)
# ---------------------------------------------------------------------------
def sfh_curve(
    *, tau_gyr: float = 1.0, max_age_gyr: float = 5.0, ngrid: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return Synthesizer's ``DelayedExponential`` :math:`\mathrm{SFR}(t_{\rm look})`, ∫ = 1 M⊙.

    Parameters
    ----------
    tau_gyr : float
        e-folding timescale [Gyr].
    max_age_gyr : float
        Age of the population at observation [Gyr].
    ngrid : int
        Number of lookback-time samples.

    Returns
    -------
    t_lookback_yr : ndarray, shape (ngrid,)
        Lookback time [yr], 0 = observation epoch.
    sfr : ndarray, shape (ngrid,)
        Star formation rate [M⊙/yr], normalized to 1 M⊙ formed.
    """
    # Synthesizer's ``SFH.DelayedExponential`` is the delayed-exponential form
    # SFR(t) ∝ t·exp(−t/τ) in time-since-onset t. We evaluate that closed form
    # directly (it *is* Synthesizer's parametrization) and normalize to 1 M⊙
    # formed, matching tengri's ``log_total_mass = 0``.
    age_since_start = np.linspace(0.0, max_age_gyr, ngrid)  # Gyr, 0 = onset
    shape = age_since_start * np.exp(-age_since_start / tau_gyr)
    mass = np.trapezoid(shape, age_since_start * 1e9)  # ∫ over yr
    sfr = shape / mass  # M⊙/yr per M⊙ formed
    t_lookback_yr = (max_age_gyr - age_since_start) * 1e9
    order = np.argsort(t_lookback_yr)
    return t_lookback_yr[order], sfr[order]


# ---------------------------------------------------------------------------
# §3, §5, §7 — composite stellar populations
# ---------------------------------------------------------------------------
def _build_stars(*, tau_gyr: float, max_age_gyr: float, metallicity: float, log_mass: float):
    """Build a parametric ``Stars`` with a delayed-τ SFH at a delta metallicity."""
    from synthesizer.parametric import SFH, Stars, ZDist
    from unyt import Gyr, Msun

    g = _load_grid(_STELLAR_GRID)
    sfh = SFH.DelayedExponential(tau=tau_gyr * Gyr, max_age=max_age_gyr * Gyr)
    zdist = ZDist.DeltaConstant(metallicity=metallicity)
    return Stars(
        g.log10ages,
        g.metallicities,
        sf_hist=sfh,
        metal_dist=zdist,
        initial_mass=(10.0**log_mass) * Msun,
    )


def stellar_sed(
    *,
    tau_gyr: float = 1.0,
    max_age_gyr: float = 5.0,
    metallicity: float = 0.02,
    log_mass: float = 10.0,
    nebular: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Integrated stellar :math:`L_\nu` for the delayed-τ SFH (no dust).

    Parameters
    ----------
    tau_gyr, max_age_gyr : float
        Delayed-τ e-folding time and observation age [Gyr].
    metallicity : float
        Absolute metallicity :math:`Z`.
    log_mass : float
        :math:`\log_{10}(M_\star/M_\odot)` formed.
    nebular : bool
        If True, return the reprocessed (stellar + nebular) spectrum; else the
        pure ``incident`` stellar continuum.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Spectral luminosity [erg/s/Hz].
    """
    from synthesizer.emission_models import IncidentEmission, ReprocessedEmission

    g = _load_grid(_STELLAR_GRID)
    stars = _build_stars(
        tau_gyr=tau_gyr, max_age_gyr=max_age_gyr, metallicity=metallicity, log_mass=log_mass
    )
    model = ReprocessedEmission(grid=g) if nebular else IncidentEmission(grid=g)
    stars.get_spectra(model)
    sed = stars.spectra[model.label]
    return U.sed_to_lnu(sed)


# ---------------------------------------------------------------------------
# §4 — dust attenuation curves
# ---------------------------------------------------------------------------
def attenuation_curve(
    name: str = "calzetti", *, wave_aa: np.ndarray | None = None, **kwargs: Any
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return :math:`A_\lambda` [mag] for a named Synthesizer attenuation law.

    Synthesizer's ``AttenuationLaw`` exposes ``get_tau(lam)``, the optical depth
    at :math:`\tau_V = 1`; :math:`A_\lambda = 1.086\,\tau(\lambda)`.

    Parameters
    ----------
    name : {"calzetti", "power_law", "calzetti_bump"}
        Attenuation law. ``calzetti_bump`` adds the 2175 Å Drude feature via
        ``Calzetti2000(ampl=...)``.
    wave_aa : array_like, optional
        Wavelength grid [Å]. Defaults to 1000–25000 Å.
    **kwargs
        Extra law parameters (e.g. ``slope`` for the power law, ``ampl`` for the
        bump amplitude).

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Wavelength [Å].
    A_lambda : ndarray, shape (n_wave,)
        Attenuation [mag] at :math:`\tau_V = 1`.
    """
    from synthesizer.emission_models import attenuation as att
    from unyt import angstrom

    if wave_aa is None:
        wave_aa = np.logspace(np.log10(1000.0), np.log10(25000.0), 2000)
    wave_aa = np.asarray(wave_aa, dtype=np.float64)

    if name == "calzetti":
        curve = att.Calzetti2000(**kwargs)
    elif name == "calzetti_bump":
        curve = att.Calzetti2000(ampl=kwargs.pop("ampl", 1.0), **kwargs)
    elif name in ("power_law", "powerlaw"):
        curve = att.PowerLaw(slope=kwargs.pop("slope", -0.7))
    else:
        curve = getattr(att, name)(**kwargs)

    tau = np.asarray(curve.get_tau(wave_aa * angstrom), dtype=np.float64)
    return wave_aa, 1.086 * tau


def attenuate(
    wave_aa: np.ndarray,
    L_nu: np.ndarray,
    *,
    name: str = "calzetti",
    av: float = 1.0,
    **kwargs: Any,
) -> np.ndarray:
    r"""Apply a screen attenuation law at :math:`A_V` to an :math:`L_\nu` array."""
    from synthesizer.emission_models import attenuation as att
    from unyt import angstrom

    curve = att.Calzetti2000(**kwargs) if name == "calzetti" else getattr(att, name)(**kwargs)
    T = np.asarray(curve.get_transmission(av / 1.086, np.asarray(wave_aa) * angstrom))
    return np.asarray(L_nu) * T


# ---------------------------------------------------------------------------
# §6, §7 — energy-balanced total emission (attenuation + Draine & Li 2007 IR)
# ---------------------------------------------------------------------------
def total_emission(
    *,
    tau_gyr: float = 1.0,
    max_age_gyr: float = 5.0,
    metallicity: float = 0.02,
    log_mass: float = 10.0,
    av: float = 1.0,
    qpah: float = 0.025,
    umin: float = 1.0,
    alpha: float = 2.0,
    gamma: float = 0.05,
    fesc: float = 0.0,
    components: tuple[str, ...] = ("incident", "attenuated", "dust_emission", "total"),
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    r"""Energy-balanced panchromatic SED via Synthesizer's ``TotalEmission`` tree.

    A Calzetti screen attenuates the reprocessed stellar+nebular spectrum and the
    Draine & Li (2007) templates re-emit the absorbed luminosity in the IR with
    energy balance enforced internally (``set_energy_balance``). Returns a dict of
    ``{label: (wave_aa, L_nu)}`` for the requested tree components (``incident``,
    ``attenuated``, ``dust_emission``, ``total``).

    Parameters
    ----------
    tau_gyr, max_age_gyr, metallicity, log_mass : float
        Delayed-τ SFH, metallicity, and formed mass of the galaxy.
    av : float
        Diffuse :math:`V`-band attenuation [mag] (Calzetti screen, ``tau_v = A_V/1.086``).
    qpah, umin, alpha, gamma : float
        Draine & Li parameters: PAH mass fraction, minimum radiation-field
        intensity, the :math:`dU \propto U^{-\alpha}` power-law index, and the PDR
        mass fraction. (Note Synthesizer's ``qpah`` is a *fraction* — 0.025 ≈
        tengri's ``qpah = 2.5``.)
    components : tuple of str
        Tree labels to return.

    Returns
    -------
    dict
        ``{label: (wave_aa [Å], L_nu [erg/s/Hz])}``.
    """
    from synthesizer.emission_models import DraineLi07, TotalEmission
    from synthesizer.emission_models.attenuation import Calzetti2000
    from unyt import Msun

    g = _load_grid(_STELLAR_GRID)
    dust_grid = _load_grid(_DUST_GRID)
    stars = _build_stars(
        tau_gyr=tau_gyr, max_age_gyr=max_age_gyr, metallicity=metallicity, log_mass=log_mass
    )
    # DL07 reads dust/hydrogen mass for its template lookup; energy balance then
    # rescales the amplitude to the absorbed luminosity, so the absolute masses
    # here are placeholders.
    stars.dust_mass = 1e7 * Msun
    stars.hydrogen_mass = 1e9 * Msun
    de = DraineLi07(grid=dust_grid, qpah=qpah, umin=umin, alpha=alpha, gamma=gamma)
    model = TotalEmission(
        grid=g, dust_curve=Calzetti2000(), dust_emission_model=de, tau_v=av / 1.086, fesc=fesc
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        stars.get_spectra(model)
    return {lab: U.sed_to_lnu(stars.spectra[lab]) for lab in components if lab in stars.spectra}


# ---------------------------------------------------------------------------
# §8 — nebular emission (Cloudy grid baked into the SSP grid)
# ---------------------------------------------------------------------------
def nebular_sed(
    *, age_gyr: float = 0.01, metallicity: float = 0.02, log_mass: float = 9.0
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return Synthesizer's nebular (lines + continuum) spectrum for a young CSF.

    Uses a constant-SFR population of age ``age_gyr`` so the nebular emission
    dominates, matching the young-burst fiducial used in the other reproduction
    notebooks. Returns the ``nebular`` spectrum from ``NebularEmission``.

    Parameters
    ----------
    age_gyr : float
        Age of the constant-SFR population [Gyr].
    metallicity : float
        Absolute metallicity :math:`Z`.
    log_mass : float
        :math:`\log_{10}(M_\star/M_\odot)` formed.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Nebular spectral luminosity [erg/s/Hz].
    """
    from synthesizer.emission_models import NebularEmission
    from synthesizer.parametric import SFH, Stars, ZDist
    from unyt import Gyr, Msun

    g = _load_grid(_STELLAR_GRID)
    sfh = SFH.Constant(max_age=age_gyr * Gyr, min_age=0.0 * Gyr)
    stars = Stars(
        g.log10ages,
        g.metallicities,
        sf_hist=sfh,
        metal_dist=ZDist.DeltaConstant(metallicity=metallicity),
        initial_mass=(10.0**log_mass) * Msun,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        stars.get_spectra(NebularEmission(grid=g))
    return U.sed_to_lnu(stars.spectra["nebular"])


# ---------------------------------------------------------------------------
# §12 — IGM transmission (Madau 1995 / Inoue 2014)
# ---------------------------------------------------------------------------
def igm_transmission(
    *, redshift: float, model: str = "inoue14", wave_obs_aa: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    r"""Return Synthesizer's IGM transmission :math:`T(\lambda_{\rm obs}, z)`.

    Parameters
    ----------
    redshift : float
        Source redshift (> 0).
    model : {"inoue14", "madau96"}
        IGM prescription.
    wave_obs_aa : array_like, optional
        Observed-frame wavelength grid [Å]. Defaults to 3000–8000 Å.

    Returns
    -------
    wave_obs_aa : ndarray, shape (n_wave,)
        Observed-frame wavelength [Å].
    T : ndarray, shape (n_wave,)
        IGM transmission in [0, 1].
    """
    from synthesizer.emission_models import attenuation as att
    from unyt import angstrom

    if wave_obs_aa is None:
        wave_obs_aa = np.linspace(3000.0, 8000.0, 4000)
    wave_obs_aa = np.asarray(wave_obs_aa, dtype=np.float64)

    igm = att.Inoue14() if model == "inoue14" else att.Madau96()
    T = np.asarray(igm.get_transmission(redshift, wave_obs_aa * angstrom), dtype=np.float64)
    return wave_obs_aa, np.clip(np.nan_to_num(T, nan=1.0), 0.0, 1.0)


# ---------------------------------------------------------------------------
# §9 — Unified AGN (disc + NLR + BLR + torus + inclination anisotropy)
# ---------------------------------------------------------------------------
# The components extracted from the UnifiedAGN tree, by Synthesizer label.
AGN_COMPONENTS = (
    "disc",
    "nlr",
    "blr",
    "torus",
    "line_regions",
    "intrinsic",
    "disc_incident",
    "disc_incident_masked",
    "disc_escaped",
    "disc_transmitted",
    "disc_averaged",
)


@lru_cache(maxsize=8)
def _unified_model(
    *, cf_nlr: float, cf_blr: float, torus_temperature_k: float, disc_transmission: str
):
    """Build and cache a :class:`UnifiedAGN` emission model."""
    from synthesizer.emission_models import Blackbody, UnifiedAGN
    from unyt import K

    nlr = _load_grid(_NLR_GRID)
    blr = _load_grid(_BLR_GRID)
    torus = Blackbody(temperature=torus_temperature_k * K)
    return UnifiedAGN(
        nlr_grid=nlr,
        blr_grid=blr,
        torus_emission_model=torus,
        covering_fraction_nlr=cf_nlr,
        covering_fraction_blr=cf_blr,
        disc_transmission=disc_transmission,
    )


def agn_unified(
    *,
    mass_msun: float = 1e8,
    eddington: float = 0.5,
    inclination_deg: float = 30.0,
    metallicity: float = 0.01,
    cf_nlr: float = 0.1,
    cf_blr: float = 0.1,
    theta_torus_deg: float = 10.0,
    torus_temperature_k: float = 1000.0,
    disc_transmission: str = "weighted_combination",
    components: tuple[str, ...] = AGN_COMPONENTS,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    r"""Generate the Synthesizer ``UnifiedAGN`` spectrum and its sub-components.

    The black hole is driven by ``(mass, accretion_rate_eddington)`` — the grid
    axes — with ``inclination`` and ``theta_torus`` setting the disc/torus
    geometry. Returns a dict mapping component label → ``(wave_aa, L_nu)``.

    Parameters
    ----------
    mass_msun : float
        Black-hole mass [:math:`M_\odot`].
    eddington : float
        Accretion rate as a fraction of Eddington.
    inclination_deg : float
        Viewing angle from the disc normal [degrees]; drives the disc/torus
        anisotropy and the torus edge-on mask.
    metallicity : float
        Absolute gas metallicity :math:`Z` for the NLR/BLR grids.
    cf_nlr, cf_blr : float
        NLR / BLR covering fractions.
    theta_torus_deg : float
        Torus half-opening angle [degrees].
    torus_temperature_k : float
        Blackbody torus temperature [K].
    disc_transmission : str
        How the observed disc spectrum is built (Synthesizer
        ``disc_transmission``; default ``"weighted_combination"``).
    components : tuple of str
        Which UnifiedAGN sub-models to return (see :data:`AGN_COMPONENTS`).

    Returns
    -------
    dict
        ``{label: (wave_aa [Å], L_nu [erg/s/Hz])}`` for each requested component.
    """
    from synthesizer.parametric import BlackHole
    from unyt import Msun, degree

    model = _unified_model(
        cf_nlr=cf_nlr,
        cf_blr=cf_blr,
        torus_temperature_k=torus_temperature_k,
        disc_transmission=disc_transmission,
    )
    bh = BlackHole(
        mass=mass_msun * Msun,
        accretion_rate_eddington=eddington,
        inclination=inclination_deg * degree,
        metallicity=metallicity,
        theta_torus=theta_torus_deg * degree,
        covering_fraction_nlr=cf_nlr,
        covering_fraction_blr=cf_blr,
    )
    with np.errstate(over="ignore", invalid="ignore"):
        bh.get_spectra(model)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label in components:
        if label in bh.spectra:
            out[label] = U.sed_to_lnu(bh.spectra[label])
    out["_bolometric_erg_s"] = float(bh.bolometric_luminosity.to("erg/s").value)
    return out


def agn_bolometric_lsun(**kwargs: Any) -> float:
    r"""Return the black hole's bolometric luminosity as :math:`\log_{10}(L/L_\odot)`.

    Convenience for matching tengri's ``agn_log_lbol`` (with the same
    :data:`units.L_SUN_ERG_PER_S`).
    """
    res = agn_unified(**kwargs)
    return float(np.log10(res["_bolometric_erg_s"] / U.L_SUN_ERG_PER_S))
