"""Thin wrappers around bagpipes for reproduction-notebook use.

Bagpipes' public API is ``bagpipes.model_galaxy(components, …)`` — a
single call instantiates the SFH, applies dust attenuation, lights up
nebular emission, attaches the energy-balanced dust IR, and convolves
through the IGM. This module exposes the resulting outputs in tengri's
unit convention (erg/s/Hz on an Angstrom rest-frame grid) and pulls out
the per-component curves the notebook compares panel by panel.

References
----------
.. [1] Carnall, A.C., et al. (2018). Inferring the star formation
       histories of massive quiescent galaxies with BAGPIPES.
       MNRAS, 480, 4379. arXiv:1712.04452.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Any

import numpy as np

from . import units as U


def bagpipes_version() -> str:
    """Installed BAGPIPES version, for the SSP-grid provenance line.

    The repackaged grid is gitignored (``*.h5``), so it is rebuilt from
    whichever BAGPIPES is installed and the §13 magnitudes move with it.
    Printing the version is what lets a reader tell a template-version
    difference from a physics one.
    """
    from importlib.metadata import version

    return version("bagpipes")


def _model_galaxy_class():
    """Lazy import so ``import reproduction.bagpipes._drivers.*`` doesn't
    pay the (~3-second) bagpipes startup cost when only ``units`` or the
    SSP repackaging is needed."""
    from bagpipes.models.model_galaxy import model_galaxy

    return model_galaxy


def _build_model(components: dict[str, Any], *, spec_wavs: np.ndarray | None = None):
    """Instantiate ``bagpipes.model_galaxy`` with z=0 unless overridden.

    A bare ``model_galaxy(components)`` call with ``redshift=0`` returns
    rest-frame :math:`L_\\lambda` in erg/s/Å — the units the rest of this
    driver expects. Setting an alternative redshift moves us into
    observed-frame and applies luminosity-distance dimming; callers that
    want that behavior should pass it explicitly via ``components``.
    """
    components = dict(components)
    components.setdefault("redshift", 0.0)
    with warnings.catch_warnings():
        # The Cloudy v25 grid emits a benign warning about line wavelengths
        # for ``redshift=0`` photometry; the spectrum_full path is
        # unaffected.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mg = _model_galaxy_class()(components, spec_wavs=spec_wavs)
    return mg


def to_lnu(mg) -> tuple[np.ndarray, np.ndarray]:
    """Convert a ``model_galaxy`` full spectrum to :math:`L_\\nu` [erg/s/Hz].

    Bagpipes builds ``spectrum_full`` in erg/s/Å (z=0) on its
    ``wavelengths`` grid (Å). The grid is rest-frame regardless of the
    redshift in the components dict, but the *units* of ``spectrum_full``
    change when ``redshift > 0`` (they pick up cm² from
    :math:`4\\pi D_L^2`). This driver is intended for z=0 only — see
    ``_build_model`` for the constraint.

    Parameters
    ----------
    mg : bagpipes.models.model_galaxy.model_galaxy
        Built via :func:`_build_model`.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength in Angstroms.
    L_nu : ndarray, shape (n_wave,)
        Spectral luminosity :math:`L_\\nu` in erg/s/Hz, integrated over
        the components attached to ``mg``. Carries an implicit factor
        :math:`10^{\\mathrm{massformed}}` from the components dict.
    """
    if mg.model_comp.get("redshift", 0.0) != 0.0:
        raise ValueError(
            "to_lnu assumes redshift=0 so spectrum_full is in erg/s/Å; got redshift={}".format(
                mg.model_comp["redshift"]
            )
        )
    wave_aa = np.asarray(mg.wavelengths, dtype=np.float64)
    L_lambda = np.asarray(mg.spectrum_full, dtype=np.float64)
    _, L_nu = U.ergs_per_aa_to_erg_per_hz(wave_aa, L_lambda)
    return wave_aa, L_nu


def stellar_only_lnu(
    *,
    massformed: float = 10.0,
    metallicity: float = 1.0,
    age_max: float = 5.0,
    age_min: float = 0.0,
    tau: float | None = None,
    sfh_type: str = "delayed",
) -> tuple[np.ndarray, np.ndarray]:
    """Return a purely stellar :math:`L_\\nu` spectrum (no dust, no neb).

    The ``constant`` and ``delayed`` Bagpipes SFH modules both accept
    ``massformed`` (log10 :math:`M_\\odot`), ``metallicity`` (Z/Zsun),
    and an age grid. We disable dust and nebular by simply omitting
    them from the components dict.

    Parameters
    ----------
    massformed : float
        :math:`\\log_{10}(M_{\\text{formed}}/M_\\odot)`. Default 10.
    metallicity : float
        Stellar metallicity in solar units. Default 1.
    age_max, age_min : float
        Bounds of the constant-SFR window in Gyr (for ``sfh_type="constant"``).
    tau : float, optional
        e-folding timescale in Gyr (for ``sfh_type="delayed"``).
    sfh_type : {"constant", "delayed"}
        SFH form.

    Returns
    -------
    wave_aa : ndarray, shape (n_wave,)
        Rest-frame wavelength [Å].
    L_nu : ndarray, shape (n_wave,)
        Stellar :math:`L_\\nu` [erg/s/Hz], no dust attenuation, no
        nebular emission.
    """
    if sfh_type == "constant":
        sfh_block = {
            "metallicity": metallicity,
            "age_min": age_min,
            "age_max": age_max,
            "massformed": massformed,
        }
        comp = {"redshift": 0.0, "constant": sfh_block}
    elif sfh_type == "delayed":
        if tau is None:
            tau = 1.0
        sfh_block = {
            "metallicity": metallicity,
            "age": age_max,
            "tau": tau,
            "massformed": massformed,
        }
        comp = {"redshift": 0.0, "delayed": sfh_block}
    else:
        raise ValueError(f"unknown sfh_type={sfh_type!r}")

    mg = _build_model(comp)
    return to_lnu(mg)


def attenuated_lnu(
    *,
    dust_block: dict[str, Any],
    nebular_block: dict[str, Any] | None = None,
    sfh_type: str = "delayed",
    massformed: float = 10.0,
    metallicity: float = 1.0,
    age: float = 5.0,
    tau: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a dust attenuation block (plus optional nebular) to a fiducial SFH.

    Mirrors the ``attenuation_curve`` semantics of the CIGALE driver but
    returns the full attenuated SED rather than just the A_λ curve, since
    bagpipes' dust modules are not separable from the SFH.
    """
    comp: dict[str, Any] = {
        "redshift": 0.0,
        "dust": deepcopy(dust_block),
    }
    if sfh_type == "delayed":
        comp["delayed"] = {
            "metallicity": metallicity,
            "age": age,
            "tau": tau,
            "massformed": massformed,
        }
    elif sfh_type == "constant":
        comp["constant"] = {
            "metallicity": metallicity,
            "age_min": 0.0,
            "age_max": age,
            "massformed": massformed,
        }
    else:
        raise ValueError(f"unknown sfh_type={sfh_type!r}")
    if nebular_block is not None:
        comp["nebular"] = deepcopy(nebular_block)
    mg = _build_model(comp)
    return to_lnu(mg)


def attenuation_curve(
    dust_block: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract :math:`A_\\lambda` [mag] from a bagpipes dust block.

    Builds two stellar-only spectra at the same SFH (1 Gyr delayed,
    :math:`Z=Z_\\odot`, :math:`\\log M=10`) — one with the requested
    dust block, one bare — and computes
    :math:`A_\\lambda = -2.5 \\log_{10}(L_{\\text{att}} / L_{\\text{int}})`.
    Wavelengths with :math:`L_{\\text{int}} = 0` are masked to zero.

    Parameters
    ----------
    dust_block : dict
        The ``dust`` entry passed to ``model_galaxy``. Must contain at
        least ``type`` and ``Av``; other keys (``eta``, ``n``, …) pass
        through.

    Returns
    -------
    wave_aa : ndarray
        Rest-frame wavelength [Å].
    A_lambda_mag : ndarray
        Attenuation magnitudes.
    """
    wave_int, L_int = attenuated_lnu(dust_block={"type": "Calzetti", "Av": 0.0})
    wave_att, L_att = attenuated_lnu(dust_block=dust_block)
    assert np.array_equal(wave_int, wave_att), "wave grid drifted"
    with np.errstate(divide="ignore", invalid="ignore"):
        A_lambda = -2.5 * np.log10(L_att / L_int)
    A_lambda = np.nan_to_num(A_lambda, nan=0.0, posinf=0.0, neginf=0.0)
    return wave_int, A_lambda


def sfh_curve(
    *,
    sfh_type: str = "delayed",
    age: float = 5.0,
    tau: float = 1.0,
    age_min: float = 0.0,
    age_max: float = 1.0,
    massformed: float = 10.0,
    metallicity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract :math:`\\mathrm{SFR}(t_{\\text{look}})` from a bagpipes SFH.

    Bagpipes' ``model_galaxy.sfh.sfh`` is in :math:`M_\\odot/\\text{yr}`
    on the ``model_galaxy.sfh.ages`` lookback-time grid (yr).
    """
    if sfh_type == "delayed":
        comp = {
            "redshift": 0.0,
            "delayed": {
                "metallicity": metallicity,
                "age": age,
                "tau": tau,
                "massformed": massformed,
            },
        }
    elif sfh_type == "constant":
        comp = {
            "redshift": 0.0,
            "constant": {
                "metallicity": metallicity,
                "age_min": age_min,
                "age_max": age_max,
                "massformed": massformed,
            },
        }
    else:
        raise ValueError(f"unknown sfh_type={sfh_type!r}")

    mg = _build_model(comp)
    lookback_yr = np.asarray(mg.sfh.ages, dtype=np.float64)
    sfr = np.asarray(mg.sfh.sfh, dtype=np.float64)
    return lookback_yr, sfr


def igm_transmission(redshift: float) -> tuple[np.ndarray, np.ndarray]:
    """Extract Bagpipes' Inoue14 IGM transmission curve at a given redshift.

    Reads the precomputed ``d_igm_grid_inoue14.fits`` table that ships
    with bagpipes and evaluates :math:`T(\\lambda_{\\text{rest}}, z)`.

    Parameters
    ----------
    redshift : float
        Source redshift. Must be ≥ 0.

    Returns
    -------
    wave_aa : ndarray
        Rest-frame wavelength [Å].
    T : ndarray
        IGM transmission in [0, 1].
    """
    from bagpipes.models.igm_model import igm

    igm_inst = igm(np.linspace(800.0, 1300.0, 5001))
    T = igm_inst.trans(redshift)
    return igm_inst.wavelengths, T
