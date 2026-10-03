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

import tempfile
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import numpy as np

# NumPy >= 2.0 renamed np.trapz to np.trapezoid; BAGPIPES 1.3.5 still calls the old name.
if not hasattr(np, "trapz"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

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


#: UV-to-NIR rest-frame range [Angstrom] over which every band integral on the
#: page is evaluated; BAGPIPES is sampled at least as finely as
#: :data:`R_COMPARISON_MIN` across it.
COMPARISON_RANGE_AA: tuple[float, float] = (1000.0, 30000.0)

#: Minimum median pixel pitch ``lambda / delta-lambda`` of the BAGPIPES model
#: grid over the comparison range. BAGPIPES' ``R_spec = 1000`` places pixels at
#: ``delta-lambda / lambda = 0.5 / R_spec``, a pitch of 2000, so 1000 is met with
#: a factor of two to spare.
R_COMPARISON_MIN: float = 1000.0

#: ``config.R_spec`` and ``config.R_other`` used for every model this driver
#: builds. BAGPIPES' default ``R_other = 20`` samples the spectrum outside the
#: spectroscopic range in 2.5 % steps; the stellar grid is point-sampled onto it, so
#: a band average over a stellar absorption feature or a nebular line picks up the
#: grid. ``R_other = 100`` (0.5 % steps) keeps the far-IR and EUV smooth enough that
#: a band average of a smooth continuum is converged (shown in the notebook).
R_SPEC_BUILD: float = 1000.0
R_OTHER_BUILD: float = 100.0


def median_pixel_pitch(wave: np.ndarray, lo: float, hi: float) -> float:
    """Median :math:`\\lambda/\\Delta\\lambda` of a wavelength grid inside ``[lo, hi]``.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Ascending wavelength grid [Angstrom].
    lo, hi : float
        Wavelength range [Angstrom] over which the pitch is evaluated.

    Returns
    -------
    float
        Median of :math:`\\lambda_i / (\\lambda_{i+1} - \\lambda_i)` over the
        pixels whose lower edge lies in ``[lo, hi]``.

    Raises
    ------
    ValueError
        If fewer than two grid points lie in the range.
    """
    w = np.asarray(wave, dtype=np.float64)
    inside = (w >= lo) & (w <= hi)
    if inside.sum() < 2:
        raise ValueError(f"fewer than two grid points inside [{lo:g}, {hi:g}] Angstrom")
    wi = w[inside]
    return float(np.median(wi[:-1] / np.diff(wi)))


def assert_sampling(
    wave: np.ndarray, lo: float, hi: float, r_min: float = R_COMPARISON_MIN
) -> float:
    """Raise unless the grid's median pixel pitch over ``[lo, hi]`` is at least ``r_min``.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Ascending wavelength grid [Angstrom].
    lo, hi : float
        Comparison range [Angstrom].
    r_min : float, optional
        Required median :math:`\\lambda/\\Delta\\lambda`.

    Returns
    -------
    float
        The measured median pixel pitch.

    Raises
    ------
    RuntimeError
        If the grid is coarser than ``r_min``, naming the fix.
    """
    pitch = median_pixel_pitch(wave, lo, hi)
    if pitch < r_min:
        raise RuntimeError(
            f"BAGPIPES model grid is too coarse for a band comparison: median "
            f"lambda/dlambda = {pitch:.0f} over {lo:g}-{hi:g} Angstrom, required >= {r_min:g}. "
            f"Build through bagpipes_driver._build_model, which passes spec_wavs and raises "
            f"config.R_spec / config.R_other; a bare model_galaxy(components) call samples at "
            f"R_other = 20 and aliases band integrals."
        )
    return pitch


@contextmanager
def _converged_sampling(scale: float = 1.0) -> Iterator[None]:
    """Set ``bagpipes.config.R_spec`` / ``R_other`` to the build values, then restore them.

    This is the single place the driver touches ``bagpipes.config``. The values are
    read when ``model_galaxy`` is constructed (``_get_wavelength_sampling``), so
    they only need to hold for the duration of the build. ``scale`` multiplies both
    values (used to show that a band ratio is converged).
    """
    import bagpipes.config as cfg

    saved = (cfg.R_spec, cfg.R_other)
    cfg.R_spec = max(cfg.R_spec, scale * R_SPEC_BUILD)
    cfg.R_other = max(cfg.R_other, scale * R_OTHER_BUILD)
    try:
        yield
    finally:
        cfg.R_spec, cfg.R_other = saved


def _build_model(
    components: dict[str, Any],
    *,
    spec_wavs: np.ndarray | None = None,
    resolution_scale: float = 1.0,
):
    """Instantiate ``bagpipes.model_galaxy`` on a converged wavelength grid.

    A bare ``model_galaxy(components)`` call with ``redshift=0`` returns
    rest-frame :math:`L_\\lambda` in erg/s/Å — the units the rest of this
    driver expects — but samples ``spectrum_full`` at ``R_other = 20`` (747
    points over 1 Å to 1e8 Å). The stellar grid is point-sampled onto that grid
    and each emission line lands in one pixel, so any band average of it
    aliases. This builder instead passes ``spec_wavs`` (default
    :data:`COMPARISON_RANGE_AA`), which makes BAGPIPES sample at ``R_spec`` across
    the range (and down to ``spec_wavs[0] / (1 + max_redshift)``), and raises
    ``R_other`` for the remainder (:func:`_converged_sampling`). The returned
    ``wavelengths`` grid is asserted to resolve ``spec_wavs`` at
    :data:`R_COMPARISON_MIN` or better.

    Setting an alternative redshift moves us into observed-frame and applies
    luminosity-distance dimming; callers that want that behavior should pass it
    explicitly via ``components``.

    Parameters
    ----------
    components : dict
        BAGPIPES ``model_components``.
    spec_wavs : array_like, shape (n,), optional
        Wavelengths [Angstrom] of the spectroscopic output; its extent is the
        range the sampling assertion covers. Default :data:`COMPARISON_RANGE_AA`.
    resolution_scale : float, optional
        Multiplies :data:`R_SPEC_BUILD` and :data:`R_OTHER_BUILD` for this build.
        Default 1.

    Returns
    -------
    bagpipes.models.model_galaxy.model_galaxy
        The built model.
    """
    components = dict(components)
    components.setdefault("redshift", 0.0)
    if spec_wavs is None:
        spec_wavs = np.asarray(COMPARISON_RANGE_AA, dtype=np.float64)
    lo, hi = float(spec_wavs[0]), float(spec_wavs[-1])
    with _converged_sampling(resolution_scale), warnings.catch_warnings():
        # The Cloudy v25 grid emits a benign warning about line wavelengths
        # for ``redshift=0`` photometry; the spectrum_full path is
        # unaffected.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mg = _model_galaxy_class()(components, spec_wavs=spec_wavs)
    assert_sampling(mg.wavelengths, lo, hi)
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


def age_of_universe_gyr(mg) -> float:
    """Age of the universe [Gyr] that ``mg``'s star formation history is anchored to.

    BAGPIPES evaluates the cosmic-time SFH forms (``dblplaw``, ``lognormal``) at
    ``age_of_universe - lookback``; a tengri ``dpl`` / ``lnorm`` carrying the same
    shape needs this value as its ``age_gyr``.

    Parameters
    ----------
    mg : bagpipes.models.model_galaxy.model_galaxy
        Built via :func:`_build_model`.

    Returns
    -------
    float
        ``mg.sfh.age_of_universe`` converted from yr to Gyr.
    """
    return float(mg.sfh.age_of_universe) / 1.0e9


def lognormal_width_dex(tmax_gyr: float, fwhm_gyr: float) -> float:
    """Width in dex of the BAGPIPES lognormal SFH defined by ``tmax`` and ``fwhm``.

    BAGPIPES solves ``lognorm_equations`` for the natural-log width
    :math:`\\sigma` and the mode offset :math:`T_0` that give a peak at ``tmax`` and
    a FWHM of ``fwhm`` (the small-width relation :math:`\\sigma \\approx
    \\mathrm{FWHM} / (2 \\sqrt{2 \\ln 2}\\, t_{\\max})` is its starting guess, not
    its answer). tengri's ``lnorm`` takes the width as :math:`\\sigma / \\ln 10`.

    Parameters
    ----------
    tmax_gyr, fwhm_gyr : float
        Peak time and FWHM of the SFH [Gyr].

    Returns
    -------
    float
        :math:`\\sigma / \\ln 10` [dex].
    """
    from bagpipes.models.star_formation_history import lognorm_equations
    from scipy.optimize import fsolve

    tmax, fwhm = tmax_gyr * 1.0e9, fwhm_gyr * 1.0e9
    tau_guess = fwhm / (2.0 * tmax * np.sqrt(2.0 * np.log(2.0)))
    t0_guess = np.log(tmax) + fwhm**2 / (8.0 * np.log(2.0) * tmax**2)
    sigma_ln, _ = fsolve(lognorm_equations, (tau_guess, t0_guess), args=([tmax, fwhm]))
    return float(sigma_ln / np.log(10.0))


def line_table(mg) -> tuple[np.ndarray, np.ndarray]:
    """BAGPIPES's own emission-line table for a built model.

    Parameters
    ----------
    mg : bagpipes.models.model_galaxy.model_galaxy
        Built with a ``nebular`` block via :func:`_build_model`.

    Returns
    -------
    wave_air : ndarray, shape (n_line,)
        Air wavelength of each line [Angstrom], as BAGPIPES labels them.
    lum : ndarray, shape (n_line,)
        Line luminosity [erg/s] after attenuation and escape, from ``mg.line_fluxes``.
    """
    from bagpipes import config

    wave = np.asarray(config.line_wavs, dtype=np.float64)
    lum = np.asarray([mg.line_fluxes[name] for name in config.line_names], dtype=np.float64)
    return wave, lum


def sum_lines(wave: np.ndarray, lum: np.ndarray, centers: tuple[float, ...], tol: float) -> float:
    """Sum the table entries within ``tol`` Angstrom of any of ``centers``.

    Parameters
    ----------
    wave, lum : array_like, shape (n_line,)
        Line wavelengths [Angstrom] and luminosities [erg/s].
    centers : tuple of float
        Wavelengths [Angstrom] (same medium as ``wave``) of the lines to add.
    tol : float
        Match tolerance [Angstrom].

    Returns
    -------
    float
        Summed luminosity [erg/s].
    """
    wave = np.asarray(wave)
    lum = np.asarray(lum)
    return float(sum(lum[np.abs(wave - c) < tol].sum() for c in centers))


def absorbed_luminosity(components: dict[str, Any]) -> float:
    """Luminosity [erg/s] BAGPIPES's energy balance moves from the UV-NIR into the IR.

    BAGPIPES re-emits :math:`\\int (S - S\\,T)\\,d\\lambda` over the whole model grid,
    including wavelengths below 912 Angstrom, where :math:`S` is the unattenuated
    spectrum (stars plus nebular emission if present) and :math:`T =
    10^{-A_V A_{\\rm cont}/2.5}` is the diffuse transmission (``eta = 1`` here, so no
    birth-cloud excess). It is evaluated from the model without the ``dust`` block and
    the transmission of the model with it.

    Parameters
    ----------
    components : dict
        BAGPIPES ``model_components`` including a ``dust`` block with ``Av``.

    Returns
    -------
    float
        Absorbed luminosity [erg/s].
    """
    bare = {k: v for k, v in components.items() if k != "dust"}
    mg_bare = _build_model(bare)
    mg_dust = _build_model(components)
    wave = np.asarray(mg_bare.wavelengths, dtype=np.float64)
    spec = np.asarray(mg_bare.spectrum_full, dtype=np.float64)
    trans = 10.0 ** (-components["dust"]["Av"] * np.asarray(mg_dust.dust_atten.A_cont) / 2.5)
    return float(np.trapezoid(spec - spec * trans, wave))


def igm_transmission_analytic(wave_rest: np.ndarray, redshift: float) -> np.ndarray:
    """Inoue et al. (2014) transmission from BAGPIPES's own table generator.

    BAGPIPES ships ``d_igm_grid_inoue14.fits``, this function sampled on a 1 Angstrom
    rest-frame grid; evaluating it directly removes the tabulation.

    Parameters
    ----------
    wave_rest : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    redshift : float
        Source redshift.

    Returns
    -------
    ndarray, shape (n_wave,)
        Transmission in [0, 1].
    """
    from bagpipes.models.making.igm_inoue2014 import get_Inoue14_trans

    return np.asarray(get_Inoue14_trans(np.asarray(wave_rest, dtype=np.float64), redshift))


def own_photometry(
    components: dict[str, Any],
    filters: list[tuple[np.ndarray, np.ndarray]],
    *,
    converged: bool = False,
) -> np.ndarray:
    """BAGPIPES's own photometry output, ``model_galaxy(components, filt_list=...)``.

    The filter curves are written to temporary two-column files (wavelength
    [Angstrom], transmission), the file format ``filt_list`` reads. Photometry is
    requested with ``phot_units="mujy"``, so each entry is
    :math:`\\langle F_\\lambda\\rangle \\lambda_{\\rm eff}^2 / c` for BAGPIPES's effective
    wavelength, where :math:`\\langle F_\\lambda\\rangle` is the
    :math:`\\lambda`-weighted band average of ``spectrum_full``. At ``redshift = 0``
    ``spectrum_full`` is the luminosity [erg/s/Angstrom], so the entry is
    :math:`\\langle L_\\nu\\rangle \\times 10^{29}` [erg/s/Hz] and a 10 pc flux follows
    by dividing by :math:`4\\pi d^2`; at ``redshift > 0`` it is a flux density in
    microjansky.

    Parameters
    ----------
    components : dict
        BAGPIPES ``model_components`` (``redshift`` included).
    filters : list of (wave, trans)
        Filter curves, wavelength [Angstrom] and transmission, one pair per band.
    converged : bool, optional
        ``False`` (default) uses BAGPIPES's configured ``R_phot = 100`` sampling across
        the filters, as an observer calling the code would. ``True`` raises
        ``R_phot`` to :data:`R_SPEC_BUILD` for the build.

    Returns
    -------
    ndarray, shape (n_filter,)
        BAGPIPES ``photometry`` in microjansky (see above).
    """
    import bagpipes.config as cfg

    model_galaxy = _model_galaxy_class()
    saved = cfg.R_phot
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, (wave, trans) in enumerate(filters):
            path = f"{tmp}/filter_{i}.dat"
            np.savetxt(path, np.column_stack([np.asarray(wave), np.asarray(trans)]))
            paths.append(path)
        if converged:
            cfg.R_phot = max(cfg.R_phot, R_SPEC_BUILD)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mg = model_galaxy(dict(components), filt_list=paths, phot_units="mujy")
        finally:
            cfg.R_phot = saved
    return np.asarray(mg.photometry, dtype=np.float64)
