# SPDX-License-Identifier: BSD-3-Clause
r"""Cloudy / photoionization NLR adapters for the AGN unified model.

The Cloudy-style backends in :mod:`tengri.components.nebular.agn_nebular`
predict **discrete line luminosities** from an AGN ionizing source. The
``unified_nlr_blr`` AGN model however expects a callable that returns a
continuous ``L_nu`` spectrum on the user's wavelength grid. This module
provides thin adapter functions that bridge the two interfaces:

1. ``compute_nlr_sed_feltre`` — Feltre, Charlot & Gutkin 2016 grid
   (CLOUDY c13.03; BEAGLE parity).
2. ``compute_nlr_sed_synthesizer`` — Synthesizer CLOUDY c23.01 AGN NLR
   grid (lazy — requires the user-supplied HDF5 path).

Both adapters use the canonical Gaussian-convolution path from the
analytic ``nlr.py`` so the resulting continuous spectrum matches the
existing line-broadening convention (``fwhm_kms`` ≈ 500 km/s for NLR).

Plug into :func:`tengri.components.agn.unified.unified_nlr_blr` via the
``nlr_fn`` keyword::

    from tengri.components.agn import unified_nlr_blr
    from tengri.components.agn.nlr_cloudy import compute_nlr_sed_feltre

    sed = unified_nlr_blr(
        wavelength,
        agn_log_lbol=12.0,
        nlr_fn=compute_nlr_sed_feltre,  # use Feltre+2016 NLR
        # additional kwargs forwarded to the adapter
        alpha_pl=-1.7,
        neb_logU=-2.0,
        neb_logn=3.0,
        neb_logZ_gas=-1.8477,
        xi_d=0.3,
    )

Closes the user-facing half of #332 (the existing backends were not
exposed at the AGN component level).

References
----------
.. [1] A. Feltre, S. Charlot, J. Gutkin, "Nuclear activity in galaxies:
   the effective slope of the ionizing spectrum," MNRAS 456, 3354 (2016).
   https://doi.org/10.1093/mnras/stv2794
.. [2] C. C. Lovell et al., "Synthesizer," MNRAS submitted (2025),
   arXiv:2004.07283.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import vmap

from tengri.components.agn._phys import gaussian_line_profile as _gaussian_line_profile
from tengri.components.nebular.agn_nebular import (
    FeltreNLRBackend,
    SynthesizerBLRBackend,
    SynthesizerNLRBackend,
    _log_qh_from_lacc,
)
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG_S

__all__ = [
    "compute_blr_sed_synthesizer",
    "compute_nlr_sed_feltre",
    "compute_nlr_sed_synthesizer",
    "get_feltre_backend",
    "get_synthesizer_blr_backend",
    "get_synthesizer_nlr_backend",
]


_FELTRE_BACKEND: FeltreNLRBackend | None = None
_SYNTHESIZER_BACKEND: SynthesizerNLRBackend | None = None
_SYNTHESIZER_BLR_BACKEND: SynthesizerBLRBackend | None = None


def get_feltre_backend(grid_path: str | None = None) -> FeltreNLRBackend:
    """Lazy singleton accessor for :class:`FeltreNLRBackend`.

    Subsequent calls return the same instance unless a fresh ``grid_path``
    is supplied. Loading the HDF5 grid is the slow step (~1 s) and we
    cache it for the duration of the Python process.

    Parameters
    ----------
    grid_path : str or None, optional
        Path to ``feltre_grid.h5``. If ``None``, uses the package default
        from :data:`tengri.components.nebular.agn_nebular._DEFAULT_FELTRE_GRID_PATH`.

    Returns
    -------
    FeltreNLRBackend
        Initialised backend, ready for ``predict_agn_nlr_lines`` calls.

    Raises
    ------
    FileNotFoundError
        If the grid file does not exist at the resolved path.
    """
    global _FELTRE_BACKEND
    if grid_path is not None or _FELTRE_BACKEND is None:
        _FELTRE_BACKEND = FeltreNLRBackend() if grid_path is None else FeltreNLRBackend(grid_path)
    return _FELTRE_BACKEND


def get_synthesizer_nlr_backend(grid_path: str) -> SynthesizerNLRBackend:
    """Lazy singleton accessor for :class:`SynthesizerNLRBackend`.

    Synthesizer AGN NLR grids are not packaged with tengri (they are
    generated via the `grid-generation` companion repo and can be many
    GB), so a path must be supplied explicitly on first call. Subsequent
    calls reuse the cached instance unless a different ``grid_path`` is
    provided.

    Parameters
    ----------
    grid_path : str
        Path to a Synthesizer CLOUDY AGN NLR HDF5 grid.

    Returns
    -------
    SynthesizerNLRBackend
        Initialised backend.
    """
    global _SYNTHESIZER_BACKEND
    if _SYNTHESIZER_BACKEND is None or _SYNTHESIZER_BACKEND.grid_path != grid_path:
        _SYNTHESIZER_BACKEND = SynthesizerNLRBackend(grid_path)
    return _SYNTHESIZER_BACKEND


def get_synthesizer_blr_backend(grid_path: str) -> SynthesizerBLRBackend:
    """Lazy singleton accessor for :class:`SynthesizerBLRBackend`.

    Mirrors :func:`get_synthesizer_nlr_backend` for the broad-line-region grid.
    A path must be supplied on first call (Synthesizer AGN grids are not packaged
    with tengri); subsequent calls reuse the cached instance unless a different
    ``grid_path`` is given.

    Parameters
    ----------
    grid_path : str
        Path to a Synthesizer CLOUDY AGN BLR HDF5 grid.

    Returns
    -------
    SynthesizerBLRBackend
        Initialised backend.
    """
    global _SYNTHESIZER_BLR_BACKEND
    if _SYNTHESIZER_BLR_BACKEND is None or _SYNTHESIZER_BLR_BACKEND.grid_path != grid_path:
        _SYNTHESIZER_BLR_BACKEND = SynthesizerBLRBackend(grid_path)
    return _SYNTHESIZER_BLR_BACKEND


def _lines_to_lnu(
    wavelength: jnp.ndarray,
    line_wavelengths: jnp.ndarray,
    line_luminosities_erg_s: jnp.ndarray,
    fwhm_kms: float,
) -> jnp.ndarray:
    r"""Convolve discrete line luminosities into a continuous :math:`L_\nu`.

    Each line contributes ``line_lum × gaussian_line_profile(wave, λ_c,
    fwhm_kms)``, where the Gaussian is per-Hz-normalised so the spectral
    integral over frequency returns ``line_lum`` in erg/s.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    line_wavelengths : array, shape (n_lines,)
        Line rest wavelengths [Å].
    line_luminosities_erg_s : array, shape (n_lines,)
        Line luminosities [erg/s] (already in CGS — convert from L_sun
        upstream if needed).
    fwhm_kms : float
        Velocity FWHM applied to each line [km/s].

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density :math:`L_\nu` [erg/s/Hz].
    """

    def _single_line(line: jnp.ndarray) -> jnp.ndarray:
        wave_c, lum = line[0], line[1]
        profile = _gaussian_line_profile(wavelength, wave_c, fwhm_kms)
        return lum * profile

    stacked = jnp.stack([line_wavelengths, line_luminosities_erg_s], axis=1)
    return vmap(_single_line)(stacked).sum(axis=0)


def compute_nlr_sed_feltre(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = 500.0,
    alpha_pl: float = -1.7,
    neb_logU: float = -2.0,
    neb_logn: float = 3.0,
    neb_logZ_gas: float = -1.8477,
    xi_d: float = 0.3,
    grid_path: str | None = None,
    **_kwargs,
) -> jnp.ndarray:
    r"""Feltre+2016 CLOUDY-grid NLR adapter for ``unified_nlr_blr``.

    Routes the requested wavelength grid through the Feltre+2016
    photoionization model and convolves the predicted lines with a
    Gaussian profile at ``fwhm_kms``. Output is :math:`L_\nu` in erg/s/Hz.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    l_disc_bol_erg : float
        AGN disc bolometric luminosity [erg/s]. The Feltre grid is
        driven by Q_H, derived from ``covering_fraction × l_disc_bol_erg``
        via the ionising-spectrum mean photon energy.
    covering_fraction : float, optional
        NLR covering factor. Default 0.1.
    fwhm_kms : float, optional
        NLR line FWHM [km/s]. Default 500.
    alpha_pl : float, optional
        AGN ionising power-law slope :math:`f_\nu \propto \nu^{\alpha}`.
        Feltre grid discretises α ∈ {−2.0, −1.7, −1.4, −1.2}; the backend
        snaps to the nearest grid point. Default −1.7.
    neb_logU : float, optional
        :math:`\log_{10}(U)` gas ionisation parameter. Default −2.0.
    neb_logn : float, optional
        :math:`\log_{10}(n_H/\mathrm{cm}^{-3})` gas density. Feltre grid
        discretises log n_H ∈ {2, 3, 4}. Default 3.0 (typical NLR).
    neb_logZ_gas : float, optional
        :math:`\log_{10}(Z_{\rm gas})` absolute gas metallicity. Default
        −1.8477 = :math:`\log_{10}(Z_\odot)`.
    xi_d : float, optional
        Dust-to-metal ratio. Feltre discretises ξ_d ∈ {0.1, 0.3, 0.5};
        snaps to nearest. Default 0.3.
    grid_path : str or None, optional
        Path to ``feltre_grid.h5``. If ``None``, uses tengri's default
        location (``data/feltre_grid.h5``) built via
        ``scripts/build_feltre_grid.py``.
    **_kwargs
        Accepted for signature compatibility with ``unified_nlr_blr``
        — ignored.

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **Not JIT-compatible at the closure level**: backend initialisation
    loads HDF5. The numerical core (``predict_agn_nlr_lines`` + Gaussian
    convolution) IS JIT-safe — call it inside a wrapping ``jax.jit`` and
    the backend object stays as a Python-level closure.

    The Feltre+2016 grid covers narrow-line emission only (n_H ≤ 10⁴
    cm⁻³). For BLR-density photoionisation use the analytic BLR template
    in :mod:`tengri.components.agn.blr` — there is no Feltre BLR grid.

    References
    ----------
    .. [1] A. Feltre, S. Charlot, J. Gutkin, MNRAS 456, 3354 (2016).
    """
    backend = get_feltre_backend(grid_path)

    # Intercepted accretion luminosity → log10(Q_H ionising photon rate)
    l_acc_erg = covering_fraction * l_disc_bol_erg
    log_qh = _log_qh_from_lacc(l_acc_erg, alpha_pl)

    line_wave_aa, line_lum_lsun = backend.predict_agn_nlr_lines(
        alpha_pl=alpha_pl,
        neb_logU=neb_logU,
        neb_logn=neb_logn,
        neb_logZ_gas=neb_logZ_gas,
        xi_d=xi_d,
        log_qh=log_qh,
        neb_fesc=0.0,
    )

    # Backend returns L_sun; convert to erg/s for downstream consumers.
    line_lum_erg = jnp.asarray(line_lum_lsun) * _L_SUN_ERG_S
    return _lines_to_lnu(
        wavelength,
        jnp.asarray(line_wave_aa),
        line_lum_erg,
        fwhm_kms,
    )


def compute_nlr_sed_synthesizer(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = 500.0,
    grid_path: str | None = None,
    log_bh_mass: float = 8.0,
    log_eddington: float = -0.3,
    cosine_inclination: float = 0.2,
    neb_logU: float = -2.0,
    neb_logn: float = 4.0,
    neb_logZ_gas: float = -1.8477,
    **_kwargs,
) -> jnp.ndarray:
    r"""Synthesizer CLOUDY-grid AGN NLR adapter for ``unified_nlr_blr``.

    The Synthesizer NLR grid uses the unified-AGN parametrisation
    (BH mass, Eddington ratio, inclination) plus the standard
    photoionisation knobs (log U, log Z). Grids are produced by the
    Synthesizer ``grid-generation`` repo running CLOUDY c23.01.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    l_disc_bol_erg : float
        Disc bolometric luminosity [erg/s]. Used only for Q_H normalisation
        via the Synthesizer backend's internal accounting.
    covering_fraction : float, optional
        NLR covering factor. Default 0.1.
    fwhm_kms : float, optional
        NLR line FWHM [km/s]. Default 500.
    grid_path : str, required
        Path to a Synthesizer AGN NLR HDF5 grid (no default — these
        grids are not packaged with tengri because they can be many GB
        in size; supply your own).
    log_bh_mass, log_eddington, cosine_inclination : float
        Synthesizer-specific physical drivers (see
        :class:`SynthesizerNLRBackend.predict_agn_nlr_lines` for details).
    neb_logU, neb_logZ_gas : float
        Standard photoionisation parameters. Defaults reflect typical
        NLR conditions.
    **_kwargs
        Ignored — for ``unified_nlr_blr`` signature compatibility.

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density :math:`L_\nu` [erg/s/Hz].
    """
    if grid_path is None:
        raise ValueError(
            "compute_nlr_sed_synthesizer requires a `grid_path` to a "
            "Synthesizer AGN NLR HDF5 grid. These grids are generated via "
            "synthesizer-project/grid-generation; tengri does not package "
            "them by default."
        )

    backend = get_synthesizer_nlr_backend(grid_path)
    l_acc_erg = covering_fraction * l_disc_bol_erg
    # The backend stores line luminosities *per ionizing photon*; the absolute
    # scale comes from Q_H. Derive log10(Q_H) from the intercepted accretion
    # luminosity so covering_fraction and l_disc_bol_erg actually drive the
    # output. Map the photoionisation knobs onto the backend's parameter names
    # (log_ionU / log_metallicity / log_nH) — passing the ``neb_*`` aliases lets
    # them fall into ``**_kwargs`` and silently revert to grid defaults.
    log_qh = _log_qh_from_lacc(l_acc_erg, alpha_pl=-1.7)
    line_wave_aa, line_lum_lsun = backend.predict_agn_nlr_lines(
        log_bh_mass=log_bh_mass,
        log_eddington=log_eddington,
        cosine_inclination=cosine_inclination,
        log_metallicity=neb_logZ_gas,
        log_ionU=neb_logU,
        log_nH=neb_logn,
        log_qh=log_qh,
    )

    line_lum_erg = jnp.asarray(line_lum_lsun) * _L_SUN_ERG_S
    return _lines_to_lnu(
        wavelength,
        jnp.asarray(line_wave_aa),
        line_lum_erg,
        fwhm_kms,
    )


def compute_blr_sed_synthesizer(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = 5000.0,
    grid_path: str | None = None,
    log_bh_mass: float = 8.0,
    log_eddington: float = -0.3,
    cosine_inclination: float = 0.2,
    neb_logU: float = -1.0,
    neb_logn: float = 10.0,
    neb_logZ_gas: float = -1.8477,
    **_kwargs,
) -> jnp.ndarray:
    r"""Synthesizer CLOUDY-grid AGN **BLR** adapter for ``unified_nlr_blr``.

    The broad-line-region sibling of :func:`compute_nlr_sed_synthesizer`. The
    Synthesizer BLR grid shares the NLR grid's six-axis structure, so this routes
    the requested wavelength grid through :class:`SynthesizerBLRBackend` (which
    reuses the NLR interpolation on the BLR grid file) and convolves the predicted
    broad lines into a continuous :math:`L_\nu`. The default ``fwhm_kms`` is
    5000 km/s — broad-line widths smear the permitted lines into the familiar
    quasar pseudo-continuum, in contrast to the ~500 km/s NLR width.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    l_disc_bol_erg : float
        Disc bolometric luminosity [erg/s] (used for :math:`Q_H` normalisation
        via the backend's internal accounting).
    covering_fraction : float, optional
        BLR covering factor. Default 0.1.
    fwhm_kms : float, optional
        BLR line FWHM [km/s]. Default 5000 (broad permitted lines).
    grid_path : str, required
        Path to a Synthesizer AGN BLR HDF5 grid (e.g.
        ``test_grid_agn-blr.hdf5``). No default — these grids are not packaged
        with tengri.
    log_bh_mass, log_eddington, cosine_inclination : float
        Synthesizer-specific physical drivers (see
        :meth:`SynthesizerBLRBackend.predict_agn_blr_lines`).
    neb_logU, neb_logZ_gas : float
        Photoionisation parameters. Defaults reflect denser, more ionised BLR
        conditions (``neb_logU = -1`` vs the NLR's ``-2``).
    **_kwargs
        Ignored — for ``unified_nlr_blr`` signature compatibility.

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **JIT** — backend init loads HDF5 (Python-level); the numerical core
    (interpolation + Gaussian convolution) is JIT-safe.

    References
    ----------
    .. [1] C. C. Lovell et al., "Synthesizer," MNRAS (2025), arXiv:2004.07283.
    """
    if grid_path is None:
        raise ValueError(
            "compute_blr_sed_synthesizer requires a `grid_path` to a Synthesizer "
            "AGN BLR HDF5 grid (e.g. test_grid_agn-blr.hdf5). These grids are "
            "generated via synthesizer-project/grid-generation; tengri does not "
            "package them by default."
        )

    backend = get_synthesizer_blr_backend(grid_path)
    l_acc_erg = covering_fraction * l_disc_bol_erg
    # See compute_nlr_sed_synthesizer: derive log10(Q_H) from the intercepted
    # accretion luminosity and map the photoionisation knobs onto the backend's
    # parameter names so covering_fraction / l_disc_bol_erg drive the amplitude.
    log_qh = _log_qh_from_lacc(l_acc_erg, alpha_pl=-1.7)
    line_wave_aa, line_lum_lsun = backend.predict_agn_blr_lines(
        log_bh_mass=log_bh_mass,
        log_eddington=log_eddington,
        cosine_inclination=cosine_inclination,
        log_metallicity=neb_logZ_gas,
        log_ionU=neb_logU,
        log_nH=neb_logn,
        log_qh=log_qh,
    )

    line_lum_erg = jnp.asarray(line_lum_lsun) * _L_SUN_ERG_S
    return _lines_to_lnu(
        wavelength,
        jnp.asarray(line_wave_aa),
        line_lum_erg,
        fwhm_kms,
    )
