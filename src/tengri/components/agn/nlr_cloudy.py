# SPDX-License-Identifier: BSD-3-Clause
r"""Cloudy / photoionization NLR adapters for the AGN unified model.

The Cloudy-style backends in :mod:`tengri.components.nebular.agn_nebular`
predict **discrete line luminosities** from an AGN ionizing source. The
``unified_nlr_blr`` AGN model however expects a callable that returns a
continuous ``L_nu`` spectrum on the user's wavelength grid. This module
provides thin adapter functions that bridge the two interfaces:

1. ``compute_nlr_sed_feltre``: Feltre, Charlot & Gutkin 2016 grid
   (CLOUDY c13.03; BEAGLE parity).
2. ``compute_nlr_sed_synthesizer``: Synthesizer CLOUDY c23.01 AGN NLR
   grid (lazy: requires the user-supplied HDF5 path).

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
.. [2] C. C. Lovell et al. 2025, Open Journal of Astrophysics, 8,
   "Synthesizer: a Software Package for Synthetic Astronomical Observables",
   https://doi.org/10.33232/001c.145766
.. [3] W. J. Roper et al. 2026, Journal of Open Source Software, 11, 9436,
   "Synthesizer: Synthetic Observables for Modern Astronomy",
   https://doi.org/10.21105/joss.09436
   (Both Synthesizer papers [2]_ [3]_ must be cited together.)
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import vmap

from tengri.components.agn._phys import gaussian_line_profile as _gaussian_line_profile
from tengri.components.nebular._constants import _LOG10_ZSUN
from tengri.components.nebular.agn_nebular import (
    FeltreNLRBackend,
    SynthesizerBLRBackend,
    SynthesizerNLRBackend,
    _log_qh_from_lacc,
    agn_nlr_cue,
)
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG_S

__all__ = [
    "compute_blr_sed_synthesizer",
    "compute_nlr_sed_cue",
    "compute_nlr_sed_feltre",
    "compute_nlr_sed_synthesizer",
    "get_cue_agn_backend",
    "get_feltre_backend",
    "get_synthesizer_blr_backend",
    "get_synthesizer_nlr_backend",
    "load_cue_agn_weights",
]

#: Default location of the Cue emulator weights (built by the user; data-gated).
_DEFAULT_CUE_WEIGHTS_PATH = "data/cue_weights.npz"

_FELTRE_BACKEND: FeltreNLRBackend | None = None
_SYNTHESIZER_BACKEND: SynthesizerNLRBackend | None = None
_SYNTHESIZER_BLR_BACKEND: SynthesizerBLRBackend | None = None
_CUE_AGN_BACKEND = None  # lazy CueBackend for the AGN-ionized NLR block


def _eager_construction():
    """Build a cached backend outside whatever trace the caller is inside.

    Every accessor below is a lazy singleton, so the **first** caller decides
    what the cache holds for the rest of the process. That is fine until the
    first caller is inside a ``jax.jit`` trace: any ``jnp`` operation run while
    a trace is active returns a tracer bound to that trace, however concrete
    its inputs are. The backend is then cached full of tracers, and the next
    reader, a later test, the next fit: fails with ``UnexpectedTracerError``
    naming the loader rather than whoever poisoned it.

    Three of these four constructors reach ``jnp``:
    :class:`CueBackend` via ``load_cue_weights`` (16x12 stacked network
    parameters), and ``FeltreNLRBackend`` / ``SynthesizerNLRBackend`` via
    ``jnp.sort`` (which ``SynthesizerBLRBackend`` inherits).

    Only the Cue path had actually been caught, as the ``test_cue_nlr_grammar``
    failure on main, and it reproduced only under a particular test order,
    which is why it passed locally and failed in CI. The others are the same
    construction under the same cache and differ only in whether anything has
    happened to call them from inside a trace yet. Guarding the cache boundary
    rather than the three constructors keeps the rule in one place: *a lazy
    singleton must not let its first caller decide.*
    """
    import jax

    return jax.ensure_compile_time_eval()


def get_feltre_backend(grid_path: str | None = None) -> FeltreNLRBackend:
    """Lazy singleton accessor for :class:`FeltreNLRBackend`.

    Subsequent calls return the same instance unless a fresh ``grid_path``
    is supplied. Loading the HDF5 grid is the slow step (~1 s) and we
    cache it for the duration of the Python process.

    Parameters
    ----------
    grid_path: str or None, optional
        Path to ``feltre_grid.h5``. If ``None``, uses the package default
        from ``tengri.components.nebular.agn_nebular._DEFAULT_FELTRE_GRID_PATH``.

    Returns
    -------
    FeltreNLRBackend
        Initialized backend, ready for ``predict_agn_nlr_lines`` calls.

    Raises
    ------
    FileNotFoundError
        If the grid file does not exist at the resolved path.
    """
    global _FELTRE_BACKEND
    if grid_path is not None or _FELTRE_BACKEND is None:
        with _eager_construction():
            _FELTRE_BACKEND = (
                FeltreNLRBackend() if grid_path is None else FeltreNLRBackend(grid_path)
            )
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
    grid_path: str
        Path to a Synthesizer CLOUDY AGN NLR HDF5 grid.

    Returns
    -------
    SynthesizerNLRBackend
        Initialized backend.
    """
    global _SYNTHESIZER_BACKEND
    if _SYNTHESIZER_BACKEND is None or _SYNTHESIZER_BACKEND.grid_path != grid_path:
        with _eager_construction():
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
    grid_path: str
        Path to a Synthesizer CLOUDY AGN BLR HDF5 grid.

    Returns
    -------
    SynthesizerBLRBackend
        Initialized backend.
    """
    global _SYNTHESIZER_BLR_BACKEND
    if _SYNTHESIZER_BLR_BACKEND is None or _SYNTHESIZER_BLR_BACKEND.grid_path != grid_path:
        with _eager_construction():
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
    fwhm_kms)``, where the Gaussian is per-Hz-normalized so the spectral
    integral over frequency returns ``line_lum`` in erg/s.

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    line_wavelengths: array, shape (n_lines,)
        Line rest wavelengths [Å].
    line_luminosities_erg_s: array, shape (n_lines,)
        Line luminosities [erg/s] (already in CGS: convert from L_sun
        upstream if needed).
    fwhm_kms: float
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
    wavelength: array, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    l_disc_bol_erg: float
        AGN disc bolometric luminosity [erg/s]. The Feltre grid is
        driven by Q_H, derived from ``covering_fraction × l_disc_bol_erg``
        via the ionizing-spectrum mean photon energy.
    covering_fraction: float, optional
        NLR covering factor. Default 0.1.
    fwhm_kms: float, optional
        NLR line FWHM [km/s]. Default 500.
    alpha_pl: float, optional
        AGN ionizing power-law slope :math:`f_\nu \propto \nu^{\alpha}`.
        Feltre grid discretizes α ∈ {−2.0, −1.7, −1.4, −1.2}; the backend
        snaps to the nearest grid point. Default −1.7.
    neb_logU: float, optional
        :math:`\log_{10}(U)` gas ionization parameter. Default −2.0.
    neb_logn: float, optional
        :math:`\log_{10}(n_H/\mathrm{cm}^{-3})` gas density. Feltre grid
        discretizes log n_H ∈ {2, 3, 4}. Default 3.0 (typical NLR).
    neb_logZ_gas: float, optional
        :math:`\log_{10}(Z_{\rm gas})` absolute gas metallicity. Default
        −1.8477 = :math:`\log_{10}(Z_\odot)`.
    xi_d: float, optional
        Dust-to-metal ratio. Feltre discretizes ξ_d ∈ {0.1, 0.3, 0.5};
        snaps to nearest. Default 0.3.
    grid_path: str or None, optional
        Path to ``feltre_grid.h5``. If ``None``, uses tengri's default
        location (``data/feltre_grid.h5``) built via
        ``scripts/build_feltre_grid.py``.
    **_kwargs
        Accepted for signature compatibility with ``unified_nlr_blr``;
        ignored.

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **Not JIT-compatible at the closure level**: backend initialization
    loads HDF5. The numerical core (``predict_agn_nlr_lines`` + Gaussian
    convolution) IS JIT-safe: call it inside a wrapping ``jax.jit`` and
    the backend object stays as a Python-level closure.

    The Feltre+2016 grid covers narrow-line emission only (n_H ≤ 10⁴
    cm⁻³). For BLR-density photoionization use the analytic BLR template
    in :mod:`tengri.components.agn.blr`: there is no Feltre BLR grid.

    References
    ----------
    .. [1] A. Feltre, S. Charlot, J. Gutkin, MNRAS 456, 3354 (2016).
    """
    backend = get_feltre_backend(grid_path)

    # Intercepted accretion luminosity → log10(Q_H ionizing photon rate)
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


def get_cue_agn_backend(weights_path: str | None = None):
    """Lazy singleton accessor for the Cue emulator (AGN-ionized NLR).

    Loading the ``cue_weights.npz`` neural-net weights is the slow step; the
    backend is cached for the process. No SSP data is threaded, the AGN NLR is
    driven by the disc ionizing spectrum, not a stellar population.

    Parameters
    ----------
    weights_path: str or None, optional
        Path to ``cue_weights.npz``. ``None`` uses
        :data:`_DEFAULT_CUE_WEIGHTS_PATH`.

    Returns
    -------
    CueBackend
        Initialized Cue emulator, ready for ``agn_nlr_cue`` calls.

    Raises
    ------
    FileNotFoundError
        If the weights file does not exist at the resolved path.

    Notes
    -----
    Construction runs inside ``jax.ensure_compile_time_eval()``. Without it,
    a backend first built *inside* a JIT trace captures ``DynamicJaxprTracer``
    values, and because it is cached in a module-level global that poisoned
    instance outlives the trace: every later out-of-trace call then raises
    ``UnexpectedTracerError``. Whoever traces a Cue-NLR model first decides
    whether the rest of the process works, which makes the failure depend on
    execution order. Same defect class as the GRAHSP template cache (#1462).
    """
    global _CUE_AGN_BACKEND
    if weights_path is not None or _CUE_AGN_BACKEND is None:
        from tengri.components.nebular.cue import CueBackend

        with _eager_construction():
            _CUE_AGN_BACKEND = CueBackend(weights_path or _DEFAULT_CUE_WEIGHTS_PATH)
    return _CUE_AGN_BACKEND


def load_cue_agn_weights():
    """Load the Cue emulator weights for the AGN NLR block (cache + discovery).

    This is the ``template_loader`` the ``nlr/cue`` block registers. It returns
    the weights **pytree**, not the backend object: only arrays can thread
    through ``jax.jit`` as arguments, and the backend is an ordinary Python
    object. Roughly 8.5 MB otherwise bakes into the graph as constants (#1383).

    Returns
    -------
    Any
        The backend's ``weights`` pytree.

    Raises
    ------
    FileNotFoundError
        If ``cue_weights.npz`` is not present.

    Notes
    -----
    **JIT-compatible**: no, deliberately: it must run before tracing.
    """
    return get_cue_agn_backend().weights


def compute_nlr_sed_cue(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    fwhm_kms: float = 500.0,
    alpha_pl: float = -1.7,
    neb_logU: float = -2.0,
    neb_logn: float = 3.0,
    neb_logZ_gas: float = -1.8477,
    weights_path: str | None = None,
    _template=None,
    **_kwargs,
) -> jnp.ndarray:
    r"""Cue-emulator AGN-ionized NLR adapter (the disc → Cue → NLR pipeline).

    The disc's power-law ionizing continuum (:math:`f_\nu \propto \nu^{\alpha}`)
    sets :math:`Q_{\rm H}` from :math:`L_{\rm acc}`, and the Cue neural-network
    emulator (Li+2025) predicts AGN-ionized narrow lines from the ionizing
    spectrum shape and gas parameters, the BEAGLE-style physical NLR, but with
    Cue's fast differentiable emulator in place of a tabulated CLOUDY grid. Lines
    are Gaussian-broadened at ``fwhm_kms``. Output is :math:`L_\nu` [erg/s/Hz].

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    l_disc_bol_erg: float
        AGN disc bolometric luminosity [erg/s]; drives :math:`Q_{\rm H}`.
    covering_fraction: float, optional
        NLR covering factor; scales the emergent line luminosity. Default 0.1.
    fwhm_kms: float, optional
        NLR line FWHM [km/s]. Default 500.
    alpha_pl: float, optional
        AGN EUV ionizing power-law slope. Default −1.7.
    neb_logU: float, optional
        :math:`\log_{10}(U)` gas ionization parameter. Default −2.0.
    neb_logn: float, optional
        :math:`\log_{10}(n_H/\mathrm{cm}^{-3})` gas density. Default 3.0.
    neb_logZ_gas: float, optional
        :math:`\log_{10}(Z_{\rm gas})` **absolute** gas metallicity (same
        convention as the Feltre block). Converted to Cue's native
        :math:`\log_{10}(Z/Z_\odot)` internally via ``_LOG10_ZSUN``. Default
        −1.8477 = solar.
    weights_path: str or None, optional
        Path to ``cue_weights.npz``. ``None`` uses the package default.
    **_kwargs
        Accepted for signature compatibility; ignored.

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **Not JIT-compatible at the closure level**: backend init loads the weights.
    The numerical core (``agn_nlr_cue`` emulator call + Gaussian convolution) is
    JIT-safe: call it inside a wrapping ``jax.jit`` with the backend held as a
    Python closure.

    References
    ----------
    .. [1] M. Li et al., ApJ 986, 9 (2025). arXiv:2405.04598.
    """
    backend = get_cue_agn_backend(weights_path)
    # agn_nlr_cue takes Cue's native log10(Z/Zsun); the block axis is absolute.
    gas_logz_rel = neb_logZ_gas - _LOG10_ZSUN
    line_wave_aa, line_lum_lsun = agn_nlr_cue(
        backend,
        l_acc_erg=l_disc_bol_erg,
        covering_fraction=covering_fraction,
        neb_logU=neb_logU,
        gas_logn=neb_logn,
        gas_logz=gas_logz_rel,
        alpha_pl=alpha_pl,
        template_data=_template,
    )
    # agn_nlr_cue already scales lines by covering_fraction; convert L_sun→erg/s.
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
    use_grid_qh: bool = True,
    **_kwargs,
) -> jnp.ndarray:
    r"""Synthesizer CLOUDY-grid AGN NLR adapter for ``unified_nlr_blr``.

    The Synthesizer NLR grid uses the unified-AGN parametrization
    (BH mass, Eddington ratio, inclination) plus the standard
    photoionization knobs (log U, log Z). Grids are produced by the
    Synthesizer ``grid-generation`` repo running CLOUDY c23.01.

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    l_disc_bol_erg: float
        Disc bolometric luminosity [erg/s]. Used only for Q_H normalization
        via the Synthesizer backend's internal accounting.
    covering_fraction: float, optional
        NLR covering factor. Default 0.1.
    fwhm_kms: float, optional
        NLR line FWHM [km/s]. Default 500.
    grid_path: str, required
        Path to a Synthesizer AGN NLR HDF5 grid (no default: these
        grids are not packaged with tengri because they can be many GB
        in size; supply your own).
    log_bh_mass, log_eddington, cosine_inclination: float
        Synthesizer-specific physical drivers (see
        :class:`SynthesizerNLRBackend.predict_agn_nlr_lines` for details).
    neb_logU, neb_logZ_gas: float
        Standard photoionization parameters. Defaults reflect typical
        NLR conditions.
    **_kwargs
        Ignored, for ``unified_nlr_blr`` signature compatibility.

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
    log_qh = _resolve_log_qh(
        backend,
        l_disc_bol_erg,
        use_grid_qh,
        log_bh_mass,
        log_eddington,
        cosine_inclination,
        neb_logZ_gas,
        neb_logU,
        neb_logn,
    )
    # Map the photoionization knobs onto the backend's parameter names
    # (log_ionU / log_metallicity / log_nH): passing the ``neb_*`` aliases lets
    # them fall into ``**_kwargs`` and silently revert to grid defaults.
    line_wave_aa, line_lum_lsun = backend.predict_agn_nlr_lines(
        log_bh_mass=log_bh_mass,
        log_eddington=log_eddington,
        cosine_inclination=cosine_inclination,
        log_metallicity=neb_logZ_gas,
        log_ionU=neb_logU,
        log_nH=neb_logn,
        log_qh=log_qh,
    )

    # Covering fraction scales the observed reprocessed lines (separate from Q_H,
    # which is the disc's intrinsic ionizing output).
    line_lum_erg = jnp.asarray(line_lum_lsun) * _L_SUN_ERG_S * covering_fraction
    return _lines_to_lnu(
        wavelength,
        jnp.asarray(line_wave_aa),
        line_lum_erg,
        fwhm_kms,
    )


def _resolve_log_qh(
    backend,
    l_disc_bol_erg,
    use_grid_qh,
    log_bh_mass,
    log_eddington,
    cosine_inclination,
    neb_logZ_gas,
    neb_logU,
    neb_logn,
):
    r"""Return log10(Q_H [photons/s]) for the line-luminosity normalization.

    With ``use_grid_qh`` (default) the grid's own specific ionizing luminosity is
    used: ``log10(Q_H) = log10(Q_H/L_bol)_grid + log10(L_bol[erg/s]) - 7`` (the
    −7 converts erg/s to W): so tengri reproduces Synthesizer's own disc-model
    :math:`Q_H` rather than assuming an ionizing-spectrum slope. The legacy path
    derives :math:`Q_H` from the accretion luminosity and an assumed slope.
    """
    if use_grid_qh and getattr(backend.grid, "log_qh_specific", None) is not None:
        log_qh_specific = backend.interp_log_qh_specific(
            log_bh_mass, log_eddington, cosine_inclination, neb_logZ_gas, neb_logU, neb_logn
        )
        return log_qh_specific + jnp.log10(l_disc_bol_erg) - 7.0
    return _log_qh_from_lacc(l_disc_bol_erg, alpha_pl=-1.7)


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
    use_grid_qh: bool = True,
    **_kwargs,
) -> jnp.ndarray:
    r"""Synthesizer CLOUDY-grid AGN **BLR** adapter for ``unified_nlr_blr``.

    The broad-line-region sibling of :func:`compute_nlr_sed_synthesizer`. The
    Synthesizer BLR grid shares the NLR grid's six-axis structure, so this routes
    the requested wavelength grid through :class:`SynthesizerBLRBackend` (which
    reuses the NLR interpolation on the BLR grid file) and convolves the predicted
    broad lines into a continuous :math:`L_\nu`. The default ``fwhm_kms`` is
    5000 km/s: broad-line widths smear the permitted lines into the familiar
    quasar pseudo-continuum, in contrast to the ~500 km/s NLR width.

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Rest-frame wavelength grid [Å].
    l_disc_bol_erg: float
        Disc bolometric luminosity [erg/s] (used for :math:`Q_H` normalization
        via the backend's internal accounting).
    covering_fraction: float, optional
        BLR covering factor. Default 0.1.
    fwhm_kms: float, optional
        BLR line FWHM [km/s]. Default 5000 (broad permitted lines).
    grid_path: str, required
        Path to a Synthesizer AGN BLR HDF5 grid (e.g.
        ``test_grid_agn-blr.hdf5``). No default: these grids are not packaged
        with tengri.
    log_bh_mass, log_eddington, cosine_inclination: float
        Synthesizer-specific physical drivers (see
        :meth:`SynthesizerBLRBackend.predict_agn_blr_lines`).
    neb_logU, neb_logZ_gas: float
        Photoionization parameters. Defaults reflect denser, more ionized BLR
        conditions (``neb_logU = -1`` vs the NLR's ``-2``).
    **_kwargs
        Ignored, for ``unified_nlr_blr`` signature compatibility.

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **JIT**: backend init loads HDF5 (Python-level); the numerical core
    (interpolation + Gaussian convolution) is JIT-safe.

    References
    ----------
    .. [1] C. C. Lovell et al. 2025, Open Journal of Astrophysics, 8,
           https://doi.org/10.33232/001c.145766
    .. [2] W. J. Roper et al. 2026, Journal of Open Source Software, 11, 9436,
           https://doi.org/10.21105/joss.09436
           (Both Synthesizer papers must be cited together.)
    """
    if grid_path is None:
        raise ValueError(
            "compute_blr_sed_synthesizer requires a `grid_path` to a Synthesizer "
            "AGN BLR HDF5 grid (e.g. test_grid_agn-blr.hdf5). These grids are "
            "generated via synthesizer-project/grid-generation; tengri does not "
            "package them by default."
        )

    backend = get_synthesizer_blr_backend(grid_path)
    # Use the grid's own Q_H normalization (see compute_nlr_sed_synthesizer); the
    # covering fraction scales the observed lines separately.
    log_qh = _resolve_log_qh(
        backend,
        l_disc_bol_erg,
        use_grid_qh,
        log_bh_mass,
        log_eddington,
        cosine_inclination,
        neb_logZ_gas,
        neb_logU,
        neb_logn,
    )
    line_wave_aa, line_lum_lsun = backend.predict_agn_blr_lines(
        log_bh_mass=log_bh_mass,
        log_eddington=log_eddington,
        cosine_inclination=cosine_inclination,
        log_metallicity=neb_logZ_gas,
        log_ionU=neb_logU,
        log_nH=neb_logn,
        log_qh=log_qh,
    )

    line_lum_erg = jnp.asarray(line_lum_lsun) * _L_SUN_ERG_S * covering_fraction
    return _lines_to_lnu(
        wavelength,
        jnp.asarray(line_wave_aa),
        line_lum_erg,
        fwhm_kms,
    )


def compute_nlr_sed_synthesizer_spectra(
    wavelength: jnp.ndarray,
    l_disc_bol_erg: float,
    covering_fraction: float = 0.1,
    grid_path: str | None = None,
    log_bh_mass: float = 8.0,
    log_eddington: float = -0.3,
    neb_logU: float = -2.0,
    neb_logn: float = 4.0,
    neb_logZ_gas: float = -2.0,
    region: str = "nlr",
    **_kwargs,
) -> jnp.ndarray:
    r"""Reprocessed NLR/BLR :math:`L_\nu` reproducing Synthesizer's UnifiedAGN.

    Unlike :func:`compute_nlr_sed_synthesizer` (which re-broadens the grid's
    discrete ``/lines`` table), this reads the grid's reprocessed
    ``/spectra/nebular`` array, the *same* product Synthesizer's ``UnifiedAGN``
    extracts (``extract="nebular"``) for its NLR/BLR components: so a unified
    AGN built through the grammar reproduces ``UnifiedAGN`` (issue #694). The
    line-region emission is isotropic (grid ``cosine_inclination`` held at 0.5,
    matching Synthesizer).

    Parameters
    ----------
    wavelength: array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    l_disc_bol_erg: float
        Disc bolometric luminosity [erg/s].
    covering_fraction: float, optional
        Line-region covering factor. Default 0.1.
    grid_path: str, required
        Path to a Synthesizer AGN HDF5 grid carrying ``/spectra/nebular``.
    log_bh_mass, log_eddington: float
        Grid drivers (``cosine_inclination`` is held at 0.5 internally).
    neb_logU, neb_logn, neb_logZ_gas: float
        Photoionization knobs (log U, log n_H, log Z absolute).
    region: {"nlr", "blr"}, optional
        Which grid (and backend) to read. Default ``"nlr"``.

    Returns
    -------
    ndarray, shape (n_wave,)
        Reprocessed nebular :math:`L_\nu` [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes (backend init is eager). ``L_nu = nebular_per_lbol *
    L_bol * f_cov``, verified against Synthesizer's UnifiedAGN ``nlr`` component
    (~2 % on amplitude, shape correlation 0.97 on the test grid; exact-node
    smoothing is the C²-interpolation caveat). Implemented to match Synthesizer
    (Lovell et al. 2025; Roper et al. 2026).
    """
    if grid_path is None:
        raise ValueError(
            "compute_nlr_sed_synthesizer_spectra requires a `grid_path` to a "
            "Synthesizer AGN HDF5 grid carrying /spectra/nebular."
        )

    backend = (
        get_synthesizer_blr_backend(grid_path)
        if region == "blr"
        else get_synthesizer_nlr_backend(grid_path)
    )
    return backend.predict_agn_nebular_spectrum(
        jnp.asarray(wavelength),
        l_bol_erg=l_disc_bol_erg,
        covering_fraction=covering_fraction,
        log_bh_mass=log_bh_mass,
        log_eddington=log_eddington,
        log_metallicity=neb_logZ_gas,
        log_ionU=neb_logU,
        log_nH=neb_logn,
    )
