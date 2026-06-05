# SPDX-License-Identifier: BSD-3-Clause
"""Narrow-line region (NLR) blocks for the composable AGN pipeline.

Wraps :func:`compute_nlr_sed` and related NLR synthesis functions so they fit
into the composable AGN pipeline. The NLR is spatially extended and isotropically
illuminated, so it is driven by the **intrinsic** AGN bolometric luminosity
(:math:`L_{\\rm bol} = 10^{\\,\\mathrm{agn\\_log\\_lbol}} L_\\odot`). Its flux is
therefore inclination-independent (the runner's Type-1/2 mask leaves it untouched).

References
----------
.. [1] Krawczyk, C. M. et al. 2013, ApJS, 206, 4 (quasar SED bolometric
       corrections).
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.nlr import compute_nlr_sed
from tengri.components.agn.nlr_cloudy import (
    compute_nlr_sed_synthesizer,
    compute_nlr_sed_synthesizer_spectra,
)
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG

__all__: list[str] = []  # registrations only

_C_AA_PER_S: float = 2.99792458e18


def _resolve_synthesizer_grid(kind: str) -> str:
    """Resolve a Synthesizer AGN grid path for the grid-backed line blocks.

    Searches ``$TENGRI_SYNTHESIZER_AGN_GRID_DIR`` then the repo-default
    ``data/synthesizer_grids/``. ``kind`` is ``"nlr"`` or ``"blr"``.

    These grids are not packaged with tengri (they ship via
    ``synthesizer-download --agn-test-grids``), so a clear error is raised if
    neither location holds ``test_grid_agn-<kind>.hdf5``.
    """
    fname = f"test_grid_agn-{kind}.hdf5"
    candidates = []
    env = os.environ.get("TENGRI_SYNTHESIZER_AGN_GRID_DIR")
    if env:
        candidates.append(Path(env) / fname)
    candidates.append(Path("data/synthesizer_grids") / fname)
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        f"Synthesizer AGN {kind.upper()} grid '{fname}' not found. Set "
        "$TENGRI_SYNTHESIZER_AGN_GRID_DIR or place it in data/synthesizer_grids/ "
        "(fetch via `synthesizer-download --agn-test-grids`)."
    )


@register_agn_block(
    "nlr",
    "analytic",
    citation="Richardson et al. 2014, ApJ, 786, 87",
    status="production",
    short_doc="Analytic NLR with Richardson+2014 line ratios and Gaussian broadening",
)
def nlr_analytic_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    agn_nlr_line_efficiency: float = 0.10,
    **_params,
) -> Array:
    r"""Narrow-line region emission as an nlr-stage block.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        log10 of the AGN bolometric luminosity [Lsun] — the **isotropic**
        illuminator of the NLR.
    l5100_disc : array, scalar
        Ignored (the NLR is illuminated by the inclination-independent
        bolometric, not the apparent — possibly foreshortened — l5100).
    agn_nlr_cf : float, optional
        NLR covering fraction. Default ``0.1``.
    agn_nlr_fwhm_kms : float, optional
        Narrow-line FWHM [km/s]. Default ``500``.
    agn_nlr_line_efficiency : float, optional
        Default ``0.10``.

    Returns
    -------
    L_lambda : tuple of ndarray, shape (n_wave,)
        Tuple of ``(maskable, isotropic)`` channels. The NLR is spatially
        extended and illuminated by the **intrinsic** AGN bolometric
        luminosity (:math:`L_{\rm bol} = 10^{\,\mathrm{agn\_log\_lbol}}
        L_\odot`), so its flux is inclination-independent — matching
        ``unified_nlr_blr``. Using the apparent ``l5100_disc`` would leak an
        inclination-dependent disc's ``cos_inc`` foreshortening into the
        (supposedly isotropic) narrow lines. Returns zero on the maskable
        channel and the NLR spectrum on the isotropic channel; the runner's
        Type-1/2 mask leaves it untouched.

    Notes
    -----
    The NLR is spatially extended -> isotropic (visible at all inclinations).
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm_kms,
        line_efficiency=agn_nlr_line_efficiency,
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    # NLR is spatially extended -> isotropic (visible at all inclinations); it is
    # the *isotropic* channel so the runner's Type-1/2 mask leaves it untouched.
    return jnp.zeros_like(L_lambda), L_lambda


@register_agn_block(
    "nlr",
    "synthesizer",
    citation="Lovell et al. 2025, arXiv:2508.03888",
    status="production",
    short_doc="Synthesizer Cloudy grid narrow-line region",
)
def nlr_synthesizer_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    neb_logU: float = -2.0,
    neb_logZ_gas: float = -1.8477,
    **_params,
) -> Array:
    r"""NLR lines from the Synthesizer Cloudy grid (grid-backed nlr block).

    Routes the composable pipeline to :func:`compute_nlr_sed_synthesizer`, so a
    unified AGN built through ``SEDModel.build`` uses the *same* photoionisation
    grid as the direct adapter (the grids the §9c reproduction panel reads). The
    grid path is resolved from ``$TENGRI_SYNTHESIZER_AGN_GRID_DIR`` /
    ``data/synthesizer_grids/`` (closes the builder-accessibility gap, #588).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored (bolometric is taken from ``l5100_disc``).
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_nlr_cf, agn_nlr_fwhm_kms : float
        Covering fraction and narrow-line FWHM [km/s].
    neb_logU, neb_logZ_gas : float
        Photoionisation knobs forwarded to the grid adapter.

    Returns
    -------
    tuple of ndarray, shape (n_wave,)
        ``(maskable, isotropic)`` channels. NLR is isotropic.

    Notes
    -----
    Like the analytic :func:`nlr_analytic_block`, the NLR is illuminated by the
    **intrinsic** bolometric luminosity (``10**agn_log_lbol * L_sun``) and is
    therefore inclination-independent (isotropic). Backend initialisation reads
    HDF5 (Python-level, not JIT-traceable); call once eagerly before any
    ``jax.jit`` over the forward model so the cached backend's interpolation
    stays JIT-safe.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed_synthesizer(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm_kms,
        grid_path=_resolve_synthesizer_grid("nlr"),
        neb_logU=neb_logU,
        neb_logZ_gas=neb_logZ_gas,
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    # Isotropic channel (extended NLR) — bypasses the runner's Type-1/2 mask.
    return jnp.zeros_like(L_lambda), L_lambda


@register_agn_block(
    "nlr",
    "synthesizer_spectra",
    citation="Lovell et al. 2025, arXiv:2508.03888",
    status="production",
    short_doc="Synthesizer UnifiedAGN NLR reprocessed nebular spectrum",
)
def nlr_synthesizer_spectra_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    neb_logU: float = -2.0,
    neb_logn: float = 4.0,
    neb_logZ_gas: float = -2.0,
    **_params,
) -> Array:
    r"""NLR reprocessed nebular spectrum reproducing Synthesizer's UnifiedAGN.

    Reads the grid's ``/spectra/nebular`` array (continuum + lines) — the exact
    product Synthesizer's ``UnifiedAGN`` extracts — instead of re-broadening the
    discrete ``/lines`` table (issue #694). Isotropic: illuminated by the
    intrinsic bolometric ``10**agn_log_lbol * L_sun`` with the grid inclination
    held at its isotropic node, so it bypasses the runner's Type-1/2 mask.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        log10(L_bol / L_sun) of the AGN.
    l5100_disc : array, scalar
        Ignored (bolometric is taken from ``agn_log_lbol``).
    agn_nlr_cf : float
        NLR covering fraction.
    neb_logU, neb_logn, neb_logZ_gas : float
        Photoionisation knobs forwarded to the grid adapter (log Z absolute).

    Returns
    -------
    tuple of ndarray, shape (n_wave,)
        ``(maskable, isotropic)`` channels; the NLR is fully isotropic.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed_synthesizer_spectra(
        wave_aa,
        l_disc_bol_erg=l_bol_erg,
        covering_fraction=agn_nlr_cf,
        grid_path=_resolve_synthesizer_grid("nlr"),
        neb_logU=neb_logU,
        neb_logn=neb_logn,
        neb_logZ_gas=neb_logZ_gas,
        region="nlr",
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    return jnp.zeros_like(L_lambda), L_lambda
