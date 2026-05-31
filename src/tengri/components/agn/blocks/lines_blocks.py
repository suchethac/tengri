# SPDX-License-Identifier: BSD-3-Clause
"""Lines-stage blocks: BLR (broad-line region) and NLR (narrow-line region).

Wraps :func:`compute_blr_sed` and :func:`compute_nlr_sed` so they fit into
the composable AGN pipeline. Both upstream functions take a *bolometric*
disc luminosity in erg/s, which is not directly available to the block
runner — only :math:`\\lambda L_\\lambda(5100\\,\\mathrm{\\AA})`
(``l5100_disc``) is. The adapters perform the bolometric-correction
conversion at the boundary:

.. math::

   L_{\\rm disc, bol} = f_{\\rm bol} \\, \\lambda L_\\lambda(5100\\,\\mathrm{\\AA})

with default :math:`f_{\\rm bol} = 9` from Krawczyk+ 2013 (a typical Type-1
quasar correction). The user can override via ``agn_blr_f_bol`` /
``agn_nlr_f_bol`` if a different correction is appropriate.

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
from tengri.components.agn.blr import compute_blr_sed
from tengri.components.agn.nlr import compute_nlr_sed
from tengri.components.agn.nlr_cloudy import (
    compute_blr_sed_synthesizer,
    compute_nlr_sed_synthesizer,
)

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


#: Default bolometric correction :math:`L_{\rm bol}/\lambda L_\lambda(5100\,\mathrm{\AA})`
#: from Krawczyk+ 2013 (2013ApJS..206....4K). Type-1 quasar median.
DEFAULT_F_BOL_5100: float = 9.0


@register_agn_block("lines", "blr")
def blr_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_blr_cf: float = 0.1,
    agn_blr_fwhm_kms: float = 5000.0,
    agn_fe2_strength: float = 0.0,
    agn_blr_line_efficiency: float = 0.08,
    agn_blr_f_bol: float = DEFAULT_F_BOL_5100,
    **_params,
) -> Array:
    r"""Broad-line region emission as a lines-stage block.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored (kept for protocol compatibility — ``l5100_disc`` provides
        the normalisation).
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_blr_cf : float, optional
        BLR covering fraction [0, 1]. Default ``0.1``.
    agn_blr_fwhm_kms : float, optional
        Broad-line FWHM [km/s]. Default ``5000``.
    agn_fe2_strength : float, optional
        :math:`R_{\rm Fe} = F({\rm FeII})/F({\rm H}\beta)`. Default ``0.0``
        (FeII pseudo-continuum disabled).
    agn_blr_line_efficiency : float, optional
        Fraction of intercepted luminosity converted to lines. Default ``0.08``.
    agn_blr_f_bol : float, optional
        Bolometric correction :math:`L_{\rm bol}/\lambda L_\lambda(5100\,\mathrm{\AA})`.
        Default :data:`DEFAULT_F_BOL_5100`.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        BLR :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    Geometric masking by the torus is **not** applied here. If the
    composable recipe also activates a torus block, double-counting is
    possible for line-of-sight inclination effects; see Section 2 of
    :mod:`tengri.components.agn.unified` for the mask convention.
    """
    del agn_log_lbol  # normalisation comes from l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = jnp.asarray(l5100_disc) * agn_blr_f_bol
    L_nu = compute_blr_sed(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_blr_cf,
        fwhm_kms=agn_blr_fwhm_kms,
        agn_fe2_strength=agn_fe2_strength,
        line_efficiency=agn_blr_line_efficiency,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("lines", "nlr")
def nlr_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    agn_nlr_line_efficiency: float = 0.10,
    agn_nlr_f_bol: float = DEFAULT_F_BOL_5100,
    **_params,
) -> Array:
    r"""Narrow-line region emission as a lines-stage block.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored.
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_nlr_cf : float, optional
        NLR covering fraction. Default ``0.1``.
    agn_nlr_fwhm_kms : float, optional
        Narrow-line FWHM [km/s]. Default ``500``.
    agn_nlr_line_efficiency : float, optional
        Default ``0.10``.
    agn_nlr_f_bol : float, optional
        Bolometric correction; default :data:`DEFAULT_F_BOL_5100`.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        NLR :math:`L_\lambda` [erg/s/Å].
    """
    del agn_log_lbol
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = jnp.asarray(l5100_disc) * agn_nlr_f_bol
    L_nu = compute_nlr_sed(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm_kms,
        line_efficiency=agn_nlr_line_efficiency,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("lines", "nlr_synthesizer")
def nlr_synthesizer_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    agn_nlr_f_bol: float = DEFAULT_F_BOL_5100,
    neb_logU: float = -2.0,
    neb_logZ_gas: float = -1.8477,
    **_params,
) -> Array:
    r"""NLR lines from the Synthesizer Cloudy grid (grid-backed lines block).

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
    agn_nlr_cf, agn_nlr_fwhm_kms, agn_nlr_f_bol : float
        Covering fraction, narrow-line FWHM [km/s], and bolometric correction.
    neb_logU, neb_logZ_gas : float
        Photoionisation knobs forwarded to the grid adapter.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        NLR :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    Backend initialisation reads HDF5 (Python-level, not JIT-traceable); call
    once eagerly before any ``jax.jit`` over the forward model so the cached
    backend's interpolation stays JIT-safe.
    """
    del agn_log_lbol
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = jnp.asarray(l5100_disc) * agn_nlr_f_bol
    L_nu = compute_nlr_sed_synthesizer(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm_kms,
        grid_path=_resolve_synthesizer_grid("nlr"),
        neb_logU=neb_logU,
        neb_logZ_gas=neb_logZ_gas,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("lines", "blr_synthesizer")
def blr_synthesizer_lines_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_blr_cf: float = 0.1,
    agn_blr_fwhm_kms: float = 5000.0,
    agn_blr_f_bol: float = DEFAULT_F_BOL_5100,
    neb_logU: float = -1.0,
    neb_logZ_gas: float = -1.8477,
    **_params,
) -> Array:
    r"""BLR lines from the Synthesizer Cloudy grid (grid-backed lines block).

    Broad-line sibling of :func:`nlr_synthesizer_lines_block`, routing to
    :func:`compute_blr_sed_synthesizer` (default FWHM 5000 km/s). Grid path is
    resolved the same way. See that function's Notes for the JIT caveat.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        Ignored (bolometric is taken from ``l5100_disc``).
    l5100_disc : array, scalar
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` of the disc [erg/s].
    agn_blr_cf, agn_blr_fwhm_kms, agn_blr_f_bol : float
        Covering fraction, broad-line FWHM [km/s], and bolometric correction.
    neb_logU, neb_logZ_gas : float
        Photoionisation knobs forwarded to the grid adapter.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        BLR :math:`L_\lambda` [erg/s/Å].
    """
    del agn_log_lbol
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = jnp.asarray(l5100_disc) * agn_blr_f_bol
    L_nu = compute_blr_sed_synthesizer(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_blr_cf,
        fwhm_kms=agn_blr_fwhm_kms,
        grid_path=_resolve_synthesizer_grid("blr"),
        neb_logU=neb_logU,
        neb_logZ_gas=neb_logZ_gas,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
