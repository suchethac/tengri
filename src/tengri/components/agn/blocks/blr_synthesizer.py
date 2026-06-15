# SPDX-License-Identifier: BSD-3-Clause
"""Synthesizer Cloudy grid broad-line region (BLR) block for the composable AGN pipeline."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._blr_common import (
    _C_AA_PER_S,
    DEFAULT_F_BOL_5100,
    _resolve_synthesizer_grid,
)
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.nlr_cloudy import compute_blr_sed_synthesizer


@register_agn_block(
    "blr",
    "synthesizer",
    citation="Lovell et al. 2025, arXiv:2508.03888",
    status="production",
    short_doc="Synthesizer Cloudy grid broad-line region",
)
def blr_synthesizer_block(
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
    r"""BLR lines from the Synthesizer Cloudy grid (grid-backed blr block).

    Broad-line sibling of :func:`nlr_synthesizer_block`, routing to
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
