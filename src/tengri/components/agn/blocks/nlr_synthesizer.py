# SPDX-License-Identifier: BSD-3-Clause
"""Synthesizer Cloudy grid narrow-line region (NLR) block for the composable AGN pipeline."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._nlr_common import _C_AA_PER_S, _resolve_synthesizer_grid
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.nlr_cloudy import compute_nlr_sed_synthesizer
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG


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
    unified AGN built through ``SEDModel.build`` uses the *same* photoionization
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
            Photoionization knobs forwarded to the grid adapter.

    Returns
    -------
    tuple of ndarray, shape (n_wave,)
            ``(maskable, isotropic)`` channels. NLR is isotropic.

    Notes
    -----
    Like the analytic :func:`nlr_analytic_block`, the NLR is illuminated by the
    **intrinsic** bolometric luminosity (``10**agn_log_lbol * L_sun``) and is
    therefore inclination-independent (isotropic). Backend initialization reads
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
