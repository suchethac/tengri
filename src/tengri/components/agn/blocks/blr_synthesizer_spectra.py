# SPDX-License-Identifier: BSD-3-Clause
"""Synthesizer UnifiedAGN BLR reprocessed nebular spectrum block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._blr_common import _C_AA_PER_S, _resolve_synthesizer_grid
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.nlr_cloudy import compute_nlr_sed_synthesizer_spectra
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG


@register_agn_block(
    "blr",
    "synthesizer_spectra",
    citation="Lovell et al. 2025, arXiv:2508.03888",
    status="production",
    short_doc="Synthesizer UnifiedAGN BLR reprocessed nebular spectrum",
)
def blr_synthesizer_spectra_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_blr_cf: float = 0.1,
    neb_logU: float = -1.0,
    neb_logn: float = 4.0,
    neb_logZ_gas: float = -2.0,
    **_params,
) -> Array:
    r"""BLR reprocessed nebular spectrum reproducing Synthesizer's UnifiedAGN.

    Broad-line sibling of :func:`nlr_synthesizer_spectra_block`, reading the
    BLR grid's ``/spectra/nebular`` array. Synthesizer extracts both line regions
    isotropically (grid ``cosine_inclination=0.5``), so this is returned on the
    isotropic channel to reproduce ``UnifiedAGN``'s ``blr`` component (issue #694)
    — the physically Type-2-obscured BLR is the ``blr``/``blr_synthesizer`` path.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
            Rest-frame wavelength [Å].
    agn_log_lbol : float
            log10(L_bol / L_sun) of the AGN.
    l5100_disc : array, scalar
            Ignored (bolometric is taken from ``agn_log_lbol``).
    agn_blr_cf : float
            BLR covering fraction.
    neb_logU, neb_logn, neb_logZ_gas : float
            Photoionization knobs forwarded to the grid adapter (log Z absolute).

    Returns
    -------
    tuple of ndarray, shape (n_wave,)
            ``(maskable, isotropic)`` channels; reproduces Synthesizer's isotropic BLR.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    l_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed_synthesizer_spectra(
        wave_aa,
        l_disc_bol_erg=l_bol_erg,
        covering_fraction=agn_blr_cf,
        grid_path=_resolve_synthesizer_grid("blr"),
        neb_logU=neb_logU,
        neb_logn=neb_logn,
        neb_logZ_gas=neb_logZ_gas,
        region="blr",
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    return jnp.zeros_like(L_lambda), L_lambda
