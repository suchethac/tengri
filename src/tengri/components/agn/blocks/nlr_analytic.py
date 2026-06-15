# SPDX-License-Identifier: BSD-3-Clause
"""Analytic narrow-line region (NLR) block for the composable AGN pipeline."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._nlr_common import _C_AA_PER_S
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.nlr import compute_nlr_sed
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG


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
