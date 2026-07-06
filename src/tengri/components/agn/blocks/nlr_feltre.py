# SPDX-License-Identifier: BSD-3-Clause
r"""Feltre+2016 CLOUDY photoionization narrow-line region (NLR) block.

The most *physical* NLR model tengri offers, and the one BEAGLE uses
(Vidal-García et al. 2022). Unlike the ``analytic`` NLR (Gaussian templates
scaled to an empirical :math:`\lambda L_\lambda(5100\,\mathrm{\AA})` anchor),
this is a self-consistent photoionization calculation:

1. the accretion disc emits an ionizing continuum, a power law
   :math:`f_\nu \propto \nu^{\alpha}` (``agn_nlr_alpha_pl``);
2. its ionizing-photon rate is
   :math:`Q_{\rm H} = \int_{\nu_{\rm Ly}}^{\infty} (L_\nu / h\nu)\,d\nu`,
   derived from the accretion luminosity (``_log_qh_from_lacc``);
3. the Feltre, Charlot & Gutkin (2016) CLOUDY c13.03 grid converts
   :math:`Q_{\rm H}` into emission lines self-consistently, over the five
   physical axes :math:`(\alpha_{\rm pl}, \log U, \log n_{\rm H}, \log Z,
   \xi_d)` — the same grid BEAGLE interpolates.

Because the line luminosities are set by the disc's ionizing-photon budget,
the NLR emission *is* reprocessed disc light — so under ``agn_norm="conserving"``
the runner debits the disc for it (energy conservation is physically exact, not
an approximation).

Requires ``data/feltre_grid.h5`` (built via ``scripts/build_feltre_grid.py``);
the block skips gracefully if the grid is absent.

References
----------
.. [1] A. Feltre, S. Charlot, J. Gutkin, "Nuclear activity in galaxies: the
   effective slope of the ionizing spectrum," MNRAS 456, 3354 (2016).
   arXiv:1511.08217. https://doi.org/10.1093/mnras/stw2180
.. [2] A. Vidal-García et al., "Modelling the nebular emission from the
   narrow-line regions of AGN in BEAGLE," MNRAS (2022). arXiv:2211.13648.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._nlr_common import _C_AA_PER_S
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.nlr_cloudy import compute_nlr_sed_feltre
from tengri.utils.physics_constants import L_SUN as _L_SUN_ERG


@register_agn_block(
    "nlr",
    "feltre",
    citation=(
        "Feltre, Charlot & Gutkin 2016, MNRAS 456, 3354 (arXiv:1511.08217); "
        "BEAGLE parity (Vidal-García et al. 2022, arXiv:2211.13648)"
    ),
    status="production",
    short_doc="Feltre+2016 CLOUDY photoionization NLR (BEAGLE parity)",
)
def nlr_feltre_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_nlr_cf: float = 0.1,
    agn_nlr_fwhm_kms: float = 500.0,
    agn_nlr_alpha_pl: float = -1.7,
    neb_logU: float = -2.0,
    neb_logn: float = 3.0,
    neb_logZ_gas: float = -1.8477,
    agn_nlr_xi_d: float = 0.3,
    **_params,
) -> tuple[Array, Array]:
    r"""Physical NLR lines from the Feltre+2016 CLOUDY grid (BEAGLE parity).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}` of the intrinsic AGN bolometric luminosity
        [:math:`L_\odot`]. Sets the ionizing-photon budget that drives the
        line luminosities (isotropic illumination — ADR-0018 §3).
    l5100_disc : array, scalar
        Ignored — the Feltre NLR is normalized by the ionizing photon rate
        :math:`Q_{\rm H}` (from :math:`L_{\rm bol}`), not the disc
        :math:`\lambda L_\lambda(5100\,\mathrm{\AA})`.
    agn_nlr_cf : float, optional
        NLR covering fraction (fraction of the ionizing continuum the NLR
        intercepts). Default 0.1.
    agn_nlr_fwhm_kms : float, optional
        Narrow-line FWHM [km/s]. Default 500.
    agn_nlr_alpha_pl : float, optional
        EUV ionizing power-law slope :math:`\alpha` (:math:`f_\nu \propto
        \nu^{\alpha}`). Default −1.7 (Feltre+2016 fiducial).
    neb_logU, neb_logn, neb_logZ_gas, agn_nlr_xi_d : float, optional
        Feltre grid axes: ionization parameter, gas density [cm⁻³, log],
        gas metallicity [log Z/Z⊙], and dust-to-metal ratio.

    Returns
    -------
    (maskable, isotropic) : tuple of ndarray, shape (n_wave,)
        NLR is spatially extended → fully isotropic (bypasses the runner's
        Type-1/2 mask). The maskable channel is zero.

    Notes
    -----
    **JIT-compatible**: the grid interpolation is pure JAX once the backend is
    loaded; the HDF5 read is Python-level — call once eagerly before ``jax.jit``.
    """
    del l5100_disc  # Feltre NLR normalizes to Q_H(L_bol), not the 5100 Å anchor.
    wave_aa = jnp.asarray(wavelength)
    l_disc_bol_erg = 10.0**agn_log_lbol * _L_SUN_ERG
    L_nu = compute_nlr_sed_feltre(
        wave_aa,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_nlr_cf,
        fwhm_kms=agn_nlr_fwhm_kms,
        alpha_pl=agn_nlr_alpha_pl,
        neb_logU=neb_logU,
        neb_logn=neb_logn,
        neb_logZ_gas=neb_logZ_gas,
        xi_d=agn_nlr_xi_d,
    )
    L_lambda = L_nu * _C_AA_PER_S / wave_aa**2
    return jnp.zeros_like(L_lambda), L_lambda
