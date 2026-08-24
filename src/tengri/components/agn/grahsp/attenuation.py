# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP bi-attenuation: SMC-like broken power-law (Prevot 1984).

Implements the ``biattenuation`` module from upstream
``JohannesBuchner/GRAHSP`` (CeCILL-v2). The galaxy and AGN components are
attenuated by **different** column densities along the line of sight:

- Galaxy components are extinguished by ``E(B-V)``.
- AGN components are extinguished by ``E(B-V) + E(B-V)-AGN``.

The attenuation curve is a broken power-law in wavelength:

.. math::

   A_{\\mathrm{SMC}}(\\lambda) = N \\left(\\frac{\\lambda}{\\lambda_b}\\right)^\\gamma,
   \\quad
   \\gamma = \\begin{cases} \\gamma_{\\mathrm{OPT}}, & \\lambda < \\lambda_b \\\\
                            \\gamma_{\\mathrm{NIR}}, & \\lambda \\geq \\lambda_b
              \\end{cases}.

The Prevot 1984 SMC defaults are :math:`\\gamma_{\\mathrm{OPT}}=-1.2`,
:math:`\\gamma_{\\mathrm{NIR}}=-3`, :math:`N=1.2`,
:math:`\\lambda_b=1100\\,\\mathrm{nm}` (Buchner+ 2024 §2.1.5).

References
----------
.. [1] Buchner, J. et al. 2024, arXiv:2405.19297, §2.1.5.
.. [2] Prevot, M. L. et al. 1984, A&A, 132, 389.
.. [3] Salvato, M. et al. 2009, ApJ, 690, 1250 (motivation for SMC).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = [
    "PREVOT_DEFAULTS",
    "attenuation_factors",
    "smc_attenuation_curve",
]

PREVOT_DEFAULTS = dict(
    opt_index=-1.2,
    nir_index=-3.0,
    norm=1.2,
    lam_break_nm=1100.0,
)
"""Default SMC-like parameters from Prevot 1984 (Buchner+ 2024 §2.1.5)."""


def smc_attenuation_curve(
    wave_nm: Array,
    opt_index: float = -1.2,
    nir_index: float = -3.0,
    norm: float = 1.2,
    lam_break_nm: float = 1100.0,
) -> Array:
    r"""Wavelength-dependent SMC-like attenuation curve.

    Parameters
    ----------
    wave_nm: array_like, shape (n_wave,)
        Wavelength grid [nm].
    opt_index: float, optional
        Power-law index :math:`\gamma_{\mathrm{OPT}}` for
        :math:`\lambda < \lambda_b`. Default ``-1.2`` (Prevot SMC).
    nir_index: float, optional
        Power-law index :math:`\gamma_{\mathrm{NIR}}` for
        :math:`\lambda \geq \lambda_b`. Default ``-3.0``.
    norm: float, optional
        Normalization :math:`N` at the break. Default ``1.2``.
    lam_break_nm: float, optional
        Break wavelength :math:`\lambda_b` [nm]. Default ``1100`` nm.

    Returns
    -------
    A: ndarray, shape (n_wave,)
        Dimensionless attenuation curve, normalized so that
        :math:`A(\lambda_b) = N`.

    Notes
    -----
    JIT/grad/vmap-compatible. Implements GRAHSP's
    ``BiAttenuationLaw.get_attenuation`` exactly (numerical agreement < 1e-12).
    """
    wave = jnp.asarray(wave_nm)
    index = jnp.where(wave < lam_break_nm, opt_index, nir_index)
    return norm * (wave / lam_break_nm) ** index


def attenuation_factors(
    wave_nm: Array,
    ebv: float,
    ebv_agn: float,
    opt_index: float = -1.2,
    nir_index: float = -3.0,
    norm: float = 1.2,
    lam_break_nm: float = 1100.0,
) -> tuple[Array, Array]:
    r"""Galaxy and AGN multiplicative attenuation factors.

    Returns ``(factor_gal, factor_agn)``, where the galaxy factor uses
    ``ebv`` and the AGN factor uses ``ebv + ebv_agn``:

    .. math::

       \mathrm{factor}_{\mathrm{gal}}(\lambda) =
           10^{-E(B-V) \cdot A_{\mathrm{SMC}}(\lambda) / 2.5},
       \quad
       \mathrm{factor}_{\mathrm{agn}}(\lambda) =
           10^{-(E(B-V) + E(B-V)_{\mathrm{AGN}}) \cdot A_{\mathrm{SMC}}(\lambda) / 2.5}.

    Parameters
    ----------
    wave_nm: array_like, shape (n_wave,)
        Wavelength grid [nm].
    ebv: float
        Galaxy line-of-sight :math:`E(B-V)` [mag].
    ebv_agn: float
        Additional AGN-only :math:`E(B-V)` [mag].
    opt_index, nir_index, norm, lam_break_nm
        See :func:`smc_attenuation_curve`.

    Returns
    -------
    factor_gal: ndarray, shape (n_wave,)
    factor_agn: ndarray, shape (n_wave,)

    Notes
    -----
    JIT-compatible. Both factors lie in :math:`(0, 1]` for non-negative
    extinctions.
    """
    curve = smc_attenuation_curve(wave_nm, opt_index, nir_index, norm, lam_break_nm)
    factor_gal = 10.0 ** (ebv * curve / -2.5)
    factor_agn = 10.0 ** ((ebv + ebv_agn) * curve / -2.5)
    return factor_gal, factor_agn
