# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP AGN emission lines + Bruhweiler+Verner 2008 FeII forest.

Implements the ``activatelines`` module from upstream
``JohannesBuchner/GRAHSP`` (CeCILL-v2). Each line is a Gaussian whose
integrated luminosity is :math:`L_{\\rm line} = r_i \\cdot L_{\\rm Hb}` where
:math:`L_{\\rm Hb} = 0.02\\,\\mathrm{l5100}` for broad lines and
:math:`0.002\\,\\mathrm{l5100}` for narrow lines, and :math:`r_i` is the
strength relative to H-beta from Mor & Netzer 2012 (Table in
``data/grahsp/grahsp_templates.h5``).

The line profile uses upstream's slightly unusual normalization
:math:`N = 510 / \\sqrt{\\pi\\sigma^2}` rather than the textbook
:math:`1/\\sqrt{2\\pi\\sigma^2}`. This is because :math:`\\mathrm{l5100}`
is :math:`\\lambda L_\\lambda` (W or erg/s) while the SED is in
:math:`L_\\lambda` (W/nm), the 510 (= 5100 Å in nm) absorbs the
:math:`\\lambda` factor; the :math:`\\sqrt{\\pi}` rather than
:math:`\\sqrt{2\\pi}` is a convention baked into upstream and matched here
exactly for fixture parity.

References
----------
.. [1] Buchner, J. et al. 2024, arXiv:2405.19297, §2.1.2.
.. [2] Netzer, H. 1990, Active Galactic Nuclei, eds. Blandford et al.
.. [3] Mor, R. & Netzer, H. 2012, MNRAS, 420, 526.
.. [4] Rakshit, S. et al. 2020, ApJS, 249, 17 (H-gamma EW).
.. [5] Bruhweiler, F. & Verner, E. 2008, ApJ, 675, 83 (FeII template).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = [
    "AGN_TYPE_BL",
    "AGN_TYPE_LINER",
    "AGN_TYPE_SY2",
    "feii_forest",
    "gaussian_lines",
]

AGN_TYPE_BL: int = 1
AGN_TYPE_SY2: int = 2
AGN_TYPE_LINER: int = 3

# Speed of light, km/s
_C_KMS: float = 299792.458
# FWHM-to-sigma conversion factor.
_FWHM_TO_SIGMA: float = 1.0 / (2.0 * (2.0 * jnp.log(2.0)) ** 0.5)
# Reference wavelength in nm (5100 Å).
_LAMBDA_5100_NM: float = 510.0
# H-beta to L(5100) ratios.
_HBETA_BROAD_RATIO: float = 0.02
_HBETA_NARROW_RATIO: float = 0.002


def _add_gaussians(
    wave_nm: Array,
    line_wave_nm: Array,
    line_lumin: Array,
    linewidth_kms: float,
) -> Array:
    """Sum of Gaussian line profiles on a shared wave grid.

    Vectorized over both ``line_wave_nm`` (lines) and ``wave_nm`` (samples)
    so the result is JIT/vmap/grad-compatible.
    """
    # Per-line widths in nm.
    width_nm = line_wave_nm * (linewidth_kms / _C_KMS)  # km/s / (km/s) = 1
    sigma = width_nm * _FWHM_TO_SIGMA  # (n_lines,)
    norm_factor = _LAMBDA_5100_NM / jnp.sqrt(jnp.pi * sigma**2)  # (n_lines,)
    # Broadcast: (n_wave, n_lines)
    diff = wave_nm[:, None] - line_wave_nm[None, :]
    shape = jnp.exp(-0.5 * diff**2 / sigma[None, :] ** 2)
    contribution = line_lumin[None, :] * shape * norm_factor[None, :]
    return contribution.sum(axis=1)


def gaussian_lines(
    wave_nm: Array,
    line_wave_nm: Array,
    line_broad: Array,
    line_narrow_sy2: Array,
    line_narrow_liner: Array,
    l5100: float,
    a_lines: float,
    linewidth_kms: float,
    agn_type: int = 1,
) -> tuple[Array, Array]:
    r"""AGN broad + narrow emission line spectrum.

    Returns ``(broad_lumin, narrow_lumin)`` such that the total emission line
    contribution is the sum.

    Parameters
    ----------
    wave_nm: array_like, shape (n_wave,)
        Output wavelength grid [nm].
    line_wave_nm: array_like, shape (n_lines,)
        Line central wavelengths [nm]. Loaded from
        ``data/grahsp/grahsp_templates.h5`` group ``netzer1990_lines``.
    line_broad, line_narrow_sy2, line_narrow_liner: array_like, shape (n_lines,)
        Line strengths relative to H-beta. From the same HDF5 group.
    l5100: float
        :math:`\lambda L_\lambda` at 5100 Å [erg/s].
    a_lines: float
        Multiplicative scale factor for line strengths (paper parameter
        ``Alines``).
    linewidth_kms: float
        FWHM of all lines [km/s].
    agn_type: {1, 2, 3}, optional
        ``1`` = broad-line AGN (Sy1/QSO; broad+narrow_sy2 + FeII enabled).
        ``2`` = Sy2 (narrow_sy2 only). ``3`` = LINER (narrow_liner only).
        Default ``1``. **static** under JIT.

    Returns
    -------
    broad_lumin: ndarray, shape (n_wave,)
        Broad-line luminosity density [erg/s/nm]. Zero for type 2 / 3.
    narrow_lumin: ndarray, shape (n_wave,)
        Narrow-line luminosity density [erg/s/nm].

    Notes
    -----
    JIT-compatible (with ``agn_type`` as a static arg). Numerical agreement
    < 1e-9 with upstream ``activatelines.add_lines`` (the same custom wave
    grid is used for fixture comparison).
    """
    l_agn = l5100 / _LAMBDA_5100_NM
    l_broad = _HBETA_BROAD_RATIO * l_agn * a_lines
    l_narrow = _HBETA_NARROW_RATIO * l_agn * a_lines
    wave_nm = jnp.asarray(wave_nm)
    line_wave_nm = jnp.asarray(line_wave_nm)
    if agn_type == AGN_TYPE_BL:
        broad_total = _add_gaussians(
            wave_nm, line_wave_nm, l_broad * jnp.asarray(line_broad), linewidth_kms
        )
        narrow_total = _add_gaussians(
            wave_nm,
            line_wave_nm,
            l_narrow * jnp.asarray(line_narrow_sy2),
            linewidth_kms,
        )
    elif agn_type == AGN_TYPE_SY2:
        broad_total = jnp.zeros_like(wave_nm)
        narrow_total = _add_gaussians(
            wave_nm,
            line_wave_nm,
            l_narrow * jnp.asarray(line_narrow_sy2),
            linewidth_kms,
        )
    elif agn_type == AGN_TYPE_LINER:
        broad_total = jnp.zeros_like(wave_nm)
        narrow_total = _add_gaussians(
            wave_nm,
            line_wave_nm,
            l_narrow * jnp.asarray(line_narrow_liner),
            linewidth_kms,
        )
    else:
        raise ValueError(f"agn_type must be 1, 2, or 3; got {agn_type!r}")
    return broad_total, narrow_total


def feii_forest(
    wave_nm: Array,
    template_wave_nm: Array,
    template_lumin: Array,
    l5100: float,
    a_lines: float,
    a_feii: float,
) -> Array:
    r"""Bruhweiler+Verner 2008 FeII forest, scaled to the broad H-beta budget.

    .. math::

       L_{\rm FeII}(\lambda) = A_{\rm FeII}\,L_{\rm Hb,broad}\,T(\lambda),

    where :math:`L_{\rm Hb,broad} = 0.02\,(\mathrm{l5100}/510)\,A_{\rm lines}`
    and :math:`T(\lambda)` is the de-redshifted template interpolated onto
    ``wave_nm``. Outside the template's support the contribution is zero.

    Parameters
    ----------
    wave_nm: array_like, shape (n_wave,)
        Output wavelength grid [nm].
    template_wave_nm, template_lumin: array_like, shape (n_template,)
        FeII template (de-redshifted from upstream's z=0.004; loaded from
        ``data/grahsp/grahsp_templates.h5`` group ``feii_bruhweiler2008``).
    l5100: float
        :math:`\lambda L_\lambda` at 5100 Å [erg/s].
    a_lines: float
        Line-strength scale factor.
    a_feii: float
        FeII strength relative to broad H-beta (paper ``AFeII``,
        reasonable range 2-10).

    Returns
    -------
    L_FeII: ndarray, shape (n_wave,)
        FeII forest contribution [erg/s/nm].

    Notes
    -----
    JIT-compatible. Uses :func:`jax.numpy.interp` with zero-padding outside
    template support: no extrapolation.
    """
    wave_nm = jnp.asarray(wave_nm)
    l_broadlines = _HBETA_BROAD_RATIO * (l5100 / _LAMBDA_5100_NM) * a_lines
    interp = jnp.interp(
        wave_nm,
        jnp.asarray(template_wave_nm),
        jnp.asarray(template_lumin),
        left=0.0,
        right=0.0,
    )
    return interp * a_feii * l_broadlines
