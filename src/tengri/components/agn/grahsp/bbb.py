# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP big blue bump: smooth bending power-law (Ryde 1999).

Implements the same model as the ``activatepl`` module of
``JohannesBuchner/GRAHSP`` (CeCILL-v2 license; see Notes).

The smooth bending power-law (SBPL) is a flexible phenomenological model for
the AGN UV-to-optical continuum with a soft transition between a UV slope
:math:`\\alpha_1` and an optical slope :math:`\\alpha_2` at a break wavelength
:math:`\\lambda_{\\mathrm{break}}`. The bend width is controlled by the
parameter :math:`\\Lambda`. The continuum is normalized at
:math:`\\lambda_0 = 5100\\,\\mathrm{\\AA}` by the parameter
:math:`\\lambda L_\\lambda(5100\\,\\mathrm{\\AA})` (``l5100``).

References
----------
.. [1] Buchner, J. et al. 2024, "Genuine Retrieval of the AGN Host Stellar
       Population (GRAHSP)", arXiv:2405.19297, Eq. 1.
.. [2] Ryde, F. 1999, ApJ, 511, 692,
       https://ui.adsabs.harvard.edu/abs/1999ApJ...511..692R.
       Smooth bending power-law parameterization.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["LAMBDA_5100_NM", "XRAY_FLOOR_NM", "floor_disc_xray", "sbpl_bbb"]

LAMBDA_5100_NM: float = 510.0
"""Reference wavelength :math:`\\lambda_0 = 5100\\,\\mathrm{\\AA} = 510\\,\\mathrm{nm}`."""

XRAY_FLOOR_NM: float = 12.4
"""Short-wavelength floor of the assembled GRAHSP disc [nm].

124 Angstrom = 0.1 keV is the blue edge of the alpha_ox corona (the exact
``wavelength < 124.0`` band in ``components/xray/xray.py``). GRAHSP has no
X-ray physics, so its disc must not emit below this edge; left free, the smooth
bending power law extrapolates unbounded into the X-ray and double-counts with
the separately-added corona (#1168). Applied in tengri's assembly layer only :
:func:`sbpl_bbb` stays upstream-faithful for the bit-exact parity fixtures.
"""


def floor_disc_xray(wave_nm: Array, l_lambda: Array) -> Array:
    r"""Zero the assembled GRAHSP disc below the alpha_ox corona's blue edge.

    Parameters
    ----------
    wave_nm: array_like, shape (n_wave,)
        Rest-frame wavelength grid. [nm]
    l_lambda: array_like, shape (n_wave,)
        Assembled disc spectrum on ``wave_nm`` [erg/s/nm or erg/s/Å].

    Returns
    -------
    ndarray, shape (n_wave,)
        ``l_lambda`` with every sample at ``wave_nm < XRAY_FLOOR_NM`` set to
        zero; unchanged at and above the floor.

    Notes
    -----
    JIT/grad-safe: a hard cut at a fixed grid wavelength (not a free
    parameter), mirroring the corona's own ``jnp.where`` band mask. The
    gradient w.r.t. the disc's shape parameters flows unchanged for
    ``wave_nm >= XRAY_FLOOR_NM``. See issue #1168.
    """
    return jnp.where(wave_nm >= XRAY_FLOOR_NM, l_lambda, 0.0)


def sbpl_bbb(
    wave_nm: Array,
    l5100: float,
    uvslope: float,
    plslope: float,
    plbendloc_nm: float,
    plbendwidth: float,
    cutoff_nm: float = -1.0,
) -> Array:
    r"""Smooth bending power-law big blue bump.

    .. math::

       L(\lambda) = \frac{\lambda L_\lambda(5100\,\mathrm{\AA})}
                         {\lambda_0}
                    \left(\frac{\lambda}{\lambda_0}\right)^{(\alpha_1 + \alpha_2 + 2)/2}
                    \left(\frac{e^{q} + e^{-q}}{e^{q_p} + e^{-q_p}}\right)^{\Lambda \delta\alpha/2}
                    \frac{\lambda_0}{\lambda},

    with :math:`q = \ln(\lambda/\lambda_{\mathrm{break}})/\Lambda` and
    :math:`q_p = \ln(\lambda_0/\lambda_{\mathrm{break}})/\Lambda`. The optional
    IR cutoff multiplies the SED by
    :math:`-\mathrm{expm1}(-\lambda_{\mathrm{cut}}/\lambda)` when
    ``cutoff_nm > 0``.

    Parameters
    ----------
    wave_nm: array_like, shape (n_wave,)
        Rest-frame wavelength grid [nm]. Note: GRAHSP / CIGALE use **nm**, not
        Å; the conversion factor of 510 is baked into the normalization.
    l5100: float
        :math:`\lambda L_\lambda` at 5100 Å [erg/s]. Upstream parameter name
        in the paper is ``L_AGN`` / ``lum5100A``.
    uvslope: float
        UV power-law index :math:`\alpha_1` (``activatepl`` parameter
        ``uvslope``). Default in upstream is 0.
    plslope: float
        Optical power-law index :math:`\alpha_2` (``plslope``).
        Must satisfy ``uvslope > plslope`` for a UV-bending continuum.
    plbendloc_nm: float
        Break wavelength :math:`\lambda_{\mathrm{break}}` [nm].
    plbendwidth: float
        Bend width :math:`\Lambda` (dimensionless).
    cutoff_nm: float, optional
        IR cutoff wavelength :math:`\lambda_{\mathrm{cut}}` [nm]. ``-1.0``
        disables the cutoff (default).

    Returns
    -------
    L_lambda: ndarray, shape (n_wave,)
        Specific luminosity :math:`L_\lambda` [W/nm; same as ``l5100``/nm].

    Notes
    -----
    JIT-compatible. Input ``wave_nm`` is used as a JAX array.

    Implements the same algorithm as ``pcigale/creation_modules/activatepl.py``
    in ``JohannesBuchner/GRAHSP`` (CeCILL-v2); validated with numerical
    agreement < 1e-9 relative (see ``tests/unit/components/agn/grahsp/
    test_bbb.py``).

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.agn.grahsp.bbb import sbpl_bbb
    >>> wave = jnp.linspace(100.0, 5000.0, 50)
    >>> L = sbpl_bbb(wave, 1.0e36, 0.0, -1.7, 100.0, 1.0)
    """
    wave = jnp.asarray(wave_nm)
    norm = l5100 / LAMBDA_5100_NM  # convert lambda*L_lambda to L_lambda at lambda_0
    q = jnp.log(wave / plbendloc_nm) / plbendwidth
    qpiv = jnp.log(LAMBDA_5100_NM / plbendloc_nm) / plbendwidth
    add_expo = (uvslope + plslope + 2.0) / 2.0
    sub_expo = (plslope - uvslope) / 2.0 * plbendwidth
    bbb = (
        norm
        * (wave / LAMBDA_5100_NM) ** add_expo
        * ((jnp.exp(q) + jnp.exp(-q)) / (jnp.exp(qpiv) + jnp.exp(-qpiv))) ** sub_expo
        * (LAMBDA_5100_NM / wave)
    )
    cutoff_factor = jnp.where(
        cutoff_nm > 0.0,
        -jnp.expm1(-cutoff_nm / wave),
        jnp.ones_like(wave),
    )
    return bbb * cutoff_factor
