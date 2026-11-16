# SPDX-License-Identifier: BSD-3-Clause
r"""Gaussian integrals over a segment, the primitives of every closed-form Gaussian readout.

A Gaussian kernel integrated against a piecewise-linear curve reduces, segment by segment,
to differences of the normal CDF and to the moments of :math:`e^{ks}` under the kernel.
These two functions are the only implementation of those integrals in tengri: the
closed-form velocity-dispersion point readout
(:func:`tengri.observation.spectrum.point_readout_gaussian_ln`) and the band projection of
velocity-broadened lines both compose them, rather than each forming its own.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import ndtr


def ndtr_diff(upper, lower):
    r"""Normal CDF difference :math:`\Phi(u) - \Phi(l)`, formed in the tail that keeps it exact.

    The mirrored form :math:`\Phi(-l) - \Phi(-u)` is used when the midpoint is positive, so the
    difference never cancels two values near one.

    Parameters
    ----------
    upper, lower : array_like
        Standardized limits :math:`u \ge l` [dimensionless], broadcastable.

    Returns
    -------
    ndarray
        :math:`\Phi(u) - \Phi(l)` [dimensionless], in the broadcast shape.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes.
    """
    direct = ndtr(upper) - ndtr(lower)
    mirrored = ndtr(-lower) - ndtr(-upper)
    return jnp.where(upper + lower > 0.0, mirrored, direct)


def gaussian_exp_moments(s_lo, s_hi, sigma):
    r"""Moments :math:`M_1, M_2` of :math:`e^{ks}` over ``[s_lo, s_hi]`` under a centered Gaussian.

    .. math::

        M_k = \int_{s_0}^{s_1} e^{ks}\,\mathcal{N}(s; 0, \sigma)\,ds
            = e^{k^2\sigma^2/2}\,\bigl[\Phi\bigl((s_1 - k\sigma^2)/\sigma\bigr)
              - \Phi\bigl((s_0 - k\sigma^2)/\sigma\bigr)\bigr],

    where :math:`s` is the offset from the kernel center (for a kernel in
    :math:`\ln\lambda`, :math:`s = \ln\lambda - \mu`), :math:`\sigma` the kernel width and
    :math:`\Phi` the normal CDF.

    Parameters
    ----------
    s_lo, s_hi : array_like
        Segment limits :math:`s_0 \le s_1` [same units as ``sigma``], broadcastable.
    sigma : array_like
        Kernel width, positive [same units as the limits], broadcastable.

    Returns
    -------
    m1, m2 : ndarray
        :math:`M_1` and :math:`M_2` [dimensionless], in the broadcast shape.

    Notes
    -----
    **JIT-compatible**: yes. **Gradient-safe**: yes for ``sigma > 0``; the caller gates the
    zero-width case.
    """
    s2 = sigma * sigma
    m1 = jnp.exp(0.5 * s2) * ndtr_diff((s_hi - s2) / sigma, (s_lo - s2) / sigma)
    m2 = jnp.exp(2.0 * s2) * ndtr_diff((s_hi - 2.0 * s2) / sigma, (s_lo - 2.0 * s2) / sigma)
    return m1, m2
