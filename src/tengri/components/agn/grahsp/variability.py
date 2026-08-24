# SPDX-License-Identifier: BSD-3-Clause
"""GRAHSP AGN variability uncertainty (Simm+ 2016 NEV).

Implements Eq. NEV from Buchner+ 2024 (arXiv:2405.19297, §2.3.2):

.. math::

   F_{\\rm var}^2 = \\mathrm{NEV} = \\min(0.1, 10^{-1.43 - 0.74\\,l_{45}}),

where :math:`l_{45} = \\log_{10}(L_{\\rm bol}^{\\rm BBB} / 10^{45}\\,\\mathrm{erg/s})`.
The fractional variability adds in quadrature to measurement and systematic
uncertainties via Bienaymé's identity (paper Eq. errorbudget).

References
----------
.. [1] Buchner, J. et al. 2024, arXiv:2405.19297, §2.3.2 Eq. NEV.
.. [2] Simm, T. et al. 2016, A&A, 585, A129 (Pan-STARRS1 NEV calibration).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

__all__ = ["normalized_excess_variance"]


def normalized_excess_variance(L_bol_BBB: Array | float) -> Array:
    r"""Pan-STARRS1-calibrated AGN normalized excess variance.

    Parameters
    ----------
    L_bol_BBB: array_like or float
        Bolometric BBB luminosity [erg/s] (the upper-limit-of-91.2-nm-integrated
        AGN BBB component, see :mod:`bbb`). Must be strictly positive.

    Returns
    -------
    NEV: ndarray or float
        Fractional variance :math:`F_{\rm var}^2`, capped at 0.1.
        :math:`F_{\rm var} = \sqrt{\mathrm{NEV}}` is the equivalent
        fractional 1-sigma flux scatter to add in quadrature.

    Notes
    -----
    JIT/grad/vmap-compatible. The 0.1 cap reflects the Simm+ 2016 saturation
    at low luminosity (~30% fractional flux variability).

    Examples
    --------
    >>> from tengri.components.agn.grahsp.variability import (
    ...     normalized_excess_variance,
    ... )
    >>> float(normalized_excess_variance(1.0e45))
    0.0371...
    >>> float(normalized_excess_variance(1.0e40))  # capped
    0.1
    """
    L = jnp.asarray(L_bol_BBB)
    l45 = jnp.log10(L) - 45.0
    raw = 10.0 ** (-1.43 - 0.74 * l45)
    return jnp.minimum(0.1, raw)
