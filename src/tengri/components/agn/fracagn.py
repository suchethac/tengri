# SPDX-License-Identifier: BSD-3-Clause
"""Utilities for fracAGN parameter conversion.

Thin wrappers for converting between the fracAGN parameter (bolometric
luminosity fraction from AGN) and the standard log_lbol parameter used in
the AGN module.

These functions are not primary model parameters but rather helper utilities
for bridging different parameterizations in AGN SED fitting.

.. note::
    Currently unwired (no importers). Kept deliberately: the AGN energy-ledger
    design (issue #929, Phase-2b investigation) explicitly defers a
    user-facing ``fagn`` parameterization, and these conversions are its
    natural seed. The energy-ledger stream owns the wire-or-delete decision :
    do not remove this module in unrelated dead-code sweeps (2026-07 audit).

References
----------

- Boquien et al. 2019, A&A, 622, A103 (CIGALE SED fitting)

"""

from __future__ import annotations

import jax.numpy as jnp


def fracagn_to_log_lbol(
    frac_agn: float,
    l_dust: float,
) -> float:
    """Convert fracAGN to log10(L_bol/L_sun) given dust luminosity.

    Maps the AGN bolometric luminosity fraction parameter (commonly used in
    SED fitting as a nuisance parameter) to the standard logarithmic bolometric
    luminosity scale. Given the dust luminosity (typically from stellar+AGN
    radiation absorbed and reemitted in the IR), the AGN luminosity is:

    .. math::

        L_{\\rm AGN} = L_{\\rm dust} \\times \\frac{f_{\\rm AGN}}{1 - f_{\\rm AGN}}

    This conversion is useful when the fit parameterization is in terms of
    the AGN fraction rather than the AGN luminosity directly.

    Parameters
    ----------
    frac_agn: float
        Fraction of total luminosity from AGN. Range: [0, 1).
        0 = no AGN, approaching 1 = pure AGN. [dimensionless]
    l_dust: float
        Dust luminosity (observed or modeled total) in solar luminosities.
        [L_sun]

    Returns
    -------
    log_lbol: float
        Logarithmic AGN bolometric luminosity [log10(L_sun)].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives.

    The relationship between fracAGN and luminosity ratio is:

    .. math::

        \\frac{L_{\\rm AGN}}{L_{\\rm dust}} = \\frac{f_{\\rm AGN}}{1 - f_{\\rm AGN}}

    This assumes that the dust luminosity is the total luminosity (stellar +
    AGN). Rearranging to solve for L_AGN:

    .. math::

        L_{\\rm AGN} = L_{\\rm dust} \\times \\frac{f_{\\rm AGN}}{1 - f_{\\rm AGN}}
    """
    # Avoid division by zero
    frac_safe = jnp.clip(frac_agn, 0.0, 0.9999)
    l_agn = l_dust * frac_safe / (1.0 - frac_safe)
    log_lbol = jnp.log10(l_agn + 1e-100)
    return log_lbol


def log_lbol_to_fracagn(
    log_lbol: float,
    l_dust: float,
) -> float:
    """Convert log10(L_bol/L_sun) to fracAGN given dust luminosity.

    Inverse of :func:`fracagn_to_log_lbol`. Maps the standard logarithmic
    bolometric luminosity parameter to the AGN bolometric luminosity fraction.

    Parameters
    ----------
    log_lbol: float
        Logarithmic AGN bolometric luminosity [log10(L_sun)].
    l_dust: float
        Dust luminosity (observed or modeled total) in solar luminosities.
        [L_sun]

    Returns
    -------
    frac_agn: float
        AGN bolometric luminosity fraction. [dimensionless, 0–1)

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp`` primitives.

    Derived from the inverse transformation:

    .. math::

        f_{\\rm AGN} = \\frac{L_{\\rm AGN}}{L_{\\rm AGN} + L_{\\rm dust}}
                     = \\frac{L_{\\rm AGN}}{L_{\\rm total}}
    """
    l_agn = 10.0**log_lbol
    l_total = l_agn + l_dust
    l_total_safe = jnp.maximum(l_total, 1e-100)
    frac_agn = l_agn / l_total_safe
    return jnp.clip(frac_agn, 0.0, 0.9999)
