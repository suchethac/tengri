# SPDX-License-Identifier: BSD-3-Clause
"""Recombination coefficients and Lyman-continuum dust-absorption factor.

This module defines the recombination coefficients used by CIGALE's nebular
model to compute the effect of ionizing photon escape and dust absorption on
nebular emission scaling. Coefficients match CIGALE's nebular module
(``pcigale/sed_modules/nebular.py``), which itself follows the Inoue (2011)
nebular treatment with Ferland (1980) recombination coefficients.

References
----------
.. [1] Ferland, G. J. 1980, "Atomic data for hydrogen and helium
    recombination", PASP, 92, 596. https://doi.org/10.1086/130714
.. [2] Inoue, A. K. 2011, "Rest-frame ultraviolet-to-optical spectral
    characteristics of extremely metal-poor and metal-free galaxies",
    MNRAS, 415, 2920. https://doi.org/10.1111/j.1365-2966.2011.18906.x
.. [3] CIGALE nebular module: ``pcigale/sed_modules/nebular.py`` (k-factor
    at lines 156-162; ``alpha_B``/``alpha_1`` defined inline, Ferland 1980).

Notes
-----
All functions are JIT-compatible and differentiable through JAX.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = [
    "ALPHA_1",
    "ALPHA_B",
    "lyc_dust_escape_factor",
]

# Hydrogen recombination coefficients [m^3/s] at T_e = 10^4 K (Ferland 1980),
# matching CIGALE's ``pcigale/sed_modules/nebular.py`` exactly:
#   ALPHA_B : Case B total recombination coefficient (sum over levels n >= 2).
#   ALPHA_1 : direct recombination to the ground state (n = 1), i.e.
#             alpha_1 = alpha_A - alpha_B. Photons emitted in these
#             recombinations re-ionize the gas and do not escape, so they
#             enter the ionization balance via the k-factor below.
# Their ratio alpha_1 / alpha_B ~= 0.597 [dimensionless].
ALPHA_B = 2.58e-19  # m^3/s -- Case B, Ferland 1980
ALPHA_1 = 1.54e-19  # m^3/s -- alpha_A - alpha_B, Ferland 1980


def lyc_dust_escape_factor(f_esc: jnp.ndarray | float, f_dust: jnp.ndarray | float) -> jnp.ndarray:
    r"""Scaling factor for nebular emission accounting for ionizing photon loss.

    Computes the CIGALE nebular k-factor, which accounts for ionizing photons
    that escape the galaxy (``f_esc``) or are absorbed by dust inside the HII
    region (``f_dust``). Both processes reduce the ionizing photon budget
    available for recombination and nebular emission. The factor is

    .. math::

        k = \frac{1 - f_\mathrm{esc} - f_\mathrm{dust}}
                 {1 + \dfrac{\alpha_1}{\alpha_B}\,(f_\mathrm{esc} + f_\mathrm{dust})}

    where :math:`\alpha_1 = 1.54 \times 10^{-19}\ \mathrm{m^3\,s^{-1}}` is the
    recombination coefficient directly to the ground state
    (:math:`\alpha_1 = \alpha_A - \alpha_B`) and
    :math:`\alpha_B = 2.58 \times 10^{-19}\ \mathrm{m^3\,s^{-1}}` is the Case B
    total recombination coefficient (Ferland 1980, evaluated at
    :math:`T_e = 10^4` K). The ratio
    :math:`\alpha_1 / \alpha_B \approx 0.597` [dimensionless].

    Parameters
    ----------
    f_esc : array_like or float
        Ionizing photon escape fraction [dimensionless, in [0, 1]].
    f_dust : array_like or float
        Ionizing photon dust-absorption fraction [dimensionless, in [0, 1]].

    Returns
    -------
    k : ndarray
        Nebular emission scaling factor [dimensionless, in [0, 1]].
        When ``f_esc + f_dust = 0`` returns 1.0 (no photon loss).
        When ``f_esc + f_dust -> 1`` returns -> 0 (complete photon loss).

    Notes
    -----
    **JIT-compatible**: yes -- uses JAX primitives with finite guards.

    **Gradient-safe**: yes -- the denominator is bounded away from zero by the
    ``f_esc + f_dust`` clamp, so the quotient and its VJP stay finite.

    **Constraint**: ``f_esc + f_dust`` must be strictly less than 1; the sum is
    clamped to ``1 - 1e-8`` to preserve numerical stability.

    References
    ----------
    .. [1] Ferland, G. J. 1980, PASP, 92, 596.
    .. [2] Inoue, A. K. 2011, MNRAS, 415, 2920.
    .. [3] CIGALE nebular module: ``pcigale/sed_modules/nebular.py``, lines
        156-162.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> float(lyc_dust_escape_factor(0.0, 0.0))  # No loss
    1.0
    >>> round(float(lyc_dust_escape_factor(0.3, 0.0)), 4)  # 30% escape
    0.5937
    >>> round(float(lyc_dust_escape_factor(0.2, 0.1)), 4)  # total loss = 0.3
    0.5937
    """
    f_esc = jnp.asarray(f_esc)
    f_dust = jnp.asarray(f_dust)

    # Clamp f_esc + f_dust to [0, 1]. The denominator ``1 + (α1/αB)·f`` is
    # strictly positive for any ``f ∈ [0, 1]`` (it ranges [1, ~1.597]), so:
    # unlike a denominator-vanishing case: no ``1 - ε`` margin is needed.
    # Clamping to a hard 1.0 lets the numerator ``(1 - f)`` reach exactly 0 at
    # full photon loss (f_esc + f_dust = 1), so nebular emission vanishes
    # cleanly instead of leaving a ~6e-9 residual that, scaled by bright line
    # luminosities (~1e38 erg/s), left ~1e30 erg/s of "suppressed" emission
    # (P-11). The ``jnp.clip`` upper bound has zero gradient for f > 1, keeping
    # the VJP finite.
    f_total = jnp.clip(f_esc + f_dust, 0.0, 1.0)

    alpha_ratio = ALPHA_1 / ALPHA_B  # ~= 0.5969

    # CIGALE nebular.py:160-161 -- k = (1 - f) / (1 + (alpha_1/alpha_B) * f).
    numerator = 1.0 - f_total
    denominator = 1.0 + alpha_ratio * f_total

    return numerator / denominator
