# SPDX-License-Identifier: BSD-3-Clause
r"""Faithful analytic ADAF spectrum (Mahadevan 1997).

Differentiable JAX implementation of the analytic scaling laws for an
advection-dominated accretion flow (ADAF / RIAF), following Mahadevan (1997
[1]_) equation-for-equation. The model gives the radio-to-X-ray spectrum of a
low-luminosity AGN from three cooling processes — cyclo-synchrotron,
bremsstrahlung, and inverse Compton — as closed-form functions of the black
hole mass, accretion rate, viscosity :math:`\alpha`, and plasma parameters
:math:`\beta` (gas-to-total pressure) and :math:`\delta` (electron viscous
heating fraction).

This module is the differentiable, native-JAX sibling of the template-based
Nemmen (2014) model (#898 Phase 2) and the AGNNES emulator (#952).

Conventions (Mahadevan 1997)
----------------------------
- Mass ``m = M / M_sun``; accretion rate ``mdot = Mdot / Mdot_Edd`` with
  ``Mdot_Edd = L_Edd / (eta_eff c^2)`` and ``eta_eff = 0.1`` (Eq. 4).
- ``beta = P_gas / P_total`` (magnetic fraction is ``1 - beta``; Eq. 1).
- ``delta`` = fraction of viscous energy heating electrons directly.
- Fiducial constants: ``c1 = 0.5``, ``c3 = 0.3``, ``r_min = 3``, ``r_max = 1000``.
- The ADAF solution requires ``mdot < mdot_crit ~ 0.28 alpha^2`` (Eq. 52).

References
----------
.. [1] R. Mahadevan, "Scaling Laws for Advection-dominated Flows: Applications
   to Low-Luminosity Galactic Nuclei," ApJ, 477, 585 (1997).
   arXiv:astro-ph/9609107. https://doi.org/10.1086/303727
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import i0 as _i0, i1 as _i1

from tengri.components.agn._phys import C_LIGHT as _C_LIGHT
from tengri.utils.physics_constants import (
    K_BOLTZ as _K_BOLTZ,
    M_ELECTRON as _M_ELECTRON,
)

# Fiducial self-similar constants (Mahadevan 1997, Narayan & Yi 1995b).
_C1: float = 0.5
_C3: float = 0.3
_R_MIN: float = 3.0
_R_MAX: float = 1000.0
_ETA_EFF: float = 0.1

# theta_e = k T_e / (m_e c^2)
_THETA_PER_TE: float = _K_BOLTZ / (_M_ELECTRON * _C_LIGHT**2)


# ── Modified Bessel K_2 (differentiable) ──────────────────────────────────
#
# JAX ships only Bessel I (i0/i1), so K_0, K_1 are built from the
# Abramowitz & Stegun (1964) rational approximations 9.8.5-9.8.8 (accurate to
# ~1e-7), and K_2 from the recurrence K_2(x) = K_0(x) + (2/x) K_1(x). We return
# the exponentially-scaled K_n(x) e^x to avoid overflow, mirroring SciPy's kve.


def _bessel_k0e_k1e(x: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    r"""Exponentially-scaled modified Bessel functions ``K_0(x) e^x`` and ``K_1(x) e^x``.

    Parameters
    ----------
    x : array_like
        Argument, ``x > 0``.

    Returns
    -------
    tuple of ndarray
        ``(K_0(x) e^x, K_1(x) e^x)``.

    Notes
    -----
    **JIT/grad-safe**: yes. Each branch is evaluated on a clipped argument so the
    unused branch of the ``jnp.where`` never overflows or produces a non-finite
    value that could poison the gradient.
    """
    x = jnp.asarray(x, dtype=jnp.float64)

    # Small-argument branch (0 < x <= 2): A&S 9.8.5 / 9.8.7. Clip so the
    # discarded branch stays finite for large x (i0/i1 grow ~ e^x).
    xs = jnp.clip(x, 1e-300, 2.0)
    t = (xs / 2.0) ** 2
    ln_half = jnp.log(xs / 2.0)
    k0_small = -ln_half * _i0(xs) + (
        -0.57721566
        + 0.42278420 * t
        + 0.23069756 * t**2
        + 0.03488590 * t**3
        + 0.00262698 * t**4
        + 0.00010750 * t**5
        + 0.00000740 * t**6
    )
    xk1_small = xs * ln_half * _i1(xs) + (
        1.0
        + 0.15443144 * t
        - 0.67278579 * t**2
        - 0.18156897 * t**3
        - 0.01919402 * t**4
        - 0.00110404 * t**5
        - 0.00004686 * t**6
    )
    k1_small = xk1_small / xs
    # Scale to K_n e^x.
    k0e_small = k0_small * jnp.exp(xs)
    k1e_small = k1_small * jnp.exp(xs)

    # Large-argument branch (x >= 2): A&S 9.8.6 / 9.8.8 give sqrt(x) e^x K_n.
    xl = jnp.clip(x, 2.0, jnp.inf)
    u = 2.0 / xl
    sqrt_xl = jnp.sqrt(xl)
    k0e_large = (
        1.25331414
        - 0.07832358 * u
        + 0.02189568 * u**2
        - 0.01062446 * u**3
        + 0.00587872 * u**4
        - 0.00251540 * u**5
        + 0.00053208 * u**6
    ) / sqrt_xl
    k1e_large = (
        1.25331414
        + 0.23498619 * u
        - 0.03655620 * u**2
        + 0.01504268 * u**3
        - 0.00780353 * u**4
        + 0.00325614 * u**5
        - 0.00068245 * u**6
    ) / sqrt_xl

    small = x <= 2.0
    return jnp.where(small, k0e_small, k0e_large), jnp.where(small, k1e_small, k1e_large)


def _bessel_k2e(x: jnp.ndarray) -> jnp.ndarray:
    r"""Exponentially-scaled modified Bessel function ``K_2(x) e^x``.

    Uses the recurrence :math:`K_2(x) = K_0(x) + (2/x) K_1(x)`.

    Parameters
    ----------
    x : array_like
        Argument, ``x > 0``.

    Returns
    -------
    ndarray
        ``K_2(x) e^x``.
    """
    k0e, k1e = _bessel_k0e_k1e(x)
    return k0e + (2.0 / x) * k1e


def _adaf_g_theta(t_e: jnp.ndarray) -> jnp.ndarray:
    r"""Relativistic Maxwellian factor ``g(theta_e)`` (Mahadevan 1997, Eq. 11).

    .. math::

        g(\theta_e) \equiv \frac{1}{K_2(1/\theta_e)}
            \left(2 + 2\theta_e + \frac{1}{\theta_e}\right) e^{-1/\theta_e}

    with :math:`\theta_e = k T_e / (m_e c^2)`. Enters the Coulomb ion-electron
    heating rate (Eq. 10) and hence the total ADAF luminosity (Eq. 49).

    Parameters
    ----------
    t_e : array_like
        Electron temperature [K].

    Returns
    -------
    ndarray
        ``g(theta_e)`` [dimensionless]. For the ADAF range (T_e ~ 1e9-1e10 K)
        this is ~1-7; ~7 at the high-mdot limit (T_e ~ 1.5e9 K).

    Notes
    -----
    **JIT/grad-safe**: yes.
    """
    theta = _THETA_PER_TE * jnp.asarray(t_e, dtype=jnp.float64)
    x = 1.0 / theta  # = 1/theta_e
    # g = (2 + 2 theta + 1/theta) e^{-x} / K_2(x) = numerator / (K_2(x) e^x).
    return (2.0 + 2.0 * theta + 1.0 / theta) / _bessel_k2e(x)
