# SPDX-License-Identifier: BSD-3-Clause
"""Relativistic thin-disc (Novikov-Thorne / Page-Thorne 1974) radial emissivity.

The Kubota & Done (2018) disc, and the QSOSED/RELQSO code that defines it, take the
disc surface temperature from the Page & Thorne (1974) flux of a thin, radiatively
efficient disc around a Kerr black hole of spin ``a`` (Page & Thorne 1974, ApJ 191,
499, Eq. 15n; Novikov & Thorne 1973; Krolik 1999, *Active Galactic Nuclei*, Sec 7.2)::

    sigma T^4(r) = 3 G M Mdot / (8 pi r^3) * Rt(r),          Rt = C / B

    B  = 1 - 3/r + 2 a r^(-3/2)
    C1 = 1 - sqrt(r_isco/r) - 3a/(2 sqrt(r)) ln(sqrt(r/r_isco))
    C2 = sum_{i=1..3}  3 (y_i - a)^2 / (y y_i (y_i - y_j)(y_i - y_k))
                       * ln[(y - y_i) / (y_isco - y_i)]
    C  = C1 - C2,    y = sqrt(r),  y_isco = sqrt(r_isco)        (r in units of R_g)

where ``y_1, y_2, y_3`` are the three roots of ``y^3 - 3 y + 2 a = 0``,
``y_1 = 2 cos[(arccos a - pi)/3]``, ``y_2 = 2 cos[(arccos a + pi)/3]``,
``y_3 = -2 cos(arccos(a)/3)``. This is the ``NTpars`` routine of the public RELQSO
source. ``Rt`` vanishes at the ISCO (zero torque) and tends to 1 far out, where it
reduces to the Newtonian ``3 G M Mdot / 8 pi sigma r^3``.

Two checks pin it (``tests/physics/gradients/test_kubota_nt_emissivity.py``):

* the energy radiated to infinity, ``int 4 pi r E(r) F dr`` with ``E`` the specific
  energy of the circular orbit, equals ``eta(a) Mdot c^2`` (a=0: 0.0572, a=0.998: 0.321);
* the dissipation in the *disc frame*, ``int 4 pi r F dr`` (the quantity K&D 2018 Eq. 2
  and RELQSO use to define ``R_hot``), is ``1.019 eta Mdot c^2`` at a=0.

The dissipation integral that defines ``R_hot`` has no closed form for ``a != 0`` in
these variables, so :func:`nt_h` evaluates it by a fixed 128-node Gauss-Legendre rule in
``ln r``: smooth, deterministic, differentiable in ``a`` by automatic differentiation.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

#: Highest spin the tengri disc accepts (``agn_a_spin`` is clipped to [0, 0.998]).
A_MAX = 0.998

#: Smallest spin used inside :func:`nt_rt`. At exactly ``a = 0`` the root ``y_2`` is 0 and
#: the ``(y_2 - a)^2 / y_2`` term is 0/0; in floating point it is a 1e-16 residue whose
#: derivative is wrong. The term is O(a), so the floor moves ``Rt`` by < 1e-8.
_A_FLOOR = 1.0e-9

_GL_T, _GL_W = np.polynomial.legendre.leggauss(128)

#: Upper limit of the ``R_hot`` bracket, in units of ``R_isco``.
X_MAX = 1.0e4


def isco_radius(a_spin):
    """Innermost stable circular orbit in units of R_g = GM/c^2.

    Bardeen, Press & Teukolsky (1972) formula for prograde orbits.

    Parameters
    ----------
    a_spin : float
        Dimensionless spin parameter (0 to 0.998).

    Returns
    -------
    float
        r_isco / R_g.

    Notes
    -----
    To ensure finite gradients at the Schwarzschild limit (a=0), the argument of the final
    square root is clamped to a small positive value (1e-20). The BPT72 formula has a
    gradient singularity at a=0 where (3-z1)->0, which makes
    sqrt((3-z1)*(3+z1+2*z2)) undefined in AD. The physical limit is correct (r_isco=6 for
    a=0), but the gradient path must be stabilized for JAX autodiff to work.
    """
    a = jnp.clip(a_spin, 0.0, A_MAX)
    z1 = 1.0 + (1.0 - a**2) ** (1.0 / 3.0) * ((1.0 + a) ** (1.0 / 3.0) + (1.0 - a) ** (1.0 / 3.0))
    z2 = jnp.sqrt(3.0 * a**2 + z1**2)
    sqrt_arg = jnp.maximum((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 1e-20)
    return 3.0 + z2 - jnp.sqrt(sqrt_arg)


def nt_rt(x, a_spin):
    """Page-Thorne dimensionless flux factor ``Rt(r)`` at ``x = r / r_isco``.

    ``sigma T^4 = 3 G M Mdot / (8 pi r^3) * Rt``; see the module docstring for the
    equations. ``Rt = 0`` at ``x = 1`` and ``-> 1`` as ``x -> inf``.

    Parameters
    ----------
    x : array_like
        Radius in units of the ISCO radius, ``x >= 1``.
    a_spin : float
        Dimensionless spin, clipped to [0, 0.998].

    Returns
    -------
    ndarray
        ``Rt(x)``, floored at 0 (rounding next to the ISCO can leave a -1e-17 residue).

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp``.
    """
    a = jnp.clip(a_spin, _A_FLOOR, A_MAX)
    r_isco = isco_radius(a)
    r = x * r_isco
    y = jnp.sqrt(r)
    y_isco = jnp.sqrt(r_isco)
    theta = jnp.arccos(a)
    y1 = 2.0 * jnp.cos((theta - jnp.pi) / 3.0)
    y2 = 2.0 * jnp.cos((theta + jnp.pi) / 3.0)
    y3 = -2.0 * jnp.cos(theta / 3.0)

    b = 1.0 - 3.0 / r + 2.0 * a / r**1.5
    c1 = 1.0 - y_isco / y - (3.0 * a / (2.0 * y)) * jnp.log(y / y_isco)

    def _term(ya, yb, yc):
        return (
            3.0
            * (ya - a) ** 2
            / (y * ya * (ya - yb) * (ya - yc))
            * jnp.log((y - ya) / (y_isco - ya))
        )

    c2 = _term(y1, y2, y3) + _term(y2, y1, y3) + _term(y3, y1, y2)
    return jnp.maximum((c1 - c2) / b, 0.0)


def nt_h(log_x, a_spin):
    """Normalized hot-flow dissipation ``h(x) = int_1^x x'^-2 Rt(x') dx'``.

    K&D 2018 Eq. 2, ``L_diss,hot = 2 int_{R_isco}^{R_hot} sigma T_NT^4 2 pi R dR``, equals
    ``L_0 * h(x_hot)`` with ``L_0 = 4 pi R_isco^2 sigma T_in^4`` and
    ``T_in^4 = 3 G M Mdot / (8 pi sigma R_isco^3)``. For ``Rt = 1 - x^-1/2`` (Newtonian,
    zero torque) ``h = 1/3 - 1/x + 2/(3 x^{3/2})``; here ``Rt`` is the Page-Thorne factor.

    Parameters
    ----------
    log_x : float
        ``ln(R_hot / R_isco)`` (>= 0).
    a_spin : float
        Dimensionless spin.

    Returns
    -------
    float
        ``h`` by 128-node Gauss-Legendre quadrature in ``ln x`` (integrand
        ``Rt(e^u) e^-u``).
    """
    u = 0.5 * log_x * (jnp.asarray(_GL_T) + 1.0)
    g = nt_rt(jnp.exp(u), a_spin) * jnp.exp(-u)
    return 0.5 * log_x * jnp.sum(jnp.asarray(_GL_W) * g)


def nt_dh_dlogx(log_x, a_spin):
    """``dh/d ln x = Rt(x) / x``, the exact integrand of :func:`nt_h`."""
    x = jnp.exp(log_x)
    return nt_rt(x, a_spin) / x


#: Fraction of ``h(X_MAX)`` the R_hot target is limited to (keeps the root in the bracket).
H_CEILING_FRACTION = 0.99


def nt_h_ceiling(a_spin):
    """Largest ``h`` the R_hot solve accepts: 0.99 of the disc's total, ``h(X_MAX)``."""
    return H_CEILING_FRACTION * nt_h(jnp.log(X_MAX), a_spin)
