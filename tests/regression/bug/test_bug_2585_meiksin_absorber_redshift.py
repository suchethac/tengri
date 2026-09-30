# SPDX-License-Identifier: BSD-3-Clause
"""Meiksin (2006) Lyman-series terms evaluated at absorber redshift, not source redshift (#2585).

The n >= 3 terms in Meiksin 2006 must use tau_alpha evaluated at the absorber
redshift z_n, not at the source redshift z. Table 1 of the paper gives the
ratios tau_n/tau_alpha as functions of the absorber redshift (the (1+z) powers
"factor out the redshift dependence" of tau_alpha itself), so every term must
be evaluated at z_n — as the n = 2 term already is correctly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate

from tengri.components.igm.meiksin06 import igm_transmission_meiksin06

pytestmark = pytest.mark.regression_bug


# ── Reference implementation from the issue reproducer ────────────────────
FACT = {3: 0.348, 4: 0.179, 5: 0.109, 6: 0.0722, 7: 0.0508, 8: 0.0373, 9: 0.0283}


def lam_n(n):
    """Lyman wavelength for transition n."""
    return 912.0 / (1 - 1 / n**2)


def tau_alpha(z):
    """Meiksin 2006 Eq. 2-3."""
    return 0.00211 * (1 + z) ** 3.7 if z < 4 else 0.00058 * (1 + z) ** 4.5


def tau_n_old(n, zn, z):
    """Old (wrong) tau_n implementation from the issue (evaluates n >= 3 at z, not z_n)."""
    if n == 2:
        return 0.00211 * (1 + zn) ** 3.7 if z <= 4 else 0.00058 * (1 + zn) ** 4.5
    base = tau_alpha(z)  # <-- WRONG: source redshift, not absorber redshift
    if n <= 9:
        return base * FACT[n] * (0.25 * (1 + zn)) ** (1 / 3 if (zn < 3 or n > 5) else 1 / 6)
    return tau_n_old(9, zn, z) * 720 / (n * (n * n - 1))


def tau_n_correct(n, zn, z, at_zn):
    """Correct tau_n (evaluates n >= 3 at z_n, not z)."""
    base = tau_alpha(zn) if at_zn else tau_alpha(z)
    if n == 2:
        if at_zn:
            return tau_alpha(zn)
        return 0.00211 * (1 + zn) ** 3.7 if z <= 4 else 0.00058 * (1 + zn) ** 4.5
    if n <= 9:
        exponent = 1 / 3 if (zn < 3 or n > 5) else 1 / 6
        return base * FACT[n] * (0.25 * (1 + zn)) ** exponent
    return tau_n_correct(9, zn, z, at_zn) * 720 / (n * (n * n - 1))


def tau_lines_old(lobs, z):
    """Old tau_lines (evaluates n >= 3 at source redshift)."""
    return sum(
        tau_n_old(n, lobs / lam_n(n) - 1, z) for n in range(2, 32) if 0 <= lobs / lam_n(n) - 1 < z
    )


def tau_lines_correct(lobs, z, at_zn):
    """Correct tau_lines with at_zn=True."""
    return sum(
        tau_n_correct(n, lobs / lam_n(n) - 1, z, at_zn)
        for n in range(2, 32)
        if 0 <= lobs / lam_n(n) - 1 < z
    )


def tau_ligm(lobs, z):
    """Lyman-alpha forest continuum opacity."""
    zL = lobs / 912.0 - 1
    return 0.805 * (1 + zL) ** 3 * (1 / (1 + zL) - 1 / (1 + z)) if zL < z else 0.0


def tau_lls(lobs, z, N0=0.25, beta=1.5, gam=1.5):
    """Lyman-Limit Systems opacity (Eq. 6, numerical)."""
    zL = lobs / 912.0 - 1
    if zL >= z:
        return 0.0

    def inner(zp):
        """Integrand for LLS opacity."""

        def integrand(t):
            return N0 * (beta - 1) * t ** (-beta) * (1 - np.exp(-t * ((1 + zL) / (1 + zp)) ** 3))

        return integrate.quad(integrand, 1, np.inf, limit=200)[0] * (1 + zp) ** gam

    return integrate.quad(inner, zL, z, limit=200)[0]


def transmission_correct(lobs, z):
    """Correct transmission (at_zn=True)."""
    return np.exp(-(tau_lines_correct(lobs, z, True) + tau_ligm(lobs, z) + tau_lls(lobs, z)))


# ── Cell 1a: Table 2 reference values at observed 1730 A ───────────────────
@pytest.mark.parametrize(
    "z,expected",
    [
        (1.5, 0.323241),
        (2.0, 0.150055),
        (2.5, 0.074558),
        (3.0, 0.038532),
        (3.5, 0.020407),
        (4.0, 0.010983),
        (4.5, 0.005976),
        (5.0, 0.003277),
        (5.5, 0.001807),
        (6.0, 0.001001),
        (6.5, 0.000556),
        (7.0, 0.000310),
    ],
)
def test_meiksin06_table2_at_1730_observed(z, expected):
    """Test transmission at observed 1730 Å equals Meiksin 2006 Table 2 to 3e-3."""
    T = float(igm_transmission_meiksin06(jnp.asarray([1730.0]), z)[0])
    assert abs(T - expected) < 3e-3 * abs(expected), (
        f"z={z}: tengri {T:.6f}, paper {expected:.6f}, rel_err={T / expected - 1:.2e}"
    )


# ── Cell 1d: Gradient ────────────────────────────────────────────────────
def test_meiksin06_grad_wrt_z():
    """Test that jax.grad of transmission w.r.t. z is finite at (1730 Å, z=3)."""

    def T_fn(z_val):
        return igm_transmission_meiksin06(jnp.asarray([1730.0]), z_val)[0]

    z_test = 3.0
    grad_T = jax.grad(T_fn)(z_test)
    assert jnp.isfinite(grad_T), f"Gradient is {grad_T}"
