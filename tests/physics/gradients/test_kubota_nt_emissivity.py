# SPDX-License-Identifier: BSD-3-Clause
"""Page-Thorne (1974) thin-disc emissivity of the Kubota & Done disc (#2572).

K&D 2018 and QSOSED/RELQSO take ``T_NT`` from the relativistic flux of a thin disc around a
Kerr black hole (Page & Thorne 1974, ApJ 191, 499). tengri used the Newtonian zero-torque
profile ``Rt = 1 - sqrt(r_isco/r)``, whose total dissipation is 1.46 L_bol at a=0 (it must be
``eta Mdot c^2``); that put ``R_hot`` at 20 R_g where the reference model has 45.

The reference here is the *defining* integral, independent of the closed form in
``_nt_emissivity``:  with G=M=c=1 and Mdot=1,
``F(r) = (1/4 pi r) (-Omega') (E - Omega L)^-2 int_{r_isco}^r (E - Omega L) L' dr``
for the circular-orbit ``Omega, E, L`` of Kerr (``sqrt(-g) = r`` in the equatorial plane).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from scipy.integrate import quad

jax.config.update("jax_enable_x64", True)

from tengri.components.agn import _nt_emissivity as N

pytestmark = pytest.mark.regression_paper

_SPINS = [0.0, 0.5, 0.9, 0.998]


def _isco(a):
    z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = np.sqrt(3 * a * a + z1 * z1)
    return 3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2 * z2))


def _orbit(r, a):
    """Circular-orbit ``(Omega, E, L)`` of Kerr, G=M=c=1."""
    d = np.sqrt(r**1.5 - 3 * np.sqrt(r) + 2 * a)
    omega = 1.0 / (r**1.5 + a)
    energy = (r**1.5 - 2 * np.sqrt(r) + a) / (r**0.75 * d)
    ang = (r * r - 2 * a * np.sqrt(r) + a * a) / (r**0.75 * d)
    return omega, energy, ang


def _flux_pt(r, a):
    """Page-Thorne flux F(r) [Mdot = 1, G=M=c=1] from the defining integral."""

    def dl_dr(rr):
        h = 1e-6 * rr
        return (_orbit(rr + h, a)[2] - _orbit(rr - h, a)[2]) / (2 * h)

    def integrand(rr):
        om, en, ang = _orbit(rr, a)
        return (en - om * ang) * dl_dr(rr)

    omega_r = (_orbit(r * (1 + 1e-6), a)[0] - _orbit(r * (1 - 1e-6), a)[0]) / (2e-6 * r)
    om, en, ang = _orbit(r, a)
    integral = quad(integrand, _isco(a), r, epsabs=0, epsrel=1e-12, limit=200)[0]
    return (-omega_r) * integral / (4 * np.pi * r * (en - om * ang) ** 2)


@pytest.mark.parametrize("a", _SPINS)
def test_nt_rt_matches_the_defining_page_thorne_integral(a):
    """(b) ``Rt`` equals F / (3 / 8 pi r^3) from the defining integral, rtol 1e-6."""
    ri = _isco(a)
    for x in (1.2, 1.5, 2.0, 4.0, 10.0, 50.0):
        r = x * ri
        expected = _flux_pt(r, a) / (3.0 / (8.0 * np.pi * r**3))
        np.testing.assert_allclose(float(N.nt_rt(x, a)), expected, rtol=1e-6, err_msg=f"x={x}")


@pytest.mark.parametrize("a", _SPINS)
def test_nt_h_matches_direct_integral_of_the_flux(a):
    """(b) ``h(x) = int_1^x x'^-2 Rt dx'`` by the 128-node rule equals the direct integral."""
    ri = _isco(a)
    for x_hot in (1.5, 5.0, 40.0, 1.0e3):
        direct = quad(lambda r: 4 * np.pi * r * _flux_pt(r, a), ri, x_hot * ri, epsrel=1e-11)[0]
        # L_0 = 4 pi r_isco^2 sigma T_in^4 -> 3 / (2 r_isco) in G=M=c=Mdot=1 units
        l0 = 1.5 / ri
        np.testing.assert_allclose(
            float(N.nt_h(np.log(x_hot), a)) * l0, direct, rtol=1e-6, err_msg=f"x={x_hot}"
        )


@pytest.mark.parametrize(
    ("a", "eta"), [(0.0, 0.0572), (0.5, 0.0821), (0.9, 0.1558), (0.998, 0.3209)]
)
def test_total_energy_at_infinity_is_the_standard_efficiency(a, eta):
    """(a) The relativistic dissipation, redshifted to infinity, is ``eta(a) Mdot c^2``.

    ``L_inf = int 4 pi r E(r) F(r) dr`` (E: specific energy of the circular orbit), to 1e-3,
    with the standard efficiencies 0.0572 (a=0) and 0.3209 (a=0.998). The disc-frame total
    ``int 4 pi r F dr`` (what K&D Eq. 2 / RELQSO integrate) is larger: 1.019 eta at a=0. A
    Newtonian zero-torque profile gives 1.457 eta: the defect fixed here.
    """
    ri = _isco(a)
    # tengri's Rt in G=M=c=Mdot=1 units: F = 3/(8 pi r^3) Rt
    x_max = 1.0e7

    def f_rt(r):
        return 3.0 / (8.0 * np.pi * r**3) * float(N.nt_rt(r / ri, a))

    def integral(weight):
        return quad(
            lambda lr: 4 * np.pi * np.exp(2 * lr) * f_rt(np.exp(lr)) * weight(np.exp(lr)),
            np.log(ri) + 1e-9,
            np.log(x_max * ri),
            epsrel=1e-10,
            limit=400,
        )[0]

    l_inf = integral(lambda r: _orbit(r, a)[1])
    l_disc = integral(lambda r: 1.0)
    efficiency = 1.0 - np.sqrt(1.0 - 2.0 / (3.0 * ri))
    np.testing.assert_allclose(efficiency, eta, atol=6e-4)
    np.testing.assert_allclose(l_inf, efficiency, rtol=1e-3)
    assert 1.0 < l_disc / efficiency < 1.2

    # the Newtonian zero-torque profile violates energy conservation by ~46% at a=0
    newt = quad(
        lambda lr: (
            4
            * np.pi
            * np.exp(2 * lr)
            * 3.0
            / (8.0 * np.pi * np.exp(3 * lr) * ri**0)
            * (1 - np.sqrt(ri / np.exp(lr)))
        ),
        np.log(ri),
        np.log(x_max * ri),
        epsrel=1e-10,
    )[0]
    if a == 0.0:
        assert newt / efficiency > 1.4


def test_disc_total_dissipation_is_the_page_thorne_one_not_newtonian():
    """(a) The production disc's own dissipation integral, to R = 1e4 R_isco, at a = 0.

    ``kubota_done_disc`` sets ``Mdot = L_bol / (eta c^2)``; its dissipation in the disc frame
    must then be 1.0191 L_bol (``int 4 pi r F dr`` of Page-Thorne, 1.0191 eta Mdot c^2), not
    the 1.457 L_bol of the Newtonian zero-torque profile it used before #2572.
    """
    import jax.numpy as jnp

    from tengri.components.agn import disc as D

    log_mbh, log_lbol = 8.5, 11.5
    _, _, r_isco, _eta, _log_ledd, mdot = D._compute_bh_params(
        log_mbh, log_lbol, 0.0, float32=False
    )
    t_in = (
        3.0
        * D._G_GRAV
        * 10.0**log_mbh
        * D._MSUN_G
        * mdot
        / (8.0 * jnp.pi * D._SIGMA_SB * r_isco**3)
    ) ** 0.25
    l_bol = 10.0**log_lbol * D._LSUN_ERG
    total = float(D._nt_l_diss_analytic(1.0e4, r_isco, t_in))
    np.testing.assert_allclose(total / l_bol, 1.0191, rtol=2e-3)
