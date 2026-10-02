# SPDX-License-Identifier: BSD-3-Clause
"""R_hot of the Kubota & Done (2018) disc: derivative and luminosity consistency (#2572).

* ``_r_hot_bisect`` is differentiated by the implicit-function theorem (a ``custom_jvp``);
  differentiating through its ``lax.scan`` bisection returned exactly 0.
* K&D 2018 Sec 4.3 ("Ldiss,hot = 0.02 LEdd, which defines rhot", Eq. 2) and the QSOSED /
  RELQSO source define ``R_hot`` through the dissipation of the hot flow,
  ``L_diss,hot = 2 int_{R_isco}^{R_hot} sigma T_NT^4 2 pi R dR``, with the Page-Thorne
  emissivity: ``R_hot`` is solved from the very ``L_hot = f_hard L_Edd`` the SED radiates,
  with no ``L_bol``-dependent cap, and for any spin.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.agn import (
    _nt_emissivity as N,
    disc as D,
)

pytestmark = pytest.mark.gradient


def _disc_scalars(log_mbh, log_lbol, a_spin=0.0):
    """``(r_g, r_isco_rg, r_isco_cm, t_in, log10_l_edd)`` as ``kubota_done_disc`` forms them."""
    r_g, r_isco_rg, r_isco_cm, _eta, log10_l_edd, mdot = D._compute_bh_params(
        log_mbh, log_lbol, a_spin, float32=False
    )
    t_in = (
        3.0
        * D._G_GRAV
        * 10.0**log_mbh
        * D._MSUN_G
        * mdot
        / (8.0 * jnp.pi * D._SIGMA_SB * r_isco_cm**3)
    ) ** 0.25
    return r_g, r_isco_rg, r_isco_cm, t_in, log10_l_edd


def _l0(r_isco_cm, t_in):
    return 4.0 * jnp.pi * r_isco_cm**2 * D._SIGMA_SB * t_in**4


@pytest.mark.parametrize("a_spin", [0.0, 0.7])
@pytest.mark.parametrize("frac", [0.01, 0.0376, 0.1])
def test_r_hot_bisect_derivative_matches_fd_in_the_interior(frac, a_spin):
    """dR_hot/d(t_in, l_target, a_spin) agree with central FD where nothing is clipped."""
    _, _, r_isco, t_in, _ = _disc_scalars(8.5, 11.5, a_spin)
    l0 = _l0(r_isco, t_in)
    target = frac * l0
    x_hot = float(D._r_hot_bisect(r_isco, t_in, target, a_spin=a_spin) / r_isco)
    print(f"a={a_spin} frac={frac}: x_hot={x_hot:.4f}")
    assert 1.001 < x_hot < 1.0e4 and frac < float(N.nt_h_ceiling(a_spin))

    cases = (
        (lambda lt: D._r_hot_bisect(r_isco, t_in, lt, a_spin=a_spin), target),
        (lambda t: D._r_hot_bisect(r_isco, t, target, a_spin=a_spin), t_in),
        (lambda a: D._r_hot_bisect(r_isco, t_in, target, a_spin=a), a_spin + 0.05),
    )
    for fn, x0 in cases:
        ad = float(jax.grad(fn)(x0))
        h = 1e-4 * max(abs(float(x0)), 1e-3)
        fd = float((fn(x0 + h) - fn(x0 - h)) / (2.0 * h))
        assert fd != 0.0, "vacuous: R_hot does not respond"
        assert abs(ad - fd) / abs(fd) < 1e-4, f"AD={ad:.8e} FD={fd:.8e}"


def test_r_hot_bisect_derivative_is_zero_where_clipped():
    """Above the ceiling R_hot is pinned, so its derivative wrt the target is 0."""
    _, _, r_isco, t_in, _ = _disc_scalars(8.5, 11.5)
    target = 0.5 * _l0(r_isco, t_in)
    assert float(jax.grad(lambda lt: D._r_hot_bisect(r_isco, t_in, lt))(target)) == 0.0
    assert float(jax.grad(lambda t: D._r_hot_bisect(r_isco, t, target))(t_in)) == 0.0


def _eq2_l_diss(x_hot, r_isco, t_in, a_spin):
    """K&D 2018 Eq. 2 by dense trapezoid quadrature in ln R (independent of ``nt_h``'s rule).

    ``L_diss,hot = 2 int_{R_isco}^{R_hot} sigma T_NT(R)^4 2 pi R dR``,
    ``sigma T_NT^4 = sigma T_in^4 x^-3 Rt(x; a)`` with the Page-Thorne factor ``Rt``.
    """
    x = jnp.exp(jnp.linspace(0.0, jnp.log(x_hot), 400001))
    t4 = t_in**4 * x**-3.0 * N.nt_rt(x, a_spin)
    r = x * r_isco
    integrand = 2.0 * D._SIGMA_SB * t4 * 2.0 * jnp.pi * r * r  # dR = R d ln R
    return float(jnp.trapezoid(integrand, jnp.log(x)))


def _hot_zone(log_mbh, log_lbol, f_hard, a_spin=0.0):
    """``(x_hot, x_out, l_diss_hot, f_hard L_Edd, ceiling luminosity)`` as the disc forms them."""
    r_g, r_isco_rg, r_isco, t_in, log10_l_edd = _disc_scalars(log_mbh, log_lbol, a_spin)
    r_hot, _, r_out = D._compute_zone_radii(
        r_g,
        r_isco_rg,
        r_isco,
        t_in,
        log_mbh,
        log_lbol,
        f_hard,
        2.0,
        log10_l_edd,
        float32=False,
        agn_a_spin=a_spin,
    )
    x_hot = float(r_hot / r_isco)
    l_diss = _eq2_l_diss(x_hot, r_isco, t_in, a_spin)
    reach = float(_l0(r_isco, t_in) * N.nt_h_ceiling(a_spin))
    return x_hot, float(r_out / r_isco), l_diss, f_hard * 10.0**log10_l_edd, reach


@pytest.mark.parametrize(
    ("log_lbol", "f_hard", "a_spin"),
    [(11.5, 0.02, 0.0), (11.5, 0.005, 0.0), (12.3, 0.02, 0.0), (12.3, 0.02, 0.7)],
    ids=["issue_point", "f_hard_0.005", "lambda_0.2", "spin_0.7"],
)
def test_dissipated_hot_flow_luminosity_is_f_hard_l_edd(log_lbol, f_hard, a_spin):
    """K&D 2018 Eq. 2 with L_diss,hot = f_hard L_Edd ("Ldiss,hot = 0.02 LEdd, which defines
    rhot"; QSOSED/RELQSO source agrees): the quadrature of Eq. 2 at R_hot equals the
    luminosity the SED radiates, rtol 1e-6, with NO L_bol/2 cap and for any spin.

    (11.5, 0.02) is lambda_Edd ~ 0.03, the paper's grid edge, where the invented cap used to
    bind (f L_Edd = 7.95e44 vs 0.5 L_bol = 6.05e44 erg/s).
    """
    x_hot, x_out, l_diss, l_edd_term, reach = _hot_zone(8.5, log_lbol, f_hard, a_spin)
    print(f"L_bol={log_lbol} f={f_hard} a={a_spin}: x_hot={x_hot:.4f} L_diss={l_diss:.6e}")
    assert l_edd_term < reach, "point must be in the reachable regime"
    assert 1.001 < x_hot < 0.5 * x_out
    np.testing.assert_allclose(l_diss, l_edd_term, rtol=1e-6)
    _, _, r_isco, t_in, ledd = _disc_scalars(8.5, log_lbol, a_spin)
    l0 = float(_l0(r_isco, t_in))
    np.testing.assert_allclose(
        float(D._hot_flow_luminosity(f_hard, ledd, l0, a_spin)), l_edd_term, rtol=1e-12
    )


def test_unreachable_hot_flow_saturates_instead_of_inventing_a_cap():
    """When the disc cannot supply f_hard L_Edd (lambda_Edd << 0.02) the whole flow is hot
    (RELQSO: "Ldiss never reaches 0.02Ledd => No upper limit for r_hot", rh = rout).

    tengri keeps static zone shapes, so L_hot saturates at ``L0 h_ceiling`` (99% of the disc's
    total) and R_hot sits at its zone ceiling 0.5 R_out; no L_bol-dependent switch.
    """
    x_hot, x_out, _, l_edd_term, reach = _hot_zone(8.5, 9.0, 0.02)
    assert l_edd_term > reach
    _, _, r_isco, t_in, ledd = _disc_scalars(8.5, 9.0)
    l0 = float(_l0(r_isco, t_in))
    np.testing.assert_allclose(float(D._hot_flow_luminosity(0.02, ledd, l0)), reach, rtol=1e-12)
    np.testing.assert_allclose(x_hot, 0.5 * x_out, rtol=1e-12)


@pytest.mark.parametrize("a_spin", [0.0, 0.5, 0.998])
def test_nt_l_diss_matches_quadrature(a_spin):
    """``_nt_l_diss_analytic`` (the integral R_hot is solved with) is Eq. 2 itself."""
    _, _, r_isco, t_in, _ = _disc_scalars(8.5, 11.5, a_spin)
    for x_hot in (1.05, 1.5, 5.0, 20.0, 300.0):
        np.testing.assert_allclose(
            float(D._nt_l_diss_analytic(x_hot, r_isco, t_in, a_spin)),
            _eq2_l_diss(x_hot, r_isco, t_in, a_spin),
            rtol=1e-6,
        )
