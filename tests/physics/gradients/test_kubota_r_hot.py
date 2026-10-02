# SPDX-License-Identifier: BSD-3-Clause
"""R_hot of the Kubota & Done (2018) disc: derivative and luminosity consistency (#2572).

Two defects, one solve:

* ``_r_hot_bisect`` differentiated *through* its ``lax.scan`` bisection, which returns
  exactly 0, so R_hot carried no sensitivity to ``M_BH``, ``L_bol`` or ``f_hard``
  wherever it was not clipped. It now has an implicit-function-theorem ``custom_jvp``.
* R_hot was solved from ``f_hard * L_Edd`` while the SED's corona used
  ``min(f_hard * L_Edd, 0.5 * L_bol)``. K&D 2018 Eq. 2 *defines* R_hot through the
  dissipation of the hot flow, ``L_diss,hot = 2 int_{R_isco}^{R_hot} sigma T_NT^4 2 pi R dR``,
  so R_hot must be solved from the very L_hot the SED radiates.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.components.agn import disc as D

_LSUN_ERG = 3.828e33


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


@pytest.mark.parametrize("frac", [0.01, 0.0376, 0.114, 0.2])
def test_r_hot_bisect_derivative_matches_fd_in_the_interior(frac):
    """d R_hot / d(t_in) and d(l_target) agree with central FD where nothing is clipped."""
    _, _, r_isco, t_in, _ = _disc_scalars(8.5, 11.5)
    l0 = _l0(r_isco, t_in)
    target = frac * l0
    x_hot = float(D._r_hot_bisect(r_isco, t_in, target) / r_isco)
    print(f"frac={frac}: x_hot={x_hot:.4f} (bracket 1.001..1e4; ceiling 0.33 L0)")
    assert 1.001 < x_hot < 1.0e4 and target < 0.33 * l0

    def r_of_target(lt):
        return D._r_hot_bisect(r_isco, t_in, lt)

    def r_of_tin(t):
        return D._r_hot_bisect(r_isco, t, target)

    for fn, x0 in ((r_of_target, target), (r_of_tin, t_in)):
        ad = float(jax.grad(fn)(x0))
        h = 1e-4 * float(x0)
        fd = float((fn(x0 + h) - fn(x0 - h)) / (2.0 * h))
        assert fd != 0.0, "vacuous: R_hot does not respond"
        assert abs(ad - fd) / abs(fd) < 1e-4, f"AD={ad:.8e} FD={fd:.8e}"


def test_r_hot_bisect_derivative_is_zero_where_clipped():
    """Above the 0.33 L0 ceiling R_hot is pinned, so its derivative wrt the target is 0."""
    _, _, r_isco, t_in, _ = _disc_scalars(8.5, 11.5)
    target = 0.5 * _l0(r_isco, t_in)
    assert float(jax.grad(lambda lt: D._r_hot_bisect(r_isco, t_in, lt))(target)) == 0.0
    assert float(jax.grad(lambda t: D._r_hot_bisect(r_isco, t, target))(t_in)) == 0.0


def _eq2_l_diss(x_hot, r_isco, t_in):
    """K&D 2018 Eq. 2 by quadrature, independent of the closed form ``h(x)``.

    ``L_diss,hot = 2 int_{R_isco}^{R_hot} sigma T_NT(R)^4 2 pi R dR`` with
    ``T_NT = T_in x^-3/4 (1 - x^-1/2)^1/4`` (x = R/R_isco), trapezoid in ln R.
    """
    x = jnp.exp(jnp.linspace(0.0, jnp.log(x_hot), 200001))
    t4 = t_in**4 * x**-3.0 * (1.0 - x**-0.5)
    r = x * r_isco
    integrand = 2.0 * D._SIGMA_SB * t4 * 2.0 * jnp.pi * r * r  # dR = R d ln R
    return float(jnp.trapezoid(integrand, jnp.log(x)))


def _hot_zone(log_mbh, log_lbol, f_hard):
    """``(x_hot, l_diss_hot, l_edd_term, cap_term)`` for the radii ``kubota_done_disc`` uses."""
    r_g, r_isco_rg, r_isco, t_in, log10_l_edd = _disc_scalars(log_mbh, log_lbol)
    r_hot, _, _ = D._compute_zone_radii(
        r_g, r_isco_rg, r_isco, t_in, log_mbh, log_lbol, f_hard, 2.0, log10_l_edd, float32=False
    )
    x_hot = r_hot / r_isco
    l_diss = _eq2_l_diss(float(x_hot), r_isco, t_in)
    return (
        float(x_hot),
        float(l_diss),
        f_hard * 10.0**log10_l_edd,
        0.5 * 10.0**log_lbol * _LSUN_ERG,
    )


@pytest.mark.parametrize(
    ("f_hard", "cap_binds"), [(0.02, True), (0.005, False)], ids=["cap_binds", "cap_inactive"]
)
def test_dissipated_hot_flow_luminosity_equals_l_hot(f_hard, cap_binds):
    """K&D 2018 Eq. 2: L_diss,hot(R_hot) == the L_hot the SED radiates, at rtol 1e-6.

    At the #2572 point (log L_bol 11.5, log M_BH 8.5) ``f_hard = 0.02`` gives
    ``f_hard L_Edd = 7.95e44`` against ``0.5 L_bol = 3.14e44`` erg/s: the cap binds.
    """
    x_hot, l_diss, l_edd_term, cap_term = _hot_zone(8.5, 11.5, f_hard)
    l_hot = min(l_edd_term, cap_term)
    print(f"f_hard={f_hard}: x_hot={x_hot:.4f} L_diss={l_diss:.6e} L_hot={l_hot:.6e}")
    assert (cap_term < l_edd_term) is cap_binds
    assert 1.001 < x_hot < 1.0e4
    np.testing.assert_allclose(l_diss, l_hot, rtol=1e-6)
    # ...and it is the very value the corona is normalised to.
    np.testing.assert_allclose(
        float(D._hot_flow_luminosity(f_hard, _disc_scalars(8.5, 11.5)[4], 11.5)), l_hot, rtol=1e-12
    )


def test_closed_form_nt_integral_matches_quadrature():
    """``_nt_l_diss_analytic`` (the closed form R_hot is solved with) is Eq. 2 itself.

    Before #2572 the closed form omitted the ``R dR`` area element's factor of ``x`` and
    understated L_diss by 0.78x at x=1.5 down to 0.30x as x -> inf; the self-referential
    tests (analytic vs analytic) could not see it.
    """
    _, _, r_isco, t_in, _ = _disc_scalars(8.5, 11.5)
    for x_hot in (1.05, 1.5, 5.0, 20.0, 300.0):
        np.testing.assert_allclose(
            float(D._nt_l_diss_analytic(x_hot, r_isco, t_in)),
            _eq2_l_diss(x_hot, r_isco, t_in),
            rtol=1e-6,
        )
