# SPDX-License-Identifier: BSD-3-Clause
"""``R_hot`` of the Kubota & Done (2018) disc is solved to round-off, not to a bracket.

``L_diss,hot(R_hot) = L_hot`` (Eq. 2) defines ``R_hot``; a 40-step bisection over the 9.2
units of ``ln x`` leaves the root known to 4e-12, and the SED depends on ``R_hot`` steeply
enough that this quantizes it into steps of about 1e-6 in its input parameters, which a
finite difference sees as noise and a JIT/eager comparison as a mismatch. The root is
polished to the residual of the dissipation integral's own evaluation.
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

pytestmark = pytest.mark.limit

#: Residual of the root in ``ln x``: ``|F(u)| / F'(u)`` with ``F = L0 h(x) - L_hot``.
_ROUND_OFF = 2e-14


def _case(log_mbh, lambda_edd, f_hard, a_spin):
    _, _, _, _, log10_l_edd, _ = D._compute_bh_params(log_mbh, 11.0, a_spin, float32=False)
    log_lbol = float(log10_l_edd + np.log10(lambda_edd) - D._LOG10_LSUN_ERG)
    _, _, r_isco_cm, _eta, log10_l_edd, mdot = D._compute_bh_params(
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
    l0 = 4.0 * jnp.pi * r_isco_cm**2 * D._SIGMA_SB * t_in**4
    return l0, f_hard * 10.0**log10_l_edd


@pytest.mark.parametrize("a_spin", [0.0, 0.7])
@pytest.mark.parametrize("f_hard", [0.02, 0.05, 0.1])
@pytest.mark.parametrize("lambda_edd", [1e-3, 0.03, 0.3, 1.0, 3.0])
@pytest.mark.parametrize("log_mbh", [6.0, 7.5, 9.0, 10.0])
def test_r_hot_residual_is_at_round_off(log_mbh, lambda_edd, f_hard, a_spin):
    l0, target = _case(log_mbh, lambda_edd, f_hard, a_spin)
    floor = l0 * N.nt_h(jnp.log(D._X_LO), a_spin)
    ceiling = l0 * D._nt_h_ceiling(a_spin)
    if not (floor < target < ceiling):
        pytest.skip("R_hot is pinned to a bracket end here (no interior root)")
    u = D._bisect_log_x(target, l0, a_spin, 40)
    residual = abs(float(l0 * N.nt_h(u, a_spin) - target)) / float(l0 * N.nt_dh_dlogx(u, a_spin))
    assert residual < _ROUND_OFF, (
        f"log M={log_mbh}, lambda_Edd={lambda_edd}, f_hard={f_hard}, a={a_spin}: "
        f"|F|/F' = {residual:.2e} in ln x"
    )
