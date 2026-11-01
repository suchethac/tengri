# SPDX-License-Identifier: BSD-3-Clause
"""Zone radii of the kubota_done disc follow Kubota & Done (2018) (#3015).

K&D 2018 Sec. 2.2 defines the hot-flow radius only by the energetics: the disc is truncated
at ``R_hot``, where the Novikov-Thorne dissipation inside it equals the corona power
``L_diss,hot = 2 int_{R_isco}^{R_hot} sigma T_NT^4 2 pi R dR = f_hard L_Edd``. Sec. 2.2 also
takes the warm zone out to the outer disc (``r_warm = r_out``), and the RELQSO source
(Hagen & Done) sets ``r_h = r_out`` when the dissipation never reaches ``f_hard L_Edd``
(``"WARNING!!! Ldiss never reaches 0.02Ledd" ... "No upper limit for r_hot"``), with
``rw > rout`` clipped to ``rout``. Neither has a ``0.5 R_out`` clip on ``R_hot`` or a
``0.9 R_out`` clip on ``R_warm``.

Before #3015 tengri clipped ``R_hot <= 0.5 R_out`` and ``R_warm <= 0.9 R_out``, so the
unreachable regime (``lambda_Edd`` below ``f_hard / 0.93``) saturated at the wrong radius.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import (
    _nt_emissivity as N,
    disc as D,
)

pytestmark = pytest.mark.regression_bug

LOG_MBH = 7.0
LOG_LSUN_ERG = 33.58265  # log10(L_sun [erg/s])
LOG_LEDD_1E7_LSUN = 11.5174  # log10(L_Edd / L_sun) for 1e7 Msun, to within 1e-4 dex


def _disc_scalars(log_lbol, a_spin=0.0):
    """``(r_g, r_isco_rg, r_isco_cm, t_in, log10_l_edd)`` as ``kubota_done_disc`` forms them."""
    r_g, r_isco_rg, r_isco_cm, _eta, log10_l_edd, mdot = D._compute_bh_params(
        LOG_MBH, log_lbol, a_spin, float32=False
    )
    t_in = (
        3.0
        * D._G_GRAV
        * 10.0**LOG_MBH
        * D._MSUN_G
        * mdot
        / (8.0 * jnp.pi * D._SIGMA_SB * r_isco_cm**3)
    ) ** 0.25
    return r_g, r_isco_rg, r_isco_cm, t_in, log10_l_edd


def _l0(r_isco_cm, t_in):
    return 4.0 * jnp.pi * r_isco_cm**2 * D._SIGMA_SB * t_in**4


def _zone(f_hard, lambda_edd):
    """Zone radii ``(r_hot, r_warm, r_out)`` [cm] and the disc scalars at ``lambda_Edd``."""
    log_lbol = LOG_LEDD_1E7_LSUN + np.log10(lambda_edd)
    r_g, r_isco_rg, r_isco_cm, t_in, log10_l_edd = _disc_scalars(log_lbol)
    r_hot, r_warm, r_out = D._compute_zone_radii(
        r_g,
        r_isco_rg,
        r_isco_cm,
        t_in,
        LOG_MBH,
        log_lbol,
        f_hard,
        2.0,
        log10_l_edd,
        float32=False,
        agn_a_spin=0.0,
    )
    return float(r_hot), float(r_warm), float(r_out), r_isco_cm, t_in, float(log10_l_edd)


@pytest.mark.parametrize(
    ("f_hard", "lambda_edd"),
    [(0.02, 0.005), (0.02, 0.02), (0.1, 0.1)],
    ids=["lambda_0.005", "lambda_0.02", "f_hard_0.1_lambda_0.1"],
)
def test_unreachable_hot_flow_takes_the_whole_disc(f_hard, lambda_edd):
    """The disc cannot supply ``f_hard L_Edd``: R_hot = R_out, as RELQSO's ``rh = rout``."""
    r_hot, r_warm, r_out, r_isco, t_in, log10_l_edd = _zone(f_hard, lambda_edd)
    l0 = float(_l0(r_isco, t_in))
    dissipated_within_r_out = l0 * float(N.nt_h(jnp.log(r_out / float(r_isco)), 0.0))
    assert dissipated_within_r_out < f_hard * 10.0**log10_l_edd, "point must be unreachable"
    np.testing.assert_allclose(r_hot, r_out, rtol=1e-12)
    np.testing.assert_allclose(r_warm, r_out, rtol=1e-12)


@pytest.mark.parametrize(
    ("f_hard", "lambda_edd"),
    [(0.02, 0.05), (0.02, 0.5)],
    ids=["lambda_0.05", "lambda_0.5"],
)
def test_reachable_hot_flow_solves_the_kd18_energy_balance(f_hard, lambda_edd):
    """Reachable: ``L0 h(R_hot / R_isco) = f_hard L_Edd`` (K&D Eq. 2), with ``R_hot < R_out``."""
    r_hot, r_warm, r_out, r_isco, t_in, log10_l_edd = _zone(f_hard, lambda_edd)
    x_hot = r_hot / float(r_isco)
    l_diss = float(_l0(r_isco, t_in)) * float(N.nt_h(jnp.log(x_hot), 0.0))
    np.testing.assert_allclose(l_diss, f_hard * 10.0**log10_l_edd, rtol=1e-6)
    assert r_hot < r_out
    assert r_hot < r_warm <= r_out


def test_clip_zone_radii_admits_up_to_the_outer_radius():
    """The shared clip: R_hot and R_warm may reach R_out, and no further."""
    r_isco, r_out = 6.0, 1000.0
    r_hot, r_warm = D._clip_zone_radii(2000.0, 3000.0, r_isco, r_out)
    assert float(r_hot) == pytest.approx(r_out)
    assert float(r_warm) == pytest.approx(r_out)
    r_hot, r_warm = D._clip_zone_radii(0.1, 0.2, r_isco, r_out)
    assert float(r_hot) == pytest.approx(r_isco * 1.01)
    assert float(r_warm) == pytest.approx(r_isco * 1.01 * 1.01)
