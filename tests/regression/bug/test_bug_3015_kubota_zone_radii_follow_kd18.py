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

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.physics_constants import L_SUN

from tengri.components.agn import (
    _nt_emissivity as N,
    disc as D,
)

pytestmark = pytest.mark.regression_bug

LOG_MBH = 7.0
LOG_LSUN_ERG = float(np.log10(L_SUN))  # log10(L_sun [erg/s]), exact to the code's constant
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
    # R_warm is floored at R_hot itself (no 1.01 factor): a zero-width warm zone is allowed.
    assert float(r_warm) == pytest.approx(r_isco * 1.01)


def test_clip_zone_radii_allows_zero_width_zones():
    """R_hot = R_out gives R_warm = R_out too, with no floor above R_out (#3015)."""
    r_hot, r_warm = D._clip_zone_radii(5000.0, 5000.0, 6.0, 1000.0)
    assert float(r_hot) == pytest.approx(1000.0)
    assert float(r_warm) == pytest.approx(1000.0)


#: Hot-flow cap at the ceiling X_MAX, the disc's total dissipation h(inf) (K&D 2018 Eq. 2).
_X_MAX = float(D._NT_X_MAX)
_LOG_MBHS = (7.0, 8.0)
_WAVE_CHECK = jnp.asarray([3000.0, 6.1992])  # 3000 A and 2 keV [A]


@pytest.mark.parametrize("log_mbh", _LOG_MBHS)
def test_sed_is_finite_with_finite_gradient_across_the_switch(log_mbh):
    """Finite L_nu and d L_nu / d agn_log_lbol on both sides of the reachable/unreachable switch.

    Grid ``agn_log_lbol`` 8-14 in steps of 0.05 dex: the switch sits near 9.5 dex at
    ``mbh = 7`` and at 10.5 dex at ``mbh = 8``, so the grid crosses it at both masses.
    """

    def l_nu(log_lbol):
        return D.kubota_done_disc(_WAVE_CHECK, agn_log_lbol=log_lbol, agn_log_mbh=log_mbh)

    value_and_grad = jax.jit(jax.value_and_grad(lambda lb: l_nu(lb)[0], argnums=0))
    value = jax.jit(l_nu)
    for log_lbol in np.arange(8.0, 14.0 + 1e-9, 0.05):
        lb = jnp.asarray(float(log_lbol))
        nu_values = np.asarray(value(lb))
        assert np.all(np.isfinite(nu_values)), (log_lbol, nu_values)
        _, grad = value_and_grad(lb)
        assert np.isfinite(float(grad)), log_lbol


def test_unreachable_regime_emits_no_disc_or_warm_light_and_corona_is_the_disc_share():
    """Unreachable (lambda_Edd = 0.01): disc + warm emit exactly 0; corona = L0 h(R_out/R_isco)."""
    lambda_edd = 0.01
    r_hot, _r_warm, r_out, r_isco, _t_in, _l_edd = _zone(0.02, lambda_edd)
    assert r_hot == pytest.approx(r_out, rel=1e-12)
    _total, l_nu_disc, _l_nu_hot, corona_fraction = D.kubota_done_disc(
        _WAVE_CHECK[:1],
        agn_log_lbol=LOG_LEDD_1E7_LSUN + np.log10(lambda_edd),
        agn_log_mbh=LOG_MBH,
        agn_f_hard=0.02,
        _return_parts=True,
    )
    assert np.max(np.abs(np.asarray(l_nu_disc))) == 0.0
    x_out = r_out / r_isco
    h_out_over_h_inf = float(N.nt_h(jnp.log(x_out), 0.0)) / float(N.nt_h(jnp.log(_X_MAX), 0.0))
    np.testing.assert_allclose(float(corona_fraction), h_out_over_h_inf, rtol=1e-9)


_SELECTORS = {
    "agn_disc_block": "kubota_done",
    "agn_nlr_block": "none",
    "agn_blr_block": "none",
    "agn_feii_block": "none",
    "agn_torus_block": "none",
    "agn_attenuation_block": "none",
}
_LEDGER_WAVE = jnp.asarray(np.geomspace(100.0, 1.0e7, 400))
_LEDGER_RTOL = 1.0e-6
_PIECE_KEYS = ("log_L_agn_disc", "log_L_agn_torus", "log_L_agn_lines", "log_L_agn_polar_dust")
_UNEMITTED_KEYS = (
    "log_L_agn_disc_unemitted",
    "log_L_agn_torus_unemitted",
    "log_L_agn_lines_unemitted",
)


def _ledger(log_lbol):
    """The composable runner's published ledger for a kubota_done disc alone [dex re erg/s]."""
    from tengri.components.agn.blocks._protocol import collect_block_templates
    from tengri.components.agn.blocks.runner import composable_agn_l_nu

    templates = collect_block_templates(
        {k[len("agn_") : -len("_block")]: v for k, v in _SELECTORS.items()}
    )
    _, comps = composable_agn_l_nu(
        _LEDGER_WAVE,
        log_lbol,
        return_components=True,
        return_ledger=True,
        template_state=templates,
        strict_keys=False,
        agn_norm="conserving",
        agn_log_mbh=LOG_MBH,
        **_SELECTORS,
    )
    return {k: float(np.asarray(v)) for k, v in comps.items() if np.ndim(v) == 0}


@pytest.mark.parametrize(
    "log_lbol", [9.5, 11.5], ids=["unreachable_lbol9.5", "reachable_lbol11.5"]
)
def test_ledger_closes_with_the_unemitted_remainder(log_lbol):
    """C2 on the disc alone: emitted pieces + unemitted pieces = L_bol, to 1e-6 (#3015).

    Both regimes: at lbol 9.5 (lambda_Edd = 0.01) the disc and warm zones are empty and the
    corona takes the disc's share inside R_out; at lbol 11.5 (lambda_Edd = 1) the hot flow is
    the reachable f_hard L_Edd.
    """
    comps = _ledger(log_lbol)
    assert comps["agn_ledger_closed"] == 1.0
    total = sum(10.0 ** comps[k] for k in _PIECE_KEYS if np.isfinite(comps[k]))
    total += sum(10.0 ** comps[k] for k in _UNEMITTED_KEYS if np.isfinite(comps[k]))
    budget = 10.0 ** (log_lbol + LOG_LSUN_ERG)
    np.testing.assert_allclose(total / budget, 1.0, rtol=_LEDGER_RTOL, atol=0.0)
    assert np.isfinite(comps["log_L_agn_disc_unemitted"])
