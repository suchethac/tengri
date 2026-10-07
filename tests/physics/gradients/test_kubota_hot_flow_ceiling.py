# SPDX-License-Identifier: BSD-3-Clause
"""The hot-flow luminosity the corona radiates is what its annuli dissipate (K&D 2018 Eq. 2).

``R_hot`` is solved from ``L_diss,hot(R_hot) = L_hot`` (Page-Thorne dissipation
``L0 h(x)``, ``x = R_hot / R_isco``) and then clipped to ``0.5 R_out`` by the zone
clip. Where the disc cannot supply ``f_hard L_Edd`` inside that clip (``lambda_Edd``
below ``f_hard``, which the ``agn_f_hard`` prior ``Uniform(0, 0.1)`` reaches up to
``lambda_Edd ~ 0.1``), ``L_hot`` must be limited to ``L0 h(x_clip)``: the corona then
radiates the power its own annuli dissipate, and ``L_diss(R_hot) = L_hot`` holds by
construction. A saturation ceiling at ``h(1e4)`` instead lets the corona radiate up to
19 % more than ``L_diss(R_hot)`` (``L_diss/L_hot = 0.812`` at ``log M = 9``,
``lambda_Edd = 0.015``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import (
    _nt_emissivity as N,
    disc as D,
)

pytestmark = pytest.mark.conservation

_X_LO_CLIP = 1.01  # the inner zone clip of ``_compute_zone_radii`` (R_hot >= 1.01 R_isco)


def _setup(log_mbh, lambda_edd, a_spin):
    """Scalars as ``kubota_done_disc`` forms them, at the ``log L_bol`` giving ``lambda_Edd``."""
    _, _, _, _, log10_l_edd, _ = D._compute_bh_params(log_mbh, 11.0, a_spin, float32=False)
    log_lbol = float(log10_l_edd + np.log10(lambda_edd) - D._LOG10_LSUN_ERG)
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
    return r_g, r_isco_rg, r_isco_cm, t_in, log10_l_edd, log_lbol


def _corona_luminosity(monkeypatch, log_mbh, log_lbol, f_hard, a_spin):
    """The ``l_hot_erg`` that ``kubota_done_disc`` hands to its corona, read where it is used."""
    seen = []
    original = D._hot_corona_lnu

    def spy(nu, l_hot_erg, *args, **kwargs):
        seen.append(float(l_hot_erg))
        return original(nu, l_hot_erg, *args, **kwargs)

    monkeypatch.setattr(D, "_hot_corona_lnu", spy)
    with jax.disable_jit():
        D.kubota_done_disc(
            jnp.geomspace(1.0, 1.0e5, 8),
            agn_log_lbol=log_lbol,
            agn_log_mbh=log_mbh,
            agn_a_spin=a_spin,
            agn_f_hard=f_hard,
        )
    assert len(seen) == 1, "the corona must be formed exactly once"
    return seen[0]


@pytest.mark.parametrize("a_spin", [0.0, 0.7])
@pytest.mark.parametrize("f_hard", [0.02, 0.05, 0.1])
@pytest.mark.parametrize(
    ("log_mbh", "lambda_edd"),
    [(7.0, 0.015), (8.0, 0.02), (9.0, 0.015), (9.0, 0.03), (9.0, 0.05), (8.5, 0.3), (9.0, 1.0)],
)
def test_corona_radiates_what_the_hot_flow_dissipates(
    monkeypatch, log_mbh, lambda_edd, f_hard, a_spin
):
    """``L_diss(R_hot) / L_hot = 1`` to 1e-6 wherever ``R_hot`` is not at its inner clip.

    ``L_hot`` is the luminosity the disc passes to its corona. Saturated (``R_hot`` at
    ``0.5 R_out``) or interior alike; at the inner clip (``1.01 R_isco``) the annuli inside
    ``R_hot`` dissipate at least ``L_hot``.
    """
    r_g, r_isco_rg, r_isco_cm, t_in, log10_l_edd, log_lbol = _setup(log_mbh, lambda_edd, a_spin)
    r_hot, _r_warm, _r_out = D._compute_zone_radii(
        r_g, r_isco_rg, r_isco_cm, t_in, log_mbh, log_lbol, f_hard, 2.0, log10_l_edd,
        False, a_spin,
    )  # fmt: skip
    l_hot = _corona_luminosity(monkeypatch, log_mbh, log_lbol, f_hard, a_spin)
    l0 = float(D._nt_l0(r_isco_cm, t_in, False))
    x_hot = float(r_hot / r_isco_cm)
    ratio = l0 * float(N.nt_h(jnp.log(x_hot), a_spin)) / l_hot
    if x_hot > _X_LO_CLIP * (1.0 + 1e-9):
        assert abs(ratio - 1.0) < 1e-6, (
            f"log M={log_mbh}, lambda_Edd={lambda_edd}, f_hard={f_hard}, a={a_spin}: "
            f"x_hot={x_hot:.4f}, L_diss(R_hot)/L_hot = {ratio:.6f}"
        )
    else:
        assert ratio >= 1.0 - 1e-6, f"inner clip: L_diss(R_hot)/L_hot = {ratio:.6f}"


def test_saturated_corona_is_pinned_at_the_zone_clip():
    """At log M = 9, lambda_Edd = 0.015, f_hard = 0.1 the zone clip binds (the bug's regime)."""
    r_g, r_isco_rg, r_isco_cm, t_in, log10_l_edd, log_lbol = _setup(9.0, 0.015, 0.0)
    r_hot, _r_warm, r_out = D._compute_zone_radii(
        r_g, r_isco_rg, r_isco_cm, t_in, 9.0, log_lbol, 0.1, 2.0, log10_l_edd, False, 0.0
    )
    assert float(r_hot / (0.5 * r_out)) == pytest.approx(1.0, rel=1e-9)
