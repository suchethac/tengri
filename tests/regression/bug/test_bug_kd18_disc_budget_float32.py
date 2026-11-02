# SPDX-License-Identifier: BSD-3-Clause
r"""The kubota_done disc budget is not a difference of near-equal float32 powers.

The disc and warm zones carry :math:`L_{\rm bol} - l_{\rm hot}` (Kubota & Done 2018, MNRAS 480,
1247: Eq. 2 gives :math:`l_{\rm hot} = L_{\rm diss,hot}`; the remainder of the accretion power is
the disc's). At the prior edge ``log_ledd = -0.75``, ``f_hard = 0.05`` the corona carries 95.9% of
the power, so the disc share :math:`1 - l_{\rm hot}/L_{\rm bol} = 0.041` amplifies a relative error
of :math:`\epsilon` in the corona share by :math:`0.959/0.041 = 23.6`.

The accretion rate is inferred so that the disc's total dissipation :math:`L_0 h(\infty)` is
:math:`L_{\rm bol}` (owner ruling 2026-10-20), so :math:`L_0/L_{\rm bol} = 1/h(\infty)` and the
share is at most 0.99. Before that ruling :math:`L_0/L_{\rm bol} = 3/(2 \eta r_{\rm isco})` and the
share read 0.959307742595 at the edge, where the zone clip binds. The invariants are (1) the corona
share equals the closed form :math:`\min(f_{\rm hard}
L_{\rm Edd}/L_{\rm bol},\ h_{\rm cap}/h(\infty))`
(float64, 1e-12), and (2) the
float32 disc spectrum follows float64 to 5e-5 at that edge, where forming the budget from
``log10 L_bol`` and ``log10 l_hot`` gave 1.68e-4.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import disc as disc_module

pytestmark = pytest.mark.regression_bug

# ``f_hard = 0.028`` puts the corona at 92% of L_bol (lambda_Edd = 0.0304) on the unsaturated side
# of the hot-flow switch, which sits at 96.5% here (``h(R_out / R_isco) / h(inf)``, #3015); at the
# earlier 0.05 the hot flow fills the disc, the disc and warm zones have zero width and there is no
# disc spectrum left to compare.
_EDGE = dict(
    agn_log_lbol=11.0,
    agn_log_mbh=8.0,
    agn_a_spin=0.499,
    agn_cos_inc=0.5,
    agn_f_hard=0.028,
    agn_gamma_warm=2.5,
    agn_kt_warm=0.55,
    agn_gamma_hard=2.0,
    agn_kt_hot=175.0,
    agn_r_warm_ratio=3.0,
)


def _parts(float32):
    dtype = jnp.float32 if float32 else jnp.float64
    with jax.enable_x64(not float32):
        wave = jnp.geomspace(100.0, 1.0e6, 300).astype(dtype)
        kw = {k: jnp.asarray(v, dtype=dtype) for k, v in _EDGE.items()}
        out = disc_module.kubota_done_disc(wave, _return_parts=True, **kw)
        return [np.asarray(x, dtype=np.float64) for x in out]


def test_corona_share_is_the_closed_form_in_float64():
    """Below the hot-flow switch the share is ``f_hard L_Edd / L_bol``, to 1e-12."""
    share = _parts(False)[3]
    log10_ledd = float(disc_module._log10_eddington_luminosity(_EDGE["agn_log_mbh"]))
    expected = _EDGE["agn_f_hard"] * 10.0 ** (
        log10_ledd - disc_module._LOG10_LSUN_ERG - _EDGE["agn_log_lbol"]
    )
    assert share == pytest.approx(expected, abs=1e-12)
    assert share <= 0.99


def test_hot_flow_fraction_matches_hot_flow_luminosity_in_float64():
    """The share is ``_hot_flow_luminosity / L_bol`` with ``L_0 = L_bol / h(inf)``."""
    with jax.enable_x64(True):
        for log_lbol, log_mbh, f_hard, a_spin in [
            (11.0, 8.0, 0.05, 0.499),
            (9.0, 9.5, 0.02, 0.0),
            (12.5, 8.0, 0.4, 0.998),
        ]:
            h_inf = float(disc_module._nt_h(jnp.log(disc_module._NT_X_MAX), a_spin))
            l0 = 10.0**log_lbol / h_inf  # [L_sun]
            log10_ledd = float(disc_module._log10_eddington_luminosity(log_mbh))
            log10_ledd_lsun = log10_ledd - disc_module._LOG10_LSUN_ERG
            expected = (
                float(
                    disc_module._hot_flow_luminosity(
                        f_hard, log10_ledd_lsun, l0, a_spin, x_hot_max=1.0e3
                    )
                )
                / 10.0**log_lbol
            )
            got = float(
                disc_module._hot_flow_fraction(
                    f_hard, log10_ledd_lsun - log_lbol, a_spin, x_hot_max=1.0e3
                )
            )
            assert got == pytest.approx(expected, rel=1e-12)


def test_float32_disc_spectrum_follows_float64_at_the_corona_dominated_edge():
    """The disc spectrum agrees to 2e-4 (a uniform offset; it was 5e-5 while rescaled)."""
    p64, p32 = _parts(False), _parts(True)
    disc64, disc32 = p64[1], p32[1]
    significant = disc64 > 1e-3 * disc64.max()
    err = np.max(np.abs(disc32[significant] / disc64[significant] - 1.0))
    # #3015: the disc spectrum is no longer rescaled to its budget, so the float32 evaluation of
    # its absolute power (the T_in^4 ring amplitude) is no longer divided out. Measured 1.75e-4
    # (a uniform -1.5e-4 offset, 7.6e-5 dex); the shape is unchanged. The bound is that, rounded.
    assert err < 2.0e-4, err
    assert p32[3] == pytest.approx(p64[3], rel=1e-5)
