# SPDX-License-Identifier: BSD-3-Clause
"""The X-ray corona anchors on ``log_L_agn_bol``, not the erg/s ``L_agn_bol`` (#2842).

The last-resort anchor of the corona is ``L_2500 = L_bol / (BC_2500 nu_2500)`` with the
Hopkins et al. (2007) bolometric correction ``BC_2500 = 5.15`` and ``nu_2500 = 1.199e15`` Hz.
Reading the erg/s ``L_agn_bol`` (1e44 erg/s for ``log_l5100 = 44``) gives ``inf`` in pure
float32 (max 3.4e38) before that division, so the whole X-ray SED was ``inf``/NaN beside an
AGN that publishes a finite ``log_L_agn_bol``. The anchor is now
``pow10(log_L_agn_bol - log10(BC_2500 nu_2500))``, which is representable (~1e28 erg/s/Hz).

Float32 is selected with ``jax.enable_x64(False)`` inside the test; ``JAX_ENABLE_X64=0`` is
never set for pytest.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.grahsp import GRAHSPSEDComponent
from tengri.components.xray.component import XRaySEDComponent
from tengri.components.xray.xray import l_2500_from_bolometric
from tengri.protocols.component import ForwardState
from tengri.utils.physics_constants import L_SUN

pytestmark = pytest.mark.regression_bug

_RTOL = 1.0e-4
_GRAHSP = {
    "agn_grahsp_log_l5100": 44.0,
    "agn_grahsp_uvslope": 0.0,
    "agn_grahsp_plslope": -1.7,
    "agn_grahsp_plbendloc_nm": 100.0,
    "agn_grahsp_plbendwidth": 1.0,
    "agn_grahsp_cutoff_nm": 10000.0,
    "agn_grahsp_a_lines": 1.0,
    "agn_grahsp_a_feii": 5.0,
    "agn_grahsp_linewidth_kms": 5000.0,
    "agn_grahsp_fcov": 0.4,
    "agn_grahsp_si": 0.0,
    "agn_grahsp_cool_lam_um": 17.0,
    "agn_grahsp_cool_width": 0.45,
    "agn_grahsp_hot_lam_um": 2.0,
    "agn_grahsp_hot_width": 0.5,
    "agn_grahsp_hot_fcov": 1.0,
    "agn_grahsp_ebv": 0.05,
    "agn_grahsp_ebv_agn": 0.05,
}
_XRAY = {
    "xray_gamma_agn": 1.8,
    "xray_gamma_hmxb": 2.0,
    "xray_gamma_lmxb": 1.56,
    "xray_E_cut": 300.0,
    "xray_delta_alpha_ox": 0.0,
    "xray_log_nh": 20.0,
    "xray_alpha_irx": 0.3,
    "xray_det_hmxb": 0.0,
    "xray_det_lmxb": 0.0,
}
#: Rest wavelengths from the UV to the hard X-ray [Angstrom]; the corona sits below ~100 A.
_WAVE = np.geomspace(0.5, 1.0e4, 120)


def _grahsp_then_xray(float32):
    """GRAHSP (``log_l5100 = 44``) feeding the X-ray component: ``(log_L_agn_bol, SED)``."""
    dtype = jnp.float32 if float32 else jnp.float64
    with jax.enable_x64(not float32):
        wave = jnp.asarray(_WAVE, dtype=dtype)
        agn = GRAHSPSEDComponent().apply(
            ForwardState(wave=wave), {k: jnp.asarray(v, dtype=dtype) for k, v in _GRAHSP.items()}
        )
        out = XRaySEDComponent().apply(
            agn.with_(sed_intrinsic=jnp.zeros_like(wave)),
            {k: jnp.asarray(v, dtype=dtype) for k, v in {**_XRAY, "redshift": 0.5}.items()},
        )
        return float(agn.derived["log_L_agn_bol"]), np.asarray(out.derived["sed_xray"], np.float64)


def test_the_anchor_is_representable_where_the_erg_per_s_luminosity_is_not():
    """``l_2500_from_bolometric`` is 1e28 erg/s/Hz in float32 for a 1e44 erg/s AGN."""
    with jax.enable_x64(False):
        log_bol = jnp.asarray(44.0, dtype=jnp.float32)
        assert not np.isfinite(float(jnp.asarray(10.0, dtype=jnp.float32) ** 44.0))
        got = float(l_2500_from_bolometric(log_bol))
    assert np.isfinite(got)
    assert got == pytest.approx(1.0e44 / (5.15 * 1.199e15), rel=1e-5)
    assert float(l_2500_from_bolometric(jnp.asarray(-jnp.inf))) == 0.0


def test_xray_beside_grahsp_is_finite_in_float32_and_matches_float64():
    """GRAHSP (``log_l5100 = 44``) + X-ray: finite float32 SED within 1e-4 of float64."""
    log_bol64, sed64 = _grahsp_then_xray(float32=False)
    log_bol32, sed32 = _grahsp_then_xray(float32=True)
    assert log_bol64 > np.log10(L_SUN) + 10.0
    assert np.isfinite(log_bol32)
    assert np.all(np.isfinite(sed32)), "X-ray SED is non-finite in float32"
    significant = sed64 > 1.0e-3 * sed64.max()
    assert significant.any() and sed64.max() > 0.0, "the corona is silent: the check is vacuous"
    err = np.max(np.abs(sed32[significant] / sed64[significant] - 1.0))
    assert err < _RTOL, f"float32 X-ray SED rel {err:.2e}"
