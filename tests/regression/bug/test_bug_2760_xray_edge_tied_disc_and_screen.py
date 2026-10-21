# SPDX-License-Identifier: BSD-3-Clause
r"""X-rays below the torus library's 10 A edge take the Morrison & McCammon (1983) column.

Owner ruling 10: the X-ray absorption behind a torus is photoelectric, :math:`\exp(-\sigma N_H)`
with :math:`N_H = 2.21\times10^{21}\,A_V` cm^-2 (Guver & Ozel 2009) and :math:`A_V` the
sightline's own V-band extinction. Two places still broke it at the library edge (#2760):

1. The tied disc took the library ratio :math:`{\rm disk}_i/{\rm disk}_0` on the caller's grid,
   which the library holds at zero below 10 A, so the tied disc emitted nothing there. Below the
   edge it must carry the same absorbed line-of-sight factor the untied path uses,
   ``disc_factor = 2 cos i T_X``.
2. The analytic fallback screen (SKIRTOR v2, no disc column) attenuated the X-rays with the SMC
   dust curve extrapolated to keV. Below the edge it must be the MM83 column of its own A_V.

Checked here: the tied factor below the edge equals the untied path's factor at the same
inclination and column; the fallback screen at 2 keV equals :math:`\exp(-\sigma N_H)`; and above
the edge the screen keeps its SMC curve.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import torus_library_transmission as tlt
from tengri.components.agn.blocks.torus_library_transmission import (
    library_edge_wavelength as _library_edge_wavelength,
)
from tengri.components.agn.blocks.torus_screen import (
    XRAY_LIBRARY_EDGE_AA,
    torus_screen_transmission,
)
from tengri.components.agn.polar_dust import smc_extinction_curve
from tests._torus_sightline_reference import HC_KEV_AA, NH_PER_AV, sigma_mm83

pytestmark = pytest.mark.regression_bug

#: The library's first node [A]; the X-ray band below it is the one ruling 10 governs.
_EDGE_AA = XRAY_LIBRARY_EDGE_AA
#: Two X-ray samples below the edge: 2 keV and 10 keV.
_XRAY_WAVE = np.array([HC_KEV_AA / 2.0, HC_KEV_AA / 10.0])
_V_WAVE = 5500.0

_SKIRTOR_PARAMS = {
    "agn_tau_skirtor": 3.0,
    "agn_p_skirtor": 1.0,
    "agn_q_skirtor": 1.0,
    "agn_oa_skirtor": 40.0,
    "agn_radius_ratio": 20.0,
}
_FRITZ_PARAMS = {"agn_fritz_oa": 20.0}


def _params(torus: str) -> dict:
    return _SKIRTOR_PARAMS if torus == "skirtor" else _FRITZ_PARAMS


@pytest.mark.parametrize(("torus", "i_deg"), [("skirtor", 70.0), ("fritz", 80.0)])
def test_tied_disc_below_the_library_edge_is_the_untied_absorbed_factor(torus, i_deg):
    """Tied factor below 10 A is nonzero and equals the untied path's ``2 cos i T_X``."""
    cos_inc = float(np.cos(np.radians(i_deg)))
    wave = jnp.asarray(np.concatenate([_XRAY_WAVE, [_V_WAVE, 1.0e3]]))
    sight = tlt.library_torus_sightline(torus, wave, cos_inc=cos_inc, params=_params(torus))
    assert sight is not None
    # The tied disc's library ratio is zero on the caller's grid below the library's edge.
    tied_ratio = jnp.zeros_like(wave)
    tied = np.asarray(tlt.tied_incl_with_library_edge(tied_ratio, wave, sight, torus))
    untied = np.asarray(sight.disc_factor)
    below = np.asarray(wave) < _EDGE_AA
    assert np.all(tied[below] > 0.0)
    np.testing.assert_allclose(tied[below], untied[below], rtol=1e-12)
    # Above the edge the tied ratio is what it was: the library ratio, here zero, untouched.
    np.testing.assert_array_equal(tied[~below], np.asarray(tied_ratio)[~below])


@pytest.mark.parametrize("torus", ["skirtor", "fritz"])
def test_tied_factor_below_the_edge_is_mm83_of_the_library_column(torus):
    """The absorbed factor is 2 cos i exp(-sigma N_H), N_H = 2.21e21 A_V, to 1e-6."""
    i_deg = 70.0 if torus == "skirtor" else 80.0
    cos_inc = float(np.cos(np.radians(i_deg)))
    wave = jnp.asarray(np.concatenate([_XRAY_WAVE, [_V_WAVE]]))
    sight = tlt.library_torus_sightline(torus, wave, cos_inc=cos_inc, params=_params(torus))
    tied = np.asarray(tlt.tied_incl_with_library_edge(jnp.zeros_like(wave), wave, sight, torus))
    t_v = float(np.asarray(sight.transmission)[-1])
    a_v = -2.5 * np.log10(t_v)
    expected = 2.0 * cos_inc * np.exp(-sigma_mm83(HC_KEV_AA / _XRAY_WAVE) * NH_PER_AV * a_v)
    np.testing.assert_allclose(tied[:2], expected, rtol=1e-6)


def test_fallback_screen_at_2_kev_is_mm83_of_its_own_column():
    """SKIRTOR v2 screen: exp(-sigma N_H) at 2 keV with A_V of the screen's own V transmission."""
    cos_inc = float(np.cos(np.radians(70.0)))
    kwargs = {"cos_inc": cos_inc, "oa_deg": 40.0, "tau_v": 3.0}
    t_x = float(np.asarray(torus_screen_transmission(jnp.asarray([HC_KEV_AA / 2.0]), **kwargs))[0])
    t_v = float(np.asarray(torus_screen_transmission(jnp.asarray([_V_WAVE]), **kwargs))[0])
    a_v = -2.5 * np.log10(t_v)
    expected = np.exp(-sigma_mm83(2.0) * NH_PER_AV * a_v)
    np.testing.assert_allclose(t_x, expected, rtol=1e-6)
    assert t_x < 1.0


def test_fallback_screen_above_the_edge_keeps_its_smc_curve():
    """Above 10 A the screen is the SMC reddening curve, as before the X-ray fix."""
    cos_inc = float(np.cos(np.radians(70.0)))
    wave = np.array([1.0e3, 1.0e2, 12.0])
    got = np.asarray(
        torus_screen_transmission(jnp.asarray(wave), cos_inc=cos_inc, oa_deg=40.0, tau_v=3.0)
    )
    k = np.asarray(smc_extinction_curve(jnp.asarray(wave)))
    k_v = float(np.asarray(smc_extinction_curve(jnp.asarray([_V_WAVE])))[0])
    t_v = float(
        np.asarray(torus_screen_transmission(jnp.asarray([_V_WAVE]), cos_inc, 40.0, 3.0))[0]
    )
    type2 = np.log(t_v) / (-3.0)
    np.testing.assert_allclose(got[:2], np.exp(-3.0 * (k[:2] / k_v) * type2), rtol=1e-9)


@pytest.mark.parametrize("torus", ["skirtor", "fritz"])
def test_screen_edge_is_the_packaged_library_first_node(torus):
    """The analytic screen's X-ray edge is the library's own first node (10 A for both)."""
    assert _library_edge_wavelength(torus) == pytest.approx(XRAY_LIBRARY_EDGE_AA, rel=1e-12)
