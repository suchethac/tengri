# SPDX-License-Identifier: BSD-3-Clause
"""Opt-in IR window for the q-based radio normalization (#2763).

``radio.sf.ir_window`` selects the rest-wavelength band of the dust-emission SED
that is integrated to form the ``L_IR`` the q relations calibrate against:
``"total"`` (default, the dust power as published), ``"tir"`` (8-1000 um, Bell
2003 ApJ 586, 794) or ``"fir"`` (42.5-122.5 um, Helou et al. 1985 ApJ 298, L7).

Reference solution: a modified blackbody whose band integral is evaluated by
adaptive quadrature in frequency (``scipy.integrate.quad``), independent of the
grid trapezoid under test.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from tengri.components.radio.component import (
    IR_WINDOWS_AA,
    RadioSEDComponent,
    RadioSEDComponentConfig,
)
from tengri.config.exceptions import ConfigError
from tengri.utils.physics_constants import C_AA
from tengri.utils.sed_quantities import log10_band_luminosity

pytestmark = pytest.mark.regression_paper

_H_OVER_K = 4.799243e-11  # h/k_B [K s]
_T_DUST = 32.0  # K
_BETA = 1.8
_Q_IR = 2.64
_L_IR_TOTAL = 1.0e45  # erg/s


def _mbb(nu):
    """Modified blackbody L_nu shape (arbitrary normalization)."""
    x = np.minimum(_H_OVER_K * nu / _T_DUST, 700.0)  # exp(700) is finite; the tail is ~1e-300
    return nu ** (3.0 + _BETA) / np.expm1(x)


@pytest.fixture(scope="module")
def grid():
    wave = np.logspace(3.0, 8.5, 8000)  # 100 A - 3.2 cm, ascending
    shape = _mbb(C_AA / wave)
    return wave, shape


def _quad_band(lo_aa, hi_aa):
    """Reference band integral of the unit-normalized shape, by adaptive quadrature."""
    nu_lo, nu_hi = C_AA / hi_aa, C_AA / lo_aa
    val, _ = quad(_mbb, nu_lo, nu_hi, epsabs=0.0, epsrel=1e-12, limit=500)
    return val


@pytest.mark.parametrize("window", ["tir", "fir"])
def test_band_integral_is_exactly_the_requested_window(grid, window):
    """The integrated band is [lo, hi] to 1e-5, edges between grid nodes."""
    wave, shape = grid
    lo, hi = IR_WINDOWS_AA[window]
    # Edges sit strictly between nodes: a node mask would be off by up to a cell.
    assert not np.any(np.isclose(wave, lo, rtol=1e-9)) and not np.any(np.isclose(wave, hi, rtol=1e-9))
    got = 10.0 ** float(log10_band_luminosity(jnp.asarray(shape), jnp.asarray(wave), lo, hi))
    assert got == pytest.approx(_quad_band(lo, hi), rel=1e-5)


def test_arbitrary_window_edges_between_nodes(grid):
    """A 31.3-77.7 um window, both edges between nodes, matches quadrature to 1e-5."""
    wave, shape = grid
    lo, hi = 3.13e5, 7.77e5
    got = 10.0 ** float(log10_band_luminosity(jnp.asarray(shape), jnp.asarray(wave), lo, hi))
    assert got == pytest.approx(_quad_band(lo, hi), rel=1e-5)


def _derived(grid):
    """Derived state of a dust block: sed_dust_ir normalized so the total power is _L_IR_TOTAL."""
    wave, shape = grid
    total = _quad_band(wave[0], wave[-1])
    sed = shape * (_L_IR_TOTAL / total)
    return wave, {
        "L_ir": jnp.asarray(_L_IR_TOTAL),
        "sed_dust_ir": jnp.asarray(sed),
    }, total


def _params():
    return {
        "radio_q_ir": jnp.asarray(_Q_IR),
        "radio_alpha_sf": jnp.asarray(0.8),
        "radio_T_e": jnp.asarray(1.0e4),
        "radio_alpha_ff": jnp.asarray(-0.1),
        "radio_loudness": jnp.asarray(-3.0),
        "radio_alpha_agn": jnp.asarray(0.7),
        "radio_log_nu_cut": jnp.asarray(30.0),
        "redshift": jnp.asarray(0.0),
    }


def _sf_1p4ghz(window, derived, wave):
    comp = RadioSEDComponent(
        config=RadioSEDComponentConfig(
            agn_radio_model="none", include_freefree=False, q_is_total=False, ir_window=window
        )
    )
    inputs = comp.emitter_inputs(derived, wave)
    nu = jnp.asarray([1.4e9])
    terms = comp.emission_terms(_params(), C_AA / nu, **inputs)
    return float(terms["sf"][0])


@pytest.mark.parametrize("window", ["tir", "fir"])
def test_radio_luminosity_scales_with_the_windowed_ir(grid, window):
    """L_nu(1.4 GHz) = L_window / (3.75e12 * 10^q): the q relation on the windowed L_IR."""
    wave, derived, total = _derived(grid)
    lo, hi = IR_WINDOWS_AA[window]
    l_window = _L_IR_TOTAL * _quad_band(lo, hi) / total
    expected = l_window / (3.75e12 * 10.0**_Q_IR)
    assert _sf_1p4ghz(window, derived, jnp.asarray(wave)) == pytest.approx(expected, rel=1e-5)


def test_default_total_uses_the_published_dust_power_unchanged(grid):
    """ir_window='total' reads L_ir as published, with no band integral at all."""
    wave, derived, _ = _derived(grid)
    expected = _L_IR_TOTAL / (3.75e12 * 10.0**_Q_IR)
    assert _sf_1p4ghz("total", derived, jnp.asarray(wave)) == pytest.approx(expected, rel=1e-12)
    # Without a wavelength grid or a dust SED, the default still works.
    comp = RadioSEDComponent()
    inputs = comp.emitter_inputs({"L_ir": jnp.asarray(_L_IR_TOTAL)})
    assert float(inputs["L_ir"]) == _L_IR_TOTAL
    assert "log_L_ir" not in inputs


def test_windowed_ir_without_dust_emission_is_refused():
    """A non-total window with no sed_dust_ir raises, rather than falling back to the total."""
    comp = RadioSEDComponent(config=RadioSEDComponentConfig(ir_window="tir"))
    with pytest.raises(ConfigError, match="sed_dust_ir"):
        comp.emitter_inputs({"L_ir": jnp.asarray(1.0e44)}, jnp.logspace(3.0, 8.0, 50))


def test_unknown_window_is_refused_at_construction():
    with pytest.raises(ValueError, match="ir_window"):
        RadioSEDComponentConfig(ir_window="mir")
