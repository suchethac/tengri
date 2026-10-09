# SPDX-License-Identifier: BSD-3-Clause
"""The composable AGN's 6 and 12 um diagnostics are point values of the emitted SED (#2745).

``log_L_6um`` and ``log_L_12um`` are ``log10(nu L_nu)`` of the disc + torus + polar SED
evaluated at exactly 6e4 A and 1.2e5 A. They are not integrals, and they are not an
interpolation of the caller's wavelength grid: the chain is evaluated at those two
wavelengths. The test uses an analytic stand-in for the chain, so the expected values are
known in closed form.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.component import _MIR_DIAGNOSTIC_AA, _mir_point_logs

pytestmark = pytest.mark.contract

C_AA = 2.99792458e18


def _power_law_chain(exponents):
    """A composable-style chain returning per-block components as power laws in wavelength."""

    def chain(wave, agn_log_lbol=10.0, return_l2500=False, return_components=False, **_):
        wave = jnp.asarray(wave)
        comps = {
            name: 10.0 ** (agn_log_lbol - 10.0) * wave**exponent
            for name, exponent in exponents.items()
        }
        if return_components:
            return None, None, None, comps
        return sum(comps.values())

    return chain


def _expected_log_nu_lnu(exponents, wave_aa, agn_log_lbol):
    lnu = sum(10.0 ** (agn_log_lbol - 10.0) * wave_aa**e for e in exponents.values())
    return float(np.log10(C_AA / wave_aa * lnu))


@pytest.mark.parametrize("agn_log_lbol", [10.0, 11.5])
def test_mir_points_are_the_emitted_sed_at_6_and_12_um(agn_log_lbol):
    """Values equal log10(nu L_nu) of the chain evaluated at exactly 6 um and 12 um."""
    exponents = {"disc": -1.2, "torus": -0.8, "polar": 0.3}
    chain = _power_law_chain(exponents)
    log_6, log_12 = _mir_point_logs("composable", chain, {}, agn_log_lbol, jnp.float64)
    assert float(log_6) == pytest.approx(
        _expected_log_nu_lnu(exponents, _MIR_DIAGNOSTIC_AA[0], agn_log_lbol), rel=1e-12
    )
    assert float(log_12) == pytest.approx(
        _expected_log_nu_lnu(exponents, _MIR_DIAGNOSTIC_AA[1], agn_log_lbol), rel=1e-12
    )


def test_mir_points_do_not_depend_on_the_caller_grid():
    """The helper takes no caller grid: the same chain gives the same values every call."""
    exponents = {"disc": -1.0, "torus": -0.5, "polar": 0.3}
    chain = _power_law_chain(exponents)
    first = _mir_point_logs("composable", chain, {}, 11.0, jnp.float64)
    second = _mir_point_logs("composable", chain, {}, 11.0, jnp.float64)
    assert np.allclose(np.asarray(first), np.asarray(second), rtol=0.0, atol=0.0)


def test_mir_points_are_differentiable_in_lbol():
    """The point values stay differentiable in agn_log_lbol (gradient-safe)."""
    exponents = {"disc": -1.2, "torus": -0.8, "polar": 0.3}
    chain = _power_law_chain(exponents)

    def loss(x):
        log_6, log_12 = _mir_point_logs("composable", chain, {}, x, jnp.float64)
        return log_6 + log_12

    grad = jax.grad(loss)(jnp.asarray(11.0))
    assert np.isfinite(float(grad))
    assert float(grad) == pytest.approx(2.0, rel=1e-9)
