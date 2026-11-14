# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the Drude UV bump on the ``power_law`` attenuation law."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import power_law
from tengri.components.dust.laws._registry import law_kwarg_names, reject_unread_law_kwargs
from tengri.utils.physics_constants import V_BAND_ANGSTROM

pytestmark = pytest.mark.contract

WAVE = np.geomspace(912.0, 25000.0, 80)


def test_default_path_is_bare_power_law_bit_identical():
    """No bump requested: the curve is the pre-bump expression, bit for bit."""
    # The pre-bump expression, evaluated by JAX as the law always was.
    old = np.asarray((jnp.asarray(WAVE) / V_BAND_ANGSTROM) ** -0.7)
    np.testing.assert_array_equal(np.asarray(power_law(jnp.asarray(WAVE))), old)
    np.testing.assert_array_equal(
        np.asarray(power_law(jnp.asarray(WAVE), dust_slope=-0.7, dust_bump_strength=0.0)),
        old,
    )


@pytest.mark.parametrize("amplitude", [0.5, 1.0, 3.0])
def test_bump_curve_is_normalized_at_v_band(amplitude):
    """k(5500 A) = 1 with the bump on, so tau_V keeps its meaning."""
    k_v = power_law(jnp.asarray([V_BAND_ANGSTROM]), dust_bump_strength=amplitude)
    np.testing.assert_allclose(np.asarray(k_v), 1.0, rtol=1e-12)


def test_bump_strength_is_read_by_power_law():
    """``bump_strength`` is no longer refused for the ``power_law`` law."""
    assert {"dust_slope", "dust_bump_strength", "dust_bump_x0", "dust_bump_gamma"} <= set(
        law_kwarg_names("power_law")
    )
    reject_unread_law_kwargs({"dust_bump_strength": 1.0}, ("power_law",), "test")


def test_bump_gradient_is_finite():
    """The amplitude is a traced prior value under inference, so its gradient must exist."""

    def total(amplitude):
        return jnp.sum(power_law(jnp.asarray(WAVE), dust_bump_strength=amplitude))

    grad = jax.grad(total)(jnp.asarray(1.0))
    assert np.isfinite(float(grad))
    assert float(grad) != 0.0
