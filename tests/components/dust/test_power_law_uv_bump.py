# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the Drude UV bump on the ``power_law`` attenuation law."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import power_law
from tengri.components.dust.laws._registry import law_kwarg_names, reject_unread_law_kwargs
from tengri.parameters import FREE, Fixed, Uniform, parse_groups
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
    """``bump_strength`` is read by the ``power_law`` law."""
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


def _dust_free_params(**dust_attenuation):
    """Dust parameters left free by a two-component model built from ``dust_attenuation``."""
    params = parse_groups(
        dust_attenuation={"type": "two_component", **dust_attenuation},
        ssp_data=None,
        redshift=Fixed(0.1),
    )
    return {name for name in params.free_params if name.startswith("dust_")}


def test_power_law_wildcard_does_not_free_the_opt_in_bump():
    """``all_params: FREE`` on a ``power_law`` screen frees the bare curve only.

    The bump is opt-in, so the wildcard keeps ``dust_bump_*`` Fixed(DEFAULT). The
    expected set is the one main frees for this model.
    """
    free = _dust_free_params(law="power_law", all_params=FREE)
    assert free == {"dust_slope", "dust_tau_bc", "dust_tau_diff"}


def test_power_law_bump_strength_named_frees_only_that_bump_parameter():
    """Naming ``bump_strength`` frees it; ``x0`` and ``gamma`` stay Fixed."""
    free = _dust_free_params(law="power_law", all_params=FREE, bump_strength=Uniform(0.0, 5.0))
    assert free == {"dust_slope", "dust_tau_bc", "dust_tau_diff", "dust_bump_strength"}


def test_noll09_wildcard_still_frees_the_bump_parameters():
    """A law that reads the bump keeps the wildcard freeing it (unchanged from main)."""
    free = _dust_free_params(law="noll09", all_params=FREE)
    assert {"dust_bump_strength", "dust_bump_x0", "dust_bump_gamma"} <= free
