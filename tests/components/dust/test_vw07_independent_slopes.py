# SPDX-License-Identifier: BSD-3-Clause
"""VW07 / Wild+2007 two-component attenuation — independent per-leaf slopes (#500)."""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import two_component_dust

pytestmark = pytest.mark.unit


class TestVW07PerLeafSlopes:
    """Birth-cloud and diffuse-ISM power-law slopes can be set independently."""

    def test_vw07_steepens_birth_cloud_uv(self):
        """With slope_bc=-1.3 (Wild+07 default), young populations see more UV attenuation."""
        wave = jnp.array([1500.0, 5500.0])
        ages_young = jnp.array([1e6])  # birth cloud dominant
        shared = two_component_dust(wave, ages_young, tau_v1=1.0, tau_v2=0.3, n_slope=-0.7)
        vw07 = two_component_dust(
            wave, ages_young, tau_v1=1.0, tau_v2=0.3, n_slope_bc=-1.3, n_slope_diff=-0.7
        )
        # Steeper BC slope ⇒ greater UV attenuation ⇒ smaller T at 1500 Å.
        assert float(vw07[0, 0]) < float(shared[0, 0])

    def test_diffuse_only_unchanged_when_slope_diff_matches(self):
        """Old population (BC sigmoid off) is essentially unaffected if slope_diff
        matches the shared slope."""
        wave = jnp.array([2000.0, 5500.0, 9000.0])
        ages_old = jnp.array([1e10])
        shared = two_component_dust(wave, ages_old, tau_v1=1.0, tau_v2=0.3, n_slope=-0.7)
        vw07 = two_component_dust(
            wave, ages_old, tau_v1=1.0, tau_v2=0.3, n_slope_bc=-1.3, n_slope_diff=-0.7
        )
        np.testing.assert_allclose(np.asarray(shared), np.asarray(vw07), atol=1e-3)

    def test_backwards_compatible_when_no_per_leaf_slopes(self):
        """Omitting n_slope_bc / n_slope_diff reproduces the shared-slope path bit-exactly."""
        wave = jnp.array([1500.0, 5500.0, 9000.0])
        ages = jnp.array([1e7, 1e9])
        baseline = two_component_dust(wave, ages, tau_v1=1.0, tau_v2=0.3, n_slope=-0.7)
        explicit_none = two_component_dust(
            wave,
            ages,
            tau_v1=1.0,
            tau_v2=0.3,
            n_slope=-0.7,
            n_slope_bc=None,
            n_slope_diff=None,
        )
        np.testing.assert_array_equal(np.asarray(baseline), np.asarray(explicit_none))


class TestVW07Recipe:
    """The recipes.vw07_attenuation preset wires slope_bc / slope_diff as free params."""

    def test_recipe_makes_per_leaf_slopes_free(self):
        from tengri import recipes
        from tengri.parameters.groups import parse_groups

        spec = parse_groups(**recipes.vw07_attenuation())
        assert "dust_slope_bc" in spec.free_params
        assert "dust_slope_diff" in spec.free_params
