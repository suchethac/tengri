"""Tests for PipelineState.add_intrinsic helper."""

import jax.numpy as jnp
import numpy as np

from tengri.protocols.component import PipelineState


class TestAddIntrinsic:
    """Unit tests for PipelineState.add_intrinsic."""

    def test_add_intrinsic_to_none(self):
        """When sed_intrinsic is None, initializes with the component."""
        wave = jnp.linspace(1000, 10000, 100)
        L_component = jnp.ones(100) * 1e38

        state = PipelineState(wave=wave, sed_intrinsic=None)
        new_state = state.add_intrinsic(L_component)

        assert new_state.sed_intrinsic is not None
        np.testing.assert_array_equal(new_state.sed_intrinsic, L_component)
        # Original state unchanged (immutability)
        assert state.sed_intrinsic is None

    def test_add_intrinsic_to_existing(self):
        """When sed_intrinsic is not None, accumulates the component."""
        wave = jnp.linspace(1000, 10000, 100)
        L_initial = jnp.ones(100) * 1e38
        L_component = jnp.ones(100) * 2e38

        state = PipelineState(wave=wave, sed_intrinsic=L_initial)
        new_state = state.add_intrinsic(L_component)

        expected = L_initial + L_component
        np.testing.assert_array_almost_equal(new_state.sed_intrinsic, expected)
        # Original state unchanged (immutability)
        np.testing.assert_array_equal(state.sed_intrinsic, L_initial)

    def test_add_intrinsic_multiple_times(self):
        """Calling add_intrinsic sequentially accumulates correctly."""
        wave = jnp.linspace(1000, 10000, 100)
        L_comp1 = jnp.ones(100) * 1e38
        L_comp2 = jnp.ones(100) * 2e38
        L_comp3 = jnp.ones(100) * 3e38

        state = PipelineState(wave=wave, sed_intrinsic=None)
        state = state.add_intrinsic(L_comp1)
        state = state.add_intrinsic(L_comp2)
        state = state.add_intrinsic(L_comp3)

        expected = L_comp1 + L_comp2 + L_comp3
        np.testing.assert_array_almost_equal(state.sed_intrinsic, expected)

    def test_add_intrinsic_preserves_other_fields(self):
        """add_intrinsic preserves sed_attenuated, sed_observed, etc."""
        wave = jnp.linspace(1000, 10000, 100)
        L_initial = jnp.ones(100) * 1e38
        L_attenuated = jnp.ones(100) * 0.9e38
        L_observed = jnp.ones(100) * 0.8e38
        L_component = jnp.ones(100) * 2e38

        state = PipelineState(
            wave=wave,
            sed_intrinsic=L_initial,
            sed_attenuated=L_attenuated,
            sed_observed=L_observed,
        )
        new_state = state.add_intrinsic(L_component)

        # New sed_intrinsic is updated
        expected_intrinsic = L_initial + L_component
        np.testing.assert_array_almost_equal(new_state.sed_intrinsic, expected_intrinsic)

        # Other fields are unchanged
        np.testing.assert_array_equal(new_state.sed_attenuated, L_attenuated)
        np.testing.assert_array_equal(new_state.sed_observed, L_observed)
        np.testing.assert_array_equal(new_state.wave, wave)

    def test_add_intrinsic_with_broadcasting(self):
        """add_intrinsic works with array broadcasting."""
        wave = jnp.linspace(1000, 10000, 100)
        L_initial = jnp.ones(100) * 1e38
        # Scalar broadcasted to shape (100,)
        L_component = 2e38

        state = PipelineState(wave=wave, sed_intrinsic=L_initial)
        new_state = state.add_intrinsic(L_component)

        expected = L_initial + L_component
        np.testing.assert_array_almost_equal(new_state.sed_intrinsic, expected)

    def test_add_intrinsic_immutability_with_derived(self):
        """add_intrinsic returns a new state; original.derived is unchanged."""
        from tengri.protocols.derived_bundle import DerivedBundle

        wave = jnp.linspace(1000, 10000, 100)
        L_initial = jnp.ones(100) * 1e38
        L_component = jnp.ones(100) * 2e38

        # Create a bundle with one field set
        bundle = DerivedBundle(log_mstar=jnp.array(10.0))
        state = PipelineState(
            wave=wave,
            sed_intrinsic=L_initial,
            derived=bundle,
        )
        new_state = state.add_intrinsic(L_component)

        # New state has accumulated sed_intrinsic
        expected_intrinsic = L_initial + L_component
        np.testing.assert_array_almost_equal(new_state.sed_intrinsic, expected_intrinsic)

        # Original state is unchanged
        np.testing.assert_array_equal(state.sed_intrinsic, L_initial)
        # Derived bundle is shared (DerivedBundle is also immutable)
        assert state.derived is new_state.derived
