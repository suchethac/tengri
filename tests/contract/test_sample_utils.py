# SPDX-License-Identifier: BSD-3-Clause
"""Tests for shared sampling utilities (_sample_utils.py).

This string was not the module docstring: an assignment preceded it, so it was
a bare expression statement and ``test_sample_utils.__doc__`` was None.
"""

import jax.numpy as jnp
import pytest

from tengri.inference._sample_utils import _mean_params, _vmap_samples_to_physical

pytestmark = pytest.mark.contract


def _dummy_unravel(flat_pos):
    return {"param_0": flat_pos[0], "param_1": flat_pos[1]}


def _dummy_to_physical(unbounded_dict):
    return {k: v for k, v in unbounded_dict.items()}


class TestVmapSamplesToPhysical:
    def test_converts_flat_samples_to_physical_dict(self):
        batch_size = 10
        n_dim = 2
        samples_flat = jnp.arange(batch_size * n_dim).reshape(batch_size, n_dim)

        result = _vmap_samples_to_physical(samples_flat, _dummy_unravel, _dummy_to_physical)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "param_0" in result
        assert "param_1" in result
        assert result["param_0"].shape == (batch_size,)
        assert result["param_1"].shape == (batch_size,)

    def test_preserves_values(self):
        batch_size = 5
        n_dim = 2
        samples_flat = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])

        result = _vmap_samples_to_physical(samples_flat, _dummy_unravel, _dummy_to_physical)

        assert jnp.allclose(result["param_0"], jnp.array([1.0, 3.0, 5.0, 7.0, 9.0]))
        assert jnp.allclose(result["param_1"], jnp.array([2.0, 4.0, 6.0, 8.0, 10.0]))


class TestMeanParams:
    def test_computes_per_param_means(self):
        samples = {
            "a": jnp.array([1.0, 2.0, 3.0]),
            "b": jnp.array([4.0, 5.0, 6.0]),
        }

        result = _mean_params(samples)

        assert result["a"] == 2.0
        assert result["b"] == 5.0

    def test_handles_multidimensional_params(self):
        samples = {
            "x": jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "y": jnp.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]]),
        }

        result = _mean_params(samples)

        assert jnp.allclose(result["x"], jnp.array([3.0, 4.0]))
        assert jnp.allclose(result["y"], jnp.array([9.0, 10.0]))

    def test_empty_dict_returns_empty_dict(self):
        samples = {}
        result = _mean_params(samples)
        assert result == {}
