# SPDX-License-Identifier: BSD-3-Clause
"""Tests for [alpha/Fe] enhancement support.

Verifies:
1. The declared default of met_alpha_fe is Fixed(0.0) and it is registered
2. Free and sampled met_alpha_fe declarations are carried by Parameters

The forward-model behavior on a 4-D [alpha/Fe] grid is in
tests/contract/test_alpha_fe_free_parameter.py; a 3-D grid refuses a non-zero
met_alpha_fe at build (#3411).
"""

import jax
import pytest
from numpy.testing import assert_allclose

from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestAlphaFeParamSpec:
    """Test that Parameters correctly handles met_alpha_fe."""

    def test_default_is_fixed_zero(self):
        """met_alpha_fe should default to Fixed(0.0)."""
        spec = Parameters(
            sfh_dpl_alpha=1.0,
            sfh_dpl_beta=1.0,
            sfh_dpl_tau_gyr=5.0,
            sfh_dpl_log_total_mass=1.0,
            mean_sfh_type="dpl",
        )
        dist = spec.get_distribution("met_alpha_fe")
        assert dist.is_fixed
        assert_allclose(dist.bounds[0], 0.0)

    def test_alpha_fe_in_all_params(self):
        """met_alpha_fe should appear in all_params."""
        spec = Parameters(mean_sfh_type="dpl")
        assert "met_alpha_fe" in spec.all_params

    def test_alpha_fe_free_when_set(self):
        """met_alpha_fe should be free when given a distribution."""
        from tengri.parameters.priors import Uniform

        spec = Parameters(
            mean_sfh_type="dpl",
            met_alpha_fe=Uniform(-0.2, 0.6),
        )
        assert "met_alpha_fe" in spec.free_params
        dist = spec.get_distribution("met_alpha_fe")
        assert not dist.is_fixed
        lo, hi = dist.bounds
        assert_allclose(lo, -0.2)
        assert_allclose(hi, 0.6)

    def test_backward_compatible_no_alpha(self):
        """Models without explicit met_alpha_fe should work unchanged."""
        spec = Parameters(
            mean_sfh_type="dpl",
            met_logzsol=-0.3,
        )
        assert "met_alpha_fe" in spec.fixed_params

    def test_sample_includes_alpha_fe(self):
        """Sampling should include met_alpha_fe."""
        from tengri.parameters.priors import Uniform

        spec = Parameters(
            mean_sfh_type="dpl",
            met_alpha_fe=Uniform(-0.2, 0.6),
        )
        params = spec.sample(jax.random.PRNGKey(42))
        assert "met_alpha_fe" in params
        val = float(params["met_alpha_fe"])
        assert -0.2 <= val <= 0.6
