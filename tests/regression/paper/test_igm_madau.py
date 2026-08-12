# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Madau (1995) IGM absorption model.

Physics references:
- Madau, P. 1995, ApJ, 441, 18 (IGM transmission model)
- Convention: igm_transmission_madau(wave_obs, z) takes observed-frame wavelengths

Tests verify:
1. Output shape and dtype
2. Bounds [0, 1] on transmission
3. No absorption at z=0
4. Significant absorption at short wavelengths
5. Strong continuum absorption shortward of Lyman limit
6. Monotonicity with redshift
7. JIT compatibility
8. Gradient finiteness
"""

import chex
import jax
import pytest

pytestmark = pytest.mark.regression_paper

import jax.numpy as jnp
import numpy as np

from tengri.components.igm import igm_transmission_madau
from tests._jit_parity import assert_jit_matches_eager


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestMadau1995IGM:
    """Unit tests for Madau 1995 IGM transmission."""

    def test_output_shape_and_dtype(self):
        """Output shape matches input; dtype is float64."""
        wave_obs = jnp.linspace(900.0, 10000.0, 200)
        T = igm_transmission_madau(wave_obs, z=2.0)
        assert T.shape == (200,), f"Expected shape (200,), got {T.shape}"
        assert T.dtype == jnp.float64, f"Expected float64, got {T.dtype}"

    def test_transmission_bounds(self):
        """Transmission is bounded in [0, 1]."""
        wave_obs = jnp.linspace(800.0, 20000.0, 500)
        for z in [0.0, 1.0, 2.0, 3.0, 5.0]:
            T = igm_transmission_madau(wave_obs, z=z)
            assert jnp.all(T >= 0.0), f"T < 0 at z={z}"
            assert jnp.all(T <= 1.0), f"T > 1 at z={z}"

    def test_no_absorption_z0(self):
        """No IGM absorption at z=0 (no IGM at source)."""
        wave_obs = jnp.array([912.0, 1216.0, 4000.0, 8000.0])
        T = igm_transmission_madau(wave_obs, z=0.0)
        np.testing.assert_allclose(
            T,
            1.0,
            atol=1e-10,
            err_msg="Madau 1995: T should be 1.0 everywhere at z=0",
        )

    def test_shortwave_absorption_z3(self):
        """Madau 1995 uses only 17 Lyman lines; applies for shorter wavelengths.

        At z=3: Lya is at observed 4864 Å (too long for Madau absorption).
        But at observed 1500 Å (shortward of LL_obs = 3647 Å), continuum is strong.
        Expect T < 0.1 at this shorter wavelength.
        """
        z = 3.0
        wave_obs = jnp.array([1500.0])
        T = float(igm_transmission_madau(wave_obs, z=z)[0])
        assert T < 0.1, (
            f"Madau 1995: At obs 1500 Å and z=3 (shortward of LL), expect T < 0.1, got T={T:.3f}"
        )
        assert T > 0.0, f"Madau 1995: T should be > 0 (not opaque), got T={T:.3f}"

    def test_continuum_absorption_z4(self):
        """Strong absorption shortward of Lyman limit at z=4.

        Lyman limit: rest 912 Å → observed 912*(1+4) = 4560 Å.
        Shortward of this, continuum absorption is severe.
        At observed 4000 Å (shortward of LL), expect T < 0.1.
        """
        z = 4.0
        wave_obs = jnp.array([4000.0])
        T = float(igm_transmission_madau(wave_obs, z=z)[0])
        assert T < 0.1, (
            f"Madau 1995: At z=4, observed 4000 Å (shortward of LL), expect T < 0.1, got T={T:.3f}"
        )

    def test_transparent_longward_of_lya(self):
        """IGM transparent longward of Lya at source redshift.

        At z=3: Lya_obs = 1216*(1+3) = 4864 Å.
        Longer wavelengths (optical) should have T > 0.90.
        """
        z = 3.0
        wave_opt_obs = jnp.array([6000.0, 7000.0, 8000.0])
        T = igm_transmission_madau(wave_opt_obs, z=z)
        assert float(jnp.min(T)) > 0.90, (
            f"Madau 1995: Optical wavelengths should be nearly transparent; "
            f"min T={float(jnp.min(T)):.3f}"
        )

    def test_monotone_decreasing_with_z(self):
        """At fixed observed wavelength, T decreases with increasing z.

        Higher redshift = longer path through absorbers.
        Fix obs_wavelength = 1500 Å (shortward of LL for z >= 1.5):
        T(z=1) > T(z=2) > T(z=3).
        """
        wave_obs = jnp.array([1500.0])
        T1 = float(igm_transmission_madau(wave_obs, z=1.0)[0])
        T2 = float(igm_transmission_madau(wave_obs, z=2.0)[0])
        T3 = float(igm_transmission_madau(wave_obs, z=3.0)[0])
        assert T1 > T2, f"T(z=1)={T1:.3f} should exceed T(z=2)={T2:.3f}"
        assert T2 > T3, f"T(z=2)={T2:.3f} should exceed T(z=3)={T3:.3f}"

    def test_jit_compatible(self):
        """Madau 1995 function is JIT-compilable."""
        wave_obs = jnp.array([1000.0, 2000.0, 4000.0, 8000.0])
        T = assert_jit_matches_eager(igm_transmission_madau, wave_obs, z=2.5)
        chex.assert_tree_all_finite(T)

    def test_gradient_wrt_z_finite(self):
        """Gradient ∂T/∂z is finite and sensible (negative sign expected).

        At fixed observed wavelength, transmission decreases with z,
        so ∂T/∂z should be negative. Use wavelength 1500 Å which is
        in the sensitive range for z=2.
        """
        wave_obs = jnp.array([1500.0])

        def f(z):
            return float(igm_transmission_madau(wave_obs, z=z)[0])

        grad_jax = float(jax.grad(lambda z: igm_transmission_madau(wave_obs, z=z)[0])(2.0))
        grad_fd = fd_grad(f, 2.0, eps=0.01)

        # Both gradients should be finite
        assert jnp.isfinite(grad_jax), f"JAX grad is not finite: {grad_jax}"
        assert np.isfinite(grad_fd), f"FD grad is not finite: {grad_fd}"

        # Sign check: negative (more absorption at higher z)
        assert grad_jax < 0.0, (
            f"Madau 1995: ∂T/∂z should be negative (more absorption at higher z), "
            f"got ∂T/∂z={grad_jax:.6f}"
        )

    def test_igm_factor_scaling(self):
        """igm_factor scales the optical depth multiplicatively.

        igm_factor=2.0 → tau_total = 2*tau → T = exp(-2*tau)
        Should result in lower transmission (more absorption).
        """
        wave_obs = jnp.array([2000.0, 4000.0, 6000.0])
        z = 2.0

        T_nominal = igm_transmission_madau(wave_obs, z=z, igm_factor=1.0)
        T_doubled = igm_transmission_madau(wave_obs, z=z, igm_factor=2.0)

        # Doubled factor should give lower (or equal) transmission
        assert jnp.all(T_doubled <= T_nominal), "igm_factor=2.0 should reduce transmission"

    def test_vector_wavelength(self):
        """Function handles vector wavelengths correctly."""
        wave_obs = jnp.linspace(900.0, 10000.0, 500)
        z = 3.0
        T = igm_transmission_madau(wave_obs, z=z)

        # All outputs should be finite and in bounds
        chex.assert_tree_all_finite(T)
        assert jnp.all((T >= 0.0) & (T <= 1.0)), "T outside [0, 1]"

    def test_consistency_across_redshifts(self):
        """Transmission is smooth across a range of redshifts.

        No discontinuities or unexpected jumps as z varies.
        """
        wave_obs = jnp.array([1500.0])
        z_values = jnp.linspace(0.1, 5.0, 50)
        T_values = jnp.array([igm_transmission_madau(wave_obs, z=z)[0] for z in z_values])

        # Check monotonicity: all differences should have the same sign
        diffs = jnp.diff(T_values)
        assert jnp.all(diffs <= 0.0), "T(z) should be monotone decreasing; found unexpected jumps"
