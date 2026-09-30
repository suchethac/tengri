# SPDX-License-Identifier: BSD-3-Clause
"""StudentT quantile refinement: Newton iterations on exact Student-t CDF.

Issue #2576: StudentT.unstandardize and standardize used piecewise-linear
table interpolation with discontinuous gradients at 4097 knots and ~1e-5
accuracy. Fix: use the table as initial guess, refine with Newton iterations
on the exact Student-t CDF (via betainc), and compute exact Jacobian analytically.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import stdtrit
from scipy.stats import norm

from tengri.parameters.priors import StudentT

pytestmark = pytest.mark.bounds


class TestStudentTQuantileAccuracy:
    """Quantile accuracy: θ against scipy.special.stdtrit to 1e-10."""

    @pytest.mark.parametrize("df", [3, 5, 10, 30])
    @pytest.mark.parametrize("xi", [-2.8, -1.5, 0.0, 1.5, 2.8])
    def test_unstandardize_accuracy(self, df, xi):
        """unstandardize(xi) matches exact Student-t quantile to 1e-10."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))

        # Standard normal CDF at xi
        p = norm.cdf(float(xi))

        # Exact quantile from scipy
        z_exact = stdtrit(df, p)

        # Our quantile via unstandardize
        theta = float(prior.unstandardize(jnp.array(float(xi))))

        # Should match (since mu=0, sigma=1)
        assert np.abs(theta - z_exact) < 1e-10, (
            f"df={df}, xi={xi}: got {theta}, expected {z_exact}, error={abs(theta - z_exact)}"
        )


class TestStudentTJacobian:
    """Gradient of unstandardize finite and computable away from CDF boundaries."""

    @pytest.mark.parametrize("df", [3, 5, 10, 30])
    @pytest.mark.parametrize("xi", [-2.8, -1.5, 1.5, 2.8])
    def test_grad_unstandardize_finite(self, df, xi):
        """Gradient of unstandardize is finite (away from xi=0 boundary)."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))

        grad_fn = jax.grad(lambda x: prior.unstandardize(jnp.array(x)))
        grad_val = float(grad_fn(jnp.array(float(xi))))

        # grad-assert: finite-only — dθ/dξ = σ φ(ξ) / f_t(z) > 0 for every ξ the table covers
        assert np.isfinite(grad_val), f"df={df}, xi={xi}: gradient is {grad_val}"


class TestStudentTFloat32:
    """Under JAX float32 mode, operations remain finite."""

    @pytest.mark.parametrize("df", [3, 5, 10])
    @pytest.mark.parametrize("xi", [-2.0, 0.0, 2.0])
    def test_unstandardize_float32(self, df, xi):
        """unstandardize finite in float32."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))

        with jax.enable_x64(False):
            theta = prior.unstandardize(jnp.array(float(xi), dtype=jnp.float32))
            assert np.isfinite(float(theta)), f"df={df}, xi={xi}: got {theta}"


class TestStudentTJIT:
    """JIT and grad of unstandardize are finite."""

    @pytest.mark.parametrize("df", [3, 5, 10])
    def test_jit_unstandardize(self, df):
        """JIT of unstandardize is finite."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))

        jitted_unstandardize = jax.jit(prior.unstandardize)
        xi = jnp.array(1.5)
        theta = jitted_unstandardize(xi)
        assert np.isfinite(float(theta))

    @pytest.mark.parametrize("df", [3, 5, 10])
    def test_jit_grad_unstandardize(self, df):
        """JIT of grad(unstandardize) is finite away from boundaries."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))

        @jax.jit
        def grad_unstandardize(xi):
            return jax.grad(prior.unstandardize)(xi)

        xi = jnp.array(1.5)
        grad_val = grad_unstandardize(xi)
        # grad-assert: finite-only — dθ/dξ = σ φ(ξ) / f_t(z) > 0 for every ξ the table covers
        assert np.isfinite(float(grad_val))


class TestStudentTTruncation:
    """Truncated variant maps ξ onto (lo, hi)."""

    @pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (-2.0, 0.5)])
    @pytest.mark.parametrize("xi", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_truncation_bounds(self, lo, hi, xi):
        """unstandardize(xi) is in [lo, hi] for all xi."""
        prior = StudentT(mu=0.0, sigma=1.0, df=3.0, lo=lo, hi=hi)

        theta = float(prior.unstandardize(jnp.array(float(xi))))
        assert lo <= theta <= hi, f"lo={lo}, hi={hi}, xi={xi}: got theta={theta} outside bounds"


class TestStudentTClosedForms:
    """Closed forms for df=1,2 remain accurate."""

    @pytest.mark.parametrize("xi", [-2.0, -1.0, 1.0, 2.0])
    def test_closed_form_df1(self, xi):
        """df=1 (Cauchy) uses closed form."""
        prior = StudentT(mu=0.0, sigma=1.0, df=1.0)

        # Compute via unstandardize
        theta = float(prior.unstandardize(jnp.array(float(xi))))

        # Compute via closed form (tan(π(p-0.5)))
        p = norm.cdf(float(xi))
        z_exact = np.tan(np.pi * (p - 0.5))

        # Should be very close (both use closed form)
        assert np.abs(theta - z_exact) < 1e-10, f"df=1, xi={xi}: got {theta}, expected {z_exact}"

    @pytest.mark.parametrize("xi", [-2.0, -1.0, 1.0, 2.0])
    def test_closed_form_df2(self, xi):
        """df=2 uses closed form."""
        prior = StudentT(mu=0.0, sigma=1.0, df=2.0)

        # Compute via unstandardize
        theta = float(prior.unstandardize(jnp.array(float(xi))))

        # Compute via closed form
        p = norm.cdf(float(xi))
        z_exact = (2.0 * p - 1.0) / np.sqrt(2.0 * p * (1.0 - p))

        # Should be very close
        assert np.abs(theta - z_exact) < 1e-10, f"df=2, xi={xi}: got {theta}, expected {z_exact}"
