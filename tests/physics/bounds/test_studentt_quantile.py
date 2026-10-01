# SPDX-License-Identifier: BSD-3-Clause
"""StudentT quantile: exact analytic gradient, never autodiffed through betainc.

Issue #2576: StudentT.unstandardize/standardize used piecewise-linear table
interpolation for df not in {1, 2} with a discontinuous (staircase) gradient
and ~1e-5 value accuracy. A first attempt (table guess + fixed Newton
iterations, gradient obtained by autodiffing through jax.scipy.special.betainc)
introduced worse defects: the CDF's ``jnp.where`` guard at the removable
singularity z=0 zeroed the autodiff tangent there, giving a 5x-wrong Jacobian
at xi=0 (the declared prior midpoint used throughout this codebase) and NaN
gradients for xi in the open neighborhood around 0; forming the CDF as
``1 - 0.5*betainc(...)`` lost precision at the tails; and ``standardize``'s
final inverse-normal step silently swapped ``erfinv`` for ``ndtri``, a 1-ULP
drift from main for df in {1, 2}.

The fix: ``dz/dp = 1/f_t(z)`` and ``dF_t/dz = f_t(z)`` are supplied directly
via ``jax.custom_jvp`` (see ``_student_t_quantile``/``_student_t_cdf`` in
``tengri.parameters.priors``), so autodiff never touches the Newton
refinement or ``betainc`` at all -- the Jacobian a sampler differentiates is
exact everywhere, including exactly at xi=0. The CDF itself is reparameterized
as ``F_t(z) = 1/2 + sign(z)/2 * I_y(1/2, df/2)``, ``y = z^2/(df+z^2)``, which
removes the near-z=0 cancellation a naive ``x=df/(df+z^2)`` formula has
(measured: an un-reparameterized CDF is wrong by ~2e-9, not ~1e-16, at
df=100, xi=1e-7 -- exactly the class of input this file's xi=+-1e-7 cases
are chosen to hit). The Newton seed is a monotone cubic Hermite guess with
EXACT knot slopes ``1/f_t(z_k)`` (free at table-construction time), cutting
the refinement to a single Newton step.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import stdtrit
from scipy.stats import norm, t as scipy_t

from tengri.parameters.priors import StudentT

pytestmark = pytest.mark.bounds

# Mandatory grid (#2576 review): xi=0 and xi=+-1e-7 straddle the removable
# CDF singularity at z=0 and MUST be included, not carved out as "near a
# boundary".
_DF_VALUES = [3, 5, 10, 30, 100]
_XI_VALUES = [-4.5, -2.8, -1.5, -1e-7, 0.0, 1e-7, 1.5, 2.8, 4.5]

# Measured absolute-error floor against scipy at the grid above (worst case
# df=3, xi=+-4.5): 5.6e-10 in theta and 3.5e-9 in the Jacobian. It is the
# residual of the single Newton step in the extreme tail, where 1/f_t(z) is
# ~7e7 and amplifies ordinary float64 rounding of the CDF residual; the
# incomplete beta itself agrees with an mpmath(dps=50) reference there, and a
# second step would halve the residual at twice the forward cost. These
# tolerances carry a ~2x margin over the measured floor, not a carve-out.
_THETA_ATOL = 1e-9
_GRAD_ATOL = 5e-9


class TestStudentTQuantileAccuracy:
    """theta and the autodiff Jacobian against scipy to the measured floor."""

    @pytest.mark.parametrize("df", _DF_VALUES)
    @pytest.mark.parametrize("xi", _XI_VALUES)
    def test_unstandardize_matches_stdtrit(self, df, xi):
        """unstandardize(xi) matches scipy stdtrit(df, Phi(xi))."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))
        p = norm.cdf(xi)
        z_exact = float(stdtrit(df, p))

        theta = float(prior.unstandardize(jnp.array(xi)))

        assert np.isfinite(theta)
        assert abs(theta - z_exact) < _THETA_ATOL, (
            f"df={df}, xi={xi}: got {theta}, expected {z_exact}"
        )

    @pytest.mark.parametrize("df", _DF_VALUES)
    @pytest.mark.parametrize("xi", _XI_VALUES)
    def test_autodiff_jacobian_matches_analytic(self, df, xi):
        """jax.grad(unstandardize) matches d(theta)/d(xi) = phi(xi)/f_t(z) exactly.

        This is the test defect #1 (the review's finding) needed: at xi=0
        the removed-singularity CDF guard used to zero this gradient's
        tangent, and just off xi=0 the unguarded betainc derivative is
        infinite -> NaN. Both failure modes would show up here as a huge
        (not merely out-of-tolerance) deviation from the analytic value, not
        a finite-but-wrong one.
        """
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))
        p = norm.cdf(xi)
        z_exact = float(stdtrit(df, p))
        expected_grad = norm.pdf(xi) / scipy_t.pdf(z_exact, df)

        grad_val = float(jax.grad(lambda x: prior.unstandardize(x))(jnp.array(xi)))

        # A NaN or zero-vs-nonzero-expected grad_val fails this comparison outright
        # (abs(nan - x) < tol is False), so this single assertion settles finite
        # AND correct together -- strictly stronger than a separate isfinite check.
        assert abs(grad_val - expected_grad) < _GRAD_ATOL, (
            f"df={df}, xi={xi}: grad={grad_val}, expected={expected_grad}"
        )


class TestStudentTTruncatedGradient:
    """Truncated priors with z=0 inside (lo, hi): the case #1651's flat seam needs.

    dtheta/dxi = sigma * phi(xi) * (F_t(hi)-F_t(lo)) / f_t(z) -- the same
    analytic tangent as the untruncated case, scaled by the CDF-space
    truncation width, and exact at xi=0 for the same reason: the gradient
    never passes through betainc's own (singular-at-z=0) derivative.
    """

    @pytest.mark.parametrize("df", _DF_VALUES)
    @pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (-2.0, 0.5)])
    @pytest.mark.parametrize("xi", [-1e-7, 0.0, 1e-7])
    def test_value_in_bounds_and_gradient_matches_analytic(self, df, lo, hi, xi):
        """unstandardize(xi) in (lo, hi); its gradient matches the truncated pushforward."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df), lo=lo, hi=hi)
        pcdf_lo = float(scipy_t.cdf(lo, df))
        pcdf_hi = float(scipy_t.cdf(hi, df))
        p = pcdf_lo + norm.cdf(xi) * (pcdf_hi - pcdf_lo)
        z_exact = float(scipy_t.ppf(p, df))
        expected_grad = norm.pdf(xi) * (pcdf_hi - pcdf_lo) / scipy_t.pdf(z_exact, df)

        theta = float(prior.unstandardize(jnp.array(xi)))
        grad_val = float(jax.grad(lambda x: prior.unstandardize(x))(jnp.array(xi)))

        assert lo < theta < hi, f"df={df}, lo={lo}, hi={hi}, xi={xi}: theta={theta} out of bounds"
        # See the finite+correct note in TestStudentTQuantileAccuracy above.
        assert abs(grad_val - expected_grad) < _GRAD_ATOL, (
            f"df={df}, lo={lo}, hi={hi}, xi={xi}: grad={grad_val}, expected={expected_grad}"
        )


class TestStudentTRoundTrip:
    """standardize(unstandardize(xi)) == xi over the full table range."""

    @pytest.mark.parametrize("df", _DF_VALUES)
    def test_round_trip_4096_points(self, df):
        """Round trip holds to 1e-10 over 4096 points spanning [-4.5, 4.5]."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))
        xi_grid = jnp.linspace(-4.5, 4.5, 4096)

        theta = jax.vmap(prior.unstandardize)(xi_grid)
        xi_back = jax.vmap(prior.standardize)(theta)

        max_err = float(jnp.max(jnp.abs(xi_back - xi_grid)))
        assert max_err < 1e-10, f"df={df}: max round-trip error {max_err:.3e}"


class TestStudentTMainParity:
    """df in {1, 2} stay bit-identical to origin/main in both directions.

    The closed-form branches below are origin/main's exact, unmodified
    expressions (verified bit-identical against a checked-out copy of
    origin/main's ``src/tengri/parameters/priors.py`` during development, in
    both unstandardize and standardize, untruncated and truncated); restated
    here as a portable reference so this test carries no dependency on a
    git ref being resolvable at test-run time. ``standardize``'s shared final
    step, ``sqrt(2)*erfinv(2u-1)``, is the one line #2576's first attempt at
    a fix silently swapped for ``ndtri`` for every df including these two
    closed forms -- this test is what catches that regression.
    """

    @staticmethod
    def _reference_unstandardize(xi, df, mu, sigma, pcdf_lo, pcdf_hi):
        phi_xi = 0.5 * (1.0 + jax.scipy.special.erf(xi / jnp.sqrt(2.0)))
        p = pcdf_lo + phi_xi * (pcdf_hi - pcdf_lo)
        p = jnp.clip(p, 1e-7, 1.0 - 1e-7)
        if df == 1.0:
            z = jnp.tan(jnp.pi * (p - 0.5))
        else:
            z = (2.0 * p - 1.0) / jnp.sqrt(2.0 * p * (1.0 - p))
        return mu + sigma * z

    @staticmethod
    def _reference_standardize(theta, df, mu, sigma, pcdf_lo, pcdf_hi):
        z = (theta - mu) / sigma
        if df == 1.0:
            p = 0.5 + jnp.arctan(z) / jnp.pi
        else:
            p = 0.5 * (1.0 + z / jnp.sqrt(2.0 + z * z))
        u = (p - pcdf_lo) / (pcdf_hi - pcdf_lo)
        u = jnp.clip(u, 1e-7, 1.0 - 1e-7)
        return jnp.sqrt(2.0) * jax.scipy.special.erfinv(2.0 * u - 1.0)

    @pytest.mark.parametrize("df", [1.0, 2.0])
    @pytest.mark.parametrize("lo,hi", [(float("-inf"), float("inf")), (-1.0, 1.0)])
    def test_bit_identical_both_directions(self, df, lo, hi):
        """unstandardize and standardize match the reference closed forms exactly."""
        prior = StudentT(mu=0.3, sigma=1.7, df=df, lo=lo, hi=hi)
        pcdf_lo = prior._pcdf_lo
        pcdf_hi = prior._pcdf_hi
        xi_grid = jnp.linspace(-6.0, 6.0, 4001)

        theta = jax.vmap(prior.unstandardize)(xi_grid)
        theta_ref = jax.vmap(
            lambda xi: self._reference_unstandardize(xi, df, 0.3, 1.7, pcdf_lo, pcdf_hi)
        )(xi_grid)
        assert jnp.array_equal(theta, theta_ref), f"df={df}, lo={lo}, hi={hi}: unstandardize drift"

        xi_back = jax.vmap(prior.standardize)(theta)
        xi_back_ref = jax.vmap(
            lambda th: self._reference_standardize(th, df, 0.3, 1.7, pcdf_lo, pcdf_hi)
        )(theta_ref)
        assert jnp.array_equal(xi_back, xi_back_ref), (
            f"df={df}, lo={lo}, hi={hi}: standardize drift"
        )


class TestStudentTFloat32:
    """float32: finite values and gradients, accurate to float32 epsilon."""

    @pytest.mark.parametrize("df", _DF_VALUES[:3])
    @pytest.mark.parametrize("xi", [-2.0, -1e-7, 0.0, 1e-7, 2.0])
    def test_value_and_grad_finite_and_accurate(self, df, xi):
        """unstandardize and its gradient are finite in float32, matching float64 to ~1e-5."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))
        p = norm.cdf(xi)
        z_exact = float(stdtrit(df, p))
        expected_grad = norm.pdf(xi) / scipy_t.pdf(z_exact, df)

        with jax.enable_x64(False):
            xi32 = jnp.array(xi, dtype=jnp.float32)
            theta = prior.unstandardize(xi32)
            grad_val = jax.grad(lambda x: prior.unstandardize(x))(xi32)

        assert theta.dtype == jnp.float32
        assert grad_val.dtype == jnp.float32
        # A NaN or zero-vs-nonzero-expected value fails these comparisons
        # outright, so finiteness is not asserted separately -- see the
        # finite+correct note in TestStudentTQuantileAccuracy above. Measured
        # worst-case float32 error ~1.7e-6 (value) / ~1.7e-6 (grad) across
        # this grid; 1e-5 carries a ~5x margin.
        assert abs(float(theta) - z_exact) < 1e-5, (
            f"df={df}, xi={xi}: theta={float(theta)}, expected={z_exact}"
        )
        assert abs(float(grad_val) - expected_grad) < 1e-5, (
            f"df={df}, xi={xi}: grad={float(grad_val)}, expected={expected_grad}"
        )


class TestStudentTDensityIntegratesToOne:
    """log_prob's exp() integrates to 1: the normalization constant is exact."""

    @pytest.mark.parametrize("df", _DF_VALUES)
    @pytest.mark.parametrize("lo,hi", [(float("-inf"), float("inf")), (-1.0, 1.0)])
    def test_density_integrates_to_one(self, df, lo, hi):
        """Numerically integrating exp(log_prob(x)) over (lo, hi) gives 1."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df), lo=lo, hi=hi)

        def density(x):
            return float(jnp.exp(prior.log_prob(jnp.array(x))))

        a = lo if np.isfinite(lo) else -200.0
        b = hi if np.isfinite(hi) else 200.0
        integral, _ = quad(density, a, b, limit=200)

        assert abs(integral - 1.0) < 1e-6, f"df={df}, lo={lo}, hi={hi}: integral={integral}"


class TestStudentTJIT:
    """JIT and grad of unstandardize are finite, including at xi=0."""

    @pytest.mark.parametrize("df", _DF_VALUES[:3])
    @pytest.mark.parametrize("xi", [0.0, 1.5])
    def test_jit_unstandardize(self, df, xi):
        """JIT of unstandardize is finite."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))

        jitted_unstandardize = jax.jit(prior.unstandardize)
        theta = jitted_unstandardize(jnp.array(xi))
        assert np.isfinite(float(theta))

    @pytest.mark.parametrize("df", _DF_VALUES[:3])
    @pytest.mark.parametrize("xi", [0.0, 1.5])
    def test_jit_grad_unstandardize(self, df, xi):
        """JIT of grad(unstandardize) is finite, including exactly at xi=0."""
        prior = StudentT(mu=0.0, sigma=1.0, df=float(df))

        @jax.jit
        def grad_unstandardize(x):
            return jax.grad(prior.unstandardize)(x)

        grad_val = grad_unstandardize(jnp.array(xi))
        # grad-assert: finite-only -- exactness against the analytic Jacobian
        # is checked (not merely finiteness) in
        # TestStudentTQuantileAccuracy.test_autodiff_jacobian_matches_analytic;
        # this class checks only that the custom_jvp rule survives jax.jit.
        assert np.isfinite(float(grad_val))


class TestStudentTClosedForms:
    """Closed forms for df=1,2 remain accurate (unaffected by the df-table fix)."""

    @pytest.mark.parametrize("xi", [-2.0, -1.0, 1.0, 2.0])
    def test_closed_form_df1(self, xi):
        """df=1 (Cauchy) uses closed form."""
        prior = StudentT(mu=0.0, sigma=1.0, df=1.0)
        theta = float(prior.unstandardize(jnp.array(xi)))
        p = norm.cdf(xi)
        z_exact = np.tan(np.pi * (p - 0.5))
        assert abs(theta - z_exact) < 1e-10, f"df=1, xi={xi}: got {theta}, expected {z_exact}"

    @pytest.mark.parametrize("xi", [-2.0, -1.0, 1.0, 2.0])
    def test_closed_form_df2(self, xi):
        """df=2 uses closed form."""
        prior = StudentT(mu=0.0, sigma=1.0, df=2.0)
        theta = float(prior.unstandardize(jnp.array(xi)))
        p = norm.cdf(xi)
        z_exact = (2.0 * p - 1.0) / np.sqrt(2.0 * p * (1.0 - p))
        assert abs(theta - z_exact) < 1e-10, f"df=2, xi={xi}: got {theta}, expected {z_exact}"


class TestStudentTTruncation:
    """Truncated variant maps xi onto (lo, hi)."""

    @pytest.mark.parametrize("lo,hi", [(-1.0, 1.0), (-2.0, 0.5)])
    @pytest.mark.parametrize("xi", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_truncation_bounds(self, lo, hi, xi):
        """unstandardize(xi) is in [lo, hi] for all xi."""
        prior = StudentT(mu=0.0, sigma=1.0, df=3.0, lo=lo, hi=hi)
        theta = float(prior.unstandardize(jnp.array(float(xi))))
        assert lo <= theta <= hi, f"lo={lo}, hi={hi}, xi={xi}: got theta={theta} outside bounds"
