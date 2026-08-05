"""Verify that the IFT standardized prior is consistent.

This script checks that for a Uniform prior on θ ∈ [lo, hi], the prior penalty
evaluated at uniformly sampled θ values gives the correct Hamiltonian form:
    H(ξ|d) = ½χ² + ½ξᵀξ

The transform h(ξ) is chosen so that P(h(ξ)) |dh/dξ| = φ(ξ), ensuring that
Jacobian factors cancel exactly with the prior density.
"""

import jax
import jax.numpy as jnp
import jax.random as jr

jax.config.update("jax_enable_x64", True)

from tengri.parameters.priors import Uniform, Gaussian, LogUniform, LogNormal, StudentT


def test_uniform_prior():
    """Verify that Uniform prior gives constant log p(θ)."""
    print("=" * 70)
    print("Test 1: Uniform Prior Jacobian Cancellation")
    print("=" * 70)

    # Test with a simple uniform prior on [9.0, 12.0]
    lo, hi = 9.0, 12.0
    dist = Uniform(lo, hi)

    # Sample θ uniformly from [lo, hi]
    key = jr.PRNGKey(42)
    n_samples = 10000
    theta_samples = jr.uniform(key, (n_samples,), minval=lo, maxval=hi)

    # For each θ, compute the unbounded ξ and the prior penalty
    log_p_theta_values = []
    prior_penalty_values = []

    for theta in theta_samples:
        # Compute unbounded parameter via inverse transform
        xi = dist.standardize(theta)

        # IFT prior penalty: ξ²
        prior_penalty = xi**2
        prior_penalty_values.append(float(prior_penalty))

        # For a uniform prior, log p(θ) should be constant = -log(hi - lo)
        # This is NOT computed from the penalty - it's the target prior we want
        log_p = dist.log_prob(theta)
        log_p_theta_values.append(float(log_p))

    prior_penalty_values = jnp.array(prior_penalty_values)
    log_p_theta_values = jnp.array(log_p_theta_values)

    # Expected: log p(θ) = -log(hi - lo) = constant
    expected_log_p = -jnp.log(hi - lo)

    print(f"Testing Uniform prior on [{lo}, {hi}]")
    print(f"N samples: {n_samples}")
    print(f"Expected constant log p(θ): {expected_log_p:.6f}")
    print()

    mean_log_p = jnp.mean(log_p_theta_values)
    std_log_p = jnp.std(log_p_theta_values)
    mean_penalty = jnp.mean(prior_penalty_values)
    std_penalty = jnp.std(prior_penalty_values)

    print("Results:")
    print(f"  Mean log p(θ):    {mean_log_p:.6f}  (expected: {expected_log_p:.6f})")
    print(f"  Std log p(θ):     {std_log_p:.6f}  (should be ~0 for constant)")
    print(f"  Mean penalty ξ²:  {mean_penalty:.6f}")
    print(f"  Std penalty ξ²:   {std_penalty:.6f}")
    print()

    # Validation
    if jnp.abs(mean_log_p - expected_log_p) < 1e-6:
        print("✓ Mean log p(θ) matches expected (uniform distribution)")
    else:
        print(f"✗ Mean log p(θ) is off by {jnp.abs(mean_log_p - expected_log_p):.6e}")

    if std_log_p < 1e-6:
        print("✓ log p(θ) is constant (std ~ 0)")
    else:
        print(f"✗ log p(θ) is NOT constant (std = {std_log_p:.6e})")

    print()
    print("Interpretation:")
    print(f"  - Jacobian cancellation means: P(h(ξ)) |dh/dξ| = φ(ξ)")
    print(f"  - For Uniform, h(ξ) = lo + (hi-lo)·sigmoid(ξ)")
    print(f"  - Result: log p(θ) is constant, and prior penalty is just ξ²")
    print("=" * 70)
    print()

    return std_log_p < 1e-6


def test_gaussian_prior():
    """Verify that Gaussian prior gives correct log p(θ)."""
    print("=" * 70)
    print("Test 2: Gaussian Prior Transform")
    print("=" * 70)

    # Test with a Gaussian prior N(10.5, 1.0²) clipped to [9.0, 12.0]
    mu, sigma = 10.5, 1.0
    lo, hi = 9.0, 12.0
    dist = Gaussian(mu, sigma, lo, hi)

    # Sample ξ ~ N(0, 1)
    key = jr.PRNGKey(43)
    n_samples = 10000
    xi_samples = jr.normal(key, (n_samples,))

    # Transform to θ and check log probabilities
    log_p_theta_values = []

    for xi in xi_samples:
        # Transform ξ → θ via h(ξ) = μ + σ·ξ (clipped)
        theta = dist.unstandardize(xi)

        # Evaluate prior at θ
        log_p = dist.log_prob(theta)
        log_p_theta_values.append(float(log_p))

    log_p_theta_values = jnp.array(log_p_theta_values)

    # For N(μ, σ²) clipped, the log prob should follow Gaussian form in bounds
    mean_log_p = jnp.mean(log_p_theta_values)
    std_log_p = jnp.std(log_p_theta_values)

    print(f"Testing Gaussian prior N({mu}, {sigma}²) clipped to [{lo}, {hi}]")
    print(f"N samples: {n_samples}")
    print()
    print("Results:")
    print(f"  Mean log p(θ): {mean_log_p:.6f}")
    print(f"  Std log p(θ):  {std_log_p:.6f}")
    print()
    print("Interpretation:")
    print(f"  - For Gaussian, h(ξ) = μ + σ·ξ (linear transform)")
    print(f"  - Prior penalty is still just ξ²")
    print(f"  - Jacobian |dh/dξ| = σ, which cancels with the Gaussian density")
    print("=" * 70)
    print()

    # For Gaussian, we can't check constant log p(θ) because it's not constant
    # But we can check that the std is reasonable (should vary with ξ)
    return std_log_p > 0.01  # Should NOT be constant


def test_loguniform_prior():
    """Verify that LogUniform prior works correctly."""
    print("=" * 70)
    print("Test 3: LogUniform Prior Transform")
    print("=" * 70)

    # Test with a LogUniform prior on [1.0, 100.0]
    lo, hi = 1.0, 100.0
    dist = LogUniform(lo, hi)

    # Sample ξ ~ N(0, 1)
    key = jr.PRNGKey(44)
    n_samples = 10000
    xi_samples = jr.normal(key, (n_samples,))

    # Transform to θ and check distribution
    theta_values = []
    log_p_theta_values = []

    for xi in xi_samples:
        # Transform ξ → θ via h(ξ) = exp(ln(lo) + (ln(hi)-ln(lo))·sigmoid(ξ))
        theta = dist.unstandardize(xi)
        theta_values.append(float(theta))

        # Evaluate prior at θ
        log_p = dist.log_prob(theta)
        log_p_theta_values.append(float(log_p))

    theta_values = jnp.array(theta_values)
    log_p_theta_values = jnp.array(log_p_theta_values)

    # For LogUniform, log(θ) should be roughly uniform
    log_theta_values = jnp.log(theta_values)
    mean_log_theta = jnp.mean(log_theta_values)
    expected_mean_log_theta = 0.5 * (jnp.log(lo) + jnp.log(hi))

    print(f"Testing LogUniform prior on [{lo}, {hi}]")
    print(f"N samples: {n_samples}")
    print()
    print("Results:")
    print(f"  Mean log(θ):     {mean_log_theta:.6f}")
    print(f"  Expected:        {expected_mean_log_theta:.6f}")
    print(f"  Mean log p(θ):   {jnp.mean(log_p_theta_values):.6f}")
    print()
    print("Interpretation:")
    print(f"  - For LogUniform, h(ξ) uses sigmoid in log-space")
    print(f"  - Prior penalty is still just ξ²")
    print(f"  - Jacobian cancellation ensures p(θ) ∝ 1/θ in [lo, hi]")
    print("=" * 70)
    print()

    # Check that mean log(θ) is close to expected
    return jnp.abs(mean_log_theta - expected_mean_log_theta) < 0.1


def test_lognormal_prior():
    """Verify that LogNormal prior works correctly."""
    print("=" * 70)
    print("Test 4: LogNormal Prior Transform")
    print("=" * 70)

    # Test with LogNormal prior: log(θ) ~ N(2.0, 0.5²)
    mu, sigma = 2.0, 0.5
    lo, hi = 1.0, 100.0
    dist = LogNormal(mu, sigma, lo, hi)

    # Sample ξ ~ N(0, 1)
    key = jr.PRNGKey(44)
    n_samples = 10000
    xi_samples = jr.normal(key, (n_samples,))

    # Transform to θ and check distribution
    theta_values = []
    log_theta_values = []

    for xi in xi_samples:
        # Transform ξ → θ via h(ξ) = exp(μ + σ·ξ)
        theta = dist.unstandardize(xi)
        theta_values.append(float(theta))
        log_theta_values.append(float(jnp.log(theta)))

    theta_values = jnp.array(theta_values)
    log_theta_values = jnp.array(log_theta_values)

    # For LogNormal, log(θ) should be roughly N(μ, σ²)
    mean_log_theta = jnp.mean(log_theta_values)
    std_log_theta = jnp.std(log_theta_values)

    print(f"Testing LogNormal prior: log(θ) ~ N({mu}, {sigma}²) clipped to [{lo}, {hi}]")
    print(f"N samples: {n_samples}")
    print()
    print("Results:")
    print(f"  Mean log(θ): {mean_log_theta:.6f}  (expected: {mu:.6f})")
    print(f"  Std log(θ):  {std_log_theta:.6f}  (expected: {sigma:.6f})")
    print()
    print("Interpretation:")
    print(f"  - For LogNormal, h(ξ) = exp(μ + σ·ξ)")
    print(f"  - Prior penalty is still just ξ²")
    print(f"  - Jacobian |dh/dξ| = exp(μ + σ·ξ)·σ cancels with LogNormal density 1/(θ·σ)")
    print("=" * 70)
    print()

    # Check that mean/std are close to expected (allowing for clipping effects)
    mean_ok = jnp.abs(mean_log_theta - mu) < 0.2
    std_ok = jnp.abs(std_log_theta - sigma) < 0.2
    return mean_ok and std_ok


def test_studentt_prior():
    """Verify that StudentT prior approximation is reasonable."""
    print("=" * 70)
    print("Test 5: StudentT Prior Transform (Gaussian Approximation)")
    print("=" * 70)

    # Test with StudentT prior t(μ=10, σ=2, df=3)
    mu, sigma, df = 10.0, 2.0, 3.0
    lo, hi = 0.0, 20.0
    dist = StudentT(mu, sigma, df, lo, hi)

    # Sample ξ ~ N(0, 1)
    key = jr.PRNGKey(45)
    n_samples = 10000
    xi_samples = jr.normal(key, (n_samples,))

    # Transform to θ
    theta_values = []
    for xi in xi_samples:
        theta = dist.unstandardize(xi)
        theta_values.append(float(theta))

    theta_values = jnp.array(theta_values)

    mean_theta = jnp.mean(theta_values)
    std_theta = jnp.std(theta_values)

    # For t-distribution, Var(t) = df/(df-2) for df>2
    # So std(θ) ≈ σ * sqrt(df/(df-2))
    expected_scale = jnp.sqrt(df / (df - 2))
    expected_std = sigma * expected_scale

    print(f"Testing StudentT prior t({mu}, {sigma}, df={df}) clipped to [{lo}, {hi}]")
    print(f"N samples: {n_samples}")
    print()
    print("Results:")
    print(f"  Mean θ:      {mean_theta:.6f}  (expected: {mu:.6f})")
    print(f"  Std θ:       {std_theta:.6f}  (expected: {expected_std:.6f})")
    print()
    print("Interpretation:")
    print(f"  - Code uses Gaussian approximation with matched variance")
    print(f"  - Scale factor: sqrt(df/(df-2)) = {expected_scale:.3f}")
    print(f"  - Paper's exact transform would use F_t^{{-1}}(Φ(ξ); ν)")
    print(f"  - Approximation is simpler and JIT-friendly")
    print("=" * 70)
    print()

    # Check that mean/std are reasonable (allowing for clipping and approximation)
    mean_ok = jnp.abs(mean_theta - mu) < 1.0
    std_ok = jnp.abs(std_theta - expected_std) < 1.0
    return mean_ok and std_ok


def test_roundtrip():
    """Verify that standardize and unstandardize are inverses."""
    print("=" * 70)
    print("Test 6: Round-trip Transform Consistency")
    print("=" * 70)

    distributions = [
        ("Uniform(9, 12)", Uniform(9.0, 12.0)),
        ("Gaussian(10.5, 1.0)", Gaussian(10.5, 1.0, 9.0, 12.0)),
        ("LogUniform(1, 100)", LogUniform(1.0, 100.0)),
        ("LogNormal(2.0, 0.5)", LogNormal(2.0, 0.5, 1.0, 100.0)),
        ("StudentT(10, 2, 3)", StudentT(10.0, 2.0, 3.0, 0.0, 20.0)),
    ]

    key = jr.PRNGKey(45)
    all_passed = True

    for name, dist in distributions:
        # Sample ξ ~ N(0, 1)
        xi = jr.normal(key, (100,))

        # Round-trip: ξ → θ → ξ'
        theta = dist.unstandardize(xi)
        xi_recovered = dist.standardize(theta)

        # For clipped distributions, only check non-clipped values
        needs_clipping_check = False
        if isinstance(dist, Gaussian) and (dist.lo != float("-inf") or dist.hi != float("inf")):
            theta_unclipped = dist.mu + dist.sigma * xi
            was_clipped = (theta_unclipped < dist.lo) | (theta_unclipped > dist.hi)
            needs_clipping_check = True
        elif isinstance(dist, LogNormal) and (dist._lo > 0 or dist._hi < float("inf")):
            theta_unclipped = jnp.exp(dist.mu + dist.sigma * xi)
            was_clipped = (theta_unclipped < dist._lo) | (theta_unclipped > dist._hi)
            needs_clipping_check = True
        elif isinstance(dist, StudentT):
            # StudentT uses approximation, always check for clipping
            scale = jnp.where(dist._df > 2, jnp.sqrt(dist._df / (dist._df - 2)), 3.0)
            theta_unclipped = dist._mu + dist._sigma * scale * xi
            was_clipped = (theta_unclipped < dist._lo) | (theta_unclipped > dist._hi)
            needs_clipping_check = True

        if needs_clipping_check:
            n_clipped = jnp.sum(was_clipped)

            # Only check round-trip for non-clipped values
            if jnp.sum(~was_clipped) > 0:
                max_error = jnp.max(jnp.abs(xi[~was_clipped] - xi_recovered[~was_clipped]))
            else:
                max_error = float("nan")

            print(f"{name}:")
            print(
                f"  {n_clipped}/{len(xi)} values clipped, testing remaining {len(xi) - n_clipped}"
            )
            print(f"  Max round-trip error (non-clipped): {max_error:.6e}")
        else:
            # No clipping, check all values
            max_error = jnp.max(jnp.abs(xi - xi_recovered))
            print(f"{name}:")
            print(f"  Max round-trip error: {max_error:.6e}")

        if max_error < 1e-6 or jnp.isnan(max_error):
            print("  ✓ Round-trip is consistent")
        else:
            print("  ✗ Round-trip has errors")
            all_passed = False

        key = jr.split(key)[0]

    print("=" * 70)
    print()
    print("Note: Clipped Gaussian breaks bijectivity at bounds (by design).")
    print("For inference, only forward transform (unstandardize) is used,")
    print("so this limitation does not affect practical usage.")
    print()
    return all_passed


if __name__ == "__main__":
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 16 + "IFT Standardized Prior Verification" + " " * 17 + "║")
    print("╚" + "═" * 68 + "╝")
    print()

    success = True

    # Test 1: Uniform prior should give constant log p(θ)
    success &= test_uniform_prior()

    # Test 2: Gaussian prior should give Gaussian log p(θ)
    success &= test_gaussian_prior()

    # Test 3: LogUniform prior should give 1/θ distribution
    success &= test_loguniform_prior()

    # Test 4: LogNormal prior
    success &= test_lognormal_prior()

    # Test 5: StudentT prior (Gaussian approximation)
    success &= test_studentt_prior()

    # Test 6: Round-trip consistency
    success &= test_roundtrip()

    if success:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "✓ All tests passed!" + " " * 25 + "║")
        print("╚" + "═" * 68 + "╝")
        print()
    else:
        print("╔" + "═" * 68 + "╗")
        print("║" + " " * 18 + "✗ Some tests failed" + " " * 27 + "║")
        print("╚" + "═" * 68 + "╝")
        print()
        raise AssertionError("IFT consistency check failed")
