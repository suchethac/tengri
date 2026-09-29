# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #2524: double_powerlaw time reversal.

The public double_powerlaw function evaluated the Carnall+2018 double power law
using lookback time as if it were cosmic time since formation, producing a
mirror-imaged SFH. This test verifies that the fix (adding explicit `age`
parameter and computing T = age - t_lookback) corrects the behavior.

Reference: Carnall et al. (2018), MNRAS 480, 4379 (arXiv:1712.04452).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import trapezoid

pytestmark = pytest.mark.regression_bug


@pytest.mark.parametrize(
    "alpha,beta,tau_yr,age_yr",
    [
        (1.5, 1.0, 3.0e9, 8.0e9),
        (3.0, 0.5, 1.0e9, 5.0e9),
        (0.7, 2.0, 5.0e9, 12.0e9),
    ],
)
@pytest.mark.parametrize("n_t_lookback", [2000, 20000])
def test_double_powerlaw_matches_dpl_shape(alpha, beta, tau_yr, age_yr, n_t_lookback):
    """Fixed double_powerlaw(t_lb, age, ...) matches dpl shape exactly.

    After the fix, double_powerlaw should evaluate the Carnall+2018 bare shape
    at cosmic time T = age - t_lookback, with no renormalization. This shape
    should be identical (up to numerical precision) to the shape computed by
    dpl when both use the same parameters.
    """
    from tengri.components.stellar.sfh.mean_sfh import double_powerlaw, dpl

    t_lookback = jnp.linspace(1.0e6, age_yr, n_t_lookback)

    # Compute the bare shape using the fixed double_powerlaw
    # (with age parameter to compute T = age - t_lookback).
    shape_double_powerlaw = np.asarray(
        double_powerlaw(t_lookback, alpha=alpha, beta=beta, tau=tau_yr, norm=1.0, age=age_yr)
    )

    # Compute via dpl with no mass renormalization (log_total_mass=0 → 10^0 = 1 M⊙ integrated).
    shape_dpl = np.asarray(
        dpl(t_lookback, alpha=alpha, beta=beta, tau=tau_yr, age=age_yr, log_total_mass=0.0)
    )

    # Normalize both to unit integral for shape comparison
    dt_yr = np.mean(np.diff(t_lookback))
    int_dpow = trapezoid(shape_double_powerlaw, t_lookback)
    int_dpl = trapezoid(shape_dpl, t_lookback)

    shape_double_powerlaw_norm = (
        shape_double_powerlaw / int_dpow if int_dpow > 0 else shape_double_powerlaw
    )
    shape_dpl_norm = shape_dpl / int_dpl if int_dpl > 0 else shape_dpl

    # Assertion (1): Half-sum absolute difference is < 1e-3
    dist_direct = 0.5 * np.sum(np.abs(shape_double_powerlaw_norm - shape_dpl_norm)) * dt_yr
    assert dist_direct < 1e-3, (
        f"Shape distance {dist_direct:.6f} >= 1e-3 for (α={alpha}, β={beta}, "
        f"τ={tau_yr:.1e}, age={age_yr:.1e}, n={n_t_lookback})"
    )


@pytest.mark.parametrize(
    "alpha,beta,tau_yr,age_yr",
    [
        (1.5, 1.0, 3.0e9, 8.0e9),
        (3.0, 0.5, 1.0e9, 5.0e9),
        (0.7, 2.0, 5.0e9, 12.0e9),
    ],
)
def test_double_powerlaw_not_mirror_image(alpha, beta, tau_yr, age_yr):
    """double_powerlaw is NOT the mirror image of dpl (guards against re-reversal).

    If someone accidentally re-reversed the fix, the shape would become the
    mirror image of dpl (reading dpl's time-reversed array backwards). This
    test verifies that distance to the mirror is > 0.05, catching re-reversals.
    """
    from tengri.components.stellar.sfh.mean_sfh import double_powerlaw, dpl

    t_lookback = jnp.linspace(1.0e6, age_yr, 5000)

    shape_double_powerlaw = np.asarray(
        double_powerlaw(t_lookback, alpha=alpha, beta=beta, tau=tau_yr, norm=1.0, age=age_yr)
    )
    shape_dpl = np.asarray(
        dpl(t_lookback, alpha=alpha, beta=beta, tau=tau_yr, age=age_yr, log_total_mass=0.0)
    )

    # Normalize
    dt_yr = np.mean(np.diff(t_lookback))
    int_dpow = trapezoid(shape_double_powerlaw, t_lookback)
    int_dpl = trapezoid(shape_dpl, t_lookback)

    shape_double_powerlaw_norm = (
        shape_double_powerlaw / int_dpow if int_dpow > 0 else shape_double_powerlaw
    )
    shape_dpl_norm = shape_dpl / int_dpl if int_dpl > 0 else shape_dpl

    # Assertion (2): Distance to mirror image > 0.05
    dist_mirror = 0.5 * np.sum(np.abs(shape_double_powerlaw_norm - shape_dpl_norm[::-1])) * dt_yr
    assert dist_mirror > 0.05, (
        f"Distance to mirror {dist_mirror:.6f} <= 0.05 for (α={alpha}, β={beta}, "
        f"τ={tau_yr:.1e}, age={age_yr:.1e}) — shape may be reversed!"
    )


@pytest.mark.parametrize(
    "alpha,beta,tau_yr,age_yr",
    [
        (1.5, 1.0, 3.0e9, 8.0e9),
        (3.0, 0.5, 1.0e9, 5.0e9),
    ],
)
def test_double_powerlaw_zero_beyond_age(alpha, beta, tau_yr, age_yr):
    """double_powerlaw is exactly 0 for t_lookback > age.

    The SFH cannot extend into the future (t_lookback > age).
    Assertion (3): SFR is exactly 0 for t_lookback > age and finite everywhere.
    """
    from tengri.components.stellar.sfh.mean_sfh import double_powerlaw

    # Grid including times beyond formation (t_lookback > age)
    t_beyond = jnp.array([0.5 * age_yr, age_yr, 1.5 * age_yr, 2.0 * age_yr])

    sfr = np.asarray(
        double_powerlaw(t_beyond, alpha=alpha, beta=beta, tau=tau_yr, norm=1.0, age=age_yr)
    )

    # Check finite everywhere
    assert np.all(np.isfinite(sfr)), f"Non-finite values in SFR: {sfr}"

    # Check zero beyond age
    mask_beyond = t_beyond > age_yr
    if np.any(mask_beyond):
        assert np.all(sfr[mask_beyond] == 0.0), f"SFR not zero beyond age: {sfr[mask_beyond]}"


@pytest.mark.parametrize(
    "alpha,beta,tau_yr,age_yr",
    [
        (1.5, 1.0, 3.0e9, 8.0e9),
        (3.0, 0.5, 1.0e9, 5.0e9),
    ],
)
def test_double_powerlaw_gradients_finite(alpha, beta, tau_yr, age_yr):
    """Gradients of double_powerlaw w.r.t. tau are finite.

    Assertion (4): jax.grad of the summed output wrt tau is finite.
    """
    from tengri.components.stellar.sfh.mean_sfh import double_powerlaw

    t_lookback = jnp.linspace(1.0e6, age_yr, 100)

    def objective(tau):
        return jnp.sum(
            double_powerlaw(t_lookback, alpha=alpha, beta=beta, tau=tau, norm=1.0, age=age_yr)
        )

    grad_fn = jax.grad(objective)
    grad_tau = grad_fn(tau_yr)

    assert np.isfinite(float(grad_tau)), f"Non-finite gradient w.r.t. tau: {grad_tau}"


def test_double_powerlaw_doctest():
    """The Examples block in the double_powerlaw docstring must work.

    This ensures that users following the docstring example get the
    correct behavior after the fix.
    """
    import jax.numpy as jnp

    from tengri import double_powerlaw

    # Example from the docstring (updated to include age parameter)
    t = jnp.linspace(1e6, 13.7e9, 100)
    age = 13.7e9  # Age of the universe
    sfr = double_powerlaw(t, alpha=1.5, beta=2.0, tau=3e9, norm=10.0, age=age)

    assert sfr.shape == (100,), f"Expected shape (100,), got {sfr.shape}"
    assert jnp.all(jnp.isfinite(sfr)), "Docstring example produces non-finite values"
    assert jnp.all(sfr >= 0), "Docstring example produces negative SFR"


@pytest.mark.parametrize(
    "tau_yr,age_yr",
    [
        (3.0e9, 8.0e9),
        (1.0e9, 5.0e9),
        (5.0e9, 12.0e9),
    ],
)
@pytest.mark.parametrize("n_t_lookback", [2000, 20000])
def test_delayed_tau_matches_sfhdelayed_shape(tau_yr, age_yr, n_t_lookback):
    """Fixed delayed_tau(t_lb, age, ...) matches sfhdelayed shape exactly.

    After the fix, delayed_tau should evaluate the delayed-exponential bare shape
    at cosmic time T = age - t_lookback, with no renormalization. This shape
    should be identical (up to numerical precision) to the shape computed by
    sfhdelayed when both use the same parameters.
    """
    from tengri.components.stellar.sfh.mean_sfh import delayed_tau, sfhdelayed

    t_lookback = jnp.linspace(1.0e6, age_yr, n_t_lookback)

    # Compute the bare shape using the fixed delayed_tau
    shape_delayed_tau = np.asarray(delayed_tau(t_lookback, tau=tau_yr, norm=1.0, age=age_yr))

    # Compute via sfhdelayed with no mass renormalization (log_total_mass=0)
    shape_sfhdelayed = np.asarray(
        sfhdelayed(t_lookback, tau=tau_yr, age=age_yr, log_total_mass=0.0)
    )

    # Normalize both to unit integral for shape comparison
    dt_yr = np.mean(np.diff(t_lookback))
    int_dtau = trapezoid(shape_delayed_tau, t_lookback)
    int_sfhd = trapezoid(shape_sfhdelayed, t_lookback)

    shape_delayed_tau_norm = shape_delayed_tau / int_dtau if int_dtau > 0 else shape_delayed_tau
    shape_sfhdelayed_norm = shape_sfhdelayed / int_sfhd if int_sfhd > 0 else shape_sfhdelayed

    # Assertion (1): Half-sum absolute difference is < 1e-3
    dist_direct = 0.5 * np.sum(np.abs(shape_delayed_tau_norm - shape_sfhdelayed_norm)) * dt_yr
    assert dist_direct < 1e-3, (
        f"Shape distance {dist_direct:.6f} >= 1e-3 for "
        f"(τ={tau_yr:.1e}, age={age_yr:.1e}, n={n_t_lookback})"
    )


@pytest.mark.parametrize(
    "tau_yr,age_yr",
    [
        (3.0e9, 8.0e9),
        (1.0e9, 5.0e9),
        (5.0e9, 12.0e9),
    ],
)
def test_delayed_tau_not_mirror_image(tau_yr, age_yr):
    """delayed_tau is NOT the mirror image of sfhdelayed (guards against re-reversal).

    If someone accidentally re-reversed the fix, the shape would become the
    mirror image of sfhdelayed. This test verifies distance to mirror > 0.05.
    """
    from tengri.components.stellar.sfh.mean_sfh import delayed_tau, sfhdelayed

    t_lookback = jnp.linspace(1.0e6, age_yr, 5000)

    shape_delayed_tau = np.asarray(delayed_tau(t_lookback, tau=tau_yr, norm=1.0, age=age_yr))
    shape_sfhdelayed = np.asarray(
        sfhdelayed(t_lookback, tau=tau_yr, age=age_yr, log_total_mass=0.0)
    )

    # Normalize
    dt_yr = np.mean(np.diff(t_lookback))
    int_dtau = trapezoid(shape_delayed_tau, t_lookback)
    int_sfhd = trapezoid(shape_sfhdelayed, t_lookback)

    shape_delayed_tau_norm = shape_delayed_tau / int_dtau if int_dtau > 0 else shape_delayed_tau
    shape_sfhdelayed_norm = shape_sfhdelayed / int_sfhd if int_sfhd > 0 else shape_sfhdelayed

    # Assertion (2): Distance to mirror image > 0.05
    dist_mirror = (
        0.5 * np.sum(np.abs(shape_delayed_tau_norm - shape_sfhdelayed_norm[::-1])) * dt_yr
    )
    assert dist_mirror > 0.05, (
        f"Distance to mirror {dist_mirror:.6f} <= 0.05 for "
        f"(τ={tau_yr:.1e}, age={age_yr:.1e}) — shape may be reversed!"
    )
