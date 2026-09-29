# SPDX-License-Identifier: BSD-3-Clause
"""Verify profile mass marginalization normalization by comparing with brute-force
quadrature.

This script evaluates both the profiled log-likelihood (analytic marginalization)
and a brute-force numerical integral over the mass prior at 3-4 different theta
(other parameters) points. A constant offset across theta points indicates a
normalization error in the analytic marginal, which is invisible to NUTS tests
and PIT calibration because both are shift-invariant in log L. A varying offset
indicates an approximation error in the quadratic amplitude. A near-zero offset
everywhere indicates the marginal is correct.

**Convention matching (critical for validity):**

1. **Theta-prior removal:** The profiled loss function from `build_profiled_loss_fn`
   returns a LOSS (negative log posterior), adding `standardized_neg_log_prior(...)`
   over the OTHER parameters. The brute-force side integrates only over M at fixed
   theta, with no theta-prior. To compare like with like, we remove the theta-prior
   contribution from the profiled side before comparing.

2. **Coordinate space:** The profiled loss works in standardized (unconstrained,
   unit-normal) coordinate space. We convert theta to physical space, compute the
   profiled stats, then convert back to standardized space for the loss evaluation.

3. **Likelihood:** Both sides use the same likelihood as the fitter evaluates it,
   including any per-band error model and normalization. The `_profile_stats`
   function replicates this exactly.

4. **Jacobian:** The mass prior is declared in log10(M) space. The quadrature
   integrates over this same space with the correct density from `mass_prior.log_prob`.

**Planted control:** A deliberately multiplied brute-force integral by e^2.0
should report a 2.0-nat difference, validating the harness can detect offsets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tengri import (
    ForwardModel,
    SEDModel,
    generate_mock,
    recipes,
)
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter
from tengri.inference.mass_profile import _log_mass_integral, _profile_stats
from tengri.observation import Observation, Photometry


def build_minimal_model(ssp_data):
    """Build a minimal model with free log_total_mass and a couple of other
    parameters.

    Uses the mock_recovery_minimal recipe with a few free dust parameters.
    Redshift is fixed.
    """
    obs = Observation(
        photometry=Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r"]
        )
    )
    return SEDModel.build(
        ssp_data=ssp_data,
        observation=obs,
        **recipes.mock_recovery_minimal(),
    )


def generate_synthetic_data(model, seed=42, snr=50.0):
    """Generate synthetic photometry data from a parameter draw."""
    key_truth, key_mock = jax.random.split(jax.random.PRNGKey(seed))
    truth = model.spec.sample(key_truth)
    mock = generate_mock(model, truth, key=key_mock, snr=snr)
    return truth, jnp.asarray(mock["flux_obs"]), jnp.asarray(mock["noise"])


def evaluate_profiled_marginal(fitter, phys_theta, flux, noise):
    """Evaluate the profiled log-likelihood at a physical parameter point.

    Returns the log of the marginal integral over mass, WITHOUT the theta-prior term.
    This is what should match the brute-force integral.

    Parameters
    ----------
    fitter : Fitter
        The unprofiled fitter (profile_mass=False).
    phys_theta : dict
        Physical parameter values for the non-mass parameters (e.g., dust_tau_bc).
    flux : ndarray
        Observed photometry data vector.
    noise : ndarray
        Per-band noise (1-sigma).

    Returns
    -------
    profiled_loglik : float
        log p(d | theta) with M marginalized out, no theta-prior term.
    """
    model = fitter.model
    mass_name = next(n for n in fitter.spec.free_params if n.endswith("log_total_mass"))
    mass_prior = fitter.spec.get_distribution(mass_name)
    ell_lo, ell_hi = mass_prior.bounds

    # Assemble the full physical params dict, keeping mass at a placeholder to
    # compute the quadratic terms (which are mass-factor-invariant).
    phys_full = {**phys_theta, mass_name: jnp.asarray(0.0)}

    # Compute the quadratic terms from _profile_stats.
    A_ref, a_star, chi2_min, ell_ref = _profile_stats(
        model,
        mass_name,
        phys_full,
        flux,
        noise,
        data_type="photometry",
    )

    # The profiled marginal: log of the quadrature integral over M.
    loglik = -0.5 * chi2_min + _log_mass_integral(
        A_ref, a_star, ell_lo, ell_hi, mass_prior, ell_ref
    )

    return float(loglik)


def evaluate_brute_force_marginal(
    fitter, phys_theta, flux, noise, n_grid=2000, planted_factor=0.0
):
    """Evaluate the mass marginal by brute-force quadrature.

    Integrates log p(d | theta, M) * p(M) over M using a fine grid in log10(M).

    Parameters
    ----------
    fitter : Fitter
        The unprofiled fitter (profile_mass=False).
    phys_theta : dict
        Physical parameter values for the non-mass parameters.
    flux : ndarray
        Observed photometry data vector.
    noise : ndarray
        Per-band noise (1-sigma).
    n_grid : int
        Number of quadrature nodes over the mass prior.
    planted_factor : float
        Add this to the log-integral, i.e. multiply the integral by
        exp(planted_factor). Used ONLY by the planted control. It MUST
        default to 0.0: a nonzero default silently inflates every ordinary
        call and is then reported as a constant offset in the thing under
        test. That is exactly what a default of 1.0 did here -- it produced
        a confident '-1.0 nat normalization error' that was the harness
        measuring its own plant.

    Returns
    -------
    loglik_marginal : float
        log p(d | theta) = log integral_M [ p(d | theta, M) p(M) dM ], plus
        the planted factor if applied.
    """
    model = fitter.model
    mass_name = next(n for n in fitter.spec.free_params if n.endswith("log_total_mass"))
    mass_prior = fitter.spec.get_distribution(mass_name)
    ell_lo, ell_hi = mass_prior.bounds

    # Build standardized theta from phys_theta, then construct full xi (including mass).
    free_names = [n for n in fitter._free_names if n != mass_name]
    xi_theta = {}
    for name in free_names:
        phys_val = phys_theta[name]
        xi_theta[name] = fitter.spec.get_distribution(name).standardize(jnp.asarray(phys_val))

    # Grid in log10(M) over the prior support.
    ell_grid = np.linspace(ell_lo, ell_hi, n_grid)
    ell_prior_density = np.asarray([float(mass_prior.log_prob(jnp.asarray(e))) for e in ell_grid])
    p_ell = np.exp(ell_prior_density)

    # Evaluate the log-likelihood at each mass value, holding theta fixed.
    # We extract it from the loss function via: log p(d | theta, M) = -nlp - log_prior
    ctx = InferenceContext.from_target(fitter)
    nlp_fn = ctx.neg_log_posterior_fn
    log_prior_fn = ctx.log_prior_fn
    data_args = ctx.data_args

    loglik_vals = np.empty(ell_grid.shape)
    for k, ell in enumerate(ell_grid):
        # Standardize the mass parameter.
        xi_mass = mass_prior.standardize(jnp.asarray(ell))
        # Full standardized parameter vector.
        xi_full = {**xi_theta, mass_name: xi_mass}
        # Evaluate loss = -log p(d | theta, M) - log p(theta, M)
        loss = nlp_fn(xi_full, data_args)
        # Evaluate the prior term log p(theta, M)
        lp = log_prior_fn(xi_full)
        # Extract log-likelihood: log p(d | theta, M) = -loss - lp
        loglik_vals[k] = float(-loss - lp)

    # Integrate: log integral [ exp(loglik) * p(ell) dell ].
    loglik_max = float(np.max(loglik_vals))
    integrand = np.exp(loglik_vals - loglik_max) * p_ell
    log_integral = float(np.log(np.trapezoid(integrand, ell_grid))) + loglik_max

    # Apply planted factor if requested.
    if abs(planted_factor) > 1e-10:
        log_integral += planted_factor

    return log_integral


def verify_normalization_at_theta_points(ssp_data, n_thetas=4):
    """Main verification: evaluate both sides at multiple theta points.

    Returns a table of results and diagnostics.
    """
    # Build models and data.
    model = build_minimal_model(ssp_data)
    forward = ForwardModel.build(sed=model)
    _truth, flux, noise = generate_synthetic_data(model, seed=0, snr=50.0)

    # Create unprofiled fitter for context and loss evaluation.
    fitter = Fitter(forward, data=flux, noise=noise, profile_mass=False)

    mass_name = next(n for n in fitter.spec.free_params if n.endswith("log_total_mass"))
    free_names = [n for n in fitter._free_names if n != mass_name]

    results = []

    # Generate n_thetas different parameter points in standardized space.
    for theta_idx in range(n_thetas):
        key = jax.random.PRNGKey(theta_idx + 100)
        # Draw standardized values for the non-mass parameters.
        xi_theta_std = {
            name: float(jax.random.normal(jax.random.fold_in(key, j)))
            for j, name in enumerate(free_names)
        }

        # Convert to physical space.
        phys_theta = {
            name: float(
                fitter.spec.get_distribution(name).unstandardize(jnp.asarray(xi_theta_std[name]))
            )
            for name in free_names
        }

        # Evaluate both sides.
        profiled_val = evaluate_profiled_marginal(fitter, phys_theta, flux, noise)
        brute_force_val = evaluate_brute_force_marginal(fitter, phys_theta, flux, noise)

        diff = profiled_val - brute_force_val

        results.append(
            {
                "theta_idx": theta_idx,
                "profiled": profiled_val,
                "brute_force": brute_force_val,
                "difference": diff,
                "rel_error": diff / abs(brute_force_val)
                if abs(brute_force_val) > 1e-10
                else np.nan,
            }
        )

    return fitter, results


def test_planted_control(ssp_data):
    """Test control: multiply brute-force integral by known factor.

    Should report a difference of exactly that factor.
    """
    model = build_minimal_model(ssp_data)
    forward = ForwardModel.build(sed=model)
    _truth, flux, noise = generate_synthetic_data(model, seed=1, snr=50.0)

    fitter = Fitter(forward, data=flux, noise=noise, profile_mass=False)

    mass_name = next(n for n in fitter.spec.free_params if n.endswith("log_total_mass"))
    free_names = [n for n in fitter._free_names if n != mass_name]

    # Evaluate at one theta point.
    key = jax.random.PRNGKey(42)
    xi_theta_std = {
        name: float(jax.random.normal(jax.random.fold_in(key, j)))
        for j, name in enumerate(free_names)
    }
    phys_theta = {
        name: float(
            fitter.spec.get_distribution(name).unstandardize(jnp.asarray(xi_theta_std[name]))
        )
        for name in free_names
    }

    # Evaluate profiled (unmodified).
    profiled_val = evaluate_profiled_marginal(fitter, phys_theta, flux, noise)

    # Evaluate brute-force with a planted factor of 2.0 nats.
    planted_nats = 2.0
    brute_force_planted = evaluate_brute_force_marginal(
        fitter, phys_theta, flux, noise, planted_factor=planted_nats
    )
    brute_force_unplanted = evaluate_brute_force_marginal(
        fitter, phys_theta, flux, noise, planted_factor=0.0
    )

    # The difference should be exactly planted_nats.
    diff_with_plant = profiled_val - brute_force_planted
    diff_without_plant = profiled_val - brute_force_unplanted

    control_result = {
        "planted_nats": planted_nats,
        "diff_with_plant": diff_with_plant,
        "diff_without_plant": diff_without_plant,
        "expected_diff_with_plant": -planted_nats,
        "passes": abs(diff_with_plant - (-planted_nats)) < 1e-10,
    }

    return control_result


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Repo root is two levels above analysis/paper1/, so parents[2] is the checkout.
    tengri_src = Path(__file__).resolve().parents[2] / "src"
    sys.path.insert(0, str(tengri_src))

    # Load SSP data (using fsps_mist_c3k as specified in the task).
    from tengri import load_ssp

    ssp = load_ssp("fsps_mist_c3k_a_chabrier")

    print("=" * 99)
    print("PROFILE MASS NORMALIZATION VERIFICATION")
    print("=" * 99)

    # Main test.
    print("\n1. Evaluating profiled vs. brute-force at multiple theta points:")
    print("-" * 99)

    fitter, results = verify_normalization_at_theta_points(ssp, n_thetas=4)

    # Print table.
    print(
        f"{'Theta':<8} {'Profiled':<18} {'Brute-Force':<18} {'Difference':<18} {'Rel Error':<12}"
    )
    print("-" * 99)
    for res in results:
        print(
            f"{res['theta_idx']:<8} "
            f"{res['profiled']:>17.10f} "
            f"{res['brute_force']:>17.10f} "
            f"{res['difference']:>17.10f} "
            f"{res['rel_error']:>11.3e}"
        )

    # Analyze results.
    diffs = np.array([r["difference"] for r in results])
    print("-" * 99)
    print(f"Mean difference: {np.mean(diffs):.10f} nats")
    print(f"Std of differences: {np.std(diffs):.10f} nats")
    print(f"Max absolute difference: {np.max(np.abs(diffs)):.10f} nats")

    # Verdict.
    max_diff = np.max(np.abs(diffs))
    if max_diff < 1e-6:
        print("\nVERDICT: Differences are ~zero. Marginal is correct.")
    elif np.std(diffs) < 0.1 * np.abs(np.mean(diffs)):
        # Constant-like offset
        print(
            f"\nVERDICT: Differences are approximately CONSTANT (~{np.mean(diffs):.6f} nats). "
            "This indicates a normalization error, invisible to shift-invariant tests (NUTS, PIT)."
        )
    else:
        print(
            f"\nVERDICT: Differences VARY with theta (~{np.std(diffs):.6f} nats std). "
            "This indicates an approximation error in the quadratic amplitude."
        )

    # Planted control.
    print("\n2. Planted control test (multiply brute-force by e^2.0):")
    print("-" * 99)
    control = test_planted_control(ssp)
    print(f"Expected difference (with plant): {control['expected_diff_with_plant']:.10f} nats")
    print(f"Actual difference (with plant):   {control['diff_with_plant']:.10f} nats")
    print(f"Actual difference (no plant):     {control['diff_without_plant']:.10f} nats")
    if control["passes"]:
        print("Control PASSES: Harness can detect a planted offset.")
    else:
        print(
            f"Control FAILS: Error is {abs(control['diff_with_plant'] - control['expected_diff_with_plant']):.3e}"
        )

    # Convention matching explanation.
    print("\n3. Convention matching summary:")
    print("-" * 99)
    print("Theta-prior handling: REMOVED from profiled side before comparison.")
    print("Coordinate space: Physical parameters evaluated at each theta;")
    print("                  _profile_stats computes quadratic terms.")
    print("Likelihood: Same as fitter uses (via _profile_stats replication).")
    print("Jacobian: Mass prior integrated in log10(M) space with correct density.")

    print("\n" + "=" * 99)
