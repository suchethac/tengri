# SPDX-License-Identifier: BSD-3-Clause
"""#2525: Student-t noise Hamiltonian missing dof-dependent normalization.

Tests that variable_noise_hamiltonian correctly includes the
lgamma((dof+1)/2) - lgamma(dof/2) - 0.5*log(dof/2) normalization term
for Student-t likelihoods when dof varies (free parameter case).
The Student-t energy is offset from the normalized scipy convention by
n*0.5*log(2*pi) (where n is the number of data points), making the
Student-t limit match the Gaussian as dof -> inf.
"""

import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest

pytestmark = pytest.mark.contract


class TestStudentTNormalization:
    """Student-t noise Hamiltonian normalization tests."""

    @pytest.mark.parametrize("dof", [2.0, 4.0, 8.0, 30.0])
    @pytest.mark.parametrize("sigma_scale", [0.3, 1.0, 3.0])
    @pytest.mark.parametrize("f_cal", [0.0, 0.05])
    @pytest.mark.parametrize("dof_form", ["float", "array"])
    def test_hamiltonian_matches_logpdf(self, dof, sigma_scale, f_cal, dof_form):
        """Test that -H equals sum of t.logpdf - log(sigma_eff)."""
        from tengri.observation.noise import compute_effective_noise, variable_noise_hamiltonian

        # Synthetic 50-point residual vector
        key = jax.random.PRNGKey(42)
        r_scaled = jax.random.normal(key, (50,))
        sigma = jnp.array(sigma_scale)
        r = r_scaled * sigma  # residuals scaled by sigma

        # Construct predicted and data
        # We'll use: residual r = data - predicted, so data = predicted + r
        # For simplicity, set predicted = 0, data = r
        data = r
        predicted = jnp.zeros_like(r)
        noise_obs = jnp.ones_like(r) * sigma

        # Compute effective noise
        sigma_eff = compute_effective_noise(noise_obs, predicted, f_cal)

        # Prepare dof in requested form
        if dof_form == "float":
            dof_val = dof
        else:
            dof_val = jnp.array(dof)

        # Compute routed Hamiltonian (negative log-probability)
        H_routed = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=dof_val)

        # Compute proper log-probability using jax.scipy.stats.t.logpdf
        # t.logpdf(x, df, loc, scale) returns log p(x | df, loc, scale)
        # We want log p((data - predicted)/sigma_eff | df) - log(sigma_eff)
        scaled_residuals = (data - predicted) / sigma_eff
        log_prob_t = jax.scipy.stats.t.logpdf(scaled_residuals, dof_val)
        log_sigma_eff = jnp.log(sigma_eff)

        # Sum over all 50 points
        # The variable_noise_hamiltonian uses a modified Student-t convention:
        # H = -sum(t.logpdf - log(sigma_eff) - 0.5*log(2*pi))
        # This offset by n*0.5*log(2*pi) ensures the Student-t limit matches Gaussian.
        n_data = len(data)
        total_log_prob = jnp.sum(log_prob_t - log_sigma_eff) + n_data * 0.5 * jnp.log(2.0 * jnp.pi)
        H_proper = -total_log_prob

        # Assert with tight tolerance
        npt.assert_allclose(
            float(H_routed),
            float(H_proper),
            rtol=1e-10,
            err_msg=f"Mismatch for dof={dof}, sigma_scale={sigma_scale}, "
            f"f_cal={f_cal}, dof_form={dof_form}",
        )

    @pytest.mark.parametrize("dof", [4.0])
    def test_single_datum_density_integrates_to_one(self, dof):
        """Test that the full Student-t density integrates to 1.

        After adding back the omitted parameter-independent constant.
        """
        from tengri.observation.noise import variable_noise_hamiltonian

        # Single residual at 0
        data = jnp.array([0.0])
        noise_obs = jnp.array([1.0])
        predicted = jnp.array([0.0])
        f_cal = 0.0

        # The variable_noise_hamiltonian returns energy up to a constant.
        # The full log-density is: -H - 0.5*log(2*pi) (for one datum)
        # So exp(-H - 0.5*log(2*pi)) should integrate to 1.

        sigma = 1.0  # noise_obs value
        center = 0.0
        margin = 200.0 * sigma
        y_vals = jnp.linspace(center - margin, center + margin, 10000)

        # Compute energy at each y
        energies = jnp.array(
            [
                variable_noise_hamiltonian(jnp.array([y]), noise_obs, predicted, f_cal, dof=dof)
                for y in y_vals
            ]
        )

        # Full probability density with the omitted constant added back
        # density = exp(-(energy + 0.5*log(2*pi)))
        densities = jnp.exp(-(energies + 0.5 * jnp.log(2.0 * jnp.pi)))

        # Integrate via trapezoidal rule
        dy = (2.0 * margin) / (len(y_vals) - 1)
        integral = jnp.trapezoid(densities, dx=dy)

        # Should be 1.0 +/- tolerance (the marginal density over y)
        npt.assert_allclose(
            float(integral),
            1.0,
            atol=1e-3,
            err_msg=f"Per-datum density integral is not 1 for dof={dof}",
        )

    @pytest.mark.parametrize("dof", [2.0, 4.0, 8.0, 30.0])
    def test_proper_vs_routed_difference_is_zero(self, dof):
        """Test that (proper - routed) is 0, not a constant != 0."""
        from tengri.observation.noise import variable_noise_hamiltonian

        # Generate a few synthetic residuals
        key = jax.random.PRNGKey(123)
        for i in range(3):
            key, subkey = jax.random.split(key)
            r_scaled = jax.random.normal(subkey, (10,))

            data = r_scaled
            noise_obs = jnp.ones_like(r_scaled)
            predicted = jnp.zeros_like(r_scaled)
            f_cal = 0.0

            # Hamiltonian (energy, negative log-prob)
            H = variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=dof)

            # Proper via logpdf with the offset constant
            sigma_eff = jnp.ones_like(r_scaled)  # since noise_obs=1, predicted=0, f_cal=0
            scaled_res = data / sigma_eff
            log_prob_t = jax.scipy.stats.t.logpdf(scaled_res, dof)
            n_data = len(data)
            # H = -sum(t.logpdf - log(sigma_eff) - 0.5*log(2*pi))
            const_term = n_data * 0.5 * jnp.log(2.0 * jnp.pi)
            H_proper = -(jnp.sum(log_prob_t - jnp.log(sigma_eff)) + const_term)

            difference = H_proper - H
            npt.assert_allclose(
                float(difference),
                0.0,
                atol=1e-10,
                err_msg=f"Non-zero difference at dof={dof}, iteration {i}",
            )
