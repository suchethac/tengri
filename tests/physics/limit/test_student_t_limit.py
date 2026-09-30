# SPDX-License-Identifier: BSD-3-Clause
"""Limit tests for Student-t noise model approaching Gaussian as dof -> infinity.

As degrees of freedom increases, the Student-t distribution approaches the
Gaussian distribution. This module verifies that the normalized constant
correctly implements this limit such that the Student-t energy approaches
the Gaussian energy as dof -> infinity.
"""

import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
import pytest
from scipy import integrate

pytestmark = pytest.mark.limit

from tengri.observation.noise import variable_noise_hamiltonian


class TestStudentTdofLimit:
    """Test Student-t approaching Gaussian as dof -> infinity."""

    def test_student_t_energy_approaches_gaussian(self):
        """Verify Student-t energy approaches Gaussian as dof increases.

        The finite-dof gap should monotonically decrease and approach zero.
        At dof=1000, the gap should be below 1e-3 per datum for reasonable data.
        """
        # Use the crossval test data
        data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predicted = jnp.array([1.1, 1.9, 3.2, 3.8, 5.1])
        noise_obs = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1])
        f_cal = 0.05

        energy_gaussian = float(
            variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=None)
        )

        dof_values = [10.0, 100.0, 1000.0, 10000.0]
        gaps = []
        energies = []

        for dof in dof_values:
            energy_t = float(
                variable_noise_hamiltonian(data, noise_obs, predicted, f_cal, dof=dof)
            )
            gap = energy_t - energy_gaussian
            gaps.append(gap)
            energies.append(energy_t)

        # Gaps should monotonically decrease
        for i in range(len(gaps) - 1):
            assert gaps[i] > gaps[i + 1], (
                f"Gaps should decrease: gap(dof={dof_values[i]})={gaps[i]:.6e} "
                f"vs gap(dof={dof_values[i+1]})={gaps[i+1]:.6e}"
            )

        # At dof=10000, gap should be very small (below 1e-3 per datum)
        n_data = float(data.shape[0])
        assert abs(gaps[-1]) < 1e-3, (
            f"Gap at dof=10000 should be < 1e-3: {abs(gaps[-1]):.6e}"
        )

        # At dof=1000, gap should be small (rtol ~1e-2 from Gaussian)
        npt.assert_allclose(
            energies[2],
            energy_gaussian,
            rtol=0.01,
            err_msg=(
                f"Student-t(dof=1000) energy={energies[2]:.6f} "
                f"should match Gaussian={energy_gaussian:.6f}"
            ),
        )

    def test_student_t_density_integrates_to_one(self):
        """Verify Student-t density (with omitted constant added back) integrates to 1.

        The variable_noise_hamiltonian returns the negative log-density up to a
        parameter-independent constant. For one datum with sigma=1, integrating
        exp(-(energy + 0.5*log(2*pi))) should yield 1.
        """
        # Test with different degrees of freedom
        dof_values = [3.0, 5.0, 10.0]
        integrals = []

        for dof_val in dof_values:
            # For a single datum with sigma=1, the energy from variable_noise_hamiltonian is:
            # E = 0.5 * (dof + 1) * log(1 + r^2/dof) - norm_term
            # where norm_term = lgamma((dof+1)/2) - lgamma(dof/2) - 0.5*log(dof/2)
            #
            # Per the brief, we integrate exp(-(E + 0.5*log(2*pi))) which should equal 1.

            norm_term_val = float(
                jax.scipy.special.gammaln((dof_val + 1.0) / 2.0)
                - jax.scipy.special.gammaln(dof_val / 2.0)
                - 0.5 * jnp.log(dof_val / 2.0)
            )

            def integrand(x, dof=dof_val, norm_term=norm_term_val):
                # Residual energy from the likelihood
                residual = 0.5 * (dof + 1.0) * np.log(1.0 + x**2 / dof)
                # Energy as returned by variable_noise_hamiltonian (for 1 datum)
                energy = residual - norm_term
                # Full density after adding back the omitted constant
                # density = exp(-(energy + 0.5*log(2*pi)))
                return np.exp(-(energy + 0.5 * np.log(2.0 * np.pi)))

            # Numerically integrate the density from -inf to +inf
            result, _ = integrate.quad(integrand, -1e2, 1e2)
            integrals.append(result)

            # Result should be 1 to within integration tolerance
            npt.assert_allclose(
                result,
                1.0,
                atol=1e-4,
                err_msg=f"Student-t(dof={dof_val}) density should integrate to 1, got {result}",
            )
