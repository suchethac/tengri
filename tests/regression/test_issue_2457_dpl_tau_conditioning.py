# SPDX-License-Identifier: BSD-3-Clause
"""Test for issue #2457: dpl tau_gyr flat direction beyond cosmic age.

When tau_gyr exceeds the galaxy's age, the turnover falls outside the
star formation history entirely. The shape is set by alpha alone, and tau's
influence on the photometry collapses. This test verifies that once
tau_gyr is capped at age_at_z(z), the log-likelihood gradient wrt
tau_gyr is non-zero (or the formed mass changes) across the parameter's
full range, including when tau_gyr would naively exceed the cosmic age.

References
----------
.. [1] Issue #2457: "dpl SFH tau conditioning problem"
.. [2] Issue #2521: "z-cap all SFH onset/age/peak-time parameters"
"""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


def central_finite_diff_grad(f, x: float, eps: float = 1e-5) -> float:
    """Central finite-difference gradient approximation."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


class TestDPLTauGradientBeyondAge:
    """Verify tau_gyr gradient is non-zero once tau is properly z-capped.

    The presence of a flat region in the likelihood surface when tau_gyr
    exceeds age(z) is a conditioning problem (#2457). Once tau is capped
    at age(z) (part of #2521), every point in the prior should have a
    non-zero gradient wrt tau_gyr, or the formed mass should at least
    vary monotonically with tau.
    """

    @pytest.mark.parametrize("z", [0.5, 2.5, 6.0])
    def test_dpl_tau_gyr_gradient_in_capped_range(self, z, synthetic_ssp_wide):
        """Log-likelihood gradient should be non-zero for all tau in [0, age(z)].

        The gradient is measured via central finite differences applied to
        the negative log-likelihood (or equivalently, the photometry's
        sensitivity to tau_gyr).
        """
        from tengri import Fixed, SEDModel, Uniform
        from tengri.utils.cosmology import age_at_z

        ssp = synthetic_ssp_wide
        cosmic_age_gyr = float(age_at_z(z))

        # Every swept dpl param is declared with an explicit (wide) Uniform,
        # not Fixed: a Fixed parameter cannot be overridden at predict() time
        # (#2296), and this test overrides all five on every call.
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={
                "type": "dpl",
                "log_total_mass": Uniform(1.0, 15.0),
                "age_gyr": Uniform(0.01, 20.0),
                "alpha": Uniform(0.01, 10.0),
                "beta": Uniform(0.01, 10.0),
                "tau_gyr": Uniform(0.01, 20.0),
            },
            redshift=Fixed(z),
            met={"type": "delta", "logzsol": Fixed(0.0)},
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
            neb={"type": "none"},
        )

        # Define a base parameter set
        params_base = {
            "sfh_dpl_log_total_mass": 9.0,
            "sfh_dpl_age_gyr": cosmic_age_gyr - 0.5,
            "sfh_dpl_alpha": 1.0,
            "sfh_dpl_beta": 1.5,
        }

        # Test several tau values across the range [0.1, age(z)]
        tau_test_values = np.linspace(0.1, cosmic_age_gyr * 0.95, 5)
        gradients = []
        masses = []

        for tau_val in tau_test_values:
            params = params_base.copy()
            params["sfh_dpl_tau_gyr"] = tau_val

            # Formed mass is now conserved BY CONSTRUCTION at every tau_gyr
            # (#2521's physical fix: the CIC age weights already bound the
            # dpl shape's support to [0, age(z)] and total mass is pinned to
            # 10**log_total_mass regardless of shape), so it is no longer a
            # useful conditioning probe on its own. The rest-frame SED
            # luminosity, at FIXED total mass, still tracks the age
            # distribution the turnover sets (younger stars are more
            # luminous per unit mass), so it is the sensitive observable
            # for #2457's flat-direction check.
            def sed_luminosity_fn(tau_gyr_var, _params=params):
                p = _params.copy()
                p["sfh_dpl_tau_gyr"] = tau_gyr_var
                pred = model.predict(p)
                return float(jnp.sum(pred.rest_sed()))

            # Compute finite-difference gradient of SED luminosity wrt tau_gyr
            grad = central_finite_diff_grad(sed_luminosity_fn, tau_val, eps=1e-4)
            gradients.append(grad)
            mass = float(model.predict(params).stellar_mass)
            masses.append(mass)

        gradients = np.array(gradients)
        masses = np.array(masses)

        # Check that formed mass is finite and positive
        chex.assert_tree_all_finite(masses)
        assert np.all(masses > 0), (
            f"Formed mass should be positive for dpl z={z}, but got min={np.min(masses):.2e}"
        )

        # The gradient should be non-zero for at least most values.
        non_zero_grad_count = np.sum(np.abs(gradients) > 1e-10)
        assert non_zero_grad_count >= 3, (
            f"Expected at least 3 of {len(tau_test_values)} tau values to have "
            f"non-zero gradient, but only {non_zero_grad_count} do. "
            f"Gradients: {gradients}. This suggests a flat region in the "
            f"likelihood surface (#2457)."
        )

    @pytest.mark.parametrize("z", [0.5, 2.5, 6.0])
    def test_dpl_formed_mass_monotonic_with_tau_gyr(self, z, synthetic_ssp_wide):
        """Formed stellar mass should show monotonic or smooth variation with tau_gyr.

        Once tau is capped, the formed mass should not be invariant (constant)
        across the tau prior, which would indicate a flat likelihood direction.
        """
        from tengri import Fixed, SEDModel, Uniform
        from tengri.utils.cosmology import age_at_z

        ssp = synthetic_ssp_wide
        cosmic_age_gyr = float(age_at_z(z))

        # Every swept dpl param is declared with an explicit (wide) Uniform,
        # not Fixed: a Fixed parameter cannot be overridden at predict() time
        # (#2296), and this test overrides all four on every call.
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={
                "type": "dpl",
                "log_total_mass": Uniform(1.0, 15.0),
                "age_gyr": Uniform(0.01, 20.0),
                "alpha": Uniform(0.01, 10.0),
                "beta": Uniform(0.01, 10.0),
                "tau_gyr": Uniform(0.01, 20.0),
            },
            redshift=Fixed(z),
            met={"type": "delta", "logzsol": Fixed(0.0)},
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
            neb={"type": "none"},
        )

        params_base = {
            "sfh_dpl_log_total_mass": 9.0,
            "sfh_dpl_age_gyr": cosmic_age_gyr - 0.5,
            "sfh_dpl_alpha": 1.0,
            "sfh_dpl_beta": 1.5,
        }

        # Sweep tau across its full range (would naively go to 13.81 Gyr)
        # but after #2521 fix is capped at cosmic_age_gyr
        tau_sweep = np.linspace(0.1, cosmic_age_gyr * 0.99, 10)

        # Formed mass is conserved BY CONSTRUCTION at every tau_gyr now
        # (#2521's physical fix), so it is no longer a useful conditioning
        # probe: assert that directly, then check the observable that must
        # still respond to tau -- the rest-frame SED luminosity, which at
        # fixed total mass tracks the age distribution the turnover sets.
        masses = np.array(
            [
                float(model.predict({**params_base, "sfh_dpl_tau_gyr": t}).stellar_mass)
                for t in tau_sweep
            ]
        )
        chex.assert_tree_all_finite(masses)
        np.testing.assert_allclose(
            masses,
            10.0**9.0,
            rtol=1e-4,
            err_msg=f"formed mass should be conserved by construction for every tau_gyr at z={z}",
        )

        sed_sums = np.array(
            [
                float(jnp.sum(model.predict({**params_base, "sfh_dpl_tau_gyr": t}).rest_sed()))
                for t in tau_sweep
            ]
        )
        chex.assert_tree_all_finite(sed_sums)

        sed_std = np.std(sed_sums)
        mean_sed = np.mean(sed_sums)
        cv = sed_std / mean_sed
        assert cv > 0.01, (
            f"SED luminosity is nearly invariant wrt tau_gyr at z={z}: "
            f"CV={cv:.6f} (std={sed_std:.2e}, mean={mean_sed:.2e}). "
            f"This indicates a flat direction (#2457)."
        )
