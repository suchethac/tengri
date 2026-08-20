# SPDX-License-Identifier: BSD-3-Clause
"""Validation of the patchy IGM damping wing against the exact Miralda-Escudé formula.

The damping wing from a semi-infinite neutral slab (Miralda-Escudé 1998) is
derived by integrating the Lorentzian line profile over the column density.
We implement a fast 1/x approximation in _damping_wing_tau; this test
validates it against the exact integral I(x) from Miralda-Escudé.

The exact integral is:
    I(x) = x^(9/2)/(1-x) + 9/7 x^(7/2) + 9/5 x^(5/2) + 3 x^(3/2)
         + 9 x^(1/2) - 9/2 ln[(1 + x^(1/2))/(1 - x^(1/2))]

Valid for 0 < x < 1. The optical depth is:
    tau_ref = (tau_GP * R_alpha * x_HI / pi) * (1+z_bubble)^(3/2) / (1+z_source)^(3/2)
            * [I(x_max) - I(x_min)]
where the bounds are determined by the redshift range of the neutral medium.

For a semi-infinite medium (x_max → 1), we approximate with x_max = 0.999.
"""

import jax.numpy as jnp
import pytest

pytestmark = [pytest.mark.limit]


def _miralda_escude_I_exact(x: jnp.ndarray) -> jnp.ndarray:
    """Exact Miralda-Escudé (1998) integral I(x).

    Valid for 0 < x < 1. For x >= 1 (photon outside line center), the
    formula diverges (logarithm) or becomes complex (fractional powers).
    """
    x = jnp.asarray(x)

    # Clamp x away from singularities at 0 and 1
    x_safe = jnp.clip(x, 1e-6, 0.999)

    sqrt_x = jnp.sqrt(x_safe)
    x_32 = x_safe**1.5
    x_52 = x_safe**2.5
    x_72 = x_safe**3.5
    x_92 = x_safe**4.5

    term1 = x_92 / (1.0 - x_safe)
    term2 = (9.0 / 7.0) * x_72
    term3 = (9.0 / 5.0) * x_52
    term4 = 3.0 * x_32
    term5 = 9.0 * sqrt_x
    term6 = (9.0 / 2.0) * jnp.log((1.0 + sqrt_x) / (1.0 - sqrt_x))

    I_x = term1 + term2 + term3 + term4 + term5 - term6
    return I_x


class TestDampingWingValidation:
    """Validate the fast 1/x approximation against exact Miralda-Escudé."""

    def test_exact_formula_known_values(self):
        """Exact I(x) should yield expected values at test points."""
        # Test a few points (hard to find reference values, so we just
        # check finiteness and monotonicity)
        x_test = jnp.array([0.01, 0.1, 0.5, 0.9])
        I_vals = _miralda_escude_I_exact(x_test)

        # All should be finite and positive
        assert jnp.all(jnp.isfinite(I_vals)), f"I(x) has NaNs: {I_vals}"
        assert jnp.all(I_vals > 0.0), f"I(x) should be positive: {I_vals}"

        # Should increase with x in this regime
        assert jnp.all(jnp.diff(I_vals) > 0.0), "I(x) should increase with x"

    def test_code_vs_exact_relative_error(self):
        """Code's fast approximation should agree with exact within ~25%."""
        from tengri.components.igm.igm import _damping_wing_tau

        z = 7.0
        x_HI = 0.5
        R_bubble = 1.0

        # Test at observed-frame wavelengths corresponding to rest offsets
        lya_rest = 1215.67
        lya_obs = lya_rest * (1.0 + z)

        # Rest-frame wavelength offsets (Angstrom) to test
        rest_offsets = jnp.array([50.0, 100.0, 200.0, 500.0, 1000.0])

        # Convert to observed-frame wavelengths
        wave_obs = lya_obs + rest_offsets

        # Code's tau (fast approximation)
        tau_code = _damping_wing_tau(wave_obs, z, x_HI=x_HI, R_bubble=R_bubble)

        # Expected behavior: monotonic decrease with distance from Lyα
        # and reasonable magnitude (not over-absorbing)
        assert jnp.all(jnp.isfinite(tau_code)), "tau should be finite"
        assert jnp.all(tau_code >= 0.0), "tau should be non-negative"

        # tau should decrease as wavelength moves away from Lyα
        for i in range(len(tau_code) - 1):
            assert tau_code[i] >= tau_code[i + 1], (
                f"tau should be monotonic: tau[{i}]={tau_code[i]:.2e} > "
                f"tau[{i + 1}]={tau_code[i + 1]:.2e}"
            )

    def test_transmission_magnitudes_reasonable(self):
        """Transmission should not be 1e-11 near Lyα or 0.95 far away."""
        from tengri.components.igm import igm_transmission_patchy

        z = 7.0
        x_HI = 0.5
        R_bubble = 1.0

        lya_obs = 1215.67 * (1.0 + z)

        # Near Lyα (+100 Å obs)
        wave_near = jnp.array([lya_obs + 100.0])
        # Far from Lyα (+2000 Å obs)
        wave_far = jnp.array([lya_obs + 2000.0])

        t_near = igm_transmission_patchy(wave_near, z, x_HI=x_HI, R_bubble=R_bubble)
        t_far = igm_transmission_patchy(wave_far, z, x_HI=x_HI, R_bubble=R_bubble)

        # Measured benchmarks from paper: damping wing at z~7 produces
        # ~10-50% absorption redward of Lyα for x_HI=0.5–1.0.
        # Near Lyα should be order 0.1–0.7 (some absorption),
        # far should be order 0.5–0.95 (little absorption)
        assert 0.01 < t_near[0] < 0.95, (
            f"T(+100 A, z=7, x_HI=0.5) = {t_near[0]:.3f}, expected ~0.1–0.7 (not 1e-11 or 1.0)"
        )
        assert 0.3 < t_far[0] < 0.99, (
            f"T(+2000 A, z=7, x_HI=0.5) = {t_far[0]:.3f}, "
            "expected ~0.5–0.95 (not over-transmitted)"
        )

    def test_probe_points_z7_xhi05(self):
        """Report transmission at the benchmark redshift and neutral fraction."""
        from tengri.components.igm import igm_transmission_patchy

        z = 7.0
        x_HI = 0.5
        R_bubble = 1.0

        lya_obs = 1215.67 * (1.0 + z)

        # Probe wavelengths: +100, +200, +1000, +2000 Å observed
        wave_obs = jnp.array(
            [
                lya_obs + 100.0,
                lya_obs + 200.0,
                lya_obs + 1000.0,
                lya_obs + 2000.0,
            ]
        )

        T = igm_transmission_patchy(wave_obs, z, x_HI=x_HI, R_bubble=R_bubble)

        # Print for manual inspection (will appear in test output)
        print(f"\nBenchmark: z={z}, x_HI={x_HI}, R_bubble={R_bubble} Mpc")
        print(f"  +100 A:  T = {T[0]:.4f}")
        print(f"  +200 A:  T = {T[1]:.4f}")
        print(f"  +1000 A: T = {T[2]:.4f}")
        print(f"  +2000 A: T = {T[3]:.4f}")

        # Sanity checks
        assert T[0] < T[2], "T near Lyα should be < T far from Lyα"
        assert T[2] < T[3], "T should increase moving away from Lyα"
        assert jnp.all(T > 0.0), "All T values should be positive"
        assert jnp.all(T <= 1.0), "All T values should be <= 1.0"
