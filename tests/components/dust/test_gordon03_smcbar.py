# SPDX-License-Identifier: BSD-3-Clause
"""Tests for Gordon et al. (2003) SMC Bar attenuation law.

Gordon et al. (2003), ApJ 594, 279, Table 4: empirical SMC-bar extinction curve.
Repackaged from dust_extinction.averages.G03_SMCBar; tabulated k(λ) values
interpolated with jnp.interp, normalized to k(5500 Å) = 1.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.bounds


class TestGordon03SMCBar:
    """Tests for the Gordon et al. (2003) SMC-bar tabulated extinction curve."""

    # Gordon+2003 Table 4 nodes: wavenumber [µm⁻¹] → A(λ)/A(V)
    # Source: dust_extinction.averages.G03_SMCBar().obsdata_x/.obsdata_axav
    TABLE4_WAVENUMBER_UM = np.array(
        [0.455, 0.606, 0.800, 1.235, 1.538, 1.818, 2.273, 2.703, 3.375, 8.625]
    )
    TABLE4_AXAV = np.array([0.110, 0.169, 0.250, 0.567, 0.801, 1.000, 1.374, 1.672, 2.000, 6.992])

    def test_table4_nodes(self):
        """k(λ)/k(5500) at Table 4 nodes matches dust_extinction model to rtol 5e-3.

        Gordon et al. (2003) Table 4 "SMC Bar" average:
        x [µm⁻¹] = 0.455, 0.606, ..., 8.625
        (R_V = 2.74)

        The dust_extinction.averages.G03_SMCBar model tabulates and extends
        these data via FM90 formula at high x (>= 3.3 µm⁻¹). This test verifies
        that our fine-grid evaluation of the model matches the model's own output.
        """
        pytest.importorskip("dust_extinction")
        import astropy.units as u
        from dust_extinction.averages import G03_SMCBar

        from tengri.components.dust.attenuation import gordon03_smcbar

        # Get reference values by evaluating the model directly
        m = G03_SMCBar()
        m_results = np.asarray(m(self.TABLE4_WAVENUMBER_UM / u.micron), dtype=np.float64)
        # Normalize to k(V-band) = 1
        # V-band is at x = 1.818 µm⁻¹ (index 5)
        k_v_ref = m_results[5]
        expected = m_results / k_v_ref

        # Convert wavenumbers [µm⁻¹] to wavelengths [Å]
        table4_wave_aa = 1e4 / self.TABLE4_WAVENUMBER_UM
        # Evaluate gordon03_smcbar at Table 4 wavelengths
        k_at_table4 = np.asarray(gordon03_smcbar(jnp.asarray(table4_wave_aa)))

        assert_allclose(k_at_table4, expected, rtol=5e-3, err_msg="Table 4 node mismatch")

    def test_v_normalization(self):
        """k(5500 Å) == 1 to 1e-6."""
        pytest.importorskip("dust_extinction")

        from tengri.components.dust.attenuation import gordon03_smcbar

        wave_v = jnp.array([5500.0])
        k_v = gordon03_smcbar(wave_v)
        assert_allclose(float(k_v[0]), 1.0, atol=1e-6, err_msg="V-band normalization")

    def test_monotonicity_optical_ir(self):
        """At optical/NIR (λ ≥ 5500 Å), k(λ) is monotonically decreasing."""
        pytest.importorskip("dust_extinction")
        from tengri.components.dust.attenuation import gordon03_smcbar

        # Sample wavelengths from V-band to far-IR
        wave = jnp.logspace(3.74, 4.34, 20)  # 5500 Å to ~20000 Å
        k = gordon03_smcbar(wave)
        # Check monotonicity
        dk = np.diff(np.asarray(k))
        assert np.all(dk <= 0), "k(λ) should be decreasing at optical/NIR wavelengths"

    def test_is_jittable(self):
        """gordon03_smcbar is JIT-compatible."""
        pytest.importorskip("dust_extinction")
        from tengri.components.dust.attenuation import gordon03_smcbar

        wave = jnp.array([3000.0, 5500.0, 10000.0])

        # Test JIT compilation
        jitted_fn = jax.jit(gordon03_smcbar)
        k_eager = gordon03_smcbar(wave)
        k_jit = jitted_fn(wave)

        assert_allclose(k_eager, k_jit, rtol=1e-10, err_msg="JIT parity mismatch")

    def test_finite_values_on_grid(self):
        """gordon03_smcbar returns finite values on a wavelength grid."""
        pytest.importorskip("dust_extinction")
        from tengri.components.dust.attenuation import gordon03_smcbar

        wave = jnp.logspace(2.5, 4.5, 100)  # 315 Å to 31623 Å
        k = gordon03_smcbar(wave)

        assert jnp.all(jnp.isfinite(k)), "All k values should be finite"
        assert_non_negative(k, name="k")
        assert jnp.all(k < 15.0), "k(λ) should not be unreasonably large"

    def test_fm90_uv(self):
        """k at UV wavenumbers matches FM90 formula to rtol 5e-3.

        Gordon et al. (2003) Table 4 SMC-bar FM90 parameters (for x >= 3.3 µm⁻¹):
        c1=-4.959, c2=2.264, c3=0.389, c4=0.461, x0=4.6, γ=1.0, R_V=2.74
        FM90 form: k(x) = c1 + c2*x + c3*D(x; x0, γ) + c4*F(x)
        where D = x²/((x²−x0²)² + x²γ²) (Drude profile)
              F(x) = 0.5392(x−5.9)² + 0.05644(x−5.9)³ for x >= 5.9, else 0
        and A(λ)/A(V) = (k(x) + R_V) / R_V
        """
        pytest.importorskip("dust_extinction")
        from tengri.components.dust.attenuation import gordon03_smcbar

        # FM90 parameters for SMC-bar
        c1, c2, c3, c4 = -4.959, 2.264, 0.389, 0.461
        x0, gamma = 4.6, 1.0
        r_v = 2.74

        # Test wavenumbers [µm⁻¹]
        x_um = np.array([4.0, 5.0, 6.5, 8.0])
        wave_aa = 1e4 / x_um

        # Compute FM90 formula
        # Drude profile: D = x²/((x²−x0²)² + x²γ²)
        drude = x_um**2 / ((x_um**2 - x0**2) ** 2 + (x_um * gamma) ** 2)
        # Far-UV term: F(x) = 0.5392(x−5.9)² + 0.05644(x−5.9)³ for x >= 5.9, else 0
        f_uv = np.where(x_um >= 5.9, 0.5392 * (x_um - 5.9) ** 2 + 0.05644 * (x_um - 5.9) ** 3, 0.0)
        # k'(x) = c1 + c2*x + c3*D + c4*F
        k_prime = c1 + c2 * x_um + c3 * drude + c4 * f_uv
        # A(λ)/A(V) = (k' + R_V) / R_V (normalized form)
        # But our gordon03_smcbar returns k(λ) = A(λ)/A(V) normalized to k(5500) = 1
        # At V-band (5500 Å = 1.818 µm⁻¹), the model value is 1.0
        axav_fm90 = (k_prime + r_v) / r_v

        # Evaluate gordon03_smcbar
        k_gordon = np.asarray(gordon03_smcbar(jnp.asarray(wave_aa)))

        # Print the numbers for verification
        print("\nFM90 UV test: wavenumbers and k values")
        for i, x in enumerate(x_um):
            print(
                f"  x={x:.1f} µm⁻¹ ({wave_aa[i]:.0f} Å): "
                f"k_gordon={k_gordon[i]:.6f}, A/Av_fm90={axav_fm90[i]:.6f}, "
                f"relative_diff={(k_gordon[i] / axav_fm90[i] - 1.0) * 100:.3f}%"
            )

        # Compare FM90 formula to gordon03_smcbar
        assert_allclose(
            k_gordon, axav_fm90, rtol=1e-5, err_msg="FM90 formula mismatch at UV wavenumbers"
        )

    def test_reachable_on_both_screens(self):
        """gordon03_smcbar is registered and resolvable in dust attenuation system."""
        pytest.importorskip("dust_extinction")
        from tengri.components.dust.laws._registry import resolve_dust_law

        # Test that gordon03_smcbar is registered and can be resolved
        law_func = resolve_dust_law("gordon03_smcbar")
        assert law_func is not None, "gordon03_smcbar should be registered"

        # Test that it returns a callable
        assert callable(law_func), "Resolved law should be callable"

        # Test that it can evaluate on a wavelength grid
        wave_test = jnp.array([1000.0, 3000.0, 5500.0, 10000.0, 30000.0])
        k_test = law_func(wave_test)

        assert k_test.shape == wave_test.shape, "Output shape should match input"
        assert jnp.all(jnp.isfinite(k_test)), "All k values should be finite"
        assert jnp.all(k_test >= 0.0), "k(λ) should be non-negative"
        assert_allclose(float(k_test[2]), 1.0, atol=1e-6, err_msg="k(5500 Å) should equal 1.0")

    def test_differs_from_pei92(self):
        """gordon03_smcbar differs significantly from Pei92 SMC at short wavelengths."""
        pytest.importorskip("dust_extinction")
        from tengri.components.dust.attenuation import gordon03_smcbar, smc

        # Evaluate at 1000 Å (far-UV)
        wave_1000 = jnp.array([1000.0])
        k_gordon = gordon03_smcbar(wave_1000)[0]
        k_pei92 = smc(wave_1000)[0]

        ratio = float(k_gordon) / float(k_pei92)

        print("\nUV comparison at 1000 Å:")
        print(f"  gordon03_smcbar: {float(k_gordon):.6f}")
        print(f"  smc (Pei 1992):  {float(k_pei92):.6f}")
        print(f"  ratio (G03/Pei): {ratio:.6f}")

        # Gordon03 should have stronger UV attenuation than Pei92
        assert ratio > 1.25, (
            f"gordon03_smcbar should exceed Pei92 SMC by >25% at 1000 Å, got {ratio:.2f}x"
        )

    def test_hold_outside_range(self):
        """Wavelengths outside [1000, 33333 Å] hold boundary values.

        k(500 Å) should equal k(1000 Å) and k(50000 Å) should equal k(33333 Å).
        """
        pytest.importorskip("dust_extinction")
        from tengri.components.dust.attenuation import gordon03_smcbar

        # Get boundary values
        wave_boundaries = jnp.array([500.0, 1000.0, 33333.0, 50000.0])
        k_boundaries = gordon03_smcbar(wave_boundaries)

        # Check that k(500 Å) == k(1000 Å)
        assert_allclose(
            k_boundaries[0], k_boundaries[1], rtol=1e-10, err_msg="k(500 Å) should hold k(1000 Å)"
        )

        # Check that k(50000 Å) == k(33333 Å)
        assert_allclose(
            k_boundaries[3],
            k_boundaries[2],
            rtol=1e-4,
            err_msg="k(50000 Å) should hold k(33333 Å)",
        )

    def test_table4_published_values(self):
        """k(λ)/k(5500) at four Table 4 nodes matches published values.

        Gordon et al. (2003) Table 4 SMC-bar average provides A/A_V values
        at specific wavenumbers. This test pins these to published values
        to detect accidental shifts in the model or precompute.

        From Gordon et al. (2003) Table 4:
        x [µm⁻¹] = 0.455, 1.235, 2.273, 3.375
        A/A_V     = 0.110, 0.567, 1.374, 2.000
        """
        pytest.importorskip("dust_extinction")
        from tengri.components.dust.attenuation import gordon03_smcbar

        # Table 4 nodes: x [µm⁻¹] and published A/A_V
        x_um_published = np.array([0.455, 1.235, 2.273, 3.375])
        axav_published = np.array([0.110, 0.567, 1.374, 2.000])

        # Convert x [µm⁻¹] to wavelength [Å]
        wave_aa = 1e4 / x_um_published

        # Evaluate gordon03_smcbar
        k_gordon = np.asarray(gordon03_smcbar(jnp.asarray(wave_aa)))

        # dust_extinction obsdata tolerance: rtol 6e-2
        assert_allclose(
            k_gordon,
            axav_published,
            rtol=6e-2,
            err_msg="Published Table 4 values mismatch [Gordon et al. 2003]",
        )
