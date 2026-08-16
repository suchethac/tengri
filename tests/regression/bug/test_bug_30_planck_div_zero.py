# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for BUG-30: Planck function divide-by-zero.

See ADR / docs/known_bugs.md for full context.
"""

import jax.numpy as jnp
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.regression_bug


class TestBug30PlanckDivZero:
    """emission.py:159-160 — exp(x)-1 must not be zero.

    At long wavelengths, x = hν/kT → 0, and the naive formula
    B_ν = 2hν³/(c² (exp(x)-1)) → inf/inf.

    Fix: use expm1(x) = exp(x)-1 to avoid cancellation, and special-case x→0.
    """

    def test_planck_finite_at_long_wavelength(self):
        """B_nu must be finite at very long wavelengths (Rayleigh-Jeans)."""
        try:
            from tengri.components.dust.emission import planck_bnu
        except ImportError:
            pytest.skip("planck_bnu not available")

        # 1 mm wavelength, T=30 K: x = hnu/kT ~ 0.005
        # 10 mm wavelength, T=30 K: x ~ 0.0005
        # Very long wavelength: x -> 0
        wave_aa = jnp.array([1e7, 1e8, 1e9])  # 1mm, 10mm, 100mm in Angstrom
        T = 30.0

        result = planck_bnu(wave_aa, T)
        assert jnp.all(jnp.isfinite(result)), f"Planck function has non-finite values: {result}"
        assert jnp.all(result > 0), "Planck function must be positive"

    def test_planck_finite_with_float32_uv_input(self):
        """B_nu must be finite when given float32 input at short UV wavelengths.

        Root cause: ssp_wave is stored as float32 in HDF5 files. With JAX's
        weak-type promotion, ``float32_array * Python_float`` stays float32 even
        with x64 enabled globally. At 5.6 Å, nu = 5.35e17 Hz and nu**3 ~ 1.5e53,
        far beyond float32 max (~3.4e38). Without an explicit float64 cast inside
        planck_bnu, nu**3 overflows to Inf and expm1(x) = Inf, giving Inf/Inf = NaN.
        """
        try:
            from tengri.components.dust.emission import planck_bnu
        except ImportError:
            pytest.skip("planck_bnu not available")

        # Mimic the actual SSP wavelength array dtype (float32 from HDF5)
        wave_aa = jnp.array([5.6, 10.0, 50.0, 100.0, 1000.0, 5000.0], dtype=jnp.float32)
        T = 35.0

        result = planck_bnu(wave_aa, T)
        assert jnp.all(jnp.isfinite(result)), (
            f"planck_bnu returned non-finite values for float32 input: {result}. "
            "Check float64 cast inside planck_bnu."
        )
        assert_non_negative(result, name="result", msg="Planck function must be non-negative")
