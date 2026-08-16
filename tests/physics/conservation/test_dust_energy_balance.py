# SPDX-License-Identifier: BSD-3-Clause
"""Tests for dust energy balance conservation.

Energy absorbed by dust must equal energy re-radiated (Kirchhoff's law).
"""

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.emission import (
    compute_absorbed_luminosity,
    energy_balance_split,
    modified_blackbody,
)

pytestmark = pytest.mark.conservation


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def wavelengths():
    """IR wavelength grid (Angstrom), 1 -- 1000 um."""
    return jnp.linspace(1e4, 1e7, 500)


@pytest.fixture
def L_absorbed():
    """Typical absorbed luminosity in Lsun."""
    return 1e10


# ── Energy conservation ───────────────────────────────────────────


class TestEnergyConservation:
    """Integral of output = eta * L_absorbed + L_agn_ir."""

    def test_default_eta(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

        sed = energy_balance_split(wavelengths, L_absorbed)
        nu = 2.99792458e10 / (wavelengths * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        assert jnp.isclose(integral, L_absorbed, rtol=0.05)

    def test_eta_half(self, wavelengths, L_absorbed):

        eta = 0.5
        sed = energy_balance_split(wavelengths, L_absorbed, eta_balance=eta)
        nu = 2.99792458e10 / (wavelengths * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        expected = eta * L_absorbed
        assert jnp.isclose(integral, expected, rtol=0.05)

    def test_eta_plus_agn(self, wavelengths, L_absorbed):

        eta = 0.8
        L_agn = 2e9
        sed = energy_balance_split(
            wavelengths,
            L_absorbed,
            L_agn_ir=L_agn,
            eta_balance=eta,
        )
        nu = 2.99792458e10 / (wavelengths * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        expected = eta * L_absorbed + L_agn
        assert jnp.isclose(integral, expected, rtol=0.05)


class TestModifiedBlackbodyEnergy:
    """Modified blackbody energy conservation."""

    @pytest.fixture
    def wave_fir(self):
        """Far-IR wavelength grid (Angstrom), 10 μm – 10 mm."""
        return jnp.logspace(5, 9, 400)

    def test_energy_conservation(self, wave_fir):
        """Integral of output ≈ L_absorbed (frequency integral)."""
        from tengri.components.dust.emission import modified_blackbody

        L_abs = 1e10
        sed = modified_blackbody(wave_fir, L_absorbed=L_abs, dust_T=30.0)
        nu = 2.99792458e10 / (wave_fir * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        np.testing.assert_allclose(float(integral), L_abs, rtol=0.02)


class TestCasey2012Energy:
    """Casey2012 model energy conservation."""

    @pytest.fixture
    def wave_ir(self):
        """Broad IR wavelength grid (Angstrom), 1 μm – 10 mm."""
        return jnp.logspace(4, 9, 500)

    def test_energy_conservation(self, wave_ir):
        """Integral of output ≈ L_absorbed."""
        from tengri.components.dust.emission import casey2012

        L_abs = 1e10
        sed = casey2012(wave_ir, L_absorbed=L_abs, dust_T=35.0)
        nu = 2.99792458e10 / (wave_ir * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        np.testing.assert_allclose(float(integral), L_abs, rtol=0.03)


class TestMbbWithCmbCorrection:
    """modified_blackbody high-z paths invoke cmb_corrected_temperature and conserve energy."""

    def test_high_z_sed_differs_from_z0(self):
        """SED at z=5 differs from z=0 due to CMB heating."""

        wave = jnp.logspace(5, 8, 200)
        sed_z0 = modified_blackbody(wave, 1e10, dust_T=20.0, redshift=0.0)
        sed_z5 = modified_blackbody(wave, 1e10, dust_T=20.0, redshift=5.0)
        # Peak should shift to shorter wavelengths at high z (higher T_eff)
        assert not jnp.allclose(sed_z0, sed_z5, rtol=0.01)

    def test_high_z_peaks_at_shorter_wavelength(self):
        """CMB heating at z=5 shifts the MBB peak to shorter wavelengths."""

        wave = jnp.logspace(5, 8, 500)
        sed_z0 = modified_blackbody(wave, 1e10, dust_T=20.0, redshift=0.0)
        sed_z5 = modified_blackbody(wave, 1e10, dust_T=20.0, redshift=5.0)
        peak_z0 = float(wave[jnp.argmax(sed_z0)])
        peak_z5 = float(wave[jnp.argmax(sed_z5)])
        assert peak_z5 < peak_z0

    def test_finite_at_extreme_redshift(self):
        """No NaN/Inf at z=10."""

        wave = jnp.logspace(4, 9, 200)
        sed = modified_blackbody(wave, 1e10, dust_T=30.0, redshift=10.0)
        chex.assert_tree_all_finite(sed)


class TestComputeAbsorbedLuminosity:
    """Tests for compute_absorbed_luminosity and compute_absorbed_luminosity_from_tau.

    Verify energy conservation in dust absorption/emission cycle.
    """

    def test_zero_absorption_gives_zero(self):
        """Perfect transmission (T=1) → no absorbed luminosity."""
        from tengri.components.dust.emission import compute_absorbed_luminosity

        wave = jnp.linspace(1e3, 1e7, 500)
        L_nu = jnp.ones_like(wave)
        result = float(compute_absorbed_luminosity(wave, L_nu, transmission=jnp.ones_like(wave)))
        assert abs(result) < 1e-10 * float(jnp.sum(L_nu))

    def test_full_absorption_gives_positive(self):
        """Zero transmission (T=0) → all energy absorbed → positive result."""

        wave = jnp.linspace(1e3, 1e7, 500)
        L_nu = jnp.ones_like(wave)
        result = float(compute_absorbed_luminosity(wave, L_nu, transmission=jnp.zeros_like(wave)))
        assert result > 0.0

    def test_partial_absorption_between_extremes(self):
        """Partial transmission produces result between zero and full-absorption."""

        wave = jnp.linspace(1e3, 1e7, 500)
        L_nu = jnp.ones_like(wave) * 1e10
        full = float(compute_absorbed_luminosity(wave, L_nu, jnp.zeros_like(wave)))
        half = float(compute_absorbed_luminosity(wave, L_nu, jnp.full_like(wave, 0.5)))
        assert 0.0 < half < full
        np.testing.assert_allclose(half, full * 0.5, rtol=1e-6)

    def test_from_tau_zero_gives_zero(self):
        """tau=0 → exp(-tau)=1 → zero absorbed."""
        from tengri.components.dust.emission import compute_absorbed_luminosity_from_tau

        wave = jnp.linspace(1e3, 1e7, 200)
        L_nu = jnp.ones_like(wave)
        result = float(compute_absorbed_luminosity_from_tau(wave, L_nu, jnp.zeros_like(wave)))
        assert abs(result) < 1e-10 * float(jnp.sum(L_nu))

    def test_from_tau_large_tau_matches_full_absorption(self):
        """Very large tau → exp(-tau)≈0 → same as full absorption."""
        from tengri.components.dust.emission import (
            compute_absorbed_luminosity,
            compute_absorbed_luminosity_from_tau,
        )

        wave = jnp.linspace(1e3, 1e7, 200)
        L_nu = jnp.ones_like(wave) * 1e8
        full = float(compute_absorbed_luminosity(wave, L_nu, jnp.zeros_like(wave)))
        large_tau = float(
            compute_absorbed_luminosity_from_tau(wave, L_nu, jnp.full_like(wave, 100.0))
        )
        np.testing.assert_allclose(large_tau, full, rtol=1e-4)


class TestAGNContribution:
    """AGN IR adds extra luminosity beyond stellar absorption."""

    @pytest.fixture
    def wavelengths(self):
        """IR wavelength grid (Angstrom), 1 -- 1000 um."""
        return jnp.linspace(1e4, 1e7, 500)

    @pytest.fixture
    def L_absorbed(self):
        """Typical absorbed luminosity in Lsun."""
        return 1e10

    def test_agn_adds_luminosity(self, wavelengths, L_absorbed):

        sed_no_agn = energy_balance_split(
            wavelengths,
            L_absorbed,
            L_agn_ir=0.0,
        )
        sed_with_agn = energy_balance_split(
            wavelengths,
            L_absorbed,
            L_agn_ir=5e9,
        )
        # With AGN, total integrated flux should be larger
        nu = 2.99792458e10 / (wavelengths * 1e-8)
        integral_no_agn = -jnp.trapezoid(sed_no_agn, nu)
        integral_with_agn = -jnp.trapezoid(sed_with_agn, nu)
        assert integral_with_agn > integral_no_agn

    def test_agn_only(self, wavelengths):
        """If L_absorbed_stellar=0, only AGN contributes."""

        L_agn = 1e10
        sed = energy_balance_split(
            wavelengths,
            L_absorbed_stellar=0.0,
            L_agn_ir=L_agn,
        )
        # Should be non-zero
        assert jnp.any(sed > 0.0)

        # Integral should approximate L_agn
        nu = 2.99792458e10 / (wavelengths * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        assert jnp.isclose(integral, L_agn, rtol=0.05)
