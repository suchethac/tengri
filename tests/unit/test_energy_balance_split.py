"""Tests for energy_balance_split two-temperature dust emission model."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wavelengths():
    """IR wavelength grid (Angstrom), 1 -- 1000 um."""
    return jnp.linspace(1e4, 1e7, 500)


@pytest.fixture
def L_absorbed():
    """Typical absorbed luminosity in Lsun."""
    return 1e10


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestRegistration:
    """Verify the model is registered in the emission model registry."""

    def test_in_registry(self):
        from tengri.models.dust.emission import DUST_EMISSION_MODELS

        assert "energy_balance_split" in DUST_EMISSION_MODELS

    def test_resolve_emission_model(self):
        from tengri.models.dust.emission import resolve_emission_model

        fn = resolve_emission_model("energy_balance_split")
        assert callable(fn)


# ---------------------------------------------------------------------------
# f_cold extremes
# ---------------------------------------------------------------------------


class TestFColdExtremes:
    """f_cold=0 gives all warm, f_cold=1 gives all cold."""

    def test_f_cold_zero_all_warm(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split, modified_blackbody

        sed = energy_balance_split(
            wavelengths,
            L_absorbed,
            f_cold=0.0,
            dust_T_warm=45.0,
            dust_T_cold=20.0,
        )
        expected = modified_blackbody(
            wavelengths,
            L_absorbed=L_absorbed,
            dust_T=45.0,
            dust_beta_ir=1.5,
        )
        assert jnp.allclose(sed, expected, rtol=1e-10)

    def test_f_cold_one_all_cold(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split, modified_blackbody

        sed = energy_balance_split(
            wavelengths,
            L_absorbed,
            f_cold=1.0,
            dust_T_warm=45.0,
            dust_T_cold=20.0,
        )
        expected = modified_blackbody(
            wavelengths,
            L_absorbed=L_absorbed,
            dust_T=20.0,
            dust_beta_ir=2.0,
        )
        assert jnp.allclose(sed, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# AGN IR contribution
# ---------------------------------------------------------------------------


class TestAGNContribution:
    """AGN IR adds extra luminosity beyond stellar absorption."""

    def test_agn_adds_luminosity(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split

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
        from tengri.models.dust.emission import energy_balance_split

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


# ---------------------------------------------------------------------------
# Energy conservation
# ---------------------------------------------------------------------------


class TestEnergyConservation:
    """Integral of output = eta * L_absorbed + L_agn_ir."""

    def test_default_eta(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split

        sed = energy_balance_split(wavelengths, L_absorbed)
        nu = 2.99792458e10 / (wavelengths * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        assert jnp.isclose(integral, L_absorbed, rtol=0.05)

    def test_eta_half(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split

        eta = 0.5
        sed = energy_balance_split(wavelengths, L_absorbed, eta_balance=eta)
        nu = 2.99792458e10 / (wavelengths * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        expected = eta * L_absorbed
        assert jnp.isclose(integral, expected, rtol=0.05)

    def test_eta_plus_agn(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split

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


# ---------------------------------------------------------------------------
# JIT compatibility
# ---------------------------------------------------------------------------


class TestJITCompatibility:
    """Model is JIT-compilable."""

    def test_jit(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split

        jitted = jax.jit(energy_balance_split)
        sed = jitted(wavelengths, L_absorbed)
        assert sed.shape == wavelengths.shape
        assert jnp.all(jnp.isfinite(sed))

    def test_vmap(self, wavelengths):
        from tengri.models.dust.emission import energy_balance_split

        L_values = jnp.array([1e9, 5e9, 1e10])
        vmapped = jax.vmap(energy_balance_split, in_axes=(None, 0))
        seds = vmapped(wavelengths, L_values)
        assert seds.shape == (3, len(wavelengths))


# ---------------------------------------------------------------------------
# Gradient compatibility
# ---------------------------------------------------------------------------


class TestGradientCompatibility:
    """Model is differentiable w.r.t. all continuous parameters."""

    def test_grad_L_absorbed(self, wavelengths):
        from tengri.models.dust.emission import energy_balance_split

        def loss(L_abs):
            sed = energy_balance_split(wavelengths, L_abs)
            return jnp.sum(sed)

        grad_fn = jax.grad(loss)
        g = grad_fn(1e10)
        assert jnp.isfinite(g)
        assert g > 0.0

    def test_grad_f_cold(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split

        def loss(f_cold):
            sed = energy_balance_split(wavelengths, L_absorbed, f_cold=f_cold)
            return jnp.sum(sed)

        grad_fn = jax.grad(loss)
        g = grad_fn(0.5)
        assert jnp.isfinite(g)

    def test_grad_temperatures(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split

        def loss(T_warm, T_cold):
            sed = energy_balance_split(
                wavelengths,
                L_absorbed,
                dust_T_warm=T_warm,
                dust_T_cold=T_cold,
            )
            return jnp.sum(sed)

        grad_fn = jax.grad(loss, argnums=(0, 1))
        g_warm, g_cold = grad_fn(45.0, 20.0)
        assert jnp.isfinite(g_warm)
        assert jnp.isfinite(g_cold)

    def test_grad_eta(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split

        def loss(eta):
            sed = energy_balance_split(wavelengths, L_absorbed, eta_balance=eta)
            return jnp.sum(sed)

        grad_fn = jax.grad(loss)
        g = grad_fn(1.0)
        assert jnp.isfinite(g)
        assert g > 0.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and physical sanity checks."""

    def test_zero_luminosity(self, wavelengths):
        from tengri.models.dust.emission import energy_balance_split

        sed = energy_balance_split(wavelengths, 0.0)
        assert jnp.allclose(sed, 0.0)

    def test_output_non_negative(self, wavelengths, L_absorbed):
        from tengri.models.dust.emission import energy_balance_split

        sed = energy_balance_split(wavelengths, L_absorbed)
        assert jnp.all(sed >= 0.0)

    def test_warm_peaks_at_shorter_wavelength(self, wavelengths, L_absorbed):
        """Warm component peaks at shorter wavelength than cold."""
        from tengri.models.dust.emission import energy_balance_split

        sed_warm_only = energy_balance_split(
            wavelengths,
            L_absorbed,
            f_cold=0.0,
            dust_T_warm=50.0,
            dust_T_cold=15.0,
        )
        sed_cold_only = energy_balance_split(
            wavelengths,
            L_absorbed,
            f_cold=1.0,
            dust_T_warm=50.0,
            dust_T_cold=15.0,
        )
        peak_warm = wavelengths[jnp.argmax(sed_warm_only)]
        peak_cold = wavelengths[jnp.argmax(sed_cold_only)]
        # Wien's law: hotter dust peaks at shorter wavelength
        assert peak_warm < peak_cold
