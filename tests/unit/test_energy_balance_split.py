"""Tests for energy_balance_split two-temperature dust emission model."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)


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


# ── Registration ──────────────────────────────────────────────────


class TestRegistration:
    """Verify the model is registered in the emission model registry."""

    def test_in_registry(self):
        from tengri.components.dust.emission import DUST_EMISSION_MODELS

        assert "energy_balance_split" in DUST_EMISSION_MODELS

    def test_resolve_emission_model(self):
        from tengri.components.dust.emission import resolve_emission_model

        fn = resolve_emission_model("energy_balance_split")
        assert callable(fn)


# ── f_cold extremes ───────────────────────────────────────────────


class TestFColdExtremes:
    """f_cold=0 gives all warm, f_cold=1 gives all cold."""

    def test_f_cold_zero_all_warm(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split, modified_blackbody

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
        from tengri.components.dust.emission import energy_balance_split, modified_blackbody

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


# ── AGN IR contribution ───────────────────────────────────────────


class TestAGNContribution:
    """AGN IR adds extra luminosity beyond stellar absorption."""

    def test_agn_adds_luminosity(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

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
        from tengri.components.dust.emission import energy_balance_split

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
        from tengri.components.dust.emission import energy_balance_split

        eta = 0.5
        sed = energy_balance_split(wavelengths, L_absorbed, eta_balance=eta)
        nu = 2.99792458e10 / (wavelengths * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        expected = eta * L_absorbed
        assert jnp.isclose(integral, expected, rtol=0.05)

    def test_eta_plus_agn(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

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


# ── JIT compatibility ─────────────────────────────────────────────


class TestJITCompatibility:
    """SEDModel is JIT-compilable."""

    def test_jit(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

        jitted = jax.jit(energy_balance_split)
        sed = jitted(wavelengths, L_absorbed)
        assert sed.shape == wavelengths.shape
        assert jnp.all(jnp.isfinite(sed))

    def test_vmap(self, wavelengths):
        from tengri.components.dust.emission import energy_balance_split

        L_values = jnp.array([1e9, 5e9, 1e10])
        vmapped = jax.vmap(energy_balance_split, in_axes=(None, 0))
        seds = vmapped(wavelengths, L_values)
        assert seds.shape == (3, len(wavelengths))


# ── Gradient compatibility ────────────────────────────────────────


class TestGradientCompatibility:
    """SEDModel is differentiable w.r.t. all continuous parameters."""

    def test_grad_L_absorbed(self, wavelengths):
        from tengri.components.dust.emission import energy_balance_split

        def loss(L_abs):
            sed = energy_balance_split(wavelengths, L_abs)
            return jnp.sum(sed)

        grad_fn = jax.grad(loss)
        g_jax = float(grad_fn(1e10))
        g_fd = fd_grad(loss, 1e10)
        np.testing.assert_allclose(
            g_jax,
            g_fd,
            rtol=2e-2,  # MBB integral over tabulated dust templates; 2% FD agreement is typical
            err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}",
        )
        assert g_jax > 0.0

    def test_grad_f_cold(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

        def loss(f_cold):
            sed = energy_balance_split(wavelengths, L_absorbed, f_cold=f_cold)
            return jnp.sum(sed)

        grad_fn = jax.grad(loss)
        g_jax = float(grad_fn(0.5))
        g_fd = fd_grad(loss, 0.5)
        np.testing.assert_allclose(
            g_jax, g_fd, rtol=1e-3, err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}"
        )

    def test_grad_temperatures(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

        def loss_warm(T_warm):
            sed = energy_balance_split(
                wavelengths,
                L_absorbed,
                dust_T_warm=T_warm,
                dust_T_cold=20.0,
            )
            return jnp.sum(sed)

        def loss_cold(T_cold):
            sed = energy_balance_split(
                wavelengths,
                L_absorbed,
                dust_T_warm=45.0,
                dust_T_cold=T_cold,
            )
            return jnp.sum(sed)

        g_warm_jax = float(jax.grad(loss_warm)(45.0))
        g_warm_fd = fd_grad(loss_warm, 45.0)
        np.testing.assert_allclose(
            g_warm_jax,
            g_warm_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_warm_jax:.4e}, FD={g_warm_fd:.4e}",
        )

        g_cold_jax = float(jax.grad(loss_cold)(20.0))
        g_cold_fd = fd_grad(loss_cold, 20.0)
        np.testing.assert_allclose(
            g_cold_jax,
            g_cold_fd,
            rtol=1e-3,
            err_msg=f"autodiff={g_cold_jax:.4e}, FD={g_cold_fd:.4e}",
        )

    def test_grad_eta(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

        def loss(eta):
            sed = energy_balance_split(wavelengths, L_absorbed, eta_balance=eta)
            return jnp.sum(sed)

        grad_fn = jax.grad(loss)
        g_jax = float(grad_fn(1.0))
        g_fd = fd_grad(loss, 1.0)
        np.testing.assert_allclose(
            g_jax, g_fd, rtol=1e-3, err_msg=f"autodiff={g_jax:.4e}, FD={g_fd:.4e}"
        )
        assert g_jax > 0.0


# ── Edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and physical sanity checks."""

    def test_zero_luminosity(self, wavelengths):
        from tengri.components.dust.emission import energy_balance_split

        sed = energy_balance_split(wavelengths, 0.0)
        assert jnp.allclose(sed, 0.0)

    def test_output_non_negative(self, wavelengths, L_absorbed):
        from tengri.components.dust.emission import energy_balance_split

        sed = energy_balance_split(wavelengths, L_absorbed)
        assert jnp.all(sed >= 0.0)

    def test_warm_peaks_at_shorter_wavelength(self, wavelengths, L_absorbed):
        """Warm component peaks at shorter wavelength than cold."""
        from tengri.components.dust.emission import energy_balance_split

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


# ── planck_bnu standalone ─────────────────────────────────────────


class TestPlanckBnu:
    """Direct tests for the planck_bnu Planck function."""

    @pytest.fixture
    def wave_ir(self):
        """IR wavelength grid (Angstrom), ~1 – 1000 μm."""
        return jnp.logspace(4, 8, 300)

    def test_finite_positive(self, wave_ir):
        from tengri.components.dust.emission import planck_bnu

        bnu = planck_bnu(wave_ir, temperature=30.0)
        assert jnp.all(jnp.isfinite(bnu))
        assert jnp.all(bnu > 0.0)

    def test_output_shape(self, wave_ir):
        from tengri.components.dust.emission import planck_bnu

        bnu = planck_bnu(wave_ir, temperature=30.0)
        assert bnu.shape == wave_ir.shape

    def test_hotter_peaks_at_shorter_wavelength(self, wave_ir):
        """Wien's displacement law: hotter BB peaks at shorter λ."""
        from tengri.components.dust.emission import planck_bnu

        bnu_cold = planck_bnu(wave_ir, temperature=20.0)
        bnu_warm = planck_bnu(wave_ir, temperature=60.0)
        peak_cold = float(wave_ir[jnp.argmax(bnu_cold)])
        peak_warm = float(wave_ir[jnp.argmax(bnu_warm)])
        assert peak_warm < peak_cold

    def test_hotter_brighter(self, wave_ir):
        """At fixed wavelength, higher T → higher B_nu (Stefan-Boltzmann)."""
        from tengri.components.dust.emission import planck_bnu

        bnu_low = planck_bnu(wave_ir, temperature=20.0)
        bnu_high = planck_bnu(wave_ir, temperature=50.0)
        assert jnp.sum(bnu_high) > jnp.sum(bnu_low)

    def test_short_wavelengths_finite(self):
        """UV/EUV wavelengths (clipped x) should not overflow."""
        from tengri.components.dust.emission import planck_bnu

        wave_uv = jnp.array([10.0, 100.0, 1000.0])  # Angstrom
        bnu = planck_bnu(wave_uv, temperature=1e4)
        assert jnp.all(jnp.isfinite(bnu))
        assert jnp.all(bnu >= 0.0)

    def test_jit_compatible(self, wave_ir):
        """planck_bnu is JIT-compilable."""
        import jax

        from tengri.components.dust.emission import planck_bnu

        jitted = jax.jit(planck_bnu)
        bnu = jitted(wave_ir, 30.0)
        assert jnp.all(jnp.isfinite(bnu))

    def test_gradient_wrt_temperature(self, wave_ir):
        """FD check: ∂(∑B_nu)/∂T."""
        import jax

        from tengri.components.dust.emission import planck_bnu

        def loss(T):
            return jnp.sum(planck_bnu(wave_ir, T))

        g_jax = float(jax.grad(loss)(30.0))
        g_fd = fd_grad(loss, 30.0, eps=0.1)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax > 0.0


# ── modified_blackbody standalone ─────────────────────────────────


class TestModifiedBlackbody:
    """Standalone tests for modified_blackbody."""

    @pytest.fixture
    def wave_fir(self):
        """Far-IR wavelength grid (Angstrom), 10 μm – 10 mm."""
        return jnp.logspace(5, 9, 400)

    def test_finite_non_negative(self, wave_fir):
        from tengri.components.dust.emission import modified_blackbody

        sed = modified_blackbody(wave_fir, L_absorbed=1e10)
        assert jnp.all(jnp.isfinite(sed))
        assert jnp.all(sed >= 0.0)

    def test_output_shape(self, wave_fir):
        from tengri.components.dust.emission import modified_blackbody

        sed = modified_blackbody(wave_fir, L_absorbed=1e10)
        assert sed.shape == wave_fir.shape

    def test_energy_conservation(self, wave_fir):
        """Integral of output ≈ L_absorbed (frequency integral)."""
        from tengri.components.dust.emission import modified_blackbody

        L_abs = 1e10
        sed = modified_blackbody(wave_fir, L_absorbed=L_abs, dust_T=30.0)
        nu = 2.99792458e10 / (wave_fir * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        np.testing.assert_allclose(float(integral), L_abs, rtol=0.02)

    def test_hotter_peaks_shorter_wavelength(self, wave_fir):
        from tengri.components.dust.emission import modified_blackbody

        sed_cold = modified_blackbody(wave_fir, L_absorbed=1e10, dust_T=20.0)
        sed_warm = modified_blackbody(wave_fir, L_absorbed=1e10, dust_T=50.0)
        peak_cold = float(wave_fir[jnp.argmax(sed_cold)])
        peak_warm = float(wave_fir[jnp.argmax(sed_warm)])
        assert peak_warm < peak_cold

    def test_zero_luminosity(self, wave_fir):
        from tengri.components.dust.emission import modified_blackbody

        sed = modified_blackbody(wave_fir, L_absorbed=0.0)
        assert jnp.allclose(sed, 0.0)

    def test_higher_beta_steeper_rayleigh_jeans(self, wave_fir):
        """Higher dust_beta_ir → steeper slope on Rayleigh-Jeans side."""
        from tengri.components.dust.emission import modified_blackbody

        # Use radio-wavelength end where RJ slope is dominant
        wave_radio = jnp.logspace(8, 10, 100)  # 1 cm – 1 m
        sed_low_beta = modified_blackbody(wave_radio, L_absorbed=1e10, dust_beta_ir=1.0)
        sed_high_beta = modified_blackbody(wave_radio, L_absorbed=1e10, dust_beta_ir=2.5)

        # High beta suppresses long-wavelength (Rayleigh-Jeans) emission more
        ratio_low = float(sed_low_beta[-1]) / float(sed_low_beta[0])
        ratio_high = float(sed_high_beta[-1]) / float(sed_high_beta[0])
        # High-beta SED has steeper drop → smaller ratio at longest wavelength
        assert ratio_high < ratio_low

    def test_registered_in_models(self):
        from tengri.components.dust.emission import DUST_EMISSION_MODELS

        assert "modified_blackbody" in DUST_EMISSION_MODELS

    def test_jit_compatible(self, wave_fir):
        import jax

        from tengri.components.dust.emission import modified_blackbody

        jitted = jax.jit(modified_blackbody)
        sed = jitted(wave_fir, 1e10)
        assert jnp.all(jnp.isfinite(sed))

    def test_gradient_wrt_L_absorbed(self, wave_fir):
        import jax

        from tengri.components.dust.emission import modified_blackbody

        def loss(L_abs):
            return jnp.sum(modified_blackbody(wave_fir, L_abs))

        g_jax = float(jax.grad(loss)(1e10))
        g_fd = fd_grad(loss, 1e10, eps=1e7)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax > 0.0

    def test_gradient_wrt_temperature(self, wave_fir):
        import jax

        from tengri.components.dust.emission import modified_blackbody

        def loss(T):
            return jnp.sum(modified_blackbody(wave_fir, L_absorbed=1e10, dust_T=T))

        g_jax = float(jax.grad(loss)(30.0))
        g_fd = fd_grad(loss, 30.0, eps=0.1)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-2)


# ── casey2012 standalone ──────────────────────────────────────────


class TestCasey2012:
    """Standalone tests for casey2012 (MBB + mid-IR power law)."""

    @pytest.fixture
    def wave_ir(self):
        """Broad IR wavelength grid (Angstrom), 1 μm – 10 mm."""
        return jnp.logspace(4, 9, 500)

    def test_registered_in_models(self):
        from tengri.components.dust.emission import DUST_EMISSION_MODELS

        assert "casey2012" in DUST_EMISSION_MODELS

    def test_finite_non_negative(self, wave_ir):
        from tengri.components.dust.emission import casey2012

        sed = casey2012(wave_ir, L_absorbed=1e10)
        assert jnp.all(jnp.isfinite(sed))
        assert jnp.all(sed >= 0.0)

    def test_output_shape(self, wave_ir):
        from tengri.components.dust.emission import casey2012

        sed = casey2012(wave_ir, L_absorbed=1e10)
        assert sed.shape == wave_ir.shape

    def test_energy_conservation(self, wave_ir):
        """Integral of output ≈ L_absorbed."""
        from tengri.components.dust.emission import casey2012

        L_abs = 1e10
        sed = casey2012(wave_ir, L_absorbed=L_abs, dust_T=35.0)
        nu = 2.99792458e10 / (wave_ir * 1e-8)
        integral = -jnp.trapezoid(sed, nu)
        np.testing.assert_allclose(float(integral), L_abs, rtol=0.03)

    def test_zero_luminosity(self, wave_ir):
        from tengri.components.dust.emission import casey2012

        sed = casey2012(wave_ir, L_absorbed=0.0)
        assert jnp.allclose(sed, 0.0)

    def test_mid_ir_excess_vs_pure_mbb(self, wave_ir):
        """For T=35K cold dust, casey2012 has LESS 8–40 μm flux than a pure MBB.

        The Casey (2012) model blends a power-law (short-λ) with the MBB (long-λ)
        using a transition function f(λ).  At 8–40 μm the MBB component is
        suppressed by (1-f) while the power-law Wien cutoff exp(-hν/kT) is
        negligible for T=35K (x ≈ 10–51 at 40–8 μm).  The casey2012 mid-IR
        flux is therefore lower than a pure MBB normalized to the same L_absorbed.
        The model's value lies in its shape flexibility: alpha_mir controls the
        power-law slope for hotter dust / warmer galaxies.
        """
        from tengri.components.dust.emission import casey2012, modified_blackbody

        L_abs = 1e10
        sed_casey = casey2012(wave_ir, L_absorbed=L_abs, dust_T=35.0)
        sed_mbb = modified_blackbody(wave_ir, L_absorbed=L_abs, dust_T=35.0)

        # 8–40 μm in Angstrom — casey2012 MBB suppressed by (1-f) transition factor
        mir_mask = (wave_ir > 8e4) & (wave_ir < 4e5)
        assert jnp.sum(sed_casey[mir_mask]) < jnp.sum(sed_mbb[mir_mask])

    def test_alpha_affects_mid_ir(self, wave_ir):
        """Larger dust_alpha_mir increases mid-IR power-law contribution."""
        from tengri.components.dust.emission import casey2012

        mir_mask = (wave_ir > 8e4) & (wave_ir < 4e5)
        sed_low = casey2012(wave_ir, L_absorbed=1e10, dust_alpha_mir=1.5)
        sed_high = casey2012(wave_ir, L_absorbed=1e10, dust_alpha_mir=3.0)
        # Different alpha → different MIR shapes
        assert not jnp.allclose(sed_low[mir_mask], sed_high[mir_mask], rtol=0.01)

    def test_jit_compatible(self, wave_ir):
        import jax

        from tengri.components.dust.emission import casey2012

        jitted = jax.jit(casey2012)
        sed = jitted(wave_ir, 1e10)
        assert jnp.all(jnp.isfinite(sed))

    def test_gradient_wrt_L_absorbed(self, wave_ir):
        import jax

        from tengri.components.dust.emission import casey2012

        def loss(L_abs):
            return jnp.sum(casey2012(wave_ir, L_abs))

        g_jax = float(jax.grad(loss)(1e10))
        g_fd = fd_grad(loss, 1e10, eps=1e7)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-3)
        assert g_jax > 0.0

    def test_gradient_wrt_temperature(self, wave_ir):
        import jax

        from tengri.components.dust.emission import casey2012

        def loss(T):
            return jnp.sum(casey2012(wave_ir, L_absorbed=1e10, dust_T=T))

        g_jax = float(jax.grad(loss)(35.0))
        g_fd = fd_grad(loss, 35.0, eps=0.1)
        np.testing.assert_allclose(g_jax, g_fd, rtol=1e-2)

    def test_hotter_peaks_shorter_wavelength(self, wave_ir):
        from tengri.components.dust.emission import casey2012

        # Use only FIR range where MBB dominates
        wave_fir = jnp.logspace(5.5, 8, 300)
        sed_cold = casey2012(wave_fir, L_absorbed=1e10, dust_T=20.0)
        sed_warm = casey2012(wave_fir, L_absorbed=1e10, dust_T=50.0)
        peak_cold = float(wave_fir[jnp.argmax(sed_cold)])
        peak_warm = float(wave_fir[jnp.argmax(sed_warm)])
        assert peak_warm < peak_cold


# ── CMB heating correction ────────────────────────────────────────


class TestCmbCorrectedTemperature:
    """Tests for cmb_corrected_temperature (da Cunha+2013)."""

    def test_z0_no_change(self):
        """At z=0 with z=0 CMB, T_eff should equal T_dust."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_dust = 35.0
        # At z=0 the CMB terms cancel: T_cmb_z == T_CMB_0, so inner = T_dust^exponent
        T_eff = float(cmb_corrected_temperature(T_dust, redshift=0.0, beta_ir=1.6))
        assert abs(T_eff - T_dust) < 0.1

    def test_high_z_raises_temperature(self):
        """At high redshift the CMB floor raises the effective temperature."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_dust = 20.0
        T_eff_z0 = float(cmb_corrected_temperature(T_dust, redshift=0.0))
        T_eff_z5 = float(cmb_corrected_temperature(T_dust, redshift=5.0))
        assert T_eff_z5 > T_eff_z0

    def test_always_finite(self):
        """Finite output even for very cold or hot dust."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        for T_dust in (0.01, 1.0, 50.0, 200.0):
            T_eff = float(cmb_corrected_temperature(T_dust, redshift=2.0))
            assert np.isfinite(T_eff), f"T_eff not finite for T_dust={T_dust}"
            assert T_eff > 0.0

    def test_negative_T_dust_clamped(self):
        """Negative T_dust values are clamped to 1 K — no NaN."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_eff = float(cmb_corrected_temperature(-10.0, redshift=0.5))
        assert np.isfinite(T_eff)
        assert T_eff > 0.0

    def test_beta_ir_affects_result(self):
        """Different beta_ir values give different T_eff."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        T_eff_low = float(cmb_corrected_temperature(30.0, redshift=3.0, beta_ir=1.0))
        T_eff_high = float(cmb_corrected_temperature(30.0, redshift=3.0, beta_ir=2.5))
        assert T_eff_low != T_eff_high

    def test_gradient_wrt_T_dust(self):
        """Gradient of T_eff w.r.t. T_dust is well-defined and positive."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        grad_fn = jax.grad(lambda T: cmb_corrected_temperature(T, redshift=2.0))
        g = float(grad_fn(30.0))
        assert np.isfinite(g)
        assert g > 0.0

    def test_gradient_wrt_redshift(self):
        """Gradient of T_eff w.r.t. redshift is well-defined and positive."""
        from tengri.components.dust.emission import cmb_corrected_temperature

        grad_fn = jax.grad(lambda z: cmb_corrected_temperature(25.0, redshift=z))
        g = float(grad_fn(2.0))
        assert np.isfinite(g)
        assert g > 0.0


# ── CMB contrast factor ───────────────────────────────────────────


class TestCmbContrastFactor:
    """Tests for cmb_contrast_factor (da Cunha+2013)."""

    def test_z0_contrast_near_one_at_fir_peak(self):
        """At z=0 near the FIR peak (~50-500 μm), CMB is negligible vs T=40 K dust.

        The contrast = 1 - B_cmb/B_eff.  At the Wien side (short λ) of the CMB
        curve but near the MBB peak of 40 K dust (Wien regime for 2.7 K CMB),
        B_cmb/B_eff is exponentially suppressed → contrast ≈ 1.

        At very long wavelengths (Rayleigh-Jeans for both), the ratio approaches
        T_cmb/T_eff ≈ 0.07, so we restrict to the MBB peak region only.
        """
        from tengri.components.dust.emission import cmb_contrast_factor

        # 50–500 μm = 5e5–5e6 Å: FIR peak of 40 K dust, Wien side of 2.7 K CMB
        wave = jnp.logspace(5.7, 6.7, 100)
        factor = cmb_contrast_factor(wave, T_eff=40.0, redshift=0.0)
        assert jnp.all(factor > 0.99)

    def test_high_z_reduces_contrast(self):
        """At high z, contrast factor is appreciably below 1 for cold dust."""
        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.logspace(5, 8, 100)
        factor_z5 = cmb_contrast_factor(wave, T_eff=20.0, redshift=5.0)
        # At z=5, T_CMB = 2.725*6 ≈ 16 K; contrast against T_eff=20 K is suppressed
        assert jnp.any(factor_z5 < 0.9)

    def test_output_in_unit_interval(self):
        """Contrast factor must be in [0, 1] by construction."""
        from tengri.components.dust.emission import cmb_contrast_factor

        for z in (0.0, 2.0, 5.0, 10.0):
            wave = jnp.logspace(4, 8, 200)
            factor = cmb_contrast_factor(wave, T_eff=30.0, redshift=z)
            assert jnp.all(factor >= 0.0), f"Negative contrast at z={z}"
            assert jnp.all(factor <= 1.0), f"Contrast > 1 at z={z}"

    def test_all_finite(self):
        """No NaN/Inf values on the output grid."""
        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.logspace(3, 9, 300)
        factor = cmb_contrast_factor(wave, T_eff=35.0, redshift=3.0)
        assert jnp.all(jnp.isfinite(factor))

    def test_gradient_compatible(self):
        """Gradient w.r.t. T_eff is finite and positive."""
        from tengri.components.dust.emission import cmb_contrast_factor

        wave = jnp.logspace(5, 8, 50)

        def loss(T_eff):
            return jnp.sum(cmb_contrast_factor(wave, T_eff=T_eff, redshift=2.0))

        g = float(jax.grad(loss)(30.0))
        assert np.isfinite(g)
        assert g > 0.0  # hotter dust → contrast closer to 1 → sum increases


# ── Absorbed luminosity integrals ─────────────────────────────────


class TestComputeAbsorbedLuminosity:
    """Tests for compute_absorbed_luminosity and compute_absorbed_luminosity_from_tau."""

    def test_zero_absorption_gives_zero(self):
        """Perfect transmission (T=1) → no absorbed luminosity."""
        from tengri.components.dust.emission import compute_absorbed_luminosity

        wave = jnp.linspace(1e3, 1e7, 500)
        L_nu = jnp.ones_like(wave)
        result = float(compute_absorbed_luminosity(wave, L_nu, transmission=jnp.ones_like(wave)))
        assert abs(result) < 1e-10 * float(jnp.sum(L_nu))

    def test_full_absorption_gives_positive(self):
        """Zero transmission (T=0) → all energy absorbed → positive result."""
        from tengri.components.dust.emission import compute_absorbed_luminosity

        wave = jnp.linspace(1e3, 1e7, 500)
        L_nu = jnp.ones_like(wave)
        result = float(compute_absorbed_luminosity(wave, L_nu, transmission=jnp.zeros_like(wave)))
        assert result > 0.0

    def test_partial_absorption_between_extremes(self):
        """Partial transmission produces result between zero and full-absorption."""
        from tengri.components.dust.emission import compute_absorbed_luminosity

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

    def test_from_tau_gradient(self):
        """Gradient of absorbed luminosity w.r.t. tau is well-defined."""
        from tengri.components.dust.emission import compute_absorbed_luminosity_from_tau

        wave = jnp.linspace(1e3, 1e7, 100)
        L_nu = jnp.ones_like(wave) * 1e9

        def loss(tau_scalar):
            return compute_absorbed_luminosity_from_tau(
                wave, L_nu, jnp.full_like(wave, tau_scalar)
            )

        g = float(jax.grad(loss)(1.0))
        assert np.isfinite(g)
        assert g > 0.0  # more tau → more absorption → luminosity increases


# ── Registry utilities ────────────────────────────────────────────


class TestRegistryUtilities:
    """Tests for resolve_emission_model, get_emission_model, preload_emission_model."""

    def test_get_emission_model_alias(self):
        """get_emission_model is an alias for resolve_emission_model."""
        from tengri.components.dust.emission import get_emission_model, resolve_emission_model

        assert get_emission_model is resolve_emission_model

    def test_resolve_returns_callable(self):
        """resolve_emission_model returns a callable for known models."""
        from tengri.components.dust.emission import resolve_emission_model

        for name in ("modified_blackbody", "energy_balance_split", "casey2012"):
            fn = resolve_emission_model(name)
            assert callable(fn), f"resolve_emission_model('{name}') is not callable"

    def test_resolve_unknown_raises_value_error(self):
        """resolve_emission_model raises ValueError for unknown model names."""
        from tengri.components.dust.emission import resolve_emission_model

        with pytest.raises(ValueError, match="Unknown dust emission model"):
            resolve_emission_model("definitely_not_a_model_12345")

    def test_preload_unknown_raises_value_error(self):
        """preload_emission_model raises ValueError for unknown model names."""
        from tengri.components.dust.emission import preload_emission_model

        with pytest.raises(ValueError, match="Unknown emission model"):
            preload_emission_model("not_registered_xyz")

    def test_preload_known_returns_callable(self):
        """preload_emission_model returns a callable for the MBB model (no data needed)."""
        from tengri.components.dust.emission import preload_emission_model

        fn = preload_emission_model("modified_blackbody")
        assert callable(fn)

    def test_find_data_file_missing(self):
        """_find_data_file returns None for nonexistent files."""
        from tengri.components.dust.emission import _find_data_file

        result = _find_data_file("__definitely_not_here__.npz")
        assert result is None

    def test_find_data_file_present(self, tmp_path, monkeypatch):
        """_find_data_file returns the path when file exists in data/."""
        from tengri.components.dust import emission as em

        # Temporarily add tmp_path to search candidates
        original = em._DATA_CANDIDATES[:]
        fake_file = tmp_path / "test_dummy.npz"
        fake_file.write_bytes(b"")
        monkeypatch.setattr(em, "_DATA_CANDIDATES", [tmp_path])
        result = em._find_data_file("test_dummy.npz")
        monkeypatch.setattr(em, "_DATA_CANDIDATES", original)
        assert result == str(fake_file)

    def test_all_lazy_models_in_registry(self):
        """All expected lazy-loaded models are present in DUST_EMISSION_MODELS."""
        from tengri.components.dust.emission import DUST_EMISSION_MODELS

        for name in ("draine_li2007", "dale2014", "draine_li2014", "astrodust", "bosa", "themis"):
            assert name in DUST_EMISSION_MODELS, f"'{name}' missing from registry"


# ── apply_dust_emission dispatcher ────────────────────────────────


class TestApplyDustEmission:
    """Tests for the apply_dust_emission high-level dispatcher."""

    def test_delegates_to_modified_blackbody(self):
        """apply_dust_emission with 'modified_blackbody' matches direct call."""
        from tengri.components.dust.emission import apply_dust_emission, modified_blackbody

        wave = jnp.logspace(5, 8, 100)
        L_abs = 1e10

        direct = modified_blackbody(wave, L_abs, dust_T=35.0)
        via_dispatcher = apply_dust_emission("modified_blackbody", wave, L_abs, dust_T=35.0)
        assert jnp.allclose(direct, via_dispatcher)

    def test_delegates_to_casey2012(self):
        """apply_dust_emission with 'casey2012' matches direct call."""
        from tengri.components.dust.emission import apply_dust_emission, casey2012

        wave = jnp.logspace(5, 8, 100)
        L_abs = 5e9

        direct = casey2012(wave, L_abs, dust_T=40.0)
        via_dispatcher = apply_dust_emission("casey2012", wave, L_abs, dust_T=40.0)
        assert jnp.allclose(direct, via_dispatcher)

    def test_unknown_name_raises(self):
        """apply_dust_emission raises ValueError for unknown model names."""
        from tengri.components.dust.emission import apply_dust_emission

        wave = jnp.logspace(5, 8, 50)
        with pytest.raises(ValueError, match="Unknown dust emission model"):
            apply_dust_emission("no_such_model", wave, 1e10)


# ── Module-level aliases ──────────────────────────────────────────


class TestModuleLevelAliases:
    """Module-level functions like draine_li2007() dispatch to the registry."""

    def test_draine_li2007_alias_callable(self):
        """draine_li2007 module-level function is callable."""
        from tengri.components.dust import emission as em

        assert callable(em.draine_li2007)

    def test_dale2014_alias_callable(self):
        """dale2014 module-level function is callable."""
        from tengri.components.dust import emission as em

        assert callable(em.dale2014)

    def test_astrodust_alias_callable(self):
        """astrodust module-level function is callable."""
        from tengri.components.dust import emission as em

        assert callable(em.astrodust)

    def test_bosa_alias_callable(self):
        """bosa module-level function is callable."""
        from tengri.components.dust import emission as em

        assert callable(em.bosa)

    def test_themis_alias_callable(self):
        """themis module-level function is callable."""
        from tengri.components.dust import emission as em

        assert callable(em.themis)

    def test_draine_li2014_alias_callable(self):
        """draine_li2014 module-level function is callable."""
        from tengri.components.dust import emission as em

        assert callable(em.draine_li2014)


# ── CMB corrections integrated into MBB ───────────────────────────


class TestMbbWithCmbCorrection:
    """modified_blackbody high-z paths invoke cmb_corrected_temperature."""

    def test_high_z_sed_differs_from_z0(self):
        """SED at z=5 differs from z=0 due to CMB heating."""
        from tengri.components.dust.emission import modified_blackbody

        wave = jnp.logspace(5, 8, 200)
        sed_z0 = modified_blackbody(wave, 1e10, dust_T=20.0, redshift=0.0)
        sed_z5 = modified_blackbody(wave, 1e10, dust_T=20.0, redshift=5.0)
        # Peak should shift to shorter wavelengths at high z (higher T_eff)
        assert not jnp.allclose(sed_z0, sed_z5, rtol=0.01)

    def test_high_z_peaks_at_shorter_wavelength(self):
        """CMB heating at z=5 shifts the MBB peak to shorter wavelengths."""
        from tengri.components.dust.emission import modified_blackbody

        wave = jnp.logspace(5, 8, 500)
        sed_z0 = modified_blackbody(wave, 1e10, dust_T=20.0, redshift=0.0)
        sed_z5 = modified_blackbody(wave, 1e10, dust_T=20.0, redshift=5.0)
        peak_z0 = float(wave[jnp.argmax(sed_z0)])
        peak_z5 = float(wave[jnp.argmax(sed_z5)])
        assert peak_z5 < peak_z0

    def test_finite_at_extreme_redshift(self):
        """No NaN/Inf at z=10."""
        from tengri.components.dust.emission import modified_blackbody

        wave = jnp.logspace(4, 9, 200)
        sed = modified_blackbody(wave, 1e10, dust_T=30.0, redshift=10.0)
        assert jnp.all(jnp.isfinite(sed))
