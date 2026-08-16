# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for pure JAX functions in sed_quantities.py.

Tests use synthetic data with known analytical results to verify
correctness. No SSP data files needed.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.bounds


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.utils.physics_constants import L_SUN as LSUN_ERG
from tengri.utils.sed_quantities import (
    C_AA,
    PC_CM,
    compute_balmer_break,
    compute_bolometric_luminosity,
    compute_dn4000,
    compute_fuv_flux,
    compute_ionizing_efficiency,
    compute_irx,
    compute_l_dust_absorbed,
    compute_l_radio_1p4ghz_from_sfr,
    compute_l_radio_thermal,
    compute_l_tir,
    compute_l_x_agn,
    compute_l_x_xrb,
    compute_luminosity_weighted_age,
    compute_luminosity_weighted_metallicity,
    compute_m_uv,
    compute_mass_weighted_age,
    compute_mass_weighted_metallicity,
    compute_nuv_flux,
    compute_q_ir,
    compute_rest_uv_color,
    compute_uv_luminosity_1600,
    compute_uv_slope_beta,
    extract_line_luminosity,
)

# ── Fixtures: synthetic wavelength grids and SEDs ─────────────────


@pytest.fixture
def wave():
    """Dense wavelength grid from 100 Å to 10^7 Å (UV through FIR)."""
    return jnp.logspace(2, 7, 5000)


@pytest.fixture
def flat_sed(wave):
    """Flat SED: L_ν = 1e30 erg/s/Hz at all wavelengths."""
    return jnp.ones_like(wave) * 1e30


# ── Mass-weighted age ─────────────────────────────────────────────


class TestMassWeightedAge:
    def test_equal_weights(self):
        weights = jnp.array([1.0, 1.0])
        ages = jnp.array([1e9, 3e9])
        result = compute_mass_weighted_age(weights, ages)
        assert jnp.isclose(result, 2.0, atol=1e-10)

    def test_single_bin(self):
        weights = jnp.array([5.0])
        ages = jnp.array([5e9])
        result = compute_mass_weighted_age(weights, ages)
        assert jnp.isclose(result, 5.0, atol=1e-10)

    def test_weighted_toward_young(self):
        weights = jnp.array([10.0, 1.0])
        ages = jnp.array([1e9, 10e9])
        result = compute_mass_weighted_age(weights, ages)
        # Weighted average: (10*1 + 1*10) / 11 = 20/11 ≈ 1.82 Gyr
        assert result < 2.0

    def test_zero_weights_returns_nan(self):
        """Zero total mass has no mass-weighted age, so the answer is NaN (#1404).

        **This assertion was inverted deliberately.** It previously required
        ``jnp.isfinite(result)`` — "no NaN or inf" — which the clamped denominator
        satisfied by returning exactly ``0.0``. But a mass-weighted age of zero
        Gyr reads as "every star just formed" rather than "there is no mass to
        weight by", and it flowed into ``pred.properties`` looking like a real
        measurement. Finiteness was a robustness preference, not a physics
        requirement; the quantity is genuinely undefined here.

        NaN-for-degenerate is already this file's convention — see
        ``TestLineExtraction.test_empty_returns_nan`` — and the property catalog
        tests assert it too (``test_property_catalog.py``, halpha / bpt_nii).
        """
        weights = jnp.array([0.0, 0.0])
        ages = jnp.array([1e9, 3e9])
        result = compute_mass_weighted_age(weights, ages)
        assert jnp.isnan(result), f"expected NaN for zero total weight, got {float(result)!r}"

    def test_nonzero_weights_stay_finite(self):
        """The #1404 guard is a gate on den>0; the ordinary path is untouched."""
        result = compute_mass_weighted_age(jnp.array([1.0, 1.0]), jnp.array([1e9, 3e9]))
        assert jnp.isfinite(result)


# ── Mass-weighted metallicity ─────────────────────────────────────


class TestMassWeightedMetallicity:
    def test_single_z(self):
        weights = jnp.array([1.0, 1.0])
        ages = jnp.array([1e9, 3e9])
        result = compute_mass_weighted_metallicity(weights, ages, log_z=-2.0)
        assert jnp.isclose(result, -2.0, atol=1e-10)

    def test_evolving_z(self):
        weights = jnp.array([1.0, 1.0])
        ages = jnp.array([0.0, 1e10])  # today and 10 Gyr ago
        result = compute_mass_weighted_metallicity(
            weights, ages, log_z=-2.0, log_z_initial=-3.0, log_z_final=-1.0
        )
        # Should be between initial and final
        assert -3.0 < float(result) < -1.0


# ── Bolometric luminosity ─────────────────────────────────────────


class TestBolometricLuminosity:
    def test_positive(self, wave, flat_sed):
        l_bol = compute_bolometric_luminosity(flat_sed, wave)
        assert l_bol > 0

    def test_zero_sed(self, wave):
        zero_sed = jnp.zeros_like(wave)
        l_bol = compute_bolometric_luminosity(zero_sed, wave)
        assert jnp.isclose(l_bol, 0.0, atol=1e-10)


# ── L_TIR ─────────────────────────────────────────────────────────


class TestLTIR:
    def test_ir_only_sed(self, wave):
        """SED nonzero only in IR should give l_tir ≈ l_bol."""
        sed = jnp.where((wave >= 8e4) & (wave <= 1e7), 1e30, 0.0)
        l_tir = compute_l_tir(sed, wave)
        l_bol = compute_bolometric_luminosity(sed, wave)
        assert jnp.isclose(l_tir, l_bol, rtol=0.01)

    def test_uv_only_sed(self, wave):
        """SED nonzero only in UV should give l_tir ≈ 0."""
        sed = jnp.where(wave < 3000.0, 1e30, 0.0)
        l_tir = compute_l_tir(sed, wave)
        assert l_tir < 1e-5 * compute_bolometric_luminosity(sed, wave)

    def test_non_negative(self, wave, flat_sed):
        l_tir = compute_l_tir(flat_sed, wave)
        assert l_tir >= 0


# ── L_dust_absorbed ───────────────────────────────────────────────


class TestLDustAbsorbed:
    def test_zero_when_no_dust(self, wave):
        sed = jnp.ones_like(wave) * 1e30
        result = compute_l_dust_absorbed(sed, sed, wave)
        assert jnp.isclose(result, 0.0, atol=1e-5)

    def test_positive_with_dust(self, wave):
        intrinsic = jnp.ones_like(wave) * 1e30
        attenuated = intrinsic * 0.5
        result = compute_l_dust_absorbed(intrinsic, attenuated, wave)
        assert result > 0


# ── UV slope beta ─────────────────────────────────────────────────


class TestUVSlopeBeta:
    def test_flat_fnu(self, wave):
        """Flat f_ν → β = -2 (since f_λ ∝ λ^-2 for flat f_ν)."""
        flat = jnp.ones_like(wave) * 1e30
        beta = compute_uv_slope_beta(flat, wave)
        assert jnp.isclose(beta, -2.0, atol=0.05)

    def test_steep_spectrum(self, wave):
        """f_ν ∝ ν^1 = (c/λ)^1 ∝ λ^-1 → β = -1 - 2 = -3."""
        nu = C_AA / wave
        sed = nu * 1e12  # f_ν ∝ ν
        beta = compute_uv_slope_beta(sed, wave)
        assert jnp.isclose(beta, -3.0, atol=0.1)


# ── Dn4000 and Balmer break ───────────────────────────────────────


class TestDn4000:
    def test_flat_sed(self, wave):
        flat = jnp.ones_like(wave) * 1e30
        dn = compute_dn4000(flat, wave)
        assert jnp.isclose(dn, 1.0, atol=0.01)

    def test_step_function(self, wave):
        """2x flux above 4000 Å → Dn4000 ≈ 2.0."""
        sed = jnp.where(wave >= 4000.0, 2e30, 1e30)
        dn = compute_dn4000(sed, wave)
        assert jnp.isclose(dn, 2.0, atol=0.05)


class TestBalmerBreak:
    def test_flat_sed(self, wave):
        flat = jnp.ones_like(wave) * 1e30
        bb = compute_balmer_break(flat, wave)
        assert jnp.isclose(bb, 1.0, atol=0.01)


# ── M_UV ──────────────────────────────────────────────────────────


class TestMUV:
    def test_known_luminosity(self, wave):
        """Check M_UV for a known flux at 1500 Å."""
        # L_ν = 1e28 erg/s/Hz at 1500 Å
        sed = jnp.ones_like(wave) * 1e28
        m_uv = compute_m_uv(sed, wave)
        # Expected: M = -2.5*log10(L/(4π*(10pc)^2)) - 48.6
        d = 10.0 * PC_CM
        f_nu = 1e28 / (4 * np.pi * d**2)
        expected = -2.5 * np.log10(f_nu) - 48.6
        assert jnp.isclose(m_uv, expected, atol=0.1)


# ── FUV, NUV fluxes ───────────────────────────────────────────────


class TestFUVNUV:
    def test_flat_sed_equal(self, wave, flat_sed):
        fuv = compute_fuv_flux(flat_sed, wave)
        nuv = compute_nuv_flux(flat_sed, wave)
        assert jnp.isclose(fuv, nuv, rtol=0.01)

    def test_positive(self, wave, flat_sed):
        assert compute_fuv_flux(flat_sed, wave) > 0
        assert compute_nuv_flux(flat_sed, wave) > 0


# ── IRX ───────────────────────────────────────────────────────────


class TestIRX:
    def test_known_ratio(self):
        # L_TIR = 1e10 Lsun, L_UV = 1e10 * LSUN_ERG erg/s → IRX = 0
        l_tir = 1e10
        l_uv = 1e10 * LSUN_ERG
        irx = compute_irx(l_tir, l_uv)
        assert jnp.isclose(irx, 0.0, atol=0.01)


# ── Rest-frame U-V color ──────────────────────────────────────────


class TestRestUVColor:
    def test_flat_sed_zero(self, wave, flat_sed):
        """Flat SED → U-V ≈ 0 (same flux in both bands)."""
        uv = compute_rest_uv_color(flat_sed, wave)
        assert jnp.isclose(uv, 0.0, atol=0.05)


# ── UV luminosity at 1600 Å ───────────────────────────────────────


class TestUVLuminosity1600:
    def test_positive(self, wave, flat_sed):
        l_uv = compute_uv_luminosity_1600(flat_sed, wave)
        assert l_uv > 0


# ── Emission line extraction ──────────────────────────────────────


class TestLineExtraction:
    def test_halpha_lookup(self):
        waves = jnp.array([4861.0, 6563.0, 6584.0])
        lums = jnp.array([1e7, 5e7, 2e7])
        result = extract_line_luminosity(waves, lums, (6563.0,))
        assert jnp.isclose(result, 5e7, rtol=0.01)

    def test_doublet_sum(self):
        waves = jnp.array([3726.0, 3729.0, 5007.0])
        lums = jnp.array([1e6, 2e6, 3e6])
        result = extract_line_luminosity(waves, lums, (3726.0, 3729.0))
        assert jnp.isclose(result, 3e6, rtol=0.01)

    def test_empty_returns_nan(self):
        waves = jnp.array([])
        lums = jnp.array([])
        result = extract_line_luminosity(waves, lums, (6563.0,))
        assert jnp.isnan(result)


# ── Radio quantities ──────────────────────────────────────────────


class TestRadio:
    def test_l_radio_positive(self):
        l = compute_l_radio_1p4ghz_from_sfr(1.0)
        assert l > 0

    def test_l_thermal_positive(self):
        l = compute_l_radio_thermal(1e53)
        assert l > 0

    def test_q_ir_finite(self):
        """q_IR should be finite and positive for physical inputs."""
        l_tir = 1e10  # Lsun
        l_radio = 1e22  # erg/s/Hz (typical for L_TIR ~ 1e10 Lsun)
        q = compute_q_ir(l_tir, l_radio)
        assert jnp.isfinite(q)
        assert float(q) > 0


# ── X-ray quantities ──────────────────────────────────────────────


class TestXRay:
    def test_xrb_positive(self):
        l_x = compute_l_x_xrb(1.0, 1e10)
        assert l_x > 0

    def test_xrb_dominated_by_hmxb_for_sfg(self):
        """For star-forming galaxies, HMXBs should dominate."""
        l_x = compute_l_x_xrb(10.0, 1e9)  # high SFR, low mass
        l_hmxb = 2.6e39 * 10.0
        assert float(l_x) > 0.5 * l_hmxb

    def test_agn_positive(self):
        l_x = compute_l_x_agn(1e44)
        assert l_x > 0


# ── Ionizing efficiency ───────────────────────────────────────────


class TestIonizingEfficiency:
    def test_typical_range(self):
        """log10(ξ_ion) should be ~25 for typical star-forming galaxies."""
        q_h = 1e54  # photons/s
        l_uv = 1e29  # erg/s/Hz (νLν ~ 1e44)
        xi = compute_ionizing_efficiency(q_h, l_uv)
        assert 24.0 < float(xi) < 26.0


# ── Luminosity-weighted quantities ────────────────────────────────


class TestLuminosityWeighted:
    def test_age_biased_to_bright(self):
        """Young bins are brighter, so L-weighted age < mass-weighted age."""
        n_age, n_wave = 10, 100
        wave = jnp.linspace(1000.0, 10000.0, n_wave)
        ages = jnp.linspace(1e8, 1e10, n_age)
        weights = jnp.ones(n_age)

        # Make young bins brighter (declining luminosity with age)
        ssp_flux = jnp.outer(1.0 / (ages / 1e8), jnp.ones(n_wave))

        age_mw = compute_mass_weighted_age(weights, ages)
        age_lw = compute_luminosity_weighted_age(weights, ssp_flux, ages, wave)
        assert age_lw < age_mw

    def test_metallicity_single_z(self):
        n_age, n_wave = 5, 50
        wave = jnp.linspace(1000.0, 10000.0, n_wave)
        ages = jnp.linspace(1e8, 1e10, n_age)
        weights = jnp.ones(n_age)
        ssp_flux = jnp.ones((n_age, n_wave))

        result = compute_luminosity_weighted_metallicity(weights, ssp_flux, ages, wave, log_z=-2.0)
        assert jnp.isclose(result, -2.0, atol=1e-10)


# ── JAX compatibility ─────────────────────────────────────────────


class TestJAXCompat:
    def test_grad_through_l_bol(self, wave):
        """Gradient flows through bolometric luminosity and matches FD."""

        def f(scale):
            sed = jnp.ones_like(wave) * scale
            return compute_bolometric_luminosity(sed, wave)

        # x0=1.0, not 1e30: at x0=1e30 eps/x0 < float64 machine epsilon so FD cancels to 0
        x0 = 1.0
        grad_jax = float(jax.grad(f)(x0))
        grad_fd = fd_grad(f, x0)
        np.testing.assert_allclose(
            grad_jax, grad_fd, rtol=1e-3, err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}"
        )
        assert grad_jax > 0

    def test_grad_through_beta(self, wave):
        """Gradient flows through UV slope and matches FD.

        For a scale-invariant quantity like UV slope beta (computed from log-log ratios),
        d(beta)/d(scale) = 0. Both autodiff and FD give 0 — use atol to handle near-zero.
        """

        def f(scale):
            sed = jnp.ones_like(wave) * scale
            return compute_uv_slope_beta(sed, wave)

        # x0=1.0, not 1e30: at x0=1e30 eps/x0 < float64 machine epsilon so FD cancels to 0
        x0 = 1.0
        grad_jax = float(jax.grad(f)(x0))
        grad_fd = fd_grad(f, x0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            atol=1e-6,
            rtol=5e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )

    def test_jit_compatible(self, wave, flat_sed):
        """All key functions work under jax.jit."""

        @jax.jit
        def compute_all(sed, w):
            return (
                compute_bolometric_luminosity(sed, w),
                compute_l_tir(sed, w),
                compute_uv_slope_beta(sed, w),
                compute_dn4000(sed, w),
                compute_m_uv(sed, w),
            )

        results = compute_all(flat_sed, wave)
        for r in results:
            assert jnp.isfinite(r)
