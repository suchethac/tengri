"""Tests for polar dust extinction and greybody reemission."""

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri.models.agn.polar_dust import (
    _type1_mask,
    polar_dust_emission,
    polar_dust_extinction,
    polar_dust_total,
)

# Shared test wavelength grid: 500 A to 10 mm (log-spaced, 500 points)
WAVELENGTH = jnp.logspace(jnp.log10(500.0), jnp.log10(1e8), 500)

# Flat input spectrum for testing (arbitrary units)
L_NU_FLAT = jnp.ones_like(WAVELENGTH)

# Realistic-ish disc spectrum (power-law, brighter in UV)
L_NU_DISC = 1e10 * (WAVELENGTH / 5000.0) ** (-1.5)


class TestPolarDustExtinction:
    """Tests for polar_dust_extinction."""

    def test_no_extinction_when_ebv_zero(self):
        """E(B-V)=0 should return input unchanged."""
        l_atten, l_abs = polar_dust_extinction(
            L_NU_DISC, WAVELENGTH, cos_inc=1.0, opening_angle_deg=40.0, ebv=0.0
        )
        assert jnp.allclose(l_atten, L_NU_DISC, rtol=1e-10)
        assert jnp.allclose(l_abs, 0.0, atol=1e-10)

    def test_no_extinction_for_type2(self):
        """Edge-on (cos_inc=0) with opening_angle=40 => Type 2, no extinction."""
        l_atten, l_abs = polar_dust_extinction(
            L_NU_DISC, WAVELENGTH, cos_inc=0.0, opening_angle_deg=40.0, ebv=0.3
        )
        # Type 2: mask ~ 0, so l_atten ~ l_nu
        assert jnp.allclose(l_atten, L_NU_DISC, rtol=1e-4)
        assert jnp.all(l_abs < 1e-4 * jnp.max(L_NU_DISC))

    def test_extinction_for_type1(self):
        """Face-on (cos_inc=1) with opening_angle=40 => Type 1, significant extinction."""
        l_atten, l_abs = polar_dust_extinction(
            L_NU_DISC, WAVELENGTH, cos_inc=1.0, opening_angle_deg=40.0, ebv=0.3
        )
        # UV should be significantly attenuated
        uv_mask = WAVELENGTH < 2000.0
        assert jnp.any(l_atten[uv_mask] < 0.5 * L_NU_DISC[uv_mask])
        # Some luminosity absorbed
        assert jnp.sum(l_abs) > 0.0

    def test_sigmoid_smooth_transition(self):
        """At the Type 1/2 boundary, extinction should be ~50%."""
        opening = 40.0
        # cos_threshold = cos(90 - 40) = cos(50 deg) ~ 0.6428
        cos_boundary = jnp.cos(jnp.radians(90.0 - opening))

        mask = _type1_mask(cos_boundary, opening)
        # At the boundary, sigmoid(0) = 0.5
        assert jnp.abs(mask - 0.5) < 0.01

    def test_higher_ebv_more_extinction(self):
        """E(B-V)=0.3 should attenuate more than E(B-V)=0.1."""
        l_atten_low, _ = polar_dust_extinction(
            L_NU_DISC, WAVELENGTH, cos_inc=1.0, opening_angle_deg=40.0, ebv=0.1
        )
        l_atten_high, _ = polar_dust_extinction(
            L_NU_DISC, WAVELENGTH, cos_inc=1.0, opening_angle_deg=40.0, ebv=0.3
        )
        # Higher ebv => lower flux (at UV wavelengths especially)
        uv_mask = WAVELENGTH < 3000.0
        assert jnp.all(l_atten_high[uv_mask] <= l_atten_low[uv_mask] + 1e-10)

    def test_wavelength_dependence(self):
        """UV should be more attenuated than IR (SMC curve steep in UV)."""
        l_atten, _ = polar_dust_extinction(
            L_NU_FLAT, WAVELENGTH, cos_inc=1.0, opening_angle_deg=40.0, ebv=0.3
        )
        # Ratio of attenuated / input
        ratio = l_atten / L_NU_FLAT
        uv_mean = jnp.mean(ratio[WAVELENGTH < 2000.0])
        ir_mean = jnp.mean(ratio[WAVELENGTH > 1e4])
        assert uv_mean < ir_mean


class TestPolarDustEmission:
    """Tests for polar_dust_emission."""

    def test_energy_conservation(self):
        """Integral of reemission should equal absorbed luminosity."""
        l_absorbed_total = 1e12  # arbitrary
        l_reemit = polar_dust_emission(l_absorbed_total, WAVELENGTH, temperature=100.0)

        # Integrate over frequency
        from tengri.models.agn.polar_dust import _C_AA

        nu = _C_AA / WAVELENGTH
        delta_nu = jnp.abs(jnp.diff(nu))
        delta_nu = jnp.concatenate(
            [delta_nu[:1], 0.5 * (delta_nu[:-1] + delta_nu[1:]), delta_nu[-1:]]
        )
        integral = jnp.sum(l_reemit * delta_nu)

        assert jnp.abs(integral - l_absorbed_total) / l_absorbed_total < 0.01

    def test_greybody_peaks_in_fir(self):
        """At T=100K, peak should be around 30 um = 3e5 A."""
        l_reemit = polar_dust_emission(1e12, WAVELENGTH, temperature=100.0)
        peak_idx = jnp.argmax(l_reemit)
        peak_wave = WAVELENGTH[peak_idx]
        # Wien's law for modified blackbody: peak ~ 29 um for T=100K
        # Allow factor-of-3 tolerance due to greybody modification
        assert 1e5 < peak_wave < 1e6, f"Peak at {peak_wave:.0f} A, expected ~3e5 A"

    def test_reemission_shape_is_greybody(self):
        """Longward of peak, emission should monotonically decrease."""
        l_reemit = polar_dust_emission(1e12, WAVELENGTH, temperature=100.0)
        peak_idx = int(jnp.argmax(l_reemit))
        # Beyond the peak (longer wavelengths = higher indices in log-spaced grid)
        longward = l_reemit[peak_idx:]
        if len(longward) > 5:
            # Check that emission generally decreases (allow small numerical wiggles)
            diffs = jnp.diff(longward)
            # Most differences should be negative (decreasing)
            frac_decreasing = jnp.sum(diffs < 0) / len(diffs)
            assert frac_decreasing > 0.9


class TestPolarDustTotal:
    """Tests for polar_dust_total convenience function."""

    def test_polar_dust_total_returns_two_arrays(self):
        """Output should be a tuple of two arrays with correct shapes."""
        l_atten, l_reemit = polar_dust_total(
            L_NU_DISC,
            WAVELENGTH,
            cos_inc=1.0,
            opening_angle_deg=40.0,
            ebv=0.2,
            temperature=100.0,
        )
        assert l_atten.shape == WAVELENGTH.shape
        assert l_reemit.shape == WAVELENGTH.shape

    def test_jit_compatible(self):
        """polar_dust_total should work under jax.jit."""
        jitted = jax.jit(polar_dust_total, static_argnames=("law",))
        l_atten, l_reemit = jitted(
            L_NU_DISC,
            WAVELENGTH,
            cos_inc=1.0,
            opening_angle_deg=40.0,
            ebv=0.2,
            temperature=100.0,
        )
        assert l_atten.shape == WAVELENGTH.shape
        assert l_reemit.shape == WAVELENGTH.shape
        assert jnp.all(jnp.isfinite(l_atten))
        assert jnp.all(jnp.isfinite(l_reemit))

    def test_gradient_flows(self):
        """Gradients should flow through ebv, cos_inc, and temperature."""

        def loss_ebv(ebv):
            l_a, l_r = polar_dust_total(L_NU_DISC, WAVELENGTH, 1.0, 40.0, ebv, temperature=100.0)
            return jnp.sum(l_a) + jnp.sum(l_r)

        def loss_cos_inc(cos_inc):
            l_a, l_r = polar_dust_total(
                L_NU_DISC, WAVELENGTH, cos_inc, 40.0, 0.2, temperature=100.0
            )
            return jnp.sum(l_a) + jnp.sum(l_r)

        def loss_temp(temperature):
            l_a, l_r = polar_dust_total(
                L_NU_DISC, WAVELENGTH, 1.0, 40.0, 0.2, temperature=temperature
            )
            return jnp.sum(l_a) + jnp.sum(l_r)

        grad_ebv = jax.grad(loss_ebv)(0.2)
        grad_cos = jax.grad(loss_cos_inc)(0.8)
        grad_temp = jax.grad(loss_temp)(100.0)

        assert jnp.isfinite(grad_ebv)
        assert jnp.isfinite(grad_cos)
        assert jnp.isfinite(grad_temp)
        # ebv gradient should be non-zero (more ebv = more extinction)
        assert jnp.abs(grad_ebv) > 0.0
