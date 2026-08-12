# SPDX-License-Identifier: BSD-3-Clause
"""Tests for polar dust extinction and graybody reemission."""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn.polar_dust import (
    _type1_mask,
    polar_dust_emission,
    polar_dust_extinction,
    polar_dust_total,
)
from tengri.utils.physics_constants import C_AA as _C_AA


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""

    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


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

    def test_no_disc_attenuation_for_type2(self):
        """Edge-on (cos_inc=0) => observer-frame disc is unattenuated, but
        polar-dust absorption is still bi-conical (Yang+2020 §2.2.2)."""
        l_atten, l_abs = polar_dust_extinction(
            L_NU_DISC, WAVELENGTH, cos_inc=0.0, opening_angle_deg=40.0, ebv=0.3
        )
        # Type 2: mask ~ 0 → observed disc unattenuated.
        assert jnp.allclose(l_atten, L_NU_DISC, rtol=1e-4)
        # But the bi-conical polar dust still absorbs disc photons → nonzero
        # l_abs drives the isotropic FIR re-emission visible from any angle.
        assert jnp.sum(l_abs) > 0.0

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
        from tengri.components.agn.polar_dust import _C_AA

        nu = _C_AA / WAVELENGTH
        delta_nu = jnp.abs(jnp.diff(nu))
        delta_nu = jnp.concatenate(
            [delta_nu[:1], 0.5 * (delta_nu[:-1] + delta_nu[1:]), delta_nu[-1:]]
        )
        integral = jnp.sum(l_reemit * delta_nu)
        assert jnp.abs(integral - l_absorbed_total) / l_absorbed_total < 0.01

    def test_graybody_peaks_in_fir(self):
        """At T=100K, peak should be around 30 um = 3e5 A."""
        l_reemit = polar_dust_emission(1e12, WAVELENGTH, temperature=100.0)
        peak_idx = jnp.argmax(l_reemit)
        peak_wave = WAVELENGTH[peak_idx]
        # Wien's law for modified blackbody: peak ~ 29 um for T=100K
        # Allow factor-of-3 tolerance due to graybody modification
        assert 1e5 < peak_wave < 1e6, f"Peak at {peak_wave:.0f} A, expected ~3e5 A"

    def test_reemission_shape_is_graybody(self):
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
        chex.assert_equal_shape([l_atten, WAVELENGTH])
        chex.assert_equal_shape([l_reemit, WAVELENGTH])

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
        chex.assert_equal_shape([l_atten, WAVELENGTH])
        chex.assert_equal_shape([l_reemit, WAVELENGTH])
        chex.assert_tree_all_finite(l_atten)
        chex.assert_tree_all_finite(l_reemit)

    def test_gradient_flows(self):
        """Gradients should flow through ebv, cos_inc, and temperature."""

        def loss_ebv(ebv):
            l_a, l_r = polar_dust_total(L_NU_DISC, WAVELENGTH, 1.0, 40.0, ebv, temperature=100.0)
            return float(jnp.sum(l_a) + jnp.sum(l_r))

        def loss_cos_inc(cos_inc):
            l_a, l_r = polar_dust_total(
                L_NU_DISC, WAVELENGTH, cos_inc, 40.0, 0.2, temperature=100.0
            )
            return float(jnp.sum(l_a) + jnp.sum(l_r))

        def loss_temp(temperature):
            l_a, l_r = polar_dust_total(
                L_NU_DISC, WAVELENGTH, 1.0, 40.0, 0.2, temperature=temperature
            )
            return float(jnp.sum(l_a) + jnp.sum(l_r))

        def grad_fn_ebv(ebv):
            l_a, l_r = polar_dust_total(L_NU_DISC, WAVELENGTH, 1.0, 40.0, ebv, temperature=100.0)
            return jnp.sum(l_a) + jnp.sum(l_r)

        grad_ebv_jax = float(jax.grad(grad_fn_ebv)(0.2))
        grad_ebv_fd = fd_grad(loss_ebv, 0.2)
        np.testing.assert_allclose(
            grad_ebv_jax,
            grad_ebv_fd,
            rtol=1e-3,
            err_msg=f"ebv: autodiff={grad_ebv_jax:.4e}, FD={grad_ebv_fd:.4e}",
        )

        def grad_fn_cos(cos_inc):
            l_a, l_r = polar_dust_total(
                L_NU_DISC, WAVELENGTH, cos_inc, 40.0, 0.2, temperature=100.0
            )
            return jnp.sum(l_a) + jnp.sum(l_r)

        grad_cos_jax = float(jax.grad(grad_fn_cos)(0.8))
        grad_cos_fd = fd_grad(loss_cos_inc, 0.8)
        np.testing.assert_allclose(
            grad_cos_jax,
            grad_cos_fd,
            rtol=1e-3,
            err_msg=f"cos_inc: autodiff={grad_cos_jax:.4e}, FD={grad_cos_fd:.4e}",
        )

        def grad_fn_temp(temperature):
            l_a, l_r = polar_dust_total(
                L_NU_DISC, WAVELENGTH, 1.0, 40.0, 0.2, temperature=temperature
            )
            return jnp.sum(l_a) + jnp.sum(l_r)

        grad_temp_jax = float(jax.grad(grad_fn_temp)(100.0))
        grad_temp_fd = fd_grad(loss_temp, 100.0)
        np.testing.assert_allclose(
            grad_temp_jax,
            grad_temp_fd,
            rtol=1e-3,
            err_msg=f"temperature: autodiff={grad_temp_jax:.4e}, FD={grad_temp_fd:.4e}",
        )
        # ebv gradient should be non-zero (more ebv = more extinction)
        assert abs(grad_ebv_jax) > 0.0


class TestSKIRTORPolarDustIntegration:
    """Integration tests for SKIRTOR + polar dust reemission."""

    @pytest.fixture
    def wave_test(self):
        """Test wavelength grid from 1000 Å to 1e7 Å (50 points logspaced)."""
        return jnp.logspace(jnp.log10(1000.0), jnp.log10(1e7), 50)

    def test_skirtor_polar_dust_type2_reemission(self, wave_test):
        """SKIRTOR Type 2 polar dust reemission increases with E(B-V).

        Test the SKIRTORTorus SEDModelComponent with polar dust parameters
        as free variables. Type 2 (edge-on, cos_inc=0.0) sightline should
        show increased FIR luminosity when polar_ebv=0.3 vs 0.0, due to
        graybody reemission of absorbed UV/optical photons.

        This validates Yang+2020 §2.2.2: isotropic absorption geometry
        for polar dust, regardless of viewing angle.

        References
        ----------
        .. [1] Yang, A., et al. 2020, MNRAS, 491, 740.
        """
        pytest.importorskip("h5py")
        from pathlib import Path

        # Check for SKIRTOR template file
        skirtor_template = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "skirtor_templates_v3.h5"
        )
        if not skirtor_template.is_file():
            pytest.skip(f"SKIRTOR template not found: {skirtor_template}")

        try:
            from tengri.components.agn.skirtor import SKIRTORTorus
        except Exception as e:
            pytest.skip(f"Cannot load SKIRTOR: {e}")

        # Create SKIRTOR torus component
        torus = SKIRTORTorus()
        torus.load(wave_test)

        # Type 2 parameters: edge-on viewing angle, varying polar dust
        params_ebv_0 = {
            "torus_tau": 5.0,
            "torus_p": 0.5,
            "torus_q": 10.0,
            "torus_oa": 30.0,
            "torus_frac": 0.5,
            "polar_ebv": 0.0,
            "polar_temperature": 100.0,
            "polar_beta": 1.6,
        }
        params_ebv_03 = params_ebv_0.copy()
        params_ebv_03["polar_ebv"] = 0.3

        # Apply at Type 2 (cos_inc=0.0)
        sed_ebv_0, _ = torus.predict(
            params_ebv_0, jnp.zeros_like(wave_test), wave_test, agn_cos_inc=0.0
        )
        sed_ebv_03, _ = torus.predict(
            params_ebv_03, jnp.zeros_like(wave_test), wave_test, agn_cos_inc=0.0
        )

        # Integrate FIR luminosity from 10 um to 1000 um (integrate in frequency)
        fir_mask = (wave_test >= 1e5) & (wave_test <= 1e8)  # 10 um to 1000 um
        nu = _C_AA / wave_test
        delta_nu = jnp.abs(jnp.diff(nu))
        delta_nu = jnp.concatenate(
            [delta_nu[:1], 0.5 * (delta_nu[:-1] + delta_nu[1:]), delta_nu[-1:]]
        )

        fir_lum_ebv_0 = jnp.sum(sed_ebv_0[fir_mask] * delta_nu[fir_mask])
        fir_lum_ebv_03 = jnp.sum(sed_ebv_03[fir_mask] * delta_nu[fir_mask])

        # Assert: ebv=0.3 produces more FIR than ebv=0.0 (reemission > 0)
        ratio = fir_lum_ebv_03 / (fir_lum_ebv_0 + 1e-20)
        assert float(ratio) > 1.05, (
            f"FIR luminosity did not increase with polar_ebv: "
            f"ratio={float(ratio):.3f}, expected > 1.05. "
            f"FIR(ebv=0.0)={float(fir_lum_ebv_0):.3e}, "
            f"FIR(ebv=0.3)={float(fir_lum_ebv_03):.3e}"
        )

    def test_composable_polar_dust_type2_reemission(self, wave_test):
        """Composable AGN runner with polar_dust block increases FIR with E(B-V).

        Call the composable_agn_l_nu runner with agn_attenuation_block="polar_dust"
        and agn_torus_block="skirtor", comparing Type 2 sightlines (cos_inc=0.0)
        with polar_ebv in {0.0, 0.3}.

        FIR luminosity should increase strictly by >5% when polar_ebv=0.3
        vs polar_ebv=0.0, due to isotropic graybody reemission.

        References
        ----------
        .. [1] Yang, A., et al. 2020, MNRAS, 491, 740.
        """
        pytest.importorskip("h5py")
        from pathlib import Path

        # Check for SKIRTOR template
        skirtor_template = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "data"
            / "skirtor_templates_v3.h5"
        )
        if not skirtor_template.is_file():
            pytest.skip(f"SKIRTOR template not found: {skirtor_template}")

        try:
            from tengri.components.agn.blocks.runner import composable_agn_l_nu
        except Exception as e:
            pytest.skip(f"Cannot load composable runner: {e}")

        # Call composable runner twice with different polar_ebv, both Type 2
        common_params = {
            "agn_log_lbol": 12.0,
            "agn_lum_ratio": 1.0,
            "agn_attenuation_block": "polar_dust",
            "agn_torus_block": "skirtor",
            "agn_disc_block": "multicolor",
            "agn_cos_inc": 0.0,  # Type 2 (edge-on)
            "agn_polar_temperature": 100.0,
            "agn_polar_beta": 1.6,
            "agn_polar_oa": 45.0,
            "agn_polar_law": "smc",
            # SKIRTOR torus params (all fixed for simplicity)
            "agn_torus_tau": 5.0,
            "agn_torus_p": 0.5,
            "agn_torus_q": 10.0,
            "agn_torus_oa": 30.0,
            "agn_torus_frac": 0.5,
        }

        params_ebv_0 = common_params.copy()
        params_ebv_0["agn_polar_ebv"] = 0.0

        params_ebv_03 = common_params.copy()
        params_ebv_03["agn_polar_ebv"] = 0.3

        sed_ebv_0 = composable_agn_l_nu(wave_test, **params_ebv_0)
        sed_ebv_03 = composable_agn_l_nu(wave_test, **params_ebv_03)

        chex.assert_equal_shape([sed_ebv_0, wave_test])
        chex.assert_equal_shape([sed_ebv_03, wave_test])
        chex.assert_tree_all_finite(sed_ebv_0)
        chex.assert_tree_all_finite(sed_ebv_03)

        # Integrate FIR from 10 um to 1000 um
        fir_mask = (wave_test >= 1e5) & (wave_test <= 1e8)
        nu = _C_AA / wave_test
        delta_nu = jnp.abs(jnp.diff(nu))
        delta_nu = jnp.concatenate(
            [delta_nu[:1], 0.5 * (delta_nu[:-1] + delta_nu[1:]), delta_nu[-1:]]
        )

        fir_lum_ebv_0 = jnp.sum(sed_ebv_0[fir_mask] * delta_nu[fir_mask])
        fir_lum_ebv_03 = jnp.sum(sed_ebv_03[fir_mask] * delta_nu[fir_mask])

        ratio = fir_lum_ebv_03 / (fir_lum_ebv_0 + 1e-20)
        assert float(ratio) > 1.05, (
            f"Composable runner: FIR luminosity did not increase with polar_ebv. "
            f"ratio={float(ratio):.3f}, expected > 1.05. "
            f"FIR(ebv=0.0)={float(fir_lum_ebv_0):.3e}, "
            f"FIR(ebv=0.3)={float(fir_lum_ebv_03):.3e}"
        )
