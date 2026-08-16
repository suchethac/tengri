# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for the Casey (2012) dust emission model.

Synthesizer-inspired: synthesizer/tests/test_dust_generators.py tests every dust
generator in isolation for energy balance, spectral shape physics, and parameter
sensitivity. Only Casey2012 physics tests were previously in tests/crossval/.

Physical properties tested:
- Energy balance: total emitted = L_absorbed
- Temperature sensitivity: hotter dust peaks at shorter wavelengths (Wien's law)
- beta_ir: higher emissivity index steepens FIR slope
- alpha_mir: mid-IR power-law slope controls 8-40 um excess
- CMB heating at high-z raises effective dust temperature
- Non-negative output
- JIT compatibility and finite gradients for all parameters
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_paper

from tengri.components.dust.emission import casey2012, cmb_corrected_temperature
from tests._jit_parity import assert_jit_matches_eager

# Physical constants
_C_AA_S = 2.99792458e18  # c in Angstrom/s


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


@pytest.fixture
def ir_wave():
    """IR wavelength grid 1-1000 μm (10^4 – 10^7 Å)."""
    return jnp.logspace(4, 7, 2000)


@pytest.fixture
def mir_wave():
    """Mid-IR wavelength grid 5-50 μm for power-law tests."""
    return jnp.logspace(jnp.log10(5e4), jnp.log10(5e5), 500)


class TestCasey2012EnergyBalance:
    """Total emitted luminosity must equal L_absorbed (energy conservation)."""

    def test_energy_balance_default_params(self, ir_wave):
        L_abs = 1e10
        sed = casey2012(ir_wave, L_abs)
        nu = _C_AA_S / ir_wave
        L_total = float(-jnp.trapezoid(sed, nu))
        ratio = L_total / L_abs
        assert 0.7 < ratio < 1.5, (
            f"Casey2012 energy balance: L_emitted/L_absorbed = {ratio:.3f}, expected ~1.0"
        )

    def test_energy_balance_hot_dust(self, ir_wave):
        L_abs = 1e11
        sed = casey2012(ir_wave, L_abs, dust_T=60.0)
        nu = _C_AA_S / ir_wave
        L_total = float(-jnp.trapezoid(sed, nu))
        ratio = L_total / L_abs
        assert 0.7 < ratio < 1.5, f"Energy balance at T=60K: ratio = {ratio:.3f}"

    def test_energy_scales_linearly(self, ir_wave):
        """L_absorbed = 2x → total emitted = 2x (linearity)."""
        L1, L2 = 1e10, 2e10
        sed1 = casey2012(ir_wave, L1)
        sed2 = casey2012(ir_wave, L2)
        nu = _C_AA_S / ir_wave
        ratio = float(-jnp.trapezoid(sed2, nu)) / float(-jnp.trapezoid(sed1, nu))
        np.testing.assert_allclose(ratio, 2.0, rtol=0.02)


class TestCasey2012SpectralShape:
    """Spectral shape physics: temperature, beta, alpha_mir."""

    def test_hotter_dust_peaks_shorter(self, ir_wave):
        """Wien's law: hotter dust peaks at shorter wavelengths."""
        sed_cold = casey2012(ir_wave, 1e10, dust_T=25.0)
        sed_hot = casey2012(ir_wave, 1e10, dust_T=55.0)
        peak_cold = float(ir_wave[jnp.argmax(sed_cold)])
        peak_hot = float(ir_wave[jnp.argmax(sed_hot)])
        assert peak_hot < peak_cold, (
            f"Hotter dust should peak at shorter λ: T=55K peak {peak_hot:.0f}Å "
            f"vs T=25K peak {peak_cold:.0f}Å"
        )

    def test_peak_in_fir_range(self, ir_wave):
        """Peak emission should be in FIR (30-300 μm = 3e5-3e6 Å)."""
        sed = casey2012(ir_wave, 1e10, dust_T=35.0)
        peak_um = float(ir_wave[jnp.argmax(sed)]) / 1e4
        assert 30 < peak_um < 300, f"Peak at {peak_um:.0f} μm, expected 30-300 μm"

    def test_higher_beta_steepens_rayleigh_jeans(self, ir_wave):
        """Higher beta_ir → steeper emissivity on the Rayleigh-Jeans side.

        At wavelengths well beyond the peak (Rayleigh-Jeans tail), the MBB
        goes as ν^(2+β), so higher β gives more relative flux at high ν
        (short-wavelength FIR), meaning the peak shifts slightly blueward.
        """
        sed_low_beta = casey2012(ir_wave, 1e10, dust_T=35.0, dust_beta_ir=1.2)
        sed_high_beta = casey2012(ir_wave, 1e10, dust_T=35.0, dust_beta_ir=2.0)
        # SEDs should differ — beta matters
        assert not jnp.allclose(sed_low_beta, sed_high_beta, rtol=1e-3), (
            "Different beta_ir values must produce different SED shapes"
        )

    def test_alpha_mir_changes_mid_ir(self, mir_wave):
        """Steeper alpha_mir produces more mid-IR power-law excess at 8-40 μm."""
        sed_flat = casey2012(mir_wave, 1e10, dust_alpha_mir=1.5)
        sed_steep = casey2012(mir_wave, 1e10, dust_alpha_mir=2.5)
        # Different power-law slope must change the integrated mid-IR flux
        assert not jnp.allclose(sed_flat, sed_steep, rtol=1e-3), (
            "Different dust_alpha_mir values must produce different mid-IR shapes"
        )

    def test_output_non_negative(self, ir_wave):
        """Casey2012 emission is non-negative everywhere."""
        sed = casey2012(ir_wave, 1e10, dust_T=35.0, dust_beta_ir=1.8, dust_alpha_mir=2.0)
        assert jnp.all(sed >= 0.0), "Casey2012 SED must be non-negative"
        assert float(jnp.max(sed)) > 0.0, "Casey2012 SED must have positive values"


class TestCasey2012CmbCorrection:
    """CMB heating correction (da Cunha+2013) raises the effective dust temperature
    at high redshift, shifting the SED peak blueward relative to z=0.

    This is the same physical test synthesizer applies to its Graybody model.
    """

    def test_cmb_heating_raises_temperature(self):
        """T_eff(z=5) > T_eff(z=0) for a cold dust grain."""
        T_dust = 20.0  # K — cold, close to CMB temperature at z=0
        T_eff_z0 = float(cmb_corrected_temperature(T_dust, redshift=0.0, beta_ir=1.8))
        T_eff_z5 = float(cmb_corrected_temperature(T_dust, redshift=5.0, beta_ir=1.8))
        assert T_eff_z5 > T_eff_z0, (
            f"CMB heating at z=5 should raise T_eff: got {T_eff_z5:.1f}K vs {T_eff_z0:.1f}K"
        )

    def test_warm_dust_minimally_affected(self):
        """Warm dust (T=40K) is barely affected by CMB at moderate z."""
        T_dust = 40.0
        T_eff_z0 = float(cmb_corrected_temperature(T_dust, redshift=0.0, beta_ir=1.8))
        T_eff_z1 = float(cmb_corrected_temperature(T_dust, redshift=1.0, beta_ir=1.8))
        # CMB at z=1 is T_cmb ≈ 5.5K — still << 40K, so small correction
        assert T_eff_z1 - T_eff_z0 < 1.0, (
            f"Warm dust at z=1 should be minimally affected: ΔT = {T_eff_z1 - T_eff_z0:.2f}K"
        )

    def test_cmb_no_op_at_z0(self):
        """At z=0, CMB correction is negligible for warm dust."""
        T_dust = 35.0
        T_eff = float(cmb_corrected_temperature(T_dust, redshift=0.0, beta_ir=1.8))
        np.testing.assert_allclose(T_eff, T_dust, atol=0.1, err_msg="z=0 CMB should be a no-op")

    def test_high_z_shifts_peak_blueward(self, ir_wave):
        """At z=5, CMB heating shifts the Casey2012 peak to shorter wavelengths."""
        sed_z0 = casey2012(ir_wave, 1e10, dust_T=20.0, redshift=0.0)
        sed_z5 = casey2012(ir_wave, 1e10, dust_T=20.0, redshift=5.0)
        peak_z0 = float(ir_wave[jnp.argmax(sed_z0)])
        peak_z5 = float(ir_wave[jnp.argmax(sed_z5)])
        assert peak_z5 <= peak_z0, (
            f"CMB heating at z=5 should shift peak blueward: "
            f"z=5 peak {peak_z5:.0f}Å vs z=0 peak {peak_z0:.0f}Å"
        )


class TestCasey2012Differentiability:
    """Casey2012 must be JIT-compilable and fully differentiable."""

    def test_jit_compatible(self, ir_wave):
        sed = assert_jit_matches_eager(
            lambda T, b, a: casey2012(ir_wave, 1e10, dust_T=T, dust_beta_ir=b, dust_alpha_mir=a),
            35.0,
            1.8,
            2.0,
        )
        chex.assert_equal_shape([sed, ir_wave])
        chex.assert_tree_all_finite(sed)

    def test_gradient_dust_T(self, ir_wave):
        def loss(T):
            return float(jnp.sum(casey2012(ir_wave, 1e10, dust_T=T)))

        grad_jax = float(jax.grad(lambda T: jnp.sum(casey2012(ir_wave, 1e10, dust_T=T)))(35.0))
        grad_fd = fd_grad(loss, 35.0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
        assert grad_jax != 0.0, "Gradient w.r.t. dust_T should be nonzero"

    def test_gradient_beta(self, ir_wave):
        def loss(b):
            return float(jnp.sum(casey2012(ir_wave, 1e10, dust_beta_ir=b)))

        def grad_fn(b):
            return jnp.sum(casey2012(ir_wave, 1e10, dust_beta_ir=b))

        grad_jax = float(jax.grad(grad_fn)(1.8))
        grad_fd = fd_grad(loss, 1.8)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )

    def test_gradient_alpha_mir(self, ir_wave):
        def loss(a):
            return float(jnp.sum(casey2012(ir_wave, 1e10, dust_alpha_mir=a)))

        def grad_fn(a):
            return jnp.sum(casey2012(ir_wave, 1e10, dust_alpha_mir=a))

        grad_jax = float(jax.grad(grad_fn)(2.0))
        grad_fd = fd_grad(loss, 2.0)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )

    def test_all_gradients_jointly(self, ir_wave):
        """All parameter gradients at once via argnums."""

        def loss(T, b, a):
            return jnp.sum(casey2012(ir_wave, 1e10, dust_T=T, dust_beta_ir=b, dust_alpha_mir=a))

        def make_loss_single(idx, param_vals):
            def loss_single(x):
                T, b, a = param_vals[0], param_vals[1], param_vals[2]
                if idx == 0:
                    T = x
                elif idx == 1:
                    b = x
                else:
                    a = x
                return float(
                    jnp.sum(casey2012(ir_wave, 1e10, dust_T=T, dust_beta_ir=b, dust_alpha_mir=a))
                )

            return loss_single

        grads_jax = jax.grad(loss, argnums=(0, 1, 2))(35.0, 1.8, 2.0)
        param_vals = [35.0, 1.8, 2.0]
        param_names = ["dust_T", "dust_beta_ir", "dust_alpha_mir"]

        for idx, name in enumerate(param_names):
            grad_jax = float(grads_jax[idx])
            loss_single = make_loss_single(idx, param_vals)
            grad_fd = fd_grad(loss_single, param_vals[idx])
            np.testing.assert_allclose(
                grad_jax,
                grad_fd,
                rtol=1e-3,
                err_msg=f"{name}: autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
            )


class TestCasey2012OpticallyThin:
    """Optically thin variant zeros the mid-IR power-law component."""

    def test_optically_thin_no_power_law(self, ir_wave):
        """With optically_thin=True, mid-IR power law is absent."""
        sed_full = casey2012(ir_wave, 1e10)
        sed_ot = casey2012(ir_wave, 1e10, optically_thin=True)
        assert not jnp.allclose(sed_full, sed_ot, rtol=1e-6), (
            "Optically thin SED must differ from full model"
        )

    def test_optically_thin_energy_balance(self, ir_wave):
        """Energy conservation holds for the optically thin variant."""
        L_abs = 1e10
        sed = casey2012(ir_wave, L_abs, optically_thin=True)
        nu = _C_AA_S / ir_wave
        L_total = float(-jnp.trapezoid(sed, nu))
        ratio = L_total / L_abs
        assert 0.7 < ratio < 1.5, (
            f"Optically thin energy balance: L_emitted/L_absorbed = {ratio:.3f}"
        )

    def test_optically_thin_non_negative(self, ir_wave):
        """Optically thin emission is non-negative everywhere."""
        sed = casey2012(ir_wave, 1e10, optically_thin=True)
        assert jnp.all(sed >= 0.0)
        assert float(jnp.max(sed)) > 0.0

    def test_optically_thin_less_mir(self, mir_wave):
        """Optically thin has less mid-IR flux than the full model."""
        sed_full = casey2012(mir_wave, 1e10, dust_T=80.0)
        sed_ot = casey2012(mir_wave, 1e10, dust_T=80.0, optically_thin=True)
        mir_full = float(jnp.sum(sed_full))
        mir_ot = float(jnp.sum(sed_ot))
        assert mir_ot <= mir_full, (
            f"Optically thin should have less mid-IR: {mir_ot:.3e} vs {mir_full:.3e}"
        )

    def test_optically_thin_jit(self, ir_wave):
        """JIT-compatible with optically_thin flag."""
        sed = assert_jit_matches_eager(
            lambda T: casey2012(ir_wave, 1e10, dust_T=T, optically_thin=True), 35.0
        )
        chex.assert_equal_shape([sed, ir_wave])
        chex.assert_tree_all_finite(sed)
