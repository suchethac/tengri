# SPDX-License-Identifier: BSD-3-Clause
"""Tests for the three-mode SFR radio physics models.

Covers:
- radio_sfr_bell2003 (backward-compat alias for radio_star_forming)
- radio_sfr_delvecchio2021 (mass+z dependent FIRRC at 1.4 GHz)
- radio_sfr_mccheyne2022  (mass+z dependent FIRRC at 150 MHz)
- _synchrotron_suppression (Bell+2003 helper)
- radio_total dispatcher (sfr_mode selection)
- radio_freefree (Murphy+2011 thermal bremsstrahlung)
- compute_radio_components (component decomposition)
"""

import chex
import pytest

pytestmark = pytest.mark.bounds
import jax
import jax.numpy as jnp
import numpy as np


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite-difference gradient. O(eps^2) accurate."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


from tengri.components.radio.radio import (
    _L0_SYNCH,
    _synchrotron_suppression,
    compute_radio_components,
    radio_freefree,
    radio_sfr_bell2003,
    radio_sfr_delvecchio2021,
    radio_sfr_mccheyne2022,
    radio_star_forming,
    radio_total,
    radio_total_dpl,
)

_C_AA = 2.99792458e18  # Angstrom/s
# Wavelengths: 10 MHz – 300 GHz, all safely in the radio band (> 1e7 A)
_WAVE_RADIO = _C_AA / jnp.logspace(7.0, 11.0, 200)  # Angstrom
# Convenient single-frequency arrays
_WAVE_14GHZ = jnp.array([_C_AA / 1.4e9])
_WAVE_150MHZ = jnp.array([_C_AA / 1.5e8])
_L_IR = 1e11  # Lsun, typical LIRG


# ── Backward-compatibility ────────────────────────────────────────
class TestRadioConstantsCGS:
    """Regression: radio constants must be in CGS (erg/s/Hz), not Lsun/Hz."""

    def test_l0_synch_cgs_value(self):
        """_L0_SYNCH must be 3.0e28 erg/s/Hz (Bell 2003 synchrotron turnover).
        Previously named _L0_SYNCH_LSUN_HZ and in Lsun/Hz. Renamed to _L0_SYNCH
        in erg/s/Hz as part of the CGS standardization (2026-04-08).
        """
        assert abs(_L0_SYNCH - 3.0e28) < 1e26, (
            f"_L0_SYNCH = {_L0_SYNCH:.2e}, expected 3.0e28 erg/s/Hz"
        )


class TestBackwardCompat:
    """radio_sfr_bell2003 must be identical to the old radio_star_forming."""

    def test_alias_identical_output(self):
        """radio_sfr_bell2003 == radio_star_forming for identical inputs."""
        L_old = radio_star_forming(_WAVE_RADIO, _L_IR)
        L_new = radio_sfr_bell2003(_WAVE_RADIO, _L_IR)
        assert jnp.allclose(L_old, L_new, rtol=0.0, atol=0.0)

    def test_dispatcher_bell2003_matches_direct_call(self):
        """radio_total(sfr_mode='bell2003', no ff, no AGN) matches radio_sfr_bell2003."""
        L_direct = radio_sfr_bell2003(_WAVE_RADIO, _L_IR)
        L_dispatch = radio_total(_WAVE_RADIO, L_ir=_L_IR, L_agn_bol=0.0, include_freefree=False)
        assert jnp.allclose(L_direct, L_dispatch, rtol=1e-12)

    def test_invalid_sfr_mode_raises(self):
        """Unknown sfr_mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown sfr_mode"):
            radio_total(_WAVE_RADIO, L_ir=_L_IR, sfr_mode="bogus")


# ── Bell+2003 synchrotron suppression ─────────────────────────────
class TestSynchrotronSuppression:
    """Bell+2003 suppression: piecewise power-law n(L) per Bell (2003) ApJ 586, 794 Eq. 3.
    n(L) = 0.9 for L >= L*; n(L) = 0.9*(L/L*)^0.3 for L < L*. L_corr = n(L)*L.
    Replaces the old quadratic formula L/(1+(L0/L)^2) which had the wrong index.
    """

    def test_bright_source_unchanged(self):
        """At L >> L*, non-thermal fraction n ≈ 0.9 (Bell 2003 Eq. 3 plateau)."""
        L_bright = jnp.array(1e6 * _L0_SYNCH)  # 1e6 × L*
        L_corr = _synchrotron_suppression(L_bright)
        ratio = float(L_corr / L_bright)
        assert abs(ratio - 0.9) < 1e-6, f"Bright source n = {ratio:.6f}, expected 0.9"

    def test_faint_source_suppressed(self):
        """At L << L*, n = 0.9*(L/L*)^0.3 (Bell 2003 Eq. 3 power-law)."""
        L_faint = jnp.array(1e-6 * _L0_SYNCH)  # 1e-6 × L*
        L_corr = _synchrotron_suppression(L_faint)
        # Expected: L_corr = 0.9 * (1e-6)^0.3 * L_faint
        n_expected = 0.9 * (1e-6) ** 0.3
        expected = n_expected * float(L_faint)
        assert abs(float(L_corr) - expected) / expected < 0.02, (
            f"Faint source L_corr {float(L_corr):.4e} != expected {expected:.4e}"
        )

    def test_suppression_monotonic(self):
        """Non-thermal fraction n(L) increases monotonically with L."""
        L_vals = jnp.logspace(-10, 2, 100) * _L0_SYNCH
        L_corr = _synchrotron_suppression(L_vals)
        factors = L_corr / L_vals
        # All factors should be <= 0.9 and non-decreasing (tolerance for float noise)
        assert jnp.all(factors <= 0.9 + 1e-12), "n(L) must never exceed 0.9"
        assert jnp.all(jnp.diff(factors) >= -1e-12), "n(L) should increase monotonically"

    def test_suppression_at_zero_safe(self):
        """L=0 should not produce NaN."""
        L_corr = _synchrotron_suppression(jnp.array(0.0))
        assert jnp.isfinite(L_corr), "Suppression at L=0 must not be NaN/Inf"


# ── Delvecchio+2021 model ─────────────────────────────────────────
class TestDelvecchio2021:
    """Tests for radio_sfr_delvecchio2021."""

    def test_q_at_fiducial_mass_z0(self):
        """At log(M★)=10, z=0: q = q0 × 1^z_slope - 0 = 2.743."""
        # L_1.4GHz = L_IR / (3.75e12 × 10^q)
        # At log(M★)=10, z=0: q = 2.743
        expected_q = 2.743
        L_ref_expected = _L_IR / (3.75e12 * 10.0**expected_q)
        L = radio_sfr_delvecchio2021(
            _WAVE_14GHZ,
            _L_IR,
            log_mstar=10.0,
            redshift=0.0,
            apply_suppression=False,
        )
        # L at 1.4 GHz (nu_ref) equals L_ref_expected: (nu/nu_ref)^-alpha = 1 at nu_ref
        L_ref_computed = float(L[0])
        assert abs(L_ref_computed - L_ref_expected) / L_ref_expected < 1e-6, (
            f"q at fiducial: computed L_ref {L_ref_computed:.4e} != expected {L_ref_expected:.4e}"
        )

    def test_massive_galaxy_more_radio(self):
        """Higher M★ → lower q → more radio luminosity per unit L_IR."""
        L_low_mass = radio_sfr_delvecchio2021(
            _WAVE_14GHZ, _L_IR, log_mstar=9.0, redshift=0.0, apply_suppression=False
        )
        L_high_mass = radio_sfr_delvecchio2021(
            _WAVE_14GHZ, _L_IR, log_mstar=11.0, redshift=0.0, apply_suppression=False
        )
        assert float(L_high_mass[0]) > float(L_low_mass[0]), (
            "Massive galaxy should be more radio-bright per unit L_IR"
        )

    def test_q_mass_dependence_correct_magnitude(self):
        """Δq over 2 dex in M★ should be 2 × mass_slope = 0.468."""
        L_m10 = radio_sfr_delvecchio2021(
            _WAVE_14GHZ, _L_IR, log_mstar=10.0, redshift=0.0, apply_suppression=False
        )
        L_m12 = radio_sfr_delvecchio2021(
            _WAVE_14GHZ, _L_IR, log_mstar=12.0, redshift=0.0, apply_suppression=False
        )
        # L ∝ 10^(-q), so Δlog(L) = Δq = 2 × 0.234 = 0.468
        delta_log_L = jnp.log10(L_m12[0]) - jnp.log10(L_m10[0])
        assert abs(float(delta_log_L) - 0.468) < 0.01, (
            f"Δlog(L) = {float(delta_log_L):.4f} expected 0.468"
        )

    def test_z_evolution_mild(self):
        """z_slope = -0.025: q decreases mildly from z=0 to z=4."""
        L_z0 = radio_sfr_delvecchio2021(
            _WAVE_14GHZ, _L_IR, log_mstar=10.0, redshift=0.0, apply_suppression=False
        )
        L_z4 = radio_sfr_delvecchio2021(
            _WAVE_14GHZ, _L_IR, log_mstar=10.0, redshift=4.0, apply_suppression=False
        )
        # q(z=4) = 2.743 * 5^(-0.025) ≈ 2.743 * 0.9604 ≈ 2.634
        # Δq ≈ 0.109 → L_z4 / L_z0 ≈ 10^0.109 ≈ 1.285
        ratio = float(L_z4[0] / L_z0[0])
        assert 1.2 < ratio < 1.4, f"z=4 / z=0 radio ratio {ratio:.3f}, expected ~1.28"

    def test_hierarchical_param_q0_increases_L(self):
        """Higher q0 → higher q → lower L_radio (L ∝ 10^{-q})."""
        L_lo = radio_sfr_delvecchio2021(
            _WAVE_14GHZ,
            _L_IR,
            log_mstar=10.0,
            redshift=0.0,
            q0=3.0,
            apply_suppression=False,
        )
        L_hi = radio_sfr_delvecchio2021(
            _WAVE_14GHZ,
            _L_IR,
            log_mstar=10.0,
            redshift=0.0,
            q0=2.0,
            apply_suppression=False,
        )
        assert float(L_hi[0]) > float(L_lo[0]), "Lower q0 should give higher L_radio"

    def test_spectral_index_0p7_default(self):
        """Default alpha_sf=0.7: S(150MHz)/S(1.4GHz) ≈ (150/1400)^{-0.7} ≈ 5.9."""
        L_14ghz = radio_sfr_delvecchio2021(
            _WAVE_14GHZ, _L_IR, log_mstar=10.0, redshift=0.0, apply_suppression=False
        )
        L_150mhz = radio_sfr_delvecchio2021(
            _WAVE_150MHZ, _L_IR, log_mstar=10.0, redshift=0.0, apply_suppression=False
        )
        expected_ratio = (150.0e6 / 1.4e9) ** (-0.7)  # ≈ 5.87
        actual_ratio = float(L_150mhz[0] / L_14ghz[0])
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01, (
            f"150MHz/1.4GHz ratio {actual_ratio:.4f} != expected {expected_ratio:.4f}"
        )

    def test_suppression_reduces_faint_source(self):
        """apply_suppression=True must reduce L for a faint source."""
        L_ir_faint = 1e6  # very low L_IR -> very faint radio
        L_nosuppr = radio_sfr_delvecchio2021(
            _WAVE_14GHZ,
            L_ir_faint,
            log_mstar=8.0,
            redshift=0.0,
            apply_suppression=False,
        )
        L_suppr = radio_sfr_delvecchio2021(
            _WAVE_14GHZ,
            L_ir_faint,
            log_mstar=8.0,
            redshift=0.0,
            apply_suppression=True,
        )
        assert float(L_suppr[0]) < float(L_nosuppr[0]), (
            "Suppression should reduce L for faint sources"
        )

    def test_jit_parity_and_shape(self):
        """JIT eager output match (parity) and output shape.

        Physical: JIT compilation must not change numerical results.
        """
        L_eager = radio_sfr_delvecchio2021(_WAVE_RADIO, _L_IR, 10.0, 0.0, apply_suppression=False)
        jitted = jax.jit(radio_sfr_delvecchio2021, static_argnames=["apply_suppression"])
        L_jit = jitted(_WAVE_RADIO, _L_IR, 10.0, 0.0, apply_suppression=False)
        chex.assert_equal_shape([L_jit, _WAVE_RADIO])
        chex.assert_trees_all_close(L_eager, L_jit, rtol=1e-6)

    def test_gradients_flow_through_log_mstar_and_redshift(self):
        """Gradients should be finite and nonzero w.r.t. log_mstar, redshift."""

        def _loss(log_mstar, redshift):
            return jnp.sum(
                radio_sfr_delvecchio2021(
                    _WAVE_14GHZ, _L_IR, log_mstar, redshift, apply_suppression=False
                )
            )

        g_mstar, g_z = jax.grad(_loss, argnums=(0, 1))(10.0, 0.5)

        def f_mstar(log_mstar: float) -> float:
            return float(_loss(log_mstar, 0.5))

        def f_z(redshift: float) -> float:
            return float(_loss(10.0, redshift))

        np.testing.assert_allclose(
            float(g_mstar),
            fd_grad(f_mstar, 10.0),
            rtol=1e-3,
            err_msg="radio_sfr_delvecchio2021: FD check ∂/∂log_mstar",
        )
        np.testing.assert_allclose(
            float(g_z),
            fd_grad(f_z, 0.5),
            rtol=1e-3,
            err_msg="radio_sfr_delvecchio2021: FD check ∂/∂redshift",
        )


# ── McCheyne+2022 model ───────────────────────────────────────────
class TestMcCheyne2022:
    """Tests for radio_sfr_mccheyne2022."""

    def test_q_at_fiducial_mass_z0(self):
        """At log(M★)=10, z=0: q = q0 + mass_slope × 0 = 1.98."""
        expected_q = 1.98
        L_ref_expected = _L_IR / (3.75e12 * 10.0**expected_q)
        L = radio_sfr_mccheyne2022(
            _WAVE_150MHZ,
            _L_IR,
            log_mstar=10.0,
            redshift=0.0,
            apply_suppression=False,
        )
        L_ref_computed = float(L[0])
        assert abs(L_ref_computed - L_ref_expected) / L_ref_expected < 1e-5, (
            f"L at fiducial {L_ref_computed:.4e} != expected {L_ref_expected:.4e}"
        )

    def test_massive_galaxy_more_radio(self):
        """Higher M★ → lower q (mass_slope=-0.22) → more L_radio."""
        L_low = radio_sfr_mccheyne2022(
            _WAVE_150MHZ, _L_IR, log_mstar=9.0, redshift=0.0, apply_suppression=False
        )
        L_high = radio_sfr_mccheyne2022(
            _WAVE_150MHZ, _L_IR, log_mstar=11.0, redshift=0.0, apply_suppression=False
        )
        assert float(L_high[0]) > float(L_low[0]), (
            "Higher M★ should give more radio emission (mass_slope < 0)"
        )

    def test_mass_slope_magnitude(self):
        """Δq over 2 dex M★ = 2 × |mass_slope| = 0.44."""
        L_m10 = radio_sfr_mccheyne2022(
            _WAVE_150MHZ, _L_IR, log_mstar=10.0, redshift=0.0, apply_suppression=False
        )
        L_m12 = radio_sfr_mccheyne2022(
            _WAVE_150MHZ, _L_IR, log_mstar=12.0, redshift=0.0, apply_suppression=False
        )
        delta_log_L = float(jnp.log10(L_m12[0]) - jnp.log10(L_m10[0]))
        # mass_slope = -0.22, so Δq = -0.22 * 2 = -0.44 → ΔlogL = +0.44
        assert abs(delta_log_L - 0.44) < 0.01, f"Δlog(L) = {delta_log_L:.4f}, expected 0.44"

    def test_hierarchical_param_monotonic(self):
        """q0 override: higher q0 → lower L_radio."""
        L_lo_q0 = radio_sfr_mccheyne2022(
            _WAVE_150MHZ,
            _L_IR,
            log_mstar=10.0,
            redshift=0.0,
            q0=2.5,
            apply_suppression=False,
        )
        L_hi_q0 = radio_sfr_mccheyne2022(
            _WAVE_150MHZ,
            _L_IR,
            log_mstar=10.0,
            redshift=0.0,
            q0=1.5,
            apply_suppression=False,
        )
        assert float(L_hi_q0[0]) > float(L_lo_q0[0]), "Lower q0 → more L_radio"

    def test_spectral_index_0p7_default(self):
        """At 150 MHz reference, extrapolating to 1.4 GHz with α=0.7."""
        L_150mhz = radio_sfr_mccheyne2022(
            _WAVE_150MHZ, _L_IR, log_mstar=10.0, redshift=0.0, apply_suppression=False
        )
        L_14ghz = radio_sfr_mccheyne2022(
            _WAVE_14GHZ, _L_IR, log_mstar=10.0, redshift=0.0, apply_suppression=False
        )
        # At 1.4 GHz: L = L_ref * (1.4e9 / 1.5e8)^{-0.7}
        expected_ratio = (1.5e8 / 1.4e9) ** (-0.7)  # > 1 (150 MHz brighter)
        actual_ratio = float(L_150mhz[0] / L_14ghz[0])
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01, (
            f"150MHz/1.4GHz ratio {actual_ratio:.4f} != expected {expected_ratio:.4f}"
        )

    def test_jit_parity_and_shape(self):
        """JIT eager output match (parity) and output shape."""
        L_eager = radio_sfr_mccheyne2022(_WAVE_RADIO, _L_IR, 10.0, 0.0, apply_suppression=False)
        jitted = jax.jit(radio_sfr_mccheyne2022, static_argnames=["apply_suppression"])
        L_jit = jitted(_WAVE_RADIO, _L_IR, 10.0, 0.0, apply_suppression=False)
        chex.assert_equal_shape([L_jit, _WAVE_RADIO])
        chex.assert_trees_all_close(L_eager, L_jit, rtol=1e-6)

    def test_gradients_flow(self):
        def _loss(log_mstar, redshift):
            return jnp.sum(
                radio_sfr_mccheyne2022(
                    _WAVE_150MHZ, _L_IR, log_mstar, redshift, apply_suppression=False
                )
            )

        g_mstar, g_z = jax.grad(_loss, argnums=(0, 1))(10.0, 0.3)

        def f_mstar(log_mstar: float) -> float:
            return float(_loss(log_mstar, 0.3))

        def f_z(redshift: float) -> float:
            return float(_loss(10.0, redshift))

        np.testing.assert_allclose(
            float(g_mstar),
            fd_grad(f_mstar, 10.0),
            rtol=1e-3,
            err_msg="radio_sfr_mccheyne2022: FD check ∂/∂log_mstar",
        )
        np.testing.assert_allclose(
            float(g_z),
            fd_grad(f_z, 0.3),
            rtol=1e-3,
            err_msg="radio_sfr_mccheyne2022: FD check ∂/∂redshift",
        )


# ── radio_total dispatcher ────────────────────────────────────────
class TestRadioTotalDispatcher:
    """radio_total correctly dispatches to all three SFR modes."""

    def test_bell2003_mode_zero_agn(self):
        """With L_agn_bol=0 and no ff, bell2003 total == radio_sfr_bell2003."""
        L_direct = radio_sfr_bell2003(_WAVE_RADIO, _L_IR)
        L_via_total = radio_total(
            _WAVE_RADIO, L_ir=_L_IR, L_agn_bol=0.0, sfr_mode="bell2003", include_freefree=False
        )
        assert jnp.allclose(L_direct, L_via_total, rtol=1e-12)

    def test_delvecchio2021_mode_dispatches(self):
        """delvecchio2021 mode should return different L than bell2003 (log_mstar != 10)."""
        L_bell = radio_total(
            _WAVE_14GHZ,
            L_ir=_L_IR,
            L_agn_bol=0.0,
            sfr_mode="bell2003",
        )
        L_delv = radio_total(
            _WAVE_14GHZ,
            L_ir=_L_IR,
            L_agn_bol=0.0,
            sfr_mode="delvecchio2021",
            log_mstar=11.0,
            redshift=1.0,
            apply_suppression=False,
        )
        assert float(L_delv[0]) != float(L_bell[0]), (
            "Delvecchio mode should give different L than Bell2003 for M★=11, z=1"
        )

    def test_mccheyne2022_mode_dispatches(self):
        """mccheyne2022 mode should return finite nonzero values."""
        L = radio_total(
            _WAVE_150MHZ,
            L_ir=_L_IR,
            L_agn_bol=0.0,
            sfr_mode="mccheyne2022",
            log_mstar=10.5,
            redshift=0.5,
            apply_suppression=False,
        )
        assert float(L[0]) > 0.0 and jnp.isfinite(L[0])

    def test_hierarchical_params_override_in_dispatcher(self):
        """q0 passed to radio_total is forwarded to the physics model."""
        L_default = radio_total(
            _WAVE_14GHZ,
            L_ir=_L_IR,
            L_agn_bol=0.0,
            sfr_mode="delvecchio2021",
            log_mstar=10.0,
            redshift=0.0,
            apply_suppression=False,
        )
        L_override = radio_total(
            _WAVE_14GHZ,
            L_ir=_L_IR,
            L_agn_bol=0.0,
            sfr_mode="delvecchio2021",
            log_mstar=10.0,
            redshift=0.0,
            q0=2.5,
            apply_suppression=False,
        )
        # q0=2.5 < 2.743 default → L should be higher (L ∝ 10^{-q})
        assert float(L_override[0]) > float(L_default[0])

    def test_total_dpl_bell2003_backward_compat(self):
        """radio_total_dpl with sfr_mode='bell2003' == old behavior."""
        from tengri.components.radio import radio_agn_dpl

        wave = _WAVE_RADIO
        L_via_total = radio_total_dpl(
            wave,
            L_ir=_L_IR,
            L_agn_bol=1e11,
            radio_loudness=1.0,
            sfr_mode="bell2003",
            include_freefree=False,
        )
        L_sf = radio_sfr_bell2003(wave, _L_IR)
        L_agn = radio_agn_dpl(wave, 1e11, radio_loudness=1.0)
        assert jnp.allclose(L_via_total, L_sf + L_agn, rtol=1e-12)

    def test_jit_parity_all_modes(self):
        """All sfr_mode options have JIT parity (eager == jitted within 1e-6 rtol).

        Physical: JIT compilation should preserve numerical outputs across all modes.
        """
        for mode in ("bell2003", "delvecchio2021", "mccheyne2022"):
            L_eager = radio_total(
                _WAVE_RADIO,
                L_ir=_L_IR,
                L_agn_bol=0.0,
                sfr_mode=mode,
                log_mstar=10.0,
                redshift=0.0,
                apply_suppression=False,
            )
            jitted = jax.jit(
                radio_total,
                static_argnames=["sfr_mode", "apply_suppression"],
            )
            L_jit = jitted(
                _WAVE_RADIO,
                L_ir=_L_IR,
                L_agn_bol=0.0,
                sfr_mode=mode,
                log_mstar=10.0,
                redshift=0.0,
                apply_suppression=False,
            )
            assert L_jit.shape == _WAVE_RADIO.shape, f"Shape mismatch for mode={mode}"
            assert jnp.allclose(L_eager, L_jit, rtol=1e-6), f"JIT parity fail for mode={mode}"

    def test_only_radio_band_emits(self):
        """All three modes must return zero at optical/UV wavelengths."""
        wave_optical = jnp.logspace(3.0, 6.9, 50)  # 1000 A – 8e6 A (below 1 mm)
        for mode in ("bell2003", "delvecchio2021", "mccheyne2022"):
            L = radio_total(
                wave_optical,
                L_ir=_L_IR,
                L_agn_bol=0.0,
                sfr_mode=mode,
                log_mstar=10.0,
                redshift=0.0,
                apply_suppression=False,
            )
            assert jnp.all(L == 0.0), f"Non-zero optical flux for mode={mode}"


# Module-level kwargs reused across TestRadioComponents
_RC_KW: dict = dict(L_ir=_L_IR, L_agn_bol=1e12, sfr_mode="bell2003", apply_suppression=False)


# ── radio_freefree — Murphy+2011 thermal bremsstrahlung ───────────
class TestFreeFree:
    """Murphy+2011 thermal free-free calibration and spectral behavior."""

    def test_calibration_1p4ghz_murphy2011(self):
        """At 1.4 GHz, Te=1e4: L_ff ≈ 5.49e-7 Lsun/Hz per M☉/yr.
        Derivation: SFR=1 M☉/yr → L_ff = C_ff × 1.4^{-0.1} ≈ 5.49e-7.
        """
        sfr = 1.0  # M☉/yr
        _SFR_IR_KENNICUTT = 1.73e10
        L_ir_1sfr = sfr * _SFR_IR_KENNICUTT  # Lsun
        L = radio_freefree(_WAVE_14GHZ, L_ir_1sfr, T_e=1e4)
        val = float(L[0])
        assert 5.0e-7 < val < 6.5e-7, (
            f"Murphy+2011 calibration: L_ff(1.4 GHz) = {val:.3e} Lsun/Hz, expected 5.0e-7 – 6.5e-7"
        )

    def test_spectral_slope(self):
        """Spectral slope is alpha_ff = -0.1: ratio = (150MHz/1.4GHz)^{0.1}."""
        L_14ghz = radio_freefree(_WAVE_14GHZ, _L_IR)
        L_150mhz = radio_freefree(_WAVE_150MHZ, _L_IR)
        expected_ratio = (150.0e6 / 1.4e9) ** 0.1  # > 1, 150 MHz slightly brighter
        actual_ratio = float(L_14ghz[0] / L_150mhz[0])
        assert abs(actual_ratio - expected_ratio) / expected_ratio < 0.01, (
            f"Slope ratio {actual_ratio:.4f} != expected {expected_ratio:.4f}"
        )

    def test_steeper_alpha_ff(self):
        """alpha_ff = -0.3 at 1.4 GHz gives less emission than alpha_ff = -0.1.
        L_ν ∝ (ν/GHz)^{alpha_ff}. At ν > 1 GHz (like 1.4 GHz), more negative
        alpha_ff yields a smaller exponent result, so L_ff is lower.
        """
        L_flat = radio_freefree(_WAVE_14GHZ, _L_IR, alpha_ff=-0.1)
        L_steep = radio_freefree(_WAVE_14GHZ, _L_IR, alpha_ff=-0.3)
        # At 1.4 GHz (above 1 GHz ref), more negative alpha_ff → less emission
        assert float(L_steep[0]) < float(L_flat[0])

    def test_te_dependence(self):
        """L_ff ∝ T_e^{0.45}: doubling T_e raises L by factor 2^{0.45}."""
        L_1e4 = radio_freefree(_WAVE_14GHZ, _L_IR, T_e=1e4)
        L_2e4 = radio_freefree(_WAVE_14GHZ, _L_IR, T_e=2e4)
        ratio = float(L_2e4[0] / L_1e4[0])
        expected = 2.0**0.45
        assert abs(ratio - expected) / expected < 0.01, (
            f"Te ratio {ratio:.4f} != (2)^0.45 = {expected:.4f}"
        )

    def test_scales_linearly_with_l_ir(self):
        """L_ff ∝ L_ir (via SFR = L_ir / K98)."""
        L_lo = radio_freefree(_WAVE_14GHZ, _L_IR)
        L_hi = radio_freefree(_WAVE_14GHZ, 10.0 * _L_IR)
        ratio = float(L_hi[0] / L_lo[0])
        assert abs(ratio - 10.0) < 1e-6, f"Linearity: ratio {ratio:.6f} != 10"

    def test_zero_below_radio_cutoff(self):
        """Free-free must vanish at optical/UV wavelengths."""
        wave_optical = jnp.logspace(3.0, 6.9, 50)
        L = radio_freefree(wave_optical, _L_IR)
        assert jnp.all(L == 0.0)

    def test_positive_in_radio_band(self):
        """All radio-band wavelengths give positive L_ff."""
        L = radio_freefree(_WAVE_RADIO, _L_IR)
        assert jnp.all(L > 0.0)

    def test_jit_parity_and_shape(self):
        """JIT eager output match (parity) and output shape.

        Physical: JIT should not alter numerical results.
        """
        L_eager = radio_freefree(_WAVE_RADIO, _L_IR)
        jitted = jax.jit(radio_freefree)
        L_jit = jitted(_WAVE_RADIO, _L_IR)
        chex.assert_equal_shape([L_jit, _WAVE_RADIO])
        chex.assert_trees_all_close(L_eager, L_jit, rtol=1e-6)

    def test_gradients_flow_through_l_ir(self):
        """FD check: ∂(∑L_ff)/∂L_ir. Murphy+2011 linear calibration."""

        def f(l):
            return float(jnp.sum(radio_freefree(_WAVE_14GHZ, l)))

        grad_jax = float(jax.grad(lambda l: jnp.sum(radio_freefree(_WAVE_14GHZ, l)))(_L_IR))
        grad_fd = fd_grad(f, _L_IR, eps=1e7)  # eps=1e7 Lsun for scale
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="radio_freefree: FD check ∂/∂L_ir",
        )

    def test_gradients_flow_through_te(self):
        """FD check: ∂(∑L_ff)/∂T_e. Thermal bremsstrahlung temperature dependence."""

        def f(t):
            return float(jnp.sum(radio_freefree(_WAVE_14GHZ, _L_IR, T_e=t)))

        grad_jax = float(
            jax.grad(lambda t: jnp.sum(radio_freefree(_WAVE_14GHZ, _L_IR, T_e=t)))(1e4)
        )
        grad_fd = fd_grad(f, 1e4, eps=10.0)  # eps=10 K for scale
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            err_msg="radio_freefree: FD check ∂/∂T_e",
        )


# ── compute_radio_components — diagnostic component decomposition ─────────
class TestRadioComponents:
    """compute_radio_components returns a dict with correct keys, values, and sum."""

    def test_dict_keys_present(self):
        comps = compute_radio_components(_WAVE_RADIO, **_RC_KW)
        assert set(comps.keys()) == {"synchrotron", "freefree", "agn", "total"}

    def test_values_sum_to_total(self):
        comps = compute_radio_components(_WAVE_RADIO, **_RC_KW, include_freefree=True)
        recon = comps["synchrotron"] + comps["freefree"] + comps["agn"]
        assert jnp.allclose(recon, comps["total"], rtol=1e-12)

    def test_freefree_zero_when_disabled(self):
        comps = compute_radio_components(_WAVE_RADIO, **_RC_KW, include_freefree=False)
        assert jnp.all(comps["freefree"] == 0.0)

    def test_freefree_positive_when_enabled(self):
        comps = compute_radio_components(_WAVE_RADIO, **_RC_KW, include_freefree=True)
        assert jnp.all(comps["freefree"] > 0.0)

    def test_total_includes_freefree_when_enabled(self):
        comps_off = compute_radio_components(_WAVE_RADIO, **_RC_KW, include_freefree=False)
        comps_on = compute_radio_components(_WAVE_RADIO, **_RC_KW, include_freefree=True)
        assert jnp.all(comps_on["total"] > comps_off["total"])

    def test_thermal_fraction_milky_way_like(self):
        """Thermal fraction at 1.4 GHz for L_ir=1e10 Lsun: 2–25% (Bell+2003 FIRRC ~5%)."""
        comps = compute_radio_components(
            _WAVE_14GHZ, L_ir=1e10, L_agn_bol=0.0, include_freefree=True, apply_suppression=False
        )
        total = float(comps["total"][0])
        ff = float(comps["freefree"][0])
        f_thermal = ff / total
        assert 0.02 < f_thermal < 0.25, f"Thermal fraction {f_thermal:.3f} outside expected 2–25%"

    def test_shape_and_finiteness(self):
        """All component shapes match wavelength grid, and all values are finite."""
        comps = compute_radio_components(_WAVE_RADIO, **_RC_KW, include_freefree=True)
        for key in ("synchrotron", "freefree", "agn", "total"):
            arr = comps[key]
            assert arr.shape == _WAVE_RADIO.shape, f"Shape mismatch for {key}"
            assert jnp.all(jnp.isfinite(arr)), f"Non-finite values in {key}"


# ── TestLayerConsistency — radio_total == components summed ───────
class TestLayerConsistency:
    """radio_total output equals component-wise sum from compute_radio_components."""

    def _check_consistency(self, sfr_mode, wave, include_freefree, **kw):
        total_direct = radio_total(
            wave,
            L_ir=_L_IR,
            L_agn_bol=0.0,
            sfr_mode=sfr_mode,
            include_freefree=include_freefree,
            apply_suppression=False,
            **kw,
        )
        comps = compute_radio_components(
            wave,
            L_ir=_L_IR,
            L_agn_bol=0.0,
            sfr_mode=sfr_mode,
            include_freefree=include_freefree,
            apply_suppression=False,
            **kw,
        )
        assert jnp.allclose(total_direct, comps["total"], rtol=1e-12)

    def test_bell2003_no_freefree_no_agn(self):
        self._check_consistency("bell2003", _WAVE_RADIO, include_freefree=False)

    def test_bell2003_with_freefree_no_agn(self):
        self._check_consistency("bell2003", _WAVE_RADIO, include_freefree=True)

    def test_delvecchio2021_no_freefree_no_agn(self):
        self._check_consistency(
            "delvecchio2021",
            _WAVE_14GHZ,
            include_freefree=False,
            log_mstar=10.5,
            redshift=1.0,
        )

    def test_mccheyne2022_no_freefree_no_agn(self):
        self._check_consistency(
            "mccheyne2022",
            _WAVE_150MHZ,
            include_freefree=False,
            log_mstar=10.5,
            redshift=0.5,
        )

    def test_bell2003_with_agn_no_freefree(self):
        """SF + AGN total equals components["total"] with nonzero AGN."""
        total = radio_total(
            _WAVE_RADIO,
            L_ir=_L_IR,
            L_agn_bol=1e12,
            radio_loudness=2.0,
            sfr_mode="bell2003",
            include_freefree=False,
            apply_suppression=False,
        )
        comps = compute_radio_components(
            _WAVE_RADIO,
            L_ir=_L_IR,
            L_agn_bol=1e12,
            radio_loudness=2.0,
            sfr_mode="bell2003",
            include_freefree=False,
            apply_suppression=False,
        )
        assert jnp.allclose(total, comps["total"], rtol=1e-12)

    def test_backward_compat_include_freefree_false(self):
        """include_freefree=False gives synchrotron-only output (no free-free)."""
        L_old = radio_sfr_bell2003(_WAVE_RADIO, _L_IR)
        L_new = radio_total(_WAVE_RADIO, L_ir=_L_IR, L_agn_bol=0.0, include_freefree=False)
        assert jnp.allclose(L_old, L_new, rtol=1e-12)
