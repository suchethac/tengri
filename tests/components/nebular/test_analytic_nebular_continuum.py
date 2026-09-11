# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for compute_analytic_nebular_continuum in _shared.py."""

import chex
import pytest

pytestmark = pytest.mark.bounds

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._shared import compute_analytic_nebular_continuum
from tests._bounds import assert_non_negative


def fd_grad(f, x: float, eps: float = 1e-4) -> float:
    """Central finite difference: (f(x+eps) - f(x-eps)) / (2*eps)."""
    return float((f(x + eps) - f(x - eps)) / (2.0 * eps))


# Default test wavelength grid covering UV through NIR
_WAVE_AA = jnp.linspace(912.0, 10000.0, 4000)

# Lyman-alpha wavelength [Å]
_LYA_AA = 1216.0

# Reference ionizing photon rate [photon/s]
_Q_H_REF = 1e50


class TestAnalyticNebularContinuumBasic:
    """Basic sanity checks on the analytic free-free + two-photon continuum."""

    def test_output_shape(self):
        """Output shape must match input wavelength grid."""
        sed = compute_analytic_nebular_continuum(_WAVE_AA, _Q_H_REF, log_z_abs=-1.848)
        chex.assert_equal_shape([sed, _WAVE_AA])

    def test_non_negative(self):
        """SED must be non-negative everywhere."""
        sed = compute_analytic_nebular_continuum(_WAVE_AA, _Q_H_REF, log_z_abs=-1.848)
        assert_non_negative(sed, name="sed", msg=f"Minimum value: {float(jnp.min(sed))}")

    def test_finite(self):
        """SED must be finite everywhere."""
        sed = compute_analytic_nebular_continuum(_WAVE_AA, _Q_H_REF, log_z_abs=-1.848)
        chex.assert_tree_all_finite(sed)

    def test_zero_qh_gives_zero_sed(self):
        """Q_H = 0 must produce a zero SED."""
        sed = compute_analytic_nebular_continuum(_WAVE_AA, 0.0, log_z_abs=-1.848)
        assert jnp.allclose(sed, 0.0), "Non-zero SED for Q_H=0"

    def test_linear_scaling_with_qh(self):
        """Doubling Q_H must double the SED at every wavelength.

        The continuum is linear in Q_H (both free-free and two-photon scale as
        Q_H / α_B, with α_B independent of Q_H).
        """
        sed1 = compute_analytic_nebular_continuum(_WAVE_AA, _Q_H_REF, log_z_abs=-1.848)
        sed2 = compute_analytic_nebular_continuum(_WAVE_AA, 2 * _Q_H_REF, log_z_abs=-1.848)
        nonzero = sed1 > 0
        ratio = sed2[nonzero] / sed1[nonzero]
        max_dev = float(jnp.max(jnp.abs(ratio - 2.0)))
        assert jnp.allclose(ratio, 2.0, rtol=1e-5), f"Ratio deviates from 2.0: {max_dev}"


class TestAnalyticNebularContinuumPhysics:
    """Physical correctness tests for the two-component continuum."""

    def test_freefree_nonzero_at_optical(self):
        """Free-free emission must be present at optical wavelengths (λ > 1216 Å).

        At λ > Lyman-α the two-photon component is zero, so any signal must
        come from free-free bremsstrahlung.
        """
        wave_optical = jnp.linspace(3000.0, 8000.0, 500)
        sed = compute_analytic_nebular_continuum(wave_optical, _Q_H_REF, log_z_abs=-1.848)
        assert jnp.all(sed > 0), "Free-free should be non-zero at optical wavelengths"

    def test_two_photon_zero_below_lya(self):
        """Two-photon component must vanish for λ < 1216 Å.

        The 2s→1s two-photon transition produces pairs of photons that together
        have the energy of one Lyα photon.  Each individual photon therefore has
        LESS than the Lyα energy, meaning λ > 1216 Å.  The spectrum is
        parameterized as y = ν/ν_Lyα = λ_Lyα/λ ∈ (0, 1), valid only for λ > λ_Lyα.

        Physical check (Nussbaumer & Schmutz 1984 / OF06 §4.5):
          • At λ < 1216 Å (y > 1): two-photon = 0; only free-free contributes.
          • At λ ≈ 2 × 1216 ≈ 2432 Å (y = 0.5): two-photon peaks.
          • The peak of the total SED (ff + 2q) in the range 1217–4000 Å must exceed
            the free-free-only level at λ < 1216 Å by a factor > 2.
        """
        wave_above_lya = jnp.linspace(1217.0, 4000.0, 100)  # two-photon active
        wave_below_lya = jnp.linspace(800.0, 1215.0, 100)  # two-photon = 0
        sed_above = compute_analytic_nebular_continuum(wave_above_lya, _Q_H_REF, log_z_abs=-1.848)
        sed_below = compute_analytic_nebular_continuum(wave_below_lya, _Q_H_REF, log_z_abs=-1.848)

        # Above 1216 Å: two-photon + free-free; peak should >> free-free alone at <1216 Å
        ratio = float(jnp.max(sed_above)) / float(jnp.max(sed_below))
        assert ratio > 2.0, (
            f"Expected two-photon peak (λ>1216) > 2× free-free level (λ<1216), got {ratio:.2f}. "
            "Two-photon continuum may not be activating above 1216 Å."
        )

        # Below 1216 Å the SED must be smoothly decreasing (no two-photon jump)
        # Free-free is monotonically decreasing with wavelength at optical/UV, so
        # the SED at 1215 Å should be close to that at 1200 Å (no discontinuity).
        wave_bracket = jnp.array([1200.0, 1215.0])
        sed_bracket = compute_analytic_nebular_continuum(wave_bracket, _Q_H_REF, log_z_abs=-1.848)
        smooth_ratio = float(sed_bracket[1]) / float(sed_bracket[0])
        assert 0.5 < smooth_ratio < 2.0, (
            f"SED discontinuity below Lyα: SED(1215)/SED(1200) = {smooth_ratio:.3f}. "
            "Two-photon may be leaking below 1216 Å."
        )

    def test_two_photon_peak_above_lya(self):
        """Two-photon spectrum peaks near λ ~ 2432 Å (y ~ 0.5).

        The Nussbaumer & Schmutz (1984) shape A₂γ(y) peaks at y = ν/ν_Lyα ≈ 0.5,
        which corresponds to λ ≈ 2 × λ_Lyα ≈ 2432 Å.  Two-photon emission spans
        λ > 1216 Å (each photon in the pair has energy < E_Lyα).

        We verify:
          1. The total SED has a clear peak in the 1500–4000 Å range.
          2. The peak wavelength lies in the range [1400, 3000] Å.
        """
        wave_nuv = jnp.linspace(1217.0, 10000.0, 5000)
        sed_nuv = compute_analytic_nebular_continuum(wave_nuv, _Q_H_REF, log_z_abs=-1.848)
        assert float(jnp.max(sed_nuv)) > 0.0, "Two-photon SED is zero everywhere above Lyα"
        peak_wave = float(wave_nuv[int(jnp.argmax(sed_nuv))])
        assert 1400.0 < peak_wave < 3000.0, (
            f"Two-photon peak wavelength {peak_wave:.0f} Å outside expected 1400–3000 Å range. "
            "Check y-convention: y = λ_Lyα / λ (not λ / λ_Lyα)."
        )

    def test_temperature_dependence(self):
        """Higher temperature gives higher free-free luminosity at optical λ.

        The total free-free luminosity scales as:
            L_ff ∝ (Q_H / α_B) × T^{-0.5} × g_ff
        where α_B ∝ T^{-0.7}, so Q_H/α_B ∝ T^{+0.7}.
        The Gaunt factor also increases with T (g_ff ∝ log(kT/hν)).
        Net result: L_ff increases with temperature.  At T=2e4 K the optical
        free-free luminosity is several times larger than at T=1e4 K.
        """
        wave_opt = jnp.linspace(3000.0, 8000.0, 500)
        sed_cool = compute_analytic_nebular_continuum(
            wave_opt, _Q_H_REF, log_z_abs=-1.848, temperature=1e4
        )
        sed_hot = compute_analytic_nebular_continuum(
            wave_opt, _Q_H_REF, log_z_abs=-1.848, temperature=2e4
        )
        assert float(jnp.sum(sed_hot)) > float(jnp.sum(sed_cool)), (
            "Free-free luminosity should increase with temperature "
            "(Q_H/α_B ∝ T^{0.7} dominates the T^{-0.5} emissivity factor)"
        )

    def test_magnitude_plausible(self):
        """Total continuum luminosity should be comparable to L_Hα for a typical HII region.

        For a 10^4 K HII region, the free-free + two-photon continuum integrated
        over UV-optical is typically 30-100% of L_Hα.
        L_Hα ~ 1.37e-12 × Q_H (Case B, T=10^4 K; Osterbrock & Ferland 2006 eq. 4.2).
        We allow 0.01× to 100× L_Hα as physically plausible bounds.
        """
        wave_uv_opt = jnp.linspace(912.0, 8000.0, 3000)
        sed = compute_analytic_nebular_continuum(wave_uv_opt, _Q_H_REF, log_z_abs=-1.848)
        # Integrate L_cont in frequency space: ν = c / λ, c = 3e18 Å/s
        nu = 3e18 / wave_uv_opt  # Hz (c in Å/s ÷ λ in Å)
        dnu = jnp.abs(jnp.gradient(nu))
        l_cont = float(jnp.sum(sed * dnu))
        # L_Hα reference
        l_ha = 1.37e-12 * float(_Q_H_REF)
        assert l_cont < 100.0 * l_ha, (
            f"Continuum luminosity ({l_cont:.2e}) implausibly >> L_Hα ({l_ha:.2e})"
        )
        assert l_cont > 0.01 * l_ha, (
            f"Continuum luminosity ({l_cont:.2e}) implausibly << L_Hα ({l_ha:.2e})"
        )


class TestAnalyticNebularContinuumDifferentiability:
    """Gradient checks for JAX differentiability."""

    def test_grad_wrt_qh_finite(self):
        """Gradient w.r.t. Q_H must be finite and positive."""
        wave = jnp.linspace(3000.0, 8000.0, 500)

        def total_flux(q_h):
            return jnp.sum(compute_analytic_nebular_continuum(wave, q_h, log_z_abs=-1.848))

        grad_jax = float(jax.grad(total_flux)(_Q_H_REF))
        grad_fd = fd_grad(total_flux, float(_Q_H_REF))
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-12,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )
        assert grad_jax > 0.0, f"Expected positive gradient w.r.t. Q_H, got {grad_jax}"

    def test_grad_wrt_temperature_finite(self):
        """Gradient w.r.t. temperature must be finite."""
        wave = jnp.linspace(3000.0, 8000.0, 500)

        def total_flux(temp):
            return jnp.sum(
                compute_analytic_nebular_continuum(
                    wave, _Q_H_REF, log_z_abs=-1.848, temperature=temp
                )
            )

        grad_jax = float(jax.grad(total_flux)(1e4))
        grad_fd = fd_grad(total_flux, 1e4)
        np.testing.assert_allclose(
            grad_jax,
            grad_fd,
            rtol=1e-3,
            atol=1e-12,
            err_msg=f"autodiff={grad_jax:.4e}, FD={grad_fd:.4e}",
        )

    def test_jit_compatible(self):
        """Function must be JIT-compilable."""
        wave = jnp.linspace(3000.0, 8000.0, 200)

        @jax.jit
        def _compute(q_h):
            return compute_analytic_nebular_continuum(wave, q_h, log_z_abs=-1.848)

        sed = _compute(_Q_H_REF)
        chex.assert_equal_shape([sed, wave])
        chex.assert_tree_all_finite(sed)


class TestAnalyticNebularContinuumLog10QH:
    """The float32-safe ``log10_q_h`` entry point (#1206 §C).

    ``compute_analytic_nebular_continuum`` forms :math:`Q_H/\\alpha_B` linearly
    by default; the linear ``q_h`` (up to ~1e53 photons/s for a real ionizing
    source, ``gas_logqion`` up to ~53) overflows float32's 3.4e38 ceiling
    whenever ``gas_logqion > 38.5``, while the ~1e28 erg/s/Hz continuum this
    function returns is representable throughout. ``log10_q_h`` carries the
    ratio as a log10 offset via ``apply_log10_scale`` instead.
    """

    def test_log10_q_h_matches_linear_q_h_in_float64(self):
        """The two entry points agree to 1e-12 relative in float64."""
        wave = jnp.linspace(912.0, 10000.0, 2000)
        log10_q_h = 50.0  # log10(_Q_H_REF), same reference rate as the file default

        via_linear = compute_analytic_nebular_continuum(wave, _Q_H_REF, log_z_abs=-1.848)
        via_log = compute_analytic_nebular_continuum(wave, log_z_abs=-1.848, log10_q_h=log10_q_h)
        np.testing.assert_allclose(
            np.asarray(via_log), np.asarray(via_linear), rtol=1e-12, atol=0.0
        )

    def test_log10_q_h_finite_in_pure_float32_at_gas_logqion_52(self):
        """The linear path overflows at ``gas_logqion = 52``; the log path does not.

        ``gas_logqion = 52`` (``Q_H ~ 1e52`` photons/s) is a representative real
        ionizing source, past float32's 3.4e38 ceiling. Confirms both halves of
        the claim: the linear path really does overflow (so this test's premise
        is live), and the log path stays finite and non-zero.
        """
        wave = jnp.linspace(912.0, 10000.0, 2000)

        with jax.enable_x64(False):
            wave32 = jnp.asarray(wave, dtype=jnp.float32)
            q_h32 = jnp.asarray(10.0**52, dtype=jnp.float32)
            assert not bool(jnp.isfinite(q_h32)), (
                f"q_h=1e52 no longer overflows float32 ({q_h32}) -- this test's "
                "premise is gone, re-check whether the log10_q_h path is still needed"
            )
            linear = compute_analytic_nebular_continuum(wave32, q_h32, log_z_abs=-1.848)
            assert not bool(jnp.all(jnp.isfinite(linear))), (
                "the linear q_h path is finite at gas_logqion=52 in float32 -- "
                "expected an overflow this test exists to route around"
            )

            log10_q_h32 = jnp.asarray(52.0, dtype=jnp.float32)
            via_log = compute_analytic_nebular_continuum(
                wave32, log_z_abs=-1.848, log10_q_h=log10_q_h32
            )

        assert via_log.dtype == jnp.float32, f"not float32: {via_log.dtype}"
        assert bool(jnp.all(jnp.isfinite(via_log))), (
            f"log10_q_h path is non-finite in pure float32 at gas_logqion=52: {via_log}"
        )
        assert bool(jnp.any(via_log != 0.0)), (
            "log10_q_h path is identically zero in pure float32 -- finite is not "
            "enough, a value that has collapsed to zero is as unusable as a NaN one"
        )

    def test_requires_q_h_or_log10_q_h(self):
        """Neither q_h nor log10_q_h given raises, rather than silently NaN-ing."""
        wave = jnp.linspace(3000.0, 8000.0, 200)
        with pytest.raises(ValueError, match="q_h"):
            compute_analytic_nebular_continuum(wave, log_z_abs=-1.848)
