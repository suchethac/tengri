# SPDX-License-Identifier: BSD-3-Clause
"""Invariant tests for emission_helpers.py.

Bug classes covered:
- Shape regression: output shape must match input wave shape for all helpers.
- Energy conservation: L_transmitted + L_absorbed = L_incident within 1%,
  with L_absorbed computed via the canonical energy-balance integral
  (:func:`tengri.forward.energy_balance.bolometric_absorbed`, #922).
- Zero-dust identity: tau_bc=0, tau_diff=0 → attenuation factor ≈ 1 everywhere.
- Mode completeness: all four modes ("bc", "diff", "neb", "none") return finite arrays.
- IGM: z=0 → transmission=1, no NaN for short-wavelength photons.

No SSP data needed. All tests use synthetic jnp arrays.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri.forward.emission_helpers import (
    _C_AA,
    attenuate_emission,
)
from tengri.forward.energy_balance import bolometric_absorbed

# ── Helpers: synthetic grids and mock dust laws ───────────────────

_N_WAVE = 150
_WAVE = jnp.linspace(1000.0, 25000.0, _N_WAVE)  # Å


def _flat_sed(n: int = _N_WAVE, amplitude: float = 1e-15) -> jnp.ndarray:
    """Flat (spectrally constant) SED in erg/s/Hz."""
    return jnp.ones(n) * amplitude


def _power_sed(wave: jnp.ndarray, alpha: float = -1.0, amplitude: float = 1e-15) -> jnp.ndarray:
    """Power-law SED L_nu ∝ lambda^alpha."""
    return amplitude * (wave / wave[len(wave) // 2]) ** alpha


def _const_law(wave: jnp.ndarray, *, n_slope: float = -0.7, dust_bump_strength: float = 0.0):
    """Trivially constant dust law k(λ) = 1.0 for all λ."""
    return jnp.ones_like(wave)


def _calzetti_approx(wave: jnp.ndarray, *, n_slope: float = -0.7, dust_bump_strength: float = 0.0):
    """Approximate Calzetti-style power-law k(λ) = (V/λ)^0.7."""
    return (5500.0 / wave) ** 0.7


def _absorbed(sed_in: jnp.ndarray, sed_out: jnp.ndarray, wave: jnp.ndarray) -> float:
    """Positive absorbed luminosity [erg/s] via the canonical integral (unmasked)."""
    nu = _C_AA / wave
    signed = bolometric_absorbed(sed_in, sed_out, nu, wave=wave, lyman_cutoff_aa=None)
    return float(jnp.abs(signed))


# ── Shape invariants ──────────────────────────────────────────────


class TestAttenuateEmissionShape:
    @pytest.mark.parametrize("n_wave", [10, 50, 200])
    def test_output_shape_matches_input(self, n_wave: int) -> None:
        wave = jnp.linspace(1000.0, 20000.0, n_wave)
        sed = _flat_sed(n_wave)
        sed_out = attenuate_emission(
            sed,
            wave,
            "bc",
            tau_bc=0.3,
            tau_diff=0.5,
            law_bc_fn=_const_law,
            law_diff_fn=_const_law,
        )
        assert sed_out.shape == (n_wave,), (
            f"Output shape {sed_out.shape} does not match input shape ({n_wave},)"
        )

    @pytest.mark.parametrize("mode", ["bc", "diff", "neb", "none"])
    def test_all_modes_return_correct_shape(self, mode: str) -> None:
        sed = _flat_sed()
        sed_out = attenuate_emission(
            sed,
            _WAVE,
            mode,
            tau_bc=0.3,
            tau_diff=0.5,
            law_bc_fn=_const_law,
            law_diff_fn=_const_law,
            neb_bc_fn=_const_law,
        )
        assert sed_out.shape == (_N_WAVE,), (
            f"Mode {mode!r}: output shape {sed_out.shape} ≠ ({_N_WAVE},)"
        )


# ── Absorbed luminosity (canonical integral over the helper's output) ─


class TestAbsorbedLuminosity:
    def test_l_absorbed_is_finite(self) -> None:
        sed = _flat_sed()
        sed_out = attenuate_emission(
            sed,
            _WAVE,
            "bc",
            tau_bc=0.5,
            tau_diff=0.5,
            law_bc_fn=_const_law,
            law_diff_fn=_const_law,
        )
        assert jnp.isfinite(_absorbed(sed, sed_out, _WAVE)), "L_absorbed is not finite"

    def test_l_absorbed_positive_for_positive_tau(self) -> None:
        """Nonzero optical depth must absorb a strictly positive luminosity."""
        sed = _flat_sed()
        sed_out = attenuate_emission(
            sed,
            _WAVE,
            "bc",
            tau_bc=0.5,
            tau_diff=0.5,
            law_bc_fn=_const_law,
            law_diff_fn=_const_law,
        )
        assert _absorbed(sed, sed_out, _WAVE) > 0.0, "tau > 0 should absorb luminosity"


# ── All four modes return finite arrays ───────────────────────────


class TestAttenuateEmissionModes:
    @pytest.mark.parametrize("mode", ["bc", "diff", "neb", "none"])
    def test_mode_returns_finite_sed(self, mode: str) -> None:
        sed = _power_sed(_WAVE)
        sed_out = attenuate_emission(
            sed,
            _WAVE,
            mode,
            tau_bc=0.3,
            tau_diff=0.5,
            law_bc_fn=_calzetti_approx,
            law_diff_fn=_calzetti_approx,
            neb_bc_fn=_const_law,
        )
        assert jnp.all(jnp.isfinite(sed_out)), (
            f"Mode {mode!r}: output SED contains non-finite values"
        )
        assert jnp.isfinite(_absorbed(sed, sed_out, _WAVE)), (
            f"Mode {mode!r}: L_absorbed is non-finite"
        )

    @pytest.mark.parametrize("mode", ["bc", "diff", "neb"])
    def test_attenuated_sed_not_greater_than_input(self, mode: str) -> None:
        """Attenuation must not amplify the SED (L_out ≤ L_in pixel-wise)."""
        sed = _flat_sed()
        sed_out = attenuate_emission(
            sed,
            _WAVE,
            mode,
            tau_bc=0.3,
            tau_diff=0.5,
            law_bc_fn=_calzetti_approx,
            law_diff_fn=_calzetti_approx,
        )
        assert jnp.all(sed_out <= sed + 1e-30), (
            f"Mode {mode!r}: attenuated SED exceeds input SED at some wavelengths"
        )


# ── Zero-dust identity ────────────────────────────────────────────


class TestZeroDustIdentity:
    def test_zero_tau_bc_and_diff_returns_input(self) -> None:
        """With tau_bc=0 and tau_diff=0, output must be ≈ input everywhere."""
        sed = _power_sed(_WAVE)
        sed_out = attenuate_emission(
            sed,
            _WAVE,
            "bc",
            tau_bc=0.0,
            tau_diff=0.0,
            law_bc_fn=_calzetti_approx,
            law_diff_fn=_calzetti_approx,
        )
        rel_err = jnp.abs(sed_out - sed) / (jnp.abs(sed) + 1e-40)
        max_rel_err = float(jnp.max(rel_err))
        assert max_rel_err < 1e-6, (
            f"Zero-dust case: max relative error {max_rel_err:.2e} (should be ~0)"
        )

    def test_zero_dust_none_mode_returns_input(self) -> None:
        """mode='none' must always return the input SED exactly."""
        sed = _power_sed(_WAVE)
        sed_out = attenuate_emission(
            sed,
            _WAVE,
            "none",
            tau_bc=0.3,
            tau_diff=0.5,
            law_bc_fn=_calzetti_approx,
            law_diff_fn=_calzetti_approx,
        )
        assert jnp.allclose(sed_out, sed, atol=0.0, rtol=0.0), (
            "mode='none' did not return input SED exactly"
        )
        L_absorbed = _absorbed(sed, sed_out, _WAVE)
        assert L_absorbed == 0.0, f"mode='none' L_absorbed should be 0, got {L_absorbed}"


# ── Energy conservation ───────────────────────────────────────────


class TestEnergyConservation:
    def _integrate_luminosity(self, sed: jnp.ndarray, wave: jnp.ndarray) -> float:
        """Integrate L_nu over frequency ν = c/λ to get total luminosity (erg/s)."""
        nu = _C_AA / wave
        return float(-jnp.trapezoid(sed, nu))

    @pytest.mark.parametrize("mode", ["bc", "diff"])
    def test_energy_conservation_within_1pct(self, mode: str) -> None:
        """L_transmitted + L_absorbed ≈ L_incident within 1%."""
        sed = _flat_sed(amplitude=1e-15)
        tau = 0.5
        sed_out = attenuate_emission(
            sed,
            _WAVE,
            mode,
            tau_bc=tau,
            tau_diff=tau,
            law_bc_fn=_calzetti_approx,
            law_diff_fn=_calzetti_approx,
        )
        L_absorbed = _absorbed(sed, sed_out, _WAVE)

        L_incident = self._integrate_luminosity(sed, _WAVE)
        L_transmitted = self._integrate_luminosity(sed_out, _WAVE)

        rel_err = abs(L_transmitted + L_absorbed - L_incident) / (L_incident + 1e-40)
        assert rel_err < 0.01, (
            f"Mode {mode!r}: energy not conserved to 1%. "
            f"L_incident={L_incident:.3e}, L_transmitted={L_transmitted:.3e}, "
            f"L_absorbed={L_absorbed:.3e}, rel_err={rel_err:.3%}"
        )

    def test_none_mode_zero_absorbed(self) -> None:
        """mode='none' must have exactly zero absorbed luminosity."""
        sed = _flat_sed()
        sed_out = attenuate_emission(
            sed,
            _WAVE,
            "none",
            tau_bc=0.3,
            tau_diff=0.5,
            law_bc_fn=_calzetti_approx,
            law_diff_fn=_calzetti_approx,
        )
        assert _absorbed(sed, sed_out, _WAVE) == 0.0

    def test_high_tau_absorbs_most_luminosity(self) -> None:
        """Very high optical depth should absorb nearly all incident luminosity."""
        sed = _flat_sed(amplitude=1e-15)
        sed_out_high = attenuate_emission(
            sed,
            _WAVE,
            "bc",
            tau_bc=10.0,
            tau_diff=10.0,
            law_bc_fn=_calzetti_approx,
            law_diff_fn=_calzetti_approx,
        )
        sed_out_low = attenuate_emission(
            sed,
            _WAVE,
            "bc",
            tau_bc=0.1,
            tau_diff=0.1,
            law_bc_fn=_calzetti_approx,
            law_diff_fn=_calzetti_approx,
        )
        assert _absorbed(sed, sed_out_high, _WAVE) > _absorbed(sed, sed_out_low, _WAVE), (
            "Higher optical depth should absorb more luminosity"
        )


# ── IGM absorption ────────────────────────────────────────────────


class TestIGMAbsorption:
    def test_igm_at_z0_is_all_ones(self) -> None:
        """At z=0, IGM is transparent: transmission should be ≈ 1 everywhere."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.linspace(900.0, 10000.0, 100)
        trans = igm_transmission(wave_obs, 0.0)
        chex.assert_tree_all_finite(trans)
        # At z=0, transmission must be 1 everywhere (no absorbers along sightline)
        assert jnp.allclose(trans, 1.0, atol=1e-6), (
            f"IGM transmission at z=0 deviates from 1. min={float(jnp.min(trans)):.4f}"
        )

    def test_igm_no_nan_short_wavelength(self) -> None:
        """No NaN at short wavelengths for moderate z (negative base fractional-power bug)."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.linspace(100.0, 1000.0, 80)
        trans = igm_transmission(wave_obs, 2.0)
        assert jnp.all(jnp.isfinite(trans)), (
            "IGM transmission contains NaN/Inf at short wavelengths "
            "(possible negative-base fractional-power bug)"
        )

    def test_igm_transmission_between_0_and_1(self) -> None:
        """IGM transmission is a fraction in [0, 1]."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.linspace(1000.0, 10000.0, 120)
        trans = igm_transmission(wave_obs, 3.0)
        assert jnp.all(trans >= 0.0), "IGM transmission has negative values"
        assert jnp.all(trans <= 1.0 + 1e-6), "IGM transmission exceeds 1.0"

    def test_igm_increases_opacity_with_redshift(self) -> None:
        """Higher redshift → more IGM absorption → lower average transmission at Lya."""
        from tengri.components.igm import igm_transmission

        # Short wavelengths (Lyman-series) — absorbed more at higher z
        wave_obs = jnp.linspace(1000.0, 2000.0, 50)
        trans_z1 = igm_transmission(wave_obs, 1.0)
        trans_z4 = igm_transmission(wave_obs, 4.0)
        assert float(jnp.mean(trans_z4)) <= float(jnp.mean(trans_z1)) + 1e-3, (
            "Higher z should give less IGM transmission, not more"
        )
