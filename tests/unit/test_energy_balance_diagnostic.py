"""Tests for the user-facing energy-balance diagnostic.

MISSING_FEATURES.md #4. Verifies that the absorbed/emitted-luminosity
diagnostic correctly reports balance for synthetic SEDs with known
input/output luminosity. Distinct from test_energy_balance.py which
tests the dust_eta_balance forward-model parameter.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.analysis.diagnostics.energy_balance import (
    dust_energy_balance,
    integrate_lnu_over_band,
)

pytestmark = pytest.mark.unit

_C_AA = 2.99792458e18  # Å/s


def _flat_l_lambda_to_l_nu(wave_aa, f_lambda):
    """Convert F_λ (per-Å) to L_ν via L_ν = (λ²/c) × F_λ."""
    return f_lambda * (wave_aa**2 / _C_AA)


# ── integrate_lnu_over_band ───────────────────────────────────────────


class TestIntegrateLnuOverBand:
    def test_flat_l_lambda_unit_continuum(self):
        wave = jnp.linspace(1000.0, 10000.0, 10000)
        l_nu = _flat_l_lambda_to_l_nu(wave, jnp.ones_like(wave))
        l_int = float(integrate_lnu_over_band(wave, l_nu, 1000.0, 10000.0))
        # ν integral of L_ν = λ integral of F_λ = 1 × (10000−1000) = 9000
        assert l_int == pytest.approx(9000.0, rel=1e-3)

    def test_band_outside_grid_returns_zero(self):
        wave = jnp.linspace(1000.0, 10000.0, 1000)
        l_nu = _flat_l_lambda_to_l_nu(wave, jnp.ones_like(wave))
        l_int = float(integrate_lnu_over_band(wave, l_nu, 50000.0, 60000.0))
        assert l_int == 0.0

    def test_band_partially_overlapping(self):
        wave = jnp.linspace(1000.0, 10000.0, 10000)
        l_nu = _flat_l_lambda_to_l_nu(wave, jnp.ones_like(wave))
        # band 5000-15000 overlaps grid only on 5000-10000 → 5000 worth
        l_int = float(integrate_lnu_over_band(wave, l_nu, 5000.0, 15000.0))
        assert l_int == pytest.approx(5000.0, rel=1e-2)


# ── dust_energy_balance ──────────────────────────────────────────────


class TestDustEnergyBalance:
    def _make_seds(self, wave, atten_uv_factor: float, l_dust_factor: float):
        unatten_f_lambda = jnp.where((wave >= 912.0) & (wave <= 30000.0), 1.0, 0.0)
        atten_f_lambda = atten_uv_factor * unatten_f_lambda
        peak = 1.0e6  # 100 μm
        sigma = 3.0e5
        ir_shape = jnp.exp(-0.5 * ((wave - peak) / sigma) ** 2) / (sigma * jnp.sqrt(2.0 * jnp.pi))
        absorbed_l_lambda = (1.0 - atten_uv_factor) * (30000.0 - 912.0)
        ir_f_lambda = l_dust_factor * absorbed_l_lambda * ir_shape
        return (
            _flat_l_lambda_to_l_nu(wave, unatten_f_lambda),
            _flat_l_lambda_to_l_nu(wave, atten_f_lambda),
            _flat_l_lambda_to_l_nu(wave, ir_f_lambda),
        )

    def test_perfect_balance(self):
        wave = jnp.logspace(np.log10(800), np.log10(3.0e6), 30000)
        l_unatten, l_atten, l_dust = self._make_seds(wave, atten_uv_factor=0.5, l_dust_factor=1.0)
        result = dust_energy_balance(wave, l_unatten, l_atten, l_dust)
        assert result["ratio"] == pytest.approx(1.0, rel=0.05)
        assert result["balanced"]
        assert result["absorbed"] > 0
        assert result["emitted"] > 0

    def test_under_emission(self):
        wave = jnp.logspace(np.log10(800), np.log10(3.0e6), 30000)
        l_unatten, l_atten, l_dust = self._make_seds(wave, atten_uv_factor=0.5, l_dust_factor=0.5)
        result = dust_energy_balance(wave, l_unatten, l_atten, l_dust)
        assert result["ratio"] == pytest.approx(0.5, rel=0.05)
        assert not result["balanced"]

    def test_over_emission(self):
        wave = jnp.logspace(np.log10(800), np.log10(3.0e6), 30000)
        l_unatten, l_atten, l_dust = self._make_seds(wave, atten_uv_factor=0.5, l_dust_factor=2.0)
        result = dust_energy_balance(wave, l_unatten, l_atten, l_dust)
        assert result["ratio"] == pytest.approx(2.0, rel=0.05)
        assert not result["balanced"]

    def test_zero_attenuation_zero_absorbed(self):
        wave = jnp.logspace(np.log10(800), np.log10(3.0e6), 5000)
        l_unatten, _, l_dust = self._make_seds(wave, atten_uv_factor=1.0, l_dust_factor=1.0)
        result = dust_energy_balance(wave, l_unatten, l_unatten, l_dust)
        assert result["absorbed"] == pytest.approx(0.0, abs=1e-10)
        # ratio is undefined when absorbed=0; expect inf, nan, or 0
        assert not np.isfinite(result["ratio"]) or result["ratio"] == 0.0

    def test_tolerance_kwarg(self):
        wave = jnp.logspace(np.log10(800), np.log10(3.0e6), 30000)
        l_unatten, l_atten, l_dust = self._make_seds(wave, atten_uv_factor=0.5, l_dust_factor=1.05)
        result_loose = dust_energy_balance(wave, l_unatten, l_atten, l_dust, tol=0.10)
        result_strict = dust_energy_balance(wave, l_unatten, l_atten, l_dust, tol=0.01)
        assert result_loose["balanced"]
        assert not result_strict["balanced"]
