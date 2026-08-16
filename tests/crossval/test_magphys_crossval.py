# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: F821
"""Cross-validation tests for MAGPHYS dust emission against literature values.

Tests verify:
- PAH line ratios match Smith+2007
- Rayleigh-Jeans slopes match expected 2+beta power laws
- Wien peaks of MBB components match analytic expectations

These tests are NOT run by default (crossval marker).
Run with: pytest -m crossval tests/crossval/test_magphys_crossval.py

NOTE: magphys_dc08 and related functions not currently implemented. Module skipped.
"""

import jax.numpy as jnp
import pytest

pytest.skip("magphys_dc08 not implemented", allow_module_level=True)

pytestmark = pytest.mark.crossval

# Fine wavelength grid for accurate peak / slope measurements
_WAVE_AA = jnp.logspace(jnp.log10(1e4), jnp.log10(1e8), 10000)
_C_CGS = 2.99792458e10
_AA_TO_CM = 1e-8
_H_PLANCK = 6.62607015e-27
_K_BOLTZMANN = 1.380649e-16


class TestPAHLineRatios:
    """PAH feature flux ratios compared to Smith+2007."""

    def _flux_at_wavelength(self, template: jnp.ndarray, target_um: float) -> float:
        """Get template flux near a target wavelength in microns."""
        target_aa = target_um * 1e4
        idx = jnp.argmin(jnp.abs(_WAVE_AA - target_aa))
        return float(template[idx])

    def test_pah_7p7_to_11p3_ratio(self):
        """PAH 7.7 / 11.3 um ratio ~ 1.9 (Smith+2007 Table 1 strengths).

        At the peak of a Drude profile D(lam_0) = S (the strength parameter),
        so the peak ratio is S_7.7 / S_11.3 = 1.0 / 0.52 ~ 1.92.
        """
        pah = _pah_template(_WAVE_AA)
        f_7p7 = self._flux_at_wavelength(pah, 7.7)
        f_11p3 = self._flux_at_wavelength(pah, 11.3)
        ratio = f_7p7 / f_11p3
        assert 1.5 < ratio < 2.5, f"PAH 7.7/11.3 ratio = {ratio:.2f}, expected ~1.9"

    def test_pah_6p2_to_7p7_ratio(self):
        """PAH 6.2 / 7.7 um ratio ~ 0.25 (Smith+2007)."""
        pah = _pah_template(_WAVE_AA)
        f_6p2 = self._flux_at_wavelength(pah, 6.2)
        f_7p7 = self._flux_at_wavelength(pah, 7.7)
        ratio = f_6p2 / f_7p7
        assert 0.15 < ratio < 0.35, f"PAH 6.2/7.7 ratio = {ratio:.2f}, expected ~0.25"


class TestRayleighJeansSlope:
    """Rayleigh-Jeans power-law slope of MBB components.

    In the RJ limit (h*nu << k*T), B_nu ~ nu^2, so
    L_nu ~ nu^beta * nu^2 = nu^(2+beta).
    """

    def _measure_rj_slope(
        self,
        temperature: float,
        beta: float,
        lam_range_um: tuple[float, float] = (3000.0, 8000.0),
    ) -> float:
        """Measure the RJ slope from the MBB component in log-log space."""
        comp = _modified_blackbody_component(_WAVE_AA, temperature, beta, 0.0)
        wavelength_cm = _WAVE_AA * _AA_TO_CM
        nu = _C_CGS / wavelength_cm

        # Select RJ tail region
        lam_um = _WAVE_AA * 1e-4
        mask = (lam_um > lam_range_um[0]) & (lam_um < lam_range_um[1])
        log_nu = jnp.log10(nu)
        log_lnu = jnp.log10(jnp.clip(comp, 1e-300, None))

        # Weighted linear regression in log-log space
        log_nu_sel = jnp.where(mask, log_nu, 0.0)
        log_lnu_sel = jnp.where(mask, log_lnu, 0.0)
        w = mask.astype(jnp.float64)
        n = jnp.sum(w)

        mean_x = jnp.sum(w * log_nu_sel) / n
        mean_y = jnp.sum(w * log_lnu_sel) / n
        dx = w * (log_nu_sel - mean_x)
        dy = w * (log_lnu_sel - mean_y)
        slope = jnp.sum(dx * dy) / jnp.sum(dx * dx)
        return float(slope)

    def test_cold_rj_slope(self):
        """Cold (20 K, beta=2.0): RJ slope = 2+2 = 4.0."""
        slope = self._measure_rj_slope(20.0, 2.0)
        assert abs(slope - 4.0) < 0.15, f"Cold RJ slope = {slope:.2f}, expected 4.0"

    def test_warm_rj_slope(self):
        """Warm (45 K, beta=1.5): RJ slope = 2+1.5 = 3.5."""
        slope = self._measure_rj_slope(45.0, 1.5)
        assert abs(slope - 3.5) < 0.15, f"Warm RJ slope = {slope:.2f}, expected 3.5"


class TestWienPeakLocations:
    """Wien peak of each MBB component matches analytic expectation.

    For nu^beta * B_nu(T), the peak satisfies (3+beta) = x * exp(x)/(exp(x)-1)
    where x = h*nu/(k*T).
    """

    def _expected_peak_um(self, temperature: float, beta: float) -> float:
        """Compute expected Wien peak wavelength in microns via Newton iteration."""
        x = 2.82 + beta
        for _ in range(20):
            ex = float(jnp.exp(jnp.clip(x, -100.0, 100.0)))
            f_val = x * ex / (ex - 1.0) - (3.0 + beta)
            dx = 1e-6
            ex2 = float(jnp.exp(jnp.clip(x + dx, -100.0, 100.0)))
            f2 = (x + dx) * ex2 / (ex2 - 1.0) - (3.0 + beta)
            deriv = (f2 - f_val) / dx
            x = x - f_val / deriv
        nu_peak = x * _K_BOLTZMANN * temperature / _H_PLANCK
        lam_peak_cm = _C_CGS / nu_peak
        return lam_peak_cm * 1e4  # cm -> um

    def _measured_peak_um(self, temperature: float, beta: float) -> float:
        """Measure peak wavelength from the computed MBB component."""
        comp = _modified_blackbody_component(_WAVE_AA, temperature, beta, 0.0)
        peak_aa = _WAVE_AA[jnp.argmax(comp)]
        return float(peak_aa) * 1e-4

    def test_hot_peak(self):
        """Hot (180 K, beta=1.5) peak location."""
        expected = self._expected_peak_um(180.0, 1.5)
        measured = self._measured_peak_um(180.0, 1.5)
        assert abs(measured / expected - 1.0) < 0.15, (
            f"Hot peak: measured={measured:.1f} um, expected={expected:.1f} um"
        )

    def test_warm_peak(self):
        """Warm (45 K, beta=1.5) peak location."""
        expected = self._expected_peak_um(45.0, 1.5)
        measured = self._measured_peak_um(45.0, 1.5)
        assert abs(measured / expected - 1.0) < 0.15, (
            f"Warm peak: measured={measured:.1f} um, expected={expected:.1f} um"
        )

    def test_cold_peak(self):
        """Cold (20 K, beta=2.0) peak location."""
        expected = self._expected_peak_um(20.0, 2.0)
        measured = self._measured_peak_um(20.0, 2.0)
        assert abs(measured / expected - 1.0) < 0.15, (
            f"Cold peak: measured={measured:.1f} um, expected={expected:.1f} um"
        )
