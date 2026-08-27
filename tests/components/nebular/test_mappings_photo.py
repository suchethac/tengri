# SPDX-License-Identifier: BSD-3-Clause
"""Unit tests for MappingsPhotoStellarBackend and MappingsPhotoAGNBackend.

These tests run without the actual HDF5 file by patching the load functions
with synthetic grids that exercise the full interpolation and Q_H pipeline.
"""

from __future__ import annotations

from unittest.mock import patch

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular._shared import _interp_index_weight, compute_qh
from tengri.components.nebular.mappings_photo import (
    MappingsAGNGridData,
    MappingsPhotoAGNBackend,
    MappingsPhotoStellarBackend,
    MappingsStellarGridData,
    _interp_stellar_grid,
    _log_z_abs_to_zo,
)
from tests._bounds import assert_non_negative

pytestmark = pytest.mark.bounds

# ── Helpers: synthetic grids for offline testing ──────────────────

_LOG10_ZSUN = -1.8477116556169435
_LSUN_ERG = 3.828e33


def _make_stellar_grid(n_z=3, n_a=4, n_s=2, n_u=4, n_n=2, n_lines=5) -> MappingsStellarGridData:
    """Build a minimal synthetic stellar grid with known constant values."""
    zo_axis = jnp.array([0.1, 0.5, 1.0])[:n_z]  # ζ_O
    logU_axis = jnp.linspace(-4.0, -1.0, n_u)
    log_age_yr_axis = jnp.array([6.0, 6.5, 7.0, 7.5])[:n_a]
    logn_axis = jnp.array([1.0, 3.0])[:n_n]
    sfh_labels = ["cont", "inst"][:n_s]
    sfh_idx_inst = sfh_labels.index("inst") if "inst" in sfh_labels else 0
    sfh_idx_cont = sfh_labels.index("cont") if "cont" in sfh_labels else 0

    shape = (n_z, n_a, n_s, n_u, n_n)
    # logHB_per_logq = -12.32 everywhere (Hβ case B value)
    logHB_per_logq = jnp.full(shape, -12.32)
    # All lines have ratio = 1.0 relative to Hβ (simplifies expected values)
    line_ratios = jnp.ones((*shape, n_lines))
    line_wavelengths = jnp.array([1215.67, 4862.68, 5008.24, 6564.61, 3728.0])[:n_lines]

    return MappingsStellarGridData(
        line_wavelengths=line_wavelengths,
        zo_axis=zo_axis,
        logU_axis=logU_axis,
        log_age_yr_axis=log_age_yr_axis,
        logn_axis=logn_axis,
        logHB_per_logq=logHB_per_logq,
        line_ratios=line_ratios,
    )


def _make_agn_grid(n_z=3, n_m=2, n_e=2, n_u=4, n_n=2, n_lines=5) -> MappingsAGNGridData:
    zo_axis = jnp.array([0.1, 0.5, 1.0])[:n_z]
    logU_axis = jnp.linspace(-4.0, -1.0, n_u)
    logmbh_axis = jnp.array([6.0, 8.0])[:n_m]
    logedd_axis = jnp.array([-2.0, -0.5])[:n_e]
    logn_axis = jnp.array([2.0, 4.0])[:n_n]

    shape = (n_z, n_m, n_e, n_u, n_n)
    # logHB_per_lum = log10(L_Hβ / L_ion) — typical AGN value ~ -2.4
    logHB_per_lum = jnp.full(shape, -2.4)
    line_ratios = jnp.ones((*shape, n_lines))
    line_wavelengths = jnp.array([1215.67, 4862.68, 5008.24, 6564.61, 3728.0])[:n_lines]

    return MappingsAGNGridData(
        line_wavelengths=line_wavelengths,
        zo_axis=zo_axis,
        logU_axis=logU_axis,
        logmbh_axis=logmbh_axis,
        logedd_axis=logedd_axis,
        logn_axis=logn_axis,
        logHB_per_lum=logHB_per_lum,
        line_ratios=line_ratios,
    )


# ── Unit helpers ──────────────────────────────────────────────────


class TestInterpIndexWeight:
    def test_exact_grid_point(self):
        # searchsorted("right") on 1.0 in [0,1,2,3] → pos=2, minus 1 → idx=1
        # w = (1.0 - grid[1]) / (grid[2] - grid[1]) = 0/1 = 0.0
        grid = jnp.array([0.0, 1.0, 2.0, 3.0])
        idx, w = _interp_index_weight(1.0, grid)
        assert int(idx) == 1
        assert float(w) == pytest.approx(0.0, abs=1e-6)

    def test_midpoint(self):
        grid = jnp.array([0.0, 2.0, 4.0])
        idx, w = _interp_index_weight(1.0, grid)
        assert int(idx) == 0
        assert float(w) == pytest.approx(0.5, abs=1e-6)

    def test_clamp_below(self):
        grid = jnp.array([1.0, 2.0, 3.0])
        idx, w = _interp_index_weight(0.0, grid)
        assert int(idx) == 0
        assert float(w) == pytest.approx(0.0, abs=1e-6)

    def test_clamp_above(self):
        grid = jnp.array([1.0, 2.0, 3.0])
        idx, w = _interp_index_weight(5.0, grid)
        # Clamped to 3.0, which is the last grid point
        assert int(idx) == 1  # clips to len-2
        assert float(w) == pytest.approx(1.0, abs=1e-6)


class TestLogZToZo:
    def test_solar(self):
        # log_z = LOG10_ZSUN → ζ_O = 1.0
        zo = _log_z_abs_to_zo(_LOG10_ZSUN)
        assert float(zo) == pytest.approx(1.0, rel=1e-4)

    def test_half_solar(self):
        import math

        zo = _log_z_abs_to_zo(_LOG10_ZSUN + math.log10(0.5))
        assert float(zo) == pytest.approx(0.5, rel=1e-4)


class TestComputeQH:
    def test_flat_spectrum_above_lyman_limit(self):
        # All flux above Lyman limit — Q_H should be zero
        wave = jnp.linspace(1000.0, 10000.0, 500)
        flux = jnp.ones_like(wave)
        qh = compute_qh(wave, flux)
        assert float(qh) == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_below_lyman_limit(self):
        # Flux below 912 Å → Q_H > 0
        wave = jnp.linspace(100.0, 2000.0, 1000)
        flux = jnp.ones_like(wave)
        qh = compute_qh(wave, flux)
        assert float(qh) > 0.0


# ── Stellar grid interpolation ────────────────────────────────────


class TestInterpStellarGrid:
    def setup_method(self):
        self.grid = _make_stellar_grid()

    def test_constant_grid_returns_constant(self):
        """With constant grid values, any query should return the constant."""
        # logHB_per_logq is constant at -12.32
        val = _interp_stellar_grid(
            self.grid.logHB_per_logq,
            self.grid,
            zo_val=0.5,
            log_age_yr_val=6.5,
            logU_val=-3.0,
            logn_val=2.0,
            sfh_idx=0,
        )
        assert float(val) == pytest.approx(-12.32, abs=1e-4)

    def test_sfh_axis_selection(self):
        """Different sfh_idx should select different SFH slices."""
        # With constant grid, both give same result — but the selection path runs
        val0 = _interp_stellar_grid(
            self.grid.logHB_per_logq,
            self.grid,
            0.5,
            6.5,
            -3.0,
            2.0,
            sfh_idx=0,
        )
        val1 = _interp_stellar_grid(
            self.grid.logHB_per_logq,
            self.grid,
            0.5,
            6.5,
            -3.0,
            2.0,
            sfh_idx=1,
        )
        # Both equal the constant value since the grid is uniform
        assert float(val0) == pytest.approx(-12.32, abs=1e-4)
        assert float(val1) == pytest.approx(-12.32, abs=1e-4)

    def test_clipping_at_boundaries(self):
        """Extrapolation beyond grid edges should clip (not crash)."""
        val = _interp_stellar_grid(
            self.grid.logHB_per_logq,
            self.grid,
            zo_val=100.0,  # way above grid max
            log_age_yr_val=100.0,
            logU_val=0.0,
            logn_val=10.0,
            sfh_idx=0,
        )
        assert jnp.isfinite(val)


# ── MappingsPhotoStellarBackend (patched load) ────────────────────


def _make_stellar_backend_with_qh() -> MappingsPhotoStellarBackend:
    """Build backend with synthetic grid and injected Q_H table."""
    with patch(
        "tengri.components.nebular.mappings_photo._load_stellar_grid",
        return_value=_make_stellar_grid(),
    ):
        backend = MappingsPhotoStellarBackend.__new__(MappingsPhotoStellarBackend)
        backend.model = "sb99"
        backend.density = "cpr"
        backend.sfh_mode = "inst"
        backend.grid = _make_stellar_grid()
        backend._sfh_idx = 1  # "inst"

        # Inject a trivial Q_H table: Q_H = 1e50 photons/s/Msun everywhere
        n_met, n_age = 3, 4
        backend._qh_table = jnp.full((n_met, n_age), 1e50)
        backend._qh_log_met = jnp.linspace(-3.0, -1.0, n_met)
        backend._qh_log_age = jnp.array([6.0, 6.5, 7.0, 7.5])
        backend._young_idx = np.arange(n_age)  # all ages young for this test

    return backend


class TestMappingsPhotoStellarBackend:
    def setup_method(self):
        self.backend = _make_stellar_backend_with_qh()

    def test_predict_returns_correct_shape(self):
        n_age = 4
        ssp_weights = jnp.ones(n_age)
        ssp_log_ages = jnp.array([6.0, 6.5, 7.0, 7.5])
        wave, lum = self.backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-2.0
        )
        chex.assert_shape(wave, (5,))
        chex.assert_shape(lum, (5,))

    def test_luminosities_positive(self):
        ssp_weights = jnp.ones(4)
        ssp_log_ages = jnp.array([6.0, 6.5, 7.0, 7.5])
        _, lum = self.backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-2.0
        )
        assert_non_negative(lum, name="lum")

    def test_fesc_unity_gives_zero(self):
        """Full escape fraction should zero out all line luminosities."""
        ssp_weights = jnp.ones(4)
        ssp_log_ages = jnp.array([6.0, 6.5, 7.0, 7.5])
        _, lum = self.backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-2.0, neb_fesc=1.0
        )
        assert jnp.allclose(lum, 0.0, atol=1e-30)

    def test_lya_fesc_reduces_only_lya(self):
        """neb_fesc_lya=1.0 should zero Lya but not other lines."""
        ssp_weights = jnp.ones(4)
        ssp_log_ages = jnp.array([6.0, 6.5, 7.0, 7.5])
        _, lum_no_lya_fesc = self.backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-2.0, neb_fesc_lya=0.0
        )
        _, lum_lya_zero = self.backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-2.0, neb_fesc_lya=1.0
        )
        lya_idx = int(jnp.argmin(jnp.abs(self.backend.grid.line_wavelengths - 1215.67)))
        assert float(lum_lya_zero[lya_idx]) == pytest.approx(0.0, abs=1e-30)
        # Other lines unchanged
        for i in range(5):
            if i != lya_idx:
                assert float(lum_lya_zero[i]) == pytest.approx(float(lum_no_lya_fesc[i]), rel=1e-5)

    def test_luminosity_scales_with_weight(self):
        """Doubling SSP weights should double line luminosities."""
        ssp_log_ages = jnp.array([6.0, 6.5, 7.0, 7.5])
        _, lum1 = self.backend.predict_nebular_line_luminosities(
            jnp.ones(4), ssp_log_ages, log_z=-2.0
        )
        _, lum2 = self.backend.predict_nebular_line_luminosities(
            2.0 * jnp.ones(4), ssp_log_ages, log_z=-2.0
        )
        assert jnp.allclose(lum2, 2.0 * lum1, rtol=1e-5)

    def test_predict_nebular_sed_shape(self):
        """predict_nebular_sed returns array matching ssp_wave shape."""
        ssp_wave = jnp.linspace(1000.0, 10000.0, 200)
        ssp_weights = jnp.ones(4)
        ssp_log_ages = jnp.array([6.0, 6.5, 7.0, 7.5])
        sed = self.backend.predict_nebular_sed(
            ssp_weights, ssp_wave, ssp_log_ages, log_z=-2.0, line_sigma_aa=5.0
        )
        chex.assert_shape(sed, (200,))
        chex.assert_tree_all_finite(sed)

    def test_predict_nebular_sed_nonnegative(self):
        ssp_wave = jnp.linspace(1000.0, 10000.0, 200)
        ssp_weights = jnp.ones(4)
        ssp_log_ages = jnp.array([6.0, 6.5, 7.0, 7.5])
        sed = self.backend.predict_nebular_sed(ssp_weights, ssp_wave, ssp_log_ages, log_z=-2.0)
        assert_non_negative(sed, name="sed")

    def test_logZ_gas_ties_to_stellar_when_none(self):
        """neb_logZ_gas=None should give same result as explicit log_z."""
        ssp_weights = jnp.ones(4)
        ssp_log_ages = jnp.array([6.0, 6.5, 7.0, 7.5])
        log_z = -2.0
        _, lum_none = self.backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=log_z, neb_logZ_gas=None
        )
        _, lum_explicit = self.backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=log_z, neb_logZ_gas=log_z
        )
        assert jnp.allclose(lum_none, lum_explicit, rtol=1e-6)


# ── MappingsPhotoAGNBackend (patched load) ────────────────────────


def _make_agn_backend() -> MappingsPhotoAGNBackend:
    backend = MappingsPhotoAGNBackend.__new__(MappingsPhotoAGNBackend)
    backend.density = "cpr"
    backend.grid = _make_agn_grid()
    return backend


class TestMappingsPhotoAGNBackend:
    def setup_method(self):
        self.backend = _make_agn_backend()

    def test_predict_returns_correct_shape(self):
        wave, lum = self.backend.predict_agn_line_luminosities(
            agn_log_l_ion_erg=45.0, neb_logZ_gas=-2.0
        )
        chex.assert_shape(wave, (5,))
        chex.assert_shape(lum, (5,))

    def test_luminosities_positive(self):
        _, lum = self.backend.predict_agn_line_luminosities(agn_log_l_ion_erg=45.0)
        assert_non_negative(lum, name="lum")

    def test_scales_linearly_with_l_ion(self):
        """Doubling L_ion should double all luminosities."""
        import math

        _, lum1 = self.backend.predict_agn_line_luminosities(agn_log_l_ion_erg=50.0)
        _, lum2 = self.backend.predict_agn_line_luminosities(
            agn_log_l_ion_erg=50.0 + math.log10(2)
        )
        assert jnp.allclose(lum2, 2.0 * lum1, rtol=1e-5)

    def test_fesc_unity_gives_zero(self):
        _, lum = self.backend.predict_agn_line_luminosities(agn_log_l_ion_erg=45.0, neb_fesc=1.0)
        assert jnp.allclose(lum, 0.0, atol=1e-30)

    def test_finite_at_grid_edges(self):
        """Queries at grid boundaries should return finite values (clipping)."""
        _, lum = self.backend.predict_agn_line_luminosities(
            agn_log_l_ion_erg=45.0,
            neb_logZ_gas=_LOG10_ZSUN + 1.0,  # well above grid max
            neb_logU=-0.01,  # near grid edge
            agn_logmbh=10.0,  # above grid max
            agn_logedd=0.0,  # above grid max
            neb_logn=10.0,  # above grid max
        )
        chex.assert_tree_all_finite(lum)

    def test_expected_luminosity_magnitude(self):
        """Sanity check: typical L_Hβ from a log L_ion=45 AGN."""
        # With ratio=1, logHB_per_lum=-2.4, agn_log_l_ion_erg=45.0:
        # L = 10^{-2.4} × 10^45 ≈ 10^{42.6} erg/s
        _, lum = self.backend.predict_agn_line_luminosities(agn_log_l_ion_erg=45.0)
        # All lines = 1.0 × Hβ so each should be ~10^{42.6} erg/s
        assert float(lum[0]) > 1e39  # sanity floor
        assert float(lum[0]) < 1e48  # sanity ceiling


# ── Q_H sanitization regression (commit 3996aba parity) ───────────


class _MockSSPData:
    """Minimal SSP data object for _precompute_qh tests."""

    def __init__(self, ssp_wave, ssp_flux, n_met=3, n_age=4):
        self.ssp_wave = ssp_wave
        self.ssp_flux = ssp_flux  # (n_met, n_age, n_wave)
        self.ssp_lgmet = jnp.linspace(-3.0, -1.0, n_met)
        self.ssp_lg_age_gyr = jnp.array([-3.0, -2.5, -2.0, -1.5])[:n_age]  # log(age/Gyr)


class TestMappingsQHSanitization:
    """Regression: _precompute_qh must sanitize Inf/NaN from pure-SSP overflow.

    cloudy_grid.py received the same fix in commit 3996aba. These tests verify
    mappings_photo.py is now consistent.
    """

    def _make_backend_shell(self) -> MappingsPhotoStellarBackend:
        """Backend shell without a loaded grid (enough for _precompute_qh)."""
        backend = MappingsPhotoStellarBackend.__new__(MappingsPhotoStellarBackend)
        backend.model = "sb99"
        backend.density = "cpr"
        backend.sfh_mode = "inst"
        backend.grid = _make_stellar_grid()
        backend._sfh_idx = 1
        backend._qh_table = None
        backend._qh_log_met = None
        backend._qh_log_age = None
        backend._young_idx = None
        return backend

    def test_qh_table_finite_with_inf_uv_flux(self):
        """_qh_table must be finite even when raw Q_H overflows to Inf.

        Simulates a pure-SSP row with extreme UV flux below the Lyman limit —
        the kind of flux that causes `compute_qh` to return Inf via
        `jnp.trapezoid(L_nu / (h * nu))` accumulation.
        """
        n_met, n_age, n_wave = 3, 4, 200
        ssp_wave = jnp.linspace(100.0, 10000.0, n_wave)

        # Normal flux everywhere, but one SSP row has extreme UV (→ Inf Q_H)
        ssp_flux = jnp.ones((n_met, n_age, n_wave)) * 1e30
        # Inject a truly huge UV spike on one row to guarantee Inf from trapezoid
        ssp_flux = ssp_flux.at[0, 0, :50].set(jnp.inf)  # n_wave[:50] < 912 Å region

        ssp_data = _MockSSPData(ssp_wave, ssp_flux, n_met=n_met, n_age=n_age)
        backend = self._make_backend_shell()
        backend._precompute_qh(ssp_data)

        assert backend._qh_table is not None
        assert jnp.all(jnp.isfinite(backend._qh_table)), (
            "_qh_table contains Inf/NaN after precomputation with extreme UV flux; "
            "jnp.where(jnp.isfinite(qh_raw), qh_raw, 0.0) sanitization is missing"
        )

    def test_qh_table_nonnegative(self):
        """_qh_table values must be ≥ 0 for any SSP input."""
        n_met, n_age, n_wave = 3, 4, 200
        ssp_wave = jnp.linspace(500.0, 10000.0, n_wave)
        ssp_flux = jnp.abs(jnp.ones((n_met, n_age, n_wave)))

        ssp_data = _MockSSPData(ssp_wave, ssp_flux, n_met=n_met, n_age=n_age)
        backend = self._make_backend_shell()
        backend._precompute_qh(ssp_data)

        assert_non_negative(backend._qh_table, name="output")

    def test_get_qh_at_nonnegative_with_normal_table(self):
        """_get_qh_at must return ≥ 0 after bilinear interpolation."""
        backend = _make_stellar_backend_with_qh()

        # Query at multiple (log_z, log_age) points including boundary extrapolation
        test_points = [
            (-3.5, 5.5),  # below both grid edges
            (-2.0, 6.5),  # mid-grid
            (-0.5, 8.0),  # above both grid edges
        ]
        for log_z, log_age in test_points:
            result = backend._get_qh_at(log_z, log_age)
            assert float(result) >= 0.0, (
                f"_get_qh_at({log_z}, {log_age}) = {result} < 0; "
                "jnp.maximum(..., 0.0) floor is missing"
            )

    def test_get_qh_at_zero_when_table_has_zero_entries(self):
        """When _qh_table is all zeros, _get_qh_at must return 0.0."""
        backend = _make_stellar_backend_with_qh()
        backend._qh_table = jnp.zeros_like(backend._qh_table)

        result = backend._get_qh_at(-2.0, 6.5)
        assert float(result) == pytest.approx(0.0, abs=1e-30)


# ── __init__.py exports ───────────────────────────────────────────


def test_module_exports():
    """Ensure new backends are exported from the nebular subpackage."""
    from tengri.components.nebular import MappingsPhotoAGNBackend, MappingsPhotoStellarBackend

    assert MappingsPhotoStellarBackend is not None
    assert MappingsPhotoAGNBackend is not None
