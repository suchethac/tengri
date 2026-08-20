# SPDX-License-Identifier: BSD-3-Clause
"""Tests for CB_19 CLOUDY photoionization grid backend.

Tests verify:
- Hβ → L_sun/Q_H unit conversion constant is correct
- CB19GridData NamedTuple structure
- _frac_idx clipping and monotonicity
- _interp_6d returns correct shape and is finite
- CB19Backend fallback/load errors
- predict_nebular_line_luminosities signature and output shape
- predict_nebular_continuum returns zero continuum
- param_spec registrations for nebular='cb19'
- JIT compatibility of prediction functions

Tests that require data/cb19_templates.h5 are skipped gracefully when missing.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import Fixed

pytestmark = pytest.mark.bounds

# ── Fixtures ──────────────────────────────────────────────────────
_CB19_H5 = Path(__file__).parents[2] / "data" / "cb19_templates.h5"
_SKIP_NO_H5 = pytest.mark.skipif(
    not _CB19_H5.exists(),
    reason="data/cb19_templates.h5 not found; run scripts/download_cb19_templates.py",
)


@pytest.fixture(scope="module")
def cb19_module():
    return importlib.import_module("tengri.components.nebular.cloudy_cb19")


@pytest.fixture(scope="module")
def fake_grid_data(cb19_module):
    """Build a minimal CB19GridData without an HDF5 file for interpolation tests."""
    mod = cb19_module
    n_oh, n_age, n_u, n_nh, n_co, n_dno, n_lines = 7, 5, 6, 4, 3, 3, 10
    log_oh = jnp.linspace(-5.06, -2.58, n_oh)
    log_age = jnp.linspace(6.0, 8.0, n_age)
    log_u = jnp.linspace(-4.0, -1.5, n_u)
    log_nh = jnp.linspace(1.0, 4.0, n_nh)
    log_co = jnp.linspace(-1.0, 0.15, n_co)
    dno = jnp.linspace(-0.25, 0.25, n_dno)
    # Random log-space ratios — constant 0.0 (ratio=1.0 = same as Hβ)
    log_ratios = jnp.zeros((n_oh, n_age, n_u, n_nh, n_co, n_dno, n_lines))
    waves = jnp.array([1215.67, 1549.0, 4862.68, 5007.0] + [3000.0] * (n_lines - 4))
    return mod.CB19GridData(
        log_OH_grid=log_oh,
        log_age_grid=log_age,
        log_U_grid=log_u,
        log_nH_grid=log_nh,
        log_CO_grid=log_co,
        dNO_grid=dno,
        line_wavelengths=waves,
        log_line_ratios=log_ratios,
        log_hb_per_qh=float(np.log10(mod._HB_PER_QH_LSUN)),
    )


# ── Unit conversion constants ─────────────────────────────────────
class TestHbConversionConstant:
    """The Hβ→L/Q_H conversion factor must match Osterbrock & Ferland 2006."""

    def test_hb_per_qh_lsun_value(self, cb19_module):
        """L_Hβ/Q_H = 4.78e-13 erg/photon / 3.828e33 erg/Lsun ≈ 1.249e-46 Lsun s."""
        val = cb19_module._HB_PER_QH_LSUN
        expected = 4.78e-13 / 3.828e33
        assert abs(val - expected) / expected < 1e-6, (
            f"_HB_PER_QH_LSUN={val:.4e} deviates from expected {expected:.4e}"
        )

    def test_hb_per_qh_in_expected_range(self, cb19_module):
        """Value should be ~1.249e-46 Lsun s/photon."""
        val = cb19_module._HB_PER_QH_LSUN
        assert 1.0e-46 < val < 2.0e-46, f"Implausible value: {val:.4e}"

    def test_log_oh_offset(self, cb19_module):
        """_LOG_OH_OFFSET = _LOG_OH_SOLAR - _LOG10_ZSUN ≈ -1.222."""
        from tengri.components.nebular._constants import (
            _LOG10_ZSUN,
            _LOG_OH_OFFSET,
            _LOG_OH_SOLAR,
        )

        assert abs(_LOG_OH_OFFSET - (_LOG_OH_SOLAR - _LOG10_ZSUN)) < 1e-6

    def test_solar_logz_maps_to_solar_logoh(self, cb19_module):
        """At solar metallicity (log10(Z)=_LOG10_ZSUN), log_OH should equal _LOG_OH_SOLAR."""
        from tengri.components.nebular._constants import (
            _LOG10_ZSUN,
            _LOG_OH_OFFSET,
            _LOG_OH_SOLAR,
        )

        log_oh = _LOG10_ZSUN + _LOG_OH_OFFSET
        assert abs(log_oh - _LOG_OH_SOLAR) < 1e-6


# ── _frac_idx ─────────────────────────────────────────────────────
class TestFracIdx:
    def test_exact_grid_points(self, cb19_module):
        """Exact grid point → integer index (fractional part zero)."""
        grid = jnp.array([1.0, 2.0, 3.0, 4.0])
        fi = cb19_module._frac_idx(2.0, grid)
        assert abs(float(fi) - 1.0) < 1e-6

    def test_midpoint(self, cb19_module):
        """Midpoint between grid[1] and grid[2] → fractional index 1.5."""
        grid = jnp.array([0.0, 1.0, 2.0, 3.0])
        fi = cb19_module._frac_idx(1.5, grid)
        assert abs(float(fi) - 1.5) < 1e-5

    def test_clips_below(self, cb19_module):
        """Values below grid minimum clip to index 0.0."""
        grid = jnp.array([1.0, 2.0, 3.0])
        fi = cb19_module._frac_idx(-10.0, grid)
        assert float(fi) >= 0.0

    def test_clips_above(self, cb19_module):
        """Values above grid maximum clip to last valid index (n-2 + frac=1)."""
        grid = jnp.array([1.0, 2.0, 3.0])
        fi = cb19_module._frac_idx(100.0, grid)
        assert float(fi) <= float(len(grid) - 1)


# ── _interp_6d ────────────────────────────────────────────────────
class TestInterp6D:
    def test_constant_grid_returns_constant(self, cb19_module, fake_grid_data):
        """Interpolation on a constant grid (all 0.0) returns 0.0 everywhere (limit test)."""
        grid = fake_grid_data
        grids = (
            grid.log_OH_grid,
            grid.log_age_grid,
            grid.log_U_grid,
            grid.log_nH_grid,
            grid.log_CO_grid,
            grid.dNO_grid,
        )
        vals = (-3.0, 7.0, -3.0, 2.0, -0.36, 0.0)
        result = cb19_module._interp_6d(grid.log_line_ratios, grids, vals)
        n_lines = grid.log_line_ratios.shape[-1]
        chex.assert_shape(result, (n_lines,))
        chex.assert_tree_all_finite(result)
        np.testing.assert_allclose(np.array(result), 0.0, atol=1e-5)

    def test_interior_point_is_finite_and_bounded(self, cb19_module, fake_grid_data):
        """Result must be finite at an interior point within bounds (bounds test)."""
        grid = fake_grid_data
        grids = (
            grid.log_OH_grid,
            grid.log_age_grid,
            grid.log_U_grid,
            grid.log_nH_grid,
            grid.log_CO_grid,
            grid.dNO_grid,
        )
        vals = (-3.5, 7.0, -2.5, 2.0, -0.5, 0.1)
        result = cb19_module._interp_6d(grid.log_line_ratios, grids, vals)
        n_lines = grid.log_line_ratios.shape[-1]
        chex.assert_shape(result, (n_lines,))
        chex.assert_tree_all_finite(result)


# ── CB19Backend (with mocked grid) ────────────────────────────────
class TestCB19BackendMocked:
    """Tests using a mocked HDF5 load to avoid needing the real data file."""

    @pytest.fixture
    def backend_with_fake_grid(self, cb19_module, fake_grid_data):
        """CB19Backend with the grid replaced by fake_grid_data."""
        backend = object.__new__(cb19_module.CB19Backend)
        backend.sed_type = "SSP"
        backend.imf = "Kroupa01"
        backend.mup = 100.0
        backend.hbfrac = 1.0
        backend.grid = fake_grid_data
        backend._log_hb_per_qh = fake_grid_data.log_hb_per_qh
        backend._max_neb_log_age = 8.0
        backend._qh_table = None
        backend._qh_log_met = None
        backend._qh_log_age = None
        backend._young_idx = None
        return backend

    def test_predict_line_lums_shape_and_finite(self, backend_with_fake_grid):
        """Line luminosities must be finite with correct shape (bounds test: finiteness)."""
        backend = backend_with_fake_grid
        n_age = 10
        n_lines = backend.grid.line_wavelengths.shape[0]
        ssp_weights = jnp.ones(n_age)
        ssp_log_ages = jnp.linspace(6.0, 9.0, n_age)
        waves, lums = backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-1.848
        )
        chex.assert_shape(waves, (n_lines,))
        chex.assert_shape(lums, (n_lines,))
        assert jnp.all(jnp.isfinite(lums)), f"Non-finite values: {lums}"

    def test_hb_conversion_applied(self, backend_with_fake_grid, cb19_module):
        """With all ratios=1.0 (log_ratio=0), L_line/Q_H should equal _HB_PER_QH_LSUN."""
        # The fake grid has log_line_ratios = 0 everywhere (ratio = 10^0 = 1)
        # So L_line/Q_H = ratio × _HB_PER_QH_LSUN = 1 × _HB_PER_QH_LSUN
        # For unit weight, Q_H=1 (fallback), contribution = _HB_PER_QH_LSUN
        backend = backend_with_fake_grid
        # Single young age bin with weight=1, Q_H=1 (no precomputed table → uses 1.0)
        ssp_weights = jnp.array([1.0])
        ssp_log_ages = jnp.array([7.0])  # 10 Myr — young
        _, lums = backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-1.848
        )
        expected_per_line = cb19_module._HB_PER_QH_LSUN
        # All lines should equal _HB_PER_QH_LSUN (ratio=1, Q_H=1, weight=1, fesc=0)
        np.testing.assert_allclose(
            np.array(lums),
            expected_per_line,
            rtol=1e-4,
            err_msg="Hβ conversion factor not correctly applied",
        )

    def test_fesc_suppresses_lines(self, backend_with_fake_grid):
        """neb_fesc=1.0 should suppress all lines to zero."""
        backend = backend_with_fake_grid
        ssp_weights = jnp.ones(3)
        ssp_log_ages = jnp.array([7.0, 7.5, 8.0])
        _, lums = backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-1.848, neb_fesc=1.0
        )
        np.testing.assert_allclose(np.array(lums), 0.0, atol=1e-30)

    def test_fesc_lya_only_suppresses_lya(self, backend_with_fake_grid):
        """neb_fesc_lya=1.0 with neb_fesc=0.0 suppresses Ly-alpha only."""
        backend = backend_with_fake_grid
        ssp_weights = jnp.ones(3)
        ssp_log_ages = jnp.array([7.0, 7.5, 8.0])
        waves, lums_nofesc = backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-1.848, neb_fesc=0.0, neb_fesc_lya=0.0
        )
        _, lums_lya_fesc = backend.predict_nebular_line_luminosities(
            ssp_weights, ssp_log_ages, log_z=-1.848, neb_fesc=0.0, neb_fesc_lya=1.0
        )
        lya_idx = int(jnp.argmin(jnp.abs(waves - 1215.67)))
        # Ly-alpha should be zeroed
        assert float(lums_lya_fesc[lya_idx]) == pytest.approx(0.0, abs=1e-40)
        # Other lines should be unchanged
        other_mask = np.ones(waves.shape[0], dtype=bool)
        other_mask[lya_idx] = False
        np.testing.assert_allclose(
            np.array(lums_lya_fesc[other_mask]),
            np.array(lums_nofesc[other_mask]),
            rtol=1e-5,
        )

    def test_continuum_is_zero(self, backend_with_fake_grid):
        """CB_19 has no continuum grid — predict_nebular_continuum returns zeros."""
        backend = backend_with_fake_grid
        _wave, cont = backend.predict_nebular_continuum(jnp.ones(3), jnp.ones(3), -1.848)
        assert jnp.all(cont == 0.0), "Continuum must be zero for CB_19"

    def test_predict_sed_shape_and_finite(self, backend_with_fake_grid):
        """SED output has correct shape and is finite (bounds test: finiteness)."""
        backend = backend_with_fake_grid
        n_wave = 50
        ssp_wave = jnp.linspace(1000.0, 10000.0, n_wave)
        sed = backend.predict_nebular_sed(
            jnp.ones(3), ssp_wave, jnp.array([7.0, 7.5, 8.0]), log_z=-1.848
        )
        chex.assert_shape(sed, (n_wave,))
        chex.assert_tree_all_finite(sed)

    def test_predict_sed_units_are_erg_s_hz(self, backend_with_fake_grid, cb19_module):
        """predict_nebular_sed must return erg/s/Hz, not Lsun/Hz.
        Regression test: prior to the fix, ``return neb_sed`` returned Lsun/Hz.
        The correct return is ``neb_sed * _LSUN_ERG``.  We verify by monkey-patching
        predict_nebular_line_luminosities to return a known 1-Lsun line, then checking
        the integrated SED has a peak ≈ 3.828e33 erg/s/Hz (not ≈ 1 Lsun/Hz).
        """
        from tengri.components.nebular._constants import _LSUN_ERG

        backend = backend_with_fake_grid
        # Wavelength grid centered on Hα (6564.61 Å) with line_sigma_aa > 0
        n_wave = 200
        ssp_wave = jnp.linspace(5000.0, 8000.0, n_wave)
        # A single line at 6564.61 Å with luminosity 1.0 Lsun
        _known_wave = jnp.array([6564.61])
        _known_lum = jnp.array([1.0])  # 1 Lsun
        original_fn = backend.predict_nebular_line_luminosities

        def _stub(*args, **kwargs):
            return _known_wave, _known_lum

        backend.predict_nebular_line_luminosities = _stub
        try:
            sed = backend.predict_nebular_sed(
                jnp.ones(1),
                ssp_wave,
                jnp.array([7.0]),
                log_z=-1.848,
                line_sigma_aa=10.0,
            )
        finally:
            backend.predict_nebular_line_luminosities = original_fn
        # Integrated line (trapezoid over freq) should equal 1 Lsun erg/s ≈ 3.828e33 erg/s.
        # Convert wavelength grid (Å) to frequency (Hz) for integration.
        _C_CGS = 2.998e18  # Å/s
        nu = _C_CGS / ssp_wave  # decreasing
        # sed is on ssp_wave grid; integrate |dnu| = integrate along reversed axis
        sed_np = np.array(sed)
        nu_np = np.array(nu)
        # Flip so nu is increasing, then trapezoid
        total_lum = float(np.trapezoid(sed_np[::-1], nu_np[::-1]))
        # Should be within a factor of 2 of 1 Lsun (≈ 3.828e33 erg/s)
        # A large fraction of the line power sits within our wavelength window.
        assert total_lum > 0.1 * _LSUN_ERG, (
            f"Integrated nebular luminosity {total_lum:.3e} erg/s is too low — "
            f"expected ~{_LSUN_ERG:.3e} erg/s (1 Lsun). "
            f"This likely means the Lsun→erg/s/Hz conversion is missing from "
            f"CB19Backend.predict_nebular_sed (unit bug reverted)."
        )
        assert total_lum < 10.0 * _LSUN_ERG, (
            f"Integrated nebular luminosity {total_lum:.3e} erg/s is too high — "
            f"expected ~{_LSUN_ERG:.3e} erg/s (1 Lsun)."
        )

    def test_repr(self, backend_with_fake_grid):
        assert "CB19Backend" in repr(backend_with_fake_grid)
        assert "SSP" in repr(backend_with_fake_grid)


# ── JIT compatibility ─────────────────────────────────────────────
class TestJITCompatibility:
    def test_interp_6d_jittable(self, cb19_module, fake_grid_data):
        """_interp_6d must trace without errors under jax.jit."""
        grid = fake_grid_data
        grids = (
            grid.log_OH_grid,
            grid.log_age_grid,
            grid.log_U_grid,
            grid.log_nH_grid,
            grid.log_CO_grid,
            grid.dNO_grid,
        )

        @jax.jit
        def _call(log_oh):
            return cb19_module._interp_6d(
                grid.log_line_ratios,
                grids,
                (log_oh, 7.0, -3.0, 2.0, -0.36, 0.0),
            )

        result = _call(jnp.array(-3.2))
        assert result.shape[0] == grid.log_line_ratios.shape[-1]

    def test_predict_lines_jittable(self, cb19_module, fake_grid_data):
        """predict_nebular_line_luminosities must be JIT-traceable."""
        backend = object.__new__(cb19_module.CB19Backend)
        backend.grid = fake_grid_data
        backend._log_hb_per_qh = fake_grid_data.log_hb_per_qh
        backend._max_neb_log_age = 8.0
        backend._qh_table = None
        backend._qh_log_met = None
        backend._qh_log_age = None
        backend._young_idx = None

        @jax.jit
        def _call(log_z):
            _, lums = backend.predict_nebular_line_luminosities(
                jnp.ones(5),
                jnp.linspace(6.5, 8.5, 5),
                log_z,
            )
            return lums

        result = _call(jnp.array(-1.848))
        chex.assert_tree_all_finite(result)


# ── preintegrate_for_photometry: CLOUDY-shape duck-typed surface ──
class TestPreintegrateForPhotometry:
    """The duck-typed surface CB19Backend exposes for the hybrid kernel.
    The kernel's nebular preint branch (``_kernels/hybrid.py``) reads:
    ``_has_preint_photometry``, ``_preint_continuum``, ``_preint_lines``,
    ``_line_lum_collapsed``, ``_qh_table``, ``_qh_log_met``, ``_qh_log_age``,
    ``_young_idx``, ``grid.line_wavelengths``. After
    ``preintegrate_for_photometry`` runs, all of these must exist and have
    shapes/units the kernel can consume.
    """

    @pytest.fixture
    def backend_for_preint(self, cb19_module, fake_grid_data):
        """CB19Backend stub bypassing HDF5 I/O — same pattern as backend_with_fake_grid."""
        backend = object.__new__(cb19_module.CB19Backend)
        backend.sed_type = "SSP"
        backend.imf = "Kroupa01"
        backend.mup = 100.0
        backend.hbfrac = 1.0
        backend.grid = fake_grid_data
        backend._log_hb_per_qh = fake_grid_data.log_hb_per_qh
        backend._max_neb_log_age = 8.0
        backend._qh_table = None
        backend._qh_log_met = None
        backend._qh_log_age = None
        backend._young_idx = None
        backend._preint_continuum = None
        backend._preint_lines = None
        backend._line_lum_collapsed = None
        backend._has_preint_photometry = False
        return backend

    @pytest.fixture
    def synthetic_filters(self):
        fw = [
            np.linspace(3000.0, 5000.0, 32),
            np.linspace(5000.0, 7000.0, 32),
            np.linspace(7000.0, 9000.0, 32),
        ]
        ft = [np.exp(-0.5 * ((w - w.mean()) / (0.3 * np.ptp(w))) ** 2) for w in fw]
        return fw, ft

    def test_sets_flag_and_attributes(self, backend_for_preint, synthetic_filters):
        """After call, the duck-typed attributes must exist and be populated."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        assert backend_for_preint._has_preint_photometry is True
        assert backend_for_preint._preint_continuum is not None
        assert backend_for_preint._preint_lines is not None
        assert backend_for_preint._line_lum_collapsed is not None

    def test_continuum_shape_and_zeros(self, backend_for_preint, synthetic_filters):
        """CB19 has no nebular continuum — _preint_continuum.phot must be all zeros
        with shape (n_met, n_age, n_logU, n_filt)."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        cont = backend_for_preint._preint_continuum
        n_met = backend_for_preint.grid.log_OH_grid.shape[0]
        n_age = backend_for_preint.grid.log_age_grid.shape[0]
        n_u = backend_for_preint.grid.log_U_grid.shape[0]
        assert cont.phot.shape == (n_met, n_age, n_u, len(fw))
        assert bool(jnp.all(cont.phot == 0.0))

    def test_axis0_relabeled_to_absolute_logz(self, backend_for_preint, synthetic_filters):
        """Axis 0 of the preint surface must be absolute log10(Z), not log10(O/H).
        The kernel queries with ``_gas_z`` (absolute log10(Z)); the relabeling
        makes ``log_OH_grid - _LOG_OH_OFFSET`` land on the right coordinate.
        """
        from tengri.components.nebular._constants import _LOG_OH_OFFSET

        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        cont = backend_for_preint._preint_continuum
        expected = backend_for_preint.grid.log_OH_grid - _LOG_OH_OFFSET
        np.testing.assert_allclose(np.array(cont.axes[0]), np.array(expected), atol=1e-6)

    def test_line_lum_collapsed_shape_and_units(self, backend_for_preint, synthetic_filters):
        """``_line_lum_collapsed`` must have shape (n_met, n_age, n_logU, n_lines)
        and contain log10(L_line/Q_H) [Lsun·s/photon] — i.e. log_line_ratios + log_hb_per_qh."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        n_met = backend_for_preint.grid.log_OH_grid.shape[0]
        n_age = backend_for_preint.grid.log_age_grid.shape[0]
        n_u = backend_for_preint.grid.log_U_grid.shape[0]
        n_lines = backend_for_preint.grid.line_wavelengths.shape[0]
        chex.assert_shape(backend_for_preint._line_lum_collapsed, (n_met, n_age, n_u, n_lines))
        # fake grid is all 0.0 ratios → collapsed value = log_hb_per_qh
        np.testing.assert_allclose(
            np.array(backend_for_preint._line_lum_collapsed),
            backend_for_preint._log_hb_per_qh,
            atol=1e-6,
        )

    def test_line_filter_weights_shape(self, backend_for_preint, synthetic_filters):
        """``_preint_lines.line_filter_weights`` must be (n_lines, n_filt) finite floats."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        weights = backend_for_preint._preint_lines.line_filter_weights
        n_lines = backend_for_preint.grid.line_wavelengths.shape[0]
        assert weights.shape == (n_lines, len(fw))
        chex.assert_tree_all_finite(weights)

    def test_fixed_collapses_axis(self, backend_for_preint, synthetic_filters):
        """Passing ``fixed={2: -3.0}`` must drop the log_U axis from line_lum and
        from _preint_lines.axes."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28, fixed={2: -3.0})
        n_met = backend_for_preint.grid.log_OH_grid.shape[0]
        n_age = backend_for_preint.grid.log_age_grid.shape[0]
        n_lines = backend_for_preint.grid.line_wavelengths.shape[0]
        chex.assert_shape(backend_for_preint._line_lum_collapsed, (n_met, n_age, n_lines))
        assert len(backend_for_preint._preint_lines.axes) == 2

    def test_preint_lines_agree_with_runtime_at_grid_point(
        self, backend_for_preint, synthetic_filters
    ):
        """Numerical equivalence: at an exact grid point, the preint-surface
        line luminosities must match what ``predict_nebular_line_luminosities``
        would compute with the same gas conditions and a unit Q_H input.
        At an exact grid point, triweight interpolation reduces to the on-grid
        value, so any disagreement here would indicate a bookkeeping bug in the
        units/relabeling rather than an interp-method difference.
        """
        from tengri.utils.grid_interp import interp_nd_triweight

        fw, ft = synthetic_filters
        b = backend_for_preint
        b.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        # Pick an exact interior grid point — last met index, middle age, middle U.
        log_oh_idx, age_idx, u_idx = 4, 2, 3
        log_oh = float(b.grid.log_OH_grid[log_oh_idx])
        log_age = float(b.grid.log_age_grid[age_idx])
        log_U = float(b.grid.log_U_grid[u_idx])
        from tengri.components.nebular._constants import _LOG_OH_OFFSET

        log_z_abs = log_oh - _LOG_OH_OFFSET
        # Preint path: interp the relabeled grid at the absolute-Z coordinate.
        log_lum_per_qh_preint = interp_nd_triweight(
            b._line_lum_collapsed,
            b._preint_lines.axes,
            b._preint_lines.edges,
            (jnp.array(log_z_abs), jnp.array(log_age), jnp.array(log_U)),
        )
        # Runtime path: the on-grid value of log_line_ratios + log_hb_per_qh
        # (the only collapses are on axes 3,4,5 at exactly the defaults — but
        # since the fake grid is constant across those axes, the collapse is
        # value-preserving).
        on_grid = b.grid.log_line_ratios[log_oh_idx, age_idx, u_idx, :, :, :, :]
        # Average over the axes 3,4,5 should equal a single grid value (constant fake grid).
        runtime_log_lum = float(on_grid.mean()) + b._log_hb_per_qh
        np.testing.assert_allclose(
            np.array(log_lum_per_qh_preint),
            runtime_log_lum * np.ones_like(np.array(log_lum_per_qh_preint)),
            atol=1e-5,
        )

    def test_kernel_consumer_path_traces_under_jit(self, backend_for_preint, synthetic_filters):
        """Mirror the kernel's line-projection path and verify it's JIT-traceable.
        This is a structural test of the duck-typed surface: the exact code path
        the kernel executes must compose with ``jax.jit`` without retracing or
        raising.
        """
        from tengri.utils.grid_interp import interp_nd_triweight

        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        line_lum = backend_for_preint._line_lum_collapsed
        line_axes = backend_for_preint._preint_lines.axes
        line_edges = backend_for_preint._preint_lines.edges
        line_weights = backend_for_preint._preint_lines.line_filter_weights

        @jax.jit
        def _line_phot(log_z, log_age, log_U, qh):
            log_lum_per_qh = interp_nd_triweight(
                line_lum, line_axes, line_edges, (log_z, log_age, log_U)
            )
            total_line_lum = qh * (10.0**log_lum_per_qh)
            return jnp.einsum("l,lf->f", total_line_lum, line_weights)

        out = _line_phot(jnp.array(-2.0), jnp.array(7.0), jnp.array(-3.0), jnp.array(1e54))
        assert out.shape == (len(fw),)
        chex.assert_tree_all_finite(out)


# ── Missing HDF5 file raises FileNotFoundError ────────────────────
class TestMissingH5:
    def test_load_raises_file_not_found(self, cb19_module):
        """load_cb19_grid raises FileNotFoundError with helpful message."""
        with pytest.raises(FileNotFoundError, match=r"cb19_templates\.h5"):
            cb19_module.load_cb19_grid(filepath="/nonexistent/path/cb19_templates.h5")

    def test_backend_init_raises_file_not_found(self, cb19_module):
        """CB19Backend init raises FileNotFoundError when HDF5 missing."""
        with pytest.raises(FileNotFoundError, match=r"cb19_templates\.h5"):
            cb19_module.CB19Backend(grid_path="/nonexistent/path/cb19_templates.h5")


# ── param_spec integration ────────────────────────────────────────
class TestParamSpec:
    _CLOUDY_GRID = Path(__file__).parents[2] / "data" / "cloudy_grid_bpss.h5"
    _SKIP_NO_GRID = pytest.mark.skipif(
        not (Path(__file__).parents[2] / "data" / "cloudy_grid_bpss.h5").exists(),
        reason="cloudy_grid_bpss.h5 not found",
    )

    def test_cb19_nebular_mode_registers_base_params(self):
        """nebular='cb19' registers the standard neb_logU, neb_logZ_gas params."""
        from tengri.parameters.parameters import Parameters

        spec = Parameters(nebular="cb19")
        assert "neb_logU" in spec._param_registry
        assert "neb_logZ_gas" in spec._param_registry

    def test_cb19_nebular_mode_registers_cb19_params(self):
        """nebular='cb19' registers neb_log_nH, neb_co, neb_dno, neb_hbfrac."""
        from tengri.parameters.parameters import Parameters

        spec = Parameters(nebular="cb19")
        for pname in ("neb_log_nH", "neb_co", "neb_dno", "neb_hbfrac"):
            assert pname in spec._param_registry, f"Missing param: {pname}"

    @_SKIP_NO_GRID
    def test_cloudy_mode_does_not_register_cb19_params(self):
        """nebular='cloudy' should NOT include the CB_19-specific extra params."""
        from tengri.parameters.parameters import Parameters

        spec = Parameters(nebular="cloudy", cloudy_grid_path=str(self._CLOUDY_GRID))
        for pname in ("neb_log_nH", "neb_co", "neb_dno", "neb_hbfrac"):
            assert pname not in spec._param_registry, (
                f"Param {pname} should not be in cloudy mode registry"
            )

    def test_cb19_param_defaults(self):
        """CB_19 params should have physically sensible default values."""
        from tengri.parameters.parameters import Fixed, Parameters

        spec = Parameters(nebular="cb19")
        defaults = spec._defaults
        assert isinstance(defaults["neb_log_nH"], Fixed)
        assert abs(float(defaults["neb_log_nH"].value) - 2.0) < 1e-6  # n_H=100
        assert isinstance(defaults["neb_co"], Fixed)
        assert abs(float(defaults["neb_co"].value) - (-0.36)) < 1e-6  # near-solar
        assert isinstance(defaults["neb_hbfrac"], Fixed)
        assert abs(float(defaults["neb_hbfrac"].value) - 1.0) < 1e-6  # radiation-bounded


# ── Real HDF5 file tests (skipped if file missing) ────────────────
@_SKIP_NO_H5
class TestCB19WithRealH5:
    def test_load_default_group(self, cb19_module):
        """Load SSP/Kroupa01/mu100 group and verify shapes."""
        grid = cb19_module.load_cb19_grid(sed_type="SSP", imf="Kroupa01", mup=100.0, hbfrac=1.0)
        assert grid.log_line_ratios.ndim == 7
        n_oh, _n_age, n_u, n_nh, _n_co, _n_dno, _n_lines = grid.log_line_ratios.shape
        assert n_oh == 7
        assert n_u == 6
        assert n_nh == 4

    def test_line_wavelengths_match_vacuum(self, cb19_module):
        """Verify key vacuum wavelengths are present (Hβ=4862.68 Å, Hα=6564.61 Å)."""
        grid = cb19_module.load_cb19_grid()
        waves = np.array(grid.line_wavelengths)
        assert np.any(np.abs(waves - 4862.68) < 1.0), "Hβ not found"
        assert np.any(np.abs(waves - 6564.61) < 1.0), "Hα not found"

    def test_hbfrac_snap_warning(self, cb19_module):
        """Requesting an unusual HbFrac value triggers a UserWarning."""
        with pytest.warns(UserWarning, match="hbfrac"):
            cb19_module.load_cb19_grid(hbfrac=0.42)

    def test_no_all_nan_slices_at_solar(self, cb19_module):
        """At solar metallicity and fiducial parameters, most lines should be finite."""
        grid = cb19_module.load_cb19_grid()
        # solar-ish OH index (≈ -3.07)
        oh_idx = int(jnp.argmin(jnp.abs(grid.log_OH_grid - (-3.07))))
        age_idx = int(jnp.argmin(jnp.abs(grid.log_age_grid - 7.0)))  # 10 Myr
        u_idx = int(jnp.argmin(jnp.abs(grid.log_U_grid - (-3.0))))
        slice_ = np.array(grid.log_line_ratios[oh_idx, age_idx, u_idx, 1, 1, 1, :])
        n_finite = np.sum(np.isfinite(slice_))
        n_total = len(slice_)
        assert n_finite > n_total * 0.5, (
            f"Too many NaN lines at solar/10Myr/logU=-3: {n_total - n_finite}/{n_total}"
        )

    def test_hb_ratio_is_unity(self, cb19_module):
        """Hβ itself should have ratio = 1.0 (log10 = 0.0) by definition."""
        grid = cb19_module.load_cb19_grid()
        waves = np.array(grid.line_wavelengths)
        hb_idx = np.argmin(np.abs(waves - 4862.68))
        hb_log_ratio = float(grid.log_line_ratios[3, 10, 2, 1, 1, 1, hb_idx])
        assert abs(hb_log_ratio - 0.0) < 0.05, f"Hβ log10(ratio) = {hb_log_ratio:.3f} ≠ 0.0"


# ── _init_nebular dispatch regression (issue #361) ────────────────
@_SKIP_NO_H5
class TestSEDModelInitNebularDispatch:
    """``neb={'type': 'cb19'}`` must instantiate ``CB19Backend``.

    Before the fix, ``_init_nebular`` had no ``elif "cb19":`` branch, so the
    requested mode silently fell through to ``BakedInBackend()`` and every
    ``Prediction.lines.*`` accessor returned NaN.  See issue #361.
    """

    def test_cb19_mode_uses_cb19_backend(self, ssp_data_wne):
        """Building a model with neb='cb19' should wire CB19Backend, not BakedInBackend."""
        import tengri
        from tengri.components.nebular import BakedInBackend, CB19Backend

        model = tengri.SEDModel.build(
            ssp_data_wne,
            sfh={
                "type": "tsnorm",
                "*": tengri.FIXED,
                "log_peak_sfr": 1.0,
                "peak_lbt_gyr": 2.0,
                "width_gyr": 1.0,
                "skew": 0.2,
                "trunc": 3.0,
                "logzsol": -0.1,
            },
            dust={
                "law": "power_law",
                "type": "two_component",
                "*": tengri.FIXED,
                "tau_bc": 0.2,
                "tau_diff": 0.1,
                "slope": -0.7,
            },
            neb={"type": "cb19", "*": tengri.FIXED},
            redshift=Fixed(0.1),
        )
        assert model.spec.nebular_mode == "cb19"
        assert isinstance(model._nebular_backend, CB19Backend)
        assert not isinstance(model._nebular_backend, BakedInBackend)
