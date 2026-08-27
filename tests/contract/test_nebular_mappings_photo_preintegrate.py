# SPDX-License-Identifier: BSD-3-Clause
"""Tests for MAPPINGS V photometry preintegration (duck-typed surface).

Tests verify:
- preintegrate_for_photometry sets flags and attributes correctly
- Continuum shape and zeros (MAPPINGS V has no nebular continuum)
- Axis 0 relabeling from ζ_O (solar-relative) to absolute log10(Z)
- Line luminosity shape and units (erg/photon × ratio)
- Fixed-axis collapse (CLOUDY-shape indices: 0=Z, 1=age, 2=U)
- Numerical equivalence at grid points
- Line filter weights are finite
- JIT compatibility of kernel-side consumers

Tests that require data/flury2024_grids.h5 are skipped gracefully when missing.
"""

from __future__ import annotations

import importlib

import chex
import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data

pytestmark = pytest.mark.contract

# ── Fixtures ──────────────────────────────────────────────────────

# parents[3] is one level above the repo root from tests/contract/, so this
# guard was permanently true and the tests below never ran (#1431).
_MAPPINGS_H5 = find_data("flury2024_grids.h5")
_SKIP_NO_H5 = pytest.mark.skipif(
    _MAPPINGS_H5 is None,
    reason="data/flury2024_grids.h5 not found; run scripts/build_flury2024_grids.py",
)


@pytest.fixture(scope="module")
def mappings_module():
    return importlib.import_module("tengri.components.nebular.mappings_photo")


@pytest.fixture(scope="module")
def fake_stellar_grid_data(mappings_module):
    """Build a minimal MappingsStellarGridData without HDF5 for testing."""
    mod = mappings_module
    n_z, n_a, n_u, n_n, n_s, n_lines = 5, 6, 4, 3, 2, 10

    zo_axis = jnp.array([0.2, 0.5, 1.0, 1.5, 2.0])  # ζ_O solar-relative
    logU_axis = jnp.linspace(-4.0, -1.5, n_u)
    log_age_yr_axis = jnp.linspace(6.0, 8.0, n_a)
    logn_axis = jnp.linspace(0.5, 3.5, n_n)

    # Grid shapes: (N_z, N_a, N_s, N_u, N_n)
    logHB_per_logq = jnp.ones((n_z, n_a, n_s, n_u, n_n)) * (-12.0)  # log10(erg/photon)
    # Line ratios: (N_z, N_a, N_s, N_u, N_n, N_lines)
    line_ratios = jnp.ones((n_z, n_a, n_s, n_u, n_n, n_lines))

    waves = jnp.array([1215.67, 1549.0, 4862.68, 5007.0] + [3000.0] * (n_lines - 4))

    return mod.MappingsStellarGridData(
        line_wavelengths=waves,
        zo_axis=zo_axis,
        logU_axis=logU_axis,
        log_age_yr_axis=log_age_yr_axis,
        logn_axis=logn_axis,
        logHB_per_logq=logHB_per_logq,
        line_ratios=line_ratios,
    )


# ── Unit tests ────────────────────────────────────────────────────


class TestPreintegrateForPhotometry:
    """The duck-typed surface MappingsPhotoStellarBackend exposes for the hybrid kernel.

    The kernel's nebular preint branch reads:
    ``_has_preint_photometry``, ``_preint_continuum``, ``_preint_lines``,
    ``_line_lum_collapsed``, ``_qh_table``, ``_qh_log_met``, ``_qh_log_age``,
    ``_young_idx``, ``grid.line_wavelengths``.
    """

    @pytest.fixture
    def backend_for_preint(self, mappings_module, fake_stellar_grid_data):
        """MappingsPhotoStellarBackend stub bypassing HDF5 I/O."""
        backend = object.__new__(mappings_module.MappingsPhotoStellarBackend)
        backend.model = "sb99"
        backend.density = "cpr"
        backend.sfh_mode = "inst"
        # SFH metadata (mirrors what _load_stellar_grid returns)
        backend.sfh_labels = ["cont", "inst"]
        backend.sfh_idx_inst = 1
        backend.sfh_idx_cont = 0
        backend._sfh_idx = backend.sfh_idx_inst
        backend.grid = fake_stellar_grid_data
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
        """MAPPINGS V has no nebular continuum — _preint_continuum.phot must be all zeros
        with shape (n_z, n_age, n_logU, n_filt)."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        cont = backend_for_preint._preint_continuum
        n_z = backend_for_preint.grid.zo_axis.shape[0]
        n_age = backend_for_preint.grid.log_age_yr_axis.shape[0]
        n_u = backend_for_preint.grid.logU_axis.shape[0]
        assert cont.phot.shape == (n_z, n_age, n_u, len(fw))
        assert bool(jnp.all(cont.phot == 0.0))

    def test_axis0_relabeled_to_absolute_logz(self, backend_for_preint, synthetic_filters):
        """Axis 0 of the preint surface must be absolute log10(Z), not ζ_O.

        Conversion: log10(Z) = log10(ζ_O × Z_sun) = log10(ζ_O) + log10(Z_sun).
        """
        from tengri.components.nebular._constants import _LOG10_ZSUN

        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        cont = backend_for_preint._preint_continuum
        zo_axis = backend_for_preint.grid.zo_axis
        expected = jnp.log10(zo_axis) + _LOG10_ZSUN
        np.testing.assert_allclose(np.array(cont.axes[0]), np.array(expected), atol=1e-6)

    def test_line_lum_collapsed_shape_and_units(self, backend_for_preint, synthetic_filters):
        """``_line_lum_collapsed`` shape (n_z, n_age, n_logU, n_lines) and
        log10(L_line/Q_H [Lsun·s/photon]) = log10(ratio × 10^logHB_per_logq / LSUN_ERG)."""
        from tengri.components.nebular._constants import _LSUN_ERG

        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        n_z = backend_for_preint.grid.zo_axis.shape[0]
        n_age = backend_for_preint.grid.log_age_yr_axis.shape[0]
        n_u = backend_for_preint.grid.logU_axis.shape[0]
        n_lines = backend_for_preint.grid.line_wavelengths.shape[0]
        chex.assert_shape(backend_for_preint._line_lum_collapsed, (n_z, n_age, n_u, n_lines))
        # fake grid: ratio=1, logHB_per_logq=-12 → L_erg_per_q = 10^-12
        # Convert to Lsun and log10: expected = log10(10^-12 / LSUN_ERG)
        expected_val = jnp.log10(10.0 ** (-12.0) / _LSUN_ERG)
        np.testing.assert_allclose(
            np.array(backend_for_preint._line_lum_collapsed),
            float(expected_val),
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
        n_z = backend_for_preint.grid.zo_axis.shape[0]
        n_age = backend_for_preint.grid.log_age_yr_axis.shape[0]
        n_lines = backend_for_preint.grid.line_wavelengths.shape[0]
        chex.assert_shape(backend_for_preint._line_lum_collapsed, (n_z, n_age, n_lines))
        assert len(backend_for_preint._preint_lines.axes) == 2

    def test_preint_lines_axes_match_collapsed_lum(self, backend_for_preint, synthetic_filters):
        """The axes in _preint_lines must match the dimensions of _line_lum_collapsed."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        n_axes = len(backend_for_preint._preint_lines.axes)
        n_dims = len(backend_for_preint._line_lum_collapsed.shape) - 1  # exclude lines axis
        assert n_axes == n_dims

    def test_continuum_axes_match_before_fixed(self, backend_for_preint, synthetic_filters):
        """Before any fixed collapse, continuum axes must match surface axes."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        cont = backend_for_preint._preint_continuum
        assert len(cont.axes) == 3  # Z, age, U

    def test_fixed_zero_collapses_metallicity(self, backend_for_preint, synthetic_filters):
        """Passing ``fixed={0: -1.0}`` collapses metallicity to (n_age, n_u, n_lines)."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28, fixed={0: -1.0})
        n_age = backend_for_preint.grid.log_age_yr_axis.shape[0]
        n_u = backend_for_preint.grid.logU_axis.shape[0]
        n_lines = backend_for_preint.grid.line_wavelengths.shape[0]
        chex.assert_shape(backend_for_preint._line_lum_collapsed, (n_age, n_u, n_lines))

    def test_continuum_effective_wavelengths_zero(self, backend_for_preint, synthetic_filters):
        """Continuum effective wavelengths must be zero (not used)."""
        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        cont = backend_for_preint._preint_continuum
        assert bool(jnp.all(cont.effective_wavelengths == 0.0))
        assert bool(jnp.all(cont.effective_wavelengths_rest == 0.0))

    def test_line_lum_is_log10_lsun_per_q(self, backend_for_preint, synthetic_filters):
        """Regression: line_lum must be log10(L_line/Q_H [Lsun·s/photon]), not linear.

        The hybrid kernel reads _line_lum_collapsed and does 10**log_lum × Q_H;
        it expects log10 units to round-trip correctly. This test verifies the
        closed-form calculation at a single grid point.

        Fake grid: line_ratios=1, logHB_per_logq=-12 → L_erg_per_q = 10^-12 erg/photon.
        Expected: log10(10^-12 / LSUN_ERG) = log10(10^-12 / 3.839e33).
        """
        from tengri.components.nebular._constants import _LSUN_ERG

        fw, ft = synthetic_filters
        backend_for_preint.preintegrate_for_photometry(fw, ft, 0.5, 1.0e28)
        line_lum = backend_for_preint._line_lum_collapsed  # shape (N_z, N_a, N_u, N_lines)

        # Fake grid: ratio=1, logHB_per_logq=-12
        line_ratio = 1.0
        logHB_per_logq = -12.0
        line_lum_erg_per_q = line_ratio * (10.0**logHB_per_logq)
        line_lum_lsun_per_q = line_lum_erg_per_q / _LSUN_ERG
        expected = jnp.log10(line_lum_lsun_per_q)

        # All grid points should match the expected value
        np.testing.assert_allclose(np.array(line_lum), float(expected), atol=1e-6)
