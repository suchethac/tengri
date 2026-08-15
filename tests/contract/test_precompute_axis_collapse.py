# SPDX-License-Identifier: BSD-3-Clause
"""Auto-collapse correctness tests for precompute adapters.

Confirms that with each grid axis individually Fixed, the collapsed grid's
lookup matches the un-collapsed lookup at that fixed value.

This test guards against regressions in slice_fixed_axes triweight collapse
machinery (e.g., Silva04, Cat3D, disc, cb19 previously used non-existent
parameters.is_fixed() API; radio, xray, dust_analytic fixed in this session).

Strategy: Test adapters directly via get_fixed_values() logic without needing
full Parameters objects (which have complex validation rules). Instead, we:
1. Build FULL adapter with parameters=None.
2. Call precompute() with a synthetic Parameters that has Fixed axes.
3. Verify collapsed lookup matches un-collapsed lookup at fixed value.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)

# Standard synthetic filter set (used across test adapters)
_CENTERS = np.array([3e5, 1e7, 1e8, 1e10])  # FIR–radio Angstrom
_WIDTHS = np.array([1e5, 3e6, 3e7, 3e9])


@pytest.fixture(scope="module")
def filter_set_radio():
    """Synthetic 4-filter set for radio precompute."""
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(_CENTERS, _WIDTHS):
        wv = np.linspace(c - 3 * w, c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


@pytest.fixture(scope="module")
def filter_set_xray():
    """Synthetic 4-filter set for X-ray precompute (0.1–100 keV)."""
    centers = np.array([1.0, 5.0, 50.0, 500.0])  # 0.1–100 keV in Angstrom
    widths = np.array([0.3, 1.5, 15.0, 150.0])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(max(c - 3 * w, 1e-2), c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


def _make_mock_params(fixed_dict: dict[str, float]) -> MagicMock:
    """Create a mock Parameters object with get_fixed_values() returning fixed_dict."""
    mock = MagicMock()
    mock.get_fixed_values.return_value = fixed_dict
    mock.free_params = []
    return mock


# ── Radio precompute tests (1 axis each) ──────────────────────────────────


class TestRadioPrecomputeAxisCollapse:
    """Test axis collapse for radio models (radio_alpha_sf, radio_alpha_ff, radio_alpha_agn)."""

    @pytest.mark.parametrize(
        "model,param_name",
        [
            ("radio_synchrotron", "radio_alpha_sf"),
            ("radio_freefree", "radio_alpha_ff"),
            ("radio_agn_jet", "radio_alpha_agn"),
        ],
    )
    def test_radio_collapse_axis(self, model, param_name, filter_set_radio):
        from tengri.components.radio import radio_precompute as adapter

        waves, trans = filter_set_radio
        redshift = 0.5

        # Build FULL adapter (no Fixed params).
        full = adapter.precompute(waves, trans, redshift, parameters=None, model=model)
        full_lookup = adapter.build_lookup(full, model=model)
        assert len(full["axes"]) == 1, f"{model} should have 1 axis"

        # Pick the midpoint value from the axis.
        ax = full["axes"][0]
        midpoint_idx = len(ax) // 2
        pinned_value = float(ax[midpoint_idx])

        # Build COLLAPSED adapter with mock Parameters.
        spec = _make_mock_params({param_name: pinned_value})
        coll = adapter.precompute(waves, trans, redshift, parameters=spec, model=model)
        coll_lookup = adapter.build_lookup(coll, model=model)

        # Collapsed lookup should have 0 grid axes (only scale).
        n_axes = len(coll["axes"])
        assert n_axes == 0, f"{model}: expected 0 axes after collapse, got {n_axes}"

        scale = jnp.float64(1.0)
        full_result = jax.jit(full_lookup)(scale, pinned_value)
        coll_result = jax.jit(coll_lookup)(scale)

        np.testing.assert_allclose(
            coll_result,
            full_result,
            rtol=1e-10,
            atol=0.0,
            err_msg=f"{model}: collapsed lookup mismatch",
        )


# ── X-ray precompute tests (2 axes each) ──────────────────────────────────


class TestXrayPrecomputeAxisCollapse:
    """Test axis collapse for X-ray models (2 free axes each)."""

    @pytest.mark.parametrize("model", ["xray_xrb", "xray_corona", "xray_corona_lopez24"])
    @pytest.mark.parametrize("fixed_axis_idx", [0, 1])
    def test_xray_collapse_axis(self, model, fixed_axis_idx, filter_set_xray):
        from tengri.components.xray import xray_precompute as adapter

        # Read the axis names off the adapter rather than restating them here.
        # A copy in the test cannot disagree with the declaration it copies, so
        # it cannot catch a name that drifted away from the live parameter — and
        # the mock below answers to whatever name it is handed, so a wrong one
        # still collapses and still passes. Both X-ray coronae declared
        # ``xray_gamma`` against a parameter named ``xray_gamma_agn`` and this
        # suite stayed green throughout (#1738).
        params = adapter.AXIS_PARAMS[model]

        waves, trans = filter_set_xray
        redshift = 0.5

        # Build FULL adapter.
        full = adapter.precompute(waves, trans, redshift, parameters=None, model=model)
        full_lookup = adapter.build_lookup(full, model=model)
        assert len(full["axes"]) == 2, f"{model} should have 2 axes"

        # Pick midpoint from the axis to collapse.
        ax_to_fix = full["axes"][fixed_axis_idx]
        midpoint_idx = len(ax_to_fix) // 2
        pinned_value = float(ax_to_fix[midpoint_idx])

        # Build COLLAPSED adapter with mock Parameters.
        param_to_fix = params[fixed_axis_idx]
        spec = _make_mock_params({param_to_fix: pinned_value})
        coll = adapter.precompute(waves, trans, redshift, parameters=spec, model=model)
        coll_lookup = adapter.build_lookup(coll, model=model)

        # After collapse, should have 1 axis.
        n_axes_coll = len(coll["axes"])
        assert n_axes_coll == 1, f"{model}: expected 1 axis after collapse, got {n_axes_coll}"

        # Sample other axis at midpoint.
        other_axis_idx = 1 - fixed_axis_idx
        other_ax = full["axes"][other_axis_idx]
        other_midpoint_idx = len(other_ax) // 2
        other_value = float(other_ax[other_midpoint_idx])

        scale = jnp.float64(1.0)

        if fixed_axis_idx == 0:
            full_result = jax.jit(full_lookup)(scale, pinned_value, other_value)
            coll_result = jax.jit(coll_lookup)(scale, other_value)
        else:
            full_result = jax.jit(full_lookup)(scale, other_value, pinned_value)
            coll_result = jax.jit(coll_lookup)(scale, other_value)

        np.testing.assert_allclose(
            coll_result,
            full_result,
            rtol=1e-10,
            atol=0.0,
            err_msg=f"{model}: axis {fixed_axis_idx} collapse mismatch",
        )


# ── AGN disc precompute tests ──────────────────────────────────────────────
# NOTE: Disc, qsogen, and cat3d adapters have build_lookup() signatures that
# require runtime parameters (agn_torus_frac, etc.) not present in the precompute
# dict. These adapters are being refactored for Wilkinson phase; tests deferred.
# See project_phase_ii3_progress.md for status.


class TestDiscPrecomputeAxisCollapse:
    """Test axis collapse for disc models (1–2 axes).

    NOTE: disc build_lookup signature requires runtime parameters (agn_torus_frac, etc.)
    not present in precompute grid axes. Tests deferred to Wilkinson phase refactor.
    """


# ── QSOgen precompute tests (2 axes) ───────────────────────────────────────
# NOTE: QSOgen's build_lookup() requires free_param_names kwarg reflecting the
# collapsed axis set. The standard template lookup doesn't handle this;
# deferred to Wilkinson phase refactor.


class TestQsogenPrecomputeAxisCollapse:
    """Test axis collapse for QSOgen (agn_plslp1, agn_ebv).

    NOTE: QSOgen's build_lookup() requires free_param_names kwarg reflecting the
    collapsed axis set. Standard template lookup doesn't handle this; deferred to
    Wilkinson phase refactor.
    """


# ── Silva04 precompute tests (1 axis) ──────────────────────────────────────


class TestSilva04PrecomputeAxisCollapse:
    """Test axis collapse for Silva04 (silva04_log_NH)."""

    _DATA = Path(__file__).parent.parent.parent.parent / "data"

    @pytest.fixture(autouse=True)
    def _skip_if_no_silva04(self):
        """Skip if Silva04 data file missing."""
        grid_file = self._DATA / "silva04_wind_torus_grid.h5"
        if not grid_file.exists():
            pytest.skip(f"Silva04 grid file not found: {grid_file}")

    def test_silva04_collapse_axis(self, filter_set_radio):
        from tengri.components.agn import silva04_precompute as adapter

        grid_file = self._DATA / "silva04_wind_torus_grid.h5"
        if not grid_file.exists():
            pytest.skip(f"Silva04 grid file not found: {grid_file}")

        waves, trans = filter_set_radio
        redshift = 0.5

        # Build FULL adapter.
        full = adapter.precompute(
            waves, trans, redshift, parameters=None, grid_path=str(grid_file)
        )
        full_lookup = adapter.build_lookup(full)
        assert len(full["axes"]) == 1, "silva04 should have 1 axis"

        # Collapse the axis.
        ax = full["axes"][0]
        midpoint_idx = len(ax) // 2
        pinned_value = float(ax[midpoint_idx])

        spec = _make_mock_params({"silva04_log_NH": pinned_value})
        coll = adapter.precompute(
            waves, trans, redshift, parameters=spec, grid_path=str(grid_file)
        )
        coll_lookup = adapter.build_lookup(coll)

        assert len(coll["axes"]) == 0, "silva04: expected 0 axes after collapse"

        scale = jnp.float64(1.0)
        full_result = jax.jit(full_lookup)(scale, pinned_value)
        coll_result = jax.jit(coll_lookup)(scale)

        np.testing.assert_allclose(
            coll_result, full_result, rtol=1e-10, atol=0.0, err_msg="silva04: collapse mismatch"
        )


# ── Cat3D precompute tests (3 axes) ────────────────────────────────────────
# NOTE: Cat3D's build_lookup() requires agn_torus_frac kwarg. Runtime parameter
# signatures mismatch precompute grid axes; deferred to Wilkinson phase refactor.


class TestCat3DPrecomputeAxisCollapse:
    """Test axis collapse for Cat3D (3 axes).

    NOTE: Cat3D's build_lookup() requires agn_torus_frac kwarg. Runtime parameter
    signatures mismatch precompute grid axes; deferred to Wilkinson phase refactor.
    """

    _DATA = Path(__file__).parent.parent.parent.parent / "data"

    @pytest.fixture(autouse=True)
    def _skip_if_no_cat3d(self):
        """Skip if CAT3D data file missing."""
        grid_file = self._DATA / "cat3d_wind_torus_grid.h5"
        if not grid_file.exists():
            pytest.skip(f"CAT3D grid file not found: {grid_file}")


# ── Dust analytic precompute tests ─────────────────────────────────────────


class TestDustAnalyticPrecomputeAxisCollapse:
    """Test axis collapse for dust analytic models."""

    def test_dust_mbb_collapse(self, filter_set_radio):
        """Test modified_blackbody (2 axes) collapse."""
        from tengri.components.dust import dust_analytic_precompute as adapter

        waves, trans = filter_set_radio
        redshift = 0.5

        # Build FULL adapter.
        full = adapter.precompute(
            waves, trans, redshift, parameters=None, model="modified_blackbody"
        )
        full_lookup = adapter.build_lookup(full, model="modified_blackbody")
        assert len(full["axes"]) == 2, "modified_blackbody should have 2 axes"

        # Collapse axis 0.
        ax0 = full["axes"][0]
        idx0 = len(ax0) // 2
        val0 = float(ax0[idx0])

        spec = _make_mock_params({"dust_T": val0})
        coll = adapter.precompute(
            waves, trans, redshift, parameters=spec, model="modified_blackbody"
        )
        coll_lookup = adapter.build_lookup(coll, model="modified_blackbody")

        assert len(coll["axes"]) == 1, "modified_blackbody: expected 1 axis after collapse"

        # Sample axis 1.
        ax1 = full["axes"][1]
        idx1 = len(ax1) // 2
        val1 = float(ax1[idx1])

        scale = jnp.float64(1.0)
        full_result = jax.jit(full_lookup)(scale, val0, val1)
        coll_result = jax.jit(coll_lookup)(scale, val1)

        np.testing.assert_allclose(
            coll_result,
            full_result,
            rtol=1e-10,
            atol=0.0,
            err_msg="modified_blackbody: collapse mismatch",
        )

    def test_dust_casey_collapse_axis0(self, filter_set_radio):
        """Test casey2012 (3 axes) collapse axis 0."""
        from tengri.components.dust import dust_analytic_precompute as adapter

        waves, trans = filter_set_radio
        redshift = 0.5

        # Build FULL adapter.
        full = adapter.precompute(waves, trans, redshift, parameters=None, model="casey2012")
        full_lookup = adapter.build_lookup(full, model="casey2012")
        assert len(full["axes"]) == 3, "casey2012 should have 3 axes"

        # Collapse axis 0.
        ax0 = full["axes"][0]
        idx0 = len(ax0) // 2
        val0 = float(ax0[idx0])

        spec = _make_mock_params({"dust_T": val0})
        coll = adapter.precompute(waves, trans, redshift, parameters=spec, model="casey2012")
        coll_lookup = adapter.build_lookup(coll, model="casey2012")

        assert len(coll["axes"]) == 2, "casey2012: expected 2 axes after collapse"

        # Sample axes 1 and 2.
        ax1 = full["axes"][1]
        idx1 = len(ax1) // 2
        val1 = float(ax1[idx1])
        ax2 = full["axes"][2]
        idx2 = len(ax2) // 2
        val2 = float(ax2[idx2])

        scale = jnp.float64(1.0)
        full_result = jax.jit(full_lookup)(scale, val0, val1, val2)
        coll_result = jax.jit(coll_lookup)(scale, val1, val2)

        np.testing.assert_allclose(
            coll_result,
            full_result,
            rtol=1e-10,
            atol=0.0,
            err_msg="casey2012 axis 0: collapse mismatch",
        )


# ── CB19 precompute tests (7 axes, conditional) ────────────────────────────


class TestCB19PrecomputeAxisCollapse:
    """Test axis collapse for CB19 nebular grid (7 axes)."""

    _DATA = Path(__file__).parent.parent.parent.parent / "data"

    @pytest.fixture(autouse=True)
    def _skip_if_no_cb19(self):
        """Skip if CB19 data file missing."""
        default_path = self._DATA / "cb19_grid.h5"
        if not default_path.exists():
            pytest.skip(f"CB19 grid file not found: {default_path}")

    def test_cb19_collapse_axis0(self, filter_set_radio):
        from tengri.components.nebular import cb19_precompute as adapter

        default_path = self._DATA / "cb19_grid.h5"
        if not default_path.exists():
            pytest.skip(f"CB19 grid file not found: {default_path}")

        waves, trans = filter_set_radio
        redshift = 0.5

        # Build FULL adapter.
        full = adapter.precompute(waves, trans, redshift, parameters=None)
        full_lookup = adapter.build_lookup(full)

        n_axes = len(full["axes"])
        assert n_axes == 7, f"CB19 should have 7 axes, got {n_axes}"

        # Collapse axis 0 (log_OH_total).
        ax = full["axes"][0]
        midpoint_idx = len(ax) // 2
        pinned_value = float(ax[midpoint_idx])

        spec = _make_mock_params({"log_OH_total": pinned_value})
        coll = adapter.precompute(waves, trans, redshift, parameters=spec)
        coll_lookup = adapter.build_lookup(coll)

        assert len(coll["axes"]) == 6, (
            f"CB19: expected 6 axes after collapsing axis 0, got {len(coll['axes'])}"
        )

        # Sample the remaining 6 axes at midpoints.
        other_values = []
        for i in range(1, 7):
            other_ax = full["axes"][i]
            other_midpoint_idx = len(other_ax) // 2
            other_values.append(float(other_ax[other_midpoint_idx]))

        scale = jnp.float64(1.0)
        full_result = jax.jit(full_lookup)(scale, pinned_value, *other_values)
        coll_result = jax.jit(coll_lookup)(scale, *other_values)

        np.testing.assert_allclose(
            coll_result,
            full_result,
            rtol=1e-10,
            atol=0.0,
            err_msg="CB19: axis 0 collapse mismatch",
        )
