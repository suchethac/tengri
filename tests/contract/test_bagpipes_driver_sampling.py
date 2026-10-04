# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the BAGPIPES driver builds models on a grid fine enough for band integrals.

``model_galaxy(components)`` without ``spec_wavs`` samples ``spectrum_full`` at
``R_other = 20`` (747 points, 2.5 % steps). The stellar grid is point-sampled onto it
and each emission line falls in one pixel, so any band average of it depends on the
grid. ``_build_model`` passes ``spec_wavs`` and raises ``R_other``, then asserts the
returned grid resolves the comparison range. These tests pin the assertion on
synthetic grids and the build wiring on a stand-in ``model_galaxy``.
"""

from __future__ import annotations

import importlib
from typing import ClassVar

import numpy as np
import pytest

pytestmark = pytest.mark.contract


def _geometric_grid(r: float, lo: float = 1.0, hi: float = 1e8) -> np.ndarray:
    """A BAGPIPES-style grid: constant ``delta-lambda / lambda = 0.5 / r``."""
    n = int(np.log(hi / lo) / np.log1p(0.5 / r)) + 1
    return lo * np.exp(np.arange(n) * np.log1p(0.5 / r))


@pytest.fixture(scope="module")
def driver():
    return importlib.import_module("reproduction.bagpipes._drivers.bagpipes_driver")


def test_default_bagpipes_grid_is_refused(driver):
    """The R_other = 20 grid (747 points) fails the assertion and the message names the fix."""
    grid = _geometric_grid(20.0)
    assert 740 < grid.size < 760
    lo, hi = driver.COMPARISON_RANGE_AA
    with pytest.raises(RuntimeError, match="spec_wavs"):
        driver.assert_sampling(grid, lo, hi)


@pytest.mark.parametrize("r", [100.0, 300.0])
def test_grid_below_the_requirement_is_refused(driver, r):
    """The pixel pitch is 2 R, so R = 100 and 300 give 200 and 600, below the required 1000."""
    lo, hi = driver.COMPARISON_RANGE_AA
    with pytest.raises(RuntimeError, match="too coarse"):
        driver.assert_sampling(_geometric_grid(r), lo, hi)


def test_grid_above_the_requirement_is_accepted(driver):
    """R = 600 gives a pitch of 1200 and passes."""
    lo, hi = driver.COMPARISON_RANGE_AA
    pitch = driver.assert_sampling(_geometric_grid(600.0), lo, hi)
    assert pitch == pytest.approx(1200.0, rel=1e-3)


def test_converged_grid_passes_with_its_pitch(driver):
    """R_spec = 1000 inside the range, R_other outside: median pitch is 2000."""
    lo, hi = driver.COMPARISON_RANGE_AA
    fine = _geometric_grid(driver.R_SPEC_BUILD, lo / 11.0, hi)
    coarse = _geometric_grid(driver.R_OTHER_BUILD, hi, 1e8)
    grid = np.concatenate([_geometric_grid(driver.R_OTHER_BUILD, 1.0, lo / 11.0), fine, coarse])
    pitch = driver.assert_sampling(grid, lo, hi)
    assert pitch == pytest.approx(2.0 * driver.R_SPEC_BUILD, rel=1e-3)


def test_too_few_points_in_range_raises(driver):
    with pytest.raises(ValueError, match="fewer than two"):
        driver.median_pixel_pitch(np.array([1.0, 10.0, 100.0]), 1000.0, 2000.0)


class _FakeModelGalaxy:
    """Records the config values and ``spec_wavs`` seen at construction."""

    seen: ClassVar[dict] = {}

    def __init__(self, components, spec_wavs=None):
        import bagpipes.config as cfg

        type(self).seen = {"R_spec": cfg.R_spec, "R_other": cfg.R_other, "spec_wavs": spec_wavs}
        self.wavelengths = _fake_grid(cfg.R_spec, cfg.R_other, spec_wavs)


def _fake_grid(r_spec: float, r_other: float, spec_wavs) -> np.ndarray:
    lo, hi = float(spec_wavs[0]), float(spec_wavs[-1])
    return np.concatenate(
        [
            _geometric_grid(r_other, 1.0, lo / 11.0),
            _geometric_grid(r_spec, lo / 11.0, hi),
            _geometric_grid(r_other, hi, 1e8),
        ]
    )


def test_build_model_passes_comparison_range_and_restores_config(driver, monkeypatch):
    """The build raises R_other for its duration, passes the comparison range, then restores."""
    cfg = pytest.importorskip("bagpipes.config")
    before = (cfg.R_spec, cfg.R_other)
    monkeypatch.setattr(driver, "_model_galaxy_class", lambda: _FakeModelGalaxy)
    driver._build_model({"delayed": {}})
    seen = _FakeModelGalaxy.seen
    assert seen["R_spec"] >= driver.R_SPEC_BUILD
    assert seen["R_other"] >= driver.R_OTHER_BUILD
    assert tuple(seen["spec_wavs"]) == driver.COMPARISON_RANGE_AA
    assert (cfg.R_spec, cfg.R_other) == before


def test_build_model_refuses_a_coarse_result(driver, monkeypatch):
    """A model_galaxy that returns the R_other = 20 grid makes ``_build_model`` raise."""
    pytest.importorskip("bagpipes.config")

    class _Coarse(_FakeModelGalaxy):
        def __init__(self, components, spec_wavs=None):
            self.wavelengths = _geometric_grid(20.0)

    monkeypatch.setattr(driver, "_model_galaxy_class", lambda: _Coarse)
    with pytest.raises(RuntimeError, match="too coarse"):
        driver._build_model({"delayed": {}})
