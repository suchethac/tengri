# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``reproduction._validation.band_average`` integrates on a chosen set of nodes.

The default ``integrate="filter"`` samples the spectrum at the filter's own nodes, so
a spectral feature narrower than the node spacing (an emission line against a 25
Angstrom filter table) is hit or missed depending on where the nodes fall.
``integrate="sed"`` integrates on the spectrum's own nodes with the filter
interpolated onto them and so recovers the band value of the underlying continuous
spectrum. These tests pin both behaviors against a dense-quadrature reference.
"""

from __future__ import annotations

import numpy as np
import pytest
from reproduction import _validation as V

pytestmark = pytest.mark.contract

_NODE_SPACING = 25.0
_LINE_SIGMA = 2.0
_LINE_AMPLITUDE = 50.0
_FILTER_LO, _FILTER_HI, _FILTER_CENTER = 4000.0, 5000.0, 4500.0

# Line centers: on a filter node, midway between two nodes, a quarter of a spacing
# from a node.
_LINE_POSITIONS = {
    "on_node": 4500.0,
    "midway": 4512.5,
    "quarter": 4506.25,
}


def _filter() -> tuple[np.ndarray, np.ndarray]:
    """A smooth bump tabulated on 25 Angstrom nodes (piecewise-linear by definition)."""
    fw = np.arange(_FILTER_LO, _FILTER_HI + 0.5 * _NODE_SPACING, _NODE_SPACING)
    ft = np.sin(np.pi * (fw - _FILTER_LO) / (_FILTER_HI - _FILTER_LO)) ** 2
    return fw, ft


def _spectrum(center: float, grid_step: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Flat continuum plus one resolved Gaussian line, on a fine grid wider than the filter."""
    wave = np.arange(_FILTER_LO - 100.0, _FILTER_HI + 100.0, grid_step)
    line = _LINE_AMPLITUDE * np.exp(-0.5 * ((wave - center) / _LINE_SIGMA) ** 2)
    return wave, 1.0 + line


def _reference(center: float) -> float:
    """Photon-weighted band average by 0.005 Angstrom quadrature of the same filter table."""
    fw, ft = _filter()
    x = np.arange(_FILTER_LO, _FILTER_HI, 0.005)
    t = np.interp(x, fw, ft)
    L = 1.0 + _LINE_AMPLITUDE * np.exp(-0.5 * ((x - center) / _LINE_SIGMA) ** 2)
    w = t / x
    return float(np.trapezoid(L * w, x) / np.trapezoid(w, x))


@pytest.mark.parametrize("position", list(_LINE_POSITIONS))
def test_sed_integration_matches_dense_quadrature(position):
    """``integrate='sed'`` equals the dense-quadrature band average to 1e-3."""
    center = _LINE_POSITIONS[position]
    wave, L = _spectrum(center)
    fw, ft = _filter()
    value = V.band_average(wave, L, fw, ft, integrate="sed")
    assert value == pytest.approx(_reference(center), rel=1e-3)


def test_filter_mode_aliases_a_line_between_nodes():
    """The default samples the line only where a filter node lands on it.

    A line midway between two nodes (12.5 A from each, six sigma) is missed
    entirely, so the filter-node band average falls more than 10 % below the
    spectrum's actual band average, which carries the line's flux.
    """
    center = _LINE_POSITIONS["midway"]
    wave, L = _spectrum(center)
    fw, ft = _filter()
    sed = V.band_average(wave, L, fw, ft, integrate="sed")
    node = V.band_average(wave, L, fw, ft)
    assert (sed - node) / sed > 0.10


@pytest.mark.parametrize("position", list(_LINE_POSITIONS))
def test_filter_mode_is_the_default(position):
    """Omitting ``integrate`` is identical to ``integrate='filter'``."""
    wave, L = _spectrum(_LINE_POSITIONS[position])
    fw, ft = _filter()
    assert V.band_average(wave, L, fw, ft) == V.band_average(wave, L, fw, ft, integrate="filter")


@pytest.mark.parametrize("weight", ["photon", "energy"])
def test_smooth_spectrum_modes_agree(weight):
    """A smooth power-law spectrum gives the same band average in both modes to 1e-4."""
    wave = np.arange(_FILTER_LO - 100.0, _FILTER_HI + 100.0, 0.5)
    L = (wave / _FILTER_CENTER) ** -1.3
    fw, ft = _filter()
    node = V.band_average(wave, L, fw, ft, weight=weight, integrate="filter")
    sed = V.band_average(wave, L, fw, ft, weight=weight, integrate="sed")
    assert sed == pytest.approx(node, rel=1e-4)


def test_coarse_spectrum_still_resolves_the_filter():
    """A spectrum coarser than the filter table keeps the filter's own nodes in the grid.

    A constant spectrum on a 200 Angstrom grid must return the constant: the
    average is weight-normalized, so only a grid that loses the filter shape would
    move it, and a grid of the spectrum nodes alone would not resolve the bump.
    """
    wave = np.arange(3000.0, 6000.0, 200.0)
    L = np.full_like(wave, 3.0)
    fw, ft = _filter()
    assert V.band_average(wave, L, fw, ft, integrate="sed") == pytest.approx(3.0, rel=1e-12)


@pytest.mark.parametrize("integrate", ["filter", "sed"])
def test_uncovered_band_is_nan(integrate):
    """A spectrum that stops short of the filter, or is zero inside it, returns NaN."""
    fw, ft = _filter()
    short = np.arange(_FILTER_LO, _FILTER_HI - 200.0, 1.0)
    assert np.isnan(V.band_average(short, np.ones_like(short), fw, ft, integrate=integrate))
    wave = np.arange(_FILTER_LO - 100.0, _FILTER_HI + 100.0, 1.0)
    L = np.where(wave < 4600.0, 1.0, 0.0)
    assert np.isnan(V.band_average(wave, L, fw, ft, integrate=integrate))


def test_unknown_integrate_raises():
    wave, L = _spectrum(4500.0)
    fw, ft = _filter()
    with pytest.raises(ValueError, match="integrate"):
        V.band_average(wave, L, fw, ft, integrate="trapezoid")


def test_integrate_threads_through_the_row_builders():
    """``filter_rows`` and ``filter_rows_native`` forward the keyword to every band."""
    filters = V.UV_TO_NIR[2:5]
    wave = np.arange(1000.0, 30000.0, 0.5)
    pivot = np.array([V.pivot_wavelength(*V.load_filter(stem)) for stem, _ in filters])
    L_ref = np.ones_like(wave)
    L_line = 1.0 + 80.0 * np.exp(-0.5 * ((wave - (pivot[0] + 11.0)) / 2.0) ** 2)
    for builder in (
        lambda **kw: V.filter_rows(wave, L_line, L_ref, filters=filters, **kw),
        lambda **kw: V.filter_rows_native(wave, L_line, wave, L_ref, filters=filters, **kw),
    ):
        default = builder()
        sed = builder(integrate="sed")
        explicit = builder(integrate="filter")
        assert default == explicit
        assert sed[0][4] != default[0][4]
