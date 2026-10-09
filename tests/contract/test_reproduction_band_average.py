# SPDX-License-Identifier: BSD-3-Clause
"""Contract: ``reproduction._validation.band_average`` integrates on a chosen set of nodes.

The default ``integrate="sed"`` integrates on the spectrum's own nodes with the filter
interpolated onto them, so it recovers the band value of the underlying continuous
spectrum. ``integrate="filter"`` samples the spectrum at the filter's own nodes, so a
spectral feature narrower than the node spacing (an emission line against a 25
Angstrom filter table) is hit or missed depending on where the nodes fall; it is kept
only to reproduce old numbers. These tests pin both behaviors against a
dense-quadrature reference.
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
    node = V.band_average(wave, L, fw, ft, integrate="filter")
    assert (sed - node) / sed > 0.10


@pytest.mark.parametrize("position", list(_LINE_POSITIONS))
def test_sed_mode_is_the_default(position):
    """Omitting ``integrate`` is identical to ``integrate='sed'``."""
    wave, L = _spectrum(_LINE_POSITIONS[position])
    fw, ft = _filter()
    assert V.band_average(wave, L, fw, ft) == V.band_average(wave, L, fw, ft, integrate="sed")


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
        node = builder(integrate="filter")
        assert default == sed
        assert node[0][4] != default[0][4]


def test_line_window_mask_removes_only_the_window_and_keeps_a_flat_median():
    wave = np.linspace(4000.0, 6000.0, 4001)
    lines = np.array([5000.0])
    mask = V.line_window_mask(wave, lines, 500.0)
    half = 500.0 / 299792.458 * wave
    assert np.array_equal(mask, np.abs(wave - 5000.0) <= np.maximum(half, 0.5))
    assert 0 < mask.sum() < wave.size

    ratio = np.full_like(wave, 1.02)
    ratio[mask] = 5.0  # a line spike inside the window
    shown = np.where(mask, np.nan, ratio)
    assert np.isnan(shown[mask]).all() and np.isfinite(shown[~mask]).all()
    assert np.median(shown[~mask]) == pytest.approx(1.02)
    # No line list: nothing to mask
    assert not V.line_window_mask(wave, np.array([])).any()


def test_sweep_fig_mask_keyword_leaves_returned_ratios_untouched():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wave = np.linspace(4000.0, 6000.0, 2001)
    ref = np.ones_like(wave)
    ref[1000] = 50.0
    tengri = np.full_like(wave, 1.02)
    cases = [("a", wave, ref, wave, tengri)]
    kw = dict(ref_label="ref", title="t", logy=False, annotate_median=True)
    _, _, r0 = V.sweep_fig(cases, **kw)
    _, (_, ax_r), r1 = V.sweep_fig(cases, mask_lines_aa=[5000.0], **kw)
    plt.close("all")
    assert np.array_equal(r0["a"], r1["a"], equal_nan=True)
    shown = [ln.get_ydata() for ln in ax_r.lines if len(ln.get_ydata()) == wave.size]
    assert shown and np.isnan(shown[-1][1000])


# ---------------------------------------------------------------------------
# Narrow line between two coarse filter nodes (#2680)
# ---------------------------------------------------------------------------

_FLAT_FILTER_NODES = np.arange(4000.0, 5000.0 + 0.5 * _NODE_SPACING, _NODE_SPACING)
_LINE_HEIGHT = 100.0  # erg/s/Hz, peak of the triangular line
_LINE_HALF_WIDTH = 0.5  # Angstrom; the line's integrated flux is height * half width = 50


def _flat_top_filter() -> tuple[np.ndarray, np.ndarray]:
    """Transmission 1 on every node from 4025 to 4975 A, ramping to 0 at the two ends.

    The 4500-4525 A stretch is flat, so a line placed there sees T = 1 and the band
    integral has a closed form.
    """
    fw = _FLAT_FILTER_NODES
    ft = np.ones_like(fw)
    ft[0] = ft[-1] = 0.0
    return fw, ft


def _narrow_line_spectrum(center: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit continuum plus a 1 A-wide triangular line, with nodes on its apex and feet."""
    base = np.arange(3900.0, 5100.0, 0.1)
    wave = np.union1d(base, [center - _LINE_HALF_WIDTH, center, center + _LINE_HALF_WIDTH])
    tri = _LINE_HEIGHT * np.clip(1.0 - np.abs(wave - center) / _LINE_HALF_WIDTH, 0.0, None)
    return wave, 1.0 + tri


def _narrow_line_analytic(center: float) -> float:
    """Closed-form photon band average: 1 + (flux / center) / integral(T / lambda).

    The line sits where T = 1, so its weighted flux is flux / center up to a relative
    (1 A / center)^2. The denominator is the filter's integral, taken on a 0.01 A grid.
    """
    fw, ft = _flat_top_filter()
    x = np.arange(fw[0], fw[-1], 0.01)
    denom = np.trapezoid(np.interp(x, fw, ft) / x, x)
    flux = _LINE_HEIGHT * _LINE_HALF_WIDTH  # triangle area = height * full base / 2
    return 1.0 + (flux / center) / denom


@pytest.mark.parametrize("center", [4512.5, 4517.5])
def test_band_average_of_narrow_line_matches_analytic_value(center):
    """A 1 A line between two 25 A nodes contributes its flux, not what the nodes see.

    The old default sampled L_nu only at the filter nodes (4500, 4525), missing the line
    entirely. The band value must equal the analytic continuum-plus-line result.
    """
    wave, L = _narrow_line_spectrum(center)
    fw, ft = _flat_top_filter()
    value = V.band_average(wave, L, fw, ft)
    assert value == pytest.approx(_narrow_line_analytic(center), rel=1e-3)


def test_band_average_is_insensitive_to_a_5_angstrom_line_shift():
    """Shifting the line by 5 A inside a flat filter region moves the band value by under 1e-3."""
    fw, ft = _flat_top_filter()
    a = V.band_average(*_narrow_line_spectrum(4512.5), fw, ft)
    b = V.band_average(*_narrow_line_spectrum(4517.5), fw, ft)
    assert a == pytest.approx(_narrow_line_analytic(4512.5), rel=1e-3)
    assert abs(a - b) / a < 1e-3
