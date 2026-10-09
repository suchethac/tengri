# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2637: spectral-index windows are hard top-hats.

Worthey et al. (1994, ApJS 94, 687, Eq. 1) and Trager et al. (1998, ApJS 116, 1,
Eqs. 1-3) define a Lick index over fixed passbands: the sideband flux is the mean
of F_λ over exactly the sideband, the pseudo-continuum F_Cλ is the line through the
two sideband means placed at the sideband mid-wavelengths, and
EW = ∫ (1 - F_λ/F_Cλ) dλ over exactly the feature passband. Balogh et al. (1999,
ApJ 527, 54) define Dn4000 as the ratio of the mean F_ν in 4000-4100 Å to the mean
F_ν in 3850-3950 Å. tengri weighted every pixel by a 1 Å sigmoid at each edge, which
shifted a Lick EW by 0.01-0.1 Å (up to 5 %) and Dn4000 by -0.14 %.

The expected values below are computed in the test from an analytic spectrum, by
fine-grid numpy integration of the definitions above, never from tengri's output.
"""

from __future__ import annotations

import dataclasses
from math import erf, pi, sqrt

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectral_indices import (
    STANDARD_INDICES,
    SpectralIndexDef,
    measure_index_jax,
    measure_indices_from_point_terms,
    precompute_index_windows,
)

pytestmark = pytest.mark.regression_bug

C_AA = 2.99792458e18  # speed of light [Å/s]

#: Tolerances requested by the owner ruling: 1e-3 Å on an EW, 1e-4 on Dn4000.
EW_TOL_AA = 1e-3
DN4000_TOL = 1e-4

_LINE_CENTER = 5000.0
_SIGMA = 3.0  # Gaussian absorption width [Å]
_DEPTH = 0.4

#: Sidebands and feature with edges that fall inside pixels of every grid below.
_EDGE_OFF_GRID = SpectralIndexDef(
    name="offgrid",
    index_type="EW",
    continuum=((4930.13, 4969.87), (5030.41, 5069.59)),
    feature=(4984.93, 5015.37),
)


def _flam(wave):
    """F_λ of a power law (F_λ ∝ λ^-1.2) with a Gaussian absorption line."""
    gauss = np.exp(-0.5 * ((wave - _LINE_CENTER) / _SIGMA) ** 2)
    return (wave / _LINE_CENTER) ** -1.2 * (1.0 - _DEPTH * gauss)


def _fnu(wave):
    return _flam(wave) * wave**2 / C_AA


def _fine_mean(fn, lo, hi, n=400001):
    x = np.linspace(lo, hi, n)
    return np.trapezoid(fn(x), x) / (hi - lo)


def trager_ew(idx, fn_flam):
    """Trager et al. (1998) Eqs. 1-3 by fine-grid integration of an analytic F_λ."""
    (b0, b1), (r0, r1) = idx.continuum
    f0, f1 = idx.feature
    x_b, x_r = 0.5 * (b0 + b1), 0.5 * (r0 + r1)
    f_b, f_r = _fine_mean(fn_flam, b0, b1), _fine_mean(fn_flam, r0, r1)
    x = np.linspace(f0, f1, 400001)
    f_c = f_b + (f_r - f_b) * (x - x_b) / (x_r - x_b)
    return np.trapezoid(1.0 - fn_flam(x) / f_c, x)


def _uniform(step):
    return np.arange(4800.0, 5300.0 + 1e-9, step)


def _irregular():
    rng = np.random.default_rng(2637)
    coarse = np.arange(4800.0, 5300.0, 0.3)
    cluster = rng.uniform(4975.0, 5025.0, 80)
    return np.unique(np.concatenate([coarse, cluster, [5300.0]]))


# ── Lick EW against the Trager definition on an analytic spectrum ──


@pytest.mark.parametrize("grid", ["uniform_0.25", "uniform_0.9", "irregular"])
@pytest.mark.parametrize("units", ["AA", "mag"])
def test_ew_with_edges_inside_pixels_matches_trager_eqs_1_to_3(grid, units):
    """Edges at 4984.93 etc. fall between pixels; the partial pixels carry the overlap."""
    wave = {
        "uniform_0.25": _uniform(0.25),
        "uniform_0.9": _uniform(0.9),
        "irregular": _irregular(),
    }[grid]
    idx = dataclasses.replace(_EDGE_OFF_GRID, units=units)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(_fnu(wave)), idx))
    truth = trager_ew(_EDGE_OFF_GRID, _flam)
    if units == "mag":
        width = _EDGE_OFF_GRID.feature[1] - _EDGE_OFF_GRID.feature[0]
        truth = -2.5 * np.log10(1.0 - truth / width)
        assert got == pytest.approx(truth, abs=EW_TOL_AA / 40.0)
    else:
        assert got == pytest.approx(truth, abs=EW_TOL_AA)


def test_gaussian_on_a_flat_continuum_has_the_closed_form_ew():
    """Flat F_λ with a Gaussian dip: EW = A σ sqrt(2π) erf(h / (σ sqrt 2)), edges off grid."""
    wave = _uniform(0.25)
    gauss = np.exp(-0.5 * ((wave - _LINE_CENTER) / _SIGMA) ** 2)
    lnu = (1.0 - _DEPTH * gauss) * wave**2 / C_AA
    lo, hi = _EDGE_OFF_GRID.feature
    # The feature is not centered on the line: integrate the Gaussian between its edges.
    cdf = lambda x: 0.5 * (1.0 + erf((x - _LINE_CENTER) / (_SIGMA * sqrt(2.0))))  # noqa: E731
    truth = _DEPTH * _SIGMA * sqrt(2.0 * pi) * (cdf(hi) - cdf(lo))
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(lnu), _EDGE_OFF_GRID))
    assert got == pytest.approx(truth, abs=EW_TOL_AA)


# ── Dn4000 against Balogh et al. (1999) ────────────────────────────


@pytest.mark.parametrize("name", ["Dn4000", "D4000"])
def test_break_of_a_power_law_equals_the_ratio_of_exact_window_means(name):
    """F_ν ∝ λ² on the 0.9 Å grid: mean of λ² over [a, b] is (b³ - a³) / (3 (b - a))."""
    idx = STANDARD_INDICES[name]
    (b0, b1), (r0, r1) = idx.continuum
    wave = np.arange(3700.0, 4300.0, 0.9)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(wave**2), idx))
    mean_sq = lambda a, b: (b**3 - a**3) / (3.0 * (b - a))  # noqa: E731
    truth = mean_sq(r0, r1) / mean_sq(b0, b1)
    assert got == pytest.approx(truth, abs=DN4000_TOL)


def test_dn4000_of_the_analytic_spectrum_matches_balogh_1999():
    """Dn4000 = <F_ν>[4000, 4100] / <F_ν>[3850, 3950] of the power law with a 4000 Å step."""
    idx = STANDARD_INDICES["Dn4000"]
    (b0, b1), (r0, r1) = idx.continuum

    def fnu(w):
        return (w / 4000.0) ** 2 * np.where(w < 4000.0, 1.0, 0.6) / C_AA

    wave = np.arange(3700.0, 4300.0, 0.05)
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(fnu(wave)), idx))
    truth = _fine_mean(fnu, r0, r1, 2000001) / _fine_mean(fnu, b0, b1, 2000001)
    assert got == pytest.approx(truth, abs=DN4000_TOL)


@pytest.mark.parametrize("name", ["Dn4000", "D4000"])
def test_flux_outside_the_window_does_not_leak_into_the_break(name):
    """Spikes one pixel outside each window edge (nodes on the edges) carry zero weight."""
    idx = STANDARD_INDICES[name]
    step = 0.5
    (b0, b1), (r0, r1) = idx.continuum
    wave = np.arange(3600.0, 4400.0, step)
    clean = np.ones_like(wave)
    spiked = clean.copy()
    for edge, side in [(b0, -1), (b1, 1), (r0, -1), (r1, 1)]:
        i = int(np.argmin(np.abs(wave - (edge + side * step))))
        assert abs(wave[i] - (edge + side * step)) < 1e-6, "edge must be on a node"
        spiked[i] = 1e6
    clean[wave >= r0] = 2.0
    spiked[wave >= r0] += clean[wave >= r0] - 1.0
    base = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(clean), idx))
    leak = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(spiked), idx))
    # Nodes lying exactly on a window edge enter with weight 0.5 step on the inside segment only
    # (the outside segment has zero overlap), so the spikes one node away are invisible.
    assert leak == pytest.approx(base, rel=1e-12)
    soft = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(spiked), idx, edge_width=1.0))
    assert abs(soft / base - 1.0) > 0.4  # the sigmoid edge sees the spikes


def test_window_mean_of_a_straight_line_is_the_edge_midpoint_for_any_edges():
    """Exact partial-pixel weights: <λ> over [lo, hi] is (lo + hi) / 2 on any grid."""
    wave = _irregular()
    idx = SpectralIndexDef(
        name="line_break",
        index_type="break",
        continuum=((4871.37, 4903.11), (5004.93, 5111.77)),
    )
    got = float(measure_index_jax(jnp.asarray(wave), jnp.asarray(wave), idx))
    (b0, b1), (r0, r1) = idx.continuum
    assert got == pytest.approx(0.5 * (r0 + r1) / (0.5 * (b0 + b1)), rel=1e-12)


# ── Window-LUT path equals the exact path ──────────────────────────


@pytest.mark.parametrize("grid", ["uniform_0.9", "irregular"])
def test_window_lut_equals_the_exact_path_and_the_definition(grid):
    wave = {"uniform_0.9": _uniform(0.9), "irregular": _irregular()}[grid]
    flux = jnp.asarray(_fnu(wave))[None, None, :]
    pc = precompute_index_windows(jnp.asarray(wave), flux, [_EDGE_OFF_GRID])
    lut = float(measure_indices_from_point_terms(pc.points.integrands[0, 0], pc)[0])
    exact = float(measure_index_jax(jnp.asarray(wave), flux[0, 0], _EDGE_OFF_GRID))
    assert lut == pytest.approx(exact, abs=1e-9)
    assert lut == pytest.approx(trager_ew(_EDGE_OFF_GRID, _flam), abs=EW_TOL_AA)


def test_window_lut_keeps_exactly_the_pixels_a_hard_window_touches():
    """Points of a hard window: the pixels inside plus the one straddling each edge."""
    wave = np.arange(3700.0, 4300.0, 0.9)
    flux = jnp.asarray(wave**2)[None, None, :]
    idx = STANDARD_INDICES["Dn4000"]
    pc = precompute_index_windows(jnp.asarray(wave), flux, [idx])
    (b0, b1), _ = idx.continuum
    in_blue = np.asarray(pc.points.window) == 0
    pts = np.asarray(pc.points.waves)[in_blue]
    assert pts.min() > b0 - 0.9 and pts.min() <= b0 + 0.9
    assert pts.max() < b1 + 0.9 and pts.max() >= b1 - 0.9


# ── Differentiability ──────────────────────────────────────────────


def test_hard_window_ew_is_differentiable_in_the_flux():
    """grad_flux HdA is finite, nonzero inside the windows, and matches a finite difference.

    Window edges are catalog constants, not parameters; the EW is linear in each node's
    partial-pixel weight, so the flux gradient needs no smoothing.
    """
    idx = STANDARD_INDICES["HdA"]
    wave = jnp.asarray(np.arange(3900.0, 4400.0, 0.9))
    flux = jnp.asarray(_fnu(np.asarray(wave)) * (1.0 + 1e-3 * np.sin(np.asarray(wave) / 7.0)))
    grad = jax.grad(lambda f: measure_index_jax(wave, f, idx))(flux)
    assert bool(jnp.all(jnp.isfinite(grad)))
    assert bool(jnp.any(grad != 0.0))
    lo = idx.continuum[0][0] - 1.0
    hi = idx.continuum[1][1] + 1.0
    outside = (np.asarray(wave) < lo - 1.0) | (np.asarray(wave) > hi + 1.0)
    assert bool(
        jnp.all(grad[outside] == 0.0)
    )  # hard edges: nothing beyond one pixel sees the flux
    direction = jnp.cos(wave / 11.0) * flux
    h = 1e-6
    fd = (
        measure_index_jax(wave, flux + h * direction, idx)
        - measure_index_jax(wave, flux - h * direction, idx)
    ) / (2.0 * h)
    assert float(jnp.vdot(grad, direction)) == pytest.approx(float(fd), rel=1e-6)
