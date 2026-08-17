# SPDX-License-Identifier: BSD-3-Clause
r"""Nebular line flux was not conserved outside the SSP grid's dense window (#1836).

Emission lines are rendered as profiles normalized to **unit area in frequency**,

.. math::

    \int \phi_i(\nu)\,{\rm d}\nu = 1
    \quad\Longrightarrow\quad
    \int L_\nu^{\rm lines}\,{\rm d}\nu = \sum_i L_i ,

which is exact in the continuum limit and false on a grid that does not resolve
:math:`\phi_i`. The triweight support is :math:`2h_i = 6\sigma_{\lambda,i}
= 6(\sigma_v/c)\lambda_i` — 2.4 Å at Lyα for the default
``neb_eline_sigma_kms = 100``.

The FSPS/MILES SSP grid is not uniform: 4423 of its 6185 points lie inside
3500–7500 Å at 0.9 Å spacing, and the rest is 8 Å (< 1000 Å), 10 Å (1000–3000 Å),
20–50 Å (7500–20000 Å) and up to 29000 Å beyond 1e5 Å. So **86 of 128 Cue lines,
carrying 69.9 % of** ``sum(line_lums)``, landed on fewer than 4 grid points.

Measured recovery on that grid before the fix — every *resolved* line correct,
every under-resolved one wrong:

===============  ==========  ==================
line (Å)         pts / 2h    recovered / L
===============  ==========  ==================
1215.7 (Lyα)     0.24        **2.0742**
6564.6 (Hα)      14.59       0.9997
5008.2 [OIII]    11.13       1.0004
4862.7 (Hβ)      10.81       1.0001
9533.2 [SIII]    0.95        **1.5714**
883323           0.22        **0.0026**
===============  ==========  ==================

whole block: ``2.2489``. Refining the grid ×64 drove every line to 1.0000, which
is the control proving the profile normalization and the target were both right
and only the sampling was wrong.

**Two defects, not one.** Beyond the flux error, a line's *total* flux varied with
``line_sigma_kms`` — Lyα ran 0.0000 → 3.0436 → 2.7072 across σ_v = 50, 100,
300 km/s — so ``d(flux)/d(sigma_v)`` was non-zero for a parameter that physically
sets only the shape, and fitting a width was partly fitting a flux.

**And the three placement modes disagreed with each other**: the same 128 lines
integrated to 2.2489x (triweight), 1.5711x (Gaussian σ=2 Å) and 1.0000x (delta),
so broadband photometry differed *between duplicate implementations of the same
physics* by up to 3.04x (GALEX NUV at z=0.5).

The fix is shared helpers rather than three separate arguments:

1. Floor the profile width at half the **local** grid spacing. The old floor used
   ``0.5*median(diff(grid))`` — a *global* statistic answering a *local* question,
   and the 4423 MILES points pin that median at 0.9 Å, so it never fired where the
   grid was coarse. Half, not the full spacing: the full width doubles the support
   and lights two pixels per line where one is right.
2. Rescale each profile by its own discrete trapezoid area in ν.
3. Scatter into the nearest pixel the lines that still vanish — a sub-pixel line
   centered exactly between two nodes puts both at ``|u| = 1``, where the
   compact-support triweight is exactly zero. The grid cannot represent a width
   there, but it must not lose the light.

All three placement modes now route through the same rendering.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

_C_CGS = 2.99792458e10

# A grid with the MILES pathology in miniature: a dense optical window bracketed
# by coarse wings, so lines fall on both sides of the resolution cliff.
_COARSE_BLUE = np.arange(1000.0, 3500.0, 10.0)
_DENSE_OPT = np.arange(3500.0, 7500.0, 0.9)
_COARSE_RED = np.arange(7500.0, 60000.0, 50.0)
_GRID = np.unique(np.concatenate([_COARSE_BLUE, _DENSE_OPT, _COARSE_RED]))

# Two resolved (inside the dense window) and three not (outside it).
_LINES = np.array([1215.7, 3729.9, 4862.7, 6564.6, 9533.2, 50000.0])
_LUMS = np.array([7.2e41, 5.9e40, 3.1e40, 8.5e40, 1.9e40, 1.5e40])


def _integral_dnu(sed: np.ndarray, grid: np.ndarray) -> float:
    """|integral sed dnu| on ``grid``, the quantity photometry ultimately weighs."""
    return abs(np.trapezoid(np.asarray(sed), _C_CGS / (np.asarray(grid) * 1e-8)))


@pytest.mark.parametrize("sigma_kms", [10.0, 50.0, 100.0, 300.0, 1000.0])
def test_triweight_conserves_flux_at_every_width(sigma_kms):
    """The default forward path returns the flux it was given, on a coarse grid."""
    from tengri.components.nebular._shared import place_line_profiles_velocity

    sed = place_line_profiles_velocity(_LINES, _LUMS, _GRID, sigma_kms)
    assert _integral_dnu(sed, _GRID) == pytest.approx(_LUMS.sum(), rel=1e-6)


@pytest.mark.parametrize("sigma_kms", [10.0, 50.0, 100.0, 300.0])
def test_every_line_individually_conserves(sigma_kms):
    """Resolved or not, each line carries its own luminosity.

    Pre-fix this failed on exactly the lines outside the dense window: Lya
    recovered 2.07x and the 5e4 A line 0.0026x, while Halpha and Hbeta were fine.
    """
    from tengri.components.nebular._shared import place_line_profiles_velocity

    for w, lum in zip(_LINES, _LUMS):
        sed = place_line_profiles_velocity(np.array([w]), np.array([lum]), _GRID, sigma_kms)
        assert _integral_dnu(sed, _GRID) == pytest.approx(lum, rel=1e-6), (
            f"line {w} A lost/invented flux at sigma_v={sigma_kms}"
        )


def test_flux_is_independent_of_the_width_parameter():
    """sigma_v sets the SHAPE; it must not move the integrated flux.

    The regression this pins: Lya's recovered flux ran 0.0000 -> 3.0436 -> 2.7072
    across sigma_v = 50, 100, 300 km/s, so a fitted width was partly fitting flux.
    """
    from tengri.components.nebular._shared import place_line_profiles_velocity

    lya, lum = np.array([1215.7]), np.array([7.2e41])
    fluxes = [
        _integral_dnu(place_line_profiles_velocity(lya, lum, _GRID, s), _GRID)
        for s in (25.0, 50.0, 100.0, 200.0, 400.0)
    ]
    assert np.ptp(fluxes) / np.mean(fluxes) < 1e-6
    assert fluxes[0] == pytest.approx(float(lum[0]), rel=1e-6)


def test_the_three_placement_modes_agree_on_flux():
    """Duplicate paths to one physics must not disagree about how much light there is.

    They differ legitimately in profile *shape*; before #1836 they also differed in
    flux (2.2489x / 1.5711x / 1.0000x on the production grid).
    """
    from tengri.components.nebular._shared import place_line_profiles

    triweight = place_line_profiles(_LINES, _LUMS, _GRID, 0.0, 100.0)
    gaussian = place_line_profiles(_LINES, _LUMS, _GRID, 2.0, 0.0)
    delta = place_line_profiles(_LINES, _LUMS, _GRID, 0.0, 0.0)
    for name, sed in (("triweight", triweight), ("gaussian", gaussian), ("delta", delta)):
        assert _integral_dnu(sed, _GRID) == pytest.approx(_LUMS.sum(), rel=1e-6), name


def test_width_still_changes_the_profile_shape():
    """Conservation must not be bought by flattening the line into a delta.

    Guards the obvious wrong fix — renormalizing something that no longer responds
    to sigma_v would pass every test above.
    """
    from tengri.components.nebular._shared import place_line_profiles_velocity

    ha, lum = np.array([6564.6]), np.array([8.5e40])
    narrow = np.asarray(place_line_profiles_velocity(ha, lum, _GRID, 50.0))
    wide = np.asarray(place_line_profiles_velocity(ha, lum, _GRID, 300.0))
    # Same flux, but the wide line is lower and broader.
    assert narrow.max() > 3.0 * wide.max()
    assert int(np.sum(wide > wide.max() / 2)) > 3 * int(np.sum(narrow > narrow.max() / 2))


def test_a_sub_pixel_line_occupies_one_pixel_not_two():
    """sigma_v -> 0 must still render the NARROWEST line the grid can hold.

    The width floor is what decides this. Flooring at the full local spacing
    doubles the support and lights two pixels per line, which is a real behavior
    regression even though flux is still conserved --
    ``tests/components/nebular/test_shock_emission.py`` asserts one pixel per
    line and caught exactly that.
    """
    from tengri.components.nebular._shared import place_line_profiles_velocity

    grid = np.linspace(3000.0, 8000.0, 5000)  # uniform, ~1.0 A
    lines = np.array([4000.0, 5000.5, 6123.7])
    lums = np.array([1.0e40, 2.0e40, 3.0e40])
    sed = np.asarray(place_line_profiles_velocity(lines, lums, grid, 0.0))
    assert int(np.sum(sed > 0)) <= lines.shape[0]
    assert _integral_dnu(sed, grid) == pytest.approx(lums.sum(), rel=1e-6)


def test_a_line_exactly_between_two_nodes_is_not_lost():
    """The degenerate case the nearest-pixel fallback exists for.

    A sub-pixel line centered exactly midway puts both neighbors at ``|u| = 1``,
    where the compact-support triweight is *exactly* zero — the profile vanishes
    and the rescale cannot recover it. Pre-#1836 that line was dropped silently.
    """
    from tengri.components.nebular._shared import place_line_profiles_velocity

    grid = np.linspace(3000.0, 8000.0, 5001)  # spacing exactly 1.0 A
    midpoint = np.array([(grid[100] + grid[101]) / 2.0])
    lum = np.array([5.0e40])
    sed = np.asarray(place_line_profiles_velocity(midpoint, lum, grid, 0.0))
    assert np.all(np.isfinite(sed))
    assert _integral_dnu(sed, grid) == pytest.approx(float(lum[0]), rel=1e-6)


def test_line_outside_the_grid_contributes_nothing_and_is_not_nan():
    """The guarded divisor must leave an off-grid line at zero, never at NaN."""
    from tengri.components.nebular._shared import place_line_profiles_velocity

    sed = np.asarray(
        place_line_profiles_velocity(np.array([1.0e9]), np.array([1.0e40]), _GRID, 100.0)
    )
    assert np.all(np.isfinite(sed))
    assert _integral_dnu(sed, _GRID) == pytest.approx(0.0, abs=1e-30)


def test_a_degenerate_single_node_grid_stays_finite():
    """A grid with no extent in nu must return zero flux, never NaN.

    Self-review catch. ``_render_conserving`` divides twice -- by the profile's
    discrete area, and by the quadrature weight of the nearest node -- and only
    the first was guarded. A one-node grid has ``quad_w == 0`` identically, so
    the second division was ``0.0 / 0.0`` and put a NaN on the tape. The
    velocity path has an explicit ``shape[0] > 1`` branch, so the degenerate
    grid is a supported input, and pre-#1836 it returned a finite value.
    """
    from tengri.components.nebular._shared import place_line_profiles_velocity

    sed = np.asarray(
        place_line_profiles_velocity(
            np.array([5000.0]), np.array([1.0e40]), np.array([5000.0]), 100.0
        )
    )
    assert np.all(np.isfinite(sed)), f"single-node grid returned {sed}"


def test_delta_mode_places_an_edge_line_on_the_edge_node():
    """The nearest node is the nearest node, including the first and the last.

    Self-review catch. Delta mode clipped its index to ``[1, n_wave - 2]``, a
    constraint inherited from the old Delta-nu, which read ``obs[i +/- 1]``. The
    quadrature weight that replaced it is defined at every node, so the clip
    only displaced a line landing on either end by a full pixel -- measured at
    5 A on a 5 A grid.
    """
    from tengri.components.nebular._shared import place_line_profiles

    grid = np.linspace(3000.0, 8000.0, 1001)  # 5 A spacing
    edges = np.array([grid[0], grid[-1]])
    lums = np.array([1.0e40, 2.0e40])
    sed = np.asarray(place_line_profiles(edges, lums, grid, 0.0, 0.0))
    assert np.array_equal(np.nonzero(sed)[0], np.array([0, grid.shape[0] - 1]))
    assert _integral_dnu(sed, grid) == pytest.approx(lums.sum(), rel=1e-6)


def test_jit_and_grad_survive_the_rescale():
    """The fix must stay usable where the forward model puts it: inside jit/grad."""
    import jax
    import jax.numpy as jnp

    from tengri.components.nebular._shared import place_line_profiles_velocity

    grid = jnp.asarray(_GRID)
    lines, lums = jnp.asarray(_LINES), jnp.asarray(_LUMS)

    @jax.jit
    def total(sigma):
        sed = place_line_profiles_velocity(lines, lums, grid, sigma)
        return jnp.abs(jnp.trapezoid(sed, _C_CGS / (grid * 1e-8)))

    assert float(total(100.0)) == pytest.approx(float(_LUMS.sum()), rel=1e-6)
    # Flux no longer depends on the width, so this gradient is now ~0 by design.
    g = float(jax.grad(total)(100.0))
    assert np.isfinite(g)
    assert abs(g) / _LUMS.sum() < 1e-9
