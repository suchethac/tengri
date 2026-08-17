# SPDX-License-Identifier: BSD-3-Clause
"""The dense-integrand resolution is chosen per SFH family, and must stay so.

A tabulated history is piecewise-*linear* between nodes that #765 already
injects as quadrature knots, so it converges fast; a binned family is
piecewise-*constant*, and every bin edge is a step the quadrature has to
resolve. Measured on one fixture at 11 bands, changing one thing at a time
(2026-08-17), median photometry error:

==================================  ==========  ==============
error source                        tabulated   non-parametric
==================================  ==========  ==============
WavePrecomp (LUT vs exact)             6.7e-05         7.6e-05
integrand at ``factor=16``             8.6e-06         2.3e-04
integrand at ``factor=8``              2.3e-05         7.3e-04
==================================  ==========  ==============

So ``8`` is below the LUT error a tabulated fit already accepts (#1747 made the
LUT the default on every fit surface), and ~10x *above* it for a binned family
whose integrand error already dominates at 16.

These tests pin the routing. The failure they exist to catch is someone
collapsing the two constants back into one — in either direction. Lowering the
parametric factor degrades the term that already dominates there; raising the
tabulated one silently gives back 1.47x the FLOPs and 1.61x the bytes of the
whole forward model for precision the LUT discards.
"""

import numpy as np
import pytest

from tengri.components.stellar import component as comp
from tengri.components.stellar.component import (
    INTEGRAND_FACTOR_PARAMETRIC,
    INTEGRAND_FACTOR_TABULATED,
)

pytestmark = pytest.mark.regression_bug


def test_tabulated_refinement_is_coarser_than_parametric():
    """The whole point: the two are different, and which way round."""
    assert INTEGRAND_FACTOR_TABULATED < INTEGRAND_FACTOR_PARAMETRIC, (
        "a tabulated history converges faster than a piecewise-constant one; "
        f"got tabulated={INTEGRAND_FACTOR_TABULATED} >= "
        f"parametric={INTEGRAND_FACTOR_PARAMETRIC}"
    )
    # Both must stay integers >= 1 or the grid construction is meaningless.
    for name, value in (
        ("INTEGRAND_FACTOR_TABULATED", INTEGRAND_FACTOR_TABULATED),
        ("INTEGRAND_FACTOR_PARAMETRIC", INTEGRAND_FACTOR_PARAMETRIC),
    ):
        assert isinstance(value, int) and value >= 1, f"{name} must be a positive int"


def test_refine_grid_size_follows_the_factor():
    """``factor`` is the number of sub-samples per SSP age interval."""
    ssp_ages = np.logspace(5.0, 10.1, 40)

    for factor in (INTEGRAND_FACTOR_TABULATED, INTEGRAND_FACTOR_PARAMETRIC):
        grid = np.asarray(comp._refine_sfh_table_ages(ssp_ages, factor=factor))
        assert grid.shape == ((40 - 1) * factor + 1,)
        assert np.all(np.diff(grid) > 0.0), "grid must be strictly ascending"


def _factors_used(monkeypatch):
    """Record every ``factor`` ``_cic_integrand`` asks the refiner for."""
    seen = []
    orig = comp._refine_sfh_table_ages

    def spy(ssp_ages_yr, factor=INTEGRAND_FACTOR_PARAMETRIC):
        seen.append(factor)
        return orig(ssp_ages_yr, factor=factor)

    monkeypatch.setattr(comp, "_refine_sfh_table_ages", spy)
    return seen


def test_cic_integrand_routes_tabulated_to_the_tabulated_factor(monkeypatch):
    """A history with its own lookback nodes gets the coarser grid."""
    seen = _factors_used(monkeypatch)
    ssp_ages = np.logspace(5.0, 10.1, 40)
    tab_lbt = np.linspace(1e6, 1e10, 24)

    comp._cic_integrand(
        ssp_ages,
        lambda age, **kw: np.ones_like(np.asarray(age, dtype=float)),
        {},
        None,
        tab_lbt,
    )

    assert seen == [INTEGRAND_FACTOR_TABULATED], (
        f"tabulated path asked for {seen}, expected [{INTEGRAND_FACTOR_TABULATED}]"
    )


def test_cic_integrand_routes_non_tabulated_to_the_parametric_factor(monkeypatch):
    """``tab_lbt_yr=None`` — every parametric and binned family — keeps 16."""
    seen = _factors_used(monkeypatch)
    ssp_ages = np.logspace(5.0, 10.1, 40)

    comp._cic_integrand(
        ssp_ages,
        lambda age, **kw: np.ones_like(np.asarray(age, dtype=float)),
        {},
        None,
        None,
    )

    assert seen == [INTEGRAND_FACTOR_PARAMETRIC], (
        f"non-tabulated path asked for {seen}, expected [{INTEGRAND_FACTOR_PARAMETRIC}]"
    )


def test_tabulated_grid_is_actually_smaller(monkeypatch):
    """Not just a different constant — fewer quadrature nodes reach the kernel.

    Guards against the routing being correct while something downstream
    re-densifies the grid and gives the saving back.
    """
    ssp_ages = np.logspace(5.0, 10.1, 40)
    tab_lbt = np.linspace(1e6, 1e10, 24)
    sfr_fn = lambda age, **kw: np.ones_like(np.asarray(age, dtype=float))  # noqa: E731

    tab_grid, _ = comp._cic_integrand(ssp_ages, sfr_fn, {}, None, tab_lbt)
    par_grid, _ = comp._cic_integrand(ssp_ages, sfr_fn, {}, None, None)

    assert tab_grid.shape[0] < par_grid.shape[0], (
        f"tabulated integrand has {tab_grid.shape[0]} nodes, parametric "
        f"{par_grid.shape[0]} — the coarser factor did not reach the grid"
    )
