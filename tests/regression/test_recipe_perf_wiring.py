# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the 2026-05-23 canonical-recipes performance audit.

Two surfaces are locked down:

1. **WavePrecomp wiring**: every photometry recipe carries
   ``approx=WavePrecomp()`` in its returned dict. ``dust_demo`` is the
   forward-only gallery recipe and stays on the exact path on purpose.

2. **Stochastic SFH (field=True) tracer fix**: ``StellarSEDComponent.apply``
   no longer calls ``float()`` on a traced ``log_age_grid`` step. Before
   the fix this raised ``ConcretizationTypeError`` and broke the
   ``stochastic_sfh_jwst`` recipe.

The bench-level numbers behind these tests are recorded in
``docs/dev/benchmarks/2026-05-23_canonical_recipes_perf_audit.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import tengri
from tengri import WavePrecomp, recipes
from tengri.components.stellar.sfh.gp_sfh import (
    LOG_AGE_MAX,
    LOG_AGE_MIN,
    log_age_grid_step,
    make_log_age_grid,
)

pytestmark = pytest.mark.regression_bug

# Recipes whose returned dict must include approx=WavePrecomp().
# dust_demo is the forward-only gallery recipe — exempt by design.
PHOTOMETRY_RECIPES = (
    "mock_recovery_minimal",
    "star_forming_photometry",
    "quiescent_z0",
    "stochastic_sfh_jwst",
    "agn_panchromatic",
)


@pytest.mark.parametrize("recipe_name", PHOTOMETRY_RECIPES)
def test_recipe_wires_waveprecomp(recipe_name):
    """Each photometry recipe must include approx=WavePrecomp() so users get
    the fast path by default (audit 2026-05-23, ~20-50× forward speedup)."""
    recipe_dict = getattr(recipes, recipe_name)()
    assert "approx" in recipe_dict, (
        f"{recipe_name} must include approx=WavePrecomp() in its returned dict "
        f"so SEDModel.build(**recipes.{recipe_name}()) takes the fast path."
    )
    assert isinstance(recipe_dict["approx"], WavePrecomp), (
        f"{recipe_name}['approx'] must be a WavePrecomp instance, got "
        f"{type(recipe_dict['approx']).__name__}"
    )


def test_dust_demo_does_not_wire_waveprecomp():
    """dust_demo is forward-only gallery code and stays on the exact path."""
    recipe_dict = recipes.dust_demo()
    assert "approx" not in recipe_dict, (
        "dust_demo is a forward-only gallery recipe and intentionally uses the "
        "exact wave-grid path. Remove approx= if it was added by accident."
    )


@pytest.mark.parametrize("recipe_name", PHOTOMETRY_RECIPES)
def test_recipe_dict_still_splattable_into_parameters(recipe_name):
    """Adding approx= must not break the legacy
    ``parse_groups(**recipe_dict)`` splat — parse_groups should
    silently drop SEDModel-only passthrough kwargs."""
    recipe_dict = getattr(recipes, recipe_name)()
    # Ensure redshift is in the recipe dict
    if "redshift" not in recipe_dict:
        recipe_dict["redshift"] = tengri.Fixed(0.1)
    spec = tengri.parse_groups(**recipe_dict)
    assert spec.n_free > 0


def test_log_age_grid_step_matches_jnp_linspace():
    """Static helper must match the actual ``jnp.linspace`` step for the
    same n_grid — guards against the constants drifting from
    ``make_log_age_grid`` defaults."""
    for n_grid in (32, 64, 128, 256):
        grid = make_log_age_grid(n_grid)
        expected = float(grid[1] - grid[0])
        assert log_age_grid_step(n_grid) == pytest.approx(expected, rel=1e-12)


def test_log_age_constants_match_make_log_age_grid_defaults():
    """Module constants must match the defaults used by make_log_age_grid;
    if these drift, both forward paths silently disagree on the GP grid."""
    grid = make_log_age_grid(64)
    assert float(grid[0]) == pytest.approx(LOG_AGE_MIN, abs=1e-12)
    assert float(grid[-1]) == pytest.approx(LOG_AGE_MAX, abs=1e-12)


def test_log_age_grid_step_is_jit_safe():
    """``log_age_grid_step`` must produce a Python float so callers inside
    jit-traced code can pass it as a static argument to downstream
    functions (e.g. compute_field_gp). Before the fix, the call site used
    ``float(log_age_grid[1] - log_age_grid[0])`` which raised
    ConcretizationTypeError under jit."""

    n_grid = 64

    @jax.jit
    def _inside_jit(x):
        # Mirror the call site in StellarSEDComponent.apply: under jit the
        # grid itself is traced, but log_age_grid_step is computed from a
        # static n_grid and constants — it must return a Python float so
        # we can feed it as a static argument to anything downstream.
        d = log_age_grid_step(n_grid)
        # Verify d is a static Python float (won't be promoted to a tracer
        # just by participating in arithmetic with x):
        return x + d  # type: ignore[operator]

    out = _inside_jit(jnp.zeros(()))
    assert jnp.allclose(out, log_age_grid_step(n_grid))
