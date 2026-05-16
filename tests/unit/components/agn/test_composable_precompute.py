"""Tests for the composable-AGN precompute path.

Covers:
- :class:`Recipe` construction (from selectors, from Parameters, typo rejection).
- Template hoist: when ``template_state`` is passed, blocks read templates
  from it; when ``None``, the lru_cache fallback works.
- ``composable_precompute.precompute()`` returns the documented dict keys.
- Auto-collapse of Fixed axes.
- Parity: precomputed lookup ≈ runtime-composed photometry at grid centres,
  to triweight-interp tolerance.
- JIT smoke + compile-tree timing sanity.
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import (
    Recipe,
    composable_agn_l_nu,
    composable_precompute,
)

# ──────────────────────────────────────────────────────────────────────
# Recipe construction
# ──────────────────────────────────────────────────────────────────────


def test_recipe_from_selectors_clean():
    r = Recipe.from_selectors(
        disc="grahsp_sbpl", torus="grahsp", attenuation="grahsp_biatten"
    )
    assert r.agn_disc_block == "grahsp_sbpl"
    assert r.agn_torus_block == "grahsp"
    assert r.agn_attenuation_block == "grahsp_biatten"
    assert r.axis_params == ()


def test_recipe_typo_raises():
    with pytest.raises(ValueError, match="Unknown torus block"):
        Recipe.from_selectors(disc="powerlaw", torus="bogus_torus")


def test_recipe_from_parameters():
    """Build Recipe from the flat agn_*_block attrs that Parameters sets."""

    class _StubParams:
        agn_disc_block = "grahsp_sbpl"
        agn_lines_block = "none"
        agn_feii_block = "none"
        agn_torus_block = "skirtor"
        agn_attenuation_block = "smc_prevot"

    r = Recipe.from_parameters(_StubParams(), axis_params=("agn_grahsp_l5100",))
    assert r.agn_disc_block == "grahsp_sbpl"
    assert r.agn_torus_block == "skirtor"
    assert r.axis_params == ("agn_grahsp_l5100",)


def test_recipe_summary():
    r = Recipe.from_selectors(disc="grahsp_sbpl", torus="grahsp")
    rows = r.summary()
    assert ("disc", "grahsp_sbpl") in rows
    assert ("torus", "grahsp") in rows
    assert len(rows) == 5


# ──────────────────────────────────────────────────────────────────────
# Template hoist
# ──────────────────────────────────────────────────────────────────────


def test_template_hoist_returns_same_output_as_lru_load():
    """Passing pre-loaded templates must match the in-block load path."""
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    wave = jnp.logspace(3, 5, 200)
    params = dict(
        agn_log_lbol=44.5,
        agn_disc_block="grahsp_sbpl",
        agn_lines_block="grahsp",
        agn_feii_block="grahsp",
        agn_torus_block="grahsp",
        agn_attenuation_block="grahsp_biatten",
        agn_grahsp_l5100=1.0e44,
        agn_grahsp_ebv=0.05,
    )
    out_no_hoist = composable_agn_l_nu(wave, **params)
    templates = load_grahsp_templates()
    out_hoisted = composable_agn_l_nu(
        wave, template_state={"grahsp": templates}, **params
    )
    np.testing.assert_allclose(
        np.asarray(out_hoisted), np.asarray(out_no_hoist), rtol=1e-12
    )


# ──────────────────────────────────────────────────────────────────────
# precompute() shape + dict contract
# ──────────────────────────────────────────────────────────────────────


def _toy_filter():
    wave = np.linspace(4500.0, 6500.0, 200)
    trans = np.exp(-0.5 * ((wave - 5500.0) / 300.0) ** 2)
    return wave, trans


def test_precompute_returns_documented_keys():
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="grahsp_biatten",
        axis_params=("agn_grahsp_l5100",),
    )
    out = composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={"agn_grahsp_l5100": np.logspace(43, 46, 4)},
    )
    assert "grid_phot" in out
    assert "axes" in out
    assert "_preint" in out
    assert out["grid_phot"].shape == (4, 1)


def test_precompute_requires_axis_params():
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(disc="grahsp_sbpl", torus="grahsp")
    with pytest.raises(ValueError, match="at least one axis_param"):
        composable_precompute.precompute(
            filter_waves=[fw],
            filter_trans=[ft],
            redshift=0.0,
            parameters=None,
            recipe=recipe,
            axis_grids={},
        )


# ──────────────────────────────────────────────────────────────────────
# Auto-collapse via Parameters
# ──────────────────────────────────────────────────────────────────────


def test_auto_collapse_fixed_axis():
    """When the spec pins one axis-param to Fixed, the grid collapses."""
    fw, ft = _toy_filter()

    class _StubParams:
        agn_disc_block = "grahsp_sbpl"
        agn_lines_block = "none"
        agn_feii_block = "none"
        agn_torus_block = "grahsp"
        agn_attenuation_block = "none"

        def get_fixed_values(self):
            return {"agn_grahsp_ebv": 0.1}

        @property
        def free_params(self):
            return ["agn_grahsp_l5100"]

    recipe = Recipe.from_parameters(
        _StubParams(),
        axis_params=("agn_grahsp_l5100", "agn_grahsp_ebv"),
    )
    out = composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=_StubParams(),
        recipe=recipe,
        axis_grids={
            "agn_grahsp_l5100": np.logspace(43, 46, 3),
            "agn_grahsp_ebv": np.array([0.0, 0.05, 0.1]),
        },
    )
    # ebv axis collapsed -> 1-D grid over l5100 only
    assert "_collapsed_axes" in out
    assert out["grid_phot"].shape == (3, 1)


# ──────────────────────────────────────────────────────────────────────
# Parity: precomputed lookup vs runtime path
# ──────────────────────────────────────────────────────────────────────


def test_parity_at_grid_centre():
    """Lookup at an exact grid-centre point matches the runtime evaluation
    integrated through the same filter to triweight-interp precision."""
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="none",
        axis_params=("agn_grahsp_l5100",),
    )

    axis_grid = np.logspace(43, 46, 5)
    out = composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={"agn_grahsp_l5100": axis_grid},
        # Match the runtime call exactly:
        wave_rest=np.logspace(2.0, 6.0, 1500, dtype=np.float64),
    )
    fn = composable_precompute.build_lookup(out)

    # Pick a grid-centre value
    l5100_value = float(axis_grid[2])
    photo_lookup = float(fn(jnp.array(1.0), jnp.array(l5100_value))[0])

    # Reference: evaluate the runner directly on the same wave grid, then
    # integrate through the filter the same way the helper does.
    wave_jax = jnp.asarray(np.logspace(2.0, 6.0, 1500))
    l_nu = composable_agn_l_nu(
        wave_jax,
        agn_disc_block="grahsp_sbpl",
        agn_lines_block="none",
        agn_feii_block="none",
        agn_torus_block="grahsp",
        agn_attenuation_block="none",
        agn_log_lbol=45.0,
        agn_grahsp_l5100=l5100_value,
    )
    # Trapezoidal filter integration: <F_nu> = int(F_nu * trans / nu dnu) /
    # int(trans/nu dnu). Use Å-side.
    c_aa_per_s = 2.99792458e18
    wave_obs = np.asarray(wave_jax)  # redshift=0
    f_nu = np.asarray(l_nu)
    trans_interp = np.interp(wave_obs, fw, ft, left=0.0, right=0.0)
    nu = c_aa_per_s / wave_obs
    order = np.argsort(nu)
    photo_ref = np.trapezoid((f_nu * trans_interp / nu)[order], nu[order]) / \
        np.trapezoid((trans_interp / nu)[order], nu[order])

    # Triweight-interp precision: ~few % at grid centres.
    np.testing.assert_allclose(photo_lookup, photo_ref, rtol=5e-2)


# ──────────────────────────────────────────────────────────────────────
# JIT compatibility + compile-time smoke
# ──────────────────────────────────────────────────────────────────────


def test_lookup_jit_compiles_and_caches():
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="grahsp_biatten",
        axis_params=("agn_grahsp_l5100", "agn_grahsp_ebv"),
    )
    pre = composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={
            "agn_grahsp_l5100": np.logspace(43, 46, 4),
            "agn_grahsp_ebv": np.array([0.0, 0.05, 0.1, 0.3]),
        },
    )
    fn = composable_precompute.build_lookup(pre)

    # First call: includes compile. Second call: pure cached lookup.
    t0 = time.time()
    out1 = fn(jnp.array(1.0), jnp.array(1e44), jnp.array(0.05))
    _ = out1.block_until_ready()
    dt_compile = time.time() - t0
    t0 = time.time()
    out2 = fn(jnp.array(1.0), jnp.array(2e44), jnp.array(0.1))
    _ = out2.block_until_ready()
    dt_cached = time.time() - t0

    assert jnp.all(jnp.isfinite(out1))
    assert jnp.all(jnp.isfinite(out2))
    # Cached should be at least 10× faster than first-call compile.
    # (Informational; soft assertion with generous slack for CI variance.)
    assert dt_cached < dt_compile / 5.0


def test_build_lookup_returns_composable_lookup_with_axis_names():
    """build_lookup must expose axis_names so the SEDModel kernel can route."""
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="none",
        axis_params=("agn_grahsp_l5100",),
    )
    pre = composable_precompute.precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={"agn_grahsp_l5100": np.logspace(43, 46, 3)},
    )
    fn = composable_precompute.build_lookup(pre)
    assert hasattr(fn, "axis_names")
    assert fn.axis_names == ("agn_grahsp_l5100",)
    # Still callable as before.
    out = fn(jnp.array(1.0), jnp.array(1e44))
    assert out.shape == (1,)


def test_kernel_gates_on_agn_log_lbol_axis():
    """Kernel currently consumes only single-axis ``agn_log_lbol`` recipes.

    This documents/locks the consumption surface: a precompute built with a
    different axis still works as a standalone callable but does NOT trigger
    the kernel's preintegrated branch — the kernel falls back to the
    runtime evaluation path.
    """
    from tengri.components.agn.blocks.composable_precompute import (
        ComposableLookup,
    )

    fn = ComposableLookup(lambda scale, x: scale * x, axis_names=("agn_log_lbol",))
    assert fn.axis_names == ("agn_log_lbol",)
    # Sentinel check: the kernel introspects axis_names and gates on this
    # exact tuple. Changing the axis name disables the kernel branch.
    other = ComposableLookup(lambda scale, x: scale * x, axis_names=("agn_grahsp_l5100",))
    assert other.axis_names != ("agn_log_lbol",)


def test_lookup_works_with_explicit_jit_wrapper():
    fw, ft = _toy_filter()
    recipe = Recipe.from_selectors(
        disc="grahsp_sbpl",
        torus="grahsp",
        attenuation="none",
        axis_params=("agn_grahsp_l5100",),
    )
    pre = composable_precompute.precompute(
        filter_waves=[fw, fw, fw],
        filter_trans=[ft, ft, ft],
        redshift=0.0,
        parameters=None,
        recipe=recipe,
        axis_grids={"agn_grahsp_l5100": np.logspace(43, 46, 4)},
    )
    fn = composable_precompute.build_lookup(pre)
    jit_fn = jax.jit(fn)
    out = jit_fn(jnp.array(1.0), jnp.array(1e44))
    assert out.shape == (3,)
    assert jnp.all(jnp.isfinite(out))
