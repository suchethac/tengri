# SPDX-License-Identifier: BSD-3-Clause
"""Tests for tengri.components.stellar.sfh.sample_sfh_prior."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import pytest

from tengri import Uniform
from tengri.components.stellar.sfh import (
    DEFAULT_AGE_GRID_YR,
    sample_sfh_prior,
)


@pytest.mark.parametrize("family", ["dpl", "tsnorm", "snorm", "norm", "exp", "dexp"])
def test_shape_matches_default_grid(family: str) -> None:
    age, curves = sample_sfh_prior(family, jax.random.PRNGKey(0), n=4)
    chex.assert_equal_shape([age, DEFAULT_AGE_GRID_YR])
    chex.assert_shape(curves, (4, age.shape[0]))


def test_curves_are_finite_and_non_negative() -> None:
    _, curves = sample_sfh_prior("tsnorm", jax.random.PRNGKey(0), n=8)
    chex.assert_tree_all_finite(curves)
    # SFR should be non-negative for all the smooth additive families.
    assert jnp.all(curves >= -1e-12)


def test_default_age_grid_is_the_helper_grid() -> None:
    """``DEFAULT_AGE_GRID_YR`` is a host-side twin of ``make_log_age_grid`` (#2271).

    #1402 made ``make_log_age_grid`` the single definition of the log-age grid
    because a second copy can desync without any test failing; the host-side
    constant is that second copy, so this pins it to the helper at float64
    rounding. The device build is the same 256 nodes over the same bounds.
    """
    from tengri.components.stellar.sfh.gp_sfh import make_log_age_grid
    from tengri.utils.grid import DEFAULT_N_GRID

    with jax.enable_x64(True):
        helper = 10.0 ** make_log_age_grid(n_grid=DEFAULT_N_GRID)
    chex.assert_trees_all_close(jnp.asarray(DEFAULT_AGE_GRID_YR), helper, rtol=1e-14, atol=0.0)


def test_deterministic_given_key() -> None:
    _, c1 = sample_sfh_prior("dpl", jax.random.PRNGKey(7), n=5)
    _, c2 = sample_sfh_prior("dpl", jax.random.PRNGKey(7), n=5)
    assert jnp.allclose(c1, c2)


def test_different_keys_give_different_curves() -> None:
    _, c1 = sample_sfh_prior("dpl", jax.random.PRNGKey(0), n=5)
    _, c2 = sample_sfh_prior("dpl", jax.random.PRNGKey(1), n=5)
    assert not jnp.allclose(c1, c2)


def test_prior_override_narrows_range() -> None:
    """Tightening the alpha prior should reduce variance across draws (loosely)."""
    key = jax.random.PRNGKey(0)
    _, wide = sample_sfh_prior("dpl", key, n=64)
    _, narrow = sample_sfh_prior("dpl", key, n=64, sfh_dpl_alpha=Uniform(0.95, 1.05))
    # Same key + same overrides => identical alpha range, but narrower alpha
    # generally lowers cross-draw variance at late times. Use a weak inequality.
    assert float(narrow.std()) <= float(wide.std()) + 1e-6


pytestmark = pytest.mark.bounds


def test_composed_tsnorm_burst() -> None:
    age, curves = sample_sfh_prior(["tsnorm", "burst"], jax.random.PRNGKey(0), n=3)
    chex.assert_shape(curves, (3, age.shape[0]))
    chex.assert_tree_all_finite(curves)


def test_field_modulator_rejected() -> None:
    with pytest.raises(NotImplementedError, match="field"):
        sample_sfh_prior(["tsnorm", "field"], jax.random.PRNGKey(0))


def test_unknown_family_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="garbage"):
        sample_sfh_prior("garbage", jax.random.PRNGKey(0))


def test_custom_age_grid() -> None:
    custom_grid = jnp.linspace(1e8, 1e10, 64)
    age, curves = sample_sfh_prior("dpl", jax.random.PRNGKey(0), n=2, age_grid_yr=custom_grid)
    chex.assert_shape(age, (64,))
    chex.assert_shape(curves, (2, 64))
