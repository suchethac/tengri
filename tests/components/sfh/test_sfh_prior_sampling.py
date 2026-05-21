"""Tests for tengri.components.stellar.sfh.sample_sfh_prior."""

from __future__ import annotations

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
    assert age.shape == DEFAULT_AGE_GRID_YR.shape
    assert curves.shape == (4, age.shape[0])


def test_curves_are_finite_and_non_negative() -> None:
    _, curves = sample_sfh_prior("tsnorm", jax.random.PRNGKey(0), n=8)
    assert jnp.all(jnp.isfinite(curves))
    # SFR should be non-negative for all the smooth additive families.
    assert jnp.all(curves >= -1e-12)


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
    assert curves.shape == (3, age.shape[0])
    assert jnp.all(jnp.isfinite(curves))


def test_field_modulator_rejected() -> None:
    with pytest.raises(NotImplementedError, match="field"):
        sample_sfh_prior(["tsnorm", "field"], jax.random.PRNGKey(0))


def test_unknown_family_raises_keyerror() -> None:
    with pytest.raises(KeyError, match="garbage"):
        sample_sfh_prior("garbage", jax.random.PRNGKey(0))


def test_custom_age_grid() -> None:
    custom_grid = jnp.linspace(1e8, 1e10, 64)
    age, curves = sample_sfh_prior("dpl", jax.random.PRNGKey(0), n=2, age_grid_yr=custom_grid)
    assert age.shape == (64,)
    assert curves.shape == (2, 64)
