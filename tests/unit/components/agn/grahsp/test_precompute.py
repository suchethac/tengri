"""Tests for the GRAHSP precompute path (filter-integrated photometry)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def _toy_filter():
    """A single Gaussian-ish filter centred at 5500 Å (rest), useful for tests."""
    wave = np.linspace(4500.0, 6500.0, 200)
    trans = np.exp(-0.5 * ((wave - 5500.0) / 300.0) ** 2)
    return wave, trans


def test_precompute_returns_expected_shapes():
    from tengri.components.agn.grahsp.precompute import precompute

    fw, ft = _toy_filter()
    out = precompute(
        filter_waves=[fw, fw],
        filter_trans=[ft, ft],
        redshift=0.0,
    )
    # Default grid is plslope (3) x ebv (5) x n_filters (2)
    assert out["grid_phot"].shape == (3, 5, 2)
    assert len(out["axes"]) == 2


def test_precompute_collapses_fixed_axes():
    """When the user pins a parameter, the corresponding axis collapses."""
    from tengri.components.agn.grahsp.precompute import precompute

    fw, ft = _toy_filter()

    # Build a Parameters object that fixes ebv but leaves plslope free.
    # NOTE: tengri.Parameters auto-detects component params; using a
    # mock-ish minimal stub is simpler than configuring the full SFH suite.
    class _StubParams:
        def __init__(self, fixed_values, free_params):
            self._fv = fixed_values
            self._fp = free_params

        def get_fixed_values(self):
            return self._fv

        @property
        def free_params(self):
            return self._fp

    params = _StubParams(
        fixed_values={"agn_grahsp_ebv": 0.1},
        free_params=["agn_grahsp_plslope"],
    )
    out = precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
        parameters=params,
    )
    # ebv axis collapsed -> only plslope (3) remains
    assert "_collapsed_axes" in out
    assert out["grid_phot"].shape == (3, 1)


def test_runtime_lookup_returns_finite_photometry():
    from tengri.components.agn.grahsp.precompute import build_lookup, precompute

    fw, ft = _toy_filter()
    pre = precompute(
        filter_waves=[fw],
        filter_trans=[ft],
        redshift=0.0,
    )
    fn = build_lookup(pre)
    # Default lookup: (scale, *grid_params). GRAHSP normalises internally so
    # the natural scale is 1.0; the grid points are (plslope, ebv).
    out = fn(jnp.array(1.0), jnp.array(-1.7), jnp.array(0.1))
    assert jnp.all(jnp.isfinite(out))
    assert out.shape == (1,)


def test_runtime_lookup_jit():
    import jax

    from tengri.components.agn.grahsp.precompute import build_lookup, precompute

    fw, ft = _toy_filter()
    pre = precompute(
        filter_waves=[fw, fw, fw],
        filter_trans=[ft, ft, ft],
        redshift=0.5,
    )
    fn = build_lookup(pre)
    fn_jit = jax.jit(fn)
    out1 = fn_jit(jnp.array(1.0), jnp.array(-1.7), jnp.array(0.1))
    out2 = fn_jit(jnp.array(1.0), jnp.array(-2.0), jnp.array(0.5))
    assert out1.shape == (3,)
    assert out2.shape == (3,)
    assert jnp.all(jnp.isfinite(out1))
    assert jnp.all(jnp.isfinite(out2))
