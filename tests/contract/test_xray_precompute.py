# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for the X-ray precompute adapter (PR 6)."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def filter_set():
    centers = np.array([1.0, 5.0, 50.0, 500.0])  # 0.1–100 keV in Angstrom
    widths = np.array([0.3, 1.5, 15.0, 150.0])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(max(c - 3 * w, 1e-2), c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


@pytest.mark.parametrize("model", ["xray_xrb", "xray_corona", "xray_corona_lopez24"])
def test_xray_precompute_and_lookup(model, filter_set):
    from tengri.components.xray import xray_precompute as adapter

    assert model in adapter.AXIS_PARAMS

    waves, trans = filter_set
    result = adapter.precompute(waves, trans, redshift=0.5, parameters=None, model=model)
    phot = np.asarray(result["grid_phot"])
    assert phot.shape[-1] == len(waves)
    chex.assert_tree_all_finite(phot)
    lookup = adapter.build_lookup(result, model=model)
    n_axes = len(adapter.AXIS_PARAMS[model])
    args = (
        jnp.float64(1.0),
        *tuple(
            jnp.asarray(np.asarray(ax)[len(np.asarray(ax)) // 2], dtype=jnp.float64)
            for ax in result["axes"]
        ),
    )
    assert len(args) == 1 + n_axes
    out = jax.jit(lookup)(*args)
    chex.assert_tree_all_finite(np.asarray(out))


@pytest.mark.parametrize("key", ["xray_xrb", "xray_corona", "xray_corona_lopez24"])
def test_xray_registered(key):
    from tengri.forward.precompute.registry import _REGISTRY

    assert _REGISTRY[key].endswith("xray_precompute")
