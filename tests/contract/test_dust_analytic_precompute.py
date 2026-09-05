# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for the analytic dust-emission precompute adapter (PR 3).

Covers ``modified_blackbody``, ``casey2012``, ``graybody``, and the ``pah_drude`` template.
"""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def filter_set():
    centers = np.array([7e4, 1.5e5, 5e5, 1e6, 5e6])  # MIR–FIR Angstrom
    widths = np.array([1e4, 3e4, 1e5, 2e5, 5e5])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


@pytest.mark.parametrize("model", ["modified_blackbody", "casey2012", "graybody", "pah_drude"])
def test_dust_analytic_precompute_and_lookup(model, filter_set):
    from tengri.components.dust import dust_analytic_precompute as adapter

    assert model in adapter.AXIS_PARAMS

    waves, trans = filter_set
    result = adapter.precompute(waves, trans, redshift=1.0, parameters=None, model=model)
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


@pytest.mark.parametrize("key", ["modified_blackbody", "casey2012", "graybody", "pah_drude"])
def test_dust_analytic_registered(key):
    from tengri.forward.precompute.registry import _REGISTRY

    assert _REGISTRY[key].endswith("dust_analytic_precompute")
