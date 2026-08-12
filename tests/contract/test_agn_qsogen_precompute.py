# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for the QSOgen precompute adapter (PR 1)."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._jit_parity import assert_jit_matches_eager

pytestmark = pytest.mark.contract

jax.config.update("jax_enable_x64", True)


@pytest.fixture(scope="module")
def filter_set():
    centers = np.array([1500.0, 4500.0, 6500.0, 12000.0])
    widths = np.array([300.0, 800.0, 1000.0, 1500.0])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


def test_precompute_shape_finite(filter_set):
    from tengri.components.agn import qsogen_precompute

    waves, trans = filter_set
    result = qsogen_precompute.precompute(waves, trans, redshift=1.0, parameters=None)
    phot = np.asarray(result["grid_phot"])
    assert phot.shape[-1] == len(waves)
    chex.assert_tree_all_finite(phot)


def test_lookup_jit(filter_set):
    from tengri.components.agn import qsogen_precompute

    waves, trans = filter_set
    result = qsogen_precompute.precompute(waves, trans, redshift=1.0, parameters=None)
    lookup = qsogen_precompute.build_lookup(result)
    n_axes = len(qsogen_precompute.AXIS_PARAMS)
    args = (jnp.float64(1.0), *tuple(jnp.float64(0.0) for _ in range(n_axes)))
    out = assert_jit_matches_eager(lookup, *args)
    chex.assert_tree_all_finite(np.asarray(out))


def test_qsogen_registered():
    from tengri.forward.precompute.registry import _REGISTRY

    assert "qsogen" in _REGISTRY
    assert _REGISTRY["qsogen"].endswith("qsogen_precompute")
