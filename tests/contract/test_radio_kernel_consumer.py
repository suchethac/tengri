# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for radio kernel consumer integration (PR 5)."""

from __future__ import annotations

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def filter_set():
    """Simple 3-filter set (FIR, millimeter, centimeter)."""
    centers = np.array([1e5, 1e7, 1e8])  # FIR, mm, cm Angstrom
    widths = np.array([3e4, 3e6, 3e7])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 32)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


@pytest.mark.parametrize("model_name", ["radio_synchrotron", "radio_freefree", "radio_agn_jet"])
def test_radio_lookups_jit_compatible(model_name, filter_set):
    """Verify precompute lookups are JIT-compatible and produce valid outputs."""
    from tengri.components.radio import radio_precompute as adapter

    waves, trans = filter_set
    result = adapter.precompute(waves, trans, redshift=0.5, parameters=None, model=model_name)
    lookup = adapter.build_lookup(result, model=model_name)

    # Test JIT compilation
    jitted_lookup = jax.jit(lookup)

    # Test call with scale and axis values
    _L_REF = 1.0e44  # erg/s
    n_axes = len(adapter.AXIS_PARAMS[model_name])

    if model_name == "radio_synchrotron":
        alpha_grid = np.linspace(0.5, 1.0, 3)
        scale = 0.5 * _L_REF
        phot = jitted_lookup(jnp.float64(scale), jnp.float64(float(alpha_grid[1])))
    elif model_name == "radio_freefree":
        alpha_grid = np.linspace(-0.2, 0.0, 3)
        scale = 0.5 * _L_REF
        phot = jitted_lookup(jnp.float64(scale), jnp.float64(float(alpha_grid[1])))
    else:  # radio_agn_jet
        alpha_grid = np.linspace(0.4, 1.2, 3)
        scale = 0.5 * _L_REF
        phot = jitted_lookup(jnp.float64(scale), jnp.float64(float(alpha_grid[1])))

    assert phot.shape == (len(waves),), f"Expected shape ({len(waves)},), got {phot.shape}"
    chex.assert_tree_all_finite(np.asarray(phot))
    # Radio photometry should be positive (L_nu > 0)
    assert_non_negative(np.asarray(phot), name="output", msg="Lookup produced negative photometry")


def test_radio_synchrotron_scale_invariance(filter_set):
    """Verify linear scaling: L_ir → photometry is linear."""
    from tengri.components.radio import radio_precompute as adapter

    # Use ONLY radio filters (not FIR which might hit suppression boundary)
    waves, _ = filter_set
    radio_waves = [waves[1], waves[2]]  # mm and cm only
    radio_trans = [
        np.exp(-0.5 * ((w - w.mean()) / ((w[1] - w[0]) * 10)) ** 2) for w in radio_waves
    ]

    result = adapter.precompute(
        radio_waves, radio_trans, redshift=0.5, parameters=None, model="radio_synchrotron"
    )
    lookup = adapter.build_lookup(result, model="radio_synchrotron")

    _L_REF = 1.0e44
    alpha_test = 0.8

    # Test two scales; output should scale linearly
    phot_1x = lookup(jnp.float64(1.0 * _L_REF), jnp.float64(alpha_test))
    phot_2x = lookup(jnp.float64(2.0 * _L_REF), jnp.float64(alpha_test))
    phot_05x = lookup(jnp.float64(0.5 * _L_REF), jnp.float64(alpha_test))

    # Scaling should be exact (linear in L_ir)
    # Only check nonzero filters
    phot_1x_arr = jnp.asarray(phot_1x)
    phot_2x_arr = jnp.asarray(phot_2x)
    phot_05x_arr = jnp.asarray(phot_05x)

    nonzero_mask = phot_1x_arr > 1e-50
    if np.any(nonzero_mask):
        ratio_2_1 = phot_2x_arr[nonzero_mask] / phot_1x_arr[nonzero_mask]
        ratio_05_1 = phot_05x_arr[nonzero_mask] / phot_1x_arr[nonzero_mask]
        np.testing.assert_allclose(ratio_2_1, 2.0, rtol=1e-9)
        np.testing.assert_allclose(ratio_05_1, 0.5, rtol=1e-9)
