# SPDX-License-Identifier: BSD-3-Clause
"""The air <-> vacuum converter pair: formula, round trip, cutoff, JAX behavior."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.air_vacuum import AIR_CUTOFF_AA, air_to_vac, vac_to_air

#: (vacuum, air) reference pairs [Angstrom], formula accuracy ~0.01.
PAIRS = [(6564.61, 6562.80), (5008.24, 5006.84), (6585.27, 6583.45), (4862.68, 4861.33)]


@pytest.mark.parametrize(("vac", "air"), PAIRS)
def test_reference_pairs(vac, air):
    assert abs(float(vac_to_air(vac)) - air) < 0.01
    assert abs(float(air_to_vac(air)) - vac) < 0.01


def test_round_trip_two_thousand_angstrom_to_five_micron():
    x = np.linspace(2000.0, 5.0e4, 100001)
    assert np.max(np.abs(air_to_vac(vac_to_air(x)) - x)) < 1e-6
    assert np.max(np.abs(vac_to_air(air_to_vac(x)) - x)) < 1e-6


def test_identity_below_cutoff():
    short = np.array([100.0, 912.0, 1215.67, 1999.0, AIR_CUTOFF_AA - 0.01])
    np.testing.assert_array_equal(air_to_vac(short), short)
    np.testing.assert_array_equal(vac_to_air(short[:4]), short[:4])


def test_vacuum_is_longer_than_air_above_cutoff():
    x = np.array([2500.0, 5000.0, 20000.0])
    assert np.all(air_to_vac(x) > x)
    assert np.all(vac_to_air(x) < x)


@pytest.mark.parametrize("fn", [air_to_vac, vac_to_air])
@pytest.mark.parametrize("x", [1500.0, 6563.0])
def test_gradient_finite_and_nonzero(fn, x):
    g = float(jax.grad(fn)(x))
    assert np.isfinite(g)
    assert g != 0.0


def test_jit_and_numpy_agree():
    x = np.array([1500.0, 4862.68, 6564.61])
    np.testing.assert_allclose(
        np.asarray(jax.jit(vac_to_air)(jnp.asarray(x))), vac_to_air(x), rtol=1e-12
    )
    assert isinstance(vac_to_air(x), np.ndarray)
    assert isinstance(vac_to_air(jnp.asarray(x)), jax.Array)
    assert vac_to_air(2500.0).shape == ()
