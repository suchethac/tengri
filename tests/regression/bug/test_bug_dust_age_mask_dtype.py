# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for the age-split dtype preservation bug.

Bug: the birth-cloud age mask hardcoded ``jnp.float64``, defeating mixed
precision. The split is now the per-node younger fraction; it must follow the
dtype of the parcel masses it is built from.
"""

import jax.numpy as jnp
import pytest

from tengri.components.stellar.age_boundary import (
    age_boundary_younger_fraction_cic,
    survival_cell_mean,
)

pytestmark = pytest.mark.regression_bug


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("width", [0.0, 0.3])
class TestAgeSplitDtype:
    """Bug: hardcoded jnp.float64 in the age mask."""

    def test_cell_mean_follows_the_input_dtype(self, dtype, width):
        lo = jnp.linspace(0.0, 1e10, 100, dtype=dtype)
        hi = lo + jnp.asarray(1e8, dtype=dtype)
        assert survival_cell_mean(lo, hi, 3e8, width).dtype == dtype

    def test_fraction_follows_the_input_dtype(self, dtype, width):
        age = jnp.linspace(0.0, 1e10, 50, dtype=dtype)
        contrib = jnp.ones(50, dtype=dtype)
        idx = jnp.minimum(jnp.arange(50) // 5, 8)
        f = jnp.full(50, 0.5, dtype=dtype)
        out = age_boundary_younger_fraction_cic(contrib, idx, f, age, 10, (3e8,), width)
        assert out.dtype == dtype
