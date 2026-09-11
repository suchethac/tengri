# SPDX-License-Identifier: BSD-3-Clause
"""#2264: vmap_chunked fails to catch Tracer*ConversionError siblings.

On JAX 0.11, TracerArrayConversionError and TracerIntegerConversionError are
SIBLINGS of ConcretizationTypeError (not subclasses), while
TracerBoolConversionError IS a subclass. The probe in vmap_chunked@batching.py
line 161 originally caught only ConcretizationTypeError, so functions raising
the two sibling exceptions would bubble up instead of triggering the documented
eager fallback. The Array sibling is the primary manifestation: np.asarray()
on a traced value raises it.

See: https://github.com/suchethac/tengri/issues/2264
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import vmap_chunked

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def sample_dict():
    """A simple parameter dict for testing."""
    return {
        "value": jnp.arange(10.0),
        "flag": jnp.ones(10, dtype=bool),
    }


def test_asarray_function_falls_back_not_raises(sample_dict):
    """np.asarray on a tracer raises TracerArrayConversionError, which must fall back."""

    def fn(p):
        # When vmap_chunked slices the batch, it passes scalar values.
        # np.asarray() on a traced scalar raises TracerArrayConversionError under jit.
        val = p["value"]  # scalar when sliced
        arr = np.asarray(val)
        return arr * 2.0

    with pytest.warns(UserWarning, match="cannot be traced"):
        result = vmap_chunked(fn, chunk_size=4)(sample_dict)

    # Should produce correct results via eager fallback
    expected = np.arange(10.0) * 2.0
    np.testing.assert_array_almost_equal(result, expected)


def test_bool_branch_function_falls_back(sample_dict):
    """TracerBoolConversionError (subclass) is caught; verify bool-branch functions fall back."""

    def fn(p):
        # When vmap_chunked slices the batch, scalars are passed.
        # bool() on a traced comparison raises TracerBoolConversionError under jit.
        val = p["value"]  # scalar when sliced
        if bool(val > 5.0):
            return val * 2.0
        else:
            return val * 1.0

    with pytest.warns(UserWarning, match="cannot be traced"):
        result = vmap_chunked(fn, chunk_size=4)(sample_dict)

    # Should produce correct results via eager fallback
    expected = np.where(np.arange(10.0) > 5.0, np.arange(10.0) * 2.0, np.arange(10.0))
    np.testing.assert_array_almost_equal(result, expected)
