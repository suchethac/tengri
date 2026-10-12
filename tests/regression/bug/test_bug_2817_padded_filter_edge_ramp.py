# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #2817: padding must not add area past a filter's table edge.

``pad_filters`` zero-pads each filter row to the longest table. The padded
nodes were moved to ``max(fw) + 1 + i`` with transmission 0, so the union-grid
trapezoid integrated a linear ramp from the last tabulated transmission down to
zero over the 1 A beyond the table. A filter whose transmission is nonzero at
its last node (WISE W1/W4, MIPS 24) therefore disagreed with its per-filter
integral. A padded batch row must equal the unpadded per-filter integral to
float64 roundoff, and must stay finite under differentiation in redshift.

Marker: regression_bug
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


def _filters():
    # Short table, nonzero transmission at its last node (the #2817 case).
    short_w = np.arange(4000.0, 6001.0, 50.0)
    short_t = np.linspace(0.2, 0.6, short_w.size)
    # Longer table, so the short one is zero-padded in the batch.
    long_w = np.arange(3000.0, 9001.0, 10.0)
    long_t = np.exp(-0.5 * ((long_w - 6000.0) / 400.0) ** 2)
    return (short_w, short_t), (long_w, long_t)


def _sed():
    wave_rest = np.geomspace(1000.0, 20000.0, 4000)
    sed = 1.0e30 * (wave_rest / 5000.0) ** -1.3
    return jnp.asarray(sed), jnp.asarray(wave_rest)


def test_padded_batch_row_equals_per_filter_integral_at_table_edge():
    from tengri.observation.photometry import (
        lnu_filter_integral,
        lnu_filter_integral_batch,
        pad_filters,
    )

    (short_w, short_t), (long_w, long_t) = _filters()
    fw_p, ft_p, _ = pad_filters([short_w, long_w], [short_t, long_t])
    sed, wave_rest = _sed()
    z = 0.3

    single = lnu_filter_integral(sed, wave_rest, jnp.asarray(short_w), jnp.asarray(short_t), z)
    batch = lnu_filter_integral_batch(sed, wave_rest, fw_p, ft_p, z)

    np.testing.assert_allclose(np.asarray(batch[0]), np.asarray(single), rtol=1e-12, atol=0.0)


def test_padded_batch_gradient_in_redshift_is_finite():
    from tengri.observation.photometry import lnu_filter_integral_batch, pad_filters

    (short_w, short_t), (long_w, long_t) = _filters()
    fw_p, ft_p, _ = pad_filters([short_w, long_w], [short_t, long_t])
    sed, wave_rest = _sed()

    grad = jax.grad(lambda z: jnp.sum(lnu_filter_integral_batch(sed, wave_rest, fw_p, ft_p, z)))
    assert np.isfinite(float(grad(0.3)))
