# SPDX-License-Identifier: BSD-3-Clause
"""Static node windows of the flagged pixels and the slice dust transmission (#3413).

A slice of the model grid must carry exactly the dust transmission the full grid carries at
the same nodes: the two-component screen is elementwise in wavelength, so the slice value is
bit-identical to the full-grid value at those nodes.
"""

import jax.numpy as jnp
import numpy as np

from tengri.components.dust._apply import two_component_dust
from tengri.components.stellar.sps.flagged_pixels import (
    flagged_pixel_mask,
    node_window,
    padded_node_slices,
)


def test_flagged_mask_marks_pixels_whose_padded_range_holds_a_breakpoint():
    edges = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    flags = flagged_pixel_mask(edges, np.array([2500.0]), pad_aa=0.0)
    np.testing.assert_array_equal(flags, [False, True, False])
    # A breakpoint on a shared edge flags both neighbours once padded.
    flags = flagged_pixel_mask(edges, np.array([2000.0]), pad_aa=10.0)
    np.testing.assert_array_equal(flags, [True, True, False])


def test_no_breakpoints_flags_nothing():
    edges = np.array([1.0, 2.0, 3.0])
    assert not flagged_pixel_mask(edges, np.array([]), pad_aa=5.0).any()


def test_padded_windows_address_the_same_nodes():
    wave = np.linspace(1.0, 100.0, 100)
    w0 = node_window(wave, 10.0, 20.0)
    w1 = node_window(wave, 50.0, 53.0)
    idx, mask = padded_node_slices([w0, w1])
    assert idx.shape == mask.shape == (2, max(w0.size, w1.size))
    np.testing.assert_array_equal(idx[0, : w0.size], w0)
    np.testing.assert_array_equal(idx[1, : w1.size], w1)
    assert mask[0].sum() == w0.size and mask[1].sum() == w1.size


def test_slice_transmission_is_bit_identical_to_full_grid():
    wave = np.geomspace(2e4, 2e5, 60)
    ages = jnp.array([1e6, 1e8, 1e10])
    kw = dict(law_bc="calzetti", law_diff="calzetti")
    full = np.asarray(two_component_dust(jnp.asarray(wave), ages, 0.5, 0.3, **kw))
    idx = node_window(wave, 1.9e4, 3.1e4)
    slice_t = np.asarray(two_component_dust(jnp.asarray(wave[idx]), ages, 0.5, 0.3, **kw))
    np.testing.assert_array_equal(slice_t, full[:, idx])
