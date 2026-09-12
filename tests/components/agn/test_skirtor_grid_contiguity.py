# SPDX-License-Identifier: BSD-3-Clause
"""The SKIRTOR grid arrays handed to JAX are C-contiguous.

``_ascending_axes`` puts descending grid axes in ascending order with
``np.flip`` / ``[::-1]``, which return numpy *views* with a negative stride.
``jnp.array`` of such a view is fine on XLA's CPU client, which copies the host
buffer, and fatal on jax-mps, whose ``BufferFromHostBuffer`` refuses it::

    [MPS ERROR] Negative byte stride (-544) at dim 5 not supported
    JaxRuntimeError: INTERNAL: Failed to create Metal buffer. GPU memory may be exhausted.

Measured on an M4 Pro (jax-mps 0.10.10): every seam of the #1206 float32 parity
sweep that includes the torus failed this way in 1.5 s, with the misleading
"memory exhausted" message, while ``np.ascontiguousarray`` of the same cube
transfers cleanly. The axis flipped is the 10-node inclination axis (dim 5 of
the ``(5, 4, 4, 8, 3, 10, 136)`` cubes) and its 1-D axis array.

This pins the loader's output, not the plugin's behavior: a contiguous host
buffer is what every backend wants from ``device_put`` anyway.
"""

import numpy as np
import pytest

from tengri._data_setup import find_data
from tengri.components.agn.skirtor import _load_grid_arrays

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def raw():
    path = find_data("skirtor_templates_v3.h5", "skirtor_templates_v2.h5")
    if path is None:
        pytest.skip("SKIRTOR template grid not available")
    return _load_grid_arrays(str(path))


def test_every_cube_handed_to_jax_is_c_contiguous(raw):
    # ``norm`` (``spectra/norm`` on the v3 grid, parameter-shaped with no
    # wavelength axis) is flipped with the same inclination axis and handed to
    # JAX as ``SkirtorDiscDustGrid.norm``, so it needs the same contiguity.
    bad = {}
    for key in ("total", "disk", "dust", "norm"):
        if key in raw:
            cube = np.asarray(raw[key])
            if not cube.flags["C_CONTIGUOUS"] or min(cube.strides) < 0:
                bad[key] = cube.strides
    assert not bad, (
        f"SKIRTOR cubes reach jnp.array as non-contiguous views (strides): {bad}. "
        "jax-mps refuses a negative host stride; make the flipped cube contiguous."
    )


def test_every_axis_handed_to_jax_is_c_contiguous(raw):
    bad = {
        i: np.asarray(ax).strides
        for i, ax in enumerate(raw["axes"])
        if not np.asarray(ax).flags["C_CONTIGUOUS"] or min(np.asarray(ax).strides) < 0
    }
    assert not bad, f"SKIRTOR axes reach jnp.array as reversed views (strides): {bad}"


def test_the_axes_really_ascend(raw):
    """Non-vacuity: the flip must still have happened, or this file guards nothing."""
    for i, ax in enumerate(raw["axes"]):
        ax = np.asarray(ax)
        if ax.size > 1:
            assert np.all(np.diff(ax) > 0), f"axis {i} does not ascend: {ax}"
