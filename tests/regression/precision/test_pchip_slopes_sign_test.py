# SPDX-License-Identifier: BSD-3-Clause
r"""The PCHIP slope limiter compares signs, so scaling monotone data keeps its slopes.

The limiter decides whether two neighbouring secants share a sign. A product of the two
secants underflows to zero in float32 once both are below ~1e-19 (1e-25 squared), so
same-signed secants were read as a sign change and the interior slope was set to 0.
Scaling monotone increasing data by a constant must scale its slopes by the same constant.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.utils.grid_interp import _pchip_slopes

pytestmark = pytest.mark.regression_bug

N_NODES = 6


def _monotone_values(dtype):
    x = jnp.asarray(np.arange(N_NODES, dtype=np.float64), dtype=dtype)
    y = jnp.exp(x)
    return x, y


@pytest.mark.parametrize(
    ("dtype_x64", "scale"),
    [
        (False, 1e-25),
        (False, 1e25),
        (True, 1e-200),
        (True, 1e200),
    ],
)
def test_pchip_slopes_survive_scaling_of_monotone_data(dtype_x64, scale):
    with jax.enable_x64(dtype_x64):
        x, y = _monotone_values(jnp.float64 if dtype_x64 else jnp.float32)
        reference = np.asarray(_pchip_slopes(x, y))
        scaled = np.asarray(_pchip_slopes(x, y * scale))

    assert np.all(np.isfinite(scaled))
    assert np.all(scaled[1:-1] > 0.0)
    np.testing.assert_allclose(scaled, reference * scale, rtol=1e-5, atol=0.0)
