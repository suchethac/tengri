# SPDX-License-Identifier: BSD-3-Clause
"""#3541: the PCHIP monotone test is a sign test, not a product test.

The Fritsch-Carlson interior tangent is zero unless the two bracketing secants share a
sign. Testing ``d_left * d_right > 0`` underflows in float32 once the secants are
~1e-23 (their product ~1e-46 is below the float32 subnormal floor), so every interior
slope of a table whose values are scaled to the SED regime came out zero in float32.
The float64 tangents of the same data are finite and non-zero. The fix tests
``sign(d_left) * sign(d_right) > 0``, which is exact in any dtype.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps.met_pchip import pchip_node_slopes
from tengri.utils.grid_interp import _pchip_slopes

pytestmark = pytest.mark.regression_bug

#: Column scale at which float32 products of secants underflow to zero.
_SCALE = 1.0e-23
_LGZ = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5])


def _column(scale=_SCALE):
    """A monotone, positive, non-uniformly-varying column at the SED flux scale."""
    return scale * np.exp(np.array([0.0, 0.3, 0.9, 1.0, 1.8, 2.5]))


def test_float32_slopes_match_float64_on_a_column_scaled_to_1e_minus_23():
    y64 = _column()
    s64 = np.asarray(_pchip_slopes(jnp.asarray(_LGZ, jnp.float64), jnp.asarray(y64, jnp.float64)))
    s32 = np.asarray(_pchip_slopes(jnp.asarray(_LGZ, jnp.float32), jnp.asarray(y64, jnp.float32)))
    assert np.all(s64[1:-1] > 0.0)
    assert np.count_nonzero(s32[1:-1] == 0.0) == 0, (
        f"float32 zeroed {np.count_nonzero(s32[1:-1] == 0.0)} interior slopes that float64 "
        f"keeps: f32={s32}, f64={s64}"
    )
    nz = s64 != 0.0
    rel = np.abs(s32[nz] - s64[nz]) / np.abs(s64[nz])
    assert rel.max() < 1e-5, f"float32 slopes differ from float64 by {rel.max():.3g} relative"


def test_native_builder_keeps_float32_slopes_on_a_scaled_table():
    """The native builder (met_pchip.pchip_node_slopes) has the same sign-test read."""
    table64 = np.stack([_column(), 2.0 * _column(), 0.5 * _column()], axis=1)  # (n_met, 3)
    lgmet = np.linspace(-2.0, 0.0, 6)
    tab64 = np.concatenate([table64, 1.3 * table64, 0.7 * table64], axis=1)  # (6, 9)
    s64 = np.asarray(pchip_node_slopes(jnp.asarray(tab64, jnp.float64), jnp.asarray(lgmet)))
    s32 = np.asarray(
        pchip_node_slopes(
            jnp.asarray(tab64 * 1.0, jnp.float32), jnp.asarray(lgmet, jnp.float32)
        )
    )
    interior64 = s64[1:-1]
    interior32 = s32[1:-1]
    zeroed = (interior32 == 0.0) & (interior64 != 0.0)
    assert not zeroed.any(), f"{int(zeroed.sum())} interior native slopes zeroed in float32"
    rel = np.abs(interior32 - interior64) / np.maximum(np.abs(interior64), 1e-300)
    assert rel[interior64 != 0.0].max() < 1e-5


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_sign_test_gives_zero_interior_slope_at_a_true_extremum(dtype):
    """Unchanged semantics: a local extremum (secants of opposite sign) has zero tangent."""
    x = jnp.asarray(np.array([0.0, 1.0, 2.0, 3.0]), dtype)
    y = jnp.asarray(np.array([1.0, 2.0, 1.5, 3.0]) * 1e-23, dtype)
    s = np.asarray(_pchip_slopes(x, y))
    # Node 1 and node 2 both sit between secants of opposite sign.
    assert s[1] == 0.0 and s[2] == 0.0
    assert np.isfinite(s).all()
