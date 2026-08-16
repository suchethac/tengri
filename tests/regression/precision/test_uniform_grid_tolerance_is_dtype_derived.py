# SPDX-License-Identifier: BSD-3-Clause
"""A uniformity tolerance written for float64 rejects a uniform float32 grid (#1206).

``_check_uniform`` compared adjacent ``linspace`` spacings with a hardcoded
``rtol=1e-9``. That is a float64 tolerance: stored in float32 the same grid's
nodes carry absolute error up to ``eps * max|g|``, so the spacings spread by
``eps * max|g| / dx``. Measured on the z-grid ``linspace(0, 4, 494)``:

    float64  spread 5.5e-14   -> passes 1e-9
    float32  spread 2.9e-05   -> FAILS 1e-9

so a genuinely uniform grid was rejected and ``compute_grid_window`` raised.
That blocked the **default fit path** in pure float32, because ``Fitter``
resolves ``approx="auto"`` to ``WavePrecomp``, whose redshift table reaches
this guard. Not a NaN — an exception, before any gradient exists.

The tolerance is now derived from the grid's dtype. These tests pin the
*derivation* and, critically, that the guard still rejects a non-uniform grid:
a tolerance widened into uselessness would also make them pass.
"""

import numpy as np
import pytest

from tengri.utils.interpolation import _check_uniform, _uniform_rtol

pytestmark = pytest.mark.regression_bug

# The WavePrecomp z-table shape that exposed this.
_N = 494
_LO, _HI = 0.0, 4.0


def _grid(dtype):
    return np.linspace(_LO, _HI, _N, dtype=dtype)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_uniform_linspace_is_accepted_at_every_precision(dtype):
    """The whole point: a uniform grid must not be rejected for being float32."""
    _check_uniform(_grid(dtype))  # must not raise


def test_float64_tolerance_is_exactly_the_historical_value():
    """float64 must be judged by the old 1e-9, not a loosened bound.

    The derived term is ~1e-13 in float64, below the 1e-9 floor, so the floor
    wins and float64 behaviour is unchanged by construction.
    """
    assert _uniform_rtol(_grid(np.float64)) == 1e-9


def test_float32_tolerance_exceeds_the_spread_it_must_admit():
    """Derived, not guessed: the bound must cover the observed spread with margin."""
    g = _grid(np.float32)
    d = np.diff(g.astype(float))
    observed = (d.max() - d.min()) / d[0]
    tol = _uniform_rtol(g)
    assert observed > 1e-9, (
        "float32 no longer spreads beyond the old tolerance — this test can no "
        "longer detect the regression it exists for"
    )
    assert tol > observed, f"tolerance {tol:.3g} does not admit spread {observed:.3g}"


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_genuinely_nonuniform_grid_is_still_rejected(dtype):
    """The guard must still bite. A tolerance widened into uselessness passes
    every test above while protecting nothing."""
    bad = np.asarray(np.geomspace(1.0, 5.0, 200), dtype=dtype)
    with pytest.raises(ValueError, match="uniform ascending grid"):
        _check_uniform(bad)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_a_descending_grid_is_still_rejected(dtype):
    bad = np.asarray(np.linspace(4.0, 0.0, 64), dtype=dtype)
    with pytest.raises(ValueError, match="uniform ascending grid"):
        _check_uniform(bad)


def test_tolerance_scales_with_dynamic_range_not_with_node_count():
    """``eps * max|g| / dx`` — a grid further from the origin needs a looser
    relative bound at the same spacing, which a fixed literal cannot express."""
    near = np.linspace(0.0, 1.0, 128, dtype=np.float32)
    far = np.linspace(1000.0, 1001.0, 128, dtype=np.float32)  # same dx, 1000x scale
    assert _uniform_rtol(far) > _uniform_rtol(near)
