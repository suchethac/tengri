# SPDX-License-Identifier: BSD-3-Clause
"""The union quadrature integrates a filter over its own support only (#2817).

``_filter_integral_union`` used to interpolate the transmission onto the union
of the SED nodes and the filter nodes with zero fill beyond the table. A filter
with nonzero transmission at its first or last node therefore ramped linearly
from that edge value to zero out to the next SED node. The ramp is as wide as the
local SED spacing, and it moved (jumped) whenever an SED node crossed the edge,
so the band flux of a smooth SED was a sawtooth in redshift (X-ray tophat, WISE
W3/W4 at the free-z band response). The quadrature now clips the SED nodes into
the filter's support, so the integral is the exact band integral and its only
redshift dependence is the smooth L_nu read at the observed nodes.

The reference is the closed form of a band average, so the test does not depend
on the SED grid spacing.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.photometry import lnu_filter_integral

pytestmark = pytest.mark.regression_bug

#: Tophat between 4000 and 5000 A with nonzero transmission at both table edges.
_FILTER_WAVE = jnp.asarray([4000.0, 5000.0])
_FILTER_TRANS = jnp.asarray([1.0, 1.0])


def _band_mean_of_lambda() -> float:
    """Exact Bessel-weighted (1/lambda) band average of L_nu = lambda over [4000, 5000]."""
    return 1000.0 / np.log(5000.0 / 4000.0)


def _sed_on(spacing: float) -> tuple[jnp.ndarray, jnp.ndarray]:
    """L_nu = lambda on a uniform rest grid of the given spacing, spanning past the band."""
    wave = jnp.arange(3000.0, 6000.0 + spacing, spacing)
    return wave, wave


def test_band_average_matches_the_closed_form_on_a_fine_sed_grid():
    """Fine SED grid: the band average equals the closed form to round-off.

    RED before the fix: the ramp past the 5000 A edge biased the average by
    -4.2e-5 at 5 A spacing (-2.1e-3 at 250 A, -4.2e-3 at 500 A); after it, -2.1e-7.
    """
    wave, L = _sed_on(5.0)
    got = float(lnu_filter_integral(L, wave, _FILTER_WAVE, _FILTER_TRANS, 0.0))
    assert got == pytest.approx(_band_mean_of_lambda(), rel=1e-6)


@pytest.mark.parametrize("spacing", [250.0, 500.0])
def test_coarse_sed_grid_adds_only_the_weight_trapezoid(spacing):
    """A coarse SED grid leaves only the smooth 1/lambda weight's trapezoid error.

    Measured after the fix: -5.2e-4 at 250 A, -2.1e-3 at 500 A, with no edge
    contribution (the pre-fix values were -2.1e-3 and -4.2e-3).
    """
    wave, L = _sed_on(spacing)
    got = float(lnu_filter_integral(L, wave, _FILTER_WAVE, _FILTER_TRANS, 0.0))
    assert got == pytest.approx(_band_mean_of_lambda(), rel=5e-3), (spacing, got)


def test_the_flux_is_continuous_in_redshift_for_a_smooth_sed():
    """A dense z scan of a smooth SED through a nonzero-edge tophat is smooth.

    Relative second differences stay tiny: the only remaining structure is the
    trapezoid on the SED's own nodes, which is far below any sawtooth.
    """
    wave = jnp.linspace(900.0, 40000.0, 4000)
    L = (wave / 1e4) ** -1.0
    z = np.linspace(0.995, 1.015, 801)
    y = np.array(
        [float(lnu_filter_integral(L, wave, _FILTER_WAVE, _FILTER_TRANS, float(zi))) for zi in z]
    )
    d2 = np.abs(np.diff(y, 2)) / np.abs(y).max()
    assert d2.max() < 1e-6, (d2.max(), z[int(np.argmax(d2)) + 1])
