# SPDX-License-Identifier: BSD-3-Clause
r"""The Fritz torus read returns the tabulated node at the cases the triweight missed (#2808).

``_interpolate_and_normalize`` in ``fritz.py`` reads the 6-D Fritz library with the
node-exact PCHIP interpolant (#2808). The triweight smoother it replaced was off the
raw node by up to 9.4 % at these points, so each case here fails on the triweight
read and passes on the PCHIP read:

    tau = 1, oa = 60 at 4.6 um   triweight 0.963 of the node
    tau = 1, oa = 60 at 100 um   triweight 1.094 of the node
    tau = 6, oa = 60 at 22 um    triweight 1.064 of the node

The other axes sit at the Fritz defaults (r_ratio 60, beta -0.5, gamma 4, psy 0.001),
which are grid nodes. The expected value is independent of the code under test: the
stored node is normalized by its own frequency integral over the template grid, the
same convention the lookup applies to its result.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from tengri.utils.physics_constants import C_AA

RTOL = 1e-10

# Fritz defaults for the axes the cases do not vary, in axis order
# (r_ratio, tau, beta, gamma, oa, psy).
_R_RATIO, _BETA, _GAMMA, _PSY = 60.0, -0.5, 4.0, 0.001

# (tau, oa, wavelength [Angstrom]) of the three cases.
CASES = (
    (1.0, 60.0, 4.6e4),
    (1.0, 60.0, 1.0e6),
    (6.0, 60.0, 2.2e5),
)


def _expected_nu_norm(wave: np.ndarray, node: np.ndarray) -> np.ndarray:
    """Fritz: unit frequency integral of L_nu = L_lam lam^2 / c over the template grid."""
    from tengri.components.agn._template_grid import native_bolometric_nu_np

    l_nu = node * wave**2 / C_AA
    return l_nu / native_bolometric_nu_np(l_nu, wave)


def _exact_index(axis: np.ndarray, value: float) -> int:
    """Index of the grid node equal to ``value``; the case is void if there is none."""
    hits = np.flatnonzero(np.isclose(np.asarray(axis), value, rtol=1e-9, atol=0.0))
    if hits.size != 1:
        pytest.fail(f"{value} is not a Fritz grid node on this axis")
    return int(hits[0])


@pytest.fixture(scope="module")
def fritz_grid():
    from tengri.components.agn import fritz

    try:
        return fritz.load_fritz_grid(fritz._find_fritz_grid())
    except OSError as exc:  # grid file absent and not downloadable
        pytest.skip(f"Fritz template grid unavailable: {exc}")


@pytest.mark.parametrize(("tau", "oa", "lam"), CASES)
def test_fritz_read_equals_node_at_triweight_miss(fritz_grid, tau, oa, lam):
    from tengri.components.agn import fritz

    g = fritz_grid
    axes = tuple(np.asarray(a) for a in g.axes)
    values = (_R_RATIO, tau, _BETA, _GAMMA, oa, _PSY)
    idx = tuple(_exact_index(a, v) for a, v in zip(axes, values, strict=True))
    point = tuple(float(a[i]) for a, i in zip(axes, idx, strict=True))

    wave = np.asarray(g.wave_grid)
    node = np.asarray(g.dust)[idx]
    got = np.asarray(
        fritz._interpolate_and_normalize(
            jnp.asarray(g.dust),
            jnp.asarray(wave),
            tuple(jnp.asarray(a) for a in axes),
            jnp.asarray(wave),
            point,
            1.0,
        )
    )
    exp = _expected_nu_norm(wave, node)

    j = int(np.argmin(np.abs(wave - lam)))
    ratio = got[j] / exp[j]
    assert abs(ratio - 1.0) <= RTOL, (
        f"tau={tau}, oa={oa}, lambda={wave[j]:.6g} A: lookup/node = {ratio!r}"
    )
    assert np.max(np.abs(got - exp)) <= RTOL * np.max(np.abs(exp))
