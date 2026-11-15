# SPDX-License-Identifier: BSD-3-Clause
r"""The multicolor disc has the same Hessian in float32 as in float64 (#3505).

The inner annulus sits at the ISCO, where the zero-torque factor
:math:`(1 - \sqrt{r_{\rm in}/r})^{1/4}` vanishes. The radial grid used to be built as
``10**linspace(log r_in, log r_out) / r_in``, which put the first node at ``1 + 2.6e-6`` in
float32 (``1 - 4e-16`` in float64): the factor read 0.034 instead of ~0, and the second
derivative with respect to ``log M_BH`` was -3366 against 389. The grid is now built in the
ratio ``r / r_in``, whose first node is exactly 1, and the floor under the factor is the
smallest one whose own second derivative is finite in float32.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import multicolor_disc

pytestmark = pytest.mark.regression_bug

NAMES = ("agn_log_lbol", "agn_log_mbh", "agn_a_spin", "agn_cos_inc")
POINTS = [
    (10.9, 8.0, 0.3, 0.8),
    (12.0, 9.5, 0.9, 0.5),
    (9.5, 6.5, 0.0, 0.99),
    (11.0, 8.0, 0.998, 0.1),
]
HESSIAN_TOL = 1e-4


def _derivatives(point, x64, euv_tail):
    with jax.enable_x64(x64):
        dtype = jnp.float64 if x64 else jnp.float32
        wave = jnp.asarray(np.geomspace(100.0, 1e5, 120), dtype)

        def energy(v):
            kwargs = dict(zip(NAMES, v, strict=True))
            return jnp.sum(
                multicolor_disc(wave, agn_lum_ratio=1.0, euv_tail=euv_tail, **kwargs) * 1e-28
            )

        v = jnp.asarray(point, dtype)
        return (
            np.asarray(jax.grad(energy)(v), dtype=np.float64),
            np.asarray(jax.hessian(energy)(v), dtype=np.float64),
        )


@pytest.mark.parametrize("euv_tail", ["powerlaw", "wien"])
@pytest.mark.parametrize("point", POINTS)
def test_float32_gradient_and_hessian_match_float64(point, euv_tail):
    g64, h64 = _derivatives(point, True, euv_tail)
    g32, h32 = _derivatives(point, False, euv_tail)
    assert np.all(np.isfinite(g32)) and np.all(np.isfinite(h32))
    assert np.max(np.abs(g32 - g64)) <= 1e-4 * np.max(np.abs(g64))
    assert np.max(np.abs(h32 - h64)) <= HESSIAN_TOL * np.max(np.abs(h64))
