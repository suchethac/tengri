# SPDX-License-Identifier: BSD-3-Clause
"""A secant loop that reaches its iteration cap without converging returns NaN.

The ADAF solves (Mahadevan 1997, Sect. 5.1.1 for T_e and Sect. 5.2 for the inversion of
L_ADAF = Q_ie + delta Q+) are bracketed secants in a :func:`jax.lax.while_loop` with a cap
``_N_MAX``. A lane still open at the cap must fail loudly rather than return an iterate that
is not a root. The cap is lowered here by monkeypatching the module constant, which the loop
reads when it is traced.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import adaf as adaf_mod

pytestmark = [pytest.mark.bounds]


def _residual(u, aux):
    """Root at u = ln 5, a smooth function that needs several secant steps from u = 0."""
    return jnp.exp(u) - 5.0, aux


def _solve_from_zero():
    return adaf_mod._bracketed_secant(
        _residual,
        jnp.asarray(0.0),
        jnp.zeros(()),
        -10.0,
        10.0,
        sign=-1.0,
        slope0=1.0,
    )


def test_converged_secant_returns_the_root():
    u, _ = _solve_from_zero()
    assert float(u) == pytest.approx(np.log(5.0), rel=1e-12)


def test_open_lane_at_the_cap_returns_nan(monkeypatch):
    monkeypatch.setattr(adaf_mod, "_N_MAX", 3)
    u, _ = _solve_from_zero()
    assert np.isnan(float(u))
