# SPDX-License-Identifier: BSD-3-Clause
"""The parity sweep's two arms must optimize the same MAP objective (#2300).

Since #2281 ``Fitter(profile_mass="auto")`` marginalizes the stellar mass analytically
under float64 and declines under float32 (``mass_profile._check_guards`` refuses when
``jax_enable_x64`` is off). Left at ``"auto"``, the sweep's float64 reference records a
marginal-likelihood optimum on every seam the guard admits, while the float32 arm
records the joint one: the ``map_loss`` column read 0.90 relative on four seams and
the ``param`` column measured "profiled vs joint" (2.3e-4), not float32. The sweep
measures arithmetic, so the objective is pinned explicitly on both arms.
"""

import sys
from pathlib import Path
from typing import ClassVar

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.unit

_SCRIPTS = Path(__file__).resolve().parents[2] / "bench" / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from benchmark_float32_mps_parity import _map_loss

_OPTIMUM = {"dust_tau_diff": jnp.asarray(0.3), "sfh_delayed_log_total_mass": jnp.asarray(10.0)}


class _Fit:
    """Stands in for ``ForwardModel``: records every ``fit`` kwarg, converges at once."""

    calls: ClassVar[list[dict]] = []

    @classmethod
    def build(cls, *, sed, observation):
        return cls()

    def fit(self, flux, noise, **kwargs):
        from tengri import Posterior

        type(self).calls.append(kwargs)
        return Posterior(
            samples=None,
            params=dict(_OPTIMUM),
            method="MAP (stub)",
            wall_time_s=0.0,
            diagnostics={"n_steps": 2, "converged": True},
            loss_history=jnp.asarray([1.0, 0.5]),
            _model=None,
        )


def test_map_stage_pins_profile_mass_off_on_both_arms():
    _Fit.calls.clear()
    truth = {"dust_tau_diff": 0.3, "sfh_delayed_log_total_mass": 10.0}
    free = ["dust_tau_diff", "sfh_delayed_log_total_mass"]
    for dtype in (jnp.float32, jnp.float64):
        loss, vector, converged, n_iters = _map_loss(
            _Fit, None, None, [1.0, 2.0], [0.1, 0.1], truth, free, dtype, n_steps=10
        )
        assert (loss, converged, n_iters) == (0.5, True, 2)
        assert vector == [0.3, 10.0]
    assert len(_Fit.calls) == 2
    for call in _Fit.calls:
        assert call.get("profile_mass") is False, (
            "the MAP stage left profile_mass at Fitter's default: float64 then marginalizes "
            "the mass and float32 does not, and the two arms optimize different objectives"
        )
