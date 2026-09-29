# SPDX-License-Identifier: BSD-3-Clause
"""``psb_wild2020``'s burst component is anchored to its own ``age``, not a hardcoded constant.

``psb_wild2020`` (registry ``psb``) built the burst's DPL cosmic-time
coordinate from the module constant ``AGEMAX_YR`` (14 Gyr) instead of the
function's own ``age`` argument -- the recent-burst episode was decoupled
from the galaxy's actual age (and hence the source redshift), so two builds
differing only in ``age`` produced burst-window shapes that agreed to the
2.3e-3 level (an edge-cell artifact of the *old* component's window, not the
burst term, which was mathematically independent of ``age``) instead of
genuinely tracking it (#2521, sweep_S3_report.md finding F-new-4).

References
----------
.. [1] Issue #2521.
.. [2] S. M. Wild et al., "Recovering star formation histories with
   dense basis representations," MNRAS 494, 529 (2020).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.conservation

from tengri.components.stellar.sfh.mean_sfh import psb_wild2020

LB = jnp.linspace(1e5, 14.0e9, 20_001)

_COMMON = dict(log_total_mass=10.0, tau=2.0e9, burstage=3e8, alpha=1.5, beta=1.0, fburst=0.3)


def _burst_only_shape(age):
    """Isolate the burst term: only it contributes at lookback < burstage."""
    sfr = np.asarray(psb_wild2020(LB, age=age, **_COMMON))
    mask = np.asarray(LB) < _COMMON["burstage"]
    burst = sfr[mask]
    return burst / np.max(burst)  # normalize shape, not amplitude


def test_burst_shape_depends_on_age():
    """The burst-window shape must differ materially between age=0.6 Gyr and age=13.5 Gyr.

    Before the fix, the burst DPL's cosmic-time coordinate used the hardcoded
    ``AGEMAX_YR`` (14 Gyr) instead of ``age``: with ``burstage`` (0.3 Gyr)
    held fixed and far below any tested age, the DPL ratio
    ``(anchor - t_lookback) / (anchor - burstage)`` stays close to 1 across
    the whole burst window whenever the anchor is large relative to
    ``burstage`` -- which a *constant* 14 Gyr anchor always is, so the shape
    was near budget-invariant to any `age` the family declared. Anchoring to
    `age` itself makes the ratio -- and hence the shape -- track `age`
    directly, most visibly once `age` is no longer large compared to
    `burstage` (e.g. a young high-z galaxy).
    """
    shape_young = _burst_only_shape(0.6e9)
    shape_old = _burst_only_shape(13.5e9)
    rel_diff = float(np.max(np.abs(shape_young - shape_old)))
    assert rel_diff > 0.05, (
        f"burst shape changed by only {rel_diff:.3e} between age=0.6 and age=13.5 Gyr "
        f"-- still anchored to a constant, not `age`"
    )


def _old_buggy_burst_shape(age):
    """The pre-fix burst term: anchored to ``AGEMAX_YR`` (14 Gyr), not ``age``."""
    from tengri.components.stellar.sfh.mean_sfh import AGEMAX_YR, window_weight

    burstage = _COMMON["burstage"]
    lb_np = np.asarray(LB)
    age_universe = AGEMAX_YR
    t_cosmic = age_universe - lb_np
    tau_burst = age_universe - burstage
    log_ratio = np.log(np.maximum(t_cosmic, 1.0) / np.maximum(tau_burst, 1.0))
    sfr = np.exp(-np.logaddexp(_COMMON["alpha"] * log_ratio, -_COMMON["beta"] * log_ratio))
    sfr = sfr * np.asarray(window_weight(LB, -jnp.inf, burstage))
    mask = lb_np < burstage
    burst = sfr[mask]
    return burst / np.max(burst)


def test_burst_shape_diverges_from_old_agemax_anchored_formula():
    """At a young `age` (comparable to burstage), the fixed shape diverges from the old one."""
    age = 0.6e9
    fixed = _burst_only_shape(age)
    old_buggy = _old_buggy_burst_shape(age)
    rel_diff = float(np.max(np.abs(fixed - old_buggy)))
    assert rel_diff > 0.05, (
        f"age={age / 1e9:.2f} Gyr: fixed burst shape only differs from the old "
        f"AGEMAX_YR-anchored one by {rel_diff:.3e} -- fix did not take effect"
    )
