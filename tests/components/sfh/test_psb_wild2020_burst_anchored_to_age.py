# SPDX-License-Identifier: BSD-3-Clause
"""``psb_wild2020``'s burst component is anchored to the age of the universe at
the model's own redshift, not a hardcoded constant and not the family's own
``age`` parameter.

The burst double power law's cosmic-time coordinate is built from
``age_universe_yr`` (``age_at_z(z)``, injected by the orchestrator from the
evaluation redshift), matching Wild et al. 2020 Eq. 5 and BAGPIPES
``star_formation_history.py``'s ``psb_wild2020`` method, both of which
measure the burst from the observation epoch ("now"), not from the old
component's own formation lookback (``age``) and not from a static module
constant.

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
_BURST_MASK = np.asarray(LB) < _COMMON["burstage"]


def _burst_only_shape(age_universe_yr, age=13.5e9):
    """Isolate the burst term: the old component is windowed to
    ``burstage < lookback < age`` and so is exactly zero at
    ``lookback < burstage`` regardless of ``age`` or ``age_universe_yr``,
    leaving only the burst's own contribution in that region."""
    sfr = np.asarray(psb_wild2020(LB, age=age, age_universe_yr=age_universe_yr, **_COMMON))
    burst = sfr[_BURST_MASK]
    return burst / np.max(burst)  # normalize shape, not amplitude


def test_burst_shape_depends_on_age_universe_yr():
    """The burst shape must differ materially between age_universe_yr=0.6 and 13.5 Gyr.

    With ``burstage`` (0.3 Gyr) held fixed and far below either anchor, the
    DPL ratio ``(age_universe_yr - t_lookback) / (age_universe_yr -
    burstage)`` tracks ``age_universe_yr`` directly, most visibly once the
    anchor is no longer large compared to ``burstage`` (e.g. a young
    high-z galaxy).
    """
    shape_young = _burst_only_shape(0.6e9)
    shape_old = _burst_only_shape(13.5e9)
    rel_diff = float(np.max(np.abs(shape_young - shape_old)))
    assert rel_diff > 0.05, (
        f"burst shape changed by only {rel_diff:.3e} between age_universe_yr=0.6 "
        f"and 13.5 Gyr -- still anchored to a constant, not age_universe_yr"
    )


def test_burst_shape_independent_of_family_age():
    """The burst term does not depend on the family's own ``age`` parameter.

    ``age`` anchors only the OLD (declining-exponential) component's own
    formation lookback; the burst's cosmic-time origin is the age of the
    universe at the model's redshift (Wild et al. 2020 Eq. 5), a quantity
    the orchestrator derives independently of the family's free ``age``.

    Isolated at ``fburst=1`` (pure burst, not the masked-region trick the
    other tests in this module use): mixing in any old-component
    contribution -- even just its edge cell straddling the ``burstage``
    window boundary -- would make the overall mass renormalization (a
    single global scale factor over the WHOLE composite) sensitive to the
    old component's own window width, which does change with ``age`` and
    would confound this comparison with an effect unrelated to the burst
    term itself.
    """
    pure_burst = {**_COMMON, "fburst": 1.0}
    sfr_a = psb_wild2020(LB, age=1.0e9, age_universe_yr=10e9, **pure_burst)
    sfr_b = psb_wild2020(LB, age=13.5e9, age_universe_yr=10e9, **pure_burst)
    np.testing.assert_allclose(np.asarray(sfr_a), np.asarray(sfr_b), rtol=1e-8, atol=1e-20)


def _old_buggy_burst_shape(age_universe_yr):
    """The pre-fix burst term: anchored to ``AGEMAX_YR`` (14 Gyr), not
    ``age_universe_yr``, and windowed to ``lookback < burstage``."""
    from tengri.components.stellar.sfh.mean_sfh import AGEMAX_YR, window_weight

    burstage = _COMMON["burstage"]
    lb_np = np.asarray(LB)
    t_cosmic = AGEMAX_YR - lb_np
    tau_burst = AGEMAX_YR - burstage
    log_ratio = np.log(np.maximum(t_cosmic, 1.0) / np.maximum(tau_burst, 1.0))
    sfr = np.exp(-np.logaddexp(_COMMON["alpha"] * log_ratio, -_COMMON["beta"] * log_ratio))
    sfr = sfr * np.asarray(window_weight(LB, -jnp.inf, burstage))
    burst = sfr[_BURST_MASK]
    return burst / np.max(burst)


def test_burst_shape_diverges_from_old_agemax_anchored_formula():
    """At a young age_universe_yr, the fixed shape diverges from the old constant-anchored one."""
    age_universe_yr = 0.6e9
    fixed = _burst_only_shape(age_universe_yr)
    old_buggy = _old_buggy_burst_shape(age_universe_yr)
    rel_diff = float(np.max(np.abs(fixed - old_buggy)))
    assert rel_diff > 0.05, (
        f"age_universe_yr={age_universe_yr / 1e9:.2f} Gyr: fixed burst shape only differs "
        f"from the old AGEMAX_YR-anchored one by {rel_diff:.3e} -- fix did not take effect"
    )
