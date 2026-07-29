# SPDX-License-Identifier: BSD-3-Clause
"""The mass-remaining normalization fails loud on a degenerate IMF (#1404).

``compute_mass_remaining_fraction`` divides the surviving mass by ``total_mass``,
the IMF mass integral. It used to clamp that denominator::

    (living_mass + dead_remnant_mass) / jnp.maximum(total_mass, 1e-30)

``total_mass`` is a sum over a fixed, strictly positive log-mass grid, so it can
only vanish if the IMF weights are identically zero — a broken registry entry,
not a reachable physical state. The clamp therefore guarded an impossible
condition, and guarded it the wrong way: with ``total_mass`` zero the numerator
is zero too, so the expression returned ``0 / 1e-30 = 0`` — a surviving fraction
of exactly zero, which reads as the perfectly plausible claim "all stellar mass
has been lost" and propagates silently into ``stellar_mass``.

That is the #1395 failure shape: a NaN guard converting a degenerate input into
a finite, believable number. The fix is the form ``utils/sed_quantities.py``
already uses — propagate NaN so it gets noticed.

The healthy-path assertions matter as much as the degenerate one: the point is
that the change is invisible for every real IMF.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.sps import mass_remaining as mr

pytestmark = pytest.mark.regression_bug

AGES_GYR = jnp.array([0.0, 0.1, 1.0, 5.0, 13.0])


def test_healthy_imf_is_unchanged():
    """Every real IMF must be untouched by the guard."""
    out = np.asarray(mr.compute_mass_remaining_fraction(AGES_GYR))

    assert np.all(np.isfinite(out)), f"non-finite surviving fraction: {out}"
    assert out[0] == pytest.approx(1.0, abs=1e-9), "no stars have died at age 0"
    assert np.all((out > 0.0) & (out <= 1.0 + 1e-9)), f"outside (0, 1]: {out}"
    assert np.all(np.diff(out) <= 1e-9), f"surviving fraction must not increase: {out}"


@pytest.mark.parametrize("imf", ["chabrier", "kroupa", "salpeter"])
def test_every_registered_imf_stays_finite(imf):
    """The guard must not fire for any IMF the registry actually offers.

    A guard that trips on real input would be worse than the clamp it replaced,
    so this is the assertion that protects users rather than the bug.
    """
    out = np.asarray(mr.compute_mass_remaining_fraction(AGES_GYR, imf=imf))
    assert np.all(np.isfinite(out)), f"{imf} produced non-finite output: {out}"
    assert np.all(out > 0.0)


def test_degenerate_imf_yields_nan_not_a_plausible_zero(monkeypatch):
    """A zero-weight IMF must produce NaN, not a surviving fraction of 0.

    Before the fix this returned exactly ``0.0`` — indistinguishable from a
    genuine physical result and silently wrong. Asserting ``isnan`` rather than
    ``not isclose(0)`` pins the *mechanism*: the value has to be unusable, not
    merely different.
    """
    monkeypatch.setitem(mr._IMF_REGISTRY, "_degenerate", lambda log_m: jnp.zeros_like(log_m))

    out = np.asarray(mr.compute_mass_remaining_fraction(AGES_GYR, imf="_degenerate"))

    assert np.all(np.isnan(out)), (
        f"degenerate IMF returned {out} instead of NaN — a zero surviving "
        "fraction reads as 'all mass lost' and propagates silently (#1404)"
    )
    assert not np.any(out == 0.0), "the old clamped path returned exactly 0.0"


def test_guard_threshold_does_not_bite_a_small_but_real_imf():
    """The ``> 1e-20`` test must key on a broken IMF, not merely a faint one.

    ``total_mass`` scales with the IMF normalization, so an IMF scaled down by a
    large constant is still perfectly valid. If the threshold were tested against
    an absolute luminosity-like scale it would reject these.
    """
    tiny = {"_tiny": lambda log_m: 1e-12 * jnp.ones_like(log_m)}
    with pytest.MonkeyPatch.context() as mp:
        mp.setitem(mr._IMF_REGISTRY, "_tiny", tiny["_tiny"])
        out = np.asarray(mr.compute_mass_remaining_fraction(AGES_GYR, imf="_tiny"))

    assert np.all(np.isfinite(out)), f"a faint-but-valid IMF was rejected: {out}"
    assert out[0] == pytest.approx(1.0, abs=1e-9)
