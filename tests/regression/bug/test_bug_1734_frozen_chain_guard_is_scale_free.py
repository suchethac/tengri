# SPDX-License-Identifier: BSD-3-Clause
"""The frozen-chain guard must not depend on the parameters' magnitude (#1734).

#1438 established that a frozen chain must not report as converged. The guard
is correct in intent; the staticness predicate underneath it was not.

Three diagnostics decided "this parameter never moved" with
``np.var(a) < 1e-30`` — an **absolute** tolerance on a quantity carrying the
square of the parameter's units. ``np.var`` of N identical floats is rounding
noise of order ``(value * eps)**2``, so the threshold's sensitivity tracked the
parameter's magnitude. A frozen parameter large enough to clear ``1e-30``
survived the filter, split R-hat scored it ~1.0 on constant data, and the
resulting non-empty dict bypassed the guard in ``Posterior.rhat`` — which fires
only when *every* parameter is dropped.

Live consequence, on a published docs page: ``06_fitting_spectroscopy`` reported
``max R-hat 0.998`` with 590/600 divergent transitions, ``p16 == p50 == p84``
for every parameter, and 0/6 coverage.

The existing guard test (``test_bug_1438_frozen_chain_is_not_converged.py``)
freezes its chains at ``0.3``, whose noise floor sits below ``1e-30``. It could
not have caught this. Every test here is therefore parametrized over magnitude —
that is the defect, and a single-value fixture is what let it through.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.analysis.diagnostics.autocorrelation import (
    effective_sample_size,
    rank_normalized_rhat,
    rhat,
)
from tengri.inference.posterior import Posterior

pytestmark = pytest.mark.regression_bug

#: Magnitudes spanning the ones tengri actually fits: an optical depth, a
#: timescale in Gyr, a log-mass, and a linear mass. ``10.634`` is a log stellar
#: mass and is the value measured to produce ``np.var == 3.155e-30`` — above the
#: old floor, so it is the one that defeated the guard.
_MAGNITUDES = [0.3, 0.693, 1.488, 4.130, 10.634, 1.0e6]

_N_DRAW = 600


def _frozen_samples(value: float, names=("a", "b", "c")) -> dict[str, np.ndarray]:
    return {n: np.full(_N_DRAW, value) for n in names}


def _frozen_posterior(value: float) -> Posterior:
    return Posterior(
        samples=_frozen_samples(value),
        params={n: value for n in ("a", "b", "c")},
        method="mcmc_hmc",
        wall_time_s=1.0,
        diagnostics={"n_divergent": 590, "n_samples": _N_DRAW},
    )


# ── the headline: a frozen chain never reports converged ──────────


@pytest.mark.parametrize("value", _MAGNITUDES)
def test_a_frozen_chain_raises_whatever_its_magnitude(value):
    """LOAD-BEARING. Neuter: restore ``np.var(a) < 1e-30`` in ``rhat``.

    At ``value=10.634`` this failed before the fix, returning
    ``{'a': 0.998}`` — a dead fit reported as converged.
    """
    with pytest.raises(ValueError, match="did not move"):
        _frozen_posterior(value).rhat()


@pytest.mark.parametrize("value", _MAGNITUDES)
def test_rhat_drops_every_frozen_parameter(value):
    """The predicate itself: no frozen parameter may survive into the result."""
    result = rhat(_frozen_samples(value))
    assert result == {}, (
        f"frozen at {value}: {sorted(result)} survived the staticness filter "
        f"(np.var = {float(np.var(np.full(_N_DRAW, value))):.3e})"
    )


@pytest.mark.parametrize("value", _MAGNITUDES)
def test_rank_normalized_rhat_is_nan_when_nothing_moved(value):
    """The third site (:func:`rank_normalized_rhat`) carried the same floor."""
    assert np.isnan(rank_normalized_rhat(np.full(_N_DRAW, value)))


@pytest.mark.parametrize("value", _MAGNITUDES)
def test_ess_skips_frozen_parameters(value):
    """The first site, on the autocorrelation path, carried it too."""
    assert effective_sample_size(_frozen_samples(value)) == {}


# ── the specific value that defeated the guard ────────────────────


def test_the_survivor_had_variance_above_the_old_floor():
    """Pins the mechanism, so a future reader sees why an absolute floor fails.

    If this stops holding, the regression above stops being a regression: the
    value would no longer exercise the path that broke.
    """
    variance = float(np.var(np.full(_N_DRAW, 10.634)))
    assert variance > 1e-30, (
        f"10.634 no longer produces variance above the old 1e-30 floor "
        f"({variance:.3e}); pick a value that does or this file tests nothing"
    )
    assert np.unique(np.full(_N_DRAW, 10.634)).size == 1


# ── chains that did move must be unaffected ───────────────────────


@pytest.mark.parametrize("value", _MAGNITUDES)
def test_a_chain_that_moved_is_still_reported(value):
    """The guard must reject only frozen chains, at every magnitude."""
    rng = np.random.default_rng(0)
    samples = {"a": value + rng.normal(0.0, max(abs(value), 1.0) * 1e-3, _N_DRAW)}

    result = rhat(samples)
    assert "a" in result, f"a moving chain at {value} was dropped as static"
    assert np.isfinite(result["a"])


def test_tiny_but_real_motion_is_kept():
    """Strictly more permissive in the right direction than the old floor.

    Motion of order 1e-16 has variance ~1e-32, below the old ``1e-30`` — it was
    dropped as static although the parameter genuinely moved. An exact predicate
    keeps it.
    """
    rng = np.random.default_rng(0)
    samples = {"a": 0.3 + rng.normal(0.0, 1e-16, _N_DRAW)}
    assert float(np.var(samples["a"])) < 1e-30
    assert "a" in rhat(samples)


def test_a_partially_frozen_posterior_still_reports_the_live_parameters():
    """Dropping static parameters and reporting the rest is documented (#1438)."""
    samples = {"pinned": np.full(_N_DRAW, 10.634)}
    rng = np.random.default_rng(0)
    samples["live"] = rng.normal(size=_N_DRAW)

    result = rhat(samples)
    assert set(result) == {"live"}
