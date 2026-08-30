# SPDX-License-Identifier: BSD-3-Clause
"""The warmup-divergence fail-fast (#2088), in isolation.

A 4-chain NUTS fit whose every transition diverged ran 227 s of warmup and
237 s of sampling before ``Posterior`` could say "dead fit". The warmup's own
divergence record already carried the verdict; blackjax was told to discard
it. These tests pin the window arithmetic and the refusal threshold; the
backend-level test file exercises the seam in ``run_nuts``.

Mutation checks:
1. ``test_window_is_a_tenth_of_warmup_with_a_floor``: change
   ``DEAD_WARMUP_WINDOW_FRAC`` or the floor.
2. ``test_fraction_is_measured_over_the_final_window_only``: average over
   every step instead of the final window.
3. ``test_refusal_threshold_is_inclusive_at_the_constant``: ``<`` -> ``<=``
   in the early return.
4. ``test_no_record_means_no_refusal``: drop the ``None`` guard.
5. ``test_an_empty_record_measures_nothing``: drop the ``flags.size == 0``
   early return (an empty mean is NaN, not a fraction).
"""

import numpy as np
import pytest

from tengri.config.exceptions import DeadFitError, InferenceError
from tengri.inference.backends.mcmc._shared import (
    DEAD_WARMUP_DIVERGENCE_FRAC,
    DEAD_WARMUP_MIN_WINDOW,
    DEAD_WARMUP_WINDOW_FRAC,
    final_window_divergence_frac,
    refuse_dead_warmup,
)


def _flags(n_warmup, divergent_tail):
    flags = np.zeros(n_warmup, dtype=bool)
    if divergent_tail:
        flags[-divergent_tail:] = True
    return flags


def test_constants_are_the_documented_values():
    assert DEAD_WARMUP_DIVERGENCE_FRAC == 0.9
    assert DEAD_WARMUP_WINDOW_FRAC == 0.1
    assert DEAD_WARMUP_MIN_WINDOW == 10


def test_window_is_a_tenth_of_warmup_with_a_floor():
    # 600 warmup steps -> final 60; only those 60 count.
    assert final_window_divergence_frac(_flags(600, 60), 600) == pytest.approx(1.0)
    assert final_window_divergence_frac(_flags(600, 30), 600) == pytest.approx(0.5)
    # 50 warmup steps -> floor of 10, not 5.
    assert final_window_divergence_frac(_flags(50, 5), 50) == pytest.approx(0.5)
    # A record shorter than the window uses everything it has.
    assert final_window_divergence_frac(_flags(4, 4), 4) == pytest.approx(1.0)


def test_fraction_is_measured_over_the_final_window_only():
    # 90 of 100 steps diverged, but the last 10 were clean: healthy by this measure.
    flags = np.ones(100, dtype=bool)
    flags[-10:] = False
    assert final_window_divergence_frac(flags, 100) == pytest.approx(0.0)


def test_no_record_means_no_refusal():
    assert final_window_divergence_frac(None, 100) is None
    refuse_dead_warmup(None, sampler="NUTS", step_size=0.1, n_warmup=100, n_samples=50)


def test_an_empty_record_measures_nothing():
    # n_warmup=0 leaves a (0,) record. The mean of nothing is NaN, which would
    # sail past the >= threshold and then crash the log line's `100.0 * frac`;
    # "not measured" is the honest answer, and the backends omit the key.
    assert final_window_divergence_frac(np.zeros(0, dtype=bool), 0) is None


def test_refusal_threshold_is_inclusive_at_the_constant():
    refuse_dead_warmup(0.89, sampler="NUTS", step_size=0.1, n_warmup=100, n_samples=50)
    with pytest.raises(DeadFitError):
        refuse_dead_warmup(0.9, sampler="NUTS", step_size=0.1, n_warmup=100, n_samples=50)


def test_the_error_carries_its_measurements_and_names_the_window():
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_warmup(1.0, sampler="NUTS", step_size=0.0438, n_warmup=600, n_samples=600)
    err = excinfo.value
    assert isinstance(err, InferenceError) and isinstance(err, RuntimeError)
    assert err.warmup_divergence_frac == pytest.approx(1.0)
    assert err.step_size == pytest.approx(0.0438)
    text = str(err)
    assert "100%" in text and "final 60" in text and "NUTS" in text and "600 draws" in text
