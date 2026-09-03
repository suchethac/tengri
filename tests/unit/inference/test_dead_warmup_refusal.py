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
5. ``test_an_empty_record_measures_nothing``: drop the
   ``flags.size < DEAD_WARMUP_MIN_WINDOW`` early return (the mean of an empty
   window is NaN, not a fraction).
6. ``test_window_is_a_tenth_of_warmup_with_a_floor``: change ``<`` to ``<=``
   in that same guard -- the ``(10, 10)`` boundary case then reads ``None``
   instead of ``1.0``.
7. ``test_the_error_survives_pickling``: drop the keyword defaults on
   ``DeadFitError.__init__`` -- the default reduce protocol re-invokes the
   constructor with ``args`` only and raises ``TypeError``.
"""

import copy
import pickle

import numpy as np
import pytest

from tengri import DeadFitError
from tengri.config.exceptions import InferenceError
from tengri.inference.backends.mcmc._shared import (
    DEAD_WARMUP_DIVERGENCE_FRAC,
    DEAD_WARMUP_MIN_WINDOW,
    DEAD_WARMUP_STEP_SIZE_FLOOR,
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
    # A record too short to fill the minimum window carries no verdict (R15).
    # BlackJAX opens dual averaging at mu = log(10 * eps0), so the first
    # proposals are made at eps ~ 2 whatever the posterior and take five or six
    # rejections to collapse: a sub-window record is that opening burst and
    # nothing else, on a healthy posterior as much as on a dead one.
    assert final_window_divergence_frac(_flags(4, 4), 4) is None
    assert final_window_divergence_frac(_flags(9, 9), 9) is None
    # Exactly the minimum window is long enough to be judged.
    assert final_window_divergence_frac(_flags(10, 10), 10) == pytest.approx(1.0)


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


def test_the_error_survives_pickling():
    """``DeadFitError`` crosses a process boundary as itself, not as a ``TypeError``.

    ``BaseException.__reduce__`` re-invokes the constructor with ``self.args``
    alone, so keyword-only arguments without defaults make every pickle and
    every ``copy`` raise ``TypeError`` -- which is what a multiprocessing
    driver would surface instead of the refusal message.
    """
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_warmup(1.0, sampler="NUTS", step_size=0.0438, n_warmup=600, n_samples=600)
    err = excinfo.value
    for revived in (pickle.loads(pickle.dumps(err)), copy.copy(err)):
        assert isinstance(revived, DeadFitError)
        assert str(revived) == str(err)
        assert revived.warmup_divergence_frac == pytest.approx(1.0)
        assert revived.step_size == pytest.approx(0.0438)


def test_step_size_floor_triggers_below_floor():
    """Step size below the floor raises, regardless of divergence fraction (#2128)."""
    # Test with low divergence fraction (below threshold) but collapsed step size
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_warmup(0.1, sampler="HMC", step_size=1e-78, n_warmup=600, n_samples=600)
    err = excinfo.value
    assert err.step_size == pytest.approx(1e-78)
    text = str(err)
    # Require step-size value AND "#2128" AND "step size", no OR.
    assert "1e-78" in text
    assert "step size" in text.lower()
    assert "#2128" in text


def test_step_size_floor_does_not_trigger_above_floor():
    """Step size above the floor does not trigger the floor check."""
    # Step size above floor but still low divergence: should pass quietly
    refuse_dead_warmup(
        0.1, sampler="HMC", step_size=DEAD_WARMUP_STEP_SIZE_FLOOR * 10, n_warmup=600, n_samples=600
    )
    # Healthy step size: definitely should pass
    refuse_dead_warmup(0.1, sampler="HMC", step_size=0.05, n_warmup=600, n_samples=600)


def test_step_size_floor_edge_exactly_at_floor():
    """Comparison direction: exactly at floor should NOT trigger (#2128 guard design)."""
    # The guard uses `<` so exactly at the floor boundary should not raise.
    # If floor is 1e-20, 1e-20 exactly should pass; only 1e-20 * 0.9 should fail.
    refuse_dead_warmup(
        0.1, sampler="HMC", step_size=DEAD_WARMUP_STEP_SIZE_FLOOR, n_warmup=600, n_samples=600
    )


def test_step_size_none_does_not_trigger_floor():
    """Absence check: None step_size (no measurement) does not trigger floor check."""
    # None step_size should return quietly (no verdict)
    refuse_dead_warmup(None, sampler="HMC", step_size=None, n_warmup=600, n_samples=600)


def test_step_size_nan_does_not_trigger_floor():
    """Absence check: NaN step_size (no measurement) does not trigger floor check."""
    # NaN step_size should not raise (isfinite guard catches it)
    refuse_dead_warmup(0.1, sampler="HMC", step_size=float("nan"), n_warmup=600, n_samples=600)


def test_step_size_zero_triggers_floor():
    """Zero step size triggers the floor check (most frozen chain possible)."""
    # 0.0 step_size is the ultimate collapsed step and should raise DeadFitError.
    # This is the most extreme case of step-size collapse.
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_warmup(0.1, sampler="HMC", step_size=0.0, n_warmup=600, n_samples=600)
    err = excinfo.value
    assert "#2128" in str(err)


def test_step_size_floor_message_names_the_threshold():
    """Error message names the DEAD_WARMUP_STEP_SIZE_FLOOR constant (#2128)."""
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_warmup(
            0.1,
            sampler="HMC",
            step_size=DEAD_WARMUP_STEP_SIZE_FLOOR * 0.1,
            n_warmup=600,
            n_samples=600,
        )
    err = excinfo.value
    text = str(err)
    # Message should reference the floor value in scientific notation
    assert "1e-20" in text or f"{DEAD_WARMUP_STEP_SIZE_FLOOR:.3g}" in text


def test_step_size_floor_with_unmeasured_divergence():
    """Frac=None with step_size below floor raises DeadFitError, not TypeError."""
    # The step-size floor check runs before the frac None-return, so it must
    # handle None gracefully and still fire. Verifies frac_text precomputation.
    with pytest.raises(DeadFitError) as excinfo:
        refuse_dead_warmup(None, sampler="HMC", step_size=1e-50, n_warmup=600, n_samples=600)
    err = excinfo.value
    text = str(err)
    # Message should mark the divergence fraction as unmeasured.
    assert "unmeasured" in text
    assert "#2128" in text
