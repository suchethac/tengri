# SPDX-License-Identifier: BSD-3-Clause
"""#1397: a non-finite MAP point must not be handed downstream.

The chain that broke ``notebooks/01_why_jax.py``::

    MAP init done (loss=nan)      <- nobody objected here
      -> Hessian at a NaN point is NaN
      -> Cholesky returns NaN
      -> preconditioning raised, blaming positive-definiteness

Metric preconditioning was the messenger, not the cause. Once it is off (the opt-in
default), the same fit runs to completion and returns a posterior in which **every**
free parameter is NaN, with ``rhat()`` all NaN — and **nothing raises**. That is the
hazard: a result-shaped object the caller can index, plot, summarize and publish,
with nothing in the return path saying it is meaningless.

(An earlier revision of this docstring justified that differently, claiming a notebook
asserting ``max_rhat < 1.01`` "never fires, because NaN compares false against any
threshold". The premise is right and the conclusion is backwards. ``nan < 1.01`` is
``False``, so ``assert max_rhat < 1.01`` **fails**, loudly — the upper-bound form is
NaN-safe. Only the negated form ``assert not (max_rhat > 1.01)`` passes vacuously,
and a census of this repo found 10 threshold assertions on convergence metrics and
**zero** in that form. The guard below is still right; it just does not need, and
should not teach, that rule.)

So the crash was doing useful work. This pins the guard that keeps it loud without
keeping it wrong: a MAP estimate containing NaN or inf is not a usable point for any
downstream backend, and every MCMC/VI backend takes ``init_from`` from exactly here.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


class TestTheGuardItself:
    """Unit-level: the check, independent of running a real optimizer."""

    def test_an_all_nan_point_is_rejected(self):
        from tengri.inference.backends.map_dispatch import _reject_nonfinite_map

        with pytest.raises(ValueError) as excinfo:
            _reject_nonfinite_map({"a": np.array(np.nan), "b": np.array(np.nan)})
        assert "nan" in str(excinfo.value).lower()

    def test_a_partially_nan_point_is_also_rejected(self):
        """One NaN coordinate makes the whole point unusable as an init."""
        from tengri.inference.backends.map_dispatch import _reject_nonfinite_map

        with pytest.raises(ValueError, match="a"):
            _reject_nonfinite_map({"a": np.array(np.nan), "b": np.array(1.0)})

    def test_an_inf_point_is_rejected(self):
        from tengri.inference.backends.map_dispatch import _reject_nonfinite_map

        with pytest.raises(ValueError):
            _reject_nonfinite_map({"a": np.array(np.inf)})

    def test_the_message_names_the_offending_parameters(self):
        from tengri.inference.backends.map_dispatch import _reject_nonfinite_map

        with pytest.raises(ValueError) as excinfo:
            _reject_nonfinite_map({"good": np.array(1.0), "dust_tau_bc": np.array(np.nan)})
        msg = str(excinfo.value)
        assert "dust_tau_bc" in msg, f"offending parameter not named: {msg}"
        assert "good" not in msg, f"names a parameter that was fine: {msg}"

    def test_a_finite_point_passes(self):
        from tengri.inference.backends.map_dispatch import _reject_nonfinite_map

        _reject_nonfinite_map({"a": np.array(1.0), "b": np.array([2.0, 3.0])})  # no raise

    def test_a_finite_vector_parameter_passes(self):
        """Field latents arrive as arrays, not scalars."""
        from tengri.inference.backends.map_dispatch import _reject_nonfinite_map

        _reject_nonfinite_map({"sfh_field_xi": np.zeros(128)})  # no raise

    def test_a_vector_parameter_with_one_nan_is_rejected(self):
        from tengri.inference.backends.map_dispatch import _reject_nonfinite_map

        bad = np.zeros(64)
        bad[17] = np.nan
        with pytest.raises(ValueError, match="sfh_field_xi"):
            _reject_nonfinite_map({"sfh_field_xi": bad})
