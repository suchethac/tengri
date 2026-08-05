# SPDX-License-Identifier: BSD-3-Clause
"""Regression: multi-start MAP must keep the best FINITE restart (#1397).

``_run_map_multistart`` runs ``n_restarts`` independent ADAM optimizations and
keeps the best -- the whole point being that some inits land in bad basins and
the good ones rescue the fit. The selection was::

    best = int(jnp.argmin(final_losses))

and ``jnp.argmin`` returns index 0 when the vector contains NaN, because every
comparison against NaN is false. So a single diverged restart made "keep the
best" keep the *worst*, discarding perfectly good optima.

Measured on ``recipes.mock_recovery_minimal()`` at ``snr=20`` (the model
``notebooks/01_why_jax.py`` teaches), 1000 steps, 8 restarts::

    per-restart final losses: [nan nan nan nan nan nan nan 4.635]
    argmin    -> idx 0, loss nan
    nanargmin -> idx 7, loss 4.635

The consequences ran downhill from there: ``_maybe_map_init`` accepted the NaN
params *and cached them on the model*, then NUTS preconditioning built a metric
at a NaN point, ``jnp.maximum(nan, floor)`` left it NaN, and the Cholesky guard
raised "metric is not positive definite -- build it with
`negative_hessian_metric`" at a caller that had just done exactly that.

These tests pin the selection rule itself, which is where the defect was.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.inference.backends.map_dispatch import _best_finite_restart

pytestmark = pytest.mark.regression_bug


class TestBestFiniteRestart:
    def test_picks_the_finite_minimum_when_a_nan_is_present(self):
        """LOAD-BEARING. Neuter: `return int(jnp.argmin(losses))`.

        This is the exact shape measured on the #1397 reproducer -- every
        restart diverged but the last. `jnp.argmin` returns 0 here.
        """
        losses = jnp.asarray([np.nan] * 7 + [4.635])
        assert _best_finite_restart(losses) == 7

    def test_nan_in_first_position_does_not_win(self):
        """`jnp.argmin` returns 0 for this input; the finite minimum is index 2."""
        losses = jnp.asarray([np.nan, 5.0, 3.0, np.nan, 7.0])
        assert _best_finite_restart(losses) == 2

    def test_infinities_are_rejected_too(self):
        """+/-inf are as unusable as NaN: an inf loss is a diverged restart.

        Note -inf would *win* a plain argmin, which is worse than losing to NaN:
        it is silently selected as the best possible fit.
        """
        losses = jnp.asarray([np.inf, 2.0, -np.inf, 9.0])
        assert _best_finite_restart(losses) == 1

    def test_all_finite_matches_plain_argmin(self):
        """The ordinary case is untouched."""
        losses = jnp.asarray([5.0, 3.0, 9.0, 4.0])
        assert _best_finite_restart(losses) == int(jnp.argmin(losses))

    def test_single_restart_is_honored(self):
        losses = jnp.asarray([2.5])
        assert _best_finite_restart(losses) == 0

    def test_all_non_finite_raises_and_says_so(self):
        """Silently returning a NaN point is what caused #1397's downstream mess.

        When there is genuinely nothing to select, the failure must name itself
        here rather than resurfacing several layers away as a Cholesky error.
        """
        losses = jnp.asarray([np.nan, np.nan, np.inf])
        with pytest.raises(ValueError) as excinfo:
            _best_finite_restart(losses)
        message = str(excinfo.value)
        assert "3" in message, "the message should say how many restarts diverged"
        assert "diverged" in message.lower()
        # It must point somewhere actionable, not just report failure.
        assert any(hint in message for hint in ("n_restarts", "learning_rate", "n_steps")), (
            f"message gives the user nothing to change: {message!r}"
        )

    def test_message_does_not_blame_the_caller_for_the_wrong_thing(self):
        """#1397's original error told the caller to do what it had already done.

        Guard against a repeat: the message must describe the optimizer
        diverging, not misattribute the failure to the data or the model being
        malformed.
        """
        with pytest.raises(ValueError) as excinfo:
            _best_finite_restart(jnp.asarray([np.nan, np.nan]))
        assert "positive definite" not in str(excinfo.value)


class TestTheCallSiteActuallyUsesIt:
    """Guards the wiring, not just the helper.

    The unit tests above pass even with ``_run_map_multistart`` reverted to
    ``jnp.argmin`` — they never touch the call site. This drives the real MAP
    path with the model from #1397, whose restarts all diverge, and asserts the
    failure is *reported*. With the old selection the diverged restart is picked
    silently and NaN params flow downstream, which is the whole bug.
    """

    def test_all_diverged_restarts_raise_instead_of_returning_nan(self, ssp_data_fsps):
        import jax
        import numpy as np

        from tengri import Observation, Photometry, SEDModel, generate_mock, recipes
        from tengri.inference.fitter import Fitter

        obs = Observation(
            photometry=Photometry.from_names(
                ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1"]
            )
        )
        model = SEDModel.build(
            ssp_data=ssp_data_fsps, observation=obs, **recipes.mock_recovery_minimal()
        )
        truth = model.spec.sample(jax.random.PRNGKey(0))
        mock = generate_mock(model, truth, key=jax.random.PRNGKey(1), snr=20.0)
        fitter = Fitter(model, mock["flux_obs"], mock["noise"], data_type="photometry")

        try:
            result = fitter._run_map(
                key=jax.random.PRNGKey(2), n_steps=1000, n_restarts=8, verbose=False
            )
        except ValueError as exc:
            assert "diverged" in str(exc).lower()
            return

        # If it did not raise, at least one restart survived — then the returned
        # params must be that finite one, never a NaN.
        assert all(bool(np.isfinite(np.asarray(v))) for v in result.params.values()), (
            "MAP returned non-finite params instead of selecting a finite restart"
        )
