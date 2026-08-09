# SPDX-License-Identifier: BSD-3-Clause
"""Regression: an unknown fit kwarg raises TypeError, and says so well (#1630).

Two decisions collided on `main` and left it red.

#1605 added ``check_unknown_kwargs`` at the dispatch seam, which was a real
improvement: before it, a typo traveled into the backend and surfaced as
``TypeError: run_map() got an unexpected keyword argument 'lines'`` — naming a
function the caller never called, inside a backend they did not choose (#1469).
The new message names neither, lists what the method does take, and suggests
near-matches.

But it raised ``ValueError``, and #1378's regression test pins ``TypeError`` —
the type Python raises for an unexpected keyword argument, and the type the
caller gets for a bad keyword on ``fit()``, which is a function they *did*
call. So `test_unknown_kwargs_still_fail_loudly` broke.

This file pins **both** halves, so restoring either one cannot silently undo
the other: the type is ``TypeError``, and the message keeps #1605's qualities.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import FIXED, SEDModel, Uniform
from tengri.forward.forward_model import ForwardModel

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=FIXED,
    )


@pytest.fixture(scope="module")
def mock_data(model):
    params = model.spec.sample(jax.random.PRNGKey(0))
    flux = np.asarray(model.predict_photometry(params))
    return flux, 0.05 * np.abs(flux)


def test_unknown_fit_kwarg_raises_typeerror(model, mock_data):
    """An unexpected keyword argument is a TypeError, as it is anywhere in Python."""
    flux, err = mock_data
    fwd = ForwardModel.build(sed=model)

    with pytest.raises(TypeError):
        fwd.fit(flux, err, method="map", n_steps=3, calibration_marginalze=True)


def test_the_message_does_not_name_the_runner(model, mock_data):
    """#1605's improvement: never name a function the caller did not call.

    Pinned separately from the type so that flipping the type back cannot
    quietly reintroduce ``run_map() got an unexpected keyword argument``.
    """
    flux, err = mock_data
    fwd = ForwardModel.build(sed=model)

    with pytest.raises(TypeError) as excinfo:
        fwd.fit(flux, err, method="map", n_steps=3, calibration_marginalze=True)

    message = str(excinfo.value)
    assert "run_map" not in message, f"message names the backend runner: {message}"
    assert "calibration_marginalze" in message, "message does not name the offending kwarg"
    assert "does not accept" in message


def test_the_message_suggests_the_near_match(model, mock_data):
    """A one-character typo should be told what it nearly was."""
    flux, err = mock_data
    fwd = ForwardModel.build(sed=model)

    with pytest.raises(TypeError) as excinfo:
        fwd.fit(flux, err, method="map", n_step=3)

    message = str(excinfo.value)
    # Assert the hint itself, not merely that "n_steps" appears: the message
    # also prints the full accepted list, so a bare substring check passes with
    # the suggestion deleted. (It did — caught by neuter-checking this test.)
    assert "Did you mean" in message, f"no near-match suggestion offered: {message}"
    assert "n_step -> n_steps" in message, f"suggestion does not name the fix: {message}"


def test_a_valid_kwarg_is_not_rejected(model, mock_data):
    """Anti-vacuity: the guard must not simply refuse everything.

    Without this, deleting the accepted-name lookup and rejecting every kwarg
    would pass all three tests above.
    """
    flux, err = mock_data
    fwd = ForwardModel.build(sed=model)

    posterior = fwd.fit(flux, err, method="map", n_steps=3, verbose=False)
    assert posterior is not None
