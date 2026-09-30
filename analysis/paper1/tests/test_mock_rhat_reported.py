#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The mock fit must report R-hat, not silently omit it.

fit_mock_joint.py read the between-chain diagnostic as
``posterior.rhats() if hasattr(posterior, "rhats") else {}``. No method named
``rhats`` exists on Posterior -- it is ``rhat`` -- so the guard fired on every
run and every mock fit ever produced wrote an empty R-hat dictionary. Nothing
failed, nothing warned, and a six-hour fit landed with no way to tell whether
its four chains had agreed. Recomputing R-hat by hand from the saved draws
afterwards gave a maximum of 1.055 with ten of sixty-eight parameters at or
above 1.01, so the diagnostic that was missing was also the one that mattered.

A hasattr guard around a method name is only safe when the name is right, and
nothing checks that. These tests check it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PAPER1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER1))

pytestmark = pytest.mark.contract


def test_posterior_exposes_rhat_and_not_rhats():
    """Pin the spelling the caller has to use."""
    posterior_mod = pytest.importorskip("tengri.inference.posterior")
    cls = posterior_mod.Posterior
    assert hasattr(cls, "rhat"), "Posterior.rhat is the between-chain diagnostic"
    assert not hasattr(cls, "rhats"), (
        "a Posterior.rhats now exists; the plural was previously a typo that "
        "silenced the diagnostic, so decide deliberately which one callers use"
    )


def test_the_mock_fit_calls_the_method_that_exists():
    src = (PAPER1 / "fit_mock_joint.py").read_text()
    assert "posterior.rhat()" in src
    assert "posterior.rhats(" not in src, "rhats() does not exist and returns nothing"


def test_rhat_is_not_hidden_behind_a_hasattr_guard():
    """The guard is the defect, not the spelling.

    Even spelled correctly, `x.rhat() if hasattr(x, "rhat") else {}` converts a
    future rename into an empty dictionary rather than an error, and an empty
    convergence diagnostic reads as "nothing to report" rather than "not
    measured".
    """
    src = (PAPER1 / "fit_mock_joint.py").read_text()
    assert 'hasattr(posterior, "rhat")' not in src
    assert 'hasattr(posterior, "rhats")' not in src
