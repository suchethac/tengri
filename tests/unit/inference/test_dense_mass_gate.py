# SPDX-License-Identifier: BSD-3-Clause
"""The dense-mass cap is one seam, and crossing it is not silent (#2166).

``use_dense = <policy> and n_dim <= 30`` existed in four places with four
behaviors: NUTS logged the downgrade at INFO and only when ``verbose``, HMC
applied it silently, dynamic HMC applied it silently from a signature that
defaults to ``dense_mass_matrix=True``, and the catalog path did not apply the
cap at all -- under a comment claiming it used "the same policy the
single-galaxy samplers use". A caller who asked for a dense metric on a wide
problem therefore got a diagonal one, or an O(D^2) allocation, depending only on
which entry point they happened to use, and in three of the four cases with no
way to find out.

That matters because the fallback is not neutral. Structural measurement on a
D = 74 field posterior (``bench/reports/2026-09-06_block_metric_structure.md``)
puts the diagonal metric's recovery of the conditioning gap at 0.074 against a
raw condition number of 5.4e4, and on the metric side it leaves the geometry
worse than it found it. Losing a dense request silently is losing the sampler's
most consequential setting.

These tests assert the policy directly rather than through a fit: a fit would
measure the machine.
"""

import re
import warnings

import pytest

from tengri.config.exceptions import measurements_of
from tengri.inference.backends.mcmc.nuts import (
    DENSE_MASS_MAX_DIM,
    _resolve_dense_mass_matrix,
    resolve_dense_mass_gate,
)

pytestmark = pytest.mark.unit


def test_the_cap_these_tests_assume_is_the_one_the_code_uses():
    """The parametrized dimensions below straddle the cap; pin it explicitly."""
    assert DENSE_MASS_MAX_DIM == 30


@pytest.mark.parametrize("n_dim", [1, 5, 7, 8, 9, 20, 30])
def test_gate_agrees_with_the_auto_policy_below_the_cap(n_dim):
    """Below the cap the gate is exactly the #319 auto-policy, unchanged."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_dense_mass_gate(None, n_dim, method="mcmc_nuts") == (
            _resolve_dense_mass_matrix(None, n_dim)
        )


@pytest.mark.parametrize("n_dim", [1, 5, 7, 30])
def test_explicit_requests_round_trip_below_the_cap(n_dim):
    """An explicit True or False is honored, and says nothing, below the cap."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_dense_mass_gate(True, n_dim, method="mcmc_hmc") is True
        assert resolve_dense_mass_gate(False, n_dim, method="mcmc_hmc") is False


@pytest.mark.parametrize("n_dim", [31, 74, 137])
def test_dense_above_the_cap_falls_back_and_warns(n_dim):
    """Above the cap the request is refused, and the caller is told."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert resolve_dense_mass_gate(True, n_dim, method="mcmc_hmc", verbose=False) is False
    assert len(caught) == 1, [str(w.message) for w in caught]
    assert "DIAGONAL" in str(caught[0].message)


def test_the_warning_carries_the_numbers_it_reports():
    """A consumer reads the dimensions off the instance, not out of the prose."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_dense_mass_gate(True, 74, method="mcmc_hmc", verbose=False)
    assert measurements_of(caught[0].message) == {"n_dim": 74.0, "max_dim": 30.0}


def test_verbose_false_still_warns_because_a_lost_setting_is_not_verbosity():
    """``verbose=False`` quiets the log line, never the refusal."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        resolve_dense_mass_gate(True, 74, method="mcmc_nuts", verbose=False)
    assert len(caught) == 1


def test_diagonal_above_the_cap_is_silent():
    """Nothing was lost, so there is nothing to say."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert resolve_dense_mass_gate(False, 74, method="mcmc_nuts") is False
        # The auto-policy already chose diagonal at this D; it was not overruled.
        assert resolve_dense_mass_gate(None, 74, method="mcmc_nuts") is False


def test_no_dense_capable_seam_re_spells_the_cap():
    """The cap must not be re-spelled per backend; that is what created this.

    A source assertion rather than a behavioral one, deliberately: the defect
    being ratcheted is *duplication*, and a behavioral test passes happily
    against six copies that currently agree. Prose mentions of the old
    expression are fine and are what the history sections are made of, so only
    an assignment to ``use_dense`` is matched.
    """
    import pathlib

    import tengri.inference.backends.mcmc.hmc as hmc_mod

    root = pathlib.Path(hmc_mod.__file__).parent.parent.parent
    seams = [
        root / "backends" / "mcmc" / "nuts.py",
        root / "backends" / "mcmc" / "hmc.py",
        root / "backends" / "mcmc" / "dynamic_hmc.py",
        root / "catalog_fitter.py",
        root / "fitter.py",
    ]
    respelled = re.compile(r"use_dense\s*=.*n_dim\s*<=\s*30")
    for path in seams:
        text = path.read_text()
        assert "DENSE_MASS_MAX_DIM" in text or "resolve_dense_mass_gate" in text, (
            f"{path.name} does not reference the shared cap"
        )
        offenders = [line for line in text.splitlines() if respelled.search(line)]
        assert not offenders, f"{path.name} re-spells the cap: {offenders}"
