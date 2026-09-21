#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Every configuration must be judged; none may fall through.

fig05 and fig06 each decided adoption with a chain of per-configuration
branches ending in a fall-through that rejected whatever it did not name.
Both named I, II and III, so Configurations IV, V and VI were refused however
many of their cells were on disk, and both figures reported them "absent"
beside the files. The tests below pin the judgement for a configuration the
rule does not name, which is the case neither copy handled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _adoption import (
    RELAXED_CONFIGS,
    divergence_rate as rate_of,
    is_adopted,
)
from _figure_style import CONFIG_ORDER

pytestmark = pytest.mark.contract

UNRELAXED = [key for key in CONFIG_ORDER if key not in RELAXED_CONFIGS]


@pytest.mark.parametrize("config", UNRELAXED)
def test_every_unrelaxed_configuration_is_judged_on_adoption_pass(config):
    """Not only I and II. A configuration the rule does not name is still judged."""
    assert is_adopted({"adoption_pass": True}, config).adopted is True
    assert is_adopted({"adoption_pass": False}, config).adopted is False


@pytest.mark.parametrize("config", UNRELAXED)
def test_a_passing_cell_is_never_refused_for_being_unnamed(config):
    """The regression: IV, V and VI were refused whatever their diagnostics said."""
    passing = {"adoption_pass": True, "rhat_max": 1.0001, "divergences": 0}
    verdict = is_adopted(passing, config)
    assert verdict.adopted is True, f"{config} refused despite {passing}: {verdict.reason}"


def test_missing_adoption_pass_is_not_adopted():
    """Absence is not consent: a cell with no verdict recorded is not drawn."""
    assert is_adopted({}, "II").adopted is False
    assert is_adopted({"adoption_pass": None}, "II").adopted is False


def test_config_iii_is_accepted_on_the_relaxed_bar():
    """III does not clear a zero-divergence bar, so it is judged differently.

    13601/III in the archive: 12 divergences in 1200 draws is 1.0%, inside the
    1.5% bar, at R-hat 1.0012.
    """
    meta = {
        "adoption_pass": False,
        "rhat_max": 1.0012,
        "divergences": 12,
        "n_samples": 300,
        "n_chains": 4,
    }
    verdict = is_adopted(meta, "III")
    assert verdict.adopted is True
    assert "relaxed" in verdict.reason


def test_config_iii_is_refused_on_convergence():
    meta = {
        "adoption_pass": False,
        "rhat_max": 1.0187,
        "divergences": 4,
        "n_samples": 300,
        "n_chains": 4,
    }
    verdict = is_adopted(meta, "III")
    assert verdict.adopted is False
    assert "rhat_max" in verdict.reason


def test_config_iii_is_refused_on_divergence_rate():
    """69 divergences in 1200 draws is 5.75%, past the 1.5% bar.

    R-hat is held inside its own bar so the refusal can only come from the
    divergence rate; 9884/III fails both at once and would not separate them.
    """
    meta = {
        "adoption_pass": False,
        "rhat_max": 1.0056,
        "divergences": 69,
        "n_samples": 300,
        "n_chains": 4,
    }
    verdict = is_adopted(meta, "III")
    assert verdict.adopted is False
    assert "divergence rate" in verdict.reason


def test_config_iii_needs_a_recorded_rhat():
    verdict = is_adopted({"adoption_pass": False, "divergences": 0}, "III")
    assert verdict.adopted is False
    assert "rhat_max" in verdict.reason


def test_the_relaxation_does_not_leak_to_other_configurations():
    """The same diagnostics that carry III past the bar must not carry II."""
    meta = {
        "adoption_pass": False,
        "rhat_max": 1.0001,
        "divergences": 1,
        "n_samples": 300,
        "n_chains": 4,
    }
    assert is_adopted(meta, "III").adopted is True
    for config in UNRELAXED:
        assert is_adopted(meta, config).adopted is False, config


def test_divergence_rate_is_per_draw_over_all_chains():
    assert rate_of({"divergences": 12, "n_samples": 300, "n_chains": 4}) == pytest.approx(0.01)
    assert rate_of({"divergences": 0, "n_samples": 300, "n_chains": 4}) == 0.0


def test_divergence_rate_survives_a_cell_with_no_draw_counts():
    """A zero denominator must not raise inside a figure run.

    Each count is zeroed on its own. Zeroing both at once cannot tell whether
    a default was substituted for either, because the surviving zero carries
    the product to zero regardless -- which is how an `or` default for one of
    them passed this test while inventing 600 draws.
    """
    assert rate_of({"divergences": 5, "n_samples": 0, "n_chains": 4}) == 0.0
    assert rate_of({"divergences": 5, "n_samples": 300, "n_chains": 0}) == 0.0
    assert rate_of({"divergences": 5, "n_samples": 0, "n_chains": 0}) == 0.0


def test_divergence_rate_defaults_only_when_a_count_is_absent():
    """Absent means absent; recorded zero means zero."""
    assert rate_of({"divergences": 12}) == pytest.approx(12 / (600 * 4))
    assert rate_of({"divergences": 12, "n_chains": 4}) == pytest.approx(12 / (600 * 4))


# --- the bar cannot see a near-frozen chain ---------------------------------


def test_a_healthy_adopted_cell_gets_no_low_ess_note():
    """Non-vacuity: the detector must be able to stay quiet."""
    from _adoption import Verdict, low_ess_note

    meta = {"adoption_pass": True, "ess_min": 380.8, "rhat_max": 1.0037}
    assert low_ess_note(meta, Verdict(True, "adoption_pass")) is None


def test_an_adopted_cell_on_three_effective_samples_is_flagged():
    """The reported failure: passes the bar, carries almost no information.

    rhat_max 1.0089 clears 1.01 and there are no divergences, so the bar
    adopts it. ESS 3 out of 1200 draws is a frozen chain. Nothing in the bar
    can see that, because rhat_max and ess_min are extrema over *different*
    parameters: the worst R-hat and the worst ESS need not belong to the same
    one, so a good rhat_max bounds nothing about ess_min.
    """
    from _adoption import Verdict, low_ess_note

    meta = {"adoption_pass": True, "ess_min": 3.0, "rhat_max": 1.0089, "divergences": 0}
    note = low_ess_note(meta, Verdict(True, "adoption_pass"))
    assert note is not None
    assert "3.0" in note


def test_an_adopted_cell_with_no_recorded_ess_is_flagged():
    """Absence is not health. An unrecorded ESS is unverified, not fine."""
    from _adoption import Verdict, low_ess_note

    note = low_ess_note({"adoption_pass": True, "rhat_max": 1.004}, Verdict(True, "x"))
    assert note is not None and "unverified" in note


def test_a_refused_cell_is_not_flagged():
    """The note is about what the bar *accepted*; a refusal needs no second reason."""
    from _adoption import Verdict, low_ess_note

    meta = {"adoption_pass": False, "ess_min": 3.0}
    assert low_ess_note(meta, Verdict(False, "did not pass the adoption bar")) is None


def test_the_bar_itself_is_unchanged_by_the_detector():
    """low_ess_note must not quietly become part of the criterion.

    The bar is the owner's to set. This pins that a cell the bar adopts is
    still reported as adopted even when the note fires, so the detector
    informs and never reclassifies.
    """
    from _adoption import is_adopted, low_ess_note

    meta = {"adoption_pass": True, "ess_min": 3.0, "rhat_max": 1.0089}
    verdict = is_adopted(meta, "V")
    assert verdict.adopted is True
    assert low_ess_note(meta, verdict) is not None
