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
        "ess_min": 300.0,
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
        "ess_min": 300.0,
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


# --- an unrecorded divergence count is not a clean one ----------------------


def test_an_absent_divergence_count_is_not_read_as_zero():
    """`or 0` made a cell nobody measured look perfectly clean.

    The relaxed bar gates on the divergence rate, so an absent count passed
    that half vacuously. is_adopted already refuses a cell with no rhat_max;
    there is no reason the divergence count should default in its own favor.
    """
    import math

    from _adoption import divergence_rate

    assert math.isnan(divergence_rate({"n_samples": 300, "n_chains": 4}))


def test_a_recorded_zero_is_still_a_clean_cell():
    """Non-vacuity: an honest zero must not be confused with an absent one."""
    from _adoption import divergence_rate

    assert divergence_rate({"divergences": 0, "n_samples": 300, "n_chains": 4}) == 0.0


def test_the_relaxed_bar_refuses_a_cell_with_no_divergence_count():
    from _adoption import is_adopted

    verdict = is_adopted({"rhat_max": 1.004, "n_samples": 300, "n_chains": 4}, "III")
    assert not verdict.adopted
    assert "divergence count" in verdict.reason


def test_the_relaxed_bar_still_adopts_a_clean_cell():
    """The guard must not refuse everything."""
    from _adoption import is_adopted

    verdict = is_adopted(
        {"rhat_max": 1.004, "divergences": 0, "ess_min": 300.0, "n_samples": 300, "n_chains": 4},
        "III",
    )
    assert verdict.adopted, verdict.reason


# --- the ESS leg the owner added to the bar, and the peak_gyr cap -----------


def test_the_relaxed_bar_refuses_a_low_ess_cell():
    """fit_one.ESS_FLOOR is part of the grid's bar; the relaxed bar computes
    its own verdict, so without this leg it would adopt what the grid refuses."""
    from _adoption import is_adopted

    verdict = is_adopted(
        {"rhat_max": 1.004, "divergences": 0, "ess_min": 3.0, "n_samples": 300, "n_chains": 4},
        "III",
    )
    assert not verdict.adopted
    assert "ess_min" in verdict.reason


def test_the_relaxed_bar_refuses_a_cell_with_no_ess_recorded():
    from _adoption import is_adopted

    verdict = is_adopted(
        {"rhat_max": 1.004, "divergences": 0, "n_samples": 300, "n_chains": 4}, "III"
    )
    assert not verdict.adopted
    assert "ess_min" in verdict.reason


def test_a_capped_v_cell_is_not_flagged():
    """Non-vacuity: a post-cap cell must pass."""
    from _adoption import uncapped_peak_note

    meta = {"priors": {"sfh_lnorm_peak_gyr": "Uniform(0.1, 5.4)"}}
    assert uncapped_peak_note(meta, "V") is None


def test_an_uncapped_v_cell_is_flagged():
    from _adoption import uncapped_peak_note

    meta = {"priors": {"sfh_lnorm_peak_gyr": "Uniform(0.1, 13.0)"}}
    note = uncapped_peak_note(meta, "V")
    assert note is not None and "13" in note


def test_a_v_cell_with_no_priors_block_is_flagged():
    """The defect this exists for: the block and the cap landed together, so
    absence means the cell predates the cap. A bound test on a missing key
    returns None and skips such a cell into the passing branch."""
    from _adoption import uncapped_peak_note

    note = uncapped_peak_note({"adoption_pass": True, "ess_min": 380.0}, "V")
    assert note is not None and "predates" in note


def test_a_v_cell_whose_prior_cannot_be_read_is_flagged():
    """Unparseable is not the same as capped."""
    from _adoption import uncapped_peak_note

    note = uncapped_peak_note({"priors": {"sfh_lnorm_peak_gyr": "custom object"}}, "V")
    assert note is not None


def test_the_peak_cap_check_is_configuration_v_only():
    """peak_gyr belongs to the log-normal SFH; III has no such parameter."""
    from _adoption import uncapped_peak_note

    assert uncapped_peak_note({}, "III") is None
    assert uncapped_peak_note({}, "VI") is None


# --- the relaxed bar follows the SFH, not the label (#2496, owner 2026-09-24) ---


def test_the_relaxed_set_is_the_two_continuity_rows_and_iii():
    """I and VI are the continuity rows; III keeps it although it no longer needs it.

    The exception was introduced for III on the grounds that a nonparametric
    continuity SFH cannot reach a zero-divergence bar at this dimensionality.
    The table was later diversified and that SFH moved to I and VI, while the
    frozenset kept the string. Pinning membership here so a future reshuffle
    has to come past a test rather than past a comment.
    """
    assert frozenset({"I", "III", "VI"}) == RELAXED_CONFIGS


@pytest.mark.parametrize(
    ("config", "divergences", "rhat", "ess"),
    [
        ("I", 6, 1.0024, 504.8),  # 24786/I, measured
        ("I", 1, 1.0015, 468.7),  # 26398/I, measured
        ("VI", 15, 1.0014, 276.7),  # 79/VI under the #2495 reparametrization
    ],
)
def test_a_continuity_row_is_adopted_on_the_relaxed_bar(config, divergences, rhat, ess):
    """Cells the strict bar refuses only for a nonzero divergence count."""
    meta = {
        "adoption_pass": False,
        "rhat_max": rhat,
        "divergences": divergences,
        "ess_min": ess,
        "n_samples": 300,
        "n_chains": 4,
    }
    verdict = is_adopted(meta, config)
    assert verdict.adopted is True, verdict.reason
    assert "relaxed" in verdict.reason
    assert rate_of(meta) <= 0.015


@pytest.mark.parametrize("config", ["I", "VI"])
def test_a_continuity_row_still_fails_past_the_rate(config):
    """The relaxation is a rate, not an amnesty: 267/VI at 6.0% stays refused."""
    meta = {
        "adoption_pass": False,
        "rhat_max": 1.0052,
        "divergences": 72,
        "ess_min": 300.0,
        "n_samples": 300,
        "n_chains": 4,
    }
    assert is_adopted(meta, config).adopted is False


@pytest.mark.parametrize("config", ["I", "VI"])
def test_a_continuity_row_still_fails_on_effective_samples(config):
    """1826/I: rate 0.0025 and R-hat 1.0010, refused on ess_min 81."""
    meta = {
        "adoption_pass": False,
        "rhat_max": 1.0010,
        "divergences": 3,
        "ess_min": 81.0,
        "n_samples": 300,
        "n_chains": 4,
    }
    verdict = is_adopted(meta, config)
    assert verdict.adopted is False
    assert "ess_min" in verdict.reason


def test_the_retune_ladder_reads_the_same_relaxed_set_as_the_judge():
    """fit_one must not restate the bar it stops retuning on.

    The ladder decides when to stop; this module decides whether the saved cell
    counts. A second copy of the membership would let them drift, and the
    failure is silent: the ladder keeps retuning a cell the census already
    adopts, and a later rung can carry fewer effective samples than the one it
    replaces. Identity, not equality, so a copied literal fails.
    """
    from importlib import import_module

    # fit_one uses relative imports, so it only loads as a package module, and
    # that loads _adoption a second time under its package-qualified name. The
    # comparison has to be against the instance fit_one itself bound, or this
    # would fail on the duplicate import rather than on a copied literal.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    fit_one = import_module("analysis.paper1.fit_one")
    shared = import_module("analysis.paper1._adoption")

    assert fit_one.RELAXED_CONFIGS is shared.RELAXED_CONFIGS
    assert fit_one.RELAXED_DIVERGENCE_RATE is shared.RELAXED_DIVERGENCE_RATE
    assert fit_one.RELAXED_RHAT_MAX is shared.RELAXED_RHAT_MAX
    # and the duplicate import must still agree in value, or the two copies of
    # the module have drifted and every other test here judges the wrong one.
    assert shared.RELAXED_CONFIGS == RELAXED_CONFIGS
