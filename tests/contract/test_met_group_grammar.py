# SPDX-License-Identifier: BSD-3-Clause
"""#1720: ``met={'type': ...}`` — the metallicity group, parallel to ``sfh``.

The grammar had two anomalies stacked, and together they produced #1677:

* every group selects its model with ``type``; ``stellar`` alone used
  ``met_mode``;
* the metallicity group was **named** ``stellar``, and ``met_mode`` was its only
  structural key — a metallicity group under a generic name.

So ``met={'type': 'table'}`` — the form both conventions imply — was the one
form the grammar rejected. ``Catalog.from_histories`` advised exactly that and
was wrong only because the grammar was the outlier. Rather than keep correcting
the advice, the grammar gained the group.

``stellar={'met_mode': ...}`` (the #311 spelling) is **gone**, not deprecated.
Two ways to say one thing is the maintenance cost this removes, so every call
site in the repo moved with it and ``stellar=`` now raises carrying the one-line
translation.
"""

from __future__ import annotations

import warnings

import pytest

from tengri import FREE, Fixed
from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS, parse_groups
from tengri.parameters.priors import Uniform

pytestmark = pytest.mark.contract


def _parse(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if "redshift" not in kwargs:
            kwargs["redshift"] = Fixed(0.1)
        return parse_groups(**kwargs)


def test_met_type_selects_the_mode_like_every_other_group():
    assert _parse(met={"type": "table"}).met_mode == "table"
    assert _parse(met={"type": "ramp"}).met_mode == "ramp"


def test_met_reads_the_same_as_sfh_for_the_same_idea():
    """The whole point: tabulating either quantity should look identical."""
    spec = _parse(sfh={"type": "table"}, met={"type": "table"})
    assert spec.mean_sfh_type == ["table"]
    assert spec.met_mode == "table"


def test_the_removed_stellar_group_says_how_to_translate():
    """A group that existed one release ago deserves better than "unknown key".

    ``difflib`` will not suggest ``met`` for ``stellar`` — they share no prefix
    — so the generic unknown-group error would leave a reader holding a name
    that used to work with no way to find what replaced it.
    """
    with pytest.raises(ValueError, match="the 'stellar' group is gone") as exc:
        _parse(stellar={"met_mode": "table"})
    message = str(exc.value)
    assert "met={'type': 'table'}" in message, "give the literal replacement line"
    assert "met={'logzsol': ...}" in message, "and the per-parameter form too"

    # A translation has to name what is being translated *from*. Asserting only
    # the replacement forms is what let this message ship reading
    # "met={'type': 'table'} becomes met={'type': 'table'}" — both halves
    # rewritten to the new spelling by the rename sweep that created them, so
    # every assertion above still passed and the source-scanning guard in
    # test_error_message_advice_parses.py was *satisfied* by the corruption,
    # since the mangled text parses and the correct text does not.
    assert "met_mode" in message, "name the old structural key, or there is no mapping"
    assert "met_logzsol" in message, "name the old parameter key too"


def test_short_form_parameters_route_into_the_met_group():
    """``met={'logzsol_0': ...}`` must reach ``met_logzsol_0``, not vanish.

    Silently accepting an unrouted key is the failure this asserts against: the
    spec would build, the fit would run, and the value the user set would simply
    not be there.
    """
    spec = _parse(met={"type": "ramp", "logzsol_0": -1.5, "logzsol_final": 0.0})
    fixed = spec.get_fixed_values()
    assert fixed["met_logzsol_0"] == pytest.approx(-1.5)
    assert fixed["met_logzsol_final"] == pytest.approx(0.0)


def test_priors_and_the_wildcard_work_in_the_met_group():
    assert (
        "met_logzsol_0" in _parse(met={"type": "ramp", "logzsol_0": Uniform(-2, -1)}).free_params
    )
    freed = _parse(met={"type": "ramp", "all_params": FREE}).free_params
    assert {"met_logzsol_0", "met_logzsol_final"} <= set(freed)


def test_an_unknown_mode_names_the_valid_ones():
    with pytest.raises(ValueError, match="Unknown metallicity mode") as exc:
        _parse(met={"type": "tabel"})
    assert "table" in str(exc.value), "a near-miss should be suggested or at least listed"


def test_met_mode_inside_the_met_group_is_refused_with_the_right_form():
    """The old key inside the new group. Say which key this group uses."""
    with pytest.raises(ValueError, match="selects its mode with 'type'"):
        _parse(met={"met_mode": "table"})


def test_there_is_exactly_one_spelling_left():
    """The point of the change: no second form to keep working, test, or teach."""
    assert "stellar" not in _GROUP_STRUCTURAL_KEYS
    assert _GROUP_STRUCTURAL_KEYS["met"] == frozenset({"type", "*", "all_params"}), (
        "the met group selects with 'type' and takes the wildcard "
        "('all_params', or the legacy '*'), nothing else"
    )


def test_the_round_trip_emits_met_and_survives_reparsing():
    """``to_groups()`` must hand back a form the grammar accepts — including its own.

    A round trip that emitted the removed ``stellar`` group would produce output
    its own parser rejects, so this asserts on both halves: what comes out, and
    that it goes back in.
    """
    spec = _parse(met={"type": "ramp", "logzsol_0": -1.5})
    groups = spec.to_groups()
    assert groups["met"]["type"] == "ramp"
    assert "stellar" not in groups
    assert _parse(**groups).met_mode == "ramp"


def test_a_default_metallicity_emits_no_met_block():
    """Never force ``met={}`` onto a round trip that did not ask for it."""
    assert "met" not in _parse(sfh={"type": "dpl"}).to_groups()


def test_the_menu_teaches_the_met_form():
    """The menu hint is the line a reader copies; it has to be the current one."""
    import tengri

    hint = str(tengri.list_metallicity_modes())
    assert "met={'type':" in hint, hint[-200:]


def test_every_declared_group_is_a_group_the_parser_accepts():
    """The two group censuses cannot diverge.

    ``_GROUP_STRUCTURAL_KEYS`` says which keys a group accepts; ``valid_groups``
    said which groups exist. They were separate hand-maintained lists, and
    nothing checked that they agreed — a group added to one alone is either
    rejected as unknown or accepts any key at all. ``met`` was the edit that
    would have hit it.

    Asserted through the parser rather than by reading the source, so it holds
    however the derivation is spelled.
    """
    for group in sorted(k for k in _GROUP_STRUCTURAL_KEYS if "." not in k):
        try:
            _parse(**{group: {}})
        except Exception as exc:
            assert "Unknown group key" not in str(exc), (
                f"{group!r} is declared in _GROUP_STRUCTURAL_KEYS but the parser "
                f"rejects it as an unknown group: {exc}"
            )
    assert "met" in _GROUP_STRUCTURAL_KEYS, "the group this issue added"
