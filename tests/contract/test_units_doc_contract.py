# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests: docs/units.md must describe the units surface that exists.

`docs/units.md` is the canonical units reference, and its code block is a list
of literal calls a reader copies. Three ways it had drifted (#1613):

* it documented ``ab_to_vega(mag_ab, band)`` with a *band name from the filter
  registry* — the parameter is a float offset, and the offsets dict was not
  reachable from ``tengri.units`` at all;
* it showed ``list_filter_conventions()`` returning a dict, which has been a
  ``_RegistryTable`` since #1295, so the documented subscript was a TypeError;
* it named 22 of 33 helpers and dropped one side of four inverse pairs.

``check_doc_examples.py`` cannot catch any of these: it resolves *symbols*, and
every symbol involved exists. What was wrong was the *call*. These tests pin
the rules instead of the three instances.
"""

import pathlib
import re

import pytest

from tengri import units

pytestmark = pytest.mark.contract

_PAGE = pathlib.Path(__file__).resolve().parents[2] / "docs" / "units.md"
_TEXT = _PAGE.read_text(encoding="utf-8")

# Every ``units.<name>`` the page mentions, in prose or in a code fence. The
# lookbehind keeps `astropy.units.Quantity` out: the page discusses astropy in
# the "Why no astropy.units" section, and those are not tengri names.
_MENTIONED = set(re.findall(r"(?<!astropy\.)\bunits\.([A-Za-z_][A-Za-z0-9_]*)", _TEXT))


def test_every_units_name_the_page_uses_exists():
    """A reader copying ``units.foo`` must get a real attribute."""
    assert _MENTIONED, "parsed no `units.` references — the parser broke, not the page"
    missing = sorted(n for n in _MENTIONED if not hasattr(units, n))
    assert not missing, f"docs/units.md uses names that do not exist on tengri.units: {missing}"


def _inverse_pairs() -> list[tuple[str, str]]:
    """``x_to_y``/``y_to_x`` pairs discovered from the public surface.

    Derived from ``__all__`` at runtime rather than listed, so a conversion
    added later is covered the day it lands.
    """
    exported = set(units.__all__)
    pairs = set()
    for name in exported:
        parts = name.split("_to_")
        if len(parts) != 2:
            continue
        mirror = f"{parts[1]}_to_{parts[0]}"
        if mirror in exported:
            pairs.add(tuple(sorted((name, mirror))))
    return sorted(pairs)


def test_inverse_pair_discovery_is_not_vacuous():
    """Guard the introspection above: if it stops finding pairs, it stops testing."""
    pairs = _inverse_pairs()
    assert len(pairs) >= 8, f"only found {len(pairs)} inverse pairs: {pairs}"
    assert ("ab_mag_to_fnu", "fnu_to_ab_mag") in pairs


@pytest.mark.parametrize(("first", "second"), _inverse_pairs())
def test_the_page_documents_both_directions_or_neither(first, second):
    """Documenting one half of a round trip strands the reader coming back.

    The page's own intent is symmetric — it annotates the Jansky pair with
    ``# inverse`` — so a one-sided pair is an omission, not a policy.
    """
    has_first, has_second = first in _MENTIONED, second in _MENTIONED
    assert has_first == has_second, (
        f"docs/units.md documents only one direction: "
        f"{first}={'documented' if has_first else 'MISSING'}, "
        f"{second}={'documented' if has_second else 'MISSING'}"
    )


def test_vega_offsets_are_reachable_from_the_module_that_needs_them():
    """``ab_to_vega``'s only other argument must live where ``ab_to_vega`` does.

    The module contract in ``tengri/units/__init__.py`` is that users can write
    ``from tengri import units`` *without reaching into* ``utils``. Until #1613
    the offsets were exported only from ``tengri.utils``, so the one documented
    use of these two functions was impossible from ``units`` alone.
    """
    assert hasattr(units, "AB_VEGA_OFFSETS")
    assert "AB_VEGA_OFFSETS" in units.__all__


def test_the_documented_vega_call_runs():
    """The page's literal example, executed."""
    offset = units.AB_VEGA_OFFSETS["V"]
    assert isinstance(offset, float)
    assert float(units.ab_to_vega(20.0, offset)) == pytest.approx(20.0 - offset)
    assert float(units.vega_to_ab(20.0 - offset, offset)) == pytest.approx(20.0)


def test_vega_offset_keys_are_short_band_names_not_registry_ids():
    """The page promises short names; a registry id must not silently work."""
    assert "V" in units.AB_VEGA_OFFSETS
    assert "r" in units.AB_VEGA_OFFSETS
    assert "sdss_r" not in units.AB_VEGA_OFFSETS


def test_page_does_not_claim_list_filter_conventions_returns_a_dict():
    """It has been a _RegistryTable since #1295; the dict subscript is a TypeError."""
    import tengri

    conv = tengri.list_filter_conventions()
    assert not isinstance(conv, dict), "return type changed — update docs/units.md"
    assert conv.names() == ["bessell", "energy"]
    # The page must not show a dict literal keyed by convention name.
    assert "# {'bessell'" not in _TEXT, "docs/units.md still shows the pre-#1295 dict output"
