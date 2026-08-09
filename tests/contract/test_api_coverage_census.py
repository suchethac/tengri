# SPDX-License-Identifier: BSD-3-Clause
"""The API-reference guard must census *both* export lists (#1606).

``tools/check_api_coverage.py`` says "every exported symbol must appear in the
API reference". It censused ``tengri.__all__`` alone, but the top-level public
surface is two lists — ``__all__`` governs ``from tengri import *`` and
``_CURATED_DIR`` governs ``tengri.<TAB>``. Eleven names live only in the second,
and six of them had no autodoc entry anywhere:

    Instrument  ParameterInformation  list_instruments
    parameter_information  print_citations  print_logo

``Instrument`` is the sharpest: a whole feature — ``Instrument.JWST_NIRCam()``
bundles filters, noise and calibration defaults — with no page in the reference,
under a guard whose stated job is to catch exactly that.

Second instance of one rule. #1574 was the first: a contract test asserting
"every ``list_*`` returns one type" was green while 6 of 29 violated it, same
cause. #1455 had already diagnosed the drift — ``__all__`` had contract tests
and ``_CURATED_DIR``'s contents had none — and fixed the *membership* of the
curated list; what remained was every other guard's census.

So these tests pin the census, not only the six names: a sweep that quietly
narrows must fail, because a narrowed census passes everything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

import tengri

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import check_api_coverage as cov

#: The six that had no entry when the census was widened (#1606). Pinned by
#: name so deleting an entry is a failure rather than a quieter run.
FORMERLY_UNDOCUMENTED = (
    "Instrument",
    "ParameterInformation",
    "list_instruments",
    "parameter_information",
    "print_citations",
    "print_logo",
)


class _Stub:
    def __init__(self, all_names, curated):
        self.__all__ = all_names
        self._CURATED_DIR = curated


# ── the live invariant ─────────────────────────────────────────────────────


def test_the_export_surface_is_still_two_lists():
    """If these ever merge, the guard's union assertion must be revisited."""
    exported = set(tengri.__all__)
    curated = set(tengri._CURATED_DIR)
    assert exported and curated
    only_curated = {n for n in curated - exported if not n.startswith("__")}
    assert only_curated, (
        "_CURATED_DIR no longer contributes any name of its own. If the lists "
        "were deliberately merged, simplify check_api_coverage.census(); do not "
        "leave an assertion that can no longer fail."
    )


def test_the_census_covers_both_lists():
    names = set(cov.census(tengri))
    assert set(tengri.__all__) - {n for n in tengri.__all__ if n.startswith("__")} <= names
    for name in FORMERLY_UNDOCUMENTED:
        assert name in names, f"{name} dropped out of the census"


def test_every_public_name_has_an_api_reference_entry():
    """The contract itself, over the wide census."""
    documented = cov.documented_names()
    import inspect

    missing = [
        name
        for name in cov.census(tengri)
        if not inspect.ismodule(getattr(tengri, name, None))
        and name not in cov.ALLOWED_UNDOCUMENTED
        and name not in documented
    ]
    assert not missing, f"no API reference entry for: {missing}"


def test_the_six_are_documented_by_name():
    """Pin the instances too — a page deleted by hand should be loud."""
    documented = cov.documented_names()
    for name in FORMERLY_UNDOCUMENTED:
        assert name in documented, f"{name} lost its API reference entry (#1606)"


# ── guard the guard ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("label", "all_names", "curated"),
    [
        ("empty __all__", [], ["a"]),
        ("empty _CURATED_DIR", ["a"], []),
        ("curated is a subset", ["a", "b"], ["a"]),
        ("second list dropped", ["a", "b"], ["a", "b"]),
    ],
)
def test_census_raises_when_the_sweep_narrows(label, all_names, curated):
    """Each of these would silently shrink coverage and still report OK."""
    with pytest.raises(ValueError):
        cov.census(_Stub(all_names, curated))


def test_census_unions_and_drops_dunders():
    assert cov.census(_Stub(["a", "b"], ["c", "__version__"])) == ["a", "b", "c"]
