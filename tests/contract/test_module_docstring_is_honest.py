# SPDX-License-Identifier: BSD-3-Clause
"""The package docstring describes the API that exists (#1283).

It opened with "``__all__`` is organized into tiers reflecting common
workflows" and then listed, under **Core (user-facing classes for a first
fit)**, eleven names ``__all__`` deliberately excludes — ``Photometry``,
``Observation``, ``Fitter``, ``Posterior``, ``Prediction``, ``NoiseModel``,
``Spectroscopy``, ``VIConfig``, ``Instrument``, ``LineList``. It also omitted
65 names ``__all__`` does contain. So it was wrong in both directions, and a
reader who trusted it wrote ``from tengri import *`` and hit ``NameError`` on
the first line of their first fit.

The ``__all__`` exclusion is correct and deliberate: ``docs/dev/api_migration_v0.x.md``
establishes the sub-namespace as the canonical import path, and keeping these
names out of ``__all__`` is what keeps star-import and ``tengri.<TAB>`` clean.
The defect was entirely in the prose.
"""

from __future__ import annotations

import re

import pytest

import tengri

pytestmark = pytest.mark.contract

DOC = tengri.__doc__ or ""

#: Names the docstring used to advertise as star-importable Core.
DELIBERATELY_NOT_IN_ALL = [
    "Photometry",
    "Observation",
    "Spectroscopy",
    "NoiseModel",
    "Fitter",
    "Posterior",
    "Prediction",
    "VIConfig",
]


def test_the_docstring_exists_and_is_substantial():
    """Guard the guard: an empty docstring would pass everything below."""
    assert len(DOC) > 800, "package docstring is unexpectedly short"


def test_it_no_longer_claims_all_is_the_tiered_map():
    assert "organized into tiers" not in DOC, (
        "the docstring claims __all__ is a tiered map of the public surface. "
        "It is not — it deliberately excludes the sub-namespace classes."
    )


def test_every_name_the_docstring_cites_exists():
    """A cited name that does not resolve is a broken instruction."""
    cited = set(re.findall(r"``(\w+)``", DOC))
    missing = sorted(n for n in cited if not hasattr(tengri, n))
    assert not missing, f"docstring cites names that do not exist: {missing}"


@pytest.mark.parametrize("name", DELIBERATELY_NOT_IN_ALL)
def test_the_excluded_names_are_still_importable_and_completable(name):
    """The premise of the fix: these are public, just not star-importable."""
    assert hasattr(tengri, name), f"{name} should remain reachable as tengri.{name}"
    assert name not in tengri.__all__, (
        f"{name} entered __all__. That may be right — but it contradicts "
        "api_migration_v0.x.md, so change the docs in the same PR."
    )


def test_it_names_the_canonical_import_path():
    """The docstring must tell the reader where these classes really live."""
    assert "from tengri.observation import" in DOC, (
        "the docstring should show the canonical sub-namespace import, since "
        "that is where Photometry/Observation/NoiseModel canonically come from"
    )
    assert "canonical" in DOC.lower()


def test_the_canonical_imports_it_shows_actually_work():
    """Execute the import lines rather than trusting them."""
    lines = [
        ln.strip()
        for ln in DOC.splitlines()
        if ln.strip().startswith("from tengri.") and " import " in ln
    ]
    assert lines, "no sub-namespace import lines found in the docstring"
    for stmt in lines:
        exec(compile(stmt, "<docstring>", "exec"), {})


def test_it_teaches_the_canonical_fit_entry_point():
    assert ".fit(" in DOC, "the docstring's worked example should run a fit"
    assert "SEDModel.build" in DOC


def test_it_does_not_teach_a_deprecated_surface():
    """SEDModel.fit and the Galaxy facade are deprecated/demoted."""
    assert "sed.fit(" not in DOC and "model.fit(" not in DOC, (
        "SEDModel.fit is deprecated in favour of ForwardModel.fit (#211)"
    )
    assert "Galaxy" not in DOC, "Galaxy is demoted; the docstring should not teach it"
