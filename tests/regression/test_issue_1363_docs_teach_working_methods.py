# SPDX-License-Identifier: BSD-3-Clause
"""#1363: no published example may teach a ``tier="broken"`` inference method.

The spec's catalog examples all read ``cat.fit(method="native_vi_linear", ...)``. Both
native VI backends are registered ``tier="broken"``, and on the batched catalog path they
raise ``NotImplementedError`` for exactly the two features those examples demonstrate
(per-galaxy redshift, presence masks). A downloading astronomer copying §6.2 hit an
exception on their first catalog fit.

Prose that *names* a broken backend is fine and often necessary — release notes, ADRs, the
method menu, and benchmark reports all legitimately discuss it. What must never happen is a
**prescriptive** ``method="..."`` in an example a reader is meant to run. This pins that
distinction rather than banning the string.

Root cause worth remembering: ``CLAUDE.md`` asserted these backends were
``tier=experimental`` long after the registry moved them to ``broken``. The instruction
file is loaded into every agent session, so one stale line kept regenerating the bad
examples. Prose that states a tier is checked against the registry here too.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.regression_bug

REPO = Path(__file__).resolve().parents[2]

#: Prescriptive call sites: ``method="x"``, ``.fit(method="x")``, ``.run("x")``.
_PRESCRIPTIVE = re.compile(r'(?:method\s*=\s*|\.run\s*\(\s*)["\']([a-z0-9_]+)["\']')

#: Docs a reader is meant to copy from. Plans and dev benchmarks are working notes and
#: legitimately record what was run at the time, including broken backends.
_SEARCH_DIRS = ("docs/internal/specs", "docs/inference", "docs/user", "docs/advanced")
_SKIP_PARTS = ("/plans/", "/dev/", "/archive/", "/adr/")


def _broken_methods() -> set[str]:
    from tengri.inference._backend_registry import _BACKENDS

    return {name for name, entry in _BACKENDS.items() if entry.tier == "broken"}


def _published_markdown() -> list[Path]:
    out: list[Path] = []
    for d in _SEARCH_DIRS:
        root = REPO / d
        if root.is_dir():
            out += [p for p in root.rglob("*.md") if not any(s in str(p) for s in _SKIP_PARTS)]
    return out


def _code_block_lines(text: str):
    """Yield ``(lineno, line)`` for fenced code-block content only.

    The prescriptive/descriptive split is exactly this: a ``method="..."`` inside a fence
    is something a reader runs, whereas the same string in prose is discussing the
    backend (release notes, benchmark write-ups, the method menu). Scanning prose too
    would force those legitimate mentions to be reworded.
    """
    in_fence = False
    for lineno, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            yield lineno, line


def test_the_registry_actually_has_broken_backends():
    """Anti-vacuity: if nothing is ``broken`` the sweep below proves nothing."""
    import tengri  # noqa: F401  (registers the backends)

    assert _broken_methods(), "no tier='broken' backends — this guard would pass vacuously"


def test_no_published_example_prescribes_a_broken_method():
    import tengri  # noqa: F401

    broken = _broken_methods()
    files = _published_markdown()
    assert files, "found no published markdown to scan — the guard would pass vacuously"

    offenders: list[str] = []
    for path in files:
        for lineno, line in _code_block_lines(path.read_text()):
            for method in _PRESCRIPTIVE.findall(line):
                if method in broken:
                    rel = path.relative_to(REPO)
                    offenders.append(f"{rel}:{lineno}: method={method!r} is tier='broken'")
    assert not offenders, (
        "published examples prescribe a broken inference method:\n  "
        + "\n  ".join(offenders)
        + "\nUse a tier='primary' method (e.g. 'mcmc_nuts', 'mcmc_hmc', 'map')."
    )


def test_claude_md_does_not_misstate_a_backend_tier():
    """The agent instruction file must not contradict the registry."""
    import tengri  # noqa: F401
    from tengri.inference._backend_registry import _BACKENDS

    text = (REPO / "CLAUDE.md").read_text()
    wrong: list[str] = []
    for claimed_tier in ("primary", "experimental", "broken"):
        for m in re.finditer(rf"`tier={claimed_tier}`", text):
            window = text[max(0, m.start() - 400) : m.start()]
            # Bind the claim to the NEAREST preceding backend name. Scanning the whole
            # window for any name mis-attributes: a sentence may mention several
            # backends and assert a tier for only the last group named.
            nearest, pos = None, -1
            for name, entry in _BACKENDS.items():
                at = window.rfind(f'"{name}"')
                if at > pos:
                    nearest, pos = (name, entry), at
            if nearest and nearest[1].tier != claimed_tier:
                wrong.append(
                    f"CLAUDE.md claims {nearest[0]} is tier={claimed_tier}, "
                    f"registry says {nearest[1].tier}"
                )
    assert not wrong, "stale tier claims in CLAUDE.md:\n  " + "\n  ".join(sorted(set(wrong)))
