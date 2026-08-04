# SPDX-License-Identifier: BSD-3-Clause
"""Contract: reading redshift with a silent 0.0 default is not allowed (#1432).

``params.get("redshift", 0.0)`` predates
:func:`tengri.parameters.resolve.resolve_fixed_params` and is a fossil. Every
dict reaching a component has already passed one of two boundaries that inject a
``Fixed`` redshift — ``Prediction`` (which applies ``resolve_fixed_params``) and
the forward pipeline (``{**fixed_values, **params}``) — so the default is
unreachable on those paths, and a default that cannot be reached is not a safety
net but a silencer.

It is not hypothetical. ``resolve.py``'s own docstring records the incident:
``redshift=Fixed(0.5)`` omitted from a params dict made ``project_photometry``
compute the flux at 10 pc, *"~16 orders of magnitude off, with no warning"*.

**This pins the rule, not the 23 instances.** Asserting a list of converted call
sites would pass forever while a new one is added next to them. Scanning for the
*pattern* is what actually holds the line — a reviewer does not have to notice
the idiom returning.

The exceptions are enumerated, with the reason each is legitimate, so adding one
is a deliberate act that shows up in review rather than a silent re-entry.
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

pytestmark = pytest.mark.contract

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "tengri"

#: ``<module path>`` -> why a redshift default is correct there.
ALLOWED: dict[str, str] = {
    "components/nebular/line_precompute.py": (
        "caller-supplied precompute reference params; wants a documented "
        "reference z, not an exception"
    ),
    "components/nebular/nebular_grid_precompute.py": (
        "same as line_precompute — reference params for the stored LUT"
    ),
    "inference/likelihood.py": (
        "chains params -> fixed_values -> 0.0, which is the correct pattern: "
        "the 0.0 is reached only if BOTH dicts lack the key"
    ),
}

# `<recv>.get("redshift", <anything>)` — any default, any receiver.
DEFAULTED = re.compile(r"""\w+\.get\(\s*['"]redshift['"]\s*,""")


def _docstring_lines(tree: ast.AST) -> set[int]:
    """Line numbers occupied by docstrings, so prose is not mistaken for code."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = node.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            out.update(range(body[0].lineno, (body[0].end_lineno or body[0].lineno) + 1))
    return out


def _offending_sites() -> list[tuple[str, int, str]]:
    found: list[tuple[str, int, str]] = []
    for py in sorted(SRC.rglob("*.py")):
        rel = str(py.relative_to(SRC))
        if rel in ALLOWED:
            continue
        text = py.read_text(encoding="utf-8")
        if "redshift" not in text:
            continue
        try:
            docs = _docstring_lines(ast.parse(text))
        except SyntaxError:  # pragma: no cover - src must parse
            docs = set()
        for i, line in enumerate(text.splitlines(), 1):
            if i in docs:
                continue  # prose *about* the idiom is fine, and resolve.py has some
            if DEFAULTED.search(line):
                found.append((rel, i, line.strip()))
    return found


def test_no_silent_redshift_default_outside_the_allowlist():
    """A silent redshift default must not reappear in src/tengri."""
    offenders = _offending_sites()
    assert not offenders, (
        "redshift read with a silent default outside the allowlist:\n"
        + "\n".join(f"  {r}:{n}\n      {ln}" for r, n, ln in offenders)
        + "\n\nUse require_redshift(params, where) from tengri.parameters.resolve. "
        "If the default is genuinely correct (the dict can legitimately lack the "
        "key), add the module to ALLOWED in this file with the reason."
    )


def test_the_allowlist_is_not_stale():
    """Every allowlisted module must still contain the pattern it excuses.

    Without this, a module stays excused after its default is removed, and the
    allowlist silently grants permission nobody is using — the same rot as a
    stale suppression comment.
    """
    stale = []
    for rel in ALLOWED:
        path = SRC / rel
        if not path.exists():
            stale.append(f"{rel} (file no longer exists)")
            continue
        if not DEFAULTED.search(path.read_text(encoding="utf-8")):
            stale.append(f"{rel} (no longer defaults redshift — drop the entry)")
    assert not stale, "stale ALLOWED entries:\n  " + "\n  ".join(stale)


def test_require_redshift_raises_with_an_actionable_message():
    """The error must name the caller and say what to do, not just what broke."""
    from tengri.parameters.resolve import require_redshift

    assert require_redshift({"redshift": 0.7}, "demo") == 0.7

    with pytest.raises(KeyError) as exc:
        require_redshift({}, "components.radio.radio_model.predict")
    msg = str(exc.value)
    assert "components.radio.radio_model.predict" in msg, "must name the caller"
    assert "resolve_fixed_params" in msg, "must name the boundary that fills it"
    assert "10 pc" in msg, "must say why a 0.0 default would be wrong"


def test_require_redshift_does_not_coerce():
    """Returns the stored object untouched, so a traced value stays traced."""
    import jax.numpy as jnp

    from tengri.parameters.resolve import require_redshift

    arr = jnp.asarray(1.25)
    assert require_redshift({"redshift": arr}, "demo") is arr
