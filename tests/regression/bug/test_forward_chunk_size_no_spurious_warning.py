# SPDX-License-Identifier: BSD-3-Clause
r"""A guard that compares against the OLD default fires on the new one (#1189 follow-up).

#1189 swept ``forward_chunk_size``'s default from ``1`` to
:data:`~tengri.inference._batching.AUTO` at all six entry points. The *defaults*
were swept; the *readers* of the old default were not. ``CatalogFitter.run``
still asked

.. code-block:: python

    if forward_chunk_size != 1:
        warnings.warn(f"forward_chunk_size={forward_chunk_size} is ignored ...")

and ``AUTO`` is the string ``"auto"``, so ``"auto" != 1`` is ``True``.
``resolve_forward_chunk_size`` is never called inside ``run``, so the value is
still the raw sentinel at that line. Measured before the fix — a plain
``run("map")``, with nothing passed:

    forward_chunk_size=auto is ignored for method='map'. ...

Every caller of a non-vmappable method (``map``, ``vi``, ``nss``) was told a
setting they never set was being ignored. The 22 tests covering this parameter
all passed throughout: nothing asserted on the *absence* of a warning.

The general form, and why the fix is a predicate rather than a wider literal:
**a literal that encodes a default is a comparison against a value that can
move.** :func:`chunking_was_requested` is asked the question instead, in the
module that owns the sentinel, so the next change of default has one place to
update rather than N call sites.
"""

from __future__ import annotations

import ast
import contextlib
import pathlib
import warnings

import pytest

from tengri.inference._batching import AUTO, chunking_was_requested

pytestmark = pytest.mark.regression_bug

_REPO = pathlib.Path(__file__).resolve().parents[3]
_SRC = _REPO / "src" / "tengri"


# --------------------------------------------------------------------------
# 1. The predicate
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", [AUTO, "auto", None, 1])
def test_nothing_is_requested_by_the_defaults_or_by_one(value):
    """AUTO, None and an explicit 1 all mean "no chunking asked for"."""
    assert chunking_was_requested(value) is False


@pytest.mark.parametrize("value", [2, 4, 64])
def test_an_explicit_width_is_a_request(value):
    """An explicit K >= 2 is a request, and has something to be ignored."""
    assert chunking_was_requested(value) is True


def test_the_shipped_default_is_not_a_request():
    """Read the default off the signature rather than repeating it.

    Repeating ``AUTO`` here would make this test agree with itself if the
    default moved again — the exact failure it exists to prevent.
    """
    import inspect

    from tengri.inference.catalog_fitter import CatalogFitter

    default = inspect.signature(CatalogFitter.run).parameters["forward_chunk_size"].default
    assert chunking_was_requested(default) is False, (
        f"the shipped default {default!r} reads as an explicit chunking request, "
        "so every caller who passes nothing will be warned"
    )


# --------------------------------------------------------------------------
# 2. The behaviour — no warning on the default path
# --------------------------------------------------------------------------


def _chunk_warnings(method, **kwargs):
    """Run a dummy catalog fit and return only the forward_chunk_size warnings.

    The warning is emitted before any fitting work, so a dummy fitter reaches it
    and then fails on the absent data — which is fine and is what we swallow.
    """
    import jax

    from tengri.inference.catalog_fitter import CatalogFitter

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitter = CatalogFitter(None, [{}])
        with contextlib.suppress(Exception):
            fitter.run(method, key=jax.random.PRNGKey(0), **kwargs)
    return [str(w.message) for w in caught if "forward_chunk_size" in str(w.message)]


def test_the_default_path_does_not_warn():
    """`run("map")` with nothing passed must say nothing about chunking."""
    assert _chunk_warnings("map") == [], (
        "a caller who passed no forward_chunk_size was warned that theirs is "
        "being ignored — the guard is comparing against the old default"
    )


def test_an_explicit_request_still_warns():
    """Guard the guard: the warning must still fire when there IS a request.

    Without this, deleting the warning outright would pass the test above.
    """
    got = _chunk_warnings("map", forward_chunk_size=4)
    assert len(got) == 1, f"expected exactly one warning, got {got}"
    assert "is ignored" in got[0]


def test_an_explicit_one_does_not_warn():
    """K=1 asks for no chunking, so there is nothing to report as ignored."""
    assert _chunk_warnings("map", forward_chunk_size=1) == []


# --------------------------------------------------------------------------
# 3. The class guard — no site may compare a sentinel default to a literal
# --------------------------------------------------------------------------

#: Module-scope names bound to a sentinel default.
_SENTINELS = {"AUTO"}

#: ``(file, param)`` pairs allowed to compare against a numeric literal, each
#: with the reason. Empty by design: the census found exactly one site and it is
#: fixed. An entry here is a decision, not a formality.
_ALLOWED: set[tuple[str, str]] = set()


def _sentinel_defaulted_params(tree):
    """Parameter names whose default is a sentinel name or a bare string."""
    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        args = node.args
        pairs = list(zip(args.args[-len(args.defaults) :], args.defaults)) if args.defaults else []
        pairs += [(a, d) for a, d in zip(args.kwonlyargs, args.kw_defaults) if d is not None]
        for arg, default in pairs:
            if (isinstance(default, ast.Name) and default.id in _SENTINELS) or (
                isinstance(default, ast.Constant) and isinstance(default.value, str)
            ):
                found.add(arg.arg)
    return found


def test_no_sentinel_defaulted_parameter_is_compared_to_a_literal():
    """AST, not regex: the offending line is ordinary-looking code.

    A parameter that defaults to a sentinel cannot be meaningfully compared with
    a number — the sentinel is not one. Every such comparison is either dead or,
    as here, silently true for everyone who passed nothing.
    """
    hits = []
    for path in sorted(_SRC.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:  # pragma: no cover — not expected in src/
            continue
        params = _sentinel_defaulted_params(tree)
        if not params:
            continue
        rel = path.relative_to(_SRC).as_posix()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare) or not isinstance(node.left, ast.Name):
                continue
            name = node.left.id
            if name not in params or (rel, name) in _ALLOWED:
                continue
            for comp in node.comparators:
                if isinstance(comp, ast.Constant) and isinstance(comp.value, (int, float)):
                    hits.append(f"  {rel}:{node.lineno}: {name} compared to {comp.value!r}")

    assert not hits, (
        "a sentinel-defaulted parameter is compared against a numeric literal, "
        "which is true for every caller who passed nothing — ask a named "
        "predicate instead (see chunking_was_requested):\n" + "\n".join(hits)
    )
