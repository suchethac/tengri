# SPDX-License-Identifier: BSD-3-Clause
"""Guard: no unauthorized raw read of an erg/s ``state.derived`` key (#1837).

Four contract keys are published in erg/s alongside an exact ``log10``
companion. The linear form is ``inf`` in float32 for an ordinary galaxy
(``L_ir``/``L_absorbed`` ~3.6e43, ``L_age`` ~3.3e42 per bin, ``line_lums``
~1e41-1e43); the companion is not. Consumers were migrated to the companions
piecemeal, and the ones left behind returned ``inf``/``nan`` for twenty
derived properties whose answers are of order 1 to 1e10 and sit comfortably
inside float32 range.

The defect was never a missing capability -- ``log_L_ir``, ``log_L_age`` and
``log_line_lums`` were already published, already finite, and already the
preferred read at other sites in the same tree. What was missing is this
guard: something that notices when a *new* consumer reaches for the linear key
next to a neighbor that does not.

Two-way gate, matching :mod:`tests.regression.precision.test_no_raw_nion_read`:
new unauthorized sites are rejected, and stale allow-list entries are flagged
so a later migration cannot leave a lie behind.

Reads are located with :mod:`ast`, not a regex, so the many prose mentions of
``state.derived["L_ir"]`` in docstrings and comments are not counted. Only
executable reads are.

See #1837; parent epic #1206.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.contract

SRC = pathlib.Path("src/tengri")

#: Contract keys published in erg/s that have an exact log10 companion.
ERG_S_KEYS = frozenset({"L_ir", "L_absorbed", "L_age", "line_lums"})

#: Authorized raw readers, each with the reason it may stay.
#:
#: Two legitimate categories, and nothing else:
#:   * the *fallback* arm of a consumer that already prefers the companion;
#:   * a producer of an **absolute** erg/s quantity, which is genuinely outside
#:     float32 range and needs the breaking unit change tracked in #1206 §3 --
#:     not something this guard can fix by swapping a read.
ALLOW = {
    "components/agn/component.py": (
        "fallback arm only -- reads log_L_ir first and drops here when a chain "
        "publishes no companion"
    ),
    "components/nebular/component.py": (
        "_line_lums_for_ratios fallback arm, plus _line_luminosity_helper which "
        "serves the 11 absolute erg/s line properties (#1206 §3)"
    ),
    "forward/component_factory.py": (
        "state_to_line_quantities builds the absolute erg/s line catalog (#1206 §3)"
    ),
    "forward/prediction.py": "pred.lines absolute erg/s catalog (#1206 §3)",
    "forward/sed_model.py": "fast-line path reconstructs the absolute erg/s catalog (#1206 §3)",
    "components/radio/component.py": (
        "packs L_ir into the RadioModel inputs dict beside log_L_ir, which the "
        "model prefers -- see radio_model.py"
    ),
}

# ``utils/sed_quantities.py`` is deliberately absent: derived_luminosity_lsun
# and derived_weights_peak_relative take the key *name* as an argument, so they
# never name a banned key as a literal and the census does not see them. That
# is the correct outcome -- they are the companion-preferring readers every
# other site should be routed through.


def _raw_reads(tree: ast.AST) -> list[tuple[int, str]]:
    """Executable reads of a banned key off a ``.derived`` mapping.

    Catches both ``derived["L_ir"]`` and ``derived.get("L_ir", ...)``, with or
    without a ``state.`` prefix. Docstrings and comments are invisible to
    ``ast``, which is the reason for using it here.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            key = node.slice
            target = node.value
            name = getattr(target, "attr", getattr(target, "id", None))
            if name == "derived" and isinstance(key, ast.Constant) and key.value in ERG_S_KEYS:
                hits.append((node.lineno, str(key.value)))
        elif isinstance(node, ast.Call):
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and getattr(func.value, "attr", getattr(func.value, "id", None)) == "derived"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value in ERG_S_KEYS
            ):
                hits.append((node.lineno, str(node.args[0].value)))
    return hits


def _census() -> dict[str, list[tuple[int, str]]]:
    found: dict[str, list[tuple[int, str]]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text())
        hits = _raw_reads(tree)
        if hits:
            found[path.relative_to(SRC).as_posix()] = hits
    return found


def test_no_unauthorized_raw_erg_s_read():
    """A new consumer must read the log companion, or justify itself here."""
    found = _census()
    offenders = {
        rel: [f"line {ln}: derived[{key!r}]" for ln, key in hits]
        for rel, hits in found.items()
        if rel not in ALLOW
    }
    assert not offenders, (
        "raw erg/s state.derived reads outside the allow-list. These keys are "
        "inf in float32; read the log companion (log_L_ir / log_L_age / "
        "log_line_lums) via derived_luminosity_lsun or "
        "derived_weights_peak_relative, or add an entry to ALLOW with the "
        f"reason it must stay (#1837): {offenders}"
    )


def test_allow_list_has_no_stale_entries():
    """A migration that removed the last raw read must remove its entry too."""
    stale = set(ALLOW) - set(_census())
    assert not stale, (
        f"allow-list entries whose files no longer read a raw erg/s key: {stale}. "
        "Remove them, so the list keeps naming only real exceptions."
    )


def test_the_guard_can_actually_fail():
    """Non-vacuity: the detector must fire on a read it is meant to catch.

    A census that silently matched nothing would make both gates above pass
    forever. This pins the detector itself rather than its current verdict.
    """
    caught = _raw_reads(ast.parse('x = jnp.asarray(state.derived["L_ir"])'))
    assert caught == [(1, "L_ir")]
    caught_get = _raw_reads(ast.parse('x = derived.get("L_age", 0.0)'))
    assert caught_get == [(1, "L_age")]
    # A log companion read is not an offense.
    assert _raw_reads(ast.parse('x = state.derived["log_L_ir"]')) == []
    # Nor is the same text inside a docstring -- the reason for using ast.
    assert _raw_reads(ast.parse('"""Reads state.derived["L_ir"] in erg/s."""')) == []
