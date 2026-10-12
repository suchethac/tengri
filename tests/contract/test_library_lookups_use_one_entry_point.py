# SPDX-License-Identifier: BSD-3-Clause
"""Contract: template-library lookups go through ``library_lookup`` (#2861).

Owner ruling: every library template lookup is node-exact PCHIP, read through
the single entry point :func:`tengri.utils.grid_interp.library_lookup`. The
triweight smoother does not reproduce stored nodes, so a direct import or call
of :func:`interp_nd_triweight` in ``src/tengri`` is a regression unless its file
is in :data:`ALLOWED` with a written reason.

The AST scan records every module that imports ``interp_nd_triweight`` (under
any alias) or calls it. The entries in :data:`ALLOWED` are either a public
re-export, the kernel's own definition, or a lookup site that has not been
switched yet. Those last ones are listed so that the remaining work is visible
here, and the list shrinks as each site is switched.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

pytestmark = pytest.mark.contract

_SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "tengri"
_NAME = "interp_nd_triweight"

#: file (relative to ``src/tengri``) -> reason it may import or call the kernel.
ALLOWED: dict[str, str] = {
    "utils/__init__.py": "public re-export of the smoother kernel; not a lookup site",
    "forward/precompute/__init__.py": "public re-export of the smoother kernel; not a lookup site",
    "components/nebular/shock.py": (
        "population-masked lookup (#2575): the B and density axes carry a population mask, "
        "and the masked triweight path is the only one that honors it. A masked PCHIP "
        "needs an owner ruling (open, #2861); not an opt-in"
    ),
}


def _kernel_sites(path: pathlib.Path) -> tuple[bool, list[int]]:
    """Return (imports the kernel, call line numbers) for one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    aliases: set[str] = set()
    imported = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == _NAME:
                    imported = True
                    aliases.add(alias.asname or alias.name)
    calls: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if (isinstance(fn, ast.Name) and fn.id in aliases) or (
            isinstance(fn, ast.Attribute) and fn.attr == _NAME
        ):
            calls.append(node.lineno)
    return imported, calls


def _scan() -> dict[str, tuple[bool, list[int]]]:
    found: dict[str, tuple[bool, list[int]]] = {}
    for path in sorted(_SRC.rglob("*.py")):
        imported, calls = _kernel_sites(path)
        if imported or calls:
            found[path.relative_to(_SRC).as_posix()] = (imported, calls)
    return found


def test_kernel_only_used_where_allowed():
    found = _scan()
    unexpected = sorted(rel for rel in found if rel not in ALLOWED)
    assert not unexpected, (
        "direct interp_nd_triweight use outside the allowlist; route the lookup through "
        f"library_lookup (#2861) or add a written reason to ALLOWED: {unexpected}"
    )


def test_switched_sites_no_longer_use_kernel():
    found = _scan()
    switched = (
        "components/dust/wg00.py",
        "components/nebular/agn_nebular.py",
        "components/nebular/cb19_precompute.py",
        "components/nebular/feltre_precompute.py",
        "components/agn/skirtor.py",
        "components/agn/skirtor_precompute.py",
        "components/agn/torus.py",
        "components/agn/disc.py",
        "components/agn/blocks/composable_precompute.py",
        "components/nebular/mappings_shock_precompute.py",
        "components/dust/dust_emission_precompute.py",
        "forward/precompute/templates.py",
        "components/_collapsed_lookup.py",
        "utils/grid_interp.py",
    )
    for rel in switched:
        assert rel not in found, (
            f"{rel} was switched to library_lookup and must stay off the kernel"
        )


def test_allowlist_has_no_stale_entries():
    found = _scan()
    stale = sorted(rel for rel in ALLOWED if rel not in found)
    assert not stale, f"ALLOWED lists modules that no longer use the kernel; remove them: {stale}"
