#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""A removed API must not survive where the sweep that removed it did not reach (#1776-#1781).

Removing a public spelling means rewriting every call site. That sweep is done by
hand, and its census is whatever the author happened to grep. When it misses a
tree nothing notices, because the tree is not imported by the test suite: the
stale calls sit there until someone runs the file.

That is not hypothetical. #1720 removed the ``stellar=`` build group, and its
commit message states "Every call site in the repo moved with it -- src, tests,
six notebooks, the gallery example". It touched six files under ``notebooks/``
and none under ``reproduction/``, which has six notebooks of its own. Five of
those six died on the removed spelling and stayed dead, because the only
PR-gated checks over ``reproduction/`` hash the *committed* PNGs -- artifacts of
a run that already happened, which keep matching perfectly while the source that
produced them no longer runs (#1776-#1781).

So the census belongs in a file rather than in one person's grep, and the roots
are the whole repo rather than the part someone remembered. Adding a rule when
you remove a public API is the cost of removing it.

Why the AST and not a regex
---------------------------
Every removed spelling appears legitimately in prose: the changelog names it, the
migration guide tabulates it, and the guard tests pass it on purpose to prove it
raises. A regex cannot tell ``stellar={...}`` in a call from the same characters
in a docstring, and a checker that cries wolf on its own documentation is one
that gets silenced. Matching call *syntax* instead of call *text* removes that
whole class: ``.md`` files are not scanned at all, and a docstring is not a call.
Notebooks are scanned through their code cells, which is where the rot was.
Markdown under ``docs/`` is scanned through its python fences only (#3358): a
published page's copy-paste block is code, and its prose is still exempt.

Usage
-----
    python tools/check_removed_apis.py
    python tools/check_removed_apis.py --root reproduction

Exit code 0 if no removed spellings are found; 1 with each one listed.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_ROOTS = (
    "src",
    "tests",
    "examples",
    "notebooks",
    "reproduction",
    "bench",
    "scripts",
    "tools",
    "docs",
)
#: Executable source only. Prose about a migration is not a call site, and the
#: changelog and migration guide must be free to name what they replaced.
SUFFIXES = (".py", ".ipynb")

#: Markdown is scanned only under ``docs/``, and only its fenced python blocks:
#: that is code a reader copies. Prose outside a fence is never a call site.
DOC_SUFFIX = ".md"
DOC_ROOT = "docs"
#: Repo-relative directories under ``docs/`` that are excluded from the fence scan.
#: ``archive``/``internal`` are excluded by EXCLUDE_PARTS; ``history`` is a frozen
#: snapshot of the docs as they were, so its fences name what was true then.
DOC_EXCLUDE_PREFIXES = ("docs/dev/history/",)
PY_FENCE = re.compile(r"^```(?:python|py)[ \t]*\n(.*?)^```", re.MULTILINE | re.DOTALL)

# Exact-match on path parts, so a renamed directory stops being excluded
# without any error. The notebook archives now share one ``archive`` parent, and
# ``docs/superpowers/`` became ``docs/internal/``.
EXCLUDE_PARTS = (
    "auto_examples",
    "_build",
    "archive",
    "internal",
)

#: Files that must call the removed spelling: they are the guards asserting it
#: still raises with its translation. Removing them to satisfy this checker
#: would delete the only proof the translation works.
EXCLUDE_FILES = frozenset(
    {
        "tools/check_removed_apis.py",
        "tests/contract/test_met_group_grammar.py",
        "tests/contract/test_build_routes_unknown_kwargs_to_grammar.py",
        "tests/regression/bug/test_catalog_met_table_advice.py",
    }
)

#: ``list_all()`` is the one lister that still returns a plain ``dict`` (of
#: ``_RegistryTable`` values), so ``.items()`` on it is correct, not the
#: pre-#1574 contract.
_DICT_RETURNING_LISTERS = frozenset({"list_all"})

_DICT_CONTRACT_ATTRS = frozenset({"items", "keys", "values"})


def _lister_name(node: ast.AST) -> str | None:
    """Return the ``list_*`` name if *node* is a call to one, else None."""
    if not isinstance(node, ast.Call):
        return None
    fn = node.func
    name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
    if not isinstance(name, str) or not name.startswith("list_"):
        return None
    if name in _DICT_RETURNING_LISTERS:
        return None
    return name


_SHOCK_BACKEND_REASON = (
    "the nebular `ShockBackend` route was removed in #3364 -- "
    "use `shock={...}`, which builds `ShockNebular`"
)
_REMOVED_SHOCK_NAMES = frozenset({"ShockBackend", "SHOCK_INLINE_PARAMS"})


_PRECOMPUTE_FOR_MODEL_REASON = (
    "`precompute_for_model` was removed -- "
    "use the registry module's `precompute` and `build_lookup`"
)


_TRANSLATE_MODULE = "tengri.parameters.translate"
_PARAM_MAP_REASON = (
    "`tengri.parameters.translate.PARAM_MAP` was removed in #3296 -- "
    "call `_build_param_map(...)` for the live table"
)


def scan_tree(tree: ast.AST):
    """Yield (lineno, reason) for each removed spelling used as real syntax."""
    yield from _scan_removed_names(tree)
    for node in ast.walk(tree):
        # #1720 -- the metallicity group is `met`; `stellar` is gone.
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg == "stellar":
                    yield node.lineno, "the `stellar=` group was removed in #1720 -- write `met=`"
                # The old group's structural key; every group selects with `type`.
                if isinstance(kw.value, ast.Dict):
                    for key in kw.value.keys:
                        if isinstance(key, ast.Constant) and key.value == "met_mode":
                            yield (
                                node.lineno,
                                "`met_mode` was removed in #1720 -- write `'type':`",
                            )

        # Dust-emission turnkey loader, deleted with its env knobs -- the registry module's
        # precompute / build_lookup are the live pair.
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "precompute_for_model" for alias in node.names
        ):
            yield node.lineno, _PRECOMPUTE_FOR_MODEL_REASON
        if isinstance(node, ast.Call):
            fn = node.func
            called = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
            if called == "precompute_for_model":
                yield node.lineno, _PRECOMPUTE_FOR_MODEL_REASON

        # #3364 -- the nebular shock backend route is gone; the shock group is `shock={...}`.
        if isinstance(node, ast.ImportFrom) and any(
            alias.name in _REMOVED_SHOCK_NAMES for alias in node.names
        ):
            yield node.lineno, _SHOCK_BACKEND_REASON
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if (
                    kw.arg == "backend"
                    and isinstance(kw.value, ast.Constant)
                    and kw.value.value == "shock"
                ):
                    yield node.lineno, _SHOCK_BACKEND_REASON

        # #1574 -- every list_* returns a _RegistryTable, not {label: callable}.
        if isinstance(node, ast.Subscript):
            name = _lister_name(node.value)
            if name and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                yield (
                    node.lineno,
                    f"`{name}()` returns a _RegistryTable since #1574 -- "
                    'string-index `.to_dict("fn")` instead',
                )
        # #3296 -- the module-level PARAM_MAP is gone; _build_param_map() is the table.
        if (
            isinstance(node, ast.ImportFrom)
            and node.module == _TRANSLATE_MODULE
            and any(alias.name == "PARAM_MAP" for alias in node.names)
        ):
            yield node.lineno, _PARAM_MAP_REASON
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "PARAM_MAP"
            and ast.unparse(node.value) == _TRANSLATE_MODULE
        ):
            yield node.lineno, _PARAM_MAP_REASON
        if isinstance(node, ast.Attribute) and node.attr in _DICT_CONTRACT_ATTRS:
            name = _lister_name(node.value)
            if name:
                yield (
                    node.lineno,
                    f"`{name}().{node.attr}()` is the pre-#1574 dict contract -- "
                    'call `.to_dict("fn")` first',
                )


#: Public names removed without a shim (#3358). ``from tengri import X`` for these
#: raises AttributeError. The replacements are the canonical names in
#: docs/dev/NAMING_CONTRACT.md.
_REMOVED_TENGRI_NAMES = {
    "Model": "`tengri.Model` was removed -- build with `SEDModel.build(...)`",
    "ParamSpec": "`tengri.ParamSpec` was removed -- use `tengri.Parameters`",
    "HierarchicalFitter": (
        "`tengri.HierarchicalFitter` was removed -- use `tengri.PopulationFitter`"
    ),
    "SpectroscopyConfig": "`tengri.SpectroscopyConfig` was removed -- use `tengri.Spectroscopy`",
    "effective_metallicity": (
        "`tengri.effective_metallicity` was removed in #3411 -- met_logzsol is the total "
        "metallicity on every grid; alpha enhancement needs a 4-D `_afe` grid"
    ),
}

_PRECOMPUTE_METHOD_REASON = (
    "`SEDModel.precompute_spectroscopy` was removed -- the spectrum table is "
    "built at build time: pass `approx=SpectrumPrecomp()` to `SEDModel.build`"
)

#: The Gyr timing prior is a Fixed-only alias since #2748; the wording mirrors
#: ``check_free_gyr_alias`` in src/tengri/parameters/timing.py.
#: Spellings of the Gyr timing prior: the flat name, and the ``sfh`` group short name.
_GYR_PRIOR_KEYS = frozenset({"sfh_tsnorm_peak_lbt_gyr", "peak_lbt_gyr"})


def _gyr_prior_reason(name: str) -> str:
    """The #2748 Fixed-only-alias message for a Gyr timing spelling ``name``."""
    frac = name.replace("gyr", "frac")
    return (
        f"`{name}` is now a Fixed-only alias: free the fraction `{frac}` instead "
        "(age = frac * age_of_universe(z)); convert the Gyr bounds by dividing by the "
        "age of the universe at the source redshift (#2748)"
    )


#: The free-prior constructors of tengri.parameters.priors (``Fixed`` is pinned).
_FREE_PRIOR_CALLS = frozenset(
    {"Uniform", "Gaussian", "LogUniform", "LogNormal", "StudentT", "Laplace"}
)


def _is_free_gyr_prior(node: ast.AST) -> bool:
    """True for a free prior: a prior constructor other than ``Fixed``, or ``FREE``.

    A bare number, ``float(...)``, ``jnp.asarray(...)`` or a ``ParamDef(...)`` is not a
    prior and is not flagged.
    """
    if isinstance(node, ast.Call):
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        return name in _FREE_PRIOR_CALLS
    if isinstance(node, ast.Name):
        return node.id == "FREE"
    if isinstance(node, ast.Attribute):
        return node.attr == "FREE"
    return False


def _sps_precompute_names(tree: ast.AST) -> frozenset[str]:
    """Names bound to the ``sps.precompute`` module, whose function is still valid."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith(".sps"):
            names.update(a.asname or a.name for a in node.names if a.name == "precompute")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.endswith(".sps.precompute"):
                    names.add(alias.asname or alias.name.split(".")[0])
    return frozenset(names)


def _scan_removed_names(tree: ast.AST):
    """Yield (lineno, reason) for the names removed in #3358."""
    module_names = _sps_precompute_names(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "tengri" and node.level == 0:
            for alias in node.names:
                if alias.name in _REMOVED_TENGRI_NAMES:
                    yield node.lineno, _REMOVED_TENGRI_NAMES[alias.name]
        if isinstance(node, ast.Call):
            fn = node.func
            if (
                isinstance(fn, ast.Attribute)
                and fn.attr == "precompute_spectroscopy"
                and not (isinstance(fn.value, ast.Name) and fn.value.id in module_names)
            ):
                yield node.lineno, _PRECOMPUTE_METHOD_REASON
            for kw in node.keywords:
                # The Gyr spelling as a keyword of any call (Parameters, build, dict).
                if kw.arg in _GYR_PRIOR_KEYS and _is_free_gyr_prior(kw.value):
                    yield node.lineno, _gyr_prior_reason(kw.arg)
        # A dict literal keyed by the Gyr spelling: the ``sfh={...}`` group form, or any
        # dict handed to SEDModel.build / Parameters. The key is specific enough that
        # every literal carrying it is flagged, whichever call it reaches.
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value in _GYR_PRIOR_KEYS
                    and _is_free_gyr_prior(value)
                ):
                    yield node.lineno, _gyr_prior_reason(key.value)


_MAGIC = re.compile(r"^\s*[%!]")


def _parse(source: str) -> ast.AST | None:
    """Parse Python, tolerating IPython magics. None if it still will not parse."""
    cleaned = "\n".join("" if _MAGIC.match(line) else line for line in source.splitlines())
    try:
        # Docs fences hold prose-escaped strings; a bad escape is not a removed call.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            return ast.parse(cleaned)
    except SyntaxError:
        return None


def scan_file(path: Path):
    """Yield (where, reason) for one .py, .ipynb, or docs .md file."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix == DOC_SUFFIX:
        for match in PY_FENCE.finditer(text):
            tree = _parse(match.group(1))
            if tree is None:
                continue
            line = text.count("\n", 0, match.start()) + 1
            for _lineno, reason in scan_tree(tree):
                yield f"fence at line {line}", reason
        return
    if path.suffix == ".py":
        tree = _parse(text)
        if tree is not None:
            yield from scan_tree(tree)
        return

    try:
        nb = json.loads(text)
    except json.JSONDecodeError:
        return
    for n, cell in enumerate(nb.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        src = cell.get("source") or ""
        joined = "".join(src) if isinstance(src, list) else src
        tree = _parse(joined)
        if tree is None:
            continue
        for _lineno, reason in scan_tree(tree):
            # Notebook line numbers are per-cell; the cell is the useful locator.
            yield f"cell {n}", reason


def _is_scanned(path: Path, rel: str) -> bool:
    """Executable source anywhere, or markdown under ``docs/``."""
    if path.suffix in SUFFIXES:
        return True
    return path.suffix == DOC_SUFFIX and rel.startswith(f"{DOC_ROOT}/")


def iter_files(roots):
    for root in roots:
        root_path = REPO_ROOT / root
        if not root_path.exists():
            continue
        candidates = [root_path] if root_path.is_file() else sorted(root_path.rglob("*"))
        for path in candidates:
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if not _is_scanned(path, rel):
                continue
            if any(part in EXCLUDE_PARTS for part in path.parts):
                continue
            if rel in EXCLUDE_FILES or rel.startswith(DOC_EXCLUDE_PREFIXES):
                continue
            yield path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="Repository-relative path to scan (repeatable). "
        f"Defaults: {', '.join(DEFAULT_ROOTS)}.",
    )
    args = parser.parse_args(argv)

    violations = []
    for path in iter_files(args.roots or list(DEFAULT_ROOTS)):
        for where, reason in scan_file(path):
            violations.append((path.relative_to(REPO_ROOT), where, reason))

    if not violations:
        print("OK: no removed API spellings in executable source")
        return 0

    print(f"FAIL: {len(violations)} removed API call site(s)\n")
    for path, where, reason in violations:
        print(f"  {path}:{where}  {reason}")
    print(
        "\nFix: rewrite the call site with the replacement named above.\n"
        "A notebook counts -- reproduction/*.ipynb and its docs/ copy are committed\n"
        "source, and nothing else in CI executes them.\n"
        "If the file must call the removed spelling to prove it raises, add it to\n"
        "EXCLUDE_FILES with a justification."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
