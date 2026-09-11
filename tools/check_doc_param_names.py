#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""CI guard: every backticked dust_*/agn_*/neb_*/sfh_* name in docs/model_reference/ is real.

#2234 added three new grammar/flat keys (``nebular_screen``,
``shock_screen``, ``agn_screen``) and, while regenerating the affected rows in
``docs/model_reference/parameters.md``, found several pre-existing identifiers
that had drifted from the live parameter registry (retired short spellings,
a #369-era rename left behind, a case mismatch) with nothing to catch it.
This guard is that catch, going forward.

A backticked identifier under one of the four physics prefixes is real if it
is either:

1. A fittable parameter, declared either as a ``ParamDeclaration("name", ...)``
   call (the primitive :func:`tengri.parameters.registry.registry` is built
   from -- every ``declared_parameters()`` entry is isinstance-checked
   against it) or as a ``"name": ParamDef(...)`` dict entry (the SFH-model
   registry's own declaration primitive, converted to ``ParamDeclaration``
   at runtime by :meth:`StellarSEDComponent.declared_parameters`). Both are
   read here by a plain AST parse, so this guard runs with no ``tengri``
   import, in the same bare-ruff lint venv as ``check_dust_law_kwargs.py`` /
   ``check_dust_group_grammar.py``.
2. A structural (non-fittable) config key or group name --
   :data:`STRUCTURAL_KEYS` below -- categorical selectors like
   ``dust_model`` / ``dust_nebular_screen`` that never enter the parameter
   registry by design (only ``ParamDeclaration``/``ParamDef`` populate it).
3. A documented pre-existing exception -- :data:`KNOWN_STALE` below -- for
   debt this guard did not create and is not the place to fix.

Known limitation: a parameter name built at runtime from an f-string or a
loop (``f"sfh_burstcont_ratio_{i}"``, six of them) is invisible to the plain
``ast.Constant`` scan below and so is not in the resolved set. None of these
currently appear in ``docs/model_reference/``; if one is added and flagged
here, extend :data:`STRUCTURAL_KEYS` (with a comment) rather than the AST
scan, mirroring how :data:`KNOWN_STALE` already documents its exceptions.

Usage
-----
    python tools/check_doc_param_names.py

Exit code 0 if every name resolves; 1 with the violations listed otherwise.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "tengri"
DOC_DIR = ROOT / "docs" / "model_reference"

#: Structural (non-fittable) config keys and group names: categorical
#: selectors passed to the grammar (``dust_attenuation={'type': ...}``) or the
#: flat ``Parameters(...)`` surface. These never enter the parameter registry
#: -- only ``ParamDeclaration`` (fittable priors) does -- so they are listed
#: by hand. A group name (``dust_attenuation``) is included too: a backticked
#: group name matches the same regex as a parameter name.
STRUCTURAL_KEYS: frozenset[str] = frozenset(
    {
        "dust_attenuation",
        "dust_emission",
        "dust_model",
        "dust_approx",
        "dust_law_bc",
        "dust_law_diff",
        "dust_law_neb",
        "dust_law_overrides",
        # #2234 (a replacement): the per-source dust-screen selector triad.
        # Flat spelling (dust_*_screen) and bare grammar spelling (agn_screen
        # is the only one of the three that also matches the doc-scan regex,
        # since only "agn_" is itself one of the four scanned prefixes).
        "dust_nebular_screen",
        "dust_shock_screen",
        "dust_agn_screen",
        "agn_screen",
        "dust_wg00_curve",
        "dust_wg00_geometry",
        "dust_wg00_structure",
        "dust_lyman_cutoff_aa",
        "dust_lyc_absorb_all",
        "dust_eb_include_lyc",
        "dust_emission_model",
        "agn_model",
        "agn_disc_block",
        "agn_nlr_block",
        "agn_blr_block",
        "agn_torus_block",
        "agn_feii_block",
        "agn_attenuation_block",
        "agn_norm",
        "sfh_model",
        "sfh_bin_edges_gyr",
    }
)

#: Pre-existing identifiers in docs/model_reference/ that predate this guard
#: (#2234) and are out of scope for it: each is either a retired
#: spelling from an earlier rename (the current registry name means the same
#: quantity under a different, not-merely-abbreviated, name) or names a
#: "planned" (not-yet-implemented, per dust.md's own MAGPHYS section) future
#: feature. Fixing the surrounding prose accurately is a separate audit; each
#: entry names where it lives and why it is not fixed here, mirroring the
#: ``ALLOWLIST`` pattern in ``check_param_defaults.py``.
KNOWN_STALE: dict[str, str] = {
    "agn_frac": "computation.md/nebular.md/parameters.md: retired, no direct current replacement",
    "agn_logedd": "nebular.md: retired short spelling of agn_log_ledd",
    "agn_logmbh": "nebular.md: retired short spelling of agn_log_mbh",
    "neb_logn": "nebular.md: retired, no current gas-density parameter under this name",
    "neb_logu": "nebular.md: case mismatch with the real neb_logU",
    "neb_logz_gas": "nebular.md: case mismatch with the real neb_logZ_gas",
    "neb_gas_logu": "parameters.md: retired 'gas_' spelling",
    "neb_gas_logz": "parameters.md: retired 'gas_' spelling",
    "sfh_burst_frac": (
        "parameters.md: retired; current name is sfh_burst_log_fburst (log, not linear)"
    ),
    "sfh_burst_age_myr": "parameters.md: retired; current name is sfh_burst_log_tpeak_myr",
    "sfh_burst_width_dex": "parameters.md: retired, no current width parameter under this name",
    "sfh_tsnorm_log_peak_sfr": "parameters.md: #369 renamed this to sfh_tsnorm_log_total_mass",
    "dust_xi_pah": "parameters.md: MAGPHYS is 'planned' (dust.md), not yet implemented",
    "dust_xi_mir": "parameters.md: MAGPHYS is 'planned' (dust.md), not yet implemented",
    "dust_xi_warm": "parameters.md: MAGPHYS is 'planned' (dust.md), not yet implemented",
}

_IDENT_RE = re.compile(r"`((?:dust|agn|neb|sfh)_[A-Za-z0-9_]*)`")


#: Call-target names that mark a parameter declaration. ``ParamDeclaration``
#: is the general primitive (``ParamDeclaration("name", prior, "desc")``);
#: ``ParamDef`` is the SFH-model registry's own (``"name": ParamDef(...)``
#: inside a dict literal), converted to ``ParamDeclaration`` at runtime.
_DECL_CALL_NAMES = frozenset({"ParamDeclaration", "ParamDef"})


def _call_func_name(node: ast.Call) -> str | None:
    func = node.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


def _declared_parameter_names() -> set[str]:
    """Every parameter name declared under src/tengri, by either primitive.

    Two shapes, both a plain AST walk with no ``tengri`` import:

    * ``ParamDeclaration("name", ...)`` -- the name is the call's first
      positional argument.
    * ``{"name": ParamDef(...), ...}`` -- the name is a dict key whose paired
      value is a ``ParamDef(...)`` call (the SFH-model registry's shape).
    """
    names: set[str] = set()
    for path in SRC.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if not any(decl in text for decl in _DECL_CALL_NAMES):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func_name = _call_func_name(node)
                if func_name == "ParamDeclaration" and node.args:
                    first = node.args[0]
                    if isinstance(first, ast.Constant) and isinstance(first.value, str):
                        names.add(first.value)
            elif isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=False):
                    if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                        continue
                    if isinstance(value, ast.Call) and _call_func_name(value) == "ParamDef":
                        names.add(key.value)
    return names


def _doc_identifiers() -> dict[str, list[tuple[Path, int]]]:
    """Every backticked dust_*/agn_*/neb_*/sfh_* identifier, with its sites."""
    sites: dict[str, list[tuple[Path, int]]] = {}
    for path in sorted(DOC_DIR.glob("*.md")):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for m in _IDENT_RE.finditer(line):
                ident = m.group(1)
                if ident.endswith("_"):
                    # A wildcard prefix, e.g. `sfh_tsnorm_burst_` naming a family
                    # of keys rather than one; nothing to resolve.
                    continue
                sites.setdefault(ident, []).append((path, lineno))
    return sites


def main() -> int:
    known = _declared_parameter_names() | STRUCTURAL_KEYS | set(KNOWN_STALE)
    sites = _doc_identifiers()
    violations = {ident: locs for ident, locs in sites.items() if ident not in known}

    if not violations:
        print(
            f"check_doc_param_names: OK -- every backticked dust_*/agn_*/neb_*/sfh_* "
            f"identifier in docs/model_reference/ resolves ({len(known)} known names)."
        )
        return 0

    print(f"check_doc_param_names: {len(violations)} unresolved identifier(s)\n")
    for ident, locs in sorted(violations.items()):
        where = ", ".join(f"{p.relative_to(ROOT)}:{n}" for p, n in locs)
        print(f"  `{ident}`  ({where})")
    print(
        "\nEach name must be a real ParamDeclaration under src/tengri, a structural "
        "config key added to STRUCTURAL_KEYS, or a documented pre-existing exception "
        "added to KNOWN_STALE (with a reason)."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
