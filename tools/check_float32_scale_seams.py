#!/usr/bin/env python3
"""Enumerate the float32 scale seams by seam, not by "a representative model" (#2178).

A **scale seam** is a site where a large physical constant or unit conversion
multiplies (or divides) a *parameter-derived* quantity. Four separate bugs have
now come out of the same shape:

===========  =========================================================
 #1388       ``apply_log10_scale`` gradient-unsafe above ~1e38
 #1439       ``multicolor_disc``'s bolometric renorm: ~1e64 (``inf``)
             against a ~1e-64 partner (flushes to 0) -> ``inf*0 = nan``
 #2100       ``_mass_scale_lnu``'s REVERSE pass materializes
             ``total_mass * L_sun`` ~3.8e43 as a standalone Jacobian
 #2178       the SAME product from the FORWARD pass, exposed by jaxlib
             0.11.1 emitting a different kernel for byte-identical HLO
===========  =========================================================

Every one was silent: the forward pass is finite and only the gradient (or, in
#2178, a differently-fused forward) is wrong. Fixing #2100's pass did not fix
#2178's, and nothing said the other was exposed. The fixes were per-site --
``stop_gradient`` at eight sites, ``optimization_barrier`` at two -- which is a
strategy that closes the seam somebody tripped and says nothing about the rest.

**The binding rule is #1436's, verbatim:**

    A float32 result established on one model configuration says nothing about a
    configuration with a different scale seam ... coverage has to be enumerated
    by seam, not by "a representative model".

So this tool enumerates the seams themselves and asks, per seam, how large the
product can get **inside that parameter's own declared prior**. A seam whose
product can leave float32's range within its declared prior is a defect waiting
to happen, whether or not anyone has tripped it yet.

**The prior is read from the registry, never copied.** ``registry()`` walks the
per-component ``_params.py`` declarations, so the range this tool measures
against is the range the parameter actually declares. Commit 45741f4cd is why
that distinction is load-bearing: a *grid axis* range copied into a test read as
a prior, and the two shared no point at all.

**AST, not regex.** The scan is over parsed syntax and *evaluated* constants;
this project replaced source-text assertions with measurements in 5d08a293e
(closes #2108) and a grep for ``L_SUN *`` would be exactly the thing that
replaced.

What the tool cannot decide, and does not try to
------------------------------------------------
Whether a seam is *safe* is not visible in the multiplication. ``L_sun``
multiplied into an SSP operand that is itself ~1e-15 never leaves range; the
same constant multiplied into a bare scalar does. The tool therefore reports the
**standalone worst case** -- the magnitude the product reaches if the large
constant ever meets the parameter with no small partner to hide behind -- and
requires every such seam to carry a recorded reason in ``_HANDLED`` saying which
grouping keeps it safe. That is the ratchet: a new seam over float32's range is
an error, an existing one is a documented decision, and a ``_HANDLED`` entry
whose seam no longer exists (or is no longer over range) is stale and also an
error, so the inventory cannot rot.

Run with ``--list`` to print the whole inventory with its reach per seam.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import math
import pathlib
import re
import sys
import warnings
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np

#: A constant is "large" at this many decades from unity. Well below the ~33.6
#: of ``L_sun`` and the ~18.5 of ``c`` in A/s, so the scan is not tuned to the
#: constants that have already bitten; ``AA_TO_CM`` (1e-8) and the O(1)
#: constants stay out, which is what keeps the inventory readable.
_BIG_DEX = 10.0

#: Shortest unqualified spelling accepted as a reference to a registry
#: parameter. See ``_scan`` for why a bare ``sfr`` is not one.
_MIN_SHORT_FORM = 5

#: float32's range. ``finfo`` rather than literals: the numbers are the dtype's,
#: not this file's opinion of them.
_F32_MAX = float(np.finfo(np.float32).max)
_F32_MIN_NORMAL = float(np.finfo(np.float32).tiny)

#: Source-local names that ARE a parameter, under a different spelling, together
#: with the registry pattern they come from and how the source gets from the
#: declared value to the runtime one. Nothing here invents a range: the range is
#: still read from whichever registry entries the pattern matches, and the
#: widest is taken. Kept small on purpose -- a local alias is a claim about the
#: code, so each one carries the line that establishes it.
_ALIASES: dict[str, tuple[str, str]] = {
    # ``total_mass = 10**log_total_mass`` is the SFH mass contract
    # (components/stellar/component.py, ``_renormalize_to_mass``).
    "total_mass": (r"^sfh_.*_log_total_mass$", "pow10"),
    # Surviving mass, bounded above by the formed mass of the same contract, so
    # the formed-mass prior is the right ceiling for it too.
    "stellar_mass": (r"^sfh_.*_log_total_mass$", "pow10"),
    # ``adaf_spectrum`` re-binds ``agn_log_lbol`` under a local name
    # (components/agn/adaf.py).
    "_lbol_shape": (r"^agn_log_lbol$", "pow10"),
}


class Seam(NamedTuple):
    """One ``constant x parameter`` site, with the reach its own prior allows."""

    module: str
    where: str  # dotted qualname of the enclosing def/class -- stable under edits
    lineno: int
    constant: str
    value: float
    param: str  # registry parameter (or alias pattern) that sets the reach
    transform: str  # "pow10" or "identity"
    param_max: float  # largest |parameter value| the declared prior admits
    reach: float  # |constant| * param_max -- the standalone worst case

    @property
    def key(self) -> str:
        """Module + enclosing qualname.

        Deliberately NOT the line number: a registration keyed on a line is
        invalidated by every edit above it, and an inventory that churns is one
        nobody reads. Two seams in the same function that share a constant and a
        parameter are the same seam wearing two line numbers.
        """
        return f"{self.module}:{self.where}"

    @property
    def over_range(self) -> bool:
        """True when the standalone product leaves float32's representable range."""
        return self.reach > _F32_MAX or (self.reach != 0.0 and self.reach < _F32_MIN_NORMAL)


#: Seams whose standalone product leaves float32 range, grouped by the PRODUCT
#: rather than by the call site. Thirty-eight of the forty-six below are the one
#: expression ``L_sun * 10**agn_log_lbol`` written in thirty-eight AGN blocks;
#: listing them as thirty-eight independent decisions would be thirty-eight
#: chances to write a reason nobody checks. Each family carries one reason and
#: the sites it covers, and ``tests/regression/precision/
#: test_float32_scale_seam_sweep.py`` requires every family here to have a
#: float32 sweep across its parameter's declared prior -- so the reason is
#: backed by a measurement rather than standing as the only evidence.
_HANDLED: dict[str, tuple[str, tuple[str, ...]]] = {
    "stellar_mass_scale": (
        "#2100 / #2178. ``total_mass * L_sun`` ~1.2e46 at the top of the declared "
        "log_total_mass prior. The grouping is stated in the graph rather than left "
        "to the backend: ``_mass_scale_lnu`` returns "
        "``optimization_barrier(total_mass * per_msun_lsun) * L_sun`` and its "
        "``custom_jvp`` carries the same barrier on ``primal_out`` and on both "
        "tangent terms, so neither pass ever forms the scalar alone. The two "
        "``sed_model`` sites and ``StellarSEDComponent.apply`` compute the scale at "
        "working precision on the derived/line path, outside any fitted gradient.",
        (
            "tengri.components.stellar.component:_mass_scale_lnu",
            "tengri.components.stellar.component:_mass_scale_lnu_jvp",
            "tengri.components.stellar.component:StellarSEDComponent.apply",
            "tengri.forward.sed_model:SEDModel._feature_fast_indices",
            "tengri.forward.sed_model:SEDModel.measure_line_fluxes",
        ),
    ),
    "agn_bolometric_renorm": (
        "#1439. ``L_sun * 10**agn_log_lbol`` reaches 3.8e47 at the top of the "
        "declared prior and is over float32's range everywhere above log_lbol ~ 4.9 "
        "-- i.e. across the whole prior, not at an edge. This is the seam #1439 was "
        "first read as an unreachable cancellation and turned out to be a grouping "
        "bug: the inner product went to ~1e64 (inf) against a ~1e-64 partner that "
        "flushed to zero, so ``inf * 0`` was nan. The renormalization now divides "
        "the shape integral out before the constant meets the SED array, and the "
        "float32 path carries the scale in log space "
        "(``utils.scale.apply_log10_scale``, #1388).",
        (
            "tengri.components.agn._phys:compute_l_12um_from_lbol",
            "tengri.components.agn._template_grid:torus_lnu_from_grid",
            "tengri.components.agn.adaf:adaf_spectrum",
            "tengri.components.agn.blocks.blr:blr_synthesizer_spectra_block",
            "tengri.components.agn.blocks.disc:_cigale_disc_lambda",
            "tengri.components.agn.blocks.grahsp_blocks:grahsp_sbpl_disc_block",
            "tengri.components.agn.blocks.nlr:nlr_analytic_block",
            "tengri.components.agn.blocks.nlr:nlr_cue_block",
            "tengri.components.agn.blocks.nlr:nlr_feltre_block",
            "tengri.components.agn.blocks.nlr:nlr_synthesizer_block",
            "tengri.components.agn.blocks.nlr:nlr_synthesizer_spectra_block",
            "tengri.components.agn.blocks.runner:compose_l_nu",
            "tengri.components.agn.blocks.torus:skirtor_torus_block",
            "tengri.components.agn.disc:_compute_bh_params",
            "tengri.components.agn.disc:_compute_zone_radii",
            "tengri.components.agn.disc:kubota_done_disc",
            "tengri.components.agn.disc:multicolor_disc",
            "tengri.components.agn.disc:powerlaw_disc",
            "tengri.components.agn.disc:relagn_disc_from_grid",
            "tengri.components.agn.fritz:create_fritz_components_from_grid.fritz_components",
            "tengri.components.agn.fritz:create_fritz_from_grid.fritz_grid",
            "tengri.components.agn.fritz:fritz_sed_from_grid",
            "tengri.components.agn.grahsp.model:compute_grahsp_sed",
            "tengri.components.agn.kd_precompute:_compute_bh_and_radii",
            "tengri.components.agn.kd_precompute:kubota_done_disc_preintegrated",
            "tengri.components.agn.qsogen:_qsogen_components",
            "tengri.components.agn.richards2006_disc:richards2006_disc",
            "tengri.components.agn.silva04:silva04_sed_from_grid",
            "tengri.components.agn.skirtor:_skirtor_grid_sed",
            "tengri.components.agn.skirtor:create_skirtor_components_from_grid.skirtor_components",
            "tengri.components.agn.skirtor:create_skirtor_raw_total_from_grid.fn",
            "tengri.components.agn.skirtor_agnfitter_precompute:build_lookup."
            "skirtor_agnfitter_phot_collapsed",
            "tengri.components.agn.skirtor_agnfitter_precompute:"
            "build_skirtor_agnfitter_photometry_lookup.skirtor_agnfitter_photometry",
            "tengri.components.agn.slone_netzer:slone_netzer_sed_from_grid",
            "tengri.components.agn.torus:create_nenkova_from_grid.nenkova_grid",
            "tengri.components.agn.torus:simple_torus",
            "tengri.components.agn.torus:two_temperature_torus",
            "tengri.components.agn.unified:unified_nlr_blr",
        ),
    ),
    "xrb_mass_scale": (
        "#722. The Lehmer+2016 LMXB normalization ``9.05e28 * stellar_mass`` "
        "reaches 2.9e41 for a 3e12 Msun galaxy at the top of the declared mass "
        "prior. It is a derived quantity read at working precision, and its float32 "
        "path is the log-domain companion ``log_l_x_xrb``.",
        ("tengri.utils.sed_quantities:compute_l_x_xrb",),
    ),
}

#: Over-range seams that are **not** handled: the enumeration found them, a
#: measurement confirmed them, and they are filed rather than fixed here.
#: Separated from ``_HANDLED`` on purpose -- an inventory that files a live
#: defect under "handled" is worse than no inventory. Every entry must name the
#: issue it is filed under, and the sweep module asserts that the arithmetic
#: claim (the product leaves float32's range inside the declared prior) still
#: holds, so an entry cannot sit here after it stops being true.
_OPEN_DEFECTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "agn_black_hole_mass": (
        "#2210. ``M_sun * 10**agn_log_mbh`` is 1.99e39 at the BOTTOM of the "
        "declared Uniform(6, 10) prior -- past float32's 3.403e38 across the "
        "entire range, with no in-range corner. Measured: the float32 forward "
        "of a kubota_done disc is ``nan`` at ``agn_log_mbh = 6`` under jaxlib "
        "0.11.1 and finite under 0.11.0, the same graph-versus-kernel split as "
        "#2178. ``_gravitational_radius`` is a regrouping away from safe; "
        "``_eddington_luminosity`` is not (L_Edd itself is ~1.26e44 erg/s at "
        "the bottom of the prior) and needs the log-domain treatment the other "
        "bolometric seams got. Filed, not fixed here: hand-fixing a second "
        "component inside the PR that builds this enumeration is the per-site "
        "habit the enumeration exists to replace.",
        (
            "tengri.components.agn.disc:_eddington_luminosity",
            "tengri.components.agn.disc:_gravitational_radius",
        ),
    ),
}

#: ``seam key -> family``. Derived, never written twice.
_FAMILY_OF: dict[str, str] = {
    key: family
    for bucket in (_HANDLED, _OPEN_DEFECTS)
    for family, (_, keys) in bucket.items()
    for key in keys
}


_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SRC_ROOT = _REPO_ROOT / "src"


def _module_name(path: pathlib.Path, root: pathlib.Path) -> str:
    rel = path.relative_to(root).with_suffix("")
    parts = [p for p in rel.parts]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _qualnames(tree: ast.AST) -> dict[int, str]:
    """``id(node) -> dotted qualname of the enclosing def/class`` for every node."""
    out: dict[int, str] = {}

    def walk(node: ast.AST, scope: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                inner = f"{scope}.{child.name}" if scope else child.name
                out[id(child)] = inner
                walk(child, inner)
            else:
                out[id(child)] = scope
                walk(child, scope)

    walk(tree, "")
    return out


def _constant_table(tree: ast.AST, module_name: str) -> dict[str, float]:
    """``name -> float`` for every float this module can see under that name.

    Three sources, in the order the interpreter would resolve them: names
    imported from another ``tengri`` module (including function-scope imports,
    which is how most of these constants actually arrive), module-level float
    literals, and finally whatever the imported module itself binds. The last
    one is the reason this is an import and not a parse: ``LOG10_ZSUN =
    math.log10(Z_SUN)`` exists only after arithmetic.
    """
    table: dict[str, float] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tengri"):
            try:
                src = importlib.import_module(node.module)
            except Exception:
                continue
            for alias in node.names:
                value = getattr(src, alias.name, None)
                if isinstance(value, float):
                    table[alias.asname or alias.name] = value
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, float):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        table[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            if isinstance(node.target, ast.Name) and isinstance(node.value.value, float):
                table[node.target.id] = node.value.value
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return table
    for attr in dir(module):
        value = getattr(module, attr, None)
        if isinstance(value, float):
            table.setdefault(attr, value)
    return table


def _as_constant(node: ast.AST, table: dict[str, float]) -> float | None:
    """The float *node* denotes, or None if it is not a constant this can see."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name):
        return table.get(node.id)
    if isinstance(node, ast.Attribute):
        return table.get(node.attr)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _as_constant(node.operand, table)
        return None if inner is None else -inner
    return None


def _pow10_of(node: ast.AST, name: str) -> bool:
    """True if *node* contains ``10 ** name`` (however deeply)."""
    for sub in ast.walk(node):
        if (
            isinstance(sub, ast.BinOp)
            and isinstance(sub.op, ast.Pow)
            and isinstance(sub.left, ast.Constant)
            and float(sub.left.value) == 10.0
            and isinstance(sub.right, ast.Name)
            and sub.right.id == name
        ):
            return True
    return False


def _prior_bounds(record) -> tuple[float, float] | None:
    """``(lo, hi)`` of the range this parameter DECLARES, from the registry.

    ``free_prior`` when the declaration has one -- that is the range ``FREE``
    expands to, i.e. the range a fit can actually reach -- else the registry
    default. A ``Fixed`` default with no ``free_prior`` contributes its single
    value; the parameter cannot leave it.
    """
    prior = record.free_prior if record.free_prior is not None else record.prior
    bounds = getattr(prior, "bounds", None)
    if bounds is not None:
        lo, hi = bounds
        return float(lo), float(hi)
    value = getattr(prior, "value", None)
    if value is not None:
        return float(value), float(value)
    return None


def _param_reach(pattern: str, transform: str, params) -> tuple[str, float] | None:
    """``(widest matching parameter, largest |value| its prior admits)``."""
    best_name, best = None, None
    for name, record in params.items():
        if not re.match(pattern, name):
            continue
        bounds = _prior_bounds(record)
        if bounds is None:
            continue
        lo, hi = bounds
        if transform == "pow10":
            reach = max(abs(10.0**lo), abs(10.0**hi))
        else:
            reach = max(abs(lo), abs(hi))
        if best is None or reach > best:
            best_name, best = name, reach
    if best_name is None:
        return None
    return best_name, best


def _scan() -> list[Seam]:
    """Every ``large constant x parameter`` site in ``src/tengri``."""
    from tengri.parameters.registry import registry

    params = registry()
    # Registry names and their short forms: ``agn_log_lbol`` is spelled in full
    # in the AGN blocks, while ``xray_log_nh`` arrives unpacked as ``log_nh``.
    direct: dict[str, str] = {name: name for name in params}
    short: dict[str, set[str]] = {}
    for name in params:
        parts = name.split("_")
        for cut in range(1, len(parts)):
            form = "_".join(parts[cut:])
            # A short form has to be specific enough to BE a reference to this
            # parameter rather than an English word that happens to appear in a
            # local. ``log_lbol`` and ``log_total_mass`` qualify; ``sfr``,
            # ``temp`` and ``alpha`` do not, and admitting them attributed a
            # local ``temperature`` in the nebular continuum to the AGN polar
            # temperature -- a seam report nobody could act on.
            if "_" in form and len(form) >= _MIN_SHORT_FORM:
                short.setdefault(form, set()).add(name)
    for form, owners in short.items():
        if len(owners) == 1 and form not in direct:
            direct[form] = next(iter(owners))

    seams: dict[str, Seam] = {}
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        module_name = _module_name(path, _SRC_ROOT)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError):
            continue
        table = _constant_table(tree, module_name)
        where = _qualnames(tree)

        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, (ast.Mult, ast.Div)):
                continue
            for side, other in ((node.left, node.right), (node.right, node.left)):
                value = _as_constant(side, table)
                if value is None or value == 0.0 or not math.isfinite(value):
                    continue
                if abs(math.log10(abs(value))) < _BIG_DEX:
                    continue
                if _as_constant(other, table) is not None:
                    continue  # constant times constant: folded, not a seam
                hit = _classify(other, direct, params)
                if hit is None:
                    continue
                param, transform, param_max = hit
                seam = Seam(
                    module=module_name,
                    where=where.get(id(node)) or "<module>",
                    lineno=node.lineno,
                    constant=ast.unparse(side),
                    value=value,
                    param=param,
                    transform=transform,
                    param_max=param_max,
                    reach=abs(value) * param_max,
                )
                seams.setdefault(seam.key, seam)
                break
    return sorted(seams.values(), key=lambda s: (s.module, s.lineno))


def _classify(expr: ast.AST, direct: dict[str, str], params) -> tuple[str, str, float] | None:
    """``(param, transform, param_max)`` if *expr* is parameter-derived."""
    names = {n.id for n in ast.walk(expr) if isinstance(n, ast.Name)}
    best: tuple[str, str, float] | None = None
    for name in sorted(names):
        if name in _ALIASES:
            pattern, transform = _ALIASES[name]
        elif name in direct:
            pattern = f"^{re.escape(direct[name])}$"
            transform = "pow10" if _pow10_of(expr, name) else "identity"
        else:
            continue
        hit = _param_reach(pattern, transform, params)
        if hit is None:
            continue
        param, param_max = hit
        if best is None or param_max > best[2]:
            best = (param, transform, param_max)
    return best


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="print the inventory and exit")
    args = parser.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        seams = _scan()
    over = [s for s in seams if s.over_range]

    if args.list:
        print(f"{len(seams)} scale seam(s); {len(over)} over float32 range within its own prior\n")
        for seam in seams:
            mark = "OVER" if seam.over_range else "ok  "
            known = _FAMILY_OF.get(seam.key, "NEW")
            print(
                f"{mark} {known:22s} {seam.reach:10.3e}  {seam.key}  (line {seam.lineno})\n"
                f"                        {seam.constant}={seam.value:.4g} x "
                f"{seam.transform}({seam.param}) <= {seam.param_max:.4g}"
            )
        return 0

    unexplained = [s for s in over if s.key not in _FAMILY_OF]
    if unexplained:
        print("float32 scale seam(s) over range within their own declared prior, unregistered:\n")
        for seam in unexplained:
            print(
                f"  {seam.key}  (src/{seam.module.replace('.', '/')}.py:{seam.lineno})\n"
                f"    {seam.constant} = {seam.value:.4g} multiplies "
                f"{seam.transform}({seam.param}), which the registry declares up to "
                f"{seam.param_max:.4g}\n"
                f"    standalone product reaches {seam.reach:.4e}; float32 holds "
                f"{_F32_MIN_NORMAL:.3e} to {_F32_MAX:.3e}"
            )
        print(
            "\nEach can become inf (or flush to zero) in float32 somewhere inside the "
            "range its own parameter declares. That is silent: the forward pass stays "
            "finite and the gradient is what goes wrong.\n"
            "Fix, in order of preference:\n"
            "  1. group so the large constant never stands alone against the "
            "parameter -- fold it into the array operand it will meet anyway, or "
            "state the grouping in the graph (jax.lax.optimization_barrier), as "
            "components/stellar/component.py::_mass_scale_lnu does;\n"
            "  2. carry the constant in log space (utils.scale.apply_log10_scale and "
            "log10_flux_scale), which is what the flux-scale seam does;\n"
            "  3. if the grouping already keeps the evaluated expression in range, "
            "register it in _HANDLED in this file with the reason -- and add it to "
            "tests/regression/precision/test_float32_scale_seam_sweep.py, which "
            "sweeps each registered seam across its declared prior in float32 and "
            "requires the gradient to come back finite AND non-zero;\n"
            "  4. if it is over range and NOT safe, file it and record it in "
            "_OPEN_DEFECTS with the issue number. That keeps the gate green "
            "without letting the inventory claim a live defect is handled."
        )
        return 1

    live = {s.key for s in over}
    stale = sorted(set(_FAMILY_OF) - live)
    if stale:
        print("_HANDLED entries that are no longer seams over float32 range -- delete them:\n")
        for key in stale:
            print(f"  {key}  (family {_FAMILY_OF[key]})")
        print(
            "\nA registration outlives its seam when the line moves or the grouping is "
            "fixed properly. Leaving it behind makes the inventory a list of claims "
            "nobody checks."
        )
        return 1

    print(
        f"OK: {len(seams)} scale seam(s) enumerated, {len(over)} over float32 range "
        f"within their own declared prior -- {len(_HANDLED)} families with a recorded "
        f"grouping, {len(_OPEN_DEFECTS)} filed as open defect(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
