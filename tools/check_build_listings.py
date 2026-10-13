#!/usr/bin/env python3
"""CI guard: every ``SEDModel.build(...)`` listing in the docs must execute.

Motivation
----------
A documentation listing is the first code a reader copies. Nothing ran
the ``SEDModel.build(...)`` listings in ``docs/``, and one of them carried
eight defects at once: names that do not exist and an inverted
Observation/Data split (#2430). ``check_doc_examples.py`` resolves names
and binds signatures, but it never builds a model, so a listing whose
names resolve can still fail at build time, and a listing that
misnames a kwarg was caught only by a reader.

This checker executes each listing against a committed fixture and
checks that the model it builds predicts finite, positive photometry at
a sampled parameter point.

What it checks
--------------
0. Execution stops before the first statement that calls ``.fit(...)`` or
   ``.run(...)``. A documented fit is minutes of sampling on data the
   listing does not define; the gate checks the build, and every statement
   up to the fit still runs.
1. Every fenced ``python`` block (and, in ``.tex`` files, every
   ``lstlisting`` and ``minted`` python body) that contains
   ``SEDModel.build(`` is executed in a fresh namespace.
2. Namespace preload: ``tengri``, ``from tengri import *``, ``jax``,
   ``jnp``, ``np``, ``ssp`` / ``ssp_data`` (the grid chosen above),
   ``obs`` / ``observation`` (five SDSS bands, the photometry
   configuration), and ``filters`` (the same five band names).
3. Literal paths in ``load_ssp_data("...")`` and ``cache_dir="..."`` are
   rewritten to the fixtures before execution, and ``download_ssp(...)``
   becomes the fixture path, because a documented path or download names
   a file the reader has, not one the repository has.
4. Every ``SEDModel`` bound in the namespace after execution is sampled
   with ``model.spec.sample(jax.random.PRNGKey(0))`` and must return
   finite, positive photometry from ``model.predict_photometry``.

Fixtures are the two SSP grids tracked in git (the CI docs job has no
download step): the bare FSPS grid, and the wNE grid for a listing that
selects ``neb={'type': 'ssp'}``.

A fence immediately preceded by a line containing ``<!-- no-exec -->``
is a deliberate fragment; it is counted as skipped and not executed.

Usage
-----
::

    python tools/check_build_listings.py              # scan docs/
    python tools/check_build_listings.py --list       # print listings found
    python tools/check_build_listings.py --no-default --paths FILE [...]

Exit status is 1 if any listing fails, 0 otherwise.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import io
import re
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = REPO_ROOT / "docs"
DATA_DIR = REPO_ROOT / "data"
# Both grids are tracked in git, so CI can read them. The bare grid serves
# every listing; the wNE grid serves a listing that selects the baked-in
# nebular backend explicitly (``neb={'type': 'ssp'}``).
BARE_SSP = "fsps_prsc_miles_chabrier.h5"
WNE_SSP = "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
WNE_NEB = re.compile(r"neb\s*=\s*\{[^}]*['\"]type['\"]\s*:\s*['\"]ssp['\"]")
BAND_NAMES = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
NO_EXEC_MARKER = "<!-- no-exec -->"
RAISES_MARKER = re.compile(r"<!--\s*raises:\s*(\w+)\s*-->")
BUILD_CALL = "SEDModel.build("

# Copied from tools/check_doc_examples.py (PY_FENCE); the tools do not import each other.
PY_FENCE = re.compile(r"```(?:python|py)\n(.*?)```", re.DOTALL)
LSTLISTING = re.compile(r"\\begin\{lstlisting\}(?:\[[^\]]*\])?(.*?)\\end\{lstlisting\}", re.DOTALL)
MINTED = re.compile(r"\\begin\{minted\}(?:\[[^\]]*\])?\{python\}(.*?)\\end\{minted\}", re.DOTALL)
SSP_LITERAL = re.compile(r"(load_ssp_data\(\s*)([\"'])[^\"']*\2")
CACHE_LITERAL = re.compile(r"(cache_dir\s*=\s*)([\"'])[^\"']*\2")
DOWNLOAD_SSP = re.compile(r"\b(?:tengri\.)?download_ssp\([^()]*\)")
# Method names that run inference. A documented fit is minutes of sampling
# and needs data the listing does not define, so execution stops before it.
INFERENCE_ATTRS = {"fit", "run"}

# Same exclusion rule as check_doc_examples.py: generated, archived, or internal pages.
EXCLUDED_DIR_PARTS = {
    "_build",
    "auto_examples",
    "archive",
    "adr",
    "internal",
    "__pycache__",
    "dev",
    "developer",
}


def is_excluded(path: Path) -> bool:
    """Return True for pages under a directory the docs guards skip."""
    return any(part in EXCLUDED_DIR_PARTS for part in path.parts)


@dataclass(frozen=True)
class Listing:
    """One documented listing: where it starts, its code, and its markers."""

    line: int
    code: str
    no_exec: bool
    expect: str | None


def _line_of(text: str, offset: int) -> int:
    """Return the 1-based line number containing ``offset``."""
    return text.count("\n", 0, offset) + 1


def _preceding_line(text: str, fence_line: int) -> str:
    """Return the line before a fence, or an empty string at the top of a file."""
    lines = text.splitlines()
    return lines[fence_line - 2] if fence_line >= 2 else ""


def _expected_raise(text: str, fence_line: int) -> str | None:
    """Return the exception name a listing must raise, from the line before it."""
    match = RAISES_MARKER.search(_preceding_line(text, fence_line))
    return match.group(1) if match else None


def listings_in_text(text: str, suffix: str) -> list[Listing]:
    """Return every listing that calls the build, with its markers."""
    found: list[Listing] = []
    for match in LSTLISTING.finditer(text) if suffix == ".tex" else PY_FENCE.finditer(text):
        code = match.group(1)
        if BUILD_CALL not in code:
            continue
        line = _line_of(text, match.start())
        no_exec = suffix != ".tex" and NO_EXEC_MARKER in _preceding_line(text, line)
        expect = None if suffix == ".tex" else _expected_raise(text, line)
        found.append(Listing(line, code, no_exec, expect))
    if suffix == ".tex":
        for match in MINTED.finditer(text):
            if BUILD_CALL in match.group(1):
                found.append(Listing(_line_of(text, match.start()), match.group(1), False, None))
    return found


def scan_paths(extra: list[Path], include_default: bool) -> list[tuple[Path, Listing]]:
    """Collect ``(path, listing)`` for every listing in scope."""
    files: list[Path] = []
    if include_default:
        files += [p for p in DOCS_ROOT.rglob("*.md") if not is_excluded(p.relative_to(DOCS_ROOT))]
    for item in extra:
        if item.is_dir():
            files += sorted(item.rglob("*.md")) + sorted(item.rglob("*.tex"))
        else:
            files.append(item)
    out = []
    for path in sorted(set(files)):
        text = path.read_text(encoding="utf-8", errors="replace")
        out += [(path, listing) for listing in listings_in_text(text, path.suffix)]
    return out


def ssp_fixture_for(code: str) -> str:
    """Name the SSP grid a listing runs against."""
    return WNE_SSP if WNE_NEB.search(code) else BARE_SSP


def rewrite_literals(code: str, ssp_file: str) -> str:
    """Point documented data paths and downloads at the committed fixtures.

    ``download_ssp(...)`` fetches a grid over the network; in a listing it
    stands for "the SSP grid", so it is replaced by the fixture path.
    """
    fixture = DATA_DIR / ssp_file
    code = DOWNLOAD_SSP.sub(f'"{fixture}"', code)
    code = SSP_LITERAL.sub(lambda m: f'{m.group(1)}"{fixture}"', code)
    return CACHE_LITERAL.sub(lambda m: f'{m.group(1)}"{DATA_DIR / "filters"}"', code)


def _runs_inference(stmt: ast.stmt) -> bool:
    """True when the statement calls one of INFERENCE_ATTRS anywhere in it."""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in INFERENCE_ATTRS
        for node in ast.walk(stmt)
    )


def build_prefix(code: str, ssp_file: str) -> ast.Module:
    """Parse a listing and keep the statements before the first inference call.

    Raises SyntaxError for a listing that does not parse, which is a finding.
    """
    tree = ast.parse(rewrite_literals(code, ssp_file))
    kept = []
    for stmt in tree.body:
        if _runs_inference(stmt):
            break
        kept.append(stmt)
    return ast.Module(body=kept, type_ignores=[])


def base_namespace(ssp_file: str) -> dict:
    """Return the preloaded fixture namespace for one SSP grid."""
    import jax
    import jax.numpy as jnp
    import numpy as np

    import tengri
    from tengri import Observation, Photometry, load_ssp_data
    from tengri.observation.filters import load_filter_set

    with warnings.catch_warnings():
        # Grid advisories (the wNE note, for one) are not findings here.
        warnings.simplefilter("ignore")
        _, _, curves = load_filter_set(BAND_NAMES)
        ssp = load_ssp_data(str(DATA_DIR / ssp_file))
    obs = Observation(photometry=Photometry(filters=tuple(curves)))
    ns: dict = {
        "tengri": tengri,
        "jax": jax,
        "jnp": jnp,
        "np": np,
        "ssp": ssp,
        "ssp_data": ssp,
        "obs": obs,
        "observation": obs,
        "filters": list(BAND_NAMES),
    }
    exec("from tengri import *", ns)
    return ns


def check_models(ns: dict) -> None:
    """Sample every SEDModel bound in ``ns`` and require positive, finite flux."""
    import jax
    import numpy as np

    from tengri import SEDModel

    seen: set[int] = set()
    for value in list(ns.values()):
        if not isinstance(value, SEDModel) or id(value) in seen:
            continue
        seen.add(id(value))
        params = value.spec.sample(jax.random.PRNGKey(0))
        flux = np.asarray(value.predict_photometry(params))
        if not (np.all(np.isfinite(flux)) and np.all(flux > 0)):
            raise AssertionError(f"photometry not finite and positive: {flux.tolist()}")


def _raised_name(exc: BaseException, expect: str) -> bool:
    """True when ``exc`` is an instance of the class named ``expect``."""
    return any(cls.__name__ == expect for cls in type(exc).__mro__)


def execute_listing(listing: Listing, base: dict, ssp_file: str) -> None:
    """Run one listing in a fresh copy of the fixture namespace.

    A listing that carries ``<!-- raises: Name -->`` must raise ``Name``;
    it passes by raising and fails by completing or raising anything else.

    The JIT caches are cleared afterwards: each listing builds its own
    compiled model, and without this the shared process grows until the
    kernel kills it.
    """
    import gc

    import jax

    ns = dict(base)
    try:
        module = build_prefix(listing.code, ssp_file)
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter("ignore")
            try:
                exec(compile(module, "<listing>", "exec"), ns)
            except Exception as exc:
                if listing.expect and _raised_name(exc, listing.expect):
                    return
                raise
            if listing.expect:
                raise AssertionError(f"expected {listing.expect}, but the listing completed")
            check_models(ns)
    finally:
        ns.clear()
        jax.clear_caches()
        gc.collect()


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip()
    return text.splitlines()[0] if text else ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--list", action="store_true", help="print listings found and exit 0")
    parser.add_argument("--no-default", action="store_true", help="skip the docs/ scan")
    parser.add_argument("--paths", nargs="*", default=[], type=Path, help="extra files or dirs")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    listings = scan_paths(args.paths, include_default=not args.no_default)
    if args.list:
        for path, listing in listings:
            tag = "  (no-exec)" if listing.no_exec else ""
            tag += f"  (raises {listing.expect})" if listing.expect else ""
            print(f"{path}:{listing.line}{tag}")
        print(f"{len(listings)} listings")
        return 0

    to_run = [(path, item) for path, item in listings if not item.no_exec]
    skipped = len(listings) - len(to_run)
    failures: list[str] = []
    bases: dict[str, dict] = {}
    for path, item in to_run:
        started = time.perf_counter()
        ssp_file = ssp_fixture_for(item.code)
        try:
            if ssp_file not in bases:
                bases[ssp_file] = base_namespace(ssp_file)
            execute_listing(item, bases[ssp_file], ssp_file)
            status = "ok"
        except Exception as exc:  # report every listing, not just the first
            status = "FAILED"
            failure = f"{path}:{item.line}: {type(exc).__name__}: {_first_line(exc)}"
            failures.append(failure)
            print(failure, flush=True)
        elapsed = time.perf_counter() - started
        print(f"  {path}:{item.line} {status} ({elapsed:.1f} s)", file=sys.stderr, flush=True)
    print(f"{len(listings)} listings, {len(to_run)} executed, {len(failures)} failed")
    if skipped:
        print(f"{skipped} skipped (no-exec)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
