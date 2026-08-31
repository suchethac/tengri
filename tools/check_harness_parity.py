#!/usr/bin/env python
# SPDX-License-Identifier: BSD-3-Clause
"""Prove that every benchmark fixture still builds the model it claims to build.

Closes the loop #2096 opened: ``bench/scripts/benchmark_notebook_sampler.py``
and ``bench/scripts/benchmark_quickstart_sampler.py`` mirror the notebooks'
models by **copying** them into Python functions, and nothing enforced that the
copy stayed a copy. Two fixtures had silently drifted -- nb05 by PR #1989
(``law_bc="calzetti"`` rewritten to ``law="calzetti"``, which sets a second
screen), nb00 by PR #2044 (tsnorm + two-component at D=7 rewritten to dpl +
single-component + nebular at D=6) -- and both kept labeling their rows with
the notebook's name.

WHAT THIS CHECKS
================

Every fixture in ``benchmark_notebook_sampler.NOTEBOOKS`` must declare a
``parity=`` provenance block. There are exactly three kinds and no default, so
a fixture added without one fails the check rather than joining the registry
unexamined -- that omission is the defect this file exists to prevent.

``kind="mirrors"``
    The fixture claims to be a copy of a published notebook's model. The check
    executes the notebook's own ``py:percent`` cells up to and including the one
    that defines its ``SEDModel``, then requires the two models to agree on

    * the canonical parameter spec (``Parameters.to_groups()``, which round-trips
      through ``parse_groups`` and carries every group's ``type``, every prior
      and every fixed value),
    * the band list, in order,
    * the SSP grid identity, and
    * **the predicted photometry at a fixed parameter vector**, to within
      floating-point noise.

``kind="historical"``
    The fixture reproduces a *superseded* model on purpose -- ``05pre`` is nb05
    before #1989, ``00``/``00pre`` are the quickstart before #2044 -- so it must
    NOT match today's notebook, and a check that flagged it would be noise. It
    is instead anchored to a sibling fixture (``anchor=``) and must differ from
    that sibling in **exactly** the spec keys it declares in ``differs_in``, no
    more and no fewer. The anchor chain must terminate at a ``mirrors`` fixture.

``kind="standalone"``
    Not a copy of anything -- the two controls. It must say why in ``why=``.

WHY THIS DESIGN, AND NOT THE OTHER TWO #2096 LISTED
===================================================

#2096 offered three candidates and deliberately picked none.

1. *Import the model builder from the notebook rather than duplicating it.*
   Rejected. It reads as the cleanest option and is the most invasive one: the
   ``SEDModel.build`` call is the pedagogical centerpiece of every one of these
   notebooks -- ``00_quickstart`` spends eleven lines of prose on the six
   arguments -- and hoisting it into an importable helper so a benchmark can
   reach it makes the teaching material worse to serve the harness. It also
   cannot express ``05pre``/``00pre`` at all: a historical fixture has no
   notebook left to import from, and those fixtures exist precisely so
   superseded reports stay reproducible. This file still *executes* the
   notebook's own code -- which is where option 1's real value was -- but does
   it from outside, leaving the notebooks untouched.

2. *A test asserting identical predictions for a fixed parameter vector.*
   Chosen, and strengthened. Predictions alone are necessary but not
   sufficient: a prior-range change (``met_logzsol`` over ``U(-2.0, 0.2)``
   against ``U(-1.5, 0.3)``, the difference between ``00`` and ``00pre``) moves
   the posterior while leaving the forward model at a fixed parameter vector
   bit-identical. So the spec comparison runs first and the prediction check
   runs second, and each catches what the other cannot: the spec sees priors,
   dispositions and group types; the prediction sees SSP grids, band ordering,
   ``approx=`` and any wiring ``to_groups()`` has historically dropped (#964,
   #1777 are two rounds of exactly that).

3. *A model-spec checksum in each report row.*
   Rejected as the primary mechanism, though the spec digest this file computes
   would serve it. A checksum is only read when someone reruns a benchmark, and
   the reason the drift accumulated for ten days is that **nobody could run
   these scripts at all** -- they raised at model build. A mechanism that needs
   the broken thing to be run before it reports is the wrong shape for this
   failure. A checksum also cannot tell a deliberate change from an accidental
   one; it just changes, and ``05pre``'s digest changing means "someone
   corrupted the historical fixture" while ``05``'s digest changing means
   "someone updated it to follow the notebook", and no digest distinguishes
   those. The ``mirrors``/``historical`` split does.

WHAT TO DO WHEN IT FAILS
========================

The failure message names the fixture, the kind of mismatch, and the exact spec
keys that differ.

* A ``mirrors`` fixture failed -> **the harness follows the notebook.** Update
  the fixture's builder to match the notebook, and consider whether the reports
  that quote its rows now describe a model that fixture no longer builds. Do
  not edit the notebook to match the fixture, and do not change a published
  measurement: if a report's *description* is now wrong, fix the description and
  say so in the report.
* If the old model still needs to be measurable -- because a published report
  reproduces only under it -- keep it as a new ``historical`` fixture anchored
  to the repaired one, the way ``05pre`` is anchored to ``05``.
* A ``historical`` fixture failed with "no longer differs from its anchor" ->
  someone repaired it into a duplicate. Either revert that, or delete the
  fixture and say in the report that its rows are now the anchor's.
* A ``historical`` fixture failed with an undeclared differing key -> the
  fixture drifted in a way its own provenance block does not admit to. Widen
  ``differs_in`` only if the new difference is intended and documented.

Usage::

    python tools/check_harness_parity.py            # every fixture
    python tools/check_harness_parity.py --fixture 05
    python tools/check_harness_parity.py --list     # provenance table, no builds

Building a model is all this needs; no fit is ever run.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPTS = REPO_ROOT / "bench" / "scripts"

#: Relative tolerance on predicted photometry. Both sides run the same code on
#: the same machine, so an honest match is bit-identical or within a few ULP;
#: this is loose enough to survive a reduction-order change and orders of
#: magnitude tighter than the 4-8% the #1989 dust-law swap moves the fluxes by.
FLUX_RTOL = 1e-9

#: The parameter vector predictions are compared at. Fixed, and drawn from the
#: *notebook's* spec so the two models are asked the same question.
PARAM_SEED = 20960


class ParityError(AssertionError):
    """A fixture does not build the model its provenance block claims."""


# --------------------------------------------------------------------------
# loading the registry and the notebooks
# --------------------------------------------------------------------------


def _load_module(path: Path, name: str):
    """Import a bench script by path -- they are scripts, not an importable package."""
    if str(BENCH_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(BENCH_SCRIPTS))
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def registry() -> dict[str, dict]:
    """``benchmark_notebook_sampler.NOTEBOOKS`` -- the one fixture registry.

    ``benchmark_quickstart_sampler.py`` and ``diagnose_ghmc_meads.py`` both
    import from it rather than keeping their own copies, which is the "repair
    it in one place" half of #2096: two independent per-worktree repairs of a
    duplicated builder are how two contradictory nb05 baselines arose.
    """
    path = BENCH_SCRIPTS / "benchmark_notebook_sampler.py"
    return _load_module(path, "benchmark_notebook_sampler").NOTEBOOKS


def _percent_code_cells(text: str) -> list[str]:
    """Code-cell sources from a jupytext ``py:percent`` file, in order.

    Delegates to ``tools/check_notebook_renders.py``'s parser, which
    ``tests/contract/test_notebook_renders.py`` already pins cell-for-cell
    against jupytext itself. Re-implementing it here would be a second parser to
    keep in step, and a parser that silently disagreed would make this guard
    decorative in the same way #1506 warned about.
    """
    sys.path.insert(0, str(REPO_ROOT / "tools"))
    try:
        from check_notebook_renders import parse_percent_code_cells
    finally:
        sys.path.pop(0)
    return parse_percent_code_cells(text)


@functools.cache
def notebook_model(rel_path: str):
    """Build the model a published notebook builds, by running the notebook's own code.

    Executes ``py:percent`` code cells in order and stops at the first one after
    which an ``SEDModel`` exists in the namespace -- imports, SSP load, filters
    and the ``SEDModel.build`` call, and nothing after it. No fit is run and no
    figure is drawn: on every notebook in the registry the model is built within
    the first few cells, which is what makes this affordable in a test.

    Notebooks ``import`` from ``notebooks/_setup.py`` and write figures relative
    to their own directory, so this runs with ``notebooks/`` on ``sys.path`` and
    as the working directory.
    """
    from tengri import SEDModel

    path = REPO_ROOT / rel_path
    if not path.is_file():
        raise ParityError(f"{rel_path}: notebook source not found")
    import contextlib
    import io

    cells = _percent_code_cells(path.read_text(encoding="utf-8"))
    ns: dict[str, Any] = {"__name__": "__notebook__", "__file__": str(path)}
    cwd = Path.cwd()
    sys.path.insert(0, str(path.parent))
    os.chdir(path.parent)
    try:
        # Notebooks print -- model summaries, citation blocks, bin edges. That is
        # their job and not this guard's output, so it is swallowed; a failure
        # here raises rather than printing, so nothing diagnostic is lost.
        with contextlib.redirect_stdout(io.StringIO()):
            for index, cell in enumerate(cells):
                exec(compile(cell, f"{rel_path}#cell{index}", "exec"), ns)
                models = [
                    v for k, v in ns.items() if isinstance(v, SEDModel) and not k.startswith("_")
                ]
                if models:
                    return models[0]
    finally:
        os.chdir(cwd)
        sys.path.remove(str(path.parent))
    raise ParityError(
        f"{rel_path}: ran every code cell without an SEDModel appearing. The "
        "notebook no longer builds a model in the way this guard can see it; "
        "fix the guard rather than deleting the fixture's parity declaration."
    )


# --------------------------------------------------------------------------
# the two comparisons
# --------------------------------------------------------------------------


def _flatten(obj: Any, prefix: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out |= _flatten(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(obj, (list, tuple)):
        out[prefix] = "[" + ", ".join(repr(v) for v in obj) + "]"
    else:
        out[prefix] = repr(obj)
    return out


def signature(sed) -> dict[str, str]:
    """A canonical, diffable description of everything a fixture declares.

    ``Parameters.to_groups()`` is the canonical form: it round-trips through
    ``parse_groups`` and its docstring guarantees identical free/fixed
    partitions, identical distributions *and* every group's ``type``. Bands and
    the SSP grid are not part of it, so they are added -- a fixture that fits
    the same physics to different filters, or against a nebular-baked grid
    instead of a bare-stellar one, is a different fixture.
    """
    ssp = getattr(sed, "ssp_data", None)
    sig = _flatten(sed.spec.to_groups())
    sig["bands"] = "[" + ", ".join(sed.observation.photometry.names) + "]"
    sig["free_params"] = "[" + ", ".join(sed.spec.free_params) + "]"
    if ssp is not None:
        sig["ssp.file"] = Path(str(getattr(ssp, "source", "?"))).name
        sig["ssp.nebular"] = repr(getattr(ssp, "nebular", None))
        sig["ssp.shape"] = repr(tuple(ssp.ssp_flux.shape))
    return sig


def signature_diff(left: dict[str, str], right: dict[str, str]) -> list[str]:
    """Spec keys on which two models disagree, including keys only one has."""
    return sorted(k for k in set(left) | set(right) if left.get(k) != right.get(k))


def max_relative_flux_difference(left, right) -> float:
    """Largest relative disagreement in predicted photometry at one shared point.

    The parameter vector is drawn once from ``left``'s spec and handed to both
    models, so the two are asked about the same galaxy rather than about two
    prior draws. Requires identical free-parameter names, which the spec
    comparison has already established by the time this runs.
    """
    import jax
    import numpy as np

    params = left.spec.sample(jax.random.PRNGKey(PARAM_SEED))
    a = np.asarray(left.predict_photometry(params), dtype=float)
    b = np.asarray(right.predict_photometry(params), dtype=float)
    if a.shape != b.shape:
        return float("inf")
    scale = np.maximum(np.abs(a), np.abs(b))
    scale = np.where(scale == 0.0, 1.0, scale)
    return float(np.max(np.abs(a - b) / scale))


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

KINDS = ("mirrors", "historical", "standalone")


def provenance(name: str, cfg: dict) -> dict:
    """The fixture's ``parity=`` block, validated for shape.

    A missing block is a failure and not a default. The whole defect #2096
    reports is a fixture that never said what it was a copy of; letting one join
    the registry silently would reproduce it.
    """
    par = cfg.get("parity")
    if not isinstance(par, dict) or "kind" not in par:
        raise ParityError(
            f"{name}: no parity= provenance block. Every fixture must declare "
            f"kind= one of {KINDS}; see tools/check_harness_parity.py."
        )
    kind = par["kind"]
    if kind not in KINDS:
        raise ParityError(f"{name}: parity kind {kind!r} is not one of {KINDS}")
    if kind == "mirrors" and not par.get("notebook"):
        raise ParityError(f"{name}: kind='mirrors' needs notebook=<path>")
    if kind == "historical":
        if not par.get("anchor"):
            raise ParityError(f"{name}: kind='historical' needs anchor=<fixture name>")
        if "differs_in" not in par:
            raise ParityError(
                f"{name}: kind='historical' needs differs_in=(...), the exact spec "
                "keys it is allowed to differ from its anchor in. An empty tuple "
                "is not allowed -- a historical fixture identical to its anchor "
                "is a duplicate, not a fixture."
            )
        if not par.get("superseded_by"):
            raise ParityError(
                f"{name}: kind='historical' needs superseded_by=<PR or commit>, so "
                "the fixture names the change it predates rather than being "
                "indefinitely exempt."
            )
    if kind == "standalone" and not par.get("why"):
        raise ParityError(f"{name}: kind='standalone' needs why=<one line>")
    return par


def anchor_chain(name: str, fixtures: dict[str, dict]) -> list[str]:
    """Follow ``anchor=`` to the ``mirrors`` fixture that grounds a historical one.

    This is what stops ``historical`` becoming an exemption. A historical
    fixture is defined *relative* to a live one, and the live one is checked
    against the notebook, so ``05pre`` is anchored to today's ``05_fitting_photometry``
    at one remove: it is required to differ from ``05`` in exactly the dust law,
    and ``05`` is required to be the notebook. Neither can rot without a failure.
    """
    seen: list[str] = [name]
    current = name
    while True:
        par = provenance(current, fixtures[current])
        if par["kind"] != "historical":
            if par["kind"] != "mirrors":
                raise ParityError(
                    f"{name}: anchor chain {' -> '.join(seen)} ends at a "
                    f"{par['kind']!r} fixture. A historical fixture must ground out "
                    "in one that mirrors a notebook, or nothing ties it to reality."
                )
            return seen
        current = par["anchor"]
        if current not in fixtures:
            raise ParityError(f"{name}: anchor {current!r} is not a fixture")
        if current in seen:
            raise ParityError(f"{name}: anchor cycle {' -> '.join([*seen, current])}")
        seen.append(current)


def check_fixture(name: str, fixtures: dict[str, dict], models: dict[str, Any]) -> str:
    """Check one fixture, returning a one-line verdict. Raises on failure."""
    par = provenance(name, fixtures[name])
    kind = par["kind"]

    if kind == "standalone":
        return f"standalone -- {par['why']}"

    if kind == "mirrors":
        rel = par["notebook"]
        theirs = notebook_model(rel)
        ours = models[name]
        diff = signature_diff(signature(ours), signature(theirs))
        if diff:
            ours_sig, their_sig = signature(ours), signature(theirs)
            detail = "\n".join(
                f"      {k}\n        fixture:  {ours_sig.get(k, '<absent>')}"
                f"\n        notebook: {their_sig.get(k, '<absent>')}"
                for k in diff
            )
            raise ParityError(
                f"{name}: no longer mirrors {rel}.\n"
                f"    {len(diff)} spec key(s) differ:\n{detail}\n"
                "    The harness follows the notebook: repair the fixture's builder. "
                "If the OLD model must stay measurable because a published report "
                "reproduces only under it, add it as a kind='historical' fixture "
                "anchored to this one -- do not edit the notebook, and do not "
                "change a published measurement."
            )
        worst = max_relative_flux_difference(ours, theirs)
        if not worst <= FLUX_RTOL:
            raise ParityError(
                f"{name}: spec matches {rel} but predicted photometry does not "
                f"(max relative difference {worst:.3e} > {FLUX_RTOL:.0e}). The "
                "specs agree, so the difference is in something to_groups() does "
                "not carry -- the SSP grid, the band ordering, approx=, or model "
                "wiring. #964 and #1777 were both structural losses of exactly "
                "that kind."
            )
        return f"mirrors {rel} (max rel flux diff {worst:.1e})"

    # historical
    chain = anchor_chain(name, fixtures)
    anchor = par["anchor"]
    diff = signature_diff(signature(models[name]), signature(models[anchor]))
    declared = sorted(par["differs_in"])
    if not diff:
        raise ParityError(
            f"{name}: is declared historical (superseded by {par['superseded_by']}) "
            f"but now builds exactly the same model as its anchor {anchor!r}. "
            "Someone repaired it into a duplicate. Either revert that, or delete "
            f"the fixture and record in the reports that its rows are {anchor!r}'s."
        )
    if diff != declared:
        undeclared = [k for k in diff if k not in declared]
        vanished = [k for k in declared if k not in diff]
        raise ParityError(
            f"{name}: differs from anchor {anchor!r} in keys its provenance block "
            f"does not describe.\n"
            f"    undeclared differences: {undeclared or 'none'}\n"
            f"    declared but no longer differing: {vanished or 'none'}\n"
            "    A historical fixture is exempt from notebook parity only in the "
            "ways it declares. Widen differs_in= if the new difference is "
            "intended and documented; otherwise the fixture has drifted."
        )
    return f"historical, {' -> '.join(chain)} (differs in {len(diff)} key(s))"


def build_models(fixtures: dict[str, dict], names: list[str]) -> dict[str, Any]:
    """Build every fixture that a check will need, sharing SSP loads."""
    import tengri

    cache: dict[str, Any] = {}
    needed = set(names)
    for name in list(names):
        par = fixtures[name].get("parity") or {}
        if par.get("kind") == "historical" and par.get("anchor"):
            needed.add(par["anchor"])
    models = {}
    for name in sorted(needed):
        ssp_name = fixtures[name].get("ssp", "fsps_prsc_miles_chabrier")
        if ssp_name not in cache:
            cache[ssp_name] = tengri.load_ssp(ssp_name, download=True)
        models[name] = fixtures[name]["build"](cache[ssp_name])
    return models


def check_all(names: list[str] | None = None) -> list[tuple[str, str]]:
    """Run every check. Returns ``(fixture, verdict)``; raises ``ParityError`` on failure."""
    fixtures = registry()
    targets = sorted(fixtures) if names is None else names
    for name in targets:
        if name not in fixtures:
            raise ParityError(f"{name!r} is not a fixture; have {sorted(fixtures)}")
    models = build_models(fixtures, targets)
    return [(name, check_fixture(name, fixtures, models)) for name in targets]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fixture", action="append", help="check only this fixture (repeatable)")
    parser.add_argument("--list", action="store_true", help="print provenance only, build nothing")
    args = parser.parse_args()

    fixtures = registry()
    if args.list:
        for name in sorted(fixtures):
            par = fixtures[name].get("parity") or {"kind": "MISSING"}
            target = par.get("notebook") or par.get("anchor") or par.get("why", "")
            print(f"  {name:<10}{par['kind']:<12}{target}")
        return 0

    targets = sorted(args.fixture or fixtures)
    models = build_models(fixtures, targets)
    failures = []
    for name in targets:
        try:
            print(f"  {name:<10}OK   {check_fixture(name, fixtures, models)}", flush=True)
        except ParityError as exc:
            failures.append(name)
            print(f"  {name:<10}FAIL {exc}", flush=True)
    if failures:
        print(f"\n{len(failures)} fixture(s) failed parity: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
