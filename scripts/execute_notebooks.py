#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Execute published notebooks so their outputs embed for nbsphinx.

Replaces ``_execute_spine.sh`` and ``_execute_spine_light.sh``, which had rotted
in three ways at once: both ``cd``'d to a hardcoded absolute path, both listed
notebook stems that no longer exist (``03_fitting_photometry``,
``14_stochastic_sfh``, ``16_simulation_interface``, ...), and one exported
``MPLBACKEND=Agg`` (#1506).

**Never set MPLBACKEND when executing a notebook you intend to publish.** Under
Agg, ``plt.show()`` is a no-op, so no ``image/png`` output is captured and the
render ships with zero figures -- while ``figures/*.png`` on disk are written
normally and look correct, which is what makes the loss invisible. Measured on a
two-cell notebook: Agg 0 figures, ipykernel's default inline backend 1. The
notebooks carry no ``%matplotlib inline`` and rely on that default.

The notebook list is derived from :mod:`sync_spine_notebooks_for_docs` so there
is one source of truth for "what is published".

Usage::

    python scripts/execute_notebooks.py --list
    python scripts/execute_notebooks.py --list --ci
    python scripts/execute_notebooks.py --list --ci --json
    python scripts/execute_notebooks.py 00_quickstart stochastic_sfh_recovery
    python scripts/execute_notebooks.py --all --timeout 1800

Writes the executed notebook to **both** ``notebooks/<slug>.ipynb`` (gitignored,
handy for inspection) and the published render under ``docs/spine/``, which is
the file that actually ships.

Writing the render here is not a convenience. ``sync_spine_notebooks_for_docs``
deliberately takes code from ``notebooks/<slug>.py`` and **outputs from the
render it finds already committed** -- on CI ``notebooks/*.ipynb`` is gitignored
and absent, so trusting it would replace real outputs with an empty notebook.
That is correct for the sync and a trap for everything else: executing a
notebook and then syncing publishes the *new source* grafted onto the *old
outputs*, with nothing to indicate it. It shipped exactly once, in #1516, where
the page ended up quoting timings from a run of the previous code. This script
is the only step that knows fresh outputs were just produced, so it is the one
that writes them.

Run ``python scripts/sync_spine_notebooks_for_docs.py`` afterwards for markdown
normalization and link retargeting; it will now find, and preserve, the outputs
written here.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sync_spine_notebooks_for_docs import (
    EXPERIMENTAL_SLUGS,
    EXPERIMENTAL_SUBDIR,
    SPINE_SLUGS,
)

ALL_SLUGS = list(SPINE_SLUGS) + list(EXPERIMENTAL_SLUGS)

#: Slugs that cannot be executed on CI runners. Maps slug to the reason.
#: Every key MUST be in ALL_SLUGS.
CI_UNEXECUTABLE = {
    "apple_mps": "Requires Apple Silicon (JAX_PLATFORMS=mps); cannot run on ubuntu-latest",
    "nvidia_cuda": (
        "Requires an NVIDIA GPU (JAX_PLATFORMS=cuda) and a jax[cuda12] wheel; "
        "ubuntu-latest runners have neither, and the notebook's whole subject is "
        "the device. Its render is produced on a CUDA box"
    ),
    "multimodel_bma_candels": (
        "Requires locally generated wNE SSP grids for MIST/Padova/BaSTI "
        "(ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0, "
        "ssp_pdva_miles_chabrier_wNE_logGasU-2.0_logGasZ0.0, "
        "ssp_bsti_miles_chabrier_wNE_logGasU-2.0_logGasZ0.0); "
        "neither tracked in git nor in the download registry"
    ),
    "12_simulation_populations": (
        "Benchmarks a model with a live CLOUDY nebular backend (gas=True); "
        "requires locally generated cloudy_grid_*.h5 files (scripts/convert_fsps_cloudy_grid.py); "
        "no cloudy_grid_*.h5 is tracked in git, so a fresh checkout cannot run it. "
        "If a grid is later committed (13 MB), this entry can be dropped"
    ),
}

# Assert that every key in CI_UNEXECUTABLE is in ALL_SLUGS.
for slug in CI_UNEXECUTABLE:
    assert slug in ALL_SLUGS, f"CI_UNEXECUTABLE key {slug!r} not in ALL_SLUGS"


def docs_render_path(slug: str) -> Path:
    """Return the published render for ``slug``.

    Mirrors the routing in :mod:`sync_spine_notebooks_for_docs`: the numbered
    spine publishes to ``docs/spine/``, the standalone demonstrations to
    ``docs/spine/<EXPERIMENTAL_SUBDIR>/``.
    """
    spine = ROOT / "docs" / "spine"
    if slug in EXPERIMENTAL_SLUGS:
        return spine / EXPERIMENTAL_SUBDIR / f"{slug}.ipynb"
    return spine / f"{slug}.ipynb"


#: A machine-specific home directory, matching ``tools/check_no_local_paths.py``.
_HOME_PATH = re.compile(r"(?:/Users|/home)/[A-Za-z0-9][A-Za-z0-9_.-]*/")

#: A worktree root -- a home directory followed by any path ending in
#: ``.claude/worktrees/<name>/``. Stripped whole, so a path rendered from a
#: worktree comes out repo-relative and identical to one rendered from the main
#: checkout. (Spelled as a pattern rather than an example on purpose: a literal
#: one in this file would itself trip ``check_no_local_paths.py``.)
_WORKTREE_ROOT = re.compile(r"(?:/Users|/home)/[^/\s\"']+?/\.claude/worktrees/[^/\s\"']+/")


def strip_local_paths(nb) -> int:
    """Rewrite machine-specific absolute paths in cell outputs. Returns the count.

    Executing a notebook bakes the *absolute* source path into every warning and
    traceback it captures -- ``/Users/<someone>/.../src/tengri/forward/sed_model.py:7796:
    WildcardPartialFreeWarning`` and the like. Those strings ship to the public
    repository inside the committed render and describe the machine that produced
    it, which ``tools/check_no_local_paths.py`` rejects (#1816).

    This runs at the write, not as a cleanup pass over the repository, because the
    executor is where the paths enter a published artifact. A repository-wide
    scrub would fix today's renders and let the next execution reintroduce them --
    which is exactly what happened when #1749 merged three minutes after #1816
    landed the guard, taking `main` red on a class that had just been repaired.

    Rewrites, in order:

    1. this checkout's root, and any ``.claude/worktrees/<name>/`` root, to
       repo-relative -- so a render is byte-identical whether it was produced from
       the main checkout or a worktree;
    2. any surviving home directory to ``~/``, which keeps the text readable
       without naming a user.
    """
    root = f"{ROOT}/"
    n = 0

    def _clean(text: str) -> str:
        nonlocal n
        before = text
        text = text.replace(root, "")
        text = _WORKTREE_ROOT.sub("", text)
        text = _HOME_PATH.sub("~/", text)
        if text != before:
            n += 1
        return text

    for cell in nb.cells:
        for output in cell.get("outputs") or []:
            if "text" in output:
                t = output["text"]
                output["text"] = [_clean(x) for x in t] if isinstance(t, list) else _clean(t)
            if "traceback" in output:
                output["traceback"] = [_clean(x) for x in output["traceback"]]
            data = output.get("data") or {}
            for key in ("text/plain", "text/html"):
                if key in data:
                    v = data[key]
                    data[key] = [_clean(x) for x in v] if isinstance(v, list) else _clean(v)
    return n


def execute(slug: str, timeout: int) -> tuple[bool, float, int, int]:
    """Execute ``notebooks/<slug>.py`` and publish the result.

    On success the executed notebook is written to both ``notebooks/<slug>.ipynb``
    and the published render. A run that raised or died writes only the former,
    so a failed execution can be inspected without shipping it.

    Returns
    -------
    tuple of (bool, float, int, int)
        ``(ok, seconds, n_figures, n_errors)``. ``ok`` is False if any cell
        raised or the kernel died.
    """
    import jupytext
    import nbformat
    from nbclient import NotebookClient

    src = ROOT / "notebooks" / f"{slug}.py"
    out = src.with_suffix(".ipynb")
    nb = jupytext.read(src)
    nb.metadata.setdefault(
        "kernelspec", {"name": "python3", "display_name": "Python 3", "language": "python"}
    )

    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        allow_errors=True,
        # cwd = notebooks/, matching how a human opens them. Notebooks should still
        # anchor their own paths at the repo root rather than rely on this (#1486).
        resources={"metadata": {"path": str(src.parent)}},
    )
    t0 = time.perf_counter()
    try:
        client.execute()
    except Exception as exc:  # kernel death is not a cell error, so catch broadly
        print(f"  kernel failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        # Scrub here too. This path writes no *render*, but it does write
        # ``notebooks/<slug>.ipynb``, and for the numbered spine that file is
        # tracked -- the 29 paths this guard caught included four of them. A
        # failed run is also the case most likely to leak: a traceback names an
        # absolute source path on every frame, not just the one line a warning
        # emits.
        strip_local_paths(nb)
        nbformat.write(nb, out)
        return False, time.perf_counter() - t0, 0, -1
    dt = time.perf_counter() - t0
    strip_local_paths(nb)
    nbformat.write(nb, out)

    figs = sum(
        1
        for c in nb.cells
        for o in (c.get("outputs") or [])
        if "image/png" in (o.get("data") or {})
    )
    errs = sum(
        1 for c in nb.cells for o in (c.get("outputs") or []) if o.get("output_type") == "error"
    )
    if errs == 0:
        render = docs_render_path(slug)
        render.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, render)
    return errs == 0, dt, figs, errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slugs", nargs="*", help="notebook stems; default is none")
    ap.add_argument("--all", action="store_true", help="execute every published notebook")
    ap.add_argument("--list", action="store_true", help="print the published notebook list")
    ap.add_argument("--ci", action="store_true", help="exclude notebooks that cannot run on CI")
    ap.add_argument(
        "--json", action="store_true", help="with --list, print as JSON array (for CI)"
    )
    ap.add_argument("--timeout", type=int, default=3000, help="per-cell timeout [s]")
    args = ap.parse_args()

    # Filter slugs based on --ci flag
    available_slugs = ALL_SLUGS
    if args.ci:
        available_slugs = [s for s in ALL_SLUGS if s not in CI_UNEXECUTABLE]

    if args.list:
        if args.json:
            print(json.dumps(available_slugs))
        else:
            for s in available_slugs:
                print(s)
        return 0

    slugs = available_slugs if args.all else args.slugs
    if not slugs:
        ap.error("give one or more slugs, or --all (see --list)")

    unknown = [s for s in slugs if s not in available_slugs]
    if unknown:
        reason_str = ""
        for s in unknown:
            if s in CI_UNEXECUTABLE:
                reason_str = f" ({CI_UNEXECUTABLE[s]})"
                break
        print(
            f"error: not available notebook(s): {', '.join(unknown)}{reason_str}",
            file=sys.stderr,
        )
        return 1

    if "MPLBACKEND" in os.environ:
        # Refuse rather than silently produce a figureless render (#1506).
        print(
            f"error: MPLBACKEND={os.environ['MPLBACKEND']!r} is set. Under a non-interactive "
            "backend plt.show() captures nothing and the render ships with zero figures. "
            "Unset it and re-run.",
            file=sys.stderr,
        )
        return 1
    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    failed = []
    for slug in slugs:
        print(f"executing {slug} ...", flush=True)
        ok, dt, figs, errs = execute(slug, args.timeout)
        status = "ok" if ok else f"FAILED ({errs} cell errors)" if errs > 0 else "FAILED (kernel)"
        print(f"  {status}  {dt / 60:.1f} min  {figs} figure(s)", flush=True)
        if not ok:
            failed.append(slug)

    if failed:
        print(f"\n{len(failed)} notebook(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("\nall requested notebooks executed cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
