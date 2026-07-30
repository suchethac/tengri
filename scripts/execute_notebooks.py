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
    python scripts/execute_notebooks.py 00_quickstart stochastic_sfh_recovery
    python scripts/execute_notebooks.py --all --timeout 1800

Writes ``notebooks/<slug>.ipynb``. Run
``python scripts/sync_spine_notebooks_for_docs.py`` afterwards to copy sources
into ``docs/spine/`` (it preserves the outputs this script produced).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sync_spine_notebooks_for_docs import EXPERIMENTAL_SLUGS, SPINE_SLUGS

ALL_SLUGS = list(SPINE_SLUGS) + list(EXPERIMENTAL_SLUGS)


def execute(slug: str, timeout: int) -> tuple[bool, float, int, int]:
    """Execute ``notebooks/<slug>.py`` in place.

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
        # cwd = notebooks/, so ../data and figures/ resolve as they do for a human.
        resources={"metadata": {"path": str(src.parent)}},
    )
    t0 = time.perf_counter()
    try:
        client.execute()
    except Exception as exc:  # kernel death is not a cell error, so catch broadly
        print(f"  kernel failure: {type(exc).__name__}: {exc}", file=sys.stderr)
        nbformat.write(nb, out)
        return False, time.perf_counter() - t0, 0, -1
    dt = time.perf_counter() - t0
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
    return errs == 0, dt, figs, errs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("slugs", nargs="*", help="notebook stems; default is none")
    ap.add_argument("--all", action="store_true", help="execute every published notebook")
    ap.add_argument("--list", action="store_true", help="print the published notebook list")
    ap.add_argument("--timeout", type=int, default=3000, help="per-cell timeout [s]")
    args = ap.parse_args()

    if args.list:
        for s in ALL_SLUGS:
            print(s)
        return 0

    slugs = ALL_SLUGS if args.all else args.slugs
    if not slugs:
        ap.error("give one or more slugs, or --all (see --list)")

    unknown = [s for s in slugs if s not in ALL_SLUGS]
    if unknown:
        print(f"error: not published notebooks: {', '.join(unknown)}", file=sys.stderr)
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
