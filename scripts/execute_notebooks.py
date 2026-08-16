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
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tools"))

from check_no_local_paths import scrub_text
from sync_spine_notebooks_for_docs import (
    EXPERIMENTAL_SLUGS,
    EXPERIMENTAL_SUBDIR,
    SPINE_SLUGS,
)

#: Output fields that carry captured console text. ``text`` is stream output,
#: ``traceback`` is an error, and the ``data`` MIME bundle holds rich reprs.
_TEXT_MIMES = ("text/plain", "text/html", "text/markdown", "text/latex")


def _scrub(value):
    """Scrub a notebook text field, which nbformat gives as ``str`` or ``list``."""
    if isinstance(value, str):
        return scrub_text(value)
    if isinstance(value, list):
        return [scrub_text(v) if isinstance(v, str) else v for v in value]
    return value


def scrub_outputs(nb) -> int:
    """Rewrite absolute repo paths captured in cell outputs. Returns the count.

    This is where the leak is stopped rather than detected. Python's warning
    format prints the absolute source path, so any notebook that emits a warning
    captures the executing machine's filesystem into output that is then
    committed and published. #1749 shipped 26 such paths one commit after #1816
    added the guard that rejects them, and a diagnostic cell printing a resolved
    ``REPO_ROOT`` added three more -- a form no warning-format fix would catch,
    which is why the scrub is applied to captured output rather than to
    warnings.

    Only outputs are touched. Cell ``source`` comes from ``notebooks/<slug>.py``
    and is never rewritten: a scrub that reaches source has stopped redacting a
    machine path and started editing code.
    """
    n = 0
    for cell in nb.cells:
        for out in cell.get("outputs") or []:
            for key in ("text", "traceback"):
                if key in out:
                    before = out[key]
                    out[key] = _scrub(before)
                    n += out[key] != before
            data = out.get("data") or {}
            for mime in _TEXT_MIMES:
                if mime in data:
                    before = data[mime]
                    data[mime] = _scrub(before)
                    n += data[mime] != before
    return n

ALL_SLUGS = list(SPINE_SLUGS) + list(EXPERIMENTAL_SLUGS)


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
        scrub_outputs(nb)
        nbformat.write(nb, out)
        return False, time.perf_counter() - t0, 0, -1
    dt = time.perf_counter() - t0
    # Before any write, so all three -- the failure path above, notebooks/, and
    # the published render -- are scrubbed by one call. A traceback captured
    # from a failed run carries absolute paths too, which is why the failure
    # path scrubs as well.
    if scrubbed := scrub_outputs(nb):
        print(f"  scrubbed {scrubbed} absolute path(s) from captured output")
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
