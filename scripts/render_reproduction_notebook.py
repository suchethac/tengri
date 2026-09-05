#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Render one reproduction notebook per ``reproduction/CONTRACT.md`` §7, and stamp it.

Runs the contract's exact recipe --
``jupytext --to ipynb``, then a headless
``PYTHONHASHSEED=0 jupyter nbconvert --to notebook --execute --inplace`` with
``MPLBACKEND`` unset so the inline-backend guard at the top of every
reproduction notebook embeds figures instead of losing them to Agg -- fails
loudly on any error output cell or a ``SystemExit``-truncated run (reusing
``tools/check_notebooks_executed.py``'s own detection logic, so a
partially-aborted render cannot slip past this script only to be caught later
by CI), then stamps the render with
``metadata["tengri_render"] = {"source_sha256", "executed_at",
"tengri_version"}`` and publishes the ``.ipynb`` (plus its ``_figs/*.png``)
to ``docs/reproduction/``.

``tools/check_repro_render_fresh.py`` recomputes ``source_sha256`` from the
checked-out ``.py`` to detect when a stamped render no longer matches its
source.

Usage::

    python scripts/render_reproduction_notebook.py <slug>
    python scripts/render_reproduction_notebook.py <slug> --timeout 1800
    python scripts/render_reproduction_notebook.py <slug> --no-publish
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools._repro_render_shared import code_cell_source_sha256
from tools.check_notebooks_executed import unexecuted


class RenderError(RuntimeError):
    """Raised when a render fails CONTRACT §7's loud-failure rule."""


def _tool_path(name: str) -> str:
    """Path to ``name`` installed alongside the running interpreter.

    A bare ``"jupytext"`` / ``"jupyter"`` resolves against ``$PATH``, which on
    a machine with more than one Python environment on it can silently pick a
    *different* jupytext/nbclient/IPython than the one this repo pins
    (``docs/requirements.txt`` floors ``matplotlib-inline`` for exactly this
    kind of version sensitivity -- see ``tools/check_figure_placement.py``).
    Resolving relative to :data:`sys.executable` runs the same virtualenv the
    caller invoked this script with.

    Parameters
    ----------
    name : str
        Executable name (``"jupytext"`` or ``"jupyter"``).

    Returns
    -------
    str
        The interpreter-adjacent path if it exists, else the bare name (a
        ``$PATH`` fallback for interpreters installed without console
        scripts alongside them).
    """
    candidate = Path(sys.executable).parent / name
    return str(candidate) if candidate.is_file() else name


def _error_cells(nb: dict) -> list[int]:
    """Indices of code cells carrying an ``error``-type output.

    Parameters
    ----------
    nb : dict
        A parsed ``.ipynb`` document.

    Returns
    -------
    list of int
        Cell indices whose outputs include at least one ``output_type ==
        "error"`` entry.
    """
    indices = []
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        for output in cell.get("outputs", []) or []:
            if output.get("output_type") == "error":
                indices.append(i)
                break
    return indices


def _tengri_version() -> str:
    """The installed ``tengri`` version, or ``"unknown"`` if it cannot import."""
    try:
        import tengri

        return getattr(tengri, "__version__", "unknown")
    # `import tengri` raises ImportError when it's absent from PYTHONPATH; the
    # `getattr` default already covers a present-but-versionless package, so
    # AttributeError here would only come from something stranger (a broken
    # partial import) that is still worth degrading gracefully for a stamp
    # field, not from a bug this function should hide. Anything else -- a
    # real error inside tengri's own import machinery -- propagates.
    except (ImportError, AttributeError):
        return "unknown"


def render_slug(slug: str, *, root: Path = ROOT, timeout: int = 1800) -> Path:
    """Execute one reproduction notebook and stamp it with its source hash.

    Parameters
    ----------
    slug : str
        Comparison folder name under ``<root>/reproduction/`` (e.g.
        ``"cigale"``).
    root : Path, optional
        Repository root under which ``reproduction/<slug>/01_<slug>.py``
        lives. Overridable so this can render a scratch fixture tree without
        touching the real repository.
    timeout : int, optional
        Per-cell execution timeout [s] passed to ``nbconvert``.

    Returns
    -------
    Path
        The rendered, stamped ``.ipynb`` under ``<root>/reproduction/<slug>/``.

    Raises
    ------
    RenderError
        If the source ``.py`` is missing, ``jupytext``/``nbconvert`` exits
        non-zero, any cell carries an ``error`` output, or any non-empty code
        cell never ran. The last case is CONTRACT §7's own warning made
        mechanical: the reproduction notebooks stop on a missing input with
        ``raise SystemExit``, which ``nbclient`` reports as a clean, zero-exit
        stop -- every cell after it is left unexecuted and the run reports
        success. See ``tools/check_notebooks_executed.py``.
    """
    slug_dir = root / "reproduction" / slug
    py_path = slug_dir / f"01_{slug}.py"
    ipynb_path = slug_dir / f"01_{slug}.ipynb"
    if not py_path.is_file():
        raise RenderError(f"{py_path} does not exist")

    subprocess.run(
        [_tool_path("jupytext"), "--to", "ipynb", py_path.name],
        cwd=slug_dir,
        check=True,
    )

    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    env["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")])
    # Unset, deliberately: a non-inline ambient MPLBACKEND (Agg is the usual
    # headless default) renders a figure-less .ipynb silently (CONTRACT §7).
    env.pop("MPLBACKEND", None)

    subprocess.run(
        [
            _tool_path("jupyter"),
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={timeout}",
            ipynb_path.name,
        ],
        cwd=slug_dir,
        check=True,
        env=env,
    )

    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))

    error_cells = _error_cells(nb)
    if error_cells:
        raise RenderError(
            f"{ipynb_path.relative_to(root)}: error output in cell(s) {error_cells} "
            "-- fix the notebook before publishing this render."
        )

    missing = unexecuted(nb)
    if missing:
        raise RenderError(
            f"{ipynb_path.relative_to(root)}: {len(missing)} code cell(s) never ran, "
            f"first at index {missing[0]} -- the run aborted (SystemExit reads as "
            "success under nbclient). Generate the missing input named in that cell "
            "and re-run."
        )

    nb.setdefault("metadata", {})["tengri_render"] = {
        "source_sha256": code_cell_source_sha256(py_path),
        "executed_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "tengri_version": _tengri_version(),
    }
    ipynb_path.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")

    return ipynb_path


def publish_slug(slug: str, ipynb_path: Path, *, root: Path = ROOT) -> Path:
    """Copy the rendered, stamped notebook and its figures to ``docs/reproduction/``.

    Parameters
    ----------
    slug : str
        Comparison folder name.
    ipynb_path : Path
        The rendered ``.ipynb`` returned by :func:`render_slug`.
    root : Path, optional
        Repository root holding ``docs/reproduction/``.

    Returns
    -------
    Path
        The published ``docs/reproduction/<slug>.ipynb`` path.
    """
    docs_dir = root / "docs" / "reproduction"
    docs_dir.mkdir(parents=True, exist_ok=True)
    docs_ipynb = docs_dir / f"{slug}.ipynb"
    shutil.copyfile(ipynb_path, docs_ipynb)

    figs_src = ipynb_path.parent / "_figs"
    if figs_src.is_dir():
        figs_dst = docs_dir / "_figs"
        figs_dst.mkdir(exist_ok=True)
        for png in figs_src.glob(f"{slug}_*.png"):
            shutil.copyfile(png, figs_dst / png.name)

    return docs_ipynb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("slug", help="comparison folder name under reproduction/")
    parser.add_argument("--timeout", type=int, default=1800, help="per-cell nbconvert timeout [s]")
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="render and stamp only; skip copying to docs/reproduction/",
    )
    args = parser.parse_args(argv)

    try:
        ipynb_path = render_slug(args.slug, timeout=args.timeout)
    except (RenderError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    print(f"OK: rendered and stamped {ipynb_path.relative_to(ROOT)}")

    if not args.no_publish:
        docs_path = publish_slug(args.slug, ipynb_path)
        print(f"OK: published {docs_path.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
