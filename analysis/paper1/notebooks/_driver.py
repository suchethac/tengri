# SPDX-License-Identifier: BSD-3-Clause
"""Shared driver for the paper's figure notebooks.

Each notebook renders one figure by running its script under ``analysis/paper1``
as a module, in the environment the figures README prescribes: the checkout's
``src`` for tengri, ``analysis`` for the ``paper1`` package, and
``analysis/paper1`` because ``configs.py`` imports ``config_metadata`` by bare
name. The plotting code stays single-sourced in the scripts; a notebook adds
input checks, provenance, the rendered image, and an optional copy into the
paper's ``figures/`` directory.

Environment overrides:

``PAPER1_CHECKOUT``   render from another checkout (e.g. the paper's pinned
                     branch) instead of the one holding this notebook.
``PAPER1_PYTHON``     interpreter to run the script with (default: this kernel's).
``PAPER_FIGURES_DIR`` if set, the rendered PDF is copied there.
``JAX_PLATFORMS``     defaults to ``cpu``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_MARKERS = ("src/tengri", "analysis/paper1/configs.py")


@dataclass(frozen=True)
class Paths:
    checkout: Path
    analysis: Path
    paper1: Path
    results: Path
    figures: Path


def find_checkout(start: Path | None = None) -> Path:
    """Locate the tengri checkout that carries ``analysis/paper1``."""
    override = os.environ.get("PAPER1_CHECKOUT")
    if override:
        root = Path(override).expanduser().resolve()
    else:
        here = (start or Path.cwd()).resolve()
        root = next(
            (c for c in (here, *here.parents) if all((c / m).exists() for m in _MARKERS)),
            None,
        )
        if root is None:
            raise FileNotFoundError(
                "no checkout with src/tengri and analysis/paper1 above "
                f"{here}; set PAPER1_CHECKOUT to the checkout holding the paper's branch"
            )
    missing = [m for m in _MARKERS if not (root / m).exists()]
    if missing:
        raise FileNotFoundError(f"{root} is missing {missing}")
    return root


def paths(checkout: Path) -> Paths:
    analysis = checkout / "analysis"
    paper1 = analysis / "paper1"
    return Paths(checkout, analysis, paper1, paper1 / "results", paper1 / "figures")


def environment(p: Paths) -> dict[str, str]:
    pythonpath = os.pathsep.join(str(d) for d in (p.checkout / "src", p.analysis, p.paper1))
    return {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu"),
        "MPLBACKEND": "Agg",
    }


def interpreter() -> str:
    return os.environ.get("PAPER1_PYTHON", sys.executable)


def check_inputs(p: Paths, relative: list[str]) -> list[Path]:
    """Fail fast, naming every missing input, before any script runs."""
    found, missing = [], []
    for rel in relative:
        path = p.paper1 / rel
        present = path.is_file() or (path.is_dir() and any(path.iterdir()))
        (found if present else missing).append(path)
    if missing:
        raise FileNotFoundError(
            "missing or empty inputs:\n  " + "\n  ".join(str(m) for m in missing)
        )
    for path in found:
        print(f"input  {path.relative_to(p.checkout)}")
    return found


def run_module(p: Paths, module: str, *args: object, tail: int = 3000) -> str:
    """Run ``python -m <module> <args>`` from the checkout root; raise on failure."""
    cmd = [interpreter(), "-m", module, *map(str, args)]
    print("$", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=p.checkout, env=environment(p), text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout[-tail:])
    if proc.returncode != 0:
        print(proc.stderr[-tail:], file=sys.stderr)
        raise RuntimeError(f"{module} exited with {proc.returncode}")
    return proc.stdout


def provenance(p: Paths) -> dict[str, str]:
    """Record what rendered the figure: commit, tree state, interpreter, library versions."""

    def git(*a: str) -> str:
        return subprocess.run(
            ["git", *a], cwd=p.checkout, text=True, capture_output=True
        ).stdout.strip()

    probe = "import jax, matplotlib, tengri; print(tengri.__version__, jax.__version__, matplotlib.__version__)"
    versions = subprocess.run(
        [interpreter(), "-c", probe],
        cwd=p.checkout,
        env=environment(p),
        text=True,
        capture_output=True,
    )
    tengri_v, jax_v, mpl_v = [*versions.stdout.split(), "?", "?", "?"][:3]
    info = {
        "checkout": str(p.checkout),
        "commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty_paths": str(
            len(git("status", "--porcelain", "--", "src", "analysis/paper1").splitlines())
        ),
        "python": interpreter(),
        "tengri": tengri_v,
        "jax": jax_v,
        "matplotlib": mpl_v,
    }
    for key, value in info.items():
        print(f"{key:12s} {value}")
    return info


def show(p: Paths, stem: str) -> None:
    """Display the rendered figure: its PNG if the script wrote one, else the PDF."""
    from IPython.display import IFrame, Image, display

    png, pdf = p.figures / f"{stem}.png", p.figures / f"{stem}.pdf"
    if not pdf.exists():
        raise FileNotFoundError(f"{pdf} was not written")
    print(f"output {pdf.relative_to(p.checkout)}  ({pdf.stat().st_size / 1024:.0f} kB)")
    if not png.exists() and shutil.which("pdftoppm"):
        subprocess.run(
            ["pdftoppm", "-png", "-r", "110", "-singlefile", str(pdf), str(pdf.with_suffix(""))],
            check=False,
        )
    if png.exists():
        display(Image(filename=str(png)))
    else:
        display(IFrame(str(pdf), width=900, height=600))


def rename_output(p: Paths, src_stem: str, dst_stem: str) -> None:
    """Copy a script's output under the name the paper includes it by."""
    for ext in (".pdf", ".png"):
        src = p.figures / f"{src_stem}{ext}"
        if src.exists():
            shutil.copy2(src, p.figures / f"{dst_stem}{ext}")
            print(f"copied {src.name} -> {dst_stem}{ext}")


def copy_to_paper(p: Paths, filename: str) -> None:
    """Copy the PDF into the paper's figures directory when PAPER_FIGURES_DIR is set."""
    target_dir = os.environ.get("PAPER_FIGURES_DIR")
    if not target_dir:
        print("PAPER_FIGURES_DIR not set; PDF left under analysis/paper1/figures")
        return
    target = Path(target_dir).expanduser() / filename
    shutil.copy2(p.figures / filename, target)
    print(f"copied -> {target}")
