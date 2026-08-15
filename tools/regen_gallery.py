#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Regenerate selected gallery examples without stripping the rest.

Why this exists
---------------

Running a plain ``make html`` while any example is stale **degrades the
gallery** (#1236). Sphinx-gallery rewrites a page whose source md5 no longer
matches its committed stamp, but ``filename_pattern`` in ``docs/conf.py``
excludes from execution every example that already has a committed figure. A
page that is rewritten but never executed comes back without anything
execution produced — silently, with the build exiting 0. Measured while 60
examples were stale: one single-example build changed 195 files and deleted
45,204 lines.

A page that is already fresh is not rewritten at all, so the damage is
proportional to the drift, and a full build over a zero-drift gallery was
measured to change nothing. This script is what you need while anything is
stale — which, by definition, is exactly when you are regenerating.

This script makes a *scoped* regeneration safe:

1. snapshot ``docs/auto_examples/``;
2. build with ``TENGRI_GALLERY_ONLY`` set, so only the requested examples
   execute (that env var replaces the skip-list pattern in ``conf.py``);
3. restore every file that does **not** belong to a requested example.

Step 3 is the fence. Without it the requested examples come back correct and
the other ~270 come back hollowed out.

Usage
-----

::

    python tools/regen_gallery.py --stale              # every drifted example
    python tools/regen_gallery.py plot_agn_hierarchy   # one example, by basename
    python tools/regen_gallery.py --stale --dry-run    # list targets, build nothing

Requires the optional data grids (SSP, SKIRTOR, astrodust, ...): the examples
actually execute. CI cannot run this — it has neither the grids nor the RAM,
which is why the rendered gallery is committed in the first place.

Batch in single digits. Every requested example runs inside **one** sphinx
process, and the per-example JAX/SSP allocations are not fully released between
them, so the peak grows with the batch. Forty at once was OOM-killed on a 48 GB
machine (``sphinx exit=-9``); eight at a time completes. The snapshot in step 3
survives the kill, so a killed batch leaves the gallery intact rather than
hollowed out — but it also does no work, so prefer several small batches over
one large one. ``--stale`` on a large drift is exactly the case to split up.
"""

from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EXAMPLES = REPO / "examples"
AUTO = REPO / "docs" / "auto_examples"
DOCS = REPO / "docs"

# A whole-gallery timing table, written from whatever this build executed. A
# scoped build therefore rewrites it to describe only the handful of examples
# it ran while still captioning it "for 279 files" — a false summary. It lives
# outside auto_examples/, so the page fence does not reach it; restore it
# explicitly and leave the table to full builds.
BUILD_ARTIFACTS = (DOCS / "sg_execution_times.rst",)


def _stale_basenames() -> list[str]:
    """Basenames of examples whose committed render has drifted."""
    sys.path.insert(0, str(REPO / "tools"))
    from check_gallery_fresh import stale_examples

    stale, unrendered = stale_examples()
    return [Path(rel).stem for rel in (*stale, *unrendered)]


def _owned_by_targets(path: Path, targets: set[str]) -> bool:
    """True when ``path`` is an output of one of the target examples.

    Sphinx-gallery names every artifact after the example basename — the
    copied ``<name>.py``, its ``.py.md5`` stamp, ``<name>.rst``,
    ``<name>.ipynb``, ``<name>.zip``, ``<name>.codeobj.json``, and the figures
    ``images/sphx_glr_<name>_NNN.png`` (plus ``images/thumb/``). Anything that
    does not carry a target's basename belongs to an example we did not run,
    so it must be restored rather than kept.

    Matched exactly, never by prefix: basenames nest (``plot_agn_disc`` is a
    prefix of ``plot_agn_disc_compare``), and a prefix rule would mark the
    longer example's files as owned by the shorter target — keeping a page the
    build hollowed out instead of restoring it.
    """
    name = path.name
    for t in targets:
        if name in {f"{t}.py", f"{t}.py.md5", f"{t}.rst", f"{t}.ipynb", f"{t}.zip"}:
            return True
        if name == f"{t}.codeobj.json" or name == f"{t}.py.md5":
            return True
        if re.fullmatch(rf"sphx_glr_{re.escape(t)}_(?:\d+|thumb)\.(?:png|svg)", name):
            return True
    return False


def _snapshot(dst: Path) -> None:
    shutil.copytree(AUTO, dst)
    for artifact in BUILD_ARTIFACTS:
        if artifact.is_file():
            shutil.copy2(artifact, dst.parent / artifact.name)


def _restore_build_artifacts(snapshot: Path) -> int:
    """Put back whole-build files a scoped build would leave partially true."""
    restored = 0
    for artifact in BUILD_ARTIFACTS:
        saved = snapshot.parent / artifact.name
        if saved.is_file() and not (
            artifact.is_file() and filecmp.cmp(saved, artifact, shallow=False)
        ):
            shutil.copy2(saved, artifact)
            restored += 1
    return restored


def _restore_untargeted(snapshot: Path, targets: set[str]) -> tuple[int, int, int]:
    """Undo the build's collateral damage.

    Returns ``(restored, removed, kept)`` — files put back from the snapshot,
    files the build created that belong to no target, and files left as the
    build wrote them.
    """
    restored = removed = kept = 0

    # Files the build changed or deleted -> put the snapshot copy back.
    for src in sorted(snapshot.rglob("*")):
        if not src.is_file():
            continue
        rel = src.relative_to(snapshot)
        live = AUTO / rel
        if _owned_by_targets(live, targets):
            kept += 1
            continue
        if live.is_file() and filecmp.cmp(src, live, shallow=False):
            continue
        live.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, live)
        restored += 1

    # Files the build created that were not in the snapshot and belong to no
    # target -> remove them, so a scoped regen leaves no stray pages behind.
    for live in sorted(AUTO.rglob("*")):
        if not live.is_file():
            continue
        rel = live.relative_to(AUTO)
        if (snapshot / rel).is_file() or _owned_by_targets(live, targets):
            continue
        live.unlink()
        removed += 1

    return restored, removed, kept


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("examples", nargs="*", help="example basenames, e.g. plot_agn_hierarchy")
    ap.add_argument("--stale", action="store_true", help="regenerate every drifted example")
    ap.add_argument("--dry-run", action="store_true", help="list targets and exit")
    args = ap.parse_args()

    targets = list(args.examples)
    if args.stale:
        targets.extend(_stale_basenames())
    targets = sorted(set(targets))

    if not targets:
        print("nothing to do: pass example basenames or --stale")
        return 0

    known = {p.stem for p in EXAMPLES.rglob("plot_*.py")}
    unknown = [t for t in targets if t not in known]
    if unknown:
        print("unknown example basename(s): " + ", ".join(unknown))
        return 2

    print(f"regenerating {len(targets)} example(s):")
    for t in targets:
        print(f"  {t}")
    if args.dry_run:
        return 0

    env = dict(os.environ)
    env["TENGRI_GALLERY_ONLY"] = ",".join(targets)
    # conf.py switches to the committed renders when CI is set; we are the
    # machine that produces them, so make sure execution is on.
    env.pop("CI", None)
    # sphinx-gallery executes each script from the gallery output directory, not
    # the repo root, so a relative ``data/filters`` lookup misses and the filter
    # loader falls through to an SVO download -- which fails without astroquery
    # and network. The script runs fine by hand and only breaks under the build,
    # which makes it a confusing failure. Pin the data directory absolutely.
    env.setdefault("TENGRI_DATA_DIR", str(REPO / "data"))

    # The snapshot deliberately outlives a crash. These builds execute real
    # science examples and can be OOM-killed mid-flight; an auto-deleting
    # temp dir would take the only copy of the untouched pages with it and
    # leave the gallery hollowed out. Removed only on a clean finish, and its
    # path is printed otherwise.
    workdir = Path(tempfile.mkdtemp(prefix="regen_gallery_"))
    snapshot = workdir / "auto_examples"
    out = workdir / "html"
    _snapshot(snapshot)

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "sphinx", "-b", "html", str(DOCS), str(out)],
            env=env,
        )
        returncode = proc.returncode
    except KeyboardInterrupt:
        print("\ninterrupted — restoring untargeted pages")
        returncode = 130
    finally:
        restored, removed, kept = _restore_untargeted(snapshot, set(targets))
        restored += _restore_build_artifacts(snapshot)

    print(
        f"\nsphinx exit={returncode}   "
        f"kept {kept} target file(s), restored {restored}, removed {removed}"
    )

    if returncode == 0:
        shutil.rmtree(workdir, ignore_errors=True)
    else:
        print(f"snapshot kept for recovery: {snapshot}")
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
