# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the published docs copy of each reproduction notebook is in sync.

The reproduction notebooks are authored under ``reproduction/<name>/`` and
*published* to the docs site as committed copies at
``docs/reproduction/<name>.ipynb`` (nbsphinx renders them from stored
outputs — ``nbsphinx_execute = "never"``). There is no auto-sync step, so
when a source notebook is re-rendered (e.g. after a physics fix) its docs
copy can silently drift and the live site keeps showing the old figures.

This bit us once: the §4 attenuation panel was fixed in the Prospector
source notebook (#552) but the docs copy from #551 kept the pre-fix
figure, so the site showed the wrong 2175 Å bump until #555.

This test pins the invariant: every published docs notebook must embed
exactly the same image outputs, in the same order, as its source. If you
re-render a ``reproduction/<name>/01_<name>.ipynb``, re-copy it to
``docs/reproduction/<name>.ipynb`` (and the ``_figs/`` PNGs) in the same
PR, or this fails.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

# Repo root: tests/contract/<this file> → parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]
_REPRO_DIR = _REPO_ROOT / "reproduction"
_DOCS_DIR = _REPO_ROOT / "docs" / "reproduction"


def _published_slugs() -> list[str]:
    """Reproduction slugs that have a published docs copy on disk."""
    if not _DOCS_DIR.is_dir():
        return []
    return sorted(
        p.stem
        for p in _DOCS_DIR.glob("*.ipynb")
        if (_REPRO_DIR / p.stem / f"01_{p.stem}.ipynb").is_file()
    )


def _embedded_figure_hashes(ipynb_path: Path) -> list[str]:
    """SHA-1 of every ``image/png`` output, in document order."""
    nb = json.loads(ipynb_path.read_text(encoding="utf-8"))
    hashes: list[str] = []
    for cell in nb.get("cells", []):
        for out in cell.get("outputs", []):
            png = out.get("data", {}).get("image/png")
            if png:
                hashes.append(hashlib.sha1(png.encode("utf-8")).hexdigest())
    return hashes


@pytest.mark.parametrize("slug", _published_slugs())
def test_docs_copy_figures_match_source(slug: str) -> None:
    """``docs/reproduction/<slug>.ipynb`` embeds the same figures as its source."""
    source = _REPRO_DIR / slug / f"01_{slug}.ipynb"
    docs = _DOCS_DIR / f"{slug}.ipynb"

    src_figs = _embedded_figure_hashes(source)
    docs_figs = _embedded_figure_hashes(docs)

    assert docs_figs == src_figs, (
        f"docs/reproduction/{slug}.ipynb is out of sync with "
        f"reproduction/{slug}/01_{slug}.ipynb "
        f"(source has {len(src_figs)} embedded figures, docs has "
        f"{len(docs_figs)}; content differs). Re-copy the re-rendered "
        f"source notebook into docs/reproduction/ in this PR — the docs "
        f"site renders the committed copy, not the source."
    )


def test_at_least_one_published_notebook() -> None:
    """Guard: the discovery glob actually finds the published notebooks."""
    slugs = _published_slugs()
    assert slugs, (
        "No published reproduction notebooks found under docs/reproduction/ "
        "with a matching reproduction/<slug>/01_<slug>.ipynb source — the "
        "sync check would silently pass on an empty set."
    )
