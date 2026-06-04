#!/usr/bin/env python3
"""Fetch AGNfitter-rX model pickles directly from GitHub (no install needed).

The AGNfitter-rX model library ships its preprocessed template pickles inside
the upstream git repository under ``models/TORUS/`` and ``models/BBB/``. They
are tracked files, so each ``build_*_grid.py`` script can obtain the one pickle
it needs straight from ``raw.githubusercontent.com`` — a contributor does not
have to clone or install AGNfitter to regenerate (and thereby verify) the
vendored HDF5 grids committed under ``data/``.

The URL is pinned to the ``AGNfitter-rX_v0.1`` tag so a rebuild years from now
reads byte-identical upstream input to what produced the committed grid.

Downloads are cached under ``~/.cache/tengri_agnfitter/`` (override with the
``TENGRI_AGNFITTER_CACHE`` environment variable) so repeated builds reuse a
single local copy.

References
----------
- Martínez-Ramírez, L. N. et al., "AGNfitter-rx: Modeling the radio-to-X-ray
  spectral energy distributions of AGNs," A&A 688, A46 (2024).
  arXiv:2405.12111. DOI: 10.1051/0004-6361/202449329.
- Upstream repository: https://github.com/GabrielaCR/AGNfitter
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

#: Upstream repository and pinned ref. The pinned tag keeps rebuilds reproducible.
AGNFITTER_REPO = "GabrielaCR/AGNfitter"
AGNFITTER_REF = "AGNfitter-rX_v0.1"

_RAW_BASE = "https://raw.githubusercontent.com"


def _cache_dir() -> Path:
    """Return the local download cache directory, creating it if absent."""
    root = os.environ.get("TENGRI_AGNFITTER_CACHE")
    cache = Path(root) if root else Path.home() / ".cache" / "tengri_agnfitter"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def raw_url(repo_relpath: str) -> str:
    """Build the raw-content URL for a repo-relative path at the pinned ref.

    Parameters
    ----------
    repo_relpath : str
        Path inside the AGNfitter repo, e.g. ``models/TORUS/S04.pickle``.

    Returns
    -------
    str
        Full ``raw.githubusercontent.com`` URL pinned to ``AGNFITTER_REF``.
    """
    return f"{_RAW_BASE}/{AGNFITTER_REPO}/{AGNFITTER_REF}/{repo_relpath.lstrip('/')}"


def fetch(repo_relpath: str, *, force: bool = False) -> Path:
    """Download an AGNfitter pickle to the local cache and return its path.

    Parameters
    ----------
    repo_relpath : str
        Path inside the AGNfitter repo, e.g. ``models/TORUS/CAT3D_mean_3p.pickle``.
    force : bool, optional
        Re-download even if a cached copy already exists (default ``False``).

    Returns
    -------
    Path
        Local path to the cached pickle.

    Raises
    ------
    urllib.error.URLError
        If the download fails (network error, 404, etc.).
    """
    dest = _cache_dir() / Path(repo_relpath).name
    if dest.is_file() and not force:
        return dest

    url = raw_url(repo_relpath)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {url}\n        -> {dest}")
    with urllib.request.urlopen(url) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    print(f"  cached {dest.stat().st_size / 1e6:.2f} MB")
    return dest


def resolve(local_path: Path, repo_relpath: str, *, download: bool) -> Path:
    """Resolve the pickle path, downloading from GitHub when requested or missing.

    Resolution order:

    1. If ``download`` is set, fetch from GitHub (cached) and use that.
    2. Else if ``local_path`` exists, use it (a manual clone / custom path).
    3. Else fall back to the cached download (auto), printing a note.

    Parameters
    ----------
    local_path : Path
        User-supplied ``--input`` path (e.g. a manual AGNfitter clone).
    repo_relpath : str
        Path inside the AGNfitter repo for autodownload.
    download : bool
        If ``True``, always (re)download instead of reading ``local_path``.

    Returns
    -------
    Path
        A path to a readable pickle file.
    """
    if download:
        return fetch(repo_relpath)
    if local_path.is_file():
        return local_path
    print(
        f"note: {local_path} not found — autodownloading {repo_relpath} from "
        f"{AGNFITTER_REPO}@{AGNFITTER_REF} (pass --download to force, or --input "
        "to point at a local AGNfitter clone)."
    )
    return fetch(repo_relpath)
