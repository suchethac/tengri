# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""SSP and template data fetch helpers.

Most tengri recipes require an SSP file that may not be present in a fresh
checkout. Four of the six recipes in :mod:`tengri.recipes` use the Cue
nebular backend, which **requires a bare-stellar SSP** (no baked-in
nebular emission) — without one they raise
:class:`~tengri.components.nebular.cue.CueWNESSPError` on
``SEDModel.build(...)``.

This module exposes a tiny helper to list and fetch the SSP files hosted
at the canonical catalog::

    from tengri.data import list_remote_ssps, download_ssp

    list_remote_ssps()  # show what's available
    path = download_ssp("fsps_prsc_miles_chabrier.h5")
    # → "data/fsps_prsc_miles_chabrier.h5"

The download is a single HTTP GET with progress reporting; no Python
package dependencies beyond the standard library. Files are written to
``./data/`` by default and skipped if they already exist (override with
``overwrite=True``).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import urllib.request
from collections.abc import Iterable
from pathlib import Path

__all__ = [
    "SSP_CATALOG_URL",
    "download_ssp",
    "list_remote_ssps",
    "local_ssp_path",
]

#: Canonical hosted catalog of tengri-compatible SSP files. Indexes a
#: directory of bare-stellar SSP HDF5 files keyed by ``<isochrone>_<lib>_<imf>.h5``
#: (e.g. ``fsps_prsc_miles_chabrier.h5``). The wNE (with-Nebular-Emission)
#: variants used by the BakedIn nebular backend are *not* shipped from
#: here; obtain those from your DSPS / FSPS install or generate via the
#: scripts under ``tools/``.
SSP_CATALOG_URL: str = "https://halos.as.arizona.edu/suchethacooray/ssp-spectra/"


def list_remote_ssps() -> list[str]:
    """Return the list of SSP filenames available at :data:`SSP_CATALOG_URL`.

    Returns
    -------
    list[str]
        Filenames in the catalog (e.g. ``"fsps_prsc_miles_chabrier.h5"``),
        in the order the server returned them, with duplicates removed.

    Raises
    ------
    urllib.error.URLError
        If the catalog is unreachable.

    Notes
    -----
    This parses the Apache autoindex page with a single regex — it does
    not validate that every linked file is a usable SSP grid.
    """
    with urllib.request.urlopen(SSP_CATALOG_URL, timeout=30) as resp:
        html = resp.read().decode("utf-8", errors="replace")
    seen: set[str] = set()
    out: list[str] = []
    for match in re.finditer(r'href="([^"]+\.h5)"', html):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


def local_ssp_path(name: str, dest_dir: str | os.PathLike = "data") -> Path:
    """Return the local filesystem path where :func:`download_ssp` writes ``name``."""
    return Path(dest_dir) / name


def _format_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TiB"


def download_ssp(
    name: str,
    dest_dir: str | os.PathLike = "data",
    *,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Download a single SSP file from :data:`SSP_CATALOG_URL`.

    Parameters
    ----------
    name : str
        Filename in the catalog (e.g. ``"fsps_prsc_miles_chabrier.h5"``).
        Use :func:`list_remote_ssps` to discover valid names.
    dest_dir : str or os.PathLike, optional
        Directory to write into. Created if missing. Default: ``"data"``
        (matches the layout the SSP loaders expect by default).
    overwrite : bool, optional
        If False (default), an existing file at the target path is kept
        and its path is returned without re-downloading. If True, any
        existing file is replaced.
    progress : bool, optional
        If True (default), print a one-line progress update to stderr
        every ~5%. Set to False for clean log output.

    Returns
    -------
    pathlib.Path
        Absolute path to the downloaded (or pre-existing) file.

    Raises
    ------
    urllib.error.URLError
        If the catalog or specific file is unreachable.
    ValueError
        If ``name`` contains a path separator (defensive against
        accidental traversal).

    Examples
    --------
    >>> from tengri.data import download_ssp  # doctest: +SKIP
    >>> path = download_ssp("fsps_prsc_miles_chabrier.h5")  # doctest: +SKIP
    >>> from tengri import load_ssp_data, recipes, SEDModel  # doctest: +SKIP
    >>> ssp = load_ssp_data(str(path))  # doctest: +SKIP
    >>> model = SEDModel.build(ssp_data=ssp, **recipes.star_forming_photometry())  # doctest: +SKIP
    """
    if "/" in name or "\\" in name or name.startswith("."):
        raise ValueError(
            f"download_ssp(name={name!r}): name must be a bare filename, "
            f"not a path. Use list_remote_ssps() to discover valid names."
        )

    target = local_ssp_path(name, dest_dir).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not overwrite:
        return target

    url = SSP_CATALOG_URL.rstrip("/") + "/" + name
    tmp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        # Stream to a side file so an interrupted download never leaves a
        # half-written SSP that load_ssp_data would happily try to read.
        with tmp.open("wb") as fh:
            if progress and total > 0:
                downloaded = 0
                next_pct = 5
                while True:
                    chunk = resp.read(1 << 20)  # 1 MiB
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    pct = downloaded * 100 // total
                    if pct >= next_pct:
                        sys.stderr.write(
                            f"\r  download {name}: "
                            f"{_format_bytes(downloaded)} / {_format_bytes(total)} "
                            f"({pct:>3d}%)"
                        )
                        sys.stderr.flush()
                        next_pct = pct + 5
                sys.stderr.write("\n")
            else:
                shutil.copyfileobj(resp, fh)

    tmp.replace(target)
    return target


def download_ssps(
    names: Iterable[str],
    dest_dir: str | os.PathLike = "data",
    *,
    overwrite: bool = False,
    progress: bool = True,
) -> list[Path]:
    """Download multiple SSP files; convenience wrapper around :func:`download_ssp`.

    Returns
    -------
    list[pathlib.Path]
        One path per input ``name``, in order.
    """
    return [download_ssp(name, dest_dir, overwrite=overwrite, progress=progress) for name in names]
