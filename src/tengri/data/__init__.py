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

from tengri._data_setup import require_remote_url
from tengri.registry import _RegistryTable

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


def list_remote_ssps() -> _RegistryTable:
    """List the SSP files available at :data:`SSP_CATALOG_URL`.

    Returns
    -------
    _RegistryTable
        One row per catalog entry, with column ``name`` (the filename,
        e.g. ``"fsps_prsc_miles_chabrier.h5"``), in the order the server
        returned them, with duplicates removed. ``.names()`` returns the
        plain list of strings this gave before #1574.

    Raises
    ------
    urllib.error.URLError
        If the catalog is unreachable.

    Notes
    -----
    Makes a live HTTP request on every call — this is the one discovery
    verb that touches the network.

    Parses the Apache autoindex page with a single regex; it does not
    validate that every linked file is a usable SSP grid.

    Examples
    --------
    >>> list_remote_ssps().names()
    """
    # B310: scheme restricted to http/https by require_remote_url.
    require_remote_url(SSP_CATALOG_URL)
    with urllib.request.urlopen(SSP_CATALOG_URL, timeout=30) as resp:  # nosec B310
        html = resp.read().decode("utf-8", errors="replace")
    seen: set[str] = set()
    rows: list[dict] = []
    for match in re.finditer(r'href="([^"]+\.h5)"', html):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            rows.append(
                {
                    "name": name,
                    "kind": "ssp",
                    "use": f"tengri.data.download_ssp({name!r})",
                }
            )
    return _RegistryTable(rows)


def local_ssp_path(name: str, dest_dir: str | os.PathLike = "data") -> Path:
    """Return the local filesystem path where :func:`download_ssp` writes ``name``."""
    return Path(dest_dir) / name


def download_ssp(
    name: str,
    dest_dir: str | os.PathLike | None = None,
    *,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """Download a single SSP file from :data:`SSP_CATALOG_URL`.

    .. deprecated::
        Use :func:`tengri.download_ssp`, which this now delegates to. The two
        were separate implementations of one job, reachable under the same name
        with incompatible signatures (``dest_dir``/``overwrite`` here versus
        ``dest``/``force`` there) — and only the other honored
        ``$TENGRI_DATA_DIR``, so a file fetched through this one could land
        somewhere the loaders never looked.

    Parameters
    ----------
    name : str
        Filename in the catalog (e.g. ``"fsps_prsc_miles_chabrier.h5"``), or a
        short identifier from ``tengri.list_known_ssps()``. Use
        :func:`list_remote_ssps` to discover catalog filenames.
    dest_dir : str or os.PathLike, optional
        Directory to write into. Created if missing. Default ``None`` →
        ``$TENGRI_DATA_DIR`` if set, else ``data/``.
    overwrite : bool, optional
        If False (default), an existing file at the target path is kept and its
        path is returned without re-downloading.
    progress : bool, optional
        If True (default), report download progress. False for clean logs.

    Returns
    -------
    pathlib.Path
        Path to the downloaded (or pre-existing) file.

    Raises
    ------
    KeyError
        If ``name`` is neither a known identifier nor a ``.h5`` filename.
    ValueError
        If ``name`` is a path rather than a bare filename.
    RuntimeError
        If the HTTP download fails.

    Examples
    --------
    >>> import tengri  # doctest: +SKIP
    >>> path = tengri.download_ssp("fsps_prsc_miles_chabrier")  # doctest: +SKIP
    """
    import warnings

    from tengri._data_setup import download_ssp as _download_ssp

    warnings.warn(
        "tengri.data.download_ssp is deprecated; use tengri.download_ssp "
        "instead. It accepts the same catalog filenames plus the short names "
        "from tengri.list_known_ssps(), and honors $TENGRI_DATA_DIR for both "
        "reads and writes.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _download_ssp(name, dest=dest_dir, force=overwrite, progress=progress)
