# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Listing of the SSP files hosted in the public catalog.

Downloading is :func:`tengri.download_ssp`, the one implementation, in
``tengri._data_setup``. This module only lists what the catalog serves::

    import tengri
    from tengri.data import list_remote_ssps

    list_remote_ssps()  # show what's available
    path = tengri.download_ssp("fsps_prsc_miles_chabrier.h5")
    # → "data/fsps_prsc_miles_chabrier.h5"

The catalog is an Apache directory index, and the listing returns every
``.h5`` file it links to. Any filename it shows can be passed to
``tengri.download_ssp``, whether or not it also has a short name in
``tengri.list_known_ssps()``.
"""

from __future__ import annotations

import re
import urllib.request

from tengri._data_setup import SSP_BASE_URL, require_remote_url
from tengri.registry import _RegistryTable

__all__ = [
    "list_remote_ssps",
]


def list_remote_ssps() -> _RegistryTable:
    """List the SSP files available at ``tengri._data_setup.SSP_BASE_URL``.

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
    Makes a live HTTP request on every call: this is the one discovery
    verb that touches the network.

    Parses the Apache autoindex page with a single regex; it does not
    validate that every linked file is a usable SSP grid.

    Examples
    --------
    >>> list_remote_ssps().names()
    """
    # B310: scheme restricted to http/https by require_remote_url.
    require_remote_url(SSP_BASE_URL)
    with urllib.request.urlopen(SSP_BASE_URL, timeout=30) as resp:  # nosec B310
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
                    "use": f"tengri.download_ssp({name!r})",
                }
            )
    return _RegistryTable(rows)
