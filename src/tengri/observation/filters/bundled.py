# SPDX-License-Identifier: BSD-3-Clause
"""Measured filter curves shipped inside the package.

Most tengri bandpasses are fetched from the SVO Filter Profile Service and
cached, and bands that are really a receiver window or a detector energy range
are served as top-hats from :mod:`tengri.observation.filters.synthetic`. This
module covers the third case: a real, measured curve that SVO does not serve in
the form the data were taken in, so the numbers have to travel with the package.

7DT is the worked example. The 23 curves are total system response, detector QE
and optics included, which is what the program's ADU were measured through. A
filter-glass-only curve fetched from elsewhere would be a different quantity.
See ``tengri/data/filters_7dt/PROVENANCE.md``.

Resolution order in :func:`tengri.observation.filters.load_filter` puts these
*after* both user routes and *before* the SVO registry, so a user can still
shadow a bundled curve with their own, exactly as they can shadow an SVO one.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import jax.numpy as jnp

from tengri.observation.photometry import FilterCurve

__all__ = ["BUNDLED_FILTER_REGISTRY", "load_bundled_filter"]

# Package holding the .dat files, and the bands it provides.
_SEVENDT_PACKAGE = "tengri.data.filters_7dt"

_SEVENDT_BANDS: tuple[str, ...] = (
    "g",
    "r",
    "i",
    *(f"m{lam}" for lam in range(400, 900, 25)),
)

# Registry: filter name -> (package, resource file name).
BUNDLED_FILTER_REGISTRY: dict[str, tuple[str, str]] = {
    f"7dt_{band}": (_SEVENDT_PACKAGE, f"7dt_{band}.dat") for band in _SEVENDT_BANDS
}


def _resource_path(name: str) -> Path:
    """Filesystem path of a bundled curve, via ``importlib.resources``.

    Parameters
    ----------
    name : str
        Registered filter name, e.g. ``"7dt_m400"``.

    Returns
    -------
    pathlib.Path
        Path to the two-column ``.dat`` file.

    Raises
    ------
    KeyError
        If *name* is not a bundled curve.
    """
    package, filename = BUNDLED_FILTER_REGISTRY[name]
    return Path(str(resources.files(package).joinpath(filename)))


def load_bundled_filter(name: str) -> FilterCurve:
    """Load a filter curve that ships inside the package.

    Parameters
    ----------
    name : str
        Registered bundled name, e.g. ``"7dt_m400"``.

    Returns
    -------
    FilterCurve
        Immutable curve with wavelength in Angstrom.

    Raises
    ------
    KeyError
        If *name* is not in :data:`BUNDLED_FILTER_REGISTRY`.

    Notes
    -----
    No unit heuristic runs here. Bundled curves are converted to Angstrom once,
    at build time by ``tools/build_7dt_filter_curves.py``, and their conversion
    is pinned by a test rather than re-guessed on every load.

    Examples
    --------
    >>> from tengri.observation.filters.bundled import load_bundled_filter
    >>> fc = load_bundled_filter("7dt_m400")
    >>> bool(fc.wave.min() > 3000.0)
    True
    """
    from tengri.observation.filters.custom import _load_filter_file, _sanitize_filter_curve

    wave, trans = _load_filter_file(_resource_path(name))
    wave, trans = _sanitize_filter_curve(wave, trans)
    return FilterCurve(wave=jnp.array(wave), trans=jnp.array(trans), name=name)


def list_bundled_filters() -> tuple[str, ...]:
    """Return every bundled filter name, sorted.

    Returns
    -------
    tuple of str
        Registered names, e.g. ``("7dt_g", "7dt_i", ...)``.
    """
    return tuple(sorted(BUNDLED_FILTER_REGISTRY))
