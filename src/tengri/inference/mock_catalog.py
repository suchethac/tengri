# SPDX-License-Identifier: BSD-3-Clause
"""The result of :meth:`Catalog.simulate`, mock observables for N galaxies (#1396).

Table-out, matching the leg :class:`CatalogPosterior` already provides: the
``to_table`` dict is a duck-type match for ``ingest_catalog``'s input, so a
simulated catalog can be written to parquet/FITS and read straight back in as
data, which is the whole point of generating one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["MockCatalog"]


@dataclass(frozen=True)
class MockCatalog:
    """Simulated observables for a catalog of galaxies.

    Parameters
    ----------
    photometry: ndarray, shape (N, n_filters)
        Observed-frame spectral flux density [erg/s/cm²/Hz].
    filter_names: tuple of str
        Band names, in ``photometry`` column order.
    lines: dict, optional
        ``{line_name: (N,) ndarray}`` observed line fluxes [erg/s/cm²].
    properties: dict, optional
        ``{property_name: (N,) ndarray}`` derived quantities, e.g.
        ``stellar_mass`` [Msun].

    Notes
    -----
    Frozen: a simulated catalog is a record of one evaluation, so mutating it in
    place would silently decouple it from the histories that produced it.
    """

    photometry: np.ndarray
    filter_names: tuple[str, ...]
    lines: dict[str, np.ndarray] = field(default_factory=dict)
    properties: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def n_galaxies(self) -> int:
        """Number of galaxies in the mock catalog."""
        return int(self.photometry.shape[0])

    def to_table(self) -> dict:
        """Export as a flat column dict (round-trips through ``ingest_catalog``).

        Returns
        -------
        dict[str, ndarray]
            One ``(N,)`` column per band (named by filter), per line, and per
            property. Flat by construction, so it survives a parquet/FITS round
            trip.

        Raises
        ------
        ValueError
            If a line or property name collides with a band name, silently
            overwriting a flux column would corrupt the file rather than fail.
        """
        table = {
            name: np.asarray(self.photometry)[:, i] for i, name in enumerate(self.filter_names)
        }
        for source in (self.lines, self.properties):
            for name, values in source.items():
                if name in table:
                    raise ValueError(
                        f"column name collision on {name!r}: a line or property "
                        f"shares a band name, and writing it would overwrite the "
                        f"flux column. Rename the filter or drop the request."
                    )
                table[name] = np.asarray(values)
        return table
