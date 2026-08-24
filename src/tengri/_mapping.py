# SPDX-License-Identifier: BSD-3-Clause
"""The read-only view shared by the three property catalogs (#1431).

``PropertyCatalog`` (a prediction's properties), ``PosteriorProperties`` (a
fit's, over the sample axis) and ``CatalogProperties`` (a catalog's, over the
galaxy axis) are the same idea at three scales. Each supplied its own
``keys`` / ``to_dict`` / read-only ``__setattr__``, byte-identical apart from
the class name in the error message.

Only those three members live here. The classes' ``__getitem__``,
``__iter__`` and ``__contains__`` are genuinely different -- one reads a
cached ``ForwardState``, one a sample array, one stacks over galaxies -- and
sharing them would mean parameterizing away the only interesting part.

Deliberately *not* a :class:`collections.abc.Mapping` base. ``PropertyCatalog``
returns plain ``list`` from ``values()`` and ``items()``; inheriting from
``Mapping`` would silently turn those into views and change what a caller can
subscript. ``PosteriorProperties`` registered as a ``Mapping`` separately in
#1459 and keeps that -- this mixin precedes it in the MRO so its list-returning
``keys`` still wins.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

__all__ = ["ReadOnlyPropertyMapping"]


class ReadOnlyPropertyMapping:
    """Shared read-only surface for a property catalog.

    A subclass supplies ``__getitem__`` and ``__iter__``; this provides the
    three members that were identical across all three implementations.

    Notes
    -----
    ``__setattr__`` raises unconditionally, so subclasses must initialize
    through ``object.__setattr__`` -- as all three already did.
    """

    def keys(self) -> list[str]:
        """Available property names, in iteration order.

        Returns
        -------
        list of str
            A list rather than a view: these names are cheap to materialize
            and callers index into them.
        """
        return list(self)

    def to_dict(self, names: Iterable[str] | None = None) -> dict[str, Any]:
        """Export properties as a plain dict, ready for a table.

        Parameters
        ----------
        names: iterable of str, optional
            Names to export. Defaults to every available property.

        Returns
        -------
        dict
            ``name -> value``. The value type is whatever the subclass's
            ``__getitem__`` yields: a scalar for a prediction, an array over
            draws for a fit, an array over galaxies for a catalog.
        """
        if names is None:
            names = list(self)
        return {name: self[name] for name in names}

    def __setattr__(self, name: str, value: Any) -> None:
        """Refuse attribute assignment -- these views are read-only."""
        raise AttributeError(f"{type(self).__name__} is read-only")
