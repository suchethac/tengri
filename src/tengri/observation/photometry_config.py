"""Photometric observation configuration.

Wraps filter transmission curves into a frozen, immutable container
with convenient factory methods for common filter sets.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax.numpy as jnp

from tengri.observation.photometry import FilterCurve


@dataclasses.dataclass(frozen=True)
class Photometry:
    """Photometric observation configuration.

    Immutable container for filter transmission curves with metadata.
    Created via factory methods rather than direct construction.

    Parameters
    ----------
    filters : tuple of FilterCurve
        Filter transmission curves.
    names : tuple of str
        Human-readable filter names (e.g. ``("sdss_r", "sdss_i")``).
    """

    filters: tuple[FilterCurve, ...] = dataclasses.field(hash=False)
    names: tuple[str, ...] = ()

    # Derived fields — set in __post_init__
    filter_waves: tuple[jnp.ndarray, ...] = dataclasses.field(default=(), hash=False, repr=False)
    filter_trans: tuple[jnp.ndarray, ...] = dataclasses.field(default=(), hash=False, repr=False)
    n_filters: int = 0

    def __post_init__(self):
        if len(self.filters) == 0:
            raise ValueError("Photometry requires at least one filter.")

        # Derive names from FilterCurve.name if not provided
        if not self.names:
            object.__setattr__(
                self,
                "names",
                tuple(f.name for f in self.filters),
            )

        # Extract wave/trans arrays for backward compatibility with Model
        object.__setattr__(
            self,
            "filter_waves",
            tuple(f.wave for f in self.filters),
        )
        object.__setattr__(
            self,
            "filter_trans",
            tuple(f.trans for f in self.filters),
        )
        object.__setattr__(self, "n_filters", len(self.filters))

    @staticmethod
    def from_names(
        names: Sequence[str],
        cache_dir: str = "data/filters",
    ) -> Photometry:
        """Create Photometry from filter registry short names.

        Parameters
        ----------
        names : sequence of str
            Short names from ``FILTER_REGISTRY`` (e.g. ``"sdss_r"``,
            ``"jwst_f200w"``).
        cache_dir : str
            Directory for cached SVO filter files.

        Returns
        -------
        Photometry
            Configured photometry with loaded filters.

        Examples
        --------
        >>> phot = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"])
        >>> phot.n_filters
        3
        """
        from tengri.observation.filters import load_filter_set

        _waves, _trans, curves = load_filter_set(list(names), cache_dir=cache_dir)
        return Photometry(filters=tuple(curves), names=tuple(names))

    @staticmethod
    def from_filter_set(
        filter_set: (
            tuple[list[jnp.ndarray], list[jnp.ndarray], list[FilterCurve]]
            | list[FilterCurve]
            | tuple
        ),
    ) -> Photometry:
        """Create Photometry from existing filter data.

        Accepts the 3-tuple returned by ``load_filter_set()`` or a list
        of ``FilterCurve`` objects — backward compatibility bridge.

        Parameters
        ----------
        filter_set
            Either a 3-tuple ``(filter_waves, filter_trans, filter_curves)``
            from ``load_filter_set()``, or a list/tuple of ``FilterCurve``.

        Returns
        -------
        Photometry
            Configured photometry.
        """
        if isinstance(filter_set, (list, tuple)):
            # 3-tuple from load_filter_set()
            if (
                len(filter_set) == 3
                and isinstance(filter_set[0], list)
                and isinstance(filter_set[2], list)
            ):
                curves = filter_set[2]
                return Photometry(filters=tuple(curves))

            # List/tuple of FilterCurve
            if all(isinstance(f, FilterCurve) for f in filter_set):
                return Photometry(filters=tuple(filter_set))

        raise TypeError(
            f"Expected 3-tuple from load_filter_set() or list of FilterCurve, "
            f"got {type(filter_set)}"
        )

    def summary(self) -> str:
        """Return a one-line summary of the photometry configuration."""
        return f"{self.n_filters} filters: {', '.join(self.names)}"
