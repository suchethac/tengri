# SPDX-License-Identifier: BSD-3-Clause
"""Photometric observation configuration.

Wraps filter transmission curves into a frozen, immutable container
with convenient factory methods for common filter sets.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import jax.numpy as jnp

from tengri.observation.photometry import FilterCurve
from tengri.utils.filter_convention import FilterConvention


@dataclasses.dataclass(frozen=True)
class Photometry:
    """Photometric observation configuration — filter set, not the fluxes.

    This class holds *which bands* the model should evaluate.  Measured
    fluxes and uncertainties are passed separately to
    :class:`tengri.Fitter` (``data=`` and ``noise=``).

    Don't call ``Photometry(...)`` directly — use the factory:

        >>> phot = tengri.Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])

    or browse the bundled set:

        >>> bandset = tengri.list_filters(survey="SDSS").names()
        >>> phot = tengri.Photometry.from_names(bandset)

    Then the full fit pattern:

        obs = tengri.Observation(photometry=phot)
        spec = tengri.Parameters(redshift=0.1, ...)
        model = tengri.SEDModel(spec, ssp_data, observation=obs)
        fitter = tengri.Fitter(model,
                               data=measured_fluxes,    # ← your fluxes go here
                               noise=measured_errors)   # ← your sigma here
        posterior = fitter.run("nuts")

    Parameters
    ----------
    filters : tuple of FilterCurve
        Filter transmission curves.
    names : tuple of str
        Human-readable filter names (e.g. ``("sdss_r", "sdss_i")``).

    Returns
    -------
    Photometry
        Photometry instance with derived fields populated.

    Attributes
    ----------
    filters : tuple[FilterCurve, ...]
        Filter transmission curves.
    names : tuple[str, ...]
        Human-readable filter names.
    filter_waves : tuple[ndarray, ...]
        Wavelength arrays for each filter [Angstrom].
    filter_trans : tuple[ndarray, ...]
        Transmission curves for each filter [dimensionless].
    n_filters : int
        Number of filters.

    Notes
    -----
    A frozen dataclass that encapsulates filter metadata and transmission
    curves. Provides factory methods (from_names, from_filter_set) for
    convenient construction. Precomputes derived fields (filter_waves,
    filter_trans, n_filters) at initialization for efficient SED projection.

    Examples
    --------
    >>> from tengri import Photometry
    >>> phot = Photometry.from_names(["sdss_r", "sdss_i"])
    >>> phot.n_filters
    2

    """

    filters: tuple[FilterCurve, ...] = dataclasses.field(hash=False)
    names: tuple[str, ...] = ()

    # Photometric filter-convolution convention (ADR-0017). ``bessell``
    # (default, photon-counting 1/lambda; matches DSPS/FSPS) or ``energy``
    # (1/lambda^2; matches CIGALE). Flows into the exact predict path.
    convention: FilterConvention = FilterConvention.BESSELL

    # Derived fields — set in __post_init__
    filter_waves: tuple[jnp.ndarray, ...] = dataclasses.field(default=(), hash=False, repr=False)
    filter_trans: tuple[jnp.ndarray, ...] = dataclasses.field(default=(), hash=False, repr=False)
    n_filters: int = 0

    # Precomputed padded arrays for the batched, JIT-friendly projection
    # path. n_filters axis is padded to FILTER_COUNT_BUCKETS so different
    # Photometry instances with similar counts share an XLA cache key.
    _fw_padded: jnp.ndarray | None = dataclasses.field(default=None, hash=False, repr=False)
    _ft_padded: jnp.ndarray | None = dataclasses.field(default=None, hash=False, repr=False)
    _n_valid: jnp.ndarray | None = dataclasses.field(default=None, hash=False, repr=False)

    def __post_init__(self):
        if len(self.filters) == 0:
            raise ValueError("Photometry requires at least one filter.")

        # Normalise a string convention (e.g. "energy") to the enum so the
        # JIT static-argument cache key is stable.
        if not isinstance(self.convention, FilterConvention):
            object.__setattr__(self, "convention", FilterConvention(self.convention))

        # Derive names from FilterCurve.name if not provided
        if not self.names:
            object.__setattr__(
                self,
                "names",
                tuple(f.name for f in self.filters),
            )

        # Materialise wave/trans arrays for direct NumPy-level access
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

        # Padded arrays for the batched projection path. n_filters_real is
        # equal to n_filters (the unpadded count); padded-out rows contribute
        # zero via the integrand. Lazy-import to avoid a top-level cycle.
        from tengri.observation.photometry import pad_filters_to_bucket

        fw_p, ft_p, n_v, _ = pad_filters_to_bucket(self.filter_waves, self.filter_trans)
        object.__setattr__(self, "_fw_padded", fw_p)
        object.__setattr__(self, "_ft_padded", ft_p)
        object.__setattr__(self, "_n_valid", n_v)

    @staticmethod
    def from_names(
        names: Sequence[str],
        cache_dir: str = "data/filters",
        convention: FilterConvention | str = FilterConvention.BESSELL,
    ) -> Photometry:
        """Create Photometry from filter registry short names.

        Parameters
        ----------
        names : sequence[str]
            Short names from ``FILTER_REGISTRY`` (e.g. ``"sdss_r"``,
            ``"jwst_f200w"``).
        cache_dir : str, optional
            Directory for cached SVO filter files. Default: ``"data/filters"``.
        convention : FilterConvention or str, optional
            Filter-convolution convention: ``"bessell"`` (default,
            photon-counting 1/lambda; DSPS/FSPS) or ``"energy"`` (1/lambda^2;
            CIGALE). See :func:`tengri.list_filter_conventions`.

        Returns
        -------
        Photometry
            Configured photometry with loaded filter transmission curves.
            Filters are validated against the global registry.

        Notes
        -----
        Loads filter transmission curves from the SVO filter service or
        local cache. Filter names are validated against the registry;
        unrecognized names raise a KeyError.

        Examples
        --------
        >>> phot = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"])
        >>> phot.n_filters
        3

        """
        from tengri.observation.filters import load_filter_set

        _waves, _trans, curves = load_filter_set(list(names), cache_dir=cache_dir)
        return Photometry(filters=tuple(curves), names=tuple(names), convention=convention)

    @staticmethod
    def from_filter_set(
        filter_set: (
            tuple[list[jnp.ndarray], list[jnp.ndarray], list[FilterCurve]]
            | list[FilterCurve]
            | tuple
        ),
        convention: FilterConvention | str = FilterConvention.BESSELL,
    ) -> Photometry:
        """Create Photometry from existing filter data.

        Accepts the 3-tuple returned by ``load_filter_set()`` or a list
        of ``FilterCurve`` objects.

        Parameters
        ----------
        filter_set : tuple | list
            Either a 3-tuple ``(filter_waves, filter_trans, filter_curves)``
            from ``load_filter_set()``, or a list/tuple of ``FilterCurve``
            objects.

        Returns
        -------
        Photometry
            Configured photometry with precomputed field (filter curves,
            wavelengths, transmissions, and filter count).

        Notes
        -----
        Flexible constructor that accepts pre-loaded filter data. Useful
        when filters are already loaded or constructed by external code.

        Examples
        --------
        >>> from tengri.observation.filters import load_filter_set
        >>> waves, trans, curves = load_filter_set(["sdss_r", "sdss_i"])
        >>> phot = Photometry.from_filter_set((waves, trans, curves))

        """
        if isinstance(filter_set, (list, tuple)):
            # 3-tuple from load_filter_set()
            if (
                len(filter_set) == 3
                and isinstance(filter_set[0], list)
                and isinstance(filter_set[2], list)
            ):
                curves = filter_set[2]
                return Photometry(filters=tuple(curves), convention=convention)

            # List/tuple of FilterCurve
            if all(isinstance(f, FilterCurve) for f in filter_set):
                return Photometry(filters=tuple(filter_set), convention=convention)

        raise TypeError(
            f"Expected 3-tuple from load_filter_set() or list of FilterCurve, "
            f"got {type(filter_set)}"
        )

    def summary(self) -> str:
        """Return a one-line summary of the photometry configuration.

        Returns
        -------
        str
            Filter count and comma-separated filter names.

        Notes
        -----
        Provides concise string representation for logging and diagnostics.

        """
        return f"{self.n_filters} filters: {', '.join(self.names)}"
