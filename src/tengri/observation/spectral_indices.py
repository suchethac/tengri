"""Spectral index definitions and observed data for direct fitting.

Supports two index types:
- **EW** (equivalent width): measures absorption or emission strength
  relative to a pseudo-continuum defined by sideband windows.
- **break** (flux ratio): ratio of mean fluxes in two continuum windows
  (e.g., Dn4000).

Index values are measured on rest-frame spectra. The forward model generates
a spectrum covering the required wavelength range, measures indices on it,
and compares against observed values via a chi2 likelihood term.

Usage::

    from tengri.observation.spectral_indices import SpectralIndexData

    indices = SpectralIndexData.from_names(
        names=["Dn4000", "HdA"],
        values=[1.8, -1.2],
        errors=[0.05, 0.3],
    )
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

# ── Index definition ──────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SpectralIndexDef:
    """Definition of a single spectral index.

    Parameters
    ----------
    name : str
        Human-readable name (e.g. ``"Dn4000"``).
    index_type : str
        Either ``"EW"`` (equivalent width) or ``"break"`` (flux ratio).
    continuum : tuple of tuple
        Continuum/sideband windows as ``((lo1, hi1), (lo2, hi2), ...)``.
        Rest-frame wavelengths in Angstrom.
        For EW indices: blue and red pseudo-continuum sidebands.
        For break indices: exactly two windows (numerator, denominator).
    feature : tuple of float or None
        Feature window ``(lo, hi)`` in Angstrom. Required for EW indices,
        None for break indices.
    units : str
        ``"AA"`` for Angstrom (default) or ``"mag"`` for magnitude.

    Examples
    --------
    >>> from tengri import SpectralIndexDef
    >>> dn4000 = SpectralIndexDef(
    ...     name="Dn4000",
    ...     index_type="break",
    ...     continuum=((3850.0, 3950.0), (4000.0, 4100.0)),
    ... )
    >>> dn4000.wave_min, dn4000.wave_max
    (3850.0, 4100.0)
    >>> hda = SpectralIndexDef(
    ...     name="HdA",
    ...     index_type="EW",
    ...     continuum=((4041.6, 4079.75), (4128.5, 4161.0)),
    ...     feature=(4083.5, 4122.25),
    ... )
    >>> hda.index_type
    'EW'
    """

    name: str
    index_type: str
    continuum: tuple[tuple[float, float], ...]
    feature: tuple[float, float] | None = None
    units: str = "AA"

    def __post_init__(self):
        if self.index_type not in ("EW", "break"):
            raise ValueError(f"index_type must be 'EW' or 'break', got {self.index_type!r}")
        if self.index_type == "EW" and self.feature is None:
            raise ValueError("EW indices require a feature window.")
        if self.index_type == "break" and len(self.continuum) != 2:
            raise ValueError("Break indices require exactly 2 continuum windows.")

    @property
    def wave_min(self) -> float:
        """Minimum wavelength needed to measure this index.

        Returns
        -------
        float
            Minimum wavelength [Angstrom] across all continuum and feature
            windows defined in this index.

        Notes
        -----
        Computed as the minimum of all window edges. Useful for determining
        the minimum wavelength coverage needed in the forward model spectrum.

        """
        vals = [w for pair in self.continuum for w in pair]
        if self.feature is not None:
            vals.extend(self.feature)
        return min(vals)

    @property
    def wave_max(self) -> float:
        """Maximum wavelength needed to measure this index.

        Returns
        -------
        float
            Maximum wavelength [Angstrom] across all continuum and feature
            windows defined in this index.

        Notes
        -----
        Computed as the maximum of all window edges. Useful for determining
        the maximum wavelength coverage needed in the forward model spectrum.

        """
        vals = [w for pair in self.continuum for w in pair]
        if self.feature is not None:
            vals.extend(self.feature)
        return max(vals)


# ── Standard index catalog ────────────────────────────────────────

STANDARD_INDICES: dict[str, SpectralIndexDef] = {
    "Dn4000": SpectralIndexDef(
        name="Dn4000",
        index_type="break",
        continuum=((3850.0, 3950.0), (4000.0, 4100.0)),
    ),
    "D4000": SpectralIndexDef(
        name="D4000",
        index_type="break",
        continuum=((3750.0, 3950.0), (4050.0, 4250.0)),
    ),
    "HdA": SpectralIndexDef(
        name="HdA",
        index_type="EW",
        continuum=((4041.60, 4079.75), (4128.50, 4161.00)),
        feature=(4083.50, 4122.25),
    ),
    "HdF": SpectralIndexDef(
        name="HdF",
        index_type="EW",
        continuum=((4057.25, 4088.50), (4114.75, 4137.25)),
        feature=(4091.00, 4112.25),
    ),
    "HgA": SpectralIndexDef(
        name="HgA",
        index_type="EW",
        continuum=((4283.50, 4319.75), (4367.25, 4419.75)),
        feature=(4319.75, 4363.50),
    ),
    "HgF": SpectralIndexDef(
        name="HgF",
        index_type="EW",
        continuum=((4283.50, 4319.75), (4354.75, 4384.75)),
        feature=(4331.25, 4352.25),
    ),
    "Mgb": SpectralIndexDef(
        name="Mgb",
        index_type="EW",
        continuum=((5142.63, 5161.38), (5191.38, 5206.38)),
        feature=(5160.13, 5192.63),
    ),
    "Fe5270": SpectralIndexDef(
        name="Fe5270",
        index_type="EW",
        continuum=((5233.15, 5248.15), (5285.65, 5318.15)),
        feature=(5245.65, 5285.65),
    ),
    "Fe5335": SpectralIndexDef(
        name="Fe5335",
        index_type="EW",
        continuum=((5304.63, 5315.88), (5353.38, 5363.38)),
        feature=(5312.13, 5352.13),
    ),
    "Hbeta": SpectralIndexDef(
        name="Hbeta",
        index_type="EW",
        continuum=((4827.88, 4847.88), (4876.63, 4891.63)),
        feature=(4847.88, 4876.63),
    ),
    "Fe4383": SpectralIndexDef(
        name="Fe4383",
        index_type="EW",
        continuum=((4359.13, 4370.38), (4442.88, 4455.38)),
        feature=(4369.13, 4420.38),
    ),
    "Ca4227": SpectralIndexDef(
        name="Ca4227",
        index_type="EW",
        continuum=((4211.00, 4219.75), (4241.00, 4251.00)),
        feature=(4222.25, 4234.75),
    ),
}


# ── JAX-compatible index measurement ──────────────────────────────


def measure_index_jax(
    wave_rest: jnp.ndarray,
    flux: jnp.ndarray,
    index_def: SpectralIndexDef,
) -> jnp.ndarray:
    """Measure a spectral index on a rest-frame spectrum.

    Parameters
    ----------
    wave_rest : ndarray, shape (n_pix,)
        Rest-frame wavelengths [Angstrom]. Must cover all windows defined
        in ``index_def``.
    flux : ndarray, shape (n_pix,)
        Flux density (any consistent units — only ratios matter).
    index_def : SpectralIndexDef
        Index definition specifying windows and index type (EW or break).

    Returns
    -------
    ndarray, shape ()
        Measured index value (scalar). Units depend on ``index_def.units``:
        [Angstrom] for EW indices, [dimensionless] for break indices.

    Notes
    -----
    **JIT-compatible**: yes — uses soft sigmoid edges for differentiability
    rather than hard window boundaries.

    **Gradient-safe**: yes — fully differentiable w.r.t. flux.

    """
    if index_def.index_type == "break":
        return _measure_break(wave_rest, flux, index_def)
    else:
        return _measure_ew(wave_rest, flux, index_def)


def _window_mean_flux(
    wave: jnp.ndarray, flux: jnp.ndarray, lo: float, hi: float, edge_width: float = 1.0
) -> jnp.ndarray:
    """Mean flux in a wavelength window, using soft sigmoid edges for differentiability."""
    w_lo = jax.nn.sigmoid((wave - lo) / edge_width)
    w_hi = jax.nn.sigmoid((hi - wave) / edge_width)
    weights = w_lo * w_hi
    n = jnp.maximum(jnp.sum(weights), 1e-10)
    return jnp.sum(flux * weights) / n


def _measure_break(wave: jnp.ndarray, flux: jnp.ndarray, idx: SpectralIndexDef) -> jnp.ndarray:
    """Compute spectral break ratio (red window mean flux / blue window mean flux)."""
    blue_lo, blue_hi = idx.continuum[0]
    red_lo, red_hi = idx.continuum[1]
    f_blue = _window_mean_flux(wave, flux, blue_lo, blue_hi)
    f_red = _window_mean_flux(wave, flux, red_lo, red_hi)
    return f_red / jnp.maximum(f_blue, 1e-30)


def _measure_ew(wave: jnp.ndarray, flux: jnp.ndarray, idx: SpectralIndexDef) -> jnp.ndarray:
    """Compute equivalent width from continuum-to-feature flux ratio and feature window width."""
    cont_fluxes = []
    for lo, hi in idx.continuum:
        cont_fluxes.append(_window_mean_flux(wave, flux, lo, hi))
    cont_flux = jnp.mean(jnp.array(cont_fluxes))

    feat_lo, feat_hi = idx.feature
    feat_flux = _window_mean_flux(wave, flux, feat_lo, feat_hi)
    feat_width = feat_hi - feat_lo

    ew = feat_width * (cont_flux - feat_flux) / jnp.maximum(cont_flux, 1e-30)

    if idx.units == "mag":
        return -2.5 * jnp.log10(jnp.maximum(1.0 - ew / feat_width, 1e-30))
    return ew


# ── Observed data container ───────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class SpectralIndexData:
    """Observed spectral index values for fitting.

    Parameters
    ----------
    index_defs : tuple of SpectralIndexDef
        Index definitions.
    values : jnp.ndarray
        Observed index values, shape ``(n_indices,)``.
    errors : jnp.ndarray
        1-sigma uncertainties, shape ``(n_indices,)``.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import SpectralIndexData
    >>> sid = SpectralIndexData.from_names(
    ...     names=["Dn4000", "HdA"],
    ...     values=[1.35, 5.2],
    ...     errors=[0.05, 0.3],
    ... )
    >>> sid.n_indices
    2
    >>> sid.names
    ('Dn4000', 'HdA')
    """

    index_defs: tuple[SpectralIndexDef, ...]
    values: jnp.ndarray = dataclasses.field(hash=False)
    errors: jnp.ndarray = dataclasses.field(hash=False)

    def __post_init__(self) -> None:
        n = len(self.index_defs)
        if n == 0:
            raise ValueError("SpectralIndexData requires at least one index.")

        values = jnp.asarray(self.values)
        errors = jnp.asarray(self.errors)

        if values.shape != (n,):
            raise ValueError(f"values shape {values.shape} does not match expected ({n},)")
        if errors.shape != (n,):
            raise ValueError(f"errors shape {errors.shape} does not match expected ({n},)")

    @property
    def n_indices(self) -> int:
        """Number of spectral indices.

        Returns
        -------
        int
            Number of indices in this dataset.

        """
        return len(self.index_defs)

    @property
    def names(self) -> tuple[str, ...]:
        """Tuple of spectral index names.

        Returns
        -------
        tuple[str, ...]
            Index names in the same order as ``index_defs``, e.g.
            ``("Dn4000", "HdA")``.

        Notes
        -----
        Names match the keys in STANDARD_INDICES.

        Examples
        --------
        >>> from tengri import SpectralIndexData
        >>> sid = SpectralIndexData.from_names(["Dn4000", "HdA"], [1.3, 5.1], [0.05, 0.3])
        >>> sid.names
        ('Dn4000', 'HdA')
        """
        return tuple(d.name for d in self.index_defs)

    @property
    def wave_range(self) -> tuple[float, float]:
        """Rest-frame wavelength range needed to measure all indices.

        Returns
        -------
        tuple[float, float]
            Tuple ``(wave_min, wave_max)`` [Angstrom] covering all continuum
            and feature windows across all indices.

        Notes
        -----
        Useful for determining minimum wavelength coverage required in the
        forward model spectrum to compute all indices.

        """
        lo = min(d.wave_min for d in self.index_defs)
        hi = max(d.wave_max for d in self.index_defs)
        return (lo, hi)

    @classmethod
    def from_names(
        cls,
        names: list[str],
        values: list[float],
        errors: list[float],
    ) -> SpectralIndexData:
        """Construct from standard index names.

        Parameters
        ----------
        names : list[str]
            Index names from ``STANDARD_INDICES`` (e.g. ``["Dn4000", "HdA"]``).
        values : list[float]
            Observed index values. Units depend on index type: [Angstrom] for
            EW indices, [dimensionless] for break indices.
        errors : list[float]
            1-sigma uncertainties (same units as ``values``).

        Returns
        -------
        SpectralIndexData
            Spectral index data object with index definitions looked up from
            ``STANDARD_INDICES``.

        Raises
        ------
        ValueError
            If any name is not in ``STANDARD_INDICES``.

        Notes
        -----
        The wavelength coverage required to measure all indices can be obtained
        via :func:`wave_range` property.

        """
        defs = []
        for name in names:
            if name not in STANDARD_INDICES:
                available = sorted(STANDARD_INDICES.keys())
                raise ValueError(f"Unknown index name {name!r}. Available: {available}")
            defs.append(STANDARD_INDICES[name])

        return cls(
            index_defs=tuple(defs),
            values=jnp.array(values),
            errors=jnp.array(errors),
        )

    def chi2(self, model_values: jnp.ndarray) -> jnp.ndarray:
        """Chi-squared statistic.

        Parameters
        ----------
        model_values : ndarray, shape (n_indices,)
            Model-predicted index values (same units as ``values``).

        Returns
        -------
        ndarray, shape ()
            Sum of ``((obs - model) / error)^2`` [dimensionless].

        Notes
        -----
        **JIT-compatible**: yes — uses only jnp primitives.

        **Gradient-safe**: yes — differentiable w.r.t. ``model_values``.

        """
        residual = (self.values - model_values) / self.errors
        return jnp.sum(residual**2)

    def log_likelihood(self, model_values: jnp.ndarray) -> jnp.ndarray:
        """Gaussian log-likelihood.

        Parameters
        ----------
        model_values : ndarray, shape (n_indices,)
            Model-predicted index values (same units as ``values``).

        Returns
        -------
        ndarray, shape ()
            Total log-likelihood summed over all indices [dimensionless].

        Notes
        -----
        **JIT-compatible**: yes — uses only jnp primitives.

        **Gradient-safe**: yes — differentiable w.r.t. ``model_values``.

        Assumes Gaussian uncertainties on the observed indices.

        """
        residual = (self.values - model_values) / self.errors
        return jnp.sum(-0.5 * residual**2 - jnp.log(self.errors) - 0.5 * jnp.log(2.0 * jnp.pi))

    def summary(self) -> str:
        """Return a human-readable summary of the spectral indices.

        Returns
        -------
        str
            Summary string (e.g., ``"2 indices (Dn4000, HdA)"``).

        Notes
        -----
        Intended for logging and diagnostics, not for programmatic parsing.

        """
        return f"{self.n_indices} indices ({', '.join(self.names)})"
