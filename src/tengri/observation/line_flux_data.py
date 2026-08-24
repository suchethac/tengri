# SPDX-License-Identifier: BSD-3-Clause
"""Observed emission line flux data for direct fitting.

Provides a declarative container for observed emission line fluxes
(integrated flux in erg/s/cm^2) that can be compared against model
predictions from the nebular backend. Unlike spectroscopic emission
line fitting (which operates on pixel-level spectra), this handles
the case where users have measured line fluxes from narrow-band
imaging, IFU analyses, or line-finding pipelines.

Usage::

    from tengri.observation.line_flux_data import LineFluxData

    lines = LineFluxData.from_dict(
        {
            "Halpha": (1.2e-16, 0.1e-16),
            "Hbeta": (3.5e-17, 0.5e-17),
            "OIII_5007": (8.0e-17, 0.8e-17),
        }
    )
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import jax.scipy.special as jsp

from tengri.observation.line_list import _DEFAULT_OPTICAL_LINES

_NAME_TO_WAVELENGTH: dict[str, float] = {t[0]: t[1] for t in _DEFAULT_OPTICAL_LINES}


@dataclasses.dataclass(frozen=True)
class LineFluxData:
    """Observed emission line fluxes for fitting.

    Parameters
    ----------
    names : tuple[str, ...]
        Line identifiers matching ``LineList`` convention
        (e.g. ``"Halpha"``, ``"OIII_5007"``).
    fluxes : jnp.ndarray
        Observed integrated line fluxes in erg/s/cm^2, shape ``(n_lines,)``.
    errors : jnp.ndarray
        1-sigma uncertainties on fluxes in erg/s/cm^2, shape ``(n_lines,)``.
    wavelengths : jnp.ndarray
        Rest-frame vacuum wavelengths in Angstrom, shape ``(n_lines,)``.
        Used to match against the nebular backend's line output.

    Returns
    -------
    LineFluxData
        Emission line flux container with validation.

    Attributes
    ----------
    names : tuple[str, ...]
        Line identifiers.
    fluxes : ndarray, shape (n_lines,)
        Observed line fluxes [erg/s/cm²].
    errors : ndarray, shape (n_lines,)
        1-sigma measurement uncertainties [erg/s/cm²].
    wavelengths : ndarray, shape (n_lines,)
        Rest-frame vacuum wavelengths [Angstrom].
    is_upper_limit : ndarray or None
        Boolean mask indicating upper limits [dimensionless].
    is_lower_limit : ndarray or None
        Boolean mask indicating lower limits [dimensionless].

    Notes
    -----
    **Immutable container**: All fields are read-only by convention. Construct
    once with validated data, do not modify.

    **Upper limits**: The ``is_upper_limit`` field marks lines that are
    non-detections (typically <2-3σ); ``fluxes`` carries the limit value.
    In the fit these enter as censored data points:
    ``ln L = ln Φ((F_lim − F_model)/σ)``, zero penalty when the model sits
    safely below the limit, smoothly rising as it crosses.

    **Lower limits**: ``is_lower_limit`` mirrors this for saturated or
    blended measurements that only bound the flux from below:
    ``ln L = ln Φ((F_model − F_lim)/σ)``. A line cannot be both an upper
    and a lower limit.

    Examples
    --------
    >>> from tengri.observation import LineFluxData
    >>> lfd = LineFluxData.from_dict(
    ...     {
    ...         "Halpha": (1.2e-16, 0.1e-16),
    ...         "Hbeta": (3.5e-17, 0.5e-17),
    ...         "OIII_5007": (8.0e-17, 0.8e-17),
    ...     }
    ... )
    >>> lfd.n_lines
    3
    >>> lfd.names
    ('Halpha', 'Hbeta', 'OIII_5007')
    """

    names: tuple[str, ...]
    fluxes: jnp.ndarray = dataclasses.field(hash=False)
    errors: jnp.ndarray = dataclasses.field(hash=False)
    wavelengths: jnp.ndarray = dataclasses.field(hash=False)
    is_upper_limit: jnp.ndarray | None = dataclasses.field(default=None, hash=False)
    is_lower_limit: jnp.ndarray | None = dataclasses.field(default=None, hash=False)

    def __post_init__(self) -> None:
        n = len(self.names)
        if n == 0:
            raise ValueError("LineFluxData requires at least one line.")

        fluxes = jnp.asarray(self.fluxes)
        errors = jnp.asarray(self.errors)
        wavelengths = jnp.asarray(self.wavelengths)

        if fluxes.shape != (n,):
            raise ValueError(
                f"fluxes shape {fluxes.shape} does not match expected ({n},) for {n} lines"
            )
        if errors.shape != (n,):
            raise ValueError(
                f"errors shape {errors.shape} does not match expected ({n},) for {n} lines"
            )
        if wavelengths.shape != (n,):
            raise ValueError(
                f"wavelengths shape {wavelengths.shape} does not match "
                f"expected ({n},) for {n} lines"
            )

        for field_name in ("is_upper_limit", "is_lower_limit"):
            mask = getattr(self, field_name)
            if mask is not None:
                mask = jnp.asarray(mask)
                if mask.shape != (n,):
                    raise ValueError(
                        f"{field_name} shape {mask.shape} does not match "
                        f"expected ({n},) for {n} lines"
                    )
        if self.is_upper_limit is not None and self.is_lower_limit is not None:
            both = jnp.asarray(self.is_upper_limit) & jnp.asarray(self.is_lower_limit)
            if bool(jnp.any(both)):
                bad = [nm for nm, b in zip(self.names, both) if bool(b)]
                raise ValueError(f"lines marked as BOTH upper and lower limit: {bad}, pick one.")

    @property
    def limit_mask(self) -> jnp.ndarray | None:
        """Trinary censoring mask: 0 = detected, +1 = upper limit, -1 = lower limit.

        Returns
        -------
        ndarray, shape (n_lines,), or None
            ``None`` when no line carries a limit flag (all detections);
            callers use this to select the plain Gaussian likelihood.
        """
        if self.is_upper_limit is None and self.is_lower_limit is None:
            return None
        n = len(self.names)
        mask = jnp.zeros(n)
        if self.is_upper_limit is not None:
            mask = jnp.where(jnp.asarray(self.is_upper_limit), 1.0, mask)
        if self.is_lower_limit is not None:
            mask = jnp.where(jnp.asarray(self.is_lower_limit), -1.0, mask)
        return mask

    @property
    def n_lines(self) -> int:
        """Number of observed lines.

        Returns
        -------
        int
            Number of lines in this dataset.

        Notes
        -----
        Computed from the length of the ``names`` tuple. Constant for
        the lifetime of the object (immutable).

        """
        return len(self.names)

    def chi2(self, model_fluxes: jnp.ndarray) -> jnp.ndarray:
        """Chi-squared statistic for detected lines (excludes upper limits).

        Parameters
        ----------
        model_fluxes : ndarray, shape (n_lines,)
            Model-predicted line fluxes [erg/s/cm^2].

        Returns
        -------
        ndarray, shape ()
            Sum of ((obs - model) / error)^2 over detected lines
            [dimensionless].

        Notes
        -----
        **JIT-compatible**: yes, uses only jnp primitives.

        **Gradient-safe**: yes, differentiable w.r.t. ``model_fluxes``.

        Upper limit lines (where ``is_upper_limit`` is True) are excluded
        from the sum.

        """
        residual = (self.fluxes - model_fluxes) / self.errors
        chi2_per_line = residual**2
        if self.is_upper_limit is not None:
            detected = ~self.is_upper_limit
            chi2_per_line = jnp.where(detected, chi2_per_line, 0.0)
        return jnp.sum(chi2_per_line)

    def log_likelihood(self, model_fluxes: jnp.ndarray) -> jnp.ndarray:
        """Log-likelihood: Gaussian for detections, survival function for upper limits.

        For detected lines:
            ln L = -0.5 * ((obs - model) / error)^2 - ln(error) - 0.5*ln(2π)

        For upper limits (non-detections reported as N-sigma limits):
            ln L = ln(0.5 * erfc((model - obs_limit) / (error * sqrt(2))))

        Parameters
        ----------
        model_fluxes : ndarray, shape (n_lines,)
            Model-predicted line fluxes [erg/s/cm^2].

        Returns
        -------
        ndarray, shape ()
            Total log-likelihood summed over all lines [dimensionless].

        Notes
        -----
        **JIT-compatible**: yes, uses only jnp primitives.

        **Gradient-safe**: yes, differentiable w.r.t. ``model_fluxes``.

        Handles both detections and upper limits (marked via ``is_upper_limit``).
        Upper limit lines use the complementary error function (erfc) to
        compute the probability that the true flux exceeds the model prediction.

        """
        residual = (self.fluxes - model_fluxes) / self.errors
        ll_gaussian = -0.5 * residual**2 - jnp.log(self.errors) - 0.5 * jnp.log(2.0 * jnp.pi)

        if self.is_upper_limit is None:
            return jnp.sum(ll_gaussian)

        x_ul = (model_fluxes - self.fluxes) / (self.errors * jnp.sqrt(2.0))
        ll_upper = jnp.log(jnp.maximum(0.5 * jsp.erfc(x_ul), 1e-30))

        ll_per_line = jnp.where(self.is_upper_limit, ll_upper, ll_gaussian)
        return jnp.sum(ll_per_line)

    @classmethod
    def from_dict(
        cls,
        line_data: dict[str, tuple],
    ) -> LineFluxData:
        """Construct from a dict of ``{name: (flux, error[, limit])}``.

        Line names are looked up in the standard optical catalog
        to determine rest-frame wavelengths.

        Parameters
        ----------
        line_data : dict[str, tuple]
            Mapping from line name to ``(flux, error)``, both
            [erg/s/cm^2], with an optional third element ``"upper"`` or
            ``"lower"`` marking the flux as a censored limit rather than a
            detection. E.g.
            ``{"Halpha": (1.2e-16, 0.1e-16), "Hbeta": (3.5e-17, 0.5e-17, "upper")}``.

        Returns
        -------
        LineFluxData
            Line flux data object with names, fluxes, errors, wavelengths,
            and any limit flags populated from the input dict.

        Raises
        ------
        ValueError
            If any line name is not found in the standard catalog, or a
            limit marker is not ``"upper"`` / ``"lower"``.

        Notes
        -----
        Wavelengths are looked up from the default optical emission line catalog
        (vacuum wavelengths). Unknown line names raise a descriptive error with
        the list of available names.

        """
        names = []
        fluxes = []
        errors = []
        wavelengths = []
        upper = []
        lower = []

        for name, entry in line_data.items():
            if name not in _NAME_TO_WAVELENGTH:
                available = sorted(_NAME_TO_WAVELENGTH.keys())
                raise ValueError(f"Unknown line name {name!r}. Available: {available}")
            flux, error, *limit = entry
            if limit and limit[0] not in ("upper", "lower"):
                raise ValueError(
                    f"Line {name!r}: limit marker must be 'upper' or 'lower', got {limit[0]!r}."
                )
            names.append(name)
            fluxes.append(flux)
            errors.append(error)
            wavelengths.append(_NAME_TO_WAVELENGTH[name])
            upper.append(bool(limit and limit[0] == "upper"))
            lower.append(bool(limit and limit[0] == "lower"))

        return cls(
            names=tuple(names),
            fluxes=jnp.array(fluxes),
            errors=jnp.array(errors),
            wavelengths=jnp.array(wavelengths),
            is_upper_limit=jnp.array(upper) if any(upper) else None,
            is_lower_limit=jnp.array(lower) if any(lower) else None,
        )

    def summary(self) -> str:
        """Return a one-line summary.

        Returns
        -------
        str
            Summary string (e.g., "3 lines (Halpha, Hbeta, OIII_5007)").

        Notes
        -----
        Intended for logging and diagnostics, not for programmatic parsing.

        """
        return f"{self.n_lines} lines ({', '.join(self.names)})"
