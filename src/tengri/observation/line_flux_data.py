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
    """

    names: tuple[str, ...]
    fluxes: jnp.ndarray = dataclasses.field(hash=False)
    errors: jnp.ndarray = dataclasses.field(hash=False)
    wavelengths: jnp.ndarray = dataclasses.field(hash=False)
    is_upper_limit: jnp.ndarray | None = dataclasses.field(default=None, hash=False)

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

        if self.is_upper_limit is not None:
            ul = jnp.asarray(self.is_upper_limit)
            if ul.shape != (n,):
                raise ValueError(
                    f"is_upper_limit shape {ul.shape} does not match expected ({n},) for {n} lines"
                )

    @property
    def n_lines(self) -> int:
        """Number of observed lines."""
        return len(self.names)

    def chi2(self, model_fluxes: jnp.ndarray) -> jnp.ndarray:
        """Chi-squared statistic for detected lines (excludes upper limits).

        Parameters
        ----------
        model_fluxes : array, shape (n_lines,)
            Model-predicted line fluxes in erg/s/cm^2.

        Returns
        -------
        jnp.ndarray, scalar
            Sum of ((obs - model) / error)^2 over detected lines.
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
        model_fluxes : array, shape (n_lines,)
            Model-predicted line fluxes in erg/s/cm^2.

        Returns
        -------
        jnp.ndarray, scalar
            Total log-likelihood summed over all lines.
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
        line_data: dict[str, tuple[float, float]],
    ) -> LineFluxData:
        """Construct from a dict of ``{name: (flux, error)}``.

        Line names are looked up in the standard optical catalog
        to determine rest-frame wavelengths.

        Parameters
        ----------
        line_data : dict
            Mapping from line name to ``(flux, error)`` tuple,
            both in erg/s/cm^2. E.g.
            ``{"Halpha": (1.2e-16, 0.1e-16), "Hbeta": (3.5e-17, 0.5e-17)}``.

        Returns
        -------
        LineFluxData

        Raises
        ------
        ValueError
            If any line name is not found in the standard catalog.
        """
        names = []
        fluxes = []
        errors = []
        wavelengths = []

        for name, (flux, error) in line_data.items():
            if name not in _NAME_TO_WAVELENGTH:
                available = sorted(_NAME_TO_WAVELENGTH.keys())
                raise ValueError(f"Unknown line name {name!r}. Available: {available}")
            names.append(name)
            fluxes.append(flux)
            errors.append(error)
            wavelengths.append(_NAME_TO_WAVELENGTH[name])

        return cls(
            names=tuple(names),
            fluxes=jnp.array(fluxes),
            errors=jnp.array(errors),
            wavelengths=jnp.array(wavelengths),
        )

    def summary(self) -> str:
        """Return a one-line summary."""
        return f"{self.n_lines} lines ({', '.join(self.names)})"
