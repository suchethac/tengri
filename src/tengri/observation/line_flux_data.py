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
from jax.scipy.stats import norm

from tengri._cache_keys import KeyPolicy, content, derive_key, shape
from tengri.observation.line_list import _DEFAULT_OPTICAL_LINES
from tengri.observation.noise import DETECTED, censored_neg_log_likelihood

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
    safely below the limit, smoothly rising as it crosses. The term is
    ``log_ndtr``-based, with finite gradients even for strongly violated
    limits (z < -15).

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

    def cache_key(self) -> tuple:
        """Return a hashable cache key for this line flux data.

        Returns
        -------
        tuple
            Cache key derived from line names, wavelengths, limit masks, and array shapes.

        Notes
        -----
        Line names, wavelengths, and limit flags are keyed by content as they
        define the likelihood function. Flux and error values are keyed by
        shape only (array values don't affect the program, only the per-galaxy
        data does).
        """
        return derive_key(self, _LINE_FLUX_DATA_CACHE_KEY_POLICY)

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
        """Chi-squared statistic: Gaussian for detections, ``-2 ln Phi(z)`` for limits.

        Parameters
        ----------
        model_fluxes : ndarray, shape (n_lines,)
            Model-predicted line fluxes [erg/s/cm^2].

        Returns
        -------
        ndarray, shape ()
            Sum over lines of ``((obs - model) / error)^2`` for detections and
            ``-2 ln Phi(z)`` for limits [dimensionless].

        Notes
        -----
        **JIT-compatible**: yes, uses only jnp primitives.

        **Gradient-safe**: yes, differentiable w.r.t. ``model_fluxes``.

        **Convention**: a limit line contributes twice its censored energy,
        ``2 * [-ln Phi(z)]`` with ``z = (limit - model) / error`` (upper) or
        ``z = (model - limit) / error`` (lower). This is the same per-datum
        number the photometry path (``CensoredLikelihood``) scores for a
        limit, so ``chi2 = 2 * E`` for the shared censored energy ``E``
        (detections: ``E = 0.5 r^2`` plus a model-independent ``ln error``
        that ``chi2`` does not carry). A limit is not dropped and not scored
        as a detection.

        """
        mask = self.limit_mask
        if mask is None:
            mask = jnp.zeros(self.n_lines)
        z_up = (self.fluxes - model_fluxes) / self.errors
        z_lo = -z_up
        chi2_limit = -2.0 * jnp.where(mask > 0, norm.logcdf(z_up), norm.logcdf(z_lo))
        return jnp.sum(jnp.where(mask == DETECTED, z_up**2, chi2_limit))

    def log_likelihood(self, model_fluxes: jnp.ndarray) -> jnp.ndarray:
        """Log-likelihood: normalized Gaussian for detections, ``ln Phi(z)`` for limits.

        For detected lines:
            ln L = -0.5 * ((obs - model) / error)^2 - ln(error) - 0.5*ln(2π)

        For upper limits:
            ln L = ln Phi((F_limit - F_model) / error)

        For lower limits:
            ln L = ln Phi((F_model - F_limit) / error)

        where Phi is the standard normal CDF.

        Parameters
        ----------
        model_fluxes : ndarray, shape (n_lines,)
            Model-predicted line fluxes [erg/s/cm^2].

        Returns
        -------
        ndarray, shape ()
            Total log-likelihood [dimensionless].

        Notes
        -----
        **JIT-compatible**: yes, uses only jnp primitives.

        **Gradient-safe**: yes, differentiable w.r.t. ``model_fluxes``, with
        finite gradients for strongly violated limits (no probability clamp).

        **Normalization**: this is a log-likelihood that can enter an
        evidence, so detections keep their full Gaussian normalization. The
        per-line shape is :func:`~tengri.observation.noise.censored_neg_log_likelihood`
        (``0.5 r^2 + ln error`` for detections, ``-ln Phi(z)`` for limits),
        which omits the model-independent ``0.5 ln(2π)`` per detection; that
        constant is restored here. A limit's term is ``ln Phi(z)`` exactly,
        with no constant.

        """
        mask = self.limit_mask
        if mask is None:
            residual = (self.fluxes - model_fluxes) / self.errors
            return jnp.sum(-0.5 * residual**2 - jnp.log(self.errors) - 0.5 * jnp.log(2.0 * jnp.pi))
        energy = censored_neg_log_likelihood(self.fluxes, self.errors, model_fluxes, mask)
        n_detected = jnp.sum(mask == DETECTED)
        return -energy - 0.5 * n_detected * jnp.log(2.0 * jnp.pi)

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


_LINE_FLUX_DATA_CACHE_KEY_POLICY: KeyPolicy = {
    "names": content("line names determine the likelihood function"),
    "fluxes": shape("per-galaxy data, not part of the structural program"),
    "errors": shape("per-galaxy data, not part of the structural program"),
    "wavelengths": content("line wavelengths determine the likelihood function"),
    "is_upper_limit": content("upper limit flags define the likelihood function"),
    "is_lower_limit": content("lower limit flags define the likelihood function"),
}
