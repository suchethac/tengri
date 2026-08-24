# SPDX-License-Identifier: BSD-3-Clause
"""Observed emission line *ratio* data for direct fitting.

A declarative container for measured emission-line ratios, e.g. the Balmer
decrement Hα/Hβ, or BPT diagnostics like [NII]/Hα, to be compared against
model predictions from the nebular backend.

Ratios are the natural data product when the *absolute* flux calibration is
uncertain but the *relative* line strengths are reliable (slit losses, aperture
corrections, and flux-calibration zero-points cancel in a ratio). Fitting the
ratio directly, rather than two absolute fluxes, avoids contaminating the fit
with that calibration uncertainty.

Usage::

    from tengri.observation import LineRatioData

    ratios = LineRatioData.from_dict(
        {
            ("Halpha", "Hbeta"): (4.2, 0.3),  # Balmer decrement
            ("NII_6584", "Halpha"): (0.35, 0.05),  # BPT-NII numerator
        }
    )

The model ratio is ``flux(numerator) / flux(denominator)`` evaluated from the
nebular backend's line luminosities (see :meth:`SEDModel.predict_line_ratios`),
so this works on both the exact and SpectrumPrecomp paths, line luminosities
are grid-independent.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from tengri.observation.line_list import _DEFAULT_OPTICAL_LINES

_NAME_TO_WAVELENGTH: dict[str, float] = {t[0]: t[1] for t in _DEFAULT_OPTICAL_LINES}


@dataclasses.dataclass(frozen=True)
class LineRatioData:
    """Observed emission line ratios for fitting.

    Parameters
    ----------
    numerators, denominators: tuple[str, ...]
        Line identifiers (``LineList`` convention) for the numerator and
        denominator of each ratio, shape ``(n_ratios,)`` each.
    ratios: ndarray, shape (n_ratios,)
        Measured flux ratios ``F_num / F_den`` [dimensionless]. When
        ``log_space=True`` these are ``log10(F_num / F_den)``.
    errors: ndarray, shape (n_ratios,)
        1-sigma uncertainties on ``ratios`` [dimensionless], in the same
        space (linear or dex) as ``ratios``.
    numerator_waves, denominator_waves: ndarray, shape (n_ratios,)
        Rest-frame vacuum wavelengths [Angstrom] used to match the nebular
        backend's line output.
    log_space: bool, default False
        If True, ``ratios``/``errors`` are interpreted in log10 space, a
        log-normal likelihood, appropriate for BPT-style diagnostics whose
        scatter is symmetric in dex. If False, a linear Gaussian on the ratio.

    Notes
    -----
    **Immutable container**: construct once with validated data.

    **Why ratios**, when absolute flux calibration is uncertain (the common
    case for slit/fiber spectra) but line ratios are robust, constraining the
    ratio avoids injecting calibration error into the fit. See
    [[project_nebular_unit_conventions]] for the line-luminosity conventions.

    Examples
    --------
    >>> from tengri.observation import LineRatioData
    >>> lrd = LineRatioData.from_dict({("Halpha", "Hbeta"): (4.2, 0.3)})
    >>> lrd.n_ratios
    1
    """

    numerators: tuple[str, ...]
    denominators: tuple[str, ...]
    ratios: jnp.ndarray = dataclasses.field(hash=False)
    errors: jnp.ndarray = dataclasses.field(hash=False)
    numerator_waves: jnp.ndarray = dataclasses.field(hash=False)
    denominator_waves: jnp.ndarray = dataclasses.field(hash=False)
    log_space: bool = False

    def __post_init__(self) -> None:
        n = len(self.numerators)
        if n == 0:
            raise ValueError("LineRatioData requires at least one ratio.")
        if len(self.denominators) != n:
            raise ValueError(
                f"numerators ({n}) and denominators ({len(self.denominators)}) "
                f"must have the same length."
            )
        for field_name in ("ratios", "errors", "numerator_waves", "denominator_waves"):
            arr = jnp.asarray(getattr(self, field_name))
            if arr.shape != (n,):
                raise ValueError(
                    f"{field_name} shape {arr.shape} does not match expected ({n},) "
                    f"for {n} ratios."
                )

    @property
    def n_ratios(self) -> int:
        """Number of observed line ratios."""
        return len(self.numerators)

    def model_ratio(
        self, model_numerator_fluxes: jnp.ndarray, model_denominator_fluxes: jnp.ndarray
    ) -> jnp.ndarray:
        """Model ratio (or log10 ratio) from per-line model fluxes.

        Parameters
        ----------
        model_numerator_fluxes, model_denominator_fluxes: ndarray, shape (n_ratios,)
            Model fluxes for the numerator and denominator lines (any
            consistent unit, the ratio is dimensionless).

        Returns
        -------
        ndarray, shape (n_ratios,)
            ``F_num / F_den`` (or ``log10`` thereof when ``log_space``).

        Notes
        -----
        **JIT-compatible**: yes. **Gradient-safe**: yes.
        """
        denom = jnp.where(jnp.abs(model_denominator_fluxes) > 0, model_denominator_fluxes, 1e-30)
        ratio = model_numerator_fluxes / denom
        if self.log_space:
            return jnp.log10(jnp.maximum(ratio, 1e-30))
        return ratio

    def chi2(
        self, model_numerator_fluxes: jnp.ndarray, model_denominator_fluxes: jnp.ndarray
    ) -> jnp.ndarray:
        """Chi-squared statistic summed over ratios.

        Notes
        -----
        **JIT-compatible**: yes. **Gradient-safe**: yes, differentiable w.r.t.
        both flux arrays.
        """
        model = self.model_ratio(model_numerator_fluxes, model_denominator_fluxes)
        residual = (self.ratios - model) / self.errors
        return jnp.sum(residual**2)

    def log_likelihood(
        self, model_numerator_fluxes: jnp.ndarray, model_denominator_fluxes: jnp.ndarray
    ) -> jnp.ndarray:
        """Gaussian (linear) / log-normal (``log_space``) log-likelihood.

        .. math::

            \\ln L = -\\tfrac{1}{2}\\left(\\frac{r_{\\rm obs} - r_{\\rm mod}}
            {\\sigma}\\right)^2 - \\ln\\sigma - \\tfrac{1}{2}\\ln 2\\pi

        where :math:`r` is the (log) ratio. Summed over all ratios.

        Notes
        -----
        **JIT-compatible**: yes. **Gradient-safe**: yes.
        """
        model = self.model_ratio(model_numerator_fluxes, model_denominator_fluxes)
        residual = (self.ratios - model) / self.errors
        ll = -0.5 * residual**2 - jnp.log(self.errors) - 0.5 * jnp.log(2.0 * jnp.pi)
        return jnp.sum(ll)

    @classmethod
    def from_dict(
        cls,
        ratio_data: dict[tuple[str, str], tuple[float, float]],
        *,
        log_space: bool = False,
    ) -> LineRatioData:
        """Construct from ``{(num_name, den_name): (ratio, error)}``.

        Line names are looked up in the standard optical catalog for their
        rest-frame vacuum wavelengths.

        Parameters
        ----------
        ratio_data: dict[tuple[str, str], tuple[float, float]]
            Mapping from a ``(numerator, denominator)`` name pair to a
            ``(ratio, error)`` tuple. E.g.
            ``{("Halpha", "Hbeta"): (4.2, 0.3)}``.
        log_space: bool, default False
            Interpret ratios/errors as log10 values (log-normal likelihood).

        Raises
        ------
        ValueError
            If any line name is not found in the standard catalog.
        """
        numerators, denominators = [], []
        ratios, errors = [], []
        num_waves, den_waves = [], []
        for (num, den), (ratio, error) in ratio_data.items():
            for name in (num, den):
                if name not in _NAME_TO_WAVELENGTH:
                    available = sorted(_NAME_TO_WAVELENGTH.keys())
                    raise ValueError(f"Unknown line name {name!r}. Available: {available}")
            numerators.append(num)
            denominators.append(den)
            ratios.append(ratio)
            errors.append(error)
            num_waves.append(_NAME_TO_WAVELENGTH[num])
            den_waves.append(_NAME_TO_WAVELENGTH[den])
        return cls(
            numerators=tuple(numerators),
            denominators=tuple(denominators),
            ratios=jnp.array(ratios),
            errors=jnp.array(errors),
            numerator_waves=jnp.array(num_waves),
            denominator_waves=jnp.array(den_waves),
            log_space=log_space,
        )

    def summary(self) -> str:
        """One-line summary for logging."""
        pairs = ", ".join(
            f"{n}/{d}" for n, d in zip(self.numerators, self.denominators, strict=True)
        )
        space = "log10" if self.log_space else "linear"
        return f"{self.n_ratios} line ratios [{space}] ({pairs})"
