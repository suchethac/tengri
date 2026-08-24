# SPDX-License-Identifier: BSD-3-Clause
"""Noise model configuration for observations.

Declarative specification of the noise model: calibration floor and
likelihood shape (Gaussian vs Student-t). Replaces the pattern of
manually adding noise_frac_cal / noise_dof to Parameters.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from tengri.parameters.priors import Distribution, Fixed


@dataclasses.dataclass(frozen=True)
class NoiseModel:
    """Noise model configuration.

    Parameters
    ----------
    calibration_floor : float, Distribution, or array
        Fractional calibration floor added in quadrature with observational
        noise: ``sigma_eff = sqrt(sigma_obs^2 + (f_cal * model)^2)``.
        A float value becomes ``Fixed(value)``; a ``Distribution`` (e.g.
        ``Uniform(0.01, 0.15)``) makes it a free parameter during inference.
        An array (shape ``(n_filters,)``) specifies a per-band fixed floor;
        arrays of Distributions are not supported.
        Default: 0.0 (no calibration floor).
    student_t_dof : float or None
        Degrees of freedom for a Student-t likelihood (heavier tails for
        outlier robustness). ``None`` uses a standard Gaussian likelihood.
        Default: None.

    Returns
    -------
    NoiseModel
        Noise model instance with configuration validated.

    Attributes
    ----------
    calibration_floor : float, Distribution, or array
        Fractional calibration uncertainty floor.
    student_t_dof : float or None
        Student-t degrees of freedom (or None for Gaussian).

    Notes
    -----
    **Immutable container**: A frozen dataclass. Fields are read-only by
    convention.

    A frozen dataclass encapsulating noise model configuration. Replaces
    the older pattern of manually creating ``noise_frac_cal`` and ``noise_dof``
    parameters. Primarily used to register observation-level hyperparameters
    with the inference engine.

    Examples
    --------
    >>> from tengri import NoiseModel, Uniform
    >>> nm = NoiseModel(calibration_floor=Uniform(0.01, 0.1))
    >>> list(nm.get_params().keys())
    ['noise_frac_cal']

    """

    calibration_floor: float | Distribution | jnp.ndarray = 0.0
    student_t_dof: float | None = None

    def __post_init__(self) -> None:
        """Validate that array calibration floors are not arrays of Distributions."""
        # Check if calibration_floor is an array with object dtype
        if (
            isinstance(self.calibration_floor, jnp.ndarray)
            and self.calibration_floor.dtype == object
        ):
            # Check if any element is a Distribution
            for elem in self.calibration_floor.flat:
                if isinstance(elem, Distribution):
                    msg = (
                        "per-band free floor (array of Distributions) is out of scope "
                        "only scalar or array of numeric values are supported"
                    )
                    raise TypeError(msg)

    def validate_array_length(self, n_filters: int) -> None:
        """Validate that array calibration_floor matches the number of filters.

        Parameters
        ----------
        n_filters : int
            Expected number of filters.

        Raises
        ------
        ValueError
            If calibration_floor is an array with length != n_filters.
        """
        if (
            isinstance(self.calibration_floor, jnp.ndarray)
            and len(self.calibration_floor) != n_filters
        ):
            msg = (
                f"calibration_floor array length mismatch: "
                f"got {len(self.calibration_floor)}, expected {n_filters}"
            )
            raise ValueError(msg)

    def get_params(self) -> dict[str, object]:
        """Return Parameters entries for the noise model.

        Returns
        -------
        dict[str, Distribution]
            Mapping of parameter names (``"noise_frac_cal"``, ``"noise_dof"``)
            to Distribution objects. Empty dict if no noise parameters are needed.
            Per-band array floors are fixed and not added as parameters.

        Notes
        -----
        Called by ``Observation.get_all_params()`` to register noise model
        hyperparameters with the inference engine. Parameters are only included
        if they are non-trivial (calibration floor > 0 or Student-t dof is not None).
        Array calibration floors are fixed and not added as free parameters.

        """
        params: dict[str, object] = {}

        if isinstance(self.calibration_floor, Distribution):
            params["noise_frac_cal"] = self.calibration_floor
        elif isinstance(self.calibration_floor, jnp.ndarray):
            # Array floors are fixed, not added as parameters
            pass
        elif self.calibration_floor > 0:
            params["noise_frac_cal"] = Fixed(self.calibration_floor)

        if self.student_t_dof is not None:
            params["noise_dof"] = Fixed(self.student_t_dof)

        return params

    def summary(self) -> str:
        """Return a one-line summary of the noise configuration.

        Returns
        -------
        str
            Single-line summary string with calibration floor and likelihood
            settings (e.g., ``"cal floor=0.05 (fixed), Student-t dof=10"``).

        Notes
        -----
        Used for logging and diagnostics. Returns ``"Gaussian (default)"`` if
        no custom noise settings are configured.

        """
        parts = []
        if isinstance(self.calibration_floor, Distribution):
            parts.append(f"cal floor={self.calibration_floor!r} (free)")
        elif isinstance(self.calibration_floor, jnp.ndarray):
            parts.append(f"cal floor=per-band array len={len(self.calibration_floor)} (fixed)")
        elif self.calibration_floor > 0:
            parts.append(f"cal floor={self.calibration_floor:.3f} (fixed)")
        if self.student_t_dof is not None:
            parts.append(f"Student-t dof={self.student_t_dof}")
        return ", ".join(parts) if parts else "Gaussian (default)"
