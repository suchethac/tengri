"""Noise model configuration for observations.

Declarative specification of the noise model: calibration floor and
likelihood shape (Gaussian vs Student-t). Replaces the pattern of
manually adding noise_frac_cal / noise_dof to Parameters.
"""

from __future__ import annotations

import dataclasses

from tengri.parameters.priors import Distribution, Fixed


@dataclasses.dataclass(frozen=True)
class NoiseModel:
    """Noise model configuration.

    Parameters
    ----------
    calibration_floor : float or Distribution
        Fractional calibration floor added in quadrature with observational
        noise: ``sigma_eff = sqrt(sigma_obs^2 + (f_cal * model)^2)``.
        A float value becomes ``Fixed(value)``; a ``Distribution`` (e.g.
        ``Uniform(0.01, 0.15)``) makes it a free parameter during inference.
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
    calibration_floor : float or Distribution
        Fractional calibration uncertainty floor.
    student_t_dof : float or None
        Student-t degrees of freedom (or None for Gaussian).

    Notes
    -----
    **Immutable container**: A frozen dataclass. Fields are read-only by
    convention.
    -----
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

    calibration_floor: float | Distribution = 0.0
    student_t_dof: float | None = None

    def get_params(self) -> dict[str, Distribution]:
        """Return Parameters entries for the noise model.

        Returns
        -------
        dict[str, Distribution]
            Mapping of parameter names (``"noise_frac_cal"``, ``"noise_dof"``)
            to Distribution objects. Empty dict if no noise parameters are needed.

        Notes
        -----
        Called by ``Observation.get_all_params()`` to register noise model
        hyperparameters with the inference engine. Parameters are only included
        if they are non-trivial (calibration floor > 0 or Student-t dof is not None).

        """
        params: dict[str, Distribution] = {}

        if isinstance(self.calibration_floor, Distribution):
            params["noise_frac_cal"] = self.calibration_floor
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
        elif self.calibration_floor > 0:
            parts.append(f"cal floor={self.calibration_floor:.3f} (fixed)")
        if self.student_t_dof is not None:
            parts.append(f"Student-t dof={self.student_t_dof}")
        return ", ".join(parts) if parts else "Gaussian (default)"
