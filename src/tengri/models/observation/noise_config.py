"""Noise model configuration for observations.

Declarative specification of the noise model: calibration floor and
likelihood shape (Gaussian vs Student-t). Replaces the pattern of
manually adding noise_frac_cal / noise_dof to ParamSpec.
"""

from __future__ import annotations

import dataclasses

from tengri.distributions import Distribution, Fixed


@dataclasses.dataclass(frozen=True)
class NoiseConfig:
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
    """

    calibration_floor: float | Distribution = 0.0
    student_t_dof: float | None = None

    def get_params(self) -> dict[str, Distribution]:
        """Return ParamSpec entries for the noise model.

        Returns
        -------
        dict
            Mapping of parameter names to Distribution objects.
            Empty dict if no noise parameters are needed.
        """
        params: dict[str, Distribution] = {}

        if isinstance(self.calibration_floor, Distribution):
            params["noise_frac_cal"] = self.calibration_floor
        elif self.calibration_floor > 0:
            params["noise_frac_cal"] = Fixed(self.calibration_floor)

        if self.student_t_dof is not None:
            params["noise_dof"] = Fixed(self.student_t_dof)

        return params
