"""Spectroscopic observation configuration.

Declarative specification of the spectroscopic instrument: wavelength grid,
resolution profile, LSF settings, calibration polynomial, and emission-line
marginalization. Includes factory methods for common instruments.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from tengri.distributions import Distribution, Gaussian


@dataclasses.dataclass(frozen=True)
class SpectroscopyConfig:
    """Spectroscopic observation configuration.

    Parameters
    ----------
    wave_obs : jnp.ndarray
        Observed-frame wavelength grid (Angstrom), shape ``(n_pix,)``.
    resolution : float, jnp.ndarray, or None
        Spectral resolution ``R = lambda / delta_lambda``.
        Scalar for constant R, per-pixel array for wavelength-dependent,
        or None to skip LSF convolution. Default: None.
    sigma_lib_kms : float
        SSP library velocity dispersion (km/s) to subtract in quadrature
        when applying the LSF. Default: 70.0 (MILES).
    lsf_n_bins : int
        Number of bins for piecewise constant approximation of
        variable-R LSF convolution. Default: 16.
    calibration_order : int
        Order of multiplicative Chebyshev calibration polynomial.
        0 = no calibration (default). Order N adds N free params
        (``cal_c1``, ..., ``cal_cN``) with ``Gaussian(0, 0.1)`` priors.
    eline_marginalize : bool
        Whether to analytically marginalize emission line amplitudes
        during likelihood computation. Default: False.
    eline_wavelengths : jnp.ndarray or None
        Custom emission line wavelengths (rest-frame Angstrom). If None,
        uses the default 13-line list (Balmer + forbidden). Default: None.
    eline_prior_sigma : float
        Prior width on emission line amplitudes for marginalization.
        Default: 100.0.
    """

    wave_obs: jnp.ndarray = dataclasses.field(hash=False)
    resolution: float | jnp.ndarray | None = dataclasses.field(default=None, hash=False)
    sigma_lib_kms: float = 70.0
    lsf_n_bins: int = 16
    calibration_order: int = 0
    eline_marginalize: bool = False
    eline_wavelengths: jnp.ndarray | None = dataclasses.field(default=None, hash=False)
    eline_prior_sigma: float = 100.0

    @property
    def n_pixels(self) -> int:
        """Number of spectral pixels."""
        return len(self.wave_obs)

    @property
    def has_lsf(self) -> bool:
        """Whether LSF convolution is configured."""
        return self.resolution is not None

    @property
    def has_calibration(self) -> bool:
        """Whether a calibration polynomial is configured."""
        return self.calibration_order > 0

    def get_calibration_params(self) -> dict[str, Distribution]:
        """Return ParamSpec entries for calibration polynomial.

        Returns
        -------
        dict
            Mapping ``cal_c1``, ..., ``cal_cN`` to ``Gaussian(0, 0.1)``
            priors. Empty dict if ``calibration_order == 0``.
        """
        if self.calibration_order == 0:
            return {}
        return {f"cal_c{i + 1}": Gaussian(0.0, 0.1) for i in range(self.calibration_order)}

    # -------------------------------------------------------------------
    # Instrument factories
    # -------------------------------------------------------------------

    @staticmethod
    def _from_resolution(
        wave_obs: jnp.ndarray,
        resolution: float | jnp.ndarray | None,
        sigma_lib_kms: float = 70.0,
        calibration_order: int = 0,
        eline_marginalize: bool = False,
        **kwargs,
    ) -> SpectroscopyConfig:
        """Shared constructor for instrument factories.

        Parameters
        ----------
        wave_obs : jnp.ndarray
            Observed wavelength grid (Angstrom).
        resolution : float, array, or None
            Spectral resolution R(lambda).
        sigma_lib_kms : float
            SSP library resolution. Default: 70.0.
        calibration_order : int
            Chebyshev calibration order. Default: 0.
        eline_marginalize : bool
            Marginalize emission lines. Default: False.
        **kwargs
            Passed to ``SpectroscopyConfig``.
        """
        return SpectroscopyConfig(
            wave_obs=jnp.asarray(wave_obs),
            resolution=resolution,
            sigma_lib_kms=sigma_lib_kms,
            calibration_order=calibration_order,
            eline_marginalize=eline_marginalize,
            **kwargs,
        )

    @staticmethod
    def nirspec_prism(wave_obs: jnp.ndarray, **kwargs) -> SpectroscopyConfig:
        """JWST NIRSpec PRISM: variable R ~ 30-330 (Jakobsen+2022)."""
        from tengri.models.observation.spectroscopy import nirspec_prism_resolution

        wave_jax = jnp.asarray(wave_obs)
        resolution = nirspec_prism_resolution(wave_jax / 1e4)
        return SpectroscopyConfig._from_resolution(wave_jax, resolution, **kwargs)

    @staticmethod
    def nirspec_g140m(wave_obs: jnp.ndarray, **kwargs) -> SpectroscopyConfig:
        """JWST NIRSpec G140M: roughly constant R ~ 1000."""
        from tengri.models.observation.spectroscopy import nirspec_g140m_resolution

        wave_jax = jnp.asarray(wave_obs)
        resolution = nirspec_g140m_resolution(wave_jax / 1e4)
        return SpectroscopyConfig._from_resolution(wave_jax, resolution, **kwargs)

    @staticmethod
    def constant_r(
        wave_obs: jnp.ndarray, R: float, **kwargs
    ) -> SpectroscopyConfig:
        """Constant-resolution spectrograph."""
        return SpectroscopyConfig._from_resolution(
            wave_obs, float(R), **kwargs
        )

    def summary(self) -> str:
        """Return a one-line summary of the spectroscopy configuration."""
        parts = [f"{self.n_pixels} pixels"]
        if self.has_lsf:
            if isinstance(self.resolution, (int, float)):
                parts.append(f"R={self.resolution:.0f}")
            else:
                r_arr = jnp.asarray(self.resolution)
                parts.append(f"R={float(r_arr.min()):.0f}-{float(r_arr.max()):.0f}")
        if self.has_calibration:
            parts.append(f"cal order={self.calibration_order}")
        if self.eline_marginalize:
            parts.append("eline marg")
        return ", ".join(parts)
