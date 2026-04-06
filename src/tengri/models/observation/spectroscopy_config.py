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
class Spectroscopy:
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
    eline_prior_sigma : float
        Prior width on emission line amplitudes for marginalization.
        Default: 100.0.
    eline_mode : str
        Emission line fitting mode. One of:

        - ``"off"``: No emission line fitting (default, backward compatible).
        - ``"fixed"``: Lines from nebular model only.
        - ``"marginalized"``: Analytically marginalize line amplitudes
          (recommended for spectroscopic fitting).
        - ``"fitted"``: Line amplitudes as free MCMC parameters.

        Default: ``"off"``.
    eline_catalog : LineCatalog or None
        Line catalog. ``None`` falls back to ``LineCatalog.default_13()`` for
        backward compatibility. Use ``LineCatalog.default_optical()`` for
        FastSpecFit parity. Default: None.
    eline_prior_type : str
        Prior type for line amplitudes. One of ``"flat"`` (uninformative) or
        ``"cloudy"`` (CLOUDY-grid-interpolated). Default: ``"flat"``.
    eline_prior_width_dex : float
        Prior scatter in dex for the ``"cloudy"`` prior. Default: 0.3.
    eline_fix_doublets : bool
        Enforce atomic physics doublet ratios. Default: True.
    eline_broad : bool
        Enable broad component for AGN candidate lines. Default: False.
    eline_broad_fwhm_min_kms : float
        Minimum FWHM for the broad component in km/s. Default: 500.0.
    """

    wave_obs: jnp.ndarray = dataclasses.field(hash=False)
    resolution: float | jnp.ndarray | None = dataclasses.field(default=None, hash=False)
    sigma_lib_kms: float = 70.0
    lsf_n_bins: int = 16
    calibration_order: int = 0
    eline_prior_sigma: float = 100.0
    eline_mode: str = "off"
    eline_catalog: object | None = dataclasses.field(default=None, hash=False)
    eline_prior_type: str = "flat"
    eline_prior_width_dex: float = 0.3
    eline_fix_doublets: bool = True
    eline_broad: bool = False
    eline_broad_fwhm_min_kms: float = 500.0

    def __post_init__(self) -> None:
        _valid_modes = ("off", "fixed", "marginalized", "fitted")
        if self.eline_mode not in _valid_modes:
            raise ValueError(f"eline_mode must be one of {_valid_modes}, got {self.eline_mode!r}")
        if self.resolution is not None and not isinstance(self.resolution, (int, float)):
            res_arr = jnp.asarray(self.resolution)
            if res_arr.ndim > 0 and res_arr.shape[0] != len(self.wave_obs):
                raise ValueError(
                    f"resolution array length {res_arr.shape[0]} does not match "
                    f"wave_obs length {len(self.wave_obs)}"
                )

    # -------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------

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

    @property
    def has_eline_fitting(self) -> bool:
        """True if emission lines are being fit (marginalized or fitted mode)."""
        return self.eline_mode in ("marginalized", "fitted")

    @property
    def effective_catalog(self) -> object:
        """Return the catalog, falling back to default_13() if not set.

        Returns
        -------
        LineCatalog
            The active line catalog.
        """
        from tengri.models.observation.line_list import LineCatalog

        if self.eline_catalog is not None:
            return self.eline_catalog
        return LineCatalog.default_13()

    # -------------------------------------------------------------------
    # Parameter helpers
    # -------------------------------------------------------------------

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
        **kwargs,
    ) -> Spectroscopy:
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
        **kwargs
            Passed to ``Spectroscopy``.
        """
        return Spectroscopy(
            wave_obs=jnp.asarray(wave_obs),
            resolution=resolution,
            sigma_lib_kms=sigma_lib_kms,
            calibration_order=calibration_order,
            **kwargs,
        )

    @staticmethod
    def nirspec_prism(wave_obs: jnp.ndarray, **kwargs) -> Spectroscopy:
        """JWST NIRSpec PRISM: variable R ~ 30-330 (Jakobsen+2022)."""
        from tengri.models.observation.spectroscopy import nirspec_prism_resolution

        wave_jax = jnp.asarray(wave_obs)
        resolution = nirspec_prism_resolution(wave_jax / 1e4)
        return SpectroscopyConfig._from_resolution(wave_jax, resolution, **kwargs)

    @staticmethod
    def nirspec_g140m(wave_obs: jnp.ndarray, **kwargs) -> Spectroscopy:
        """JWST NIRSpec G140M: roughly constant R ~ 1000."""
        from tengri.models.observation.spectroscopy import nirspec_g140m_resolution

        wave_jax = jnp.asarray(wave_obs)
        resolution = nirspec_g140m_resolution(wave_jax / 1e4)
        return SpectroscopyConfig._from_resolution(wave_jax, resolution, **kwargs)

    @staticmethod
    def constant_r(wave_obs: jnp.ndarray, R: float, **kwargs) -> Spectroscopy:
        """Constant-resolution spectrograph."""
        return SpectroscopyConfig._from_resolution(wave_obs, float(R), **kwargs)

    @classmethod
    def desi_like(
        cls, wave_obs: jnp.ndarray, resolution: float = 2500.0, **kwargs
    ) -> Spectroscopy:
        """DESI-like spectroscopic configuration with full line fitting.

        Pre-configured with:

        - ~40-line optical catalog (FastSpecFit parity)
        - Marginalized emission lines with CLOUDY priors
        - Doublet constraints enabled
        - Calibration polynomial order 3

        Parameters
        ----------
        wave_obs : jnp.ndarray
            Observed-frame wavelength grid (Angstrom).
        resolution : float
            Spectral resolution R = lambda/delta_lambda. Default: 2500 (DESI).
        **kwargs
            Additional fields passed to ``Spectroscopy``.

        Returns
        -------
        SpectroscopyConfig
            Configured for DESI-like spectroscopic fitting.
        """
        from tengri.models.observation.line_list import LineCatalog

        return cls(
            wave_obs=wave_obs,
            resolution=resolution,
            calibration_order=3,
            eline_mode="marginalized",
            eline_catalog=LineCatalog.default_optical(),
            eline_prior_type="cloudy",
            eline_fix_doublets=True,
            **kwargs,
        )

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------

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
        if self.eline_mode != "off":
            parts.append(f"eline={self.eline_mode}")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Deprecated alias — removed in tengri v1.0
# ---------------------------------------------------------------------------


def _make_deprecated_spectroscopy_config():
    import warnings

    class SpectroscopyConfig(Spectroscopy):
        def __init__(self, *args, **kwargs):
            warnings.warn(
                "SpectroscopyConfig is deprecated. Use Spectroscopy instead. "
                "Will be removed in tengri v1.0.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    SpectroscopyConfig.__name__ = "SpectroscopyConfig"
    SpectroscopyConfig.__qualname__ = "SpectroscopyConfig"
    return SpectroscopyConfig


SpectroscopyConfig = _make_deprecated_spectroscopy_config()
