"""Spectroscopic observation configuration.

Declarative specification of the spectroscopic instrument: wavelength grid,
resolution profile, LSF settings, calibration polynomial, and emission-line
marginalization. Includes factory methods for common instruments.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from tengri.parameters.priors import Distribution, Gaussian


@dataclasses.dataclass(frozen=True)
class Spectroscopy:
    """Spectroscopic observation configuration.

    Parameters
    ----------
    wave_obs : jnp.ndarray
        Observed-frame wavelength grid [Angstrom], shape ``(n_pix,)``.
    resolution : float, jnp.ndarray, or None
        Spectral resolution ``R = lambda / delta_lambda``.
        Scalar for constant R, per-pixel array for wavelength-dependent,
        or None to skip LSF convolution. Default: None.
    sigma_lib_kms : float
        SSP library velocity dispersion [km/s] to subtract in quadrature
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

        - ``"off"``: No emission line fitting (default).
        - ``"fixed"``: Lines from nebular model only.
        - ``"marginalized"``: Analytically marginalize line amplitudes
          (recommended for spectroscopic fitting).
        - ``"fitted"``: Line amplitudes as free MCMC parameters.

        Default: ``"off"``.
    eline_catalog : LineList or None
        Line catalog. ``None`` falls back to ``LineList.default_13()`` for
        Use ``LineList.default_optical()`` for
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
        Minimum FWHM for the broad component [km/s]. Default: 500.0.
    covariance : jnp.ndarray or None
        Full spectral covariance matrix, shape ``(n_pix, n_pix)``.
        When provided, the likelihood uses ``diff @ C^{-1} @ diff``
        instead of per-pixel ``sum((diff/sigma)^2)``. The inverse is
        precomputed at construction time.  Default: None (diagonal noise).

    Notes
    -----
    A frozen dataclass that encapsulates spectroscopic instrument metadata,
    including wavelength grid, resolution profile, calibration strategy,
    and emission-line fitting configuration. Precomputes the inverse
    covariance matrix at initialization for efficient likelihood evaluation.
    Used by SEDModel to configure spectral prediction and by the inference
    engine to set up calibration priors.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import Spectroscopy
    >>> wave = jnp.linspace(4000.0, 9000.0, 500)
    >>> spec = Spectroscopy(wave_obs=wave, resolution=1000.0)
    >>> spec.n_pixels
    500

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
    covariance: jnp.ndarray | None = dataclasses.field(default=None, hash=False)

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
        if self.covariance is not None:
            cov = jnp.asarray(self.covariance)
            n = len(self.wave_obs)
            if cov.shape != (n, n):
                raise ValueError(
                    f"covariance shape {cov.shape} does not match expected ({n}, {n})"
                )
            object.__setattr__(self, "_cov_inv", jnp.linalg.inv(cov))
        else:
            object.__setattr__(self, "_cov_inv", None)

    # ── Properties ────────────────────────────────────────────────

    @property
    def n_pixels(self) -> int:
        """Number of spectral pixels.

        Returns
        -------
        int
            Number of wavelength grid points.

        Notes
        -----
        Read-only property; computed from the length of ``wave_obs``.

        """
        return len(self.wave_obs)

    @property
    def has_covariance(self) -> bool:
        """Whether a full covariance matrix is configured.

        Returns
        -------
        bool
            True if covariance matrix is present.

        Notes
        -----
        Read-only property; determined at initialization.

        """
        return self._cov_inv is not None

    @property
    def cov_inv(self) -> jnp.ndarray | None:
        """Precomputed inverse covariance matrix, or None.

        Returns
        -------
        ndarray or None
            Inverse covariance matrix, shape ``(n_pix, n_pix)``, or None
            if diagonal noise is assumed.

        Notes
        -----
        The inverse is precomputed at initialization for efficient likelihood
        evaluation. This property is read-only.

        """
        return self._cov_inv

    @property
    def has_lsf(self) -> bool:
        """Whether LSF convolution is configured.

        Returns
        -------
        bool
            True if resolution profile is specified.

        Notes
        -----
        Read-only property; determines whether LSF convolution is applied
        in the forward model.

        """
        return self.resolution is not None

    @property
    def has_calibration(self) -> bool:
        """Whether a calibration polynomial is configured.

        Returns
        -------
        bool
            True if calibration order is > 0.

        Notes
        -----
        Read-only property; determines whether calibration coefficients
        are registered as free parameters.

        """
        return self.calibration_order > 0

    @property
    def has_eline_fitting(self) -> bool:
        """True if emission lines are being fit (marginalized or fitted mode).

        Returns
        -------
        bool
            True if eline_mode is "marginalized" or "fitted".

        Notes
        -----
        Read-only property; determines whether emission line amplitudes are
        treated as free parameters or analytically marginalized.

        """
        return self.eline_mode in ("marginalized", "fitted")

    @property
    def effective_catalog(self) -> object:
        """Return the catalog, falling back to default_13() if not set.

        Returns
        -------
        LineList
            The active line catalog (either explicitly configured or default).

        Notes
        -----
        Provides convenient fallback logic: if eline_catalog is None,
        automatically returns the default 13-line catalog.

        """
        from tengri.observation.line_list import LineList

        if self.eline_catalog is not None:
            return self.eline_catalog
        return LineList.default_13()

    # ── Parameter helpers ─────────────────────────────────────────

    def get_calibration_params(self) -> dict[str, Distribution]:
        """Return Parameters entries for calibration polynomial.

        Returns
        -------
        dict
            Mapping of calibration coefficient names (``cal_c1``, ..., ``cal_cN``)
            to ``Gaussian(0, 0.1)`` priors. Empty dict if ``calibration_order == 0``.

        Notes
        -----
        Called by Observation.get_all_params() to register observation-level
        parameters with the inference engine. Each coefficient has a weak
        Gaussian prior centered at unity (in log space: 0).

        """
        if self.calibration_order == 0:
            return {}
        return {f"cal_c{i + 1}": Gaussian(0.0, 0.1) for i in range(self.calibration_order)}

    # ── Instrument factories ──────────────────────────────────────

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
            Observed wavelength grid [Angstrom].
        resolution : float, array, or None
            Spectral resolution R(lambda).
        sigma_lib_kms : float
            SSP library resolution [km/s]. Default: 70.0.
        calibration_order : int
            Chebyshev calibration order. Default: 0.
        **kwargs
            Passed to ``Spectroscopy``.

        Returns
        -------
        Spectroscopy
            Configured spectroscopy object.

        Notes
        -----
        Internal helper used by instrument-specific factories (nirspec_prism,
        nirspec_g140m, constant_r). Not intended for direct public use.

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
        """JWST NIRSpec PRISM: variable R ~ 30-330 (Jakobsen+2022).

        Parameters
        ----------
        wave_obs : jnp.ndarray
            Observed-frame wavelength grid [Angstrom].
        **kwargs
            Additional arguments passed to Spectroscopy.

        Returns
        -------
        Spectroscopy
            Configured NIRSpec PRISM spectroscopy.

        Notes
        -----
        Applies wavelength-dependent resolution appropriate for NIRSpec's
        PRISM mode. Resolution varies from R~30 in the red to R~330 in
        the blue (Jakobsen et al. 2022).

        """
        from tengri.observation.spectrum import nirspec_prism_resolution

        wave_jax = jnp.asarray(wave_obs)
        resolution = nirspec_prism_resolution(wave_jax / 1e4)
        return Spectroscopy._from_resolution(wave_jax, resolution, **kwargs)

    @staticmethod
    def nirspec_g140m(wave_obs: jnp.ndarray, **kwargs) -> Spectroscopy:
        """JWST NIRSpec G140M: roughly constant R ~ 1000.

        Parameters
        ----------
        wave_obs : jnp.ndarray
            Observed-frame wavelength grid [Angstrom].
        **kwargs
            Additional arguments passed to Spectroscopy.

        Returns
        -------
        Spectroscopy
            Configured NIRSpec G140M spectroscopy.

        Notes
        -----
        Applies wavelength-dependent resolution for NIRSpec's medium-resolution
        G140M mode. Resolution is approximately constant at R~1000 across
        the wavelength range.

        """
        from tengri.observation.spectrum import nirspec_g140m_resolution

        wave_jax = jnp.asarray(wave_obs)
        resolution = nirspec_g140m_resolution(wave_jax / 1e4)
        return Spectroscopy._from_resolution(wave_jax, resolution, **kwargs)

    @staticmethod
    def constant_r(wave_obs: jnp.ndarray, R: float, **kwargs) -> Spectroscopy:
        """Constant-resolution spectrograph.

        Parameters
        ----------
        wave_obs : jnp.ndarray
            Observed-frame wavelength grid [Angstrom].
        R : float
            Spectral resolution (constant across all wavelengths).
        **kwargs
            Additional arguments passed to Spectroscopy.

        Returns
        -------
        Spectroscopy
            Configured spectrograph with constant resolution.

        Notes
        -----
        Convenient factory for instruments with wavelength-independent
        spectral resolution, such as low-resolution JWST NIRSpec PRISM
        approximations or ideal spectrographs.

        """
        return Spectroscopy._from_resolution(wave_obs, float(R), **kwargs)

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
            Observed-frame wavelength grid [Angstrom].
        resolution : float
            Spectral resolution R = lambda/delta_lambda. Default: 2500 (DESI).
        **kwargs
            Additional fields passed to ``Spectroscopy``.

        Returns
        -------
        Spectroscopy
            Fully configured spectroscopy for DESI-like emission-line fitting.

        Notes
        -----
        This configuration mirrors DESI's optical spectroscopy capabilities,
        including marginalized emission line amplitudes with CLOUDY-based
        priors and atomic physics constraints. Suitable for large spectroscopic
        surveys of emission-line galaxies.

        """
        from tengri.observation.line_list import LineList

        return cls(
            wave_obs=wave_obs,
            resolution=resolution,
            calibration_order=3,
            eline_mode="marginalized",
            eline_catalog=LineList.default_optical(),
            eline_prior_type="cloudy",
            eline_fix_doublets=True,
            **kwargs,
        )

    # ── Summary ───────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a one-line summary of the spectroscopy configuration.

        Returns
        -------
        str
            Comma-separated summary (e.g., "500 pixels, R=2500, eline=marginalized").
            Includes pixel count, resolution, calibration order, emission-line mode,
            and covariance info.

        Notes
        -----
        Used for logging and diagnostics. Provides a compact, human-readable
        representation of the instrument configuration. Intended for display to users,
        not for programmatic parsing.

        """
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
        if self.has_covariance:
            parts.append("cov_matrix")
        return ", ".join(parts)


# ── Wavelength masking ────────────────────────────────────────────


def apply_wavelength_mask(
    noise: jnp.ndarray,
    wave_obs: jnp.ndarray,
    mask_ranges: list[tuple[float, float]],
) -> jnp.ndarray:
    """Mask spectral regions by setting noise to infinity.

    Masked pixels are effectively removed from the likelihood (chi2
    contribution → 0). Returns a new array — does not mutate the input.

    Parameters
    ----------
    noise : array, shape (n_pix,)
        Per-pixel 1-sigma noise [flux units].
    wave_obs : array, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom].
    mask_ranges : list of (lo, hi)
        Wavelength ranges to mask [Angstrom]. Each ``(lo, hi)`` pair
        defines a region where ``lo <= wave <= hi`` is masked.

    Returns
    -------
    jnp.ndarray
        Copy of ``noise`` with masked pixels set to ``inf``.

    Notes
    -----
    Immutable operation: returns a new array without modifying the input.
    Masked pixels contribute zero to the likelihood due to infinite noise.
    Useful for excluding sky emission lines, detector artifacts, or other
    unreliable spectral regions from inference.

    Examples
    --------
    Mask the 5577 A sky line and a detector gap::

        noise_masked = apply_wavelength_mask(
            noise,
            wave_obs,
            mask_ranges=[(5560, 5590), (7580, 7680)],
        )

    """
    noise = jnp.array(noise)
    wave_obs = jnp.array(wave_obs)
    for lo, hi in mask_ranges:
        in_range = (wave_obs >= lo) & (wave_obs <= hi)
        noise = jnp.where(in_range, jnp.inf, noise)
    return noise


# ── Deprecated alias — removed in tengri v1.0 ─────────────────────
