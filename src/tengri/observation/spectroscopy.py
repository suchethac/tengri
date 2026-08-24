# SPDX-License-Identifier: BSD-3-Clause
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
    wave_obs: jnp.ndarray
        Observed-frame wavelength grid [Angstrom], shape ``(n_pix,)``.
    resolution: float, jnp.ndarray, or None
        Spectral resolution ``R = lambda / delta_lambda``.
        Scalar for constant R, per-pixel array for wavelength-dependent,
        or None to skip LSF convolution. Default: None.
    sigma_lib_kms: float
        SSP library velocity dispersion [km/s] to subtract in quadrature
        when applying the LSF. Default: 70.0 (MILES).
    lsf_n_bins: int
        Number of bins for piecewise constant approximation of
        variable-R LSF convolution. Default: 16.
    calibration_order: int
        Order of multiplicative Chebyshev calibration polynomial.
        0 = no calibration (default). Order N adds N free params
        (``cal_c1``, ..., ``cal_cN``) with ``Gaussian(0, 0.1)`` priors.
    eline_prior_sigma: float
        Prior width on emission line amplitudes for marginalization.
        Default: 100.0.
    eline_mode: str
        Emission line fitting mode. One of:

        - ``"off"``: No emission line fitting (default).
        - ``"fixed"``: Lines from nebular model only.
        - ``"marginalized"``: Analytically marginalize line amplitudes
          (recommended for spectroscopic fitting).
        - ``"fitted"``: Line amplitudes as free MCMC parameters.

        Default: ``"off"``.
    eline_catalog: LineList or None
        Line catalog. ``None`` falls back to ``LineList.default_13()`` for
        Use ``LineList.default_optical()`` for
        FastSpecFit parity. Default: None.
    eline_prior_type: str
        Prior type for line amplitudes. One of ``"flat"`` (uninformative) or
        ``"cloudy"`` (CLOUDY-grid-interpolated). Default: ``"flat"``.
    eline_prior_width_dex: float
        Prior scatter in dex for the ``"cloudy"`` prior. Default: 0.3.
    eline_fix_doublets: bool
        Enforce atomic physics doublet ratios. Default: True.
    eline_broad: bool
        Enable broad component for AGN candidate lines. Default: False.
    eline_broad_fwhm_min_kms: float
        Minimum FWHM for the broad component [km/s]. Default: 500.0.
    covariance: jnp.ndarray or None
        Full spectral covariance matrix, shape ``(n_pix, n_pix)``.
        When provided, the likelihood uses ``diff @ C^{-1} @ diff``
        instead of per-pixel ``sum((diff/sigma)^2)``. The inverse is
        precomputed at construction time.  Default: None (diagonal noise).

    Returns
    -------
    Spectroscopy
        Spectroscopy instance with covariance matrix inverted and metadata set.

    Attributes
    ----------
    wave_obs: ndarray, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom].
    resolution: float, ndarray, or None
        Spectral resolution.
    sigma_lib_kms: float
        SSP library velocity dispersion [km/s].
    lsf_n_bins: int
        Number of LSF approximation bins.
    calibration_order: int
        Chebyshev polynomial order.
    eline_prior_sigma: float
        Emission line prior width.
    eline_mode: str
        Emission line fitting mode.
    eline_catalog: LineList or None
        Emission line catalog.
    eline_prior_type: str
        Prior type for line marginalization.
    eline_prior_width_dex: float
        Prior scatter [dex].
    eline_fix_doublets: bool
        Whether to enforce doublet ratios.
    eline_broad: bool
        Whether broad AGN component is enabled.
    eline_broad_fwhm_min_kms: float
        Minimum broad component FWHM [km/s].
    covariance: ndarray, shape (n_pix, n_pix) or None
        Spectral covariance matrix.
    covariance_inv: ndarray, shape (n_pix, n_pix) or None
        Inverse covariance matrix (precomputed).

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
    resample: str = "point"
    eline_prior_sigma: float = 100.0
    eline_mode: str = "off"
    eline_catalog: object | None = dataclasses.field(default=None, hash=False)
    eline_prior_type: str = "flat"
    eline_prior_width_dex: float = 0.3
    eline_fix_doublets: bool = True
    eline_broad: bool = False
    eline_broad_fwhm_min_kms: float = 500.0
    covariance: jnp.ndarray | None = dataclasses.field(default=None, hash=False)
    resolution_matrix: object | None = dataclasses.field(default=None, hash=False)

    def __post_init__(self) -> None:
        _valid_modes = ("off", "fixed", "marginalized", "fitted")
        if self.eline_mode not in _valid_modes:
            raise ValueError(f"eline_mode must be one of {_valid_modes}, got {self.eline_mode!r}")
        _valid_resample = ("point", "conserving", "auto")
        if self.resample not in _valid_resample:
            raise ValueError(f"resample must be one of {_valid_resample}, got {self.resample!r}")
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
        if self.resolution_matrix is not None:
            data = jnp.asarray(self.resolution_matrix.data)
            if data.shape[1] != len(self.wave_obs):
                raise ValueError(
                    f"resolution_matrix has {data.shape[1]} columns but wave_obs "
                    f"has length {len(self.wave_obs)}"
                )

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

    def resolve_conserving(self, wave_rest_model, z_ref: float = 0.0) -> bool:
        """Resolve the ``resample`` mode to a flux-conserving flag (#1166).

        ``"point"`` maps to ``False`` and ``"conserving"`` to ``True``. ``"auto"``
        returns ``True`` only when the observed pixels are coarse enough that point
        interpolation would skip model bins, the median rest-frame pixel spacing
        exceeds the median model-grid spacing over their overlap, evaluated at
        ``z_ref`` (pass the lowest redshift in the prior, the worst case for
        under-sampling). A pure-Python decision made once before tracing, so the
        forward branch stays static.

        Parameters
        ----------
        wave_rest_model: array_like, shape (n_wave,)
            Rest-frame model wavelength grid [Angstrom].
        z_ref: float, optional
            Redshift at which observed pixels are mapped to the rest frame.
            Default 0.0.

        Returns
        -------
        bool
            Whether to use the flux-conserving resample.
        """
        if self.resample != "auto":
            return self.resample == "conserving"
        import numpy as np

        wr = np.asarray(wave_rest_model)
        wq = np.asarray(self.wave_obs) / (1.0 + float(z_ref))
        lo = max(float(wq.min()), float(wr.min()))
        hi = min(float(wq.max()), float(wr.max()))
        qm = (wq >= lo) & (wq <= hi)
        mm = (wr >= lo) & (wr <= hi)
        if int(qm.sum()) < 2 or int(mm.sum()) < 2:
            return False
        return bool(np.median(np.diff(wq[qm])) > np.median(np.diff(wr[mm])))

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
    def has_resolution_matrix(self) -> bool:
        """Whether a banded instrument resolution matrix is configured (#1163).

        Returns
        -------
        bool
            True if a :class:`~tengri.observation.banded.BandedMatrix` is set,
            in which case it replaces the Gaussian ``apply_lsf`` in projection
            (the DESI/PFS spectro-perfectionism convention; Bolton & Schlegel
            2010).

        Notes
        -----
        Read-only property; determined at initialization.
        """
        return self.resolution_matrix is not None

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
        dict[str, Distribution]
            Mapping of calibration coefficient names (``cal_c1``, ..., ``cal_cN``)
            to ``Gaussian(0, 0.1)`` priors. Empty dict if ``calibration_order == 0``.

        Notes
        -----
        Called by ``Observation.get_all_params()`` to register observation-level
        parameters with the inference engine. Each coefficient has a weak
        Gaussian prior centered at 0 in log space (unity in linear space).

        """
        if self.calibration_order == 0:
            return {}
        return {f"cal_c{i + 1}": Gaussian(0.0, 0.1) for i in range(self.calibration_order)}

    def calibration_coeffs(self, params) -> jnp.ndarray | None:
        """Extract calibration coefficients from parameter dict.

        Returns
        -------
        ndarray, shape (calibration_order,), or None
            Chebyshev polynomial coefficients ``[c_1, c_2, ..., c_N]``,
            extracted from ``params["cal_c1"]``, ..., ``params["cal_cN"]``.
            Returns ``None`` when ``calibration_order == 0``.

        Notes
        -----
        Keys are constructed with the same f-string shape as
        :meth:`get_calibration_params` to prevent drift.

        """
        if self.calibration_order == 0:
            return None
        return jnp.asarray([params[f"cal_c{i + 1}"] for i in range(self.calibration_order)])

    @property
    def calibration_wave_range(self):
        """Wavelength range anchoring the calibration polynomial to [-1, 1].

        Returns
        -------
        tuple of scalar
            ``(wave_min, wave_max)`` [Angstrom] of the configured ``wave_obs``.

        Notes
        -----
        **JIT-compatible**: yes. The bounds are returned as JAX scalars, never
        Python floats: ``Spectroscopy`` is a pytree, so ``wave_obs`` is a tracer
        inside ``predict_observables_jit`` and ``float()`` on it would raise
        ``ConcretizationTypeError``. They are only ever consumed arithmetically
        (by :func:`~tengri.observation.calibration.calibration_polynomial`), so
        tracers are fine.

        The polynomial is normalized to the *configured instrument* grid, not to
        whatever grid a caller passes, so a given ``cal_cN`` keeps the same
        meaning when the model is evaluated on a custom ``wave_obs``.

        """
        return (self.wave_obs.min(), self.wave_obs.max())

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
        wave_obs: ndarray, shape (n_pix,)
            Observed-frame wavelength grid [Angstrom].
        resolution: float, ndarray, or None
            Spectral resolution ``R = lambda / delta_lambda``. Scalar for
            constant R, per-pixel array for wavelength-dependent, or None
            to skip LSF convolution.
        sigma_lib_kms: float, optional
            SSP library velocity dispersion to subtract in quadrature [km/s].
            Default: 70.0 (MILES standard).
        calibration_order: int, optional
            Order of multiplicative Chebyshev calibration polynomial.
            Default: 0 (no calibration).
        **kwargs
            Additional keyword arguments passed to ``Spectroscopy.__init__``.

        Returns
        -------
        Spectroscopy
            Configured spectroscopy object with all parameters set.

        Notes
        -----
        Internal helper used by instrument-specific factories (``nirspec_prism``,
        ``nirspec_g140m``, ``constant_r``). Not intended for direct public use.

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
        wave_obs: ndarray, shape (n_pix,)
            Observed-frame wavelength grid [Angstrom].
        **kwargs
            Additional keyword arguments passed to ``Spectroscopy``.

        Returns
        -------
        Spectroscopy
            Configured NIRSpec PRISM spectroscopy with wavelength-dependent
            resolution.

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
        wave_obs: ndarray, shape (n_pix,)
            Observed-frame wavelength grid [Angstrom].
        **kwargs
            Additional keyword arguments passed to ``Spectroscopy``.

        Returns
        -------
        Spectroscopy
            Configured NIRSpec G140M spectroscopy with approximately constant
            resolution R~1000.

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
        wave_obs: ndarray, shape (n_pix,)
            Observed-frame wavelength grid [Angstrom].
        R: float
            Spectral resolution ``R = lambda / delta_lambda`` (constant across
            all wavelengths, dimensionless).
        **kwargs
            Additional keyword arguments passed to ``Spectroscopy``.

        Returns
        -------
        Spectroscopy
            Configured spectrograph with constant wavelength-independent
            resolution.

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
        wave_obs: ndarray, shape (n_pix,)
            Observed-frame wavelength grid [Angstrom].
        resolution: float, optional
            Spectral resolution ``R = lambda / delta_lambda`` (dimensionless).
            Default: 2500 (DESI standard).
        **kwargs
            Additional keyword arguments passed to ``Spectroscopy``.

        Returns
        -------
        Spectroscopy
            Fully configured spectroscopy for DESI-like emission-line fitting
            with all standard settings pre-applied.

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

    # ── FITS I/O factories ────────────────────────────────────────

    @classmethod
    def from_jwst_x1d(
        cls,
        fits_path: str,
        *,
        ext: int | str = 1,
        resolution: float | jnp.ndarray | None = None,
        **kwargs,
    ) -> tuple[Spectroscopy, jnp.ndarray, jnp.ndarray]:
        """Load from a JWST x1d/x1dints FITS spectrum.

        Reads WAVELENGTH (µm), FLUX (µJy), and FLUX_ERROR (µJy) from the
        specified extension and converts to tengri internal units
        (Angstrom, erg/s/cm²/Hz).

        Parameters
        ----------
        fits_path: str
            Path to the x1d FITS file.
        ext: int or str
            FITS extension containing the spectrum table. Default: 1.
        resolution: float, ndarray, or None
            Spectral resolution override.  If None, auto-selects based on
            the ``FILTER`` or ``GRATING`` header keyword. Default: None.
        **kwargs
            Additional keyword arguments passed to ``Spectroscopy``.

        Returns
        -------
        tuple[Spectroscopy, ndarray, ndarray]
            ``(spec_config, flux_obs, flux_err)`` where flux values are in
            erg/s/cm²/Hz and wavelengths in Angstrom.

        Notes
        -----
        Requires ``astropy``.  NaN pixels are masked by setting their
        error to infinity.

        Unit conversions:

        - Wavelength: µm → Å (×10⁴)
        - Flux: µJy → erg/s/cm²/Hz (×10⁻²⁹)

        """
        from astropy.io import fits as pyfits

        with pyfits.open(fits_path) as hdul:
            data = hdul[ext].data
            header = hdul[ext].header if hasattr(hdul[ext], "header") else {}
            primary_header = hdul[0].header

            wave_um = jnp.asarray(data["WAVELENGTH"], dtype=jnp.float64)
            flux_ujy = jnp.asarray(data["FLUX"], dtype=jnp.float64)
            err_ujy = jnp.asarray(data["FLUX_ERROR"], dtype=jnp.float64)

        wave_aa = wave_um * 1e4
        flux_cgs = flux_ujy * 1e-29
        err_cgs = err_ujy * 1e-29

        nan_mask = ~(jnp.isfinite(flux_cgs) & jnp.isfinite(err_cgs) & (err_cgs > 0.0))
        err_cgs = jnp.where(nan_mask, jnp.inf, err_cgs)
        flux_cgs = jnp.where(nan_mask, 0.0, flux_cgs)

        if resolution is None:
            grating = str(header.get("GRATING", primary_header.get("GRATING", ""))).upper()
            if "PRISM" in grating:
                from tengri.observation.spectrum import nirspec_prism_resolution

                resolution = nirspec_prism_resolution(wave_um)
            elif "G140M" in grating:
                from tengri.observation.spectrum import nirspec_g140m_resolution

                resolution = nirspec_g140m_resolution(wave_um)
            else:
                resolution = None

        spec = cls._from_resolution(wave_aa, resolution, **kwargs)
        return spec, flux_cgs, err_cgs

    @classmethod
    def from_fits(
        cls,
        fits_path: str,
        *,
        wave_col: str = "WAVELENGTH",
        flux_col: str = "FLUX",
        err_col: str = "FLUX_ERROR",
        wave_unit_aa: float = 1.0,
        flux_unit_cgs: float = 1.0,
        ext: int | str = 1,
        resolution: float | jnp.ndarray | None = None,
        **kwargs,
    ) -> tuple[Spectroscopy, jnp.ndarray, jnp.ndarray]:
        """Load from a generic FITS binary table spectrum.

        Parameters
        ----------
        fits_path: str
            Path to the FITS file.
        wave_col: str
            Column name for wavelength. Default: ``"WAVELENGTH"``.
        flux_col: str
            Column name for flux. Default: ``"FLUX"``.
        err_col: str
            Column name for flux error. Default: ``"FLUX_ERROR"``.
        wave_unit_aa: float
            Multiplicative factor to convert wavelength column to Angstrom.
            E.g., 1e4 for µm input. Default: 1.0 (already Å).
        flux_unit_cgs: float
            Multiplicative factor to convert flux column to erg/s/cm²/Hz.
            E.g., 1e-29 for µJy. Default: 1.0 (already CGS).
        ext: int or str
            FITS extension. Default: 1.
        resolution: float, ndarray, or None
            Spectral resolution. Default: None.
        **kwargs
            Additional keyword arguments passed to ``Spectroscopy``.

        Returns
        -------
        tuple[Spectroscopy, ndarray, ndarray]
            ``(spec_config, flux_obs, flux_err)`` in Angstrom and CGS.

        Notes
        -----
        Requires ``astropy``.  Generic reader for any FITS binary table
        spectrum.  For JWST x1d files, prefer ``from_jwst_x1d`` which
        handles unit conversion and resolution auto-detection.

        """
        from astropy.io import fits as pyfits

        with pyfits.open(fits_path) as hdul:
            data = hdul[ext].data
            wave_raw = jnp.asarray(data[wave_col], dtype=jnp.float64)
            flux_raw = jnp.asarray(data[flux_col], dtype=jnp.float64)
            err_raw = jnp.asarray(data[err_col], dtype=jnp.float64)

        wave_aa = wave_raw * wave_unit_aa
        flux_cgs = flux_raw * flux_unit_cgs
        err_cgs = err_raw * flux_unit_cgs

        nan_mask = ~(jnp.isfinite(flux_cgs) & jnp.isfinite(err_cgs) & (err_cgs > 0.0))
        err_cgs = jnp.where(nan_mask, jnp.inf, err_cgs)
        flux_cgs = jnp.where(nan_mask, 0.0, flux_cgs)

        spec = cls._from_resolution(wave_aa, resolution, **kwargs)
        return spec, flux_cgs, err_cgs

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


# ── Wavelength grid builder ──────────────────────────────────────


def build_wavelength_grid(
    resolution_fn: callable,
    wave_min_aa: float,
    wave_max_aa: float,
    n_pix_per_resel: float = 2.5,
) -> jnp.ndarray:
    """Build an optimal wavelength grid from an instrument resolution profile.

    Steps through wavelength space with pixel spacing matched to the local
    resolution element: Δλ = λ / (n_pix_per_resel · R(λ)).

    Parameters
    ----------
    resolution_fn: callable
        Function R(wave_um) → spectral resolution, where wave_um is in
        micrometers.  Must accept and return arrays.  Use e.g.
        ``nirspec_prism_resolution`` or ``nirspec_g140m_resolution``.
    wave_min_aa: float
        Minimum wavelength [Å].
    wave_max_aa: float
        Maximum wavelength [Å].
    n_pix_per_resel: float
        Number of pixels per resolution element (Nyquist = 2.0).
        Default: 2.5 (slight oversampling for interpolation safety).

    Returns
    -------
    ndarray, shape (n_pix,)
        Wavelength grid in Angstrom, non-uniformly spaced to match the
        instrument resolution.

    Notes
    -----
    Not JIT-compatible (uses a Python while-loop to build the grid).

    The grid construction follows the CIGALE ``new_wavegrid`` approach
    (Jakobsen+2022 convention of ~2.2 pixels per resolution element for
    NIRSpec PRISM). Here we allow the user to set this via ``n_pix_per_resel``.

    Examples
    --------
    >>> from tengri.observation.spectrum import nirspec_prism_resolution
    >>> grid = build_wavelength_grid(nirspec_prism_resolution, 6000.0, 53000.0)
    >>> grid.shape[0]  # ~400-600 pixels depending on sampling

    """
    import numpy as np

    wave_list = [wave_min_aa]
    while wave_list[-1] < wave_max_aa:
        lam = wave_list[-1]
        lam_um = lam / 1e4
        r_local = float(resolution_fn(jnp.array([lam_um]))[0])
        r_local = max(r_local, 1.0)
        step = lam / (n_pix_per_resel * r_local)
        wave_list.append(lam + step)

    grid = np.array(wave_list, dtype=np.float64)
    grid = grid[grid <= wave_max_aa]
    return jnp.asarray(grid)


# ── Wavelength masking ────────────────────────────────────────────


def apply_wavelength_mask(
    noise: jnp.ndarray,
    wave_obs: jnp.ndarray,
    mask_ranges: list[tuple[float, float]],
) -> jnp.ndarray:
    """Mask spectral regions by setting noise to infinity.

    Masked pixels are effectively removed from the likelihood (chi2
    contribution → 0). Returns a new array, does not mutate the input.

    Parameters
    ----------
    noise: ndarray, shape (n_pix,)
        Per-pixel 1-sigma noise [flux units].
    wave_obs: ndarray, shape (n_pix,)
        Observed-frame wavelength grid [Angstrom].
    mask_ranges: list[tuple[float, float]]
        Wavelength ranges to mask [Angstrom]. Each ``(lo, hi)`` pair
        defines a region where ``lo <= wave <= hi`` is masked.

    Returns
    -------
    ndarray, shape (n_pix,)
        Copy of ``noise`` with masked pixels set to ``inf``. Pixels in
        masked wavelength ranges have noise = infinity.

    Notes
    -----
    Immutable operation: returns a new array without modifying the input.
    Masked pixels contribute zero to the likelihood due to infinite noise.
    Useful for excluding sky emission lines, detector artifacts, or other
    unreliable spectral regions from inference.

    Examples
    --------
    Mask the 5577 Å sky line and a detector gap::

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


# ── Deprecated alias, removed in tengri v1.0 ─────────────────────
