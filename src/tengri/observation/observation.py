# SPDX-License-Identifier: BSD-3-Clause
"""Unified observation configuration for tengri SED fitting.

Bundles photometric and/or spectroscopic setup with noise configuration
into a single declarative object. Follows the same Instrument pattern as
Synthesizer, in tengri's JAX/differentiable inference context.

The Observation class is a frozen configuration container, it never
enters JAX-traced code. It configures what the Model precomputes and
what the Fitter dispatches.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from tengri.observation.line_flux_data import LineFluxData
from tengri.observation.line_ratio_data import LineRatioData
from tengri.observation.noise_model import NoiseModel
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectral_indices import SpectralIndexData
from tengri.observation.spectroscopy import Spectroscopy
from tengri.parameters.priors import Distribution
from tengri.parameters.resolve import require_redshift
from tengri.utils.scale import apply_log10_scale, log10_flux_scale

_OBSERVATION_DEPRECATION_WARNED = False


def _restband_lnu(state) -> jnp.ndarray:
    r"""Total rest-frame band luminosity for ``phot_rest_fnu`` (#1148).

    ``phot_rest_fnu`` (and so ``Observables.mag_absolute``) is the SED reprojected
    at :math:`z=0`, :math:`d_L=10\,{\rm pc}`: *the galaxy as it is*. The filter
    therefore sits in the **rest** frame and samples the rest SED at its own pivot.

    The LUT used to reuse ``total_lnu``, the **observed**-band sum, which samples
    rest :math:`\lambda_{\rm eff}/(1+z)`. Those are different physical quantities:
    against the exact path the LUT ran 769 % out in ``des_g`` at z=0.5 and orders of
    magnitude out in the blue, so an object's *absolute* magnitude depended on its
    redshift and on which ``approx`` was passed.

    Assembled from the ``*_restband_lnu_precomp`` family, **auto-discovered** exactly
    as ``predict_via_precomp`` discovers the observed ``*_phot_lnu_precomp`` family,
    so a new emitter gets rest-frame photometry for free and cannot silently drop out
    of it, which is how the fast path lost AGN and nebular emission once already
    (#737/#740).

    The galaxy's own dust stays (it is part of the SED); the IGM does not (it is a
    line-of-sight absorber between us and the source, #1115).

    Parameters
    ----------
    state : PipelineState
        Carrying the ``*_restband_lnu_precomp`` family and the rest-band dust screens.

    Returns
    -------
    ndarray, shape (n_filters,)
        Rest-frame band luminosity [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes, pure JAX.

    Falls back to the observed-band sum when no emitter published a rest-band tensor
    (a model built without ``WavePrecomp``, whose ``phot_rest_fnu`` the exact
    projector computes instead).
    """
    keys = [k for k in state.derived.field_names() if k.endswith("_restband_lnu_precomp")]
    contribs = [state.derived.get(k) for k in keys]
    contribs = [c for c in contribs if c is not None]
    if not contribs:
        # No rest-band LUT (e.g. n_subbands / WavePrecomp off). The caller's
        # observed-band sum is the pre-#1148 behavior; keep it rather than return
        # zeros, so this can never silently blank out a channel.
        return state.derived.get("stellar_phot_lnu_precomp")

    total = contribs[0]
    for c in contribs[1:]:
        total = total + c

    # Dust reddens the stellar + nebular + shock bucket, exactly as in the
    # observed band. Everything else (AGN, radio, X-ray) carries its own
    # attenuation already. Shock rides with nebular rather than in the
    # unattenuated remainder because the exact path sums sed_shock into
    # sed_intrinsic *before* dust, so it is reddened there too (#1375).
    stellar = state.derived.get("stellar_restband_lnu_precomp")
    nebular = state.derived.get("nebular_restband_lnu_precomp")
    shock = state.derived.get("shock_restband_lnu_precomp")
    stellar = stellar if stellar is not None else jnp.zeros_like(total)
    nebular = nebular if nebular is not None else jnp.zeros_like(total)
    # Shock kept separately as well as summed in: the sum defines the screened bucket
    # that ``unattenuated`` is the complement of, but only nebular has an exact
    # band-integrated screen (#1738), so the two halves are reddened differently below.
    shock_only = shock if shock is not None else jnp.zeros_like(total)
    nebular = nebular + shock_only
    unattenuated = total - stellar - nebular

    a_bc = state.derived.get("dust_bc_restband_attenuation_precomp")
    a_diff = state.derived.get("dust_diff_restband_attenuation_precomp")
    a_single = state.derived.get("dust_restband_attenuation_precomp")
    sub_per_age = state.derived.get("stellar_restband_lnu_per_age_subband_precomp")
    y_age = state.derived.get("dust_young_indicator")

    if a_bc is not None and a_diff is not None:
        # Two-component (Charlot & Fall): T(a, λ) = T_diff(λ)·T_bc(λ)^y(a).
        a_bc_sub = state.derived.get("dust_bc_restband_attenuation_subband_precomp")
        a_diff_sub = state.derived.get("dust_diff_restband_attenuation_subband_precomp")
        if a_bc_sub is not None and sub_per_age is not None and y_age is not None:
            # K-point quadrature across the rest band, the screen is EVALUATED at
            # each node, not extrapolated from the pivot (#1122).
            t_sub = a_diff_sub * a_bc_sub ** y_age[:, None, None]
            stellar_att = jnp.sum(sub_per_age * t_sub, axis=(0, 2))
        else:
            stellar_att = a_diff * a_bc * stellar
        # Nebular arises in the HII regions around the youngest stars, so it sees the
        # full young-limit screen (y=1), matching the exact path. Where the dust
        # component published the reddened continuum integrated through the rest band,
        # prefer it: sampling the screen at the pivot is only correct where the screen
        # is flat across the filter, and nebular emission is line-dominated (#1738).
        nebular_exact = state.derived.get("nebular_restband_lnu_attenuated_precomp")
        if nebular_exact is not None:
            return stellar_att + nebular_exact + a_diff * a_bc * shock_only + unattenuated
        return stellar_att + a_diff * a_bc * nebular + unattenuated

    if a_single is not None:
        a_sub = state.derived.get("dust_restband_attenuation_subband_precomp")
        if a_sub is not None and sub_per_age is not None:
            stellar_att = jnp.sum(sub_per_age * a_sub, axis=(0, 2))
        else:
            stellar_att = a_single * stellar
        return stellar_att + a_single * nebular + unattenuated

    return total


@dataclasses.dataclass(frozen=True)
class Observation:
    """Unified observation configuration.

    Composes optional photometric, spectroscopic, and noise model
    configurations. At least one of ``photometry`` or ``spectroscopy``
    must be provided.

    Parameters
    ----------
    photometry : Photometry or None
        Photometric filter configuration.
    spectroscopy : Spectroscopy or None
        Spectroscopic instrument configuration.
    noise : NoiseModel or None
        Noise model configuration (calibration floor, Student-t dof).
    line_fluxes : LineFluxData or None
        Observed emission line fluxes for direct fitting.
        When provided, the likelihood includes an additive chi-squared
        term comparing model line luminosities against these fluxes.

    Returns
    -------
    Observation
        Validated observation container with at least one data modality.

    Attributes
    ----------
    photometry : Photometry or None
        Photometric filter configuration.
    spectroscopy : Spectroscopy or None
        Spectroscopic instrument configuration.
    noise : NoiseModel or None
        Noise model configuration.
    line_fluxes : LineFluxData or None
        Observed emission line fluxes.
    spectral_indices : SpectralIndexData or None
        Observed spectral indices for fitting.

    Notes
    -----
    A frozen, immutable dataclass that serves as a declarative container
    for all observation metadata. Never enters JAX-traced code; used solely
    for configuration dispatch to precomputation and inference steps.
    Inspired by Synthesizer's Instrument pattern, adapted for tengri's
    differentiable context.

    Examples
    --------
    Photometry-only::

        obs = Observation(
            photometry=Photometry.from_names(["sdss_r", "sdss_i"]),
        )

    Joint photometry + spectroscopy::

        obs = Observation(
            photometry=Photometry.from_names(["jwst_f200w", "jwst_f356w"]),
            spectroscopy=Spectroscopy.nirspec_prism(wave_obs),
            noise=NoiseModel(calibration_floor=Uniform(0.01, 0.15)),
        )

    """

    photometry: Photometry | None = None
    spectroscopy: Spectroscopy | None = None
    noise: NoiseModel | None = None
    line_fluxes: LineFluxData | None = None
    spectral_indices: SpectralIndexData | None = None
    line_ratios: LineRatioData | None = None
    lines: object | None = None

    def __post_init__(self):
        # Emit one-shot deprecation warning for value-carrying fields
        global _OBSERVATION_DEPRECATION_WARNED
        if not _OBSERVATION_DEPRECATION_WARNED and (
            self.line_fluxes is not None
            or self.spectral_indices is not None
            or self.line_ratios is not None
        ):
            import warnings

            warnings.warn(
                "Observation(line_fluxes=...) / spectral_indices=... / "
                "line_ratios=... carries measured values on the instrument schema "
                "and is deprecated: declare WHICH lines with "
                "lines=LineList.from_names([...]) and supply the VALUES per galaxy "
                "via Data(lines=...). See #1321.",
                DeprecationWarning,
                stacklevel=2,
            )
            _OBSERVATION_DEPRECATION_WARNED = True

        if (
            self.photometry is None
            and self.spectroscopy is None
            and self.line_fluxes is None
            and self.spectral_indices is None
            and self.line_ratios is None
        ):
            raise ValueError(
                "Observation requires at least one of photometry, spectroscopy, "
                "line_fluxes, line_ratios, or spectral_indices."
            )

    @property
    def can_do_photometry(self) -> bool:
        """Whether photometric filters are configured.

        Returns
        -------
        bool
            True if photometry is configured.

        Notes
        -----
        Query method for capability checking. Safe to call even if
        photometry was not provided to this Observation.

        """
        return self.photometry is not None

    @property
    def can_do_spectroscopy(self) -> bool:
        """Whether a spectroscopic wavelength grid is configured.

        Returns
        -------
        bool
            True if spectroscopy is configured.

        Notes
        -----
        Query method for capability checking. Safe to call even if
        spectroscopy was not provided to this Observation.

        """
        return self.spectroscopy is not None

    @property
    def has_line_fluxes(self) -> bool:
        """Whether observed emission line fluxes are configured.

        Returns
        -------
        bool
            True if line flux data is configured.

        Notes
        -----
        Query method for capability checking. Safe to call even if
        line fluxes were not provided to this Observation.

        """
        return self.line_fluxes is not None

    @property
    def has_spectral_indices(self) -> bool:
        """Whether observed spectral indices are configured.

        Returns
        -------
        bool
            True if spectral index data is configured.

        Notes
        -----
        Query method for capability checking. Safe to call even if
        spectral indices were not provided to this Observation.

        """
        return self.spectral_indices is not None

    @property
    def has_line_ratios(self) -> bool:
        """Whether observed emission line ratios are configured.

        Returns
        -------
        bool
            True if line ratio data is configured.
        """
        return self.line_ratios is not None

    @property
    def is_joint(self) -> bool:
        """Whether both photometry and spectroscopy are configured.

        Returns
        -------
        bool
            True if both photometry and spectroscopy are present.

        Notes
        -----
        Convenience predicate for detecting joint photometry+spectroscopy
        fitting (vs. photometry-only or spectroscopy-only).

        """
        return self.can_do_photometry and self.can_do_spectroscopy

    @property
    def data_type(self) -> str:
        """Inferred data type string (photometry/spectroscopy/joint).

        Returns
        -------
        str
            One of ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.

        Notes
        -----
        Returns a string representation of the configured data types,
        useful for logging and dispatch logic.

        """
        if self.is_joint:
            return "joint"
        elif self.can_do_photometry:
            return "photometry"
        else:
            return "spectroscopy"

    # ── Data dimensions ───────────────────────────────────────────

    @property
    def n_data_phot(self) -> int:
        """Number of photometric data points (filters).

        Returns
        -------
        int
            Number of filters, or 0 if no photometry configured.

        Notes
        -----
        Returns 0 safely if photometry is not configured; may be used in
        conditional logic without prior capability checks.

        """
        return self.photometry.n_filters if self.photometry else 0

    @property
    def n_data_spec(self) -> int:
        """Number of spectroscopic data points (pixels).

        Returns
        -------
        int
            Number of spectral pixels, or 0 if no spectroscopy configured.

        Notes
        -----
        Returns 0 safely if spectroscopy is not configured; may be used in
        conditional logic without prior capability checks.

        """
        return self.spectroscopy.n_pixels if self.spectroscopy else 0

    @property
    def n_data_lines(self) -> int:
        """Number of emission line flux data points.

        Returns
        -------
        int
            Number of emission lines, or 0 if no line flux data configured.

        Notes
        -----
        Returns 0 safely if line flux data is not configured; may be used
        in conditional logic without prior capability checks.

        """
        return self.line_fluxes.n_lines if self.line_fluxes else 0

    @property
    def n_data_indices(self) -> int:
        """Number of spectral index data points.

        Returns
        -------
        int
            Number of spectral indices, or 0 if no index data configured.

        Notes
        -----
        Returns 0 safely if spectral index data is not configured; may be
        used in conditional logic without prior capability checks.

        """
        return self.spectral_indices.n_indices if self.spectral_indices else 0

    @property
    def n_data_ratios(self) -> int:
        """Number of emission line ratio data points.

        Returns 0 safely if no line ratio data is configured.
        """
        return self.line_ratios.n_ratios if self.line_ratios else 0

    @property
    def n_data(self) -> int:
        """Total number of data points.

        Returns
        -------
        int
            Sum of all photometric, spectroscopic, line flux, line ratio,
            and spectral index data points.

        Notes
        -----
        Aggregates counts across all observation modalities. Useful for
        data dimensionality checks and prior/posterior shape validation.

        """
        return (
            self.n_data_phot
            + self.n_data_spec
            + self.n_data_lines
            + self.n_data_indices
            + self.n_data_ratios
        )

    # ── Data packing / unpacking ──────────────────────────────────

    def pack_data(
        self,
        phot: jnp.ndarray | None = None,
        spec: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Concatenate photometry and spectroscopy data in canonical order.

        Validates array shapes against the observation configuration.
        Canonical order: ``[photometry, spectroscopy]``.

        Parameters
        ----------
        phot : array or None
            Photometric data, shape ``(n_filters,)``.
        spec : array or None
            Spectroscopic data, shape ``(n_pixels,)``.

        Returns
        -------
        jnp.ndarray
            Packed data array, shape ``(n_data,)``.

        Raises
        ------
        ValueError
            If array shapes don't match the observation configuration.

        Notes
        -----
        Both arrays are optional but at least one must be provided and
        configured in the Observation. Useful for likelihood evaluation
        and parameter inference pipelines.

        """
        arrays = []

        if self.can_do_photometry:
            if phot is None:
                raise ValueError("Observation has photometry but phot= was not provided.")
            phot = jnp.asarray(phot)
            if phot.shape != (self.n_data_phot,):
                raise ValueError(
                    f"phot shape {phot.shape} doesn't match expected ({self.n_data_phot},)"
                )
            arrays.append(phot)

        if self.can_do_spectroscopy:
            if spec is None:
                raise ValueError("Observation has spectroscopy but spec= was not provided.")
            spec = jnp.asarray(spec)
            if spec.shape != (self.n_data_spec,):
                raise ValueError(
                    f"spec shape {spec.shape} doesn't match expected ({self.n_data_spec},)"
                )
            arrays.append(spec)

        if len(arrays) == 1:
            return arrays[0]
        return jnp.concatenate(arrays)

    def unpack_prediction(
        self,
        predicted: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Split a concatenated prediction into photometry and spectroscopy.

        Inverse of ``pack_data``: reverses the concatenation to extract
        predictions for each observation modality.

        Parameters
        ----------
        predicted : array
            Packed prediction array, shape ``(n_data,)``.

        Returns
        -------
        dict
            Keys are ``"photometry"`` and/or ``"spectroscopy"``, values
            are the corresponding sub-arrays.

        Notes
        -----
        Only keys corresponding to configured observation modalities
        will be present in the returned dictionary.

        """
        result = {}
        idx = 0

        if self.can_do_photometry:
            result["photometry"] = predicted[idx : idx + self.n_data_phot]
            idx += self.n_data_phot

        if self.can_do_spectroscopy:
            result["spectroscopy"] = predicted[idx : idx + self.n_data_spec]

        return result

    # ── Parameter generation ──────────────────────────────────────

    def get_all_params(self) -> dict[str, Distribution]:
        """Collect all observation-driven parameters.

        Merges calibration polynomial params from spectroscopy config
        and noise model params from noise config.

        Returns
        -------
        dict
            Parameter name → Distribution mapping. Empty if no observation
            params are needed (e.g. photometry-only with no noise config).

        Notes
        -----
        This method is called by the inference engine to set up the prior
        structure. Observation parameters include calibration coefficients
        and noise model hyperparameters, but not SED or SFH params.

        """
        params: dict[str, Distribution] = {}

        if self.spectroscopy is not None:
            params.update(self.spectroscopy.get_calibration_params())

        if self.noise is not None:
            params.update(self.noise.get_params())

        return params

    # ── Observation projection (rest SED → observed fluxes) ───────

    def observe_photometry(self, sed_result, z: float, dl_cm: float) -> jnp.ndarray:
        """Project an observed-frame SED through photometric filters.

        Parameters
        ----------
        sed_result : SEDResult
            Observed-frame SED with ``wavelength`` and ``sed``.
        z : float
            Redshift.
        dl_cm : float
            Luminosity distance [cm].

        Returns
        -------
        jnp.ndarray, shape (n_filters,)
            Photometric fluxes in each filter [erg/s/Hz].

        Notes
        -----
        Requires photometry to be configured. Computes rest-frame wavelengths
        from the observed-frame input and integrates the SED against each
        filter transmission curve.

        """
        if self.photometry is None:
            raise ValueError("No photometry configured in this Observation.")

        from tengri.observation.photometry import compute_flux_density

        wave_rest = sed_result.wavelength / (1.0 + z)
        fluxes = []
        for fw, ft in zip(self.photometry.filter_waves, self.photometry.filter_trans):
            f = compute_flux_density(sed_result.sed, wave_rest, fw, ft, z, dl_cm)
            fluxes.append(f)
        return jnp.array(fluxes)

    def observe_spectrum(
        self,
        sed_result,
        z: float,
        dl_cm: float,
        sigma_v_kms: float = 0.0,
        cal_coeffs: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Project an observed-frame SED onto spectroscopic pixel grid.

        Parameters
        ----------
        sed_result : SEDResult
            Observed-frame SED with ``wavelength`` and ``sed``.
        z : float
            Redshift.
        dl_cm : float
            Luminosity distance [cm].
        sigma_v_kms : float, optional
            Intrinsic velocity dispersion [km/s]. Default 0.0.
        cal_coeffs : ndarray, shape (order,), or None, optional
            Calibration polynomial coefficients to apply after LSF.
            If ``None`` (default), no calibration is applied.

        Returns
        -------
        jnp.ndarray, shape (n_pixels,)
            Spectroscopic flux at each pixel [erg/s/cm^2/Hz].

        Notes
        -----
        Requires spectroscopy to be configured. Applies LSF convolution
        if a resolution profile is specified. Applies flux-calibration
        polynomial if ``cal_coeffs`` is provided. Returns data ready for
        likelihood evaluation against observed spectra.

        """
        if self.spectroscopy is None:
            raise ValueError("No spectroscopy configured in this Observation.")

        from tengri.observation.spectrum import project_spectrum

        wave_rest = sed_result.wavelength / (1.0 + z)
        wave_obs = self.spectroscopy.wave_obs
        conserving = self.spectroscopy.resolve_conserving(sed_result.wavelength)
        flux = project_spectrum(
            sed_result.sed,
            wave_rest,
            wave_obs,
            z,
            dl_cm,
            resolution=self.spectroscopy.resolution,
            sigma_lib_kms=self.spectroscopy.sigma_lib_kms,
            sigma_v_kms=sigma_v_kms,
            cal_coeffs=cal_coeffs,
            cal_wave_range=self.spectroscopy.calibration_wave_range,
            conserving=conserving,
            resolution_matrix=self.spectroscopy.resolution_matrix,
        )
        return flux

    # ── Unified projection (ObservationModel Protocol) ────────────

    def predict(
        self,
        state,
        params,
        *,
        dl_cm=None,
        wave_obs=None,
        sigma_v_kms: float = 0.0,
        lsf_resolution=None,
        lsf_sigma_lib_kms: float | None = None,
        lsf_n_bins: int | None = None,
        observables_type=None,
    ) -> dict[str, jnp.ndarray]:
        """Project an orchestrator :class:`ForwardState` into observable channels.

        Unified projection seam matching the :class:`ObservationModel`
        Protocol in :mod:`tengri.protocols.observation`. Part of the
        forward-projection unification
        (``docs/dev/archive/photometry_path_unification.md``).

        Parameters
        ----------
        state : ForwardState
            Orchestrator output. Reads ``state.sed_intrinsic`` (rest-frame
            L_nu in erg/s/Hz) and ``state.wave`` (rest-frame Angstrom).
        params : Mapping[str, jnp.ndarray]
            Parameter dict. Reads ``redshift`` for the cosmology calculation.
        dl_cm : float or jnp.ndarray, optional
            Luminosity distance [cm]. If ``None``, derived from
            ``params["redshift"]`` via :func:`tengri.utils.cosmology.luminosity_distance`.
        wave_obs : jnp.ndarray, optional
            Observed-frame wavelength grid for the spectrum. Defaults to
            ``self.spectroscopy.wave_obs``.
        sigma_v_kms : float, default 0.0
            Velocity dispersion [km/s] for LSF convolution.
        lsf_resolution : float, ndarray, or None
            Override LSF resolution. ``None`` reuses
            ``self.spectroscopy.resolution``.
        lsf_sigma_lib_kms : float, optional
            Override SSP library sigma [km/s]. ``None`` reuses
            ``self.spectroscopy.sigma_lib_kms``.
        lsf_n_bins : int, optional
            Override piecewise-constant LSF bin count. ``None`` reuses
            ``self.spectroscopy.lsf_n_bins``.
        observables_type : type or None
            If provided, a :class:`typing.NamedTuple` class produced by
            :func:`build_observables_class`. When ``None``, returns a dict
            (backward-compat). When provided, populates and returns an instance
            of this class.

        Returns
        -------
        dict[str, jnp.ndarray] or Observables
            When ``observables_type`` is ``None``: Observable channels as dict,
            keyed by which sub-blocks are configured:

            - ``"phot_fnu"`` if :attr:`can_do_photometry`, shape ``(n_filters,)``,
              F_nu [erg/s/cm^2/Hz].
            - ``"spec_fnu"`` if :attr:`can_do_spectroscopy`, shape ``(n_pix,)``,
              F_nu [erg/s/cm^2/Hz].

            When ``observables_type`` is provided: an instance of the passed
            NamedTuple class with fields populated in order: ``phot_fnu``,
            ``phot_rest_fnu``, ``spec_fnu``, ``lines_flux``, ``indices``.

        Notes
        -----
        **JIT-compatible**: yes, same kernels as
        :meth:`observe_photometry` and :meth:`observe_spectrum`.

        Joint observations (``is_joint``) return both keys from a single
        forward pass. ``loss_functions._build_prediction`` still makes
        two separate calls for joint fits; collapsing that branch onto
        this single call is an open consolidation.

        When ``observables_type`` is provided and the observation has
        ``line_fluxes`` or ``spectral_indices`` configured, raises
        ``NotImplementedError``, those channels are not routed through
        this entry point yet.
        """
        from tengri.cosmology import luminosity_distance
        from tengri.observation.photometry import compute_flux_density_batch, project_photometry
        from tengri.observation.spectrum import project_spectrum

        z = jnp.asarray(require_redshift(params, "observation.observation.predict"))
        if dl_cm is None:
            dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        else:
            dl_cm = jnp.asarray(dl_cm)

        sed_rest = state.sed_intrinsic
        wave_rest = state.wave

        # IGM attenuation is an observed-frame transmission the IGM component
        # publishes on the rest grid (``T`` evaluated at ``wave_obs =
        # wave*(1+z)``). Every projection below redshifts the rest SED
        # internally, so attenuating here (once, before projection) is what
        # captures the sharp Lyman break across broad bands and spectral pixels
        # at high redshift (#932); a single per-band effective-wavelength factor
        # would not. ``T`` shares the rest grid with ``sed_rest``, and the key is
        # absent (structural no-op) when IGM is disabled, so low-z / IGM-off
        # models are bit-unchanged.
        #
        # ``sed_atten`` feeds the spectroscopy block ONLY; that is an observed-frame
        # channel, where the absorber belongs.
        #
        # The observed-photometry block does NOT use it: ``project_photometry`` reads
        # ``state.sed_intrinsic`` and applies the same transmission itself, so that
        # arbitrary post-build filters (``Prediction.photometry(filters=...)``) go
        # through the identical kernel instead of a copy that could silently omit the
        # IGM factor. Handing it ``sed_atten`` would square the transmission.
        #
        # The rest-frame-photometry block does NOT use it either (#1115): the IGM is a
        # line-of-sight absorber, not part of the galaxy's rest-frame SED. See there.
        igm_trans = (
            state.derived.get("igm_transmission", None) if state.derived is not None else None
        )
        sed_atten = sed_rest if igm_trans is None else sed_rest * igm_trans

        out: dict[str, jnp.ndarray] = {}

        if self.can_do_photometry:
            # ``predict`` is the canonical exact (compositional) path: it
            # integrates ``state.sed_intrinsic`` through each filter without
            # approximation. The precompute LUT path is opt-in via
            # :meth:`predict_via_precomp` (or its callers); this method does
            # NOT fall through to the LUT by default, exact-first is the
            # default semantics for ``observation.predict``.
            out["phot_fnu"] = project_photometry(state, params, self.photometry, dl_cm=dl_cm)

        if self.can_do_spectroscopy:
            sed_spec = sed_atten

            wo = wave_obs if wave_obs is not None else self.spectroscopy.wave_obs
            resolution = (
                lsf_resolution if lsf_resolution is not None else self.spectroscopy.resolution
            )
            sigma_lib = (
                lsf_sigma_lib_kms
                if lsf_sigma_lib_kms is not None
                else self.spectroscopy.sigma_lib_kms
            )
            n_bins = lsf_n_bins if lsf_n_bins is not None else self.spectroscopy.lsf_n_bins
            conserving = self.spectroscopy.resolve_conserving(state.wave)
            flux = project_spectrum(
                sed_spec,
                wave_rest,
                wo,
                z,
                dl_cm,
                resolution=resolution,
                sigma_lib_kms=sigma_lib,
                n_bins=n_bins,
                sigma_v_kms=sigma_v_kms,
                cal_coeffs=self.spectroscopy.calibration_coeffs(params),
                cal_wave_range=self.spectroscopy.calibration_wave_range,
                conserving=conserving,
                resolution_matrix=self.spectroscopy.resolution_matrix,
            )
            out["spec_fnu"] = flux

        # If observables_type is provided, populate and return the NamedTuple.
        # Line fluxes / line ratios / spectral indices are NOT projection
        # observables; they are scalar measurables computed separately
        # (predict_line_fluxes / predict_line_ratios / predict_spectral_indices)
        # and composed into the likelihood via the prediction dict, so their
        # presence on the Observation no longer blocks the projection here.
        if observables_type is not None:
            # Compute phot_rest_fnu: rest-frame photometry at z=0, d_L=10pc
            phot_rest = None
            if self.can_do_photometry:
                from tengri.utils.physics_constants import TEN_PC_CM

                dl_rest = TEN_PC_CM  # 10 pc in cm
                n_real = self.photometry.n_filters
                # ``sed_rest``, NOT ``sed_atten`` (#1115). ``phot_rest_fnu`` is the
                # SED reprojected at z=0, d_L=10 pc, the galaxy as it is. The IGM is
                # a line-of-sight absorber *between us and the source*; it is not part
                # of the galaxy's rest-frame SED. Feeding the attenuated SED here made
                # an object's absolute magnitude depend on how far away it happens to
                # be. (The galaxy's own LyC absorption (``neb_fesc``, dust) already
                # lives in ``sed_rest`` and correctly stays.)
                #
                # This projects at z=0, so the filter's OWN wavelengths are read as
                # REST wavelengths, and T is stored on the rest grid as
                # T(λ_rest·(1+z)). The corruption was therefore confined to filters
                # with support at rest λ < 1216 Å, blueward of Lyα, where Madau/Inoue
                # absorption begins, and the (1+z) cancels, so that boundary is
                # redshift-invariant while its depth is not. Measured on a rest-900 Å
                # band: −5.2 % at z=1, −30.0 % at z=3, −95.9 % (≈ −3.5 mag) at z=6.
                #
                # Zero-diff for every shipped filter (the bluest, GALEX FUV, starts at
                # 1341 Å and has no throughput below Lyα), which is exactly why it
                # lands now, before a Lyman-continuum band added for escape-fraction
                # work quietly returns an answer that scales with source redshift.
                # ``predict_via_precomp`` already did this; the two paths now agree.
                phot_rest = compute_flux_density_batch(
                    sed_rest,
                    wave_rest,
                    self.photometry._fw_padded,
                    self.photometry._ft_padded,
                    jnp.asarray(0.0),
                    jnp.asarray(dl_rest),
                    convention=self.photometry.convention,
                )[:n_real]

            # Build positional arguments in order: phot_fnu, phot_rest_fnu, spec_fnu
            args = []
            if self.can_do_photometry:
                args.append(out["phot_fnu"])
                args.append(phot_rest)
            if self.can_do_spectroscopy:
                args.append(out["spec_fnu"])

            return observables_type(*args)

        return out

    def predict_via_precomp(
        self,
        state,
        params,
        *,
        observables_type=None,
    ):
        r"""Project observables via the photometric LUT instead of integrating ``sed_intrinsic``.

        Opt-in fast path for ``approx=WavePrecomp()``. Sums all
        ``*_phot_lnu_lut`` keys present in ``state.derived`` (the
        rest-frame Lν contributions from each component that publishes
        one) and applies the cosmology factor ``(1+z)/(4π·dl²)`` to
        convert to observed F_ν.

        Components that publish a LUT entry:

        - ``stellar_phot_lnu_precomp``: :class:`StellarSEDComponent`.
          For ``BakedIn`` nebular backends this already contains the nebular
          contribution, since the SSP grid carries baked-in nebular emission.
        - ``nebular_phot_lnu_precomp``: :class:`NebularSEDComponent`
          (when the backend supports filter-level precomputation;
          non-BakedIn backends only).

        Any additional ``*_phot_lnu_precomp`` entries (AGN, …) sum in
        automatically.

        Photometry only, spectroscopy has its own LUT path
        (:meth:`predict_spectrum_via_precomp`).

        Parameters
        ----------
        state : ForwardState
            Orchestrator state with at least ``stellar_phot_lnu_precomp``.
        params : Mapping[str, jnp.ndarray]
            Param dict; reads ``redshift``.
        observables_type : type, optional
            Per-model :class:`Observables` NamedTuple class (from
            :meth:`SEDModel.Observables`). When provided, returns an
            instance; when ``None``, returns a dict.

        Returns
        -------
        Observables or dict
            ``phot_fnu`` and ``phot_rest_fnu`` populated.

        Raises
        ------
        ValueError
            If no ``*_phot_lnu_lut`` keys are present (i.e. the model was
            not built with ``approx=WavePrecomp()``).
        NotImplementedError
            If the observation has spectroscopy, line_fluxes, or
            spectral_indices.

        Notes
        -----
        **JIT-compatible**: yes, pure JAX arithmetic on the LUT entries.

        **This is a second implementation of :meth:`predict`, not a spelling of
        it.** The two are kept in sync by hand, deliberately: the speedup exists
        because this path reaches its answer without ever referencing the dense
        SED, which lets XLA dead-code-eliminate the full-resolution chain
        (#1109). Collapsing the two would delete the optimization. The standing
        goal is therefore not to merge them but to shrink and *bound* the
        divergence channel by channel, and to keep the remaining approximations
        named with their measured size, which is what follows.

        **Accuracy ledger, where this path differs from** :meth:`predict`:

        - **Stellar continuum**: K-point sub-band quadrature (#1122). The
          screen is *evaluated* at each node rather than sampled once, and
          converges as :math:`1/K^2`. K=5 (default): ≲0.6 % worst case in GALEX
          FUV; 3.2e-05 to 7.8e-04 on an FSPS/SDSS *griz* reference model over
          :math:`\tau_{\rm diff} \le 2`, :math:`z \le 1`. **This is the floor
          for the whole path**, no other channel can do better than the
          bucket that dominates the broadband.
        - **Nebular, under** ``dust_attenuation={'type': 'two_component'}``: *exact* since
          #1738. That component publishes the reddened continuum integrated
          through each band (``nebular_phot_lnu_attenuated_precomp``), so there
          is no band-averaging error left to quote: ≤3.5e-06 against the exact
          path on a fixture built to maximize it. Previously screened at
          :math:`\lambda_{\rm eff}`, which inflated the total gap by up to 26x
          over the stellar floor while carrying only 0.8-3.5 % of the band flux.
        - **Nebular, under** ``dust_attenuation={'type': 'single_component'}``: **still at**
          :math:`\lambda_{\rm eff}`. The qualifier above is not pedantry: this
          docstring claimed nebular was exact full stop, and it was measured
          wrong within a day of being written. :class:`DustAttenuationSEDComponent`
          declares ``sed_nebular`` an *optional* input purely as a topological
          ordering edge, its own docstring notes the screen "does not read the
          key directly (it acts on the already-summed ``sed_intrinsic``") so no
          separately reddened nebular SED exists there to project. Measured on an
          FSPS SSP through SDSS *gri*: 1.787e-03 at :math:`\tau_v`\ =1/z=0.05 and
          1.955e-03 at :math:`\tau_v`\ =2/z=1, against a stellar-only floor of
          ~6.1e-04, a ~3x inflation, versus the 26x removed on two-component.
          Bounded in ``tests/contract/test_precomp_channel_drift.py``.

          Fixing it means computing ``sed_neb · exp(-tau_v · k)`` in that
          component and projecting it through the same seam. Deliberately
          sequenced **after** #1808, which asks whether ``k(λ)`` may be
          precomputed at all: a nebular term reading today's cached ``k`` would
          inherit the freeze, and a later fix would move the stellar term onto
          the live curve while leaving nebular on the stale one. Two screens
          disagreeing inside one model is worse than the uniform staleness
          there now.
        - **Shock**: the worst remaining channel by two orders of magnitude, and
          **not** a band-averaging error despite what this docstring said for a
          long time. This path multiplies shock by ``a_diff·a_bc``; the exact
          path applies *no* stellar dust screen to it at all, because
          ``two_component`` never reads ``sed_shock`` and adds the non-nebular
          remainder unattenuated. Measured on an FSPS SSP through SDSS *gri*:
          **4.5 %** at :math:`\tau_{\rm bc}`\ =1/z=0.05, **7.4 %** at
          :math:`\tau`\ =2/z=0.05, **37.7 %** at :math:`\tau`\ =2/z=1, and in
          the exact path shock's contribution over its intrinsic value is
          1.66e-55 whatever :math:`\tau` is, i.e. the bare cosmology factor.
          A quadrature cannot close a factor-of-40 disagreement about *whether*
          a screen applies. Which path is right is a physics question, so the
          gap is bounded (``tests/contract/test_precomp_channel_drift.py``)
          rather than silently resolved.
        - **IGM**: ``igm_phot_factor`` band-averages :math:`T` alone,
          forming :math:`\langle S\rangle\langle T\rangle` where the flux needs
          :math:`\langle S\,T\rangle`. Across GALEX FUV at :math:`z\approx0.8`
          the transmission runs ~1 to ~0 *inside* the band and that covariance
          term reaches **−9.5 %**. Folded into the sub-band weights (#1135)
          wherever a mean-IGM model is precomputable; patchy reionization and
          DLAs read free parameters, so those configs keep the live path.
        - **Additive emitters** (dust IR, radio, X-ray, AGN), exact via the
          rank-1/rank-K band response where the emitter factorizes, else a dense
          band integral, else a single :math:`\lambda_{\rm eff}` sample; see
          ``tengri.components._band_projection.project_additive_onto_photometry``
          for which branch a given component takes.

        Re-measure rather than quoting these: every figure above is a property
        of an SSP grid, a filter set and a K, none of which are fixed by this
        method.

        See Also
        --------
        predict : The default projection path that integrates
            ``sed_intrinsic`` through filters. Stays the canonical
            reference; the exact path remains the default (``approx=None``).
        """
        from tengri.cosmology import luminosity_distance

        # Sum all *_phot_lnu_precomp contributions from components that published
        # one. New components add their precompute field to DerivedState and
        # apply(), no change to predict_via_precomp required.
        precomp_keys = [k for k in state.derived.field_names() if k.endswith("_phot_lnu_precomp")]
        precomp_contribs = [state.derived[k] for k in precomp_keys if k in state.derived]
        if not precomp_contribs:
            raise ValueError(
                "predict_via_precomp requires at least one *_phot_lnu_precomp in state.derived. "
                "Build the model with approx=WavePrecomp()."
            )
        # Part A (joint): this projector produces ONLY the photometry channel
        # (phot_fnu / phot_rest_fnu). On a joint photometry+spectroscopy model
        # the spectrum channel is projected separately by
        # ``predict_spectrum_via_precomp`` and merged by the caller, and line
        # fluxes / spectral indices are served by their own grid-independent
        # predict_* methods, so the presence of spectroscopy/lines/indices is
        # not a blocker here.
        if not self.can_do_photometry:
            raise ValueError("predict_via_precomp requires photometry to be configured.")

        # Sum all precompute contributions, rest-frame Lν at the source's z.
        total_phi = precomp_contribs[0]
        for c in precomp_contribs[1:]:
            total_phi = total_phi + c

        # 2026-05-20: the previous runtime guards used
        # ``float(L_ir) > 0`` / ``float(jnp.max(sed_nebular)) > 0`` to detect
        # "component ran on the wave grid but didn't publish a per-filter
        # precompute". They were correct in spirit but broke under
        # ``jax.jit`` tracing (float concretization on traced arrays). The
        # checks are dropped in favor of trusting the build-time wiring:
        # when ``approx=WavePrecomp(...)`` is set, the stellar component
        # publishes ``filter_eff_waves`` which downstream dust / nebular
        # components see and use to publish their own precomps. The top-
        # level "no ``*_phot_lnu_precomp`` keys at all" check above still
        # catches the user-error case of calling ``predict_via_precomp``
        # on a model built without WavePrecomp.
        a_lut = state.derived.get("dust_attenuation_precomp")
        a_bc_lut = state.derived.get("dust_bc_attenuation_precomp")
        # Dust attenuation applies to STELLAR + nebular (both arise from the
        # photosphere + birth cloud / diffuse ISM). AGN has its own attenuation
        # parameters (``agn_ebv_*``) and is not attenuated by the stellar dust
        # component. Compute the dust-attenuable bucket first; the rest is
        # added unattenuated afterwards.
        stellar_phi = state.derived.get("stellar_phot_lnu_precomp")
        stellar_phi = stellar_phi if stellar_phi is not None else jnp.zeros_like(total_phi)
        nebular_phi_for_dust = state.derived.get("nebular_phot_lnu_precomp")
        nebular_phi_for_dust = (
            nebular_phi_for_dust if nebular_phi_for_dust is not None else jnp.zeros_like(total_phi)
        )
        # Shock joins the young-limit bucket rather than the unattenuated
        # remainder: the exact path sums ``sed_shock`` into ``sed_intrinsic``
        # before dust runs, so it is reddened by the same screen. Leaving it in
        # ``unattenuated_phi`` would make the LUT read high wherever the screen
        # bites, and only for models that enable shock (#1375, #851).
        #
        # KNOWN RESIDUAL, and NOT the band-averaging one this comment used to
        # claim. It read "like the nebular bucket, shock is a single number per
        # filter, so the screen is applied at λ_eff... fixing it means giving
        # shock a sub-band LUT". Measured, that is the wrong mechanism: the
        # EXACT path applies no stellar screen to shock at all. two_component
        # never reads sed_shock, it forms non_stellar_other = non_stellar_pre_dust
        # - sed_neb and adds that bucket unattenuated, while this line multiplies
        # shock by a_diff·a_bc (0.025-0.109 at tau_bc=2). In the exact path
        # shock's photometric contribution over its intrinsic value measures
        # 1.66e-55 whatever tau is: the bare cosmology factor.
        #
        # So the gap is 4.5 % at tau_bc=1/z=0.05, 7.4 % at tau=2/z=0.05 and
        # 37.7 % at tau=2/z=1 on an FSPS SSP through SDSS gri, a disagreement
        # about WHETHER the screen applies, which no quadrature can close. With
        # tau=0 the paths agree to roundoff, localizing it here rather than in
        # the filter integration. Whether shocked gas should sit behind the
        # stellar screen is a physics decision, so this is left as measured and
        # bounded in tests/contract/test_precomp_channel_drift.py.
        # Shock attenuation (#1434): consume the band-integrated attenuated form published
        # by dust.two_component.apply (lines 963-989). This is the unified seam: both
        # exact and precomp paths apply the same young-limit dust screen
        # (tau_bc·k_bc + tau_diff·k_diff). Precomp consumes the band-integrated
        # product, not re-multiplying at lambda_eff, which prevents drift.
        #
        # Shock handling (#1434): depends on whether dust publishes attenuated form.
        # Two-component dust publishes shock_phot_lnu_attenuated_precomp (band-integrated
        # attenuated form, consumed by this path). Single-component dust and no-dust models
        # do not publish this, so shock must be handled via the legacy λ_eff screen.
        shock_phi_intrinsic = state.derived.get("shock_phot_lnu_precomp")
        shock_phi_attenuated_precomp = state.derived.get("shock_phot_lnu_attenuated_precomp")
        consume_attenuated = shock_phi_attenuated_precomp is not None
        # Detect dust presence: if dust publishes attenuation factors, it is active.
        dust_is_active = state.derived.get("dust_bc_attenuation_precomp") is not None
        # Structural gate (two-component only): if shock exists and dust is active,
        # the attenuated form MUST be published (not a silent failure on key absence).
        if (
            shock_phi_intrinsic is not None
            and dust_is_active
            and shock_phi_attenuated_precomp is None
        ):
            raise KeyError(
                "#1434: dust component active but shock_phot_lnu_attenuated_precomp "
                "not published by two_component.apply. Build ordering bug: sed_shock "
                "must be in optional_inputs() so ShockNebular runs before "
                "DustSEDComponent, making sed_shock available for dust to read and "
                "publish the attenuated form."
            )
        # Track shock separately for diagnostics (intrinsic form).
        shock_only_phi = (
            shock_phi_intrinsic if shock_phi_intrinsic is not None else jnp.zeros_like(total_phi)
        )
        # Bucket accounting depends on whether attenuated form is available:
        if consume_attenuated:
            # Two-component dust (#1434): shock is being replaced by attenuated form.
            # Subtract intrinsic shock from unattenuated_phi to avoid it leaking through.
            dust_attenuable_phi = stellar_phi + nebular_phi_for_dust
            unattenuated_phi = (
                total_phi - dust_attenuable_phi - shock_only_phi
                if shock_phi_intrinsic is not None
                else total_phi - dust_attenuable_phi
            )
        else:
            # Single-component dust or no dust: shock gets λ_eff screen (legacy behavior).
            # Include shock in nebular_phi_for_dust so it gets screened normally.
            if shock_phi_intrinsic is not None:
                nebular_phi_for_dust = nebular_phi_for_dust + shock_only_phi
            dust_attenuable_phi = stellar_phi + nebular_phi_for_dust
            unattenuated_phi = total_phi - dust_attenuable_phi

        # Two-component (Charlot & Fall) dust LUT.
        # Factorization: T(a, λ) = T_diff(λ) × T_bc(λ)^y(a).
        # At the filter level, with per-age stellar LUT
        # ``stellar_phot_lnu_per_age_precomp[a, b]``:
        #
        #     stellar_attenuated_b = Σ_a per_age[a, b] × A_diff(b) × A_bc(b)^y(a)
        #
        # ``A_bc(b)^y(a)`` interpolates between BC-attenuated (y=1) and
        # bare (y=0) stars. Smooth y(a) handles the transition.
        per_age = state.derived.get("stellar_phot_lnu_per_age_precomp")
        sub_per_age = state.derived.get("stellar_phot_lnu_per_age_subband_precomp")
        a_bc_sub = state.derived.get("dust_bc_attenuation_subband_precomp")
        _have_subband = a_bc_sub is not None and sub_per_age is not None

        # The sub-band tensor with the IGM already folded in at the quadrature
        # nodes (#1135). Present only when a mean-IGM model is precomputable.
        # ``igm_phot_factor`` band-averages T *alone*, unweighted by the spectrum
        # ⟨S⟩·⟨T⟩ where the flux needs ⟨S·T⟩; across GALEX FUV at z≈0.8 the
        # transmission runs from ~1 to ~0 inside the band and that covariance term
        # reaches −9.5 %. Contracting this tensor against the same dust screen
        # captures SED × dust × IGM in one sum.
        #
        # Carried SEPARATELY from ``stellar_attenuated`` rather than replacing it:
        # ``phot_rest_fnu`` is projected at z=0 and carries no IGM by contract.
        sub_per_age_igm = state.derived.get("stellar_phot_lnu_per_age_subband_igm_precomp")
        stellar_attenuated_igm = None

        if a_bc_lut is not None and (_have_subband or per_age is not None):
            a_diff_lut = state.derived["dust_diff_attenuation_precomp"]
            y_age = state.derived["dust_young_indicator"]

            if _have_subband:
                # K-point sub-band quadrature (#1122), supersedes the Taylor form.
                # Same Charlot & Fall factorization T(a, λ) = T_diff(λ)·T_bc(λ)^y(a),
                # but the screen is EVALUATED at each sub-band's quadrature node
                # rather than extrapolated from λ_eff:
                #
                #   stellar_b = Σ_a Σ_k Φ[a,b,k]·T_diff[a,b,k]·T_bc[a,b,k]^y(a)
                #
                # Converges as 1/K² (K=5: ≲0.6 % worst case in GALEX FUV) where the
                # Taylor extrapolation diverges (+45 % at z=0.05 → +215 % at z=1).
                a_diff_sub = state.derived["dust_diff_attenuation_subband_precomp"]
                t_sub = a_diff_sub * a_bc_sub ** y_age[:, None, None]
                stellar_attenuated = jnp.sum(sub_per_age * t_sub, axis=(0, 2))
                if sub_per_age_igm is not None:
                    # Same screen, same nodes, only the weights carry T (#1135).
                    stellar_attenuated_igm = jnp.sum(sub_per_age_igm * t_sub, axis=(0, 2))
            else:
                atten_bc_per_age = a_bc_lut[None, :] ** y_age[:, None]  # A_bc(λ_eff)^y(a)
                t_per_age = a_diff_lut[None, :] * atten_bc_per_age  # A_diff·A_bc^y
                stellar_attenuated = jnp.sum(per_age * t_per_age, axis=0)
                # First-order Taylor (Ψ) correction, only when the moment tensor was
                # built (approx=WavePrecomp(taylor_correction=True); #617).
                # Expand T_a(λ) ≈ T_a(λ_eff) + T_a'(λ_eff)·(λ−λ_eff). Using the
                # log-derivative identity T_a'/T_a = (ln A_diff)' + y·(ln A_bc)':
                #   T_a' = T_a · (logslope_diff + y·logslope_bc)
                # This avoids the A_bc^(y−1) pole, at X-ray/UV bands far off the
                # dust curve A_bc → 0, but T_a → 0 too, so T_a' → 0 (no 0·inf NaN).
                moment_per_age = state.derived.get("stellar_phot_moment_per_age_precomp")
                logslope_diff = state.derived.get("dust_diff_log_attenuation_slope_precomp")
                logslope_bc = state.derived.get("dust_bc_log_attenuation_slope_precomp")
                if (
                    moment_per_age is not None
                    and logslope_diff is not None
                    and logslope_bc is not None
                ):
                    t_slope_per_age = t_per_age * (
                        logslope_diff[None, :] + y_age[:, None] * logslope_bc[None, :]
                    )
                    stellar_attenuated = stellar_attenuated + jnp.sum(
                        moment_per_age * t_slope_per_age, axis=0
                    )

            # Nebular emission (Cue / CloudyGrid) arises in the HII regions
            # around the youngest stars, so it sees the full young-limit screen
            # (birth cloud AND diffuse (A_bc·A_diff, i.e. y=1)) matching the
            # exact path (two_component.py reddens the nebular SED by both
            # screens). The earlier diffuse-only ``A_diff·Φ_neb`` left
            # nebular-line-dominated bands ~18 % (τ=0.5) to ~37 % (τ=1) too
            # bright.
            #
            # No longer evaluated at λ_eff (#1738). The dust component publishes the
            # reddened nebular continuum integrated THROUGH each band, which is the
            # quantity the exact path computes, not the K-point approximation to it
            # that the stellar continuum uses (#1122), because the reddened nebular
            # continuum is already on the full grid wherever a dust component runs at
            # all. Sampling the screen at one wavelength per band is only correct
            # where the screen is flat across the filter, and nebular emission is
            # line-dominated: measured on a real FSPS SSP through SDSS griz it
            # inflated the precomp-vs-exact gap by up to 26x over the stellar-only
            # floor, on a bucket carrying just 0.8-3.5 % of the band flux.
            #
            # Shock is attenuated by the same dust screen (#1434). The dust component
            # publishes shock_phot_lnu_attenuated_precomp, the band-integrated attenuated
            # form. This is added as-is (not rescreened) to avoid drift from the exact path.
            nebular_exact = state.derived.get("nebular_phot_lnu_attenuated_precomp")
            shock_attenuated_for_output = (
                shock_phi_attenuated_precomp
                if shock_phi_attenuated_precomp is not None
                else jnp.zeros_like(total_phi)
            )
            if nebular_exact is not None:
                nebular_attenuated = nebular_exact + shock_attenuated_for_output
            else:
                nebular_attenuated = (
                    a_diff_lut * a_bc_lut * nebular_phi_for_dust + shock_attenuated_for_output
                )
            total_lnu = stellar_attenuated + nebular_attenuated + unattenuated_phi

        # Single-component dust via the Taylor expansion
        # f_b = A(λ_eff)·Φ_b + A'(λ_eff)·Ψ_b (Zacharegkas+2025).
        # When dust precompute is present, the Taylor moment Ψ MUST also be
        # present (the dust expansion is only valid with the second term).
        elif a_lut is not None:
            # Sub-band quadrature (#1122), single screen. Per-age Phi_k contracted
            # against the law EVALUATED at each node. Must be checked before the
            # Taylor form: the quadrature supersedes it, and without this branch a
            # single-component model would silently drop to the bare A(lam_eff)
            # form once ``taylor_correction`` defaulted off.
            a_sub = state.derived.get("dust_attenuation_subband_precomp")
            if a_sub is not None and sub_per_age is not None:
                # Per-age Φ_k contracted against the law EVALUATED at each node.
                # Checked before the Taylor form because the quadrature supersedes
                # it: without this branch a single-component model would silently
                # drop to the bare A(λ_eff) form once ``taylor_correction``
                # defaulted off, worse than what it replaced.
                stellar_attenuated = jnp.sum(sub_per_age * a_sub, axis=(0, 2))
                if sub_per_age_igm is not None:
                    # Same screen, same nodes, only the weights carry T (#1135).
                    stellar_attenuated_igm = jnp.sum(sub_per_age_igm * a_sub, axis=(0, 2))
                # Nebular (if any) publishes no sub-band tensors; keep it at λ_eff.
                dust_attenuated = stellar_attenuated + a_lut * nebular_phi_for_dust
            else:
                # Zeroth order: flat attenuation at the filter effective wavelength.
                dust_attenuated = a_lut * dust_attenuable_phi
                # First-order Taylor (Ψ) correction, applied only when the moment
                # tensor and attenuation slope were built, i.e.
                # approx=WavePrecomp(taylor_correction=True) (#617). With
                # taylor_correction=False neither is published, so the flat
                # A(λ_eff)·Φ form stands. Only stellar publishes a moment; nebular
                # is treated as Φ-only.
                a_slope_lut = state.derived.get("dust_attenuation_slope_precomp")
                stellar_psi = state.derived.get("stellar_phot_moment_precomp")
                if a_slope_lut is not None and stellar_psi is not None:
                    dust_attenuated = dust_attenuated + a_slope_lut * stellar_psi
            total_lnu = dust_attenuated + unattenuated_phi
        elif sub_per_age_igm is not None and sub_per_age is not None:
            # No dust component at all, but the IGM still needs the quadrature;
            # the ⟨S⟩·⟨T⟩ gap is a property of the band average, not of the dust.
            # Rebuild the stellar term from the sub-band sums so the IGM-free and
            # IGM-folded halves are consistent (Σ_k Φ_k = Φ exactly, the partition
            # is flux-conserving by construction, asserted in subband_quadrature).
            stellar_attenuated = jnp.sum(sub_per_age, axis=(0, 2))
            stellar_attenuated_igm = jnp.sum(sub_per_age_igm, axis=(0, 2))
            total_lnu = stellar_attenuated + (total_phi - stellar_phi)
        else:
            total_lnu = total_phi

        z = jnp.asarray(require_redshift(params, "observation.observation.predict_via_precomp"))
        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        log10_cos = log10_flux_scale(z, dl_cm)
        # Apply the flux scale using loss-scaling approach (#1388).
        # The flux projection seam has log10_flux_scale ~ -58 dex, which causes
        # the reverse-pass Jacobian 10**(-58) to underflow in float32. By boosting
        # and then dividing back, we lift the cotangent into range without
        # perturbing float64 (boost is an exact power of 2).
        # This is the scaled-SED contract: apply the scale without materializing
        # the problematic Jacobian inside the differentiated region.
        from tengri.utils.scale import pow10

        # Apply the scale using custom_vjp to reorder the backward pass and
        # avoid materializing the problematic Jacobian (~10^-58) in float32.
        # The custom backward splits the exponent into two representable chunks
        # (#1388).
        from tengri.observation.photometry import _apply_flux_scale_safe

        phot_fnu = _apply_flux_scale_safe(total_lnu, log10_cos)

        # phot_rest_fnu: the SED reprojected at z=0, d_L=10 pc, the galaxy as it is.
        from tengri.utils.physics_constants import TEN_PC_CM

        cosmology_rest = 1.0 / (4.0 * jnp.pi * TEN_PC_CM**2)
        phot_rest_fnu = _restband_lnu(state) * cosmology_rest

        # IGM / DLA attenuation. The IGM component publishes its full
        # transmission curve on the dense rest grid even under WavePrecomp, so
        # the per-band factor is the *filter-weighted mean* ⟨T⟩ through each
        # true bandpass, the same union-grid quadrature the exact path uses
        # (#1026). A point sample at the effective wavelength is blind to the
        # sharp Lyman-α edge for bands straddling it (T(λ_eff) ≈ 1 while a
        # third of the band sits in the forest, u band at z ≈ 1.9 read 4–7%
        # high). ⟨T⟩ drops only the covariance of the in-band SED structure
        # with T: −0.5% end-to-end for the same u band (the stellar Lyα
        # absorption sits on the smeared T edge), reaching a few % only in
        # Lyman-limit dropout bands (in-band T contrast ~1 × steep in-band
        # SED), use ``approx=None`` for precision work there.
        # A Taylor-Ψ cross-term cannot recover it: the moment expansion
        # assumes a smooth screen, and T_IGM steps within the band. The point
        # sample remains as the fallback when no padded filter curves were
        # published (#932 behavior).
        # Only the observed-frame flux is attenuated; ``phot_rest_fnu`` (z=0)
        # carries no IGM.
        # Prefer the build-time band factors. <T>_f depends only on (z, filter,
        # convention); the transmission is averaged alone, unweighted by the SED,
        # so the IGM component tabulates it against z at build time and publishes
        # ``igm_phot_factor``. Consuming the full-grid ``igm_transmission`` here
        # instead forced a 5994-point Inoue+2014 evaluation on EVERY call (12.1
        # MFLOPs to produce n_filters numbers) and kept the full-resolution grid
        # live, defeating the dead-code elimination that is the entire WavePrecomp
        # speedup: 108 us -> 1764 us since #932. Bit-identical at the z-nodes; a
        # fixed-z model has a single node and is exact.
        igm_factor = state.derived.get("igm_phot_factor")
        igm_trans = state.derived.get("igm_transmission")
        eff_waves = state.derived.get("filter_eff_waves")
        if igm_factor is None and igm_trans is not None and eff_waves is not None:
            # Fallback: patchy reionization / DLA read free parameters, so <T>_f
            # is not a function of redshift alone and cannot be tabulated.
            fw_pad = state.derived.get("phot_filter_waves_padded")
            ft_pad = state.derived.get("phot_filter_trans_padded")
            if fw_pad is not None and ft_pad is not None:
                from tengri.observation.photometry import lnu_filter_integral_batch

                igm_factor = lnu_filter_integral_batch(
                    igm_trans,
                    state.wave,
                    fw_pad,
                    ft_pad,
                    z,
                    convention=self.photometry.convention,
                )
            else:
                igm_factor = jnp.interp(jnp.asarray(eff_waves), state.wave, igm_trans)
        if igm_factor is not None:
            if stellar_attenuated_igm is not None:
                # Stellar already carries T evaluated AT the quadrature nodes
                # (#1135), so the band factor must not touch it; that would apply
                # the IGM twice. Everything the quadrature cannot reach (nebular
                # lines, AGN, dust emission) keeps ⟨T⟩_f, which is what it had
                # before; the stellar continuum dominates the broadband and is now
                # the accurate term.
                other_lnu = total_lnu - stellar_attenuated
                igm_lnu = other_lnu * igm_factor + stellar_attenuated_igm
                # Apply the scale using the same custom_vjp approach for consistency
                from tengri.observation.photometry import _apply_flux_scale_safe

                phot_fnu = _apply_flux_scale_safe(igm_lnu, log10_cos)
            else:
                phot_fnu = phot_fnu * igm_factor

        out = {"phot_fnu": phot_fnu, "phot_rest_fnu": phot_rest_fnu}

        if observables_type is not None:
            return observables_type(phot_fnu, phot_rest_fnu)
        return out

    def predict_spectrum_via_precomp(
        self,
        state,
        params,
        *,
        observables_type=None,
    ):
        """Project spectrum observables via the spectrum LUT.

        Opt-in fast path for ``approx=SpectrumPrecomp()``. Sums all ``*_spec_lnu_precomp``
        keys present in ``state.derived`` and applies the cosmology factor
        ``(1+z)/(4π·dl²)`` to convert to observed F_ν.

        Parameters
        ----------
        state : ForwardState
            Orchestrator state with at least ``spec_eff_waves`` and
            one or more ``*_spec_lnu_precomp`` entries.
        params : Mapping[str, jnp.ndarray]
            Param dict; reads ``redshift``.
        observables_type : type, optional
            Per-model :class:`Observables` NamedTuple class (from
            :meth:`SEDModel.Observables`). When provided, returns an
            instance; when ``None``, returns a dict.

        Returns
        -------
        Observables or dict
            ``spec_fnu`` populated from the spectrum LUT path.

        Raises
        ------
        ValueError
            If no ``*_spec_lnu_precomp`` keys are present.

        Notes
        -----
        **JIT-compatible**: yes, pure JAX arithmetic on the LUT entries.
        """
        from tengri.utils.cosmology import luminosity_distance

        # Sum all *_spec_lnu_precomp contributions from components that published one.
        precomp_keys = [k for k in state.derived.field_names() if k.endswith("_spec_lnu_precomp")]
        precomp_contribs = [state.derived[k] for k in precomp_keys if k in state.derived]
        if not precomp_contribs:
            raise ValueError(
                "predict_spectrum_via_precomp requires at least one *_spec_lnu_precomp "
                "in state.derived. Build the model with approx=SpectrumPrecomp()."
            )

        # Sum all per-pixel emitter contributions, rest-frame Lν at the
        # source's z. Emitters (stellar, nebular continuum, AGN) publish
        # ``*_spec_lnu_precomp``; the dust component publishes per-pixel
        # *transmission* (not an Lν), applied below.
        total_spec_lnu = precomp_contribs[0]
        for c in precomp_contribs[1:]:
            total_spec_lnu = total_spec_lnu + c

        # ── Dust attenuation on the pixel grid ──────────────────────────
        # A spectrum pixel is a single wavelength, so transmission T(λ_pix)
        # is exact, no Taylor moment (contrast predict_via_precomp). Dust
        # attenuates the stellar + nebular-continuum bucket; AGN carries its
        # own attenuation and is added unattenuated.
        stellar_phi = state.derived.get("stellar_spec_lnu_precomp")
        stellar_phi = stellar_phi if stellar_phi is not None else jnp.zeros_like(total_spec_lnu)
        nebular_phi = state.derived.get("nebular_spec_lnu_precomp")
        nebular_phi = nebular_phi if nebular_phi is not None else jnp.zeros_like(total_spec_lnu)
        dust_attenuable = stellar_phi + nebular_phi
        unattenuated = total_spec_lnu - dust_attenuable

        t_bc = state.derived.get("dust_spec_bc_transmission_precomp")
        t_diff = state.derived.get("dust_spec_diff_transmission_precomp")
        t_single = state.derived.get("dust_spec_transmission_precomp")
        per_age = state.derived.get("stellar_spec_lnu_per_age_precomp")

        if t_bc is not None and t_diff is not None and per_age is not None:
            # Two-component (Charlot & Fall): T(a, λ) = T_diff(λ)·T_bc(λ)^y(a).
            y_age = state.derived["dust_young_indicator"]
            atten_bc_per_age = t_bc[None, :] ** y_age[:, None]  # (n_age, n_pix)
            stellar_attenuated = jnp.sum(per_age * atten_bc_per_age, axis=0) * t_diff
            # Nebular emission arises in the HII regions around the youngest
            # stars, so it sees the full young-limit screen, birth cloud AND
            # diffuse (T_bc · T_diff, i.e. y=1), matching the exact path's
            # emission treatment (two_component.py reddens the nebular SED by
            # τ_bc·k_bc + τ_diff·k_diff). Applying only T_diff here under-
            # attenuated the nebular lines by the missing 1/T_bc factor.
            nebular_attenuated = t_diff * t_bc * nebular_phi
            total_spec_lnu = stellar_attenuated + nebular_attenuated + unattenuated
        elif t_single is not None:
            # Single-component: uniform screen T(λ_pix) on the attenuable bucket.
            total_spec_lnu = dust_attenuable * t_single + unattenuated
        # else: no dust LUT published, leave total_spec_lnu unattenuated.

        # Apply cosmology: observed F_ν = L_ν / (4π·d_L²) × (1 + z)
        z = jnp.asarray(
            require_redshift(params, "observation.observation.predict_spectrum_via_precomp")
        )
        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        log10_cos = log10_flux_scale(z, dl_cm)
        spec_fnu = apply_log10_scale(total_spec_lnu, log10_cos)

        # spec_rest_fnu: same LUT sum projected at z=0, d_L=10pc.
        from tengri.utils.physics_constants import TEN_PC_CM

        cosmology_rest = 1.0 / (4.0 * jnp.pi * TEN_PC_CM**2)
        spec_rest_fnu = total_spec_lnu * cosmology_rest

        # IGM / DLA attenuation. Unlike the photometry LUT, the spectrum LUT is
        # per-pixel, so the observed-frame transmission is applied *exactly*:
        # sample the transmission the IGM component published on the rest grid
        # at each pixel's rest effective wavelength. Only the observed-frame flux
        # is attenuated; spec_rest_fnu (z=0) carries no IGM (#932).
        # Prefer the build-time per-pixel table. A pixel's rest effective wavelength
        # is wave_obs/(1+z) and the curve is T(wave_rest*(1+z), z), so the sample
        # collapses to T at the fixed observed instrument grid, a function of
        # (z, pixel) alone. Sampling the full-grid curve here instead forced a
        # 5994-point Inoue+2014 evaluation on every call and left the LUT buying
        # NOTHING (2120 us exact vs 2098 us LUT). Bit-identical at the z-nodes.
        igm_factor = state.derived.get("igm_spec_factor")
        igm_trans = state.derived.get("igm_transmission")
        eff_waves = state.derived.get("spec_eff_waves")
        if igm_factor is None and igm_trans is not None and eff_waves is not None:
            # Fallback: patchy / DLA read free parameters, so the transmission is
            # not a function of redshift alone and cannot be tabulated.
            igm_factor = jnp.interp(jnp.asarray(eff_waves), state.wave, igm_trans)
        if igm_factor is not None:
            spec_fnu = spec_fnu * igm_factor

        # Flux calibration, the same Chebyshev polynomial the exact path applies
        # inside ``project_spectrum`` (#1046). The LUT path must apply it itself:
        # it never calls ``project_spectrum``, so without this a model built with
        # ``approx=SpectrumPrecomp()`` silently DROPPED its calibration while the
        # exact model applied it, cal_c* gradients were exactly zero under the
        # LUT. Same failure as the LUT dropping AGN + nebular (#737/#740): the
        # speed knob quietly changing the physics.
        #
        # Observed frame only. The calibration models the *instrument's* flux
        # error, which has no meaning for the rest-frame (z=0, 10 pc) spectrum,
        # so ``spec_rest_fnu`` is deliberately left uncalibrated.
        cal_coeffs = self.spectroscopy.calibration_coeffs(params)
        if cal_coeffs is not None:
            from tengri.observation.calibration import apply_calibration

            wmin, wmax = self.spectroscopy.calibration_wave_range
            spec_fnu = apply_calibration(
                spec_fnu, self.spectroscopy.wave_obs, cal_coeffs, wmin, wmax
            )

        out = {"spec_fnu": spec_fnu, "spec_rest_fnu": spec_rest_fnu}

        if observables_type is not None:
            # Only populate fields the per-model Observables NamedTuple
            # actually declares (it carries spec_fnu but not necessarily
            # spec_rest_fnu, and may also carry phot_* for joint models).
            avail = {k: v for k, v in out.items() if k in observables_type._fields}
            return observables_type(**avail)
        return out

    # ── Display ───────────────────────────────────────────────────

    def summary(self) -> str:
        """Return a human-readable summary of the observation.

        Returns
        -------
        str
            Multi-line summary string with filter counts, instrument config,
            noise settings, and total data point count.

        Notes
        -----
        Used for logging and diagnostics. Does not execute any inference;
        purely informational output.

        """
        lines = ["Observation"]
        lines.append("-" * 50)

        if self.photometry is not None:
            lines.append(f"  Photometry : {self.photometry.summary()}")

        if self.spectroscopy is not None:
            lines.append(f"  Spectroscopy : {self.spectroscopy.summary()}")

        if self.line_fluxes is not None:
            lines.append(f"  Line fluxes: {self.line_fluxes.summary()}")

        if self.spectral_indices is not None:
            lines.append(f"  Indices    : {self.spectral_indices.summary()}")

        if self.noise is not None:
            lines.append(f"  Noise      : {self.noise.summary()}")

        lines.append(f"  Data type  : {self.data_type}")
        lines.append(f"  N data     : {self.n_data}")
        if self.is_joint:
            lines.append(f"               ({self.n_data_phot} phot + {self.n_data_spec} spec)")
        if self.has_line_fluxes:
            lines.append(f"               + {self.n_data_lines} line fluxes")
        if self.has_spectral_indices:
            lines.append(f"               + {self.n_data_indices} spectral indices")

        obs_params = self.get_all_params()
        if obs_params:
            lines.append(f"  Auto params: {', '.join(sorted(obs_params.keys()))}")

        return "\n".join(lines)
