# SPDX-License-Identifier: BSD-3-Clause
"""Unified observation configuration for tengri SED fitting.

Bundles photometric and/or spectroscopic setup with noise configuration
into a single declarative object. Inspired by Synthesizer's Instrument
pattern, adapted for tengri's JAX/differentiable inference context.

The Observation class is a frozen configuration container — it never
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

    def __post_init__(self):
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

    # ── Capability queries ────────────────────────────────────────

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

        Returns
        -------
        jnp.ndarray, shape (n_pixels,)
            Spectroscopic flux at each pixel [erg/s/Hz].

        Notes
        -----
        Requires spectroscopy to be configured. Applies LSF convolution
        if a resolution profile is specified. Returns data ready for
        likelihood evaluation against observed spectra.

        """
        if self.spectroscopy is None:
            raise ValueError("No spectroscopy configured in this Observation.")

        from tengri.observation.spectrum import apply_lsf, compute_spectrum

        wave_rest = sed_result.wavelength / (1.0 + z)
        wave_obs = self.spectroscopy.wave_obs
        flux = compute_spectrum(sed_result.sed, wave_rest, wave_obs, z, dl_cm)

        if self.spectroscopy.resolution is not None:
            flux = apply_lsf(
                flux,
                wave_obs,
                self.spectroscopy.resolution,
                sigma_lib_kms=self.spectroscopy.sigma_lib_kms,
                sigma_v_kms=sigma_v_kms,
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
        Protocol in :mod:`tengri.protocols.observation`. Phase 1 of the
        forward-projection unification (``docs/dev/photometry_path_unification.md``).

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

            - ``"phot_fnu"`` if :attr:`can_do_photometry` — shape ``(n_filters,)``,
              F_nu [erg/s/cm^2/Hz].
            - ``"spec_fnu"`` if :attr:`can_do_spectroscopy` — shape ``(n_pix,)``,
              F_nu [erg/s/cm^2/Hz].

            When ``observables_type`` is provided: an instance of the passed
            NamedTuple class with fields populated in order: ``phot_fnu``,
            ``phot_rest_fnu``, ``spec_fnu``, ``lines_flux``, ``indices``.

        Notes
        -----
        **JIT-compatible**: yes — same kernels as
        :meth:`observe_photometry` and :meth:`observe_spectrum`.

        Joint observations (``is_joint``) return both keys from a single
        forward pass — Phase 2 of the unification plan will collapse
        ``loss_functions._build_prediction``'s two-call joint branch
        onto this single call.

        When ``observables_type`` is provided and the observation has
        ``line_fluxes`` or ``spectral_indices`` configured, raises
        ``NotImplementedError`` — Phase 3+ territory.
        """
        from tengri.cosmology import luminosity_distance
        from tengri.observation.photometry import compute_flux_density_batch
        from tengri.observation.spectrum import apply_lsf, compute_spectrum

        z = jnp.asarray(params.get("redshift", 0.0))
        if dl_cm is None:
            dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        else:
            dl_cm = jnp.asarray(dl_cm)

        sed_rest = state.sed_intrinsic
        wave_rest = state.wave

        # IGM attenuation is an observed-frame transmission the IGM component
        # publishes on the rest grid (``T`` evaluated at ``wave_obs =
        # wave*(1+z)``). Both projections below redshift ``sed_rest``
        # internally, so multiplying here — before projection — attenuates the
        # observed SED for photometry *and* spectroscopy at once. Applying the
        # full transmission curve (not a single per-band effective-wavelength
        # factor) is what captures the sharp Lyman break across broad bands and
        # spectral pixels at high redshift (#932). ``T`` shares the rest grid
        # with ``sed_rest``; the key is absent (structural no-op) when IGM is
        # disabled, so low-z / IGM-off models are bit-unchanged.
        igm_trans = (
            state.derived.get("igm_transmission", None) if state.derived is not None else None
        )
        if igm_trans is not None:
            sed_rest = sed_rest * igm_trans

        out: dict[str, jnp.ndarray] = {}

        if self.can_do_photometry:
            # ``predict`` is the canonical exact (compositional) path: it
            # integrates ``state.sed_intrinsic`` through each filter without
            # approximation. The precompute LUT path is opt-in via
            # :meth:`predict_via_precomp` (or its callers); this method does
            # NOT fall through to the LUT by default — exact-first is the
            # default semantics for ``observation.predict``.
            #
            # Use the batched (vmapped) projection over filters padded to
            # ``FILTER_COUNT_BUCKETS`` so distinct Photometry instances with
            # similar filter counts share an XLA compile. Padded rows have
            # all-zero transmission and contribute zero by construction.
            n_real = self.photometry.n_filters
            phot = compute_flux_density_batch(
                sed_rest,
                wave_rest,
                self.photometry._fw_padded,
                self.photometry._ft_padded,
                z,
                dl_cm,
                convention=self.photometry.convention,
            )[:n_real]
            out["phot_fnu"] = phot

        if self.can_do_spectroscopy:
            wo = wave_obs if wave_obs is not None else self.spectroscopy.wave_obs
            flux = compute_spectrum(sed_rest, wave_rest, wo, z, dl_cm)
            resolution = (
                lsf_resolution if lsf_resolution is not None else self.spectroscopy.resolution
            )
            if resolution is not None:
                sigma_lib = (
                    lsf_sigma_lib_kms
                    if lsf_sigma_lib_kms is not None
                    else self.spectroscopy.sigma_lib_kms
                )
                n_bins = lsf_n_bins if lsf_n_bins is not None else self.spectroscopy.lsf_n_bins
                flux = apply_lsf(
                    flux,
                    wo,
                    resolution,
                    sigma_lib_kms=sigma_lib,
                    n_bins=n_bins,
                    sigma_v_kms=sigma_v_kms,
                )
            out["spec_fnu"] = flux

        # Phase 2: if observables_type is provided, populate and return NamedTuple.
        # Line fluxes / line ratios / spectral indices are NOT projection
        # observables — they are scalar measurables computed separately
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
        """Project observables via the photometric LUT instead of integrating ``sed_intrinsic``.

        Phase 3c-3 opt-in fast path. Sums all ``*_phot_lnu_lut`` keys
        present in ``state.derived`` (the rest-frame Lν contributions
        from each component that publishes one) and applies the cosmology
        factor ``(1+z)/(4π·dl²)`` to convert to observed F_ν.

        Components that publish a LUT entry as of this PR:

        - ``stellar_phot_lnu_precomp`` — :class:`StellarSEDComponent` (Phase 3b/3c-1).
          For ``BakedIn`` nebular backends this already contains the nebular
          contribution, since the SSP grid carries baked-in nebular emission.
        - ``nebular_phot_lnu_precomp`` — :class:`NebularSEDComponent` (Phase 3c-3b
          and later, when the backend supports filter-level precomputation;
          non-BakedIn backends only).

        Future entries (``dust_*``, ``agn_*``, …) sum in automatically.

        Photometry only — spectroscopy / line_fluxes / spectral_indices
        land in Phase 3c-3 final scope.

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
        **JIT-compatible**: yes — pure JAX arithmetic on the LUT entries.

        See Also
        --------
        predict : The default projection path that integrates
            ``sed_intrinsic`` through filters. Stays the canonical
            reference until Phase 3c-3e flips the default.
        """
        from tengri.cosmology import luminosity_distance

        # Sum all *_phot_lnu_precomp contributions from components that published
        # one. New components add their precompute field to DerivedState and
        # apply() — no change to predict_via_precomp required.
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
        # predict_* methods — so the presence of spectroscopy/lines/indices is
        # not a blocker here.
        if not self.can_do_photometry:
            raise ValueError("predict_via_precomp requires photometry to be configured.")

        # Sum all precompute contributions — rest-frame Lν at the source's z.
        total_phi = precomp_contribs[0]
        for c in precomp_contribs[1:]:
            total_phi = total_phi + c

        # Phase 3d-5 (2026-05-20): the previous runtime guards used
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
        dust_attenuable_phi = stellar_phi + nebular_phi_for_dust
        unattenuated_phi = total_phi - dust_attenuable_phi

        # Phase 3c-3c-iv-c: two-component (Charlot & Fall) dust LUT.
        # Factorization: T(a, λ) = T_diff(λ) × T_bc(λ)^y(a).
        # At the filter level, with per-age stellar LUT
        # ``stellar_phot_lnu_per_age_precomp[a, b]``:
        #
        #     stellar_attenuated_b = Σ_a per_age[a, b] × A_diff(b) × A_bc(b)^y(a)
        #
        # ``A_bc(b)^y(a)`` interpolates between BC-attenuated (y=1) and
        # bare (y=0) stars. Smooth y(a) handles the transition.
        per_age = state.derived.get("stellar_phot_lnu_per_age_precomp")
        if a_bc_lut is not None and per_age is not None:
            a_diff_lut = state.derived["dust_diff_attenuation_precomp"]
            y_age = state.derived["dust_young_indicator"]
            atten_bc_per_age = a_bc_lut[None, :] ** y_age[:, None]  # A_bc(λ_eff)^y(a)
            t_per_age = a_diff_lut[None, :] * atten_bc_per_age  # T_a(λ_eff) = A_diff·A_bc^y
            stellar_attenuated = jnp.sum(per_age * t_per_age, axis=0)
            # First-order Taylor (Ψ) correction — only when the moment tensor was
            # built (approx=WavePrecomp(taylor_correction=True), the default; #617).
            # Expand T_a(λ) ≈ T_a(λ_eff) + T_a'(λ_eff)·(λ−λ_eff). Using the
            # log-derivative identity T_a'/T_a = (ln A_diff)' + y·(ln A_bc)':
            #   T_a' = T_a · (logslope_diff + y·logslope_bc)
            # This avoids the A_bc^(y−1) pole — at X-ray/UV bands far off the dust
            # curve A_bc → 0, but T_a → 0 too, so T_a' → 0 cleanly (no 0·inf NaN).
            moment_per_age = state.derived.get("stellar_phot_moment_per_age_precomp")
            logslope_diff = state.derived.get("dust_diff_log_attenuation_slope_precomp")
            logslope_bc = state.derived.get("dust_bc_log_attenuation_slope_precomp")
            _have_taylor = (
                moment_per_age is not None
                and logslope_diff is not None
                and logslope_bc is not None
            )
            if _have_taylor:
                t_slope_per_age = t_per_age * (
                    logslope_diff[None, :] + y_age[:, None] * logslope_bc[None, :]
                )
                stellar_attenuated = stellar_attenuated + jnp.sum(
                    moment_per_age * t_slope_per_age, axis=0
                )
            # Nebular emission (Cue / CloudyGrid) arises in the HII regions
            # around the youngest stars, so it sees the full young-limit screen
            # — birth cloud AND diffuse (A_bc·A_diff, i.e. y=1) — matching the
            # exact path (two_component.py reddens the nebular SED by both
            # screens). The earlier diffuse-only ``A_diff·Φ_neb`` left
            # nebular-line-dominated bands ~18 % (τ=0.5) to ~37 % (τ=1) too
            # bright. Zeroth order in the filter; the residual intra-filter
            # Taylor term (#617) is the same one the stellar continuum carries.
            nebular_attenuated = a_diff_lut * a_bc_lut * nebular_phi_for_dust
            total_lnu = stellar_attenuated + nebular_attenuated + unattenuated_phi

        # Phase 3c-3c-iii: single-component dust via the Taylor expansion
        # f_b = A(λ_eff)·Φ_b + A'(λ_eff)·Ψ_b (Zacharegkas+2025).
        # When dust precompute is present, the Taylor moment Ψ MUST also be
        # present (the dust expansion is only valid with the second term).
        elif a_lut is not None:
            # Zeroth order: flat attenuation at the filter effective wavelength.
            dust_attenuated = a_lut * dust_attenuable_phi
            # First-order Taylor (Ψ) correction — applied only when the moment
            # tensor and attenuation slope were built, i.e.
            # approx=WavePrecomp(taylor_correction=True) (the default; #617).
            # With taylor_correction=False neither is published, so the flat
            # A(λ_eff)·Φ form stands. Only stellar publishes a moment; nebular is
            # treated as Φ-only.
            a_slope_lut = state.derived.get("dust_attenuation_slope_precomp")
            stellar_psi = state.derived.get("stellar_phot_moment_precomp")
            if a_slope_lut is not None and stellar_psi is not None:
                dust_attenuated = dust_attenuated + a_slope_lut * stellar_psi
            total_lnu = dust_attenuated + unattenuated_phi
        else:
            total_lnu = total_phi

        z = jnp.asarray(params.get("redshift", 0.0))
        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        cosmology = (1.0 + z) / (4.0 * jnp.pi * dl_cm**2)
        phot_fnu = total_lnu * cosmology

        # phot_rest_fnu: same LUT sum projected at z=0, d_L=10pc.
        from tengri.utils.physics_constants import TEN_PC_CM

        cosmology_rest = 1.0 / (4.0 * jnp.pi * TEN_PC_CM**2)
        phot_rest_fnu = total_lnu * cosmology_rest

        # IGM / DLA attenuation. The LUT path carries no per-λ observed SED, so
        # the exact full-curve transmission used by the wave-grid projection
        # (observation.predict) is unavailable here. Instead sample the
        # transmission the IGM component published on the rest grid at each
        # filter's rest effective wavelength — a per-band (effective-wavelength)
        # approximation. It is coarse across the *sharp* Lyman break for bands
        # straddling it (use ``approx=None`` for unbiased rest-UV IGM at high z,
        # mirroring the WavePrecomp dust-Taylor caveat), but recovers the bulk
        # of the attenuation instead of silently dropping it entirely (#932).
        # Only the observed-frame flux is attenuated; ``phot_rest_fnu`` (z=0)
        # carries no IGM.
        igm_trans = state.derived.get("igm_transmission")
        eff_waves = state.derived.get("filter_eff_waves")
        if igm_trans is not None and eff_waves is not None:
            igm_factor = jnp.interp(jnp.asarray(eff_waves), state.wave, igm_trans)
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
        """Project spectrum observables via the spectrum LUT (Phase 5).

        Phase 5 opt-in fast path for spectroscopy. Sums all ``*_spec_lnu_precomp``
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
        **JIT-compatible**: yes — pure JAX arithmetic on the LUT entries.
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

        # Sum all per-pixel emitter contributions — rest-frame Lν at the
        # source's z. Emitters (stellar, nebular continuum, AGN) publish
        # ``*_spec_lnu_precomp``; the dust component publishes per-pixel
        # *transmission* (not an Lν), applied below.
        total_spec_lnu = precomp_contribs[0]
        for c in precomp_contribs[1:]:
            total_spec_lnu = total_spec_lnu + c

        # ── Dust attenuation on the pixel grid ──────────────────────────
        # A spectrum pixel is a single wavelength, so transmission T(λ_pix)
        # is exact — no Taylor moment (contrast predict_via_precomp). Dust
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
            # stars, so it sees the full young-limit screen — birth cloud AND
            # diffuse (T_bc · T_diff, i.e. y=1) — matching the exact path's
            # emission treatment (two_component.py reddens the nebular SED by
            # τ_bc·k_bc + τ_diff·k_diff). Applying only T_diff here under-
            # attenuated the nebular lines by the missing 1/T_bc factor.
            nebular_attenuated = t_diff * t_bc * nebular_phi
            total_spec_lnu = stellar_attenuated + nebular_attenuated + unattenuated
        elif t_single is not None:
            # Single-component: uniform screen T(λ_pix) on the attenuable bucket.
            total_spec_lnu = dust_attenuable * t_single + unattenuated
        # else: no dust LUT published — leave total_spec_lnu unattenuated.

        # Apply cosmology: observed F_ν = L_ν / (4π·d_L²) × (1 + z)
        z = jnp.asarray(params.get("redshift", 0.0))
        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        cosmology = (1.0 + z) / (4.0 * jnp.pi * dl_cm**2)
        spec_fnu = total_spec_lnu * cosmology

        # spec_rest_fnu: same LUT sum projected at z=0, d_L=10pc.
        from tengri.utils.physics_constants import TEN_PC_CM

        cosmology_rest = 1.0 / (4.0 * jnp.pi * TEN_PC_CM**2)
        spec_rest_fnu = total_spec_lnu * cosmology_rest

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
