# SPDX-License-Identifier: BSD-3-Clause
"""ForwardModel, the outer shell of the forward chain.

Owns a tuple of :class:`Population` objects and an :class:`Observation`.
Exposes a single ``.predict(params)`` method that inference calls.
The architecture spec is at
``docs/dev/archive/forward-model-architecture.md`` §5–§6.

Single-population fits use bare parameter names
(``sfh_dpl_alpha``, ``dust_tau_v``). Multi-population fits use the
``{population_name}.{prefix}_{param}`` namespace defined in
`ADR-0012 <../adr/0012-forward-model-population.md>`_. Cross-population
state keys follow the same convention via the typed
:class:`tengri.protocols.DerivedState`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from tengri.forward.population import Population
from tengri.inference._backend_registry import DEFAULT_METHOD
from tengri.protocols.component import ForwardState
from tengri.protocols.derived_state import DerivedState

__all__ = ["ForwardModel"]


def _filters_fingerprint(obs):
    """Compute content-hash of filter transmission curves.

    Parameters
    ----------
    obs: object
        Observation model with optional ``.photometry`` attribute.

    Returns
    -------
    int or None
        Hash of filter transmission curves, or None if obs has no photometry.
    """
    if obs is None or not hasattr(obs, "photometry") or obs.photometry is None:
        return None
    return hash(tuple(np.asarray(t).tobytes() for t in obs.photometry.filter_trans))


def _approx_family_map(approx):
    """Split an ``approx`` grammar value into per-family configs.

    Returns ``None`` when the value is not parseable (unknown member or a
    duplicated family), the caller then falls through to ``with_approx``,
    whose canonical validation owns the teaching error.
    """
    from tengri.forward.sed_model import FeaturePrecomp, SpectrumPrecomp, WavePrecomp

    items = approx if isinstance(approx, tuple) else (approx,)
    fam = {"wave": None, "spectrum": None, "feature": None}
    for item in items:
        if isinstance(item, WavePrecomp):
            key = "wave"
        elif isinstance(item, SpectrumPrecomp):
            key = "spectrum"
        elif isinstance(item, FeaturePrecomp):
            key = "feature"
        else:
            return None
        if fam[key] is not None:
            return None
        fam[key] = item
    return fam


def _resolve_sed_approx(sed, approx, observation):
    """Spec §5 reuse-on-match (#1367): reuse a sed that already carries exactly
    the requested LUT against the same filters; otherwise rebuild it against
    the authoritative observation."""
    fam = _approx_family_map(approx)
    sed_obs = getattr(sed, "observation", None)
    same_filters = sed_obs is not None and _filters_fingerprint(sed_obs) == _filters_fingerprint(
        observation
    )
    if (
        fam is not None
        and same_filters
        and getattr(sed, "_approx_config_wave", None) == fam["wave"]
        and getattr(sed, "_approx_config_spec", None) == fam["spectrum"]
        and getattr(sed, "_approx_config_feature", None) == fam["feature"]
    ):
        return sed
    return sed.with_approx(approx, observation=observation)


@dataclass(frozen=True)
class ForwardModel:
    """The outer shell of the forward model, and the surface inference consumes.

    Holds one or more populations plus an observation, and exposes
    ``.predict(params)`` as the single API every inference backend talks to.
    A backend never has to ask which channels exist, the returned dict says
    so, which is what lets one ``Fitter`` drive photometry, spectroscopy,
    joint, and hierarchical fits without branching.

    Build the SED chain, wrap it, fit it. :meth:`fit` is the canonical entry
    point for inference (issue #211); it is exactly
    ``Fitter(self, data, noise).run(method)``.

    Parameters
    ----------
    populations: tuple of Population
        One or more populations. Each carries an SED ``SubModel`` and
        optionally a spatial ``SubModel``. Names must be distinct.
    observation: object
        Observation model exposing ``predict(state, params) -> dict``
        (single-population) and ``predict_summed(per_pop_states, params)``
        (multi-population).

    Notes
    -----
    **Construction.** Prefer :meth:`build` over the raw constructor, it
    accepts the ``sed=`` / ``population=`` / ``populations=`` forms and wraps
    them into the uniform ``populations`` tuple for you.

    **Parameter names.** Single-population fits use flat names
    (``sfh_dpl_alpha``, ``dust_tau_v``). Multi-population decompositions
    namespace by population, ``<population>.<component_prefix>_<param>``,
    ``disc.sfh_dpl_alpha``, ``bulge.dust_tau_v``, so three stellar components
    in three populations no longer collide (ADR-0012).

    **JIT.** ``predict`` and the channel-specific ``predict_photometry`` /
    ``predict_spectrum`` are traced by the inference backends under
    ``jax.jit``. Prefer :meth:`predict_photometry` on the fitting hot path;
    :meth:`predict_observables` returns the full dict and bypasses the
    photometry lookup table.

    See Also
    --------
    build: construct one, the recommended path.
    fit: run inference; the canonical entry point.
    tengri.SEDModel: the SED physics chain that goes inside a population.
    tengri.Fitter: the inference engine :meth:`fit` delegates to.

    Examples
    --------
    Wrap an SED chain and fit photometry:

    >>> from tengri import ForwardModel, SEDModel  # doctest: +SKIP
    >>> sed = SEDModel.build(ssp_data=ssp, observation=obs, **config)  # doctest: +SKIP
    >>> forward = ForwardModel.build(sed=sed, observation=obs)  # doctest: +SKIP
    >>> result = forward.fit(flux, noise, method="mcmc_nuts")  # doctest: +SKIP
    >>> result.properties["stellar_mass"].shape  # doctest: +SKIP
    (4000,)

    Hierarchical fit over a population sharing PSD hyperparameters:

    >>> pop = PopulationSEDModel(  # doctest: +SKIP
    ...     sed=sed,
    ...     galaxies=galaxies,
    ...     shared=("sfh_field_psd_sigma", "sfh_field_psd_tau_myr"),
    ... )
    >>> forward = ForwardModel.build(population=pop, observation=obs)  # doctest: +SKIP
    >>> result = forward.fit(method="vi")  # doctest: +SKIP
    """

    populations: tuple[Population, ...]
    observation: Any
    mode: str = "single"

    # ── Legacy-SEDModel delegations (for Fitter consumption) ─────────
    # The Fitter inner machinery (loss_fn, JIT compile, posterior
    # warm-start) reaches into ~12 SEDModel attributes. These
    # properties delegate so a ForwardModel can stand in for an
    # SEDModel anywhere the Fitter needs it. For hierarchical fits,
    # the inner SubModel is a PopulationSEDModel whose own attributes
    # delegate to its template SEDModel, three-level chain.

    @property
    def spec(self):
        """The :class:`Parameters`-shaped spec the Fitter consumes.

        Delegates to ``self.populations[0].sed.spec``. For single-
        galaxy fits, that's the :class:`SEDModel`'s scalar spec; for
        hierarchical fits (PopulationSEDModel), it's the batched
        :class:`PopulationSpecView` (see PR #241). The Fitter sees
        the same Protocol surface either way and doesn't need to
        know which.

        Notes
        -----
        For multi-population galaxy decompositions (ADR-0012), this
        returns the *first* population's spec, those fits use
        namespaced parameter names and have their own conventions
        handled elsewhere in the pipeline.
        """
        return self.populations[0].sed.spec

    def _inner_sed_for_delegation(self):
        """Resolve the underlying SEDModel held by the first population.

        For plain SEDModel forwards, the inner SED is ``populations[0].sed``;
        for PopulationSEDModel-wrapped forwards, it's
        ``populations[0].sed.sed`` (the template). Migration-2 properties
        share this resolver to avoid duplicating the walk.
        """
        sub = self.populations[0].sed
        return getattr(sub, "sed", sub)

    @property
    def wave_obs(self):
        """Observed-frame spectroscopy wavelength grid, or ``None``.

        Delegates to the first population's inner SED, single-population
        and hierarchical fits share one spectroscopy grid across the
        channel. Migration 2 step 1 promotes this from the legacy
        ``__getattr__`` fall-through to a first-class property.
        """
        return getattr(self._inner_sed_for_delegation(), "wave_obs", None)

    @property
    def has_fixedz_photometry_precompute(self) -> bool:
        """Fast-path eligibility, delegated from the inner SED (#620)."""
        return bool(
            getattr(self._inner_sed_for_delegation(), "has_fixedz_photometry_precompute", False)
        )

    # ── Inner-SED delegation (#1300) ─────────────────────────────────
    #
    # ForwardModel is the canonical inference surface (#211) but has no
    # ``__getattr__`` fall-through, it forwards through an explicit list. The
    # inference stack does ``model = fitter.model`` and then calls SEDModel
    # methods on it, so anything absent from that list raised AttributeError:
    # the DEPRECATED ``Fitter(sed_model, ...)`` path worked and the recommended
    # one did not. A sweep of ``src/tengri/inference`` for ``model.<attr>``
    # found twelve such methods, reached from loss_functions, jit_engine,
    # fitter, posterior and _sample_utils.
    #
    # Forwarding is semantically correct for a SINGLE population: the JIT cache
    # is keyed on ``compile_signature()`` rather than object identity, and the
    # ``*_via_state`` pair runs the component chain against the SED's own
    # observation. It is NOT correct for a multi-population forward, where
    # ``populations[0]`` is an arbitrary pick, those raise instead.

    _DELEGATED_TO_INNER_SED = (
        "_has_line_catalog",
        "measure_line_fluxes",
        "_get_or_build_predict_observables_jit",
        "_photometry_via_state",
        "_spectrum_via_state",
        "_predict_derived",
        "_predict_rest_sed",
        "_predict_sfh_quantities",
        "_template_data_for_jit",
        "ssp_data",
        "mock",
        "name",
        "predict_observables_jit",
        "predict_properties",
    )

    def _single_inner_sed(self, what: str):
        """Inner SED for a delegation that only makes sense for one population.

        Raises rather than silently answering for ``populations[0]``: picking an
        arbitrary population would fail open, which is the shape of #1271.
        """
        if len(self.populations) != 1:
            raise NotImplementedError(
                f"{type(self).__name__}.{what} delegates to a single inner SED, but this "
                f"ForwardModel holds {len(self.populations)} populations. Call it on the "
                f"population you mean (e.g. forward.populations[i].sed.{what})."
            )
        return self._inner_sed_for_delegation()

    def _has_line_catalog(self) -> bool:
        """Whether the nebular backend publishes a discrete line catalog.

        ``loss_functions`` calls this whenever the observation carries line
        fluxes, to choose between *predicting* lines (photoionization backends)
        and *measuring* them off the spectrum (wNE / shock).
        """
        fn = getattr(self._single_inner_sed("_has_line_catalog"), "_has_line_catalog", None)
        return bool(fn()) if callable(fn) else False

    @property
    def available_properties(self):
        """Derived-property catalog of the inner SED. Delegated (#1300)."""
        return self._single_inner_sed("available_properties").available_properties

    def _supports_jit_threading(self) -> bool:
        """Whether the threaded (``data_args``) forward is valid for this topology.

        The threaded forward, ``predict_observables_jit`` / ``predict_state`` with
        ``ssp_data`` and ``template_data`` passed in, is written for a plain
        single-population SED forward. On a hierarchical
        (:class:`PopulationSEDModel`-wrapped) forward it mis-broadcasts the galaxy
        axis against the SFH grid (``mul got incompatible shapes (256,), (3,)``).

        Hierarchical and multi-population forwards were previously excluded by
        *accident*: ``Fitter._build_data_args`` read ``model.ssp_data``, which did
        not exist, and a ``contextlib.suppress`` swallowed the ``AttributeError``.
        Adding that delegation turned threading on for topologies it does not
        support. This states the exclusion instead of relying on a missing
        attribute, and keeps the topology test next to the other topology guards.

        Returns
        -------
        bool
            True only for a single, non-spatial, non-``PopulationSEDModel``
            population.
        """
        if len(self.populations) != 1 or self.populations[0].spatial is not None:
            return False
        sub = self.populations[0].sed
        # A PopulationSEDModel wraps its template as ``.sed``; a plain SEDModel
        # does not, so this distinguishes hierarchical from single-galaxy.
        return getattr(sub, "sed", sub) is sub

    @property
    def ssp_data(self):
        """SSP grid of the inner SED. Delegated.

        A **property**, not one of the auto-installed method delegations: callers
        read ``model.ssp_data`` as an attribute, so installing it via
        :func:`_install_inner_sed_delegations` would hand them a bound method and
        thread that instead of the grid.

        Load-bearing for JIT data threading. ``Fitter._build_data_args`` reads this
        to populate ``data_args["_jit_inputs"]``, and did so inside a
        ``contextlib.suppress(AttributeError, TypeError)``. While this property was
        missing, that read raised, the suppress swallowed it, and the whole
        ``_jit_inputs`` assignment was skipped, so the SSP grid closure-captured
        into every compiled loss on the *canonical* inference surface. On a real
        grid (15x93x5994 float64) it inlined twice as hex, 267.6 MB of a 274.6 MB
        program, and XLA compilation was OOM-killed. Guarded by
        ``tests/contract/test_loss_ssp_threading.py``.
        """
        return self._single_inner_sed("ssp_data").ssp_data

    @property
    def hybrid(self):
        """Hybrid kernel container delegated from the inner SED."""
        return getattr(self._inner_sed_for_delegation(), "hybrid", None)

    @property
    def z_fixed(self):
        """Fixed redshift, or ``None``, delegated from the inner SED."""
        return getattr(self._inner_sed_for_delegation(), "z_fixed", None)

    @property
    def dl_cm_fixed(self):
        """Fixed luminosity distance [cm], or ``None``, delegated from the inner SED."""
        return getattr(self._inner_sed_for_delegation(), "dl_cm_fixed", None)

    @property
    def n_grid(self):
        """PSD-grid resolution, or ``0``, delegated from the inner SED."""
        return getattr(self._inner_sed_for_delegation(), "n_grid", 0)

    @property
    def uses_stochastic_sfh(self) -> bool:
        """Stochastic-SFH flag delegated from the inner SED."""
        return bool(getattr(self._inner_sed_for_delegation(), "uses_stochastic_sfh", False))

    # ── Explicit delegates (Migration 2 step 4) ──────────────────────
    # These methods/properties forward to the first population's inner
    # SED so callers (Fitter, loss_functions, jit_engine, hierarchical,
    # standardized) don't rely on the legacy ``__getattr__`` fall-through.
    # Single-population semantics; multi-population fits use the
    # namespace machinery in :meth:`predict` directly.

    @property
    def wavelengths(self):
        """Rest-frame wavelength grid delegated from the inner SED."""
        return self._inner_sed_for_delegation().wavelengths

    def compile_signature(self):
        """Structural signature of the inner SED (used for JIT cache keys)."""
        return self._inner_sed_for_delegation().compile_signature()

    def predict_spectrum(
        self, params, wave_obs=None, wave_chunk_size=None, *, ssp_data=None, template_data=None
    ):
        """Channel-specific prediction: ``spec_fnu`` extracted from :meth:`predict_observables`.

        Symmetric with :meth:`predict_photometry`. Routing the default
        (no explicit ``wave_obs``/``wave_chunk_size``) spectrum call through
        :meth:`predict_observables` keeps the **standardized forward seam**:
        single-galaxy fits return shape ``(n_pix,)`` and hierarchical fits
        (:class:`PopulationSEDModel`) return ``(N_gal, n_pix)``, the per-galaxy
        batching and the spectrum projection are vmapped *below* this method
        (see :func:`_predict_observation`). The previous delegation to the inner
        *scalar* SED bypassed the population vmap, so stacked per-galaxy SFH
        params collided with the age grid (suchethac/tengri#711, Gap 2).

        An explicit ``wave_obs``/``wave_chunk_size`` (interactive plotting on a
        custom grid) still delegates to the inner SED, which evaluates the SED
        on the requested grid (suchethac/tengri#707).

        ``ssp_data``/``template_data`` are the JIT-threading channel, see
        :meth:`SEDModel.predict_photometry` (#1753).
        """
        if wave_obs is None and wave_chunk_size is None:
            pred = self.predict_observables(params, ssp_data=ssp_data, template_data=template_data)
            for key in ("spec_fnu", "spec_obs"):
                if key in pred:
                    return pred[key]
            raise KeyError(
                f"predict_spectrum: no spectroscopic channel in prediction dict "
                f"(saw keys: {list(pred)})"
            )
        return self._inner_sed_for_delegation().predict_spectrum(
            params,
            wave_obs=wave_obs,
            wave_chunk_size=wave_chunk_size,
            ssp_data=ssp_data,
            template_data=template_data,
        )

    def predict_photometry_components(self, params):
        """Delegate to :meth:`SEDModel.predict_photometry_components` on the inner SED."""
        return self._inner_sed_for_delegation()._photometry_via_state(params)

    def predict_spectrum_components(self, params, wave_obs=None):
        """Delegate to :meth:`SEDModel.predict_spectrum_components` on the inner SED."""
        return self._inner_sed_for_delegation()._spectrum_via_state(params, wave_obs=wave_obs)

    def predict_line_fluxes(self, params, target_wavelengths=None, **kwargs):
        """Delegate to :meth:`SEDModel.predict_line_fluxes` on the inner SED.

        Forwards ``**kwargs`` (``tolerance_aa``, the shared-forward ``state=``)
        so the joint-loss fast path works on the ForwardModel path.
        """
        return self._inner_sed_for_delegation().predict_line_fluxes(
            params, target_wavelengths=target_wavelengths, **kwargs
        )

    def predict_line_ratios(self, params, line_ratio_data, **kwargs):
        """Delegate to :meth:`SEDModel.predict_line_ratios` on the inner SED."""
        return self._inner_sed_for_delegation().predict_line_ratios(
            params, line_ratio_data, **kwargs
        )

    def predict_spectral_indices(self, params, index_defs, **kwargs):
        """Delegate to :meth:`SEDModel.predict_spectral_indices` on the inner SED."""
        return self._inner_sed_for_delegation().predict_spectral_indices(
            params, index_defs, **kwargs
        )

    def predict_state(
        self,
        params,
        fixed_values=None,
        ssp_data=None,
        template_data=None,
        *,
        observables_only=False,
    ):
        """Delegate to :meth:`SEDModel.predict_state` on the inner SED.

        The three optional positional arguments are the **JIT-threading**
        channel: when the loss builder has them, it passes the SSP grid and
        template arrays in rather than letting the forward close over them,
        which keeps them out of the compiled program as constants.
        ``observables_only`` is the publication-shortcut opt-in; see
        :meth:`SEDModel.predict_state`. Defaulting it to ``False`` here means a
        delegated call cannot lose a published SED by omission (#1673).

        This signature must track :meth:`SEDModel.predict_state`. It previously took
        ``params`` only, so the threaded feature-channel call raised ``TypeError``
        here, invisible, because the caller only reached this branch when
        ``_jit_inputs`` was populated, and a missing ``ssp_data`` delegation meant it
        never was. Two omissions masking each other.
        """
        return self._inner_sed_for_delegation().predict_state(
            params,
            fixed_values=fixed_values,
            ssp_data=ssp_data,
            template_data=template_data,
            observables_only=observables_only,
        )

    def predict_derived(self, params):
        """Delegate to :meth:`SEDModel.predict_derived` on the inner SED."""
        return self._inner_sed_for_delegation()._predict_derived(params)

    def predict_sfh(self, params, *args, **kwargs):
        """Delegate to :meth:`SEDModel.predict_sfh` on the inner SED."""
        return self._inner_sed_for_delegation().predict_sfh(params, *args, **kwargs)

    def predict_sfh_quantities(self, params, *args, **kwargs):
        """Delegate to :meth:`SEDModel.predict_sfh_quantities` on the inner SED."""
        return self._inner_sed_for_delegation()._predict_sfh_quantities(params, *args, **kwargs)

    def predict_rest_sed(self, params, *args, **kwargs):
        """Delegate to :meth:`SEDModel.predict_rest_sed` on the inner SED."""
        return self._inner_sed_for_delegation()._predict_rest_sed(params, *args, **kwargs)

    def predict_magnitudes(self, params):
        """Delegate to :meth:`SEDModel.predict_magnitudes` on the inner SED."""
        return self._inner_sed_for_delegation()._predict_magnitudes(params)

    def xi_to_params(self, xi):
        """Delegate to :meth:`SEDModel.xi_to_params` on the inner SED."""
        return self._inner_sed_for_delegation().xi_to_params(xi)

    def predict_photometry(self, params, *, ssp_data=None, template_data=None):
        """Channel-specific prediction: ``phot_fnu`` extracted from :meth:`predict_observables`.

        Single-galaxy fits return shape ``(n_filters,)``; hierarchical
        fits (PopulationSEDModel) return shape ``(N_gal, n_filters)``.
        The Fitter's legacy loss_fn calls this directly; providing it
        on ForwardModel ensures the batched output flows through rather
        than reaching for the inner scalar SED.

        Parameters
        ----------
        params: Mapping
            Free-parameter dict.
        ssp_data, template_data: Any | None, keyword-only, optional
            The JIT-threading channel, see
            :meth:`SEDModel.predict_photometry`. Pass these only when wrapping
            this method in your own JAX transform; ``None`` (default) uses the
            model's own arrays, which is correct for every ordinary call.
        """
        pred = self.predict_observables(params, ssp_data=ssp_data, template_data=template_data)
        for key in ("phot_fnu", "fnu_obs"):
            if key in pred:
                return pred[key]
        raise KeyError(
            f"predict_photometry: no photometric channel in prediction dict "
            f"(saw keys: {list(pred)})"
        )

    @classmethod
    def build(
        cls,
        *,
        sed: Any | None = None,
        spatial: Any | None = None,
        population: Any | None = None,
        populations: Iterable[Population] | None = None,
        mode: str | None = None,
        observation: Any | None = None,
        approx: Any | None = None,
    ) -> ForwardModel:
        """Construct a :class:`ForwardModel`.

        Three forms:

        - **Single-population sugar (the common case):** pass
          ``sed=<SEDModel>`` and ``observation=<Observation>``,
          optionally ``spatial=<SpatialModel>``. They are wrapped into
          a one-element ``populations`` tuple with ``name="default"``.
        - **Hierarchical population:** pass
          ``population=<PopulationSEDModel>`` for a hierarchical fit
          across many galaxies sharing some parameters (e.g. PSD).
          The PopulationSEDModel holds the SED template, the per-galaxy
          data, the names of shared parameters, and their priors.
          Inference goes through the standard
          ``Fitter(forward, ...).run('vi')`` path; the Fitter routes
          to the hierarchical machinery when it detects a
          PopulationSEDModel.
        - **Multi-population explicit:** pass ``populations=[...]``
          for galaxy decompositions (AGN + bulge + disc, ADR-0012).
          Names must be unique and must not contain ``.``.

        Parameters
        ----------
        sed: SEDModel or SubModel, optional
            Single-population shortcut. Mutually exclusive with
            ``population`` and ``populations``.
        spatial: SpatialModel or SubModel, optional
            Single-population spatial side. Only valid when ``sed=``
            is also given.
        population: PopulationSEDModel or SubModel, optional
            Hierarchical-population shortcut. Mutually exclusive with
            ``sed`` and ``populations``. The PopulationSEDModel is held
            inside ``Population(name="default", sed=population)``,             the outer-shell
            signature stays uniform.
        populations: iterable of Population, optional
            Explicit population list for galaxy decompositions.
            Mutually exclusive with ``sed`` and ``population``.
        observation: object, optional
            Observation model. Inherited from ``sed`` when omitted.
        approx: WavePrecomp or SpectrumPrecomp or FeaturePrecomp or tuple, optional
            LUT policy, built against the **authoritative** observation
            (spec §5, #1367). Same grammar as ``SEDModel.build(approx=...)``.
            Reuse-on-match: a sed already carrying exactly this LUT against
            the same filters is used as-is; a different-filter LUT is rebuilt
            (superseding the mismatch guard below, the rebuild is the
            guard's own suggested remedy); no LUT → built. Only valid with
            the ``sed=`` form; ``None`` (default) leaves the sed untouched.

        Returns
        -------
        ForwardModel

        Raises
        ------
        ValueError
            If construction doesn't get exactly one of ``sed``,
            ``population``, or ``populations``; if ``spatial=`` is
            given without ``sed=``; or if ``populations`` contains
            duplicate names.
        """
        if spatial is not None and sed is None:
            raise ValueError(
                "ForwardModel.build(spatial=...) requires sed=... too. "
                "Use populations=[...] for explicit pairing."
            )
        if approx is not None and sed is None:
            raise ValueError(
                "ForwardModel.build(approx=...) applies to the single-sed form "
                "(sed=...). For populations=[...] / population=..., build each "
                "SEDModel with its own approx= (SEDModel.build(..., approx=...))."
            )
        # Resolve observation: inherit from sed if omitted
        if observation is None:
            if sed is None:
                raise TypeError(
                    "ForwardModel.build(populations=...)/(population=...) requires "
                    "observation=... explicitly (no single sed to inherit from)."
                )
            observation = getattr(sed, "observation", None)
            if observation is None:
                raise TypeError("ForwardModel.build needs observation=... (the sed carries none).")
        elif sed is not None:
            sed_obs = getattr(sed, "observation", None)
            if (
                approx is None
                and sed_obs is not None
                and _filters_fingerprint(sed_obs) != _filters_fingerprint(observation)
                and getattr(sed, "_approx", {}).get("wave_precomp")
            ):
                raise ValueError(
                    "This sed carries a WavePrecomp LUT integrated against different "
                    "filters than the observation passed to ForwardModel.build, its "
                    "photometry would be silently wrong (#1315). Rebuild the sed with "
                    "this observation, or build it without approx=, or pass approx= "
                    "here to rebuild the LUT against this observation (#1367)."
                )

        provided = sum(x is not None for x in (sed, population, populations))
        if provided != 1:
            raise ValueError(
                "ForwardModel.build needs exactly one of sed=..., "
                "population=..., or populations=..."
            )

        if sed is not None:
            if approx is not None:
                sed = _resolve_sed_approx(sed, approx, observation)
            pops = (Population(name="default", sed=sed, spatial=spatial),)
        elif population is not None:
            pops = (Population(name="default", sed=population),)
        else:
            assert populations is not None
            pops = tuple(populations)

        if len(pops) == 0:
            raise ValueError("ForwardModel needs at least one population.")
        names = [p.name for p in pops]
        if len(set(names)) != len(names):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(
                f"ForwardModel.build: population names must be distinct; "
                f"got duplicates {duplicates}."
            )

        # Validate and infer mode
        valid_modes = ("single", "multi_population", "hierarchical")
        if mode is not None and mode not in valid_modes:
            raise ValueError(f"mode={mode!r} is not valid. Choose from: {', '.join(valid_modes)}")

        # Infer mode based on which kwargs were provided
        if populations is not None:
            inferred_mode = "multi_population"
        elif population is not None:
            inferred_mode = "hierarchical"
        else:
            inferred_mode = "single"

        # Check for hierarchical mode (reserved for #1319)
        if mode == "hierarchical" and population is None:
            raise NotImplementedError(
                "mode='hierarchical' requires the #1319 shared= parameter "
                "(not yet implemented). See issue #1319."
            )

        # Validate mode assertion
        if mode is not None and mode != inferred_mode:
            if mode == "multi_population":
                raise ValueError(
                    f"mode='multi_population' requires populations=... "
                    f"(got {['sed=', 'population='][population is not None]}); "
                    f"inferred mode would be {inferred_mode!r}"
                )
            else:
                needed = {
                    "single": "sed=",
                    "hierarchical": "population=",
                }[mode]
                got_kwarg = ["sed=", "population=", "populations="][
                    [sed is not None, population is not None, populations is not None].index(True)
                ]
                raise ValueError(
                    f"mode={mode!r} requires {needed} (got {got_kwarg}); "
                    f"inferred mode would be {inferred_mode!r}"
                )

        return cls(populations=pops, observation=observation, mode=inferred_mode)

    def predict(self, params: Mapping[str, Any]) -> Any:
        """Lazy :class:`~tengri.forward.prediction.Prediction` for derived quantities.

        Symmetric with :meth:`SEDModel.predict`. Delegates to the inner
        SED's ``predict``, so the rich attribute groups
        (``.sfh``, ``.sed``, ``.lines``, ``.radio``, ``.xray``,
        ``.ionizing``, ``.photometry()``, ``.spectrum()``, ``.magnitudes()``)
        are available directly on a ``ForwardModel``.

        For the JIT-safe channel dict consumed by the inference path
        (loss functions, ``Observation.predict_summed``,
        ``fiber_spectroscopy``), use :meth:`predict_observables`
        instead, that's the path the Fitter dispatches into.

        Parameters
        ----------
        params: mapping of str -> array
            Free parameter values. Single-population fits use bare
            names (``"sfh_dpl_alpha"``).

        Returns
        -------
        Prediction
            Lazy caching wrapper with ``.sfh``, ``.sed``, ``.lines``,
            ``.radio``, ``.xray``, ``.ionizing``, ``.photometry()``,
            ``.spectrum``, and ``.magnitudes`` property groups.

        Raises
        ------
        NotImplementedError
            For multi-population forward models. ``Prediction`` is a
            single-population shape; explicit galaxy decompositions
            (AGN + bulge + disc) should iterate populations and call
            ``.predict`` on each, or use :meth:`predict_observables`
            for the combined channel dict.

        Notes
        -----
        **JIT-compatible**: no, :class:`Prediction` uses Python-side
        caching. Use :meth:`predict_observables` inside JIT.
        """
        if len(self.populations) != 1:
            raise NotImplementedError(
                "ForwardModel.predict() (lazy Prediction) is single-population only. "
                "Multi-population galaxy decompositions should call .predict() on "
                "each population's inner SED, or use .predict_observables() for the "
                "combined channel dict."
            )
        sub = self.populations[0].sed
        inner = getattr(sub, "sed", sub)
        if not hasattr(inner, "predict"):
            raise AttributeError(
                f"Inner population SED ({type(inner).__name__}) does not implement "
                "predict(); cannot produce a Prediction. Use .predict_observables() "
                "for the channel dict instead."
            )
        return inner.predict(params)

    def predict_observables(
        self,
        params: Mapping[str, jnp.ndarray],
        *,
        ssp_data=None,
        template_data=None,
    ) -> Mapping[str, jnp.ndarray]:
        """Predicted observables for the given parameters.

        Per-population orchestration (architecture spec §9.1):

        1. For each population, slice ``params`` by namespace
           (``"<name>.<key>"`` → ``"<key>"``; bare names like
           ``redshift`` pass through to all populations).
        2. Merge the population's fixed-parameter values.
        3. Run ``pop.sed.run(state, full_params)`` then
           ``pop.spatial.run(state, full_params)`` if present.
        4. Hand all per-population states to the observation.

        For a single-population fit, the observation receives the one
        state via ``observation.predict(state, params)``. For
        multi-population, the observation receives the dict of states
        via ``observation.predict_summed(per_pop_states, params)``,         falling back to a
        default sum if the observation does not
        provide ``predict_summed``.

        Cross-population reads (a component in one population reading
        a derived key published by another) are supported by copying
        each population's typed derived bundle into every other
        population's state under the namespaced key
        (``"agn.L_bolometric"``, etc.) before the next sub-model runs.

        Parameters
        ----------
        params: mapping of str -> array
            Free parameter values. Single-population fits use bare
            names (``"sfh_dpl_alpha"``); multi-population fits use
            namespaced names (``"disc.sfh_dpl_alpha"``).

        Returns
        -------
        mapping of str -> array
            Prediction dict, keyed by observation channel.
        """
        # Fast path (issue #281, predict-half): a single-population,
        # photometry-only model built with a WavePrecomp LUT can serve its
        # photometry straight from the inner SEDModel's LUT-aware orchestrator
        # (``predict_observables_jit``), skipping the full-resolution component
        # cube that the general per-population path below builds. That cube,         # not the LUT
        # projection, is the ~11-16x cost on plain photometry, so
        # without this the LUT never helps ``predict_observables`` (the fit path
        # already routes through ``predict_photometry`` and was unaffected).
        # Tightly guarded: multi-population, spatial, spectroscopy/joint,
        # hierarchical (PopulationSEDModel), and exact (non-LUT) models all fall
        # through to the unchanged orchestration.
        if len(self.populations) == 1 and self.populations[0].spatial is None:
            only = self.populations[0].sed
            obs = self.observation
            if (
                obs is not None
                and getattr(obs, "can_do_photometry", False)
                and not getattr(obs, "can_do_spectroscopy", False)
                and not hasattr(only, "sed")  # plain SEDModel, not a PopulationSEDModel
                and hasattr(only, "predict_observables_jit")
                and bool(getattr(only, "_approx", {}).get("wave_precomp"))
            ):
                return dict(
                    only.predict_observables_jit(
                        params, ssp_data=ssp_data, template_data=template_data
                    )._asdict()
                )

        is_multipop = len(self.populations) > 1

        # ── Pass 1: run each population's SED + Spatial, collect states.
        per_pop_states: dict[str, ForwardState] = {}
        per_pop_derived: dict[str, dict[str, Any]] = {}
        per_pop_params: dict[str, dict[str, Any]] = {}
        # The JIT-threading channel reaches ``run`` only on the topologies the
        # threaded forward is actually written for. ``_supports_jit_threading``
        # is that predicate already, single, non-spatial, non-PopulationSEDModel,
        # so ask it rather than restating the condition here. Restating it is
        # how the loss path acquired the bug its docstring records: hierarchical
        # forwards were excluded only by a swallowed AttributeError, and threading
        # silently turned on for a topology that mis-broadcasts the galaxy axis
        # against the SFH grid (see tests/contract/test_loss_ssp_threading.py).
        # Passing nothing keeps the call shape byte-for-byte what it was.
        threaded = {}
        if (ssp_data is not None or template_data is not None) and self._supports_jit_threading():
            threaded = {"ssp_data": ssp_data, "template_data": template_data}

        for pop in self.populations:
            full_params = self._params_for_population(params, pop)
            per_pop_params[pop.name] = full_params
            init_state = ForwardState(wave=jnp.zeros(1))
            state = pop.sed.run(init_state, full_params, **threaded)
            if pop.spatial is not None:
                state = pop.spatial.run(state, full_params)
            per_pop_states[pop.name] = state
            # Snapshot this population's derived bundle for cross-pop reads.
            per_pop_derived[pop.name] = dict(state.derived)

        # ── Pass 2 (multi-pop only): inject every other population's
        # namespaced derived keys into each state's derived bundle so
        # downstream observation models can read e.g. ``agn.L_bolometric``.
        if is_multipop:
            namespaced_extras: dict[str, Any] = {}
            for pop_name, pop_derived in per_pop_derived.items():
                for key, value in pop_derived.items():
                    namespaced_extras[f"{pop_name}.{key}"] = value
            for name, state in list(per_pop_states.items()):
                # Use _extras for namespaced keys (typed bundle rejects them).
                merged_extras = {**state.derived._extras, **namespaced_extras}
                new_derived = DerivedState(
                    **{
                        field: getattr(state.derived, field)
                        for field in DerivedState.field_names()
                        if field != "_extras"
                    },
                    _extras=merged_extras,
                )
                per_pop_states[name] = state.with_(derived=new_derived)

        # ── Pass 3: hand to observation.
        if not is_multipop:
            (only_state,) = per_pop_states.values()
            (only_params,) = per_pop_params.values()
            (only_pop,) = self.populations
            return _predict_observation(self.observation, only_pop.sed, only_state, only_params)

        if hasattr(self.observation, "predict_summed"):
            return self.observation.predict_summed(per_pop_states, per_pop_params)

        # Fallback: synthesize predict_summed by summing per-population
        # observation.predict outputs in linear flux, key-by-key.
        per_pop_pred = {
            name: self.observation.predict(state, per_pop_params[name])
            for name, state in per_pop_states.items()
        }
        return _linear_flux_sum(per_pop_pred)

    @property
    def approx(self):
        """The effective approximation state of the wrapped SED.

        Delegates to :attr:`SEDModel.approx`, so the question "is a LUT live?"
        has one spelling and one answer whether it is asked of a
        :class:`~tengri.forward.sed_model.SEDModel` or of the
        :class:`ForwardModel` wrapping it. Reaching past this for an inner
        attribute (``model._approx``) reads a detail the wrapper does not carry
        and silently reports "exact".

        Returns
        -------
        ApproxState
            Frozen summary of the active LUTs; falsy for an exact model.
        """
        from tengri.forward.sed_model import ApproxState

        inner = self._inner_sed_for_delegation()
        state = getattr(inner, "approx", None)
        return state if isinstance(state, ApproxState) else ApproxState()

    @property
    def approx_configs(self) -> tuple:
        """The wrapped SED's active precompute **configs**, in ``approx=`` form.

        Delegates to :attr:`SEDModel.approx_configs` for the same reason
        :attr:`approx` does: one spelling, one answer, whether asked of the SED
        or of the wrapper. Lets a caller add a LUT family without discarding
        another's settings.

        Returns
        -------
        tuple
            Active config objects; empty for an exact model.
        """
        inner = self._inner_sed_for_delegation()
        configs = getattr(inner, "approx_configs", ())
        return tuple(configs)

    def _has_modern_approx(self) -> bool:
        """Whether the wrapped SED carries a build-time ``approx=`` LUT."""
        inner = self._inner_sed_for_delegation()
        fn = getattr(inner, "_has_modern_approx", None)
        return bool(fn()) if callable(fn) else False

    def with_approx(self, approx):
        """Return a copy of this forward model with a different ``approx`` policy.

        Clones the wrapped SED via :meth:`SEDModel.with_approx` and re-wraps it,
        preserving the observation. Only the common single-population, SED-only
        forward is cloned; multi-population, spatial, and hierarchical
        (PopulationSEDModel) forwards are returned unchanged, as is any request
        that resolves to a no-op.

        Parameters
        ----------
        approx: WavePrecomp or SpectrumPrecomp or FeaturePrecomp or tuple or None
            Approximation policy for the clone (same grammar as ``approx=`` on
            :class:`SEDModel`).

        Returns
        -------
        ForwardModel
            A new forward model on the requested approximation path, or ``self``
            when the request is a no-op or the topology is not a plain
            single-population SED forward.
        """
        if len(self.populations) != 1 or self.populations[0].spatial is not None:
            return self
        sub = self.populations[0].sed
        inner = getattr(sub, "sed", sub)
        if inner is not sub:
            # PopulationSEDModel-wrapped (hierarchical), leave unchanged.
            return self
        if not hasattr(inner, "with_approx"):
            return self
        new_inner = inner.with_approx(approx)
        if new_inner is inner:
            return self
        return ForwardModel.build(sed=new_inner, observation=self.observation)

    def fit(
        self,
        data: Any = None,
        noise: Any = None,
        method: str = DEFAULT_METHOD,
        *,
        approx: Any = "auto",
        key: Any = None,
        params: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """Run inference. Canonical convenience entry point.

        Equivalent to ``Fitter(self, data, noise, approx=approx).run(method,
        **kwargs)``, wires the standard inference pipeline through the
        :class:`ForwardModel` exactly as the architecture spec prescribes
        ('inference is always through ForwardModel', issue #211).

        Parameters
        ----------
        data: array_like or Data, optional
            Observed flux (photometry / spectroscopy) or a :class:`Data`
            record. Optional for hierarchical fits where the per-galaxy data
            lives on the :class:`PopulationSEDModel`.
        noise: array, optional
            1-sigma uncertainties matching ``data``. Must be ``None`` if
            ``data`` is a :class:`Data` record.
        method: str, default ``"vi"``
            Inference method. Any value accepted by
            :meth:`Fitter.run` (``"vi"``, ``"mcmc_nuts"``, ``"map"``,
            …).
        approx: {"auto", None} or WavePrecomp or SpectrumPrecomp or tuple, default ``"auto"``
            Approximation policy for the fit. ``"auto"`` (default) routes the
            fit through the fast precompute LUT selected by data type
            (``WavePrecomp`` for photometry, ``SpectrumPrecomp`` for
            spectroscopy/joint, plus ``FeaturePrecomp`` when emission lines are
            fit); ``None`` forces the exact wave-grid path; an explicit config
            (or tuple) overrides. Model **prediction** stays exact regardless,             only
            the fit is accelerated. The user's model object is left
            unchanged; the returned posterior references the fit clone.
        key: jax.random.PRNGKey, optional
            Inference seed.
        params: dict, optional
            Per-fit parameter override dict. Keys that name fixed parameters
            (not free) will override their values for this fit only; the model
            object is left unchanged. Useful for catalog-fitting with per-galaxy
            fixed values (e.g. ``params={"redshift": z_i}``). Keys must name
            valid parameters (raise ``ValueError`` if not); keys naming free
            parameters raise ``ValueError`` (you cannot pin a parameter being fit).
            Default ``None`` (no override).
        **kwargs: Any
            Forwarded to :meth:`Fitter.run` (e.g. ``prewarm=``, JIT-compile the
            loss/sampler/predict surface before the fit loop, default ``True``).

        Returns
        -------
        Posterior
            Same return as :meth:`Fitter.run`.

        Notes
        -----
        Prefer this entry over :meth:`SEDModel.fit` for new code.
        ``Fitter(forward, data, noise).run(method)`` remains the
        low-level path; ``forward.fit(...)`` is just the shortcut.
        """
        from tengri.inference.fitter import Fitter, split_fitter_kwargs
        from tengri.observation.data import Data as _Data

        # Constructor-owned kwargs (calibration_marginalize, likelihood, ...)
        # go to Fitter(...); the rest to run(), spec §7's fit-time flags (#1378).
        ctor_kwargs, kwargs = split_fitter_kwargs(kwargs)

        data_mask = None
        if isinstance(data, _Data):
            if noise is not None:
                raise TypeError(
                    "fit(Data, noise=...) is ambiguous: the Data "
                    "record already carries its uncertainties."
                )
            v = data.validate_against(self.observation)
            data_mask = v.censor
            if v.spec_flux is not None and v.flux is not None:
                # Joint: the Fitter takes ONE concatenated vector plus
                # ``data_type="joint"`` -- photometry first, then spectrum, the
                # order the joint likelihood splits on. Handing it
                # ``photometry=``/``spectrum=`` instead names ``SEDModel.fit``
                # parameters that ``Fitter`` does not have: they fell through to
                # ``run()`` while ``data=None`` tripped the constructor's
                # "requires data and noise" guard, so every joint Data record
                # raised. ``data_type`` cannot ride ``**kwargs`` here -- it is
                # in ``_FIT_SURFACE_MANAGED``, so ``split_fitter_kwargs`` keeps
                # it out of ``ctor_kwargs`` for the surface to set (#1366).
                data = jnp.concatenate([jnp.asarray(v.flux), jnp.asarray(v.spec_flux)])
                noise = jnp.concatenate([jnp.asarray(v.noise), jnp.asarray(v.spec_noise)])
                ctor_kwargs.setdefault("data_type", "joint")
            elif v.spec_flux is not None:
                data, noise = v.spec_flux, v.spec_noise
            else:
                data, noise = v.flux, v.noise
            if v.line_values is not None:
                # Route Data.lines through to the Fitter using schema wavelengths
                # and per-galaxy values. Create a temporary Observation with the
                # line_fluxes populated, keeping all wavelengths from schema.
                # The cache key uses wavelengths (constant across fits), not values,
                # so fits with the same schema reuse the compiled program.
                lines_schema = getattr(self.observation, "lines", None)
                if lines_schema is None:
                    raise ValueError(
                        "Data has lines but Observation.lines is not declared. "
                        "Declare which lines with lines=LineList.from_names([...]) "
                        "in the schema."
                    )
                # Extract wavelengths from schema for matching line names
                line_names = tuple(sorted(v.line_values.keys()))
                wavelengths = []
                for name in line_names:
                    # Find the wavelength in the schema
                    try:
                        idx = lines_schema.names.index(name)
                        wavelengths.append(float(lines_schema.wavelengths[idx]))
                    except (ValueError, AttributeError, TypeError) as e:
                        raise ValueError(
                            f"Line '{name}' in Data.lines not found in Observation.lines"
                        ) from e
                wavelengths_arr = jnp.asarray(wavelengths)

                # Build LineFluxData with schema wavelengths and per-galaxy values
                line_fluxes_arr = jnp.array([v.line_values[n][0] for n in line_names])
                line_errors_arr = jnp.array([v.line_values[n][1] for n in line_names])

                import dataclasses

                from tengri.observation.line_flux_data import LineFluxData

                # Censoring rides with the value it belongs to, so a limit
                # cannot drift out of alignment with its line. Reading only
                # [0] and [1] here is what silently turned a non-detection
                # into a measurement (#1460).
                markers = [
                    (v.line_values[n][2] if len(v.line_values[n]) > 2 else None)
                    for n in line_names
                ]
                upper = [m == "upper" for m in markers]
                lower = [m == "lower" for m in markers]

                # This rebuild replaces the schema's ``line_fluxes`` wholesale.
                # If the user declared limits there and supplied none here,
                # dropping them would be silent and would bias the fit, so say
                # so instead. The deprecation on ``Observation(line_fluxes=...)``
                # points at ``Data``, and limits have to make that trip too.
                declared = getattr(self.observation, "line_fluxes", None)
                declared_mask = getattr(declared, "limit_mask", None)
                if declared_mask is not None and not any(upper) and not any(lower):
                    flagged = [
                        nm
                        for nm, flag in zip(getattr(declared, "names", ()), declared_mask)
                        if float(flag) != 0.0
                    ]
                    raise ValueError(
                        f"Observation.line_fluxes marks {flagged} as censored "
                        "limits, but Data.lines supplies no limit markers, "
                        "those flags would be dropped and the lines fit as "
                        "detections. Pass them with the values instead: "
                        "Data(lines={'<line>': (flux, err, 'upper')})."
                    )

                temp_line_flux_data = LineFluxData(
                    names=line_names,
                    wavelengths=wavelengths_arr,
                    fluxes=line_fluxes_arr,
                    errors=line_errors_arr,
                    is_upper_limit=jnp.asarray(upper) if any(upper) else None,
                    is_lower_limit=jnp.asarray(lower) if any(lower) else None,
                )

                # Use dataclasses.replace to create a new Observation with the
                # LineFluxData, keeping all other fields (including schema wavelengths).
                obs_with_lines = dataclasses.replace(
                    self.observation, line_fluxes=temp_line_flux_data
                )

                # Create a ForwardModel wrapper that overrides only the observation
                # This avoids rebuilding the model, only changing the observation's data.
                fwd_with_lines = dataclasses.replace(self, observation=obs_with_lines)

                fitter = Fitter(
                    fwd_with_lines,
                    data=data,
                    noise=noise,
                    data_mask=data_mask,
                    approx=approx,
                    params_override=params,
                    **ctor_kwargs,
                )
                return fitter.run(method, key=key, **kwargs)

        fitter = Fitter(
            self,
            data=data,
            noise=noise,
            data_mask=data_mask,
            approx=approx,
            params_override=params,
            **ctor_kwargs,
        )
        return fitter.run(method, key=key, **kwargs)

    def prewarm(self, *, data_shape=None, method: str = "mcmc_nuts", **kwargs):
        """Pre-compile the JIT kernels for ``method`` against this model's shape.

        After this returns, a subsequent :meth:`fit` call with the same
        ``method`` skips XLA compilation. It still runs its own sampler warmup
        adaptation on the real data: the dummy (zeros) data used here would give a
        meaningless step size / mass matrix, so that adaptation is discarded and
        only the value-independent compile is kept.

        Parameters
        ----------
        data_shape: tuple of int or None, optional
            Shape of the data to pre-warm against. If ``None``, uses the
            observation's photometry shape. Dummy data (zeros) of this shape
            are created internally, pre-warm is value-independent and only
            needs the compile signature (shape, filters, wavelengths, etc.).
        method: str, default ``"mcmc_nuts"``
            Inference method to pre-warm. Any name accepted by
            :meth:`fit`.
        **kwargs
            Forwarded to the underlying :meth:`Fitter.prewarm` call
            (e.g. ``n_chains=4`` for multichain compilation).

        Returns
        -------
        None

        Notes
        -----
        Pre-warm is idempotent, a second call with the same method is a
        fast no-op, reusing the compiled kernels from the first call.
        The persistent XLA cache (``~/.cache/tengri_jax_cache``) also
        captures the compile, so a fresh Python process sees a warm
        XLA cache too.

        See Also
        --------
        Fitter.prewarm: Low-level pre-warm interface

        Examples
        --------
        >>> fwd = ForwardModel.build(sed=sed, observation=obs)
        >>> fwd.prewarm(method="mcmc_nuts", n_chains=4)
        >>> posterior = fwd.fit(flux, flux_err, method="mcmc_nuts", n_chains=4)
        """
        from tengri.inference.fitter import Fitter

        # Determine the observation shape for dummy data
        obs = self.observation
        if data_shape is None:
            # Use photometry shape if available, else None (hierarchical fit)
            if obs is not None and hasattr(obs, "photometry") and obs.photometry is not None:
                n_filters = obs.photometry.n_filters
                data_shape = (n_filters,)
            else:
                data_shape = None

        # Without a data shape (e.g. a hierarchical model with no top-level
        # photometry) there is nothing to build a dummy Fitter against, passing
        # data=None would crash. Skip rather than raise: prewarm is a best-effort
        # optimization.
        if data_shape is None:
            return

        # Dummy data (zeros). The XLA compile is value-independent, but the
        # sampler *adaptation* (step size + mass matrix) is NOT, a warmup on
        # zeros produces geometry tuned to a meaningless posterior. We keep the
        # compile and DROP that adaptation below so a real fit re-adapts.
        data = jnp.zeros(data_shape)
        noise = jnp.ones(data_shape)

        # Build Fitter with dummy data and call its prewarm (compiles kernels
        # AND writes a per-model adaptation entry keyed structurally on shape).
        fitter = Fitter(self, data=data, noise=noise)
        fitter.prewarm(method=method, **kwargs)

        # Discard the zeros-data adaptation so a subsequent real fit with the same
        # shape does not silently reuse it (wrong mass matrix → degraded sampling,
        # divergence risk). The XLA compile persists in the JAX cache; only the
        # value-dependent adaptation half is dropped.
        try:
            from tengri.inference._model_cache import _default_owner

            _default_owner.get_or_compile_model(self).get("adaptation", {}).clear()
        except Exception:
            pass

    def _params_for_population(
        self,
        params: Mapping[str, Any],
        pop: Population,
    ) -> dict[str, Any]:
        """Build the parameter dict this population's SubModel sees.

        Strips the ``"<pop.name>."`` namespace from any namespaced
        params; passes bare names (no ``.``) through unchanged.
        Cross-population namespaced reads (e.g. ``"agn.L_bol"`` in a
        component's ``reads`` dict) are NOT stripped, they flow
        through to the component's ``apply``, which looks them up in
        ``state.derived`` after the cross-pop merge.

        Then merges the population's spec fixed values so the
        downstream projection has everything (redshift, calibration
        coefficients, etc.).
        """
        prefix = f"{pop.name}."
        sliced: dict[str, Any] = {}
        for k, v in params.items():
            if k.startswith(prefix):
                sliced[k[len(prefix) :]] = v
            elif "." not in k:
                sliced[k] = v

        full: dict[str, Any] = {}
        spec = getattr(pop.sed, "spec", None)
        if spec is not None and hasattr(spec, "get_fixed_values"):
            full.update(spec.get_fixed_values())
        full.update(sliced)
        return full


def _predict_observation(
    observation: Any,
    sub_model: Any,
    state: ForwardState,
    params: Mapping[str, Any],
) -> Mapping[str, jnp.ndarray]:
    """Run ``observation.predict``, vmapping over any axes the SubModel published.

    SubModels publish their batched axes via :attr:`batched_axes`
    (``{name: axis_position}``, default ``{}``). When non-empty, the
    SubModel's :meth:`run` has already returned a batched
    :class:`ForwardState` along those axes. This helper vmaps
    ``observation.predict`` once per batched axis so the predicted
    observables, the channel dict :math:`\\hat{\\mathbf{d}}` that
    enters the data term of the information Hamiltonian, carry
    matching leading axes.

    The shape-consistency contract: regardless of which SubModel is
    held, the prediction dict's keys and per-channel array shapes
    line up with whatever data array the likelihood is fed. For a
    single-galaxy fit, ``phot_fnu`` has shape ``(n_filters,)``; for
    a hierarchical fit with :class:`PopulationSEDModel`,
    ``phot_fnu`` has shape ``(N_gal, n_filters)`` and broadcasts
    against per-galaxy data and noise in the χ² term of
    :math:`\\mathcal{H}_{\\rm hier}` (see paper §4 hierarchical
    inference).

    Composability: ``observation.predict`` is the un-batched
    primitive. Callers wanting outer ``pmap`` / ``shard_map`` can
    wrap this helper (or just ``observation.predict``) themselves,     the hidden batching here is
    the default, not the only path.
    """
    import jax

    batched_axes = getattr(sub_model, "batched_axes", {}) or {}
    if not batched_axes:
        return observation.predict(state, params)

    # Vmap once per batched axis. ``params`` axes are inferred from
    # the SubModel's parameter_axes when available; otherwise broadcast.
    predict_fn = observation.predict
    if hasattr(sub_model, "parameter_axes"):
        params_axes = sub_model.parameter_axes(params)
    else:
        params_axes = {name: None for name in params}
    for axis_pos in batched_axes.values():
        predict_fn = jax.vmap(predict_fn, in_axes=(axis_pos, params_axes))
    return predict_fn(state, params)


def _linear_flux_sum(
    per_pop_pred: dict[str, Mapping[str, jnp.ndarray]],
) -> dict[str, jnp.ndarray]:
    """Sum a per-population prediction-dict in linear flux, key-by-key."""
    all_keys: set[str] = set()
    for pred in per_pop_pred.values():
        all_keys.update(pred.keys())
    summed: dict[str, jnp.ndarray] = {}
    for key in all_keys:
        contributions = [pred[key] for pred in per_pop_pred.values() if key in pred]
        if not contributions:
            continue
        total = contributions[0]
        for c in contributions[1:]:
            total = total + c
        summed[key] = total
    return summed


def _install_inner_sed_delegations() -> None:
    """Attach the method delegations named in ``ForwardModel._DELEGATED_TO_INNER_SED``.

    Generated from the tuple rather than written out thirteen times, so the
    declared list and the installed methods cannot drift apart, that drift is
    what produced #1300 in the first place. ``_has_line_catalog`` and
    ``available_properties`` are defined explicitly on the class (one coerces to
    bool, the other is a property) and are skipped here.
    """
    explicit = {"_has_line_catalog", "available_properties"}

    def _make(name: str):
        def _delegated(self, *args, **kwargs):
            return getattr(self._single_inner_sed(name), name)(*args, **kwargs)

        _delegated.__name__ = name
        _delegated.__qualname__ = f"ForwardModel.{name}"
        _delegated.__doc__ = (
            f"Delegated to the inner SEDModel's ``{name}`` (#1300).\n\n"
            "The inference stack calls this on ``fitter.model``, which is a\n"
            "ForwardModel on the canonical path. Raises for multi-population\n"
            "forwards rather than answering for an arbitrary population."
        )
        return _delegated

    for name in ForwardModel._DELEGATED_TO_INNER_SED:
        if name in explicit or name in vars(ForwardModel):
            continue
        setattr(ForwardModel, name, _make(name))


_install_inner_sed_delegations()
