# SPDX-License-Identifier: BSD-3-Clause
"""ForwardModel — the outer shell of the forward chain.

Owns a tuple of :class:`Population`s and an :class:`Observation`.
Exposes a single ``.predict(params)`` method that inference calls.
The architecture spec is at
``docs/dev/forward-model-architecture.md`` §5–§6.

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

from tengri.forward.population import Population
from tengri.protocols.component import ForwardState
from tengri.protocols.derived_state import DerivedState

__all__ = ["ForwardModel"]


@dataclass(frozen=True)
class ForwardModel:
    """The outer shell of the forward model.

    Holds populations + observation; exposes ``.predict(params)`` as
    the sole API inference consumes. Supports both single-population
    (the common case) and multi-population galaxy decompositions
    (AGN + bulge + disc, ADR-0012).

    Parameters
    ----------
    populations : tuple of Population
        One or more populations. Each carries an SED ``SubModel`` and
        optionally a spatial ``SubModel``. Names must be distinct.
    observation : object
        Observation model exposing ``predict(state, params) → dict``
        (single-population) and ``predict_summed(per_pop_states, params)``
        (multi-population).
    """

    populations: tuple[Population, ...]
    observation: Any

    # ── Legacy-SEDModel delegations (for Fitter consumption) ─────────
    # The Fitter inner machinery (loss_fn, JIT compile, posterior
    # warm-start) reaches into ~12 SEDModel attributes. These
    # properties delegate so a ForwardModel can stand in for an
    # SEDModel anywhere the Fitter needs it. For hierarchical fits,
    # the inner SubModel is a PopulationSEDModel whose own attributes
    # delegate to its template SEDModel — three-level chain.

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
        returns the *first* population's spec — those fits use
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

        Delegates to the first population's inner SED — single-population
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

    def predict_spectrum(self, params, wave_obs=None, wave_chunk_size=None):
        """Channel-specific prediction: ``spec_fnu`` extracted from :meth:`predict_observables`.

        Symmetric with :meth:`predict_photometry`. Routing the default
        (no explicit ``wave_obs``/``wave_chunk_size``) spectrum call through
        :meth:`predict_observables` keeps the **standardized forward seam**:
        single-galaxy fits return shape ``(n_pix,)`` and hierarchical fits
        (:class:`PopulationSEDModel`) return ``(N_gal, n_pix)`` — the per-galaxy
        batching and the spectrum projection are vmapped *below* this method
        (see :func:`_predict_observation`). The previous delegation to the inner
        *scalar* SED bypassed the population vmap, so stacked per-galaxy SFH
        params collided with the age grid (suchethac/tengri#711, Gap 2).

        An explicit ``wave_obs``/``wave_chunk_size`` (interactive plotting on a
        custom grid) still delegates to the inner SED, which evaluates the SED
        on the requested grid (suchethac/tengri#707).
        """
        if wave_obs is None and wave_chunk_size is None:
            pred = self.predict_observables(params)
            for key in ("spec_fnu", "spec_obs"):
                if key in pred:
                    return pred[key]
            raise KeyError(
                f"predict_spectrum: no spectroscopic channel in prediction dict "
                f"(saw keys: {list(pred)})"
            )
        return self._inner_sed_for_delegation().predict_spectrum(
            params, wave_obs=wave_obs, wave_chunk_size=wave_chunk_size
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

    def predict_state(self, params):
        """Delegate to :meth:`SEDModel.predict_state` on the inner SED."""
        return self._inner_sed_for_delegation().predict_state(params)

    def predict_derived(self, params):
        """Delegate to :meth:`SEDModel.predict_derived` on the inner SED."""
        return self._inner_sed_for_delegation().predict_derived(params)

    def predict_sfh(self, params, *args, **kwargs):
        """Delegate to :meth:`SEDModel.predict_sfh` on the inner SED."""
        return self._inner_sed_for_delegation().predict_sfh(params, *args, **kwargs)

    def predict_sfh_quantities(self, params, *args, **kwargs):
        """Delegate to :meth:`SEDModel.predict_sfh_quantities` on the inner SED."""
        return self._inner_sed_for_delegation().predict_sfh_quantities(params, *args, **kwargs)

    def predict_rest_sed(self, params, *args, **kwargs):
        """Delegate to :meth:`SEDModel.predict_rest_sed` on the inner SED."""
        return self._inner_sed_for_delegation().predict_rest_sed(params, *args, **kwargs)

    def predict_magnitudes(self, params):
        """Delegate to :meth:`SEDModel.predict_magnitudes` on the inner SED."""
        return self._inner_sed_for_delegation().predict_magnitudes(params)

    def xi_to_params(self, xi):
        """Delegate to :meth:`SEDModel.xi_to_params` on the inner SED."""
        return self._inner_sed_for_delegation().xi_to_params(xi)

    def ensure_photometry_precomputed(self) -> bool:
        """Lazily precompute photometry on the inner SED if not yet done.

        Delegates to :meth:`SEDModel.ensure_photometry_precomputed` on
        the first population's inner SED. Returns the inner method's
        result (``True`` if precomputation ran on this call). Multi-
        population galaxy decompositions iterate across populations.
        """
        ran = False
        for pop in self.populations:
            sub = pop.sed
            inner = getattr(sub, "sed", sub)
            fn = getattr(inner, "ensure_photometry_precomputed", None)
            if fn is not None:
                ran = bool(fn()) or ran
        return ran

    def predict_photometry(self, params):
        """Channel-specific prediction: ``phot_fnu`` extracted from :meth:`predict_observables`.

        Single-galaxy fits return shape ``(n_filters,)``; hierarchical
        fits (PopulationSEDModel) return shape ``(N_gal, n_filters)``.
        The Fitter's legacy loss_fn calls this directly; providing it
        on ForwardModel ensures the batched output flows through rather
        than reaching for the inner scalar SED.
        """
        pred = self.predict_observables(params)
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
        observation: Any,
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
          for galaxy decompositions (AGN + bulge + disc — ADR-0012).
          Names must be unique and must not contain ``.``.

        Parameters
        ----------
        sed : SEDModel or SubModel, optional
            Single-population shortcut. Mutually exclusive with
            ``population`` and ``populations``.
        spatial : SpatialModel or SubModel, optional
            Single-population spatial side. Only valid when ``sed=``
            is also given.
        population : PopulationSEDModel or SubModel, optional
            Hierarchical-population shortcut. Mutually exclusive with
            ``sed`` and ``populations``. The PopulationSEDModel is held
            inside ``Population(name="default", sed=population)`` —
            the outer-shell signature stays uniform.
        populations : iterable of Population, optional
            Explicit population list for galaxy decompositions.
            Mutually exclusive with ``sed`` and ``population``.
        observation : object
            Observation model.

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
        provided = sum(x is not None for x in (sed, population, populations))
        if provided != 1:
            raise ValueError(
                "ForwardModel.build needs exactly one of sed=..., "
                "population=..., or populations=..."
            )

        if sed is not None:
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

        return cls(populations=pops, observation=observation)

    def predict(self, params: Mapping[str, Any]) -> Any:
        """Lazy :class:`~tengri.forward.prediction.Prediction` for derived quantities.

        Symmetric with :meth:`SEDModel.predict`. Delegates to the inner
        SED's ``predict``, so the rich attribute groups
        (``.sfh``, ``.sed``, ``.lines``, ``.radio``, ``.xray``,
        ``.ionizing``, ``.photometry``, ``.spectrum``, ``.magnitudes``)
        are available directly on a ``ForwardModel``.

        For the JIT-safe channel dict consumed by the inference path
        (loss functions, ``Observation.predict_summed``,
        ``fiber_spectroscopy``), use :meth:`predict_observables`
        instead — that's the path the Fitter dispatches into.

        Parameters
        ----------
        params : mapping of str -> array
            Free parameter values. Single-population fits use bare
            names (``"sfh_dpl_alpha"``).

        Returns
        -------
        Prediction
            Lazy caching wrapper with ``.sfh``, ``.sed``, ``.lines``,
            ``.radio``, ``.xray``, ``.ionizing``, ``.photometry``,
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
        **JIT-compatible**: no — :class:`Prediction` uses Python-side
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
        via ``observation.predict_summed(per_pop_states, params)`` —
        falling back to a default sum if the observation does not
        provide ``predict_summed``.

        Cross-population reads (a component in one population reading
        a derived key published by another) are supported by copying
        each population's typed derived bundle into every other
        population's state under the namespaced key
        (``"agn.L_bolometric"``, etc.) before the next sub-model runs.

        Parameters
        ----------
        params : mapping of str -> array
            Free parameter values. Single-population fits use bare
            names (``"sfh_dpl_alpha"``); multi-population fits use
            namespaced names (``"disc.sfh_dpl_alpha"``).

        Returns
        -------
        mapping of str -> array
            Prediction dict, keyed by observation channel.
        """
        is_multipop = len(self.populations) > 1

        # ── Pass 1: run each population's SED + Spatial, collect states.
        per_pop_states: dict[str, ForwardState] = {}
        per_pop_derived: dict[str, dict[str, Any]] = {}
        per_pop_params: dict[str, dict[str, Any]] = {}
        for pop in self.populations:
            full_params = self._params_for_population(params, pop)
            per_pop_params[pop.name] = full_params
            init_state = ForwardState(wave=jnp.zeros(1))
            state = pop.sed.run(init_state, full_params)
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

    def fit(
        self,
        data: Any = None,
        noise: Any = None,
        method: str = "vi",
        *,
        key: Any = None,
        **kwargs: Any,
    ):
        """Run inference. Canonical convenience entry point.

        Equivalent to ``Fitter(self, data, noise).run(method, **kwargs)``
        — wires the standard inference pipeline through the
        :class:`ForwardModel` exactly as the architecture spec
        prescribes ('inference is always through ForwardModel',
        issue #211).

        Parameters
        ----------
        data : array, optional
            Observed flux (photometry / spectroscopy). Optional for
            hierarchical fits where the per-galaxy data lives on the
            :class:`PopulationSEDModel`.
        noise : array, optional
            1-sigma uncertainties matching ``data``.
        method : str, default ``"vi"``
            Inference method. Any value accepted by
            :meth:`Fitter.run` (``"vi"``, ``"mcmc_nuts"``, ``"map"``,
            …).
        key : jax.random.PRNGKey, optional
            Inference seed.
        **kwargs : Any
            Forwarded to :meth:`Fitter.run`.

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
        from tengri.inference.fitter import Fitter

        fitter = Fitter(self, data=data, noise=noise)
        return fitter.run(method, key=key, **kwargs)

    def _params_for_population(
        self,
        params: Mapping[str, Any],
        pop: Population,
    ) -> dict[str, Any]:
        """Build the parameter dict this population's SubModel sees.

        Strips the ``"<pop.name>."`` namespace from any namespaced
        params; passes bare names (no ``.``) through unchanged.
        Cross-population namespaced reads (e.g. ``"agn.L_bol"`` in a
        component's ``reads`` dict) are NOT stripped — they flow
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
    observables — the channel dict :math:`\\hat{\\mathbf{d}}` that
    enters the data term of the information Hamiltonian — carry
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
    wrap this helper (or just ``observation.predict``) themselves —
    the hidden batching here is the default, not the only path.
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
