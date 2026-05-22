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

    def predict(
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
            return self.observation.predict(only_state, only_params)

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
