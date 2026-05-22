"""ForwardModel — the outer shell of the forward chain.

Owns a tuple of :class:`Population`s and an :class:`Observation`.
Exposes a single ``.predict(params)`` method that inference calls.
The architecture spec is at
``docs/dev/forward-model-architecture.md`` §5; this file implements
the tracer-bullet single-population slice.

Subsequent plans add:
  * Multi-population orchestration (ADR-0012)
  * Spatial submodel composition (spatial-model plan)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tengri.forward.population import Population
from tengri.protocols.component import ForwardState

__all__ = ["ForwardModel"]


@dataclass(frozen=True)
class ForwardModel:
    """The outer shell of the forward model.

    Holds populations + observation; exposes ``.predict(params)`` as
    the sole API inference consumes.

    Parameters
    ----------
    populations : tuple of Population
        At least one population; tracer-bullet enforces exactly one.
    observation : object
        Observation model exposing the existing
        ``Observation`` / ``ObservationModel`` API used by the SED
        forward pipeline.

    Notes
    -----
    Tracer-bullet limitations:
      * Single population only.
      * Spatial SubModels are not constructed (subsequent plan).
      * Parameter names are not namespaced (ADR-0012 plan).
    """

    populations: tuple[Population, ...]
    observation: Any

    @classmethod
    def build(
        cls,
        *,
        sed: Any | None = None,
        spatial: Any | None = None,
        populations: Iterable[Population] | None = None,
        observation: Any,
    ) -> ForwardModel:
        """Construct a :class:`ForwardModel`.

        Convenience entry point. Two forms:

        - **Single-population sugar (the common case):** pass
          ``sed=<SEDModel>`` and ``observation=<Observation>``. The
          SED is wrapped into a one-element ``populations`` tuple
          with ``name="default"``.
        - **Explicit populations:** pass ``populations=[...]``. Used
          once multi-population lands (ADR-0012). The tracer-bullet
          accepts the form but raises on >1 entry.

        Parameters
        ----------
        sed : SEDModel or SubModel, optional
            Single-population shortcut. Mutually exclusive with
            ``populations``.
        spatial : optional
            Reserved for the spatial-model plan. Raises if provided
            in the tracer-bullet.
        populations : iterable of Population, optional
            Explicit population list. Mutually exclusive with ``sed``.
        observation : object
            Observation model.

        Returns
        -------
        ForwardModel

        Raises
        ------
        ValueError
            If neither ``sed`` nor ``populations`` is given, or both.
        NotImplementedError
            If ``len(populations) > 1`` (ADR-0012) or if ``spatial``
            is provided (subsequent plan).
        """
        if spatial is not None:
            raise NotImplementedError(
                "ForwardModel.build(spatial=...) is reserved for the spatial-model plan."
            )
        if (sed is None) == (populations is None):
            raise ValueError("ForwardModel.build needs exactly one of sed=... or populations=...")

        if sed is not None:
            pops = (Population(name="default", sed=sed),)
        else:
            assert populations is not None
            pops = tuple(populations)

        if len(pops) > 1:
            raise NotImplementedError(
                "Multi-population ForwardModel is deferred to the ADR-0012 "
                "plan. This tracer-bullet ships single-population only."
            )
        if len(pops) == 0:
            raise ValueError("ForwardModel needs at least one population.")

        return cls(populations=pops, observation=observation)

    def predict(
        self,
        params: Mapping[str, jnp.ndarray],
    ) -> Mapping[str, jnp.ndarray]:
        """Predicted observables for the given parameters.

        Iterates over ``populations`` (per-population orchestration —
        architecture spec §9.1). For the tracer-bullet single-population
        case, the loop is a one-element loop and the result is returned
        directly.

        For each population, the SED ``SubModel`` produces a
        :class:`tengri.protocols.ForwardState`; the observation then
        projects that state into the channel dict
        (``{"phot_fnu": ...}``, ``{"spec_fnu": ...}``, joint, …) via
        :meth:`tengri.observation.Observation.predict`.

        Parameters
        ----------
        params : mapping of str -> array
            Free parameter values.

        Returns
        -------
        mapping of str -> array
            Prediction dict, keyed by observation channel. The keys
            depend on the observation's configuration (photometric,
            spectroscopic, or joint).
        """
        per_pop: dict[str, Mapping[str, jnp.ndarray]] = {}
        for pop in self.populations:
            if pop.spatial is not None:
                raise NotImplementedError(
                    "Population.spatial is reserved for the spatial-model plan."
                )
            # Merge the user's free-parameter values with the SubModel's
            # fixed-parameter values so the downstream projection has
            # everything it needs (redshift, calibration coefficients, etc.).
            # Free params override fixed ones when names collide — the user's
            # value wins. The legacy ``SEDModel.predict_photometry`` did this
            # merge internally; ``Observation.predict`` does not, so the
            # outer shell threads it.
            full_params: dict[str, Any] = {}
            spec = getattr(pop.sed, "spec", None)
            if spec is not None and hasattr(spec, "get_fixed_values"):
                full_params.update(spec.get_fixed_values())
            full_params.update(params)

            # The initial ForwardState is a placeholder — SED is the head of
            # the per-population chain and the SubModel's run() produces a
            # freshly-populated state. The 1-element wave is overwritten.
            init_state = ForwardState(wave=jnp.zeros(1))
            state = pop.sed.run(init_state, full_params)
            per_pop[pop.name] = self.observation.predict(state, full_params)

        if len(per_pop) == 1:
            (only,) = per_pop.values()
            return only
        raise NotImplementedError("Multi-population summing is the ADR-0012 plan, not this slice.")
