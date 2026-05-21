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

from tengri.forward._sed_submodel_adapter import _LegacySEDSubModel
from tengri.forward.population import Population

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
            # Local import to avoid an import cycle through
            # tengri.forward.sed_model -> tengri.forward.forward_model.
            from tengri.forward.sed_model import SEDModel

            sub = sed if not isinstance(sed, SEDModel) else _LegacySEDSubModel(sed)
            pops = (Population(name="default", sed=sub),)
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

        Tracer-bullet implementation: single-population only. Detailed
        implementation lands in Task 5.

        Raises
        ------
        NotImplementedError
            Always raised in this Task-4 skeleton. Task 5 fills in
            the actual prediction path.
        """
        raise NotImplementedError(
            "ForwardModel.predict is implemented in Task 5 of the tracer-bullet plan."
        )
