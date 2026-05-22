"""Population — one (SED, spatial) pair inside a :class:`ForwardModel`.

A galaxy decomposition (AGN point source + Sérsic bulge + exponential
disc) is expressed as multiple :class:`Population`s. The tracer-bullet
implementation supports single-population only; ADR-0012 will lift
this to multi-population with namespaced parameter names.

See ``docs/dev/forward-model-architecture.md`` §5 and ADR-0012.
"""

from __future__ import annotations

from dataclasses import dataclass

from tengri.protocols.submodel import SubModel

__all__ = ["Population"]


@dataclass(frozen=True)
class Population:
    """One (SED, spatial) pair inside a :class:`ForwardModel`.

    Parameters
    ----------
    name : str
        Population namespace. ``"default"`` for the convenience
        single-population path. Used by ADR-0012 multi-population
        parameter naming (not yet active in this slice).
    sed : SubModel
        SED SubModel for this population.
    spatial : SubModel or None, optional
        Spatial SubModel for this population. ``None`` for SED-only
        populations.
    """

    name: str
    sed: SubModel
    spatial: SubModel | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Population.name must be a non-empty string.")
