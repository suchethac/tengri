"""Population — a single SED SubModel + metadata tuple.

Tracer-bullet scope: holds one SubModel (SEDModel wrapped via
_LegacySEDSubModel). Multi-population composition is deferred to the
ADR-0012 plan.

See ``docs/dev/forward-model-architecture.md`` §5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["Population"]


@dataclass(frozen=True)
class Population:
    """A single population in a forward model.

    Holds a SubModel instance and a descriptive name. Tracer-bullet
    ships exactly one; multi-population lands in ADR-0012.

    Parameters
    ----------
    name : str
        Descriptive name (e.g. "default", "disk", "bulge").
    sed : SubModel
        Instance conforming to the SubModel Protocol
        (tengri.protocols.component.SubModel). Typically a
        _LegacySEDSubModel wrapping an SEDModel.
    """

    name: str
    sed: Any
