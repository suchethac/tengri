# SPDX-License-Identifier: BSD-3-Clause
"""Population, one (SED, spatial) pair inside a :class:`ForwardModel`.

A galaxy decomposition (AGN point source + Sérsic bulge + exponential
disc) is expressed as multiple :class:`Population` objects. The population's
``name`` is the outer namespace in parameter names, e.g.
``disc.sfh_dpl_alpha`` for the disc population's stellar
parameter. See architecture spec §6 + ADR-0012 for the full namespace
contract.

See ``docs/dev/archive/forward-model-architecture.md`` §5 and ADR-0012.
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
        Population namespace. Used as the outer prefix in parameter
        names: ``{name}.{prefix}_{param}`` (ADR-0012). Must be
        non-empty and must not contain ``.`` (the namespace separator).
        Convention: use ``"default"`` for single-population fits;
        ``"agn"``, ``"bulge"``, ``"disc"`` for galaxy decompositions.
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
        if "." in self.name:
            raise ValueError(
                f"Population.name {self.name!r} contains '.', which is reserved "
                f"as the parameter namespace separator (ADR-0012). Use "
                f"underscores or hyphens instead."
            )
