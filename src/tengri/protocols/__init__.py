# SPDX-License-Identifier: BSD-3-Clause
"""Core protocols for the tengri SED forward-model pipeline (Part II-1).

This subpackage defines the **contracts** that physics components,
observation models, and likelihoods must implement so that future
phases can migrate :class:`tengri.SEDModel` from a hardcoded tier
dispatch (~2957 lines today) into a thin orchestrator over a list of
:class:`SEDComponent` objects.

This is intentionally a **scaffold**: nothing in `tengri` consumes
these protocols yet. The classes live here so:

1. Future component implementations have a single, stable interface
   to target (`StellarSEDComponent`, `DustSEDComponent`, …).
2. The contract is reviewable on its own, before any migration touches
   `forward/sed_model.py`.
3. A unit test asserts the protocol shape so accidental breaking
   changes during the migration are caught early.

See ``docs/dev/REFACTOR.md`` and the in-tree design plans for the
full re-architecture proposal.
"""

from __future__ import annotations

from tengri.protocols.component import (
    BARE_NAME_ALLOWLIST,
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    PipelineContractError,
    PipelineState,  # soft alias of ForwardState; removed in v1.0
    SEDComponent,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.protocols.derived_bundle import DerivedBundle
from tengri.protocols.likelihood import Likelihood
from tengri.protocols.observation import ObservationModel
from tengri.protocols.spatial import SpatialComponent
from tengri.protocols.submodel import SubModel

__all__ = [
    "BARE_NAME_ALLOWLIST",
    "DerivedBundle",
    "DerivedKey",
    "ForwardState",
    "Likelihood",
    "ObservationModel",
    "ParamDeclaration",
    "PipelineContractError",
    "PipelineState",
    "SEDComponent",
    "SEDComponentConfig",
    "SEDComponentState",
    "SpatialComponent",
    "SubModel",
]
