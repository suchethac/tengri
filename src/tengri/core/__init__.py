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

See ``~/.claude/plans/i-want-you-to-soft-torvalds.md`` Part II for the
full re-architecture proposal and Phase II-2 onwards.
"""

from __future__ import annotations

from tengri.core.component import (
    BARE_NAME_ALLOWLIST,
    ParamDeclaration,
    PipelineState,
    SEDComponent,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.core.likelihood import Likelihood
from tengri.core.observation import ObservationModel

__all__ = [
    "BARE_NAME_ALLOWLIST",
    "Likelihood",
    "ObservationModel",
    "ParamDeclaration",
    "PipelineState",
    "SEDComponent",
    "SEDComponentConfig",
    "SEDComponentState",
]
