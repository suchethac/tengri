# SPDX-License-Identifier: BSD-3-Clause
"""Inter-component interfaces for the tengri SED forward model.

A `SEDModel` is a list of physics blocks (stellar, dust, nebular, AGN,
radio, X-ray, IGM, …) evaluated in order. Each block reads and writes
a typed bag of derived quantities: `ForwardState.derived`, a
:class:`DerivedState`; so a stellar block can publish ``L_uv`` for a
downstream dust block to read.

The objects defined here are the small, stable surface that physics
components target:

- :class:`SEDComponent`: the protocol every physics block satisfies.
- :class:`ForwardState`, :class:`DerivedState`: the typed bags that
  flow between components.
- :class:`DerivedKey`, :class:`ParamDeclaration`: the labels a
  component uses to declare its inputs, outputs, and free parameters.
- :class:`Likelihood`, :class:`ObservationModel`: analogous shapes
  for the observation layer.
- :class:`ComponentIOError` (alias :data:`PipelineContractError`,
  deprecated); raised when one block's declared inputs disagree with
  what an upstream block publishes.

See ``docs/architecture/`` and the in-tree ADRs for the wider design.
"""

from __future__ import annotations

from tengri.protocols.component import (
    BARE_NAME_ALLOWLIST,
    ComponentIOError,
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    PipelineContractError,  # deprecated alias of ComponentIOError; removed in v1.0
    PipelineState,  # soft alias of ForwardState; removed in v1.0
    SEDComponent,
    SEDComponentConfig,
    SEDComponentState,
)
from tengri.protocols.derived_state import DerivedState
from tengri.protocols.likelihood import Likelihood
from tengri.protocols.observation import ObservationModel
from tengri.protocols.spatial import SpatialComponent
from tengri.protocols.submodel import SubModel

__all__ = [
    "BARE_NAME_ALLOWLIST",
    "ComponentIOError",
    "DerivedKey",
    "DerivedState",
    "ForwardState",
    "Likelihood",
    "ObservationModel",
    "ParamDeclaration",
    "PipelineState",
    "SEDComponent",
    "SEDComponentConfig",
    "SEDComponentState",
    "SpatialComponent",
    "SubModel",
]


_RENAMED_SYMBOLS = {
    "DerivedBundle": ("DerivedState", "tengri.protocols.DerivedState"),
}


def __getattr__(name: str) -> object:
    if name in _RENAMED_SYMBOLS:
        new_name, new_path = _RENAMED_SYMBOLS[name]
        from tengri._deprecated import deprecated_attribute

        new_obj = globals()[new_name]
        return deprecated_attribute(
            new_obj, old_name=f"tengri.protocols.{name}", new_name=new_path
        )
    raise AttributeError(f"module 'tengri.protocols' has no attribute {name!r}")
