# SPDX-License-Identifier: BSD-3-Clause
"""SpatialComponent Protocol: contract for spatial-physics blocks.

Mirror of :class:`tengri.protocols.component.SEDComponent` on the spatial
side of the forward model. Each spatial component owns one piece of the
2D surface-brightness profile: a Sérsic envelope, an exponential disk,
a flat aperture, etc.: plus the parameters and precomputed tensors that
go with it.

See architecture spec ``docs/dev/archive/forward-model-architecture.md`` §3.2
for the astronomer-facing convenience base ``SpatialModelComponent``
that satisfies this Protocol with auto-discovery and a default apply().
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp

from tengri.protocols.component import (
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = [
    "SpatialComponent",
    "SpatialComponentConfig",
    "SpatialComponentState",
]


SpatialComponentConfig = SEDComponentConfig
SpatialComponentState = SEDComponentState


@runtime_checkable
class SpatialComponent(Protocol):
    """Contract for one block of the spatial forward model.

    Concrete subclasses (``Sersic``, ``Exponential``, ``FlatSlab``,
    eventually ``BulgeDisk`` and ``GPSpatialField``) live in
    :mod:`tengri.components.spatial.<name>`. They publish a 2D
    surface-brightness profile into :attr:`ForwardState.derived` under
    the ``"spatial_profile_2d"`` key.

    Required attributes mirror :class:`SEDComponent` exactly:

    - ``name: str``: stable identifier
    - ``parameter_prefix: str``: always ``"spatial_"``
    - ``config: SpatialComponentConfig``: frozen structural knobs

    Required methods are identical in shape to :class:`SEDComponent`:

    - ``declared_parameters()`` → list of :class:`ParamDeclaration`
    - ``precompute(...)`` → :class:`SpatialComponentState` (eager)
    - ``apply(state, params)`` → :class:`ForwardState` (pure JAX)

    """

    name: str
    parameter_prefix: str
    config: SpatialComponentConfig

    def declared_parameters(self) -> list[ParamDeclaration]: ...

    def precompute(self, **kwargs: Any) -> SpatialComponentState: ...

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState: ...
