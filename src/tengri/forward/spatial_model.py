"""SpatialModel — SubModel composer over a list of :class:`SpatialComponent`s.

Mirror of :class:`tengri.forward.sed_model.SEDModel` on the spatial side,
at the SubModel layer. Holds a list of :class:`SpatialComponent`
instances and threads :class:`ForwardState` through them in
:meth:`run`.

A :class:`SpatialModel` is one of the three concrete sub-models that
:class:`tengri.ForwardModel` orchestrates per architecture spec §4 —
the others being ``SEDModel`` (already in place) and
``SpatialSEDModel`` (the joint composer, also in this file).

The wave-grid field of the incoming :class:`ForwardState` is preserved
as a placeholder; spatial sub-models operate on
``state.derived["spatial_grid_xy_kpc"]`` which the caller (typically
:class:`ForwardModel.predict`) inserts before running.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tengri.protocols.component import ForwardState, ParamDeclaration
from tengri.protocols.spatial import SpatialComponent

__all__ = ["SpatialModel", "SpatialSEDModel"]


@dataclass(frozen=True)
class SpatialModel:
    """Composer over a list of :class:`SpatialComponent`s.

    Satisfies :class:`tengri.protocols.SubModel` — has ``name``,
    :meth:`declared_parameters`, and :meth:`run`.

    Parameters
    ----------
    components : sequence of :class:`SpatialComponent`
        Spatial physics blocks to thread state through, in order.

    Notes
    -----
    JIT/grad/vmap-compatible: :meth:`run` is pure JAX.
    """

    components: tuple[SpatialComponent, ...]
    name: str = "spatial"

    def __init__(self, components: Sequence[SpatialComponent]) -> None:
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "name", "spatial")

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Aggregated parameter declarations across all spatial components."""
        decls: list[ParamDeclaration] = []
        for comp in self.components:
            decls.extend(comp.declared_parameters())
        return decls

    def run(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Thread ``state`` through each :attr:`components` in order.

        Reads :attr:`state.derived["spatial_grid_xy_kpc"]` — must be
        inserted by the caller before :meth:`run`. Each component
        consumes and updates :attr:`state.derived["spatial_profile_2d"]`.

        Parameters
        ----------
        state : ForwardState
            Incoming state. Must already carry the spatial grid.
        params : Mapping
            Free parameter values.

        Returns
        -------
        ForwardState
            New state with ``spatial_profile_2d`` populated.
        """
        for comp in self.components:
            state = comp.apply(state, params)
        return state


@dataclass(frozen=True)
class SpatialSEDModel:
    """The joint spatial+SED sub-model.

    Composes one :class:`tengri.SEDModel` and one :class:`SpatialModel`
    into a single :class:`tengri.protocols.SubModel`. SED runs first
    (head of the chain); spatial runs second so spatial components can
    optionally read SED-derived keys from ``state.derived``
    (architecture spec §4.3 — SED → Spatial ordering).

    Parameters
    ----------
    sed : :class:`tengri.SEDModel`
        SED sub-model (or anything satisfying :class:`SubModel`).
    spatial : :class:`SpatialModel`
        Spatial sub-model.

    Notes
    -----
    Composer-only — no physics of its own. Aggregates declared
    parameters from both halves; runs SED, then spatial, on the same
    threaded :class:`ForwardState`.

    JIT/grad/vmap-compatible.
    """

    sed: Any
    spatial: SpatialModel
    name: str = "spatial_sed"

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Union of SED and spatial declared parameters."""
        decls = list(self.sed.declared_parameters())
        decls.extend(self.spatial.declared_parameters())
        return decls

    def run(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Run SED first, then spatial. Pure JAX."""
        state = self.sed.run(state, params)
        return self.spatial.run(state, params)
