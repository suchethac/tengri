# SPDX-License-Identifier: BSD-3-Clause
"""SpatialModel, SubModel composer over a list of :class:`SpatialComponent` objects.

Mirror of :class:`tengri.forward.sed_model.SEDModel` on the spatial side,
at the SubModel layer. Holds a list of :class:`SpatialComponent`
instances and threads :class:`ForwardState` through them in
:meth:`run`.

A :class:`SpatialModel` is one of the three concrete sub-models that
:class:`tengri.ForwardModel` orchestrates per architecture spec §4, the others being ``SEDModel``
(already in place) and
``SpatialSEDModel`` (the joint composer, also in this file).

The wave-grid field of the incoming :class:`ForwardState` is preserved
as a placeholder; spatial sub-models operate on
``state.derived["spatial_grid_xy_kpc"]`` which the caller (typically
:class:`ForwardModel.predict_observables`) inserts before running.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.protocols.component import ForwardState, ParamDeclaration
from tengri.protocols.spatial import SpatialComponent

__all__ = ["SpatialModel", "SpatialSEDModel", "default_grid_kpc"]


def default_grid_kpc(
    n: int = 64,
    extent_kpc: float = 10.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """A square ``(n, n)`` grid spanning ``[-extent_kpc, +extent_kpc]`` per axis.

    Returns ``(x_grid, y_grid)``, both 2D arrays of shape ``(n, n)``.
    Used as the default for :class:`SpatialModel` when no explicit grid
    is provided.

    Parameters
    ----------
    n: int, default 64
        Number of points per axis.
    extent_kpc: float, default 10.0
        Half-width of the grid in kpc; the full grid spans
        ``[-extent_kpc, +extent_kpc]`` in both x and y.

    Returns
    -------
    tuple[ndarray, ndarray]
        ``(x_grid, y_grid)`` of shape ``(n, n)`` each.
    """
    axis = jnp.linspace(-extent_kpc, extent_kpc, n)
    return jnp.meshgrid(axis, axis)


@dataclass(frozen=True)
class SpatialModel:
    """Composer over a list of :class:`SpatialComponent` objects.

    Satisfies :class:`tengri.protocols.SubModel`, has ``name``,
    :meth:`declared_parameters`, and :meth:`run`.

    Parameters
    ----------
    components: sequence of :class:`SpatialComponent`
        Spatial physics blocks to thread state through, in order.
    grid_kpc: tuple of (ndarray, ndarray), optional
        ``(x_grid_kpc, y_grid_kpc)``, the 2D spatial coordinate grids,
        each of shape ``(ny, nx)``. If omitted, a sensible default
        ``64×64`` grid spanning ±10 kpc is constructed via
        :func:`default_grid_kpc`. Pass an explicit grid when the
        production fit needs different resolution or extent.

    Notes
    -----
    JIT/grad/vmap-compatible: :meth:`run` is pure JAX. The grid is held
    as a static attribute, not closed over.
    """

    components: tuple[SpatialComponent, ...]
    grid_kpc: tuple[jnp.ndarray, jnp.ndarray]
    #: The :class:`SubModel` identifier. ``init=False`` because the protocol
    #: calls it a *stable* identifier: this constructor never accepted it, and
    #: an ordinary field declaration advertised otherwise to
    #: ``dataclasses.fields()`` and to the docs.
    name: str = field(default="spatial", init=False)

    def __init__(
        self,
        components: Sequence[SpatialComponent],
        grid_kpc: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    ) -> None:
        if grid_kpc is None:
            grid_kpc = default_grid_kpc()
        object.__setattr__(self, "components", tuple(components))
        object.__setattr__(self, "grid_kpc", grid_kpc)

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

        Inserts :attr:`grid_kpc` into ``state.derived["spatial_grid_xy_kpc"]``
        if not already present (callers can override by setting it on
        the incoming state). Each component then consumes the grid and
        updates :attr:`state.derived["spatial_profile_2d"]`.

        Parameters
        ----------
        state: ForwardState
            Incoming state. Wave grid is preserved; spatial grid is
            inserted (or kept if caller already set it).
        params: Mapping
            Free parameter values.

        Returns
        -------
        ForwardState
            New state with ``spatial_profile_2d`` populated.
        """
        if "spatial_grid_xy_kpc" not in state.derived:
            state = state.with_(derived=state.derived.with_(spatial_grid_xy_kpc=self.grid_kpc))
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
    (architecture spec §4.3, SED → Spatial ordering).

    Parameters
    ----------
    sed: :class:`tengri.SEDModel`
        SED sub-model (or anything satisfying :class:`SubModel`).
    spatial: :class:`SpatialModel`
        Spatial sub-model.

    Notes
    -----
    Composer-only, no physics of its own. Aggregates declared
    parameters from both halves; runs SED, then spatial, on the same
    threaded :class:`ForwardState`.

    JIT/grad/vmap-compatible.
    """

    sed: Any
    spatial: SpatialModel
    #: The :class:`SubModel` identifier. ``init=False`` for the same reason as
    #: on :class:`SpatialModel`, and here it also *closes* a hole: the
    #: generated constructor accepted ``name=`` and let a caller overwrite the
    #: identifier the protocol promises is stable.
    name: str = field(default="spatial_sed", init=False)

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
