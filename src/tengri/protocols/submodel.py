# SPDX-License-Identifier: BSD-3-Clause
"""SubModel protocol: one mode of the ForwardModel (SED, spatial, joint).

Defined by the forward-model architecture spec
(``docs/dev/archive/forward-model-architecture.md``) §4. A SubModel is the thin
composer over a list of components. Each population carries one SED
SubModel and optionally one spatial SubModel; ``ForwardModel`` runs the
populations in sequence and hands the result to ``ObservationModel``.

This is a Protocol, not an ABC: implementations satisfy it by shape.
The runtime-checkable variant is provided so smoke tests can assert
``isinstance(obj, SubModel)`` without importing a concrete base.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

import jax.numpy as jnp

from tengri.protocols.component import ForwardState, ParamDeclaration

__all__ = ["SubModel"]


@runtime_checkable
class SubModel(Protocol):
    """Contract for one mode of a :class:`ForwardModel`.

    Required attributes
    -------------------
    name : str
        Stable identifier for diagnostics. Examples: ``"sed"``,
        ``"spatial"``, ``"spatial_sed"``.

    Required methods
    ----------------
    declared_parameters() -> list[ParamDeclaration]
        Aggregated parameter declarations across this SubModel's
        components. Consumed by ``Parameters`` and by the CI prefix
        guard.

    run(state, params) -> ForwardState
        Pure JAX. Threads the input state through the SubModel's
        components and returns a new ``ForwardState`` with the
        SubModel's contribution applied. Must not mutate the input.

    Notes
    -----
    JIT/grad/vmap-compatible: ``run`` is pure JAX. Static configuration
    (component list, frozen pytree state) is held on ``self`` as
    Python attributes, not captured by closure.
    """

    name: str

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Aggregated parameter declarations from this SubModel's components.

        Returns
        -------
        list of :class:`ParamDeclaration`
            One entry per free parameter this SubModel exposes,
            aggregated across its components. The orchestrator hands
            these to :class:`tengri.Parameters` so the user doesn't
            have to register them by hand.

        Notes
        -----
        Every ``name`` MUST start with the corresponding component's
        :attr:`parameter_prefix` *or* be listed in
        :data:`BARE_NAME_ALLOWLIST` (currently just ``redshift``). The
        param-prefix CI guard at ``tools/check_param_prefixes.py``
        enforces this across the aggregated set.
        """
        ...

    def run(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Pure JAX step. Thread state through this SubModel's components.

        Reads ``params`` (already sliced to this SubModel's parameter
        prefixes by the caller) plus any frozen tensors held by ``self``,
        then returns a *new* :class:`ForwardState` with this SubModel's
        contribution applied.

        Parameters
        ----------
        state : ForwardState
            Current state from upstream sub-models (e.g. spatial reads
            keys produced by SED in :class:`SpatialSEDModel`).
        params : mapping of str -> array
            Free parameter values for parameters owned by this
            SubModel's components.

        Returns
        -------
        ForwardState
            New state. MUST NOT mutate the input.
        """
        ...
