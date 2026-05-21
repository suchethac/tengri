"""_LegacySEDSubModel — migration shim wrapping :class:`SEDModel` as a :class:`SubModel`.

This adapter exists to land the ``ForwardModel`` outer shell without
disturbing the existing :class:`tengri.forward.sed_model.SEDModel`
internals. Subsequent plans factor SEDModel's chain into a first-class
SubModel; at that point this adapter is deleted.

See the tracer-bullet plan at
``docs/superpowers/plans/2026-05-21-forward-model-tracer-bullet.md``.

Underscore-prefixed because it is not part of the public API.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp

from tengri.protocols.component import ForwardState, ParamDeclaration

if TYPE_CHECKING:
    from tengri.forward.sed_model import SEDModel

__all__ = ["_LegacySEDSubModel"]


@dataclass(frozen=True)
class _LegacySEDSubModel:
    """SubModel adapter around an existing :class:`SEDModel`.

    Parameters
    ----------
    sed_model : SEDModel
        Constructed SEDModel instance. Held by reference; not copied.

    Notes
    -----
    The adapter does no physics of its own. ``run(state, params)``
    delegates to the wrapped SEDModel's internal forward pipeline.
    ``declared_parameters`` re-shapes ``sed_model.spec`` into a list
    of :class:`ParamDeclaration` so :class:`ForwardModel` can
    introspect free parameters uniformly across populations.
    """

    sed_model: "SEDModel"
    name: str = "sed"

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Free parameter declarations from the wrapped SEDModel.

        Returns the same set as ``sed_model.spec.free_params``, lifted
        into :class:`ParamDeclaration` tuples so the result type matches
        the SubModel Protocol.
        """
        spec = self.sed_model.spec
        decls: list[ParamDeclaration] = []
        for pname in spec.free_params:
            prior = spec._distributions[pname]
            decls.append(
                ParamDeclaration(
                    name=pname,
                    prior=prior,
                    description="",
                    units="",
                )
            )
        return decls

    def run(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Run the wrapped SEDModel's forward pipeline.

        Delegates to the existing SEDModel's ``predict_state`` method
        that produces a ``ForwardState``. The method threads ``params``
        through the orchestrator chain without modification.

        Parameters
        ----------
        state : ForwardState
            Input state (typically with ``wave`` field initialized).
        params : mapping of str -> array
            Free parameter values.

        Returns
        -------
        ForwardState
            New state with SED contributions applied. Pure JAX.
        """
        return self.sed_model.predict_state(params)
