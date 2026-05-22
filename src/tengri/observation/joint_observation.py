"""JointObservation — composer that merges multiple observation models.

Used when a single fit consumes multiple data channels — total-flux
photometry plus fiber spectroscopy plus resolved imaging plus … — each
with its own observation model. The composer calls each wrapped model's
``predict`` and merges the dicts.

Example
-------

.. code-block:: python

    obs = JointObservation(
        TotalPhotometryObservation(...),
        FiberSpectroscopyObservation(...),
    )
    pred = obs.predict(state, params)
    # pred["phot_fnu"] from the first child; pred["spec_fnu"] (aperture-scaled)
    # from the second.

No physics of its own. Pure dict merging. Last child wins on shared
keys — child order matters when the same observable is published by
multiple wrapped models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tengri.protocols.component import ForwardState

__all__ = ["JointObservation"]


@dataclass(frozen=True)
class JointObservation:
    """Merge multiple observation models into one ``predict`` dict.

    Parameters
    ----------
    *children : object
        Each must expose ``predict(state, params) → mapping``.
    name : str, default "joint"
        Identifier for diagnostics.

    Notes
    -----
    JIT/grad/vmap-compatible (every child must be).
    """

    children: tuple[Any, ...]
    name: str = "joint"

    def __init__(self, *children: Any, name: str = "joint") -> None:
        object.__setattr__(self, "children", tuple(children))
        object.__setattr__(self, "name", name)

    def predict(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> Mapping[str, jnp.ndarray]:
        out: dict[str, jnp.ndarray] = {}
        for child in self.children:
            out.update(child.predict(state, params))
        return out
