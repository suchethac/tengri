# SPDX-License-Identifier: BSD-3-Clause
"""JointObservation, composer that merges multiple observation models.

Used when a single fit consumes multiple data channels, total-flux
photometry plus fiber spectroscopy plus resolved imaging plus …, each
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
keys, child order matters when the same observable is published by
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
    *children: object
        Each must expose ``predict(state, params) → mapping``.
    name: str, default "joint"
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

    def predict_summed(
        self,
        per_pop_states: Mapping[str, ForwardState],
        per_pop_params: Mapping[str, Mapping[str, jnp.ndarray]],
    ) -> Mapping[str, jnp.ndarray]:
        """Sum each child observation's prediction across populations.

        For each child observation, run its ``predict`` on every
        population's state, then sum the resulting per-population
        dicts key-by-key (linear flux sum). Returns the merged dict.

        This is the multi-population entry point used by
        :meth:`tengri.ForwardModel.predict_observables` when more than one
        :class:`Population` is wired up.

        Parameters
        ----------
        per_pop_states: mapping of population name -> ForwardState
            One state per population (already namespace-merged in
            ``derived`` so cross-population reads work).
        per_pop_params: mapping of population name -> params dict
            The fixed-values-merged parameter dict each population
            saw, useful when child observations read
            population-specific parameters (e.g. per-population
            redshifts).

        Returns
        -------
        mapping of str -> array
            Merged + summed prediction dict, same shape as
            :meth:`predict`.
        """
        summed: dict[str, jnp.ndarray] = {}
        for child in self.children:
            per_pop_pred = {
                name: child.predict(state, per_pop_params[name])
                for name, state in per_pop_states.items()
            }
            all_keys: set[str] = set()
            for pred in per_pop_pred.values():
                all_keys.update(pred.keys())
            for key in all_keys:
                contributions = [pred[key] for pred in per_pop_pred.values() if key in pred]
                if not contributions:
                    continue
                total = contributions[0]
                for c in contributions[1:]:
                    total = total + c
                summed[key] = total  # last child wins on collision (same as predict)
        return summed
