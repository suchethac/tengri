# SPDX-License-Identifier: BSD-3-Clause
"""Recipe: frozen envelope of AGN block selectors.

A :class:`Recipe` is the internal representation of a composable AGN
choice: which block implements each of the five pipeline stages
(``disc → lines → feii → torus → attenuation``). It is used by both the
:func:`composable_agn_l_nu` runner (when called via JIT-composable mode)
and the :mod:`composable_precompute` builder (precompute mode).

**Users never construct ``Recipe`` directly.** The intended construction
sites are:

1. :meth:`Recipe.from_parameters`: read selectors off the flat
   ``agn_*_block`` attributes that :class:`tengri.Parameters` already
   carries today.
2. The companion nested-dict parser (see plan
   ``i-feel-like-its-serene-emerson.md``), once it lands, will produce
   a ``Recipe`` from an ``agn={'disc': {'type': 'grahsp_sbpl'}, ...}``
   group and stash it on ``Parameters._agn_recipe``.

The ``template_state`` and ``axis_params`` fields are optional and only
populated when the recipe is being prepared for the precompute path.

References
----------
Plan: ``~/.claude/plans/enumerated-watching-rainbow.md``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from tengri.components.agn.blocks._protocol import BLOCK_CATEGORIES
from tengri.components.agn.blocks.runner import validate_block_recipe

__all__ = ["Recipe"]


@dataclass(frozen=True)
class Recipe:
    r"""Frozen 6-tuple of AGN block selectors plus precompute metadata.

    Attributes
    ----------
    agn_disc_block, agn_nlr_block, agn_blr_block, agn_feii_block, \
agn_torus_block, agn_attenuation_block: str
        Names of registered block implementations (see
        :data:`tengri.components.agn.blocks.AGN_BLOCKS`). Each defaults
        to ``"none"`` so a bare ``Recipe()`` is the no-op pipeline.
    axis_params: tuple[str, ...]
        Parameter names to vary in the precompute grid; everything else
        is held at the user's fixed value (or registry default). Empty
        tuple means "no precompute axes: recipe is fully scalar."
    template_state: Any, optional
        Pre-loaded template bundle to thread through block bodies so they
        skip in-block ``load_*_templates()`` calls. ``None`` triggers the
        backwards-compatible lru_cache fallback inside each block. Set
        by :func:`composable_precompute.precompute` once per grid build.
    """

    agn_disc_block: str = "none"
    agn_nlr_block: str = "none"
    agn_blr_block: str = "none"
    agn_feii_block: str = "none"
    agn_torus_block: str = "none"
    agn_attenuation_block: str = "none"
    axis_params: tuple[str, ...] = field(default_factory=tuple)
    template_state: Any = None

    def __post_init__(self) -> None:
        """Eagerly run :func:`validate_block_recipe` on construction.

        Typos raise :class:`ValueError` immediately rather than silently
        falling through to a misbehaving runtime.
        """
        validate_block_recipe(
            agn_disc_block=self.agn_disc_block,
            agn_nlr_block=self.agn_nlr_block,
            agn_blr_block=self.agn_blr_block,
            agn_feii_block=self.agn_feii_block,
            agn_torus_block=self.agn_torus_block,
            agn_attenuation_block=self.agn_attenuation_block,
        )

    # ──────────────────────────────────────────────────────────────────
    # Constructors
    # ──────────────────────────────────────────────────────────────────

    @classmethod
    def from_parameters(
        cls,
        params: Any,
        *,
        axis_params: Sequence[str] = (),
        template_state: Any = None,
    ) -> Recipe:
        """Read selectors off a :class:`tengri.Parameters` instance.

        Parameters
        ----------
        params: Parameters
            Tengri ``Parameters`` object whose ``agn_*_block`` attributes
            were populated either via the flat ``Parameters(...)`` kwargs
            or by the companion nested-dict parser.
        axis_params: sequence of str, optional
            Parameter names that the precompute will vary over a grid.
            Default ``()`` (no precompute).
        template_state: Any, optional
            Pre-loaded template bundle. Default ``None``.

        Returns
        -------
        Recipe
        """
        return cls(
            agn_disc_block=getattr(params, "agn_disc_block", "none"),
            agn_nlr_block=getattr(params, "agn_nlr_block", "none"),
            agn_blr_block=getattr(params, "agn_blr_block", "none"),
            agn_feii_block=getattr(params, "agn_feii_block", "none"),
            agn_torus_block=getattr(params, "agn_torus_block", "none"),
            agn_attenuation_block=getattr(params, "agn_attenuation_block", "none"),
            axis_params=tuple(axis_params),
            template_state=template_state,
        )

    @classmethod
    def from_selectors(
        cls,
        *,
        disc: str = "none",
        nlr: str = "none",
        blr: str = "none",
        feii: str = "none",
        torus: str = "none",
        attenuation: str = "none",
        axis_params: Sequence[str] = (),
        template_state: Any = None,
    ) -> Recipe:
        """Construct a :class:`Recipe` from short positional category names.

        Convenience for callers who don't have a :class:`Parameters` object
        yet (e.g. interactive use, tests).
        """
        return cls(
            agn_disc_block=disc,
            agn_nlr_block=nlr,
            agn_blr_block=blr,
            agn_feii_block=feii,
            agn_torus_block=torus,
            agn_attenuation_block=attenuation,
            axis_params=tuple(axis_params),
            template_state=template_state,
        )

    # ──────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────

    def as_selector_dict(self) -> dict[str, str]:
        """Return ``{stage_name: block_name}`` for splicing into kwargs."""
        return {
            "agn_disc_block": self.agn_disc_block,
            "agn_nlr_block": self.agn_nlr_block,
            "agn_blr_block": self.agn_blr_block,
            "agn_feii_block": self.agn_feii_block,
            "agn_torus_block": self.agn_torus_block,
            "agn_attenuation_block": self.agn_attenuation_block,
        }

    def summary(self) -> list[tuple[str, str]]:
        """Return ``[(stage, block_name), ...]`` for ``model.summary()``.

        Consumed by the companion nested-dict plan's summary renderer; each
        row becomes a line under the ``agn`` group with provenance.
        """
        sel = self.as_selector_dict()
        return [(cat, sel[f"agn_{cat}_block"]) for cat in BLOCK_CATEGORIES]
