# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Shared helper for the six AGN sub-block factory modules.

The AGN grammar has six orthogonal composable axes: ``disc``,
``torus``, ``nlr``, ``blr``, ``feii``, ``atten``. Each axis has a fixed set
of valid variant strings (the parser's
``_VALID_AGN_<AXIS>_TYPES`` enums) and a sub-block-specific param
partition: all variants within an axis share the same parameter set
(the variant string selects the physics).

This helper centralizes the introspection + factory generation so
each ``tengri.builders.agn.<axis>`` submodule is a thin two-liner.
"""

from __future__ import annotations

from collections.abc import Callable

from tengri.builders._factory import make_factory, short_form
from tengri.parameters.groups import _AGN_PARTITION
from tengri.parameters.registry import recipe_parameters
from tengri.parameters.sentinels import FREE, WILDCARD_ALIAS


def _discover_sub_block_params(axis: str, representative_variant: str) -> list[str]:
    """Return short-form param names this AGN sub-block axis exposes.

    All variants within an axis activate the same param set per
    ``_AGN_PARTITION``, so introspecting one representative variant is
    enough. We filter to ``_AGN_PARTITION[name] == f"agn.{axis}"`` —
    those are the parameters the parser routes into the sub-block dict.
    """
    target_path = f"agn.{axis}"
    recipe = {
        "sfh": {"type": "dpl"},
        "agn": {
            "type": "composable",
            WILDCARD_ALIAS: FREE,
            axis: {"type": representative_variant, WILDCARD_ALIAS: FREE},
        },
    }
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = recipe_parameters(recipe, free_only=False)
    full_names = sorted(rec.name for rec in records if _AGN_PARTITION.get(rec.name) == target_path)
    # Short-form: strip leading ``agn_`` (the parser's
    # _extract_short_name for AGN sub-blocks strips just the agn_
    # prefix, leaving e.g. ``tau_skirtor`` short for ``agn_tau_skirtor``).
    return [short_form(name, prefixes=("agn_",)) for name in full_names]


def build_axis_factories(
    *,
    axis: str,
    variants: set[str],
    representative_variant: str,
) -> dict[str, Callable[..., dict]]:
    """Build one factory per variant for a single AGN sub-block axis.

    Parameters
    ----------
    axis : str
        One of ``"disc"``, ``"torus"``, ``"nlr"``, ``"blr"``, ``"feii"``,
        ``"atten"``.
    variants : set[str]
        Variant names for this axis (from ``_VALID_AGN_<AXIS>_TYPES``).
    representative_variant : str
        Variant used to probe parameter activation. Any non-``none``
        entry works; choose one with a stable activation profile.
    """
    short_params = _discover_sub_block_params(axis, representative_variant)
    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(variants):
        factories[variant] = make_factory(
            variant=variant,
            short_params=short_params,
            qualname_prefix=f"tengri.builders.agn.{axis}",
            module_name=f"tengri.builders.agn.{axis}",
            short_doc=f"AGN {axis} sub-block: {variant!r}.",
        )
    return factories
