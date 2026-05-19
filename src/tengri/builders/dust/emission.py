# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for the dust emission sub-block.

The grammar nests the emission model inside the dust group:

>>> dust = {
...     "type": "two_component",
...     "law_bc": "calzetti",
...     "emission": {"type": "dale2014", "*": FIXED, "alpha_dale": Fixed(2.0)},
... }

Each emission variant in
:data:`tengri.parameters.groups._VALID_DUST_EMISSION_TYPES` gets a
factory in this module. The parser activates a single superset of
dust-emission params regardless of which model is chosen, so all
factories share an identical signature; the variant string selects
the physics.

Examples
--------
>>> from tengri import builders, FIXED, Fixed
>>> builders.dust.emission.dale2014(_=FIXED, alpha_dale=Fixed(2.0))  # doctest: +SKIP
{'type': 'dale2014', '*': FIXED, 'alpha_dale': Fixed(2.0)}
"""

from __future__ import annotations

from collections.abc import Callable

from tengri.builders._factory import make_factory, short_form
from tengri.parameters.groups import _VALID_DUST_EMISSION_TYPES
from tengri.parameters.registry import recipe_parameters
from tengri.parameters.sentinels import FREE

_PREFIXES = ("dust_",)
# Param names that belong to dust *emission* rather than attenuation.
# The parser activates these only when an emission model is selected;
# they're the natural set for the emission factories.
_EMISSION_PREFIXES = (
    "dust_T",
    "dust_beta_ir",
    "dust_alpha_dale",
    "dust_umin",
    "dust_umax",
    "dust_gamma_dl",
    "dust_qpah",
    "dust_pah",
    "dust_lgU",
    "dust_log_",
)


def _discover_params(variant: str) -> list[str]:
    """Return short-form names for the emission-side dust params.

    Each emission variant activates the same superset (the parser is
    conservative); we still introspect per variant so that future
    variants with extra params surface automatically.
    """
    recipe = {
        "sfh": {"type": "dpl"},
        "dust": {
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "*": FREE,
            "emission": {"type": variant, "*": FREE},
        },
    }
    records = recipe_parameters(recipe, free_only=False)
    out: list[str] = []
    for rec in records:
        if not rec.name.startswith(_EMISSION_PREFIXES):
            continue
        out.append(short_form(rec.name, prefixes=_PREFIXES))
    return out


def _populate_factories() -> dict[str, Callable[..., dict]]:
    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(_VALID_DUST_EMISSION_TYPES):
        factories[variant] = make_factory(
            variant=variant,
            short_params=_discover_params(variant),
            qualname_prefix="tengri.builders.dust.emission",
            module_name="tengri.builders.dust.emission",
            short_doc=f"Dust IR emission model: {variant!r}.",
        )
    return factories


_FACTORIES = _populate_factories()
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of emission-model variants exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
