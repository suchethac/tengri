# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for radio config dicts.

Variants come from :data:`tengri.parameters.groups._VALID_RADIO_TYPES`
(currently ``none`` and ``condon92``). The free-parameter set is shared
across variants — the variant string selects the physics model; the
params describe model knobs (e.g. ``q_ir``, ``alpha_thin``,
``alpha_thick``).

.. note::
   The ``*`` (wildcard) key on radio config dicts is accepted by the
   parser but currently does **not** flip individual radio params to
   FREE — this is a pre-existing limitation of
   :func:`tengri.parameters.groups.parse_groups` for radio / X-ray /
   IGM groups (the wildcard only works for components whose registry
   declares the per-variant param list explicitly, like SFH). To make
   a radio param free, pass an explicit :class:`Distribution`:
   ``builders.radio.condon92(q_ir=Uniform(2.0, 3.0))``. The factory
   itself faithfully emits the wildcard into the dict; the parser
   behaviour is identical to a hand-written dict.

Examples
--------
>>> from tengri import builders, Uniform
>>> builders.radio.condon92(q_ir=Uniform(2.0, 3.0))  # doctest: +SKIP
{'type': 'condon92', '*': FIXED, 'q_ir': Uniform(...)}
"""

from __future__ import annotations

from collections.abc import Callable

from tengri.builders._factory import make_factory, short_form
from tengri.parameters.groups import _VALID_RADIO_TYPES
from tengri.parameters.registry import recipe_parameters

_PREFIXES = ("radio_",)


def _discover_params(variant: str) -> list[str]:
    if variant == "none":
        return []
    recipe = {
        "sfh": {"type": "dpl"},
        "radio": {"type": variant, "*": "FREE_PLACEHOLDER"},
    }
    # Use FREE so all conditional params surface.
    from tengri.parameters.sentinels import FREE

    recipe["radio"]["*"] = FREE
    records = recipe_parameters(recipe, free_only=False)
    return [
        short_form(rec.name, prefixes=_PREFIXES)
        for rec in records
        if rec.name.startswith(_PREFIXES)
    ]


def _populate_factories() -> dict[str, Callable[..., dict]]:
    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(_VALID_RADIO_TYPES):
        factories[variant] = make_factory(
            variant=variant,
            short_params=_discover_params(variant),
            qualname_prefix="tengri.builders.radio",
            module_name="tengri.builders.radio",
            short_doc=f"Radio model: {variant!r}.",
        )
    return factories


_FACTORIES = _populate_factories()
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of radio variant names exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]
