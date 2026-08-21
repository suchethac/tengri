# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for IGM (intergalactic medium) config dicts.

The IGM grammar accepts a ``type`` selecting the transmission curve
(``none`` / ``madau`` / ``inoue14``) plus two optional boolean sub-flags
that activate additional free parameters:

- ``patchy=True`` enables a patchy-reionization parameterization, adding
  ``igm_bubble_mpc`` and ``igm_x_HI``.
- ``dla=True`` enables a damped-Lyman-α absorber, adding the
  ``dla_log_n_hi``, ``dla_z``, ``dla_b_turb``, ``dla_temp`` params.

Variants are introspected from
:data:`tengri.components.igm.IGM_TRANSMISSION_MODELS` (canonical names
only — aliases like ``"inoue"`` are validator-side back-compat and get
no separate factory); the activated free-param set comes from a one-shot
call to :func:`tengri.parameters.registry.recipe_parameters`, so adding
a new DLA / patchy parameter in the registry surfaces here automatically.

Examples
--------
>>> from tengri import builders, FREE, Uniform
>>> # Plain Inoue+14 (no extras)
>>> builders.igm.inoue14()
{'type': 'inoue14', 'all_params': FIXED}
>>> # Toggle patchy reionization, override the bubble size prior
>>> builders.igm.inoue14(all_params=FREE, patchy=True, bubble_mpc=Uniform(5, 50))  # doctest: +SKIP
{'type': 'inoue14', 'all_params': FREE, 'patchy': True, 'bubble_mpc': Uniform(...)}
"""

from __future__ import annotations

from collections.abc import Callable

from tengri._completion import curated_dir
from tengri.builders._factory import make_factory, short_form
from tengri.parameters.groups import _valid_igm_types
from tengri.parameters.registry import recipe_parameters

_PREFIXES = ("igm_", "dla_")  # The igm group dict carries both prefixes.

# Map: which boolean flag activates each short-form param. Used by the
# factory so that ``builders.igm.inoue14(bubble_mpc=Uniform(...))`` is
# enough to imply ``patchy=True`` without forcing the user to set both.
_PATCHY_SHORT = ("bubble_mpc", "x_HI")
_DLA_SHORT = ("log_n_hi", "z", "b_turb", "temp")


def _discover_params(variant: str) -> tuple[list[str], dict[str, str]]:
    """Return (short_params, flag_param_map) for one IGM variant.

    Runs :func:`recipe_parameters` once with both flags enabled to
    surface every potentially-activated parameter. The factory's runtime
    behavior then auto-enables the right flag when the user provides
    one of the conditional params.
    """
    if variant == "none":
        return [], {}
    recipe = {
        "sfh": {"type": "dpl"},  # any additive SFH so the parser doesn't error
        "igm": {"type": variant, "patchy": True, "dla": True},
    }
    records = recipe_parameters(recipe, free_only=False)
    short_params: list[str] = []
    flag_map: dict[str, str] = {}
    for rec in records:
        if not any(rec.name.startswith(p) for p in _PREFIXES):
            continue
        short = short_form(rec.name, prefixes=_PREFIXES)
        short_params.append(short)
        if rec.name.startswith("dla_"):
            flag_map[short] = "dla"
        elif short in _PATCHY_SHORT:
            flag_map[short] = "patchy"
    return short_params, flag_map


def _populate_factories() -> dict[str, Callable[..., dict]]:
    # Aliases (e.g. ``"inoue"`` → ``"inoue14"``) are validator-side back-compat
    # and don't get a separate user-facing factory — the canonical entry serves
    # both. Filter them out so ``builders.igm.available()`` only lists canonical
    # names + ``"none"``. (Contract: ``test_igm_builder_factories_skip_aliases``.)
    from tengri.components.igm.igm import _IGM_ALIASES

    factories: dict[str, Callable[..., dict]] = {}
    for variant in sorted(_valid_igm_types()):
        if variant in _IGM_ALIASES:
            continue
        short_params, flag_map = _discover_params(variant)
        factories[variant] = make_factory(
            variant=variant,
            short_params=short_params,
            qualname_prefix="tengri.builders.igm",
            module_name="tengri.builders.igm",
            short_doc=f"IGM transmission: {variant!r}.",
            bool_flags=("patchy", "dla") if variant != "none" else (),
            flag_param_map=flag_map,
        )
    return factories


_FACTORIES = _populate_factories()
globals().update(_FACTORIES)


def available() -> list[str]:
    """Return the list of IGM variant names exposed by this module."""
    return sorted(_FACTORIES)


__all__ = ["available", *sorted(_FACTORIES)]


__dir__ = curated_dir(__all__)
