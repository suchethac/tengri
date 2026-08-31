# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for the AGN group: most complex of the builders.

AGN composition has five orthogonal sub-block axes (``disc``,
``torus``, ``nlr``, ``blr``, ``feii``, ``atten``) plus a top-level
``composable`` model that orchestrates them. The grammar's intent:

>>> agn = {
...     "type": "composable",
...     "all_params": FREE,
...     "log_lbol": Uniform(9.42, 13.42),
...     "disc": {"type": "multicolor", "all_params": FREE},
...     "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
...     "nlr": {"type": "analytic"},
...     "blr": {"type": "analytic"},
...     "feii": {"type": "none"},
...     "atten": {"law": "prevot_smc", "all_params": Fixed(DEFAULT)},
... }

The factory mirror:

>>> from tengri import builders, FREE, Fixed, DEFAULT, Uniform
>>> agn = builders.agn.composable(
...     all_params=FREE,
...     log_lbol=Uniform(9.42, 13.42),
...     disc=builders.agn.disc.multicolor(all_params=FREE),
...     torus=builders.agn.torus.skirtor(all_params=Fixed(DEFAULT)),
...     nlr=builders.agn.nlr.analytic(),
...     blr=builders.agn.blr.analytic(),
...     feii=builders.agn.feii.none(),
...     atten=builders.agn.atten.smc_prevot(all_params=Fixed(DEFAULT)),
... )

All 14 top-level AGN models are exposed as factories:

- ``composable``: orchestrator for the six sub-blocks
- ``simple``: power-law disc + single-temperature torus
- ``standard``: multi-color disc + two-temperature torus
- ``multicolor_agn``: Shakura-Sunyaev disc + 2-T torus (Kubota & Done 2018)
- ``kubota_done``: outer-zone K&D disc + clumpy torus (alias for ``multicolor_agn``)
- ``kubota_done_full``: full 3-zone K&D disc + torus + Comptonization
- ``skirtor``: power-law disc + SKIRTOR clumpy torus
- ``silva04``: power-law disc + Silva+04 smooth torus
- ``cat3d_wind``: power-law disc + CAT3D-Wind clumpy torus
- ``adaf``: ADAF + truncated disc + simple torus (low-luminosity)
- ``relagn``: RELAGN relativistic disc + 2-T torus
- ``grahsp``: multi-component semi-analytical AGN (Granada+Hönig+Ruiz)
- ``qsogen``: Temple, Hewett & Banerji (2021) empirical quasar SED
- ``unified_nlr_blr``: unified AGN + NLR/BLR decomposition + geometric masking

Examples
--------
>>> from tengri import builders, FREE, Uniform
>>> agn = builders.agn.skirtor(all_params=FREE, log_lbol=Uniform(9.42, 13.42))
>>> agn = builders.agn.simple(log_mbh=Uniform(6, 9))
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from tengri._completion import curated_dir
from tengri.builders._factory import (
    _DEFAULT_WILDCARD,
    UNSET,
    _pop_wildcard,
    _validate_wildcard,
    make_factory,
    short_form,
)
from tengri.builders.agn import atten, blr, disc, feii, nlr, torus
from tengri.parameters.groups import _AGN_PARTITION
from tengri.parameters.registry import recipe_parameters
from tengri.parameters.sentinels import FREE, WILDCARD_ALIAS

_AXIS_MODULES = {
    "disc": disc,
    "torus": torus,
    "nlr": nlr,
    "blr": blr,
    "feii": feii,
    "atten": atten,
}


def _discover_shared_params() -> list[str]:
    """Short-form names for params the parser routes to ``agn`` (not a sub-block).

    These are the cross-block knobs (log_lbol, log_mbh, a_spin, …) the
    ``composable`` orchestrator exposes at the top level. Sub-block-
    specific params live inside their sub-block dicts and belong to the
    sub-block factories instead.
    """
    recipe = {
        "sfh": {"type": "dpl"},
        "agn": {
            "type": "composable",
            WILDCARD_ALIAS: FREE,
            "disc": {"type": "powerlaw"},
            "torus": {"type": "skirtor"},
            "nlr": {"type": "analytic"},
            "blr": {"type": "analytic"},
            "feii": {"type": "none"},
            "atten": {"law": "prevot_smc"},
        },
    }
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = recipe_parameters(recipe, free_only=False)
    full_names = sorted(rec.name for rec in records if _AGN_PARTITION.get(rec.name) == "agn")
    return [short_form(name, prefixes=("agn_",)) for name in full_names]


_SHARED_SHORT_PARAMS = _discover_shared_params()


def composable(**kwargs: Any) -> dict:
    """Build a ``composable`` AGN config dict with six sub-block selectors."""
    wildcard = _pop_wildcard("agn.composable", kwargs)
    _validate_wildcard("agn.composable", wildcard)
    sub_blocks = {axis: kwargs.pop(axis, None) for axis in _AXIS_MODULES}
    for axis, value in sub_blocks.items():
        if value is None:
            continue
        if not isinstance(value, dict):
            raise TypeError(
                f"agn.composable({axis}=...): expected a dict from "
                f"builders.agn.{axis}.<variant>(...), got "
                f"{type(value).__name__}."
            )
    unknown = [k for k in kwargs if k not in _SHARED_SHORT_PARAMS]
    if unknown:
        raise TypeError(
            f"agn.composable() got unexpected keyword arguments: {unknown}. "
            f"Valid sub-blocks: {list(_AXIS_MODULES)}. "
            f"Valid shared params: {_SHARED_SHORT_PARAMS}. "
            f"(Pass ``all_params=FREE`` or ``all_params=Fixed(DEFAULT)`` -- or the "
            f"synonym ``other_params=`` -- to set the policy.)"
        )
    out: dict[str, Any] = {"type": "composable", WILDCARD_ALIAS: wildcard}
    for short in _SHARED_SHORT_PARAMS:
        if short in kwargs and kwargs[short] is not UNSET:
            out[short] = kwargs[short]
    for axis, value in sub_blocks.items():
        if value is not None:
            out[axis] = value
    return out


# Attach a real signature so IDEs see the sub-block + shared-param kwargs.
_sig_params = [
    inspect.Parameter(
        "all_params", inspect.Parameter.KEYWORD_ONLY, default=_DEFAULT_WILDCARD, annotation=Any
    ),
    inspect.Parameter(
        "other_params", inspect.Parameter.KEYWORD_ONLY, default=UNSET, annotation=Any
    ),
]
for axis in _AXIS_MODULES:
    _sig_params.append(
        inspect.Parameter(axis, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=dict)
    )
for short in _SHARED_SHORT_PARAMS:
    _sig_params.append(
        inspect.Parameter(short, inspect.Parameter.KEYWORD_ONLY, default=UNSET, annotation=Any)
    )
composable.__signature__ = inspect.Signature(_sig_params, return_annotation=dict)
composable.__qualname__ = "tengri.builders.agn.composable"
composable.__module__ = "tengri.builders.agn"


def _discover_top_level_params() -> list[str]:
    """Short-form names for params all 13 top-level AGN models activate.

    All non-composable top-level models (simple, standard, skirtor, etc.)
    share the same 48-param superset. Discover them once via the simplest
    model and use the same set for all 13.
    """
    recipe = {
        "sfh": {"type": "dpl"},
        "agn": {"type": "simple", WILDCARD_ALIAS: FREE},
    }
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = recipe_parameters(recipe, free_only=False)
    full_names = sorted(rec.name for rec in records if rec.name.startswith("agn_"))
    return [short_form(name, prefixes=("agn_",)) for name in full_names]


_TOP_LEVEL_SHORT_PARAMS = _discover_top_level_params()

# List of top-level AGN model names (excluding composable).
_TOP_LEVEL_MODELS = [
    "adaf",
    "cat3d_wind",
    "grahsp",
    "kubota_done",
    "kubota_done_full",
    "multicolor_agn",
    "qsogen",
    "relagn",
    "silva04",
    "simple",
    "skirtor",
    "standard",
    "unified_nlr_blr",
]


def _populate_top_level_factories() -> dict[str, Callable[..., dict]]:
    """Build one factory per top-level AGN model."""
    factories: dict[str, Callable[..., dict]] = {}
    for model_name in _TOP_LEVEL_MODELS:
        factories[model_name] = make_factory(
            variant=model_name,
            short_params=_TOP_LEVEL_SHORT_PARAMS,
            qualname_prefix="tengri.builders.agn",
            module_name="tengri.builders.agn",
            short_doc=f"AGN top-level model: {model_name!r}.",
        )
    return factories


_TOP_LEVEL_FACTORIES = _populate_top_level_factories()
globals().update(_TOP_LEVEL_FACTORIES)


def available() -> list[str]:
    """Return the AGN factories exposed at the top level.

    Lists the ``"composable"`` orchestrator plus 13 top-level models.
    """
    return sorted(["composable", *_TOP_LEVEL_FACTORIES])


def available_axes() -> dict[str, list[str]]:
    """Return ``{axis: [variants]}`` for the six composable sub-block axes."""
    return {axis: mod.available() for axis, mod in _AXIS_MODULES.items()}


_factory_callable: Callable[..., dict] = composable  # type: ignore[assignment]

__all__ = [
    "atten",
    "available",
    "available_axes",
    "blr",
    "composable",
    "disc",
    "feii",
    "nlr",
    "torus",
    *_TOP_LEVEL_MODELS,
]


__dir__ = curated_dir(__all__)
