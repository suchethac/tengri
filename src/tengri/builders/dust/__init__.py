# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for the dust group (attenuation + emission).

Two top-level factories — ``single_component`` and ``two_component`` —
plus a nested :mod:`~tengri.builders.dust.emission` submodule for the IR
emission sub-block.

The grammar:

>>> dust = {
...     "type": "two_component",
...     "law_bc": "calzetti",
...     "law_diff": "calzetti",
...     "all_params": FREE,
...     "tau_bc": Uniform(0, 4),
...     "emission": {"type": "dale2014", "all_params": FIXED},
... }

The factory mirror:

>>> from tengri import builders, FREE, Uniform, FIXED
>>> dust = builders.dust.two_component(
...     law_bc="calzetti",
...     law_diff="calzetti",
...     defaults=FREE,
...     tau_bc=Uniform(0, 4),
...     emission=builders.dust.emission.dale2014(defaults=FIXED),
... )

The ``law_bc`` / ``law_diff`` kwargs accept any key registered in
:data:`tengri.components.dust.attenuation.DUST_LAWS`. Single-component
factories use ``law`` (a singular alias that the grammar parser
accepts) — the attenuation law string is the only setting that
distinguishes one ``two_component`` model from another (the parameter
list is identical across laws).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from tengri._completion import curated_dir
from tengri.builders._factory import UNSET, _pop_wildcard, short_form
from tengri.builders.dust import emission  # nested factory namespace
from tengri.components.dust.attenuation import DUST_LAWS
from tengri.parameters.registry import recipe_parameters
from tengri.parameters.sentinels import FIXED, FREE, WILDCARD_ALIAS

# Param shortlists per dust_model — discovered at import time so adding a
# new attenuation knob in the registry surfaces here automatically.
_PREFIXES = ("dust_",)
_ATTEN_PREFIXES_TO_KEEP = (
    "dust_tau_v",
    "dust_tau_bc",
    "dust_tau_diff",
    "dust_slope",
    "dust_f_obscuration",
    "dust_bump_strength",
    "dust_delta",
    "dust_Rv",
)


def _discover_attenuation_params(dust_model: str) -> list[str]:
    """Activate every attenuation param this dust_model accepts.

    Returns the short-form list. ``single_component`` carries
    ``tau_v`` (single optical depth); ``two_component`` carries
    ``tau_bc`` + ``tau_diff`` instead. Both share the other knobs
    (``slope``, ``Rv``, ``f_obscuration``, etc.).
    """
    recipe = {
        "sfh": {"type": "dpl"},
        "dust": {
            "type": dust_model,
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            WILDCARD_ALIAS: FREE,
        },
    }
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        records = recipe_parameters(recipe, free_only=False)
    return sorted(
        short_form(rec.name, prefixes=_PREFIXES)
        for rec in records
        if rec.name.startswith(_ATTEN_PREFIXES_TO_KEEP)
    )


def _make_dust_factory(
    *,
    dust_model: str,
    short_params: list[str],
    setting_names: tuple[str, ...],
    setting_defaults: dict[str, str],
) -> Callable[..., dict]:
    """Build a top-level dust factory (single_component or two_component).

    Differs from the generic :func:`make_factory` in that it accepts
    string-valued settings (``law_bc``, ``law_diff``) and a nested
    ``emission`` sub-block kwarg. The output dict matches what the
    grammar parser expects.
    """

    def factory(**kwargs: Any) -> dict:
        wildcard = _pop_wildcard(f"dust.{dust_model}", kwargs)
        if wildcard not in (FREE, FIXED):
            raise ValueError(
                f"dust.{dust_model}(defaults=...): expected FREE or FIXED, got "
                f"{wildcard!r}. Use tengri.FREE or tengri.FIXED."
            )
        # String settings (e.g. law_bc, law_diff).
        settings: dict[str, str] = {}
        for s in setting_names:
            if s in kwargs:
                value = kwargs.pop(s)
                if value not in DUST_LAWS:
                    raise ValueError(
                        f"dust.{dust_model}({s}={value!r}): unknown attenuation "
                        f"law. Valid: {sorted(DUST_LAWS)}."
                    )
                settings[s] = value
            else:
                settings[s] = setting_defaults[s]
        emission_block = kwargs.pop("emission", None)
        valid_kwargs = ["defaults", *setting_names, "emission", *short_params]
        unknown = [k for k in kwargs if k not in short_params]
        if unknown:
            raise TypeError(
                f"dust.{dust_model}() got unexpected keyword arguments: "
                f"{unknown}. Valid: {valid_kwargs}."
            )

        out: dict[str, Any] = {"type": dust_model, WILDCARD_ALIAS: wildcard, **settings}
        for short in short_params:
            if short in kwargs and kwargs[short] is not UNSET:
                out[short] = kwargs[short]
        if emission_block is not None:
            if not isinstance(emission_block, dict):
                raise TypeError(
                    f"dust.{dust_model}(emission=...): expected a dict (e.g. "
                    "from builders.dust.emission.<variant>(...)), got "
                    f"{type(emission_block).__name__}."
                )
            out["emission"] = emission_block
        return out

    sig_params = [
        inspect.Parameter(
            "defaults", inspect.Parameter.KEYWORD_ONLY, default=FIXED, annotation=Any
        ),
    ]
    for s in setting_names:
        sig_params.append(
            inspect.Parameter(
                s,
                inspect.Parameter.KEYWORD_ONLY,
                default=setting_defaults[s],
                annotation=str,
            )
        )
    for short in short_params:
        sig_params.append(
            inspect.Parameter(short, inspect.Parameter.KEYWORD_ONLY, default=UNSET, annotation=Any)
        )
    sig_params.append(
        inspect.Parameter(
            "emission",
            inspect.Parameter.KEYWORD_ONLY,
            default=None,
            annotation=dict,
        )
    )
    factory.__signature__ = inspect.Signature(sig_params, return_annotation=dict)
    factory.__name__ = dust_model
    factory.__qualname__ = f"tengri.builders.dust.{dust_model}"
    factory.__module__ = "tengri.builders.dust"

    doc_lines: list[str] = []
    doc_lines.append(f"Build a config dict for the {dust_model!r} attenuation model.")
    doc_lines.append("")
    settings_desc = ", ".join(f"``{s}``" for s in setting_names)
    doc_lines.append(
        f"Settings: {settings_desc}. Each accepts any key registered in "
        f"``tengri.components.dust.attenuation.DUST_LAWS`` "
        f"(e.g. ``'calzetti'``, ``'smc'``, ``'cardelli'``)."
    )
    doc_lines.append("")
    doc_lines.append("Parameters")
    doc_lines.append("----------")
    doc_lines.append("defaults : sentinel, optional")
    doc_lines.append(
        "    Wildcard policy. ``FREE`` makes unspecified attenuation params "
        "fit; ``FIXED`` (default) pins them to registry defaults."
    )
    for s in setting_names:
        doc_lines.append(f"{s} : str, optional")
        doc_lines.append(
            f"    Attenuation law name from DUST_LAWS. Default {setting_defaults[s]!r}."
        )
    doc_lines.append("emission : dict, optional")
    doc_lines.append(
        "    Nested config from ``builders.dust.emission.<variant>(...)`` "
        "selecting the IR re-emission model."
    )
    for short in short_params:
        doc_lines.append(f"{short} : Distribution, sentinel, or scalar, optional")
        doc_lines.append("    Override the registry-default prior.")
    doc_lines.append("")
    doc_lines.append("Returns")
    doc_lines.append("-------")
    doc_lines.append("dict")
    doc_lines.append("    Config dict matching the :meth:`SEDModel.build` grammar.")
    factory.__doc__ = "\n".join(doc_lines)
    return factory


# Build the two top-level factories at import time.
single_component = _make_dust_factory(
    dust_model="single_component",
    short_params=_discover_attenuation_params("single_component"),
    setting_names=("law_bc",),
    setting_defaults={"law_bc": "calzetti"},
)
two_component = _make_dust_factory(
    dust_model="two_component",
    short_params=_discover_attenuation_params("two_component"),
    setting_names=("law_bc", "law_diff"),
    setting_defaults={"law_bc": "calzetti", "law_diff": "calzetti"},
)


def available() -> list[str]:
    """Return the list of top-level dust variants exposed."""
    return ["single_component", "two_component"]


__all__ = ["available", "emission", "single_component", "two_component"]


__dir__ = curated_dir(__all__)
