# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories for the dust group (attenuation + emission).

Two top-level factories — ``single_component`` and ``two_component`` —
plus a nested :mod:`~tengri.builders.dust.emission` submodule for the IR
emission sub-block.

The grammar (single-component example):

>>> dust = {
...     "type": "single_component",
...     "law": "calzetti",  # Required: no defaults in grammar
...     "all_params": FREE,
...     "tau_v": Uniform(0, 4),
... }

The grammar (two-component example):

>>> dust = {
...     "type": "two_component",
...     "law": "calzetti",  # Shared law, or use law_bc/law_diff separately
...     "all_params": FREE,
...     "tau_bc": Uniform(0, 2),
...     "tau_diff": Uniform(0, 4),
...     "emission": {"type": "dale2014", "all_params": FIXED},
... }

The factory mirror (single-component):

>>> from tengri import builders, FREE, Uniform, FIXED
>>> dust = builders.dust.single_component(
...     law="calzetti",  # Required
...     defaults=FREE,
...     tau_v=Uniform(0, 4),
... )

The factory mirror (two-component):

>>> dust = builders.dust.two_component(
...     law="calzetti",  # Shared law, or use law_bc/law_diff
...     defaults=FREE,
...     tau_bc=Uniform(0, 2),
...     tau_diff=Uniform(0, 4),
...     emission=builders.dust.emission.dale2014(defaults=FIXED),
... )

The ``law`` / ``law_bc`` / ``law_diff`` kwargs accept any key registered in
:data:`tengri.components.dust.attenuation.DUST_LAWS`. Single-component requires
``law`` (singular); two-component accepts either a single shared ``law`` or
separate ``law_bc`` and ``law_diff``. Grammar builds require explicit law
specification; flat-kwarg Parameter construction uses power_law defaults.
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
            "law": "calzetti",  # Shared law for discovery; grammar validates this later
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
        # String settings (e.g. law, law_bc, law_diff). No defaults are applied;
        # grammar requires explicit specification. Flat-kwarg builds get power_law
        # defaults in Parameters.
        settings: dict[str, str] = {}

        if dust_model == "two_component":
            # law (shared) XOR (law_bc AND law_diff) (per-screen). Handled here
            # rather than the generic per-setting loop below, since the two
            # forms are mutually exclusive, not independently required.
            law = kwargs.pop("law", None)
            law_bc = kwargs.pop("law_bc", None)
            law_diff = kwargs.pop("law_diff", None)
            if law is not None and (law_bc is not None or law_diff is not None):
                raise ValueError(
                    "dust.two_component(): grammar is ambiguous: cannot specify "
                    "both 'law' and 'law_bc'/'law_diff'. Use EITHER 'law' (shared "
                    "by both screens) OR both 'law_bc' and 'law_diff' (per-screen). "
                    "Example 1: dust.two_component(law='calzetti', ...) "
                    "Example 2: dust.two_component(law_bc='calzetti', "
                    "law_diff='power_law', ...)"
                )
            if law is not None:
                for name, value in (("law", law),):
                    if value not in DUST_LAWS:
                        raise ValueError(
                            f"dust.two_component({name}={value!r}): unknown "
                            f"attenuation law. Valid: {sorted(DUST_LAWS)}."
                        )
                settings["law"] = law
            elif law_bc is not None and law_diff is not None:
                for name, value in (("law_bc", law_bc), ("law_diff", law_diff)):
                    if value not in DUST_LAWS:
                        raise ValueError(
                            f"dust.two_component({name}={value!r}): unknown "
                            f"attenuation law. Valid: {sorted(DUST_LAWS)}."
                        )
                settings["law_bc"] = law_bc
                settings["law_diff"] = law_diff
            elif law_bc is not None or law_diff is not None:
                _pairs = (("law_bc", law_bc), ("law_diff", law_diff))
                given = [n for n, v in _pairs if v is not None]
                raise ValueError(
                    "dust.two_component(): requires BOTH 'law_bc' and 'law_diff' "
                    "for per-screen specification, or 'law' for a shared law. "
                    f"You gave: {given}. "
                    "Example 1: dust.two_component(law='calzetti', ...) "
                    "Example 2: dust.two_component(law_bc='calzetti', "
                    "law_diff='power_law', ...)"
                )
            else:
                raise TypeError(
                    "dust.two_component(): requires either 'law' (shared) or "
                    "both 'law_bc' and 'law_diff' (per-screen). "
                    f"Valid laws: {sorted(DUST_LAWS)}. "
                    "Example 1: dust.two_component(law='calzetti', ...) "
                    "Example 2: dust.two_component(law_bc='calzetti', "
                    "law_diff='power_law', ...)"
                )

        for s in setting_names:
            if s in kwargs:
                value = kwargs.pop(s)
                if value not in DUST_LAWS:
                    raise ValueError(
                        f"dust.{dust_model}({s}={value!r}): unknown attenuation "
                        f"law. Valid: {sorted(DUST_LAWS)}."
                    )
                settings[s] = value
            elif s in setting_defaults:
                # Fallback default if one is provided
                settings[s] = setting_defaults[s]
            else:
                # No default and not provided: required.
                raise TypeError(
                    f"dust.{dust_model}() missing required argument: {s!r}. "
                    f"Example: dust.{dust_model}({s}='calzetti', ...)."
                )
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
        default = setting_defaults.get(s, inspect.Parameter.empty)
        sig_params.append(
            inspect.Parameter(
                s,
                inspect.Parameter.KEYWORD_ONLY,
                default=default,
                annotation=str,
            )
        )
    if dust_model == "two_component":
        # law/law_bc/law_diff form an XOR group (see factory() above); shown
        # here as optional (UNSET) for IDE autocomplete, not enforced by the
        # signature itself — factory() enforces the actual XOR requirement.
        for law_kw in ("law", "law_bc", "law_diff"):
            sig_params.append(
                inspect.Parameter(
                    law_kw, inspect.Parameter.KEYWORD_ONLY, default=UNSET, annotation=str
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
    if dust_model == "two_component":
        doc_lines.append(
            "Law: ``law`` (shared by both screens) XOR both ``law_bc`` and "
            "``law_diff`` (per-screen). Required — no default. Each accepts "
            "any key registered in ``tengri.components.dust.attenuation.DUST_LAWS`` "
            "(e.g. ``'calzetti'``, ``'smc'``, ``'cardelli'``)."
        )
    else:
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
    if dust_model == "two_component":
        doc_lines.append("law : str, optional")
        doc_lines.append(
            "    Attenuation law shared by both screens. Mutually exclusive with law_bc/law_diff."
        )
        doc_lines.append("law_bc : str, optional")
        doc_lines.append("    Birth-cloud attenuation law. Requires law_diff too.")
        doc_lines.append("law_diff : str, optional")
        doc_lines.append("    Diffuse-ISM attenuation law. Requires law_bc too.")
    for s in setting_names:
        doc_lines.append(f"{s} : str")
        if s in setting_defaults:
            doc_lines.append(
                f"    Attenuation law name from DUST_LAWS. Default {setting_defaults[s]!r}."
            )
        else:
            doc_lines.append("    Attenuation law name from DUST_LAWS. Required (no default).")
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
# Note: builders now require explicit laws (no defaults). Grammar validation
# enforces this; flat-kwarg builds apply power_law defaults in Parameters.
single_component = _make_dust_factory(
    dust_model="single_component",
    short_params=_discover_attenuation_params("single_component"),
    setting_names=("law",),
    setting_defaults={},  # No defaults; grammar requires explicit law
)
two_component = _make_dust_factory(
    dust_model="two_component",
    short_params=_discover_attenuation_params("two_component"),
    # law/law_bc/law_diff are handled by the dedicated XOR branch in factory()
    # above, not the generic per-setting loop (they are mutually exclusive
    # forms, not independently required settings).
    setting_names=(),
    setting_defaults={},
)


def available() -> list[str]:
    """Return the list of top-level dust variants exposed."""
    return ["single_component", "two_component"]


__all__ = ["available", "emission", "single_component", "two_component"]


__dir__ = curated_dir(__all__)
