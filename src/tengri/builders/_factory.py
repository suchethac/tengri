# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Shared helpers for building config-dict factory callables.

Each ``tengri.builders.<component>`` module uses :func:`make_factory` to
synthesize one callable per variant. The helpers here keep the per-module
files focused on *what* varies (variant list, prefix, short-name rule);
signature and docstring attachment lives here.

The SFH module (:mod:`tengri.builders.sfh`) was the first consumer and
predates this helper; it carries its own equivalent inline. Future
unification can fold SFH onto these helpers when convenient.
"""

from __future__ import annotations

import inspect
import warnings
from collections.abc import Callable
from typing import Any

from tengri.parameters.sentinels import FIXED, FREE, WILDCARD_ALIAS

# Sentinel marking a parameter that was not specified at call time. We
# can't use ``None`` because ``None`` is a legitimate dict value users
# might want to pass through; can't use ``FREE`` / ``FIXED`` because those
# carry semantic meaning.
UNSET = object()


def make_factory(
    *,
    variant: str,
    short_params: list[str],
    qualname_prefix: str,
    module_name: str,
    short_doc: str = "",
    bool_flags: tuple[str, ...] = (),
    flag_param_map: dict[str, str] | None = None,
) -> Callable[..., dict]:
    """Build one factory callable for a single variant.

    Parameters
    ----------
    variant : str
        Variant name (becomes the ``'type'`` key in the emitted dict).
    short_params : list of str
        Short-form parameter names the variant accepts (matches what the
        grammar parser's ``_extract_short_name`` will look up).
    qualname_prefix : str
        Dotted prefix for ``__qualname__`` (e.g.
        ``"tengri.builders.radio"``).
    module_name : str
        ``__module__`` for the factory function.
    short_doc : str, optional
        One-line description placed at the top of the docstring.
    bool_flags : tuple of str, optional
        Names of boolean sub-keys this variant accepts (e.g. ``patchy``,
        ``dla`` for IGM). They're surfaced in the signature with default
        ``False`` and only included in the output dict when set ``True``.
    flag_param_map : dict, optional
        Map ``short_param`` → ``flag_name``: providing the param
        auto-enables the flag. Useful for IGM where setting
        ``bubble_mpc=...`` implies ``patchy=True``.
    """
    flag_param_map = flag_param_map or {}

    def factory(**kwargs: Any) -> dict:
        wildcard = _pop_wildcard(variant, kwargs)
        if wildcard not in (FREE, FIXED):
            raise ValueError(
                f"{variant}(all_params=...): expected FREE or FIXED, got "
                f"{wildcard!r}. Use tengri.FREE or tengri.FIXED."
            )
        flag_values: dict[str, bool] = {f: bool(kwargs.pop(f, False)) for f in bool_flags}
        unknown = [k for k in kwargs if k not in short_params]
        if unknown:
            valid_kwargs = [*bool_flags, *short_params]
            raise TypeError(
                f"{variant}() got unexpected keyword arguments: {unknown}. "
                f"Valid: {valid_kwargs}. "
                f"(Pass ``all_params=FREE`` or ``all_params=FIXED`` to set the policy.)"
            )

        # Auto-enable flag when a flag-conditional param was given.
        for short, flag in flag_param_map.items():
            if short in kwargs and kwargs[short] is not UNSET:
                flag_values[flag] = True

        out: dict[str, Any] = {"type": variant, WILDCARD_ALIAS: wildcard}
        for flag, value in flag_values.items():
            if value:
                out[flag] = True
        for short in short_params:
            if short in kwargs and kwargs[short] is not UNSET:
                out[short] = kwargs[short]
        return out

    sig_params = [
        inspect.Parameter(
            "all_params",
            inspect.Parameter.KEYWORD_ONLY,
            default=FIXED,
            annotation=Any,
        ),
    ]
    for flag in bool_flags:
        sig_params.append(
            inspect.Parameter(
                flag,
                inspect.Parameter.KEYWORD_ONLY,
                default=False,
                annotation=bool,
            )
        )
    for short in short_params:
        sig_params.append(
            inspect.Parameter(
                short,
                inspect.Parameter.KEYWORD_ONLY,
                default=UNSET,
                annotation=Any,
            )
        )
    factory.__signature__ = inspect.Signature(sig_params, return_annotation=dict)
    factory.__name__ = variant
    factory.__qualname__ = f"{qualname_prefix}.{variant}"
    factory.__module__ = module_name

    lines: list[str] = []
    lines.append(f"Build a config dict for the {variant!r} variant.")
    lines.append("")
    if short_doc:
        lines.append(short_doc)
        lines.append("")
    lines.append("Parameters")
    lines.append("----------")
    lines.append("all_params : sentinel, optional")
    lines.append(
        "    Wildcard policy for parameters not explicitly named. ``FREE`` "
        "makes them fit; ``FIXED`` (default) pins them to their registry "
        "center. Matches the ``'all_params'`` key in the dict grammar."
    )
    for flag in bool_flags:
        lines.append(f"{flag} : bool, optional")
        lines.append(
            f"    Toggle sub-feature {flag!r}; when ``True``, the relevant "
            "free parameters become activatable by the parser. Defaults to "
            "``False``."
        )
    for short in short_params:
        flag_note = (
            f" (auto-enables ``{flag_param_map[short]}=True``)" if short in flag_param_map else ""
        )
        lines.append(f"{short} : Distribution, sentinel, or scalar, optional")
        lines.append(
            f"    Override the registry default prior for the matching free parameter.{flag_note}"
        )
    lines.append("")
    lines.append("Returns")
    lines.append("-------")
    lines.append("dict")
    lines.append("    Config dict matching the :meth:`SEDModel.build` grammar.")
    factory.__doc__ = "\n".join(lines)
    return factory


def _pop_wildcard(variant: str, kwargs: dict[str, Any]) -> Any:
    """Pop the wildcard kwarg, supporting ``all_params=`` only.

    The canonical builder name for the wildcard policy is ``all_params=``,
    mirroring the dict grammar's ``'all_params'`` key. The retired aliases
    ``defaults=`` and ``_=`` raise ``TypeError`` naming ``all_params=`` as
    the required spelling.

    Raises ``TypeError`` if ``defaults=`` or ``_=`` are passed.
    """
    has_canonical = "all_params" in kwargs
    has_deprecated = "defaults" in kwargs
    has_legacy = "_" in kwargs

    # Reject deprecated aliases with hard errors naming the replacement
    if has_deprecated:
        raise TypeError(
            f"The `defaults=` alias has been retired; the wildcard parameter is "
            f"`all_params=`. Write all_params=FREE instead of defaults=FREE."
        )
    if has_legacy:
        raise TypeError(
            f"The `_=` alias has been retired; the wildcard parameter is "
            f"`all_params=`. Write all_params=FREE instead of _=FREE."
        )

    # Pop the canonical form or return default
    if has_canonical:
        return kwargs.pop("all_params")
    return FIXED


def short_form(full_name: str, *, prefixes: tuple[str, ...]) -> str:
    """Strip a prefix and return the short-form name.

    The grammar parser's ``tengri.parameters.groups._extract_short_name``
    strips the component prefix to derive the short form (e.g.
    ``radio_q_ir`` → ``q_ir``). Factories must use the same convention.
    """
    for p in prefixes:
        if full_name.startswith(p):
            return full_name[len(p) :]
    return full_name
