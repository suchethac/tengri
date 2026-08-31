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
from collections.abc import Callable
from typing import Any

from tengri.parameters.priors import Fixed, _is_default_fixed
from tengri.parameters.sentinels import DEFAULT, FREE, WILDCARD_ALIAS

# Sentinel marking a parameter that was not specified at call time. We
# can't use ``None`` because ``None`` is a legitimate dict value users
# might want to pass through; can't use ``FREE`` / ``Fixed(DEFAULT)`` because
# those carry semantic meaning.
UNSET = object()

#: The wildcard's default value: every factory signature (and ``_pop_wildcard``'s
#: fallback) shares this one instance rather than constructing a fresh
#: ``Fixed(DEFAULT)`` per call/signature.
_DEFAULT_WILDCARD = Fixed(DEFAULT)


def _validate_wildcard(label: str, wildcard: Any) -> None:
    """Validate a builder factory's ``all_params=`` / ``other_params=`` value.

    Accepts ``FREE`` or an unresolved ``Fixed(DEFAULT)`` token (the
    per-parameter "pin at the registry default" spelling, also legal in
    the wildcard slot). Rejects everything else, including a concrete
    ``Fixed(v)`` and a bare ``DEFAULT`` sentinel -- the latter gets its own
    message pointing at ``Fixed(DEFAULT)``.

    Shared by every ``tengri.builders.*`` factory (``_factory.py``'s
    ``make_factory``, and the SFH/AGN/dust modules' own inline factories) so
    the four near-identical copies of this check stay in exact sync.

    Parameters
    ----------
    label : str
        The factory call site as it should read in the error, e.g.
        ``"sfh.dpl"`` or ``"agn.composable"``.
    wildcard : object
        The value popped from ``all_params=`` / ``other_params=``.

    Raises
    ------
    ValueError
        If ``wildcard`` is not ``FREE`` or ``Fixed(DEFAULT)``.
    """
    if wildcard is FREE or _is_default_fixed(wildcard):
        return
    if wildcard is DEFAULT:
        raise ValueError(
            f"{label}(all_params=...): bare DEFAULT is not a valid wildcard value. "
            f"DEFAULT is only legal as the argument of Fixed(...); did you mean "
            f"all_params=Fixed(DEFAULT)?"
        )
    if isinstance(wildcard, Fixed):
        raise ValueError(
            f"{label}(all_params=...): expected FREE or Fixed(DEFAULT), got "
            f"{wildcard!r}. A concrete Fixed(v) cannot be the wildcard value: one "
            f"literal value cannot apply across different parameters -- give this "
            f"parameter its own keyword instead."
        )
    raise ValueError(
        f"{label}(all_params=...): expected FREE or Fixed(DEFAULT), got "
        f"{wildcard!r}. Use tengri.FREE or tengri.Fixed(tengri.DEFAULT)."
    )


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
        _validate_wildcard(variant, wildcard)
        flag_values: dict[str, bool] = {f: bool(kwargs.pop(f, False)) for f in bool_flags}
        unknown = [k for k in kwargs if k not in short_params]
        if unknown:
            valid_kwargs = [*bool_flags, *short_params]
            raise TypeError(
                f"{variant}() got unexpected keyword arguments: {unknown}. "
                f"Valid: {valid_kwargs}. "
                f"(Pass ``all_params=FREE`` or ``all_params=Fixed(DEFAULT)`` -- or the "
                f"synonym ``other_params=`` -- to set the policy.)"
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
            default=_DEFAULT_WILDCARD,
            annotation=Any,
        ),
        inspect.Parameter(
            "other_params",
            inspect.Parameter.KEYWORD_ONLY,
            default=UNSET,
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
    lines.append("all_params : sentinel or Fixed, optional")
    lines.append(
        "    Wildcard policy for parameters not explicitly named. ``FREE`` "
        "makes them fit; ``Fixed(DEFAULT)`` (default) pins them to their registry "
        "center. Matches the ``'all_params'`` key in the dict grammar."
    )
    lines.append("other_params : sentinel, optional")
    lines.append(
        "    Exact synonym of ``all_params``; give only one. Reads best written "
        'last, after explicit per-parameter overrides, as "the others". Matches '
        "the ``'other_params'`` key in the dict grammar."
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
    """Pop the wildcard kwarg, supporting ``all_params=`` and its exact synonym
    ``other_params=``.

    The canonical builder name for the wildcard policy is ``all_params=``,
    mirroring the dict grammar's ``'all_params'`` key; ``other_params=`` is an
    exact synonym, mirroring the grammar's ``'other_params'`` key (see
    ``tengri.parameters.sentinels.WILDCARD_ALIAS_OTHER``). Only one of the two
    may be given. The retired aliases ``defaults=`` and ``_=`` raise
    ``TypeError`` naming ``all_params=`` as the required spelling.

    Raises ``TypeError`` if ``defaults=`` or ``_=`` are passed. Raises
    ``ValueError`` if both ``all_params=`` and ``other_params=`` are passed.
    """
    has_canonical = "all_params" in kwargs
    has_other = "other_params" in kwargs
    has_deprecated = "defaults" in kwargs
    has_legacy = "_" in kwargs

    # Reject deprecated aliases with hard errors naming the replacement
    if has_deprecated:
        raise TypeError(
            "The `defaults=` alias has been retired; the wildcard parameter is "
            "`all_params=`. Write all_params=FREE instead of defaults=FREE."
        )
    if has_legacy:
        raise TypeError(
            "The `_=` alias has been retired; the wildcard parameter is "
            "`all_params=`. Write all_params=FREE instead of _=FREE."
        )
    # The two spellings set the same policy; giving both is a contradiction,
    # not a redundancy to silently resolve.
    if has_canonical and has_other:
        raise ValueError(
            "`all_params=` and `other_params=` are synonyms for the same wildcard "
            "parameter; give only one. Write all_params=FREE or other_params=FREE, "
            "not both."
        )

    # Pop the canonical form (or its synonym) or return default
    if has_canonical:
        return kwargs.pop("all_params")
    if has_other:
        return kwargs.pop("other_params")
    return _DEFAULT_WILDCARD


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
