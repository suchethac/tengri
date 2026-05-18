# SPDX-License-Identifier: BSD-3-Clause
"""Introspection registry for tengri free parameters.

Walks every per-component ``_params.py`` module under
:mod:`tengri.components` and exposes a single, queryable view of every
:class:`~tengri.core.component.ParamDeclaration` the codebase declares.
The underlying data ownership is unchanged — each component still owns
its own ``_params.py`` (the decentralisation that landed pre-ADR-0005).
This module just gives users a single API to ask:

- **What parameters exist?** ``tengri.list_parameters()``
- **Where does this parameter live?** ``tengri.describe_parameter("dust_tau_v")``
- **Which components own a given prefix?** ``registry().owners_of_prefix("agn_")``

Conscious choice: the registry is *flat*, not configuration-aware. A
parameter like ``dust_tau_bc`` lives in ``DustSEDComponent`` whether or
not the two-component dust model is enabled in any specific
:class:`tengri.SEDModel`. Per-model views are constructed from
``model.spec.free_params`` instead.

See ADR-0005 for the full rationale.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import NamedTuple

from tengri.core.component import ParamDeclaration

__all__ = [
    "ParameterRecord",
    "describe_parameter",
    "list_parameters",
    "registry",
]


class ParameterRecord(NamedTuple):
    """One free parameter and the module that owns it.

    Mirrors :class:`tengri.core.component.ParamDeclaration` (the
    declaration shape) with an extra ``owner`` field naming the
    ``_params.py`` module that exports it. Used by the registry to
    answer "where does this parameter live?" queries.

    Fields
    ------
    name : str
        Parameter name (e.g. ``"dust_tau_v"``). Same convention as
        :class:`ParamDeclaration.name`.
    prior : object
        Default prior (:class:`tengri.parameters.priors.Distribution`).
        Identical to :class:`ParamDeclaration.prior`.
    description : str
        One-line human-readable description.
    owner : str
        Fully-qualified module path of the ``_params.py`` that exports
        this parameter, e.g.
        ``"tengri.components.dust._params"``.
    group : str
        The tuple attribute name within ``owner`` (e.g. ``"PARAMS"``,
        ``"ATTENUATION_PARAMS"``, ``"SINGLE_COMPONENT_PARAMS"``).
        Useful when a component splits its parameters across multiple
        named tuples to indicate which configuration toggles them.
    """

    name: str
    prior: object
    description: str
    owner: str
    group: str


def _walk_param_modules() -> dict[str, ParameterRecord]:
    """Walk every ``_params.py`` under ``tengri.components`` plus the
    legacy ``_NON_SFH_PARAMS`` bucket in :mod:`tengri.parameters._param_defs`,
    and collect every declared free parameter.

    Build order is deterministic (``pkgutil.walk_packages`` returns
    modules in package-traversal order). When the same parameter name
    appears in more than one source, the *first* occurrence wins —
    matching how the legacy aggregator at
    :mod:`tengri.parameters._param_defs` resolves duplicates.

    Parameters declared in component ``_params.py`` modules use the
    :class:`ParamDeclaration` shape directly. Parameters in the legacy
    ``_NON_SFH_PARAMS`` bucket use a 4-tuple
    ``(description, bound_check, bound_error, default_prior)`` and are
    adapted in-place; their ``owner`` field reads
    ``"tengri.parameters._param_defs:_NON_SFH_PARAMS"`` so introspection
    can distinguish them from cleanly-migrated component parameters.

    Private. Re-evaluated lazily by :func:`registry` and cached.
    """
    import tengri.components as components_pkg

    out: dict[str, ParameterRecord] = {}
    for module_info in pkgutil.walk_packages(
        components_pkg.__path__, prefix=components_pkg.__name__ + "."
    ):
        if not module_info.name.endswith("._params"):
            continue
        try:
            mod = importlib.import_module(module_info.name)
        except Exception:
            # A subpackage might fail to import in some environments
            # (e.g. optional dep missing). Skip rather than break
            # introspection — the registry is a best-effort view.
            continue
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            attr = getattr(mod, attr_name)
            if not isinstance(attr, tuple):
                continue
            if not all(isinstance(x, ParamDeclaration) for x in attr):
                continue
            for decl in attr:
                if decl.name in out:
                    continue  # first-wins; matches legacy aggregator
                out[decl.name] = ParameterRecord(
                    name=decl.name,
                    prior=decl.prior,
                    description=decl.description,
                    owner=module_info.name,
                    group=attr_name,
                )

    # Legacy shared bucket: redshift, met_logzsol, noise_*, sigma_v_kms.
    # These have not yet been migrated to a component-owned _params.py.
    # See ADR-0005, "Known gaps" section.
    try:
        from tengri.parameters import _param_defs as _legacy
    except Exception:
        _legacy = None  # type: ignore[assignment]
    if _legacy is not None:
        legacy_bucket = getattr(_legacy, "_NON_SFH_PARAMS", {}) or {}
        for name, payload in legacy_bucket.items():
            if name in out:
                continue
            # Legacy 4-tuple shape: (description, bound_check, bound_error, prior).
            description, _bcheck, _berr, prior = payload
            out[name] = ParameterRecord(
                name=name,
                prior=prior,
                description=description,
                owner="tengri.parameters._param_defs",
                group="_NON_SFH_PARAMS",
            )
    return out


_CACHE: dict[str, ParameterRecord] | None = None


def registry() -> dict[str, ParameterRecord]:
    """Return the full parameter registry as a name → record map.

    Lazily built on first call, then cached for the process lifetime.
    The map is a fresh ``dict`` returned by reference — callers should
    not mutate it. To force a rebuild (useful after editing
    ``_params.py`` in a live REPL), call :func:`_clear_cache`.
    """
    global _CACHE
    if _CACHE is None:
        _CACHE = _walk_param_modules()
    return _CACHE


def _clear_cache() -> None:
    """Drop the cached registry. Re-imports happen on the next call."""
    global _CACHE
    _CACHE = None


def list_parameters(prefix: str | None = None) -> list[str]:
    """List every free-parameter name in the registry, optionally filtered.

    Parameters
    ----------
    prefix : str, optional
        If given, only return names starting with this prefix
        (e.g. ``"dust_"``, ``"agn_"``). Useful for surveying a
        physics domain.

    Returns
    -------
    list of str
        Parameter names, sorted alphabetically for stable output.

    Examples
    --------
    >>> import tengri
    >>> tengri.list_parameters(prefix="radio_")[:3]
    ['radio_alpha_agn', 'radio_alpha_ff', 'radio_alpha_inj']
    """
    names = registry().keys()
    if prefix is not None:
        names = [n for n in names if n.startswith(prefix)]
    return sorted(names)


def describe_parameter(name: str) -> ParameterRecord:
    """Return the :class:`ParameterRecord` for ``name``.

    Raises
    ------
    KeyError
        If ``name`` is not in the registry. The message lists the
        Levenshtein-closest known parameter as a "Did you mean: ..."
        hint, matching the style used by
        :func:`tengri.forward.orchestrator.validate_pipeline`.

    Examples
    --------
    >>> import tengri
    >>> rec = tengri.describe_parameter("dust_tau_v")
    >>> rec.owner
    'tengri.components.dust._params'
    """
    reg = registry()
    if name in reg:
        return reg[name]
    hint = _closest(name, reg.keys())
    suffix = f" (Did you mean: {hint!r}?)" if hint is not None else ""
    raise KeyError(f"No parameter named {name!r} in the registry.{suffix}")


def _closest(target: str, options) -> str | None:
    """Closest option by Levenshtein distance (≤ 2). None if no match."""

    def lev(a: str, b: str) -> int:
        if a == b:
            return 0
        if not a:
            return len(b)
        if not b:
            return len(a)
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            curr = [i]
            for j, cb in enumerate(b, 1):
                curr.append(min(curr[-1] + 1, prev[j] + 1, prev[j - 1] + (0 if ca == cb else 1)))
            prev = curr
        return prev[-1]

    best_name: str | None = None
    best_dist = 3
    for k in options:
        d = lev(target, k)
        if d < best_dist:
            best_dist = d
            best_name = k
    return best_name
