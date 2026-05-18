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
    "as_param_map",
    "describe_parameter",
    "list_parameters",
    "recipe_parameters",
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
    """Walk every ``_params.py`` under ``tengri.components`` and
    ``tengri.parameters._shared``, and collect every declared free parameter.

    Build order is deterministic (``pkgutil.walk_packages`` returns
    modules in package-traversal order). When the same parameter name
    appears in more than one source, the *first* occurrence wins —
    matching how the legacy aggregator at
    :mod:`tengri.parameters._param_defs` resolves duplicates.

    Parameters declared in component ``_params.py`` modules and
    ``tengri.parameters._shared`` use the :class:`ParamDeclaration` shape
    directly. The legacy ``_NON_SFH_PARAMS`` bucket in
    :mod:`tengri.parameters._param_defs` is a 4-tuple
    ``(description, bound_check, bound_error, default_prior)`` and is
    kept for backwards compatibility; it is derived from
    ``_shared.PARAMS``, so introspection should prefer the _shared
    source when both are present.

    Private. Re-evaluated lazily by :func:`registry` and cached.
    """
    import tengri.components as components_pkg

    out: dict[str, ParameterRecord] = {}

    # Walk all component _params.py modules.
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

    # Shared parameters: redshift, met_logzsol, noise_*, sigma_v_kms.
    # These are declared cleanly in tengri.parameters._shared.PARAMS
    # as of ADR-0005 follow-up #1. Import and walk like a component.
    try:
        from tengri.parameters import _shared as shared_module
    except Exception:
        shared_module = None  # type: ignore[assignment]
    if shared_module is not None:
        for attr_name in dir(shared_module):
            if attr_name.startswith("_"):
                continue
            attr = getattr(shared_module, attr_name)
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
                    owner="tengri.parameters._shared",
                    group=attr_name,
                )

    # Fallback: legacy shared bucket from _param_defs for any params
    # that didn't get picked up above (defensive; all should now be in
    # _shared.PARAMS).
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


def as_param_map() -> dict[str, tuple[str, float, float]]:
    """Return all registry parameters as a translation map.

    Derives a ``{public_name: (internal_name, scale, offset)}`` dictionary
    from the parameter declarations in the registry. For parameters with no
    declared unit conversion, defaults to identity mapping
    ``(name, 1.0, 0.0)``.

    This is the canonical source for parameter translation; ``translate._build_param_map``
    wraps this function, applying SFH-resolution and dust-model selection on top.

    Returns
    -------
    dict
        ``{public_name: (internal_name, scale, offset)}`` for every parameter
        in the registry. Non-SFH parameters use identity translation unless
        overridden in the unit-conversion map (see Notes).

    Notes
    -----
    Unit conversions for a fixed set of parameters are defined in
    :mod:`tengri.parameters.translate`:

    - ``met_logzsol``: ``log10(Z/Zsun)`` → ``log10(Z)`` with ``LOG10_ZSUN`` offset
    - ``neb_logZ_gas``: ``log10(Z_gas/Zsun)`` → ``log10(Z)`` with ``LOG10_ZSUN`` offset
    - ``dust_tau_bc``, ``dust_tau_diff``: internal rename only, no scale/offset change
    - ``redshift``, ``noise_frac_cal``, ``noise_dof``, ``sigma_v_kms``: identity
    - All other registered parameters: identity mapping ``(name, 1.0, 0.0)``

    The offset ``LOG10_ZSUN = -1.8477116556169435`` (Asplund 2009) bridges
    the convention mismatch between public-facing relative metallicity and
    internal absolute metallicity.

    See Also
    --------
    translate._build_param_map : Applies SFH resolution and dust-model selection.
    """
    # Import locally to avoid circular deps and keep translate.py free from
    # transitive physics-module imports at module level.
    from tengri.parameters import translate as translate_module

    reg = registry()
    result: dict[str, tuple[str, float, float]] = {}

    # Start with the manually-defined unit conversions for shared/sentinel params
    result.update(translate_module._NON_SFH_PARAM_MAP)

    # Add identity mappings for all remaining parameters in the registry
    # (parameters not in _NON_SFH_PARAM_MAP and not in _PARAMS_WITH_UNIT_CONVERSION).
    for pub_name in reg:
        if pub_name not in result:
            # Identity mapping: no unit conversion
            result[pub_name] = (pub_name, 1.0, 0.0)

    return result


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


def recipe_parameters(recipe_dict: dict, free_only: bool = True) -> list[ParameterRecord]:
    """Introspect a recipe dict and return the parameters it activates.

    Takes a nested-dict recipe (output of e.g. :func:`tengri.recipes.star_forming_photometry()`)
    and returns a sorted list of :class:`ParameterRecord` objects for each
    parameter that the recipe would activate — WITHOUT requiring SSP data or
    building an :class:`~tengri.SEDModel`.

    Parameters
    ----------
    recipe_dict : dict
        A recipe dictionary matching the format of :mod:`tengri.recipes`.
        Example::

            {
                "sfh": {"type": "dpl", "*": FREE},
                "dust": {"type": "two_component", "law_bc": "calzetti", "*": FREE},
                "neb": {"type": "cue", "*": FIXED},
                "redshift": Uniform(0.01, 6.0),
            }

    free_only : bool, optional
        If True (default), return only the free parameters (entries with
        non-fixed priors). If False, return all parameters the recipe
        activates (including FIXED ones). Default is True.

    Returns
    -------
    list of ParameterRecord
        A sorted list of :class:`ParameterRecord` objects corresponding to
        the parameters that the recipe activates. Sorted by parameter name.

    Raises
    ------
    ValueError
        If the recipe dict is invalid (e.g., unknown group, unknown type,
        invalid parameter name).

    Notes
    -----
    **Does not require SSP data.** Unlike :class:`~tengri.SEDModel.from_groups()`,
    which needs SSP data to build the full model, this function only translates
    the recipe structure to a :class:`~tengri.Parameters` object and introspects
    its parameter names — the heaviest operation is a pure-Python dict traversal.

    **Resolves sentinels.** FREE and FIXED sentinels are expanded to their
    registry defaults; if ``free_only=True``, FIXED entries are filtered out.

    Examples
    --------
    List all free parameters activated by a recipe::

        >>> from tengri import recipes, recipe_parameters
        >>> recipe = recipes.star_forming_photometry()
        >>> params = recipe_parameters(recipe)
        >>> len(params)
        14
        >>> params[0].name
        'agn_a_spin'

    Or include fixed parameters::

        >>> all_params = recipe_parameters(recipe, free_only=False)
        >>> len(all_params) > len(params)
        True

    See Also
    --------
    ~tengri.recipes : Curated recipe functions.
    describe_parameter : Look up a single parameter by name.
    """
    from tengri.parameters.parameters import Parameters

    # Translate recipe to Parameters (no SSP data needed)
    params = Parameters.from_groups(**recipe_dict)

    # Get the list of parameter names to introspect
    if free_only:
        param_names = params.free_params
    else:
        param_names = params.all_params

    # Build ParameterRecord list by looking up each name in the registry
    reg = registry()
    records: list[ParameterRecord] = []

    for name in param_names:
        if name in reg:
            records.append(reg[name])
        # If name is not in registry, skip it (e.g., structural settings
        # like mean_sfh_type, dust_model, etc.)

    # Sort by parameter name for stable output
    records.sort(key=lambda r: r.name)
    return records


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
