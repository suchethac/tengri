# SPDX-License-Identifier: BSD-3-Clause
"""Introspection registry for tengri free parameters.

Walks every per-component ``_params.py`` module under :mod:`tengri.components`,
:mod:`tengri.observation`, and :mod:`tengri.parameters._shared`, then exposes
a single, queryable view of every :class:`~tengri.protocols.component.ParamDeclaration`
the codebase declares. The underlying data ownership is unchanged — each
component/module still owns its own ``_params.py`` (the decentralization that
landed pre-ADR-0005). This module just gives users a single API to ask:

- **What parameters exist?** ``tengri.list_parameters()``
- **Where does this parameter live?** ``tengri.describe_parameter("dust_tau_v")``
- **Which components own a given prefix?** ``registry().owners_of_prefix("agn_")``

Conscious choice: the registry is *flat*, not configuration-aware. A
parameter like ``dust_tau_bc`` lives in ``DustSEDComponent`` whether or
not the two-component dust model is enabled in any specific
:class:`tengri.SEDModel`. Per-model views are constructed from
``model.spec.free_params`` instead.

See ADR-0005 and Step E (Observation cleanup) for the full rationale.
"""

from __future__ import annotations

import importlib
import pkgutil
import warnings
from typing import NamedTuple

from tengri.protocols.component import ParamDeclaration

__all__ = [
    "ParameterRecord",
    "as_param_map",
    "describe_parameter",
    "list_parameters",
    "recipe_parameters",
    "registry",
]


class ParameterRecord(NamedTuple):
    """Where a free parameter lives: name, prior, description, units, owner module, owner tuple."""

    name: str
    prior: object
    description: str
    units: str
    owner: str  # fully-qualified module path of the ``_params.py`` that exports it
    group: str  # tuple attribute on ``owner``, e.g. "PARAMS" or "ATTENUATION_PARAMS"


def _walk_param_modules() -> dict[str, ParameterRecord]:
    """Walk every ``_params.py`` under components, observation, and parameters._shared.

    First-wins on name collisions (deterministic via ``pkgutil.walk_packages`` order).
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
        except ImportError as exc:
            # A component may legitimately be unimportable when an optional
            # dependency is absent. Degrade rather than break introspection —
            # but say so: a silently vanishing component reads as "this
            # parameter does not exist" and has shipped as a bug twice
            # (#1165, #1179). Anything that is not an ImportError is a real
            # defect and propagates.
            warnings.warn(
                f"tengri: parameters from {module_info.name!r} are missing from the "
                f"registry because the module could not be imported ({exc}). Any "
                f"parameter it declares will be reported as unknown.",
                RuntimeWarning,
                stacklevel=2,
            )
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
                    units=decl.units,
                    owner=module_info.name,
                    group=attr_name,
                )

    # Observation _params.py module (noise model parameters).
    # As of Step E, noise parameters are owned by the observation module,
    # not the shared parameters module.
    # These are first-party modules with no optional dependency: if one fails
    # to import that is a defect, and swallowing it would silently yield an
    # incomplete registry. Let it raise.
    from tengri.observation import _params as obs_params_module

    for attr_name in dir(obs_params_module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(obs_params_module, attr_name)
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
                units=decl.units,
                owner="tengri.observation._params",
                group=attr_name,
            )

    # Shared parameters: redshift, met_logzsol, sigma_v_kms.
    # These are declared cleanly in tengri.parameters._shared.PARAMS
    # as of ADR-0005 follow-up #1. Import and walk like a component.
    from tengri.parameters import _shared as shared_module

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
                units=decl.units,
                owner="tengri.parameters._shared",
                group=attr_name,
            )

    # Legacy ``_NON_SFH_PARAMS`` bucket: provides ``noise_frac_cal`` and
    # ``noise_dof`` which aren't yet declared via the ParamDeclaration
    # path. 4-tuple shape ``(description, bound_check, bound_error, prior)``.
    from tengri.parameters._builders import _NON_SFH_PARAMS

    if _NON_SFH_PARAMS:
        for name, payload in _NON_SFH_PARAMS.items():
            if name in out:
                continue
            description, _bcheck, _berr, prior = payload
            out[name] = ParameterRecord(
                name=name,
                prior=prior,
                description=description,
                units="",
                owner="tengri.parameters._builders",
                group="_NON_SFH_PARAMS",
            )

    # ``neb_xid`` orphan from AGN module: kept in _builders._AGN_EXTRAS
    # for the Feltre NLR backend. Not part of any component's _params.py
    # but must be registered for the parameter system to function.
    from tengri.parameters._builders import _AGN_EXTRAS

    if _AGN_EXTRAS:
        for name, payload in _AGN_EXTRAS.items():
            if name in out:
                continue
            description, _bcheck, _berr, prior = payload
            out[name] = ParameterRecord(
                name=name,
                prior=prior,
                description=description,
                units="",
                owner="tengri.parameters._builders",
                group="_AGN_EXTRAS",
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


def as_param_map() -> dict[str, tuple[str, float, float, str]]:
    """Return parameter map as {public: (internal, scale, offset, units)}.

    This is the canonical view for parameter translation. Each entry maps
    a public parameter name to a 4-tuple of:

    - internal name (used internally in computations)
    - scale factor (multiplicative conversion)
    - offset (additive conversion: internal = scale * public + offset)
    - units string (e.g. "Myr", "erg/s/Hz", "Z/Zsun")

    For parameters with identity translation (internal == public), the
    scale is 1.0, offset is 0.0, and units still documents the parameter.

    This function returns the base translation map without SFH-specific
    logic (which is handled by :func:`tengri.parameters.translate._build_param_map`).

    Returns
    -------
    dict[str, tuple[str, float, float, str]]
        A mapping from public parameter name to (internal, scale, offset, units).
        The map includes all declared parameters from the registry. The
        ``tengri.parameters.translate`` module applies SFH resolution
        and dust-model selection on top of this base.

    Notes
    -----
    For most parameters, this returns identity mappings with units
    inherited from the component declarations. Some parameters may need
    unit conversion; those are defined explicitly elsewhere (e.g., in
    ``tengri.parameters._shared.py``) with non-unit-scale entries.
    """
    reg = registry()
    result: dict[str, tuple[str, float, float, str]] = {}
    for name, record in reg.items():
        # Default identity mapping: internal == public, scale=1.0, offset=0.0
        # The translate module will override specific entries with unit conversions.
        result[name] = (name, 1.0, 0.0, record.units)
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
                "sfh": {"type": "dpl", "all_params": FREE},
                "dust": {"type": "two_component", "law_bc": "calzetti", "all_params": FREE},
                "neb": {"type": "cue", "all_params": FIXED},
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
    **Does not require SSP data.** Unlike :meth:`~tengri.SEDModel.build`,
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
    from tengri.parameters.groups import parse_groups

    # Translate recipe to Parameters (no SSP data needed)
    params = parse_groups(**recipe_dict)

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
    from tengri.utils.strings import closest

    return closest(target, options, max_distance=2)
