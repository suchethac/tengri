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
    # The distribution ``FREE`` expands to. ``prior`` is the registry *default*
    # (usually Fixed); ``free_prior`` is the admissible range. None means the
    # parameter declares no defensible range, and FREE refuses rather than
    # silently leaving it pinned (#1264). Last field, so positional
    # construction of the historical 6-tuple keeps working.
    free_prior: object = None


#: Units implied by a parameter-name suffix. The naming contract
#: (NAMING_CONTRACT §3) requires a unit-bearing SFH parameter to state its unit
#: in its own name, so reading the suffix is transcription, not inference.
#: ``ParamDef`` carries no units field, so this is the only source available
#: for SFH parameters.
#: Ordered longest-meaning-first: a *semantic stem* names the whole quantity
#: (``log_total_mass``), a *bare suffix* only names its unit (``_gyr``). Stems
#: must be tried first or ``sfh_db_log_sfr_inst`` would match nothing while
#: ``sfh_snorm_burst_burst_sfr`` matched ``_yr`` inside "Msun/yr".
#:
#: Log quantities declare ``log10(<unit>)``, not ``<unit>``: ``log10(M/Msun)``
#: is dimensionless, and saying "Msun" would invite exactly the units error
#: this project keeps finding. ``dex`` is used when the log is of a ratio or
#: an offset, where there is no underlying unit to name.
_SUFFIX_UNITS: tuple[tuple[str, str], ...] = (
    # semantic stems (more specific — must precede the bare unit suffixes)
    ("log_total_mass", "log10(Msun)"),
    ("log_sfr_inst", "log10(Msun/yr)"),
    ("burst_sfr", "Msun/yr"),
    ("met_logzsol_scatter", "dex"),
    # bare unit suffixes
    ("_gyr", "Gyr"),
    ("_myr", "Myr"),
    ("_kms", "km/s"),
    ("_km_s", "km/s"),
    ("_yr", "yr"),
)

#: Patterns whose index varies, so a fixed suffix cannot catch them.
_PATTERN_UNITS: tuple[tuple[str, str], ...] = (
    # "log10 flex bin SFR ratio N (controls bin width)" — a log ratio.
    (r"_flex_\d+$", "dex"),
)


def _units_from_name(name: str) -> str:
    """Units implied by a parameter name, or ``""`` if none applies.

    Notes
    -----
    Inference-from-name is how every model-registry parameter (SFH, MET) gets
    its units — those registries have no per-parameter ``units`` field, unlike
    the component ``_params.py`` declarations. Before the stems above existed,
    that meant only the five bare unit suffixes were recognized, so all 137
    ``sfh_*`` parameters between them declared just ``Gyr``/``Myr``/``km/s``
    and every ``log_total_mass`` declared nothing (#1296).
    """
    import re

    for suffix, units in _SUFFIX_UNITS:
        if name.endswith(suffix):
            # A bare unit suffix on a log-valued name states the unit of the
            # quantity *inside* the log, and log10(t/Myr) is dimensionless.
            # `sfh_burst_log_tmax_myr` declared "Myr" on exactly that mistake.
            # The semantic stems above already carry their own log10(...) form,
            # so only the bare suffixes need wrapping.
            if units.startswith("log10(") or units == "dex":
                return units
            return f"log10({units})" if _is_log_valued(name) else units
    for pattern, units in _PATTERN_UNITS:
        if re.search(pattern, name):
            return units
    return ""


def _is_log_valued(name: str) -> bool:
    """True when the parameter's *value* is a logarithm, from its name alone."""
    import re

    return bool(re.search(r"(^|_)l(og|g)[a-z0-9_]", name, flags=re.I))


#: Registries that own their parameters' *translation* as well as their
#: declaration, via a per-model ``internal_param_map``. Their parameters belong
#: in the introspection registry (so ``describe_parameter`` can find them) but
#: must NOT contribute identity entries to :func:`as_param_map`: the real
#: mapping is model-dependent and often non-identity — ``met_logzsol_0`` maps to
#: ``log_z_abs_initial`` with a ``-LOG10_ZSUN`` offset, and an identity entry
#: alongside it raises ``ParameterMapError`` for conflicting mappings.
_TRANSLATION_OWNED_ELSEWHERE: frozenset[str] = frozenset(
    {
        "tengri.components.stellar.sfh.registry",
        "tengri.components.stellar.sfh.met_registry",
    }
)


def _register_model_registry_params(
    out: dict[str, ParameterRecord],
    model_registry: dict[str, object],
    *,
    owner: str,
    label: str,
) -> None:
    """Add a model registry's per-model parameter declarations, in place.

    Both the SFH and metallicity registries map a model name to a spec whose
    ``params`` attribute is a ``{param_name: ParamDef}`` dict. ``ParamDef``
    stores the *distribution* under ``.default``, which is this registry's
    ``prior``.

    First-wins on name collisions, matching the component walk: a parameter
    several models declare (composition variants share names) is recorded once,
    against the first model that declares it.

    Parameters
    ----------
    out : dict
        Registry map being built. Mutated in place.
    model_registry : dict
        Name -> model spec carrying a ``params`` mapping.
    owner : str
        Fully-qualified module path recorded on each record.
    label : str
        Registry name used to build the record's ``group`` field, e.g.
        ``"SFH_REGISTRY"``.
    """
    for model_name in sorted(model_registry):
        spec = model_registry[model_name]
        for pname, pdef in (getattr(spec, "params", None) or {}).items():
            if pname in out:
                continue
            out[pname] = ParameterRecord(
                name=pname,
                prior=getattr(pdef, "default", None),
                description=getattr(pdef, "description", ""),
                units=_units_from_name(pname),
                owner=owner,
                group=f"{label}[{model_name!r}].params",
                # Threading this is what makes an SFH/MET parameter freeable at
                # all. The component ``_params.py`` walk above passes
                # ``decl.free_prior``; this branch did not, so every ``sfh_*``
                # record arrived with ``free_prior=None`` regardless of its
                # declaration and ``all_params: FREE`` could never expand one
                # (#887). ``getattr`` with a default keeps older registries that
                # predate the ``ParamDef`` field working.
                free_prior=getattr(pdef, "free_prior", None),
            )


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
                    free_prior=decl.free_prior,
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
                free_prior=decl.free_prior,
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
                free_prior=decl.free_prior,
            )

    # Star-formation-history and metallicity parameters.
    #
    # Neither owns a ``_params.py``: their parameters are declared per model in
    # ``SFH_REGISTRY[<type>].params`` / ``MET_REGISTRY[<type>].params``, a
    # different mechanism that the walk above cannot see. The result was that
    # *every* SFH parameter was missing from introspection —
    # ``list_parameters()`` returned 189 names with no ``sfh_*`` at all, and
    # ``describe_parameter("sfh_dpl_alpha")`` raised ``KeyError`` for the very
    # identifier the naming contract uses as its worked example (#1264).
    #
    # Walk the registries rather than hand-listing names here: models arrive by
    # registration, and a hand-kept copy would go stale exactly the way this
    # gap appeared in the first place.
    from tengri.components.stellar.sfh.met_registry import MET_REGISTRY
    from tengri.components.stellar.sfh.registry import SFH_REGISTRY

    _register_model_registry_params(
        out,
        SFH_REGISTRY,
        owner="tengri.components.stellar.sfh.registry",
        label="SFH_REGISTRY",
    )
    _register_model_registry_params(
        out,
        MET_REGISTRY,
        owner="tengri.components.stellar.sfh.met_registry",
        label="MET_REGISTRY",
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
        # SFH / metallicity parameters are declared in their own model
        # registries, which also own the translation via ``internal_param_map``.
        # Emitting an identity entry here would collide with it — see
        # ``_TRANSLATION_OWNED_ELSEWHERE``.
        if record.owner in _TRANSLATION_OWNED_ELSEWHERE:
            continue
        # Default identity mapping: internal == public, scale=1.0, offset=0.0
        # The translate module will override specific entries with unit conversions.
        result[name] = (name, 1.0, 0.0, record.units)
    return result


def list_parameters(prefix: str | None = None):
    """List every free parameter in the registry, optionally filtered.

    Parameters
    ----------
    prefix : str, optional
        If given, only return names starting with this prefix
        (e.g. ``"dust_"``, ``"agn_"``). Useful for surveying a
        physics domain.

    Returns
    -------
    _RegistryTable
        One row per parameter, sorted by name, with keys ``name``,
        ``description``, ``units``, ``owner`` and ``group``. Renders as a
        table in a notebook.

        This used to return ``list[str]``, the only ``list_*`` that did
        (#1285). Use ``.names()`` for the old shape — and note the bare names
        were throwing away the description and units the registry stores.

    Examples
    --------
    >>> import tengri
    >>> tengri.list_parameters(prefix="radio_").names()[:3]
    ['radio_T_e', 'radio_alpha_agn', 'radio_alpha_ff']
    """
    from tengri.registry import _RegistryTable

    reg = registry()
    names = sorted(reg.keys())
    if prefix is not None:
        names = [n for n in names if n.startswith(prefix)]

    rows = []
    for name in names:
        rec = reg[name]
        rows.append(
            {
                "name": name,
                "kind": "parameter",
                "description": getattr(rec, "description", "") or "",
                "units": getattr(rec, "units", "") or "",
                "owner": getattr(rec, "owner", "") or "",
                "group": getattr(rec, "group", "") or "",
            }
        )
    return _RegistryTable(rows)


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

    # Translate recipe to Parameters (no SSP data needed).
    #
    # ``_allow_empty_wildcard``: this is a *discovery* call, not a model the
    # caller intends to fit. Introspection recipes use ``all_params: FREE`` to
    # mean "surface every parameter of this variant" and then read
    # ``all_params`` regardless of free/fixed, so a wildcard that frees nothing
    # is harmless here — unlike in user model construction, where it silently
    # pins the physics being fitted.
    params = parse_groups(**recipe_dict, _allow_empty_wildcard=True)

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
