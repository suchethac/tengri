# SPDX-License-Identifier: BSD-3-Clause
"""Property registry for computed derived quantities.

Provides a mechanism to declare, register, and query derived quantities
(properties) that are computed from the orchestrator :class:`ForwardState`.
Properties are grouped (e.g., ``"sfh"``, ``"sed"``) and each has a pure
function that reads from ``state.derived`` and returns a scalar.

Examples
--------
Define properties on a component::

    class StellarSEDComponent(SEDModelComponent):
        properties = {
            "stellar_mass": Property(
                units="Msun",
                group="sfh",
                doc="Total formed stellar mass.",
                fn=lambda state, params: 10 ** state.derived["log_mstar_formed"],
            ),
        }

Access properties on a model::

    model = SEDModel.build(...)
    props = model.predict_properties(params, names=("stellar_mass",))
    # or
    pred = model.predict(params)
    stellar_mass = pred.stellar_mass

Notes
-----
**Collision semantics**: The global registry maps name → list[PropertyEntry].
A collision error fires only when TWO SIMULTANEOUSLY-ACTIVE components in a
built model declare the same property name. This is checked at
``SEDModel.available_properties`` assembly.

**JIT-compatible**: Property functions must be pure JAX, as they are called
with the ``ForwardState`` during forward passes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

__all__ = [
    "PROPERTY_REGISTRY",
    "Property",
    "PropertyEntry",
    "assemble_available_properties",
    "missing_property_message",
    "register_properties",
    "warn_if_lines_are_unavailable",
]

# Module-level registry: name → list[PropertyEntry]
PROPERTY_REGISTRY: dict[str, list[PropertyEntry]] = {}


@dataclass(frozen=True)
class Property:
    """A derived quantity computed from :class:`ForwardState`.

    Parameters
    ----------
    units : str
        Physical units of the quantity (e.g., ``"Msun"``, ``"1/yr"``).
    group : str
        Group name for related properties (e.g., ``"sfh"``, ``"sed"``).
    doc : str
        One-line description of the quantity.
    fn : Callable
        Pure function ``fn(state, params) -> scalar`` that reads
        ``state.derived[...]`` and returns a JAX scalar.

    Notes
    -----
    The ``fn`` field is not compared or hashed (marked ``compare=False``)
    so frozen dataclasses can store it.
    """

    units: str
    group: str
    doc: str
    fn: Callable = field(compare=False, hash=False)


@dataclass(frozen=True)
class PropertyEntry:
    """Registered entry pairing a property name to its metadata and function.

    Parameters
    ----------
    name : str
        Canonical name of the property (e.g., ``"stellar_mass"``).
    units : str
        Physical units (e.g., ``"Msun"``).
    group : str
        Group name (e.g., ``"sfh"``).
    doc : str
        Short description.
    component_name : str
        Name of the component that declared this property.
    fn : Callable
        Pure function ``fn(state, params) -> scalar``.
    """

    name: str
    units: str
    group: str
    doc: str
    component_name: str
    fn: Callable = field(compare=False, hash=False)


def register_properties(component_name: str, props: dict[str, Property]) -> None:
    """Register properties declared by a component.

    Called by :meth:`SEDModelComponent.__init_subclass__` during class definition.

    Parameters
    ----------
    component_name : str
        Name of the component (e.g., ``"stellar"``).
    props : dict[str, Property]
        Mapping of property name to :class:`Property` instance.

    Notes
    -----
    This function populates :data:`PROPERTY_REGISTRY` by appending
    :class:`PropertyEntry` instances. Multiple components may declare
    the same property name (e.g., if they are conditional variants);
    the collision check happens at model build time in
    :func:`assemble_available_properties`.
    """
    for name, prop in props.items():
        entry = PropertyEntry(
            name=name,
            units=prop.units,
            group=prop.group,
            doc=prop.doc,
            component_name=component_name,
            fn=prop.fn,
        )
        if name not in PROPERTY_REGISTRY:
            PROPERTY_REGISTRY[name] = []
        PROPERTY_REGISTRY[name].append(entry)


def assemble_available_properties(active_component_names: set[str]) -> dict[str, PropertyEntry]:
    """Assemble properties available in a built model.

    Filters :data:`PROPERTY_REGISTRY` to only entries whose component
    is active in the model. Raises :exc:`ValueError` if two active
    components declare the same property name.

    Parameters
    ----------
    active_component_names : set[str]
        Set of component names present in the model
        (e.g., ``{"stellar", "dust", "neb"}``).

    Returns
    -------
    dict[str, PropertyEntry]
        Mapping of property name to its entry, filtered to active components.

    Raises
    ------
    ValueError
        If two active components declare the same property name.
    """
    catalog: dict[str, PropertyEntry] = {}
    for name, entries in PROPERTY_REGISTRY.items():
        active_entries = [e for e in entries if e.component_name in active_component_names]
        if len(active_entries) > 1:
            component_list = ", ".join(e.component_name for e in active_entries)
            raise ValueError(
                f"Property {name!r} declared by multiple active components: {component_list}. "
                f"Remove the collision or use a model variant with only one of them."
            )
        if active_entries:
            catalog[name] = active_entries[0]
    return catalog


def _grammar_hint(component_name: str) -> str:
    """How to add the missing component, naming only things that exist.

    Two things are checked rather than assumed. The component name is *not*
    always the grammar group, ``nebular`` declares the line properties but the
    group is ``neb``, so the group is named only when the grammar accepts it.
    And the menu verb is named only when it is actually exported. Advice that
    does not resolve is worse than no advice.
    """
    from tengri.parameters.groups import _GROUP_STRUCTURAL_KEYS

    if component_name not in _GROUP_STRUCTURAL_KEYS:
        return ""
    hint = f" Add the {component_name!r} group when you build the model"
    import tengri

    lister = f"list_{component_name}_models"
    if hasattr(tengri, lister):
        hint += f", tengri.{lister}() lists the choices"
    return hint + "."


def _diagnose(name: str, known: list[str]) -> str:
    """One property's diagnosis, without the shared list of what *is* available."""
    if name in PROPERTY_REGISTRY:
        components = sorted({e.component_name for e in PROPERTY_REGISTRY[name]})
        owner = " or ".join(repr(c) for c in components)
        hint = _grammar_hint(components[0]) if len(components) == 1 else ""
        return (
            f"{name!r} comes from the {owner} component, which this model does "
            f"not include, so it cannot be computed.{hint} It is a real "
            f"property, tengri.describe_property({name!r}) documents it."
        )
    import difflib

    close = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    suggestion = f" Did you mean {close}?" if close else ""
    return f"Unknown property {name!r}.{suggestion}"


def missing_property_message(*names: str, available: dict | set | list) -> str:
    """Why a property lookup failed, misspelling, or a component not built?

    Parameters
    ----------
    *names : str
        The properties the caller asked for that could not be served.
    available : dict | set | list
        The names this model can compute.

    Returns
    -------
    str
        Message body for the raised :exc:`KeyError`. One diagnosis per name,
        then the available list **once**, repeating 43 names per bad name
        turned a two-name mistake into a 1600-character wall.

    Notes
    -----
    ``list_properties()`` advertises every registered property regardless of
    what any one model contains, so "not available here" is the *common* case
    and "you misspelled it" is the rare one. Reporting both as ``Unknown
    property`` sent readers hunting for a typo in a name they had just copied
    off the menu, the component was simply not in their model.
    """
    known = sorted(available)
    diagnoses = [_diagnose(name, known) for name in names]
    body = "\n".join(diagnoses) if len(diagnoses) > 1 else diagnoses[0]
    return f"{body}\nAvailable on this model: {known}"


def line_property_names() -> frozenset[str]:
    """Registered properties that need a per-line luminosity catalog.

    Returns
    -------
    frozenset of str
        Every registered property in the ``lines`` group.

    Notes
    -----
    Read off the registry rather than listed by hand, so a new line diagnostic
    is covered by :func:`warn_if_lines_are_unavailable` the moment it is
    registered.
    """
    return frozenset(
        name
        for name, entries in PROPERTY_REGISTRY.items()
        if any(entry.group == "lines" for entry in entries)
    )


def _headline_line_target_waves(name: str):
    """The :data:`~tengri.utils.sed_quantities.KEY_LINES` wavelengths a line
    property depends on, if any.

    Parameters
    ----------
    name : str
        A property name from the ``lines`` group.

    Returns
    -------
    tuple of float or None
        Rest-frame target wavelengths [Angstrom], read from ``KEY_LINES``
        under ``name`` itself or, for a log companion, under ``name`` with
        its ``log_`` prefix stripped. ``None`` for a property this check does
        not cover -- a ratio diagnostic (``bpt_nii``, ``o32``, ...) combines
        several ``KEY_LINES`` entries rather than owning one wavelength, so
        it is left to the coarser catalog-publishes-nothing check above.
    """
    from tengri.utils.sed_quantities import KEY_LINES

    key = name[4:] if name.startswith("log_") else name
    return KEY_LINES.get(key)


def _warn_if_headline_line_uncovered(model, backend, requested, state) -> None:
    """Warn when a requested headline line has no catalog match within tolerance.

    Parameters
    ----------
    model : SEDModel
        The model being queried; only used to read ``model.spec.cue_full_catalog``
        for the remedy text.
    backend : object
        The active nebular backend. Already confirmed (by the caller) to
        publish a per-line catalog at all -- this checks whether it carries
        the *specific* requested line.
    requested : set of str
        Property names from the ``lines`` group the caller asked for.
    state : ForwardState
        The orchestrator state carrying ``derived["line_waves"]``.

    Notes
    -----
    Eager-path only: the ``float``/``np.asarray`` casts raise
    :exc:`~jax.errors.ConcretizationTypeError` under any ``jax`` transform
    (e.g. ``jax.jit(model.predict_properties)``), where there is no concrete
    catalog to inspect yet and nothing to say.

    Generalizes across backends: only the *published*
    ``state.derived["line_waves"]`` is read, not any backend-specific
    attribute, so cue, cloudy, cb19 and mappings are covered by the same
    tolerance check that :func:`~tengri.utils.sed_quantities.extract_line_luminosity`
    itself uses (``sed_quantities._LINE_MATCH_TOL_AA``) -- the same
    comparison that decides whether the property comes back NaN, so this
    warns exactly when (and only when) that NaN is about to happen.
    """
    import warnings

    import jax
    import numpy as np

    derived = getattr(state, "derived", None)
    if derived is None or "line_waves" not in derived:
        return
    try:
        line_waves = np.asarray(derived["line_waves"], dtype=float)
    except jax.errors.ConcretizationTypeError:
        return
    if line_waves.size == 0:
        return

    from tengri.utils.sed_quantities import _LINE_MATCH_TOL_AA

    backend_name = type(backend).__name__
    is_cue_subset = backend_name == "CueBackend" and not bool(
        getattr(getattr(model, "spec", None), "cue_full_catalog", True)
    )
    remedy = (
        "rebuild the model with neb={'type': 'cue', 'full_catalog': True} to "
        "reach the full catalog"
        if is_cue_subset
        else "the selected nebular backend/grid does not carry this line"
    )

    for name in sorted(requested):
        target_waves = _headline_line_target_waves(name)
        if target_waves is None:
            continue
        offsets = [float(np.min(np.abs(line_waves - tw))) for tw in target_waves]
        best_offset = min(offsets)
        if best_offset <= _LINE_MATCH_TOL_AA:
            continue
        nearest_target = target_waves[offsets.index(best_offset)]
        nearest_line_aa = float(line_waves[int(np.argmin(np.abs(line_waves - nearest_target)))])
        warnings.warn(
            f"{name!r} has no catalog line within {_LINE_MATCH_TOL_AA:.0f} "
            f"Angstrom of its target wavelength on this {backend_name!r} "
            f"catalog (nearest catalog line {nearest_line_aa:.2f} Å, "
            f"{best_offset:.1f} Å away), so {name!r} will be NaN. "
            f"{remedy}. See #2239.",
            UserWarning,
            stacklevel=4,
        )


def warn_if_lines_are_unavailable(model, names, *, state=None) -> None:
    """Warn when a requested line property can only come back NaN.

    Parameters
    ----------
    model : SEDModel
        The model whose nebular backend is inspected.
    names : iterable of str
        Property names the caller asked for.
    state : ForwardState, optional
        The orchestrator state, if already computed. When given, and the
        backend does publish a catalog, also checks whether the *specific*
        requested headline line (:data:`~tengri.utils.sed_quantities.KEY_LINES`)
        is actually covered by that catalog -- see
        :func:`_warn_if_headline_line_uncovered`. Omit only when no state is
        available yet (the coarser "no catalog at all" check below still runs).

    Notes
    -----
    ``BakedInBackend`` and the shock backends publish no per-line catalog, so
    every ``lines`` property is NaN. ``Prediction._ensure_lines`` has warned
    about that since #361, but only on the ``pred.lines.*`` route. The dict
    accessor and :meth:`~tengri.SEDModel.predict_properties`, the documented
    jit/vmap surface for derived quantities, returned the same NaN in silence.
    One helper, called by all three, is what keeps them from drifting again.

    A backend can publish *a* catalog and still not carry *this* line --
    cue's legacy 128-line subset has no C IV entry (#2239). That is a
    narrower, per-line question than "does this backend publish lines at
    all", so it is checked separately, by
    :func:`_warn_if_headline_line_uncovered`, only when ``state`` is given.

    Fires at trace time under ``jax.jit``, like the accessor's warning.
    """
    import warnings

    requested = line_property_names().intersection(names)
    if not requested:
        return
    backend = getattr(model, "_nebular_backend", None)
    if backend is None or not hasattr(backend, "predict_nebular_line_luminosities"):
        backend_name = type(backend).__name__ if backend is not None else "None"
        warnings.warn(
            f"Nebular backend {backend_name!r} does not publish a per-line "
            f"luminosity catalog, so {sorted(requested)[:4]} and the other "
            f"'lines' properties will be NaN. To get discrete line luminosities, "
            f"rebuild the model with neb={{'type': 'cue'}}, 'cloudy', or 'cb19' "
            f"(each requires a compatible SSP and any backing grid; see "
            f"tengri.list_nebular_backends() for details). See #361.",
            UserWarning,
            stacklevel=3,
        )
        return
    if state is not None:
        _warn_if_headline_line_uncovered(model, backend, requested, state)
