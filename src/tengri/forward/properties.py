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
    always the grammar group — ``nebular`` declares the line properties but the
    group is ``neb`` — so the group is named only when the grammar accepts it.
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
        hint += f" — tengri.{lister}() lists the choices"
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
            f"property — tengri.describe_property({name!r}) documents it."
        )
    import difflib

    close = difflib.get_close_matches(name, known, n=3, cutoff=0.6)
    suggestion = f" Did you mean {close}?" if close else ""
    return f"Unknown property {name!r}.{suggestion}"


def missing_property_message(*names: str, available: dict | set | list) -> str:
    """Why a property lookup failed — misspelling, or a component not built?

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
        then the available list **once** — repeating 43 names per bad name
        turned a two-name mistake into a 1600-character wall.

    Notes
    -----
    ``list_properties()`` advertises every registered property regardless of
    what any one model contains, so "not available here" is the *common* case
    and "you misspelled it" is the rare one. Reporting both as ``Unknown
    property`` sent readers hunting for a typo in a name they had just copied
    off the menu — the component was simply not in their model.
    """
    known = sorted(available)
    diagnoses = [_diagnose(name, known) for name in names]
    body = "\n".join(diagnoses) if len(diagnoses) > 1 else diagnoses[0]
    return f"{body}\nAvailable on this model: {known}"
