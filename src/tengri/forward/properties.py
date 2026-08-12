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


def warn_if_lines_are_unavailable(model, names) -> None:
    """Warn when a requested line property can only come back NaN.

    Parameters
    ----------
    model : SEDModel
        The model whose nebular backend is inspected.
    names : iterable of str
        Property names the caller asked for.

    Notes
    -----
    ``BakedInBackend`` and the shock backends publish no per-line catalog, so
    every ``lines`` property is NaN. ``Prediction._ensure_lines`` has warned
    about that since #361 — but only on the ``pred.lines.*`` route. The dict
    accessor and :meth:`~tengri.SEDModel.predict_properties`, the documented
    jit/vmap surface for derived quantities, returned the same NaN in silence.
    One helper, called by all three, is what keeps them from drifting again.

    Fires at trace time under ``jax.jit``, like the accessor's warning.
    """
    import warnings

    requested = line_property_names().intersection(names)
    if not requested:
        return
    backend = getattr(model, "_nebular_backend", None)
    if backend is not None and hasattr(backend, "predict_nebular_line_luminosities"):
        return
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
