# SPDX-License-Identifier: BSD-3-Clause
"""Component registry for closure-friendly SED component precomputation.

This module provides a registry-driven design for adding new SED components
(dust IR, AGN torus, etc.) without modifying the hybrid kernel or PrecomputedData.

Each component is registered with:
- A precompute function that builds runtime arrays (grid, axes, etc.)
- An extract_arrays function that shapes these arrays for JIT threading
- A build_lookup function that returns a JIT-compiled callable
- An apply_signature tuple defining the positional parameter order
- An activation predicate that determines if the component fires at runtime

This design ensures that adding a new component requires only:
1. Writing the precompute/build_lookup/extract functions
2. Creating a ComponentSpec and calling register(...)
3. No edits to sed_model.py, hybrid.py, or PrecomputedData

Example:

    @dataclass(frozen=True)
    class MyComponentSpec(ComponentSpec):
        name: str = "my_component:backend_name"

    spec = MyComponentSpec(
        name="my_component:backend_name",
        precompute=lambda filters, trans, z, params: precompute_my_thing(...),
        extract_arrays=lambda precomp: (precomp["array1"], precomp["array2"]),
        build_lookup=lambda precomp, *, grid_arrays_traced=None: build_my_lookup(...),
        apply_signature=("my_param1", "my_param2"),
        activation=lambda spec, params: params.spec.my_param1 is not None,
    )
    register(spec)
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["REGISTRY", "ComponentSpec", "get_component", "register"]


@dataclass(frozen=True)
class ComponentSpec:
    """Registry entry for a closure-friendly SED component.

    Parameters
    ----------
    name : str
        Unique key, e.g. "dust_ir:draine_li2007" or "agn:skirtor".
        Format is "{family}:{model}" to group related models.

    precompute : callable
        Signature: ``(filter_waves, filter_trans, redshift, parameters) -> Precomputed``.
        Must return ``None`` when the component is inactive or data is missing
        (so callers can skip it). Otherwise returns a dict-like object containing
        precomputed grids, axes, and metadata.

    extract_arrays : callable
        Signature: ``(precompute_result) -> tuple[jax.Array, ...]``.
        These arrays will be threaded as JIT runtime arguments to avoid
        closure capture of large arrays. Should return empty tuple if
        no arrays need tracing (for analytic models).

    build_lookup : callable
        Signature: ``(precompute_result, *, grid_arrays_traced=None) -> JIT-callable``.
        Returns a JIT-compiled callable whose positional arguments are
        component-specific (parameter values matching apply_signature).
        When ``grid_arrays_traced`` is supplied, the callable uses those traced
        arrays; when ``None``, falls back to closure capture (back-compat).

    apply_signature : tuple[str, ...]
        Parameter names that the lookup accepts (positional), in order.
        Example: ("dust_umin", "dust_gamma_dl", "dust_qpah") for DL07.
        Adding a new component → fill this list once.

    activation : callable
        Signature: ``(spec, parameters) -> bool`` — does this component fire?
        Parameters is the user's Parameters object.
        Encapsulates the if/elif chain that was in sed_model.py / hybrid.py.
        Example: ``lambda spec, params: params._dust_emission_model == "draine_li2007"``.

    Notes
    -----
    **Immutable by design**: Frozen dataclass prevents accidental mutation.

    **JIT compatibility**: extract_arrays and build_lookup must return
    JAX pytrees (arrays, tuples, dicts with array leaves). The returned
    lookup callable receives traced arrays as kwargs, so it's safe inside
    a JIT body.
    """

    name: str
    precompute: Callable[..., Any]
    extract_arrays: Callable[[Any], tuple]
    build_lookup: Callable[..., Callable]
    apply_signature: tuple[str, ...]
    activation: Callable[[ComponentSpec, Any], bool]


REGISTRY: dict[str, ComponentSpec] = {}
"""Global component registry, keyed by ComponentSpec.name."""


def register(spec: ComponentSpec) -> ComponentSpec:
    """Register a component in the global registry.

    Parameters
    ----------
    spec : ComponentSpec
        The component specification to register.

    Returns
    -------
    ComponentSpec
        The same spec (for use as a decorator).

    Raises
    ------
    ValueError
        If a component with the same name is already registered.

    Examples
    --------
    .. code-block:: python

        spec = ComponentSpec(name="dust_ir:draine_li2007", ...)
        register(spec)
    """
    if spec.name in REGISTRY:
        raise ValueError(f"Duplicate component registration: {spec.name}")
    REGISTRY[spec.name] = spec
    return spec


def get_component(name: str) -> ComponentSpec | None:
    """Retrieve a registered component by name.

    Parameters
    ----------
    name : str
        Component name, e.g. "dust_ir:draine_li2007".

    Returns
    -------
    ComponentSpec or None
        The registered spec, or None if not found.
    """
    return REGISTRY.get(name)


def list_components(family: str | None = None) -> dict[str, ComponentSpec]:
    """List all registered components, optionally filtered by family.

    Parameters
    ----------
    family : str, optional
        Filter to this family (e.g., "dust_ir" → all "dust_ir:*" entries).
        If None, returns all registered components.

    Returns
    -------
    dict[str, ComponentSpec]
        Mapping of name → spec for matching components.

    Examples
    --------
    .. code-block:: python

        dust_models = list_components(family="dust_ir")
        agn_models = list_components(family="agn")
    """
    if family is None:
        return dict(REGISTRY)
    prefix = f"{family}:"
    return {k: v for k, v in REGISTRY.items() if k.startswith(prefix)}
