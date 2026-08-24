# SPDX-License-Identifier: BSD-3-Clause
"""Forward model components: stellar, dust, nebular, AGN, IGM, radio, X-ray.

Physics blocks for the SEDModel pipeline. Each block owns its parameter
definitions and a SEDComponent implementation that receives upstream
derived quantities (redshift, stellar mass, SFR, AGN luminosity) and
computes rest-frame specific luminosity L_nu over its spectral domain.
"""

# Import canonical submodules eagerly so users can write
# ``tengri.components.dust`` etc. without an explicit submodule import.
from tengri.components import agn, dust, igm, nebular, radio, stellar, xray
from tengri.components.sed_model_component import SEDModelComponent

__all__ = [
    "SEDModelComponent",
    "agn",
    "dust",
    "igm",
    "nebular",
    "radio",
    "register_component",
    "stellar",
    "xray",
]


# ──────────────────────────────────────────────────────────────────
# SEDComponent discovery registry
#
# Holds every SEDComponent class: built-ins (auto-registered the first
# time `_get_registered_components()` is called) and any contributor
# class decorated with `@register_component`. Consumed by
# `tengri.parameters.translate._build_param_map` to auto-derive identity
# entries from each component's `declared_parameters()`.
# ──────────────────────────────────────────────────────────────────

_REGISTERED_COMPONENTS: list[type] = []
_BUILTINS_LOADED: bool = False


def register_component(cls: type) -> type:
    """Register an SEDComponent class for parameter auto-discovery.

    Decorate a contributor's :class:`SEDComponent`-conforming class to
    have its :meth:`declared_parameters` consulted at param-map-build
    time, so any new free parameters become available to
    :class:`tengri.Parameters` without editing
    ``tengri.parameters.translate``.

    Parameters
    ----------
    cls: type
        A class conforming to the :class:`SEDComponent` protocol. It must
        expose ``name``, ``parameter_prefix``, and a
        ``declared_parameters()`` method returning the parameter
        declarations the component contributes.

    Returns
    -------
    type
        The same class, unchanged, so the decorator is transparent.

    Notes
    -----
    Registration is idempotent: the class is appended once and
    re-decorating an already-registered class is a no-op. This function
    is the parameter-discovery seam only. Subclassing
    :class:`tengri.SEDModelComponent` registers a component for
    ``SEDModel.build`` dispatch as well and is the recommended path for
    new physics blocks (ADR-0011); reach for this decorator when a class
    cannot take that base.

    Examples
    --------
    >>> from tengri import register_component
    >>> @register_component
    ... class MyAGNSEDComponent:
    ...     name = "my_agn"
    ...     parameter_prefix = "my_agn_"
    ...
    ...     def declared_parameters(self):
    ...         return [ParamDeclaration("my_agn_param", ...)]  # doctest: +SKIP
    """
    if cls not in _REGISTERED_COMPONENTS:
        _REGISTERED_COMPONENTS.append(cls)
    return cls


def _ensure_builtins_loaded() -> None:
    """Lazy-register the seven built-in SEDComponent adapters.

    Done lazily (not at module import) to avoid the circular imports
    that would arise if every component module pulled in
    ``tengri.components`` at top level.
    """
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True

    from tengri.components.agn.component import AGNSEDComponent
    from tengri.components.dust.component import DustAttenuationSEDComponent
    from tengri.components.igm.component import IGMSEDComponent
    from tengri.components.nebular.component import NebularSEDComponent
    from tengri.components.radio.component import RadioSEDComponent
    from tengri.components.stellar.component import StellarSEDComponent
    from tengri.components.xray.component import XRaySEDComponent

    for cls in (
        StellarSEDComponent,
        DustAttenuationSEDComponent,
        NebularSEDComponent,
        AGNSEDComponent,
        IGMSEDComponent,
        RadioSEDComponent,
        XRaySEDComponent,
    ):
        register_component(cls)


def _get_registered_components() -> list:
    """Return all registered SEDComponent classes (built-ins + contributors).

    Returns
    -------
    list
        SEDComponent classes available in the current process. The
        list is freshly copied; callers may not mutate it; use
        :func:`register_component` to add entries.
    """
    _ensure_builtins_loaded()
    return list(_REGISTERED_COMPONENTS)
