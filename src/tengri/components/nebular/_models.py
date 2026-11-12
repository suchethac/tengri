# SPDX-License-Identifier: BSD-3-Clause
"""Runtime registry of nebular emission backends.

Mirrors ``tengri.components.xray._models``, ``tengri.components.radio._models``,
and ``tengri.components.igm._models`` (#355) and the older
``SFH_REGISTRY`` / ``AGN_MODELS`` patterns. ``_VALID_NEBULAR_TYPES``
in :mod:`tengri.parameters.groups` derives from
:data:`NEBULAR_MODELS.keys()` per ADR-0005 / ADR-0008 (single source of
truth): adding a new backend is one ``register_nebular_model`` call,
not a parallel edit to a validator set.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True)
class NebularRegistryEntry:
    """Registry entry for a nebular emission backend.

    Attributes
    ----------
    callable : Callable or None
        The backend class (or any reference). ``None`` for the
        ``'none'`` disable-toggle. Dispatch in
        ``tengri.parameters.groups._translate_neb`` still routes
        through ``nebular_ssp`` / ``nebular_cue`` / ``nebular`` flags
        on :class:`Parameters`; this field is metadata.
    citation : str
        Academic citation. Empty for the disable-toggle.
    status : str
        ``"production"`` / ``"experimental"`` / ``"demo"`` / ``"deprecated"``.
    short_doc : str
        One-line description.
    config_backend : str or None
        The ``config.backend`` string this grammar name lowers to, or ``None``
        for ``'none'``. ``'mappings_agn'`` carries ``'mappings'``, which is the
        name the build's class-name inference reads today.
    """

    callable: Callable | None
    citation: str = ""
    status: str = "production"
    short_doc: str = ""
    config_backend: str | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        fn = object.__getattribute__(self, "callable")
        if fn is None:
            raise TypeError("The 'none' nebular model has no callable.")
        return fn(*args, **kwargs)


NEBULAR_MODELS: dict[str, NebularRegistryEntry] = {}


@functools.cache
def _nebular_backend_classes() -> tuple[type[Any], ...]:
    """Return the seven nebular backend classes, imported on first use."""
    from tengri.components.nebular.baked_in import BakedInBackend
    from tengri.components.nebular.cloudy_cb19 import CB19Backend
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend
    from tengri.components.nebular.cue import CueBackend
    from tengri.components.nebular.gutkin16 import Gutkin16Backend
    from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend
    from tengri.components.nebular.shock import ShockBackend

    return (
        BakedInBackend,
        CB19Backend,
        CloudyGridBackend,
        CueBackend,
        Gutkin16Backend,
        MappingsPhotoStellarBackend,
        ShockBackend,
    )


@functools.cache
def nebular_backend_classes() -> Mapping[str, type[Any]]:
    """Map each ``config.backend`` string to the backend class that it selects.

    The keys are the classes' own ``config_name`` values. The mapping is built
    on first use because the backend modules import heavy dependencies and the
    package ``__init__`` imports this module first.
    """
    return MappingProxyType({cls.config_name: cls for cls in _nebular_backend_classes()})


def resolve_nebular_backend(name: str) -> type[Any]:
    """Return the backend class that ``config.backend == name`` selects.

    Parameters
    ----------
    name : str
        A ``config.backend`` string, for example ``"cloudy_grid"``.

    Returns
    -------
    type
        The backend class registered under ``name``.

    Raises
    ------
    ConfigError
        If ``name`` is not a registered config name. The message lists the valid
        names and a close match when one exists.
    """
    import difflib

    from tengri.config.exceptions import ConfigError

    classes = nebular_backend_classes()
    if name in classes:
        return classes[name]
    valid = sorted(classes)
    suggestions = difflib.get_close_matches(name, valid, n=2, cutoff=0.6)
    hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
    raise ConfigError(
        f"Unknown nebular backend {name!r}; valid names are {', '.join(valid)}.{hint}"
    )


@functools.cache
def nebular_backends_with_freefree_continuum() -> frozenset[str]:
    """Return the config names whose backend carries a thermal free-free continuum.

    Derived from each class's ``carries_freefree``. A backend instance that sets
    ``carries_freefree_continuum`` (Gutkin+2016, per ``continuum=``) overrides this
    set in :func:`nebular_backend_carries_freefree`.

    Notes
    -----
    Membership decides whether the radio block's Murphy+2011 ``radio_freefree``
    term is added on top of the nebular continuum (issue #2346, #2574). A
    nebular-included SSP carries its free-free inside the SSP flux, and the radio
    factory reads that from the SSP's ``nebular`` stamp, not from this set.
    """
    return frozenset(
        name for name, cls in nebular_backend_classes().items() if cls.carries_freefree
    )


@functools.cache
def nebular_photoionized_backends() -> frozenset[str]:
    """Return the config names of the photoionized nebular backends.

    Derived from ``supports_split_sed``: the photoionized backends are exactly the
    ones with a continuum plus line-catalog split (Cue, CloudyGrid, CB19, Gutkin16,
    MAPPINGS). The equality is asserted in the contract tests.
    """
    return frozenset(
        name for name, cls in nebular_backend_classes().items() if cls.supports_split_sed
    )


def nebular_backend_carries_freefree(name: str | None, backend: object | None = None) -> bool:
    """Check if a declared nebular backend carries a free-free continuum.

    Used by :func:`tengri.forward.component_factory.build_components` to
    auto-resolve ``radio.include_freefree=None`` to ``False`` when the declared
    nebular backend publishes its own thermal free-free (issue #2346): one
    thermal term on the whole SED grid, not separate radio + nebular terms.

    Parameters
    ----------
    name : str or None
        Nebular backend name from the factory's vocabulary:
        ``"baked_in"`` (default), ``"cb19"``, ``"cloudy_grid"``, ``"gutkin16"``,
        ``"mappings"``, ``"cue"``, ``"shock"``, or ``None`` to omit nebular.
    backend : object or None, optional
        The pre-constructed backend instance, when one was passed. If it carries a
        ``carries_freefree_continuum`` attribute, that attribute decides.

    Returns
    -------
    bool
        ``True`` if the backend carries a free-free continuum. ``False`` otherwise,
        including ``None``.

    Notes
    -----
    The rule rests on what the backend's continuum supplies, decided in this order:

    1. A backend instance that declares ``carries_freefree_continuum`` (Gutkin+2016,
       true iff ``continuum="analytic"``) answers for itself.
    2. Otherwise the declared name decides, by membership in
       :func:`nebular_backends_with_freefree_continuum`.

    Neither step reads the published ``sed_nebular`` array or an approximation flag:
    the fast nebular grid path publishes ``sed_nebular`` as all zeros for the same
    physical model, and structural decisions must never be gated on approximation
    toggles.
    """
    declared = getattr(backend, "carries_freefree_continuum", None)
    if declared is not None:
        return bool(declared)
    return name in nebular_backends_with_freefree_continuum()


def nebular_carries_freefree(name: str | None, instance: object | None = None) -> bool:
    """Check whether the nebular side of a model publishes thermal free-free (#2962).

    The one predicate both radio free-free gates read: the factory's
    ``include_freefree`` resolution and the ``bell2003_split`` thermal law. It extends
    :func:`nebular_backend_carries_freefree` with the backend object, because a
    ``NebularContinuumFallback`` wraps a
    line-only backend under the declared name (``"cb19"``, ``"mappings"``) and its
    analytic tier supplies the free-free and two-photon continuum over the whole SSP
    grid.

    Parameters
    ----------
    name : str or None
        Declared nebular backend name, as for :func:`nebular_backend_carries_freefree`.
    instance : object or None
        The pre-constructed backend object, when one was passed.

    Returns
    -------
    bool
        ``True`` if the declared name or the backend object carries a free-free
        continuum.

    Notes
    -----
    **JIT-compatible**: no, configuration read.
    """
    from tengri.components.nebular._shared import NebularContinuumFallback

    if nebular_backend_carries_freefree(name, instance):
        return True
    # The wrapper always carries free-free: its Tier 1 is a continuum-capable backend
    # and its Tier 2 is the analytic free-free plus two-photon continuum.
    return isinstance(instance, NebularContinuumFallback)


def register_nebular_model(
    name: str,
    *,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
    config_backend: str | None = None,
) -> Callable[[Callable | None], Callable | None]:
    """Register a nebular emission backend in :data:`NEBULAR_MODELS`."""

    def decorator(fn: Callable | None) -> Callable | None:
        NEBULAR_MODELS[name] = NebularRegistryEntry(
            callable=fn,
            citation=citation,
            status=status,
            short_doc=short_doc,
            config_backend=config_backend,
        )
        return fn

    return decorator
