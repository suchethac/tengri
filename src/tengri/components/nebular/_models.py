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

from collections.abc import Callable
from dataclasses import dataclass
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
    """

    callable: Callable | None
    citation: str = ""
    status: str = "production"
    short_doc: str = ""

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        fn = object.__getattribute__(self, "callable")
        if fn is None:
            raise TypeError("The 'none' nebular model has no callable.")
        return fn(*args, **kwargs)


NEBULAR_MODELS: dict[str, NebularRegistryEntry] = {}

#: Nebular backends that carry their own thermal free-free continuum (issue #2346).
#: These backends publish ``sed_nebular`` with a free-free component, so the radio
#: block's Murphy+2011 ``radio_freefree`` term should not run in addition
#: (would double-count the thermal emission). Keyed on the factory's ``nebular_backend``
#: vocabulary: ``"baked_in"``, ``"cb19"``, ``"cloudy_grid"``, ``"mappings"``,
#: ``"cue"``, ``"shock"``, or ``None``.
#:
#: - ``"cue"``: Li et al. (2025) neural emulator, carries a Q_H-normalized
#:   continuum with free-free.
#: - ``"cloudy_grid"``: tabulated Cloudy grid (Byler+2017 axis), carries
#:   free-free.
#: - All others (``"baked_in"``, ``"cb19"``, ``"mappings"``, ``"shock"``, ``None``):
#:   do NOT carry free-free. ``"cb19"`` publishes zeros (no continuum at all).
NEBULAR_BACKENDS_WITH_FREEFREE_CONTINUUM: frozenset[str] = frozenset({"cue", "cloudy_grid"})


def nebular_backend_carries_freefree(name: str | None) -> bool:
    """Check if a declared nebular backend carries a free-free continuum.

    Used by :func:`tengri.forward.component_factory.build_components` to
    auto-resolve ``radio.include_freefree=None`` to ``False`` when the declared
    nebular backend publishes its own thermal free-free (issue #2346): one
    thermal term on the whole SED grid, not separate radio + nebular terms.

    Parameters
    ----------
    name : str or None
        Nebular backend name from the factory's vocabulary:
        ``"baked_in"`` (default), ``"cb19"``, ``"cloudy_grid"``, ``"mappings"``,
        ``"cue"``, ``"shock"``, or ``None`` to omit nebular.

    Returns
    -------
    bool
        ``True`` if the backend carries a free-free continuum (Cue, CloudyGrid).
        ``False`` otherwise, including ``None``.

    Notes
    -----
    The rule is keyed on the *declared* backend name, never on the published
    ``sed_nebular`` array or an approximation flag: the fast nebular grid path
    publishes ``sed_nebular`` as all zeros for the same physical model, and
    structural decisions must never be gated on approximation toggles.
    """
    return name in NEBULAR_BACKENDS_WITH_FREEFREE_CONTINUUM


def register_nebular_model(
    name: str,
    *,
    citation: str = "",
    status: str = "production",
    short_doc: str = "",
) -> Callable[[Callable | None], Callable | None]:
    """Register a nebular emission backend in :data:`NEBULAR_MODELS`."""

    def decorator(fn: Callable | None) -> Callable | None:
        NEBULAR_MODELS[name] = NebularRegistryEntry(
            callable=fn, citation=citation, status=status, short_doc=short_doc
        )
        return fn

    return decorator
