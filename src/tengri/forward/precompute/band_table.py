# SPDX-License-Identifier: BSD-3-Clause
"""Band-table providers: the adapter contract for build-time band tables.

A band table is a build-time integral of one component's emission through the model's filters,
threaded into the JIT as a pytree. A provider is a precompute adapter module that owns the
construction of the tables for one component. It answers one call per slot with either a
:class:`BandTable` or a :class:`Decline` that carries the reason it gives.

The provider is a sibling of :class:`~tengri.forward.precompute.protocol.PrecomputeModule`, not
a member of it: a module can implement both, but the band-table contract is a separate seam.
:class:`~tengri.forward.sed_model.SEDModel` keeps one memo per component, ``_band_tables``, which
holds the verdict for each slot it has asked for.

See ADR-0021 (``docs/adr/0021-band-table-providers.md``) for the decisions behind this shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class Decline:
    """A provider's refusal to build a band table.

    Parameters
    ----------
    reason : str or None
        The reason, as the adapter words it. ``None`` is a silent gate: the table is not built
        and no reason is recorded.
    """

    reason: str | None


@dataclass(frozen=True)
class BandTable:
    """A built band table, as threaded into the JIT.

    Parameters
    ----------
    arrays : Any
        The pytree the adapter built, key for key. Its structure, dtypes and values are
        JIT-structural: a change to any of them changes the compiled program.
    """

    arrays: Any


@dataclass(frozen=True)
class BandTableContext:
    """Everything a band-table provider sees about the model that asks for a table.

    Parameters
    ----------
    spec : Parameters
        The model's parameter specification.
    wave : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    filters : tuple
        The raw filter curves of the observation.
    padded_filters : tuple
        The filter curves padded to the shared wavelength grid.
    z_nodes : ndarray, shape (n_z,), or None
        Redshift nodes of a free-redshift table; ``None`` at a fixed redshift.
    wave_precomp : object
        The ``WavePrecomp`` approximation settings in force.
    chain : list
        The component chain the table is built from (chain0, before any grid chain attaches).
    dust_ir_diffuse_screen : object
        The diffuse-screen dust IR configuration, or ``None``.
    atten_free_ok : frozenset of str
        Parameter names that leave the attenuation shape fixed. Passed, not copied.
    agn_axis_grids : object
        The AGN axis-grid overrides, or ``None``.
    ask : callable
        ``ask(name, slot)`` returns the memo entry of another slot of the same component.
    """

    spec: Any
    wave: Any
    filters: Any
    padded_filters: Any
    z_nodes: Any
    wave_precomp: Any
    chain: Any
    dust_ir_diffuse_screen: Any
    atten_free_ok: Any
    agn_axis_grids: Any
    ask: Callable[[str, str], Any]


@runtime_checkable
class BandTableProvider(Protocol):
    """A precompute adapter that owns the band tables of one component.

    Attributes
    ----------
    BAND_SLOTS : tuple of str
        The slots this adapter builds, for example ``("band_table",)``.
    """

    BAND_SLOTS: tuple[str, ...]

    def band_table(
        self, ctx: BandTableContext, *, component: str, slot: str
    ) -> BandTable | Decline:
        """Return the band table of one slot of one component, or a refusal of it.

        Parameters
        ----------
        ctx : BandTableContext
            The model state the table is built from.
        component : str
            The component whose slot is requested.
        slot : str
            One of ``BAND_SLOTS``.

        Returns
        -------
        BandTable or Decline
            The built table, or the adapter's refusal with its reason.
        """
        ...


def arrays_of(entry: BandTable | Decline | None) -> Any:
    """Return the arrays of a built band table, or None for a refusal or an absent entry.

    Parameters
    ----------
    entry : BandTable, Decline or None
        A memo entry.

    Returns
    -------
    Any or None
        ``entry.arrays`` for a :class:`BandTable`, otherwise None.
    """
    return entry.arrays if isinstance(entry, BandTable) else None
