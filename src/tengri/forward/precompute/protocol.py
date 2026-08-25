# SPDX-License-Identifier: BSD-3-Clause
"""Precompute Protocol, the contract each component's precompute module implements.

Each component that supports preintegration (wavelength-integral cached against
filter curves, optionally with parameter-axis triweight interpolation) exposes
a module that satisfies :class:`PrecomputeModule`. This makes
:class:`~tengri.SEDModel` discover and wire precompute uniformly across
every physics component via :mod:`tengri.forward.precompute.registry`.

Shape of a compliant module
---------------------------

Each `components/<component>/<name>_precompute.py` file defines:

- ``AXIS_PARAMS``: an ordered tuple (or dict keyed by model variant) naming the
  user-facing parameter for each grid axis. Empty tuple means no grid axes
  (scalar template, e.g. a fixed-shape IR template scaled by L_absorbed only).

  Each entry must be a name the component actually declares, the *full*
  prefixed parameter (``agn_cos_inc``, not ``cat3d_cos_inc`` and not the bare
  ``cos_inc``), because it is matched against ``Parameters.get_fixed_values()``
  keys. Nothing forces the two namespaces to agree, so a rename on either side
  severs the link silently: no axis collapses, and nothing is raised. Six
  modules shipped exactly that (#1738). The order is equally load-bearing:
  entry ``i`` must be axis ``i`` of the grid, since the position *is* the axis
  index passed to the collapse.

- ``precompute(filter_waves, filter_trans, redshift, parameters, **kwargs)``:
  build the preintegrated grid.  The function is responsible for:

    1. Loading templates (disk / memory).
    2. Building grid axes.
    3. Calling
       :func:`tengri.forward.precompute.grid.preintegrate_grid` (or the
       component-specific equivalent, K&D has its own dataclass).
    4. Auto-collapsing axes whose corresponding parameter in ``parameters`` is
       :class:`~tengri.parameters.priors.Fixed`, by calling
       :func:`tengri.forward.precompute.templates.collapse_fixed_axes`. Do not
       hand-roll this step: it was copied into eleven modules, which is how six
       of them came to declare axis names that could never match and how two
       shipped declarations that disagree with their own grid, with no error in
       either case (#1738). The shared helper carries both checks.
    5. Returning a precompute result (``PreintegratedGrid`` or component
       dataclass such as ``KDPreintegratedData``).

- ``build_lookup(preint, **kwargs)``: return a JIT-compiled callable that takes
  (scale, *free_params) → filter photometry. The free_params list matches
  AXIS_PARAMS entries whose prior is NOT Fixed.

Auto-collapse is a first-class feature, not an afterthought. A user who fixes
``dust_qpah`` should get a 1D DL07 grid instead of a 2D one, for free.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PreintegratedResult(Protocol):
    """Explicit contract for precompute result shape.

    Every :meth:`PrecomputeModule.precompute` must return an object satisfying
    this Protocol. Implementations may be:

    - A :class:`~tengri.utils.grid_interp.PreintegratedGrid` (most common).
    - A custom frozen dataclass such as
      :class:`~tengri.components.agn.kd_precompute.KDPreintegratedData` (K&D disc).
    - A dict-like object with documented keys (legacy paths, being phased out).

    Attributes
    ----------
    This is a **structural Protocol**, the caller is guaranteed only that:

    1. The object is hashable (frozen dataclass or immutable wrapper).
    2. It has a ``.data`` attribute or is itself array-like with a ``.shape``
       attribute (for shape introspection in registration validation).
    3. It is compatible with downstream :meth:`PrecomputeModule.build_lookup`
       calls that read model-specific attributes (e.g., ``.phot``, ``.axes``).

    Implementations SHOULD document any grid-indexing metadata (axis names,
    units, whether axes are sorted, etc.) that runtime lookup code expects.

    Notes
    -----
    This Protocol is **runtime-checkable** and is used by the registry validator
    to detect shape mismatches at registration time, rather than deep inside
    JIT compilation. On registration failure, the validator reports which
    attributes were accessed but missing, with module and attribute names.

    **Not enforced in all __init__ paths** (only where validators are called);
    implementations should still satisfy this contract for safe interchange.
    """

    # Intentionally sparse: we check existence of critical attributes
    # (shape, data) at runtime via hasattr, not via Protocol enforcement.


@runtime_checkable
class PrecomputeModule(Protocol):
    """Structural type for a component precompute module.

    This is a runtime-checkable Protocol, so `isinstance(module, PrecomputeModule)`
    validates that a component module exports the right surface. We use module
    objects (not classes) because each file is the natural unit of registration.

    Attributes
    ----------
    AXIS_PARAMS : tuple or dict
        Ordered tuple of parameter names defining preintegration grid axes,
        or dict mapping model variant to tuple (for multi-variant components).
        Empty tuple means scalar template (no grid axes). [dimensionless]

    Notes
    -----
    Implementations live in ``components/<component>/<component>_precompute.py``
    and are auto-discovered via :mod:`~tengri.forward.precompute.registry`.
    """

    AXIS_PARAMS: Any  # tuple[str, ...] or dict[str, tuple[str, ...]] for multi-variant components

    def precompute(
        self,
        filter_waves: list,
        filter_trans: list,
        redshift: float,
        parameters: Any,
        **kwargs: Any,
    ) -> PreintegratedResult:
        """Build preintegrated grid through filters with auto-collapse of fixed axes.

        Parameters
        ----------
        filter_waves : list of array_like
            Per-filter wavelength arrays. [Angstrom]
        filter_trans : list of array_like
            Per-filter transmission curves (normalized). [dimensionless]
        redshift : float
            Source redshift. [dimensionless]
        parameters : Any
            Free and fixed parameter specification.
        **kwargs : Any
            Component-specific options.

        Returns
        -------
        PreintegratedResult
            Preintegrated result satisfying the result shape contract.
            Typically :class:`~tengri.utils.grid_interp.PreintegratedGrid`
            or a component-specific frozen dataclass like
            :class:`~tengri.components.agn.kd_precompute.KDPreintegratedData`.

        Notes
        -----
        **JIT-compatible**: no, factory function, runs at model-init time outside JIT.

        Implementations must auto-collapse grid axes whose parameters are Fixed
        in ``parameters``, reducing dimensionality and lookup cost, by calling
        :func:`tengri.forward.precompute.templates.collapse_fixed_axes` rather
        than reimplementing the step (#1738).
        """
        ...

    def build_lookup(self, preint: PreintegratedResult, **kwargs: Any) -> Callable:
        """Build JIT-compiled lookup function from preintegrated grid.

        Parameters
        ----------
        preint : PreintegratedResult
            Output of :meth:`precompute`.
        **kwargs : Any
            Component-specific options.

        Returns
        -------
        Callable
            JIT-compiled function: ``(scale, *free_params) -> photometry_array``,
            where ``photometry_array`` has shape matching the filter set passed
            to precompute. [erg/s/cm^2/Hz]

            The callable signature is component-specific (e.g., K&D takes
            different arguments than dust models), but all return filter photometry
            broadband fluxes with units matching the precomputed grid.

        Notes
        -----
        **JIT-compatible**: yes, returned function must be decorated with
        ``@jax.jit`` and use only ``jnp`` primitives for gradient-safe
        interpolation.

        **Gradient-safe**: yes, differentiable w.r.t. all free parameters.
        """
        ...
