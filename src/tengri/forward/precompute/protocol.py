"""Precompute Protocol — the contract each component's precompute module implements.

Each component that supports preintegration (wavelength-integral cached against
filter curves, optionally with parameter-axis triweight interpolation) exposes
a module that satisfies :class:`PrecomputeModule`. This makes
:class:`~tengri.forward.SEDModel` discover and wire precompute uniformly across
every physics component via :mod:`tengri.forward.precompute.registry`.

Shape of a compliant module
---------------------------

Each `components/<component>/<name>_precompute.py` file defines:

- ``AXIS_PARAMS``: an ordered tuple (or dict keyed by model variant) naming the
  user-facing parameter for each grid axis. Empty tuple means no grid axes
  (scalar template, e.g. a fixed-shape IR template scaled by L_absorbed only).

- ``precompute(filter_waves, filter_trans, redshift, parameters, **kwargs)``:
  build the preintegrated grid.  The function is responsible for:

    1. Loading templates (disk / memory).
    2. Building grid axes.
    3. Calling
       :func:`tengri.forward.precompute.grid.preintegrate_grid` (or the
       component-specific equivalent — K&D has its own dataclass).
    4. Auto-collapsing axes whose corresponding parameter in ``parameters`` is
       :class:`~tengri.parameters.priors.Fixed`, via
       :func:`tengri.forward.precompute.grid.slice_fixed_axes` (or equivalent).
    5. Returning a precompute result (``PreintegratedGrid`` or component
       dataclass such as ``KDPreintegratedData``).

- ``build_lookup(preint, **kwargs)``: return a JIT-compiled callable that takes
  (scale, *free_params) → filter photometry. The free_params list matches
  AXIS_PARAMS entries whose prior is NOT Fixed.

Auto-collapse is a first-class feature, not an afterthought. A user who fixes
``dust_qpah`` should get a 1D DL07 grid instead of a 2D one, for free.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PrecomputeModule(Protocol):
    """Structural type for a component precompute module.

    This is a runtime-checkable Protocol, so `isinstance(module, PrecomputeModule)`
    validates that a component module exports the right surface. We use module
    objects (not classes) because each file is the natural unit of registration.
    """

    AXIS_PARAMS: Any  # tuple[str, ...] or dict[str, tuple[str, ...]] for multi-variant components

    def precompute(
        self,
        filter_waves: list,
        filter_trans: list,
        redshift: float,
        parameters: Any,
        **kwargs: Any,
    ) -> Any: ...

    def build_lookup(self, preint: Any, **kwargs: Any) -> Any: ...
