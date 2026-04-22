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
    ) -> Any:
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
        Any
            Preintegrated result (PreintegratedGrid or component-specific dataclass).

        Notes
        -----
        **JIT-compatible**: no — factory function, runs at model-init time outside JIT.

        Implementations must auto-collapse grid axes whose parameters are Fixed
        in ``parameters``, reducing dimensionality and lookup cost.
        """
        ...

    def build_lookup(self, preint: Any, **kwargs: Any) -> Any:
        """Build JIT-compiled lookup function from preintegrated grid.

        Parameters
        ----------
        preint : Any
            Output of :meth:`precompute`.
        **kwargs : Any
            Component-specific options.

        Returns
        -------
        callable
            JIT-compiled function: ``(scale, *free_params) -> photometry_array``,
            where ``photometry_array`` has shape matching the filter set passed
            to precompute. [erg/s/cm^2/Hz]

        Notes
        -----
        **JIT-compatible**: yes — returned function must be decorated with
        ``@jax.jit`` and use only ``jnp`` primitives for gradient-safe
        interpolation.

        **Gradient-safe**: yes — differentiable w.r.t. all free parameters.
        """
        ...
