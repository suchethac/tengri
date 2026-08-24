# SPDX-License-Identifier: BSD-3-Clause
"""SpatialModelComponent: astronomer-facing base for spatial physics blocks.

Mirror of :class:`tengri.components.sed_model_component.SEDModelComponent`
on the spatial side. Provides the same auto-discovery of free parameters
(class-level :class:`Distribution` attrs), the same ``reads``/``publishes``
contract, and a sensible default :meth:`apply` orchestration that:

1. Slices ``params`` by ``parameter_prefix`` (always ``"spatial_"``).
2. Pulls cross-component reads from ``state.derived``.
3. Reads the spatial grid from ``state.derived["spatial_grid_xy_kpc"]``
   (a tuple of ``(x_grid_kpc, y_grid_kpc)`` 2D arrays).
4. Calls subclass :meth:`predict`.
5. Writes the resulting profile to ``state.derived["spatial_profile_2d"]``
   and any publishes the subclass declared.

Subclasses MUST override :meth:`predict` with signature
``predict(p, profile_in, grid_kpc) -> (profile_out, published)``.

See architecture spec ``docs/dev/archive/forward-model-architecture.md`` §3.2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.parameters.priors import Distribution
from tengri.protocols.component import (
    DerivedKey,
    ForwardState,
    ParamDeclaration,
    SEDComponentConfig,
)

__all__ = ["SpatialModelComponent"]


_SPATIAL_REGISTRY: dict[str, type[SpatialModelComponent]] = {}


class SpatialModelComponent:
    """Astronomer-facing base for spatial physics blocks.

    Subclasses declare:

      * ``name`` (str) and ``parameter_prefix`` (always ``"spatial_"``)
      * Class-level :class:`Distribution` attributes: free parameters
      * Optional ``reads`` (dict of name -> units) and ``publishes``
        (dict of name -> units) for the cross-component contract
      * A ``predict(p, profile_in, grid_kpc, **reads_kwargs)`` method

    Mirror of :class:`SEDModelComponent`. Behavior is identical except:

      * Default :attr:`parameter_prefix` is ``"spatial_"``
      * The default :meth:`apply` updates
        ``state.derived["spatial_profile_2d"]`` (not ``state.sed_intrinsic``)
      * :meth:`predict` is called with ``grid_kpc`` (not ``wave``)

    """

    name: str = "spatial_component"
    parameter_prefix: str = "spatial_"
    reads: ClassVar[dict[str, str]] = {}
    publishes: ClassVar[dict[str, str]] = {"spatial_profile_2d": ""}
    config: SEDComponentConfig

    _free_param_attrs: ClassVar[tuple[str, ...]] = ()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        free_attrs: list[str] = []
        for attr_name, attr in vars(cls).items():
            if attr_name.startswith("_"):
                continue
            if isinstance(attr, Distribution):
                free_attrs.append(attr_name)
        cls._free_param_attrs = tuple(free_attrs)

        component_name = getattr(cls, "name", None)
        if component_name and component_name != "spatial_component":
            registered = _SPATIAL_REGISTRY.get(component_name)
            if registered is not None and registered is not cls:
                existing = registered
                raise ValueError(
                    f"SpatialModelComponent name collision: {component_name!r} is "
                    f"already registered to {existing.__module__}.{existing.__name__}; "
                    f"{cls.__module__}.{cls.__name__} cannot also claim it."
                )
            _SPATIAL_REGISTRY[component_name] = cls

    def __init__(self) -> None:
        self.config = SEDComponentConfig(name=self.name)

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Lift class-level Distribution attrs into ParamDeclaration tuples."""
        decls: list[ParamDeclaration] = []
        for attr_name in self._free_param_attrs:
            dist = getattr(type(self), attr_name)
            decls.append(
                ParamDeclaration(
                    name=f"{self.parameter_prefix}{attr_name}",
                    prior=dist,
                    description=dist.description,
                    units=dist.units,
                )
            )
        return decls

    def inputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component reads, derived from the ``reads`` dict."""
        return tuple(DerivedKey(name=k, units=u, description="") for k, u in self.reads.items())

    def outputs(self) -> tuple[DerivedKey, ...]:
        """Cross-component publishes, derived from the ``publishes`` dict."""
        return tuple(
            DerivedKey(name=k, units=u, description="") for k, u in self.publishes.items()
        )

    def precompute(self, **kwargs: Any):
        """No-op by default. Subclasses override if they need cached tensors."""
        from tengri.protocols.spatial import SpatialComponentState

        return SpatialComponentState(name=self.name)

    def apply(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> ForwardState:
        """Default orchestration: slice params, lookup grid, call predict, write state."""
        prefix_len = len(self.parameter_prefix)
        p_sliced = {
            k[prefix_len:]: v for k, v in params.items() if k.startswith(self.parameter_prefix)
        }

        input_kwargs = {}
        for input_key in self.inputs():
            key_name = input_key.name
            if key_name not in state.derived:
                raise KeyError(
                    f"SpatialModelComponent {self.name!r} declares required input "
                    f"{key_name!r}, but it is not in state.derived. "
                    f"Available: {list(state.derived.keys())}"
                )
            input_kwargs[key_name] = state.derived[key_name]

        if "spatial_grid_xy_kpc" not in state.derived:
            raise KeyError(
                f"SpatialModelComponent {self.name!r}: state.derived does not "
                f"contain 'spatial_grid_xy_kpc'. The grid must be set up by "
                f"SpatialModel or the caller before running spatial components. "
                f"Available: {list(state.derived.keys())}"
            )
        grid_kpc = state.derived["spatial_grid_xy_kpc"]

        if "spatial_profile_2d" in state.derived:
            profile_in = state.derived["spatial_profile_2d"]
        else:
            x_grid, _ = grid_kpc
            profile_in = jnp.zeros_like(x_grid)

        profile_out, published = self.predict(p_sliced, profile_in, grid_kpc, **input_kwargs)

        new_derived = state.derived.with_(
            spatial_profile_2d=profile_out,
            **published,
        )
        return state.with_(derived=new_derived)

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        profile_in: jnp.ndarray,
        grid_kpc: Any,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX prediction step. MUST be implemented by subclasses.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped (``p["re_kpc"]``, not
            ``p["spatial_re_kpc"]``).
        profile_in : ndarray, shape (ny, nx)
            Running 2D surface-brightness profile from upstream components,
            or zeros if this component is first.
        grid_kpc : tuple of (ndarray, ndarray), each shape (ny, nx)
            ``(x_grid_kpc, y_grid_kpc)``: the 2D spatial coordinate grids.
        **inputs : ndarray
            Cross-component reads keyed by names in :attr:`reads`.

        Returns
        -------
        tuple[ndarray, mapping[str, ndarray]]
            ``(profile_out, published)``. ``profile_out`` has shape
            ``(ny, nx)`` and replaces (or extends) ``profile_in``.
            ``published`` is a dict matching :attr:`publishes` keys
            other than ``spatial_profile_2d``, which the base class
            handles automatically.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.predict() must be implemented by subclass"
        )
