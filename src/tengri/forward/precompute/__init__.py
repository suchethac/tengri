"""Precompute algorithm, Protocol, and registry for template-based components.

Key extension point: implement :class:`PrecomputeModule` to make a component
eligible for preintegration (filter-convolution cached at model-build time).

Usage::

    from tengri.forward.precompute import PrecomputeModule, PreintegratedGrid
    from tengri.forward.precompute import preintegrate_grid, slice_fixed_axes

Where precompute strategies live
--------------------------------

This package is the **protocol + registry** for precompute. The concrete
strategy for each component lives next to that component's physics::

    components/agn/{disc,cat3d,silva04,skirtor}_precompute.py
    components/dust/{dust_analytic,dust_emission}_precompute.py
    components/nebular/{cloudy,cue,mappings_photo,feltre}_precompute.py
    components/stellar/sps/precompute.py

Each implements :class:`PrecomputeModule` and registers via
``@register_precompute_module(...)``. To add preintegration to a new
component, copy the smallest existing implementation
(``stellar/sps/precompute.py``) and adapt.
"""

from tengri.forward.precompute.protocol import PrecomputeModule
from tengri.forward.precompute.registry import registered_components, resolve
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    precompute_template_photometry,
)
from tengri.utils.grid_interp import (
    PreintegratedGrid,
    PreintegratedLines,
    interp_nd_triweight,
    preintegrate_grid,
    preintegrate_lines,
    slice_fixed_axes,
)

__all__ = [
    "PrecomputeModule",
    "PreintegratedGrid",
    "PreintegratedLines",
    "build_template_photometry_lookup",
    "interp_nd_triweight",
    "precompute_template_photometry",
    "preintegrate_grid",
    "preintegrate_lines",
    "registered_components",
    "resolve",
    "slice_fixed_axes",
]
