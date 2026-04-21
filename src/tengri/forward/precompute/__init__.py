"""Precompute algorithm, Protocol, and registry for template-based components.

Key extension point: implement :class:`PrecomputeModule` to make a component
eligible for preintegration (filter-convolution cached at model-build time).

Usage::

    from tengri.forward.precompute import PrecomputeModule, PreintegratedGrid
    from tengri.forward.precompute import preintegrate_grid, slice_fixed_axes
"""

from tengri.forward.precompute.grid import (
    PreintegratedGrid,
    PreintegratedLines,
    interp_nd_triweight,
    preintegrate_grid,
    preintegrate_lines,
    slice_fixed_axes,
)
from tengri.forward.precompute.protocol import PrecomputeModule
from tengri.forward.precompute.registry import registered_components, resolve
from tengri.forward.precompute.templates import (
    build_template_photometry_lookup,
    precompute_template_photometry,
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
