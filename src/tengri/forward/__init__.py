# SPDX-License-Identifier: BSD-3-Clause
"""Forward-model orchestration: SEDModel, pipeline, kernels, precompute.

Key class: ``SEDModel``, the main forward model class.

An alternative **component-orchestrator** path is also exposed:

- :func:`build_components`: assemble an ordered list of
  :class:`SEDComponent` adapters from a flat keyword-argument call.
- :func:`run_components`: thread a
  :class:`tengri.protocols.ForwardState` through the chain.

Usage::

    from tengri import SEDModel
    from tengri.forward.result import SEDResult
    from tengri.forward import build_components, run_components
"""

from tengri.forward.component_factory import (
    IonizingQuantities,
    RadioQuantities,
    XRayQuantities,
    build_components,
    chain_summary,
    state_to_emission_lines,
    state_to_ionizing_quantities,
    state_to_radio_quantities,
    state_to_sed_components,
    state_to_sed_quantities,
    state_to_sfh_quantities,
    state_to_xray_quantities,
)
from tengri.forward.filter_preintegrate import preintegrate_ssp_filter_grid
from tengri.forward.orchestrator import run_components
from tengri.forward.result import SEDResult

__all__ = [
    "IonizingQuantities",
    "RadioQuantities",
    "SEDResult",
    "XRayQuantities",
    "build_components",
    "chain_summary",
    "preintegrate_ssp_filter_grid",
    "run_components",
    "state_to_emission_lines",
    "state_to_ionizing_quantities",
    "state_to_radio_quantities",
    "state_to_sed_components",
    "state_to_sed_quantities",
    "state_to_sfh_quantities",
    "state_to_xray_quantities",
]
