# SPDX-License-Identifier: BSD-3-Clause
"""User-facing model presets.

Self-describing: query menus and details via Python rather than REGISTRY.md.
New presets register themselves via the ``@register_preset(name, ...)`` decorator.

Quick start::

    from tengri.presets import synthesizer_default, list_presets, describe_preset

    # Build a ready-to-fit model
    model, params = synthesizer_default()

    # List all available presets
    presets = list_presets()  # _RegistryTable, one row per preset

    # Get full details on one preset
    details = describe_preset("synthesizer_default")
    print(details["citations"])  # ['Bruzual_2003', 'Calzetti_2000', ...]

Two factory contracts coexist in the registry:

- **Model presets** (e.g. ``synthesizer_default``) return ``(SEDModel,
  Parameters)``: ready to fit.
- **Galaxy-type parameter presets** (``starforming``, ``quiescent``,
  ``high_z``, ``photoz``, ``jwst_spec``, ``agn_host``; in
  :mod:`~tengri.presets.param_presets`) return ``(Parameters,
  SEDModelConfig)``; the legacy expert surface consumed by
  :class:`tengri.Galaxy` via :func:`resolve_preset`. For new code prefer the
  grammar recipes in :mod:`tengri.recipes`.

"""

from __future__ import annotations

from tengri.presets._registry import (
    describe_preset,
    list_presets,
    register_preset,
)
from tengri.presets.param_presets import (
    agn_host,
    describe,
    high_z,
    jwst_spec,
    photoz,
    quiescent,
    resolve_preset,
    starforming,
)
from tengri.presets.synthesizer import synthesizer_default

__all__ = [
    "agn_host",
    "describe",
    "describe_preset",
    "high_z",
    "jwst_spec",
    "list_presets",
    "photoz",
    "quiescent",
    "register_preset",
    "resolve_preset",
    "starforming",
    "synthesizer_default",
]
