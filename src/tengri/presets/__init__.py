# SPDX-License-Identifier: BSD-3-Clause
"""User-facing model presets.

Self-describing: query menus and details via Python rather than REGISTRY.md.
New presets register themselves via the ``@register_preset(name, ...)`` decorator.

Quick start::

    from tengri.presets import synthesizer_default, list_presets, describe_preset

    # Build a ready-to-fit model
    model, params = synthesizer_default()

    # List all available presets
    presets = list_presets()  # dict[str, dict]

    # Get full details on one preset
    details = describe_preset("synthesizer_default")
    print(details["citations"])  # ['Bruzual_2003', 'Calzetti_2000', ...]
"""

from __future__ import annotations

from tengri.presets._registry import (
    describe_preset,
    list_presets,
    register_preset,
)
from tengri.presets.synthesizer import synthesizer_default

__all__ = [
    "describe_preset",
    "list_presets",
    "register_preset",
    "synthesizer_default",
]
