# SPDX-License-Identifier: BSD-3-Clause
"""One sampler driver shared by every inference engine.

``target`` holds the engine-independent problem (:class:`FlatTarget`, :class:`LatentLayout`)
and the metric layer; ``chains`` the multi-chain layer; ``driver`` the sampler registry and
:func:`run_sampler`. Names are resolved lazily so that ``chain_program`` can import the
chain layer without importing the driver that imports it.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "FlatTarget": "target",
    "LatentLayout": "target",
    "pilot_metric": "target",
    "masked_covariance": "target",
    "Sampler": "driver",
    "SamplerResult": "driver",
    "run_sampler": "driver",
    "get_sampler": "driver",
    "sampler_names": "driver",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{module}"), name)
