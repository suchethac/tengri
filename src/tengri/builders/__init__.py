# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories that emit config dicts for :meth:`tengri.SEDModel.build`.

The :meth:`SEDModel.build` nested-dict grammar is concise and
JSON-serializable, but the inner-key namespace is opaque to IDEs:
typing ``{'type': 'dpl', 'beat': Uniform(1, 3)}`` (note the typo) only
fails at construction time. This subpackage layers per-variant
callables on top that:

- Carry real :class:`inspect.Signature` objects, so hovering in a
  notebook or IDE surfaces the actual parameter names and short docs.
- Validate kwargs eagerly: unknown parameter names raise
  :class:`TypeError` with the list of valid alternatives.
- Return plain dicts in the exact grammar
  :meth:`SEDModel.build` already accepts, so the factory output and the
  hand-written-dict path are freely interchangeable and equally
  JSON-round-trippable.

Examples
--------
>>> from tengri import SEDModel, builders, FREE, FIXED, Uniform, Fixed
>>> model = SEDModel.build(
...     ssp_data=ssp,
...     observation=obs,
...     sfh=builders.sfh.dpl(beta=Uniform(1, 3)),
...     redshift=Fixed(0.05),
... )

Scope
-----

- :mod:`tengri.builders.sfh`: one factory per canonical SFH variant
  in :data:`SFH_REGISTRY` (PR #79).
- :mod:`tengri.builders.igm` / :mod:`~tengri.builders.radio` /
  :mod:`~tengri.builders.xray`; simple components with
  variant-string selectors and component-wide parameter sets.
- :mod:`tengri.builders.neb`: nebular emission backends
  (none/ssp/cue/cloudy/cb19).
- :mod:`tengri.builders.dust` (+ nested
  :mod:`~tengri.builders.dust.emission`); attenuation
  top-level + IR emission sub-block.
- :mod:`tengri.builders.agn`: top-level ``composable``
  orchestrator + five sub-block submodules (``disc``, ``torus``,
  ``lines``, ``feii``, ``atten``).

See also
--------
tengri.recipes
    Curated starting points (returns dicts; pre-dates the factory layer
    and continues to work).
"""

from __future__ import annotations

from tengri._completion import curated_dir
from tengri.builders import agn, dust, igm, neb, radio, sfh, xray

__all__ = ["agn", "dust", "igm", "neb", "radio", "sfh", "xray"]


__dir__ = curated_dir(__all__)
