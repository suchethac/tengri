# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Callable factories that emit config dicts for :meth:`tengri.SEDModel.build`.

The :meth:`SEDModel.build` nested-dict grammar is concise and
JSON-serialisable, but the inner-key namespace is opaque to IDEs:
typing ``{'type': 'dpl', 'beat': Uniform(1, 3)}`` (note the typo) only
fails at construction time. This subpackage layers per-variant
callables on top that:

- Carry real :class:`inspect.Signature` objects, so hovering in a
  notebook or IDE surfaces the actual parameter names and short docs.
- Validate kwargs eagerly — unknown parameter names raise
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
- :mod:`tengri.builders.sfh` — Phase 1 (PR #79). One factory per
  canonical SFH variant in :data:`SFH_REGISTRY`.
- :mod:`tengri.builders.igm` / :mod:`~tengri.builders.radio` /
  :mod:`~tengri.builders.xray` — Phase 2. Simple components with
  variant-string selectors and component-wide parameter sets.
- :mod:`tengri.builders.neb` — Phase 3. Nebular emission backends
  (none/ssp/cue/cloudy/cb19).

Dust and AGN factories are tracked in #82 and #83.

See also
--------
tengri.recipes
    Curated starting points (returns dicts; pre-dates the factory layer
    and continues to work).
"""

from __future__ import annotations

from tengri.builders import igm, neb, radio, sfh, xray

__all__ = ["igm", "neb", "radio", "sfh", "xray"]
