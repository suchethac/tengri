# SPDX-License-Identifier: BSD-3-Clause
"""Params dicts derived from a component's own declarations, not hand-rolled.

A hand-written "complete" params dict is a copy of a declaration, and a copy
cannot disagree with its source until someone adds a parameter — at which point
the component indexes past the end of the dict and raises ``KeyError`` from
whichever site happens to read first. That has now happened three times on the
X-ray component alone:

* ``xray_log_nh`` (#870, #768). This is why
  ``tests/integration/test_retrace_guards.py`` stopped hand-rolling and grew the
  local ``_component_default_params`` helper that this module generalizes.
* ``xray_det_hmxb`` / ``xray_det_lmxb`` (#1706), which broke 50 tests across six
  integration files — every X-ray file that had kept hand-rolling. The one that
  derives was untouched (#1832).

Both breakages landed on a tree auto-marked ``slow`` (``_SLOW_TREES`` in
``tests/conftest.py``), deselected from the default run *and* from the PR gate,
so neither surfaced until a scheduled run a day later.

Deriving closes the class: a newly declared parameter arrives at its declared
default rather than as a ``KeyError``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def component_params(*components: Any, **overrides: Any) -> dict[str, jnp.ndarray]:
    """Build a params dict from each component's declared defaults.

    Parameters
    ----------
    *components : SEDComponent
        Components whose ``declared_parameters()`` seed the dict. Later
        components win over earlier ones where declarations overlap.
    **overrides : float or array_like
        Values pinned by the caller, applied last. Use these for the quantities
        a test's assertions actually depend on — everything else rides on the
        declaration, so adding a parameter upstream cannot break the caller.

    Returns
    -------
    dict of str to ndarray
        Full parameter dict, every value passed through ``jnp.asarray``.

    Notes
    -----
    Not JIT-traced: this runs at fixture-construction time and builds the
    concrete dict a component's ``apply()`` is then called with.

    Examples
    --------
    >>> params = component_params(XRaySEDComponent(), redshift=0.1)  # doctest: +SKIP
    >>> params["xray_det_hmxb"]  # declared default, not hand-written  # doctest: +SKIP
    Array(0., dtype=float64)
    """
    params: dict[str, jnp.ndarray] = {}
    for component in components:
        for decl in component.declared_parameters():
            params[decl.name] = jnp.asarray(decl.prior.default)
    params.update({name: jnp.asarray(value) for name, value in overrides.items()})
    return params
