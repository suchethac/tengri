# SPDX-License-Identifier: BSD-3-Clause
"""Shared accessor for the build-time term band response of an additive emitter.

An additive emitter (dust IR, X-ray, radio) is a *sum of rank-1 terms*: each a
scalar amplitude times a spectral shape fixed by the emitter's shape parameters.
Because the filter integral is linear, each term's per-filter response is a
constant of the evaluation redshift and the band flux collapses to
``sum_k A_k * R_kf``. The response is built once, tabulated over redshift, in
``tengri.SEDModel._additive_term_band_response`` and threaded into the JIT as
``template_data``; this module is the single reader,
so the namespace key cannot drift between the producer and its consumers.

See ``docs/dev/sed-model-components.md`` and #1109.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from tengri.components._z_response import interp_z_table

#: Key under which each emitter's namespace carries its term band response.
TERM_BAND_RESPONSE_KEY = "term_band_response"


def term_band_response(
    template_data: Any, name: str, params: Mapping[str, Any] | None = None
) -> Mapping[str, Any] | None:
    """Read an emitter's term band response at the evaluation redshift.

    The response is tabulated over redshift at build time (a term's filter
    integral and, for radio, its shape both move with ``z``), so the reader
    interpolates it at the redshift this evaluation runs at, the merged
    ``params["redshift"]`` the emitter's own ``apply`` receives. A model with one
    ``Fixed`` redshift has a one-node table; reading it needs no redshift.

    Parameters
    ----------
    template_data : Any
        The threaded template data. Any non-mapping value (including ``None``,
        the common case when the model was built without ``approx=WavePrecomp``)
        yields ``None``.
    name : str
        Emitter namespace: ``"xray"`` or ``"radio"``.
    params : mapping, optional
        The emitter's params, carrying the bare ``redshift``. Needed only when
        the table has more than one redshift node.

    Returns
    -------
    mapping or None
        ``{"R": (n_terms, n_filters), "lam_ref": (n_terms,), "S_ref": (n_terms,)}``
        at the evaluation redshift, or ``None`` when no response was built: in
        which case the caller must keep the exact per-call dense filter integral.
        Term order matches the emitter's ``emission_terms`` dict order.

    Notes
    -----
    **JIT-compatible**: yes.
    """
    if not isinstance(template_data, Mapping):
        return None
    namespace = template_data.get(name)
    if not isinstance(namespace, Mapping):
        return None
    table = namespace.get(TERM_BAND_RESPONSE_KEY)
    if table is None:
        return None
    nodes = table["ln1pz"]
    if nodes.shape[0] == 1:
        return {"R": table["R"][0], "lam_ref": table["lam_ref"], "S_ref": table["S_ref"][0]}
    from tengri.parameters.resolve import require_redshift

    z = require_redshift(params or {}, f"components.{name}.term_band_response")
    return {
        "R": interp_z_table({"ln1pz": nodes, "values": table["R"]}, z),
        "lam_ref": table["lam_ref"],
        "S_ref": interp_z_table({"ln1pz": nodes, "values": table["S_ref"]}, z),
    }


def term_shape_response(
    template_data: Any, name: str, params: Mapping[str, Any], terms: tuple[str, ...]
) -> Mapping[str, Any] | None:
    """Read an emitter's shape-only term tables at its current shape parameters.

    The fallback when no exact band response exists (a shape parameter is free). Each term's
    table gives the band flux of that term's shape, normalized at the term's ``lam_ref``, so the
    component multiplies it by the term evaluated at ``lam_ref``, which carries the amplitude
    exactly. See :mod:`tengri.components._term_shape_table`.

    Parameters
    ----------
    template_data : Any
        The threaded template data; any non-mapping value yields ``None``.
    name : str
        Emitter namespace: ``"xray"`` or ``"radio"``.
    params : mapping
        The emitter's params, carrying the free shape parameters the tables read.
    terms : tuple of str
        Term keys in the emitter's ``emission_terms`` order.

    Returns
    -------
    mapping or None
        ``{"R": (n_terms, n_filters), "lam_ref": (n_terms,)}``, or ``None`` when any term has no
        table (the caller keeps the dense per-call integral for the whole emitter).

    Notes
    -----
    **JIT-compatible**: yes.
    """
    from tengri.components._term_shape_table import (
        TERM_SHAPE_TABLE_KEY,
        lookup_term_shape,
    )

    if not isinstance(template_data, Mapping):
        return None
    namespace = template_data.get(name)
    if not isinstance(namespace, Mapping):
        return None
    tables = namespace.get(TERM_SHAPE_TABLE_KEY)
    if tables is None:
        return None
    if any(tables.get(term) is None for term in terms):
        return None
    rows = [lookup_term_shape(tables[term], params) for term in terms]
    lam_ref = [tables[term]["lam_ref"] for term in terms]
    return {"R": jnp.stack(rows), "lam_ref": jnp.stack(lam_ref)}
