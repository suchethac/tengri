# SPDX-License-Identifier: BSD-3-Clause
"""Shared accessor for the build-time term band response of an additive emitter.

An additive emitter (dust IR, X-ray, radio) is a *sum of rank-1 terms* — each a
scalar amplitude times a spectral shape fixed by the emitter's shape parameters.
Because the filter integral is linear, each term's per-filter response is a
build-time constant and the band flux collapses to ``sum_k A_k * R_kf``. The
response is built once in ``tengri.SEDModel._additive_term_band_response``
and threaded into the JIT as ``template_data``; this module is the single reader,
so the namespace key cannot drift between the producer and its consumers.

See ``docs/dev/sed-model-components.md`` and #1109.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Key under which each emitter's namespace carries its term band response.
TERM_BAND_RESPONSE_KEY = "term_band_response"


def term_band_response(template_data: Any, name: str) -> Mapping[str, Any] | None:
    """Read an emitter's build-time term band response out of ``template_data``.

    Parameters
    ----------
    template_data : Any
        The threaded template data. Any non-mapping value (including ``None``,
        the common case when the model was built without ``approx=WavePrecomp``)
        yields ``None``.
    name : str
        Emitter namespace — ``"xray"`` or ``"radio"``.

    Returns
    -------
    mapping or None
        ``{"R": (n_terms, n_filters), "lam_ref": (n_terms,), "S_ref": (n_terms,)}``,
        or ``None`` when no response was built — in which case the caller must keep
        the exact per-call dense filter integral. Term order matches the emitter's
        ``emission_terms`` dict order.
    """
    if not isinstance(template_data, Mapping):
        return None
    namespace = template_data.get(name)
    if not isinstance(namespace, Mapping):
        return None
    return namespace.get(TERM_BAND_RESPONSE_KEY)
