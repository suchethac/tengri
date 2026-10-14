# SPDX-License-Identifier: BSD-3-Clause
r"""Shape-only band tables for the rank-1 terms of an additive emitter.

An additive emitter (radio, X-ray) is a sum of rank-1 terms ``A_k * S_k(lambda; shape)``.
When a shape parameter is free the exact band response (one constant per filter) no longer
exists, and the component integrates the filters on every call. This module tabulates only the
shape: for each term it stores the band flux of the term's shape **normalized at a fixed
reference wavelength** ``lam_ref``, as a function of the shape parameters alone:

.. math::

    \tilde R_{kf}(\text{shape}) = \frac{\int S_k(\lambda;\text{shape})\, R_f(\lambda)\, d\lambda}
                                      {S_k(\lambda_{\rm ref};\text{shape})}

The amplitude is not tabulated. At predict the component evaluates the term once at
``lam_ref``, which carries the whole amplitude logic exactly (FIRRC, redshift evolution,
suppression, the Bell q split), and the band flux is ``term_k(lam_ref) * R~_kf``. The
normalization makes ``term_k(lam_ref) = A_k S_k(lam_ref)`` cancel against the table's
denominator, so the only approximation is the PCHIP interpolation of ``ln R~`` between the
shape nodes, which passes exactly through every node.

The tables are built by the registry adapters (``radio_precompute``, ``xray_precompute``), which
supply the unit-amplitude shape function; this module owns the integral, the normalization and the
lookup, so every family uses one method.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from tengri.observation.photometry import lnu_filter_integral_batch
from tengri.utils.grid_interp import interp_nd_pchip

#: Key under which an additive emitter's namespace carries its shape-only term tables.
TERM_SHAPE_TABLE_KEY = "term_shape_table"

#: ``ln`` of the band-flux floor ``1e-300``. A filter the term does not reach is zero, and
#: ``ln 0`` is undefined for the PCHIP; ``exp`` of this value is numerically zero. The floor is
#: stored in log space because ``1e-300`` itself is exactly 0.0 in float32 and would be a silent
#: zero if it ever entered a float32 array (#1568). ``-690.7755278982137`` is ``np.log(1e-300)``
#: in float64 (pinned by ``tests/unit/test_term_shape_table_ln_floor.py``).
_LN_FLOOR_LOG = -690.7755278982137


def _ln_band(band: np.ndarray) -> np.ndarray:
    """``ln(max(band, 1e-300))`` in float64, computed without the float32-unrepresentable floor.

    Bit-identical to ``np.log(np.maximum(band, 1e-300))`` for every ``band`` (for ``band > 1e-300``
    the log is above the floor and is returned unchanged; for ``band <= 0`` and below the floor
    the result is ``_LN_FLOOR_LOG``). NaN propagates, as it did before.
    """
    positive = band > 0.0
    ln_positive = np.log(np.where(positive, band, 1.0))
    floored = np.where(positive, np.maximum(ln_positive, _LN_FLOOR_LOG), _LN_FLOOR_LOG)
    return np.where(np.isnan(band), band, floored)


def build_term_shape_table(
    shape_fn: Callable[[Sequence[float]], Any],
    axes: Mapping[str, np.ndarray],
    wave: Any,
    filter_waves_padded: Any,
    filter_trans_padded: Any,
    redshift: Any,
) -> dict:
    """Tabulate one term's shape-only band flux over its shape axes.

    Parameters
    ----------
    shape_fn : callable
        ``shape_fn(values) -> S(wave)``: the term's unit-amplitude spectral shape [erg/s/Hz] on
        ``wave`` at the shape parameters ``values`` (one per axis, in ``axes`` order). Any
        overall scale is harmless: the table is normalized at ``lam_ref``.
    axes : mapping of str to ndarray
        Parameter name to its strictly ascending node values.
    wave : ndarray, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom], the grid the term is evaluated on at predict.
    filter_waves_padded, filter_trans_padded : ndarray, shape (n_filters, max_len)
        Padded observed-frame filter curves, as :func:`lnu_filter_integral_batch` takes them.
    redshift : float
        The model's single ``Fixed`` redshift. The table is not tabulated over redshift.

    Returns
    -------
    dict
        ``{"axes": {name: (n_nodes,)}, "ln_R": (*n_nodes, n_filters), "lam_ref": scalar}``.

    Raises
    ------
    ValueError
        When the term vanishes at ``lam_ref`` on some node. The normalization would divide by
        zero there, so the caller must keep the dense per-call integral.
    """
    names = tuple(axes)
    nodes = tuple(np.asarray(axes[n], dtype=np.float64) for n in names)
    mid = tuple(float(ax[ax.size // 2]) for ax in nodes)
    wave_np = np.asarray(wave, dtype=np.float64)
    peak = int(np.argmax(np.abs(np.asarray(shape_fn(mid), dtype=np.float64))))
    lam_ref = float(wave_np[peak])

    shape_len = tuple(ax.size for ax in nodes)
    n_filters = int(np.shape(filter_waves_padded)[0])
    band = np.empty((*shape_len, n_filters), dtype=np.float64)
    for idx in np.ndindex(*shape_len):
        values = tuple(float(nodes[d][i]) for d, i in enumerate(idx))
        s = jnp.asarray(shape_fn(values), dtype=jnp.result_type(float))
        ref = float(s[peak])
        if not ref > 0.0:
            raise ValueError(
                f"term shape vanishes or flips sign at lam_ref={lam_ref:.4g} A on node {values}; "
                "the normalized table cannot represent it"
            )
        row = lnu_filter_integral_batch(
            s, wave, filter_waves_padded, filter_trans_padded, redshift
        )
        band[idx] = np.asarray(row, dtype=np.float64) / ref

    return {
        "axes": {n: jnp.asarray(ax) for n, ax in zip(names, nodes)},
        "ln_R": jnp.asarray(_ln_band(band)),
        "lam_ref": jnp.asarray(lam_ref),
    }


def lookup_term_shape(table: Mapping[str, Any], params: Mapping[str, Any]) -> jnp.ndarray:
    """Shape-only band flux ``R~_f`` of one term at the current shape parameters.

    Parameters
    ----------
    table : mapping
        One term's table from :func:`build_term_shape_table`.
    params : mapping
        The emitter's params. Each table axis name is read from it.

    Returns
    -------
    ndarray, shape (n_filters,)
        ``exp`` of the PCHIP interpolant of ``ln R~``, node-exact at the nodes.

    Notes
    -----
    **JIT-compatible**: yes; the table arrays are traced template data.
    """
    names = tuple(table["axes"])
    if not names:
        return jnp.exp(table["ln_R"])
    axes = tuple(table["axes"][n] for n in names)
    point = tuple(jnp.asarray(params[n]) for n in names)
    return jnp.exp(interp_nd_pchip(table["ln_R"], axes, point))
