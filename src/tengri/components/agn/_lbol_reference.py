# SPDX-License-Identifier: BSD-3-Clause
"""Float32 safety for AGN luminosity evaluation (#1206, #2321).

The composable runner and the monolithic AGN component paths share one
reference-evaluation factoring: when operating in pure float32, blocks are
evaluated at a low reference L_bol to keep internal sums finite, then the true
magnitude is recovered in log space.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
from jax import Array

from tengri.utils.physics_constants import L_SUN
from tengri.utils.scale import apply_log10_scale

#: log10 of the solar luminosity [dex], for folding the AGN bolometric scale
#: into log space (float32 safety, #1206). L_SUN ~3.828e33 erg/s.
_LOG10_L_SUN: float = math.log10(L_SUN)

#: Reference AGN ``agn_log_lbol`` (= log10(L_bol/L_sun)) at which every block is
#: evaluated for the float32 factoring (#1206). Chosen so L_bol = 1e10 erg/s: low
#: enough that the *squares* of the internal bolometric integrals stay in float32
#: range (``(1e10)**2 = 1e20 << 3.4e38``), yet high enough that the runner's
#: ``max(faceon/L_sun, 1e-30)`` zero-protection floor never engages (faceon/L_sun
#: ~1e-23, seven decades clear). The true 10^agn_log_lbol is re-applied afterward.
_AGN_LBOL_REF: float = 10.0 - _LOG10_L_SUN


def reference_evaluation(
    agn_log_lbol: Array,
    wave: Array,
) -> tuple[Array, bool, Array]:
    r"""Apply float32 reference evaluation guard to AGN bolometric luminosity.

    When operating in pure float32, an erg/s-scale luminosity overflows before
    the log-domain path is reached. This guard evaluates blocks at a reference
    ``agn_log_lbol`` value (low enough that all integrals stay finite in
    float32), then rescales in log space afterward.

    The reference-evaluation split — shape on true ``agn_log_lbol``, magnitude
    on reference — is exact for shape-invariant blocks, the SKIRTOR torus
    template and the power-law disc scale linearly with L_bol. Since the
    ``agn_log_lbol_shape`` hand-off below, exact for the multicolor disc too:
    the true L_bol drives the temperature and geometry while only the magnitude
    is factored, and the magnitude is linear by construction. Measured in
    float64, where float32 round-off cannot mask a shape error: max relative
    deviation from a direct evaluation at the true L_bol is 2.2e-16 at log
    L_bol 11–14. Without the shape hand-off the same comparison is off by 100%
    at log L_bol = 11 and 3685% at 14.

    Parameters
    ----------
    agn_log_lbol : float or array
        Input AGN bolometric luminosity :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    wave : array
        Wavelength array. Its dtype is inspected to detect float32 mode.

    Returns
    -------
    agn_log_lbol_eval : float or array
        The luminosity to pass to blocks for evaluation. When float32 is
        enabled, this is the reference value ``_AGN_LBOL_REF``; otherwise the
        true input value.
    use_ref : bool
        Whether reference evaluation is active (True in float32 mode).
    log_scale_offset : float
        The log10-space offset to apply to block outputs: ``agn_log_lbol -
        _AGN_LBOL_REF`` when ``use_ref`` is True, else 0.0.
    """
    use_ref = wave.dtype == jnp.float32
    if use_ref:
        agn_log_lbol_eval = jnp.full_like(
            jnp.asarray(agn_log_lbol, dtype=wave.dtype), _AGN_LBOL_REF
        )
        log_scale_offset = agn_log_lbol - _AGN_LBOL_REF
    else:
        agn_log_lbol_eval = agn_log_lbol
        log_scale_offset = 0.0
    return agn_log_lbol_eval, use_ref, log_scale_offset


def rescale(value: Array | dict, offset: Array) -> Array | dict:
    """Apply log10-space offset to a scalar, array, or dict of arrays.

    Parameters
    ----------
    value : array or dict
        The value(s) to rescale. If a dict, every value is rescaled.
    offset : array
        The log10-space offset to apply via ``apply_log10_scale``.

    Returns
    -------
    rescaled : same type as value
        The rescaled value(s).
    """
    if isinstance(value, dict):
        return {k: apply_log10_scale(v, offset) for k, v in value.items()}
    return apply_log10_scale(value, offset)
