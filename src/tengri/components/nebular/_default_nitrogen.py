# SPDX-License-Identifier: BSD-3-Clause
"""Default N/O--O/H abundance relation for the nebular backends (#2693).

Photoionization-grid backends (CB_19, CloudyGrid, MAPPINGS) bake a nitrogen
abundance that rises with oxygen into their CLOUDY grids, and ``neb_dno`` is
an offset *from* it. Cue takes [N/O] as a free input, so its default has to
supply the same metallicity dependence explicitly: :func:`default_nitrogen_offset`
is that default, and ``gas_logno`` is an offset from it.

The shipped grids store only their axes (no N/H or O/H table) and the
converters in ``scripts/`` carry no N/O formula, so the relation they embody
is not recoverable from the files; the form adopted is the empirical
two-regime relation of Nicholls et al. (2017), with the grids' agreement
checked at the line-ratio level (see the regression test).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from tengri.components.nebular._constants import _LOG_OH_SOLAR

#: Primary-production floor, log10(N/O) (Nicholls+2017 N/O fit).
_LOG_NO_PRIMARY: float = -1.732
#: Secondary-production intercept: log10(N/O)_sec = log10(O/H) + 2.19
#: (same fit), with O/H the absolute number ratio.
_LOG_NO_SECONDARY_OFFSET: float = 2.19
_LN10: float = math.log(10.0)


def _log_no_absolute(log_oh):
    """log10(N/O) = log10(10**-1.732 + 10**(log10(O/H) + 2.19)), overflow-safe."""
    return (
        jnp.logaddexp(_LOG_NO_PRIMARY * _LN10, (log_oh + _LOG_NO_SECONDARY_OFFSET) * _LN10) / _LN10
    )


@jax.jit
def default_nitrogen_offset(gas_logz):
    r"""Default [N/O] (dex relative to solar) at gas metallicity ``gas_logz``.

    The N/O fit of Nicholls et al. (2017, MNRAS 466, 4403) gives the nitrogen-to-oxygen
    ratio of the interstellar medium as a primary floor plus a secondary term
    that grows with oxygen:

    .. math::

        \log_{10}({\rm N/O}) = \log_{10}\left(10^{-1.732}
            + 10^{\log_{10}({\rm O/H}) + 2.19}\right).

    The two terms are equal at :math:`12+\log({\rm O/H}) = 8.08`: below it
    N/O is flat (primary nitrogen), above it N/O rises linearly with O/H
    (secondary nitrogen). The regime switch is in *absolute* O/H, so the
    relation is evaluated at ``log(O/H) = log(O/H)_sun + gas_logz``, with
    ``log(O/H)_sun = -3.07`` (12+log(O/H) = 8.93) the CB19 solar scale; the
    FSPS grids' scale is not recorded on disk. Anchoring at Asplund's 8.69
    instead would move [N/O] by 0.17 dex at 0.1 Z_sun and 0.05 dex at
    0.3 Z_sun. The relation is referenced to its own solar value, so the
    return is 0 at ``gas_logz = 0`` (to float64 rounding, below 1e-16 dex,
    which the float32 network input absorbs).

    Parameters
    ----------
    gas_logz : float or array_like
        Gas metallicity :math:`\log_{10}(Z/Z_\odot)` (O/H scales with Z).

    Returns
    -------
    jax.Array
        :math:`[{\rm N/O}]` in dex relative to solar, same shape as the input.
        Pure JAX: jit-, vmap- and grad-safe. Evaluated in the widest enabled
        float dtype, so eager, traced and batched callers round to the same
        float32 value.

    References
    ----------
    .. [1] Nicholls et al. 2017, MNRAS 466, 4403.
    """
    # Evaluate in the widest float JAX has enabled, so the float32 rounding of
    # the result does not depend on how the caller vectorises or fuses it.
    gas_logz = jnp.asarray(gas_logz, dtype=jnp.result_type(float))
    # The solar reference runs through the identical op sequence on zeros.
    return _log_no_absolute(_LOG_OH_SOLAR + gas_logz) - _log_no_absolute(
        _LOG_OH_SOLAR + gas_logz * 0.0
    )
