# SPDX-License-Identifier: BSD-3-Clause
"""Nenkova et al. 2008 CLUMPY torus block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blocks._torus_common import _C_AA_PER_S
from tengri.components.agn.torus import nenkova_torus


@register_agn_block(
    "torus",
    "nenkova",
    citation="Nenkova et al. 2008, ApJ, 685, 147",
    status="production",
    short_doc="Nenkova et al. 2008 CLUMPY radiative-transfer torus",
)
def nenkova_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_tau: float = 30.0,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""Nenkova+ 2008 CLUMPY torus block.

    Production-quality clumpy radiative-transfer torus templates. ``agn_tau``
    is the equatorial optical depth at 0.55 µm; ``agn_torus_frac`` sets the
    fraction of :math:`L_{\rm bol}` re-emitted by the torus.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol : float
    l5100_disc : array
        Ignored (kept for protocol compatibility — this block normalizes
        from ``agn_log_lbol``).
    agn_tau : float, optional
        Equatorial optical depth (5-150). Default ``30``.
    agn_torus_frac : float, optional
        Covering factor [0, 1]. Default ``0.5``.

    References
    ----------
    .. [1] Nenkova, M. et al. 2008, ApJ, 685, 147.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = nenkova_torus(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_tau=agn_tau,
        agn_torus_frac=agn_torus_frac,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
