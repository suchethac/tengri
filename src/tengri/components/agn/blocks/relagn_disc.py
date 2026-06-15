# SPDX-License-Identifier: BSD-3-Clause
"""RELAGN relativistic Kerr accretion disc (grid-backed) block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._disc_common import _C_AA_PER_S
from tengri.components.agn.blocks._protocol import register_agn_block


@register_agn_block(
    "disc",
    "relagn",
    citation="Hagen & Done 2023, MNRAS, 521, 251",
    status="production",
    short_doc="RELAGN relativistic Kerr accretion disc (grid-backed)",
)
def relagn_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_mdot: float = -1.0,
    agn_astar: float = 0.0,
    agn_cos_inc: float = 0.86602540378443864,
    **_params,
) -> Array:
    r"""RELAGN relativistic Kerr accretion disc block.

    Interpolates the RELAGN grid (Hagen & Done 2023) which uses KYCONV
    (Dovciak, Karas & Yaqoob 2004) per-annulus Kerr ray-tracing for a
    relativistic disc in a strong gravitational field. The disc
    luminosity is self-consistent with black hole mass and accretion rate;
    no separate ``agn_log_lbol`` parameter needed.

    .. warning::

       ``agn_log_lbol`` is ignored — RELAGN disc luminosity is derived from
       ``agn_log_mbh`` and ``agn_log_mdot`` via the grid; the parameter is
       retained for block-protocol compatibility only.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        *Unused.* Kept for block protocol; RELAGN sets its luminosity from
        M_BH and Mdot.
    agn_log_mbh : float, optional
        :math:`\log_{10}(M_{\rm BH}/M_\odot)`, range [6, 10]. Default ``8.0``.
    agn_log_mdot : float, optional
        :math:`\log_{10}(\dot M/\dot M_{\rm Edd})`, range [-1.5, 0.3].
        Default ``-1.0``.
    agn_astar : float, optional
        Dimensionless black hole spin a* (prograde only), range [0, 0.998].
        Default ``0.0``.
    agn_cos_inc : float, optional
        Cosine of inclination (1 = face-on, 0 = edge-on). Default ``0.866``
        (≈30°, matching CIGALE convention).

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Disc :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    **JIT-compatible**: yes — triweight kernel interpolation on RELAGN grid.

    **Gradient-safe**: yes — C² continuous on all axes.

    **Grid required**: ``data/relagn_disc_grid.h5`` (gitignored). Build via
    ``scripts/build_relagn_disc_grid.py`` (requires HEASOFT/XSPEC + KYCONV).

    References
    ----------
    .. [1] Dovciak, M., Karas, V., & Yaqoob, T. (2004). ApJS, 153, 205.
       KYCONV: Emission from the accretion disk of a Kerr black hole.
       https://doi.org/10.1086/421115
    .. [2] Hagen, S. & Done, C. (2023). MNRAS, 521, 251. RELAGN: A relativistic
       accretion disc model for high spin and high inclination. High-spin AGN.
       https://doi.org/10.1093/mnras/stad478
    """
    from tengri.components.agn.disc import create_relagn_disc_from_grid
    from tengri.components.agn.unified import _find_relagn_grid

    wave_aa = jnp.asarray(wavelength)
    disc_fn = create_relagn_disc_from_grid(_find_relagn_grid())
    L_nu = disc_fn(
        wave_aa,
        agn_log_mbh=agn_log_mbh,
        agn_log_mdot=agn_log_mdot,
        agn_astar=agn_astar,
        agn_cos_inc=agn_cos_inc,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
