# SPDX-License-Identifier: BSD-3-Clause
"""Fritz et al. 2006 smooth-dust torus block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.blocks._torus_common import _C_AA_PER_S
from tengri.components.agn.fritz import fritz_sed


@register_agn_block(
    "torus",
    "fritz",
    citation="Fritz et al. 2006, A&A, 470, 221",
    status="production",
    short_doc="Fritz et al. 2006 smooth-dust torus",
)
def fritz_torus_block(
    wavelength: Array,
    agn_log_lbol: float,
    l5100_disc: Array,
    *,
    agn_fritz_r_ratio: float = 60.0,
    agn_fritz_tau: float = 1.0,
    agn_fritz_beta: float = -0.5,
    agn_fritz_gamma: float = 4.0,
    agn_fritz_oa: float = 60.0,
    agn_fritz_psy: float = 0.001,
    agn_torus_frac: float = 0.5,
    **_params,
) -> Array:
    r"""Fritz+ 2006 smooth-dust torus block.

    Six-dimensional template grid with triweight interpolation on
    ``(r_ratio, tau, beta, gamma, opening_angle, psy)``. The torus
    covering factor scales the template by ``agn_torus_frac × L_bol``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol : float
    l5100_disc : array
        Ignored.
    agn_fritz_r_ratio : float, optional
        Dust torus radius ratio (r_max / r_min) [dimensionless].
        Default ``60.0``. Allowed: 10, 30, 60, 100, 150.
    agn_fritz_tau : float, optional
        Optical depth at 9.7 µm [dimensionless].
        Default ``1.0``. Allowed: 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 6.0, 10.0.
    agn_fritz_beta : float, optional
        Radial dust density power-law index [dimensionless].
        Default ``-0.5``. Allowed: -1.0, -0.75, -0.5, -0.25, 0.0.
    agn_fritz_gamma : float, optional
        Polar dust density gradient [dimensionless].
        Default ``4.0``. Allowed: 0, 2, 4, 6.
    agn_fritz_oa : float, optional
        Dust torus half-opening angle [degrees], as keyed in CIGALE's
        ``SimpleDatabase`` (the user-facing "full opening angle" 60/100/140 is
        mapped to this half-angle via ``(180 - oa) / 2`` in CIGALE).
        Default ``60.0``. Allowed: 20, 40, 60.
    agn_fritz_psy : float, optional
        Viewing angle from torus axis [degrees].
        Default ``0.001`` (type-2 edge-on).
        Allowed: 0.001, 10.1, 20.1, 30.1, 40.1, 50.1, 60.1, 70.1, 80.1, 89.99.
        Values: 0° = type-2 AGN (edge-on), 90° = type-1 AGN (face-on).
    agn_torus_frac : float, optional
        Covering factor [0, 1]. Default ``0.5``.

    References
    ----------
    .. [1] Fritz, O. et al. 2006, A&A, 470, 221. arXiv:0606147.
    .. [2] Boquien, M. et al. 2019, A&A, 622, A103. CIGALE. arXiv:1811.03094.
    """
    del l5100_disc
    wave_aa = jnp.asarray(wavelength)
    L_nu = fritz_sed(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
        agn_fritz_r_ratio=agn_fritz_r_ratio,
        agn_fritz_tau=agn_fritz_tau,
        agn_fritz_beta=agn_fritz_beta,
        agn_fritz_gamma=agn_fritz_gamma,
        agn_fritz_oa=agn_fritz_oa,
        agn_fritz_psy=agn_fritz_psy,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2
