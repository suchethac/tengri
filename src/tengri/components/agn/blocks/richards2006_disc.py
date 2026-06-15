# SPDX-License-Identifier: BSD-3-Clause
"""Richards et al. 2006 mean SDSS quasar composite SED block."""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._disc_common import _C_AA_PER_S
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.richards2006_disc import richards2006_disc


@register_agn_block(
    "disc",
    "richards2006",
    citation="Richards et al. 2006, ApJ, 166, 470",
    status="production",
    short_doc="Richards et al. 2006 mean SDSS quasar composite SED",
)
def richards2006_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    **_params,
) -> Array:
    r"""Richards+2006 mean SDSS quasar SED block.

    Empirical disc template from a composite of SDSS quasars (Richards
    et al. 2006). The template is a fixed UV-optical shape, normalized to
    the requested bolometric luminosity. Wavelength coverage 30.5 Å —
    3×10⁸ Å with zero flux outside.

    This template carries no free spectral-shape parameters. Use it when
    reproducing SDSS composites or as a fixed-shape disc alternative to
    physically motivated discs (multicolor, Kubota & Done). For variations
    in disc shape, prefer :func:`multicolor_disc_block` or
    :func:`kubota_done_disc_block`.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Disc :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    **JIT-compatible**: yes.

    **Upstream**: Empirical composite from SDSS Data Release 3
    (Richards et al. 2006).

    References
    ----------
    .. [1] Richards, G. T., et al. 2006, ApJ, 166, 470. Supermassive Black
       Holes in SDSS Quasars and the Role of Quasar Triggering. Published
       2006 May 10. https://doi.org/10.1086/506525
    """
    wave_aa = jnp.asarray(wavelength)
    L_nu = richards2006_disc(wave_aa, log_lbol=agn_log_lbol)
    return L_nu * _C_AA_PER_S / wave_aa**2
