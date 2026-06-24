# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE SKIRTOR2016 empirical disc (disk_type=0) block."""

from __future__ import annotations

from jax import Array

from tengri.components.agn.blocks._disc_common import _cigale_disc_lambda
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc_cigale import skirtor_disk_spectrum


@register_agn_block(
    "disc",
    "skirtor",
    citation="Stalevski et al. 2016, MNRAS, 458, 2288",
    status="production",
    short_doc="CIGALE SKIRTOR2016 empirical disc (disk_type=0)",
)
def cigale_skirtor_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_cigale_disk_delta: float = 0.0,
    **_params,
) -> Array:
    r"""CIGALE ``skirtor2016`` empirical disc spectrum.

    Piecewise power law with breakpoints at λ = 8, 10, 100, 5000, 10⁶ nm
    and indices :math:`\alpha = (0.2, -1.0, -1.5 + \delta, -4.0)`. This
    is the disc shape bundled in CIGALE's ``skirtor2016`` module when
    ``disk_type = 0``; pair it with the ``torus="skirtor"`` block for a
    bit-for-bit reproduction of CIGALE's SKIRTOR2016 AGN SED (closes the
    UV-optical disc disagreement documented in
    ``reproduction/cigale/01_cigale.py`` §9).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`. Sets the integrated disc
        luminosity.
    agn_cigale_disk_delta : float, optional
        Slope modulator (paper ``delta``). The 100-5000 nm power-law
        index becomes :math:`-1.5 + \delta`. Default ``0.0``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    **JIT-compatible**: yes.

    **Upstream**: CIGALE ``pcigale.sed_modules.skirtor2016.skirtor_disk``
    (Boquien et al. 2019 [B19]_).

    References
    ----------
    .. [S12] Stalevski, M. et al. 2012, MNRAS, 420, 2756. 3D radiative
       transfer modeling of the dusty torus around AGN: the influence
       of clumping. arXiv:1109.1286.
       https://doi.org/10.1111/j.1365-2966.2011.19775.x
    .. [S16] Stalevski, M. et al. 2016, MNRAS, 458, 2288. The dust
       covering factor in active galactic nuclei. arXiv:1602.06954.
       https://doi.org/10.1093/mnras/stw444
    .. [B19] Boquien, M. et al. 2019, A&A, 622, A103. CIGALE: a Python
       Code Investigating GALaxy Emission. arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    return _cigale_disc_lambda(
        wavelength,
        agn_log_lbol,
        skirtor_disk_spectrum,
        delta=agn_cigale_disk_delta,
    )
