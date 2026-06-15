# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE SKIRTOR2016 Schartmann 2005 disc (disk_type=1) block."""

from __future__ import annotations

from jax import Array

from tengri.components.agn.blocks._disc_common import _cigale_disc_lambda
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc_cigale import schartmann2005_disk_spectrum


@register_agn_block(
    "disc",
    "schartmann2005",
    citation="Schartmann et al. 2005, A&A, 437, 861",
    status="production",
    short_doc="CIGALE SKIRTOR2016 Schartmann 2005 disc (disk_type=1)",
)
def cigale_schartmann_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_cigale_disk_delta: float = 0.0,
    **_params,
) -> Array:
    r"""CIGALE ``skirtor2016`` Schartmann (2005) disc spectrum.

    Piecewise power law with breakpoints at λ = 8, 50, 125, 10⁴, 10⁶ nm
    and indices :math:`\alpha = (1.0, -0.2, -1.5 + \delta, -4.0)`. Used
    by CIGALE's ``skirtor2016`` module when ``disk_type = 1`` (the
    CIGALE default).

    The Schartmann shape has a shallower near-IR slope and a smoother
    1200-Å bend than the SKIRTOR analytic disc.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_cigale_disk_delta : float, optional
        Slope modulator (paper ``delta``); 100-10000 nm index becomes
        :math:`-1.5 + \delta`. Default ``0.0``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    **JIT-compatible**: yes.

    **Upstream**: CIGALE
    ``pcigale.sed_modules.skirtor2016.schartmann2005_disk``
    (Boquien et al. 2019 [B19]_).

    References
    ----------
    .. [Sch05] Schartmann, M., Meisenheimer, K., Camenzind, M., Wolf, S.,
       & Henning, T. 2005, A&A, 437, 861. Towards a physical model of
       dust tori in active galactic nuclei. Radiative transfer
       calculations for a hydrostatic torus model.
       https://doi.org/10.1051/0004-6361:20042363
    .. [B19] Boquien, M. et al. 2019, A&A, 622, A103. CIGALE: a Python
       Code Investigating GALaxy Emission. arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    return _cigale_disc_lambda(
        wavelength,
        agn_log_lbol,
        schartmann2005_disk_spectrum,
        delta=agn_cigale_disk_delta,
    )
