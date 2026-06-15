# SPDX-License-Identifier: BSD-3-Clause
"""CIGALE SKIRTOR2016 ADAF-thin disc blend (disk_type=2) block."""

from __future__ import annotations

from jax import Array

from tengri.components.agn.blocks._disc_common import _cigale_disc_lambda
from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc_cigale import adaf_disk_spectrum


@register_agn_block(
    "disc",
    "adaf_lopez2024",
    citation="Lopez et al. 2024, A&A, 691, A163",
    status="production",
    short_doc="CIGALE SKIRTOR2016 ADAF-thin disc blend (disk_type=2)",
)
def cigale_adaf_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_cigale_disk_delta: float = 0.0,
    **_params,
) -> Array:
    r"""CIGALE ``skirtor2016`` ADAF↔thin-disc transitional spectrum.

    Empirical blend
    :math:`(1-\delta)\,f_{\rm ADAF}(\lambda) + \delta\,f_{\rm disc}(\lambda)`
    between an ADAF-like multi-segment power law and a δ-modulated thin-disc
    power law, mimicking the LLAGN → quasar accretion-mode transition.
    Used by CIGALE's ``skirtor2016`` module when ``disk_type = 2``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_cigale_disk_delta : float, optional
        Blend weight in ``[0, 1]`` (paper ``delta``). ``0`` -> pure ADAF;
        ``1`` -> pure thin disc. Note: for this block the parameter is a
        blend weight, **not** a slope modulator. Default ``0.0``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        :math:`L_\lambda` [erg/s/Å].

    Notes
    -----
    **JIT-compatible**: yes.

    **Upstream**: CIGALE ``pcigale.sed_modules.skirtor2016.adaf_disk``
    (Lopez et al. 2024 [Lop24]_, Boquien et al. 2019 [B19]_).

    References
    ----------
    .. [Lop24] Lopez, I. E. et al. 2024, A&A, 691, A163. Modelling the
       X-ray emission of AGN in CIGALE and application to eROSITA.
       arXiv:2407.16182. https://doi.org/10.1051/0004-6361/202449801
    .. [B19] Boquien, M. et al. 2019, A&A, 622, A103. CIGALE: a Python
       Code Investigating GALaxy Emission. arXiv:1811.03094.
       https://doi.org/10.1051/0004-6361/201834156
    """
    return _cigale_disc_lambda(
        wavelength,
        agn_log_lbol,
        adaf_disk_spectrum,
        delta=agn_cigale_disk_delta,
    )
