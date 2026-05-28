# SPDX-License-Identifier: BSD-3-Clause
"""Non-GRAHSP disc-stage blocks: multicolor / Kubota-Done / ADAF.

Wraps three existing tengri disc functions in the
:mod:`tengri.components.agn.blocks._protocol` block-protocol signature so
they can be selected via ``agn_disc_block=...`` in a composable AGN recipe.

Each adapter performs the standard :math:`L_\\nu \\to L_\\lambda` conversion
:math:`L_\\lambda = L_\\nu \\, c / \\lambda^2` at the block boundary, where
:math:`c = 2.99792458 \\times 10^{18}` Å·Hz.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array

from tengri.components.agn.blocks._protocol import register_agn_block
from tengri.components.agn.disc import (
    adaf_disc,
    kubota_done_disc,
    multicolor_disc,
)
from tengri.components.agn.disc_cigale import (
    adaf_disk_spectrum,
    schartmann2005_disk_spectrum,
    skirtor_disk_spectrum,
)
from tengri.utils.physics_constants import L_SUN

__all__: list[str] = []  # registrations only

#: Speed of light in Å × Hz, for L_ν → L_λ conversion.
_C_AA_PER_S: float = 2.99792458e18

#: Solar luminosity [erg/s] — IAU 2015 nominal (already in erg/s, see
#: :data:`tengri.utils.physics_constants.L_SUN`).
_L_SUN_ERG: float = L_SUN


@register_agn_block("disc", "multicolor")
def multicolor_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.86602540378443864,
    **_params,
) -> Array:
    r"""Shakura-Sunyaev multi-color thin-disc block.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    agn_log_mbh : float, optional
        :math:`\log_{10}(M_{\rm BH}/M_\odot)`. Default ``8.0``.
    agn_log_ledd : float, optional
        :math:`\log_{10}(\lambda_{\rm Edd})`. Default ``-1.0``.
    agn_a_spin : float, optional
        BH spin parameter. Default ``0.0``.
    agn_cos_inc : float, optional
        Cosine of viewing inclination. Default ``0.5``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        Disc :math:`L_\lambda` [erg/s/Å].

    References
    ----------
    .. [1] Shakura, N. I. & Sunyaev, R. A. 1973, A&A, 24, 337.
    """
    wave_aa = jnp.asarray(wavelength)
    L_nu = multicolor_disc(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("disc", "kubota_done")
def kubota_done_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.86602540378443864,
    agn_f_hard: float = 0.02,
    agn_gamma_warm: float = 2.5,
    agn_kt_warm: float = 0.2,
    agn_gamma_hard: float = 1.8,
    agn_kt_hot: float = 100.0,
    agn_r_warm_ratio: float = 2.0,
    **_params,
) -> Array:
    r"""Kubota & Done (2018) three-zone disc + corona block.

    Three radial zones: cool outer SS thin disc, warm Comptonising region,
    hot inner corona. Designed for intermediate-to-high accretion rates.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol, agn_log_mbh, agn_log_ledd, agn_a_spin, agn_cos_inc
        Standard disc parameters; see :func:`multicolor_disc_block`.
    agn_f_hard : float, optional
        Coronal luminosity fraction. Default ``0.02``.
    agn_gamma_warm, agn_kt_warm : float, optional
        Warm Comptonisation photon index and electron temperature [keV].
    agn_gamma_hard, agn_kt_hot : float, optional
        Hot Comptonisation photon index and electron temperature [keV].
    agn_r_warm_ratio : float, optional
        :math:`R_{\rm warm}/R_{\rm hot}`. Default ``2.0``.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        :math:`L_\lambda` [erg/s/Å].

    References
    ----------
    .. [1] Kubota, A. & Done, C. 2018, MNRAS, 480, 1247,
       https://doi.org/10.1093/mnras/sty1890.
    """
    wave_aa = jnp.asarray(wavelength)
    L_nu = kubota_done_disc(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
        agn_f_hard=agn_f_hard,
        agn_gamma_warm=agn_gamma_warm,
        agn_kt_warm=agn_kt_warm,
        agn_gamma_hard=agn_gamma_hard,
        agn_kt_hot=agn_kt_hot,
        agn_r_warm_ratio=agn_r_warm_ratio,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


@register_agn_block("disc", "adaf")
def adaf_disc_block(
    wavelength: Array,
    agn_log_lbol: float,
    *,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -3.0,
    agn_r_tr: float = 100.0,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.01,
    agn_cos_inc: float = 0.86602540378443864,
    **_params,
) -> Array:
    r"""ADAF + truncated disc for low-luminosity AGN.

    Inner accretion flow is advection-dominated (radiatively inefficient);
    outer flow is a SS thin disc truncated at :math:`r_{\rm tr}`.

    .. warning::

       The ADAF inner flow does **not** produce a meaningful 5100 Å
       continuum; pairing this disc with GRAHSP-style downstream blocks
       (which normalise to :math:`\lambda L_\lambda(5100\,\mathrm{\AA})`)
       will trigger :class:`RecipeWarning`. Use only when a UV/optical
       contribution is genuinely absent from the source.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
    agn_log_lbol, agn_log_mbh, agn_log_ledd, agn_cos_inc
        Standard disc parameters.
    agn_r_tr : float, optional
        Truncation radius :math:`r_{\rm tr}` [:math:`R_g`]. Default ``100``.
    agn_adaf_beta : float, optional
        Magnetic-to-gas pressure ratio. Default ``0.5``.
    agn_adaf_delta : float, optional
        Electron heating fraction. Default ``0.01``.

    References
    ----------
    .. [1] Mahadevan, R. 1997, ApJ, 477, 585. ADAF emission spectra.
    .. [2] Lopez Navas, E. et al. 2024 (low-luminosity AGN application).
    """
    wave_aa = jnp.asarray(wavelength)
    L_nu = adaf_disc(
        wave_aa,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_r_tr=agn_r_tr,
        agn_adaf_beta=agn_adaf_beta,
        agn_adaf_delta=agn_adaf_delta,
        agn_cos_inc=agn_cos_inc,
    )
    return L_nu * _C_AA_PER_S / wave_aa**2


# ──────────────────────────────────────────────────────────────────────
# CIGALE skirtor2016 piecewise-power-law disc blocks
# ──────────────────────────────────────────────────────────────────────
#
# Three empirical disc spectra ported from CIGALE's ``skirtor2016`` module
# (Boquien et al. 2019 [B19]_). Each is a piecewise power law normalised
# to unit area; the block adapter scales by :math:`L_{\rm bol}` to produce
# :math:`L_\lambda` [erg/s/Å]. These are the *bit-for-bit* CIGALE disc
# shapes — use them when reproducing CIGALE fits. For differentiable
# physical discs use ``multicolor``, ``kubota_done`` or ``adaf`` instead.
#
# **Wavelength unit bridge**: ``disc_cigale.*_spectrum`` functions return
# a spectrum normalised in CIGALE's native **nanometre** grid; we evaluate
# them on ``wavelength_aa / 10`` and divide the returned per-nm spectrum
# by 10 to convert to per-Å. (For a unit-normalised distribution,
# rescaling the axis by k multiplies values by 1/k.)
#
# .. [B19] Boquien, M. et al. 2019, A&A, 622, A103. CIGALE: a Python
#    Code Investigating GALaxy Emission. arXiv:1811.03094.
#    https://doi.org/10.1051/0004-6361/201834156


def _cigale_disc_lambda(
    wavelength_aa: Array,
    agn_log_lbol: float,
    spectrum_per_nm_fn,
    delta: float,
) -> Array:
    r"""Common L_λ scaffold for CIGALE piecewise-power-law disc blocks.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Rest-frame wavelength [Å].
    agn_log_lbol : float
        :math:`\log_{10}(L_{\rm bol}/L_\odot)`.
    spectrum_per_nm_fn : callable
        One of :func:`skirtor_disk_spectrum`,
        :func:`schartmann2005_disk_spectrum`, or
        :func:`adaf_disk_spectrum`. Takes ``(wave_nm, delta)`` and returns
        a dimensionless spectrum normalised so its integral over the
        nm axis equals one.
    delta : float
        CIGALE ``delta`` slope/blend modulator.

    Returns
    -------
    L_lambda : ndarray, shape (n_wave,)
        :math:`L_\lambda` [erg/s/Å].
    """
    wave_aa = jnp.asarray(wavelength_aa)
    wave_nm = wave_aa / 10.0
    # Unit-normalised spectrum on the nm grid (integral over nm = 1).
    s_per_nm = spectrum_per_nm_fn(wave_nm, delta=delta)
    # Convert to a unit-normalised density on the Å grid (÷10).
    s_per_aa = s_per_nm / 10.0
    L_bol_erg = (10.0**agn_log_lbol) * _L_SUN_ERG
    return s_per_aa * L_bol_erg


@register_agn_block("disc", "skirtor")
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
       transfer modelling of the dusty torus around AGN: the influence
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


@register_agn_block("disc", "schartmann2005")
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


@register_agn_block("disc", "adaf_lopez2024")
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
