# SPDX-License-Identifier: BSD-3-Clause
"""X-ray SED models: binaries, AGN corona, hot gas.

Predicts X-ray emission (0.1–10 keV, λ < 124 Å) from three physical components:

1. **X-ray binaries** (HMXB + LMXB): power-law + cutoff, SFR- and mass-dependent
2. **AGN corona**: power-law continuum with exponential high-energy cutoff,
   optionally tied to disc UV luminosity (self-consistent disc-corona coupling)
3. **Hot gas**: optional thermal bremsstrahlung from diffuse ISM/CGM

All functions are pure JAX, JIT-compatible, fully differentiable.

**Self-consistent disc-corona**: The function xray_agn_corona_from_disc computes
the X-ray photon index and normalisation from disc UV luminosity using empirical
α_ox–L_2500 correlations (Just et al. 2007; Yang et al. 2022). This enforces
physical consistency between UV and X-ray SED components during inference.

**IRX-based X-ray (Lopez+2024)**: The function xray_agn_corona_lopez24 uses the
α_IRX parameter (12μm-to-X-ray ratio) instead of α_ox, which is more robust
for obscured and low-luminosity AGN where UV is unreliable.

**Design basis**: Adapted from CIGALE modules (Yang+2020, Lopez+2024) with
full JAX reimplementation for differentiability and gradient-based inference.
"""

import jax.numpy as jnp

from tengri.utils.physics_constants import (
    C_AA as _C_AA,
    H_PLANCK as _H_PLANCK,
    KEV_TO_ERG as _KEV_TO_ERG,
    KEV_TO_HZ as _KEV_TO_HZ,
)


def xray_xrb(
    wavelength: jnp.ndarray,
    sfr: float,
    stellar_mass: float,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    E_cut: float = 100.0,
    log_L_hmxb_offset: float = 0.0,
    log_L_lmxb_offset: float = 0.0,
    metallicity_z: float = 0.02,
    stellar_age_gyr: float = 5.0,
) -> jnp.ndarray:
    r"""Predict X-ray SED from accretion-powered binaries (Lehmer+2016 / Yang+22).

    Computes the combined X-ray emission from high-mass (HMXB) and low-mass
    (LMXB) X-ray binary populations using the Lehmer et al. (2016, 2019)
    metallicity- and age-dependent scaling relations adopted by Yang
    et al. (2022) in CIGALE ``yang20``. Each population is modelled as a
    power-law with exponential cutoff.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame). [Å]
    sfr : float
        Star formation rate. [Msun/yr]
    stellar_mass : float
        Stellar mass. [Msun]
    gamma_hmxb : float, optional
        HMXB photon index (Γ, where F_ν ∝ ν^{−Γ}). Default: 2.0.
    gamma_lmxb : float, optional
        LMXB photon index. Default: 1.6.
    E_cut : float, optional
        Exponential cutoff energy for both populations. Default: 100 keV. [keV]
    log_L_hmxb_offset : float, optional
        Departure from mean SFR relation (δ_HMXB in Yang+22). Default 0.0. [dex]
    log_L_lmxb_offset : float, optional
        Departure from mean stellar-mass relation (δ_LMXB). Default 0.0. [dex]
    metallicity_z : float, optional
        Stellar metallicity (mass fraction, ``Z = 0.02`` ≈ solar).
        Drives the HMXB quartic; sub-solar metallicity makes HMXBs more
        X-ray luminous (more massive black holes). Default 0.02. [dimensionless]
    stellar_age_gyr : float, optional
        Mass-weighted stellar age. Drives the LMXB age quartic in
        ``log10(age/Gyr)``. Default 5.0. [Gyr]

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density of X-ray binary populations.
        [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **HMXB luminosity scaling** — Lehmer et al. (2019, ApJ 883, 109) Eq. 8
    via the polynomial fit of Fragos et al. (2013) adopted by Yang et al.
    (2022, ApJ 927, 192) in CIGALE ``yang20`` lines 207–214. HMXBs trace
    the instantaneous SFR with a metallicity-dependent normalisation
    (low-Z stars form more massive black holes, raising the integrated
    luminosity per unit SFR):

    .. math::

        \log_{10}\!\left[\frac{L_X^{\mathrm{HMXB}}(2\text{–}10)}{\mathrm{SFR}}\right]
        = 33.28 - 62.12\,Z + 569.44\,Z^2 - 1833.80\,Z^3 + 1968.33\,Z^4
        \;\;[\mathrm{W}/(\,M_\odot\,\mathrm{yr}^{-1})]

    converted to erg/s by multiplying by 10⁷.

    **LMXB luminosity scaling** — Lehmer et al. (2014, ApJ 789, 52) age-
    dependent quartic in ``logT ≡ log10(age/Gyr)`` adopted by Yang+22
    (``yang20`` lines 217–224). LMXBs are old (> 1 Gyr) systems so the
    integrated luminosity per unit stellar mass declines with age as
    the progenitor binaries deplete:

    .. math::

        \log_{10}\!\left[\frac{L_X^{\mathrm{LMXB}}(2\text{–}10)}{M_\star/10^{10}\,M_\odot}\right]
        = 33.276 - 1.503\,\log T - 0.423\,\log T^2
                + 0.425\,\log T^3 + 0.136\,\log T^4
        \;\;[\mathrm{W}]

    converted to erg/s by multiplying by 10⁷.

    **Spectral shape**: power-law with exponential cutoff (photoelectric
    absorption, intrinsic torque limits):

    .. math::

        F_\nu \propto \nu^{-\Gamma}\, \exp(-h\nu / E_{\mathrm{cut}})

    Typical cutoffs: E_cut ≈ 100 keV. The :math:`\log L_{\mathrm{*}}` offsets
    capture per-galaxy scatter (Yang+22 δ_HMXB / δ_LMXB).

    **Why this changed from Grimm+03 / Gilfanov 2004**: the earlier
    ``2.6e39·SFR`` (HMXB) and ``8.3e28·M*`` (LMXB) constants ignore
    metallicity and stellar age, which Yang+22 found to drive ≈ 0.5 dex
    of systematic offset at z > 3 sub-solar regimes. At Z = Z_sun and
    age = 10 Gyr the new quartics reproduce the old constants to ~ 10 %.

    References
    ----------
    .. [1] B. D. Lehmer et al., "The metallicity dependence of the
       high-mass X-ray binary luminosity function," ApJ, 883, 109 (2019).
       https://doi.org/10.3847/1538-4357/ab3104
    .. [2] B. D. Lehmer et al., "Star-formation-driven X-ray emission in
       redshift z = 0–8 galaxies," ApJ, 789, 52 (2014).
       https://doi.org/10.1088/0004-637X/789/1/52
    .. [3] T. Fragos et al., "X-ray binary evolution across cosmic time,"
       ApJ, 776, L31 (2013). https://doi.org/10.1088/2041-8205/776/2/L31
    .. [4] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022),
       module ``yang20``. https://doi.org/10.3847/1538-4357/ac4971
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / _KEV_TO_ERG  # convert to keV

    # Lehmer+2019 / Fragos+2013 metallicity-quartic HMXB normalisation
    # (Yang+22 Eq. used in CIGALE yang20.py:207–214). 10^7 converts W → erg/s.
    Z = metallicity_z
    log_L_hmxb_per_sfr_W = 33.28 - 62.12 * Z + 569.44 * Z**2 - 1833.80 * Z**3 + 1968.33 * Z**4
    L_hmxb_ref = (
        sfr * 10.0 ** (log_L_hmxb_per_sfr_W + log_L_hmxb_offset) * 1e7  # erg/s
    )

    # Lehmer+2014 age-quartic LMXB normalisation (yang20.py:217–224).
    # Floor age at 0.1 Myr to keep log defined for stellar tracers that
    # may pass zero on the first integration step.
    logT = jnp.log10(jnp.maximum(stellar_age_gyr, 1e-4))
    log_L_lmxb_per_1e10_W = (
        33.276 - 1.503 * logT - 0.423 * logT**2 + 0.425 * logT**3 + 0.136 * logT**4
    )
    L_lmxb_ref = (
        (stellar_mass / 1e10) * 10.0 ** (log_L_lmxb_per_1e10_W + log_L_lmxb_offset) * 1e7  # erg/s
    )

    # Power-law with exponential cutoff: L_nu ∝ (E/E_ref)^{-Γ+1} * exp(-E/E_cut)
    # Normalise by integrating the spectral shape over the 2-10 keV reference band
    # (not by a single-point bandwidth, which gives ~2-3x error in absolute luminosity).
    E_ref = 5.0  # keV (reference for spectral shape evaluation)

    spec_hmxb = (E_keV / E_ref) ** (-gamma_hmxb + 1) * jnp.exp(-E_keV / E_cut)
    spec_lmxb = (E_keV / E_ref) ** (-gamma_lmxb + 1) * jnp.exp(-E_keV / E_cut)

    # Compute band integral of each spectral shape over 2-10 keV on a fine grid.
    # ∫L_nu dnu = L_ref  → L_nu = L_ref * spec / ∫_band spec dnu
    E_fine = jnp.linspace(2.0, 10.0, 200)  # keV
    nu_fine = E_fine * _KEV_TO_HZ
    spec_hmxb_fine = (E_fine / E_ref) ** (-gamma_hmxb + 1) * jnp.exp(-E_fine / E_cut)
    spec_lmxb_fine = (E_fine / E_ref) ** (-gamma_lmxb + 1) * jnp.exp(-E_fine / E_cut)
    band_int_hmxb = jnp.maximum(jnp.trapezoid(spec_hmxb_fine, nu_fine), 1e-60)
    band_int_lmxb = jnp.maximum(jnp.trapezoid(spec_lmxb_fine, nu_fine), 1e-60)

    L_nu_hmxb = L_hmxb_ref / band_int_hmxb * spec_hmxb
    L_nu_lmxb = L_lmxb_ref / band_int_lmxb * spec_lmxb

    # X-ray only (E > 0.1 keV = lambda < 124 A)
    xray_mask = wavelength < 124.0
    return jnp.where(xray_mask, L_nu_hmxb + L_nu_lmxb, 0.0)


def alpha_ox_from_l2500(l_2500_erg_hz: float) -> float:
    r"""Compute alpha_ox from monochromatic 2500 A luminosity (Just+2007).

    Parameters
    ----------
    l_2500_erg_hz : float
        Monochromatic luminosity density at rest-frame 2500 A. [erg/s/Hz]

    Returns
    -------
    float
        Optical-to-X-ray spectral index alpha_ox, the slope between 2500 A
        and 2 keV monochromatic fluxes. Typical AGN range: -2.0 to -1.0.
        [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    **Empirical correlation** (Just et al. 2007 [1]_, Eq. 3):
    derived from optically-bright AGN; valid for
    :math:`28 \lesssim \log_{10}(L_{2500}/[\mathrm{erg\,s^{-1}\,Hz^{-1}}]) \lesssim 33`.

    .. math::

        \alpha_{\mathrm{ox}} = -0.137 \, \log_{10}\!\left(
            L_{2500}\,[\mathrm{erg\,s^{-1}\,Hz^{-1}}]
        \right) + 2.638

    More luminous AGN are X-ray weaker (steeper, more negative
    :math:`\alpha_{\mathrm{ox}}`).

    References
    ----------
    .. [1] Just, D. W. et al., 2007, ApJ, 665, 1004, Eq. 3.
    """
    return -0.137 * jnp.log10(l_2500_erg_hz) + 2.638


def xray_hotgas(
    wavelength: jnp.ndarray,
    sfr: float,
) -> jnp.ndarray:
    r"""Diffuse hot-gas X-ray emission from star-forming galaxies.

    Soft (≲ 2 keV) thermal-bremsstrahlung-like template scaled by SFR,
    following CIGALE ``yang20`` lines 110–116 and 203. The spectral
    shape is the same shape used by Yang+22:

    .. math::

        L_\nu(\lambda) \propto \lambda^{-2}\, \exp(-\lambda_{1\,\mathrm{keV}}/\lambda)

    normalised so that the 0.5–2 keV band integral equals
    :math:`8.3 \times 10^{38}\,\mathrm{SFR}\;[\mathrm{erg/s}]` (the
    Mineo et al. 2012 hot-gas–SFR scaling). This is the SFR-driven
    diffuse component that sits below the XRB power-law in the soft
    band of star-forming galaxies and dominates < 1 keV for SFR-poor
    LMXB hosts.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame). [Å]
    sfr : float
        Star formation rate. [Msun/yr]

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density of the hot-gas component.
        [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — pure ``jnp`` primitives.

    **Normalisation** — Yang+22 CIGALE ``yang20.py:204``: the 0.5–2 keV
    integral is :math:`8.3 \times 10^{31}\,\mathrm{SFR}\;[\mathrm{W}]
    = 8.3 \times 10^{38}\,\mathrm{SFR}\;[\mathrm{erg/s}]`. The factor
    is derived from the Mineo et al. (2012) hot-gas X-ray–SFR scaling
    on local star-forming galaxies.

    References
    ----------
    .. [1] S. Mineo, M. Gilfanov and R. Sunyaev, "X-ray emission from
       star-forming galaxies — II. Hot interstellar medium," MNRAS,
       426, 1870 (2012). https://doi.org/10.1111/j.1365-2966.2012.21831.x
    .. [2] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022),
       ``yang20`` lines 110–116, 203. https://doi.org/10.3847/1538-4357/ac4971
    """
    # Wavelength of 1 keV in Å (= h c / (1 keV)). Using the imported
    # physics constants keeps the conversion consistent with the rest
    # of this module.
    lam_1kev = _C_AA * _H_PLANCK / _KEV_TO_ERG  # Å
    lam_0p5kev = lam_1kev * 2.0  # 0.5 keV → λ = 24.8 Å
    lam_2kev = lam_1kev / 2.0  # 2 keV   → λ = 6.2 Å

    # Yang+22 template shape, normalised on a fine band grid.
    spec = wavelength ** (-2.0) * jnp.exp(-lam_1kev / wavelength)

    # Band integral over 0.5–2 keV in frequency. Trapezoid needs an
    # ascending grid, so build ν directly (= ascending in energy).
    nu_fine = jnp.linspace(_C_AA / lam_0p5kev, _C_AA / lam_2kev, 200)
    wave_fine = _C_AA / nu_fine
    spec_fine = wave_fine ** (-2.0) * jnp.exp(-lam_1kev / wave_fine)
    band_int = jnp.maximum(jnp.trapezoid(spec_fine, nu_fine), 1e-60)

    # 8.3e38 erg/s per Msun/yr (Mineo+2012 hot-gas scaling, Yang+22).
    L_0p5_2 = 8.3e38 * sfr
    L_nu = L_0p5_2 / band_int * spec

    xray_mask = wavelength < 124.0
    return jnp.where(xray_mask, L_nu, 0.0)


def xray_anisotropy(
    l_x: jnp.ndarray,
    cos_inc: float,
    a1: float = 0.5,
    a2: float = 0.0,
) -> jnp.ndarray:
    r"""Apply viewing-angle anisotropy to X-ray luminosity (Yang+2022).

    Parameters
    ----------
    l_x : array_like, shape (n_wave,)
        Isotropic (face-on) X-ray luminosity spectrum. [erg/s/Hz]
    cos_inc : float
        Cosine of inclination angle (1 = face-on, 0 = edge-on).
        [dimensionless]
    a1 : float, optional
        Linear anisotropy coefficient. Default 0.5. [dimensionless]
    a2 : float, optional
        Quadratic anisotropy coefficient. Default 0.0. [dimensionless]

    Returns
    -------
    ndarray, shape (n_wave,)
        Anisotropy-corrected L_X. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    **Empirical correction** (Yang et al. 2022, ApJ 927, 192, lines
    231–234 of CIGALE ``yang20.py``): polynomial in
    :math:`\mu \equiv \cos\theta`, normalised so that the *reference*
    viewing angle :math:`\theta_{\rm ref} = 30°` returns the input
    ``l_x`` unchanged. The reason is that L_2500 (which sets the X-ray
    normalisation through α_ox) is itself a θ = 30° quantity in the
    SKIRTOR disc convention, so the anisotropy factor must be 1 at
    :math:`\mu_{30} = \cos 30° = 0.866`:

    .. math::

        f(\mu) = \frac{a_1\,\mu + a_2\,\mu^2 + (1 - a_1 - a_2)}
                      {1 - 0.13397\,a_1 - 0.25\,a_2},
        \qquad
        L_X^{\rm obs} = f(\mu)\, L_X^{\rm iso}

    The denominator :math:`1 - 0.13397 a_1 - 0.25 a_2 = f_{\rm num}(\mu_{30})`
    is the numerator evaluated at :math:`\mu_{30}`. The default
    :math:`a_1 = 0.5,\, a_2 = 0` is the X-CIGALE intermediate-obscuration
    solution. For the defaults, ``f(1) ≈ 1.072`` and ``f(0) ≈ 0.535`` —
    face-on is ~ 7 % brighter than the L_2500-referenced normalisation.

    Pre-2026-05 versions omitted the denominator, biasing observed
    luminosities low by ~ 7 % at face-on; the fix is consequential for
    α_ox–L_2500 calibrations even though the dispersion in the data is
    larger.

    References
    ----------
    .. [1] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    """
    numerator = a1 * cos_inc + a2 * cos_inc**2 + (1.0 - a1 - a2)
    denominator = 1.0 - 0.13397 * a1 - 0.25 * a2
    return l_x * numerator / denominator


def xray_agn_corona_from_disc(
    wavelength: jnp.ndarray,
    l_2500_erg_hz: float,
    cos_inc: float = 1.0,
    delta_alpha_ox: float = 0.0,
    gamma: float = 1.8,
    E_cut: float = 300.0,
    apply_anisotropy: bool = True,
    a1: float = 0.5,
    a2: float = 0.0,
) -> jnp.ndarray:
    """Self-consistent AGN corona emission derived from disc UV luminosity.

    Computes alpha_ox from L_2500 via the Just+2007 relation, derives
    L_2keV, builds the X-ray power-law spectrum, and optionally applies
    viewing-angle anisotropy (Yang+2022).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength [Angstrom].
    l_2500_erg_hz : float
        Monochromatic luminosity density at 2500 A [erg/s/Hz].
    cos_inc : float
        Cosine of inclination (1 = face-on). Default 1.0.
    delta_alpha_ox : float
        Additive offset to the Just+2007 alpha_ox. Default 0.0.
    gamma : float
        Photon index. Default 1.8. Range: 1.4-2.4.
    E_cut : float
        Exponential cutoff energy [keV]. Default 300.
    apply_anisotropy : bool
        Whether to apply Yang+2022 viewing-angle correction.
    a1 : float
        Linear anisotropy coefficient. Default 0.5.
    a2 : float
        Quadratic anisotropy coefficient. Default 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.
    """
    # alpha_ox from disc UV luminosity
    alpha_ox = alpha_ox_from_l2500(l_2500_erg_hz) + delta_alpha_ox

    # Derive L_2keV from alpha_ox definition:
    #   alpha_ox = 0.384 * log10(L_2keV / L_2500)
    #   => L_2keV = L_2500 * 10^(alpha_ox / 0.384)
    l_2kev_erg_hz = l_2500_erg_hz * 10.0 ** (alpha_ox / 0.384)

    # Build power-law spectrum with exponential cutoff
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / 1.6022e-9  # convert to keV
    E_ref = 2.0  # keV
    spec = (E_keV / E_ref) ** (-gamma + 1) * jnp.exp(-E_keV / E_cut)

    # Normalise at 2 keV. ``l_2kev_erg_hz`` is already L_nu(2 keV) in erg/s/Hz
    # (alpha_ox is defined on monochromatic L_nu values, Tananbaum+1979), so
    # multiplying by the dimensionless ``spec`` (=1 at E=E_ref) gives L_nu(E).
    l_nu = l_2kev_erg_hz * spec

    # X-ray mask (E > 0.1 keV => lambda < 124 A)
    l_nu = jnp.where(wavelength < 124.0, l_nu, 0.0)

    # Optional anisotropy correction
    if apply_anisotropy:
        l_nu = xray_anisotropy(l_nu, cos_inc, a1=a1, a2=a2)

    return l_nu


def xray_agn_corona(
    wavelength: jnp.ndarray,
    L_agn_bol: float,
    gamma: float = 1.8,
    E_cut: float = 300.0,
    alpha_ox: float = -1.4,
) -> jnp.ndarray:
    """X-ray emission from AGN corona.

    Power law with photon index Gamma and exponential cutoff,
    normalized via the alpha_ox relation:
      alpha_ox = 0.384 * log(L_2keV / L_2500A)

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength [Angstrom].
    L_agn_bol : float
        AGN bolometric luminosity [erg/s].
    gamma : float
        Photon index. Default 1.8. Range: 1.4-2.4.
    E_cut : float
        Cutoff energy [keV]. Default 300.
    alpha_ox : float
        UV-to-X-ray slope. Default -1.4. Range: -2.0 to -1.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / (1.6022e-9)

    # Monochromatic luminosity density at 2500 A in erg/s/Hz.
    # L_bol = BC_2500 * nu_2500 * L_nu(2500) => L_nu = L_bol / (BC * nu)
    # BC_2500 ~ 5.15 (Hopkins+2007), nu_2500 = c / 2500 A = 1.199e15 Hz
    _NU_2500 = 1.199e15  # Hz
    _BC_2500 = 5.15  # Hopkins+2007 bolometric correction at 2500 A
    L_2500 = L_agn_bol / (_BC_2500 * _NU_2500)  # erg/s/Hz

    # alpha_ox = 0.384 * log10(L_2keV / L_2500A)
    # => L_2keV = L_2500 * 10^(alpha_ox / 0.384)
    L_2keV = L_2500 * 10.0 ** (alpha_ox / 0.384)

    # Power-law spectrum
    E_ref = 2.0  # keV
    spec = (E_keV / E_ref) ** (-gamma + 1) * jnp.exp(-E_keV / E_cut)

    # Normalise at 2 keV. ``L_2keV`` is already L_nu(2 keV) in erg/s/Hz
    # (alpha_ox is defined on monochromatic L_nu values, Tananbaum+1979), so
    # multiplying by the dimensionless ``spec`` (=1 at E=E_ref) gives L_nu(E).
    L_nu = L_2keV * spec

    xray_mask = wavelength < 124.0
    return jnp.where(xray_mask, L_nu, 0.0)


def xray_total(
    wavelength: jnp.ndarray,
    sfr: float = 1.0,
    stellar_mass: float = 1e10,
    L_agn_bol: float = 0.0,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    gamma_agn: float = 1.8,
    E_cut: float = 300.0,
    alpha_ox: float = -1.4,
    metallicity_z: float = 0.02,
    stellar_age_gyr: float = 5.0,
    include_hotgas: bool = True,
    **_kwargs,
) -> jnp.ndarray:
    """Total X-ray emission: XRB + diffuse hot gas + AGN corona.

    Combines the three CIGALE ``yang20`` building blocks: Lehmer+2016/2019
    metallicity- and age-dependent X-ray binaries
    (:func:`xray_xrb`), the SFR-driven diffuse hot-gas template
    (:func:`xray_hotgas`), and the α_ox-normalised AGN corona power-law
    (:func:`xray_agn_corona`).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength [Angstrom].
    sfr : float
        Star formation rate [Msun/yr].
    stellar_mass : float
        Stellar mass [Msun].
    L_agn_bol : float
        AGN bolometric luminosity [erg/s]. 0 = no AGN X-ray.
    gamma_hmxb : float
        HMXB photon index. Default 2.0.
    gamma_lmxb : float
        LMXB photon index. Default 1.6.
    gamma_agn : float
        AGN photon index. Default 1.8.
    E_cut : float
        Exponential cutoff energy [keV]. Default 300.
    alpha_ox : float
        UV-to-X-ray slope. Default -1.4.
    metallicity_z : float
        Stellar metallicity (mass fraction). Default 0.02 (solar).
    stellar_age_gyr : float
        Mass-weighted stellar age. Default 5.0. [Gyr]
    include_hotgas : bool
        Add the Mineo+2012 diffuse hot-gas component. Default True.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.
    """
    xrb = xray_xrb(
        wavelength,
        sfr,
        stellar_mass,
        gamma_hmxb,
        gamma_lmxb,
        E_cut,
        metallicity_z=metallicity_z,
        stellar_age_gyr=stellar_age_gyr,
    )
    agn = xray_agn_corona(wavelength, L_agn_bol, gamma_agn, E_cut, alpha_ox)
    hotgas = jnp.where(include_hotgas, xray_hotgas(wavelength, sfr), jnp.zeros_like(wavelength))
    return xrb + agn + hotgas


# ── Lopez+2024 IRX-based X-ray (for low-luminosity AGN) ─────────


def xray_bolometric_correction_duras(l_bol_erg: float) -> float:
    r"""X-ray bolometric correction k_X (Duras et al. 2020, Eq. 2).

    Returns the ratio k_X = L_bol / L_X(2-10 keV). For low-luminosity
    AGN (L_bol < 10^43 erg/s), the correction approaches a constant
    k_X ≈ 9.65 (Lopez et al. 2024 Eq. 6).

    Parameters
    ----------
    l_bol_erg : float
        AGN bolometric luminosity. [erg/s]

    Returns
    -------
    float
        Bolometric correction factor k_X = L_bol / L_X. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.

    The general form from Duras et al. (2020) is:

    .. math::

        k_X(L_{\rm bol}) = a \left[1 + \left(\frac{\log_{10}(L_{\rm bol})}{b}\right)^c\right]

    with best-fit coefficients a = 15.33, b = 11.48, c = 16.20
    (Duras et al. 2020, Table 2, X-ray correction).

    For the low-luminosity regime (L_bol ~ 10^38–10^43 erg/s),
    Lopez et al. (2024, Eq. 6) find a linear fit:

    .. math::

        L_{\rm bol} = (9.65 \pm 1.004) \times L_X

    References
    ----------
    .. [1] K. Duras et al., "Universal bolometric corrections for active
       galactic nuclei over seven luminosity decades," A&A, 636, A73 (2020).
       https://doi.org/10.1051/0004-6361/201936817
    .. [2] I. E. Lopez et al., "IRX-CIGALE: a tailored module for
       Low-Luminosity AGN," A&A, 692, A209 (2024). arXiv:2404.16938.
       https://doi.org/10.1051/0004-6361/202449801
    """
    a, b, c = 15.33, 11.48, 16.20
    log_l = jnp.log10(jnp.maximum(l_bol_erg, 1e-100))
    k_x = a * (1.0 + (log_l / b) ** c)
    return jnp.maximum(k_x, 1.0)


def xray_agn_corona_lopez24(
    wavelength: jnp.ndarray,
    l_12um_erg_hz: float,
    alpha_irx: float = 0.3,
    gamma: float = 1.8,
    E_cut: float = 300.0,
    cos_inc: float = 1.0,
    apply_anisotropy: bool = True,
    a1: float = 0.5,
    a2: float = 0.0,
) -> jnp.ndarray:
    r"""AGN corona X-ray emission from 12μm luminosity (Lopez et al. 2024).

    Uses the α_IRX parameter (ratio of 2–10 keV X-ray luminosity to nuclear
    12μm luminosity) to derive the X-ray luminosity. This approach is more
    robust than α_ox for obscured and low-luminosity AGN where UV emission
    is unreliable.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom (rest-frame). [Å]
    l_12um_erg_hz : float
        Nuclear monochromatic luminosity density at 12 μm. [erg/s/Hz]
    alpha_irx : float
        Log ratio of 2–10 keV to 12μm luminosity:
        α_IRX = log₁₀(L_X(2–10 keV) / L_12μm). [dimensionless]
        Default: 0.3. Typical range: 0.0–0.6.
    gamma : float
        X-ray photon index (Γ, where F_ν ∝ ν^{1−Γ}). [dimensionless]
        Default: 1.8. Typical range: 1.4–3.5.
    E_cut : float
        Exponential cutoff energy. [keV]
        Default: 300.0.
    cos_inc : float
        Cosine of inclination angle (1 = face-on, 0 = edge-on). [dimensionless]
        Default: 1.0.
    apply_anisotropy : bool
        Whether to apply Yang+2022 viewing-angle correction.
        Default: True.
    a1 : float
        Linear anisotropy coefficient. [dimensionless]
        Default: 0.5.
    a2 : float
        Quadratic anisotropy coefficient. [dimensionless]
        Default: 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives.

    The α_IRX parameter connects the X-ray and mid-IR luminosities via
    the Gandhi et al. (2009) L_X–L_12μm relation:

    .. math::

        \alpha_{\rm IRX} = \log_{10}\left(\frac{L_X^{2\text{--}10\,\rm keV}}{L_{12\mu\rm m}}\right)

    The intrinsic 2–10 keV luminosity is:

    .. math::

        L_X^{2\text{--}10\,\rm keV} = 10^{\alpha_{\rm IRX}} \times L_{12\mu\rm m}

    The advantage over α_ox: 12μm emission is dominated by the torus and
    is relatively unaffected by obscuration (scatter ≈ 0.33 dex vs UV
    which can be absorbed by orders of magnitude). This makes α_IRX
    especially suitable for LLAGN and obscured Seyferts.

    **Upstream**: Ported from CIGALE IRX-CIGALE module
    (Lopez et al. 2024 [1]_).

    References
    ----------
    .. [1] I. E. Lopez et al., "IRX-CIGALE: a tailored module for
       Low-Luminosity AGN," A&A, 692, A209 (2024). arXiv:2404.16938.
       https://doi.org/10.1051/0004-6361/202449801
    .. [2] P. Gandhi et al., "Resolving the mid-infrared cores of local
       Seyferts," A&A, 502, 457 (2009). arXiv:0905.2577.
       https://doi.org/10.1051/0004-6361/200811368
    .. [3] D. Asmus et al., "Local AGN survey (LASr): I. Galaxy sample,
       infrared and X-ray data," MNRAS, 454, 766 (2015).
       https://doi.org/10.1093/mnras/stv1950
    .. [4] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / 1.6022e-9

    # Derive L_X(2-10 keV) from α_IRX and L_12μm
    # L_12μm in erg/s: convert from erg/s/Hz using ν_12μm bandwidth
    nu_12um = _C_AA / 1.2e5  # 12 μm = 120000 Å
    l_12um_erg = l_12um_erg_hz * nu_12um
    l_x_2_10 = 10.0**alpha_irx * l_12um_erg

    # Build power-law spectrum with exponential cutoff
    E_ref = 5.0  # keV (mid-band reference)
    spec = (E_keV / E_ref) ** (-gamma + 1) * jnp.exp(-E_keV / E_cut)

    # Normalize by integrating spectral shape over 2-10 keV
    E_fine = jnp.linspace(2.0, 10.0, 200)
    nu_fine = E_fine * _KEV_TO_HZ
    spec_fine = (E_fine / E_ref) ** (-gamma + 1) * jnp.exp(-E_fine / E_cut)
    band_integral = jnp.maximum(jnp.trapezoid(spec_fine, nu_fine), 1e-60)

    l_nu = l_x_2_10 / band_integral * spec

    # X-ray mask (E > 0.1 keV => λ < 124 Å)
    l_nu = jnp.where(wavelength < 124.0, l_nu, 0.0)

    # Optional anisotropy correction (Yang+2022)
    if apply_anisotropy:
        l_nu = xray_anisotropy(l_nu, cos_inc, a1=a1, a2=a2)

    return l_nu


def xray_total_lopez24(
    wavelength: jnp.ndarray,
    sfr: float = 1.0,
    stellar_mass: float = 1e10,
    l_12um_erg_hz: float = 0.0,
    alpha_irx: float = 0.3,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    gamma_agn: float = 1.8,
    E_cut: float = 300.0,
    metallicity_z: float = 0.02,
    stellar_age_gyr: float = 5.0,
    include_hotgas: bool = True,
    **_kwargs,
) -> jnp.ndarray:
    r"""Total X-ray emission using IRX-based AGN (Lopez+2024) + XRBs.

    Combines X-ray binary emission (HMXB + LMXB) with AGN corona emission
    derived from the 12μm luminosity via α_IRX.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Angstrom. [Å]
    sfr : float
        Star formation rate. [Msun/yr]
    stellar_mass : float
        Stellar mass. [Msun]
    l_12um_erg_hz : float
        Nuclear 12μm luminosity density. [erg/s/Hz]
        0 = no AGN X-ray contribution.
    alpha_irx : float
        Log ratio of L_X to L_12μm. [dimensionless]
        Default: 0.3.
    gamma_hmxb : float
        HMXB photon index. Default: 2.0.
    gamma_lmxb : float
        LMXB photon index. Default: 1.6.
    gamma_agn : float
        AGN photon index. Default: 1.8.
    E_cut : float
        Exponential cutoff energy. [keV] Default: 300.

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    See :func:`xray_agn_corona_lopez24` for the α_IRX model details
    and :func:`xray_xrb` for the XRB component.
    """
    xrb = xray_xrb(
        wavelength,
        sfr,
        stellar_mass,
        gamma_hmxb,
        gamma_lmxb,
        E_cut,
        metallicity_z=metallicity_z,
        stellar_age_gyr=stellar_age_gyr,
    )
    agn = xray_agn_corona_lopez24(
        wavelength,
        l_12um_erg_hz,
        alpha_irx,
        gamma_agn,
        E_cut,
        apply_anisotropy=False,
    )
    hotgas = jnp.where(include_hotgas, xray_hotgas(wavelength, sfr), jnp.zeros_like(wavelength))
    return xrb + agn + hotgas
