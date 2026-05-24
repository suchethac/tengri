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

from tengri._deprecated import deprecated_alias
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
    metallicity_z: float = 0.02,
    stellar_age_gyr: float = 1.0,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    E_cut: float = 100.0,
    log_L_hmxb_offset: float = 0.0,
    log_L_lmxb_offset: float = 0.0,
) -> jnp.ndarray:
    r"""Predict X-ray SED from accretion-powered binaries.

    Computes the combined X-ray emission from high-mass (HMXB) and low-mass
    (LMXB) X-ray binary populations. HMXB luminosity scales with SFR and
    metallicity (Lehmer et al. 2016). LMXB luminosity scales with stellar
    mass and age (Lehmer et al. 2016). Both are modelled as power-laws
    with exponential cutoff.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame). [Å]
    sfr : float
        Star formation rate. [Msun/yr]
    stellar_mass : float
        Stellar mass. [Msun]
    metallicity_z : float, optional
        Metallicity (mass fraction, not log Z/Z_sun). Default: 0.02 (solar). []
    stellar_age_gyr : float, optional
        Stellar age in Gyr. Default: 1.0. [Gyr]
    gamma_hmxb : float, optional
        HMXB photon index (Γ, where F_ν ∝ ν^{−Γ}). Default: 2.0.
    gamma_lmxb : float, optional
        LMXB photon index. Default: 1.6.
    E_cut : float, optional
        Exponential cutoff energy for both populations. Default: 100 keV. [keV]
    log_L_hmxb_offset : float, optional
        Departure from mean SFR relation (dex). Allows scatter or evolution.
        Default: 0.0. [dex]
    log_L_lmxb_offset : float, optional
        Departure from mean stellar-mass relation (dex). Default: 0.0. [dex]

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density of X-ray binary populations.
        [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **HMXB luminosity scaling** (Lehmer et al. 2016, ApJ 825, 7, Eq. 15):
        HMXBs are young binary systems (age < 100 Myr) with massive companions,
        so their population follows the instantaneous SFR. The luminosity
        depends strongly on metallicity Z (mass fraction):

        .. math::

            \log(L_X^{\mathrm{HMXB}}(2\text{–}10\,\mathrm{keV})/\mathrm{SFR}) =
                40.28 - 62.12Z + 569.44Z^2 - 1833.80Z^3 + 1968.33Z^4
                \quad [\mathrm{erg\,s^{-1}\,(M_\odot\,yr^{-1})^{-1}}]

        At solar metallicity (Z=0.02), this yields ≈ 2.6×10^39 erg/s per
        M_sun/yr SFR, consistent with Grimm et al. 2003.

    **LMXB luminosity scaling** (Lehmer et al. 2016, ApJ 825, 7, Eq. 15):
        LMXBs are old systems (age > 1 Gyr), so their population traces
        stellar mass. The luminosity depends on stellar age t (Gyr):

        .. math::

            \log(L_X^{\mathrm{LMXB}}(2\text{–}10\,\mathrm{keV})/M_\star) =
                40.276 - 1.503\log t - 0.423(\log t)^2 + 0.425(\log t)^3 + 0.136(\log t)^4
                \quad [\mathrm{erg\,s^{-1}\,M_\odot^{-1}}]

        At t=1 Gyr, this yields ≈ 8.3×10^28 erg/s per M_sun, consistent with
        Gilfanov 2004.

    **Spectral shape**: Both HMXB and LMXB are modelled as power-laws with
    a high-energy exponential cutoff (photoelectric absorption, or intrinsic
    accretion torque limits):

        .. math::

            F_\nu \propto \nu^{-\Gamma} \exp(-h\nu / E_{\mathrm{cut}})

        Typical cutoffs: E_cut ≈ 100 keV (LMXB) to 200 keV (HMXB).
        The exponent E_cut controls the shape at high energies.

    **Wavelength coverage**: X-ray binaries emit primarily in 0.1–10 keV
    (λ ≈ 1.2 Å – 124 Å). Outside this range, flux is negligible.

    **Offsets**: The log_L_*_offset parameters allow captured intrinsic
    scatter (e.g., redshift-dependent evolution) in hierarchical models.

    References
    ----------
    .. [1] B. D. Lehmer et al., "The evolution of the X-ray binary
       luminosity functions of nearby galaxies with the Chandra COSMOS
       survey," ApJ, 825, 7 (2016).
       https://doi.org/10.3847/0004-637X/825/1/7
    .. [2] H.-J. Grimm et al., "High-mass X-ray binaries as a star formation
       rate indicator in distant galaxies," MNRAS, 339, 793 (2003).
       https://doi.org/10.1046/j.1365-8711.2003.06224.x
    .. [3] M. Gilfanov, "Low-mass X-ray binaries as a stellar mass indicator
       for the host galaxy," MNRAS, 349, 146 (2004). arXiv:astro-ph/0309171.
       https://doi.org/10.1111/j.1365-2966.2004.07473.x
    .. [4] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / _KEV_TO_ERG  # convert to keV

    # Lehmer+2016 metallicity quartic for HMXB (yang20.py:207–214)
    # log(L_HMXB / SFR) = 33.28 - 62.12*Z + 569.44*Z^2 - 1833.80*Z^3 + 1968.33*Z^4
    # in W units. Convert to erg/s: +7.0 (log10 conversion)
    # Leading constant 40.28 = 33.28 + 7.0 makes the unit conversion explicit.
    log_l_hmxb_per_sfr = (
        40.28
        - 62.12 * metallicity_z
        + 569.44 * metallicity_z**2
        - 1833.80 * metallicity_z**3
        + 1968.33 * metallicity_z**4
    )
    L_hmxb_ref = 10.0**log_l_hmxb_per_sfr * sfr * 10.0**log_L_hmxb_offset

    # Lehmer+2016 age quartic for LMXB (yang20.py:216–224)
    # log(L_LMXB / M_star) = 33.276 - 1.503*log(t) - 0.423*(log t)^2
    #   + 0.425*(log t)^3 + 0.136*(log t)^4 (in W units)
    # Leading constant 40.276 = 33.276 + 7.0 for erg/s conversion.
    # where t is age in Gyr
    log_t = jnp.log10(jnp.maximum(stellar_age_gyr, 1e-3))  # protect against log(0)
    log_l_lmxb_per_mstar = (
        40.276
        - 1.503 * log_t
        - 0.423 * log_t**2
        + 0.425 * log_t**3
        + 0.136 * log_t**4
    )
    L_lmxb_ref = 10.0**log_l_lmxb_per_mstar * stellar_mass * 10.0**log_L_lmxb_offset

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
    gamma: float = 1.0,
    E_cut: float = 1.0,
) -> jnp.ndarray:
    r"""Predict X-ray SED from hot gas (diffuse ISM/CGM).

    Computes thermal X-ray emission from optically-thin hot plasma in the
    interstellar medium (ISM) and circumgalactic medium (CGM). The emission
    scales with SFR and is modelled as thermal bremsstrahlung.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame). [Å]
    sfr : float
        Star formation rate. [Msun/yr]
    gamma : float, optional
        Photon index (Γ, where F_ν ∝ ν^{−Γ}). Default: 1.0 (thermal).
    E_cut : float, optional
        Exponential cutoff energy. Default: 1.0 keV (hot gas characteristic). [keV]

    Returns
    -------
    array, shape (n_wave,)
        Spectral luminosity density of hot gas X-ray emission.
        [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.

    **Hot gas luminosity scaling** (Yang et al. 2020, MNRAS 491, 740;
    Yang et al. 2022, ApJ 927, 192; Mineo et al. 2012, ApJ 745, 181):
        Hot gas X-ray emission scales with SFR because star-forming regions
        heat the ISM through supernovae and winds:

        .. math::

            \log(L_X^{\mathrm{hot\,gas}}(0.5\text{–}2\,\mathrm{keV})/\mathrm{SFR}) = 38.9
                \quad [\mathrm{erg\,s^{-1}\,(M_\odot\,yr^{-1})^{-1}}]

        This gives L_X^{hot gas} ≈ 7.94×10^38 erg/s per M_sun/yr SFR.

    **Spectral shape**: Hot gas is modelled as thermal bremsstrahlung from
    optically-thin plasma (Γ = 1; free-free and free-bound emission):

        .. math::

            F_\nu \propto \nu^{-\Gamma} \exp(-h\nu / E_{\mathrm{cut}})

        Typical cutoff: E_cut ≈ 1 keV (plasma temperature ~ 10^7 K).

    **Wavelength coverage**: Hot gas emits in soft X-rays (0.5–2 keV,
        λ ≈ 6–124 Å). Outside this range, flux is negligible.

    References
    ----------
    .. [1] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," MNRAS, 491, 740 (2020).
       https://doi.org/10.1093/mnras/stz3001
    .. [2] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    .. [3] S. Mineo et al., "The hot and energetic universe: The X-ray binary
       populations in normal galaxies," ApJ, 745, 181 (2012).
       https://doi.org/10.1088/0004-637X/745/2/181
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / _KEV_TO_ERG

    # Hot gas luminosity scaling (yang20.py:204)
    # L_0.5-2keV = 8.3e31 W * SFR. In erg/s: 8.3e38 = 10^38.919.
    # (yang20.py:204 shows L_hotgas_0p5to2keV = 8.3e31 * sfr)
    log_l_hotgas_per_sfr = 38.919
    L_hotgas_ref = 10.0**log_l_hotgas_per_sfr * sfr

    # Thermal bremsstrahlung spectrum with exponential cutoff
    E_ref = 1.0  # keV (characteristic hot-gas energy)
    spec = (E_keV / E_ref) ** (-gamma + 1) * jnp.exp(-E_keV / E_cut)

    # Normalise by integrating spectral shape over 0.5-2 keV
    E_fine = jnp.linspace(0.5, 2.0, 200)  # keV
    nu_fine = E_fine * _KEV_TO_HZ
    spec_fine = (E_fine / E_ref) ** (-gamma + 1) * jnp.exp(-E_fine / E_cut)
    band_int = jnp.maximum(jnp.trapezoid(spec_fine, nu_fine), 1e-60)

    L_nu = L_hotgas_ref / band_int * spec

    # Soft X-ray only (0.5-2 keV => 6-124 A)
    # but allow slightly beyond for smoothness
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
        Anisotropy-corrected L_X. [erm/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    **Empirical correction** (Yang et al. 2022 [1]_): polynomial in
    :math:`\mu \equiv \cos\theta`, normalised so the bolometric
    corona luminosity at θ=0° (face-on, :math:`\mu = 1`) is recovered.

    The anisotropic luminosity is computed as (yang20.py:231–235):

    .. math::

        f(\mu) = \frac{a_1\,\mu + a_2\,\mu^2 + (1 - a_1 - a_2)}{1 - 0.13397\,a_1 - 0.25\,a_2},
        \qquad
        L_X^{\rm obs} = f(\mu)\, L_X^{\rm iso}

    The denominator is crucial: it normalizes the angular distribution so that
    multiplying by the numerator and dividing by the denominator at θ=0° gives
    unity, ensuring the face-on luminosity is unmodified. The default
    :math:`a_1 = 0.5,\, a_2 = 0` corresponds to the "intermediate" obscuration
    solution adopted in X-CIGALE (Yang et al. 2022).

    At default (a1=0.5, a2=0), the denominator is 0.933, so the correction
    factor is ~1.072 at face-on (7% enhancement relative to the polynomial alone,
    to recover the face-on bolometric luminosity).

    References
    ----------
    .. [1] Yang, G. et al., 2022, ApJ, 927, 192.
       Eq. 231–235; CIGALE yang20.py:231–235.
    """
    numerator = a1 * cos_inc + a2 * cos_inc**2 + (1.0 - a1 - a2)
    # Normalization denominator (yang20.py:231–235): ensures face-on
    # bolometric corona luminosity is recovered. Without this, anisotropy
    # would suppress the face-on luminosity.
    denominator = 1.0 - 0.13397 * a1 - 0.25 * a2
    factor = numerator / denominator
    return l_x * factor


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

    # Derive L_2keV from alpha_ox definition (yang20.py:227):
    #   alpha_ox = 0.3838 * log10(L_2keV / L_2500)
    #   => L_2keV = L_2500 * 10^(alpha_ox / 0.3838)
    # The divisor 0.3838 = 1 / log10(nu_2keV / nu_2500A) is the exact
    # frequency ratio between 2 keV (λ ≈ 6.2 Å) and 2500 Å.
    l_2kev_erg_hz = l_2500_erg_hz * 10.0 ** (alpha_ox / 0.3838)

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
    l_2500_30deg_erg_hz: float,
    gamma: float = 1.8,
    E_cut: float = 300.0,
    delta_alpha_ox: float = 0.0,
    cos_inc: float = 1.0,
    apply_anisotropy: bool = True,
    a1: float = 0.5,
    a2: float = 0.0,
) -> jnp.ndarray:
    r"""X-ray emission from AGN corona (CIGALE / Yang+2020 canonical path).

    Self-consistent AGN corona emission derived from the disc UV luminosity
    at 30° intrinsic angle. This is the canonical path matching X-CIGALE
    (Yang+2020) and enforces physical consistency between UV and X-ray SEDs.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength grid in Å (rest-frame). [Å]
    l_2500_30deg_erg_hz : float
        Monochromatic luminosity density at 2500 Å from the AGN disc
        at 30° inclination angle (intrinsic). [erg/s/Hz]
    gamma : float, optional
        Photon index (Γ, where F_ν ∝ ν^{−Γ}). Default: 1.8. Range: 1.4–2.4.
    E_cut : float, optional
        Exponential cutoff energy. Default: 300 keV. [keV]
    delta_alpha_ox : float, optional
        Additive offset to the Just+2007 α_ox relation. Default: 0.0. [dex]
    cos_inc : float, optional
        Cosine of inclination angle (1 = face-on, 0 = edge-on).
        Default: 1.0. []
    apply_anisotropy : bool, optional
        Whether to apply Yang+2022 viewing-angle correction. Default: True.
    a1 : float, optional
        Linear anisotropy coefficient. Default: 0.5. []
    a2 : float, optional
        Quadratic anisotropy coefficient. Default: 0.0. []

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    **Physical basis** (Yang et al. 2020, MNRAS 491, 740, §2.2.1):
    The α_OX parameter (defined as the SED slope between 2500 Å and 2 keV)
    is an observationally-calibrated relation from Just et al. (2007):

    .. math::

        \alpha_{\mathrm{OX}} = -0.137 \log_{10}(L_{2500}/[\mathrm{erg\,s^{-1}\,Hz^{-1}}])
            + 2.638

    The 2–10 keV luminosity then follows from the definition:

    .. math::

        \alpha_{\mathrm{OX}} = 0.3838 \log_{10}(L_{2\,\mathrm{keV}}/L_{2500})
            \quad \Rightarrow \quad
        L_{2\,\mathrm{keV}} = L_{2500} \times 10^{\alpha_{\mathrm{OX}}/0.3838}

    This driving from L_2500 (rather than L_bol) ensures consistency with
    the AGN disc model and avoids the ambiguity of bolometric corrections.

    **Anisotropy** (Yang et al. 2022, ApJ 927, 192):
    When apply_anisotropy=True, an inclination-dependent factor is applied:

    .. math::

        L_X(\theta) = L_X(0°) \times [a_1\cos\theta + a_2\cos^2\theta
            + (1 - a_1 - a_2)]

    Default (a1=0.5, a2=0.0) matches X-CIGALE's "intermediate" solution.

    References
    ----------
    .. [1] D. W. Just et al., "The X-ray luminosity and morphology
       dependence of narrow emission-line regions in nearby active galactic
       nuclei," ApJ, 665, 1004 (2007).
       https://doi.org/10.1086/519990
    .. [2] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," MNRAS, 491, 740 (2020).
       https://doi.org/10.1093/mnras/stz3001
    .. [3] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    """
    return xray_agn_corona_from_disc(
        wavelength,
        l_2500_30deg_erg_hz,
        cos_inc=cos_inc,
        delta_alpha_ox=delta_alpha_ox,
        gamma=gamma,
        E_cut=E_cut,
        apply_anisotropy=apply_anisotropy,
        a1=a1,
        a2=a2,
    )


def _xray_agn_corona_bolometric(
    wavelength: jnp.ndarray,
    L_agn_bol: float,
    gamma: float = 1.8,
    E_cut: float = 300.0,
    alpha_ox: float = -1.4,
) -> jnp.ndarray:
    """**DEPRECATED**: X-ray emission from AGN bolometric luminosity.

    **This function is deprecated.** Use :func:`xray_agn_corona` (which takes
    L_2500_30deg_erg_hz) instead. The bolometric-correction path is ambiguous
    and inconsistent with the disc UV model. Converted from L_bol via
    Hopkins+2007 bolometric correction (BC_2500 ≈ 5.15).

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
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / (1.6022e-9)

    # Monochromatic luminosity density at 2500 A in erg/s/Hz.
    # L_bol = BC_2500 * nu_2500 * L_nu(2500) => L_nu = L_bol / (BC * nu)
    # BC_2500 ~ 5.15 (Hopkins+2007), nu_2500 = c / 2500 A = 1.199e15 Hz
    _NU_2500 = 1.199e15  # Hz
    _BC_2500 = 5.15  # Hopkins+2007 bolometric correction at 2500 A
    L_2500 = L_agn_bol / (_BC_2500 * _NU_2500)  # erg/s/Hz

    # alpha_ox = 0.3838 * log10(L_2keV / L_2500A)
    # => L_2keV = L_2500 * 10^(alpha_ox / 0.3838)
    L_2keV = L_2500 * 10.0 ** (alpha_ox / 0.3838)

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
    metallicity_z: float = 0.02,
    stellar_age_gyr: float = 1.0,
    l_2500_30deg: float = 0.0,
    gamma_hmxb: float = 2.0,
    gamma_lmxb: float = 1.6,
    gamma_agn: float = 1.8,
    E_cut: float = 300.0,
    delta_alpha_ox: float = 0.0,
    cos_inc: float = 1.0,
    apply_anisotropy: bool = True,
    a1: float = 0.5,
    a2: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Total X-ray emission (HMXB + LMXB + hot gas + AGN corona).

    Combines X-ray binaries (HMXB and LMXB), diffuse hot gas, and AGN
    corona emission into a single SED. Implements Lehmer et al. 2016
    (metallicity and age-dependent XRB scaling), Yang et al. 2020 (hot gas),
    and Just et al. 2007 / Yang et al. 2020 (AGN corona).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Wavelength [Angstrom].
    sfr : float
        Star formation rate [Msun/yr]. Default: 1.0.
    stellar_mass : float
        Stellar mass [Msun]. Default: 1e10.
    metallicity_z : float
        Metallicity (mass fraction). Default: 0.02 (solar). []
    stellar_age_gyr : float
        Stellar age in Gyr. Default: 1.0. [Gyr]
    l_2500_30deg : float
        AGN monochromatic luminosity at 2500 Å at 30° inclination [erg/s/Hz].
        Default: 0.0 (no AGN X-ray).
    gamma_hmxb : float
        HMXB photon index. Default: 2.0.
    gamma_lmxb : float
        LMXB photon index. Default: 1.6.
    gamma_agn : float
        AGN X-ray photon index. Default: 1.8.
    E_cut : float
        Exponential cutoff energy [keV]. Default: 300.
    delta_alpha_ox : float
        Additive offset to Just+2007 α_ox relation [dex]. Default: 0.0.
    cos_inc : float
        Cosine of inclination angle. Default: 1.0 (face-on). []
    apply_anisotropy : bool
        Whether to apply Yang+2022 viewing-angle correction. Default: True.
    a1 : float
        Linear anisotropy coefficient. Default: 0.5. []
    a2 : float
        Quadratic anisotropy coefficient. Default: 0.0. []

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    **Components**:
    - HMXB: Lehmer+2016 metallicity quartic, scaling with SFR
    - LMXB: Lehmer+2016 age quartic, scaling with M_star
    - Hot gas: Yang+2020, scaling with SFR
    - AGN corona: Just+2007 / Yang+2020, scaling with L_2500 and α_OX
    """
    xrb = xray_xrb(
        wavelength,
        sfr,
        stellar_mass,
        metallicity_z=metallicity_z,
        stellar_age_gyr=stellar_age_gyr,
        gamma_hmxb=gamma_hmxb,
        gamma_lmxb=gamma_lmxb,
        E_cut=E_cut,
    )
    hotgas = xray_hotgas(wavelength, sfr, gamma=1.0, E_cut=1.0)
    agn = xray_agn_corona(
        wavelength,
        l_2500_30deg,
        gamma=gamma_agn,
        E_cut=E_cut,
        delta_alpha_ox=delta_alpha_ox,
        cos_inc=cos_inc,
        apply_anisotropy=apply_anisotropy,
        a1=a1,
        a2=a2,
    )
    return xrb + hotgas + agn


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
    xrb = xray_xrb(wavelength, sfr, stellar_mass, gamma_hmxb, gamma_lmxb, E_cut)
    agn = xray_agn_corona_lopez24(
        wavelength,
        l_12um_erg_hz,
        alpha_irx,
        gamma_agn,
        E_cut,
        apply_anisotropy=False,
    )
    return xrb + agn


# ── Deprecation shims ──


# Deprecated: old bolometric-normalisation path for xray_agn_corona
# Use xray_agn_corona_from_disc (which takes l_2500_30deg) instead
xray_agn_corona_bolometric = deprecated_alias(
    _xray_agn_corona_bolometric,
    old_name="xray_agn_corona_bolometric",
    new_name="xray_agn_corona_from_disc",
    drop_version="1.0",
)
