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
) -> jnp.ndarray:
    r"""Predict X-ray SED from accretion-powered binaries.

    Computes the combined X-ray emission from high-mass (HMXB) and low-mass
    (LMXB) X-ray binary populations, scaled by SFR and stellar mass respectively.
    Each population is modelled as a power-law with exponential cutoff.

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

    **HMXB luminosity scaling** (Grimm et al. 2003, MNRAS 339, 793, Eq. 1):
        HMXBs are young binary systems (age < 100 Myr) with massive companions,
        so their population follows the instantaneous SFR:

        .. math::

            L_X^{\mathrm{HMXB}}(2\text{–}10\,\mathrm{keV}) =
                2.6 \times 10^{39} \times \left(\frac{\mathrm{SFR}}{M_\odot/\mathrm{yr}}\right)
                \quad [\mathrm{erg/s}]

        A coefficient offset (log_L_hmxb_offset) captures intrinsic scatter
        or evolutionary effects.

    **LMXB luminosity scaling** (Gilfanov 2004, MNRAS 349, 146, Eq. 1):
        LMXBs are old systems (age > 1 Gyr), so their population traces
        stellar mass:

        .. math::

            L_X^{\mathrm{LMXB}}(2\text{–}10\,\mathrm{keV}) =
                8.3 \times 10^{28} \times \left(\frac{M_\star}{M_\odot}\right)
                \quad [\mathrm{erg/s}]

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
    scatter (e.g., metallicity effects on binary evolution) or
    redshift-dependent evolution in hierarchical models.

    References
    ----------
    .. [1] H.-J. Grimm et al., "High-mass X-ray binaries as a star formation
       rate indicator in distant galaxies," MNRAS, 339, 793 (2003).
       https://doi.org/10.1046/j.1365-8711.2003.06224.x
    .. [2] M. Gilfanov, "Low-mass X-ray binaries as a stellar mass indicator
       for the host galaxy," MNRAS, 349, 146 (2004). arXiv:astro-ph/0309171.
       https://doi.org/10.1111/j.1365-2966.2004.07473.x
    .. [3] G. Yang et al., "Fitting AGN/galaxy X-ray-to-radio SEDs with
       CIGALE and improvement of the code," ApJ, 927, 192 (2022).
       https://doi.org/10.3847/1538-4357/ac4971
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / (1.6022e-9)  # convert to keV

    # Reference luminosities (erg/s in 2-10 keV)
    L_hmxb_ref = 2.6e39 * sfr * 10.0**log_L_hmxb_offset
    L_lmxb_ref = 8.3e28 * stellar_mass * 10.0**log_L_lmxb_offset

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
    """Compute alpha_ox from monochromatic 2500 A luminosity (Just+2007).

    Parameters
    ----------
    l_2500_erg_hz : float
        Monochromatic luminosity density at 2500 A [erg/s/Hz].

    Returns
    -------
    float
        alpha_ox (typically between -2.0 and -1.0) [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    References
    ----------
    Just et al. 2007, ApJ, 665, 1004, Eq. 3.
    """
    return -0.137 * jnp.log10(l_2500_erg_hz) + 2.638


def xray_anisotropy(
    l_x: jnp.ndarray,
    cos_inc: float,
    a1: float = 0.5,
    a2: float = 0.0,
) -> jnp.ndarray:
    """Apply viewing-angle anisotropy to X-ray luminosity (Yang+2022).

    The correction factor is normalised so that face-on (cos_inc=1)
    gives maximum luminosity:

      f(theta) = a1 * cos(theta) + a2 * cos^2(theta) + (1 - a1 - a2)

    Parameters
    ----------
    l_x : array
        Isotropic (face-on) X-ray luminosity spectrum.
    cos_inc : float
        Cosine of inclination angle (1 = face-on, 0 = edge-on).
    a1 : float
        Linear anisotropy coefficient. Default 0.5.
    a2 : float
        Quadratic anisotropy coefficient. Default 0.0.

    Returns
    -------
    array
        Anisotropy-corrected L_X, same shape as ``l_x``.

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.

    References
    ----------
    Yang et al. 2022, ApJ, 927, 42.
    """
    factor = a1 * cos_inc + a2 * cos_inc**2 + (1.0 - a1 - a2)
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

    # Derive L_2keV from alpha_ox definition:
    #   alpha_ox = 0.384 * log10(L_2keV / L_2500)
    #   => L_2keV = L_2500 * 10^(alpha_ox / 0.384)
    l_2kev_erg_hz = l_2500_erg_hz * 10.0 ** (alpha_ox / 0.384)

    # Build power-law spectrum with exponential cutoff
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / 1.6022e-9  # convert to keV
    E_ref = 2.0  # keV
    spec = (E_keV / E_ref) ** (-gamma + 1) * jnp.exp(-E_keV / E_cut)

    # Normalise at 2 keV
    l_nu = l_2kev_erg_hz / (_KEV_TO_HZ * E_ref) * spec

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

    # Normalize at 2 keV
    L_nu = L_2keV / (_KEV_TO_HZ * E_ref) * spec

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
    **_kwargs,
) -> jnp.ndarray:
    """Total X-ray emission (XRB + AGN corona).

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

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — pure JAX function.
    """
    xrb = xray_xrb(wavelength, sfr, stellar_mass, gamma_hmxb, gamma_lmxb, E_cut)
    agn = xray_agn_corona(wavelength, L_agn_bol, gamma_agn, E_cut, alpha_ox)
    return xrb + agn


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
