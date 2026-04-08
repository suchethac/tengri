"""X-ray emission models for galaxies.

Predicts X-ray SED from three components:
1. X-ray binaries (HMXB + LMXB) — scaled from SFR and stellar mass
2. AGN corona — power-law with exponential cutoff
3. Hot gas — thermal bremsstrahlung (optional)

Self-consistent disc-corona coupling: ``xray_agn_corona_from_disc``
derives alpha_ox from the disc UV luminosity (Just+2007), then builds
an anisotropic X-ray power law (Yang+2022).

Based on CIGALE's Yang+2020 and Lopez+2024 modules.

All pure JAX, JIT-compatible.

References
----------
- Grimm et al. 2003, MNRAS, 339, 793 (HMXB-SFR relation)
- Gilfanov 2004, MNRAS, 349, 146 (LMXB-mass relation)
- Yang et al. 2020 (CIGALE X-ray module)
- Lopez et al. 2024 (updated IR-to-X-ray relation)
- Just et al. 2007, ApJ, 665, 1004 (alpha_ox-L_2500 relation)
- Yang et al. 2022, ApJ, 927, 42 (X-ray anisotropy)
"""

import jax.numpy as jnp

# Physical constants
_C_AA = 2.99792458e18  # Angstrom/s
_H_PLANCK = 6.62607015e-27  # erg s
_K_BOLTZ = 1.380649e-16  # erg/K
_KEV_TO_HZ = 2.418e17  # 1 keV = 2.418e17 Hz
_LSUN = 3.828e33  # erg/s


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
    """X-ray emission from X-ray binaries (HMXB + LMXB).

    HMXB luminosity scales with SFR (Grimm+2003):
      L_X(HMXB) = 2.6e39 * (SFR / Msun/yr) erg/s  (2-10 keV)

    LMXB luminosity scales with stellar mass (Gilfanov 2004):
      L_X(LMXB) = 8.3e28 * (M_star / Msun) erg/s  (2-10 keV)

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    sfr : float
        Star formation rate (Msun/yr).
    stellar_mass : float
        Stellar mass (Msun).
    gamma_hmxb : float
        HMXB photon index. Default 2.0.
    gamma_lmxb : float
        LMXB photon index. Default 1.6.
    E_cut : float
        Exponential cutoff energy (keV). Default 100.
    log_L_hmxb_offset : float
        Deviation from mean L_HMXB-SFR relation (dex). Default 0.
    log_L_lmxb_offset : float
        Deviation from mean L_LMXB-M* relation (dex). Default 0.

    Returns
    -------
    array (n_wave,)
        L_nu in erg/s/Hz.
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
        Monochromatic luminosity density at 2500 A in erg/s/Hz.

    Returns
    -------
    float
        alpha_ox (typically between -2.0 and -1.0).

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
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    l_2500_erg_hz : float
        Monochromatic luminosity density at 2500 A (erg/s/Hz).
    cos_inc : float
        Cosine of inclination (1 = face-on). Default 1.0.
    delta_alpha_ox : float
        Additive offset to the Just+2007 alpha_ox. Default 0.0.
    gamma : float
        Photon index. Default 1.8. Range: 1.4-2.4.
    E_cut : float
        Exponential cutoff energy (keV). Default 300.
    apply_anisotropy : bool
        Whether to apply Yang+2022 viewing-angle correction.
    a1 : float
        Linear anisotropy coefficient. Default 0.5.
    a2 : float
        Quadratic anisotropy coefficient. Default 0.0.

    Returns
    -------
    array (n_wave,)
        L_nu in erg/s/Hz.
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
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_agn_bol : float
        AGN bolometric luminosity (erg/s).
    gamma : float
        Photon index. Default 1.8. Range: 1.4-2.4.
    E_cut : float
        Cutoff energy (keV). Default 300.
    alpha_ox : float
        UV-to-X-ray slope. Default -1.4. Range: -2.0 to -1.0.

    Returns
    -------
    array (n_wave,)
        L_nu in erg/s/Hz.
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
    sfr : float
        Star formation rate (Msun/yr).
    stellar_mass : float
        Stellar mass (Msun).
    L_agn_bol : float
        AGN bolometric luminosity (Lsun). 0 = no AGN X-ray.
    """
    xrb = xray_xrb(wavelength, sfr, stellar_mass, gamma_hmxb, gamma_lmxb, E_cut)
    agn = xray_agn_corona(wavelength, L_agn_bol, gamma_agn, E_cut, alpha_ox)
    return xrb + agn
