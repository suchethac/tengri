"""X-ray emission models for galaxies.

Predicts X-ray SED from three components:
1. X-ray binaries (HMXB + LMXB) — scaled from SFR and stellar mass
2. AGN corona — power-law with exponential cutoff
3. Hot gas — thermal bremsstrahlung (optional)

Based on CIGALE's Yang+2020 and Lopez+2024 modules.

All pure JAX, JIT-compatible.

References
----------
- Grimm et al. 2003, MNRAS, 339, 793 (HMXB-SFR relation)
- Gilfanov 2004, MNRAS, 349, 146 (LMXB-mass relation)
- Yang et al. 2020 (CIGALE X-ray module)
- Lopez et al. 2024 (updated IR-to-X-ray relation)
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
        L_nu in Lsun/Hz.
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / (1.6022e-9)  # convert to keV

    # Reference luminosities (erg/s in 2-10 keV)
    L_hmxb_ref = 2.6e39 * sfr * 10.0**log_L_hmxb_offset
    L_lmxb_ref = 8.3e28 * stellar_mass * 10.0**log_L_lmxb_offset

    # Power-law with exponential cutoff: dN/dE ~ E^{-Gamma} * exp(-E/E_cut)
    # L_nu = L_ref * (E/E_ref)^{-Gamma+1} * exp(-E/E_cut) (energy flux per Hz)
    E_ref = 5.0  # keV (geometric mean of 2-10 keV)

    spec_hmxb = (E_keV / E_ref) ** (-gamma_hmxb + 1) * jnp.exp(-E_keV / E_cut)
    spec_lmxb = (E_keV / E_ref) ** (-gamma_lmxb + 1) * jnp.exp(-E_keV / E_cut)

    # Normalize: integral over 2-10 keV should give the reference luminosity
    # For simplicity, normalize at E_ref and scale
    L_nu_hmxb = (L_hmxb_ref / _LSUN) / (E_ref * _KEV_TO_HZ) * spec_hmxb
    L_nu_lmxb = (L_lmxb_ref / _LSUN) / (E_ref * _KEV_TO_HZ) * spec_lmxb

    # X-ray only (E > 0.1 keV = lambda < 124 A)
    xray_mask = wavelength < 124.0
    return jnp.where(xray_mask, L_nu_hmxb + L_nu_lmxb, 0.0)


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
        AGN bolometric luminosity (Lsun).
    gamma : float
        Photon index. Default 1.8. Range: 1.4-2.4.
    E_cut : float
        Cutoff energy (keV). Default 300.
    alpha_ox : float
        UV-to-X-ray slope. Default -1.4. Range: -2.0 to -1.0.

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.
    """
    nu = _C_AA / wavelength
    E_keV = _H_PLANCK * nu / (1.6022e-9)

    # L_2500A from bolometric correction: L_2500 ~ L_bol / 5
    L_2500 = L_agn_bol / 5.0  # Lsun/Hz (rough)

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
