"""Radio emission models (synchrotron + free-free).

Predicts radio SED from star formation rate (via FIR-radio correlation)
and AGN (via radio-loudness parameter).

The total radio luminosity has two components:
1. Star-forming: synchrotron from supernova remnants, scaled via q_IR
2. AGN: radio jets/lobes, parameterized by radio-loudness R

All pure JAX, JIT-compatible.

References
----------
- Condon 1992, ARA&A, 30, 575 (radio emission from galaxies)
- Bell 2003, ApJ, 586, 794 (FIR-radio correlation)
- Delhaize et al. 2017, A&A, 602, A4 (q_IR evolution)
- Delvecchio et al. 2021, A&A, 647, A123 (radio-AGN)
- Martinez-Ramirez et al. 2024, A&A (AGNfitter-rx double power-law)
"""

import jax.numpy as jnp

# Physical constants
_C_CGS = 2.99792458e10  # cm/s
_C_AA = 2.99792458e18  # Angstrom/s
_LSUN = 3.828e33  # erg/s
_JY = 1e-23  # erg/s/cm^2/Hz


def radio_star_forming(
    wavelength: jnp.ndarray,
    L_ir: float,
    q_ir: float = 2.64,
    alpha_sf: float = 0.8,
    nu_ref: float = 1.4e9,
) -> jnp.ndarray:
    """Synchrotron emission from star formation via FIR-radio correlation.

    L_nu(radio) = L_IR / (10^q_IR) * (nu/nu_ref)^{-alpha}

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_ir : float
        Total infrared luminosity (8-1000 um) in Lsun.
    q_ir : float
        FIR-radio correlation parameter. Default 2.64 (Bell 2003).
        Evolves with redshift: q_IR(z) ~ 2.64 * (1+z)^{-0.15} (Delhaize+2017).
    alpha_sf : float
        Synchrotron spectral index. Default 0.8.
    nu_ref : float
        Reference frequency in Hz. Default 1.4 GHz.

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.
    """
    nu = _C_AA / wavelength  # Hz

    # L_1.4GHz from FIR-radio correlation: q_IR = log10(L_IR / (3.75e12 * L_1.4GHz))
    # => L_1.4GHz = L_IR / (3.75e12 * 10^q_IR)
    L_ref = L_ir / (3.75e12 * 10.0**q_ir)  # Lsun/Hz at nu_ref

    # Power-law extrapolation
    L_nu = L_ref * (nu / nu_ref) ** (-alpha_sf)

    # Only emit at radio frequencies (nu < 300 GHz = lambda > 1 mm = 1e7 A)
    radio_mask = wavelength > 1e7
    return jnp.where(radio_mask, L_nu, 0.0)


def radio_agn(
    wavelength: jnp.ndarray,
    L_agn_bol: float,
    radio_loudness: float = 0.0,
    alpha_agn: float = 0.7,
    nu_ref: float = 1.4e9,
) -> jnp.ndarray:
    """Radio emission from AGN jets/lobes.

    L_nu(radio) = R * L_5GHz_from_Lbol * (nu/5GHz)^{-alpha}

    The radio-loudness parameter R = log10(L_5GHz / L_B) where L_B is
    the B-band luminosity. R > 1 = radio-loud, R < 1 = radio-quiet.

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_agn_bol : float
        AGN bolometric luminosity in Lsun.
    radio_loudness : float
        log10(L_5GHz / L_B). Default 0 (radio-quiet). Range: -2 to 5.
    alpha_agn : float
        AGN radio spectral index. Default 0.7.
    nu_ref : float
        Reference frequency. Default 1.4 GHz.

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.
    """
    nu = _C_AA / wavelength

    # Monochromatic luminosity density at B-band (4400 A) in Lsun/Hz.
    # L_bol = BC_B * nu_B * L_nu(4400) => L_nu = L_bol / (BC_B * nu_B)
    # BC_B ~ 5.15 (Hopkins+2007, Table 1), nu_B = c / 4400 A = 6.818e14 Hz
    _NU_B = 6.818e14  # Hz
    _BC_B = 5.15  # Hopkins+2007 bolometric correction at 4400 A
    L_B = L_agn_bol / (_BC_B * _NU_B)  # Lsun/Hz

    # L_5GHz from radio-loudness definition
    L_5GHz = L_B * 10.0**radio_loudness  # Lsun/Hz

    # Power-law extrapolation from 5 GHz
    nu_5GHz = 5.0e9
    L_nu = L_5GHz * (nu / nu_5GHz) ** (-alpha_agn)

    radio_mask = wavelength > 1e7
    return jnp.where(radio_mask, L_nu, 0.0)


def radio_total(
    wavelength: jnp.ndarray,
    L_ir: float = 0.0,
    L_agn_bol: float = 0.0,
    q_ir: float = 2.64,
    alpha_sf: float = 0.8,
    radio_loudness: float = 0.0,
    alpha_agn: float = 0.7,
    **_kwargs,
) -> jnp.ndarray:
    """Total radio emission (star-forming + AGN).

    Parameters
    ----------
    L_ir : float
        Total IR luminosity (Lsun) for SF component.
    L_agn_bol : float
        AGN bolometric luminosity (Lsun) for AGN component.
    q_ir : float
        FIR-radio correlation parameter.
    alpha_sf : float
        SF synchrotron spectral index.
    radio_loudness : float
        AGN radio-loudness log10(L_5GHz/L_B).
    alpha_agn : float
        AGN radio spectral index.
    """
    sf = radio_star_forming(wavelength, L_ir, q_ir, alpha_sf)
    agn = radio_agn(wavelength, L_agn_bol, radio_loudness, alpha_agn)
    return sf + agn


def radio_agn_dpl(
    wavelength: jnp.ndarray,
    L_agn_bol: float,
    radio_loudness: float = 0.0,
    alpha1: float = -0.75,
    alpha2: float = -0.1,
    log_nu_t: float = 10.0,
    log_nu_cut: float = 13.0,
    nu_ref: float = 5.0e9,
) -> jnp.ndarray:
    """Double power-law AGN radio emission (AGNfitter-rx).

    Implements the broken power-law model from Martinez-Ramirez+2024 (Eq. 9-10)
    with an optically-thick low-frequency regime, an optically-thin steep
    high-frequency regime, and a synchrotron aging exponential cutoff.

    .. math::

        L_\\nu = L_{5\\mathrm{GHz}} \\left(\\frac{\\nu}{\\nu_t}\\right)^{\\alpha_1}
        \\left[1 - \\exp\\left(-\\left(\\frac{\\nu_t}{\\nu}\\right)^{\\alpha_1 - \\alpha_2}
        \\right)\\right] \\exp\\left(-\\frac{\\nu}{\\nu_{\\mathrm{cut}}}\\right)

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_agn_bol : float
        AGN bolometric luminosity in Lsun.
    radio_loudness : float
        log10(L_5GHz / L_B). Default 0 (radio-quiet). Range: [-2, 5].
    alpha1 : float
        Optically thin (steep) spectral slope. Default -0.75. Range: [-2, 0].
    alpha2 : float
        Optically thick (flat/inverted) spectral slope. Default -0.1.
        Range: [-1, 1].
    log_nu_t : float
        log10(transition frequency / Hz). Default 10.0. Range: [7, 13].
    log_nu_cut : float
        log10(synchrotron aging cutoff frequency / Hz). Default 13.0.
    nu_ref : float
        Reference frequency for L_5GHz normalization. Default 5 GHz.

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.

    Notes
    -----
    At frequencies well below nu_t the spectrum flattens to ~nu^alpha2
    (self-absorbed synchrotron). Above nu_t the spectrum steepens to ~nu^alpha1
    (optically thin synchrotron). The exponential cutoff at nu_cut models
    synchrotron aging losses.
    """
    nu = _C_AA / wavelength  # Hz

    # B-band luminosity density (same as radio_agn)
    _NU_B = 6.818e14  # Hz
    _BC_B = 5.15  # Hopkins+2007
    L_B = L_agn_bol * _LSUN / (_BC_B * _NU_B) / _LSUN  # Lsun/Hz

    # L_5GHz from radio-loudness definition
    L_5GHz = L_B * 10.0**radio_loudness  # Lsun/Hz

    nu_t = 10.0**log_nu_t
    nu_cut = 10.0**log_nu_cut

    # Double power-law (Martinez-Ramirez+2024 Eq. 9-10)
    # L_nu = L_5GHz * (nu/nu_t)^alpha1
    #        * [1 - exp(-(nu_t/nu)^(alpha1 - alpha2))]
    #        * exp(-nu/nu_cut)
    #
    # Normalization: evaluate the DPL shape at nu_ref=5 GHz and divide
    # so that L_nu(nu_ref) = L_5GHz exactly.
    def _dpl_shape(freq):
        ratio = freq / nu_t
        thick_thin = jnp.power(ratio, alpha1)
        turnover = 1.0 - jnp.exp(-jnp.power(nu_t / freq, alpha1 - alpha2))
        cutoff = jnp.exp(-freq / nu_cut)
        return thick_thin * turnover * cutoff

    shape = _dpl_shape(nu)
    shape_ref = _dpl_shape(nu_ref)
    # Guard against shape_ref ~ 0 (should not happen for physical params)
    shape_ref_safe = jnp.where(shape_ref > 0.0, shape_ref, 1.0)
    L_nu = L_5GHz * shape / shape_ref_safe

    # Only emit at radio frequencies (nu < 300 GHz => lambda > 1 mm = 1e7 A)
    radio_mask = wavelength > 1e7
    return jnp.where(radio_mask, L_nu, 0.0)


def radio_total_dpl(
    wavelength: jnp.ndarray,
    L_ir: float = 0.0,
    L_agn_bol: float = 0.0,
    q_ir: float = 2.64,
    alpha_sf: float = 0.8,
    radio_loudness: float = 0.0,
    alpha1: float = -0.75,
    alpha2: float = -0.1,
    log_nu_t: float = 10.0,
    log_nu_cut: float = 13.0,
    **_kwargs,
) -> jnp.ndarray:
    """Total radio emission: star-forming + AGN double power-law.

    Combines the FIR-radio-correlation star-forming component with the
    AGNfitter-rx double power-law AGN component.

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_ir : float
        Total IR luminosity (Lsun) for SF component.
    L_agn_bol : float
        AGN bolometric luminosity (Lsun) for AGN component.
    q_ir : float
        FIR-radio correlation parameter.
    alpha_sf : float
        SF synchrotron spectral index.
    radio_loudness : float
        AGN radio-loudness log10(L_5GHz/L_B).
    alpha1 : float
        Optically thin (steep) spectral slope.
    alpha2 : float
        Optically thick (flat/inverted) spectral slope.
    log_nu_t : float
        log10(transition frequency / Hz).
    log_nu_cut : float
        log10(synchrotron aging cutoff frequency / Hz).

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.
    """
    sf = radio_star_forming(wavelength, L_ir, q_ir, alpha_sf)
    agn = radio_agn_dpl(
        wavelength,
        L_agn_bol,
        radio_loudness,
        alpha1,
        alpha2,
        log_nu_t,
        log_nu_cut,
    )
    return sf + agn
