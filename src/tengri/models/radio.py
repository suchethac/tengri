"""Radio emission models (synchrotron + free-free).

Predicts radio SED from star formation rate (via FIR-radio correlation)
and AGN (via radio-loudness parameter).

The total radio luminosity has three components:
1. Star-forming synchrotron: from supernova remnants, scaled via FIRRC
2. Thermal free-free: bremsstrahlung from HII regions (Murphy+2011)
3. AGN: radio jets/lobes, parameterized by radio-loudness R

Three SFR physics models are available (select via ``sfr_mode`` in
``radio_total`` / ``radio_total_dpl``):

``"bell2003"``
    Fixed scalar q_IR (Bell 2003). No mass or redshift dependence.
    Backward-compatible default.

``"delvecchio2021"``
    Mass- and redshift-dependent FIRRC at 1.4 GHz (Delvecchio+2021, UV+FIR).
    Correlation parameters (q0, mass_slope, z_slope) are exposed as function
    arguments so they can later be treated as hierarchical prior hyperparameters
    in HierarchicalFitter.

``"mccheyne2022"``
    Mass- and redshift-dependent FIRRC calibrated at 150 MHz (McCheyne+2022).
    Same hierarchical-prior architecture as Delvecchio2021.

The FIRRC correlation parameters in the latter two models are fixed to their
literature best-fit values by default, but are architecturally free so they can
be promoted to Uniform priors in a hierarchical fit without touching this code.

All functions are pure JAX, JIT-compatible.

References
----------
- Condon 1992, ARA&A, 30, 575
- Bell 2003, ApJ, 586, 794 (FIR-radio correlation + synchrotron suppression)
- Delhaize et al. 2017, A&A, 602, A4 (q_IR redshift evolution)
- Delvecchio et al. 2021, A&A, 647, A123 (mass + redshift dependent FIRRC, 1.4 GHz)
- McCheyne et al. 2022, A&A (mass + redshift dependent FIRRC, 150 MHz)
- Mancuso et al. 2017 (synchrotron suppression calibration)
- Martinez-Ramirez et al. 2024, A&A (AGNfitter-rx double power-law)
- Giulietti et al. 2025 (arXiv:2503.20525, SEMPER; equations 4-5 adopted here)
"""

import jax.numpy as jnp

# Physical constants
_C_CGS = 2.99792458e10  # cm/s
_C_AA = 2.99792458e18  # Angstrom/s
_LSUN = 3.828e33  # erg/s
_JY = 1e-23  # erg/s/cm^2/Hz

# Bell+2003 synchrotron suppression threshold at 1.4 GHz reference.
# L0 = 3e28 erg/s/Hz converted to Lsun/Hz.
_L0_SYNCH_LSUN_HZ: float = 3.0e28 / _LSUN  # ≈ 7.84e-6 Lsun/Hz

# Wavelength boundary separating radio from IR (1 mm = 1e7 Angstrom = 300 GHz)
_RADIO_WAVE_MIN_AA: float = 1.0e7

# Murphy+2011 Eq. 11 free-free calibration constant [Lsun/Hz per (M☉/yr) at 1 GHz, Te=1e4 K]
# Derivation: SFR/(M☉/yr) = 4.6e-28 × (Te/1e4)^{-0.45} × (ν/GHz)^{0.1} × L_ν[erg/s/Hz]
# → L_ν[Lsun/Hz] = SFR × (Te/1e4)^{0.45} × (ν/GHz)^{-0.1} / (4.6e-28 × _LSUN)
# At 1.4 GHz, Te=1e4: 5.68e-7 × (1.4)^{-0.1} ≈ 5.49e-7 Lsun/Hz per M☉/yr
_C_FF_LSUN_HZ: float = 1.0 / (4.6e-28 * _LSUN)  # ≈ 5.68e-7

# Kennicutt+1998 IR-SFR calibration: L_IR [Lsun] → SFR [M☉/yr]
_SFR_IR_KENNICUTT: float = 1.73e10


def _synchrotron_suppression(L_ref: jnp.ndarray) -> jnp.ndarray:
    """Bell+2003 correction for low-SFR synchrotron suppression.

    Low-mass, low-SFR galaxies are less efficient synchrotron emitters due to
    cosmic-ray losses (Klein+1984; Chi+1990; Price+1992; Bell 2003). This
    correction smoothly suppresses emission below a threshold luminosity.

    .. math::

        L_{\\rm corr} = \\frac{L}{1 + (L_0 / L)^2}

    At L >> L0: L_corr ≈ L (unaffected).
    At L << L0: L_corr ≈ L^3 / L0^2 (quadratic suppression).

    Parameters
    ----------
    L_ref : array or float
        Luminosity density at the reference frequency in Lsun/Hz.

    Returns
    -------
    array or float
        Corrected luminosity density in Lsun/Hz.

    Notes
    -----
    L0 = 3e28 erg/s/Hz at 1.4 GHz (Mancuso+2017 / Bell+2003).
    The correction is applied at the reference frequency before spectral
    extrapolation so that the power-law shape is preserved.
    """
    L_safe = jnp.where(L_ref > 0.0, L_ref, 1.0e-40)
    return L_ref / (1.0 + (_L0_SYNCH_LSUN_HZ / L_safe) ** 2)


def radio_sfr_bell2003(
    wavelength: jnp.ndarray,
    L_ir: float,
    q_ir: float = 2.64,
    alpha_sf: float = 0.8,
    nu_ref: float = 1.4e9,
) -> jnp.ndarray:
    """Star-forming synchrotron via fixed scalar FIRRC (Bell 2003).

    This is the original Bell+2003 model with a constant q_IR.  It is
    equivalent to the former ``radio_star_forming`` function and preserved
    here for backward-compatibility and as the ``sfr_mode="bell2003"``
    option in ``radio_total``.

    .. math::

        L_\\nu(\\text{radio}) = \\frac{L_{\\rm IR}}{3.75 \\times 10^{12}
        \\times 10^{q_{\\rm IR}}} \\left(\\frac{\\nu}{\\nu_{\\rm ref}}
        \\right)^{-\\alpha}

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_ir : float
        Total infrared luminosity (8-1000 μm) in Lsun.
    q_ir : float
        FIR-radio correlation parameter. Default 2.64 (Bell 2003, z=0).
    alpha_sf : float
        Synchrotron spectral index (S_ν ∝ ν^{-α}). Default 0.8.
    nu_ref : float
        Reference frequency in Hz. Default 1.4 GHz.

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.
    """
    nu = _C_AA / wavelength
    L_ref = L_ir / (3.75e12 * 10.0**q_ir)  # Lsun/Hz at nu_ref
    L_nu = L_ref * (nu / nu_ref) ** (-alpha_sf)
    return jnp.where(wavelength > _RADIO_WAVE_MIN_AA, L_nu, 0.0)


# Backward-compatible alias
radio_star_forming = radio_sfr_bell2003


def radio_sfr_delvecchio2021(
    wavelength: jnp.ndarray,
    L_ir: float,
    log_mstar: float,
    redshift: float,
    q0: float = 2.743,
    mass_slope: float = 0.234,
    z_slope: float = -0.025,
    alpha_sf: float = 0.7,
    nu_ref: float = 1.4e9,
    apply_suppression: bool = True,
) -> jnp.ndarray:
    """Star-forming synchrotron via mass- and redshift-dependent FIRRC at 1.4 GHz.

    Implements the UV+FIR FIRRC from Delvecchio+2021 (Eq. 4 in SEMPER,
    arXiv:2503.20525), calibrated on >400 000 SFGs in COSMOS at 0.1 < z < 4:

    .. math::

        q(M_\\star, z) = q_0 \\, (1+z)^{z_{\\rm slope}}
        - (\\log M_\\star/M_\\odot - 10) \\times m_{\\rm slope}

    The three FIRRC parameters (``q0``, ``mass_slope``, ``z_slope``) are
    exposed as function arguments so they can be promoted to hierarchical
    prior hyperparameters in ``HierarchicalFitter`` without modifying this
    code.  Their defaults are the Delvecchio+2021 best-fit values.

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_ir : float
        Total infrared luminosity (8-1000 μm) in Lsun.
    log_mstar : float
        log10(M★ / M⊙). Typical range [8, 12].
    redshift : float
        Galaxy redshift. Valid range [0.1, 4] per Delvecchio+2021.
    q0 : float
        FIRRC normalization at log(M★)=10, z=0. Default 2.743.
        Hierarchical prior suggestion: Uniform(2.4, 3.1).
    mass_slope : float
        ∂q / ∂log(M★). Default 0.234 (negative: massive → lower q → more radio).
        Hierarchical prior suggestion: Uniform(0.0, 0.5).
    z_slope : float
        Power-law exponent on (1+z). Default -0.025 (slight decline with z).
        Hierarchical prior suggestion: Uniform(-0.2, 0.05).
    alpha_sf : float
        Synchrotron spectral index. Default 0.7 (consensus: Novak+2017, SEMPER).
    nu_ref : float
        Reference frequency in Hz. Default 1.4 GHz (calibration frequency).
    apply_suppression : bool
        Apply Bell+2003 synchrotron suppression for low-SFR galaxies.
        Default True.

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.

    Notes
    -----
    q decreases with increasing M★ (radio-brighter per unit IR for massive
    galaxies), consistent with stronger magnetic fields and denser ISM.
    The z_slope = -0.025 implies much more modest redshift evolution than the
    older Delhaize+2017 value of -0.15.
    """
    nu = _C_AA / wavelength

    # Mass- and redshift-dependent q_IR (Delvecchio+2021 UV+FIR, SEMPER Eq. 4)
    q_ir = q0 * (1.0 + redshift) ** z_slope - (log_mstar - 10.0) * mass_slope

    # L at 1.4 GHz reference from FIRRC definition
    L_ref = L_ir / (3.75e12 * 10.0**q_ir)  # Lsun/Hz

    # Optional Bell+2003 synchrotron suppression (low-SFR galaxies)
    L_ref = jnp.where(apply_suppression, _synchrotron_suppression(L_ref), L_ref)

    L_nu = L_ref * (nu / nu_ref) ** (-alpha_sf)
    return jnp.where(wavelength > _RADIO_WAVE_MIN_AA, L_nu, 0.0)


def radio_sfr_mccheyne2022(
    wavelength: jnp.ndarray,
    L_ir: float,
    log_mstar: float,
    redshift: float,
    q0: float = 1.98,
    mass_slope: float = -0.22,
    z_slope: float = 0.02,
    alpha_sf: float = 0.7,
    nu_ref: float = 1.5e8,
    apply_suppression: bool = True,
) -> jnp.ndarray:
    """Star-forming synchrotron via mass- and redshift-dependent FIRRC at 150 MHz.

    Implements the FIRRC from McCheyne+2022 (Eq. 5 in SEMPER, arXiv:2503.20525),
    derived from LOFAR 150 MHz observations of a mass-complete sample in
    ELAIS-N1 at z < 1:

    .. math::

        q(z, M_\\star) = q_0 \\, (1+z)^{z_{\\rm slope}}
        + m_{\\rm slope} \\times (\\log M_\\star/M_\\odot - 10)

    This relation is natively calibrated at 150 MHz and is preferred over
    spectral-index rescaling from 1.4 GHz for low-frequency data. For z > 1,
    the Delvecchio+2021 model (rescaled to 150 MHz) is better constrained.

    The three FIRRC parameters (``q0``, ``mass_slope``, ``z_slope``) are
    exposed as hierarchical prior parameters; their defaults are the
    McCheyne+2022 best-fit values.

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_ir : float
        Total infrared luminosity (8-1000 μm) in Lsun.
    log_mstar : float
        log10(M★ / M⊙). Typical range [10.05, 11.4] per McCheyne+2022.
    redshift : float
        Galaxy redshift. Valid range [0, 1] per McCheyne+2022.
    q0 : float
        FIRRC normalization at log(M★)=10, z=0. Default 1.98.
        Hierarchical prior suggestion: Uniform(1.5, 2.5).
    mass_slope : float
        ∂q / ∂log(M★). Default -0.22 (negative sign convention differs from
        Delvecchio: here higher M★ → lower q directly via negative coefficient).
        Hierarchical prior suggestion: Uniform(-0.5, 0.0).
    z_slope : float
        Power-law exponent on (1+z). Default 0.02 (nearly no z evolution).
        Hierarchical prior suggestion: Uniform(-0.1, 0.2).
    alpha_sf : float
        Synchrotron spectral index. Default 0.7.
    nu_ref : float
        Reference frequency in Hz. Default 150 MHz (calibration frequency).
    apply_suppression : bool
        Apply Bell+2003 synchrotron suppression. Default True.

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.

    Notes
    -----
    McCheyne+2022 reports a steeper mass dependence than Delvecchio+2021.
    The discrepancy is reconciled when using α = -0.59 instead of -0.7 for
    spectral index conversion between 1.4 GHz and 150 MHz.
    """
    nu = _C_AA / wavelength

    # Mass- and redshift-dependent q at 150 MHz (McCheyne+2022, SEMPER Eq. 5)
    q_ir = q0 * (1.0 + redshift) ** z_slope + mass_slope * (log_mstar - 10.0)

    # L at 150 MHz reference from FIRRC definition
    L_ref = L_ir / (3.75e12 * 10.0**q_ir)  # Lsun/Hz

    # Optional Bell+2003 synchrotron suppression
    L_ref = jnp.where(apply_suppression, _synchrotron_suppression(L_ref), L_ref)

    L_nu = L_ref * (nu / nu_ref) ** (-alpha_sf)
    return jnp.where(wavelength > _RADIO_WAVE_MIN_AA, L_nu, 0.0)


def radio_freefree(
    wavelength: jnp.ndarray,
    L_ir: float,
    T_e: float = 1e4,
    alpha_ff: float = -0.1,
) -> jnp.ndarray:
    """Thermal free-free (bremsstrahlung) emission from HII regions.

    Traces instantaneous SFR via the Kennicutt+1998 IR calibration and the
    Murphy+2011 radio-SFR relation (Eq. 11).  At 1.4 GHz a typical star-forming
    galaxy contributes ~5–15% of its total radio flux from free-free, depending
    on the FIRRC calibration used for the synchrotron component.

    .. math::

        L_\\nu^{\\rm ff} = C_{\\rm ff} \\, (T_e / 10^4\\,{\\rm K})^{0.45}
        \\left(\\frac{\\nu}{\\rm GHz}\\right)^{\\alpha_{\\rm ff}}
        \\frac{L_{\\rm IR}}{L_{\\rm IR,\\odot}}

    where :math:`C_{\\rm ff} = 1 / (4.6 \\times 10^{-28} \\, L_\\odot)
    \\approx 5.68 \\times 10^{-7}` Lsun/Hz per M☉/yr at 1 GHz and
    :math:`L_{\\rm IR,\\odot} = 1.73 \\times 10^{10}` Lsun (Kennicutt+1998).

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_ir : float
        Total infrared luminosity (8-1000 μm) in Lsun.
    T_e : float
        Electron temperature in K. Default 1e4.
        Prior suggestion: LogUniform(5e3, 2e4).
    alpha_ff : float
        Free-free spectral index (L_ν ∝ ν^{α_ff}). Default −0.1 (nearly flat).
        Prior suggestion: Uniform(−0.15, 0.0).

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.

    Notes
    -----
    Calibration check: at 1.4 GHz, Te=1e4 K, L_IR=1e10 Lsun (SFR≈0.58 M☉/yr):
    L_ff ≈ 5.49e-7 × 0.58 ≈ 3.2e-7 Lsun/Hz (Murphy+2011 Table 1 consistent).

    References
    ----------
    - Murphy et al. 2011, ApJ, 737, 67 (Eq. 11)
    - Condon 1992, ARA&A, 30, 575
    - Kennicutt 1998, ARA&A, 36, 189
    """
    nu = _C_AA / wavelength  # Hz
    nu_ghz = nu / 1.0e9  # GHz
    sfr = L_ir / _SFR_IR_KENNICUTT  # M☉/yr
    # Murphy+2011 Eq. 11 inverted; (T_e/1e4)^0.45 factor from ionized gas physics
    L_nu = _C_FF_LSUN_HZ * (T_e / 1.0e4) ** 0.45 * nu_ghz**alpha_ff * sfr
    return jnp.where(wavelength > _RADIO_WAVE_MIN_AA, L_nu, 0.0)


def _dispatch_sfr(
    wavelength: jnp.ndarray,
    L_ir: float,
    sfr_mode: str,
    q_ir: float,
    alpha_sf: float,
    log_mstar: float,
    redshift: float,
    q0: float | None,
    mass_slope: float | None,
    z_slope: float | None,
    apply_suppression: bool,
) -> jnp.ndarray:
    """Dispatch SFR synchrotron component by mode (private helper).

    Eliminates duplicated ``if/elif sfr_mode`` blocks in ``radio_total`` and
    ``radio_total_dpl``.  Only the FIRRC normalization (Layer 1) differs between
    modes; the spectral shape machinery (Layer 3) is shared inside each wrapper.

    Parameters
    ----------
    wavelength : array
        Wavelength in Angstrom.
    L_ir : float
        IR luminosity in Lsun.
    sfr_mode : str
        One of ``"bell2003"``, ``"delvecchio2021"``, ``"mccheyne2022"``.
    q_ir : float
        Fixed q_IR (bell2003 mode only).
    alpha_sf : float
        Synchrotron spectral index.
    log_mstar : float
        log10(M★/M⊙).
    redshift : float
        Galaxy redshift.
    q0 : float or None
        FIRRC normalization override (None = use mode default).
    mass_slope : float or None
        Mass slope override.
    z_slope : float or None
        Redshift slope override.
    apply_suppression : bool
        Apply Bell+2003 synchrotron suppression.

    Returns
    -------
    array
        L_nu in Lsun/Hz.
    """
    if sfr_mode == "bell2003":
        return radio_sfr_bell2003(wavelength, L_ir, q_ir, alpha_sf)
    elif sfr_mode == "delvecchio2021":
        kw = {}
        if q0 is not None:
            kw["q0"] = q0
        if mass_slope is not None:
            kw["mass_slope"] = mass_slope
        if z_slope is not None:
            kw["z_slope"] = z_slope
        return radio_sfr_delvecchio2021(
            wavelength, L_ir, log_mstar, redshift, apply_suppression=apply_suppression, **kw
        )
    elif sfr_mode == "mccheyne2022":
        kw = {}
        if q0 is not None:
            kw["q0"] = q0
        if mass_slope is not None:
            kw["mass_slope"] = mass_slope
        if z_slope is not None:
            kw["z_slope"] = z_slope
        return radio_sfr_mccheyne2022(
            wavelength, L_ir, log_mstar, redshift, apply_suppression=apply_suppression, **kw
        )
    else:
        raise ValueError(
            f"Unknown sfr_mode {sfr_mode!r}. "
            "Choose 'bell2003', 'delvecchio2021', or 'mccheyne2022'."
        )


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

    return jnp.where(wavelength > _RADIO_WAVE_MIN_AA, L_nu, 0.0)


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

    return jnp.where(wavelength > _RADIO_WAVE_MIN_AA, L_nu, 0.0)


def radio_total(
    wavelength: jnp.ndarray,
    L_ir: float = 0.0,
    L_agn_bol: float = 0.0,
    q_ir: float = 2.64,
    alpha_sf: float = 0.8,
    radio_loudness: float = 0.0,
    alpha_agn: float = 0.7,
    sfr_mode: str = "bell2003",
    log_mstar: float = 10.0,
    redshift: float = 0.0,
    q0: float | None = None,
    mass_slope: float | None = None,
    z_slope: float | None = None,
    apply_suppression: bool = True,
    include_freefree: bool = False,
    T_e: float = 1e4,
    alpha_ff: float = -0.1,
    **_kwargs,
) -> jnp.ndarray:
    """Total radio emission (star-forming synchrotron + optional free-free + AGN power-law).

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_ir : float
        Total IR luminosity (Lsun) for SF component.
    L_agn_bol : float
        AGN bolometric luminosity (Lsun) for AGN component.
    q_ir : float
        FIR-radio correlation parameter (bell2003 mode only).
    alpha_sf : float
        SF synchrotron spectral index. Default 0.8 in bell2003 mode; the
        Delvecchio/McCheyne modes default to 0.7 internally.
    radio_loudness : float
        AGN radio-loudness log10(L_5GHz/L_B).
    alpha_agn : float
        AGN radio spectral index.
    sfr_mode : str
        Which SFR physics model to use. One of:
        - ``"bell2003"`` (default): fixed q_IR, no mass/z dependence.
        - ``"delvecchio2021"``: mass+z dependent FIRRC at 1.4 GHz.
        - ``"mccheyne2022"``: mass+z dependent FIRRC at 150 MHz.
    log_mstar : float
        log10(M★/M⊙). Used by delvecchio2021 and mccheyne2022 modes.
    redshift : float
        Galaxy redshift. Used by delvecchio2021 and mccheyne2022 modes.
    q0 : float or None
        FIRRC normalization override. None uses the mode's literature default.
    mass_slope : float or None
        Mass slope override. None uses the mode's literature default.
    z_slope : float or None
        Redshift slope override. None uses the mode's literature default.
    apply_suppression : bool
        Apply Bell+2003 synchrotron suppression (delvecchio/mccheyne modes).
    include_freefree : bool
        Add thermal free-free component (Murphy+2011). Default False to preserve
        backward-compatible behavior.
    T_e : float
        Electron temperature [K] for free-free component. Default 1e4.
    alpha_ff : float
        Free-free spectral index (L_ν ∝ ν^{α}). Default -0.1.

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.
    """
    sf = _dispatch_sfr(
        wavelength,
        L_ir,
        sfr_mode,
        q_ir,
        alpha_sf,
        log_mstar,
        redshift,
        q0,
        mass_slope,
        z_slope,
        apply_suppression,
    )
    agn = radio_agn(wavelength, L_agn_bol, radio_loudness, alpha_agn)
    ff = radio_freefree(wavelength, L_ir, T_e, alpha_ff) if include_freefree else 0.0
    return sf + ff + agn


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
    sfr_mode: str = "bell2003",
    log_mstar: float = 10.0,
    redshift: float = 0.0,
    q0: float | None = None,
    mass_slope: float | None = None,
    z_slope: float | None = None,
    apply_suppression: bool = True,
    include_freefree: bool = False,
    T_e: float = 1e4,
    alpha_ff: float = -0.1,
    **_kwargs,
) -> jnp.ndarray:
    """Total radio emission: star-forming + optional free-free + AGN double power-law.

    Combines the SFR component (dispatched by ``sfr_mode``) with the optional
    thermal free-free component and the AGNfitter-rx double power-law AGN component.

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_ir : float
        Total IR luminosity (Lsun) for SF component.
    L_agn_bol : float
        AGN bolometric luminosity (Lsun) for AGN component.
    q_ir : float
        FIR-radio correlation parameter (bell2003 mode only).
    alpha_sf : float
        SF synchrotron spectral index.
    radio_loudness : float
        AGN radio-loudness log10(L_5GHz/L_B).
    alpha1 : float
        Optically thin spectral slope.
    alpha2 : float
        Optically thick spectral slope.
    log_nu_t : float
        log10(transition frequency / Hz).
    log_nu_cut : float
        log10(synchrotron aging cutoff frequency / Hz).
    sfr_mode : str
        SFR physics model. See ``radio_total`` for options.
    log_mstar : float
        log10(M★/M⊙). Used by delvecchio2021 and mccheyne2022 modes.
    redshift : float
        Galaxy redshift. Used by delvecchio2021 and mccheyne2022 modes.
    q0 : float or None
        FIRRC normalization override.
    mass_slope : float or None
        Mass slope override.
    z_slope : float or None
        Redshift slope override.
    apply_suppression : bool
        Apply Bell+2003 synchrotron suppression.
    include_freefree : bool
        Add thermal free-free component (Murphy+2011). Default False.
    T_e : float
        Electron temperature [K] for free-free component. Default 1e4.
    alpha_ff : float
        Free-free spectral index. Default -0.1.

    Returns
    -------
    array (n_wave,)
        L_nu in Lsun/Hz.
    """
    sf = _dispatch_sfr(
        wavelength,
        L_ir,
        sfr_mode,
        q_ir,
        alpha_sf,
        log_mstar,
        redshift,
        q0,
        mass_slope,
        z_slope,
        apply_suppression,
    )
    agn = radio_agn_dpl(
        wavelength, L_agn_bol, radio_loudness, alpha1, alpha2, log_nu_t, log_nu_cut
    )
    ff = radio_freefree(wavelength, L_ir, T_e, alpha_ff) if include_freefree else 0.0
    return sf + ff + agn


def radio_components(
    wavelength: jnp.ndarray,
    L_ir: float = 0.0,
    L_agn_bol: float = 0.0,
    q_ir: float = 2.64,
    alpha_sf: float = 0.8,
    radio_loudness: float = 0.0,
    alpha_agn: float = 0.7,
    sfr_mode: str = "bell2003",
    log_mstar: float = 10.0,
    redshift: float = 0.0,
    q0: float | None = None,
    mass_slope: float | None = None,
    z_slope: float | None = None,
    apply_suppression: bool = True,
    include_freefree: bool = True,
    T_e: float = 1e4,
    alpha_ff: float = -0.1,
    **_kwargs,
) -> dict:
    """Decompose total radio emission into physical components.

    Returns a dict with individual component SEDs and their sum. Useful for
    visualizing the relative contribution of synchrotron, free-free, and AGN,
    and for computing thermal fractions (f_ff = freefree / total).

    Parameters
    ----------
    wavelength : array (n_wave,)
        Wavelength in Angstrom.
    L_ir : float
        IR luminosity (Lsun).
    L_agn_bol : float
        AGN bolometric luminosity (Lsun).
    q_ir : float
        FIR-radio q_IR (bell2003 mode only).
    alpha_sf : float
        Synchrotron spectral index.
    radio_loudness : float
        AGN radio-loudness log10(L_5GHz/L_B).
    alpha_agn : float
        AGN spectral index.
    sfr_mode : str
        SFR physics mode. See ``radio_total`` for options.
    log_mstar : float
        log10(M★/M⊙).
    redshift : float
        Galaxy redshift.
    q0, mass_slope, z_slope : float or None
        FIRRC parameter overrides.
    apply_suppression : bool
        Apply Bell+2003 synchrotron suppression.
    include_freefree : bool
        Include free-free component. Default True (diagnostic function).
    T_e : float
        Electron temperature for free-free. Default 1e4 K.
    alpha_ff : float
        Free-free spectral index. Default -0.1.

    Returns
    -------
    dict with keys:
        ``"synchrotron"`` : array (n_wave,) — SFR synchrotron L_nu [Lsun/Hz]
        ``"freefree"`` : array (n_wave,) — thermal free-free L_nu [Lsun/Hz]
        ``"agn"`` : array (n_wave,) — AGN radio L_nu [Lsun/Hz]
        ``"total"`` : array (n_wave,) — sum of above [Lsun/Hz]
    """
    synchrotron = _dispatch_sfr(
        wavelength,
        L_ir,
        sfr_mode,
        q_ir,
        alpha_sf,
        log_mstar,
        redshift,
        q0,
        mass_slope,
        z_slope,
        apply_suppression,
    )
    agn = radio_agn(wavelength, L_agn_bol, radio_loudness, alpha_agn)
    ff = (
        radio_freefree(wavelength, L_ir, T_e, alpha_ff)
        if include_freefree
        else jnp.zeros_like(wavelength)
    )
    total = synchrotron + ff + agn
    return {"synchrotron": synchrotron, "freefree": ff, "agn": agn, "total": total}
