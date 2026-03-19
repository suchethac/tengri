r"""QSOgen quasar SED model (Temple, Hewett & Banerji 2021).

Empirical quasar SED producing the characteristic "v-shaped" spectrum:
blue power-law from the accretion disc falling toward the optical, then
rising again in the IR from hot dust.  Four additive components in
f_nu space::

        Emission lines (Lya, CIV, MgII, Ha, Hb...)
             |   |    |       |     |
    f_nu  \     /\  /\   /\      /\    /\
           \   /  \/  \ /  \    /  \  /  \     /  Hot dust BB
            \ /       \/    \  /    \/    \   /   (T~1240K)
             X  <-break->    \/           \ /
            / \              power-law     X
           /   \             continuum    / \
          EUV    UV        Optical       NIR      MIR
         <1200A  1200-3880A             1-2um    2-5um

Parameters
----------
=========== ======= =================================== ==========================================
Parameter   Default Physical meaning                     Effect on SED
=========== ======= =================================== ==========================================
agn_plslp1  -0.349  Blue/UV slope (f_nu ~ nu^alpha)      Steeper -> more UV flux, bluer u-g
agn_plslp2  +0.593  Red/optical slope                    Steeper -> redder optical continuum
agn_plbrk   3880 A  Break wavelength                     Shifts the "valley" of the v-shape
agn_tbb     1240 K  Hot dust temperature                 Higher T -> dust peak shifts bluer (Wien)
agn_bbnorm  3.96    Dust BB / continuum at 2 um          Higher -> stronger MIR excess
agn_emline  1.0     Emission line strength multiplier    Scales all line EWs
agn_ebv     0.0     SMC-like dust reddening E(B-V)       Reddens UV, suppresses blue flux
=========== ======= =================================== ==========================================

Recommended Inference Strategy
------------------------------
When fitting broadband photometry of AGN/quasars:

1. **Always free:** ``agn_log_lbol`` (or ``agn_frac``) -- the AGN
   contribution strength.  This is the minimum viable AGN model.
2. **Optionally free:** ``agn_ebv`` -- AGN reddening is the strongest
   shape variation in real quasars and is degenerate with galaxy dust.
3. **Optionally free:** ``agn_plslp1`` -- UV slope varies significantly
   between quasars (alpha ~ -1.5 to +0.5).
4. **Keep fixed:** ``agn_plslp2``, ``agn_plbrk``, ``agn_tbb``,
   ``agn_bbnorm`` are well-constrained by the SDSS composite and add
   more degeneracies than information unless you have MIR data.

For a typical galaxy+AGN decomposition with UV-to-NIR photometry,
start with 1--2 free AGN parameters and add more only if the data
warrant it (chi-squared still poor).

If WISE W1/W2 data are available, the hot dust BB becomes strongly
constraining and ``agn_bbnorm`` could be freed.

Diagnostic colours for AGN contamination:

- **W1-W2 > 0.8** (Stern+12) -- hot dust makes W2 brighter than W1
- **u-g bluer than expected** -- UV power-law excess
- **NUV-r very blue** -- strong UV continuum + emission lines
- **Rest 1-3 um inflection** -- stellar SED drops but AGN dust rises

Baldwin Effect
--------------
Emission line EWs scale inversely with luminosity:

    EW(L) = EW_ref * (L / L_ref)^{-0.2}

Brighter quasars have weaker lines relative to continuum.  Physically,
the more luminous the central engine, the more the continuum "drowns
out" the line-emitting gas.

Differences from the Original QSOgen
-------------------------------------
===================== ================================== ====================================
Feature               Original (Python/numpy)             This JAX version
===================== ================================== ====================================
Power-law transitions Hard np.where                       Smooth sigmoid (differentiable)
Emission lines        4 empirical template interpolation  18 analytic Gaussians (VdB+01)
Baldwin effect slope  0.183                               0.2
Host galaxy template  S0 SWIRE included                   Not included (handled by Model)
IGM absorption        Becker+13 tau_eff                   Handled by diffsed IGM module
Autodiff              No                                  Yes (JAX JIT-compatible)
===================== ================================== ====================================

References
----------
- Temple, Hewett & Banerji 2021, MNRAS, 508, 737
- Vanden Berk et al. 2001, AJ, 122, 549 (emission line EWs)
- Gordon et al. 2003, ApJ, 594, 279 (SMC extinction)
- Stern et al. 2012, ApJ, 753, 30 (W1-W2 AGN selection)
"""

import jax
import jax.numpy as jnp

from diffsed.models.agn.unified import register_agn_model
from diffsed.models.dust.attenuation import smc as smc_curve

# ===================================================================
# Physical constants (CGS)
# ===================================================================

_H_PLANCK = 6.62607015e-27  # Planck constant [erg s]
_K_BOLTZ = 1.380649e-16  # Boltzmann constant [erg K^-1]
_C_LIGHT = 2.99792458e10  # Speed of light [cm s^-1]
_C_LIGHT_KMS = 2.99792458e5  # Speed of light [km/s]
_LSUN_ERG = 3.828e33  # Solar luminosity [erg s^-1]
_ANGSTROM_CM = 1e-8  # Angstrom -> cm

# Normalization wavelength
_LAMBDA_NORM = 5500.0  # Angstrom

# Reference wavelength for hot dust anchoring
_LAMBDA_BB_ANCHOR = 20000.0  # 2 um in Angstrom

# ===================================================================
# Default parameters (Temple+2021 Table 3)
# ===================================================================

_DEFAULT_PLSLP1 = -0.349  # Blue power-law slope (f_nu ~ nu^alpha)
_DEFAULT_PLSLP2 = 0.593  # Red power-law slope
_DEFAULT_PLBRK = 3880.0  # Break wavelength [Angstrom]
_DEFAULT_TBB = 1240.0  # Hot dust temperature [K]
_DEFAULT_BBNORM = 3.96  # Hot dust normalization
_DEFAULT_EMLINE_SCALE = 1.0  # Emission line strength multiplier
_DEFAULT_EBV = 0.0  # E(B-V) dust reddening

# Baldwin effect slope (negative = brighter quasars have weaker lines)
_BALDWIN_SLOPE = -0.2

# Reference log bolometric luminosity for Baldwin effect normalization
_LOG_LBOL_REF = 45.0  # erg/s

# Sigmoid transition half-width in log-wavelength space
_SIGMOID_WIDTH = 0.02  # dex (~5% in wavelength)


# ===================================================================
# Emission line table
# ===================================================================

# Each row: [rest wavelength (A), equivalent width (A), FWHM (km/s)]
# EWs from Vanden Berk et al. (2001) SDSS composite.
# Broad lines: FWHM ~ 5000 km/s; narrow lines: FWHM ~ 500 km/s.
_EMISSION_LINES = jnp.array(
    [
        # Broad lines
        [1034.0, 3.0, 5000.0],  # OVI 1034
        [1216.0, 90.0, 5000.0],  # Ly-alpha
        [1240.0, 8.0, 5000.0],  # NV 1240
        [1397.0, 5.0, 5000.0],  # SiIV 1397
        [1549.0, 25.0, 5000.0],  # CIV 1549
        [1909.0, 20.0, 5000.0],  # CIII] 1909
        [2800.0, 30.0, 5000.0],  # MgII 2800
        [4340.0, 3.0, 5000.0],  # H-gamma
        [4861.0, 15.0, 5000.0],  # H-beta
        [6563.0, 40.0, 5000.0],  # H-alpha
        [18750.0, 5.0, 5000.0],  # Pa-alpha
        # Narrow lines
        [3727.0, 4.0, 500.0],  # [OII] 3727
        [3869.0, 2.0, 500.0],  # [NeIII] 3869
        [4686.0, 2.0, 500.0],  # HeII 4686
        [5007.0, 5.0, 500.0],  # [OIII] 5007
        [6583.0, 3.0, 500.0],  # [NII] 6583
        [6717.0, 1.5, 500.0],  # [SII] 6717
        [6731.0, 1.5, 500.0],  # [SII] 6731
    ]
)

_LINE_WAVELENGTHS = _EMISSION_LINES[:, 0]
_LINE_EWS = _EMISSION_LINES[:, 1]
_LINE_FWHMS = _EMISSION_LINES[:, 2]


# ===================================================================
# Helper functions
# ===================================================================


def _wavelength_to_nu(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength (Angstrom) to frequency (Hz)."""
    return _C_LIGHT / (wavelength * _ANGSTROM_CM)


def _planck_blambda(
    wavelength: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Planck function B_lambda(T) [erg s^-1 cm^-2 A^-1 sr^-1].

    Parameters
    ----------
    wavelength : array
        Wavelength [Angstrom].
    temperature : float
        Temperature [K].

    Returns
    -------
    array
        B_lambda(T) in per-Angstrom units.
    """
    t_safe = jnp.maximum(temperature, 1.0)
    lam_cm = wavelength * _ANGSTROM_CM
    x = _H_PLANCK * _C_LIGHT / (lam_cm * _K_BOLTZ * t_safe)
    x_clip = jnp.clip(x, 0.0, 500.0)
    prefactor = 2.0 * _H_PLANCK * _C_LIGHT**2 / lam_cm**5
    # Return in per-Angstrom: divide by 1e8 (cm -> A)
    return prefactor / (jnp.exp(x_clip) - 1.0) * _ANGSTROM_CM


def _broken_powerlaw_continuum(
    wavelength: jnp.ndarray,
    plslp1: float,
    plslp2: float,
    plbrk: float,
    plstep: float = -1.0,
    plbrk3: float = 1200.0,
) -> jnp.ndarray:
    """Broken power-law continuum in f_nu (matching original qsogen exactly).

    Works in f_nu space: f_nu = const * wavelength^sl, where sl = -alpha_nu.
    Three segments with smooth sigmoid transitions (differentiable):
    - lambda > plbrk: sl2 = -plslp2 (red/optical)
    - plbrk3 < lambda < plbrk: sl1 = -plslp1 (blue/UV)
    - lambda < plbrk3: sl3 = sl1 - plstep (EUV, steepened)

    Normalized to 1.0 at ``_LAMBDA_NORM`` (5500 A).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    plslp1 : float
        Blue f_nu spectral index. f_nu ~ nu^plslp1 = wavelength^(-plslp1).
    plslp2 : float
        Red f_nu spectral index.
    plbrk : float
        Break wavelength between blue and red [Angstrom].
    plstep : float
        Extra steepening in the EUV below plbrk3. Default -1.0.
    plbrk3 : float
        EUV break wavelength [Angstrom]. Default 1200.

    Returns
    -------
    array, shape (n_wave,)
        f_nu (arbitrary units, normalized to 1.0 at 5500 A).
    """
    # f_nu slopes in wavelength space (matching original qsosed.py exactly)
    sl1 = -plslp1
    sl2 = -plslp2
    sl3 = sl1 - plstep

    # Normalization constants for continuity
    const2 = 1.0 / (_LAMBDA_NORM**sl2)
    const1 = const2 * (plbrk**sl2) / (plbrk**sl1)
    const3 = const1 * (plbrk3**sl1) / (plbrk3**sl3)

    # Power-law segments
    f_red = const2 * wavelength**sl2
    f_blue = const1 * wavelength**sl1
    f_euv = const3 * wavelength**sl3

    # Differentiable sigmoid transitions (replaces original np.where)
    steepness = 1.0 / _SIGMOID_WIDTH
    log_wave = jnp.log10(wavelength)
    log_brk = jnp.log10(jnp.maximum(plbrk, 100.0))
    log_brk3 = jnp.log10(jnp.maximum(plbrk3, 100.0))

    w_red = jax.nn.sigmoid(steepness * (log_wave - log_brk))
    w_euv = jax.nn.sigmoid(-steepness * (log_wave - log_brk3))
    w_blue = jnp.clip(1.0 - w_red - w_euv, 0.0, 1.0)

    f_nu = w_euv * f_euv + w_blue * f_blue + w_red * f_red

    # Normalize to 1.0 at 5500 A
    log_norm = jnp.log10(_LAMBDA_NORM)
    w_red_n = jax.nn.sigmoid(steepness * (log_norm - log_brk))
    w_euv_n = jax.nn.sigmoid(-steepness * (log_norm - log_brk3))
    w_blue_n = jnp.clip(1.0 - w_red_n - w_euv_n, 0.0, 1.0)

    f_norm = (
        w_euv_n * const3 * _LAMBDA_NORM**sl3
        + w_blue_n * const1 * _LAMBDA_NORM**sl1
        + w_red_n * const2 * _LAMBDA_NORM**sl2
    )

    return f_nu / jnp.maximum(f_norm, 1e-30)


def _hot_dust_blackbody(
    wavelength: jnp.ndarray,
    continuum_flam: jnp.ndarray,
    tbb: float,
    bbnorm: float,
) -> jnp.ndarray:
    """Hot dust blackbody component.

    Adds a blackbody at temperature ``tbb`` normalized so that at 2 um
    the blackbody flux equals ``bbnorm`` times the power-law continuum flux.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    continuum_flam : array, shape (n_wave,)
        Power-law continuum f_lambda (normalized at 5500 A).
    tbb : float
        Hot dust temperature [K].
    bbnorm : float
        Normalization: ratio of blackbody to continuum at 2 um.

    Returns
    -------
    array, shape (n_wave,)
        Hot dust f_lambda contribution (same units as continuum_flam).
    """
    # B_nu as function of wavelength (matching original bb() exactly):
    #   bb(T, wav) = wav^{-3} / (exp(hc/kT*wav) - 1)
    hc_over_k = 1.43877735e8  # h*c/k_B in Kelvin*Angstrom
    x = hc_over_k / (tbb * jnp.maximum(wavelength, 1.0))
    bb_fnu = wavelength ** (-3.0) / (jnp.exp(jnp.clip(x, 0.0, 500.0)) - 1.0)

    # Normalize: bb_flux(anchor) = bbnorm (ABSOLUTE f_nu, not relative)
    x_anchor = hc_over_k / (tbb * _LAMBDA_BB_ANCHOR)
    bb_anchor = _LAMBDA_BB_ANCHOR ** (-3.0) / (jnp.exp(jnp.clip(x_anchor, 0.0, 500.0)) - 1.0)
    cmult = bbnorm / jnp.maximum(bb_anchor, 1e-60)

    return cmult * bb_fnu


def _balmer_continuum(
    wavelength: jnp.ndarray,
    continuum_flam: jnp.ndarray,
    bcnorm: float = 1.0,
    tbc: float = 15000.0,
    taube: float = 1.0,
    wavbe: float = 3646.0,
) -> jnp.ndarray:
    """Balmer continuum emission (Grandi 1982).

    Adds hydrogen recombination continuum below the Balmer edge at 3646 A.
    Matches the original qsogen prescription: B_nu(T_BC) * (1 - exp(-tau))
    where tau = tau_BE * (nu_BE / nu)^3.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    continuum_flam : array, shape (n_wave,)
        Power-law continuum (for normalization reference at 3000 A).
    bcnorm : float
        Balmer continuum strength relative to power-law at 3000 A.
        Default 1.0 (from original qsogen).
    tbc : float
        BC temperature [K]. Default 15000.
    taube : float
        Optical depth at Balmer edge. Default 1.0.
    wavbe : float
        Balmer edge wavelength [Angstrom]. Default 3646.

    Returns
    -------
    array, shape (n_wave,)
        Balmer continuum f_lambda contribution.
    """
    # B_nu as function of wavelength (matching original bb() convention)
    # bb(T, wav) = wav^{-3} / (exp(hc/kT*wav) - 1)
    hc_over_k = 1.43877735e8  # h*c/k_B in Kelvin * Angstrom
    x = hc_over_k / (tbc * jnp.maximum(wavelength, 1.0))
    x_clip = jnp.clip(x, 0.0, 500.0)
    b_nu_wav = wavelength ** (-3.0) / (jnp.exp(x_clip) - 1.0)

    # Optical depth: tau(nu) = taube * (nu_BE / nu)^3 = taube * (wav / wavbe)^3
    tau = taube * (wavelength / wavbe) ** 3
    tau_clip = jnp.clip(tau, 0.0, 50.0)
    absorption = 1.0 - jnp.exp(-tau_clip)

    bc_flux = b_nu_wav * absorption

    # Normalize at 3000 A relative to continuum
    wnorm = 3000.0
    log_wave = jnp.log10(wavelength)
    log_wnorm = jnp.log10(wnorm)
    cont_at_3000 = jnp.interp(log_wnorm, log_wave, continuum_flam)
    cont_at_3000 = jnp.maximum(cont_at_3000, 1e-30)

    # BC value at wnorm for normalization
    x_norm = hc_over_k / (tbc * wnorm)
    b_norm = wnorm ** (-3.0) / (jnp.exp(jnp.clip(x_norm, 0.0, 500.0)) - 1.0)
    tau_norm = taube * (wnorm / wavbe) ** 3
    bc_at_3000 = b_norm * (1.0 - jnp.exp(-jnp.clip(tau_norm, 0.0, 50.0)))
    bc_at_3000 = jnp.maximum(bc_at_3000, 1e-60)

    scale = bcnorm * cont_at_3000 / bc_at_3000

    # Only below Balmer edge (smooth sigmoid transition)
    steepness = 1.0 / _SIGMOID_WIDTH
    below_edge = jax.nn.sigmoid(-steepness * (jnp.log10(wavelength) - jnp.log10(wavbe)))

    return scale * bc_flux * below_edge


def _emission_line_spectrum(
    wavelength: jnp.ndarray,
    continuum_flam: jnp.ndarray,
    emline_scale: float,
    log_lbol: float,
) -> jnp.ndarray:
    """Emission line spectrum as sum of Gaussians with Baldwin effect.

    Each line is a Gaussian with rest-frame equivalent width from
    the Vanden Berk et al. (2001) composite, scaled by:
    - ``emline_scale``: overall strength multiplier
    - Baldwin effect: EW ~ (L/L_ref)^beslope

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    continuum_flam : array, shape (n_wave,)
        Continuum f_lambda (for EW anchoring).
    emline_scale : float
        Overall emission line strength multiplier.
    log_lbol : float
        log10(L_bol / erg s^-1) for Baldwin effect.

    Returns
    -------
    array, shape (n_wave,)
        Emission line f_lambda (same units as continuum_flam).
    """
    # Baldwin effect scaling: EW ~ (L/L_ref)^beslope
    # In magnitude form: scal ~ (log_lbol - log_lbol_ref)^beslope
    # Simpler: ratio of luminosities to reference
    lbol_ratio = 10.0 ** (log_lbol - _LOG_LBOL_REF)
    baldwin_factor = lbol_ratio**_BALDWIN_SLOPE

    # Scale factor
    scale = emline_scale * baldwin_factor

    def _single_line(line_data):
        """Compute Gaussian profile for one emission line."""
        lam_c = line_data[0]
        ew_rest = line_data[1]
        fwhm_kms = line_data[2]

        # Gaussian sigma in Angstrom
        sigma_ang = lam_c * (fwhm_kms / _C_LIGHT_KMS) / 2.3548
        sigma_ang = jnp.maximum(sigma_ang, 0.01)

        # Gaussian profile (normalized so integral over dlambda = 1)
        profile = jnp.exp(-0.5 * ((wavelength - lam_c) / sigma_ang) ** 2) / (
            sigma_ang * jnp.sqrt(2.0 * jnp.pi)
        )

        # Interpolate continuum at line center for EW conversion
        log_wave = jnp.log10(wavelength)
        log_lam_c = jnp.log10(lam_c)
        cont_at_line = jnp.interp(log_lam_c, log_wave, continuum_flam)
        cont_at_line = jnp.maximum(cont_at_line, 1e-30)

        # Line flux = EW * continuum_at_line (in f_lambda)
        line_flux = scale * ew_rest * cont_at_line

        return line_flux * profile

    # Vectorize over all lines
    line_spectra = jax.vmap(_single_line)(_EMISSION_LINES)
    return jnp.sum(line_spectra, axis=0)


def _apply_dust_reddening(
    wavelength: jnp.ndarray,
    f_lam: jnp.ndarray,
    ebv: float,
) -> jnp.ndarray:
    """Apply SMC-like dust reddening.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    f_lam : array, shape (n_wave,)
        Input spectrum (f_lambda).
    ebv : float
        E(B-V) reddening [mag].

    Returns
    -------
    array, shape (n_wave,)
        Reddened f_lambda.
    """
    # k(lambda) from SMC curve (normalized at V-band)
    k_lam = smc_curve(wavelength)
    # A(lambda) = E(B-V) * k(lambda) * R_V, but our smc returns A/A_V
    # so A(lambda) = E(B-V) * R_V * k(lambda) where R_V ~ 2.93 for SMC
    rv_smc = 2.93
    a_lam = ebv * rv_smc * k_lam
    return f_lam * 10.0 ** (-0.4 * a_lam)


# ===================================================================
# Main QSOgen SED function
# ===================================================================


def qsogen_sed(
    wavelength: jnp.ndarray,
    agn_plslp1: float = _DEFAULT_PLSLP1,
    agn_plslp2: float = _DEFAULT_PLSLP2,
    agn_plbrk: float = _DEFAULT_PLBRK,
    agn_tbb: float = _DEFAULT_TBB,
    agn_bbnorm: float = _DEFAULT_BBNORM,
    agn_emline_scale: float = _DEFAULT_EMLINE_SCALE,
    agn_ebv: float = _DEFAULT_EBV,
    agn_log_lbol: float = 45.0,
    agn_frac: float = 1.0,
    **_kwargs,
) -> jnp.ndarray:
    """QSOgen quasar SED (Temple, Hewett & Banerji 2021).

    Generates a rest-frame quasar SED from 912-100000 Angstrom by
    combining a broken power-law continuum, hot dust blackbody,
    empirical emission lines (with Baldwin effect), and optional
    SMC-like dust reddening.

    The output is L_nu in Lsun/Hz, normalized via ``agn_log_lbol``.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_plslp1 : float
        Blue power-law slope in f_nu (UV/blue side). Default -0.349.
    agn_plslp2 : float
        Red power-law slope in f_nu (optical/red side). Default 0.593.
    agn_plbrk : float
        Break wavelength [Angstrom]. Default 3880.
    agn_tbb : float
        Hot dust temperature [K]. Default 1240.
    agn_bbnorm : float
        Hot dust normalization (relative to continuum at 2 um).
        Default 3.96.
    agn_emline_scale : float
        Emission line strength multiplier. Default 1.0.
    agn_ebv : float
        E(B-V) dust reddening [mag]. Default 0.0 (no reddening).
    agn_log_lbol : float
        log10(L_bol / Lsun). Bolometric luminosity. Default 45.0.
    agn_frac : float
        Overall AGN fraction scaling. Default 1.0.

    Returns
    -------
    array, shape (n_wave,)
        L_nu [Lsun Hz^-1].
    """
    # --- Component 1: Broken power-law continuum ---
    continuum = _broken_powerlaw_continuum(
        wavelength,
        agn_plslp1,
        agn_plslp2,
        agn_plbrk,
    )

    # --- Component 2: Hot dust blackbody ---
    hot_dust = _hot_dust_blackbody(wavelength, continuum, agn_tbb, agn_bbnorm)

    # --- Component 3: Emission lines ---
    # Convert agn_log_lbol from Lsun to erg/s for Baldwin effect
    log_lbol_erg = agn_log_lbol + jnp.log10(_LSUN_ERG)
    emission_lines = _emission_line_spectrum(
        wavelength,
        continuum,
        agn_emline_scale,
        log_lbol_erg,
    )

    # --- Combine components (all in f_nu, normalized ~1 at 5500 A) ---
    # NOTE: Balmer continuum (_balmer_continuum) is available but NOT
    # included by default, matching the original qsogen where bcnorm
    # must be explicitly set. Add via a future agn_bcnorm parameter.
    f_nu_total = continuum + hot_dust + emission_lines

    # --- Component 4: Dust reddening ---
    # Reddening operates on f_lambda, but the extinction law A(lambda) is
    # the same in f_nu: 10^(-0.4 * A_lam) is a multiplicative factor.
    f_nu_total = _apply_dust_reddening(wavelength, f_nu_total, agn_ebv)

    # --- Scale to bolometric luminosity ---
    # Integrate f_nu * dnu to find shape normalization
    nu = _wavelength_to_nu(wavelength)
    idx_sort = jnp.argsort(nu)
    integral_nu = jnp.trapezoid(f_nu_total[idx_sort], nu[idx_sort])
    integral_nu = jnp.maximum(jnp.abs(integral_nu), 1e-30)

    # Scale to L_nu in Lsun/Hz
    l_bol_lsun = 10.0**agn_log_lbol
    l_nu = f_nu_total * (l_bol_lsun / integral_nu)

    return l_nu * agn_frac


# ===================================================================
# Register in AGN_MODELS
# ===================================================================


@register_agn_model("qsogen")
def qsogen(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = 45.0,
    agn_frac: float = 1.0,
    agn_plslp1: float = _DEFAULT_PLSLP1,
    agn_plslp2: float = _DEFAULT_PLSLP2,
    agn_plbrk: float = _DEFAULT_PLBRK,
    agn_tbb: float = _DEFAULT_TBB,
    agn_bbnorm: float = _DEFAULT_BBNORM,
    agn_emline_scale: float = _DEFAULT_EMLINE_SCALE,
    agn_ebv: float = _DEFAULT_EBV,
    **_kwargs,
) -> jnp.ndarray:
    """QSOgen quasar SED (Temple+2021) — registered model entry point.

    Thin wrapper around ``qsogen_sed`` matching the AGN_MODELS registry
    signature: ``fn(wavelength, agn_log_lbol, **kwargs) -> L_nu``.

    See ``qsogen_sed`` for full parameter documentation.
    """
    return qsogen_sed(
        wavelength,
        agn_plslp1=agn_plslp1,
        agn_plslp2=agn_plslp2,
        agn_plbrk=agn_plbrk,
        agn_tbb=agn_tbb,
        agn_bbnorm=agn_bbnorm,
        agn_emline_scale=agn_emline_scale,
        agn_ebv=agn_ebv,
        agn_log_lbol=agn_log_lbol,
        agn_frac=agn_frac,
    )
