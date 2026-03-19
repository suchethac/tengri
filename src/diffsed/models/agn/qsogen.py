"""QSOgen quasar SED model (Temple, Hewett & Banerji 2021).

Empirical quasar SED generator producing rest-frame 912-100000 Angstrom
spectra from seven physically-motivated components:

1. **Broken power-law continuum** with smooth sigmoid transitions.
2. **Hot dust blackbody** emission at ~1240 K.
3. **Emission lines** as analytic Gaussians with Baldwin effect.
4. **Dust reddening** via SMC-like extinction.

The model is parameterized by 7 free parameters that map to the
physical quasar SED shape, making it suitable for photometric fitting
of Type 1 AGN across UV to MIR wavelengths.

All functions are pure JAX and JIT-compilable.

References
----------
- Temple, Hewett & Banerji 2021, MNRAS, 508, 737
- Vanden Berk et al. 2001, AJ, 122, 549 (emission line EWs)
- Gordon et al. 2003, ApJ, 594, 279 (SMC extinction)
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
_EMISSION_LINES = jnp.array([
    # Broad lines
    [1034.0, 3.0, 5000.0],   # OVI 1034
    [1216.0, 90.0, 5000.0],  # Ly-alpha
    [1240.0, 8.0, 5000.0],   # NV 1240
    [1397.0, 5.0, 5000.0],   # SiIV 1397
    [1549.0, 25.0, 5000.0],  # CIV 1549
    [1909.0, 20.0, 5000.0],  # CIII] 1909
    [2800.0, 30.0, 5000.0],  # MgII 2800
    [4340.0, 3.0, 5000.0],   # H-gamma
    [4861.0, 15.0, 5000.0],  # H-beta
    [6563.0, 40.0, 5000.0],  # H-alpha
    [18750.0, 5.0, 5000.0],  # Pa-alpha
    # Narrow lines
    [3727.0, 4.0, 500.0],    # [OII] 3727
    [3869.0, 2.0, 500.0],    # [NeIII] 3869
    [4686.0, 2.0, 500.0],    # HeII 4686
    [5007.0, 5.0, 500.0],    # [OIII] 5007
    [6583.0, 3.0, 500.0],    # [NII] 6583
    [6717.0, 1.5, 500.0],    # [SII] 6717
    [6731.0, 1.5, 500.0],    # [SII] 6731
])

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
) -> jnp.ndarray:
    """Broken power-law continuum in f_lambda.

    Three segments with smooth sigmoid transitions:
    - lambda < 1200 A: steepened slope (alpha_3 = plslp1 - 1 in f_nu)
    - 1200 A < lambda < plbrk: blue slope plslp1 (in f_nu)
    - lambda > plbrk: red slope plslp2 (in f_nu)

    f_nu ~ nu^alpha is equivalent to f_lambda ~ lambda^(-alpha - 2).

    The continuum is normalized to 1.0 at ``_LAMBDA_NORM`` (5500 A).

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    plslp1 : float
        Blue power-law slope in f_nu (UV side of break).
    plslp2 : float
        Red power-law slope in f_nu (optical side of break).
    plbrk : float
        Break wavelength [Angstrom].

    Returns
    -------
    array, shape (n_wave,)
        f_lambda (arbitrary units, normalized at 5500 A).
    """
    # Convert f_nu slopes to f_lambda slopes: f_lam ~ lam^(-alpha_nu - 2)
    slope_blue = -plslp1 - 2.0
    slope_red = -plslp2 - 2.0
    slope_euv = -(plslp1 - 1.0) - 2.0  # Steepened below 1200 A

    # Log-wavelength for sigmoid transitions
    log_wave = jnp.log10(wavelength)
    log_brk = jnp.log10(jnp.maximum(plbrk, 100.0))
    log_1200 = jnp.log10(1200.0)

    steepness = 1.0 / _SIGMOID_WIDTH

    # Sigmoid weights:
    # w_red ~ 1 for lambda >> plbrk, ~ 0 for lambda << plbrk
    w_red = jax.nn.sigmoid(steepness * (log_wave - log_brk))
    # w_euv ~ 1 for lambda << 1200, ~ 0 for lambda >> 1200
    w_euv = jax.nn.sigmoid(-steepness * (log_wave - log_1200))
    # w_blue fills the remainder
    w_blue = 1.0 - w_red - w_euv
    # Clip to avoid negative weights from sigmoid overlap
    w_blue = jnp.clip(w_blue, 0.0, 1.0)

    # Power-law segments, each referenced to its own pivot
    # then anchored together through normalization
    f_blue = (wavelength / plbrk) ** slope_blue
    f_red = (wavelength / plbrk) ** slope_red
    # EUV segment: match blue at 1200 A, then steepen
    f_euv = (wavelength / 1200.0) ** slope_euv * (1200.0 / plbrk) ** slope_blue

    f_lam = w_euv * f_euv + w_blue * f_blue + w_red * f_red

    # Normalize at 5500 A
    log_norm = jnp.log10(_LAMBDA_NORM)
    w_red_n = jax.nn.sigmoid(steepness * (log_norm - log_brk))
    w_euv_n = jax.nn.sigmoid(-steepness * (log_norm - log_1200))
    w_blue_n = jnp.clip(1.0 - w_red_n - w_euv_n, 0.0, 1.0)

    f_blue_n = (_LAMBDA_NORM / plbrk) ** slope_blue
    f_red_n = (_LAMBDA_NORM / plbrk) ** slope_red
    f_euv_n = (_LAMBDA_NORM / 1200.0) ** slope_euv * (1200.0 / plbrk) ** slope_blue

    f_norm = w_euv_n * f_euv_n + w_blue_n * f_blue_n + w_red_n * f_red_n
    f_norm = jnp.maximum(f_norm, 1e-30)

    return f_lam / f_norm


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
    b_lam = _planck_blambda(wavelength, tbb)
    b_lam_anchor = _planck_blambda(jnp.array(_LAMBDA_BB_ANCHOR), tbb)
    b_lam_anchor = jnp.maximum(b_lam_anchor, 1e-60)

    # Evaluate continuum at 2 um by interpolation in log space
    # Use smooth approach: find continuum value at anchor wavelength
    # Since continuum_flam is already on our wavelength grid, we
    # evaluate the Planck shape relative to its own anchor value
    # and scale by bbnorm * continuum_at_anchor.
    # For the continuum at 2 um, evaluate the broken power law there:
    # it's embedded in continuum_flam, so we interpolate.
    log_wave = jnp.log10(wavelength)
    log_anchor = jnp.log10(_LAMBDA_BB_ANCHOR)
    continuum_at_anchor = jnp.interp(
        log_anchor, log_wave, continuum_flam,
    )
    continuum_at_anchor = jnp.maximum(continuum_at_anchor, 1e-30)

    # Blackbody shape normalized at anchor
    bb_shape = b_lam / b_lam_anchor

    return bbnorm * continuum_at_anchor * bb_shape


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
    baldwin_factor = lbol_ratio ** _BALDWIN_SLOPE

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
        profile = jnp.exp(
            -0.5 * ((wavelength - lam_c) / sigma_ang) ** 2
        ) / (sigma_ang * jnp.sqrt(2.0 * jnp.pi))

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
        wavelength, agn_plslp1, agn_plslp2, agn_plbrk,
    )

    # --- Component 2: Hot dust blackbody ---
    hot_dust = _hot_dust_blackbody(wavelength, continuum, agn_tbb, agn_bbnorm)

    # --- Component 3: Emission lines ---
    # Convert agn_log_lbol from Lsun to erg/s for Baldwin effect
    log_lbol_erg = agn_log_lbol + jnp.log10(_LSUN_ERG)
    emission_lines = _emission_line_spectrum(
        wavelength, continuum, agn_emline_scale, log_lbol_erg,
    )

    # --- Combine components (all in f_lambda, normalized at 5500 A) ---
    f_lam_total = continuum + hot_dust + emission_lines

    # --- Component 4: Dust reddening ---
    f_lam_total = _apply_dust_reddening(wavelength, f_lam_total, agn_ebv)

    # --- Convert to L_nu and scale to bolometric luminosity ---
    # f_lam is in arbitrary units (normalized at 5500 A).
    # Integrate f_lam * dlam to get bolometric "shape integral",
    # then scale so total luminosity = 10^agn_log_lbol Lsun.

    # L_nu = L_lam * lam^2 / c  (with c in Angstrom/s)
    c_ang = _C_LIGHT / _ANGSTROM_CM  # c in Angstrom/s
    f_nu = f_lam_total * wavelength**2 / c_ang

    # Integrate L_nu * dnu to find shape normalization
    nu = _wavelength_to_nu(wavelength)
    idx_sort = jnp.argsort(nu)
    integral_nu = jnp.trapezoid(f_nu[idx_sort], nu[idx_sort])
    integral_nu = jnp.maximum(jnp.abs(integral_nu), 1e-30)

    # Scale to bolometric luminosity
    l_bol_lsun = 10.0**agn_log_lbol
    l_nu = f_nu * (l_bol_lsun / integral_nu)

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
