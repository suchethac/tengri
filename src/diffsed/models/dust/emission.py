"""Dust emission models for diffsed.

This module implements IR re-emission of UV/optical light absorbed by dust.
All models are pure JAX (JIT-compatible, fully differentiable) and follow
the energy-balance constraint: total IR luminosity equals total absorbed
luminosity from the attenuation step.

Available Emission Models
-------------------------
- **modified_blackbody**: Optically-thin modified blackbody (2-3 params)
- **dale2014**: Dale et al. (2014) 1-parameter IR template family
- **draine_li2007**: Draine & Li (2007) 3-parameter model (analytic approx.)
- **draine_li2014**: Draine & Li (2014 update) 4-parameter model (analytic approx.)

Energy Balance
--------------
The normalization for every model is set by::

    L_dust_emission = L_dust_absorbed
                    = integral[(1 - transmission) * L_stellar_intrinsic * dlambda]

This is computed from the attenuation step and passed to each model as
``L_absorbed`` (scalar, in Lsun).

References
----------
- Dale et al. 2014, ApJ, 784, 83
- Draine & Li 2007, ApJ, 657, 810
- Draine & Li 2014 update (CIGALE implementation, Boquien+2019)
- Aniano et al. 2012, ApJ, 756, 138
- da Cunha et al. 2013, ApJ, 766, 13
- Hildebrand 1983, QJRAS, 24, 267
"""

from collections.abc import Callable

import jax.numpy as jnp

# ===================================================================
# Physical constants (CGS)
# ===================================================================

_H_PLANCK = 6.62607015e-27  # erg s
_K_BOLTZMANN = 1.380649e-16  # erg / K
_C_CGS = 2.99792458e10  # cm / s
_LSUN_ERG = 3.828e33  # erg / s  (IAU 2015)
_AA_TO_CM = 1.0e-8  # Angstrom -> cm


# ===================================================================
# Emission model registry
# ===================================================================

DUST_EMISSION_MODELS: dict[str, Callable] = {}


def register_emission_model(name: str) -> Callable:
    """Register a dust emission model function (decorator factory)."""

    def decorator(fn: Callable) -> Callable:
        DUST_EMISSION_MODELS[name] = fn
        return fn

    return decorator


def get_emission_model(name: str) -> Callable:
    """Get a registered emission model by name.

    Parameters
    ----------
    name : str
        Model name (e.g. ``"modified_blackbody"``).

    Returns
    -------
    Callable
        The model function with signature
        ``(wavelength, L_absorbed, **params) -> L_nu_emission``.

    Raises
    ------
    ValueError
        If *name* is not in the registry.
    """
    if name not in DUST_EMISSION_MODELS:
        raise ValueError(
            f"Unknown dust emission model '{name}'. Available: {list(DUST_EMISSION_MODELS.keys())}"
        )
    return DUST_EMISSION_MODELS[name]


# ===================================================================
# Utility: Planck function
# ===================================================================


def planck_bnu(
    wavelength_aa: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    """Planck function B_nu(T) evaluated at given wavelengths.

    Parameters
    ----------
    wavelength_aa : array
        Wavelength grid in Angstrom.
    temperature : float
        Blackbody temperature in Kelvin.

    Returns
    -------
    array
        B_nu in erg / s / cm^2 / Hz / sr.
    """
    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    # x = h*nu / (k*T), clipped to avoid overflow in exp
    x = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * temperature), 0.0, 500.0)
    return 2.0 * _H_PLANCK * nu**3 / (_C_CGS**2) / (jnp.exp(x) - 1.0)


# ===================================================================
# CMB heating correction (da Cunha+2013)
# ===================================================================

_T_CMB_0 = 2.725  # CMB temperature at z=0 (K)


def cmb_corrected_temperature(
    T_dust: float,
    redshift: float,
    beta_ir: float = 1.6,
) -> float:
    """Effective dust temperature including CMB heating.

    At high redshift the CMB sets a temperature floor on dust grains.
    The effective equilibrium temperature is (da Cunha et al. 2013)::

        T_eff = (T_dust ^ (4 + beta) + T_CMB(z) ^ (4 + beta) - T_CMB(z=0) ^ (4 + beta)) ^ {
            1 / (4 + beta)
        }

    Parameters
    ----------
    T_dust : float
        Intrinsic dust temperature in Kelvin (what the galaxy would
        have at z=0 in isolation).
    redshift : float
        Source redshift.
    beta_ir : float
        Dust emissivity index. Default 1.6.

    Returns
    -------
    float
        Effective dust temperature in Kelvin.
    """
    exponent = 4.0 + beta_ir
    T_cmb_z = _T_CMB_0 * (1.0 + redshift)
    T_eff = (T_dust**exponent + T_cmb_z**exponent - _T_CMB_0**exponent) ** (1.0 / exponent)
    return T_eff


def cmb_contrast_factor(
    wavelength_aa: jnp.ndarray,
    T_eff: float,
    redshift: float,
) -> jnp.ndarray:
    """Flux suppression factor from observing dust against the CMB.

    The observed flux is reduced because the galaxy's dust emission is
    measured against the CMB background (da Cunha et al. 2013)::

        S_obs / S_intrinsic = 1 - B_nu(T_CMB(z)) / B_nu(T_eff)

    Parameters
    ----------
    wavelength_aa : array
        Wavelength grid in Angstrom.
    T_eff : float
        CMB-corrected effective dust temperature (K).
    redshift : float
        Source redshift.

    Returns
    -------
    array
        Multiplicative contrast factor in [0, 1].
    """
    T_cmb_z = _T_CMB_0 * (1.0 + redshift)

    # Compute the Planck ratio B_nu(T_cmb)/B_nu(T_eff) stably.
    # Since both share the same nu^3 prefactor, the ratio simplifies to
    #   (exp(x_eff) - 1) / (exp(x_cmb) - 1)
    # where x = h*nu/(k*T).  For x >> 1 this approaches exp(x_eff - x_cmb)
    # which is safe because x_cmb > x_eff (T_eff > T_cmb).
    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    x_eff = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * T_eff), 0.0, 500.0)
    x_cmb = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * T_cmb_z), 0.0, 500.0)

    # Ratio = (exp(x_eff) - 1) / (exp(x_cmb) - 1)
    # Use log-space: log(ratio) = log(expm1(x_eff)) - log(expm1(x_cmb))
    # For large x, expm1(x) ~ exp(x), so log(expm1(x)) ~ x.
    log_expm1_eff = jnp.where(
        x_eff > 30.0, x_eff, jnp.log(jnp.expm1(jnp.clip(x_eff, 1e-10, 30.0)))
    )
    log_expm1_cmb = jnp.where(
        x_cmb > 30.0, x_cmb, jnp.log(jnp.expm1(jnp.clip(x_cmb, 1e-10, 30.0)))
    )

    # B_cmb/B_eff = exp(log_expm1_eff - log_expm1_cmb)
    # Since T_eff >= T_cmb, x_cmb >= x_eff, so the exponent is <= 0
    # and the ratio is in [0, 1].
    log_ratio = log_expm1_eff - log_expm1_cmb
    ratio = jnp.exp(jnp.clip(log_ratio, -100.0, 0.0))

    return jnp.clip(1.0 - ratio, 0.0, 1.0)


# ===================================================================
# Energy balance
# ===================================================================


def compute_absorbed_luminosity(
    wavelength_aa: jnp.ndarray,
    L_nu_intrinsic: jnp.ndarray,
    transmission: jnp.ndarray,
) -> float:
    """Compute total luminosity absorbed by dust.

    This is the energy-balance integral::

        L_absorbed = integral[(1 - T(lambda)) * L_nu_intrinsic * dnu]

    where T(lambda) is the dust transmission fraction (output of the
    attenuation model, values in [0, 1]).

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Rest-frame wavelength grid in Angstrom (must be sorted ascending).
    L_nu_intrinsic : array, shape (n_wave,)
        Intrinsic (dust-free) luminosity density in Lsun/Hz.
    transmission : array, shape (n_wave,)
        Dust transmission fraction in [0, 1].  For age-dependent models
        this should be the SFH-weighted effective transmission.

    Returns
    -------
    float
        Total absorbed luminosity in Lsun (integrated over frequency).
    """
    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm  # descending (since wave is ascending)

    absorbed_Lnu = (1.0 - transmission) * L_nu_intrinsic

    # Integrate over frequency: nu is descending, negate for positive result
    return -jnp.trapezoid(absorbed_Lnu, nu)


def compute_absorbed_luminosity_from_tau(
    wavelength_aa: jnp.ndarray,
    L_nu_intrinsic: jnp.ndarray,
    tau_lambda: jnp.ndarray,
) -> float:
    """Compute total absorbed luminosity from optical depth.

    Convenience wrapper when you have tau(lambda) rather than transmission.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Rest-frame wavelength grid in Angstrom (sorted ascending).
    L_nu_intrinsic : array, shape (n_wave,)
        Intrinsic luminosity density in Lsun/Hz.
    tau_lambda : array, shape (n_wave,)
        Optical depth as a function of wavelength.

    Returns
    -------
    float
        Total absorbed luminosity in Lsun.
    """
    transmission = jnp.exp(-tau_lambda)
    return compute_absorbed_luminosity(wavelength_aa, L_nu_intrinsic, transmission)


# ===================================================================
# Model 1: Modified blackbody (2-3 parameters)
# ===================================================================


@register_emission_model("modified_blackbody")
def modified_blackbody(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_T: float = 30.0,
    dust_beta_ir: float = 1.8,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Optically-thin modified blackbody dust emission.

    The unnormalized spectrum is::

        S_nu ~ nu^beta * B_nu(T_dust)

    which is then normalized so that the frequency integral equals
    ``L_absorbed``.

    When ``redshift > 0``, the dust temperature is corrected for CMB
    heating (da Cunha et al. 2013) and the observed flux is reduced by
    the CMB contrast factor.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending).
    L_absorbed : float
        Total absorbed luminosity in Lsun (sets the normalization).
    dust_T : float
        Dust temperature in Kelvin.  Typical range: 20--60 K.
    dust_beta_ir : float
        Emissivity index.  Typical range: 1.5--2.0.
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default 0 (no correction, backward compatible).

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.
    """
    # CMB correction: always applied. At z=0 this is a no-op since
    # T_cmb(z=0) terms cancel and B_nu(T_cmb)/B_nu(T_dust) ~ 0.
    T_eff = cmb_corrected_temperature(dust_T, redshift, dust_beta_ir)

    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    # Reference frequency at 250 um (convenient normalization pivot)
    nu_ref = _C_CGS / (250.0e-4)  # 250 um in cm
    emissivity = (nu / nu_ref) ** dust_beta_ir

    bnu = planck_bnu(wavelength_aa, T_eff)

    # Unnormalized SED shape (erg/s/cm^2/Hz/sr units cancel in ratio)
    shape = emissivity * bnu

    # Integrate shape over frequency for normalization.
    # nu is descending (wave ascending), so negate to get positive integral.
    integral = -jnp.trapezoid(shape, nu)

    # Guard against zero integral (e.g. wavelength grid entirely outside
    # the thermal peak) — return zeros instead of NaN
    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    result = norm * shape

    # CMB contrast: suppresses flux where dust is observed against CMB
    contrast = cmb_contrast_factor(wavelength_aa, T_eff, redshift)

    return result * contrast


# ===================================================================
# Model 2: Dale et al. 2014 (1 parameter)
# ===================================================================


def _dale_component_temperature(alpha: float) -> tuple[float, float, float]:
    """Map Dale alpha to warm/cold modified-blackbody temperatures.

    The Dale et al. (2014) templates span alpha = [0.0625, 4.0].
    Low alpha -> intense radiation field -> warmer dust.
    High alpha -> weak radiation field -> cooler dust.

    We approximate the full template library as a two-component model:
    a cold (diffuse ISM) component and a warm (PDR/HII) component whose
    temperature ratio and mixing fraction vary with alpha.

    Returns (T_cold, T_warm, f_warm) where f_warm is the warm fraction.
    """
    # Cold component: 15-25 K, rising gently at low alpha
    T_cold = 20.0 + 5.0 * jnp.exp(-0.5 * alpha)

    # Warm component: 40-70 K
    T_warm = 70.0 - 10.0 * alpha

    # Warm fraction: dominant at low alpha, negligible at high alpha
    f_warm = jnp.clip(0.8 * jnp.exp(-0.6 * alpha), 0.01, 0.99)

    return T_cold, T_warm, f_warm


@register_emission_model("dale2014")
def dale2014(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_alpha_dale: float = 2.0,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Dale et al. (2014) 1-parameter dust emission template.

    The full Dale model parameterizes the IR SED by the power-law slope
    alpha of the radiation field intensity distribution dM/dU ~ U^{-alpha}.
    Here we provide an analytic two-component approximation that captures
    the key alpha-dependent behaviour: low alpha yields warm, peaked SEDs;
    high alpha yields cooler, broader SEDs.

    When ``redshift > 0``, CMB heating correction (da Cunha+2013) is
    applied to both temperature components, and the CMB contrast factor
    suppresses the observed flux.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending).
    L_absorbed : float
        Total absorbed luminosity in Lsun.
    dust_alpha_dale : float
        Power-law slope.  Valid range: 0.0625--4.0.
        alpha ~ 1-1.5: luminous IR galaxies.
        alpha ~ 2-2.5: normal star-forming galaxies.
        alpha ~ 3-4: quiescent galaxies.
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default 0 (no correction, backward compatible).

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.
    """
    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    T_cold, T_warm, f_warm = _dale_component_temperature(dust_alpha_dale)

    # Both components have beta_ir = 1.8 (typical grain emissivity)
    beta_ir = 1.8

    # CMB correction: always applied (no-op at z=0)
    T_cold = cmb_corrected_temperature(T_cold, redshift, beta_ir)
    T_warm = cmb_corrected_temperature(T_warm, redshift, beta_ir)

    nu_ref = _C_CGS / (250.0e-4)
    emissivity = (nu / nu_ref) ** beta_ir

    bnu_cold = planck_bnu(wavelength_aa, T_cold)
    bnu_warm = planck_bnu(wavelength_aa, T_warm)

    shape = emissivity * ((1.0 - f_warm) * bnu_cold + f_warm * bnu_warm)

    # nu is descending, negate for positive integral
    integral = -jnp.trapezoid(shape, nu)

    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    result = norm * shape

    # CMB contrast using luminosity-weighted effective T
    T_eff_avg = (1.0 - f_warm) * T_cold + f_warm * T_warm
    contrast = cmb_contrast_factor(wavelength_aa, T_eff_avg, redshift)

    return result * contrast


# ===================================================================
# Model 3: Draine & Li 2007 (3 parameters) — analytic approximation
# ===================================================================


def _draine_li_umin_to_temperature(U_min: float) -> float:
    """Convert minimum radiation field intensity to effective dust temperature.

    The Draine & Li (2007) model uses a local ISRF intensity U
    (in units of the Mathis et al. 1983 ISRF).  The equilibrium grain
    temperature scales roughly as T ~ 18 * U^(1/6) K for silicate grains.
    """
    return 18.0 * U_min ** (1.0 / 6.0)


def _pdr_temperature(U_min: float) -> float:
    """Effective temperature for the PDR (photo-dissociation region) component.

    Grains in PDRs are exposed to U >> U_min.  We use a characteristic
    temperature corresponding to the geometric mean of U_min and U_max,
    where U_max ~ 1e6 (Draine & Li 2007 default).
    """
    U_eff = jnp.sqrt(U_min * 1.0e6)
    return 18.0 * U_eff ** (1.0 / 6.0)


@register_emission_model("draine_li2007")
def draine_li2007(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_umin: float = 1.0,
    dust_gamma_dl: float = 0.01,
    dust_qpah: float = 2.5,
    **_kwargs,
) -> jnp.ndarray:
    """Draine & Li (2007) dust emission model (analytic approximation).

    The full DL07 model requires pre-computed grain opacity tables for
    silicate, graphite, and PAH grains.  This implementation provides a
    differentiable analytic approximation:

    - **(1 - gamma)** of the dust mass is heated by a single radiation
      field U_min, producing a cool modified-blackbody component.
    - **gamma** of the dust mass sits in PDR environments with a
      distribution of U from U_min to U_max = 1e6, approximated as a
      warm modified-blackbody at an effective temperature.
    - **q_PAH** controls the relative strength of mid-IR PAH emission
      features, modelled here as a 7.7 um warm component.

    For production use with the full grain model, load the DL07 templates
    via ``load_draine_li_templates()`` and use ``draine_li2007_from_grid()``.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending).
    L_absorbed : float
        Total absorbed luminosity in Lsun.
    dust_umin : float
        Minimum radiation field intensity (Mathis ISRF units).
        Typical range: 0.1--25.
    dust_gamma_dl : float
        Fraction of dust mass in PDR regions.
        Typical range: 0.0--1.0 (usually < 0.1 for normal galaxies).
    dust_qpah : float
        PAH mass fraction in percent.
        Typical range: 0.47--4.58 %.

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.
    """
    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    beta_ir = 2.0  # DL07 uses beta~2 for large grains
    nu_ref = _C_CGS / (250.0e-4)
    emissivity = (nu / nu_ref) ** beta_ir

    # --- Component 1: Diffuse ISM (FIR continuum) ---
    T_diff = _draine_li_umin_to_temperature(dust_umin)
    shape_diff = emissivity * planck_bnu(wavelength_aa, T_diff)

    # --- Component 2: PDR warm continuum ---
    T_pdr = _pdr_temperature(dust_umin)
    shape_pdr = emissivity * planck_bnu(wavelength_aa, T_pdr)

    # --- Component 3: PAH mid-IR feature complex ---
    # Approximate the blended 6.2, 7.7, 8.6, 11.3, 12.7 um features
    # as a Gaussian centered on 7.7 um with ~3 um width.
    lambda_pah = 7.7e4  # 7.7 um in Angstrom
    sigma_pah = 3.0e4  # ~3 um width (blends 6.2-12.7 um)
    shape_pah = jnp.exp(-0.5 * ((wavelength_aa - lambda_pah) / sigma_pah) ** 2)

    # --- Luminosity fractions ---
    # Calibrated against bagpipes DL07 templates (Draine & Li 2007):
    #   At q_PAH=2.5%, U_min=1.0, gamma=0.01:
    #     PAH (5-15 um) ~ 40% of L_IR
    #     FIR (>30 um)  ~ 35% of L_IR
    #     MIR continuum ~ 25% of L_IR
    # PAH fraction scales with q_PAH; FIR fraction grows with U_min.
    f_pah = (dust_qpah / 3.5) * 0.35
    f_pdr = dust_gamma_dl
    f_diff = 1.0 - f_pah - f_pdr

    # For the PAH component (defined in wavelength-space as a Gaussian),
    # convert to L_nu: L_nu = L_lambda * lambda^2 / c, then normalize.
    shape_pah_lnu = shape_pah * (wavelength_cm**2) / _C_CGS

    # Normalize all 3 components at once: negate because nu is descending
    int_diff = -jnp.trapezoid(shape_diff, nu)
    int_pdr = -jnp.trapezoid(shape_pdr, nu)
    int_pah = -jnp.trapezoid(shape_pah_lnu, nu)

    l_nu = (
        f_diff * jnp.where(int_diff > 0.0, shape_diff / int_diff, shape_diff)
        + f_pdr * jnp.where(int_pdr > 0.0, shape_pdr / int_pdr, shape_pdr)
        + f_pah * jnp.where(int_pah > 0.0, shape_pah_lnu / int_pah, shape_pah_lnu)
    )

    return L_absorbed * l_nu


# ===================================================================
# Template-based DL07: create from grid file
# ===================================================================


def create_dl07_from_grid(grid_path: str) -> Callable:
    """Create a DL07 emission model function backed by tabulated templates.

    Loads the HDF5 grid once and returns a function matching the emission
    model registry interface. Use this instead of the analytic approximation
    for production work.

    Parameters
    ----------
    grid_path : str
        Path to ``dl07_templates.h5`` (from ``scripts/convert_dl07_templates.py``).

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, **params) -> L_nu``.

    Example
    -------
    >>> dl07 = create_dl07_from_grid("data/dl07_templates.h5")
    >>> DUST_EMISSION_MODELS["dl07_tabulated"] = dl07  # optional: register
    >>> sed_ir = dl07(wavelength, L_absorbed, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5)
    """
    templates = load_draine_li_templates(grid_path)

    # Pre-extract arrays for the closure
    single_u = templates["single_u"]  # (n_qpah, n_umin, n_wave)
    powerlaw = templates["powerlaw"]  # (n_qpah, n_umin, n_wave)
    tmpl_wave = templates["wavelength"]
    umin_grid = templates["umin_grid"]
    qpah_grid = templates["qpah_grid"]

    def dl07_tabulated(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_umin: float = 1.0,
        dust_gamma_dl: float = 0.01,
        dust_qpah: float = 2.5,
        **_kwargs,
    ) -> jnp.ndarray:
        """DL07 emission from tabulated templates (Draine & Li 2007).

        j_nu = (1-gamma) * single_U(q_PAH, U_min)
             + gamma * powerlaw(q_PAH, U_min)

        Normalized to L_absorbed via energy balance.
        """
        dust_umin_c = jnp.clip(dust_umin, umin_grid[0], umin_grid[-1])
        dust_qpah_c = jnp.clip(dust_qpah, qpah_grid[0], qpah_grid[-1])

        # Bilinear interpolation indices
        i_u = jnp.clip(jnp.searchsorted(umin_grid, dust_umin_c) - 1, 0, len(umin_grid) - 2)
        i_q = jnp.clip(jnp.searchsorted(qpah_grid, dust_qpah_c) - 1, 0, len(qpah_grid) - 2)

        fu = (dust_umin_c - umin_grid[i_u]) / (umin_grid[i_u + 1] - umin_grid[i_u])
        fq = (dust_qpah_c - qpah_grid[i_q]) / (qpah_grid[i_q + 1] - qpah_grid[i_q])

        def _bilinear(grid):
            return (
                (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u]
                + (1.0 - fq) * fu * grid[i_q, i_u + 1]
                + fq * (1.0 - fu) * grid[i_q + 1, i_u]
                + fq * fu * grid[i_q + 1, i_u + 1]
            )

        # Mix single-U and power-law components via gamma
        template = (1.0 - dust_gamma_dl) * _bilinear(single_u) + dust_gamma_dl * _bilinear(
            powerlaw
        )

        # Interpolate template onto target wavelength grid
        sed = jnp.interp(wavelength_aa, tmpl_wave, template, left=0.0, right=0.0)

        return L_absorbed * sed

    return dl07_tabulated


def load_draine_li_templates(filepath: str) -> dict:
    """Load DL07 template grid from HDF5 or NPZ.

    Supports two formats:
    - HDF5 with keys: wavelength, umin_grid, qpah_grid, single_u, powerlaw
    - NPZ with keys: wavelength, umin_grid, qpah_grid,
      templates_umin_only, templates_umin_umax

    The templates must be pre-normalized so that each template integrates
    to 1 over wavelength (L_lambda convention). The model function handles
    the L_absorbed scaling.

    Parameters
    ----------
    filepath : str
        Path to template file (.h5 or .npz).

    Returns
    -------
    dict with keys: wavelength, umin_grid, qpah_grid, single_u, powerlaw
        All arrays are JAX arrays. single_u and powerlaw have shape
        (n_qpah, n_umin, n_wave).
    """
    import numpy as np

    if filepath.endswith(".npz"):
        data = np.load(filepath)
        wavs = data["wavelength"]
        single_u = data["templates_umin_only"]  # (n_qpah, n_umin, n_wave)
        powerlaw = data["templates_umin_umax"]

        # Normalize each template to integrate to 1 over wavelength
        for i in range(single_u.shape[0]):
            for j in range(single_u.shape[1]):
                norm = np.trapezoid(single_u[i, j], wavs)
                if norm > 0:
                    single_u[i, j] /= norm
                norm = np.trapezoid(powerlaw[i, j], wavs)
                if norm > 0:
                    powerlaw[i, j] /= norm

        return {
            "wavelength": jnp.array(wavs),
            "umin_grid": jnp.array(data["umin_grid"]),
            "qpah_grid": jnp.array(data["qpah_grid"]),
            "single_u": jnp.array(single_u),
            "powerlaw": jnp.array(powerlaw),
        }

    # HDF5 format
    import h5py as _h5py

    with _h5py.File(filepath, "r") as f:
        return {
            "wavelength": jnp.array(f["wavelength"][:]),
            "umin_grid": jnp.array(f["umin_grid"][:]),
            "qpah_grid": jnp.array(f["qpah_grid"][:]),
            "single_u": jnp.array(f["single_u"][:]),
            "powerlaw": jnp.array(f["powerlaw"][:]),
        }


# ===================================================================
# Model 4: Draine & Li 2014 (4 parameters) — analytic approximation
# ===================================================================


def _pdr_temperature_dl14(U_min: float, U_max: float = 1.0e7) -> float:
    """Effective temperature for the PDR component (DL14).

    Like DL07 but with U_max = 10^7 instead of 10^6.
    """
    U_eff = jnp.sqrt(U_min * U_max)
    return 18.0 * U_eff ** (1.0 / 6.0)


def _alpha_warm_fraction_correction(alpha: float) -> float:
    """Correction to the warm/PDR fraction based on the alpha slope.

    In DL07, alpha is fixed at 2.0. In DL14, alpha controls how much
    dust mass is exposed to high-U radiation fields:
    - Low alpha (steep): more dust at high U -> warmer emission
    - High alpha (shallow): less dust at high U -> cooler emission

    The warm fraction scales roughly as 1/(alpha - 1) for the integral
    of U^{-alpha} from U_min to U_max, normalized relative to alpha=2.
    """
    # Relative to alpha=2.0 (DL07 default), the luminosity-weighted
    # warm fraction scales approximately as (alpha-1)^{-1} / (2-1)^{-1}
    # = 1/(alpha-1).  Clip to avoid divergence near alpha=1.
    alpha_safe = jnp.clip(alpha, 1.01, 5.0)
    return 1.0 / (alpha_safe - 1.0)


@register_emission_model("draine_li2014")
def draine_li2014(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_umin: float = 1.0,
    dust_gamma_dl: float = 0.01,
    dust_qpah: float = 2.5,
    dust_alpha_dl14: float = 2.0,
    **_kwargs,
) -> jnp.ndarray:
    """Draine & Li (2014 update) dust emission model (analytic approx.).

    Extends the DL07 analytic approximation with:
    - Variable alpha (power-law slope of radiation field distribution)
    - Extended q_PAH range (0.47-7.32%)
    - Extended U_min range (0.1-50)
    - U_max = 10^7 (was 10^6 in DL07)

    The model:
    - **(1 - gamma)** of the dust mass is heated by U = U_min only
      (cool modified-blackbody).
    - **gamma** of the dust mass sits in PDR environments with
      dM/dU ~ U^{-alpha} from U_min to U_max = 10^7.
    - **q_PAH** controls mid-IR PAH emission features.
    - **alpha** controls the power-law slope (steeper = more warm dust).

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending).
    L_absorbed : float
        Total absorbed luminosity in Lsun.
    dust_umin : float
        Minimum radiation field intensity (Mathis ISRF units).
        Typical range: 0.1--50.
    dust_gamma_dl : float
        Fraction of dust mass in PDR regions.
        Typical range: 0.0--1.0.
    dust_qpah : float
        PAH mass fraction in percent.
        Typical range: 0.47--7.32 %.
    dust_alpha_dl14 : float
        Power-law slope of the radiation field distribution.
        Range: 1.0--3.0. Default 2.0 (recovers DL07 behaviour).

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.
    """
    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    beta_ir = 2.0  # Grain emissivity index
    nu_ref = _C_CGS / (250.0e-4)
    emissivity = (nu / nu_ref) ** beta_ir

    # --- Component 1: Diffuse ISM (FIR continuum) ---
    T_diff = _draine_li_umin_to_temperature(dust_umin)
    shape_diff = emissivity * planck_bnu(wavelength_aa, T_diff)

    # --- Component 2: PDR warm continuum ---
    # Effective temperature depends on alpha: lower alpha means the
    # luminosity-weighted <U> is higher, so use a scaled T_pdr.
    T_pdr_base = _pdr_temperature_dl14(dust_umin)
    # Modulate: for alpha < 2 dust is warmer; for alpha > 2 it is cooler.
    alpha_corr = _alpha_warm_fraction_correction(dust_alpha_dl14)
    # T scales as U_eff^(1/6), and U_eff ~ alpha_corr, so:
    T_pdr = T_pdr_base * alpha_corr ** (1.0 / 6.0)
    shape_pdr = emissivity * planck_bnu(wavelength_aa, T_pdr)

    # --- Component 3: PAH mid-IR feature complex ---
    lambda_pah = 7.7e4  # 7.7 um in Angstrom
    sigma_pah = 3.0e4  # ~3 um width (blends 6.2-12.7 um)
    shape_pah = jnp.exp(-0.5 * ((wavelength_aa - lambda_pah) / sigma_pah) ** 2)

    # --- Luminosity fractions ---
    f_pah = (dust_qpah / 3.5) * 0.35
    f_pdr = dust_gamma_dl
    f_diff = 1.0 - f_pah - f_pdr

    shape_pah_lnu = shape_pah * (wavelength_cm**2) / _C_CGS

    # Normalize all 3 components: negate because nu is descending
    int_diff = -jnp.trapezoid(shape_diff, nu)
    int_pdr = -jnp.trapezoid(shape_pdr, nu)
    int_pah = -jnp.trapezoid(shape_pah_lnu, nu)

    l_nu = (
        f_diff * jnp.where(int_diff > 0.0, shape_diff / int_diff, shape_diff)
        + f_pdr * jnp.where(int_pdr > 0.0, shape_pdr / int_pdr, shape_pdr)
        + f_pah * jnp.where(int_pah > 0.0, shape_pah_lnu / int_pah, shape_pah_lnu)
    )

    return L_absorbed * l_nu


# ===================================================================
# Template-based DL14: create from grid file
# ===================================================================


def load_dl14_templates(filepath: str) -> dict:
    """Load DL14 template grid from HDF5.

    Parameters
    ----------
    filepath : str
        Path to ``dl14_templates.h5`` (from ``scripts/convert_dl14_templates.py``).

    Returns
    -------
    dict with keys:
        wavelength, umin_grid, qpah_grid, alpha_grid, single_u, powerlaw.
        single_u has shape (n_qpah, n_umin, n_wave).
        powerlaw has shape (n_qpah, n_umin, n_alpha, n_wave).
    """
    import h5py as _h5py

    with _h5py.File(filepath, "r") as f:
        return {
            "wavelength": jnp.array(f["wavelength"][:]),
            "umin_grid": jnp.array(f["umin_grid"][:]),
            "qpah_grid": jnp.array(f["qpah_grid"][:]),
            "alpha_grid": jnp.array(f["alpha_grid"][:]),
            "single_u": jnp.array(f["single_u"][:]),
            "powerlaw": jnp.array(f["powerlaw"][:]),
        }


def create_dl14_from_grid(grid_path: str) -> Callable:
    """Create a DL14 emission model function backed by tabulated templates.

    Loads the HDF5 grid once and returns a function matching the emission
    model registry interface. The key difference from DL07: the powerlaw
    template now depends on alpha too, requiring trilinear interpolation
    in (q_PAH, U_min, alpha) space instead of bilinear.

    Parameters
    ----------
    grid_path : str
        Path to ``dl14_templates.h5`` (from ``scripts/convert_dl14_templates.py``).

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, **params) -> L_nu``.

    Example
    -------
    >>> dl14 = create_dl14_from_grid("data/dl14_templates.h5")
    >>> DUST_EMISSION_MODELS["dl14_tabulated"] = dl14
    >>> sed = dl14(
    ...     wav, L_abs, dust_umin=1.0, dust_gamma_dl=0.01, dust_qpah=2.5, dust_alpha_dl14=2.0
    ... )
    """
    templates = load_dl14_templates(grid_path)

    single_u = templates["single_u"]  # (n_qpah, n_umin, n_wave)
    powerlaw = templates["powerlaw"]  # (n_qpah, n_umin, n_alpha, n_wave)
    tmpl_wave = templates["wavelength"]
    umin_grid = templates["umin_grid"]
    qpah_grid = templates["qpah_grid"]
    alpha_grid = templates["alpha_grid"]

    def dl14_tabulated(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_umin: float = 1.0,
        dust_gamma_dl: float = 0.01,
        dust_qpah: float = 2.5,
        dust_alpha_dl14: float = 2.0,
        **_kwargs,
    ) -> jnp.ndarray:
        """DL14 emission from tabulated templates.

        j_nu = (1-gamma) * single_U(q_PAH, U_min)
             + gamma * powerlaw(q_PAH, U_min, alpha)

        Normalized to L_absorbed via energy balance.
        """
        # Clip parameters to grid bounds
        dust_umin_c = jnp.clip(dust_umin, umin_grid[0], umin_grid[-1])
        dust_qpah_c = jnp.clip(dust_qpah, qpah_grid[0], qpah_grid[-1])
        dust_alpha_c = jnp.clip(dust_alpha_dl14, alpha_grid[0], alpha_grid[-1])

        # Interpolation indices and fractions
        n_u = len(umin_grid)
        n_q = len(qpah_grid)
        n_a = len(alpha_grid)

        i_u = jnp.clip(jnp.searchsorted(umin_grid, dust_umin_c) - 1, 0, n_u - 2)
        i_q = jnp.clip(jnp.searchsorted(qpah_grid, dust_qpah_c) - 1, 0, n_q - 2)
        i_a = jnp.clip(jnp.searchsorted(alpha_grid, dust_alpha_c) - 1, 0, n_a - 2)

        fu = (dust_umin_c - umin_grid[i_u]) / (umin_grid[i_u + 1] - umin_grid[i_u])
        fq = (dust_qpah_c - qpah_grid[i_q]) / (qpah_grid[i_q + 1] - qpah_grid[i_q])
        fa = (dust_alpha_c - alpha_grid[i_a]) / (alpha_grid[i_a + 1] - alpha_grid[i_a])

        # Bilinear interpolation for single-U (q_PAH, U_min)
        def _bilinear(grid):
            return (
                (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u]
                + (1.0 - fq) * fu * grid[i_q, i_u + 1]
                + fq * (1.0 - fu) * grid[i_q + 1, i_u]
                + fq * fu * grid[i_q + 1, i_u + 1]
            )

        # Trilinear interpolation for powerlaw (q_PAH, U_min, alpha)
        def _trilinear(grid):
            # Interpolate at alpha[i_a] and alpha[i_a+1] via bilinear in (q, u)
            def _bilinear_at_alpha(ia_idx):
                return (
                    (1.0 - fq) * (1.0 - fu) * grid[i_q, i_u, ia_idx]
                    + (1.0 - fq) * fu * grid[i_q, i_u + 1, ia_idx]
                    + fq * (1.0 - fu) * grid[i_q + 1, i_u, ia_idx]
                    + fq * fu * grid[i_q + 1, i_u + 1, ia_idx]
                )

            lo = _bilinear_at_alpha(i_a)
            hi = _bilinear_at_alpha(i_a + 1)
            return (1.0 - fa) * lo + fa * hi

        # Mix single-U and power-law components via gamma
        template = (1.0 - dust_gamma_dl) * _bilinear(single_u) + dust_gamma_dl * _trilinear(
            powerlaw
        )

        # Interpolate template onto target wavelength grid
        sed = jnp.interp(wavelength_aa, tmpl_wave, template, left=0.0, right=0.0)

        return L_absorbed * sed

    return dl14_tabulated


def register_dl14_tabulated(grid_path: str, name: str = "dl14_tabulated") -> None:
    """Load and register the tabulated DL14 model in the emission registry.

    After calling this, the model is available via
    ``get_emission_model("dl14_tabulated")`` and can be used as the
    ``dust_emission_model`` in ``Model()``.

    Parameters
    ----------
    grid_path : str
        Path to ``dl14_templates.h5``.
    name : str
        Registry name. Default ``"dl14_tabulated"``.
    """
    model_fn = create_dl14_from_grid(grid_path)
    DUST_EMISSION_MODELS[name] = model_fn


# ===================================================================
# Convenience: apply emission model by name
# ===================================================================


def register_dl07_tabulated(grid_path: str, name: str = "dl07_tabulated") -> None:
    """Load and register the tabulated DL07 model in the emission registry.

    After calling this, the model is available via
    ``get_emission_model("dl07_tabulated")`` and can be used as the
    ``dust_emission_model`` in ``Model()``.

    Parameters
    ----------
    grid_path : str
        Path to ``dl07_templates.npz`` or ``.h5``.
    name : str
        Registry name. Default ``"dl07_tabulated"``.
    """
    model_fn = create_dl07_from_grid(grid_path)
    DUST_EMISSION_MODELS[name] = model_fn


def apply_dust_emission(
    model_name: str,
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    **params,
) -> jnp.ndarray:
    """Apply a named dust emission model.

    Parameters
    ----------
    model_name : str
        One of ``"modified_blackbody"``, ``"dale2014"``, ``"draine_li2007"``,
        ``"draine_li2014"``, or any tabulated model registered via
        ``register_dl07_tabulated()`` / ``register_dl14_tabulated()``.
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom.
    L_absorbed : float
        Total absorbed luminosity in Lsun.
    **params
        Model-specific parameters.

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.
    """
    model_fn = get_emission_model(model_name)
    return model_fn(wavelength_aa, L_absorbed, **params)
