"""Dust emission models for tengri.

This module implements IR re-emission of UV/optical light absorbed by dust.
All models are pure JAX (JIT-compatible, fully differentiable) and follow
the energy-balance constraint: total IR luminosity equals total absorbed
luminosity from the attenuation step.

Available Emission Models
-------------------------
- **modified_blackbody**: Optically-thin modified blackbody (2-3 params)
- **dale2014**: Dale et al. (2014) 1-parameter IR template family (tabulated)
- **draine_li2007**: Draine & Li (2007) 3-parameter model (tabulated)
- **draine_li2014**: Draine & Li (2014 update) 4-parameter model (tabulated)

Template Auto-Loading
---------------------
The ``"draine_li2007"``, ``"dale2014"``, and ``"draine_li2014"`` models
auto-load tabulated templates from the ``data/`` directory on first use.
If templates are not found, they fall back to analytic approximations with
a warning.  The analytic fallbacks are crude (single-Gaussian PAH, hand-tuned
temperatures) and should NOT be used for science.

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

import warnings
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp

# ===================================================================
# Template search paths (resolved once, reused for all models)
# ===================================================================

_DATA_CANDIDATES = [
    Path(__file__).resolve().parents[4] / "data",
    Path("data"),
]


def _find_data_file(filename: str) -> str | None:
    """Search standard data directories for a template file."""
    for d in _DATA_CANDIDATES:
        candidate = d / filename
        if candidate.is_file():
            return str(candidate)
    return None


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
# Model 1b: Casey (2012) modified blackbody + mid-IR power law
# ===================================================================

# Empirical coefficients for turnover wavelength (Casey 2012, Eq. 3, errata)
_CASEY_B1_UM = 26.68  # μm
_CASEY_B2_UM_PER_K = 6.246e-3  # μm / K


@register_emission_model("casey2012")
def casey2012(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_T: float = 35.0,
    dust_beta_ir: float = 1.8,
    dust_alpha_mir: float = 2.0,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Casey (2012) modified blackbody + mid-IR power law dust emission.

    Combines a modified blackbody (FIR peak from cold/warm dust) with a
    mid-IR power law (Wien-side excess from warm dust continuum), joined
    by a smooth sigmoid transition function.  The key advantage over a
    pure modified blackbody is capturing the 8--40 μm excess seen in real
    galaxy SEDs.

    The model is (Casey 2012, MNRAS, 425, 3094, Eq. 2)::

        S(ν) = N_pl * ν^α_mid * f(ν)
             + N_bb * ν^(3+β) / (exp(hν/kT) - 1) * (1 - f(ν))

    where the transition function is::

        f(ν) = 1 / (1 + (λ_0 / λ)^2)

    and the empirical turnover wavelength is (Eq. 3, with errata)::

        λ_0 = b1 + b2 * T   [μm]

    with ``b1 = 26.68 μm``, ``b2 = 6.246e-3 μm/K``.

    Both components are normalized so that the total frequency integral
    equals ``L_absorbed``.

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
        Dust temperature in Kelvin.  Typical range: 25--60 K.
    dust_beta_ir : float
        Dust emissivity index for the MBB component.
        Typical range: 1.5--2.0.
    dust_alpha_mir : float
        Mid-IR power-law slope.  Typical range: 1.5--2.5.
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default 0 (no correction).

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.

    References
    ----------
    Casey, C. M., 2012, MNRAS, 425, 3094.
    da Cunha, E. et al., 2013, ApJ, 766, 13 (CMB corrections).
    """
    # CMB correction (no-op at z=0)
    T_eff = cmb_corrected_temperature(dust_T, redshift, dust_beta_ir)

    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm  # Hz, descending

    # Empirical turnover wavelength (Casey 2012, Eq. 3 with errata)
    lambda0_cm = (_CASEY_B1_UM + _CASEY_B2_UM_PER_K * T_eff) * 1.0e-4  # μm -> cm

    # Transition function: f(ν) = 1 / (1 + (λ_0/λ)^2)
    # λ_0/λ = ν / ν_0 where ν_0 = c / λ_0
    nu0 = _C_CGS / lambda0_cm
    f_transition = 1.0 / (1.0 + (nu / nu0) ** 2)

    # --- Mid-IR power-law component ---
    # S_pl(ν) ~ ν^α_mid * f(ν)
    # Use a reference frequency to keep the amplitude sensible
    nu_ref = _C_CGS / (100.0e-4)  # 100 μm pivot in Hz
    power_law = (nu / nu_ref) ** dust_alpha_mir * f_transition

    # --- Modified blackbody component ---
    # S_bb(ν) ~ ν^(3+β) / (exp(hν/kT) - 1) * (1 - f(ν))
    x = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * T_eff), 0.0, 500.0)
    mbb = (nu / nu_ref) ** (3.0 + dust_beta_ir) / (jnp.exp(x) - 1.0)
    mbb = mbb * (1.0 - f_transition)

    # Combined unnormalized shape
    shape = power_law + mbb

    # Normalize so integral over frequency = L_absorbed
    # nu is descending (wave ascending), negate for positive integral
    integral = -jnp.trapezoid(shape, nu)
    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    result = norm * shape

    # CMB contrast suppression
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


def _dale2014_analytic_fallback(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_alpha_dale: float = 2.0,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Dale et al. (2014) ANALYTIC FALLBACK — not for science.

    .. deprecated::
        This crude approximation replaces the full Dale template library
        with two hand-tuned modified blackbodies.  Use the tabulated
        version (the default ``"dale2014"`` registry entry, which
        auto-loads ``data/dale2014_templates.npz``).

    Only used when template files are not found.
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


def _measure_dl07_pah_fraction(grid_path: str) -> float:
    """Measure the PAH luminosity fraction from tabulated DL07 templates.

    Computes the fraction of total L_nu in the 5--15 um PAH band for the
    reference parameters qpah=2.5%, umin=1.0, gamma=0.01.  This value
    calibrates the analytic DL07 model's PAH component.

    Parameters
    ----------
    grid_path : str
        Path to ``dl07_templates.npz`` (or ``.h5``).

    Returns
    -------
    float
        PAH luminosity fraction (integrated 5--15 um / total).
    """
    import numpy as np

    data = np.load(grid_path)
    wavs = data["wavelength"]
    umin_grid = data["umin_grid"]
    qpah_grid = data["qpah_grid"]
    single_u = data["templates_umin_only"]
    powerlaw = data["templates_umin_umax"]

    # Reference indices: qpah ~ 2.5%, umin ~ 1.0
    i_q = int(np.argmin(np.abs(qpah_grid - 2.5)))
    i_u = int(np.argmin(np.abs(umin_grid - 1.0)))

    # Mixed template: (1-gamma)*single_u + gamma*powerlaw, gamma=0.01
    tmpl = 0.99 * single_u[i_q, i_u] + 0.01 * powerlaw[i_q, i_u]

    # Normalize in L_lambda space
    norm = np.trapezoid(tmpl, wavs)
    if norm > 0:
        tmpl = tmpl / norm

    # Convert to L_nu: L_nu = L_lambda * lambda^2 / c
    wave_cm = wavs * 1.0e-8
    nu = 2.99792458e10 / wave_cm
    tmpl_lnu = tmpl * wave_cm**2 / 2.99792458e10

    # Normalize in L_nu space (nu descending -> negate)
    int_lnu = -np.trapezoid(tmpl_lnu, nu)
    if int_lnu > 0:
        tmpl_lnu = tmpl_lnu / int_lnu

    # PAH band: 5-15 um (50000-150000 Angstrom)
    pah_mask = (wavs >= 50000) & (wavs <= 150000)
    pah_frac = float(-np.trapezoid(tmpl_lnu[pah_mask], nu[pah_mask]))

    return pah_frac


# Lazy-loaded calibration: measured from DL07 tabulated templates at
# the reference point qpah=2.5%, umin=1.0, gamma=0.01.
# Call calibrate_dl07_pah_fraction() to update from your grid file.
_DL07_PAH_FRAC_REF: float | None = None

# Reference qpah for the calibration measurement
_DL07_QPAH_REF: float = 2.5


def calibrate_dl07_pah_fraction(grid_path: str) -> float:
    """Measure and cache the DL07 PAH fraction from tabulated templates.

    After calling this, the analytic ``draine_li2007()`` will use the
    calibrated f_pah instead of the default approximation.

    Parameters
    ----------
    grid_path : str
        Path to ``dl07_templates.npz`` (or ``.h5``).

    Returns
    -------
    float
        The measured PAH luminosity fraction at qpah=2.5%, umin=1.0.
    """
    global _DL07_PAH_FRAC_REF
    _DL07_PAH_FRAC_REF = _measure_dl07_pah_fraction(grid_path)
    return _DL07_PAH_FRAC_REF


def _get_dl07_pah_frac_at_ref() -> float:
    """Return the calibrated PAH fraction, or a reasonable default.

    The default (0.10) is a conservative estimate that produces FIR-dominated
    emission consistent with the tabulated templates.  For accurate results,
    call ``calibrate_dl07_pah_fraction()`` with the DL07 grid path.
    """
    if _DL07_PAH_FRAC_REF is not None:
        return _DL07_PAH_FRAC_REF
    # Fallback: measured from DL07 templates (qpah=2.5%, umin=1.0, gamma=0.01)
    # PAH band (5-15 um) contains ~10% of total L_nu.  The old value of 0.25
    # was too high, pulling the centroid to ~40 um vs the correct ~100-200 um.
    return 0.10


def _draine_li2007_analytic_fallback(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_umin: float = 1.0,
    dust_gamma_dl: float = 0.01,
    dust_qpah: float = 2.5,
    **_kwargs,
) -> jnp.ndarray:
    """Draine & Li (2007) ANALYTIC FALLBACK — not for science.

    .. deprecated::
        This crude approximation models the entire PAH feature complex as
        a single Gaussian at 7.7 μm and uses hand-tuned temperature
        formulas.  Use the tabulated version (the default ``"draine_li2007"``
        registry entry, which auto-loads ``data/dl07_templates.npz``).

    Only used when template files are not found.
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
    # PAH fraction calibrated against tabulated DL07 templates:
    # At the reference point (qpah=2.5%, umin=1.0, gamma=0.01), the
    # 5-15 um PAH band contains ~10% of total L_nu.  Scale linearly
    # with qpah relative to the reference qpah.
    f_pah_ref = _get_dl07_pah_frac_at_ref()
    f_pah = (dust_qpah / _DL07_QPAH_REF) * f_pah_ref
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

        Templates are in L_lambda convention (normalized to integrate to
        1 over wavelength).  This function converts to L_nu (Lsun/Hz)
        and scales by L_absorbed to enforce energy balance.

        Returns L_nu in Lsun/Hz.
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
        # Template is in L_lambda space (integral over wavelength = 1)
        sed_llam = jnp.interp(wavelength_aa, tmpl_wave, template, left=0.0, right=0.0)

        # Convert L_lambda -> L_nu: L_nu = L_lambda * lambda^2 / c
        wavelength_cm = wavelength_aa * _AA_TO_CM
        nu = _C_CGS / wavelength_cm
        sed_lnu = sed_llam * (wavelength_cm**2) / _C_CGS

        # Renormalize so that integral(L_nu, d_nu) = L_absorbed
        # nu is descending (wavelength ascending), so negate
        integral = -jnp.trapezoid(sed_lnu, nu)
        norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

        return norm * sed_lnu

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
# Template-based Dale+2014: create from grid file
# ===================================================================


def create_dale2014_from_grid(grid_path: str) -> Callable:
    """Create a Dale+2014 emission model backed by tabulated templates.

    Loads the NPZ grid once and returns a function matching the emission
    model registry interface.  The NPZ file must contain:

    - ``wavelength_aa``: rest-frame wavelength grid in Angstrom (n_wave,)
    - ``alpha_grid``: array of alpha values (n_alpha,)
    - ``templates_sf``: star-forming templates (n_alpha, n_wave) in
      L_lambda units (will be normalized internally so that each
      template integrates to 1 over frequency).

    The returned function performs 1-D linear interpolation in alpha
    and normalizes the result so that the frequency integral equals
    ``L_absorbed``.

    Parameters
    ----------
    grid_path : str
        Path to ``dale2014_templates.npz``.

    Returns
    -------
    Callable
        Model function with signature
        ``(wavelength_aa, L_absorbed, dust_alpha_dale=2.0, **kw) -> L_nu``.

    Example
    -------
    >>> dale = create_dale2014_from_grid("data/dale2014_templates.npz")
    >>> DUST_EMISSION_MODELS["dale2014_tabulated"] = dale
    >>> sed = dale(wav, L_abs, dust_alpha_dale=1.5)
    """
    import numpy as np

    data = np.load(grid_path)
    tmpl_wave = jnp.array(data["wavelength_aa"])  # (n_wave,)
    alpha_grid = jnp.array(data["alpha_grid"])  # (n_alpha,)
    templates_raw = np.array(data["templates_sf"])
    # Handle both (n_alpha, n_wave) and (n_wave, n_alpha) layouts
    if templates_raw.shape[0] == len(tmpl_wave) and templates_raw.shape[1] == len(alpha_grid):
        templates_raw = templates_raw.T  # -> (n_alpha, n_wave)

    # Convert from L_lambda to L_nu: L_nu = L_lambda * lambda^2 / c
    wave_cm = np.array(tmpl_wave) * _AA_TO_CM
    nu = _C_CGS / wave_cm  # descending for ascending wavelengths

    templates_lnu = templates_raw * (wave_cm**2)[None, :] / _C_CGS

    # Normalize each template so that integral(L_nu, dnu) = 1
    # nu is descending, so negate for positive integral
    for i in range(templates_lnu.shape[0]):
        integral = -np.trapezoid(templates_lnu[i], nu)
        if integral > 0:
            templates_lnu[i] /= integral

    templates = jnp.array(templates_lnu)  # (n_alpha, n_wave)

    def dale2014_tabulated(
        wavelength_aa: jnp.ndarray,
        L_absorbed: float,
        dust_alpha_dale: float = 2.0,
        **_kwargs,
    ) -> jnp.ndarray:
        """Dale+2014 emission from tabulated templates.

        Performs 1-D linear interpolation in alpha and normalizes
        the template so the frequency integral equals L_absorbed.

        Parameters
        ----------
        wavelength_aa : array, shape (n_wave,)
            Target wavelength grid in Angstrom (sorted ascending).
        L_absorbed : float
            Total absorbed luminosity in Lsun.
        dust_alpha_dale : float
            Power-law slope of the radiation field distribution.
            Valid range determined by the grid (typically 0.0625--4.0).

        Returns
        -------
        array, shape (n_wave,)
            Dust emission L_nu in Lsun/Hz.
        """
        alpha_c = jnp.clip(dust_alpha_dale, alpha_grid[0], alpha_grid[-1])

        # Linear interpolation index
        i_a = jnp.clip(
            jnp.searchsorted(alpha_grid, alpha_c) - 1,
            0,
            len(alpha_grid) - 2,
        )
        fa = (alpha_c - alpha_grid[i_a]) / (alpha_grid[i_a + 1] - alpha_grid[i_a])

        template = (1.0 - fa) * templates[i_a] + fa * templates[i_a + 1]

        # Interpolate onto target wavelength grid
        sed = jnp.interp(wavelength_aa, tmpl_wave, template, left=0.0, right=0.0)

        return L_absorbed * sed

    return dale2014_tabulated


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


def _draine_li2014_analytic_fallback(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_umin: float = 1.0,
    dust_gamma_dl: float = 0.01,
    dust_qpah: float = 2.5,
    dust_alpha_dl14: float = 2.0,
    **_kwargs,
) -> jnp.ndarray:
    """Draine & Li (2014 update) ANALYTIC FALLBACK — not for science.

    .. deprecated::
        This crude approximation uses single-Gaussian PAH and hand-tuned
        temperature formulas.  Use the tabulated version (the default
        ``"draine_li2014"`` registry entry, which auto-loads
        ``data/dl14_templates.h5``).

    Only used when template files are not found.
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
    # PAH fraction calibrated against DL07 tabulated templates (same
    # grain model).  See calibrate_dl07_pah_fraction() and draine_li2007().
    f_pah_ref = _get_dl07_pah_frac_at_ref()
    f_pah = (dust_qpah / _DL07_QPAH_REF) * f_pah_ref
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


def register_dale2014_tabulated(grid_path: str, name: str = "dale2014_tabulated") -> None:
    """Load and register the tabulated Dale+2014 model in the emission registry.

    After calling this, the model is available via
    ``get_emission_model("dale2014_tabulated")`` and can be used as the
    ``dust_emission_model`` in ``Model()``.

    Parameters
    ----------
    grid_path : str
        Path to ``dale2014_templates.npz``.
    name : str
        Registry name. Default ``"dale2014_tabulated"``.
    """
    model_fn = create_dale2014_from_grid(grid_path)
    DUST_EMISSION_MODELS[name] = model_fn


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


# ===================================================================
# Lazy auto-loading: template-based models as defaults
# ===================================================================
#
# On first call, each model tries to load tabulated templates from
# data/.  If found, the template-based version replaces itself in
# the registry.  If not found, falls back to the analytic
# approximation with a loud warning.
#
# This gives:
#   - Zero-config: ``dust_emission="draine_li2007"`` just works
#   - Fast import: no I/O at import time
#   - Correct physics: tabulated templates used by default
# ===================================================================

# Track which models have been resolved (to avoid repeated warnings)
_resolved: set[str] = set()


def _make_lazy_loader(
    name: str,
    template_filename: str,
    loader_fn_name: str,
    fallback_fn: Callable,
) -> Callable:
    """Create a lazy-loading wrapper that auto-loads templates on first call.

    Parameters
    ----------
    name : str
        Registry name (e.g. ``"draine_li2007"``).
    template_filename : str
        Filename to search for in data/ (e.g. ``"dl07_templates.npz"``).
    loader_fn_name : str
        Name of the ``create_*_from_grid`` function in this module.
    fallback_fn : Callable
        The analytic fallback function.
    """

    def _lazy_wrapper(*args, **kwargs):
        if name not in _resolved:
            _resolved.add(name)
            path = _find_data_file(template_filename)
            if path is not None:
                try:
                    loader = globals()[loader_fn_name]
                    tabulated = loader(path)
                    DUST_EMISSION_MODELS[name] = tabulated
                    return tabulated(*args, **kwargs)
                except Exception as e:
                    warnings.warn(
                        f"Failed to load {template_filename}: {e}. "
                        f"Falling back to analytic {name} (NOT suitable for "
                        f"science — crude single-Gaussian PAH approximation).",
                        stacklevel=2,
                    )
                    DUST_EMISSION_MODELS[name] = fallback_fn
            else:
                warnings.warn(
                    f"Template file '{template_filename}' not found in data/. "
                    f"Falling back to analytic {name} (NOT suitable for "
                    f"science — crude approximation with hand-tuned "
                    f"temperatures). Download templates or set the path "
                    f"manually via register_*_tabulated().",
                    stacklevel=2,
                )
                DUST_EMISSION_MODELS[name] = fallback_fn
        return DUST_EMISSION_MODELS[name](*args, **kwargs)

    _lazy_wrapper.__name__ = name
    _lazy_wrapper.__doc__ = (
        f"Lazy-loading wrapper for {name}. Auto-loads tabulated templates "
        f"from data/{template_filename} on first call."
    )
    return _lazy_wrapper


# --- DL07: tries dl07_templates.npz, then .h5 ---
def _find_dl07_templates() -> str | None:
    for fn in ("dl07_templates.npz", "dl07_templates.h5"):
        path = _find_data_file(fn)
        if path is not None:
            return path
    return None


def _dl07_lazy_wrapper(*args, **kwargs):
    """Draine & Li (2007) — auto-loads tabulated templates on first call."""
    if "draine_li2007" not in _resolved:
        _resolved.add("draine_li2007")
        path = _find_dl07_templates()
        if path is not None:
            try:
                tabulated = create_dl07_from_grid(path)
                DUST_EMISSION_MODELS["draine_li2007"] = tabulated
                DUST_EMISSION_MODELS["dl07_tabulated"] = tabulated
                return tabulated(*args, **kwargs)
            except Exception as e:
                warnings.warn(
                    f"Failed to load DL07 templates: {e}. "
                    f"Falling back to analytic DL07 (NOT suitable for "
                    f"science — crude single-Gaussian PAH approximation).",
                    stacklevel=2,
                )
                DUST_EMISSION_MODELS["draine_li2007"] = _draine_li2007_analytic_fallback
        else:
            warnings.warn(
                "DL07 template files (dl07_templates.npz/.h5) not found "
                "in data/. Falling back to analytic DL07 (NOT suitable "
                "for science). Run: python scripts/convert_dl07_templates.py",
                stacklevel=2,
            )
            DUST_EMISSION_MODELS["draine_li2007"] = _draine_li2007_analytic_fallback
    return DUST_EMISSION_MODELS["draine_li2007"](*args, **kwargs)


DUST_EMISSION_MODELS["draine_li2007"] = _dl07_lazy_wrapper


# --- Dale+2014: tries dale2014_templates.npz ---
DUST_EMISSION_MODELS["dale2014"] = _make_lazy_loader(
    "dale2014",
    "dale2014_templates.npz",
    "create_dale2014_from_grid",
    _dale2014_analytic_fallback,
)


# --- DL14: tries dl14_templates.h5 ---
DUST_EMISSION_MODELS["draine_li2014"] = _make_lazy_loader(
    "draine_li2014",
    "dl14_templates.h5",
    "create_dl14_from_grid",
    _draine_li2014_analytic_fallback,
)


# ===================================================================
# Backward-compatible module-level aliases for direct imports
# ===================================================================
# Tests and user code may do ``from tengri.models.dust.emission import draine_li2007``.
# These aliases point to the lazy wrappers (which auto-load templates on call).


def draine_li2007(*args, **kwargs):
    """Draine & Li (2007) — dispatches to the registry (auto-loads templates)."""
    return DUST_EMISSION_MODELS["draine_li2007"](*args, **kwargs)


def dale2014(*args, **kwargs):
    """Dale et al. (2014) — dispatches to the registry (auto-loads templates)."""
    return DUST_EMISSION_MODELS["dale2014"](*args, **kwargs)


def draine_li2014(*args, **kwargs):
    """Draine & Li (2014) — dispatches to the registry (auto-loads templates)."""
    return DUST_EMISSION_MODELS["draine_li2014"](*args, **kwargs)
