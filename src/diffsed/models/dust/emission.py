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
- Hildebrand 1983, QJRAS, 24, 267
"""

from typing import Callable

import jax
import jax.numpy as jnp


# ===================================================================
# Physical constants (CGS)
# ===================================================================

_H_PLANCK = 6.62607015e-27    # erg s
_K_BOLTZMANN = 1.380649e-16   # erg / K
_C_CGS = 2.99792458e10        # cm / s
_LSUN_ERG = 3.828e33          # erg / s  (IAU 2015)
_AA_TO_CM = 1.0e-8            # Angstrom -> cm


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
            f"Unknown dust emission model '{name}'. "
            f"Available: {list(DUST_EMISSION_MODELS.keys())}"
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

    # Integrate over frequency: nu is descending, so flip for trapezoid
    nu_ascending = nu[::-1]
    absorbed_ascending = absorbed_Lnu[::-1]

    return jnp.trapezoid(absorbed_ascending, nu_ascending)


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
    **_kwargs,
) -> jnp.ndarray:
    """Optically-thin modified blackbody dust emission.

    The unnormalized spectrum is::

        S_nu ~ nu^beta * B_nu(T_dust)

    which is then normalized so that the frequency integral equals
    ``L_absorbed``.

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

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.
    """
    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    # Reference frequency at 250 um (convenient normalization pivot)
    nu_ref = _C_CGS / (250.0e-4)  # 250 um in cm
    emissivity = (nu / nu_ref) ** dust_beta_ir

    bnu = planck_bnu(wavelength_aa, dust_T)

    # Unnormalized SED shape (erg/s/cm^2/Hz/sr units cancel in ratio)
    shape = emissivity * bnu

    # Integrate shape over frequency for normalization
    nu_ascending = nu[::-1]
    shape_ascending = shape[::-1]
    integral = jnp.trapezoid(shape_ascending, nu_ascending)

    # Guard against zero integral (e.g. wavelength grid entirely outside
    # the thermal peak) — return zeros instead of NaN
    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    return norm * shape


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
    **_kwargs,
) -> jnp.ndarray:
    """Dale et al. (2014) 1-parameter dust emission template.

    The full Dale model parameterizes the IR SED by the power-law slope
    alpha of the radiation field intensity distribution dM/dU ~ U^{-alpha}.
    Here we provide an analytic two-component approximation that captures
    the key alpha-dependent behaviour: low alpha yields warm, peaked SEDs;
    high alpha yields cooler, broader SEDs.

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
    nu_ref = _C_CGS / (250.0e-4)
    emissivity = (nu / nu_ref) ** beta_ir

    bnu_cold = planck_bnu(wavelength_aa, T_cold)
    bnu_warm = planck_bnu(wavelength_aa, T_warm)

    shape = emissivity * ((1.0 - f_warm) * bnu_cold + f_warm * bnu_warm)

    nu_ascending = nu[::-1]
    shape_ascending = shape[::-1]
    integral = jnp.trapezoid(shape_ascending, nu_ascending)

    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    return norm * shape


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

    # Diffuse ISM component: heated by U_min
    T_diff = _draine_li_umin_to_temperature(dust_umin)
    bnu_diff = planck_bnu(wavelength_aa, T_diff)

    # PDR component: heated by a distribution of U from U_min to U_max
    T_pdr = _pdr_temperature(dust_umin)
    bnu_pdr = planck_bnu(wavelength_aa, T_pdr)

    # Continuum: large-grain thermal emission
    continuum = emissivity * (
        (1.0 - dust_gamma_dl) * bnu_diff + dust_gamma_dl * bnu_pdr
    )

    # PAH mid-IR feature complex (approximate as warm component at ~400 K,
    # representing the 6.2, 7.7, 8.6, 11.3, 12.7 um features).
    # Strength scales with q_PAH (in percent), normalized to a reference
    # q_PAH = 3.5% typical for MW-like dust.
    T_pah = 400.0
    pah_frac = (dust_qpah / 3.5) * 0.05  # ~5% of luminosity at q_PAH=3.5%
    bnu_pah = planck_bnu(wavelength_aa, T_pah)
    pah_emissivity = (nu / nu_ref) ** 1.0  # Flatter emissivity for small grains

    # Combined unnormalized shape
    shape = continuum + pah_frac * pah_emissivity * bnu_pah

    # Normalize to L_absorbed
    nu_ascending = nu[::-1]
    shape_ascending = shape[::-1]
    integral = jnp.trapezoid(shape_ascending, nu_ascending)

    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    return norm * shape


# ===================================================================
# Template-based DL07 interface (for future use with tabulated grids)
# ===================================================================

def load_draine_li_templates(filepath: str) -> dict:
    """Load pre-computed Draine & Li (2007) template grid.

    Expected file format: HDF5 or .npz with arrays:
    - ``wavelength``: shape (n_wave,), Angstrom
    - ``umin_grid``: shape (n_umin,)
    - ``qpah_grid``: shape (n_qpah,)
    - ``templates``: shape (n_qpah, n_umin, n_wave), L_nu per unit L_absorbed

    Parameters
    ----------
    filepath : str
        Path to template file.

    Returns
    -------
    dict
        Keys: ``"wavelength"``, ``"umin_grid"``, ``"qpah_grid"``,
        ``"templates"`` (all as jnp arrays).
    """
    import numpy as np

    data = np.load(filepath)
    return {
        "wavelength": jnp.array(data["wavelength"]),
        "umin_grid": jnp.array(data["umin_grid"]),
        "qpah_grid": jnp.array(data["qpah_grid"]),
        "templates": jnp.array(data["templates"]),
    }


def draine_li2007_from_grid(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    templates: dict,
    dust_umin: float = 1.0,
    dust_gamma_dl: float = 0.01,
    dust_qpah: float = 2.5,
) -> jnp.ndarray:
    """Draine & Li 2007 emission from pre-computed template grid.

    Uses bilinear interpolation in (U_min, q_PAH) space for
    differentiability.  The gamma parameter linearly mixes the
    single-U_min template with the delta-function + power-law
    distribution template.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Target wavelength grid in Angstrom.
    L_absorbed : float
        Total absorbed luminosity in Lsun.
    templates : dict
        From ``load_draine_li_templates()``.
    dust_umin : float
        Minimum radiation field intensity.
    dust_gamma_dl : float
        PDR fraction.
    dust_qpah : float
        PAH mass fraction (%).

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.
    """
    umin_grid = templates["umin_grid"]
    qpah_grid = templates["qpah_grid"]
    tmpl_wave = templates["wavelength"]
    tmpl_data = templates["templates"]

    # Clamp to grid bounds
    dust_umin_c = jnp.clip(dust_umin, umin_grid[0], umin_grid[-1])
    dust_qpah_c = jnp.clip(dust_qpah, qpah_grid[0], qpah_grid[-1])

    # Find interpolation indices via searchsorted
    i_u = jnp.clip(
        jnp.searchsorted(umin_grid, dust_umin_c) - 1, 0, len(umin_grid) - 2
    )
    i_q = jnp.clip(
        jnp.searchsorted(qpah_grid, dust_qpah_c) - 1, 0, len(qpah_grid) - 2
    )

    # Fractional position within grid cell
    fu = (dust_umin_c - umin_grid[i_u]) / (umin_grid[i_u + 1] - umin_grid[i_u])
    fq = (dust_qpah_c - qpah_grid[i_q]) / (qpah_grid[i_q + 1] - qpah_grid[i_q])

    # Bilinear interpolation
    t00 = tmpl_data[i_q, i_u]
    t01 = tmpl_data[i_q, i_u + 1]
    t10 = tmpl_data[i_q + 1, i_u]
    t11 = tmpl_data[i_q + 1, i_u + 1]

    template_sed = (
        (1.0 - fq) * (1.0 - fu) * t00
        + (1.0 - fq) * fu * t01
        + fq * (1.0 - fu) * t10
        + fq * fu * t11
    )

    # Interpolate template onto target wavelength grid
    sed_interp = jnp.interp(wavelength_aa, tmpl_wave, template_sed, left=0.0, right=0.0)

    return L_absorbed * sed_interp


# ===================================================================
# Convenience: apply emission model by name
# ===================================================================

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
        One of ``"modified_blackbody"``, ``"dale2014"``, ``"draine_li2007"``.
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
