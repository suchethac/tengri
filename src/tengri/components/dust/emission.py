"""Dust emission models for tengri.

This module implements IR re-emission of UV/optical light absorbed by dust.
All models are pure JAX (JIT-compatible, fully differentiable) and follow
the energy-balance constraint: total IR luminosity equals total absorbed
luminosity from the attenuation step.

Available Emission Models
-------------------------
- **modified_blackbody**: Optically-thin modified blackbody (2-3 params)
- **casey2012**: Casey (2012) modified blackbody + mid-IR power law (3 params)
- **pah_drude**: Smith et al. (2007) PAH Drude profiles (0 params, pure template)
- **dale2014**: Dale et al. (2014) 1-parameter IR template family (tabulated)
- **draine_li2007**: Draine & Li (2007) 3-parameter model (tabulated)
- **draine_li2014**: Draine & Li (2014 update) 4-parameter model (tabulated)
- **astrodust**: Hensley & Draine (2023) Astrodust+PAH model (tabulated)
- **bosa**: Boquien & Salim (2021) (L_TIR, sSFR)-parameterized model (tabulated)
- **themis**: Jones et al. (2017) THEMIS/DustEM model (tabulated)

Template Auto-Loading
---------------------
The ``"draine_li2007"``, ``"dale2014"``, ``"draine_li2014"``,
``"astrodust"``, ``"bosa"``, and ``"themis"`` models auto-load tabulated
templates from the ``data/`` directory on first use.  If templates are not
found, they fall back to analytic approximations with a warning.  The
analytic fallbacks are crude (single-Gaussian PAH, hand-tuned temperatures)
and should NOT be used for science.

Energy Balance
--------------
The normalization for every model is set by::

    L_dust_emission = L_dust_absorbed
                    = integral[(1 - transmission) * L_stellar_intrinsic * dlambda]

This is computed from the attenuation step and passed to each model as
``L_absorbed`` (scalar, in Lsun).

References
----------
- Casey 2012, MNRAS, 425, 3094
- Dale et al. 2014, ApJ, 784, 83
- Draine & Li 2007, ApJ, 657, 810
- Draine & Li 2014 update (CIGALE implementation, Boquien+2019)
- Aniano et al. 2012, ApJ, 756, 138
- da Cunha et al. 2013, ApJ, 766, 13
- Hildebrand 1983, QJRAS, 24, 267
- Hensley & Draine 2023, ApJ, 948, 55 (Astrodust+PAH)
- Boquien & Salim 2021, A&A, 653, A149 (BOSA templates)
- Jones et al. 2017, A&A, 602, A46 (THEMIS dust model)

"""

import contextlib
import functools
from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp

from tengri.utils.physics_constants import (
    AA_TO_CM as _AA_TO_CM,
    C_CGS as _C_CGS,
    H_PLANCK as _H_PLANCK,
    K_BOLTZ as _K_BOLTZMANN,
)

# ── Template search paths (resolved once, reused for all models) ──

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


# ── Emission model registry ───────────────────────────────────────

DUST_EMISSION_MODELS: dict[str, Callable] = {}

# Track which lazy loaders have been resolved to avoid duplicate loading
_resolved: set[str] = set()


def register_emission_model(name: str) -> Callable:
    """Decorator factory that registers a dust emission model under a name.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"dale2014"``, ``"draine_li2007"``).

    Returns
    -------
    Callable
        Decorator that registers the decorated function and returns it unchanged.

    Notes
    -----
    **JIT-compatible**: no — registration happens at factory time before JIT.

    Decorated functions must implement the ``DustEmissionTemplate`` protocol:
    accept a wavelength array, absorbed luminosity, and keyword arguments,
    returning an emission SED ``L_ν`` [erg/s/Hz].

    """

    def decorator(fn: Callable) -> Callable:
        """Inner decorator that registers function in DUST_EMISSION_MODELS dict.

        Parameters
        ----------
        fn : Callable
            Dust emission model function matching ``DustEmissionTemplate`` protocol.

        Returns
        -------
        Callable
            The input function unchanged (enables use as a decorator).

        Notes
        -----
        **JIT-compatible**: depends on the decorated function.

        """
        DUST_EMISSION_MODELS[name] = fn
        return fn

    return decorator


def resolve_emission_model(name: str) -> Callable:
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

    Notes
    -----
    **JIT-compatible**: no — registry lookup happens at factory time.

    The returned function matches the ``DustEmissionTemplate`` protocol and can
    be called with wavelengths, absorbed luminosity, and model-specific parameters.

    """
    if name not in DUST_EMISSION_MODELS:
        raise ValueError(
            f"Unknown dust emission model '{name}'. Available: {list(DUST_EMISSION_MODELS.keys())}"
        )
    return DUST_EMISSION_MODELS[name]


def preload_emission_model(name: str) -> Callable:
    """Force lazy template loading outside any JAX JIT scope.

    Template-based emission models use lazy loaders that fire on first call.
    If the first call happens inside a ``@jax.jit`` scope, ``jnp.array()``
    inside the loader creates ``DynamicJaxprTracer`` objects that escape into
    closures, causing ``UnexpectedTracerError`` on subsequent non-JIT calls.

    Call this function at factory time (outside JIT) so templates are loaded
    into ``DUST_EMISSION_MODELS[name]`` as regular ``DeviceArray`` objects.
    Dynamic JAX indexing inside JIT then works correctly.

    Parameters
    ----------
    name : str
        Registry name (e.g. ``"draine_li2007"``).

    Returns
    -------
    Callable
        The loaded (real) emission function — NOT a lazy wrapper.

    Notes
    -----
    **JIT-compatible**: no — template loading happens at factory time.

    Safe to call at ``SEDModel.__init__`` time to prevent tracer leaks
    when models are first called inside a ``@jax.jit`` scope.

    """
    if name not in DUST_EMISSION_MODELS:
        raise ValueError(
            f"Unknown emission model '{name}'. Available: {list(DUST_EMISSION_MODELS.keys())}"
        )
    if name not in _resolved:
        # Trigger lazy loading with dummy inputs; ignore computation output —
        # we only want the side effect of loading templates into the registry.
        import numpy as _np

        _dummy_wave = _np.linspace(1e3, 1e7, 5, dtype=_np.float64)
        with contextlib.suppress(Exception):
            DUST_EMISSION_MODELS[name](_dummy_wave, 1.0)
    return DUST_EMISSION_MODELS[name]


# ── Utility: Planck function ──────────────────────────────────────


def planck_bnu(
    wavelength_aa: jnp.ndarray,
    temperature: float,
) -> jnp.ndarray:
    r"""Planck function B_ν(T) evaluated at given wavelengths.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    temperature : float
        Blackbody temperature. [K]

    Returns
    -------
    ndarray, shape (n_wave,)
        Planck brightness. [erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The Planck function is:

    .. math::

        B_\nu(T) = \frac{2 h \nu^3}{c^2} \frac{1}{\exp(h\nu / k_B T) - 1}

    where :math:`\nu = c / \lambda` is the frequency, :math:`h` is Planck's constant,
    :math:`k_B` is Boltzmann's constant, and :math:`c` is the speed of light.

    """
    # Cast to float64 before computing nu to prevent nu**3 overflow.
    # float32 max is ~3.4e38; at 5.6 Å, nu = 5.35e17 Hz so nu**3 ~ 1.5e53 —
    # far beyond float32 range. JAX weak-type promotion keeps float32 arrays
    # float32 even when combined with Python float scalars, so the cast must
    # be explicit even though x64 is enabled globally.
    wavelength_cm = jnp.asarray(wavelength_aa, dtype=jnp.float64) * _AA_TO_CM
    nu = _C_CGS / wavelength_cm

    # Clamp x = hν/kT to [1e-10, 500].  At x=500, expm1 ≈ 1.4e217
    # (finite in float64).  The clamp avoids both expm1 overflow and
    # division-by-zero, and keeps gradients finite everywhere.
    x = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * temperature), 1e-10, 500.0)
    return 2.0 * _H_PLANCK * nu**3 / (_C_CGS**2) / jnp.expm1(x)


# ── CMB heating correction (da Cunha+2013) ────────────────────────

_T_CMB_0 = 2.725  # CMB temperature at z=0 (K)


def cmb_corrected_temperature(
    T_dust: float,
    redshift: float,
    beta_ir: float = 1.6,
) -> float:
    r"""Effective dust temperature including CMB heating.

    At high redshift the CMB sets a temperature floor on dust grains.
    The effective equilibrium temperature is (da Cunha et al. 2013).

    Parameters
    ----------
    T_dust : float
        Intrinsic dust temperature (what the galaxy would have at z=0 in isolation). [K]
    redshift : float
        Source redshift. [dimensionless]
    beta_ir : float
        Dust emissivity index. [dimensionless] Default: 1.6.

    Returns
    -------
    float
        Effective dust temperature including CMB heating. [K]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The effective temperature is:

    .. math::

        T_{\rm eff} = \left[T_{\rm dust}^{4+\beta} + T_{\rm CMB}(z)^{4+\beta}
        - T_{\rm CMB}(z=0)^{4+\beta}\right]^{1/(4+\beta)}

    where :math:`T_{\rm CMB}(z) = T_{\rm CMB,0} (1 + z)` with :math:`T_{\rm CMB,0} = 2.725` K.

    """
    exponent = 4.0 + beta_ir
    T_cmb_z = _T_CMB_0 * (1.0 + redshift)
    # Clamp T_dust to positive values before raising to a fractional exponent.
    # Negative T_dust (possible during unconstrained sampling) would give NaN.
    T_dust_safe = jnp.maximum(T_dust, 1.0)
    inner = jnp.maximum(T_dust_safe**exponent + T_cmb_z**exponent - _T_CMB_0**exponent, 0.0)
    T_eff = inner ** (1.0 / exponent)
    return T_eff


def cmb_contrast_factor(
    wavelength_aa: jnp.ndarray,
    T_eff: float,
    redshift: float,
) -> jnp.ndarray:
    r"""Flux suppression factor from observing dust against the CMB.

    The observed flux is reduced because the galaxy's dust emission is
    measured against the CMB background (da Cunha et al. 2013).

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    T_eff : float
        CMB-corrected effective dust temperature. [K]
    redshift : float
        Source redshift. [dimensionless]

    Returns
    -------
    ndarray, shape (n_wave,)
        Multiplicative contrast factor in [0, 1]. [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The contrast factor is:

    .. math::

        C(\lambda) = 1 - \frac{B_\nu(T_{\rm CMB}(z))}{B_\nu(T_{\rm eff})}

    Since :math:`T_{\rm eff} > T_{\rm CMB}(z)`, we have :math:`0 \leq C(\lambda) \leq 1`.

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


# ── Energy balance ────────────────────────────────────────────────


def compute_absorbed_luminosity(
    wavelength_aa: jnp.ndarray,
    L_nu_intrinsic: jnp.ndarray,
    transmission: jnp.ndarray,
) -> float:
    r"""Compute total luminosity absorbed by dust.

    Integrates (1 - transmission) × L_nu over frequency to get the total
    absorbed energy, which must be re-emitted in the IR (energy balance).

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å] Must be sorted ascending.
    L_nu_intrinsic : array_like, shape (n_wave,)
        Intrinsic (dust-free) luminosity density. [Lsun Hz⁻¹]
    transmission : array_like, shape (n_wave,)
        Dust transmission fraction in [0, 1]. For age-dependent models
        this should be the SFH-weighted effective transmission.

    Returns
    -------
    float
        Total absorbed luminosity. [Lsun]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The absorbed luminosity is:

    .. math::

        L_{\rm absorbed} = \int [1 - T(\lambda)] L_\nu(\lambda) d\nu

    where the integral is over frequency (ν is descending as λ is ascending).

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
    r"""Compute total absorbed luminosity from optical depth.

    Convenience wrapper when you have τ(λ) rather than transmission T(λ).

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å] Must be sorted ascending.
    L_nu_intrinsic : array_like, shape (n_wave,)
        Intrinsic luminosity density. [Lsun Hz⁻¹]
    tau_lambda : array_like, shape (n_wave,)
        Optical depth as a function of wavelength. [dimensionless]

    Returns
    -------
    float
        Total absorbed luminosity. [Lsun]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    Internally converts τ(λ) to transmission via T(λ) = exp(−τ(λ))
    then calls ``compute_absorbed_luminosity``.

    """
    transmission = jnp.exp(-tau_lambda)
    return compute_absorbed_luminosity(wavelength_aa, L_nu_intrinsic, transmission)


# ── Model 1: Modified blackbody (2-3 parameters) ──────────────────


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
        Wavelength grid in Angstrom (sorted ascending). [Å]
    L_absorbed : float
        Total absorbed luminosity.  Unit-agnostic: the output L_nu will be
        in the same units per Hz (e.g. pass erg/s → get erg/s/Hz; pass
        Lsun → get Lsun/Hz).  In ``sed_pipeline.py`` the pipeline passes
        erg/s (from a frequency-integrated trapezoid) and receives erg/s/Hz.
    dust_T : float
        Dust temperature in Kelvin.  Typical range: 20--60 K. [K]
    dust_beta_ir : float
        Emissivity index.  Typical range: 1.5--2.0. [dimensionless]
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default 0 (no correction). [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in ``[L_absorbed units] / Hz``.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

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


# ── Model 1b: Casey (2012) modified blackbody + mid-IR power law ──

# Empirical coefficients for turnover wavelength (Casey 2012, Eq. 3, errata)
_CASEY_B1_UM = 26.68  # μm
_CASEY_B2_UM_PER_K = 6.246e-3  # μm / K


def _casey_transition_function(
    wavelength_cm: jnp.ndarray,
    T_eff: float,
) -> jnp.ndarray:
    """Compute Casey (2012) transition function between power law and MBB.

    Parameters
    ----------
    wavelength_cm : array, shape (n_wave,)
        Wavelength grid in cm. [cm]
    T_eff : float
        Effective dust temperature (already CMB-corrected). [K]

    Returns
    -------
    array, shape (n_wave,)
        Transition function f(λ) = 1 / (1 + (λ/λ_0)^2).
        f→1 at short λ (power law), f→0 at long λ (MBB). [dimensionless]

    Notes
    -----
    **JIT-compatible**: yes.

    """
    # Empirical turnover wavelength (Casey 2012, Eq. 3 with errata)
    lambda0_cm = (_CASEY_B1_UM + _CASEY_B2_UM_PER_K * T_eff) * 1.0e-4  # μm -> cm

    # Transition function: f(λ) = 1 / (1 + (λ/λ_0)^2)
    # f→1 at short λ (mid-IR power law dominates), f→0 at long λ (MBB dominates)
    # Casey 2012 convention: power law for Wien side, MBB for Rayleigh-Jeans
    return 1.0 / (1.0 + (wavelength_cm / lambda0_cm) ** 2)


def _casey_powerlaw_component(
    nu: jnp.ndarray,
    nu_ref: float,
    f_transition: jnp.ndarray,
    x: jnp.ndarray,
    dust_alpha_mir: float,
    optically_thin: bool,
) -> jnp.ndarray:
    """Compute Casey (2012) mid-IR power-law component.

    Parameters
    ----------
    nu : array, shape (n_wave,)
        Frequency in Hz (descending). [Hz]
    nu_ref : float
        Reference frequency (100 μm pivot). [Hz]
    f_transition : array, shape (n_wave,)
        Transition function from _casey_transition_function. [dimensionless]
    x : array, shape (n_wave,)
        Planck argument h*nu/(k*T). [dimensionless]
    dust_alpha_mir : float
        Mid-IR power-law slope. [dimensionless]
    optically_thin : bool
        If True, return zero (power law suppressed). [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Mid-IR power-law component S_pl(ν). [erg/s/Hz] (before normalization)

    Notes
    -----
    **JIT-compatible**: yes.

    The exponential cutoff prevents the power law from diverging at
    UV/optical wavelengths. This follows Casey (2012) Eq. 2 where the
    power law implicitly operates only in the IR regime.

    """
    # S_pl(ν) ~ ν^α_mid * f(ν) * exp(-hν/kT) [Wien cutoff]
    wien_cutoff = jnp.exp(-x)
    power_law = (nu / nu_ref) ** dust_alpha_mir * f_transition * wien_cutoff
    power_law = power_law * (1.0 - optically_thin)
    return power_law


def _casey_mbb_component(
    nu: jnp.ndarray,
    nu_ref: float,
    f_transition: jnp.ndarray,
    x: jnp.ndarray,
    dust_beta_ir: float,
) -> jnp.ndarray:
    """Compute Casey (2012) modified blackbody component.

    Parameters
    ----------
    nu : array, shape (n_wave,)
        Frequency in Hz (descending). [Hz]
    nu_ref : float
        Reference frequency (100 μm pivot). [Hz]
    f_transition : array, shape (n_wave,)
        Transition function from _casey_transition_function. [dimensionless]
    x : array, shape (n_wave,)
        Planck argument h*nu/(k*T). [dimensionless]
    dust_beta_ir : float
        Dust emissivity index. [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Modified blackbody component S_bb(ν). [erg/s/Hz] (before normalization)

    Notes
    -----
    **JIT-compatible**: yes.

    """
    # S_bb(ν) ~ ν^(3+β) / (exp(hν/kT) - 1) * (1 - f(ν))
    mbb = (nu / nu_ref) ** (3.0 + dust_beta_ir) / (jnp.exp(x) - 1.0)
    mbb = mbb * (1.0 - f_transition)
    return mbb


@register_emission_model("casey2012")
def casey2012(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_T: float = 35.0,
    dust_beta_ir: float = 1.8,
    dust_alpha_mir: float = 2.0,
    optically_thin: bool = False,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Casey (2012) modified blackbody + mid-IR power law dust emission.

    Combines a modified blackbody (FIR peak from cold/warm dust) with a
    mid-IR power law (Wien-side excess from warm dust continuum), joined
    by a smooth sigmoid transition function.

    When ``optically_thin=True``, the mid-IR power-law component is zeroed,
    leaving only the modified blackbody.  This variant is useful for cold
    dust-dominated galaxies where the power law is unphysical.

    .. note::

        The mid-IR power-law contribution is only significant for **warm/hot
        dust** (T ≳ 60 K).  For typical cold ISM dust (T = 25–60 K) the Wien
        cutoff exp(-hν/kT) kills the power-law component at 8–40 μm (x ≈ 10–51
        at those wavelengths), so the model produces *less* 8–40 μm flux than a
        pure MBB normalised to the same L_absorbed.  The 8–40 μm advantage
        described in Casey (2012) applies to warmer starburst / AGN-heated dust
        components where T ≳ 80–100 K.

    The implemented model uses the following convention (see code comments)::

        S(ν) = N_pl * ν^α_mid * f(λ)         [mid-IR power law, f→1 at short λ]
             + N_bb * ν^(3+β) / (exp(hν/kT) - 1) * (1 - f(λ))   [FIR MBB, 1-f→1 at long λ]

    where the transition function (f→1 selects power law at short λ) is::

        f(λ) = 1 / (1 + (λ / λ_0)^2)

    Note: Casey (2012, MNRAS 425 3094) Eq. 2 defines the carrier function differently;
    the code's convention has f→1 at short λ (mid-IR) and 1-f→1 at long λ (FIR).
    The shapes produced are equivalent; only the labelling of f vs (1-f) differs.

    The empirical turnover wavelength is (Eq. 3, with errata)::

        λ_0 = b1 + b2 * T[μm]

    with ``b1 = 26.68 μm``, ``b2 = 6.246e-3 μm/K``.

    Both components are normalized so that the total frequency integral
    equals ``L_absorbed``.

    When ``redshift > 0``, the dust temperature is corrected for CMB
    heating (da Cunha et al. 2013) and the observed flux is reduced by
    the CMB contrast factor.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending). [Å]
    L_absorbed : float
        Total absorbed luminosity in Lsun (sets the normalization). [L_sun]
    dust_T : float
        Dust temperature in Kelvin.  Typical range: 25--60 K. [K]
    dust_beta_ir : float
        Dust emissivity index for the MBB component.
        Typical range: 1.5--2.0. [dimensionless]
    dust_alpha_mir : float
        Mid-IR power-law slope.  Typical range: 1.5--2.5. [dimensionless]
    optically_thin : bool
        If True, zero the mid-IR power-law component, leaving only the
        modified blackbody.  Default: False. [dimensionless]
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default 0 (no correction). [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in Lsun/Hz.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere.

    References
    ----------
    Casey, C. M., 2012, MNRAS, 425, 3094.
    da Cunha, E. et al., 2013, ApJ, 766, 13 (CMB corrections).

    """
    # CMB correction (no-op at z=0)
    T_eff = cmb_corrected_temperature(dust_T, redshift, dust_beta_ir)

    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm  # Hz, descending

    # Planck argument (shared by both components)
    x = jnp.clip(_H_PLANCK * nu / (_K_BOLTZMANN * T_eff), 0.0, 500.0)
    nu_ref = _C_CGS / (100.0e-4)  # 100 μm pivot in Hz

    # Compute transition function between power law and MBB
    f_transition = _casey_transition_function(wavelength_cm, T_eff)

    # Compute mid-IR power-law and modified blackbody components
    power_law = _casey_powerlaw_component(
        nu, nu_ref, f_transition, x, dust_alpha_mir, optically_thin
    )
    mbb = _casey_mbb_component(nu, nu_ref, f_transition, x, dust_beta_ir)

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


# ── Model 1c: Schreiber et al. (2016) dust continuum + PAH ────────


def _drude_profile(
    wavelength_aa: jnp.ndarray,
    lambda0_aa: float,
    fwhm_um: float,
) -> jnp.ndarray:
    r"""Drude profile for PAH emission feature.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid in Ångstrom.
    lambda0_aa : float
        Center wavelength in Ångstrom.
    fwhm_um : float
        FWHM in micrometers.

    Returns
    -------
    ndarray, shape (n_wave,)
        Normalized Drude profile.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The Drude profile is:

    .. math::

        D(\lambda) = \frac{2}{\pi} \frac{\gamma}{
            (\lambda/\lambda_0 - \lambda_0/\lambda)^2 + \gamma^2}

    where :math:`\gamma = \text{FWHM} / \lambda_0`.

    """
    lambda0_um = lambda0_aa / 1e4
    fwhm_um_safe = jnp.maximum(fwhm_um, 1e-6)
    gamma = fwhm_um_safe / lambda0_um

    wavelength_um = wavelength_aa / 1e4
    ratio = wavelength_um / lambda0_um

    denominator = (ratio - 1.0 / ratio) ** 2 + gamma**2
    return 2.0 / jnp.pi * gamma / denominator


# ── Model 2: PAH Drude Profile Template ──


@register_emission_model("pah_drude")
def pah_drude(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Smith et al. (2007) PAH Drude profiles dust emission.

    A sum of 18 PAH Drude profiles (normalized to Smith+2007 SINGS median
    strengths). Runtime evaluation prefers the precomputed lookup from
    :mod:`~tengri.components.dust.dust_analytic_precompute`; this function
    provides a direct full-wavelength evaluation used when precompute is
    unavailable.

    Parameters
    ----------
    wavelength_aa : array, shape (n_wave,)
        Wavelength grid in Angstrom (sorted ascending). [Å]
    L_absorbed : float
        Total absorbed luminosity. [Lsun or erg/s, as passed]
    redshift : float
        Source redshift (unused for this model). [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        Dust emission L_nu in ``[L_absorbed units] / Hz``.

    Notes
    -----
    **JIT-compatible**: yes — computed via triweight interpolation in precompute
    mode; this function is a fallback stub.

    **Gradient-safe**: yes — inherited from precompute lookup.

    The PAH template is a pure shape (no free axes). Runtime evaluation uses the
    precomputed lookup from :mod:`~tengri.components.dust.dust_analytic_precompute`
    and skips the full-wavelength evaluation in the hybrid kernel.

    References
    ----------
    .. [1] Smith, J. D., et al., "The mid-infrared emission of ultraluminous
           infrared galaxies," ApJ, 656, 770 (2007). arXiv:astro-ph/0701042.
           https://doi.org/10.1086/510378

    """
    from tengri.components.dust.drude_profiles import compute_pah_template

    # Convert to microns for compute_pah_template
    wavelength_um = wavelength_aa / 1e4

    # Compute PAH template in L_lambda (dimensionless relative units)
    pah_llam = compute_pah_template(wavelength_um, strengths=None)

    # Convert L_lambda to L_nu: L_nu = L_lambda * lambda^2 / c
    wave_cm = wavelength_aa * 1.0e-8  # Angstrom -> cm
    c_cgs = 2.99792458e10  # cm/s
    lnu = L_absorbed * pah_llam * (wave_cm**2) / c_cgs

    return lnu


@register_emission_model("schreiber2016")
def schreiber2016(
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    dust_T: float = 30.0,
    dust_f_pah: float = 0.05,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    r"""Schreiber et al. (2016) 2-parameter dust emission model.

    Mixes dust continuum and PAH emission by a fractional parameter.
    The dust continuum is a modified blackbody (modified_blackbody with
    beta=1.5). The PAH component is approximated as a sum of Drude profiles
    at standard wavelengths.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid in Ångstrom (sorted ascending).
    L_absorbed : float
        Total absorbed luminosity. Unit-agnostic: the output L_nu will be
        in the same units per Hz.
    dust_T : float
        Dust continuum temperature in Kelvin.
        Typical range: 15--60 K. Default: 30.0.
    dust_f_pah : float
        Fractional contribution from PAH emission in [0, 1].
        Default: 0.05.
    redshift : float
        Source redshift. When > 0, CMB heating correction is applied.
        Default: 0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_nu in ``[L_absorbed units] / Hz``.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The model composition is:

    .. math::

        L_\nu = (1 - f_{\rm PAH}) L_\nu^{\rm continuum} + f_{\rm PAH} L_\nu^{\rm PAH}

    where:

    - Dust continuum: modified blackbody with temperature T_dust and emissivity
      index β = 1.5, using the same normalization as ``modified_blackbody``.
    - PAH: sum of Drude profiles at standard rest wavelengths (3.3, 6.2, 7.7,
      8.6, 11.3, 12.7 μm) with relative strengths from Smith et al. (2007).

    The total integral over frequency is normalized to ``L_absorbed``.

    References
    ----------
    .. [1] Schreiber, C., Elbaz, D., Sparre, M., et al., 2016,
           A&A, 589, A35 (https://doi.org/10.1051/0004-6361/201527923)

    .. [2] Smith, J. D. T., Draine, B. T., Dale, D. A., et al., 2007,
           ApJ, 656, 770 (PAH profile templates).

    """
    # CMB correction (no-op at z=0)
    T_eff = cmb_corrected_temperature(dust_T, redshift, 1.5)

    wavelength_cm = wavelength_aa * _AA_TO_CM
    nu = _C_CGS / wavelength_cm  # Hz, descending

    # ─ Dust continuum component (modified blackbody with beta=1.5) ─
    nu_ref = _C_CGS / (250.0e-4)
    emissivity = (nu / nu_ref) ** 1.5
    bnu = planck_bnu(wavelength_aa, T_eff)
    continuum = emissivity * bnu

    # ─ PAH component (sum of Drude profiles) ─
    # PAH features from Smith et al. (2007), with rest wavelengths and FWHM
    pah_features = [
        (33000.0, 0.05, 0.04),  # 3.3 μm, FWHM 0.05 μm, strength 0.04
        (62000.0, 0.19, 0.14),  # 6.2 μm, FWHM 0.19 μm, strength 0.14
        (77000.0, 0.47, 0.42),  # 7.7 μm, FWHM 0.47 μm, strength 0.42
        (86000.0, 0.27, 0.11),  # 8.6 μm, FWHM 0.27 μm, strength 0.11
        (113000.0, 0.18, 0.19),  # 11.3 μm, FWHM 0.18 μm, strength 0.19
        (127000.0, 0.32, 0.10),  # 12.7 μm, FWHM 0.32 μm, strength 0.10
    ]

    pah_sum = jnp.zeros_like(wavelength_aa)
    for lambda0_aa, fwhm_um, strength in pah_features:
        drude = _drude_profile(wavelength_aa, lambda0_aa, fwhm_um)
        pah_sum = pah_sum + strength * drude

    # Normalize PAH to unit integral
    pah_integral = -jnp.trapezoid(pah_sum, nu)
    pah_normalized = jnp.where(pah_integral > 0.0, pah_sum / pah_integral, 0.0)

    # ─ Mix components ─
    f_pah_clipped = jnp.clip(dust_f_pah, 0.0, 1.0)
    mixed_shape = (1.0 - f_pah_clipped) * continuum + f_pah_clipped * pah_normalized

    # ─ Normalize total to L_absorbed ─
    integral = -jnp.trapezoid(mixed_shape, nu)
    norm = jnp.where(integral > 0.0, L_absorbed / integral, 0.0)

    result = norm * mixed_shape

    # CMB contrast
    contrast = cmb_contrast_factor(wavelength_aa, T_eff, redshift)

    return result * contrast


# ── Backward-compatible module-level aliases for direct imports ───
# Tests and user code may do ``from tengri.components.dust.emission import draine_li2007``.
# These aliases point to the lazy wrappers (which auto-load templates on call).


def draine_li2007(*args, **kwargs):
    """Draine & Li (2007) — dispatches to the registry (auto-loads templates).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters (alpha, U_min, gamma, q_pah). See registered
        function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded tabulated DL07 model. Auto-loads HDF5 templates
    on first call.

    """
    return DUST_EMISSION_MODELS["draine_li2007"](*args, **kwargs)


def dale2014(*args, **kwargs):
    """Dale et al. (2014) — dispatches to the registry (auto-loads templates).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters (alpha). See registered function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded Dale2014 template model. Auto-loads HDF5
    templates on first call.

    """
    return DUST_EMISSION_MODELS["dale2014"](*args, **kwargs)


def draine_li2014(*args, **kwargs):
    """Draine & Li (2014) — dispatches to the registry (auto-loads templates).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters (alpha, U_min, gamma, q_pah). See registered
        function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded DL14 template model. Auto-loads HDF5 templates
    on first call. DL14 is the 2014 update to Draine & Li 2007 with additional
    alpha (radiation field) dependence.

    """
    return DUST_EMISSION_MODELS["draine_li2014"](*args, **kwargs)


def astrodust(*args, **kwargs):
    """Astrodust+PAH (Hensley & Draine 2023) — dispatches to the registry.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters. See registered function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded Astrodust+PAH template model. Auto-loads
    HDF5 templates on first call.

    """
    return DUST_EMISSION_MODELS["astrodust"](*args, **kwargs)


def bosa(*args, **kwargs):
    """BOSA (Boquien & Salim 2021) — dispatches to the registry.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters. See registered function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded BOSA template model. Auto-loads HDF5
    templates on first call. BOSA parameterizes dust by (L_TIR, sSFR).

    """
    return DUST_EMISSION_MODELS["bosa"](*args, **kwargs)


def themis(*args, **kwargs):
    """THEMIS (Jones+2017) — dispatches to the registry.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    L_absorbed : float
        Total absorbed luminosity. [L_sun]
    **kwargs
        Model-specific parameters. See registered function for details.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν [erg/s/Hz or L_sun/Hz].

    Notes
    -----
    **JIT-compatible**: yes (returned function is JIT-compiled).

    Dispatches to the lazy-loaded THEMIS/DustEM template model. Auto-loads
    HDF5 templates on first call. Uses a-C(:H) aromatic carbon composition
    rather than PAH fraction.

    """
    return DUST_EMISSION_MODELS["themis"](*args, **kwargs)


# ── Energy-balance decomposition models ───────────────────────────


def energy_balance_split(
    wavelength_aa: jnp.ndarray,
    L_absorbed_stellar: float,
    L_agn_ir: float = 0.0,
    eta_balance: float = 1.0,
    f_cold: float = 0.5,
    dust_T_warm: float = 45.0,
    dust_T_cold: float = 20.0,
    dust_beta_warm: float = 1.5,
    dust_beta_cold: float = 2.0,
    redshift: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    r"""Two-temperature energy balance with AGN contribution.

    Extends simple eta_balance by decomposing IR into warm (SF-heated)
    and cold (diffuse ISM) components, plus optional AGN IR contribution.

    Parameters
    ----------
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid. [Å] Must be sorted ascending.
    L_absorbed_stellar : float
        Total absorbed stellar luminosity. [Lsun]
    L_agn_ir : float
        Additional AGN-heated IR luminosity. [Lsun] Default: 0.0.
    eta_balance : float
        Energy balance parameter: ratio of re-emitted to absorbed stellar luminosity.
        [dimensionless] Default: 1.0 (strict energy balance).
    f_cold : float
        Fraction of total IR luminosity in the cold component.
        [dimensionless, in [0, 1]] Default: 0.5.
    dust_T_warm : float
        Warm dust temperature. [K] Default: 45.0.
    dust_T_cold : float
        Cold dust temperature. [K] Default: 20.0.
    dust_beta_warm : float
        Warm component emissivity index. [dimensionless] Default: 1.5.
    dust_beta_cold : float
        Cold component emissivity index. [dimensionless] Default: 2.0.
    redshift : float
        Source redshift. [dimensionless] When > 0, CMB heating correction is applied
        to both components. Default: 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν. [Lsun Hz⁻¹]

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    The total IR luminosity budget is:

    .. math::

        L_{\rm IR,total} = \eta_{\rm balance} L_{\rm absorbed,\star} + L_{\rm AGN,IR}

        L_{\rm warm} = (1 - f_{\rm cold}) L_{\rm IR,total}

        L_{\rm cold} = f_{\rm cold} L_{\rm IR,total}

    Each component is a modified blackbody (via ``modified_blackbody``).

    References
    ----------
    .. [1] V. Kokorev et al., "STARDUST: Spectral Template Analysis and
       Recovery of Dust and Ultraviolet Spectral features,"
       ApJ, 921, 40 (2021). https://doi.org/10.3847/1538-4357/ac1aa7

    .. [2] E. da Cunha et al., "MAGPHYS: a new code to compute and interpret
       the Spectral Energy Distribution of the Galaxy," MNRAS, 388, 1595 (2008).
       https://doi.org/10.1111/j.1365-2966.2008.13535.x

    """
    L_ir_total = eta_balance * L_absorbed_stellar + L_agn_ir

    L_warm = (1.0 - f_cold) * L_ir_total
    L_cold = f_cold * L_ir_total

    # Each component is a modified blackbody
    sed_warm = modified_blackbody(
        wavelength_aa,
        L_absorbed=L_warm,
        dust_T=dust_T_warm,
        dust_beta_ir=dust_beta_warm,
        redshift=redshift,
    )
    sed_cold = modified_blackbody(
        wavelength_aa,
        L_absorbed=L_cold,
        dust_T=dust_T_cold,
        dust_beta_ir=dust_beta_cold,
        redshift=redshift,
    )

    return sed_warm + sed_cold


# ── Application layer: model dispatchers and utilities ────────────


def apply_dust_emission(
    model_name: str,
    wavelength_aa: jnp.ndarray,
    L_absorbed: float,
    **params,
) -> jnp.ndarray:
    r"""Apply a named dust emission model.

    Dispatches to a registered model function by name.

    Parameters
    ----------
    model_name : str
        Registered model name (e.g. "modified_blackbody", "draine_li2007").
    wavelength_aa : array_like, shape (n_wave,)
        Wavelength grid. [Å]
    L_absorbed : float
        Absorbed luminosity. [Lsun]
    **params
        Model-specific keyword arguments (e.g., dust_T, dust_umin, dust_gamma_dl).

    Returns
    -------
    ndarray, shape (n_wave,)
        Dust emission L_ν. [Lsun Hz⁻¹]

    Raises
    ------
    ValueError
        If model_name is not registered.

    Notes
    -----
    **JIT-compatible**: yes if the underlying model is JIT-compatible.

    """
    fn = resolve_emission_model(model_name)
    return fn(wavelength_aa, L_absorbed, **params)


# ── Lazy loading infrastructure for template-based models ─────────


def _make_lazy_loader(
    name: str,
    template_filename: str,
    loader_fn_name: str,
) -> Callable:
    """Create a lazy-loading wrapper that auto-loads templates on first call.

    Parameters
    ----------
    name : str
        Registry name (e.g. ``"dale2014"``).
    template_filename : str
        Canonical HDF5 filename to search for in data/ (e.g. ``"dale2014_templates.h5"``).
        The v2 variant (``"*_v2.h5"``) is tried first if present.
    loader_fn_name : str
        Name of the ``create_*_from_grid`` function in this module.

    """

    def _lazy_wrapper(*args, **kwargs):
        """Resolve and cache the dust emission template on first call, then delegate to it."""
        if name not in _resolved:
            _resolved.add(name)
            # Try v2 HDF5 first (improved grid), then canonical HDF5
            stem = template_filename.rsplit(".", 1)[0]
            v2_name = stem + "_v2.h5"
            path = _find_data_file(v2_name) or _find_data_file(template_filename)
            if path is not None:
                loader = globals()[loader_fn_name]
                tabulated = loader(path)
                DUST_EMISSION_MODELS[name] = tabulated
                return tabulated(*args, **kwargs)
            else:
                raise FileNotFoundError(
                    f"Template file '{template_filename}' not found in data/. "
                    f"The analytic fallback for {name} has been removed because it "
                    f"produced scientifically incorrect results. Download templates "
                    f"or register manually via register_*_tabulated()."
                )
        return DUST_EMISSION_MODELS[name](*args, **kwargs)

    _lazy_wrapper.__name__ = name
    _lazy_wrapper.__doc__ = (
        f"Lazy-loading wrapper for {name}. Auto-loads tabulated templates "
        f"from data/{template_filename} on first call (v2 grid preferred if present)."
    )
    return _lazy_wrapper


def _find_dl07_templates() -> str | None:
    """Find DL07 template files, preferring v2 grid."""
    for fn in ("dl07_templates_v2.h5", "dl07_templates.h5"):
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
            from .emission_templates import create_dl07_from_grid

            tabulated = create_dl07_from_grid(path)
            DUST_EMISSION_MODELS["draine_li2007"] = tabulated
            DUST_EMISSION_MODELS["dl07_tabulated"] = tabulated
            return tabulated(*args, **kwargs)
        else:
            raise FileNotFoundError(
                "DL07 template files (dl07_templates_v2.h5 / dl07_templates.h5) "
                "not found in data/. "
                "The analytic fallback has been removed because it produced "
                "scientifically incorrect results (single-Gaussian PAH approximation). "
                "Run: python scripts/convert_dl07_templates.py"
            )
    return DUST_EMISSION_MODELS["draine_li2007"](*args, **kwargs)


# ── Import emission template functions ───────────────────────────

from .emission_templates import (
    create_astrodust_from_grid as create_astrodust_from_grid,
    create_bosa_from_grid as create_bosa_from_grid,
    create_dale2014_from_grid as create_dale2014_from_grid,
    create_dl07_from_grid as create_dl07_from_grid,
    create_dl14_from_grid as create_dl14_from_grid,
    create_themis_from_grid as create_themis_from_grid,
    load_astrodust_templates as load_astrodust_templates,
    load_bosa_templates as load_bosa_templates,
    load_dale2014_templates as load_dale2014_templates,
    load_dl14_templates as load_dl14_templates,
    load_draine_li_templates as load_draine_li_templates,
    load_themis_templates as load_themis_templates,
    register_astrodust_tabulated as register_astrodust_tabulated,
    register_bosa_tabulated as register_bosa_tabulated,
    register_dale2014_tabulated as register_dale2014_tabulated,
    register_dl07_tabulated as register_dl07_tabulated,
    register_dl14_tabulated as register_dl14_tabulated,
    register_themis_tabulated as register_themis_tabulated,
)

# Register models at module load time
DUST_EMISSION_MODELS["energy_balance_split"] = energy_balance_split

# Register lazy loaders at module load time
# These will auto-load templates on first call
DUST_EMISSION_MODELS["draine_li2007"] = _dl07_lazy_wrapper

DUST_EMISSION_MODELS["dale2014"] = _make_lazy_loader(
    "dale2014",
    "dale2014_templates.h5",
    "create_dale2014_from_grid",
)


@functools.cache
def _load_dl14_fn():
    """Load DL14 template grid from file."""
    from .emission_templates import create_dl14_from_grid

    for fname in ("dl14_templates_v2.h5", "dl14_templates.h5"):
        path = _find_data_file(fname)
        if path is not None:
            return create_dl14_from_grid(path)
    raise FileNotFoundError(
        "DL14 template files not found (dl14_templates_v2.h5 or dl14_templates.h5). "
        "The analytic fallback has been removed because it produced scientifically "
        "incorrect results. Run: python scripts/download_dl14_templates.py"
    )


def _dl14_lazy_wrapper(*args, **kwargs):
    """Lazy loader for DL14: prioritizes v2 grid, falls back to legacy grid."""
    fn = _load_dl14_fn()
    return fn(*args, **kwargs)


DUST_EMISSION_MODELS["draine_li2014"] = _dl14_lazy_wrapper

DUST_EMISSION_MODELS["astrodust"] = _make_lazy_loader(
    "astrodust",
    "astrodust_templates.h5",
    "create_astrodust_from_grid",
)

DUST_EMISSION_MODELS["bosa"] = _make_lazy_loader(
    "bosa",
    "bosa_templates.h5",
    "create_bosa_from_grid",
)

DUST_EMISSION_MODELS["themis"] = _make_lazy_loader(
    "themis",
    "themis_templates.h5",
    "create_themis_from_grid",
)
