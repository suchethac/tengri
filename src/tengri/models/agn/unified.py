"""Unified AGN SED: combined disc + torus emission.

Combines accretion disc and dust torus models into a single AGN SED.
The disc emission is partly absorbed by the torus (via covering factor)
and re-emitted in the IR.

Three pre-registered configurations:

- **simple**: power-law disc + single-temperature torus (3 free params).
- **standard**: multi-color disc + two-temperature torus (5-6 free params).
- **kubota_done**: multi-color disc with BH physics + clumpy torus (8+ params).

Usage::

    from tengri.models.agn.unified import unified_agn, get_agn_model

    # Use a named configuration
    model_fn = get_agn_model("simple")
    l_nu = model_fn(wavelength, agn_log_lbol=44.0, agn_frac=0.1, ...)

    # Or use the generic combiner directly
    l_nu = unified_agn(wavelength, agn_log_lbol=44.0, disc_model="powerlaw", ...)
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp

from tengri.models.agn.blr import blr_emission
from tengri.models.agn.disc import multicolor_disc, powerlaw_disc
from tengri.models.agn.nlr import nlr_emission
from tengri.models.agn.skirtor import create_skirtor_from_grid

# Auto-load tabulated SKIRTOR templates (preferred over analytic)
_skirtor_fn = None
from tengri.models.agn.torus import simple_torus, two_temperature_torus

# ===================================================================
# AGN model registry
# ===================================================================

AGN_MODELS: dict[str, Callable] = {}


def register_agn_model(name: str) -> Callable:
    """Register an AGN model function (decorator factory).

    The registered function must have signature:
        fn(wavelength, agn_log_lbol, **kwargs) -> L_nu [Lsun Hz^-1]
    """

    def decorator(fn: Callable) -> Callable:
        AGN_MODELS[name] = fn
        return fn

    return decorator


def get_agn_model(name: str) -> Callable:
    """Retrieve a registered AGN model by name.

    Parameters
    ----------
    name : str
        Model name. One of: "simple", "standard", "kubota_done".

    Returns
    -------
    callable
        Model function: fn(wavelength, agn_log_lbol, **kwargs) -> L_nu.

    Raises
    ------
    ValueError
        If name is not registered.
    """
    if name not in AGN_MODELS:
        raise ValueError(f"Unknown AGN model '{name}'. Available: {list(AGN_MODELS.keys())}")
    return AGN_MODELS[name]


# ===================================================================
# Generic unified AGN combiner
# ===================================================================


def unified_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    disc_model: str = "powerlaw",
    torus_model: str = "simple",
    agn_torus_frac: float = 0.5,
    **kwargs,
) -> jnp.ndarray:
    """Compute unified AGN SED: disc + torus.

    The torus re-emits a fraction ``agn_torus_frac`` of the bolometric
    luminosity, while the disc emits the remaining ``(1 - agn_torus_frac)``.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity.
    disc_model : str
        Disc model name: "powerlaw" or "multicolor". Default "powerlaw".
    torus_model : str
        Torus model name: "simple" or "two_temperature". Default "simple".
    agn_torus_frac : float
        Covering factor: fraction of L_bol intercepted and re-emitted by
        the torus. The disc gets (1 - agn_torus_frac). Default 0.5.
    **kwargs
        Passed through to both disc and torus functions (they ignore
        unrecognized kwargs via **_kwargs).

    Returns
    -------
    array, shape (n_wave,)
        Total AGN L_nu [Lsun Hz^-1] = L_disc + L_torus.
    """
    from tengri.models.agn.skirtor import skirtor_analytic

    disc_fns = {
        "powerlaw": powerlaw_disc,
        "multicolor": multicolor_disc,
    }
    torus_fns = {
        "simple": simple_torus,
        "two_temperature": two_temperature_torus,
        "skirtor": skirtor_analytic,
    }

    disc_fn = disc_fns[disc_model]
    torus_fn = torus_fns[torus_model]

    # Disc gets (1 - covering_factor) of L_bol
    l_disc = disc_fn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0 - agn_torus_frac,
        **kwargs,
    )

    # Torus re-emits covering_factor of L_bol
    l_torus = torus_fn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
        **kwargs,
    )

    return l_disc + l_torus


# ===================================================================
# Pre-registered AGN models
# ===================================================================


@register_agn_model("simple")
def simple_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 0.1,
    agn_alpha: float = -1.0,
    agn_T_torus: float = 1000.0,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    """Simple AGN: power-law disc + single-temperature torus.

    3 free parameters (+ agn_frac scaling):
    - agn_alpha: disc spectral slope
    - agn_T_torus: torus temperature
    - agn_torus_frac: covering factor

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun).
    agn_frac : float
        AGN luminosity as fraction of total galaxy luminosity.
        Applied as overall scaling. Default 0.1.
    agn_alpha : float
        Disc spectral slope. Default -1.0.
    agn_T_torus : float
        Torus temperature [K]. Default 1000.
    agn_torus_frac : float
        Covering factor. Default 0.5.

    Returns
    -------
    array, shape (n_wave,)
        L_nu [Lsun Hz^-1].
    """
    l_nu = unified_agn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        disc_model="powerlaw",
        torus_model="simple",
        agn_alpha=agn_alpha,
        agn_T_torus=agn_T_torus,
        agn_torus_frac=agn_torus_frac,
    )
    return l_nu * agn_frac


@register_agn_model("standard")
def standard_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 0.1,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_T_hot: float = 1200.0,
    agn_T_warm: float = 300.0,
    agn_frac_hot: float = 0.3,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    """Standard AGN: multi-color disc + two-temperature torus.

    5-6 free parameters:
    - agn_log_mbh: black hole mass
    - agn_log_ledd: Eddington ratio
    - agn_T_hot, agn_T_warm: torus temperatures
    - agn_frac_hot: hot/warm ratio
    - agn_torus_frac: covering factor

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun).
    agn_frac : float
        Overall AGN fraction. Default 0.1.
    agn_log_mbh : float
        log10(M_BH / Msun). Default 8.0.
    agn_log_ledd : float
        log10(L/L_Edd). Default -1.0.
    agn_T_hot : float
        Hot dust temperature [K]. Default 1200.
    agn_T_warm : float
        Warm dust temperature [K]. Default 300.
    agn_frac_hot : float
        Fraction in hot component. Default 0.3.
    agn_torus_frac : float
        Covering factor. Default 0.5.

    Returns
    -------
    array, shape (n_wave,)
        L_nu [Lsun Hz^-1].
    """
    l_nu = unified_agn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        disc_model="multicolor",
        torus_model="two_temperature",
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_T_hot=agn_T_hot,
        agn_T_warm=agn_T_warm,
        agn_frac_hot=agn_frac_hot,
        agn_torus_frac=agn_torus_frac,
    )
    return l_nu * agn_frac


@register_agn_model("kubota_done")
def kubota_done_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 0.1,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.5,
    agn_T_hot: float = 1200.0,
    agn_T_warm: float = 300.0,
    agn_frac_hot: float = 0.3,
    agn_tau_torus: float = 5.0,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    """Kubota & Done (2018) disc + clumpy torus.

    8+ free parameters:
    - agn_log_mbh: black hole mass
    - agn_log_ledd: Eddington ratio
    - agn_a_spin: BH spin
    - agn_cos_inc: inclination
    - agn_T_hot, agn_T_warm: torus temperatures
    - agn_frac_hot: hot/warm ratio
    - agn_tau_torus: torus optical depth
    - agn_torus_frac: covering factor

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun).
    agn_frac : float
        Overall AGN fraction. Default 0.1.
    agn_log_mbh : float
        log10(M_BH / Msun). Default 8.0.
    agn_log_ledd : float
        log10(L/L_Edd). Default -1.0.
    agn_a_spin : float
        BH spin (0 to 0.998). Default 0.0.
    agn_cos_inc : float
        cos(inclination). Default 0.5.
    agn_T_hot : float
        Hot dust temperature [K]. Default 1200.
    agn_T_warm : float
        Warm dust temperature [K]. Default 300.
    agn_frac_hot : float
        Fraction in hot component. Default 0.3.
    agn_tau_torus : float
        Torus optical depth at 9.7 um. Default 5.
    agn_torus_frac : float
        Covering factor. Default 0.5.

    Returns
    -------
    array, shape (n_wave,)
        L_nu [Lsun Hz^-1].
    """
    l_nu = unified_agn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        disc_model="multicolor",
        torus_model="two_temperature",
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
        agn_T_hot=agn_T_hot,
        agn_T_warm=agn_T_warm,
        agn_frac_hot=agn_frac_hot,
        agn_tau_torus=agn_tau_torus,
        agn_torus_frac=agn_torus_frac,
    )
    return l_nu * agn_frac


@register_agn_model("skirtor")
def skirtor_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = 44.0,
    agn_frac: float = 0.1,
    agn_tau_skirtor: float = 7.0,
    agn_p_skirtor: float = 1.0,
    agn_q_skirtor: float = 1.0,
    agn_oa_skirtor: float = 40.0,
    agn_cos_inc: float = 0.5,
    agn_torus_frac: float = 0.5,
    **_kwargs,
) -> jnp.ndarray:
    """SKIRTOR clumpy torus AGN: disc + SKIRTOR torus (analytic).

    Uses the analytic SKIRTOR approximation (Stalevski et al. 2012, 2016)
    for the torus emission, combined with a power-law accretion disc.

    5 SKIRTOR-specific free parameters:
    - agn_tau_skirtor: optical depth at 9.7 um (3-11)
    - agn_p_skirtor: radial density gradient (0-1.5)
    - agn_q_skirtor: polar density gradient (0-1.5)
    - agn_oa_skirtor: opening angle in degrees (20-60)
    - agn_cos_inc: cosine of inclination (0=edge-on, 1=face-on)

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Default 44.0.
    agn_frac : float
        Overall AGN fraction scaling. Default 0.1.
    agn_tau_skirtor : float
        9.7 um optical depth (3-11). Default 7.0.
    agn_p_skirtor : float
        Radial density gradient (0-1.5). Default 1.0.
    agn_q_skirtor : float
        Polar density gradient (0-1.5). Default 1.0.
    agn_oa_skirtor : float
        Opening angle [degrees] (20-60). Default 40.0.
    agn_cos_inc : float
        cos(inclination), 0=edge-on, 1=face-on. Default 0.5.
    agn_torus_frac : float
        Covering fraction. Default 0.5.

    Returns
    -------
    array, shape (n_wave,)
        L_nu [Lsun Hz^-1].
    """
    # Disc gets (1 - covering_factor) of L_bol
    l_disc = powerlaw_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0 - agn_torus_frac,
    )

    # SKIRTOR torus re-emits covering_factor of L_bol
    # Auto-load tabulated templates on first call
    global _skirtor_fn
    if _skirtor_fn is None:
        from pathlib import Path

        grid_path = Path(__file__).resolve().parents[2] / "data" / "skirtor_templates.npz"
        if not grid_path.is_file():
            # Search alternative locations
            for candidate in [
                Path("data/skirtor_templates.npz"),
                Path.home() / "Projects/tengri/data/skirtor_templates.npz",
            ]:
                if candidate.is_file():
                    grid_path = candidate
                    break
        if grid_path.is_file():
            _skirtor_fn = create_skirtor_from_grid(str(grid_path))
        else:
            raise FileNotFoundError(
                "SKIRTOR templates not found. Download from "
                "https://sites.google.com/site/skirtorus/sed-library "
                "and convert with scripts/convert_skirtor_templates.py"
            )

    l_torus = _skirtor_fn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_tau_skirtor=agn_tau_skirtor,
        agn_p_skirtor=agn_p_skirtor,
        agn_q_skirtor=agn_q_skirtor,
        agn_oa_skirtor=agn_oa_skirtor,
        agn_cos_inc=agn_cos_inc,
        agn_torus_frac=agn_torus_frac,
    )

    return (l_disc + l_torus) * agn_frac


# ===================================================================
# Geometric masking (smooth sigmoid for differentiability)
# ===================================================================

_LSUN_ERG = 3.828e33  # Solar luminosity [erg s^-1]


def _sigmoid_mask(
    cos_inc: float,
    theta_torus: float,
    width: float = 2.0,
) -> float:
    """Smooth geometric mask for disc/BLR visibility.

    Returns ~1 (visible) for face-on orientations and ~0 (obscured)
    when the line of sight passes through the torus.

    The transition occurs at inclination = 90 - theta_torus (edge of
    the torus opening cone). A smooth sigmoid replaces the hard cutoff
    to maintain differentiability.

    Parameters
    ----------
    cos_inc : float
        Cosine of inclination (0 = edge-on, 1 = face-on).
    theta_torus : float
        Torus half-opening angle [degrees].
    width : float
        Sigmoid transition width [degrees]. Default 2.0.

    Returns
    -------
    float
        Visibility fraction in [0, 1].
    """
    # Inclination in degrees: inc = arccos(cos_inc)
    inc_deg = jnp.degrees(jnp.arccos(jnp.clip(cos_inc, 0.0, 1.0)))
    # Critical angle: above this the torus blocks the view
    inc_crit = 90.0 - jnp.clip(theta_torus, 0.0, 90.0)
    # Sigmoid: 1 when inc << inc_crit, 0 when inc >> inc_crit
    return jax.nn.sigmoid(-(inc_deg - inc_crit) / jnp.maximum(width, 0.1))


# ===================================================================
# Unified AGN with NLR + BLR decomposition
# ===================================================================


@register_agn_model("unified_nlr_blr")
def unified_nlr_blr(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = 44.0,
    agn_cos_inc: float = 0.5,
    agn_theta_torus: float = 30.0,
    agn_covering_nlr: float = 0.1,
    agn_covering_blr: float = 0.1,
    agn_log_mbh: float = 7.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_T_hot: float = 1200.0,
    agn_T_warm: float = 300.0,
    agn_frac_hot: float = 0.3,
    agn_tau_torus: float = 5.0,
    agn_torus_frac: float = 0.5,
    agn_frac: float = 0.1,
    agn_blr_fwhm: float = 5000.0,
    agn_nlr_fwhm: float = 500.0,
    **_kwargs,
) -> jnp.ndarray:
    """Unified AGN SED with NLR/BLR decomposition and geometric masking.

    Extends the ``kubota_done`` model with narrow and broad line region
    emission, inspired by Synthesizer's UnifiedAGN. The disc and BLR
    are masked by the torus at high inclinations using a smooth sigmoid
    transition. The NLR and torus are always visible (isotropic).

    Total SED::

        L_agn = mask_disc * L_disc
              + L_torus
              + covering_nlr * L_disc_bol * eta_nlr   (NLR, isotropic)
              + mask_blr * covering_blr * L_disc_bol * eta_blr  (BLR, masked)

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity. Default 44.0.
    agn_cos_inc : float
        Cosine of inclination angle (0 = edge-on/Type 2,
        1 = face-on/Type 1). Default 0.5.
    agn_theta_torus : float
        Torus half-opening angle [degrees]. Default 30.0.
    agn_covering_nlr : float
        NLR covering fraction (0 to 1). Default 0.1.
    agn_covering_blr : float
        BLR covering fraction (0 to 1). Default 0.1.
    agn_log_mbh : float
        log10(M_BH / Msun). Default 7.0.
    agn_log_ledd : float
        log10(L/L_Edd). Default -1.0.
    agn_a_spin : float
        BH spin (0 to 0.998). Default 0.0.
    agn_T_hot : float
        Hot dust temperature [K]. Default 1200.
    agn_T_warm : float
        Warm dust temperature [K]. Default 300.
    agn_frac_hot : float
        Hot-to-warm dust fraction. Default 0.3.
    agn_tau_torus : float
        Torus optical depth at 9.7 um. Default 5.
    agn_torus_frac : float
        Torus covering factor (fraction of L_bol intercepted by torus).
        Default 0.5.
    agn_frac : float
        Overall AGN fraction scaling. Default 0.1.
    agn_blr_fwhm : float
        BLR line FWHM [km/s]. Default 5000.
    agn_nlr_fwhm : float
        NLR line FWHM [km/s]. Default 500.

    Returns
    -------
    array, shape (n_wave,)
        Total AGN L_nu [Lsun Hz^-1].
    """
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG

    # --- Geometric masks ---
    mask_disc = _sigmoid_mask(agn_cos_inc, agn_theta_torus)
    mask_blr = _sigmoid_mask(agn_cos_inc, agn_theta_torus)

    # --- Disc emission (intrinsic, before masking) ---
    # Disc gets (1 - torus_frac) of L_bol
    l_disc = multicolor_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0 - agn_torus_frac,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
    )
    # Apply geometric masking
    l_disc_masked = mask_disc * l_disc

    # --- Torus emission (always visible) ---
    l_torus = two_temperature_torus(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
        agn_T_hot=agn_T_hot,
        agn_T_warm=agn_T_warm,
        agn_frac_hot=agn_frac_hot,
        agn_tau_torus=agn_tau_torus,
    )

    # --- NLR emission (isotropic, always visible) ---
    # NLR is illuminated by the disc bolometric luminosity
    l_disc_bol_erg = (1.0 - agn_torus_frac) * l_bol_erg
    l_nlr_erg = nlr_emission(
        wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_covering_nlr,
        fwhm_kms=agn_nlr_fwhm,
    )
    l_nlr = l_nlr_erg / _LSUN_ERG

    # --- BLR emission (masked by torus) ---
    l_blr_erg = blr_emission(
        wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_covering_blr,
        fwhm_kms=agn_blr_fwhm,
    )
    l_blr = mask_blr * l_blr_erg / _LSUN_ERG

    # --- Total ---
    l_total = l_disc_masked + l_torus + l_nlr + l_blr
    return l_total * agn_frac
