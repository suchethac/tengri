"""Unified AGN SED: combined disc + torus + NLR + BLR emission.

Combines accretion disc and dust torus models into a single AGN SED.
The disc emission is partly absorbed by the torus (via covering factor)
and re-emitted in the IR.

Architecture overview
---------------------
The component tree and geometric masking design in this module is inspired
by the ``UnifiedAGN`` class in the Synthesizer package
(Lovell et al. 2025, Open J. Astrophys., arXiv:2508.03888;
 Roper et al. 2025, arXiv:2506.15811; https://github.com/synthesizer-project/synthesizer).

Synthesizer's model defines:

- An accretion disc whose inclination-dependent emission is extracted from
  precomputed photoionisation **grids** (CLOUDY-based).
- NLR emission as grid-extracted nebular spectra, *not* masked by the torus
  (isotropic: always visible at all inclinations).
- BLR emission as grid-extracted nebular spectra, masked by the torus using a
  **hard binary condition**: ``inclination + theta_torus > 90°`` → zeroed.
- A torus that reprocesses the isotropic disc emission, scaled by
  ``torus_fraction = theta_torus / 90°`` (geometric fraction of sky covered).
- Optional diffuse dust attenuation layered on top.

Differences in this implementation
-----------------------------------
tengri adopts the same conceptual decomposition but diverges in several ways
for compatibility with gradient-based inference (VI, HMC):

1. **Analytic disc models, not grids**: disc emission uses the closed-form
   multicolor/Shakura-Sunyaev or power-law models from ``disc.py`` rather
   than a photoionisation grid. This makes the disc JIT/grad-compatible.

2. **Analytic NLR/BLR templates, not grids**: NLR and BLR use empirically
   calibrated Gaussian line templates (Vanden Berk et al. 2001; Groves et al.
   2004) rather than CLOUDY-generated photoionisation grids. Grids would
   require non-differentiable table look-ups over (U, n_H) axes.

3. **Smooth sigmoid geometric mask, not a hard binary**: Synthesizer zeros
   the disc/BLR whenever ``inclination + theta_torus > 90°``. tengri replaces
   this with a smooth sigmoid centred at the critical angle (see
   ``_sigmoid_mask``). The sigmoid preserves the gradient through the
   inclination parameter, which would otherwise be zero almost everywhere
   under the hard mask.

4. **Explicit ``agn_torus_frac`` parameter, not derived from theta_torus**:
   Synthesizer internally computes ``torus_fraction = theta_torus / 90°``,
   coupling the torus covering factor to the opening angle. tengri keeps
   ``agn_torus_frac`` as an independent free parameter. This decoupling is
   intentional: deriving the covering factor from ``cos(theta_torus)`` in
   the forward pass introduces a gradient discontinuity at ``theta_torus=0``
   and at ``theta_torus=90°`` (see CLAUDE.md gotcha).

5. **Polar dust reddening (SMC law)**: an optional ``agn_polar_ebv`` parameter
   applies SMC-law extinction to the disc and BLR for Type 1 sightlines.
   This is absent from Synthesizer's basic UnifiedAGN but present in the
   CIGALE ``skirtor2016`` module (Yang et al. 2020, MNRAS, 491, 740).

Pre-registered configurations
------------------------------
- **simple**: power-law disc + single-temperature torus (3 free params).
- **standard**: multi-color disc + two-temperature torus (5-6 free params).
- **kubota_done**: multi-color disc with BH physics + clumpy torus (8+ params).
- **unified_nlr_blr**: full Synthesizer-inspired model with NLR/BLR + polar dust.

Usage::

    from tengri.components.agn.unified import unified_agn, resolve_agn_model

    # Use a named configuration
    model_fn = resolve_agn_model("simple")
    l_nu = model_fn(wavelength, agn_log_lbol=44.0, agn_frac=0.1, ...)

    # Or use the generic combiner directly
    l_nu = unified_agn(wavelength, agn_log_lbol=44.0, disc_model="powerlaw", ...)

References
----------
.. [1] Lovell C. C. et al. 2025, Open Journal of Astrophysics,
       "Synthesizer: a Software Package for Synthetic Astronomical Observables",
       arXiv:2508.03888, https://doi.org/10.48550/arXiv.2508.03888
.. [2] Roper W. J. et al. 2025, arXiv:2506.15811,
       "Synthesizer: Synthetic Observables For Modern Astronomy",
       https://doi.org/10.48550/arXiv.2506.15811
.. [3] synthesizer source: https://github.com/synthesizer-project/synthesizer
       (src/synthesizer/emission_models/agn/unified_agn.py)
"""

import functools
import warnings
from collections.abc import Callable

import jax
import jax.numpy as jnp

from tengri.components.agn.blr import blr_emission
from tengri.components.agn.cat3d_wind import cat3d_wind_analytic
from tengri.components.agn.disc import (
    adaf_disc,
    create_relagn_disc_from_grid,
    kubota_done_disc,
    multicolor_disc,
    powerlaw_disc,
)
from tengri.components.agn.nlr import nlr_emission
from tengri.components.agn.silva04 import silva04_analytic
from tengri.components.agn.skirtor import _find_skirtor_grid, create_skirtor_from_grid
from tengri.components.agn.torus import simple_torus, two_temperature_torus
from tengri.components.dust.attenuation import prevot_smc
from tengri.utils.physics_constants import L_SUN as _LSUN_ERG


def _redden_disc(
    wavelength: jnp.ndarray,
    l_disc: jnp.ndarray,
    agn_ebv_disc: float,
) -> jnp.ndarray:
    r"""Apply Prévot SMC extinction to the disc SED.

    The Prévot et al. 1984 SMC law with ``R_V = 2.72`` is the standard
    AGN-disc obscuration prescription used by AGNfitter
    (``BBBred_Prevot`` in ``MODEL_AGNfitter.py``).  When
    ``agn_ebv_disc`` is 0.0 this is a no-op (returns the input unchanged).

    Parameters
    ----------
    wavelength : ndarray, shape (n_wave,)
        Rest-frame wavelength grid. [Å]
    l_disc : ndarray, shape (n_wave,)
        Unreddened disc SED. [erg/s/Hz]
    agn_ebv_disc : float
        Colour excess :math:`E(B-V)` applied to the disc. [mag]

    Returns
    -------
    ndarray, shape (n_wave,)
        Reddened disc SED. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes.

    **Gradient-safe**: yes — ``prevot_smc`` uses a smooth sigmoid ramp
    through the X-ray region.

    Attenuation at V band:

    .. math::

        A(V) = R_V \cdot E(B-V), \qquad
        L_{\rm red}(\lambda) = L(\lambda)\, 10^{-0.4\, k(\lambda)\, R_V\, E(B-V)}

    where :math:`k(\lambda) = A(\lambda)/A(V)` follows tengri's
    ``k(V) = 1`` dust-law convention. The :math:`R_V = 2.72` factor
    converts ``agn_ebv_disc`` (the user-facing :math:`E(B-V)`) to
    :math:`A(V)` before applying the wavelength-dependent
    :math:`k(\lambda)` curve.
    """
    _R_V_PREVOT_SMC = 2.72  # Prevot+1984
    k = prevot_smc(wavelength)
    return l_disc * jnp.power(10.0, -0.4 * k * _R_V_PREVOT_SMC * agn_ebv_disc)


@functools.cache
def _load_skirtor_fn():
    """Load SKIRTOR template grid from file.

    Delegates to ``skirtor._find_skirtor_grid()`` which searches v3, v2,
    and npz formats in priority order.
    """
    return create_skirtor_from_grid(_find_skirtor_grid())


def _find_relagn_grid() -> str:
    """Locate the RELAGN outer-disc grid file.

    Searches ``data/relagn_disc_grid.h5`` relative to the repository root.

    Raises
    ------
    FileNotFoundError
        If no grid file is found. Run ``scripts/build_relagn_disc_grid.py``.
    """
    from pathlib import Path

    base = Path(__file__).resolve().parents[4]
    candidates = [
        base / "data" / "relagn_disc_grid.h5",
        Path("data/relagn_disc_grid.h5"),
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        "RELAGN disc grid not found. Run: "
        "conda run -n henv python scripts/build_relagn_disc_grid.py"
    )


@functools.cache
def _load_relagn_fn():
    """Load RELAGN outer-disc grid from HDF5 (cached)."""
    return create_relagn_disc_from_grid(_find_relagn_grid())


# ── AGN model registry ────────────────────────────────────────────

AGN_MODELS: dict[str, Callable] = {}


def register_agn_model(name: str) -> Callable:
    """Register an AGN model function (decorator factory).

    Parameters
    ----------
    name : str
        Unique model name for the registry.

    Returns
    -------
    callable
        Decorator that registers the decorated function in AGN_MODELS.

    Notes
    -----
    **JIT-compatible**: no — registers at module load time (not JIT-compilable).

    The registered function must have signature:
        fn(wavelength, agn_log_lbol, **kwargs) -> L_nu [erg/s/Hz]
    """

    def decorator(fn: Callable) -> Callable:
        """Inner decorator that registers function in AGN_MODELS dict."""
        AGN_MODELS[name] = fn
        return fn

    return decorator


def resolve_agn_model(name: str) -> Callable:
    """Retrieve a registered AGN model by name.

    Parameters
    ----------
    name : str
        Model name (e.g., "simple", "standard", "multicolor_agn", "adaf").

    Returns
    -------
    callable
        Model function: fn(wavelength, agn_log_lbol, **kwargs) -> L_nu [erg/s/Hz].

    Raises
    ------
    ValueError
        If name is not registered.

    Notes
    -----
    **JIT-compatible**: no — performs dictionary lookup at initialization time.
    """
    if name not in AGN_MODELS:
        raise ValueError(f"Unknown AGN model '{name}'. Available: {list(AGN_MODELS.keys())}")

    # Emit deprecation warning for old model names
    if name == "kubota_done":
        warnings.warn(
            "'kubota_done' is deprecated. Use 'multicolor_agn' instead.",
            DeprecationWarning,
            stacklevel=2,
        )

    return AGN_MODELS[name]


from tengri._deprecated import deprecated_alias

get_agn_model = deprecated_alias(resolve_agn_model, old_name="get_agn_model")


# ── Generic unified AGN combiner ──────────────────────────────────


def unified_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    disc_model: str = "powerlaw",
    torus_model: str = "simple",
    agn_torus_frac: float = 0.5,
    agn_ebv_disc: float = 0.0,
    **kwargs,
) -> jnp.ndarray:
    """Compute unified AGN SED: disc + torus.

    The torus re-emits a fraction ``agn_torus_frac`` of the bolometric
    luminosity, while the disc emits the remaining ``(1 - agn_torus_frac)``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    disc_model : str, optional
        Disc model name: "powerlaw", "multicolor", "kubota_done_3zone", "adaf".
        Default "powerlaw".
    torus_model : str, optional
        Torus model name: "simple", "two_temperature", "skirtor".
        Default "simple".
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless], range [0, 1]. Disc gets
        (1 - agn_torus_frac). Default 0.5.
    **kwargs
        Additional parameters passed to disc and torus functions
        (they ignore unrecognized kwargs).

    Returns
    -------
    ndarray, shape (n_wave,)
        Total AGN L_nu [erg/s/Hz] = L_disc + L_torus.

    References
    ----------
    .. [1] See :mod:`tengri.components.agn.disc` and
           :mod:`tengri.components.agn.torus` for individual model references.

    Notes
    -----
    **JIT-compatible**: depends on disc and torus models (mostly yes for analytic).

    **Gradient-safe**: yes when using analytic disc and torus models.
    """
    from tengri.components.agn.skirtor import skirtor_analytic

    disc_fns = {
        "powerlaw": powerlaw_disc,
        "multicolor": multicolor_disc,
        "kubota_done_3zone": kubota_done_disc,
        "adaf": adaf_disc,
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
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)

    # Torus re-emits covering_factor of L_bol
    l_torus = torus_fn(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
        **kwargs,
    )

    return l_disc + l_torus


# ── Pre-registered AGN models ─────────────────────────────────────


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

    Minimal 3-parameter model (+ overall luminosity scaling).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_frac : float, optional
        AGN luminosity fraction of total galaxy SED [dimensionless].
        Applied as overall scaling. Default 0.1.
    agn_alpha : float, optional
        Disc spectral slope [dimensionless]. Default -1.0.
    agn_T_torus : float, optional
        Torus temperature [K]. Default 1000.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless], range [0, 1]. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — delegates to :func:`unified_agn`.
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

    5-6 free parameters controlling BH accretion physics and dust emission.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_frac : float, optional
        AGN luminosity fraction [dimensionless]. Default 0.1.
    agn_log_mbh : float, optional
        log10 of black hole mass [Msun]. Default 8.0.
    agn_log_ledd : float, optional
        log10 of Eddington ratio [dimensionless]. Default -1.0.
    agn_T_hot : float, optional
        Hot dust temperature [K]. Default 1200.
    agn_T_warm : float, optional
        Warm dust temperature [K]. Default 300.
    agn_frac_hot : float, optional
        Hot component mass fraction [dimensionless]. Default 0.3.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless]. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — delegates to :func:`unified_agn` with multicolor disc.
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


@register_agn_model("multicolor_agn")
def multicolor_agn(
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
    """Multicolor Shakura-Sunyaev disc + two-temperature torus.

    Standard thin-disc SED with spin-dependent ISCO and Novikov-Thorne
    radiative efficiency. This is the outer standard disc only — no warm
    Comptonization or hot corona (for the full 3-zone model, see kubota_done_full).

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_frac : float, optional
        AGN luminosity fraction [dimensionless]. Default 0.1.
    agn_log_mbh : float, optional
        log10 of black hole mass [Msun]. Default 8.0.
    agn_log_ledd : float, optional
        log10 of Eddington ratio [dimensionless]. Default -1.0.
    agn_a_spin : float, optional
        Black hole spin [dimensionless], range [0, 0.998]. Default 0.0.
    agn_cos_inc : float, optional
        cos(inclination) [dimensionless], range [0, 1]. Default 0.5.
    agn_T_hot : float, optional
        Hot dust temperature [K]. Default 1200.
    agn_T_warm : float, optional
        Warm dust temperature [K]. Default 300.
    agn_frac_hot : float, optional
        Hot dust mass fraction [dimensionless]. Default 0.3.
    agn_tau_torus : float, optional
        Torus optical depth at 9.7 um [dimensionless]. Default 5.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless]. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses :func:`multicolor_disc` and :func:`two_temperature_torus`.

    Also registered as "kubota_done" (deprecated alias).
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


# "kubota_done" is a registered alias for multicolor_agn in the model registry
AGN_MODELS["kubota_done"] = multicolor_agn


@register_agn_model("kubota_done_full")
def kubota_done_full_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 0.1,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -1.0,
    agn_a_spin: float = 0.0,
    agn_cos_inc: float = 0.5,
    agn_f_hard: float = 0.02,
    agn_gamma_warm: float = 2.5,
    agn_kt_warm: float = 0.2,
    agn_gamma_hard: float = 1.8,
    agn_kt_hot: float = 100.0,
    agn_r_warm_ratio: float = 2.0,
    agn_T_hot: float = 1200.0,
    agn_T_warm: float = 300.0,
    agn_frac_hot: float = 0.3,
    agn_tau_torus: float = 5.0,
    agn_torus_frac: float = 0.5,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Full Kubota & Done (2018) 3-zone disc + two-temperature torus.

    Extends ``multicolor_agn`` with the full K&D 3-zone disc model:
    outer standard disc, warm Comptonization (soft X-ray excess), and
    hot corona (hard X-ray power law). Combined with a two-temperature dust torus.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_frac : float, optional
        AGN luminosity fraction [dimensionless]. Default 0.1.
    agn_log_mbh : float, optional
        log10 of black hole mass [Msun]. Default 8.0.
    agn_log_ledd : float, optional
        log10 of Eddington ratio [dimensionless]. Default -1.0.
    agn_a_spin : float, optional
        Black hole spin [dimensionless], range [0, 0.998]. Default 0.0.
    agn_cos_inc : float, optional
        cos(inclination) [dimensionless]. Default 0.5.
    agn_f_hard : float, optional
        Fraction of L_Edd in corona [dimensionless]. Default 0.02.
    agn_gamma_warm : float, optional
        Warm Comptonization photon index [dimensionless]. Default 2.5.
    agn_kt_warm : float, optional
        Warm electron temperature [keV]. Default 0.2.
    agn_gamma_hard : float, optional
        Hard X-ray photon index [dimensionless]. Default 1.8.
    agn_kt_hot : float, optional
        Hot corona temperature [keV]. Default 100.0.
    agn_r_warm_ratio : float, optional
        R_warm / R_hot ratio [dimensionless]. Default 2.0.
    agn_T_hot : float, optional
        Hot dust temperature [K]. Default 1200.
    agn_T_warm : float, optional
        Warm dust temperature [K]. Default 300.
    agn_frac_hot : float, optional
        Hot dust mass fraction [dimensionless]. Default 0.3.
    agn_tau_torus : float, optional
        Torus optical depth at 9.7 um [dimensionless]. Default 5.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless]. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses :func:`kubota_done_disc` and :func:`two_temperature_torus`.
    """
    # 3-zone disc gets (1 - covering_factor) of L_bol
    l_disc = kubota_done_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0 - agn_torus_frac,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_a_spin=agn_a_spin,
        agn_cos_inc=agn_cos_inc,
        agn_f_hard=agn_f_hard,
        agn_gamma_warm=agn_gamma_warm,
        agn_kt_warm=agn_kt_warm,
        agn_gamma_hard=agn_gamma_hard,
        agn_kt_hot=agn_kt_hot,
        agn_r_warm_ratio=agn_r_warm_ratio,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)

    # Torus re-emits covering_factor of L_bol
    l_torus = two_temperature_torus(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
        agn_T_hot=agn_T_hot,
        agn_T_warm=agn_T_warm,
        agn_frac_hot=agn_frac_hot,
        agn_tau_torus=agn_tau_torus,
    )

    return (l_disc + l_torus) * agn_frac


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
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """SKIRTOR clumpy torus AGN: power-law disc + SKIRTOR torus (analytic).

    Uses the analytic SKIRTOR approximation (Stalevski et al. 2012, 2016)
    for the torus emission, combined with a power-law accretion disc.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float, optional
        log10 of bolometric luminosity [Lsun]. Default 44.0.
    agn_frac : float, optional
        AGN luminosity fraction [dimensionless]. Default 0.1.
    agn_tau_skirtor : float, optional
        Optical depth at 9.7 um [dimensionless], range [3, 11]. Default 7.0.
    agn_p_skirtor : float, optional
        Radial density gradient [dimensionless], range [0, 1.5]. Default 1.0.
    agn_q_skirtor : float, optional
        Polar density gradient [dimensionless], range [0, 1.5]. Default 1.0.
    agn_oa_skirtor : float, optional
        Opening angle [degrees], range [20, 60]. Default 40.0.
    agn_cos_inc : float, optional
        cos(inclination) [dimensionless], 0=edge-on, 1=face-on. Default 0.5.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless]. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: no — requires SKIRTOR template interpolation (non-differentiable).
    """
    # Disc gets (1 - covering_factor) of L_bol
    l_disc = powerlaw_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0 - agn_torus_frac,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)

    # SKIRTOR torus re-emits covering_factor of L_bol
    # Auto-load tabulated templates on first call
    l_torus = _load_skirtor_fn()(
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


@register_agn_model("silva04")
def silva04_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = 44.0,
    agn_frac: float = 0.1,
    agn_log_nh_silva: float = 23.0,
    agn_torus_frac: float = 0.5,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """Silva+04 smooth-torus AGN: power-law disc + Silva+04 torus.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 44.0.
    agn_frac : float, optional
        Overall AGN luminosity fraction applied on top of the
        disc-plus-torus sum. Default 0.1.
    agn_log_nh_silva : float, optional
        Hydrogen column density, ``log10(N_H / cm^-2)``. Default 23.0.
    agn_torus_frac : float, optional
        Fraction of L_bol reprocessed by the torus. Disc receives
        ``1 - agn_torus_frac``. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        Combined AGN SED. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes — both the power-law disc and the Silva+04
    grid interpolation are pure JAX.

    Grid templates ported from AGNfitter (Calistro Rivera et al. 2016);
    see :mod:`tengri.components.agn.silva04` and
    ``scripts/build_silva04_grid.py``.
    """
    l_disc = powerlaw_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0 - agn_torus_frac,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)
    l_torus = silva04_analytic(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_log_nh_silva=agn_log_nh_silva,
        agn_torus_frac=agn_torus_frac,
    )
    return (l_disc + l_torus) * agn_frac


@register_agn_model("cat3d_wind")
def cat3d_wind_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = 44.0,
    agn_frac: float = 0.1,
    agn_cos_inc: float = 0.5,
    agn_a_cat3d: float = -2.0,
    agn_fwd_cat3d: float = 0.45,
    agn_torus_frac: float = 0.5,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """CAT3D-Wind AGN: power-law disc + clumpy-disc + polar-wind torus.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_lbol : float, optional
        ``log10(L_bol / L_sun)``. Default 44.0.
    agn_frac : float, optional
        Overall AGN luminosity fraction applied on top of the
        disc-plus-torus sum. Default 0.1.
    agn_cos_inc : float, optional
        Cosine of inclination. Default 0.5.
    agn_a_cat3d : float, optional
        Radial power-law index of the clumpy-cloud distribution. Default
        −2.0.
    agn_fwd_cat3d : float, optional
        Polar-wind mass fraction. Default 0.2.
    agn_torus_frac : float, optional
        Fraction of L_bol reprocessed by the torus. Disc receives
        ``1 - agn_torus_frac``. Default 0.5.

    Returns
    -------
    ndarray, shape (n_wave,)
        Combined AGN SED. [erg/s/Hz]

    Notes
    -----
    **JIT-compatible**: yes.

    Grid templates ported from AGNfitter-rX (Martínez-Ramírez
    et al. 2024, A&A 688, A46, arXiv:2405.12111) — Hönig & Kishimoto 2017
    CAT3D-Wind three-parameter projection. See
    :mod:`tengri.components.agn.cat3d_wind` and
    ``scripts/build_cat3d_wind_grid.py``.
    """
    l_disc = powerlaw_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0 - agn_torus_frac,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)
    l_torus = cat3d_wind_analytic(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_cos_inc=agn_cos_inc,
        agn_a_cat3d=agn_a_cat3d,
        agn_fwd_cat3d=agn_fwd_cat3d,
        agn_torus_frac=agn_torus_frac,
    )
    return (l_disc + l_torus) * agn_frac


@register_agn_model("adaf")
def adaf_agn(
    wavelength: jnp.ndarray,
    agn_log_lbol: float,
    agn_frac: float = 0.1,
    agn_log_mbh: float = 8.0,
    agn_log_ledd: float = -3.0,
    agn_r_tr: float = 100.0,
    agn_adaf_beta: float = 0.5,
    agn_adaf_delta: float = 0.01,
    agn_cos_inc: float = 0.5,
    agn_torus_frac: float = 0.3,
    agn_T_torus: float = 500.0,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """ADAF + truncated disc + simple torus for low-luminosity AGN.

    At low accretion rates (L/L_Edd < 0.01), the inner disc transitions
    to an ADAF. The outer disc remains as a truncated Shakura-Sunyaev disc.
    A simple torus re-emits a fraction of the bolometric luminosity in the IR.

    6 free parameters (+ agn_frac scaling):

    - agn_log_mbh: BH mass
    - agn_log_ledd: Eddington ratio (should be < -2 for ADAF regime)
    - agn_r_tr: truncation radius [R_g]
    - agn_adaf_beta: magnetic pressure fraction
    - agn_adaf_delta: electron heating fraction
    - agn_torus_frac: torus covering factor

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10 of bolometric luminosity [Lsun].
    agn_frac : float, optional
        AGN luminosity fraction [dimensionless]. Default 0.1.
    agn_log_mbh : float, optional
        log10 of black hole mass [Msun]. Default 8.0.
    agn_log_ledd : float, optional
        log10 of Eddington ratio [dimensionless]. Default -3.0.
    agn_r_tr : float, optional
        Truncation radius [R_g]. Default 100.
    agn_adaf_beta : float, optional
        Magnetic pressure fraction [dimensionless], range [0, 1]. Default 0.5.
    agn_adaf_delta : float, optional
        Electron heating fraction [dimensionless], range [0, 1]. Default 0.01.
    agn_cos_inc : float, optional
        cos(inclination) [dimensionless]. Default 0.5.
    agn_torus_frac : float, optional
        Torus covering factor [dimensionless]. Default 0.3.
    agn_T_torus : float, optional
        Torus temperature [K]. Default 500.

    Returns
    -------
    ndarray, shape (n_wave,)
        L_nu [erg/s/Hz].

    Notes
    -----
    **JIT-compatible**: yes — uses :func:`adaf_disc` and :func:`simple_torus`.
    """
    # ADAF + truncated disc gets (1 - torus_frac) of L_bol
    l_disc = adaf_disc(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_frac=1.0 - agn_torus_frac,
        agn_log_mbh=agn_log_mbh,
        agn_log_ledd=agn_log_ledd,
        agn_r_tr=agn_r_tr,
        agn_adaf_beta=agn_adaf_beta,
        agn_adaf_delta=agn_adaf_delta,
        agn_cos_inc=agn_cos_inc,
    )
    l_disc = _redden_disc(wavelength, l_disc, agn_ebv_disc)

    # Simple torus re-emits torus_frac of L_bol
    l_torus = simple_torus(
        wavelength,
        agn_log_lbol=agn_log_lbol,
        agn_torus_frac=agn_torus_frac,
        agn_T_torus=agn_T_torus,
    )

    return (l_disc + l_torus) * agn_frac


@register_agn_model("relagn")
def relagn_agn(
    wavelength: jnp.ndarray,
    agn_log_mbh: float = 8.0,
    agn_log_mdot: float = -1.0,
    agn_astar: float = 0.0,
    agn_cos_inc: float = 0.5,
    agn_torus_frac: float = 0.5,
    agn_T_hot: float = 1200.0,
    agn_T_warm: float = 300.0,
    agn_frac_hot: float = 0.3,
    agn_ebv_disc: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """RELAGN relativistic outer disc + two-temperature dust torus.

    Uses the precomputed RELAGN grid (Hagen & Done 2023) with KYCONV
    (Dovciak, Karas & Yaqoob 2004) per-annulus Kerr ray-tracing for the
    disc, and a two-temperature modified blackbody for the torus.

    The disc luminosity is self-consistent with BH mass and accretion rate;
    the torus re-emits ``agn_torus_frac`` of the disc bolometric output.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength. [Å]
    agn_log_mbh : float, optional
        ``log10(M_BH / M_sun)``, range [7, 10]. Default 8.0.
    agn_log_mdot : float, optional
        ``log10(Ṁ / Ṁ_Edd)``, range [−1.5, 0.3]. Default −1.0.
    agn_astar : float, optional
        Dimensionless BH spin, prograde only, range [0, 0.998]. Default 0.0.
    agn_cos_inc : float, optional
        Cosine of inclination (1 = face-on). Default 0.5.
    agn_torus_frac : float, optional
        Torus covering factor; torus re-emits this fraction of disc L_bol.
        Disc is attenuated by ``(1 − agn_torus_frac)``. Default 0.5.
    agn_T_hot : float, optional
        Hot dust temperature [K]. Default 1200.
    agn_T_warm : float, optional
        Warm dust temperature [K]. Default 300.
    agn_frac_hot : float, optional
        Hot dust mass fraction. Default 0.3.
    agn_ebv_disc : float, optional
        SMC-law colour excess applied to disc [mag]. Default 0.0.

    Returns
    -------
    ndarray, shape (n_wave,)
        Total AGN L_ν. [erg s⁻¹ Hz⁻¹]

    Notes
    -----
    **JIT-compatible**: yes — disc interpolation is pure JAX triweight kernel.

    **Gradient-safe**: yes — C²-continuous triweight kernel on all grid axes.

    **Grid required**: ``data/relagn_disc_grid.h5`` built by
    ``scripts/build_relagn_disc_grid.py`` (requires HEASOFT/XSPEC + KYCONV).

    **Torus normalization**: derived by integrating the disc L_ν over the
    output wavelength grid via ``jnp.trapezoid`` — no separate ``agn_log_lbol``
    parameter needed.

    References
    ----------
    .. [1] Dovciak, M., Karas, V., & Yaqoob, T. (2004).
       ApJS, 153, 205. doi:10.1086/421115  [KYCONV]

    .. [2] Hagen, S. & Done, C. (2023).
       MNRAS, 521, 251. doi:10.1093/mnras/stad478  [RELAGN]
    """
    disc_fn = _load_relagn_fn()

    # Full disc at reference inclination from precomputed KYCONV grid
    l_disc_full = disc_fn(
        wavelength,
        agn_log_mbh=agn_log_mbh,
        agn_log_mdot=agn_log_mdot,
        agn_astar=agn_astar,
        agn_cos_inc=agn_cos_inc,
    )
    l_disc_full = _redden_disc(wavelength, l_disc_full, agn_ebv_disc)

    # Attenuate disc by torus covering factor
    l_disc = l_disc_full * (1.0 - agn_torus_frac)

    # Derive disc L_bol by integrating L_ν over ν (trapezoid in JAX)
    _c_aa = 2.99792458e18  # Å/s
    nu = _c_aa / wavelength  # decreasing
    # Sort ascending for trapezoid
    lbol_disc_erg = jnp.trapezoid(jnp.flip(l_disc_full), jnp.flip(nu))
    log_lbol_lsun = jnp.log10(jnp.maximum(lbol_disc_erg, 1e30)) - jnp.log10(3.839e33)

    # Torus re-emits agn_torus_frac of disc L_bol
    l_torus = two_temperature_torus(
        wavelength,
        agn_log_lbol=log_lbol_lsun,
        agn_torus_frac=agn_torus_frac,
        agn_T_hot=agn_T_hot,
        agn_T_warm=agn_T_warm,
        agn_frac_hot=agn_frac_hot,
    )

    return l_disc + l_torus


# ── Geometric masking (smooth sigmoid for differentiability) ──────


def _sigmoid_mask(
    cos_inc: float,
    theta_torus: float,
    width: float = 2.0,
) -> float:
    """Smooth geometric mask for disc/BLR visibility.

    Returns ~1 (visible) for face-on orientations and ~0 (obscured)
    when the line of sight passes through the torus.

    The critical angle is ``inc_crit = 90° - theta_torus``: above this
    the torus intercepts the sightline. In Synthesizer [1]_ this condition
    is implemented as a hard binary step (``inclination + theta_torus > 90°``
    → zeroed). tengri replaces the hard step with a smooth sigmoid to
    preserve differentiability for gradient-based inference.

    .. math::

        \\sigma(i, \\theta_t) = \\mathrm{sigmoid}\\!\\left(
            -\\frac{\\arccos(\\cos i) - (90^\\circ - \\theta_t)}{w}
        \\right)

    where :math:`w` is the transition half-width in degrees (default 2°).
    At :math:`w \\to 0` this converges to Synthesizer's hard binary mask.

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

    Notes
    -----
    JIT/grad/vmap compatible.

    References
    ----------
    .. [1] Synthesizer ``torus_edgeon_condition``:
           https://github.com/synthesizer-project/synthesizer/blob/main/src/synthesizer/emission_models/agn/unified_agn.py
    """
    # Inclination in degrees: inc = arccos(cos_inc)
    inc_deg = jnp.degrees(jnp.arccos(jnp.clip(cos_inc, 0.0, 1.0)))
    # Critical angle: above this the torus blocks the view
    inc_crit = 90.0 - jnp.clip(theta_torus, 0.0, 90.0)
    # Sigmoid: 1 when inc << inc_crit, 0 when inc >> inc_crit
    return jax.nn.sigmoid(-(inc_deg - inc_crit) / jnp.maximum(width, 0.1))


# ── Unified AGN with NLR + BLR decomposition ──────────────────────


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
    agn_polar_ebv: float = 0.0,
    nlr_fn: "Callable | None" = None,
    blr_fn: "Callable | None" = None,
    **_kwargs,
) -> jnp.ndarray:
    """Unified AGN SED with NLR/BLR decomposition and geometric masking.

    Implements the same conceptual decomposition as the ``UnifiedAGN`` class
    in Synthesizer (Lovell et al. 2025 [1]_, Roper et al. 2025 [2]_):
    an accretion disc, dusty torus, narrow line region (NLR), and broad line
    region (BLR), combined with inclination-dependent geometric masking.
    Polar dust reddening (SMC law) is additionally adapted from CIGALE's
    ``skirtor2016`` module [3]_.

    .. rubric:: Correspondence with Synthesizer's UnifiedAGN

    The following table maps tengri parameters to Synthesizer attributes
    (``synthesizer.components.blackhole.BlackholeComponent``):

    ==================  =====================================  ============================
    tengri              Synthesizer                            Note
    ==================  =====================================  ============================
    agn_cos_inc         ``cos(inclination)``                   Synth uses inclination [deg]
    agn_theta_torus     ``theta_torus`` [deg]                  Same meaning
    agn_covering_nlr    ``covering_fraction_nlr``              Identical semantics
    agn_covering_blr    ``covering_fraction_blr``              Identical semantics
    agn_torus_frac      ``torus_fraction = theta_torus/90°``   **Different**: independent param
    agn_blr_fwhm        ``velocity_dispersion_blr`` [km/s]     tengri uses FWHM directly
    agn_nlr_fwhm        ``velocity_dispersion_nlr`` [km/s]     Same
    ==================  =====================================  ============================

    .. rubric:: Deliberate differences from Synthesizer

    1. **Analytic disc, not a grid**: Synthesizer extracts disc emission from
       precomputed CLOUDY photoionisation grids. tengri uses the closed-form
       ``multicolor_disc`` (Shakura-Sunyaev / Novikov-Thorne) from
       ``disc.py``. Rationale: grid look-ups are not JAX-jittable under
       gradient tape; analytic models are differentiable by construction.

    2. **Analytic NLR/BLR templates, not grids**: Synthesizer uses CLOUDY
       grids over (logU, log n_H) for both NLR and BLR emission. tengri uses
       empirically calibrated Gaussian line templates (Vanden Berk et al.
       2001 for BLR; Groves et al. 2004 for NLR). Rationale: same as above.

    3. **Smooth sigmoid mask, not a hard binary**: Synthesizer zeros the disc
       and BLR when ``inclination + theta_torus > 90°`` — a hard step
       function. tengri replaces this with a smooth sigmoid (see
       ``_sigmoid_mask``) centred at the same critical angle with a ~2°
       transition width. Rationale: the hard mask has zero gradient with
       respect to inclination almost everywhere, making gradient-based VI
       and HMC blind to inclination information near the critical angle.

    4. **``agn_torus_frac`` decoupled from ``theta_torus``**: Synthesizer
       internally sets ``torus_fraction = theta_torus / 90°``, coupling the
       reprocessed fraction to the geometric opening angle. tengri keeps
       ``agn_torus_frac`` as an independent free parameter. Rationale:
       auto-deriving from ``cos(theta_torus)`` in the forward pass introduces
       a gradient discontinuity at the poles (CLAUDE.md gotcha). The coupling
       can be enforced at the ``Parameters`` level via a fixed/derived param.

    5. **Polar dust reddening (SMC law)**: ``agn_polar_ebv`` applies SMC
       extinction to the disc and BLR for Type 1 sightlines (mask > 0).
       Absent from Synthesizer's basic ``UnifiedAGNIntrinsic``; adapted from
       CIGALE ``skirtor2016`` (Yang et al. 2020 [3]_).

    .. rubric:: Geometry summary

    .. math::

        L_{\\rm AGN}(\\lambda) =
            \\sigma(i, \\theta_t)\\, A_{\\rm pol}(\\lambda)\\, L_{\\rm disc}(\\lambda)
            + L_{\\rm torus}(\\lambda)
            + f_{\\rm NLR}\\, \\eta_{\\rm NLR}(\\lambda)\\, L_{\\rm bol,disc}
            + \\sigma(i, \\theta_t)\\, A_{\\rm pol}(\\lambda)\\,
              f_{\\rm BLR}\\, \\eta_{\\rm BLR}(\\lambda)\\, L_{\\rm bol,disc}

    where :math:`\\sigma(i, \\theta_t)` is the smooth sigmoid mask
    (1 = visible, 0 = obscured), :math:`A_{\\rm pol}` is the SMC polar dust
    transmission, :math:`f_{\\rm NLR/BLR}` are the covering fractions, and
    :math:`\\eta_{\\rm NLR/BLR}` are the normalised line templates.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_log_lbol : float
        log10(L_bol / Lsun). Total AGN bolometric luminosity. Default 44.0.
    agn_cos_inc : float
        Cosine of inclination angle (0 = edge-on/Type 2,
        1 = face-on/Type 1). Synthesizer uses inclination in degrees;
        tengri uses cos(inclination). Default 0.5.
    agn_theta_torus : float
        Torus half-opening angle [degrees]. Controls the critical inclination
        above which the disc and BLR are obscured. Same meaning as
        ``theta_torus`` in Synthesizer. Default 30.0.
    agn_covering_nlr : float
        NLR covering fraction (0 to 1). Fraction of disc luminosity
        reprocessed into NLR emission. Synthesizer: ``covering_fraction_nlr``.
        Default 0.1.
    agn_covering_blr : float
        BLR covering fraction (0 to 1). Fraction of disc luminosity
        reprocessed into BLR emission. Synthesizer: ``covering_fraction_blr``.
        Default 0.1.
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
        In Synthesizer this is derived as ``theta_torus / 90°``; here it is
        an independent free parameter. Default 0.5.
    agn_frac : float
        Overall AGN fraction scaling. Default 0.1.
    agn_blr_fwhm : float
        BLR line FWHM [km/s]. Synthesizer uses ``velocity_dispersion_blr``;
        tengri uses FWHM directly. Default 5000.
    agn_nlr_fwhm : float
        NLR line FWHM [km/s]. Default 500.
    agn_polar_ebv : float
        Polar dust E(B-V) applied to disc + BLR for Type 1 views (SMC law).
        Adapted from CIGALE skirtor2016 [3]_. Not in Synthesizer's basic
        UnifiedAGN. Default 0.0 (no reddening).
    nlr_fn : callable or None
        NLR emission backend. Signature::

            nlr_fn(wavelength, l_disc_bol_erg, covering_fraction,
                   fwhm_kms=500.0, **kwargs) -> ndarray, shape (n_wave,)

        where the return value is L_nu [erg/s/Hz]. Default ``None`` uses the
        built-in analytic template from :func:`~tengri.components.agn.nlr.nlr_emission`.
        To use the Cue neural-net emulator, pass the result of
        :func:`make_cue_nlr_fn`.
    blr_fn : callable or None
        BLR emission backend. Same signature as ``nlr_fn``. Default ``None``
        uses the built-in analytic template from
        :func:`~tengri.components.agn.blr.blr_emission`.

    Returns
    -------
    array, shape (n_wave,)
        Total AGN L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    JIT/grad/vmap compatible when ``nlr_fn`` and ``blr_fn`` are JIT-compatible
    (the default analytic templates are; Cue-backed closures are also JIT-safe
    since the weights are closed over as static pytree leaves).

    References
    ----------
    .. [1] Lovell C. C. et al. 2025, Open Journal of Astrophysics,
           "Synthesizer: a Software Package for Synthetic Astronomical Observables",
           arXiv:2508.03888, https://doi.org/10.48550/arXiv.2508.03888
    .. [2] Roper W. J. et al. 2025, arXiv:2506.15811,
           "Synthesizer: Synthetic Observables For Modern Astronomy",
           https://doi.org/10.48550/arXiv.2506.15811
    .. [3] G. Yang et al. 2020, MNRAS, 491, 740, "X-CIGALE: Fitting AGN/galaxy
           SEDs from X-ray to infrared" (skirtor2016 module),
           https://doi.org/10.1093/mnras/stz3001

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import unified_nlr_blr
    >>> wave = jnp.linspace(1000.0, 30000.0, 512)
    >>> sed = unified_nlr_blr(wave, agn_log_lbol=45.0, agn_cos_inc=0.8)
    >>> sed.shape
    (512,)
    >>> bool(jnp.all(sed >= 0))
    True
    """
    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG

    # Derive torus covering factor from opening angle for consistency:
    # covering_factor ~ cos(theta_torus). If agn_torus_frac is at default
    # (0.5), use the geometric value; otherwise honour the explicit setting.
    geom_cf = jnp.cos(jnp.radians(jnp.clip(agn_theta_torus, 0.0, 90.0)))
    # When agn_torus_frac is a free parameter, use it directly.
    # The jnp.where(|x-0.5| < 1e-6, geom_cf, x) creates a likelihood discontinuity
    # at torus_frac=0.5±1e-6 that corrupts gradient-based inference (VI, MAP).
    # Auto-derivation from theta_torus is done at the Parameters level via fixed values,
    # not in the forward pass.
    _ = geom_cf  # retained for reference; used when agn_torus_frac is fixed by Parameters

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
    # --- Polar dust extinction (Type 1 reddening, SMC law) ---
    # Applied to disc and BLR when visible (mask > 0).
    # Uses SMC law since AGN sightlines typically lack a 2175 A bump.
    from tengri.components.dust.attenuation import smc as _smc_law

    k_polar = _smc_law(wavelength)  # k(λ)/k(V), normalized at 5500 A
    # R_V(SMC) = 2.93; A(λ) = E(B-V) * R_V * k(λ)
    _RV_SMC = 2.93
    polar_trans = jnp.exp(-0.921 * agn_polar_ebv * _RV_SMC * k_polar)

    # Apply geometric masking + polar dust to disc
    l_disc_masked = mask_disc * l_disc * polar_trans

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
    # NLR is illuminated by the disc bolometric luminosity.
    # Pluggable: use nlr_fn if provided, else the built-in analytic template.
    l_disc_bol_erg = (1.0 - agn_torus_frac) * l_bol_erg
    _nlr_fn = nlr_fn if nlr_fn is not None else nlr_emission
    l_nlr = _nlr_fn(
        wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_covering_nlr,
        fwhm_kms=agn_nlr_fwhm,
        **_kwargs,
    )

    # --- BLR emission (masked by torus) ---
    # Pluggable: use blr_fn if provided, else the built-in analytic template.
    _blr_fn = blr_fn if blr_fn is not None else blr_emission
    l_blr_raw = _blr_fn(
        wavelength,
        l_disc_bol_erg=l_disc_bol_erg,
        covering_fraction=agn_covering_blr,
        fwhm_kms=agn_blr_fwhm,
        **_kwargs,
    )
    l_blr = mask_blr * polar_trans * l_blr_raw

    # --- Total ---
    l_total = l_disc_masked + l_torus + l_nlr + l_blr
    return l_total * agn_frac


# ── Pluggable NLR/BLR backend factories ──────────────────────────


def make_cue_nlr_fn(
    weights,
    gas_logU: float = -2.0,
    gas_logZ: float = 0.0,
    gas_log_nH: float = 2.0,
    gas_xi_d: float = 0.3,
    ionspec_index1: float = 1.7,
    ionspec_logLratio1: float = 0.0,
    ionspec_index2: float = -0.5,
    ionspec_logLratio2: float = 0.0,
    ionspec_index3: float = -1.5,
    ionspec_logLratio3: float = 0.0,
    ionspec_index4: float = -3.0,
    gas_logqion: float = 49.1,
):
    """Create a Cue-backed NLR emission callable for ``unified_nlr_blr``.

    Returns a closure that uses the Cue neural-net emulator (Li et al. 2024
    [1]_) to predict NLR emission lines and continuum as a function of
    ionising spectrum shape and gas properties. The closure matches the
    standard NLR backend interface expected by :func:`unified_nlr_blr`.

    The Cue weights are closed over as static pytree leaves, so the returned
    callable is JIT-compatible.

    Parameters
    ----------
    weights : CueWeights
        Pre-loaded Cue network weights from
        :func:`~tengri.components.nebular.cue.load_cue_weights`.
    gas_logU : float
        log10(ionization parameter U). Default -2.0.
    gas_logZ : float
        log10(Z/Zsun) of the NLR gas. Default 0.0 (solar).
    gas_log_nH : float
        log10(hydrogen number density [cm^-3]). Default 2.0.
    gas_xi_d : float
        Dust-to-metal ratio. Default 0.3.
    ionspec_index1..4 : float
        Power-law indices of the ionising spectrum piecewise approximation.
        Defaults correspond to a typical Seyfert 1 AGN spectrum.
    ionspec_logLratio1..3 : float
        log10(L) ratios between power-law segments. Default 0.0.
    gas_logqion : float
        log10(Q_H) — total ionising photon rate [photons/s]. Used for
        normalisation. Default 49.1 (typical AGN at log L_bol ~ 44).

    Returns
    -------
    callable
        NLR emission function with signature::

            fn(wavelength, l_disc_bol_erg, covering_fraction,
               fwhm_kms=500.0, **kwargs) -> ndarray, shape (n_wave,)

        The returned L_nu is in [erg/s/Hz].

    Notes
    -----
    JIT/grad compatible when Cue weights are registered as JAX pytrees.
    The weights must first be loaded via
    ``tengri.components.nebular.cue.load_cue_weights``.

    The Cue emulator was trained for AGN-illuminated gas (not stellar HII
    regions). It is appropriate for NLR emission; it does NOT model the BLR
    (which requires a separate, denser photoionisation model with
    n_H ~ 10^10 cm^-3 and no dust).

    References
    ----------
    .. [1] Li Z. et al. 2024, ApJ, 969, 28,
           "Cue: An Emulator for AGN-Dominated Emission",
           https://doi.org/10.3847/1538-4357/ad44a8
    """
    from tengri.components.agn._phys import gaussian_line_profile as _glp
    from tengri.components.nebular.cue import _prepare_nn_params, predict_all_lines
    from tengri.utils.physics_constants import L_SUN as _LSUN

    # Build the fixed 12-element NN parameter vector once.
    # These are the gas/ionspec parameters that do NOT change per call.
    # l_disc_bol_erg rescales logqion per call below.
    _nn_params_base = _prepare_nn_params(
        gas_logU=gas_logU,
        gas_logZ=gas_logZ,
        gas_log_nH=gas_log_nH,
        gas_xi_d=gas_xi_d,
        ionspec_index1=ionspec_index1,
        ionspec_logLratio1=ionspec_logLratio1,
        ionspec_index2=ionspec_index2,
        ionspec_logLratio2=ionspec_logLratio2,
        ionspec_index3=ionspec_index3,
        ionspec_logLratio3=ionspec_logLratio3,
        ionspec_index4=ionspec_index4,
    )

    def _cue_nlr(
        wavelength: jnp.ndarray,
        l_disc_bol_erg: float,
        covering_fraction: float = 0.1,
        fwhm_kms: float = 500.0,
        **kw,
    ) -> jnp.ndarray:
        """Predict narrow-line region emission lines from disc ionizing photon rate."""
        # Infer log10(Q_H) from l_disc_bol_erg if provided.
        # Approximate: log10(Q_H) ~ log10(L_bol_erg) - log10(13.6 eV)
        # For a power-law spectrum with <E> ~ 13.6 eV (1 Ryd).
        _logqion = jnp.log10(jnp.maximum(l_disc_bol_erg, 1e30)) - jnp.log10(2.18e-11)

        # Predict line wavelengths [Angstrom] and luminosities [Lsun]
        wav_lines, lum_lsun = predict_all_lines(
            _nn_params_base, weights, gas_logU + jnp.log10(4.0), _logqion
        )

        # Convert Lsun → erg/s and apply covering fraction
        lum_erg = lum_lsun * _LSUN * covering_fraction

        # Place each line on the wavelength grid via Gaussian broadening
        l_nu = jnp.zeros_like(wavelength)
        for i in range(wav_lines.shape[0]):
            profile = _glp(wavelength, wav_lines[i], fwhm_kms)
            l_nu = l_nu + lum_erg[i] * profile

        return l_nu

    return _cue_nlr
