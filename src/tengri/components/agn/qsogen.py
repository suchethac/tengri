# SPDX-License-Identifier: BSD-3-Clause
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

1. **Always free:** ``agn_log_lbol`` (or ``agn_lum_ratio``) -- the AGN
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

Diagnostic colors for AGN contamination:

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
Emission lines        4 empirical template interpolation  Same templates (qsosed_emlines)
Baldwin effect slope  0.183                               0.183 (matching original)
Balmer continuum      Added before flux normalization     Added after L_bol normalization
Host galaxy template  S0 SWIRE included                   Not included (handled by Model)
IGM absorption        Becker+13 tau_eff                   Handled by tengri IGM module
Autodiff              No                                  Yes (JAX JIT-compatible)
===================== ================================== ====================================

References
----------

- Temple, Hewett & Banerji 2021, MNRAS, 508, 737
- Vanden Berk et al. 2001, AJ, 122, 549 (emission line EWs)
- Gordon et al. 2003, ApJ, 594, 279 (SMC extinction)
- Stern et al. 2012, ApJ, 753, 30 (W1-W2 AGN selection)

"""

import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.agn._params import DEFAULT_AGN_LOG_LBOL, DEFAULT_AGN_LUM_RATIO
from tengri.components.agn._phys import bolometric_integral_nu as _bolometric_integral_nu
from tengri.components.dust.attenuation import smc as smc_curve

# ── Physical constants (CGS) ──────────────────────────────────────
from tengri.utils.physics_constants import (
    AA_TO_CM as _ANGSTROM_CM,
    C_CGS as _C_LIGHT,
    L_SUN as _LSUN_ERG,
)
from tengri.utils.scale import representable_exponent

# Normalization wavelength
_LAMBDA_NORM = 5500.0  # Angstrom

# Reference wavelength for hot dust anchoring
_LAMBDA_BB_ANCHOR = 20000.0  # 2 um in Angstrom

# Short-wavelength floor of the disc continuum [Angstrom]. QSOgen is an
# empirical template with no X-ray physics; its broken power law would
# otherwise extrapolate unbounded into the X-ray and double-count with the
# alpha_ox corona. 124 A = 0.1 keV is the corona's blue edge — the exact
# ``wavelength < 124.0`` band used in components/xray/xray.py — so the disc
# owns lambda >= 124 A (optical/UV/EUV) and the corona owns lambda < 124 A
# (0.1-10 keV) with no overlap. See issue #1113.
_XRAY_FLOOR_LAMBDA_AA = 124.0

# ── Default parameters (Temple+2021 Table 3) ──────────────────────

_DEFAULT_PLSLP1 = -0.349  # Blue power-law slope (f_nu ~ nu^alpha)
_DEFAULT_PLSLP2 = 0.593  # Red power-law slope
_DEFAULT_PLBRK = 3880.0  # Break wavelength [Angstrom]
_DEFAULT_TBB = 1240.0  # Hot dust temperature [K]
_DEFAULT_BBNORM = 3.96  # Hot dust normalization
_DEFAULT_EMLINE_SCALE = 1.0  # Emission line strength multiplier
_DEFAULT_EBV = 0.0  # E(B-V) dust reddening

# Baldwin effect slope (original Temple+2021 value)
_BALDWIN_SLOPE = 0.183

# Reference M_i for Baldwin effect normalization (original benorm)
_BENORM = -27.0

# Sigmoid transition half-width in log-wavelength space
_SIGMOID_WIDTH = 0.02  # dex (~5% in wavelength)


# ── Emission line template (loaded lazily from data file) ─────────
# Template format: 6 rows x N wavelengths
#   row 0: wavelength (Angstrom)
#   row 1: median emission lines (flux units, relative to reference continuum)
#   row 2: reference continuum (for EW-scaling normalization)
#   row 3: peaky (high-EW) line template
#   row 4: windy (high-blueshift) line template
#   row 5: narrow optical line template
#
# The original qsogen uses EW-scaling (scal_emline < 0):
#   flux_with_lines = flux * (1 + |scal| * linval / conval)
# This preserves EW ratios relative to the actual continuum.


def _load_emline_template_arrays():
    """Load the empirical emission line template arrays eagerly (NumPy, not JAX).

    This function loads the emission-line template data at import time to avoid
    file I/O and generator evaluation inside JIT-traced functions, which would
    cause UnexpectedTracerError (BUG-NSS-03).

    Returns
    -------
    tuple of np.ndarray
        (wavelength, median_lines, reference_continuum, peaky, windy, narrow)

    Raises
    ------
    FileNotFoundError
        If the template file is not found in any of the searched locations.
    """
    from tengri._data_setup import find_data

    # find_data covers the old parents[4] and cwd-relative candidates, plus
    # $TENGRI_DATA_DIR (#1431); the /tmp path is the upstream drop location.
    candidates = [
        find_data("qsogen_emline_template.dat"),
        Path("/tmp/qsogen/qsosed_emlines_20210625.dat"),
    ]
    for path in candidates:
        if path is None:
            continue
        if path.is_file():
            data = np.genfromtxt(str(path), unpack=True)
            # Materialize each row as a fully-realized np.ndarray (not generator)
            # to avoid tracer leaks when accessed inside JIT-compiled functions.
            wavelength = np.asarray(data[0], dtype=np.float64)
            median_lines = np.asarray(data[1], dtype=np.float64)
            reference_continuum = np.asarray(data[2], dtype=np.float64)
            peaky = np.asarray(data[3], dtype=np.float64)
            windy = np.asarray(data[4], dtype=np.float64)
            narrow = np.asarray(data[5], dtype=np.float64)
            return wavelength, median_lines, reference_continuum, peaky, windy, narrow

    raise FileNotFoundError(
        "QSOGen emission line template not found. Expected at data/qsogen_emline_template.dat"
    )


# Load emission-line template arrays at module import time to avoid tracer leaks
# in JIT-compiled functions (BUG-NSS-03 fix). Arrays are materialized as NumPy
# and stored as module-level closures referenced by _empirical_emission_lines.
try:
    _EMLINE_WAV, _EMLINE_MED, _EMLINE_REF, _EMLINE_PEAKY, _EMLINE_WINDY, _EMLINE_NARROW = (
        _load_emline_template_arrays()
    )
except FileNotFoundError:
    # If template file is missing, set to None. _empirical_emission_lines will
    # raise a clear error at runtime when called.
    _EMLINE_WAV = None

# Redshift-luminosity relation from SDSS DR16Q (Temple+2021 config.py)
_ZLUM = np.array([0.23, 0.34, 0.6, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.3, 3.7, 4.13, 4.5])
_LUMVAL = np.array(
    [-21.76, -22.9, -24.1, -25.4, -26.0, -26.6, -27.1, -27.6, -27.9, -28.1, -28.4, -28.6, -28.9]
)


# ── Helper functions ──────────────────────────────────────────────


def _wavelength_to_nu(wavelength: jnp.ndarray) -> jnp.ndarray:
    """Convert wavelength (Angstrom) to frequency (Hz)."""
    return _C_LIGHT / (wavelength * _ANGSTROM_CM)


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

    # Floor the continuum below the alpha_ox corona's blue edge (0.1 keV). The
    # broken power law has no intrinsic short-wavelength cutoff, so left free it
    # extrapolates the EUV segment into the X-ray (nu*L_nu ~ nu^-0.349) and
    # double-counts with the corona (#1113). This is a hard cut at a fixed grid
    # wavelength (not a free parameter), mirroring the corona's own
    # ``wavelength < 124.0`` band; the gradient w.r.t. the shape parameters
    # flows unchanged for lambda >= 124 A. Applied here, before the caller's
    # bolometric normalization, so the removed flux no longer dilutes L_bol.
    continuum = f_nu / jnp.maximum(f_norm, 1e-30)
    return jnp.where(wavelength >= _XRAY_FLOOR_LAMBDA_AA, continuum, 0.0)


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
        Hot dust f_lambda **component only** (not the total SED).
        Returns zero everywhere when ``bbnorm=0``.
        The caller (``qsogen_sed``) adds this to the other components.
    """
    # B_nu as function of wavelength (matching original bb() exactly):
    #   bb(T, wav) = wav^{-3} / (exp(hc/kT*wav) - 1)
    hc_over_k = 1.43877735e8  # h*c/k_B in Kelvin*Angstrom
    x = hc_over_k / (tbb * jnp.maximum(wavelength, 1.0))
    bb_fnu = wavelength ** (-3.0) / (
        jnp.exp(jnp.clip(x, 0.0, representable_exponent(500.0, base=math.e))) - 1.0
    )

    # Normalize: bbnorm is the ratio f_bb(2μm) / f_cont(2μm) (Temple+2021).
    # Evaluate the BB and continuum at the 2μm anchor to get the relative scale.
    x_anchor = hc_over_k / (tbb * _LAMBDA_BB_ANCHOR)
    bb_anchor = _LAMBDA_BB_ANCHOR ** (-3.0) / (
        jnp.exp(jnp.clip(x_anchor, 0.0, representable_exponent(500.0, base=math.e))) - 1.0
    )
    cont_at_anchor = jnp.interp(
        jnp.array([_LAMBDA_BB_ANCHOR]), wavelength, continuum_flam, left=0.0, right=0.0
    )[0]
    cmult = bbnorm * jnp.maximum(cont_at_anchor, 1e-60) / jnp.maximum(bb_anchor, 1e-60)

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
        Balmer continuum f_lambda **component only** (not the total SED).
        Returns zero everywhere when ``bcnorm=0`` or above the Balmer edge.
        The caller (``qsogen_sed``) adds this to the other components.
    """
    # B_nu as function of wavelength (matching original bb() convention)
    # bb(T, wav) = wav^{-3} / (exp(hc/kT*wav) - 1)
    hc_over_k = 1.43877735e8  # h*c/k_B in Kelvin * Angstrom
    x = hc_over_k / (tbc * jnp.maximum(wavelength, 1.0))
    x_clip = jnp.clip(x, 0.0, representable_exponent(500.0, base=math.e))
    b_nu_wav = wavelength ** (-3.0) / (jnp.exp(x_clip) - 1.0)

    # Optical depth: sigma_bf(nu) ~ nu^{-3} (Osterbrock & Ferland, AGN^2 Eq. 2.4), so
    # tau(lambda) = tau_BE * (lambda_BE / lambda)^3 — tau INCREASES at shorter wavelengths
    # (higher frequencies), reaching tau_BE at the Balmer edge and falling beyond.
    tau = taube * (wavbe / wavelength) ** 3
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
    b_norm = wnorm ** (-3.0) / (
        jnp.exp(jnp.clip(x_norm, 0.0, representable_exponent(500.0, base=math.e))) - 1.0
    )
    tau_norm = taube * (wnorm / wavbe) ** 3
    bc_at_3000 = b_norm * (1.0 - jnp.exp(-jnp.clip(tau_norm, 0.0, 50.0)))
    bc_at_3000 = jnp.maximum(bc_at_3000, 1e-60)

    scale = bcnorm * cont_at_3000 / bc_at_3000

    # Only below Balmer edge (smooth sigmoid transition)
    steepness = 1.0 / _SIGMOID_WIDTH
    below_edge = jax.nn.sigmoid(-steepness * (jnp.log10(wavelength) - jnp.log10(wavbe)))

    return scale * bc_flux * below_edge


def _empirical_emission_lines(
    wavelength: jnp.ndarray,
    continuum_fnu: jnp.ndarray,
    emline_scale: float,
    m_i: float,
) -> jnp.ndarray:
    """Empirical emission line spectrum (Temple+2021 templates).

    Uses the full empirical emission line templates from the original
    qsogen, which include FeII pseudo-continuum, blended line complexes,
    and realistic line profiles from the SDSS DR16Q composite.

    The template is applied via EW-scaling (matching the original's
    ``scal_emline < 0`` path): ``flux_lines = |scal| * linval * flux / conval``.
    This preserves equivalent widths relative to the actual continuum.

    The dimensionless ratio ``linval / conval`` is the same in f_nu and
    f_lambda, so this works directly in f_nu space.

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    continuum_fnu : array, shape (n_wave,)
        Continuum f_nu (power-law + hot dust, normalized ~1 at 5500 A).
    emline_scale : float
        Overall emission line strength multiplier (positive values scale
        line fluxes; negative values preserve EW ratios). Default usage
        is negative (EW-scaling), matching original qsogen.
    m_i : float
        Absolute i-band magnitude (at z=2) for Baldwin effect scaling.
        Controls the ``emline_type`` parameter via ``beslope``.

    Returns
    -------
    array, shape (n_wave,)
        Emission line f_nu contribution (same units as continuum_fnu).
    """
    # Use module-level emission-line template arrays (loaded at import time
    # to avoid file I/O and tracer leaks inside JIT scope). BUG-NSS-03 fix.
    if _EMLINE_WAV is None:
        raise FileNotFoundError(
            "QSOGen emission line template not found. Expected at data/qsogen_emline_template.dat"
        )

    linwav = _EMLINE_WAV
    medval = _EMLINE_MED
    conval_raw = _EMLINE_REF
    pkyval = _EMLINE_PEAKY
    wdyval = _EMLINE_WINDY
    _nlr = _EMLINE_NARROW

    # Baldwin effect: emline_type = (M_i - benorm) * beslope
    # beslope > 0, benorm = -27 -> brighter quasars (more negative M_i)
    # get emline_type < 0 -> more blueshifted lines
    emline_type = (m_i - _BENORM) * _BALDWIN_SLOPE

    # Combine templates based on emline_type (matching original exactly)
    # emline_type = 0: average template
    # emline_type > 0: blend toward peaky (high-EW)
    # emline_type < 0: blend toward windy (high-blueshift)
    varlin_pos = jnp.clip(emline_type, 0.0, 3.0)
    varlin_neg = jnp.clip(-emline_type, 0.0, 2.0)

    # Smooth blend between positive and negative regimes
    is_positive = jax.nn.sigmoid(emline_type * 20.0)  # sharp but differentiable

    linval_pos = varlin_pos * pkyval + (1.0 - varlin_pos) * medval
    linval_neg = varlin_neg * wdyval + (1.0 - varlin_neg) * medval
    linval = is_positive * linval_pos + (1.0 - is_positive) * linval_neg

    # Remove negative dips from extreme extrapolation
    # (smooth clamp instead of hard zeroing for differentiability)
    linval = jnp.maximum(linval, 0.0)

    # Interpolate template and reference continuum onto working wavelength grid
    linval_interp = jnp.interp(wavelength, linwav, linval)
    conval_interp = jnp.interp(wavelength, linwav, conval_raw)

    # Prevent division by zero in reference continuum
    conval_safe = jnp.maximum(conval_interp, 1e-30)

    # EW-scaling: preserves equivalent widths relative to actual continuum
    # Original: flux += |scalin| * linval * flux / conval
    # In f_nu: the dimensionless ratio linval/conval is identical to f_lambda
    scale = jnp.abs(emline_scale)
    line_fnu = scale * linval_interp * continuum_fnu / conval_safe

    return line_fnu


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


# ── Main QSOgen SED function ──────────────────────────────────────


def _lbol_to_m_i(log_lbol_lsun: float) -> float:
    """Convert log10(L_bol / Lsun) to approximate absolute i-band magnitude.

    Uses a rough bolometric correction to map L_bol to M_i for the
    Baldwin effect. The default M_i = -27 at log_lbol ~ 12.5 (Lsun).

    Parameters
    ----------
    log_lbol_lsun : float
        log10(L_bol / Lsun).

    Returns
    -------
    float
        Approximate M_i (AB magnitude).
    """
    # L_bol(Lsun) -> L_bol(erg/s) -> M_i via bolometric correction
    # M_i ~ -2.5 * log10(L_bol / L_ref) + M_i_ref
    # Calibrated so log_lbol_lsun=12.5 -> M_i ~ -27 (typical SDSS z~2 quasar)
    return -2.5 * (log_lbol_lsun - 12.5) + _BENORM


def _qsogen_components(
    wavelength: jnp.ndarray,
    *,
    agn_plslp1: float,
    agn_plslp2: float,
    agn_plbrk: float,
    agn_tbb: float,
    agn_bbnorm: float,
    agn_emline_scale: float,
    agn_ebv: float,
    agn_log_lbol: float,
    agn_bcnorm: float,
) -> dict[str, jnp.ndarray]:
    """Compute per-component ``L_nu`` for a QSOgen recipe.

    Returns a dict keyed by component (``"continuum"``, ``"hot_dust"``,
    ``"emission_lines"``, ``"balmer_continuum"``, ``"smc_factor"``) so each
    can be addressed independently by the
    :mod:`tengri.components.agn.blocks` runner.

    Each spectral component is the *L_nu contribution* before SMC reddening;
    ``smc_factor`` is the multiplicative attenuation factor in (0, 1].
    Reconstructing the monolithic :func:`compute_qsogen_sed` output is::

        sum_l_nu = continuum + hot_dust + emission_lines + balmer_continuum
        result = sum_l_nu * smc_factor

    All components share the joint cont+BB bolometric normalization so the
    sum bit-for-bit reproduces :func:`compute_qsogen_sed` (rtol < 1e-12).

    Notes
    -----
    JIT-compatible. Internal helper; not part of the public API. The block
    adapters in :mod:`tengri.components.agn.blocks.qsogen_blocks` call this
    once per block invocation; JIT folds the redundant trace.
    """
    continuum_unscaled = _broken_powerlaw_continuum(wavelength, agn_plslp1, agn_plslp2, agn_plbrk)
    hot_dust_unscaled = _hot_dust_blackbody(wavelength, continuum_unscaled, agn_tbb, agn_bbnorm)
    f_nu_cont = continuum_unscaled + hot_dust_unscaled

    nu = _wavelength_to_nu(wavelength)
    integral_nu = _bolometric_integral_nu(f_nu_cont, nu, floor=1e-30)

    l_bol_erg = 10.0**agn_log_lbol * _LSUN_ERG
    norm_factor = l_bol_erg / integral_nu

    continuum = continuum_unscaled * norm_factor
    hot_dust = hot_dust_unscaled * norm_factor

    # Emission lines anchor on the normalized continuum-only L_nu (matches
    # upstream order: lines added before BC, BC sees lines).
    m_i = _lbol_to_m_i(agn_log_lbol)
    emission_lines = _empirical_emission_lines(
        wavelength,
        continuum + hot_dust,
        agn_emline_scale,
        m_i,
    )

    # Balmer continuum sees cont + hot_dust + lines.
    balmer_continuum = _balmer_continuum(
        wavelength,
        continuum + hot_dust + emission_lines,
        agn_bcnorm,
    )

    # SMC reddening as a pure multiplicative factor.
    k_lam = smc_curve(wavelength)
    rv_smc = 2.93
    a_lam = agn_ebv * rv_smc * k_lam
    smc_factor = 10.0 ** (-0.4 * a_lam)

    return {
        "continuum": continuum,
        "hot_dust": hot_dust,
        "emission_lines": emission_lines,
        "balmer_continuum": balmer_continuum,
        "smc_factor": smc_factor,
    }


def compute_qsogen_sed(
    wavelength: jnp.ndarray,
    agn_plslp1: float = _DEFAULT_PLSLP1,
    agn_plslp2: float = _DEFAULT_PLSLP2,
    agn_plbrk: float = _DEFAULT_PLBRK,
    agn_tbb: float = _DEFAULT_TBB,
    agn_bbnorm: float = _DEFAULT_BBNORM,
    agn_emline_scale: float = _DEFAULT_EMLINE_SCALE,
    agn_ebv: float = _DEFAULT_EBV,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_bcnorm: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """QSOgen quasar SED (Temple, Hewett & Banerji 2021).

    Generates a rest-frame quasar SED from 912-100000 Angstrom by
    combining a broken power-law continuum, hot dust blackbody,
    empirical emission lines (with Baldwin effect), Balmer continuum,
    and optional SMC-like dust reddening.

    The output is L_nu in erg/s/Hz, normalized via ``agn_log_lbol``.

    Pipeline order (matching original qsogen):
    1. Continuum + hot dust (shape SED)
    2. Normalize to L_bol (bolometric integral of cont+BB only)
    3. Add emission lines as additive excess (not re-normalized)
    4. Add Balmer continuum as additive excess (not re-normalized)
    5. Dust reddening (attenuate full quasar SED)

    Parameters
    ----------
    wavelength : array, shape (n_wave,)
        Rest-frame wavelength [Angstrom].
    agn_plslp1 : float
        Blue power-law slope in f_nu (UV/blue side). Default -0.349.
        [dimensionless]
    agn_plslp2 : float
        Red power-law slope in f_nu (optical/red side). Default 0.593.
        [dimensionless]
    agn_plbrk : float
        Break wavelength [Angstrom]. Default 3880.
    agn_tbb : float
        Hot dust temperature [K]. Default 1240.
    agn_bbnorm : float
        Hot dust normalization (relative to continuum at 2 um).
        Default 3.96. [dimensionless]
    agn_emline_scale : float
        Emission line strength multiplier. Default 1.0 (negative values
        use EW-scaling, matching the original qsogen). [dimensionless]
    agn_ebv : float
        E(B-V) dust reddening [mag]. Default 0.0 (no reddening).
    agn_log_lbol : float
        log10(L_bol / Lsun). Bolometric luminosity. Defaults to the declared
        ``agn_log_lbol`` default.
        [log10(L_sun)]
    agn_lum_ratio : float
        Overall AGN fraction scaling. Default 1.0. [dimensionless]
    agn_bcnorm : float
        Balmer continuum normalization. Default 0.0 (disabled).
        Set to ~1.0 to enable Balmer continuum emission. [dimensionless]

    Returns
    -------
    array, shape (n_wave,)
        L_nu [erg s^-1 Hz^-1].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and smooth sigmoid
    transitions.

    The emission line template is loaded from disk at module import time
    to avoid file I/O inside JIT-traced functions.
    """
    components = _qsogen_components(
        wavelength,
        agn_plslp1=agn_plslp1,
        agn_plslp2=agn_plslp2,
        agn_plbrk=agn_plbrk,
        agn_tbb=agn_tbb,
        agn_bbnorm=agn_bbnorm,
        agn_emline_scale=agn_emline_scale,
        agn_ebv=agn_ebv,
        agn_log_lbol=agn_log_lbol,
        agn_bcnorm=agn_bcnorm,
    )
    l_nu = (
        components["continuum"]
        + components["hot_dust"]
        + components["emission_lines"]
        + components["balmer_continuum"]
    ) * components["smc_factor"]
    return l_nu * agn_lum_ratio


# ── Register in AGN_MODELS ────────────────────────────────────────


# Deprecated: qsogen is no longer registered in AGN_MODELS.
# Use composable AGN blocks instead: agn_disc_block="qsogen_sbpl_disc" + nlr/blr blocks.
# This function is retained for backward compatibility if imported directly.
def qsogen(
    wavelength: jnp.ndarray,
    agn_log_lbol: float = DEFAULT_AGN_LOG_LBOL,
    agn_lum_ratio: float = DEFAULT_AGN_LUM_RATIO,
    agn_plslp1: float = _DEFAULT_PLSLP1,
    agn_plslp2: float = _DEFAULT_PLSLP2,
    agn_plbrk: float = _DEFAULT_PLBRK,
    agn_tbb: float = _DEFAULT_TBB,
    agn_bbnorm: float = _DEFAULT_BBNORM,
    agn_emline_scale: float = _DEFAULT_EMLINE_SCALE,
    agn_ebv: float = _DEFAULT_EBV,
    agn_bcnorm: float = 0.0,
    **_kwargs,
) -> jnp.ndarray:
    """QSOgen quasar SED (Temple+2021) — registered model entry point.

    Thin wrapper around ``compute_qsogen_sed`` matching the AGN_MODELS registry
    signature: ``fn(wavelength, agn_log_lbol, **kwargs) -> L_nu``.

    Parameters
    ----------
    wavelength : array_like, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    agn_log_lbol : float, optional
        Total AGN bolometric luminosity [log10(L_sun)]. Defaults to the declared
        ``agn_log_lbol`` default.
    agn_lum_ratio : float, optional
        Fraction of bolometric luminosity emitted by this component.
        Default: 1.0. [dimensionless, 0–1]
    agn_plslp1 : float, optional
        Power-law slope for the UV/optical continuum. Default: -0.5.
        [dimensionless]
    agn_plslp2 : float, optional
        Power-law slope for the X-ray continuum. Default: -1.0.
        [dimensionless]
    agn_plbrk : float, optional
        Break wavelength between UV/optical and X-ray slopes [Angstrom].
        Default: 3000.0.
    agn_tbb : float, optional
        Hot dust blackbody temperature [K]. Default: 1000.0.
    agn_bbnorm : float, optional
        Hot dust blackbody normalization [dimensionless]. Default: 0.5.
    agn_emline_scale : float, optional
        Emission-line template normalization [dimensionless]. Default: 1.0.
    agn_ebv : float, optional
        Dust reddening (color excess) [dimensionless, E(B-V)]. Default: 0.0.
    agn_bcnorm : float, optional
        Balmer continuum normalization [dimensionless]. Default: 0.0.
    **_kwargs
        Additional keyword arguments (ignored, for compatibility with AGN
        model registry).

    Returns
    -------
    ndarray, shape (n_wave,)
        Spectral luminosity density L_ν [erg/s/Hz].

    Notes
    -----
    See ``compute_qsogen_sed`` for full parameter documentation and physics
    details.
    """
    return compute_qsogen_sed(
        wavelength,
        agn_plslp1=agn_plslp1,
        agn_plslp2=agn_plslp2,
        agn_plbrk=agn_plbrk,
        agn_tbb=agn_tbb,
        agn_bbnorm=agn_bbnorm,
        agn_emline_scale=agn_emline_scale,
        agn_ebv=agn_ebv,
        agn_log_lbol=agn_log_lbol,
        agn_lum_ratio=agn_lum_ratio,
        agn_bcnorm=agn_bcnorm,
    )
