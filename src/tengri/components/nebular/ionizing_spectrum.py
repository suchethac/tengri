# SPDX-License-Identifier: BSD-3-Clause
"""Ionizing spectrum fitting for the Cue neural emulator.

This module fits piecewise power-law parameterisations of the hydrogen-ionizing
portion (λ < 912 Å) of stellar population synthesis spectra. These 7-parameter
descriptions (4 power-law slopes + 3 flux ratios across ionization edges) serve
as inputs to the Cue neural network emulator for fast, differentiable nebular
emission predictions (Li et al. 2025).

Precomputation pipeline: fit all (metallicity, age) SSPs once, store in a
table, and interpolate at inference time for rapid gradient evaluation.
"""

import jax.numpy as jnp
import numpy as np

from tengri.components.nebular._constants import _C_AA, _H_PLANCK

# numpy < 2.0 compat: numpy 2.0 removed `trapz`; numpy >= 1.26 provides `trapezoid`.
# Guarded import avoids the eager `np.trapz` lookup that crashes on numpy >= 2.0.
try:
    from numpy import trapezoid as _np_trapz
except ImportError:  # numpy < 1.26
    from numpy import trapz as _np_trapz  # type: ignore[no-redef]

# Physical constants
from tengri.utils.physics_constants import L_SUN as _LSUN


def _fit_segment(
    seg_wave: np.ndarray,
    seg_flux: np.ndarray,
    norm: float,
) -> np.ndarray:
    """Fit power-law model to a single ionization-regime segment.

    **Internal helper** — fits L_ν ∝ λ^α to one wavelength segment using
    least-squares regression with a photon-count constraint to preserve Q_H.

    Parameters
    ----------
    seg_wave : array, shape (n_seg,)
        Segment wavelength grid [Å]
    seg_flux : array, shape (n_seg,)
        Normalized segment flux (already multiplied by norm factor for stability)
    norm : float
        Normalization factor applied to flux (used to denormalize log_A later)

    Returns
    -------
    coeff : array, shape (2,)
        [slope, log_norm_denormalized] — power-law parameters α and log10(A).
        If segment has no positive flux, returns [0.0, -inf].

    Notes
    -----
    Uses scipy L-BFGS-B optimization to minimize:
        0.5 * ||log_flux - pred||^2 + 0.5 * len(seg) * (log_Q - log_Q_pred)^2

    Constraint term ensures photon rate Q_H matches the input SED, preventing
    slope-fitting from drifting when flux amplitude is small or noisy.

    """
    from scipy.optimize import minimize

    pos = seg_flux > 0
    if not np.any(pos):
        return np.array([0.0, -np.inf])

    log_wave = np.log10(seg_wave)
    log_flux = np.log10(np.maximum(seg_flux, 1e-99))

    # Initial guess: linear fit in log-log
    init_slope = (log_flux[-1] - log_flux[0]) / max(log_wave[-1] - log_wave[0], 1e-10)
    init_norm = log_flux[-1] - init_slope * log_wave[-1]

    # Q_H for this segment — cast to float64 defensively. Per-segment
    # magnitudes are typically smaller than the full-LyC integration
    # below, but the same overflow path applies on float32 SSPs (see
    # fit_ionizing_spectrum Q_H block + issue #458).
    nu = (_C_AA / seg_wave).astype(np.float64)
    integrand = seg_flux.astype(np.float64) / (_H_PLANCK * nu)
    Q_seg = np.abs(_np_trapz(integrand[::-1], x=nu[::-1]))
    log_Q = np.log10(max(Q_seg, 1e-99))

    def objective(
        params,
        _log_wave=log_wave,
        _seg_wave=seg_wave,
        _log_flux=log_flux,
        _log_Q=log_Q,
    ):
        """Compute power-law residual and Q_H constraint for ionizing spectrum fitting."""
        pred = params[1] + params[0] * _log_wave
        log_Q_pred = (
            params[1]
            - np.log10(_H_PLANCK)
            + np.log10(
                np.abs((_seg_wave[-1] ** params[0] - _seg_wave[0] ** params[0]) / params[0])
            )
        )
        return (
            0.5 * np.sum((_log_flux - pred) ** 2)
            + 0.5 * len(_seg_wave) * (_log_Q - log_Q_pred) ** 2
        )

    res = minimize(
        objective,
        [init_slope, init_norm],
        method="L-BFGS-B",
        bounds=[(-40, 100), (-200, 100)],
    )
    coeff = np.array(res.x)
    coeff[1] -= np.log10(norm)
    return coeff


def _compute_segment_luminosities(
    coeff: np.ndarray,
    edges: np.ndarray,
) -> np.ndarray:
    """Compute integrated luminosity per segment from power-law parameters.

    **Internal helper** — integrates L_ν ∝ λ^α over wavelength for each
    of the 4 ionization segments, handling the special case α ≈ 1.

    Parameters
    ----------
    coeff : array, shape (4, 2)
        [slope, log_norm] pairs for each segment [Å, dimensionless]
    edges : array, shape (5,)
        Segment boundaries [Å]: [1, HeII, OII, HeI, HI_limit]

    Returns
    -------
    log_L : array, shape (4,)
        log10(integrated luminosity) for each segment [log10(erg/s)]

    Notes
    -----
    For α ≠ 1:
        L = A ∫_λ_lo^λ_hi λ^α dλ = A (λ_hi^(α+1) - λ_lo^(α+1)) / (α + 1)

    For α ≈ 1 (handled specially to avoid division by zero):
        L = A ∫_λ_lo^λ_hi λ dλ = A ln(λ_hi / λ_lo)

    """
    log_L = np.zeros(4)
    for i in range(4):
        lam_lo, lam_hi = edges[i], edges[i + 1]
        alpha = coeff[i, 0]
        log_A = coeff[i, 1]
        if abs(alpha - 1.0) > 1e-8:
            log_L[i] = (
                log_A
                + np.log10(_C_AA * _LSUN)
                + np.log10(np.abs((lam_hi ** (alpha - 1) - lam_lo ** (alpha - 1)) / (alpha - 1)))
            )
        else:
            log_L[i] = (
                log_A + np.log10(_C_AA * _LSUN) + np.log10(np.abs(np.log(lam_hi) - np.log(lam_lo)))
            )
    return log_L


# Ionization edges (Angstrom) — from cue/constants.py
HEII_EDGE = 1e8 / 438908.8789  # 227.84 A
OII_EDGE = 1e8 / 283270.9  # 353.07 A
HEI_EDGE = 1e8 / 198310.66637  # 504.26 A
HI_LIMIT = 911.76  # Lyman limit (physical: 911.7633 A)

# Segment boundaries: [1, HeII, OII, HeI, HI]
SEGMENT_EDGES = np.array([1.0, HEII_EDGE, OII_EDGE, HEI_EDGE, HI_LIMIT])

# Cue parameter ranges (for clipping)
_CLIP_RANGES = {
    "ionspec_index1": (1.0, 42.0),
    "ionspec_index2": (-0.3, 30.0),
    "ionspec_index3": (-1.0, 14.0),
    "ionspec_index4": (-1.7, 8.0),
    "ionspec_logLratio1": (-1.0, 10.1),
    "ionspec_logLratio2": (-0.5, 1.9),
    "ionspec_logLratio3": (-0.4, 2.2),
}


def fit_ionizing_spectrum(
    wave: np.ndarray,
    flux: np.ndarray,
    edges: np.ndarray = SEGMENT_EDGES,
) -> dict:
    r"""Fit 4-segment piecewise power law to ionizing SED.

    Given an SSP or composite stellar population spectrum, fits independent
    power-law models (L_ν ∝ λ^α) to each of 4 ionization-regime segments
    (HeII, OII, HeI, HI Lyman limit). Returns 7 parameters suitable for input
    to the Cue neural emulator: 4 power-law slopes + 3 log-flux ratios.

    **Not JAX-compatible** — uses numpy and scipy.optimize; intended for
    one-time precomputation of SSP grids before inference.

    Parameters
    ----------
    wave : array, shape (n_wave,)
        Wavelength grid in Å (must cover λ < 912 Å). [Å]
    flux : array, shape (n_wave,)
        Spectral luminosity density [erg/s/Hz/Msun] or [erg/s/Hz];
        units cancel in power-law fit.
    edges : array, shape (5,), optional
        Segment boundaries in Å: [1, HeII, OII, HeI, HI_limit].
        Default: SEGMENT_EDGES (1, 227.84, 353.07, 504.26, 911.76 Å).

    Returns
    -------
    dict
        Dictionary with keys:

        - **ionspec_index1...4** (float): Power-law slopes α for each segment
        - **ionspec_logLratio1...3** (float): log10(L_{k+1}/L_k) integrated fluxes
        - **gas_logqion** (float): log10(Q_H) total ionizing photon rate [log10(photons/s)]
        - **powerlaw_params** (array, shape (4, 2)): [slope, log_norm] per segment

    Notes
    -----
    **Fitting strategy** (Li et al. 2025):
        Each of the 4 segments is independently fit to a power law:

        .. math::

            L_\nu(\lambda) = A_k \, \lambda^{\alpha_k}

        in log-log space via least-squares regression. To preserve ionizing photon
        content, a penalty term |Q_{H,\mathrm{fit}} - Q_{H,\mathrm{data}}| is added
        to the objective (scipy L-BFGS-B minimization). This ensures that Q_H
        (computed by integrating L_ν / hν) matches the true photon rate, not just
        the spectral shape.

    **Ionization regimes**:
        - **Segment 1** [1, 227.84 Å]: HeII ionization (E > 54.4 eV)
        - **Segment 2** [227.84, 353.07 Å]: OII→HeII (40.8–54.4 eV)
        - **Segment 3** [353.07, 504.26 Å]: HeI→OII (24.6–40.8 eV)
        - **Segment 4** [504.26, 911.76 Å]: HI→HeI (13.6–24.6 eV, Lyman continuum)

        Edges correspond to ionization thresholds of heavy elements; Cue encodes
        metallicity implicitly via their position in ionization parameter space.

    **Integrated flux ratios**:
        For each segment k, compute integrated luminosity:

        .. math::

            L_k = \int_{\lambda_k}^{\lambda_{k+1}} L_\nu(\lambda) \, \frac{\mathrm{d}\nu}{c}

        Then define:

        .. math::

            \log L_{\mathrm{ratio}, k} = \log_{10}\left(\frac{L_{k+1}}{L_k}\right)

        These 3 ratios encode the relative ionizing flux in adjacent segments,
        independent of absolute normalisation.

    **Error handling**:
        Returns sensible defaults (zero slope, −∞ norm) for segments with
        no/negligible ionizing flux (e.g., segment 1 in solar-metallicity old SSPs).

    **Clipping**:
        Output values are clipped to physically motivated ranges stored in
        _CLIP_RANGES to prevent Cue emulator extrapolation failures.

    References
    ----------
    .. [1] M. Li et al., "The Cue Nebular Emulator: Fast, Interpretable
       Predictions of Emission-Line Strengths from Stellar Populations,"
       ApJ, 986, 9 (2025). arXiv:2405.04598.
       https://doi.org/10.3847/1538-4357/ad7fe3

    """
    wave = np.asarray(wave)
    flux = np.asarray(flux)

    # Find bin edges in wavelength array
    ind_bin = np.array([max(np.where(wave <= e)[0]) for e in edges[1:]]) + 1
    ind_bin = np.insert(ind_bin, 0, 0)

    coeff = np.zeros((4, 2))

    # Normalize spectrum for numerical stability
    ref_flux = np.median(flux[ind_bin[-1] - 1 : ind_bin[-1] + 1])
    norm = 1e-18 / max(ref_flux, 1e-99)
    normalized = np.clip(flux * norm, 1e-70 * norm, np.inf)

    # Fit each segment independently
    for i in range(4):
        seg_wave = wave[ind_bin[i] : ind_bin[i + 1]]
        seg_flux = normalized[ind_bin[i] : ind_bin[i + 1]]
        coeff[i] = _fit_segment(seg_wave, seg_flux, norm)

    # Compute integrated luminosities per segment
    log_L = _compute_segment_luminosities(coeff, edges)
    logLratios = np.diff(log_L)

    # Total Q_H — integrate photon rate over frequency.
    # wave is increasing → nu_all is decreasing.  Both integrand and
    # x must share the same element ordering for np.trapz.
    #
    # Cast to float64 for the integration. SSPs may ship in float32 (e.g.
    # BC03-from-CIGALE) to save disk; ``(flux * L_SUN) / (h * nu)`` then
    # produces intermediates ~ 1e30 and ``trapezoid`` over the ~ 1e16 Hz
    # bandwidth integrates to ~ 1e46 — well past float32's 3.4e38 max,
    # so the running sum overflows to ``inf`` from a few terms in. Result:
    # every (Z, age) bin in the cached table collapses to ``log10(inf) =
    # inf``, downstream treats the SSP as wNE / dead, and Cue silently
    # emits ~zero nebular emission (issue #458). The slope fits above
    # are float32-safe because ``normalized = flux * 1e-18 / ref_flux``
    # rescales each segment before fitting.
    ionizing_mask = wave <= HI_LIMIT
    flux_iz = flux[ionizing_mask].astype(np.float64)
    nu_all = (_C_AA / wave[ionizing_mask]).astype(np.float64)
    Q_total = np.abs(
        _np_trapz(
            (flux_iz * _LSUN) / (_H_PLANCK * nu_all),
            x=nu_all,
        )
    )
    log_qion = np.log10(max(Q_total, 1e-99))

    return {
        "ionspec_index1": np.clip(coeff[0, 0], *_CLIP_RANGES["ionspec_index1"]),
        "ionspec_index2": np.clip(coeff[1, 0], *_CLIP_RANGES["ionspec_index2"]),
        "ionspec_index3": np.clip(coeff[2, 0], *_CLIP_RANGES["ionspec_index3"]),
        "ionspec_index4": np.clip(coeff[3, 0], *_CLIP_RANGES["ionspec_index4"]),
        "ionspec_logLratio1": np.clip(logLratios[0], *_CLIP_RANGES["ionspec_logLratio1"]),
        "ionspec_logLratio2": np.clip(logLratios[1], *_CLIP_RANGES["ionspec_logLratio2"]),
        "ionspec_logLratio3": np.clip(logLratios[2], *_CLIP_RANGES["ionspec_logLratio3"]),
        "gas_logqion": log_qion,
        "powerlaw_params": coeff,
    }


#: Max log10(age/yr) at which Cue's downstream Q_H weighting can still receive
#: a non-zero contribution. Bins older than this have ``weighted_qh`` zeroed in
#: :meth:`CueBackend._compute_weighted_cue_params`, so fitting them here is
#: pure waste. Kept in sync with ``cue._MAX_NEB_LOG_AGE``.
_PRECOMPUTE_MAX_LOG_AGE_YR: float = 8.0  # 100 Myr


def precompute_ionizing_params_table(
    ssp_wave: np.ndarray,
    ssp_flux: np.ndarray,
    ssp_lgmet: np.ndarray,
    ssp_log_age_yr: np.ndarray | None = None,
) -> dict:
    """Precompute Cue ionizing parameters for a full SSP grid.

    Batch-fits piecewise power laws to (metallicity, age) SSP spectra younger
    than 100 Myr (older bins contribute nothing to Cue's weighted Q_H so they
    are skipped — ``logqion_table[im, ia] = -99`` for those). Called once at
    model initialization; results are stored and interpolated at runtime.

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        Wavelength grid in Å. [Å]
    ssp_flux : array, shape (n_met, n_age, n_wave)
        SSP spectra on (metallicity, age) grid. [erg/s/Hz/Msun]
    ssp_lgmet : array, shape (n_met,)
        Metallicity grid in log10(Z). [log10(Z)]
    ssp_log_age_yr : array, shape (n_age,), optional
        log10(age/yr) for each age bin. When provided, bins older than
        :data:`_PRECOMPUTE_MAX_LOG_AGE_YR` (100 Myr) are skipped without
        invoking scipy — a ~140× speedup for unusually fine age grids
        (BC03-from-CIGALE has 13700 ages of which only ~10 are young).
        When ``None`` (legacy callers), every bin is attempted; the per-bin
        ``np.max(flux_iz) <= 0`` early-exit still skips dead bins.

    Returns
    -------
    dict
        Dictionary with keys:

        - **ionspec_table** (array, shape (n_met, n_age, 7)): Ionizing parameters
          [ionspec_index1..4, ionspec_logLratio1..3] per SSP
        - **logqion_table** (array, shape (n_met, n_age)): Q_H ionizing photon rates
          [log10(photons/s)]
        - **n_met, n_age** (int): Grid dimensions

    Notes
    -----
    **One-time cost**: This function is not JAX-compatible (uses scipy). Call
    once per Cue model initialization and store results (e.g., in h5 file).
    Precomputation is O(n_met × n_age_young × n_segments) where n_age_young is
    the count of bins ≤ 100 Myr; typically < 1 second for modern SSP grids.

    **Metadata**: Unfittable SSPs (old ages with zero ionizing flux, corrupted
    data) are left with all parameters = 0 (or −99 for log-space). Downstream
    code handles these gracefully via clipping and default fallback values.

    """
    n_met, n_age, _ = ssp_flux.shape
    ionspec_table = np.zeros((n_met, n_age, 7))
    logqion_table = np.full((n_met, n_age), -99.0)

    wave_np = np.asarray(ssp_wave)

    # Optional age-cutoff: skip the scipy fit on bins older than the
    # downstream Q_H weighting threshold. Saves ~99 % of fits on fine-age
    # grids (e.g. BC03-from-CIGALE: 13700 ages → ~10 young bins).
    if ssp_log_age_yr is not None:
        young_age_mask = np.asarray(ssp_log_age_yr) <= _PRECOMPUTE_MAX_LOG_AGE_YR
    else:
        young_age_mask = np.ones(n_age, dtype=bool)

    for im in range(n_met):
        for ia in range(n_age):
            if not young_age_mask[ia]:
                continue
            flux_np = np.asarray(ssp_flux[im, ia, :])

            # Skip if no ionizing flux
            ionizing_mask = wave_np <= HI_LIMIT
            if np.max(flux_np[ionizing_mask]) <= 0:
                continue

            try:
                result = fit_ionizing_spectrum(wave_np, flux_np)
                ionspec_table[im, ia, 0] = result["ionspec_index1"]
                ionspec_table[im, ia, 1] = result["ionspec_index2"]
                ionspec_table[im, ia, 2] = result["ionspec_index3"]
                ionspec_table[im, ia, 3] = result["ionspec_index4"]
                ionspec_table[im, ia, 4] = result["ionspec_logLratio1"]
                ionspec_table[im, ia, 5] = result["ionspec_logLratio2"]
                ionspec_table[im, ia, 6] = result["ionspec_logLratio3"]
                logqion_table[im, ia] = result["gas_logqion"]
            except (ValueError, IndexError, RuntimeError):
                # ValueError: invalid input data (NaN, zero flux, wrong wavelength range)
                # IndexError: wavelength array doesn't cover ionizing regime (<912 A)
                # RuntimeError: scipy optimization failed to converge
                continue

    return {
        "ionspec_table": ionspec_table,
        "logqion_table": logqion_table,
        "n_met": n_met,
        "n_age": n_age,
    }


def interpolate_ionizing_params(
    ionspec_table: jnp.ndarray,
    logqion_table: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    ssp_log_age_yr: jnp.ndarray,
    log_z: float,
    log_age_yr: float,
) -> tuple[jnp.ndarray, float]:
    """Bilinearly interpolate ionizing spectrum parameters at target (Z, age).

    Looks up precomputed ionizing parameters on the (metallicity, age) SSP grid
    and returns bilinearly interpolated values at the target point.

    Parameters
    ----------
    ionspec_table : array, shape (n_met, n_age, 7)
        Precomputed ionising spectrum parameters (ionspec_index1..4, logLratio1..3).
    logqion_table : array, shape (n_met, n_age)
        Ionizing photon rates Q_H. [log10(photons/s)]
    ssp_lgmet : array, shape (n_met,)
        SSP metallicity grid. [log10(Z)]
    ssp_log_age_yr : array, shape (n_age,)
        SSP age grid. [log10(yr)]
    log_z : float
        Target metallicity. [log10(Z)]
    log_age_yr : float
        Target age. [log10(yr)]

    Returns
    -------
    ionspec_7 : array, shape (7,)
        Ionizing spectrum parameters: [index1, index2, index3, index4,
        logLratio1, logLratio2, logLratio3]
    logqion : float
        Interpolated Q_H. [log10(photons/s)]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives with
    searchsorted and linear interpolation.

    **Clipping**: Target (log_z, log_age_yr) are clipped to grid bounds
    before interpolation to prevent extrapolation artifacts.

    **Bilinear interpolation**: Uses 2×2 neighbourhood of grid points,
    with fractional weights (fz, fa) computed from target position within
    the bracketing cell.

    """
    # Bilinear interpolation
    log_z_c = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
    log_age_c = jnp.clip(log_age_yr, ssp_log_age_yr[0], ssp_log_age_yr[-1])

    iz = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
    ia = jnp.clip(jnp.searchsorted(ssp_log_age_yr, log_age_c) - 1, 0, len(ssp_log_age_yr) - 2)

    fz = (log_z_c - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])
    fa = (log_age_c - ssp_log_age_yr[ia]) / (ssp_log_age_yr[ia + 1] - ssp_log_age_yr[ia])

    # Bilinear for 7-vector
    t00 = ionspec_table[iz, ia]
    t01 = ionspec_table[iz, ia + 1]
    t10 = ionspec_table[iz + 1, ia]
    t11 = ionspec_table[iz + 1, ia + 1]

    ionspec = (1 - fz) * (1 - fa) * t00 + (1 - fz) * fa * t01 + fz * (1 - fa) * t10 + fz * fa * t11

    # Bilinear for scalar Q_H
    q00 = logqion_table[iz, ia]
    q01 = logqion_table[iz, ia + 1]
    q10 = logqion_table[iz + 1, ia]
    q11 = logqion_table[iz + 1, ia + 1]

    logqion = (1 - fz) * (1 - fa) * q00 + (1 - fz) * fa * q01 + fz * (1 - fa) * q10 + fz * fa * q11

    return ionspec, logqion
