"""Compute ionizing spectrum parameters for the Cue emulator.

Given an SSP spectrum, fits a 4-segment piecewise power law to the
ionizing portion (lambda < 912 A) and returns the 7 Cue parameters:
4 slopes + 3 log flux ratios.

The 4 segments are bounded by ionization edges:
- Segment 1: [1, 227.84] A  (He II edge)
- Segment 2: [227.84, 353.07] A  (O II edge)
- Segment 3: [353.07, 504.26] A  (He I edge)
- Segment 4: [504.26, 911.6] A  (H I Lyman limit)

Each segment is fit with: log10(F_nu) = alpha * log10(lambda) + log10(A)

The 3 flux ratios are: logLratio_k = log10(L_k+1 / L_k) where L_k is
the integrated luminosity in segment k.

Based on Li et al. (2025, ApJ 986, 9), arXiv:2405.04598.

References
----------
- Li et al. 2025, ApJ, 986, 9
- fit_4loglinear_ionparam() in cue/utils.py
"""

import jax.numpy as jnp
import numpy as np

# numpy < 2.0 compat
_np_trapz = getattr(np, "trapezoid", np.trapz)

# Physical constants
_C_AA = 2.9979e18     # c in Angstrom/s
_H_PLANCK = 6.626e-27 # erg s
_LSUN = 3.828e33      # erg/s

# Ionization edges (Angstrom) — from cue/constants.py
HEII_EDGE = 1e8 / 438908.8789   # 227.84 A
OII_EDGE = 1e8 / 283270.9       # 353.07 A
HEI_EDGE = 1e8 / 198310.66637   # 504.26 A
HI_LIMIT = 911.6                 # Lyman limit

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
    """Fit piecewise power law to ionizing spectrum (numpy, for precomputation).

    This is a numpy function (not JAX) intended for one-time precomputation
    of ionizing parameters from SSP templates. It uses scipy.optimize for
    the fit, matching the original Cue implementation.

    Parameters
    ----------
    wave : array, shape (n_wave,)
        Wavelength in Angstrom (must include lambda < 912 A).
    flux : array, shape (n_wave,)
        Flux density in Lsun/Hz.
    edges : array
        Segment boundaries [1, HeII, OII, HeI, 911.6].

    Returns
    -------
    dict with keys:
        ionspec_index1..4 : float — power-law slopes
        ionspec_logLratio1..3 : float — log flux ratios
        gas_logqion : float — log10(Q_H) total ionizing photon rate
        powerlaw_params : array (4, 2) — [slope, log_norm] per segment
    """
    from scipy.optimize import minimize

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

    for i in range(4):
        seg_wave = wave[ind_bin[i] : ind_bin[i + 1]]
        seg_flux = normalized[ind_bin[i] : ind_bin[i + 1]]

        pos = seg_flux > 0
        if not np.any(pos):
            coeff[i] = [0.0, -np.inf]
            continue

        log_wave = np.log10(seg_wave)
        log_flux = np.log10(np.maximum(seg_flux, 1e-99))

        # Initial guess: linear fit in log-log
        init_slope = (log_flux[-1] - log_flux[0]) / max(log_wave[-1] - log_wave[0], 1e-10)
        init_norm = log_flux[-1] - init_slope * log_wave[-1]

        # Q_H for this segment
        nu = _C_AA / seg_wave
        integrand = seg_flux / (_H_PLANCK * nu)
        Q_seg = np.abs(_np_trapz(integrand[::-1], x=nu[::-1]))
        log_Q = np.log10(max(Q_seg, 1e-99))

        def objective(params):
            pred = params[1] + params[0] * log_wave
            log_Q_pred = params[1] - np.log10(_H_PLANCK) + np.log10(
                np.abs((seg_wave[-1] ** params[0] - seg_wave[0] ** params[0]) / params[0])
            )
            return (
                0.5 * np.sum((log_flux - pred) ** 2)
                + 0.5 * len(seg_wave) * (log_Q - log_Q_pred) ** 2
            )

        res = minimize(
            objective,
            [init_slope, init_norm],
            method="L-BFGS-B",
            bounds=[(-40, 100), (-200, 100)],
        )
        coeff[i] = res.x
        coeff[i, 1] -= np.log10(norm)

    # Compute integrated luminosities per segment
    log_L = np.zeros(4)
    for i in range(4):
        lam_lo, lam_hi = edges[i], edges[i + 1]
        alpha = coeff[i, 0]
        log_A = coeff[i, 1]
        if abs(alpha - 1.0) > 1e-8:
            log_L[i] = log_A + np.log10(_C_AA * _LSUN) + np.log10(
                np.abs((lam_hi ** (alpha - 1) - lam_lo ** (alpha - 1)) / (alpha - 1))
            )
        else:
            log_L[i] = log_A + np.log10(_C_AA * _LSUN) + np.log10(
                np.abs(np.log(lam_hi) - np.log(lam_lo))
            )

    logLratios = np.diff(log_L)

    # Total Q_H — integrate photon rate over frequency.
    # wave is increasing → nu_all is decreasing.  Both integrand and
    # x must share the same element ordering for np.trapz.
    ionizing_mask = wave <= HI_LIMIT
    nu_all = _C_AA / wave[ionizing_mask]
    Q_total = np.abs(_np_trapz(
        (flux[ionizing_mask] * _LSUN) / (_H_PLANCK * nu_all),
        x=nu_all,
    ))
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


def precompute_ionizing_params_table(
    ssp_wave: np.ndarray,
    ssp_flux: np.ndarray,
    ssp_lgmet: np.ndarray,
) -> dict:
    """Precompute ionizing spectrum parameters for an SSP grid.

    Fits piecewise power laws to each (metallicity, age) SSP spectrum.
    Only fits young ages (< 100 Myr) since older SSPs have negligible
    ionizing flux.

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        SSP wavelength grid (Angstrom).
    ssp_flux : array, shape (n_met, n_age, n_wave)
        SSP spectra in Lsun/Hz/Msun.
    ssp_lgmet : array, shape (n_met,)
        SSP log metallicity grid.

    Returns
    -------
    dict with:
        ionspec_table : array (n_met, n_age, 7) — the 7 ionizing params
        logqion_table : array (n_met, n_age) — log10(Q_H)
        n_met, n_age : int
    """
    n_met, n_age, _ = ssp_flux.shape
    ionspec_table = np.zeros((n_met, n_age, 7))
    logqion_table = np.full((n_met, n_age), -99.0)

    wave_np = np.asarray(ssp_wave)

    for im in range(n_met):
        for ia in range(n_age):
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
            except Exception:
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
    """Interpolate precomputed ionizing params at (Z, age).

    Parameters
    ----------
    ionspec_table : array (n_met, n_age, 7)
    logqion_table : array (n_met, n_age)
    ssp_lgmet : array (n_met,)
    ssp_log_age_yr : array (n_age,)
    log_z : float — target metallicity
    log_age_yr : float — target log age

    Returns
    -------
    ionspec_7 : array (7,) — the 7 ionizing spectrum parameters
    logqion : float — log10(Q_H)
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

    ionspec = (
        (1 - fz) * (1 - fa) * t00
        + (1 - fz) * fa * t01
        + fz * (1 - fa) * t10
        + fz * fa * t11
    )

    # Bilinear for scalar Q_H
    q00 = logqion_table[iz, ia]
    q01 = logqion_table[iz, ia + 1]
    q10 = logqion_table[iz + 1, ia]
    q11 = logqion_table[iz + 1, ia + 1]

    logqion = (1 - fz) * (1 - fa) * q00 + (1 - fz) * fa * q01 + fz * (1 - fa) * q10 + fz * fa * q11

    return ionspec, logqion
