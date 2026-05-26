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

# Module-level memoization for ``precompute_ionizing_params_table``.
#
# Same SSP → identical ionizing-parameter table. The scipy curve-fit loop
# costs ~6 s on a modern MILES grid (15×93 SSPs), and re-runs on every
# ``SEDModel.build(neb={'type':'cue', ...})``. Cache keyed on a stable
# fingerprint of the SSP wavelength + metallicity grids (small, fast to
# hash); flux identity is established by matching grids since SSP files
# pair a unique flux cube with a unique (wave, lgmet) pair.
#
# The cache also persists to disk under ``<jax_cache_dir>/ionspec_tables/``,
# keyed on a SHA-256 of the fingerprint. This matters because the scipy
# curve-fits underlying the table aren't bit-stable across processes
# (BLAS threading + LAPACK perturbations → coefficient differences in
# the last few ULPs). Those coefficients then bake into the JAX trace as
# constants, so cross-process processes produce slightly different HLO
# modules → the persistent XLA cache misses on every Cue compile. Loading
# a bit-identical table from disk on every process keeps the HLO stable
# and lets the JAX persistent cache actually hit on cold-start runs.

import hashlib as _hashlib
import os as _os
from pathlib import Path as _Path

_IONSPEC_TABLE_CACHE: dict[tuple, dict] = {}


def _ssp_fingerprint(ssp_wave: np.ndarray, ssp_flux: np.ndarray, ssp_lgmet: np.ndarray) -> tuple:
    """Cheap, stable fingerprint for an SSP grid."""
    return (
        tuple(ssp_flux.shape),
        str(ssp_flux.dtype),
        bytes(np.asarray(ssp_wave).tobytes()),
        bytes(np.asarray(ssp_lgmet).tobytes()),
    )


def _fingerprint_hash(key: tuple) -> str:
    """SHA-256 of the fingerprint, hex-encoded, for use as a disk filename."""
    h = _hashlib.sha256()
    h.update(repr(key[:2]).encode())  # shape + dtype
    h.update(key[2])  # raw wave bytes
    h.update(key[3])  # raw lgmet bytes
    return h.hexdigest()


def _ionspec_disk_cache_dir() -> _Path:
    """Resolve the on-disk ionspec table cache directory.

    Co-locates with the JAX persistent cache so wiping one wipes the other.
    """
    env_dir = _os.environ.get("TENGRI_JAX_CACHE_DIR", "").strip()
    if env_dir:
        base = _Path(env_dir).expanduser()
    else:
        xdg = _os.environ.get("XDG_CACHE_HOME", "").strip()
        base = (_Path(xdg).expanduser() if xdg else _Path.home() / ".cache") / "tengri_jax_cache"
    out = base / "ionspec_tables"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _load_ionspec_disk(key: tuple) -> dict | None:
    """Load a cached ionspec table from disk if it exists, else return None."""
    try:
        path = _ionspec_disk_cache_dir() / f"{_fingerprint_hash(key)}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            return {
                "ionspec_table": data["ionspec_table"],
                "logqion_table": data["logqion_table"],
                "n_met": int(data["n_met"]),
                "n_age": int(data["n_age"]),
            }
    except (OSError, ValueError, KeyError):
        # Corrupted file / permission issue → treat as miss
        return None


def _store_ionspec_disk(key: tuple, result: dict) -> None:
    """Persist an ionspec table to disk. Failures are silently swallowed."""
    try:
        path = _ionspec_disk_cache_dir() / f"{_fingerprint_hash(key)}.npz"
        np.savez(
            path,
            ionspec_table=result["ionspec_table"],
            logqion_table=result["logqion_table"],
            n_met=np.asarray(result["n_met"]),
            n_age=np.asarray(result["n_age"]),
        )
    except OSError:
        pass


class _single_thread_blas:
    """Context manager forcing single-threaded BLAS / OpenMP.

    The piecewise-power-law fits in :func:`precompute_ionizing_params_table`
    use scipy / numpy linear algebra under the hood. Multi-threaded BLAS
    introduces non-determinism (thread scheduling perturbs the last ULPs of
    the LSQ solution), and those bits then bake into the JAX trace as
    constants — so different processes produce different HLO modules. Pinning
    BLAS / OpenMP to one thread for the duration of the fit loop gives
    bit-identical coefficients across processes and is also slightly faster
    on small (~10×80) SSP grids where the per-call overhead dominates.

    Falls back to a no-op if ``threadpoolctl`` is unavailable.
    """

    def __enter__(self):
        try:
            from threadpoolctl import threadpool_limits

            self._ctx = threadpool_limits(limits=1)
        except ImportError:
            self._ctx = None
        return self

    def __exit__(self, *exc):
        if self._ctx is not None:
            self._ctx.__exit__(*exc)
        return False


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

    # Q_H for this segment
    nu = _C_AA / seg_wave
    integrand = seg_flux / (_H_PLANCK * nu)
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
    ionizing_mask = wave <= HI_LIMIT
    nu_all = _C_AA / wave[ionizing_mask]
    Q_total = np.abs(
        _np_trapz(
            (flux[ionizing_mask] * _LSUN) / (_H_PLANCK * nu_all),
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


def precompute_ionizing_params_table(
    ssp_wave: np.ndarray,
    ssp_flux: np.ndarray,
    ssp_lgmet: np.ndarray,
) -> dict:
    """Precompute Cue ionizing parameters for a full SSP grid.

    Batch-fits piecewise power laws to all (metallicity, age) SSP spectra.
    Silently skips SSPs with negligible ionizing flux (age > ~100 Myr).
    Called once at model initialization; results are stored and interpolated
    at runtime.

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        Wavelength grid in Å. [Å]
    ssp_flux : array, shape (n_met, n_age, n_wave)
        SSP spectra on (metallicity, age) grid. [erg/s/Hz/Msun]
    ssp_lgmet : array, shape (n_met,)
        Metallicity grid in log10(Z). [log10(Z)]

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
    Precomputation is O(n_met × n_age × n_segments), typically < 1 second
    for modern SSP grids (n_met ~ 50, n_age ~ 300).

    **Metadata**: Unfittable SSPs (old ages with zero ionizing flux, corrupted
    data) are left with all parameters = 0 (or −99 for log-space). Downstream
    code handles these gracefully via clipping and default fallback values.

    """
    # Memoize: same SSP grid → identical result. Two-tier cache:
    # 1. In-process dict for fast repeat builds within a single process
    #    (introduced in #418).
    # 2. On-disk ``.npz`` keyed on a SHA-256 of the SSP fingerprint, so
    #    cold processes load a bit-identical table instead of re-fitting.
    #    Without this, scipy's curve-fits perturb the coefficients by a
    #    few ULPs each run, and those bits bake into the JAX trace as
    #    constants → the JAX persistent compilation cache misses on every
    #    fresh process. The single_thread_blas wrapper below removes the
    #    remaining BLAS-threading non-determinism inside the fits.
    key = _ssp_fingerprint(ssp_wave, ssp_flux, ssp_lgmet)
    cached = _IONSPEC_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    disk_cached = _load_ionspec_disk(key)
    if disk_cached is not None:
        _IONSPEC_TABLE_CACHE[key] = disk_cached
        return disk_cached

    n_met, n_age, _ = ssp_flux.shape
    ionspec_table = np.zeros((n_met, n_age, 7))
    logqion_table = np.full((n_met, n_age), -99.0)

    wave_np = np.asarray(ssp_wave)

    with _single_thread_blas():
        for im in range(n_met):
            for ia in range(n_age):
                flux_np = np.asarray(ssp_flux[im, ia, :])

                # Skip if no ionizing flux
                ionizing_mask = wave_np <= HI_LIMIT
                if np.max(flux_np[ionizing_mask]) <= 0:
                    continue

                try:
                    fit = fit_ionizing_spectrum(wave_np, flux_np)
                    ionspec_table[im, ia, 0] = fit["ionspec_index1"]
                    ionspec_table[im, ia, 1] = fit["ionspec_index2"]
                    ionspec_table[im, ia, 2] = fit["ionspec_index3"]
                    ionspec_table[im, ia, 3] = fit["ionspec_index4"]
                    ionspec_table[im, ia, 4] = fit["ionspec_logLratio1"]
                    ionspec_table[im, ia, 5] = fit["ionspec_logLratio2"]
                    ionspec_table[im, ia, 6] = fit["ionspec_logLratio3"]
                    logqion_table[im, ia] = fit["gas_logqion"]
                except (ValueError, IndexError, RuntimeError):
                    # ValueError: invalid input data (NaN, zero flux, wrong wavelength range)
                    # IndexError: wavelength array doesn't cover ionizing regime (<912 A)
                    # RuntimeError: scipy optimization failed to converge
                    continue

    result = {
        "ionspec_table": ionspec_table,
        "logqion_table": logqion_table,
        "n_met": n_met,
        "n_age": n_age,
    }
    _IONSPEC_TABLE_CACHE[key] = result
    _store_ionspec_disk(key, result)
    return result


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
