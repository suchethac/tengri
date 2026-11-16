# SPDX-License-Identifier: BSD-3-Clause
"""Ionizing spectrum fitting for the Cue neural emulator.

This module fits piecewise power-law parameterizations of the hydrogen-ionizing
portion (λ < 912 Å) of stellar population synthesis spectra. These 7-parameter
descriptions (4 power-law slopes + 3 flux ratios across ionization edges) serve
as inputs to the Cue neural network emulator for fast, differentiable nebular
emission predictions (Li et al. 2025).

Precomputation pipeline: fit all (metallicity, age) SSPs once, store in a
table, and interpolate at inference time for rapid gradient evaluation.
"""

# Module-level memoization for ``precompute_ionizing_params_table``.
#
# Same SSP → identical ionizing-parameter table. The per-SSP fit loop would
# otherwise re-run on every ``SEDModel.build(neb={'type':'cue', ...})``. Cache keyed on a stable
# fingerprint of the SSP wavelength + metallicity grids (small, fast to
# hash); flux identity is established by matching grids since SSP files
# pair a unique flux cube with a unique (wave, lgmet) pair.
#
# The cache also persists to disk under ``<jax_cache_dir>/ionspec_tables/``,
# keyed on a SHA-256 of the fingerprint. A table whose coefficients
# differed in the last few ULPs between processes would bake into the JAX
# trace as constants, so cross-process processes would produce slightly
# different HLO modules and the persistent XLA cache would miss on every
# Cue compile. Loading
# a bit-identical table from disk on every process keeps the HLO stable
# and lets the JAX persistent cache actually hit on cold-start runs.
import hashlib as _hashlib
import os as _os
from collections.abc import Mapping
from pathlib import Path as _Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np

from tengri.components.lyc import LYMAN_LIMIT_AA, edge_trapezoid
from tengri.components.nebular._constants import _C_AA, _H_PLANCK
from tengri.config.exceptions import ConfigError
from tengri.utils.host_array import host_array
from tengri.utils.physics_constants import L_SUN as _LSUN

#: Version constant for ionizing-spectrum precomputation table.
#: Bump when the curve-fit algorithm, the table layout, or field set changes.
#: One version constant per cache (disk + in-memory) to ensure a stale entry
#: cannot be reused after an algorithm change.
#: 2: ``gas_logqion`` integrates via the shared
#: :func:`tengri.components.lyc.edge_trapezoid` step model (#537 partial-bin
#: correction), not a hard ``wave <= HI_LIMIT`` mask.
#: 3: ``gas_logqion`` integrates in the log domain (#2849), and the computation
#: dtype is part of the key, so entries written under version 2 are not reused.
#: 4: bins older than 100 Myr carry a Q_H and a shape (every age is counted by
#: default); the age axis, the age cutoff and the old-bin thinning limit are part
#: of the key, so the young-only tables written under version 3 are never served.
#: 5: segments are solved to their exact stationary point by
#: :func:`tengri.components.nebular.ionizing_fit.fit_ionizing_shape`, where SLSQP
#: stopped up to 0.06 in slope short of it inside the trained box.
_IONSPEC_CACHE_VERSION = 5

_IONSPEC_TABLE_CACHE: dict[tuple, Mapping] = {}

#: :func:`~tengri.components.lyc.edge_trapezoid` traced once per integrand shape. The
#: table builder calls it from NumPy code, once per fitted bin and once per chunk of old
#: bins; dispatched eagerly, each of its ``jnp`` operations compiled as its own XLA
#: program on every cold table build (34 on the FSPS grid, none above the persistent
#: cache's 0.05 s floor, #2914). Traced, each shape is one program, and the persistent
#: cache serves it to the next process.
_edge_trapezoid_jit = jax.jit(
    edge_trapezoid, static_argnames=("variable", "side", "edge_aa", "axis")
)


def _ssp_fingerprint(
    ssp_wave: np.ndarray,
    ssp_flux: np.ndarray,
    ssp_lgmet: np.ndarray,
    ssp_log_age_yr: np.ndarray | None = None,
    max_log_age: float | None = None,
) -> tuple:
    """Content fingerprint for an SSP grid: including the flux VALUES.

    Parameters
    ----------
    ssp_wave : array_like, shape (n_wave,)
        SSP wavelength grid [Angstrom].
    ssp_flux : array_like, shape (n_met, n_age, n_wave)
        SSP flux grid [Lsun/Hz/Msun].
    ssp_lgmet : array_like, shape (n_met,)
        SSP metallicity grid, log10(Z) absolute.
    ssp_log_age_yr : array_like, shape (n_age,), optional
        SSP age axis, log10(age/yr). Which bins are fitted, which are thinned and
        which are counted at all follow from it, so it is part of the key.
    max_log_age : float, optional
        Oldest counted age, log10(age/yr); ``None`` counts every age. A table
        built under one cutoff is never served to another.

    Returns
    -------
    tuple
        Hashable key. Two SSP grids share it only if they are the same grid.

    Notes
    -----
    The flux is digested, not merely shaped. ``ssp_wave`` and ``ssp_lgmet`` are
    hashed byte for byte, and the flux enters as a SHA-256 digest. A bare-stellar
    grid and its with-nebular-emission twin share a wavelength axis, a metallicity
    axis, a shape and a dtype, and differ only in the flux (by a factor of about
    100 in the Lyman continuum, which is the part this table integrates). Keying on
    the flux therefore keeps the two grids in separate entries, in memory and on
    disk. The key also carries the computation dtype (#2849), so the
    :func:`_load_ionspec_disk` rejection of non-finite tables (#458) is a second
    check and not the only one.

    The digest costs one SHA-256 pass over the flux array per fingerprint, which is
    a fraction of the SSP's own load time and is dwarfed by the power-law fits it
    guards.
    """
    flux = np.ascontiguousarray(np.asarray(ssp_flux))
    return (
        _IONSPEC_CACHE_VERSION,
        tuple(flux.shape),
        str(flux.dtype),
        bytes(np.asarray(ssp_wave).tobytes()),
        bytes(np.asarray(ssp_lgmet).tobytes()),
        _hashlib.sha256(flux.tobytes()).hexdigest(),
        _computation_dtype(),
        None if ssp_log_age_yr is None else bytes(np.asarray(ssp_log_age_yr).tobytes()),
        None if max_log_age is None else float(max_log_age),
        (MAX_OLD_FITS_PER_METALLICITY, MIN_COUNTED_OLD_LOG_QION),
    )


def _computation_dtype() -> str:
    """Float precision the table is computed in: ``float64`` with x64 on, else ``float32``.

    The edge quadrature runs in JAX, so its precision follows ``jax_enable_x64``,
    not the dtype of the input array. A float64 SSP built with x64 off therefore
    yields a float32 table, and that table does not share a key with the x64-on one
    (#2849).

    Returns
    -------
    str
        ``"float64"`` or ``"float32"``, the dtype JAX returns for a Python float
        under the current ``jax_enable_x64`` setting.
    """
    return str(jnp.result_type(float))


def _fingerprint_hash(key: tuple) -> str:
    """SHA-256 of the fingerprint, hex-encoded, for use as a disk filename.

    Includes the version in the digest so that cache invalidation is automatic
    when the algorithm or table format changes; old entries are simply
    unaddressable under the new hash.
    """
    h = _hashlib.sha256()
    h.update(repr(key[0]).encode())  # version
    h.update(repr(key[1:3]).encode())  # shape + dtype
    h.update(key[3])  # raw wave bytes
    h.update(key[4])  # raw lgmet bytes
    h.update(key[5].encode())  # flux content digest: without it, see _ssp_fingerprint
    h.update(key[6].encode())  # computation dtype: float32 and float64 tables differ
    h.update(b"-" if key[7] is None else key[7])  # age axis: selects which bins are fitted
    h.update(repr(key[8]).encode())  # age cutoff: a young-only table is not an all-age table
    h.update(repr(key[9]).encode())  # old-bin thinning limit: it changes the copied shapes
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
    """Load a cached ionspec table from disk if it exists, else return None.

    Rejects caches whose ``logqion_table`` contains ``inf`` (legacy artefact
    of the float32 overflow bug: issue #458). Such tables were written by
    older versions and would silently poison Cue's nebular forward pass on
    every subsequent process. Treating them as a miss forces a refit, which
    after the float32 fix yields finite values.
    """
    try:
        path = _ionspec_disk_cache_dir() / f"{_fingerprint_hash(key)}.npz"
        if not path.exists():
            return None
        with np.load(path) as data:
            logqion = np.asarray(data["logqion_table"])
            if np.isinf(logqion).any():
                # An overflowed (inf) table is not usable: discard and recompute.
                return None
            return {
                "ionspec_table": data["ionspec_table"],
                "logqion_table": logqion,
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
    run reductions that multi-threaded BLAS / OpenMP may order differently
    from run to run (thread scheduling perturbs the last ULPs of the
    solution), and those bits then bake into the JAX trace as
    constants: so different processes produce different HLO modules. Pinning
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


# Ionization edges (Angstrom): from cue/constants.py
HEII_EDGE = 1e8 / 438908.8789  # 227.84 A
OII_EDGE = 1e8 / 283270.9  # 353.07 A
HEI_EDGE = 1e8 / 198310.66637  # 504.26 A
# The one Lyman edge (tengri.components.lyc); local alias kept for the
# existing HI_LIMIT call sites in this module and one test import.
HI_LIMIT = LYMAN_LIMIT_AA

# Segment boundaries: [1, HeII, OII, HeI, HI]
SEGMENT_EDGES = host_array([1.0, HEII_EDGE, OII_EDGE, HEI_EDGE, HI_LIMIT])


def require_ionizing_coverage(ssp_wave, *, backend: str) -> None:
    """Refuse an SSP wavelength axis with no point at or below the Lyman limit (#3444).

    Parameters
    ----------
    ssp_wave : array_like, shape (n_wave,)
        SSP wavelength axis [Angstrom].
    backend : str
        Name of the photoionized backend that asks, quoted in the error.

    Raises
    ------
    ConfigError
        If no wavelength is at or below ``HI_LIMIT`` [Angstrom]. Such a grid has no
        ionizing spectrum, so the backend cannot predict nebular emission from it.

    Notes
    -----
    The mask is the same ``wave <= HI_LIMIT`` the ionizing-table builder uses, so the
    check and the builder cannot disagree. Pure NumPy, no JAX transform.
    """
    wave_axis = np.asarray(ssp_wave, dtype=np.float64)
    if np.any(wave_axis <= HI_LIMIT):
        return
    raise ConfigError(
        f"neb backend '{backend}' cannot use this SSP grid: its wavelength axis "
        f"starts at {wave_axis.min():.6g} A, above the Lyman limit at {HI_LIMIT:.0f} A, "
        "so it carries no ionizing spectrum from which to predict nebular emission. "
        "A grid truncated above the limit is not a dark grid, so the build is refused.\n"
        "\n"
        "Fix (one of):\n"
        "  1. Use an SSP grid whose wavelength axis extends below 912 A.\n"
        "  2. Use a nebular backend that does not need the ionizing spectrum.\n"
        "  3. Drop the nebular emission: neb={'type': 'none'}."
    )


# Cue-emulator range of the 7 ionizing-shape parameters, used to clip the
# auto-derived (SSP-fit) ionspec coefficients so the Cue neural emulator is never
# evaluated outside the range it was trained on.
#
# Source: the Cue reference code (github.com/efburnham/cue; ``emulator.py``
# docstring and the ``np.clip`` bounds of the SSP-fit routine in ``utils.py``):
# index1 [1, 42], index2 [-0.3, 30], index3 [-1, 14], index4 [-1.7, 8],
# logLratio1 [-1, 10.1], logLratio2 [-0.5, 1.9], logLratio3 [-0.4, 2.2].
#
# Discrepancy with the paper: Li et al. 2025 (ApJ 986, 9, arXiv:2405.04598),
# Table 1, prints index3 [-1.1, 14] and logLratio1 [-0.1, 10.1]. The other five
# agree. The code values are used because the paper's own text places AGN power
# laws with 1.2 <= alpha <= 2 inside the training set, and those have
# logLratio1 between -0.86 and -0.26 (Eq. 2 definition), which a -0.1 floor would
# exclude; the printed -0.1 is most likely a dropped digit of -1.0. Measured on
# the emulator itself, lines vary smoothly in logLratio1 from -1.0 to +0.3.
#
# These are DISTINCT from, and deliberately NOT unified with, the user-facing
# prior bounds in ``components/nebular/_params.py::CUE_IONSPEC_PARAMS`` (e.g.
# index1 clips to [1, 42] here vs. the prior range [0, 50] there): the prior
# range is what a user may sample, the clip range is what the emulator can
# faithfully evaluate. See #887 (which confirmed they must stay separate).
_CLIP_RANGES = {
    "ionspec_index1": (1.0, 42.0),
    "ionspec_index2": (-0.3, 30.0),
    "ionspec_index3": (-1.0, 14.0),
    "ionspec_index4": (-1.7, 8.0),
    "ionspec_logLratio1": (-1.0, 10.1),
    "ionspec_logLratio2": (-0.5, 1.9),
    "ionspec_logLratio3": (-0.4, 2.2),
}

#: Parameter names in the order of the 7-vector the Cue backend carries.
IONSPEC_NAMES: tuple[str, ...] = tuple(_CLIP_RANGES)


def clamp_ionspec_straight_through(i7):
    """Clamp a 7-vector of ionizing-shape parameters to the Cue training box.

    Parameters
    ----------
    i7 : array_like, shape (7,)
        ``[ionspec_index1..4, ionspec_logLratio1..3]`` [dimensionless].

    Returns
    -------
    jnp.ndarray, shape (7,)
        The input moved onto the box ``_CLIP_RANGES`` (Cue's reference bounds; the
        comment above ``_CLIP_RANGES`` records how they differ from Li et al. 2025,
        Table 1).

    Notes
    -----
    Straight-through: the forward value is ``clip(x)`` and, for finite input, the
    gradient is the identity, so a fit driven past the box keeps a signal (a plain
    clip would zero it). The forward value is ``clip(x)`` plus ``x - x`` (exactly
    0.0), so it equals the bound out of range and the input bit-for-bit in range.

    The out-of-range gradient is deliberately inconsistent with the pinned
    value: the value stays at the bound while the gradient still says "move".
    Non-finite input: ``+/-inf`` returns the bound (zero gradient), ``nan`` stays
    ``nan``. JIT/grad/vmap safe.
    """
    x = jnp.asarray(i7)
    lo = jnp.asarray([_CLIP_RANGES[k][0] for k in IONSPEC_NAMES], dtype=x.dtype)
    hi = jnp.asarray([_CLIP_RANGES[k][1] for k in IONSPEC_NAMES], dtype=x.dtype)
    # Double-where: the straight-through arithmetic only ever sees finite input
    # (``inf - inf`` is NaN, and a single outer ``where`` still differentiates that
    # branch), and the non-finite branch is a constant for autodiff, so a +/-inf or
    # NaN component carries neither a NaN nor a tangent into the finite ones.
    finite = jnp.isfinite(x)
    x_safe = jnp.where(finite, x, 0.5 * (lo + hi))
    clipped_safe = jnp.clip(x_safe, lo, hi)
    through = jax.lax.stop_gradient(clipped_safe) + (x_safe - jax.lax.stop_gradient(x_safe))
    bound = jax.lax.stop_gradient(jnp.clip(x, lo, hi))  # +/-inf -> bound, nan -> nan
    return jnp.where(finite, through, bound)


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

    **Host routine**: NumPy in, NumPy out (the fit itself runs in JAX, float64);
    intended for one-time precomputation of SSP grids before inference.

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

        in log-log space via least-squares regression, with a term on the segment
        photon rate so the fit keeps the ionizing photon count, not just the
        spectral shape. The minimizer is the exact stationary point of that
        objective, from the one solver
        :func:`~tengri.components.nebular.ionizing_fit.fit_ionizing_shape`
        (run in float64), which the Cue backend also uses on composite spectra.

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
        independent of absolute normalization.

    **Error handling**:
        Returns sensible defaults (zero slope, −∞ norm) for segments with
        no/negligible ionizing flux (e.g., segment 1 in solar-metallicity old SSPs).

    **Clipping**:
        Output values are clipped to physically motivated ranges stored in
        _CLIP_RANGES to prevent Cue emulator extrapolation failures.

    References
    ----------
    .. [1] Y. Li et al., "Cue: A Fast and Flexible Photoionization Emulator
       for Modeling Nebular Emission Powered by Almost Any Ionizing Source,"
       ApJ, 986, 9 (2025). arXiv:2405.04598.
       https://doi.org/10.3847/1538-4357/adcab4

    """
    # float64 for the whole fit, whatever the SSP shipped as (#1206).
    #
    # The reference fit's floors (``1e-99`` on the flux, ``1e-70 * norm`` on the
    # normalized flux) lie below what float32 can hold; on a float32 SSP they become
    # exactly 0.0 and ``log10(0) = -inf`` would enter the objective (#1206: 198 of
    # 5580 table entries went ``-inf``). The fit runs in log space and in float64
    # here, and the Q_H integrand below is formed in float64 for the same reason. This
    # is a build-time, once-per-SSP routine producing a small table, so float64 costs
    # nothing.
    #
    # float64 input is unaffected: ``astype`` on a float64 array is a no-op copy.
    wave = np.asarray(wave, dtype=np.float64)
    flux = np.asarray(flux, dtype=np.float64)

    from tengri.components.nebular.ionizing_fit import compiled_fit

    # The fit reads the pixels up to the one just past the HI edge; a wider grid is fine.
    _, fit = compiled_fit(wave, edges)
    with jax.enable_x64(True):
        log_flux = np.log10(np.maximum(flux, np.finfo(np.float64).tiny))
        shape = fit(jnp.asarray(log_flux, dtype=jnp.float64))
        slopes = np.asarray(shape.slopes, dtype=np.float64)
        coeff = np.stack([slopes, np.asarray(shape.log_amp, dtype=np.float64)], axis=1)
        log_L = np.asarray(shape.log_seglum, dtype=np.float64) + np.log10(_C_AA * _LSUN)
    logLratios = np.diff(log_L)

    # Total Q_H (#458, #2849). The integrand (flux * L_sun) / (h nu) is formed in
    # float64 NumPy. Its integral reaches ~1e46-1e47, past float32's ceiling, so
    # the quadrature runs on the integrand divided by a power of two (exact in
    # binary floating point) and the exponent is restored in float64. The exponent
    # is set from the finite ionizing pixels only: a NaN or inf beyond the Lyman
    # edge does not enter the Q_H integral, so it must not set the scale either.
    # The log is taken in NumPy.
    integrand = (flux * _LSUN) / (_H_PLANCK * (_C_AA / wave))
    ionizing_mag = np.where((wave < HI_LIMIT) & np.isfinite(integrand), np.abs(integrand), 0.0)
    exponent = int(np.frexp(np.max(ionizing_mag, initial=0.0))[1])
    q_scaled = float(
        _edge_trapezoid_jit(
            np.ldexp(integrand, -exponent),
            wave,
            variable="nu",
            side="ionizing",
            edge_aa=HI_LIMIT,
        )
    )
    log_qion = float(np.log10(max(np.ldexp(q_scaled, exponent), 1e-99)))

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
        # Absolute integrated luminosity per segment, log10. The 3 ``logLratio``
        # params are ``diff(log_seglum)`` and therefore discard this
        # normalization -- but combining several stellar populations needs it,
        # because segment luminosities ADD linearly across populations while
        # their log-ratios do not (#1018).
        "log_seglum": log_L,
    }


#: log10(age/yr) of 100 Myr. Ages up to here are always fitted bin by bin; older ages
#: are counted (by default) from the thinned fits described at
#: :data:`MAX_OLD_FITS_PER_METALLICITY`. It is also the cutoff the former hard age cut
#: used, ``neb={'type': 'cue', 'lyc_max_age_yr': 1e8}``. Lives in :mod:`ionizing_spectrum`
#: rather than :mod:`cue` to avoid a circular import (``cue`` already depends on this
#: module).
MAX_NEB_LOG_AGE: float = 8.0  # 100 Myr

#: Value of ``neb={'type': 'cue', 'lyc_max_age_yr': ...}`` that counts every stellar age.
LYC_ALL_AGES: str = "all"


def validate_lyc_max_age_yr(value: object) -> str | float:
    """Validate a ``lyc_max_age_yr`` value: ``'all'`` or a positive finite age [yr].

    Parameters
    ----------
    value : object
        The user's value: ``'all'`` or a number of years (``1e8`` is the former 100 Myr cut).

    Returns
    -------
    str or float
        ``'all'`` unchanged, or the age as a float [yr].

    Raises
    ------
    ConfigError
        If ``value`` is neither ``'all'`` nor a positive finite real number (a ``bool`` is
        refused: ``True`` would read as 1 yr).
    """
    from tengri.config.exceptions import ConfigError

    if isinstance(value, str) and value == LYC_ALL_AGES:
        return LYC_ALL_AGES
    is_number = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )
    if not is_number or not np.isfinite(value) or value <= 0:
        raise ConfigError(
            f"neb lyc_max_age_yr must be 'all' or a positive finite age in years "
            f"(1e8 is the former 100 Myr cut); got {value!r}."
        )
    return float(value)


def lyc_max_log_age(value: str | float) -> float | None:
    """The cutoff as log10(age/yr), or ``None`` when every age is counted.

    Parameters
    ----------
    value : str or float
        A validated ``lyc_max_age_yr``: ``'all'`` or an age [yr].

    Returns
    -------
    float or None
        ``log10(value)`` [log10(yr)]; ``None`` for ``'all'``. ``1e8`` maps to exactly 8.0,
        which is :data:`MAX_NEB_LOG_AGE`, so the former mask is reproduced bit for bit.
    """
    return None if value == LYC_ALL_AGES else float(np.log10(value))


#: Most old bins (older than :data:`MAX_NEB_LOG_AGE`) fitted per metallicity. A grid with
#: more old ages than this fits a log-spaced subset of that size; every other old bin
#: takes the shape of the nearest fitted bin and its own exact Q_H. FSPS and MIST grids
#: (46 old ages) fit every bin; BC03 from CIGALE (13600 old ages) fits this many,
#: which takes seconds where fitting every bin takes 7-14 minutes.
MAX_OLD_FITS_PER_METALLICITY: int = 128

_NO_IONIZING_LOG_Q: float = -99.0  #: ``logqion_table`` value of a bin with no Lyman continuum

#: Old bins (older than :data:`MAX_NEB_LOG_AGE`) with ``log10 Q_H`` [photons/s/Msun] below
#: this are left unfitted, like a bin with no Lyman continuum. A galaxy of 1e13 Msun at
#: this rate emits 1e37 photons/s, an Halpha luminosity of order 1e25 erg/s, which is
#: nothing next to any observed galaxy; and the SSP flux that yields it (below 1e-38 Lsun/Hz
#: in the Lyman continuum) is at the edge of float32 storage, so a fit of its shape would
#: depend on the storage dtype rather than on the stellar population.
MIN_COUNTED_OLD_LOG_QION: float = 24.0


def _thinned_old_indices(old_idx: np.ndarray, log_age: np.ndarray, limit: int) -> np.ndarray:
    """The subset of old bins that gets its own fit, log-spaced in age.

    Parameters
    ----------
    old_idx : ndarray of int, shape (n_old,)
        Indices of the old bins in the SSP age axis, ascending in age.
    log_age : ndarray, shape (n_age,)
        log10(age/yr) of every SSP age bin.
    limit : int
        Most bins to select.

    Returns
    -------
    ndarray of int, shape (<= limit,)
        ``old_idx`` itself when it already fits within ``limit``; otherwise the bins
        nearest to ``limit`` log-spaced ages across the old range, endpoints included.
    """
    if old_idx.size <= limit:
        return old_idx
    ages = log_age[old_idx]
    targets = np.linspace(ages[0], ages[-1], limit)
    nearest = np.abs(ages[None, :] - targets[:, None]).argmin(axis=1)
    return old_idx[np.unique(nearest)]


def _exact_log_qion(
    ssp_wave: np.ndarray, ssp_flux: np.ndarray, rows: np.ndarray, chunk: int = 2048
) -> np.ndarray:
    """``log10 Q_H`` of the selected spectra by the same quadrature the per-bin fit uses.

    Parameters
    ----------
    ssp_wave : ndarray, shape (n_wave,)
        Wavelength grid, ascending [Angstrom].
    ssp_flux : ndarray, shape (n_age, n_wave)
        Spectra of one metallicity [Lsun/Hz/Msun].
    rows : ndarray of int, shape (n_rows,)
        Which spectra to integrate.
    chunk : int, optional
        Spectra integrated per pass (bounds memory on very fine age grids).

    Returns
    -------
    ndarray, shape (n_rows,)
        ``log10`` of the ionizing photon rate [photons/s/Msun]; -99 where there is none.

    Notes
    -----
    The integrand ``L_nu / (h nu)`` is formed in float64 NumPy and divided row by row by a
    power of two before the JAX quadrature, then restored in float64, as in
    :func:`fit_ionizing_spectrum` (#458, #2849), so the result is finite in pure float32.
    Only the grid up to the first point at or above the Lyman edge enters the ionizing
    side of :func:`~tengri.components.lyc.edge_trapezoid`, so the rest is dropped.
    """
    wave = np.asarray(ssp_wave, dtype=np.float64)
    last = int(np.searchsorted(wave, HI_LIMIT, side="left"))
    cut = min(last + 1, wave.size)
    wave_ion = wave[:cut]
    nu = _C_AA / wave_ion
    out = np.empty(rows.size)
    for start in range(0, rows.size, chunk):
        flux = np.asarray(ssp_flux[rows[start : start + chunk], :cut], dtype=np.float64)
        integrand = (flux * _LSUN) / (_H_PLANCK * nu)
        magnitude = np.where(
            (wave_ion < HI_LIMIT) & np.isfinite(integrand), np.abs(integrand), 0.0
        ).max(axis=1, initial=0.0)
        exponent = np.frexp(magnitude)[1]
        q_scaled = np.asarray(
            _edge_trapezoid_jit(
                np.ldexp(integrand, -exponent[:, None]),
                wave_ion,
                variable="nu",
                side="ionizing",
                edge_aa=HI_LIMIT,
            )
        )
        q = np.ldexp(q_scaled.astype(np.float64), exponent)
        # log10 of the positive rates only; a bin with none keeps the -99 'no photons' value.
        positive = q > 0.0
        log_q = np.log10(np.where(positive, q, 1.0))
        out[start : start + chunk] = np.where(
            positive, np.maximum(log_q, _NO_IONIZING_LOG_Q), _NO_IONIZING_LOG_Q
        )
    return out


def _share_nearest_fit(
    tables: dict,
    log_age: np.ndarray,
    fitted: np.ndarray,
    old_idx: np.ndarray,
    exact_log_q: np.ndarray,
    live: np.ndarray,
) -> None:
    """Give every unfitted live old bin the nearest fitted shape and its own exact Q_H.

    Parameters
    ----------
    tables : dict
        ``ionspec_table`` (n_met, n_age, 7) and ``logqion_table`` (n_met, n_age);
        modified in place.
    log_age : ndarray, shape (n_age,)
        log10(age/yr) of each SSP age bin.
    fitted : ndarray of bool, shape (n_met, n_age)
        Bins whose own fit succeeded.
    old_idx : ndarray of int, shape (n_old,)
        Counted bins older than :data:`MAX_NEB_LOG_AGE`.
    exact_log_q : ndarray, shape (n_met, n_old)
        ``log10 Q_H`` of each old bin by :func:`_exact_log_qion` [photons/s/Msun].
    live : ndarray of bool, shape (n_met, n_old)
        Old bins above :data:`MIN_COUNTED_OLD_LOG_QION`.

    Notes
    -----
    The donor is the nearest fitted old bin in log-age at the same metallicity: the
    slopes and segment ratios are the donor's, and the bin's own photon rate is
    exact. A bin below the Q_H floor, or one with no fitted donor at its metallicity,
    keeps the unfitted convention (zeros, -99).
    """
    ionspec, logq = tables["ionspec_table"], tables["logqion_table"]
    for im in range(ionspec.shape[0]):
        donors = old_idx[fitted[im, old_idx]]
        pending = live[im] & ~fitted[im, old_idx]
        if donors.size == 0 or not pending.any():
            continue
        todo, exact = old_idx[pending], exact_log_q[im, pending]
        nearest = np.abs(log_age[todo][:, None] - log_age[donors][None, :]).argmin(axis=1)
        donor = donors[nearest]
        ionspec[im, todo] = ionspec[im, donor]
        logq[im, todo] = exact


def _memoize_ionspec_table(key: tuple, table: dict) -> Mapping:
    """Store ``table`` in the process memo as a read-only mapping and return it.

    The memo is shared by every model built on the same SSP content, and callers
    key their own bookkeeping on the identity of the entry, so the same object is
    returned every time. It must therefore not be writable: the mapping refuses
    assignment and the arrays are marked read-only. A caller that wants to change
    a field copies first (``dict(table)``).
    """
    for value in table.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    frozen = MappingProxyType(table)
    _IONSPEC_TABLE_CACHE[key] = frozen
    return frozen


def precompute_ionizing_params_table(
    ssp_wave: np.ndarray,
    ssp_flux: np.ndarray,
    ssp_lgmet: np.ndarray,
    ssp_log_age_yr: np.ndarray | None = None,
    max_log_age: float | None = None,
) -> Mapping:
    """Precompute Cue ionizing parameters for a full SSP grid.

    Batch-fits piecewise power laws to (metallicity, age) SSP spectra. Bins up to
    100 Myr are fitted one by one. Older bins are counted too (unless ``max_log_age``
    excludes them): a grid with more than :data:`MAX_OLD_FITS_PER_METALLICITY` old ages
    fits a log-spaced subset, and every other old bin takes the shape of the nearest
    fitted one with its own exact Q_H. Called once at model initialization; results
    are stored and interpolated at runtime.

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        Wavelength grid in Å. [Å]
    ssp_flux : array, shape (n_met, n_age, n_wave)
        SSP spectra on (metallicity, age) grid. [erg/s/Hz/Msun]
    ssp_lgmet : array, shape (n_met,)
        Metallicity grid in log10(Z). [log10(Z)]
    ssp_log_age_yr : array, shape (n_age,), optional
        log10(age/yr) for each age bin. Required for the old-bin handling: when
        ``None`` (legacy callers) every bin is attempted, none is thinned, and
        ``max_log_age`` is ignored; the per-bin ``np.max(flux_iz) <= 0`` early-exit
        still skips dead bins.
    max_log_age : float, optional
        Oldest age to count, log10(age/yr). Bins older than this are left unfitted
        (``logqion_table = -99``), the table the former 100 Myr cut used when this is
        :data:`MAX_NEB_LOG_AGE`. ``None`` (default) counts every age.

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
    **One-time cost**: a host routine (NumPy tables, the fit in JAX). Call
    once per Cue model initialization and store results (e.g., in h5 file).
    Precomputation is O(n_met × n_fitted_ages × n_segments); n_fitted_ages is the number
    of bins up to 100 Myr plus at most :data:`MAX_OLD_FITS_PER_METALLICITY` older ones.

    **Old bins**: ``logqion_table`` of a thinned bin is its exact edge-aware Q_H, the
    same quadrature as a fitted bin; only its slopes and segment ratios are borrowed.

    **Metadata**: Unfittable SSPs (zero ionizing flux, corrupted data) are left with all
    parameters = 0 (or −99 for log-space). Downstream code handles these gracefully via
    clipping and default fallback values.

    """
    # float64 before anything else, including the cache key (#1206).
    #
    # Same reason as :func:`fit_ionizing_spectrum`, which this calls per bin, but
    # the ORDER matters here for a second reason: the fingerprint below is taken
    # from these arrays, so casting first makes a float32 SSP and its float64
    # twin hash to the SAME key. Without that they key separately, and a session
    # running with ``JAX_ENABLE_X64=0`` would both compute a corrupt table and
    # persist it to the shared on-disk cache under its own key, where it would be
    # served to every later float32 process. Casting first also strands any such
    # entry already written by a pre-fix run: it lives under a key nothing
    # computes any more.
    ssp_wave = np.asarray(ssp_wave, dtype=np.float64)
    ssp_flux = np.asarray(ssp_flux, dtype=np.float64)
    ssp_lgmet = np.asarray(ssp_lgmet, dtype=np.float64)

    # Memoize: same SSP grid → identical result. Two-tier cache:
    # 1. In-process dict for fast repeat builds within a single process
    #    (introduced in #418).
    # 2. On-disk ``.npz`` keyed on a SHA-256 of the SSP fingerprint, so
    #    cold processes load a bit-identical table instead of re-fitting.
    #    Coefficients that differ by a few ULPs between processes would bake
    #    into the JAX trace as constants, and the JAX persistent compilation
    #    cache would miss on every fresh process. The single_thread_blas
    #    wrapper below removes BLAS-threading non-determinism from the fits.
    # The age axis and the cutoff are in the key: they decide which bins are fitted.
    key = _ssp_fingerprint(
        ssp_wave, ssp_flux, ssp_lgmet, ssp_log_age_yr=ssp_log_age_yr, max_log_age=max_log_age
    )
    cached = _IONSPEC_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    disk_cached = _load_ionspec_disk(key)
    if disk_cached is not None:
        return _memoize_ionspec_table(key, disk_cached)

    n_met, n_age, _ = ssp_flux.shape
    ionspec_table = np.zeros((n_met, n_age, 7))
    logqion_table = np.full((n_met, n_age), _NO_IONIZING_LOG_Q)

    wave_np = np.asarray(ssp_wave)

    # Which bins get their own fit. Without an age axis every bin does. With one,
    # bins up to 100 Myr always do; counted older bins do too, up to the thinning limit
    # per metallicity, among those with a photon rate worth counting (the rest of the
    # live ones are filled from their nearest fitted neighbor below).
    fit_mask = np.ones((n_met, n_age), dtype=bool)
    old_idx = np.zeros(0, dtype=int)
    if ssp_log_age_yr is not None:
        log_age = np.asarray(ssp_log_age_yr, dtype=np.float64)
        counted = np.ones(n_age, dtype=bool) if max_log_age is None else log_age <= max_log_age
        old = counted & (log_age > MAX_NEB_LOG_AGE)
        old_idx = np.where(old)[0]
        fit_mask[:] = (counted & ~old)[None, :]
    else:
        log_age = np.zeros(n_age)
    exact_log_q = np.zeros((n_met, old_idx.size))
    live = np.zeros((n_met, old_idx.size), dtype=bool)
    for im in range(n_met if old_idx.size else 0):
        exact_log_q[im] = _exact_log_qion(wave_np, ssp_flux[im], old_idx)
        live[im] = exact_log_q[im] >= MIN_COUNTED_OLD_LOG_QION
        chosen = _thinned_old_indices(old_idx[live[im]], log_age, MAX_OLD_FITS_PER_METALLICITY)
        fit_mask[im, chosen] = True

    fitted = np.zeros((n_met, n_age), dtype=bool)
    # The ionizing slice depends on the wavelength axis alone, so it is formed once. Its
    # reduction is empty only when the axis has no point at or below the Lyman limit; the
    # backend refuses such a grid by name before it gets here (#3417).
    ionizing_mask = wave_np <= HI_LIMIT
    with _single_thread_blas():
        for im in range(n_met):
            for ia in range(n_age):
                if not fit_mask[im, ia]:
                    continue
                flux_np = np.asarray(ssp_flux[im, ia, :])

                # Skip if no ionizing flux
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
                    fitted[im, ia] = True
                except (ValueError, IndexError):
                    # ValueError: invalid input data (NaN, zero flux, wrong wavelength range)
                    # IndexError: wavelength array doesn't cover ionizing regime (<912 A)
                    continue

    if old_idx.size:
        _share_nearest_fit(
            {
                "ionspec_table": ionspec_table,
                "logqion_table": logqion_table,
            },
            log_age,
            fitted,
            old_idx,
            exact_log_q,
            live,
        )

    result = {
        "ionspec_table": ionspec_table,
        "logqion_table": logqion_table,
        "n_met": n_met,
        "n_age": n_age,
    }
    _store_ionspec_disk(key, result)
    return _memoize_ionspec_table(key, result)


def _cell_fraction(target, grid, idx):
    """Fractional position of ``target`` within grid cell ``idx``, -inf-safe (#1001).

    SSP grids with an age-0 anchor carry ``log10(age) = -inf`` as the first entry,
    making that cell's span infinite; the naive fraction is ``inf/inf`` -> NaN and
    silently poisons every downstream Cue prediction. Degenerate cells snap to the
    nearest node. Used by :func:`interpolate_ionizing_params`.
    """
    left = grid[idx]
    span = grid[idx + 1] - left
    finite = jnp.isfinite(left) & jnp.isfinite(span)
    left_safe = jnp.where(finite, left, 0.0)
    span_safe = jnp.where(finite, span, 1.0)
    frac = (target - left_safe) / span_safe
    return jnp.where(finite, frac, jnp.where(target == left, 0.0, 1.0))


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
        Precomputed ionizing spectrum parameters (ionspec_index1..4, logLratio1..3).
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
    **JIT-compatible**: yes, all operations use ``jnp`` primitives with
    searchsorted and linear interpolation.

    **Clipping**: Target (log_z, log_age_yr) are clipped to grid bounds
    before interpolation to prevent extrapolation artifacts.

    **Bilinear interpolation**: Uses 2×2 neighborhood of grid points,
    with fractional weights (fz, fa) computed from target position within
    the bracketing cell.

    **Infinite grid edges** (#1001): SSP grids with an age-0 anchor bin
    carry ``log10(age) = -inf`` as the first age entry, making the first
    cell's span infinite. The naive fraction ``(x - (-inf)) / (edge -
    (-inf)) = inf/inf`` is NaN and silently poisoned every downstream Cue
    prediction whose SFH weighted the age-0 bin. Such degenerate cells now
    snap to the nearest node: the fraction is 0 when the target sits on
    the infinite edge itself and 1 otherwise (any finite target is
    infinitely far, in log space, from an age-0 edge). Guarded with
    finite-dummy ``jnp.where`` so no NaN enters forward or gradient passes.

    """

    # Bilinear interpolation
    log_z_c = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
    log_age_c = jnp.clip(log_age_yr, ssp_log_age_yr[0], ssp_log_age_yr[-1])

    iz = jnp.clip(jnp.searchsorted(ssp_lgmet, log_z_c) - 1, 0, len(ssp_lgmet) - 2)
    ia = jnp.clip(jnp.searchsorted(ssp_log_age_yr, log_age_c) - 1, 0, len(ssp_log_age_yr) - 2)

    fz = _cell_fraction(log_z_c, ssp_lgmet, iz)
    fa = _cell_fraction(log_age_c, ssp_log_age_yr, ia)

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
