# SPDX-License-Identifier: BSD-3-Clause
"""DSPS integration: differentiable CSP synthesis and SSP template management.

This module wraps the Differentiable Stellar Population Synthesis (DSPS) library
(Hearin et al. 2023), which provides the core forward model operation: integrating
a star formation history (SFH) with SSP templates to produce a composite stellar
population (CSP) spectrum. All operations are JAX-native and fully differentiable
via automatic differentiation.

The CSP integral is:

    L_CSP(λ) = ∫ SFR(t) × L_SSP(λ|age,Z) dt

See 3-forward-model.tex, Eq. 3.1–3.5 for the mathematical formulation and
Appendix A.1 for metallicity marginalization and precomputation schemes.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SSPData(NamedTuple):
    """Immutable container for SSP template library.

    Holds wavelength grid, flux templates, and metadata (age/metallicity grids)
    needed by the CSP integration engine. Typically loaded once from disk and
    reused across many forward-model evaluations.

    Parameters
    ----------
    ssp_wave : array, shape (n_wave,)
        Rest-frame wavelength grid [Angstrom].
    ssp_flux : array, shape (n_met, n_age, n_wave)
        Spectral luminosity density of simple stellar populations (SSPs)
        per unit stellar mass [erg/s/Hz/Msun].
        Origin: BC03, BPASS, FSPS, ProGeny, or other DSPS-compatible library.
    ssp_lg_age_gyr : array, shape (n_age,)
        Age grid in log10 space [log10(Gyr)].
    ssp_lgmet : array, shape (n_met,)
        Metallicity grid (absolute, NOT solar-relative) [log10(Z)].
        Offset: log10(Z_sun) ≈ −1.848 (Asplund+2009).
        Do NOT confuse with user-facing log10(Z/Z_sun).
    ssp_mass_remaining : array, shape (n_met, n_age), optional
        Fraction of initial stellar mass still present (living stars + remnants)
        at each (age, metallicity) [dimensionless, ∈ [0, 1]].
        Used for stellar mass normalization in CSP integral. Depends on IMF
        and isochrone library; None if unavailable.
    ssp_alpha_fe : array, optional
        Alpha enhancement grid (for future use). Currently None.
        When implemented: ssp_flux will be (n_met, n_alpha, n_age, n_wave).

    Notes
    -----
    **Metallicity convention**: ssp_lgmet is absolute log10(Z), NOT relative
    to solar. To convert user-supplied log10(Z/Z_sun) to grid coordinates,
    add LOG10_ZSUN ≈ −1.848.

    **Future extension**: ssp_alpha_fe support for alpha-element abundance
    variations (Vazdekis+2015, MIST, etc.) is planned. Currently, metallicity
    is the only dimension; alpha is fixed (typically solar, α = 0).

    **Survival mass**: ssp_mass_remaining encodes stellar mass loss due to
    stellar evolution (main-sequence turnoff, white dwarf cooling, etc.).
    It is essential for mass-based inferences but may not be present in
    older SSP libraries.

    Examples
    --------
    >>> from tengri import SSPData, load_ssp_data
    >>> ssp = load_ssp_data("data/ssp_miles.h5")  # doctest: +SKIP
    >>> ssp.ssp_flux.shape  # (n_met, n_age, n_wave)  # doctest: +SKIP
    (22, 107, 4563)

    """

    ssp_wave: jnp.ndarray
    ssp_flux: jnp.ndarray  # (n_met, n_age, n_wave) or future (n_met, n_alpha, n_age, n_wave)
    ssp_lg_age_gyr: jnp.ndarray
    ssp_lgmet: jnp.ndarray
    ssp_mass_remaining: jnp.ndarray | None = None
    # Future: ssp_alpha_fe grid for alpha-enhanced templates (Vazdekis+2015, MIST)
    # When available, ssp_flux becomes (n_met, n_alpha, n_age, n_wave) and
    # interpolation adds a third dimension. The current met_alpha_fe parameter
    # uses effective_metallicity() as an approximation for 2D grids.
    ssp_alpha_fe: jnp.ndarray | None = None
    # Initial mass function the SSP was computed under, e.g. ``"chabrier"``,
    # ``"kroupa"``, ``"salpeter"``. Surfaced by :func:`load_ssp_data` from
    # the file's HDF5 metadata if present, else parsed from the filename
    # tail (last underscore-delimited token before ``.h5``). ``"unknown"``
    # when neither path yields a match. See issue #307 for the discovery
    # gap this closes; the IMF is invisible in the model spec otherwise.
    imf: str = "unknown"
    # Provenance name the grid was loaded from (filename stem, e.g.
    # ``ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0``). Surfaced by
    # :func:`load_ssp_data` so the citation machinery can infer the SPS code,
    # isochrone set, and spectral library from the
    # ``<code>_<isochrone>_<library>_<imf>`` token convention. Empty when
    # unknown. Metadata only — never a JIT leaf.
    source: str = ""


def _sspdata_flatten(s):
    # ``imf``/``source`` are metadata, not JIT leaves — keep strings out of the trace.
    children = (
        s.ssp_wave,
        s.ssp_flux,
        s.ssp_lg_age_gyr,
        s.ssp_lgmet,
        s.ssp_mass_remaining,
        s.ssp_alpha_fe,
    )
    aux = (s.imf, s.source)
    return children, aux


def _sspdata_unflatten(aux, children):
    return SSPData(*children, imf=aux[0], source=aux[1])


jax.tree_util.register_pytree_node(SSPData, _sspdata_flatten, _sspdata_unflatten)


_LOAD_SSP_PRESETS: dict[str, str] = {
    # Short alias → full filename. Bare-stellar entries come from
    # ``_KNOWN_SSPS`` (the auto-fetch catalog); wNE entries are produced
    # locally and recorded here so demo scripts can ``load_ssp("…wNE")``
    # without spelling out the gas-grid suffix.
    "prsc_miles_chabrier_wNE": "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    "mist_c3k_a_chabrier_wNE": "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
}


def load_ssp(name: str | None = None) -> "SSPData":
    """Load an SSP grid by short name, walking parent dirs for ``data/``.

    Convenience wrapper around :func:`load_ssp_data` for tutorial and
    gallery scripts. Resolves a short alias (``"prsc_miles_chabrier_wNE"``)
    or a bare filename to a full path by walking ``Path.cwd()`` upward
    until it finds a ``data/`` directory containing the file.

    Parameters
    ----------
    name : str or None, optional
        Short alias from ``_LOAD_SSP_PRESETS``, a key from
        ``tengri.list_known_ssps()``, or a literal filename (with or
        without ``.h5``). ``None`` (default) loads the wNE PRSC/MILES
        Chabrier grid used by the dust/nebular demo scripts.

    Returns
    -------
    SSPData
        Loaded SSP container, same as :func:`load_ssp_data`.

    Raises
    ------
    FileNotFoundError
        If no ``data/<filename>.h5`` exists in any ancestor directory.

    Examples
    --------
    >>> from tengri import load_ssp
    >>> ssp = load_ssp()  # default wNE SSP
    >>> ssp = load_ssp("prsc_miles_chabrier")  # bare-stellar (for Cue)
    """
    from pathlib import Path

    from tengri._data_setup import _KNOWN_SSPS

    if name is None:
        filename = _LOAD_SSP_PRESETS["prsc_miles_chabrier_wNE"]
    elif name in _LOAD_SSP_PRESETS:
        filename = _LOAD_SSP_PRESETS[name]
    elif name in _KNOWN_SSPS:
        filename = _KNOWN_SSPS[name]
    else:
        # Accept an explicit absolute/relative path to an .h5 file directly
        # (closes #496 — reproduction notebooks ship SSPs under
        # ``reproduction/<code>/_drivers/data/`` rather than ``<root>/data/``).
        as_path = Path(name)
        if as_path.suffix == ".h5" and as_path.exists():
            return load_ssp_data(str(as_path))
        filename = name if name.endswith(".h5") else name + ".h5"

    for parent in [Path.cwd(), *Path.cwd().parents]:
        candidate = parent / "data" / filename
        if candidate.exists():
            return load_ssp_data(str(candidate))
    raise FileNotFoundError(
        f"SSP file 'data/{filename}' not found in any ancestor of {Path.cwd()}. "
        f"Place the file under <project_root>/data/ or call "
        f"tengri.download_ssp('<short_name>') to fetch a bundled SSP."
    )


def load_ssp_data(filepath: str) -> SSPData:
    """Load SSP templates from a DSPS-format HDF5 file.

    Reads stellar population synthesis templates stored in HDF5 format
    (compatible with DSPS library and distributed SSP libraries: BC03,
    BPASS, FSPS, ProGeny). Handles optional fields (ssp_mass_remaining,
    ssp_alpha_fe) gracefully.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file. Expected fields: ssp_wave, ssp_flux,
        ssp_lg_age_gyr, ssp_lgmet. Optional: ssp_mass_remaining, ssp_alpha_fe.

    Returns
    -------
    SSPData
        Loaded SSP container with all template data and metadata.

    Raises
    ------
    ImportError
        If h5py is not installed.
    KeyError
        If required HDF5 fields are missing.
    OSError
        If filepath does not exist or is not readable.

    Notes
    -----
    **JIT-compatible**: yes — only file I/O occurs; returned SSPData is
    immutable and suitable for use in JAX operations.

    **File format**: Standard DSPS HDF5 layout. See DSPS documentation
    and distributed templates on halos.as.arizona.edu for format details.

    Examples
    --------
    >>> from tengri.components.stellar.sps import load_ssp_data
    >>> ssp = load_ssp_data("data/ssp_bc03.h5")
    >>> print(ssp.ssp_wave.shape, ssp.ssp_flux.shape)
    (6000,) (50, 300, 6000)

    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for SSP loading: pip install h5py") from None

    import os
    from pathlib import Path

    fp = Path(filepath)
    if not fp.exists() and not os.environ.get("TENGRI_DISABLE_SSP_AUTODOWNLOAD"):
        # Auto-fetch from the public catalog if the basename is known.
        from tengri._data_setup import _KNOWN_SSPS, KNOWN_SSP_FILENAMES, download_ssp

        if fp.name in KNOWN_SSP_FILENAMES:
            short = next(k for k, v in _KNOWN_SSPS.items() if v == fp.name)
            print(f"[tengri] {fp} not found — fetching '{short}' from public catalog...")
            download_ssp(short, dest=fp.parent if fp.parent != Path("") else "data")

    with h5py.File(filepath, "r") as f:
        ssp_lg_age_gyr = jnp.array(f["ssp_lg_age_gyr"][:])
        ssp_lgmet = jnp.array(f["ssp_lgmet"][:])

        # IMF discovery (#307): HDF5 attribute wins, else parse filename
        # tail. Surfaced as ``ssp.imf`` so model spec / summary / gallery
        # plots can introspect the assumed IMF without grepping the
        # filename. Resolved before ``ssp_mass_remaining`` so the
        # surviving-mass synthesizer can pick the right DSPS calibration.
        imf = _detect_imf(f, fp.name)

        # Solar-luminosity unit contract (#969): ``ssp_flux`` is stored in
        # "Lsun/Hz per Msun", but WHICH Lsun depends on the code that wrote
        # the file — the ported ``fsps_*`` catalogue grids carry FSPS's
        # native numbers (Lsun = 3.839e33 erg/s, ``sps_vars.f90``), while
        # tengri converts to erg/s with the IAU 2015 value (3.828e33), a
        # flat 0.29 % absolute-flux offset. Rescale on load so the
        # in-memory arrays are IAU-Lsun-normalized and every downstream
        # conversion is exact.
        ssp_flux = jnp.array(f["ssp_flux"][:])
        native_lsun = _detect_native_lsun(f, fp.name)
        if native_lsun is not None:
            from tengri.utils.physics_constants import L_SUN

            if abs(native_lsun / L_SUN - 1.0) > 1e-12:
                ssp_flux = ssp_flux * (native_lsun / L_SUN)

        if "ssp_mass_remaining" in f:
            mass_remaining = jnp.array(f["ssp_mass_remaining"][:])
        else:
            mass_remaining = _synthesize_mass_remaining(
                filepath, ssp_lg_age_gyr, ssp_lgmet, imf_tag=imf
            )

        alpha_fe = None
        if "ssp_alpha_fe" in f:
            alpha_fe = jnp.array(f["ssp_alpha_fe"][:])

        return SSPData(
            ssp_wave=jnp.array(f["ssp_wave"][:]),
            ssp_flux=ssp_flux,
            ssp_lg_age_gyr=ssp_lg_age_gyr,
            ssp_lgmet=ssp_lgmet,
            ssp_mass_remaining=mass_remaining,
            ssp_alpha_fe=alpha_fe,
            imf=imf,
            source=fp.stem,
        )


#: FSPS's internal solar luminosity (``sps_vars.f90``, ``lsun = 3.839d33``)
#: [erg/s]. The ported ``fsps_*`` catalogue grids store flux in units of
#: this Lsun — verified by the reproduction notebook's §1 bit-match of the
#: raw HDF5 values against live ``python-fsps`` output (#969).
_FSPS_LSUN_ERG_PER_S: float = 3.839e33


def _detect_native_lsun(h5_file, filename: str) -> float | None:
    """Solar-luminosity constant the SSP file's flux units are based on (#969).

    Resolution order (mirrors :func:`_detect_imf`):

    1. ``h5_file.attrs["lsun_erg_per_s"]`` — the explicit unit contract;
       ports should write this going forward.
    2. Filename prefix ``fsps_`` → FSPS's native 3.839e33 erg/s (the
       published catalogue grids store FSPS's raw numbers; §1 of the
       Prospector reproduction pins this bit-exactly).
    3. ``None`` — unknown provenance; the loader assumes the file is
       already IAU-normalized and applies no rescale. Guessing a wrong
       constant is worse than a documented 0.3 %-scale uncertainty.

    Parameters
    ----------
    h5_file : h5py.File
        Open HDF5 handle (for the attribute lookup).
    filename : str
        Basename of the file (for the catalogue-prefix fallback).

    Returns
    -------
    float or None
        The file's native Lsun [erg/s], or ``None`` when unknown.
    """
    attr = h5_file.attrs.get("lsun_erg_per_s") if hasattr(h5_file, "attrs") else None
    if attr is not None:
        return float(attr)
    if filename.startswith("fsps_"):
        return _FSPS_LSUN_ERG_PER_S
    return None


#: IMFs that ship in the public SSP catalog. Parsed out of HDF5 metadata
#: or filename tails. Extend when a new IMF lands in ``_data_setup._KNOWN_SSPS``.
_KNOWN_IMFS: tuple[str, ...] = ("chabrier", "kroupa", "salpeter")


def _detect_imf(h5_file, filename: str) -> str:
    """Detect the IMF an SSP grid was computed under (#307).

    Resolution order:

    1. ``h5_file.attrs["imf"]`` (when SSP files start shipping the metadata).
    2. Filename tail matched against :data:`_KNOWN_IMFS`
       — e.g. ``"fsps_prsc_miles_chabrier.h5"`` → ``"chabrier"``.
    3. Fallback: ``"unknown"``.

    Falsely returning a wrong IMF is worse than returning ``"unknown"``,
    so the filename match requires an exact ``_<imf>`` token, not a
    substring (avoids ``"_kroupa_burst"`` → ``"kroupa"`` if someone
    appends qualifiers).
    """
    attr_imf = h5_file.attrs.get("imf") if hasattr(h5_file, "attrs") else None
    if attr_imf:
        # h5py returns ``bytes`` on Python 3 for string attrs from older files.
        if isinstance(attr_imf, bytes):
            attr_imf = attr_imf.decode("utf-8", errors="replace")
        return str(attr_imf).strip().lower()

    stem = filename.rsplit(".h5", 1)[0]
    tokens = stem.split("_")
    for tok in reversed(tokens):
        low = tok.lower()
        if low in _KNOWN_IMFS:
            return low
    return "unknown"


def _synthesize_mass_remaining(
    filepath, ssp_lg_age_gyr: jnp.ndarray, ssp_lgmet: jnp.ndarray, imf_tag=None
) -> jnp.ndarray:
    """Fill missing ssp_mass_remaining when the SSP HDF5 lacks the table.

    Uses :func:`dsps.imf.surviving_mstar.surviving_mstar`, a 9-parameter
    sigmoid fit to FSPS with shipped per-IMF calibrations (Chabrier,
    Salpeter, Kroupa, van Dokkum). This is the same fallback diffsky and
    other DSPS-stack codes use, so tengri's reported surviving masses stay
    bit-aligned with that ecosystem.

    Resolves the IMF via :func:`_detect_imf` (HDF5 attribute → filename
    tail → ``"unknown"``). When the IMF is ``"unknown"`` or absent, falls
    back to Chabrier with a one-shot ``UserWarning`` naming the file and
    convention. The DSPS sigmoid agrees with FSPS to 1–2 % including at
    young ages, where the IMF-turnoff integrator in
    :mod:`tengri.components.stellar.sps.mass_remaining` undercounts mass
    return by missing MS/post-MS stellar-wind losses (e.g. 0% vs 2-3% at
    1 Myr).

    Returns
    -------
    array, shape (n_met, n_age)
        Surviving mass fraction broadcast over the metallicity axis (Z
        dependence is dropped here by design; the table-supplied version,
        when present, is what carries it).
    """
    import warnings

    from dsps.imf.surviving_mstar import (
        CHABRIER_PARAMS,
        KROUPA_PARAMS,
        SALPETER_PARAMS,
        VAN_DOKKUM_PARAMS,
        surviving_mstar,
    )

    _IMF_PARAMS = {
        "chabrier": CHABRIER_PARAMS,
        "salpeter": SALPETER_PARAMS,
        "kroupa": KROUPA_PARAMS,
        "van_dokkum": VAN_DOKKUM_PARAMS,
    }

    if isinstance(imf_tag, bytes):
        imf_tag = imf_tag.decode("utf-8", errors="replace")
    imf = (imf_tag or "unknown").strip().lower()
    params = _IMF_PARAMS.get(imf, CHABRIER_PARAMS)
    if imf not in _IMF_PARAMS:
        warnings.warn(
            f"SSP file {filepath!s} has no 'ssp_mass_remaining' table and "
            f"IMF could not be resolved (got {imf!r}); synthesizing "
            "surviving-mass fractions from dsps.imf.surviving_mstar with "
            "the Chabrier-fit-to-FSPS parameters. Set the 'imf' HDF5 "
            "attribute or use a filename suffix matching _detect_imf's "
            "_KNOWN_IMFS to silence, or port the upstream table for a "
            "bit-exact match.",
            UserWarning,
            stacklevel=3,
        )

    # surviving_mstar takes log10(age/yr): ssp_lg_age_gyr is log10(age/Gyr).
    lg_age_yr = ssp_lg_age_gyr + 9.0
    f_surv_age = surviving_mstar(lg_age_yr, **params)
    return jnp.broadcast_to(f_surv_age, (ssp_lgmet.shape[0], lg_age_yr.shape[0]))


def load_ssp_data_dsps(filepath: str) -> SSPData:
    """Load SSP templates using DSPS native loader.

    Falls back to load_ssp_data() if DSPS is not installed.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file in DSPS format.

    Returns
    -------
    SSPData
        Loaded SSP template data.

    Notes
    -----
    **JIT-compatible**: yes — only file I/O occurs outside JAX traced code.
    Falls back gracefully to load_ssp_data() if dsps is not available.

    """
    try:
        from dsps import load_ssp_templates

        ssp_data = load_ssp_templates(fn=filepath)
        return SSPData(
            ssp_wave=jnp.array(ssp_data.ssp_wave),
            ssp_flux=_rescale_flux_to_iau_lsun(jnp.array(ssp_data.ssp_flux), filepath),
            ssp_lg_age_gyr=jnp.array(ssp_data.ssp_lg_age_gyr),
            ssp_lgmet=jnp.array(ssp_data.ssp_lgmet),
        )
    except ImportError:
        return load_ssp_data(filepath)


def _rescale_flux_to_iau_lsun(ssp_flux: jnp.ndarray, filepath: str) -> jnp.ndarray:
    """Apply the #969 native-Lsun rescale on the DSPS-native loader path."""
    from pathlib import Path

    try:
        import h5py

        with h5py.File(filepath, "r") as f:
            native = _detect_native_lsun(f, Path(filepath).name)
    except OSError:
        # Unreadable as HDF5 — fall back to the filename heuristic alone.
        name = Path(filepath).name
        native = _FSPS_LSUN_ERG_PER_S if name.startswith("fsps_") else None
    if native is None:
        return ssp_flux
    from tengri.utils.physics_constants import L_SUN

    if abs(native / L_SUN - 1.0) <= 1e-12:
        return ssp_flux
    return ssp_flux * (native / L_SUN)


def csp_age_dt(ssp_ages_yr: jnp.ndarray, method: str = "trapz") -> jnp.ndarray:
    """Compute CSP quadrature bin widths for a given integration method.

    Both methods implement trapezoidal integration of the CSP integral
    ∫ SFR(t) dt, but differ in the quadrature variable:

    ``"trapz"`` — standard trapezoidal rule in **linear age**:

        dt_i = 0.5 * (t_{i+1} - t_{i-1})   [interior]
        dt_0 = 0.5 * (t_1 - t_0)            [left endpoint]
        dt_N = 0.5 * (t_N - t_{N-1})        [right endpoint]

    ``"log_trapz"`` — trapezoidal rule in **log₁₀-age** with Jacobian:

        dt_i = t_i * ln(10) * d(log₁₀ t)_i

    where d(log₁₀ t)_i are the half-widths in log₁₀-age space.
    This is equivalent to the substitution x = log₁₀(t), dt = t·ln(10)·dx
    (Johnson et al. 2021, Appendix B). For log-spaced SSP grids (equal
    Δ(log₁₀ t) per bin), this achieves uniform quadrature accuracy across
    all ages, while linear trapz over-resolves old stars and under-resolves
    young stars.

    Parameters
    ----------
    ssp_ages_yr : array, shape (n_age,)
        SSP ages in years [yr], sorted ascending.
    method : {"trapz", "log_trapz"}
        Integration scheme. Default ``"trapz"`` matches DSPS.

    Returns
    -------
    array, shape (n_age,)
        Effective linear-age bin widths [yr]. Multiply by SFR [Msun/yr]
        to get mass formed per bin [Msun].

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    **Gradient-safe**: yes.

    References
    ----------
    .. [1] C. Johnson et al., "Prospector: Bayesian Stellar Population
       Inference with Separable Star Formation Histories," ApJ, 927, 74
       (2021). https://doi.org/10.3847/1538-4357/ac4867

    """
    if method == "trapz":
        return jnp.concatenate(
            [
                jnp.array([0.5 * (ssp_ages_yr[1] - ssp_ages_yr[0])]),
                0.5 * (ssp_ages_yr[2:] - ssp_ages_yr[:-2]),
                jnp.array([0.5 * (ssp_ages_yr[-1] - ssp_ages_yr[-2])]),
            ]
        )
    elif method == "log_trapz":
        log10_ages = jnp.log10(ssp_ages_yr)
        d_log10 = jnp.concatenate(
            [
                jnp.array([0.5 * (log10_ages[1] - log10_ages[0])]),
                0.5 * (log10_ages[2:] - log10_ages[:-2]),
                jnp.array([0.5 * (log10_ages[-1] - log10_ages[-2])]),
            ]
        )
        return ssp_ages_yr * jnp.log(10.0) * d_log10
    else:
        raise ValueError(
            f"Unknown CSP integration method: {method!r}. "
            "Valid options: 'trapz', 'log_trapz', 'log_interp'."
        )


def csp_log_interp_matrix(ssp_ages_yr, n_gl: int = 5):
    """Johnson+2021 log-linear SSP interpolation weight matrix.

    Returns an N×N matrix A such that ``m = A @ sfr`` gives the CSP mass
    weights when SSP spectra are interpolated **linearly in log(t)** between
    grid points and the SFR is assumed **piecewise-linear in t** between SSP
    ages (Johnson et al. 2021, ApJS 254, 22, Appendix B, Eq. B3).

    The CSP integral is approximated as:

        F_λ = ∫ SFR(t) · S_λ(t) dt
            ≈ Σ_j m_j · S_λ(t_j)

    where S_λ(t) between grid points is the log-linear interpolant:

        S_λ(t) = a_j(t)·S_λ(t_j) + b_{j+1}(t)·S_λ(t_{j+1}),  t ∈ [t_j, t_{j+1}]

        a_j(t)     = (log t_{j+1} − log t) / (log t_{j+1} − log t_j)   [falls 1→0]
        b_{j+1}(t) = (log t − log t_j) / (log t_{j+1} − log t_j)       [rises 0→1]

    Substituting a piecewise-linear SFR and integrating each interval gives:

        m_j = Σ_{intervals touching j} ∫ SFR(t) · φ_j(t) dt

    where φ_j is the hat basis function (a_j on the right interval, b_j on
    the left). This is computed via 5-point Gauss-Legendre quadrature per
    interval, exact for polynomials up to degree 9.

    The returned matrix is symmetric tridiagonal:
    - A[j, j-1] = contribution from left interval via b_j
    - A[j, j]   = sum of right-interval a_j and left-interval b_j contributions
    - A[j, j+1] = contribution from right interval via a_j (symmetric)

    Parameters
    ----------
    ssp_ages_yr : array-like, shape (n_age,)
        SSP ages in years [yr], sorted ascending.
    n_gl : int, optional
        Number of Gauss-Legendre quadrature points per interval. Default 5
        (exact for degree-9 polynomials; more than sufficient).

    Returns
    -------
    ndarray, shape (n_age, n_age)
        Weight matrix A (dimensionless, evaluated in years). Use as
        ``weights = A @ sfr_on_ssp`` to integrate the CSP.

    Notes
    -----
    **JIT-compatible**: no — uses numpy and does not support traced evaluation.
    Precompute the matrix at startup or outside JAX functions.
    **Gradient-safe**: not applicable (CPU-only computation).

    """
    import numpy as np

    ages = np.asarray(ssp_ages_yr, dtype=float)
    N = len(ages)
    A = np.zeros((N, N))

    # 5-point Gauss-Legendre nodes on [-1,1], mapped to [0,1]
    xi, wi = np.polynomial.legendre.leggauss(n_gl)
    p_nodes = (xi + 1.0) / 2.0  # in [0, 1]
    p_weights = wi / 2.0  # sum = 1

    for j in range(N - 1):
        t_lo, t_hi = ages[j], ages[j + 1]
        delta_t = t_hi - t_lo
        delta_u = np.log10(t_hi) - np.log10(t_lo)  # always > 0

        # Quadrature points in linear t
        t_q = t_lo + p_nodes * delta_t

        # Log-linear basis functions at quadrature points
        a_j = (np.log10(t_hi) - np.log10(t_q)) / delta_u  # falls 1→0
        b_j1 = 1.0 - a_j  # rises 0→1

        # SFR(t_q) = SFR_j*(1-p) + SFR_{j+1}*p  (piecewise-linear in t)
        # Contribution to m_j (integrate SFR · a_j dt over [t_j, t_{j+1}]):
        A[j, j] += delta_t * np.dot(p_weights, (1.0 - p_nodes) * a_j)
        A[j, j + 1] += delta_t * np.dot(p_weights, p_nodes * a_j)
        # Contribution to m_{j+1} (integrate SFR · b_{j+1} dt):
        A[j + 1, j] += delta_t * np.dot(p_weights, (1.0 - p_nodes) * b_j1)
        A[j + 1, j + 1] += delta_t * np.dot(p_weights, p_nodes * b_j1)

    return A


def enforce_increasing_cosmic_time(t_cosmic_asc: jnp.ndarray) -> jnp.ndarray:
    r"""Project an ascending cosmic-time table to *strictly* increasing.

    DSPS' SFH-table kernels interpolate on ``gal_t_table`` and return NaN
    when the knots are non-monotone or duplicated. At high observation
    redshift the youngest valid cosmic-time bins can fall below
    ``T_TABLE_MIN`` and clamp to the same floor as the cosmically-invalid
    ramp, so the table dips below the ramp and DSPS NaNs the whole weight
    tensor (the invalid-bin ramp alone does not cover boundary-valid bins).

    Uses a subtract-ramp / cumulative-maximum / add-ramp projection: it
    removes any inversion or tie while staying an *exact* no-op for tables
    that already increase by more than ``eps`` per step (the normal low-z
    case) — then ``t - ramp`` is still increasing, ``cummax`` is the
    identity, and adding ``ramp`` back recovers the input bit-for-bit.

    Parameters
    ----------
    t_cosmic_asc : array_like, shape (n_age,)
        Cosmic-time knots [Gyr], ascending.

    Returns
    -------
    ndarray, shape (n_age,)
        Strictly-increasing cosmic-time knots [Gyr].

    Notes
    -----
    **JIT-compatible**: yes. **Differentiable**: yes. The ``eps`` step
    (1e-6 Gyr = 1000 yr) is far below the smallest SSP age-grid spacing,
    so the perturbation to genuinely-clamped bins is negligible.

    References
    ----------
    .. [1] suchethac/tengri#683 — SFH age weights NaN at high redshift.
    """
    eps = 0.01 * 1e-4  # 1e-6 Gyr (1000 yr); far below any SSP age step
    ramp = jnp.arange(t_cosmic_asc.shape[0]) * eps
    return jnp.maximum.accumulate(t_cosmic_asc - ramp) + ramp


def compute_dsps_native_weights(
    sfr_on_ssp_ages: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    ssp_lg_age_gyr: jnp.ndarray,
    ssp_flux: jnp.ndarray,
    t_obs_gyr: float,
    lgmet: float,
    lgmet_scatter: float = 0.2,
) -> tuple:
    """Compute CSP age weights and metallicity-marginalized SSP flux via DSPS.

    **This is tengri's primary (recommended) CSP integration mode**,
    selected via ``SEDModel(..., csp_integration="dsps_native")``.

    Uses DSPS's triweight metallicity kernel (Hearin et al. 2023, Eq. 10)
    to convolve the stellar metallicity distribution with SSP templates,
    then integrates the SFH via DSPS's trapezoidal scheme on cosmic time.
    Age and metallicity integration are performed in a single DSPS call,
    eliminating the need for a separate :func:`compute_csp_weights` +
    ``interp_metallicity`` step.

    Unlike the trapezoidal quadrature modes (``trapz``, ``log_trapz``),
    this method computes the CSP on **cosmic** (not lookback) time, which
    avoids endpoint-weighting errors at young ages.  The metallicity
    distribution is a lognormal (Gaussian in log10 Z) with scatter
    ``lgmet_scatter``, matching the Prospector/DSPS convention
    (Johnson et al. 2021).  The resulting ``ssp_flux_at_z`` is already
    marginalized over the full metallicity PDF and flows into tengri's
    existing dust and AGN pipeline unchanged.

    Parameters
    ----------
    sfr_on_ssp_ages : array, shape (n_age,)
        Star formation rate (Msun/yr) evaluated at each SSP lookback age,
        sorted **ascending by age** (youngest = index 0).
    ssp_ages_yr : array, shape (n_age,)
        SSP lookback ages in years (ascending).
    ssp_lgmet : array, shape (n_met,)
        log10(Z) metallicity grid of the SSP library (absolute, not Z/Zsun).
    ssp_lg_age_gyr : array, shape (n_age,)
        log10(age/Gyr) of SSP templates.
    ssp_flux : array, shape (n_met, n_age, n_wave)
        SSP spectra in Lsun/Hz/Msun.
    t_obs_gyr : float
        Age of the universe in Gyr at the observation redshift.
        Computed from tengri's cosmology (not DSPS's DEFAULT_COSMOLOGY).
    lgmet : float
        log10(Z) metallicity of the galaxy (absolute, same units as ssp_lgmet).
    lgmet_scatter : float, optional
        Gaussian scatter in log10(Z) (dex). Default 0.2 dex, matching DSPS
        and Prospector conventions (Conroy & van Dokkum 2009; Johnson+2021).

    Returns
    -------
    age_weights_msun : array, shape (n_age,)
        Mass formed per SSP age bin (Msun), sorted ascending by age.
        Sum = total stellar mass formed.  Directly replaces the output of
        :func:`compute_csp_weights`.
    ssp_flux_at_z : array, shape (n_age, n_wave)
        SSP flux marginalized over the metallicity distribution
        (Lsun/Hz/Msun).

    Notes
    -----
    SSP ages in tengri are **lookback times** (youngest = smallest).  DSPS
    needs **cosmic times** sorted ascending.  The conversion is::

        t_cosmic_gyr = clip(t_obs_gyr - ssp_ages_yr / 1e9, min=1e-3)

    Reversal (youngest→oldest in tengri ↔ oldest→youngest in cosmic time)
    is handled internally; the returned ``age_weights_msun`` is sorted
    back to tengri's ascending-age convention.

    Requires ``dsps`` to be installed (``pip install dsps``).

    References
    ----------
    Hearin et al. 2023, arXiv:2112.08423, Eq. 10 (triweight kernel).

    """
    try:
        from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_lognormal_mdf
    except ImportError:
        raise ImportError(
            "dsps is required for csp_integration='dsps_native'. Install with: pip install dsps"
        ) from None

    # SSP ages are lookback times (young→old, ascending).
    # DSPS needs cosmic times (old→young = ascending cosmic time).
    # Reverse so that gal_t_table is sorted ascending for DSPS.
    ssp_age_gyr = ssp_ages_yr / 1e9
    t_cosmic_raw = t_obs_gyr - ssp_age_gyr  # may go ≤ 0 for SSP ages > t_obs

    # Mask out invalid bins (stars formed before the Big Bang). For
    # those bins, set SFR to zero so they contribute no mass to the
    # CSP integral. The cosmic-time array must be **strictly**
    # monotonic (no duplicates) AND every entry must satisfy
    # ``t >= T_TABLE_MIN = 0.01 Gyr`` for DSPS's internal
    # ``cumulative_mstar_formed`` and ``log10(M*)`` calls to behave
    # sanely. We give each bin a small linear ramp starting at
    # T_TABLE_MIN; valid bins keep their actual cosmic time but
    # also get a floor at T_TABLE_MIN so very-high-z observations
    # don't underflow.
    T_TABLE_MIN = 0.01  # Gyr; matches dsps.constants.T_TABLE_MIN
    n_ssp = ssp_ages_yr.shape[0]
    # Floor every cosmic time to T_TABLE_MIN.
    t_cosmic_floor = jnp.maximum(t_cosmic_raw, T_TABLE_MIN)
    # Identify invalid bins (originally ≤0).
    valid = t_cosmic_raw > 0.0

    # ``valid`` is in lookback (young→old) order; reverse to align
    # with ascending cosmic time.
    valid_asc = valid[::-1]
    t_cosmic_asc_raw = t_cosmic_floor[::-1]
    sfr_asc_raw = sfr_on_ssp_ages[::-1]

    # In ascending-cosmic-time order, invalid bins occupy the first
    # ``k = n_invalid`` indices. Give them a strict-monotonic ramp
    # in ``[T_TABLE_MIN, T_TABLE_MIN + ε]`` so they stay below the
    # youngest valid bin (which has t_cosmic > 0 by definition,
    # though potentially small).
    n_invalid = jnp.sum(~valid_asc)
    idx = jnp.arange(n_ssp)
    is_invalid_pos = idx < n_invalid
    # Ramp from T_TABLE_MIN to T_TABLE_MIN * 1.5, strictly increasing.
    ramp = T_TABLE_MIN + (T_TABLE_MIN * 0.5) * (idx + 1) / jnp.maximum(n_invalid, 1)
    t_cosmic_asc = jnp.where(is_invalid_pos, ramp, t_cosmic_asc_raw)
    # Guarantee strictly-increasing knots: at high z boundary-valid bins can
    # clamp to T_TABLE_MIN below the invalid-bin ramp, which DSPS NaNs on. #683
    t_cosmic_asc = enforce_increasing_cosmic_time(t_cosmic_asc)
    sfr_asc = jnp.where(is_invalid_pos, 0.0, sfr_asc_raw)

    result = calc_rest_sed_sfh_table_lognormal_mdf(
        gal_t_table=t_cosmic_asc,
        gal_sfr_table=sfr_asc,
        gal_lgmet=lgmet,
        gal_lgmet_scatter=lgmet_scatter,
        ssp_lgmet=ssp_lgmet,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
        ssp_flux=ssp_flux,
        t_obs=t_obs_gyr,
    )

    # ``result.weights`` is the joint (n_met, n_age) probability
    # distribution (sums to 1) used by DSPS internally to build
    # ``rest_sed``. The joint is **non-separable**: the outer product
    # of the marginals (lgmet_weights ⊗ age_weights) gives the right
    # marginals but the wrong per-bin product when convolved with
    # ``ssp_flux``, over-scaling the CSP SED by orders of magnitude.
    # Use the joint directly.
    total_mass = jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9)

    joint = result.weights  # (n_met, n_age) sum=1
    age_weights_norm = joint.sum(axis=0)  # (n_age,) sum=1
    age_weights_msun = age_weights_norm * jnp.maximum(total_mass, 0.0)

    # Per-age conditional metallicity-weighted SSP flux (Lsun/Hz/Msun).
    # weighted_ssp[a] = sum_m(joint[m, a] × ssp_flux[m, a, :])
    # ssp_flux_at_z[a] = weighted_ssp[a] / age_weights_norm[a]
    # so that ``age_weights_msun[a] × ssp_flux_at_z[a, :]`` reproduces
    # ``total_mass × weighted_ssp[a]``. Sum over a → rest_sed.
    weighted_ssp = jnp.einsum("ma,maw->aw", joint, ssp_flux)  # (n_age, n_wave)
    age_weights_safe = jnp.maximum(age_weights_norm, 1e-30)
    ssp_flux_at_z = weighted_ssp / age_weights_safe[:, None]

    return age_weights_msun, ssp_flux_at_z


def compute_dsps_age_weights(
    sfr_on_ssp_ages: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    ssp_lg_age_gyr: jnp.ndarray,
    t_obs_gyr: float,
) -> jnp.ndarray:
    r"""DSPS-canonical age weights only (no metallicity dispatch).

    Produces the SFH→age weight tensor (Hearin+ 2021 Eq. 9) on the
    SSP age grid in absolute mass units (Msun per age bin), without
    doing the metallicity marginalization. Useful when the caller
    runs an independent metallicity dispatch (bilinear on a 4D
    α-grid, ramp, chem-evol, etc.) and only needs DSPS-canonical
    SFH integration.

    Implements the same negative-cosmic-time safety as
    :func:`compute_dsps_native_weights` (invalid SSP bins masked
    via ``T_TABLE_MIN`` ramp + zero SFR).

    Parameters
    ----------
    sfr_on_ssp_ages : array, shape (n_age,)
        Star formation rate (Msun/yr) at each SSP lookback age,
        sorted **ascending by age** (youngest = index 0).
    ssp_ages_yr : array, shape (n_age,)
        SSP lookback ages in years (ascending).
    ssp_lg_age_gyr : array, shape (n_age,)
        log10(age/Gyr) of SSP templates (DSPS convention).
    t_obs_gyr : float
        Age of the universe in Gyr at the observation redshift.

    Returns
    -------
    age_weights_msun : ndarray, shape (n_age,)
        Mass formed per SSP age bin (Msun), sorted ascending by age.
        Sum = total stellar mass formed.

    Notes
    -----
    **JIT-compatible**: yes. **Differentiable**: yes — pure JAX,
    no shape changes from inputs.

    References
    ----------
    .. [1] Hearin et al. 2021, "DSPS: Differentiable Stellar
       Population Synthesis", arXiv:2112.06830, Eq. 9.
    """
    try:
        from dsps.sed.ssp_weights import calc_age_weights_from_sfh_table
    except ImportError:
        raise ImportError(
            "dsps is required for DSPS-canonical age weights. Install with: pip install dsps"
        ) from None

    ssp_age_gyr = ssp_ages_yr / 1e9
    t_cosmic_raw = t_obs_gyr - ssp_age_gyr

    # NaN-safety: floor + invalid-bin ramp (see compute_dsps_native_weights).
    T_TABLE_MIN = 0.01  # Gyr; matches dsps.constants.T_TABLE_MIN
    n_ssp = ssp_ages_yr.shape[0]
    t_cosmic_floor = jnp.maximum(t_cosmic_raw, T_TABLE_MIN)
    valid = t_cosmic_raw > 0.0
    valid_asc = valid[::-1]
    t_cosmic_asc_raw = t_cosmic_floor[::-1]
    sfr_asc_raw = sfr_on_ssp_ages[::-1]
    n_invalid = jnp.sum(~valid_asc)
    idx = jnp.arange(n_ssp)
    is_invalid_pos = idx < n_invalid
    ramp = T_TABLE_MIN + (T_TABLE_MIN * 0.5) * (idx + 1) / jnp.maximum(n_invalid, 1)
    t_cosmic_asc = jnp.where(is_invalid_pos, ramp, t_cosmic_asc_raw)
    # Guarantee strictly-increasing knots: at high z boundary-valid bins can
    # clamp to T_TABLE_MIN below the invalid-bin ramp, which DSPS NaNs on. #683
    t_cosmic_asc = enforce_increasing_cosmic_time(t_cosmic_asc)
    sfr_asc = jnp.where(is_invalid_pos, 0.0, sfr_asc_raw)

    # DSPS canonical trapezoidal-in-cosmic-time SFH integration.
    age_weights_norm = calc_age_weights_from_sfh_table(
        gal_t_table=t_cosmic_asc,
        gal_sfr_table=sfr_asc,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
        t_obs=t_obs_gyr,
    )
    total_mass = jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9)
    return age_weights_norm * jnp.maximum(total_mass, 0.0)


def compute_dsps_met_table_weights(
    sfr_on_ssp_ages: jnp.ndarray,
    lgmet_on_ssp_ages: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    ssp_lg_age_gyr: jnp.ndarray,
    ssp_flux: jnp.ndarray,
    t_obs_gyr: float,
    lgmet_scatter: float = 0.2,
) -> tuple:
    """Compute CSP age weights and metallicity-marginalized SSP flux via DSPS.

    Uses a per-age metallicity table (time-evolving Z(t)).

    Selected via ``SEDModel(..., csp_integration="dsps_met_table")``.  Unlike
    :func:`compute_dsps_native_weights` which uses a single scalar ``lgmet``
    with a lognormal MDF, this function accepts a per-SSP-age metallicity
    array so each age bin can have its own metallicity and lognormal scatter
    (Hearin et al. 2023, Eq. 11).  This is the natural mode for models with
    an evolving chemical history (``_met_mode="ramp"``).

    For a constant-metallicity model, pass a uniform array
    ``jnp.full_like(ssp_ages_yr, log_z_abs)``; the result is numerically
    equivalent to :func:`compute_dsps_native_weights` but computed via the
    met-table DSPS path.

    Parameters
    ----------
    sfr_on_ssp_ages : array, shape (n_age,)
        Star formation rate (Msun/yr) at each SSP lookback age,
        sorted **ascending by age** (youngest = index 0).
    lgmet_on_ssp_ages : array, shape (n_age,)
        log10(Z) metallicity at each SSP lookback age (absolute, not Z/Zsun),
        sorted ascending by age (youngest = index 0).
    ssp_ages_yr : array, shape (n_age,)
        SSP lookback ages in years (ascending).
    ssp_lgmet : array, shape (n_met,)
        log10(Z) metallicity grid of the SSP library (absolute).
    ssp_lg_age_gyr : array, shape (n_age,)
        log10(age/Gyr) of SSP templates.
    ssp_flux : array, shape (n_met, n_age, n_wave)
        SSP spectra in Lsun/Hz/Msun.
    t_obs_gyr : float
        Age of the universe in Gyr at the observation redshift.
    lgmet_scatter : float, optional
        Gaussian scatter in log10(Z) per age bin (dex). Default 0.2 dex.

    Returns
    -------
    age_weights_msun : array, shape (n_age,)
        Mass formed per SSP age bin (Msun), ascending by age (youngest first).
    ssp_flux_at_z : array, shape (n_age, n_wave)
        SSP flux marginalized over the per-age metallicity distribution
        (Lsun/Hz/Msun), ascending by age.

    Notes
    -----
    DSPS returns ``lgmet_weights`` with shape ``(n_met, n_age)`` in ascending
    **cosmic** time order (oldest first).  We flip the age axis back with
    ``lgmet_weights[:, ::-1]`` before the ``"ma,maw->aw"`` einsum so the
    metallicity weights are correctly paired with tengri's youngest-first SSP
    convention.

    References
    ----------
    Hearin et al. 2023, arXiv:2112.08423, Eq. 11 (met-table kernel).

    """
    try:
        from dsps.sed.stellar_sed import calc_rest_sed_sfh_table_met_table
    except ImportError:
        raise ImportError(
            "dsps is required for csp_integration='dsps_met_table'. Install with: pip install dsps"
        ) from None

    # SSP ages are lookback times (young→old, ascending).
    # DSPS needs cosmic times sorted ascending (oldest first).
    ssp_age_gyr = ssp_ages_yr / 1e9
    t_cosmic_gyr = jnp.clip(t_obs_gyr - ssp_age_gyr, min=1e-3)

    t_cosmic_asc = t_cosmic_gyr[::-1]  # oldest first
    sfr_asc = sfr_on_ssp_ages[::-1]
    lgmet_asc = lgmet_on_ssp_ages[::-1]  # metallicity aligned with cosmic time

    result = calc_rest_sed_sfh_table_met_table(
        gal_t_table=t_cosmic_asc,
        gal_sfr_table=sfr_asc,
        gal_lgmet_table=lgmet_asc,
        gal_lgmet_scatter=lgmet_scatter,
        ssp_lgmet=ssp_lgmet,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
        ssp_flux=ssp_flux,
        t_obs=t_obs_gyr,
    )

    # See ``compute_dsps_native_weights`` for the rationale: use the
    # joint (n_met, n_age) ``result.weights`` directly. DSPS aligns its
    # weights' age axis with the SSP grid (lookback-time ascending) —
    # no axis flips required to dot with ``ssp_flux``.
    total_mass = jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9)

    joint = result.weights  # (n_met, n_age) sum=1
    age_weights_norm = joint.sum(axis=0)  # (n_age,) sum=1
    age_weights_msun = age_weights_norm * jnp.maximum(total_mass, 0.0)

    weighted_ssp = jnp.einsum("ma,maw->aw", joint, ssp_flux)  # (n_age, n_wave) per Msun_formed
    age_weights_safe = jnp.maximum(age_weights_norm, 1e-30)
    ssp_flux_at_z = weighted_ssp / age_weights_safe[:, None]

    return age_weights_msun, ssp_flux_at_z


def compute_csp_weights(
    sfr_on_ssp_ages: jnp.ndarray,
    ssp_ages_yr: jnp.ndarray,
    method: str = "trapz",
    _log_interp_matrix=None,
) -> jnp.ndarray:
    """Compute SFH weights (mass formed per SSP age bin).

    Returns the stellar mass formed in each age bin (Msun), NOT
    normalized to sum=1. This way the CSP SED = sum(w_i * SSP_i)
    is in Lsun/Hz (same as DSPS), not Lsun/Hz/Msun.

    The total stellar mass formed is sum(weights).

    Parameters
    ----------
    sfr_on_ssp_ages : array, shape (n_age,)
        Star formation rate at each SSP age [Msun/yr].
    ssp_ages_yr : array, shape (n_age,)
        SSP ages [yr], sorted ascending.
    method : {"trapz", "log_trapz", "log_interp"}
        Integration method. See :func:`csp_age_dt` for details.
        Default ``"trapz"`` is the DSPS-compatible linear-age trapezoid rule.
        ``"log_trapz"`` applies the log-age Jacobian.
        ``"log_interp"`` uses Johnson+2021 log-linear interpolation (matrix
        multiply); requires ``_log_interp_matrix`` to be supplied.
    _log_interp_matrix : array, shape (n_age, n_age), optional
        Precomputed weight matrix from :func:`csp_log_interp_matrix`.
        Required when ``method="log_interp"``.

    Returns
    -------
    array, shape (n_age,)
        Mass formed per age bin [Msun]. Sum = total mass formed.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    **Gradient-safe**: yes.

    """
    if method == "log_interp":
        if _log_interp_matrix is None:
            _log_interp_matrix = jnp.array(
                csp_log_interp_matrix(ssp_ages_yr), dtype=sfr_on_ssp_ages.dtype
            )
        return _log_interp_matrix @ sfr_on_ssp_ages
    dt = csp_age_dt(ssp_ages_yr, method)
    return sfr_on_ssp_ages * dt


# ── Alpha-element enhancement ─────────────────────────────────────

# Coefficient converting [alpha/Fe] to total metallicity offset.
# Alpha elements (O, Mg, Si, Ca, Ti) dominate the metal mass budget,
# so [Z/H]_eff ≈ [Fe/H] + A * [alpha/Fe] with A ~ 0.75.
# Reference: Thomas, Maraston & Bender 2003; Vazdekis et al. 2015.
_ALPHA_TO_Z_COEFF = 0.75

# Salaris relation coefficients (Salaris, Chieffi & Straniero 1993;
# Knowles et al. 2023 Eq. 2). This is a semi-empirical fit to detailed
# stellar interior models with different abundance mixtures.
_SALARIS_LINEAR = 0.66154
_SALARIS_QUADRATIC = 0.20465


def salaris_mh_from_feh(feh: float, alpha_fe: float) -> float:
    """Convert [Fe/H] + [α/Fe] to total metallicity [M/H].

    Uses the Salaris, Chieffi & Straniero (1993) relation as parameterized
    by Knowles et al. (2023) Eq. 2::

        [M/H] = [Fe/H] + 0.66154 × [α/Fe] + 0.20465 × [α/Fe]²

    At solar [α/Fe] = 0.0, [M/H] = [Fe/H] exactly.

    Parameters
    ----------
    feh : float
        Iron abundance [Fe/H] (relative to solar, dimensionless).
    alpha_fe : float
        Alpha-element enhancement [α/Fe] (relative to solar, dimensionless).

    Returns
    -------
    float
        Total metallicity [M/H] (relative to solar, dimensionless).

    Notes
    -----
    **JIT-compatible**: yes — pure arithmetic operations.
    **Gradient-safe**: yes.

    References
    ----------
    Salaris, Chieffi & Straniero 1993, ApJ, 414, 580.
    Knowles et al. 2023, Eq. 2.

    Examples
    --------
    >>> from tengri import salaris_mh_from_feh
    >>> round(float(salaris_mh_from_feh(feh=-0.5, alpha_fe=0.2)), 4)
    -0.3676

    """
    return feh + _SALARIS_LINEAR * alpha_fe + _SALARIS_QUADRATIC * alpha_fe**2


def salaris_feh_from_mh(mh: float, alpha_fe: float) -> float:
    """Convert total metallicity [M/H] + [α/Fe] to iron abundance [Fe/H].

    Inverse of the Salaris relation::

        [Fe/H] = [M/H] − 0.66154 × [α/Fe] − 0.20465 × [α/Fe]²

    At solar [α/Fe] = 0.0, [Fe/H] = [M/H] exactly.

    Parameters
    ----------
    mh : float
        Total metallicity [M/H] (relative to solar, dimensionless).
    alpha_fe : float
        Alpha-element enhancement [α/Fe] (relative to solar, dimensionless).

    Returns
    -------
    float
        Iron abundance [Fe/H] (relative to solar, dimensionless).

    Notes
    -----
    **JIT-compatible**: yes — pure arithmetic operations.
    **Gradient-safe**: yes.

    References
    ----------
    Salaris, Chieffi & Straniero 1993, ApJ, 414, 580 (inverse formula).

    Examples
    --------
    >>> from tengri import salaris_feh_from_mh
    >>> round(float(salaris_feh_from_mh(mh=-0.5, alpha_fe=0.2)), 4)
    -0.6324

    """
    return mh - _SALARIS_LINEAR * alpha_fe - _SALARIS_QUADRATIC * alpha_fe**2


@jax.jit
def effective_metallicity(log_z_fe: float, alpha_fe: float = 0.0) -> float:
    r"""Convert [Fe/H] + [alpha/Fe] to effective total metallicity.

    Approximates the effect of alpha-element enhancement on the SED as
    a shift in the total metallicity used for SSP interpolation. Used
    when SSP templates are computed at fixed solar abundance ratios and
    cannot vary [alpha/Fe] at runtime.

    Parameters
    ----------
    log_z_fe : float
        Iron abundance [Fe/H] (equivalently, log10(Z/Zsun) when
        ``alpha_fe = 0``). [dex]
    alpha_fe : float, optional
        Alpha-element enhancement [alpha/Fe] relative to solar.
        Default 0.0 (solar abundance ratios). [dex]

    Returns
    -------
    float
        Effective total metallicity log10(Z_eff/Zsun). Same units as
        ``log_z_fe``. [dex]

    Notes
    -----
    **JIT-compatible**: yes — pure arithmetic; decorated with ``@jax.jit``.
    **Gradient-safe**: yes.

    **Approximation** (Thomas, Maraston & Bender 2003 [1]_):
    only valid for SSP grids that lack an explicit [alpha/Fe] axis. When
    the grid does include one, prefer bilinear (Z, [alpha/Fe]) interpolation;
    use :func:`has_alpha_grid` to test the SSP container at construction.

    .. math::

        [Z/H]_{\mathrm{eff}} = [\mathrm{Fe}/\mathrm{H}]
        + 0.75 \, [\alpha/\mathrm{Fe}]

    The coefficient 0.75 (``_ALPHA_TO_Z_COEFF``) is the empirical
    enhancement-to-metallicity scaling adopted by the Vazdekis et al.
    2015 [2]_ MILES library for E-MILES alpha-enhanced SSPs.

    References
    ----------
    .. [1] Thomas, D., Maraston, C., Bender, R., 2003, MNRAS, 339, 897.
    .. [2] Vazdekis, A. et al., 2015, MNRAS, 449, 1177.

    Examples
    --------
    >>> from tengri import effective_metallicity
    >>> round(float(effective_metallicity(-0.5, alpha_fe=0.3)), 4)
    -0.275

    """
    return log_z_fe + _ALPHA_TO_Z_COEFF * alpha_fe


def has_alpha_grid(ssp_data: SSPData) -> bool:
    """Check if SSP data includes an [alpha/Fe] grid dimension.

    When True, ssp_flux has shape (n_met, n_alpha, n_age, n_wave) and
    proper bilinear (Z, [α/Fe]) interpolation should be used instead of
    the effective_metallicity approximation.

    Parameters
    ----------
    ssp_data : SSPData
        Loaded SSP template data.

    Returns
    -------
    bool
        True if ssp_alpha_fe is present and ssp_flux is 4D.

    Notes
    -----
    **JIT-compatible**: yes — pure shape checking and conditionals.
    **Gradient-safe**: yes.

    Examples
    --------
    >>> from tengri import has_alpha_grid, load_ssp_data
    >>> # ssp = load_ssp_data("data/ssp_BC03_Chabrier.h5")
    >>> # has_alpha_grid(ssp)  # True if file contains [alpha/Fe] axis
    >>> # False for standard BC03 grids (no alpha grid)

    """
    return ssp_data.ssp_alpha_fe is not None and ssp_data.ssp_flux.ndim == 4


@jax.jit
def interpolate_alpha_only(
    ssp_flux: jnp.ndarray,
    ssp_alpha_fe: jnp.ndarray,
    alpha_fe: float,
) -> jnp.ndarray:
    """Linear interpolation along the [α/Fe] axis only — 4D → 3D.

    Collapses the [α/Fe] dimension of a 4D SSP grid at a single target
    value, leaving the metallicity axis intact. The result feeds the
    standard 3D DSPS lognormal-MDF kernel, so the Z marginalization
    behaves identically to a no-α-grid run with the same met scatter.

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_alpha, n_age, n_wave)
        SSP flux on the full 4D (Z, [α/Fe]) grid. [Lsun/Hz/Msun]
    ssp_alpha_fe : array, shape (n_alpha,)
        [α/Fe] grid values, relative to solar.
    alpha_fe : float
        Target [α/Fe] (dimensionless, relative to solar). Clipped to
        the grid bounds before interpolation.

    Returns
    -------
    array, shape (n_met, n_age, n_wave)
        α-collapsed SSP flux, ready for the 3D MDF kernel.

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    **Gradient-safe**: yes — linear interpolation is differentiable.

    Complements :func:`interpolate_met_alpha`, which collapses both
    axes to a single (Z, [α/Fe]) point. Samplers that want lognormal
    MDF marginalization over metallicity should use this α-only path
    so the 4D and 3D code paths share the same Z kernel.
    """
    afe = jnp.clip(alpha_fe, ssp_alpha_fe[0], ssp_alpha_fe[-1])
    ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe, afe) - 1, 0, ssp_alpha_fe.shape[0] - 2)
    fa = (afe - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])
    return (1.0 - fa) * ssp_flux[:, ia] + fa * ssp_flux[:, ia + 1]


@jax.jit
def interpolate_met_alpha(
    ssp_flux: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    ssp_alpha_fe: jnp.ndarray,
    log_z: float,
    alpha_fe: float,
) -> jnp.ndarray:
    """Bilinear interpolation in (metallicity, [α/Fe]) for 4D SSP grids.

    This is the correct approach when alpha-enhanced SSP templates are
    available (e.g., sMILES, BPASS v2.3, α-MC).  It replaces the
    ``effective_metallicity()`` approximation, which is only valid when
    α-enhanced templates are NOT available.

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_alpha, n_age, n_wave)
        SSP flux [Lsun/Hz/Msun] on the full (Z, [α/Fe]) grid.
    ssp_lgmet : array, shape (n_met,)
        [Fe/H] iron abundance grid (log10 relative to solar, dimensionless).
        All source libraries must be converted to [Fe/H] at load time.
    ssp_alpha_fe : array, shape (n_alpha,)
        [α/Fe] grid values (relative to solar, dimensionless;
        e.g., [-0.2, 0.0, +0.2, +0.4, +0.6]).
    log_z : float
        Target [Fe/H] (iron abundance, log10 relative to solar).
    alpha_fe : float
        Target [α/Fe] (relative to solar, dimensionless).

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux [Lsun/Hz/Msun] at the target (Z, [α/Fe]).

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    **Gradient-safe**: yes — bilinear interpolation is differentiable.

    """
    # Metallicity index and fraction
    lz = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
    iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz) - 1, 0, len(ssp_lgmet) - 2)
    fz = (lz - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])

    # Alpha index and fraction
    afe = jnp.clip(alpha_fe, ssp_alpha_fe[0], ssp_alpha_fe[-1])
    ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe, afe) - 1, 0, len(ssp_alpha_fe) - 2)
    fa = (afe - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])

    # Bilinear: four corners → (n_age, n_wave)
    return (
        (1.0 - fz) * (1.0 - fa) * ssp_flux[iz, ia]
        + fz * (1.0 - fa) * ssp_flux[iz + 1, ia]
        + (1.0 - fz) * fa * ssp_flux[iz, ia + 1]
        + fz * fa * ssp_flux[iz + 1, ia + 1]
    )


@jax.jit
def interpolate_met_alpha_evolving(
    ssp_flux: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    ssp_alpha_fe: jnp.ndarray,
    log_z_per_age: jnp.ndarray,
    alpha_fe_per_age: jnp.ndarray,
) -> jnp.ndarray:
    """Per-age bilinear interpolation in (Z, [α/Fe]) for time-evolving abundances.

    Each SSP age bin can have a different metallicity AND a different
    [α/Fe], enabling physically motivated chemical evolution where old
    stars are α-enhanced and young stars are solar-scaled.

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_alpha, n_age, n_wave)
        SSP flux [Lsun/Hz/Msun] on the full (Z, [α/Fe]) grid.
    ssp_lgmet : array, shape (n_met,)
        [Fe/H] iron abundance grid (log10 relative to solar, dimensionless).
    ssp_alpha_fe : array, shape (n_alpha,)
        [α/Fe] grid values (relative to solar, dimensionless).
    log_z_per_age : array, shape (n_age,)
        Target [Fe/H] at each SSP age bin (dimensionless).
    alpha_fe_per_age : array, shape (n_age,)
        Target [α/Fe] at each SSP age bin (relative to solar, dimensionless).

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux [Lsun/Hz/Msun] with per-age (Z, [α/Fe]).

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.vmap`` for vectorized interpolation.
    **Gradient-safe**: yes.

    """

    def _interp_one_age(lz_i, afe_i, flux_at_age_i):
        """Bilinear interpolation over metallicity and alpha-element abundance."""
        # flux_at_age_i: (n_met, n_alpha, n_wave)
        lz = jnp.clip(lz_i, ssp_lgmet[0], ssp_lgmet[-1])
        iz = jnp.clip(jnp.searchsorted(ssp_lgmet, lz) - 1, 0, len(ssp_lgmet) - 2)
        fz = (lz - ssp_lgmet[iz]) / (ssp_lgmet[iz + 1] - ssp_lgmet[iz])

        afe = jnp.clip(afe_i, ssp_alpha_fe[0], ssp_alpha_fe[-1])
        ia = jnp.clip(jnp.searchsorted(ssp_alpha_fe, afe) - 1, 0, len(ssp_alpha_fe) - 2)
        fa = (afe - ssp_alpha_fe[ia]) / (ssp_alpha_fe[ia + 1] - ssp_alpha_fe[ia])

        return (
            (1.0 - fz) * (1.0 - fa) * flux_at_age_i[iz, ia]
            + fz * (1.0 - fa) * flux_at_age_i[iz + 1, ia]
            + (1.0 - fz) * fa * flux_at_age_i[iz, ia + 1]
            + fz * fa * flux_at_age_i[iz + 1, ia + 1]
        )

    # Transpose: (n_met, n_alpha, n_age, n_wave) → (n_age, n_met, n_alpha, n_wave)
    flux_by_age = jnp.transpose(ssp_flux, (2, 0, 1, 3))
    return jax.vmap(_interp_one_age)(log_z_per_age, alpha_fe_per_age, flux_by_age)


@jax.jit
def compute_alpha_fe_evolving(
    ssp_lg_age_gyr: jnp.ndarray,
    alpha_fe_old: float,
    alpha_fe_young: float,
    t_universe_gyr: float,
) -> jnp.ndarray:
    """Compute per-age [α/Fe] from a linear ramp in lookback time.

    Old stars (large lookback time) have high [α/Fe] (formed before
    Type Ia SNe enriched Fe).  Young stars have lower [α/Fe] (solar
    or sub-solar).  This is the standard chemical evolution prediction.

    The ramp is linear in lookback time::

        [α/Fe](t_lookback) = α_young + (α_old - α_young) * t_lookback / t_universe

    Parameters
    ----------
    ssp_lg_age_gyr : array, shape (n_age,)
        Log10(age [Gyr]) of SSP templates (= lookback time for SSP bins).
    alpha_fe_old : float
        [α/Fe] of the oldest stars (at t_lookback = t_universe, dimensionless).
        Typically +0.3 to +0.5 for massive ellipticals.
    alpha_fe_young : float
        [α/Fe] at present day (t_lookback ≈ 0, dimensionless).
        Typically ~0.0 (solar) for disk galaxies.
    t_universe_gyr : float
        Age of the universe at the observed redshift [Gyr].

    Returns
    -------
    array, shape (n_age,)
        [α/Fe] at each SSP age bin (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    **Gradient-safe**: yes.

    """
    age_gyr = 10.0**ssp_lg_age_gyr
    t_frac = jnp.clip(age_gyr / t_universe_gyr, 0.0, 1.0)
    return alpha_fe_young + (alpha_fe_old - alpha_fe_young) * t_frac


from tengri.utils.physics_constants import L_SUN as LSUN_ERG_PER_S


@jax.jit
def compute_csp_sed(
    weights: jnp.ndarray, ssp_flux_at_met: jnp.ndarray, dust_attenuation: jnp.ndarray
) -> jnp.ndarray:
    """Compute composite stellar population SED.

    SED = Lsun * sum_i (weight_i * dust_i * ssp_flux_i)

    where weights are in Msun (mass formed per bin) and SSP flux
    is in Lsun/Hz/Msun. The result is in erg/s/Hz.

    Parameters
    ----------
    weights : array, shape (n_age,)
        Mass formed per age bin [Msun] from :func:`compute_csp_weights`.
    ssp_flux_at_met : array, shape (n_age, n_wave)
        SSP spectra at fixed metallicity [Lsun/Hz/Msun].
    dust_attenuation : array, shape (n_age, n_wave)
        Multiplicative dust transmission per age and wavelength
        (dimensionless, in [0, 1]).

    Returns
    -------
    array, shape (n_wave,)
        Composite SED [erg/s/Hz] (rest-frame luminosity density).

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp.einsum`` for vectorized multiplication.
    **Gradient-safe**: yes.

    """
    # weights [Msun] * ssp [Lsun/Hz/Msun] * dust [dimensionless] -> Lsun/Hz
    sed_lsun = jnp.einsum("i,iw,iw->w", weights, dust_attenuation, ssp_flux_at_met)
    return sed_lsun * LSUN_ERG_PER_S  # -> erg/s/Hz


@jax.jit
def interpolate_metallicity(
    ssp_flux: jnp.ndarray, ssp_lgmet: jnp.ndarray, log_z: float
) -> jnp.ndarray:
    r"""Interpolate SSP flux to a target metallicity.

    Linear interpolation in log(Z/Zsun) space between the two
    nearest metallicity grid points.

    Parameters
    ----------
    ssp_flux : array_like, shape (n_met, n_age, n_wave)
        Full SSP flux grid. [Lsun/Hz/Msun]
    ssp_lgmet : array_like, shape (n_met,)
        log10(Z/Zsun) grid points. [dimensionless]
    log_z : float
        Target metallicity log10(Z/Zsun). Values outside the grid
        bounds are clamped to ``[ssp_lgmet[0], ssp_lgmet[-1]]``. [dimensionless]

    Returns
    -------
    ndarray, shape (n_age, n_wave)
        SSP flux interpolated to the target metallicity. [Lsun/Hz/Msun]

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    **Gradient-safe**: yes — linear interpolation is differentiable.

    **Approximation**: piecewise-linear interpolation in
    :math:`\log_{10}(Z/Z_\odot)`. Strictly valid only for grids whose
    flux varies smoothly with metallicity; for sharply varying lines or
    edges, prefer the smooth triweight kernel in
    :func:`compute_lgmet_weights`.

    Given bracketing grid indices :math:`i, i+1` with
    :math:`\log Z_i \le \log Z \le \log Z_{i+1}`,

    .. math::

        f = \frac{\log Z - \log Z_i}{\log Z_{i+1} - \log Z_i},
        \qquad
        F(\log Z) = (1 - f)\, F_i + f\, F_{i+1}

    where :math:`F_i \equiv` ``ssp_flux[i]`` [Lsun/Hz/Msun].

    """
    # Clamp to grid bounds
    log_z_clamped = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])

    # Find bracketing indices
    idx = jnp.searchsorted(ssp_lgmet, log_z_clamped) - 1
    idx = jnp.clip(idx, 0, len(ssp_lgmet) - 2)

    # Linear interpolation weight
    frac = (log_z_clamped - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])

    return (1.0 - frac) * ssp_flux[idx] + frac * ssp_flux[idx + 1]


# ── Smooth metallicity interpolation (triweight kernel, DSPS-compatible)

_LGMET_LO = -4.0
_LGMET_HI = 0.5


@jax.jit
def _tw_cuml_kern(x, m, h):
    """Triweight kernel CDF (same as DSPS _tw_cuml_kern).

    Cumulative distribution of the triweight kernel with support |z| < 3.
    Returns 0 for z < -3, 1 for z > 3, smooth polynomial between.
    """
    z = (x - m) / h
    val = -5.0 * z**7 / 69984.0 + 7.0 * z**5 / 2592.0 - 35.0 * z**3 / 864.0 + 35.0 * z / 96.0 + 0.5
    val = jnp.where(z < -3.0, 0.0, val)
    val = jnp.where(z > 3.0, 1.0, val)
    return val


@jax.jit
def _get_lgmet_bin_edges(grid, lo=_LGMET_LO, hi=_LGMET_HI):
    """Bin edges from midpoints, matching DSPS convention.

    Uses half-spacing on each side, with outer edges clamped.
    """
    edges = jnp.concatenate([jnp.array([lo]), 0.5 * (grid[:-1] + grid[1:]), jnp.array([hi])])
    return edges


@jax.jit
def compute_lgmet_weights(log_z, ssp_lgmet, lgmet_scatter=0.1):
    """Metallicity weights via triweight CDF integration (DSPS-compatible).

    Integrates the triweight kernel CDF between bin edges, exactly
    matching the DSPS ``triweighted_histogram`` approach. The kernel
    has support at |z| < 3σ, giving smooth multi-bin weights.

    Parameters
    ----------
    log_z : float
        Target log10(Z/Zsun) (dimensionless).
    ssp_lgmet : array, shape (n_met,)
        SSP metallicity grid [log10(Z/Zsun)], sorted ascending.
    lgmet_scatter : float
        Kernel bandwidth [dex]. DSPS default: 0.1.

    Returns
    -------
    array, shape (n_met,)
        Normalized weights summing to 1 (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp`` primitives and custom kernel CDF.
    **Gradient-safe**: yes.

    """
    edges = _get_lgmet_bin_edges(ssp_lgmet)
    # CDF difference: probability mass in each bin
    # Note: CDF(lo) - CDF(hi) gives the mass between lo and hi
    # because _tw_cuml_kern returns CDF of the flipped kernel.
    # DSPS convention: _tw_cuml_kern(x, lo, sig) - _tw_cuml_kern(x, hi, sig)
    # where x is the galaxy metallicity, lo/hi are bin edges.
    cdf_lo = _tw_cuml_kern(log_z, edges[:-1], lgmet_scatter)
    cdf_hi = _tw_cuml_kern(log_z, edges[1:], lgmet_scatter)
    raw = cdf_lo - cdf_hi

    total = jnp.sum(raw)
    nearest = jnp.argmin(jnp.abs(ssp_lgmet - log_z))
    fallback = jnp.zeros_like(raw).at[nearest].set(1.0)
    return jnp.where(total > 0, raw / total, fallback)


@jax.jit
def interpolate_metallicity_smooth(ssp_flux, ssp_lgmet, log_z, lgmet_scatter=0.1):
    """Interpolate SSP flux using triweight kernel over metallicity.

    C²-continuous gradients. Matches DSPS approach (Hearin+2023).

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_age, n_wave)
        Full SSP flux grid [Lsun/Hz/Msun].
    ssp_lgmet : array, shape (n_met,)
        SSP metallicity grid [log10(Z/Zsun)], sorted ascending.
    log_z : float
        Target log10(Z/Zsun).
    lgmet_scatter : float
        Kernel bandwidth [dex]. Default 0.1.

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux [Lsun/Hz/Msun].

    Notes
    -----
    **JIT-compatible**: yes — uses triweight kernel via :func:`compute_lgmet_weights`.
    **Gradient-safe**: yes — C²-continuous gradients.

    """
    w = compute_lgmet_weights(log_z, ssp_lgmet, lgmet_scatter)
    return jnp.einsum("m,maw->aw", w, ssp_flux)


@jax.jit
def interpolate_metallicity_smooth_evolving(ssp_flux, ssp_lgmet, log_z_per_age, lgmet_scatter=0.1):
    """Triweight metallicity interpolation with per-age Z.

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_age, n_wave)
        Full SSP flux grid [Lsun/Hz/Msun].
    ssp_lgmet : array, shape (n_met,)
        SSP metallicity grid [log10(Z/Zsun)], sorted ascending.
    log_z_per_age : array, shape (n_age,)
        Target log10(Z/Zsun) at each SSP age bin (dimensionless).
    lgmet_scatter : float
        Kernel bandwidth [dex]. Default 0.1.

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux [Lsun/Hz/Msun].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.vmap`` for per-age interpolation.
    **Gradient-safe**: yes — C²-continuous gradients.

    """

    def _one_age(log_z_i, flux_at_age_i):
        """Marginalize SSP flux over metallicity using triweight kernel."""
        w = compute_lgmet_weights(log_z_i, ssp_lgmet, lgmet_scatter)
        return jnp.einsum("m,mw->w", w, flux_at_age_i)

    flux_by_age = jnp.transpose(ssp_flux, (1, 0, 2))
    return jax.vmap(_one_age)(log_z_per_age, flux_by_age)


@jax.jit
def interpolate_mass_remaining_smooth(ssp_mass_remaining, ssp_lgmet, log_z, lgmet_scatter=0.1):
    """Smooth mass-remaining interpolation using triweight kernel.

    Interpolates the surviving mass fraction to a target metallicity
    using the same triweight kernel as :func:`interpolate_metallicity_smooth`.

    Parameters
    ----------
    ssp_mass_remaining : array, shape (n_met, n_age)
        Surviving mass fraction per metallicity and age (dimensionless, in [0, 1]).
    ssp_lgmet : array, shape (n_met,)
        SSP metallicity grid [log10(Z/Zsun)], sorted ascending.
    log_z : float
        Target log10(Z/Zsun).
    lgmet_scatter : float
        Kernel bandwidth [dex]. Default 0.1.

    Returns
    -------
    array, shape (n_age,)
        Interpolated mass-remaining fraction per age (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes — uses triweight kernel via :func:`compute_lgmet_weights`.
    **Gradient-safe**: yes — C²-continuous gradients.

    """
    w = compute_lgmet_weights(log_z, ssp_lgmet, lgmet_scatter)
    return jnp.einsum("m,ma->a", w, ssp_mass_remaining)


@jax.jit
def interpolate_metallicity_evolving(
    ssp_flux: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    log_z_per_age: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate SSP flux with a different metallicity per age bin.

    Each SSP age bin is interpolated at its own metallicity, enabling
    time-evolving metallicity models (e.g., chemical enrichment).

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_age, n_wave)
        Full SSP flux grid [Lsun/Hz/Msun].
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid (dimensionless), sorted ascending.
    log_z_per_age : array, shape (n_age,)
        Target log10(Z/Zsun) at each age bin (dimensionless).

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux [Lsun/Hz/Msun] with per-age metallicity.

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.vmap`` for vectorized interpolation.
    **Gradient-safe**: yes — linear interpolation is differentiable.

    """

    def _interp_one_age(log_z_i, ssp_flux_at_age_i):
        """Interpolate a single age bin at its metallicity.

        Parameters
        ----------
        log_z_i : scalar
            Target log10(Z/Zsun) for this age bin.
        ssp_flux_at_age_i : array, shape (n_met, n_wave)
            SSP flux at all metallicities for this age bin.

        Returns
        -------
        array, shape (n_wave,)
            Interpolated flux [Lsun/Hz/Msun].

        """
        log_z_c = jnp.clip(log_z_i, ssp_lgmet[0], ssp_lgmet[-1])
        idx = jnp.clip(
            jnp.searchsorted(ssp_lgmet, log_z_c) - 1,
            0,
            len(ssp_lgmet) - 2,
        )
        frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
        return (1.0 - frac) * ssp_flux_at_age_i[idx] + frac * ssp_flux_at_age_i[idx + 1]

    # ssp_flux is (n_met, n_age, n_wave); transpose to (n_age, n_met, n_wave)
    # so vmap over the leading (age) axis pairs each age with its metallicity
    ssp_flux_by_age = jnp.transpose(ssp_flux, (1, 0, 2))  # (n_age, n_met, n_wave)
    return jax.vmap(_interp_one_age)(log_z_per_age, ssp_flux_by_age)


@jax.jit
def interpolate_mass_remaining_evolving(
    ssp_mass_remaining: jnp.ndarray,
    ssp_lgmet: jnp.ndarray,
    log_z_per_age: jnp.ndarray,
) -> jnp.ndarray:
    """Interpolate mass-remaining with a different metallicity per age bin.

    Parameters
    ----------
    ssp_mass_remaining : array, shape (n_met, n_age)
        Surviving mass fraction per metallicity and age (dimensionless, in [0, 1]).
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid (dimensionless), sorted ascending.
    log_z_per_age : array, shape (n_age,)
        Target log10(Z/Zsun) at each age bin (dimensionless).

    Returns
    -------
    array, shape (n_age,)
        Interpolated mass-remaining fraction per age bin (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes — uses ``jax.vmap`` for vectorized interpolation.
    **Gradient-safe**: yes — linear interpolation is differentiable.

    """

    def _interp_one_age(log_z_i, mr_at_age_i):
        """Linear interpolation of mass-remaining at a single age bin."""
        log_z_c = jnp.clip(log_z_i, ssp_lgmet[0], ssp_lgmet[-1])
        idx = jnp.clip(
            jnp.searchsorted(ssp_lgmet, log_z_c) - 1,
            0,
            len(ssp_lgmet) - 2,
        )
        frac = (log_z_c - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
        return (1.0 - frac) * mr_at_age_i[idx] + frac * mr_at_age_i[idx + 1]

    # ssp_mass_remaining is (n_met, n_age); transpose to (n_age, n_met)
    mr_by_age = jnp.transpose(ssp_mass_remaining, (1, 0))  # (n_age, n_met)
    return jax.vmap(_interp_one_age)(log_z_per_age, mr_by_age)


@jax.jit
def compute_log_z_evolving(
    ssp_lg_age_gyr: jnp.ndarray,
    log_z_initial: float,
    log_z_final: float,
    t_universe_gyr: float,
) -> jnp.ndarray:
    """Compute per-age-bin metallicity from a linear-in-log ramp.

    The metallicity evolves linearly in log(Z/Zsun) space:

        log_z(t_lookback) = log_z_final + (log_z_initial - log_z_final)
                            * t_lookback / t_universe

    where t_lookback=0 is today (log_z_final) and t_lookback=t_universe
    is the oldest stars (log_z_initial). SSP ages are lookback times.

    Parameters
    ----------
    ssp_lg_age_gyr : array, shape (n_age,)
        Log10(age [Gyr]) of SSP templates (= lookback time for SSP bins).
    log_z_initial : float
        Metallicity of the oldest stars (at t_lookback = t_universe),
        in log10(Z/Zsun) (dimensionless).
    log_z_final : float
        Metallicity at present day (t_lookback = 0) [log10(Z/Zsun)].
    t_universe_gyr : float
        Age of the universe at the observed redshift [Gyr].

    Returns
    -------
    array, shape (n_age,)
        log10(Z/Zsun) at each SSP age bin (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    **Gradient-safe**: yes.

    """
    age_gyr = 10.0**ssp_lg_age_gyr
    # Clamp lookback time to [0, t_universe] so extrapolation is safe
    t_frac = jnp.clip(age_gyr / t_universe_gyr, 0.0, 1.0)
    return log_z_final + (log_z_initial - log_z_final) * t_frac


@jax.jit
def interpolate_mass_remaining(
    ssp_mass_remaining: jnp.ndarray, ssp_lgmet: jnp.ndarray, log_z: float
) -> jnp.ndarray:
    """Interpolate mass-remaining fraction to a target metallicity.

    Parameters
    ----------
    ssp_mass_remaining : array, shape (n_met, n_age)
        Surviving mass fraction per metallicity and age (dimensionless, in [0, 1]).
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid (dimensionless), sorted ascending.
    log_z : float
        Target log10(Z/Zsun).

    Returns
    -------
    array, shape (n_age,)
        Interpolated mass-remaining fraction per age (dimensionless).

    Notes
    -----
    **JIT-compatible**: yes — all operations use ``jnp`` primitives.
    **Gradient-safe**: yes — linear interpolation is differentiable.

    """
    log_z_clamped = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])
    idx = jnp.searchsorted(ssp_lgmet, log_z_clamped) - 1
    idx = jnp.clip(idx, 0, len(ssp_lgmet) - 2)
    frac = (log_z_clamped - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])
    return (1.0 - frac) * ssp_mass_remaining[idx] + frac * ssp_mass_remaining[idx + 1]


@jax.jit
def compute_surviving_mass(weights: jnp.ndarray, mass_remaining_at_met: jnp.ndarray) -> float:
    """Compute surviving stellar mass from CSP weights and mass-remaining.

    Parameters
    ----------
    weights : array, shape (n_age,)
        Mass formed per age bin [Msun] from :func:`compute_csp_weights`.
    mass_remaining_at_met : array, shape (n_age,)
        Fraction of formed mass surviving at each age (dimensionless, in [0, 1])
        from :func:`interpolate_mass_remaining`.

    Returns
    -------
    float
        Total surviving stellar mass [Msun].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp.sum`` for reduction.
    **Gradient-safe**: yes.

    """
    return jnp.sum(weights * mass_remaining_at_met)


def predict_surviving_mass(
    sfr: jnp.ndarray,
    t_lookback_yr: jnp.ndarray,
    ssp: "SSPData",
    log_z_zsun: float = 0.0,
) -> jnp.ndarray:
    """Predict surviving stellar mass directly from an SFR(t) array.

    Standalone helper for putting **priors on the surviving stellar mass**
    without running the full SED forward pass. Computes

    .. math::

        M_{\\star,{\\rm surv}} = \\int \\mathrm{SFR}(t_{\\rm lb}) \\,
                                  f_{\\rm surv}(t_{\\rm lb}; Z) \\, dt_{\\rm lb}

    where :math:`f_{\\rm surv}(t)` is the surviving mass fraction from the
    SSP's ``ssp_mass_remaining`` table, evaluated at the user-supplied
    metallicity via linear interpolation in :math:`\\log_{10}(Z/Z_\\odot)`
    and resampled onto the SFH lookback grid via linear interp in
    :math:`\\log_{10}({\\rm age})`.

    This formula follows from the post-2026-05-25 SFH normalization contract
    (``trapezoid(sfr, t) = 10**log_total_mass`` for every parametric SFH):
    the survivor fraction at each age becomes a *weight* on the SFR, and the
    integral collapses to a single trapezoid rule on the SFH grid — no DSPS
    convolution required.

    Parameters
    ----------
    sfr : array_like, shape (n_lb,)
        SFR on the lookback grid [Msun/yr]. Typically the output of one of
        the parametric SFH callables (e.g. ``tau(t, log_total_mass=10.0, …)``).
    t_lookback_yr : array_like, shape (n_lb,)
        Lookback time grid [yr], ascending.
    ssp : SSPData
        SSP container with populated ``ssp_mass_remaining`` (n_met, n_age)
        and ``ssp_lg_age_gyr``, ``ssp_lgmet`` axes. Raises if
        ``ssp_mass_remaining`` is ``None``.
    log_z_zsun : float, optional
        Metallicity at which to evaluate the surviving-mass fraction,
        :math:`\\log_{10}(Z/Z_\\odot)`. Default 0.0 (solar). Add
        :data:`tengri.utils.physics_constants.LOG10_ZSUN` if you have
        absolute :math:`\\log_{10}(Z)`.

    Returns
    -------
    jnp.ndarray (scalar)
        Surviving stellar mass [Msun].

    Notes
    -----
    **JIT-compatible**: yes — uses ``jnp.interp``, ``jnp.trapezoid``,
    and the existing :func:`interpolate_mass_remaining` helper.
    **Gradient-safe**: yes — differentiable w.r.t. any SFH parameter
    through ``sfr``.

    The result is also published as ``state.derived["log_mstar"]`` after
    every full forward pass through ``SEDModel.predict_*``. This helper
    is the cheap standalone path for the prior layer:

    .. code-block:: python

        # Put a Gaussian prior on log10(M*,surv)
        sfr = tau(t_lb, log_total_mass=lgM_formed, tau=tau_yr, age=age_yr)
        log_mstar_surv = jnp.log10(predict_surviving_mass(sfr, t_lb, ssp))
        log_prior = -0.5 * ((log_mstar_surv - 10.5) / 0.3) ** 2

    Raises
    ------
    ValueError
        If ``ssp.ssp_mass_remaining`` is ``None`` (e.g., legacy SSP file
        without a stored mass-remaining table).

    See Also
    --------
    compute_surviving_mass : low-level helper that takes pre-computed weights.
    interpolate_mass_remaining : metallicity interpolation used internally.

    """
    if ssp.ssp_mass_remaining is None:
        raise ValueError(
            "SSP has no ssp_mass_remaining table. Surviving mass cannot be "
            "computed without it. Load an SSP file with stellar-evolution "
            "mass-loss data (see tengri.load_ssp_data)."
        )
    # Mass-remaining fraction on the SSP age grid at the requested metallicity.
    f_surv_on_ssp = interpolate_mass_remaining(
        ssp.ssp_mass_remaining, ssp.ssp_lgmet, log_z_zsun
    )  # (n_age,)
    # Resample to the SFH lookback grid via linear interp in log10(age).
    # SSP ages are in log10(Gyr); SFH lookback is in linear yr. Convert both
    # to log10(yr) for the interp x-axis.
    ssp_log_age_yr = ssp.ssp_lg_age_gyr + 9.0
    log_lb_yr = jnp.log10(jnp.maximum(t_lookback_yr, 1.0))
    f_surv_on_lb = jnp.interp(log_lb_yr, ssp_log_age_yr, f_surv_on_ssp)
    # Clamp to [0, 1] — extrapolation outside the SSP age range can drift.
    f_surv_on_lb = jnp.clip(f_surv_on_lb, 0.0, 1.0)
    # Trapezoid: M_surv = ∫ SFR(t) f_surv(t) dt
    return jnp.trapezoid(sfr * f_surv_on_lb, t_lookback_yr)
