"""DSPS (Differentiable Stellar Population Synthesis) integration.

Wraps the DSPS CSP integral and SSP template loading. DSPS provides
the differentiable mapping from SFH weights → composite stellar
population spectrum, which is the core of the forward model.

References
----------
- Hearin et al. 2023 (arXiv:2112.08423): DSPS
- SSP templates: https://halos.as.arizona.edu/suchethacooray/ssp-spectra/
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class SSPData(NamedTuple):
    """Container for SSP template data.

    Attributes
    ----------
    ssp_wave : array, shape (n_wave,)
        Rest-frame wavelength grid (Angstrom).
    ssp_flux : array, shape (n_met, n_age, n_wave)
        SSP luminosity per unit mass (Lsun/Hz/Msun or similar).
    ssp_lg_age_gyr : array, shape (n_age,)
        Log10(age/Gyr) of SSP templates.
    ssp_lgmet : array, shape (n_met,)
        log10(Z) metallicity grid (absolute log10 metallicity, NOT log10(Z/Zsun)).
        Offset from solar: log10(Zsun) ≈ -1.848. See CLAUDE.md conventions.
    ssp_mass_remaining : array, shape (n_met, n_age), optional
        Fraction of formed mass still in living stars + remnants
        at each age and metallicity. Computed from stellar evolution
        tracks; depends on IMF and isochrone library. None if not
        available (surviving mass cannot be computed).
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


def load_ssp_data(filepath: str) -> SSPData:
    """Load SSP templates from DSPS-compatible HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file with fields: ssp_wave, ssp_flux,
        ssp_lg_age_gyr, ssp_lgmet.

    Returns
    -------
    SSPData
        Loaded SSP template data.
    """
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py required for SSP loading: pip install h5py") from None

    with h5py.File(filepath, "r") as f:
        mass_remaining = None
        if "ssp_mass_remaining" in f:
            mass_remaining = jnp.array(f["ssp_mass_remaining"][:])

        alpha_fe = None
        if "ssp_alpha_fe" in f:
            alpha_fe = jnp.array(f["ssp_alpha_fe"][:])

        return SSPData(
            ssp_wave=jnp.array(f["ssp_wave"][:]),
            ssp_flux=jnp.array(f["ssp_flux"][:]),
            ssp_lg_age_gyr=jnp.array(f["ssp_lg_age_gyr"][:]),
            ssp_lgmet=jnp.array(f["ssp_lgmet"][:]),
            ssp_mass_remaining=mass_remaining,
            ssp_alpha_fe=alpha_fe,
        )


def load_ssp_data_dsps(filepath: str) -> SSPData:
    """Load SSP templates using DSPS native loader.

    Falls back to load_ssp_data() if DSPS is not installed.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.

    Returns
    -------
    SSPData
        Loaded SSP template data.
    """
    try:
        from dsps import load_ssp_templates

        ssp_data = load_ssp_templates(fn=filepath)
        return SSPData(
            ssp_wave=jnp.array(ssp_data.ssp_wave),
            ssp_flux=jnp.array(ssp_data.ssp_flux),
            ssp_lg_age_gyr=jnp.array(ssp_data.ssp_lg_age_gyr),
            ssp_lgmet=jnp.array(ssp_data.ssp_lgmet),
        )
    except ImportError:
        return load_ssp_data(filepath)


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
        SSP ages in years, sorted ascending.
    method : {"trapz", "log_trapz"}
        Integration scheme. Default ``"trapz"`` matches DSPS.

    Returns
    -------
    array, shape (n_age,)
        Effective linear-age bin widths (years). Multiply by SFR (Msun/yr)
        to get mass formed per bin (Msun).
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
        SSP ages in years, sorted ascending.
    n_gl : int, optional
        Number of Gauss-Legendre quadrature points per interval. Default 5
        (exact for degree-9 polynomials; more than sufficient).

    Returns
    -------
    ndarray, shape (n_age, n_age)
        Weight matrix A. Use as ``weights = A @ sfr_on_ssp``.
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
    selected via ``Model(..., csp_integration="dsps_native")``.

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
        (Lsun/Hz/Msun).  Replaces the output of
        ``interp_met_alpha_dispatch``.

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
    t_cosmic_gyr = jnp.clip(t_obs_gyr - ssp_age_gyr, min=1e-3)

    # Flip to ascending cosmic time (oldest universe-age first).
    t_cosmic_asc = t_cosmic_gyr[::-1]
    sfr_asc = sfr_on_ssp_ages[::-1]

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

    # result.age_weights is normalized (sum=1, DSPS convention).
    # Scale to absolute mass (Msun) using trapezoidal integral of SFR dt.
    total_mass = jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9)
    # age_weights from DSPS are in reversed (ascending cosmic time) order.
    # Flip back to tengri's ascending-age convention.
    age_weights_msun = result.age_weights[::-1] * jnp.maximum(total_mass, 0.0)

    # Metallicity-marginalized SSP flux per age bin.
    # result.lgmet_weights shape: (n_met,) — fractional weights summing to 1.
    lgmet_w = result.lgmet_weights  # (n_met,)
    lgmet_w_safe = lgmet_w / jnp.maximum(lgmet_w.sum(), 1e-30)
    # Broadcast over age axis: ssp_flux shape is (n_met, n_age, n_wave).
    ssp_flux_at_z = jnp.einsum("m,maw->aw", lgmet_w_safe, ssp_flux)  # (n_age, n_wave)

    return age_weights_msun, ssp_flux_at_z


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
    """Compute CSP age weights and metallicity-marginalized SSP flux via DSPS
    with a per-age metallicity table (time-evolving Z(t)).

    Selected via ``Model(..., csp_integration="dsps_met_table")``.  Unlike
    :func:`compute_dsps_native_weights` which uses a single scalar ``lgmet``
    with a lognormal MDF, this function accepts a per-SSP-age metallicity
    array so each age bin can have its own metallicity and lognormal scatter
    (Hearin et al. 2023, Eq. 11).  This is the natural mode for models with
    an evolving chemical history (``_evolving_metallicity=True``).

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

    # Scale normalized age weights to absolute mass (Msun).
    total_mass = jnp.trapezoid(sfr_asc, t_cosmic_asc * 1e9)
    # result.age_weights is in ascending cosmic time (oldest first) → flip back.
    age_weights_msun = result.age_weights[::-1] * jnp.maximum(total_mass, 0.0)

    # lgmet_weights: (n_met, n_age) in ascending cosmic time (oldest first).
    # Flip age axis → youngest first, matching tengri's ssp_flux axis order.
    lgmet_w = result.lgmet_weights[:, ::-1]  # (n_met, n_age)
    lgmet_w_safe = lgmet_w / jnp.maximum(jnp.sum(lgmet_w, axis=0, keepdims=True), 1e-30)
    # Per-age metallicity-marginalized SSP flux.
    ssp_flux_at_z = jnp.einsum("ma,maw->aw", lgmet_w_safe, ssp_flux)  # (n_age, n_wave)

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
        Star formation rate at each SSP age (Msun/yr).
    ssp_ages_yr : array, shape (n_age,)
        SSP ages in years.
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
        Mass formed per age bin (Msun). Sum = total mass formed.
    """
    if method == "log_interp":
        if _log_interp_matrix is None:
            _log_interp_matrix = jnp.array(
                csp_log_interp_matrix(ssp_ages_yr), dtype=sfr_on_ssp_ages.dtype
            )
        return _log_interp_matrix @ sfr_on_ssp_ages
    dt = csp_age_dt(ssp_ages_yr, method)
    return sfr_on_ssp_ages * dt


# ---------------------------------------------------------------------------
# Alpha-element enhancement
# ---------------------------------------------------------------------------

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
        Iron abundance [Fe/H] (dex, relative to solar).
    alpha_fe : float
        Alpha-element enhancement [α/Fe] (dex).

    Returns
    -------
    float
        Total metallicity [M/H] (dex, relative to solar).
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
        Total metallicity [M/H] (dex, relative to solar).
    alpha_fe : float
        Alpha-element enhancement [α/Fe] (dex).

    Returns
    -------
    float
        Iron abundance [Fe/H] (dex, relative to solar).
    """
    return mh - _SALARIS_LINEAR * alpha_fe - _SALARIS_QUADRATIC * alpha_fe**2


@jax.jit
def effective_metallicity(log_z_fe: float, alpha_fe: float = 0.0) -> float:
    """Convert [Fe/H] + [alpha/Fe] to effective total metallicity.

    Approximates the effect of alpha-element enhancement on the SED
    as a shift in the total metallicity used for SSP interpolation:

        [Z/H]_eff = [Fe/H] + 0.75 * [alpha/Fe]

    This is the standard approach when SSP templates are computed at
    fixed abundance ratios and cannot be changed at runtime.

    Parameters
    ----------
    log_z_fe : float
        Iron abundance [Fe/H] (or equivalently, log10(Z) when
        [alpha/Fe] = 0, i.e. the existing ``log_z`` parameter).
    alpha_fe : float, optional
        Alpha-element enhancement [alpha/Fe] in dex.
        Default is 0.0 (solar abundance ratios).

    Returns
    -------
    float
        Effective total metallicity log10(Z_eff) in the same
        units as ``log_z_fe``.

    References
    ----------
    Thomas, Maraston & Bender 2003, MNRAS 339, 897
    Vazdekis et al. 2015, MNRAS 449, 1177
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
    """
    return ssp_data.ssp_alpha_fe is not None and ssp_data.ssp_flux.ndim == 4


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
        SSP flux on the full (Z, [α/Fe]) grid.
    ssp_lgmet : array, shape (n_met,)
        [Fe/H] iron abundance grid (log10 relative to solar).
        All source libraries must be converted to [Fe/H] at load time.
    ssp_alpha_fe : array, shape (n_alpha,)
        [α/Fe] grid values (e.g., [-0.2, 0.0, +0.2, +0.4, +0.6]).
    log_z : float
        Target [Fe/H] (iron abundance, log10 relative to solar).
    alpha_fe : float
        Target [α/Fe] in dex.

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux at the target (Z, [α/Fe]).
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
        SSP flux on the full (Z, [α/Fe]) grid.
    ssp_lgmet : array, shape (n_met,)
        [Fe/H] iron abundance grid (log10 relative to solar).
    ssp_alpha_fe : array, shape (n_alpha,)
        [α/Fe] grid values.
    log_z_per_age : array, shape (n_age,)
        Target [M/H] at each SSP age bin.
    alpha_fe_per_age : array, shape (n_age,)
        Target [α/Fe] at each SSP age bin.

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux with per-age (Z, [α/Fe]).
    """

    def _interp_one_age(lz_i, afe_i, flux_at_age_i):
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
        Log10(age/Gyr) of SSP templates (= lookback time for SSP bins).
    alpha_fe_old : float
        [α/Fe] of the oldest stars (at t_lookback = t_universe).
        Typically +0.3 to +0.5 for massive ellipticals.
    alpha_fe_young : float
        [α/Fe] at present day (t_lookback ≈ 0).
        Typically ~0.0 (solar) for disk galaxies.
    t_universe_gyr : float
        Age of the universe at the observed redshift (Gyr).

    Returns
    -------
    array, shape (n_age,)
        [α/Fe] at each SSP age bin.
    """
    age_gyr = 10.0**ssp_lg_age_gyr
    t_frac = jnp.clip(age_gyr / t_universe_gyr, 0.0, 1.0)
    return alpha_fe_young + (alpha_fe_old - alpha_fe_young) * t_frac


LSUN_ERG_PER_S = 3.828e33  # erg/s (IAU 2015)


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
        Mass formed per age bin (Msun) from compute_csp_weights.
    ssp_flux_at_met : array, shape (n_age, n_wave)
        SSP spectra at fixed metallicity (Lsun/Hz/Msun).
    dust_attenuation : array, shape (n_age, n_wave)
        Multiplicative dust transmission per age and wavelength.

    Returns
    -------
    array, shape (n_wave,)
        Composite SED in erg/s/Hz (rest-frame luminosity density).
    """
    # weights [Msun] * ssp [Lsun/Hz/Msun] * dust [dimensionless] -> Lsun/Hz
    sed_lsun = jnp.einsum("i,iw,iw->w", weights, dust_attenuation, ssp_flux_at_met)
    return sed_lsun * LSUN_ERG_PER_S  # -> erg/s/Hz


@jax.jit
def interpolate_metallicity(
    ssp_flux: jnp.ndarray, ssp_lgmet: jnp.ndarray, log_z: float
) -> jnp.ndarray:
    """Interpolate SSP flux to a target metallicity.

    Linear interpolation in log(Z/Zsun) space between the two
    nearest metallicity grid points.

    Parameters
    ----------
    ssp_flux : array, shape (n_met, n_age, n_wave)
        Full SSP flux grid.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid.
    log_z : float
        Target log10(Z/Zsun).

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux.
    """
    # Clamp to grid bounds
    log_z_clamped = jnp.clip(log_z, ssp_lgmet[0], ssp_lgmet[-1])

    # Find bracketing indices
    idx = jnp.searchsorted(ssp_lgmet, log_z_clamped) - 1
    idx = jnp.clip(idx, 0, len(ssp_lgmet) - 2)

    # Linear interpolation weight
    frac = (log_z_clamped - ssp_lgmet[idx]) / (ssp_lgmet[idx + 1] - ssp_lgmet[idx])

    return (1.0 - frac) * ssp_flux[idx] + frac * ssp_flux[idx + 1]


# ---------------------------------------------------------------------------
# Smooth metallicity interpolation (triweight kernel, DSPS-compatible)
# ---------------------------------------------------------------------------

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
        Target log10(Z/Zsun).
    ssp_lgmet : array (n_met,)
        SSP metallicity grid.
    lgmet_scatter : float
        Kernel bandwidth in dex (DSPS default: 0.1).

    Returns
    -------
    array (n_met,) — normalized weights summing to 1.
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

    C2-continuous gradients. Matches DSPS approach (Hearin+2023).

    Parameters
    ----------
    ssp_flux : array (n_met, n_age, n_wave)
    ssp_lgmet : array (n_met,)
    log_z : float — target log10(Z/Zsun)
    lgmet_scatter : float — kernel bandwidth in dex

    Returns
    -------
    array (n_age, n_wave)
    """
    w = compute_lgmet_weights(log_z, ssp_lgmet, lgmet_scatter)
    return jnp.einsum("m,maw->aw", w, ssp_flux)


@jax.jit
def interpolate_metallicity_smooth_evolving(ssp_flux, ssp_lgmet, log_z_per_age, lgmet_scatter=0.1):
    """Triweight metallicity interpolation with per-age Z.

    Parameters
    ----------
    ssp_flux : array (n_met, n_age, n_wave)
    ssp_lgmet : array (n_met,)
    log_z_per_age : array (n_age,) — per-bin log10(Z/Zsun)
    lgmet_scatter : float

    Returns
    -------
    array (n_age, n_wave)
    """

    def _one_age(log_z_i, flux_at_age_i):
        w = compute_lgmet_weights(log_z_i, ssp_lgmet, lgmet_scatter)
        return jnp.einsum("m,mw->w", w, flux_at_age_i)

    flux_by_age = jnp.transpose(ssp_flux, (1, 0, 2))
    return jax.vmap(_one_age)(log_z_per_age, flux_by_age)


@jax.jit
def interpolate_mass_remaining_smooth(ssp_mass_remaining, ssp_lgmet, log_z, lgmet_scatter=0.1):
    """Smooth mass-remaining interpolation using triweight kernel."""
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
        Full SSP flux grid.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid.
    log_z_per_age : array, shape (n_age,)
        Target log10(Z/Zsun) at each age bin.

    Returns
    -------
    array, shape (n_age, n_wave)
        Interpolated SSP flux with per-age metallicity.
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
            Interpolated flux.
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
        Fraction of formed mass surviving at each age and metallicity.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid.
    log_z_per_age : array, shape (n_age,)
        Target log10(Z/Zsun) at each age bin.

    Returns
    -------
    array, shape (n_age,)
        Interpolated mass-remaining fraction per age bin.
    """

    def _interp_one_age(log_z_i, mr_at_age_i):
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
        Log10(age/Gyr) of SSP templates.
    log_z_initial : float
        Metallicity of the oldest stars (at t_lookback = t_universe),
        in log10(Z/Zsun) internally (absolute log10(Z)).
    log_z_final : float
        Metallicity at present day (t_lookback = 0), in log10(Z).
    t_universe_gyr : float
        Age of the universe at the observed redshift (Gyr).

    Returns
    -------
    array, shape (n_age,)
        log10(Z) at each SSP age bin.
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
        Fraction of formed mass surviving at each age and metallicity.
    ssp_lgmet : array, shape (n_met,)
        Log10(Z/Zsun) grid.
    log_z : float
        Target log10(Z/Zsun).

    Returns
    -------
    array, shape (n_age,)
        Interpolated mass-remaining fraction.
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
        Mass formed per age bin (Msun) from compute_csp_weights.
    mass_remaining_at_met : array, shape (n_age,)
        Fraction of formed mass surviving at each age (from
        interpolate_mass_remaining).

    Returns
    -------
    float
        Total surviving stellar mass (Msun).
    """
    return jnp.sum(weights * mass_remaining_at_met)
