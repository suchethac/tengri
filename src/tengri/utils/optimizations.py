"""Performance and memory optimizations for JAX-based SED fitting.

Implements tricks from NIFTy.re (Edenhofer+2024) and Zacharegkas+2025
to reduce memory usage and speed up computation.

Key optimizations:
1. Hartley transform (real-to-real FFT, avoids complex arrays)
2. Gradient checkpointing for the forward model
3. Approximate photometry (pre-computed SSP broadband fluxes)
4. Memory-efficient vmap patterns
"""

import jax
import jax.numpy as jnp

from tengri.utils.conversions import lnu_to_fnu

# ── 1. Hartley transform (from NIFTy.re) ──────────────────────────


@jax.jit
def hartley(x: jnp.ndarray) -> jnp.ndarray:
    """Hartley transform: real-to-real alternative to FFT.

    H(x) = Re(FFT(x)) - Im(FFT(x))

    The Hartley transform is its own inverse (self-reciprocal) and
    operates entirely in real space, halving memory compared to
    complex FFT. Used by NIFTy.re for correlated field generation.

    Parameters
    ----------
    x : array, shape (N,)
        Real-valued input.

    Returns
    -------
    array, shape (N,)
        Real-valued Hartley transform.
    """
    tmp = jnp.fft.fft(x)
    return tmp.real - tmp.imag


@jax.jit
def inverse_hartley(x: jnp.ndarray) -> jnp.ndarray:
    """Inverse Hartley transform (same as forward, with 1/N scaling)."""
    return hartley(x) / x.shape[0]


@jax.jit
def gp_from_xi_hartley(xi: jnp.ndarray, amplitude: jnp.ndarray) -> jnp.ndarray:
    """Generate GP realization using Hartley transform (memory-efficient).

    Unlike the rfft-based version, this avoids complex arrays entirely.
    The amplitude array must be in the full Fourier space (N values),
    not just the positive frequencies.

    The correct sequence: Hartley-transform xi to frequency domain,
    multiply by amplitude, then inverse-Hartley back. This ensures
    the GP has the correct variance (matching the rfft-based version).

    Parameters
    ----------
    xi : array, shape (N,)
        Standardized latent vector.
    amplitude : array, shape (N,)
        Full-space amplitude operator (sqrt of PSD, properly normalized).

    Returns
    -------
    array, shape (N,)
        GP realization.
    """
    return inverse_hartley(amplitude * hartley(xi))


def compute_full_amplitude_drw(
    n_points: int, d_log_age: float, sigma_ps: float, tau_ps: float, log_age_ref: float = 8.0
) -> jnp.ndarray:
    """Compute amplitude operator in full Fourier space for Hartley transform.

    Parameters
    ----------
    n_points : int
        Grid size.
    d_log_age : float
        Grid spacing in dex.
    sigma_ps, tau_ps : float
        DRW PSD parameters.
    log_age_ref : float
        Reference log-age for Jacobian correction.

    Returns
    -------
    array, shape (n_points,)
        Full-space amplitude operator.
    """
    from tengri.components.sfh.psd_models import psd_drw

    t_ref = 10.0**log_age_ref
    ln10 = jnp.log(10.0)

    # Full FFT frequencies
    freqs = jnp.fft.fftfreq(n_points, d=d_log_age)
    q = 2.0 * jnp.pi * freqs
    omega_phys = q / (t_ref * ln10)

    p_phys = psd_drw(jnp.abs(omega_phys), sigma_ps, tau_ps)
    p_logage = p_phys / (t_ref * ln10)

    return jnp.sqrt(jnp.maximum(p_logage, 1e-30) / d_log_age)


# ── 2. Gradient checkpointing ─────────────────────────────────────


def checkpointed_forward_model(model_fn):
    """Wrap a forward model with gradient checkpointing.

    Trades compute for memory: intermediate activations are recomputed
    during the backward pass instead of stored. Critical for fitting
    many galaxies in parallel via vmap.

    Usage:
        model_fn = checkpointed_forward_model(model.__call__)
        # Gradients now use O(1) memory per forward-model layer
        grad_fn = jax.grad(lambda p: loss(model_fn(p), data))

    Parameters
    ----------
    model_fn : callable
        Forward model function: params -> predictions.

    Returns
    -------
    callable
        Checkpointed version with same signature.
    """
    return jax.checkpoint(model_fn)


# ── 3. Approximate photometry (Zacharegkas+2025 Section 3) ────────


@jax.jit
def precompute_ssp_photometry(
    ssp_flux: jnp.ndarray,
    ssp_wave: jnp.ndarray,
    filter_wave: jnp.ndarray,
    filter_trans: jnp.ndarray,
    redshift: float,
) -> jnp.ndarray:
    """Pre-compute SSP broadband fluxes for approximate photometry.

    c_SSP(tau_age, Z) = int T(lambda|z) * L_SSP(lambda|tau_age, Z) * lambda dlambda
                        / int T(lambda|z) * lambda dlambda

    This is Equation 7 of Zacharegkas+2025. Once computed, galaxy
    photometry is just a weighted sum over this grid, eliminating
    the expensive wavelength integral from the MCMC loop.

    Parameters
    ----------
    ssp_flux : array, shape (n_age, n_wave)
        SSP spectra at fixed metallicity.
    ssp_wave : array, shape (n_wave,)
        Rest-frame wavelength grid (Angstrom).
    filter_wave : array, shape (n_filt,)
        Filter wavelength grid (observed frame).
    filter_trans : array, shape (n_filt,)
        Filter transmission.
    redshift : float
        Source redshift.

    Returns
    -------
    array, shape (n_age,)
        Pre-computed SSP broadband flux per age bin through this filter.
    """
    # Redshift SSP wavelengths to observed frame
    wave_obs = ssp_wave * (1.0 + redshift)

    # Denominator: int T(lambda) * lambda dlambda
    denom = jnp.trapezoid(filter_trans * filter_wave, filter_wave)

    def _single_age(ssp_spectrum):
        """Compute broadband SSP flux through a single filter."""
        # Interpolate SSP onto filter wavelength grid
        sed_on_filter = jnp.interp(filter_wave, wave_obs, ssp_spectrum, left=0.0, right=0.0)
        num = jnp.trapezoid(sed_on_filter * filter_trans * filter_wave, filter_wave)
        return num / jnp.maximum(denom, 1e-30)

    return jax.vmap(_single_age)(ssp_flux)


@jax.jit
def effective_wavelength(filter_wave: jnp.ndarray, filter_trans: jnp.ndarray) -> float:
    """Effective wavelength of a filter: lambda_eff = int(T*lam^2 dlam) / int(T*lam dlam).

    Zacharegkas+2025 Equation 5.
    """
    num = jnp.trapezoid(filter_trans * filter_wave**2, filter_wave)
    denom = jnp.trapezoid(filter_trans * filter_wave, filter_wave)
    return num / jnp.maximum(denom, 1e-30)


@jax.jit
def approximate_photometry(
    weights: jnp.ndarray,
    ssp_phot: jnp.ndarray,
    dust_atten_at_eff: jnp.ndarray,
    dl_cm: float,
    redshift: float,
) -> float:
    """Fast approximate photometry using pre-computed SSP fluxes.

    c_gal ≈ F_att(lambda_eff) * sum_i P_SFH(tau_i) * c_SSP(tau_i)

    Pulls dust attenuation outside the integral (evaluated at effective
    wavelength). Error <0.1% for broadband filters (Zacharegkas+2025).

    Parameters
    ----------
    weights : array, shape (n_age,)
        Normalized SFH weights.
    ssp_phot : array, shape (n_age,)
        Pre-computed SSP broadband fluxes (from precompute_ssp_photometry).
    dust_atten_at_eff : array, shape (n_age,)
        Dust transmission at effective wavelength per age bin.
    dl_cm : float
        Luminosity distance (cm).
    redshift : float
        Source redshift.

    Returns
    -------
    float
        Approximate flux density (erg/s/cm^2/Hz).
    """
    # Weighted sum with dust
    flux_intrinsic = jnp.sum(weights * ssp_phot * dust_atten_at_eff)
    flux_scale = lnu_to_fnu(1.0, dl_cm, redshift)
    return flux_scale * flux_intrinsic


# ── 4. Memory-efficient vmap patterns ─────────────────────────────


def batched_forward(model_fn, params_batch, batch_size=100):
    """Process galaxies in memory-efficient batches.

    For large catalogs, vmapping over all galaxies at once can exhaust
    GPU memory. This processes them in chunks.

    Parameters
    ----------
    model_fn : callable
        Forward model for a single galaxy.
    params_batch : dict of arrays
        Parameters with leading batch dimension.
    batch_size : int
        Number of galaxies per chunk.

    Returns
    -------
    array
        Concatenated results.
    """
    n_total = jax.tree.leaves(params_batch)[0].shape[0]
    results = []

    for start in range(0, n_total, batch_size):
        end = min(start + batch_size, n_total)
        chunk = jax.tree.map(lambda x, _s=start, _e=end: x[_s:_e], params_batch)
        result = jax.vmap(model_fn)(chunk)
        results.append(result)

    return jnp.concatenate(results, axis=0)
