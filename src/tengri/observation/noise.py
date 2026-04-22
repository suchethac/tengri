"""Noise models for SED fitting likelihoods.

Following NIFTy Information Field Theory (IFT) principles, noise parameters
are part of the generative model and jointly inferred with the signal.

The data model is: d = R(s) + n, where n ~ N(0, N) and N is the noise
covariance. When the noise covariance depends on model parameters, we use
NIFTy's ``VariableCovarianceGaussian`` likelihood:

    E = ½ ‖(d - f) · τ‖² - Σ log(τ)

where τ = 1/σ_eff is the inverse effective noise standard deviation, and
the -Σlog(τ) = +Σlog(σ_eff) term is the log-determinant that prevents the
trivial solution σ → ∞.

The effective noise combines observational uncertainties with a fractional
calibration floor:

    σ²_eff,k = σ²_obs,k + (f_cal · m_k)²

where f_cal is a free parameter inferred from data (typically 1-15%).

References
----------
- Enßlin et al. (2009): Information field theory
- Knollmüller & Enßlin (2019): Encoding prior knowledge in the structure of
  the likelihood
- Johnson et al. (2021): Prospector noise model
- Alsing et al. (2022): Hierarchical noise model for photometric SED fitting

Usage
-----
When ``noise_frac_cal`` is ``Fixed(0.0)`` (default), the noise model is
inactive and the code uses the standard ``jft.Gaussian`` likelihood. When
``noise_frac_cal`` is a free parameter (e.g., ``Uniform(0.01, 0.2)``), the
code switches to ``jft.VariableCovarianceGaussian``.

Example::

    spec = Parameters(
        noise_frac_cal=Uniform(0.01, 0.2),  # enable noise model
        sfh_dpl_alpha=Uniform(0.5, 4.0),
        ...
    )

"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# ── Core noise computation (pure JAX, JIT-compatible) ─────────────


def compute_effective_noise(
    noise_obs: jnp.ndarray,
    model_flux: jnp.ndarray,
    f_cal: float | jnp.ndarray,
) -> jnp.ndarray:
    """Compute effective noise with calibration floor.

    σ_eff = sqrt(σ²_obs + (f_cal · model)²)

    Parameters
    ----------
    noise_obs : array, shape (n_bands,)
        Observed 1-sigma uncertainties.
    model_flux : array, shape (n_bands,)
        Model-predicted fluxes (absolute value used for calibration term).
    f_cal : float or scalar array
        Fractional calibration uncertainty. Typical range: 0.01-0.15.

    Returns
    -------
    array, shape (n_bands,)
        Effective noise standard deviation.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import compute_effective_noise
    >>> noise = jnp.array([0.1, 0.2, 0.15])
    >>> model = jnp.array([1.0, 2.0, 1.5])
    >>> sigma_eff = compute_effective_noise(noise, model, f_cal=0.05)
    >>> sigma_eff.shape
    (3,)
    """
    cal_noise = f_cal * jnp.abs(model_flux)
    return jnp.sqrt(noise_obs**2 + cal_noise**2)


def compute_std_inv(
    noise_obs: jnp.ndarray,
    model_flux: jnp.ndarray,
    f_cal: float | jnp.ndarray,
) -> jnp.ndarray:
    """Compute inverse effective noise (precision).

    τ = 1/σ_eff. This is the second output expected by NIFTy's
    ``VariableCovarianceGaussian`` likelihood.

    Parameters
    ----------
    noise_obs : array, shape (n_bands,)
        Observed 1-sigma uncertainties.
    model_flux : array, shape (n_bands,)
        Model-predicted fluxes.
    f_cal : float or scalar array
        Fractional calibration uncertainty.

    Returns
    -------
    array, shape (n_bands,)
        Inverse noise standard deviation τ = 1/σ_eff.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import compute_std_inv
    >>> noise = jnp.array([0.1, 0.2, 0.15])
    >>> model = jnp.array([1.0, 2.0, 1.5])
    >>> tau = compute_std_inv(noise, model, f_cal=0.05)
    >>> tau.shape
    (3,)
    """
    sigma_eff = compute_effective_noise(noise_obs, model_flux, f_cal)
    return 1.0 / sigma_eff


# ── Detection: is the noise model active? ─────────────────────────


def has_noise_model(spec) -> bool:
    """Check if any noise parameter is free (not Fixed at 0).

    Parameters
    ----------
    spec : Parameters
        Parameter specification.

    Returns
    -------
    bool
        True if the noise model is active.

    Examples
    --------
    >>> from tengri import Parameters, Uniform, has_noise_model
    >>> spec = Parameters(dust_tau_bc=Uniform(0.1, 4.0))
    >>> has_noise_model(spec)
    False
    >>> spec2 = Parameters(dust_tau_bc=Uniform(0.1, 4.0), noise_frac_cal=Uniform(0.01, 0.2))
    >>> has_noise_model(spec2)
    True
    """
    from tengri.parameters.priors import Fixed

    for name in spec.free_params:
        if name.startswith("noise_"):
            return True
    # Also check if noise_frac_cal is Fixed but nonzero
    if "noise_frac_cal" in spec.all_params:
        dist = spec.get_distribution("noise_frac_cal")
        if isinstance(dist, Fixed) and dist.value != 0.0:
            return True
    return False


def get_noise_dof(spec) -> float | None:
    """Get the Student-t degrees of freedom, or None if Gaussian.

    Parameters
    ----------
    spec : Parameters
        Parameter specification.

    Returns
    -------
    float or None
        Degrees of freedom if noise_dof is set (Fixed or free), None otherwise.

    """
    from tengri.parameters.priors import Fixed

    if "noise_dof" not in spec.all_params:
        return None
    dist = spec.get_distribution("noise_dof")
    if isinstance(dist, Fixed):
        return dist.value
    # noise_dof is free — return None to signal it's in the latent vector
    return None


def uses_student_t(spec) -> bool:
    """Check if Student-t likelihood should be used.

    Returns True if noise_dof is set to a nonzero value (fixed or free).

    Parameters
    ----------
    spec : Parameters
        Parameter specification.

    Returns
    -------
    bool
        True if Student-t likelihood should be used.

    Examples
    --------
    >>> from tengri import Parameters, Uniform, uses_student_t
    >>> spec = Parameters(dust_tau_bc=Uniform(0.1, 4.0))
    >>> uses_student_t(spec)
    False
    """
    if "noise_dof" not in spec.all_params:
        return False
    from tengri.parameters.priors import Fixed

    dist = spec.get_distribution("noise_dof")
    if isinstance(dist, Fixed):
        return dist.value > 0.0
    return True  # free noise_dof → Student-t


# ── Data mask constants for censored likelihood ───────────────────

DETECTED = 0
UPPER_LIMIT = 1
LOWER_LIMIT = -1


# ── Censored likelihood for upper/lower limits ────────────────────


def censored_log_likelihood(
    data: jnp.ndarray,
    noise_obs: jnp.ndarray,
    predicted: jnp.ndarray,
    mask: jnp.ndarray,
    f_cal: float | jnp.ndarray = 0.0,
    dof: float | None = None,
) -> jnp.ndarray:
    """Negative log-likelihood with per-band censoring.

    For detected bands (mask=0), uses the standard Gaussian (or
    Student-t) likelihood. For upper limits (mask=1), uses the normal
    CDF: ``ln L_k = ln Phi((f_upper - m_k) / sigma_k)``. For lower
    limits (mask=-1): ``ln L_k = ln Phi((m_k - f_lower) / sigma_k)``.

    All branches are computed via ``jnp.where`` for JIT compatibility
    (no Python control flow per band).

    Parameters
    ----------
    data : array, shape (n_bands,)
        Observed fluxes.  For censored bands, this holds the limit value.
    noise_obs : array, shape (n_bands,)
        Observed 1-sigma uncertainties.
    predicted : array, shape (n_bands,)
        Model-predicted fluxes.
    mask : array, shape (n_bands,)
        Per-band type: 0=detected, 1=upper limit, -1=lower limit.
    f_cal : float or scalar array
        Fractional calibration uncertainty (applied to detected bands
        only).  Default 0.0.
    dof : float or None
        Student-t degrees of freedom for detected bands.  None =
        Gaussian.

    Returns
    -------
    scalar
        Total energy (negative log-likelihood, up to additive constant).

    """
    sigma_eff = compute_effective_noise(noise_obs, predicted, f_cal)

    # --- Detected band energy ---
    r = (data - predicted) / sigma_eff
    if dof is not None:
        e_detected = 0.5 * (dof + 1.0) * jnp.log(1.0 + r**2 / dof) + jnp.log(sigma_eff)
    else:
        e_detected = 0.5 * r**2 + jnp.log(sigma_eff)

    # --- Upper limit: ln L = ln Phi((f_upper - m) / sigma) ---
    z_upper = (data - predicted) / sigma_eff
    e_upper = -jax.scipy.stats.norm.logcdf(z_upper)

    # --- Lower limit: ln L = ln Phi((m - f_lower) / sigma) ---
    z_lower = (predicted - data) / sigma_eff
    e_lower = -jax.scipy.stats.norm.logcdf(z_lower)

    # Per-band dispatch
    e_per_band = jnp.where(
        mask == UPPER_LIMIT,
        e_upper,
        jnp.where(mask == LOWER_LIMIT, e_lower, e_detected),
    )
    return jnp.sum(e_per_band)


# ── Energy and metric for JIT EVI engine ──────────────────────────


def variable_noise_hamiltonian(
    data: jnp.ndarray,
    noise_obs: jnp.ndarray,
    predicted: jnp.ndarray,
    f_cal: float | jnp.ndarray,
    dof: float | None = None,
) -> jnp.ndarray:
    """Hamiltonian (energy) for variable-covariance likelihood.

    Gaussian (dof=None):
        E_lh = ½ Σ_k (d_k - m_k)² / σ²_eff,k + Σ_k log(σ_eff,k)

    Student-t (dof set):
        E_lh = (ν+1)/2 Σ_k log(1 + r²_k/ν) + Σ_k log(σ_eff,k)

    where r_k = (d_k - m_k) / σ_eff,k.

    The log-determinant term Σlog(σ_eff) prevents the trivial solution
    of σ → ∞. This matches NIFTy's ``VariableCovarianceGaussian`` and
    ``VariableCovarianceStudentT`` energy functions.

    Note: Does NOT include the ½ξᵀξ prior term. Caller adds that.

    Parameters
    ----------
    data : array, shape (n_bands,)
        Observed fluxes.
    noise_obs : array, shape (n_bands,)
        Observed 1-sigma uncertainties.
    predicted : array, shape (n_bands,)
        Model-predicted fluxes.
    f_cal : float or scalar array
        Fractional calibration uncertainty.
    dof : float or None
        Student-t degrees of freedom. None = Gaussian (default).
        Typical values: 2 (heavy tails, Alsing+2022), 4 (moderate).

    Returns
    -------
    scalar
        Likelihood energy (negative log-likelihood up to constant).

    """
    sigma_eff = compute_effective_noise(noise_obs, predicted, f_cal)
    r = (data - predicted) / sigma_eff
    logdet = jnp.sum(jnp.log(sigma_eff))

    if dof is not None:
        # Student-t: (ν+1)/2 · Σ log(1 + r²/ν) + Σ log(σ_eff)
        return 0.5 * (dof + 1.0) * jnp.sum(jnp.log(1.0 + r**2 / dof)) + logdet
    else:
        # Gaussian: ½ Σ r² + Σ log(σ_eff)
        return 0.5 * jnp.sum(r**2) + logdet


def variable_noise_metric_vec(
    xi: jnp.ndarray,
    v: jnp.ndarray,
    signal_noise_fn,
    data: jnp.ndarray,
    unflatten,
    flatten,
) -> jnp.ndarray:
    """GGN metric-vector product for VariableCovarianceGaussian.

    Computes M @ v = J^T H_E J v + v, where:
    - J is the Jacobian of (f, τ) w.r.t. ξ
    - H_E is the Hessian of E w.r.t. (f, τ):
        d²E/df² = τ²
        d²E/dτ² = (d-f)² + 1/τ²
        d²E/(df dτ) = -2(d-f)τ

    This is the exact Gauss-Newton metric, matching NIFTy's internal
    ``VariableCovarianceGaussian.metric()`` computation.

    Parameters
    ----------
    xi : array, shape (n_latent,)
        Flattened latent parameters.
    v : array, shape (n_latent,)
        Vector to multiply.
    signal_noise_fn : callable
        Maps primals dict → (predicted, std_inv) tuple.
    data : array, shape (n_bands,)
        Observed data.
    unflatten : callable
        xi flat array → dict.
    flatten : callable
        dict → xi flat array.

    Returns
    -------
    array, shape (n_latent,)
        M @ v = J^T H_E J v + v.

    """
    xi_d = unflatten(xi)
    v_d = unflatten(v)

    # Forward + JVP: get outputs and directional derivatives
    (f, tau), (Jv_f, Jv_tau) = jax.jvp(signal_noise_fn, (xi_d,), (v_d,))

    # Hessian blocks of E(f, τ) — all diagonal in data space
    residual = data - f
    H_ff = tau**2
    H_tt = residual**2 + 1.0 / tau**2
    H_ft = -2.0 * residual * tau

    # H_E @ [Jv_f, Jv_tau]
    w_f = H_ff * Jv_f + H_ft * Jv_tau
    w_t = H_ft * Jv_f + H_tt * Jv_tau

    # J^T @ w via VJP
    _, vjp_fn = jax.vjp(signal_noise_fn, xi_d)
    (JTw,) = vjp_fn((w_f, w_t))

    # M @ v = J^T H_E J v + v (prior metric is identity)
    return flatten(JTw) + v
