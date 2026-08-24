# SPDX-License-Identifier: BSD-3-Clause
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

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tengri.utils.scale import whiten

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
    noise_obs: array, shape (n_bands,)
        Observed 1-sigma uncertainties [flux units].
    model_flux: array, shape (n_bands,)
        Model-predicted fluxes [flux units] (absolute value used for
        calibration term).
    f_cal: float or array, shape (n_bands,)
        Fractional calibration uncertainty [dimensionless].
        A scalar applies the same floor to all bands. An array applies
        a per-band floor; shape must match ``noise_obs`` and ``model_flux``.
        Typical range: 0.01–0.15.

    Returns
    -------
    array, shape (n_bands,)
        Effective noise standard deviation [same units as inputs].

    Notes
    -----
    **JIT-compatible**: yes, uses only jnp primitives.

    The calibration term ``f_cal * |model|`` adds a flux-dependent
    floor to the noise budget, preventing zero-noise solutions when
    measurement uncertainties are very small. Per-band floors allow
    calibration uncertainty to vary across filters, e.g., due to
    detector or photometry pipeline differences.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import compute_effective_noise
    >>> noise = jnp.array([0.1, 0.2, 0.15])
    >>> model = jnp.array([1.0, 2.0, 1.5])
    >>> sigma_eff = compute_effective_noise(noise, model, f_cal=0.05)
    >>> sigma_eff.shape
    (3,)

    Per-band floor example:

    >>> f_cal_per_band = jnp.array([0.02, 0.10, 0.05])
    >>> sigma_eff = compute_effective_noise(noise, model, f_cal=f_cal_per_band)
    >>> sigma_eff.shape
    (3,)
    """

    cal_noise = f_cal * jnp.abs(model_flux)
    # hypot, not sqrt(a**2 + b**2): flux uncertainties are ~1e-30, so their
    # squares (~1e-60) underflow float32 to zero, sigma_eff collapses to 0 and
    # the likelihood residual (data - pred)/sigma_eff becomes NaN. hypot factors
    # out the larger term, keeping the intermediate O(1). Identical in float64
    # to the last bit (#1206).
    return jnp.hypot(noise_obs, cal_noise)


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
    noise_obs: array, shape (n_bands,)
        Observed 1-sigma uncertainties [flux units].
    model_flux: array, shape (n_bands,)
        Model-predicted fluxes [flux units].
    f_cal: float or scalar array
        Fractional calibration uncertainty [dimensionless].

    Returns
    -------
    array, shape (n_bands,)
        Inverse noise standard deviation τ = 1/σ_eff [1/flux_units].

    Notes
    -----
    **JIT-compatible**: yes, delegates to :func:`compute_effective_noise`
    which is pure JAX.

    Used in variable-covariance likelihoods where the noise is a traced
    parameter. See :func:`variable_noise_hamiltonian` for integration into
    the likelihood energy function.

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
    spec: Parameters
        Parameter specification.

    Returns
    -------
    bool
        True if the noise model is active (any noise parameter is free or
        Fixed to nonzero value).

    Notes
    -----
    Not JIT-compatible (uses Python control flow and class introspection).

    This function checks if any parameter whose name starts with ``"noise_"``
    is in the free parameter list, or if ``noise_frac_cal`` is explicitly
    fixed to a nonzero value.

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
    spec: Parameters
        Parameter specification.

    Returns
    -------
    float or None
        Degrees of freedom (scalar, dimensionless) if ``noise_dof`` is fixed
        to a specific value. Returns ``None`` if ``noise_dof`` is not set or
        is a free (fitted) parameter.

    Notes
    -----
    Not JIT-compatible (uses Python class introspection).

    When ``noise_dof`` is free (a fitted parameter), this function returns
    ``None`` to signal that the DOF is part of the latent parameter vector
    and must be handled separately in the inference engine.

    """
    from tengri.parameters.priors import Fixed

    if "noise_dof" not in spec.all_params:
        return None
    dist = spec.get_distribution("noise_dof")
    if isinstance(dist, Fixed):
        return dist.value
    # noise_dof is free, return None to signal it's in the latent vector
    return None


def uses_student_t(spec) -> bool:
    """Check if Student-t likelihood should be used.

    Returns True if noise_dof is set to a nonzero value (fixed or free).

    Parameters
    ----------
    spec: Parameters
        Parameter specification.

    Returns
    -------
    bool
        True if the Student-t likelihood (with heavy tails) should be used
        instead of the standard Gaussian.

    Notes
    -----
    Not JIT-compatible (uses Python class introspection).

    Student-t likelihood with ``dof`` degrees of freedom is heavier-tailed
    than Gaussian and more robust to outliers. If ``noise_dof`` is free,
    it is fitted jointly with other parameters.

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


def censored_neg_log_likelihood(
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
    data: array, shape (n_bands,)
        Observed fluxes [flux units]. For censored bands, this holds the
        limit value.
    noise_obs: array, shape (n_bands,)
        Observed 1-sigma uncertainties [flux units].
    predicted: array, shape (n_bands,)
        Model-predicted fluxes [flux units].
    mask: array, shape (n_bands,)
        Per-band type: 0 = detected, 1 = upper limit, -1 = lower limit
        [dimensionless].
    f_cal: float or scalar array
        Fractional calibration uncertainty (applied to detected bands
        only) [dimensionless]. Default 0.0.
    dof: float or None
        Student-t degrees of freedom for detected bands. None = Gaussian
        (default).

    Returns
    -------
    scalar
        Total energy (negative log-likelihood, up to additive constant)
        [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses ``jnp.where`` dispatch (no Python
    control flow). Differentiable w.r.t. predicted fluxes and f_cal.

    **Censoring model**:

    - **Detected** (mask=0): Gaussian or Student-t likelihood of residual.
    - **Upper limit** (mask=1): ln L = ln Phi((f_upper - m)/σ).
    - **Lower limit** (mask=-1): ln L = ln Phi((m - f_lower)/σ).

    where Phi is the standard normal CDF.
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

    # Apply the noise model band by band
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
    data: array, shape (n_bands,)
        Observed fluxes [flux units].
    noise_obs: array, shape (n_bands,)
        Observed 1-sigma uncertainties [flux units].
    predicted: array, shape (n_bands,)
        Model-predicted fluxes [flux units].
    f_cal: float or scalar array
        Fractional calibration uncertainty [dimensionless].
    dof: float or None
        Student-t degrees of freedom. None = Gaussian (default).
        Typical values: 2 (heavy tails, Alsing+2022), 4 (moderate).

    Returns
    -------
    scalar
        Likelihood energy (negative log-likelihood up to constant)
        [dimensionless].

    Notes
    -----
    **JIT-compatible**: yes, uses only jnp primitives.

    The log-determinant term ``Σ log(σ_eff)`` is crucial: it prevents
    the trivial solution σ → ∞ and makes the likelihood fully specified.
    Does NOT include any prior term on ``f_cal`` or other parameters;
    the caller is responsible for adding the ½ξᵀξ prior.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import variable_noise_hamiltonian
    >>> data = jnp.array([1.0, 2.0, 1.5])
    >>> noise = jnp.array([0.1, 0.2, 0.15])
    >>> pred = jnp.array([0.9, 2.1, 1.4])
    >>> e = variable_noise_hamiltonian(data, noise, pred, f_cal=0.05)
    >>> float(e) > 0
    True
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
    xi: array, shape (n_latent,)
        Flattened latent parameters.
    v: array, shape (n_latent,)
        Vector to multiply.
    signal_noise_fn: callable
        Maps primals dict → (predicted, std_inv) tuple.
    data: array, shape (n_bands,)
        Observed data.
    unflatten: callable
        xi flat array → dict.
    flatten: callable
        dict → xi flat array.

    Returns
    -------
    array, shape (n_latent,)
        M @ v = J^T H_E J v + v.

    Notes
    -----
    **JIT-compatible**: yes, uses only jax primitives (jax.jvp, jax.vjp).

    **Gradient-safe**: yes, differentiable w.r.t. data and model parameters.

    Implements the metric-vector product for the Gauss-Newton approximation
    of the variational Hessian, used in information field theory inference.
    The prior metric is assumed to be the identity.

    """
    xi_d = unflatten(xi)
    v_d = unflatten(v)

    # Forward + JVP: get outputs and directional derivatives
    (f, tau), (Jv_f, Jv_tau) = jax.jvp(signal_noise_fn, (xi_d,), (v_d,))

    # Hessian blocks of E(f, τ), all diagonal in data space.
    #
    # Applied in factored form (#1617). Written directly, two of the four blocks
    # are destroyed in float32 at a real photometric sigma, in opposite
    # directions, measured, not inferred:
    #
    #     H_ff = tau**2               1.111e+59  ->  inf
    #     H_tt = r**2 + 1/tau**2      3.611e-56  ->  0.0   (both terms underflow)
    #
    # The second is the dangerous one: the curvature along the noise direction
    # is silently *removed* rather than poisoned, so the metric stays finite and
    # looks usable. Each block factors into representable pieces, so no
    # rescaling of the metric is needed:
    #
    #     H_ff Jv_f    = tau**2 Jv_f          = (Jv_f/sigma)/sigma,  sigma = 1/tau
    #     H_tt Jv_tau  = ((r tau)**2 + 1) Jv_tau / tau**2
    #
    # ``r_std = residual * tau`` is the standardized residual, O(1) by
    # construction, so neither underflowing term is ever formed. ``whiten``
    # carries the optimization_barrier that stops XLA re-associating the pairs
    # back into the overflowing square (#1535/#1588).
    residual = data - f
    sigma_eff = 1.0 / tau  # ~3e-30, representable where tau**2 is not
    r_std = residual * tau  # standardized residual, O(1)
    H_ft = -2.0 * r_std  # == -2 * residual * tau

    h_ff_jv = whiten(whiten(Jv_f, sigma_eff), sigma_eff)  # tau**2 * Jv_f
    h_tt_jv = (r_std**2 + 1.0) * whiten(whiten(Jv_tau, tau), tau)

    # H_E @ [Jv_f, Jv_tau]
    w_f = h_ff_jv + H_ft * Jv_tau
    w_t = H_ft * Jv_f + h_tt_jv

    # J^T @ w via VJP
    _, vjp_fn = jax.vjp(signal_noise_fn, xi_d)
    (JTw,) = vjp_fn((w_f, w_t))

    # M @ v = J^T H_E J v + v (prior metric is identity)
    return flatten(JTw) + v


# ── GP covariance kernels for correlated spectral noise ───────────


def exp_squared_kernel(
    x: jnp.ndarray,
    amplitude: float | jnp.ndarray,
    length_scale: float | jnp.ndarray,
    x2: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Squared exponential (RBF) covariance kernel.

    Computes the squared exponential kernel matrix for Gaussian process
    modeling of correlated spectral noise.

    Parameters
    ----------
    x: array, shape (n,)
        Coordinate values [Angstrom] (or any real-valued coordinate).
    amplitude: float or scalar array
        Kernel amplitude :math:`\sigma`. Controls overall variance.
    length_scale: float or scalar array
        Kernel length scale :math:`\ell`. Controls correlation length.
    x2: array, shape (m,), optional
        Second set of coordinates for cross-covariance. If None,
        computes auto-covariance (x vs x).

    Returns
    -------
    K: ndarray, shape (n, n) or (n, m)
        Covariance matrix. If x2 is None, returns symmetric (n, n)
        auto-covariance; otherwise returns (n, m) cross-covariance.

    Notes
    -----
    .. math::

        K(x, x') = \sigma^2 \exp\!\left(-\frac{(x - x')^2}{2 \ell^2}\right)

    where :math:`\sigma` is the amplitude and :math:`\ell` is the length scale.

    JIT-compatible and differentiable w.r.t. all float inputs.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import exp_squared_kernel
    >>> x = jnp.linspace(4000.0, 8000.0, 20)
    >>> K = exp_squared_kernel(x, amplitude=1.0, length_scale=500.0)
    >>> K.shape
    (20, 20)
    >>> K[0, 0]  # diagonal entry equals amplitude^2
    DeviceArray(1., dtype=float32)
    """
    x2 = x if x2 is None else x2
    diff = x[:, None] - x2[None, :]
    return amplitude**2 * jnp.exp(-0.5 * (diff / length_scale) ** 2)


def matern32_kernel(
    x: jnp.ndarray,
    amplitude: float | jnp.ndarray,
    length_scale: float | jnp.ndarray,
    x2: jnp.ndarray | None = None,
) -> jnp.ndarray:
    r"""Matérn 3/2 covariance kernel.

    Once-differentiable GP kernel, smoother than the exponential kernel,
    rougher than squared-exponential. A good default for correlated spectral
    noise where the autocorrelation length is finite but the residuals are
    not infinitely smooth.

    Parameters
    ----------
    x: array, shape (n,)
        Coordinate values [Angstrom] (or any real coordinate).
    amplitude: float or scalar array
        Kernel amplitude :math:`\sigma`.
    length_scale: float or scalar array
        Correlation length :math:`\ell`.
    x2: array, shape (m,), optional
        Second set of coordinates for cross-covariance. Defaults to ``x``
        (auto-covariance).

    Returns
    -------
    K: ndarray, shape (n, n) or (n, m)

    Notes
    -----
    .. math::

        K(x, x') = \sigma^2 \left(1 + \frac{\sqrt{3}\,r}{\ell}\right)
        \exp\!\left(-\frac{\sqrt{3}\,r}{\ell}\right), \quad r = |x - x'|

    JIT-compatible. :math:`r` is softened to :math:`\sqrt{r^2 + 10^{-20}}`
    so the gradient is defined at the diagonal.

    Examples
    --------
    >>> x = jnp.linspace(4000.0, 8000.0, 20)
    >>> K = matern32_kernel(x, amplitude=1.0, length_scale=500.0)
    >>> K.shape
    (20, 20)
    """
    x2 = x if x2 is None else x2
    diff = x[:, None] - x2[None, :]
    # Soft |r| keeps the gradient defined at r=0 (the diagonal).
    r = jnp.sqrt(diff**2 + 1e-20)
    arg = jnp.sqrt(3.0) * r / length_scale
    return amplitude**2 * (1.0 + arg) * jnp.exp(-arg)


def gp_noise_covariance(
    wavelength: jnp.ndarray,
    noise_obs: jnp.ndarray,
    gp_amplitude: float | jnp.ndarray,
    gp_length_scale: float | jnp.ndarray,
    kernel: str = "exp_squared",
) -> jnp.ndarray:
    r"""Compute noise covariance as white noise + GP kernel.

    Combines observational uncertainties (white noise diagonal) with
    a Gaussian process kernel to model correlated spectral noise.

    Parameters
    ----------
    wavelength: array, shape (n_wave,)
        Wavelengths [Angstrom].
    noise_obs: array, shape (n_wave,)
        Observed 1-sigma uncertainties [same units as flux].
    gp_amplitude: float or scalar array
        GP kernel amplitude. Dimensionless scaling of kernel.
    gp_length_scale: float or scalar array
        GP kernel length scale [Angstrom].
    kernel: str, optional
        Kernel type: "exp_squared" (default) or "matern32".

    Returns
    -------
    N: ndarray, shape (n_wave, n_wave)
        Covariance matrix :math:`N = \text{diag}(\sigma_{\text{obs}}^2) + K_{\text{gp}}`.

    Notes
    -----
    The returned matrix combines white noise variance on the diagonal
    with off-diagonal correlations from the chosen GP kernel. This is
    suitable for spectral fitting likelihoods that support full
    covariance matrices.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri import gp_noise_covariance
    >>> wave = jnp.linspace(4000.0, 8000.0, 50)
    >>> noise = jnp.ones(50) * 0.1
    >>> N = gp_noise_covariance(
    ...     wave, noise, gp_amplitude=0.5, gp_length_scale=300.0, kernel="exp_squared"
    ... )
    >>> N.shape
    (50, 50)
    """
    # White noise diagonal
    diag_noise = jnp.diag(noise_obs**2)

    # Choose kernel
    if kernel == "exp_squared":
        K_gp = exp_squared_kernel(wavelength, gp_amplitude, gp_length_scale)
    elif kernel == "matern32":
        K_gp = matern32_kernel(wavelength, gp_amplitude, gp_length_scale)
    else:
        raise ValueError(f"Unknown kernel '{kernel}'. Must be 'exp_squared' or 'matern32'.")

    return diag_noise + K_gp


def apply_zp_floor(
    flux: jnp.ndarray,
    noise: jnp.ndarray,
    floor: float | jnp.ndarray,
) -> jnp.ndarray:
    r"""Inflate per-band noise with a fractional zero-point systematic floor.

    Combines the existing per-band statistical uncertainty with a
    multiplicative ZP-calibration term in quadrature:

    .. math::

        \sigma_{\rm eff}^2 = \sigma_{\rm data}^2 +
        \bigl(f_{\rm floor} \cdot |F|\bigr)^2

    Different surveys / passbands have different ZP calibration
    floors (e.g. SDSS ~2%, JWST NIRCam ~5%, Pan-STARRS ~1%). When the
    photometric pipeline doesn't already fold these into the reported
    noise, apply this preprocessing step before constructing the
    Fitter so the fit can't be falsely confident below the
    calibration limit.

    Parameters
    ----------
    flux: array_like, shape (n_bands,)
        Observed flux density per band. [erg/s/cm^2/Hz]
    noise: array_like, shape (n_bands,)
        Statistical 1-sigma noise per band. Same units as ``flux``.
    floor: float or array_like, shape (n_bands,)
        Fractional ZP floor (e.g. ``0.02`` for 2%). Scalar applies
        the same floor to all bands; array gives per-band values.
        Must be non-negative.

    Returns
    -------
    ndarray, shape (n_bands,)
        Effective per-band 1-sigma noise. Same units as ``noise``.

    Raises
    ------
    ValueError
        If ``floor`` has shape incompatible with ``flux``, or any
        ``floor`` value is negative.

    Notes
    -----
    **JIT-compatible**: yes, pure ``jnp`` arithmetic.

    Caps the achievable per-band SNR at :math:`1/f_{\rm floor}` (e.g.
    a 2% floor caps SNR at 50). Uses ``|F|`` so non-detections with
    negative flux still yield a finite, positive noise term.

    The fractional floor is *not* the same as the existing
    ``noise_frac_cal`` parameter, which applies a global Student-t-style
    calibration term inside the likelihood. Use this utility upstream
    for known per-band ZP uncertainties; reserve ``noise_frac_cal``
    for free-parameter calibration nuisances.

    Examples
    --------
    >>> flux = jnp.array([1.0, 2.0, 3.0])
    >>> noise = jnp.array([0.05, 0.10, 0.15])
    >>> # SDSS optical: 2% per band
    >>> sigma_eff = apply_zp_floor(flux, noise, 0.02)
    """
    flux_arr = jnp.asarray(flux)
    noise_arr = jnp.asarray(noise)
    floor_arr = jnp.asarray(floor)

    if floor_arr.ndim == 0:
        if float(floor_arr) < 0.0:
            raise ValueError("zp floor must be non-negative (got < 0)")
    else:
        if floor_arr.shape != flux_arr.shape:
            raise ValueError(f"floor shape {floor_arr.shape} != flux shape {flux_arr.shape}")
        if not bool(jnp.all(floor_arr >= 0.0)):
            raise ValueError("zp floor must be non-negative (got < 0 in some band)")

    sys_term = floor_arr * jnp.abs(flux_arr)
    # hypot avoids the float32 underflow of squaring ~1e-30 flux uncertainties
    # (see compute_effective_noise). Identical in float64 (#1206).
    return jnp.hypot(noise_arr, sys_term)


# ── Photon-limited Poisson likelihood ─────────────────────────────


@dataclass(frozen=True)
class PoissonNoiseLikelihood:
    r"""Photon-limited Poisson likelihood with sky and read-noise terms.

    Implements Gaussian approximation to Poisson counting noise, valid
    when observed count rates are ≥ ~5 photons (Newberry 1991).

    The variance combines:

    - Poisson shot noise: σ²_shot = F / g (counts → e⁻)
    - Sky background: σ²_sky = σ_sky² / g
    - Read noise: σ²_read = σ_read² / g²
    - Flux-dependent systematic floor: σ²_sys = (f_sys · F)²

    Total: σ²_eff = F/g + σ_sky²/g + σ_read²/g² + (f_sys · F)²

    For SED fitting (source-level photometry), the per-source variance
    in detected counts is typically σ² = F + σ_back², where F is the
    source photon count and σ_back is the background variance in count
    space.

    Parameters
    ----------
    gain: float
        CCD gain in electrons per ADU [e⁻/ADU]. Default 1.0 (pure Poisson
        in ADU counts).
    sky_var: float
        Background (sky + dark) variance in count space [counts²].
        Default 0.0.
    read_noise: float
        Read noise standard deviation in electrons [e⁻]. Default 0.0.
    systematic_floor: float
        Fractional flux-dependent systematic uncertainty added in quadrature
        [dimensionless]. Typical range: 0.01–0.05. Default 0.0.

    Attributes
    ----------
    gain: float
    sky_var: float
    read_noise: float
    systematic_floor: float

    Notes
    -----
    **JIT-compatible**: yes, pure JAX, differentiable w.r.t. predicted flux.

    For source counts F (in detected photons), the effective Gaussian
    variance is:

    .. math::

        \sigma^2_{\rm eff} = \frac{F}{g} + \frac{\sigma^2_{\rm sky}}{g}
        + \frac{\sigma^2_{\rm read}}{g^2} + \bigl(f_{\rm sys} \cdot F\bigr)^2

    When all background terms are zero, reduces to σ² = F (pure Poisson).
    Handles predicted=0 safely by clamping to small positive value.

    References
    ----------
    .. [1] Bevington, P. R. & Robinson, D. K. (2003). *Data Reduction and
           Error Analysis for the Physical Sciences*, 3rd edn.
           McGraw-Hill. Section 3.6 (Poisson distributions).
    """

    gain: float = 1.0
    sky_var: float = 0.0
    read_noise: float = 0.0
    systematic_floor: float = 0.0

    def log_prob(
        self,
        observed: jnp.ndarray,
        predicted: jnp.ndarray,
    ) -> jnp.ndarray:
        """Gaussian log-likelihood for Poisson photon counts.

        Parameters
        ----------
        observed: array, shape (n_data,)
            Observed flux or counts [counts or flux units, depending on gain].
        predicted: array, shape (n_data,)
            Model-predicted flux [same units as observed].

        Returns
        -------
        log_prob: array, shape (n_data,)
            Per-datum log-likelihood (likelihood, not log-likelihood energy).
            Caller is responsible for summing and negating for energy.

        Notes
        -----
        **JIT-compatible**: yes, pure ``jnp`` arithmetic.

        Computes log(p(obs | pred)) under Gaussian approximation to Poisson.
        Avoids singularities by clamping predicted flux to ≥ eps internally.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from tengri.observation.noise import PoissonNoiseLikelihood
        >>> lh = PoissonNoiseLikelihood(gain=1.0, sky_var=0.0)
        >>> obs = jnp.array([100.0, 200.0])
        >>> pred = jnp.array([95.0, 205.0])
        >>> lp = lh.log_prob(obs, pred)
        >>> lp.shape
        (2,)
        """
        # Clamp predicted to small positive value to avoid div-by-zero
        eps = 1e-8
        pred_safe = jnp.maximum(predicted, eps)

        # Effective variance: shot + sky + read + systematic
        shot_var = pred_safe / self.gain
        sky_term = self.sky_var / self.gain
        read_term = (self.read_noise / self.gain) ** 2
        sys_term = (self.systematic_floor * pred_safe) ** 2

        sigma2_eff = shot_var + sky_term + read_term + sys_term
        sigma_eff = jnp.sqrt(sigma2_eff)

        # Gaussian log-likelihood: -0.5 * (residual / sigma)^2 - log(sigma)
        residual = observed - predicted
        chi2 = (residual / sigma_eff) ** 2
        return -0.5 * chi2 - jnp.log(sigma_eff)


# ── Student-t robust likelihood (outlier-resistant) ────────────────


@dataclass(frozen=True)
class StudentTLikelihood:
    r"""Heavy-tailed Student-t likelihood for outlier-robust SED fitting.

    Implements the outlier-robust noise model of Hogg, Bovy & Lang (2010),
    using a Student-t distribution with ν degrees of freedom.

    For ν → ∞, reduces to Gaussian. For low ν (e.g., 2–4), the heavy
    tails make outliers far less damaging to the posterior. Typical use:
    ν is either fixed (e.g., 2, 4, 10) or fitted as a free parameter.

    Parameters
    ----------
    dof: float
        Degrees of freedom ν [dimensionless]. Default 4.0. Smaller values
        give heavier tails; ν=1 is Cauchy. Typical range: [1, 30].

    Attributes
    ----------
    dof: float
        Degrees of freedom.

    Notes
    -----
    **JIT-compatible**: yes, uses ``jax.scipy.stats.t.logpdf``.

    The log-likelihood per data point is:

    .. math::

        \ln p(d | m, \sigma, \nu) = \ln T_\nu\!\left(\frac{d - m}{\sigma}; \nu\right)
        - \ln \sigma

    where :math:`T_\nu` is the Student-t PDF with :math:`\nu` degrees of freedom.

    The residual scale σ should be supplied by the caller; this class does
    not manage observational uncertainties, use a noise model (e.g.
    ``compute_effective_noise``) to construct σ from data.

    References
    ----------
    .. [1] Hogg, D. W., Bovy, J., & Lang, D. (2010).
           "Data analysis recipes. I. Fitting a model to data."
           arXiv:1008.4686
           ADS: 2010arXiv1008.4686H
           https://doi.org/10.48550/arXiv.1008.4686
    """

    dof: float = 4.0

    def log_prob(
        self,
        observed: jnp.ndarray,
        predicted: jnp.ndarray,
        sigma: jnp.ndarray,
    ) -> jnp.ndarray:
        """Student-t log-likelihood.

        Parameters
        ----------
        observed: array, shape (n_data,)
            Observed values [arbitrary units].
        predicted: array, shape (n_data,)
            Model-predicted values [same units as observed].
        sigma: array, shape (n_data,)
            Noise standard deviation per datum [same units as observed].
            Typically computed via ``compute_effective_noise(...)`` or
            similar noise model.

        Returns
        -------
        log_prob: array, shape (n_data,)
            Per-datum log-likelihood. Caller sums for total likelihood.

        Notes
        -----
        **JIT-compatible**: yes, delegates to
        ``jax.scipy.stats.t.logpdf``.

        Avoids numerical instability by clamping σ to ≥ eps internally.

        Examples
        --------
        >>> import jax.numpy as jnp
        >>> from tengri.observation.noise import StudentTLikelihood
        >>> lh = StudentTLikelihood(dof=4.0)
        >>> obs = jnp.array([1.0, 2.0, 5.0])  # 5.0 is outlier
        >>> pred = jnp.array([1.0, 2.0, 1.5])
        >>> sigma = jnp.array([0.5, 0.5, 0.5])
        >>> lp = lh.log_prob(obs, pred, sigma)
        >>> lp.shape
        (3,)
        """
        # Clamp sigma to avoid numerical issues
        eps = 1e-8
        sigma_safe = jnp.maximum(sigma, eps)

        # Standardize residuals
        residual = (observed - predicted) / sigma_safe

        # Student-t log-pdf from JAX; include -log(sigma) normalization
        return jax.scipy.stats.t.logpdf(residual, self.dof) - jnp.log(sigma_safe)
