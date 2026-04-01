"""Autocorrelation time estimator using Sokal's windowing method.

Reimplementation of the C acor_estimate utility in Python/NumPy,
using FFT for O(N log N) autocorrelation computation.
"""

import numpy as np


def autocorrelation_fft(x):
    """Compute normalized autocorrelation function using FFT.

    Args:
        x: 1D array of MCMC samples, shape (N,).

    Returns:
        Normalized autocorrelation C(t) for t = 0, 1, ..., N-1.
        C(0) = 1 by construction.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    x_centered = x - x.mean()

    # Zero-pad to avoid circular correlation artifacts
    fft_size = 2 * n
    f = np.fft.rfft(x_centered, n=fft_size)
    acf = np.fft.irfft(f * np.conj(f), n=fft_size)[:n]

    # Normalize so C(0) = 1
    if acf[0] == 0:
        return np.zeros(n)
    return acf / acf[0]


def integrated_autocorrelation_time(x, c=5.0):
    """Estimate integrated autocorrelation time using Sokal's convention.

    Computes tau_int = 1/2 + sum_{t=1}^{M} C(t), where M is the
    smallest integer such that M >= c * tau_int(M).

    With this convention, ESS = N / (2 * tau_int).

    Args:
        x: 1D array of MCMC samples.
        c: Sokal's window parameter (default 5.0).

    Returns:
        (tau, converged): Estimated autocorrelation time and whether
        the window criterion was satisfied before reaching N/2.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    acf = autocorrelation_fft(x)

    tau = 0.5
    max_lag = n // 2
    for t in range(1, max_lag):
        tau += acf[t]
        if t >= c * tau:
            return tau, True

    return tau, False


def effective_sample_size(x):
    """Compute effective sample size.

    ESS = N / (2 * tau_int). For i.i.d. samples, ESS ~ N.

    Args:
        x: 1D array of MCMC samples.

    Returns:
        (ess, converged): Effective sample size and convergence flag.
    """
    x = np.asarray(x, dtype=np.float64)
    tau, converged = integrated_autocorrelation_time(x)
    ess = len(x) / (2.0 * tau) if tau > 0 else float(len(x))
    return ess, converged


def batch_acor(samples, c=5.0):
    """Compute autocorrelation time for each parameter dimension.

    Args:
        samples: 2D array of shape (N, D), where N is the number
            of samples and D is the number of dimensions.
        c: Sokal's window parameter.

    Returns:
        (taus, converged): Arrays of shape (D,) with autocorrelation
        times and convergence flags per dimension.
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    n_dims = samples.shape[1]
    taus = np.zeros(n_dims)
    converged = np.zeros(n_dims, dtype=bool)

    for d in range(n_dims):
        taus[d], converged[d] = integrated_autocorrelation_time(
            samples[:, d], c=c
        )

    return taus, converged


def split_rhat(samples):
    """Split-Rhat convergence diagnostic (Vehtari et al. 2021).

    Splits the chain in half and compares between-half variance to
    within-half variance. R-hat near 1.0 indicates convergence.
    R-hat > 1.05 suggests the chain has not converged.

    Args:
        samples: 2D array of shape (N, D) or 1D array of shape (N,).

    Returns:
        Array of R-hat values, shape (D,).
    """
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    n = samples.shape[0]
    mid = n // 2
    chains = [samples[:mid], samples[mid:2 * mid]]  # equal length halves

    m = len(chains)  # 2
    n_half = chains[0].shape[0]

    # Per-chain means and variances
    chain_means = np.array([c.mean(axis=0) for c in chains])  # (m, D)
    chain_vars = np.array([c.var(axis=0, ddof=1) for c in chains])  # (m, D)

    # Within-chain variance (average of per-chain variances)
    W = chain_vars.mean(axis=0)  # (D,)

    # Between-chain variance
    grand_mean = chain_means.mean(axis=0)
    B = n_half / (m - 1) * ((chain_means - grand_mean) ** 2).sum(axis=0)

    # Pooled variance estimate
    var_hat = (n_half - 1) / n_half * W + B / n_half

    # R-hat
    rhat = np.sqrt(var_hat / np.maximum(W, 1e-30))
    return rhat
