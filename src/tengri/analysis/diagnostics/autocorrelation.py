# SPDX-License-Identifier: BSD-3-Clause
"""Autocorrelation time estimation (Sokal method, following Behroozi 2025).

Implements the autocorrelation estimator from the Ray Tracing Sampler
(Behroozi 2025, arXiv:2510.25824). Uses Sokal's self-consistent window:

    τ = 1 + 2 Σ_{k=1}^{K} ρ(k)

where K is determined by the condition k > 5τ (the window adapts to
the actual correlation length).

Two modes:

- **Standard**: ρ(k) = Cor(x_i, x_{i+k}) — standard Pearson autocorrelation
- **Absolute**: ρ(k) = Cor(|x_i - μ|, |x_{i+k} - μ|) — catches non-Gaussian
  correlations (e.g., skewed posteriors, multimodal chains)

The effective sample size is N_eff = N / τ.

Reference
---------
Sokal, A. (1997), "Monte Carlo Methods in Statistical Mechanics:
    Foundations and New Algorithms", Lectures at the Cargèse Summer School.
Behroozi, P. (2025), "The Ray Tracing Sampler", arXiv:2510.25824.
    Source: https://bitbucket.org/pbehroozi/ray-tracing-sampler/
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "autocorrelation_at_lag",
    "autocorrelation_time",
    "check_chain_length",
    "effective_sample_size",
    "rank_normalized_rhat",
    "rhat",
    "split_rhat",
]


def _never_moved(values: np.ndarray) -> bool:
    """True when every draw is identical — the parameter never moved (#1734).

    The staticness test these diagnostics need is exact and scale-free: *did any
    two draws differ?* Three call sites used to ask ``np.var(a) < 1e-30``
    instead, which is an **absolute** tolerance on a quantity carrying the
    square of the parameter's units. ``np.var`` of N identical floats is not
    exactly zero — it is rounding noise of order ``(value * eps)**2`` — so the
    threshold's sensitivity drifted with the parameter's magnitude.

    Measured on 600 identical draws, the same completely frozen chain:

    ==========  =======================  ===========================
    value       ``np.var``               vs ``1e-30``
    ==========  =======================  ===========================
    0.693       4.930e-32                below — correctly skipped
    4.130       0.000e+00                below — correctly skipped
    10.634      **3.155e-30**            **above — survived**
    ==========  =======================  ===========================

    The survivor reached :func:`rhat` as a live parameter, split R-hat scored it
    ~1.0 (within- and between-chain variance are both zero on constant data),
    and the non-empty result then bypassed the frozen-chain guard in
    ``Posterior.rhat`` — which raises only when *every* parameter is dropped.
    A dead fit reported ``max R-hat 0.998``. Whether the guard fired depended on
    how large the numbers happened to be.

    ``np.ptp`` states the intent directly and cannot drift with scale. It is
    also strictly more permissive in the right direction: a parameter that moved
    by a genuinely tiny amount is kept rather than silently dropped as static.

    Parameters
    ----------
    values : ndarray
        Draws for one parameter, shape ``(n_draw,)`` or ``(n_chain, n_draw)``.

    Returns
    -------
    bool
        ``True`` when the array is empty or every element is identical.

    Notes
    -----
    **JIT-compatible**: no — a NumPy diagnostic, called outside traced code.
    """
    if values.size == 0:
        return True
    return float(np.ptp(values)) == 0.0


def autocorrelation_at_lag(
    x: np.ndarray,
    lag: int,
    absolute: bool = False,
) -> float:
    """Compute normalized autocorrelation at a given lag.

    Follows Behroozi's ``autocorrelation()`` from ``acor_estimate.c``:
    Pearson correlation between x[0:N-lag] and x[lag:N], optionally
    using absolute deviations.

    Parameters
    ----------
    x : array, shape (N,)
        1D chain (single parameter).
    lag : int
        Lag in samples (must be >= 1).
    absolute : bool
        If True, compute correlation of |x - mean| (catches non-Gaussian
        correlations in magnitude of deviations).

    Returns
    -------
    float
        Absolute value of the normalized correlation coefficient at this lag.
    """
    n = len(x)
    if lag < 1:
        lag = 1
    if lag >= n:
        lag = n - 1

    max_i = n - lag
    head = x[:max_i]
    tail = x[lag : lag + max_i]

    mean_head = np.mean(head)
    mean_tail = np.mean(tail)

    if absolute:
        # Absolute deviation mode: |x - mean|
        abs_head = np.abs(head - mean_head)
        abs_tail = np.abs(tail - mean_tail)
        aa_head = np.mean(abs_head)
        aa_tail = np.mean(abs_tail)
        d_head = abs_head - aa_head
        d_tail = abs_tail - aa_tail
    else:
        d_head = head - mean_head
        d_tail = tail - mean_tail

    var_head = np.mean(d_head**2)
    var_tail = np.mean(d_tail**2)
    cov = np.mean(d_head * d_tail)

    if var_head <= 0 or var_tail <= 0:
        return 0.0

    return float(np.abs(cov / np.sqrt(var_head * var_tail)))


def autocorrelation_time(
    x: np.ndarray,
    absolute: bool = False,
) -> float:
    """Estimate integrated autocorrelation time using Sokal's method.

    Follows Behroozi's ``autocorrelation_time_estimate()`` from
    ``acor_estimate.c``:

        τ = 1 + 2 Σ ρ(k)

    with adaptive truncation at k > 5τ and skip-size doubling for
    efficiency on long chains.

    Parameters
    ----------
    x : array, shape (N,)
        1D chain (single parameter).
    absolute : bool
        If True, use absolute-deviation autocorrelation.

    Returns
    -------
    float
        Estimated integrated autocorrelation time (in samples).
        τ = 1 means uncorrelated; τ = 10 means ~10 samples per
        independent draw.
    """
    n = len(x)
    if n < 4:
        return float(n)

    tau = 1.0
    skip_size = 1.0
    k = 1

    while k < n:
        rho = autocorrelation_at_lag(x, int(k), absolute=absolute)
        tau += 2.0 * skip_size * rho

        # Behroozi's efficiency trick: double skip_size for large lags
        if k > 15 * skip_size:
            skip_size *= 2.0

        # Sokal's self-consistent window: stop when lag > 5τ
        if k > 5.0 * tau:
            break

        k += skip_size

    return tau


def autocorrelation_time_combined(x: np.ndarray) -> dict:
    """Compute both standard and absolute autocorrelation times.

    Returns the max of the two (following Behroozi's diagnostic output),
    which is the most conservative estimate.

    Parameters
    ----------
    x : array, shape (N,)
        1D chain.

    Returns
    -------
    dict
        Keys: 'tau_standard', 'tau_absolute', 'tau_max',
        'ess', 'chain_converged'.
    """
    n = len(x)
    tau_std = autocorrelation_time(x, absolute=False)
    tau_abs = autocorrelation_time(x, absolute=True)
    tau_max = max(tau_std, tau_abs)
    ess = n / tau_max if tau_max > 0 else float(n)
    converged = tau_max * 5 <= n

    return {
        "tau_standard": tau_std,
        "tau_absolute": tau_abs,
        "tau_max": tau_max,
        "ess": ess,
        "chain_converged": converged,
    }


def effective_sample_size(
    chains: dict[str, np.ndarray],
    exclude_prefixes: tuple[str, ...] = ("psd_xi",),
) -> dict:
    """Compute ESS for all scalar parameters using Sokal's method.

    Parameters
    ----------
    chains : dict
        Parameter name → array of shape (N,) or (N, ...).
    exclude_prefixes : tuple of str
        Parameter name prefixes to skip (e.g., GP latent vector).

    Returns
    -------
    dict
        Parameter name → dict with 'tau_standard', 'tau_absolute',
        'tau_max', 'ess', 'chain_converged'.
    """
    result = {}
    for name, arr in chains.items():
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        arr = np.asarray(arr)
        if arr.ndim != 1:
            continue
        # Skip static parameters — exactly, not by an absolute variance floor
        # whose sensitivity tracks the parameter's magnitude (#1734).
        if _never_moved(arr):
            continue
        result[name] = autocorrelation_time_combined(arr)
    return result


def check_chain_length(
    chains: dict[str, np.ndarray],
    exclude_prefixes: tuple[str, ...] = ("psd_xi",),
    verbose: bool = True,
) -> dict:
    """Check chain convergence using autocorrelation time estimates.

    Follows Behroozi's criterion: chain is converged when N > 5τ for
    all parameters. Reports both standard and absolute ACT.

    Parameters
    ----------
    chains : dict
        Parameter name → array of shape (N,).
    exclude_prefixes : tuple of str
        Prefixes to skip.
    verbose : bool
        Print diagnostics.

    Returns
    -------
    dict
        Keys: 'all_converged' (bool), 'params' (per-parameter info),
        'warnings' (list of str).
    """
    ess_info = effective_sample_size(chains, exclude_prefixes)

    if not ess_info:
        return {"all_converged": True, "params": {}, "warnings": []}

    n = len(next(iter(chains.values())))
    warnings = []
    all_converged = True

    for name, info in ess_info.items():
        if not info["chain_converged"]:
            all_converged = False
            warnings.append(
                f"{name}: τ={info['tau_max']:.1f} but need N > 5τ = "
                f"{5 * info['tau_max']:.0f} (have N={n}). "
                f"Autocorrelation time is likely underestimated."
            )

    if verbose:
        header = (
            f"{'Parameter':<22s} {'τ (std)':>8s} {'τ (abs)':>8s} "
            f"{'τ (max)':>8s} {'ESS':>8s} {'Conv?':>6s}"
        )
        print(header)
        print("-" * len(header))
        for name, info in sorted(ess_info.items()):
            conv = "OK" if info["chain_converged"] else "WARN"
            print(
                f"{name:<22s} {info['tau_standard']:>8.1f} "
                f"{info['tau_absolute']:>8.1f} {info['tau_max']:>8.1f} "
                f"{info['ess']:>8.0f} {conv:>6s}"
            )
        if warnings:
            print()
            for w in warnings:
                print(f"  >> {w}")

    return {
        "all_converged": all_converged,
        "params": ess_info,
        "warnings": warnings,
    }


# ── Split-Rhat (Gelman-Rubin) ─────────────────────────────────────────


def split_rhat(chain: np.ndarray) -> float:
    r"""Split-:math:`\hat R` (Gelman-Rubin) convergence diagnostic.

    Parameters
    ----------
    chain : array_like
        Either a 1-D array of length ``N`` (single chain split into two
        halves), or a 2-D array of shape ``(m, n)`` with ``m`` chains of
        length ``n`` (used as-is, no further splitting). [any units]

    Returns
    -------
    float
        :math:`\hat R`. Values close to 1.0 indicate convergence; values
        :math:`> 1.01` (Vehtari+2021 [2]_) or :math:`> 1.05` (looser,
        Gelman-Rubin 1992 [1]_) indicate failure to mix. Returns
        ``np.nan`` for chains too short to split or with zero variance.
        [dimensionless]

    Notes
    -----
    For a 1-D input of length :math:`N`, the chain is split into two
    halves of length :math:`n = \lfloor N/2 \rfloor` to detect
    within-chain non-stationarity. With :math:`m` chains of length
    :math:`n` and chain means :math:`\bar x_j`, overall mean
    :math:`\bar x_{\cdot\cdot}`, and chain sample variances
    :math:`s_j^2`,

    .. math::

        B &= \frac{n}{m - 1} \sum_{j=1}^{m} (\bar x_j - \bar x_{\cdot\cdot})^2 \\
        W &= \frac{1}{m} \sum_{j=1}^{m} s_j^2 \\
        \hat V &= \frac{n - 1}{n}\, W + \frac{1}{n}\, B \\
        \hat R &= \sqrt{\hat V / W}

    The numerator :math:`\hat V` over-estimates the marginal posterior
    variance until the chains have mixed; :math:`W` under-estimates it.
    Their ratio approaches 1 from above as mixing improves.

    This is the **classical** split-:math:`\hat R`. The rank-normalized
    folded variant of Vehtari et al. 2021 is more robust to heavy tails
    but adds rank/folding pre-processing steps; consider that variant
    for production diagnostics on heavy-tailed posteriors.

    References
    ----------
    .. [1] Gelman, A., Rubin, D. B., 1992, Statistical Science, 7, 457.
    .. [2] Vehtari, A. et al., 2021, Bayesian Analysis, 16, 667.
    """
    arr = np.asarray(chain)
    if arr.ndim == 1:
        n_total = arr.shape[0]
        n = n_total // 2
        if n < 2:
            return float("nan")
        chains = np.stack([arr[:n], arr[n : 2 * n]], axis=0)
    elif arr.ndim == 2:
        chains = arr
        n = chains.shape[1]
        if n < 2 or chains.shape[0] < 2:
            return float("nan")
    else:
        raise ValueError(f"split_rhat expects 1-D or 2-D array, got ndim={arr.ndim}")

    m = chains.shape[0]
    chain_means = chains.mean(axis=1)
    chain_vars = chains.var(axis=1, ddof=1)
    overall_mean = chain_means.mean()

    W = chain_vars.mean()
    if W <= 0.0 or not np.isfinite(W):
        return float("nan")

    B = (n / (m - 1)) * float(np.sum((chain_means - overall_mean) ** 2))
    var_hat = (n - 1) / n * W + B / n
    return float(np.sqrt(var_hat / W))


def rhat(
    chains: dict[str, np.ndarray],
    exclude_prefixes: tuple[str, ...] = ("psd_xi",),
) -> dict[str, float]:
    r"""Per-parameter split-:math:`\hat R`.

    Parameters
    ----------
    chains : dict
        Parameter name → 1-D chain array (or 2-D array of multiple
        chains).
    exclude_prefixes : tuple of str, optional
        Parameter name prefixes to skip (default skips ``psd_xi`` GP
        latent fields).

    Returns
    -------
    dict
        Parameter name → :math:`\hat R`. Static (zero-variance) and
        excluded parameters are dropped from the output.

    See Also
    --------
    split_rhat : underlying scalar/vector implementation.
    """
    result: dict[str, float] = {}
    for name, arr in chains.items():
        if any(name.startswith(p) for p in exclude_prefixes):
            continue
        a = np.asarray(arr)
        if a.ndim not in (1, 2):
            continue
        # The site that mattered: a frozen parameter surviving this filter lands
        # in ``result``, and a non-empty result is what ``Posterior.rhat``'s
        # frozen-chain guard treats as proof the chain moved (#1734, #1438).
        if _never_moved(a):
            continue
        result[name] = split_rhat(a)
    return result


# ── Vehtari+2021 rank-normalized folded split-Rhat ─────────────────────


def _rank_normalize(values: np.ndarray) -> np.ndarray:
    r"""Rank-normalize a flat array via the standard-normal inverse-CDF.

    Replaces each value with :math:`\Phi^{-1}((r - 0.5)/N)` where
    :math:`r` is its average rank (ties resolved by averaging) and
    :math:`N` is the total sample count. The output is approximately
    standard-normal regardless of the input distribution.

    Notes
    -----
    Uses ``scipy.stats.norm.ppf`` if available, else the rational
    approximation in Beasley & Springer (1977) via ``math.erfinv``.
    Pure-numpy implementation to keep this module dependency-light.
    """
    flat = values.ravel()
    n_total = flat.shape[0]
    # Average ranks in [1, n_total]
    order = flat.argsort()
    ranks = np.empty_like(flat, dtype=np.float64)
    # Assign tied values their mean rank.
    sorted_vals = flat[order]
    i = 0
    while i < n_total:
        j = i + 1
        while j < n_total and sorted_vals[j] == sorted_vals[i]:
            j += 1
        mean_rank = 0.5 * (i + 1 + j)  # 1-based mean
        ranks[order[i:j]] = mean_rank
        i = j
    # Map rank to (0, 1) via (r - 0.5)/N, then to standard normal via
    # the inverse normal CDF Φ⁻¹.
    p = (ranks - 0.5) / n_total
    from scipy.special import ndtri  # Φ⁻¹

    z = ndtri(p).astype(np.float64)
    return z.reshape(values.shape)


def rank_normalized_rhat(chain: np.ndarray) -> float:
    r"""Vehtari+2021 rank-normalized folded split-:math:`\hat R`.

    More robust than the classical Gelman-Rubin diagnostic to heavy
    tails and to chains that mix in mean but not in scale. Computes the
    classical split-:math:`\hat R` after **rank-normalizing** the
    samples (so the underlying distribution is Gaussian by
    construction), and additionally on the **folded** samples
    :math:`|x - \mathrm{median}(x)|` to detect scale drift. Returns the
    maximum of the two — the recommended convergence statistic of
    Vehtari et al. 2021 [1]_.

    Parameters
    ----------
    chain : array_like
        1-D chain of length ``N`` (split into halves), or 2-D ``(m, n)``
        with ``m`` chains used as-is.

    Returns
    -------
    float
        :math:`\max(\hat R_{\rm rank}, \hat R_{\rm rank, folded})`.
        Convergence threshold ``< 1.01`` (Vehtari+2021). Returns
        ``np.nan`` for chains too short to split or with zero variance.

    Notes
    -----
    The rank step makes the test robust to non-Gaussian / heavy-tailed
    posteriors where the classical :math:`\hat R` can be noisy. The
    folded step (Vehtari+2021 §4.2) catches scenarios where the chains
    agree on the mean but disagree on the scale — the classical
    diagnostic misses these because chain means converge while
    variances do not.

    References
    ----------
    .. [1] Vehtari, A. et al., 2021,
       "Rank-Normalization, Folding, and Localization: An Improved R̂
       for Assessing Convergence of MCMC,"
       Bayesian Analysis, 16, 667-718. arXiv:1903.08008.
    """
    arr = np.asarray(chain)
    if arr.ndim == 1:
        n_total = arr.shape[0]
        n = n_total // 2
        if n < 2:
            return float("nan")
        chains = np.stack([arr[:n], arr[n : 2 * n]], axis=0)
    elif arr.ndim == 2:
        chains = arr
        if chains.shape[1] < 2 or chains.shape[0] < 2:
            return float("nan")
    else:
        raise ValueError(f"rank_normalized_rhat expects 1-D or 2-D array, got ndim={arr.ndim}")
    if not np.isfinite(np.var(chains)) or _never_moved(chains):
        return float("nan")

    # Standard rank-normalization across the pooled sample.
    z = _rank_normalize(chains)
    r_basic = split_rhat(z)
    # Folded: rank-normalize |x - pooled_median|.
    folded = np.abs(chains - np.median(chains))
    z_fold = _rank_normalize(folded)
    r_fold = split_rhat(z_fold)
    if not (np.isfinite(r_basic) and np.isfinite(r_fold)):
        return float("nan")
    return float(max(r_basic, r_fold))
