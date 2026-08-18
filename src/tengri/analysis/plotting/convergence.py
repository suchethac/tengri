# SPDX-License-Identifier: BSD-3-Clause
"""Convergence diagnostics and posterior summary visualization.

Utilities for monitoring MCMC/VI convergence and reporting effective sample sizes,
acceptance rates, and other diagnostic metrics.
"""

import matplotlib.pyplot as plt
import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Diagnostics table
# ═══════════════════════════════════════════════════════════════════


def diagnostics_table(results, names=None):
    """Print a formatted diagnostics comparison table.

    Parameters
    ----------
    results : dict
        Mapping from method name to :class:`Posterior` (e.g. from :meth:`Fitter.run`).
    names : list of str, optional
        Display order. Defaults to ``list(results.keys())``.

    Returns
    -------
    None

    Examples
    --------
    .. code-block:: python

        from tengri import diagnostics_table

        results = {"NUTS": posterior_nuts, "VI": posterior_vi}
        diagnostics_table(results)
        # Method          Wall time   ESS (min)   ESS (med)   Accept %
        # ----------------------------------------------------------------
        # NUTS               42.3s         812         934      82.4%
        # VI                  3.1s           —           —         —
    """
    if names is None:
        names = list(results.keys())

    header = (
        f"{'Method':<15} {'Wall time':>10} {'ESS (min)':>10} {'ESS (med)':>10} {'Accept %':>10}"
    )
    print(header)
    print("-" * len(header))

    for name in names:
        res = results[name]
        wt = f"{res.wall_time_s:.1f}s"

        if res.samples is not None:
            ess = res.effective_sample_size()
            ess_vals = [v for k, v in ess.items() if k != "psd_xi"]
            ess_min = f"{min(ess_vals):.0f}" if ess_vals else "—"
            ess_med = f"{np.median(ess_vals):.0f}" if ess_vals else "—"
        else:
            ess_min, ess_med = "—", "—"

        accept = res.diagnostics.get(
            "accept_rate_post_burnin", res.diagnostics.get("mean_accept_prob", None)
        )
        accept_str = f"{accept:.1%}" if accept is not None else "—"
        print(f"{name:<15} {wt:>10} {ess_min:>10} {ess_med:>10} {accept_str:>10}")


# ═══════════════════════════════════════════════════════════════════
# Posterior SFH plot convenience wrapper
# ═══════════════════════════════════════════════════════════════════


def posterior_plot_sfh(result, truth_sfh=None, ax=None):
    """Plot SFH posterior with optional truth.

    Convenience function for Posterior.plot_sfh() — shows posterior median
    and 68% credible band over lookback time.

    Parameters
    ----------
    result : Posterior
        Posterior inference result with model reference.
    truth_sfh : dict, optional
        Truth SFH parameters to plot as dashed line. Should be a
        parameter dict (will be passed to model.predict_sfh).
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes

    Raises
    ------
    RuntimeError
        If model reference is not available.
    """
    if result._model is None:
        raise RuntimeError("plot_sfh() requires model reference")

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    # Compute SFH for MAP
    sfh_map = result._model.predict_sfh(result.params)
    t_gyr = np.array(sfh_map["t_gyr"])
    sfr_mean = np.array(sfh_map["sfr_mean"])

    # If samples available, compute posterior band
    if result.samples is not None:
        n_samples = next(iter(result.samples.values())).shape[0]
        sfh_samples = []
        for i in range(n_samples):
            sample_i = {k: v[i] for k, v in result.samples.items()}
            sfh_i = result._model.predict_sfh(sample_i)
            sfh_samples.append(np.array(sfh_i["sfr_mean"]))
        sfh_array = np.array(sfh_samples)

        # Plot posterior band
        sfr_lo = np.percentile(sfh_array, 16, axis=0)
        sfr_hi = np.percentile(sfh_array, 84, axis=0)
        ax.fill_between(t_gyr, sfr_lo, sfr_hi, alpha=0.3, color="C0", label="68% credible region")

    # Plot posterior median
    ax.plot(t_gyr, sfr_mean, "-", color="C0", lw=2, label="Posterior median")

    # Overlay truth if provided
    if truth_sfh is not None:
        try:
            sfh_true = result._model.predict_sfh(truth_sfh)
            t_gyr_true = np.array(sfh_true["t_gyr"])
            sfr_true = np.array(sfh_true["sfr_mean"])
            ax.plot(t_gyr_true, sfr_true, "--", color="k", lw=2, label="Truth")
        except (KeyError, AttributeError, ValueError, TypeError):
            # KeyError: sfr_mean or t_gyr missing from output
            # AttributeError: predict_sfh method doesn't exist
            # ValueError/TypeError: array conversion failed
            pass

    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_ylabel("SFR [M☉/yr]")
    ax.legend(frameon=False)

    return fig, ax


# ═══════════════════════════════════════════════════════════════════
# Convergence thresholds and checks
# ═══════════════════════════════════════════════════════════════════

# Thresholds following Vehtari et al. (2021) "Rank-normalization,
# folding, and localization" and Stan/ArviZ conventions.
CONVERGENCE_THRESHOLDS = {
    "ess_bulk_min": 100,  # minimum bulk ESS per parameter
    "ess_total_target": 400,  # target total ESS for reliable summaries
    "divergence_warn": 0,  # any divergence warrants investigation
    "divergence_fail_pct": 5,  # >5% divergences = serious problem
    "accept_rt_lo": 0.20,  # RT acceptance too low = step_size too large
    "accept_rt_hi": 0.90,  # RT acceptance too high = barely moving
    "accept_nuts_target": 0.80,
}


def convergence_check(result, method_name="", verbose=True):
    """Run industry-standard convergence diagnostics on a Posterior result.

    Checks ESS, acceptance rate, and divergences against standard
    thresholds (Vehtari et al. 2021; Stan/ArviZ conventions).

    Parameters
    ----------
    result : Posterior
        Inference result with .samples and .diagnostics.
    method_name : str, optional
        Label for printing (e.g., "RT", "NUTS", "geoVI"). Default "".
    verbose : bool, optional
        If True, print detailed diagnostics. Default True.

    Returns
    -------
    dict
        Keys: 'converged' (bool), 'warnings' (list of str),
        'ess_min', 'ess_median', 'n_params_low_ess'.

    Examples
    --------
    .. code-block:: python

        from tengri import convergence_check

        info = convergence_check(posterior, method_name="NUTS", verbose=True)
        if not info["converged"]:
            print("Warnings:", info["warnings"])
    """
    warnings = []
    info = {}
    th = CONVERGENCE_THRESHOLDS

    name = method_name or result.diagnostics.get("method", "Sampler")

    # --- ESS + Autocorrelation Time (Sokal/Behroozi method) ---
    if result.samples is not None:
        ess = result.effective_sample_size()
        # Exclude GP latent vector from summary (too many params)
        ess_phys = {k: v for k, v in ess.items() if not k.startswith("psd_xi")}
        if ess_phys:
            ess_vals = list(ess_phys.values())
            ess_min = min(ess_vals)
            ess_med = float(np.median(ess_vals))
            n_low = sum(1 for v in ess_vals if v < th["ess_bulk_min"])

            info["ess_min"] = ess_min
            info["ess_median"] = ess_med
            info["n_params_low_ess"] = n_low

            if ess_min < th["ess_bulk_min"]:
                low_params = [k for k, v in ess_phys.items() if v < th["ess_bulk_min"]]
                warnings.append(
                    f"Low ESS: {n_low}/{len(ess_phys)} params below "
                    f"{th['ess_bulk_min']} "
                    f"(min ESS = {ess_min:.0f} for "
                    f"{low_params[0] if low_params else '?'})"
                )
            if ess_med < th["ess_total_target"]:
                warnings.append(
                    f"Median ESS = {ess_med:.0f} < {th['ess_total_target']} "
                    f"target — consider more samples"
                )

        # Autocorrelation time check (Behroozi 2025 criterion: N > 5τ)
        try:
            act_info = result.autocorrelation_time()
            act_phys = {k: v for k, v in act_info.items() if not k.startswith("psd_xi")}
            if act_phys:
                tau_max_vals = [v["tau_max"] for v in act_phys.values()]
                info["tau_max_max"] = max(tau_max_vals)
                info["tau_max_median"] = float(np.median(tau_max_vals))
                not_converged = [k for k, v in act_phys.items() if not v["chain_converged"]]
                info["n_params_short_chain"] = len(not_converged)
                if not_converged:
                    worst = max(act_phys.items(), key=lambda kv: kv[1]["tau_max"])
                    warnings.append(
                        f"Chain too short for {len(not_converged)} params "
                        f"(need N > 5τ; worst: {worst[0]} with "
                        f"τ={worst[1]['tau_max']:.1f})"
                    )
        except Exception:
            pass  # gracefully degrade if ACT fails
    else:
        info["ess_min"] = None
        info["ess_median"] = None
        info["n_params_low_ess"] = None

    # --- Frozen chain detection: all samples identical per parameter (#1437) ---
    # A frozen chain has ~0 within-chain and ~0 between-chain variance, so split-R-hat
    # scores it ~1.0 (perfect convergence) even though the sampler never moved.
    # The guard: detect parameters where n_unique == 1 and surface as a failure.
    frozen_params = []
    if result.samples is not None:
        for param_name, samples in result.samples.items():
            if param_name.startswith("psd_xi"):
                continue
            arr = np.asarray(samples)
            if arr.ndim >= 1 and float(np.ptp(arr)) == 0.0:
                frozen_params.append(param_name)

        if frozen_params:
            info["frozen_params"] = frozen_params
            warnings.append(
                f"FROZEN: {len(frozen_params)} parameter(s) never moved "
                f"({', '.join(frozen_params[:3])}{'...' if len(frozen_params) > 3 else ''}) "
                f"— R-hat cannot detect this (both variances ~0, ratio ~1)"
            )

    # --- Divergences (NUTS) ---
    n_div = result.diagnostics.get("n_divergent", None)
    if n_div is not None:
        n_samples = result.diagnostics.get("n_samples", 1)
        div_pct = 100 * n_div / max(n_samples, 1)
        info["n_divergent"] = n_div
        info["divergence_pct"] = div_pct

        # Explicit detection: all samples diverged (#1437)
        if n_div == n_samples and n_samples > 0:
            info["all_samples_divergent"] = True
            warnings.append(
                f"CRITICAL: {n_div}/{n_samples} divergent transitions (100%) "
                f"— the sampler rejected every proposal. This is a dead fit, not a converged one."
            )
        elif n_div > th["divergence_warn"]:
            severity = "SERIOUS" if div_pct > th["divergence_fail_pct"] else "WARNING"
            warnings.append(
                f"{severity}: {n_div}/{n_samples} divergent transitions "
                f"({div_pct:.1f}%) — posterior may be unreliable"
            )
        else:
            info["all_samples_divergent"] = False
    else:
        info["all_samples_divergent"] = False

    # --- Acceptance rate (RT) ---
    accept = result.diagnostics.get("accept_rate_post_burnin", None)
    if accept is not None:
        info["acceptance_rate"] = accept
        if accept < th["accept_rt_lo"]:
            warnings.append(f"RT acceptance {accept:.0%} too low — reduce step_size")
        elif accept > th["accept_rt_hi"]:
            warnings.append(
                f"RT acceptance {accept:.0%} too high — chain barely moving, increase step_size"
            )

    # --- Acceptance rate (NUTS) ---
    accept_nuts = result.diagnostics.get("mean_accept_prob", None)
    if accept_nuts is not None:
        info["acceptance_rate"] = accept_nuts

    # --- Overall verdict ---
    converged = len(warnings) == 0
    info["converged"] = converged
    info["warnings"] = warnings

    # --- Print ---
    if verbose:
        status = "CONVERGED" if converged else "WARNINGS"
        print(f"\n{'=' * 60}")
        print(f"  Convergence diagnostics: {name}  [{status}]")
        print(f"{'=' * 60}")
        if info.get("ess_min") is not None:
            print(f"  ESS (min / median): {info['ess_min']:.0f} / {info['ess_median']:.0f}")
        if info.get("tau_max_max") is not None:
            print(
                f"  ACT τ (max / median): {info['tau_max_max']:.1f} / {info['tau_max_median']:.1f}"
            )
        if "acceptance_rate" in info:
            print(f"  Acceptance rate:    {info['acceptance_rate']:.1%}")
        if "n_divergent" in info:
            print(
                f"  Divergences:        {info['n_divergent']} / "
                f"{result.diagnostics.get('n_samples', '?')}"
            )
        if warnings:
            for w in warnings:
                print(f"  >> {w}")
        else:
            print("  All diagnostics passed.")
        print(f"{'=' * 60}\n")

    return info


def convergence_table(results_dict, verbose=True):
    """Run convergence checks on multiple results, print comparison table.

    Parameters
    ----------
    results_dict : dict
        Mapping from method name to :class:`Posterior`.
    verbose : bool, optional
        If True, print formatted table. Default True.

    Returns
    -------
    dict
        Mapping from method name to convergence_check info dict.

    Examples
    --------
    .. code-block:: python

        from tengri import convergence_table

        results = {"NUTS": posterior_nuts, "VI": posterior_vi}
        info = convergence_table(results, verbose=True)
    """
    all_info = {}
    for name, res in results_dict.items():
        all_info[name] = convergence_check(res, method_name=name, verbose=False)

    if verbose:
        # Compact table
        header = (
            f"{'Method':<15} {'ESS min':>8} {'ESS med':>8} "
            f"{'τ max':>8} {'Accept':>8} {'Diverg':>8} {'Status':>10}"
        )
        print(header)
        print("-" * len(header))
        for name, info in all_info.items():
            ess_min = f"{info['ess_min']:.0f}" if info.get("ess_min") is not None else "—"
            ess_med = f"{info['ess_median']:.0f}" if info.get("ess_median") is not None else "—"
            tau_max = f"{info['tau_max_max']:.1f}" if info.get("tau_max_max") is not None else "—"
            accept = f"{info['acceptance_rate']:.0%}" if "acceptance_rate" in info else "—"
            diverg = f"{info['n_divergent']}" if "n_divergent" in info else "—"
            status = "OK" if info["converged"] else "WARN"
            print(
                f"{name:<15} {ess_min:>8} {ess_med:>8} "
                f"{tau_max:>8} {accept:>8} {diverg:>8} {status:>10}"
            )

        # Print warnings below
        any_warns = any(info["warnings"] for info in all_info.values())
        if any_warns:
            print("\nWarnings:")
            for name, info in all_info.items():
                for w in info["warnings"]:
                    print(f"  [{name}] {w}")

    return all_info


def plot_autocorrelation(result, params=None, max_lag=None, figsize=None):
    """Plot autocorrelation function and mark Sokal window for each parameter.

    Parameters
    ----------
    result : Posterior
        Inference result with samples.
    params : list of str, optional
        Parameters to plot. Default: all scalar params (excluding psd_xi).
    max_lag : int, optional
        Maximum lag to display. Default: min(500, n_samples // 2).
    figsize : tuple, optional
        Figure size. Default auto-scaled from number of parameters.

    Returns
    -------
    fig : matplotlib Figure

    Examples
    --------
    .. code-block:: python

        from tengri import plot_autocorrelation

        fig = plot_autocorrelation(posterior, params=["sfh_dpl_alpha", "dust_tau_bc"])
        fig.savefig("autocorr.pdf")
    """
    if result.samples is None:
        raise ValueError("ACF plot requires samples (not MAP)")

    acfs = result.autocorrelation(max_lag=max_lag)
    act_info = result.autocorrelation_time()

    if params is not None:
        acfs = {k: v for k, v in acfs.items() if k in params}
        act_info = {k: v for k, v in act_info.items() if k in params}

    n_params = len(acfs)
    if n_params == 0:
        raise ValueError("No scalar parameters to plot")

    ncols = min(3, n_params)
    nrows = (n_params + ncols - 1) // ncols
    if figsize is None:
        figsize = (4 * ncols, 3 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.ravel()

    for idx, (name, acf) in enumerate(sorted(acfs.items())):
        ax = axes_flat[idx]
        lags = np.arange(len(acf))

        # Truncate display at max_lag or where ACF is negligible
        display_max = min(len(acf), max_lag or 500)
        ax.plot(lags[:display_max], acf[:display_max], color="C0", lw=1.0)

        # Mark Sokal window (5τ) if we have ACT info
        if name in act_info:
            tau = act_info[name]["tau_max"]
            window = min(5 * tau, display_max)
            ax.axvline(window, color="C3", ls="--", lw=1.0, alpha=0.7, label=f"5τ = {window:.0f}")
            ax.set_title(f"{name}  (τ={tau:.1f}, ESS={act_info[name]['ess']:.0f})", fontsize=9)
        else:
            ax.set_title(name, fontsize=9)

        ax.axhline(0, color="gray", ls="-", lw=0.5, alpha=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("ACF")
        ax.legend(fontsize=10)

    # Hide unused axes
    for idx in range(n_params, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    return fig
