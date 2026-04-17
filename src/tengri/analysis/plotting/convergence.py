"""Convergence diagnostics and posterior summary visualization.

Utilities for monitoring MCMC/VI convergence and reporting effective sample sizes,
acceptance rates, and other diagnostic metrics.
"""

import numpy as np

# ═══════════════════════════════════════════════════════════════════
# Diagnostics table
# ═══════════════════════════════════════════════════════════════════


def diagnostics_table(results, names=None):
    """Print a formatted diagnostics comparison table.

    Parameters
    ----------
    results : dict of {method_name: Posterior}
    names : list of str, optional — order
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
    import matplotlib.pyplot as plt

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
