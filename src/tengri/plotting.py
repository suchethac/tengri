"""Plotting helper functions for tengri.

Reusable, publication-quality plotting utilities for SFH recovery,
SED fitting, spectroscopic fits, corner plot comparisons, and
diagnostics tables. Style configuration (rcParams) is left to the
caller — notebooks set their own style via ``setup_style()``.

Inspired by BAGPIPES (Carnall+2018), Prospector (Johnson+2021),
and CIGALE (Boquien+2019). Designed for ApJ/MNRAS figures.

Usage::

    from tengri.plotting import plot_sfh, plot_sed_fit, safe_corner
    from tengri.plotting import COLORS, SDSS_WAVE_EFF
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════
# Color palette — colorblind-safe, print-friendly
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    # Sampler colors (consistent across all notebooks)
    "map": "#888888",  # grey — point estimate
    "rt": "#1f77b4",  # blue — Ray Tracing (exact MCMC)
    # Canonical names
    "vi": "#ff7f0e",  # orange — geoVI (variational)
    "vi_linear": "#9467bd",  # purple — MGVI (linear VI)
    "mcmc_nuts": "#2ca02c",  # green — NUTS (gold standard)
    "mcmc_raytrace": "#1f77b4",  # blue — Ray Tracing (exact MCMC)
    # Legacy names (deprecated but still supported)
    "geovi": "#ff7f0e",  # orange — geoVI (variational)
    "nuts": "#2ca02c",  # green — NUTS (gold standard)
    "mgvi": "#9467bd",  # purple — MGVI (linear VI)
    # Data colors
    "truth": "#1a1a1a",  # near-black — ground truth
    "data": "#333333",  # dark grey — observed data
    "model": "#d62728",  # red — model prediction
    # SFH components
    "sfh_mean": "#1f77b4",  # blue — mean SFH backbone
    "sfh_full": "#ff7f0e",  # orange — full SFH (mean + GP)
    "sfh_gp": "#2ca02c",  # green — GP contribution
    # Band colors (SDSS)
    "u": "#7b3294",
    "g": "#008837",
    "r": "#d73027",
    "i": "#fc8d59",
    "z": "#4575b4",
    # Sequential for progressive reveal
    "seq": ["#d4d4d4", "#a8a8a8", "#1f77b4", "#2ca02c", "#d62728"],
}

# Named sampler styles for consistent legends
SAMPLER_STYLE = {
    "MAP": {"color": COLORS["map"], "ls": "--", "lw": 1.5, "alpha": 1.0},
    "RT": {"color": COLORS["rt"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    # Canonical names
    "VI": {"color": COLORS["vi"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    "VI_Linear": {"color": COLORS["vi_linear"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    "MCMC_NUTS": {"color": COLORS["mcmc_nuts"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    # Legacy names (deprecated but still supported)
    "geoVI": {"color": COLORS["geovi"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    "NUTS": {"color": COLORS["nuts"], "ls": "-", "lw": 1.5, "alpha": 1.0},
    "MGVI": {"color": COLORS["mgvi"], "ls": "-", "lw": 1.5, "alpha": 1.0},
}

# SDSS effective wavelengths (Angstrom)
SDSS_BANDS = {"u": 3551, "g": 4686, "r": 6166, "i": 7480, "z": 8932}
SDSS_WAVE_EFF = np.array([3551, 4686, 6166, 7480, 8932])
SDSS_BAND_NAMES = ["u", "g", "r", "i", "z"]
SDSS_BAND_COLORS = [COLORS["u"], COLORS["g"], COLORS["r"], COLORS["i"], COLORS["z"]]


# ═══════════════════════════════════════════════════════════════════
# Style setup
# ═══════════════════════════════════════════════════════════════════


def setup_style():
    """Configure matplotlib for publication-quality astronomy figures.

    Follows BAGPIPES (Carnall+2018) styling closely:
    - Large axis labels (18pt) and tick labels (14pt)
    - Thick lines (2pt data, 1.5pt axes)
    - Inward ticks on all four sides
    - No frame on legends
    """
    plt.rcParams.update(
        {
            # Figure
            "figure.dpi": 150,
            "figure.facecolor": "white",
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
            # Font — BAGPIPES uses Helvetica/sans-serif; we use DejaVu Serif
            # for journal compatibility without requiring LaTeX installation
            "font.size": 14,
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            # Axes labels — BAGPIPES: 18pt labels, 14pt ticks
            "axes.labelsize": 18,
            "axes.titlesize": 16,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 12,
            # Axes frame — BAGPIPES: 1.5pt
            "axes.linewidth": 1.5,
            "axes.grid": False,
            # Ticks — BAGPIPES: inward, all four sides
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.width": 0.7,
            "ytick.minor.width": 0.7,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            # Legend — BAGPIPES: no frame
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            # Lines — BAGPIPES: 2pt
            "lines.linewidth": 2.0,
        }
    )


# ═══════════════════════════════════════════════════════════════════
# SFH plotting (BAGPIPES-inspired)
# ═══════════════════════════════════════════════════════════════════


def plot_sfh(
    model,
    posterior,
    true_params=None,
    ax=None,
    color=None,
    label="Posterior",
    method="RT",
    show_draws=True,
    n_draws=30,
    ci_levels=(16, 84),
    xlim=(0, 13.5),
    show_mean_sfh=True,
):
    """Plot SFH posterior with uncertainty band — BAGPIPES/Prospector style.

    Features:
    - Filled 68% credible interval (not sample spaghetti)
    - Optional faint sample draws underneath
    - Truth as solid black line
    - Mean SFH backbone as dashed line
    - Clean axis labels with proper units

    Parameters
    ----------
    model : Model
    posterior : Posterior
    true_params : dict, optional
    ax : Axes, optional
    color : str, optional — defaults to sampler color
    label : str
    method : str — "RT", "geoVI", "NUTS", "MAP" for auto-coloring
    show_draws : bool — show faint sample lines
    n_draws : int — number of sample draws
    ci_levels : tuple — percentile levels for fill
    xlim : tuple — x-axis limits in Gyr
    show_mean_sfh : bool — show the smooth backbone as dashed

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    if color is None:
        color = SAMPLER_STYLE.get(method, {}).get("color", COLORS["rt"])

    if posterior.samples is not None:
        # Compute SFH draws
        n_total = len(next(iter(posterior.samples.values())))
        sfh_draws = []
        for i in range(n_total):
            s_i = {k: posterior.samples[k][i] for k in posterior.samples}
            sfh_i = model.predict_sfh(s_i)
            key = "sfr_full" if model.spec.stochastic else "sfr_mean"
            sfh_draws.append(sfh_i[key])

        sfh_arr = np.array(sfh_draws)
        t_gyr = np.array(model.predict_sfh(posterior.params)["t_gyr"])

        # Credible interval fill
        lo = np.percentile(sfh_arr, ci_levels[0], axis=0)
        hi = np.percentile(sfh_arr, ci_levels[1], axis=0)
        median = np.median(sfh_arr, axis=0)

        ax.fill_between(
            t_gyr, lo, hi, color=color, alpha=0.25, edgecolor="none", label=f"{label} (68% CI)"
        )

        # Faint sample draws (Prospector style)
        if show_draws:
            idx = np.linspace(0, n_total - 1, min(n_draws, n_total), dtype=int)
            for j in idx:
                ax.plot(t_gyr, sfh_arr[j], color=color, alpha=0.06, lw=0.5)

        # Median line
        ax.plot(t_gyr, median, color=color, lw=1.8, zorder=4)

    else:
        # MAP: point estimate
        sfh = model.predict_sfh(posterior.params)
        t_gyr = np.array(sfh["t_gyr"])
        key = "sfr_full" if model.spec.stochastic else "sfr_mean"
        ax.plot(t_gyr, sfh[key], color=COLORS["map"], lw=1.8, ls="--", label="MAP", zorder=3)

    # Truth overlay (BAGPIPES convention: solid black)
    if true_params is not None:
        sfh_true = model.predict_sfh(true_params)
        key = "sfr_full" if model.spec.stochastic else "sfr_mean"
        ax.plot(
            sfh_true["t_gyr"],
            sfh_true[key],
            color=COLORS["truth"],
            lw=2.0,
            label="Truth",
            zorder=10,
        )
        # Show mean SFH backbone for stochastic models
        if model.spec.stochastic and show_mean_sfh:
            ax.plot(
                sfh_true["t_gyr"],
                sfh_true["sfr_mean"],
                color=COLORS["truth"],
                lw=1.0,
                ls=":",
                alpha=0.4,
            )

    # BAGPIPES convention: lookback time with present at right
    ax.set_xlabel(r"$\mathrm{Lookback\ time\ /\ Gyr}$")
    ax.set_ylabel(r"$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")
    ax.set_xlim(xlim[1], xlim[0])  # reversed: high lookback at left, present at right
    ax.set_ylim(bottom=0.0)
    ax.legend(loc="upper left")

    return ax


def plot_sfh_comparison(model, results, true_params=None, methods=None, figsize=(15, 4)):
    """Side-by-side SFH recovery for multiple methods.

    BAGPIPES-style multi-panel layout.

    Parameters
    ----------
    model : Model
    results : dict of {method_name: Posterior}
    true_params : dict, optional
    methods : list of str, optional — order of panels
    figsize : tuple

    Returns
    -------
    fig, axes
    """
    if methods is None:
        methods = list(results.keys())
    n = len(methods)

    fig, axes = plt.subplots(1, n, figsize=figsize, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        plot_sfh(
            model, results[method], true_params=true_params, ax=ax, method=method, label=method
        )
        ax.set_title(method, fontsize=12, fontweight="bold")
        if ax != axes[0]:
            ax.set_ylabel("")

    fig.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════
# SED / Photometry plotting (Prospector-inspired)
# ═══════════════════════════════════════════════════════════════════


def plot_sed_fit(
    wave_eff,
    flux_obs,
    noise,
    flux_true=None,
    posterior_draws=None,
    ax=None,
    band_names=None,
    show_residuals=True,
):
    """Plot observed photometry with model fit — Prospector style.

    Parameters
    ----------
    wave_eff : array — effective wavelengths (Angstrom)
    flux_obs : array — observed fluxes
    noise : array — 1-sigma uncertainties
    flux_true : array, optional — true fluxes
    posterior_draws : array, optional — shape (n_draws, n_bands)
    ax : Axes, optional
    band_names : list of str, optional
    show_residuals : bool

    Returns
    -------
    fig or ax
    """
    if show_residuals:
        fig = plt.figure(figsize=(8, 5))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_main)
    else:
        if ax is None:
            fig, ax_main = plt.subplots(figsize=(8, 4))
        else:
            ax_main = ax
            fig = ax.figure

    # Data points with error bars
    ax_main.errorbar(
        wave_eff,
        flux_obs,
        yerr=noise,
        fmt="o",
        ms=7,
        color=COLORS["data"],
        capsize=3,
        capthick=1.0,
        elinewidth=1.0,
        zorder=5,
        label="Observed",
    )

    # True fluxes
    if flux_true is not None:
        ax_main.scatter(
            wave_eff,
            flux_true,
            marker="D",
            s=40,
            facecolors="none",
            edgecolors=COLORS["truth"],
            linewidths=1.2,
            zorder=6,
            label="Truth",
        )

    # Posterior predictive draws (faint lines)
    if posterior_draws is not None:
        for draw in posterior_draws:
            ax_main.plot(wave_eff, draw, "-", color=COLORS["rt"], alpha=0.08, lw=0.8)
        median_pred = np.median(posterior_draws, axis=0)
        ax_main.plot(
            wave_eff, median_pred, "s", ms=5, color=COLORS["rt"], zorder=4, label="Model (median)"
        )

    ax_main.set_ylabel(r"$f_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
    ax_main.legend(loc="upper right")

    # Band labels
    if band_names is not None:
        for weff, bname, bcol in zip(wave_eff, band_names, SDSS_BAND_COLORS):
            ax_main.annotate(
                bname,
                (weff, flux_obs[list(wave_eff).index(weff)]),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8,
                color=bcol,
            )

    if show_residuals and posterior_draws is not None:
        median_pred = np.median(posterior_draws, axis=0)
        residuals = (flux_obs - median_pred) / noise
        ax_res.axhline(0, color="0.5", ls="--", lw=0.8)
        ax_res.axhspan(-1, 1, alpha=0.05, color="0.5")
        ax_res.axhspan(-2, 2, alpha=0.03, color="0.5")
        colors = [
            SDSS_BAND_COLORS[i] if i < len(SDSS_BAND_COLORS) else COLORS["data"]
            for i in range(len(wave_eff))
        ]
        ax_res.bar(wave_eff, residuals, width=200, color=colors, alpha=0.7)
        ax_res.set_xlabel(r"Wavelength ($\AA$)")
        ax_res.set_ylabel(r"$(d - f)/\sigma$")
        ax_res.set_ylim(-4, 4)
        plt.setp(ax_main.get_xticklabels(), visible=False)
        return fig
    else:
        ax_main.set_xlabel(r"Wavelength ($\AA$)")
        return ax_main


# ═══════════════════════════════════════════════════════════════════
# Spectrum plotting
# ═══════════════════════════════════════════════════════════════════


def plot_spectrum_fit(
    wave_obs, spec_obs, noise, spec_true=None, spec_draws=None, features=None, z=0.1
):
    """Plot spectroscopic fit with residuals — CIGALE style.

    Parameters
    ----------
    wave_obs : array — observed wavelength grid (Angstrom)
    spec_obs : array — observed spectrum
    noise : array — 1-sigma uncertainties
    spec_true : array, optional — true spectrum
    spec_draws : array, optional — posterior draws, shape (n, n_pix)
    features : dict, optional — {name: rest_wavelength} for annotation
    z : float — redshift for feature marking

    Returns
    -------
    fig
    """
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax_main = fig.add_subplot(gs[0])
    ax_res = fig.add_subplot(gs[1], sharex=ax_main)

    # Data
    ax_main.plot(wave_obs, spec_obs, color="0.6", lw=0.5, alpha=0.7)
    ax_main.fill_between(
        np.array(wave_obs),
        np.array(spec_obs - noise),
        np.array(spec_obs + noise),
        color="0.8",
        alpha=0.3,
    )

    # Truth
    if spec_true is not None:
        ax_main.plot(wave_obs, spec_true, color=COLORS["truth"], lw=1.0, alpha=0.5, label="Truth")

    # Posterior
    if spec_draws is not None:
        median = np.median(spec_draws, axis=0)
        lo, hi = np.percentile(spec_draws, [16, 84], axis=0)
        ax_main.fill_between(np.array(wave_obs), lo, hi, color=COLORS["rt"], alpha=0.2)
        ax_main.plot(wave_obs, median, color=COLORS["rt"], lw=1.2, label="Model (68% CI)")

        # Residuals
        residuals = (np.array(spec_obs) - median) / np.array(noise)
        ax_res.plot(wave_obs, residuals, color=COLORS["rt"], lw=0.5)
        ax_res.axhline(0, color="0.5", ls="--", lw=0.8)
        ax_res.axhspan(-2, 2, alpha=0.05, color="0.5")
        ax_res.set_ylim(-4, 4)
        ax_res.set_ylabel(r"$(d-f)/\sigma$")

    # Feature annotations
    if features is not None:
        for name, lam_rest in features.items():
            lam_obs = lam_rest * (1 + z)
            if float(wave_obs[0]) <= lam_obs <= float(wave_obs[-1]):
                for ax in [ax_main, ax_res]:
                    ax.axvline(lam_obs, color="0.7", ls=":", lw=0.6)
                ax_main.text(
                    lam_obs,
                    ax_main.get_ylim()[1] * 0.95,
                    name,
                    fontsize=7,
                    ha="center",
                    color="0.4",
                    rotation=90,
                    va="top",
                )

    ax_res.set_xlabel(r"Observed wavelength ($\AA$)")
    ax_main.set_ylabel(r"$f_\nu$ (erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$)")
    ax_main.legend(loc="upper right")
    plt.setp(ax_main.get_xticklabels(), visible=False)

    return fig


# ═══════════════════════════════════════════════════════════════════
# Corner plot wrapper (safe against degenerate posteriors)
# ═══════════════════════════════════════════════════════════════════


def safe_corner(posterior, **kwargs):
    """Wrapper around Posterior.plot_corner that handles degenerate posteriors.

    Returns fig on success, None on failure.
    """
    try:
        return posterior.plot_corner(**kwargs)
    except (ValueError, np.linalg.LinAlgError) as e:
        print(f"Corner plot skipped: {e}")
        return None


def plot_corner_comparison(posteriors, labels, colors=None, truths=None, params=None):
    """Overlay multiple posteriors on a single corner plot.

    Parameters
    ----------
    posteriors : list of Posterior
    labels : list of str
    colors : list of str, optional
    truths : dict, optional
    params : list of str, optional

    Returns
    -------
    fig or None
    """
    if colors is None:
        default_colors = [
            COLORS["rt"],
            COLORS["geovi"],
            COLORS["nuts"],
            COLORS["mgvi"],
            COLORS["map"],
        ]
        colors = default_colors[: len(posteriors)]

    fig = None
    for post, label, color in zip(posteriors, labels, colors):
        try:
            fig = post.plot_corner(params=params, truths=truths, color=color, label=label, fig=fig)
        except (ValueError, np.linalg.LinAlgError):
            print(f"Corner plot skipped for {label}")
            continue

    return fig


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
# Common spectral features for annotation
# ═══════════════════════════════════════════════════════════════════

SPECTRAL_FEATURES = {
    r"Ly$\alpha$": 1216.0,
    "D4000": 4000.0,
    r"H$\delta$": 4102.0,
    r"H$\gamma$": 4340.0,
    r"H$\beta$": 4861.0,
    "[O III]": 5007.0,
    "Mg b": 5175.0,
    "Na D": 5893.0,
    r"H$\alpha$": 6563.0,
}


# ═══════════════════════════════════════════════════════════════════
# Burstiness plane (for NB01)
# ═══════════════════════════════════════════════════════════════════

# Galaxy type annotations for the sigma-tau grid
GALAXY_ANNOTATIONS = {
    (0, 0): "Dead elliptical",
    (0, 2): "Secular disk",
    (1, 1): "Normal SF galaxy",
    (2, 0): "Extreme dwarf",
    (2, 2): "Post-starburst",
}


# ═══════════════════════════════════════════════════════════════════
# Visual language specification — import from here for consistency
# ═══════════════════════════════════════════════════════════════════

SED_XLIM = (912, 1e7)  # Å, rest-frame
SED_XSCALE = "log"
SED_YLABEL = r"$\lambda F_\lambda$ (normalized at 5500 Å)"
SED_XLABEL = r"Rest-frame wavelength (Å)"

SFH_XLABEL = "Lookback time (Gyr)"
SFH_YLABEL = r"SFR (M$_\odot$ yr$^{-1}$)"

SWEEP_CMAPS = {
    "dust": "YlOrRd",  # yellow→red for reddening
    "agn": "PuRd",  # purple→red for AGN dominance
    "sfh": "Blues",  # light→dark for SFH variation
    "nebular": "Greens",  # for ionization
    "radio": "cool",  # blue→purple for radio
    "redshift": "plasma",  # for redshift sweeps
}

REFERENCE_STYLE = dict(color="0.75", lw=1.5, zorder=0, label="reference")


# ═══════════════════════════════════════════════════════════════════
# Sweep utilities — the engine behind all gallery notebooks
# ═══════════════════════════════════════════════════════════════════


def sweep_parameter(
    model,
    param_name: str,
    values,
    *,
    ax=None,
    cmap: str = "viridis",
    label_fmt: str = "{:.2f}",
    unit: str = "",
    log_scale: bool = False,
    components: bool = False,
    reference_idx: int | None = None,
    wave_range: tuple[float, float] | None = None,
    normalize_at: float | None = 5500.0,
) -> tuple:
    """Sweep one parameter across values, plot resulting SEDs colormapped low→high.

    Parameters
    ----------
    model : tengri.Model
        Model instance with a ``predict`` or ``sed`` callable.
    param_name : str
        Full parameter name (e.g. ``"dust_tau_bc"``).
    values : sequence of float
        Parameter values to sweep.
    ax : Axes, optional
        Existing axes to plot into. Creates new figure if None.
    cmap : str
        Matplotlib colormap name. Use SWEEP_CMAPS[key] for standard sweeps.
    label_fmt : str
        Format string for legend labels, e.g. ``"τ_BC = {:.1f}"``.
    unit : str
        Unit string appended to label, e.g. ``"K"``.
    log_scale : bool
        If True, log-scale the y-axis.
    components : bool
        If True, also overplot individual SED components as dashed lines.
    reference_idx : int or None
        Index into ``values`` to plot in REFERENCE_STYLE (gray). Others in cmap.
    wave_range : (lo, hi) or None
        Wavelength range in Å to plot. Defaults to SED_XLIM.
    normalize_at : float or None
        Normalize SEDs at this rest-frame wavelength (Å). None = no normalization.

    Returns
    -------
    fig, ax : Figure, Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    else:
        fig = ax.get_figure()

    cmap_obj = plt.get_cmap(cmap)
    n = len(values)
    colors = [cmap_obj(i / max(n - 1, 1)) for i in range(n)]

    xlim = wave_range if wave_range is not None else SED_XLIM

    for i, val in enumerate(values):
        # Override single parameter; use model defaults for rest
        override = {param_name: float(val)}
        try:
            wave, lnu = model.sed(override)
        except Exception:
            # Fallback: try predict interface
            pred = model.predict(override)
            wave = np.asarray(pred.wavelengths)
            lnu = np.asarray(pred.lnu)

        wave = np.asarray(wave)
        lnu = np.asarray(lnu)

        # Normalize
        if normalize_at is not None:
            idx_norm = int(np.argmin(np.abs(wave - normalize_at)))
            norm = lnu[idx_norm]
            if norm > 0:
                lnu = lnu / norm

        # λF_λ = ν F_ν ∝ (c/λ²) × F_ν ∝ F_ν / λ * const → for normalized SEDs use lnu * (c/λ)
        # approximate: plot lnu × wave (proportional to λ F_λ)
        y = lnu * wave

        mask = (wave >= xlim[0]) & (wave <= xlim[1])
        if mask.sum() == 0:
            continue

        style = REFERENCE_STYLE.copy() if i == reference_idx else {"color": colors[i], "lw": 1.8}
        label_str = label_fmt.format(val)
        if unit:
            label_str += f" {unit}"
        ax.plot(wave[mask], y[mask], label=label_str, **style)

    ax.set_xscale(SED_XSCALE)
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(SED_XLABEL)
    ax.set_ylabel(SED_YLABEL)
    ax.set_xlim(xlim)
    ax.legend(fontsize=10, ncol=2)
    return fig, ax


def parameter_gallery(
    model,
    param_sweep_specs: list[dict],
    *,
    ncols: int = 3,
    figsize_per_panel: tuple[float, float] = (4, 3),
) -> "plt.Figure":
    """Multi-panel gallery: one panel per entry in param_sweep_specs.

    Parameters
    ----------
    model : tengri.Model
    param_sweep_specs : list of dict
        Each dict: ``{"param": "dust_tau_bc", "values": [...], "label": "τ_BC",
        "cmap": "YlOrRd", "label_fmt": "{:.1f}"}``.
    ncols : int
        Number of columns in the grid.
    figsize_per_panel : (w, h)
        Size per panel in inches.

    Returns
    -------
    fig : Figure
    """
    n = len(param_sweep_specs)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_panel[0] * ncols, figsize_per_panel[1] * nrows),
        squeeze=False,
    )

    for idx, spec in enumerate(param_sweep_specs):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        sweep_parameter(
            model,
            spec["param"],
            spec["values"],
            ax=ax,
            cmap=spec.get("cmap", "viridis"),
            label_fmt=spec.get("label_fmt", "{:.2f}"),
            unit=spec.get("unit", ""),
        )
        ax.set_title(spec.get("label", spec["param"]), fontsize=12)

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig


def sfh_sed_comparison(
    model,
    param_name: str,
    values,
    *,
    cmap: str = "plasma",
    n_stochastic: int = 0,
    key=None,
) -> "plt.Figure":
    """Two-panel figure: SFH realizations (left) + corresponding SEDs (right).

    Essential for the SFH gallery and burstiness story.

    Parameters
    ----------
    model : tengri.Model
    param_name : str
        Parameter to sweep across ``values``.
    values : sequence
        Parameter values.
    cmap : str
        Colormap for the sweep.
    n_stochastic : int
        If > 0, draw this many stochastic SFH samples per value (thin, alpha=0.2).
    key : jax.random.PRNGKey, optional
        Required when n_stochastic > 0.

    Returns
    -------
    fig : Figure  (two-panel, figsize=(12, 4))
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_sfh, ax_sed = axes

    cmap_obj = plt.get_cmap(cmap)
    n = len(values)
    colors = [cmap_obj(i / max(n - 1, 1)) for i in range(n)]

    for i, val in enumerate(values):
        override = {param_name: float(val)}
        color = colors[i]
        label = f"{param_name} = {val:.2g}"

        # SFH panel
        try:
            t_lookback, sfr = model.sfh(override)
            ax_sfh.plot(np.asarray(t_lookback), np.asarray(sfr), color=color, lw=1.8, label=label)
        except Exception:
            pass  # model may not support sfh() — skip

        # SED panel
        try:
            wave, lnu = model.sed(override)
            wave = np.asarray(wave)
            lnu = np.asarray(lnu)
            idx_norm = int(np.argmin(np.abs(wave - 5500.0)))
            norm = lnu[idx_norm]
            if norm > 0:
                lnu = lnu / norm
            y = lnu * wave
            mask = (wave >= SED_XLIM[0]) & (wave <= SED_XLIM[1])
            ax_sed.plot(wave[mask], y[mask], color=color, lw=1.8, label=label)
        except Exception:
            pass

    ax_sfh.set_xlabel(SFH_XLABEL)
    ax_sfh.set_ylabel(SFH_YLABEL)
    ax_sfh.legend(fontsize=9)

    ax_sed.set_xscale(SED_XSCALE)
    ax_sed.set_xlabel(SED_XLABEL)
    ax_sed.set_ylabel(SED_YLABEL)
    ax_sed.set_xlim(SED_XLIM)

    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════
# Convenience methods for MockData and Posterior
# ═══════════════════════════════════════════════════════════════════


def mock_plot(mock, ax=None):
    """Plot mock photometry and true SED.

    Convenience function for MockData.plot() — shows observed photometry
    with errorbars and the true noiseless SED as a line.

    Parameters
    ----------
    mock : MockData
        Mock observation from model.mock().
    ax : matplotlib Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    else:
        fig = ax.get_figure()

    # Index for photometric bands (x-axis as indices)
    n_bands = len(mock.flux_true)
    x_phot = np.arange(n_bands)

    # Plot true SED as a line connecting the noiseless fluxes
    ax.plot(x_phot, np.array(mock.flux_true), "-", color="gray", lw=2, label="True SED")

    # Plot observed photometry with noise errorbars
    ax.errorbar(
        x_phot,
        np.array(mock.flux_obs),
        yerr=np.array(mock.noise),
        fmt="o",
        color="k",
        ms=6,
        ecolor="k",
        capsize=3,
        label="Observed",
    )

    ax.set_xlabel("Band index")
    ax.set_ylabel("Flux density [erg/s/cm²/Hz]")
    ax.set_yscale("log")
    ax.legend(frameon=False)

    return fig, ax


def posterior_plot_sed(result, mock=None, ax=None):
    """Plot SED posterior with optional mock data.

    Convenience function for Posterior.plot_sed() — creates a two-panel
    figure showing the posterior SED band and the star formation history
    over lookback time.

    Parameters
    ----------
    result : Posterior
        Posterior inference result with model reference.
    mock : MockData, optional
        Mock observation to overlay. If provided, shows observed
        photometry with errorbars.
    ax : matplotlib Axes or array of Axes, optional
        Axes to plot on. If None, creates a new figure.

    Returns
    -------
    fig : matplotlib Figure
    axes : array of matplotlib Axes

    Raises
    ------
    RuntimeError
        If model reference is not available.
    """
    if result._model is None:
        raise RuntimeError("plot_sed() requires model reference")

    if ax is None:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    else:
        if isinstance(ax, np.ndarray):
            axes = ax
            fig = axes.flat[0].get_figure()
        else:
            axes = np.array([[ax]])
            fig = ax.get_figure()

    ax_sed = axes.flat[0]
    ax_sfh = axes.flat[1]

    # Left panel: SED
    if result.samples is not None:
        # Compute SED for each sample
        n_samples = next(iter(result.samples.values())).shape[0]
        sed_samples = []
        for i in range(n_samples):
            sample_i = {k: v[i] for k, v in result.samples.items()}
            sed_i = result._model.predict_rest_sed(sample_i).sed
            sed_samples.append(sed_i)
        sed_array = np.array(sed_samples)

        # Get wavelengths
        wave_rest = np.array(result._model.wavelengths)

        # Plot posterior band (16th to 84th percentile)
        sed_lo = np.percentile(sed_array, 16, axis=0)
        sed_hi = np.percentile(sed_array, 84, axis=0)
        ax_sed.fill_between(
            wave_rest, sed_lo, sed_hi, alpha=0.3, color="C0", label="68% credible region"
        )

        # Plot MAP SED
        sed_map = result._model.predict_rest_sed(result.params).sed
        ax_sed.plot(wave_rest, sed_map, "-", color="C0", lw=2, label="MAP SED")
    else:
        # MAP only: just plot the best fit
        wave_rest = np.array(result._model.wavelengths)
        sed_map = result._model.predict_rest_sed(result.params).sed
        ax_sed.plot(wave_rest, sed_map, "-", color="C0", lw=2, label="Best fit")

    # Overlay mock data if provided
    if mock is not None:
        n_bands = len(mock.flux_true)
        x_phot = np.arange(n_bands)
        ax_sed.errorbar(
            x_phot,
            np.array(mock.flux_obs),
            yerr=np.array(mock.noise),
            fmt="o",
            color="k",
            ms=6,
            ecolor="k",
            capsize=3,
            label="Observed",
        )

    ax_sed.set_xlabel("Wavelength [Å]")
    ax_sed.set_ylabel("Flux density [erg/s/cm²/Hz]")
    ax_sed.set_yscale("log")
    ax_sed.legend(frameon=False)

    # Right panel: SFH
    sfh_map = result._model.predict_sfh(result.params)
    t_gyr = np.array(sfh_map["t_gyr"])
    sfr_mean = np.array(sfh_map["sfr_mean"])

    if result.samples is not None:
        # Compute SFH for each sample
        sfh_samples = []
        for i in range(n_samples):
            sample_i = {k: v[i] for k, v in result.samples.items()}
            sfh_i = result._model.predict_sfh(sample_i)
            sfh_samples.append(np.array(sfh_i["sfr_mean"]))
        sfh_array = np.array(sfh_samples)

        # Plot posterior band
        sfr_lo = np.percentile(sfh_array, 16, axis=0)
        sfr_hi = np.percentile(sfh_array, 84, axis=0)
        ax_sfh.fill_between(
            t_gyr, sfr_lo, sfr_hi, alpha=0.3, color="C0", label="68% credible region"
        )

    ax_sfh.plot(t_gyr, sfr_mean, "-", color="C0", lw=2, label="Posterior median")

    # Overlay truth SFH if available in mock
    if mock is not None:
        try:
            sfh_true = result._model.predict_sfh(mock.params)
            t_gyr_true = np.array(sfh_true["t_gyr"])
            sfr_true = np.array(sfh_true["sfr_mean"])
            ax_sfh.plot(t_gyr_true, sfr_true, "--", color="k", lw=2, label="Truth")
        except Exception:
            pass

    ax_sfh.set_xlabel("Lookback time [Gyr]")
    ax_sfh.set_ylabel("SFR [M☉/yr]")
    ax_sfh.set_title("Star Formation History")
    ax_sfh.legend(frameon=False)

    plt.tight_layout()
    return fig, axes


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
        except Exception:
            pass

    ax.set_xlabel("Lookback time [Gyr]")
    ax.set_ylabel("SFR [M☉/yr]")
    ax.legend(frameon=False)

    return fig, ax
