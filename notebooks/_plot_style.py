"""Publication-quality plotting style and helper functions for tengri notebooks.

Inspired by BAGPIPES (Carnall+2018), Prospector (Johnson+2021), and CIGALE
(Boquien+2019) plotting conventions. Designed for astronomy journals (ApJ/MNRAS).

Usage in notebooks:
    from _plot_style import setup_style, plot_sfh, plot_sed, plot_corner_comparison
    setup_style()
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

# ═══════════════════════════════════════════════════════════════════
# Color palette — colorblind-safe, print-friendly
# ═══════════════════════════════════════════════════════════════════

COLORS = {
    # Sampler colors (consistent across all notebooks)
    "map":       "#888888",   # grey — point estimate
    "rt":        "#1f77b4",   # blue — Ray Tracing (exact MCMC)
    "geovi":     "#ff7f0e",   # orange — geoVI (variational)
    "nuts":      "#2ca02c",   # green — NUTS (gold standard)
    "mgvi":      "#9467bd",   # purple — MGVI (linear VI)
    "pathfinder": "#8c564b",  # brown — Pathfinder (approximate)
    "ess":       "#e377c2",   # pink — Elliptical Slice Sampling
    "laplace":   "#bcbd22",   # olive/yellow-green — Laplace
    # Data colors
    "truth":     "#1a1a1a",   # near-black — ground truth
    "data":      "#333333",   # dark grey — observed data
    "model":     "#d62728",   # red — model prediction
    # SFH components
    "sfh_mean":  "#1f77b4",   # blue — mean SFH backbone
    "sfh_full":  "#ff7f0e",   # orange — full SFH (mean + GP)
    "sfh_gp":    "#2ca02c",   # green — GP contribution
    # Band colors (SDSS)
    "u": "#7b3294", "g": "#008837", "r": "#d73027",
    "i": "#fc8d59", "z": "#4575b4",
    # Sequential for progressive reveal
    "seq": ["#d4d4d4", "#a8a8a8", "#1f77b4", "#2ca02c", "#d62728"],
}

# Named sampler styles for consistent legends
SAMPLER_STYLE = {
    "MAP":  {"color": COLORS["map"],   "ls": "--", "lw": 1.5, "alpha": 1.0},
    "RT":   {"color": COLORS["rt"],    "ls": "-",  "lw": 1.5, "alpha": 1.0},
    "geoVI":{"color": COLORS["geovi"], "ls": "-",  "lw": 1.5, "alpha": 1.0},
    "NUTS": {"color": COLORS["nuts"],  "ls": "-",  "lw": 1.5, "alpha": 1.0},
    "MGVI": {"color": COLORS["mgvi"],  "ls": "-",  "lw": 1.5, "alpha": 1.0},
    "Pathfinder": {"color": COLORS["pathfinder"], "ls": "-.", "lw": 1.5, "alpha": 1.0},
    "ESS":  {"color": COLORS["ess"],   "ls": "-",  "lw": 1.5, "alpha": 1.0},
    "Laplace": {"color": COLORS["laplace"], "ls": ":", "lw": 2.0, "alpha": 1.0},
}

# SDSS effective wavelengths (Angstrom)
SDSS_BANDS = {"u": 3551, "g": 4686, "r": 6166, "i": 7480, "z": 8932}
SDSS_WAVE_EFF = np.array([3551, 4686, 6166, 7480, 8932])
SDSS_BAND_NAMES = ["u", "g", "r", "i", "z"]
SDSS_BAND_COLORS = [COLORS["u"], COLORS["g"], COLORS["r"],
                     COLORS["i"], COLORS["z"]]


# ═══════════════════════════════════════════════════════════════════
# Style setup
# ═══════════════════════════════════════════════════════════════════

def setup_style():
    """Configure matplotlib for publication-quality astronomy figures.

    Uses SciencePlots 'science' base style + AAS journal requirements:
    - Min 6pt fonts, min 0.5pt lines (AAS mandatory)
    - Inward ticks on all four sides (astronomy convention)
    - Colorblind-safe palette (AAS recommended)
    - No chartjunk (Rougier et al. 2014 Rule 8)

    References:
        - SciencePlots (garrett403/SciencePlots)
        - AAS Graphics Guide (journals.aas.org/graphics-guide/)
        - Ten Simple Rules (Rougier et al. 2014, PLOS Comp Bio)
        - BAGPIPES (Carnall+2018) for astronomy conventions
    """
    # Use SciencePlots base + no-latex (avoids LaTeX dependency)
    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        pass  # fall back to manual configuration

    # Override with astronomy-specific settings
    plt.rcParams.update({
        # Figure — AAS: 300+ DPI for publication
        "figure.dpi": 150,
        "figure.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        # Font — AAS: Times/Helvetica/Symbol, min 6pt
        # SciencePlots sets serif; we keep it and increase sizes
        "font.size": 12,
        "font.family": "serif",
        "mathtext.fontset": "dejavuserif",
        # Axes labels — visible at journal column width
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        # Axes frame — AAS: min 0.5pt lines
        "axes.linewidth": 1.0,
        "axes.grid": False,
        # Ticks — astronomy convention: inward, all four sides
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        # Legend — no frame, compact
        "legend.frameon": False,
        "legend.handlelength": 1.5,
        "legend.borderpad": 0.3,
        "legend.labelspacing": 0.3,
        # Lines — visible at journal scale (AAS: min 0.5pt)
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        # Errorbar
        "errorbar.capsize": 3,
    })


# ═══════════════════════════════════════════════════════════════════
# SFH plotting (BAGPIPES-inspired)
# ═══════════════════════════════════════════════════════════════════

def plot_sfh(model, posterior, true_params=None, ax=None,
             color=None, label="Posterior", method="RT",
             show_draws=True, n_draws=30, ci_levels=(16, 84),
             xlim=(0, 13.5), show_mean_sfh=True):
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

        ax.fill_between(t_gyr, lo, hi, color=color, alpha=0.25,
                         edgecolor="none", lw=0,
                         label=f"{label} (68% CI)", zorder=3,
                         rasterized=True)

        # Faint sample draws (Prospector style)
        if show_draws:
            idx = np.linspace(0, n_total - 1, min(n_draws, n_total), dtype=int)
            for j in idx:
                ax.plot(t_gyr, sfh_arr[j], color=color, alpha=0.06, lw=0.5,
                        rasterized=True)

        # Median line (always on top — BAGPIPES convention)
        ax.plot(t_gyr, median, color=color, lw=1.8, zorder=4)

    else:
        # MAP: point estimate
        sfh = model.predict_sfh(posterior.params)
        t_gyr = np.array(sfh["t_gyr"])
        key = "sfr_full" if model.spec.stochastic else "sfr_mean"
        ax.plot(t_gyr, sfh[key], color=COLORS["map"], lw=1.8,
                ls="--", label="MAP", zorder=3)

    # Truth overlay (BAGPIPES convention: solid black)
    if true_params is not None:
        sfh_true = model.predict_sfh(true_params)
        key = "sfr_full" if model.spec.stochastic else "sfr_mean"
        ax.plot(sfh_true["t_gyr"], sfh_true[key], color=COLORS["truth"],
                lw=3.0, label="Truth", zorder=10)
        # Show mean SFH backbone for stochastic models
        if model.spec.stochastic and show_mean_sfh:
            ax.plot(sfh_true["t_gyr"], sfh_true["sfr_mean"],
                    color=COLORS["truth"], lw=1.0, ls=":", alpha=0.4)

    # Lookback time: 0 (present) at left, high lookback at right
    ax.set_xlabel(r"$\mathrm{Lookback\ time\ /\ Gyr}$")
    ax.set_ylabel(r"$\mathrm{SFR\ /\ M_\odot\ yr^{-1}}$")
    ax.set_xlim(xlim[0], xlim[1])

    # Sensible ylim: bottom at 0, top at 1.5× the 84th percentile peak
    ax.set_ylim(bottom=0.)
    if posterior.samples is not None and len(sfh_draws) > 0:
        hi_peak = np.max(np.percentile(sfh_arr, 84, axis=0))
        if hi_peak > 0:
            ax.set_ylim(top=hi_peak * 1.5)

    ax.legend(loc="upper right")

    # 200 Myr inset (always for stochastic models)
    if posterior.samples is not None and model.spec.stochastic:
        ax_in = add_sfh_inset(ax, t_gyr, np.median(sfh_arr, axis=0),
                              color=color, lw=1.2)
        if true_params is not None:
            sfh_t = model.predict_sfh(true_params)
            t_key = "sfr_full" if model.spec.stochastic else "sfr_mean"
            t_myr = np.asarray(sfh_t["t_gyr"]) * 1e3
            mask = t_myr <= 200
            if mask.sum() > 2:
                ax_in.plot(t_myr[mask], np.asarray(sfh_t[t_key])[mask],
                           color=COLORS["truth"], lw=1.5)

    return ax


def add_sfh_inset(ax, t_gyr, sfr, inset_range_myr=200, **kwargs):
    """Add a zoom inset showing recent SFH (last 200 Myr by default).

    Parameters
    ----------
    ax : matplotlib Axes
        Parent axes to attach the inset to.
    t_gyr : array
        Lookback time in Gyr.
    sfr : array
        Star formation rate.
    inset_range_myr : float
        Maximum lookback time to show in the inset (Myr).
    **kwargs
        Passed to ax_in.plot().

    Returns
    -------
    ax_in : matplotlib Axes
        The inset axes (for further customization).
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    ax_in = inset_axes(ax, width="35%", height="40%", loc="upper right",
                       borderpad=1.5)
    t_myr = np.asarray(t_gyr) * 1e3
    mask = t_myr <= inset_range_myr
    if mask.sum() > 2:
        ax_in.plot(t_myr[mask], np.asarray(sfr)[mask], **kwargs)
    ax_in.set_xlim(0, inset_range_myr)
    ax_in.set_xlabel("Lookback (Myr)", fontsize=6)
    ax_in.set_ylabel("SFR", fontsize=6)
    ax_in.tick_params(labelsize=5)
    return ax_in


def plot_sfh_comparison(model, results, true_params=None,
                        methods=None, figsize=(15, 4)):
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
        plot_sfh(model, results[method], true_params=true_params,
                 ax=ax, method=method, label=method)
        ax.set_title(method, fontsize=12, fontweight="bold")
        if ax != axes[0]:
            ax.set_ylabel("")

    fig.tight_layout()
    return fig, axes


# ═══════════════════════════════════════════════════════════════════
# SED / Photometry plotting (Prospector-inspired)
# ═══════════════════════════════════════════════════════════════════

def plot_sed_fit(wave_eff, flux_obs, noise, flux_true=None,
                 posterior_draws=None, ax=None, band_names=None,
                 show_residuals=True):
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
    ax_main.errorbar(wave_eff, flux_obs, yerr=noise,
                      fmt="o", ms=7, color=COLORS["data"],
                      capsize=3, capthick=1.0, elinewidth=1.0,
                      zorder=5, label="Observed")

    # True fluxes
    if flux_true is not None:
        ax_main.scatter(wave_eff, flux_true, marker="D", s=40,
                         facecolors="none", edgecolors=COLORS["truth"],
                         linewidths=1.2, zorder=6, label="Truth")

    # Posterior predictive draws (faint lines)
    if posterior_draws is not None:
        for draw in posterior_draws:
            ax_main.plot(wave_eff, draw, "-", color=COLORS["rt"],
                          alpha=0.08, lw=0.8)
        median_pred = np.median(posterior_draws, axis=0)
        ax_main.plot(wave_eff, median_pred, "s", ms=5,
                      color=COLORS["rt"], zorder=4, label="Model (median)")

    ax_main.set_ylabel(r"$f_\nu$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)")
    ax_main.legend(loc="upper right")

    # Band labels
    if band_names is not None:
        for weff, bname, bcol in zip(wave_eff, band_names, SDSS_BAND_COLORS):
            ax_main.annotate(bname, (weff, flux_obs[list(wave_eff).index(weff)]),
                              textcoords="offset points", xytext=(0, 12),
                              ha="center", fontsize=8, color=bcol)

    if show_residuals and posterior_draws is not None:
        median_pred = np.median(posterior_draws, axis=0)
        residuals = (flux_obs - median_pred) / noise
        ax_res.axhline(0, color="0.5", ls="--", lw=0.8)
        ax_res.axhspan(-1, 1, alpha=0.05, color="0.5")
        ax_res.axhspan(-2, 2, alpha=0.03, color="0.5")
        colors = [SDSS_BAND_COLORS[i] if i < len(SDSS_BAND_COLORS)
                  else COLORS["data"] for i in range(len(wave_eff))]
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

def plot_spectrum_fit(wave_obs, spec_obs, noise, spec_true=None,
                      spec_draws=None, features=None, z=0.1):
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
    ax_main.fill_between(np.array(wave_obs),
                          np.array(spec_obs - noise),
                          np.array(spec_obs + noise),
                          color="0.8", alpha=0.3)

    # Truth
    if spec_true is not None:
        ax_main.plot(wave_obs, spec_true, color=COLORS["truth"],
                      lw=1.0, alpha=0.5, label="Truth")

    # Posterior
    if spec_draws is not None:
        median = np.median(spec_draws, axis=0)
        lo, hi = np.percentile(spec_draws, [16, 84], axis=0)
        ax_main.fill_between(np.array(wave_obs), lo, hi,
                              color=COLORS["rt"], alpha=0.2)
        ax_main.plot(wave_obs, median, color=COLORS["rt"],
                      lw=1.2, label="Model (68% CI)")

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
                ax_main.text(lam_obs, ax_main.get_ylim()[1] * 0.95, name,
                              fontsize=7, ha="center", color="0.4",
                              rotation=90, va="top")

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


def plot_corner_comparison(posteriors, labels, colors=None,
                           truths=None, params=None):
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
        default_colors = [COLORS["rt"], COLORS["geovi"], COLORS["nuts"],
                          COLORS["mgvi"], COLORS["map"]]
        colors = default_colors[:len(posteriors)]

    fig = None
    for post, label, color in zip(posteriors, labels, colors):
        try:
            fig = post.plot_corner(params=params, truths=truths,
                                    color=color, label=label, fig=fig)
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

    header = f"{'Method':<15} {'Wall time':>10} {'ESS (min)':>10} {'ESS (med)':>10} {'Accept %':>10}"
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

        accept = res.diagnostics.get("accept_rate_post_burnin",
                 res.diagnostics.get("mean_accept_prob", None))
        accept_str = f"{accept:.1%}" if accept is not None else "—"
        print(f"{name:<15} {wt:>10} {ess_min:>10} {ess_med:>10} {accept_str:>10}")


# ═══════════════════════════════════════════════════════════════════
# Convergence diagnostics (industry standard: Stan/ArviZ/BlackJAX)
# ═══════════════════════════════════════════════════════════════════

# Thresholds following Vehtari et al. (2021) "Rank-normalization,
# folding, and localization" and Stan/ArviZ conventions.
CONVERGENCE_THRESHOLDS = {
    "ess_bulk_min": 100,       # minimum bulk ESS per parameter
    "ess_total_target": 400,   # target total ESS for reliable summaries
    "divergence_warn": 0,      # any divergence warrants investigation
    "divergence_fail_pct": 5,  # >5% divergences = serious problem
    "accept_rt_lo": 0.20,     # RT acceptance too low = step_size too large
    "accept_rt_hi": 0.90,     # RT acceptance too high = barely moving
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
    method_name : str
        Label for printing (e.g., "RT", "NUTS", "geoVI").
    verbose : bool
        If True, print detailed diagnostics.

    Returns
    -------
    dict
        Keys: 'converged' (bool), 'warnings' (list of str),
        'ess_min', 'ess_median', 'n_params_low_ess'.
    """
    warnings = []
    info = {}
    th = CONVERGENCE_THRESHOLDS

    name = method_name or result.diagnostics.get("method", "Sampler")

    # --- ESS ---
    if result.samples is not None:
        ess = result.effective_sample_size()
        # Exclude GP latent vector from summary (too many params)
        ess_phys = {k: v for k, v in ess.items()
                    if not k.startswith("psd_xi")}
        if ess_phys:
            ess_vals = list(ess_phys.values())
            ess_min = min(ess_vals)
            ess_med = float(np.median(ess_vals))
            n_low = sum(1 for v in ess_vals if v < th["ess_bulk_min"])

            info["ess_min"] = ess_min
            info["ess_median"] = ess_med
            info["n_params_low_ess"] = n_low

            if ess_min < th["ess_bulk_min"]:
                low_params = [k for k, v in ess_phys.items()
                              if v < th["ess_bulk_min"]]
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
    else:
        info["ess_min"] = None
        info["ess_median"] = None
        info["n_params_low_ess"] = None

    # --- Divergences (NUTS) ---
    n_div = result.diagnostics.get("n_divergent", None)
    if n_div is not None:
        n_samples = result.diagnostics.get("n_samples", 1)
        div_pct = 100 * n_div / max(n_samples, 1)
        info["n_divergent"] = n_div
        info["divergence_pct"] = div_pct
        if n_div > th["divergence_warn"]:
            severity = "SERIOUS" if div_pct > th["divergence_fail_pct"] else "WARNING"
            warnings.append(
                f"{severity}: {n_div}/{n_samples} divergent transitions "
                f"({div_pct:.1f}%) — posterior may be unreliable"
            )

    # --- Acceptance rate (RT) ---
    accept = result.diagnostics.get("accept_rate_post_burnin", None)
    if accept is not None:
        info["acceptance_rate"] = accept
        if accept < th["accept_rt_lo"]:
            warnings.append(
                f"RT acceptance {accept:.0%} too low — reduce step_size"
            )
        elif accept > th["accept_rt_hi"]:
            warnings.append(
                f"RT acceptance {accept:.0%} too high — chain barely moving, "
                f"increase step_size"
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
            print(f"  ESS (min / median): {info['ess_min']:.0f} / "
                  f"{info['ess_median']:.0f}")
        if "acceptance_rate" in info:
            print(f"  Acceptance rate:    {info['acceptance_rate']:.1%}")
        if "n_divergent" in info:
            print(f"  Divergences:        {info['n_divergent']} / "
                  f"{result.diagnostics.get('n_samples', '?')}")
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
    results_dict : dict of {name: Posterior}
    verbose : bool

    Returns
    -------
    dict of {name: convergence_info}
    """
    all_info = {}
    for name, res in results_dict.items():
        all_info[name] = convergence_check(res, method_name=name,
                                           verbose=False)

    if verbose:
        # Compact table
        header = (f"{'Method':<15} {'ESS min':>8} {'ESS med':>8} "
                  f"{'Accept':>8} {'Diverg':>8} {'Status':>10}")
        print(header)
        print("-" * len(header))
        for name, info in all_info.items():
            ess_min = (f"{info['ess_min']:.0f}"
                       if info.get("ess_min") is not None else "—")
            ess_med = (f"{info['ess_median']:.0f}"
                       if info.get("ess_median") is not None else "—")
            accept = (f"{info['acceptance_rate']:.0%}"
                      if "acceptance_rate" in info else "—")
            diverg = (f"{info['n_divergent']}"
                      if "n_divergent" in info else "—")
            status = "OK" if info["converged"] else "WARN"
            print(f"{name:<15} {ess_min:>8} {ess_med:>8} "
                  f"{accept:>8} {diverg:>8} {status:>10}")

        # Print warnings below
        any_warns = any(info["warnings"] for info in all_info.values())
        if any_warns:
            print("\nWarnings:")
            for name, info in all_info.items():
                for w in info["warnings"]:
                    print(f"  [{name}] {w}")

    return all_info


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
