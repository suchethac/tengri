# SPDX-License-Identifier: BSD-3-Clause
"""SED, spectrum, and parameter sweep plotting utilities.

Publication-quality spectral energy distribution visualization for photometry,
spectroscopy, and parameter sweeps. Inspired by Prospector (Johnson+2021)
and CIGALE (Boquien+2019).
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from .styles import (
    COLORS,
    REFERENCE_STYLE,
    SDSS_BAND_COLORS,
    SED_XLABEL,
    SED_XLIM,
    SED_XSCALE,
    SED_YLABEL,
    SFH_XLABEL,
    SFH_YLABEL,
)

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
    """Plot observed photometry with model fit: Prospector style.

    Parameters
    ----------
    wave_eff : array_like, shape (n_bands,)
        Effective filter wavelengths. [Angstrom]
    flux_obs : array_like, shape (n_bands,)
        Observed photometric fluxes. [erg/s/cm²/Hz]
    noise : array_like, shape (n_bands,)
        1-sigma photometric uncertainties. [erg/s/cm²/Hz]
    flux_true : array_like, shape (n_bands,), optional
        True (noiseless) fluxes for mock recovery plots.
    posterior_draws : array_like, shape (n_draws, n_bands), optional
        Posterior predictive flux draws for uncertainty shading.
    ax : matplotlib Axes, optional
        Axes to plot on. Creates new figure if None.
    band_names : list of str, optional
        Filter names for axis labels.
    show_residuals : bool
        If True, adds a residual panel below the main SED panel. Default True.

    Returns
    -------
    fig : matplotlib Figure

    Examples
    --------
    .. code-block:: python

        from tengri import plot_sed_fit
        import numpy as np

        wave = np.array([4800.0, 6200.0, 7700.0, 9000.0])
        flux = np.array([1.2e-18, 1.8e-18, 2.1e-18, 1.9e-18])
        noise = flux * 0.05
        fig = plot_sed_fit(wave, flux, noise, show_residuals=False)
    """
    # The residual panel needs `posterior_draws` to have anything in it, so it
    # is built only when both are present. Keying it on `show_residuals` alone
    # meant the DEFAULT call: observed photometry, no posterior yet; drew an
    # empty box under the SED. `ax=` is honored whenever no residual panel is
    # being added; a caller's own Axes cannot be split in two.
    want_residuals = show_residuals and posterior_draws is not None
    ax_res = None
    if want_residuals and ax is None:
        fig = plt.figure(figsize=(8, 5))
        gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(gs[0])
        ax_res = fig.add_subplot(gs[1], sharex=ax_main)
    elif ax is None:
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

    if ax_res is not None:
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
    else:
        ax_main.set_xlabel(r"Wavelength ($\AA$)")
    # One return type. This used to hand back a Figure on the residual branch
    # and an Axes otherwise, so what the caller held depended on whether they
    # had passed `posterior_draws`: and `fig.savefig(...)`, which the
    # docstring, the module example and the sibling plot_spectrum_fit all
    # imply, raised AttributeError on the common path.
    return fig


# ═══════════════════════════════════════════════════════════════════
# Spectrum plotting
# ═══════════════════════════════════════════════════════════════════


def plot_spectrum_fit(
    wave_obs, spec_obs, noise, spec_true=None, spec_draws=None, features=None, z=0.1
):
    """Plot spectroscopic fit with residuals: CIGALE style.

    Parameters
    ----------
    wave_obs : array_like, shape (n_pix,)
        Observed-frame wavelength grid. [Angstrom]
    spec_obs : array_like, shape (n_pix,)
        Observed spectrum flux density. [erg/s/cm²/Hz]
    noise : array_like, shape (n_pix,)
        1-sigma per-pixel uncertainties. [erg/s/cm²/Hz]
    spec_true : array_like, shape (n_pix,), optional
        True (noiseless) spectrum for mock recovery plots.
    spec_draws : array_like, shape (n_draws, n_pix), optional
        Posterior predictive spectrum draws for uncertainty shading.
    features : dict, optional
        Rest-frame line/feature wavelengths to annotate, e.g. ``{"Hα": 6563.0}``.
    z : float
        Redshift for shifting feature wavelengths to observed frame. Default 0.1.

    Returns
    -------
    fig : matplotlib Figure

    Examples
    --------
    .. code-block:: python

        from tengri import plot_spectrum_fit

        fig = plot_spectrum_fit(
            wave_obs,
            spec_obs,
            noise,
            spec_draws=posterior_spec_draws,
            features={"Hα": 6563.0, "Hβ": 4861.0},
            z=0.5,
        )
        fig.savefig("spectrum_fit.pdf")
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
# Sweep utilities: the engine behind all gallery notebooks
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
    model : tengri.SEDModel
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
    # Clamp to SWEEP_VMIN..SWEEP_VMAX so the bright-yellow tail of viridis
    # (>0.85) doesn't wash out on white backgrounds.
    from .styles import SWEEP_VMAX, SWEEP_VMIN

    colors = [
        cmap_obj(SWEEP_VMIN + (SWEEP_VMAX - SWEEP_VMIN) * i / max(n - 1, 1)) for i in range(n)
    ]

    xlim = wave_range if wave_range is not None else SED_XLIM

    for i, val in enumerate(values):
        # Override single parameter; use model defaults for rest
        override = {param_name: float(val)}
        pred = model._predict_rest_sed(override)
        wave = np.asarray(pred.wavelength)
        lnu = np.asarray(pred.sed)

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

        style = REFERENCE_STYLE.copy() if i == reference_idx else {"color": colors[i], "lw": 2.2}
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
    model : tengri.SEDModel
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
    model : tengri.SEDModel
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
    import jax

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax_sfh, ax_sed = axes

    cmap_obj = plt.get_cmap(cmap)
    n = len(values)
    # Clamp to SWEEP_VMIN..SWEEP_VMAX so the bright-yellow tail of viridis
    # (>0.85) doesn't wash out on white backgrounds.
    from .styles import SWEEP_VMAX, SWEEP_VMIN

    colors = [
        cmap_obj(SWEEP_VMIN + (SWEEP_VMAX - SWEEP_VMIN) * i / max(n - 1, 1)) for i in range(n)
    ]

    for i, val in enumerate(values):
        override = {param_name: float(val)}
        color = colors[i]
        label = f"{param_name} = {val:.2g}"

        # Plot stochastic samples first (thin lines underneath)
        if n_stochastic > 0 and key is not None and model.spec.stochastic:
            for s in range(n_stochastic):
                # Generate unique key for this sample
                sample_key = jax.random.fold_in(key, i * 1000 + s)
                # Generate xi vector for stochastic SFH
                xi = jax.random.normal(sample_key, shape=(model.n_grid,))
                override_with_xi = {**override, "sfh_field_xi": xi}

                # SFH panel - stochastic
                try:
                    sfh = model.predict_sfh(override_with_xi)
                    t_lookback = np.asarray(sfh["t_gyr"])
                    sfr = np.asarray(sfh.get("sfr_full", sfh.get("sfr_mean")))
                    ax_sfh.plot(t_lookback, sfr, color=color, lw=0.5, alpha=0.4)
                except (AttributeError, TypeError, ValueError, KeyError):
                    pass

                # SED panel - stochastic
                try:
                    pred = model._predict_rest_sed(override_with_xi)
                    wave = np.asarray(pred.wavelength)
                    lnu = np.asarray(pred.sed)
                    idx_norm = int(np.argmin(np.abs(wave - 5500.0)))
                    norm = lnu[idx_norm]
                    if norm > 0:
                        lnu = lnu / norm
                    y = lnu * wave
                    mask = (wave >= SED_XLIM[0]) & (wave <= SED_XLIM[1])
                    ax_sed.plot(wave[mask], y[mask], color=color, lw=0.5, alpha=0.4)
                except (IndexError, ValueError, TypeError, AttributeError):
                    pass

        # Plot deterministic mean on top
        try:
            sfh = model.predict_sfh(override)
            t_lookback = np.asarray(sfh["t_gyr"])
            sfr = np.asarray(sfh.get("sfr_mean"))
            ax_sfh.plot(t_lookback, sfr, color=color, lw=2.0, label=label)
        except (AttributeError, TypeError, ValueError, KeyError):
            pass

        # SED panel - deterministic mean
        try:
            pred = model._predict_rest_sed(override)
            wave = np.asarray(pred.wavelength)
            lnu = np.asarray(pred.sed)
            idx_norm = int(np.argmin(np.abs(wave - 5500.0)))
            norm = lnu[idx_norm]
            if norm > 0:
                lnu = lnu / norm
            y = lnu * wave
            mask = (wave >= SED_XLIM[0]) & (wave <= SED_XLIM[1])
            ax_sed.plot(wave[mask], y[mask], color=color, lw=2.0, label=label)
        except (IndexError, ValueError, TypeError, AttributeError):
            pass

    ax_sfh.set_xlabel(SFH_XLABEL)
    ax_sfh.set_ylabel(SFH_YLABEL)
    ax_sfh.legend(fontsize=9)

    ax_sed.set_xscale(SED_XSCALE)
    ax_sed.set_yscale("log")
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

    Convenience function for MockData.plot(): shows observed photometry
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

    Convenience function for Posterior.plot_sed(): creates a two-panel
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
            sed_i = result._model._predict_rest_sed(sample_i).sed
            sed_samples.append(sed_i)
        sed_array = np.array(sed_samples)

        # Get wavelengths
        wave_rest = np.array(result._model.ssp_data.ssp_wave)

        # Plot posterior band (16th to 84th percentile)
        sed_lo = np.percentile(sed_array, 16, axis=0)
        sed_hi = np.percentile(sed_array, 84, axis=0)
        ax_sed.fill_between(
            wave_rest, sed_lo, sed_hi, alpha=0.3, color="C0", label="68% credible region"
        )

        # Plot MAP SED
        sed_map = result._model._predict_rest_sed(result.params).sed
        ax_sed.plot(wave_rest, sed_map, "-", color="C0", lw=2, label="MAP SED")
    else:
        # MAP only: just plot the best fit
        wave_rest = np.array(result._model.ssp_data.ssp_wave)
        sed_map = result._model._predict_rest_sed(result.params).sed
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
        except (KeyError, AttributeError, ValueError, TypeError):
            # KeyError: sfr_mean or t_gyr missing from output
            # AttributeError: predict_sfh method doesn't exist
            # ValueError/TypeError: array conversion failed
            pass

    ax_sfh.set_xlabel("Lookback time [Gyr]")
    ax_sfh.set_ylabel("SFR [M☉/yr]")
    ax_sfh.set_title("Star Formation History")
    ax_sfh.legend(frameon=False)

    plt.tight_layout()
    return fig, axes
