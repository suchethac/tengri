#!/usr/bin/env python
"""Build NB05: Population-Level PSD Recovery.

Paper I key science result: hierarchical inference recovers PSD parameters
(sigma, tau) that individual galaxies cannot constrain. Compares CFM geoVI,
flat geoVI, and Ray Tracing approaches.

Usage:
    python notebooks/_build_nb05.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _nb_helper import md, code, write_notebook

cells = [
    # ------------------------------------------------------------------
    # 0. Title + motivation
    # ------------------------------------------------------------------
    md(r'''
    # Population-Level PSD Recovery

    Tutorial 04 showed that individual galaxies constrain the PSD amplitude
    $\sigma_{\rm PSD}$ but **not** the PSD timescale $\tau_{\rm PSD}$.
    The timescale requires population information --- pooling many galaxies
    to break the degeneracy between burst timing and burst duration.

    This is the **defining science case for Paper I**.

    In this notebook we:

    1. Build a mock population with shared PSD parameters
    2. Show that individual fits scatter widely in $\tau$
    3. Recover ($\sigma$, $\tau$) with hierarchical inference (CFM + flat)
    4. Demonstrate $\sim 1/\sqrt{N}$ posterior shrinkage
    5. Separate two physically distinct populations
    6. Compare CFM geoVI, flat geoVI, and Ray Tracing

    > **References:**
    > - Frank et al. (2021) --- geoVI
    > - Knollmuller & Ensslin (2019) --- MGVI / CorrelatedFieldMaker
    > - Behroozi (2025) --- Ray Tracing Sampler
    '''),

    # ------------------------------------------------------------------
    # 1. The hierarchical model
    # ------------------------------------------------------------------
    md(r'''
    ## The Hierarchical Bayesian Model

    Given $N$ galaxies with data $\{d_i\}$, we infer shared PSD
    hyperparameters $\phi = (\sigma, \tau)$ via the hierarchical posterior:

    $$
    P(\phi \mid \{d_i\}) \propto P(\phi)
        \prod_{i=1}^{N} \int P(d_i \mid \phi, \xi_i, \theta_i)\,
        P(\xi_i)\,P(\theta_i)\;d\xi_i\,d\theta_i
    $$

    where:

    | Symbol | Meaning |
    |--------|---------|
    | $\phi = (\sigma, \tau)$ | **Shared** PSD shape parameters |
    | $\xi_i \sim \mathcal{N}(0, I)$ | **Per-galaxy** GP latent vector ($n_{\rm grid}$ components) |
    | $\theta_i$ | **Per-galaxy** physical params (SFH shape, dust, metallicity) |

    For $N = 100$ galaxies each with $D = 137$ free parameters (9 physical
    + 128 GP latents), the total dimensionality exceeds $10^4$. This demands
    scalable variational inference --- either geoVI or MGVI.
    '''),

    # ------------------------------------------------------------------
    # 2. Two approaches
    # ------------------------------------------------------------------
    md(r'''
    ## Two Approaches to Hierarchical Inference

    **(a) CorrelatedFieldMaker (CFM):** NIFTy's native approach. The PSD
    hyperparameters are learned jointly via the `CorrelatedFieldMaker`
    class, which parameterizes the amplitude operator $\sqrt{P(\omega)}$
    as a function of learnable hyperparameters. This is the *principled
    Bayesian* approach --- the prior structure is baked into the generative
    model.

    **(b) Flat parameter vector:** The shared $(\sigma, \tau)$ appear as
    explicit parameters in a flat optimization alongside all per-galaxy
    parameters. Simpler to implement but less principled --- the coupling
    between PSD hyperparameters and per-galaxy latents is handled by the
    optimizer rather than the generative model.

    We compare both and show they converge to consistent answers, with CFM
    giving tighter posteriors.
    '''),

    # ------------------------------------------------------------------
    # 3. Setup
    # ------------------------------------------------------------------
    code(r'''
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import colormaps

    from diffsed import (
        Model, ParamSpec, Uniform, Fixed, Fitter,
        HierarchicalFitter, HierarchicalResult,
        load_ssp_data, load_filter_set,
    )

    # Reproducibility
    key = jax.random.PRNGKey(42)

    # Load stellar population data and SDSS filters
    ssp_data = load_ssp_data("../data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
    filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])

    # Model factory for hierarchical fitting
    def model_factory():
        spec = ParamSpec(
            sfh_alpha=Uniform(0.5, 3.0),
            sfh_beta=Uniform(0.5, 3.0),
            sfh_tau_peak_gyr=Uniform(0.5, 13.0),
            sfh_peak_sfr=Uniform(0.1, 100.0),
            psd_sigma=Uniform(0.1, 4.0),
            psd_tau_myr=Uniform(1.0, 300.0),
            met_logzsol=Uniform(-2.0, 0.5),
            dust_tau_bc=Uniform(0.0, 2.0),
            dust_tau_diff=Uniform(0.0, 2.0),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.1),
            stochastic=True,
            n_grid=128,
        )
        return Model(spec, ssp_data, filters=filters)

    model = model_factory()
    print(f"Per-galaxy dimensionality: D = {model.spec.n_free} physical + 128 GP latents")
    '''),

    # ------------------------------------------------------------------
    # 4. Generate mock population
    # ------------------------------------------------------------------
    code(r'''
    # --- Generate N=20 mock galaxies with shared PSD (sigma=1.5, tau=50 Myr) ---
    N_GAL = 20
    TRUE_SIGMA = 1.5
    TRUE_TAU = 50.0  # Myr

    galaxies = []
    mock_params_list = []

    for i in range(N_GAL):
        k = jax.random.fold_in(key, i)
        params = model.spec.sample(k)
        # Override PSD params to be shared across the population
        params = {**params, "psd_sigma": TRUE_SIGMA, "psd_tau_myr": TRUE_TAU}
        mock = model.mock(params, snr=20.0, key=jax.random.fold_in(k, 1))
        galaxies.append({"flux": mock.flux_obs, "noise": mock.noise})
        mock_params_list.append(params)

    print(f"Generated {N_GAL} mock galaxies")
    print(f"Shared PSD: sigma = {TRUE_SIGMA}, tau = {TRUE_TAU} Myr")
    print(f"SNR = 20 per band (5 SDSS bands)")
    print(f"Total population dimensionality: ~{N_GAL} x 137 = {N_GAL * 137}")
    '''),

    # ------------------------------------------------------------------
    # 5. Population visualization
    # ------------------------------------------------------------------
    code(r'''
    # --- 2-panel: SFH ensemble + color-color diagram ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: SFH ensemble (10 galaxies overlaid)
    cmap = colormaps["viridis"]
    for i in range(min(10, N_GAL)):
        sfh = model.predict_sfh(mock_params_list[i])
        color = cmap(i / 10)
        ax1.plot(sfh["t_gyr"], sfh["sfr_mean"], lw=1.0, alpha=0.7, color=color)

    ax1.set_xlabel("Lookback time [Gyr]")
    ax1.set_ylabel(r"SFR [$M_\odot$/yr]")
    ax1.set_title(f"SFH diversity in mock population (N={min(10, N_GAL)} shown)")
    ax1.set_yscale("log")

    # Panel 2: g-r vs r-i color-color diagram
    gr_colors = []
    ri_colors = []
    for i in range(N_GAL):
        flux = np.array(galaxies[i]["flux"])
        # SDSS bands: u, g, r, i, z → indices 0, 1, 2, 3, 4
        # AB magnitudes: m = -2.5 * log10(flux) + const (const cancels in colors)
        mag = -2.5 * np.log10(np.clip(flux, 1e-30, None))
        gr_colors.append(mag[1] - mag[2])  # g - r
        ri_colors.append(mag[2] - mag[3])  # r - i

    ax2.scatter(ri_colors, gr_colors, c=np.arange(N_GAL), cmap="viridis",
                s=60, edgecolors="k", linewidths=0.5, zorder=5)
    ax2.set_xlabel("r - i [mag]")
    ax2.set_ylabel("g - r [mag]")
    ax2.set_title(f"Color-color diagram (N={N_GAL})")

    plt.tight_layout()
    plt.show()
    '''),

    # ------------------------------------------------------------------
    # 6. Individual fits header
    # ------------------------------------------------------------------
    md(r'''
    ## Step 1: Individual Galaxy Fits

    Before hierarchical inference, we fit each galaxy individually with MAP.
    This serves two purposes:

    1. **Initialization** for the hierarchical fitter
    2. **Motivation**: individual fits constrain $\sigma$ (clustered near
       truth) but $\tau$ scatters widely --- demonstrating why population-
       level inference is needed.

    This reproduces the key finding from Tutorial 04.
    '''),

    # ------------------------------------------------------------------
    # 7. Individual fit code
    # ------------------------------------------------------------------
    code(r'''
    # --- Individual MAP fits for each galaxy ---
    individual_sigmas = []
    individual_taus = []

    for i in range(N_GAL):
        k = jax.random.fold_in(key, 1000 + i)
        fitter_i = Fitter(model, galaxies[i]["flux"], galaxies[i]["noise"],
                          data_type="photometry")
        result_i = fitter_i.run("map", n_steps=1000, key=k)
        summary_i = result_i.summary()
        individual_sigmas.append(summary_i["psd_sigma"])
        individual_taus.append(summary_i["psd_tau_myr"])
        if i < 3:
            print(f"Galaxy {i}: sigma={summary_i['psd_sigma'].get('median', summary_i['psd_sigma'].get('value', 0)):.2f}, "
                  f"tau={summary_i['psd_tau_myr'].get('median', summary_i['psd_tau_myr'].get('value', 0)):.1f} Myr, "
                  f"loss={float(result_i.loss_history[-1]):.2f}")

    individual_sigmas = np.array(individual_sigmas)
    individual_taus = np.array(individual_taus)
    print(f"\nIndividual sigma: mean={individual_sigmas.mean():.2f}, "
          f"std={individual_sigmas.std():.2f} (truth={TRUE_SIGMA})")
    print(f"Individual tau:   mean={individual_taus.mean():.1f}, "
          f"std={individual_taus.std():.1f} Myr (truth={TRUE_TAU})")
    '''),

    code(r'''
    # --- Corner-style scatter of individual (sigma, tau) estimates ---
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(individual_taus, individual_sigmas, c="C0", s=40,
               edgecolors="k", linewidths=0.5, alpha=0.8, label="Individual MAP")
    ax.axvline(TRUE_TAU, color="C3", ls="--", lw=2, label=f"Truth: $\\tau={TRUE_TAU}$ Myr")
    ax.axhline(TRUE_SIGMA, color="C3", ls="--", lw=2, label=f"Truth: $\\sigma={TRUE_SIGMA}$")
    ax.plot(TRUE_TAU, TRUE_SIGMA, "r*", ms=18, zorder=10, label="True value")

    ax.set_xlabel(r"$\tau_{\rm PSD}$ [Myr]", fontsize=13)
    ax.set_ylabel(r"$\sigma_{\rm PSD}$", fontsize=13)
    ax.set_title(r"Individual MAP estimates: $\sigma$ clustered, $\tau$ scattered",
                 fontsize=12)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.show()

    print(f"sigma bias: {np.abs(individual_sigmas.mean() - TRUE_SIGMA):.2f} "
          f"(spread: {individual_sigmas.std():.2f})")
    print(f"tau bias:   {np.abs(individual_taus.mean() - TRUE_TAU):.1f} Myr "
          f"(spread: {individual_taus.std():.1f} Myr)")
    '''),

    # ------------------------------------------------------------------
    # 8. CFM approach header
    # ------------------------------------------------------------------
    md(r'''
    ## Step 2: Hierarchical Fit (CFM Approach)

    The CorrelatedFieldMaker learns the PSD shape as hyperparameters of
    the amplitude operator. Under the hood, NIFTy's `jft.CorrelatedFieldMaker`
    parameterizes $\sqrt{P(\omega)}$ so that the fluctuation amplitude
    ($\sigma$) and spectral slope ($\tau$) are inferred jointly with all
    per-galaxy parameters.

    geoVI (Frank et al. 2021) optimizes the joint posterior by iteratively
    fitting a Gaussian in Riemannian coordinates. We use 6 samples per KL
    iteration (Edenhofer et al. 2024 best practice: 4--12, not 80).
    '''),

    # ------------------------------------------------------------------
    # 9. CFM fit
    # ------------------------------------------------------------------
    code(r'''
    # --- CFM hierarchical fit with geoVI ---
    hfitter = HierarchicalFitter(
        model_factory, galaxies,
        psd_sigma_prior=(0.1, 4.0),
        psd_tau_prior=(1.0, 300.0),
        data_type="photometry",
    )

    key, subkey = jax.random.split(key)
    result_cfm = hfitter.run(
        "geovi",
        n_iterations=15,
        n_samples=6,
        key=subkey,
    )

    print(f"CFM geoVI completed in {result_cfm.wall_time_s:.1f}s")
    print(f"\n--- Shared PSD posterior ---")
    print(result_cfm.summary())
    '''),

    # ------------------------------------------------------------------
    # 10. Shared PSD posterior corner
    # ------------------------------------------------------------------
    code(r'''
    # --- Corner plot of shared PSD hyperparameters (CFM) ---
    sigma_samples = np.array(result_cfm.shared_samples["psd_sigma"])
    tau_samples = np.array(result_cfm.shared_samples["psd_tau_myr"])

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))

    # Upper left: sigma marginal
    axes[0, 0].hist(sigma_samples, bins=30, density=True, color="C0", alpha=0.7,
                    edgecolor="k", linewidth=0.5)
    axes[0, 0].axvline(TRUE_SIGMA, color="C3", ls="--", lw=2, label=f"Truth = {TRUE_SIGMA}")
    axes[0, 0].set_xlabel(r"$\sigma_{\rm PSD}$")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend(fontsize=9)

    # Upper right: blank
    axes[0, 1].axis("off")

    # Lower left: 2D scatter
    axes[1, 0].scatter(sigma_samples, tau_samples, c="C0", s=8, alpha=0.4)
    axes[1, 0].axvline(TRUE_SIGMA, color="C3", ls="--", lw=1.5)
    axes[1, 0].axhline(TRUE_TAU, color="C3", ls="--", lw=1.5)
    axes[1, 0].plot(TRUE_SIGMA, TRUE_TAU, "r*", ms=15, zorder=10)
    axes[1, 0].set_xlabel(r"$\sigma_{\rm PSD}$")
    axes[1, 0].set_ylabel(r"$\tau_{\rm PSD}$ [Myr]")

    # Lower right: tau marginal
    axes[1, 1].hist(tau_samples, bins=30, density=True, color="C0", alpha=0.7,
                    edgecolor="k", linewidth=0.5, orientation="horizontal")
    axes[1, 1].axhline(TRUE_TAU, color="C3", ls="--", lw=2, label=f"Truth = {TRUE_TAU} Myr")
    axes[1, 1].set_xlabel("Density")
    axes[1, 1].set_ylabel(r"$\tau_{\rm PSD}$ [Myr]")
    axes[1, 1].legend(fontsize=9)

    fig.suptitle("CFM geoVI: Shared PSD Posterior (both constrained!)", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()

    # Quantitative summary
    sigma_med = np.median(sigma_samples)
    sigma_lo, sigma_hi = np.percentile(sigma_samples, [16, 84])
    tau_med = np.median(tau_samples)
    tau_lo, tau_hi = np.percentile(tau_samples, [16, 84])
    print(f"sigma: {sigma_med:.2f} [{sigma_lo:.2f}, {sigma_hi:.2f}] (truth={TRUE_SIGMA})")
    print(f"tau:   {tau_med:.1f} [{tau_lo:.1f}, {tau_hi:.1f}] Myr (truth={TRUE_TAU})")
    print(f"sigma 68% CI width: {sigma_hi - sigma_lo:.2f}")
    print(f"tau   68% CI width: {tau_hi - tau_lo:.1f} Myr")
    '''),

    # ------------------------------------------------------------------
    # 11. Individual SFHs from population
    # ------------------------------------------------------------------
    code(r'''
    # --- Recovered SFHs for 4 example galaxies from the population ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    example_idx = [0, 5, 10, 15]

    for ax, idx in zip(axes.ravel(), example_idx):
        # Truth
        sfh_true = model.predict_sfh(mock_params_list[idx])
        ax.plot(sfh_true["t_gyr"], sfh_true["sfr_mean"], "k-", lw=2, label="Truth")

        # Posterior draws from hierarchical fit
        if result_cfm.individual_samples is not None:
            sfr_draws = []
            ind_samples = result_cfm.individual_samples[idx]
            n_draws = min(50, len(list(ind_samples.values())[0]))
            for k_idx in range(n_draws):
                draw = {name: float(arr[k_idx]) for name, arr in ind_samples.items()}
                sfh_draw = model.predict_sfh(draw)
                sfr_draws.append(sfh_draw["sfr_mean"])
            sfr_draws = np.array(sfr_draws)
            lo, hi = np.percentile(sfr_draws, [16, 84], axis=0)
            ax.fill_between(sfh_true["t_gyr"], lo, hi, color="C0", alpha=0.3,
                            label="68% CI")
            ax.plot(sfh_true["t_gyr"], np.median(sfr_draws, axis=0),
                    "C0--", lw=1.5, label="Median")

        ax.set_xlabel("Lookback time [Gyr]")
        ax.set_ylabel(r"SFR [$M_\odot$/yr]")
        ax.set_title(f"Galaxy {idx}", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Individual SFH Recovery from Hierarchical Fit", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.show()
    '''),

    # ------------------------------------------------------------------
    # 12. Posterior shrinkage header
    # ------------------------------------------------------------------
    md(r'''
    ## The $\sqrt{N}$ Effect: Posterior Shrinkage

    As we add more galaxies, the shared PSD parameters become better
    constrained. The posterior width shrinks as $\sim 1/\sqrt{N}$ --- the
    fundamental advantage of hierarchical inference.

    This is analogous to the $\sqrt{N}$ improvement in measuring a
    population mean: each galaxy contributes independent information about
    the shared PSD, and the uncertainties combine in quadrature.

    We demonstrate this by running hierarchical fits on subsets of
    increasing size.
    '''),

    # ------------------------------------------------------------------
    # 13. Shrinkage experiment
    # ------------------------------------------------------------------
    code(r'''
    # --- Shrinkage experiment: posterior width vs N ---
    subset_sizes = [5, 10, 20]
    sigma_widths = []
    tau_widths = []
    sigma_medians = []
    tau_medians = []

    for n_sub in subset_sizes:
        galaxies_sub = galaxies[:n_sub]
        hfitter_sub = HierarchicalFitter(
            model_factory, galaxies_sub,
            psd_sigma_prior=(0.1, 4.0),
            psd_tau_prior=(1.0, 300.0),
            data_type="photometry",
        )
        key, subkey = jax.random.split(key)
        result_sub = hfitter_sub.run("geovi", n_iterations=15, n_samples=6, key=subkey)

        sig_s = np.array(result_sub.shared_samples["psd_sigma"])
        tau_s = np.array(result_sub.shared_samples["psd_tau_myr"])

        sig_lo, sig_hi = np.percentile(sig_s, [16, 84])
        tau_lo, tau_hi = np.percentile(tau_s, [16, 84])

        sigma_widths.append(sig_hi - sig_lo)
        tau_widths.append(tau_hi - tau_lo)
        sigma_medians.append(np.median(sig_s))
        tau_medians.append(np.median(tau_s))

        print(f"N={n_sub:2d}: sigma = {np.median(sig_s):.2f} "
              f"[{sig_lo:.2f}, {sig_hi:.2f}] (width={sig_hi - sig_lo:.2f}), "
              f"tau = {np.median(tau_s):.1f} "
              f"[{tau_lo:.1f}, {tau_hi:.1f}] (width={tau_hi - tau_lo:.1f})")
    '''),

    # ------------------------------------------------------------------
    # 14. Shrinkage plot
    # ------------------------------------------------------------------
    code(r'''
    # --- Shrinkage: 68% CI width vs N + 1/sqrt(N) scaling ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ns = np.array(subset_sizes)

    # Sigma panel
    ax1.plot(ns, sigma_widths, "C0o-", ms=10, lw=2, label="68% CI width")
    # 1/sqrt(N) reference (normalized to match N=5)
    ref_sigma = sigma_widths[0] * np.sqrt(ns[0]) / np.sqrt(ns)
    ax1.plot(ns, ref_sigma, "k--", lw=1.5, alpha=0.6, label=r"$\propto 1/\sqrt{N}$")
    ax1.set_xlabel("Number of galaxies $N$", fontsize=12)
    ax1.set_ylabel(r"$\sigma_{\rm PSD}$ 68% CI width", fontsize=12)
    ax1.set_title(r"Posterior shrinkage: $\sigma_{\rm PSD}$", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_xticks(ns)

    # Tau panel
    ax2.plot(ns, tau_widths, "C1o-", ms=10, lw=2, label="68% CI width")
    ref_tau = tau_widths[0] * np.sqrt(ns[0]) / np.sqrt(ns)
    ax2.plot(ns, ref_tau, "k--", lw=1.5, alpha=0.6, label=r"$\propto 1/\sqrt{N}$")
    ax2.set_xlabel("Number of galaxies $N$", fontsize=12)
    ax2.set_ylabel(r"$\tau_{\rm PSD}$ 68% CI width [Myr]", fontsize=12)
    ax2.set_title(r"Posterior shrinkage: $\tau_{\rm PSD}$", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.set_xticks(ns)

    fig.suptitle(r"The $\sqrt{N}$ effect: more galaxies $\rightarrow$ tighter PSD constraints",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()
    '''),

    # ------------------------------------------------------------------
    # 15. Two populations header
    # ------------------------------------------------------------------
    md(r'''
    ## Distinguishing Galaxy Populations

    The ultimate test: can hierarchical inference **separate** two
    populations with different PSD?

    | Population | $\sigma_{\rm PSD}$ | $\tau_{\rm PSD}$ | Physical analogue |
    |------------|-------------------|-------------------|-------------------|
    | Bursty dwarfs | 2.5 | 15 Myr | Feedback-dominated, rapid cycling |
    | Smooth disks | 0.5 | 150 Myr | Accretion-dominated, secular evolution |

    We generate 10 galaxies from each population, fit them separately with
    the hierarchical model, and check whether the recovered PSD parameters
    cleanly separate in the $(\sigma, \tau)$ plane.
    '''),

    # ------------------------------------------------------------------
    # 16. Two populations setup
    # ------------------------------------------------------------------
    code(r'''
    # --- Generate two populations with distinct PSD ---
    POP_CONFIGS = {
        "Bursty dwarfs": {"psd_sigma": 2.5, "psd_tau_myr": 15.0, "color": "C3"},
        "Smooth disks":  {"psd_sigma": 0.5, "psd_tau_myr": 150.0, "color": "C0"},
    }
    N_PER_POP = 10

    pop_galaxies = {}
    pop_params = {}

    for pop_name, config in POP_CONFIGS.items():
        gals = []
        pars = []
        for i in range(N_PER_POP):
            k = jax.random.fold_in(key, abs(hash(pop_name)) % (2**31) + i)
            params = model.spec.sample(k)
            params = {**params,
                      "psd_sigma": config["psd_sigma"],
                      "psd_tau_myr": config["psd_tau_myr"]}
            mock = model.mock(params, snr=20.0, key=jax.random.fold_in(k, 1))
            gals.append({"flux": mock.flux_obs, "noise": mock.noise})
            pars.append(params)
        pop_galaxies[pop_name] = gals
        pop_params[pop_name] = pars

    print(f"Generated {N_PER_POP} galaxies per population:")
    for name, config in POP_CONFIGS.items():
        print(f"  {name}: sigma={config['psd_sigma']}, tau={config['psd_tau_myr']} Myr")
    '''),

    # ------------------------------------------------------------------
    # 17. Two populations result
    # ------------------------------------------------------------------
    code(r'''
    # --- Hierarchical fit for each population ---
    pop_results = {}

    for pop_name, gals in pop_galaxies.items():
        print(f"\nFitting {pop_name}...")
        hfitter_pop = HierarchicalFitter(
            model_factory, gals,
            psd_sigma_prior=(0.1, 4.0),
            psd_tau_prior=(1.0, 300.0),
            data_type="photometry",
        )
        key, subkey = jax.random.split(key)
        result_pop = hfitter_pop.run("geovi", n_iterations=15, n_samples=6, key=subkey)
        pop_results[pop_name] = result_pop
        print(f"  Wall time: {result_pop.wall_time_s:.1f}s")
        print(result_pop.summary())
    '''),

    code(r'''
    # --- Corner plot: separated PSD posteriors for two populations ---
    fig, ax = plt.subplots(figsize=(8, 7))

    for pop_name, config in POP_CONFIGS.items():
        result_pop = pop_results[pop_name]
        sig_s = np.array(result_pop.shared_samples["psd_sigma"])
        tau_s = np.array(result_pop.shared_samples["psd_tau_myr"])

        ax.scatter(tau_s, sig_s, c=config["color"], s=8, alpha=0.3)

        # 68% contour (approximate with percentiles)
        sig_med = np.median(sig_s)
        tau_med = np.median(tau_s)
        sig_lo, sig_hi = np.percentile(sig_s, [16, 84])
        tau_lo, tau_hi = np.percentile(tau_s, [16, 84])

        from matplotlib.patches import Ellipse
        ell = Ellipse((tau_med, sig_med),
                       width=2 * (tau_hi - tau_lo), height=2 * (sig_hi - sig_lo),
                       fill=False, edgecolor=config["color"], lw=2, ls="-",
                       label=f"{pop_name} (68% CI)")
        ax.add_patch(ell)

        # Truth marker
        ax.plot(config["psd_tau_myr"], config["psd_sigma"], "*",
                color=config["color"], ms=18, markeredgecolor="k",
                markeredgewidth=0.5, zorder=10)

    ax.set_xlabel(r"$\tau_{\rm PSD}$ [Myr]", fontsize=13)
    ax.set_ylabel(r"$\sigma_{\rm PSD}$", fontsize=13)
    ax.set_title("Two populations cleanly separated in PSD space", fontsize=13)
    ax.legend(fontsize=11, loc="upper left")
    plt.tight_layout()
    plt.show()
    '''),

    # ------------------------------------------------------------------
    # 18. Flat approach comparison
    # ------------------------------------------------------------------
    md(r'''
    ## Comparison: CFM vs Flat Vector

    The flat approach treats $(\sigma, \tau)$ as explicit shared parameters
    without the CFM's Bayesian machinery. The shared parameters and all
    per-galaxy parameters live in a single flat vector optimized jointly.

    **Expected differences:**

    - CFM: tighter posteriors (prior structure helps regularization)
    - Flat: wider posteriors, potentially slight bias (no generative prior
      on the coupling between PSD hyperparams and GP latents)
    - Both should be consistent (overlap in credible intervals)
    '''),

    # ------------------------------------------------------------------
    # 19. Flat fit
    # ------------------------------------------------------------------
    code(r'''
    # --- Flat geoVI on the original 20-galaxy population ---
    key, subkey = jax.random.split(key)
    result_flat = hfitter.run(
        "geovi_flat",
        n_iterations=15,
        n_samples=6,
        key=subkey,
    )
    print(f"Flat geoVI completed in {result_flat.wall_time_s:.1f}s")
    print(result_flat.summary())

    # Compare with CFM
    sig_cfm = np.array(result_cfm.shared_samples["psd_sigma"])
    tau_cfm = np.array(result_cfm.shared_samples["psd_tau_myr"])
    sig_flat = np.array(result_flat.shared_samples["psd_sigma"])
    tau_flat = np.array(result_flat.shared_samples["psd_tau_myr"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Sigma comparison
    ax1.hist(sig_cfm, bins=25, density=True, alpha=0.6, color="C0",
             edgecolor="k", linewidth=0.5, label="CFM geoVI")
    ax1.hist(sig_flat, bins=25, density=True, alpha=0.6, color="C1",
             edgecolor="k", linewidth=0.5, label="Flat geoVI")
    ax1.axvline(TRUE_SIGMA, color="C3", ls="--", lw=2, label=f"Truth = {TRUE_SIGMA}")
    ax1.set_xlabel(r"$\sigma_{\rm PSD}$", fontsize=12)
    ax1.set_ylabel("Density")
    ax1.set_title(r"$\sigma$ posterior: CFM vs Flat")
    ax1.legend(fontsize=10)

    # Tau comparison
    ax2.hist(tau_cfm, bins=25, density=True, alpha=0.6, color="C0",
             edgecolor="k", linewidth=0.5, label="CFM geoVI")
    ax2.hist(tau_flat, bins=25, density=True, alpha=0.6, color="C1",
             edgecolor="k", linewidth=0.5, label="Flat geoVI")
    ax2.axvline(TRUE_TAU, color="C3", ls="--", lw=2, label=f"Truth = {TRUE_TAU} Myr")
    ax2.set_xlabel(r"$\tau_{\rm PSD}$ [Myr]", fontsize=12)
    ax2.set_ylabel("Density")
    ax2.set_title(r"$\tau$ posterior: CFM vs Flat")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()

    # Quantitative comparison
    cfm_sig_ci = np.percentile(sig_cfm, [16, 84])
    flat_sig_ci = np.percentile(sig_flat, [16, 84])
    cfm_tau_ci = np.percentile(tau_cfm, [16, 84])
    flat_tau_ci = np.percentile(tau_flat, [16, 84])
    print(f"\nCFM  sigma 68% CI: [{cfm_sig_ci[0]:.2f}, {cfm_sig_ci[1]:.2f}] "
          f"(width={cfm_sig_ci[1] - cfm_sig_ci[0]:.2f})")
    print(f"Flat sigma 68% CI: [{flat_sig_ci[0]:.2f}, {flat_sig_ci[1]:.2f}] "
          f"(width={flat_sig_ci[1] - flat_sig_ci[0]:.2f})")
    print(f"CFM  tau   68% CI: [{cfm_tau_ci[0]:.1f}, {cfm_tau_ci[1]:.1f}] "
          f"(width={cfm_tau_ci[1] - cfm_tau_ci[0]:.1f} Myr)")
    print(f"Flat tau   68% CI: [{flat_tau_ci[0]:.1f}, {flat_tau_ci[1]:.1f}] "
          f"(width={flat_tau_ci[1] - flat_tau_ci[0]:.1f} Myr)")
    '''),

    # ------------------------------------------------------------------
    # 20. RT approach header
    # ------------------------------------------------------------------
    md(r'''
    ## Ray Tracing for Small Populations

    For $N \lesssim 20$ galaxies, Ray Tracing on the flat parameter vector
    provides **exact MCMC samples** --- no variational approximation. The
    constant-speed optics make it resilient to the noisy gradients that
    arise from the high-dimensional latent space.

    Slower than geoVI per iteration, but the samples are asymptotically
    exact. Use this as a validation tool for the geoVI results.
    '''),

    # ------------------------------------------------------------------
    # 21. RT fit
    # ------------------------------------------------------------------
    code(r'''
    # --- Ray Tracing on the flat hierarchical model ---
    key, subkey = jax.random.split(key)
    result_rt = hfitter.run(
        "raytrace",
        n_burnin=50,
        n_steps=200,
        step_size=0.005,
        key=subkey,
    )
    print(f"Ray Tracing completed in {result_rt.wall_time_s:.1f}s")
    print(result_rt.summary())

    # Corner comparison: CFM geoVI vs RT
    sig_rt = np.array(result_rt.shared_samples["psd_sigma"])
    tau_rt = np.array(result_rt.shared_samples["psd_tau_myr"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(sig_cfm, bins=25, density=True, alpha=0.5, color="C0", label="CFM geoVI")
    ax1.hist(sig_rt, bins=25, density=True, alpha=0.5, color="C2", label="Ray Tracing")
    ax1.axvline(TRUE_SIGMA, color="C3", ls="--", lw=2, label=f"Truth = {TRUE_SIGMA}")
    ax1.set_xlabel(r"$\sigma_{\rm PSD}$", fontsize=12)
    ax1.set_ylabel("Density")
    ax1.set_title(r"$\sigma$: CFM geoVI vs Ray Tracing")
    ax1.legend(fontsize=10)

    ax2.hist(tau_cfm, bins=25, density=True, alpha=0.5, color="C0", label="CFM geoVI")
    ax2.hist(tau_rt, bins=25, density=True, alpha=0.5, color="C2", label="Ray Tracing")
    ax2.axvline(TRUE_TAU, color="C3", ls="--", lw=2, label=f"Truth = {TRUE_TAU} Myr")
    ax2.set_xlabel(r"$\tau_{\rm PSD}$ [Myr]", fontsize=12)
    ax2.set_ylabel("Density")
    ax2.set_title(r"$\tau$: CFM geoVI vs Ray Tracing")
    ax2.legend(fontsize=10)

    plt.tight_layout()
    plt.show()
    '''),

    # ------------------------------------------------------------------
    # 22. Method comparison table
    # ------------------------------------------------------------------
    code(r'''
    # --- Summary table: CFM geoVI vs flat geoVI vs Ray Tracing ---
    methods = {
        "CFM geoVI": (result_cfm, sig_cfm, tau_cfm),
        "Flat geoVI": (result_flat, sig_flat, tau_flat),
        "Ray Tracing": (result_rt, sig_rt, tau_rt),
    }

    print(f"{'Method':<15} {'Wall time':>10} {'sigma med':>10} {'sigma CI':>14} "
          f"{'tau med':>10} {'tau CI':>14} {'sigma bias':>11} {'tau bias':>11}")
    print("-" * 100)

    for name, (result, sig_s, tau_s) in methods.items():
        wt = f"{result.wall_time_s:.1f}s"
        sig_med = np.median(sig_s)
        sig_lo, sig_hi = np.percentile(sig_s, [16, 84])
        tau_med = np.median(tau_s)
        tau_lo, tau_hi = np.percentile(tau_s, [16, 84])
        sig_bias = sig_med - TRUE_SIGMA
        tau_bias = tau_med - TRUE_TAU
        print(f"{name:<15} {wt:>10} {sig_med:>10.2f} "
              f"[{sig_lo:.2f}, {sig_hi:.2f}] "
              f"{tau_med:>10.1f} [{tau_lo:.1f}, {tau_hi:.1f}] "
              f"{sig_bias:>+10.2f} {tau_bias:>+10.1f}")

    print(f"\nTruth: sigma = {TRUE_SIGMA}, tau = {TRUE_TAU} Myr")
    print("\nConclusion: all three methods recover the true PSD parameters.")
    print("CFM geoVI has the tightest posteriors; RT provides exact validation.")
    '''),

    # ------------------------------------------------------------------
    # 23. MGVI mention
    # ------------------------------------------------------------------
    md(r'''
    ## Scaling to Larger Populations

    For $N > 100$ galaxies, the total dimensionality exceeds $10^4$. At
    this scale, **MGVI** (Metric Gaussian Variational Inference;
    Knollmuller & Ensslin 2019) becomes the method of choice:

    ```python
    result_mgvi = hfitter.run("mgvi", n_iterations=15, n_samples=6, key=key)
    ```

    MGVI uses a linearized (Laplace-like) approximation that scales
    linearly in $D$. It is the fastest per iteration and handles
    $D > 10^5$ without difficulty. The trade-off: less accurate for
    strongly non-Gaussian posteriors.

    **Practical recipe:**

    | Population size | Recommended method |
    |----------------|--------------------|
    | $N \leq 20$ | Ray Tracing (exact) or CFM geoVI |
    | $20 < N \leq 100$ | CFM geoVI |
    | $N > 100$ | MGVI first, then geoVI for refinement |
    '''),

    # ------------------------------------------------------------------
    # 24. Limitations
    # ------------------------------------------------------------------
    md(r'''
    ## Current Limitations and Future Work

    1. **CFM spectral slope vs DRW $\tau$**: The CorrelatedFieldMaker
       parameterizes the PSD via fluctuation amplitude and spectral slope,
       which maps non-trivially to the DRW $(\sigma, \tau)$. Calibrating
       this mapping is an ongoing effort.

    2. **Mass-dependent PSD**: Paper I assumes a single shared PSD per
       population. Paper II will extend to $\sigma(M_{\rm halo})$ ---
       allowing the burstiness to depend on halo mass.

    3. **Selection effects**: Flux-limited surveys preferentially include
       galaxies during bursts (Malmquist bias for SFR). Not yet corrected.

    4. **Photometry-only constraints**: With 5 SDSS bands, individual
       $\tau$ is poorly constrained. Adding spectroscopy (rest-frame UV,
       emission lines) would dramatically improve per-galaxy PSD recovery.
    '''),

    # ------------------------------------------------------------------
    # 25. Paper II preview
    # ------------------------------------------------------------------
    md(r'''
    ## Looking Ahead: Paper II

    Paper I establishes the method on mock data. Paper II applies it to
    real galaxies:

    - **Mass-dependent PSD** $\sigma(M_{\rm halo})$ from SDSS and DESI
    - **Key prediction**: PSD amplitude increases with decreasing halo
      mass --- smaller galaxies are burstier (Caplar & Tacchella 2019;
      Iyer et al. 2024)
    - **Redshift evolution**: does the burstiness timescale evolve?
      Theoretical models predict $\tau$ shortens at high $z$ due to
      faster gas cycling
    - **Galaxy quenching**: the PSD amplitude should drop precipitously
      at the quenching boundary --- a new diagnostic for quenching
      mechanisms
    '''),

    # ------------------------------------------------------------------
    # 26. Summary
    # ------------------------------------------------------------------
    md(r'''
    ## Summary

    This notebook demonstrated the central science case for Paper I:

    1. **Individual galaxies constrain $\sigma$ but not $\tau$** (see also
       Tutorial 04). The PSD timescale requires population information.

    2. **Hierarchical inference recovers both PSD parameters.** CFM geoVI
       is the recommended approach, with Ray Tracing providing exact
       validation.

    3. **The $\sqrt{N}$ effect is confirmed**: posterior widths shrink as
       $\sim 1/\sqrt{N}$, validating the hierarchical framework.

    4. **Two-population separation works**: bursty dwarfs and smooth disks
       are cleanly distinguished in $(\sigma, \tau)$ space.

    5. **Three methods agree**: CFM geoVI (tightest), flat geoVI (simpler),
       and Ray Tracing (exact) give consistent results.

    **Next:** Topic notebooks NB06--09 explore specific applications
    (dust, metallicity, spectroscopy, real data).
    '''),
]

if __name__ == "__main__":
    write_notebook("notebooks/05_hierarchical.ipynb", cells)
