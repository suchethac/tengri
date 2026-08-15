Examples gallery
================

170+ standalone scripts demonstrating tengri's physics components, fitting workflows, and end-to-end use cases.

Run a script locally with ``python examples/quickstart/plot_first_fit.py``. Physics examples (dust curves, SFH shapes, AGN spectra) require only core dependencies; fits additionally need an SSP grid. Fetch one via ``import tengri; tengri.download_ssp()``.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. thumbnail-parent-div-close

.. raw:: html

    </div>

Quick Start
===========

End-to-end fit walkthrough: from mock data to posterior corner plots and convergence diagnostics.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="MAP returns a point estimate; nothing here estimates uncertainty. Six free parameters, which is the validated ceiling for method=&quot;mcmc_nuts&quot;; method=&quot;laplace&quot; is the cheaper route to intervals, from the Hessian at the MAP. vi and mcmc_raytrace target D ≳ 20. See the method-selection page for the full decision table.">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_first_fit_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_first_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recovering stellar mass from 5-band SDSS photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Understanding model structure through parameter provenance tags">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_model_summary_walkthrough_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_model_summary_walkthrough`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Understanding model structure through parameter provenance tags</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust attenuation across the SED: intrinsic, attenuated, and absorbed">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_sed_components_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_sed_components`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation across the SED: intrinsic, attenuated, and absorbed</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hα and [O III]+Hβ are produced by gas reprocessing the ionizing continuum from O/B stars. The SFH is a young starburst (peak age ≈ 30 Myr).">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_swap_nebular_backend_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_swap_nebular_backend`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Swapping the nebular backend on, then off, on a young starburst</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Recipes
=======

Common workflows: prior comparison, photometry I/O, redshift fixing, filter set swapping, posterior persistence.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six curated recipes span galaxy populations: star-forming at 0–6 (bare-stellar SSP), quiescent at z ≈ 0.05 (bare-stellar, τ_diff-free to trace dust), AGN panchromatic (bare-stellar, full AGN composite with disc+torus+radio+xray), stochastic JWST high-z with burstiness (bare-stellar, DPL+field at 0.5–12), mock-recovery minimal (any SSP, 4–5 free params for benchmarking), and dust-demo (wNE only — baked nebular emission visualized). All use WavePrecomp() except photoz (ztable does not cover z &gt; 12). Use load_ssp(&quot;*.wNE&quot;) only for dust_demo; others silently under-predict if fed wNE.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_compare_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">What each shipped tengri recipe produces</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a FilterCurve from a Gaussian transmission profile and combine it with standard filters. The Photometry object merges them, then SEDModel predicts photometry on all bands at once — custom filters compose naturally with the standard library.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_custom_filter_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_custom_filter`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Register and use a custom photometric filter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Call tengri.list_recipes() to see the shipped menu with SSP requirements (bare-stellar, wNE, or any) and tengri.describe_recipe(name) to fetch a recipe&#x27;s docstring. Three models showcase the morphological diversity: star-forming (DPL+Cue nebular, free z to 6), quiescent at z=0.05 (dexp, lower dust ceiling), and AGN-panchromatic (full composite, z to 6). All require bare-stellar SSP (Cue backend).">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_introspection_tour_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_introspection_tour`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recipe introspection and SED morphology comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Mock 3 galaxies, fit each independently with MAP. The workflow is: sample true parameters → generate mock fluxes + noise → fit with free SFH/dust and fixed redshift. Demonstrates vectorizing catalog-scale fits when redshift is already known (e.g., spectroscopy).">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_load_real_csv_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_load_real_csv`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Load and fit photometry from CSV</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="MAP-fit a model, serialize the Posterior to HDF5 with .save(), reload in a new session with Posterior.load(), and recover the fit parameters and diagnostics. Enables checkpoint-driven analysis pipelines and multi-stage fits.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_save_load_posterior_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_save_load_posterior`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Save and load a posterior to disk</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two fits on the same mock data: one with redshift fixed (spectroscopic known, free SFH/dust/met), one with redshift free (photometric only). The fixed-z fit converges to truth; free-z is degenerate with dust and SFH, showing why spectroscopy breaks the age-dust-redshift degeneracies that plague photometry-only fitting.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_specific_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_specific_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Redshift constraint: spectroscopy vs photometry alone</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Workflows
=========

End-to-end workflows: BPT classification, dust resampling, high-z LBG fits, method comparison, post-starburst recovery.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="If data is informative, the MAP estimate sits at the likelihood maximum and prior choice barely matters. If data is uninformative, the MAP slides toward the prior mode. At low S/N, the posterior shifts away from truth when the prior is strong; at high S/N both priors converge.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_prior_systematic_dust_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_prior_systematic_dust`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">How priors push the dust posterior — flat vs narrow prior on τ_diff</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="BPT ([OIII]/Hβ vs [NII]/Hα) line ratios computed directly from the rest-frame SED via continuum-subtracted boxcar integration around each line center, swept across a stellar metallicity grid. The Kewley+2001 and Kauffmann+2003 demarcation lines distinguish star-forming galaxies from AGN.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_bpt_classification_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_bpt_classification`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram: emission lines from the baked-in nebular SSP</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="References: Calzetti+2000; Conroy+2013.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_dust_mc_resampling_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_dust_mc_resampling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation: uncertainty in SED from dust parameter estimation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Lyman-break signature (sharp UV dropout at observed ≈ 4 μm) constrains age and metallicity even with just 4 bands.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_high_z_lbg_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_high_z_lbg`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">High-redshift Lyman-break galaxy: Lyman dropout signatures in JWST/HST</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reference: Conroy+2013.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_method_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_method_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recovering a truncated-skew-normal SFH from SDSS photometry via MAP</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two galaxies with different physical properties can produce nearly identical broadband fluxes when the 4000 Å break of a dusty low-z galaxy and the Lyman break of a high-z galaxy land at the same observed wavelength.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_photoz_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_photoz_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The photo-z degeneracy: dusty z ≈ 0.3 vs unobscured z ≈ 3.5</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A post-starburst galaxy shows a recent burst followed by quenching. When fit with a smooth exponential (incorrect), the fit biases the recovered SFH.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_post_starburst_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_post_starburst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Model misspecification: post-starburst galaxies reveal wrong SFH</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When did star formation stop? Optical color, the 4000 Å break, and Hα equivalent width respond on different timescales: NUV − r reddens within ~100 Myr (loss of O/B stars); D_n(4000) rises over 1–3 Gyr (A-star evolution); Hα EW drops fastest (≲10 Myr, youngest ionizing photons only).">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_quenching_diagnostics_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_quenching_diagnostics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Three diagnostics of quenching epoch in one figure</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Replacing metallicity history Z(t) with its mass-weighted mean introduces 10–23% flux errors in u and 1–6% in z. The SED is a nonlinear mass-weighted sum of SSP templates; young metal-rich stars (dominant in UV) and old metal-poor stars do not average.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_simulation_seds_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_simulation_seds`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Predicting SEDs for a simulated population: what collapsing Z(t) costs</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Stellar Population Synthesis
============================

DSPS-based SSP grids: age, metallicity, and spectral properties.

- ``plot_ssp_grid.py`` — SSP grid visualization (age, metallicity, spectrum)


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Per-SSP-library solar metallicity differs: MIST Z☉ = 0.0142, BC03/Padova Z☉ = 0.0190, PARSEC Z☉ = 0.0152, BASTI Z☉ = 0.0200. A given logzsol is only meaningful against its library.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_age_metallicity_color_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_age_metallicity_color_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-metallicity color degeneracy in SDSS colors</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The stellar populations in massive elliptical galaxies are typically α-enhanced ([α/Fe] &gt; 0) due to rapid star formation timescales that terminate before iron-peak elements fully enrich the gas. Increasing [α/Fe] shifts absorption-feature strengths — particularly Mg b and Fe5270 — which serve as diagnostics of the galaxy&#x27;s star-formation history timescale. We sweep [α/Fe] from 0.0 to 0.6 at fixed age (5 Gyr) and solar metallicity, showing the full rest-frame SED and a zoom on the optical feature region.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_alpha_enhanced_population_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_alpha_enhanced_population`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Alpha-element enhancement shifts absorption features in old stellar populations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The bolometric correction in band X is BC_X = M_bol − M_X (equivalently 2.5 log10(L_X / L_bol) up to a sign). For a single-burst SSP it traces which part of the spectrum carries the bolometric luminosity at each age: at young ages the UV dominates so BC_UV is small and BC_K is large (positive); as the population ages the SED reddens and the correction inverts — BC_K shrinks while BC_UV blows up.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_bolometric_correction_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_bolometric_correction_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bolometric correction per band as a single burst ages</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Energy is conserved: dust attenuation removes UV/optical flux, which is re-radiated in the far-infrared (Dale 2014 templates restore the balance). IGM absorption (Inoue 2014) sculpts the rest-frame continuum below the Lyman break (912 Å).">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_component_buildup_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_component_buildup`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Building the panchromatic SED component by component</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The same stellar population SED looks different depending on the units chosen for visualization. a single galaxy SED in three complementary representations on a 3-panel grid:">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_fnu_vs_flambda_units_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_fnu_vs_flambda_units`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SED Conventions: F_λ vs F_ν vs νF_ν</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Initial Mass Function (IMF) parameterizes the fraction of massive versus low-mass stars born during star formation. Chabrier, Kroupa, and Salpeter IMFs differ most in the high-mass end: Salpeter has more massive stars, producing a higher M/L ratio (more mass per unit light) and harder UV continua. We vary IMF while fixing SFH, age, and metallicity, overlaying rest-frame νL_ν to reveal the IMF signature in the SED continuum shape and M/L.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_imf_choice_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_imf_choice_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Initial Mass Function choice and stellar mass-to-light ratio</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The hydrogen-ionizing photon production rate Q_H of a simple stellar population drops by ~5 dex from 1 Myr to 100 Myr as O stars die. Different SSP libraries predict different Q_H(t) because they differ in upper-IMF treatment, stellar rotation, and (most dramatically) whether massive binaries are included — BPASS extends the Q_H-producing phase to ~30 Myr.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ionizing_lum_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ionizing_lum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionizing-photon production rate vs SSP age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The M_★/L_band ratio depends on population age, but the sensitivity varies dramatically by band. At short wavelengths (u, V), M/L is very age-sensitive: young starbursts are bright, so M/L is small; old populations are faint in the UV, so M/L grows rapidly (factor ~100 over 10 Gyr).">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_mass_to_light_band_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_mass_to_light_band_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar mass-to-light ratios across bands: age sensitivity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Different stellar population synthesis codes use different stellar spectral libraries, isochrone families, and binary treatments. The SED of a ~1 Gyr-old, solar-metallicity simple stellar population already shows visible differences in the UV (BPASS binaries add a hot continuum) and in the NIR (treatment of TP-AGB).">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_sps_library_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_sps_library_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP library comparison at a fixed age and metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A single stellar population transitions from UV-dominated (young, hot) to NIR-dominated (old, red) with age. Peak-normalized λF_λ on log-log axes makes the temperature inversion visible across five representative ages at solar metallicity.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_age_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_age_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Population Aging: SSP at Solar Metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The choice of SSP library propagates into the colors a photometric fitter recovers — a single fixed galaxy SFH and dust law, rebuilt with FSPS-MIST, FSPS-Padova/MILES, BPASS, BC03, and CB19 in turn, produces a noticeable spread in NUV − r, u − g, g − r, and r − K.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_color_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_color_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Galaxy broadband colors depend on the SSP library</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Per-SSP-library solar metallicity differs: MIST Z☉ = 0.0142, BC03/Padova Z☉ = 0.0190, PARSEC Z☉ = 0.0152, BASTI Z☉ = 0.0200.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP Grid: Age and Metallicity Evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Different Initial Mass Functions produce different continuum shapes at fixed age and metallicity. Salpeter (top-heavy) produces harder UV and near-IR continua. We compare the rest-frame νL_ν at 1 Gyr solar metallicity, peak-normalized at 5500 Å to reveal chromatic differences. The NIR is most diagnostic of IMF choice because massive stars dominate the red-giant branch.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_imf_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_imf_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">IMF choice revealed in SED continuum shape: Chabrier vs Kroupa vs Salpeter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar population synthesis templates differ across stellar evolution codes and isochrone libraries, producing measurable offsets in predicted spectra even at fixed age and metallicity. This gallery script loads four representative SSP libraries shipped with tengri (BC03, FSPS MILES, FSPS C3K, BPASS, ProGeny), constructs minimal SEDModel instances at age = 5 Gyr and Z = 0 (solar), and overlays rest-frame SED predictions (νL_ν) on log-log axes to reveal template-dependent uncertainties and continuum shape differences.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_library_shootout_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_library_shootout`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP Library Shootout: Comparing Spectral Predictions at 5 Gyr, Z=0</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Metallicity reddens the optical continuum and shifts iron-peak absorption features in the near-IR. We show five metallicity points spanning the SSP grid at fixed age (1 Gyr). Peak-normalized λF_λ makes spectral shape variations visible without large luminosity differences obscuring them.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_metallicity_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_metallicity_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Metallicity Effects on SED</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust-free UV continuum slope of an SSP swept from 10 Myr to 1 Gyr. β is fit in the Calzetti+1994 windows (1268–2580 Å) to F_λ ∝ λ^β.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_uv_slope_age_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_uv_slope_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Intrinsic UV continuum slope β vs single-burst age</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Star Formation Histories
========================

Default kernel: CIC (cloud-in-cell). `'dsps'` kernel available for cross-code parity but interpolates in log-space, annihilating the first SSP node (~3.8% mass loss) and shifting age gradients by 43%. Parametric forms (DPL, delayed-exponential, lognormal) and non-parametric (PSD-governed stochastic). Mismatch between true and assumed SFH form can produce unrecognizable posteriors.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The bursty continuity prior (Tacchella+2022, ApJ 926, 134) shares the piecewise-constant continuity SFH with Leja+2019 but doubles the Student-t scale on log-SFR ratios whose younger bin edge is recent (&lt; 1 Gyr lookback). The result is a prior that lets recent SFR variations swing by ~1 dex while keeping older history smooth (σ = 0.3 dex).">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_bursty_continuity_sigma_schedule_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_bursty_continuity_sigma_schedule`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Bursty continuity prior: bin-edge-dependent σ schedule (Tacchella+2022)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four representative (σ, τ) pairs define burstiness regimes: Smooth (σ=0.3, τ=100 Myr), Moderate (σ=1.0, τ=50 Myr), Bursty (σ=2.0, τ=20 Myr), and Extreme (σ=3.0, τ=5 Myr). Each panel shows one forward-model draw with the smooth mean SFH overlaid, illustrating the range of morphologies that each regime produces before inference.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_bursty_recovery_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_bursty_recovery`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Four regimes of stochastic-SFH burstiness from smooth to extreme</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six perspectives on chemical evolution: (1) closed-box model with varying SFR timescales; (2) cumulative metallicity from different exponential SFHs; (3) leaky-box model showing how outflow rates suppress Z; (4) age-metallicity relation across galactic radii; (5) three metallicity evolution scenarios (constant solar, linear ramp, two-step); (6) resulting integrated SEDs showing how Z(t) pathways alter optical/UV colors and absorption features. Together they show how star formation, galactic winds, and chemical enrichment control the Z(t) history and observable photometry.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_chemical_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_chemical_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Chemical evolution: How SFH and outflows shape metal enrichment history</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed mean SFH and stellar mass, continuity (Leja+2019) and field (PSD-governed) priors yield strikingly different stochastic realizations: continuity produces smooth log-normal transitions; field produces controlled burstiness governed by σ_field.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_continuity_vs_bursty_psd_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_continuity_vs_bursty_psd`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Continuity Prior vs PSD-Governed Prior: Stochastic Structure at Fixed Mean</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The timescale τ of a delayed-exponential SFH sets how quickly star formation falls after its peak: short τ means rapid decline and old stars, long τ means a sustained tail and younger mean age. We vary τ across the prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dexp_tau_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dexp_tau_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Delayed-exponential timescale τ controls decay after peak SFR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing how the rising slope α (columns) and falling slope β (rows) together control the full SFH morphology. Early-time α determines assembly speed; late-time β sets the post-peak decay. The optical SED responds across each cell. Bottom panels show representative 1D sweeps: α alone (left, at fixed β) and β alone (right, at fixed α), illustrating how each parameter independently shapes the full UV-to-IR SED.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dpl_alpha_beta_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dpl_alpha_beta_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double power-law SFH parameter space: early growth α vs late quenching β</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduction of the star-formation-history overview figure of Buchner et al. (2024, GRAHSP): galaxy SEDs for a delayed-\tau SFH (\mathrm{SFR}\propto t\,e^{-t/\tau}, CIGALE sfh_delayed) whose cutoff timescale \tau is swept from 100 Myr (yellow; SFR truncates early, old-star-dominated) to 10 Gyr (dark blue; continuously rising, young, nebular- and dust-rich). Minimal attenuation E(B-V)=0.01 is applied. The inset shows the corresponding star-formation histories.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_grahsp_paper_sfh_tau_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_grahsp_paper_sfh_tau_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP SFH figure reproduction: delayed-tau galaxy SED sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The peak time of a log-normal SFH controls when most stars formed, shifting the age structure and dramatically affecting UV slope, 4000 Å break strength, and NIR luminosity. Following Carnall+2018 / BAGPIPES, the peak is measured in cosmic time since formation (T = age - lookback); larger peak times correspond to more recent star formation. We vary the peak across its prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_lnorm_peak_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_lnorm_peak_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Log-normal peak time shifts stellar age and SED morphology</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tengri ships the non-parametric SFH priors that appear most often in Prospector papers, all with the published prior on the SFR ratios:">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_prospector_priors_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_prospector_priors_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Prospector prior families: continuity vs bursty vs Dirichlet vs PSB</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing five stochastic-SFH realizations for each combination of amplitude σ (vertical axis) and damping timescale τ (horizontal axis). Larger σ produces more dramatic bursts; longer τ sustains those bursts. Each panel shows the mean smooth SFH (dashed) and colored realizations. Bottom panels show representative SEDs for σ alone (left) and τ alone (right), illustrating how each parameter independently shapes the UV continuum and optical colors.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_burstiness_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_burstiness`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD parameter space: amplitude σ and timescale τ control burstiness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare three star-formation histories representing distinct quenching scenarios: (1) Constantly star-forming (no quenching), (2) Slowly quenched exponential decay (tau=4 Gyr, peak 6 Gyr ago), and (3) Rapidly quenched post-starburst (truncated skew-normal, peak 2 Gyr ago, width 0.3 Gyr). The resulting rest-frame SEDs exhibit markedly different colors, equivalent widths (Hα), and spectral slopes, highlighting how quenching timescale imprints on observable photometry and spectroscopy.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_quenching_pathway_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_quenching_pathway_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quenching pathways: fast vs slow termination of star formation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="CIGALE&#x27;s sfh2exp star-formation history superposes an old, exponentially declining main population with a second, more recent exponential burst that contributes a fixed fraction f_burst of the total stellar mass formed. It is the classic parametrization for post-starburst and rejuvenated systems.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh2exp_main_plus_burst_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh2exp_main_plus_burst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">sfh2exp: double declining exponential (old population + recent burst)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each parametric SFH in tengri encodes a different prior on when a galaxy forms its stars. We overlay the SFR(t) shape of nine production-status forms at their default parameter values, all integrated to the same total stellar mass, so the differences are entirely in the shape — not the normalization.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_form_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_form_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Parametric SFH form atlas</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The parametric SFH atlas (``plot_sfh_form_compare.py``) shows seven classical analytic SFH shapes. Beyond those, tengri ships three non-parametric families that bin the mass formed in successive lookback intervals — useful when the data resolve more than ~5 SFR bins and you want a flexible prior that doesn&#x27;t impose a strong shape.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_nonparametric_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_nonparametric_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Non-parametric SFH families compared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four quenching scenarios—constant SFR, exponential decline, sharp truncation, and recent burst—produce distinct SED shapes. Constant SFR yields a young, blue galaxy; sharp quenching creates old red colors; a recent burst injects young stars atop an old population. The SED reveals the full assembly history.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_quenching_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_quenching_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Quenching morphology sets the age mix and resulting SED colors</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Generate stochastic star-formation histories using the Fourier-space GP correlated field model, governed by a damped-random-walk power spectrum. Left panel shows mild burstiness (σ=0.3, τ=300 Myr); right shows strong burstiness (σ=1.0, τ=100 Myr). Five realizations appear in each panel, with the smooth mean SFH overlaid.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_stochastic_sfh_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_stochastic_sfh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stochastic SFH samples from GP-correlated fields with different burstiness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How observable is an underlying ancient burst (10 Gyr ago) beneath a young (300 Myr) starburst? outshining problem in broadband photometry (Trager+ 2000, Renzini 2006): the young burst&#x27;s UV emission completely dominates over the ancient burst&#x27;s optical/IR, rendering the ancient population invisible to broadband SED fitting.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_two_burst_observability_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_two_burst_observability`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The outshining problem: young bursts eclipse ancient populations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A common SED-fitting failure mode: pick a smooth parametric SFH (delayed exponential, tau-model, lognormal) for a galaxy whose true star-formation history has short-timescale bursts. The continuum-anchored bands (optical, NIR) absorb the mass and the fit looks plausible — but the UV bands, where young O/B stars dominate, carry the residual of the recent burst.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_wrong_model_trap_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_wrong_model_trap`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fitting a stochastic SFH with a smooth parametric prior leaves a UV residual</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Metallicity
===========

Per-SSP-library Z☉ differs: MIST 0.0142, BC03/Padova 0.0190, PARSEC 0.0152, BASTI 0.0200. Cross-code comparisons must reason in absolute log(Z). Stellar and gas-phase Z are separate knobs. Age–metallicity degeneracy in broadband data. α-element enhancement.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The [α/Fe] abundance ratio encodes the chemical enrichment history: rapid enrichment by core-collapse supernovae before Type Ia SNe begin leads to high [α/Fe]. In the SED, enhanced alpha-elements suppress iron absorption lines in the optical (especially around 4000–5000 Å) because the higher abundance of alpha elements shifts the line-blanketing opacity. We sweep [α/Fe] on a quiescent passively evolving galaxy where iron features dominate the continuum absorption.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_alpha_fe_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_alpha_fe_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Alpha-element enhancement suppresses iron absorption features</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar metallicity affects the stellar continuum shape and overall energy balance. Dust emission responds to absorbed stellar photons: metal-poor hot stars emit bluer light with less IR-absorbed energy, while metal-rich cooler stars are less bright in the UV but more absorbed in the optical/NIR. We sweep stellar metallicity on a young star-forming galaxy at z = 0.2 with dust attenuation and thermal emission from warm dust.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_logzsol_panchromatic_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_logzsol_panchromatic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Metallicity shapes panchromatic SED with dust emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Metal-rich young populations and metal-poor old populations can produce similar optical colors — a fundamental degeneracy in galaxy fitting. This 3×4 grid shows normalized rest-frame continua at nine points in the age–metallicity plane, with each row fixed at one lookback-formation age and each column fixed at one metallicity. Dust is zeroed to expose the clean stellar continuum shape.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_metallicity_age_grid_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_metallicity_age_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-metallicity degeneracy in the stellar continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Metallicity evolution Z(t) depends on the balance between metal production (in supernovae) and metal removal (via outflows). This four-panel figure shows how different star formation timescales and outflow efficiencies η alter the enrichment history relative to a closed box (zero outflow). Top-left: closed-box enrichment timescale dependence. Top-right: impact of variable outflow rates. Bottom-left: closed vs leaky enrichment under constant SFR. Bottom-right: age-metallicity relation analog — how different assembly epochs lead to different final metal content.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_zh_evolution_compare_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_zh_evolution_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Chemical evolution: closed-box vs leaky-box enrichment histories</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Nebular Emission
================

Emission lines are vacuum throughout: Hα is 6564.61 Å, not the 6562.8 Å air
value. Mixing the two shifts every line centroid.

``neb={'type': ...}`` takes ``ssp``, ``cue``, ``cb19``, ``cloudy`` or ``none``.
The default, ``ssp``, uses the emission already baked into a with-nebular (wNE)
SSP grid. The live backends instead compute it, and expect a bare stellar grid.
Feed a bare grid to the baked-in path and both continuum and line fluxes come
out low, with no error raised.

Gas-phase metallicity is its own knob and does not follow the stellar one.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cue neural emulator responds to 12+ parameters. We show how each knob (ionization, metallicity, density, abundances, ionizing slope) moves a galaxy on the BPT-N plane log [OIII]/Hβ vs log [NII]/Hα. Each panel sweeps one parameter while holding fiducial values fixed. Kewley+2001 and Kauffmann+2003 demarcations shown for reference.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_cue_flexibility_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_cue_flexibility`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue nebular knobs affect BPT positions individually</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Show how the Cue neural emulator (Li+2025) maps the 2D parameter space (log U, log Z_gas) onto three classical BPT diagnostic diagrams. Lines of constant log U (varying metallicity) and constant log Z (varying ionization) show the full grid&#x27;s coverage and demarcation positions.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_cue_grid_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_cue_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue nebular grid on BPT diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Emission lines are vacuum throughout: [OIII] = 5008.24 Å, [NII] = 6585.28 Å, Hα = 6564.61 Å, Hβ = 4862.68 Å. Overlays Kewley+2001 SF/AGN demarcation and Kauffmann+2003 SF/composite line.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_diagram_population_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_diagram_population`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram: star-forming galaxies, AGN, and shocks</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cue knobs fesc (ionizing-photon escape fraction) and logU (HII region ionization parameter) jointly govern the line spectrum of a star-forming galaxy: escape fraction sets how many ionizing photons reach the gas, logU shifts the resulting ionization balance of the gas they ionize. We map the response of three diagnostic lines/ratios on a 2-D grid.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_fesc_logu_atlas_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_fesc_logu_atlas`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue 2-D atlas: ionizing escape fraction × ionization parameter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cue (Li, Leja &amp; Speagle 2023) maps a four-dimensional HII region control space — ionization parameter log U, gas-phase metallicity log Z_gas, ionizing-spectrum shape, and dust-to-metal ratio — onto an emission-line spectrum. A two-dimensional sweep over the two knobs most users will turn (``log U`` and log Z_gas) is shown for four diagnostic line ratios.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_parameter_atlas_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_parameter_atlas`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comprehensive 2D sweep of ionization parameter and metallicity (Cue)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Lyα-specific escape fraction f_esc_lya sets what fraction of Lyα photons can escape the ISM without scattering. Higher f_esc_lya suppresses the Lyα emission line while leaving other nebular lines unchanged.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_fesc_lya_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_fesc_lya_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyα escape fraction controls Lyman-alpha strength</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We sweep the ionizing-photon escape fraction f_esc from 0 to 1.0 at fixed log U and metallicity, showing both the broadband SED response and a zoomed view of the critical Lyman-continuum (912 A) region. The Lyman edge deepens as ionizing photons escape the ISM unabsorbed, suppressing optical line ratios simultaneously.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_fesc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_fesc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Escape fraction reshapes the SED from the Lyman continuum to optical lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nebular free-free, free-bound, and two-photon emission respond to gas-phase metallicity (``logZ_gas``) through changes in metal cooling efficiency and ionization balance. metallicity sensitivity of the nebular continuum at fixed ionization parameter.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_gas_z_continuum_effect_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_gas_z_continuum_effect`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas-phase metallicity effect on nebular continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Murphy+2011 SFR-Hα relation requires ionizing photons from stars younger than ~10 Myr. Constant-SFR models at ages 1–300 Myr show the calibration breaks at young (&lt;10 Myr; insufficient ionizing photons) and old (&gt;100 Myr; all stars too old to ionize) populations. We sweep stellar metallicity to show the calibration validity range is weakly sensitive to Z: higher Z reduces ionizing photon production, compressing the valid age window slightly toward older ages.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_halpha_sfr_calibration_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_halpha_sfr_calibration_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hα SFR calibration breaks at young ages, weakly dependent on metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A galaxy&#x27;s velocity dispersion sigma_v_kms broadens every spectral feature — including the nebular emission lines — from a few tens of km/s (dynamically cold disks) to several hundred km/s (dispersion-dominated spheroids and AGN narrow-line regions). The broadening is a forward-model convolution applied to the predicted spectrum, so it is only visible when the instrument line-spread function is finer than the velocity width: we therefore predict a spectrum on a high-resolution grid (R ~ 10000) around the [O III] λλ4959,5007 + Hβ region and sweep sigma_v_kms.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_line_sigma_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_line_sigma_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission line broadening traces gas kinematics</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 1-D log Z_gas sweep on the SED scale, complementing the 2-D atlas in plot_cue_parameter_atlas.py and the line-ratio projection in plot_strong_line_metallicity_diagnostics.py. Reader sees how every strong optical line moves together as Z_gas climbs, with [N II]/Hα and [O III]/Hbeta the textbook diagnostics.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logz_gas_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logz_gas_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas metallicity reshapes the optical nebular continuum and line forest</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Lyα rest-frame wavelength is 1216 Å (vacuum). EW peaks at 3–5 Myr when O-type stars dominate ionization, then decays past 10 Myr. Higher metallicity suppresses ionizing photon production, reducing peak EW.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_lyalpha_ew_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_lyalpha_ew_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha equivalent width peaks at young ages, varies with gas metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Ionizing photon production declines rapidly with stellar population age (~t^-1). We show how nebular line strength evolves from young (50 Myr) to old (5 Gyr) populations.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_age_dependence_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_age_dependence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular emission fades with stellar population age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Young massive stars produce harder ionizing continua and drive the nebular emission toward higher [O III]/Hbeta. We sweep the SFH timescale tau_gyr from 0.1 to 2 Gyr on a single dual power-law model and plot the resulting line ratios against the Kewley+2001 / Kauffmann+2003 demarcation curves. The locus migrates from the star-forming wing into the composite region as the population ages — SFH timescale is the upstream knob behind the BPT ionization sequence.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_bpt_logu_grid_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_bpt_logu_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar-population age moves a galaxy on the BPT diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nebular gas density controls ionization balance and recombination rates, affecting emission line strengths. Higher density increases cooling efficiency, shifting line ratios through recombination rate changes.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_density_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_density_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular density affects recombination and cooling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare four nebular emission backends on identical star-forming spectra:">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_nebular_backends_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_nebular_backends`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular backends: Cue, CloudyGrid, SSP-embedded, and BakedIn</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The optical [O III] 5007 / Hβ ratio is set primarily by the ionization parameter log U: more energetic Lyman continuum photons per H atom ionize more O+ to O++, while Hβ recombination depends mostly on the ionizing photon rate (``Q_H``) and is roughly insensitive to log U. The ratio therefore rises monotonically with log U at fixed gas metallicity.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_oiii_hbeta_logu_at_fixed_z_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_oiii_hbeta_logu_at_fixed_z`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">[O III]/Hβ vs ionization parameter at fixed gas metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hydrogen-ionizing photon production (Q_H, photons/s per solar mass) depends critically on stellar population age. Young starbursts (age ≈ 3–5 Myr) produce ionizing photons at peak rates; by 100 Myr, Q_H drops by ~3 orders of magnitude. We show how this evolution varies across metallicity Z = [-1.0, -0.5, 0.0, +0.3] using FSPS bare-stellar (non-nebular) SSP templates, as ionizing photons are consumed by CLOUDY during wNE SSP generation and would appear suppressed.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_qh_vs_age_metallicity_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_qh_vs_age_metallicity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionizing photon production rate Q_H peaks sharply with stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Shock emission (MAPPINGS V models) can mimic AGN on the BPT diagram. We show how shock velocity, gas density, and magnetic field strength affect line ratios and diagnostic positions. Four-panel layout shows velocity and density sequences on BPT, line ratios vs velocity, and magnetic field strength.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_shock_emission_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_shock_emission`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAPPINGS V shocks: velocity, density, and magnetic field effects</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Five widely-used optical strong-line metallicity diagnostics evaluated across the Cue logZ_gas prior. Each one carries a different systematic — Pettini &amp; Pagel 2004 O3N2 saturates at high Z, the R23 ratio is double-valued (Pagel+1979), N2 (Marino+2013) is monotonic but small dynamic range, the [O III]/[O II] diagnostic tracks ionization, and [S II]/[O II] is a low-ion proxy.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_strong_line_metallicity_diagnostics_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_strong_line_metallicity_diagnostics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Strong-line gas-phase metallicity diagnostics</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Dust Attenuation
================

Two-component Charlot & Fall geometry: ``dust_tau_bc`` on the birth clouds,
``dust_tau_diff`` on the diffuse ISM. ``dust_slope`` defaults to -0.7, the
diffuse-ISM value; -1.3 is the birth-cloud one. The 2175 Å bump is a separate
always-on modifier, ``dust_bump_strength``, defaulting to 0.0 — Calzetti
carries no bump unless you ask for one.

Dust emission templates load from ``data/``. There is no analytic fallback: a
missing template raises ``FileNotFoundError`` rather than quietly substituting
a worse model.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Charlot &amp; Fall 2000 two-component dust model splits attenuation into a birth-cloud component (``τ_bc``) that only the youngest stellar ages see, and a diffuse-ISM component (``τ_diff``) that attenuates all stellar light. The two are degenerate for an old population but separate cleanly for a young one.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_birth_cloud_vs_diffuse_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_birth_cloud_vs_diffuse`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth-cloud vs diffuse-ISM dust: age dependence and parameter degeneracies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Kriek &amp; Conroy attenuation law has two degrees of freedom: bump strength and UV slope (δ). Varying both reveals how steeper UV slopes suppress the apparent prominence of the 2175 Å bump relative to the surrounding continuum. We show a 2×2 grid: rows sweep bump strength (0–2 at fixed δ), columns sweep δ slope (−1, +0.5 at fixed bump), revealing the synergy — a steep negative slope (blue wing) enhances bump visibility, while shallow positive slopes (flattened UV) bury the bump in the continuum.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_bump_delta_joint_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_bump_delta_joint_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2175 Å bump × UV slope interaction in Kriek & Conroy attenuation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cardelli+1989 Milky Way attenuation curve is a family parameterized by R_V = A_V / E(B-V). Smaller R_V (≲ 3) gives a steeper UV rise and stronger 2175 Å bump (denser lines of sight, small grains dominate); larger R_V (≳ 4.5) flattens the UV slope (processed grains, larger sizes).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_cardelli_rv_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_cardelli_rv_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cardelli MW attenuation: sweeping R_V</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust geometry determines how dust affects starlight. A screen (foreground dust) filters the light as it leaves the galaxy: transmission = exp(-τ_λ). A mixed geometry (dust uniformly distributed with stars) is more gentle: transmission = (1 - exp(-τ_λ)) / τ_λ.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_geometry_screen_vs_mixed_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_geometry_screen_vs_mixed`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Screen vs. mixed dust geometry: identical optical depths, different SEDs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each of the bundled dust-attenuation laws applied to the same intrinsic SED at the same V-band optical depth — so the differences between the curves are entirely in the wavelength dependence of the attenuation. The intrinsic (unreddened) SED is shown in black for reference.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_law_application_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_law_application`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The same galaxy reddened by every attenuation law in the registry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For a fixed star-forming galaxy with τ_V = 1 (a moderate attenuation), six common attenuation laws produce six visibly different reddened UV slopes β. The intrinsic SED has β ≈ −2.3; SMC steepens β to ≈ +0.4; Calzetti / Salim leave a flatter β ≈ −0.5. The spread (~1 mag of UV slope at fixed τ_V) is the systematic an SED fitter inherits if its dust-law assumption is wrong.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_law_uv_slope_response_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_law_uv_slope_response`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Attenuation law leaves a distinct UV-slope fingerprint</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The power-law slope δ steepens (negative) or flattens (positive) UV attenuation relative to the optical, controlling whether dust absorbs more or less light at short wavelengths. We vary δ with elevated τ_bc and τ_diff to make slope effects visible (low dust opacities wash out the continuum slope).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_slope_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_slope_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation curve slope controls UV vs optical hardness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Public dust attenuation laws applied to the same intrinsic SED at the same V-band optical depth (τ_V = 1.0), illustrating how dust geometry and grain-size composition vary across the local universe.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_galactic_zoo_dust_laws_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_galactic_zoo_dust_laws`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation laws across the galaxy zoo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduction of Fig. 7 of Buchner et al. (2024, GRAHSP): a star-forming galaxy SED from intrinsic (dark blue) to strongly attenuated (dark red) as the diffuse color excess E(B-V) is swept from 0.01 to 10. Energy balance routes the attenuated UV/optical light into the far-IR dust bump (Dale 2014), so the curves pivot about the FIR peak while the UV is progressively suppressed.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_grahsp_paper_fig7_galaxy_attenuation_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_grahsp_paper_fig7_galaxy_attenuation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP Fig. 7 reproduction: attenuation of the galaxy model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reddy et al. (2015) derived a dust attenuation curve from Balmer decrements of z ~ 1.4–2.6 star-forming galaxies in the MOSDEF survey. It is shallower in the UV than the SMC curve but has a lower total-to-selective ratio (``R_V = 2.505``) than Calzetti&#x27;s local starburst law (``R_V = 4.05``) — a combination relevant when fitting rest-UV/optical SEDs of high-z galaxies. FSPS exposes this curve; tengri provides it as the reddy15 dust law.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_reddy15_highz_curve_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_reddy15_highz_curve`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Reddy+2015 high-redshift attenuation curve</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed UV slope β_UV (the observable astronomers measure), many (R_V, A_V) pairs produce identical colors — this is a classical dust modeling pitfall. Shows β_UV as contours on the (R_V, A_V) grid for Cardelli MW attenuation. Standard reference points (SMC, LMC, Milky Way diffuse, Calzetti starburst) sit on different iso-β_UV contours, illustrating why dust-law assumptions strongly bias inferred properties.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_rv_av_uv_slope_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_rv_av_uv_slope_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rv and Av degeneracy in UV slope: the Calzetti trap</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Charlot &amp; Fall two-component dust model separates birth-cloud dust (young stars only, age &lt; ~10 Myr) from diffuse ISM dust (all stars). Two panels show: (left) V-band transmission versus age for three (τ_bc, τ_diff) combinations, revealing the sharp ~10 Myr transition; (right) full transmission spectra for 1 Myr and 1 Gyr stars under the same dust column.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_two_component_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_two_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Two-component dust: birth cloud obscures only young stars</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The IRX–β relation connects the UV continuum slope β (1250–2600 Å) with the infrared excess IRX = log₁₀(L_IR / L_UV). This diagram reveals dust reddening and star formation rate indicators in galaxies. Here we:">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_usecase_irx_beta_meurer_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_usecase_irx_beta_meurer`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">IRX–β diagram: infra-red excess vs UV slope (Meurer+1999)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 2175 Å UV bump sits atop a power-law continuum. Varying the slope parameter δ (delta) in the Kriek &amp; Conroy attenuation law steepens or flattens the UV continuum, which changes the bump&#x27;s prominence relative to the surrounding curve. We zoom on rest-frame 1500–3500 Å to isolate the bump region and show how δ ∈ [−1, +0.5] reshapes the attenuation curve.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_uv_bump_strength_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_uv_bump_strength_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV bump shape controlled by attenuation curve slope</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Charlot &amp; Fall 2000 two-component dust model conserves energy: every UV photon attenuated by the dust must come back out as IR re-emission. We sweep τ_diff from 0 to 2 mag and on each step plot two quantities — the absorbed UV power L_abs(λ&lt;3000 Å) inferred from the difference of (no-dust) minus (with-dust) attenuated SEDs, and the integrated IR luminosity L_IR(8–1000 μm) from the IR re-emission template (Dale+2014 here).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_uv_ir_energy_balance_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_uv_ir_energy_balance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV-IR energy balance: absorbed = re-emitted</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WG00 radiative-transfer grid (FSPS dust_type=3) spans three large-scale star-dust geometries — shell (a foreground screen), cloudy (a homogeneous star-dust mix), and dusty (a clumpy two-phase medium) — crossed with two grain populations (Milky-Way and SMC). At a fixed tau_V these choices set the shape of the transmission exp(-A(lambda)): the foreground screen is the reddest (steepest UV), while the mixed and clumpy geometries are progressively grayer because short-wavelength photons escape through low-opacity sightlines.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_wg00_geometry_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_wg00_geometry_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Witt & Gordon 2000: geometry and grain type at fixed optical depth</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For a foreground dust screen the attenuation curve has a fixed shape — its amplitude scales with tau_V but the UV-to-optical ratio is constant, so a single k(lambda) law captures it. Witt &amp; Gordon (2000) showed this breaks down once dust and stars are mixed: high-``tau_V`` sightlines self-shield, the short-wavelength photons preferentially escape through low-opacity channels, and the effective curve greys (flattens) as tau_V rises. The curve shape is therefore a function of tau_V — which is exactly why tengri ships WG00 as a radiative-transfer table (FSPS dust_type=3), interpolated in tau_V, rather than a fixed-shape law.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_wg00_tau_v_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_wg00_tau_v_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Witt & Gordon 2000: the attenuation shape greys with optical depth</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Dust Emission
=============

Dust emission templates auto-load from ``data/``; analytic fallbacks are not suitable for science. PAH features in Draine & Li templates (q_PAH and U_min sweeps). Temperature sweeps. Template libraries: BOSA, THEMIS, PAHspec, Astrodust (HD23).


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Per-component breakdown (Astrodust continuum, PAHs, spinning dust) at the Hensley &amp; Draine 2023 fiducial ionization parameter \log_{10} U = 0.2.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_03_components_at_fiducial_U_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_03_components_at_fiducial_U`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH per-component decomposition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The BOSA infrared template library is parametrized jointly by total infrared luminosity log L_TIR and specific star formation rate log sSFR. Neither axis alone tells the full story: at fixed sSFR the FIR peak migrates with L_TIR (dust temperature), while at fixed L_TIR the PAH mid-IR forest brightens with sSFR. Three side-by-side panels at fixed sSFR overlay three L_TIR values each, making the 2-D dependence legible in a single figure rather than two skinny 1-D loops.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA library: PAH features and FIR peak depend on both sSFR and L_TIR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Dale et al. (2014) IR template family can be combined with a pure-AGN (&quot;quasar&quot;) template to represent dust heated by an obscured AGN in addition to the star-forming ISM. tengri reproduces CIGALE&#x27;s convention, where the AGN is a separate power source added on top of the stellar-heated dust:">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dale2014_agn_fraction_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dale2014_agn_fraction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dale 2014 dust IR: AGN fraction (CIGALE-faithful additive mixing)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust temperature T sets the far-infrared peak via Wien&#x27;s displacement law. Higher T shifts the peak blueward into the mid-IR; lower T shifts it redward toward the submillimeter.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_T_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_T_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modified Blackbody Dust Temperature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 2-D grid on the Draine &amp; Li 2007 template library: rows step through PAH mass fraction q_PAH (controls mid-IR PAH-feature strength), columns through the minimum radiation field U_min (sets the diffuse dust temperature, i.e. the FIR peak position). The two axes act nearly orthogonally — a surprise for anyone who would lump them together as &quot;PAH knobs.&quot;">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_qpah_umin_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_qpah_umin_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The q_PAH and U_min knobs move PAH amplitude and FIR peak independently</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All dust IR-emission libraries shipped in tengri, shown on two scales:">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_ir_library_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_ir_library_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR-emission library comparison: models and templates</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The mid-infrared ionization-parameter sensitivity is library-specific, but the FIR-peak migration with rising log U is a universal prediction. We overlay the Hensley &amp; Draine 2023 (Astrodust+PAH) and the Draine+2021 PAHspec libraries at the same three log U values to surface where the two agree (FIR peak position) and where they differ (MIR PAH-feature strength and the Astrodust silicate plateau near 18 microns).">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_logu_cross_library_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_logu_cross_library`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Two PAH libraries respond to log U with the same FIR-peak migration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Casey 2012 modified blackbody dust SEDs across the canonical fitter&#x27;s two knobs — dust temperature T_dust and emissivity index β. Each curve in the top panel is a fixed β = 1.8 MBB swept in T; the bottom panel fixes T = 30 K and sweeps β. The peak shifts by ~40 μm per 10 K of warming; the sub-mm slope steepens by one power-law index per Δβ = 1.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_mbb_temperature_beta_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_mbb_temperature_beta_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Modified blackbody: T_dust × β grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Draine &amp; Li (2007) dust model naturally separates three emission regimes via its parameters. Varying q_PAH (PAH mass fraction) and U_min (minimum radiation-field intensity) traces three archetypal SED shapes:">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_pah_warm_cold_split_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_pah_warm_cold_split`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR SED: PAH / Warm grain / Cold grain decomposition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep across the 13 published PAHspec starlight spectra (mMMP, m31bulge, BC03/BPASS SSPs) at fixed ionization parameter. Demonstrates strong dependence of PAH features on starlight hardness.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_pahspec_starlight_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_pahspec_starlight_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine+2021 PAHspec: starlight-spectrum sweep at fixed log U</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="PAH mass fraction controls strength of polycyclic aromatic hydrocarbon mid-infrared emission features. Higher q_PAH produces stronger features at 3.3, 6.2, 7.7, 8.6, 11.3 μm. Range varies by dust model.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_qpah_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_qpah_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PAH Mass Fraction (q_PAH)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For Draine &amp; Li (2007) dust at fixed mass, raising the diffuse radiation field intensity U_min does two things at once: it shifts the SED peak blueward (warmer dust) and proportionally boosts the total far-IR luminosity (``L_IR`` ∝ U_min). The standard T_peak–``L_IR`` correlation seen in observations is the joint footprint of these two effects.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_tdust_vs_lir_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_tdust_vs_lir`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radiation field strength sets both dust peak temperature and L_IR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Jones et al. (2017) THEMIS dust model distributes grains over a range of starlight intensities U with a power law dU/dM \propto U^{-\alpha}. The slope alpha controls how much warm, intensely-illuminated dust contributes relative to the cold diffuse component: a smaller alpha puts more mass at high U, shifting the FIR peak blueward and filling in the mid-IR.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_alpha_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_alpha_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS dust IR: radiation-field slope (alpha)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep hydrocarbon grain content across the THEMIS grid at fixed minimum radiation field strength. PAH-like mid-IR features strengthen with q_HAC while FIR continuum remains essentially unchanged.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_qhac_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_qhac_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: q_HAC sweep at fixed U_min</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The starlight intensity floor U_min sets the temperature of the diffuse-ISM component in template-based dust libraries. Two perspectives:">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_umin_cross_library_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_umin_cross_library`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Minimum radiation field U_min: DL07 vs THEMIS FIR peak and sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust re-radiates absorbed starlight across a broad range of temperatures: colder dust (e.g., diffuse cirrus at ~20 K) peaks in the far-infrared (~250 μm), while warmer dust grains (e.g., starburst regions at ~40 K) peak at shorter wavelengths (~50–100 μm).">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_warm_cold_dust_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_warm_cold_dust_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR SED: Warm and cold dust decomposition</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

AGN Models
==========

Torus models in `components/agn/torus.py` are toy models; SKIRTOR is the one for science. Disc continua (multicolor, KD18, relagn, qsogen), narrow-/broad-line and FeII emission, polar-dust and Type 1/2 attenuation, X-ray corona via α_ox relation. Cross-validated against CIGALE, GRAHSP, AGNfitter.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The X-ray corona response of an AGN depends jointly on bolometric luminosity (which sets the X-ray normalization through the Lusso &amp; Risaliti L_X-L_UV correlation) and on the UV-to-X-ray slope alpha_OX (which sets the relative balance of UV and X-ray emission). Four panels at log L_bol = 44, 45, 46, 47 erg/s overlay three alpha_OX values each, showing that the absolute X-ray luminosity scales with L_bol while the X-ray-to-UV ratio is set independently by alpha_OX.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_alpha_ox_lbol_2d_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_alpha_ox_lbol_2d`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray corona shape across the alpha_OX vs log L_bol plane</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Shakura-Sunyaev thin disc model shows how the big blue bump (BBB) peak shifts to longer wavelengths as black-hole mass increases. At fixed Eddington ratio log(L_bol / L_Edd) = -1.0, the disc temperature scales as T_{\rm in} \propto (\dot{m} / m_\odot)^{1/4}, where the inner temperature determines the location of peak νLν. Higher mass → lower accretion rate → cooler disc → redder peak.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_bbb_mbh_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_bbb_mbh_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Big Blue Bump: multicolor disc temperature evolution with black-hole mass</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A single log L_bol = 12.5 composable AGN built up component by component — disc alone, +torus, +narrow lines, +broad lines — so the reader can see what each block contributes to the total spectrum. The bottom panel shows the same decomposition stacked so the layers add up to the full SED.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_components_breakdown_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_components_breakdown`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN composite SED: per-block decomposition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All thirteen accretion-disc backbones registered under agn.disc.type, at fixed bolometric luminosity log L_bol = 12.5 (in log L_sun), evaluated in isolation with the host suppressed and no torus/lines/dust. The differences between the curves are entirely how each model partitions the disc power across wavelength: pure blackbody vs warm Comptonization, relativistic vs Newtonian potential, radiatively efficient thin disc vs inefficient ADAF, empirical composite vs first-principles continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_disc_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_disc_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN disc continuum: every registered model at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Iron pseudo-continuum (Fe II) emission in AGN produces characteristic humps in the near-UV and optical bands. The strength and shape are governed by the Fe II equivalent width and ionization state, parameterized in tengri by the agn_fe2_strength parameter relative to H-beta (Balmer lines).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_feii_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_feii_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fe II pseudo-continuum strength evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Until recently the agn_ parameters were declared with fixed* defaults and no prior range, so the build grammar&#x27;s FREE controls (``agn={&#x27;all_params&#x27;: FREE}``, recipes.agn_panchromatic()) silently resolved every AGN parameter to a constant — a fit would freeze the entire AGN sector with no error. The registry now gives each parameter a physically-motivated Uniform/``LogUniform`` prior (Nenkova+2008, Kubota &amp; Done 2018, Stalevski+2016 grid extents), so FREE actually frees them.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_free_param_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_free_param_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN parameters are free-able now — and every one moves the SED</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four AGN configurations of increasing physical complexity at the same bolometric luminosity (log L_bol = 12.5 in L_sun units) — bare multicolor disc, +SKIRTOR torus, +NLR narrow-line forest, and an empirical QSOgen template that bundles all of the above. The reader sees which spectral feature each block introduces (mid-IR torus bump, optical narrow lines, broad UV continuum) and which are essentially universal across the modeling choice.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_hierarchy_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_hierarchy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Building up an AGN SED: disc, then torus, then lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A Seyfert galaxy SED is decomposed photometrically by varying the AGN contribution fraction agn_lum_ratio from 0 (pure host) to 1.0 (pure AGN) to 0.5 (composite). how to isolate the AGN contribution from the host galaxy using a single model and varying a structural parameter — useful for diagnosing photometric AGN contamination.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_host_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_host_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN host-galaxy decomposition: disentangling Seyfert contributions</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The renderable line backbones registered under the three composable line selectors — agn.nlr (narrow-line region), agn.blr (broad-line region), and agn.feii (iron pseudo-continuum) — each layered on the same disc + torus at fixed log L_bol = 12.5. The backbone controls which optical/UV features the model produces: narrow forbidden lines, broad permitted lines, or the blended Fe II forest.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_lines_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_lines_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN emission-line backbones compared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust-free quasar spectra are intrinsically blue in the UV and optical. Adding a polar-dust attenuation component reddens the accretion-disc continuum: increasing the polar-dust reddening agn_polar_ebv (E(B−V), [mag]) from 0 to 0.4 walks the SED from unobscured type-1 QSO colors to a moderately dust-reddened continuum, while the absorbed UV energy is re-radiated as a polar-dust infrared bump.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_qsogen_ebv_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_qsogen_ebv_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSO continuum: polar-dust reddening tunes UV to optical color</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The QSOgen model (Temple+2021) includes empirical UV/optical emission-line forest and broad Balmer continuum. The relative strength of these line features with respect to the continuum obeys the Baldwin effect: luminous quasars show weaker equivalent-width emission lines (the line flux grows sublinearly with continuum). This sweep shows the Baldwin effect in the QSOgen template across six decades of bolometric luminosity (log L_bol = 9 to 13 L_sun), revealing the Ly-alpha + C IV feature cluster around 1000–1600 Å and optical hydrogen Balmer lines (Hα, Hβ).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_qsogen_emline_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_qsogen_emline_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen emission lines: Baldwin effect across AGN luminosity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="All ten dusty-torus libraries registered under agn.torus.type, reprocessing the same accretion-disc continuum at fixed log L_bol = 12.5 (in log L_sun) and standard inclination. The disc is held at multicolor (Kubota &amp; Done 2018) so the differences in the curves are entirely how each torus library geometrically distributes hot grains and re-emits the absorbed UV in the MIR — clumpy radiative transfer (SKIRTOR, CLUMPY, CAT3D-WIND) vs smooth-dust grids (Fritz, Silva) vs phenomenological graybodies.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_torus_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_torus_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN dusty torus: library comparison at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The ultraviolet-to-X-ray spectral slope α_OX encodes the fundamental physics of accretion discs. At higher bolometric luminosities, discs shift toward cooler effective temperatures and steeper UV slopes, reducing the X-ray-to-UV flux ratio. We compute α_OX for 15 tengri AGN disc models (multicolor, no torus/lines) across log L_bol ∈ [10.5, 14.0], measuring at rest-frame 2500 Å (UV) and 2 keV (X-ray). The Lusso &amp; Risaliti 2016 fit α_OX = −0.166 log L_2500 + 4.74 captures the observational trend that luminous quasars are more UV-bright and X-ray-weak.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_alpha_ox_lusso_risaliti_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_alpha_ox_lusso_risaliti`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lusso & Risaliti 2016: α_OX – L_UV relation for AGN discs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduces the UV-to-X-ray connection panel from Yang et al. 2020 (X-CIGALE Fig. 3): the X-ray corona is normalized through the Just+07 alpha_OX-L_2500 relation, anchored at the disc-derived L_2500. Offsets delta_alpha_OX from -0.3 to +0.3 dex pivot the X-ray power-law about the 2500 A anchor — the disc UV stays fixed (single curve at log lam &gt; 1), only the X-ray normalization moves.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_alpha_ox_uv_xray_connection_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_alpha_ox_uv_xray_connection`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">delta_alpha_OX pivots the X-ray spectrum about the disc UV anchor</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The CAT3D-Wind torus (Hönig &amp; Kishimoto 2017) splits the circumnuclear dust into a mid-plane clumpy disc plus a polar outflow (&quot;wind&quot;). Its infrared reprocessing is controlled by three observables: the wind mass fraction fwd, the radial cloud-distribution index a, and the viewing angle cos i.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_cat3d_wind_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_cat3d_wind_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">CAT3D-Wind clumpy torus: wind fraction and viewing angle</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The agn.disc, agn.lines, agn.feii, agn.torus, agn.atten sub-blocks of SEDModel.build are composable: turning one on at a time and overlaying the all-on reference (dashed gray) shows which features each sub-block contributes. Five panels at fixed log L_bol = 12.0, all built via the public nested-dict grammar:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_composable_block_toggles_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_composable_block_toggles`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cumulative buildup of the GRAHSP AGN recipe, one sub-block at a time</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The composable AGN grammar (``agn.disc``, agn.torus, agn.lines, agn.feii, agn.atten) lets the user mix sub-blocks across model families. Same SEDModel.build call, three different physics tuples:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_composable_recipes_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_composable_recipes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Three AGN recipes built by swapping selectors, not call sites</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A toy single-temperature blackbody torus implemented as a modern SEDModelComponent subclass, discoverable through SEDModel.build and composable with other AGN blocks. The SEDModelComponent pattern is the recommended path for any new SED physics — AGN, dust, or stellar.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_custom_torus_extension_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_custom_torus_extension`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Custom AGN torus model via SEDModelComponent and direct integration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The GRAHSP AGN model (Buchner+ 2024) optionally adds a Balmer continuum following Grandi (1982): a 15,000 K blackbody truncated at the Balmer edge (3646 Å) and Gaussian-broadened by the line width. Together with the FeII forest it builds the &quot;small blue bump&quot; seen blueward of ~4000 Å in type-1 quasars.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_balmer_continuum_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_balmer_continuum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP Balmer continuum: building the small blue bump</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The GRAHSP big blue bump can be modeled two ways. The default is a smooth bending power-law (Ryde 1998 form) with free UV/optical slopes and a bend wavelength. The physical alternative is the Netzer accretion-disc grid (Netzer &amp; Trakhtenbrot 2014), tabulated over black-hole mass, spin and Eddington ratio — selected with disc_model=&quot;netzer&quot; plus disc_m / disc_a / disc_mdot.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_disc_vs_bbb_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_disc_vs_bbb`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP accretion disc: Netzer templates vs the bending power-law</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The iron pseudo-continuum (the &quot;FeII forest&quot;) is a defining feature of type-1 AGN optical/UV spectra. GRAHSP offers two templates: the photoionization model of Bruhweiler &amp; Verner (2008) (the upstream default) and the empirical Veron-Cetty, Joly &amp; Veron (2004) template. They differ most in the relative strength and shape of the UV (2200–3000 Å) and optical (4400–5400 Å) multiplet blends.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_feii_templates_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_feii_templates`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP FeII forest: Bruhweiler+Verner 2008 vs Veron-Cetty 2004</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduction of Fig. 1 of Buchner et al. (2024, GRAHSP): how the individual model components sum to the total emission (black). The AGN side is the GRAHSP bending power-law disk/BBB (blue), iron + emission-line forest (red), and the dusty torus (yellow dashed), normalized so the disk has L_{5100\,\mathrm{\AA}}^{\rm AGN}=10^{44}\,\mathrm{erg\,s^{-1}} =10^{37}\,\mathrm{W} (blue square); the torus is anchored at 12 μm (yellow diamond). The host is a stellar population (purple) and its reprocessed dust emission (green).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_paper_fig1_overview_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_paper_fig1_overview`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP Fig. 1 reproduction: panchromatic AGN + host overview</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Faithful reproduction of Fig. 9 of Buchner et al. (2024, GRAHSP): the AGN spectrum from intrinsic (blue, top) to strongly attenuated (red, bottom) as the AGN-only color excess agn_grahsp_ebv_agn is swept from 0.01 to 1. GRAHSP attenuates the AGN side with an SMC/Prevot (1984) law (paper §2.1.5), which rises steeply into the UV — so the UV/optical continuum is suppressed far more than the near-IR, and the heaviest attenuation eventually bites into the torus too. The intrinsic torus component is overplotted dashed black.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_paper_fig9_agn_attenuation_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_paper_fig9_agn_attenuation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP Fig. 9 reproduction: attenuation of the AGN model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="GRAHSP ships two torus prescriptions. The default is an empirical log-Gaussian cool+hot dust continuum (``activategtorus``). The alternative is the Mor &amp; Netzer 2012 template torus (``activatetorus``), which interpolates between mean / 25th / 75th-percentile observed AGN mid-IR SEDs via agn_grahsp_tor_temp and applies a short-wavelength Gaussian cutoff at agn_grahsp_tor_cutoff_um.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_grahsp_torus_modes_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_grahsp_torus_modes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">GRAHSP torus: empirical log-Gaussian vs Mor & Netzer 2012 templates</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Kubota &amp; Done (2018) three-zone accretion disc model shows how the big blue bump (BBB) peaks at different wavelengths depending on black-hole mass and Eddington ratio. Sweeping across the accretion-state plane from low-luminosity advection-dominated (ADAF-like) to high-Eddington thin-disc reveals the transition: high mass + low Eddington gives cool outer discs peaking in the NIR; low mass + high Eddington gives hot inner zones peaking in the FUV/UV.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_kd18_disc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_kd18_disc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kubota & Done 2018 disc: Accretion state effects on continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Nenkova et al. (2008) CLUMPY library is the AGN dusty-torus model used by FSPS and Prospector. tengri ships the same templates (vendored from FSPS as data/nenkova08_torus_grid.h5) and interpolates them with a pure-JAX triweight kernel, so the equatorial optical depth agn_tau is a fully differentiable, fitted parameter — it can be sampled by NUTS, optimized by MAP, or marginalized by VI, just like in Prospector.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_nenkova_tau_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_nenkova_tau_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">CLUMPY torus (Nenkova+2008): optical depth as a fitted parameter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Identical AGN configuration (multicolor disc + SKIRTOR torus at log L_bol = 12.5), one with the narrow-line region (FWHM ~ a few hundred km/s, characteristic Type-2 spectrum) and the other with the broad-line region (FWHM ~ thousands of km/s, Type-1). Side-by-side zooms on the UV (Ly-alpha, C IV) and the optical (Hbeta, [O III], Hα) make the velocity-width contrast unmistakable while controlling for continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_nlr_blr_lines_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_nlr_blr_lines`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Narrow vs broad line region: a velocity-width contrast in two windows</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Polar dust disc attenuation applies only to Type 1 (face-on) sightlines — the equatorial torus already screens the disc for Type 2. The bi-conical polar dust absorbs disc photons regardless of viewing angle, however, and re-emits them isotropically as a FIR graybody (Casey 2012). So both Type 1 and Type 2 sweeps show the FIR re-emission bump growing with E(B-V); only the UV/optical attenuation is gated by sightline.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_polar_dust_ebv_type12_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_polar_dust_ebv_type12_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polar dust E(B-V) reddens Type 1 & 2 AGN differently</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In a relativistic accretion-disc model the inner boundary sits at the innermost stable circular orbit (ISCO). Higher spin shrinks the ISCO, raises the inner-disc temperature, and shifts disc power blueward — the UV spectral slope alpha (L_nu ~ nu^alpha across 912 to 3000 Å) hardens monotonically with spin. We sweep a_spin from 0 to 0.998 on the Kubota &amp; Done (2018) disc backbone, the public-API entry point for spin-sensitive disc physics in tengri, and report alpha alongside the SEDs.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_relagn_spin_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_relagn_spin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Black-hole spin hardens the UV slope through ISCO migration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three accretion-disc backbones at the same bolometric anchor (log L_bol / L_sun = 12.5): the Richards et al. 2006 empirical mean Type-1 SDSS quasar template, the Temple, Hewett &amp; Banerji 2021 empirical QSOgen, and the Shakura-Sunyaev multicolor disc (the outer-disc component of Kubota &amp; Done 2018). Each is normalized to the same bolometric output so the differences are entirely in spectral shape — Richards+2006 is broader than QSOgen and carries the infrared bump from its host-galaxy-corrected composite, while the multicolor disc cuts off sharply on either side of the big blue bump.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_richards2006_template_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_richards2006_template`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Richards+2006 empirical Type-1 quasar template alongside physical discs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three distinct AGN types overlaid to show how AGN morphology and obscuration evolve with luminosity:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_seyfert_quasar_blazar_archetypes_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_seyfert_quasar_blazar_archetypes`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN archetypes: Seyfert, quasar, and LIRG/Sy across bolometric luminosity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Silva, Maiolino &amp; Granato (2004) AGN torus templates are empirical reprocessed-dust SEDs binned by line-of-sight hydrogen column density agn_log_nh_silva. As the column rises from unobscured (Type-1-like, N_\mathrm{H} \sim 10^{22}\,\mathrm{cm^{-2}}) to Compton-thick (N_\mathrm{H} \sim 10^{25}\,\mathrm{cm^{-2}}):">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_silva04_nh_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_silva04_nh_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Silva+04 torus: Obscuration and the 9.7 μm silicate feature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="SKIRTOR (Stalevski et al. 2016) is a clumpy radiative transfer torus model with a three-dimensional parameter space (half-opening angle, inclination, optical depth). Two different implementations exist in tengri:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_agnfitter_vs_cigale_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_agnfitter_vs_cigale`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: AGNfitter-averaged vs. X-CIGALE full grid</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrate how the SKIRTOR clumpy radiative-transfer torus (Stalevski+ 2012, 2016) reprocesses the hot accretion disc and dust as a function of viewing angle.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_inclination_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_inclination_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR AGN torus: inclination-dependent obscuration and silicate features</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduces the SKIRTOR vs Fritz comparison from Yang et al. 2020 (X-CIGALE Fig. 2). Both libraries re-emit the same disc-absorbed luminosity in the mid-IR; the mid-IR peak amplitude differs by ~0.5 dex because SKIRTOR&#x27;s clumpy 3-D Stalevski+2016 RT redistributes heating more efficiently into the bright NIR-MIR continuum than a smooth-density torus. tengri does not ship Fritz+2006 directly; we substitute Silva+04 (template-based smooth torus, the closest contemporary analog) — the qualitative contrast (clumpy bright MIR vs smooth fainter MIR) is preserved.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_vs_smooth_torus_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_vs_smooth_torus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR clumpy vs Silva+04 smooth-torus comparison (X-CIGALE Fig. 2)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="This is tengri&#x27;s full-grid SKIRTOR torus (Stalevski+2012, 2016), following the X-CIGALE skirtor2016 conventions: a 5-D clumpy two-phase library indexed by equatorial optical depth tau, radial and polar density gradients p / q, half-opening angle oa, and inclination cos i (plus an optional Casey-2012 polar-dust graybody). It is the science-grade counterpart to the parameter-averaged skirtor_agnfitter library — and, having the full grid, it responds strongly to its parameters.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_xcigale_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_xcigale_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus (full X-CIGALE grid): optical depth and inclination</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Slone &amp; Netzer (2012) accretion-disc library (SN12, as packaged by AGNfitter-rX) tabulates the big-blue-bump continuum over black-hole mass and Eddington ratio. The disc&#x27;s characteristic temperature scales as T_\mathrm{max} \propto (\dot m / M_\mathrm{BH})^{1/4}, so the spectral peak walks across the UV/optical as those two knobs change:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_slone_netzer_disc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_slone_netzer_disc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Slone & Netzer 2012 disc: Black-hole mass and Eddington ratio</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The growth of a supermassive black hole (SMBH) traces a path through the (M_BH, L_bol) plane. Starting as a dormant low-mass hole, accretion during mergers builds both mass and luminosity. Peak luminosity occurs as a luminous QSO before accretion slows and the system fades. This example traces four key evolutionary stages and plots both the track on the (M_BH, L_bol) diagram and the corresponding SEDs.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_smbh_growth_track_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_smbh_growth_track`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN SMBH growth track: dormant → merger → QSO → fading</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Different radiative-transfer and empirical torus libraries encode the Type-1↔Type-2 unified-model transition differently. SKIRTOR uses a 3D clumpy model with a smooth density distribution and produces symmetric silicate absorption/emission features. CAT3D-WIND employs a wind-like clumpy geometry. Nenkova et al. (CLUMPY) offers a simpler analytical approach. This grid shows how each library&#x27;s silicate 9.7 μm feature and overall IR reprocessing vary with inclination at fixed L_bol and (where applicable) opening angle, revealing library-specific anisotropies and feature depths.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_torus_library_inclination_grid_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_torus_library_inclination_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN torus libraries across viewing angle: silicate feature and geometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The SKIRTOR clumpy torus geometry is controlled by the half-opening angle (``agn_oa_skirtor``), which determines how much of the accretion disc the dusty material covers. Smaller opening angles (more pole-on geometry, ~20–30°) produce a compact torus that exposes the hot inner disc; larger angles (more flared, ~50–60°) create a covering geometry that obscures the disc and reprocess more UV/optical photons into the mid-infrared.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_torus_opening_angle_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_torus_opening_angle_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus opening angle: geometry controls IR silicate and FIR bump</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The composable AGN runner sums disc + broad/narrow lines + FeII + torus, but a real dusty torus also obscures the central engine along edge-on sightlines while its own infrared emission is not re-extinguished by that same screen. tengri applies this inclination-dependent torus screen automatically whenever the torus is one of the two CIGALE production grids (``skirtor`` or fritz).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_torus_screen_disc_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_torus_screen_disc`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN unification: the torus screens the disc with inclination</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Unified AGN models explain the Type 1/Type 2 dichotomy as a purely geometric effect — the same accretion disc + dusty torus system appears as:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_type1_type2_unified_model_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_type1_type2_unified_model`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Type 1 vs Type 2 AGN: Unified viewing-angle classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sanders et al. (1988) proposed that Ultra-Luminous Infrared Galaxies (ULIRGs) are the dust-shrouded precursors to optical QSOs. This sequence traces progressive unveiling of a buried AGN through five stages:">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_ulirg_to_qso_transition_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_ulirg_to_qso_transition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">ULIRG→QSO evolutionary sequence: dust-obscured starburst to bare quasar</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Radio
=====

Star formation (free-free and synchrotron) and AGN (radio-loud) components. Far-infrared–radio correlation and non-thermal spectral slopes.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The synchrotron spectral index α_sf controls how steeply the radio spectrum falls with frequency. Star-forming galaxies typically have α_sf ≈ 0.7–0.8. Flat spectra (α ≈ 0) signal strong free-free contribution; steep spectra (α &gt; 1) indicate cosmic-ray electron aging. We vary α_sf ∈ [0.3, 1.2] at fixed L_IR = 10^11 L_sun and show normalized spectra (reference 1.4 GHz).">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_alpha_sf_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_alpha_sf_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Synchrotron spectral index: steeper α_sf dims the high-frequency tail</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The dimensionless parameter q_IR characterizes the FIR-radio correlation, linking far-infrared luminosity to 1.4 GHz synchrotron emission. Higher q_IR means relatively weaker radio per unit star formation. We vary q_IR across the observationally motivated range 2.0–3.3 at fixed L_IR = 10^11 L_sun, demonstrating how radio loudness evolves (Bell 2003).">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_q_ir_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_q_ir_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FIR-radio correlation: q_IR sets radio loudness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A star-forming galaxy&#x27;s GHz continuum is set by two components: non-thermal synchrotron from supernova remnants (steep, L_ν ∝ ν^{-α_sf}) and thermal free-free from H II regions (flat, L_ν ∝ ν^{-0.1}). Their ratio at fixed frequency depends sensitively on the synchrotron spectral index α_sf — flatter spectra leave more of the GHz luminosity to free-free, steeper spectra are synchrotron-dominated until the (sub-mm) crossover.">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_crossover_frequency_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_crossover_frequency`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Synchrotron / free-free balance vs synchrotron slope α_sf</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The FIR-radio correlation links far-infrared luminosity (dust-reprocessed star-formation energy) to 1.4 GHz synchrotron emission. The dimensionless q_IR parameter relates the two via L_IR ∝ L_1.4GHz^(10^q_IR/2.5). Brighter starbursts emit stronger radio across all frequencies. We sweep L_IR over 10^10–10^13 L_sun at fixed q_IR = 2.64 (canonical; Bell 2003).">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_lir_relation_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_lir_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FIR-radio correlation: L_IR × q_IR sets radio loudness scale</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Radio loudness R = log₁₀(L_5GHz / L_B) quantifies the ratio of AGN radio to optical luminosity. Radio-quiet AGN have R ≲ 1; radio-loud sources (FR I/II, blazars) reach R ∼ 3–5. Each decade in R corresponds to an order of magnitude increase in jet radio luminosity at fixed bolometric AGN power. We sweep R ∈ [0, 4] at fixed L_bol = 10^44 erg/s (Seyfert-1-like) and α_agn = 0.7.">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_loudness_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_loudness_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN radio loudness R: orders of magnitude in jet power</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed host (constant SFR = 3 M☉/yr, Condon-92 synchrotron + free-free) we sweep the composable AGN&#x27;s bolometric luminosity agn_log_lbol from 9 to 13 (in log L_sun). The host alone produces a power-law GHz continuum; the AGN superposes a flatter-spectrum jet component that takes over above log L_bol ≳ 11.5 — the classic radio-loud / radio-quiet division emerges from this competition.">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_vs_agn_lbol_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_vs_agn_lbol`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radio SED response to AGN bolometric luminosity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Decompose a star-forming galaxy&#x27;s radio SED into its physical components: synchrotron (steep, slope ~ -0.8) from supernova remnants and thermal free-free (flat, slope ~ -0.1) from HII regions. At radio frequencies, synchrotron dominates below ~30 GHz, while free-free becomes progressively important above. This example uses the Condon (1992) framework with Murphy+2011 thermal calibration.">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_synchrotron_thermal_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_synchrotron_thermal_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radio SED decomposition: synchrotron vs thermal free-free</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

X-ray Emission
==============

X-ray binaries (HMXB, LMXB) scaled with SFR and stellar mass. AGN coronae: luminosity, photon index γ, exponential cutoff E_cut, UV-to-X-ray slope α_ox.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The X-ray power-law spectrum steepens above an exponential cutoff E_cut. Compact coronae with low optical depth have low E_cut (~100 keV); thick, optically-deep coronae extend to higher E_cut (~1 TeV). Variation of E_cut at fixed γ=1.8 and α_ox=−1.4 shows how the hard X-ray tail responds to changes in coronal geometry or magnetic field.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_E_cut_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_E_cut_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN X-ray hard-tail rollover: exponential cutoff E_cut governs high-energy turnover</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The CIGALE-faithful corona derives the X-ray normalization from L_2500 via the empirical alpha_OX-L_2500 correlation. tengri ships three published parametrizations:">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_alpha_ox_relations_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_alpha_ox_relations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Three empirical alpha_OX-L_2500 prescriptions diverge at the quasar peak</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The UV-to-X-ray spectral slope alpha_OX (defined as log F_X minus log F_UV divided by log nu_X minus log nu_UV) separates X-ray-loud quasars (alpha_OX around -1.2, strong X-ray relative to the UV continuum) from X-ray-quiet systems (alpha_OX around -1.8, suppressed X-ray). The CIGALE-faithful corona derives alpha_OX from L_2500 via the Just+2007 relation by default; here we sweep delta_alpha_ox to apply offsets from -0.4 to +0.4 around that empirical value, at fixed L_2500 (= L_bol = 1e45 erg/s through the standard Hopkins+2007 bolometric correction). More positive delta brightens the corona; more negative suppresses it.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_alpha_ox_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_alpha_ox_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN UV-to-X-ray power-law slope alpha_OX controls X-ray normalization</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="X-ray absorption in AGN undergoes a qualitative shift at N_H ≈ 1e24 cm⁻², where the cross-section for Compton scattering becomes comparable to photoelectric absorption. Below this threshold, soft photons (E &lt; 10 keV) are suppressed by the Thompson cross-section σ_T ≈ 0.66 Barn, creating a steep spectral curvature in the soft band. Above it, the entire 2–10 keV continuum is suppressed equally, flattening the spectrum and leaving only a scattered component (~1% of the intrinsic flux) observable.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_compton_thick_vs_thin_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_compton_thick_vs_thin`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photoelectric vs. Compton-thick regimes: the N_H = 1e24 cm−2 transition</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The AGN X-ray corona produces a cut-off power-law (photon index Gamma roughly 1.8, E_cut around 300 keV) normalized through the alpha_OX-L_2500 relation (Lusso &amp; Risaliti 2016). At fixed Gamma and alpha_OX, increasing bolometric luminosity shifts the whole spectrum upward but leaves the spectral shape nearly intact — the sub-linear alpha_OX relation only steepens the shape at the top of the quasar regime.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_agn_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_agn`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN corona: bolometric luminosity sets normalization, not shape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The CIGALE-faithful obscured-AGN spectral model combines two knobs that classification surveys often confound: delta_alpha_ox (offset from the empirical alpha_OX-L_2500 relation, controlling the intrinsic X-ray-to-UV ratio) and log N_H (line-of-sight column density, suppressing soft-band flux through zphabs × cabs). We compute the hardness ratio HR = (H - S) / (H + S) with S = 0.5-2 keV and H = 2–10 keV across the joint (delta_alpha_ox, log N_H) plane on a fixed L_2500 anchor (= L_bol = 1e45 erg/s through the Hopkins+2007 bolometric correction).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_alpha_ox_nh_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_alpha_ox_nh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hardness ratio across the alpha_OX vs log N_H plane</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The X-CIGALE X-ray module (Yang et al. 2020) sums four physically distinct emitters: the AGN corona (a cut-off power law normalized through the α_OX–L_2500 relation), low- and high-mass X-ray binaries (LMXB ∝ M⋆, HMXB ∝ SFR; Lehmer et al. 2016 metallicity/age scalings), and a hot interstellar-gas term (∝ SFR). This reproduces Yang+2020 Figure 1 for a typical AGN host: L_2–10 keV = 10⁴³ erg s⁻¹, M⋆ = 10¹¹ M⊙, SFR = 10 M⊙ yr⁻¹, T = 1 Gyr, Z = 0.02.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_component_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_component_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray SED decomposition: AGN, LMXB, HMXB, hot gas</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The X-ray photon index γ controls how rapidly the AGN corona&#x27;s power-law spectrum falls off above a few keV. Flat spectra (low γ ~1.4) extend more photons to high energies; steep spectra (high γ ~2.4) drop quickly. We vary γ across its typical observational range at fixed bolometric luminosity.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_gamma_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_gamma_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN X-ray spectral hardness: photon index γ controls power-law steepness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The line-of-sight column density N_H reshapes the AGN X-ray spectrum in two regimes: photoelectric absorption (``zphabs``) suppresses the soft band roughly as \exp(-\sigma(E)\,N_H) with cross-section \sigma \propto E^{-3}, while Compton down-scattering (``cabs``) adds an energy-independent suppression \exp(-\sigma_T\,N_H) that becomes dominant once log N_H ≳ 24 (the Compton-thick boundary). A constant warm-electron scattered fraction (~1 % of the intrinsic continuum) is added back, which is the only flux observable in the soft band for nearly opaque columns and explains why Compton-thick AGN are still marginally detectable in soft-band stacks (Matsumoto et al. 2026 Fig. 11/12).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_nh_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_nh_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">N_H column density sweep: from unobscured to Compton-thick</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In Compton-thick AGN (log N_H ≳ 24 cm⁻²), the line-of-sight obscurer extinguishes the primary AGN corona below ~ 10 keV. What&#x27;s left is the reflected component — the fraction of corona photons that hit the cold accretion disc, Compton-scatter off bound electrons, and emerge along the line of sight without being photoelectrically absorbed. The resulting spectrum peaks around 30 keV (the famous Compton hump) and is the smoking-gun signature that NuSTAR / Swift-BAT surveys use to confirm buried supermassive black holes (Ricci+2017, Matsumoto+26).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_pexrav_compton_hump_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_pexrav_compton_hump`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Compton hump in obscured AGN: pexrav reflection across log N_H</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="X-ray binaries (XRBs) are the dominant X-ray sources in star-forming galaxies once an AGN is excluded. High-mass XRBs trace the recent star-formation rate (Mineo+2012), while low-mass XRBs trace the integrated stellar mass (Lehmer+2019). The two scalings have different spectral shapes too: HMXBs are slightly harder, LMXBs slightly softer. Two side-by-side sweeps — SFR (left) at fixed M_star = 1e11 M☉, and M_star (right) at fixed SFR = 10 M☉/yr — separate the two channels on the same axes.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_sf_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_sf`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray binary luminosity scales with SFR (HMXB) and stellar mass (LMXB)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed host (constant SFR = 3 M☉/yr, Mineo+12 HMXB contribution) we sweep the composable AGN&#x27;s bolometric luminosity agn_log_lbol from 9 to 13 (in log L_sun). The host XRB component is a flat power-law below ~10 keV; the AGN corona contributes a much harder power-law that dominates above log L_bol ≳ 11.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_vs_agn_lbol_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_vs_agn_lbol`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray SED response to AGN bolometric luminosity</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

IGM
===

Intergalactic-medium absorption: Madau vs Inoue prescriptions, Lyα forest, damped Lyα systems. Lyman-break/dropout signature in high-z photometric selection. IGM `igm_transmission(wave_obs, z)` takes observed-frame wavelengths (not rest-frame).


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Damped Lyman-alpha (DLA) systems imprint strong absorption features blueward of the Lyman-alpha line (1216 Å rest-frame). We sweep column density log(N_H) ∈ {19.0, 19.5, 20.0, 20.3, 20.8} cm^{-2} at fixed redshift z=3, showing how higher column density systems deepen the Lyman forest and suppress flux in the UV-to-optical SED.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_dla_absorption_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_dla_absorption`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DLA column density sculpts the Lyman alpha forest at z=3</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Damped Lyman-alpha (DLA) systems imprint deep absorption troughs across the UV-to-optical range, with the strength and profile shape depending sensitively on the absorber&#x27;s redshift. We hold column density at the classic DLA threshold log(N_H) = 20.3 cm⁻² and sweep the absorber redshift over z ∈ {1, 2, 3, 4, 5, 6}, showing how the damping wing pattern shifts to longer observed wavelengths and the Lyman-alpha forest structure evolves. This complements the fixed-z, variable-N_H absorption pattern by isolating the redshift dependence.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_dla_redshift_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_dla_redshift_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DLA damping wing evolves with absorber redshift at fixed column density</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four IGM transmission variants available in tengri are compared at z=7, applied to a young star-forming SED. This diagnostic isolates the differences between models around the Lyman-alpha forest:">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_igm_models_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_igm_models_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparison of IGM absorption models at high redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The intergalactic medium (IGM) imprints wavelength-dependent opacity on observed galaxy SEDs via Lyman-series and Lyman-continuum absorption. The Lyman break at 912 Å rest-frame shifts to longer observed wavelengths at higher z, enabling photometric redshift estimation via the dropout technique. We vary redshift z ∈ {0.5, 1, 2, 3, 4, 6, 8} across the Inoue et al. (2014) transmission model to show how IGM opacity increases with z.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_igm_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_igm_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">IGM transmission curves evolve sharply with redshift as Lyman forest deepens</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The two standard IGM transmission prescriptions diverge most visibly across the Lyman-alpha forest and the Lyman limit. Madau (1995) is the original analytic Lyman-series effective optical depth; Inoue+2014 added Lyman-continuum and damped-Lyα systems in a more careful integral over the H I distribution.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_inoue_vs_madau_z5_z7_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_inoue_vs_madau_z5_z7`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Inoue+2014 vs Madau 1995 across the Lyman break at z=5 and z=7</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Lyman-alpha (Lyα) emission line at rest-frame 1216 Å is one of the strongest hydrogen recombination features in star-forming galaxies. As the redshift increases from z = 2 to z = 7, the IGM becomes progressively opaque at wavelengths shortward of Lyα (the &quot;blue wing&quot;), due to cumulative Lyman-series absorption from neutral hydrogen in the intergalactic medium.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_lyman_alpha_igm_attenuation_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_lyman_alpha_igm_attenuation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha profile and IGM blue-wing absorption across redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A young Lyman-break galaxy SED is built once at rest frame, then redshifted to a sequence of observed-frame epochs (``z = 1, 3, 5, 7``) with the Inoue et al. 2014 IGM transmission stamped on top. The characteristic spectral signatures move with redshift:">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_sed_with_igm_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_sed_with_igm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Full galaxy SED with IGM absorption applied at multiple redshifts</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Photometry
==========

Broadband filter selection, k-corrections, cosmological dimming. Diagnostic planes: UVJ, NUV–r, WISE/IRAC AGN wedges. Photometric-redshift color degeneracies.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 4000 Å break is a sharp discontinuity in the stellar continuum at the boundary between the Balmer and Paschen series, caused by hydrogen Lyman absorption blanketing in the overlying atmosphere. In the rest frame it sits at 4000 Å for all galaxies. In the observer frame, the break shifts to longer wavelengths with increasing redshift: z × 4000 Å. This is why different photometric bands probe the break at different redshifts — the fundamental principle behind photo-z estimation and dust/age degeneracies.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_balmer_break_redshift_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_balmer_break_redshift_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Balmer break (4000 Å) position in observed-frame filters vs redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How many photometric bands are needed to recover stellar mass accurately? We mock a single galaxy with fixed parameters at different signal-to-noise levels using progressively larger filter sets, then MAP-fit to measure the recovered mass uncertainty. The figure shows that stellar mass constraints improve dramatically with filter count: a 2-band measurement is degenerate (wide posterior), while a 10-band panchromatic set (optical + NIR + mid-IR) tightens the mass estimate by an order of magnitude or more.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_band_count_mass_recovery_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_band_count_mass_recovery`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar mass recovery with increasing photometric band count</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does a galaxy&#x27;s location in color–color space evolve with redshift? We compute SDSS g − r and r − z colors for two galaxy populations — a young star-forming and an old quiescent — across z = 0 to 3, with arrows marking the integer redshift stops. This is the reference picture for photometric redshift classifiers and for stellar-template grids.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_color_tracks_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_color_tracks_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photometric color tracks vs redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does the observed photometric flux of a FIXED-luminosity galaxy decline with redshift? We track a star-forming galaxy (log M* = 10.5, SFR = 10 M☉/yr) across z = 0.1 to 6 in three optical/infrared bands (SDSS r, JWST J, JWST H), visualizing the three physical effects:">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_cosmic_dimming_observed_flux_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_cosmic_dimming_observed_flux`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cosmic dimming and K-correction with redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The same star-forming galaxy SED is intercepted by three different filter sets — SDSS ugriz (optical), 2MASS JHKs (near-infrared), and HST ACS F435W/F606W/F814W (UV-optical). Each panel overlays the survey&#x27;s throughputs on the shared SED so the reader sees, at a glance, which spectral features (the 4000-Å break, Hα + [N II], the 1.6-μm stellar bump) fall inside each band.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_set_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_set_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Filter placement decides which spectral features a survey can see</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Show a typical star-forming galaxy SED at z=1 with observed-frame filter throughputs overlaid as semi-transparent fills from 0.3 to 25 μm. This helps visualize which rest-frame stellar and dust features each photometric system samples across the spectrum.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_throughput_overlay_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_throughput_overlay`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">HST+JWST+LSST+Spitzer Filter Overlay on Star-Forming SED at z=1</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="What each photometric system measures depends on where its filters sit relative to the rest-frame spectral features. We overlay six common filter sets (GALEX NUV, SDSS ugriz, 2MASS JHK, WISE W1/W2/W3, Euclid YJH, JWST NIRCam wide bands) on top of an observed-frame star-forming galaxy SED at z = 0.5.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_galaxy_with_filters_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_galaxy_with_filters`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Galaxy SED with photometric filter coverage</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Subaru HSC and Blanco DECam i-bands have different red-edge cutoffs (HSC i-2 at ~850 nm, DECam i at ~870 nm). This 20 nm difference produces measurable color offsets when a sharp spectral feature sweeps through the i-band — particularly the Lyman break at z~3.5–4.5. We show (r − i) colors for an LBG template across both filter sets to highlight the divergence in the high-redshift regime.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_hsc_vs_des_color_high_z_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_hsc_vs_des_color_high_z`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">HSC vs DES filter i-band differences at high redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do K-corrections vary with redshift for different galaxy populations? K-corrections quantify the shift in filter response as galaxies move to higher redshifts: K(z) = −2.5 log₁₀[(1+z) × F_ν(z) / F_ν(0)] for a fixed rest-frame filter. We compute K(z) for the SDSS r-band across four galaxy types — young star-forming, old star-forming, red-sequence elliptical, and post-starburst — from z = 0.01 to z = 2.0. This illustrates why stellar mass measurements require careful K-corrections at high redshift and why color-matched template sets dominate photometric redshift algorithms.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_k_correction_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_k_correction_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">K-corrections as a function of redshift for different SED types</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The NUV−r color is a sensitive probe of stellar age in galaxies. We show how a single-burst star formation history (tsnorm, truncated-skew-normal) evolves across the GALEX green valley (NUV−r ≈ 4–5 mag) as the stellar population ages from 0.05 to 5.5 Gyr. The color exhibits a sharp discontinuity as the stellar population cools through the transition between young, UV-bright stars and older, redder populations.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_nuv_r_age_track_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_nuv_r_age_track`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NUV−r color vs stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A higher-redshift counterpart to the SDSS quickstart fit. We mock JWST NIRCam wide-band photometry of a star-forming galaxy at z=3 (S/N=15), run a MAP fit, and show the recovered SED + per-band residuals. NIRCam samples the rest-frame UV-optical at this redshift, so the SFH and dust attenuation are the dominant levers.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_photometric_fit_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_photometric_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recovering a z=3 galaxy from JWST NIRCam photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two galaxies with very different star formation histories and dust can collide in color–color space, making photo-z ambiguous. Here, a young dusty star-forming galaxy at z≈0.5 and an old quiescent galaxy at z≈2 follow nearly identical (u-g, g-r) tracks and intersect at a single point. This shows why intermediate-wavelength photometry is essential for robust photo-z classification.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_photoz_color_degeneracy_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_photoz_color_degeneracy_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photo-z degeneracy in color–color space: low-z dusty vs high-z quiescent</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Galaxy color–magnitude diagram showing the distinct red and blue populations. We model two populations — 25 quiescent old galaxies (peak SFH ~8 Gyr) and 25 star-forming galaxies (continuous SFR) — varying stellar mass via log_total_mass. Each sample is placed at z = 0.05, computing u − r color and rest-frame M_r magnitude. The color bimodality and green valley are key signatures of galaxy assembly across cosmic time (Strateva et al. 2001 SDSS, Baldry et al. 2004).">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_red_sequence_blue_cloud_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_red_sequence_blue_cloud`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Red Sequence vs Blue Cloud Bimodality</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Same rest-frame star-forming SED at four observed redshifts, with the SDSS ugriz throughputs plotted in their observed position so the reader sees which rest-frame features each band samples. The Balmer break enters the u band by z=1; by z=2 the bands fall longward of the 4000-A break entirely. This is the geometric source of the k-correction&#x27;s sign.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_redshift_filter_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_redshift_filter_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SDSS ugriz sweep through a galaxy SED as z grows from 0.1 to 2</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Same star-forming galaxy, same SDSS ugriz set, three signal-to-noise levels (5, 20, 100). For each S/N we mock the photometry, run a MAP fit, and overlay the recovered SED on the truth. The figure surfaces the expected scaling — posterior offset and band-by-band residuals shrink as 1/S/N — and makes the inference cost concrete: even at S/N=5 the dust amplitude is degenerate enough that a single MAP run misses it by ~0.3 mag in the u band.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_snr_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_snr_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Posterior width tracks 1/S/N for fixed-truth SDSS photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Color-color diagram in Spitzer IRAC bands (3.6, 4.5, 5.8, 8.0 μm) showing the Lacy+2007 / Donley+2012 AGN selection wedge. Population of 50 star-forming galaxies (z=0–2) are plotted as blue cloud; 10 AGN with varying bolometric luminosity cluster inside the wedge (red region) demonstrating the diagnostic power of mid-infrared colors for AGN identification.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_spitzer_irac_agn_wedge_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_spitzer_irac_agn_wedge`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Spitzer IRAC AGN Wedge Diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The UVJ (U−V vs V−J) diagram is a classic method for separating star-forming from quiescent galaxies. We populate it with four model tracks: (1) constant star-forming galaxies with varying dust optical depth, (2) an old quiescent population, (3) a post-starburst galaxy, and (4) a dusty starburst. The gray box marks the &quot;quiescent region&quot; from Williams+2009, a visual guide for identifying passive galaxies.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The UVJ color–color diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WISE color-color diagram (Stern et al. 2012) is a tool for separating AGN from star-forming galaxies using mid-infrared colors. The diagnostic exploits the fact that AGN emit power-law SEDs (flat in νLν) while star-forming galaxies have cooler dust emission (Rayleigh-Jeans slope at long wavelengths).">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_wise_agn_color_color_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_wise_agn_color_color`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">WISE W1–W2 vs W2–W3 Color-Color Diagram with Stern+2012 AGN Wedge</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Spectroscopy
============

Absorption-line indices (D4000, Hδ) from stellar age and metallicity. Velocity dispersion and line broadening. Instrumental resolution effects (prism vs grating). High-redshift examples: JWST/NIRSpec out to z ≈ 6 Lyα emitters.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar absorption features, especially the Mg b and Fe5270 line strengths, encode both age and metallicity in a classical anti-correlation pattern: at fixed metallicity, both features strengthen with age (population becomes older, cooler); at fixed age, increasing metallicity also strengthens the features (enhanced α-element abundances + stronger metal absorption).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_bandheads_age_metallicity_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_bandheads_age_metallicity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar absorption bandheads: age and metallicity anti-correlation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Population diagnostics: single-burst SSP populations (3 SFH shapes × 5 ages × 3 metallicities = 45 points) colored by SFH shape and marked by metallicity. The Hδ_A vs D_n(4000) diagram discriminates starburst (high Hδ_A, low D_n(4000)) from quiescent (low Hδ_A, high D_n(4000)) populations and is sensitive to recent star formation and metal enrichment.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_d4000_hdelta_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_d4000_hdelta_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kauffmann+2003 D_n(4000) vs Hδ_A Diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="High-redshift star-forming galaxy with strong rest-frame UV and optical emission lines redshifted into the JWST NIRSpec G395M window (2.9–5.1 μm). Lines include Lyα, CIV, HeII, CIII], [OII], Hβ, [OIII], and Hα, each annotated with rest wavelength and vacuum-frame position.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_jwst_nirspec_high_z_spectrum_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_jwst_nirspec_high_z_spectrum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mock JWST NIRSpec G395M spectrum of a z=7 star-forming galaxy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="High-redshift Lyα emitter at z=6 with young age (~10 Myr), low metallicity (Z~0.1 Z☉), and minimal dust. The observed-frame spectrum (7000–13000 Å) reveals the redshifted Lyα emission line at 8512 Å, the Lyman break at 6384 Å, characteristic IGM blue-wing absorption, and the rest-UV continuum. Demonstrates Lyα radiative transfer and reionization-era observability.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_lae_spectrum_z6_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_lae_spectrum_z6`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha emitter spectrum at z=6: IGM absorption and Lyα escape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A z=5 JADES-like star-forming galaxy observed with JWST NIRSpec in two modes: PRISM (R~100, low-resolution) and G395M grating (R~1000, medium-resolution). The Hα line at rest 6564.61 Å appears as a single blob in PRISM but resolves into three peaks in the grating: Hα + [NII] λλ6549,6585 Å doublet.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_nirspec_prism_vs_grating_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_nirspec_prism_vs_grating`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JWST NIRSpec PRISM vs G395M grating: Hα + [NII] resolution comparison</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Hδ_A peaks ~100 Myr post-quench, then decays as A-type stars die out. This absorption index traces the lifetime of A-type stars responsible for Balmer absorption, a key driver of galaxy quenching (Worthey &amp; Ottaviani 1997).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_post_starburst_diagnostic_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_post_starburst_diagnostic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Post-Starburst K+A Diagnostic: Hδ_A vs Time Since Quench</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spectral resolution R determines whether the Hα + [N II] emission-line complex appears as a single blended feature (low R) or resolves into three distinct lines (high R). Varying R from 100 to 10000 reveals the transition from kinematically degenerate at R~100 (SDSS/DESI-like) to fully resolved at R~5000 (JWST-like).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_resolution_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_resolution_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Instrumental resolution controls Hα + [N II] line blending</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Mg b 5170 Å region of an old stellar population observed at spectral resolution R = 3000, convolved with increasing stellar velocity dispersion σ_v from 50 to 400 km/s. The classic kinematic diagnostic — line core depth tracks σ_v, asymmetric wings appear with rotational broadening (not modeled here, sigma only).">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_sigma_v_absorption_broadening_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_sigma_v_absorption_broadening`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Velocity-dispersion broadening of stellar absorption features</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="D4000, Hδ absorption, and the Mg b feature respond differently to stellar age and metallicity, providing complementary constraints when used together. D4000 rises with age; Hδ peaks at intermediate ages (A-star dominated); Mg b traces metallicity on the RGB/AGB branch.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_spectral_features_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_spectral_features`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Key Spectral Features as Age and Metallicity Probes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three of the most-used optical absorption / emission diagnostics evaluated on a single-burst stellar population from 30 Myr to 13 Gyr, at solar metallicity, no dust. The figure makes obvious which diagnostic responds on which timescale:">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_spectral_indices_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_spectral_indices_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Classic spectral indices vs single-burst age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Rest-frame spectrum with stellar population ages">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_spectrum_fit_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_spectrum_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rest-frame spectrum with stellar population ages</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep stellar velocity dispersion σ_v ∈ {50, 100, 150, 250, 400} km/s to show how line broadening increases with dynamical heating. The Mg b absorption feature (~5170 Å) widens progressively, demonstrating the kinematic signature of higher-velocity stellar populations.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_velocity_dispersion_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_velocity_dispersion_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Velocity Dispersion Sweep</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Narrow-line regions sit at σ_v ≈ 50–300 km/s; a broad Hα component from the AGN accretion disk reaches thousands of km/s. The [NII] doublet is separated by 35.4 Å (6549.86 and 6585.28 Å vacuum), which corresponds to σ_v ≈ 1600 km/s — above that the two lines merge into the wing of Hα.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_velocity_offset_lines_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_velocity_offset_lines`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission-line velocity dispersion: narrow [NII] to broad Hα</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Multiwavelength
===============

Panchromatic SEDs from X-ray to radio.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six physics blocks added cumulatively to the same star-forming host so the contribution of each is visible at every wavelength.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_components_isolated_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_components_isolated`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Each tengri SED component shown in isolation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A low-mass, low-metallicity dwarf irregular (M*~10^8 M☉, Z~0.1 Z☉) with high specific star formation rate. The SED highlights: strong UV continuum from young stars, dominant optical emission lines (Hα 6563 Å, [OIII] 5007 Å, Hβ) on a faint continuum, minimal dust attenuation, and negligible far-infrared. Equivalent width of Hα is extreme (~100s Å). Metal-poor stellar populations and active star formation drive the starburst signature visible from UV through optical.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_dwarf_irregular_sed_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_dwarf_irregular_sed`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Panchromatic SED of a low-metallicity dwarf irregular galaxy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The FIR–radio correlation (van der Kruit 1971; Helou et al. 1985) holds over five decades in galaxy luminosity. Sweeps IR luminosity and radio spectral index to show the tight linear correlation and how the empirical parameter q_IR varies with model calibration.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_fir_radio_correlation_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_fir_radio_correlation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FIR–radio correlation across IR luminosity and spectral shape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A single intrinsic LBG (young dust-poor star-forming galaxy) shown in the observer frame at four redshifts. The Lyman break sweeps redward into the u-, then g- and r-band dropout regimes, the Inoue+2014 IGM transmission removes more and more flux blueward of Lyα, and the apparent magnitude faint-end falls by ~2.5 mag from z = 2 → 8 due to luminosity distance alone.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_lbg_observed_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_lbg_observed_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Observed SED of a Lyman-break galaxy at z = 2, 4, 6, 8</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="M82 (NGC 3034) is a nearby starburst galaxy with intense nuclear star formation (SFR ~ 10 M☉/yr), stellar mass M* ~ 1×10^10 M☉, and moderate-to-high dust opacity (τ_V ~ 2 in the starburst core). The panchromatic SED spans from UV (young stars) through optical (attenuated by dust) to far-infrared (warm dust re-emission at ~50 μm) and radio (free-free continuum from ionized regions and synchrotron from supernovae).">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_m82_starburst_panchromatic_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_m82_starburst_panchromatic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Panchromatic SED: M82 Starburst Analog</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spiral galaxies exhibit radial metallicity gradients: metal-rich centers and metal-poor discs (e.g. NGC 891, Searle 1971). Three common gradient scenarios—steep positive, flat, and inverted depletion—reshape the integrated SED when weighted by disc area.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_metallicity_radial_gradient_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_metallicity_radial_gradient`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radial metallicity gradients and integrated-light SED</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 5–30 μm rest-frame spectrum showcases distinct infrared tracers: dust polycyclic aromatic hydrocarbon (PAH) emission peaks at 6.2, 7.7, 8.6, 11.3, and 12.7 μm in star-forming galaxies, while silicate absorption (9.7 μm Si–O stretch) and AGN heating suppress PAH and introduce continuum growth in AGN-dominated systems. We model three templates: (a) pure starburst (no AGN), (b) pure AGN (no star formation), and (c) composite with AGN fraction = 0.5. the diagnostic power of mid-IR spectroscopy: PAH strength probes star formation rate, while continuum slope and silicate depth reveal AGN heating and dust temperature.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_mid_ir_pah_features_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_mid_ir_pah_features`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mid-IR PAH features in star-forming, AGN, and composite galaxies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Panchromatic SED spanning hard X-rays through centimeter radio of a luminous quasar with radio-loud jets. Combines AGN disc continuum, X-ray corona, and radio components, showing how AGN dominate across 0.1 keV through centimeter wavelengths.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_panchromatic_agn_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_panchromatic_agn`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray to radio SED of a luminous AGN</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Active galactic nuclei dominate UV to infrared SEDs. Sweeps AGN luminosity fraction from pure starburst to pure AGN, showing the transition in SED morphology as the accretion disc continuum increasingly dominates stellar and dust emission.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_panchromatic_agn_fraction_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_panchromatic_agn_fraction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Blending star-forming galaxy and AGN accretion disc continua</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust absorbs UV and optical photons and re-emits at infrared wavelengths. Sweeping diffuse ISM optical depth τ_diff shows how UV absorption transfers energy into the infrared, demonstrating energy conservation between the attenuation and emission components.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_panchromatic_dust_balance_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_panchromatic_dust_balance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV attenuation and infrared re-emission balance dust energy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A heavily obscured starburst archetype (Arp 220-class ULIRG) with high optical depth and extreme far-infrared dominance. The SED shows: stellar intrinsic (suppressed by dust), stellar attenuated, dust re-emission dominating at 60 μm, and radio extension. Demonstrates how dust attenuation redirects all UV/optical photons into the infrared, completely transforming the SED from young, luminous starbursts.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_panchromatic_dusty_starburst_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_panchromatic_dusty_starburst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV-to-radio SED of a dusty starburst ULIRG</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="UV-to-radio SED of a star-forming galaxy">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_panchromatic_galaxy_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_panchromatic_galaxy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV-to-radio SED of a star-forming galaxy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A nearby Milky Way-mass galaxy (M*~5×10^10 M☉, SFR~2 M☉/yr) across the full electromagnetic spectrum from X-ray (10 Å) to radio (10^9 Å).">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_panchromatic_milky_way_analog_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_panchromatic_milky_way_analog`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Panchromatic SED: Milky Way Analog</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Same galaxy rest-frame panchromatic SED (UV through radio) observed at increasing redshifts. Cosmological redshift transforms rest-frame wavelengths and dims luminosity, shifting spectral features to infrared bands at high redshift where ground-based surveys probe star formation epochs.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_panchromatic_redshift_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_panchromatic_redshift_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rest-to-observer-frame transformation of panchromatic SEDs with redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust-obscured starburst at z = 3 with a heavily attenuated diffuse ISM (`tau_diff = 3.0`). The negative K-correction is what makes submillimeter selection nearly distance-independent over z ≈ 1–6: as a source recedes, the observing band walks up the steep Rayleigh-Jeans side of the dust peak, and the two effects very nearly cancel (Blain+2002).">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_submillimeter_galaxy_sed_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_submillimeter_galaxy_sed`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Submillimeter galaxy SED: dust-obscured starburst at z=3</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="An archetype obscured starburst built up from the tengri component toolkit: a 100 Myr ongoing burst with τ_diff = 2 mag of diffuse dust and a 1 mag birth-cloud opacity. The Dale+2014 IR template re-emits the absorbed UV/optical power into a 60 μm peak; the Condon-92 radio extends to the GHz with the right FIR–radio ratio; the SDSS r-band sits ~3 magnitudes below the FIR peak because almost all the UV/optical has been reprocessed.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_ulirg_arp220_analog_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_ulirg_arp220_analog`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Arp 220 analog: panchromatic SED of a heavily obscured ULIRG</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Inference Methods
=================

Method selection by dimensionality: `mcmc_nuts` for D ≤ 6, `mcmc_hmc` for D ~ 7–8, `mcmc_raytrace`/`vi` for D >~ 20, `laplace` for cheap intervals from MAP Hessian. `vi` and `native_vi_*` are not posterior-equivalent; both native backends are tier=broken and must never be taught in an example. Convergence diagnostics: split-R-hat, ESS, prior-vs-posterior comparisons, corner plots, posterior-predictive checks.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reference: Conroy+2013.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_convergence_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_convergence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAP fit recovery: star-formation history from mock photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Galaxy photometry is degenerate in redshift and stellar mass — the same galaxy can look identical at different redshifts if the mass is adjusted.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_photoz_chi2_grid_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_photoz_chi2_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photo-z degeneracy: chi² landscape over redshift and stellar mass</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The prior predictive envelope — which combinations of parameters the model can produce under the chosen priors, without data — reveals silent pathologies. Coverage: does the envelope contain the data? If not, the posterior will shift to prior boundaries. Width: narrow bands indicate parameters already constrained by the prior alone; data cannot improve estimates there.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_prior_predictive_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_prior_predictive`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Prior predictive check: what does the model predict before it sees data?</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reference: Conroy+2013.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_sfh_recovery_test_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_sfh_recovery_test`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SFH recovery with MAP: double power-law against mock photometry</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Use Cases
=========

Paper-style diagnostics: UVJ, JWST color-color, SFR indicators, mass completeness, age–dust degeneracy, emission-line correlations.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Left panel: The age–dust degeneracy as seen in optical g−r color. A 5 Gyr stellar population with no dust is nearly indistinguishable from a 1 Gyr population reddened by τ_diff = 0.4 when observed in optical broadband colors alone. A 2-D grid in (age, τ_diff) with iso-color contours reveals the orientation of the degeneracy—lines of constant color show why optical colors alone cannot break this ambiguity.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_2d_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_2d`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-dust degeneracy: optical colors vs. UV constraining power</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Optical photometry alone cannot uniquely break the degeneracy between stellar age, dust attenuation, and redshift — a fundamental limitation in photo-z and SED fitting. Three physically distinct galaxy populations can produce nearly identical SDSS ugriz photometry:">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_redshift_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_redshift_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The age-dust-redshift degeneracy in photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Balmer decrement measures dust attenuation via hydrogen recombination line ratios: Hα / H-beta is sensitive to extinction (Calzetti et al. 2000). Without dust, the intrinsic ratio is ~2.78–2.86 (Case B). Here we sweep dust optical depth (τ_diff ∈ [0, 2]) and measure how the predicted Hα and H-beta change. We derive A_V = 1.086 × τ_diff and compare against the Calzetti+2000 expectation.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_balmer_decrement_av_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_balmer_decrement_av`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Balmer Decrement Tests Dust Attenuation on Emission Lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Planck 2018: H₀ = 67.4 km/s/Mpc, Ω_M = 0.315. Riess et al. 2022: H₀ = 73.04 ± 1.04 km/s/Mpc, Ω_M = 0.30. Apparent magnitude shift is ~0.15 mag across z = 0.05–3.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_cosmology_distance_modulus_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_cosmology_distance_modulus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hubble Tension: Cosmology-dependent distance modulus</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="tengri.cosmology exposes the standard FRW distances — comoving, luminosity, angular-diameter — and lookback time as pure-JAX functions over a Planck-18 default. They are differentiable, JIT-able, and interchangeable with astropy&#x27;s API for tengri&#x27;s own forward model.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_cosmology_ladder_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_cosmology_ladder`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cosmological distance ladder and the K-correction for a flat SED</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Kauffmann+2003 separation of star-forming and quiescent SDSS galaxies plotted as a sample track: stellar-burst age varied from 30 Myr to 11 Gyr (single-burst SSP), with each model giving a (``D_n(4000)``, sSFR) pair.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_d4000_vs_ssfr_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_d4000_vs_ssfr`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">D_n(4000) – specific SFR: the Kauffmann+2003 sequence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Steidel+1996 U-dropout box is calibrated for a specific filter set and does not transfer to arbitrary filters: (U − G) &gt; 1.0, (G − R) &lt; 1.5, (U − G) &gt; 1.5(G − R) + 0.3. True z~3 galaxies cluster inside; lower-redshift galaxies fall outside.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_dropout_selection_z3_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_dropout_selection_z3`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">z~3 Lyman-break galaxy U-dropout selection: color-color diagnosis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Mannucci+2010 fundamental metallicity relation (FMR) describes how a galaxy&#x27;s gas-phase metallicity (Z) depends not only on its stellar mass (M) but also on its star formation rate (SFR). This three-parameter relation is a schematic* demonstration of the physical interplay between assembly, star formation, and chemical enrichment.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_fundamental_metallicity_relation_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_fundamental_metallicity_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Fundamental Metallicity Relation: M*-Z-SFR three-body interaction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Coleman, Wu &amp; Weedman 1980 spectral templates remain the textbook illustration of how the integrated SED morphs along the Hubble sequence — from quiescent ellipticals with deep 4000 Å breaks to gas-rich irregulars dominated by ongoing star formation and nebular emission.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_hubble_sequence_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_hubble_sequence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">A morphological atlas: E, Sa, Sb, Sc, Im galaxy SEDs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="JWST NIRCam color-color diagnostics for high-z galaxy classification">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_jwst_color_color_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_jwst_color_color`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JWST NIRCam color-color diagnostics for high-z galaxy classification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Kennicutt+1998 baseline calibrations under constant-SFR assumption: L_FUV(1500 Å): SFR/L_FUV = 1.4 × 10⁻²⁸; L_Hα: SFR/L_Hα = 7.9 × 10⁻⁴²; L_IR(8–1000 μm): SFR/L_IR = 4.5 × 10⁻⁴⁴. Stochastic SFH introduces variance in each indicator; Hα most sensitive to recent star formation.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_kennicutt_sfr_calibrations_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_kennicutt_sfr_calibrations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kennicutt+1998 SFR calibrations: baseline + stochastic variance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates how strong gravitational lensing elevates intrinsically-faint high-redshift (z=7) galaxies above the JWST NIRCam 5σ detection threshold.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_lensed_galaxy_magnification_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_lensed_galaxy_magnification`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Strong-lensing magnification: EoR galaxy SEDs at μ = 1, 5, 20, 100</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The star-forming main sequence (MS) defines a tight relation between stellar mass (M*) and star formation rate (SFR) for actively forming galaxies. This example demonstrates how the MS shifts upward by ~0.7 dex from z=0 to z=2, reflecting the Universe&#x27;s peak epoch of star formation. The left panel shows recovery of the z~0 MS from mock SEDModel photometry; the right panel reveals MS evolution to high-z.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_main_sequence_cosmic_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_main_sequence_cosmic_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Star-forming main sequence: z = 0 → 2 cosmic evolution + recovery</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a population of N=200 quiescent galaxies replicating the SDSS Luminous Red Galaxy (LRG) sample selection (Eisenstein et al. 2001, SDSS-I): old, massive systems at z~0.3 with log M* ≈ 11 and ages sampling the red-sequence range Uniform(6, 11) Gyr (Thomas et al. 2005).">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sdss_lrg_stack_template_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sdss_lrg_stack_template`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SDSS Luminous Red Galaxy Stacked Template Spectrum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The cosmic star formation rate density (SFRD) — the total stellar mass created per unit time per unit comoving volume — rises from z~0 to peak at z~2, then declines toward higher redshift. Madau &amp; Dickinson 2014 assembled multi-wavelength observational data and fit a smooth analytic form:">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sfh_to_madau_dickinson_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sfh_to_madau_dickinson`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Madau-Dickinson 2014 cosmic SFRD(z) from a population of mock galaxies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Star formation rate calibrations depend on which wavelengths we observe. At high dust optical depth, UV-only SFR estimators severely underestimate the true SFR because dusty starbursts radiate most energy in the infrared. The hybrid SFR(UV+IR) recipe recovers the true SFR by combining both tracers.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sfr_uv_ir_consistency_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sfr_uv_ir_consistency`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SFR calibrations: UV only vs UV+IR hybrid estimators vs dust optical depth</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The stellar mass function (SMF) describes the number density of galaxies as a function of stellar mass, a fundamental probe of galaxy assembly. The Schechter function provides an excellent fit to observed SMF across cosmic time:">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_stellar_mass_luminosity_function_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_stellar_mass_luminosity_function`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar Mass and Luminosity Functions from Mock Survey</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Tully-Fisher relation is a tight empirical correlation between the baryonic mass of disc galaxies and their observed rotation velocity, parametrized as M_baryon ∝ V_rot^4 (slope 4.0 on the log-log plane).">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_tully_fisher_relation_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_tully_fisher_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Tully-Fisher Relation: Baryonic Mass — Rotation Velocity Scaling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The infrared excess (IRX = L_IR / L_FUV) versus UV-continuum slope β diagram is the standard tool for inferring attenuation in star-forming galaxies. However, β is degenerate between dust and stellar age: young dusty and old dust-free populations both exhibit red UV continua.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uv_slope_beta_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uv_slope_beta`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV slope β degeneracy: dust optical depth and stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="For any self-consistent dust model: L_IR ≈ L_UV_absorbed. Using tabulated Dale14 templates, agreement reaches ~5% across τ_V ∈ {0, 0.1, …, 4}. Non-conservation flags calibration issues in the dust emission routing.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uv_to_ir_bolometric_balance_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uv_to_ir_bolometric_balance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust energy balance: L_IR = L_UV_absorbed across opacity variations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Rest-frame U−V vs V−J colors separate star-forming from quiescent galaxies. The Williams+2009 quiescent wedge marks the boundary between dusty star-forming and passive systems.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UVJ diagram: rest-frame colors separate star-forming from quiescent</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Advanced Topics
===============

Hierarchical population inference, gradient diagnostics, batch fitting, panchromatic multi-component SEDs, joint photometry + spectroscopy.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The on-ramp for adding a custom physics block to tengri. Subclass SEDModelComponent, declare name, parameter_prefix, priors as class attributes, and implement predict(p, sed_in, wave). __init_subclass__ registers the new variant and auto-fills the inputs() / outputs() contracts.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_custom_attenuation_component_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_custom_attenuation_component`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Authoring a new physics block with SEDModelComponent</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Validates that AB magnitude zero-point definitions are consistent across filters. Compares photometry converted to magnitude via the formula m_AB = -2.5 log10(F_ν) - 48.6 against tengri&#x27;s built-in magnitude conversion. The AB magnitude system requires this relationship to hold across all filters—any deviation signals a zero-point calibration issue.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_ab_mag_zero_point_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_ab_mag_zero_point`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AB Magnitude Zero-point Consistency Check</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Verify tengri&#x27;s Calzetti implementation against Eq. 1 in Calzetti et al. 2000 (ApJ 533, 682). The canonical k(V=5500 Å) = 4.05 must be reproduced exactly.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_calzetti_kv_norm_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_calzetti_kv_norm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diagnostic: Calzetti 2000 attenuation law vs. published formula</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="In the dust-free limit with Case B recombination (T_e=10,000 K, n_e=100 cm^-3), the intrinsic Hα/Hβ ratio is 2.86, nearly independent of ionization parameter and metallicity below ~0.5 Z☉ (Storey &amp; Hummer 1995, MNRAS 272, 41). This diagnostic checks that tengri&#x27;s Cue nebular emulator reproduces the canonical value across its (logU, logZ_gas) grid, identifying any library drift or implementation errors.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_case_b_balmer_ratio_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_case_b_balmer_ratio`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Case B Hα/Hβ ratio across ionization and metallicity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="External ground truth: Chabrier 2003 PASP 115 763, Eq. 16–17.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_chabrier_imf_norm_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_chabrier_imf_norm`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Chabrier 2003 IMF — analytic normalization and SSP mean stellar mass</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compares tengri&#x27;s Planck18 cosmology implementation (DSPS-backed, Ω_m = 0.315, h = 0.674) against astropy.cosmology.Planck18 (which uses slightly different parameter values) across z = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0]. Validates luminosity distance d_L(z), comoving distance d_C(z), age(z), and comoving volume element consistency. Residuals should be stable across z and &lt;1% due to underlying parameter differences rather than numerical bugs. Tengri&#x27;s PLANCK18 parameters (Om0=0.315, h=0.674) match Planck 2018 published values.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_cosmology_vs_astropy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_cosmology_vs_astropy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cosmological Distance Validation: tengri vs Astropy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Validates that the νF_ν peak position of Draine &amp; Li (2007) dust emission templates follows Wien&#x27;s displacement law, an effective dust temperature diagnostic. The DL07 templates encode different dust temperatures for different U_min values; the Wien law applied to the νF_ν peak recovers this temperature.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_dl07_temperature_proxy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_dl07_temperature_proxy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Draine & Li 2007: dust temperature from SED peak position</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diagnostic figure comparing tengri&#x27;s Calzetti and Cardelli/CCM89 attenuation laws against the reference implementations in the dust_extinction package (Barbary et al., widely used by astropy workflows). Residuals reveal systematic offsets and validity ranges. If k(λ) residuals exceed 5% outside known singularities, the implementation may need verification against the original papers.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_dust_extinction_vs_pypi_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_dust_extinction_vs_pypi`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation law validation: tengri vs dust_extinction PyPI package</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Tests dust attenuation–emission consistency via energy conservation. Sweeps diffuse optical depth τ_diff while measuring agreement between independent attenuation and emission modules. Ratio = L_emitted / L_absorbed should equal 1.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_energy_balance_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_energy_balance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Energy balance: dust absorption vs. emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When ionizing photons escape (f_esc &gt; 0), fewer LyC photons ionize the ISM within the galaxy, suppressing all nebular line emission proportionally: L(Hα) ∝ (1 − f_esc) × Q_H, where Q_H is the intrinsic ionizing photon rate.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_fesc_lyc_conservation_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_fesc_lyc_conservation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman continuum escape fraction conservation in Cue nebular model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Validates the AB magnitude photometric filter convolution formula by computing the effective F_ν through a photometric filter manually and comparing against predict_photometry(). The AB convention defines the filter-weighted flux as">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_filter_integral_manual_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_filter_integral_manual`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Manual Filter Integral vs predict_photometry Consistency Check</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="tengri is a differentiable JAX package. Every model gradient ∂L/∂θ computed via jax.grad() should numerically match a central finite-difference approximation. This diagnostic builds a star-forming model with several free parameters, defines a chi-squared loss, and compares autodiff vs FD gradients for each parameter. A mismatch (&gt;1e-3) indicates a non-differentiable operation.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_gradient_finite_difference_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_gradient_finite_difference`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Autodiff gradients vs. finite-difference derivatives: diagnostic verification</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Verifies that JIT-compiled predictions are bit-identical to eager-mode evaluations. For predict_photometry and predict(params).lines, we sample random parameter sets and compare max relative difference between eager and JIT outputs. A value &lt; 1e-10 confirms no spurious numerical divergence; &gt; 1e-10 suggests platform-dependent floating-point behavior.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_jit_concrete_identity_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_jit_concrete_identity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">JAX JIT Compilation: Eager vs Compiled Numerical Equivalence</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diagnostic: the Hα luminosity traces the ionizing photon rate from young stars, which correlates with the instantaneous SFR. Kennicutt (1998, ApJ 498 541, Eq. 2) calibrated this relationship for Salpeter IMF; for Chabrier IMF (used by tengri), the coefficient is 4.97e-42: SFR / (M☉/yr) = 4.97e-42 × L(Hα) / (erg/s).">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_kennicutt_halpha_sfr_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_kennicutt_halpha_sfr`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hα-to-SFR calibration against Kennicutt (1998)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diagnostic: Lyman-series line optical depth τ_LS vs observed wavelength in the Lyman-alpha forest, comparing tengri&#x27;s Madau+1995 model to manual calculation from published coefficients (Madau 1995 Table 1, Eq. 15).">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_madau_published_table_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_madau_published_table`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Validating IGM transmission against Madau 1995 published table</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Internal consistency check: the cumulative SFR integral ∫₀ᵗ SFR(t) dt should equal the stellar mass returned by predict_properties(). This diagnostic varies the DPL SFH parameters and verifies that the two pathways (manual trapz of the trajectory vs library integration) agree to ~0.1%. Discrepancies &gt; 5% trigger a warning and would indicate a bug in either the SFH trajectory or the mass integration kernel.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_mass_conservation_sfh_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_mass_conservation_sfh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Mass conservation in SFH: manual integration vs predict_properties</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When constructing a model with priors like Uniform(0, 2), the sampling method model.spec.sample(key) should actually draw from that declared distribution. This example verifies the sampling implementation empirically: we draw 10000 samples from a model with mixed prior types (Uniform, LogUniform) and compare each empirical histogram against its theoretical PDF.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_prior_sample_distributions_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_prior_sample_distributions`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diagnosing prior sampling distributions with empirical histograms</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The rest-frame SED depends only on intrinsic galaxy properties (SFH, dust, metallicity, nebular, AGN) and is independent of redshift. Redshift only enters via the observation (wavelength shift, distance dimming, IGM attenuation). This diagnostic verifies that Prediction.rest_sed returns bit-identical SEDs across a range of redshifts for identical intrinsic parameters. Age-of-the-Universe constraints at high-z may truncate the SFH legitimately, producing smooth variation; any non-smooth jump signals a coupling bug.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_redshift_rest_invariance_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_redshift_rest_invariance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rest-frame SED Redshift Invariance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Verifies the SED chain is additive by comparing the full pred.rest_sed() output against a manual sum of per-component SEDs. The forward model chains stellar continuum through dust attenuation, dust emission, and nebular processing; if modular, the sum should reconstruct the total.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_sed_additivity_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_sed_additivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SED additivity: stellar, dust attenuation, emission, and nebular components</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Stellar population synthesis grids cover finite (age, metallicity) ranges. This diagnostic probes what happens at boundaries: clip, extrapolate, or error? We fix the SFH and vary stellar metallicity across the SSP grid boundary—inside, at the edge, and beyond. The resulting SEDs reveal the interpolation behavior; any NaN or error surfaces immediately in the plot.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_ssp_grid_edge_behavior_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_ssp_grid_edge_behavior`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP grid edge behavior: clipping, extrapolation, NaN</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WavePrecomp approximation pre-integrates SSP × filter LUTs and interpolates photometry through a redshift table, trading exact calculations for speed. This diagnostic compares exact-wave-grid photometry against WavePrecomp variants at different ztable densities n_z, showing how fractional errors decrease with finer redshift grids.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_diag_waveprecomp_accuracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_diag_waveprecomp_accuracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">WavePrecomp photometric accuracy across redshift grids</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Cramér-Rao bound from the Fisher Information Matrix shows that SDSS 5-band photometry alone cannot separately constrain age, dust, and metallicity. Adding NIR or MIR bands breaks the degeneracy by factors of 2–5×, quantifying the information gain from multiwavelength coverage.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_fisher_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_fisher_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-Dust-Metallicity Degeneracy: Fisher Analysis</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Computes the Jacobian d(flux)/d(theta) of the forward model and displays it as a heatmap showing which photometric bands are sensitive to which physical parameters. Each column shows normalized sensitivity to one parameter; dark blue/red indicates strong dependence.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_gradient_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_gradient_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gradient Sensitivity Heatmap</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates tengri&#x27;s Observation API for joint fitting across two data streams. Creates a mock galaxy with SDSS photometry and low-resolution spectroscopy, then recovers parameters via MAP. Shows how spectroscopy breaks photometric degeneracies.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_joint_fit_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_joint_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Joint Photometry + Spectroscopy Fit</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Low-level component orchestration using build_components and run_components. For production use, see plot_joint_fit.py and plot_radio_xray.py which use the SEDModel.build() nested-dict grammar.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_orchestrator_demo_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_orchestrator_demo`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Component Orchestrator End-to-End</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a full galaxy SED spanning X-ray to radio wavelengths. Shows stellar emission, dust attenuation, dust IR emission, radio synchrotron, and X-ray binary contributions. Demonstrates tengri&#x27;s multiwavelength physics modules for radio and X-ray—no SSP data required for these components.">

.. only:: html

  .. image:: /auto_examples/advanced/images/thumb/sphx_glr_plot_radio_xray_thumb.png
    :alt:

  :doc:`/auto_examples/advanced/plot_radio_xray`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Panchromatic SED: UV to Radio</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Showcase
========

Full-stack demonstrations: population forward modeling, gradient diagnostics, end-to-end workflows.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Fisher Information Matrix quantifies which parameter combinations are constrained by data and which are degenerate. Age-dust degeneracy: at fixed stellar mass, older stars + more dust produce the same multiwavelength SED as younger stars + less dust.">

.. only:: html

  .. image:: /auto_examples/showcase/images/thumb/sphx_glr_plot_gradient_degeneracy_direction_thumb.png
    :alt:

  :doc:`/auto_examples/showcase/plot_gradient_degeneracy_direction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Fisher Information Ellipses from the Hessian</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compute logarithmic sensitivities ∂(log F) / ∂(log θ) for each photometric band. Finite-difference methods (∂F/∂θ ≈ [F(θ+δ) − F(θ−δ)] / (2δ)) are slow and fragile; JAX autodiff computes exact sensitivities via one forward and reverse pass per parameter.">

.. only:: html

  .. image:: /auto_examples/showcase/images/thumb/sphx_glr_plot_jax_gradient_sensitivity_thumb.png
    :alt:

  :doc:`/auto_examples/showcase/plot_jax_gradient_sensitivity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Automatic differentiation: parameter sensitivities via jax.grad</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Recipes for common science cases">

.. only:: html

  .. image:: /auto_examples/showcase/images/thumb/sphx_glr_plot_recipes_gallery_thumb.png
    :alt:

  :doc:`/auto_examples/showcase/plot_recipes_gallery`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recipes for common science cases</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
