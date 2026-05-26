Examples gallery
================

.. toctree::
   :hidden:
   :titlesonly:
   :maxdepth: 1

   /auto_examples/quickstart/index.rst
   /auto_examples/recipes/index.rst
   /auto_examples/workflows/index.rst
   /auto_examples/sps/index.rst
   /auto_examples/sfh/index.rst
   /auto_examples/metallicity/index.rst
   /auto_examples/nebular/index.rst
   /auto_examples/dust_attenuation/index.rst
   /auto_examples/dust_emission/index.rst
   /auto_examples/agn/index.rst
   /auto_examples/radio/index.rst
   /auto_examples/xray/index.rst
   /auto_examples/igm/index.rst
   /auto_examples/photometry/index.rst
   /auto_examples/spectroscopy/index.rst
   /auto_examples/multiwavelength/index.rst
   /auto_examples/inference/index.rst
   /auto_examples/usecases/index.rst
   /auto_examples/advanced/index.rst


170+ standalone scripts demonstrating tengri's physics components, fitting
workflows, and end-to-end use cases. Each card below links to a per-script
page with the rendered figure, the full source, and a downloadable Jupyter
notebook.

**Browse by category.** Cards are organised into sections — quickstart and
workflows for end-to-end recipes, physics components for one-knob sweeps,
inference for fitter behaviour, and use cases for paper-style figures.

How to run an example locally
-----------------------------

Each script is a normal Python program::

    python examples/quickstart/plot_first_fit.py

Most physics examples (dust curves, SFH shapes, AGN spectra) need only
tengri's core dependencies. Fitting examples additionally require an SSP
grid — fetch one with::

    import tengri
    tengri.download_ssp()  # default fsps_prsc_miles_chabrier; see list_known_ssps()


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

Getting started with tengri — first fit and SED visualization.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The simplest end-to-end tengri workflow. We build a model with a truncated-skew-normal SFH and a two-component Calzetti dust attenuation, mock SDSS ugriz photometry at S/N = 20, then run a MAP fit to recover the input parameters. The figure shows the full rest-frame SED behind the five observed bands and the residuals of the MAP fit relative to the noise level.">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_first_fit_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_first_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Recovering a star-forming galaxy from 5-band SDSS photometry</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Build a model with both stellar and dust components. Predict the full SED with attenuation, then predict without dust absorption to isolate the absorbed UV-optical flux. The filled region shows how much light dust removes from the intrinsic stellar continuum.">

.. only:: html

  .. image:: /auto_examples/quickstart/images/thumb/sphx_glr_plot_sed_components_thumb.png
    :alt:

  :doc:`/auto_examples/quickstart/plot_sed_components`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation across the SED: intrinsic, attenuated, and absorbed</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Recipes
=======

Short, focused snippets for common how-to questions — comparing priors,
loading photometry from CSV, fixing redshift, swapping filter sets, and
saving/loading a posterior to disk.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The astronomer&#x27;s-eye-view of the tengri ingest path. Starting from a single CSV row of SDSS ugriz fluxes and per-band errors (the same shape pandas would hand you from a survey catalogue), we parse the row, build the photometric Observation from the column names, fit with MAP, and overlay the recovered SED on the observed bands with normalised residuals.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_real_data_fit_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_real_data_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">From a CSV row to a MAP SED fit, end to end</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="tengri.recipes ships several curated starting-point model configs that map common astronomer use-cases onto the nested-dict SEDModel.build grammar. This card overlays the rest-frame SED of every shipped recipe so users can pick by eye:">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_compare_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">What each shipped tengri recipe produces</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_compare_priors_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_compare_priors`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I combine a custom photometric filter with standard filters? This recipe generates a synthetic Gaussian filter at 2 μm and pairs it with SDSS optical bands, then predicts the full SED and photometry.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_custom_filter_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_custom_filter`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Register and use a custom photometric filter</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I load measured photometry from a table and fit it? This recipe generates mock photometry for 3 galaxies and fits each one independently with a MAP fit, demonstrating the workflow for catalogue-scale SED fitting.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_load_real_csv_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_load_real_csv`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Load and fit photometry from CSV</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How do I persist a posterior between sessions? This recipe runs a MAP fit, saves the result to HDF5, reloads it, and demonstrates basic analysis. Posterior objects can be checkpointed for long-running fits or multi-stage analysis pipelines.">

.. only:: html

  .. image:: /auto_examples/recipes/images/thumb/sphx_glr_plot_recipe_save_load_posterior_thumb.png
    :alt:

  :doc:`/auto_examples/recipes/plot_recipe_save_load_posterior`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Save and load a posterior to disk</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When redshift is known from spectroscopy, the SED fit is more precise than when inferring redshift from photometry alone. This recipe generates mock photometry at a known redshift, then fits it with redshift fixed (spectroscopic) and redshift free (photometric only), showing how redshift degeneracies affect parameter recovery.">

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

End-to-end fitting workflows — BPT classification, dust Monte-Carlo
resampling, high-z LBG fits, method comparison, and post-starburst recovery.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_bpt_classification_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_bpt_classification`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_dust_mc_resampling_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_dust_mc_resampling`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_high_z_lbg_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_high_z_lbg`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_method_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_method_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Two galaxies with very different physical properties — a dusty star-forming galaxy at z=0.3 and an unobscured Lyman-break galaxy at z=3.5 — can produce nearly identical ugrizY broadband fluxes. The 4000 Å break of the dusty low-z galaxy and the Lyman break of the high-z galaxy land at the same observed wavelength, so without intermediate bands or IR coverage the photo-z is bimodal.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_photoz_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_photoz_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The photo-z degeneracy: dusty z≈0.3 vs unobscured z≈3.5</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_post_starburst_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_post_starburst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="When did star formation in a galaxy stop? Optical-only color, the 4000 Å break, and Hα equivalent width respond on different timescales: NUV − r reddens within ~100 Myr of quenching (loss of O/B stars), D_n(4000) continues to rise over 1–3 Gyr as A stars evolve, and Hα EW drops fastest of all (within ~10 Myr) since it tracks only the youngest ionizing photons.">

.. only:: html

  .. image:: /auto_examples/workflows/images/thumb/sphx_glr_plot_workflow_quenching_diagnostics_thumb.png
    :alt:

  :doc:`/auto_examples/workflows/plot_workflow_quenching_diagnostics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Three diagnostics of quenching epoch in one figure</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Stellar Population Synthesis
=============================

DSPS-based SSP grids: age, metallicity, and spectral properties.

- ``plot_ssp_grid.py`` — SSP grid visualization (age, metallicity, spectrum)


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Young, metal-rich and old, metal-poor stellar populations can produce similar colours — a fundamental degeneracy in stellar population inference. This example builds a 2D grid of single-burst SSP-like models varying age (log10(t/Gyr) = -2 to 1.1) and metallicity (log10(Z/Zsun) = -2 to 0.4), then plots three SDSS broadband colours (u − r, g − r, NUV − r) as pcolormesh grids to visualize the degeneracy.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_age_metallicity_color_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_age_metallicity_color_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-metallicity colour degeneracy in SDSS colours</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The stellar populations in massive elliptical galaxies are typically α-enhanced ([α/Fe] &gt; 0) due to rapid star formation timescales that terminate before iron-peak elements fully enrich the gas (Thomas et al. 2005). This example demonstrates how increasing [α/Fe] shifts absorption features — particularly the Mg b and Fe5270 indices — which serve as diagnostics of star-formation history timescale.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_alpha_enhanced_population_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_alpha_enhanced_population`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Alpha-element enhancement in quiescent stellar populations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The same stellar population SED looks different depending on the units chosen for visualization. This example shows a single galaxy SED in three complementary representations on a 3-panel grid:">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_fnu_vs_flambda_units_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_fnu_vs_flambda_units`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SED Conventions: F_λ vs F_ν vs νF_ν</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The hydrogen-ionising photon production rate Q_H of a simple stellar population drops by ~5 dex from 1 Myr to 100 Myr as O stars die. Different SSP libraries predict different Q_H(t) because they differ in upper-IMF treatment, stellar rotation, and (most dramatically) whether massive binaries are included — BPASS extends the Q_H-producing phase to ~30 Myr.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ionising_lum_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ionising_lum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionising-photon production rate vs SSP age</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="M_★/L_band rises with stellar age in every band; the rise is steepest in g, where massive young stars dominate, and shallowest in K_s, where red giants contribute at every age past the first ~100 Myr.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_mass_to_light_ratios_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_mass_to_light_ratios`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar mass-to-light ratios vs SSP age, per band</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Four-panel overview of the DSPS SSP grid: age evolution at fixed metallicity, metallicity evolution at fixed age, monochromatic flux vs age, and color-color diagram across the full grid. Shows how stellar populations age from UV-hot to IR-red as they cool.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SSP Grid: Age and Metallicity Evolution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Different Initial Mass Functions produce different M/L ratios at fixed age and metallicity. We rescale a Chabrier SSP by literature M/L values for Chabrier, Kroupa, and Salpeter at 1 Gyr, solar metallicity. The NIR (where massive stars dominate the mass budget) is most diagnostic of IMF choice.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_ssp_imf_compare_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_ssp_imf_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">IMF Comparison: Mass-to-Light Ratio</div>
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

Parametric and stochastic star formation history models.


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

    <div class="sphx-glr-thumbcontainer" tooltip="Four perspectives on chemical evolution: (1) closed-box model with varying SFR timescales; (2) cumulative metallicity from different exponential SFHs; (3) leaky-box model showing how outflow rates suppress Z; and (4) age-metallicity relation across galactic radii. Together they show how star formation and galactic winds control the Z(t) history.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_chemical_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_chemical_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Chemical evolution: How SFH and outflows shape metal enrichment history</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The chemical composition of stars encodes the assembly history of galaxies. This figure demonstrates three metallicity evolution pathways available in tengri: (1) constant solar Z, (2) linear ramp from Z = 0.1 Zsun to Zsun over 13 Gyr of cosmic time, and (3) two-step enrichment (low-metallicity plateau at early times, then a sharp jump at lookback time 8 Gyr ago).">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_chemical_evolution_ramp_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_chemical_evolution_ramp`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Metallicity evolution: three scenarios for Z(t) and resulting SED</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The continuity prior (Leja+2019) penalises sharp transitions in adjacent-bin log-SFR ratios with a Student-t distribution (mu=0, sigma=0.3, df=2). This visualisation shows 200 independent draws from the registry default prior, displayed as percentile bands (5th, 25th, 50th, 75th, 95th) versus lookback time.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_continuity_prior_visualisation_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_continuity_prior_visualisation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Non-Parametric Continuity Prior: 200 Sample Draws</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing how the rising slope α (columns) and falling slope β (rows) together control the full SFH morphology. Early-time α determines assembly speed; late-time β sets the post-peak decay. The optical SED responds across each cell, revealing how parameter space maps to stellar age.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dpl_alpha_beta_grid_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dpl_alpha_beta_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Double power-law SFH parameter space: early growth α vs late quenching β</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The rising slope α of a double-power-law star formation history controls how rapidly the galaxy assembled its mass before the peak. Larger α means a more abrupt onset of star formation, leaving a younger O/B-star population at the time of observation and a steeper rest-frame UV slope. We vary α across the prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dpl_alpha_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dpl_alpha_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Early-time SFH slope α shapes the UV continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The falling slope β of a double-power-law SFH controls quenching after the peak. Large β means rapid quenching and an old stellar population; small β means a gentle tail and more mixed ages. We vary β across its prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_dpl_beta_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_dpl_beta_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Post-peak quenching slope β shapes stellar age distribution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The peak lookback time of a log-normal SFH controls when most stars formed, shifting the age structure and dramatically affecting UV slope, 4000 Å break strength, and NIR luminosity. We vary the peak time across its prior range with every other parameter fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_lnorm_peak_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_lnorm_peak_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Log-normal peak lookback time shifts stellar age and SED morphology</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Compare all parametric SFH models available in tengri. Each is evaluated on a lookback-time grid with representative parameters, showing the range of morphologies from smooth exponentials to sharp truncations. No SSP data required.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_parametric_sfh_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_parametric_sfh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Catalog of parametric star-formation-history models</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Three PSD models govern the frequency structure of stochastic SFHs: the default damped random walk (DRW), the Matern family (which includes DRW as a special case), and the extended regulator model. Plotted in frequency space at representative parameters. No SSP data required.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_alternatives_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_alternatives`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparison of power-spectral-density models for stochastic SFHs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 3×3 grid showing five stochastic-SFH realizations for each combination of amplitude σ (vertical axis) and damping timescale τ (horizontal axis). Larger σ produces more dramatic bursts; longer τ sustains those bursts. Each panel shows the mean smooth SFH (dashed) and colored realizations, revealing how the two PSD parameters together map to observable burstiness regimes.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_burstiness_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_burstiness`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD parameter space: amplitude σ (rows) and timescale τ (columns) control burstiness</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The field SFH employs a damped random walk (DRW) power spectral density (PSD) to govern stochastic star formation history realizations. Two parameters control the prior distribution of SFR time-variability:">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_burstiness_prior_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_burstiness_prior`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD-Governed Stochastic SFH Prior: Burstiness Corner Cases</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The amplitude σ of the power spectral density sets how dramatically star formation fluctuates around the smooth trend: σ ≈ 0 means nearly constant SFR, large σ produces dramatic bursts that leave imprints in UV slope, optical colors, and stellar masses. We vary σ across its prior range with the timescale τ fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_sigma_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_sigma_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD amplitude σ controls burst magnitude in stochastic SFHs</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The damping timescale τ (in Myr) of the power spectral density governs how long star-formation bursts persist. Short τ means rapid flickering; long τ means sustained episodes that leave their imprint on the SED. We vary τ across the prior range with the burst amplitude σ fixed.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_psd_tau_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_psd_tau_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">PSD timescale τ controls burst duration in stochastic SFHs</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="A galaxy with two separated bursts—one at 10 Gyr (old) and one at 0.3 Gyr (recent)— produces a SED that blends young hot and old cool stellar populations. Left panel shows the optical-to-NIR region in linear scale; right panel shows the full panchromatic SED in log-log, revealing the emission from both young and old stars.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_sfh_double_burst_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_sfh_double_burst`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dual-epoch star formation: old and recent bursts leave distinct SED signatures</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Each parametric SFH in tengri encodes a different prior on when a galaxy forms its stars. We overlay the SFR(t) shape of nine production-status forms at their default parameter values, all integrated to the same total stellar mass, so the differences are entirely in the shape — not the normalisation.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="How observable is an underlying ancient burst (10 Gyr ago) beneath a young (300 Myr) starburst? This example demonstrates the outshining problem in broadband photometry (Trager+ 2000, Renzini 2006): the young burst&#x27;s UV emission completely dominates over the ancient burst&#x27;s optical/IR, rendering the ancient population invisible to broadband SED fitting.">

.. only:: html

  .. image:: /auto_examples/sfh/images/thumb/sphx_glr_plot_two_burst_observability_thumb.png
    :alt:

  :doc:`/auto_examples/sfh/plot_two_burst_observability`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The outshining problem: young bursts eclipse ancient populations</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Metallicity
===========

Stellar metallicity Z and α-element enhancement effects on the SED.


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

    <div class="sphx-glr-thumbcontainer" tooltip="Metal-poor stars are hotter and bluer (less line blanketing), while metal-rich stars are redder due to increased opacity. We sweep stellar metallicity across the prior range with every other parameter fixed on a typical intermediate-age galaxy with modest dust attenuation.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_logzsol_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_logzsol_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Stellar metallicity drives UV-optical SED colour</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Metal-rich young populations and metal-poor old populations can produce similar optical colours — a fundamental degeneracy in galaxy fitting. This 3×4 grid shows normalised rest-frame continua at nine points in the age–metallicity plane, with each row fixed at one lookback-formation age and each column fixed at one metallicity. Dust is zeroed to expose the clean stellar continuum shape.">

.. only:: html

  .. image:: /auto_examples/metallicity/images/thumb/sphx_glr_plot_metallicity_age_grid_thumb.png
    :alt:

  :doc:`/auto_examples/metallicity/plot_metallicity_age_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Age-metallicity degeneracy in the stellar continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Metallicity evolution Z(t) depends on the balance between metal production (in supernovae) and metal removal (via outflows). This four-panel figure shows how different star formation timescales and outflow efficiencies η alter the enrichment history relative to a closed box (zero outflow). Top-left: closed-box enrichment timescale dependence. Top-right: impact of variable outflow rates. Bottom-left: closed vs leaky enrichment under constant SFR. Bottom-right: age-metallicity relation analogue — how different assembly epochs lead to different final metal content.">

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

Nebular emission backends comparison.


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

    <div class="sphx-glr-thumbcontainer" tooltip="The BPT diagram ([OIII]/Hβ vs [NII]/Hα) separates ionizing sources. Shocks (MAPPINGS V, Allen+2008) trace a sequence from HII regions through composite regions into Seyfert regions as velocity increases. We plot shock models alongside the standard demarcation lines.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_diagnostics_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_diagnostics`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram separates star formation from shocks and AGN</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Baldwin-Phillips-Terlevich (BPT) diagram ([OIII]/Hβ vs [NII]/Hα) separates ionization mechanisms: star formation, AGN, and composites.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_bpt_diagram_population_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_bpt_diagram_population`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BPT diagram population with star-forming galaxies and AGN-like models</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Cue has six tuning knobs that control HII-region ionization and the diffuse ionized gas. This six-panel tour sweeps each knob individually and reports the L_Hα response relative to the baseline, in dex. A flat line means the parameter has no effect on Hα at fixed other knobs.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_flex_tour_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_flex_tour`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue knob flexibility: six dimensions of HII region control</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The ionization parameter logU controls the hardness of the ionizing radiation field and drives rapid changes in optical line ratios. We show how [OIII]/[OII] (O32) and [OIII]/Hβ respond to logU from -4 to -1 at fixed metallicity (Z/Zsun = -0.5), demonstrating the use of O32 as a logU diagnostic (Kewley &amp; Dolphin 2002). Cue (Li et al. 2024, 2025) samples the ionizing spectrum flexibility and provides smooth gradients through metallicity, density, and ionization parameters for joint SED fitting.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_logu_line_ratios_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_logu_line_ratios`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionization parameter (logU) controls emission-line diagnostics</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Cue (Li, Leja &amp; Speagle 2023) maps a four-dimensional HII region control space — ionization parameter log U, gas-phase metallicity log Z_gas, ionizing-spectrum shape, and dust-to-metal ratio — onto an emission-line spectrum. A two-dimensional sweep over the two knobs most users will turn (``log U`` and log Z_gas) is shown for four diagnostic line ratios.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_cue_parameter_atlas_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_cue_parameter_atlas`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comprehensive sweep of the Cue nebular parameters</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diffuse ionized gas (DIG) has lower ionization parameter than HII regions, shifting galaxies toward the LINER region on the BPT diagram. We vary the DIG fraction from pure HII (0) to mixed gas (0.8).">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_dig_frac_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_dig_frac_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diffuse ionized gas suppresses strong optical lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Zoomed rest-frame spectrum of an ionised-gas-dominated SF galaxy with the strongest optical / near-UV emission lines labelled. Wavelengths are vacuum; line positions follow NIST/Atomic Line List.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_emission_line_atlas_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_emission_line_atlas`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optical emission-line atlas of a young star-forming galaxy</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="We sweep the ionising-photon escape fraction f_esc from 0 to 0.9 at fixed log U and metallicity, and read out the response in diagnostic-ratio space ([O III]/Hbeta etc.). Companion to plot_lyman_continuum_escape.py, which shows the same physics in SED space focused on the Lyman edge.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_fesc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_fesc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Escape fraction suppresses the optical line ratios, not just amplitudes</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Nebular free-free, free-bound, and two-photon emission respond to gas-phase metallicity (``logZ_gas``) through changes in metal cooling efficiency and ionization balance. This example demonstrates the metallicity sensitivity of the nebular continuum at fixed ionization parameter.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_gas_z_continuum_effect_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_gas_z_continuum_effect`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas-phase metallicity effect on nebular continuum</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Murphy+2011 SFR-Hα relation requires ionizing photons from stars younger than ~10 Myr. Constant-SFR models at ages 1–300 Myr show the calibration breaks at young (&lt;10 Myr; insufficient ionizing photons) and old (&gt;100 Myr; all stars too old to ionize) populations.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_halpha_sfr_calibration_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_halpha_sfr_calibration_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hα SFR calibration breaks at young ages</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three classical strong-line metallicity diagnostics computed as a function of gas-phase metallicity (``logZ_gas``). The plot spans 12 + log(O/H) from ~7 to ~9 and illustrates key observational features: the saturation of [O III]/H-beta at high metallicity (Kewley &amp; Dopita 2002), the monotonic but small dynamic range of [N II]/H-alpha (Marino et al. 2013), and the famous double-valued R23 ratio which peaks near 12 + log(O/H) ≈ 8.3 (Pagel et al. 1979).">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_line_ratios_metallicity_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_line_ratios_metallicity_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Optical line-ratio diagnostics along the metallicity gradient</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Emission line velocity dispersion broadens lines from a few km/s (narrow, kinematically resolved) to hundreds of km/s (unresolved at typical spectroscopic resolution). We show the [OIII] region broadened across the dynamical range.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_line_sigma_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_line_sigma_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Emission line broadening traces gas kinematics</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Varying log U from -4 to -1.5 on a young star-forming galaxy at fixed metallicity changes every strong optical line simultaneously — Hbeta, [O III], Halpha, [N II], [S II] all move together. We plot the full 4000-7500 A SED so the continuum context is visible alongside the line forest. Companion to plot_cue_logu_line_ratios.py, which projects the same sweep onto two-line diagnostic axes.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logu_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logu_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Ionisation parameter reshapes the full optical SED, not just line ratios</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 1-D log Z_gas sweep on the SED scale, complementing the 2-D atlas in plot_cue_parameter_atlas.py and the line-ratio projection in plot_strong_line_metallicity_diagnostics.py. Reader sees how every strong optical line moves together as Z_gas climbs, with [N II]/Halpha and [O III]/Hbeta the textbook diagnostics.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_logz_gas_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_logz_gas_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Gas metallicity reshapes the optical nebular continuum and line forest</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Lyman-alpha (Lyα) equivalent width (EW) traces stellar population age through the presence and strength of massive O stars. We construct a sequence of constant star-formation-rate (CSF) models with ages ranging from 1 Myr to 30 Myr at fixed metallicity (Z = Zsun; logZ = 0), compute the rest-frame Lyα emission line luminosity and the underlying continuum at 1216 Å, then derive EW(Lyα) = L(Lyα) / L_continuum.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_lyalpha_ew_vs_age_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_lyalpha_ew_vs_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-alpha equivalent width peaks during O-star dominance</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="We zoom on the Lyman-continuum region (rest 800-1300 A) and sweep the escape fraction f_esc to show how the 912 A discontinuity deepens as more ionising photons leave the ISM unabsorbed. Companion to plot_fesc_sweep.py, which projects the same physics into optical line-ratio diagnostics.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_lyman_continuum_escape_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_lyman_continuum_escape`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman-continuum escape fraction reshapes the SED around the 912 A edge</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Two nebular backends, same SFH, same dust, same metallicity. BakedIn pulls line ratios from the SSP grid (Conroy + Byler wNE templates); Cue (Li, Leja &amp; Speagle 2023) is a neural emulator over the CLOUDY parameter space, run here at log U = -3.0.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_neb_backend_compare_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_neb_backend_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Nebular backends side-by-side: BakedIn vs Cue</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Young massive stars produce harder ionising continua and drive the nebular emission toward higher [O III]/Hbeta. We sweep the SFH timescale tau_gyr from 0.1 to 2 Gyr on a single dual power-law model and plot the resulting line ratios against the Kewley+2001 / Kauffmann+2003 demarcation curves. The locus migrates from the star-forming wing into the composite region as the population ages — SFH timescale is the upstream knob behind the BPT ionisation sequence.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Compare Cue (neural emulator; current recommended path) against traditional photoionization grids (CloudyGrid) and SSP-embedded nebular. Shows [OIII] and H-alpha regions on a young starburst.">

.. only:: html

  .. image:: /auto_examples/nebular/images/thumb/sphx_glr_plot_nebular_backends_thumb.png
    :alt:

  :doc:`/auto_examples/nebular/plot_nebular_backends`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cue nebular emulator vs alternatives</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Four widely-used optical strong-line metallicity diagnostics evaluated across the Cue logZ_gas prior. Each one carries a different systematic — Pettini &amp; Pagel 2004 O3N2 saturates at high Z, the R23 ratio is double-valued, N2 (Marino+2013) is monotonic but small dynamic range, and the [Ne III]/[O II] diagnostic is weakly Z-dependent.">

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

How starlight is extincted on its way out of the galaxy — Calzetti vs
power-law slopes, the 2175 Å UV bump, birth-cloud and diffuse-ISM optical
depths, two-component geometry, and law comparisons.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The tengri library offers six attenuation laws covering the morphology-geometry spectrum: Milky Way (Cardelli), SMC (Pei), starburst (Calzetti, Conroy), and theoretical models (Kriek &amp; Conroy, power law). At fixed τ_V = 1, their curves expose the 2175 Å bump (MW/Cardelli), slope differences (SMC is greyer, Calzetti is redder), and parametric extensions (Kriek &amp; Conroy).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_attenuation_law_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_attenuation_law_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The six headline dust attenuation laws span MW, SMC, and starburst geometries</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust attenuation laws encode how interstellar dust preferentially absorbs short-wavelength (UV) starlight relative to optical/IR. The wavelength dependence is empirically calibrated to extinction measurements in the Milky Way (Cardelli+1989), Large/Small Magellanic Clouds (Pei 1992), and starburst galaxies (Calzetti+2000, Kriek+Conroy 2013).">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_attenuation_law_family_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_attenuation_law_family`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation laws: family comparison across UV to near-IR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_birth_cloud_vs_diffuse_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_birth_cloud_vs_diffuse`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Charlot &amp; Fall 2000 two-component dust model splits attenuation into: - τ_bc (birth-cloud): attenuates only young stellar ages (&lt; 10 Myr) - τ_diff (diffuse ISM): attenuates all stellar light">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_birth_cloud_vs_diffuse_age_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_birth_cloud_vs_diffuse_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth-cloud attenuation age dependence: Charlot & Fall 2000</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_cardelli_rv_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_cardelli_rv_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The six headline dust attenuation laws plotted over the full UV-through-NIR range (0.1–3 μm), extending beyond the 2175 Å bump region to show how curves flatten in the infrared. Red-shifted galaxies observe longer wavelengths at rest frame, so the IR slope controls K-correction factors and SED fitting degeneracies.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_curves_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_curves`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation laws from UV through near-infrared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_geometry_screen_vs_mixed_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_geometry_screen_vs_mixed`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three dust geometries—foreground screen (power-law), mixed slab (Calzetti), and clumpy two-phase (SMC)—proxy different physical arrangements via their attenuation laws. At fixed τ_V = 1, geometry controls the spectral shape: screens are reddest, clumpy geometries are greyest. Transmission curves show how each law transforms a stellar continuum.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_dust_geometry_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_dust_geometry_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust geometry shapes the extinction: screen vs mixed vs clumpy</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Four named attenuation laws applied to the same intrinsic SED at the same V-band optical depth (τ_V = 1.0), illustrating how dust geometry and grain-size composition vary across the local universe.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_galactic_zoo_dust_laws_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_galactic_zoo_dust_laws`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust attenuation laws across the galaxy zoo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Birth-cloud dust optical depth τ_bc attenuates only the youngest stellar light (age &lt; ~10 Myr), controlling nebular emission from embedded HII regions. τ_bc effects are clearest on young star-forming populations; we use a 500 Myr starburst and vary τ_bc across the prior range.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_tau_bc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_tau_bc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Birth cloud dust suppresses young-stellar UV and nebular emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Diffuse ISM dust optical depth τ_diff attenuates all stellar light (young + old). Higher τ_diff reddens the optical continuum and weakens the 4000 Å break, signaling aging stellar populations. We vary τ_diff across a range with every other parameter fixed on a typical star-forming galaxy.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_tau_diff_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_tau_diff_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Diffuse ISM dust attenuates all stellar populations</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The 2175 Å UV bump from PAHs and small graphite grains sweeps from absent to Milky-Way strength via the dust_bump_strength knob. At zero, the attenuation curve is a smooth power law; at MW-like values, the bump dominates the UV. We show the attenuation law (not a galaxy SED) to isolate the curve shape.">

.. only:: html

  .. image:: /auto_examples/dust_attenuation/images/thumb/sphx_glr_plot_uv_bump_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_attenuation/plot_uv_bump_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The 2175 Å UV bump traces small-grain dust populations</div>
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


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Dust Emission
=============

How dust re-radiates absorbed starlight in the IR — PAH features and the
q_PAH / U_min sweeps of Draine & Li templates, modified-blackbody
temperature sweeps, and dives into the BOSA, THEMIS, PAHspec, and
Astrodust (HD23) template grids.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Per-H grain volume distribution versus grain radius for the Hensley &amp; Draine 2023 fiducial size distribution (MW high-latitude R_V=3.1 sightline).">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_01_size_distribution_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_01_size_distribution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH size distribution</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Emission per H per ionization parameter U across the Hensley &amp; Draine 2023 grid. Dividing by U reveals its effect: PAH-to-FIR ratio plateaus in FIR (U-independent) but rises steeply with U in MIR.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_02_emission_vs_lgU_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_02_emission_vs_lgU`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH emission vs log U</div>
    </div>


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

    <div class="sphx-glr-thumbcontainer" tooltip="Compare dust emission templates at fixed infrared luminosity. Shows how spectral shape changes across modified-blackbody, Draine+2021 PAHspec, and Hensley &amp; Draine 2023 Astrodust while bolometric output remains conserved.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_04_sedmodel_dust_emission_swap_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_04_sedmodel_dust_emission_swap`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">DustEmissionSEDComponent — swap MBB / PAHspec / Astrodust</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Ionization fraction and alignment efficiency versus grain size for the Hensley &amp; Draine 2023 fiducial size distribution.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_05_ionization_alignment_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_05_ionization_alignment`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH ionization fraction and alignment</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Extinction opacity, polarized extinction, and single-scattering albedo for the Hensley &amp; Draine 2023 fiducial size distribution.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_06_extinction_and_scattering_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_06_extinction_and_scattering`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH extinction, scattering, and albedo</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spinning dust microwave emission across 10–100 GHz, decomposed by grain (Astrodust/PAH) and phase (CNM/WNM), for the Hensley &amp; Draine 2023 fiducial.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_07_spinning_dust_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_07_spinning_dust`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH spinning-dust microwave emission</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Polarized emission and polarization fraction from Astrodust grains at the Hensley &amp; Draine 2023 fiducial ionization parameter.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_astrodust_hd23_08_polarized_emission_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_astrodust_hd23_08_polarized_emission`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Astrodust+PAH polarized emission and polarization fraction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The BOSA infrared template library is parametrised jointly by total infrared luminosity log L_TIR and specific star formation rate log sSFR. Neither axis alone tells the full story: at fixed sSFR the FIR peak migrates with L_TIR (dust temperature), while at fixed L_TIR the PAH mid-IR forest brightens with sSFR. Three side-by-side panels at fixed sSFR overlay three L_TIR values each, making the 2-D dependence legible in a single figure rather than two skinny 1-D loops.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_grid_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA library: PAH features and FIR peak depend on both sSFR and L_TIR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep infrared luminosity across the BOSA grid at fixed specific star formation rate. Increasing L_TIR heats dust, shifting FIR peak blueward and enhancing PAH relative to continuum. Library is normalised by ∫Lν dν=1; shape variation with L_TIR is intentionally small.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_bosa_ltir_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_bosa_ltir_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">BOSA: log L_TIR sweep at fixed log sSFR</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="All six dust-emission ingredients shipped with tengri, called with the same absorbed bolometric luminosity (1e10 L_sun) and the same warm-dust temperature (35 K). Analytic models (modified BB, Casey 2012, energy-balance split) drop sharply blue-ward of the warm-dust peak; template-based libraries (DL07, DL14, Dale+2014) carry PAH features in the 3-20 μm window. Template models silently skip if the data files aren&#x27;t available.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_dust_emission_models_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_dust_emission_models`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust-emission model family at fixed L_abs</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="All seven shipped dust IR-emission libraries reprocessing the same absorbed UV power into the IR, normalised so the integrated L_IR(8–1000 μm) is identical across curves. The differences then sit entirely in the SED shape — peak wavelength (T_dust proxy), PAH-feature amplitude in the 3–20 μm window, and how steeply the sub-mm tail falls.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_ir_library_compare_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_ir_library_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust IR-emission library comparison at fixed L_dust</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The mid-infrared ionisation-parameter sensitivity is library-specific, but the FIR-peak migration with rising log U is a universal prediction. We overlay the Hensley &amp; Draine 2023 (Astrodust+PAH) and the Draine+2021 PAHspec libraries at the same three log U values to surface where the two agree (FIR peak position) and where they differ (MIR PAH-feature strength and the Astrodust silicate plateau near 18 microns).">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Sweep radiation-field distribution slope across the THEMIS grid at fixed grain content and minimum intensity. Lower alpha shifts weight toward high U, warming dust and shifting FIR peak blueward; higher alpha approaches single-U.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_themis_alpha_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_themis_alpha_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">THEMIS: power-law slope alpha sweep</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The starlight intensity floor U_min sets the temperature of the diffuse-ISM component in template-based dust libraries. We compare the Draine &amp; Li 2007 grid (fixed q_PAH = 2.5%) and the THEMIS grid (fixed q_HAC = 0.17) at three matched U_min values to highlight that the FIR-peak position is remarkably consistent between the two grain-physics paradigms, while THEMIS predicts a stronger mid-IR continuum from its hydrogenated amorphous carbon component.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_umin_cross_library_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_umin_cross_library`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Minimum radiation field U_min: DL07 and THEMIS agree on the FIR peak</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Minimum radiation field intensity U_{\min} controls diffuse dust heating — higher U_{\min} → hotter dust → FIR peak shifted blueward. DL07 (Draine &amp; Li 2007) and THEMIS (Jones et al. 2017) are overlaid so that their response to U_{\min} can be compared on the same axes.">

.. only:: html

  .. image:: /auto_examples/dust_emission/images/thumb/sphx_glr_plot_umin_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/dust_emission/plot_umin_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">U_min sweep: DL07 vs THEMIS FIR peak migration</div>
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

AGN disc and torus SED templates.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The X-ray corona response of an AGN depends jointly on bolometric luminosity (which sets the X-ray normalisation through the Lusso &amp; Risaliti L_X-L_UV correlation) and on the UV-to-X-ray slope alpha_OX (which sets the relative balance of UV and X-ray emission). Four panels at log L_bol = 44, 45, 46, 47 erg/s overlay three alpha_OX values each, showing that the absolute X-ray luminosity scales with L_bol while the X-ray-to-UV ratio is set independently by alpha_OX.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="The bolometric correction K_X = L_{\rm bol} / L_X relates the total AGN luminosity to the flux in a single observational band. For X-ray selected AGN, this is essential for converting observed X-ray fluxes back to total AGN power.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_bolometric_correction_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_bolometric_correction`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN Bolometric Correction: K_X(L_bol) Across Four Bands</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The torus inclination angle determines how much cold dust emission we observe. Face-on (high cos_inc) views show a smooth thermal bump; edge-on (low cos_inc) views expose more reprocessed mid-infrared flux and can show silicate absorption features.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_cos_inc_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_cos_inc_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: viewing angle tunes IR profile shape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six accretion-disc backbones at fixed bolometric luminosity log L_bol = 12.5 (in log L_sun), evaluated in isolation with the host suppressed and no torus/lines/dust. The differences between the curves are entirely how each model partitions the disc power across wavelength: pure blackbody vs warm Comptonization, relativistic vs Newtonian potential, empirical-fit vs first-principles continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_disc_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_disc_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN disc continuum: model comparison at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four AGN configurations of increasing physical complexity at the same bolometric luminosity (log L_bol = 12.5 in L_sun units) — bare multicolour disc, +SKIRTOR torus, +NLR narrow-line forest, and an empirical QSOgen template that bundles all of the above. The reader sees which spectral feature each block introduces (mid-IR torus bump, optical narrow lines, broad UV continuum) and which are essentially universal across the modelling choice.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_hierarchy_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_hierarchy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Building up an AGN SED: disc, then torus, then lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A Seyfert galaxy SED is decomposed photometrically by varying the AGN contribution fraction agn_frac from 0 (pure host) to 1.0 (pure AGN) to 0.5 (composite). This demonstrates how to isolate the AGN contribution from the host galaxy using a single model and varying a structural parameter — useful for diagnosing photometric AGN contamination.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_host_decomposition_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_host_decomposition`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN host-galaxy decomposition: disentangling Seyfert contributions</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Four production line backbones layered on top of the same disc + torus at fixed log L_bol = 12.5. The line backbone controls which optical/UV emission features the model produces — narrow-line region forbidden lines, broad-line permitted lines, or pre-canned empirical line lists.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_lines_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_lines_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN emission-line backbones compared</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The disc continuum normalisation tracks bolometric luminosity directly; the disc temperature shifts more subtly with the implied accretion rate. Varying agn_log_lbol from 10 to 14 (in log10 L_sun) sweeps four orders of magnitude in disc luminosity, comparable to typical Seyfert through bright-QSO regimes. The spectral shape (slope, peak position) remains nearly fixed.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_log_lbol_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_log_lbol_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen disc: bolometric luminosity controls overall flux</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The torus opening angle (``oa_skirtor``) sets how much of the central disc is visible. A narrower torus (smaller opening angle) hides the disc and relies on reprocessed torus emission; a more open torus exposes the hot disc continuum and shifts the SED blueward.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_oa_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_oa_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: opening angle controls exposed disc fraction</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Dust-free quasar spectra are intrinsically blue in the UV and optical. Intrinsic dust reddening ebv (E(B−V)) reddens the continuum via extinction. Varying ebv from 0 to 0.4 shows the transition from unobscured type-1 QSO colours to moderately dust-enshrouded systems.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_qsogen_ebv_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_qsogen_ebv_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen disc: dust reddening tunes UV to optical colour</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The QSOgen model includes a UV/optical emission-line forest and broad Balmer continuum on top of the underlying disc. The relative strength of these line features with respect to the continuum controls the slope and colour of the UV–optical SED.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_qsogen_emline_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_qsogen_emline_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen lines: emission-line contributions vary with luminosity</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The SKIRTOR clumpy torus has a radial dust-density profile with power-law index p. Steeper profiles (higher p) concentrate more dust closer to the disc, reducing the mid-IR peak temperature and shifting flux toward the far-IR. Flatter profiles distribute dust more uniformly and hotter on average.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_skirtor_p_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_skirtor_p_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: radial density profile tunes IR emission peak</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 9.7 μm optical depth tau_97 controls the strength of silicate dust absorption/emission in the mid-infrared. Thin tori (tau ~3) show weak features and more continuum; thick tori (tau ~11) develop deep absorption troughs or bright emission depending on viewing angle.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_tau_skirtor_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_tau_skirtor_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: optical depth governs silicate feature strength</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Six dusty-torus libraries reprocessing the same accretion-disc continuum at fixed log L_bol = 12.5 (in log L_sun) and standard inclination. The disc is held at multicolor (Kubota &amp; Done 2018) so the differences in the curves are entirely how each torus library geometrically distributes hot grains and re-emits the absorbed UV in the MIR.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_torus_compare_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_torus_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN dusty torus: library comparison at fixed L_bol</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The unified AGN model attributes the Type-1 vs Type-2 dichotomy to geometry alone. Three inclinations of an identical disc + SKIRTOR torus + broad-line region (Type 1, face-on, cos i = 0.95), torus edge (intermediate, cos i = 0.5), and edge-on (Type 2, cos i = 0.1). The broad UV bump and BLR lines vanish behind the torus at high inclination; the mid-IR torus reprocessed emission stays.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_agn_type12_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_agn_type12`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Same AGN, different viewing angle: Type 1 to Type 2 by inclination</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduces the UV-to-X-ray connection panel from Yang et al. 2020 (X-CIGALE Fig. 3): the X-ray corona is normalised through the Just+07 alpha_OX-L_2500 relation, anchored at the disc-derived L_2500. Offsets delta_alpha_OX from -0.3 to +0.3 dex pivot the X-ray power-law about the 2500 A anchor — the disc UV stays fixed (single curve at log lam &gt; 1), only the X-ray normalisation moves.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_alpha_ox_uv_xray_connection_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_alpha_ox_uv_xray_connection`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">delta_alpha_OX pivots the X-ray spectrum about the disc UV anchor</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The dimensionless spin parameter a determines the innermost stable circular orbit (ISCO). Higher spin pushes ISCO inward, raising peak disc temperature and shifting the UV bump bluer. This demonstrates the classic Kerr black hole effect on thin disc accretion: Schwarzschild (a=0) → near-extremal Kerr (a*=0.998).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_bh_spin_disc_continuum_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_bh_spin_disc_continuum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Black hole spin effect on accretion disc UV peak temperature</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The agn.disc, agn.lines, agn.feii, agn.torus, agn.atten sub-blocks of SEDModel.build are composable: turning one on at a time and overlaying the all-on reference (dashed grey) shows which features each sub-block contributes. Five panels at fixed log L_bol = 12.0, all built via the public nested-dict grammar:">

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

    <div class="sphx-glr-thumbcontainer" tooltip="The collaborator workflow for adding a new AGN model. We define a toy single-temperature blackbody torus, register it with register_agn_model, confirm it is discoverable through tengri.list_agn_models and tengri.describe, then evaluate it on the public SEDModel.build path and plot it next to the production SKIRTOR torus at the same bolometric luminosity. The toy curve is a greybody; the SKIRTOR curve carries the silicate 9.7 micron feature and the inclination-dependent geometry the toy elides.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_custom_torus_extension_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_custom_torus_extension`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Registering a custom AGN torus model and using it through SEDModel.build</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed black hole mass M_BH = 10^8 M_sun, the accretion disc luminosity and spectral shape scale with Eddington ratio λ_Edd = L_bol / L_Edd. Here we sweep λ_Edd from 0.001 to 1.0 at five logarithmic steps and overlay the disc continuum (100–3000 Å) to show how lower accretion rates produce fainter discs with unchanged spectral shape (Shakura &amp; Sunyaev 1973).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_eddington_ratio_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_eddington_ratio_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Eddington ratio sweep: multicolor disc thermal scaling</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The black hole mass (M_BH) and stellar bulge mass (M_) of galaxies follow a tight empirical scaling relation. This example builds 12 mock AGN-hosting galaxies sweeping log M_ from 9 to 12 M_☉, derives M_BH from the published Kormendy &amp; Ho (2013) and Reines &amp; Volonteri (2015) relations, and constrains the AGN bolometric luminosity via a random Eddington ratio (λ_Edd ∈ [0.001, 0.1]).">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_mbh_mstar_relation_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_mbh_mstar_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">M_BH–M_* scaling relation: Kormendy & Ho 2013 and Reines & Volonteri 2015</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Identical AGN configuration (multicolour disc + SKIRTOR torus at log L_bol = 12.5), one with the narrow-line region (FWHM ~ a few hundred km/s, characteristic Type-2 spectrum) and the other with the broad-line region (FWHM ~ thousands of km/s, Type-1). Side-by-side zooms on the UV (Ly-alpha, C IV) and the optical (Hbeta, [O III], Halpha) make the velocity-width contrast unmistakable while controlling for continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_nlr_blr_lines_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_nlr_blr_lines`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Narrow vs broad line region: a velocity-width contrast in two windows</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduces Figure 1 of Yang et al. 2020 (the X-CIGALE polar-dust introduction): SMC-law attenuation of the AGN disc by dust above the torus, plus an energy-conserving mid-IR greybody re-emission. Two panels at cos_inc = 0.95 (Type-1, face-on into the polar cone) and cos_inc = 0.10 (Type-2, edge-on view of the torus) for opening angle 40°. We sweep agn_polar_ebv from 0.00 to 0.30 — covering the empirical range Yang+2020 anchor against red quasars.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_polar_dust_ebv_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_polar_dust_ebv_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polar-dust E(B-V) sweep for Type 1 and Type 2 AGN sightlines (X-CIGALE)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Polar dust disc attenuation applies only to Type 1 (face-on) sightlines — the equatorial torus already screens the disc for Type 2. The bi-conical polar dust absorbs disc photons regardless of viewing angle, however, and re-emits them isotropically as a FIR greybody (Casey 2012). So both Type 1 and Type 2 sweeps show the FIR re-emission bump growing with E(B-V); only the UV/optical attenuation is gated by sightline.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_polar_dust_ebv_type12_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_polar_dust_ebv_type12_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Polar dust E(B-V) reddens Type 1 & 2 AGN differently</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Temple, Hewett &amp; Banerji (2021) QSOgen empirical template, used as the agn.disc.type=&quot;qsogen&quot; selector. We sweep log L_bol from 10.0 to 13.5 (in L_sun units) at fixed redshift to show that the template&#x27;s spectral shape is approximately self-similar across the quasar luminosity function — the only knob that moves features (the Baldwin-effect drop in C IV/Ly-alpha equivalent width) is the bolometric normalisation.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_qsogen_spectrum_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_qsogen_spectrum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSOgen empirical quasar SED across four decades of bolometric luminosity</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_reverberation_size_luminosity_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_reverberation_size_luminosity`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three accretion-disc backbones at the same bolometric anchor (log L_bol / L_sun = 12.5): the Richards et al. 2006 empirical mean Type-1 SDSS quasar template, the Temple, Hewett &amp; Banerji 2021 empirical QSOgen, and the Shakura-Sunyaev multicolour disc (the outer-disc component of Kubota &amp; Done 2018). Each is normalised to the same bolometric output so the differences are entirely in spectral shape — Richards+2006 is broader than QSOgen and carries the infrared bump from its host-galaxy-corrected composite, while the multicolour disc cuts off sharply on either side of the big blue bump.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrate how the SKIRTOR clumpy radiative-transfer torus (Stalevski+2016) reprocesses the hot accretion disc as a function of viewing angle.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_inclination_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_inclination_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR AGN torus: inclination-dependent obscuration</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 9.7 μm and 18 μm silicate bands are strong diagnostics of AGN torus orientation. When viewing the torus face-on (high cos_inc), dust emission dominates and silicates appear in emission. Edge-on views (low cos_inc) show silicates in absorption against the hot dust continuum.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_silicate_features_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_silicate_features`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR Torus: Silicate features from face-on to edge-on</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The SKIRTOR clumpy torus model (Stalevski et al. 2016) emits thermal IR radiation that depends strongly on two parameters: viewing angle (inclination θ via cos_inc) and optical depth (``tau_97`` at 9.7 μm). Face-on systems show a smooth thermal continuum; edge-on systems develop deep 9.7 μm silicate absorption. Higher τ increases reprocessed flux.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_variants_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_variants`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus: viewing angle and optical depth effects</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Reproduces the SKIRTOR vs Fritz comparison from Yang et al. 2020 (X-CIGALE Fig. 2). Both libraries re-emit the same disc-absorbed luminosity in the mid-IR; the mid-IR peak amplitude differs by ~0.5 dex because SKIRTOR&#x27;s clumpy 3-D Stalevski+2016 RT redistributes heating more efficiently into the bright NIR-MIR continuum than a smooth-density torus. tengri does not ship Fritz+2006 directly; we substitute Silva+04 (template-based smooth torus, the closest contemporary analogue) — the qualitative contrast (clumpy bright MIR vs smooth fainter MIR) is preserved.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_skirtor_vs_smooth_torus_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_skirtor_vs_smooth_torus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR clumpy vs Silva+04 smooth-torus comparison (X-CIGALE Fig. 2)</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="The torus half-opening angle (OA, polar half-angle in degrees) controls the covering fraction and the relative strength of direct vs. re-processed AGN emission as a function of observer inclination. Smaller OA (narrow torus) covers a smaller solid angle, reducing the fraction of reprocessed emission visible face-on and increasing direct continuum. Larger OA (flared torus) increases covering, suppressing direct light and boosting thermal re-emission in the mid-infrared.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_torus_opening_angle_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_torus_opening_angle`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SKIRTOR torus opening angle sweep: covering factor and MIR emission</div>
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


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Accretion disc reverberation mapping reveals how the hot UV-emitting inner disc responds to ionizing source changes. Fausnaugh+2016 observed NGC 5548 using HST multi-band photometry (UV, optical) and found that UV variations lead optical by τ(λ) — the light-crossing time across the effective emission radius at wavelength λ.">

.. only:: html

  .. image:: /auto_examples/agn/images/thumb/sphx_glr_plot_variability_continuum_lag_thumb.png
    :alt:

  :doc:`/auto_examples/agn/plot_variability_continuum_lag`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN UV→optical continuum reverberation: light-crossing time lags</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Radio
=====

Star-formation radio emission and the FIR–radio correlation.


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

    <div class="sphx-glr-thumbcontainer" tooltip="The FIR-radio correlation links far-infrared luminosity (dust-reprocessed star-formation energy) to 1.4 GHz synchrotron emission. The dimensionless q_IR parameter relates the two via L_IR ∝ L_1.4GHz^(10^q_IR/2.5). Brighter starbursts emit stronger radio across all frequencies. We sweep L_IR over 10^10–10^13 L_sun at fixed q_IR = 2.64 (canonical; Bell 2003).">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_lir_relation_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_lir_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FIR-radio correlation: L_IR × q_IR sets radio loudness scale</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Radio loudness R = log_10(L_5GHz / L_B) quantifies the ratio of AGN radio to optical luminosity. Radio-quiet AGN have R ≲ 1; radio-loud sources (FR I/II, blazars) reach R ∼ 3–5. Each decade in R corresponds to an order of magnitude increase in jet radio luminosity at fixed bolometric AGN power. We sweep R ∈ [0, 4] at fixed L_bol = 10^44 erg/s (Seyfert-1-like) and α_agn = 0.7.">

.. only:: html

  .. image:: /auto_examples/radio/images/thumb/sphx_glr_plot_radio_loudness_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/radio/plot_radio_loudness_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN radio loudness R: orders of magnitude in jet power</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed host (constant SFR = 3 M_sun/yr, Condon-92 synchrotron + free-free) we sweep the composable AGN&#x27;s bolometric luminosity agn_log_lbol from 9 to 13 (in log L_sun). The host alone produces a power-law GHz continuum; the AGN superposes a flatter-spectrum jet component that takes over above log L_bol ≳ 11.5 — the classic radio-loud / radio-quiet division emerges from this competition.">

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

Multi-wavelength X-ray components: X-ray binaries (HMXB + LMXB) and AGN coronae.

Star-Forming Galaxies
^^^^^^^^^^^^^^^^^^^^^

- ``plot_xray_sf.py`` — X-ray binary scaling with SFR and stellar mass

AGN Coronae
^^^^^^^^^^^

- ``plot_xray_agn.py`` — AGN X-ray coronae: luminosity sequence and spectral hardness
- ``plot_xray_gamma_sweep.py`` — Photon index γ controls power-law steepness
- ``plot_E_cut_sweep.py`` — Exponential cutoff E_cut governs hard-tail rollover
- ``plot_alpha_ox_sweep.py`` — UV-to-X-ray slope α_ox controls normalisation


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

    <div class="sphx-glr-thumbcontainer" tooltip="The CIGALE-faithful corona derives the X-ray normalisation from L_2500 via the empirical alpha_OX-L_2500 correlation. tengri ships three published parametrisations:">

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

      <div class="sphx-glr-thumbnail-title">AGN UV-to-X-ray power-law slope alpha_OX controls X-ray normalisation</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The AGN X-ray corona produces a cut-off power-law (photon index Gamma roughly 1.8, E_cut around 300 keV) normalised through the alpha_OX-L_2500 relation (Lusso &amp; Risaliti 2016). At fixed Gamma and alpha_OX, increasing bolometric luminosity shifts the whole spectrum upward but leaves the spectral shape nearly intact — the sub-linear alpha_OX relation only steepens the shape at the top of the quasar regime.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_agn_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_agn`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">AGN corona: bolometric luminosity sets normalisation, not shape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The CIGALE-faithful obscured-AGN spectral model combines two knobs that classification surveys often confound: delta_alpha_ox (offset from the empirical alpha_OX-L_2500 relation, controlling the intrinsic X-ray-to-UV ratio) and log N_H (line-of-sight column density, suppressing soft-band flux through zphabs × cabs). We compute the hardness ratio HR = (H - S) / (H + S) with S = 0.5-2 keV and H = 2-10 keV across the joint (delta_alpha_ox, log N_H) plane on a fixed L_2500 anchor (= L_bol = 1e45 erg/s through the Hopkins+2007 bolometric correction).">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_alpha_ox_nh_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_alpha_ox_nh`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hardness ratio across the alpha_OX vs log N_H plane</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="X-ray binaries (XRBs) are the dominant X-ray sources in star-forming galaxies once an AGN is excluded. High-mass XRBs trace the recent star-formation rate (Mineo+2012), while low-mass XRBs trace the integrated stellar mass (Lehmer+2019). The two scalings have different spectral shapes too: HMXBs are slightly harder, LMXBs slightly softer. Two side-by-side sweeps — SFR (left) at fixed M_star = 1e11 M_sun, and M_star (right) at fixed SFR = 10 M_sun/yr — separate the two channels on the same axes.">

.. only:: html

  .. image:: /auto_examples/xray/images/thumb/sphx_glr_plot_xray_sf_thumb.png
    :alt:

  :doc:`/auto_examples/xray/plot_xray_sf`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">X-ray binary luminosity scales with SFR (HMXB) and stellar mass (LMXB)</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="At fixed host (constant SFR = 3 M_sun/yr, Mineo+12 HMXB contribution) we sweep the composable AGN&#x27;s bolometric luminosity agn_log_lbol from 9 to 13 (in log L_sun). The host XRB component is a flat power-law below ~10 keV; the AGN corona contributes a much harder power-law that dominates above log L_bol ≳ 11.">

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

Intergalactic medium absorption and Lyman-forest effects on observed SEDs.


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

    <div class="sphx-glr-thumbcontainer" tooltip="The intergalactic medium (IGM) opacity increases dramatically with redshift due to the expanding neutral hydrogen fraction. We sweep redshift z ∈ {2, 3, 4, 5, 6, 7, 8} on Inoue et al. (2014) IGM transmission curves, showing how the Lyman alpha forest deepens and the Lyman break (912 Å rest-frame) shifts to longer observed wavelengths at higher z, suppressing flux blueward of the break.">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_igm_z_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_igm_z_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Lyman forest deepens with redshift: high-z IGM opacity suppresses UV flux</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A power-law QSO continuum (Vanden Berk et al. 2001 composite slope \alpha_{\nu} = -0.5) is built with tengri&#x27;s AGN multicolor disc and pure-stellar synthesis. The Inoue et al. 2014 intergalactic-medium transmission is then applied to the observed frame, suppressing the blue side below \lambda_{\rm obs} &lt; \lambda_{\rm Ly\alpha}(1+z).">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_lyman_alpha_forest_QSO_template_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_lyman_alpha_forest_QSO_template`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">QSO continuum with Lyman-alpha forest absorption at z=3</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/igm/images/thumb/sphx_glr_plot_lyman_dropout_redshift_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/igm/plot_lyman_dropout_redshift_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
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

Filter curves and photometric fitting.


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

    <div class="sphx-glr-thumbcontainer" tooltip="How does a galaxy&#x27;s location in colour–colour space evolve with redshift? We compute SDSS g − r and r − z colours for two galaxy populations — a young star-forming and an old quiescent — across z = 0 to 3, with arrows marking the integer redshift stops. This is the reference picture for photometric redshift classifiers and for stellar-template grids.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_color_tracks_redshift_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_color_tracks_redshift`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Photometric colour tracks vs redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="How does the observed photometric flux of a FIXED-luminosity galaxy decline with redshift? We track a star-forming galaxy (log M* = 10.5, SFR = 10 Msun/yr) across z = 0.1 to 6 in three optical/infrared bands (SDSS r, JWST J, JWST H), visualizing the three physical effects:">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_cosmic_dimming_observed_flux_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_cosmic_dimming_observed_flux`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Cosmic dimming and K-correction with redshift</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Plot the ugriz filter transmission curves from the SDSS photometric system. Filters are loaded from the SVO Filter Profile Service via tengri&#x27;s filter registry.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_curves_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_curves`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SDSS Filter Transmission Curves</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_filter_set_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_filter_set_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="How do K-corrections vary with redshift for different galaxy populations? K-corrections quantify the shift in filter response as galaxies move to higher redshifts: K(z) = −2.5 log₁₀[(1+z) × F_ν(z) / F_ν(0)] for a fixed rest-frame filter. We compute K(z) for the SDSS r-band across four galaxy types — young star-forming, old star-forming, red-sequence elliptical, and post-starburst — from z = 0.01 to z = 2.0. This illustrates why stellar mass measurements require careful K-corrections at high redshift and why colour-matched template sets dominate photometric redshift algorithms.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_k_correction_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_k_correction_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">K-corrections as a function of redshift for different SED types</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The NUV−r colour is a sensitive probe of stellar age in galaxies. We show how a single-burst star formation history (tsnorm, truncated-skew-normal) evolves across the GALEX green valley (NUV−r ≈ 4–5 mag) as the stellar population ages from 0.05 to 5.5 Gyr. The colour exhibits a sharp discontinuity as the stellar population cools through the transition between young, UV-bright stars and older, redder populations.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_nuv_r_age_track_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_nuv_r_age_track`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">NUV−r colour vs stellar age</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_photometric_fit_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_photometric_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_red_sequence_blue_cloud_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_red_sequence_blue_cloud`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_redshift_filter_grid_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_redshift_filter_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_snr_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_snr_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_spitzer_irac_agn_wedge_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_spitzer_irac_agn_wedge`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The UVJ (U−V vs V−J) diagram is a classic method for separating star-forming from quiescent galaxies. We populate it with four model tracks: (1) constant star-forming galaxies with varying dust optical depth, (2) an old quiescent population, (3) a post-starburst galaxy, and (4) a dusty starburst. The grey box marks the &quot;quiescent region&quot; from Williams+2009, a visual guide for identifying passive galaxies.">

.. only:: html

  .. image:: /auto_examples/photometry/images/thumb/sphx_glr_plot_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/photometry/plot_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The UVJ colour–colour diagram</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The WISE color-color diagram (Stern et al. 2012) is a powerful tool for separating AGN from star-forming galaxies using mid-infrared colors. The diagnostic exploits the fact that AGN emit power-law SEDs (flat in νLν) while star-forming galaxies have cooler dust emission (Rayleigh-Jeans slope at long wavelengths).">

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

Spectroscopic fitting and spectral features.


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

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_jwst_nirspec_high_z_spectrum_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_jwst_nirspec_high_z_spectrum`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="A classic post-starburst (PSB) / K+A galaxy signature: strong Balmer absorption lines (high Hδ_A) with no emission, visible only in a narrow window after a recent burst of star formation has been abruptly quenched.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_post_starburst_diagnostic_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_post_starburst_diagnostic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Post-Starburst K+A Diagnostic: Hδ_A vs Time Since Quench</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrate how instrumental resolution affects spectral line profile visibility by observing the same intrinsic SED at resolutions R = 100 (SDSS lores), 500 (DESI), 2000 (KMOS), 5000 (MUSE), and 25000 (HARPS). The Hα + [N II] complex (rest ~6550–6600 Å) transitions from fully blended at low R to completely resolved at high R, revealing the forbidden and Balmer lines separately.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_resolution_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_resolution_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Instrumental Resolution Sweep: Hα Line Blending</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Mg b 5170 Å region of an old stellar population observed at spectral resolution R = 3000, convolved with increasing stellar velocity dispersion σ_v from 50 to 400 km/s. The classic kinematic diagnostic — line core depth tracks σ_v, asymmetric wings appear with rotational broadening (not modelled here, sigma only).">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Compare the rest-frame spectrum of a young and old galaxy at fixed redshift. Shows how the optical continuum color, Balmer decrement, and absorption line strengths depend on mean stellar age, holding metallicity and dust fixed.">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_spectrum_fit_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_spectrum_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Rest-frame spectrum with stellar population ages</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/spectroscopy/images/thumb/sphx_glr_plot_velocity_dispersion_sweep_thumb.png
    :alt:

  :doc:`/auto_examples/spectroscopy/plot_velocity_dispersion_sweep`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Multiwavelength
===============

Panchromatic SED examples spanning X-ray to radio.

These short scripts complement the main tutorial notebooks — each produces a
single figure. For full narrative walkthroughs, see the ``notebooks/`` spine.


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

    <div class="sphx-glr-thumbcontainer" tooltip="The FIR–radio correlation (van der Kruit 1971; Helou et al. 1985) holds over five decades in galaxy luminosity. Sweeps IR luminosity and radio spectral index to show the tight linear correlation and how the empirical parameter q_IR varies with model calibration.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_fir_radio_correlation_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_fir_radio_correlation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">FIR–radio correlation across IR luminosity and spectral shape</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A single intrinsic LBG (young dust-poor star-forming galaxy) shown in the observer frame at four redshifts. The Lyman break sweeps redward into the u- and then g- and r-band dropout regimes, the Inoue+2014 IGM transmission removes more and more flux blueward of Lyα, and the apparent magnitude faint-end falls by ~2.5 mag from z = 2 → 8 due to luminosity distance alone.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_lbg_observed_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_lbg_observed_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Observed SED of a Lyman-break galaxy at z = 2, 4, 6, 8</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="M82 (NGC 3034) is a nearby starburst galaxy with intense nuclear star formation (SFR ~ 10 Msun/yr), stellar mass M* ~ 1×10^10 Msun, and moderate-to-high dust opacity (τ_V ~ 2 in the starburst core). The panchromatic SED spans from UV (young stars) through optical (attenuated by dust) to far-infrared (warm dust re-emission at ~50 μm) and radio (free-free continuum from ionized regions and synchrotron from supernovae).">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_m82_starburst_panchromatic_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_m82_starburst_panchromatic`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Panchromatic SED: M82 Starburst Analog</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Spiral galaxies exhibit radial metallicity gradients: metal-rich centres and metal-poor discs (e.g. NGC 891, Searle 1971). This example illustrates how three common gradient scenarios—steep positive, flat, and inverted depletion—reshape the integrated SED when weighted by disc area.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_metallicity_radial_gradient_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_metallicity_radial_gradient`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Radial metallicity gradients and integrated-light SED</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 5–30 μm rest-frame spectrum showcases distinct infrared tracers: dust polycyclic aromatic hydrocarbon (PAH) emission peaks at 6.2, 7.7, 8.6, 11.3, and 12.7 μm in star-forming galaxies, while silicate absorption (9.7 μm Si–O stretch) and AGN heating suppress PAH and introduce continuum growth in AGN-dominated systems. We model three templates: (a) pure starburst (no AGN), (b) pure AGN (no star formation), and (c) composite with AGN fraction = 0.5. This illustrates the diagnostic power of mid-IR spectroscopy: PAH strength probes star formation rate, while continuum slope and silicate depth reveal AGN heating and dust temperature.">

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

    <div class="sphx-glr-thumbcontainer" tooltip="Full panchromatic SED combining stellar continuum, dust absorption, dust infrared emission, and radio synchrotron. Demonstrates how a unified model spans from ultraviolet through centimeter wavelengths with continuous physics from stellar populations through dust and synchrotron emission.">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_panchromatic_galaxy_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_panchromatic_galaxy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">UV-to-radio SED of a star-forming galaxy</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/multiwavelength/images/thumb/sphx_glr_plot_panchromatic_milky_way_analog_thumb.png
    :alt:

  :doc:`/auto_examples/multiwavelength/plot_panchromatic_milky_way_analog`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Submillimeter galaxies (SMGs) are the most luminous starbursts in the universe, hidden behind massive dust columns. This example constructs a z=3 SMG SED with M* = 2×10^11 Msun, SFR = 500 Msun/yr, and τ_V ≈ 3.5 — typical of ALMA-detected sources and the SCUBA-2 850 μm parent population.">

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

Comparing inference methods and convergence diagnostics.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The convergence diagnostic shows how the negative log posterior (loss) decays across optimizer iterations. We fit mock photometry using MAP (maximum a posteriori) optimization with Adam and display the loss curve, showing when the optimizer has effectively converged. The right panel overlays the recovered SFH against the truth.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_convergence_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_convergence`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">MAP fit convergence: loss decay across iterations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates parameter degeneracies and individual 1-D marginalized posteriors after fitting mock 5-band SDSS photometry. The corner plot shows the full 2-D covariance structure between parameters; blue lines mark the injected truth. Note: for demonstration scale; production runs use 10× more VI iterations and samples.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_corner_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_corner`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Posterior corner plot from variational inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_hierarchical_population_fit_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_hierarchical_population_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates how adding rest-frame optical emission-line equivalent widths (H-alpha, [OIII]) to broadband photometry dramatically tightens parameter constraints and breaks the notorious photo-z/dust degeneracy. Two panels compare posterior widths: (a) redshift posterior with and without line constraints, and (b) dust attenuation posterior showing dramatic improvement.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_joint_photometry_line_fit_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_joint_photometry_line_fit`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Joint photometry + emission-line fitting breaks photo-z and dust degeneracies</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates convergence behavior of two inference methods: MAP (point-estimate via optimization) and pure-JAX geometric variational inference (native VI). Both are initialized from the same MAP fit, then evolve independently to show how they explore the posterior. The SFH panel on the right shows the recovered star-formation history from each method overlaid on the truth.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_method_comparison_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_method_comparison`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Comparing MAP and pure-JAX variational inference</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates the n_chains knob on the gradient-based MCMC backends. Each chain shares the cached step size and mass matrix (so this is only meaningful on the second call against the same model), then jax.vmap dispatches the chains in parallel across XLA SIMD lanes (CPU) or accelerator cores (GPU/TPU).">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_multichain_speedup_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_multichain_speedup`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Multi-chain MCMC speedup via jax.vmap</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_photoz_chi2_grid_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_photoz_chi2_grid`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates MCMC parameter estimation and posterior covariance structure after fitting mock 5-band SDSS photometry with a double-power-law (dpl) star formation history. The corner plot visualizes all 1-D marginalized posteriors and 2-D joint distributions, with blue lines marking the injected truth values.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_posterior_corner_dpl_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_posterior_corner_dpl`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Posterior corner plot from MCMC: double-power-law SFH</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The posterior predictive check (PPC) is the gold-standard Bayesian goodness-of-fit diagnostic (Rubin 1984; Gelman et al. 1996). We fit mock SDSS photometry, draw 100 samples from the posterior, regenerate mock photometry for each sample, and overlay the band-by-band model envelope (16th, 50th, 84th percentiles) against observed data with residuals normalized by noise. For well-fit models, residuals cluster within ±2σ.">

.. only:: html

  .. image:: /auto_examples/inference/images/thumb/sphx_glr_plot_posterior_predictive_check_thumb.png
    :alt:

  :doc:`/auto_examples/inference/plot_posterior_predictive_check`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Posterior predictive check: Bayesian goodness-of-fit diagnostic</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Simulate a galaxy with a double-power-law (dpl) star formation history, mock SDSS photometry at S/N=20, and recover the SFH using MAP optimization. The figure compares the true and recovered SFH as a function of time, with the bottom panel showing photometric residuals normalized by noise. The recovery demonstrates how well the SED fitting posterior reconstructs SFH shape despite degeneracies with dust and metallicity.">

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

Paper-style figures and diagnostic plots — UVJ diagram, JWST color-color,
SFR-indicator comparison, mass completeness, age–dust degeneracy, and
emission-line Pearson coefficients.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A 5 Gyr stellar population with no dust is nearly indistinguishable from a 1 Gyr population reddened by τ_diff = 0.4 when observed in optical broadband colors alone. This is the central degeneracy that limits SED-fitting accuracy from optical-only photometry, and the reason FUV/NUV (sensitive to recent star formation) or rest-frame IR (sensitive to dust mass) bands break the ambiguity.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_2d_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_2d`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The age–dust degeneracy on the optical g − r color</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_age_dust_redshift_degeneracy_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_age_dust_redshift_degeneracy`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Balmer decrement measures dust attenuation via hydrogen recombination line ratios: H-alpha / H-beta is sensitive to extinction (Calzetti et al. 2000). Without dust, the intrinsic ratio is ~2.78–2.86 (Case B). Here we sweep dust optical depth (τ_diff ∈ [0, 2]) and measure how the predicted H-alpha and H-beta change. We derive A_V = 1.086 × τ_diff and compare against the Calzetti+2000 expectation.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_balmer_decrement_av_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_balmer_decrement_av`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Balmer Decrement Tests Dust Attenuation on Emission Lines</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Observed-frame flux of a rest-frame SED depends on cosmological distances, which vary with H0 and Ω_M. This example quantifies the Hubble tension (H0 tension between local measurements ~73 km/s/Mpc and CMB measurements ~67.4 km/s/Mpc) by showing how apparent magnitude shifts by ~0.15 mag across cosmic time.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_cosmology_distance_modulus_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_cosmology_distance_modulus`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Hubble Tension: Cosmology-dependent distance modulus</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The 4000 Å break D_n(4000) — Bruzual 1983, Balogh+1999 — measures the discontinuity around 4000 Å produced by the line-blanketing of ionised metals in the atmospheres of old stars. It rises monotonically with the mass-weighted age of the stellar population and is one of the most widely used age indicators in SDSS-style optical-only data (Kauffmann+2003).">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_d4000_age_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_d4000_age`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The 4000 Å break as a stellar age proxy</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_dropout_selection_z3_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_dropout_selection_z3`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_emission_line_pcc_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_emission_line_pcc`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_jwst_color_color_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_jwst_color_color`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Three of the most-used star-formation-rate indicators agree only for specific assumed SFHs. We mock a constant-SFR galaxy across SFR = 0.01 to 100 M☉/yr and read each indicator out:">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_kennicutt_sfr_calibrations_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_kennicutt_sfr_calibrations`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Kennicutt+1998 SFR calibrations: UV, Hα, and L_IR</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Demonstrates how strong gravitational lensing elevates intrinsically-faint high-redshift (z=7) galaxies above the JWST NIRCam 5σ detection threshold.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_lensed_galaxy_magnification_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_lensed_galaxy_magnification`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Strong-lensing magnification: EoR galaxy detection boost</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_main_sequence_cosmic_evolution_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_main_sequence_cosmic_evolution`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The Speagle et al. 2014 star-forming main sequence defines the locus of star-forming galaxies in the log SFR vs. log M plane. This example generates 30 mock star-forming galaxies by sampling M uniformly and computing SFR from the Speagle+2014 relation. We then build minimal-configuration tengri SEDModels for each galaxy and verify the population using the public API.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_main_sequence_recovery_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_main_sequence_recovery`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The Star-Forming Main Sequence: M*-SFR Galaxy Population</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_mass_completeness_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_mass_completeness`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_sfr_indicator_compare_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_sfr_indicator_compare`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
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

    <div class="sphx-glr-thumbcontainer" tooltip="UV slope (β) is degenerate between dust optical depth and stellar age: young dusty and old dust-free populations both show red UV continua. This script sweeps BOTH dust (τ_diff ∈ [0, 1.5]) and stellar age (t_burst ∈ [0.01, 10] Gyr) on a single-burst SFH (tsnorm) and plots the resulting UV slope β as a 2D heatmap. We expect the age and dust axes to BOTH affect β: old stars are redder, dust reddens UV.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_tau_age_2d_uv_slope_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_tau_age_2d_uv_slope`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">2D Degeneracy: Dust Optical Depth vs Stellar Age via UV Slope</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_tully_fisher_relation_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_tully_fisher_relation`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="The infrared excess (IRX = L_IR / L_FUV) versus UV-continuum slope β diagram (Meurer+1999) is the standard tool for inferring attenuation in unresolved star-forming galaxies. We mock a population of star-forming galaxies with a fixed SFH and a range of diffuse dust optical depths, measure each galaxy&#x27;s β by fitting a power-law to its rest-frame UV continuum (1268–2580 Å, Calzetti+1994 windows), and overplot the empirical Meurer+1999 starburst relation.">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uv_slope_beta_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uv_slope_beta`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">The IRX–β relation emerges from the dust model</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="A cornerstone of dust modeling is energy conservation: the UV light absorbed by dust must be re-radiated in the infrared. This example constructs 15 tengri SEDModels with optical depth τ_V ∈ {0, 0.1, ..., 4} and validates that integrated infrared luminosity (8–1000 μm) matches the absorbed UV (912–3000 Å rest-frame).">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uv_to_ir_bolometric_balance_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uv_to_ir_bolometric_balance`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Dust energy balance: L_IR = L_UV_absorbed across opacity variations</div>
    </div>


.. raw:: html

    <div class="sphx-glr-thumbcontainer" tooltip="Example script with invalid Python syntax ">

.. only:: html

  .. image:: /auto_examples/usecases/images/thumb/sphx_glr_plot_usecase_uvj_diagram_thumb.png
    :alt:

  :doc:`/auto_examples/usecases/plot_usecase_uvj_diagram`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">SyntaxError</div>
    </div>


.. thumbnail-parent-div-close

.. raw:: html

    </div>

Advanced Topics
===============

Hierarchical inference, gradient sensitivity, batch fitting, panchromatic SED
with radio and X-ray components, and joint photometry + spectroscopy fitting.


.. raw:: html

  <div id='sg-tag-list' class='sphx-glr-tag-list'></div>


.. raw:: html

    <div class="sphx-glr-thumbnails">

.. thumbnail-parent-div-open

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

    <div class="sphx-glr-thumbcontainer" tooltip="TODO[examples-sweep]: This script uses low-level component orchestration (build_components, run_components) which is experimental Phase II-2.6 API intended for infrastructure use, not recommended for user-facing examples.">

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


.. only:: html

 .. rst-class:: sphx-glr-signature

    `Gallery generated by Sphinx-Gallery <https://sphinx-gallery.github.io>`_
