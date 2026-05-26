:orphan:



.. _sphx_glr_auto_examples_sps:

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

    <div class="sphx-glr-thumbcontainer" tooltip="The stellar populations in massive elliptical galaxies are typically α-enhanced ([α/Fe] &gt; 0) due to rapid star formation timescales that terminate before iron-peak elements fully enrich the gas (Thomas et al. 2005). increasing [α/Fe] shifts absorption features — particularly the Mg b and Fe5270 indices — which serve as diagnostics of star-formation history timescale.">

.. only:: html

  .. image:: /auto_examples/sps/images/thumb/sphx_glr_plot_alpha_enhanced_population_thumb.png
    :alt:

  :doc:`/auto_examples/sps/plot_alpha_enhanced_population`

.. raw:: html

      <div class="sphx-glr-thumbnail-title">Alpha-element enhancement in quiescent stellar populations</div>
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


.. toctree::
   :hidden:

   /auto_examples/sps/plot_age_metallicity_color_grid
   /auto_examples/sps/plot_alpha_enhanced_population
   /auto_examples/sps/plot_fnu_vs_flambda_units
   /auto_examples/sps/plot_ionising_lum
   /auto_examples/sps/plot_mass_to_light_band_comparison
   /auto_examples/sps/plot_mass_to_light_ratios
   /auto_examples/sps/plot_sps_library_compare
   /auto_examples/sps/plot_ssp_age_sweep
   /auto_examples/sps/plot_ssp_color_compare
   /auto_examples/sps/plot_ssp_grid
   /auto_examples/sps/plot_ssp_imf_compare
   /auto_examples/sps/plot_ssp_library_shootout
   /auto_examples/sps/plot_ssp_metallicity_sweep
   /auto_examples/sps/plot_uv_slope_age

